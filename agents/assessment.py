from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from core.bdi_modules import ITEM_TO_MODULES, MODULE_NAMES, MODULE_TO_ITEMS, MODULE_WEIGHTS
from core.calibration import build_feature_vector, get_calibrator_bundle, predict_with_explanations
from core.llm import LLMBudgetExceeded, get_llm
from core.prompts import get_prompt
from core.state import (
    AgentState,
    BDI_ITEM_NAMES,
    EvidenceRecord,
    ItemBelief,
    SPECIALIST_ITEM_MAP,
    SYMPTOM_NAME_TO_ITEM,
    bump_failure_counter,
    top_symptoms_from_beliefs,
    top_symptoms_from_scores,
)

LEXICAL_EVIDENCE_CUES: Dict[int, List[str]] = {
    1: ["sad", "down", "low", "heavy", "empty", "dark cloud"],
    2: [
        "hopeless",
        "no future",
        "nothing will change",
        "pointless",
        "worrying about the future",
        "no way out",
    ],
    3: [
        "failure",
        "failed",
        "not good enough",
        "falling behind",
        "playing catch-up",
        "accomplished nothing",
        "wasted the whole day",
    ],
    4: [
        "no pleasure",
        "enjoy nothing",
        "hollow",
        "nothing feels good",
        "going through the motions",
        "feel like a chore",
        "not really enjoying",
        "disconnected",
        "not really being present",
    ],
    5: ["guilty", "guilt", "blame myself", "regret"],
    7: ["hate myself", "dislike myself"],
    8: [
        "beat myself up",
        "self-doubt",
        "self critical",
        "second-guessing",
        "what i've done wrong",
        "pretending to be okay",
        "facade",
    ],
    9: ["better off dead", "end it", "kill myself", "self harm", "hurt myself"],
    11: ["restless", "agitated", "can't sit still"],
    12: ["withdrawing", "keeping to myself", "avoid people"],
    14: ["worthless", "burden", "useless", "not enough"],
    15: [
        "no energy",
        "drained",
        "exhausted",
        "running on empty",
        "hard to muster",
        "no motivation",
        "struggle to get started",
    ],
    16: [
        "can't sleep",
        "trouble sleeping",
        "lying awake",
        "insomnia",
        "wake up tired",
        "trouble falling asleep",
        "wake up feeling",
    ],
    18: ["no appetite", "appetite", "eating less", "eating more"],
    19: [
        "can't focus",
        "concentrate",
        "distracted",
        "brain fog",
        "mind was racing",
        "lose focus",
        "zoning out",
        "mind is always elsewhere",
    ],
    20: ["fatigue", "tired all day", "worn out", "sluggish"],
}


def _latest_persona_message_with_index(state: AgentState) -> Tuple[str, int]:
    messages = list(state.get("messages", []))
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")), idx
    return "", -1


def _recent_context(state: AgentState, limit: int = 4) -> str:
    turns = state.get("messages", [])[-limit:]
    lines = []
    for msg in turns:
        role = "Detector" if msg.get("role") == "user" else "Persona"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _parse_json_payload(raw_text: str) -> Tuple[Any, bool]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    starts = []
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start != -1:
        starts.append(obj_start)
    if arr_start != -1:
        starts.append(arr_start)
    if not starts:
        return {}, False
    start = min(starts)

    obj_end = text.rfind("}")
    arr_end = text.rfind("]")
    end = max(obj_end, arr_end)
    if end == -1 or end < start:
        return {}, False

    candidate = text[start : end + 1].strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, (dict, list)):
            return payload, True
        return {}, False
    except json.JSONDecodeError:
        return {}, False


def _number_in_range(value, low: float, high: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return low <= numeric <= high


def _sentence_for_cue(text: str, cue: str) -> str:
    chunks = [part.strip() for part in text.replace("!", ".").replace("?", ".").split(".")]
    lower_cue = cue.lower()
    for chunk in chunks:
        if lower_cue in chunk.lower():
            return chunk
    return text[:220].strip()


def _fallback_evidence_from_text(node_name: str, turn: int, text: str) -> List[EvidenceRecord]:
    lowered = text.lower()
    records: List[EvidenceRecord] = []
    for item_id, cues in LEXICAL_EVIDENCE_CUES.items():
        hits = [cue for cue in cues if cue in lowered]
        if not hits:
            continue
        intensity = min(3.0, 1.0 + (0.35 * len(hits)))
        confidence = min(0.85, 0.45 + (0.1 * len(hits)))
        evidence_text = _sentence_for_cue(text, hits[0])
        records.append(
            EvidenceRecord(
                turn=turn,
                node=node_name if node_name in {"somatic", "cognitive", "risk"} else "cognitive",
                item_id=item_id,
                symptom_name=BDI_ITEM_NAMES.get(item_id, f"Item {item_id}"),
                direction="increase",
                intensity=float(intensity),
                confidence=float(confidence),
                evidence_text=evidence_text,
                reason=f"lexical cue match: {', '.join(hits[:3])}",
                method="lexical_fallback",
            )
        )
    records.sort(key=lambda record: (record.confidence, record.intensity), reverse=True)
    return records[:3]


def _coerce_item_id(raw_item_id, raw_symptom_name: str) -> int | None:
    try:
        item_id = int(raw_item_id)
        if 1 <= item_id <= 21:
            return item_id
    except (TypeError, ValueError):
        pass

    symptom = raw_symptom_name.strip().lower()
    if symptom in SYMPTOM_NAME_TO_ITEM:
        return SYMPTOM_NAME_TO_ITEM[symptom]
    return None


def _coerce_evidence_record(node_name: str, turn: int, item: Dict, fallback_text: str) -> EvidenceRecord | None:
    symptom_name = str(item.get("symptom_name", "")).strip()
    item_id = _coerce_item_id(item.get("item_id"), symptom_name)
    if item_id is None:
        return None
    if not symptom_name:
        symptom_name = BDI_ITEM_NAMES[item_id]

    direction = str(item.get("direction", "increase")).strip().lower()
    if direction not in {"increase", "decrease", "neutral"}:
        direction = "increase"

    try:
        intensity = float(item.get("intensity", 0.0))
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(3.0, intensity))

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_text = str(item.get("evidence_text", "")).strip() or fallback_text[:220]
    reason = str(item.get("reason", "")).strip() or "implicit affective signal"
    method = str(item.get("method", "llm_extractor")).strip() or "llm_extractor"

    return EvidenceRecord(
        turn=turn,
        node=node_name,
        item_id=item_id,
        symptom_name=symptom_name,
        direction=direction,
        intensity=intensity,
        confidence=confidence,
        evidence_text=evidence_text,
        reason=reason,
        method=method,
    )


def extract_evidence(state: AgentState) -> Dict:
    node_name = str(state.get("active_node", "cognitive"))
    latest_message, latest_persona_idx = _latest_persona_message_with_index(state)
    turn = int(state.get("turn_index", 0)) + 1
    last_processed_idx = int(state.get("last_processed_persona_msg_idx", -1))
    has_new_persona_input = latest_persona_idx > last_processed_idx

    if node_name not in {"somatic", "cognitive", "risk"}:
        node_name = "cognitive"

    if not has_new_persona_input:
        turn_trace = {
            "extract_evidence": {
                "turn": turn,
                "source": "skip_no_new_persona",
                "has_new_persona_input": False,
                "latest_persona_idx": latest_persona_idx,
                "last_processed_persona_msg_idx": last_processed_idx,
                "kept_items_count": 0,
                "empty_streak": int(state.get("empty_evidence_streak", 0)),
            }
        }
        return {
            "latest_turn_evidence": [],
            "specialist_debug": "Evidence extraction: waiting for persona input",
            "turn_trace": turn_trace,
            "has_new_persona_input": False,
        }

    evidence_records: List[EvidenceRecord] = []
    raw_nonempty = False
    json_parse_ok = False
    raw_items_count = 0
    dropped_unknown = 0
    dropped_invalid = 0
    source = "llm_extractor"
    counters = dict(state.get("failure_counters", {}))
    lexical_prefilter: List[EvidenceRecord] = []
    llm_on_lexical_hit = os.getenv("EVIDENCE_LLM_ON_LEXICAL_HIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if latest_message.strip():
        lexical_prefilter = _fallback_evidence_from_text(node_name, turn, latest_message)
        if lexical_prefilter and not llm_on_lexical_hit:
            evidence_records = lexical_prefilter
            source = "lexical_prefilter"
        else:
            prompt = get_prompt("evidence_extraction").format(
                node_name=node_name,
                recent_context=_recent_context(state) or "none",
                latest_message=latest_message,
            )
            try:
                llm = get_llm()
                raw = llm.invoke([("system", prompt)]).content
                raw_nonempty = bool(str(raw).strip())
                parsed, json_parse_ok = _parse_json_payload(str(raw))
                items: List[Any] = []
                if isinstance(parsed, dict):
                    maybe_items = parsed.get("evidence", [])
                    if isinstance(maybe_items, list):
                        items = maybe_items
                elif isinstance(parsed, list):
                    items = parsed
                if raw_nonempty and not json_parse_ok:
                    counters = bump_failure_counter(counters, "extract_json_parse_fail")
                if isinstance(items, list):
                    raw_items_count = len(items)
                    for raw_item in items:
                        if not isinstance(raw_item, dict):
                            dropped_invalid += 1
                            continue
                        symptom_name = str(raw_item.get("symptom_name", "")).strip()
                        resolved_item_id = _coerce_item_id(raw_item.get("item_id"), symptom_name)
                        if resolved_item_id is None:
                            dropped_unknown += 1
                            continue
                        if not _number_in_range(raw_item.get("intensity"), 0.0, 3.0):
                            dropped_invalid += 1
                            continue
                        if not _number_in_range(raw_item.get("confidence"), 0.0, 1.0):
                            dropped_invalid += 1
                            continue
                        record = _coerce_evidence_record(node_name, turn, raw_item, latest_message)
                        if record is not None:
                            evidence_records.append(record)
                else:
                    source = "llm_extractor_non_list_payload"
            except LLMBudgetExceeded:
                raise
            except Exception:
                source = "llm_extractor_error"
                counters = bump_failure_counter(counters, "extract_json_parse_fail")
    else:
        source = "skip_empty_message"

    if dropped_unknown > 0:
        counters = bump_failure_counter(counters, "extract_item_map_fail", amount=dropped_unknown)
    fallback_records: List[EvidenceRecord] = []
    if not evidence_records and latest_message.strip():
        fallback_records = _fallback_evidence_from_text(node_name, turn, latest_message)
        if fallback_records:
            evidence_records = fallback_records
            source = "lexical_fallback"

    if not evidence_records:
        counters = bump_failure_counter(counters, "extract_empty")
        empty_streak = int(state.get("empty_evidence_streak", 0)) + 1
    else:
        empty_streak = 0

    turn_trace = {
        "extract_evidence": {
            "turn": turn,
            "source": source,
            "raw_nonempty": raw_nonempty,
            "json_parse_ok": json_parse_ok,
            "raw_items_count": raw_items_count,
            "kept_items_count": len(evidence_records),
            "drop_unknown_item_count": dropped_unknown,
            "drop_invalid_range_count": dropped_invalid,
            "prefilter_count": len(lexical_prefilter),
            "llm_on_lexical_hit": llm_on_lexical_hit,
            "fallback_used": bool(fallback_records),
            "empty_streak": empty_streak,
            "has_new_persona_input": True,
            "latest_persona_idx": latest_persona_idx,
            "last_processed_persona_msg_idx": last_processed_idx,
        }
    }
    summary = (
        f"{state.get('specialist_debug', '')} | evidence_count={len(evidence_records)}"
        if state.get("specialist_debug")
        else f"Evidence extraction: count={len(evidence_records)}"
    )
    return {
        "latest_turn_evidence": evidence_records,
        "evidence_log": evidence_records,
        "specialist_debug": summary,
        "turn_trace": turn_trace,
        "failure_counters": counters,
        "empty_evidence_streak": empty_streak,
        "has_new_persona_input": True,
    }


def _coerce_belief(item_id: int, value) -> ItemBelief:
    if isinstance(value, ItemBelief):
        return value
    if isinstance(value, dict):
        try:
            return ItemBelief(**value)
        except Exception:
            pass
    return ItemBelief(
        item_id=item_id,
        mean_score=0.0,
        uncertainty=1.0,
        support_count=0,
        last_update_turn=0,
    )


def _update_single_belief(belief: ItemBelief, evidence: EvidenceRecord, turn: int) -> ItemBelief:
    prior_n = belief.support_count
    new_n = prior_n + 1
    weighted_observation = evidence.intensity * evidence.confidence
    new_mean = ((belief.mean_score * prior_n) + weighted_observation) / new_n
    new_uncertainty = max(0.05, 1.0 / (new_n + 1.0))
    return ItemBelief(
        item_id=belief.item_id,
        mean_score=max(0.0, min(3.0, new_mean)),
        uncertainty=max(0.0, min(1.0, new_uncertainty)),
        support_count=new_n,
        last_update_turn=turn,
    )


def update_beliefs(state: AgentState) -> Dict:
    turn = int(state.get("turn_index", 0)) + 1
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    if not has_new_persona_input:
        turn_trace = dict(state.get("turn_trace", {}))
        turn_trace["update_beliefs"] = {
            "turn": turn,
            "skipped_no_new_persona_input": True,
            "active_node": str(state.get("active_node", "cognitive")),
            "updated_item_ids": [],
            "risk_flag": bool(state.get("risk_flag", False)),
            "calibrator_mode": str(state.get("calibrator_mode", "deterministic_default")),
            "global_confidence": round(float(state.get("global_confidence", 0.0)), 4),
            "positive_features": [],
            "negative_features": [],
        }
        return {
            "turn_trace": turn_trace,
            "specialist_debug": "Belief update: skipped (no new persona input)",
        }

    latest_evidence = list(state.get("latest_turn_evidence", []))
    prior_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = _coerce_belief(item_id, prior_beliefs.get(item_id))

    for record in latest_evidence:
        if 1 <= record.item_id <= 21:
            beliefs[record.item_id] = _update_single_belief(beliefs[record.item_id], record, turn)

    risk_flag = bool(state.get("risk_flag", False)) or any(
        rec.item_id == 9 and rec.intensity >= 0.75 for rec in latest_evidence
    )

    evidence_confidences = [float(rec.confidence) for rec in latest_evidence]
    feature_vector = build_feature_vector(beliefs, evidence_confidences, risk_flag)
    bundle = get_calibrator_bundle()
    prediction = predict_with_explanations(feature_vector, bundle)
    counters = dict(state.get("failure_counters", {}))
    if prediction.mode == "deterministic_default" and getattr(bundle, "fallback_reason", "") == "load_failed":
        counters = bump_failure_counter(counters, "calibrator_fallback_cache")

    positive = [
        {
            "feature": item.feature,
            "value": item.value,
            "weight": item.weight,
            "impact": item.impact,
        }
        for item in prediction.positive_contributions
    ]
    negative = [
        {
            "feature": item.feature,
            "value": item.value,
            "weight": item.weight,
            "impact": item.impact,
        }
        for item in prediction.negative_contributions
    ]

    active_node = str(state.get("active_node", "cognitive"))
    node_items = SPECIALIST_ITEM_MAP.get(active_node, [])
    node_summary = ", ".join(str(item_id) for item_id in node_items[:3]) or "n/a"
    updated_item_ids = sorted({int(record.item_id) for record in latest_evidence})
    positive_names = [row["feature"] for row in positive[:3]]
    negative_names = [row["feature"] for row in negative[:3]]
    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["update_beliefs"] = {
        "turn": turn,
        "skipped_no_new_persona_input": False,
        "active_node": active_node,
        "updated_item_ids": updated_item_ids,
        "risk_flag": risk_flag,
        "calibrator_mode": prediction.mode,
        "global_confidence": round(float(prediction.global_confidence), 4),
        "positive_features": positive_names,
        "negative_features": negative_names,
    }
    debug = (
        f"{state.get('specialist_debug', '')} | "
        f"beliefs_updated={len(latest_evidence)}; node_items={node_summary}; "
        f"cal_mode={prediction.mode}; conf={prediction.global_confidence:.2f}"
    )

    return {
        "item_beliefs": beliefs,
        "risk_flag": risk_flag,
        "latest_feature_vector": feature_vector,
        "calibrator_mode": prediction.mode,
        "positive_contributions": positive,
        "negative_contributions": negative,
        "global_confidence": prediction.global_confidence,
        "predicted_bdi_score": prediction.predicted_bdi_score,
        "predicted_label": prediction.predicted_label,
        "predicted_key_symptoms": top_symptoms_from_beliefs(beliefs, limit=4),
        "specialist_debug": debug,
        "turn_trace": turn_trace,
        "failure_counters": counters,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _module_stats_from_beliefs(item_beliefs: Dict[int, ItemBelief]) -> Dict[int, Dict[str, float | List[int]]]:
    module_stats: Dict[int, Dict[str, float | List[int]]] = {}
    for module_id, module_items in MODULE_TO_ITEMS.items():
        observed_items = [item_id for item_id in module_items if int(item_beliefs[item_id].support_count) > 0]
        if not observed_items:
            continue

        coverage = float(len(observed_items)) / float(max(1, len(module_items)))
        avg_support = sum(float(item_beliefs[item_id].support_count) for item_id in observed_items) / float(
            len(observed_items)
        )
        support_strength = _clamp(avg_support / 2.0, 0.0, 1.0)
        module_conf = _clamp((0.20 + (0.50 * coverage) + (0.30 * support_strength)), 0.0, 1.0)

        weighted_sum = 0.0
        weight_total = 0.0
        for item_id in observed_items:
            belief = item_beliefs[item_id]
            local_weight = _clamp(0.5 + (0.25 * float(belief.support_count)), 0.0, 1.0)
            weighted_sum += float(belief.mean_score) * local_weight
            weight_total += local_weight
        module_mean = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        module_signal = module_mean * module_conf

        module_stats[module_id] = {
            "module_id": module_id,
            "module_name": MODULE_NAMES.get(module_id, f"Module {module_id}"),
            "items": list(module_items),
            "observed_items": observed_items,
            "coverage": round(coverage, 6),
            "avg_support": round(avg_support, 6),
            "support_strength": round(support_strength, 6),
            "module_conf": round(module_conf, 6),
            "module_mean": round(module_mean, 6),
            "module_signal": round(module_signal, 6),
        }
    return module_stats


def _impute_missing_item_score(
    item_id: int,
    module_stats: Dict[int, Dict[str, float | List[int]]],
) -> Tuple[float, List[Dict[str, float | int | str]]]:
    candidates = ITEM_TO_MODULES.get(item_id, [])
    contributions: List[Dict[str, float | int | str]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for module_id in candidates:
        module_row = module_stats.get(module_id)
        if not module_row:
            continue
        module_conf = float(module_row.get("module_conf", 0.0))
        module_signal = float(module_row.get("module_signal", 0.0))
        module_weight = float(MODULE_WEIGHTS.get(module_id, 1.0)) * module_conf
        if module_weight <= 0:
            continue
        weighted_sum += module_signal * module_weight
        weight_total += module_weight
        contributions.append(
            {
                "module_id": module_id,
                "module_name": MODULE_NAMES.get(module_id, f"Module {module_id}"),
                "weight": round(module_weight, 6),
                "module_signal": round(module_signal, 6),
                "module_conf": round(module_conf, 6),
                "module_mean": round(float(module_row.get("module_mean", 0.0)), 6),
                "coverage": round(float(module_row.get("coverage", 0.0)), 6),
            }
        )

    if weight_total > 0:
        imputed_float = weighted_sum / weight_total
    else:
        imputed_float = 0.0

    # Moderate strong-peer floor.
    for module_id in candidates:
        module_row = module_stats.get(module_id)
        if not module_row:
            continue
        coverage = float(module_row.get("coverage", 0.0))
        module_mean = float(module_row.get("module_mean", 0.0))
        if coverage >= 0.66 and module_mean >= 2.5:
            imputed_float = max(imputed_float, 2.0)
        if coverage >= 0.80 and module_mean >= 2.8:
            imputed_float = max(imputed_float, 2.5)

    return _clamp(imputed_float, 0.0, 3.0), contributions


def finalize_with_module_imputation(state: AgentState) -> Dict:
    raw_predicted_bdi_score = int(state.get("predicted_bdi_score") or 0)
    raw_predicted_label = str(state.get("predicted_label") or "control")
    risk_flag = bool(state.get("risk_flag", False))
    bdi_threshold = int(os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"))

    prior_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = _coerce_belief(item_id, prior_beliefs.get(item_id))

    module_stats = _module_stats_from_beliefs(beliefs)
    final_item_scores: Dict[int, int] = {}
    item_details: Dict[str, Dict[str, object]] = {}
    imputed_item_count = 0

    for item_id in range(1, 22):
        belief = beliefs[item_id]
        if int(belief.support_count) > 0:
            observed_float = _clamp(float(belief.mean_score), 0.0, 3.0)
            observed_int = int(round(observed_float))
            final_item_scores[item_id] = max(0, min(3, observed_int))
            item_details[str(item_id)] = {
                "source": "observed",
                "support_count": int(belief.support_count),
                "mean_score": round(float(belief.mean_score), 6),
                "final_score": final_item_scores[item_id],
                "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
            }
            continue

        imputed_float, contributions = _impute_missing_item_score(item_id, module_stats)
        final_item_scores[item_id] = max(0, min(3, int(round(imputed_float))))
        if final_item_scores[item_id] > 0:
            imputed_item_count += 1
        item_details[str(item_id)] = {
            "source": "imputed",
            "support_count": 0,
            "imputed_float": round(float(imputed_float), 6),
            "final_score": final_item_scores[item_id],
            "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
            "contributions": contributions,
        }

    final_bdi_score = max(0, min(63, sum(int(final_item_scores[item_id]) for item_id in range(1, 22))))

    core_item_ids = [2, 3, 4, 5, 7, 8, 14, 15, 16, 19, 20]
    final_core_item_min_hits = int(os.getenv("FINAL_CORE_ITEM_MIN_HITS", "2"))
    final_core_signal_gate = float(os.getenv("FINAL_CORE_SIGNAL_GATE", "1.0"))
    core_hits = sum(1 for item_id in core_item_ids if int(final_item_scores.get(item_id, 0)) >= 1)
    module_signal_total = sum(float(module_row.get("module_signal", 0.0)) for module_row in module_stats.values())

    depression_from_bdi = final_bdi_score >= bdi_threshold
    depression_from_core_coverage = (
        core_hits >= max(1, final_core_item_min_hits) and module_signal_total >= final_core_signal_gate
    )
    final_label = "depressed" if (depression_from_bdi or risk_flag or depression_from_core_coverage) else "control"
    final_key_symptoms = top_symptoms_from_scores(final_item_scores, limit=4)

    return {
        "raw_predicted_bdi_score": raw_predicted_bdi_score,
        "raw_predicted_label": raw_predicted_label,
        "predicted_bdi_score": final_bdi_score,
        "predicted_label": final_label,
        "predicted_key_symptoms": final_key_symptoms,
        "final_item_scores": final_item_scores,
        "module_imputation": {
            "module_stats": module_stats,
            "item_details": item_details,
            "imputed_item_count": imputed_item_count,
            "threshold": bdi_threshold,
            "risk_flag": risk_flag,
            "core_hits": core_hits,
            "core_item_ids": core_item_ids,
            "final_core_item_min_hits": final_core_item_min_hits,
            "module_signal_total": round(module_signal_total, 6),
            "final_core_signal_gate": final_core_signal_gate,
            "depression_from_bdi": depression_from_bdi,
            "depression_from_core_coverage": depression_from_core_coverage,
            "raw_predicted_bdi_score": raw_predicted_bdi_score,
            "raw_predicted_label": raw_predicted_label,
            "final_bdi_score": final_bdi_score,
            "final_label": final_label,
        },
    }
