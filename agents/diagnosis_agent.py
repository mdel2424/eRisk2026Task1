from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from core.llm import get_llm
from core.prompts import get_prompt
from core.probabilistic_runtime import ITEM_PARENT_WEIGHTS
from core.state import AgentState, DiagnosisDecision, FinalState, symptom_name_from_item, top_symptoms_from_scores


def _state_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _bound_quotes_for_item(state: AgentState, item_id: int, limit: int = 2) -> List[str]:
    quotes: List[str] = []
    for row in reversed(list(state.get("assertion_log", []))):
        try:
            row_item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if row_item_id != item_id:
            continue
        binding_status = str(_state_value(row, "binding_status", "") or "")
        if binding_status not in {"exact", "normalized_exact"}:
            continue
        quote = str(_state_value(row, "anchor_quote", "") or "").strip()
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= limit:
            break
    return list(reversed(quotes))


def _deterministic_item_score(presence_prob: float, expected_score: float) -> int:
    if presence_prob < 0.22 and expected_score < 0.55:
        return 0
    if expected_score < 0.75:
        return 0 if presence_prob < 0.35 else 1
    if expected_score < 1.55:
        return 1
    if expected_score < 2.35:
        return 2
    return 3


def _direct_bound_support_count(state: AgentState, item_id: int) -> int:
    count = 0
    for row in list(state.get("assertion_log", [])):
        try:
            row_item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if row_item_id != item_id:
            continue
        label = str(_state_value(row, "assertion_label", "") or "")
        binding_status = str(_state_value(row, "binding_status", "") or "")
        if label in {"present", "conditional", "contrastive"} and binding_status in {"exact", "normalized_exact"}:
            count += 1
    return count


def _max_posterior_mass(posterior: List[float]) -> float:
    if not posterior:
        return 0.0
    return max(max(0.0, min(1.0, float(prob))) for prob in posterior)


def _dominant_parent_support(state: AgentState, item_id: int) -> tuple[str, float]:
    parent_weights = dict(ITEM_PARENT_WEIGHTS.get(int(item_id), {}) or {})
    if not parent_weights:
        return "cognitive_affective", 0.0
    dominant_parent = max(parent_weights.items(), key=lambda pair: (float(pair[1]), pair[0]))[0]
    bayes_nodes = dict(state.get("bayes_nodes", {}) or {})
    node_state = bayes_nodes.get(dominant_parent)
    probability = max(0.0, min(1.0, float(_state_value(node_state, "probability", 0.0) or 0.0)))
    return dominant_parent, probability


def _calibrated_item_score(
    *,
    state: AgentState,
    item_id: int,
    presence_prob: float,
    expected_score: float,
    posterior: List[float],
) -> int:
    direct_support_count = _direct_bound_support_count(state, item_id)
    max_mass = _max_posterior_mass(posterior)
    dominant_parent, dominant_parent_prob = _dominant_parent_support(state, item_id)

    if int(item_id) == 9 and direct_support_count <= 0:
        return 0

    if direct_support_count <= 0:
        if presence_prob < 0.45 or expected_score < 0.80 or dominant_parent_prob < 0.30:
            return 0
        if (
            presence_prob >= 0.80
            and expected_score >= 1.75
            and dominant_parent_prob >= 0.66
            and max_mass >= 0.46
        ):
            return 2
        if (
            presence_prob >= 0.60
            and expected_score >= 1.00
            and dominant_parent_prob >= 0.40
            and max_mass >= 0.34
        ):
            return 1
        return 0

    if direct_support_count == 1:
        if presence_prob < 0.26 and expected_score < 0.80:
            return 0
        if expected_score < 1.15 or max_mass < 0.44:
            return 0 if presence_prob < 0.48 else 1
        if expected_score < 1.85 or max_mass < 0.52:
            return 1
        if expected_score < 2.65:
            return 2
        return 3 if presence_prob >= 0.86 else 2

    return _deterministic_item_score(presence_prob, expected_score)


def _deterministic_diagnosis(state: AgentState) -> DiagnosisDecision:
    bayes_items = dict(state.get("bayes_items", {}))
    item_scores: Dict[int, int] = {}
    rationale_by_item: Dict[str, str] = {}
    quote_links_by_item: Dict[str, List[str]] = {}
    max_probabilities: List[float] = []
    direct_bound_support_total = 0

    for item_id in range(1, 22):
        item_state = bayes_items.get(item_id)
        presence_prob = float(_state_value(item_state, "presence_prob", 0.0) or 0.0)
        expected_score = float(_state_value(item_state, "expected_score", 0.0) or 0.0)
        posterior = list(_state_value(item_state, "score_posterior", [1.0, 0.0, 0.0, 0.0]) or [1.0, 0.0, 0.0, 0.0])
        if posterior:
            max_probabilities.append(max(float(prob) for prob in posterior))
        direct_support_count = _direct_bound_support_count(state, item_id)
        direct_bound_support_total += direct_support_count
        item_scores[item_id] = _calibrated_item_score(
            state=state,
            item_id=item_id,
            presence_prob=presence_prob,
            expected_score=expected_score,
            posterior=posterior,
        )
        quotes = _bound_quotes_for_item(state, item_id)
        quote_links_by_item[str(item_id)] = quotes
        if item_scores[item_id] > 0:
            dominant_parent, dominant_parent_prob = _dominant_parent_support(state, item_id)
            if direct_support_count > 0 and quotes:
                rationale = (
                    f"{symptom_name_from_item(item_id)} scored from a directly quote-grounded posterior "
                    f"(presence={presence_prob:.2f}, expected={expected_score:.2f}); grounded in: {quotes[0]}"
                )
            elif direct_support_count > 0:
                rationale = (
                    f"{symptom_name_from_item(item_id)} scored from directly quote-grounded support "
                    f"(presence={presence_prob:.2f}, expected={expected_score:.2f})"
                )
            else:
                rationale = (
                    f"{symptom_name_from_item(item_id)} scored from posterior-supported, corroborated cluster state "
                    f"({dominant_parent}={dominant_parent_prob:.2f}, presence={presence_prob:.2f}, expected={expected_score:.2f})"
                )
            rationale_by_item[str(item_id)] = rationale

    total_bdi = max(0, min(63, sum(item_scores.values())))
    predicted_label = "depressed" if total_bdi >= int(os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14")) else "control"
    supported_item_count = sum(1 for score in item_scores.values() if int(score) > 0)
    base_confidence = sum(max_probabilities) / float(len(max_probabilities)) if max_probabilities else 0.0
    support_factor = min(1.0, (float(supported_item_count) + min(float(direct_bound_support_total), 4.0)) / 8.0)
    confidence = max(0.12, base_confidence * (0.30 + (0.70 * support_factor)))
    return DiagnosisDecision(
        item_scores=item_scores,
        total_bdi=total_bdi,
        predicted_label=predicted_label,
        confidence=max(0.0, min(1.0, confidence)),
        rationale_by_item=rationale_by_item,
        quote_links_by_item=quote_links_by_item,
        used_llm=False,
        synthesis_mode="deterministic",
    )


def _maybe_llm_diagnosis(state: AgentState, fallback: DiagnosisDecision) -> DiagnosisDecision:
    if os.getenv("DIAGNOSIS_AGENT_USE_LLM", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return fallback

    prompt_template = get_prompt("diagnosis_synthesis")
    if not prompt_template.strip():
        return fallback

    bayes_items = dict(state.get("bayes_items", {}))
    item_lines = []
    for item_id in range(1, 22):
        item_state = bayes_items.get(item_id)
        item_lines.append(
            {
                "item_id": item_id,
                "symptom_name": symptom_name_from_item(item_id),
                "presence_prob": round(float(_state_value(item_state, "presence_prob", 0.0) or 0.0), 4),
                "expected_score": round(float(_state_value(item_state, "expected_score", 0.0) or 0.0), 4),
                "quotes": _bound_quotes_for_item(state, item_id),
            }
        )

    prompt = prompt_template.format(
        item_state_json=json.dumps(item_lines, ensure_ascii=True),
        total_expected_bdi=round(sum(float(_state_value(bayes_items.get(item_id), "expected_score", 0.0) or 0.0) for item_id in range(1, 22)), 4),
    )

    try:
        raw = str(get_llm().invoke([("system", prompt)]).content or "").strip()
        payload = json.loads(raw)
    except Exception:
        return fallback

    try:
        raw_scores = dict(payload.get("item_scores", {}))
        item_scores = {item_id: max(0, min(3, int(raw_scores.get(str(item_id), raw_scores.get(item_id, fallback.item_scores.get(item_id, 0))) or 0))) for item_id in range(1, 22)}
        total_bdi = max(0, min(63, int(payload.get("total_bdi", sum(item_scores.values())) or sum(item_scores.values()))))
        predicted_label = str(payload.get("predicted_label", "depressed" if total_bdi >= 14 else "control") or "control").strip().lower()
        if predicted_label not in {"control", "depressed"}:
            predicted_label = "depressed" if total_bdi >= 14 else "control"
        confidence = max(0.0, min(1.0, float(payload.get("confidence", fallback.confidence) or fallback.confidence)))
        rationale_by_item = {
            str(item_id): str(value)
            for item_id, value in dict(payload.get("rationale_by_item", fallback.rationale_by_item) or fallback.rationale_by_item).items()
        }
        quote_links_by_item = {
            str(item_id): [str(v) for v in list(values or [])]
            for item_id, values in dict(payload.get("quote_links_by_item", fallback.quote_links_by_item) or fallback.quote_links_by_item).items()
        }
    except Exception:
        return fallback

    return DiagnosisDecision(
        item_scores=item_scores,
        total_bdi=total_bdi,
        predicted_label=predicted_label,
        confidence=confidence,
        rationale_by_item=rationale_by_item,
        quote_links_by_item=quote_links_by_item,
        used_llm=True,
        synthesis_mode="llm",
    )


def diagnosis_agent(state: AgentState) -> Dict[str, Any]:
    diagnosis = _maybe_llm_diagnosis(state, _deterministic_diagnosis(state))
    predicted_key_item_ids = [
        item_id
        for item_id, score in sorted(diagnosis.item_scores.items(), key=lambda pair: (-int(pair[1]), int(pair[0])))
        if int(score) > 0
    ][:4]
    predicted_key_symptoms = [symptom_name_from_item(item_id) for item_id in predicted_key_item_ids]

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["diagnosis_agent"] = {
        "total_bdi": int(diagnosis.total_bdi),
        "predicted_label": str(diagnosis.predicted_label),
        "confidence": round(float(diagnosis.confidence), 4),
        "used_llm": bool(diagnosis.used_llm),
        "synthesis_mode": diagnosis.synthesis_mode,
        "supported_item_count": int(sum(1 for score in diagnosis.item_scores.values() if int(score) > 0)),
        "top_symptoms": list(predicted_key_symptoms),
    }

    return {
        "diagnosis": diagnosis,
        "predicted_bdi_score": int(diagnosis.total_bdi),
        "predicted_label": str(diagnosis.predicted_label),
        "predicted_key_item_ids": predicted_key_item_ids,
        "predicted_key_symptoms": predicted_key_symptoms,
        "final_item_scores": dict(diagnosis.item_scores),
        "final": FinalState(
            predicted_bdi_score=int(diagnosis.total_bdi),
            predicted_label=str(diagnosis.predicted_label),
            top_symptoms=list(predicted_key_symptoms),
            evidence_report={
                "rationale_by_item": dict(diagnosis.rationale_by_item),
                "quote_links_by_item": dict(diagnosis.quote_links_by_item),
            },
            risk_flag=bool(state.get("risk_flag", False)),
            debug_trace=dict(turn_trace["diagnosis_agent"]),
        ),
        "turn_trace": turn_trace,
    }
