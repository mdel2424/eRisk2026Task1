from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.bdi_modules import ITEM_TO_MODULES, MODULE_NAMES, MODULE_TO_ITEMS, MODULE_WEIGHTS
from core.state import (
    AgentState,
    ControlState,
    FinalState,
    ItemBelief,
    coerce_item_belief,
    symptom_name_from_item,
    top_symptoms_from_scores,
)



def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return float(value)



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
            weighted_sum += float(belief.expected_score) * local_weight
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

    return _clamp(imputed_float, 0.0, 3.0), contributions



def _evidence_report(state: AgentState) -> Dict[str, object]:
    evidence_rows = list(state.get("evidence_log", []))
    top_rows: List[Dict[str, object]] = []
    for row in evidence_rows[-6:]:
        item_id = int(getattr(row, "item_id", 0) or 0)
        top_rows.append(
            {
                "item_id": item_id,
                "symptom_name": str(getattr(row, "symptom_name", "") or ""),
                "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
                "intensity": float(getattr(row, "intensity", 0.0) or 0.0),
                "method": str(getattr(row, "method", "") or ""),
                "evidence_text": str(getattr(row, "evidence_text", "") or ""),
            }
        )
    return {
        "evidence_count": len(evidence_rows),
        "recent_evidence": top_rows,
    }


def _rank_key_items(final_item_scores: Dict[int, int], item_details: Dict[str, Dict[str, object]], limit: int = 4) -> List[int]:
    def _rank_key(item_id: int) -> tuple[int, int, int, int]:
        score = int(final_item_scores.get(item_id, 0))
        detail = item_details.get(str(item_id), {})
        source = str(detail.get("source", "imputed"))
        observed_rank = 0 if source in {"observed", "observed_blended"} else 1
        support_count = int(detail.get("support_count", 0) or 0)
        return (-score, observed_rank, -support_count, item_id)

    candidates = [int(item_id) for item_id, score in final_item_scores.items() if int(score) > 0]
    candidates.sort(key=_rank_key)
    return candidates[:limit]



def finalize_outputs(state: AgentState) -> Dict:
    control = state.get("control")
    control_stop = bool(getattr(control, "stop", False))
    control_reason = str(getattr(control, "stop_reason", "") or "")
    should_finalize = control_stop

    turn_trace = dict(state.get("turn_trace", {}))
    final_trace = {
        "turn": int(state.get("turn_index", 0)),
        "ran_final_imputation": bool(should_finalize),
        "control_stop": control_stop,
        "reason": control_reason if control_reason else "continue",
    }

    if not should_finalize:
        final_state = FinalState(
            predicted_bdi_score=int(state.get("predicted_bdi_score") or state.get("raw_predicted_bdi_score") or 0),
            predicted_label=str(state.get("predicted_label") or state.get("raw_predicted_label") or "control"),
            top_symptoms=list(state.get("predicted_key_symptoms") or []),
            evidence_report=_evidence_report(state),
            risk_flag=bool(state.get("risk_flag", False)),
            debug_trace=final_trace,
        )
        turn_trace["finalize_outputs"] = final_trace
        return {
            "final": final_state,
            "turn_trace": turn_trace,
        }

    raw_predicted_bdi_score = int(state.get("raw_predicted_bdi_score") or state.get("predicted_bdi_score") or 0)
    raw_predicted_label = str(state.get("raw_predicted_label") or state.get("predicted_label") or "control")
    risk_flag = bool(state.get("risk_flag", False))
    bdi_threshold = int(os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"))

    prior_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = coerce_item_belief(item_id, prior_beliefs.get(item_id))

    module_stats = _module_stats_from_beliefs(beliefs)
    final_item_scores: Dict[int, int] = {}
    item_details: Dict[str, Dict[str, object]] = {}
    imputed_item_count = 0
    blended_observed_item_count = 0
    blended_item_ids: List[int] = []

    obs_blend_enabled = _env_bool("FINAL_OBS_BLEND_ENABLED", "1")
    obs_blend_conf_threshold = _clamp(_env_float("FINAL_OBS_BLEND_CONF_THRESHOLD", "0.60"), 0.0, 1.0)
    obs_blend_support_max = max(1, int(_env_float("FINAL_OBS_BLEND_SUPPORT_MAX", "2")))
    obs_blend_module_conf_min = _clamp(_env_float("FINAL_OBS_BLEND_MODULE_CONF_MIN", "0.50"), 0.0, 1.0)
    obs_blend_max_alpha = _clamp(_env_float("FINAL_OBS_BLEND_MAX_ALPHA", "0.35"), 0.0, 1.0)

    for item_id in range(1, 22):
        belief = beliefs[item_id]
        if int(belief.support_count) > 0:
            observed_float = _clamp(float(belief.expected_score), 0.0, 3.0)
            observed_int = int(round(observed_float))
            observed_confidence = _clamp(1.0 - float(belief.uncertainty), 0.0, 1.0)
            support_count = int(belief.support_count)
            module_estimate_float, contributions = _impute_missing_item_score(item_id, module_stats)
            best_module_conf = 0.0
            for contribution in contributions:
                try:
                    best_module_conf = max(best_module_conf, float(contribution.get("module_conf", 0.0)))
                except (TypeError, ValueError):
                    continue

            blend_applied = False
            blend_alpha = 0.0
            blend_reason = "high_conf_kept"
            final_float = observed_float

            if not obs_blend_enabled:
                blend_reason = "blend_disabled"
            elif not contributions:
                blend_reason = "no_module_signal"
            elif observed_confidence >= obs_blend_conf_threshold:
                blend_reason = "high_conf_kept"
            elif support_count > obs_blend_support_max:
                blend_reason = "high_support_kept"
            elif best_module_conf < obs_blend_module_conf_min:
                blend_reason = "low_module_conf_kept"
            else:
                confidence_gap = (obs_blend_conf_threshold - observed_confidence) / max(obs_blend_conf_threshold, 1e-6)
                confidence_gap = _clamp(confidence_gap, 0.0, 1.0)
                support_factor = _clamp(
                    float((obs_blend_support_max + 1) - support_count) / float(max(1, obs_blend_support_max)),
                    0.0,
                    1.0,
                )
                module_factor = _clamp(best_module_conf, 0.0, 1.0)
                blend_alpha = _clamp(confidence_gap * support_factor * module_factor, 0.0, obs_blend_max_alpha)
                if blend_alpha > 0.0:
                    final_float = _clamp(
                        ((1.0 - blend_alpha) * observed_float) + (blend_alpha * module_estimate_float),
                        0.0,
                        3.0,
                    )
                    blend_applied = True
                    blend_reason = "low_conf_blended"
                else:
                    blend_reason = "low_conf_blend_zero_alpha"

            final_int = max(0, min(3, int(round(final_float))))
            # Safety rule: never down-adjust observed non-zero risk item.
            if item_id == 9 and observed_int >= 1 and final_int < observed_int:
                final_int = observed_int
                if blend_applied:
                    blend_reason = "risk_item9_floor_applied"

            final_item_scores[item_id] = final_int
            source = "observed_blended" if blend_applied else "observed"
            if blend_applied:
                blended_observed_item_count += 1
                blended_item_ids.append(item_id)
            item_details[str(item_id)] = {
                "source": source,
                "support_count": support_count,
                "expected_score": round(float(belief.expected_score), 6),
                "observed_confidence": round(observed_confidence, 6),
                "module_estimate_float": round(float(module_estimate_float), 6),
                "best_module_conf": round(float(best_module_conf), 6),
                "blend_alpha": round(float(blend_alpha), 6),
                "blend_applied": bool(blend_applied),
                "blend_reason": blend_reason,
                "final_score": final_item_scores[item_id],
                "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
                "contributions": contributions,
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
            "observed_confidence": None,
            "module_estimate_float": round(float(imputed_float), 6),
            "best_module_conf": round(
                max((float(c.get("module_conf", 0.0)) for c in contributions), default=0.0),
                6,
            ),
            "blend_alpha": 0.0,
            "blend_applied": False,
            "blend_reason": "missing_imputed",
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
    ranked_key_item_ids = _rank_key_items(final_item_scores, item_details=item_details, limit=4)
    if ranked_key_item_ids:
        final_key_symptoms = [symptom_name_from_item(item_id) for item_id in ranked_key_item_ids]
    else:
        final_key_symptoms = top_symptoms_from_scores(final_item_scores, limit=4)

    final_trace.update(
        {
            "raw_predicted_bdi_score": raw_predicted_bdi_score,
            "raw_predicted_label": raw_predicted_label,
            "final_bdi_score": final_bdi_score,
            "final_label": final_label,
            "imputed_item_count": imputed_item_count,
            "blended_observed_item_count": blended_observed_item_count,
            "blended_item_ids": blended_item_ids,
            "predicted_key_item_ids": ranked_key_item_ids,
        }
    )

    final_state = FinalState(
        predicted_bdi_score=final_bdi_score,
        predicted_label=final_label,
        top_symptoms=final_key_symptoms,
        evidence_report=_evidence_report(state),
        risk_flag=risk_flag,
        debug_trace=final_trace,
    )

    module_imputation = {
        "module_stats": module_stats,
        "item_details": item_details,
        "imputed_item_count": imputed_item_count,
        "blended_observed_item_count": blended_observed_item_count,
        "blended_item_ids": blended_item_ids,
        "obs_blend_enabled": obs_blend_enabled,
        "obs_blend_conf_threshold": obs_blend_conf_threshold,
        "obs_blend_support_max": obs_blend_support_max,
        "obs_blend_module_conf_min": obs_blend_module_conf_min,
        "obs_blend_max_alpha": obs_blend_max_alpha,
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
    }

    turn_trace["finalize_outputs"] = final_trace

    return {
        "control": ControlState(
            stop=True,
            stop_reason=control_reason or "finalized",
        ),
        "should_stop": True,
        "final": final_state,
        "raw_predicted_bdi_score": raw_predicted_bdi_score,
        "raw_predicted_label": raw_predicted_label,
        "predicted_bdi_score": final_bdi_score,
        "predicted_label": final_label,
        "predicted_key_symptoms": final_key_symptoms,
        "predicted_key_item_ids": ranked_key_item_ids,
        "final_item_scores": final_item_scores,
        "module_imputation": module_imputation,
        "turn_trace": turn_trace,
    }
