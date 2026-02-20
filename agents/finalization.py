from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.bdi_modules import ITEM_TO_MODULES, MODULE_NAMES, MODULE_TO_ITEMS, MODULE_WEIGHTS
from core.state import AgentState, ItemBelief, top_symptoms_from_scores

from agents.belief_update import _coerce_belief


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
