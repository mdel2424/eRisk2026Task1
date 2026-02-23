from __future__ import annotations

import math
from typing import Dict, List

from core.calibration import (
    build_feature_vector,
    get_calibrator_bundle,
    predict_with_explanations,
)
from core.state import AgentState, ItemBelief, PolicyMetricsState, coerce_item_belief



def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)



def policy_metrics(state: AgentState) -> Dict:
    raw_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = coerce_item_belief(item_id, raw_beliefs.get(item_id))

    total_expected_bdi = sum(float(belief.expected_score) for belief in beliefs.values())
    covered_items = [item_id for item_id, belief in beliefs.items() if int(belief.support_count) > 0]
    coverage = float(len(covered_items)) / 21.0
    mean_entropy = sum(float(belief.entropy) for belief in beliefs.values()) / 21.0

    ranked_entropy = sorted(
        beliefs.items(),
        key=lambda pair: (float(pair[1].entropy), -int(pair[1].support_count)),
        reverse=True,
    )
    top_uncertain_items = [item_id for item_id, _ in ranked_entropy[:5]]

    ig_estimates: Dict[int, float] = {}
    for item_id, belief in beliefs.items():
        unresolved_bonus = 1.0 if int(belief.support_count) == 0 else 0.25
        ig_estimates[item_id] = round(float(belief.entropy) * unresolved_bonus, 6)

    confidence_rows = []
    for row in list(state.get("latest_turn_likelihoods", [])):
        try:
            confidence_rows.append(float(getattr(row, "extract_confidence", 0.0)))
        except (TypeError, ValueError):
            continue

    risk_flag = bool(state.get("risk_flag", False))
    feature_vector = build_feature_vector(beliefs, confidence_rows, risk_flag)
    bundle = get_calibrator_bundle()
    prediction = predict_with_explanations(feature_vector, bundle)

    label_prob = _sigmoid(float(prediction.raw_label_score))
    metrics = PolicyMetricsState(
        total_expected_bdi=max(0.0, min(63.0, total_expected_bdi)),
        label_prob=max(0.0, min(1.0, label_prob)),
        coverage=max(0.0, min(1.0, coverage)),
        mean_entropy=max(0.0, min(2.0, mean_entropy)),
        top_uncertain_items=top_uncertain_items,
        last_ig_estimates=ig_estimates,
    )

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

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["policy_metrics"] = {
        "turn": int(state.get("turn_index", 0)),
        "coverage": round(metrics.coverage, 4),
        "mean_entropy": round(metrics.mean_entropy, 4),
        "total_expected_bdi": round(metrics.total_expected_bdi, 4),
        "label_prob": round(metrics.label_prob or 0.0, 4),
    }
    # Compatibility key expected by diagnostics/eval counters.
    turn_trace["update_beliefs"] = {
        "turn": int(state.get("turn_index", 0)),
        "active_node": str(state.get("active_node", "cognitive")),
        "updated_item_ids": [
            int(item_id)
            for item_id in range(1, 22)
            if int(beliefs[item_id].last_update_turn) == int(state.get("turn_index", 0))
        ],
        "risk_flag": risk_flag,
        "calibrator_mode": prediction.mode,
        "global_confidence": round(float(prediction.global_confidence), 4),
        "positive_features": [row["feature"] for row in positive[:3]],
        "negative_features": [row["feature"] for row in negative[:3]],
    }

    return {
        "metrics": metrics,
        "latest_feature_vector": feature_vector,
        "calibrator_mode": prediction.mode,
        "positive_contributions": positive,
        "negative_contributions": negative,
        "global_confidence": float(prediction.global_confidence),
        "raw_predicted_bdi_score": int(prediction.predicted_bdi_score),
        "raw_predicted_label": str(prediction.predicted_label),
        "turn_trace": turn_trace,
    }
