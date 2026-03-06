from __future__ import annotations

import math
import os
from typing import Dict, List

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

    risk_flag = bool(state.get("risk_flag", False))

    # Single clinician-style confidence: support + coverage saturation with near-monotonic smoothing.
    try:
        conf_support_tau = float(os.getenv("CONF_SUPPORT_TAU", "1.25"))
    except (TypeError, ValueError):
        conf_support_tau = 1.25
    conf_support_tau = max(1e-6, conf_support_tau)

    try:
        conf_depth_weight = float(os.getenv("CONF_DEPTH_WEIGHT", "0.70"))
    except (TypeError, ValueError):
        conf_depth_weight = 0.70
    try:
        conf_coverage_weight = float(os.getenv("CONF_COVERAGE_WEIGHT", "0.30"))
    except (TypeError, ValueError):
        conf_coverage_weight = 0.30
    conf_depth_weight = max(0.0, conf_depth_weight)
    conf_coverage_weight = max(0.0, conf_coverage_weight)
    weight_sum = conf_depth_weight + conf_coverage_weight
    if weight_sum <= 1e-8:
        conf_depth_weight = 0.70
        conf_coverage_weight = 0.30
    else:
        conf_depth_weight = conf_depth_weight / weight_sum
        conf_coverage_weight = conf_coverage_weight / weight_sum

    item_confidences = []
    for item_id in range(1, 22):
        support_i = max(0.0, float(getattr(beliefs[item_id], "support_count", 0)))
        item_conf = 1.0 - math.exp(-support_i / conf_support_tau)
        item_confidences.append(max(0.0, min(1.0, item_conf)))
    depth_confidence = sum(item_confidences) / float(len(item_confidences)) if item_confidences else 0.0
    coverage_confidence = max(0.0, min(1.0, coverage))
    target_confidence = max(
        0.0,
        min(1.0, (conf_depth_weight * depth_confidence) + (conf_coverage_weight * coverage_confidence)),
    )

    try:
        conf_up_alpha = float(os.getenv("CONF_UP_ALPHA", "0.55"))
    except (TypeError, ValueError):
        conf_up_alpha = 0.55
    conf_up_alpha = max(0.0, min(1.0, conf_up_alpha))

    try:
        conf_decay_streak_start = int(os.getenv("CONF_DECAY_STREAK_START", "6"))
    except (TypeError, ValueError):
        conf_decay_streak_start = 6
    try:
        conf_decay_per_turn = float(os.getenv("CONF_DECAY_PER_TURN", "0.002"))
    except (TypeError, ValueError):
        conf_decay_per_turn = 0.002
    try:
        conf_decay_max = float(os.getenv("CONF_DECAY_MAX", "0.01"))
    except (TypeError, ValueError):
        conf_decay_max = 0.01
    try:
        conf_max_drop_per_turn = float(os.getenv("CONF_MAX_DROP_PER_TURN", "0.01"))
    except (TypeError, ValueError):
        conf_max_drop_per_turn = 0.01
    conf_decay_streak_start = max(1, conf_decay_streak_start)
    conf_decay_per_turn = max(0.0, conf_decay_per_turn)
    conf_decay_max = max(0.0, conf_decay_max)
    conf_max_drop_per_turn = max(0.0, conf_max_drop_per_turn)

    prev_confidence = max(0.0, min(1.0, float(state.get("global_confidence", 0.0))))
    if target_confidence > prev_confidence:
        smoothed_confidence = prev_confidence + (conf_up_alpha * (target_confidence - prev_confidence))
    else:
        smoothed_confidence = prev_confidence

    empty_evidence_streak = max(0, int(state.get("empty_evidence_streak", 0)))
    decay_applied = 0.0
    if empty_evidence_streak >= conf_decay_streak_start:
        decay_applied = min(
            conf_decay_max,
            conf_decay_per_turn * float((empty_evidence_streak - conf_decay_streak_start) + 1),
        )
        smoothed_confidence -= decay_applied

    global_confidence = max(0.0, min(1.0, smoothed_confidence))
    global_confidence = max(global_confidence, prev_confidence - conf_max_drop_per_turn)
    global_confidence = max(0.0, min(1.0, global_confidence))

    # Simple sigmoid estimate for label probability (logging only).
    bdi_threshold = float(os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"))
    label_prob = _sigmoid((total_expected_bdi - bdi_threshold) / 6.0)

    predicted_bdi_score = max(0, min(63, int(round(total_expected_bdi))))
    predicted_label = "depressed" if (predicted_bdi_score >= int(bdi_threshold) or risk_flag) else "control"

    metrics = PolicyMetricsState(
        total_expected_bdi=max(0.0, min(63.0, total_expected_bdi)),
        label_prob=max(0.0, min(1.0, label_prob)),
        coverage=max(0.0, min(1.0, coverage)),
        mean_entropy=max(0.0, min(2.0, mean_entropy)),
        top_uncertain_items=top_uncertain_items,
        last_ig_estimates=ig_estimates,
    )

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["policy_metrics"] = {
        "turn": int(state.get("turn_index", 0)),
        "coverage": round(metrics.coverage, 4),
        "mean_entropy": round(metrics.mean_entropy, 4),
        "total_expected_bdi": round(metrics.total_expected_bdi, 4),
        "label_prob": round(metrics.label_prob or 0.0, 4),
        "global_confidence": round(global_confidence, 4),
        "target_confidence": round(target_confidence, 4),
        "depth_confidence": round(depth_confidence, 4),
        "coverage_confidence": round(coverage_confidence, 4),
        "empty_evidence_streak": empty_evidence_streak,
        "decay_applied": round(decay_applied, 6),
        "confidence_source": "support_coverage_saturation_smoothed",
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
        "global_confidence": round(global_confidence, 4),
        "target_confidence": round(target_confidence, 4),
        "depth_confidence": round(depth_confidence, 4),
        "coverage_confidence": round(coverage_confidence, 4),
        "empty_evidence_streak": empty_evidence_streak,
        "decay_applied": round(decay_applied, 6),
        "confidence_source": "support_coverage_saturation_smoothed",
    }

    return {
        "metrics": metrics,
        "global_confidence": float(global_confidence),
        "raw_predicted_bdi_score": predicted_bdi_score,
        "raw_predicted_label": predicted_label,
        "turn_trace": turn_trace,
    }
