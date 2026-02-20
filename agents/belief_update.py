from __future__ import annotations

from typing import Dict

from core.calibration import build_feature_vector, get_calibrator_bundle, predict_with_explanations
from core.state import (
    AgentState,
    ItemBelief,
    SPECIALIST_ITEM_MAP,
    bump_failure_counter,
    top_symptoms_from_beliefs,
)


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


def _update_single_belief(belief: ItemBelief, evidence, turn: int) -> ItemBelief:
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
