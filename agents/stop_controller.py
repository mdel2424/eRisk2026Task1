from __future__ import annotations

import os
from typing import Any, Dict

from core.state import AgentState, ControlState, StopDecision


def _state_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _mean_uncertainty(state: AgentState) -> float:
    bayes_items = dict(state.get("bayes_items", {}))
    if not bayes_items:
        return 1.0
    return sum(float(_state_value(row, "uncertainty", 1.0) or 1.0) for row in bayes_items.values()) / float(len(bayes_items))


def _coverage_ratio(state: AgentState) -> float:
    bayes_items = dict(state.get("bayes_items", {}))
    if not bayes_items:
        return 0.0
    resolved = 0
    for row in bayes_items.values():
        presence = float(_state_value(row, "presence_prob", 0.0) or 0.0)
        uncertainty = float(_state_value(row, "uncertainty", 1.0) or 1.0)
        if uncertainty <= 0.35 or presence >= 0.55 or presence <= 0.10:
            resolved += 1
    return float(resolved) / float(len(bayes_items))


def stop_controller(state: AgentState) -> Dict[str, Any]:
    turn_index = int(state.get("turn_index", 0) or 0)
    min_turns = max(1, int(os.getenv("MIN_TURNS", "20")))
    max_turns = max(min_turns + 2, int(os.getenv("MAX_TURNS", "40")))
    stop_confidence = max(0.0, min(1.0, float(os.getenv("STOP_CONFIDENCE", "0.66"))))

    diagnosis = state.get("diagnosis")
    predicted_label = str(_state_value(diagnosis, "predicted_label", state.get("predicted_label", "control")) or "control")
    predicted_bdi_score = int(_state_value(diagnosis, "total_bdi", state.get("predicted_bdi_score", 0)) or 0)
    mean_uncertainty = _mean_uncertainty(state)
    coverage_ratio = _coverage_ratio(state)
    risk_prob = float(state.get("risk_prob", 0.0) or 0.0)
    global_confidence = max(0.0, min(1.0, (0.55 * coverage_ratio) + (0.45 * (1.0 - mean_uncertainty))))

    should_stop = False
    reason = "continue_collecting"
    if turn_index >= max_turns:
        should_stop = True
        reason = "max_turn_limit"
    elif turn_index >= min_turns and global_confidence >= stop_confidence and coverage_ratio >= 0.72:
        should_stop = True
        reason = "posterior_convergence"
    elif turn_index >= max(8, min_turns // 2) and risk_prob >= 0.72 and global_confidence >= 0.55:
        should_stop = True
        reason = "risk_escalation_stabilized"

    stop_decision = StopDecision(
        turn=max(1, turn_index),
        should_stop=bool(should_stop),
        reason=reason,
        predicted_label=predicted_label if predicted_label in {"control", "depressed"} else "control",  # type: ignore[arg-type]
        predicted_bdi_score=max(0, min(63, predicted_bdi_score)),
        confidence=global_confidence,
    )

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["stop_controller"] = {
        "turn": max(1, turn_index),
        "should_stop": bool(should_stop),
        "reason": reason,
        "global_confidence": round(float(global_confidence), 4),
        "coverage_ratio": round(float(coverage_ratio), 4),
        "mean_uncertainty": round(float(mean_uncertainty), 4),
        "risk_prob": round(float(risk_prob), 4),
    }

    return {
        "control": ControlState(stop=bool(should_stop), stop_reason=reason),
        "should_stop": bool(should_stop),
        "stop_history": [stop_decision],
        "global_confidence": float(global_confidence),
        "stop_debug": (
            f"Stop controller: stop={bool(should_stop)}; reason={reason}; "
            f"coverage={coverage_ratio:.3f}; mean_uncertainty={mean_uncertainty:.3f}"
        ),
        "turn_trace": turn_trace,
    }

