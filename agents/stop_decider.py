from __future__ import annotations

import os
from typing import Dict, Tuple

from core.state import AgentState, ControlState, StopDecision



def compute_stop_decision(state: AgentState) -> Tuple[bool, str, float]:
    min_turns = int(os.getenv("MIN_TURNS", "20"))
    max_turns = int(os.getenv("MAX_TURNS", "40"))
    stop_confidence = float(os.getenv("STOP_CONFIDENCE", "0.66"))

    turn_index = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    confidence = float(state.get("global_confidence", 0.0))

    should_stop = False
    reason = "continue"

    if not has_new_persona_input:
        reason = "opening_turn" if turn_index == 0 else "awaiting_persona_input"
    else:
        if turn_index >= max_turns:
            should_stop = True
            reason = "max_turns_reached"
        elif turn_index >= min_turns and confidence >= stop_confidence:
            should_stop = True
            reason = "confidence_threshold_reached"

    return should_stop, reason, confidence



def stop_decider(state: AgentState) -> Dict:
    should_stop, stop_reason, confidence = compute_stop_decision(state)

    predicted_label = str(state.get("raw_predicted_label") or state.get("predicted_label") or "control")
    if predicted_label not in {"control", "depressed"}:
        predicted_label = "control"

    predicted_bdi_score = int(state.get("raw_predicted_bdi_score") or state.get("predicted_bdi_score") or 0)

    stop_history_payload = []
    if bool(state.get("has_new_persona_input", False)):
        stop_record = StopDecision(
            turn=max(1, int(state.get("turn_index", 0))),
            should_stop=should_stop,
            reason=stop_reason,
            predicted_label=predicted_label,
            predicted_bdi_score=max(0, min(63, predicted_bdi_score)),
            confidence=max(0.0, min(1.0, confidence)),
        )
        stop_history_payload = [stop_record]

    debug_line = (
        f"Stop decider: turn={int(state.get('turn_index', 0))}, "
        f"conf={confidence:.2f}, threshold={float(os.getenv('STOP_CONFIDENCE', '0.66')):.2f}, "
        f"risk={bool(state.get('risk_flag', False))}, "
        f"stop={should_stop} ({stop_reason})"
    )

    turn_trace = dict(state.get("turn_trace", {}))
    stop_trace = {
        "turn": int(state.get("turn_index", 0)),
        "confidence": round(confidence, 4),
        "confidence_source": "support_coverage_saturation_smoothed",
        "should_stop": should_stop,
        "reason": stop_reason,
        "label": predicted_label,
        "risk_flag": bool(state.get("risk_flag", False)),
        "min_turns": int(os.getenv("MIN_TURNS", "20")),
        "max_turns": int(os.getenv("MAX_TURNS", "40")),
        "stop_confidence": float(os.getenv("STOP_CONFIDENCE", "0.66")),
    }
    turn_trace["stop_decider"] = stop_trace
    turn_trace["stop"] = stop_trace

    return {
        "control": ControlState(stop=should_stop, stop_reason=stop_reason),
        "should_stop": should_stop,
        "stop_debug": debug_line,
        "stop_history": stop_history_payload,
        "turn_trace": turn_trace,
    }
