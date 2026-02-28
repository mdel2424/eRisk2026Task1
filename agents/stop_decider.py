from __future__ import annotations

import os
from typing import Dict, Tuple

from core.state import AgentState, ControlState, StopDecision



def compute_stop_decision(
    state: AgentState,
    force_risk_stop: bool = False,
) -> Tuple[bool, str, float, int, bool, int, bool]:
    min_turns = int(os.getenv("MIN_TURNS", "4"))
    max_turns = int(os.getenv("MAX_TURNS", "10"))
    stop_confidence = float(os.getenv("STOP_CONFIDENCE", "0.75"))
    min_evidence_for_conf_stop = int(os.getenv("MIN_EVIDENCE_FOR_CONF_STOP", "2"))
    min_items_observed = int(os.getenv("MIN_ITEMS_OBSERVED_FOR_CONF_STOP", "4"))

    turn_index = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    risk_flag = bool(state.get("risk_flag", False))
    confidence = float(state.get("global_confidence", 0.0))
    evidence_total = len(state.get("evidence_log", []))
    evidence_gate_met = evidence_total >= max(0, min_evidence_for_conf_stop)
    item_beliefs = state.get("item_beliefs", {})
    items_observed = 0
    for item_id in range(1, 22):
        belief = item_beliefs.get(item_id)
        if belief is None:
            continue
        try:
            if int(getattr(belief, "support_count", 0)) > 0:
                items_observed += 1
        except (TypeError, ValueError):
            continue
    observed_items_gate_met = items_observed >= max(0, min_items_observed)
    confidence_gate_met = confidence >= stop_confidence and evidence_gate_met and observed_items_gate_met

    should_stop = False
    reason = "continue"

    if force_risk_stop and risk_flag:
        should_stop = True
        reason = "risk_short_circuit"
    elif not has_new_persona_input:
        reason = "opening_turn" if turn_index == 0 else "awaiting_persona_input"
    else:
        if turn_index >= max_turns:
            should_stop = True
            reason = "max_turns reached"
        elif turn_index >= min_turns and (confidence_gate_met or risk_flag):
            should_stop = True
            reason = "calibrated confidence/risk threshold reached"
        elif turn_index >= min_turns and confidence >= stop_confidence and not evidence_gate_met:
            reason = (
                "confidence threshold met but evidence gate blocked "
                f"({evidence_total}/{min_evidence_for_conf_stop})"
            )
        elif turn_index >= min_turns and confidence >= stop_confidence and not observed_items_gate_met:
            reason = (
                "confidence threshold met but observed-item gate blocked "
                f"({items_observed}/{min_items_observed})"
            )

    return should_stop, reason, confidence, evidence_total, evidence_gate_met, items_observed, observed_items_gate_met



def stop_decider(state: AgentState) -> Dict:
    should_stop, stop_reason, confidence, evidence_total, evidence_gate_met, items_observed, observed_items_gate_met = (
        compute_stop_decision(state)
    )

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
        f"conf={confidence:.2f}, risk={bool(state.get('risk_flag', False))}, "
        f"evidence={evidence_total}, gate={evidence_gate_met}, "
        f"items_observed={items_observed}, items_gate={observed_items_gate_met}, "
        f"stop={should_stop} ({stop_reason})"
    )

    turn_trace = dict(state.get("turn_trace", {}))
    stop_trace = {
        "turn": int(state.get("turn_index", 0)),
        "confidence": round(confidence, 4),
        "evidence_total": evidence_total,
        "evidence_gate_met": evidence_gate_met,
        "items_observed": items_observed,
        "observed_items_gate_met": observed_items_gate_met,
        "should_stop": should_stop,
        "reason": stop_reason,
        "label": predicted_label,
        "risk_flag": bool(state.get("risk_flag", False)),
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
