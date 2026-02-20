import os

from langgraph.graph import END, StateGraph

from agents.assessment import extract_evidence, update_beliefs
from agents.specialists import cognitive_specialist, risk_specialist, somatic_specialist
from agents.supervisor import supervisor_router
from core.state import (
    AgentState,
    StopDecision,
)


def assess_stop(state: AgentState):
    min_turns = int(os.getenv("MIN_TURNS", "4"))
    max_turns = int(os.getenv("MAX_TURNS", "10"))
    stop_confidence = float(os.getenv("STOP_CONFIDENCE", "0.75"))
    min_evidence_for_conf_stop = int(os.getenv("MIN_EVIDENCE_FOR_CONF_STOP", "2"))

    turn_index = state.get("turn_index", 0) + 1
    risk_flag = bool(state.get("risk_flag", False))
    predicted_bdi_score = int(state.get("predicted_bdi_score") or 0)
    predicted_label = str(state.get("predicted_label") or "control")
    if predicted_label not in {"control", "depressed"}:
        predicted_label = "control"
    confidence = float(state.get("global_confidence", 0.0))
    evidence_total = len(state.get("evidence_log", []))
    evidence_gate_met = evidence_total >= max(0, min_evidence_for_conf_stop)
    confidence_gate_met = confidence >= stop_confidence and evidence_gate_met

    should_stop = False
    stop_reason = "continue"
    if turn_index >= max_turns:
        should_stop = True
        stop_reason = "max_turns reached"
    elif turn_index >= min_turns and (confidence_gate_met or risk_flag):
        should_stop = True
        stop_reason = "calibrated confidence/risk threshold reached"
    elif turn_index >= min_turns and confidence >= stop_confidence and not evidence_gate_met:
        stop_reason = (
            "confidence threshold met but evidence gate blocked "
            f"({evidence_total}/{min_evidence_for_conf_stop})"
        )

    stop_record = StopDecision(
        turn=turn_index,
        should_stop=should_stop,
        reason=stop_reason,
        predicted_label=predicted_label,
        predicted_bdi_score=max(0, min(63, predicted_bdi_score)),
        confidence=max(0.0, min(1.0, confidence)),
    )

    debug_line = (
        f"Assess stop: turn={turn_index}, bdi={predicted_bdi_score}, "
        f"conf={confidence:.2f}, risk={risk_flag}, label={predicted_label}, "
        f"evidence={evidence_total}, gate={evidence_gate_met}, "
        f"stop={should_stop} ({stop_reason})"
    )
    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["stop"] = {
        "turn": turn_index,
        "confidence": round(confidence, 4),
        "stop_confidence": stop_confidence,
        "evidence_total": evidence_total,
        "min_evidence_for_conf_stop": min_evidence_for_conf_stop,
        "evidence_gate_met": evidence_gate_met,
        "should_stop": should_stop,
        "reason": stop_reason,
        "label": predicted_label,
        "risk_flag": risk_flag,
    }
    trace_entry = {
        "turn": turn_index,
        "turn_trace": turn_trace,
        "route_debug": state.get("route_debug", ""),
        "specialist_debug": state.get("specialist_debug", ""),
        "stop_debug": debug_line,
        "failure_counters": dict(state.get("failure_counters", {})),
        "empty_evidence_streak": int(state.get("empty_evidence_streak", 0)),
    }

    return {
        "turn_index": turn_index,
        "predicted_label": predicted_label,
        "should_stop": should_stop,
        "stop_debug": debug_line,
        "stop_history": [stop_record],
        "turn_trace": turn_trace,
        "trace_log": [trace_entry],
    }


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("somatic", somatic_specialist)
workflow.add_node("cognitive", cognitive_specialist)
workflow.add_node("risk", risk_specialist)
workflow.add_node("extract_evidence", extract_evidence)
workflow.add_node("update_beliefs", update_beliefs)
workflow.add_node("assess_stop", assess_stop)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "somatic": "somatic",
        "cognitive": "cognitive",
        "risk": "risk",
    },
)

workflow.add_edge("somatic", "extract_evidence")
workflow.add_edge("cognitive", "extract_evidence")
workflow.add_edge("risk", "extract_evidence")

workflow.add_edge("extract_evidence", "update_beliefs")
workflow.add_edge("update_beliefs", "assess_stop")

workflow.add_edge("assess_stop", END)

app = workflow.compile()
