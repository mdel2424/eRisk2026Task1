import os

from langgraph.graph import END, StateGraph

from agents.specialists import cognitive_specialist, risk_specialist, somatic_specialist
from agents.supervisor import supervisor_router
from core.state import (
    AgentState,
    bdi_score_from_scalar,
    top_symptoms_from_hits,
)


def assess_stop(state: AgentState):
    min_turns = int(os.getenv("MIN_TURNS", "4"))
    max_turns = int(os.getenv("MAX_TURNS", "10"))
    stop_confidence = float(os.getenv("STOP_CONFIDENCE", "0.75"))

    turn_index = state.get("turn_index", 0) + 1
    score = float(state.get("depression_score", 0.0))
    risk_flag = bool(state.get("risk_flag", False))
    symptom_hits = list(state.get("symptom_hits", []))

    predicted_bdi_score = bdi_score_from_scalar(score)
    predicted_key_symptoms = top_symptoms_from_hits(symptom_hits, limit=4)
    predicted_label = "depressed" if (predicted_bdi_score >= 14 or risk_flag) else "control"

    should_stop = False
    stop_reason = "continue"
    if turn_index >= max_turns:
        should_stop = True
        stop_reason = "max_turns reached"
    elif turn_index >= min_turns and (score >= stop_confidence or risk_flag):
        should_stop = True
        stop_reason = "confidence/risk threshold reached"

    debug_line = (
        f"Assess stop: turn={turn_index}, score={score:.2f}, risk={risk_flag}, "
        f"label={predicted_label}, stop={should_stop} ({stop_reason})"
    )

    return {
        "turn_index": turn_index,
        "predicted_bdi_score": predicted_bdi_score,
        "predicted_key_symptoms": predicted_key_symptoms,
        "predicted_label": predicted_label,
        "should_stop": should_stop,
        "stop_debug": debug_line,
    }


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("somatic", somatic_specialist)
workflow.add_node("cognitive", cognitive_specialist)
workflow.add_node("risk", risk_specialist)
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

workflow.add_edge("somatic", "assess_stop")
workflow.add_edge("cognitive", "assess_stop")
workflow.add_edge("risk", "assess_stop")

workflow.add_edge("assess_stop", END)

app = workflow.compile()
