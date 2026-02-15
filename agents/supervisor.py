from core.state import AgentState

def supervisor_router(state: AgentState):
    text = state["messages"][-1]["content"].lower() if state["messages"] else ""

    risk_cues = ["die", "dead", "worthless", "suicide", "end it"]
    somatic_cues = ["sleep", "tired", "appetite", "eat", "energy"]

    if any(cue in text for cue in risk_cues):
        next_node = "risk"
    elif any(cue in text for cue in somatic_cues):
        next_node = "somatic"
    else:
        next_node = "cognitive"

    return {"next_node": next_node}
