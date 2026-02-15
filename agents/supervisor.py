from core.state import AgentState

def supervisor_router(state: AgentState):
    text = state["messages"][-1]["content"].lower() if state["messages"] else ""

    risk_cues = ["die", "dead", "worthless", "suicide", "end it