from core.state import AgentState

def somatic_specialist(state: AgentState):
    return {"messages": [{"role": "assistant", "content": "Somatic follow-up (indirect)."}]}

def cognitive_specialist(state: AgentState):
    return {"messages": [{"role": "assistant", "content": "Cognitive follow-up (indirect)."}]}

def risk_specialist(state: AgentState):
    return {"messages": [{"role": "assistant", "content": "Risk-aware safety follow-up (indirect)."}]}
