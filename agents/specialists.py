from core.state import AgentState

def somatic_specialist(state: AgentState):
    # Logic to probe 11, 15, 16,18, 20
    return {"messages": [{"role": "assistant", "content": "I'm the Somatic Agent."}]}

def cognitive_specialist(state: AgentState):
    # Logic to probe 2, 3, 5, 7, 8, 14
    return {"messages": [{"role": "assistant", "content": "I'm the Cognitive Agent."}]}
