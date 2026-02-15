from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.specialists import somatic_specialist, cognitive_specialist

workflow = StateGraph(AgentState)

workflow.add_node("somatic", somatic_specialist)
workflow.add_node("cognitive", cognitive_specialist)

workflow.set_entry_point("somatic")
workflow.add_edge("somatic", END)
