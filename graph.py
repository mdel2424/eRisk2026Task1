from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.supervisor import supervisor_router
from agents.specialists import somatic_specialist, cognitive_specialist, risk_specialist

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("somatic", somatic_specialist)
workflow.add_node("cognitive", cognitive_specialist)
workflow.add_node("risk", risk_specialist)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "somatic": "somatic",
        "cognitive": "cognitive",
        "risk": "risk"
    },
)


workflow.add_edge("somatic", END)
workflow.add_edge("cognitive", END)
workflow.add_edge("risk", END)

app = workflow.compile()