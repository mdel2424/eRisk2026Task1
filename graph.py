from __future__ import annotations

import os
from typing import Any, Callable, Dict

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal test/runtime environments
    END = "__END__"

    class StateGraph:  # type: ignore[override]
        def __init__(self, _state_type) -> None:
            self._nodes: Dict[str, Callable] = {}
            self._entry_point: str | None = None
            self._edges: Dict[str, str] = {}
            self._conditional_edges: Dict[str, tuple[Callable, Dict[str, str]]] = {}

        def add_node(self, name: str, fn: Callable) -> None:
            self._nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self._entry_point = name

        def add_edge(self, source: str, target: str) -> None:
            self._edges[source] = target

        def add_conditional_edges(
            self,
            source: str,
            condition: Callable,
            mapping: Dict[str, str],
        ) -> None:
            self._conditional_edges[source] = (condition, mapping)

        def compile(self):
            if not self._entry_point:
                raise ValueError("Entry point is required before compile().")

            nodes = dict(self._nodes)
            edges = dict(self._edges)
            conditional_edges = dict(self._conditional_edges)
            entry = str(self._entry_point)

            class _CompiledGraph:
                def __init__(self) -> None:
                    self._nodes = nodes
                    self._edges = edges
                    self._conditional_edges = conditional_edges
                    self._entry = entry

                @staticmethod
                def _merge_state(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
                    merged = dict(state)
                    for key, value in delta.items():
                        if key in {"messages", "evidence_log", "route_history", "stop_history", "trace_log"}:
                            prev = merged.get(key, [])
                            if isinstance(prev, list) and isinstance(value, list):
                                merged[key] = prev + value
                                continue
                        merged[key] = value
                    return merged

                def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
                    current = self._entry
                    working = dict(state)
                    while current != END:
                        if current not in self._nodes:
                            raise KeyError(f"Node '{current}' is not registered")
                        delta = self._nodes[current](working)
                        if not isinstance(delta, dict):
                            raise TypeError(f"Node '{current}' returned non-dict delta: {type(delta)}")
                        working = self._merge_state(working, delta)
                        if current in self._conditional_edges:
                            condition, mapping = self._conditional_edges[current]
                            branch = condition(working)
                            if branch not in mapping:
                                raise KeyError(
                                    f"Conditional branch '{branch}' is not mapped from node '{current}'"
                                )
                            current = mapping[branch]
                        elif current in self._edges:
                            current = self._edges[current]
                        else:
                            raise KeyError(f"No outgoing edge from node '{current}'")
                    return working

            return _CompiledGraph()

from agents.bayes_state_update import bayes_state_update
from agents.diagnosis_agent import diagnosis_agent
from agents.ingest_turn import ingest_turn
from agents.judgment_agent import judgment_agent
from agents.navigation_agent import navigation_agent
from agents.question_agent import question_agent
from agents.stop_controller import stop_controller
from core.state import AgentState

NodeFn = Callable[[AgentState], Dict[str, Any]]


SINGLE_WRITER_KEYS: Dict[str, set[str]] = {
    "risk_flag": {"bayes_state_update"},
    "risk_prob": {"bayes_state_update"},
    "control": {"stop_controller"},
    "should_stop": {"stop_controller"},
    "stop_history": {"stop_controller"},
    "next_action": {"navigation_agent"},
    "question_plan": {"navigation_agent"},
    "next_node": {"navigation_agent"},
    "active_node": {"navigation_agent"},
    "route_history": {"navigation_agent"},
    "route_debug": {"navigation_agent"},
    "outgoing": {"question_agent"},
    "messages": {"question_agent"},
    "judgment": {"judgment_agent"},
    "opening_signal_cluster": {"judgment_agent"},
    "opening_signal_item_ids": {"judgment_agent"},
    "opening_signal_turn": {"judgment_agent"},
    "opening_bootstrap_applied": {"judgment_agent"},
    "bayes_nodes": {"bayes_state_update"},
    "bayes_items": {"bayes_state_update"},
    "diagnosis": {"diagnosis_agent"},
    "predicted_label": {"diagnosis_agent"},
    "predicted_bdi_score": {"diagnosis_agent"},
    "predicted_key_item_ids": {"diagnosis_agent"},
    "predicted_key_symptoms": {"diagnosis_agent"},
    "final_item_scores": {"diagnosis_agent"},
    "opening_followup_cluster": {"navigation_agent"},
    "opening_cognitive_anchor_preserved": {"navigation_agent"},
}

NODE_FORBIDDEN_KEYS: Dict[str, set[str]] = {
    "judgment_agent": {
        "control",
        "should_stop",
        "next_action",
        "next_node",
        "active_node",
        "route_history",
        "route_debug",
        "outgoing",
    },
    "bayes_state_update": {
        "control",
        "should_stop",
        "next_action",
        "next_node",
        "active_node",
        "route_history",
        "route_debug",
        "outgoing",
        "messages",
    },
    "diagnosis_agent": {
        "control",
        "should_stop",
        "next_action",
        "next_node",
        "active_node",
        "route_history",
        "route_debug",
        "outgoing",
        "messages",
    },
}



def _strict_invariants_enabled() -> bool:
    raw = os.getenv("GRAPH_VALIDATE_INVARIANTS", "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}



def _assert_invariants(node_name: str, delta: Dict[str, Any]) -> None:
    keys = set(delta.keys())

    for key, writers in SINGLE_WRITER_KEYS.items():
        if key in keys and node_name not in writers:
            raise AssertionError(
                f"Invariant violation: key '{key}' can only be written by {sorted(writers)}, got {node_name}"
            )

    forbidden = NODE_FORBIDDEN_KEYS.get(node_name, set())
    illegal = sorted(key for key in keys if key in forbidden)
    if illegal:
        raise AssertionError(
            f"Invariant violation: node '{node_name}' wrote forbidden keys: {illegal}"
        )



def _wrap_node(name: str, node_fn: NodeFn) -> NodeFn:
    def _wrapped(state: AgentState) -> Dict[str, Any]:
        delta = node_fn(state)
        if not isinstance(delta, dict):
            raise TypeError(f"Node '{name}' must return dict, got {type(delta)}")
        if _strict_invariants_enabled():
            _assert_invariants(name, delta)
        return delta

    return _wrapped


def _stop_branch(state: AgentState) -> str:
    return "stop" if bool(state.get("should_stop", False)) else "continue"



def build_app(node_overrides: dict[str, NodeFn] | None = None):
    node_overrides = dict(node_overrides or {})

    node_map: Dict[str, NodeFn] = {
        "ingest_turn": ingest_turn,
        "judgment_agent": judgment_agent,
        "bayes_state_update": bayes_state_update,
        "diagnosis_agent": diagnosis_agent,
        "stop_controller": stop_controller,
        "navigation_agent": navigation_agent,
        "question_agent": question_agent,
    }

    workflow = StateGraph(AgentState)

    for node_name, default_fn in node_map.items():
        impl = node_overrides.get(node_name, default_fn)
        workflow.add_node(node_name, _wrap_node(node_name, impl))

    workflow.set_entry_point("ingest_turn")

    workflow.add_edge("ingest_turn", "judgment_agent")
    workflow.add_edge("judgment_agent", "bayes_state_update")
    workflow.add_edge("bayes_state_update", "diagnosis_agent")
    workflow.add_edge("diagnosis_agent", "stop_controller")

    workflow.add_conditional_edges(
        "stop_controller",
        _stop_branch,
        {
            "stop": END,
            "continue": "navigation_agent",
        },
    )

    workflow.add_edge("navigation_agent", "question_agent")
    workflow.add_edge("question_agent", END)

    return workflow.compile()


app = build_app()
