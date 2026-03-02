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

from agents.belief_update import update_beliefs
from agents.evidence_extraction import extract_likelihoods
from agents.finalize_outputs import finalize_outputs
from agents.ingest_turn import ingest_turn
from agents.policy_metrics import policy_metrics
from agents.question_generator import question_generator
from agents.risk_sentinel import risk_sentinel
from agents.stop_decider import stop_decider
from agents.target_selector import target_selector
from core.state import AgentState

NodeFn = Callable[[AgentState], Dict[str, Any]]


SINGLE_WRITER_KEYS: Dict[str, set[str]] = {
    "risk": {"risk_sentinel"},
    "risk_flag": {"risk_sentinel"},
    "risk_prob": {"risk_sentinel"},
    "control": {"stop_decider", "finalize_outputs"},
    "should_stop": {"stop_decider", "finalize_outputs"},
    "stop_history": {"stop_decider"},
    "next_action": {"target_selector"},
    "next_node": {"target_selector"},
    "active_node": {"target_selector"},
    "route_history": {"target_selector"},
    "route_debug": {"target_selector"},
    "outgoing": {"question_generator"},
    "messages": {"question_generator"},
}

NODE_FORBIDDEN_KEYS: Dict[str, set[str]] = {
    "extract_likelihoods": {
        "beliefs",
        "item_beliefs",
        "control",
        "should_stop",
        "next_action",
        "next_node",
        "active_node",
        "route_history",
        "route_debug",
        "outgoing",
    },
    "belief_update": {
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
    "policy_metrics": {
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
    "question_generator": {
        "predicted_label",
        "predicted_bdi_score",
        "raw_predicted_label",
        "raw_predicted_bdi_score",
        "final_item_scores",
        "module_imputation",
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
        "risk_sentinel": risk_sentinel,
        "extract_likelihoods": extract_likelihoods,
        "belief_update": update_beliefs,
        "policy_metrics": policy_metrics,
        "stop_decider": stop_decider,
        "target_selector": target_selector,
        "question_generator": question_generator,
        "finalize_outputs": finalize_outputs,
    }

    workflow = StateGraph(AgentState)

    for node_name, default_fn in node_map.items():
        impl = node_overrides.get(node_name, default_fn)
        workflow.add_node(node_name, _wrap_node(node_name, impl))

    workflow.set_entry_point("ingest_turn")

    workflow.add_edge("ingest_turn", "risk_sentinel")
    workflow.add_edge("risk_sentinel", "extract_likelihoods")

    workflow.add_edge("extract_likelihoods", "belief_update")
    workflow.add_edge("belief_update", "policy_metrics")
    workflow.add_edge("policy_metrics", "stop_decider")

    workflow.add_conditional_edges(
        "stop_decider",
        _stop_branch,
        {
            "stop": "finalize_outputs",
            "continue": "target_selector",
        },
    )

    workflow.add_edge("target_selector", "question_generator")
    workflow.add_edge("question_generator", "finalize_outputs")

    workflow.add_edge("finalize_outputs", END)

    return workflow.compile()


app = build_app()
