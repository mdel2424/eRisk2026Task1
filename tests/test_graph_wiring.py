from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core.state import (
    AgentState,
    ControlState,
    FinalState,
    NextAction,
    PolicyMetricsState,
    RiskState,
    RouteDecision,
    StopDecision,
    TurnState,
    build_initial_state,
)
from graph import build_app



def _base_state() -> AgentState:
    state = build_initial_state(persona_id="test")
    return state


class GraphWiringTests(unittest.TestCase):
    def test_risk_utterance_short_circuits_to_finalization(self) -> None:
        state = _base_state()
        state["messages"].append({"role": "assistant", "content": "Sometimes I feel better off dead."})

        def ingest_node(_: AgentState):
            return {
                "turn": TurnState(
                    latest_text_raw="Sometimes I feel better off dead.",
                    latest_text_norm="sometimes i feel better off dead",
                    latest_sentences=["Sometimes I feel better off dead"],
                    turn_id=1,
                ),
                "turn_index": 1,
                "has_new_persona_input": True,
                "last_processed_persona_msg_idx": 0,
            }

        def risk_node(_: AgentState):
            return {
                "risk": RiskState(
                    risk_prob=0.99,
                    risk_flag=True,
                    evidence_spans=["better off dead"],
                    reason="active lexical cue",
                    last_updated_turn=1,
                    short_circuit=True,
                ),
                "risk_flag": True,
                "risk_prob": 0.99,
            }

        def finalize_node(_: AgentState):
            return {
                "final": FinalState(
                    predicted_bdi_score=22,
                    predicted_label="depressed",
                    top_symptoms=["Suicidal Thoughts or Wishes"],
                    evidence_report={"evidence_count": 1},
                    risk_flag=True,
                    debug_trace={"path": "risk_short_circuit"},
                ),
                "should_stop": True,
            }

        def fail_if_called(_: AgentState):
            raise AssertionError("This node should not run on risk short-circuit")

        app = build_app(
            {
                "ingest_turn": ingest_node,
                "risk_sentinel": risk_node,
                "extract_likelihoods": fail_if_called,
                "belief_update": fail_if_called,
                "policy_metrics": fail_if_called,
                "stop_decider": fail_if_called,
                "target_selector": fail_if_called,
                "question_generator": fail_if_called,
                "finalize_outputs": finalize_node,
            }
        )

        out = app.invoke(state)
        self.assertTrue(out["risk_flag"])
        self.assertTrue(out["should_stop"])
        self.assertEqual(out["final"].debug_trace.get("path"), "risk_short_circuit")

    def test_non_risk_routes_to_question_generator(self) -> None:
        state = _base_state()
        state["messages"].append({"role": "assistant", "content": "I have just been tired lately."})

        def ingest_node(_: AgentState):
            return {
                "turn": TurnState(
                    latest_text_raw="I have just been tired lately.",
                    latest_text_norm="i have just been tired lately",
                    latest_sentences=["I have just been tired lately"],
                    turn_id=1,
                ),
                "turn_index": 1,
                "has_new_persona_input": True,
                "last_processed_persona_msg_idx": 0,
            }

        def risk_node(_: AgentState):
            return {
                "risk": RiskState(risk_prob=0.05, risk_flag=False, evidence_spans=[], reason="none", last_updated_turn=1),
                "risk_flag": False,
                "risk_prob": 0.05,
            }

        def extract_node(_: AgentState):
            return {"latest_turn_likelihoods": [], "latest_turn_evidence": [], "evidence_log": []}

        def belief_node(s: AgentState):
            return {"beliefs": s["beliefs"], "item_beliefs": s["item_beliefs"]}

        def metrics_node(_: AgentState):
            return {
                "metrics": PolicyMetricsState(
                    total_expected_bdi=2.0,
                    label_prob=0.2,
                    coverage=0.1,
                    mean_entropy=1.8,
                    top_uncertain_items=[15, 2, 4],
                    last_ig_estimates={15: 2.2, 2: 1.7, 4: 1.2},
                ),
                "global_confidence": 0.62,
                "raw_predicted_bdi_score": 6,
                "raw_predicted_label": "control",
            }

        def stop_node(_: AgentState):
            return {
                "control": ControlState(stop=False, stop_reason="continue"),
                "should_stop": False,
                "stop_history": [
                    StopDecision(
                        turn=1,
                        should_stop=False,
                        reason="continue",
                        predicted_label="control",
                        predicted_bdi_score=6,
                        confidence=0.62,
                    )
                ],
            }

        def target_node(_: AgentState):
            return {
                "next_action": NextAction(
                    target_item_id=15,
                    route="somatic",
                    style="gentle_probe",
                    rationale="highest uncertainty",
                ),
                "next_node": "somatic",
                "active_node": "somatic",
                "route_history": [
                    RouteDecision(
                        turn=1,
                        chosen_node="somatic",
                        policy="entropy_penalized",
                        reason="highest uncertainty",
                        target_items=[15],
                        expected_gain=2.2,
                    )
                ],
            }

        def question_node(_: AgentState):
            text = "How has your energy changed during a typical day this week?"
            return {
                "outgoing": {"detector_message": text},
                "messages": [{"role": "user", "content": text}],
            }

        def finalize_node(s: AgentState):
            return {
                "final": FinalState(
                    predicted_bdi_score=int(s.get("raw_predicted_bdi_score", 0) or 0),
                    predicted_label="control",
                    top_symptoms=[],
                    evidence_report={"evidence_count": len(s.get("evidence_log", []))},
                    risk_flag=False,
                    debug_trace={"path": "continue"},
                )
            }

        app = build_app(
            {
                "ingest_turn": ingest_node,
                "risk_sentinel": risk_node,
                "extract_likelihoods": extract_node,
                "belief_update": belief_node,
                "policy_metrics": metrics_node,
                "stop_decider": stop_node,
                "target_selector": target_node,
                "question_generator": question_node,
                "finalize_outputs": finalize_node,
            }
        )

        out = app.invoke(state)
        self.assertFalse(out["should_stop"])
        self.assertEqual(out["messages"][-1]["role"], "user")
        self.assertIn("energy", out["messages"][-1]["content"].lower())

    def test_empty_extraction_still_selects_reasonable_next_action(self) -> None:
        state = _base_state()
        state["messages"].append({"role": "assistant", "content": "I'm okay, nothing much to add."})

        def ingest_node(_: AgentState):
            return {
                "turn": TurnState(
                    latest_text_raw="I'm okay, nothing much to add.",
                    latest_text_norm="i'm okay, nothing much to add",
                    latest_sentences=["I'm okay, nothing much to add"],
                    turn_id=1,
                ),
                "turn_index": 1,
                "has_new_persona_input": True,
                "last_processed_persona_msg_idx": 0,
            }

        def risk_node(_: AgentState):
            return {
                "risk": RiskState(risk_prob=0.0, risk_flag=False, evidence_spans=[], reason="none", last_updated_turn=1),
                "risk_flag": False,
                "risk_prob": 0.0,
            }

        def extract_node(_: AgentState):
            return {
                "latest_turn_likelihoods": [],
                "latest_turn_evidence": [],
                "evidence_log": [],
            }

        def belief_node(s: AgentState):
            return {"beliefs": s["beliefs"], "item_beliefs": s["item_beliefs"]}

        def metrics_node(_: AgentState):
            return {
                "metrics": PolicyMetricsState(
                    total_expected_bdi=0.0,
                    label_prob=0.12,
                    coverage=0.0,
                    mean_entropy=2.0,
                    top_uncertain_items=[2, 3, 15],
                    last_ig_estimates={2: 1.9, 3: 1.8, 15: 1.7},
                ),
                "global_confidence": 0.88,
                "raw_predicted_bdi_score": 0,
                "raw_predicted_label": "control",
            }

        def stop_node(_: AgentState):
            return {
                "control": ControlState(stop=False, stop_reason="continue"),
                "should_stop": False,
            }

        def question_node(_: AgentState):
            text = "Could you share one recent example of how that has felt for you?"
            return {
                "outgoing": {"detector_message": text},
                "messages": [{"role": "user", "content": text}],
            }

        def finalize_node(_: AgentState):
            return {"final": FinalState()}

        app = build_app(
            {
                "ingest_turn": ingest_node,
                "risk_sentinel": risk_node,
                "extract_likelihoods": extract_node,
                "belief_update": belief_node,
                "policy_metrics": metrics_node,
                "stop_decider": stop_node,
                "question_generator": question_node,
                "finalize_outputs": finalize_node,
            }
        )

        out = app.invoke(state)
        next_action = out["next_action"]
        self.assertIsNotNone(next_action.target_item_id)
        self.assertIn(next_action.route, {"cognitive", "somatic", "risk"})
        self.assertEqual(out["messages"][-1]["role"], "user")

    def test_stop_conditions_at_min_max_turn_boundaries_route_correctly(self) -> None:
        min_turns = "4"
        max_turns = "6"
        stop_conf = "0.8"

        def ingest_passthrough(s: AgentState):
            return {
                "turn": TurnState(
                    latest_text_raw="status",
                    latest_text_norm="status",
                    latest_sentences=["status"],
                    turn_id=int(s.get("turn_index", 0)),
                ),
                "has_new_persona_input": True,
            }

        def risk_node(_: AgentState):
            return {
                "risk": RiskState(risk_prob=0.0, risk_flag=False, evidence_spans=[], reason="none", last_updated_turn=1),
                "risk_flag": False,
                "risk_prob": 0.0,
            }

        def extract_node(_: AgentState):
            return {"latest_turn_likelihoods": [], "latest_turn_evidence": []}

        def belief_node(s: AgentState):
            return {"beliefs": s["beliefs"], "item_beliefs": s["item_beliefs"]}

        def metrics_node(_: AgentState):
            return {
                "metrics": PolicyMetricsState(
                    total_expected_bdi=8.0,
                    label_prob=0.35,
                    coverage=0.4,
                    mean_entropy=1.3,
                    top_uncertain_items=[2],
                    last_ig_estimates={2: 1.1},
                ),
                "global_confidence": 0.90,
                "raw_predicted_bdi_score": 8,
                "raw_predicted_label": "control",
            }

        def target_node(_: AgentState):
            return {
                "next_action": NextAction(target_item_id=2, route="cognitive", style="gentle_probe", rationale="test"),
                "next_node": "cognitive",
                "active_node": "cognitive",
                "route_history": [
                    RouteDecision(
                        turn=1,
                        chosen_node="cognitive",
                        policy="entropy_penalized",
                        reason="test",
                        target_items=[2],
                        expected_gain=1.1,
                    )
                ],
            }

        def question_node(_: AgentState):
            return {
                "outgoing": {"detector_message": "Can you tell me more?"},
                "messages": [{"role": "user", "content": "Can you tell me more?"}],
            }

        def finalize_node(_: AgentState):
            return {"final": FinalState()}

        with patch.dict(
            os.environ,
            {
                "MIN_TURNS": min_turns,
                "MAX_TURNS": max_turns,
                "STOP_CONFIDENCE": stop_conf,
                "MIN_EVIDENCE_FOR_CONF_STOP": "2",
            },
            clear=False,
        ):
            app = build_app(
                {
                    "ingest_turn": ingest_passthrough,
                    "risk_sentinel": risk_node,
                    "extract_likelihoods": extract_node,
                    "belief_update": belief_node,
                    "policy_metrics": metrics_node,
                    "target_selector": target_node,
                    "question_generator": question_node,
                    "finalize_outputs": finalize_node,
                }
            )

            state_continue = _base_state()
            state_continue["turn_index"] = 3
            state_continue["messages"].append({"role": "assistant", "content": "I am coping."})
            out_continue = app.invoke(state_continue)
            self.assertFalse(out_continue["should_stop"])
            self.assertEqual(out_continue["messages"][-1]["role"], "user")

            state_stop = _base_state()
            state_stop["turn_index"] = 6
            state_stop["messages"].append({"role": "assistant", "content": "I am coping."})
            out_stop = app.invoke(state_stop)
            self.assertTrue(out_stop["should_stop"])


if __name__ == "__main__":
    unittest.main()
