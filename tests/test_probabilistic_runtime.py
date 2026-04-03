from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.bayes_state_update import bayes_state_update
from agents.diagnosis_agent import diagnosis_agent
from agents.ingest_turn import ingest_turn
from agents.judgment_agent import judgment_agent
from agents.navigation_agent import navigation_agent
from agents.question_agent import question_agent
from core.state import AssertionRecord, BayesItemState, build_initial_state
from graph import build_app
from persona.atomic_memory import build_persona_memory, verify_reply_against_memory


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages):
        return _FakeResponse(self._content)


class ProbabilisticRuntimeTests(unittest.TestCase):
    def test_judgment_agent_summarizes_bound_assertions(self) -> None:
        state = build_initial_state(persona_id="judge-test")
        state["active_node"] = "cognitive"

        fake_result = {
            "latest_turn_assertions": [
                AssertionRecord(
                    turn=1,
                    node="cognitive",
                    item_id=14,
                    symptom_name="Worthlessness",
                    assertion_label="present",
                    confidence=0.82,
                    intensity=2.0,
                    anchor_quote="I feel like a burden",
                    reason="bound quote",
                    binding_status="exact",
                ),
                AssertionRecord(
                    turn=1,
                    node="cognitive",
                    item_id=7,
                    symptom_name="Self-Dislike",
                    assertion_label="uncertain",
                    confidence=0.0,
                    intensity=0.0,
                    anchor_quote="",
                    reason="no signal",
                    binding_status="unbound",
                ),
            ],
            "latest_turn_evidence": [],
            "turn_trace": {},
        }

        with patch("agents.judgment_agent._run_judgment_extraction", return_value=fake_result):
            result = judgment_agent(state)

        self.assertEqual(result["judgment"].active_cluster, "cognitive_affective")
        self.assertEqual(int(result["judgment"].bound_positive_assertion_count), 1)
        self.assertAlmostEqual(float(result["judgment"].evidence_binding_coverage), 1.0)
        self.assertIn("judgment_agent", result["turn_trace"])

    def test_judgment_agent_bootstraps_opening_cognitive_signal_without_llm(self) -> None:
        state = build_initial_state(persona_id="judge-opening-bootstrap")
        state["messages"] = [
            {"role": "user", "content": "What has been feeling most different for you lately?"},
            {
                "role": "assistant",
                "content": (
                    "I have been pretty hard on myself and stuck in my own head lately, "
                    "and the bigger change is that things have felt heavier and darker lately."
                ),
            },
        ]
        state.update(ingest_turn(state))

        with patch("agents.judgment_agent.get_extractor_llm", side_effect=RuntimeError("network unavailable")):
            result = judgment_agent(state)

        self.assertTrue(bool(result["judgment"].opening_bootstrap_applied))
        self.assertEqual(str(result["judgment"].opening_bootstrap_cluster), "cognitive_affective")
        self.assertEqual(int(result["judgment"].opening_bootstrap_item_ids[0]), 8)
        self.assertIn(int(result["judgment"].opening_bootstrap_item_ids[1]), {1, 2})
        self.assertIn(1, list(result["judgment"].allowed_item_ids))
        self.assertIn(4, list(result["judgment"].allowed_item_ids))
        self.assertIn(12, list(result["judgment"].allowed_item_ids))
        self.assertEqual(str(result["opening_signal_cluster"]), "cognitive_affective")
        self.assertEqual(int(result["opening_signal_item_ids"][0]), 8)
        self.assertIn(int(result["opening_signal_item_ids"][1]), {1, 2})
        self.assertGreaterEqual(int(result["judgment"].bound_positive_assertion_count), 2)

    def test_bayes_state_update_derives_probabilistic_and_compatibility_beliefs(self) -> None:
        state = build_initial_state(persona_id="bayes-test")
        state["turn_index"] = 3
        state["latest_turn_assertions"] = [
            AssertionRecord(
                turn=3,
                node="cognitive",
                item_id=14,
                symptom_name="Worthlessness",
                assertion_label="present",
                confidence=0.84,
                intensity=2.1,
                anchor_quote="I feel like a burden",
                reason="bound worthlessness quote",
                binding_status="exact",
            )
        ]
        state["evidence_log"] = [
            {
                "turn": 3,
                "item_id": 14,
                "evidence_text": "I feel like a burden",
                "assertion_label": "present",
            }
        ]

        result = bayes_state_update(state)

        self.assertGreater(float(result["bayes_items"][14].presence_prob), 0.28)
        self.assertGreater(float(result["bayes_nodes"]["negative_self_schema"].probability), 0.14)
        self.assertEqual(int(result["item_beliefs"][14].support_count), 1)
        self.assertFalse(bool(result["risk_flag"]))

    def test_diagnosis_agent_scores_from_bayes_state(self) -> None:
        state = build_initial_state(persona_id="diagnosis-test")
        state["bayes_nodes"]["negative_self_schema"].probability = 0.71
        state["bayes_items"][14] = BayesItemState(
            item_id=14,
            presence_prob=0.78,
            score_posterior=[0.08, 0.16, 0.52, 0.24],
            expected_score=1.92,
            uncertainty=0.48,
            audit_trail=[{"anchor_quote": "I feel like a burden"}],
        )
        state["assertion_log"] = [
            AssertionRecord(
                turn=2,
                node="cognitive",
                item_id=14,
                symptom_name="Worthlessness",
                assertion_label="present",
                confidence=0.80,
                intensity=2.0,
                anchor_quote="I feel like a burden",
                reason="bound quote",
                binding_status="exact",
            )
        ]

        result = diagnosis_agent(state)

        self.assertGreaterEqual(int(result["final_item_scores"][14]), 2)
        self.assertIn(14, list(result["predicted_key_item_ids"]))
        self.assertIn("diagnosis_agent", result["turn_trace"])

    def test_diagnosis_agent_allows_posterior_supported_nonrisk_score_without_direct_quote(self) -> None:
        state = build_initial_state(persona_id="diagnosis-posterior-supported")
        state["bayes_nodes"]["negative_self_schema"].probability = 0.64
        state["bayes_items"][14] = BayesItemState(
            item_id=14,
            presence_prob=0.69,
            score_posterior=[0.18, 0.50, 0.22, 0.10],
            expected_score=1.18,
            uncertainty=0.50,
            audit_trail=[],
        )

        result = diagnosis_agent(state)

        self.assertEqual(int(result["final_item_scores"][14]), 1)
        self.assertIn("posterior-supported, corroborated cluster state", result["diagnosis"].rationale_by_item["14"])

    def test_diagnosis_agent_requires_stronger_cluster_support_for_score_two_without_direct_quote(self) -> None:
        state = build_initial_state(persona_id="diagnosis-score-two")
        state["bayes_nodes"]["negative_self_schema"].probability = 0.78
        state["bayes_items"][7] = BayesItemState(
            item_id=7,
            presence_prob=0.83,
            score_posterior=[0.10, 0.18, 0.48, 0.24],
            expected_score=1.94,
            uncertainty=0.44,
            audit_trail=[],
        )

        result = diagnosis_agent(state)

        self.assertEqual(int(result["final_item_scores"][7]), 2)

    def test_diagnosis_agent_keeps_item_nine_direct_evidence_only(self) -> None:
        state = build_initial_state(persona_id="diagnosis-item-nine-strict")
        state["bayes_nodes"]["risk"].probability = 0.91
        state["bayes_items"][9] = BayesItemState(
            item_id=9,
            presence_prob=0.88,
            score_posterior=[0.08, 0.16, 0.42, 0.34],
            expected_score=2.02,
            uncertainty=0.42,
            audit_trail=[],
        )

        result = diagnosis_agent(state)

        self.assertEqual(int(result["final_item_scores"][9]), 0)

    def test_navigation_agent_prefers_same_cluster_threading(self) -> None:
        state = build_initial_state(persona_id="nav-test")
        state["turn_index"] = 7
        state["conversation_thread"].active = True
        state["conversation_thread"].route = "cognitive"
        state["conversation_thread"].module_id = 3
        state["conversation_thread"].source_item_id = 14
        state["conversation_thread"].question_count = 1
        state["bayes_items"][14] = BayesItemState(
            item_id=14,
            presence_prob=0.72,
            score_posterior=[0.10, 0.20, 0.44, 0.26],
            expected_score=1.86,
            uncertainty=0.56,
            audit_trail=[],
        )

        result = navigation_agent(state)

        self.assertEqual(str(result["question_plan"].active_cluster), "cognitive_affective")
        self.assertEqual(int(result["next_action"].target_item_id), 3)
        self.assertEqual(str(result["next_action"].question_kind), "same_module_followup")

    def test_navigation_agent_breaks_near_ties_toward_recent_cognitive_signal(self) -> None:
        state = build_initial_state(persona_id="nav-tie-break")
        state["turn_index"] = 6
        state["latest_turn_assertions"] = [
            AssertionRecord(
                turn=6,
                node="cognitive",
                item_id=14,
                symptom_name="Worthlessness",
                assertion_label="present",
                confidence=0.84,
                intensity=2.0,
                anchor_quote="I feel like a burden",
                reason="recent quote",
                binding_status="exact",
            )
        ]
        state["judgment"].bound_positive_assertion_count = 1
        state["bayes_nodes"]["cognitive_affective"].probability = 0.42
        state["bayes_nodes"]["somatic_vegetative"].probability = 0.41

        result = navigation_agent(state)

        self.assertEqual(str(result["question_plan"].active_cluster), "cognitive_affective")
        self.assertEqual(
            str(result["turn_trace"]["navigation_agent"]["cluster_reselection_reason"]),
            "recent_bound_assertion_preference",
        )

    def test_navigation_agent_keeps_opening_cognitive_anchor_on_near_tie(self) -> None:
        state = build_initial_state(persona_id="nav-opening-cognitive-anchor")
        state["turn_index"] = 1
        state["opening_signal_cluster"] = "cognitive_affective"
        state["opening_signal_item_ids"] = [8, 2]
        state["bayes_nodes"]["cognitive_affective"].probability = 0.40
        state["bayes_nodes"]["somatic_vegetative"].probability = 0.44
        state["judgment"].bound_positive_assertion_count = 1

        result = navigation_agent(state)

        self.assertEqual(str(result["question_plan"].active_cluster), "cognitive_affective")
        self.assertEqual(
            str(result["turn_trace"]["navigation_agent"]["cluster_reselection_reason"]),
            "opening_cognitive_anchor",
        )

    def test_navigation_agent_rebalances_after_weak_early_somatic_lockin(self) -> None:
        state = build_initial_state(persona_id="nav-opening-lockin-rebalance")
        state["turn_index"] = 2
        state["opening_signal_cluster"] = "cognitive_affective"
        state["opening_signal_item_ids"] = [8, 2]
        state["opening_followup_cluster"] = "somatic_vegetative"
        state["conversation_thread"].active = True
        state["conversation_thread"].route = "somatic"
        state["conversation_thread"].module_id = 6
        state["conversation_thread"].source_item_id = 20
        state["conversation_thread"].question_count = 1
        state["judgment"].bound_positive_assertion_count = 1
        state["judgment"].emitted_evidence_count = 1
        state["bayes_nodes"]["cognitive_affective"].probability = 0.36
        state["bayes_nodes"]["somatic_vegetative"].probability = 0.42

        result = navigation_agent(state)

        self.assertEqual(str(result["question_plan"].active_cluster), "cognitive_affective")
        self.assertEqual(
            str(result["turn_trace"]["navigation_agent"]["cluster_reselection_reason"]),
            "opening_cognitive_anchor",
        )
        self.assertGreaterEqual(
            int(result["runtime_counters"].get("opening_somatic_pivot_after_cognitive_signal_count", 0)),
            1,
        )

    def test_navigation_agent_exits_stale_same_item_loop(self) -> None:
        state = build_initial_state(persona_id="nav-stale-thread")
        state["turn_index"] = 8
        state["conversation_thread"].active = True
        state["conversation_thread"].route = "somatic"
        state["conversation_thread"].module_id = 6
        state["conversation_thread"].source_item_id = 20
        state["conversation_thread"].question_count = 1
        state["messages"] = [
            {"role": "user", "content": "How has your energy been?"},
            {"role": "assistant", "content": "I'm wiped out by the afternoon."},
            {"role": "user", "content": "When does that tiredness hit most?"},
            {"role": "assistant", "content": "I'm wiped out by the afternoon."},
        ]
        state["bayes_items"][20] = BayesItemState(
            item_id=20,
            presence_prob=0.70,
            score_posterior=[0.10, 0.20, 0.48, 0.22],
            expected_score=1.82,
            uncertainty=0.58,
            audit_trail=[],
        )
        state["bayes_items"][16] = BayesItemState(
            item_id=16,
            presence_prob=0.52,
            score_posterior=[0.18, 0.34, 0.33, 0.15],
            expected_score=1.45,
            uncertainty=0.64,
            audit_trail=[],
        )

        result = navigation_agent(state)

        self.assertEqual(str(result["next_action"].question_kind), "same_module_followup")
        self.assertNotEqual(int(result["next_action"].target_item_id), 20)
        self.assertGreaterEqual(int(result["runtime_counters"].get("stale_thread_count", 0)), 1)
        self.assertGreaterEqual(int(result["runtime_counters"].get("same_item_loop_exit_count", 0)), 1)

    def test_question_agent_emits_plan_aware_followup(self) -> None:
        state = build_initial_state(persona_id="question-test")
        state["turn_index"] = 4
        state["messages"] = [
            {"role": "user", "content": "How have you been talking to yourself lately?"},
            {"role": "assistant", "content": "I keep feeling like a burden at work."},
        ]
        state["next_action"].target_item_id = 14
        state["next_action"].route = "cognitive"
        state["next_action"].style = "gentle_probe"
        state["next_action"].question_kind = "same_item_followup"
        state["next_action"].timeframe_mode = "clarify"
        state["next_action"].anchor_text = "I keep feeling like a burden"
        state["question_plan"].active_cluster = "cognitive_affective"
        state["question_plan"].transition_reason = "bounded_thread_continuation"
        state["question_plan"].urgency_mode = "adaptive"

        with patch("agents.question_agent.get_llm", return_value=_FakeLLM("What goes through your mind when that burden feeling spikes")):
            result = question_agent(state)

        self.assertEqual(result["messages"][-1]["role"], "user")
        self.assertTrue(result["messages"][-1]["content"].endswith("?"))
        self.assertIn("question_agent", result["turn_trace"])
        self.assertEqual(str(result["turn_trace"]["question_agent"]["question_kind"]), "same_item_followup")

    def test_question_agent_updates_thread_runtime_counters(self) -> None:
        state = build_initial_state(persona_id="question-counters")
        state["turn_index"] = 5
        state["next_action"].target_item_id = 16
        state["next_action"].route = "somatic"
        state["next_action"].style = "clarify_frequency"
        state["next_action"].question_kind = "same_item_followup"
        state["next_action"].timeframe_mode = "clarify"
        state["next_action"].thread_turn_index = 2
        state["question_plan"].active_cluster = "somatic_vegetative"

        with patch("agents.question_agent.get_llm", return_value=_FakeLLM("When that happens, how often are you awake in the night")):
            result = question_agent(state)

        self.assertEqual(int(result["runtime_counters"].get("threaded_followup_count", 0)), 1)
        self.assertEqual(int(result["runtime_counters"].get("thread_question_total", 0)), 1)

    def test_diagnosis_agent_keeps_structural_only_item_at_zero(self) -> None:
        state = build_initial_state(persona_id="diagnosis-structural-only")
        state["bayes_items"][18] = BayesItemState(
            item_id=18,
            presence_prob=0.76,
            score_posterior=[0.18, 0.36, 0.30, 0.16],
            expected_score=1.52,
            uncertainty=0.64,
            audit_trail=[],
        )

        result = diagnosis_agent(state)

        self.assertEqual(int(result["final_item_scores"][18]), 0)
        self.assertLess(float(result["diagnosis"].confidence), 0.5)

    def test_graph_stops_before_question_generation(self) -> None:
        order = []

        def _node(name: str, delta: dict):
            def _impl(_state):
                order.append(name)
                return dict(delta)

            return _impl

        app = build_app(
            {
                "ingest_turn": _node("ingest_turn", {}),
                "judgment_agent": _node("judgment_agent", {}),
                "bayes_state_update": _node("bayes_state_update", {}),
                "diagnosis_agent": _node("diagnosis_agent", {}),
                "stop_controller": _node(
                    "stop_controller",
                    {"should_stop": True, "control": {"stop": True, "stop_reason": "test"}},
                ),
                "navigation_agent": _node("navigation_agent", {}),
                "question_agent": _node("question_agent", {}),
            }
        )

        app.invoke(build_initial_state(persona_id="graph-stop"))
        self.assertEqual(
            order,
            ["ingest_turn", "judgment_agent", "bayes_state_update", "diagnosis_agent", "stop_controller"],
        )

    def test_graph_continues_to_question_agent_when_not_stopping(self) -> None:
        order = []

        def _node(name: str, delta: dict):
            def _impl(_state):
                order.append(name)
                return dict(delta)

            return _impl

        app = build_app(
            {
                "ingest_turn": _node("ingest_turn", {}),
                "judgment_agent": _node("judgment_agent", {}),
                "bayes_state_update": _node("bayes_state_update", {}),
                "diagnosis_agent": _node("diagnosis_agent", {}),
                "stop_controller": _node(
                    "stop_controller",
                    {"should_stop": False, "control": {"stop": False, "stop_reason": "continue"}},
                ),
                "navigation_agent": _node("navigation_agent", {}),
                "question_agent": _node("question_agent", {"messages": [{"role": "user", "content": "test?"}]}),
            }
        )

        result = app.invoke(build_initial_state(persona_id="graph-continue"))
        self.assertEqual(
            order,
            [
                "ingest_turn",
                "judgment_agent",
                "bayes_state_update",
                "diagnosis_agent",
                "stop_controller",
                "navigation_agent",
                "question_agent",
            ],
        )
        self.assertEqual(result["messages"][-1]["content"], "test?")

    def test_atomic_memory_flags_unexpected_positive_for_zero_item(self) -> None:
        memory = build_persona_memory(
            bdi_scores={item_id: 0 for item_id in range(1, 22)},
            context_tag="routine_stable",
            style_tag="open_but_flat",
        )
        valid, reason = verify_reply_against_memory(
            memory,
            target_item_id=14,
            target_score=0,
            reply_text="I feel like a burden and I do not matter.",
        )

        self.assertFalse(valid)
        self.assertTrue(reason.startswith("unexpected_positive_lexeme"))


if __name__ == "__main__":
    unittest.main()
