from __future__ import annotations

import unittest

from agents.target_selector import target_selector
from core.state import ItemBelief, PolicyMetricsState, RouteDecision, build_initial_state


def _set_belief(
    state,
    item_id: int,
    *,
    support_count: int = 0,
    expected_score: float = 0.0,
    entropy: float = 2.0,
) -> None:
    state["item_beliefs"][item_id] = ItemBelief(
        item_id=item_id,
        support_count=support_count,
        expected_score=expected_score,
        entropy=entropy,
    )


def _set_metrics(state, *, ig_estimates: dict[int, float], total_expected_bdi: float = 0.0) -> None:
    state["metrics"] = PolicyMetricsState(
        total_expected_bdi=float(total_expected_bdi),
        top_uncertain_items=list(range(1, 22)),
        last_ig_estimates=dict(ig_estimates),
    )


def _deprioritize_other_items(state, *, keep_item_ids: set[int]) -> None:
    for item_id in range(1, 22):
        if item_id in keep_item_ids:
            continue
        _set_belief(state, item_id, support_count=1, expected_score=0.0, entropy=0.2)


def _set_productive_trace(
    state,
    *,
    route: str,
    target_item_id: int,
    target_module_id: int,
    updated_item_ids: list[int],
    support_increments_count: int = 1,
) -> None:
    state["has_new_persona_input"] = True
    state["active_node"] = route
    state["turn_trace"] = {
        "update_beliefs": {
            "support_increments_count": support_increments_count,
            "updated_item_ids": list(updated_item_ids),
        },
        "specialist": {
            "node": route,
            "target_item_id": target_item_id,
            "target_module_id": target_module_id,
        },
    }


class TargetSelectorTests(unittest.TestCase):
    def test_productive_turn_prefers_same_module_followup_over_same_item(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 5
        _deprioritize_other_items(state, keep_item_ids={5, 14})
        _set_belief(state, 14, support_count=1, expected_score=1.4, entropy=1.6)
        _set_belief(state, 5, support_count=0, expected_score=0.0, entropy=2.0)
        _set_metrics(state, ig_estimates={14: 1.4, 5: 2.0})
        _set_productive_trace(
            state,
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            updated_item_ids=[14],
            support_increments_count=1,
        )

        result = target_selector(state)

        self.assertEqual(int(result["next_action"].target_item_id), 5)
        self.assertEqual(str(result["next_action"].route), "cognitive")
        self.assertEqual(str(result["route_history"][0].policy), "evidence_followup_same_module")
        self.assertEqual(int(result["turn_trace"]["target_selector"]["selected_module_id"]), 3)
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["same_module_followup_eligible"]))
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["same_item_followup_eligible"]))

    def test_same_item_followup_still_fires_when_no_eligible_sibling_exists(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 6
        _deprioritize_other_items(state, keep_item_ids={14})
        _set_belief(state, 14, support_count=1, expected_score=1.4, entropy=1.6)
        _set_metrics(state, ig_estimates={14: 1.4})
        _set_productive_trace(
            state,
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            updated_item_ids=[14],
            support_increments_count=1,
        )

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_followup_same_item")
        self.assertEqual(int(result["next_action"].target_item_id), 14)
        self.assertFalse(bool(result["turn_trace"]["target_selector"]["same_module_followup_eligible"]))
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["same_item_followup_eligible"]))

    def test_same_module_followup_chooses_highest_entropy_unresolved_sibling(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 7
        _deprioritize_other_items(state, keep_item_ids={5, 7, 14})
        _set_belief(state, 7, support_count=2, expected_score=1.5, entropy=0.9)
        _set_belief(state, 5, support_count=0, expected_score=0.0, entropy=1.8)
        _set_belief(state, 14, support_count=0, expected_score=0.0, entropy=1.4)
        _set_metrics(state, ig_estimates={7: 1.0, 5: 1.8, 14: 1.4})
        _set_productive_trace(
            state,
            route="cognitive",
            target_item_id=7,
            target_module_id=3,
            updated_item_ids=[7],
            support_increments_count=1,
        )

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_followup_same_module")
        self.assertEqual(int(result["next_action"].target_item_id), 5)
        self.assertEqual(int(result["turn_trace"]["target_selector"]["followup_source_item_id"]), 7)
        self.assertEqual(int(result["turn_trace"]["target_selector"]["followup_source_module_id"]), 3)

    def test_risk_route_never_uses_evidence_followup_policy(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 8
        _deprioritize_other_items(state, keep_item_ids={9})
        _set_belief(state, 9, support_count=1, expected_score=1.3, entropy=1.7)
        _set_metrics(state, ig_estimates={9: 2.0})
        _set_productive_trace(
            state,
            route="risk",
            target_item_id=9,
            target_module_id=9,
            updated_item_ids=[9],
            support_increments_count=1,
        )

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_weighted_global")
        self.assertEqual(int(result["next_action"].target_item_id), 9)
        self.assertEqual(str(result["next_action"].route), "risk")

    def test_item_nine_is_not_selected_globally_without_recent_risk_signal(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 9
        state["has_new_persona_input"] = False
        for item_id in range(1, 22):
            _set_belief(state, item_id, support_count=0, expected_score=0.0, entropy=0.1)
        _set_belief(state, 9, support_count=0, expected_score=0.0, entropy=2.0)
        _set_belief(state, 13, support_count=0, expected_score=0.0, entropy=1.8)
        _set_metrics(state, ig_estimates={9: 3.5, 13: 1.7})

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_weighted_global")
        self.assertEqual(int(result["next_action"].target_item_id), 13)
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["risk_dampener_applied"]))

    def test_item_nine_becomes_globally_eligible_with_risk_signal(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 10
        state["risk_flag"] = True
        state["has_new_persona_input"] = False
        for item_id in range(1, 22):
            _set_belief(state, item_id, support_count=0, expected_score=0.0, entropy=0.1)
        _set_belief(state, 9, support_count=0, expected_score=0.0, entropy=2.0)
        _set_belief(state, 13, support_count=0, expected_score=0.0, entropy=1.8)
        _set_metrics(state, ig_estimates={9: 3.5, 13: 1.7})

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_weighted_global")
        self.assertEqual(int(result["next_action"].target_item_id), 9)
        self.assertEqual(str(result["next_action"].route), "risk")
        self.assertFalse(bool(result["turn_trace"]["target_selector"]["risk_dampener_applied"]))
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["risk_reentry_eligible"]))
        self.assertEqual(str(result["turn_trace"]["target_selector"]["risk_reentry_reason"]), "risk_flag")

    def test_item_nine_becomes_globally_eligible_with_high_expected_bdi(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 11
        state["has_new_persona_input"] = False
        for item_id in range(1, 22):
            _set_belief(state, item_id, support_count=0, expected_score=0.0, entropy=0.1)
        _set_belief(state, 9, support_count=0, expected_score=0.0, entropy=2.0)
        _set_belief(state, 13, support_count=0, expected_score=0.0, entropy=1.8)
        _set_metrics(state, ig_estimates={9: 3.5, 13: 1.7}, total_expected_bdi=22.0)

        result = target_selector(state)

        self.assertEqual(int(result["next_action"].target_item_id), 9)
        self.assertEqual(str(result["next_action"].route), "risk")
        self.assertEqual(str(result["turn_trace"]["target_selector"]["risk_reentry_reason"]), "high_expected_bdi")

    def test_recent_risk_attempt_suppresses_repeated_risk_reentry(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 12
        state["risk_flag"] = True
        state["has_new_persona_input"] = False
        for item_id in range(1, 22):
            _set_belief(state, item_id, support_count=0, expected_score=0.0, entropy=0.1)
        _set_belief(state, 9, support_count=0, expected_score=0.0, entropy=2.0)
        _set_belief(state, 13, support_count=0, expected_score=0.0, entropy=1.8)
        _set_metrics(state, ig_estimates={9: 3.5, 13: 1.7})
        state["route_history"] = [
            RouteDecision(
                turn=10,
                chosen_node="risk",
                policy="evidence_weighted_global",
                reason="recent risk probe",
                target_items=[9],
                expected_gain=1.0,
            )
        ]

        result = target_selector(state)

        self.assertEqual(int(result["next_action"].target_item_id), 13)
        self.assertFalse(bool(result["turn_trace"]["target_selector"]["risk_reentry_eligible"]))
        self.assertTrue(bool(result["turn_trace"]["target_selector"]["risk_recent_attempted"]))

    def test_module_saturation_penalty_pushes_global_selection_away_from_overexplored_module(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 13
        state["has_new_persona_input"] = False
        for item_id in range(1, 22):
            _set_belief(state, item_id, support_count=1, expected_score=0.0, entropy=0.2)
        _set_belief(state, 14, support_count=0, expected_score=0.0, entropy=1.8)
        _set_belief(state, 16, support_count=0, expected_score=0.0, entropy=1.7)
        _set_belief(state, 18, support_count=1, expected_score=0.0, entropy=0.2)
        _set_metrics(state, ig_estimates={14: 1.8, 16: 1.7})

        result = target_selector(state)

        self.assertEqual(str(result["route_history"][0].policy), "evidence_weighted_global")
        self.assertEqual(int(result["next_action"].target_item_id), 16)
        self.assertGreater(float(result["turn_trace"]["target_selector"]["ranking_top_candidates"][1]["score_components"]["module_saturation_penalty"]), 0.0)

    def test_global_fallback_is_module_aware_for_item_twenty_one(self) -> None:
        state = build_initial_state(persona_id="selector-test")
        state["turn_index"] = 14
        state["has_new_persona_input"] = False
        _deprioritize_other_items(state, keep_item_ids={21})
        _set_belief(state, 21, support_count=0, expected_score=0.0, entropy=1.9)
        _set_metrics(state, ig_estimates={21: 1.9})

        result = target_selector(state)

        self.assertEqual(int(result["next_action"].target_item_id), 21)
        self.assertEqual(str(result["next_action"].route), "somatic")
        self.assertEqual(int(result["turn_trace"]["target_selector"]["selected_module_id"]), 7)


if __name__ == "__main__":
    unittest.main()
