from __future__ import annotations

import unittest

from agents.stop_decider import stop_decider
from core.state import ItemBelief, PolicyMetricsState, RouteDecision, build_initial_state


def _set_belief(
    state,
    item_id: int,
    *,
    support_count: int = 0,
    expected_score: float = 0.0,
    entropy: float = 2.0,
    last_update_turn: int = 0,
) -> None:
    state["item_beliefs"][item_id] = ItemBelief(
        item_id=item_id,
        support_count=support_count,
        expected_score=expected_score,
        entropy=entropy,
        last_update_turn=last_update_turn,
    )


def _set_all_beliefs(
    state,
    *,
    support_count: int = 1,
    expected_score: float = 0.0,
    entropy: float = 0.4,
    last_update_turn: int = 18,
) -> None:
    for item_id in range(1, 22):
        _set_belief(
            state,
            item_id,
            support_count=support_count,
            expected_score=expected_score,
            entropy=entropy,
            last_update_turn=last_update_turn,
        )


def _route_for_item(item_id: int) -> str:
    if item_id == 9:
        return "risk"
    if item_id in {11, 15, 16, 18, 20, 21}:
        return "somatic"
    return "cognitive"


def _set_attempted_items(state, item_ids: list[int]) -> None:
    history = []
    for turn, item_id in enumerate(item_ids, start=1):
        history.append(
            RouteDecision(
                turn=turn,
                chosen_node=_route_for_item(item_id),
                policy="evidence_weighted_global",
                reason="test coverage",
                target_items=[item_id],
                expected_gain=1.0,
            )
        )
    state["route_history"] = history


def _set_metrics(state, *, total_expected_bdi: float = 0.0) -> None:
    state["metrics"] = PolicyMetricsState(total_expected_bdi=float(total_expected_bdi))


class StopDeciderTests(unittest.TestCase):
    def test_evidence_saturation_met_when_confident_and_stalled(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.80
        state["empty_evidence_streak"] = 3
        _set_attempted_items(state, list(range(1, 13)))
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=18)
        for item_id in [13, 14, 15, 16, 17]:
            _set_belief(state, item_id, support_count=0, entropy=1.4, last_update_turn=0)

        result = stop_decider(state)

        self.assertTrue(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "evidence_saturation_met")
        self.assertTrue(bool(result["turn_trace"]["stop_decider"]["evidence_saturation_eligible"]))

    def test_unattempted_high_entropy_items_do_not_block_evidence_saturation(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.80
        state["empty_evidence_streak"] = 3
        _set_attempted_items(state, list(range(1, 13)))
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=18)
        for item_id in [13, 14, 15, 16, 17, 18, 19]:
            _set_belief(state, item_id, support_count=0, entropy=1.6, last_update_turn=0)

        result = stop_decider(state)

        self.assertTrue(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "evidence_saturation_met")
        self.assertEqual(int(result["turn_trace"]["stop_decider"]["high_entropy_unresolved_count"]), 0)

    def test_recent_updates_block_evidence_saturation(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.80
        state["empty_evidence_streak"] = 0
        _set_attempted_items(state, list(range(1, 13)))
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=18)
        for item_id in [13, 14, 15, 16, 17]:
            _set_belief(state, item_id, support_count=0, entropy=1.4, last_update_turn=0)
        _set_belief(state, 5, support_count=1, entropy=0.4, last_update_turn=22)

        result = stop_decider(state)

        self.assertFalse(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "continue")
        self.assertEqual(int(result["turn_trace"]["stop_decider"]["recent_updated_item_count"]), 1)

    def test_high_unresolved_entropy_blocks_evidence_saturation(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.80
        state["empty_evidence_streak"] = 3
        _set_attempted_items(state, list(range(1, 13)))
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=18)
        for item_id in [1, 2, 3, 4, 5, 6]:
            _set_belief(state, item_id, support_count=0, entropy=1.4, last_update_turn=0)

        result = stop_decider(state)

        self.assertFalse(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "continue")
        self.assertEqual(int(result["turn_trace"]["stop_decider"]["high_entropy_unresolved_count"]), 6)

    def test_structural_fallback_still_applies_when_saturation_does_not(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.60
        state["empty_evidence_streak"] = 0
        _set_attempted_items(state, list(range(1, 16)))
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=21)

        result = stop_decider(state)

        self.assertTrue(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "structural_coverage_met")
        self.assertFalse(bool(result["turn_trace"]["stop_decider"]["evidence_saturation_eligible"]))

    def test_pending_risk_probe_blocks_evidence_saturation(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 22
        state["has_new_persona_input"] = True
        state["global_confidence"] = 0.80
        state["empty_evidence_streak"] = 3
        state["risk_flag"] = True
        _set_metrics(state, total_expected_bdi=22.0)
        _set_attempted_items(state, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
        _set_all_beliefs(state, support_count=1, entropy=0.4, last_update_turn=18)
        _set_belief(state, 9, support_count=0, entropy=1.5, last_update_turn=0)
        for item_id in [13, 14, 15, 16]:
            _set_belief(state, item_id, support_count=0, entropy=1.4, last_update_turn=0)

        result = stop_decider(state)

        self.assertFalse(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "continue")
        self.assertTrue(bool(result["turn_trace"]["stop_decider"]["risk_probe_pending"]))

    def test_max_turns_reached_is_unchanged(self) -> None:
        state = build_initial_state(persona_id="stop-test")
        state["turn_index"] = 40
        state["has_new_persona_input"] = True

        result = stop_decider(state)

        self.assertTrue(bool(result["should_stop"]))
        self.assertEqual(str(result["control"].stop_reason), "max_turns_reached")


if __name__ == "__main__":
    unittest.main()
