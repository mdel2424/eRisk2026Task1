from __future__ import annotations

import unittest

from agents.belief_update import update_beliefs
from core.state import LikelihoodEvidence, build_initial_state


def _belief_state(persona_id: str = "belief-test"):
    state = build_initial_state(persona_id=persona_id)
    state["turn_index"] = 1
    return state


def _positive_row(
    *,
    item_id: int,
    evidence_type: str,
    extract_confidence: float,
    extract_intensity: float,
    support_increment_blocked: bool = False,
) -> LikelihoodEvidence:
    return LikelihoodEvidence(
        item_id=item_id,
        likelihood=[0.55, 0.75, 1.20, 1.60],
        spans=["test evidence"],
        extract_confidence=extract_confidence,
        extract_intensity=extract_intensity,
        evidence_type=evidence_type,
        symptom_name="test",
        direction="increase",
        evidence_id=f"{item_id}-{evidence_type}-{extract_confidence}-{extract_intensity}-{support_increment_blocked}",
        method_weight_hint=0.0,
        precision_gate_action="soft_clamped" if support_increment_blocked else "kept",
        support_increment_blocked=support_increment_blocked,
    )


def _per_item_stat(trace: dict, item_id: int) -> dict:
    for row in trace.get("per_item_stats", []):
        if int(row.get("item_id", 0) or 0) == item_id:
            return row
    raise AssertionError(f"missing per_item_stats entry for item {item_id}")


class BeliefUpdateGuardedSupportTests(unittest.TestCase):
    def test_weak_lexical_guarded_row_updates_posterior_but_not_support(self) -> None:
        state = _belief_state("weak-lexical-guarded")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=9, evidence_type="lexical_fallback", extract_confidence=0.4, extract_intensity=1.0)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][9]
        trace = result["turn_trace"]["belief_update"]

        self.assertGreater(float(belief.expected_score), 1.5)
        self.assertEqual(int(belief.support_count), 0)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 1)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 1)

    def test_weak_salvage_guarded_row_updates_posterior_but_not_support(self) -> None:
        state = _belief_state("weak-salvage-guarded")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=4, evidence_type="llm_salvage", extract_confidence=0.4, extract_intensity=1.0)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][4]
        trace = result["turn_trace"]["belief_update"]

        self.assertGreater(float(belief.expected_score), 1.5)
        self.assertEqual(int(belief.support_count), 0)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 1)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 1)

    def test_soft_clamped_guarded_row_never_increments_support(self) -> None:
        state = _belief_state("soft-clamped-guarded")
        state["latest_turn_likelihoods"] = [
            _positive_row(
                item_id=19,
                evidence_type="llm_salvage",
                extract_confidence=0.8,
                extract_intensity=2.0,
                support_increment_blocked=True,
            )
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][19]
        trace = result["turn_trace"]["belief_update"]

        self.assertGreater(float(belief.expected_score), 1.5)
        self.assertEqual(int(belief.support_count), 0)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 1)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 1)

    def test_strong_llm_extractor_guarded_row_still_increments_support(self) -> None:
        state = _belief_state("strong-llm-guarded")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=9, evidence_type="llm_extractor", extract_confidence=0.8, extract_intensity=2.0)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][9]
        trace = result["turn_trace"]["belief_update"]

        self.assertEqual(int(belief.support_count), 1)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 0)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 0)

    def test_non_guarded_somatic_fallback_keeps_current_support_behavior(self) -> None:
        state = _belief_state("non-guarded-somatic")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=16, evidence_type="lexical_fallback", extract_confidence=0.4, extract_intensity=1.0)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][16]
        trace = result["turn_trace"]["belief_update"]

        self.assertEqual(int(belief.support_count), 1)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 0)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 0)

    def test_llm_opportunistic_non_guarded_row_uses_medium_weight_and_increments_support(self) -> None:
        state = _belief_state("opportunistic-non-guarded")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=16, evidence_type="llm_opportunistic", extract_confidence=0.6, extract_intensity=1.6)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][16]
        trace = result["turn_trace"]["belief_update"]
        item_stat = _per_item_stat(trace, 16)

        self.assertEqual(int(belief.support_count), 1)
        self.assertEqual(int(trace["method_counts"]["llm_opportunistic"]), 1)
        self.assertAlmostEqual(float(item_stat["effective_weight_mean"]), 0.75, places=6)
        self.assertEqual(int(item_stat["support_increments"]), 1)

    def test_llm_opportunistic_guarded_row_requires_stricter_threshold(self) -> None:
        state = _belief_state("opportunistic-guarded-weak")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=9, evidence_type="llm_opportunistic", extract_confidence=0.6, extract_intensity=1.6)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][9]
        trace = result["turn_trace"]["belief_update"]

        self.assertGreater(float(belief.expected_score), 1.5)
        self.assertEqual(int(belief.support_count), 0)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 1)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 1)

    def test_llm_opportunistic_guarded_strong_row_increments_support(self) -> None:
        state = _belief_state("opportunistic-guarded-strong")
        state["latest_turn_likelihoods"] = [
            _positive_row(item_id=9, evidence_type="llm_opportunistic", extract_confidence=0.7, extract_intensity=1.8)
        ]

        result = update_beliefs(state)
        belief = result["item_beliefs"][9]
        trace = result["turn_trace"]["belief_update"]

        self.assertEqual(int(belief.support_count), 1)
        self.assertEqual(int(trace["support_rejected_by_method_count"]), 0)
        self.assertEqual(int(trace["support_rejected_guarded_item_count"]), 0)


if __name__ == "__main__":
    unittest.main()
