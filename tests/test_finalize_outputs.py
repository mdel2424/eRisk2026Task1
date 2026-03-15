from __future__ import annotations

import unittest

from agents.finalize_outputs import finalize_outputs
from core.state import (
    ControlState,
    EvidenceRecord,
    ItemBelief,
    RiskState,
    build_initial_state,
    posterior_from_expected_score,
)


def _finalizing_state(persona_id: str = "finalizer-test"):
    state = build_initial_state(persona_id=persona_id)
    state["control"] = ControlState(stop=True, stop_reason="test")
    state["raw_predicted_bdi_score"] = 0
    state["raw_predicted_label"] = "control"
    state["risk_flag"] = False
    state["risk"] = RiskState(risk_prob=0.0, risk_flag=False, reason="no lexical risk cue")
    state["turn_index"] = 1
    return state


def _set_item_belief(
    state,
    *,
    item_id: int,
    expected_score: float,
    support_count: int,
    entropy: float,
) -> None:
    state["item_beliefs"][item_id] = ItemBelief(
        item_id=item_id,
        posterior=posterior_from_expected_score(expected_score),
        entropy=entropy,
        expected_score=expected_score,
        support_count=support_count,
        last_update_turn=1,
    )


def _add_evidence(
    state,
    *,
    item_id: int,
    turn: int,
    method: str = "llm_extractor",
    blocked: bool = False,
    confidence: float = 0.7,
    intensity: float = 1.5,
) -> None:
    state["evidence_log"].append(
        EvidenceRecord(
            turn=turn,
            node="risk" if item_id == 9 else ("somatic" if item_id in {11, 15, 16, 18, 20} else "cognitive"),
            item_id=item_id,
            symptom_name=f"item-{item_id}",
            direction="increase",
            intensity=float(intensity),
            confidence=float(confidence),
            evidence_text=f"evidence for item {item_id}",
            reason="test fixture",
            method=method,
            support_increment_blocked=blocked,
        )
    )


def _add_persona_reply(state, message: str) -> None:
    state["messages"].append({"role": "assistant", "content": message})


def _add_detector_question(state, message: str) -> None:
    state["messages"].append({"role": "user", "content": message})


class FinalizeOutputsTests(unittest.TestCase):
    def test_low_signal_sparse_control_profile_activates_guardrail(self) -> None:
        state = _finalizing_state("low-signal-active")
        _set_item_belief(state, item_id=16, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertTrue(bool(module_imputation["low_signal_guardrail_active"]))
        self.assertEqual(int(module_imputation["observed_positive_breadth"]), 1)
        self.assertEqual(int(module_imputation["imputed_point_budget"]), 0)

    def test_single_repeated_weak_item_does_not_disable_guardrail(self) -> None:
        state = _finalizing_state("repeated-weak-item")
        _set_item_belief(state, item_id=4, expected_score=2.1, support_count=2, entropy=0.80)
        _add_evidence(state, item_id=4, turn=1)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertTrue(bool(module_imputation["low_signal_guardrail_active"]))
        self.assertTrue(bool(module_imputation["support_concentration_dominant"]))
        self.assertEqual(module_imputation["corroborated_item_ids"], [])
        self.assertEqual(int(module_imputation["dominant_support_item_id"]), 4)

    def test_broad_mild_support_without_anchors_keeps_guardrail_active(self) -> None:
        state = _finalizing_state("anchor-gated-guardrail")
        _set_item_belief(state, item_id=4, expected_score=2.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=10, expected_score=1.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=5, expected_score=2.1, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.20)
        _add_evidence(state, item_id=4, turn=1)
        _add_evidence(state, item_id=10, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_evidence(state, item_id=14, turn=2)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertTrue(bool(module_imputation["support_geometry_candidate_bypass"]))
        self.assertTrue(bool(module_imputation["anchor_gated_guardrail_blocked"]))
        self.assertTrue(bool(module_imputation["low_signal_guardrail_active"]))
        self.assertFalse(bool(module_imputation["severe_recovery_mode_active"]))
        self.assertGreaterEqual(int(module_imputation["corroborated_core_hits"]), 2)
        self.assertGreaterEqual(int(module_imputation["corroborated_affective_cognitive_module_breadth"]), 2)
        self.assertEqual(module_imputation["guardrail_bypass_source"], "none")
        self.assertIsNotNone(module_imputation["imputed_point_budget"])

    def test_few_mild_supported_items_still_keep_guardrail_active(self) -> None:
        state = _finalizing_state("few-mild-items")
        _set_item_belief(state, item_id=4, expected_score=1.2, support_count=1, entropy=1.00)
        _set_item_belief(state, item_id=15, expected_score=1.2, support_count=1, entropy=1.00)
        _set_item_belief(state, item_id=16, expected_score=1.2, support_count=1, entropy=1.00)
        _set_item_belief(state, item_id=20, expected_score=1.2, support_count=1, entropy=1.00)

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["low_signal_guardrail_active"]))

    def test_one_positive_profile_gets_zero_imputed_points(self) -> None:
        state = _finalizing_state("imputed-budget-zero")
        _set_item_belief(state, item_id=1, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertEqual(int(module_imputation["imputed_point_budget"]), 0)
        self.assertEqual(int(module_imputation["imputed_points_after_guardrail"]), 0)
        self.assertEqual(sum(1 for item_id, detail in module_imputation["item_details"].items() if detail["source"] == "imputed" and int(result["final_item_scores"][int(item_id)]) > 0), 0)

    def test_two_or_three_positive_profile_is_capped_to_one_imputed_point(self) -> None:
        state = _finalizing_state("imputed-budget-one")
        _set_item_belief(state, item_id=1, expected_score=1.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=4, expected_score=1.6, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertEqual(int(module_imputation["imputed_point_budget"]), 1)
        self.assertLessEqual(int(module_imputation["imputed_points_after_guardrail"]), 1)
        self.assertGreaterEqual(int(module_imputation["imputed_points_before_guardrail"]), int(module_imputation["imputed_points_after_guardrail"]))

    def test_four_positive_low_signal_profile_is_capped_to_two_imputed_points(self) -> None:
        state = _finalizing_state("imputed-budget-two")
        _set_item_belief(state, item_id=1, expected_score=1.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=4, expected_score=1.6, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=10, expected_score=1.1, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=11, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertTrue(bool(module_imputation["low_signal_guardrail_active"]))
        self.assertEqual(int(module_imputation["observed_positive_breadth"]), 4)
        self.assertEqual(int(module_imputation["observed_core_hits"]), 1)
        self.assertEqual(int(module_imputation["imputed_point_budget"]), 1)
        self.assertLessEqual(int(module_imputation["imputed_points_after_guardrail"]), 1)

    def test_low_signal_somatic_only_evidence_cannot_create_somatic_spillover(self) -> None:
        state = _finalizing_state("somatic-only-block")
        _set_item_belief(state, item_id=11, expected_score=1.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=16, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertEqual(int(result["final_item_scores"][15]), 0)
        self.assertEqual(int(result["final_item_scores"][20]), 0)
        self.assertEqual(int(result["final_item_scores"][18]), 0)
        self.assertIn(15, module_imputation["somatic_corroboration_blocked_item_ids"])
        self.assertIn(18, module_imputation["somatic_corroboration_blocked_item_ids"])

    def test_somatic_spillover_is_allowed_with_affective_cognitive_corroboration(self) -> None:
        state = _finalizing_state("somatic-corroborated")
        _set_item_belief(state, item_id=1, expected_score=1.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=11, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)

        self.assertGreaterEqual(int(result["final_item_scores"][15]) + int(result["final_item_scores"][20]), 1)
        self.assertTrue(bool(result["module_imputation"]["affective_cognitive_corroboration"]))

    def test_low_signal_observed_nonrisk_two_without_corroboration_is_capped_to_one(self) -> None:
        state = _finalizing_state("observed-two-capped")
        _set_item_belief(state, item_id=6, expected_score=2.2, support_count=2, entropy=0.90)
        _add_evidence(state, item_id=6, turn=1)

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["6"]

        self.assertEqual(int(result["final_item_scores"][6]), 1)
        self.assertFalse(bool(detail["is_corroborated_item"]))
        self.assertTrue(bool(detail["low_signal_observed_cap_applied"]))
        self.assertIn(6, result["module_imputation"]["low_signal_observed_cap_item_ids"])

    def test_weak_singleton_nonrisk_positive_is_trimmed_to_zero(self) -> None:
        state = _finalizing_state("singleton-zero")
        _set_item_belief(state, item_id=15, expected_score=1.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["15"]

        self.assertEqual(int(result["final_item_scores"][15]), 0)
        self.assertTrue(bool(detail["low_signal_singleton_trim_applied"]))
        self.assertIn(15, result["module_imputation"]["low_signal_singleton_trimmed_item_ids"])

    def test_borderline_singleton_nonrisk_positive_is_trimmed_to_one(self) -> None:
        state = _finalizing_state("singleton-one")
        _set_item_belief(state, item_id=15, expected_score=2.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["15"]

        self.assertEqual(int(result["final_item_scores"][15]), 1)
        self.assertTrue(bool(detail["low_signal_singleton_trim_applied"]))

    def test_strong_anchor_observed_item_bypasses_local_trim(self) -> None:
        state = _finalizing_state("singleton-bypass")
        _set_item_belief(state, item_id=4, expected_score=2.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=10, expected_score=1.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=5, expected_score=2.1, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=15, expected_score=2.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=20, expected_score=1.2, support_count=1, entropy=0.20)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["15"]

        self.assertTrue(bool(result["module_imputation"]["low_signal_guardrail_active"]))
        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(result["module_imputation"]["guardrail_bypass_source"], "item_local_severe_recovery")
        self.assertEqual(int(result["final_item_scores"][15]), 2)
        self.assertFalse(bool(detail["low_signal_singleton_trim_applied"]))
        self.assertTrue(bool(detail["strong_anchor_local_bypass_applied"]))

    def test_severe_recovery_still_trims_non_anchor_singleton_observed_items(self) -> None:
        state = _finalizing_state("severe-recovery-non-anchor-trim")
        _set_item_belief(state, item_id=13, expected_score=2.2, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=15, turn=2)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["13"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][13]), 1)
        self.assertTrue(bool(detail["low_signal_singleton_trim_applied"]))
        self.assertFalse(bool(detail["strong_anchor_local_bypass_applied"]))

    def test_low_signal_single_passive_risk_without_core_corroboration_forces_zero(self) -> None:
        state = _finalizing_state("item9-passive-zero")
        state["risk"] = RiskState(risk_prob=0.68, risk_flag=True, reason="passive death ideation cue")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["9"]

        self.assertEqual(int(result["final_item_scores"][9]), 0)
        self.assertTrue(bool(detail["low_signal_item9_guardrail_applied"]))
        self.assertEqual(result["module_imputation"]["low_signal_item9_cap_reason"], "passive_without_corroborated_core_forced_zero")

    def test_low_signal_single_passive_risk_with_core_corroboration_caps_at_one(self) -> None:
        state = _finalizing_state("item9-passive-one")
        state["risk"] = RiskState(risk_prob=0.68, risk_flag=True, reason="passive death ideation cue")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=4, expected_score=1.6, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=10, expected_score=1.2, support_count=1, entropy=1.20)
        _add_evidence(state, item_id=4, turn=1)
        _add_evidence(state, item_id=10, turn=2)

        result = finalize_outputs(state)

        self.assertEqual(int(result["final_item_scores"][9]), 1)

    def test_low_signal_multiple_passive_risk_caps_at_one(self) -> None:
        state = _finalizing_state("item9-multi-passive")
        state["risk"] = RiskState(risk_prob=0.82, risk_flag=True, reason="multiple passive death ideation cues")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.7, support_count=2, entropy=0.60)
        _add_evidence(state, item_id=9, turn=1)
        _add_evidence(state, item_id=9, turn=2)

        result = finalize_outputs(state)

        self.assertEqual(int(result["final_item_scores"][9]), 1)
        self.assertEqual(result["module_imputation"]["low_signal_item9_cap_reason"], "multiple_passive_capped_at_one")

    def test_low_signal_active_risk_can_stay_positive(self) -> None:
        state = _finalizing_state("item9-active")
        state["risk"] = RiskState(risk_prob=0.98, risk_flag=True, reason="active self-harm cue match")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.2, support_count=1, entropy=1.20)

        result = finalize_outputs(state)

        self.assertGreaterEqual(int(result["final_item_scores"][9]), 1)

    def test_high_signal_item_nine_behavior_is_unchanged(self) -> None:
        state = _finalizing_state("item9-high-signal")
        state["risk"] = RiskState(risk_prob=0.68, risk_flag=True, reason="passive death ideation cue")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.2, support_count=1, entropy=1.20)
        _set_item_belief(state, item_id=4, expected_score=2.0, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=10, expected_score=1.2, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=5, expected_score=2.0, support_count=1, entropy=0.20)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.20)
        _add_evidence(state, item_id=4, turn=1)
        _add_evidence(state, item_id=10, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_evidence(state, item_id=14, turn=2)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][9]), 1)

    def test_severe_recovery_keeps_anchored_observed_twos(self) -> None:
        state = _finalizing_state("severe-recovery-observed")
        _set_item_belief(state, item_id=14, expected_score=2.2, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.2, support_count=1, entropy=0.90)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        module_imputation = result["module_imputation"]

        self.assertTrue(bool(module_imputation["low_signal_guardrail_active"]))
        self.assertTrue(bool(module_imputation["severe_recovery_mode_active"]))
        self.assertEqual(module_imputation["guardrail_bypass_source"], "item_local_severe_recovery")
        self.assertEqual(int(result["final_item_scores"][14]), 2)
        self.assertEqual(int(result["final_item_scores"][15]), 2)
        self.assertEqual(module_imputation["severe_recovery_reason"], "multiple_strong_anchor_modules")
        self.assertIn(14, module_imputation["strong_anchor_local_bypass_item_ids"])
        self.assertIn(15, module_imputation["strong_anchor_local_bypass_item_ids"])

    def test_severe_amplitude_observed_item_recovers_to_two(self) -> None:
        state = _finalizing_state("severe-amplitude-observed")
        _set_item_belief(state, item_id=14, expected_score=1.6, support_count=1, entropy=1.40)
        _set_item_belief(state, item_id=5, expected_score=1.0, support_count=1, entropy=1.20)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][14]), 2)
        self.assertTrue(bool(detail["severe_amplitude_observed_applied"]))
        self.assertIn(14, result["module_imputation"]["severe_amplitude_observed_item_ids"])
        self.assertFalse(bool(detail["severe_amplitude_observed_to_three"]))

    def test_observed_somatic_severe_item_can_recover_to_three(self) -> None:
        state = _finalizing_state("severe-amplitude-observed-three")
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.4, support_count=2, entropy=0.80)
        _add_evidence(state, item_id=14, turn=1, confidence=0.80, intensity=2.0)
        _add_evidence(state, item_id=15, turn=1, confidence=0.80, intensity=2.0)
        _add_evidence(state, item_id=15, turn=2, confidence=0.82, intensity=2.2)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["15"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][15]), 3)
        self.assertTrue(bool(detail["severe_amplitude_observed_applied"]))
        self.assertTrue(bool(detail["severe_amplitude_observed_to_three"]))
        self.assertIn(15, result["module_imputation"]["severe_amplitude_observed_to_three_item_ids"])

    def test_non_somatic_observed_severe_item_still_caps_at_two(self) -> None:
        state = _finalizing_state("severe-amplitude-observed-two-only")
        _set_item_belief(state, item_id=14, expected_score=2.4, support_count=2, entropy=0.80)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1, confidence=0.80, intensity=2.0)
        _add_evidence(state, item_id=14, turn=2, confidence=0.82, intensity=2.2)
        _add_evidence(state, item_id=15, turn=3, confidence=0.80, intensity=2.0)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][14]), 2)
        self.assertFalse(bool(detail["severe_amplitude_observed_to_three"]))

    def test_anchored_observed_without_evidence_strength_does_not_uplift(self) -> None:
        state = _finalizing_state("severe-amplitude-observed-none")
        _set_item_belief(state, item_id=14, expected_score=1.4, support_count=1, entropy=1.40)
        _set_item_belief(state, item_id=5, expected_score=1.0, support_count=1, entropy=1.20)
        _add_evidence(state, item_id=14, turn=1, confidence=0.40, intensity=1.0)
        _add_evidence(state, item_id=5, turn=2, confidence=0.40, intensity=1.0)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][14]), 1)
        self.assertFalse(bool(detail["severe_amplitude_observed_applied"]))
        self.assertEqual(result["module_imputation"]["severe_amplitude_observed_item_ids"], [])

    def test_severe_recovery_false_profiles_never_get_amplitude_rescue(self) -> None:
        state = _finalizing_state("severe-amplitude-off")
        _set_item_belief(state, item_id=15, expected_score=1.8, support_count=1, entropy=1.30)
        _add_evidence(state, item_id=15, turn=1)

        result = finalize_outputs(state)

        self.assertFalse(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(result["module_imputation"]["severe_amplitude_observed_item_ids"], [])
        self.assertEqual(result["module_imputation"]["severe_amplitude_imputed_item_ids"], [])
        self.assertFalse(bool(result["module_imputation"]["severe_item9_rescued"]))

    def test_strong_anchor_module_imputed_candidate_recovers_to_two(self) -> None:
        state = _finalizing_state("severe-recovery-imputed")
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=15, turn=2)
        _add_persona_reply(state, "I feel like a burden and I do not matter.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertTrue(bool(result["module_imputation"]["low_signal_guardrail_active"]))
        self.assertEqual(int(result["final_item_scores"][11]), 2)
        self.assertEqual(int(result["final_item_scores"][20]), 1)
        self.assertEqual(int(result["final_item_scores"][18]), 0)
        self.assertIn(11, result["module_imputation"]["severe_amplitude_imputed_item_ids"])
        self.assertNotIn(20, result["module_imputation"]["severe_amplitude_imputed_item_ids"])

    def test_module_three_severe_restore_recovers_up_to_two_zeroed_items(self) -> None:
        state = _finalizing_state("module-three-severe-restore")
        _set_item_belief(state, item_id=5, expected_score=2.4, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=12, expected_score=2.0, support_count=2, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=2, entropy=0.90)
        _add_evidence(state, item_id=5, turn=1, confidence=0.75, intensity=2.4)
        _add_evidence(state, item_id=12, turn=2, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=12, turn=3, confidence=0.78, intensity=2.1)
        _add_evidence(state, item_id=15, turn=4, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=15, turn=5, confidence=0.78, intensity=2.1)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        restored_item_ids = result["module_imputation"]["severe_module3_restored_item_ids"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["module_imputation"]["severe_module3_restore_budget"]), 2)
        self.assertEqual(len(restored_item_ids), 2)
        self.assertTrue(all(item_id in {3, 6, 7, 8, 14} for item_id in restored_item_ids))
        self.assertTrue(all(bool(result["module_imputation"]["item_details"][str(item_id)]["severe_module3_restore_applied"]) for item_id in restored_item_ids))
        self.assertTrue(all(int(result["final_item_scores"][item_id]) >= 1 for item_id in restored_item_ids))

    def test_module_three_severe_restore_requires_observed_module_three_severity(self) -> None:
        state = _finalizing_state("module-three-severe-restore-off")
        _set_item_belief(state, item_id=12, expected_score=2.0, support_count=2, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=2, entropy=0.90)
        _add_evidence(state, item_id=12, turn=1, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=12, turn=2, confidence=0.78, intensity=2.1)
        _add_evidence(state, item_id=15, turn=3, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=15, turn=4, confidence=0.78, intensity=2.1)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(result["module_imputation"]["severe_module3_restored_item_ids"], [])

    def test_module_three_severe_restore_does_not_restore_non_module_three_items(self) -> None:
        state = _finalizing_state("module-three-severe-restore-only-module-three")
        _set_item_belief(state, item_id=5, expected_score=2.4, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=12, expected_score=2.0, support_count=2, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=2, entropy=0.90)
        _add_evidence(state, item_id=5, turn=1, confidence=0.75, intensity=2.4)
        _add_evidence(state, item_id=12, turn=2, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=12, turn=3, confidence=0.78, intensity=2.1)
        _add_evidence(state, item_id=15, turn=4, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=15, turn=5, confidence=0.78, intensity=2.1)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        restored_item_ids = result["module_imputation"]["severe_module3_restored_item_ids"]
        self.assertTrue(restored_item_ids)
        self.assertTrue(all(item_id in {3, 5, 6, 7, 8, 14} for item_id in restored_item_ids))

    def test_module_three_severe_restore_prioritizes_item_fourteen_on_worthlessness_hits(self) -> None:
        state = _finalizing_state("module-three-item14-priority")
        _set_item_belief(state, item_id=5, expected_score=2.4, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=12, expected_score=2.0, support_count=2, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=2, entropy=0.90)
        _add_evidence(state, item_id=5, turn=1, confidence=0.75, intensity=2.4)
        _add_evidence(state, item_id=12, turn=2, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=12, turn=3, confidence=0.78, intensity=2.1)
        _add_evidence(state, item_id=15, turn=4, confidence=0.75, intensity=2.0)
        _add_evidence(state, item_id=15, turn=5, confidence=0.78, intensity=2.1)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_module3_item14_priority_applied"]))
        self.assertIn(14, result["module_imputation"]["severe_module3_restored_item_ids"])

    def test_non_strong_anchor_module_imputed_candidate_is_not_restored(self) -> None:
        state = _finalizing_state("non-strong-anchor-imputed")
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=20, expected_score=1.2, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_evidence(state, item_id=15, turn=3)
        _add_evidence(state, item_id=20, turn=4)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "It takes a little more effort to get going than it used to.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertLessEqual(int(result["final_item_scores"][11]), 1)
        self.assertNotIn(11, result["module_imputation"]["severe_recovered_item_ids"])
        self.assertNotIn(11, result["module_imputation"]["severe_amplitude_imputed_item_ids"])

    def test_strong_anchor_imputed_restore_budget_is_one_per_module(self) -> None:
        state = _finalizing_state("module-three-budget")
        _set_item_belief(state, item_id=14, expected_score=2.2, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_evidence(state, item_id=15, turn=3)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        restored_module_three = result["module_imputation"]["severe_recovered_item_ids_by_module"].get("3", [])

        self.assertEqual(len(restored_module_three), 1)
        self.assertEqual(
            len(
                [
                    item_id
                    for item_id in restored_module_three
                    if item_id in {3, 6, 7, 8}
                ]
            ),
            1,
        )

    def test_severe_recovery_does_not_loosen_item_nine(self) -> None:
        state = _finalizing_state("severe-recovery-item9")
        state["risk"] = RiskState(risk_prob=0.68, risk_flag=True, reason="passive death ideation cue")
        state["risk_flag"] = True
        _set_item_belief(state, item_id=9, expected_score=2.4, support_count=1, entropy=0.70)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=15, expected_score=2.0, support_count=1, entropy=0.90)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][9]), 0)

    def test_item_twenty_one_mild_observed_signal_is_retained_to_one(self) -> None:
        state = _finalizing_state("item21-mild-retained")
        _set_item_belief(state, item_id=21, expected_score=1.2, support_count=1, entropy=1.30)
        _add_evidence(state, item_id=21, turn=1, confidence=0.6, intensity=1.0)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")
        _add_persona_reply(state, "That side of things is a little lower than usual, not a big change though.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["21"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertTrue(bool(result["module_imputation"]["item21_mild_observed_retained"]))
        self.assertEqual(int(result["final_item_scores"][21]), 1)
        self.assertTrue(bool(detail["item21_mild_observed_retained"]))

    def test_item_twenty_one_without_explicit_phrase_is_not_retained(self) -> None:
        state = _finalizing_state("item21-mild-not-retained")
        _set_item_belief(state, item_id=21, expected_score=1.2, support_count=1, entropy=1.30)
        _add_evidence(state, item_id=21, turn=1, confidence=0.6, intensity=1.0)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")
        _add_persona_reply(state, "That side of things has been fine.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["21"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertFalse(bool(result["module_imputation"]["item21_mild_observed_retained"]))
        self.assertEqual(int(result["final_item_scores"][21]), 0)
        self.assertFalse(bool(detail["item21_mild_observed_retained"]))

    def test_item_fourteen_latent_restore_fires_with_module_three_corroboration(self) -> None:
        state = _finalizing_state("item14-latent-restored")
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=1.00)
        _set_item_belief(state, item_id=7, expected_score=1.6, support_count=1, entropy=1.00)
        _add_evidence(state, item_id=5, turn=1, confidence=0.6, intensity=1.6)
        _add_evidence(state, item_id=7, turn=2, confidence=0.6, intensity=1.6)
        _add_persona_reply(state, "I feel like it's my own fault and I don't like who I am right now.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertEqual(int(result["final_item_scores"][14]), 2)
        self.assertIn(14, result["module_imputation"]["item14_latent_restored_item_ids"])
        self.assertTrue(bool(detail["item14_latent_restore_applied"]))

    def test_item_fourteen_latent_restore_does_not_fire_for_generic_stress(self) -> None:
        state = _finalizing_state("item14-latent-no-generic-stress")
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=1.00)
        _set_item_belief(state, item_id=7, expected_score=1.6, support_count=1, entropy=1.00)
        _add_evidence(state, item_id=5, turn=1, confidence=0.6, intensity=1.6)
        _add_evidence(state, item_id=7, turn=2, confidence=0.6, intensity=1.6)
        _add_persona_reply(state, "Work has been busy and everything feels heavier lately.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertEqual(int(result["final_item_scores"][14]), 0)
        self.assertEqual(result["module_imputation"]["item14_latent_restored_item_ids"], [])
        self.assertFalse(bool(detail["item14_latent_restore_applied"]))

    def test_item_fourteen_latent_restore_requires_more_than_one_weak_companion_without_severe_mode(self) -> None:
        state = _finalizing_state("item14-latent-one-companion")
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=1.00)
        _add_evidence(state, item_id=5, turn=1, confidence=0.6, intensity=1.6)
        _add_persona_reply(state, "I feel like it's my own fault and I don't like who I am right now.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["14"]

        self.assertFalse(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][14]), 0)
        self.assertFalse(bool(detail["item14_latent_restore_applied"]))

    def test_item_twenty_one_imputed_restore_fires_with_question_history_and_one_soft_denial(self) -> None:
        state = _finalizing_state("item21-imputed-restored")
        _set_item_belief(state, item_id=20, expected_score=1.7, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=20, turn=1, confidence=0.7, intensity=1.8)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")
        _add_detector_question(
            state,
            "In the past two weeks, how often have you noticed a reduced interest in sexual activity compared with your usual level?",
        )
        _add_persona_reply(state, "That side of things has been fine.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["21"]

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertTrue(bool(result["module_imputation"]["item21_question_history_hit"]))
        self.assertEqual(int(result["module_imputation"]["item21_direct_denial_count"]), 1)
        self.assertTrue(bool(result["module_imputation"]["item21_imputed_restored"]))
        self.assertEqual(int(result["final_item_scores"][21]), 1)
        self.assertTrue(bool(detail["item21_imputed_restore_applied"]))

    def test_item_twenty_one_imputed_restore_requires_question_history(self) -> None:
        state = _finalizing_state("item21-imputed-no-question")
        _set_item_belief(state, item_id=20, expected_score=1.7, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=20, turn=1, confidence=0.7, intensity=1.8)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["21"]

        self.assertFalse(bool(result["module_imputation"]["item21_question_history_hit"]))
        self.assertFalse(bool(result["module_imputation"]["item21_imputed_restored"]))
        self.assertEqual(int(result["final_item_scores"][21]), 0)
        self.assertFalse(bool(detail["item21_imputed_restore_applied"]))

    def test_item_twenty_one_imputed_restore_is_blocked_by_repeated_direct_denials(self) -> None:
        state = _finalizing_state("item21-imputed-two-denials")
        _set_item_belief(state, item_id=20, expected_score=1.7, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=20, turn=1, confidence=0.7, intensity=1.8)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")
        _add_detector_question(
            state,
            "In the past two weeks, how often have you noticed a reduced interest in sexual activity compared with your usual level?",
        )
        _add_persona_reply(state, "That side of things has been fine.")
        _add_detector_question(
            state,
            "In the past two weeks, how often have you noticed a reduced interest in sexual activity compared with your usual level?",
        )
        _add_persona_reply(state, "Okay honestly, that side of things has been fine.")

        result = finalize_outputs(state)
        detail = result["module_imputation"]["item_details"]["21"]

        self.assertTrue(bool(result["module_imputation"]["item21_question_history_hit"]))
        self.assertEqual(int(result["module_imputation"]["item21_direct_denial_count"]), 2)
        self.assertFalse(bool(result["module_imputation"]["item21_imputed_restored"]))
        self.assertEqual(int(result["final_item_scores"][21]), 0)
        self.assertFalse(bool(detail["item21_imputed_restore_applied"]))

    def test_severe_item_nine_rescue_recovers_to_one(self) -> None:
        state = _finalizing_state("severe-item9-rescue")
        state["risk"] = RiskState(risk_prob=0.20, risk_flag=False, reason="no lexical risk cue")
        _set_item_belief(state, item_id=9, expected_score=1.6, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=9, turn=1)
        _add_evidence(state, item_id=14, turn=2)
        _add_evidence(state, item_id=5, turn=3)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][9]), 1)
        self.assertTrue(bool(result["module_imputation"]["severe_item9_rescued"]))

    def test_severe_item_nine_without_direct_support_stays_unchanged(self) -> None:
        state = _finalizing_state("severe-item9-no-rescue")
        state["risk"] = RiskState(risk_prob=0.20, risk_flag=False, reason="no lexical risk cue")
        _set_item_belief(state, item_id=9, expected_score=1.6, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=14, expected_score=2.0, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=14, turn=1)
        _add_evidence(state, item_id=5, turn=2)
        _add_persona_reply(state, "I genuinely feel worthless and like I do not contribute anything that matters.")
        _add_persona_reply(state, "Even getting out of bed is a battle and everything takes so much energy.")

        result = finalize_outputs(state)

        self.assertTrue(bool(result["module_imputation"]["severe_recovery_mode_active"]))
        self.assertEqual(int(result["final_item_scores"][9]), 0)
        self.assertFalse(bool(result["module_imputation"]["severe_item9_rescued"]))

    def test_same_module_weak_pairs_do_not_self_validate(self) -> None:
        state = _finalizing_state("weak-pairs-not-corroborated")
        _set_item_belief(state, item_id=3, expected_score=1.2, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=1.2, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=3, turn=1)
        _add_evidence(state, item_id=5, turn=2)

        result = finalize_outputs(state)
        item_three = result["module_imputation"]["item_details"]["3"]
        item_five = result["module_imputation"]["item_details"]["5"]

        self.assertFalse(bool(item_three["is_corroborated_item"]))
        self.assertFalse(bool(item_five["is_corroborated_item"]))
        self.assertEqual(item_three["same_module_corroborated_item_count"], 0)
        self.assertEqual(item_five["same_module_corroborated_item_count"], 0)

    def test_same_module_stronger_pair_can_still_corroborate(self) -> None:
        state = _finalizing_state("stronger-pair-corroborated")
        _set_item_belief(state, item_id=3, expected_score=1.2, support_count=1, entropy=0.90)
        _set_item_belief(state, item_id=5, expected_score=1.6, support_count=1, entropy=0.90)
        _add_evidence(state, item_id=3, turn=1)
        _add_evidence(state, item_id=5, turn=2)

        result = finalize_outputs(state)
        item_three = result["module_imputation"]["item_details"]["3"]
        item_five = result["module_imputation"]["item_details"]["5"]

        self.assertTrue(bool(item_three["is_corroborated_item"]))
        self.assertTrue(bool(item_five["is_corroborated_item"]))
        self.assertIn(5, item_three["same_module_corroborated_item_ids"])
        self.assertIn(3, item_five["same_module_corroborated_item_ids"])


if __name__ == "__main__":
    unittest.main()
