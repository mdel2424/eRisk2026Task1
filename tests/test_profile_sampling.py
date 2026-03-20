from __future__ import annotations

import builtins
import random
import unittest
from unittest.mock import patch

from agents.finalize_outputs import finalize_outputs
from core.bdi_modules import MODULE_TO_ITEMS
from core.state import ControlState, ItemBelief, build_initial_state, posterior_from_expected_score
from persona.profile_sampling import (
    FAMILY_BLUEPRINTS,
    SEVERITY_TIERS,
    _family_module_emphasis,
    _sample_bdi_scores_for_family,
    generate_persona_pool,
)
from persona.sim_behavior import response_style_flags
from persona.sim_templates import QUALIFIED_UNSURE_PHRASES
from persona.simulated_persona import SimulatedPersona


class ProfileSamplingTests(unittest.TestCase):
    def test_finalize_outputs_still_emits_module_imputation(self) -> None:
        state = build_initial_state(persona_id="regression")
        state["control"] = ControlState(stop=True, stop_reason="test")
        state["raw_predicted_bdi_score"] = 6
        state["raw_predicted_label"] = "control"
        state["risk_flag"] = False
        state["item_beliefs"][16] = ItemBelief(
            item_id=16,
            posterior=posterior_from_expected_score(2.4),
            entropy=0.30,
            expected_score=2.4,
            support_count=2,
            last_update_turn=1,
        )

        result = finalize_outputs(state)

        self.assertIn("module_imputation", result)
        module_imputation = result["module_imputation"]
        self.assertIn(6, module_imputation["module_stats"])
        self.assertEqual(module_imputation["item_details"]["18"]["source"], "imputed")
        self.assertGreaterEqual(int(module_imputation["imputed_points_before_guardrail"]), 1)
        self.assertEqual(int(module_imputation["imputed_points_after_guardrail"]), 0)
        self.assertIn(18, module_imputation["suppressed_imputed_item_ids"])

    def test_module_six_items_stay_soft_coupled(self) -> None:
        for seed in range(1, 121):
            rng = random.Random(seed)
            for family, blueprint in FAMILY_BLUEPRINTS.items():
                severity = "minimal" if not blueprint["depressed"] else "moderate"
                scores = _sample_bdi_scores_for_family(family, rng, severity=severity)
                self.assertLessEqual(abs(int(scores[16]) - int(scores[18])), 1, msg=f"{family} seed={seed}")

    def test_family_module_emphasis_produces_distinct_sleep_appetite_weight(self) -> None:
        somatic_emphasis = _family_module_emphasis(FAMILY_BLUEPRINTS["somatic_evasive"])
        cognitive_emphasis = _family_module_emphasis(FAMILY_BLUEPRINTS["cognitive_ruminative"])
        control_emphasis = _family_module_emphasis(FAMILY_BLUEPRINTS["control_neutral"])

        self.assertGreater(float(somatic_emphasis[6]), float(cognitive_emphasis[6]))
        self.assertGreater(float(somatic_emphasis[6]), float(control_emphasis[6]))

    def test_family_patterns_show_up_in_generated_scores(self) -> None:
        somatic_scores = []
        control_scores = []
        for seed in range(50):
            somatic_scores.append(_sample_bdi_scores_for_family("somatic_evasive", random.Random(seed), severity="moderate"))
            control_scores.append(_sample_bdi_scores_for_family("control_neutral", random.Random(seed), severity="minimal"))

        somatic_sleep_appetite = sum((row[16] + row[18]) for row in somatic_scores) / float(len(somatic_scores))
        control_sleep_appetite = sum((row[16] + row[18]) for row in control_scores) / float(len(control_scores))
        self.assertGreater(somatic_sleep_appetite, control_sleep_appetite)

    def test_depressed_family_totals_stay_within_severity_ranges(self) -> None:
        depressed_families = [family for family, blueprint in FAMILY_BLUEPRINTS.items() if blueprint["depressed"]]

        for family in depressed_families:
            for severity, tier in SEVERITY_TIERS.items():
                for seed in range(10):
                    scores = _sample_bdi_scores_for_family(family, random.Random((seed + 1) * 17), severity=severity)
                    total = sum(int(value) for value in scores.values())
                    self.assertGreaterEqual(total, int(tier["floor"]), msg=f"{family} {severity} {seed}")
                    self.assertLessEqual(total, int(tier["ceiling"]), msg=f"{family} {severity} {seed}")

    def test_risk_leaning_profiles_keep_cognitive_support_when_risk_is_high(self) -> None:
        cognitive_support = [2, 3, 5, 8, 14]
        saw_high_risk = False

        for seed in range(1, 101):
            scores = _sample_bdi_scores_for_family("risk_leaning", random.Random(seed), severity="severe")
            if int(scores[9]) < 2:
                continue
            saw_high_risk = True
            support_hits = sum(1 for item_id in cognitive_support if int(scores[item_id]) >= 2)
            self.assertGreaterEqual(support_hits, 2, msg=f"seed={seed}")

        self.assertTrue(saw_high_risk)

    def test_minimal_risk_leaning_profiles_respect_nonrisk_breadth_caps(self) -> None:
        nonrisk_core_ids = {2, 5, 8, 14}
        secondary_ids = {3, 4, 12, 15, 19}

        for seed in range(1, 81):
            scores = _sample_bdi_scores_for_family("risk_leaning", random.Random(seed), severity="minimal")
            nonrisk_positive_ids = [item_id for item_id, score in scores.items() if item_id != 9 and int(score) > 0]
            self.assertLessEqual(len(nonrisk_positive_ids), 4, msg=f"seed={seed}")
            self.assertTrue(all(int(scores[item_id]) <= 1 for item_id in nonrisk_positive_ids), msg=f"seed={seed}")
            self.assertLessEqual(
                sum(1 for item_id in nonrisk_core_ids if int(scores[item_id]) > 0),
                2,
                msg=f"seed={seed}",
            )
            self.assertLessEqual(
                sum(1 for item_id in secondary_ids if int(scores[item_id]) > 0),
                1,
                msg=f"seed={seed}",
            )

    def test_minimal_cognitive_ruminative_profiles_respect_spillover_caps(self) -> None:
        module34_item_ids = set(MODULE_TO_ITEMS[3]) | set(MODULE_TO_ITEMS[4])

        for seed in range(1, 81):
            scores = _sample_bdi_scores_for_family("cognitive_ruminative", random.Random(seed), severity="minimal")
            positive_ids = [item_id for item_id, score in scores.items() if int(score) > 0]
            outside_ids = [item_id for item_id in positive_ids if item_id not in module34_item_ids]
            self.assertLessEqual(len(positive_ids), 4, msg=f"seed={seed}")
            self.assertLessEqual(len(outside_ids), 1, msg=f"seed={seed}")
            self.assertTrue(all(int(scores[item_id]) <= 1 for item_id in outside_ids), msg=f"seed={seed}")

    def test_generate_persona_pool_is_deterministic_for_seed(self) -> None:
        left = generate_persona_pool(count=12, seed=42)
        right = generate_persona_pool(count=12, seed=42)

        self.assertEqual(
            [
                (
                    profile.persona_id,
                    profile.family,
                    profile.severity_tier,
                    profile.subtype_tag,
                    profile.context_tag,
                    profile.style_tag,
                    profile.bdi_scores,
                )
                for profile in left
            ],
            [
                (
                    profile.persona_id,
                    profile.family,
                    profile.severity_tier,
                    profile.subtype_tag,
                    profile.context_tag,
                    profile.style_tag,
                    profile.bdi_scores,
                )
                for profile in right
            ],
        )

    def test_generate_persona_pool_does_not_read_eval_artifacts(self) -> None:
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected file read")):
            profiles = generate_persona_pool(count=6, seed=7)

        self.assertEqual(len(profiles), 6)

    def test_generate_persona_pool_exposes_diversity_metadata(self) -> None:
        profiles = generate_persona_pool(count=24, seed=42)

        subtype_tags = {profile.subtype_tag for profile in profiles}
        context_tags = {profile.context_tag for profile in profiles}
        style_tags = {profile.style_tag for profile in profiles}
        severity_tiers = {profile.severity_tier for profile in profiles}

        self.assertGreaterEqual(len(subtype_tags), 6)
        self.assertGreaterEqual(len(context_tags), 5)
        self.assertGreaterEqual(len(style_tags), 4)
        self.assertTrue({"minimal", "mild", "moderate", "severe"} & severity_tiers)

    def test_generate_persona_pool_reduces_within_family_repetition(self) -> None:
        profiles = generate_persona_pool(count=48, seed=42)

        by_family: dict[str, set[str]] = {}
        for profile in profiles:
            by_family.setdefault(profile.family, set()).add(profile.subtype_tag)

        diverse_families = [family for family, subtype_tags in by_family.items() if len(subtype_tags) >= 2]
        self.assertGreaterEqual(len(diverse_families), 4)

    def test_simulated_persona_tags_change_surface_form_deterministically(self) -> None:
        probe_intent = {
            "target_item_id": 15,
            "route": "somatic",
            "style": "impact",
            "mode": "normal",
            "directness": "indirect",
            "priority": 0.7,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[15] = 2
        base_behavior = {
            "evasiveness": 0.42,
            "verbosity": 0.48,
            "contradiction": 0.08,
            "affect_volatility": 0.18,
            "hedge_rate": 0.60,
            "normalization_rate": 0.40,
            "context_anchor_rate": 0.70,
            "direct_answer_rate": 0.80,
        }

        workload_persona = SimulatedPersona(
            persona_id="101",
            bdi_scores=dict(scores),
            family="functional_masked",
            split="eval",
            context_tag="workload",
            style_tag="minimizing_practical",
            behavior_params=dict(base_behavior),
        )
        caregiving_persona = SimulatedPersona(
            persona_id="101",
            bdi_scores=dict(scores),
            family="functional_masked",
            split="eval",
            context_tag="caregiving",
            style_tag="contextual_reflective",
            behavior_params=dict(base_behavior),
        )

        left = workload_persona.reply([], dict(probe_intent))
        right = caregiving_persona.reply([], dict(probe_intent))

        self.assertNotEqual(left, right)
        self.assertEqual(left, SimulatedPersona(
            persona_id="101",
            bdi_scores=dict(scores),
            family="functional_masked",
            split="eval",
            context_tag="workload",
            style_tag="minimizing_practical",
            behavior_params=dict(base_behavior),
        ).reply([], dict(probe_intent)))

    def test_simulated_persona_can_produce_mixed_contrastive_answer(self) -> None:
        probe_intent = {
            "target_item_id": 15,
            "route": "somatic",
            "style": "functional_impact",
            "mode": "normal",
            "directness": "indirect",
            "priority": 0.8,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[4] = 2
        scores[15] = 2
        scores[20] = 2

        persona = SimulatedPersona(
            persona_id="202",
            bdi_scores=scores,
            family="mixed_moderate",
            split="eval",
            context_tag="workload",
            style_tag="contextual_reflective",
            behavior_params={
                "evasiveness": 0.35,
                "verbosity": 0.55,
                "contradiction": 0.05,
                "affect_volatility": 0.18,
                "hedge_rate": 0.45,
                "normalization_rate": 0.30,
                "context_anchor_rate": 0.65,
                "direct_answer_rate": 0.88,
            },
        )

        history = [{"role": "user", "content": "Earlier probe"}, {"role": "assistant", "content": "Earlier answer."}]
        reply = persona.reply(history, dict(probe_intent))
        flags = response_style_flags(reply)

        self.assertTrue(flags["mixed_answer"])
        self.assertTrue("both" in reply.lower() or "bit of both" in reply.lower())

    def test_simulated_persona_opening_turn_produces_context_summary(self) -> None:
        probe_intent = {
            "target_item_id": 2,
            "route": "cognitive",
            "style": "opening",
            "mode": "normal",
            "directness": "indirect",
            "priority": 0.7,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[15] = 3
        scores[16] = 2
        scores[20] = 2

        persona = SimulatedPersona(
            persona_id="210",
            bdi_scores=scores,
            family="somatic_evasive",
            split="eval",
            context_tag="health_stress",
            style_tag="contextual_reflective",
            behavior_params={
                "evasiveness": 0.35,
                "verbosity": 0.52,
                "contradiction": 0.05,
                "affect_volatility": 0.18,
                "hedge_rate": 0.45,
                "normalization_rate": 0.16,
                "context_anchor_rate": 0.62,
                "direct_answer_rate": 0.88,
            },
        )

        reply = persona.reply([], dict(probe_intent))
        flags = response_style_flags(reply)

        self.assertFalse(flags["soft_denial"])
        self.assertTrue("main change" in reply.lower() or "what stands out" in reply.lower() or "biggest shift" in reply.lower())
        self.assertEqual(int(persona.style_stats()["opening_summary_count"]), 1)

    def test_simulated_persona_uses_contrastive_negative_for_zero_target_with_active_cluster(self) -> None:
        probe_intent = {
            "target_item_id": 1,
            "route": "cognitive",
            "style": "gentle_probe",
            "mode": "wrapup",
            "directness": "direct",
            "priority": 0.7,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[17] = 2
        scores[20] = 2

        persona = SimulatedPersona(
            persona_id="203",
            bdi_scores=scores,
            family="somatic_evasive",
            split="eval",
            context_tag="health_stress",
            style_tag="minimizing_practical",
            behavior_params={
                "evasiveness": 0.30,
                "verbosity": 0.42,
                "contradiction": 0.03,
                "affect_volatility": 0.10,
                "hedge_rate": 0.40,
                "normalization_rate": 0.18,
                "context_anchor_rate": 0.45,
                "direct_answer_rate": 0.90,
            },
        )

        history = [
            {"role": "user", "content": "Earlier probe"},
            {"role": "assistant", "content": "That part feels pretty close to normal for me."},
        ]
        reply = persona.reply(history, dict(probe_intent))
        flags = response_style_flags(reply)

        self.assertFalse(flags["soft_denial"])
        self.assertIn("irritability", reply.lower())
        self.assertEqual(int(persona.style_stats()["contrastive_negative_count"]), 1)

    def test_same_item_followup_elaborates_instead_of_resetting(self) -> None:
        probe_intent = {
            "target_item_id": 14,
            "route": "cognitive",
            "style": "clarify_frequency",
            "mode": "normal",
            "directness": "direct",
            "priority": 0.8,
            "question_kind": "same_item_followup",
            "thread_turn_index": 2,
            "thread_module_id": 3,
            "thread_source_item_id": 14,
            "timeframe_mode": "carry",
            "anchor_text": "I feel like a burden",
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[14] = 2

        persona = SimulatedPersona(
            persona_id="214",
            bdi_scores=scores,
            family="cognitive_ruminative",
            split="eval",
            context_tag="workload",
            style_tag="contextual_reflective",
        )

        history = [
            {"role": "user", "content": "Lately, what do you tend to tell yourself when things feel heavy?"},
            {"role": "assistant", "content": "I feel like a burden, especially when messages pile up."},
        ]
        reply = persona.reply(history, dict(probe_intent))

        self.assertNotIn("seems a little different", reply.lower())
        self.assertTrue("most days" in reply.lower() or "hours" in reply.lower() or "messages" in reply.lower())

    def test_repeated_denial_followup_stays_brief_and_consistent(self) -> None:
        probe_intent = {
            "target_item_id": 18,
            "route": "somatic",
            "style": "gentle_probe",
            "mode": "normal",
            "directness": "indirect",
            "priority": 0.5,
            "question_kind": "same_item_followup",
            "thread_turn_index": 3,
            "thread_module_id": 6,
            "thread_source_item_id": 18,
            "timeframe_mode": "carry",
            "anchor_text": "appetite feels normal",
        }
        scores = {item_id: 0 for item_id in range(1, 22)}

        persona = SimulatedPersona(
            persona_id="215",
            bdi_scores=scores,
            family="control_stressed",
            split="eval",
            context_tag="routine_stable",
            style_tag="minimizing_practical",
        )

        history = [
            {"role": "user", "content": "Has your appetite changed much lately?"},
            {"role": "assistant", "content": "Not really, that still feels about the same."},
        ]
        reply = persona.reply(history, dict(probe_intent))

        self.assertLessEqual(len(reply.split()), 12)
        self.assertTrue("same" in reply.lower() or "normal" in reply.lower() or "not really" in reply.lower())
        self.assertNotIn("seems a little different", reply.lower())

    def test_simulated_persona_soft_denial_still_exists_for_low_signal_control_case(self) -> None:
        probe_intent = {
            "target_item_id": 18,
            "route": "somatic",
            "style": "gentle_probe",
            "mode": "normal",
            "directness": "direct",
            "priority": 0.7,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}

        persona = SimulatedPersona(
            persona_id="204",
            bdi_scores=scores,
            family="control_neutral",
            split="eval",
            context_tag="routine_stable",
            style_tag="minimizing_practical",
            behavior_params={
                "evasiveness": 0.25,
                "verbosity": 0.38,
                "contradiction": 0.02,
                "affect_volatility": 0.08,
                "hedge_rate": 0.35,
                "normalization_rate": 0.12,
                "context_anchor_rate": 0.35,
                "direct_answer_rate": 0.92,
            },
        )

        history = [{"role": "user", "content": "Earlier probe"}, {"role": "assistant", "content": "Earlier answer."}]
        reply = persona.reply(history, dict(probe_intent))
        flags = response_style_flags(reply)

        self.assertTrue(flags["soft_denial"])
        self.assertTrue("normal" in reply.lower() or "same" in reply.lower() or "not really" in reply.lower())

    def test_low_severity_off_target_probe_prefers_soft_denial_over_generic_uncertainty(self) -> None:
        probe_intent = {
            "target_item_id": 18,
            "route": "somatic",
            "style": "gentle_probe",
            "mode": "normal",
            "directness": "direct",
            "priority": 0.6,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[2] = 1
        scores[9] = 1

        persona = SimulatedPersona(
            persona_id="305",
            bdi_scores=scores,
            family="risk_leaning",
            split="eval",
            context_tag="financial_pressure",
            style_tag="hedged_uncertain",
            behavior_params={
                "evasiveness": 0.40,
                "verbosity": 0.42,
                "contradiction": 0.05,
                "affect_volatility": 0.12,
                "hedge_rate": 0.50,
                "normalization_rate": 0.16,
                "context_anchor_rate": 0.42,
                "direct_answer_rate": 0.84,
            },
        )

        history = [
            {"role": "user", "content": "Opening question"},
            {"role": "assistant", "content": "Opening answer."},
            {"role": "user", "content": "How has your appetite been?"},
        ]
        reply = persona.reply(history, dict(probe_intent))
        flags = response_style_flags(reply)

        self.assertTrue(flags["soft_denial"])
        self.assertFalse(any(phrase.lower() in reply.lower() for phrase in QUALIFIED_UNSURE_PHRASES))

    def test_low_severity_profile_uses_qualified_unsure_at_most_once_in_fixed_battery(self) -> None:
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[2] = 1
        scores[14] = 1
        persona = SimulatedPersona(
            persona_id="306",
            bdi_scores=scores,
            family="cognitive_ruminative",
            split="eval",
            context_tag="school",
            style_tag="hedged_uncertain",
            behavior_params={
                "evasiveness": 0.42,
                "verbosity": 0.46,
                "contradiction": 0.06,
                "affect_volatility": 0.14,
                "hedge_rate": 0.52,
                "normalization_rate": 0.18,
                "context_anchor_rate": 0.45,
                "direct_answer_rate": 0.82,
            },
        )
        history: list[dict] = []
        probes = [
            {"target_item_id": 18, "route": "somatic", "style": "gentle_probe", "mode": "normal", "directness": "direct", "priority": 0.6},
            {"target_item_id": 21, "route": "somatic", "style": "gentle_probe", "mode": "normal", "directness": "direct", "priority": 0.6},
            {"target_item_id": 16, "route": "somatic", "style": "clarify_frequency", "mode": "normal", "directness": "direct", "priority": 0.6},
        ]

        replies = []
        for probe in probes:
            history.append({"role": "user", "content": "Probe"})
            reply = persona.reply(list(history), dict(probe))
            replies.append(reply)
            history.append({"role": "assistant", "content": reply})

        qualified_unsure_hits = sum(
            1
            for reply in replies
            if any(phrase.lower() in reply.lower() for phrase in QUALIFIED_UNSURE_PHRASES)
        )
        self.assertLessEqual(qualified_unsure_hits, 1)

    def test_simulated_persona_can_surface_sleep_and_appetite_variability_language(self) -> None:
        sleep_probe = {
            "target_item_id": 16,
            "route": "somatic",
            "style": "clarify_frequency",
            "mode": "normal",
            "directness": "direct",
            "priority": 0.75,
        }
        appetite_probe = {
            "target_item_id": 18,
            "route": "somatic",
            "style": "gentle_probe",
            "mode": "normal",
            "directness": "direct",
            "priority": 0.75,
        }
        scores = {item_id: 0 for item_id in range(1, 22)}
        scores[16] = 3
        scores[18] = 3
        scores[20] = 2
        history = [{"role": "user", "content": "Earlier probe"}, {"role": "assistant", "content": "Earlier answer."}]

        sleep_persona = SimulatedPersona(
            persona_id="205",
            bdi_scores=dict(scores),
            family="somatic_evasive",
            split="eval",
            context_tag="health_stress",
            style_tag="open_but_flat",
            behavior_params={
                "evasiveness": 0.30,
                "verbosity": 0.46,
                "contradiction": 0.03,
                "affect_volatility": 0.10,
                "hedge_rate": 0.32,
                "normalization_rate": 0.12,
                "context_anchor_rate": 0.45,
                "direct_answer_rate": 0.92,
            },
        )
        appetite_persona = SimulatedPersona(
            persona_id="206",
            bdi_scores=dict(scores),
            family="somatic_evasive",
            split="eval",
            context_tag="health_stress",
            style_tag="contextual_reflective",
            behavior_params={
                "evasiveness": 0.30,
                "verbosity": 0.50,
                "contradiction": 0.03,
                "affect_volatility": 0.10,
                "hedge_rate": 0.36,
                "normalization_rate": 0.14,
                "context_anchor_rate": 0.48,
                "direct_answer_rate": 0.92,
            },
        )

        sleep_reply = sleep_persona.reply(list(history), dict(sleep_probe))
        appetite_reply = appetite_persona.reply(list(history), dict(appetite_probe))

        self.assertTrue(
            "sleep" in sleep_reply.lower() or "night" in sleep_reply.lower() or "sleeping too much" in sleep_reply.lower()
        )
        self.assertTrue(
            "food" in appetite_reply.lower()
            or "eating" in appetite_reply.lower()
            or "up and down" in appetite_reply.lower()
        )


if __name__ == "__main__":
    unittest.main()
