from __future__ import annotations

import builtins
import random
import unittest
from unittest.mock import patch

from agents.finalize_outputs import finalize_outputs
from core.state import ControlState, ItemBelief, build_initial_state, posterior_from_expected_score
from persona.profile_sampling import (
    DEFAULT_SIM_GENERATOR_VERSION,
    FAMILY_BLUEPRINTS,
    SEVERITY_TIERS,
    _family_module_emphasis,
    _sample_bdi_scores_for_family,
    generate_persona_pool,
)


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
        self.assertGreaterEqual(int(module_imputation["imputed_item_count"]), 1)

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

    def test_generate_persona_pool_is_deterministic_for_seed(self) -> None:
        left = generate_persona_pool(count=12, seed=42)
        right = generate_persona_pool(count=12, seed=42)

        self.assertEqual(
            [(profile.persona_id, profile.family, profile.bdi_scores) for profile in left],
            [(profile.persona_id, profile.family, profile.bdi_scores) for profile in right],
        )
        self.assertTrue(all(profile.generator_version == DEFAULT_SIM_GENERATOR_VERSION for profile in left))

    def test_generate_persona_pool_does_not_read_eval_artifacts(self) -> None:
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected file read")):
            profiles = generate_persona_pool(count=6, seed=7)

        self.assertEqual(len(profiles), 6)


if __name__ == "__main__":
    unittest.main()
