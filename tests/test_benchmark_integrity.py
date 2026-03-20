from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.io_schema import PersonaResult


class BenchmarkIntegrityTests(unittest.TestCase):
    def test_integrity_report_flags_result_id_mismatch(self) -> None:
        from app.eval_artifacts import write_eval_artifacts

        manifest_payload = {
            "run_config": {"manifest_schema_version": 4, "persona_count": 2, "seed": 42},
            "persona_count": 2,
            "profiles": [
                {
                    "persona_id": "alpha",
                    "split": "eval",
                    "family": "control_neutral",
                    "severity_tier": "minimal",
                    "subtype_tag": "routine_stable",
                    "context_tag": "routine_stable",
                    "style_tag": "open_but_flat",
                    "source": "synthetic",
                    "has_ground_truth": True,
                    "depressed": False,
                    "bdi_scores": {"1": 0},
                    "bdi_total": 0,
                    "key_symptoms": [],
                    "risk_signal": False,
                    "behavior_params": {},
                    "template_bank": "default",
                    "generation_seed": 42001,
                },
                {
                    "persona_id": "beta",
                    "split": "eval",
                    "family": "control_stressed",
                    "severity_tier": "minimal",
                    "subtype_tag": "workload_strained",
                    "context_tag": "workload",
                    "style_tag": "minimizing_practical",
                    "source": "synthetic",
                    "has_ground_truth": True,
                    "depressed": False,
                    "bdi_scores": {"1": 1},
                    "bdi_total": 1,
                    "key_symptoms": ["Sadness"],
                    "risk_signal": False,
                    "behavior_params": {},
                    "template_bank": "default",
                    "generation_seed": 42002,
                },
            ],
        }

        results = [
            PersonaResult(LLM="alpha", bdi_score=0, key_symptoms=[], item_scores={"1": 0}, item_support_counts={"1": 0})
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, integrity_payload, _, _, _ = write_eval_artifacts(
                output_dir=Path(tmp_dir),
                conversations=[],
                results=results,
                diagnostics_payload=[],
                overall_rows=[],
                route_distribution=Counter(),
                turns_total=0,
                evidence_turns_nonempty=0,
                evidence_records_total=0,
                extract_source_distribution=Counter(),
                extract_recovery_distribution=Counter(),
                route_policy_distribution=Counter(),
                duplicate_evidence_rows_total=0,
                contradiction_evidence_rows_total=0,
                support_increments_total=0,
                method_weight_usage=Counter(),
                post_floor_new_items_total=0,
                post_floor_nonempty_turns_total=0,
                post_floor_turns_total=0,
                min_turns_for_productivity=1,
                early_stop_reason_distribution=Counter(),
                extract_parse_fail_log_entries=[],
                run_failure_counters=Counter(),
                eval_ids=["alpha"],
                manifest_hash="abc123",
                manifest_payload=manifest_payload,
                prior_manifest_info={"exists": False, "hash": None, "profile_count": 0, "read_error": None},
                seed=42,
                persona_count=2,
                processed_profiles=1,
                trace_level="off",
                max_api_calls=100,
                save_diagnostics=False,
                debug_outputs=False,
                run_profile="lean",
                requested_save_diagnostics=False,
                requested_trace_level="off",
                requested_debug_outputs=False,
                all_profiles=[],
            )

        self.assertFalse(integrity_payload["pass"])
        self.assertFalse(integrity_payload["results_alignment"]["pass"])
        self.assertEqual(integrity_payload["results_alignment"]["missing_in_results"], ["beta"])

    def test_integrity_prior_manifest_mismatch_is_informational_only(self) -> None:
        from app.eval_artifacts import write_eval_artifacts

        manifest_payload = {
            "run_config": {"manifest_schema_version": 4, "persona_count": 1, "seed": 42},
            "persona_count": 1,
            "profiles": [
                {
                    "persona_id": "alpha",
                    "split": "eval",
                    "family": "control_neutral",
                    "severity_tier": "minimal",
                    "subtype_tag": "routine_stable",
                    "context_tag": "routine_stable",
                    "style_tag": "open_but_flat",
                    "source": "synthetic",
                    "has_ground_truth": True,
                    "depressed": False,
                    "bdi_scores": {"1": 0},
                    "bdi_total": 0,
                    "key_symptoms": [],
                    "risk_signal": False,
                    "behavior_params": {},
                    "template_bank": "default",
                    "generation_seed": 42001,
                }
            ],
        }
        results = [
            PersonaResult(LLM="alpha", bdi_score=0, key_symptoms=[], item_scores={"1": 0}, item_support_counts={"1": 0})
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, integrity_payload, _, _, _ = write_eval_artifacts(
                output_dir=Path(tmp_dir),
                conversations=[],
                results=results,
                diagnostics_payload=[],
                overall_rows=[],
                route_distribution=Counter(),
                turns_total=0,
                evidence_turns_nonempty=0,
                evidence_records_total=0,
                extract_source_distribution=Counter(),
                extract_recovery_distribution=Counter(),
                route_policy_distribution=Counter(),
                duplicate_evidence_rows_total=0,
                contradiction_evidence_rows_total=0,
                support_increments_total=0,
                method_weight_usage=Counter(),
                post_floor_new_items_total=0,
                post_floor_nonempty_turns_total=0,
                post_floor_turns_total=0,
                min_turns_for_productivity=1,
                early_stop_reason_distribution=Counter(),
                extract_parse_fail_log_entries=[],
                run_failure_counters=Counter(),
                eval_ids=["alpha"],
                manifest_hash="newhash",
                manifest_payload=manifest_payload,
                prior_manifest_info={"exists": True, "hash": "oldhash", "profile_count": 1, "read_error": None},
                seed=42,
                persona_count=1,
                processed_profiles=1,
                trace_level="off",
                max_api_calls=100,
                save_diagnostics=False,
                debug_outputs=False,
                run_profile="lean",
                requested_save_diagnostics=False,
                requested_trace_level="off",
                requested_debug_outputs=False,
                all_profiles=[],
            )

        self.assertFalse(integrity_payload["prior_manifest"]["matches_current"])
        self.assertTrue(integrity_payload["pass"])

    def test_failure_report_excludes_non_failure_extractor_log_rows(self) -> None:
        from app.eval_artifacts import write_eval_artifacts

        manifest_payload = {
            "run_config": {"manifest_schema_version": 4, "persona_count": 1, "seed": 42},
            "persona_count": 1,
            "profiles": [
                {
                    "persona_id": "alpha",
                    "split": "eval",
                    "family": "control_neutral",
                    "severity_tier": "minimal",
                    "subtype_tag": "routine_stable",
                    "context_tag": "routine_stable",
                    "style_tag": "open_but_flat",
                    "source": "synthetic",
                    "has_ground_truth": True,
                    "depressed": False,
                    "bdi_scores": {"1": 0},
                    "bdi_total": 0,
                    "key_symptoms": [],
                    "risk_signal": False,
                    "behavior_params": {},
                    "template_bank": "default",
                    "generation_seed": 42001,
                }
            ],
        }
        results = [
            PersonaResult(LLM="alpha", bdi_score=0, key_symptoms=[], item_scores={"1": 0}, item_support_counts={"1": 0})
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, failure_report_payload, _, _, _, _ = write_eval_artifacts(
                output_dir=Path(tmp_dir),
                conversations=[],
                results=results,
                diagnostics_payload=[
                    {
                        "final_state": {
                            "sim_style_stats": {
                                "responses_total": 4,
                                "response_words_total": 88,
                                "qualifier_response_count": 2,
                                "hedged_response_count": 2,
                                "context_anchor_count": 1,
                                "mixed_answer_count": 1,
                                "soft_denial_count": 1,
                                "deflect_response_count": 0,
                                "baseline_comparison_count": 2,
                            }
                        }
                    }
                ],
                overall_rows=[
                    {
                        "family": "control_neutral",
                        "subtype_tag": "routine_stable",
                        "context_tag": "routine_stable",
                        "style_tag": "open_but_flat",
                        "bdi_true": 0,
                        "bdi_pred": 0,
                        "turns": 1,
                        "item_scores_true": {"1": 0},
                        "item_scores_pred": {"1": 0},
                    }
                ],
                route_distribution=Counter(),
                turns_total=1,
                evidence_turns_nonempty=0,
                evidence_records_total=0,
                extract_source_distribution=Counter(),
                extract_recovery_distribution=Counter(),
                route_policy_distribution=Counter(),
                duplicate_evidence_rows_total=0,
                contradiction_evidence_rows_total=0,
                support_increments_total=0,
                method_weight_usage=Counter(),
                post_floor_new_items_total=0,
                post_floor_nonempty_turns_total=0,
                post_floor_turns_total=0,
                min_turns_for_productivity=1,
                early_stop_reason_distribution=Counter(),
                extract_parse_fail_log_entries=[
                    {"failure_reason": "genuine_no_signal_all_unsupported", "counts_as_failure": False},
                    {"failure_reason": "scoped_empty_then_opportunistic_empty", "counts_as_failure": True},
                ],
                run_failure_counters=Counter(),
                eval_ids=["alpha"],
                manifest_hash="abc123",
                manifest_payload=manifest_payload,
                prior_manifest_info={"exists": False, "hash": None, "profile_count": 0, "read_error": None},
                seed=42,
                persona_count=1,
                processed_profiles=1,
                trace_level="compact",
                max_api_calls=100,
                save_diagnostics=True,
                debug_outputs=True,
                run_profile="debug",
                requested_save_diagnostics=True,
                requested_trace_level="compact",
                requested_debug_outputs=True,
                all_profiles=[],
            )

        self.assertEqual(int(failure_report_payload["extract_parse_fail_log_count"]), 1)
        self.assertEqual(int(failure_report_payload["extract_non_failure_log_count"]), 1)
        self.assertEqual(failure_report_payload["subtype_count"], {"routine_stable": 1})
        self.assertEqual(failure_report_payload["context_count"], {"routine_stable": 1})
        self.assertEqual(failure_report_payload["style_count"], {"open_but_flat": 1})
        self.assertEqual(int(failure_report_payload["sim_style_summary"]["responses_total"]), 4)
        self.assertAlmostEqual(float(failure_report_payload["sim_style_summary"]["avg_response_words"]), 22.0)


if __name__ == "__main__":
    unittest.main()
