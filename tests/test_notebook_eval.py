from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.notebook_eval import (
    ITEM_ERROR_COLUMNS,
    NOTEBOOK_STABILITY_PERSONA_THRESHOLD,
    PERSONA_ERROR_COLUMNS,
    build_compact_diagnostics_summary,
    build_eval_stability_notice,
    build_item_error_table,
    build_persona_error_table,
    compare_style_summaries,
    inspect_artifact_run_consistency,
    load_eval_records,
    resolve_runtime_artifact_metadata,
    summarize_simulated_style,
    summarize_transcript_style,
)
from core.io_schema import PersonaResult
from core.state import symptom_name_from_item


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class NotebookEvalTests(unittest.TestCase):
    def test_persona_result_serializes_finalizer_summary(self) -> None:
        payload = PersonaResult(
            LLM="alpha",
            bdi_score=4,
            key_symptoms=["Worthlessness"],
            item_scores={"1": 1},
            item_support_counts={"1": 2},
            finalizer_summary={
                "low_signal_guardrail_active": True,
                "guardrail_bypass_source": "none",
                "severe_recovery_mode_active": False,
                "severe_amplitude_observed_item_ids": [14],
                "severe_item9_rescued": False,
            },
        ).to_erisk_dict()

        self.assertIn("finalizer_summary", payload)
        self.assertTrue(bool(payload["finalizer_summary"]["low_signal_guardrail_active"]))

    def test_load_eval_records_joins_prediction_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "persona_manifest_run_local.json",
                {
                    "profiles": [
                        {
                            "persona_id": "alpha",
                            "split": "val",
                            "family": "cognitive_ruminative",
                            "severity_tier": "moderate",
                            "subtype_tag": "worthlessness_heavy",
                            "context_tag": "school",
                            "style_tag": "contextual_reflective",
                            "source": "synthetic",
                            "bdi_scores": {"1": 0, "2": 1},
                            "bdi_total": 1,
                            "key_symptoms": ["Pessimism"],
                        },
                        {
                            "persona_id": "beta",
                            "split": "test",
                            "family": "somatic_fatigue",
                            "severity_tier": "mild",
                            "subtype_tag": "sleep_heavy",
                            "context_tag": "workload",
                            "style_tag": "minimizing_practical",
                            "source": "synthetic",
                            "bdi_scores": {"1": 1, "2": 0},
                            "bdi_total": 1,
                            "key_symptoms": ["Sadness"],
                        },
                    ]
                },
            )
            _write_json(
                root / "results_run_local.json",
                [
                    {
                        "LLM": "alpha",
                        "bdi-score": 2,
                        "key-symptoms": ["Pessimism"],
                        "item-scores": {"1": 1, "2": 2},
                        "finalizer_summary": {
                            "low_signal_guardrail_active": True,
                            "support_geometry_candidate_bypass": False,
                            "anchor_gated_guardrail_blocked": False,
                            "guardrail_bypass_source": "none",
                            "severe_recovery_mode_active": False,
                            "severe_recovery_reason": "",
                            "severe_amplitude_observed_item_ids": [14],
                            "severe_amplitude_imputed_item_ids": [11],
                            "severe_item9_rescued": True,
                        },
                    },
                    {
                        "LLM": "beta",
                        "bdi-score": 0,
                        "key-symptoms": [],
                        "item-scores": {"1": 0, "2": 0},
                    },
                ],
            )

            records_df = load_eval_records(root)

        self.assertEqual(len(records_df), 2)
        self.assertEqual(records_df.loc[0, "persona_id"], "alpha")
        self.assertEqual(records_df.loc[0, "split"], "val")
        self.assertEqual(records_df.loc[0, "family"], "cognitive_ruminative")
        self.assertEqual(records_df.loc[0, "severity_tier"], "moderate")
        self.assertEqual(records_df.loc[0, "subtype_tag"], "worthlessness_heavy")
        self.assertEqual(records_df.loc[0, "context_tag"], "school")
        self.assertEqual(records_df.loc[0, "style_tag"], "contextual_reflective")
        self.assertEqual(records_df.loc[0, "bdi_true"], 1)
        self.assertEqual(records_df.loc[0, "bdi_pred"], 2)
        self.assertEqual(records_df.loc[0, "item_1_true"], 0)
        self.assertEqual(records_df.loc[0, "item_1_pred"], 1)
        self.assertEqual(records_df.loc[1, "item_2_true"], 0)
        self.assertEqual(records_df.loc[1, "item_2_pred"], 0)
        self.assertEqual(records_df.loc[0, "item_scores_true"]["2"], 1)
        self.assertEqual(records_df.loc[0, "item_scores_pred"]["2"], 2)
        self.assertTrue(bool(records_df.loc[0, "finalizer_low_signal_guardrail_active"]))
        self.assertFalse(bool(records_df.loc[0, "finalizer_support_geometry_candidate_bypass"]))
        self.assertFalse(bool(records_df.loc[0, "finalizer_anchor_gated_guardrail_blocked"]))
        self.assertEqual(records_df.loc[0, "finalizer_guardrail_bypass_source"], "none")
        self.assertFalse(bool(records_df.loc[0, "finalizer_severe_recovery_mode_active"]))
        self.assertEqual(records_df.loc[0, "finalizer_severe_amplitude_observed_item_ids"], [14])
        self.assertEqual(records_df.loc[0, "finalizer_severe_amplitude_imputed_item_ids"], [11])
        self.assertTrue(bool(records_df.loc[0, "finalizer_severe_item9_rescued"]))
        self.assertTrue(bool(records_df.loc[0, "finalizer_guardrail_consistency_ok"]))

    def test_load_eval_records_handles_missing_finalizer_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "persona_manifest_run_local.json",
                {
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
                            "bdi_scores": {"1": 0},
                            "bdi_total": 0,
                            "key_symptoms": [],
                        }
                    ]
                },
            )
            _write_json(
                root / "results_run_local.json",
                [
                    {
                        "LLM": "alpha",
                        "bdi-score": 0,
                        "key-symptoms": [],
                        "item-scores": {"1": 0},
                    }
                ],
            )

            records_df = load_eval_records(root)

        self.assertIn("finalizer_summary", records_df.columns)
        self.assertEqual(records_df.loc[0, "finalizer_summary"], {})
        self.assertFalse(bool(records_df.loc[0, "finalizer_severe_recovery_mode_active"]))
        self.assertEqual(records_df.loc[0, "finalizer_severe_amplitude_observed_item_ids"], [])
        self.assertEqual(records_df.loc[0, "finalizer_severe_amplitude_imputed_item_ids"], [])
        self.assertFalse(bool(records_df.loc[0, "finalizer_severe_item9_rescued"]))
        self.assertTrue(bool(records_df.loc[0, "finalizer_guardrail_consistency_ok"]))

    def test_build_item_error_table_computes_expected_means(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "item_scores_true": {"1": 0, "2": 1},
                    "item_scores_pred": {"1": 1, "2": 2},
                },
                {
                    "item_scores_true": {"1": 1, "2": 0},
                    "item_scores_pred": {"1": 0, "2": 1},
                },
            ]
        )

        item_error_df = build_item_error_table(records_df)

        self.assertEqual(len(item_error_df), 21)
        self.assertEqual(list(item_error_df.columns), ITEM_ERROR_COLUMNS)

        item_1 = item_error_df[item_error_df["item_id"] == 1].iloc[0]
        self.assertEqual(item_1["symptom_name"], symptom_name_from_item(1))
        self.assertAlmostEqual(float(item_1["avg_pred"]), 0.5)
        self.assertAlmostEqual(float(item_1["avg_true"]), 0.5)
        self.assertAlmostEqual(float(item_1["mean_error"]), 0.0)
        self.assertAlmostEqual(float(item_1["abs_mean_error"]), 0.0)
        self.assertEqual(int(item_1["n_profiles"]), 2)

        item_2 = item_error_df[item_error_df["item_id"] == 2].iloc[0]
        self.assertEqual(item_2["symptom_name"], symptom_name_from_item(2))
        self.assertAlmostEqual(float(item_2["avg_pred"]), 1.5)
        self.assertAlmostEqual(float(item_2["avg_true"]), 0.5)
        self.assertAlmostEqual(float(item_2["mean_error"]), 1.0)
        self.assertAlmostEqual(float(item_2["abs_mean_error"]), 1.0)

    def test_build_item_error_table_handles_empty_input(self) -> None:
        item_error_df = build_item_error_table(pd.DataFrame())

        self.assertTrue(item_error_df.empty)
        self.assertEqual(list(item_error_df.columns), ITEM_ERROR_COLUMNS)

    def test_summarize_transcript_style_extracts_patient_lines(self) -> None:
        transcript = """
Patient: I think it has been more up and down than usual lately.
Patient: That part feels pretty close to normal for me.
Patient: It is a bit of both, honestly; getting started takes effort and I get less out of it.
"""

        summary = summarize_transcript_style(transcript)

        self.assertEqual(summary["response_count"], 3)
        self.assertGreater(float(summary["qualifier_rate"]), 0.0)
        self.assertGreater(float(summary["mixed_answer_rate"]), 0.0)
        self.assertGreaterEqual(float(summary["soft_denial_rate"]), 0.0)

    def test_summarize_simulated_style_returns_expected_keys(self) -> None:
        summary = summarize_simulated_style(persona_count=4, seed=42)

        self.assertEqual(summary["persona_count"], 4)
        self.assertGreater(int(summary["response_count"]), 0)
        self.assertIn("avg_response_words", summary)
        self.assertIn("qualifier_rate", summary)
        self.assertIn("mixed_answer_rate", summary)
        self.assertIn("soft_denial_rate", summary)

    def test_summarize_simulated_style_falls_within_calibration_envelope(self) -> None:
        summary = summarize_simulated_style(persona_count=8, seed=42)

        self.assertGreaterEqual(float(summary["avg_response_words"]), 20.0)
        self.assertLessEqual(float(summary["avg_response_words"]), 24.0)
        self.assertGreaterEqual(float(summary["qualifier_rate"]), 0.40)
        self.assertLessEqual(float(summary["qualifier_rate"]), 0.60)
        self.assertGreaterEqual(float(summary["baseline_comparison_rate"]), 0.06)
        self.assertLessEqual(float(summary["baseline_comparison_rate"]), 0.22)
        self.assertLess(float(summary["soft_denial_rate"]), 0.40)
        self.assertGreaterEqual(float(summary["mixed_answer_rate"]), 0.04)

    def test_build_eval_stability_notice_warns_on_small_persona_count(self) -> None:
        notice = build_eval_stability_notice({"persona_count": 4})

        self.assertTrue(bool(notice["is_low_stability"]))
        self.assertEqual(int(notice["stable_threshold"]), NOTEBOOK_STABILITY_PERSONA_THRESHOLD)
        self.assertIn("Low-stability read", str(notice["message"]))

    def test_runtime_metadata_comes_from_artifacts_not_raw_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "config_used.json",
                {
                    "resolved_backends": {
                        "detector_backend": "openrouter",
                        "detector_target": "artifact/model",
                        "persona_runtime": "deterministic_simulator",
                    }
                },
            )
            _write_json(
                root / "benchmark_integrity_run_local.json",
                {
                    "detector": {
                        "backend": "ollama",
                        "target": "stale/notebook-guess",
                    }
                },
            )

            metadata = resolve_runtime_artifact_metadata(root)

        self.assertEqual(metadata["detector_backend"], "openrouter")
        self.assertEqual(metadata["detector_target"], "artifact/model")
        self.assertEqual(metadata["source"], "config_used")

    def test_compact_diagnostics_summary_highlights_module3_target_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "failure_report_run_local.json",
                {
                    "artifact_run_id": "run-a",
                    "generated_at": "2026-04-20T12:00:00+00:00",
                    "extract_empty_rate": 0.5,
                    "evidence_nonempty_rate": 0.4,
                    "extract_failure_log_count": 3,
                    "extract_true_parse_fail_log_count": 0,
                    "extract_runtime_error_log_count": 1,
                    "extract_failure_kind_distribution": {
                        "extractor_call_failure": 1,
                        "opportunistic_no_candidate": 2,
                    },
                    "extract_failure_reason_distribution": {
                        "opportunistic_shortlist_no_candidates": 2,
                        "llm_call_failed": 1,
                    },
                    "extract_source_distribution": {
                        "module3_scoped_recovery": 1,
                        "llm_extractor_error": 1,
                    },
                },
            )
            _write_json(
                root / "diagnostics_run_local.json",
                [
                    {
                        "LLM": "alpha",
                        "artifact_run_id": "run-a",
                        "generated_at": "2026-04-20T12:00:00+00:00",
                        "timeline": [
                            {
                                "turn": 1,
                                "route_decision": {"target_items": [7]},
                                "turn_trace": {
                                    "extract_likelihoods": {
                                        "source": "module3_scoped_recovery",
                                        "detail_module3_scoped_recovery_trigger": "llm_extractor_error",
                                    },
                                    "belief_update": {"updated_item_ids": [5]},
                                },
                            }
                        ],
                    }
                ],
            )

            diagnostics = build_compact_diagnostics_summary(
                root,
                item_error_df=pd.DataFrame(
                    [
                        {"item_id": 7, "symptom_name": "Self-Dislike", "mean_error": -1.5},
                        {"item_id": 8, "symptom_name": "Self-Criticalness", "mean_error": -1.2},
                        {"item_id": 14, "symptom_name": "Worthlessness", "mean_error": -1.0},
                    ]
                ),
            )

        self.assertEqual(diagnostics["module3_target_coverage"], {7: 1, 8: 0, 14: 0})
        self.assertEqual(int(diagnostics["llm_extractor_error_turns"]), 1)
        self.assertEqual(int(diagnostics["runtime_error_count"]), 1)
        self.assertEqual(int(diagnostics["opportunistic_no_candidate_count"]), 2)
        self.assertIn("collapsed onto item 7", str(diagnostics["coverage_message"]))
        self.assertIn("opportunistic shortlist no-candidate", str(diagnostics["dominant_issue"]))

    def test_inspect_artifact_run_consistency_flags_mixed_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "failure_report_run_local.json",
                {"artifact_run_id": "run-a", "generated_at": "2026-04-20T12:00:00+00:00"},
            )
            _write_json(
                root / "config_used.json",
                {"artifact_run_id": "run-b", "generated_at": "2026-04-20T12:05:00+00:00"},
            )

            summary = inspect_artifact_run_consistency(root)

        self.assertTrue(bool(summary["mixed_run_detected"]))
        self.assertEqual(sorted(summary["unique_run_ids"]), ["run-a", "run-b"])
        self.assertIn("Mixed-run artifact set detected", str(summary["warning"]))

    def test_compare_style_summaries_returns_deltas(self) -> None:
        reference = {"response_count": 8, "avg_response_words": 20.0, "qualifier_rate": 0.4, "hedge_rate": 0.4, "context_anchor_rate": 0.3, "mixed_answer_rate": 0.2, "soft_denial_rate": 0.1, "baseline_comparison_rate": 0.25}
        simulated = {"response_count": 10, "avg_response_words": 24.0, "qualifier_rate": 0.5, "hedge_rate": 0.5, "context_anchor_rate": 0.4, "mixed_answer_rate": 0.35, "soft_denial_rate": 0.05, "baseline_comparison_rate": 0.3}

        comparison = compare_style_summaries(reference, simulated)

        self.assertEqual(comparison["reference_response_count"], 8)
        self.assertEqual(comparison["simulated_response_count"], 10)
        self.assertAlmostEqual(float(comparison["delta_avg_response_words"]), 4.0)
        self.assertAlmostEqual(float(comparison["delta_mixed_answer_rate"]), 0.15)

    def test_build_persona_error_table_sorts_by_absolute_bdi_error(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "persona_id": "a",
                    "split": "val",
                    "family": "alpha",
                    "source": "synthetic",
                    "bdi_true": 12,
                    "bdi_pred": 20,
                },
                {
                    "persona_id": "b",
                    "split": "test",
                    "family": "beta",
                    "source": "synthetic",
                    "bdi_true": 10,
                    "bdi_pred": 8,
                },
            ]
        )

        persona_error_df = build_persona_error_table(records_df)

        self.assertEqual(list(persona_error_df.columns), PERSONA_ERROR_COLUMNS)
        self.assertEqual(persona_error_df.loc[0, "persona_id"], "a")
        self.assertAlmostEqual(float(persona_error_df.loc[0, "bdi_error"]), 8.0)
        self.assertAlmostEqual(float(persona_error_df.loc[0, "bdi_abs_error"]), 8.0)
        self.assertEqual(persona_error_df.loc[1, "persona_id"], "b")
        self.assertAlmostEqual(float(persona_error_df.loc[1, "bdi_error"]), -2.0)
        self.assertAlmostEqual(float(persona_error_df.loc[1, "bdi_abs_error"]), 2.0)


if __name__ == "__main__":
    unittest.main()
