from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.notebook_eval import (
    ITEM_ERROR_COLUMNS,
    OPENING_SIGNAL_COLUMNS,
    PERSONA_ERROR_COLUMNS,
    RUNTIME_ANOMALY_COLUMNS,
    RUNTIME_SUPPORT_GAP_COLUMNS,
    build_item_error_table,
    build_cluster_collapse_table,
    build_opening_signal_table,
    build_persona_error_table,
    build_runtime_anomaly_table,
    build_runtime_support_gap_table,
    compare_style_summaries,
    load_eval_records,
    summarize_simulated_style,
    summarize_transcript_style,
)
from core.io_schema import PersonaResult
from core.state import symptom_name_from_item


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class NotebookEvalTests(unittest.TestCase):
    def test_persona_result_serializes_runtime_summary(self) -> None:
        payload = PersonaResult(
            LLM="alpha",
            bdi_score=4,
            key_symptoms=["Worthlessness"],
            item_scores={"1": 1},
            item_support_counts={"1": 2},
            runtime_summary={
                "diagnosis": {"confidence": 0.73, "used_llm": False, "synthesis_mode": "deterministic"},
                "judgment": {"active_cluster": "cognitive_affective", "evidence_binding_coverage": 1.0},
                "navigation": {"opening_followup_cluster": "cognitive_affective"},
                "bayes": {"node_posteriors": {"cognitive_affective": 0.62}},
            },
        ).to_erisk_dict()

        self.assertIn("runtime_summary", payload)
        self.assertAlmostEqual(float(payload["runtime_summary"]["diagnosis"]["confidence"]), 0.73)

    def test_load_eval_records_joins_prediction_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "persona_manifest_run_local.json",
                {
                    "profiles": [
                        {
                            "persona_id": "alpha",
                            "split": "eval",
                            "family": "cognitive_ruminative",
                            "severity_tier": "moderate",
                            "subtype_tag": "worthlessness_heavy",
                            "context_tag": "school",
                            "style_tag": "contextual_reflective",
                            "source": "synthetic",
                            "bdi_scores": {"1": 0, "2": 1},
                            "bdi_total": 1,
                            "key_symptoms": ["Pessimism"],
                        }
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
                        "runtime_summary": {
                            "diagnosis": {
                                "confidence": 0.68,
                                "used_llm": False,
                                "synthesis_mode": "deterministic",
                                "supported_item_count": 2,
                            },
                            "judgment": {
                                "active_cluster": "cognitive_affective",
                                "evidence_binding_coverage": 1.0,
                                "bound_positive_assertion_count": 2,
                                "emitted_evidence_count": 2,
                                "opening_bootstrap_applied": True,
                                "opening_bootstrap_cluster": "cognitive_affective",
                                "opening_bootstrap_item_ids": [8, 2],
                            },
                            "navigation": {
                                "opening_signal_cluster": "cognitive_affective",
                                "opening_signal_item_ids": [8, 2],
                                "opening_followup_cluster": "cognitive_affective",
                                "opening_cognitive_anchor_preserved": True,
                            },
                            "bayes": {"node_posteriors": {"cognitive_affective": 0.71}},
                        },
                    }
                ],
            )

            records_df = load_eval_records(root)

        self.assertEqual(len(records_df), 1)
        self.assertEqual(records_df.loc[0, "persona_id"], "alpha")
        self.assertEqual(records_df.loc[0, "family"], "cognitive_ruminative")
        self.assertEqual(records_df.loc[0, "bdi_true"], 1)
        self.assertEqual(records_df.loc[0, "bdi_pred"], 2)
        self.assertEqual(records_df.loc[0, "item_1_true"], 0)
        self.assertEqual(records_df.loc[0, "item_1_pred"], 1)
        self.assertEqual(records_df.loc[0, "runtime_active_cluster"], "cognitive_affective")
        self.assertAlmostEqual(float(records_df.loc[0, "runtime_diagnosis_confidence"]), 0.68)
        self.assertTrue(bool(records_df.loc[0, "runtime_opening_bootstrap_applied"]))
        self.assertEqual(records_df.loc[0, "runtime_opening_followup_cluster"], "cognitive_affective")
        self.assertTrue(bool(records_df.loc[0, "runtime_bayes_node_posteriors"]["cognitive_affective"] > 0.7))

    def test_load_eval_records_handles_missing_runtime_summary(self) -> None:
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
                [{"LLM": "alpha", "bdi-score": 0, "key-symptoms": [], "item-scores": {"1": 0}}],
            )

            records_df = load_eval_records(root)

        self.assertIn("runtime_summary", records_df.columns)
        self.assertEqual(records_df.loc[0, "runtime_summary"], {})
        self.assertEqual(records_df.loc[0, "runtime_active_cluster"], "")
        self.assertEqual(float(records_df.loc[0, "runtime_evidence_binding_coverage"]), 1.0)
        self.assertFalse(bool(records_df.loc[0, "runtime_opening_bootstrap_applied"]))

    def test_build_item_error_table_computes_expected_means(self) -> None:
        records_df = pd.DataFrame(
            [
                {"item_scores_true": {"1": 0, "2": 1}, "item_scores_pred": {"1": 1, "2": 2}},
                {"item_scores_true": {"1": 1, "2": 0}, "item_scores_pred": {"1": 0, "2": 1}},
            ]
        )

        item_error_df = build_item_error_table(records_df)

        self.assertEqual(len(item_error_df), 21)
        self.assertEqual(list(item_error_df.columns), ITEM_ERROR_COLUMNS)

        item_1 = item_error_df[item_error_df["item_id"] == 1].iloc[0]
        self.assertEqual(item_1["symptom_name"], symptom_name_from_item(1))
        self.assertAlmostEqual(float(item_1["avg_pred"]), 0.5)
        self.assertAlmostEqual(float(item_1["avg_true"]), 0.5)

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

    def test_summarize_simulated_style_returns_expected_keys(self) -> None:
        summary = summarize_simulated_style(persona_count=4, seed=42)

        self.assertEqual(summary["persona_count"], 4)
        self.assertGreater(int(summary["response_count"]), 0)
        self.assertIn("avg_response_words", summary)
        self.assertIn("qualifier_rate", summary)
        self.assertIn("mixed_answer_rate", summary)
        self.assertIn("soft_denial_rate", summary)

    def test_compare_style_summaries_returns_deltas(self) -> None:
        comparison = compare_style_summaries(
            {"response_count": 10, "avg_response_words": 15.0, "qualifier_rate": 0.2},
            {"response_count": 12, "avg_response_words": 18.0, "qualifier_rate": 0.35},
        )

        self.assertEqual(comparison["reference_response_count"], 10)
        self.assertEqual(comparison["simulated_response_count"], 12)
        self.assertAlmostEqual(float(comparison["delta_avg_response_words"]), 3.0)
        self.assertAlmostEqual(float(comparison["delta_qualifier_rate"]), 0.15)

    def test_build_persona_error_table_sorts_descending_abs_error(self) -> None:
        records_df = pd.DataFrame(
            [
                {"persona_id": "1", "split": "eval", "family": "a", "source": "synthetic", "bdi_true": 2, "bdi_pred": 7},
                {"persona_id": "2", "split": "eval", "family": "b", "source": "synthetic", "bdi_true": 4, "bdi_pred": 5},
            ]
        )

        persona_df = build_persona_error_table(records_df)
        self.assertEqual(list(persona_df.columns), PERSONA_ERROR_COLUMNS)
        self.assertEqual(list(persona_df["persona_id"]), ["1", "2"])

    def test_build_runtime_anomaly_table_flags_cluster_collapse_cases(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "persona_id": "1",
                    "family": "risk_leaning",
                    "source": "synthetic",
                    "bdi_true": 28,
                    "bdi_pred": 11,
                    "runtime_active_cluster": "somatic_vegetative",
                    "runtime_evidence_binding_coverage": 1.0,
                    "runtime_diagnosis_confidence": 0.72,
                    "runtime_supported_item_count": 1,
                    "runtime_bayes_node_posteriors": {
                        "somatic_vegetative": 1.0,
                        "cognitive_affective": 0.22,
                    },
                    "item_scores_pred": {"15": 2, "16": 2, "18": 2, "20": 2, "21": 1},
                },
                {
                    "persona_id": "2",
                    "family": "control_stressed",
                    "source": "synthetic",
                    "bdi_true": 4,
                    "bdi_pred": 5,
                    "runtime_active_cluster": "cognitive_affective",
                    "runtime_evidence_binding_coverage": 1.0,
                    "runtime_diagnosis_confidence": 0.40,
                    "runtime_supported_item_count": 3,
                    "runtime_bayes_node_posteriors": {
                        "somatic_vegetative": 0.25,
                        "cognitive_affective": 0.41,
                    },
                    "item_scores_pred": {"2": 1, "5": 1, "19": 1},
                },
            ]
        )

        anomaly_df = build_runtime_anomaly_table(records_df)
        collapse_df = build_cluster_collapse_table(records_df)

        self.assertEqual(list(anomaly_df.columns), RUNTIME_ANOMALY_COLUMNS)
        self.assertEqual(list(anomaly_df["persona_id"]), ["1"])
        self.assertEqual(list(collapse_df["persona_id"]), ["1"])

    def test_build_runtime_support_gap_table_flags_low_support_high_error_cases(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "persona_id": "1",
                    "family": "risk_leaning",
                    "source": "synthetic",
                    "bdi_true": 28,
                    "bdi_pred": 9,
                    "runtime_supported_item_count": 1,
                    "runtime_diagnosis_confidence": 0.66,
                    "runtime_active_cluster": "somatic_vegetative",
                },
                {
                    "persona_id": "2",
                    "family": "control_stressed",
                    "source": "synthetic",
                    "bdi_true": 4,
                    "bdi_pred": 5,
                    "runtime_supported_item_count": 3,
                    "runtime_diagnosis_confidence": 0.44,
                    "runtime_active_cluster": "cognitive_affective",
                },
            ]
        )

        support_gap_df = build_runtime_support_gap_table(records_df)

        self.assertEqual(list(support_gap_df.columns), RUNTIME_SUPPORT_GAP_COLUMNS)
        self.assertEqual(list(support_gap_df["persona_id"]), ["1"])

    def test_build_opening_signal_table_lists_bootstrap_personas(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "persona_id": "1",
                    "family": "cognitive_ruminative",
                    "source": "synthetic",
                    "runtime_opening_bootstrap_applied": True,
                    "runtime_opening_bootstrap_cluster": "cognitive_affective",
                    "runtime_opening_bootstrap_item_ids": [8, 2],
                    "runtime_opening_followup_cluster": "cognitive_affective",
                    "runtime_opening_cognitive_anchor_preserved": True,
                    "runtime_supported_item_count": 4,
                    "bdi_true": 21,
                    "bdi_pred": 18,
                },
                {
                    "persona_id": "2",
                    "family": "somatic_evasive",
                    "source": "synthetic",
                    "runtime_opening_bootstrap_applied": False,
                    "runtime_opening_bootstrap_cluster": "",
                    "runtime_opening_bootstrap_item_ids": [],
                    "runtime_opening_followup_cluster": "",
                    "runtime_opening_cognitive_anchor_preserved": False,
                    "runtime_supported_item_count": 2,
                    "bdi_true": 32,
                    "bdi_pred": 4,
                },
            ]
        )

        opening_df = build_opening_signal_table(records_df)

        self.assertEqual(list(opening_df.columns), OPENING_SIGNAL_COLUMNS)
        self.assertEqual(list(opening_df["persona_id"]), ["1"])

    def test_eval_notebook_no_longer_references_finalizer_columns(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "eval_item_error_analysis.ipynb"
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
        self.assertNotIn("finalizer_guardrail_consistency_ok", joined)
        self.assertNotIn("finalizer_low_signal_guardrail_active", joined)


if __name__ == "__main__":
    unittest.main()
