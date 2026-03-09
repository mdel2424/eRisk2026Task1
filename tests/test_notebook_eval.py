from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.notebook_eval import (
    ITEM_ERROR_COLUMNS,
    PERSONA_ERROR_COLUMNS,
    build_item_error_table,
    build_persona_error_table,
    load_eval_records,
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
                            "source": "synthetic",
                            "bdi_scores": {"1": 0, "2": 1},
                            "bdi_total": 1,
                            "key_symptoms": ["Pessimism"],
                        },
                        {
                            "persona_id": "beta",
                            "split": "test",
                            "family": "somatic_fatigue",
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
