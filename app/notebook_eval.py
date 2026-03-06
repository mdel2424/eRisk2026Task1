from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.state import symptom_name_from_item

RECORD_BASE_COLUMNS = [
    "persona_id",
    "split",
    "family",
    "source",
    "bdi_true",
    "bdi_pred",
    "key_symptoms_true",
    "key_symptoms_pred",
    "item_scores_true",
    "item_scores_pred",
]

ITEM_ERROR_COLUMNS = [
    "item_id",
    "symptom_name",
    "avg_pred",
    "avg_true",
    "mean_error",
    "abs_mean_error",
    "n_profiles",
]

PERSONA_ERROR_COLUMNS = [
    "persona_id",
    "split",
    "family",
    "source",
    "bdi_true",
    "bdi_pred",
    "bdi_error",
    "bdi_abs_error",
]


def _item_value_columns(prefix: str) -> List[str]:
    return [f"item_{item_id}_{prefix}" for item_id in range(1, 22)]


RECORD_COLUMNS = RECORD_BASE_COLUMNS + _item_value_columns("true") + _item_value_columns("pred")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_item_scores(raw_scores: Dict[Any, Any] | None) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    source = dict(raw_scores or {})
    for item_id in range(1, 22):
        raw_value = source.get(str(item_id), source.get(item_id, 0))
        try:
            value = int(raw_value or 0)
        except (TypeError, ValueError):
            value = 0
        normalized[str(item_id)] = max(0, min(3, value))
    return normalized


def _empty_records_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RECORD_COLUMNS)


def _empty_item_error_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ITEM_ERROR_COLUMNS)


def _scores_from_row(row: pd.Series, key: str) -> Dict[str, int]:
    scores = row.get(key, {})
    if isinstance(scores, dict):
        return _normalize_item_scores(scores)
    return _normalize_item_scores({})


def run_eval_notebook(
    *,
    persona_count: int = 10,
    seed: int = 42,
    eval_mode: str = "mixed_holdout",
    prompt_version: str = "v1",
    save_diagnostics: bool = False,
    max_api_calls: int = 500,
    trace_level: str = "off",
    fit_calibrator_policy: str = "auto",
    randomize_eval_split: bool = True,
    debug_outputs: bool = False,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    from app.cli_eval import run_eval

    resolved_output_dir = Path(output_dir)
    result = run_eval(
        persona_count=persona_count,
        seed=seed,
        eval_mode=eval_mode,
        prompt_version=prompt_version,
        save_diagnostics=save_diagnostics,
        max_api_calls=max_api_calls,
        trace_level=trace_level,
        fit_calibrator_policy=fit_calibrator_policy,
        randomize_eval_split=randomize_eval_split,
        debug_outputs=debug_outputs,
        output_dir=resolved_output_dir,
    )
    payload = dict(result)
    payload["output_dir"] = str(Path(result.get("output_dir", resolved_output_dir)).resolve())
    return payload


def load_eval_metrics(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    metrics_path = artifact_dir / "metrics_run_local.json"
    payload = _load_json(metrics_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metrics payload to be a JSON object: {metrics_path}")
    return payload


def load_eval_records(output_dir: str | Path) -> pd.DataFrame:
    artifact_dir = Path(output_dir)
    results_path = artifact_dir / "results_run_local.json"
    manifest_path = artifact_dir / "persona_manifest_run_local.json"

    raw_results = list(_load_json(results_path) or [])
    raw_manifest = dict(_load_json(manifest_path) or {})
    profiles = list(raw_manifest.get("profiles", []) or [])
    profiles_by_id = {str(profile.get("persona_id", "")).strip(): profile for profile in profiles}

    rows: List[Dict[str, Any]] = []
    missing_persona_ids: List[str] = []

    for result in raw_results:
        persona_id = str(result.get("LLM", "")).strip()
        if not persona_id:
            continue
        profile = profiles_by_id.get(persona_id)
        if profile is None:
            missing_persona_ids.append(persona_id)
            continue

        true_scores = _normalize_item_scores(profile.get("bdi_scores", {}))
        pred_scores = _normalize_item_scores(result.get("item-scores", {}))
        row: Dict[str, Any] = {
            "persona_id": persona_id,
            "split": str(profile.get("split", "")),
            "family": str(profile.get("family", "")),
            "source": str(profile.get("source", "")),
            "bdi_true": int(profile.get("bdi_total", sum(true_scores.values())) or 0),
            "bdi_pred": int(result.get("bdi-score", 0) or 0),
            "key_symptoms_true": list(profile.get("key_symptoms", []) or []),
            "key_symptoms_pred": list(result.get("key-symptoms", []) or []),
            "item_scores_true": true_scores,
            "item_scores_pred": pred_scores,
        }
        for item_id in range(1, 22):
            key = str(item_id)
            row[f"item_{item_id}_true"] = int(true_scores[key])
            row[f"item_{item_id}_pred"] = int(pred_scores[key])
        rows.append(row)

    if missing_persona_ids:
        missing = ", ".join(sorted(missing_persona_ids))
        raise KeyError(f"Prediction artifacts reference personas missing from manifest: {missing}")

    if not rows:
        return _empty_records_frame()
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def build_persona_error_table(records_df: pd.DataFrame | None) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=PERSONA_ERROR_COLUMNS)

    persona_df = records_df.copy()
    persona_df["bdi_error"] = pd.to_numeric(persona_df["bdi_pred"], errors="coerce").fillna(0.0) - pd.to_numeric(
        persona_df["bdi_true"], errors="coerce"
    ).fillna(0.0)
    persona_df["bdi_abs_error"] = persona_df["bdi_error"].abs()
    return (
        persona_df[PERSONA_ERROR_COLUMNS]
        .sort_values(["bdi_abs_error", "persona_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_item_error_table(records_df: pd.DataFrame | None) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        return _empty_item_error_frame()

    rows: List[Dict[str, Any]] = []
    n_profiles = int(len(records_df.index))

    for item_id in range(1, 22):
        pred_col = f"item_{item_id}_pred"
        true_col = f"item_{item_id}_true"

        if pred_col in records_df.columns and true_col in records_df.columns:
            pred_values = pd.to_numeric(records_df[pred_col], errors="coerce").fillna(0.0)
            true_values = pd.to_numeric(records_df[true_col], errors="coerce").fillna(0.0)
        else:
            pred_values = pd.Series(
                [float(_scores_from_row(row, "item_scores_pred").get(str(item_id), 0)) for _, row in records_df.iterrows()]
            )
            true_values = pd.Series(
                [float(_scores_from_row(row, "item_scores_true").get(str(item_id), 0)) for _, row in records_df.iterrows()]
            )

        avg_pred = float(pred_values.mean()) if not pred_values.empty else 0.0
        avg_true = float(true_values.mean()) if not true_values.empty else 0.0
        mean_error = avg_pred - avg_true
        rows.append(
            {
                "item_id": item_id,
                "symptom_name": symptom_name_from_item(item_id),
                "avg_pred": round(avg_pred, 4),
                "avg_true": round(avg_true, 4),
                "mean_error": round(mean_error, 4),
                "abs_mean_error": round(abs(mean_error), 4),
                "n_profiles": n_profiles,
            }
        )

    return pd.DataFrame(rows, columns=ITEM_ERROR_COLUMNS)


def split_item_error_table(item_error_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if item_error_df.empty:
        empty = _empty_item_error_frame()
        return {"all_items": empty, "under_predicted": empty.copy(), "over_predicted": empty.copy()}

    all_items = item_error_df.sort_values(["mean_error", "item_id"], ascending=[True, True]).reset_index(drop=True)
    under_predicted = (
        item_error_df[item_error_df["mean_error"] < 0]
        .sort_values(["mean_error", "item_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    over_predicted = (
        item_error_df[item_error_df["mean_error"] > 0]
        .sort_values(["mean_error", "item_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {
        "all_items": all_items,
        "under_predicted": under_predicted,
        "over_predicted": over_predicted,
    }


__all__ = [
    "ITEM_ERROR_COLUMNS",
    "PERSONA_ERROR_COLUMNS",
    "build_persona_error_table",
    "RECORD_COLUMNS",
    "build_item_error_table",
    "load_eval_records",
    "load_eval_metrics",
    "run_eval_notebook",
    "split_item_error_table",
]
