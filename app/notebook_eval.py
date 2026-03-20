from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.finalizer_summary import (
    FINALIZER_GUARDRAIL_FIELDS,
    FINALIZER_SEVERE_AMPLITUDE_FIELDS,
    FINALIZER_SEVERE_RECOVERY_FIELDS,
    default_finalizer_summary_value,
)
from core.state import symptom_name_from_item
from persona import create_persona, generate_persona_pool
from persona.sim_behavior import response_style_flags

FINALIZER_GROUPED_COLUMNS = (
    [f"finalizer_{field}" for field in FINALIZER_GUARDRAIL_FIELDS]
    + [f"finalizer_{field}" for field in FINALIZER_SEVERE_RECOVERY_FIELDS]
    + [f"finalizer_{field}" for field in FINALIZER_SEVERE_AMPLITUDE_FIELDS]
)

RECORD_BASE_COLUMNS = [
    "persona_id",
    "split",
    "family",
    "severity_tier",
    "subtype_tag",
    "context_tag",
    "style_tag",
    "source",
    "bdi_true",
    "bdi_pred",
    "key_symptoms_true",
    "key_symptoms_pred",
    "item_scores_true",
    "item_scores_pred",
    "finalizer_summary",
    *FINALIZER_GROUPED_COLUMNS,
    "finalizer_guardrail_consistency_ok",
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

SIM_STYLE_CALIBRATION_PROBES: List[Dict[str, object]] = [
    {"target_item_id": 1, "route": "cognitive", "style": "gentle_probe", "mode": "normal", "directness": "indirect", "priority": 0.5},
    {"target_item_id": 4, "route": "cognitive", "style": "functional_impact", "mode": "normal", "directness": "indirect", "priority": 0.6},
    {"target_item_id": 14, "route": "cognitive", "style": "gentle_probe", "mode": "normal", "directness": "direct", "priority": 0.75},
    {"target_item_id": 15, "route": "somatic", "style": "functional_impact", "mode": "normal", "directness": "indirect", "priority": 0.6},
    {"target_item_id": 16, "route": "somatic", "style": "clarify_frequency", "mode": "normal", "directness": "direct", "priority": 0.7},
    {"target_item_id": 18, "route": "somatic", "style": "gentle_probe", "mode": "normal", "directness": "direct", "priority": 0.7},
    {"target_item_id": 21, "route": "somatic", "style": "gentle_probe", "mode": "normal", "directness": "direct", "priority": 0.7},
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
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("score", 0)
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


def _style_summary_from_responses(responses: List[str]) -> Dict[str, Any]:
    cleaned = [str(response or "").strip() for response in responses if str(response or "").strip()]
    response_count = len(cleaned)
    if response_count <= 0:
        return {
            "response_count": 0,
            "avg_response_words": 0.0,
            "qualifier_rate": 0.0,
            "hedge_rate": 0.0,
            "context_anchor_rate": 0.0,
            "mixed_answer_rate": 0.0,
            "soft_denial_rate": 0.0,
            "baseline_comparison_rate": 0.0,
        }

    word_counts = [len(response.split()) for response in cleaned]
    flags = [response_style_flags(response) for response in cleaned]

    def _rate(key: str) -> float:
        return round(sum(1 for flag in flags if bool(flag.get(key))) / float(response_count), 4)

    return {
        "response_count": response_count,
        "avg_response_words": round(sum(word_counts) / float(response_count), 4),
        "qualifier_rate": _rate("qualifier"),
        "hedge_rate": _rate("hedged"),
        "context_anchor_rate": _rate("context_anchor"),
        "mixed_answer_rate": _rate("mixed_answer"),
        "soft_denial_rate": _rate("soft_denial"),
        "baseline_comparison_rate": _rate("baseline_comparison"),
    }


def _scores_from_row(row: pd.Series, key: str) -> Dict[str, int]:
    scores = row.get(key, {})
    if isinstance(scores, dict):
        return _normalize_item_scores(scores)
    return _normalize_item_scores({})


def _flatten_finalizer_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for field in FINALIZER_GUARDRAIL_FIELDS + FINALIZER_SEVERE_RECOVERY_FIELDS + FINALIZER_SEVERE_AMPLITUDE_FIELDS:
        default_value = default_finalizer_summary_value(field)
        raw_value = summary.get(field, default_value)
        if isinstance(default_value, bool):
            value = bool(raw_value)
        elif isinstance(default_value, int):
            value = int(raw_value or 0)
        elif isinstance(default_value, list):
            value = list(raw_value or [])
        else:
            value = str(raw_value or "")
        flattened[f"finalizer_{field}"] = value
    return flattened


def run_eval_notebook(
    *,
    persona_count: int = 10,
    seed: int = 42,
    save_diagnostics: bool = False,
    max_api_calls: int = 500,
    trace_level: str = "off",
    debug_outputs: bool = False,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    from app.cli_eval import run_eval

    resolved_output_dir = Path(output_dir)
    result = run_eval(
        persona_count=persona_count,
        seed=seed,
        save_diagnostics=save_diagnostics,
        max_api_calls=max_api_calls,
        trace_level=trace_level,
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


def load_benchmark_integrity(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    integrity_path = artifact_dir / "benchmark_integrity_run_local.json"
    payload = _load_json(integrity_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected integrity payload to be a JSON object: {integrity_path}")
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
        finalizer_summary = dict(result.get("finalizer_summary", {}) or {})
        row: Dict[str, Any] = {
            "persona_id": persona_id,
            "split": str(profile.get("split", "")),
            "family": str(profile.get("family", "")),
            "severity_tier": str(profile.get("severity_tier", "")),
            "subtype_tag": str(profile.get("subtype_tag", "")),
            "context_tag": str(profile.get("context_tag", "")),
            "style_tag": str(profile.get("style_tag", "")),
            "source": str(profile.get("source", "")),
            "bdi_true": int(profile.get("bdi_total", sum(true_scores.values())) or 0),
            "bdi_pred": int(result.get("bdi-score", 0) or 0),
            "key_symptoms_true": list(profile.get("key_symptoms", []) or []),
            "key_symptoms_pred": list(result.get("key-symptoms", []) or []),
            "item_scores_true": true_scores,
            "item_scores_pred": pred_scores,
            "finalizer_summary": finalizer_summary,
            **_flatten_finalizer_summary(finalizer_summary),
            "finalizer_guardrail_consistency_ok": (
                not finalizer_summary
                or bool(finalizer_summary.get("low_signal_guardrail_active", False))
                or bool(finalizer_summary.get("severe_recovery_mode_active", False))
            ),
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


def summarize_transcript_style(transcript_source: str | Path) -> Dict[str, Any]:
    candidate_path = Path(transcript_source)
    if candidate_path.exists():
        text = candidate_path.read_text(encoding="utf-8")
    else:
        text = str(transcript_source)
    responses = re.findall(r"^(?:Patient|Persona):\s*(.*)$", text, flags=re.M)
    summary = _style_summary_from_responses(responses)
    summary["source_kind"] = "path" if candidate_path.exists() else "inline_text"
    return summary


def summarize_simulated_style(
    *,
    persona_count: int = 12,
    seed: int = 42,
    probe_battery: List[Dict[str, object]] | None = None,
) -> Dict[str, Any]:
    probes = list(probe_battery or SIM_STYLE_CALIBRATION_PROBES)
    profiles = generate_persona_pool(count=persona_count, seed=seed)
    responses: List[str] = []
    for profile in profiles:
        persona = create_persona(profile)
        history: List[Dict[str, object]] = []
        for probe in probes:
            reply = persona.reply(history, dict(probe))
            responses.append(reply)
            history.append({"role": "user", "content": f"probe-{probe.get('target_item_id', '')}"})
            history.append({"role": "assistant", "content": reply})

    summary = _style_summary_from_responses(responses)
    summary["persona_count"] = int(persona_count)
    summary["probe_count"] = len(probes)
    summary["seed"] = int(seed)
    return summary


def compare_style_summaries(reference: Dict[str, Any], simulated: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "avg_response_words",
        "qualifier_rate",
        "hedge_rate",
        "context_anchor_rate",
        "mixed_answer_rate",
        "soft_denial_rate",
        "baseline_comparison_rate",
    ]
    comparison: Dict[str, Any] = {
        "reference_response_count": int(reference.get("response_count", 0) or 0),
        "simulated_response_count": int(simulated.get("response_count", 0) or 0),
    }
    for key in keys:
        reference_value = float(reference.get(key, 0.0) or 0.0)
        simulated_value = float(simulated.get(key, 0.0) or 0.0)
        comparison[f"reference_{key}"] = round(reference_value, 4)
        comparison[f"simulated_{key}"] = round(simulated_value, 4)
        comparison[f"delta_{key}"] = round(simulated_value - reference_value, 4)
    return comparison


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
    "SIM_STYLE_CALIBRATION_PROBES",
    "build_persona_error_table",
    "RECORD_COLUMNS",
    "build_item_error_table",
    "compare_style_summaries",
    "load_eval_records",
    "load_eval_metrics",
    "run_eval_notebook",
    "summarize_simulated_style",
    "summarize_transcript_style",
    "split_item_error_table",
]
