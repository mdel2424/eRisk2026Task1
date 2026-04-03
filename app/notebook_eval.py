from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.probabilistic_runtime import CLUSTER_TO_ITEMS
from core.state import symptom_name_from_item
from persona import create_persona, generate_persona_pool
from persona.sim_behavior import response_style_flags

RUNTIME_SUMMARY_COLUMNS = [
    "runtime_active_cluster",
    "runtime_evidence_binding_coverage",
    "runtime_bound_positive_assertion_count",
    "runtime_emitted_evidence_count",
    "runtime_opening_bootstrap_applied",
    "runtime_opening_bootstrap_cluster",
    "runtime_opening_bootstrap_item_ids",
    "runtime_opening_signal_cluster",
    "runtime_opening_signal_item_ids",
    "runtime_opening_followup_cluster",
    "runtime_opening_cognitive_anchor_preserved",
    "runtime_diagnosis_confidence",
    "runtime_diagnosis_used_llm",
    "runtime_diagnosis_synthesis_mode",
    "runtime_supported_item_count",
    "runtime_bayes_node_posteriors",
]

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
    "runtime_summary",
    *RUNTIME_SUMMARY_COLUMNS,
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

RUNTIME_ANOMALY_COLUMNS = [
    "persona_id",
    "family",
    "source",
    "bdi_true",
    "bdi_pred",
    "runtime_active_cluster",
    "runtime_evidence_binding_coverage",
    "runtime_diagnosis_confidence",
    "runtime_supported_item_count",
    "runtime_somatic_posterior",
    "runtime_cognitive_posterior",
    "predicted_nonzero_item_count",
    "predicted_somatic_item_count",
    "predicted_cognitive_item_count",
    "dominant_predicted_cluster",
    "somatic_cluster_share",
    "bdi_abs_error",
]

RUNTIME_SUPPORT_GAP_COLUMNS = [
    "persona_id",
    "family",
    "source",
    "bdi_true",
    "bdi_pred",
    "bdi_abs_error",
    "runtime_supported_item_count",
    "runtime_diagnosis_confidence",
    "runtime_active_cluster",
]

OPENING_SIGNAL_COLUMNS = [
    "persona_id",
    "family",
    "source",
    "runtime_opening_bootstrap_applied",
    "runtime_opening_bootstrap_cluster",
    "runtime_opening_bootstrap_item_ids",
    "runtime_opening_followup_cluster",
    "runtime_opening_cognitive_anchor_preserved",
    "runtime_supported_item_count",
    "bdi_true",
    "bdi_pred",
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


def _cluster_counts_from_scores(scores: Dict[str, int]) -> Dict[str, int]:
    normalized = _normalize_item_scores(scores)
    cognitive_ids = set(CLUSTER_TO_ITEMS.get("cognitive_affective", []))
    somatic_ids = set(CLUSTER_TO_ITEMS.get("somatic_vegetative", []))
    positive_ids = [int(item_id) for item_id, score in normalized.items() if int(score) >= 1]
    cognitive_count = sum(1 for item_id in positive_ids if item_id in cognitive_ids)
    somatic_count = sum(1 for item_id in positive_ids if item_id in somatic_ids)
    return {
        "predicted_nonzero_item_count": len(positive_ids),
        "predicted_cognitive_item_count": cognitive_count,
        "predicted_somatic_item_count": somatic_count,
    }


def _flatten_runtime_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = dict(summary.get("diagnosis", {}) or {})
    judgment = dict(summary.get("judgment", {}) or {})
    navigation = dict(summary.get("navigation", {}) or {})
    bayes = dict(summary.get("bayes", {}) or {})
    return {
        "runtime_active_cluster": str(judgment.get("active_cluster", "") or ""),
        "runtime_evidence_binding_coverage": float(judgment.get("evidence_binding_coverage", 1.0) or 1.0),
        "runtime_bound_positive_assertion_count": int(judgment.get("bound_positive_assertion_count", 0) or 0),
        "runtime_emitted_evidence_count": int(judgment.get("emitted_evidence_count", 0) or 0),
        "runtime_opening_bootstrap_applied": bool(judgment.get("opening_bootstrap_applied", False)),
        "runtime_opening_bootstrap_cluster": str(judgment.get("opening_bootstrap_cluster", "") or ""),
        "runtime_opening_bootstrap_item_ids": [
            int(item_id)
            for item_id in list(judgment.get("opening_bootstrap_item_ids", []) or [])
            if 1 <= int(item_id) <= 21
        ],
        "runtime_opening_signal_cluster": str(navigation.get("opening_signal_cluster", "") or ""),
        "runtime_opening_signal_item_ids": [
            int(item_id)
            for item_id in list(navigation.get("opening_signal_item_ids", []) or [])
            if 1 <= int(item_id) <= 21
        ],
        "runtime_opening_followup_cluster": str(navigation.get("opening_followup_cluster", "") or ""),
        "runtime_opening_cognitive_anchor_preserved": bool(
            navigation.get("opening_cognitive_anchor_preserved", False)
        ),
        "runtime_diagnosis_confidence": float(diagnosis.get("confidence", 0.0) or 0.0),
        "runtime_diagnosis_used_llm": bool(diagnosis.get("used_llm", False)),
        "runtime_diagnosis_synthesis_mode": str(diagnosis.get("synthesis_mode", "") or ""),
        "runtime_supported_item_count": int(diagnosis.get("supported_item_count", 0) or 0),
        "runtime_bayes_node_posteriors": dict(bayes.get("node_posteriors", {}) or {}),
    }


def build_runtime_anomaly_table(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=RUNTIME_ANOMALY_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, row in records_df.iterrows():
        pred_scores = _scores_from_row(row, "item_scores_pred")
        cluster_counts = _cluster_counts_from_scores(pred_scores)
        node_posteriors = dict(row.get("runtime_bayes_node_posteriors", {}) or {})
        somatic_posterior = float(node_posteriors.get("somatic_vegetative", 0.0) or 0.0)
        cognitive_posterior = float(node_posteriors.get("cognitive_affective", 0.0) or 0.0)
        predicted_nonzero_item_count = int(cluster_counts["predicted_nonzero_item_count"])
        predicted_somatic_item_count = int(cluster_counts["predicted_somatic_item_count"])
        predicted_cognitive_item_count = int(cluster_counts["predicted_cognitive_item_count"])
        dominant_cluster = "none"
        if predicted_nonzero_item_count > 0:
            if predicted_somatic_item_count > predicted_cognitive_item_count:
                dominant_cluster = "somatic_vegetative"
            elif predicted_cognitive_item_count > predicted_somatic_item_count:
                dominant_cluster = "cognitive_affective"
            else:
                dominant_cluster = "balanced"
        somatic_share = (
            float(predicted_somatic_item_count) / float(predicted_nonzero_item_count)
            if predicted_nonzero_item_count > 0
            else 0.0
        )
        bdi_true = int(row.get("bdi_true", 0) or 0)
        bdi_pred = int(row.get("bdi_pred", 0) or 0)
        supported_item_count = int(row.get("runtime_supported_item_count", 0) or 0)
        diagnosis_confidence = float(row.get("runtime_diagnosis_confidence", 0.0) or 0.0)
        anomaly = (
            (diagnosis_confidence >= 0.70 and supported_item_count <= 1)
            or (predicted_nonzero_item_count >= 3 and somatic_share >= 0.75)
            or (somatic_posterior >= 0.80 and cognitive_posterior <= 0.35)
            or abs(bdi_pred - bdi_true) >= 6
        )
        if not anomaly:
            continue
        rows.append(
            {
                "persona_id": row.get("persona_id", ""),
                "family": row.get("family", ""),
                "source": row.get("source", ""),
                "bdi_true": bdi_true,
                "bdi_pred": bdi_pred,
                "runtime_active_cluster": row.get("runtime_active_cluster", ""),
                "runtime_evidence_binding_coverage": float(row.get("runtime_evidence_binding_coverage", 1.0) or 1.0),
                "runtime_diagnosis_confidence": diagnosis_confidence,
                "runtime_supported_item_count": supported_item_count,
                "runtime_somatic_posterior": round(somatic_posterior, 4),
                "runtime_cognitive_posterior": round(cognitive_posterior, 4),
                "predicted_nonzero_item_count": predicted_nonzero_item_count,
                "predicted_somatic_item_count": predicted_somatic_item_count,
                "predicted_cognitive_item_count": predicted_cognitive_item_count,
                "dominant_predicted_cluster": dominant_cluster,
                "somatic_cluster_share": round(somatic_share, 4),
                "bdi_abs_error": abs(bdi_pred - bdi_true),
            }
        )

    anomaly_df = pd.DataFrame(rows, columns=RUNTIME_ANOMALY_COLUMNS)
    if anomaly_df.empty:
        return anomaly_df
    return anomaly_df.sort_values(
        by=["bdi_abs_error", "runtime_diagnosis_confidence", "predicted_somatic_item_count"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def build_cluster_collapse_table(records_df: pd.DataFrame) -> pd.DataFrame:
    anomaly_df = build_runtime_anomaly_table(records_df)
    if anomaly_df.empty:
        return anomaly_df
    return anomaly_df.loc[
        (
            (anomaly_df["dominant_predicted_cluster"] == "somatic_vegetative")
            & (anomaly_df["somatic_cluster_share"] >= 0.75)
        )
        | (
            (anomaly_df["runtime_diagnosis_confidence"] >= 0.65)
            & (anomaly_df["runtime_supported_item_count"] <= 1)
        )
    ].reset_index(drop=True)


def build_runtime_support_gap_table(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=RUNTIME_SUPPORT_GAP_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, row in records_df.iterrows():
        bdi_true = int(row.get("bdi_true", 0) or 0)
        bdi_pred = int(row.get("bdi_pred", 0) or 0)
        bdi_abs_error = abs(bdi_pred - bdi_true)
        supported_item_count = int(row.get("runtime_supported_item_count", 0) or 0)
        if bdi_abs_error < 6 or supported_item_count > 2:
            continue
        rows.append(
            {
                "persona_id": row.get("persona_id", ""),
                "family": row.get("family", ""),
                "source": row.get("source", ""),
                "bdi_true": bdi_true,
                "bdi_pred": bdi_pred,
                "bdi_abs_error": bdi_abs_error,
                "runtime_supported_item_count": supported_item_count,
                "runtime_diagnosis_confidence": float(row.get("runtime_diagnosis_confidence", 0.0) or 0.0),
                "runtime_active_cluster": row.get("runtime_active_cluster", ""),
            }
        )

    support_gap_df = pd.DataFrame(rows, columns=RUNTIME_SUPPORT_GAP_COLUMNS)
    if support_gap_df.empty:
        return support_gap_df
    return support_gap_df.sort_values(
        by=["bdi_abs_error", "runtime_supported_item_count", "runtime_diagnosis_confidence"],
        ascending=[False, True, False],
        ignore_index=True,
    )


def build_opening_signal_table(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=OPENING_SIGNAL_COLUMNS)

    opening_df = records_df.loc[
        records_df["runtime_opening_bootstrap_applied"].fillna(False).astype(bool)
    ].copy()
    if opening_df.empty:
        return pd.DataFrame(columns=OPENING_SIGNAL_COLUMNS)

    opening_df["bdi_abs_error"] = (
        pd.to_numeric(opening_df["bdi_pred"], errors="coerce").fillna(0.0)
        - pd.to_numeric(opening_df["bdi_true"], errors="coerce").fillna(0.0)
    ).abs()
    return opening_df[OPENING_SIGNAL_COLUMNS].sort_values(
        by=["bdi_abs_error", "runtime_supported_item_count", "persona_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


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
        runtime_summary = dict(result.get("runtime_summary", {}) or {})
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
            "runtime_summary": runtime_summary,
            **_flatten_runtime_summary(runtime_summary),
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
    "RECORD_COLUMNS",
    "build_item_error_table",
    "build_persona_error_table",
    "compare_style_summaries",
    "load_benchmark_integrity",
    "load_eval_metrics",
    "load_eval_records",
    "run_eval_notebook",
    "split_item_error_table",
    "summarize_simulated_style",
    "summarize_transcript_style",
]
