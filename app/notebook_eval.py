from __future__ import annotations

import json
import re
from collections import Counter
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

NOTEBOOK_STABILITY_PERSONA_THRESHOLD = 10
SELF_WORTH_CLUSTER_ITEM_IDS = (7, 8, 14)
ARTIFACT_RUN_CONSISTENCY_FILES = (
    "metrics_run_local.json",
    "failure_report_run_local.json",
    "benchmark_integrity_run_local.json",
    "config_used.json",
    "diagnostics_run_local.json",
    "extract_failure_log_run_local.json",
    "extract_true_parse_fail_log_run_local.json",
    "extract_runtime_error_log_run_local.json",
    "extract_parse_fail_log_run_local.json",
)


def _item_value_columns(prefix: str) -> List[str]:
    return [f"item_{item_id}_{prefix}" for item_id in range(1, 22)]


RECORD_COLUMNS = RECORD_BASE_COLUMNS + _item_value_columns("true") + _item_value_columns("pred")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
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


def load_failure_report(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    failure_path = artifact_dir / "failure_report_run_local.json"
    payload = _load_json(failure_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected failure report payload to be a JSON object: {failure_path}")
    return payload


def load_config_used(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    config_path = artifact_dir / "config_used.json"
    payload = _load_json(config_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected config payload to be a JSON object: {config_path}")
    return payload


def load_diagnostics(output_dir: str | Path) -> List[Dict[str, Any]]:
    artifact_dir = Path(output_dir)
    diagnostics_path = artifact_dir / "diagnostics_run_local.json"
    payload = _load_json(diagnostics_path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected diagnostics payload to be a JSON array: {diagnostics_path}")
    return [row for row in payload if isinstance(row, dict)]


def _artifact_dict(path: Path) -> Dict[str, Any]:
    payload = _load_json_if_exists(path)
    if isinstance(payload, dict):
        return payload
    return {}


def _artifact_list(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json_if_exists(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _artifact_run_id(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("artifact_run_id", "") or "").strip()
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return str(payload[0].get("artifact_run_id", "") or "").strip()
    return ""


def _artifact_generated_at(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("generated_at", "") or "").strip()
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return str(payload[0].get("generated_at", "") or "").strip()
    return ""


def inspect_artifact_run_consistency(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_rows: List[Dict[str, Any]] = []
    run_ids: List[str] = []
    for filename in ARTIFACT_RUN_CONSISTENCY_FILES:
        path = artifact_dir / filename
        payload = _load_json_if_exists(path)
        if payload is None:
            continue
        run_id = _artifact_run_id(payload)
        generated_at = _artifact_generated_at(payload)
        if run_id:
            run_ids.append(run_id)
        artifact_rows.append(
            {
                "artifact": filename,
                "artifact_run_id": run_id,
                "generated_at": generated_at,
            }
        )

    run_id_counts = Counter(run_ids)
    unique_run_ids = sorted(run_id_counts.keys())
    missing_run_id_artifacts = sorted(
        row["artifact"] for row in artifact_rows if not str(row.get("artifact_run_id", "")).strip()
    )
    mixed_run_detected = len(unique_run_ids) > 1
    if mixed_run_detected:
        warning = (
            "Mixed-run artifact set detected in outputs/: multiple artifact_run_id values are present. "
            "Do not compare these files as if they came from one eval run."
        )
    elif missing_run_id_artifacts and artifact_rows:
        warning = (
            "Some artifacts are missing artifact_run_id metadata, so run consistency could not be fully verified."
        )
    else:
        warning = ""

    return {
        "mixed_run_detected": bool(mixed_run_detected),
        "artifact_count_checked": len(artifact_rows),
        "artifact_run_rows": artifact_rows,
        "unique_run_ids": unique_run_ids,
        "missing_run_id_artifacts": missing_run_id_artifacts,
        "warning": warning,
    }


def _coerce_item_ids(raw_values: Any) -> List[int]:
    if not isinstance(raw_values, list):
        return []
    item_ids: List[int] = []
    for raw_value in raw_values:
        try:
            item_id = int(raw_value)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in item_ids:
            item_ids.append(item_id)
    return item_ids


def _turn_trace_extract(turn: Dict[str, Any]) -> Dict[str, Any]:
    trace = dict(turn.get("turn_trace", {}) or {})
    extract_trace = trace.get("extract_likelihoods")
    if not isinstance(extract_trace, dict):
        extract_trace = trace.get("extract_evidence")
    if not isinstance(extract_trace, dict):
        extract_trace = {}
    return dict(extract_trace)


def _turn_trace_belief(turn: Dict[str, Any]) -> Dict[str, Any]:
    trace = dict(turn.get("turn_trace", {}) or {})
    belief_trace = trace.get("belief_update")
    if not isinstance(belief_trace, dict):
        belief_trace = trace.get("update_beliefs")
    if not isinstance(belief_trace, dict):
        belief_trace = {}
    return dict(belief_trace)


def _iter_timeline_turns(diagnostics_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for record in diagnostics_payload:
        timeline = record.get("timeline", [])
        if not isinstance(timeline, list):
            continue
        for turn in timeline:
            if isinstance(turn, dict):
                turns.append(turn)
    return turns


def build_eval_stability_notice(
    metrics: Dict[str, Any] | None = None,
    *,
    persona_count: int | None = None,
    stable_threshold: int = NOTEBOOK_STABILITY_PERSONA_THRESHOLD,
) -> Dict[str, Any]:
    resolved_persona_count = int(persona_count or 0)
    if resolved_persona_count <= 0 and isinstance(metrics, dict):
        try:
            resolved_persona_count = int(metrics.get("persona_count", 0) or 0)
        except (TypeError, ValueError):
            resolved_persona_count = 0

    is_low_stability = resolved_persona_count > 0 and resolved_persona_count < int(stable_threshold)
    if is_low_stability:
        message = (
            f"Low-stability read: this run only has {resolved_persona_count} personas. "
            f"Keep it for quick debugging, but rerun with at least {stable_threshold} personas before treating "
            "headline metrics as benchmark-quality."
        )
        level = "warning"
    elif resolved_persona_count > 0:
        message = (
            f"Sample size is more stable for interpretation ({resolved_persona_count} personas, "
            f"threshold {stable_threshold})."
        )
        level = "ok"
    else:
        message = "Persona count is unavailable, so stability could not be assessed."
        level = "unknown"

    return {
        "persona_count": int(resolved_persona_count),
        "stable_threshold": int(stable_threshold),
        "is_low_stability": bool(is_low_stability),
        "level": level,
        "message": message,
    }


def resolve_runtime_artifact_metadata(output_dir: str | Path) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    config_payload = _artifact_dict(artifact_dir / "config_used.json")
    resolved = dict(config_payload.get("resolved_backends", {}) or {})
    if resolved:
        return {
            "detector_backend": str(resolved.get("detector_backend", "") or ""),
            "detector_target": str(resolved.get("detector_target", "") or ""),
            "persona_runtime": str(resolved.get("persona_runtime", "") or ""),
            "source": "config_used",
        }

    integrity_payload = _artifact_dict(artifact_dir / "benchmark_integrity_run_local.json")
    detector = dict(integrity_payload.get("detector", {}) or {})
    if detector:
        return {
            "detector_backend": str(detector.get("backend", "") or ""),
            "detector_target": str(detector.get("target", "") or ""),
            "persona_runtime": "deterministic_simulator",
            "source": "benchmark_integrity",
        }

    return {
        "detector_backend": "",
        "detector_target": "",
        "persona_runtime": "",
        "source": "unavailable",
    }


def build_compact_diagnostics_summary(
    output_dir: str | Path,
    *,
    item_error_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    artifact_dir = Path(output_dir)
    failure_report = _artifact_dict(artifact_dir / "failure_report_run_local.json")
    diagnostics_payload = _artifact_list(artifact_dir / "diagnostics_run_local.json")
    extract_failure_entries = _artifact_list(artifact_dir / "extract_failure_log_run_local.json")
    true_parse_entries = _artifact_list(artifact_dir / "extract_true_parse_fail_log_run_local.json")
    runtime_error_entries = _artifact_list(artifact_dir / "extract_runtime_error_log_run_local.json")
    artifact_consistency = inspect_artifact_run_consistency(artifact_dir)
    timeline_turns = _iter_timeline_turns(diagnostics_payload)

    extract_turn_count = 0
    llm_extractor_error_turns = 0
    module3_target_coverage = {item_id: 0 for item_id in SELF_WORTH_CLUSTER_ITEM_IDS}
    module3_target_mismatch_examples: List[Dict[str, Any]] = []
    module3_targeted_turns = 0

    for turn in timeline_turns:
        route_decision = dict(turn.get("route_decision", {}) or {})
        target_items = _coerce_item_ids(route_decision.get("target_items", []))
        extract_trace = _turn_trace_extract(turn)
        belief_trace = _turn_trace_belief(turn)
        extract_source = str(extract_trace.get("source", "") or "").strip()
        recovery_trigger = str(extract_trace.get("detail_module3_scoped_recovery_trigger", "") or "").strip()
        if extract_source:
            extract_turn_count += 1
        if extract_source == "llm_extractor_error" or recovery_trigger == "llm_extractor_error":
            llm_extractor_error_turns += 1

        updated_item_ids = _coerce_item_ids(belief_trace.get("updated_item_ids", []))
        for item_id in target_items:
            if item_id not in module3_target_coverage:
                continue
            module3_target_coverage[item_id] += 1
            module3_targeted_turns += 1
            if updated_item_ids and item_id not in updated_item_ids and len(module3_target_mismatch_examples) < 5:
                module3_target_mismatch_examples.append(
                    {
                        "turn": int(turn.get("turn", 0) or 0),
                        "target_item_id": int(item_id),
                        "updated_item_ids": list(updated_item_ids),
                        "extract_source": extract_source,
                    }
                )

    if extract_turn_count <= 0:
        source_distribution = dict(failure_report.get("extract_source_distribution", {}) or {})
        extract_turn_count = sum(
            int(value or 0)
            for value in source_distribution.values()
            if isinstance(value, (int, float))
        )
        llm_extractor_error_turns = int(source_distribution.get("llm_extractor_error", 0) or 0)

    llm_extractor_error_rate = (
        round(float(llm_extractor_error_turns) / float(extract_turn_count), 4)
        if extract_turn_count > 0
        else 0.0
    )

    failure_kind_distribution = dict(failure_report.get("extract_failure_kind_distribution", {}) or {})
    if not failure_kind_distribution and extract_failure_entries:
        failure_kind_distribution = dict(
            Counter(str(entry.get("entry_kind", "") or "") for entry in extract_failure_entries if str(entry.get("entry_kind", "") or ""))
        )
    failure_reason_distribution = dict(failure_report.get("extract_failure_reason_distribution", {}) or {})
    if not failure_reason_distribution and extract_failure_entries:
        failure_reason_distribution = dict(
            Counter(
                str(entry.get("failure_reason", "") or "")
                for entry in extract_failure_entries
                if str(entry.get("failure_reason", "") or "")
            )
        )

    extract_failure_log_count = int(
        failure_report.get(
            "extract_failure_log_count",
            len(extract_failure_entries) or int(failure_report.get("extract_parse_fail_log_count", 0) or 0),
        )
        or 0
    )
    true_parse_fail_count = int(
        failure_report.get("extract_true_parse_fail_log_count", len(true_parse_entries)) or 0
    )
    runtime_error_count = int(
        failure_report.get("extract_runtime_error_log_count", len(runtime_error_entries)) or 0
    )
    non_failure_info_count = int(failure_report.get("extract_non_failure_log_count", 0) or 0)
    opportunistic_no_candidate_count = int(
        failure_kind_distribution.get(
            "opportunistic_no_candidate",
            failure_reason_distribution.get("opportunistic_shortlist_no_candidates", 0),
        )
        or 0
    )
    empty_or_unsupported_count = int(
        failure_kind_distribution.get(
            "empty_or_unsupported",
            max(0, extract_failure_log_count - true_parse_fail_count - runtime_error_count - opportunistic_no_candidate_count),
        )
        or 0
    )
    top_failure_reasons = [
        {"failure_reason": reason, "count": int(count)}
        for reason, count in sorted(
            failure_reason_distribution.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:5]
    ]

    dominant_issue = "No extraction failure dominant issue detected."
    if artifact_consistency["mixed_run_detected"]:
        dominant_issue = "Mixed-run artifact set detected; clean outputs or rerun before interpreting the failure mix."
    else:
        dominant_candidates = {
            "malformed JSON": int(true_parse_fail_count),
            "backend/runtime exception": int(runtime_error_count),
            "valid-but-empty extraction": int(empty_or_unsupported_count),
            "opportunistic shortlist no-candidate": int(opportunistic_no_candidate_count),
        }
        dominant_label, dominant_count = max(
            dominant_candidates.items(),
            key=lambda item: (int(item[1]), item[0]),
        )
        if dominant_count > 0:
            dominant_issue = f"Dominant extraction issue: {dominant_label} ({dominant_count} turns)."

    if item_error_df is None:
        records_df = load_eval_records(artifact_dir)
        item_error_df = build_item_error_table(records_df)

    worst_under_predicted_items: List[Dict[str, Any]] = []
    if item_error_df is not None and not item_error_df.empty:
        under_predicted = (
            item_error_df[item_error_df["mean_error"] < 0]
            .sort_values(["mean_error", "item_id"], ascending=[True, True])
            .head(3)
        )
        for _, row in under_predicted.iterrows():
            worst_under_predicted_items.append(
                {
                    "item_id": int(row["item_id"]),
                    "symptom_name": str(row["symptom_name"]),
                    "mean_error": round(float(row["mean_error"]), 4),
                }
            )

    item7_count = int(module3_target_coverage.get(7, 0) or 0)
    item8_count = int(module3_target_coverage.get(8, 0) or 0)
    item14_count = int(module3_target_coverage.get(14, 0) or 0)
    if module3_targeted_turns <= 0:
        coverage_message = "Module-3 self-worth items were not targeted in the saved diagnostics."
    elif item7_count > 0 and item8_count == 0 and item14_count == 0:
        coverage_message = (
            f"Module-3 self-worth coverage is collapsed onto item 7 ({item7_count} turns); "
            "items 8 and 14 were never directly targeted."
        )
    else:
        coverage_message = (
            f"Module-3 self-worth target coverage: item 7 -> {item7_count}, "
            f"item 8 -> {item8_count}, item 14 -> {item14_count}."
        )

    return {
        "extract_empty_rate": round(float(failure_report.get("extract_empty_rate", 0.0) or 0.0), 4),
        "evidence_nonempty_rate": round(float(failure_report.get("evidence_nonempty_rate", 0.0) or 0.0), 4),
        "extract_turn_count": int(extract_turn_count),
        "llm_extractor_error_turns": int(llm_extractor_error_turns),
        "llm_extractor_error_rate": round(float(llm_extractor_error_rate), 4),
        "extract_failure_log_count": int(extract_failure_log_count),
        "true_parse_fail_count": int(true_parse_fail_count),
        "runtime_error_count": int(runtime_error_count),
        "empty_or_unsupported_count": int(empty_or_unsupported_count),
        "opportunistic_no_candidate_count": int(opportunistic_no_candidate_count),
        "non_failure_info_count": int(non_failure_info_count),
        "failure_kind_distribution": failure_kind_distribution,
        "failure_reason_distribution": failure_reason_distribution,
        "top_failure_reasons": top_failure_reasons,
        "dominant_issue": dominant_issue,
        "worst_under_predicted_items": worst_under_predicted_items,
        "module3_target_coverage": module3_target_coverage,
        "module3_targeted_turns": int(module3_targeted_turns),
        "module3_target_mismatch_examples": module3_target_mismatch_examples,
        "coverage_message": coverage_message,
        "artifact_consistency": artifact_consistency,
        "artifact_warning": str(artifact_consistency.get("warning", "") or ""),
    }


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
    "NOTEBOOK_STABILITY_PERSONA_THRESHOLD",
    "PERSONA_ERROR_COLUMNS",
    "SIM_STYLE_CALIBRATION_PROBES",
    "build_compact_diagnostics_summary",
    "build_eval_stability_notice",
    "build_persona_error_table",
    "RECORD_COLUMNS",
    "build_item_error_table",
    "compare_style_summaries",
    "inspect_artifact_run_consistency",
    "load_config_used",
    "load_diagnostics",
    "load_eval_records",
    "load_failure_report",
    "load_eval_metrics",
    "resolve_runtime_artifact_metadata",
    "run_eval_notebook",
    "summarize_simulated_style",
    "summarize_transcript_style",
    "split_item_error_table",
]
