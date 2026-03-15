from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from core.llm import get_llm_usage
from core.runtime_policy import resolve_detector_backend
from core.state import symptom_name_from_item

from app.cli_common import _current_git_hash, _git_is_dirty, _serialize, _write_json
from app.cli_eval_helpers import _family_summary, _profile_meta, _select_primary_metrics, _with_objective


def _evaluation_stability_warnings(metrics: Dict[str, Any], split_name: str) -> List[str]:
    warnings: List[str] = []
    if not metrics:
        return [f"{split_name}:no_records"]
    return warnings


def _predicted_key_pairs(item_ids: List[int], symptom_names: List[str]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for idx, item_id in enumerate(item_ids):
        name = symptom_names[idx] if idx < len(symptom_names) else ""
        pairs.append({"item_id": int(item_id), "symptom_name": str(name)})
    return pairs


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn <= 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _build_error_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "profiles_evaluated": 0,
            "item_metrics": [],
            "family_item_mae": {},
            "worst_personas_by_bdi_abs_error": [],
        }

    item_metrics: List[Dict[str, Any]] = []
    for item_id in range(1, 22):
        tp = fp = fn = 0
        abs_errors: List[float] = []
        true_pos_count = 0
        pred_pos_count = 0
        for row in rows:
            scores_true = dict(row.get("item_scores_true", {}))
            scores_pred = dict(row.get("item_scores_pred", {}))
            true_score = int(scores_true.get(str(item_id), scores_true.get(item_id, 0)) or 0)
            pred_score = int(scores_pred.get(str(item_id), scores_pred.get(item_id, 0)) or 0)
            true_pos = true_score >= 1
            pred_pos = pred_score >= 1
            if true_pos and pred_pos:
                tp += 1
            elif (not true_pos) and pred_pos:
                fp += 1
            elif true_pos and (not pred_pos):
                fn += 1
            if true_pos:
                true_pos_count += 1
            if pred_pos:
                pred_pos_count += 1
            abs_errors.append(abs(float(true_score) - float(pred_score)))
        item_mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        item_f1 = _f1_from_counts(tp, fp, fn)
        item_metrics.append(
            {
                "item_id": item_id,
                "symptom_name": symptom_name_from_item(item_id),
                "f1_at_1": round(item_f1, 4),
                "mae": round(item_mae, 4),
                "true_positive_count": true_pos_count,
                "pred_positive_count": pred_pos_count,
            }
        )

    family_errors: Dict[str, List[float]] = {}
    for row in rows:
        family = str(row.get("family", "unknown"))
        bucket = family_errors.setdefault(family, [])
        scores_true = dict(row.get("item_scores_true", {}))
        scores_pred = dict(row.get("item_scores_pred", {}))
        for item_id in range(1, 22):
            true_score = int(scores_true.get(str(item_id), scores_true.get(item_id, 0)) or 0)
            pred_score = int(scores_pred.get(str(item_id), scores_pred.get(item_id, 0)) or 0)
            bucket.append(abs(float(true_score) - float(pred_score)))
    family_item_mae = {
        family: round((sum(values) / len(values)) if values else 0.0, 4)
        for family, values in sorted(family_errors.items(), key=lambda pair: pair[0])
    }

    worst_personas = sorted(
        (
            {
                "llm": str(row.get("llm", "")),
                "family": str(row.get("family", "unknown")),
                "split": str(row.get("split", "unknown")),
                "bdi_true": int(row.get("bdi_true", 0)),
                "bdi_pred": int(row.get("bdi_pred", 0)),
                "bdi_abs_error": abs(int(row.get("bdi_true", 0)) - int(row.get("bdi_pred", 0))),
            }
            for row in rows
        ),
        key=lambda row: (-int(row["bdi_abs_error"]), row["llm"]),
    )[:10]

    return {
        "profiles_evaluated": len(rows),
        "item_metrics": item_metrics,
        "item_metrics_sorted_by_mae_desc": sorted(item_metrics, key=lambda row: (-float(row["mae"]), int(row["item_id"]))),
        "family_item_mae": family_item_mae,
        "worst_personas_by_bdi_abs_error": worst_personas,
    }


def build_eval_diagnostics_entry(
    *,
    profile,
    final_state: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    style_stats: Dict[str, Any],
    trace_level: str,
) -> Dict[str, Any]:
    return {
        "LLM": profile.persona_id,
        "source": profile.source,
        "has_ground_truth": profile.has_ground_truth,
        "persona_meta": _profile_meta(profile),
        "timeline": _serialize(timeline if trace_level == "compact" else []),
        "final_state": _serialize(
            {
                "turn_index": final_state.get("turn_index", 0),
                "predicted_label": final_state.get("predicted_label"),
                "predicted_bdi_score": final_state.get("predicted_bdi_score"),
                "raw_predicted_label": final_state.get("raw_predicted_label"),
                "raw_predicted_bdi_score": final_state.get("raw_predicted_bdi_score"),
                "predicted_key_symptoms": final_state.get("predicted_key_symptoms", []),
                "predicted_key_item_ids": list(final_state.get("predicted_key_item_ids", []))[:4],
                "predicted_key_pairs": _predicted_key_pairs(
                    list(final_state.get("predicted_key_item_ids", []))[:4],
                    list(final_state.get("predicted_key_symptoms", []))[:4],
                ),
                "risk_flag": bool(final_state.get("risk_flag", False)),
                "global_confidence": final_state.get("global_confidence", 0.0),
                "evidence_log_count": len(final_state.get("evidence_log", [])),
                "last_route_decision": (
                    _serialize(final_state.get("route_history", [])[-1])
                    if final_state.get("route_history")
                    else None
                ),
                "last_stop_decision": (
                    _serialize(final_state.get("stop_history", [])[-1])
                    if final_state.get("stop_history")
                    else None
                ),
                "final_item_scores_non_zero": {
                    str(k): int(v)
                    for k, v in dict(final_state.get("final_item_scores", {})).items()
                    if int(v) > 0
                },
                "imputed_item_count": (
                    int(final_state.get("module_imputation", {}).get("imputed_item_count", 0))
                    if isinstance(final_state.get("module_imputation", {}), dict)
                    else 0
                ),
                "blended_observed_item_count": (
                    int(final_state.get("module_imputation", {}).get("blended_observed_item_count", 0))
                    if isinstance(final_state.get("module_imputation", {}), dict)
                    else 0
                ),
                "failure_counters": final_state.get("failure_counters", {}),
                "sim_style_stats": style_stats,
            }
        ),
    }


def _detector_target() -> str:
    detector_backend = resolve_detector_backend()
    if detector_backend == "ollama":
        return os.getenv("OLLAMA_DETECTOR_MODEL", "qwen3.5:4b")
    return os.getenv("OPENROUTER_DETECTOR_MODEL", "openrouter/auto")


def _persona_ids_from_results(results: List[Any]) -> List[str]:
    persona_ids: List[str] = []
    for result in results:
        raw_id = getattr(result, "LLM", None)
        if raw_id is None and isinstance(result, dict):
            raw_id = result.get("LLM")
        persona_id = str(raw_id or "").strip()
        if persona_id:
            persona_ids.append(persona_id)
    return persona_ids


def _duplicate_ids(values: List[str]) -> List[str]:
    counts = Counter(value for value in values if value)
    return sorted([value for value, count in counts.items() if count > 1])


def _build_benchmark_integrity(
    *,
    manifest_payload: Dict[str, Any],
    results: List[Any],
    eval_ids: List[str],
    manifest_hash: str,
    prior_manifest_info: Dict[str, Any],
    prompt_version: str,
) -> Dict[str, Any]:
    profiles = list(manifest_payload.get("profiles", []) or [])
    manifest_persona_ids = [str(profile.get("persona_id", "")).strip() for profile in profiles if str(profile.get("persona_id", "")).strip()]
    result_persona_ids = _persona_ids_from_results(results)
    eval_loop_ids = [str(persona_id).strip() for persona_id in eval_ids if str(persona_id).strip()]

    manifest_duplicates = _duplicate_ids(manifest_persona_ids)
    result_duplicates = _duplicate_ids(result_persona_ids)
    eval_duplicates = _duplicate_ids(eval_loop_ids)

    manifest_id_set = set(manifest_persona_ids)
    result_id_set = set(result_persona_ids)
    eval_id_set = set(eval_loop_ids)

    alignment_missing_results = sorted(manifest_id_set - result_id_set)
    alignment_unexpected_results = sorted(result_id_set - manifest_id_set)
    alignment_missing_eval = sorted(manifest_id_set - eval_id_set)
    alignment_unexpected_eval = sorted(eval_id_set - manifest_id_set)
    results_alignment_pass = not (
        manifest_duplicates
        or result_duplicates
        or eval_duplicates
        or alignment_missing_results
        or alignment_unexpected_results
        or alignment_missing_eval
        or alignment_unexpected_eval
    )

    non_synthetic_ids: List[str] = []
    missing_ground_truth_ids: List[str] = []
    inconsistent_total_ids: List[str] = []
    unexpected_split_ids: List[str] = []
    manifest_issues: List[str] = []

    for profile in profiles:
        persona_id = str(profile.get("persona_id", "")).strip()
        if str(profile.get("source", "")).strip().lower() != "synthetic":
            non_synthetic_ids.append(persona_id)
        if not bool(profile.get("has_ground_truth", False)):
            missing_ground_truth_ids.append(persona_id)
        if str(profile.get("split", "")).strip() != "eval":
            unexpected_split_ids.append(persona_id)
        if "generator_version" in profile:
            manifest_issues.append(f"{persona_id}:legacy_generator_version_present")
        raw_scores = dict(profile.get("bdi_scores", {}) or {})
        computed_total = sum(int(raw_scores.get(str(item_id), raw_scores.get(item_id, 0)) or 0) for item_id in range(1, 22))
        stored_total = int(profile.get("bdi_total", 0) or 0)
        if stored_total != min(computed_total, 63):
            inconsistent_total_ids.append(persona_id)

    run_config = dict(manifest_payload.get("run_config", {}) or {})
    if manifest_payload.get("persona_count") != len(profiles):
        manifest_issues.append("persona_count_mismatch")
    if run_config.get("persona_count") != len(profiles):
        manifest_issues.append("run_config_persona_count_mismatch")
    if "generator_version" in run_config:
        manifest_issues.append("legacy_run_config_generator_version_present")
    if "generator_version" in manifest_payload:
        manifest_issues.append("legacy_manifest_generator_version_present")
    if manifest_duplicates:
        manifest_issues.append("duplicate_manifest_persona_ids")
    if inconsistent_total_ids:
        manifest_issues.append("bdi_total_mismatch")
    if unexpected_split_ids:
        manifest_issues.append("unexpected_profile_split")

    synthetic_ground_truth_pass = not non_synthetic_ids and not missing_ground_truth_ids
    manifest_consistency_pass = not manifest_issues
    git_dirty = _git_is_dirty()
    prior_hash = prior_manifest_info.get("hash")
    prior_read_error = prior_manifest_info.get("read_error")

    issues: List[str] = []
    if not results_alignment_pass:
        issues.append("results_alignment_failed")
    if not synthetic_ground_truth_pass:
        issues.append("synthetic_ground_truth_check_failed")
    if not manifest_consistency_pass:
        issues.append("manifest_consistency_failed")

    return {
        "pass": results_alignment_pass and synthetic_ground_truth_pass and manifest_consistency_pass,
        "evaluation_mode": "synthetic",
        "persona_regeneration_policy": "always_regenerate",
        "manifest_hash": manifest_hash,
        "prompt_version": prompt_version,
        "detector": {
            "backend": resolve_detector_backend(),
            "target": _detector_target(),
        },
        "git": {
            "commit": _current_git_hash(),
            "dirty": git_dirty,
        },
        "prior_manifest": {
            "exists": bool(prior_manifest_info.get("exists", False)),
            "hash": prior_hash,
            "profile_count": int(prior_manifest_info.get("profile_count", 0) or 0),
            "matches_current": (prior_hash == manifest_hash) if prior_hash else None,
            "read_error": prior_read_error,
        },
        "results_alignment": {
            "pass": results_alignment_pass,
            "manifest_persona_count": len(manifest_persona_ids),
            "result_persona_count": len(result_persona_ids),
            "eval_loop_persona_count": len(eval_loop_ids),
            "duplicate_manifest_persona_ids": manifest_duplicates,
            "duplicate_result_persona_ids": result_duplicates,
            "duplicate_eval_loop_persona_ids": eval_duplicates,
            "missing_in_results": alignment_missing_results,
            "unexpected_in_results": alignment_unexpected_results,
            "missing_in_eval_loop": alignment_missing_eval,
            "unexpected_in_eval_loop": alignment_unexpected_eval,
        },
        "synthetic_ground_truth": {
            "pass": synthetic_ground_truth_pass,
            "all_synthetic": not non_synthetic_ids,
            "all_have_ground_truth": not missing_ground_truth_ids,
            "non_synthetic_ids": non_synthetic_ids,
            "missing_ground_truth_ids": missing_ground_truth_ids,
        },
        "manifest_consistency": {
            "pass": manifest_consistency_pass,
            "issues": manifest_issues,
            "inconsistent_bdi_total_ids": inconsistent_total_ids,
            "unexpected_split_ids": unexpected_split_ids,
        },
        "issues": issues,
    }


def write_eval_artifacts(
    *,
    output_dir: Path,
    conversations: List,
    results: List,
    diagnostics_payload: List[Dict[str, Any]],
    overall_rows: List[Dict[str, Any]],
    route_distribution: Counter[str],
    turns_total: int,
    evidence_turns_nonempty: int,
    evidence_records_total: int,
    extract_source_distribution: Counter[str],
    extract_recovery_distribution: Counter[str],
    route_policy_distribution: Counter[str],
    duplicate_evidence_rows_total: int,
    contradiction_evidence_rows_total: int,
    support_increments_total: int,
    method_weight_usage: Counter[str],
    post_floor_new_items_total: int,
    post_floor_nonempty_turns_total: int,
    post_floor_turns_total: int,
    min_turns_for_productivity: int,
    early_stop_reason_distribution: Counter[str],
    extract_parse_fail_log_entries: List[Dict[str, Any]],
    run_failure_counters: Counter[str],
    eval_ids: List[str],
    manifest_hash: str,
    manifest_payload: Dict[str, Any],
    prior_manifest_info: Dict[str, Any],
    prompt_version: str,
    seed: int,
    persona_count: int,
    processed_profiles: int,
    trace_level: str,
    max_api_calls: int,
    save_diagnostics: bool,
    debug_outputs: bool,
    run_profile: str,
    requested_save_diagnostics: bool,
    requested_trace_level: str,
    requested_debug_outputs: bool,
    all_profiles: List,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    overall_metrics = compute_metrics(overall_rows) if overall_rows else {}
    max_turns = int(os.getenv("MAX_TURNS", "40"))
    overall_metrics_payload = _with_objective(overall_metrics, max_turns=max_turns)
    primary_split, primary_metrics = _select_primary_metrics(
        overall_labeled=overall_metrics_payload,
    )
    evaluation_stability_warnings = []
    evaluation_stability_warnings.extend(_evaluation_stability_warnings(overall_metrics_payload, "overall_labeled"))

    metrics_payload: Dict[str, Any] = {
        "evaluation_mode": "synthetic",
        "prompt_version": prompt_version,
        "persona_count": len(all_profiles),
        "overall_labeled": overall_metrics_payload,
        "primary_eval_split": primary_split,
        "primary_metrics": primary_metrics,
        "evaluation_stability_warnings": evaluation_stability_warnings,
        "metric_semantics_warnings": [
            "Evaluation uses item-level BDI metrics only; binary depressed/control metrics are deprecated.",
            "symptom_f1_at_4 is an alias of item_f1_macro_at_1 (full 21-item macro F1 with positive threshold >=1).",
        ],
        "llm_usage": get_llm_usage(),
        "extract_json_parse_failures": int(run_failure_counters.get("extract_json_parse_fail", 0)),
    }
    if primary_metrics:
        metrics_payload.update(primary_metrics)

    interactions_payload = [conv.model_dump() for conv in conversations]
    results_payload = [result.to_erisk_dict() for result in results]
    _write_json(output_dir / "interactions_run_local.json", interactions_payload)
    _write_json(output_dir / "results_run_local.json", results_payload)
    _write_json(output_dir / "metrics_run_local.json", metrics_payload)
    error_report_payload: Dict[str, Any] = {
        "status": "disabled_in_lean_mode",
        "debug_outputs": False,
    }
    failure_report_payload: Dict[str, Any] = {
        "status": "disabled_in_lean_mode",
        "debug_outputs": False,
    }
    if debug_outputs:
        error_report_payload = _build_error_report(overall_rows)
        _write_json(output_dir / "error_report_run_local.json", error_report_payload)
        _write_json(output_dir / "extract_parse_fail_log_run_local.json", extract_parse_fail_log_entries)

    extractor_failure_entries = [
        entry for entry in extract_parse_fail_log_entries if bool(entry.get("counts_as_failure", True))
    ]
    extractor_non_failure_entries = [
        entry for entry in extract_parse_fail_log_entries if not bool(entry.get("counts_as_failure", True))
    ]

    evidence_nonempty_rate = (evidence_turns_nonempty / turns_total) if turns_total else 0.0
    avg_evidence_per_turn = (evidence_records_total / turns_total) if turns_total else 0.0
    extract_parse_fail_rate = (
        float(run_failure_counters.get("extract_json_parse_fail", 0)) / float(turns_total) if turns_total else 0.0
    )
    extract_empty_rate = float(run_failure_counters.get("extract_empty", 0)) / float(turns_total) if turns_total else 0.0
    duplicate_evidence_rows_rate = (
        float(duplicate_evidence_rows_total) / float(turns_total) if turns_total else 0.0
    )
    contradiction_evidence_rows_rate = (
        float(contradiction_evidence_rows_total) / float(turns_total) if turns_total else 0.0
    )
    support_increments_rate = (
        float(support_increments_total) / float(turns_total) if turns_total else 0.0
    )
    avg_new_items_after_min_turns = (
        float(post_floor_new_items_total) / float(post_floor_turns_total) if post_floor_turns_total else 0.0
    )
    avg_nonempty_turns_after_min_turns = (
        float(post_floor_nonempty_turns_total) / float(post_floor_turns_total) if post_floor_turns_total else 0.0
    )
    blended_observed_total = 0
    for entry in diagnostics_payload:
        final_state_payload = dict(entry.get("final_state", {}) if isinstance(entry, dict) else {})
        module_imputation = final_state_payload.get("module_imputation", {})
        if isinstance(module_imputation, dict):
            blended_observed_total += int(module_imputation.get("blended_observed_item_count", 0) or 0)
            continue
        blended_observed_total += int(final_state_payload.get("blended_observed_item_count", 0) or 0)
    blended_observed_mean = (
        float(blended_observed_total) / float(processed_profiles) if processed_profiles > 0 else 0.0
    )
    if debug_outputs:
        family_summary = _family_summary(overall_rows)
        failure_report_payload = {
            "run_summary": {
                "evaluation_mode": "synthetic",
                "prompt_version": prompt_version,
                "personas_requested": persona_count,
                "profiles_evaluated": processed_profiles,
                "turns_total": turns_total,
                "trace_level": trace_level,
                "max_api_calls": (max_api_calls if max_api_calls > 0 else None),
            },
            "failure_counters": dict(run_failure_counters),
            "route_distribution": dict(route_distribution),
            "route_policy_distribution": dict(route_policy_distribution),
            "evidence_nonempty_rate": round(evidence_nonempty_rate, 4),
            "avg_evidence_per_turn": round(avg_evidence_per_turn, 4),
            "extract_parse_fail_rate": round(extract_parse_fail_rate, 4),
            "extract_empty_rate": round(extract_empty_rate, 4),
            "extract_source_distribution": dict(extract_source_distribution),
            "extract_recovery_distribution": dict(extract_recovery_distribution),
            "method_weight_usage": dict(method_weight_usage),
            "extract_parse_fail_log_count": len(extractor_failure_entries),
            "extract_non_failure_log_count": len(extractor_non_failure_entries),
            "duplicate_evidence_rows_total": int(duplicate_evidence_rows_total),
            "duplicate_evidence_rows_rate": round(duplicate_evidence_rows_rate, 4),
            "contradiction_evidence_rows_total": int(contradiction_evidence_rows_total),
            "contradiction_evidence_rows_rate": round(contradiction_evidence_rows_rate, 4),
            "support_increments_total": int(support_increments_total),
            "support_increments_rate": round(support_increments_rate, 4),
            "confidence_semantics": (
                "global_confidence uses support+coverage saturation with near-monotonic smoothing: "
                "item_conf=1-exp(-support/CONF_SUPPORT_TAU); "
                "target=CONF_DEPTH_WEIGHT*depth_conf + CONF_COVERAGE_WEIGHT*coverage_conf; "
                "smoothed with CONF_UP_ALPHA and bounded no-info decay"
            ),
            "blended_observed_item_count_total": int(blended_observed_total),
            "blended_observed_item_count_mean_per_profile": round(blended_observed_mean, 4),
            "post_floor_productivity": {
                "min_turns_threshold": int(min_turns_for_productivity),
                "turns_after_min_turns": int(post_floor_turns_total),
                "avg_new_items_after_min_turns": round(avg_new_items_after_min_turns, 4),
                "avg_nonempty_turns_after_min_turns": round(avg_nonempty_turns_after_min_turns, 4),
                "early_stop_reason_distribution": dict(early_stop_reason_distribution),
            },
            "evaluation_stability_warnings": evaluation_stability_warnings,
            "item_f1_macro_at_1": float(primary_metrics.get("item_f1_macro_at_1", 0.0)) if primary_metrics else 0.0,
            "item_mae": float(primary_metrics.get("item_mae", 0.0)) if primary_metrics else 0.0,
            "llm_usage": get_llm_usage(),
            **family_summary,
        }
        _write_json(output_dir / "failure_report_run_local.json", failure_report_payload)

    benchmark_integrity_payload = _build_benchmark_integrity(
        manifest_payload=manifest_payload,
        results=results,
        eval_ids=eval_ids,
        manifest_hash=manifest_hash,
        prior_manifest_info=prior_manifest_info,
        prompt_version=prompt_version,
    )
    _write_json(output_dir / "benchmark_integrity_run_local.json", benchmark_integrity_payload)

    if debug_outputs and save_diagnostics:
        _write_json(output_dir / "diagnostics_run_local.json", diagnostics_payload)

    config_snapshot = {
        "args": {
            "mode": "eval",
            "personas": persona_count,
            "seed": seed,
            "prompt_version": prompt_version,
            "save_diagnostics_requested": bool(requested_save_diagnostics),
            "save_diagnostics_effective": bool(save_diagnostics),
            "trace_level_requested": requested_trace_level,
            "trace_level_effective": trace_level,
            "debug_outputs_requested": bool(requested_debug_outputs),
            "debug_outputs_effective": bool(debug_outputs),
            "run_profile": run_profile,
            "max_api_calls": max_api_calls,
        },
        "runtime": {
            "evaluation_mode": "synthetic",
            "persona_runtime": "deterministic_simulator",
        },
        "env": {
            "DETECTOR_BACKEND": os.getenv("DETECTOR_BACKEND", "openrouter"),
            "OPENROUTER_DETECTOR_MODEL": os.getenv("OPENROUTER_DETECTOR_MODEL", ""),
            "OPENROUTER_PROVIDER_ORDER": os.getenv("OPENROUTER_PROVIDER_ORDER", ""),
            "OPENROUTER_REQUIRE_PROVIDER_ORDER": os.getenv("OPENROUTER_REQUIRE_PROVIDER_ORDER", "0"),
            "OPENROUTER_REASONING_EFFORT": os.getenv("OPENROUTER_REASONING_EFFORT", ""),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "OLLAMA_DETECTOR_MODEL": os.getenv("OLLAMA_DETECTOR_MODEL", "qwen3.5:4b"),
            "OLLAMA_THINK_MODE": os.getenv("OLLAMA_THINK_MODE", "auto"),
            "DETECTOR_MAX_NEW_TOKENS": os.getenv("DETECTOR_MAX_NEW_TOKENS", "96"),
            "DETECTOR_TEMPERATURE": os.getenv("DETECTOR_TEMPERATURE", "0.2"),
            "DETECTOR_EXTRACTOR_MAX_NEW_TOKENS": os.getenv(
                "DETECTOR_EXTRACTOR_MAX_NEW_TOKENS",
                os.getenv("DETECTOR_MAX_NEW_TOKENS", "96"),
            ),
            "MIN_TURNS": os.getenv("MIN_TURNS", "20"),
            "MAX_TURNS": os.getenv("MAX_TURNS", "40"),
            "STOP_CONFIDENCE": os.getenv("STOP_CONFIDENCE", "0.66"),
            "CONF_SUPPORT_TAU": os.getenv("CONF_SUPPORT_TAU", "1.25"),
            "CONF_DEPTH_WEIGHT": os.getenv("CONF_DEPTH_WEIGHT", "0.70"),
            "CONF_UP_ALPHA": os.getenv("CONF_UP_ALPHA", "0.55"),
            "SUPERVISOR_EVIDENCE_MIN_SCORE": os.getenv("SUPERVISOR_EVIDENCE_MIN_SCORE", "0.30"),
            "SUPERVISOR_EVIDENCE_RISK_THRESHOLD": os.getenv("SUPERVISOR_EVIDENCE_RISK_THRESHOLD", "0.22"),
            "SUPERVISOR_ESCAPE_EMPTY_STREAK": os.getenv("SUPERVISOR_ESCAPE_EMPTY_STREAK", "2"),
            "RISK_SENTINEL_FLAG_THRESHOLD": os.getenv("RISK_SENTINEL_FLAG_THRESHOLD", "0.45"),
            "RISK_SENTINEL_SHORTCIRCUIT_THRESHOLD": os.getenv("RISK_SENTINEL_SHORTCIRCUIT_THRESHOLD", "1.1"),
            "SIM_HEDGE_RATE": os.getenv("SIM_HEDGE_RATE", "0.60"),
            "SIM_NORMALIZATION_RATE": os.getenv("SIM_NORMALIZATION_RATE", "0.40"),
            "SIM_CONTEXT_ANCHOR_RATE": os.getenv("SIM_CONTEXT_ANCHOR_RATE", "0.55"),
            "SIM_DIRECT_ANSWER_RATE": os.getenv("SIM_DIRECT_ANSWER_RATE", "0.82"),
        },
        "resolved_backends": {
            "detector_backend": resolve_detector_backend(),
            "detector_target": _detector_target(),
            "persona_runtime": "deterministic_simulator",
        },
        "llm_usage": get_llm_usage(),
        "manifest_hash": manifest_hash,
        "git_commit": _current_git_hash(),
        "git_dirty": _git_is_dirty(),
        "benchmark_provenance": {
            "evaluation_mode": "synthetic",
            "persona_regeneration_policy": "always_regenerate",
            "prior_manifest_exists": bool(prior_manifest_info.get("exists", False)),
            "prior_manifest_matches_current": (
                prior_manifest_info.get("hash") == manifest_hash if prior_manifest_info.get("hash") else None
            ),
        },
    }
    _write_json(output_dir / "config_used.json", config_snapshot)

    return (
        metrics_payload,
        failure_report_payload,
        benchmark_integrity_payload,
        config_snapshot,
        primary_metrics,
        error_report_payload,
    )
