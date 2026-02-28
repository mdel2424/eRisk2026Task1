from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from core.llm import get_llm_usage
from core.runtime_policy import auto_backend_switch_enabled, resolve_detector_backend, resolve_persona_backend
from core.state import symptom_name_from_item

from app.cli_common import _current_git_hash, _serialize, _write_json
from app.cli_eval_helpers import _family_summary, _profile_meta, _select_primary_metrics, _with_objective


def _evaluation_stability_warnings(metrics: Dict[str, Any], split_name: str) -> List[str]:
    warnings: List[str] = []
    if not metrics:
        return [f"{split_name}:no_records"]
    class_counts = metrics.get("class_counts", {}) if isinstance(metrics, dict) else {}
    depressed_true = int(class_counts.get("depressed_true", 0) or 0)
    control_true = int(class_counts.get("control_true", 0) or 0)
    if depressed_true == 0 or control_true == 0:
        warnings.append(f"{split_name}:single_class_eval_split")
    if not bool(metrics.get("binary_f1_defined", False)):
        warnings.append(f"{split_name}:binary_f1_unstable")
    if not bool(metrics.get("risk_recall_defined", False)):
        warnings.append(f"{split_name}:risk_recall_undefined")
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
                "failure_counters": final_state.get("failure_counters", {}),
                "calibrator_mode": final_state.get("calibrator_mode", ""),
                "sim_style_stats": style_stats,
            }
        ),
    }


def write_eval_artifacts(
    *,
    output_dir: Path,
    conversations: List,
    results: List,
    diagnostics_payload: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    overall_rows: List[Dict[str, Any]],
    route_distribution: Counter[str],
    calibrator_mode_counts: Counter[str],
    turns_total: int,
    evidence_turns_nonempty: int,
    evidence_records_total: int,
    run_failure_counters: Counter[str],
    calibrator_status: Dict[str, Any],
    id_overlap_counts: Dict[str, int],
    template_overlap_counts: Dict[str, int],
    template_validator: Dict[str, Any],
    leakage_reasons: List[str],
    calibrator_train_ids: List[str],
    eval_ids: List[str],
    manifest_hash: str,
    requested_eval_mode: str,
    effective_eval_mode: str,
    prompt_version: str,
    seed: int,
    persona_count: int,
    processed_profiles: int,
    trace_level: str,
    max_api_calls: int,
    save_diagnostics: bool,
    fit_calibrator_policy: str,
    train_profiles: List,
    val_profiles: List,
    test_profiles: List,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    train_eval_overlap = len(set(calibrator_train_ids).intersection(set(eval_ids)))
    if train_eval_overlap > 0:
        leakage_reasons.append("calibrator_train_eval_overlap")

    val_metrics = compute_metrics(val_rows) if val_rows else {}
    test_metrics = compute_metrics(test_rows) if test_rows else {}
    overall_metrics = compute_metrics(overall_rows) if overall_rows else {}
    max_turns = int(os.getenv("MAX_TURNS", "10"))
    val_metrics_payload = _with_objective(val_metrics, max_turns=max_turns)
    test_metrics_payload = _with_objective(test_metrics, max_turns=max_turns)
    overall_metrics_payload = _with_objective(overall_metrics, max_turns=max_turns)
    primary_split, primary_metrics = _select_primary_metrics(
        synthetic_val=val_metrics_payload,
        synthetic_test=test_metrics_payload,
        overall_labeled=overall_metrics_payload,
    )
    evaluation_stability_warnings = []
    evaluation_stability_warnings.extend(_evaluation_stability_warnings(val_metrics_payload, "synthetic_val"))
    evaluation_stability_warnings.extend(_evaluation_stability_warnings(test_metrics_payload, "synthetic_test"))
    evaluation_stability_warnings.extend(_evaluation_stability_warnings(overall_metrics_payload, "overall_labeled"))

    metrics_payload: Dict[str, Any] = {
        "eval_mode_requested": requested_eval_mode,
        "eval_mode_effective": effective_eval_mode,
        "prompt_version": prompt_version,
        "synthetic_train_count": len(train_profiles),
        "synthetic_val_count": len(val_profiles),
        "synthetic_test_count": len(test_profiles),
        "split_counts": {"train": len(train_profiles), "val": len(val_profiles), "test": len(test_profiles)},
        "synthetic_val": val_metrics_payload,
        "synthetic_test": test_metrics_payload,
        "overall_labeled": overall_metrics_payload,
        "primary_eval_split": primary_split,
        "primary_metrics": primary_metrics,
        "evaluation_stability_warnings": evaluation_stability_warnings,
        "metric_semantics_warnings": [
            "symptom_f1_at_4 is an alias of item_f1_macro_at_1 (full 21-item macro F1 with positive threshold >=1)."
        ],
        "calibrator_status": calibrator_status,
        "llm_usage": get_llm_usage(),
    }
    if primary_metrics:
        metrics_payload.update(primary_metrics)

    interactions_payload = [conv.model_dump() for conv in conversations]
    results_payload = [result.to_erisk_dict() for result in results]
    error_report_payload = _build_error_report(overall_rows)
    _write_json(output_dir / "interactions_run_local.json", interactions_payload)
    _write_json(output_dir / "results_run_local.json", results_payload)
    _write_json(output_dir / "metrics_run_local.json", metrics_payload)
    _write_json(output_dir / "error_report_run_local.json", error_report_payload)

    evidence_nonempty_rate = (evidence_turns_nonempty / turns_total) if turns_total else 0.0
    avg_evidence_per_turn = (evidence_records_total / turns_total) if turns_total else 0.0
    family_summary = _family_summary(overall_rows)
    failure_report_payload = {
        "run_summary": {
            "eval_mode_requested": requested_eval_mode,
            "eval_mode_effective": effective_eval_mode,
            "prompt_version": prompt_version,
            "personas_requested": persona_count,
            "profiles_evaluated": processed_profiles,
            "turns_total": turns_total,
            "trace_level": trace_level,
            "max_api_calls": (max_api_calls if max_api_calls > 0 else None),
        },
        "failure_counters": dict(run_failure_counters),
        "route_distribution": dict(route_distribution),
        "evidence_nonempty_rate": round(evidence_nonempty_rate, 4),
        "avg_evidence_per_turn": round(avg_evidence_per_turn, 4),
        "calibrator_mode_counts": dict(calibrator_mode_counts),
        "evaluation_stability_warnings": evaluation_stability_warnings,
        "class_counts": dict(primary_metrics.get("class_counts", {}) if isinstance(primary_metrics, dict) else {}),
        "binary_f1_defined": bool(primary_metrics.get("binary_f1_defined", False)) if primary_metrics else False,
        "risk_recall_defined": bool(primary_metrics.get("risk_recall_defined", False)) if primary_metrics else False,
        "llm_usage": get_llm_usage(),
        "calibrator_status": calibrator_status,
        **family_summary,
    }
    _write_json(output_dir / "failure_report_run_local.json", failure_report_payload)

    leakage_report_payload = {
        "split_sizes": {"train": len(train_profiles), "val": len(val_profiles), "test": len(test_profiles)},
        "id_overlap_counts": id_overlap_counts,
        "template_overlap_counts": template_overlap_counts,
        "template_phrase_overlap": template_validator,
        "calibrator_train_ids_count": len(set(calibrator_train_ids)),
        "eval_ids_count": len(set(eval_ids)),
        "train_eval_overlap_count": train_eval_overlap,
        "manifest_hash": manifest_hash,
        "strict_pass": len(leakage_reasons) == 0,
        "failure_reasons": leakage_reasons,
    }
    _write_json(output_dir / "leakage_report_run_local.json", leakage_report_payload)

    if save_diagnostics:
        _write_json(output_dir / "diagnostics_run_local.json", diagnostics_payload)

    config_snapshot = {
        "args": {
            "mode": "eval",
            "personas": persona_count,
            "seed": seed,
            "eval_mode": requested_eval_mode,
            "prompt_version": prompt_version,
            "save_diagnostics": save_diagnostics,
            "max_api_calls": max_api_calls,
            "trace_level": trace_level,
            "fit_calibrator": fit_calibrator_policy,
        },
        "env": {
            "PROMPT_VERSION": os.getenv("PROMPT_VERSION", "v1"),
            "AUTO_BACKEND_SWITCH": os.getenv("AUTO_BACKEND_SWITCH", "1"),
            "DETECTOR_BACKEND": os.getenv("DETECTOR_BACKEND", "local_hf"),
            "DETECTOR_MODEL": os.getenv("DETECTOR_MODEL", ""),
            "OPENROUTER_DETECTOR_MODEL": os.getenv("OPENROUTER_DETECTOR_MODEL", ""),
            "PERSONA_BACKEND": os.getenv("PERSONA_BACKEND", "openrouter_sim"),
            "PERSONA_RUNTIME_MODE": "deterministic_sim_only",
            "PROBE_INTENT_REQUIRED": "1",
            "MIN_CUDA_VRAM_GB": os.getenv("MIN_CUDA_VRAM_GB", "8"),
            "MIN_TURNS": os.getenv("MIN_TURNS", ""),
            "MAX_TURNS": os.getenv("MAX_TURNS", ""),
            "STOP_CONFIDENCE": os.getenv("STOP_CONFIDENCE", ""),
            "MIN_EVIDENCE_FOR_CONF_STOP": os.getenv("MIN_EVIDENCE_FOR_CONF_STOP", "2"),
            "MIN_ITEMS_OBSERVED_FOR_CONF_STOP": os.getenv("MIN_ITEMS_OBSERVED_FOR_CONF_STOP", "4"),
            "DETERMINISTIC_BDI_LABEL_THRESHOLD": os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"),
            "DETERMINISTIC_CORE_ITEM_MEAN_THRESHOLD": os.getenv("DETERMINISTIC_CORE_ITEM_MEAN_THRESHOLD", "0.6"),
            "DETERMINISTIC_CORE_ITEM_MIN_HITS": os.getenv("DETERMINISTIC_CORE_ITEM_MIN_HITS", "4"),
            "EVIDENCE_LLM_ON_LEXICAL_HIT": os.getenv("EVIDENCE_LLM_ON_LEXICAL_HIT", "0"),
            "FORCE_RISK_PROBE_TURN": os.getenv("FORCE_RISK_PROBE_TURN", "3"),
            "FORCE_SOMATIC_PROBE_TURN": os.getenv("FORCE_SOMATIC_PROBE_TURN", "4"),
            "SUPERVISOR_EVIDENCE_MIN_SCORE": os.getenv("SUPERVISOR_EVIDENCE_MIN_SCORE", "0.30"),
            "SUPERVISOR_EVIDENCE_RISK_THRESHOLD": os.getenv("SUPERVISOR_EVIDENCE_RISK_THRESHOLD", "0.22"),
            "SUPERVISOR_ESCAPE_EMPTY_STREAK": os.getenv("SUPERVISOR_ESCAPE_EMPTY_STREAK", "2"),
            "CALIBRATOR_MIN_TRAIN_RECORDS": os.getenv("CALIBRATOR_MIN_TRAIN_RECORDS", "10"),
            "CALIBRATOR_PATH": os.getenv("CALIBRATOR_PATH", ""),
            "STRICT_SPLIT_LOCK": os.getenv("STRICT_SPLIT_LOCK", "1"),
            "EVAL_STRATIFIED_STRICT": os.getenv("EVAL_STRATIFIED_STRICT", "1"),
            "SIM_GENERATOR_VERSION": os.getenv("SIM_GENERATOR_VERSION", "sim_v3"),
            "SIM_TEMPLATE_DISJOINT_ENFORCE": os.getenv("SIM_TEMPLATE_DISJOINT_ENFORCE", "1"),
            "SIM_HEDGE_RATE": os.getenv("SIM_HEDGE_RATE", "0.65"),
            "SIM_NORMALIZATION_RATE": os.getenv("SIM_NORMALIZATION_RATE", "0.45"),
            "SIM_CONTEXT_ANCHOR_RATE": os.getenv("SIM_CONTEXT_ANCHOR_RATE", "0.55"),
            "SIM_DIRECT_ANSWER_RATE": os.getenv("SIM_DIRECT_ANSWER_RATE", "0.78"),
            "SIM_DEPRESSED_TARGET_BDI": os.getenv("SIM_DEPRESSED_TARGET_BDI", "30"),
            "SIM_DEPRESSED_TARGET_JITTER": os.getenv("SIM_DEPRESSED_TARGET_JITTER", "4"),
            "SIM_DEPRESSED_TARGET_BLEND": os.getenv("SIM_DEPRESSED_TARGET_BLEND", "0.85"),
        },
        "resolved_backends": {
            "auto_enabled": auto_backend_switch_enabled(),
            "detector_backend": resolve_detector_backend(),
            "persona_backend": resolve_persona_backend(),
        },
        "llm_usage": get_llm_usage(),
        "manifest_hash": manifest_hash,
        "git_commit": _current_git_hash(),
    }
    _write_json(output_dir / "config_used.json", config_snapshot)

    return metrics_payload, failure_report_payload, leakage_report_payload, config_snapshot, primary_metrics, error_report_payload
