from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from core.llm import get_llm_usage
from core.runtime_policy import auto_backend_switch_enabled, resolve_detector_backend, resolve_persona_backend

from app.cli_common import _current_git_hash, _serialize, _write_json
from app.cli_eval_helpers import _family_summary, _profile_meta, _select_primary_metrics, _with_objective


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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, float]]:
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
        "calibrator_status": calibrator_status,
        "llm_usage": get_llm_usage(),
    }
    if primary_metrics:
        metrics_payload.update(primary_metrics)

    interactions_payload = [conv.model_dump() for conv in conversations]
    results_payload = [result.to_erisk_dict() for result in results]
    _write_json(output_dir / "interactions_run_local.json", interactions_payload)
    _write_json(output_dir / "results_run_local.json", results_payload)
    _write_json(output_dir / "metrics_run_local.json", metrics_payload)

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
            "PERSONA_BACKEND": os.getenv("PERSONA_BACKEND", "hf_adapter"),
            "OPENROUTER_PERSONA_MODEL": os.getenv("OPENROUTER_PERSONA_MODEL", ""),
            "ERISK_BASE_MODEL": os.getenv("ERISK_BASE_MODEL", ""),
            "ERISK_ADAPTER_ID": os.getenv("ERISK_ADAPTER_ID", ""),
            "ERISK_ADAPTER_TEMPLATE": os.getenv("ERISK_ADAPTER_TEMPLATE", ""),
            "MIN_CUDA_VRAM_GB": os.getenv("MIN_CUDA_VRAM_GB", "8"),
            "MIN_TURNS": os.getenv("MIN_TURNS", ""),
            "MAX_TURNS": os.getenv("MAX_TURNS", ""),
            "STOP_CONFIDENCE": os.getenv("STOP_CONFIDENCE", ""),
            "MIN_EVIDENCE_FOR_CONF_STOP": os.getenv("MIN_EVIDENCE_FOR_CONF_STOP", "2"),
            "DETERMINISTIC_BDI_LABEL_THRESHOLD": os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"),
            "DETERMINISTIC_CORE_ITEM_MEAN_THRESHOLD": os.getenv("DETERMINISTIC_CORE_ITEM_MEAN_THRESHOLD", "0.6"),
            "DETERMINISTIC_CORE_ITEM_MIN_HITS": os.getenv("DETERMINISTIC_CORE_ITEM_MIN_HITS", "4"),
            "EVIDENCE_LLM_ON_LEXICAL_HIT": os.getenv("EVIDENCE_LLM_ON_LEXICAL_HIT", "0"),
            "SUPERVISOR_EVIDENCE_MIN_SCORE": os.getenv("SUPERVISOR_EVIDENCE_MIN_SCORE", "0.30"),
            "SUPERVISOR_EVIDENCE_RISK_THRESHOLD": os.getenv("SUPERVISOR_EVIDENCE_RISK_THRESHOLD", "0.22"),
            "SUPERVISOR_ESCAPE_EMPTY_STREAK": os.getenv("SUPERVISOR_ESCAPE_EMPTY_STREAK", "2"),
            "CALIBRATOR_MIN_TRAIN_RECORDS": os.getenv("CALIBRATOR_MIN_TRAIN_RECORDS", "10"),
            "CALIBRATOR_PATH": os.getenv("CALIBRATOR_PATH", ""),
            "STRICT_SPLIT_LOCK": os.getenv("STRICT_SPLIT_LOCK", "1"),
            "SIM_GENERATOR_VERSION": os.getenv("SIM_GENERATOR_VERSION", "sim_v2"),
            "SIM_PARAPHRASE_ENABLED": os.getenv("SIM_PARAPHRASE_ENABLED", "1"),
            "SIM_PARAPHRASE_RATE": os.getenv("SIM_PARAPHRASE_RATE", "0.5"),
            "SIM_TEMPLATE_DISJOINT_ENFORCE": os.getenv("SIM_TEMPLATE_DISJOINT_ENFORCE", "1"),
            "SIM_HEDGE_RATE": os.getenv("SIM_HEDGE_RATE", "0.65"),
            "SIM_NORMALIZATION_RATE": os.getenv("SIM_NORMALIZATION_RATE", "0.45"),
            "SIM_CONTEXT_ANCHOR_RATE": os.getenv("SIM_CONTEXT_ANCHOR_RATE", "0.55"),
            "SIM_DIRECT_ANSWER_RATE": os.getenv("SIM_DIRECT_ANSWER_RATE", "0.78"),
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

    return metrics_payload, failure_report_payload, leakage_report_payload, config_snapshot, primary_metrics
