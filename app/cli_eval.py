from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from core.calibration import clear_calibrator_cache
from core.io_schema import PersonaConversation, PersonaResult
from core.llm import get_llm_usage, reset_llm_usage, set_llm_call_budget
from persona import PersonaProfile, build_split_profiles
from persona.sim_behavior import validate_template_disjointness

from app.cli_common import _parse_bool, _parse_int, _write_json
from app.cli_eval_helpers import (
    _enforce_manifest_lock,
    _manifest_hash,
    _manifest_payload,
    _resolve_effective_eval_mode,
    _resolve_fit_calibrator_policy,
    _split_overlap_count,
    _strict_split_lock_enabled,
    _template_overlap_counts,
)
from app.cli_runtime import _run_profile
from app.cli_runtime_helpers import _assert_openrouter_ready, _print_backend_info, _print_progress, _to_turns, _usage_snippet
from app.eval_artifacts import build_eval_diagnostics_entry, write_eval_artifacts
from app.eval_calibration import fit_calibrator_from_train_profiles


def _result_record(profile: PersonaProfile, state: Dict) -> Dict:
    predicted_label = state.get("predicted_label", "control")
    predicted_bdi = int(state.get("predicted_bdi_score") or 0)
    predicted_symptoms = list(state.get("predicted_key_symptoms") or [])[:4]
    risk_flag = bool(state.get("risk_flag", False))
    final_scores_raw = dict(state.get("final_item_scores", {}))
    item_scores_pred = {
        str(item_id): int(final_scores_raw.get(item_id, final_scores_raw.get(str(item_id), 0)) or 0)
        for item_id in range(1, 22)
    }
    item_scores_true = {str(item_id): int(profile.bdi_scores.get(item_id, 0) or 0) for item_id in range(1, 22)}

    return {
        "llm": profile.persona_id,
        "family": profile.family,
        "split": profile.split,
        "y_true": profile.depressed,
        "y_pred": predicted_label == "depressed",
        "bdi_true": profile.bdi_total,
        "bdi_pred": predicted_bdi,
        "symptoms_true": profile.key_symptoms,
        "symptoms_pred": predicted_symptoms,
        "item_scores_true": item_scores_true,
        "item_scores_pred": item_scores_pred,
        "turns": int(state.get("turn_index", 0)),
        "risk_true": profile.has_risk_signal,
        "risk_pred": risk_flag,
    }


def run_eval(
    persona_count: int,
    seed: int,
    eval_mode: str,
    prompt_version: str,
    save_diagnostics: bool,
    max_api_calls: int,
    trace_level: str,
    fit_calibrator_policy: str,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    from graph import app as graph_app

    verbose_console = _parse_bool(os.getenv("CLI_VERBOSE", "0"))
    live_status = _parse_bool(os.getenv("CLI_LIVE_STATUS", "1"))
    ci_mode = _parse_bool(os.getenv("CI", "0"))
    if ci_mode:
        live_status = False

    os.environ["PROMPT_VERSION"] = prompt_version
    os.environ.pop("CALIBRATOR_PATH", None)
    clear_calibrator_cache()
    _assert_openrouter_ready()
    set_llm_call_budget(max_api_calls if max_api_calls > 0 else None)
    reset_llm_usage()

    fit_calibrator_enabled = _resolve_fit_calibrator_policy(fit_calibrator_policy)
    min_train_records = _parse_int(os.getenv("CALIBRATOR_MIN_TRAIN_RECORDS", "10"), 10)
    strict_split_lock = _strict_split_lock_enabled()
    generator_version = os.getenv("SIM_GENERATOR_VERSION", "sim_v3").strip() or "sim_v3"

    requested_eval_mode, effective_eval_mode = _resolve_effective_eval_mode(eval_mode)

    if verbose_console:
        print(
            f"--- Eval Mode: {requested_eval_mode} | personas={persona_count} | seed={seed} | prompts={prompt_version} ---"
        )
        _print_backend_info(max_api_calls=max_api_calls if max_api_calls > 0 else None, trace_level=trace_level)
        print(
            "Calibrator policy: "
            f"requested={fit_calibrator_policy} | enabled={fit_calibrator_enabled} | min_train_records={min_train_records}"
        )
        print(
            f"Synthetic generator: version={generator_version} | strict_split_lock={'on' if strict_split_lock else 'off'}"
        )
    elif live_status:
        print(
            f"Running eval: mode={requested_eval_mode}, personas={persona_count}, "
            f"prompt={prompt_version}, live_status=on"
        )

    splits = build_split_profiles(count=persona_count, seed=seed)
    train_profiles = splits["synthetic_train"]
    val_profiles = splits["synthetic_val"]
    test_profiles = splits["synthetic_test"]

    manifest_payload = _manifest_payload(
        persona_count=persona_count,
        seed=seed,
        generator_version=generator_version,
        train_profiles=train_profiles,
        val_profiles=val_profiles,
        test_profiles=test_profiles,
    )
    manifest_hash = _manifest_hash(manifest_payload)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "persona_manifest_run_local.json"
    manifest_hash_path = output_dir / "persona_manifest_hash_run_local.txt"

    if strict_split_lock:
        _enforce_manifest_lock(manifest_path, manifest_payload, manifest_hash)

    _write_json(manifest_path, manifest_payload)
    manifest_hash_path.write_text(manifest_hash + "\n", encoding="utf-8")

    split_ids = {
        "train": [profile.persona_id for profile in train_profiles],
        "val": [profile.persona_id for profile in val_profiles],
        "test": [profile.persona_id for profile in test_profiles],
    }
    id_overlap_counts = {
        "train_val": _split_overlap_count(split_ids["train"], split_ids["val"]),
        "train_test": _split_overlap_count(split_ids["train"], split_ids["test"]),
        "val_test": _split_overlap_count(split_ids["val"], split_ids["test"]),
    }
    template_overlap_counts = _template_overlap_counts(train_profiles, val_profiles, test_profiles)
    template_validator = validate_template_disjointness()

    leakage_reasons: List[str] = []
    if any(value > 0 for value in id_overlap_counts.values()):
        leakage_reasons.append("persona_id_overlap_across_splits")
    if any(value > 0 for value in template_overlap_counts.values()):
        leakage_reasons.append("template_bank_overlap_across_splits")
    if not bool(template_validator.get("strict_pass", False)):
        leakage_reasons.append("template_phrase_overlap_detected")

    calibrator_status, calibrator_train_ids, calibrator_failure_counters = fit_calibrator_from_train_profiles(
        train_profiles=train_profiles,
        graph_app=graph_app,
        fit_enabled=fit_calibrator_enabled,
        min_train_records=min_train_records,
        output_dir=output_dir,
        verbose_console=verbose_console,
        live_status=live_status,
    )
    calibrator_status["requested_policy"] = fit_calibrator_policy

    run_failure_counters: Counter[str] = Counter(calibrator_failure_counters)

    conversations: List[PersonaConversation] = []
    results: List[PersonaResult] = []
    diagnostics_payload: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []
    route_distribution: Counter[str] = Counter()
    calibrator_mode_counts: Counter[str] = Counter()
    turns_total = 0
    evidence_turns_nonempty = 0
    evidence_records_total = 0

    eval_profiles = val_profiles + test_profiles
    eval_total = len(eval_profiles)
    processed_profiles = 0
    eval_ids: List[str] = []

    for idx, profile in enumerate(eval_profiles, start=1):
        if verbose_console:
            print(f"\n=== Persona {profile.persona_id} ({profile.source}/{profile.split}/{profile.family}) ===")
        final_state, timeline, style_stats = _run_profile(
            profile,
            graph_app,
            verbose=False,
            progress_prefix=f"[eval {idx}/{eval_total} persona={profile.persona_id}]",
            live_status=live_status,
        )
        processed_profiles += 1
        eval_ids.append(profile.persona_id)

        turns = _to_turns(final_state["messages"])
        conversations.append(PersonaConversation(LLM=profile.persona_id, conversation=turns))
        final_scores = dict(final_state.get("final_item_scores", {}))
        item_scores_map = {
            str(item_id): int(final_scores.get(item_id, final_scores.get(str(item_id), 0)) or 0)
            for item_id in range(1, 22)
        }
        results.append(
            PersonaResult(
                LLM=profile.persona_id,
                bdi_score=int(final_state.get("predicted_bdi_score") or 0),
                key_symptoms=list(final_state.get("predicted_key_symptoms") or [])[:4],
                item_scores=item_scores_map,
            )
        )

        row = _result_record(profile, final_state)
        overall_rows.append(row)
        if profile in val_profiles:
            val_rows.append(row)
        if profile in test_profiles:
            test_rows.append(row)

        diagnostics_payload.append(
            build_eval_diagnostics_entry(
                profile=profile,
                final_state=final_state,
                timeline=timeline,
                style_stats=style_stats,
                trace_level=trace_level,
            )
        )

        for entry in timeline:
            turns_total += 1
            latest_evidence = entry.get("latest_evidence", [])
            ev_count = len(latest_evidence) if isinstance(latest_evidence, list) else 0
            evidence_records_total += ev_count
            if ev_count > 0:
                evidence_turns_nonempty += 1

            route_decision = entry.get("route_decision")
            if isinstance(route_decision, dict):
                chosen = str(route_decision.get("chosen_node", "")).strip()
                if chosen:
                    route_distribution[chosen] += 1

            turn_trace = entry.get("turn_trace", {})
            if isinstance(turn_trace, dict):
                belief_trace = turn_trace.get("update_beliefs", {})
                if isinstance(belief_trace, dict):
                    mode = str(belief_trace.get("calibrator_mode", "")).strip()
                    if mode:
                        calibrator_mode_counts[mode] += 1

        for key, value in dict(final_state.get("failure_counters", {})).items():
            try:
                run_failure_counters[str(key)] += int(value)
            except (TypeError, ValueError):
                continue

        if verbose_console:
            _print_progress(f"Evaluation {_usage_snippet()}", idx, eval_total)
        usage_now = get_llm_usage()
        max_calls_now = usage_now.get("max_calls")
        if max_calls_now is not None and int(usage_now.get("calls_total", 0)) >= int(max_calls_now):
            if verbose_console:
                print("\nStopping eval early: API call budget reached.")
            break

    metrics_payload, failure_report_payload, leakage_report_payload, config_snapshot, primary_metrics, error_report_payload = (
        write_eval_artifacts(
            output_dir=output_dir,
            conversations=conversations,
            results=results,
            diagnostics_payload=diagnostics_payload,
            val_rows=val_rows,
            test_rows=test_rows,
            overall_rows=overall_rows,
            route_distribution=route_distribution,
            calibrator_mode_counts=calibrator_mode_counts,
            turns_total=turns_total,
            evidence_turns_nonempty=evidence_turns_nonempty,
            evidence_records_total=evidence_records_total,
            run_failure_counters=run_failure_counters,
            calibrator_status=calibrator_status,
            id_overlap_counts=id_overlap_counts,
            template_overlap_counts=template_overlap_counts,
            template_validator=template_validator,
            leakage_reasons=leakage_reasons,
            calibrator_train_ids=calibrator_train_ids,
            eval_ids=eval_ids,
            manifest_hash=manifest_hash,
            requested_eval_mode=requested_eval_mode,
            effective_eval_mode=effective_eval_mode,
            prompt_version=prompt_version,
            seed=seed,
            persona_count=persona_count,
            processed_profiles=processed_profiles,
            trace_level=trace_level,
            max_api_calls=max_api_calls,
            save_diagnostics=save_diagnostics,
            fit_calibrator_policy=fit_calibrator_policy,
            train_profiles=train_profiles,
            val_profiles=val_profiles,
            test_profiles=test_profiles,
        )
    )

    if primary_metrics:
        print(
            f"binary_f1={float(primary_metrics.get('binary_f1', 0.0)):.4f} "
            f"objective={float(primary_metrics.get('objective', 0.0)):.4f}"
        )
    else:
        print("binary_f1=0.0000 objective=0.0000")

    if verbose_console:
        print("\n--- Evaluation Summary ---")
        print(metrics_payload)
        print("\nLeakage report:")
        print(leakage_report_payload)
        print("\nWrote:")
        print(f" - {output_dir / 'persona_manifest_run_local.json'}")
        print(f" - {output_dir / 'persona_manifest_hash_run_local.txt'}")
        print(f" - {output_dir / 'interactions_run_local.json'}")
        print(f" - {output_dir / 'results_run_local.json'}")
        print(f" - {output_dir / 'metrics_run_local.json'}")
        print(f" - {output_dir / 'error_report_run_local.json'}")
        print(f" - {output_dir / 'failure_report_run_local.json'}")
        print(f" - {output_dir / 'leakage_report_run_local.json'}")
        if save_diagnostics:
            print(f" - {output_dir / 'diagnostics_run_local.json'}")
        print(f" - {output_dir / 'config_used.json'}")

    return {
        "metrics": metrics_payload,
        "failure_report": failure_report_payload,
        "error_report": error_report_payload,
        "leakage_report": leakage_report_payload,
        "config": config_snapshot,
        "output_dir": str(output_dir),
        "profiles_evaluated": processed_profiles,
        "expected_eval_profiles": len(eval_profiles),
        "route_distribution": dict(route_distribution),
        "failure_counters": dict(run_failure_counters),
    }
