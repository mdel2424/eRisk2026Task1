from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from core.io_schema import PersonaConversation, PersonaResult
from core.llm import get_llm_usage, reset_llm_usage, set_llm_call_budget
from persona import PersonaProfile, generate_persona_pool

from app.cli_common import _parse_bool, _write_json
from app.cli_eval_helpers import (
    _manifest_hash,
    _manifest_payload,
    _load_previous_manifest_info,
)
from app.cli_runtime import _run_profile
from app.cli_runtime_helpers import (
    _assert_detector_backend_ready,
    _print_backend_info,
    _print_progress,
    _to_turns,
    _usage_snippet,
)
from app.eval_artifacts import build_eval_diagnostics_entry, write_eval_artifacts


def _result_record(profile: PersonaProfile, state: Dict) -> Dict:
    predicted_bdi = int(state.get("predicted_bdi_score") or 0)
    predicted_symptoms = list(state.get("predicted_key_symptoms") or [])[:4]
    final_scores_raw = dict(state.get("final_item_scores", {}))
    item_scores_pred = {
        str(item_id): int(final_scores_raw.get(item_id, final_scores_raw.get(str(item_id), 0)) or 0)
        for item_id in range(1, 22)
    }
    item_scores_true = {str(item_id): int(profile.bdi_scores.get(item_id, 0) or 0) for item_id in range(1, 22)}

    return {
        "llm": profile.persona_id,
        "family": profile.family,
        "severity_tier": profile.severity_tier,
        "subtype_tag": profile.subtype_tag,
        "context_tag": profile.context_tag,
        "style_tag": profile.style_tag,
        "split": profile.split,
        "bdi_true": profile.bdi_total,
        "bdi_pred": predicted_bdi,
        "symptoms_true": profile.key_symptoms,
        "symptoms_pred": predicted_symptoms,
        "item_scores_true": item_scores_true,
        "item_scores_pred": item_scores_pred,
        "turns": int(state.get("turn_index", 0)),
    }


def _runtime_summary_record(state: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = state.get("diagnosis")
    judgment = state.get("judgment")
    bayes_nodes = dict(state.get("bayes_nodes", {}) or {})
    return {
        "diagnosis": {
            "confidence": float(getattr(diagnosis, "confidence", 0.0) or 0.0),
            "used_llm": bool(getattr(diagnosis, "used_llm", False)),
            "synthesis_mode": str(getattr(diagnosis, "synthesis_mode", "") or ""),
            "supported_item_count": int(
                sum(
                    1
                    for value in dict(getattr(diagnosis, "item_scores", {}) or {}).values()
                    if int(value) > 0
                )
            ),
        },
        "judgment": {
            "active_cluster": str(getattr(judgment, "active_cluster", "") or ""),
            "evidence_binding_coverage": float(getattr(judgment, "evidence_binding_coverage", 1.0) or 1.0),
            "bound_positive_assertion_count": int(getattr(judgment, "bound_positive_assertion_count", 0) or 0),
            "emitted_evidence_count": int(getattr(judgment, "emitted_evidence_count", 0) or 0),
            "opening_bootstrap_applied": bool(getattr(judgment, "opening_bootstrap_applied", False)),
            "opening_bootstrap_cluster": str(getattr(judgment, "opening_bootstrap_cluster", "") or ""),
            "opening_bootstrap_item_ids": [
                int(item_id)
                for item_id in list(getattr(judgment, "opening_bootstrap_item_ids", []) or [])
                if 1 <= int(item_id) <= 21
            ],
        },
        "navigation": {
            "opening_signal_cluster": str(state.get("opening_signal_cluster", "") or ""),
            "opening_signal_item_ids": [
                int(item_id)
                for item_id in list(state.get("opening_signal_item_ids", []) or [])
                if 1 <= int(item_id) <= 21
            ],
            "opening_followup_cluster": str(state.get("opening_followup_cluster", "") or ""),
            "opening_cognitive_anchor_preserved": bool(state.get("opening_cognitive_anchor_preserved", False)),
        },
        "bayes": {
            "node_posteriors": {
                str(node_id): round(float(getattr(node_state, "probability", 0.0) or 0.0), 4)
                for node_id, node_state in bayes_nodes.items()
            }
        },
    }


def run_eval(
    persona_count: int,
    seed: int,
    save_diagnostics: bool,
    max_api_calls: int,
    trace_level: str,
    debug_outputs: bool = False,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    from graph import app as graph_app

    verbose_console = _parse_bool(os.getenv("CLI_VERBOSE", "0"))
    live_status = _parse_bool(os.getenv("CLI_LIVE_STATUS", "1"))
    ci_mode = _parse_bool(os.getenv("CI", "0"))
    if ci_mode:
        live_status = False

    _assert_detector_backend_ready()
    set_llm_call_budget(max_api_calls if max_api_calls > 0 else None)
    reset_llm_usage()

    stop_policy = {
        "MIN_TURNS": os.getenv("MIN_TURNS", "20"),
        "MAX_TURNS": os.getenv("MAX_TURNS", "40"),
        "STOP_CONFIDENCE": os.getenv("STOP_CONFIDENCE", "0.66"),
    }
    runtime_controls = {
        "DIAGNOSIS_AGENT_USE_LLM": os.getenv("DIAGNOSIS_AGENT_USE_LLM", "0"),
        "DETERMINISTIC_BDI_LABEL_THRESHOLD": os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"),
        "DETECTOR_EXTRACTOR_MAX_NEW_TOKENS": os.getenv(
            "DETECTOR_EXTRACTOR_MAX_NEW_TOKENS",
            os.getenv("DETECTOR_MAX_NEW_TOKENS", "96"),
        ),
    }

    requested_trace_level = str(trace_level).strip().lower()
    requested_save_diagnostics = bool(save_diagnostics)
    effective_debug_outputs = bool(debug_outputs) or requested_save_diagnostics or (requested_trace_level == "compact")
    trace_level_effective = requested_trace_level if effective_debug_outputs else "off"
    save_diagnostics_effective = requested_save_diagnostics if effective_debug_outputs else False
    run_profile = "debug" if effective_debug_outputs else "lean"

    if verbose_console and effective_debug_outputs:
        print(f"--- Synthetic Eval | personas={persona_count} | seed={seed} ---")
        _print_backend_info(max_api_calls=max_api_calls if max_api_calls > 0 else None, trace_level=trace_level_effective)
        print(
            "Stop policy: "
            + " | ".join(f"{key}={value}" for key, value in stop_policy.items())
        )
        print(
            "Runtime controls: "
            + " | ".join(f"{key}={value}" for key, value in runtime_controls.items())
        )
    elif live_status:
        print(f"Running synthetic eval: personas={persona_count}, live_status=on")
        if effective_debug_outputs:
            print(
                "Stop policy: "
                + " | ".join(f"{key}={value}" for key, value in stop_policy.items())
            )
            print(
                "Runtime controls: "
                + " | ".join(f"{key}={value}" for key, value in runtime_controls.items())
            )

    all_profiles = generate_persona_pool(count=persona_count, seed=seed)

    manifest_payload = _manifest_payload(
        persona_count=persona_count,
        seed=seed,
        profiles=all_profiles,
    )
    manifest_hash = _manifest_hash(manifest_payload)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "persona_manifest_run_local.json"
    manifest_hash_path = output_dir / "persona_manifest_hash_run_local.txt"
    prior_manifest_info = _load_previous_manifest_info(manifest_path)

    _write_json(manifest_path, manifest_payload)
    manifest_hash_path.write_text(manifest_hash + "\n", encoding="utf-8")

    run_failure_counters: Counter[str] = Counter()

    conversations: List[PersonaConversation] = []
    results: List[PersonaResult] = []
    diagnostics_payload: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []
    route_distribution: Counter[str] = Counter()
    route_policy_distribution: Counter[str] = Counter()
    extract_source_distribution: Counter[str] = Counter()
    extract_recovery_distribution: Counter[str] = Counter()
    extract_parse_fail_log_entries: List[Dict[str, Any]] = []
    duplicate_evidence_rows_total = 0
    contradiction_evidence_rows_total = 0
    support_increments_total = 0
    method_weight_usage: Counter[str] = Counter()
    turns_total = 0
    evidence_turns_nonempty = 0
    evidence_records_total = 0
    min_turns_for_productivity = max(1, int(os.getenv("MIN_TURNS", "20")))
    post_floor_new_items_total = 0
    post_floor_nonempty_turns_total = 0
    post_floor_turns_total = 0
    early_stop_reason_distribution: Counter[str] = Counter()

    eval_profiles = all_profiles
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
        item_beliefs = final_state.get("item_beliefs", {})
        item_support_map = {
            str(item_id): int(getattr(item_beliefs.get(item_id), "support_count", 0) or 0)
            for item_id in range(1, 22)
        }
        results.append(
            PersonaResult(
                LLM=profile.persona_id,
                bdi_score=int(final_state.get("predicted_bdi_score") or 0),
                key_symptoms=list(final_state.get("predicted_key_symptoms") or [])[:4],
                item_scores=item_scores_map,
                item_support_counts=item_support_map,
                runtime_summary=_runtime_summary_record(final_state),
            )
        )

        row = _result_record(profile, final_state)
        overall_rows.append(row)

        if effective_debug_outputs:
            diagnostics_payload.append(
                build_eval_diagnostics_entry(
                    profile=profile,
                    final_state=final_state,
                    timeline=timeline,
                    style_stats=style_stats,
                    trace_level=trace_level_effective,
                )
            )
        stop_history = list(final_state.get("stop_history", []))
        if stop_history:
            latest_stop = stop_history[-1]
            if isinstance(latest_stop, dict):
                stop_reason = str(latest_stop.get("reason", "")).strip()
            else:
                stop_reason = str(getattr(latest_stop, "reason", "")).strip()
            if stop_reason:
                early_stop_reason_distribution[stop_reason] += 1

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
                policy = str(route_decision.get("policy", "")).strip()
                if policy:
                    route_policy_distribution[policy] += 1

            turn_trace = entry.get("turn_trace", {})
            if isinstance(turn_trace, dict):
                judgment_trace = turn_trace.get("judgment_agent", {})
                bayes_trace = turn_trace.get("bayes_state_update", {})
                if isinstance(judgment_trace, dict):
                    source = str(judgment_trace.get("source", "")).strip() or "judgment_agent"
                    extract_source_distribution[source] += 1
                    if bool(judgment_trace.get("salvage_used", False)):
                        extract_recovery_distribution["salvage"] += 1
                    if int(judgment_trace.get("key_alias_used_count", 0) or 0) > 0:
                        extract_recovery_distribution["key_alias"] += int(judgment_trace.get("key_alias_used_count", 0) or 0)
                    if int(judgment_trace.get("schema_coerce_used_count", 0) or 0) > 0:
                        extract_recovery_distribution["schema_coerce"] += int(judgment_trace.get("schema_coerce_used_count", 0) or 0)
                    support_increments_total += int(judgment_trace.get("bound_positive_assertion_count", 0) or 0)
                    if bool(judgment_trace.get("llm_called", False)) and int(judgment_trace.get("kept_items_count", 0) or 0) <= 0:
                        extract_parse_fail_log_entries.append(
                            {
                                "llm": profile.persona_id,
                                "split": profile.split,
                                "family": profile.family,
                                "turn": int(entry.get("turn", judgment_trace.get("turn", 0)) or 0),
                                "source": source,
                                "failure_reason": str(judgment_trace.get("parse_error_kind", "") or "no_bound_evidence"),
                                "counts_as_failure": bool(not judgment_trace.get("genuine_no_signal_turn", False)),
                                "parse_error_kind": str(judgment_trace.get("parse_error_kind", "") or ""),
                                "parse_error_message": str(judgment_trace.get("parse_error_message", "") or ""),
                                "target_item_id": int(judgment_trace.get("target_item_id", 0) or 0),
                                "target_module_id": int(judgment_trace.get("target_module_id", 0) or 0),
                                "allowed_item_ids": list(judgment_trace.get("allowed_item_ids", []) or []),
                                "route_node": (
                                    str(route_decision.get("chosen_node", "")).strip()
                                    if isinstance(route_decision, dict)
                                    else ""
                                ),
                                "route_policy": (
                                    str(route_decision.get("policy", "")).strip()
                                    if isinstance(route_decision, dict)
                                    else ""
                                ),
                            }
                        )
                    if bool(judgment_trace.get("has_new_persona_input", False)):
                        effective_turn = int(judgment_trace.get("turn", entry.get("turn", 0)) or 0)
                        if effective_turn >= min_turns_for_productivity:
                            post_floor_turns_total += 1
                            if ev_count > 0:
                                post_floor_nonempty_turns_total += 1
                            post_floor_new_items_total += max(0, int(judgment_trace.get("bound_positive_assertion_count", 0) or 0))
                if isinstance(bayes_trace, dict):
                    method_weight_usage["noisy_or_bayes_update"] += 1

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

    metrics_payload, failure_report_payload, benchmark_integrity_payload, config_snapshot, primary_metrics, error_report_payload = (
        write_eval_artifacts(
            output_dir=output_dir,
            conversations=conversations,
            results=results,
            diagnostics_payload=diagnostics_payload,
            overall_rows=overall_rows,
            route_distribution=route_distribution,
            turns_total=turns_total,
            evidence_turns_nonempty=evidence_turns_nonempty,
            evidence_records_total=evidence_records_total,
            extract_source_distribution=extract_source_distribution,
            extract_recovery_distribution=extract_recovery_distribution,
            route_policy_distribution=route_policy_distribution,
            duplicate_evidence_rows_total=duplicate_evidence_rows_total,
            contradiction_evidence_rows_total=contradiction_evidence_rows_total,
            support_increments_total=support_increments_total,
            method_weight_usage=method_weight_usage,
            post_floor_new_items_total=post_floor_new_items_total,
            post_floor_nonempty_turns_total=post_floor_nonempty_turns_total,
            post_floor_turns_total=post_floor_turns_total,
            min_turns_for_productivity=min_turns_for_productivity,
            early_stop_reason_distribution=early_stop_reason_distribution,
            extract_parse_fail_log_entries=extract_parse_fail_log_entries,
            run_failure_counters=run_failure_counters,
            eval_ids=eval_ids,
            manifest_hash=manifest_hash,
            manifest_payload=manifest_payload,
            prior_manifest_info=prior_manifest_info,
            seed=seed,
            persona_count=persona_count,
            processed_profiles=processed_profiles,
            trace_level=trace_level_effective,
            max_api_calls=max_api_calls,
            save_diagnostics=save_diagnostics_effective,
            debug_outputs=effective_debug_outputs,
            run_profile=run_profile,
            requested_save_diagnostics=requested_save_diagnostics,
            requested_trace_level=requested_trace_level,
            requested_debug_outputs=bool(debug_outputs),
            all_profiles=all_profiles,
        )
    )

    if primary_metrics:
        print(
            f"item_f1={float(primary_metrics.get('item_f1_macro_at_1', 0.0)):.4f} "
            f"objective={float(primary_metrics.get('objective', 0.0)):.4f}"
        )
    else:
        print("item_f1=0.0000 objective=0.0000")

    if verbose_console:
        print("\n--- Evaluation Summary ---")
        print(metrics_payload)
        print("\nBenchmark integrity:")
        print(benchmark_integrity_payload)
        print("\nWrote:")
        print(f" - {output_dir / 'persona_manifest_run_local.json'}")
        print(f" - {output_dir / 'persona_manifest_hash_run_local.txt'}")
        print(f" - {output_dir / 'interactions_run_local.json'}")
        print(f" - {output_dir / 'results_run_local.json'}")
        print(f" - {output_dir / 'metrics_run_local.json'}")
        if effective_debug_outputs:
            print(f" - {output_dir / 'error_report_run_local.json'}")
            print(f" - {output_dir / 'extract_parse_fail_log_run_local.json'}")
            print(f" - {output_dir / 'failure_report_run_local.json'}")
        print(f" - {output_dir / 'benchmark_integrity_run_local.json'}")
        if save_diagnostics_effective:
            print(f" - {output_dir / 'diagnostics_run_local.json'}")
        print(f" - {output_dir / 'config_used.json'}")

    return {
        "metrics": metrics_payload,
        "failure_report": failure_report_payload,
        "error_report": error_report_payload,
        "benchmark_integrity": benchmark_integrity_payload,
        "config": config_snapshot,
        "output_dir": str(output_dir),
        "profiles_evaluated": processed_profiles,
        "expected_eval_profiles": len(eval_profiles),
        "route_distribution": dict(route_distribution),
        "failure_counters": dict(run_failure_counters),
    }
