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
from app.finalizer_summary import compact_finalizer_summary
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
    confidence_model = {
        "CONF_SUPPORT_TAU": os.getenv("CONF_SUPPORT_TAU", "1.25"),
        "CONF_DEPTH_WEIGHT": os.getenv("CONF_DEPTH_WEIGHT", "0.70"),
        "CONF_COVERAGE_WEIGHT": os.getenv("CONF_COVERAGE_WEIGHT", "0.30"),
        "CONF_UP_ALPHA": os.getenv("CONF_UP_ALPHA", "0.55"),
        "CONF_DECAY_STREAK_START": os.getenv("CONF_DECAY_STREAK_START", "6"),
        "CONF_DECAY_PER_TURN": os.getenv("CONF_DECAY_PER_TURN", "0.002"),
        "CONF_DECAY_MAX": os.getenv("CONF_DECAY_MAX", "0.01"),
        "CONF_MAX_DROP_PER_TURN": os.getenv("CONF_MAX_DROP_PER_TURN", "0.01"),
    }
    risk_policy = {
        "RISK_SENTINEL_FLAG_THRESHOLD": os.getenv("RISK_SENTINEL_FLAG_THRESHOLD", "0.45"),
        "RISK_SENTINEL_SHORTCIRCUIT_THRESHOLD": os.getenv("RISK_SENTINEL_SHORTCIRCUIT_THRESHOLD", "1.1"),
        "RISK_SENTINEL_ACTIVE_SHORTCIRCUIT": os.getenv("RISK_SENTINEL_ACTIVE_SHORTCIRCUIT", "0"),
        "EXTRACTOR_MIN_RECORDS_TARGET": os.getenv("EXTRACTOR_MIN_RECORDS_TARGET", "1"),
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
            "Confidence model: "
            + " | ".join(f"{key}={value}" for key, value in confidence_model.items())
        )
        print(
            "Risk/extractor controls: "
            + " | ".join(f"{key}={value}" for key, value in risk_policy.items())
        )
    elif live_status:
        print(f"Running synthetic eval: personas={persona_count}, live_status=on")
        if effective_debug_outputs:
            print(
                "Stop policy: "
                + " | ".join(f"{key}={value}" for key, value in stop_policy.items())
            )
            print(
                "Confidence model: "
                + " | ".join(f"{key}={value}" for key, value in confidence_model.items())
            )
            print(
                "Risk/extractor controls: "
                + " | ".join(f"{key}={value}" for key, value in risk_policy.items())
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
                finalizer_summary=compact_finalizer_summary(final_state.get("module_imputation")),
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
                belief_trace = turn_trace.get("belief_update")
                if not isinstance(belief_trace, dict):
                    belief_trace = turn_trace.get("update_beliefs", {})
                if isinstance(belief_trace, dict):
                    duplicate_evidence_rows_total += int(belief_trace.get("duplicate_rows_count", 0) or 0)
                    contradiction_evidence_rows_total += int(belief_trace.get("contradiction_rows_count", 0) or 0)
                    support_increments_total += int(belief_trace.get("support_increments_count", 0) or 0)
                    method_counts = belief_trace.get("method_counts", {})
                    if isinstance(method_counts, dict):
                        for method_name, count in method_counts.items():
                            try:
                                method_weight_usage[str(method_name)] += int(count)
                            except (TypeError, ValueError):
                                continue
                extract_trace = turn_trace.get("extract_likelihoods")
                if not isinstance(extract_trace, dict):
                    extract_trace = turn_trace.get("extract_evidence", {})
                if isinstance(extract_trace, dict):
                    source = str(extract_trace.get("source", "")).strip()
                    if source:
                        extract_source_distribution[source] += 1
                    if bool(extract_trace.get("salvage_used", False)):
                        extract_recovery_distribution["salvage"] += 1
                    alias_used = int(extract_trace.get("key_alias_used_count", 0) or 0)
                    schema_used = int(extract_trace.get("schema_coerce_used_count", 0) or 0)
                    if alias_used > 0:
                        extract_recovery_distribution["key_alias"] += alias_used
                    if schema_used > 0:
                        extract_recovery_distribution["schema_coerce"] += schema_used

                    llm_called = bool(extract_trace.get("llm_called", False))
                    raw_nonempty = bool(extract_trace.get("raw_nonempty", False))
                    kept_items_count = int(extract_trace.get("kept_items_count", 0) or 0)
                    opportunistic_kept_items_count = int(extract_trace.get("opportunistic_kept_items_count", 0) or 0)
                    support_increments_this_turn = 0
                    if isinstance(belief_trace, dict):
                        support_increments_this_turn = int(belief_trace.get("support_increments_count", 0) or 0)
                    should_log_extract_event = llm_called and (
                        kept_items_count == 0
                        or (opportunistic_kept_items_count > 0 and support_increments_this_turn == 0)
                    )
                    if should_log_extract_event:
                        dropped_unknown_item_count = int(extract_trace.get("drop_unknown_item_count", 0) or 0)
                        dropped_invalid_range_count = int(extract_trace.get("drop_invalid_range_count", 0) or 0)
                        detail_scored_item_count = int(extract_trace.get("detail_scored_item_count", 0) or 0)
                        detail_supported_item_count = int(extract_trace.get("detail_supported_item_count", 0) or 0)
                        detail_missing_allowed_item_count = int(
                            extract_trace.get("detail_missing_allowed_item_count", 0) or 0
                        )
                        detail_supported_rows_dropped_by_item1 = int(
                            extract_trace.get("detail_supported_rows_dropped_by_item1", 0) or 0
                        )
                        detail_supported_rows_dropped_by_item9 = int(
                            extract_trace.get("detail_supported_rows_dropped_by_item9", 0) or 0
                        )
                        detail_supported_rows_kept_post_validation = int(
                            extract_trace.get("detail_supported_rows_kept_post_validation", 0) or 0
                        )
                        genuine_no_signal_turn = bool(extract_trace.get("genuine_no_signal_turn", False))
                        json_parse_ok = bool(extract_trace.get("json_parse_ok", False))
                        parse_error_kind = str(extract_trace.get("parse_error_kind", "") or "")
                        parse_error_message = str(extract_trace.get("parse_error_message", "") or "")
                        opportunistic_called = bool(extract_trace.get("opportunistic_called", False))
                        opportunistic_shortlist_called = bool(
                            extract_trace.get("opportunistic_shortlist_called", False)
                        )
                        opportunistic_shortlist_parse_ok = bool(
                            extract_trace.get("opportunistic_shortlist_parse_ok", False)
                        )
                        opportunistic_has_strong_offtarget_signal = bool(
                            extract_trace.get("opportunistic_has_strong_offtarget_signal", False)
                        )
                        opportunistic_score_called = bool(extract_trace.get("opportunistic_score_called", False))
                        opportunistic_score_parse_ok = bool(
                            extract_trace.get("opportunistic_score_parse_ok", False)
                        )
                        opportunistic_supported_item_count = int(
                            extract_trace.get("opportunistic_supported_item_count", 0) or 0
                        )
                        counts_as_failure = True
                        if opportunistic_kept_items_count > 0 and support_increments_this_turn == 0:
                            failure_reason = "opportunistic_kept_but_no_support_increment"
                        elif genuine_no_signal_turn and kept_items_count == 0:
                            failure_reason = "genuine_no_signal_all_unsupported"
                            counts_as_failure = False
                        elif not json_parse_ok:
                            failure_reason = parse_error_kind or "invalid_or_truncated_json"
                        elif detail_supported_item_count > 0 and kept_items_count == 0:
                            if detail_supported_rows_dropped_by_item9 > 0:
                                failure_reason = "scorer_supported_items_dropped_by_item9"
                            elif detail_supported_rows_dropped_by_item1 > 0:
                                failure_reason = "scorer_supported_items_dropped_by_item1"
                            else:
                                failure_reason = "scorer_supported_items_dropped_post_validation"
                        elif opportunistic_shortlist_called and not opportunistic_shortlist_parse_ok:
                            failure_reason = "opportunistic_shortlist_parse_fail"
                        elif (
                            opportunistic_shortlist_called
                            and opportunistic_shortlist_parse_ok
                            and not opportunistic_has_strong_offtarget_signal
                        ):
                            failure_reason = "opportunistic_shortlist_no_candidates"
                        elif opportunistic_score_called and not opportunistic_score_parse_ok:
                            failure_reason = "opportunistic_score_parse_fail"
                        elif opportunistic_supported_item_count > 0 and opportunistic_kept_items_count == 0:
                            if int(extract_trace.get("opportunistic_dropped_weak_count", 0) or 0) > 0:
                                failure_reason = "opportunistic_supported_rows_dropped_by_strong_only_threshold"
                            else:
                                failure_reason = "opportunistic_supported_items_dropped_post_validation"
                        elif detail_missing_allowed_item_count > 0 and kept_items_count == 0:
                            failure_reason = "scorer_partial_allowed_item_coverage"
                        elif detail_scored_item_count > 0 and detail_supported_item_count == 0 and opportunistic_called:
                            failure_reason = "scorer_all_unsupported_then_opportunistic_empty"
                        elif detail_scored_item_count > 0 and detail_supported_item_count == 0:
                            failure_reason = "scorer_all_unsupported"
                        elif opportunistic_called and kept_items_count == 0:
                            failure_reason = "scoped_empty_then_opportunistic_empty"
                        elif kept_items_count == 0:
                            failure_reason = "scoped_empty_only"
                        elif dropped_unknown_item_count > 0 and dropped_invalid_range_count == 0:
                            failure_reason = "unknown_item_mapping"
                        elif dropped_invalid_range_count > 0:
                            failure_reason = "invalid_intensity_or_confidence_range"
                        else:
                            failure_reason = "parsed_but_no_usable_evidence"
                        extract_parse_fail_log_entries.append(
                            {
                                "llm": profile.persona_id,
                                "split": profile.split,
                                "family": profile.family,
                                "turn": int(entry.get("turn", extract_trace.get("turn", 0)) or 0),
                                "source": source or "llm_extractor",
                                "failure_reason": failure_reason,
                                "counts_as_failure": counts_as_failure,
                                "json_parse_ok": json_parse_ok,
                                "parse_error_kind": parse_error_kind,
                                "parse_error_message": parse_error_message,
                                "parse_error_line": int(extract_trace.get("parse_error_line", 0) or 0),
                                "parse_error_column": int(extract_trace.get("parse_error_column", 0) or 0),
                                "parse_error_position": int(extract_trace.get("parse_error_position", 0) or 0),
                                "parse_fail_stage": str(extract_trace.get("parse_fail_stage", "") or ""),
                                "parse_balance": extract_trace.get("parse_balance", {}),
                                "raw_items_count": int(extract_trace.get("raw_items_count", 0) or 0),
                                "kept_items_count": kept_items_count,
                                "drop_unknown_item_count": dropped_unknown_item_count,
                                "drop_invalid_range_count": dropped_invalid_range_count,
                                "target_item_id": int(extract_trace.get("target_item_id", 0) or 0),
                                "target_module_id": int(extract_trace.get("target_module_id", 0) or 0),
                                "allowed_item_ids": list(extract_trace.get("allowed_item_ids", []) or []),
                                "gate_candidate_item_ids": list(extract_trace.get("gate_candidate_item_ids", []) or []),
                                "gate_called": bool(extract_trace.get("gate_called", False)),
                                "gate_parse_ok": bool(extract_trace.get("gate_parse_ok", False)),
                                "gate_target_relevant": bool(extract_trace.get("gate_target_relevant", False)),
                                "stage2_called": bool(extract_trace.get("stage2_called", False)),
                                "stage2_parse_ok": bool(extract_trace.get("stage2_parse_ok", False)),
                                "clear_no_symptom_skip": bool(extract_trace.get("clear_no_symptom_skip", False)),
                                "genuine_no_signal_turn": genuine_no_signal_turn,
                                "gate_soft_false_overridden": bool(
                                    extract_trace.get("gate_soft_false_overridden", False)
                                ),
                                "detail_called_due_to_gate_false": bool(
                                    extract_trace.get("detail_called_due_to_gate_false", False)
                                ),
                                "detail_called_due_to_gate_parse_fail": bool(
                                    extract_trace.get("detail_called_due_to_gate_parse_fail", False)
                                ),
                                "detail_empty_after_gate_true": bool(
                                    extract_trace.get("detail_empty_after_gate_true", False)
                                ),
                                "detail_empty_after_gate_false": bool(
                                    extract_trace.get("detail_empty_after_gate_false", False)
                                ),
                                "detail_scored_item_count": detail_scored_item_count,
                                "detail_supported_item_count": detail_supported_item_count,
                                "detail_unsupported_item_count": int(
                                    extract_trace.get("detail_unsupported_item_count", 0) or 0
                                ),
                                "detail_supported_item_ids": list(
                                    extract_trace.get("detail_supported_item_ids", []) or []
                                ),
                                "detail_missing_allowed_item_count": detail_missing_allowed_item_count,
                                "detail_supported_rows_dropped_by_item1": detail_supported_rows_dropped_by_item1,
                                "detail_supported_rows_dropped_by_item9": detail_supported_rows_dropped_by_item9,
                                "detail_supported_rows_kept_post_validation": (
                                    detail_supported_rows_kept_post_validation
                                ),
                                "opportunistic_called": opportunistic_called,
                                "opportunistic_skipped_on_risk": bool(
                                    extract_trace.get("opportunistic_skipped_on_risk", False)
                                ),
                                "opportunistic_shortlist_called": opportunistic_shortlist_called,
                                "opportunistic_shortlist_parse_ok": opportunistic_shortlist_parse_ok,
                                "opportunistic_has_strong_offtarget_signal": (
                                    opportunistic_has_strong_offtarget_signal
                                ),
                                "opportunistic_candidate_item_ids": list(
                                    extract_trace.get("opportunistic_candidate_item_ids", []) or []
                                ),
                                "opportunistic_score_called": opportunistic_score_called,
                                "opportunistic_score_parse_ok": opportunistic_score_parse_ok,
                                "opportunistic_parse_ok": bool(extract_trace.get("opportunistic_parse_ok", False)),
                                "opportunistic_raw_items_count": int(
                                    extract_trace.get("opportunistic_raw_items_count", 0) or 0
                                ),
                                "opportunistic_kept_items_count": opportunistic_kept_items_count,
                                "opportunistic_scored_item_count": int(
                                    extract_trace.get("opportunistic_scored_item_count", 0) or 0
                                ),
                                "opportunistic_supported_item_count": opportunistic_supported_item_count,
                                "opportunistic_unsupported_item_count": int(
                                    extract_trace.get("opportunistic_unsupported_item_count", 0) or 0
                                ),
                                "opportunistic_supported_item_ids": list(
                                    extract_trace.get("opportunistic_supported_item_ids", []) or []
                                ),
                                "opportunistic_missing_item_count": int(
                                    extract_trace.get("opportunistic_missing_item_count", 0) or 0
                                ),
                                "opportunistic_dropped_weak_count": int(
                                    extract_trace.get("opportunistic_dropped_weak_count", 0) or 0
                                ),
                                "opportunistic_salvage_used": bool(
                                    extract_trace.get("opportunistic_salvage_used", False)
                                ),
                                "opportunistic_item_ids": list(
                                    extract_trace.get("opportunistic_item_ids", []) or []
                                ),
                                "support_increments_this_turn": support_increments_this_turn,
                                "key_alias_used_count": int(extract_trace.get("key_alias_used_count", 0) or 0),
                                "schema_coerce_used_count": int(extract_trace.get("schema_coerce_used_count", 0) or 0),
                                "salvage_used": bool(extract_trace.get("salvage_used", False)),
                                "fallback_used": bool(extract_trace.get("fallback_used", False)),
                                "raw_extractor_payload": str(extract_trace.get("raw_extractor_payload", "") or ""),
                                "latest_message": str(extract_trace.get("latest_message", "") or ""),
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

                    if bool(extract_trace.get("has_new_persona_input", False)):
                        effective_turn = int(extract_trace.get("turn", entry.get("turn", 0)) or 0)
                        if effective_turn >= min_turns_for_productivity:
                            post_floor_turns_total += 1
                            if ev_count > 0:
                                post_floor_nonempty_turns_total += 1
                            new_items_this_turn = int(belief_trace.get("new_items_this_turn", 0) or 0)
                            if new_items_this_turn <= 0:
                                updated_items = belief_trace.get("updated_item_ids", [])
                                if isinstance(updated_items, list):
                                    new_items_this_turn = len(updated_items)
                            post_floor_new_items_total += max(0, int(new_items_this_turn))

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
