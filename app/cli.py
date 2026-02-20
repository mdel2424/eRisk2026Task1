from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from core.calibration import clear_calibrator_cache, fit_calibrator, save_calibrator_bundle
from core.evaluation import compute_metrics
from core.io_schema import PersonaConversation, PersonaResult, Turn
from core.llm import LLMBudgetExceeded, get_llm_usage, reset_llm_usage, set_llm_call_budget
from core.runtime_policy import (
    auto_backend_switch_enabled,
    cuda_runtime,
    min_cuda_vram_gb,
    resolve_detector_backend,
    resolve_persona_backend,
)
from core.state import build_initial_state
from persona import PersonaProfile, build_split_profiles, create_persona, generate_persona_profiles
from persona.sim_behavior import validate_template_disjointness

load_dotenv()

MANIFEST_SCHEMA_VERSION = 2


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | int | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _safe_model_dump(value):
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return value
    return value


def _serialize(value):
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    dumped = _safe_model_dump(value)
    if dumped is not value:
        return _serialize(dumped)
    return value


def _format_debug(state: Dict) -> str:
    return " | ".join(
        [
            f"route={state.get('route_debug', '')}",
            f"specialist={state.get('specialist_debug', '')}",
            f"stop={state.get('stop_debug', '')}",
        ]
    )


def _print_progress(label: str, current: int, total: int, width: int = 24) -> None:
    total = max(1, total)
    current = max(0, min(current, total))
    filled = int(width * (current / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label} [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def _print_backend_info(max_api_calls: int | None = None, trace_level: str = "compact") -> None:
    auto_on = auto_backend_switch_enabled()
    detector_backend = resolve_detector_backend()
    persona_backend = resolve_persona_backend()
    cuda_available, vram_gb = cuda_runtime()
    min_vram = min_cuda_vram_gb()
    cuda_gate = "pass" if (cuda_available and vram_gb >= min_vram) else "fail"

    if detector_backend == "openrouter":
        detector_target = os.getenv("OPENROUTER_DETECTOR_MODEL", "openrouter/auto")
    else:
        detector_target = os.getenv("DETECTOR_MODEL", "")

    if persona_backend == "openrouter_sim":
        persona_target = os.getenv("OPENROUTER_PERSONA_MODEL", "openrouter/auto")
    else:
        persona_target = os.getenv("ERISK_ADAPTER_ID", "")

    print(
        "Backend info: "
        f"auto_switch={'on' if auto_on else 'off'} | "
        f"cuda_available={cuda_available} | vram_gb={vram_gb:.2f} | "
        f"min_vram_gb={min_vram:.2f} | cuda_gate={cuda_gate}"
    )
    print(
        "Resolved backends: "
        f"detector={detector_backend} [{detector_target}] | "
        f"persona={persona_backend} [{persona_target}]"
    )
    call_budget_text = "none" if max_api_calls is None or max_api_calls <= 0 else str(max_api_calls)
    print(f"Runtime controls: trace_level={trace_level} | max_api_calls={call_budget_text}")


def _assert_openrouter_ready() -> None:
    detector_backend = resolve_detector_backend()
    persona_backend = resolve_persona_backend()
    if detector_backend == "openrouter" or persona_backend == "openrouter_sim":
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise ValueError(
                "OPENROUTER_API_KEY is required because at least one resolved backend uses OpenRouter."
            )


def _to_turns(messages: List[dict]) -> List[Turn]:
    turns: List[Turn] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"user", "assistant"}:
            turns.append(Turn(role=role, message=str(msg.get("content", ""))))
    return turns


def _snapshot_turn(state: Dict) -> Dict:
    route_history = state.get("route_history", [])
    stop_history = state.get("stop_history", [])
    return {
        "turn": int(state.get("turn_index", 0)),
        "route_decision": _serialize(route_history[-1]) if route_history else None,
        "latest_evidence": _serialize(state.get("latest_turn_evidence", [])),
        "item_beliefs": _serialize(state.get("item_beliefs", {})),
        "positive_contributions": _serialize(state.get("positive_contributions", [])),
        "negative_contributions": _serialize(state.get("negative_contributions", [])),
        "stop_decision": _serialize(stop_history[-1]) if stop_history else None,
        "predicted_label": state.get("predicted_label"),
        "predicted_bdi_score": state.get("predicted_bdi_score"),
        "raw_predicted_label": state.get("raw_predicted_label"),
        "raw_predicted_bdi_score": state.get("raw_predicted_bdi_score"),
        "final_item_scores": _serialize(state.get("final_item_scores", {})),
        "module_imputation": _serialize(state.get("module_imputation", {})),
        "global_confidence": state.get("global_confidence", 0.0),
        "route_debug": state.get("route_debug", ""),
        "specialist_debug": state.get("specialist_debug", ""),
        "stop_debug": state.get("stop_debug", ""),
        "turn_trace": _serialize(state.get("turn_trace", {})),
        "failure_counters": _serialize(state.get("failure_counters", {})),
        "empty_evidence_streak": int(state.get("empty_evidence_streak", 0)),
    }


def _mark_budget_exceeded(state: Dict, where: str, exc: Exception) -> Dict:
    counters = dict(state.get("failure_counters", {}))
    counters["budget_exceeded"] = int(counters.get("budget_exceeded", 0)) + 1
    trace = dict(state.get("turn_trace", {}))
    trace["budget"] = {"where": where, "error": str(exc)}
    state["failure_counters"] = counters
    state["turn_trace"] = trace
    state["should_stop"] = True
    state["stop_debug"] = f"Budget exceeded at {where}: {exc}"
    trace_log = list(state.get("trace_log", []))
    trace_log.append(
        {
            "turn": int(state.get("turn_index", 0)),
            "turn_trace": trace,
            "stop_debug": state["stop_debug"],
            "failure_counters": counters,
        }
    )
    state["trace_log"] = trace_log
    return state


def _run_detector_until_stop(
    state: Dict,
    persona,
    graph_app,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    timeline: List[Dict] = []
    while True:
        try:
            state = graph_app.invoke(state)
        except LLMBudgetExceeded as exc:
            state = _mark_budget_exceeded(state, "detector_graph", exc)
            timeline.append(_snapshot_turn(state))
            return state, timeline

        timeline.append(_snapshot_turn(state))
        if state.get("should_stop"):
            return state, timeline

        detector_message = state["messages"][-1]["content"]
        try:
            persona_reply = persona.reply(state["messages"])
        except LLMBudgetExceeded as exc:
            state = _mark_budget_exceeded(state, "persona_reply", exc)
            timeline.append(_snapshot_turn(state))
            return state, timeline

        state["messages"].append({"role": "assistant", "content": persona_reply})
        if verbose:
            print(f"\nDetector: {detector_message}")
            print(f"Persona: {persona_reply}")


def run_interactive() -> None:
    from graph import app as graph_app

    print("--- eRisk 2026: Conversational Depression Detection MVP (Interactive) ---")
    _assert_openrouter_ready()
    set_llm_call_budget(None)
    reset_llm_usage()
    _print_backend_info(max_api_calls=None, trace_level="compact")

    profiles = generate_persona_profiles(count=10, seed=42)
    profile = profiles[0]
    persona = create_persona(profile)

    state = build_initial_state(persona_id=profile.persona_id)
    state = graph_app.invoke(state)
    print(f"\nDetector: {state['messages'][-1]['content']}")
    print(f"Debug: {_format_debug(state)}")

    while True:
        user_input = input("\nPersona (enter for auto / exit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input:
            state["messages"].append({"role": "assistant", "content": user_input})
        else:
            state["messages"].append({"role": "assistant", "content": persona.reply(state["messages"])})

        state = graph_app.invoke(state)

        detector_msg = state["messages"][-1]["content"]
        print(f"\nDetector: {detector_msg}")
        print(
            f"Turn={state['turn_index']} | PredLabel={state.get('predicted_label')} | "
            f"PredBDI={state.get('predicted_bdi_score')} | Stop={state.get('should_stop')}"
        )
        print(f"Debug: {_format_debug(state)}")
        if state.get("should_stop"):
            print(f"Key symptoms: {state.get('predicted_key_symptoms', [])}")
            break


def _result_record(profile: PersonaProfile, state: Dict) -> Dict:
    predicted_label = state.get("predicted_label", "control")
    predicted_bdi = int(state.get("predicted_bdi_score") or 0)
    predicted_symptoms = list(state.get("predicted_key_symptoms") or [])[:4]
    risk_flag = bool(state.get("risk_flag", False))

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
        "turns": int(state.get("turn_index", 0)),
        "risk_true": profile.has_risk_signal,
        "risk_pred": risk_flag,
    }


def _objective(metrics: Dict[str, float], max_turns: int, latency_lambda: float = 0.15) -> float:
    if not metrics:
        return 0.0
    binary_f1 = float(metrics.get("binary_f1", 0.0))
    avg_turns = float(metrics.get("avg_turns_to_decision", 0.0))
    normalized_turns = min(1.0, avg_turns / max(1, max_turns))
    return round(binary_f1 - (latency_lambda * normalized_turns), 4)


def _with_objective(metrics: Dict[str, float], max_turns: int) -> Dict[str, float]:
    if not metrics:
        return {}
    return {**metrics, "objective": _objective(metrics, max_turns=max_turns)}


def _select_primary_metrics(
    synthetic_val: Dict[str, float],
    synthetic_test: Dict[str, float],
    overall_labeled: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    if overall_labeled:
        return "overall_labeled", overall_labeled
    if synthetic_test:
        return "synthetic_test", synthetic_test
    if synthetic_val:
        return "synthetic_val", synthetic_val
    return "overall_labeled", {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _current_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _resolve_fit_calibrator_policy(policy: str) -> bool:
    value = str(policy).strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    detector_backend = resolve_detector_backend()
    persona_backend = resolve_persona_backend()
    return not (detector_backend == "openrouter" or persona_backend == "openrouter_sim")


def _usage_snippet() -> str:
    usage = get_llm_usage()
    max_calls = usage.get("max_calls")
    calls_total = int(usage.get("calls_total", 0))
    if max_calls is None:
        return f"calls={calls_total}/inf"
    return f"calls={calls_total}/{int(max_calls)}"


def _run_profile(profile: PersonaProfile, graph_app, verbose: bool = False) -> Tuple[Dict, List[Dict]]:
    persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)
    return _run_detector_until_stop(state, persona, graph_app, verbose=verbose)


def _strict_split_lock_enabled() -> bool:
    return _parse_bool(os.getenv("STRICT_SPLIT_LOCK", "1"))


def _manifest_payload(
    *,
    persona_count: int,
    seed: int,
    generator_version: str,
    train_profiles: List[PersonaProfile],
    val_profiles: List[PersonaProfile],
    test_profiles: List[PersonaProfile],
) -> Dict[str, Any]:
    def _profile_dict(profile: PersonaProfile) -> Dict[str, Any]:
        return {
            "persona_id": profile.persona_id,
            "split": profile.split,
            "family": profile.family,
            "source": profile.source,
            "has_ground_truth": profile.has_ground_truth,
            "depressed": profile.depressed,
            "bdi_scores": dict(profile.bdi_scores),
            "bdi_total": profile.bdi_total,
            "key_symptoms": profile.key_symptoms,
            "risk_signal": profile.has_risk_signal,
            "behavior_params": dict(profile.behavior_params),
            "template_bank": profile.template_bank,
            "generation_seed": profile.generation_seed,
            "generator_version": profile.generator_version,
        }

    all_profiles = train_profiles + val_profiles + test_profiles
    return {
        "run_config": {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "persona_count": persona_count,
            "seed": seed,
            "generator_version": generator_version,
            "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "split_counts": {
            "train": len(train_profiles),
            "val": len(val_profiles),
            "test": len(test_profiles),
        },
        "profiles": [_profile_dict(profile) for profile in all_profiles],
    }


def _manifest_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enforce_manifest_lock(manifest_path: Path, current_payload: Dict[str, Any], current_hash: str) -> None:
    if not manifest_path.exists():
        return
    try:
        previous_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    prev_config = previous_payload.get("run_config", {})
    curr_config = current_payload.get("run_config", {})
    if prev_config != curr_config:
        return

    previous_hash = _manifest_hash(previous_payload)
    if previous_hash != current_hash:
        raise ValueError(
            "STRICT_SPLIT_LOCK failed: split manifest hash mismatch for identical run_config. "
            "Possible leakage/non-deterministic generation detected. "
            "If this is an intentional simulator update, bump SIM_GENERATOR_VERSION "
            "or remove outputs/persona_manifest_run_local.json."
        )


def _split_overlap_count(a: List[str], b: List[str]) -> int:
    return len(set(a).intersection(set(b)))


def _template_overlap_counts(train_profiles: List[PersonaProfile], val_profiles: List[PersonaProfile], test_profiles: List[PersonaProfile]) -> Dict[str, int]:
    train_banks = [profile.template_bank for profile in train_profiles]
    val_banks = [profile.template_bank for profile in val_profiles]
    test_banks = [profile.template_bank for profile in test_profiles]
    return {
        "train_val": _split_overlap_count(train_banks, val_banks),
        "train_test": _split_overlap_count(train_banks, test_banks),
        "val_test": _split_overlap_count(val_banks, test_banks),
    }


def _family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family", "unknown"))].append(row)

    family_count = {family: len(items) for family, items in grouped.items()}
    binary_f1_by_family: Dict[str, float] = {}
    bdi_mae_by_family: Dict[str, float] = {}
    for family, items in grouped.items():
        metrics = compute_metrics(items)
        binary_f1_by_family[family] = float(metrics.get("binary_f1", 0.0))
        bdi_mae_by_family[family] = float(metrics.get("bdi_mae", 0.0))
    return {
        "family_count": family_count,
        "binary_f1_by_family": binary_f1_by_family,
        "bdi_mae_by_family": bdi_mae_by_family,
    }


def _resolve_effective_eval_mode(eval_mode: str) -> tuple[str, str]:
    requested = str(eval_mode).strip().lower()
    if requested in {"mixed_holdout", "synthetic_only"}:
        return requested, "synthetic_holdout"
    return requested, "synthetic_holdout"


def _profile_meta(profile: PersonaProfile) -> Dict[str, Any]:
    return {
        "split": profile.split,
        "family": profile.family,
        "generator_version": profile.generator_version,
        "generation_seed": profile.generation_seed,
        "template_bank": profile.template_bank,
        "behavior_params": dict(profile.behavior_params),
        "bdi_total": profile.bdi_total,
        "risk_signal": profile.has_risk_signal,
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
) -> None:
    from graph import app as graph_app

    os.environ["PROMPT_VERSION"] = prompt_version
    os.environ.pop("CALIBRATOR_PATH", None)
    clear_calibrator_cache()
    _assert_openrouter_ready()
    set_llm_call_budget(max_api_calls if max_api_calls > 0 else None)
    reset_llm_usage()

    fit_calibrator_enabled = _resolve_fit_calibrator_policy(fit_calibrator_policy)
    min_train_records = _parse_int(os.getenv("CALIBRATOR_MIN_TRAIN_RECORDS", "10"), 10)
    strict_split_lock = _strict_split_lock_enabled()
    generator_version = os.getenv("SIM_GENERATOR_VERSION", "sim_v2").strip() or "sim_v2"

    requested_eval_mode, effective_eval_mode = _resolve_effective_eval_mode(eval_mode)

    print(f"--- Eval Mode: {requested_eval_mode} | personas={persona_count} | seed={seed} | prompts={prompt_version} ---")
    _print_backend_info(max_api_calls=max_api_calls if max_api_calls > 0 else None, trace_level=trace_level)
    print(
        "Calibrator policy: "
        f"requested={fit_calibrator_policy} | enabled={fit_calibrator_enabled} | min_train_records={min_train_records}"
    )
    print(
        f"Synthetic generator: version={generator_version} | strict_split_lock={'on' if strict_split_lock else 'off'}"
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

    output_dir = Path("outputs")
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

    eval_profiles = val_profiles + test_profiles

    run_failure_counters: Counter[str] = Counter()
    calibrator_status: Dict[str, Any] = {
        "requested_policy": fit_calibrator_policy,
        "enabled": fit_calibrator_enabled,
        "mode": "deterministic_default",
        "reason": "disabled_by_policy",
        "train_records": 0,
        "saved_path": "",
    }

    calibrator_train_records: List[Dict] = []
    calibrator_train_ids: List[str] = []

    if train_profiles and fit_calibrator_enabled:
        if len(train_profiles) < min_train_records:
            calibrator_status["reason"] = "small_train_split"
            calibrator_status["mode"] = "skipped_small_train"
            run_failure_counters["calibrator_fallback_small_train"] += 1
            print(
                f"Skipping calibrator fit: train split too small "
                f"({len(train_profiles)} < {min_train_records})."
            )
        else:
            print(f"Fitting calibrator from synthetic train split ({len(train_profiles)} personas)...")
            train_total = len(train_profiles)
            for idx, profile in enumerate(train_profiles, start=1):
                final_state, _ = _run_profile(profile, graph_app, verbose=False)
                calibrator_train_ids.append(profile.persona_id)
                feature_vector = dict(final_state.get("latest_feature_vector", {}))
                calibrator_train_records.append(
                    {
                        "features": feature_vector,
                        "bdi_true": profile.bdi_total,
                        "y_true": profile.depressed,
                    }
                )
                _print_progress(f"Calibrator fit {_usage_snippet()}", idx, train_total)

            calibrator_status["train_records"] = len(calibrator_train_records)
            if calibrator_train_records:
                bundle, fit_reason = fit_calibrator(calibrator_train_records, min_records=min_train_records)
                calibrator_status["mode"] = bundle.mode
                calibrator_status["reason"] = fit_reason or ""
                if bundle.mode == "sklearn_fitted":
                    calibrator_path = output_dir / "calibrator_bundle_local.json"
                    save_calibrator_bundle(calibrator_path, bundle)
                    os.environ["CALIBRATOR_PATH"] = str(calibrator_path)
                    calibrator_status["saved_path"] = str(calibrator_path)
                    print(f"Calibrator saved: {calibrator_path}")
                else:
                    if fit_reason == "small_train_split":
                        run_failure_counters["calibrator_fallback_small_train"] += 1
                    print(f"Calibrator fallback mode: {bundle.mode} ({fit_reason or 'n/a'})")
                clear_calibrator_cache()
    elif not fit_calibrator_enabled:
        calibrator_status["reason"] = "disabled_by_policy"
    else:
        calibrator_status["reason"] = "no_train_profiles"

    conversations: List[PersonaConversation] = []
    results: List[PersonaResult] = []
    diagnostics_payload: List[Dict[str, Any]] = []
    val_rows: List[Dict] = []
    test_rows: List[Dict] = []
    overall_rows: List[Dict] = []
    route_distribution: Counter[str] = Counter()
    calibrator_mode_counts: Counter[str] = Counter()
    turns_total = 0
    evidence_turns_nonempty = 0
    evidence_records_total = 0

    eval_total = len(eval_profiles)
    processed_profiles = 0
    eval_ids: List[str] = []

    for idx, profile in enumerate(eval_profiles, start=1):
        print(f"\n=== Persona {profile.persona_id} ({profile.source}/{profile.split}/{profile.family}) ===")
        final_state, timeline = _run_profile(profile, graph_app, verbose=False)
        processed_profiles += 1
        eval_ids.append(profile.persona_id)

        turns = _to_turns(final_state["messages"])
        conversations.append(PersonaConversation(LLM=profile.persona_id, conversation=turns))
        results.append(
            PersonaResult(
                LLM=profile.persona_id,
                bdi_score=int(final_state.get("predicted_bdi_score") or 0),
                key_symptoms=list(final_state.get("predicted_key_symptoms") or [])[:4],
            )
        )

        row = _result_record(profile, final_state)
        overall_rows.append(row)
        if profile in val_profiles:
            val_rows.append(row)
        if profile in test_profiles:
            test_rows.append(row)

        diagnostics_payload.append(
            {
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
                        "final_item_scores": final_state.get("final_item_scores", {}),
                        "module_imputation": final_state.get("module_imputation", {}),
                        "global_confidence": final_state.get("global_confidence", 0.0),
                        "route_history": final_state.get("route_history", []),
                        "evidence_log": final_state.get("evidence_log", []),
                        "item_beliefs": final_state.get("item_beliefs", {}),
                        "stop_history": final_state.get("stop_history", []),
                        "trace_log": final_state.get("trace_log", []) if trace_level == "compact" else [],
                        "failure_counters": final_state.get("failure_counters", {}),
                        "calibrator_mode": final_state.get("calibrator_mode", ""),
                    }
                ),
            }
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

        _print_progress(f"Evaluation {_usage_snippet()}", idx, eval_total)
        usage_now = get_llm_usage()
        max_calls_now = usage_now.get("max_calls")
        if max_calls_now is not None and int(usage_now.get("calls_total", 0)) >= int(max_calls_now):
            print("\nStopping eval early: API call budget reached.")
            break

    train_eval_overlap = _split_overlap_count(calibrator_train_ids, eval_ids)
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
            "CALIBRATOR_MIN_TRAIN_RECORDS": os.getenv("CALIBRATOR_MIN_TRAIN_RECORDS", "10"),
            "CALIBRATOR_PATH": os.getenv("CALIBRATOR_PATH", ""),
            "STRICT_SPLIT_LOCK": os.getenv("STRICT_SPLIT_LOCK", "1"),
            "SIM_GENERATOR_VERSION": os.getenv("SIM_GENERATOR_VERSION", "sim_v2"),
            "SIM_PARAPHRASE_ENABLED": os.getenv("SIM_PARAPHRASE_ENABLED", "1"),
            "SIM_PARAPHRASE_RATE": os.getenv("SIM_PARAPHRASE_RATE", "0.5"),
            "SIM_TEMPLATE_DISJOINT_ENFORCE": os.getenv("SIM_TEMPLATE_DISJOINT_ENFORCE", "1"),
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

    print("\n--- Evaluation Summary ---")
    if primary_metrics:
        print(
            f"Primary split: {primary_split} | "
            f"binary_f1={float(primary_metrics.get('binary_f1', 0.0)):.4f} | "
            f"bdi_mae={float(primary_metrics.get('bdi_mae', 0.0)):.4f}"
        )
    print(json.dumps(metrics_payload, indent=2))
    print("\nLeakage report:")
    print(json.dumps(leakage_report_payload, indent=2))
    print("\nWrote:")
    print(" - outputs/persona_manifest_run_local.json")
    print(" - outputs/persona_manifest_hash_run_local.txt")
    print(" - outputs/interactions_run_local.json")
    print(" - outputs/results_run_local.json")
    print(" - outputs/metrics_run_local.json")
    print(" - outputs/failure_report_run_local.json")
    print(" - outputs/leakage_report_run_local.json")
    if save_diagnostics:
        print(" - outputs/diagnostics_run_local.json")
    print(" - outputs/config_used.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eRisk 2026 Conversational Depression Detection PoC")
    parser.add_argument("--mode", choices=["interactive", "eval"], default="interactive")
    parser.add_argument("--personas", type=int, default=10, help="Number of synthetic personas")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic persona generation")
    parser.add_argument(
        "--eval_mode",
        choices=["mixed_holdout", "synthetic_only"],
        default="mixed_holdout",
    )
    parser.add_argument("--prompt_version", default="v1")
    parser.add_argument("--save_diagnostics", default="true")
    parser.add_argument("--max_api_calls", type=int, default=180)
    parser.add_argument("--trace_level", choices=["compact", "off"], default="compact")
    parser.add_argument("--fit_calibrator", choices=["auto", "on", "off"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "interactive":
        run_interactive()
    else:
        run_eval(
            persona_count=args.personas,
            seed=args.seed,
            eval_mode=args.eval_mode,
            prompt_version=args.prompt_version,
            save_diagnostics=_parse_bool(args.save_diagnostics),
            max_api_calls=args.max_api_calls,
            trace_level=args.trace_level,
            fit_calibrator_policy=args.fit_calibrator,
        )


if __name__ == "__main__":
    main()
