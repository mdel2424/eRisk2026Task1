from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from core.calibration import fit_calibrator, save_calibrator_bundle
from core.evaluation import compute_metrics
from core.io_schema import PersonaConversation, PersonaResult, Turn
from core.state import build_initial_state
from persona import (
    PersonaProfile,
    build_official_tracking_profiles,
    create_persona,
    generate_persona_profiles,
    split_synthetic_profiles,
)

load_dotenv()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
        "global_confidence": state.get("global_confidence", 0.0),
        "route_debug": state.get("route_debug", ""),
        "specialist_debug": state.get("specialist_debug", ""),
        "stop_debug": state.get("stop_debug", ""),
    }


def _run_detector_until_stop(state: Dict, persona, graph_app, verbose: bool = False) -> Tuple[Dict, List[Dict]]:
    timeline: List[Dict] = []
    while True:
        state = graph_app.invoke(state)
        timeline.append(_snapshot_turn(state))
        if state.get("should_stop"):
            return state, timeline

        detector_message = state["messages"][-1]["content"]
        persona_reply = persona.reply(state["messages"])
        state["messages"].append({"role": "assistant", "content": persona_reply})
        if verbose:
            print(f"\nDetector: {detector_message}")
            print(f"Persona: {persona_reply}")


def run_interactive() -> None:
    from graph import app as graph_app

    print("--- eRisk 2026: Conversational Depression Detection MVP (Interactive) ---")
    state = build_initial_state(persona_id="interactive")
    state = graph_app.invoke(state)
    print(f"\nDetector: {state['messages'][-1]['content']}")
    print(f"Debug: {_format_debug(state)}")

    while True:
        user_input = input("\nPersona: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        state["messages"].append({"role": "assistant", "content": user_input})
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


def _result_record(profile: PersonaProfile, state: Dict) -> Dict | None:
    if not profile.has_ground_truth:
        return None

    predicted_label = state.get("predicted_label", "control")
    predicted_bdi = int(state.get("predicted_bdi_score") or 0)
    predicted_symptoms = list(state.get("predicted_key_symptoms") or [])[:4]
    risk_flag = bool(state.get("risk_flag", False))

    return {
        "llm": profile.persona_id,
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


def _prefix_profiles(profiles: List[PersonaProfile], prefix: str) -> List[PersonaProfile]:
    prefixed: List[PersonaProfile] = []
    for profile in profiles:
        prefixed.append(replace(profile, persona_id=f"{prefix}-{profile.persona_id}"))
    return prefixed


def _run_profile(profile: PersonaProfile, graph_app, verbose: bool = False) -> Tuple[Dict, List[Dict]]:
    persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)
    return _run_detector_until_stop(state, persona, graph_app, verbose=verbose)


def _released_official_ids() -> List[str]:
    raw = os.getenv("OFFICIAL_RELEASED_PERSONAS", "1,2").strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts if parts else ["1", "2"]


def run_eval(
    persona_count: int,
    seed: int,
    eval_mode: str,
    prompt_version: str,
    save_diagnostics: bool,
) -> None:
    from graph import app as graph_app

    os.environ["PROMPT_VERSION"] = prompt_version
    os.environ.pop("CALIBRATOR_PATH", None)

    print(f"--- Eval Mode: {eval_mode} | personas={persona_count} | seed={seed} | prompts={prompt_version} ---")

    synthetic_profiles = _prefix_profiles(generate_persona_profiles(persona_count, seed=seed), "synth")
    splits = split_synthetic_profiles(synthetic_profiles, seed=seed)
    official_profiles = build_official_tracking_profiles(_released_official_ids())

    train_profiles = splits["synthetic_train"]
    val_profiles = splits["synthetic_val"]
    test_profiles = splits["synthetic_test"]

    calibrator_train_records: List[Dict] = []
    if eval_mode in {"mixed_holdout", "synthetic_only"} and train_profiles:
        print(f"Fitting calibrator from synthetic train split ({len(train_profiles)} personas)...")
        for profile in train_profiles:
            final_state, _ = _run_profile(profile, graph_app, verbose=False)
            feature_vector = dict(final_state.get("latest_feature_vector", {}))
            calibrator_train_records.append(
                {
                    "features": feature_vector,
                    "bdi_true": profile.bdi_total,
                    "y_true": profile.depressed,
                }
            )

        if calibrator_train_records:
            bundle = fit_calibrator(calibrator_train_records)
            calibrator_path = Path("outputs/calibrator_bundle_local.json")
            save_calibrator_bundle(calibrator_path, bundle)
            os.environ["CALIBRATOR_PATH"] = str(calibrator_path)
            print(f"Calibrator saved: {calibrator_path}")

    if eval_mode == "mixed_holdout":
        eval_profiles = val_profiles + test_profiles + official_profiles
    elif eval_mode == "official_only":
        eval_profiles = official_profiles
    else:
        eval_profiles = val_profiles + test_profiles

    conversations: List[PersonaConversation] = []
    results: List[PersonaResult] = []
    diagnostics_payload: List[Dict[str, Any]] = []
    val_rows: List[Dict] = []
    test_rows: List[Dict] = []
    overall_rows: List[Dict] = []

    for profile in eval_profiles:
        print(f"\n=== Persona {profile.persona_id} ({profile.source}) ===")
        final_state, timeline = _run_profile(profile, graph_app, verbose=False)

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
        if row is not None:
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
                "timeline": _serialize(timeline),
                "final_state": _serialize(
                    {
                        "turn_index": final_state.get("turn_index", 0),
                        "predicted_label": final_state.get("predicted_label"),
                        "predicted_bdi_score": final_state.get("predicted_bdi_score"),
                        "predicted_key_symptoms": final_state.get("predicted_key_symptoms", []),
                        "global_confidence": final_state.get("global_confidence", 0.0),
                        "route_history": final_state.get("route_history", []),
                        "evidence_log": final_state.get("evidence_log", []),
                        "item_beliefs": final_state.get("item_beliefs", {}),
                        "stop_history": final_state.get("stop_history", []),
                    }
                ),
            }
        )

    val_metrics = compute_metrics(val_rows) if val_rows else {}
    test_metrics = compute_metrics(test_rows) if test_rows else {}
    overall_metrics = compute_metrics(overall_rows) if overall_rows else {}
    max_turns = int(os.getenv("MAX_TURNS", "10"))

    metrics_payload: Dict[str, Any] = {
        "eval_mode": eval_mode,
        "prompt_version": prompt_version,
        "synthetic_train_count": len(train_profiles),
        "synthetic_val_count": len(val_profiles),
        "synthetic_test_count": len(test_profiles),
        "official_tracking_count": len(official_profiles),
        "synthetic_val": {
            **val_metrics,
            "objective": _objective(val_metrics, max_turns=max_turns) if val_metrics else 0.0,
        },
        "synthetic_test": {
            **test_metrics,
            "objective": _objective(test_metrics, max_turns=max_turns) if test_metrics else 0.0,
        },
        "overall_labeled": {
            **overall_metrics,
            "objective": _objective(overall_metrics, max_turns=max_turns) if overall_metrics else 0.0,
        },
    }
    if test_metrics:
        metrics_payload.update(test_metrics)
    elif overall_metrics:
        metrics_payload.update(overall_metrics)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions_payload = [conv.model_dump() for conv in conversations]
    results_payload = [result.to_erisk_dict() for result in results]

    (output_dir / "interactions_run_local.json").write_text(
        json.dumps(interactions_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "results_run_local.json").write_text(
        json.dumps(results_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics_run_local.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    if save_diagnostics:
        (output_dir / "diagnostics_run_local.json").write_text(
            json.dumps(diagnostics_payload, indent=2),
            encoding="utf-8",
        )

    config_snapshot = {
        "args": {
            "mode": "eval",
            "personas": persona_count,
            "seed": seed,
            "eval_mode": eval_mode,
            "prompt_version": prompt_version,
            "save_diagnostics": save_diagnostics,
        },
        "env": {
            "PROMPT_VERSION": os.getenv("PROMPT_VERSION", "v1"),
            "DETECTOR_MODEL": os.getenv("DETECTOR_MODEL", ""),
            "ERISK_BASE_MODEL": os.getenv("ERISK_BASE_MODEL", ""),
            "ERISK_ADAPTER_ID": os.getenv("ERISK_ADAPTER_ID", ""),
            "ERISK_ADAPTER_TEMPLATE": os.getenv("ERISK_ADAPTER_TEMPLATE", ""),
            "MIN_TURNS": os.getenv("MIN_TURNS", ""),
            "MAX_TURNS": os.getenv("MAX_TURNS", ""),
            "STOP_CONFIDENCE": os.getenv("STOP_CONFIDENCE", ""),
            "CALIBRATOR_PATH": os.getenv("CALIBRATOR_PATH", ""),
        },
        "git_commit": _current_git_hash(),
    }
    (output_dir / "config_used.json").write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )

    print("\n--- Evaluation Summary ---")
    print(json.dumps(metrics_payload, indent=2))
    print("\nWrote:")
    print(" - outputs/interactions_run_local.json")
    print(" - outputs/results_run_local.json")
    print(" - outputs/metrics_run_local.json")
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
        choices=["mixed_holdout", "official_only", "synthetic_only"],
        default="mixed_holdout",
    )
    parser.add_argument("--prompt_version", default="v1")
    parser.add_argument("--save_diagnostics", default="true")
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
        )


if __name__ == "__main__":
    main()
