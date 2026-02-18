from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from core.evaluation import compute_metrics
from core.io_schema import PersonaConversation, PersonaResult, Turn
from core.state import build_initial_state
from persona import PersonaProfile, create_persona, generate_persona_profiles

load_dotenv()


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


def _run_detector_until_stop(state: Dict, persona, graph_app) -> Dict:
    while True:
        state = graph_app.invoke(state)
        if state.get("should_stop"):
            return state

        detector_message = state["messages"][-1]["content"]
        persona_reply = persona.reply(state["messages"])
        state["messages"].append({"role": "assistant", "content": persona_reply})
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


def _result_record(profile: PersonaProfile, state: Dict) -> Dict:
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


def run_eval(persona_count: int, seed: int) -> None:
    from graph import app as graph_app

    print(f"--- Eval Mode: {persona_count} personas | seed={seed} ---")
    profiles = generate_persona_profiles(persona_count, seed=seed)

    conversations: List[PersonaConversation] = []
    results: List[PersonaResult] = []
    metric_rows: List[Dict] = []

    for profile in profiles:
        print(f"\n=== Persona {profile.persona_id} ===")
        persona = create_persona(profile)
        state = build_initial_state(persona_id=profile.persona_id)

        state = _run_detector_until_stop(state, persona, graph_app)

        turns = _to_turns(state["messages"])
        conversations.append(PersonaConversation(LLM=profile.persona_id, conversation=turns))

        result = PersonaResult(
            LLM=profile.persona_id,
            bdi_score=int(state.get("predicted_bdi_score") or 0),
            key_symptoms=list(state.get("predicted_key_symptoms") or [])[:4],
        )
        results.append(result)
        metric_rows.append(_result_record(profile, state))

    metrics = compute_metrics(metric_rows)
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
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n--- Evaluation Summary ---")
    print(json.dumps(metrics, indent=2))
    print("\nWrote:")
    print(" - outputs/interactions_run_local.json")
    print(" - outputs/results_run_local.json")
    print(" - outputs/metrics_run_local.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eRisk 2026 Conversational Depression Detection PoC")
    parser.add_argument("--mode", choices=["interactive", "eval"], default="interactive")
    parser.add_argument("--personas", type=int, default=10, help="Number of personas in eval mode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic persona generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "interactive":
        run_interactive()
    else:
        run_eval(persona_count=args.personas, seed=args.seed)


if __name__ == "__main__":
    main()
