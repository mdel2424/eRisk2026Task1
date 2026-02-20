from __future__ import annotations

from core.llm import reset_llm_usage, set_llm_call_budget
from core.state import build_initial_state
from persona import create_persona, generate_persona_profiles

from app.cli_runtime_helpers import _assert_openrouter_ready, _format_debug, _print_backend_info


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
            persona_text = user_input
        else:
            persona_text = persona.reply(state["messages"])
        state["messages"].append({"role": "assistant", "content": persona_text})
        print(f"Persona: {persona_text}")

        state = graph_app.invoke(state)

        print(f"Debug: {_format_debug(state)}")
        detector_msg = state["messages"][-1]["content"] if state.get("messages") else ""
        if not state.get("should_stop") and state.get("messages", []) and state["messages"][-1].get("role") == "user":
            print(f"\nDetector: {detector_msg}")
        print(
            f"Turn={state['turn_index']} | PredLabel={state.get('predicted_label')} | "
            f"PredBDI={state.get('predicted_bdi_score')} | Stop={state.get('should_stop')}"
        )
        if state.get("should_stop"):
            print(f"Key symptoms: {state.get('predicted_key_symptoms', [])}")
            break
