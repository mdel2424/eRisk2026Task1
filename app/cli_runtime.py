from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.llm import LLMBudgetExceeded
from core.state import build_initial_state
from persona import PersonaProfile, create_persona

from app.cli_runtime_helpers import _mark_budget_exceeded, _snapshot_turn, _usage_snippet


def _run_detector_until_stop(
    state: Dict,
    persona,
    graph_app,
    verbose: bool = False,
    progress_prefix: str = "",
    live_status: bool = False,
) -> Tuple[Dict, List[Dict]]:
    timeline: List[Dict] = []
    cycle = 0
    while True:
        cycle += 1
        if live_status:
            print(
                f"\r{progress_prefix} cycle={cycle} turn={state.get('turn_index', 0)} "
                f"stage=detector_graph {_usage_snippet()}",
                end="",
                flush=True,
            )
        try:
            state = graph_app.invoke(state)
        except LLMBudgetExceeded as exc:
            state = _mark_budget_exceeded(state, "detector_graph", exc)
            timeline.append(_snapshot_turn(state))
            if live_status:
                print()
            return state, timeline

        timeline.append(_snapshot_turn(state))
        if state.get("should_stop"):
            if live_status:
                print()
            return state, timeline

        detector_message = state["messages"][-1]["content"]
        if live_status:
            print(
                f"\r{progress_prefix} cycle={cycle} turn={state.get('turn_index', 0)} "
                f"stage=persona_reply {_usage_snippet()}",
                end="",
                flush=True,
            )
        try:
            persona_reply = persona.reply(state["messages"])
        except LLMBudgetExceeded as exc:
            state = _mark_budget_exceeded(state, "persona_reply", exc)
            timeline.append(_snapshot_turn(state))
            if live_status:
                print()
            return state, timeline

        state["messages"].append({"role": "assistant", "content": persona_reply})
        if verbose:
            print(f"\nDetector: {detector_message}")
            print(f"Persona: {persona_reply}")


def _run_profile(
    profile: PersonaProfile,
    graph_app,
    verbose: bool = False,
    progress_prefix: str = "",
    live_status: bool = False,
) -> Tuple[Dict, List[Dict], Dict[str, Any]]:
    persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)
    final_state, timeline = _run_detector_until_stop(
        state,
        persona,
        graph_app,
        verbose=verbose,
        progress_prefix=progress_prefix,
        live_status=live_status,
    )
    style_stats: Dict[str, Any] = {}
    if hasattr(persona, "style_stats"):
        try:
            stats = persona.style_stats()
            if isinstance(stats, dict):
                style_stats = stats
        except Exception:
            style_stats = {}
    return final_state, timeline, style_stats
