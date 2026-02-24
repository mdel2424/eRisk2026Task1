from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.llm import LLMBudgetExceeded
from core.state import build_initial_state
from persona import PersonaProfile, create_persona

from app.cli_runtime_helpers import _mark_budget_exceeded, _snapshot_turn, _usage_snippet


def _build_probe_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    next_action = state.get("next_action")
    if next_action is None:
        raise RuntimeError("Missing probe_intent for persona handoff: next_action is absent.")

    target_item_id = getattr(next_action, "target_item_id", None)
    route = getattr(next_action, "route", None)
    style = getattr(next_action, "style", None)
    mode = getattr(next_action, "mode", None)
    directness = getattr(next_action, "directness", None)
    priority = getattr(next_action, "priority", None)

    if target_item_id is None or route is None or style is None:
        raise RuntimeError("Missing probe_intent for persona handoff: incomplete next_action payload.")

    try:
        target_item_id = int(target_item_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Missing probe_intent for persona handoff: target_item_id is invalid.") from exc
    if target_item_id < 1 or target_item_id > 21:
        raise RuntimeError("Missing probe_intent for persona handoff: target_item_id is out of range.")

    route = str(route).strip().lower()
    if route not in {"somatic", "cognitive", "risk"}:
        raise RuntimeError("Missing probe_intent for persona handoff: route is invalid.")

    style = str(style).strip()
    if not style:
        raise RuntimeError("Missing probe_intent for persona handoff: style is empty.")

    mode = str(mode or "normal").strip().lower()
    if mode not in {"normal", "wrapup"}:
        raise RuntimeError("Missing probe_intent for persona handoff: mode is invalid.")

    directness = str(directness or "indirect").strip().lower()
    if directness not in {"indirect", "direct"}:
        raise RuntimeError("Missing probe_intent for persona handoff: directness is invalid.")

    try:
        priority = float(priority if priority is not None else 0.5)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Missing probe_intent for persona handoff: priority is invalid.") from exc
    if priority < 0.0 or priority > 1.0:
        raise RuntimeError("Missing probe_intent for persona handoff: priority is out of range.")

    return {
        "target_item_id": target_item_id,
        "route": route,
        "style": style,
        "mode": mode,
        "directness": directness,
        "priority": priority,
    }


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

        if state.get("should_stop"):
            timeline.append(_snapshot_turn(state))
            if live_status:
                print()
            return state, timeline

        probe_intent = _build_probe_intent(state)
        turn_trace = dict(state.get("turn_trace", {}))
        turn_trace["persona_handoff"] = {
            "target_item_id": probe_intent["target_item_id"],
            "route": probe_intent["route"],
            "style": probe_intent["style"],
            "mode": probe_intent["mode"],
            "directness": probe_intent["directness"],
            "priority": round(float(probe_intent["priority"]), 4),
        }
        state["turn_trace"] = turn_trace
        timeline.append(_snapshot_turn(state))

        detector_message = state["messages"][-1]["content"]
        if live_status:
            print(
                f"\r{progress_prefix} cycle={cycle} turn={state.get('turn_index', 0)} "
                f"stage=persona_reply {_usage_snippet()}",
                end="",
                flush=True,
            )
        try:
            persona_reply = persona.reply(state["messages"], probe_intent)
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
