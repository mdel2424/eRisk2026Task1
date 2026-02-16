from __future__ import annotations

from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from core.state import build_initial_state
from graph import app
from persona import PersonaProfile, create_persona, generate_persona_profiles

load_dotenv()

st.set_page_config(page_title="eRisk PoC", layout="wide")
st.title("eRisk PoC")


def _selected_profile(profiles: List[PersonaProfile], persona_id: str) -> PersonaProfile:
    for profile in profiles:
        if profile.persona_id == persona_id:
            return profile
    return profiles[0]


def _attach_trace_to_latest_detector_message(state: Dict) -> None:
    if not state.get("messages"):
        return
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            msg["trace"] = {
                "route": state.get("route_debug", ""),
                "specialist": state.get("specialist_debug", ""),
                "stop": state.get("stop_debug", ""),
            }
            return


def _reset_conversation(profile: PersonaProfile) -> None:
    st.session_state.persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)
    state = app.invoke(state)
    _attach_trace_to_latest_detector_message(state)
    st.session_state.agent_state = state
    st.session_state.awaiting_persona = True


def _session_needs_hard_reset(state: Dict) -> bool:
    if not state:
        return True
    if "route_debug" not in state or "specialist_debug" not in state or "stop_debug" not in state:
        return True
    messages = state.get("messages", [])
    if not messages:
        return True
    if messages[0].get("role") != "user":
        return True
    first_detector = next((msg for msg in messages if msg.get("role") == "user"), None)
    if first_detector and "trace" not in first_detector:
        return True
    return False


def _init_session() -> None:
    if "profiles" not in st.session_state:
        st.session_state.profiles = generate_persona_profiles(10, seed=42)
    if "profile_id" not in st.session_state:
        st.session_state.profile_id = st.session_state.profiles[0].persona_id
    if "awaiting_persona" not in st.session_state:
        st.session_state.awaiting_persona = True
    if "agent_state" not in st.session_state or "persona" not in st.session_state:
        profile = _selected_profile(st.session_state.profiles, st.session_state.profile_id)
        _reset_conversation(profile)
    else:
        state = st.session_state.agent_state
        if _session_needs_hard_reset(state):
            profile = _selected_profile(st.session_state.profiles, st.session_state.profile_id)
            _reset_conversation(profile)


def _run_next_turn() -> None:
    state: Dict = st.session_state.agent_state

    if state.get("should_stop"):
        st.session_state.agent_state = state
        return

    if st.session_state.awaiting_persona:
        persona_reply = st.session_state.persona.reply(state["messages"])
        state["messages"].append({"role": "assistant", "content": persona_reply})
        st.session_state.awaiting_persona = False

    state = app.invoke(state)
    _attach_trace_to_latest_detector_message(state)
    if not state.get("should_stop"):
        st.session_state.awaiting_persona = True
    st.session_state.agent_state = state


def _run_until_stop(max_steps: int = 20) -> None:
    steps = 0
    while not st.session_state.agent_state.get("should_stop") and steps < max_steps:
        _run_next_turn()
        steps += 1


_init_session()

with st.sidebar:
    persona_ids = [p.persona_id for p in st.session_state.profiles]
    selected_id = st.selectbox(
        "Persona ID",
        persona_ids,
        index=persona_ids.index(st.session_state.profile_id),
    )

    if st.button("Reset"):
        st.session_state.profile_id = selected_id
        profile = _selected_profile(st.session_state.profiles, selected_id)
        _reset_conversation(profile)
        st.rerun()

    if st.button("Run Next Turn"):
        _run_next_turn()
        st.rerun()

    if st.button("Auto-Run To Decision"):
        _run_until_stop()
        st.rerun()

state = st.session_state.agent_state
profile = _selected_profile(st.session_state.profiles, st.session_state.profile_id)

st.subheader("Conversation")
for msg in state.get("messages", []):
    role = msg.get("role", "assistant")
    content = str(msg.get("content", ""))
    ui_role = "assistant" if role == "user" else "user"
    avatar = "🤖" if role == "user" else "👤"
    with st.chat_message(ui_role, avatar=avatar):
        st.write(content)
        if role == "user":
            trace = msg.get("trace", {})
            with st.expander("Trace", expanded=False):
                st.caption(f"Route: {trace.get('route', '')}")
                st.caption(f"Specialist: {trace.get('specialist', '')}")
                st.caption(f"Stop: {trace.get('stop', '')}")

st.subheader("Current Prediction")
st.write(f"Persona ID: **{profile.persona_id}**")
st.write(f"Turn: **{state.get('turn_index', 0)}**")
st.write(f"Predicted label: **{state.get('predicted_label', 'n/a')}**")
st.write(f"Predicted BDI score: **{state.get('predicted_bdi_score', 'n/a')}**")
st.write(f"Top symptoms: **{state.get('predicted_key_symptoms', [])}**")
st.write(f"Stop: **{state.get('should_stop', False)}**")
