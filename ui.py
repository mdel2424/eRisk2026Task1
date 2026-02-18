from __future__ import annotations

import json
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from core.state import build_initial_state
from graph import app
from persona import PersonaProfile, create_persona, generate_persona_profiles

load_dotenv()

st.set_page_config(page_title="eRisk PoC", layout="wide")
st.title("eRisk PoC")


def _serialize(value):
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _serialize(value.model_dump())
        except Exception:
            return str(value)
    return value


def _selected_profile(profiles: List[PersonaProfile], persona_id: str) -> PersonaProfile:
    for profile in profiles:
        if profile.persona_id == persona_id:
            return profile
    return profiles[0]


def _attach_turn_debug_to_latest_detector_message(state: Dict) -> None:
    if not state.get("messages"):
        return

    payload = {
        "trace": {
            "route": state.get("route_debug", ""),
            "specialist": state.get("specialist_debug", ""),
            "stop": state.get("stop_debug", ""),
        },
        "evidence": _serialize(state.get("latest_turn_evidence", [])),
        "positive_contributions": _serialize(state.get("positive_contributions", [])),
        "negative_contributions": _serialize(state.get("negative_contributions", [])),
    }
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            msg["turn_debug"] = payload
            return


def _reset_conversation(profile: PersonaProfile) -> None:
    st.session_state.persona = create_persona(profile)
    state = build_initial_state(persona_id=profile.persona_id)
    state = app.invoke(state)
    _attach_turn_debug_to_latest_detector_message(state)
    st.session_state.agent_state = state
    st.session_state.awaiting_persona = True


def _session_needs_hard_reset(state: Dict) -> bool:
    if not state:
        return True
    required_state_keys = [
        "route_debug",
        "specialist_debug",
        "stop_debug",
        "item_beliefs",
        "evidence_log",
        "route_history",
        "stop_history",
    ]
    if any(key not in state for key in required_state_keys):
        return True
    messages = state.get("messages", [])
    if not messages:
        return True
    if messages[0].get("role") != "user":
        return True
    first_detector = next((msg for msg in messages if msg.get("role") == "user"), None)
    if first_detector and "turn_debug" not in first_detector:
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
    _attach_turn_debug_to_latest_detector_message(state)
    if not state.get("should_stop"):
        st.session_state.awaiting_persona = True
    st.session_state.agent_state = state


def _run_until_stop(max_steps: int = 20) -> None:
    steps = 0
    while not st.session_state.agent_state.get("should_stop") and steps < max_steps:
        _run_next_turn()
        steps += 1


def _conversation_payload(state: Dict) -> Dict:
    return {
        "persona_id": state.get("persona_id"),
        "messages": state.get("messages", []),
        "predicted_label": state.get("predicted_label"),
        "predicted_bdi_score": state.get("predicted_bdi_score"),
        "predicted_key_symptoms": state.get("predicted_key_symptoms", []),
        "global_confidence": state.get("global_confidence", 0.0),
    }


def _diagnostics_payload(state: Dict) -> Dict:
    return {
        "route_history": _serialize(state.get("route_history", [])),
        "evidence_log": _serialize(state.get("evidence_log", [])),
        "item_beliefs": _serialize(state.get("item_beliefs", {})),
        "stop_history": _serialize(state.get("stop_history", [])),
        "latest_feature_vector": _serialize(state.get("latest_feature_vector", {})),
        "positive_contributions": _serialize(state.get("positive_contributions", [])),
        "negative_contributions": _serialize(state.get("negative_contributions", [])),
        "route_debug": state.get("route_debug", ""),
        "specialist_debug": state.get("specialist_debug", ""),
        "stop_debug": state.get("stop_debug", ""),
    }


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

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Conversation")
    for msg in state.get("messages", []):
        role = msg.get("role", "assistant")
        content = str(msg.get("content", ""))
        ui_role = "assistant" if role == "user" else "user"
        avatar = "🤖" if role == "user" else "👤"
        with st.chat_message(ui_role, avatar=avatar):
            st.write(content)
            if role == "user":
                turn_debug = msg.get("turn_debug", {})
                trace = turn_debug.get("trace", {})
                with st.expander("Trace", expanded=False):
                    st.caption(f"Route: {trace.get('route', '')}")
                    st.caption(f"Specialist: {trace.get('specialist', '')}")
                    st.caption(f"Stop: {trace.get('stop', '')}")

                    evidence = turn_debug.get("evidence", [])
                    if evidence:
                        st.write("Evidence")
                        st.dataframe(evidence, use_container_width=True)
                    else:
                        st.caption("Evidence: none extracted this turn")

with right_col:
    st.subheader("Current Prediction")
    st.write(f"Persona ID: **{profile.persona_id}**")
    st.write(f"Turn: **{state.get('turn_index', 0)}**")
    st.write(f"Predicted label: **{state.get('predicted_label', 'n/a')}**")
    st.write(f"Predicted BDI score: **{state.get('predicted_bdi_score', 'n/a')}**")
    st.write(f"Global confidence: **{state.get('global_confidence', 0.0):.2f}**")
    st.write(f"Top symptoms: **{state.get('predicted_key_symptoms', [])}**")
    st.write(f"Stop: **{state.get('should_stop', False)}**")

    stop_history = state.get("stop_history", [])
    if stop_history:
        latest_stop = _serialize(stop_history[-1])
        st.caption(
            f"Stop rationale: {latest_stop.get('reason', '')} | "
            f"confidence={latest_stop.get('confidence', 0.0):.2f}"
        )

    st.markdown("---")
    st.write("Item Beliefs")
    item_beliefs = _serialize(state.get("item_beliefs", {}))
    belief_rows = []
    for item_id_str, belief in item_beliefs.items():
        if isinstance(belief, dict):
            belief_rows.append(
                {
                    "item_id": int(item_id_str),
                    "mean_score": round(float(belief.get("mean_score", 0.0)), 3),
                    "uncertainty": round(float(belief.get("uncertainty", 1.0)), 3),
                    "support_count": int(belief.get("support_count", 0)),
                }
            )
    belief_rows = sorted(belief_rows, key=lambda row: row["item_id"])
    st.dataframe(belief_rows, use_container_width=True, height=280)

    st.write("Top Positive Contributions")
    st.dataframe(_serialize(state.get("positive_contributions", [])), use_container_width=True)
    st.write("Top Negative Contributions")
    st.dataframe(_serialize(state.get("negative_contributions", [])), use_container_width=True)

    st.markdown("---")
    conversation_json = json.dumps(_serialize(_conversation_payload(state)), indent=2)
    diagnostics_json = json.dumps(_serialize(_diagnostics_payload(state)), indent=2)
    st.download_button(
        "Download Conversation JSON",
        data=conversation_json,
        file_name=f"conversation_persona_{profile.persona_id}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download Diagnostics JSON",
        data=diagnostics_json,
        file_name=f"diagnostics_persona_{profile.persona_id}.json",
        mime="application/json",
        use_container_width=True,
    )
