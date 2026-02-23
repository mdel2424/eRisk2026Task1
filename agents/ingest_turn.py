from __future__ import annotations

import re
from typing import Dict, Tuple

from core.state import AgentState, TurnState


def _latest_persona_message_with_index(state: AgentState) -> Tuple[str, int]:
    messages = list(state.get("messages", []))
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")), idx
    return "", -1


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return " ".join(lowered.split())


def _split_sentences(text: str) -> list[str]:
    chunks = [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]
    return chunks[:8]


def ingest_turn(state: AgentState) -> Dict:
    latest_message, latest_persona_idx = _latest_persona_message_with_index(state)
    last_processed_idx = int(state.get("last_processed_persona_msg_idx", -1))
    has_new_persona_input = latest_persona_idx > last_processed_idx

    prior_turn_index = int(state.get("turn_index", 0))
    turn_index = prior_turn_index + 1 if has_new_persona_input else prior_turn_index
    turn_id = turn_index if turn_index > 0 else 1

    text_raw = latest_message if has_new_persona_input else ""
    text_norm = _normalize_text(text_raw)
    sentences = _split_sentences(text_raw)
    turn = TurnState(
        latest_text_raw=text_raw,
        latest_text_norm=text_norm,
        latest_sentences=sentences,
        turn_id=turn_id,
    )

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["ingest_turn"] = {
        "turn": turn_id,
        "has_new_persona_input": has_new_persona_input,
        "latest_persona_idx": latest_persona_idx,
        "last_processed_persona_msg_idx": last_processed_idx,
        "text_len": len(text_raw),
        "sentence_count": len(sentences),
    }

    return {
        "turn": turn,
        "turn_index": turn_index,
        "has_new_persona_input": has_new_persona_input,
        "last_processed_persona_msg_idx": (
            latest_persona_idx if has_new_persona_input else last_processed_idx
        ),
        "turn_trace": turn_trace,
    }
