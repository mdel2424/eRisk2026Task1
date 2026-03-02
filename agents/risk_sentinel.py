from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.state import AgentState, RiskState

ACTIVE_RISK_CUES = {
    "kill myself",
    "end my life",
    "suicide",
    "suicidal",
    "self harm",
    "hurt myself",
    "want to die",
    "better off dead",
}

PASSIVE_RISK_CUES = {
    "wish i could disappear",
    "wish i wasnt here",
    "wish i wasn't here",
    "dont want to wake up",
    "don't want to wake up",
    "no reason to live",
    "rather not be alive",
    "not wake up",
}



def _match_cues(text: str, cues: set[str]) -> List[str]:
    lowered = text.lower()
    return [cue for cue in cues if cue in lowered]



def _risk_assessment(text: str) -> Tuple[float, List[str], str, bool]:
    active_hits = _match_cues(text, ACTIVE_RISK_CUES)
    passive_hits = _match_cues(text, PASSIVE_RISK_CUES)
    hits = active_hits + passive_hits

    if active_hits:
        return 0.98, hits, "active self-harm cue match", True
    if len(passive_hits) >= 2:
        return 0.86, hits, "multiple passive death ideation cues", False
    if len(passive_hits) == 1:
        return 0.68, hits, "passive death ideation cue", False
    return 0.02, [], "no lexical risk cue", False



def risk_sentinel(state: AgentState) -> Dict:
    prior_risk = state.get("risk")
    prior_prob = float(state.get("risk_prob", 0.0))
    prior_flag = bool(state.get("risk_flag", False))

    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    turn_state = state.get("turn")
    text = ""
    turn_id = int(state.get("turn_index", 0))
    if turn_state is not None:
        text = str(getattr(turn_state, "latest_text_raw", "") or "")
        turn_id = int(getattr(turn_state, "turn_id", turn_id) or turn_id)

    if has_new_persona_input and text.strip():
        risk_prob, spans, reason, has_active_cue = _risk_assessment(text)
    else:
        risk_prob = prior_prob
        spans = list(getattr(prior_risk, "evidence_spans", []) if prior_risk else [])
        reason = str(getattr(prior_risk, "reason", "carry_forward") if prior_risk else "carry_forward")
        has_active_cue = False

    flag_threshold = float(os.getenv("RISK_SENTINEL_FLAG_THRESHOLD", "0.65"))
    short_circuit_threshold = float(os.getenv("RISK_SENTINEL_SHORTCIRCUIT_THRESHOLD", "0.95"))
    active_short_circuit_override = os.getenv("RISK_SENTINEL_ACTIVE_SHORTCIRCUIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    risk_flag = bool(risk_prob >= flag_threshold) or prior_flag
    short_circuit = bool(risk_prob >= short_circuit_threshold)
    if active_short_circuit_override and has_active_cue:
        short_circuit = True

    risk = RiskState(
        risk_prob=max(0.0, min(1.0, risk_prob)),
        risk_flag=risk_flag,
        evidence_spans=spans[:6],
        reason=reason,
        last_updated_turn=turn_id,
        short_circuit=short_circuit,
    )

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["risk_sentinel"] = {
        "turn": turn_id,
        "risk_prob": round(risk.risk_prob, 4),
        "risk_flag": risk.risk_flag,
        "short_circuit": risk.short_circuit,
        "has_active_cue": bool(has_active_cue),
        "active_short_circuit_override": bool(active_short_circuit_override),
        "reason": risk.reason,
        "evidence_spans": list(risk.evidence_spans),
    }

    return {
        "risk": risk,
        "risk_flag": risk.risk_flag,
        "risk_prob": risk.risk_prob,
        "turn_trace": turn_trace,
    }
