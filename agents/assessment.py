from __future__ import annotations

import json
from typing import Dict, List

from core.calibration import build_feature_vector, get_calibrator_bundle, predict_with_explanations
from core.llm import get_llm
from core.prompts import get_prompt
from core.state import (
    AgentState,
    BDI_ITEM_NAMES,
    EvidenceRecord,
    ItemBelief,
    SPECIALIST_ITEM_MAP,
    SYMPTOM_NAME_TO_ITEM,
    top_symptoms_from_beliefs,
)


def _latest_persona_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _recent_context(state: AgentState, limit: int = 4) -> str:
    turns = state.get("messages", [])[-limit:]
    lines = []
    for msg in turns:
        role = "Detector" if msg.get("role") == "user" else "Persona"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _parse_json_object(raw_text: str) -> Dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _coerce_item_id(raw_item_id, raw_symptom_name: str) -> int | None:
    try:
        item_id = int(raw_item_id)
        if 1 <= item_id <= 21:
            return item_id
    except (TypeError, ValueError):
        pass

    symptom = raw_symptom_name.strip().lower()
    if symptom in SYMPTOM_NAME_TO_ITEM:
        return SYMPTOM_NAME_TO_ITEM[symptom]
    return None


def _coerce_evidence_record(node_name: str, turn: int, item: Dict, fallback_text: str) -> EvidenceRecord | None:
    symptom_name = str(item.get("symptom_name", "")).strip()
    item_id = _coerce_item_id(item.get("item_id"), symptom_name)
    if item_id is None:
        return None
    if not symptom_name:
        symptom_name = BDI_ITEM_NAMES[item_id]

    direction = str(item.get("direction", "increase")).strip().lower()
    if direction not in {"increase", "decrease", "neutral"}:
        direction = "increase"

    try:
        intensity = float(item.get("intensity", 0.0))
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(3.0, intensity))

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_text = str(item.get("evidence_text", "")).strip() or fallback_text[:220]
    reason = str(item.get("reason", "")).strip() or "implicit affective signal"
    method = str(item.get("method", "llm_extractor")).strip() or "llm_extractor"

    return EvidenceRecord(
        turn=turn,
        node=node_name,  # type: ignore[arg-type]
        item_id=item_id,
        symptom_name=symptom_name,
        direction=direction,  # type: ignore[arg-type]
        intensity=intensity,
        confidence=confidence,
        evidence_text=evidence_text,
        reason=reason,
        method=method,
    )


def extract_evidence(state: AgentState) -> Dict:
    node_name = str(state.get("active_node", "cognitive"))
    latest_message = _latest_persona_message(state)
    turn = int(state.get("turn_index", 0)) + 1

    if node_name not in {"somatic", "cognitive", "risk"}:
        node_name = "cognitive"

    evidence_records: List[EvidenceRecord] = []
    if latest_message.strip():
        prompt = get_prompt("evidence_extraction").format(
            node_name=node_name,
            recent_context=_recent_context(state) or "none",
            latest_message=latest_message,
        )
        try:
            llm = get_llm()
            raw = llm.invoke([("system", prompt)]).content
            parsed = _parse_json_object(str(raw))
            items = parsed.get("evidence", [])
            if isinstance(items, list):
                for raw_item in items:
                    if not isinstance(raw_item, dict):
                        continue
                    record = _coerce_evidence_record(node_name, turn, raw_item, latest_message)
                    if record is not None:
                        evidence_records.append(record)
        except Exception:
            evidence_records = []

    summary = (
        f"{state.get('specialist_debug', '')} | evidence_count={len(evidence_records)}"
        if state.get("specialist_debug")
        else f"Evidence extraction: count={len(evidence_records)}"
    )
    return {
        "latest_turn_evidence": evidence_records,
        "evidence_log": evidence_records,
        "specialist_debug": summary,
    }


def _coerce_belief(item_id: int, value) -> ItemBelief:
    if isinstance(value, ItemBelief):
        return value
    if isinstance(value, dict):
        try:
            return ItemBelief(**value)
        except Exception:
            pass
    return ItemBelief(
        item_id=item_id,
        mean_score=0.0,
        uncertainty=1.0,
        support_count=0,
        last_update_turn=0,
    )


def _update_single_belief(belief: ItemBelief, evidence: EvidenceRecord, turn: int) -> ItemBelief:
    prior_n = belief.support_count
    new_n = prior_n + 1
    weighted_observation = evidence.intensity * evidence.confidence
    new_mean = ((belief.mean_score * prior_n) + weighted_observation) / new_n
    new_uncertainty = max(0.05, 1.0 / (new_n + 1.0))
    return ItemBelief(
        item_id=belief.item_id,
        mean_score=max(0.0, min(3.0, new_mean)),
        uncertainty=max(0.0, min(1.0, new_uncertainty)),
        support_count=new_n,
        last_update_turn=turn,
    )


def update_beliefs(state: AgentState) -> Dict:
    turn = int(state.get("turn_index", 0)) + 1
    latest_evidence = list(state.get("latest_turn_evidence", []))
    prior_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = _coerce_belief(item_id, prior_beliefs.get(item_id))

    for record in latest_evidence:
        if 1 <= record.item_id <= 21:
            beliefs[record.item_id] = _update_single_belief(beliefs[record.item_id], record, turn)

    risk_flag = bool(state.get("risk_flag", False)) or any(
        rec.item_id == 9 and rec.intensity >= 0.75 for rec in latest_evidence
    )

    evidence_confidences = [float(rec.confidence) for rec in latest_evidence]
    feature_vector = build_feature_vector(beliefs, evidence_confidences, risk_flag)
    bundle = get_calibrator_bundle()
    prediction = predict_with_explanations(feature_vector, bundle)

    positive = [
        {
            "feature": item.feature,
            "value": item.value,
            "weight": item.weight,
            "impact": item.impact,
        }
        for item in prediction.positive_contributions
    ]
    negative = [
        {
            "feature": item.feature,
            "value": item.value,
            "weight": item.weight,
            "impact": item.impact,
        }
        for item in prediction.negative_contributions
    ]

    active_node = str(state.get("active_node", "cognitive"))
    node_items = SPECIALIST_ITEM_MAP.get(active_node, [])
    node_summary = ", ".join(str(item_id) for item_id in node_items[:3]) or "n/a"
    debug = (
        f"{state.get('specialist_debug', '')} | "
        f"beliefs_updated={len(latest_evidence)}; node_items={node_summary}; "
        f"cal_mode={prediction.mode}; conf={prediction.global_confidence:.2f}"
    )

    return {
        "item_beliefs": beliefs,
        "risk_flag": risk_flag,
        "latest_feature_vector": feature_vector,
        "calibrator_mode": prediction.mode,
        "positive_contributions": positive,
        "negative_contributions": negative,
        "global_confidence": prediction.global_confidence,
        "predicted_bdi_score": prediction.predicted_bdi_score,
        "predicted_label": prediction.predicted_label,
        "predicted_key_symptoms": top_symptoms_from_beliefs(beliefs, limit=4),
        "specialist_debug": debug,
    }
