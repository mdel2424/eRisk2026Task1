from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

BDI_ITEM_NAMES: Dict[int, str] = {
    1: "Sadness",
    2: "Pessimism",
    3: "Past Failure",
    4: "Loss of Pleasure",
    5: "Guilty Feelings",
    6: "Punishment Feelings",
    7: "Self-Dislike",
    8: "Self-Criticalness",
    9: "Suicidal Thoughts or Wishes",
    10: "Crying",
    11: "Agitation",
    12: "Loss of Interest",
    13: "Indecisiveness",
    14: "Worthlessness",
    15: "Loss of Energy",
    16: "Changes in Sleeping Pattern",
    17: "Irritability",
    18: "Changes in Appetite",
    19: "Concentration Difficulty",
    20: "Tiredness or Fatigue",
    21: "Loss of Interest in Sex",
}

SYMPTOM_NAME_TO_ITEM: Dict[str, int] = {
    name.lower(): item_id for item_id, name in BDI_ITEM_NAMES.items()
}

SPECIALIST_ITEM_MAP: Dict[str, List[int]] = {
    "somatic": [11, 15, 16, 18, 20],
    "cognitive": [2, 3, 5, 7, 8, 14],
    "risk": [9],
}


class EvidenceRecord(BaseModel):
    turn: int = Field(ge=1)
    node: Literal["somatic", "cognitive", "risk"]
    item_id: int = Field(ge=1, le=21)
    symptom_name: str
    direction: Literal["increase", "decrease", "neutral"] = "increase"
    intensity: float = Field(ge=0.0, le=3.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str
    reason: str
    method: str = "llm_extractor"


class ItemBelief(BaseModel):
    item_id: int = Field(ge=1, le=21)
    mean_score: float = Field(ge=0.0, le=3.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(ge=0)
    last_update_turn: int = Field(ge=0)


class RouteDecision(BaseModel):
    turn: int = Field(ge=1)
    chosen_node: Literal["somatic", "cognitive", "risk"]
    policy: str
    reason: str
    target_items: List[int] = Field(default_factory=list)
    expected_gain: float = Field(ge=0.0)


class StopDecision(BaseModel):
    turn: int = Field(ge=1)
    should_stop: bool
    reason: str
    predicted_label: Literal["control", "depressed"]
    predicted_bdi_score: int = Field(ge=0, le=63)
    confidence: float = Field(ge=0.0, le=1.0)


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[List[dict], operator.add]
    turn_index: int
    persona_id: Optional[str]
    last_processed_persona_msg_idx: int
    has_new_persona_input: bool

    # Routing
    next_node: str
    active_node: str
    route_history: Annotated[List[RouteDecision], operator.add]

    # Evidence + beliefs
    latest_turn_evidence: List[EvidenceRecord]
    evidence_log: Annotated[List[EvidenceRecord], operator.add]
    item_beliefs: Dict[int, ItemBelief]

    # Prediction
    predicted_label: Optional[str]
    predicted_bdi_score: Optional[int]
    predicted_key_symptoms: List[str]
    raw_predicted_label: Optional[str]
    raw_predicted_bdi_score: Optional[int]
    final_item_scores: Dict[int, int]
    module_imputation: Dict[str, dict]
    global_confidence: float
    risk_flag: bool
    should_stop: bool
    stop_history: Annotated[List[StopDecision], operator.add]

    # Explainability (UI)
    turn_trace: Dict[str, dict]
    trace_log: Annotated[List[dict], operator.add]
    failure_counters: Dict[str, int]
    empty_evidence_streak: int
    route_debug: str
    specialist_debug: str
    stop_debug: str
    latest_feature_vector: Dict[str, float]
    calibrator_mode: str
    positive_contributions: List[dict]
    negative_contributions: List[dict]


def _initial_item_beliefs() -> Dict[int, ItemBelief]:
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = ItemBelief(
            item_id=item_id,
            mean_score=0.0,
            uncertainty=1.0,
            support_count=0,
            last_update_turn=0,
        )
    return beliefs


def build_initial_state(persona_id: Optional[str] = None) -> AgentState:
    return AgentState(
        messages=[],
        next_node="cognitive",
        active_node="cognitive",
        turn_index=0,
        last_processed_persona_msg_idx=-1,
        has_new_persona_input=False,
        risk_flag=False,
        route_debug="",
        specialist_debug="",
        stop_debug="",
        turn_trace={},
        trace_log=[],
        failure_counters={},
        empty_evidence_streak=0,
        predicted_label=None,
        predicted_bdi_score=None,
        predicted_key_symptoms=[],
        raw_predicted_label=None,
        raw_predicted_bdi_score=None,
        final_item_scores={},
        module_imputation={},
        should_stop=False,
        persona_id=persona_id,
        evidence_log=[],
        latest_turn_evidence=[],
        item_beliefs=_initial_item_beliefs(),
        route_history=[],
        stop_history=[],
        global_confidence=0.0,
        latest_feature_vector={},
        calibrator_mode="deterministic_default",
        positive_contributions=[],
        negative_contributions=[],
    )


def symptom_name_from_item(item_id: int) -> str:
    return BDI_ITEM_NAMES.get(item_id, f"Item {item_id}")


def top_symptoms_from_beliefs(item_beliefs: Dict[int, ItemBelief], limit: int = 4) -> List[str]:
    ranked = sorted(
        item_beliefs.items(),
        key=lambda pair: pair[1].mean_score,
        reverse=True,
    )
    top = [symptom_name_from_item(item_id) for item_id, belief in ranked if belief.mean_score > 0.25]
    return top[:limit]


def top_symptoms_from_scores(scores_by_item: Dict[int, int], limit: int = 4) -> List[str]:
    ranked = sorted(scores_by_item.items(), key=lambda pair: pair[1], reverse=True)
    top = [symptom_name_from_item(item_id) for item_id, score in ranked if score > 0]
    return top[:limit]


def bump_failure_counter(
    counters: Dict[str, int],
    key: str,
    amount: int = 1,
) -> Dict[str, int]:
    next_counters = dict(counters or {})
    next_counters[key] = int(next_counters.get(key, 0)) + max(0, int(amount))
    return next_counters
