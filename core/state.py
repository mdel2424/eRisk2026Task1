from __future__ import annotations

import math
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

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

MAX_ITEM_ENTROPY = 2.0


def _normalize_distribution(values: List[float]) -> List[float]:
    clipped = [max(1e-8, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [value / total for value in clipped]


def _entropy(posterior: List[float]) -> float:
    value = 0.0
    for prob in posterior:
        p = max(1e-12, min(1.0, float(prob)))
        value -= p * math.log2(p)
    return max(0.0, min(MAX_ITEM_ENTROPY, value))


def _expected_score(posterior: List[float]) -> float:
    return max(0.0, min(3.0, sum(idx * prob for idx, prob in enumerate(posterior))))


def posterior_from_expected_score(expected_score: float) -> List[float]:
    target = max(0.0, min(3.0, float(expected_score)))
    weights = [1.0 / (1.0 + abs(float(idx) - target)) for idx in range(4)]
    return _normalize_distribution(weights)


class TurnState(BaseModel):
    latest_text_raw: str = ""
    latest_text_norm: str = ""
    latest_sentences: List[str] = Field(default_factory=list)
    turn_id: int = Field(default=0, ge=0)


class RiskState(BaseModel):
    risk_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_flag: bool = False
    evidence_spans: List[str] = Field(default_factory=list)
    reason: str = ""
    last_updated_turn: int = Field(default=0, ge=0)
    short_circuit: bool = False


class LikelihoodEvidence(BaseModel):
    item_id: int = Field(ge=1, le=21)
    likelihood: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0], min_length=4, max_length=4)
    spans: List[str] = Field(default_factory=list)
    extract_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extract_intensity: float = Field(default=0.0, ge=0.0, le=3.0)
    evidence_type: str = "llm_extractor"
    symptom_name: str = ""
    direction: Literal["increase", "decrease", "neutral"] = "neutral"
    evidence_id: str = ""
    method_weight_hint: float = Field(default=0.0, ge=0.0, le=2.0)
    precision_gate_action: str = "kept"
    support_increment_blocked: bool = False

    @field_validator("likelihood")
    @classmethod
    def _validate_likelihood(cls, value: List[float]) -> List[float]:
        normalized = [max(1e-8, float(v)) for v in value[:4]]
        if len(normalized) < 4:
            normalized.extend([1.0] * (4 - len(normalized)))
        return normalized


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
    precision_gate_action: str = "kept"
    support_increment_blocked: bool = False


class ItemBelief(BaseModel):
    item_id: int = Field(ge=1, le=21)
    posterior: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25], min_length=4, max_length=4)
    entropy: float = Field(default=MAX_ITEM_ENTROPY, ge=0.0, le=MAX_ITEM_ENTROPY)
    expected_score: float = Field(default=0.0, ge=0.0, le=3.0)
    support_count: int = Field(default=0, ge=0)
    last_update_turn: int = Field(default=0, ge=0)

    @field_validator("posterior")
    @classmethod
    def _normalize_posterior(cls, value: List[float]) -> List[float]:
        return _normalize_distribution(list(value))

    @field_validator("entropy", mode="before")
    @classmethod
    def _coerce_entropy(cls, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = MAX_ITEM_ENTROPY
        return max(0.0, min(MAX_ITEM_ENTROPY, parsed))

    @field_validator("expected_score", mode="before")
    @classmethod
    def _coerce_expected(cls, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        return max(0.0, min(3.0, parsed))

    @property
    def mean_score(self) -> float:
        return float(self.expected_score)

    @property
    def uncertainty(self) -> float:
        return max(0.0, min(1.0, float(self.entropy) / MAX_ITEM_ENTROPY))


class BeliefState(BaseModel):
    items: Dict[int, ItemBelief] = Field(default_factory=dict)


class PolicyMetricsState(BaseModel):
    total_expected_bdi: float = Field(default=0.0, ge=0.0, le=63.0)
    label_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_entropy: float = Field(default=MAX_ITEM_ENTROPY, ge=0.0, le=MAX_ITEM_ENTROPY)
    top_uncertain_items: List[int] = Field(default_factory=list)
    last_ig_estimates: Optional[Dict[int, float]] = Field(default_factory=dict)


class ControlState(BaseModel):
    stop: bool = False
    stop_reason: str = ""


class NextAction(BaseModel):
    target_item_id: int = Field(default=2, ge=1, le=21)
    route: Literal["somatic", "cognitive", "risk"] = "cognitive"
    style: str = "gentle_probe"
    mode: Literal["normal", "wrapup"] = "normal"
    directness: Literal["indirect", "direct"] = "indirect"
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class OutgoingState(BaseModel):
    detector_message: str = ""


class FinalState(BaseModel):
    predicted_bdi_score: int = Field(default=0, ge=0, le=63)
    predicted_label: Literal["control", "depressed"] = "control"
    top_symptoms: List[str] = Field(default_factory=list)
    evidence_report: Dict[str, Any] = Field(default_factory=dict)
    risk_flag: bool = False
    debug_trace: Dict[str, Any] = Field(default_factory=dict)


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

    # New slices
    turn: TurnState
    risk: RiskState
    beliefs: BeliefState
    metrics: PolicyMetricsState
    control: ControlState
    next_action: NextAction
    outgoing: OutgoingState
    final: FinalState

    # Routing
    next_node: str
    active_node: str
    route_history: Annotated[List[RouteDecision], operator.add]

    # Evidence + beliefs
    latest_turn_likelihoods: List[LikelihoodEvidence]
    latest_turn_evidence: List[EvidenceRecord]
    evidence_log: Annotated[List[EvidenceRecord], operator.add]
    item_beliefs: Dict[int, ItemBelief]
    item_evidence_memory: Dict[int, List[str]]
    item_direction_tally: Dict[int, Dict[str, int]]

    # Compatibility predictions
    predicted_label: Optional[str]
    predicted_bdi_score: Optional[int]
    predicted_key_symptoms: List[str]
    predicted_key_item_ids: List[int]
    raw_predicted_label: Optional[str]
    raw_predicted_bdi_score: Optional[int]
    final_item_scores: Dict[int, int]
    module_imputation: Dict[str, dict]
    global_confidence: float
    risk_flag: bool
    risk_prob: float
    should_stop: bool
    stop_history: Annotated[List[StopDecision], operator.add]

    # Explainability (UI)
    turn_trace: Dict[str, dict]
    trace_log: Annotated[List[dict], operator.add]
    failure_counters: Dict[str, int]
    empty_evidence_streak: int
    new_items_this_turn: int
    recent_new_items_window: List[int]
    recent_nonempty_window: List[int]
    route_debug: str
    specialist_debug: str
    stop_debug: str


def coerce_item_belief(item_id: int, value: Any) -> ItemBelief:
    if isinstance(value, ItemBelief):
        return value
    if isinstance(value, dict):
        if "posterior" in value:
            try:
                posterior = _normalize_distribution(list(value.get("posterior", [0.25] * 4)))
                return ItemBelief(
                    item_id=int(value.get("item_id", item_id)),
                    posterior=posterior,
                    entropy=_entropy(posterior),
                    expected_score=_expected_score(posterior),
                    support_count=int(value.get("support_count", 0) or 0),
                    last_update_turn=int(value.get("last_update_turn", 0) or 0),
                )
            except Exception:
                pass

        legacy_mean = value.get("mean_score", value.get("expected_score", 0.0))
        legacy_unc = value.get("uncertainty")
        posterior = posterior_from_expected_score(float(legacy_mean or 0.0))
        entropy = _entropy(posterior)
        if legacy_unc is not None:
            try:
                entropy = max(0.0, min(MAX_ITEM_ENTROPY, float(legacy_unc) * MAX_ITEM_ENTROPY))
            except (TypeError, ValueError):
                entropy = _entropy(posterior)
        return ItemBelief(
            item_id=item_id,
            posterior=posterior,
            entropy=entropy,
            expected_score=_expected_score(posterior),
            support_count=int(value.get("support_count", 0) or 0),
            last_update_turn=int(value.get("last_update_turn", 0) or 0),
        )

    return ItemBelief(item_id=item_id)


def _initial_item_beliefs() -> Dict[int, ItemBelief]:
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = ItemBelief(item_id=item_id)
    return beliefs


def build_initial_state(persona_id: Optional[str] = None) -> AgentState:
    beliefs = _initial_item_beliefs()
    return AgentState(
        messages=[],
        next_node="cognitive",
        active_node="cognitive",
        turn_index=0,
        last_processed_persona_msg_idx=-1,
        has_new_persona_input=False,
        turn=TurnState(),
        risk=RiskState(),
        beliefs=BeliefState(items=beliefs),
        metrics=PolicyMetricsState(),
        control=ControlState(),
        next_action=NextAction(),
        outgoing=OutgoingState(),
        final=FinalState(),
        risk_flag=False,
        risk_prob=0.0,
        route_debug="",
        specialist_debug="",
        stop_debug="",
        turn_trace={},
        trace_log=[],
        failure_counters={},
        empty_evidence_streak=0,
        new_items_this_turn=0,
        recent_new_items_window=[],
        recent_nonempty_window=[],
        predicted_label=None,
        predicted_bdi_score=None,
        predicted_key_symptoms=[],
        predicted_key_item_ids=[],
        raw_predicted_label=None,
        raw_predicted_bdi_score=None,
        final_item_scores={},
        module_imputation={},
        should_stop=False,
        persona_id=persona_id,
        evidence_log=[],
        latest_turn_likelihoods=[],
        latest_turn_evidence=[],
        item_beliefs=beliefs,
        item_evidence_memory={},
        item_direction_tally={},
        route_history=[],
        stop_history=[],
        global_confidence=0.0,
    )


def symptom_name_from_item(item_id: int) -> str:
    return BDI_ITEM_NAMES.get(item_id, f"Item {item_id}")


def top_symptoms_from_scores(scores_by_item: Dict[int, int], limit: int = 4) -> List[str]:
    ranked = sorted(scores_by_item.items(), key=lambda pair: (-int(pair[1]), int(pair[0])))
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
