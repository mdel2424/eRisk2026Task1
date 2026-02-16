import operator
from collections import Counter
from typing import Annotated, Dict, List, Optional, TypedDict

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

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    symptom_hits: Annotated[List[str], operator.add]
    next_node: str
    turn_index: int
    depression_score: float
    risk_flag: bool
    route_debug: str
    specialist_debug: str
    stop_debug: str
    predicted_label: Optional[str]
    predicted_bdi_score: Optional[int]
    predicted_key_symptoms: List[str]
    should_stop: bool
    persona_id: Optional[str]


def build_initial_state(persona_id: Optional[str] = None) -> AgentState:
    return AgentState(
        messages=[],
        symptom_hits=[],
        next_node="cognitive",
        turn_index=0,
        depression_score=0.0,
        risk_flag=False,
        route_debug="",
        specialist_debug="",
        stop_debug="",
        predicted_label=None,
        predicted_bdi_score=None,
        predicted_key_symptoms=[],
        should_stop=False,
        persona_id=persona_id,
    )  # type: ignore[typeddict-item]


def top_symptoms_from_scores(scores_by_item: Dict[int, int], limit: int = 4) -> List[str]:
    ranked = sorted(scores_by_item.items(), key=lambda pair: pair[1], reverse=True)
    top = [BDI_ITEM_NAMES[item_id] for item_id, score in ranked if score > 0]
    return top[:limit]


def bdi_score_from_scalar(depression_score: float) -> int:
    bounded = max(0.0, min(1.0, depression_score))
    return min(63, int(round(bounded * 63)))


def top_symptoms_from_hits(symptom_hits: List[str], limit: int = 4) -> List[str]:
    if not symptom_hits:
        return []
    counts = Counter(symptom_hits)
    return [name for name, _ in counts.most_common(limit)]
