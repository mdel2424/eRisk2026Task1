from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Protocol

from core.state import top_symptoms_from_scores

SplitName = Literal["train", "val", "test"]


class PersonaResponder(Protocol):
    persona_id: str
    bdi_scores: Dict[int, int]

    def reply(self, history: List[dict], probe_intent: Dict[str, object]) -> str:
        ...


@dataclass
class PersonaProfile:
    persona_id: str
    split: SplitName
    family: str
    bdi_scores: Dict[int, int]
    depressed: bool
    source: str = "synthetic"
    has_ground_truth: bool = True
    behavior_params: Dict[str, float | str] = None  # type: ignore[assignment]
    template_bank: str = "train_bank_v1"
    generation_seed: int = 0
    generator_version: str = "sim_v3"

    def __post_init__(self) -> None:
        if self.behavior_params is None:
            self.behavior_params = {
                "evasiveness": 0.45,
                "verbosity": 0.45,
                "contradiction": 0.08,
                "affect_volatility": 0.2,
                "hedge_rate": 0.65,
                "normalization_rate": 0.45,
                "context_anchor_rate": 0.55,
                "direct_answer_rate": 0.78,
            }

    @property
    def bdi_total(self) -> int:
        if not self.has_ground_truth:
            return 0
        return min(sum(int(score) for score in self.bdi_scores.values()), 63)

    @property
    def key_symptoms(self) -> List[str]:
        if not self.has_ground_truth:
            return []
        return top_symptoms_from_scores(self.bdi_scores, limit=4)

    @property
    def has_risk_signal(self) -> bool:
        return int(self.bdi_scores.get(9, 0)) > 0
