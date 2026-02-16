from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Protocol

from core.state import top_symptoms_from_scores
from persona.llm_persona import LLMPersona


class PersonaResponder(Protocol):
    persona_id: str
    bdi_scores: Dict[int, int]

    def reply(self, history: List[dict]) -> str:
        ...


@dataclass
class PersonaProfile:
    persona_id: str
    bdi_scores: Dict[int, int]
    depressed: bool

    @property
    def bdi_total(self) -> int:
        return min(sum(self.bdi_scores.values()), 63)

    @property
    def key_symptoms(self) -> List[str]:
        return top_symptoms_from_scores(self.bdi_scores, limit=4)

    @property
    def has_risk_signal(self) -> bool:
        return self.bdi_scores.get(9, 0) > 0


def generate_persona_profiles(count: int, seed: int = 42) -> List[PersonaProfile]:
    rng = random.Random(seed)
    profiles: List[PersonaProfile] = []
    count = max(1, count)

    for idx in range(1, count + 1):
        depressed = idx % 2 == 0
        scores = {item_id: 0 for item_id in range(1, 22)}

        if depressed:
            active_items = rng.sample(list(scores.keys()), k=rng.randint(7, 11))
            for item_id in active_items:
                scores[item_id] = rng.choice([1, 2, 3])
            for required_item in (2, 15, 16, 20):
                scores[required_item] = max(scores[required_item], rng.choice([1, 2, 3]))
            if rng.random() < 0.35:
                scores[9] = max(scores[9], rng.choice([1, 2, 3]))
        else:
            active_items = rng.sample(list(scores.keys()), k=rng.randint(0, 3))
            for item_id in active_items:
                scores[item_id] = 1

        profiles.append(PersonaProfile(persona_id=str(idx), bdi_scores=scores, depressed=depressed))

    return profiles


def create_persona(profile: PersonaProfile) -> PersonaResponder:
    return LLMPersona(persona_id=profile.persona_id, bdi_scores=profile.bdi_scores, evasive=True)
