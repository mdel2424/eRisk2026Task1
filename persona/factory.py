from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Protocol

from core.runtime_policy import resolve_persona_backend
from core.state import top_symptoms_from_scores
from persona.llm_persona import LLMPersona
from persona.openrouter_persona import OpenRouterSimPersona


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
    source: str = "synthetic"
    has_ground_truth: bool = True

    @property
    def bdi_total(self) -> int:
        if not self.has_ground_truth:
            return 0
        return min(sum(self.bdi_scores.values()), 63)

    @property
    def key_symptoms(self) -> List[str]:
        if not self.has_ground_truth:
            return []
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

        profiles.append(
            PersonaProfile(
                persona_id=str(idx),
                bdi_scores=scores,
                depressed=depressed,
                source="synthetic",
                has_ground_truth=True,
            )
        )

    return profiles


def create_persona(profile: PersonaProfile) -> PersonaResponder:
    backend = resolve_persona_backend()
    if backend == "openrouter_sim":
        return OpenRouterSimPersona(persona_id=profile.persona_id, bdi_scores=profile.bdi_scores, evasive=True)
    if backend != "hf_adapter":
        raise ValueError(f"Unsupported PERSONA_BACKEND: {backend}")
    return LLMPersona(persona_id=profile.persona_id, bdi_scores=profile.bdi_scores, evasive=True)


def split_synthetic_profiles(
    profiles: List[PersonaProfile],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, List[PersonaProfile]]:
    synthetic = [profile for profile in profiles if profile.source == "synthetic"]
    rng = random.Random(seed)
    shuffled = synthetic[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if n >= 3:
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        val_end = min(val_end, n - 1)

    return {
        "synthetic_train": shuffled[:train_end],
        "synthetic_val": shuffled[train_end:val_end],
        "synthetic_test": shuffled[val_end:],
    }


def build_official_tracking_profiles(persona_ids: List[str]) -> List[PersonaProfile]:
    profiles: List[PersonaProfile] = []
    for persona_id in persona_ids:
        scores = {item_id: 0 for item_id in range(1, 22)}
        profiles.append(
            PersonaProfile(
                persona_id=persona_id,
                bdi_scores=scores,
                depressed=False,
                source="official",
                has_ground_truth=False,
            )
        )
    return profiles
