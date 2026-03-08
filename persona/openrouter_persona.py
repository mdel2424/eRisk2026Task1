from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from persona.sim_behavior import build_deterministic_reply, normalize_response, response_style_flags


@dataclass
class OpenRouterSimPersona:
    persona_id: str
    bdi_scores: Dict[int, int]
    evasive: bool = True
    context_window: int = 8
    rng_seed: int = 31
    family: str = "mixed_moderate"
    split: str = "test"
    behavior_params: Dict[str, float | str] = field(default_factory=dict)
    template_bank: str = "test_bank_v1"
    generator_version: str = "sim_v4"
    last_response: str = field(default="", init=False)
    recent_responses: List[str] = field(default_factory=list, init=False)
    responses_total: int = field(default=0, init=False)
    hedged_response_count: int = field(default=0, init=False)
    deflect_response_count: int = field(default=0, init=False)
    context_anchor_count: int = field(default=0, init=False)
    response_words_total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        try:
            pid = int(self.persona_id.split("-")[-1])
        except Exception:
            pid = 0
        self._rng = random.Random(self.rng_seed + pid)
        if not self.behavior_params:
            self.behavior_params = {
                "evasiveness": 0.45,
                "verbosity": 0.45,
                "contradiction": 0.08,
                "affect_volatility": 0.2,
            }

    def _fallback_variant(self, history: List[dict], probe_intent: Mapping[str, object]) -> str:
        return build_deterministic_reply(
            family=self.family,
            split=self.split,
            bdi_scores=self.bdi_scores,
            behavior_params=self.behavior_params,
            history=history,
            probe_intent=dict(probe_intent),
            evasive=self.evasive,
            rng=self._rng,
        )

    def _dedupe(self, candidate: str, history: List[dict], probe_intent: Mapping[str, object]) -> str:
        lowered_recent = {text.lower().strip() for text in self.recent_responses[-4:]}
        text = normalize_response(candidate)
        if not text:
            text = normalize_response(self._fallback_variant(history, probe_intent))
        if text.lower().strip() not in lowered_recent:
            return text
        for _ in range(3):
            retry = normalize_response(self._fallback_variant(history, probe_intent))
            if retry and retry.lower().strip() not in lowered_recent:
                return retry
        return text

    def style_stats(self) -> Dict[str, float | int]:
        avg_words = (self.response_words_total / self.responses_total) if self.responses_total else 0.0
        return {
            "responses_total": int(self.responses_total),
            "hedged_response_count": int(self.hedged_response_count),
            "deflect_response_count": int(self.deflect_response_count),
            "context_anchor_count": int(self.context_anchor_count),
            "avg_response_words": round(float(avg_words), 3),
        }

    def reply(self, history: List[dict], probe_intent: Dict[str, object]) -> str:
        base = build_deterministic_reply(
            family=self.family,
            split=self.split,
            bdi_scores=self.bdi_scores,
            behavior_params=self.behavior_params,
            history=history,
            probe_intent=probe_intent,
            evasive=self.evasive,
            rng=self._rng,
        )
        text = normalize_response(base)
        text = self._dedupe(text, history, probe_intent)

        if not text:
            text = "I can describe my day-to-day if that helps more."

        self.last_response = text
        self.recent_responses.append(text)
        if len(self.recent_responses) > 6:
            self.recent_responses = self.recent_responses[-6:]
        flags = response_style_flags(text)
        self.responses_total += 1
        if flags.get("hedged"):
            self.hedged_response_count += 1
        if flags.get("deflect"):
            self.deflect_response_count += 1
        if flags.get("context_anchor"):
            self.context_anchor_count += 1
        self.response_words_total += len(text.split())
        return text
