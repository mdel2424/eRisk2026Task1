from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from persona.sim_behavior import _build_deterministic_reply_payload, build_deterministic_reply, normalize_response, response_style_flags


@dataclass
class SimulatedPersona:
    persona_id: str
    bdi_scores: Dict[int, int]
    evasive: bool = True
    context_window: int = 8
    rng_seed: int = 31
    family: str = "mixed_moderate"
    split: str = "eval"
    context_tag: str = "routine_stable"
    style_tag: str = "open_but_flat"
    behavior_params: Dict[str, float | str] = field(default_factory=dict)
    template_bank: str = "default"
    last_response: str = field(default="", init=False)
    recent_responses: List[str] = field(default_factory=list, init=False)
    responses_total: int = field(default=0, init=False)
    hedged_response_count: int = field(default=0, init=False)
    qualifier_response_count: int = field(default=0, init=False)
    deflect_response_count: int = field(default=0, init=False)
    context_anchor_count: int = field(default=0, init=False)
    mixed_answer_count: int = field(default=0, init=False)
    soft_denial_count: int = field(default=0, init=False)
    baseline_comparison_count: int = field(default=0, init=False)
    opening_summary_count: int = field(default=0, init=False)
    contrastive_negative_count: int = field(default=0, init=False)
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
            context_tag=self.context_tag,
            style_tag=self.style_tag,
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
            "qualifier_response_count": int(self.qualifier_response_count),
            "deflect_response_count": int(self.deflect_response_count),
            "context_anchor_count": int(self.context_anchor_count),
            "mixed_answer_count": int(self.mixed_answer_count),
            "soft_denial_count": int(self.soft_denial_count),
            "baseline_comparison_count": int(self.baseline_comparison_count),
            "opening_summary_count": int(self.opening_summary_count),
            "contrastive_negative_count": int(self.contrastive_negative_count),
            "avg_response_words": round(float(avg_words), 3),
            "response_words_total": int(self.response_words_total),
        }

    def reply(self, history: List[dict], probe_intent: Dict[str, object]) -> str:
        base, mode_name = _build_deterministic_reply_payload(
            family=self.family,
            split=self.split,
            context_tag=self.context_tag,
            style_tag=self.style_tag,
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
        if flags.get("qualifier"):
            self.qualifier_response_count += 1
        if flags.get("deflect"):
            self.deflect_response_count += 1
        if flags.get("context_anchor"):
            self.context_anchor_count += 1
        if flags.get("mixed_answer"):
            self.mixed_answer_count += 1
        if flags.get("soft_denial"):
            self.soft_denial_count += 1
        if flags.get("baseline_comparison"):
            self.baseline_comparison_count += 1
        if mode_name == "opening_summary":
            self.opening_summary_count += 1
        if mode_name == "contrastive_negative":
            self.contrastive_negative_count += 1
        self.response_words_total += len(text.split())
        return text
