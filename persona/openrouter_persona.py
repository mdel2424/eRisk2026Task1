from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

from core.llm import get_persona_openrouter_llm
from core.state import BDI_ITEM_NAMES
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
    generator_version: str = "sim_v2"
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

    def _active_symptom_names(self) -> List[str]:
        active = [
            BDI_ITEM_NAMES.get(item_id, f"Item {item_id}")
            for item_id, score in self.bdi_scores.items()
            if int(score) > 0
        ]
        return active[:10]

    def _paraphrase_enabled(self) -> bool:
        raw = os.getenv("SIM_PARAPHRASE_ENABLED", "1").strip().lower()
        return raw in {"1", "true", "yes", "y", "on"}

    def _paraphrase_rate(self) -> float:
        try:
            value = float(os.getenv("SIM_PARAPHRASE_RATE", "0.5"))
        except ValueError:
            value = 0.5
        return max(0.0, min(1.0, value))

    def _latest_detector_question(self, history: List[dict]) -> str:
        for msg in reversed(history):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return ""

    def _paraphrase(self, base_text: str, history: List[dict]) -> str:
        llm = get_persona_openrouter_llm()
        latest_question = self._latest_detector_question(history)
        allowed = ", ".join(self._active_symptom_names()) or "general stress only"

        prompt = (
            "Rewrite the draft patient reply to sound natural while preserving meaning and severity. "
            "Use a cooperative but hedged style: answer first, soften uncertainty second. "
            "Keep first-person voice, 1-2 short sentences, no bullets, no diagnosis labels. "
            "Include one concrete life detail when available (work/family/routine/messages). "
            "Avoid hard refusal unless directly asked for diagnostic labels. "
            "Do not add symptoms not supported by the allowed symptom set. "
            f"Allowed symptoms: {allowed}. "
            f"Context split={self.split}, family={self.family}, template_bank={self.template_bank}. "
            f"Latest clinician question: {latest_question or 'none'}. "
            f"Draft reply: {base_text}"
        )

        text = llm.invoke([("system", prompt)]).content
        return normalize_response(text)

    def _fallback_variant(self, history: List[dict]) -> str:
        return build_deterministic_reply(
            family=self.family,
            split=self.split,
            bdi_scores=self.bdi_scores,
            behavior_params=self.behavior_params,
            history=history,
            evasive=self.evasive,
            rng=self._rng,
        )

    def _dedupe(self, candidate: str, history: List[dict]) -> str:
        lowered_recent = {text.lower().strip() for text in self.recent_responses[-4:]}
        text = normalize_response(candidate)
        if not text:
            text = normalize_response(self._fallback_variant(history))
        if text.lower().strip() not in lowered_recent:
            return text
        for _ in range(3):
            retry = normalize_response(self._fallback_variant(history))
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

    def reply(self, history: List[dict]) -> str:
        base = build_deterministic_reply(
            family=self.family,
            split=self.split,
            bdi_scores=self.bdi_scores,
            behavior_params=self.behavior_params,
            history=history,
            evasive=self.evasive,
            rng=self._rng,
        )
        text = normalize_response(base)

        if self._paraphrase_enabled() and self._rng.random() < self._paraphrase_rate():
            try:
                candidate = self._paraphrase(base, history)
                if candidate:
                    text = candidate
            except Exception:
                text = normalize_response(base)

        text = self._dedupe(text, history)

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
