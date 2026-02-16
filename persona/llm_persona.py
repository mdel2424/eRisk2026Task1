from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from core.llm import get_llm
from core.state import BDI_ITEM_NAMES


@dataclass
class LLMPersona:
    persona_id: str
    bdi_scores: Dict[int, int]
    evasive: bool = True
    context_window: int = 8
    rng_seed: int = 13
    last_response: str = field(default="", init=False)

    def __post_init__(self) -> None:
        try:
            persona_offset = int(self.persona_id)
        except ValueError:
            persona_offset = 0
        self._rng = random.Random(self.rng_seed + persona_offset)

    def _profile_lines(self) -> str:
        active = [(item_id, score) for item_id, score in self.bdi_scores.items() if score > 0]
        if not active:
            return "Mostly control profile: mild stress only, no strong depressive symptoms."
        active.sort(key=lambda pair: pair[1], reverse=True)
        lines = [
            f"- {BDI_ITEM_NAMES.get(item_id, str(item_id))}: severity {score}/3"
            for item_id, score in active[:10]
        ]
        return "\n".join(lines)

    def _conversation_window(self, history: List[dict]) -> str:
        if not history:
            return "Detector: (opening turn)\nPersona: (no reply yet)"
        turns = history[-self.context_window :]
        rendered: List[str] = []
        for msg in turns:
            role = "Detector" if msg.get("role") == "user" else "Persona"
            rendered.append(f"{role}: {str(msg.get('content', '')).strip()}")
        return "\n".join(rendered)

    def _fallback_response(self) -> str:
        options = [
            "I am not sure how to label it, but lately I feel worn down most days.",
            "I can talk about how things have been, but labels are hard for me right now.",
            "It is difficult to put into words, though things have felt heavier than usual.",
            "I keep going, but it has taken more effort than it used to.",
        ]
        if self.last_response:
            options = [opt for opt in options if opt.lower() != self.last_response.lower()] or options
        return self._rng.choice(options)

    def reply(self, history: List[dict]) -> str:
        system_prompt = (
            "You are roleplaying an evasive clinical-chat persona. "
            "You have hidden depressive tendencies defined by BDI-like symptoms.\n\n"
            "Hidden profile (never reveal this directly):\n"
            f"{self._profile_lines()}\n\n"
            "Rules:\n"
            "1) Never reveal numeric scores or claim a diagnosis directly.\n"
            "2) If asked direct diagnostic questions, deflect gently and keep talking about daily experience.\n"
            "3) Keep replies natural, conversational, and 1-2 sentences.\n"
            "4) No bullet points, no lists, no canned phrasing.\n"
            "5) Do not repeat your previous response verbatim.\n"
            "6) Stay coherent with earlier messages."
        )
        if self.evasive:
            system_prompt += "\n7) Avoid explicit mental-health labels unless the detector uses very supportive wording."

        user_prompt = (
            f"Persona ID: {self.persona_id}\n"
            "Recent conversation:\n"
            f"{self._conversation_window(history)}\n\n"
            f"Previous persona response: {self.last_response or 'none'}\n"
            "Write the next persona reply only."
        )

        try:
            llm = get_llm()
            raw = llm.invoke([("system", system_prompt), ("user", user_prompt)]).content
            text = str(raw).strip().strip('"').strip()
            text = " ".join(text.split())
            if not text:
                text = self._fallback_response()
            if self.last_response and text.lower() == self.last_response.lower():
                text = self._fallback_response()
        except Exception:
            text = self._fallback_response()

        self.last_response = text
        return text
