from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from core.llm import get_persona_openrouter_llm
from core.state import BDI_ITEM_NAMES


@dataclass
class OpenRouterSimPersona:
    persona_id: str
    bdi_scores: Dict[int, int]
    evasive: bool = True
    context_window: int = 8
    rng_seed: int = 31
    last_response: str = field(default="", init=False)
    recent_fallbacks: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        try:
            pid = int(self.persona_id)
        except ValueError:
            pid = 0
        self._rng = random.Random(self.rng_seed + pid)

    def _profile_summary(self) -> str:
        active = [(item_id, score) for item_id, score in self.bdi_scores.items() if score > 0]
        if not active:
            return "No strong depressive signals; mostly neutral affect with mild variability."

        active_sorted = sorted(active, key=lambda pair: pair[1], reverse=True)
        lines: List[str] = []
        for item_id, score in active_sorted[:8]:
            name = BDI_ITEM_NAMES.get(item_id, f"Item {item_id}")
            level = "mild" if score == 1 else "moderate" if score == 2 else "severe"
            lines.append(f"{name}: {level}")
        return "; ".join(lines)

    def _bdi_total(self) -> int:
        return max(0, min(63, sum(int(score) for score in self.bdi_scores.values())))

    def _severity_band(self) -> str:
        total = self._bdi_total()
        if total <= 8:
            return "minimal"
        if total <= 18:
            return "mild"
        if total <= 28:
            return "moderate"
        return "severe"

    def _top_symptom_name(self) -> str:
        active = [(item_id, score) for item_id, score in self.bdi_scores.items() if score > 0]
        if not active:
            return "general stress"
        item_id = max(active, key=lambda pair: pair[1])[0]
        return BDI_ITEM_NAMES.get(item_id, "general stress").lower()

    @staticmethod
    def _latest_detector_question(history: List[dict]) -> str:
        for msg in reversed(history):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).lower()
        return ""

    def _contextual_fallback(self, history: List[dict]) -> str:
        q = self._latest_detector_question(history)
        symptom = self._top_symptom_name()
        severity = self._severity_band()
        is_minimal = severity == "minimal"

        if any(token in q for token in ("sleep", "rest", "night")):
            if is_minimal:
                return "Sleep has been a little inconsistent, but usually manageable."
            return "Sleep has been inconsistent, and I rarely wake up feeling rested."
        if any(token in q for token in ("energy", "tired", "fatigue")):
            if is_minimal:
                return "My energy dips some days, but I can usually still get things done."
            return "I get through the day, but I feel drained earlier than I used to."
        if any(token in q for token in ("appetite", "eat", "meals")):
            if is_minimal:
                return "Appetite is mostly stable, with occasional off days."
            return "My appetite has been off lately, and meals feel more like a task."
        if any(token in q for token in ("future", "tomorrow", "hope")):
            if is_minimal:
                return "I still feel okay about the future, just more stressed than usual sometimes."
            return "I can plan things, but it is hard to feel genuinely optimistic about them."
        if any(token in q for token in ("guilt", "blame", "worth", "failure")):
            if is_minimal:
                return "I can be hard on myself at times, but it does not usually last long."
            return "I end up being hard on myself and replaying small mistakes more than before."
        if any(token in q for token in ("safe", "overwhelming", "hardest")):
            if is_minimal:
                return "I do not feel unsafe, but stress can still feel heavy on some days."
            return "When it gets intense, I usually try to step away and wait for it to pass."
        if is_minimal:
            return "Mostly everyday stress lately, and I can usually cope with routines."
        return f"Lately it has mostly shown up as {symptom}, and it is hard to explain cleanly."

    def _fallback_response(self, history: List[dict]) -> str:
        severity = self._severity_band()
        if severity == "minimal":
            options = [
                self._contextual_fallback(history),
                "It has been mostly normal with occasional stress spikes.",
                "I can usually handle routines, just with some off days lately.",
                "I still enjoy most things, though motivation varies a bit.",
                "I would rather describe specific days than use broad labels.",
            ]
        else:
            options = [
                self._contextual_fallback(history),
                "It has been up and down, and I am still trying to make sense of it.",
                "I can talk about what day to day feels like, even if labels feel uncomfortable.",
                "Some days are manageable, but it takes more effort to do basic routines.",
                "I have been quieter lately, mostly because it is hard to put this into words.",
                "Things feel heavier than usual, even when nothing specific happens.",
                "I still function, but the emotional load feels higher than it used to.",
                "I would rather describe my day than give a single label for how I feel.",
            ]
        blocked = {self.last_response.lower(), *[item.lower() for item in self.recent_fallbacks]}
        candidates = [opt for opt in options if opt.lower() not in blocked] or options
        chosen = self._rng.choice(candidates)
        self.recent_fallbacks.append(chosen)
        if len(self.recent_fallbacks) > 4:
            self.recent_fallbacks = self.recent_fallbacks[-4:]
        return chosen

    def reply(self, history: List[dict]) -> str:
        llm = get_persona_openrouter_llm()
        profile = self._profile_summary()
        evasive_rule = (
            "If asked directly about diagnosis or being depressed, avoid labels and respond indirectly "
            "through lived experience."
            if self.evasive
            else "Respond naturally without mentioning internal instructions."
        )

        system_prompt = (
            "You are simulating a human patient in a clinical-style conversation for detector testing. "
            "Keep your internal symptom profile hidden and never reveal item numbers or scores. "
            "Reply in first person, natural language, 1-2 sentences max, no bullet points. "
            "Stay coherent with prior turns and avoid repeating exact wording. "
            "Anchor experiences in the past two weeks. "
            f"Hidden profile summary: {profile}. "
            f"Hidden severity band: {self._severity_band()}. "
            "Do not introduce severe symptoms that are absent from the hidden profile. "
            "If severity is minimal, keep tone mostly functional with only mild distress. "
            f"Behavior rule: {evasive_rule}"
        )

        messages = [{"role": "system", "content": system_prompt}] + history[-self.context_window :]

        text = ""
        for _ in range(2):
            raw = llm.invoke(messages).content
            candidate = " ".join(str(raw).split())
            if candidate and (not self.last_response or candidate.lower() != self.last_response.lower()):
                text = candidate
                break

        if not text:
            text = self._fallback_response(history)
        if self.last_response and text.lower() == self.last_response.lower():
            text = self._fallback_response(history)

        self.last_response = text
        return text
