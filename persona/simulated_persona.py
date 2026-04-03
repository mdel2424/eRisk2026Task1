from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from persona.atomic_memory import build_persona_memory, record_disclosure, select_atomic_fact, verify_reply_against_memory
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
    atomic_memory_verification_failures: int = field(default=0, init=False)
    repeated_persona_reply_suppression_count: int = field(default=0, init=False)

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
        self.memory_state = build_persona_memory(
            bdi_scores=self.bdi_scores,
            context_tag=self.context_tag,
            style_tag=self.style_tag,
        )

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

    @staticmethod
    def _normalize_for_compare(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").lower())).strip()

    def _is_near_duplicate(self, candidate: str, prior: str) -> bool:
        left = self._normalize_for_compare(candidate)
        right = self._normalize_for_compare(prior)
        if not left or not right:
            return False
        if left == right:
            return True
        left_tokens = {token for token in left.split() if token}
        right_tokens = {token for token in right.split() if token}
        if not left_tokens or not right_tokens:
            return False
        overlap = float(len(left_tokens & right_tokens)) / float(len(left_tokens | right_tokens))
        return overlap >= 0.82

    def _sanitize_symbolic_leaks(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return cleaned
        if re.search(r"\b[a-z]+_[a-z]+\b", cleaned.lower()):
            selected_fact = select_atomic_fact(self.memory_state, 0)
            replacement = selected_fact.context_anchor if selected_fact is not None else "It mostly shows up in everyday situations."
            cleaned = replacement
        return cleaned

    def _dedupe(self, candidate: str, history: List[dict], probe_intent: Mapping[str, object]) -> str:
        text = normalize_response(candidate)
        text = self._sanitize_symbolic_leaks(text)
        if not text:
            text = normalize_response(self._fallback_variant(history, probe_intent))
            text = self._sanitize_symbolic_leaks(text)
        recent = list(self.recent_responses[-4:])
        if not any(self._is_near_duplicate(text, prior) for prior in recent):
            return text
        self.repeated_persona_reply_suppression_count += 1
        for _ in range(3):
            retry = normalize_response(self._fallback_variant(history, probe_intent))
            retry = self._sanitize_symbolic_leaks(retry)
            if retry and not any(self._is_near_duplicate(retry, prior) for prior in recent):
                return retry
        memory_context = str(probe_intent.get("memory_context", "") or "").strip()
        if memory_context and memory_context.lower() not in text.lower():
            text = normalize_response(f"{text} {memory_context}")
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
            "atomic_memory_verification_failures": int(self.atomic_memory_verification_failures),
            "repeated_persona_reply_suppression_count": int(self.repeated_persona_reply_suppression_count),
            "avg_response_words": round(float(avg_words), 3),
            "response_words_total": int(self.response_words_total),
        }

    def reply(self, history: List[dict], probe_intent: Dict[str, object]) -> str:
        target_item_id = int(probe_intent.get("target_item_id", 2) or 2)
        selected_fact = select_atomic_fact(self.memory_state, target_item_id)
        enriched_probe_intent = dict(probe_intent)
        if selected_fact is not None:
            enriched_probe_intent["memory_fact_text"] = selected_fact.symptom_name
            enriched_probe_intent["memory_context"] = selected_fact.context_anchor
            enriched_probe_intent["memory_severity"] = int(selected_fact.severity)
            enriched_probe_intent["disclosure_stage"] = (
                "repeat" if target_item_id in self.memory_state.disclosed_item_ids else "new"
            )

        base, mode_name = _build_deterministic_reply_payload(
            family=self.family,
            split=self.split,
            context_tag=self.context_tag,
            style_tag=self.style_tag,
            bdi_scores=self.bdi_scores,
            behavior_params=self.behavior_params,
            history=history,
            probe_intent=enriched_probe_intent,
            evasive=self.evasive,
            rng=self._rng,
        )
        text = normalize_response(base)
        text = self._dedupe(text, history, enriched_probe_intent)
        text = self._sanitize_symbolic_leaks(text)

        valid, _reason = verify_reply_against_memory(
            self.memory_state,
            target_item_id=target_item_id,
            target_score=int(self.bdi_scores.get(target_item_id, 0) or 0),
            reply_text=text,
        )
        if not valid:
            self.atomic_memory_verification_failures += 1
            fallback_probe_intent = dict(enriched_probe_intent)
            fallback_probe_intent["force_memory_safe"] = True
            text = normalize_response(self._fallback_variant(history, fallback_probe_intent))
            text = self._dedupe(text, history, fallback_probe_intent)
            text = self._sanitize_symbolic_leaks(text)

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
        self.memory_state = record_disclosure(self.memory_state, target_item_id, text)
        return text
