from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import List, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from core.llm_types import LLMResponse
from core.llm_usage import _record_llm_error, _record_token_usage, _reserve_llm_call
from core.runtime_policy import min_cuda_vram_gb


def _assert_cuda_ready(min_vram_gb: float) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for local inference")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram_gb < min_vram_gb:
        raise RuntimeError(
            f"GPU VRAM is too low: {total_vram_gb:.2f} GB available, requires >= {min_vram_gb:.2f} GB"
        )


@lru_cache(maxsize=4)
def _load_base_chat_model(model_id: str, hf_token: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _assert_cuda_ready(min_vram_gb=min_cuda_vram_gb())

    token = hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": token, "device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


class LocalHFChatLLM:
    def __init__(
        self,
        model_id: str,
        hf_token: str,
        load_in_4bit: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model_id = model_id
        self.tokenizer, self.model = _load_base_chat_model(model_id, hf_token, load_in_4bit)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def _normalize_messages(messages: Sequence[dict | tuple[str, str]]) -> List[dict]:
        normalized: List[dict] = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
                normalized.append({"role": role, "content": content})
            else:
                normalized.append(
                    {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                )
        return normalized

    def invoke(self, messages: Sequence[dict | tuple[str, str]]) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        if not normalized:
            return LLMResponse(content="", backend="local_hf", model_id=self.model_id, latency_ms=0.0)

        started = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(
            normalized,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)

        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(0.0, self.temperature),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = output[0][inputs.shape[-1] :]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            content=text,
            backend="local_hf",
            model_id=self.model_id,
            latency_ms=latency_ms,
        )


class OpenRouterChatLLM:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_sec: int = 120,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for openrouter backend")
        self.model_id = model_id
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    @staticmethod
    def _normalize_messages(messages: Sequence[dict | tuple[str, str]]) -> List[dict]:
        normalized: List[dict] = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
                normalized.append({"role": str(role), "content": str(content)})
            else:
                normalized.append(
                    {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                )
        return normalized

    @staticmethod
    def _collect_text(value: object) -> List[str]:
        chunks: List[str] = []
        if value is None:
            return chunks
        if isinstance(value, str):
            text = value.strip()
            if text:
                chunks.append(text)
            return chunks
        if isinstance(value, (int, float, bool)):
            chunks.append(str(value))
            return chunks
        if isinstance(value, list):
            for item in value:
                chunks.extend(OpenRouterChatLLM._collect_text(item))
            return chunks
        if isinstance(value, dict):
            # Prefer common text-bearing keys first.
            preferred_keys = (
                "text",
                "content",
                "output_text",
                "value",
                "message",
                "final",
            )
            for key in preferred_keys:
                if key in value:
                    chunks.extend(OpenRouterChatLLM._collect_text(value.get(key)))
            # Generic fallback: walk remaining keys.
            for key, nested in value.items():
                if key in preferred_keys:
                    continue
                chunks.extend(OpenRouterChatLLM._collect_text(nested))
            return chunks
        return chunks

    @staticmethod
    def _extract_content(payload: dict) -> str:
        if not isinstance(payload, dict):
            return ""

        candidates: List[object] = []
        choices = payload.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                # OpenAI/OpenRouter chat format
                message = first.get("message")
                if message is not None:
                    candidates.append(message)
                # Legacy completion-style text fallback
                if "text" in first:
                    candidates.append(first.get("text"))
                # Some providers return delta-like structures even in non-stream calls
                if "delta" in first:
                    candidates.append(first.get("delta"))

        # Responses-style / provider-specific fallbacks.
        candidates.extend(
            [
                payload.get("output_text"),
                payload.get("output"),
                payload.get("response"),
                payload.get("data"),
                payload.get("message"),
            ]
        )

        for candidate in candidates:
            parts = OpenRouterChatLLM._collect_text(candidate)
            if parts:
                text = " ".join(part for part in parts if part).strip()
                if text:
                    return text
        return ""

    def invoke(self, messages: Sequence[dict | tuple[str, str]]) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        if not normalized:
            return LLMResponse(content="", backend="openrouter", model_id=self.model_id, latency_ms=0.0)

        _reserve_llm_call()
        started = time.perf_counter()

        payload = {
            "model": self.model_id,
            "messages": normalized,
            "temperature": max(0.0, self.temperature),
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name

        req = urllib_request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            _record_llm_error()
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            _record_llm_error()
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _record_llm_error()
            raise RuntimeError(f"OpenRouter returned invalid JSON: {raw[:200]}") from exc

        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if isinstance(usage, dict):
            try:
                prompt_tokens = int(
                    usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokens", 0)))
                )
            except (TypeError, ValueError):
                prompt_tokens = 0
            try:
                completion_tokens = int(
                    usage.get(
                        "completion_tokens",
                        usage.get("output_tokens", usage.get("completionTokens", 0)),
                    )
                )
            except (TypeError, ValueError):
                completion_tokens = 0
            try:
                total_tokens = int(usage.get("total_tokens", usage.get("totalTokens", 0)))
            except (TypeError, ValueError):
                total_tokens = 0
        if total_tokens <= 0:
            total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
        _record_token_usage(prompt_tokens, completion_tokens, total_tokens)

        content = self._extract_content(parsed)
        strict_nonempty = os.getenv("OPENROUTER_STRICT_NONEMPTY", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if strict_nonempty and not content.strip():
            _record_llm_error()
            preview = raw[:500].replace("\n", " ")
            raise RuntimeError(
                "OpenRouter returned an empty assistant content payload. "
                f"Model={self.model_id}. Raw preview={preview}"
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            content=content,
            backend="openrouter",
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_tokens=max(0, prompt_tokens),
            completion_tokens=max(0, completion_tokens),
            total_tokens=max(0, total_tokens),
        )
