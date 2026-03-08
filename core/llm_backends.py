from __future__ import annotations

import json
import os
import random
import socket
import time
from functools import lru_cache
from typing import List, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from core.llm_types import LLMResponse
from core.llm_usage import _record_llm_error, _record_token_usage, _reserve_llm_call
from core.runtime_policy import cuda_runtime, min_cuda_vram_gb


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = (base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4].rstrip("/")
    return normalized or "http://127.0.0.1:11434"


def _load_json_response(req: urllib_request.Request | str, *, timeout_sec: int) -> dict:
    with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Received invalid JSON payload: {raw[:200]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Received non-object JSON payload")
    return parsed


def list_ollama_models(base_url: str, *, timeout_sec: int) -> list[str]:
    normalized_base_url = normalize_ollama_base_url(base_url)
    req = urllib_request.Request(
        url=f"{normalized_base_url}/api/tags",
        method="GET",
    )
    try:
        parsed = _load_json_response(req, timeout_sec=timeout_sec)
    except urllib_error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    names: list[str] = []
    models = parsed.get("models", [])
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("name", "model"):
                raw_name = str(model.get(key, "") or "").strip()
                if raw_name and raw_name not in names:
                    names.append(raw_name)
    return names


def resolve_ollama_think_value() -> bool:
    raw = os.getenv("OLLAMA_THINK_MODE", "auto").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    cuda_available, _ = cuda_runtime()
    return bool(cuda_available)


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


class OllamaChatLLM:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: int = 120,
    ) -> None:
        if not model_id:
            raise ValueError("OLLAMA_DETECTOR_MODEL is required for ollama backend")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = normalize_ollama_base_url(base_url)
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

    def invoke(self, messages: Sequence[dict | tuple[str, str]]) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        if not normalized:
            return LLMResponse(content="", backend="ollama", model_id=self.model_id, latency_ms=0.0)

        _reserve_llm_call()
        started = time.perf_counter()

        payload = {
            "model": self.model_id,
            "messages": normalized,
            "stream": False,
            "think": resolve_ollama_think_value(),
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": max(0.0, self.temperature),
                "top_p": self.top_p,
            },
        }
        req = urllib_request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            parsed = _load_json_response(req, timeout_sec=self.timeout_sec)
        except urllib_error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            _record_llm_error()
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            _record_llm_error()
            raise RuntimeError(
                f"Ollama request failed for model '{self.model_id}' at {self.base_url}: {exc}"
            ) from exc

        prompt_tokens = 0
        completion_tokens = 0
        try:
            prompt_tokens = int(parsed.get("prompt_eval_count", 0) or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        try:
            completion_tokens = int(parsed.get("eval_count", 0) or 0)
        except (TypeError, ValueError):
            completion_tokens = 0
        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
        _record_token_usage(prompt_tokens, completion_tokens, total_tokens)

        message = parsed.get("message", {})
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content", "") or "").strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            content=content,
            backend="ollama",
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_tokens=max(0, prompt_tokens),
            completion_tokens=max(0, completion_tokens),
            total_tokens=max(0, total_tokens),
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

    # Keys that must never be treated as assistant content text.
    _SKIP_KEYS = frozenset({
        "role", "refusal", "tool_calls", "function_call", "name",
        "reasoning", "reasoning_content", "reasoning_tokens",
        "annotations", "audio", "logprobs",
    })

    @staticmethod
    def _collect_text(value: object, *, _depth: int = 0) -> List[str]:
        """Recursively extract human-readable text fragments from a value.

        Skips keys known to carry reasoning traces, role labels, or
        tool-call metadata so that only the actual assistant *content*
        is returned.
        """
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
                chunks.extend(OpenRouterChatLLM._collect_text(item, _depth=_depth + 1))
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
                if key in value and key not in OpenRouterChatLLM._SKIP_KEYS:
                    chunks.extend(OpenRouterChatLLM._collect_text(value.get(key), _depth=_depth + 1))
            # Generic fallback: walk remaining keys, skipping non-content fields.
            for key, nested in value.items():
                if key in preferred_keys or key in OpenRouterChatLLM._SKIP_KEYS:
                    continue
                chunks.extend(OpenRouterChatLLM._collect_text(nested, _depth=_depth + 1))
            return chunks
        return chunks

    @staticmethod
    def _extract_content(payload: dict) -> str:
        """Extract the assistant's content text from an OpenRouter/OpenAI response.

        For chat completions the canonical path is
        ``choices[0].message.content``.  We try that first so we never
        accidentally include reasoning traces, role strings, or other
        metadata that live as sibling keys on the ``message`` dict.
        """
        if not isinstance(payload, dict):
            return ""

        # ── Fast path: choices[0].message.content ──────────────────────
        choices = payload.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    # Direct content extraction — avoids walking into
                    # reasoning / role / tool_calls siblings.
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    # Some providers nest content in a list of parts.
                    if isinstance(content, list):
                        parts = OpenRouterChatLLM._collect_text(content)
                        if parts:
                            return " ".join(parts).strip()

                # Legacy completion-style text fallback
                if "text" in first:
                    text_val = first.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        return text_val.strip()

                # Delta fallback (some providers in non-stream mode)
                if "delta" in first:
                    delta = first.get("delta")
                    if isinstance(delta, dict):
                        dc = delta.get("content")
                        if isinstance(dc, str) and dc.strip():
                            return dc.strip()

        # ── Responses-style / provider-specific fallbacks ──────────────
        for key in ("output_text", "output", "response", "data", "message"):
            fallback = payload.get(key)
            if fallback is not None:
                parts = OpenRouterChatLLM._collect_text(fallback)
                if parts:
                    text = " ".join(parts).strip()
                    if text:
                        return text
        return ""

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return True
        if isinstance(exc, urllib_error.URLError):
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return True
            reason_text = str(reason or "").lower()
            if "timed out" in reason_text or "temporary" in reason_text or "unreachable" in reason_text:
                return True
        if isinstance(exc, OSError):
            msg = str(exc).lower()
            if "timed out" in msg or "temporary failure" in msg or "network is unreachable" in msg:
                return True
        return False

    @staticmethod
    def _sleep_before_retry(attempt_index: int, *, base_ms: int, jitter_ms: int) -> None:
        # attempt_index is 1-based for retries (1, 2, 3...)
        delay_ms = max(0, base_ms) * (2 ** max(0, attempt_index - 1))
        if jitter_ms > 0:
            delay_ms += random.randint(0, jitter_ms)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    def invoke(self, messages: Sequence[dict | tuple[str, str]]) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        if not normalized:
            return LLMResponse(content="", backend="openrouter", model_id=self.model_id, latency_ms=0.0)

        _reserve_llm_call()
        started = time.perf_counter()

        payload: dict = {
            "model": self.model_id,
            "messages": normalized,
            "temperature": max(0.0, self.temperature),
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }

        # Reasoning-model budget control — cap reasoning effort so the
        # model reserves enough of max_tokens for actual content.
        reasoning_effort = os.getenv("OPENROUTER_REASONING_EFFORT", "").strip().lower()
        if reasoning_effort in {"low", "medium", "high"}:
            payload["reasoning"] = {"effort": reasoning_effort}

        # Provider routing — read from env so callers can pin fast providers.
        provider_order_raw = os.getenv("OPENROUTER_PROVIDER_ORDER", "").strip()
        if provider_order_raw:
            provider_cfg: dict = {
                "order": [p.strip() for p in provider_order_raw.split(",") if p.strip()],
            }
            if os.getenv("OPENROUTER_REQUIRE_PROVIDER_ORDER", "0").strip() in {
                "1", "true", "yes", "y", "on",
            }:
                provider_cfg["require_parameters"] = True
            payload["provider"] = provider_cfg

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

        max_retries = max(0, int(os.getenv("OPENROUTER_MAX_RETRIES", "3")))
        retry_base_ms = max(0, int(os.getenv("OPENROUTER_RETRY_BASE_MS", "400")))
        retry_jitter_ms = max(0, int(os.getenv("OPENROUTER_RETRY_JITTER_MS", "250")))

        raw = ""
        for attempt in range(max_retries + 1):
            try:
                with urllib_request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib_error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = str(exc)
                can_retry = self._is_retryable_http_status(int(exc.code)) and attempt < max_retries
                if can_retry:
                    self._sleep_before_retry(
                        attempt + 1,
                        base_ms=retry_base_ms,
                        jitter_ms=retry_jitter_ms,
                    )
                    continue
                _record_llm_error()
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                can_retry = self._is_retryable_exception(exc) and attempt < max_retries
                if can_retry:
                    self._sleep_before_retry(
                        attempt + 1,
                        base_ms=retry_base_ms,
                        jitter_ms=retry_jitter_ms,
                    )
                    continue
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
