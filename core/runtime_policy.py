from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def min_cuda_vram_gb() -> float:
    raw = os.getenv("MIN_CUDA_VRAM_GB", "8").strip()
    try:
        return float(raw)
    except ValueError:
        return 8.0


@lru_cache(maxsize=1)
def cuda_runtime() -> tuple[bool, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, 0.0
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return True, float(total_vram_gb)
    except Exception:
        return False, 0.0


@lru_cache(maxsize=1)
def has_cuda_with_min_vram() -> bool:
    cuda_available, total_vram_gb = cuda_runtime()
    return cuda_available and total_vram_gb >= min_cuda_vram_gb()


def auto_backend_switch_enabled() -> bool:
    return _env_flag("AUTO_BACKEND_SWITCH", default=True)


def resolve_detector_backend() -> Literal["local_hf", "openrouter"]:
    if auto_backend_switch_enabled():
        return "local_hf" if has_cuda_with_min_vram() else "openrouter"

    backend = os.getenv("DETECTOR_BACKEND", "local_hf").strip().lower()
    if backend in {"local_hf", "openrouter"}:
        return backend  # type: ignore[return-value]
    return "local_hf"


def resolve_persona_backend() -> Literal["openrouter_sim"]:
    if auto_backend_switch_enabled():
        return "openrouter_sim"

    backend = os.getenv("PERSONA_BACKEND", "openrouter_sim").strip().lower()
    if backend in {"", "openrouter_sim"}:
        return "openrouter_sim"

    raise ValueError(
        "Persona backend "
        f"'{backend}' disabled: deterministic simulator is the only supported persona backend in this build. "
        "Set PERSONA_BACKEND=openrouter_sim (or enable AUTO_BACKEND_SWITCH=1)."
    )
