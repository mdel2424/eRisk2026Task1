from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal


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


def resolve_detector_backend() -> Literal["openrouter", "ollama"]:
    backend = os.getenv("DETECTOR_BACKEND", "openrouter").strip().lower()
    if backend in {"openrouter", "ollama"}:
        return backend  # type: ignore[return-value]
    return "openrouter"
