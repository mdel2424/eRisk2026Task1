from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | int | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _safe_model_dump(value):
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return value
    return value


def _serialize(value):
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    dumped = _safe_model_dump(value)
    if dumped is not value:
        return _serialize(dumped)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _current_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
