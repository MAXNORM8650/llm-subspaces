"""Small shared utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def threshold_token(fraction: float) -> str:
    return format(float(fraction) * 100.0, ".10g").replace(".", "_")


def validate_thresholds(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result or any(value <= 0.0 or value > 1.0 for value in result):
        raise ValueError("energy thresholds must be unique values in (0, 1]")
    return result
