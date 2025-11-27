"""Optional onepip maker worker (lazy import)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["onepip_maker_s1_worker"]


def __getattr__(name: str) -> Any:
    if name != "onepip_maker_s1_worker":
        raise AttributeError(
            f"module 'workers.onepip_maker_s1' has no attribute {name!r}"
        )
    try:
        module = import_module("workers.onepip_maker_s1.worker")
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "onepip_maker_s1 worker is unavailable in this build"
        ) from exc
    return getattr(module, name)
