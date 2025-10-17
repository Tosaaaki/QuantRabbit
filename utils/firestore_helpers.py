"""Utility helpers for Firestore queries with backwards-compatible filters."""
from __future__ import annotations

from typing import Any

try:  # Firestore >= 2.7
    from google.cloud.firestore_v1 import FieldFilter  # type: ignore
except Exception:  # pragma: no cover - library missing/older version
    FieldFilter = None  # type: ignore


def apply_filter(query: Any, field: str, op: str, value: Any):
    """Attach a where-filter, preferring FieldFilter when available."""
    if FieldFilter is not None:
        try:
            return query.where(filter=FieldFilter(field, op, value))
        except Exception:
            return query.where(field, op, value)
    return query.where(field, op, value)
