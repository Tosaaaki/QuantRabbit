"""Utilities for building QuantRabbit client order identifiers."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

__all__ = ["build_client_order_id"]


def build_client_order_id(focus_tag: Optional[str], strategy_tag: str) -> str:
    """Build a mostly-unique client order id per AGENT.me spec.

    Format: ``qr-{ts_ms}-{focus_tag}-{tag}-{hash9}``
    """

    ts_ms = int(time.time() * 1000)
    focus_part = (focus_tag or "hybrid")[:6]
    base_tag = (strategy_tag or "").split("-", 1)[0]
    clean_tag = "".join(ch for ch in base_tag if ch.isalnum())[:16] or "sig"
    seed = f"{ts_ms}-{focus_part}-{clean_tag}-{os.getpid()}-{time.monotonic_ns()}".encode()
    digest = hashlib.sha1(seed).hexdigest()[:9]
    return f"qr-{ts_ms}-{focus_part}-{clean_tag}-{digest}"
