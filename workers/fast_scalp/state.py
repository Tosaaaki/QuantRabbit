"""
Shared state between the main 60s loop and the FastScalp worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Optional


@dataclass(frozen=True)
class FastScalpSnapshot:
    account_equity: float = 10000.0
    margin_available: float = 0.0
    margin_rate: float = 0.0
    weight_scalp: Optional[float] = None
    focus_tag: str = "micro"
    risk_pct_override: Optional[float] = None
    range_active: bool = False
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FastScalpState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = FastScalpSnapshot()

    def update_from_main(
        self,
        *,
        account_equity: float,
        margin_available: float,
        margin_rate: float,
        weight_scalp: Optional[float],
        focus_tag: str,
        risk_pct_override: Optional[float],
        range_active: bool,
    ) -> None:
        snap = FastScalpSnapshot(
            account_equity=max(1.0, float(account_equity or 0.0)),
            margin_available=max(0.0, float(margin_available or 0.0)),
            margin_rate=max(0.0, float(margin_rate or 0.0)),
            weight_scalp=weight_scalp,
            focus_tag=focus_tag,
            risk_pct_override=risk_pct_override,
            range_active=range_active,
        )
        with self._lock:
            self._snapshot = snap

    def snapshot(self) -> FastScalpSnapshot:
        with self._lock:
            return self._snapshot

