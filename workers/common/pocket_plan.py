"""
Shared plan payloads between main planner and pocket executors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal


PocketType = Literal["macro", "scalp"]


@dataclass(slots=True)
class PocketPlan:
    generated_at: datetime
    pocket: PocketType
    focus_tag: str
    focus_pockets: List[str]
    range_active: bool
    range_soft_active: bool
    range_ctx: Dict[str, Any]
    event_soon: bool
    spread_gate_active: bool
    spread_gate_reason: str
    spread_log_context: str
    lot_allocation: float
    risk_override: float
    weight_macro: float
    scalp_share: float
    signals: List[Dict[str, Any]]
    perf_snapshot: Dict[str, Any]
    factors_m1: Dict[str, Any]
    factors_h4: Dict[str, Any]
    notes: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self, *, now: datetime | None = None) -> float:
        now_ts = now or datetime.utcnow()
        return max(0.0, (now_ts - self.generated_at).total_seconds())

    def is_stale(self, max_age_sec: float, *, now: datetime | None = None) -> bool:
        return self.age_seconds(now=now) > max_age_sec
