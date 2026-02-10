"""Execution-side helpers for entry quality gating.

These helpers are intentionally stdlib-only so they can run on the VM runtime
without adding heavy quant/ML dependencies.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Sequence

PIP_VALUE = 0.01  # USD/JPY: 1 pip = 0.01


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile in [0, 100]. Returns 0.0 for empty input."""
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, float(pct)))
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(float(v) for v in values)
    if pct >= 100.0:
        return float(sorted_vals[-1])
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return float(sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac)


@dataclass(frozen=True, slots=True)
class MicrostructureSnapshot:
    age_ms: int
    tick_count: int
    span_seconds: float
    tick_density: float
    spread_pips: float
    spread_p25: float
    spread_median: float
    spread_p95: float
    spread_max: float
    momentum_pips: float
    velocity_pips_per_sec: float
    source: str = "tick_window"


def compute_microstructure_snapshot(
    ticks: Sequence[dict],
    *,
    now_epoch: Optional[float] = None,
    pip_value: float = PIP_VALUE,
    source: str = "tick_window",
) -> Optional[MicrostructureSnapshot]:
    """Compute microstructure stats from a list of tick dicts.

    Expected keys per tick: epoch, bid, ask (mid optional).
    Returns None if not enough valid ticks are present.
    """
    if not ticks:
        return None
    rows: list[tuple[float, float, float, float]] = []
    for tick in ticks:
        if not isinstance(tick, dict):
            continue
        try:
            epoch = float(tick.get("epoch") or 0.0)
            bid = float(tick.get("bid") or 0.0)
            ask = float(tick.get("ask") or 0.0)
        except Exception:
            continue
        if epoch <= 0.0 or bid <= 0.0 or ask <= 0.0 or ask < bid:
            continue
        mid_raw = tick.get("mid")
        try:
            mid = float(mid_raw) if mid_raw is not None else (bid + ask) / 2.0
        except Exception:
            mid = (bid + ask) / 2.0
        rows.append((epoch, bid, ask, mid))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[0])
    first_epoch = rows[0][0]
    last_epoch = rows[-1][0]
    span = max(0.0, last_epoch - first_epoch)
    spreads: list[float] = [max(0.0, (ask - bid) / pip_value) for _, bid, ask, _ in rows]
    spread_pips = spreads[-1] if spreads else 0.0
    spread_p25 = percentile(spreads, 25.0)
    spread_med = percentile(spreads, 50.0)
    spread_p95 = percentile(spreads, 95.0)
    spread_max = max(spreads) if spreads else 0.0
    first_mid = rows[0][3]
    last_mid = rows[-1][3]
    momentum_pips = (last_mid - first_mid) / pip_value
    density = float(len(rows)) / span if span > 0 else 0.0
    velocity = abs(momentum_pips) / span if span > 0 else 0.0
    now_val = float(time.time() if now_epoch is None else now_epoch)
    age_ms = int(max(0.0, (now_val - last_epoch) * 1000.0))
    return MicrostructureSnapshot(
        age_ms=age_ms,
        tick_count=len(rows),
        span_seconds=span,
        tick_density=density,
        spread_pips=spread_pips,
        spread_p25=spread_p25,
        spread_median=spread_med,
        spread_p95=spread_p95,
        spread_max=spread_max,
        momentum_pips=momentum_pips,
        velocity_pips_per_sec=velocity,
        source=str(source or "tick_window"),
    )


_SPLIT_RE = re.compile(r"[-_/]+")

# Heuristic tags (kept aligned with execution.risk_guard).
_RANGE_HINTS = {
    "range",
    "revert",
    "reversion",
    "fade",
    "bbrsi",
    "bb_rsi",
    "vwapbound",
    "vwaprevert",
    "levelreactor",
    "magnet",
}
_TREND_HINTS = {
    "trend",
    "donchian",
    "break",
    "breakout",
    "rangebreak",
    "pullback",
    "momentum",
    "impulse",
    "trendma",
    "h1momentum",
    "m1scalper",
    "london",
    "session_open",
    "squeeze",
}


def infer_strategy_style(strategy_tag: Optional[str]) -> Optional[str]:
    """Return 'range', 'trend', or None (unknown/mixed)."""
    raw = str(strategy_tag or "").strip().lower()
    if not raw:
        return None
    base = raw.split("-", 1)[0]
    range_hit = any(token in base for token in _RANGE_HINTS)
    trend_hit = any(token in base for token in _TREND_HINTS)
    if range_hit and not trend_hit:
        return "range"
    if trend_hit and not range_hit:
        return "trend"
    # Fall back to token matching (for snake_case tags, etc.).
    tokens = [t for t in _SPLIT_RE.split(raw) if t]
    if tokens:
        range_hits = sum(1 for t in tokens if t in _RANGE_HINTS)
        trend_hits = sum(1 for t in tokens if t in _TREND_HINTS)
        if range_hits > trend_hits:
            return "range"
        if trend_hits > range_hits:
            return "trend"
    return None


def infer_regime_group(regime_label: Optional[str]) -> Optional[str]:
    """Normalize a regime label into a coarse bucket."""
    raw = str(regime_label or "").strip().lower()
    if not raw:
        return None
    if raw.startswith("range"):
        return "range"
    if raw.startswith("trend"):
        return "trend"
    if raw.startswith("breakout"):
        return "breakout"
    if raw.startswith("event"):
        return "event"
    if raw.startswith("mixed"):
        return "mixed"
    return None

