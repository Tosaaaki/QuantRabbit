from __future__ import annotations

import time
from typing import List

import os

from market_data import spread_monitor, tick_window

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    if pct == 100.0:
        return sorted_vals[-1]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def spread_ok(*, max_pips: float | None = None, p25_max: float | None = None):
    state = spread_monitor.get_state()
    if state is None:
        ticks = tick_window.recent_ticks(seconds=8.0, limit=120)
        if not ticks:
            return False, None
        spreads: list[float] = []
        for tick in ticks:
            try:
                bid = float(tick.get("bid") or 0.0)
                ask = float(tick.get("ask") or 0.0)
            except Exception:
                continue
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue
            spreads.append((ask - bid) / 0.01)
        if not spreads:
            return False, None
        try:
            last_epoch = float(ticks[-1].get("epoch") or 0.0)
        except Exception:
            last_epoch = 0.0
        age_ms = int(max(0.0, (time.time() - (last_epoch or time.time())) * 1000.0))
        state = {
            "spread_pips": spreads[-1],
            "p25_pips": _percentile(spreads, 25.0),
            "median_pips": _percentile(spreads, 50.0),
            "p95_pips": _percentile(spreads, 95.0),
            "samples": len(spreads),
            "age_ms": age_ms,
            "stale": age_ms > 5000,
            "source": "tick_cache",
        }
    if state.get("stale"):
        return False, state
    spread = state.get("spread_pips", 999.0)
    try:
        spread = float(spread)
    except (TypeError, ValueError):
        spread = 999.0
    if max_pips is not None and spread > max_pips:
        return False, state
    if p25_max is not None:
        p25 = state.get("p25_pips")
        try:
            p25 = float(p25)
        except (TypeError, ValueError):
            return False, state
        if spread > max(p25_max, p25):
            return False, state
    return True, state


def test_spread_ok_falls_back_to_tick_cache(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 4.0)
    monkeypatch.setattr(spread_monitor, "get_state", lambda: None)
    monkeypatch.setattr(
        tick_window,
        "recent_ticks",
        lambda seconds=60.0, limit=None: [
            {"epoch": 2.0, "bid": 156.000, "ask": 156.010, "mid": 156.005},
            {"epoch": 3.0, "bid": 156.000, "ask": 156.008, "mid": 156.004},
            {"epoch": 4.0, "bid": 156.000, "ask": 156.006, "mid": 156.003},
        ],
    )

    ok, state = spread_ok(max_pips=1.2, p25_max=1.0)
    assert ok is True
    assert state is not None
    assert state.get("source") == "tick_cache"
