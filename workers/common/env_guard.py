"""Shared environment guard utilities for high-frequency scalp workers."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from market_data import spread_monitor, tick_window

PIP_VALUE = 0.01  # USD/JPY 1 pip


def _mid_from_tick(tick: dict, fallback: float) -> float:
    mid_val = tick.get("mid")
    if mid_val is not None:
        try:
            return float(mid_val)
        except (TypeError, ValueError):
            pass
    bid = tick.get("bid")
    ask = tick.get("ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            return fallback
    return fallback


def mean_reversion_allowed(
    *,
    spread_p50_limit: float,
    return_pips_limit: float,
    return_window_sec: float,
    instant_move_limit: float,
    tick_gap_ms_limit: float,
    tick_gap_move_pips: float,
    ticks: Optional[Iterable[dict]] = None,
) -> Tuple[bool, str]:
    """
    Evaluate whether mean-reversion style entries should be allowed.

    Args:
        spread_p50_limit: Maximum acceptable baseline median spread (pips). 0 disables check.
        return_pips_limit: Maximum acceptable absolute price move (pips) across the window. 0 disables check.
        return_window_sec: Window in seconds to measure the price move. Ignored when <=0.
        instant_move_limit: Abort if the latest tick-to-tick move exceeds this (pips). 0 disables check.
        tick_gap_ms_limit: Abort if a tick gap above this (ms) coincides with a large move. 0 disables check.
        tick_gap_move_pips: Required move (pips) during a tick gap to trigger the guard.
        ticks: Optional iterable of recent ticks (dicts with epoch/bid/ask/mid). When omitted, pulls from tick_window.

    Returns:
        Tuple[bool, str]: (allowed flag, blocking reason)
    """
    state = spread_monitor.get_state()
    if spread_p50_limit > 0.0 and state:
        median = state.get("baseline_p50_pips")
        if median is None:
            median = state.get("avg_pips")
        if median is not None and median > spread_p50_limit:
            return False, f"spread_p50={median:.2f}p>limit({spread_p50_limit:.2f}p)"

    if ticks is None:
        ticks = tick_window.recent_ticks(
            seconds=max(return_window_sec, 5.0), limit=int(max(return_window_sec, 5.0) * 12)
        )
    tick_list = list(ticks)
    if return_pips_limit > 0.0 and return_window_sec > 0.0 and tick_list:
        window_ticks = [
            t for t in tick_list if t.get("epoch") and t.get("epoch") >= tick_list[-1].get("epoch", 0.0) - return_window_sec
        ]
        mids = [
            _mid_from_tick(t, float(tick_list[-1].get("mid") or 0.0))
            for t in window_ticks
        ]
        if len(mids) >= 2:
            move_pips = abs(mids[-1] - mids[0]) / PIP_VALUE
            if move_pips > return_pips_limit:
                return False, f"return={move_pips:.2f}p>limit({return_pips_limit:.2f}p)"

    if instant_move_limit > 0.0 and len(tick_list) >= 2:
        latest_mid = _mid_from_tick(tick_list[-1], float(tick_list[-1].get("mid") or 0.0))
        prev_mid = _mid_from_tick(tick_list[-2], latest_mid)
        instant_move = abs(latest_mid - prev_mid) / PIP_VALUE
        if instant_move > instant_move_limit:
            return False, f"instant_move={instant_move:.2f}p>limit({instant_move_limit:.2f}p)"

    if (
        tick_gap_ms_limit > 0.0
        and tick_gap_move_pips > 0.0
        and len(tick_list) >= 2
    ):
        for prev, curr in zip(tick_list[-10:-1], tick_list[-9:]):
            try:
                prev_epoch = float(prev.get("epoch"))
                curr_epoch = float(curr.get("epoch"))
            except (TypeError, ValueError):
                continue
            if prev_epoch <= 0.0 or curr_epoch <= 0.0:
                continue
            gap_ms = abs(curr_epoch - prev_epoch) * 1000.0
            if gap_ms < tick_gap_ms_limit:
                continue
            prev_mid = _mid_from_tick(prev, float(prev.get("mid") or 0.0))
            curr_mid = _mid_from_tick(curr, prev_mid)
            move_pips = abs(curr_mid - prev_mid) / PIP_VALUE
            if move_pips >= tick_gap_move_pips:
                return (
                    False,
                    f"tick_gap={gap_ms:.0f}ms move={move_pips:.2f}p>limit({tick_gap_move_pips:.2f}p)",
                )

    return True, ""
