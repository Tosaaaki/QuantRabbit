"""
market_data.tick_window
~~~~~~~~~~~~~~~~~~~~~~~
秒足レベルでのティック履歴を保持し、スキャル戦略が
直近 1〜2 分間のマイクロストラクチャを参照できるようにする。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

_MAX_SECONDS = 180  # 3 分あれば十分
_MAX_TICKS = 1800   # 10tick/sec を見込んだ上限


@dataclass(slots=True)
class _TickRow:
    epoch: float
    bid: float
    ask: float
    mid: float


_TICKS: Deque[_TickRow] = deque(maxlen=_MAX_TICKS)


def record(tick) -> None:  # type: ignore[no-untyped-def]
    """
    market_data.tick_fetcher.Tick を想定。
    """
    try:
        bid = float(tick.bid)
        ask = float(tick.ask)
        ts = float(tick.time.timestamp())
    except (AttributeError, TypeError, ValueError):
        return
    mid = round((bid + ask) / 2.0, 5)
    _TICKS.append(_TickRow(epoch=ts, bid=bid, ask=ask, mid=mid))


def _iter_recent(seconds: float) -> Iterable[_TickRow]:
    if not _TICKS:
        return ()
    cutoff = _TICKS[-1].epoch - max(0.0, seconds)
    return (row for row in reversed(_TICKS) if row.epoch >= cutoff)


def recent_ticks(seconds: float = 60.0, *, limit: int | None = None) -> List[Dict[str, float]]:
    """
    直近 seconds 秒以内のティックを新しい順で返す。
    """
    rows = []
    for idx, row in enumerate(_iter_recent(seconds)):
        rows.append({"epoch": row.epoch, "bid": row.bid, "ask": row.ask, "mid": row.mid})
        if limit is not None and idx + 1 >= limit:
            break
    rows.reverse()
    return rows


def summarize(seconds: float = 60.0) -> Dict[str, float]:
    """
    直近 seconds 秒の簡易サマリ。スキャルエントリーの
    指値位置を決める際の参考値を返す。
    """
    rows = list(_iter_recent(seconds))
    if not rows:
        return {}
    highs = max(row.bid for row in rows), max(row.ask for row in rows), max(row.mid for row in rows)
    lows = min(row.bid for row in rows), min(row.ask for row in rows), min(row.mid for row in rows)
    latest = rows[0]
    span = latest.epoch - rows[-1].epoch if len(rows) > 1 else 0.0
    return {
        "latest_bid": latest.bid,
        "latest_ask": latest.ask,
        "latest_mid": latest.mid,
        "high_bid": highs[0],
        "high_ask": highs[1],
        "high_mid": highs[2],
        "low_bid": lows[0],
        "low_ask": lows[1],
        "low_mid": lows[2],
        "span_seconds": span,
        "tick_count": float(len(rows)),
    }

