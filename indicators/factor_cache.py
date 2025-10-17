"""
indicators.factor_cache
~~~~~~~~~~~~~~~~~~~~~~~
・Candle を逐次受け取り DataFrame に蓄積
・最新指標を cache(dict) に保持し他モジュールへ提供
"""

from __future__ import annotations
import asyncio
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, Literal

from indicators.calc_core import IndicatorEngine

TimeFrame = Literal["M1", "H4"]

_CANDLES_MAX = {"M1": 2000, "H4": 500}  # M1: ~33h, H4: ~83d
_CANDLES: Dict[TimeFrame, deque] = {
    "M1": deque(maxlen=_CANDLES_MAX["M1"]),
    "H4": deque(maxlen=_CANDLES_MAX["H4"]),
}
_FACTORS: Dict[TimeFrame, Dict[str, float]] = defaultdict(dict)
_LAST_TICK_METRICS: Dict[TimeFrame, Dict[str, float]] = defaultdict(dict)

_LOCK = asyncio.Lock()


def _compute_factors(rows: list[dict]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    price_df = df[["open", "high", "low", "close"]]
    factors = IndicatorEngine.compute(price_df)

    latest = rows[-1]
    factors["open"] = float(latest["open"])
    factors["high"] = float(latest["high"])
    factors["low"] = float(latest["low"])
    factors["close"] = float(latest["close"])
    factors["candles"] = rows
    return factors


async def on_candle(tf: TimeFrame, candle: Dict[str, float]):
    """確定したローソク足を登録し、終値ベースの指標を更新"""
    async with _LOCK:
        q = _CANDLES[tf]
        q.append(
            {
                "timestamp": candle["time"].isoformat(),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
            }
        )

        if len(q) < 20:  # 計算に必要な最小限のデータを待つ
            return

        factors = _compute_factors(list(q))
        factors["is_live"] = False

        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
        if _LAST_TICK_METRICS.get(tf):
            _FACTORS[tf].update(_LAST_TICK_METRICS[tf])


async def update_live(
    tf: TimeFrame,
    candle: Dict[str, float],
    tick_metrics: Dict[str, float] | None = None,
):
    """進行中のローソク（Tick）で指標を更新"""
    async with _LOCK:
        q = _CANDLES[tf]
        if len(q) < 19:
            return

        rows = list(q)
        rows.append(
            {
                "timestamp": candle["time"].isoformat(),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
            }
        )
        factors = _compute_factors(rows)
        factors["is_live"] = True

        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)

        if tick_metrics:
            _LAST_TICK_METRICS[tf] = dict(tick_metrics)
        if _LAST_TICK_METRICS.get(tf):
            _FACTORS[tf].update(_LAST_TICK_METRICS[tf])


def all_factors() -> Dict[TimeFrame, Dict[str, float]]:
    """全タイムフレームの指標dictを返す"""
    return dict(_FACTORS)


# ---------- self-test ----------
if __name__ == "__main__":
    import random
    import datetime
    import asyncio

    async def main():
        base = 157.00
        now = datetime.datetime.utcnow()
        # M1
        for i in range(30):
            ts = now + datetime.timedelta(minutes=i)
            price = base + random.uniform(-0.1, 0.1)
            await on_candle(
                "M1",
                {
                    "open": price,
                    "high": price + 0.03,
                    "low": price - 0.03,
                    "close": price,
                    "time": ts,
                },
            )
        # H4
        for i in range(30):
            ts = now + datetime.timedelta(hours=i * 4)
            price = base + random.uniform(-1.0, 1.0)
            await on_candle(
                "H4",
                {
                    "open": price,
                    "high": price + 0.3,
                    "low": price - 0.3,
                    "close": price,
                    "time": ts,
                },
            )

        import pprint

        pprint.pprint(all_factors())

    asyncio.run(main())
