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

TimeFrame = Literal["M1", "H1", "H4"]

_CANDLES_MAX = {"M1": 2000, "H1": 1000, "H4": 500}  # M1: ~33h, H1: ~41d, H4: ~83d
_CANDLES: Dict[TimeFrame, deque] = {
    "M1": deque(maxlen=_CANDLES_MAX["M1"]),
    "H1": deque(maxlen=_CANDLES_MAX["H1"]),
    "H4": deque(maxlen=_CANDLES_MAX["H4"]),
}
_FACTORS: Dict[TimeFrame, Dict[str, float]] = defaultdict(dict)

_LOCK = asyncio.Lock()


async def on_candle(tf: TimeFrame, candle: Dict[str, float]):
    """
    market_data.candle_fetcher から呼ばれる想定
    """
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

        df = pd.DataFrame(q)
        factors = IndicatorEngine.compute(df)

        # Donchian戦略で必要になるため、生ローソクも格納
        factors["candles"] = list(q)
        last = q[-1]
        factors.update(
            {
                "close": last["close"],
                "open": last["open"],
                "high": last["high"],
                "low": last["low"],
                "timestamp": last["timestamp"],
            }
        )

        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)


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
