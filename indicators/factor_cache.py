"""
indicators.factor_cache
~~~~~~~~~~~~~~~~~~~~~~~
・Candle を逐次受け取り DataFrame に蓄積
・最新指標を cache(dict) に保持し他モジュールへ提供
"""

from __future__ import annotations

import asyncio
import pandas as pd
from collections import deque
from typing import Dict

from indicators.calc_core import IndicatorEngine

_CANDLES_MAX = 2000           # 約 33 時間分 (M1)
_CANDLES: deque = deque(maxlen=_CANDLES_MAX)
_LATEST: Dict[str, float] = {}

_LOCK = asyncio.Lock()        # 取引ループと別スレッド安全用


async def on_candle(key: str, candle: Dict[str, float]):
    """
    market_data.candle_fetcher から呼ばれる想定
    key    : "YYYY-MM-DDTHH:MM" UTC
    candle : {"open":..,"high":..,"low":..,"close":..}
    """
    async with _LOCK:
        _CANDLES.append(
            {
                "timestamp": key,
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
            }
        )
        df = pd.DataFrame(_CANDLES)
        _LATEST.clear()
        _LATEST.update(IndicatorEngine.compute(df))


def get(name: str, default: float | None = None) -> float | None:
    """単一指標を取得"""
    return _LATEST.get(name, default)


def all_factors() -> Dict[str, float]:
    """最新指標 dict を shallow copy で返す"""
    return dict(_LATEST)


# ---------- self-test ----------
if __name__ == "__main__":
    # ダミー 20 本のローソクでテスト
    import random, datetime, asyncio

    async def main():
        base = 157.00
        for i in range(30):
            ts = (datetime.datetime.utcnow() + datetime.timedelta(minutes=i)).strftime(
                "%Y-%m-%dT%H:%M"
            )
            price = base + random.uniform(-0.1, 0.1)
            await on_candle(ts, {"open": price, "high": price + 0.03,
                                 "low": price - 0.03, "close": price})
        print(all_factors())

    asyncio.run(main())