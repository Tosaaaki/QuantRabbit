"""
market_data.candle_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tick を受け取り、任意のタイムフレームのローソク足を逐次生成する。
現在は **M1** 固定。必要に応じて dict 内に他 TF を追加可。
"""

from __future__ import annotations
import asyncio, datetime
from collections import defaultdict
from typing import Callable, Awaitable, Dict, List
from market_data.tick_fetcher import Tick

Candle = dict[str, float]  # open, high, low, close

class CandleAggregator:
    def __init__(self):
        self.current: Dict[str, Candle] = {}
        self.subscribers: List[Callable[[str, Candle], Awaitable[None]]] = []

    def subscribe(self, coro: Callable[[str, Candle], Awaitable[None]]):
        self.subscribers.append(coro)

    async def on_tick(self, tick: Tick):
        ts = tick.time.replace(tzinfo=datetime.timezone.utc)
        minute_key = ts.strftime("%Y-%m-%dT%H:%M")

        cndl = self.current.get(minute_key)
        price = (tick.bid + tick.ask) / 2
        if cndl is None:
            cndl = {"open": price, "high": price, "low": price, "close": price}
            self.current[minute_key] = cndl
        else:
            cndl["high"] = max(cndl["high"], price)
            cndl["low"] = min(cndl["low"], price)
            cndl["close"] = price

        #   次の分に切り替わった時点で確定ローソクを通知
        keys = list(self.current.keys())
        for k in keys:
            if k != minute_key:
                finalized = self.current.pop(k)
                for sub in self.subscribers:
                    await sub(k, finalized)


# ------ 便利ラッパ ------

async def start_candle_stream(instrument: str,
                              candle_handler: Callable[[str, Candle], Awaitable[None]]):
    """
    instrument: 例 "USD_JPY"
    candle_handler: async def candle_handler(key:str, cndl:Candle)
        key = "YYYY-MM-DDTHH:MM" (UTC)
    """
    agg = CandleAggregator()
    agg.subscribe(candle_handler)

    async def tick_cb(tick: Tick):
        await agg.on_tick(tick)

    from market_data.tick_fetcher import run_price_stream
    await run_price_stream(instrument, tick_cb)

# ---------- self test ----------
if __name__ == "__main__":
    import pprint, sys
    async def debug_candle(k, c):
        pprint.pprint((k, c))
    try:
        asyncio.run(start_candle_stream("USD_JPY", debug_candle))
    except KeyboardInterrupt:
        sys.exit(0)