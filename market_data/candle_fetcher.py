"""
market_data.candle_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tick を受け取り、任意のタイムフレームのローソク足を逐次生成する。
現在は **M1** 固定。必要に応じて dict 内に他 TF を追加可。
"""

from __future__ import annotations
import asyncio, datetime
from collections import defaultdict
from typing import Callable, Awaitable, Dict, List, Tuple
from market_data.tick_fetcher import Tick

Candle = dict[str, float]  # open, high, low, close
TimeFrame = Literal["M1", "H4"]

class CandleAggregator:
    def __init__(self, timeframes: List[TimeFrame]):
        self.timeframes = timeframes
        self.current_candles: Dict[TimeFrame, Candle] = {}
        self.last_keys: Dict[TimeFrame, str] = {}
        self.subscribers: Dict[TimeFrame, List[Callable[[Candle], Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, tf: TimeFrame, coro: Callable[[Candle], Awaitable[None]]):
        if tf in self.timeframes:
            self.subscribers[tf].append(coro)

    def _get_key(self, tf: TimeFrame, ts: datetime.datetime) -> str:
        if tf == "M1":
            return ts.strftime("%Y-%m-%dT%H:%M")
        if tf == "H4":
            # 4時間足の区切り (0, 4, 8, 12, 16, 20時 UTC)
            hour = (ts.hour // 4) * 4
            return ts.strftime(f"%Y-%m-%dT{hour:02d}:00")
        raise ValueError(f"Unsupported timeframe: {tf}")

    async def on_tick(self, tick: Tick):
        ts = tick.time.replace(tzinfo=datetime.timezone.utc)
        price = (tick.bid + tick.ask) / 2

        for tf in self.timeframes:
            key = self._get_key(tf, ts)
            
            # 新しいローソク足か判定
            if self.last_keys.get(tf) != key:
                # 古い足が確定
                if tf in self.current_candles:
                    finalized_candle = self.current_candles[tf]
                    for sub in self.subscribers[tf]:
                        await sub(finalized_candle)
                
                # 新しい足を開始
                self.current_candles[tf] = {"open": price, "high": price, "low": price, "close": price, "time": ts}
                self.last_keys[tf] = key
            else:
                # 現在の足を更新
                c = self.current_candles[tf]
                c["high"] = max(c["high"], price)
                c["low"] = min(c["low"], price)
                c["close"] = price
                c["time"] = ts


# ------ 便利ラッパ ------

async def start_candle_stream(instrument: str,
                              handlers: List[Tuple[TimeFrame, Callable[[Candle], Awaitable[None]]]]):
    """
    instrument: 例 "USD_JPY"
    handlers: [(TimeFrame, handler), ...]
    """
    timeframes = [tf for tf, _ in handlers]
    agg = CandleAggregator(timeframes)
    for tf, handler in handlers:
        agg.subscribe(tf, handler)

    async def tick_cb(tick: Tick):
        await agg.on_tick(tick)

    from market_data.tick_fetcher import run_price_stream
    await run_price_stream(instrument, tick_cb)

# ---------- self test ----------
if __name__ == "__main__":
    import pprint, sys
    async def debug_m1_candle(c):
        print("--- M1 Candle ---")
        pprint.pprint(c)

    async def debug_h4_candle(c):
        print("--- H4 Candle ---")
        pprint.pprint(c)

    try:
        handlers_to_run = [
            ("M1", debug_m1_candle),
            ("H4", debug_h4_candle),
        ]
        asyncio.run(start_candle_stream("USD_JPY", handlers_to_run))
    except KeyboardInterrupt:
        sys.exit(0)