"""
market_data.candle_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tick を受け取り、任意のタイムフレームのローソク足を逐次生成する。
現在は **M1** 固定。必要に応じて dict 内に他 TF を追加可。
"""

from __future__ import annotations

import asyncio
import datetime
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List, Literal, Tuple

import httpx
from utils.secrets import get_secret
from market_data.tick_fetcher import Tick, _parse_time
from market_data.replay_logger import log_candle
from market_data import spread_monitor, tick_window

Candle = dict[str, float]  # open, high, low, close
TimeFrame = Literal["M1", "M5", "H1", "H4", "D1"]


TOKEN = get_secret("oanda_token")
# Secret Manager または env.toml の `oanda_practice` を参照（未設定なら本番）
try:
    PRACT = str(get_secret("oanda_practice")).lower() == "true"
except Exception:
    PRACT = False
REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class CandleAggregator:
    def __init__(self, timeframes: List[TimeFrame], instrument: str):
        self.timeframes = timeframes
        self.instrument = instrument
        self.current_candles: Dict[TimeFrame, Candle] = {}
        self.last_keys: Dict[TimeFrame, str] = {}
        self.subscribers: Dict[TimeFrame, List[Callable[[Candle], Awaitable[None]]]] = (
            defaultdict(list)
        )

    def subscribe(self, tf: TimeFrame, coro: Callable[[Candle], Awaitable[None]]):
        if tf in self.timeframes:
            self.subscribers[tf].append(coro)

    def _get_key(self, tf: TimeFrame, ts: datetime.datetime) -> str:
        if tf == "M1":
            return ts.strftime("%Y-%m-%dT%H:%M")
        if tf == "M5":
            minute = (ts.minute // 5) * 5
            return ts.strftime(f"%Y-%m-%dT%H:{minute:02d}")
        if tf == "H1":
            return ts.strftime("%Y-%m-%dT%H:00")
        if tf == "H4":
            # 4時間足の区切り (0, 4, 8, 12, 16, 20時 UTC)
            hour = (ts.hour // 4) * 4
            return ts.strftime(f"%Y-%m-%dT{hour:02d}:00")
        if tf == "D1":
            return ts.strftime("%Y-%m-%dT00:00")
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
                    finalized_candle = dict(self.current_candles[tf])
                    try:
                        log_candle(self.instrument, tf, finalized_candle)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[replay] failed to log candle: {exc}")
                    for sub in self.subscribers[tf]:
                        await sub(finalized_candle)

                # 新しい足を開始
                self.current_candles[tf] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "time": ts,
                }
                self.last_keys[tf] = key
            else:
                # 現在の足を更新
                c = self.current_candles[tf]
                c["high"] = max(c["high"], price)
                c["low"] = min(c["low"], price)
                c["close"] = price
                c["time"] = ts


# ------ 便利ラッパ ------


async def start_candle_stream(
    instrument: str,
    handlers: List[Tuple[TimeFrame, Callable[[Candle], Awaitable[None]]]],
):
    """
    instrument: 例 "USD_JPY"
    handlers: [(TimeFrame, handler), ...]
    """
    timeframes = [tf for tf, _ in handlers]
    agg = CandleAggregator(timeframes, instrument)
    for tf, handler in handlers:
        agg.subscribe(tf, handler)

    async def tick_cb(tick: Tick):
        try:
            spread_monitor.update_from_tick(tick)
        except Exception as exc:  # noqa: BLE001
            print(f"[spread] failed to update monitor: {exc}")
        try:
            tick_window.record(tick)
        except Exception as exc:  # noqa: BLE001
            print(f"[tick_window] failed to record tick: {exc}")
        await agg.on_tick(tick)

    from market_data.tick_fetcher import run_price_stream

    await run_price_stream(instrument, tick_cb)


async def fetch_historical_candles(
    instrument: str, granularity: TimeFrame, count: int
) -> List[Candle]:
    """OANDA REST から過去ローソク足を取得する（失敗時は空配列）。"""
    url = f"{REST_HOST}/v3/instruments/{instrument}/candles"
    # Map internal TF to OANDA API granularity
    gran = granularity
    if granularity == "D1":
        gran = "D"
    params = {"granularity": gran, "count": count, "price": "M"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=HEADERS, params=params, timeout=7)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    out: List[Candle] = []
    for c in data.get("candles", []):
        # OANDA API returns nanoseconds, but fromisoformat only supports microseconds.
        # Truncate to microseconds.
        ts = _parse_time(c["time"])
        out.append(
            {
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "time": ts,
            }
        )
    out.sort(key=lambda x: x["time"])
    return out


async def initialize_history(instrument: str):
    """起動時に過去ローソクを取得し factor_cache を埋める"""
    from indicators.factor_cache import on_candle

    preload_counts = {"M1": 60, "M5": 120, "H1": 120, "H4": 30, "D1": 120}
    for tf in ("M1", "M5", "H1", "H4", "D1"):
        count = preload_counts.get(tf, 20)
        candles = await fetch_historical_candles(instrument, tf, count)
        for c in candles:
            await on_candle(tf, c)


# ---------- self test ----------
if __name__ == "__main__":
    import pprint
    import sys

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
