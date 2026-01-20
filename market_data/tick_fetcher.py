from __future__ import annotations

import asyncio
import json
import os
import random
import datetime
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, Tuple

import httpx
from utils.secrets import get_secret
from market_data.replay_logger import log_tick

# ---------- 読み込み：env.toml ----------
TOKEN: str = get_secret("oanda_token")
ACCOUNT: str = get_secret("oanda_account_id")
# Secret Manager または env.toml の `oanda_practice` を参照（未設定なら本番）
try:
    PRACTICE: bool = str(get_secret("oanda_practice")).lower() == "true"
except Exception:
    PRACTICE = False
MOCK_STREAM: bool = os.getenv("MOCK_TICK_STREAM", "0") == "1"
_STREAM_READ_TIMEOUT = float(os.getenv("TICK_STREAM_READ_TIMEOUT_SEC", "12"))
_STREAM_CONNECT_TIMEOUT = float(os.getenv("TICK_STREAM_CONNECT_TIMEOUT_SEC", "5"))
_STREAM_WRITE_TIMEOUT = float(os.getenv("TICK_STREAM_WRITE_TIMEOUT_SEC", "5"))
_STREAM_POOL_TIMEOUT = float(os.getenv("TICK_STREAM_POOL_TIMEOUT_SEC", "5"))
_STREAM_MAX_IDLE_SEC = float(os.getenv("TICK_STREAM_MAX_IDLE_SEC", "10"))
_STREAM_MAX_IDLE_STRIKES = int(os.getenv("TICK_STREAM_MAX_IDLE_STRIKES", "2"))

STREAM_HOST = (
    "stream-fxtrade.oanda.com" if not PRACTICE else "stream-fxpractice.oanda.com"
)
STREAM_URL = f"https://{STREAM_HOST}/v3/accounts/{ACCOUNT}/pricing/stream"


DepthLevels = Tuple[Tuple[float, float], ...]


@dataclass
class Tick:
    instrument: str
    time: datetime.datetime
    bid: float
    ask: float
    liquidity: int
    bids: DepthLevels = tuple()
    asks: DepthLevels = tuple()


# ---------- メイン ----------


def _parse_time(value: str) -> datetime.datetime:
    """Convert OANDA timestamp (nanosecond precision) into datetime."""
    iso = value.replace("Z", "+00:00")
    if "." not in iso:
        return datetime.datetime.fromisoformat(iso)

    head, frac_and_tz = iso.split(".", 1)
    tz = "+00:00"
    if "+" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("+", 1)
        tz = "+" + tz_tail
    elif "-" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("-", 1)
        tz = "-" + tz_tail
    else:
        frac = frac_and_tz

    frac = (frac[:6]).ljust(6, "0")
    return datetime.datetime.fromisoformat(f"{head}.{frac}{tz}")


async def _connect(instrument: str, callback: Callable[[Tick], Awaitable[None]]):
    """
    内部：リコネクトループ
    """
    params = {"instruments": instrument}
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept-Datetime-Format": "RFC3339",
    }

    while True:
        try:
            timeout = httpx.Timeout(
                connect=_STREAM_CONNECT_TIMEOUT,
                read=_STREAM_READ_TIMEOUT,
                write=_STREAM_WRITE_TIMEOUT,
                pool=_STREAM_POOL_TIMEOUT,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "GET", STREAM_URL, headers=headers, params=params
                ) as r:
                    r.raise_for_status()
                    last_price_mono: Optional[float] = None
                    idle_strikes = 0
                    async for raw in r.aiter_lines():
                        if not raw:
                            continue
                        msg = json.loads(raw)
                        msg_type = msg.get("type")
                        if msg_type == "HEARTBEAT":
                            if (
                                last_price_mono is not None
                                and _STREAM_MAX_IDLE_SEC > 0
                                and _STREAM_MAX_IDLE_STRIKES > 0
                            ):
                                idle_for = time.monotonic() - last_price_mono
                                if idle_for >= _STREAM_MAX_IDLE_SEC:
                                    idle_strikes += 1
                                    if idle_strikes >= _STREAM_MAX_IDLE_STRIKES:
                                        raise RuntimeError(
                                            f"stream idle {idle_for:.1f}s without PRICE"
                                        )
                            continue
                        if msg_type != "PRICE":
                            continue
                        last_price_mono = time.monotonic()
                        idle_strikes = 0
                        raw_bids = msg.get("bids", [])
                        raw_asks = msg.get("asks", [])
                        bids = tuple(
                            (
                                float(entry.get("price")),
                                float(entry.get("liquidity", 0.0)),
                            )
                            for entry in raw_bids
                            if entry.get("price") is not None
                        )
                        asks = tuple(
                            (
                                float(entry.get("price")),
                                float(entry.get("liquidity", 0.0)),
                            )
                            for entry in raw_asks
                            if entry.get("price") is not None
                        )
                        top_bid = bids[0][0] if bids else float(raw_bids[0]["price"]) if raw_bids else 0.0
                        top_ask = asks[0][0] if asks else float(raw_asks[0]["price"]) if raw_asks else 0.0
                        top_liquidity = int(raw_bids[0].get("liquidity", 0)) if raw_bids else 0
                        tick = Tick(
                            instrument=msg["instrument"],
                            time=_parse_time(msg["time"]),
                            bid=top_bid,
                            ask=top_ask,
                            liquidity=top_liquidity,
                            bids=bids,
                            asks=asks,
                        )
                        try:
                            log_tick(tick)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[replay] failed to log tick: {exc}")
                        await callback(tick)
        except Exception as e:
            print("tick_fetcher reconnect:", e)
            await asyncio.sleep(3)  # バックオフして再接続


async def _mock_stream(instrument: str, callback: Callable[[Tick], Awaitable[None]]):
    """ネット接続不可時用の簡易ティック生成"""
    price = 150.0
    while True:
        move = random.uniform(-0.05, 0.05)
        bid = round(price + move, 3)
        ask = round(bid + 0.003, 3)
        price = (bid + ask) / 2
        tick = Tick(
            instrument=instrument,
            time=datetime.datetime.now(datetime.timezone.utc),
            bid=bid,
            ask=ask,
            liquidity=1000000,
            bids=((bid, 1000000.0),),
            asks=((ask, 1000000.0),),
        )
        try:
            log_tick(tick)
        except Exception as exc:  # noqa: BLE001
            print(f"[replay] failed to log tick: {exc}")
        await callback(tick)
        await asyncio.sleep(1)


async def run_price_stream(
    instrument: str, callback: Callable[[Tick], Awaitable[None]]
):
    """
    Public API
    ----------
    `instrument` : 例 "USD_JPY"
    `callback`   : async def tick_handler(Tick)
    """
    if MOCK_STREAM:
        await _mock_stream(instrument, callback)
    else:
        await _connect(instrument, callback)
