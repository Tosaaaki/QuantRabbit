from __future__ import annotations

import asyncio
import json
import os
import random
import datetime
from dataclasses import dataclass
from typing import Callable, Awaitable

import httpx
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN: str = get_secret("oanda_token")
ACCOUNT: str = get_secret("oanda_account_id")
# Secret Manager または env.toml の `oanda_practice` を参照（未設定なら本番）
try:
    PRACTICE: bool = str(get_secret("oanda_practice")).lower() == "true"
except Exception:
    PRACTICE = False
MOCK_STREAM: bool = os.getenv("MOCK_TICK_STREAM", "0") == "1"

STREAM_HOST = (
    "stream-fxtrade.oanda.com" if not PRACTICE else "stream-fxpractice.oanda.com"
)
STREAM_URL = f"https://{STREAM_HOST}/v3/accounts/{ACCOUNT}/pricing/stream"


@dataclass
class Tick:
    instrument: str
    time: datetime.datetime
    bid: float
    ask: float
    liquidity: int


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
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET", STREAM_URL, headers=headers, params=params
                ) as r:
                    r.raise_for_status()
                    async for raw in r.aiter_lines():
                        if not raw:
                            continue
                        msg = json.loads(raw)
                        if msg.get("type") != "PRICE":
                            continue
                        tick = Tick(
                            instrument=msg["instrument"],
                            time=_parse_time(msg["time"]),
                            bid=float(msg["bids"][0]["price"]),
                            ask=float(msg["asks"][0]["price"]),
                            liquidity=int(msg["bids"][0]["liquidity"]),
                        )
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
        )
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
