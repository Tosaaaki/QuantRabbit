"""
market_data.tick_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
OANDA プライス・ストリーム (v20) を非同期で購読し、
ティックをコールバックに渡すユーティリティ。

外部モジュール例
----------------
from market_data.tick_fetcher import run_price_stream

async def on_tick(tick):
    print(tick)

asyncio.run(run_price_stream("USD_JPY", on_tick))
"""

from __future__ import annotations
import asyncio, json, ssl, datetime
from dataclasses import dataclass
from typing import Callable, Awaitable
import websockets
import tomllib
import pathlib

# ---------- 読み込み：env.toml ----------
CONF = tomllib.loads(pathlib.Path("config/env.toml").read_text())
TOKEN: str = CONF["oanda"]["token"]
ACCOUNT: str = CONF["oanda"]["account"]
PRACTICE: bool = CONF["oanda"].get("practice", True)

STREAM_HOST = "stream-fxpractice.oanda.com" if PRACTICE else "stream-fxtrade.oanda.com"
STREAM_URL = f"wss://{STREAM_HOST}/v3/accounts/{ACCOUNT}/pricing/stream"

@dataclass
class Tick:
    instrument: str
    time: datetime.datetime
    bid: float
    ask: float
    liquidity: int

# ---------- メイン ----------

async def _connect(instrument: str, callback: Callable[[Tick], Awaitable[None]]):
    """
    内部：リコネクトループ
    """
    params = f"instruments={instrument}"
    uri = f"{STREAM_URL}?{params}"
    ssl_ctx = ssl.create_default_context()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    while True:
        try:
            async with websockets.connect(uri, ssl=ssl_ctx, extra_headers=headers, ping_interval=20) as ws:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg["type"] != "PRICE":      # HEARTBEAT などは無視
                        continue
                    tick = Tick(
                        instrument=msg["instrument"],
                        time=datetime.datetime.fromisoformat(msg["time"].replace("Z", "+00:00")),
                        bid=float(msg["bids"][0]["price"]),
                        ask=float(msg["asks"][0]["price"]),
                        liquidity=int(msg["bids"][0]["liquidity"]),
                    )
                    await callback(tick)
        except Exception as e:
            print("tick_fetcher reconnect:", e)
            await asyncio.sleep(3)   # バックオフして再接続


async def run_price_stream(instrument: str, callback: Callable[[Tick], Awaitable[None]]):
    """
    Public API
    ----------
    `instrument` : 例 "USD_JPY"
    `callback`   : async def tick_handler(Tick)
    """
    await _connect(instrument, callback)