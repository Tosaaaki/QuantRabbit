from __future__ import annotations
import asyncio
import json
import ssl
import datetime
from dataclasses import dataclass
from typing import Callable, Awaitable
import websockets
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
TOKEN: str = get_secret("oanda_token")
ACCOUNT: str = get_secret("oanda_account_id")
PRACTICE: bool = False  # 本番口座なので False に設定。必要に応じて True に変更

STREAM_HOST = (
    "stream-fxtrade.oanda.com" if not PRACTICE else "stream-fxpractice.oanda.com"
)
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
            async with websockets.connect(
                uri, ssl=ssl_ctx, extra_headers=headers, ping_interval=20
            ) as ws:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg["type"] != "PRICE":  # HEARTBEAT などは無視
                        continue
                    tick = Tick(
                        instrument=msg["instrument"],
                        time=datetime.datetime.fromisoformat(
                            msg["time"].replace("Z", "+00:00")
                        ),
                        bid=float(msg["bids"][0]["price"]),
                        ask=float(msg["asks"][0]["price"]),
                        liquidity=int(msg["bids"][0]["liquidity"]),
                    )
                    await callback(tick)
        except Exception as e:
            print("tick_fetcher reconnect:", e)
            await asyncio.sleep(3)  # バックオフして再接続


async def run_price_stream(
    instrument: str, callback: Callable[[Tick], Awaitable[None]]
):
    """
    Public API
    ----------
    `instrument` : 例 "USD_JPY"
    `callback`   : async def tick_handler(Tick)
    """
    await _connect(instrument, callback)
