from __future__ import annotations

import asyncio
import json
import os
import random
import datetime
import time
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, Tuple

import httpx
from utils.secrets import get_secret
from utils.market_hours import is_market_open
from market_data.replay_logger import log_tick

# ---------- 読み込み：env.toml ----------
# NOTE: Don't resolve secrets at import time. Unit tests and fresh checkouts may not
# have env.toml/Secret Manager configured; defer until first network use.
_OANDA_TOKEN: str | None = None
_OANDA_ACCOUNT: str | None = None
_OANDA_PRACTICE: bool | None = None


def _load_oanda_config() -> tuple[str, str, bool]:
    global _OANDA_TOKEN, _OANDA_ACCOUNT, _OANDA_PRACTICE
    if _OANDA_TOKEN and _OANDA_ACCOUNT and _OANDA_PRACTICE is not None:
        return _OANDA_TOKEN, _OANDA_ACCOUNT, _OANDA_PRACTICE

    token = os.environ.get("OANDA_TOKEN") or None
    account = os.environ.get("OANDA_ACCOUNT") or None
    practice_raw = os.environ.get("OANDA_PRACTICE")
    practice: bool | None = None
    if practice_raw is not None:
        practice = str(practice_raw).strip().lower() == "true"

    if not token:
        token = str(get_secret("oanda_token"))
    if not account:
        account = str(get_secret("oanda_account_id"))
    if practice is None:
        try:
            practice = str(get_secret("oanda_practice")).lower() == "true"
        except Exception:
            practice = False

    _OANDA_TOKEN = token
    _OANDA_ACCOUNT = account
    _OANDA_PRACTICE = bool(practice)
    return token, account, bool(practice)
MOCK_STREAM: bool = os.getenv("MOCK_TICK_STREAM", "0") == "1"
_STREAM_READ_TIMEOUT = float(os.getenv("TICK_STREAM_READ_TIMEOUT_SEC", "25"))
_STREAM_CONNECT_TIMEOUT = float(os.getenv("TICK_STREAM_CONNECT_TIMEOUT_SEC", "5"))
_STREAM_WRITE_TIMEOUT = float(os.getenv("TICK_STREAM_WRITE_TIMEOUT_SEC", "5"))
_STREAM_POOL_TIMEOUT = float(os.getenv("TICK_STREAM_POOL_TIMEOUT_SEC", "5"))
_STREAM_MAX_IDLE_SEC = float(os.getenv("TICK_STREAM_MAX_IDLE_SEC", "20"))
_STREAM_MAX_IDLE_STRIKES = int(os.getenv("TICK_STREAM_MAX_IDLE_STRIKES", "3"))
_STREAM_IDLE_IGNORE_CLOSED = os.getenv("TICK_STREAM_IDLE_IGNORE_CLOSED", "1").strip().lower() not in {"0","false","no","off",""}


DepthLevels = Tuple[Tuple[float, float], ...]


LOG = logging.getLogger(__name__)

_STREAM_RESET_EVENT = asyncio.Event()
_STREAM_RESET_REASON: Optional[str] = None
_STREAM_RESET_TS: float = 0.0


def request_stream_reset(reason: str = "unspecified") -> None:
    """Request a soft reset of the pricing stream (reconnect without full process restart)."""
    global _STREAM_RESET_REASON, _STREAM_RESET_TS
    _STREAM_RESET_REASON = reason or "unspecified"
    _STREAM_RESET_TS = time.time()
    try:
        _STREAM_RESET_EVENT.set()
    except Exception:  # pragma: no cover - defensive
        LOG.warning("tick_fetcher reset request failed to signal event")


async def _await_stream_reset(response: httpx.Response) -> None:
    await _STREAM_RESET_EVENT.wait()
    _STREAM_RESET_EVENT.clear()
    reason = _STREAM_RESET_REASON or "unspecified"
    age = time.time() - _STREAM_RESET_TS if _STREAM_RESET_TS else 0.0
    if age > 0:
        LOG.warning("tick_fetcher reset requested: %s (age=%.1fs)", reason, age)
    else:
        LOG.warning("tick_fetcher reset requested: %s", reason)
    try:
        await response.aclose()
    except Exception:
        pass


def _log_stream_reset_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except Exception:
        return
    if exc:
        LOG.debug("tick_fetcher reset waiter ended with error: %s", exc)


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
    token, account, practice = _load_oanda_config()
    stream_host = "stream-fxtrade.oanda.com" if not practice else "stream-fxpractice.oanda.com"
    stream_url = f"https://{stream_host}/v3/accounts/{account}/pricing/stream"
    params = {"instruments": instrument}
    headers = {
        "Authorization": f"Bearer {token}",
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
                    "GET", stream_url, headers=headers, params=params
                ) as r:
                    r.raise_for_status()
                    last_price_mono: Optional[float] = None
                    idle_strikes = 0
                    reset_task = asyncio.create_task(_await_stream_reset(r))
                    reset_task.add_done_callback(_log_stream_reset_done)
                    try:
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
                                        if _STREAM_IDLE_IGNORE_CLOSED and not is_market_open():
                                            continue
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
                    finally:
                        reset_task.cancel()
                        try:
                            await reset_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
        except Exception as e:
            LOG.warning("tick_fetcher reconnect: %s", e)
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
