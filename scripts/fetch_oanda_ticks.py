#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OANDA Pricing Stream から USD/JPY のティックを取得して JSONL に保存します。

保存形式は `market_data/replay_logger.log_tick` と同一で、
`logs/replay/<instrument>/<instrument>_ticks_YYYYMMDD.jsonl` に追記します。

例）IAP/OS Login なしのローカル実行（環境変数/Secrets 必要）:
  python scripts/fetch_oanda_ticks.py --minutes 15 --instrument USD_JPY

注意:
  - OANDA は「過去のティック履歴 API」を提供していないため、本スクリプトは“これから先”のライブティックを保存します。
  - `config/env.toml` や OS 環境変数に `oanda_token`, `oanda_account_id`, `oanda_practice` を設定してください。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.secrets import get_secret  # noqa: E402
from market_data import replay_logger  # noqa: E402


@dataclass
class _Tick:
    instrument: str
    bid: float
    ask: float
    time: datetime


def _pricing_host(practice: bool) -> str:
    return "https://stream-fxpractice.oanda.com" if practice else "https://stream-fxtrade.oanda.com"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_price_obj(obj: dict, instrument: str) -> Optional[_Tick]:
    if not obj or obj.get("type") != "PRICE":
        return None
    try:
        bids = obj.get("bids") or []
        asks = obj.get("asks") or []
        bid = float(bids[0]["price"]) if bids else None
        ask = float(asks[0]["price"]) if asks else None
        ts_raw = obj.get("time")
        if ts_raw:
            ts_raw = ts_raw.replace("Z", "+00:00")
            if "." in ts_raw:
                head, frac = ts_raw.split(".", 1)
                if "+" in frac:
                    frac, tz_tail = frac.split("+", 1)
                    ts_raw = f"{head}.{frac[:6].ljust(6, '0')}+{tz_tail}"
                elif "-" in frac:
                    frac, tz_tail = frac.split("-", 1)
                    ts_raw = f"{head}.{frac[:6].ljust(6, '0')}-{tz_tail}"
                else:
                    ts_raw = f"{head}.{frac[:6].ljust(6, '0')}+00:00"
            elif "+" not in ts_raw and "-" not in ts_raw[-6:]:
                ts_raw = ts_raw + "+00:00"
            ts = datetime.fromisoformat(ts_raw)
        else:
            ts = _now_utc()
    except Exception:
        return None
    if bid is None or ask is None:
        return None
    return _Tick(instrument=instrument, bid=bid, ask=ask, time=ts)


async def stream_ticks(instrument: str, minutes: float) -> None:
    token = get_secret("oanda_token")
    account_id = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").strip().lower() in {"1", "true", "yes"}
    except Exception:
        practice = False

    base = _pricing_host(practice)
    url = f"{base}/v3/accounts/{account_id}/pricing/stream"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"instruments": instrument}

    deadline = _now_utc() + timedelta(minutes=max(0.1, minutes))
    timeout = httpx.Timeout(10.0, read=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url, headers=headers, params=params) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    if _now_utc() >= deadline:
                        break
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tick = _parse_price_obj(obj, instrument)
                if tick is None:
                    if _now_utc() >= deadline:
                        break
                    continue
                # 保存（replay_loggerと同一フォーマット）
                replay_logger.log_tick(tick)
                if _now_utc() >= deadline:
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch live ticks from OANDA pricing stream and store to logs/replay.")
    ap.add_argument("--instrument", default="USD_JPY", help="Instrument (default USD_JPY)")
    ap.add_argument("--minutes", type=float, default=10.0, help="Duration to stream (minutes)")
    args = ap.parse_args()
    try:
        asyncio.run(stream_ticks(args.instrument, args.minutes))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
