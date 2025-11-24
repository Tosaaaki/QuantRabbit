terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.24" }
  }
  backend "gcs" { bucket = "quantrabbit-tf-state" }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_compute_instance" "vm" {
  name         = "fx-trader-vm"
  machine_type = "e2-small"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # 外向け IP
    }
  }

        metadata_startup_script = <<-EOT
            #!/bin/bash
            # Updated 2025-07-22
            set -ex # Enable tracing and exit on error

            echo "Starting startup script..."
            echo "Sleeping for 30 seconds..."
            sleep 30

            

            echo "Installing system prerequisites..."
            sudo DEBIAN_FRONTEND=noninteractive apt-get update
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-venv python3-pip ca-certificates curl
            echo "Prerequisites installed."

            # echo "Installing Google Cloud Ops Agent..."
            # while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            #   echo "Waiting for other package manager to finish..."
            #   sleep 1
            # done
            # export DEBIAN_FRONTEND=noninteractive
            # sudo curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
            # sudo bash add-google-cloud-ops-agent-repo.sh --also-install
            # echo "Google Cloud Ops Agent installed."

            # echo "Configuring Google Cloud Ops Agent for Docker logs..."
            # sudo mkdir -p /etc/google-cloud-ops-agent/config.yaml.d
            # sudo tee /etc/google-cloud-ops-agent/config.yaml.d/docker-logs.yaml > /dev/null <<EOF
# logging:
#   receivers:
#     docker_receiver:
#       type: docker_json
#       include_all_docker_containers: true
#   service:
#     pipelines:
#       default_pipeline:
#         receivers: [docker_receiver]
# EOF
            # sudo systemctl restart google-cloud-ops-agent
            # echo "Google Cloud Ops Agent configured."

            echo "Cloning & preparing QuantRabbit repository..."
            sudo -u tossaki bash <<'EOS'
set -euo pipefail
cd /home/tossaki
if [ ! -d QuantRabbit ]; then
  git clone https://github.com/Tosaaaki/QuantRabbit.git
fi
cd /home/tossaki/QuantRabbit
git pull --ff-only || true
sed -i '/pandas[_-]ta/d' requirements.txt cloudrun/requirements.txt || true
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cat > analysis/gpt_decider.py <<'PY'
"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import asyncio
import json
from openai import AsyncOpenAI

from typing import Dict

from utils.cost_guard import add_tokens
from utils.secrets import get_secret
from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)

client = AsyncOpenAI(api_key=get_secret("openai_api_key"))


_SCHEMA = {
    "focus_tag": str,
    "weight_macro": float,
    "ranked_strategies": list,
}


class GPTTimeout(Exception): ...


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    if not add_tokens(0, MAX_TOKENS_MONTH):
        raise RuntimeError("GPT token limit exceeded")

    msgs = build_messages(payload)

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.2,
            max_tokens=120,
            timeout=7,
        )
    except Exception as e:
        raise GPTTimeout(str(e)) from e

    usage_in = resp.usage.prompt_tokens
    usage_out = resp.usage.completion_tokens
    add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)

    content = resp.choices[0].message.content.strip()
    content = content.lstrip("```json").rstrip("```").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {content}") from e

    for k, typ in _SCHEMA.items():
        if k not in data:
            raise ValueError(f"key {k} missing")
        if not isinstance(data[k], typ):
            raise ValueError(f"{k} type error")
    data["weight_macro"] = round(float(data["weight_macro"]), 2)
    return data


async def get_decision(payload: Dict) -> Dict:
    """GPT 呼び出し（フォールバックなし、リトライあり）"""
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return await asyncio.wait_for(call_openai(payload), timeout=9)
        except Exception as e:
            last_exc = e
            await asyncio.sleep(1.5)
    raise GPTTimeout(str(last_exc) if last_exc else "unknown error")


if __name__ == "__main__":
    import datetime
    import pprint

    dummy = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "reg_macro": "Trend",
        "reg_micro": "Range",
        "factors_m1": {"ma10": 157.2, "ma20": 157.1, "adx": 30},
        "factors_h4": {"ma10": 157.0, "ma20": 156.8, "adx": 25},
        "news_short": [],
        "news_long": [],
        "perf": {"macro_pf": 1.3, "micro_pf": 1.1},
    }

    res = asyncio.run(get_decision(dummy))
    pprint.pp(res)
PY

cat > indicators/calc_core.py <<'PY'
"""Core indicator calculations without pandas-ta."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class IndicatorEngine:
    """Compute technical indicators from OHLC DataFrame."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {
                "ma10": 0.0,
                "ma20": 0.0,
                "ema20": 0.0,
                "rsi": 0.0,
                "atr": 0.0,
                "adx": 0.0,
                "bbw": 0.0,
            }

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ma10 = close.rolling(window=10, min_periods=10).mean()
        ma20 = close.rolling(window=20, min_periods=20).mean()
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()

        rsi = _rsi(close, period=14)
        atr = _atr(high, low, close, period=14)
        adx = _adx(high, low, close, period=14)

        upper, middle, lower = _bollinger(close, period=20, std_mult=2.0)
        bbw_series = np.where(middle != 0, (upper - lower) / middle, 0.0)

        out: Dict[str, float] = {
            "ma10": float(ma10.iloc[-1]) if not ma10.empty else 0.0,
            "ma20": float(ma20.iloc[-1]) if not ma20.empty else 0.0,
            "ema20": float(ema20.iloc[-1]) if not ema20.empty else 0.0,
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 0.0,
            "atr": float(atr.iloc[-1]) if not atr.empty else 0.0,
            "adx": float(adx.iloc[-1]) if not adx.empty else 0.0,
            "bbw": float(bbw_series[-1]) if bbw_series.size else 0.0,
        }

        for k, v in out.items():
            if not np.isfinite(v):
                out[k] = 0.0
        return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def _dm(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    up = high.diff().clip(lower=0.0)
    down = -low.diff().clip(lower=0.0)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    return pd.Series(plus_dm, index=high.index), pd.Series(minus_dm, index=high.index)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    atr = _atr(high, low, close, period)
    plus_dm, minus_dm = _dm(high, low)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def _bollinger(close: pd.Series, period: int, std_mult: float):
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper.fillna(0.0), middle.fillna(0.0), lower.fillna(0.0)
PY

cat > market_data/tick_fetcher.py <<'PY'
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


def _parse_time(value: str) -> datetime.datetime:
    """Convert OANDA nanosecond timestamp to datetime (microsecond precision)."""
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
            await asyncio.sleep(3)


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
PY

cat > market_data/replay_logger.py <<'PY'
from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any, Mapping

_BASE_DIR = Path("logs/replay")
_LOCK = threading.Lock()


def _ensure_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_utc_iso(ts: datetime.datetime | str | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts.astimezone(datetime.timezone.utc).isoformat()


def _day_key(ts: datetime.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts.astimezone(datetime.timezone.utc).strftime("%Y%m%d")


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        fh_path = _ensure_path(path)
        with fh_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def log_tick(tick) -> None:
    ts = tick.time
    day = _day_key(ts)
    path = _BASE_DIR / tick.instrument / f"{tick.instrument}_ticks_{day}.jsonl"
    payload = {
        "ts": _to_utc_iso(ts),
        "instrument": tick.instrument,
        "bid": getattr(tick, "bid", None),
        "ask": getattr(tick, "ask", None),
        "mid": (getattr(tick, "bid", 0.0) + getattr(tick, "ask", 0.0)) / 2
        if getattr(tick, "bid", None) is not None and getattr(tick, "ask", None) is not None
        else None,
        "liquidity": getattr(tick, "liquidity", None),
    }
    _write_jsonl(path, payload)


def log_candle(instrument: str, timeframe: str, candle: Mapping[str, Any]) -> None:
    ts = candle.get("time")
    if isinstance(ts, str):
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        dt = ts or datetime.datetime.now(datetime.timezone.utc)
    day = _day_key(dt)
    path = _BASE_DIR / instrument / f"{instrument}_{timeframe}_{day}.jsonl"
    payload = {
        "ts": _to_utc_iso(dt),
        "timeframe": timeframe,
        "open": float(candle.get("open")) if candle.get("open") is not None else None,
        "high": float(candle.get("high")) if candle.get("high") is not None else None,
        "low": float(candle.get("low")) if candle.get("low") is not None else None,
        "close": float(candle.get("close")) if candle.get("close") is not None else None,
        "volume": candle.get("volume"),
    }
    _write_jsonl(path, payload)
PY

cat > market_data/candle_fetcher.py <<'PY'
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

Candle = dict[str, float]
TimeFrame = Literal["M1", "H4"]

TOKEN = get_secret("oanda_token")
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
        if tf == "H4":
            hour = (ts.hour // 4) * 4
            return ts.strftime(f"%Y-%m-%dT{hour:02d}:00")
        raise ValueError(f"Unsupported timeframe: {tf}")

    async def on_tick(self, tick: Tick):
        ts = tick.time.replace(tzinfo=datetime.timezone.utc)
        price = (tick.bid + tick.ask) / 2

        for tf in self.timeframes:
            key = self._get_key(tf, ts)

            if self.last_keys.get(tf) != key:
                if tf in self.current_candles:
                    finalized = dict(self.current_candles[tf])
                    try:
                        log_candle(self.instrument, tf, finalized)
                    except Exception as exc:
                        print(f"[replay] failed to log candle: {exc}")
                    for sub in self.subscribers[tf]:
                        await sub(finalized)

                self.current_candles[tf] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "time": ts,
                }
                self.last_keys[tf] = key
            else:
                candle = self.current_candles[tf]
                candle["high"] = max(candle["high"], price)
                candle["low"] = min(candle["low"], price)
                candle["close"] = price
                candle["time"] = ts


async def start_candle_stream(
    instrument: str,
    handlers: List[Tuple[TimeFrame, Callable[[Candle], Awaitable[None]]]],
):
    timeframes = [tf for tf, _ in handlers]
    agg = CandleAggregator(timeframes, instrument)
    for tf, handler in handlers:
        agg.subscribe(tf, handler)

    async def tick_cb(tick: Tick):
        await agg.on_tick(tick)

    from market_data.tick_fetcher import run_price_stream

    await run_price_stream(instrument, tick_cb)


async def fetch_historical_candles(
    instrument: str, granularity: TimeFrame, count: int
) -> List[Candle]:
    url = f"{REST_HOST}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=HEADERS, params=params, timeout=7)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    out: List[Candle] = []
    for c in data.get("candles", []):
        time_str = c["time"].replace("Z", "+00:00")
        if "." in time_str:
            main_part, frac_part = time_str.split(".")
            time_str = f"{main_part}.{frac_part[:6]}{frac_part[-6:]}"
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
    from indicators.factor_cache import on_candle

    for tf in ("M1", "H4"):
        candles = await fetch_historical_candles(instrument, tf, 20)
        for c in candles:
            await on_candle(tf, c)
PY
EOS
            echo "Repository prepared."

            echo "Adjusting permissions and runtime directories..."
            sudo chown -R tossaki:tossaki /home/tossaki/QuantRabbit
            sudo -u tossaki mkdir -p /home/tossaki/.cache

            echo "Skipping local config copies (using Secret Manager)."

            echo "Creating systemd service..."
            sudo tee /etc/systemd/system/quantrabbit.service >/dev/null <<'UNIT'
[Unit]
Description=QuantRabbit Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
User=tossaki
WorkingDirectory=/home/tossaki/QuantRabbit
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/home/tossaki
Environment=TUNER_ENABLE=1
Environment=TUNER_SHADOW_MODE=false
ExecStart=/home/tossaki/QuantRabbit/.venv/bin/python /home/tossaki/QuantRabbit/main.py
Restart=always
RestartSec=5
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
UNIT

            sudo systemctl daemon-reload
            sudo systemctl enable --now quantrabbit.service
            echo "quantrabbit.service started."
        EOT
  tags = ["fx-vm"]
  service_account {
    email  = "fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }
}

data "google_iam_policy" "sa" {
  binding {
    role = "roles/iam.serviceAccountUser"
    members = [
      "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com",
    ]
  }
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}
