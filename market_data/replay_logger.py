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
    """Tick を JSONL に保存する。"""
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
    """ローソク足を JSONL に保存する。"""
    ts = candle.get("time")
    if isinstance(ts, str):
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        dt = ts or datetime.datetime.now(datetime.timezone.utc)
    day = _day_key(dt)
    path = (
        _BASE_DIR
        / instrument
        / f"{instrument}_{timeframe}_{day}.jsonl"
    )
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
