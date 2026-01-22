from __future__ import annotations

import atexit
import datetime
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Mapping

_BASE_DIR = Path("logs/replay")
_LOCK = threading.Lock()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


_ASYNC_ENABLED = os.getenv("REPLAY_LOG_ASYNC", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_QUEUE_MAX = max(1, _env_int("REPLAY_LOG_QUEUE_MAX", 8000))
_WRITE_QUEUE: "queue.Queue[tuple[Path, Mapping[str, Any]]]" = queue.Queue(
    maxsize=_QUEUE_MAX
)
_WRITER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def _writer_loop() -> None:
    while True:
        try:
            item = _WRITE_QUEUE.get(timeout=0.5)
        except queue.Empty:
            if _STOP_EVENT.is_set():
                return
            continue
        try:
            path, payload = item
            _write_jsonl(path, payload)
        except Exception:
            pass
        finally:
            _WRITE_QUEUE.task_done()


def _start_writer() -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD and _WRITER_THREAD.is_alive():
        return
    _WRITER_THREAD = threading.Thread(
        target=_writer_loop,
        name="replay-log-writer",
        daemon=True,
    )
    _WRITER_THREAD.start()


def _enqueue_write(path: Path, payload: Mapping[str, Any]) -> None:
    if not _ASYNC_ENABLED:
        _write_jsonl(path, payload)
        return
    _start_writer()
    try:
        _WRITE_QUEUE.put_nowait((path, payload))
    except queue.Full:
        _write_jsonl(path, payload)


def _shutdown_writer() -> None:
    if not _ASYNC_ENABLED:
        return
    _STOP_EVENT.set()
    if _WRITER_THREAD and _WRITER_THREAD.is_alive():
        _WRITER_THREAD.join(timeout=2.0)
    while True:
        try:
            path, payload = _WRITE_QUEUE.get_nowait()
        except queue.Empty:
            break
        _write_jsonl(path, payload)
        _WRITE_QUEUE.task_done()


atexit.register(_shutdown_writer)


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
        "bids": list(getattr(tick, "bids", ())[:5]) if getattr(tick, "bids", ()) else None,
        "asks": list(getattr(tick, "asks", ())[:5]) if getattr(tick, "asks", ()) else None,
    }
    _enqueue_write(path, payload)


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
    _enqueue_write(path, payload)
