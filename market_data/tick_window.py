"""
market_data.tick_window
~~~~~~~~~~~~~~~~~~~~~~~
秒足レベルでのティック履歴を保持し、スキャル戦略が
直近 1〜2 分間のマイクロストラクチャを参照できるようにする。
"""

from __future__ import annotations

import time
from collections import deque
import os
import json
import threading
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import logging

# 保持ウィンドウを戦略要件（最大で ~6 分〜10 分）に合わせて拡張
_MAX_SECONDS = 600  # 10 分保持（S5×60本などの要件に対応）
_MAX_TICKS = 6000   # 10 tick/sec を想定した上限
_BASE_DIR = Path(__file__).resolve().parents[1]
_CACHE_PATH = (_BASE_DIR / "logs" / "tick_cache.json").resolve()
# ディスクキャッシュに含める最大件数（ワーカーが十分な窓幅を再構成できるよう拡張）
_CACHE_LIMIT = 3000  # およそ 5〜10 分分（tick/sec に依存）
_LOGGER = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_FLUSH_INTERVAL_SEC = _env_float("TICK_CACHE_FLUSH_INTERVAL_SEC", 1.0)
_MIN_RELOAD_INTERVAL_SEC = _env_float("TICK_CACHE_RELOAD_INTERVAL_SEC", 0.2)


@dataclass(slots=True)
class _TickRow:
    epoch: float
    bid: float
    ask: float
    mid: float


_TICKS: Deque[_TickRow] = deque(maxlen=_MAX_TICKS)
_last_flush_ts: float = 0.0
_last_persist_error_ts: float = 0.0
_cache_mtime: float = 0.0
_last_reload_ts: float = 0.0
_persist_lock = threading.Lock()
_persist_inflight = False
_persist_payload: list[dict[str, float]] | None = None


def _load_cache() -> None:
    if not _CACHE_PATH.exists():
        return
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, list):
        return
    rows: List[_TickRow] = []
    for entry in payload[-_MAX_TICKS:]:
        if not isinstance(entry, dict):
            continue
        try:
            epoch = float(entry["epoch"])
            bid = float(entry["bid"])
            ask = float(entry["ask"])
            mid = float(entry["mid"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(_TickRow(epoch=epoch, bid=bid, ask=ask, mid=mid))
    if rows:
        for row in rows[-_MAX_TICKS:]:
            _TICKS.append(row)
        try:
            stat = _CACHE_PATH.stat()
            globals()["_cache_mtime"] = float(stat.st_mtime)
        except Exception:
            globals()["_cache_mtime"] = 0.0
        _LOGGER.info("[TICK_CACHE] restored ticks=%d", len(rows))
    else:
        _LOGGER.info("[TICK_CACHE] no cached ticks found")


def _write_cache_payload(payload: list[dict[str, float]]) -> None:
    global _last_persist_error_ts, _cache_mtime
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_CACHE_PATH.parent,
            prefix=_CACHE_PATH.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(json.dumps(payload, separators=(",", ":")))
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(_CACHE_PATH)
        try:
            _cache_mtime = float(_CACHE_PATH.stat().st_mtime)
        except Exception:
            pass
        _LOGGER.debug("[TICK_CACHE] persisted=%d path=%s", len(payload), _CACHE_PATH)
    except Exception as exc:
        if "tmp_path" in locals() and isinstance(tmp_path, Path):
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if time.time() - _last_persist_error_ts >= 30.0:
            _LOGGER.warning("[TICK_CACHE] persist failed: %s", exc)
            _last_persist_error_ts = time.time()


def _persist_worker() -> None:
    global _persist_inflight, _persist_payload
    while True:
        with _persist_lock:
            payload = _persist_payload
            _persist_payload = None
            if payload is None:
                _persist_inflight = False
                return
        _write_cache_payload(payload)


def _persist_cache() -> None:
    global _last_flush_ts, _persist_inflight, _persist_payload
    now = time.time()
    if now - _last_flush_ts < _FLUSH_INTERVAL_SEC:
        return
    _last_flush_ts = now
    window = list(_TICKS)[-min(len(_TICKS), _CACHE_LIMIT) :]
    payload = [
        {"epoch": row.epoch, "bid": row.bid, "ask": row.ask, "mid": row.mid}
        for row in window
    ]
    with _persist_lock:
        _persist_payload = payload
        if _persist_inflight:
            return
        _persist_inflight = True
    thread = threading.Thread(target=_persist_worker, name="tick-cache-persist", daemon=True)
    thread.start()


_load_cache()


def _reload_cache_if_updated() -> None:
    """Reload on-disk cache if it changed since last load.

    This enables cross-process workers to receive fresh ticks that were
    persisted by the main process without maintaining their own stream.
    Rate-limited to avoid excessive filesystem I/O.
    """
    global _cache_mtime, _last_reload_ts
    now = time.time()
    if now - _last_reload_ts < _MIN_RELOAD_INTERVAL_SEC:
        return
    _last_reload_ts = now
    try:
        stat = _CACHE_PATH.stat()
    except FileNotFoundError:
        return
    except Exception:
        return
    mtime = float(stat.st_mtime)
    if mtime <= _cache_mtime:
        return
    # Cache updated by another process – reload a slim view.
    # Keep fresher in-memory ticks when the file update is older.
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            _cache_mtime = mtime
            return
        rows: list[_TickRow] = []
        file_latest_epoch = float("-inf")
        for entry in payload[-_CACHE_LIMIT:]:
            if not isinstance(entry, dict):
                continue
            try:
                epoch = float(entry["epoch"])
                bid = float(entry["bid"])
                ask = float(entry["ask"])
                mid = float(entry["mid"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(_TickRow(epoch=epoch, bid=bid, ask=ask, mid=mid))
            if epoch > file_latest_epoch:
                file_latest_epoch = epoch
        if not rows:
            _cache_mtime = mtime
            return
        if _TICKS:
            mem_latest_epoch = _TICKS[-1].epoch
            if file_latest_epoch <= mem_latest_epoch:
                # File cache is older than local stream/snapshot fallback.
                _cache_mtime = mtime
                return
            for row in sorted(rows, key=lambda item: item.epoch):
                if row.epoch > mem_latest_epoch:
                    _TICKS.append(row)
        else:
            for row in rows[-_MAX_TICKS:]:
                _TICKS.append(row)
        _cache_mtime = mtime
    except Exception:
        # Ignore reload failures; next iteration will retry
        return


def record(tick) -> None:  # type: ignore[no-untyped-def]
    """
    market_data.tick_fetcher.Tick を想定。
    """
    try:
        bid = float(tick.bid)
        ask = float(tick.ask)
        ts = float(tick.time.timestamp())
    except (AttributeError, TypeError, ValueError):
        return
    mid = round((bid + ask) / 2.0, 5)
    _TICKS.append(_TickRow(epoch=ts, bid=bid, ask=ask, mid=mid))
    _persist_cache()


def _iter_recent(seconds: float) -> Iterable[_TickRow]:
    # Ensure we see new ticks written by other processes.
    _reload_cache_if_updated()
    if not _TICKS:
        return ()
    cutoff = _TICKS[-1].epoch - max(0.0, seconds)
    return (row for row in reversed(_TICKS) if row.epoch >= cutoff)


def recent_ticks(seconds: float = 60.0, *, limit: int | None = None) -> List[Dict[str, float]]:
    """
    直近 seconds 秒以内のティックを新しい順で返す。
    """
    # Pull in any fresh cache before serving reads
    _reload_cache_if_updated()
    rows = []
    for idx, row in enumerate(_iter_recent(seconds)):
        rows.append({"epoch": row.epoch, "bid": row.bid, "ask": row.ask, "mid": row.mid})
        if limit is not None and idx + 1 >= limit:
            break
    rows.reverse()
    return rows


def summarize(seconds: float = 60.0) -> Dict[str, float]:
    """
    直近 seconds 秒の簡易サマリ。スキャルエントリーの
    指値位置を決める際の参考値を返す。
    """
    _reload_cache_if_updated()
    rows = list(_iter_recent(seconds))
    if not rows:
        return {}
    highs = max(row.bid for row in rows), max(row.ask for row in rows), max(row.mid for row in rows)
    lows = min(row.bid for row in rows), min(row.ask for row in rows), min(row.mid for row in rows)
    latest = rows[0]
    span = latest.epoch - rows[-1].epoch if len(rows) > 1 else 0.0
    return {
        "latest_bid": latest.bid,
        "latest_ask": latest.ask,
        "latest_mid": latest.mid,
        "high_bid": highs[0],
        "high_ask": highs[1],
        "high_mid": highs[2],
        "low_bid": lows[0],
        "low_ask": lows[1],
        "low_mid": lows[2],
        "span_seconds": span,
        "tick_count": float(len(rows)),
    }
