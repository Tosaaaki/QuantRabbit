"""
indicators.factor_cache
~~~~~~~~~~~~~~~~~~~~~~~
・Candle を逐次受け取り DataFrame に蓄積
・最新指標を cache(dict) に保持し他モジュールへ提供
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Literal

import pandas as pd

from indicators.calc_core import IndicatorEngine
from analysis.regime_classifier import classify

TimeFrame = Literal["M1", "M5", "H1", "H4", "D1"]

_CANDLES_MAX = {
    "M1": 2000,
    "M5": 1200,
    "H1": 1000,
    "H4": 500,
    "D1": 400,  # ~1.5y of daily bars for long-term factors
}
_CANDLES: Dict[TimeFrame, deque] = {
    "M1": deque(maxlen=_CANDLES_MAX["M1"]),
    "M5": deque(maxlen=_CANDLES_MAX["M5"]),
    "H1": deque(maxlen=_CANDLES_MAX["H1"]),
    "H4": deque(maxlen=_CANDLES_MAX["H4"]),
    "D1": deque(maxlen=_CANDLES_MAX["D1"]),
}
_FACTORS: Dict[TimeFrame, Dict[str, float]] = defaultdict(dict)
_CACHE_PATH = Path("logs/factor_cache.json")
_LAST_RESTORE_MTIME: float | None = None
_LAST_REGIME: Dict[TimeFrame, dict] = {}

_LOCK = asyncio.Lock()


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def _update_regime(tf: TimeFrame, factors: Dict[str, float]) -> None:
    if tf not in {"M1", "H1", "H4"}:
        return
    try:
        label = classify(factors, tf)
    except Exception:
        return
    if not label:
        return
    ts = factors.get("timestamp")
    factors["regime"] = label
    factors["regime_ts"] = ts
    _LAST_REGIME[tf] = {"regime": label, "timestamp": ts}


def _persist_cache() -> None:
    """Persist factor snapshot to disk so restart can reuse warm data."""
    try:
        payload: Dict[str, Dict[str, object]] = {}
        for tf, factors in _FACTORS.items():
            if not factors:
                continue
            snapshot: Dict[str, object] = {}
            for key, value in factors.items():
                if key == "candles":
                    continue
                snapshot[key] = _serialize(value)

            # Attach candles separately after serialization to avoid numpy leakage
            raw_candles = factors.get("candles")
            if isinstance(raw_candles, list):
                serial_candles = []
                limit = _CANDLES_MAX.get(tf, len(raw_candles))
                for cndl in raw_candles[-limit:]:
                    if not isinstance(cndl, dict):
                        continue
                    serial_candles.append(
                        {
                            "timestamp": cndl.get("timestamp"),
                            "open": _serialize(cndl.get("open")),
                            "high": _serialize(cndl.get("high")),
                            "low": _serialize(cndl.get("low")),
                            "close": _serialize(cndl.get("close")),
                        }
                    )
                snapshot["candles"] = serial_candles

            payload[tf] = snapshot
        if not payload:
            return
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(_CACHE_PATH, json.dumps(payload))
        try:
            global _LAST_RESTORE_MTIME
            _LAST_RESTORE_MTIME = _CACHE_PATH.stat().st_mtime
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] persist failed: %s", exc)


def _restore_cache() -> bool:
    """Load persisted factors (if any) back into memory."""
    if not _CACHE_PATH.exists():
        return False
    global _LAST_RESTORE_MTIME
    try:
        payload = _CACHE_PATH.read_text(encoding="utf-8")
        if not payload.strip():
            return False
        data = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] restore failed: %s", exc)
        return False
    try:
        _LAST_RESTORE_MTIME = _CACHE_PATH.stat().st_mtime  # type: ignore[assignment]
    except Exception:
        pass
    for tf, snapshot in data.items():
        if tf not in _CANDLES:
            continue
        try:
            label = snapshot.get("regime") or snapshot.get("regime_label")
            if label:
                _LAST_REGIME[tf] = {
                    "regime": str(label),
                    "timestamp": snapshot.get("regime_ts") or snapshot.get("timestamp"),
                }
            candles = snapshot.get("candles") or []
            dq = _CANDLES[tf]
            dq.clear()
            for cndl in candles[-_CANDLES_MAX[tf]:]:
                if not isinstance(cndl, dict):
                    continue
                dq.append(
                    {
                        "timestamp": cndl.get("timestamp"),
                        "open": cndl.get("open"),
                        "high": cndl.get("high"),
                        "low": cndl.get("low"),
                        "close": cndl.get("close"),
                    }
                )
            factors = dict(snapshot)
            if "candles" in factors:
                factors["candles"] = list(dq)
            if dq:
                last = dq[-1]
                factors.setdefault("close", last.get("close"))
                factors.setdefault("open", last.get("open"))
                factors.setdefault("high", last.get("high"))
                factors.setdefault("low", last.get("low"))
                factors.setdefault("timestamp", last.get("timestamp"))
            _FACTORS[tf].clear()
            _FACTORS[tf].update(factors)
            if "regime" not in _FACTORS[tf]:
                _update_regime(tf, _FACTORS[tf])
        except Exception as exc:  # noqa: BLE001
            logging.warning("[FACTOR_CACHE] failed to restore timeframe %s: %s", tf, exc)
    return True


_restore_cache()

for tf, dq in _CANDLES.items():
    if not dq:
        continue
    try:
        df = pd.DataFrame(list(dq))
        factors = IndicatorEngine.compute(df)
        factors["candles"] = list(dq)
        last = dq[-1]
        factors.update(
            {
                "close": last.get("close"),
                "open": last.get("open"),
                "high": last.get("high"),
                "low": last.get("low"),
                "timestamp": last.get("timestamp"),
            }
        )
        _update_regime(tf, factors)
        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] warm recompute failed tf=%s: %s", tf, exc)


def refresh_cache_from_disk() -> bool:
    """
    Reload cache from disk if the persisted file is newer than the last in-memory restore.
    Returns True when a reload was performed.
    """
    global _LAST_RESTORE_MTIME
    try:
        stat = _CACHE_PATH.stat()
    except FileNotFoundError:
        return False
    except Exception:
        return False
    if _LAST_RESTORE_MTIME is not None and stat.st_mtime <= _LAST_RESTORE_MTIME:
        return False
    if _restore_cache():
        _LAST_RESTORE_MTIME = stat.st_mtime
        return True
    return False


def _serialize(value: object) -> object:
    """Convert numpy/pandas scalars into plain Python types."""
    try:
        import numpy as np  # local import to avoid hard dependency if missing
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    return value


def get_candles_snapshot(tf: TimeFrame, *, limit: int | None = None) -> list[dict]:
    """Return a shallow copy of cached candles for a timeframe."""
    try:
        refresh_cache_from_disk()
    except Exception:  # defensive: 失敗しても古いキャッシュを返す
        pass
    dq = _CANDLES.get(tf)
    if not dq:
        return []
    candles = list(dq)
    if limit is not None and limit > 0:
        return candles[-limit:]
    return candles


def ensure_factors(tf: TimeFrame) -> Dict[str, float] | None:
    """Ensure factors exist for the timeframe (best-effort from cached candles)."""
    try:
        refresh_cache_from_disk()
    except Exception:
        pass
    existing = _FACTORS.get(tf)
    if existing:
        return existing
    dq = _CANDLES.get(tf)
    if not dq or len(dq) < 20:
        return None
    try:
        df = pd.DataFrame(list(dq))
        factors = IndicatorEngine.compute(df)
        factors["candles"] = list(dq)
        last = dq[-1]
        factors.update(
            {
                "close": last.get("close"),
                "open": last.get("open"),
                "high": last.get("high"),
                "low": last.get("low"),
                "timestamp": last.get("timestamp"),
            }
        )
        _update_regime(tf, factors)
        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
        _persist_cache()
        return _FACTORS[tf]
    except Exception:
        return None


def get_last_regime(tf: TimeFrame) -> tuple[str | None, object | None]:
    """Return last known regime label and timestamp for timeframe."""
    try:
        refresh_cache_from_disk()
    except Exception:
        pass
    factors = _FACTORS.get(tf) or {}
    label = factors.get("regime") or factors.get("regime_label")
    if label:
        return str(label), factors.get("regime_ts") or factors.get("timestamp")
    cached = _LAST_REGIME.get(tf) or {}
    return cached.get("regime"), cached.get("timestamp")


async def on_candle(tf: TimeFrame, candle: Dict[str, float]):
    """
    market_data.candle_fetcher から呼ばれる想定
    """
    async with _LOCK:
        q = _CANDLES[tf]
        q.append(
            {
                "timestamp": candle["time"].isoformat(),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
            }
        )

        if len(q) < 20:  # 計算に必要な最小限のデータを待つ
            return

        df = pd.DataFrame(q)
        factors = IndicatorEngine.compute(df)

        # Donchian戦略で必要になるため、生ローソクも格納
        factors["candles"] = list(q)
        last = q[-1]
        factors.update(
            {
                "close": last["close"],
                "open": last["open"],
                "high": last["high"],
                "low": last["low"],
                "timestamp": last["timestamp"],
            }
        )
        _update_regime(tf, factors)

        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
        _persist_cache()


def all_factors() -> Dict[TimeFrame, Dict[str, float]]:
    """全タイムフレームの指標dictを返す（ディスクキャッシュが新しければ自動リロード）。"""
    try:
        refresh_cache_from_disk()
    except Exception:  # defensive: 失敗しても古いキャッシュを返す
        pass
    return dict(_FACTORS)


# ---------- self-test ----------
if __name__ == "__main__":
    import asyncio
    import datetime
    import random

    async def main():
        base = 157.00
        now = datetime.datetime.utcnow()
        # M1
        for i in range(30):
            ts = now + datetime.timedelta(minutes=i)
            price = base + random.uniform(-0.1, 0.1)
            await on_candle(
                "M1",
                {
                    "open": price,
                    "high": price + 0.03,
                    "low": price - 0.03,
                    "close": price,
                    "time": ts,
                },
            )
        # H4
        for i in range(30):
            ts = now + datetime.timedelta(hours=i * 4)
            price = base + random.uniform(-1.0, 1.0)
            await on_candle(
                "H4",
                {
                    "open": price,
                    "high": price + 0.3,
                    "low": price - 0.3,
                    "close": price,
                    "time": ts,
                },
            )

        import pprint

        pprint.pprint(all_factors())

    asyncio.run(main())
