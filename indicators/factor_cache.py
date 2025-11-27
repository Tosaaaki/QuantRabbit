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
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Literal

import pandas as pd

from indicators.calc_core import IndicatorEngine

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

_LOCK = asyncio.Lock()


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
        _CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] persist failed: %s", exc)


def _restore_cache() -> None:
    """Load persisted factors (if any) back into memory."""
    if not _CACHE_PATH.exists():
        return
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] restore failed: %s", exc)
        return

    for tf, snapshot in data.items():
        if tf not in _CANDLES:
            continue
        try:
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
        except Exception as exc:  # noqa: BLE001
            logging.warning("[FACTOR_CACHE] failed to restore timeframe %s: %s", tf, exc)


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
        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[FACTOR_CACHE] warm recompute failed tf=%s: %s", tf, exc)


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

        _FACTORS[tf].clear()
        _FACTORS[tf].update(factors)
        _persist_cache()


def all_factors() -> Dict[TimeFrame, Dict[str, float]]:
    """全タイムフレームの指標dictを返す"""
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
