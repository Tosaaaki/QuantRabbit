from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from indicators import factor_cache


def _run(coro):
    return asyncio.run(coro)


def _seed_closed_m1(count: int = 40, start_price: float = 150.0) -> datetime:
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    async def _seed() -> None:
        for i in range(count):
            open_px = start_price + i * 0.02
            close_px = open_px + 0.01
            await factor_cache.on_candle(
                "M1",
                {
                    "open": open_px,
                    "high": close_px + 0.02,
                    "low": open_px - 0.02,
                    "close": close_px,
                    "time": base + timedelta(minutes=i),
                },
            )

    _run(_seed())
    return base


@pytest.fixture(autouse=True)
def _reset_factor_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(factor_cache, "_CACHE_PATH", tmp_path / "factor_cache_test.json", raising=False)
    monkeypatch.setattr(factor_cache, "_LIVE_UPDATE_ENABLED", True, raising=False)
    monkeypatch.setattr(factor_cache, "_LIVE_UPDATE_TFS", {"M1"}, raising=False)
    monkeypatch.setattr(factor_cache, "_LIVE_UPDATE_MIN_INTERVAL_SEC", 0.0, raising=False)
    monkeypatch.setattr(factor_cache, "_INCLUDE_LIVE_CANDLE_DEFAULT", True, raising=False)

    factor_cache._LAST_RESTORE_MTIME = None
    factor_cache._LAST_REGIME.clear()
    factor_cache._LIVE_CANDLES.clear()
    factor_cache._LIVE_LAST_COMPUTE_MONO.clear()
    factor_cache._FACTORS.clear()
    for dq in factor_cache._CANDLES.values():
        dq.clear()

    yield

    factor_cache._LAST_RESTORE_MTIME = None
    factor_cache._LAST_REGIME.clear()
    factor_cache._LIVE_CANDLES.clear()
    factor_cache._LIVE_LAST_COMPUTE_MONO.clear()
    factor_cache._FACTORS.clear()
    for dq in factor_cache._CANDLES.values():
        dq.clear()


def test_live_candle_updates_factors_and_snapshot():
    base = _seed_closed_m1()
    closed_factors = factor_cache.all_factors().get("M1") or {}
    closed_close = float(closed_factors["close"])

    live_ts = base + timedelta(minutes=40, seconds=15)
    live_close = closed_close + 0.85
    _run(
        factor_cache.on_candle_live(
            "M1",
            {
                "open": closed_close,
                "high": live_close + 0.03,
                "low": closed_close - 0.03,
                "close": live_close,
                "time": live_ts,
            },
        )
    )

    factors = factor_cache.all_factors().get("M1") or {}
    assert factors.get("live_candle") is True
    assert factors.get("timestamp") == live_ts.isoformat()
    assert float(factors.get("close")) == pytest.approx(live_close)
    assert factors.get("last_closed_timestamp") != factors.get("timestamp")

    candles = factor_cache.get_candles_snapshot("M1", limit=2)
    assert len(candles) == 2
    assert candles[-1]["timestamp"] == live_ts.isoformat()
    assert float(candles[-1]["close"]) == pytest.approx(live_close)


def test_snapshot_can_exclude_live_candle():
    base = _seed_closed_m1()
    live_ts = base + timedelta(minutes=40, seconds=20)

    _run(
        factor_cache.on_candle_live(
            "M1",
            {
                "open": 150.0,
                "high": 151.0,
                "low": 149.9,
                "close": 150.9,
                "time": live_ts,
            },
        )
    )

    with_live = factor_cache.get_candles_snapshot("M1", include_live=True)
    without_live = factor_cache.get_candles_snapshot("M1", include_live=False)
    assert len(with_live) == len(without_live) + 1
    assert with_live[-1]["timestamp"] == live_ts.isoformat()


def test_finalized_candle_clears_live_state():
    base = _seed_closed_m1()
    final_ts = base + timedelta(minutes=40, seconds=58)
    final_candle = {
        "open": 150.7,
        "high": 151.2,
        "low": 150.6,
        "close": 151.1,
        "time": final_ts,
    }

    _run(factor_cache.on_candle_live("M1", final_candle))
    _run(factor_cache.on_candle("M1", final_candle))

    factors = factor_cache.all_factors().get("M1") or {}
    assert factors.get("live_candle") is False
    assert factors.get("timestamp") == final_ts.isoformat()
    assert factors.get("last_closed_timestamp") == final_ts.isoformat()
    assert "M1" not in factor_cache._LIVE_CANDLES

    with_live = factor_cache.get_candles_snapshot("M1", include_live=True)
    without_live = factor_cache.get_candles_snapshot("M1", include_live=False)
    assert len(with_live) == len(without_live)
