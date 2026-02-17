from __future__ import annotations

import asyncio

from workers.market_data_feed import worker as market_data_feed_worker


def test_build_handlers_passes_timeframe_to_factor_cache(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def _fake_on_candle(tf: str, candle: dict) -> None:
        calls.append((tf, candle))

    monkeypatch.setattr("indicators.factor_cache.on_candle", _fake_on_candle, raising=True)

    candle = {"open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "time": "2026-02-17T00:00:00+00:00"}
    handlers = market_data_feed_worker._build_handlers(["M1", "H1"])
    for _, handler in handlers:
        asyncio.run(handler(candle))

    assert [tf for tf, _ in calls] == ["M1", "H1"]
    assert all(payload is candle for _, payload in calls)


def test_bind_factor_handler_accepts_sync_callback() -> None:
    calls: list[tuple[str, float]] = []

    def _sync_on_candle(tf: str, candle: dict) -> None:
        calls.append((tf, float(candle["close"])))

    handler = market_data_feed_worker._bind_factor_handler("M5", _sync_on_candle)
    asyncio.run(handler({"close": 155.123}))

    assert calls == [("M5", 155.123)]
