from __future__ import annotations

import datetime
import os

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _candle(o: float, h: float, l: float, c: float) -> dict[str, float]:
    return {"open": o, "high": h, "low": l, "close": c}


def _set_quickshot_env(monkeypatch, worker) -> None:
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_ENABLED", True)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_REQUIRE_BREAKOUT_RETEST", True)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_TARGET_JPY", 100.0)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_EST_COST_JPY", 12.0)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_MAX_SPREAD_PIPS", 0.30)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_BREAK_LOOKBACK_M5", 8)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_BREAK_MARGIN_PIPS", 0.25)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_PULLBACK_BAND_PIPS", 2.2)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_RETRACE_MIN_PIPS", 0.35)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_TP_ATR_MULT", 2.1)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_TP_PIPS_MIN", 6.0)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_TP_PIPS_MAX", 12.0)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_SL_TP_RATIO", 0.60)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_SL_PIPS_MIN", 3.5)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_SL_PIPS_MAX", 9.0)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_MIN_ENTRY_PROBABILITY", 0.55)
    monkeypatch.setattr(worker.config, "USDJPY_QUICKSHOT_BLOCK_JST_HOURS", frozenset({7}))


def _seed_candles(monkeypatch, worker) -> None:
    m5 = [
        _candle(155.88, 155.94, 155.82, 155.90),
        _candle(155.90, 155.95, 155.85, 155.93),
        _candle(155.93, 155.97, 155.89, 155.96),
        _candle(155.96, 155.99, 155.92, 155.98),
        _candle(155.98, 156.00, 155.95, 155.99),
        _candle(155.99, 156.00, 155.96, 155.98),
        _candle(155.98, 155.99, 155.94, 155.97),
        _candle(155.97, 156.00, 155.93, 155.99),
        _candle(156.00, 156.10, 155.99, 156.08),
    ]
    m1 = [
        _candle(156.06, 156.07, 156.04, 156.05),
        _candle(156.05, 156.06, 156.02, 156.03),
        _candle(156.03, 156.04, 156.01, 156.02),
        _candle(156.02, 156.03, 156.01, 156.02),
        _candle(156.02, 156.03, 156.01, 156.02),
    ]

    def _fake_snapshot(tf: str, limit: int = 0):
        if tf == "M5":
            return m5
        if tf == "M1":
            return m1
        return []

    monkeypatch.setattr(worker, "get_candles_snapshot", _fake_snapshot)


def test_detect_usdjpy_quickshot_plan_allows_long_breakout_pullback(monkeypatch) -> None:
    from workers.scalp_m1scalper import worker

    _set_quickshot_env(monkeypatch, worker)
    _seed_candles(monkeypatch, worker)

    allow, reason, detail = worker._detect_usdjpy_quickshot_plan(
        signal_side="long",
        signal={"notes": {"mode": "breakout_retest"}},
        price=156.02,
        spread_pips=0.18,
        atr_pips=3.8,
        now_utc=datetime.datetime(2026, 2, 27, 7, 10, tzinfo=datetime.timezone.utc),
    )

    assert allow is True
    assert reason == "allow"
    assert detail.get("mode") == "quickshot_breakout_pullback"
    assert detail.get("side") == "long"
    assert detail.get("target_units", 0) >= 1300
    assert detail.get("target_units", 0) <= 1600
    assert detail.get("tp_pips", 0) > detail.get("sl_pips", 0)
    assert detail.get("entry_probability", 0) >= 0.55


def test_detect_usdjpy_quickshot_plan_blocks_jst_maintenance_hour(monkeypatch) -> None:
    from workers.scalp_m1scalper import worker

    _set_quickshot_env(monkeypatch, worker)
    _seed_candles(monkeypatch, worker)

    allow, reason, detail = worker._detect_usdjpy_quickshot_plan(
        signal_side="long",
        signal={"notes": {"mode": "breakout_retest"}},
        price=156.02,
        spread_pips=0.18,
        atr_pips=3.8,
        now_utc=datetime.datetime(2026, 2, 26, 22, 10, tzinfo=datetime.timezone.utc),  # JST 07:10
    )

    assert allow is False
    assert reason == "quickshot_maintenance_hour"
    assert detail.get("jst_hour") == 7


def test_detect_usdjpy_quickshot_plan_blocks_side_mismatch(monkeypatch) -> None:
    from workers.scalp_m1scalper import worker

    _set_quickshot_env(monkeypatch, worker)
    _seed_candles(monkeypatch, worker)

    allow, reason, detail = worker._detect_usdjpy_quickshot_plan(
        signal_side="short",
        signal={"notes": {"mode": "breakout_retest"}},
        price=156.02,
        spread_pips=0.18,
        atr_pips=3.8,
        now_utc=datetime.datetime(2026, 2, 27, 7, 10, tzinfo=datetime.timezone.utc),
    )

    assert allow is False
    assert reason == "quickshot_side_mismatch"
    assert detail.get("signal_side") == "short"
    assert detail.get("m5_anchor_side") == "long"
