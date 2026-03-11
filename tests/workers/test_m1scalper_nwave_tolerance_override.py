from __future__ import annotations

from types import SimpleNamespace


def _fac(close: float) -> dict:
    return {
        "candles": [
            {"open": 156.000, "high": 156.004, "low": 155.996, "close": 156.000},
            {"open": 156.000, "high": 156.004, "low": 155.996, "close": 156.000},
        ],
        "close": close,
        "ema20": 156.000,
        "ema10": 156.000,
        "rsi": 50.0,
        "atr": 0.01,
        "vol_5m": 0.2,
        "adx": 12.0,
        "bbw": 0.002,
    }


def _trend_flip_fac() -> dict:
    candles = [
        {"open": 150.000, "high": 150.020, "low": 149.995, "close": 150.010},
        {"open": 150.010, "high": 150.065, "low": 150.005, "close": 150.060},
    ]
    return {
        "candles": candles,
        "open": candles[-1]["open"],
        "high": candles[-1]["high"],
        "low": candles[-1]["low"],
        "close": candles[-1]["close"],
        "ema20": 150.000,
        "ema10": 150.032,
        "rsi": 60.0,
        "atr": 0.02,
        "atr_pips": 2.0,
        "vol_5m": 0.3,
        "adx": 24.0,
        "bbw": 0.0011,
    }


def _breakout_retest_fac() -> dict:
    candles = [
        {"open": 155.992, "high": 156.010, "low": 155.988, "close": 155.998},
        {"open": 155.998, "high": 156.015, "low": 155.992, "close": 156.000},
        {"open": 156.000, "high": 156.020, "low": 155.990, "close": 156.000},
        {"open": 156.000, "high": 156.025, "low": 155.995, "close": 156.010},
        {"open": 156.010, "high": 156.030, "low": 156.000, "close": 156.015},
        {"open": 156.015, "high": 156.028, "low": 156.000, "close": 156.005},
        {"open": 156.005, "high": 156.022, "low": 155.998, "close": 156.000},
        {"open": 156.000, "high": 156.018, "low": 155.996, "close": 156.008},
        {"open": 156.032, "high": 156.065, "low": 156.026, "close": 156.060},
        {"open": 156.060, "high": 156.062, "low": 156.034, "close": 156.038},
    ]
    return {
        "candles": candles,
        "open": candles[-1]["open"],
        "high": candles[-1]["high"],
        "low": candles[-1]["low"],
        "close": candles[-1]["close"],
        "ema20": 156.000,
        "ema10": 156.022,
        "rsi": 57.0,
        "atr": 0.02,
        "atr_pips": 2.0,
        "vol_5m": 0.28,
        "adx": 24.0,
        "bbw": 0.0012,
        "range_active": False,
        "range_score": 0.14,
    }


def test_nwave_tolerance_default_can_be_overridden_by_env(monkeypatch):
    from strategies.scalping import m1_scalper as strategy

    monkeypatch.setattr(strategy, "_load_scalper_config", lambda: {})
    monkeypatch.setattr(strategy, "_NWAVE_ALIGN_ENABLED", False)
    monkeypatch.setattr(
        strategy,
        "detect_latest_n_wave",
        lambda _candles: SimpleNamespace(
            direction="long",
            entry_price=156.000,
            pullback_pips=1.2,
            amplitude_pips=3.0,
            quality=1.0,
        ),
    )
    monkeypatch.delenv("M1SCALP_NWAVE_TOL_DEF_PIPS", raising=False)
    monkeypatch.delenv("M1SCALP_NWAVE_TOLERANCE_DEFAULT_PIPS", raising=False)

    # close is 0.45p above entry. default tolerance floor is 0.42p, so this is skipped as late.
    signal = strategy.M1Scalper.check(_fac(close=156.0045))
    assert signal is None

    monkeypatch.setenv("M1SCALP_NWAVE_TOL_DEF_PIPS", "0.50")
    signal = strategy.M1Scalper.check(_fac(close=156.0045))
    assert isinstance(signal, dict)
    assert signal.get("action") == "OPEN_LONG"
    assert float(signal.get("entry_tolerance_pips") or 0.0) >= 0.50


def test_trend_flip_extreme_guard_blocks_extended_trend_long(monkeypatch):
    from strategies.scalping import m1_scalper as strategy

    monkeypatch.setattr(strategy, "_load_scalper_config", lambda: {})
    monkeypatch.setattr(strategy, "_shock_guard", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(strategy, "_liquidity_guard", lambda: True)
    monkeypatch.setattr(strategy, "detect_latest_n_wave", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(strategy, "_BREAKOUT_RETEST_ENABLED", False)
    monkeypatch.setattr(strategy, "_VSHAPE_REBOUND_ENABLED", False)
    monkeypatch.setattr(strategy, "derive_pattern_signature", None)

    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GUARD_ENABLED", False)
    signal = strategy.M1Scalper.check(_trend_flip_fac())
    assert isinstance(signal, dict)
    assert signal.get("tag") == "M1Scalper-trend-long"

    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GUARD_ENABLED", True)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_ATR_MAX_PIPS", 3.2)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_BBW_MAX", 0.0014)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GAP_MIN_PIPS", 2.5)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_LONG_RSI_MIN", 58.0)

    blocked = strategy.M1Scalper.check(_trend_flip_fac())
    assert blocked is None


def test_breakout_retest_signal_exposes_dynamic_setup_metadata(monkeypatch):
    from strategies.scalping import m1_scalper as strategy

    fac = _breakout_retest_fac()
    signal = strategy._breakout_retest_signal(
        candles=fac["candles"],
        close=fac["close"],
        ema20=fac["ema20"],
        rsi=fac["rsi"],
        atr_pips=fac["atr_pips"],
        adx=fac["adx"],
        range_active=fac["range_active"],
        range_score=fac["range_score"],
        price_gap_pips=(fac["close"] - fac["ema20"]) / 0.01,
        ema_gap_pips=(fac["ema10"] - fac["ema20"]) / 0.01,
        bbw=fac["bbw"],
        vol5=fac["vol_5m"],
        trend_up=True,
        trend_down=False,
        strong_up=False,
        strong_down=False,
        range_reversion_only=False,
        fast_cut=5.0,
        fast_cut_time=60.0,
        tp_dyn=6.0,
        sl_dyn=4.2,
        adjust_tp=lambda tp, _conf: tp,
    )

    assert isinstance(signal, dict)
    assert signal.get("tag") == "M1Scalper-breakout-retest-long"
    assert 0.40 <= float(signal.get("entry_probability") or 0.0) <= 0.95
    assert 0.0 <= float(signal.get("setup_quality") or 0.0) <= 1.0
    assert 0.65 <= float(signal.get("setup_size_mult") or 0.0) <= 1.35
    notes = signal.get("notes") or {}
    assert notes.get("mode") == "breakout_retest"
    assert notes.get("flow_regime")
    assert notes.get("setup_fingerprint")
    assert isinstance(notes.get("continuation_pressure"), int)
