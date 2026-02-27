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
