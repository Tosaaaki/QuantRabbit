from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

from workers.scalp_extrema_reversal import worker


def _range_ctx(*, active: bool, score: float, mode: str) -> SimpleNamespace:
    return SimpleNamespace(active=active, score=score, mode=mode, reason=mode.lower())


def test_extrema_trend_gate_blocks_countertrend_continuation(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_SCORE_MIN", 0.40)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS", 1.0)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 24.0, "ma10": 158.452, "ma20": 158.434},
        range_ctx=_range_ctx(active=False, score=0.26, mode="TREND"),
    )

    assert ok is False
    assert diag["range_score"] == 0.26
    assert diag["against_gap_pips"] > 1.2


def test_extrema_trend_gate_blocks_trend_mode_without_range_ready(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_SCORE_MIN", 0.40)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS", 1.0)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 14.5, "ma10": 158.4402, "ma20": 158.4402},
        range_ctx=_range_ctx(active=False, score=0.2909, mode="TREND"),
    )

    assert ok is False
    assert diag["range_mode"] == 1.0
    assert diag["range_score"] == 0.2909


def test_extrema_trend_gate_allows_when_range_context_is_ready(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_SCORE_MIN", 0.40)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS", 1.0)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 24.0, "ma10": 158.444, "ma20": 158.434},
        range_ctx=_range_ctx(active=True, score=0.42, mode="RANGE"),
    )

    assert ok is True
    assert diag["range_active"] == 1.0


def test_extrema_trend_gate_blocks_weak_range_mode_even_if_active(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MIN_RANGE_SCORE", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_ADX_MIN", 18.0)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_MA_GAP_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_SCORE_MIN", 0.40)
    monkeypatch.setattr(worker, "EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS", 1.0)

    ok, diag = worker._extrema_trend_gate_ok(
        "short",
        {"adx": 9.7, "ma10": 158.4524, "ma20": 158.44},
        range_ctx=_range_ctx(active=True, score=0.36, mode="RANGE"),
    )

    assert ok is False
    assert diag["range_active"] == 1.0
    assert diag["range_score"] == 0.36
    assert diag["against_gap_pips"] > 1.0


def test_place_order_uses_actual_free_ratio_for_cap(monkeypatch):
    called: dict[str, float] = {}

    def fake_compute_cap(**kwargs):
        called["free_ratio"] = kwargs["free_ratio"]
        return SimpleNamespace(cap=0.5, reasons={})

    async def fake_market_order(**kwargs):
        return kwargs

    monkeypatch.setattr(worker, "compute_cap", fake_compute_cap)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 3.0)
    monkeypatch.setattr(worker, "_adx", lambda *_args, **_kwargs: 18.0)
    monkeypatch.setattr(
        worker,
        "compute_units",
        lambda **_kwargs: SimpleNamespace(units=100, factors={"free_ratio": 0.07}),
    )
    monkeypatch.setattr(worker, "clamp_sl_tp", lambda **kwargs: (kwargs["sl"], kwargs["tp"]))
    monkeypatch.setattr(worker, "market_order", fake_market_order)
    monkeypatch.setattr(worker.spread_monitor, "get_state", lambda: {"spread_pips": 0.8})

    import utils.oanda_account as oanda_account

    monkeypatch.setattr(
        oanda_account,
        "get_account_snapshot",
        lambda: SimpleNamespace(free_margin_ratio=0.07),
    )

    result = asyncio.run(
        worker._place_order(
            {
                "action": "OPEN_SHORT",
                "sl_pips": 1.5,
                "tp_pips": 2.0,
                "confidence": 70,
                "tag": "scalp_extrema_reversal_live",
                "reason": "extrema_reversal",
                "extrema": {},
            },
            fac_m1={"close": 158.450, "adx": 18.0, "atr_pips": 3.0, "rsi": 58.0},
            fac_h4={},
            range_ctx=_range_ctx(active=False, score=0.32, mode="RANGE"),
        )
    )

    assert called["free_ratio"] == 0.07
    assert result["meta"]["cap"] == 0.5


def test_signal_extrema_reversal_uses_wider_strategy_local_sl_tp_caps(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_HIGH_BAND_PIPS", 1.5)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 1.5)
    monkeypatch.setattr(worker, "EXTREMA_RSI_SHORT_MIN", 54.0)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_SL_ATR_MULT", 0.95)
    monkeypatch.setattr(worker, "EXTREMA_TP_ATR_MULT", 1.25)
    monkeypatch.setattr(worker, "EXTREMA_SL_MIN_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_SL_MAX_PIPS", 2.6)
    monkeypatch.setattr(worker, "EXTREMA_TP_MIN_PIPS", 1.4)
    monkeypatch.setattr(worker, "EXTREMA_TP_MAX_PIPS", 3.2)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_rsi", lambda *_args, **_kwargs: 58.0)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 3.0)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.462, "low": 158.430}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.456, low=158.430),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.462, 158.459, 158.455, 158.452, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "short", 1.2))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {"close": 158.450, "adx": 18.0, "atr_pips": 3.0, "rsi": 58.0},
        range_ctx=_range_ctx(active=True, score=0.42, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["sl_pips"] == 2.6
    assert signal["tp_pips"] == 3.2


def test_signal_extrema_reversal_allows_supportive_long_when_m5_is_bullish(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_SL_ATR_MULT", 0.95)
    monkeypatch.setattr(worker, "EXTREMA_TP_ATR_MULT", 1.25)
    monkeypatch.setattr(worker, "EXTREMA_SL_MIN_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_SL_MAX_PIPS", 2.6)
    monkeypatch.setattr(worker, "EXTREMA_TP_MIN_PIPS", 1.4)
    monkeypatch.setattr(worker, "EXTREMA_TP_MAX_PIPS", 3.2)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_RSI_MIN", 56.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_DI_GAP_MIN", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_EMA_SLOPE_MIN", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M1_ADX_MAX", 24.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M1_EMA_GAP_MAX_PIPS", 1.4)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_RSI_CAP", 50.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_LOW_BAND_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_CONF_BONUS", 4)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.447}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.447),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.450, 158.449, 158.448, 158.447, 158.448, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.8))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    fac_m1 = {"close": 158.450, "ema20": 158.446, "adx": 18.0, "atr_pips": 1.8, "rsi": 49.0}
    fac_m5 = {"close": 158.470, "ema20": 158.440, "rsi": 61.0, "plus_di": 25.0, "minus_di": 16.0, "ema_slope_10": 0.01}
    signal = worker._signal_extrema_reversal(
        fac_m1,
        fac_m5=fac_m5,
        range_ctx=_range_ctx(active=True, score=0.44, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["confidence"] == 66
    assert signal["extrema"]["supportive_long"] is True
    assert signal["extrema"]["long_rsi_cap"] == 50.0
    assert signal["extrema"]["long_low_band_pips"] == 1.2


def test_signal_extrema_reversal_requires_supportive_context_for_shallow_long(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.440}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.440),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.446, 158.444, 158.441, 158.439, 158.442, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.8))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {"close": 158.450, "ema20": 158.446, "adx": 18.0, "atr_pips": 1.8, "rsi": 49.0},
        fac_m5={"close": 158.430, "ema20": 158.440, "rsi": 52.0, "plus_di": 16.0, "minus_di": 20.0, "ema_slope_10": -0.01},
        range_ctx=_range_ctx(active=True, score=0.44, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is None


def test_signal_extrema_reversal_blocks_non_supportive_long_under_countertrend_gap(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS", 0.5)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.440}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.440),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.446, 158.444, 158.441, 158.439, 158.442, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.8))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {
            "close": 158.450,
            "ma10": 158.440,
            "ma20": 158.446,
            "ema20": 158.446,
            "adx": 18.0,
            "atr_pips": 1.8,
            "rsi": 40.0,
        },
        fac_m5={"close": 158.430, "ema20": 158.440, "rsi": 52.0, "plus_di": 16.0, "minus_di": 20.0, "ema_slope_10": -0.01},
        range_ctx=_range_ctx(active=True, score=0.44, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is None


def test_signal_extrema_reversal_keeps_supportive_long_under_same_countertrend_gap(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS", 0.5)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_RSI_MIN", 56.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_DI_GAP_MIN", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M5_EMA_SLOPE_MIN", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M1_ADX_MAX", 24.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_M1_EMA_GAP_MAX_PIPS", 1.4)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_RSI_CAP", 50.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_LOW_BAND_PIPS", 1.2)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_CONF_BONUS", 4)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.440}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.440),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.446, 158.444, 158.441, 158.439, 158.442, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.8))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {
            "close": 158.450,
            "ma10": 158.440,
            "ma20": 158.446,
            "ema20": 158.446,
            "adx": 18.0,
            "atr_pips": 1.8,
            "rsi": 49.0,
        },
        fac_m5={"close": 158.470, "ema20": 158.440, "rsi": 61.0, "plus_di": 25.0, "minus_di": 16.0, "ema_slope_10": 0.01},
        range_ctx=_range_ctx(active=True, score=0.44, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["extrema"]["supportive_long"] is True


def test_signal_extrema_reversal_blocks_non_supportive_shallow_probe_long(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS", 0.5)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_BOUNCE_MAX_PIPS", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_TICK_STRENGTH_MAX", 0.20)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_ADX_MAX", 13.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_RANGE_SCORE_MAX", 0.32)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.440}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.440),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.446, 158.444, 158.441, 158.439, 158.442, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.2))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {"close": 158.450, "ema20": 158.446, "adx": 12.6, "atr_pips": 1.8, "rsi": 40.2},
        fac_m5={"close": 158.430, "ema20": 158.440, "rsi": 52.0, "plus_di": 16.0, "minus_di": 20.0, "ema_slope_10": -0.01},
        range_ctx=_range_ctx(active=True, score=0.31, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is None


def test_signal_extrema_reversal_keeps_non_supportive_long_with_deeper_probe(monkeypatch):
    monkeypatch.setattr(worker, "EXTREMA_ALLOWED_REGIMES", set())
    monkeypatch.setattr(worker, "EXTREMA_SPREAD_P25_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_ADX_MAX", 35.0)
    monkeypatch.setattr(worker, "EXTREMA_ATR_MAX", 0.0)
    monkeypatch.setattr(worker, "EXTREMA_SHORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LOW_BAND_PIPS", 0.9)
    monkeypatch.setattr(worker, "EXTREMA_RSI_LONG_MAX", 46.0)
    monkeypatch.setattr(worker, "EXTREMA_SWEEP_MIN_PIPS", 0.06)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SUPPORT_ENABLED", True)
    monkeypatch.setattr(worker, "EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS", 0.5)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_BOUNCE_MAX_PIPS", 0.30)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_TICK_STRENGTH_MAX", 0.20)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_ADX_MAX", 13.0)
    monkeypatch.setattr(worker, "EXTREMA_LONG_SHALLOW_PROBE_RANGE_SCORE_MAX", 0.32)
    monkeypatch.setattr(worker, "_latest_price", lambda *_args, **_kwargs: 158.450)
    monkeypatch.setattr(worker, "_atr_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(
        worker,
        "get_candles_snapshot",
        lambda *_args, **_kwargs: [{"high": 158.464, "low": 158.444}] * 80,
    )
    monkeypatch.setattr(
        worker,
        "compute_range_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(high=158.464, low=158.444),
    )
    monkeypatch.setattr(
        worker,
        "tick_snapshot",
        lambda *_args, **_kwargs: ([158.449, 158.447, 158.444, 158.445, 158.447, 158.450], None),
    )
    monkeypatch.setattr(worker, "tick_reversal", lambda *_args, **_kwargs: (True, "long", 0.6))
    monkeypatch.setattr(worker, "_extrema_trend_gate_ok", lambda *_args, **_kwargs: (True, {}))

    signal = worker._signal_extrema_reversal(
        {"close": 158.450, "ema20": 158.446, "adx": 12.9, "atr_pips": 1.8, "rsi": 45.3},
        fac_m5={"close": 158.430, "ema20": 158.440, "rsi": 52.0, "plus_di": 16.0, "minus_di": 20.0, "ema_slope_10": -0.01},
        range_ctx=_range_ctx(active=True, score=0.31, mode="RANGE"),
        tag="scalp_extrema_reversal_live",
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
