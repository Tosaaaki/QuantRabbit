from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple


def _load_worker_namespace() -> dict[str, object]:
    worker_path = (
        Path(__file__).resolve().parents[2]
        / "workers/scalp_wick_reversal_blend/worker.py"
    )
    source = worker_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(worker_path))
    wanted = {
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_drought_revert_setup_pressure",
        "_wick_blend_long_pressure_blocked",
        "_attach_flow_guard_context",
        "_signal_drought_revert",
        "_signal_precision_lowvol",
        "_signal_wick_reversal_blend",
        "_build_entry_thesis",
    }
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict[str, object] = {
        "Dict": Dict,
        "Optional": Optional,
        "Tuple": Tuple,
        "PIP": 0.01,
        "_to_probability": lambda value: max(0.0, min(1.0, float(value) / 100.0)),
        "config": SimpleNamespace(
            MAX_SPREAD_PIPS=1.2,
            DROUGHT_RANGE_SCORE=0.38,
            DROUGHT_SPREAD_P25=1.0,
            DROUGHT_ADX_MAX=26.0,
            DROUGHT_BBW_MAX=0.0016,
            DROUGHT_ATR_MIN=0.5,
            DROUGHT_ATR_MAX=4.0,
            DROUGHT_BB_TOUCH_PIPS=1.0,
            DROUGHT_RSI_LONG_MAX=49.0,
            DROUGHT_RSI_SHORT_MIN=51.0,
            DROUGHT_SETUP_PRESSURE_ALLOW_TOUCH_RATIO_MIN=0.50,
            DROUGHT_SETUP_PRESSURE_ALLOW_REV_STRENGTH_MIN=0.82,
            DROUGHT_SETUP_PRESSURE_ALLOW_SETUP_QUALITY_MIN=0.44,
            DROUGHT_SETUP_PRESSURE_ALLOW_REVERSION_SUPPORT_MIN=0.70,
            DROUGHT_SETUP_PRESSURE_ALLOW_PROJECTION_SCORE_MIN=0.10,
            DROUGHT_SETUP_PRESSURE_BLOCK_PROJECTION_SCORE_MAX=0.08,
            DROUGHT_SETUP_PRESSURE_BLOCK_SETUP_QUALITY_MAX=0.40,
            DROUGHT_SETUP_PRESSURE_BLOCK_REVERSION_SUPPORT_MAX=0.60,
            DROUGHT_SETUP_PRESSURE_BLOCK_CONTINUATION_PRESSURE_MIN=0.33,
            PREC_LOWVOL_RANGE_SCORE=0.25,
            PREC_LOWVOL_ADX_MAX=30.0,
            PREC_LOWVOL_BBW_MAX=0.0010,
            PREC_LOWVOL_ATR_MIN=0.3,
            PREC_LOWVOL_ATR_MAX=6.0,
            PREC_LOWVOL_BB_TOUCH_PIPS=1.6,
            PREC_LOWVOL_RSI_LONG_MAX=51.0,
            PREC_LOWVOL_RSI_SHORT_MIN=49.0,
            PREC_LOWVOL_STOCH_LONG_MAX=0.65,
            PREC_LOWVOL_STOCH_SHORT_MIN=0.35,
            PREC_LOWVOL_VWAP_GAP_MIN=0.15,
            PREC_LOWVOL_VWAP_GAP_BLOCK=1.8,
            PREC_LOWVOL_SPREAD_P25=1.8,
            PREC_LOWVOL_REV_MIN_STRENGTH=0.28,
            PREC_LOWVOL_WEAK_SHORT_GUARD_ENABLED=True,
            PREC_LOWVOL_WEAK_SHORT_RSI_MIN=60.0,
            PREC_LOWVOL_WEAK_SHORT_PROJECTION_SCORE_MAX=0.0,
            PREC_LOWVOL_WEAK_SHORT_SETUP_QUALITY_MAX=0.46,
            PREC_LOWVOL_MARGINAL_SHORT_GUARD_ENABLED=True,
            PREC_LOWVOL_MARGINAL_SHORT_RSI_MIN=59.0,
            PREC_LOWVOL_MARGINAL_SHORT_PROJECTION_SCORE_MAX=0.08,
            PREC_LOWVOL_MARGINAL_SHORT_SETUP_QUALITY_MAX=0.44,
            PREC_LOWVOL_MARGINAL_SHORT_CONTINUATION_PRESSURE_MIN=0.33,
            ENV_PREFIX="SCALP_PRECISION",
        ),
        "SPREAD_REV_TICK_MIN": 6,
        "TICK_IMB_PATTERN_GATE_OPT_IN": False,
        "TICK_IMB_PATTERN_GATE_ALLOW_GENERIC": False,
        "_latest_price": lambda fac: float(fac.get("close") or 0.0),
        "bb_levels": lambda fac: (
            float(fac["upper"]),
            float(fac.get("mid", fac["close"])),
            float(fac["lower"]),
            float(fac.get("bbw", 0.0)),
            float(fac["span_pips"]),
        ),
        "spread_ok": lambda **_: (True, 0.8),
        "_adx": lambda fac: float(fac.get("adx") or 0.0),
        "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
        "_atr_pips": lambda fac: float(fac.get("atr_pips") or 0.0),
        "_rsi": lambda fac: float(fac.get("rsi") or 0.0),
        "_stoch_rsi": lambda fac: float(fac.get("stoch_rsi") or 0.0),
        "_macd_hist_pips": lambda fac: float(fac.get("macd_hist_pips", fac.get("macd_hist") or 0.0)),
        "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0),
        "_vwap_gap_pips": lambda fac: float(fac.get("vwap_gap") or 0.0),
        "_div_score": lambda fac: float(fac.get("div_score") or 0.0),
        "tick_snapshot": lambda *_args, **_kwargs: ([1.0, 2.0, 3.0], 1.0),
        "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.38),
        "projection_decision": lambda side, mode="range": (True, 1.0, {"side": side, "mode": mode}),
        "wick_blend_entry_quality": lambda **_kwargs: {"allow": True, "quality": 0.75, "components": {}},
        "get_candles_snapshot": lambda *_args, **_kwargs: [
            {"open": 158.18, "high": 158.19, "low": 158.15, "close": 158.18}
        ],
        "_drought_revert_setup_pressure": lambda *_args, **_kwargs: {},
        "_wick_blend_long_setup_pressure": lambda *_args, **_kwargs: {},
        "_precision_lowvol_setup_pressure": lambda *_args, **_kwargs: {},
        "WICK_BLEND_RANGE_SCORE_MIN": 0.45,
        "WICK_BLEND_SPREAD_P25": 1.0,
        "WICK_BLEND_ADX_MIN": 0.0,
        "WICK_BLEND_ADX_MAX": 24.0,
        "WICK_BLEND_BBW_MAX": 0.0014,
        "WICK_BLEND_ATR_MIN": 0.8,
        "WICK_BLEND_ATR_MAX": 4.0,
        "WICK_BLEND_RANGE_MIN_PIPS": 1.0,
        "WICK_BLEND_BODY_MAX_PIPS": 2.2,
        "WICK_BLEND_BODY_RATIO_MAX": 0.75,
        "WICK_BLEND_WICK_RATIO_MIN": 0.35,
        "WICK_BLEND_BB_TOUCH_PIPS": 1.1,
        "WICK_BLEND_BB_TOUCH_RATIO": 0.22,
        "WICK_BLEND_REQUIRE_TICK_REV": True,
        "WICK_BLEND_TICK_WINDOW_SEC": 10.0,
        "WICK_BLEND_TICK_MIN_TICKS": 6,
        "WICK_BLEND_TICK_MIN_STRENGTH": 0.28,
        "WICK_BLEND_FOLLOW_PIPS": 0.0,
        "WICK_BLEND_EXTREME_RETRACE_MIN_PIPS": 0.0,
        "WICK_BLEND_DIAG": False,
        "WICK_BLEND_DIAG_INTERVAL_SEC": 20.0,
        "WICK_BLEND_LONG_SETUP_PRESSURE_ENABLED": True,
        "WICK_BLEND_LONG_SETUP_PRESSURE_BBW_MAX": 0.00055,
        "WICK_BLEND_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN": 0.40,
        "WICK_BLEND_LONG_SETUP_PRESSURE_RSI_MAX": 50.0,
        "WICK_BLEND_LONG_SETUP_PRESSURE_QUALITY_MAX": 0.83,
        "WICK_BLEND_LONG_SETUP_PRESSURE_PROJECTION_SCORE_MAX": 0.15,
    }
    exec(compile(module, str(worker_path), "exec"), namespace)
    return namespace


def test_reversion_short_flow_guard_blocks_current_like_continuation_short() -> None:
    ns = _load_worker_namespace()
    guard = ns["_reversion_short_flow_guard"]
    fac = {
        "adx": 16.2,
        "ema20": 158.022,
        "ema_slope_10": 0.018,
        "ema_slope_20": 0.012,
        "macd_hist": 0.93,
        "vwap_gap": 22.8,
        "plus_di": 28.7,
        "minus_di": 21.0,
    }

    allow, detail = guard(
        fac_m1=fac,
        price=158.046,
        dist_upper_pips=0.8,
        band_pips=2.2,
        range_score=0.337,
        rev_strength=0.32,
        profile="DroughtRevert",
    )

    assert allow is False
    assert detail["continuation_pressure"] > detail["max_pressure"]
    assert detail["stretch_pressure"] > 0.6
    assert detail["setup_quality"] < 0.25


def test_drought_revert_requires_projection_gate() -> None:
    ns = _load_worker_namespace()
    ns["projection_decision"] = lambda side, mode="range": (False, 1.0, {"side": side, "mode": mode})
    signal_fn = ns["_signal_drought_revert"]
    fac = {
        "close": 158.046,
        "upper": 158.054,
        "lower": 157.944,
        "span_pips": 11.0,
        "adx": 15.5,
        "bbw": 0.0008,
        "atr_pips": 2.4,
        "rsi": 56.0,
        "ema20": 158.020,
        "ema_slope_10": -0.01,
        "ema_slope_20": -0.01,
        "macd_hist": -0.1,
        "vwap_gap": 3.0,
        "plus_di": 18.0,
        "minus_di": 22.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.41)

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is None


def test_drought_revert_blocks_falling_knife_long_lane() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_drought_revert"]
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.66)
    fac = {
        "close": 158.012,
        "upper": 158.094,
        "lower": 158.006,
        "span_pips": 8.8,
        "adx": 21.0,
        "bbw": 0.0009,
        "atr_pips": 1.6,
        "rsi": 46.5,
        "ema20": 158.040,
        "ma10": 158.022,
        "ma20": 158.040,
        "ema_slope_10": -0.020,
        "ema_slope_20": -0.010,
        "macd_hist": -0.25,
        "vwap_gap": -2.1,
        "plus_di": 16.0,
        "minus_di": 24.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.42)

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is None


def test_drought_revert_boosts_strong_reclaim_long_lane() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_drought_revert"]
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.88)
    fac = {
        "close": 158.012,
        "upper": 158.094,
        "lower": 158.006,
        "span_pips": 8.8,
        "adx": 16.5,
        "bbw": 0.0008,
        "atr_pips": 1.6,
        "rsi": 42.0,
        "ema20": 158.024,
        "ma10": 158.020,
        "ma20": 158.028,
        "ema_slope_10": -0.005,
        "ema_slope_20": -0.003,
        "macd_hist": -0.06,
        "vwap_gap": -1.2,
        "plus_di": 19.0,
        "minus_di": 22.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.46)

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tp_pips"] >= 1.4
    assert signal["size_mult"] >= 0.95
    assert signal["sl_pips"] >= 1.7


def test_drought_revert_blocks_flat_gap_oversold_long_with_deep_mean_stretch() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_drought_revert"]
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.82)
    fac = {
        "close": 158.512,
        "upper": 158.598,
        "lower": 158.522,
        "span_pips": 7.6,
        "adx": 12.1,
        "bbw": 0.0006,
        "atr_pips": 2.88,
        "rsi": 44.7,
        "ema20": 158.544,
        "ma10": 158.542,
        "ma20": 158.5382,
        "ema_slope_10": 0.002,
        "ema_slope_20": 0.001,
        "macd_hist": -0.08,
        "vwap_gap": 37.8,
        "plus_di": 23.0,
        "minus_di": 29.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.35)

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is None


def test_drought_revert_blocks_weak_long_under_recent_setup_pressure() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_drought_revert"]
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.70)
    ns["_drought_revert_setup_pressure"] = lambda *_args, **_kwargs: {
        "trades": 8.0,
        "sl_rate": 0.625,
        "fast_sl_rate": 0.50,
        "net_jpy": -31.7,
        "last_close_age_sec": 1800.0,
        "active": 1.0,
    }
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.04},
    )
    fac = {
        "close": 158.017,
        "upper": 158.094,
        "lower": 158.006,
        "span_pips": 8.8,
        "adx": 17.7,
        "bbw": 0.0008,
        "atr_pips": 1.6,
        "rsi": 46.1,
        "ema20": 158.028,
        "ma10": 158.021,
        "ma20": 158.029,
        "ema_slope_10": -0.004,
        "ema_slope_20": -0.003,
        "macd_hist": -0.10,
        "vwap_gap": -1.1,
        "plus_di": 19.0,
        "minus_di": 25.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.46, reason="volatility_compression")

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is None


def test_drought_revert_keeps_strong_long_under_recent_setup_pressure() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_drought_revert"]
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.90)
    ns["_drought_revert_setup_pressure"] = lambda *_args, **_kwargs: {
        "trades": 8.0,
        "sl_rate": 0.625,
        "fast_sl_rate": 0.50,
        "net_jpy": -31.7,
        "last_close_age_sec": 1800.0,
        "active": 1.0,
    }
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.02},
    )
    fac = {
        "close": 158.007,
        "upper": 158.094,
        "lower": 158.006,
        "span_pips": 8.8,
        "adx": 16.5,
        "bbw": 0.0008,
        "atr_pips": 1.6,
        "rsi": 42.0,
        "ema20": 158.024,
        "ma10": 158.020,
        "ma20": 158.028,
        "ema_slope_10": -0.005,
        "ema_slope_20": -0.003,
        "macd_hist": -0.06,
        "vwap_gap": -1.2,
        "plus_di": 19.0,
        "minus_di": 22.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.46, reason="volatility_compression")

    signal = signal_fn(fac, range_ctx, tag="DroughtRevert")

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["setup_pressure"]["active"] == 1.0


def test_precision_lowvol_disables_vgap_bonus_when_flow_guard_is_marginal() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 17.0,
        "bbw": 0.00075,
        "atr_pips": 2.8,
        "rsi": 50.5,
        "stoch_rsi": 0.72,
        "vwap_gap": 2.2,
    }
    range_ctx = SimpleNamespace(active=True, score=0.44)

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.18,
            "max_pressure": 0.60,
            "setup_quality": 0.82,
            "reversion_support": 0.74,
        },
    )
    low_pressure = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.60,
            "max_pressure": 0.63,
            "setup_quality": 0.61,
            "reversion_support": 0.51,
        },
    )
    high_pressure = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert low_pressure is not None
    assert high_pressure is not None
    assert low_pressure["confidence"] > high_pressure["confidence"]
    assert low_pressure["size_mult"] > high_pressure["size_mult"]
    assert low_pressure["sl_pips"] >= 1.7
    assert high_pressure["sl_pips"] >= 1.7


def test_precision_lowvol_blocks_weak_short_under_recent_setup_pressure() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 17.0,
        "bbw": 0.00075,
        "atr_pips": 2.2,
        "rsi": 53.5,
        "stoch_rsi": 0.74,
        "vwap_gap": 1.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.44, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.25,
            "max_pressure": 0.58,
            "setup_quality": 0.34,
            "reversion_support": 0.56,
        },
    )
    ns["_precision_lowvol_setup_pressure"] = lambda *_args, **_kwargs: {
        "trades": 4.0,
        "sl_rate": 0.75,
        "fast_sl_rate": 0.50,
        "net_jpy": -26.0,
        "stop_loss_streak": 2.0,
        "fast_stop_loss_streak": 1.0,
        "last_close_age_sec": 45.0,
        "active": 1.0,
    }

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is None


def test_precision_lowvol_blocks_current_weak_overbought_short_lane() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 15.5,
        "bbw": 0.00036,
        "atr_pips": 2.2,
        "rsi": 61.2,
        "stoch_rsi": 1.0,
        "vwap_gap": 1.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.45, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.26,
            "max_pressure": 0.60,
            "setup_quality": 0.43,
            "reversion_support": 0.73,
        },
    )
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": -0.05},
    )

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is None


def test_precision_lowvol_keeps_higher_projection_short_when_rsi_is_high() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 15.5,
        "bbw": 0.00036,
        "atr_pips": 2.2,
        "rsi": 61.2,
        "stoch_rsi": 1.0,
        "vwap_gap": 1.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.45, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.22,
            "max_pressure": 0.60,
            "setup_quality": 0.43,
            "reversion_support": 0.73,
        },
    )
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.12},
    )

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_precision_lowvol_blocks_marginal_short_under_continuation_headwind() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 15.5,
        "bbw": 0.00036,
        "atr_pips": 2.2,
        "rsi": 59.4,
        "stoch_rsi": 0.91,
        "vwap_gap": 1.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.45, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.36,
            "max_pressure": 0.60,
            "setup_quality": 0.42,
            "reversion_support": 0.70,
        },
    )
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.05},
    )

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is None


def test_precision_lowvol_keeps_marginal_short_when_headwind_is_absent() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 15.5,
        "bbw": 0.00036,
        "atr_pips": 2.2,
        "rsi": 59.4,
        "stoch_rsi": 0.91,
        "vwap_gap": 1.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.45, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.18,
            "max_pressure": 0.60,
            "setup_quality": 0.42,
            "reversion_support": 0.70,
        },
    )
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.05},
    )

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_precision_lowvol_keeps_stronger_short_under_recent_setup_pressure() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_precision_lowvol"]
    fac = {
        "close": 158.046,
        "upper": 158.055,
        "lower": 157.945,
        "span_pips": 11.0,
        "adx": 17.0,
        "bbw": 0.00075,
        "atr_pips": 2.2,
        "rsi": 55.5,
        "stoch_rsi": 0.82,
        "vwap_gap": 1.8,
    }
    range_ctx = SimpleNamespace(active=True, score=0.44, reason="volatility_compression")

    ns["_reversion_short_flow_guard"] = lambda **_kwargs: (
        True,
        {
            "continuation_pressure": 0.19,
            "max_pressure": 0.60,
            "setup_quality": 0.38,
            "reversion_support": 0.74,
        },
    )
    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "short", 0.82)
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.12},
    )
    ns["_precision_lowvol_setup_pressure"] = lambda *_args, **_kwargs: {
        "trades": 4.0,
        "sl_rate": 0.75,
        "fast_sl_rate": 0.50,
        "net_jpy": -26.0,
        "stop_loss_streak": 2.0,
        "fast_stop_loss_streak": 1.0,
        "last_close_age_sec": 45.0,
        "active": 1.0,
    }

    signal = signal_fn(dict(fac), range_ctx, tag="PrecisionLowVol")

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["setup_pressure"]["active"] == 1.0


def test_wick_blend_long_pressure_blocked_for_current_breakout_loser_lane() -> None:
    ns = _load_worker_namespace()
    blocked = ns["_wick_blend_long_pressure_blocked"]

    assert (
        blocked(
            range_reason="volatility_compression",
            side="long",
            setup_pressure={"active": 1.0},
            bbw=0.00041,
            range_score=0.412,
            rsi=46.9,
            wick_quality=0.746,
            projection_score=0.14,
        )
        is True
    )


def test_wick_blend_long_pressure_keeps_stronger_lane() -> None:
    ns = _load_worker_namespace()
    blocked = ns["_wick_blend_long_pressure_blocked"]

    assert (
        blocked(
            range_reason="volatility_compression",
            side="long",
            setup_pressure={"active": 1.0},
            bbw=0.00050,
            range_score=0.323,
            rsi=62.7,
            wick_quality=0.841,
            projection_score=-0.265,
        )
        is False
    )


def test_wick_blend_signal_blocks_current_breakout_loser_lane() -> None:
    ns = _load_worker_namespace()
    signal_fn = ns["_signal_wick_reversal_blend"]
    fac = {
        "close": 159.18,
        "upper": 159.205,
        "lower": 159.161,
        "span_pips": 4.4,
        "adx": 22.1,
        "bbw": 0.000399,
        "atr_pips": 2.43,
        "rsi": 46.97,
    }
    range_ctx = SimpleNamespace(active=True, score=0.411, reason="volatility_compression")

    ns["tick_reversal"] = lambda *_args, **_kwargs: (True, "long", 0.9)
    ns["projection_decision"] = lambda side, mode="range": (
        True,
        1.0,
        {"side": side, "mode": mode, "score": 0.14},
    )
    ns["wick_blend_entry_quality"] = lambda **_kwargs: {
        "allow": True,
        "quality": 0.746,
        "components": {"range": 0.914},
    }
    ns["_wick_blend_long_setup_pressure"] = lambda *_args, **_kwargs: {"active": 1.0}

    signal = signal_fn(dict(fac), range_ctx, tag="WickReversalBlend")

    assert signal is None


def test_build_entry_thesis_promotes_flow_guard_to_dynamic_fields() -> None:
    ns = _load_worker_namespace()
    build_entry_thesis = ns["_build_entry_thesis"]
    signal = {
        "confidence": 68,
        "tag": "PrecisionLowVol",
        "reason": "precision_lowvol",
        "sl_pips": 1.2,
        "tp_pips": 1.8,
        "projection": {"side": "short", "mode": "range"},
        "flow_guard": {
            "continuation_pressure": 0.64,
            "reversion_support": 0.41,
            "setup_quality": 0.38,
        },
    }
    fac = {
        "rsi": 49.5,
        "adx": 18.0,
        "atr_pips": 2.5,
        "bbw": 0.0007,
        "stoch_rsi": 0.66,
        "macd_hist": 0.2,
        "vwap_gap": 4.1,
        "ema_slope_10": 0.04,
        "plus_di": 24.0,
        "minus_di": 19.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.42, reason="volatility_compression", mode="RANGE")

    thesis = build_entry_thesis(signal, fac, range_ctx)

    assert thesis["flow_guard"] == signal["flow_guard"]
    assert thesis["continuation_pressure"] == 0.64
    assert thesis["setup_quality"] == 0.38
    assert thesis["flow_headwind_regime"] == "continuation_headwind"
    assert "flow_regime" not in thesis
    assert thesis["plus_di"] == 24.0
    assert thesis["minus_di"] == 19.0


def test_build_entry_thesis_preserves_direct_dynamic_fields_without_nested_flow_guard() -> None:
    ns = _load_worker_namespace()
    build_entry_thesis = ns["_build_entry_thesis"]
    signal = {
        "confidence": 68,
        "tag": "VwapRevertS",
        "reason": "vwap_revert",
        "sl_pips": 1.8,
        "tp_pips": 2.2,
        "projection": {"side": "short", "mode": "range"},
        "continuation_pressure": 0.58,
        "reversion_support": 0.49,
        "setup_quality": 0.44,
        "flow_regime": "range_fade",
    }
    fac = {
        "rsi": 63.8,
        "adx": 21.4,
        "atr_pips": 2.53,
        "bbw": 0.00063,
        "stoch_rsi": 0.91,
        "macd_hist": 0.25,
        "vwap_gap": 18.4,
        "ema_slope_10": 0.05,
        "plus_di": 27.0,
        "minus_di": 18.0,
    }
    range_ctx = SimpleNamespace(active=True, score=0.37, reason="volatility_compression", mode="RANGE")

    thesis = build_entry_thesis(signal, fac, range_ctx)

    assert thesis["continuation_pressure"] == 0.58
    assert thesis["reversion_support"] == 0.49
    assert thesis["setup_quality"] == 0.44
    assert thesis["flow_regime"] == "range_fade"
