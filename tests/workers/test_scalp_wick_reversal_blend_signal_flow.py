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
        "_signal_drought_revert",
        "_signal_precision_lowvol",
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
    assert thesis["flow_regime"] == "continuation_headwind"
    assert thesis["plus_di"] == 24.0
    assert thesis["minus_di"] == 19.0
