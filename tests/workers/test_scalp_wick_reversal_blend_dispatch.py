from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Tuple

import pytest


def _worker_tree() -> tuple[Path, ast.Module]:
    worker_path = (
        Path(__file__).resolve().parents[2]
        / "workers/scalp_wick_reversal_blend/worker.py"
    )
    source = worker_path.read_text(encoding="utf-8")
    return worker_path, ast.parse(source, filename=str(worker_path))


def _load_dispatch_helper():
    worker_path, tree = _worker_tree()
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_RANGE_CTX_SIGNAL_NAMES",
                    "_M5_SIGNAL_NAMES",
                }:
                    selected.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name == "_dispatch_strategy_signal":
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "Callable": Callable,
        "Dict": Dict,
        "Optional": Optional,
        "datetime": dt,
    }
    exec(compile(module, str(worker_path), "exec"), namespace)
    return namespace["_dispatch_strategy_signal"]


def _load_worker_functions(*names: str):
    worker_path, tree = _worker_tree()
    selected = []
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted.union({"_attach_flow_guard_context"}):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "Dict": Dict,
        "Optional": Optional,
        "Tuple": Tuple,
        "PIP": 0.01,
        "_precision_lowvol_setup_pressure": lambda *_args, **_kwargs: {},
    }
    exec(compile(module, str(worker_path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", ["DroughtRevert", "PrecisionLowVol"])
def test_dispatch_strategy_signal_passes_range_ctx(name: str) -> None:
    dispatch = _load_dispatch_helper()
    marker = object()

    def fake_signal(fac_m1, range_ctx, **kwargs):
        assert fac_m1 == {"close": 150.0}
        assert range_ctx is marker
        assert kwargs == {"tag": name}
        return {"tag": name}

    signal = dispatch(
        name=name,
        fn=fake_signal,
        fac_m1={"close": 150.0},
        fac_h1={},
        fac_m5={},
        range_ctx=marker,
        now_utc=dt.datetime(2026, 3, 10, 12, 0, 0),
        kwargs={"tag": name},
    )

    assert signal == {"tag": name}


@pytest.mark.parametrize(
    ("func_name", "tag", "config_attrs"),
    [
        (
            "_signal_drought_revert",
            "DroughtRevert",
            {
                "MAX_SPREAD_PIPS": 2.0,
                "DROUGHT_RANGE_SCORE": 0.38,
                "DROUGHT_SPREAD_P25": 1.0,
                "DROUGHT_ADX_MAX": 26.0,
                "DROUGHT_BBW_MAX": 0.0016,
                "DROUGHT_ATR_MIN": 0.5,
                "DROUGHT_ATR_MAX": 4.0,
                "DROUGHT_BB_TOUCH_PIPS": 1.0,
                "DROUGHT_RSI_LONG_MAX": 49.0,
                "DROUGHT_RSI_SHORT_MIN": 51.0,
            },
        ),
        (
            "_signal_precision_lowvol",
            "PrecisionLowVol",
            {
                "MAX_SPREAD_PIPS": 2.0,
                "PREC_LOWVOL_RANGE_SCORE": 0.25,
                "PREC_LOWVOL_ADX_MAX": 30.0,
                "PREC_LOWVOL_BBW_MAX": 0.0010,
                "PREC_LOWVOL_ATR_MIN": 0.3,
                "PREC_LOWVOL_ATR_MAX": 6.0,
                "PREC_LOWVOL_BB_TOUCH_PIPS": 1.6,
                "PREC_LOWVOL_RSI_LONG_MAX": 51.0,
                "PREC_LOWVOL_RSI_SHORT_MIN": 49.0,
                "PREC_LOWVOL_STOCH_LONG_MAX": 0.65,
                "PREC_LOWVOL_STOCH_SHORT_MIN": 0.35,
                "PREC_LOWVOL_VWAP_GAP_MIN": 0.15,
                "PREC_LOWVOL_VWAP_GAP_BLOCK": 1.8,
                "PREC_LOWVOL_SPREAD_P25": 1.8,
                "PREC_LOWVOL_REV_MIN_STRENGTH": 0.28,
            },
        ),
        (
            "_signal_vwap_revert",
            "VwapRevertS",
            {
                "MAX_SPREAD_PIPS": 2.0,
            },
        ),
    ],
)
def test_short_reversion_signals_block_wrong_way_bullish_headwind(
    func_name: str,
    tag: str,
    config_attrs: dict[str, float],
) -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_drought_revert",
        "_signal_precision_lowvol",
        "_signal_vwap_revert",
    )
    namespace.update(
        {
            "config": SimpleNamespace(**config_attrs),
            "SPREAD_REV_TICK_MIN": 6,
            "VWAP_REV_RANGE_SCORE": 0.40,
            "VWAP_REV_SPREAD_P25": 0.9,
            "VWAP_REV_ADX_MAX": 22.0,
            "VWAP_REV_BBW_MAX": 0.0014,
            "VWAP_REV_ATR_MIN": 0.7,
            "VWAP_REV_ATR_MAX": 3.2,
            "VWAP_REV_GAP_MIN": 1.2,
            "VWAP_REV_BB_TOUCH_PIPS": 0.9,
            "VWAP_REV_RSI_LONG_MAX": 46.0,
            "VWAP_REV_RSI_SHORT_MIN": 54.0,
            "VWAP_REV_STOCH_LONG_MAX": 0.2,
            "VWAP_REV_STOCH_SHORT_MIN": 0.8,
            "_latest_price": lambda _fac: 158.198,
            "bb_levels": lambda _fac: (158.200, 158.160, 158.120, 0.080, 8.0),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0) / 0.01,
            "_macd_hist_pips": lambda fac: float(fac.get("macd_hist") or 0.0) / 0.01,
            "tick_snapshot": lambda *_args, **_kwargs: ([158.180, 158.191, 158.198], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.82),
            "projection_decision": lambda side, mode="range": (side == "short", 1.0, {"mode": mode}),
        }
    )

    fac_m1 = {
        "adx": 23.0,
        "bbw": 0.0008,
        "atr": 1.4,
        "rsi": 58.0,
        "stoch": 0.92,
        "vgap": 1.7,
        "ema20": 158.150,
        "ema_slope_10": 0.0028,
        "ema_slope_20": 0.0018,
        "macd_hist": 0.0058,
    }
    range_ctx = SimpleNamespace(active=True, score=0.62)

    signal = namespace[func_name](fac_m1, range_ctx, tag=tag)

    assert signal is None


def test_precision_lowvol_short_keeps_local_flow_guard_when_headwind_is_mild() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_precision_lowvol",
    )
    namespace.update(
        {
            "config": SimpleNamespace(
                MAX_SPREAD_PIPS=2.0,
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
            ),
            "_latest_price": lambda _fac: 158.198,
            "bb_levels": lambda _fac: (158.200, 158.180, 158.160, 0.040, 4.0),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0) / 0.01,
            "_macd_hist_pips": lambda fac: float(fac.get("macd_hist") or 0.0) / 0.01,
            "tick_snapshot": lambda *_args, **_kwargs: ([158.188, 158.196, 158.198], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.88),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.08,
                {"mode": mode, "score": 0.22},
            ),
        }
    )

    fac_m1 = {
        "adx": 18.0,
        "bbw": 0.0007,
        "atr": 1.0,
        "rsi": 56.0,
        "stoch": 0.86,
        "vgap": 1.1,
        "ema20": 158.186,
        "ema_slope_10": 0.0003,
        "ema_slope_20": 0.0001,
        "macd_hist": 0.0010,
    }
    range_ctx = SimpleNamespace(active=True, score=0.66)

    signal = namespace["_signal_precision_lowvol"](fac_m1, range_ctx, tag="PrecisionLowVol")

    assert isinstance(signal, dict)
    assert signal["action"] == "OPEN_SHORT"
    assert signal["tag"] == "PrecisionLowVol"
    assert signal["reason"] == "precision_lowvol"
    assert signal["size_mult"] < 1.08
    assert signal["flow_guard"]["setup_quality"] > 0.5
    assert signal["flow_guard"]["continuation_pressure"] < signal["flow_guard"]["max_pressure"]


def test_precision_lowvol_blocks_weak_gap_stretched_hostile_projection_short_lane() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_precision_lowvol",
    )
    namespace.update(
        {
            "config": SimpleNamespace(
                MAX_SPREAD_PIPS=2.0,
                PREC_LOWVOL_RANGE_SCORE=0.18,
                PREC_LOWVOL_ADX_MAX=24.0,
                PREC_LOWVOL_BBW_MAX=0.0016,
                PREC_LOWVOL_ATR_MIN=0.4,
                PREC_LOWVOL_ATR_MAX=4.4,
                PREC_LOWVOL_BB_TOUCH_PIPS=1.2,
                PREC_LOWVOL_RSI_LONG_MAX=48.0,
                PREC_LOWVOL_RSI_SHORT_MIN=52.0,
                PREC_LOWVOL_STOCH_LONG_MAX=0.55,
                PREC_LOWVOL_STOCH_SHORT_MIN=0.45,
                PREC_LOWVOL_VWAP_GAP_MIN=0.08,
                PREC_LOWVOL_VWAP_GAP_BLOCK=64.0,
                PREC_LOWVOL_SPREAD_P25=1.0,
                PREC_LOWVOL_REV_MIN_STRENGTH=0.18,
            ),
            "_latest_price": lambda _fac: 158.286,
            "bb_levels": lambda _fac: (158.288, 158.272, 158.244, 0.044, 4.4),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "tick_snapshot": lambda *_args, **_kwargs: ([158.278, 158.283, 158.286], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.58),
            "_reversion_short_flow_guard": lambda **_kwargs: (
                True,
                {
                    "continuation_pressure": 0.34,
                    "max_pressure": 0.41,
                    "setup_quality": 0.32,
                    "reversion_support": 0.45,
                },
            ),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.0,
                {"mode": mode, "score": -0.125},
            ),
        }
    )

    fac_m1 = {
        "adx": 16.8,
        "bbw": 0.00054,
        "atr": 2.13,
        "rsi": 60.7,
        "stoch": 0.81,
        "vgap": 29.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.43)

    signal = namespace["_signal_precision_lowvol"](fac_m1, range_ctx, tag="PrecisionLowVol")

    assert signal is None


def test_precision_lowvol_keeps_strong_reversal_probe_under_hostile_projection() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_precision_lowvol",
    )
    namespace.update(
        {
            "config": SimpleNamespace(
                MAX_SPREAD_PIPS=2.0,
                PREC_LOWVOL_RANGE_SCORE=0.18,
                PREC_LOWVOL_ADX_MAX=24.0,
                PREC_LOWVOL_BBW_MAX=0.0016,
                PREC_LOWVOL_ATR_MIN=0.4,
                PREC_LOWVOL_ATR_MAX=4.4,
                PREC_LOWVOL_BB_TOUCH_PIPS=1.2,
                PREC_LOWVOL_RSI_LONG_MAX=48.0,
                PREC_LOWVOL_RSI_SHORT_MIN=52.0,
                PREC_LOWVOL_STOCH_LONG_MAX=0.55,
                PREC_LOWVOL_STOCH_SHORT_MIN=0.45,
                PREC_LOWVOL_VWAP_GAP_MIN=0.08,
                PREC_LOWVOL_VWAP_GAP_BLOCK=64.0,
                PREC_LOWVOL_SPREAD_P25=1.0,
                PREC_LOWVOL_REV_MIN_STRENGTH=0.18,
            ),
            "_latest_price": lambda _fac: 158.287,
            "bb_levels": lambda _fac: (158.288, 158.272, 158.244, 0.044, 4.4),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "tick_snapshot": lambda *_args, **_kwargs: ([158.278, 158.284, 158.287], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.88),
            "_reversion_short_flow_guard": lambda **_kwargs: (
                True,
                {
                    "continuation_pressure": 0.34,
                    "max_pressure": 0.41,
                    "setup_quality": 0.34,
                    "reversion_support": 0.47,
                },
            ),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.08,
                {"mode": mode, "score": -0.125},
            ),
        }
    )

    fac_m1 = {
        "adx": 16.8,
        "bbw": 0.00054,
        "atr": 2.13,
        "rsi": 60.7,
        "stoch": 0.81,
        "vgap": 29.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.43)

    signal = namespace["_signal_precision_lowvol"](fac_m1, range_ctx, tag="PrecisionLowVol")

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["confidence"] < 80
    assert signal["size_mult"] <= 1.02


def test_precision_lowvol_prefers_short_up_lean_over_down_flat_lane() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_precision_lowvol",
    )
    namespace.update(
        {
            "config": SimpleNamespace(
                MAX_SPREAD_PIPS=2.0,
                PREC_LOWVOL_RANGE_SCORE=0.18,
                PREC_LOWVOL_ADX_MAX=24.0,
                PREC_LOWVOL_BBW_MAX=0.0016,
                PREC_LOWVOL_ATR_MIN=0.4,
                PREC_LOWVOL_ATR_MAX=4.4,
                PREC_LOWVOL_BB_TOUCH_PIPS=1.2,
                PREC_LOWVOL_RSI_LONG_MAX=48.0,
                PREC_LOWVOL_RSI_SHORT_MIN=52.0,
                PREC_LOWVOL_STOCH_LONG_MAX=0.55,
                PREC_LOWVOL_STOCH_SHORT_MIN=0.45,
                PREC_LOWVOL_VWAP_GAP_MIN=0.08,
                PREC_LOWVOL_VWAP_GAP_BLOCK=64.0,
                PREC_LOWVOL_SPREAD_P25=1.0,
                PREC_LOWVOL_REV_MIN_STRENGTH=0.18,
            ),
            "_latest_price": lambda _fac: 158.286,
            "bb_levels": lambda _fac: (158.288, 158.272, 158.244, 0.044, 4.4),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0) / 0.01,
            "_macd_hist_pips": lambda fac: float(fac.get("macd_hist") or 0.0) / 0.01,
            "tick_snapshot": lambda *_args, **_kwargs: ([158.278, 158.284, 158.286], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.88),
            "_reversion_short_flow_guard": lambda **_kwargs: (
                True,
                {
                    "continuation_pressure": 0.22,
                    "max_pressure": 0.46,
                    "setup_quality": 0.71,
                    "reversion_support": 0.62,
                },
            ),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.06,
                {"mode": mode, "score": 0.18},
            ),
        }
    )

    base_fac = {
        "adx": 16.8,
        "bbw": 0.00054,
        "atr": 2.13,
        "rsi": 60.7,
        "stoch": 0.81,
        "vgap": 2.4,
    }
    range_ctx = SimpleNamespace(active=True, score=0.43)

    up_lean = namespace["_signal_precision_lowvol"](
        {
            **base_fac,
            "ma10": 158.304,
            "ma20": 158.298,
        },
        range_ctx,
        tag="PrecisionLowVol",
    )
    down_flat = namespace["_signal_precision_lowvol"](
        {
            **base_fac,
            "ma10": 158.294,
            "ma20": 158.296,
        },
        range_ctx,
        tag="PrecisionLowVol",
    )

    assert up_lean is not None
    assert down_flat is not None
    assert up_lean["confidence"] > down_flat["confidence"]
    assert up_lean["size_mult"] > down_flat["size_mult"]


def test_vwap_revert_short_blocks_gap_strong_hostile_projection_lane() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_vwap_revert",
    )
    namespace.update(
        {
            "config": SimpleNamespace(MAX_SPREAD_PIPS=2.0),
            "VWAP_REV_RANGE_SCORE": 0.40,
            "VWAP_REV_SPREAD_P25": 0.9,
            "VWAP_REV_ADX_MAX": 22.0,
            "VWAP_REV_BBW_MAX": 0.0014,
            "VWAP_REV_ATR_MIN": 0.7,
            "VWAP_REV_ATR_MAX": 3.2,
            "VWAP_REV_GAP_MIN": 1.2,
            "VWAP_REV_BB_TOUCH_PIPS": 0.9,
            "VWAP_REV_RSI_LONG_MAX": 46.0,
            "VWAP_REV_RSI_SHORT_MIN": 54.0,
            "VWAP_REV_STOCH_LONG_MAX": 0.2,
            "VWAP_REV_STOCH_SHORT_MIN": 0.8,
            "_latest_price": lambda _fac: 158.046,
            "bb_levels": lambda _fac: (158.048, 158.020, 157.992, 0.056, 5.6),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0) / 0.01,
            "_macd_hist_pips": lambda fac: float(fac.get("macd_hist") or 0.0) / 0.01,
            "tick_snapshot": lambda *_args, **_kwargs: ([158.020, 158.034, 158.046], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.82),
            "_reversion_short_flow_guard": lambda **_kwargs: (
                True,
                {
                    "continuation_pressure": 0.41,
                    "max_pressure": 0.44,
                    "setup_quality": 0.54,
                    "reversion_support": 0.47,
                },
            ),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.0,
                {"mode": mode, "score": -0.265},
            ),
        }
    )

    fac_m1 = {
        "adx": 16.2,
        "bbw": 0.00079,
        "atr": 2.72,
        "rsi": 56.2,
        "stoch": 0.92,
        "vgap": 22.87,
        "ema20": 158.022,
        "ema_slope_10": 0.0018,
        "ema_slope_20": 0.0012,
        "macd_hist": 0.0093,
    }
    range_ctx = SimpleNamespace(active=True, score=0.337)

    signal = namespace["_signal_vwap_revert"](fac_m1, range_ctx, tag="VwapRevertS")

    assert signal is None


def test_vwap_revert_prefers_short_up_lean_lane() -> None:
    namespace = _load_worker_functions(
        "_unit_bound",
        "_positive_norm",
        "_plus_di",
        "_minus_di",
        "_reversion_short_flow_guard",
        "_signal_vwap_revert",
    )
    namespace.update(
        {
            "config": SimpleNamespace(MAX_SPREAD_PIPS=2.0),
            "VWAP_REV_RANGE_SCORE": 0.40,
            "VWAP_REV_SPREAD_P25": 0.9,
            "VWAP_REV_ADX_MAX": 22.0,
            "VWAP_REV_BBW_MAX": 0.0014,
            "VWAP_REV_ATR_MIN": 0.7,
            "VWAP_REV_ATR_MAX": 3.2,
            "VWAP_REV_GAP_MIN": 1.2,
            "VWAP_REV_BB_TOUCH_PIPS": 0.9,
            "VWAP_REV_RSI_LONG_MAX": 46.0,
            "VWAP_REV_RSI_SHORT_MIN": 54.0,
            "VWAP_REV_STOCH_LONG_MAX": 0.2,
            "VWAP_REV_STOCH_SHORT_MIN": 0.8,
            "_latest_price": lambda _fac: 158.046,
            "bb_levels": lambda _fac: (158.048, 158.020, 157.992, 0.056, 5.6),
            "spread_ok": lambda **_kwargs: (True, {}),
            "_adx": lambda fac: float(fac.get("adx") or 0.0),
            "_bbw": lambda fac: float(fac.get("bbw") or 0.0),
            "_atr_pips": lambda fac: float(fac.get("atr") or 0.0),
            "_rsi": lambda fac: float(fac.get("rsi") or 50.0),
            "_stoch_rsi": lambda fac: float(fac.get("stoch") or 0.0),
            "_vwap_gap_pips": lambda fac: float(fac.get("vgap") or 0.0),
            "_ema_slope_pips": lambda fac, key: float(fac.get(key) or 0.0) / 0.01,
            "_macd_hist_pips": lambda fac: float(fac.get("macd_hist") or 0.0) / 0.01,
            "tick_snapshot": lambda *_args, **_kwargs: ([158.020, 158.034, 158.046], 6.0),
            "tick_reversal": lambda *_args, **_kwargs: (True, "short", 0.86),
            "_reversion_short_flow_guard": lambda **_kwargs: (
                True,
                {
                    "continuation_pressure": 0.28,
                    "max_pressure": 0.48,
                    "setup_quality": 0.72,
                    "reversion_support": 0.63,
                },
            ),
            "projection_decision": lambda side, mode="range": (
                side == "short",
                1.04,
                {"mode": mode, "score": 0.12},
            ),
        }
    )

    base_fac = {
        "adx": 16.2,
        "bbw": 0.00079,
        "atr": 2.72,
        "rsi": 56.2,
        "stoch": 0.92,
        "vgap": 8.4,
        "ema20": 158.022,
        "ema_slope_10": 0.0006,
        "ema_slope_20": 0.0002,
        "macd_hist": 0.0023,
    }
    range_ctx = SimpleNamespace(active=True, score=0.337)

    up_lean = namespace["_signal_vwap_revert"](
        {
            **base_fac,
            "ma10": 158.035,
            "ma20": 158.028,
        },
        range_ctx,
        tag="VwapRevertS",
    )
    down_flat = namespace["_signal_vwap_revert"](
        {
            **base_fac,
            "ma10": 158.021,
            "ma20": 158.023,
        },
        range_ctx,
        tag="VwapRevertS",
    )

    assert up_lean is not None
    assert down_flat is not None
    assert up_lean["confidence"] > down_flat["confidence"]
    assert up_lean["size_mult"] > down_flat["size_mult"]
