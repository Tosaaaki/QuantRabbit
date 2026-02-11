from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.order_manager import _exit_end_reversal_eval


def _base_policy() -> dict:
    return {
        "enabled": True,
        "score_min": 0.55,
        "min_profit_pips": 0.2,
        "max_factor_age_sec": 120.0,
        "reasons": ["take_profit", "rsi_take", "lock_floor", "candle_*"],
        "rsi_center": 35.0,
        "rsi_band": 12.0,
        "adx_center": 24.0,
        "adx_band": 12.0,
        "gap_ref_pips": 8.0,
        "gap_band_pips": 10.0,
        "vwap_ref": 16.0,
        "slope_ref_pips": 6.0,
    }


def test_end_reversal_triggers_for_short_exhaustion() -> None:
    decision = _exit_end_reversal_eval(
        exit_reason="take_profit",
        strategy_tag="SqueezePulseBreak",
        units_ctx=-5000,
        est_pips=0.7,
        instrument="USD_JPY",
        exit_context={
            "factor_age_m1_sec": 20.0,
            "factors": {
                "M1": {"rsi": 25.0, "adx": 17.0, "ma10": 153.312, "ma20": 153.326, "vwap_gap": 3.5},
                "M5": {"ma10": 153.255, "ma20": 153.338},
            },
        },
        policy=_base_policy(),
    )
    assert decision["triggered"] is True
    assert decision["state"] == "triggered"
    assert float(decision["score"]) >= 0.55


def test_end_reversal_does_not_trigger_in_strong_trend_extension() -> None:
    decision = _exit_end_reversal_eval(
        exit_reason="take_profit",
        strategy_tag="SqueezePulseBreak",
        units_ctx=-5000,
        est_pips=0.8,
        instrument="USD_JPY",
        exit_context={
            "factor_age_m1_sec": 18.0,
            "factors": {
                "M1": {"rsi": 46.0, "adx": 35.0, "ma10": 153.196, "ma20": 153.332, "vwap_gap": 24.0},
                "M5": {"ma10": 153.225, "ma20": 153.349},
            },
        },
        policy=_base_policy(),
    )
    assert decision["triggered"] is False
    assert decision["state"] == "below_threshold"
    assert float(decision["score"]) < 0.55


def test_end_reversal_requires_target_exit_reason() -> None:
    decision = _exit_end_reversal_eval(
        exit_reason="max_adverse",
        strategy_tag="SqueezePulseBreak",
        units_ctx=-3000,
        est_pips=1.0,
        instrument="USD_JPY",
        exit_context={"factor_age_m1_sec": 12.0, "factors": {"M1": {"rsi": 22.0}}},
        policy=_base_policy(),
    )
    assert decision["triggered"] is False
    assert decision["state"] == "reason_mismatch"


def test_end_reversal_blocks_when_factors_are_stale() -> None:
    decision = _exit_end_reversal_eval(
        exit_reason="lock_floor",
        strategy_tag="WickReversalHF",
        units_ctx=-2500,
        est_pips=0.6,
        instrument="USD_JPY",
        exit_context={"factor_age_m1_sec": 240.0, "factors": {"M1": {"rsi": 21.0}}},
        policy=_base_policy(),
    )
    assert decision["triggered"] is False
    assert decision["state"] == "stale_factors"

