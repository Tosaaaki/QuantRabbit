import pathlib
import sys
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.exit_manager import ExitManager


def build_open_trade(side: str, seconds_ago: float, units: int = 1000) -> dict:
    opened_at = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return {
        "side": side,
        "units": units,
        "open_time": opened_at.isoformat(),
    }


def test_scalp_long_keeps_position_when_price_above_ema():
    manager = ExitManager()
    open_positions = {
        "scalp": {
            "long_units": 1000,
            "short_units": 0,
            "long_avg_price": 150.000,
            "open_trades": [build_open_trade("long", seconds_ago=5.0)],
        }
    }
    fac_m1 = {
        "close": 150.030,
        "ema20": 150.020,
        "atr_pips": 1.2,
        "candles": [],
    }
    fac_h4 = {"atr_pips": 8.0}

    decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_m1,
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
    )

    assert decisions == []


def test_scalp_long_exits_when_price_drops_below_ema():
    manager = ExitManager()
    open_positions = {
        "scalp": {
            "long_units": 1000,
            "short_units": 0,
            "long_avg_price": 150.030,
            "open_trades": [build_open_trade("long", seconds_ago=12.0)],
        }
    }
    fac_m1 = {
        "close": 150.010,
        "ema20": 150.020,
        "atr_pips": 1.2,
        "candles": [],
    }
    fac_h4 = {"atr_pips": 8.0}

    decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_m1,
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.pocket == "scalp"
    assert decision.reason == "scalp_momentum_flip"
    assert decision.units < 0


def test_macro_profit_lock_triggers_for_h1momentum():
    manager = ExitManager()
    now = datetime.now(timezone.utc)
    open_positions = {
        "macro": {
            "long_units": 1000,
            "short_units": 0,
            "avg_price": 150.000,
            "long_avg_price": 150.000,
            "open_trades": [
                {
                    "side": "long",
                    "units": 1000,
                    "price": 150.000,
                    "open_time": (now - timedelta(minutes=60)).isoformat(),
                    "entry_thesis": {
                        "strategy": "H1Momentum",
                        "note": {"insurance_sl": 30.0},
                    },
                }
            ],
        }
    }
    fac_m1 = {
        "close": 150.150,
        "ema20": 150.120,
        "atr_pips": 2.5,
        "rsi": 55.0,
        "candles": [
            {
                "timestamp": (now - timedelta(minutes=30)).isoformat(),
                "high": 150.400,
                "low": 150.020,
            },
            {
                "timestamp": now.isoformat(),
                "high": 150.180,
                "low": 150.060,
            },
        ],
    }
    fac_h4 = {
        "ma10": 150.040,
        "ma20": 150.000,
        "adx": 28.0,
        "ema20": 150.080,
        "atr_pips": 18.0,
    }

    decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_m1,
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
        now=now,
    )

    assert decisions, "Expected profit-lock exit decision."
    decision = decisions[0]
    assert decision.reason == "macro_profit_lock"
    assert decision.units < 0


def test_micro_low_vol_event_budget_exit():
    manager = ExitManager()
    base_time = datetime.now(timezone.utc)
    open_positions = {
        "micro": {
            "long_units": 1000,
            "short_units": 0,
            "avg_price": 150.000,
            "long_avg_price": 150.000,
            "open_trades": [
                {
                    "side": "long",
                    "units": 1000,
                    "price": 150.000,
                    "open_time": (base_time - timedelta(seconds=4.2)).isoformat(),
                }
            ],
        }
    }
    fac_m1 = {
        "close": 149.997,
        "ema20": 150.000,
        "atr_pips": 2.1,
        "vol_5m": 0.83,
        "candles": [
            {
                "timestamp": (base_time - timedelta(seconds=3.0)).isoformat(),
                "high": 150.001,
                "low": 149.996,
            },
            {
                "timestamp": (base_time - timedelta(seconds=1.0)).isoformat(),
                "high": 149.999,
                "low": 149.995,
            },
        ],
    }
    fac_h4 = {"atr_pips": 8.0}
    low_vol_profile = {"low_vol": True, "low_vol_like": True, "score": 0.74, "atr": 2.1}

    decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_m1,
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
        now=base_time,
        low_vol_profile=low_vol_profile,
        low_vol_quiet=True,
        news_status="quiet",
    )

    assert decisions, "Expected event budget exit decision"
    reason = decisions[0].reason
    assert reason == "low_vol_event_budget"


def test_micro_low_vol_hazard_exit_requires_two_observations():
    manager = ExitManager()
    base_time = datetime.now(timezone.utc)
    open_trade = {
        "side": "long",
        "units": 1000,
        "price": 150.000,
        "open_time": (base_time - timedelta(seconds=3.0)).isoformat(),
    }
    open_positions = {
        "micro": {
            "long_units": 1000,
            "short_units": 0,
            "avg_price": 150.000,
            "long_avg_price": 150.000,
            "open_trades": [open_trade],
        }
    }
    low_vol_profile = {"low_vol": True, "low_vol_like": True, "score": 0.7, "atr": 2.0}

    def fac_snapshot(now_time: datetime, close_px: float) -> dict:
        return {
            "close": close_px,
            "ema20": 150.002,
            "atr_pips": 2.0,
            "vol_5m": 0.82,
            "candles": [
                {
                    "timestamp": (now_time - timedelta(seconds=3.0)).isoformat(),
                    "high": 150.0056,
                    "low": 149.996,
                },
                {
                    "timestamp": (now_time - timedelta(seconds=0.5)).isoformat(),
                    "high": 150.004,
                    "low": 149.994,
                },
            ],
        }

    fac_h4 = {"atr_pips": 8.0}

    first_decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_snapshot(base_time, 149.995),
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
        now=base_time,
        low_vol_profile=low_vol_profile,
        low_vol_quiet=True,
        news_status="quiet",
    )
    assert first_decisions == []

    later_time = base_time + timedelta(seconds=0.8)
    second_decisions = manager.plan_closures(
        open_positions=open_positions,
        signals=[],
        fac_m1=fac_snapshot(later_time, 149.994),
        fac_h4=fac_h4,
        event_soon=False,
        range_mode=False,
        now=later_time,
        low_vol_profile=low_vol_profile,
        low_vol_quiet=True,
        news_status="quiet",
    )

    assert second_decisions, "Hazard exit should trigger on second observation"
    assert second_decisions[0].reason == "low_vol_hazard_exit"
