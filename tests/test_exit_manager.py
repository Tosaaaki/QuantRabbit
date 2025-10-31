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
