import pathlib
import sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.micro_lowvol.momentum_pulse import MomentumPulse
from strategies.micro_lowvol.vol_compression_break import VolCompressionBreak
from strategies.micro_lowvol.bb_rsi_fast import BBRsiFast
from strategies.micro_lowvol.micro_vwap_revert import MicroVWAPRevert


def _ts(offset_seconds: float = 0.0) -> str:
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stamp = base + timedelta(seconds=offset_seconds)
    return stamp.isoformat()


def test_momentum_pulse_generates_long_signal():
    candles = [
        {"timestamp": _ts(-90), "open": 150.000, "close": 149.996, "high": 150.004, "low": 149.990},
        {"timestamp": _ts(-60), "open": 149.996, "close": 150.001, "high": 150.005, "low": 149.994},
        {"timestamp": _ts(-30), "open": 150.001, "close": 150.007, "high": 150.010, "low": 149.999},
        {"timestamp": _ts(-5), "open": 150.007, "close": 150.012, "high": 150.015, "low": 150.004},
    ]
    fac = {
        "close": 150.012,
        "ema12": 150.0115,
        "ema20": 150.0075,
        "ma10": 150.0105,
        "adx": 22.0,
        "vol_5m": 0.92,
        "bbw": 0.19,
        "atr_pips": 2.0,
        "candles": candles,
    }
    signal = MomentumPulse.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert 1.2 <= signal["sl_pips"] <= 2.7
    assert signal["tag"].startswith("MomentumPulse")


def test_vol_compression_break_detects_short_breakout():
    candles = [
        {"timestamp": _ts(-80), "open": 150.050, "close": 150.046, "high": 150.051, "low": 150.045},
        {"timestamp": _ts(-60), "open": 150.046, "close": 150.044, "high": 150.047, "low": 150.042},
        {"timestamp": _ts(-40), "open": 150.044, "close": 150.040, "high": 150.046, "low": 150.038},
        {"timestamp": _ts(-20), "open": 150.040, "close": 150.036, "high": 150.041, "low": 150.035},
        {"timestamp": _ts(-5), "open": 150.036, "close": 150.028, "high": 150.036, "low": 150.025},
    ]
    fac = {
        "close": 150.028,
        "ema12": 150.034,
        "ema20": 150.035,
        "bbw": 0.16,
        "vol_5m": 0.88,
        "adx": 18.0,
        "atr_pips": 2.4,
        "candles": candles,
    }
    signal = VolCompressionBreak.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert 1.2 <= signal["sl_pips"] <= 2.4


def test_bb_rsi_fast_prefers_long_in_range():
    fac = {
        "close": 149.972,
        "ma20": 150.000,
        "ema20": 150.000,
        "rsi": 36.5,
        "bbw": 0.0004,
        "adx": 19.0,
        "vol_5m": 0.88,
        "atr_pips": 2.4,
        "candles": [
            {"timestamp": _ts(-60), "open": 150.001, "close": 149.990},
            {"timestamp": _ts(-30), "open": 149.990, "close": 149.978},
            {"timestamp": _ts(-5), "open": 149.978, "close": 149.972},
        ],
    }
    signal = BBRsiFast.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["sl_pips"] >= signal["tp_pips"]


def test_micro_vwap_revert_flags_short_bias():
    typical_prices = [
        {"timestamp": _ts(-70), "open": 150.010, "high": 150.015, "low": 150.008, "close": 150.012},
        {"timestamp": _ts(-50), "open": 150.008, "high": 150.012, "low": 150.004, "close": 150.007},
        {"timestamp": _ts(-30), "open": 150.007, "high": 150.010, "low": 150.003, "close": 150.006},
        {"timestamp": _ts(-10), "open": 150.006, "high": 150.009, "low": 150.002, "close": 150.005},
    ]
    fac = {
        "close": 150.020,
        "ema20": 150.010,
        "ma10": 150.011,
        "vol_5m": 0.85,
        "bbw": 0.18,
        "adx": 19.5,
        "atr_pips": 2.0,
        "candles": typical_prices,
    }
    signal = MicroVWAPRevert.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert 1.1 <= signal["sl_pips"] <= 2.4
