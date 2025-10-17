import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.mean_reversion.bb_rsi import BBRsi


@pytest.fixture(autouse=True)
def no_micro_guard(monkeypatch):
    monkeypatch.setattr(
        "strategies.mean_reversion.bb_rsi.micro_loss_cooldown_active",
        lambda: False,
    )
    monkeypatch.setattr(
        "strategies.mean_reversion.bb_rsi.micro_recent_loss_guard",
        lambda: False,
    )


def _base_fac(price: float, bbw: float, rsi: float, tick_mom: float) -> dict:
    ts = datetime.datetime(2024, 1, 1, 4, 45, tzinfo=datetime.timezone.utc).isoformat()
    return {
        "rsi": rsi,
        "bbw": bbw,
        "ma20": 150.0,
        "close": price,
        "atr": 0.12,
        "adx": 20.0,
        "ma_slope": 0.02,
        "tick_momentum_5": tick_mom,
        "tick_velocity_30s": tick_mom,
        "tick_vwap_delta": tick_mom / 2,
        "tick_range_30s": 1.2,
        "candles": [{"timestamp": ts}],
    }


def test_bb_rsi_generates_long_signal_when_conditions_met():
    fac = _base_fac(price=143.5, bbw=0.08, rsi=42.0, tick_mom=-0.8)
    fac_h4 = {"ma10": 151.0, "ma20": 149.5, "adx": 28.0}

    signal = BBRsi.check(fac, fac_h4)

    assert signal is not None
    assert signal["action"] == "buy"
    assert signal["sl_pips"] > 0
    assert signal["tp_pips"] > 0


def test_bb_rsi_blocks_long_when_h4_trend_down():
    fac = _base_fac(price=143.5, bbw=0.08, rsi=42.0, tick_mom=-0.8)
    fac_h4 = {"ma10": 148.0, "ma20": 150.5, "adx": 28.0}

    signal = BBRsi.check(fac, fac_h4)

    assert signal is None


def test_bb_rsi_blocks_when_momentum_strong():
    fac = _base_fac(price=143.5, bbw=0.08, rsi=42.0, tick_mom=-3.0)
    fac_h4 = {"ma10": 151.0, "ma20": 149.5, "adx": 28.0}

    signal = BBRsi.check(fac, fac_h4)

    assert signal is None
