from __future__ import annotations

from array import array
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.adaptive_exact_s5_profit_engine import ExactS5Series
from quant_rabbit.range_rail_shadow import RangeRailError, resolve_range_rail_rotation

UTC = timezone.utc
DECISION = datetime(2026, 5, 12, 9, 0, 0, tzinfo=UTC)
SHA = "a" * 64
SPREAD = 0.0002


def _series(mids: list[float]) -> ExactS5Series:
    epochs = array("q")
    bids = array("d")
    asks = array("d")
    base = int(DECISION.timestamp())
    for i, mid in enumerate(mids):
        epochs.append(base + 5 * i)
        bids.append(mid - SPREAD / 2)
        asks.append(mid + SPREAD / 2)
    return ExactS5Series(
        points=(), minute_epochs=(), s5_epochs=epochs, bid_opens=bids, ask_opens=asks
    )


def _resolve(mids, **over):
    kw = {
        "side": "SHORT",
        "rail_price": 1.1020,
        "mid_price": 1.1000,
        "stop_price": 1.1035,
        "rail_provenance_sha256": SHA,
        "decision_utc": DECISION,
        "entry_ttl_seconds": 90,
        "horizon_seconds": 3600,
        "pip_factor": 10_000.0,
    }
    kw.update(over)
    return resolve_range_rail_rotation(_series(mids), **kw)


def test_short_rail_reverts_to_mid_for_a_win() -> None:
    # Price rises into the 1.1020 rail (fills the passive sell), then reverts
    # down through mid 1.1000 -> buy back at mid for ~+20 pips.
    mids = [1.1015, 1.1021, 1.1010, 1.0999, 1.0995]
    result = _resolve(mids)
    assert result["filled"] is True
    assert result["exit_reason"] == "MEAN_REVERT_TO_MID"
    assert result["realized_pips"] == pytest.approx(20.0)


def test_short_rail_breakout_hits_stop() -> None:
    # Price fills the rail then keeps rising through the 1.1035 stop.
    mids = [1.1021, 1.1030, 1.1040, 1.1045]
    result = _resolve(mids)
    assert result["filled"] is True
    assert result["exit_reason"] == "STOP_LOSS"
    assert result["realized_pips"] < 0.0


def test_no_fill_when_price_never_reaches_rail() -> None:
    mids = [1.1000, 1.1005, 1.1008, 1.1010]  # never rises to 1.1020 within TTL
    result = _resolve(mids)
    assert result["filled"] is False
    assert result["status"] == "NO_FILL_WITHIN_TTL"
    assert result["result_available"] is False


def test_long_rail_reverts_up_to_mid() -> None:
    # LONG buys the lower rail as price falls into it, reverts up to mid.
    mids = [1.0985, 1.0979, 1.0990, 1.1001, 1.1005]
    result = _resolve(
        mids,
        side="LONG",
        rail_price=1.0980,
        mid_price=1.1000,
        stop_price=1.0965,
    )
    assert result["exit_reason"] == "MEAN_REVERT_TO_MID"
    assert result["realized_pips"] == pytest.approx(20.0)


def test_geometry_and_provenance_fail_closed() -> None:
    with pytest.raises(RangeRailError, match="stop > rail > mid"):
        _resolve([1.1021], stop_price=1.1010)  # stop below rail for a SHORT
    with pytest.raises(RangeRailError, match="sha256"):
        _resolve([1.1021], rail_provenance_sha256="nope")
    with pytest.raises(RangeRailError, match="horizon"):
        _resolve([1.1021], horizon_seconds=50)
