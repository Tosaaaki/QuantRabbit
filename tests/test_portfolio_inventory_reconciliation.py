from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.portfolio_inventory_reconciliation import (
    InventoryReconciliationError,
    _canonical_sha,
    build_portfolio_inventory,
)

NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def _broker(pid: str, pair: str, side: str, frac: float) -> dict:
    return {
        "position_id": pid,
        "pair": pair,
        "side": side,
        "nav_exposure_fraction": frac,
    }


def _ledger(pid: str, lane: str = "S5_SURVIVOR", thesis: str = "STILL_VALID") -> dict:
    return {"position_id": pid, "lane_id": lane, "thesis_state": thesis}


def _build(broker, ledger, manual=()):
    return build_portfolio_inventory(
        broker_positions=broker,
        ledger_open_positions=ledger,
        manual_no_touch_ids=list(manual),
        nav_account_currency=1_000_000.0,
        as_of_utc=NOW,
        broker_snapshot_sha256="a" * 64,
        ledger_tip_sha256="b" * 64,
    )


def test_three_way_match_reconciles_and_excludes_manual_exposure() -> None:
    inventory = _build(
        [
            _broker("p1", "EUR_USD", "LONG", 0.2),
            _broker("p2", "USD_JPY", "SHORT", 0.2),
            _broker("m1", "GBP_JPY", "LONG", 0.5),
        ],
        [_ledger("p1"), _ledger("p2", thesis="BROKEN")],
        manual=("m1",),
    )

    assert inventory["status"] == "RECONCILED"
    assert inventory["new_entries_admissible"] is True
    assert inventory["manual_no_touch_count"] == 1
    # Manual GBP_JPY must not contaminate trader exposure: USD stacks to -0.4.
    assert inventory["net_currency_exposure_excl_manual"]["USD"] == pytest.approx(
        -0.4
    )
    assert "GBP" not in inventory["net_currency_exposure_excl_manual"]
    assert inventory["positions_requiring_thesis_review"] == ["p2"]
    body = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    assert inventory["inventory_sha256"] == _canonical_sha(body)


def test_untracked_and_orphaned_positions_fail_closed() -> None:
    untracked = _build(
        [_broker("p1", "EUR_USD", "LONG", 0.2), _broker("ghost", "AUD_USD", "LONG", 0.1)],
        [_ledger("p1")],
    )
    assert untracked["status"] == "UNRECONCILED_FAIL_CLOSED"
    assert untracked["new_entries_admissible"] is False
    assert untracked["untracked_at_broker"] == ["ghost"]

    orphaned = _build([_broker("p1", "EUR_USD", "LONG", 0.2)], [_ledger("p1"), _ledger("gone")])
    assert orphaned["status"] == "UNRECONCILED_FAIL_CLOSED"
    assert orphaned["orphaned_in_ledger"] == ["gone"]

    with pytest.raises(InventoryReconciliationError, match="thesis state"):
        _build([_broker("p1", "EUR_USD", "LONG", 0.2)], [_ledger("p1", thesis="OK")])
    with pytest.raises(InventoryReconciliationError, match="owning lane"):
        _build(
            [_broker("p1", "EUR_USD", "LONG", 0.2)],
            [{"position_id": "p1", "lane_id": " ", "thesis_state": "STILL_VALID"}],
        )
