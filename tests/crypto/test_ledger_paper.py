from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_rabbit.crypto.ledger import CryptoLedger, LedgerIntegrityError
from quant_rabbit.crypto.paper import PaperEngine


def test_ledger_is_append_only_deduplicated_and_restartable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.db"
    ledger = CryptoLedger(path)
    event_id, created = ledger.append(
        "DECISION", "btc_jpy", {"candidate": False}, dedupe_key="decision:1"
    )
    same_id, duplicate_created = ledger.append(
        "DECISION", "btc_jpy", {"candidate": False}, dedupe_key="decision:1"
    )
    assert event_id == same_id
    assert created is True
    assert duplicate_created is False
    assert CryptoLedger(path).verify()["event_count"] == 1
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM crypto_events")


def test_ledger_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    ledger = CryptoLedger(path)
    ledger.append("DECISION", "btc_jpy", {"ok": True}, dedupe_key="one")
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER crypto_events_no_update")
        conn.execute(
            "UPDATE crypto_events SET payload_json='{}' WHERE sequence=1"
        )
    with pytest.raises(LedgerIntegrityError):
        CryptoLedger(path)


def test_incremental_ledger_verify_anchors_verified_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.db"
    ledger = CryptoLedger(path)
    ledger.append("DECISION", "btc_jpy", {"n": 1}, dedupe_key="one")
    checkpoint = ledger.verify()
    ledger.append("DECISION", "btc_jpy", {"n": 2}, dedupe_key="two")
    reopened = CryptoLedger(path, verify_on_open=False)
    result = reopened.verify_incremental(
        event_count=int(checkpoint["event_count"]),
        head_hash=str(checkpoint["head_hash"]),
    )
    assert result == ledger.verify()


def test_incremental_ledger_verify_rejects_wrong_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.db"
    ledger = CryptoLedger(path)
    ledger.append("DECISION", "btc_jpy", {"n": 1}, dedupe_key="one")
    reopened = CryptoLedger(path, verify_on_open=False)
    with pytest.raises(LedgerIntegrityError):
        reopened.verify_incremental(event_count=1, head_hash="f" * 64)


def test_ledger_reads_bounded_utc_window(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    ledger = CryptoLedger(path)
    ledger.append(
        "DECISION",
        "btc_jpy",
        {"n": 1},
        dedupe_key="one",
        created_at=datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )
    ledger.append(
        "DECISION",
        "btc_jpy",
        {"n": 2},
        dedupe_key="two",
        created_at=datetime(2026, 7, 29, 1, 0, tzinfo=timezone.utc),
    )
    rows = list(
        ledger.events_between(
            "2026-07-29T00:30:00+00:00",
            "2026-07-29T01:30:00+00:00",
        )
    )
    assert [row["payload"]["n"] for row in rows] == [2]


def test_paper_partial_fill_restart_and_duplicate_are_deterministic(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "paper.db")
    intent = {
        "intent_id": "intent-1",
        "pair": "btc_jpy",
        "side": "BUY",
        "amount": "1",
        "order_style": "PAPER_MAKER_LIMIT",
        "regime": "TREND_UP",
        "authority": "NONE",
        "live_permission": False,
    }
    depth = {"bids": [["100", "10"]], "asks": [["101", "10"]]}
    engine = PaperEngine(ledger, initial_cash_jpy=Decimal("1000"))
    first = engine.process_intent(
        intent,
        depth=depth,
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.0012"),
    )
    assert first["status"] == "PARTIALLY_FILLED"
    assert first["filled_amount"] == "0.25"
    restarted = PaperEngine(CryptoLedger(tmp_path / "paper.db"))
    before = restarted.state.as_dict()
    duplicate = restarted.process_intent(
        intent,
        depth=depth,
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.0012"),
    )
    assert duplicate == first
    assert restarted.state.as_dict() == before
    metrics = restarted.mark_to_market({"btc_jpy": Decimal("102")})
    assert metrics["trade_count"] == 1
    assert metrics["fill_count"] == 1
    assert metrics["completed_trade_count"] == 0
    assert metrics["turnover_jpy"] == "25.00"
    assert metrics["trade_count_semantics"].startswith("DEPRECATED_")
    assert metrics["partial_fill_ratio"] == 1.0
    assert metrics["maker_fill_count"] == 1
    assert "TREND_UP" in metrics["by_regime_pnl_jpy"]


def test_paper_rejects_any_live_authority(tmp_path: Path) -> None:
    engine = PaperEngine(CryptoLedger(tmp_path / "paper.db"))
    with pytest.raises(RuntimeError):
        engine.process_intent(
            {
                "intent_id": "unsafe",
                "pair": "btc_jpy",
                "side": "BUY",
                "amount": "1",
                "order_style": "PAPER_TAKER",
                "authority": "ORDER",
                "live_permission": True,
            },
            depth={"asks": [["100", "1"]], "bids": [["99", "1"]]},
            maker_fee_rate=Decimal("0"),
            taker_fee_rate=Decimal("0"),
        )


def test_paper_sell_closes_virtual_position_without_shorting(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "paper.db")
    engine = PaperEngine(
        ledger,
        initial_cash_jpy=Decimal("1000"),
        maker_fill_fraction=Decimal("1"),
    )
    depth = {"bids": [["100", "100"]], "asks": [["101", "100"]]}
    buy = engine.process_intent(
        {
            "intent_id": "buy",
            "pair": "btc_jpy",
            "side": "BUY",
            "amount": "1",
            "order_style": "PAPER_TAKER",
            "regime": "FAST",
            "authority": "NONE",
            "live_permission": False,
        },
        depth=depth,
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    assert buy["filled_amount"] == "1"
    sell = engine.process_intent(
        {
            "intent_id": "sell",
            "pair": "btc_jpy",
            "side": "SELL",
            "amount": "2",
            "order_style": "PAPER_TAKER",
            "regime": "FAST",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"bids": [["102", "100"]], "asks": [["103", "100"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    assert sell["filled_amount"] == "1"
    assert engine.state.positions["btc_jpy"] == 0
    assert engine.state.round_trips == 1
    assert engine.state.realized_pnl_by_pair["btc_jpy"] == Decimal("1")
    assert engine.mark_to_market({"btc_jpy": Decimal("102")})[
        "profit_factor"
    ] is None


def test_margin_paper_opens_and_closes_short_with_interest(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "margin.db")
    engine = PaperEngine(
        ledger,
        initial_cash_jpy=Decimal("1000"),
        maker_fill_fraction=Decimal("1"),
        allow_short=True,
        max_leverage=Decimal("2"),
    )
    opened = engine.process_intent(
        {
            "intent_id": "short-open",
            "pair": "btc_jpy",
            "side": "SELL",
            "position_effect": "OPEN",
            "amount": "10",
            "order_style": "PAPER_TAKER",
            "regime": "FAST_SHORT",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"bids": [["100", "100"]], "asks": [["101", "100"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    assert opened["filled_amount"] == "10"
    assert engine.state.positions["btc_jpy"] == Decimal("-10")
    interest = engine.accrue_interest(
        {"btc_jpy": (Decimal("0.0004"), Decimal("0.0004"))},
        elapsed_sec=86400,
        cause_id="day-1",
    )
    assert interest == Decimal("0.4")
    closed = engine.process_intent(
        {
            "intent_id": "short-close",
            "pair": "btc_jpy",
            "side": "BUY",
            "position_effect": "CLOSE",
            "amount": "10",
            "order_style": "PAPER_TAKER",
            "regime": "FAST_SHORT",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"bids": [["89", "100"]], "asks": [["90", "100"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    assert closed["filled_amount"] == "10"
    assert engine.state.positions["btc_jpy"] == 0
    metrics = engine.mark_to_market(
        {"btc_jpy": Decimal("89")},
        {"btc_jpy": Decimal("90")},
    )
    assert Decimal(metrics["net_pnl_jpy"]) == Decimal("99.6")
    assert metrics["round_trip_count"] == 1
    assert metrics["short_position_count"] == 0


def test_margin_paper_caps_opening_at_two_x_and_models_losscut(
    tmp_path: Path,
) -> None:
    engine = PaperEngine(
        CryptoLedger(tmp_path / "margin.db"),
        initial_cash_jpy=Decimal("1000"),
        allow_short=True,
        max_leverage=Decimal("2"),
    )
    fill = engine.process_intent(
        {
            "intent_id": "leveraged-short",
            "pair": "btc_jpy",
            "side": "SELL",
            "position_effect": "OPEN",
            "amount": "100",
            "order_style": "PAPER_TAKER",
            "regime": "FAST_SHORT",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"bids": [["100", "100"]], "asks": [["101", "100"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    assert fill["filled_amount"] == "20"
    snapshot = engine.margin_snapshot(
        {"btc_jpy": Decimal("119")},
        {"btc_jpy": Decimal("120")},
    )
    assert snapshot["margin_ratio"] == Decimal("0.25")
    assert snapshot["status"] == "MODELED_LOSSCUT"
