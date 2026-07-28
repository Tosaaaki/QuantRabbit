from __future__ import annotations

import sqlite3
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
