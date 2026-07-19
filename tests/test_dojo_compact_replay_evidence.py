from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_rabbit.dojo_compact_replay_evidence import (
    MANIFEST_CONTRACT,
    CompactReplayEvidenceError,
    CompactReplayEvidenceWriter,
    verify_compact_replay_evidence,
)


BINDINGS = {
    "source_sha256": "1" * 64,
    "plan_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "cost_sha256": "4" * 64,
}


def _writer(tmp_path: Path, name: str = "segment") -> CompactReplayEvidenceWriter:
    return CompactReplayEvidenceWriter.create(
        tmp_path / name,
        evidence_id=f"evidence-{name}",
        segment_id=f"segment-{name}",
        replay_start_utc="2025-06-01T00:00:00Z",
        replay_end_utc="2025-06-02T00:00:00Z",
        initial_balance_jpy=200_000,
        bindings=BINDINGS,
    )


def _append_round_trip(writer: CompactReplayEvidenceWriter) -> None:
    writer.append(
        "BOT",
        event_id="event-bot-submit",
        event_at_utc="2025-06-01T00:01:00Z",
        payload={
            "bot_id": "spike-fade-v1",
            "decision_id": "decision-1",
            "pair": "USD_JPY",
            "decision": "SUBMIT_ORDER",
            "reason_code": "SPIKE_FADE_SIGNAL",
            "signal_sha256": "5" * 64,
            "related_order_or_trade_id": "order-1",
        },
    )
    writer.append(
        "ORDER",
        event_id="event-order-submit",
        event_at_utc="2025-06-01T00:01:00Z",
        payload={
            "order_id": "order-1",
            "pair": "USD_JPY",
            "side": "LONG",
            "order_type": "MARKET",
            "status": "SUBMITTED",
            "units": 1000,
            "requested_price": None,
            "stop_loss_price": 149.5,
            "take_profit_price": 150.5,
        },
    )
    writer.append(
        "FILL",
        event_id="event-fill",
        event_at_utc="2025-06-01T00:01:01Z",
        payload={
            "fill_id": "fill-1",
            "order_id": "order-1",
            "trade_id": "trade-1",
            "pair": "USD_JPY",
            "side": "LONG",
            "units": 1000,
            "fill_price": 150.0,
            "spread_pips": 0.2,
            "slippage_pips": 0.3,
            "fee_jpy": 30.0,
        },
    )
    writer.append(
        "MARGIN",
        event_id="event-margin-open",
        event_at_utc="2025-06-01T00:01:02Z",
        payload={
            "balance_jpy": 199_970.0,
            "equity_jpy": 199_900.0,
            "used_margin_jpy": 30_000.0,
            "free_margin_jpy": 169_900.0,
        },
    )
    writer.append(
        "CHECKPOINT",
        event_id="event-checkpoint-open",
        event_at_utc="2025-06-01T00:02:00Z",
        payload={
            "checkpoint_id": "checkpoint-1",
            "source_cursor_sha256": "6" * 64,
            "state_sha256": "7" * 64,
            "open_trade_count": 1,
            "pending_order_count": 0,
            "balance_jpy": 199_970.0,
            "equity_jpy": 200_100.0,
        },
    )
    writer.append(
        "BOT",
        event_id="event-bot-close",
        event_at_utc="2025-06-01T01:01:00Z",
        payload={
            "bot_id": "spike-fade-v1",
            "decision_id": "decision-2",
            "pair": "USD_JPY",
            "decision": "CLOSE_TRADE",
            "reason_code": "TIME_EXIT",
            "signal_sha256": "8" * 64,
            "related_order_or_trade_id": "trade-1",
        },
    )
    writer.append(
        "EXIT",
        event_id="event-exit",
        event_at_utc="2025-06-01T01:01:01Z",
        payload={
            "exit_id": "exit-1",
            "trade_id": "trade-1",
            "pair": "USD_JPY",
            "reason": "TIME",
            "units": 1000,
            "exit_price": 150.2,
            "quote_to_jpy_rate": 1.0,
            "realized_pnl_jpy": 200.0,
            "financing_jpy": 0.0,
        },
    )
    writer.append(
        "CHECKPOINT",
        event_id="event-checkpoint-flat",
        event_at_utc="2025-06-01T01:02:00Z",
        payload={
            "checkpoint_id": "checkpoint-2",
            "source_cursor_sha256": "9" * 64,
            "state_sha256": "a" * 64,
            "open_trade_count": 0,
            "pending_order_count": 0,
            "balance_jpy": 200_170.0,
            "equity_jpy": 200_170.0,
        },
    )
    writer.stop(
        event_id="event-stop",
        stopped_at_utc="2025-06-02T00:00:00Z",
        status="COMPLETED",
        reason_code="WINDOW_COMPLETE",
        source_cursor_sha256="b" * 64,
        terminal_balance_jpy=200_170.0,
        terminal_equity_jpy=200_170.0,
        open_trade_count=0,
        pending_order_count=0,
    )


def test_round_trip_final_manifest_binds_event_only_chain(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _append_round_trip(writer)
    manifest = writer.finalize(finalized_at_utc="2025-06-02T00:00:01Z")
    writer.close()

    verified = verify_compact_replay_evidence(tmp_path / "segment")
    assert verified.finalized is True
    assert verified.event_count == 10
    assert verified.bindings == BINDINGS
    assert verified.event_type_counts == {
        "SEGMENT_START": 1,
        "BOT": 2,
        "ORDER": 1,
        "FILL": 1,
        "EXIT": 1,
        "MARGIN": 1,
        "CHECKPOINT": 2,
        "SEGMENT_STOP": 1,
    }
    assert manifest["contract"] == MANIFEST_CONTRACT
    assert manifest["events_sha256"] == verified.events_sha256
    assert manifest["live_permission"] is False
    assert manifest["order_authority"] == "NONE"
    assert manifest["broker_mutation_allowed"] is False


def test_duplicate_unknown_nan_and_invalid_lifecycle_fail_before_append(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    before = (tmp_path / "segment" / "events.jsonl").read_bytes()
    payload = {
        "balance_jpy": 200_000.0,
        "equity_jpy": 200_000.0,
        "used_margin_jpy": 0.0,
        "free_margin_jpy": 200_000.0,
    }
    writer.append(
        "MARGIN",
        event_id="margin-1",
        event_at_utc="2025-06-01T00:01:00Z",
        payload=payload,
    )
    after_valid = (tmp_path / "segment" / "events.jsonl").read_bytes()
    assert after_valid != before

    with pytest.raises(CompactReplayEvidenceError, match="duplicate event_id"):
        writer.append(
            "MARGIN",
            event_id="margin-1",
            event_at_utc="2025-06-01T00:02:00Z",
            payload=payload,
        )
    with pytest.raises(CompactReplayEvidenceError, match="unknown event_type"):
        writer.append(
            "CANDLE",
            event_id="bad-candle",
            event_at_utc="2025-06-01T00:02:00Z",
            payload={},
        )
    with pytest.raises(CompactReplayEvidenceError, match=r"unknown=\['extra'\]"):
        writer.append(
            "MARGIN",
            event_id="bad-extra",
            event_at_utc="2025-06-01T00:02:00Z",
            payload={**payload, "extra": 1},
        )
    with pytest.raises(CompactReplayEvidenceError, match="finite"):
        writer.append(
            "MARGIN",
            event_id="bad-nan",
            event_at_utc="2025-06-01T00:02:00Z",
            payload={**payload, "equity_jpy": float("nan")},
        )
    assert (tmp_path / "segment" / "events.jsonl").read_bytes() == after_valid
    writer.close()


def test_fill_without_pending_order_and_checkpoint_count_fail_closed(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(CompactReplayEvidenceError, match="pending order"):
        writer.append(
            "FILL",
            event_id="orphan-fill",
            event_at_utc="2025-06-01T00:01:00Z",
            payload={
                "fill_id": "fill-x",
                "order_id": "missing-order",
                "trade_id": "trade-x",
                "pair": "EUR_USD",
                "side": "SHORT",
                "units": 1000,
                "fill_price": 1.1,
                "spread_pips": 0.2,
                "slippage_pips": 0.3,
                "fee_jpy": 1.0,
            },
        )
    with pytest.raises(CompactReplayEvidenceError, match="open-trade count"):
        writer.append(
            "CHECKPOINT",
            event_id="bad-checkpoint",
            event_at_utc="2025-06-01T00:01:00Z",
            payload={
                "checkpoint_id": "checkpoint-x",
                "source_cursor_sha256": "c" * 64,
                "state_sha256": "d" * 64,
                "open_trade_count": 1,
                "pending_order_count": 0,
                "balance_jpy": 200_000.0,
                "equity_jpy": 200_000.0,
            },
        )
    writer.close()


def test_tamper_noncanonical_and_unknown_path_are_rejected(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.close()
    ledger = tmp_path / "segment" / "events.jsonl"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    ledger.write_text(json.dumps(event, separators=(", ", ": ")) + "\n", encoding="utf-8")
    with pytest.raises(CompactReplayEvidenceError, match="not canonical"):
        verify_compact_replay_evidence(tmp_path / "segment")

    other = _writer(tmp_path, "unknown")
    other.close()
    (tmp_path / "unknown" / "surprise.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CompactReplayEvidenceError, match="unknown evidence path"):
        verify_compact_replay_evidence(tmp_path / "unknown")


def test_symlink_paths_and_binding_schema_fail_closed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(CompactReplayEvidenceError, match="symlink"):
        CompactReplayEvidenceWriter.create(
            link / "segment",
            evidence_id="unsafe",
            segment_id="unsafe",
            replay_start_utc="2025-06-01T00:00:00Z",
            replay_end_utc="2025-06-02T00:00:00Z",
            initial_balance_jpy=200_000,
            bindings=BINDINGS,
        )
    with pytest.raises(CompactReplayEvidenceError, match="schema mismatch"):
        CompactReplayEvidenceWriter.create(
            tmp_path / "bad-bindings",
            evidence_id="bad-bindings",
            segment_id="bad-bindings",
            replay_start_utc="2025-06-01T00:00:00Z",
            replay_end_utc="2025-06-02T00:00:00Z",
            initial_balance_jpy=200_000,
            bindings={**BINDINGS, "unknown_sha256": "e" * 64},
        )


def test_manifest_blocks_later_append_and_manifest_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.stop(
        event_id="stop",
        stopped_at_utc="2025-06-02T00:00:00Z",
        status="COMPLETED",
        reason_code="WINDOW_COMPLETE",
        source_cursor_sha256="f" * 64,
        terminal_balance_jpy=200_000.0,
        terminal_equity_jpy=200_000.0,
        open_trade_count=0,
        pending_order_count=0,
    )
    writer.finalize(finalized_at_utc="2025-06-02T00:00:01Z")
    with pytest.raises(CompactReplayEvidenceError, match="cannot be appended|follow"):
        writer.append(
            "MARGIN",
            event_id="too-late",
            event_at_utc="2025-06-02T00:00:00Z",
            payload={
                "balance_jpy": 1.0,
                "equity_jpy": 1.0,
                "used_margin_jpy": 0.0,
                "free_margin_jpy": 1.0,
            },
        )
    writer.close()

    manifest_path = tmp_path / "segment" / "final-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["event_count"] += 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CompactReplayEvidenceError, match="manifest SHA-256 mismatch"):
        verify_compact_replay_evidence(tmp_path / "segment")


def test_completed_stop_cannot_truncate_the_declared_replay_window(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    before = (tmp_path / "segment" / "events.jsonl").read_bytes()
    with pytest.raises(CompactReplayEvidenceError, match="equal replay_end_utc"):
        writer.stop(
            event_id="early-complete",
            stopped_at_utc="2025-06-01T12:00:00Z",
            status="COMPLETED",
            reason_code="WINDOW_COMPLETE",
            source_cursor_sha256="1" * 64,
            terminal_balance_jpy=200_000.0,
            terminal_equity_jpy=200_000.0,
            open_trade_count=0,
            pending_order_count=0,
        )
    assert (tmp_path / "segment" / "events.jsonl").read_bytes() == before
    with pytest.raises(CompactReplayEvidenceError, match="SEGMENT_STOP is required"):
        writer.finalize(finalized_at_utc="2025-06-02T00:00:01Z")
    writer.close()


def test_fill_units_must_equal_sealed_order_units(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append(
        "ORDER",
        event_id="one-unit-order",
        event_at_utc="2025-06-01T00:01:00Z",
        payload={
            "order_id": "order-one",
            "pair": "USD_JPY",
            "side": "LONG",
            "order_type": "MARKET",
            "status": "SUBMITTED",
            "units": 1,
            "requested_price": None,
            "stop_loss_price": 149.5,
            "take_profit_price": 150.5,
        },
    )
    before = (tmp_path / "segment" / "events.jsonl").read_bytes()
    with pytest.raises(CompactReplayEvidenceError, match="units differ"):
        writer.append(
            "FILL",
            event_id="inflated-fill",
            event_at_utc="2025-06-01T00:01:01Z",
            payload={
                "fill_id": "fill-inflated",
                "order_id": "order-one",
                "trade_id": "trade-inflated",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 1000,
                "fill_price": 150.0,
                "spread_pips": 0.2,
                "slippage_pips": 0.3,
                "fee_jpy": 0.0,
            },
        )
    assert (tmp_path / "segment" / "events.jsonl").read_bytes() == before
    writer.close()


def test_no_trade_segment_cannot_mint_terminal_profit(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    before = (tmp_path / "segment" / "events.jsonl").read_bytes()
    with pytest.raises(CompactReplayEvidenceError, match="accumulated balance"):
        writer.stop(
            event_id="invented-profit",
            stopped_at_utc="2025-06-02T00:00:00Z",
            status="COMPLETED",
            reason_code="WINDOW_COMPLETE",
            source_cursor_sha256="2" * 64,
            terminal_balance_jpy=300_000.0,
            terminal_equity_jpy=300_000.0,
            open_trade_count=0,
            pending_order_count=0,
        )
    assert (tmp_path / "segment" / "events.jsonl").read_bytes() == before
    writer.close()


def test_exit_profit_and_margin_are_independently_reconciled(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append(
        "ORDER",
        event_id="order-event",
        event_at_utc="2025-06-01T00:01:00Z",
        payload={
            "order_id": "order-pnl",
            "pair": "USD_JPY",
            "side": "LONG",
            "order_type": "MARKET",
            "status": "SUBMITTED",
            "units": 1000,
            "requested_price": None,
            "stop_loss_price": 149.5,
            "take_profit_price": 150.5,
        },
    )
    writer.append(
        "FILL",
        event_id="fill-event",
        event_at_utc="2025-06-01T00:01:01Z",
        payload={
            "fill_id": "fill-pnl",
            "order_id": "order-pnl",
            "trade_id": "trade-pnl",
            "pair": "USD_JPY",
            "side": "LONG",
            "units": 1000,
            "fill_price": 150.0,
            "spread_pips": 0.2,
            "slippage_pips": 0.3,
            "fee_jpy": 30.0,
        },
    )
    with pytest.raises(CompactReplayEvidenceError, match="margin equity"):
        writer.append(
            "MARGIN",
            event_id="false-margin",
            event_at_utc="2025-06-01T00:01:02Z",
            payload={
                "balance_jpy": 199_970.0,
                "equity_jpy": 199_900.0,
                "used_margin_jpy": 30_000.0,
                "free_margin_jpy": 180_000.0,
            },
        )
    with pytest.raises(CompactReplayEvidenceError, match="price calculation"):
        writer.append(
            "EXIT",
            event_id="invented-exit-pnl",
            event_at_utc="2025-06-01T01:01:01Z",
            payload={
                "exit_id": "exit-pnl",
                "trade_id": "trade-pnl",
                "pair": "USD_JPY",
                "reason": "TIME",
                "units": 1000,
                "exit_price": 150.2,
                "quote_to_jpy_rate": 1.0,
                "realized_pnl_jpy": 50_000.0,
                "financing_jpy": 0.0,
            },
        )
    writer.close()
