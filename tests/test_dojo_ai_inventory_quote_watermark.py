from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_rabbit import dojo_ai_inventory_quote_watermark as watermark


OPEN = datetime(2026, 7, 23, 12, 0, 30, tzinfo=timezone.utc)
WEEKEND = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def _room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(watermark, "_trusted_repository_root", lambda: tmp_path)
    room = (
        tmp_path
        / "research"
        / "data"
        / "dojo_paper_ai_inventory_v1"
        / "rooms"
        / "paper-ai-inventory-experiment-v1"
        / "paper-ai-inventory-room-v1"
    )
    room.mkdir(parents=True)
    return room


def _append(room: Path) -> dict[str, object]:
    return watermark.append_ai_inventory_quote_watermark(
        room,
        pair="USD_JPY",
        bid=163.0,
        ask=163.01,
        timestamp_utc="2026-07-23T12:00:00Z",
        slippage_pips_per_fill=0.3,
        financing_pips_per_day=0.8,
        acquisition_receipt_sha256="a" * 64,
    )


def test_writer_persists_content_addressed_source_and_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)

    row = _append(room)
    retried = _append(room)
    validation = watermark.validate_ai_inventory_quote_watermarks(
        room / watermark.QUOTE_WATERMARK_LEDGER_NAME
    )

    assert retried == row
    assert validation == {
        "valid": True,
        "row_count": 1,
        "terminal_quote_sha256": row["quote_sha256"],
    }
    source = (
        room
        / watermark.QUOTE_SOURCE_DIRECTORY
        / f"{row['source_sha256']}.json"
    )
    assert source.is_file()
    assert source.is_symlink() is False
    assert row["paper_only"] is True
    assert row["order_authority"] == "NONE"
    assert row["live_permission"] is False
    assert row["acquisition_receipt_sha256"] == "a" * 64


def test_retry_after_source_persisted_before_ledger_append_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    original_open = watermark._open_locked_ledger

    monkeypatch.setattr(
        watermark,
        "_open_locked_ledger",
        lambda _path: (_ for _ in ()).throw(
            watermark.AiInventoryQuoteWatermarkError(
                "injected failure after source persistence"
            )
        ),
    )
    with pytest.raises(
        watermark.AiInventoryQuoteWatermarkError,
        match="injected failure",
    ):
        _append(room)

    sources = list((room / watermark.QUOTE_SOURCE_DIRECTORY).iterdir())
    assert len(sources) == 1
    assert not (room / watermark.QUOTE_WATERMARK_LEDGER_NAME).exists()

    monkeypatch.setattr(watermark, "_open_locked_ledger", original_open)
    row = _append(room)
    validation = watermark.validate_ai_inventory_quote_watermarks(
        room / watermark.QUOTE_WATERMARK_LEDGER_NAME
    )

    assert validation["row_count"] == 1
    assert validation["terminal_quote_sha256"] == row["quote_sha256"]
    assert list((room / watermark.QUOTE_SOURCE_DIRECTORY).iterdir()) == sources


def test_retry_after_empty_ledger_created_before_row_append_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    original_read = watermark._read_validate_locked_ledger

    monkeypatch.setattr(
        watermark,
        "_read_validate_locked_ledger",
        lambda _handle, _path: (_ for _ in ()).throw(
            watermark.AiInventoryQuoteWatermarkError(
                "injected failure before row append"
            )
        ),
    )
    with pytest.raises(
        watermark.AiInventoryQuoteWatermarkError,
        match="injected failure",
    ):
        _append(room)

    ledger = room / watermark.QUOTE_WATERMARK_LEDGER_NAME
    assert ledger.is_file()
    assert ledger.read_bytes() == b""

    monkeypatch.setattr(watermark, "_read_validate_locked_ledger", original_read)
    row = _append(room)
    validation = watermark.validate_ai_inventory_quote_watermarks(ledger)

    assert validation["row_count"] == 1
    assert validation["terminal_quote_sha256"] == row["quote_sha256"]


def test_same_pair_timestamp_conflict_never_rewrites_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    _append(room)
    ledger = room / watermark.QUOTE_WATERMARK_LEDGER_NAME
    before = ledger.read_bytes()

    with pytest.raises(watermark.AiInventoryQuoteWatermarkConflictError):
        watermark.append_ai_inventory_quote_watermark(
            room,
            pair="USD_JPY",
            bid=162.99,
            ask=163.0,
            timestamp_utc="2026-07-23T12:00:00Z",
            slippage_pips_per_fill=0.3,
            financing_pips_per_day=0.8,
            acquisition_receipt_sha256="a" * 64,
        )

    assert ledger.read_bytes() == before


def test_same_quote_with_different_acquisition_receipt_is_a_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    _append(room)
    ledger = room / watermark.QUOTE_WATERMARK_LEDGER_NAME
    before = ledger.read_bytes()

    with pytest.raises(watermark.AiInventoryQuoteWatermarkConflictError):
        watermark.append_ai_inventory_quote_watermark(
            room,
            pair="USD_JPY",
            bid=163.0,
            ask=163.01,
            timestamp_utc="2026-07-23T12:00:00Z",
            slippage_pips_per_fill=0.3,
            financing_pips_per_day=0.8,
            acquisition_receipt_sha256="b" * 64,
        )

    assert ledger.read_bytes() == before


def test_weekend_fails_before_room_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watermark, "_utc_now", lambda: WEEKEND)
    monkeypatch.setattr(
        watermark,
        "_trusted_repository_root",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve root")),
    )

    with pytest.raises(watermark.AiInventoryQuoteWatermarkMarketClosedError):
        watermark.append_ai_inventory_quote_watermark(
            tmp_path / "missing",
            pair="USD_JPY",
            bid=163.0,
            ask=163.01,
            timestamp_utc="2026-07-25T12:00:00Z",
            slippage_pips_per_fill=0.0,
            financing_pips_per_day=0.0,
            acquisition_receipt_sha256="a" * 64,
        )

    assert not (tmp_path / "missing").exists()


def test_source_tamper_fails_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    row = _append(room)
    source = (
        room
        / watermark.QUOTE_SOURCE_DIRECTORY
        / f"{row['source_sha256']}.json"
    )
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        watermark.AiInventoryQuoteWatermarkError,
        match="source digest mismatch",
    ):
        watermark.validate_ai_inventory_quote_watermarks(
            room / watermark.QUOTE_WATERMARK_LEDGER_NAME
        )


def test_symlink_ledger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(watermark, "_utc_now", lambda: OPEN)
    target = tmp_path / "outside.jsonl"
    target.write_bytes(b"")
    (room / watermark.QUOTE_WATERMARK_LEDGER_NAME).symlink_to(target)

    with pytest.raises(watermark.AiInventoryQuoteWatermarkError):
        _append(room)
