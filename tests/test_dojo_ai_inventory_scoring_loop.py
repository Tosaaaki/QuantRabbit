from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from quant_rabbit import dojo_ai_inventory_evaluator as evaluator
from quant_rabbit import dojo_ai_inventory_scoring_loop as scoring
from tests.test_dojo_ai_inventory_evaluator import _trusted_room


REGISTER_AT = datetime(2026, 7, 23, 12, 0, 10, tzinfo=timezone.utc)
SCORE_AT = datetime(2026, 7, 23, 12, 3, 0, tzinfo=timezone.utc)
WEEKEND = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _preflight(room: Path) -> dict[str, object]:
    return {
        "experiment_id": room.parent.name,
        "room_id": room.name,
        "launch_preflight_token_sha256": "a" * 64,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _patch_open_clock(
    monkeypatch: pytest.MonkeyPatch, instant: datetime
) -> None:
    monkeypatch.setattr(scoring, "_utc_now", lambda: instant)
    monkeypatch.setattr(evaluator, "_utc_now", lambda: instant)


def _prepare_prospective_room(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    flat_allow: bool = False,
) -> dict[str, object]:
    fixture = _trusted_room(tmp_path, monkeypatch, flat_allow=flat_allow)
    room = fixture["room"]
    assert isinstance(room, Path)
    monkeypatch.setattr(scoring, "_trusted_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        scoring,
        "verify_paper_ai_inventory_launch_preflight",
        lambda *_args, **_kwargs: _preflight(room),
    )
    broker = fixture["broker_path"]
    quote = fixture["quote_path"]
    assert isinstance(broker, Path)
    assert isinstance(quote, Path)
    full_broker = broker.read_bytes()
    full_quote = quote.read_bytes()
    broker_rows = full_broker.splitlines(keepends=True)
    quote_rows = full_quote.splitlines(keepends=True)
    broker.write_bytes(
        b"".join(broker_rows if flat_allow else broker_rows[:-1])
    )
    quote.write_bytes(b"".join(quote_rows[:2]))
    return {
        **fixture,
        "full_broker": full_broker,
        "full_quote": full_quote,
    }


def _restore_outcome_sources(fixture: dict[str, object]) -> None:
    broker = fixture["broker_path"]
    quote = fixture["quote_path"]
    assert isinstance(broker, Path)
    assert isinstance(quote, Path)
    broker.write_bytes(fixture["full_broker"])
    quote.write_bytes(fixture["full_quote"])


def _receipt(request: dict[str, object]) -> dict[str, object]:
    body = {
        "contract": scoring.SCORING_FEEDBACK_RECEIPT_CONTRACT,
        "feedback_identity_sha256": request["feedback_identity_sha256"],
        "candidate_id": request["candidate_id"],
        "decision_sha256": request["decision_sha256"],
        "evaluation_sha256": request["evaluation_sha256"],
        "sink_id": "candidate-feedback-test-sink",
        "sink_event_sha256": "b" * 64,
        "sink_ledger_tip_sha256": "c" * 64,
        "accepted_at_utc": "2026-07-23T12:03:00Z",
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "receipt_sha256": scoring.feedback_receipt_sha256(body)}


def test_weekend_stops_before_room_resolution_or_filesystem_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring, "_utc_now", lambda: WEEKEND)
    monkeypatch.setattr(
        scoring,
        "_require_registered_room",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("room must not be resolved")
        ),
    )

    with pytest.raises(scoring.AiInventoryScoringMarketClosedError):
        scoring.run_ai_inventory_scoring_cycle(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()


def test_unregistered_room_fails_before_decision_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    room = fixture["room"]
    assert isinstance(room, Path)
    monkeypatch.setattr(scoring, "_trusted_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        scoring,
        "verify_paper_ai_inventory_launch_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("not registered")
        ),
    )
    _patch_open_clock(monkeypatch, REGISTER_AT)

    with patch.object(scoring, "_read_validate_decisions") as reader:
        with pytest.raises(scoring.AiInventoryScoringRegistrationError):
            scoring.run_ai_inventory_scoring_cycle(room)
    reader.assert_not_called()


def test_future_decision_fails_before_outcome_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    room = fixture["room"]
    assert isinstance(room, Path)
    monkeypatch.setattr(scoring, "_trusted_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        scoring,
        "verify_paper_ai_inventory_launch_preflight",
        lambda *_args, **_kwargs: _preflight(room),
    )
    _patch_open_clock(
        monkeypatch,
        datetime(2026, 7, 23, 11, 59, 59, tzinfo=timezone.utc),
    )

    with patch.object(scoring, "_read_validate_broker_rows") as broker_reader:
        with pytest.raises(scoring.AiInventoryScoringIntegrityError):
            scoring.run_ai_inventory_scoring_cycle(room)

    broker_reader.assert_not_called()
    assert not (room / scoring.SCORING_CHECKPOINT_LEDGER_NAME).exists()


def test_pending_settlement_score_feedback_and_retry_are_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    decision = fixture["decision"]
    assert isinstance(room, Path)
    assert isinstance(decision, dict)
    _patch_open_clock(monkeypatch, REGISTER_AT)

    pending = scoring.run_ai_inventory_scoring_cycle(room)

    assert pending.pending_decisions == (decision["decision_sha256"],)
    assert [row["event_type"] for row in pending.checkpoint_events] == [
        scoring.EVENT_DECISION_PENDING
    ]

    _restore_outcome_sources(fixture)
    _patch_open_clock(monkeypatch, SCORE_AT)
    callback_requests: list[dict[str, object]] = []

    def callback(request: dict[str, object]) -> dict[str, object]:
        callback_requests.append(request)
        return _receipt(request)

    scored = scoring.run_ai_inventory_scoring_cycle(
        room,
        feedback_callback=callback,
    )

    assert scored.scored_decisions == (decision["decision_sha256"],)
    assert scored.unscored_decisions == ()
    assert [row["event_type"] for row in scored.checkpoint_events] == [
        scoring.EVENT_SCORED,
        scoring.EVENT_FEEDBACK_PENDING,
        scoring.EVENT_FEEDBACK_ACKNOWLEDGED,
    ]
    assert len(callback_requests) == 1
    evaluation = room / evaluator.EVALUATION_LEDGER_NAME
    checkpoint = room / scoring.SCORING_CHECKPOINT_LEDGER_NAME
    assert evaluation.read_bytes().count(b"\n") == 1
    assert checkpoint.read_bytes().count(b"\n") == 4

    retried = scoring.run_ai_inventory_scoring_cycle(
        room,
        feedback_callback=callback,
    )

    assert retried.checkpoint_events == ()
    assert retried.feedback_requests == ()
    assert len(callback_requests) == 1
    assert evaluation.read_bytes().count(b"\n") == 1
    assert checkpoint.read_bytes().count(b"\n") == 4


def test_flat_entry_gate_scores_at_precommitted_fixed_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch, flat_allow=True)
    room = fixture["room"]
    decision = fixture["decision"]
    assert isinstance(room, Path)
    assert isinstance(decision, dict)
    monkeypatch.setattr(scoring, "FIXED_HORIZON_SECONDS", 119)
    _patch_open_clock(monkeypatch, REGISTER_AT)
    first = scoring.run_ai_inventory_scoring_cycle(room)
    assert first.pending_decisions == (decision["decision_sha256"],)

    _restore_outcome_sources(fixture)
    _patch_open_clock(monkeypatch, SCORE_AT)
    result = scoring.run_ai_inventory_scoring_cycle(room)

    assert result.scored_decisions == (decision["decision_sha256"],)
    assert len(result.feedback_requests) == 1
    evaluation_rows = [
        json.loads(line)
        for line in (room / evaluator.EVALUATION_LEDGER_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert evaluation_rows[0]["outcome_kind"] == "FIXED_HORIZON"
    assert evaluation_rows[0]["horizon_end_at_utc"] == "2026-07-23T12:02:00Z"


def test_first_scan_after_outcome_is_preserved_as_unscored_not_backfilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    decision = fixture["decision"]
    assert isinstance(room, Path)
    assert isinstance(decision, dict)
    _restore_outcome_sources(fixture)
    _patch_open_clock(monkeypatch, SCORE_AT)

    with patch.object(scoring, "evaluate_ai_inventory_outcome") as evaluate:
        result = scoring.run_ai_inventory_scoring_cycle(room)

    evaluate.assert_not_called()
    assert result.unscored_decisions == (decision["decision_sha256"],)
    assert [row["event_type"] for row in result.checkpoint_events] == [
        scoring.EVENT_DECISION_PENDING,
        scoring.EVENT_UNSCORED,
    ]
    assert (
        result.checkpoint_events[-1]["payload"]["reason_code"]
        == "MISSED_PROSPECTIVE_REGISTRATION"
    )
    assert not (room / evaluator.EVALUATION_LEDGER_NAME).exists()


def test_source_integrity_defect_after_pending_stays_ai_shadow_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    decision = fixture["decision"]
    assert isinstance(room, Path)
    assert isinstance(decision, dict)
    _patch_open_clock(monkeypatch, REGISTER_AT)
    scoring.run_ai_inventory_scoring_cycle(room)
    _restore_outcome_sources(fixture)
    quote = fixture["quote_path"]
    assert isinstance(quote, Path)
    final_quote = json.loads(quote.read_text(encoding="utf-8").splitlines()[-1])
    source = room / "quote_sources" / f"{final_quote['source_sha256']}.json"
    source.write_text("{}\n", encoding="utf-8")
    _patch_open_clock(monkeypatch, SCORE_AT)

    with patch.object(scoring, "evaluate_ai_inventory_outcome") as evaluate:
        result = scoring.run_ai_inventory_scoring_cycle(room)

    evaluate.assert_not_called()
    assert result.unscored_decisions == (decision["decision_sha256"],)
    unscored = result.checkpoint_events[-1]
    assert unscored["event_type"] == scoring.EVENT_UNSCORED
    assert unscored["payload"]["reason_code"] == "SOURCE_INTEGRITY_DEFECT"
    assert unscored["payload"]["defect"]["message_sha256"]
    retry = scoring.run_ai_inventory_scoring_cycle(room)
    assert retry.checkpoint_events == ()


def test_invalid_callback_receipt_leaves_retryable_feedback_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    assert isinstance(room, Path)
    _patch_open_clock(monkeypatch, REGISTER_AT)
    scoring.run_ai_inventory_scoring_cycle(room)
    _restore_outcome_sources(fixture)
    _patch_open_clock(monkeypatch, SCORE_AT)

    with pytest.raises(scoring.AiInventoryScoringIntegrityError):
        scoring.run_ai_inventory_scoring_cycle(
            room,
            feedback_callback=lambda _request: {},
        )

    checkpoint_rows = [
        json.loads(line)
        for line in (room / scoring.SCORING_CHECKPOINT_LEDGER_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert checkpoint_rows[-1]["event_type"] == scoring.EVENT_FEEDBACK_PENDING
    request = checkpoint_rows[-1]["payload"]
    backdated = _receipt(request)
    backdated["accepted_at_utc"] = "2026-07-23T12:02:59Z"
    backdated["receipt_sha256"] = scoring.feedback_receipt_sha256(backdated)
    with pytest.raises(
        scoring.AiInventoryScoringIntegrityError,
        match="predates",
    ):
        scoring.acknowledge_ai_inventory_scoring_feedback(room, backdated)
    acknowledged = scoring.acknowledge_ai_inventory_scoring_feedback(
        room,
        _receipt(request),
    )
    assert acknowledged["event_type"] == scoring.EVENT_FEEDBACK_ACKNOWLEDGED


def test_checkpoint_tamper_blocks_later_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    assert isinstance(room, Path)
    _patch_open_clock(monkeypatch, REGISTER_AT)
    scoring.run_ai_inventory_scoring_cycle(room)
    checkpoint = room / scoring.SCORING_CHECKPOINT_LEDGER_NAME
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    row["payload"]["status"] = "FORGED"
    checkpoint.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    validation = scoring.validate_ai_inventory_scoring_checkpoint_ledger(
        checkpoint
    )
    assert validation["valid"] is False
    with pytest.raises(scoring.AiInventoryScoringIntegrityError):
        scoring.run_ai_inventory_scoring_cycle(room)


def test_semantically_invalid_checkpoint_is_rejected_even_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_prospective_room(tmp_path, monkeypatch)
    room = fixture["room"]
    assert isinstance(room, Path)
    _patch_open_clock(monkeypatch, REGISTER_AT)
    scoring.run_ai_inventory_scoring_cycle(room)
    checkpoint = room / scoring.SCORING_CHECKPOINT_LEDGER_NAME
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    row["payload"]["target_horizon_at_utc"] = row["payload"][
        "decision_cutoff_at_utc"
    ]
    row["payload_sha256"] = scoring._sha256(
        scoring._canonical_json(row["payload"]).encode()
    )
    identity_body = {
        "event_type": row["event_type"],
        "decision_sha256": row["decision_sha256"],
        "payload": row["payload"],
    }
    row["checkpoint_identity_sha256"] = scoring._sha256(
        scoring._canonical_json(identity_body).encode()
    )
    checkpoint_body = {
        key: value for key, value in row.items() if key != "checkpoint_sha256"
    }
    row["checkpoint_sha256"] = scoring._sha256(
        scoring._canonical_json(checkpoint_body).encode()
    )
    checkpoint.write_text(
        scoring._canonical_json(row) + "\n",
        encoding="utf-8",
    )

    validation = scoring.validate_ai_inventory_scoring_checkpoint_ledger(
        checkpoint
    )

    assert validation["valid"] is False
