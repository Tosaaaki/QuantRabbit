from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.exit_decision import (
    ExitAction,
    ExitDecision,
    ExitDecisionError,
    ExitExecutionState,
    ExitExecutionStore,
    NoTouchError,
    OwnerBinding,
    PositionSide,
    PositionSnapshot,
    StalePositionError,
)


NOW = datetime(2026, 9, 4, 5, 0, tzinfo=timezone.utc)
AI_OWNER = OwnerBinding(
    owner_kind="AI_SYSTEM",
    owner_id="quant-rabbit-ai",
    client_extension_id="qrv1-ai-campaign-7",
    campaign_id="campaign-7",
)


def _position(**changes: object) -> PositionSnapshot:
    values: dict[str, object] = {
        "cycle_id": "cycle-20260904-0500",
        "broker_epoch": "broker-epoch-42",
        "position_revision": "position-revision-8",
        "trade_id": "900123",
        "instrument": "EUR_USD",
        "side": PositionSide.LONG,
        "units": 800,
        "owner_binding": AI_OWNER,
        "bid": "1.10200",
        "ask": "1.10210",
        "stop_loss": "1.09500",
        "take_profit": "1.11000",
    }
    values.update(changes)
    return PositionSnapshot(**values)  # type: ignore[arg-type]


def _decision(action: ExitAction = ExitAction.CLOSE_ALL, **changes: object) -> ExitDecision:
    values: dict[str, object] = {
        "action": action,
        "cycle_id": "cycle-20260904-0500",
        "broker_epoch": "broker-epoch-42",
        "position_revision": "position-revision-8",
        "trade_id": "900123",
        "instrument": "EUR_USD",
        "owner_binding": AI_OWNER,
        "created_at_utc": NOW,
        "reason": "fixed exit decision",
        "evidence_refs": ("evidence:sealed:1",),
    }
    values.update(changes)
    return ExitDecision.create(**values)  # type: ignore[arg-type]


def test_qrx_id_is_content_addressed_and_tamper_fails() -> None:
    decision = _decision()
    assert decision.decision_id.startswith("qrx_")
    assert len(decision.decision_id) == 68
    assert decision == _decision()
    assert decision.resource_claims == (
        "position:900123",
        "reverse-entry:cycle-20260904-0500:EUR_USD",
    )

    tampered = decision.to_dict()
    tampered["position_revision"] = "position-revision-9"
    with pytest.raises(ExitDecisionError, match="content address"):
        ExitDecision.from_mapping(tampered)


def test_manual_usdjpy_like_position_is_always_no_touch() -> None:
    decision = _decision(instrument="USD_JPY", trade_id="manual-777")
    manual = {
        "cycle_id": decision.cycle_id,
        "broker_epoch": decision.broker_epoch,
        "position_revision": decision.position_revision,
        "trade_id": "manual-777",
        "instrument": "USD_JPY",
        "side": "SHORT",
        "units": -2500,
        "owner_binding": {
            "owner_kind": "OPERATOR",
            "owner_id": "human",
            "client_extension_id": "manual-ticket",
            "campaign_id": "manual-campaign",
        },
        "bid": "148.100",
        "ask": "148.110",
    }
    with pytest.raises(NoTouchError) as exc_info:
        decision.validate_for_position(manual, now=NOW)
    assert exc_info.value.code == "NO_TOUCH"

    tagless = dict(manual)
    tagless.pop("owner_binding")
    with pytest.raises(NoTouchError) as exc_info:
        decision.validate_for_position(tagless, now=NOW)
    assert exc_info.value.code == "NO_TOUCH"


def test_stale_position_revision_and_broker_epoch_are_rejected() -> None:
    decision = _decision()
    with pytest.raises(StalePositionError) as exc_info:
        decision.validate_for_position(
            _position(position_revision="position-revision-9"),
            now=NOW,
        )
    assert exc_info.value.code == "STALE_POSITION_REVISION"

    with pytest.raises(StalePositionError) as exc_info:
        decision.validate_for_position(_position(broker_epoch="broker-epoch-43"), now=NOW)
    assert exc_info.value.code == "STALE_BROKER_EPOCH"


def test_reduce_requires_strict_partial_geometry() -> None:
    partial = _decision(ExitAction.REDUCE, units=300)
    partial.validate_for_position(_position(), now=NOW)

    for units in (800, 801):
        invalid = _decision(ExitAction.REDUCE, units=units)
        with pytest.raises(ExitDecisionError) as exc_info:
            invalid.validate_for_position(_position(), now=NOW)
        assert exc_info.value.code == "INVALID_REDUCE_GEOMETRY"

    with pytest.raises(ExitDecisionError, match="fields do not match REDUCE"):
        _decision(ExitAction.REDUCE)
    with pytest.raises(ExitDecisionError, match="fields do not match CLOSE_ALL"):
        _decision(ExitAction.CLOSE_ALL, units=800)


def test_ttl_and_price_geometry_fail_closed() -> None:
    expired = _decision()
    with pytest.raises(StalePositionError) as exc_info:
        expired.validate_for_position(_position(), now=NOW + timedelta(minutes=5))
    assert exc_info.value.code == "DECISION_EXPIRED"

    tightened = _decision(ExitAction.TIGHTEN_SL, stop_loss="1.09900")
    tightened.validate_for_position(_position(), now=NOW)
    invalid = _decision(ExitAction.TIGHTEN_SL, stop_loss="1.09400")
    with pytest.raises(ExitDecisionError) as exc_info:
        invalid.validate_for_position(_position(), now=NOW)
    assert exc_info.value.code == "INVALID_SL_GEOMETRY"


def test_duplicate_and_concurrent_reservation_claim_post_once(tmp_path) -> None:
    store = ExitExecutionStore(tmp_path / "exit-state")
    decision = _decision()

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda _: store.reserve(decision, _position(), now=NOW),
                range(16),
            )
        )
    assert sum(outcome.may_post for outcome in outcomes) == 1
    assert all(outcome.receipt.decision.decision_id == decision.decision_id for outcome in outcomes)
    assert store.read(decision.decision_id).state is ExitExecutionState.RESERVE_PRE_POST  # type: ignore[union-attr]

    # A cosmetically different decision still conflicts on the same exact
    # position and reverse-entry cycle resources.
    conflicting = _decision(reason="different prose cannot mint another claim")
    blocked = store.reserve(conflicting, _position(), now=NOW)
    assert blocked.may_post is False
    assert blocked.receipt.state is ExitExecutionState.UNKNOWN_NO_RESEND
    assert blocked.receipt.unknown_reason.startswith("RESOURCE_ALREADY_RESERVED:")  # type: ignore[union-attr]


class SimulatedProcessCrash(BaseException):
    pass


def test_crash_after_post_boundary_leaves_consumed_reservation(tmp_path) -> None:
    store = ExitExecutionStore(tmp_path / "exit-state")
    decision = _decision()
    calls = 0

    def crash() -> None:
        nonlocal calls
        calls += 1
        raise SimulatedProcessCrash

    with pytest.raises(SimulatedProcessCrash):
        store.run_post_once(decision, _position(), post=crash, now=NOW)
    assert calls == 1
    assert store.read(decision.decision_id).state is ExitExecutionState.POST_ATTEMPTED  # type: ignore[union-attr]

    restarted = ExitExecutionStore(tmp_path / "exit-state")
    retried = restarted.run_post_once(
        decision,
        _position(),
        post=lambda: pytest.fail("existing reservation must never resend"),
        now=NOW + timedelta(seconds=1),
    )
    assert retried.may_post is False
    assert retried.receipt.state is ExitExecutionState.POST_ATTEMPTED
    assert calls == 1


def test_transport_exception_is_terminal_unknown_and_never_resent(tmp_path) -> None:
    store = ExitExecutionStore(tmp_path / "exit-state")
    decision = _decision()
    calls = 0

    def transport_failure() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("ambiguous broker response")

    result = store.run_post_once(
        decision,
        _position(),
        post=transport_failure,
        now=NOW,
    )
    assert calls == 1
    assert result.receipt.state is ExitExecutionState.UNKNOWN_NO_RESEND
    assert result.receipt.unknown_reason == "TRANSPORT_EXCEPTION:TimeoutError"

    repeated = store.run_post_once(
        decision,
        _position(),
        post=lambda: pytest.fail("UNKNOWN_NO_RESEND must consume the signal"),
        now=NOW + timedelta(seconds=1),
    )
    assert repeated.may_post is False
    assert repeated.receipt.state is ExitExecutionState.UNKNOWN_NO_RESEND


def test_successful_post_requires_explicit_reconciliation_terminal(tmp_path) -> None:
    store = ExitExecutionStore(tmp_path / "exit-state")
    decision = _decision(ExitAction.REDUCE, units=200)
    attempted = store.run_post_once(
        decision,
        _position(),
        post=lambda: {"order_id": "broker-order-1"},
        now=NOW,
    )
    assert attempted.receipt.state is ExitExecutionState.RECONCILING
    assert attempted.receipt.broker_result_digest
    terminal = store.mark_terminal(
        decision.decision_id,
        outcome="PARTIAL_REDUCE_CONFIRMED",
        now=NOW + timedelta(seconds=1),
    )
    assert terminal.state is ExitExecutionState.TERMINAL
    assert terminal.terminal_outcome == "PARTIAL_REDUCE_CONFIRMED"

