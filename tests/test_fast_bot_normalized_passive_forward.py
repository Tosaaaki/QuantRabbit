from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.contextual_technical_forward import BidAskCandle, Ohlc
from quant_rabbit.fast_bot_normalized_passive_forward import (
    build_decision,
    build_scorecard,
    decision_window,
    load_policy,
    resolve_signal,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "fast_bot_normalized_passive_forward_v1.json"


def _m1_candles(
    decision: datetime,
    *,
    slope: float = -0.00001,
    spread: float = 0.00010,
) -> list[BidAskCandle]:
    start = decision - timedelta(minutes=240)
    candles: list[BidAskCandle] = []
    for index in range(241):
        timestamp = start + timedelta(minutes=index)
        mid = 1.11000 + slope * index
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        candles.append(
            BidAskCandle(
                timestamp_utc=timestamp,
                pair="EUR_USD",
                bid=Ohlc(bid, bid + 0.00002, bid - 0.00002, bid),
                ask=Ohlc(ask, ask + 0.00002, ask - 0.00002, ask),
            )
        )
    return candles


def _s5(
    timestamp: datetime,
    *,
    bid_open: float,
    ask_low: float,
    spread: float = 0.00010,
) -> BidAskCandle:
    ask_open = bid_open + spread
    return BidAskCandle(
        timestamp_utc=timestamp,
        pair="EUR_USD",
        bid=Ohlc(bid_open, bid_open + 0.00002, bid_open - 0.00002, bid_open),
        ask=Ohlc(ask_open, ask_open + 0.00002, ask_low, ask_open),
    )


def _emitted_decision() -> tuple[dict, dict]:
    policy = load_policy(POLICY_PATH)
    decision = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    row = build_decision(
        policy,
        policy_sha256="a" * 64,
        decision_at_utc=decision,
        observed_at_utc=decision + timedelta(minutes=1, seconds=10),
        candles=_m1_candles(decision),
        truth_chunk_sha256=["b" * 64],
    )
    assert row["status"] == "EMITTED"
    return policy, row


def test_policy_binds_rejected_historical_artifact_and_zero_authority() -> None:
    policy = load_policy(POLICY_PATH)
    assert policy["selection_disclosure"]["selection_status"] == "HOLDOUT_REJECT"
    assert policy["selection_disclosure"]["shadow_candidate_admitted"] is False
    assert policy["primary_trading_candidate_allowed"] is False
    assert policy["execution_authority"] == "NONE"
    assert policy["external_orders"] == 0


def test_policy_rejects_tampered_historical_artifact(tmp_path: Path) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    source = Path(policy["selection_disclosure"]["source_artifact_path"])
    tampered = tmp_path / "research.json"
    tampered.write_bytes(source.read_bytes() + b"\n")
    policy["selection_disclosure"]["source_artifact_path"] = str(tampered)
    candidate = tmp_path / "policy.json"
    candidate.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash changed"):
        load_policy(candidate)


def test_decision_window_opens_only_after_complete_hourly_m1_and_after_lock() -> None:
    policy = load_policy(POLICY_PATH)
    waiting = decision_window(
        policy,
        as_of_utc=datetime(2026, 9, 1, 12, 0, 30, tzinfo=timezone.utc),
    )
    opened = decision_window(
        policy,
        as_of_utc=datetime(2026, 9, 1, 12, 1, 15, tzinfo=timezone.utc),
    )
    late = decision_window(
        policy,
        as_of_utc=datetime(2026, 9, 1, 12, 2, 31, tzinfo=timezone.utc),
    )
    assert waiting["status"] == "WAITING_FOR_COMPLETE_M1"
    assert opened["status"] == "OPEN"
    assert late["status"] == "OUTSIDE_COLLECTION_WINDOW"


def test_frozen_down_return_emits_long_limit_with_exact_rounding() -> None:
    policy, row = _emitted_decision()
    signal = row["signals"][0]
    expected_raw = row["decision_bid_close"] + (
        row["decision_ask_close"] - row["decision_bid_close"]
    ) * policy["vehicle"]["entry_spread_fraction"]
    expected = (int((expected_raw + 1e-12) / 0.00001)) * 0.00001
    assert row["source_direction"] == "DOWN"
    assert row["selected_side"] == "LONG"
    assert row["normalized_return"] >= 1.25
    assert signal["side"] == "LONG"
    assert abs(signal["entry_price"] - expected) < 1e-9
    assert signal["order_intents"] == []
    assert signal["execution_authority"] == "NONE"


def test_opposite_direction_is_side_filtered_but_reserves_overlap() -> None:
    policy = load_policy(POLICY_PATH)
    decision = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    row = build_decision(
        policy,
        policy_sha256="a" * 64,
        decision_at_utc=decision,
        observed_at_utc=decision + timedelta(minutes=1, seconds=10),
        candles=_m1_candles(decision, slope=0.00001),
    )
    assert row["status"] == "SIDE_FILTERED_RESERVED"
    assert row["selected_side"] == "SHORT"
    assert row["signals"] == []
    assert row["reservation_until_utc"] is not None


def test_prior_reservation_suppresses_otherwise_eligible_signal() -> None:
    policy, first = _emitted_decision()
    next_decision = datetime(2026, 9, 2, 13, 0, tzinfo=timezone.utc)
    row = build_decision(
        policy,
        policy_sha256="a" * 64,
        decision_at_utc=next_decision,
        observed_at_utc=next_decision + timedelta(minutes=1, seconds=10),
        candles=_m1_candles(next_decision),
        prior_decisions=[first],
    )
    assert row["status"] == "SKIPPED_OVERLAP"
    assert row["signals"] == []
    assert row["overlap_reservation_source_decision_id"] == first["decision_id"]


def test_s5_touch_fills_at_limit_and_exits_at_exact_time_close() -> None:
    _policy, decision = _emitted_decision()
    signal = decision["signals"][0]
    activation = datetime.fromisoformat(signal["activation_at_utc"])
    limit = float(signal["entry_price"])
    fill = _s5(activation + timedelta(seconds=10), bid_open=limit - 0.00004, ask_low=limit)
    exit_at = (activation + timedelta(seconds=10)).replace(second=0) + timedelta(minutes=240)
    exit_candle = _s5(exit_at, bid_open=limit + 0.00030, ask_low=limit + 0.00035)
    outcome = resolve_signal(
        signal,
        [fill, exit_candle],
        policy_sha256="a" * 64,
        resolved_at_utc=datetime.fromisoformat(signal["maturity_at_utc"]) + timedelta(minutes=1),
        truth_chunk_sha256=["c" * 64],
    )
    assert outcome["status"] == "FILLED_TIME_CLOSE"
    assert outcome["entry_price"] == limit
    assert outcome["realized_pips"] == 3.0
    assert outcome["execution_authority"] == "NONE"


def test_s5_without_touch_records_unfilled_not_zero_return() -> None:
    _policy, decision = _emitted_decision()
    signal = decision["signals"][0]
    activation = datetime.fromisoformat(signal["activation_at_utc"])
    limit = float(signal["entry_price"])
    outcome = resolve_signal(
        signal,
        [_s5(activation, bid_open=limit, ask_low=limit + 0.00001)],
        policy_sha256="a" * 64,
        resolved_at_utc=datetime.fromisoformat(signal["maturity_at_utc"]) + timedelta(minutes=1),
    )
    assert outcome["status"] == "UNFILLED"
    assert outcome["filled"] is False
    assert outcome["realized_pips"] is None


def test_missing_exact_time_close_fails_instead_of_inventing_truth() -> None:
    _policy, decision = _emitted_decision()
    signal = decision["signals"][0]
    activation = datetime.fromisoformat(signal["activation_at_utc"])
    limit = float(signal["entry_price"])
    with pytest.raises(ValueError, match="time-close quote is missing"):
        resolve_signal(
            signal,
            [_s5(activation, bid_open=limit - 0.00004, ask_low=limit)],
            policy_sha256="a" * 64,
            resolved_at_utc=datetime.fromisoformat(signal["maturity_at_utc"]) + timedelta(minutes=1),
        )


def test_scorecard_can_pass_research_gate_but_never_grants_live_authority() -> None:
    policy, decision = _emitted_decision()
    signal = decision["signals"][0]
    outcomes = []
    decisions = []
    start = datetime(2026, 9, 2, tzinfo=timezone.utc)
    for index in range(100):
        signal_id = hashlib.sha256(str(index).encode()).hexdigest()
        activation = start + timedelta(days=index // 10, minutes=index % 10)
        decisions.append(
            {
                "signals": [
                    {
                        **signal,
                        "signal_sha256": signal_id,
                        "activation_at_utc": activation.isoformat(),
                    }
                ]
            }
        )
        outcomes.append(
            {
                "signal_sha256": signal_id,
                "filled": True,
                "fill_at_utc": activation.isoformat(),
                "realized_pips": 2.0,
            }
        )
    scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256="a" * 64,
        as_of_utc=start + timedelta(days=20),
    )
    assert scorecard["prospective_gate_passed"] is True
    assert scorecard["status"] == "PROSPECTIVE_EVIDENCE_PASSED_REVIEW_REQUIRED"
    assert scorecard["primary_trading_candidate_allowed"] is False
    assert scorecard["promotion_allowed"] is False
    assert scorecard["live_permission"] is False
    assert scorecard["external_order_attempts"] == 0
    assert scorecard["external_orders"] == 0
