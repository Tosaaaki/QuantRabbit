from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.contextual_technical_forward import BidAskCandle, Ohlc
from quant_rabbit.fast_bot_normalized_passive_family_forward import (
    ANCHOR_ID,
    EXPLORATORY_ID,
    build_decision,
    build_scorecard,
    decision_window,
    load_policy,
    resolve_signal,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "fast_bot_normalized_passive_forward_v2.json"


def _m1_candles(
    decision: datetime,
    *,
    pair: str,
    slope: float,
    spread: float,
) -> list[BidAskCandle]:
    start = decision - timedelta(minutes=240)
    base = 150.0 if pair == "USD_JPY" else 1.11
    wick = 0.002 if pair == "USD_JPY" else 0.00002
    candles: list[BidAskCandle] = []
    for index in range(241):
        timestamp = start + timedelta(minutes=index)
        mid = base + slope * index
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        candles.append(
            BidAskCandle(
                timestamp_utc=timestamp,
                pair=pair,
                bid=Ohlc(bid, bid + wick, bid - wick, bid),
                ask=Ohlc(ask, ask + wick, ask - wick, ask),
            )
        )
    return candles


def _s5(
    timestamp: datetime,
    *,
    pair: str,
    bid_open: float,
    ask_open: float,
    bid_high: float | None = None,
    ask_low: float | None = None,
) -> BidAskCandle:
    small = 0.001 if pair == "USD_JPY" else 0.00001
    return BidAskCandle(
        timestamp_utc=timestamp,
        pair=pair,
        bid=Ohlc(
            bid_open,
            bid_high if bid_high is not None else bid_open + small,
            bid_open - small,
            bid_open,
        ),
        ask=Ohlc(
            ask_open,
            ask_open + small,
            ask_low if ask_low is not None else ask_open - small,
            ask_open,
        ),
    )


def _candidate(policy: dict, candidate_id: str) -> dict:
    return next(row for row in policy["candidates"] if row["candidate_id"] == candidate_id)


def _synthetic_family_rows(
    policy: dict,
    *,
    candidate_id: str,
    values: list[float],
) -> tuple[list[dict], list[dict]]:
    candidate = _candidate(policy, candidate_id)
    start = datetime(2026, 9, 2, tzinfo=timezone.utc)
    decisions: list[dict] = []
    outcomes: list[dict] = []
    for index, value in enumerate(values):
        signal_id = hashlib.sha256(f"signal-{index}".encode()).hexdigest()
        activation = start + timedelta(days=index // 10, minutes=index % 10)
        decisions.append(
            {
                "candidate_id": candidate_id,
                "signals": [
                    {
                        "candidate_id": candidate_id,
                        "signal_sha256": signal_id,
                        "activation_at_utc": activation.isoformat(),
                    }
                ],
            }
        )
        outcomes.append(
            {
                "candidate_id": candidate_id,
                "signal_sha256": signal_id,
                "filled": True,
                "fill_at_utc": activation.isoformat(),
                "realized_pips": value,
                "pair": candidate["selector"]["pair"],
            }
        )
    return decisions, outcomes


def test_policy_freezes_two_candidates_without_promoting_either() -> None:
    policy = load_policy(POLICY_PATH)
    assert [row["candidate_id"] for row in policy["candidates"]] == [
        ANCHOR_ID,
        EXPLORATORY_ID,
    ]
    assert policy["candidates"][0]["historical_status"] == "HOLDOUT_REJECT"
    assert policy["candidates"][1]["historical_holdout"] is None
    assert policy["candidates"][1]["historical_holdout_inspected"] is False
    assert policy["family_evaluation"]["multiple_testing_correction"] == "BONFERRONI"
    assert policy["primary_trading_candidate_allowed"] is False
    assert policy["automatic_replacement_allowed"] is False
    assert policy["execution_authority"] == "NONE"


def test_policy_rejects_candidate_family_tampering(tmp_path: Path) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["candidates"][1]["selector"]["normalized_threshold"] = 1.25
    candidate = tmp_path / "policy.json"
    candidate.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate selector changed"):
        load_policy(candidate)


def test_v2_window_starts_only_at_new_future_cutoff() -> None:
    policy = load_policy(POLICY_PATH)
    before = decision_window(
        policy,
        as_of_utc=datetime(2026, 9, 1, 13, 1, 10, tzinfo=timezone.utc),
    )
    opened = decision_window(
        policy,
        as_of_utc=datetime(2026, 9, 1, 14, 1, 10, tzinfo=timezone.utc),
    )
    assert before["status"] == "BEFORE_FORWARD_LOCK"
    assert opened["status"] == "OPEN"


def test_anchor_and_independent_candidate_emit_from_their_own_rules() -> None:
    policy = load_policy(POLICY_PATH)
    decision = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    anchor = build_decision(
        policy,
        _candidate(policy, ANCHOR_ID),
        policy_sha256="a" * 64,
        decision_at_utc=decision,
        observed_at_utc=decision + timedelta(minutes=1, seconds=10),
        candles=_m1_candles(
            decision,
            pair="EUR_USD",
            slope=-0.00001,
            spread=0.00010,
        ),
    )
    exploratory = build_decision(
        policy,
        _candidate(policy, EXPLORATORY_ID),
        policy_sha256="a" * 64,
        decision_at_utc=decision,
        observed_at_utc=decision + timedelta(minutes=1, seconds=10),
        candles=_m1_candles(
            decision,
            pair="USD_JPY",
            slope=0.01,
            spread=0.01,
        ),
        prior_decisions=[anchor],
    )
    assert anchor["status"] == "EMITTED"
    assert anchor["signals"][0]["side"] == "LONG"
    assert exploratory["status"] == "EMITTED"
    assert exploratory["signals"][0]["pair"] == "USD_JPY"
    assert exploratory["overlap_reservation_source_decision_id"] is None
    assert exploratory["execution_authority"] == "NONE"


def test_generic_short_resolution_uses_bid_fill_and_ask_time_close() -> None:
    activation = datetime(2026, 9, 2, 12, 1, tzinfo=timezone.utc)
    limit = 150.005
    signal = {
        "family_id": "family",
        "candidate_id": "candidate",
        "decision_id": "decision",
        "signal_sha256": "b" * 64,
        "pair": "USD_JPY",
        "side": "SHORT",
        "direction": "DOWN",
        "decision_at_utc": (activation - timedelta(minutes=1)).isoformat(),
        "activation_at_utc": activation.isoformat(),
        "fill_window_end_exclusive_utc": (activation + timedelta(minutes=5)).isoformat(),
        "maturity_at_utc": (activation + timedelta(minutes=245)).isoformat(),
        "entry_price": limit,
        "holding_minutes": 240,
    }
    fill = _s5(
        activation + timedelta(seconds=10),
        pair="USD_JPY",
        bid_open=150.006,
        ask_open=150.016,
        bid_high=150.007,
    )
    exit_at = activation + timedelta(minutes=240)
    exit_candle = _s5(
        exit_at,
        pair="USD_JPY",
        bid_open=149.98,
        ask_open=149.99,
    )
    outcome = resolve_signal(
        signal,
        [fill, exit_candle],
        policy_sha256="a" * 64,
        resolved_at_utc=activation + timedelta(minutes=246),
    )
    assert outcome["status"] == "FILLED_TIME_CLOSE"
    assert outcome["exit_price"] == 149.99
    assert outcome["realized_pips"] == 1.5
    assert outcome["gap_through_fill"] is True
    assert outcome["broker_mutation_allowed"] is False


def test_thirty_loss_prefix_stops_candidate_without_replacement() -> None:
    policy = load_policy(POLICY_PATH)
    decisions, outcomes = _synthetic_family_rows(
        policy,
        candidate_id=EXPLORATORY_ID,
        values=[-1.0] * 30,
    )
    scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256="a" * 64,
        as_of_utc=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    row = next(
        item
        for item in scorecard["candidate_scorecards"]
        if item["candidate_id"] == EXPLORATORY_ID
    )
    assert row["status"] == "FUTILITY_REJECTED_COLLECTION_STOPPED"
    assert row["interim_futility"]["rejected_at_sample_prefix"] == 30
    assert row["automatic_adoption_allowed"] is False
    assert scorecard["automatic_replacement_allowed"] is False
    assert scorecard["replacement_policy"] == (
        "NEW_VERSION_AND_UNTOUCHED_FUTURE_CUTOFF_REQUIRED"
    )


def test_final_decision_uses_fixed_first_100_and_never_late_revival_or_reversal() -> None:
    policy = load_policy(POLICY_PATH)
    decisions, outcomes = _synthetic_family_rows(
        policy,
        candidate_id=ANCHOR_ID,
        values=[2.0] * 100 + [-1000.0],
    )
    scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256="a" * 64,
        as_of_utc=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    row = next(
        item
        for item in scorecard["candidate_scorecards"]
        if item["candidate_id"] == ANCHOR_ID
    )
    assert row["filled_signal_count_total"] == 101
    assert row["evaluated_fixed_prefix_count"] == 100
    assert row["metrics"]["net_pips"] == 200.0
    assert row["status"] == "FINAL_EVIDENCE_PASSED_REVIEW_REQUIRED"
    assert row["prospective_gate_passed"] is True
    assert scorecard["familywise_correction"]["per_candidate_final_alpha"] == 0.025
    assert scorecard["primary_trading_candidate_allowed"] is False
    assert scorecard["promotion_allowed"] is False
    assert scorecard["live_permission"] is False


def test_failed_fixed_prefix_stays_terminal_even_if_later_rows_would_help() -> None:
    policy = load_policy(POLICY_PATH)
    decisions, outcomes = _synthetic_family_rows(
        policy,
        candidate_id=ANCHOR_ID,
        values=[-1.0] * 100 + [1000.0] * 20,
    )
    scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256="a" * 64,
        as_of_utc=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    row = next(
        item
        for item in scorecard["candidate_scorecards"]
        if item["candidate_id"] == ANCHOR_ID
    )
    assert row["evaluated_fixed_prefix_count"] == 100
    assert row["metrics"]["net_pips"] == -100.0
    assert row["status"] == "FINAL_EVIDENCE_REJECTED_COLLECTION_STOPPED"
    assert row["prospective_gate_passed"] is False
