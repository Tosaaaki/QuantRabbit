from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.fast_bot_truth import (
    SCORING_POLICY,
    build_fast_bot_scorecard,
    resolve_due_fast_bot_outcomes_from_oanda,
    resolve_fast_bot_signal,
)
from quant_rabbit.fast_bot import (
    ENTRY_EXPERIMENT_CONTRACT,
    SIGNAL_CONTRACT,
    _entry_experiment_arms,
    _shadow_geometry_pips,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _signal(*, signal_id: str = "signal-1", generated: datetime = NOW, side: str = "LONG") -> dict:
    if side == "LONG":
        entry, tp, sl = 1.1000, 1.1003, 1.0997
    else:
        entry, tp, sl = 1.1001, 1.0998, 1.1004
    body = {
        "contract": SIGNAL_CONTRACT,
        "schema_version": 1,
        "signal_id": hashlib.sha256(signal_id.encode("utf-8")).hexdigest()[:24],
        "pair": "EUR_USD",
        "side": side,
        "method": "RANGE_ROTATION",
        "m1_closed_candle_utc": (generated - timedelta(minutes=1)).isoformat(),
        "regime_contract_sha256": "b" * 64,
        "generated_at_utc": generated.isoformat(),
        "quote_timestamp_utc": generated.isoformat(),
        "order_type": "LIMIT",
        "entry_reference": "PASSIVE_NEAR_SIDE",
        "entry": entry,
        "take_profit": tp,
        "stop_loss": sl,
        "take_profit_pips": 3.0,
        "stop_loss_pips": 3.0,
        "reward_risk": 1.0,
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "attached_take_profit_required": True,
        "attached_stop_loss_required": True,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**body, "signal_sha256": hashlib.sha256(raw).hexdigest()}


def _signal_v2(
    *,
    signal_id: str = "signal-v2",
    generated: datetime = NOW,
    bid: float = 1.1000,
    ask: float = 1.1002,
) -> dict:
    spread = round((ask - bid) * 10000, 6)
    m5_atr = 5.0
    tp_pips, sl_pips = _shadow_geometry_pips(
        "RANGE_ROTATION",
        spread=spread,
        m5_atr=m5_atr,
    )
    arms = _entry_experiment_arms(
        pair="EUR_USD",
        side="LONG",
        bid=bid,
        ask=ask,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
    )
    primary = arms[0]
    body = {
        "contract": SIGNAL_CONTRACT,
        "schema_version": 2,
        "signal_id": hashlib.sha256(signal_id.encode("utf-8")).hexdigest()[:24],
        "pair": "EUR_USD",
        "side": "LONG",
        "method": "RANGE_ROTATION",
        "m1_closed_candle_utc": (generated - timedelta(minutes=1)).isoformat(),
        "regime_contract_sha256": "b" * 64,
        "generated_at_utc": generated.isoformat(),
        "quote_timestamp_utc": generated.isoformat(),
        "quote_bid": bid,
        "quote_ask": ask,
        "spread_pips": spread,
        "m5_atr_pips": m5_atr,
        "geometry_policy": "METHOD_SPREAD_M5_ATR_V1",
        "order_type": "LIMIT",
        "entry_reference": "PASSIVE_NEAR_SIDE",
        "entry": primary["entry"],
        "take_profit": primary["take_profit"],
        "stop_loss": primary["stop_loss"],
        "take_profit_pips": primary["take_profit_pips"],
        "stop_loss_pips": primary["stop_loss_pips"],
        "reward_risk": round(
            primary["take_profit_pips"] / primary["stop_loss_pips"],
            6,
        ),
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "entry_experiment_contract": ENTRY_EXPERIMENT_CONTRACT,
        "entry_experiment_arms": arms,
        "attached_take_profit_required": True,
        "attached_stop_loss_required": True,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _reseal_signal(body)


def _reseal_signal(body: dict) -> dict:
    unsealed = {key: value for key, value in body.items() if key != "signal_sha256"}
    raw = json.dumps(
        unsealed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**unsealed, "signal_sha256": hashlib.sha256(raw).hexdigest()}


def _reseal_outcome(body: dict) -> dict:
    unsealed = {key: value for key, value in body.items() if key != "contract_sha256"}
    raw = json.dumps(
        unsealed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**unsealed, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _candle(
    seconds: float,
    *,
    generated: datetime = NOW,
    bid_o: float = 1.1000,
    bid_h: float = 1.1000,
    bid_l: float = 1.0999,
    ask_o: float = 1.1002,
    ask_h: float = 1.1003,
    ask_l: float = 1.1002,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=generated + timedelta(seconds=seconds),
        bid_o=bid_o,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=1.1000,
        ask_o=ask_o,
        ask_h=ask_h,
        ask_l=ask_l,
        ask_c=ask_o,
    )


def _complete_truth(
    *candles: S5BidAskCandle,
    generated: datetime = NOW,
) -> list[S5BidAskCandle]:
    maturity = generated + timedelta(seconds=990)
    epoch_us = int(round(generated.timestamp() * 1_000_000))
    first_us = ((epoch_us + 4_999_999) // 5_000_000) * 5_000_000
    floor_us = int(round(maturity.timestamp() * 1_000_000)) // 5_000_000 * 5_000_000
    first = datetime.fromtimestamp(first_us / 1_000_000, tz=timezone.utc)
    expected = {
        first + timedelta(seconds=offset): _candle(
            (first + timedelta(seconds=offset) - generated).total_seconds(),
            generated=generated,
        )
        for offset in range(0, int((floor_us - first_us) / 1_000_000), 5)
    }
    expected.update({candle.timestamp_utc: candle for candle in candles})
    return [expected[timestamp] for timestamp in sorted(expected)]


class FastBotTruthTest(unittest.TestCase):
    def test_long_limit_uses_ask_to_fill_and_bid_to_take_profit(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(
                _candle(5, ask_l=1.0999),
                _candle(10, bid_h=1.10035, bid_l=1.0999),
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(outcome["realized_pips"], 3.0)
        self.assertFalse(outcome["live_permission"])
        self.assertFalse(outcome["broker_mutation"])

    def test_same_s5_touch_is_stop_first(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(_candle(5, ask_l=1.0999, bid_h=1.1004, bid_l=1.0996)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(outcome["exit_reason"], "STOP_LOSS_AMBIGUOUS_FILL_S5")
        self.assertEqual(outcome["realized_pips"], -3.0)
        self.assertTrue(outcome["ambiguous_same_s5"])

    def test_fill_candle_tp_only_is_conservatively_full_stop(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(_candle(5, ask_l=1.0999, bid_h=1.1004)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(outcome["exit_reason"], "STOP_LOSS_AMBIGUOUS_FILL_S5")
        self.assertEqual(outcome["realized_pips"], -3.0)
        self.assertTrue(outcome["ambiguous_same_s5"])

    def test_stop_gap_uses_executable_open_beyond_attached_stop(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(
                _candle(5, ask_l=1.0999),
                _candle(
                    10,
                    bid_o=1.0990,
                    bid_h=1.0991,
                    bid_l=1.0989,
                ),
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(outcome["exit_reason"], "STOP_LOSS_GAP")
        self.assertEqual(outcome["realized_pips"], -10.0)

    def test_gap_through_fill_candle_is_charged_at_executable_open(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(
                _candle(
                    5,
                    bid_o=1.0990,
                    bid_h=1.0991,
                    bid_l=1.0989,
                    ask_o=1.0991,
                    ask_h=1.0992,
                    ask_l=1.0990,
                ),
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(
            outcome["exit_reason"],
            "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
        )
        self.assertEqual(outcome["realized_pips"], -10.0)

    def test_ttl_straddling_s5_candle_cannot_fill(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(_candle(90, ask_l=1.0999)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertFalse(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "UNFILLED")

    def test_v2_precommitted_entry_arms_share_one_frozen_s5_path(self) -> None:
        signal = _signal_v2()
        outcome = resolve_fast_bot_signal(
            signal,
            _complete_truth(
                _candle(
                    5,
                    ask_l=1.10008,
                    bid_l=1.10005,
                ),
                _candle(10, bid_h=1.1008, bid_l=1.10005),
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )
        arms = {
            arm["arm_id"]: arm
            for arm in outcome["entry_experiment"]["arms"]
        }

        self.assertFalse(outcome["filled"])
        self.assertFalse(arms["PASSIVE_NEAR_SIDE"]["filled"])
        self.assertFalse(arms["PASSIVE_QUARTER_SPREAD"]["filled"])
        self.assertTrue(arms["PASSIVE_MID_SPREAD"]["filled"])
        self.assertTrue(arms["PASSIVE_THREE_QUARTER_SPREAD"]["filled"])
        self.assertEqual(arms["PASSIVE_MID_SPREAD"]["exit_reason"], "TAKE_PROFIT")
        self.assertFalse(
            outcome["entry_experiment"]["automatic_parameter_change_allowed"]
        )

        scorecard = build_fast_bot_scorecard(
            [signal],
            [outcome],
            as_of_utc=NOW + timedelta(days=1),
        )
        score_arms = {
            arm["arm_id"]: arm
            for arm in scorecard["entry_experiment"]["arms"]
        }
        self.assertEqual(scorecard["resolved_signals"], 1)
        self.assertEqual(scorecard["filled_signals"], 0)
        self.assertEqual(score_arms["PASSIVE_NEAR_SIDE"]["filled_signals"], 0)
        self.assertEqual(score_arms["PASSIVE_MID_SPREAD"]["filled_signals"], 1)
        self.assertEqual(score_arms["PASSIVE_MID_SPREAD"]["net_pips"], 6.0)
        self.assertEqual(
            score_arms["PASSIVE_MID_SPREAD"]["net_delta_pips_vs_near_side"],
            6.0,
        )
        self.assertEqual(
            score_arms["PASSIVE_MID_SPREAD"]["delta_inference_unit"],
            "PAIRED_SIGNAL_DAY",
        )
        self.assertFalse(
            scorecard["entry_experiment"]["automatic_parameter_change_allowed"]
        )
        self.assertIsNone(scorecard["entry_experiment"]["selected_arm_id"])

    def test_collapsed_entry_arms_cannot_become_review_ready(self) -> None:
        signals = []
        outcomes = []
        for index in range(100):
            generated = NOW - timedelta(days=index % 10, minutes=index // 10)
            signal = _signal_v2(
                signal_id=f"collapsed-{index}",
                generated=generated,
                bid=1.10000,
                ask=1.10001,
            )
            signals.append(signal)
            outcomes.append(
                resolve_fast_bot_signal(
                    signal,
                    _complete_truth(generated=generated),
                    resolved_at_utc=generated + timedelta(minutes=20),
                    truth_chunk_sha256=["a" * 64],
                )
            )

        experiment = build_fast_bot_scorecard(
            signals,
            outcomes,
            as_of_utc=NOW + timedelta(days=1),
        )["entry_experiment"]

        self.assertEqual(experiment["resolved_precommitted_signals"], 100)
        self.assertFalse(experiment["all_alternatives_have_100_distinct_ticks"])
        self.assertFalse(experiment["review_ready"])
        for arm in experiment["arms"][1:]:
            self.assertEqual(arm["distinct_from_near_side_signals"], 0)
            self.assertEqual(arm["entry_collapse_rate_vs_near_side"], 1.0)

    def test_v2_resealed_arm_tampering_is_rejected(self) -> None:
        signal = _signal_v2()
        signal["entry_experiment_arms"][1]["entry"] += 0.00001
        tampered = _reseal_signal(signal)

        with self.assertRaisesRegex(ValueError, "invalid fast-bot signal"):
            resolve_fast_bot_signal(
                tampered,
                _complete_truth(_candle(5)),
                resolved_at_utc=NOW + timedelta(minutes=20),
            )

    def test_resealed_outcome_arm_profit_forgery_is_excluded(self) -> None:
        signal = _signal_v2()
        outcome = resolve_fast_bot_signal(
            signal,
            _complete_truth(),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )
        outcome["entry_experiment"]["arms"][1]["realized_pips"] = 999.0
        forged = _reseal_outcome(outcome)

        scorecard = build_fast_bot_scorecard(
            [signal],
            [forged],
            as_of_utc=NOW + timedelta(days=1),
        )

        self.assertEqual(scorecard["resolved_signals"], 0)
        self.assertEqual(
            scorecard["entry_experiment"]["resolved_precommitted_signals"],
            0,
        )

    def test_filled_unresolved_horizon_is_full_stop_not_market_close(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(_candle(5, ask_l=1.0999)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(outcome["exit_reason"], "HORIZON_FULL_STOP_LOSS")
        self.assertEqual(outcome["realized_pips"], -3.0)

    def test_forward_scorecard_can_pass_but_never_grants_live_permission(self) -> None:
        signals = []
        outcomes = []
        for index in range(100):
            generated = (
                NOW
                - timedelta(days=index % 10)
                - timedelta(minutes=index // 10)
            )
            signal = _signal(signal_id=f"signal-{index}", generated=generated)
            shifted = _complete_truth(
                _candle(5, generated=generated, ask_l=1.0999),
                _candle(10, generated=generated, bid_h=1.1004),
                generated=generated,
            )
            signals.append(signal)
            outcomes.append(
                resolve_fast_bot_signal(
                    signal,
                    shifted,
                    resolved_at_utc=generated + timedelta(minutes=20),
                    truth_chunk_sha256=["a" * 64],
                )
            )
        scorecard = build_fast_bot_scorecard(
            signals,
            outcomes,
            as_of_utc=NOW + timedelta(days=1),
        )

        self.assertTrue(scorecard["forward_evidence_passed"])
        self.assertEqual(scorecard["filled_signals"], 100)
        self.assertFalse(scorecard["live_permission"])
        self.assertFalse(scorecard["promotion_allowed"])
        self.assertEqual(
            scorecard["promotion_blockers"],
            [
                "OVERLAPPING_AI_TRADER_ENTRY_AUTHORITY_RETIREMENT_REQUIRED",
                "SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED",
            ],
        )

    def test_unresolved_signal_days_cannot_inflate_filled_active_days(self) -> None:
        signals = []
        outcomes = []
        for index in range(100):
            generated = NOW - timedelta(minutes=index)
            signal = _signal(signal_id=f"filled-{index}", generated=generated)
            signals.append(signal)
            outcomes.append(
                resolve_fast_bot_signal(
                    signal,
                    _complete_truth(
                        _candle(5, generated=generated, ask_l=1.0999),
                        _candle(10, generated=generated, bid_h=1.1004),
                        generated=generated,
                    ),
                    resolved_at_utc=generated + timedelta(minutes=20),
                    truth_chunk_sha256=["a" * 64],
                )
            )
        signals.extend(
            _signal(
                signal_id=f"unresolved-day-{day}",
                generated=NOW - timedelta(days=day),
            )
            for day in range(1, 10)
        )

        scorecard = build_fast_bot_scorecard(
            signals,
            outcomes,
            as_of_utc=NOW + timedelta(days=1),
        )

        self.assertEqual(scorecard["filled_signals"], 100)
        self.assertEqual(scorecard["active_days"], 1)
        self.assertFalse(scorecard["forward_evidence_passed"])

    def test_expectancy_lower_bound_uses_filled_day_blocks(self) -> None:
        signals = []
        outcomes = []
        for index in range(100):
            day = index // 10
            generated = NOW - timedelta(days=day, minutes=index % 10)
            signal = _signal(signal_id=f"daily-{index}", generated=generated)
            winning_day = day < 6
            candles = _complete_truth(
                _candle(5, generated=generated, ask_l=1.0999),
                _candle(
                    10,
                    generated=generated,
                    **({"bid_h": 1.1004} if winning_day else {"bid_l": 1.0996}),
                ),
                generated=generated,
            )
            signals.append(signal)
            outcomes.append(
                resolve_fast_bot_signal(
                    signal,
                    candles,
                    resolved_at_utc=generated + timedelta(minutes=20),
                    truth_chunk_sha256=["a" * 64],
                )
            )

        scorecard = build_fast_bot_scorecard(
            signals,
            outcomes,
            as_of_utc=NOW + timedelta(days=1),
        )

        self.assertEqual(scorecard["filled_signals"], 100)
        self.assertEqual(scorecard["profit_factor"], 1.5)
        self.assertEqual(scorecard["expectancy_inference_unit"], "FILLED_SIGNAL_DAY")
        self.assertLess(scorecard["one_sided_95_daily_mean_lower_pips"], 0.0)
        self.assertFalse(scorecard["forward_evidence_passed"])

    def test_same_pair_m1_identity_is_counted_once(self) -> None:
        first = _signal(signal_id="first")
        second_body = {
            **_signal(signal_id="second", generated=NOW + timedelta(seconds=30)),
            "m1_closed_candle_utc": first["m1_closed_candle_utc"],
        }
        second_body.pop("signal_sha256")
        raw = json.dumps(second_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        second = {**second_body, "signal_sha256": hashlib.sha256(raw).hexdigest()}
        first_outcome = resolve_fast_bot_signal(
            first,
            _complete_truth(_candle(5, ask_l=1.0999, bid_h=1.1004)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        scorecard = build_fast_bot_scorecard(
            [first, second],
            [first_outcome],
            as_of_utc=NOW + timedelta(days=1),
        )

        self.assertEqual(scorecard["emitted_signals"], 1)
        self.assertEqual(scorecard["resolved_signals"], 1)
        self.assertEqual(scorecard["duplicate_identity_signals_ignored"], 1)
        self.assertIsNone(scorecard["one_sided_95_mean_lower_pips"])

    def test_no_due_signals_does_not_open_oanda_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = resolve_due_fast_bot_outcomes_from_oanda(
                shadow_ledger_path=root / "missing-shadow.jsonl",
                outcome_ledger_path=root / "outcomes.jsonl",
                scorecard_path=root / "scorecard.json",
                client_factory=lambda: (_ for _ in ()).throw(AssertionError("must not open client")),
                clock=lambda: NOW,
            )

        self.assertEqual(result["status"], "NO_DUE_SIGNALS")
        self.assertFalse(result["broker_read"])

    def test_tampered_signal_cannot_be_resolved_or_counted(self) -> None:
        signal = _signal()
        outcome = resolve_fast_bot_signal(
            signal,
            _complete_truth(_candle(5, ask_l=1.0999, bid_h=1.1004)),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )
        tampered = {**signal, "take_profit_pips": 30.0}

        with self.assertRaisesRegex(ValueError, "invalid fast-bot signal"):
            resolve_fast_bot_signal(
                tampered,
                _complete_truth(_candle(5, ask_l=1.0999, bid_h=1.1004)),
                resolved_at_utc=NOW + timedelta(minutes=20),
            )
        scorecard = build_fast_bot_scorecard(
            [tampered],
            [outcome],
            as_of_utc=NOW + timedelta(days=1),
        )
        self.assertEqual(scorecard["emitted_signals"], 0)
        self.assertEqual(scorecard["resolved_signals"], 0)

    def test_off_grid_truth_cannot_be_scored_as_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid fast-bot S5 truth grid coverage"):
            resolve_fast_bot_signal(
                _signal(),
                [_candle(6)],
                resolved_at_utc=NOW + timedelta(minutes=20),
                truth_chunk_sha256=["a" * 64],
            )

    def test_receipted_internal_s5_gap_is_preserved_as_no_tick(self) -> None:
        truth = _complete_truth()
        del truth[40]

        outcome = resolve_fast_bot_signal(
            _signal(),
            truth,
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertEqual(outcome["truth_grid_slot_count"], 198)
        self.assertEqual(outcome["truth_candle_count"], 197)
        self.assertEqual(outcome["truth_no_tick_slot_count"], 1)

    def test_empty_truth_hash_evidence_cannot_create_an_outcome(self) -> None:
        with self.assertRaisesRegex(ValueError, "truth chunk SHA-256"):
            resolve_fast_bot_signal(
                _signal(),
                _complete_truth(),
                resolved_at_utc=NOW + timedelta(minutes=20),
            )

    def test_resealed_bad_timing_or_truth_hash_is_excluded(self) -> None:
        signal = _signal()
        outcome = resolve_fast_bot_signal(
            signal,
            _complete_truth(
                _candle(5, ask_l=1.0999),
                _candle(10, bid_h=1.1004),
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )
        bad_rows = []
        bad_hash = {**outcome, "truth_chunk_sha256": ["not-a-sha"]}
        bad_rows.append(_reseal_outcome(bad_hash))
        bad_time = {
            **outcome,
            "fill_at_utc": (NOW + timedelta(seconds=6)).isoformat(),
        }
        bad_rows.append(_reseal_outcome(bad_time))
        for bad in bad_rows:
            with self.subTest(defect=bad["contract_sha256"]):
                scorecard = build_fast_bot_scorecard(
                    [signal],
                    [bad],
                    as_of_utc=NOW + timedelta(days=1),
                )
                self.assertEqual(scorecard["resolved_signals"], 0)

    def test_old_or_invalid_policy_outcome_is_re_resolved(self) -> None:
        signal = _signal(generated=NOW - timedelta(minutes=20))
        truth = _complete_truth(generated=NOW - timedelta(minutes=20))
        valid = resolve_fast_bot_signal(
            signal,
            truth,
            resolved_at_utc=NOW,
            truth_chunk_sha256=["a" * 64],
        )
        old_body = {
            key: value
            for key, value in valid.items()
            if key not in {"contract_sha256", "scoring_policy"}
        }
        old = _reseal_outcome(old_body)
        invalid_current = _reseal_outcome({**valid, "realized_pips": 999.0})

        for prior in (old, invalid_current):
            with self.subTest(prior_policy=prior.get("scoring_policy")):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    shadow_path = root / "shadow.jsonl"
                    outcome_path = root / "outcomes.jsonl"
                    scorecard_path = root / "scorecard.json"
                    shadow_path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
                    outcome_path.write_text(json.dumps(prior) + "\n", encoding="utf-8")
                    with patch(
                        "quant_rabbit.fast_bot_truth.fetch_frozen_s5_truth",
                        return_value=(truth, ["b" * 64]),
                    ):
                        result = resolve_due_fast_bot_outcomes_from_oanda(
                            shadow_ledger_path=shadow_path,
                            outcome_ledger_path=outcome_path,
                            scorecard_path=scorecard_path,
                            client_factory=object,
                            clock=lambda: NOW,
                        )
                    rows = [
                        json.loads(line)
                        for line in outcome_path.read_text().splitlines()
                    ]
                    scorecard = json.loads(scorecard_path.read_text())

                self.assertEqual(result["ledger_appended"], 1)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1]["scoring_policy"], SCORING_POLICY)
                self.assertEqual(scorecard["resolved_signals"], 1)

    def test_due_rotation_does_not_starve_the_thirteenth_failed_signal(self) -> None:
        signals = [
            _signal(
                signal_id=f"due-{index}",
                generated=NOW - timedelta(minutes=20 + index),
            )
            for index in range(13)
        ]
        attempted: set[str] = set()

        def fail_truth(*_args, **kwargs):
            attempted.add(kwargs["time_from"].isoformat())
            raise ValueError("incomplete fast-bot S5 truth coverage")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shadow_path = root / "shadow.jsonl"
            shadow_path.write_text(
                "".join(json.dumps(signal) + "\n" for signal in signals),
                encoding="utf-8",
            )
            kwargs = dict(
                shadow_ledger_path=shadow_path,
                outcome_ledger_path=root / "outcomes.jsonl",
                scorecard_path=root / "scorecard.json",
                client_factory=object,
            )
            with patch(
                "quant_rabbit.fast_bot_truth.fetch_frozen_s5_truth",
                side_effect=fail_truth,
            ):
                resolve_due_fast_bot_outcomes_from_oanda(
                    **kwargs,
                    clock=lambda: NOW,
                )
                resolve_due_fast_bot_outcomes_from_oanda(
                    **kwargs,
                    clock=lambda: NOW + timedelta(seconds=30),
                )

        self.assertEqual(len(attempted), 13)

    def test_non_grid_signal_requires_the_exact_aligned_s5_grid(self) -> None:
        generated = NOW + timedelta(microseconds=54940)
        signal = _signal(generated=generated)

        outcome = resolve_fast_bot_signal(
            signal,
            _complete_truth(
                _candle(5, ask_l=1.0999),
                generated=generated,
            ),
            resolved_at_utc=NOW + timedelta(minutes=20),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "HORIZON_FULL_STOP_LOSS")

    def test_immature_signal_cannot_be_scored(self) -> None:
        with self.assertRaisesRegex(ValueError, "fast-bot signal is not mature"):
            resolve_fast_bot_signal(
                _signal(),
                _complete_truth(_candle(5)),
                resolved_at_utc=NOW + timedelta(minutes=5),
                truth_chunk_sha256=["a" * 64],
            )


if __name__ == "__main__":
    unittest.main()
