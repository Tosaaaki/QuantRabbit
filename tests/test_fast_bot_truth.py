from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot_truth import (
    build_fast_bot_scorecard,
    resolve_due_fast_bot_outcomes_from_oanda,
    resolve_fast_bot_signal,
)
from quant_rabbit.fast_bot import SIGNAL_CONTRACT
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


def _candle(
    seconds: int,
    *,
    bid_h: float = 1.1001,
    bid_l: float = 1.0999,
    ask_h: float = 1.1002,
    ask_l: float = 1.1000,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=NOW + timedelta(seconds=seconds),
        bid_o=1.1000,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=1.1000,
        ask_o=1.1001,
        ask_h=ask_h,
        ask_l=ask_l,
        ask_c=1.1001,
    )


def _complete_truth(*candles: S5BidAskCandle) -> list[S5BidAskCandle]:
    return [*candles, _candle(985)]


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
        )

        self.assertEqual(outcome["exit_reason"], "STOP_LOSS_AMBIGUOUS_SAME_S5")
        self.assertEqual(outcome["realized_pips"], -3.0)
        self.assertTrue(outcome["ambiguous_same_s5"])

    def test_filled_unresolved_horizon_is_full_stop_not_market_close(self) -> None:
        outcome = resolve_fast_bot_signal(
            _signal(),
            _complete_truth(_candle(5, ask_l=1.0999)),
            resolved_at_utc=NOW + timedelta(minutes=20),
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
            shifted = [
                S5BidAskCandle(
                    timestamp_utc=generated + timedelta(seconds=5),
                    bid_o=1.1000,
                    bid_h=1.1004,
                    bid_l=1.0999,
                    bid_c=1.1003,
                    ask_o=1.1001,
                    ask_h=1.1005,
                    ask_l=1.0999,
                    ask_c=1.1004,
                ),
                S5BidAskCandle(
                    timestamp_utc=generated + timedelta(seconds=985),
                    bid_o=1.1000,
                    bid_h=1.1001,
                    bid_l=1.0999,
                    bid_c=1.1000,
                    ask_o=1.1001,
                    ask_h=1.1002,
                    ask_l=1.1000,
                    ask_c=1.1001,
                ),
            ]
            signals.append(signal)
            outcomes.append(
                resolve_fast_bot_signal(
                    signal,
                    shifted,
                    resolved_at_utc=generated + timedelta(minutes=20),
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
            ["SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED"],
        )

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

    def test_incomplete_truth_cannot_be_scored_as_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete fast-bot S5 truth coverage"):
            resolve_fast_bot_signal(
                _signal(),
                [_candle(5)],
                resolved_at_utc=NOW + timedelta(minutes=20),
            )

    def test_complete_candle_end_can_satisfy_nine_second_tail_coverage(self) -> None:
        generated = NOW + timedelta(microseconds=54940)
        signal = _signal(generated=generated)

        outcome = resolve_fast_bot_signal(
            signal,
            [
                _candle(5, ask_l=1.0999),
                _candle(980),
            ],
            resolved_at_utc=NOW + timedelta(minutes=20),
        )

        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "HORIZON_FULL_STOP_LOSS")

    def test_immature_signal_cannot_be_scored(self) -> None:
        with self.assertRaisesRegex(ValueError, "fast-bot signal is not mature"):
            resolve_fast_bot_signal(
                _signal(),
                _complete_truth(_candle(5)),
                resolved_at_utc=NOW + timedelta(minutes=5),
            )


if __name__ == "__main__":
    unittest.main()
