from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.technical_forecast_forward_outcome import (
    S5BidAskCandle,
    append_forward_outcomes,
    build_forward_scorecard,
    load_forward_outcomes,
    load_forward_shadows,
    pending_forward_signals,
    resolve_frozen_forward_signal,
)
from quant_rabbit.technical_forecast_forward_shadow import (
    build_forward_shadow,
    load_forward_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "config" / "technical_forecast_forward_candidate_v1.json"
CANDIDATE_SHA = hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()


class TechnicalForecastForwardOutcomeTest(unittest.TestCase):
    def test_long_frozen_limit_fills_on_ask_and_takes_profit_on_bid(self) -> None:
        candidate, _shadow, task, decision = _first_task()
        candles = [
            # This candle starts before the 00:00:02 quote and must be ignored,
            # even though its range crosses both frozen exits.
            _bar(decision, bid=(1.0000, 1.0020, 0.9960, 1.0001)),
            _bar(
                decision + timedelta(seconds=5),
                bid=(0.9999, 1.0002, 0.9997, 1.0001),
            ),
            _bar(
                decision + timedelta(seconds=10),
                bid=(1.0001, 1.0016, 1.0000, 1.0015),
            ),
        ]

        outcome = resolve_frozen_forward_signal(
            candidate,
            task,
            candles,
            candidate_sha256=CANDIDATE_SHA,
            resolved_at_utc=decision + timedelta(days=1, seconds=10),
            truth_chunk_sha256=["a" * 64],
        )

        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(outcome["conservative_pips"], 15.0)
        self.assertFalse(outcome["promotion_allowed"])
        self.assertFalse(outcome["broker_mutation_allowed"])

    def test_unfilled_limit_is_zero_not_a_synthetic_trade(self) -> None:
        candidate, _shadow, task, decision = _first_task()
        candles = [
            _bar(
                decision + timedelta(seconds=5),
                bid=(1.0003, 1.0005, 1.0001, 1.0004),
            ),
        ]

        outcome = resolve_frozen_forward_signal(
            candidate,
            task,
            candles,
            candidate_sha256=CANDIDATE_SHA,
            resolved_at_utc=decision + timedelta(days=1, seconds=10),
            truth_chunk_sha256=["b" * 64],
        )

        self.assertFalse(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "UNFILLED_EXPIRED")
        self.assertEqual(outcome["conservative_pips"], 0.0)

    def test_same_s5_tp_sl_ordering_is_charged_as_full_stop(self) -> None:
        candidate, _shadow, task, decision = _first_task()
        candles = [
            _bar(
                decision + timedelta(seconds=5),
                bid=(1.0000, 1.0020, 0.9960, 1.0001),
            ),
        ]

        outcome = resolve_frozen_forward_signal(
            candidate,
            task,
            candles,
            candidate_sha256=CANDIDATE_SHA,
            resolved_at_utc=decision + timedelta(days=1, seconds=10),
            truth_chunk_sha256=["c" * 64],
        )

        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "AMBIGUOUS_TP_SL_ORDERING")
        self.assertEqual(outcome["conservative_pips"], -30.0)

    def test_open_at_horizon_is_charged_as_full_stop(self) -> None:
        candidate, _shadow, task, decision = _first_task()
        candles = [
            _bar(
                decision + timedelta(seconds=5),
                bid=(0.9999, 1.0002, 0.9997, 1.0001),
            ),
            _bar(
                decision + timedelta(hours=23),
                bid=(1.0001, 1.0004, 0.9998, 1.0002),
            ),
        ]

        outcome = resolve_frozen_forward_signal(
            candidate,
            task,
            candles,
            candidate_sha256=CANDIDATE_SHA,
            resolved_at_utc=decision + timedelta(days=1, seconds=10),
            truth_chunk_sha256=["d" * 64],
        )

        self.assertEqual(outcome["exit_reason"], "OPEN_UNRESOLVED")
        self.assertEqual(outcome["conservative_pips"], -30.0)

    def test_outcome_ledger_is_idempotent_and_tamper_evident(self) -> None:
        candidate, _shadow, task, decision = _first_task()
        outcome = resolve_frozen_forward_signal(
            candidate,
            task,
            [
                _bar(
                    decision + timedelta(seconds=5),
                    bid=(0.9999, 1.0002, 0.9997, 1.0001),
                ),
                _bar(
                    decision + timedelta(seconds=10),
                    bid=(1.0001, 1.0016, 1.0000, 1.0015),
                ),
            ],
            candidate_sha256=CANDIDATE_SHA,
            resolved_at_utc=decision + timedelta(days=1, seconds=10),
            truth_chunk_sha256=["e" * 64],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outcomes.jsonl"
            self.assertEqual(
                append_forward_outcomes(
                    path,
                    [outcome],
                    candidate_sha256=CANDIDATE_SHA,
                ),
                1,
            )
            self.assertEqual(
                append_forward_outcomes(
                    path,
                    [outcome],
                    candidate_sha256=CANDIDATE_SHA,
                ),
                0,
            )
            self.assertEqual(
                len(load_forward_outcomes(path, candidate_sha256=CANDIDATE_SHA)),
                1,
            )
            path.write_text(
                path.read_text().replace("TAKE_PROFIT", "STOP_LOSS"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_forward_outcomes(path, candidate_sha256=CANDIDATE_SHA)

    def test_shadow_ledger_rejects_post_recording_signal_edits(self) -> None:
        shadow = _shadow_for_day(0)
        shadow["signals"][0]["technical_score_pips"] = 999.0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadows.jsonl"
            path.write_text(
                json.dumps(shadow, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_forward_shadows(path, candidate_sha256=CANDIDATE_SHA)

    def test_scorecard_freezes_first_complete_positive_cohort(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)
        shadows = [_shadow_for_day(day) for day in range(20)]
        outcomes = []
        for day_index, shadow in enumerate(shadows):
            for signal in shadow["signals"]:
                positive = day_index < 17
                outcomes.append(
                    {
                        "signal_sha256": signal["signal_sha256"],
                        "decision_at_utc": shadow["decision_at_utc"],
                        "filled": True,
                        "exit_reason": "TAKE_PROFIT" if positive else "STOP_LOSS",
                        "conservative_pips": 15.0 if positive else -30.0,
                    }
                )

        scorecard = build_forward_scorecard(
            candidate,
            shadows,
            outcomes,
            candidate_sha256=CANDIDATE_SHA,
            as_of_utc=datetime(2026, 8, 6, 0, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(
            scorecard["status"],
            "FORWARD_EVIDENCE_PASSED_REVIEW_REQUIRED",
        )
        self.assertTrue(scorecard["forward_evidence_passed"])
        self.assertFalse(scorecard["promotion_allowed"])
        self.assertEqual(scorecard["metrics"]["mature_signals"], 51)
        self.assertEqual(scorecard["post_cohort_outcome_count"], 9)
        self.assertTrue(all(gate["passed"] for gate in scorecard["gates"]))


def _first_task():
    candidate = load_forward_candidate(CANDIDATE_PATH)
    shadow = _shadow_for_day(0, pair_count=1)
    decision = datetime.fromisoformat(shadow["decision_at_utc"])
    plan = pending_forward_signals(
        candidate,
        [shadow],
        [],
        as_of_utc=decision + timedelta(days=1, seconds=10),
    )
    return candidate, shadow, plan["selected"][0], decision


def _shadow_for_day(day: int, *, pair_count: int = 10):
    terminal = datetime(2026, 7, 15, 23, 55, tzinfo=timezone.utc) + timedelta(
        days=day
    )
    charts, quotes = _market(pair_count=pair_count, terminal=terminal)
    return build_forward_shadow(
        load_forward_candidate(CANDIDATE_PATH),
        charts,
        quotes,
        candidate_sha256=CANDIDATE_SHA,
        observed_at_utc=terminal + timedelta(minutes=5, seconds=5),
    )


def _market(*, pair_count: int, terminal: datetime):
    pairs = [
        "EUR_USD",
        "AUD_USD",
        "USD_CAD",
        "EUR_GBP",
        "AUD_CAD",
        "EUR_CHF",
        "USD_CHF",
        "GBP_USD",
        "NZD_USD",
        "USD_JPY",
    ][:pair_count]
    charts = []
    quotes = {}
    for index, pair in enumerate(pairs, start=1):
        pip_size = 0.01 if pair.endswith("JPY") else 0.0001
        start = 150.0 if pair.endswith("JPY") else 1.0
        signed_steps = -index if index % 2 == 0 else index
        candles = []
        for offset in range(13):
            candle_time = terminal - timedelta(minutes=5 * (12 - offset))
            close = start + signed_steps * pip_size * offset / 12.0
            candles.append(
                {
                    "t": candle_time.isoformat(),
                    "o": close,
                    "h": close,
                    "l": close,
                    "c": close,
                    "v": 100,
                    "complete": True,
                }
            )
        charts.append(
            {
                "pair": pair,
                "views": [{"granularity": "M5", "recent_candles": candles}],
            }
        )
        quotes[pair] = {
            "bid": start,
            "ask": start + pip_size,
            "timestamp_utc": (terminal + timedelta(minutes=5, seconds=2)).isoformat(),
        }
    return (
        {
            "generated_at_utc": (
                terminal + timedelta(minutes=5, seconds=3)
            ).isoformat(),
            "charts": charts,
        },
        quotes,
    )


def _bar(
    timestamp: datetime,
    *,
    bid: tuple[float, float, float, float],
    spread: float = 0.0001,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=timestamp,
        bid_o=bid[0],
        bid_h=bid[1],
        bid_l=bid[2],
        bid_c=bid[3],
        ask_o=bid[0] + spread,
        ask_h=bid[1] + spread,
        ask_l=bid[2] + spread,
        ask_c=bid[3] + spread,
    )


if __name__ == "__main__":
    unittest.main()
