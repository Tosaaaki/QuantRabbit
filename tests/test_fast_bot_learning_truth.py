from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.fast_bot_learning import build_fast_bot_learning_shadow
from quant_rabbit.fast_bot_learning_truth import (
    build_fast_bot_learning_scorecard,
    resolve_due_fast_bot_learning_outcomes_from_oanda,
    resolve_fast_bot_learning_seat,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
HASH = "a" * 64
METHODS = ("BREAKOUT_FAILURE", "RANGE_ROTATION", "TREND_CONTINUATION")
SIDES = ("LONG", "SHORT")


def _sha(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(body: dict) -> dict:
    return {**body, "contract_sha256": _sha(body)}


def _six_candidate_seat() -> dict:
    snapshot = {
        "fetched_at_utc": NOW.isoformat(),
        "quotes": {
            "EUR_USD": {
                "bid": 1.10000,
                "ask": 1.10008,
                "timestamp_utc": NOW.isoformat(),
            }
        },
        "positions": [],
        "orders": [],
    }
    rows = [
        {
            "pair": "EUR_USD",
            "side": side,
            "method": method,
            "state": "CAUTION",
            "score": 1.0,
            "execution_enabled": False,
            "hard_blockers": [],
            "caution_reasons": ["TRIGGER_NOT_READY"],
            "m1_closed_candle_utc": NOW.isoformat(),
            "m5_atr_pips": 5.0,
            "spread_pips": 0.8,
            "spread_to_m5_atr": 0.16,
            "failed_break_direction": "NONE",
        }
        for method in METHODS
        for side in SIDES
    ]
    regime = _seal(
        {
            "contract": "QR_HIERARCHICAL_BOT_REGIME_V1",
            "schema_version": 1,
            "generated_at_utc": NOW.isoformat(),
            "rows": rows,
            "sources": {"broker_snapshot_sha256": _sha(snapshot)},
        }
    )
    shadow = build_fast_bot_learning_shadow(regime, snapshot, now_utc=NOW)
    seat = shadow["seats"][0]
    if len(seat["candidates"]) != 6:  # make failures explain the producer contract
        raise AssertionError("test requires all six pair/side/method candidates")
    return seat


def _path() -> list[S5BidAskCandle]:
    return [
        S5BidAskCandle(
            timestamp_utc=NOW + timedelta(seconds=5),
            bid_o=1.10000,
            bid_h=1.10020,
            bid_l=1.09990,
            bid_c=1.10005,
            ask_o=1.10008,
            ask_h=1.10028,
            ask_l=1.09998,
            ask_c=1.10013,
        ),
        S5BidAskCandle(
            timestamp_utc=NOW + timedelta(seconds=10),
            bid_o=1.10005,
            bid_h=1.10045,
            bid_l=1.09995,
            bid_c=1.10035,
            ask_o=1.10013,
            ask_h=1.10053,
            ask_l=1.10003,
            ask_c=1.10043,
        ),
    ]


class FastBotLearningTruthTest(unittest.TestCase):
    def test_one_frozen_fetch_scores_six_candidates_and_all_eight_arms(self) -> None:
        seat = _six_candidate_seat()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shadow = root / "shadow.jsonl"
            outcome = root / "outcome.jsonl"
            scorecard = root / "scorecard.json"
            shadow.write_text(json.dumps(seat, sort_keys=True) + "\n", encoding="utf-8")
            with patch(
                "quant_rabbit.fast_bot_learning_truth.fetch_frozen_s5_truth",
                return_value=(_path(), [HASH]),
            ) as fetch:
                result = resolve_due_fast_bot_learning_outcomes_from_oanda(
                    shadow_ledger_path=shadow,
                    outcome_ledger_path=outcome,
                    scorecard_path=scorecard,
                    client_factory=object,
                    clock=lambda: NOW + timedelta(minutes=32),
                )
            self.assertEqual(result["status"], "RESOLVED")
            self.assertEqual(result["ledger_appended"], 1)
            self.assertEqual(fetch.call_count, 1)
            row = json.loads(outcome.read_text(encoding="utf-8"))
            self.assertEqual(len(row["candidates"]), 6)
            self.assertTrue(all(len(item["arms"]) == 8 for item in row["candidates"]))
            paths = {
                arm["truth_path_sha256"]
                for item in row["candidates"]
                for arm in item["arms"]
            }
            hashes = {
                tuple(arm["truth_chunk_sha256"])
                for item in row["candidates"]
                for arm in item["arms"]
            }
            self.assertEqual(paths, {row["truth_path_sha256"]})
            self.assertEqual(hashes, {(HASH,)})

    def test_long_short_use_executable_sides_and_record_mfe_mae(self) -> None:
        row = resolve_fast_bot_learning_seat(
            _six_candidate_seat(),
            _path(),
            resolved_at_utc=NOW + timedelta(minutes=32),
            truth_chunk_sha256=[HASH],
        )
        long_base = next(
            arm
            for candidate in row["candidates"]
            if candidate["side"] == "LONG"
            for arm in candidate["arms"]
            if arm["arm_id"] == "BASE"
        )
        short_base = next(
            arm
            for candidate in row["candidates"]
            if candidate["side"] == "SHORT"
            for arm in candidate["arms"]
            if arm["arm_id"] == "BASE"
        )
        self.assertTrue(long_base["filled"])
        self.assertTrue(short_base["filled"])
        self.assertGreaterEqual(long_base["maximum_favorable_excursion_pips"], 0.0)
        self.assertGreater(long_base["maximum_adverse_excursion_pips"], 0.0)
        self.assertGreaterEqual(short_base["maximum_favorable_excursion_pips"], 0.0)
        self.assertGreater(short_base["maximum_adverse_excursion_pips"], 0.0)
        self.assertEqual(long_base["time_to_fill_seconds"], 5.0)
        self.assertEqual(short_base["time_to_fill_seconds"], 5.0)
        self.assertEqual(long_base["post_cost_realized_pips"], long_base["realized_pips"])
        self.assertEqual(short_base["post_cost_realized_pips"], short_base["realized_pips"])

    def test_same_s5_fill_and_attached_exit_is_conservative_stop_first(self) -> None:
        candle = S5BidAskCandle(
            timestamp_utc=NOW + timedelta(seconds=5),
            bid_o=1.10000,
            bid_h=1.10100,
            bid_l=1.09900,
            bid_c=1.10000,
            ask_o=1.10008,
            ask_h=1.10108,
            ask_l=1.09908,
            ask_c=1.10008,
        )
        row = resolve_fast_bot_learning_seat(
            _six_candidate_seat(),
            [candle],
            resolved_at_utc=NOW + timedelta(minutes=32),
            truth_chunk_sha256=[HASH],
        )
        bases = [
            arm
            for candidate in row["candidates"]
            for arm in candidate["arms"]
            if arm["arm_id"] == "BASE"
        ]
        self.assertEqual(len(bases), 6)
        self.assertTrue(
            all("AMBIGUOUS_FILL_S5" in arm["exit_reason"] for arm in bases)
        )
        self.assertTrue(all(arm["post_cost_realized_pips"] < 0.0 for arm in bases))
        self.assertTrue(all(arm["maximum_favorable_excursion_pips"] == 0.0 for arm in bases))

    def test_tampered_candidate_or_arm_geometry_is_rejected(self) -> None:
        seat = _six_candidate_seat()
        tampered = json.loads(json.dumps(seat))
        tampered["candidates"][0]["arms"][0]["entry"] += 0.00001
        # Re-sealing only the seat cannot repair the candidate/arm commitment.
        body = {key: value for key, value in tampered.items() if key != "contract_sha256"}
        tampered["contract_sha256"] = _sha(body)
        with self.assertRaisesRegex(ValueError, "invalid fast-bot learning seat"):
            resolve_fast_bot_learning_seat(
                tampered,
                _path(),
                resolved_at_utc=NOW + timedelta(minutes=32),
                truth_chunk_sha256=[HASH],
            )

    def test_duplicate_is_idempotent_and_does_not_repeat_broker_read(self) -> None:
        seat = _six_candidate_seat()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shadow = root / "shadow.jsonl"
            outcome = root / "outcome.jsonl"
            scorecard = root / "scorecard.json"
            shadow.write_text(json.dumps(seat, sort_keys=True) + "\n", encoding="utf-8")
            with patch(
                "quant_rabbit.fast_bot_learning_truth.fetch_frozen_s5_truth",
                return_value=(_path(), [HASH]),
            ) as fetch:
                first = resolve_due_fast_bot_learning_outcomes_from_oanda(
                    shadow_ledger_path=shadow,
                    outcome_ledger_path=outcome,
                    scorecard_path=scorecard,
                    client_factory=object,
                    clock=lambda: NOW + timedelta(minutes=32),
                )
                second = resolve_due_fast_bot_learning_outcomes_from_oanda(
                    shadow_ledger_path=shadow,
                    outcome_ledger_path=outcome,
                    scorecard_path=scorecard,
                    client_factory=object,
                    clock=lambda: NOW + timedelta(minutes=33),
                )
            self.assertEqual(first["ledger_appended"], 1)
            self.assertEqual(second["status"], "NO_DUE_SEATS")
            self.assertFalse(second["broker_read"])
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(len(outcome.read_text(encoding="utf-8").splitlines()), 1)

    def test_malformed_duplicate_current_policy_fails_before_broker_read(self) -> None:
        seat = _six_candidate_seat()
        valid = resolve_fast_bot_learning_seat(
            seat,
            _path(),
            resolved_at_utc=NOW + timedelta(minutes=32),
            truth_chunk_sha256=[HASH],
        )
        malformed = {**valid, "candidate_count": 999}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shadow = root / "shadow.jsonl"
            outcome = root / "outcome.jsonl"
            scorecard = root / "scorecard.json"
            shadow.write_text(json.dumps(seat, sort_keys=True) + "\n", encoding="utf-8")
            outcome.write_text(
                json.dumps(valid, sort_keys=True) + "\n" + json.dumps(malformed, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with patch(
                "quant_rabbit.fast_bot_learning_truth.fetch_frozen_s5_truth"
            ) as fetch:
                result = resolve_due_fast_bot_learning_outcomes_from_oanda(
                    shadow_ledger_path=shadow,
                    outcome_ledger_path=outcome,
                    scorecard_path=scorecard,
                    client_factory=object,
                    clock=lambda: NOW + timedelta(minutes=33),
                )
            self.assertEqual(result["status"], "OUTCOME_IDENTITY_CONFLICT")
            self.assertEqual(result["outcome_identity_conflict_count"], 1)
            self.assertFalse(result["broker_read"])
            fetch.assert_not_called()

    def test_scorecard_preserves_groups_paired_delta_and_shadow_safety(self) -> None:
        seat = _six_candidate_seat()
        outcome = resolve_fast_bot_learning_seat(
            seat,
            _path(),
            resolved_at_utc=NOW + timedelta(minutes=32),
            truth_chunk_sha256=[HASH],
        )
        scorecard = build_fast_bot_learning_scorecard(
            [seat], [outcome], as_of_utc=NOW + timedelta(minutes=32)
        )
        self.assertEqual(
            scorecard["grouping_dimensions"],
            [
                "candidate_class",
                "cost_pressure_bucket",
                "pair",
                "side",
                "method",
                "horizon_lane",
                "arm_id",
            ],
        )
        self.assertTrue(scorecard["groups"])
        self.assertTrue(
            all(group["paired_count_vs_base"] == 1 for group in scorecard["groups"])
        )
        self.assertTrue(
            any(
                group["arm_id"] != "BASE"
                and group["mean_paired_delta_pips_vs_base"] is not None
                for group in scorecard["groups"]
            )
        )
        for item in (outcome, scorecard):
            self.assertFalse(item["automatic_promotion_allowed"])
            self.assertFalse(item["primary_effect"])
            self.assertFalse(item["risk_effect"])
            self.assertEqual(item["order_authority"], "NONE")
            self.assertTrue(item["shadow_only"])
            self.assertFalse(item["live_permission"])
            self.assertFalse(item["broker_mutation_allowed"])


if __name__ == "__main__":
    unittest.main()
