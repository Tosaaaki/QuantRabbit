from __future__ import annotations

import copy
import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.strategy.failed_break_evidence import (
    build_m5_failed_break_evidence_from_candles,
    failed_break_for_side,
    verify_m5_failed_break_evidence,
)
from quant_rabbit.strategy.intent_generator import (
    _oanda_m5_failed_break,
    _oanda_m5_failed_break_evidence,
)


class FailedBreakEvidenceTests(unittest.TestCase):
    def _window(self, current: dict[str, float]) -> list[dict[str, object]]:
        start = datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)
        prior = [
            {
                "t": (start + timedelta(minutes=5 * index)).isoformat(),
                "o": 150.0,
                "h": 200.0,
                "l": 100.0,
                "c": 150.0,
                "complete": True,
            }
            for index in range(20)
        ]
        return [
            *prior,
            {
                "t": (start + timedelta(minutes=100)).isoformat(),
                **current,
                "complete": True,
            },
        ]

    @staticmethod
    def _rehash(value: dict[str, object]) -> None:
        body = {key: item for key, item in value.items() if key != "evidence_sha256"}
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        value["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()

    def test_long_requires_low_pierce_and_five_percent_reacceptance(self) -> None:
        evidence = build_m5_failed_break_evidence_from_candles(
            self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        )

        self.assertEqual(verify_m5_failed_break_evidence(evidence), (True, None))
        self.assertEqual(evidence["direction"], "LONG")
        self.assertEqual(evidence["prior_low"], 100.0)
        self.assertEqual(evidence["prior_width"], 100.0)
        self.assertEqual(evidence["acceptance_zone"], 100.0)
        self.assertEqual(failed_break_for_side(evidence["candles"], side="LONG"), (True, 100.0))
        self.assertEqual(failed_break_for_side(evidence["candles"], side="SHORT"), (False, None))

    def test_short_requires_high_pierce_and_five_percent_reacceptance(self) -> None:
        candles = self._window({"o": 190.0, "h": 201.0, "l": 140.0, "c": 194.0})
        evidence = build_m5_failed_break_evidence_from_candles(candles)

        self.assertEqual(verify_m5_failed_break_evidence(evidence), (True, None))
        self.assertEqual(evidence["direction"], "SHORT")
        self.assertEqual(evidence["acceptance_zone"], 200.0)
        self.assertEqual(_oanda_m5_failed_break_evidence(candles, side="SHORT"), (True, 200.0))
        self.assertTrue(_oanda_m5_failed_break(candles, side="SHORT"))
        self.assertFalse(_oanda_m5_failed_break(candles, side="LONG"))

    def test_threshold_is_strict_and_not_touch_inclusive(self) -> None:
        evidence = build_m5_failed_break_evidence_from_candles(
            self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 105.0})
        )

        self.assertEqual(evidence["direction"], "NONE")
        self.assertEqual(evidence["reason"], "NO_FAILED_BREAK")

    def test_both_sides_pierced_is_ambiguous_for_every_side(self) -> None:
        candles = self._window({"o": 150.0, "h": 201.0, "l": 99.0, "c": 150.0})
        evidence = build_m5_failed_break_evidence_from_candles(candles)

        self.assertEqual(evidence["direction"], "NONE")
        self.assertEqual(evidence["reason"], "BOTH_SIDES_AMBIGUOUS")
        self.assertEqual(_oanda_m5_failed_break_evidence(candles, side="LONG"), (False, None))
        self.assertEqual(_oanda_m5_failed_break_evidence(candles, side="SHORT"), (False, None))

    def test_last_complete_candle_is_current_and_only_prior_twenty_are_used(self) -> None:
        candles = self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        first_time = datetime.fromisoformat(str(candles[0]["t"])) - timedelta(minutes=5)
        candles.insert(
            0,
            {
                "t": first_time.isoformat(),
                "o": 500.0,
                "h": 900.0,
                "l": 1.0,
                "c": 500.0,
                "complete": True,
            },
        )
        candles.append(
            {
                "t": "2026-07-13T01:45:00+00:00",
                "o": 190.0,
                "h": 250.0,
                "l": 140.0,
                "c": 150.0,
                "complete": False,
            }
        )

        evidence = build_m5_failed_break_evidence_from_candles(candles)

        self.assertEqual(evidence["direction"], "LONG")
        self.assertEqual(evidence["prior_low"], 100.0)
        self.assertEqual(evidence["prior_high"], 200.0)
        self.assertEqual(len(evidence["candles"]), 21)
        self.assertEqual(evidence["candles"][-1]["t"], "2026-07-13T01:40:00Z")

    def test_gap_in_selected_complete_window_fails_closed(self) -> None:
        candles = self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        candles[10]["t"] = "2026-07-13T00:51:00+00:00"

        evidence = build_m5_failed_break_evidence_from_candles(candles)

        self.assertEqual(evidence["status"], "UNAVAILABLE")
        self.assertEqual(evidence["reason"], "M5_CANDLE_SEQUENCE_INVALID")
        self.assertEqual(verify_m5_failed_break_evidence(evidence), (True, None))
        self.assertEqual(failed_break_for_side(candles, side="LONG"), (False, None))

    def test_prices_that_round_to_zero_publish_verifiable_unavailable_evidence(self) -> None:
        candles = self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        for candle in candles:
            candle.update({"o": 0.00000000001, "h": 0.00000000001, "l": 0.00000000001, "c": 0.00000000001})

        evidence = build_m5_failed_break_evidence_from_candles(candles)

        self.assertEqual(evidence["status"], "UNAVAILABLE")
        self.assertEqual(evidence["reason"], "M5_CANDLE_INVALID")
        self.assertEqual(verify_m5_failed_break_evidence(evidence), (True, None))

    def test_verifier_recomputes_derivation_after_valid_rehash(self) -> None:
        evidence = build_m5_failed_break_evidence_from_candles(
            self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        )
        tampered = copy.deepcopy(evidence)
        tampered["direction"] = "SHORT"
        self._rehash(tampered)

        self.assertEqual(
            verify_m5_failed_break_evidence(tampered),
            (False, "M5_FAILED_BREAK_EVIDENCE_DERIVATION_MISMATCH"),
        )

    def test_unhashable_direction_is_rejected_instead_of_raising(self) -> None:
        evidence = build_m5_failed_break_evidence_from_candles(
            self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        )
        tampered = copy.deepcopy(evidence)
        tampered["direction"] = ["LONG"]
        self._rehash(tampered)

        self.assertEqual(
            verify_m5_failed_break_evidence(tampered),
            (False, "M5_FAILED_BREAK_EVIDENCE_DIRECTION_INVALID"),
        )

    def test_timestamp_free_legacy_wrapper_keeps_exact_one_side_semantics(self) -> None:
        timed = self._window({"o": 110.0, "h": 160.0, "l": 99.0, "c": 106.0})
        legacy = [{key: value for key, value in candle.items() if key != "t"} for candle in timed]

        self.assertEqual(_oanda_m5_failed_break_evidence(legacy, side="LONG"), (True, 100.0))
        self.assertEqual(_oanda_m5_failed_break_evidence(legacy, side="SHORT"), (False, None))

        legacy[-1].update({"o": 150.0, "h": 201.0, "l": 99.0, "c": 150.0})
        self.assertFalse(_oanda_m5_failed_break(legacy, side="LONG"))
        self.assertFalse(_oanda_m5_failed_break(legacy, side="SHORT"))


if __name__ == "__main__":
    unittest.main()
