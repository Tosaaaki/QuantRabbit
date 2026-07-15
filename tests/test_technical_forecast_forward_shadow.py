from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.technical_forecast_forward_shadow import (
    append_shadow_once,
    build_forward_shadow,
    forward_collection_window,
    load_forward_candidate,
    validate_forward_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "config" / "technical_forecast_forward_candidate_v1.json"
CANDIDATE_SHA = hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()


class TechnicalForecastForwardShadowTest(unittest.TestCase):
    def test_candidate_discloses_holdout_and_cannot_enable_live_orders(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)

        self.assertEqual(validate_forward_candidate(candidate), ())
        self.assertFalse(candidate["live_order_enabled"])
        self.assertFalse(candidate["promotion_allowed"])
        self.assertTrue(
            candidate["selection_disclosure"]["initial_holdout_inspected_before_lock"]
        )

        candidate["live_order_enabled"] = True
        self.assertIn("live_order_enabled must be false", validate_forward_candidate(candidate))

    def test_due_snapshot_selects_strongest_thirty_percent_with_fixed_geometry(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)
        charts, quotes = _market(pair_count=10)
        observed = datetime(2026, 7, 16, 0, 0, 5, tzinfo=timezone.utc)

        shadow = build_forward_shadow(
            candidate,
            charts,
            quotes,
            candidate_sha256=CANDIDATE_SHA,
            observed_at_utc=observed,
        )

        self.assertEqual(shadow["status"], "EMITTED")
        self.assertEqual(shadow["selected_pair_count"], 3)
        self.assertEqual(
            [row["pair"] for row in shadow["signals"]],
            ["USD_JPY", "NZD_USD", "GBP_USD"],
        )
        strongest = shadow["signals"][0]
        self.assertEqual(strongest["predicted_direction"], "DOWN")
        self.assertEqual(strongest["side"], "SHORT")
        self.assertEqual(strongest["entry_ttl_min"], 5)
        self.assertEqual(strongest["take_profit_pips"], 15.0)
        self.assertEqual(strongest["stop_loss_pips"], 30.0)
        self.assertFalse(strongest["live_ready"])
        self.assertEqual(shadow["order_intents"], [])
        self.assertFalse(shadow["broker_mutation_allowed"])

    def test_missed_decision_window_emits_no_signal(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)
        charts, quotes = _market(
            pair_count=2,
            terminal=datetime(2026, 7, 17, 22, 55, tzinfo=timezone.utc),
        )

        shadow = build_forward_shadow(
            candidate,
            charts,
            quotes,
            candidate_sha256=CANDIDATE_SHA,
            observed_at_utc=datetime(2026, 7, 17, 23, 0, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(shadow["status"], "DECISION_WINDOW_MISSED")
        self.assertEqual(shadow["signals"], [])

    def test_collection_window_is_open_for_only_sixty_seconds(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)

        open_window = forward_collection_window(
            candidate,
            datetime(2026, 7, 16, 0, 0, 59, tzinfo=timezone.utc),
        )
        missed = forward_collection_window(
            candidate,
            datetime(2026, 7, 16, 0, 1, 1, tzinfo=timezone.utc),
        )

        self.assertTrue(open_window["open"])
        self.assertEqual(missed["status"], "DECISION_WINDOW_MISSED")
        self.assertFalse(missed["open"])

    def test_stale_spread_quotes_fail_closed(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)
        charts, quotes = _market(pair_count=2)
        for quote in quotes.values():
            quote["timestamp_utc"] = "2026-07-15T23:00:00+00:00"

        shadow = build_forward_shadow(
            candidate,
            charts,
            quotes,
            candidate_sha256=CANDIDATE_SHA,
            observed_at_utc=datetime(2026, 7, 16, 0, 0, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(shadow["status"], "NO_SPREAD_FRESH_ELIGIBLE_PAIRS")
        self.assertEqual(shadow["signals"], [])

    def test_ledger_deduplicates_one_decision(self) -> None:
        candidate = load_forward_candidate(CANDIDATE_PATH)
        charts, quotes = _market(pair_count=2)
        shadow = build_forward_shadow(
            candidate,
            charts,
            quotes,
            candidate_sha256=CANDIDATE_SHA,
            observed_at_utc=datetime(2026, 7, 16, 0, 0, 5, tzinfo=timezone.utc),
        )
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"

            self.assertTrue(append_shadow_once(ledger, shadow))
            self.assertFalse(append_shadow_once(ledger, shadow))
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision_id"], shadow["decision_id"])


def _market(
    *,
    pair_count: int,
    terminal: datetime = datetime(2026, 7, 15, 23, 55, tzinfo=timezone.utc),
):
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
            "generated_at_utc": (terminal + timedelta(minutes=5, seconds=3)).isoformat(),
            "charts": charts,
        },
        quotes,
    )


if __name__ == "__main__":
    unittest.main()
