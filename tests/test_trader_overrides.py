"""Unit tests for strategy/trader_overrides.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.strategy.trader_overrides import (
    TraderOverrides,
    load_trader_overrides,
    overrides_block_check,
    overrides_score_delta,
)


def _write_overrides(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "trader_overrides.json"
    path.write_text(json.dumps(payload))
    return path


class TraderOverridesLoadTest(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ov = load_trader_overrides(Path(tmp))
            self.assertEqual(ov.bias_modifiers, {})
            self.assertEqual(ov.blocked_lane_ids, frozenset())
            self.assertEqual(ov.narrative_summary, "")

    def test_malformed_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trader_overrides.json").write_text("{not json}")
            ov = load_trader_overrides(root)
            self.assertEqual(ov.bias_modifiers, {})

    def test_loads_bias_overrides_per_pair_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
            _write_overrides(root, {
                "expires_at_utc": future,
                "bias_overrides": {
                    "GBP_USD": {"LONG": -25.0, "SHORT": 0.0},
                    "EUR_USD": {"LONG": -15.0},
                },
            })
            ov = load_trader_overrides(root)
            self.assertAlmostEqual(ov.bias_modifiers[("GBP_USD", "LONG")], -25.0)
            self.assertAlmostEqual(ov.bias_modifiers[("GBP_USD", "SHORT")], 0.0)
            self.assertAlmostEqual(ov.bias_modifiers[("EUR_USD", "LONG")], -15.0)

    def test_loads_blocked_lane_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
            _write_overrides(root, {
                "expires_at_utc": future,
                "blocked_lanes": ["failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE"],
            })
            ov = load_trader_overrides(root)
            self.assertIn("failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE", ov.blocked_lane_ids)

    def test_expired_overrides_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            _write_overrides(root, {
                "expires_at_utc": past,
                "bias_overrides": {"GBP_USD": {"LONG": -25.0}},
                "blocked_lanes": ["xxx"],
            })
            ov = load_trader_overrides(root)
            self.assertEqual(ov.bias_modifiers, {})
            self.assertEqual(ov.blocked_lane_ids, frozenset())

    def test_missing_expiry_treated_as_non_expiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_overrides(root, {
                "bias_overrides": {"USD_JPY": {"LONG": +10.0}},
            })
            ov = load_trader_overrides(root)
            self.assertAlmostEqual(ov.bias_modifiers[("USD_JPY", "LONG")], +10.0)


class TraderOverridesLookupTest(unittest.TestCase):
    def test_score_delta_for_known_pair_direction(self) -> None:
        ov = TraderOverrides(
            bias_modifiers={("GBP_USD", "LONG"): -25.0},
            blocked_lane_ids=frozenset(),
            narrative_summary="",
            expires_at_utc=None,
            source_path=None,
        )
        delta, rationale = overrides_score_delta(ov, "GBP_USD", "LONG")
        self.assertAlmostEqual(delta, -25.0)
        self.assertIsNotNone(rationale)
        self.assertIn("GBP_USD", rationale)

    def test_score_delta_unknown_pair_returns_zero(self) -> None:
        ov = TraderOverrides(
            bias_modifiers={("GBP_USD", "LONG"): -25.0},
            blocked_lane_ids=frozenset(),
            narrative_summary="",
            expires_at_utc=None,
            source_path=None,
        )
        delta, rationale = overrides_score_delta(ov, "EUR_USD", "SHORT")
        self.assertEqual(delta, 0.0)
        self.assertIsNone(rationale)

    def test_block_check_for_listed_lane(self) -> None:
        ov = TraderOverrides(
            bias_modifiers={},
            blocked_lane_ids=frozenset(["lane-x"]),
            narrative_summary="",
            expires_at_utc=None,
            source_path=None,
        )
        blocked, msg = overrides_block_check(ov, "lane-x")
        self.assertTrue(blocked)
        self.assertIn("lane-x", msg or "")

    def test_block_check_for_unlisted_lane(self) -> None:
        ov = TraderOverrides(
            bias_modifiers={},
            blocked_lane_ids=frozenset(["lane-x"]),
            narrative_summary="",
            expires_at_utc=None,
            source_path=None,
        )
        blocked, msg = overrides_block_check(ov, "lane-y")
        self.assertFalse(blocked)
        self.assertIsNone(msg)


if __name__ == "__main__":
    unittest.main()
