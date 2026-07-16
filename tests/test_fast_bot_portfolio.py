from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.fast_bot_portfolio import build_fast_bot_portfolio


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _signal(
    signal_id: str,
    *,
    pair: str,
    side: str,
    method: str,
    horizon: str,
    start_minutes: int = 0,
    hold_minutes: int = 15,
    priority: float = 1.0,
    units: float = 100.0,
    margin: float = 10.0,
    entry: float = 1.1,
) -> dict:
    start = NOW + timedelta(minutes=start_minutes)
    body = {
        "signal_id": signal_id,
        "pair": pair,
        "side": side,
        "method": method,
        "horizon_lane": horizon,
        "generated_at_utc": NOW.isoformat(),
        "holding_window": {
            "start_utc": start.isoformat(),
            "end_utc": (start + timedelta(minutes=hold_minutes)).isoformat(),
        },
        "estimated_margin_jpy": margin,
        "notional_units": units,
        "entry": entry,
        "portfolio_priority": priority,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**body, "signal_sha256": hashlib.sha256(raw).hexdigest()}


def _constraints(
    *,
    margin: float = 1_000.0,
    gross: float = 1_000.0,
    concurrent: int = 10,
) -> dict:
    return {
        "margin_budget_jpy": margin,
        "max_currency_gross": gross,
        "max_concurrent_signals_per_horizon": concurrent,
    }


class FastBotPortfolioTest(unittest.TestCase):
    def test_multiple_pairs_and_methods_are_selected_simultaneously(self) -> None:
        signals = [
            _signal("eur-trend", pair="EUR_USD", side="LONG", method="TREND", horizon="SHORT"),
            _signal("gbp-range", pair="GBP_USD", side="SHORT", method="RANGE", horizon="SHORT"),
            _signal("eur-range", pair="EUR_USD", side="LONG", method="RANGE", horizon="SHORT"),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 3)
        self.assertFalse(result["top_one_selection"])
        self.assertEqual({row["signal_id"] for row in result["selected"]}, {"eur-trend", "gbp-range", "eur-range"})
        self.assertFalse(result["live_permission"])
        self.assertFalse(result["broker_mutation"])

    def test_same_pair_short_long_and_medium_short_coexist_when_hedging(self) -> None:
        signals = [
            _signal("short-long", pair="EUR_USD", side="LONG", method="TREND", horizon="SHORT"),
            _signal("medium-short", pair="EUR_USD", side="SHORT", method="RANGE", horizon="MEDIUM", hold_minutes=60),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 2)
        coexistence = result["opposite_side_coexistence"][0]
        self.assertFalse(coexistence["same_horizon"])
        self.assertTrue(coexistence["holding_windows_overlap"])
        self.assertEqual(coexistence["broker_projection"], "BROKER_HEDGE_CAPABLE_IF_SEPARATELY_PROMOTED")

    def test_non_overlapping_shared_currency_is_not_correlation_blocked(self) -> None:
        signals = [
            _signal(
                "first",
                pair="EUR_USD",
                side="LONG",
                method="TREND",
                horizon="SHORT",
                units=100,
            ),
            _signal(
                "later",
                pair="GBP_USD",
                side="LONG",
                method="TREND",
                horizon="SHORT",
                start_minutes=15,
                units=100,
            ),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(gross=110),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 2)
        self.assertEqual(result["shadow_rejected_count"], 0)

    def test_overlapping_cross_horizon_paths_share_margin_and_currency_gross(self) -> None:
        signals = [
            _signal(
                "short-high",
                pair="EUR_USD",
                side="LONG",
                method="TREND",
                horizon="SHORT",
                priority=2,
                units=100,
                margin=60,
            ),
            _signal(
                "medium-low",
                pair="GBP_USD",
                side="LONG",
                method="RANGE",
                horizon="MEDIUM",
                priority=1,
                units=100,
                margin=60,
            ),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(margin=100, gross=110),
            now_utc=NOW,
        )

        self.assertEqual({row["signal_id"] for row in result["selected"]}, {"short-high"})
        rejected = next(row for row in result["shadow_rejected"] if row["signal_id"] == "medium-low")
        self.assertIn("MARGIN_BUDGET_JPY", rejected["reasons"])
        self.assertIn("MAX_CURRENCY_GROSS:USD", rejected["reasons"])

    def test_non_overlapping_cross_horizon_paths_do_not_share_account_risk(self) -> None:
        signals = [
            _signal(
                "short",
                pair="EUR_USD",
                side="LONG",
                method="TREND",
                horizon="SHORT",
                units=100,
                margin=60,
            ),
            _signal(
                "medium-later",
                pair="GBP_USD",
                side="LONG",
                method="RANGE",
                horizon="MEDIUM",
                start_minutes=15,
                units=100,
                margin=60,
            ),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(margin=60, gross=110),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 2)
        self.assertEqual(result["shadow_rejected_count"], 0)

    def test_quote_currency_leg_uses_reference_entry_price(self) -> None:
        result = build_fast_bot_portfolio(
            [
                _signal(
                    "eur-long",
                    pair="EUR_USD",
                    side="LONG",
                    method="TREND",
                    horizon="SHORT",
                    units=100,
                    entry=1.1,
                )
            ],
            hedging_enabled=True,
            constraints=_constraints(),
            now_utc=NOW,
        )

        exposure = result["currency_legs_and_exposure"]["by_currency"]
        self.assertEqual(exposure["EUR"]["net_units"], 100.0)
        self.assertAlmostEqual(exposure["USD"]["net_units"], -110.0)
        self.assertAlmostEqual(exposure["USD"]["gross_units"], 110.0)
        self.assertEqual(
            result["currency_legs_and_exposure"]["unit_basis"],
            "BASE_UNITS_AND_REFERENCE_ENTRY_QUOTE_UNITS",
        )

    def test_missing_shadow_size_and_margin_are_rejected_not_fabricated(self) -> None:
        signal = _signal(
            "unsized",
            pair="EUR_USD",
            side="LONG",
            method="TREND",
            horizon="SHORT",
        )
        body = {key: value for key, value in signal.items() if key != "signal_sha256"}
        body.pop("notional_units")
        body.pop("estimated_margin_jpy")
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        signal = {**body, "signal_sha256": hashlib.sha256(raw).hexdigest()}

        result = build_fast_bot_portfolio(
            [signal],
            hedging_enabled=True,
            constraints=_constraints(),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 0)
        self.assertIn("NOTIONAL_UNITS_INVALID", result["shadow_rejected"][0]["reasons"])
        self.assertIn("ESTIMATED_MARGIN_JPY_INVALID", result["shadow_rejected"][0]["reasons"])
        self.assertEqual(
            result["source_schema_compatibility"]["missing_sizing_policy"],
            "SHADOW_REJECT_WITHOUT_FABRICATION",
        )

    def test_binding_constraint_rejects_lowest_priority_but_retains_shadow_record(self) -> None:
        signals = [
            _signal("high", pair="EUR_USD", side="LONG", method="TREND", horizon="SHORT", priority=3),
            _signal("middle", pair="GBP_JPY", side="LONG", method="RANGE", horizon="SHORT", priority=2),
            _signal("low", pair="AUD_CAD", side="SHORT", method="FAILURE", horizon="SHORT", priority=1),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=True,
            constraints=_constraints(concurrent=2),
            now_utc=NOW,
        )

        self.assertEqual({row["signal_id"] for row in result["selected"]}, {"high", "middle"})
        rejected = next(row for row in result["shadow_rejected"] if row["signal_id"] == "low")
        self.assertIn("MAX_CONCURRENT_SIGNALS_PER_HORIZON", rejected["reasons"])
        self.assertTrue(rejected["shadow_record_retained"])
        self.assertIn("AUD_CAD:SHORT:FAILURE:SHORT", result["eligible_signals_by_key"])
        counterfactual = next(row for row in result["counterfactual_path_audit"] if row["signal_id"] == "low")
        self.assertTrue(counterfactual["would_select_if_constraints_unbound"])

    def test_netting_account_keeps_opposing_paths_but_marks_virtual_only(self) -> None:
        signals = [
            _signal("long", pair="EUR_USD", side="LONG", method="TREND", horizon="SHORT"),
            _signal("short", pair="EUR_USD", side="SHORT", method="RANGE", horizon="MEDIUM"),
        ]

        result = build_fast_bot_portfolio(
            signals,
            hedging_enabled=False,
            constraints=_constraints(),
            now_utc=NOW,
        )

        self.assertEqual(result["selected_signal_count"], 2)
        self.assertEqual(result["opposite_side_coexistence"][0]["broker_projection"], "VIRTUAL_ONLY_NETTING_ACCOUNT")
        self.assertTrue(all(row["broker_projection"] == "VIRTUAL_ONLY_NETTING_ACCOUNT" for row in result["selected"]))


if __name__ == "__main__":
    unittest.main()
