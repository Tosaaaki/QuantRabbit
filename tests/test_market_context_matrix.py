from __future__ import annotations

import unittest

from quant_rabbit.analysis.market_context_matrix import (
    build_market_context_matrix_from_payloads,
    matrix_summary_for_intent,
)


class MarketContextMatrixTest(unittest.TestCase):
    def test_maps_dxy_and_strength_to_eurusd_direction_without_blocking_policy(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts(),
            cross_asset={
                "synthetic_dxy": {"change_pct_24h": 0.4},
                "assets": [],
                "issues": [],
            },
            flow=_flow(),
            currency_strength={
                "scores": [
                    {"currency": "EUR", "rank": 2, "score_pct": 0.1},
                    {"currency": "USD", "rank": 1, "score_pct": 0.5},
                ],
                "issues": [],
            },
            levels=_levels(),
            calendar=_calendar(),
            cot=_cot(),
            option_skew={"readings": [], "issues": ["MISSING_OPTION_SKEW_FEED"]},
        )

        self.assertEqual(payload["trade_count_policy"], "ADVISORY_ONLY_DOES_NOT_BLOCK_OR_DEMOTE_LANES")
        short = payload["pairs"]["EUR_USD"]["SHORT"]
        long = payload["pairs"]["EUR_USD"]["LONG"]
        short_support_codes = {item["code"] for item in short["supports"]}
        long_reject_codes = {item["code"] for item in long["rejects"]}

        self.assertIn("DXY_24H_DIRECTION", short_support_codes)
        self.assertIn("QUOTE_STRENGTH_EXCEEDS_BASE", short_support_codes)
        self.assertIn("DXY_24H_DIRECTION", long_reject_codes)
        self.assertIn("QUOTE_STRENGTH_EXCEEDS_BASE", long_reject_codes)

    def test_missing_option_skew_is_recorded_not_invented(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts(),
            cross_asset={"synthetic_dxy": {"change_pct_24h": 0.0}, "assets": [], "issues": []},
            flow=_flow(),
            currency_strength={"scores": [], "issues": []},
            levels=_levels(),
            calendar=_calendar(),
            cot={"reports": [], "issues": []},
            option_skew={
                "readings": [
                    {
                        "pair": "EUR_USD",
                        "tenor": "1W",
                        "rr_25d": None,
                        "issue": "MISSING_OPTION_SKEW_FEED",
                    }
                ],
                "issues": ["MISSING_OPTION_SKEW_FEED"],
            },
        )

        long = payload["pairs"]["EUR_USD"]["LONG"]
        missing_codes = {item["code"] for item in long["missing"]}
        option_supports = [item for item in long["supports"] if item["layer"] == "option_skew"]

        self.assertIn("MISSING_OPTION_SKEW_FEED", missing_codes)
        self.assertEqual(option_supports, [])

    def test_disabled_option_skew_is_ignored_not_counted_missing(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts(),
            cross_asset={"synthetic_dxy": {"change_pct_24h": 0.0}, "assets": [], "issues": []},
            flow=_flow(),
            currency_strength={"scores": [], "issues": []},
            levels=_levels(),
            calendar=_calendar(),
            cot={"reports": [], "issues": []},
            option_skew={
                "enabled": False,
                "disabled_reason": "NO_OPTION_SKEW_PROVIDER",
                "readings": [],
                "issues": [],
            },
        )

        long = payload["pairs"]["EUR_USD"]["LONG"]
        missing_codes = {item["code"] for item in long["missing"]}

        self.assertNotIn("MISSING_OPTION_SKEW_PAIR", missing_codes)
        self.assertNotIn("MISSING_OPTION_SKEW_FEED", missing_codes)

    def test_cot_is_longer_term_warning_not_short_term_reject(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts(),
            cross_asset={"synthetic_dxy": {"change_pct_24h": 0.0}, "assets": [], "issues": []},
            flow=_flow(),
            currency_strength={
                "scores": [
                    {"currency": "EUR", "rank": 1, "score_pct": 0.4},
                    {"currency": "USD", "rank": 2, "score_pct": 0.2},
                ],
                "issues": [],
            },
            levels=_levels(),
            calendar=_calendar(),
            cot=_cot(),
            option_skew={"readings": [], "issues": ["MISSING_OPTION_SKEW_FEED"]},
        )

        short = payload["pairs"]["EUR_USD"]["SHORT"]
        cot_warnings = [item for item in short["warnings"] if item["layer"] == "cot"]
        cot_rejects = [item for item in short["rejects"] if item["layer"] == "cot"]

        self.assertTrue(cot_warnings)
        self.assertEqual(cot_rejects, [])

    def test_gold_and_oil_are_cross_asset_context_not_trade_blocks(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts("EUR_USD", "USD_CAD"),
            context_asset_charts=_context_asset_charts("XAU_USD", "WTICO_USD"),
            cross_asset={
                "synthetic_dxy": {"change_pct_24h": 0.0},
                "assets": [
                    {"instrument": "XAU_USD", "change_pct_24h": 1.2},
                    {"instrument": "WTICO_USD", "change_pct_24h": 2.5},
                ],
                "issues": [],
            },
            flow=_flow(),
            currency_strength={"scores": [], "issues": []},
            levels=_levels(),
            calendar=_calendar(),
            cot={"reports": [], "issues": []},
            option_skew={"readings": [], "issues": ["MISSING_OPTION_SKEW_FEED"]},
        )

        eurusd_long_codes = {item["code"] for item in payload["pairs"]["EUR_USD"]["LONG"]["supports"]}
        usdcad_short_codes = {item["code"] for item in payload["pairs"]["USD_CAD"]["SHORT"]["supports"]}
        usdcad_long_warning_codes = {item["code"] for item in payload["pairs"]["USD_CAD"]["LONG"]["warnings"]}
        eurusd_gold_context = [
            item
            for item in payload["pairs"]["EUR_USD"]["LONG"]["supports"]
            if item["code"] == "GOLD_CONTEXT_TECHNICAL_DIRECTION"
        ]

        self.assertIn("GOLD_USD_PRESSURE_DIRECTION", eurusd_long_codes)
        self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", eurusd_long_codes)
        self.assertIn("GOLD_USD_PRESSURE_DIRECTION", usdcad_short_codes)
        self.assertIn("OIL_CAD_DIRECTION", usdcad_short_codes)
        self.assertIn("OIL_CONTEXT_TECHNICAL_DIRECTION", usdcad_short_codes)
        self.assertIn("OIL_CAD_DIRECTION", usdcad_long_warning_codes)
        self.assertEqual(eurusd_gold_context[0]["evidence_refs"], ["context_asset:XAU_USD"])
        self.assertEqual(payload["trade_count_policy"], "ADVISORY_ONLY_DOES_NOT_BLOCK_OR_DEMOTE_LANES")
        summary = matrix_summary_for_intent(payload, "EUR_USD", "LONG")
        self.assertIn("context_asset_chart", summary["matrix_support_layers"])
        self.assertTrue(
            any("GOLD_CONTEXT_TECHNICAL_DIRECTION" in item for item in summary["matrix_support_context"])
        )
        self.assertNotIn("supports", summary)

    def test_intent_summary_keeps_matrix_compact(self) -> None:
        payload = build_market_context_matrix_from_payloads(
            pair_charts=_pair_charts(),
            cross_asset={"synthetic_dxy": {"change_pct_24h": -0.2}, "assets": [], "issues": []},
            flow=_flow(),
            currency_strength={
                "scores": [
                    {"currency": "EUR", "rank": 1, "score_pct": 0.4},
                    {"currency": "USD", "rank": 2, "score_pct": 0.2},
                ],
                "issues": [],
            },
            levels=_levels(),
            calendar=_calendar(),
            cot=_cot(),
            option_skew={"readings": [], "issues": ["MISSING_OPTION_SKEW_FEED"]},
        )

        summary = matrix_summary_for_intent(payload, "EUR_USD", "LONG")

        self.assertEqual(summary["market_context_matrix_ref"], "matrix:EUR_USD:LONG")
        self.assertIn("matrix_support_count", summary)
        self.assertIn("strongest_matrix_reject", summary)
        self.assertNotIn("supports", summary)


def _pair_charts(*pairs: str) -> dict:
    pairs = pairs or ("EUR_USD",)
    return {
        "charts": [
            {
                "pair": pair,
                "dominant_regime": "TREND_UP",
                "long_score": 0.7,
                "short_score": 0.3,
                "confluence": {
                    "score_balance": "LONG_LEAN",
                    "dominant_regime": "TREND_UP",
                    "higher_tf_alignment": "ALIGNED",
                },
                "views": [],
            }
            for pair in pairs
        ]
    }


def _context_asset_charts(*assets: str) -> dict:
    return {
        "role": "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION",
        "charts": [
            {
                "pair": asset,
                "dominant_regime": "TREND_UP",
                "long_score": 0.8,
                "short_score": 0.2,
                "confluence": {
                    "score_balance": "LONG_LEAN",
                    "dominant_regime": "TREND_UP",
                },
                "views": [],
            }
            for asset in assets
        ],
        "issues": [],
    }


def _flow() -> dict:
    return {
        "spreads": [
            {
                "instrument": "EUR_USD",
                "current_pips": 0.8,
                "median_pips": 1.2,
                "stress_flag": "NORMAL",
            }
        ],
        "order_books": [{"instrument": "EUR_USD", "issue": "ORDERBOOK_FEED_UNAUTHORIZED"}],
        "position_books": [{"instrument": "EUR_USD", "issue": "POSITIONBOOK_FEED_UNAUTHORIZED"}],
        "issues": [],
    }


def _levels() -> dict:
    return {
        "pairs": [
            {
                "pair": "EUR_USD",
                "last_close": 1.101,
                "daily_open": 1.1,
                "weekly_open": 1.099,
                "pdc": 1.1005,
                "round_numbers": [{"price": 1.1, "distance_pips": -10.0}],
            }
        ],
        "issues": [],
    }


def _calendar() -> dict:
    return {
        "pair_windows": [{"pair": "EUR_USD", "in_window": False, "reason": "next event outside window"}],
        "issues": [],
    }


def _cot() -> dict:
    return {
        "reports": [
            {"currency": "EUR", "leveraged_net": -500.0},
            {"currency": "USD", "leveraged_net": 1000.0},
        ],
        "issues": [],
    }


if __name__ == "__main__":
    unittest.main()
