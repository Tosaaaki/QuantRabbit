"""Regression tests for forward projection direction semantics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.forward_projection import (
    ProjectionSignal,
    aggregate_projection_score,
    detect_forward_projections,
)
from quant_rabbit.strategy.predictive_limit_orders import (
    PredictiveLimitOrder,
    apply_limit_orders,
    generate_limits_from_projections,
)


_QUIET_SESSION_NOW = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)


@dataclass
class _Sig:
    name: str
    direction: str
    rationale: str


def _chart_with_liquidity(*, side: str, price: float) -> dict:
    return {
        "pair": "EUR_USD",
        "views": [
            {
                "granularity": "M5",
                "indicators": {"atr_pips": 10.0},
                "structure": {
                    "liquidity": [
                        {"side": side, "price": price, "indices": [1, 2, 3]},
                    ],
                },
            }
        ],
    }


class LiquiditySweepDirectionTest(unittest.TestCase):
    def test_equal_high_sweep_is_short_entry_bias(self) -> None:
        signals = detect_forward_projections(
            _chart_with_liquidity(side="EQ_HIGH", price=1.1003),
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        sweep = next(s for s in signals if s.name == "liquidity_sweep_high")
        self.assertEqual(sweep.direction, "DOWN")

        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")

        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_equal_low_sweep_is_long_entry_bias(self) -> None:
        signals = detect_forward_projections(
            _chart_with_liquidity(side="EQ_LOW", price=1.0997),
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        sweep = next(s for s in signals if s.name == "liquidity_sweep_low")
        self.assertEqual(sweep.direction, "UP")

        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")

        self.assertGreater(long_score, 0.0, long_reasons)
        self.assertLess(short_score, 0.0, short_reasons)

    def test_sweep_inside_operating_noise_is_not_projection_signal(self) -> None:
        signals = detect_forward_projections(
            _chart_with_liquidity(side="EQ_HIGH", price=1.1001),
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        self.assertFalse(any(s.name == "liquidity_sweep_high" for s in signals), signals)

    def test_sweep_inside_current_spread_is_not_projection_signal(self) -> None:
        chart = _chart_with_liquidity(side="EQ_LOW", price=1.0997)
        chart["views"][0]["indicators"]["atr_pips"] = 0.8
        chart["views"][0]["indicators"]["spread_pips"] = 4.0

        signals = detect_forward_projections(
            chart,
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        self.assertFalse(any(s.name == "liquidity_sweep_low" for s in signals), signals)

    def test_current_cross_asset_schema_emits_dxy_lag_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cross_asset_path = Path(tmp) / "cross_asset_snapshot.json"
            cross_asset_path.write_text(json.dumps({"synthetic_dxy": {"change_pct_24h": 0.6}}))

            signals = detect_forward_projections(
                None,
                pair="EUR_USD",
                current_price=1.1000,
                cross_asset_path=cross_asset_path,
                now=_QUIET_SESSION_NOW,
            )

        signal = next(s for s in signals if s.name == "cross_asset_dxy_lag")
        self.assertEqual(signal.direction, "DOWN")
        self.assertIn("DXY moved +0.60%", signal.rationale)

    def test_projection_score_prefers_direction_specific_calibration(self) -> None:
        signals = [
            ProjectionSignal(
                name="liquidity_sweep_high",
                timeframe="M5",
                direction="UP",
                lead_time_min=10,
                confidence=1.0,
                bonus_magnitude=10.0,
                rationale="EUR_USD sweep-high UP historically fails",
            )
        ]
        hit_rates = {
            "liquidity_sweep_high": {
                "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100},
            },
            "liquidity_sweep_high_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100},
            },
        }

        score, reasons = aggregate_projection_score(
            signals,
            "LONG",
            hit_rates=hit_rates,
            pair="EUR_USD",
            regime="TREND",
        )

        self.assertLess(score, 5.0)
        self.assertTrue(any("[cal×" in reason for reason in reasons), reasons)

    def test_predictive_limit_fades_equal_high_by_signal_name(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_high",
                "DOWN",
                f"M5 equal-highs at {1.1003 + i * 0.00001:.5f} (3.0pip up)",
            )
            for i in range(4)
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "views": [{"granularity": "M15", "indicators": {"atr_pips": 10.0}}],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertTrue(orders)
        self.assertTrue(all(o.side == "SHORT" for o in orders))

    def test_predictive_limit_fades_equal_low_by_signal_name(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                f"M5 equal-lows at {1.0997 - i * 0.00001:.5f} (3.0pip down)",
            )
            for i in range(4)
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "views": [{"granularity": "M15", "indicators": {"atr_pips": 10.0}}],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertTrue(orders)
        self.assertTrue(all(o.side == "LONG" for o in orders))

    def test_predictive_limit_allows_small_grade_b_early_turn_equal_low(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                "M15 equal-lows at 1.09970 (3.0pip down)",
            )
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.05, "price_percentile_7d": 0.02},
                "views": [
                    {
                        "granularity": "M15",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0996,
                            "bb_middle": 1.1005,
                            "close": 1.09975,
                            "rsi_14": 31.0,
                            "williams_r_14": -92.0,
                        },
                        "structure": {"last_event": {"kind": "BOS_DOWN", "close_confirmed": False}},
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "LONG")
        self.assertEqual(orders[0].grade, "B")
        self.assertLess(orders[0].units, 5000)

    def test_predictive_limit_rejects_thin_grade_b_without_extreme_context(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                "M15 equal-lows at 1.09970 (3.0pip down)",
            )
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.45, "price_percentile_7d": 0.50},
                "views": [
                    {
                        "granularity": "M15",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0990,
                            "bb_middle": 1.1005,
                            "close": 1.1002,
                            "rsi_14": 49.0,
                            "williams_r_14": -45.0,
                        },
                        "structure": {"last_event": {"kind": "BOS_DOWN", "close_confirmed": False}},
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(orders, [])

    def test_high_impact_actual_beat_projects_currency_followthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calendar = Path(tmp) / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                                "forecast": "85K",
                                "actual": "172K",
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                now=datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc),
            )

        event_signal = next(s for s in signals if s.name == "event_surprise_followthrough")
        self.assertEqual(event_signal.direction, "DOWN")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_unemployment_miss_uses_lower_is_better_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calendar = Path(tmp) / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Unemployment Rate",
                                "forecast": "4.3%",
                                "actual": "4.6%",
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="USD_JPY",
                current_price=160.0,
                calendar_path=calendar,
                now=datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc),
            )

        event_signal = next(s for s in signals if s.name == "event_surprise_followthrough")
        self.assertEqual(event_signal.direction, "DOWN")

    def test_news_items_feed_directional_followthrough_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = root / "news_digest.md"
            digest.write_text("")
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "lookback_hours": 24,
                        "items": [
                            {
                                "title": "Euro drops as strong US jobs data lifts Greenback",
                                "summary": "NFP steamrolls US Dollar bears and sends the US Dollar higher after payrolls.",
                                "published_at_utc": "2026-06-05T16:50:00+00:00",
                                "currencies": ["USD"],
                                "pairs": ["EUR_USD"],
                                "topics": ["employment"],
                            }
                        ],
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                news_digest_path=digest,
                news_items_path=news_items,
                now=datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc),
            )

        news_signal = next(s for s in signals if s.name == "news_theme_followthrough")
        self.assertEqual(news_signal.direction, "DOWN")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_pre_event_us_employment_nowcast_projects_usd_strength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                                "forecast": "85K",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "title": "ADP private payrolls rise 122K in May, above expectations",
                                "summary": "Hiring was broad-based and labor market momentum stayed resilient before NFP.",
                                "published_at_utc": "2026-06-03T12:45:00+00:00",
                                "currencies": ["USD"],
                                "topics": ["employment"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "us_employment_nowcast")
        self.assertEqual(signal.direction, "DOWN")
        self.assertGreaterEqual(signal.confidence, 0.70)
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_pre_event_us_employment_nowcast_projects_usd_weakness_from_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                                "forecast": "165K",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "title": "Initial jobless claims rise above consensus",
                                "summary": "Weekly claims increase as more workers file for unemployment insurance.",
                                "published_at_utc": "2026-06-04T12:45:00+00:00",
                                "currencies": ["USD"],
                                "topics": ["employment"],
                            },
                            {
                                "title": "Continuing jobless claims rise again",
                                "summary": "Claims move higher for another week.",
                                "published_at_utc": "2026-06-04T12:45:00+00:00",
                                "currencies": ["USD"],
                                "topics": ["employment"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="USD_JPY",
                current_price=160.0,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 4, 13, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "us_employment_nowcast")
        self.assertEqual(signal.direction, "DOWN")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_pre_event_us_employment_nowcast_ignores_future_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                                "forecast": "85K",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            digest = root / "news_digest.md"
            digest.write_text("- ADP private payrolls rise above expectations; hiring broad-based before NFP.\n")
            future_mtime = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc).timestamp()
            os.utime(digest, (future_mtime, future_mtime))

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_digest_path=digest,
                now=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(any(s.name == "us_employment_nowcast" for s in signals))

    def test_pre_event_macro_nowcast_projects_usd_strength_from_cpi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-10T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Core CPI m/m",
                                "forecast": "0.2%",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "source": "MacroWire",
                                "title": "US core CPI seen sticky and above consensus",
                                "summary": "Hot inflation keeps Fed hawkish as yields rise before CPI.",
                                "published_at_utc": "2026-06-09T12:00:00+00:00",
                                "currencies": ["USD"],
                                "topics": ["inflation", "central_bank"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 9, 13, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "macro_event_nowcast_inflation")
        self.assertEqual(signal.direction, "DOWN")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_pre_event_macro_nowcast_projects_eur_strength_from_ecb_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-11T12:15:00+00:00",
                                "currency": "EUR",
                                "impact": "High",
                                "title": "Main Refinancing Rate",
                                "forecast": "3.75%",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            digest = root / "news_digest.md"
            digest.write_text("- EUR / ECB: hawkish rate-hike risk, yields higher before rate decision.\n")
            mtime = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(digest, (mtime, mtime))

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_digest_path=digest,
                now=datetime(2026, 6, 10, 13, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "macro_event_nowcast_central_bank")
        self.assertEqual(signal.direction, "UP")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        self.assertGreater(long_score, 0.0, long_reasons)
        self.assertLess(short_score, 0.0, short_reasons)

    def test_pre_event_macro_nowcast_reads_rate_cut_bets_fade_as_hawkish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-17T18:00:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "FOMC Statement",
                                "forecast": None,
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "source": "MacroWire",
                                "title": "US rate cut bets fade before FOMC",
                                "summary": "Treasury yields are higher as Fed easing expectations recede.",
                                "published_at_utc": "2026-06-16T14:00:00+00:00",
                                "currencies": ["USD"],
                                "topics": ["central_bank", "yields"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 16, 15, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "macro_event_nowcast_central_bank")
        self.assertEqual(signal.direction, "DOWN")

    def test_pre_event_macro_nowcast_projects_gbp_weakness_from_retail_sales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-12T06:00:00+00:00",
                                "currency": "GBP",
                                "impact": "Medium",
                                "title": "Retail Sales m/m",
                                "forecast": "0.3%",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "source": "MacroWire",
                                "title": "Sterling retail sales preview points below consensus",
                                "summary": "UK consumer spending slows and confidence is weaker before retail sales.",
                                "published_at_utc": "2026-06-11T08:00:00+00:00",
                                "currencies": ["GBP"],
                                "topics": ["consumption"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="GBP_USD",
                current_price=1.3400,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "macro_event_nowcast_consumption")
        self.assertEqual(signal.direction, "DOWN")

    def test_pre_event_macro_nowcast_projects_aud_weakness_from_gdp_trade_drag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-03T01:30:00+00:00",
                                "currency": "AUD",
                                "impact": "High",
                                "title": "GDP q/q",
                                "forecast": "0.5%",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "source": "MacroWire",
                                "title": "Australia net trade and flat government spending cloud GDP outlook",
                                "summary": (
                                    "Net trade will subtract from Q1 GDP as imports surged and "
                                    "commodity exports fell, with government spending flat."
                                ),
                                "published_at_utc": "2026-06-02T02:36:49+00:00",
                                "currencies": ["AUD"],
                                "topics": ["growth", "trade"],
                            }
                        ]
                    }
                )
            )

            signals = detect_forward_projections(
                {"views": []},
                pair="AUD_USD",
                current_price=0.7150,
                calendar_path=calendar,
                news_items_path=news_items,
                now=datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc),
            )

        signal = next(s for s in signals if s.name == "macro_event_nowcast_growth")
        self.assertEqual(signal.direction, "DOWN")

    def test_pre_event_macro_nowcast_ignores_ambiguous_digest_without_currency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-10T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "CPI m/m",
                                "forecast": "0.2%",
                                "actual": None,
                            }
                        ]
                    }
                )
            )
            digest = root / "news_digest.md"
            digest.write_text("- CPI could be hot and above consensus before release.\n")
            mtime = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(digest, (mtime, mtime))

            signals = detect_forward_projections(
                {"views": []},
                pair="EUR_USD",
                current_price=1.1600,
                calendar_path=calendar,
                news_digest_path=digest,
                now=datetime(2026, 6, 9, 13, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(any(s.name.startswith("macro_event_nowcast_") for s in signals))

    def test_predictive_limit_dedupes_same_liquidity_level(self) -> None:
        signals = [
            _Sig("liquidity_sweep_low", "UP", "M5 equal-lows at 1.09970 (3.0pip down)"),
            _Sig("liquidity_sweep_low", "UP", "M15 equal-lows at 1.09970 (3.0pip down)"),
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.03},
                "views": [
                    {
                        "granularity": "M5",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0996,
                            "bb_middle": 1.1005,
                            "close": 1.0997,
                            "rsi_14": 35.0,
                        },
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(len(orders), 1)

    def test_predictive_limit_dedupes_nearby_same_trap_prices(self) -> None:
        signals = [
            _Sig("liquidity_sweep_low", "UP", "M1 equal-lows at 1.09970 (3.0pip down)"),
            _Sig("liquidity_sweep_low", "UP", "M5 equal-lows at 1.09973 (2.7pip down)"),
            _Sig("liquidity_sweep_low", "UP", "M15 equal-lows at 1.09980 (2.0pip down)"),
            _Sig("liquidity_sweep_low", "UP", "M30 equal-lows at 1.09970 (3.0pip down)"),
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "views": [{"granularity": "M15", "indicators": {"atr_pips": 10.0}}],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].grade, "A")

    def test_predictive_limit_send_requires_live_confirmation(self) -> None:
        class _Broker:
            def __init__(self) -> None:
                self.requests: list[dict] = []

            def post_order_json(self, payload: dict) -> None:
                self.requests.append(payload)

        old_live_enabled = os.environ.pop("QR_LIVE_ENABLED", None)
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("QR_LIVE_ENABLED", old_live_enabled)
                if old_live_enabled is not None
                else os.environ.pop("QR_LIVE_ENABLED", None)
            )
        )
        broker = _Broker()
        order = PredictiveLimitOrder(
            pair="EUR_USD",
            side="LONG",
            limit_price=1.0997,
            take_profit_price=1.1017,
            units=5000,
            rationale="liquidity sweep fade",
            source="liquidity_sweep_fade",
            grade="A",
            gtd_utc="2026-06-16T06:00:00Z",
        )

        blocked = apply_limit_orders([order], broker, dry_run=False, confirm_live=False)
        self.assertEqual(broker.requests, [])
        self.assertFalse(blocked[0]["sent"])
        self.assertIn("PREDICTIVE_LIMIT_LIVE_GATE_BLOCKED", blocked[0]["error"])

        os.environ["QR_LIVE_ENABLED"] = "1"
        sent = apply_limit_orders([order], broker, dry_run=False, confirm_live=True)
        self.assertTrue(sent[0]["sent"])
        self.assertEqual(len(broker.requests), 1)
