from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.intent_generator import IntentGenerator


class IntentGeneratorTest(unittest.TestCase):
    def test_requires_snapshot_before_pricing_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            output = root / "intents.json"
            report = root / "intents.md"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.generated, 0)
            self.assertEqual(summary.needs_snapshot, 1)
            self.assertIn("NEEDS_BROKER_SNAPSHOT", report.read_text())

    def test_generates_and_risk_checks_priced_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"
            report = root / "intents.md"

            cap_jpy = 500.0
            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=cap_jpy,
            ).run(snapshot_path=snapshot)

            self.assertEqual(summary.generated, 2)
            self.assertEqual(summary.dry_run_passed, 2)
            self.assertEqual(summary.live_ready, 0)
            payload = json.loads(output.read_text())
            order_types = {item["intent"]["order_type"] for item in payload["results"]}
            self.assertEqual(order_types, {"STOP-ENTRY", "MARKET"})
            result = next(item for item in payload["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            self.assertEqual(result["intent"]["pair"], "EUR_USD")
            self.assertEqual(result["intent"]["market_context"]["method"], "TREND_CONTINUATION")
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], cap_jpy)
            self.assertGreater(result["risk_metrics"]["spread_pips"], 0.0)
            self.assertTrue(result["live_blockers"])

            market = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            self.assertEqual(market["lane_id"], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertEqual(market["intent"]["metadata"]["parent_lane_id"], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(market["intent"]["metadata"]["order_timing"], "NOW_MARKET")

    def test_carries_current_regime_and_session_bucket_from_pair_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                pair_charts_path=_pair_charts_with_context(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            intent = payload["results"][0]["intent"]
            self.assertEqual(intent["metadata"]["regime_state"], "TREND_DOWN")
            self.assertEqual(intent["metadata"]["session_bucket"], "NY")
            self.assertEqual(intent["market_context"]["session"], "NY")
            self.assertIn("TREND_DOWN current", intent["market_context"]["regime"])

    def test_blocks_chart_direction_conflict_before_live_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root, direction="SHORT"),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="TREND_UP",
                    m5_regime="TREND_UP",
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))

    def test_blocks_trend_market_chase_when_m5_is_not_trending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="TREND_UP",
                    m5_regime="RANGE",
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            stop_entry = next(item for item in payload["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            market = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            market_issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(stop_entry["status"], "LIVE_READY")
            self.assertEqual(market["status"], "DRY_RUN_BLOCKED")
            self.assertIn("TREND_MARKET_NOT_OPERATING_TREND", market_issue_codes)

    def test_trigger_receipt_required_does_not_create_market_chase_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_trigger_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}
            order_types = {item["intent"]["order_type"] for item in payload["results"]}

            self.assertEqual(summary.generated, 1)
            self.assertEqual(order_types, {"STOP-ENTRY"})
            self.assertEqual(lane_ids, {"failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"})
            self.assertFalse(any(lane_id.endswith(":MARKET") for lane_id in lane_ids))

    def test_sizes_repair_receipt_to_use_loss_budget_without_breaking_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=snapshot)

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            intent = result["intent"]
            # With ATR-derived geometry the SL distance is the larger of
            # 1*ATR(M5) and 6*spread, so unit count is bounded by the new
            # geometry; assert risk fits the cap rather than a fixed unit.
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)
            self.assertGreater(intent["units"], 0)

    def test_generic_stop_sits_beyond_current_adverse_wick_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = next(item for item in json.loads(output.read_text())["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            intent = result["intent"]
            metadata = intent["metadata"]

            self.assertEqual(metadata["geometry_model"], "ATR_SPREAD_STRUCTURE")
            self.assertTrue(metadata["structural_stop_outside_level"])
            self.assertLess(intent["sl"], metadata["structural_stop_level"])
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)

    def test_sizes_usd_quote_pair_from_snapshot_conversion_not_static_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root, usd_jpy=300.0))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # Unit count depends on conversion rate AND the (now ATR-derived)
            # SL distance — assert risk fits cap rather than fixed unit count.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)

    def test_uses_campaign_runner_reward_risk_for_tp_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root, target_reward_risk=4.0),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["intent"]["metadata"]["target_reward_risk"], 4.0)
            self.assertAlmostEqual(result["risk_metrics"]["reward_risk"], 4.0)

    def test_range_rotation_uses_rail_limit_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            intent = result["intent"]
            metadata = intent["metadata"]
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(intent["order_type"], "LIMIT")
            self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
            self.assertAlmostEqual(metadata["range_support"], 1.17100)
            self.assertAlmostEqual(metadata["range_resistance"], 1.17600)
            self.assertLess(abs(intent["entry"] - metadata["range_support"]), 0.0002)
            self.assertLess(intent["sl"], metadata["range_support"])
            self.assertTrue(metadata["range_tp_is_inside_box"])

    def test_range_rotation_adds_market_reclaim_when_quote_is_at_rail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root, eur_bid=1.17110, eur_ask=1.17118))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))

            self.assertEqual(market["status"], "LIVE_READY")
            self.assertEqual(market["intent"]["order_type"], "MARKET")
            self.assertEqual(market["intent"]["metadata"]["geometry_model"], "RANGE_RAIL_MARKET")
            self.assertTrue(market["intent"]["metadata"]["range_tp_is_inside_box"])

    def test_range_rotation_market_variant_blocks_when_quote_is_mid_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))
            issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(market["status"], "DRY_RUN_BLOCKED")
            self.assertIn("RANGE_MARKET_NOT_AT_RAIL", issue_codes)

    def test_low_vol_directional_range_market_uses_tight_risk_budgeted_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="UNCLEAR",
                    m5_regime="RANGE",
                    m5_long_bias=0.72,
                    m5_short_bias=0.18,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                ),
                max_loss_jpy=140.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))
            issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(market["status"], "LIVE_READY")
            self.assertEqual(market["intent"]["metadata"]["geometry_model"], "RANGE_DIRECTIONAL_MARKET")
            self.assertEqual(market["intent"]["metadata"]["regime_state"], "RANGE")
            self.assertEqual(market["intent"]["metadata"]["regime_stop_widen_mult"], 1.0)
            self.assertGreaterEqual(market["intent"]["units"], 2000)
            self.assertLessEqual(market["risk_metrics"]["risk_jpy"], 140.0)
            self.assertNotIn("RANGE_MARKET_NOT_AT_RAIL", issue_codes)

    def test_range_direction_conflict_uses_m5_bias_not_aggregate_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root, direction="SHORT"),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.82,
                    short_score=0.11,
                    dominant_regime="UNCLEAR",
                    m5_regime="RANGE",
                    m5_long_bias=0.12,
                    m5_short_bias=0.76,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                ),
                max_loss_jpy=140.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))

            self.assertEqual(market["status"], "LIVE_READY")
            self.assertNotIn("CHART_DIRECTION_CONFLICT", issue_codes)

    def test_open_position_with_per_trade_sized_risk_does_not_block_new_entry(self) -> None:
        # AGENT_CONTRACT §3.5 regression: portfolio cap is the WHOLE-DAY risk
        # budget, not the per-trade slice. A previous bug fed `max_loss_jpy`
        # (per-trade cap, e.g. 1050 JPY) into `max_portfolio_loss_jpy` so the
        # second any position opened, every fresh-entry candidate failed
        # `open_risk + candidate_risk > 1051` and the trader fell back to WAIT
        # for the rest of the day. With the fix, when no whole-day cap is
        # available (no daily_target_state.json on disk) the portfolio gate is
        # a no-op — it does NOT silently inherit the per-trade cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot_with_position(root)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=1050.0,
            ).run(snapshot_path=snapshot)

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            issue_codes = {i["code"] for i in result["risk_issues"]}
            # Per-trade cap (1050) is still enforced on the new candidate via
            # `loss_cap`; portfolio cap (whole day) is absent in this test
            # because no ledger exists, so the portfolio gate is a no-op.
            self.assertNotIn("PORTFOLIO_LOSS_CAP_EXCEEDED", issue_codes)

    def test_sizes_units_with_percentage_risk_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_pct=1.0,
                risk_equity_jpy=100_000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # 1% of 100,000 = 1000 JPY cap. Unit count is derived from cap and
            # the (now ATR-aware) SL distance; assert risk respects the cap.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1000.0)

    def test_sizes_units_to_margin_budget_before_live_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=5_000.0,
            ).run(
                snapshot_path=_snapshot(
                    root,
                    nav_jpy=220_145.7765,
                    balance_jpy=208_945.7765,
                    margin_used_jpy=156_414.0,
                    margin_available_jpy=63_831.7765,
                )
            )

            payload = json.loads(output.read_text())
            for result in payload["results"]:
                self.assertLessEqual(result["intent"]["units"], 6000)
                self.assertLessEqual(result["risk_metrics"]["margin_utilization_after_pct"], 92.0)
                issue_codes = {issue["code"] for issue in result["risk_issues"]}
                self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", issue_codes)
                self.assertNotIn("MARGIN_AVAILABLE_EXCEEDED", issue_codes)


def _campaign(root: Path, *, target_reward_risk: float | None = None, direction: str = "LONG") -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": direction,
                        "method": "TREND_CONTINUATION",
                        "adoption": "RISK_REPAIR_DRY_RUN",
                        "campaign_role": "NOW_IF_REPAIRED",
                        "reason": "trend continuation pressure",
                        "required_receipt": "dry-run under loss cap",
                        **({"target_reward_risk": target_reward_risk} if target_reward_risk is not None else {}),
                        "blockers": ["old sizing broke the loss cap"],
                        "story_examples": ["quality_audit: green staircase into upper band"],
                    },
                    {
                        "desk": "event_risk_trader",
                        "pair": "EUR_USD",
                        "direction": "BOTH",
                        "method": "EVENT_RISK",
                        "adoption": "RISK_OVERLAY",
                    },
                ]
            }
        )
    )
    return path


def _range_campaign(root: Path, *, direction: str = "LONG") -> Path:
    path = root / "range_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "range_trader",
                        "pair": "EUR_USD",
                        "direction": direction,
                        "method": "RANGE_ROTATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW",
                        "reason": "range rail rotation pressure",
                        "required_receipt": "enter only at lower rail and rotate toward box interior",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["quality_audit: lower rail box reclaim into midpoint"],
                    }
                ]
            }
        )
    )
    return path


def _trigger_campaign(root: Path) -> Path:
    path = root / "trigger_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "failure_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "adoption": "TRIGGER_RECEIPT_REQUIRED",
                        "campaign_role": "BACKUP_OR_RELOAD",
                        "reason": "missed-edge trigger pressure",
                        "required_receipt": "Arm only a trigger/pending-entry receipt; no market chase.",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["quality_audit: failed downside break reclaimed the box"],
                    }
                ]
            }
        )
    )
    return path


def _strategy(root: Path, *, status: str = "RISK_REPAIR_CANDIDATE", direction: str = "LONG") -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": direction,
                        "status": status,
                        "required_fix": "edge exists but old sizing broke the loss cap",
                    }
                ]
            }
        )
    )
    return path


def _pair_charts(root: Path) -> Path:
    path = root / "pair_charts.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "views": [
                            {
                                "granularity": "M5",
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _pair_charts_with_context(root: Path) -> Path:
    path = root / "pair_charts_context.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_DOWN",
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime_reading": {"state": "TREND_WEAK", "confidence": 0.5, "atr_percentile": 80.0},
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _pair_charts_with_direction(
    root: Path,
    *,
    long_score: float,
    short_score: float,
    dominant_regime: str,
    m5_regime: str,
    m5_long_bias: float | None = None,
    m5_short_bias: float | None = None,
    regime_quantile: str = "NORMAL",
    atr_pips: float = 8.0,
) -> Path:
    path = root / "pair_charts_direction.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": dominant_regime,
                        "long_score": long_score,
                        "short_score": short_score,
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": m5_regime,
                                "long_bias": long_score if m5_long_bias is None else m5_long_bias,
                                "short_bias": short_score if m5_short_bias is None else m5_short_bias,
                                "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6, "atr_percentile": 50.0},
                                "family_scores": {
                                    "mean_rev_score": 1.1,
                                    "trend_score": 0.2,
                                    "breakout_score": 0.1,
                                    "disagreement": 0.2,
                                },
                                "indicators": {
                                    "atr_pips": atr_pips,
                                    "regime_quantile": regime_quantile,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _snapshot(
    root: Path,
    *,
    usd_jpy: float = 156.64,
    eur_bid: float = 1.17322,
    eur_ask: float = 1.17330,
    nav_jpy: float = 200_000.0,
    balance_jpy: float = 200_000.0,
    margin_used_jpy: float = 0.0,
    margin_available_jpy: float = 200_000.0,
) -> Path:
    path = root / "snapshot.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": eur_bid, "ask": eur_ask, "timestamp_utc": now},
                    "USD_JPY": {"bid": usd_jpy, "ask": usd_jpy + 0.008, "timestamp_utc": now},
                },
                "account": {
                    "nav_jpy": nav_jpy,
                    "balance_jpy": balance_jpy,
                    "margin_used_jpy": margin_used_jpy,
                    "margin_available_jpy": margin_available_jpy,
                    "fetched_at_utc": now,
                },
            }
        )
    )
    return path


def _snapshot_with_position(root: Path) -> Path:
    """Snapshot carrying an open trader position whose worst-case loss is
    near the per-trade cap (≈1000 JPY @ 5000u EUR_USD with a 20-pip stop)."""
    path = root / "snapshot.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 5000,
                        "entry_price": 1.1710,
                        "take_profit": 1.1750,
                        "stop_loss": 1.1690,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": 1.17322, "ask": 1.17330, "timestamp_utc": now},
                    "USD_JPY": {"bid": 156.64, "ask": 156.648, "timestamp_utc": now},
                },
                "account": {
                    "nav_jpy": 200_000.0,
                    "balance_jpy": 200_000.0,
                    "margin_used_jpy": 0.0,
                    "margin_available_jpy": 200_000.0,
                    "fetched_at_utc": now,
                },
            }
        )
    )
    return path


class RegimeAwareGeometryHelpersTest(unittest.TestCase):
    """Unit tests for regime-derived reward_risk and SL widening helpers.

    Per AGENT_CONTRACT §3.5: TP/SL must be regime-derived. These tests pin the
    multiplier mapping so a future refactor cannot silently revert to a single
    fixed reward_risk floor.
    """

    def test_range_regime_shortens_target(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_RANGE_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("RANGE"), REGIME_REWARD_RISK_RANGE_MULT)
        self.assertLess(REGIME_REWARD_RISK_RANGE_MULT, 1.0)

    def test_trend_regime_widens_target(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_TREND_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("TREND_UP"), REGIME_REWARD_RISK_TREND_MULT)
        self.assertEqual(_regime_reward_risk_multiplier("TREND_DOWN"), REGIME_REWARD_RISK_TREND_MULT)
        self.assertGreater(REGIME_REWARD_RISK_TREND_MULT, 1.0)

    def test_impulse_regime_extends_target_furthest(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_IMPULSE_MULT,
            REGIME_REWARD_RISK_TREND_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("IMPULSE_UP"), REGIME_REWARD_RISK_IMPULSE_MULT)
        self.assertGreaterEqual(REGIME_REWARD_RISK_IMPULSE_MULT, REGIME_REWARD_RISK_TREND_MULT)

    def test_unknown_or_unclear_regime_returns_unchanged(self) -> None:
        from quant_rabbit.strategy.intent_generator import _regime_reward_risk_multiplier

        self.assertEqual(_regime_reward_risk_multiplier(None), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier(""), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier("UNCLEAR"), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier("FAILURE_RISK"), 1.0)

    def test_low_confidence_widens_stop(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_LOW_CONFIDENCE_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        reading = {"confidence": 0.2, "atr_percentile": 0.5}
        self.assertEqual(_regime_stop_widening_multiplier(reading), REGIME_LOW_CONFIDENCE_STOP_MULT)

    def test_high_volatility_widens_stop(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_HIGH_VOL_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        reading = {"confidence": 0.9, "atr_percentile": 0.95}
        self.assertEqual(_regime_stop_widening_multiplier(reading), REGIME_HIGH_VOL_STOP_MULT)

    def test_percent_scale_atr_percentile_does_not_turn_quiet_tape_into_high_vol(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_HIGH_VOL_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        quiet_percent_scale = {"confidence": 0.9, "atr_percentile": 35.0}
        hot_percent_scale = {"confidence": 0.9, "atr_percentile": 95.0}

        self.assertEqual(_regime_stop_widening_multiplier(quiet_percent_scale), 1.0)
        self.assertEqual(_regime_stop_widening_multiplier(hot_percent_scale), REGIME_HIGH_VOL_STOP_MULT)

    def test_widening_is_clamped_at_max(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_MAX_STOP_WIDEN,
            _regime_stop_widening_multiplier,
        )

        # Both signals trigger; result should not exceed the documented ceiling.
        reading = {"confidence": 0.1, "atr_percentile": 0.99}
        result = _regime_stop_widening_multiplier(reading)
        self.assertLessEqual(result, REGIME_MAX_STOP_WIDEN)

    def test_missing_reading_does_not_widen(self) -> None:
        from quant_rabbit.strategy.intent_generator import _regime_stop_widening_multiplier

        self.assertEqual(_regime_stop_widening_multiplier(None), 1.0)
        self.assertEqual(_regime_stop_widening_multiplier({}), 1.0)


class RangeRewardRiskFloorTest(unittest.TestCase):
    """Risk policy must allow rr < min_reward_risk for RANGE entries.

    Regression: before the regime-aware floor, range_trader rotations were
    forced to ≥1.2R even when the opposing rail capped TP closer. The
    risk validator must read `intent.metadata['regime_state']` and apply
    `policy.range_min_reward_risk` instead of the default floor.
    """

    def test_range_state_uses_range_min_reward_risk(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        policy = RiskPolicy()
        self.assertLess(policy.range_min_reward_risk, policy.min_reward_risk)

    def test_default_min_reward_risk_unchanged_for_non_range(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        # The default floor for trend/breakout/unclear remains conservative.
        self.assertGreaterEqual(RiskPolicy().min_reward_risk, 1.2)


class PerTradeFloorTest(unittest.TestCase):
    """Per-trade risk must be floored when pace×budget shrinks below an
    equity-derived minimum, but only when pace was not explicitly set by
    operator CLI. Per feedback_high_conviction_execution.md.
    """

    def test_floor_applied_when_pace_is_derived(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        policy = RiskPolicy()
        self.assertIsNotNone(policy.min_per_trade_risk_pct)
        self.assertGreater(policy.min_per_trade_risk_pct, 0.0)


class NavPctSizingTest(unittest.TestCase):
    """`_nav_pct_position_units` and the SL-free sizing precedence path.

    User directive 2026-05-08「BaseUnitを決めると、資産が増えたときに追従
    できないよ。％で決めないといけなくない？」: position size must be
    NAV-relative so it auto-scales with equity.
    """

    def _account(self, nav_jpy: float = 227000.0):
        from quant_rabbit.models import AccountSummary
        return AccountSummary(
            balance_jpy=nav_jpy,
            nav_jpy=nav_jpy,
            margin_used_jpy=0.0,
            margin_available_jpy=nav_jpy,
            unrealized_pl_jpy=0.0,
            financing_jpy=0.0,
            pl_jpy=0.0,
            fetched_at_utc=datetime.now(timezone.utc),
            hedging_enabled=True,
            last_transaction_id="0",
        )

    def _snapshot(self, nav_jpy: float = 227000.0):
        from quant_rabbit.models import BrokerSnapshot, Quote
        return BrokerSnapshot(
            fetched_at_utc=datetime.now(timezone.utc),
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote(
                    pair="EUR_USD",
                    bid=1.17280,
                    ask=1.17290,
                    timestamp_utc=datetime.now(timezone.utc),
                ),
                "USD_JPY": Quote(
                    pair="USD_JPY",
                    bid=156.886,
                    ask=156.894,
                    timestamp_utc=datetime.now(timezone.utc),
                ),
            },
            account=self._account(nav_jpy),
            home_conversions={"USD": 157.0, "JPY": 1.0},
        )

    def test_nav_pct_returns_none_when_env_unset(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        prior = os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
        try:
            result = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot())
            self.assertIsNone(result)
        finally:
            if prior is not None:
                os.environ["QR_TRADER_POSITION_NAV_PCT"] = prior

    def test_nav_pct_30_yields_about_10000u_for_eur_usd_at_227k_nav(self) -> None:
        # 30% × 227,000 JPY = 68,100 JPY margin.
        # EUR_USD at 1.17290 with USDJPY=157 → margin/u ≈ 1.17290 × 157 × 0.04
        # = 7.366 JPY/u. → 68,100 / 7.366 ≈ 9,245 units.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            result = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot())
            self.assertIsNotNone(result)
            assert result is not None
            self.assertGreater(result, 8500.0)
            self.assertLess(result, 10500.0)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_auto_scales_with_higher_nav(self) -> None:
        # Bumping NAV from 227k to 250k should grow units proportionally.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            small = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=227000.0))
            big = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=250000.0))
            self.assertIsNotNone(small)
            self.assertIsNotNone(big)
            assert small is not None and big is not None
            # 250/227 × small ≈ big within 5% tolerance.
            expected_ratio = 250000.0 / 227000.0
            actual_ratio = big / small
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.02)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_auto_scales_down_with_lower_nav(self) -> None:
        # Drawdown to 200k should shrink units proportionally.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            normal = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=227000.0))
            shrunk = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=200000.0))
            self.assertIsNotNone(normal)
            self.assertIsNotNone(shrunk)
            assert normal is not None and shrunk is not None
            self.assertLess(shrunk, normal)
            expected_ratio = 200000.0 / 227000.0
            actual_ratio = shrunk / normal
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.02)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_invalid_value_returns_none(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        for bad in ("abc", "0", "-5", "  "):
            os.environ["QR_TRADER_POSITION_NAV_PCT"] = bad
            try:
                self.assertIsNone(_nav_pct_position_units("EUR_USD", 1.17290, self._snapshot()))
            finally:
                os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)


class MinLotFloorIntentTest(unittest.TestCase):
    """Coverage for 2026-05-12 emergency fix B in
    `_risk_budgeted_units` + the `MARGIN_TOO_THIN_FOR_MIN_LOT` BLOCK that
    the intent_generator now emits when the budget can only fund a
    sub-`MIN_PRODUCTION_LOT_UNITS` lot. The bug surfaced when 470901 (201u
    EUR/USD), 470904 (322u AUD/JPY), 470907 (2u GBP/USD) all filled at
    micro size after a tight-margin cycle. Each lot's round-trip spread
    cost exceeded any realistic pip target — guaranteed-loss trades.
    """

    def _stub_snapshot(self, margin_used: float = 0.0, margin_available: float = 200000.0):
        from quant_rabbit.models import AccountSummary, BrokerSnapshot, Quote
        now = datetime.now(timezone.utc)
        return BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote(pair="EUR_USD", bid=1.17280, ask=1.17290, timestamp_utc=now),
                "USD_JPY": Quote(pair="USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            home_conversions={"USD": 157.005},
            account=AccountSummary(
                balance_jpy=227000.0,
                nav_jpy=227000.0,
                margin_used_jpy=margin_used,
                margin_available_jpy=margin_available,
                unrealized_pl_jpy=0.0,
                financing_jpy=0.0,
                pl_jpy=0.0,
                fetched_at_utc=now,
                hedging_enabled=True,
                last_transaction_id="0",
            ),
        )

    def setUp(self) -> None:
        import os
        self._prior = os.environ.pop("QR_ALLOW_TEST_MICRO_LOT", None)

    def tearDown(self) -> None:
        import os
        if self._prior is None:
            os.environ.pop("QR_ALLOW_TEST_MICRO_LOT", None)
        else:
            os.environ["QR_ALLOW_TEST_MICRO_LOT"] = self._prior

    def test_risk_budgeted_units_returns_zero_when_budget_subfloor(self) -> None:
        # Budget so small that loss_budget_units < 1000.
        # max_loss_jpy=50 JPY, stop ≈ 20 pip → loss_budget ≈ 159 units.
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=50.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertEqual(units, 0)

    def test_risk_budgeted_units_returns_1000_when_just_at_floor(self) -> None:
        # max_loss_jpy ~315 JPY → loss_budget ≈ 1003 units → rounds to 1000.
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=315.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertGreaterEqual(units, 1000)
        self.assertEqual(units % 1000, 0)  # rounded down to 1000-step

    def test_risk_budgeted_units_returns_5000_for_clear_budget(self) -> None:
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=2000.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertGreaterEqual(units, 5000)
        self.assertEqual(units % 1000, 0)

    def test_test_micro_lot_override_restores_legacy_fallback(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        os.environ["QR_ALLOW_TEST_MICRO_LOT"] = "1"
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=50.0,
            snapshot=self._stub_snapshot(),
        )
        # Override active → falls back to legacy `max(1, int(max_units))`.
        self.assertGreater(units, 0)
        self.assertLess(units, 1000)

    def test_account_none_keeps_legacy_fallback_for_test_fixtures(self) -> None:
        # Fixture-style snapshot without an account must not trigger the
        # production floor — many legacy test fixtures construct snapshots
        # without an `AccountSummary` and rely on the historical
        # micro-unit fallback.
        from quant_rabbit.models import BrokerSnapshot, Quote
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        now = datetime.now(timezone.utc)
        no_account = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote(pair="EUR_USD", bid=1.17280, ask=1.17290, timestamp_utc=now),
                "USD_JPY": Quote(pair="USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            home_conversions={"USD": 157.005},
            account=None,
        )
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=50.0,
            snapshot=no_account,
        )
        self.assertGreater(units, 0)


class ExhaustionRangeChaseTest(unittest.TestCase):
    """Coverage for 2026-05-13 filter C: refuse same-direction entries
    after a 2σ-equivalent 24h range extension. Operates via
    `_method_context_issues` against the intent's metadata, so it
    fires at intent-generation time without touching open positions.
    """

    def _intent(self, *, side, sigma_mult, price_pct_24h, pair: str = "EUR_USD"):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
        return OrderIntent(
            pair=pair,
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.MARKET,
            units=5000,
            tp=1.18 if side == "LONG" else 1.17,
            sl=1.17 if side == "LONG" else 1.18,
            thesis="test thesis",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="test",
                chart_story="test",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="sl trades",
            ),
            metadata={
                "range_24h_sigma_multiple": sigma_mult,
                "price_percentile_24h": price_pct_24h,
            },
        )

    def test_long_at_top_after_2sigma_range_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=2.5, price_pct_24h=0.92)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_short_at_bottom_after_2sigma_range_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="SHORT", sigma_mult=2.5, price_pct_24h=0.08)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_long_at_bottom_after_2sigma_range_passes(self) -> None:
        # LONG mean-reversion entry at low — not a chase.
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=2.5, price_pct_24h=0.10)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_sigma_below_threshold_passes(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=1.5, price_pct_24h=0.97)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_missing_sigma_no_block(self) -> None:
        # AGENT_CONTRACT §3.5: no data → no filter.
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=None, price_pct_24h=0.97)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)


if __name__ == "__main__":
    unittest.main()
