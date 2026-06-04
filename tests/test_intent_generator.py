from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.strategy.intent_generator import IntentGenerator


class IntentGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._prior_require_forecast_for_live = os.environ.pop(
            "QR_REQUIRE_FORECAST_FOR_LIVE",
            None,
        )
        self._prior_require_telemetry_for_live = os.environ.pop(
            "QR_REQUIRE_TELEMETRY_FOR_LIVE",
            None,
        )

    def tearDown(self) -> None:
        if self._prior_require_forecast_for_live is None:
            os.environ.pop("QR_REQUIRE_FORECAST_FOR_LIVE", None)
        else:
            os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = self._prior_require_forecast_for_live
        if self._prior_require_telemetry_for_live is None:
            os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
        else:
            os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = self._prior_require_telemetry_for_live

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

    def test_live_entry_requires_fresh_forecast_when_live_default_active(self) -> None:
        prior = os.environ.get("QR_REQUIRE_FORECAST_FOR_LIVE")
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                summary = IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_REQUIRE_FORECAST_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = prior

        self.assertEqual(summary.live_ready, 0)
        self.assertGreater(summary.dry_run_passed, 0)
        issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
        blockers = [blocker for item in payload["results"] for blocker in item["live_blockers"]]
        self.assertIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertTrue(any("no fresh executable pair forecast" in blocker for blocker in blockers))

    def test_live_entry_requires_forecast_telemetry_when_live_default_active(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                    patch(
                        "quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry",
                        return_value=None,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {issue["code"] for item in short_results for issue in item["risk_issues"]}
        self.assertEqual(summary.live_ready, 0)
        self.assertTrue(short_results)
        self.assertTrue(any(item["status"] == "DRY_RUN_PASSED" for item in short_results))
        self.assertTrue(all(item["status"] != "LIVE_READY" for item in short_results))
        self.assertIn("TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE", issue_codes)

    def test_generate_intents_records_pre_entry_forecast_telemetry_before_live_validation(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                snapshot = _snapshot(root)
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
                rows = (root / "data" / "forecast_history.jsonl").read_text().splitlines()
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        self.assertTrue(rows)
        latest = json.loads(rows[-1])
        self.assertEqual(latest["pair"], "EUR_USD")
        self.assertTrue(str(latest["cycle_id"]).startswith("pre-entry-forecast-refresh:"))

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE", issue_codes)
        self.assertNotIn("TELEMETRY_DIRECTIONAL_PROJECTION_REQUIRED_FOR_LIVE", issue_codes)

    def test_matching_forecast_telemetry_allows_live_ready(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                snapshot = _snapshot(root)
                _write_forecast_telemetry(root, direction="DOWN", confidence=0.91)
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        self.assertGreater(summary.live_ready, 0)
        self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
        self.assertTrue(
            all(
                "TELEMETRY_" not in issue["code"]
                for item in short_results
                for issue in item["risk_issues"]
            )
        )

    def test_same_cycle_forecast_telemetry_allows_snapshot_timestamp_ordering_skew(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime.now(timezone.utc) + timedelta(minutes=5)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=fetched_at.isoformat(),
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
                forecast_rows = (root / "data" / "forecast_history.jsonl").read_text().splitlines()
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        latest = json.loads(forecast_rows[-1])
        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertGreater(summary.live_ready, 0)
        self.assertEqual(latest["pair"], "EUR_USD")
        self.assertEqual(latest["cycle_id"], short_results[0]["intent"]["metadata"]["forecast_cycle_id"])
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", issue_codes)

    def test_forecast_telemetry_from_different_cycle_cannot_be_live_ready(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime.now(timezone.utc) - timedelta(minutes=2)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=fetched_at.isoformat(),
                )
                _write_forecast_telemetry(root, direction="DOWN", confidence=0.91, cycle_id="older-cycle")
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                    patch(
                        "quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry",
                        return_value=None,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertEqual(summary.live_ready, 0)
        self.assertIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", issue_codes)

    def test_snapshot_packet_time_prevents_self_stale_intents(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime.now(timezone.utc) - timedelta(minutes=2)
                quote_at = fetched_at - timedelta(seconds=10)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=quote_at.isoformat(),
                )
                _write_forecast_telemetry(root, direction="DOWN", confidence=0.91)
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        stale_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
            if issue["code"] == "STALE_QUOTE"
        }
        self.assertGreater(summary.live_ready, 0)
        self.assertFalse(stale_codes)

    def test_watch_only_strategy_profile_cannot_become_live_ready_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="WATCH_ONLY"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        statuses = {item["status"] for item in payload["results"]}
        blockers = [blocker for item in payload["results"] for blocker in item["live_blockers"]]
        self.assertNotIn("LIVE_READY", statuses)
        self.assertIn("DRY_RUN_PASSED", statuses)
        self.assertTrue(any("WATCH_ONLY" in blocker for blocker in blockers))

    def test_forecast_seed_pending_trigger_can_repair_watch_only_profile_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="UP",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1762,
                    invalidation_price=1.1718,
                    horizon_min=60,
                    rationale_summary="UP forecast from current tape",
                    drivers_for=("sell-side sweep fade",),
                    drivers_against=("old profile is watch-only",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="SHORT"),
                        strategy_profile=_strategy(root, status="WATCH_ONLY", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.82,
                            short_score=0.18,
                            dominant_regime="TREND_UP",
                            m5_regime="TREND_UP",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        trigger = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
        )
        market = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET"
        )
        self.assertTrue(trigger["intent"]["metadata"]["forecast_seed"])
        self.assertEqual(trigger["status"], "LIVE_READY")
        self.assertEqual(trigger["live_blockers"], [])
        self.assertEqual(market["status"], "DRY_RUN_PASSED")
        self.assertTrue(any("WATCH_ONLY" in blocker for blocker in market["live_blockers"]))

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

    def test_carries_tf_location_and_nearby_level_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                pair_charts_path=_pair_charts_location_map(root),
                levels_path=_levels_snapshot(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            intent = payload["results"][0]["intent"]
            metadata = intent["metadata"]

            self.assertEqual(metadata["tf_regime_map"]["M5"]["classification"], "RANGE")
            self.assertEqual(metadata["tf_regime_map"]["H1"]["classification"], "TREND_UP")
            self.assertIn("M5", metadata["range_timeframes"])
            self.assertIn("H1:TREND_UP", metadata["trend_timeframes"])
            self.assertTrue(metadata["nearest_levels_below"])
            self.assertTrue(metadata["nearest_levels_above"])
            self.assertTrue(metadata["level_clusters_near"])
            self.assertTrue(all(item["count"] > 1 for item in metadata["level_clusters_near"]))
            self.assertTrue(
                any(
                    str(item["source"]).startswith("levels:")
                    for item in metadata["nearest_levels_below"] + metadata["nearest_levels_above"]
                )
            )
            self.assertIn("M5 RANGE", metadata["market_location_story"])
            self.assertIn("H1 TREND_UP", metadata["market_location_story"])
            self.assertIn(metadata["market_location_story"], intent["market_context"]["chart_story"])

    def test_forecast_first_seed_adds_current_direction_before_mirror_candidates(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1688,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.18,
                            short_score=0.82,
                            dominant_regime="TREND_DOWN",
                            m5_regime="TREND_DOWN",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                results = payload["results"]
                seed = next(
                    item
                    for item in results
                    if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                )

                self.assertTrue(seed["intent"]["metadata"]["forecast_seed"])
                self.assertEqual(seed["intent"]["metadata"]["forecast_direction"], "DOWN")
                self.assertEqual(seed["intent"]["metadata"]["forecast_confidence"], 0.91)
                self.assertEqual(seed["intent"]["tp"], 1.1688)
                self.assertEqual(seed["intent"]["metadata"]["tp_target_source"], "FORECAST_CAPPED_ATR_RR")
                self.assertIn("forecast-first", seed["intent"]["market_context"]["narrative"])
                self.assertEqual(
                    sum(
                        1
                        for item in results
                        if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                    ),
                    1,
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_forecast_first_blocks_opposite_direction_candidates(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                long_results = [item for item in payload["results"] if item["intent"]["side"] == "LONG"]
                short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
                long_issue_codes = {
                    issue["code"]
                    for item in long_results
                    for issue in item["risk_issues"]
                    if issue["severity"] == "BLOCK"
                }

                self.assertGreater(summary.live_ready, 0)
                self.assertTrue(short_results)
                self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
                self.assertTrue(long_results)
                self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in long_results))
                self.assertIn("FORECAST_DIRECTION_CONFLICT", long_issue_codes)
                self.assertTrue(
                    all(
                        item["intent"]["metadata"]["forecast_direction"] == "DOWN"
                        and item["intent"]["metadata"]["forecast_confidence"] == 0.91
                        for item in payload["results"]
                    )
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_forecast_tp_does_not_cap_when_target_fails_execution_floor(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1708,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.18,
                            short_score=0.82,
                            dominant_regime="TREND_DOWN",
                            m5_regime="TREND_DOWN",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                seed = next(
                    item
                    for item in payload["results"]
                    if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                )

                self.assertTrue(seed["intent"]["metadata"]["forecast_seed"])
                self.assertEqual(seed["intent"]["metadata"]["forecast_target_price"], 1.1708)
                self.assertNotEqual(seed["intent"]["tp"], 1.1708)
                self.assertEqual(seed["intent"]["metadata"]["tp_target_source"], "ATR_RR")
                self.assertIn(
                    "forecast target skipped: forecast TP RR",
                    seed["intent"]["metadata"]["tp_target_reason"],
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_below_entry_forecast_still_marks_existing_lanes_as_forecast_context(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.51,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast below fresh-entry threshold",
                drivers_for=("pair_chart LONG_LEAN",),
                drivers_against=("confidence below entry gate",),
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.72,
                        short_score=0.28,
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
        self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", issue_codes)
        self.assertNotIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertTrue(
            all(
                item["intent"]["metadata"]["forecast_direction"] == "UP"
                and item["intent"]["metadata"]["forecast_confidence"] == 0.51
                and not item["intent"]["metadata"]["forecast_seed"]
                for item in payload["results"]
                if item["intent"]["pair"] == "EUR_USD"
            )
        )

    def test_reversal_recovery_hedge_uses_recovery_forecast_floor_for_live_context(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import (
            _forecast_live_readiness_issue,
            _method_context_issues,
            _telemetry_live_readiness_issues,
        )

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        recovery_metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_timing_class": "REVERSAL",
            "forecast_direction": "UP",
            "forecast_confidence": 0.51,
            "forecast_target_price": 1.1762,
            "forecast_invalidation_price": 1.1718,
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="reversal recovery hedge",
            market_context=MarketContext(
                regime="UNCLEAR current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=recovery_metadata,
        )

        fresh_entry_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.STOP_ENTRY,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="fresh entry",
                market_context=intent.market_context,
                metadata={**recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            ),
            {**recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            TradeMethod.TREND_CONTINUATION,
        )

        self.assertIsNone(_forecast_live_readiness_issue(intent, recovery_metadata, TradeMethod.TREND_CONTINUATION))
        self.assertEqual(fresh_entry_issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

        range_recovery_metadata = {
            **recovery_metadata,
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.82,
        }
        self.assertIsNone(
            _forecast_live_readiness_issue(intent, range_recovery_metadata, TradeMethod.TREND_CONTINUATION)
        )
        range_fresh_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="fresh range trend entry",
                market_context=intent.market_context,
                metadata={**range_recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            ),
            {**range_recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            TradeMethod.TREND_CONTINUATION,
        )
        self.assertEqual(range_fresh_issue["code"], "RANGE_FORECAST_REQUIRES_RANGE_ROTATION")

        chart_reversal_metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_timing_class": "REVERSAL",
            "chart_score_balance": "LONG_LEAN",
            "chart_score_gap": 0.183,
            "pattern_reversal_dominant_side": "LONG",
            "pattern_reversal_weight_long": 44.25,
            "pattern_reversal_weight_short": 30.42,
            "trend_timeframes": ["M1:TREND_UP", "M5:TREND_UP"],
        }
        market_recovery = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="chart-confirmed reversal recovery hedge",
            market_context=intent.market_context,
            metadata=chart_reversal_metadata,
        )
        self.assertIsNone(
            _forecast_live_readiness_issue(market_recovery, chart_reversal_metadata, TradeMethod.TREND_CONTINUATION)
        )

        now = datetime.now(timezone.utc)
        with patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            return_value={
                "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                "direction": "UNCLEAR",
                "confidence": 0.18,
                "cycle_id": "cycle",
            },
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            return_value=0,
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            telemetry_codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    market_recovery,
                    chart_reversal_metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                    ),
                    now,
                )
            }
        self.assertNotIn("TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE", telemetry_codes)

        weak_chart_reversal = {
            **chart_reversal_metadata,
            "pattern_reversal_dominant_side": "SHORT",
            "pattern_reversal_weight_long": 33.75,
            "pattern_reversal_weight_short": 36.08,
        }
        weak_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="weak recovery hedge",
                market_context=intent.market_context,
                metadata=weak_chart_reversal,
            ),
            weak_chart_reversal,
            TradeMethod.TREND_CONTINUATION,
        )
        self.assertEqual(weak_issue["code"], "FORECAST_CONTEXT_REQUIRED_FOR_LIVE")

        tolerated_confidence_metadata = {
            **chart_reversal_metadata,
            "forecast_direction": "UP",
            "forecast_confidence": 0.6133,
            "forecast_cycle_id": "cycle",
        }
        tolerated_confidence_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="rounded confidence recovery hedge",
            market_context=intent.market_context,
            metadata=tolerated_confidence_metadata,
        )

        def telemetry_codes_for_latest_confidence(latest_confidence: float) -> set[str]:
            with patch(
                "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
                return_value={
                    "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                    "direction": "UP",
                    "confidence": latest_confidence,
                    "cycle_id": "cycle",
                },
            ), patch(
                "quant_rabbit.strategy.intent_generator._directional_projection_recorded",
                return_value=True,
            ), patch(
                "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
                return_value=0,
            ), patch(
                "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
                return_value=None,
            ):
                return {
                    issue["code"]
                    for issue in _telemetry_live_readiness_issues(
                        tolerated_confidence_intent,
                        tolerated_confidence_metadata,
                        BrokerSnapshot(
                            fetched_at_utc=now,
                            quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                        ),
                        now,
                    )
                }

        self.assertNotIn(
            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
            telemetry_codes_for_latest_confidence(0.6134),
        )
        self.assertIn(
            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
            telemetry_codes_for_latest_confidence(0.616),
        )

        opposed = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1730,
            tp=1.1700,
            sl=1.1750,
            thesis="opposed reversal recovery hedge",
            market_context=intent.market_context,
            metadata=recovery_metadata,
        )
        opposed_codes = {issue["code"] for issue in _method_context_issues(opposed)}
        self.assertIn("FORECAST_DIRECTION_CONFLICT", opposed_codes)

    def test_range_rotation_uses_range_probability_floor_for_live_context(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.51,
            "geometry_model": "RANGE_RAIL_LIMIT",
            "range_support": 1.1618,
            "range_resistance": 1.1650,
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=4000,
            entry=1.1619,
            tp=1.1633,
            sl=1.1601,
            thesis="range rail rotation",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        self.assertIsNone(_forecast_live_readiness_issue(intent, metadata, TradeMethod.RANGE_ROTATION))

        weak_metadata = {**metadata, "forecast_confidence": 0.49}
        weak_issue = _forecast_live_readiness_issue(intent, weak_metadata, TradeMethod.RANGE_ROTATION)
        self.assertEqual(weak_issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

    def test_projection_expiry_grace_avoids_same_cycle_false_blocker(self) -> None:
        from quant_rabbit.strategy.intent_generator import _expired_pending_projection_count

        emitted_at = datetime(2026, 6, 1, 6, 56, 21, tzinfo=timezone.utc)
        row = {
            "timestamp_emitted_utc": emitted_at.isoformat().replace("+00:00", "Z"),
            "pair": "EUR_USD",
            "signal_name": "bb_squeeze_expansion_imminent",
            "direction": "EITHER",
            "resolution_window_min": 150.0,
            "resolution_status": "PENDING",
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "projection_ledger.jsonl").write_text(json.dumps(row) + "\n")
            with patch(
                "quant_rabbit.strategy.intent_generator.PROJECTION_PENDING_EXPIRY_GRACE_SECONDS",
                300.0,
            ):
                just_after_expiry = emitted_at + timedelta(minutes=150, seconds=60)
                stale_after_grace = emitted_at + timedelta(minutes=150, seconds=301)

                self.assertEqual(
                    _expired_pending_projection_count(
                        data_root=data_root,
                        validation_time_utc=just_after_expiry,
                    ),
                    0,
                )
                self.assertEqual(
                    _expired_pending_projection_count(
                        data_root=data_root,
                        validation_time_utc=stale_after_grace,
                    ),
                    1,
                )

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

    def test_sl_free_still_blocks_decisive_trend_continuation_conflict(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                summary = IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                issues = [issue for item in payload["results"] for issue in item["risk_issues"]]
                long_results = [item for item in payload["results"] if item["intent"]["side"] == "LONG"]
                short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]

                self.assertGreater(summary.live_ready, 0)
                self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in long_results))
                self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
                self.assertTrue(
                    any(
                        issue["code"] == "CHART_DIRECTION_CONFLICT"
                        and issue["severity"] == "BLOCK"
                        and "trend-continuation hard gate" in issue["message"]
                        for issue in issues
                    ),
                    f"expected hard trend conflict blocker, got {issues}",
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_preserves_range_and_failure_entries_when_trend_lane_blocks(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)

                range_output = root / "range_intents.json"
                IntentGenerator(
                    campaign_plan=_range_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=range_output,
                    report_path=root / "range_intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="RANGE",
                        m5_long_bias=0.76,
                        m5_short_bias=0.12,
                        regime_quantile="QUIET",
                        atr_pips=3.2,
                    ),
                    max_loss_jpy=140.0,
                ).run(snapshot_path=_snapshot(root, eur_bid=1.17110, eur_ask=1.17118))

                range_results = json.loads(range_output.read_text())["results"]
                self.assertTrue(
                    any(item["status"] == "LIVE_READY" for item in range_results),
                    f"range/fade entries should remain available, got {range_results}",
                )

                failure_output = root / "failure_intents.json"
                IntentGenerator(
                    campaign_plan=_trigger_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=failure_output,
                    report_path=root / "failure_intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                failure_results = json.loads(failure_output.read_text())["results"]
                self.assertEqual({item["intent"]["order_type"] for item in failure_results}, {"LIMIT", "STOP-ENTRY"})
                self.assertTrue(
                    any(item["status"] == "LIVE_READY" for item in failure_results),
                    f"breakout-failure trigger entries should remain available, got {failure_results}",
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

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

            self.assertEqual(summary.generated, 2)
            self.assertEqual(order_types, {"LIMIT", "STOP-ENTRY"})
            self.assertEqual(
                lane_ids,
                {
                    "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                    "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                },
            )
            self.assertFalse(any(lane_id.endswith(":MARKET") for lane_id in lane_ids))

    def test_breakout_failure_generates_retest_limit_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            by_lane_id = {item["lane_id"]: item for item in payload["results"]}
            parent = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"

            self.assertEqual(summary.generated, 3)
            self.assertEqual(
                {lane_id: item["intent"]["order_type"] for lane_id, item in by_lane_id.items()},
                {
                    f"{parent}:LIMIT": "LIMIT",
                    parent: "STOP-ENTRY",
                    f"{parent}:MARKET": "MARKET",
                },
            )
            limit = by_lane_id[f"{parent}:LIMIT"]
            self.assertEqual(limit["intent"]["metadata"]["parent_lane_id"], parent)
            self.assertEqual(limit["intent"]["metadata"]["order_timing"], "PENDING_TRIGGER")

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
                pair_charts_path=_pair_charts(root),
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
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=1_000.0,
            ).run(snapshot_path=_snapshot(root, usd_jpy=300.0))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # Unit count depends on conversion rate AND the (now ATR-derived)
            # SL distance — assert risk fits cap rather than fixed unit count.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1_000.0)

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
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["intent"]["metadata"]["target_reward_risk"], 4.0)
            self.assertAlmostEqual(result["risk_metrics"]["reward_risk"], 4.0)

    def test_small_wave_attaches_structural_harvest_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_tp_mode(root, adx=16.0, atr_percentile=0.2),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            metadata = result["intent"]["metadata"]
            self.assertTrue(metadata["attach_take_profit_on_fill"])
            self.assertEqual(metadata["tp_execution_mode"], "ATTACHED_TECHNICAL_TP")
            self.assertEqual(metadata["tp_target_source"], "STRUCTURAL_HARVEST")
            self.assertIn("ADX", metadata["tp_attach_reason"])

    def test_attached_harvest_missing_structure_uses_operating_floor_tp(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.SHORT,
            method=TradeMethod.BREAKOUT_FAILURE,
            order_type=OrderType.MARKET,
            quote=Quote(pair="EUR_USD", bid=1.16264, ask=1.16272),
            entry=1.16272,
            tp=1.14355,
            sl=1.16728,
            reward_risk=4.2,
            execution_regime="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 6.5},
            pair_chart={"pair": "EUR_USD", "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}]},
            atr_pips=6.8,
        )

        self.assertEqual(tp, 1.15998)
        self.assertEqual(metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertEqual(metadata["tp_target_distance_pips"], 27.4)
        self.assertEqual(metadata["virtual_take_profit_reward_risk"], 0.601)
        self.assertIn("structural anchor missing", metadata["tp_target_reason"])

    def test_recovery_hedge_missing_harvest_structure_uses_operating_floor_tp(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.LONG,
            method=TradeMethod.TREND_CONTINUATION,
            order_type=OrderType.STOP_ENTRY,
            quote=Quote(pair="EUR_USD", bid=1.16264, ask=1.16272),
            entry=1.16288,
            tp=1.17200,
            sl=1.16220,
            reward_risk=4.2,
            execution_regime="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 2.8},
            pair_chart={
                "pair": "EUR_USD",
                "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}],
            },
            atr_pips=6.8,
            hedge_recovery=True,
        )

        self.assertEqual(tp, 1.16329)
        self.assertEqual(metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertLessEqual(metadata["tp_target_distance_pips"], 27.2)
        self.assertIn("structural anchor missing", metadata["tp_target_reason"])

    def test_recovery_hedge_intent_passes_recovery_state_into_tp_plan(self) -> None:
        from quant_rabbit.models import (
            AccountSummary,
            BrokerPosition,
            BrokerSnapshot,
            OrderType,
            Owner,
            Quote,
            Side,
        )
        from quant_rabbit.strategy.intent_generator import _intent_from_lane, _method_context_issues

        now = datetime.now(timezone.utc)
        quote = Quote(pair="EUR_USD", bid=1.16264, ask=1.16272, timestamp_utc=now)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="trapped-short",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=22_000,
                    entry_price=1.15800,
                    unrealized_pl_jpy=-23_000.0,
                    owner=Owner.TRADER,
                ),
            ),
            orders=(),
            quotes={
                "EUR_USD": quote,
                "USD_JPY": Quote(pair="USD_JPY", bid=156.64, ask=156.648, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=220_000.0,
                margin_used_jpy=120_000.0,
                margin_available_jpy=80_000.0,
                hedging_enabled=True,
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 156.64},
        )
        intent = _intent_from_lane(
            {
                "desk": "trend_trader",
                "pair": "EUR_USD",
                "direction": "LONG",
                "method": "TREND_CONTINUATION",
                "adoption": "RISK_REPAIR_DRY_RUN",
                "campaign_role": "NOW_IF_REPAIRED",
                "reason": "reversal recovery hedge",
                "required_receipt": "dry-run under loss cap",
                "forecast_direction": "UP",
                "forecast_confidence": 0.51,
                "forecast_target_price": 1.1662,
                "forecast_invalidation_price": 1.1618,
            },
            quote,
            snapshot,
            max_loss_jpy=500.0,
            atr_pips=6.8,
            order_type_override=OrderType.STOP_ENTRY,
            regime_state="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 2.8},
            pair_chart={
                "pair": "EUR_USD",
                "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}],
            },
        )

        self.assertEqual(intent.metadata["position_intent"], "HEDGE")
        self.assertTrue(intent.metadata["hedge_recovery"])
        self.assertEqual(intent.metadata["hedge_timing_class"], "REVERSAL")
        self.assertEqual(intent.metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertNotIn(
            "HARVEST_TP_STRUCTURE_MISSING",
            {issue["code"] for issue in _method_context_issues(intent)},
        )

    def test_strong_trend_uses_runner_no_broker_tp_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_tp_mode(root, adx=31.0, atr_percentile=0.8),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            metadata = result["intent"]["metadata"]
            self.assertFalse(metadata["attach_take_profit_on_fill"])
            self.assertEqual(metadata["tp_execution_mode"], "RUNNER_NO_BROKER_TP")
            self.assertEqual(metadata["tp_target_source"], "STRUCTURAL_EXTEND")
            self.assertIn("qualifies as runner", metadata["tp_attach_reason"])

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

    def test_stable_current_range_synthesizes_range_rotation_from_trend_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="RANGE",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )

            self.assertEqual(range_limit["status"], "LIVE_READY")
            self.assertEqual(range_limit["intent"]["order_type"], "LIMIT")
            self.assertEqual(range_limit["intent"]["metadata"]["geometry_model"], "RANGE_RAIL_LIMIT")

    def test_breakout_pending_range_does_not_synthesize_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}

            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)

    def test_squeeze_range_does_not_synthesize_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="RANGE",
                    bb_squeeze=True,
                    atr_percentile=0.12,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}

            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)

    def test_range_forming_synthesizes_limit_only_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="UNCLEAR",
                    m5_regime="TREND_WEAK",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="TREND_WEAK",
                    adx=18.0,
                    choppiness=55.0,
                    bb_width_percentile=0.30,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )

            self.assertIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)
            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION:MARKET", lane_ids)
            self.assertEqual(range_limit["intent"]["order_type"], "LIMIT")
            self.assertEqual(range_limit["intent"]["metadata"]["range_phase"], "RANGE_FORMING")

    def test_confirmed_range_breakout_synthesizes_breakout_stop_entry_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.64,
                    short_score=0.36,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.66,
                    m5_short_bias=0.34,
                    regime_quantile="NORMAL",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                    close=1.17645,
                    trend_score=0.75,
                    breakout_score=0.85,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            breakout = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
            )

            self.assertEqual(breakout["intent"]["order_type"], "STOP-ENTRY")
            self.assertEqual(breakout["intent"]["metadata"]["range_phase"], "BREAKOUT_UP")
            self.assertEqual(breakout["intent"]["metadata"]["range_breakout_direction"], "UP")

    def test_existing_range_rotation_blocks_during_breakout_pending(self) -> None:
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
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            issue_codes = {issue["code"] for issue in range_limit["risk_issues"]}

            self.assertEqual(range_limit["status"], "DRY_RUN_BLOCKED")
            self.assertIn("RANGE_PHASE_NOT_ROTATION", issue_codes)

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
                pair_charts_path=_pair_charts(root),
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

    def test_recovery_hedge_uses_conviction_scaled_tranche_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                IntentGenerator(
                    campaign_plan=_campaign(root, direction="SHORT"),
                    strategy_profile=_strategy(root, status="MINE_MISSED_EDGE", direction="SHORT"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.25,
                        short_score=0.75,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    max_loss_jpy=20_000.0,
                ).run(snapshot_path=_snapshot_with_underwater_long(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        result = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
        )
        intent = result["intent"]
        metadata = intent["metadata"]

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_recovery_reference_units"], 22_000)
        self.assertLess(intent["units"], 22_000)
        self.assertEqual(intent["units"], metadata["hedge_recovery_units"])
        self.assertGreaterEqual(intent["units"], 1_000)
        self.assertLess(metadata["hedge_recovery_size_scale"], 1.0)
        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")
        self.assertTrue(metadata["hedge_unwind_plan_required"])

    def test_recovery_hedge_reversal_confirmation_allows_larger_time_efficient_tranche(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import (
            RECOVERY_HEDGE_CONTINUATION_MAX_SCALE,
            _hedge_timing_metadata,
            _recovery_hedge_sizing_metadata,
        )

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.75,
            "chart_long_score": 0.25,
            "chart_score_gap": -0.50,
            "tf_agreement_score": 0.80,
            "chart_story_structural": "M15 BOS_DOWN close_confirmed after failed reclaim",
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))
        sizing = _recovery_hedge_sizing_metadata(Side.SHORT, metadata, chart_context, {})

        self.assertEqual(metadata["hedge_timing_class"], "REVERSAL")
        self.assertGreater(sizing["hedge_recovery_size_scale"], RECOVERY_HEDGE_CONTINUATION_MAX_SCALE)
        self.assertGreater(sizing["hedge_recovery_units"], 7_000)

    def test_recovery_hedge_m5_opposition_prevents_reversal_classification(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _hedge_timing_metadata

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.75,
            "chart_long_score": 0.25,
            "chart_score_gap": -0.50,
            "tf_agreement_score": 0.80,
            "m5_long_bias": 0.625,
            "m5_short_bias": 0.25,
            "chart_story_structural": (
                "M5(FAILURE_RISK struct=CHOCH_UP@1.1593); "
                "H1(TREND_DOWN struct=BOS_DOWN@1.1594)"
            ),
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))

        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")

    def test_recovery_hedge_market_blocks_when_m5_opposes_side(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=6000,
            tp=1.15181,
            sl=1.16131,
            thesis="market recovery hedge while M5 opposes",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "position_intent": "HEDGE",
                "hedge_recovery": True,
                "chart_direction_bias": Side.SHORT.value,
                "chart_score_gap": -0.42,
                "m5_long_bias": 0.625,
                "m5_short_bias": 0.25,
                "chart_story_structural": (
                    "M5(FAILURE_RISK struct=CHOCH_UP@1.1593); "
                    "H1(TREND_DOWN struct=BOS_DOWN@1.1594)"
                ),
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["RECOVERY_HEDGE_MARKET_OPPOSED_BY_M5"]["severity"], "BLOCK")

    def test_breakout_failure_market_blocks_short_before_upper_retest(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.16136,
            tp=1.15971,
            sl=1.16198,
            thesis="short failed-break market chase near support",
            market_context=MarketContext(
                regime="UNCLEAR current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "classification": "TREND_WEAK",
                        "range_position": 0.226,
                        "nearest_support_distance_pips": -0.74,
                        "nearest_resistance_distance_pips": 2.54,
                    },
                    "M15": {
                        "classification": "RANGE",
                        "range_position": 0.041,
                        "nearest_support_distance_pips": -0.25,
                        "nearest_resistance_distance_pips": 5.75,
                    },
                }
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["BREAKOUT_FAILURE_MARKET_NOT_RETESTED"]["severity"], "BLOCK")

    def test_breakout_failure_market_allows_short_after_upper_retest(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.16176,
            tp=1.15971,
            sl=1.16224,
            thesis="short failed-break after resistance retest",
            market_context=MarketContext(
                regime="RANGE current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "classification": "RANGE",
                        "range_position": 0.82,
                        "nearest_support_distance_pips": -4.2,
                        "nearest_resistance_distance_pips": 0.6,
                    },
                    "M15": {
                        "classification": "RANGE",
                        "range_position": 0.71,
                        "nearest_support_distance_pips": -6.4,
                        "nearest_resistance_distance_pips": 1.8,
                    },
                }
            },
        )

        issues = _method_context_issues(intent)
        issue_codes = {issue["code"] for issue in issues}

        self.assertNotIn("BREAKOUT_FAILURE_MARKET_NOT_RETESTED", issue_codes)

    def test_breakout_failure_blocks_distant_atr_rr_when_harvest_tp_missing(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=3000,
            entry=1.15964,
            tp=1.15017,
            sl=1.16172,
            thesis="failed break must not use far ATR/RR fallback",
            market_context=MarketContext(
                regime="TREND_DOWN current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_target_source": "ATR_RR",
                "tp_target_reason": "structural target too close; using ATR/RR virtual target",
                "tp_target_distance_pips": 94.7,
                "tp_atr_pips": 3.0,
                "position_intent": "HEDGE",
                "hedge_recovery": True,
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["HARVEST_TP_STRUCTURE_MISSING"]["severity"], "BLOCK")

    def test_recovery_hedge_without_reversal_confirmation_caps_continuation_chase(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import (
            RECOVERY_HEDGE_CONTINUATION_MAX_SCALE,
            _hedge_timing_metadata,
            _recovery_hedge_sizing_metadata,
        )

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.52,
            "chart_long_score": 0.48,
            "chart_score_gap": -0.04,
            "tf_agreement_score": 0.34,
            "chart_story_structural": "M15 range churn without close-confirmed BOS",
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))
        sizing = _recovery_hedge_sizing_metadata(Side.SHORT, metadata, chart_context, {})

        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")
        self.assertLessEqual(sizing["hedge_recovery_size_scale"], RECOVERY_HEDGE_CONTINUATION_MAX_SCALE)
        self.assertLess(sizing["hedge_recovery_units"], 11_000)


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


def _breakout_failure_campaign(root: Path) -> Path:
    path = root / "breakout_failure_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "failure_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW",
                        "reason": "failed downside break reclaimed support",
                        "required_receipt": "wait for failed-break retest or confirmed trigger; no blind chase.",
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


def _pair_charts_location_map(root: Path) -> Path:
    path = root / "pair_charts_location_map.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "long_score": 0.66,
                        "short_score": 0.22,
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "RANGE",
                                "long_bias": 0.54,
                                "short_bias": 0.46,
                                "regime_reading": {"state": "RANGE", "confidence": 0.72, "atr_percentile": 28.0},
                                "indicators": {
                                    "close": 1.17325,
                                    "atr_pips": 6.0,
                                    "adx_14": 16.0,
                                    "choppiness_14": 64.0,
                                    "linreg_slope_20": 0.01,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1732,
                                    "avwap_anchor": 1.1731,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_UP",
                                "long_bias": 0.72,
                                "short_bias": 0.18,
                                "regime_reading": {"state": "TREND_UP", "confidence": 0.77, "atr_percentile": 68.0},
                                "indicators": {
                                    "close": 1.17325,
                                    "atr_pips": 24.0,
                                    "adx_14": 31.0,
                                    "choppiness_14": 38.0,
                                    "linreg_slope_20": 0.52,
                                    "bb_lower": 1.1688,
                                    "bb_upper": 1.1802,
                                    "bb_middle": 1.1745,
                                    "donchian_low": 1.1684,
                                    "donchian_high": 1.1810,
                                    "vwap": 1.1729,
                                    "avwap_anchor": 1.1719,
                                    "linreg_channel_lower": 1.1700,
                                    "linreg_channel_upper": 1.1790,
                                    "swing_low": 1.1678,
                                    "swing_high": 1.1820,
                                },
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _levels_snapshot(root: Path) -> Path:
    path = root / "levels_snapshot.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair": "EUR_USD",
                        "pdh": 1.1762,
                        "pdl": 1.1691,
                        "pdc": 1.1728,
                        "daily_open": 1.1718,
                        "weekly_open": 1.1700,
                        "monthly_open": 1.1650,
                        "last_close": 1.1732,
                        "pivots": [
                            {
                                "style": "STANDARD",
                                "pp": 1.1727,
                                "r1": 1.1763,
                                "r2": 1.1792,
                                "s1": 1.1692,
                                "s2": 1.1664,
                            }
                        ],
                        "sessions": [
                            {"name": "ASIA", "high": 1.1742, "low": 1.1711, "range_pips": 31.0},
                            {"name": "LONDON", "high": 1.1764, "low": 1.1708, "range_pips": 56.0},
                        ],
                        "round_numbers": [
                            {"price": 1.17, "distance_pips": -33.0},
                            {"price": 1.175, "distance_pips": 17.0},
                        ],
                    }
                ],
                "issues": [],
            }
        )
    )
    return path


def _pair_charts_tp_mode(root: Path, *, adx: float, atr_percentile: float) -> Path:
    path = root / "pair_charts_tp_mode.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "long_score": 0.78,
                        "short_score": 0.12,
                        "confluence": {
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.66,
                            "dominant_regime": "TREND_UP",
                            "tf_agreement_score": 1.0,
                            "atr_percentile_24h": atr_percentile,
                            "range_24h_sigma_multiple": 1.1,
                        },
                        "session": {"current_tag": "LONDON_NY_OVERLAP"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "long_bias": 0.78,
                                "short_bias": 0.12,
                                "regime_reading": {"state": "TREND_UP", "confidence": 0.8, "atr_percentile": 70.0},
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 28.0, "adx_14": adx},
                                "smc": {"dealing_range": {"swing_high": 1.2300, "swing_low": 1.1500}},
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 42.0, "adx_14": adx},
                                "smc": {"dealing_range": {"swing_high": 1.2300, "swing_low": 1.1500}},
                            },
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
    regime_state: str = "TREND_WEAK",
    bb_squeeze: bool = False,
    atr_percentile: float = 50.0,
    adx: float = 22.0,
    choppiness: float = 50.0,
    bb_width_percentile: float = 50.0,
    close: float = 1.1735,
    trend_score: float = 0.2,
    breakout_score: float = 0.1,
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
                        "confluence": {
                            "score_balance": (
                                "TIED"
                                if abs(round(long_score - short_score, 4)) <= 0.05
                                else ("LONG_LEAN" if long_score > short_score else "SHORT_LEAN")
                            ),
                            "score_gap": round(long_score - short_score, 4),
                            "dominant_regime": dominant_regime,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": m5_regime,
                                "long_bias": long_score if m5_long_bias is None else m5_long_bias,
                                "short_bias": short_score if m5_short_bias is None else m5_short_bias,
                                "regime_reading": {"state": regime_state, "confidence": 0.6, "atr_percentile": atr_percentile},
                                "family_scores": {
                                    "mean_rev_score": 1.1,
                                    "trend_score": trend_score,
                                    "breakout_score": breakout_score,
                                    "disagreement": 0.2,
                                },
                                "indicators": {
                                    "close": close,
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
                                    "adx_14": adx,
                                    "choppiness_14": choppiness,
                                    "atr_percentile_100": atr_percentile,
                                    "bb_width_percentile_100": bb_width_percentile,
                                    "bb_squeeze": bb_squeeze,
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
    fetched_at_utc: str | None = None,
    quote_timestamp_utc: str | None = None,
) -> Path:
    path = root / "snapshot.json"
    now = fetched_at_utc or datetime.now(timezone.utc).isoformat()
    quote_time = quote_timestamp_utc or now
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": eur_bid, "ask": eur_ask, "timestamp_utc": quote_time},
                    "USD_JPY": {"bid": usd_jpy, "ask": usd_jpy + 0.008, "timestamp_utc": quote_time},
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


def _write_forecast_telemetry(
    root: Path,
    *,
    direction: str,
    confidence: float,
    cycle_id: str = "test-cycle",
) -> None:
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (data_root / "forecast_history.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "cycle_id": cycle_id,
                "pair": "EUR_USD",
                "direction": direction,
                "confidence": confidence,
                "current_price": 1.17326,
                "target_price": 1.1712,
                "invalidation_price": 1.1742,
                "horizon_min": 60,
            }
        )
        + "\n"
    )
    (data_root / "projection_ledger.jsonl").write_text(
        json.dumps(
            {
                "timestamp_emitted_utc": ts,
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": direction,
                "lead_time_min": 60,
                "confidence": confidence,
                "entry_price": 1.17326,
                "predicted_target_price": 1.1712,
                "predicted_invalidation_price": 1.1742,
                "resolution_window_min": 120,
                "resolution_status": "PENDING",
                "cycle_id": cycle_id,
            }
        )
        + "\n"
    )


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


def _snapshot_with_underwater_long(root: Path) -> Path:
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
                        "units": 22_000,
                        "entry_price": 1.16688,
                        "unrealized_pl_jpy": -23_000.0,
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": 1.16042, "ask": 1.16050, "timestamp_utc": now},
                    "USD_JPY": {"bid": 156.64, "ask": 156.648, "timestamp_utc": now},
                },
                "account": {
                    "nav_jpy": 170_000.0,
                    "balance_jpy": 192_000.0,
                    "margin_used_jpy": 162_000.0,
                    "margin_available_jpy": 7_000.0,
                    "unrealized_pl_jpy": -23_000.0,
                    "hedging_enabled": True,
                    "fetched_at_utc": now,
                },
                "home_conversions": {"USD": 156.64, "JPY": 1.0},
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

    def _stub_snapshot(self, margin_used: float = 0.0, margin_available: float = 200000.0, positions=()):
        from quant_rabbit.models import AccountSummary, BrokerSnapshot, Quote
        now = datetime.now(timezone.utc)
        return BrokerSnapshot(
            fetched_at_utc=now,
            positions=tuple(positions),
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

    def test_risk_budgeted_units_caps_same_pair_hedge_by_loss_budget(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        open_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.17290,
                sl=1.17490,
                max_loss_jpy=2_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(open_long,),
                ),
                side=Side.SHORT,
                position_intent="HEDGE",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

        self.assertEqual(units, 6_000)

    def test_risk_budgeted_units_ignores_manual_opposing_units_for_hedge_target(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_700,
            entry_price=1.16013,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.16013,
                sl=1.16317,
                max_loss_jpy=10_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(trader_long, manual_long, existing_short),
                ),
                side=Side.SHORT,
                position_intent="HEDGE",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

        self.assertEqual(units, 1_000)

    def test_risk_budgeted_units_uses_manual_exposure_for_broker_margin_offset(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_nav_pct = os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=8_400,
            entry_price=1.16013,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.16013,
                sl=1.16317,
                max_loss_jpy=10_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(trader_long, manual_long, existing_short),
                ),
                side=Side.SHORT,
                position_intent="PYRAMID",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_nav_pct is not None:
                os.environ["QR_TRADER_POSITION_NAV_PCT"] = prior_nav_pct

        self.assertEqual(units, 3_000)

    def test_position_intent_metadata_marks_underwater_opposite_side_as_recovery_hedge(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7000,
            entry_price=1.16688,
            unrealized_pl_jpy=-500.0,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1000.0,
            owner=Owner.UNKNOWN,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, manual_long)))

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_reference_scope"], "trader_owned_only")
        self.assertEqual(metadata["hedge_recovery_units"], 7_000)
        self.assertEqual(metadata["hedge_recovery_unrealized_pl_jpy"], -500.0)
        self.assertEqual(metadata["hedge_non_trader_opposing_units_ignored"], 15_000)
        self.assertEqual(metadata["hedge_non_trader_opposing_unrealized_pl_jpy_ignored"], -1000.0)

    def test_position_intent_metadata_caps_recovery_hedge_to_uncovered_opposite_units(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_000,
            entry_price=1.17000,
            unrealized_pl_jpy=400.0,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, existing_short)))

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_gross_opposing_units"], 22_000)
        self.assertEqual(metadata["hedge_existing_same_side_units"], 5_000)
        self.assertEqual(metadata["hedge_recovery_units"], 17_000)

    def test_position_intent_metadata_suppresses_covered_recovery_hedge(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=22_000,
            entry_price=1.17000,
            unrealized_pl_jpy=900.0,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, existing_short)))

        self.assertEqual(metadata["position_intent"], "PYRAMID")
        self.assertEqual(metadata["hedge_reference_units"], 0)
        self.assertEqual(metadata["hedge_existing_same_side_units"], 22_000)
        self.assertEqual(metadata["hedge_suppressed_reason"], "opposite_exposure_already_covered")

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

    def _intent(
        self,
        *,
        side,
        sigma_mult,
        price_pct_24h,
        pair: str = "EUR_USD",
        metadata_extra: dict | None = None,
        order_type: str = "MARKET",
        method: str = "TREND_CONTINUATION",
        entry: float | None = None,
    ):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
        metadata = {
            "range_24h_sigma_multiple": sigma_mult,
            "price_percentile_24h": price_pct_24h,
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        return OrderIntent(
            pair=pair,
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=5000,
            entry=entry,
            tp=1.18 if side == "LONG" else 1.17,
            sl=1.17 if side == "LONG" else 1.18,
            thesis="test thesis",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="test",
                chart_story="test",
                method=TradeMethod.parse(method),
                invalidation="sl trades",
            ),
            metadata=metadata,
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

    def test_recovery_hedge_market_at_extended_side_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(
            side="SHORT",
            sigma_mult=2.5,
            price_pct_24h=0.08,
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(
            issue for issue in _method_context_issues(intent) if issue["code"] == "EXHAUSTION_RANGE_CHASE"
        )
        self.assertEqual(issue["severity"], "BLOCK")

    def test_recovery_hedge_stop_at_extended_side_warns_instead_of_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(
            side="SHORT",
            sigma_mult=2.5,
            price_pct_24h=0.08,
            order_type="STOP-ENTRY",
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(
            issue for issue in _method_context_issues(intent) if issue["code"] == "EXHAUSTION_RANGE_CHASE"
        )
        self.assertEqual(issue["severity"], "WARN")

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

    def test_failed_break_short_market_at_upper_retest_not_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"range_position": 0.80},
                    "M15": {"range_position": 0.78},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_stop_near_lower_side_still_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            order_type="STOP-ENTRY",
            entry=1.16080,
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16000, "nearest_resistance": 1.16400},
                    "M15": {"nearest_support": 1.15900, "nearest_resistance": 1.16500},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_limit_at_upper_retest_not_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            order_type="LIMIT",
            entry=1.16360,
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16000, "nearest_resistance": 1.16400},
                    "M15": {"nearest_support": 1.15900, "nearest_resistance": 1.16500},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_range_rotation_short_at_tiny_upper_rail_but_broader_discount_still_chases(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=8.5,
            price_pct_24h=0.09,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16089,
            metadata_extra={
                "price_percentile_7d": 0.04,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.15900, "nearest_resistance": 1.16100},
                    "M15": {"nearest_support": 1.15880, "nearest_resistance": 1.16120},
                },
            },
        )
        issues = _method_context_issues(intent)
        issue = next(issue for issue in issues if issue["code"] == "EXHAUSTION_RANGE_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_range_rotation_long_at_broader_premium_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=None,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16670,
            metadata_extra={
                "price_percentile_7d": 0.92,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16650, "nearest_resistance": 1.16740},
                    "M15": {"nearest_support": 1.16580, "nearest_resistance": 1.16820},
                },
            },
        )
        issues = _method_context_issues(intent)
        issue = next(issue for issue in issues if issue["code"] == "RANGE_ROTATION_BROADER_LOCATION_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_range_rotation_short_at_broader_premium_keeps_retest_carveout(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=7.0,
            price_pct_24h=0.86,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16500,
            metadata_extra={
                "price_percentile_7d": 0.83,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16380, "nearest_resistance": 1.16520},
                    "M15": {"nearest_support": 1.16270, "nearest_resistance": 1.16580},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)


class BreakoutFailureStopChaseTest(unittest.TestCase):
    def _intent(self, *, side: str, order_type: str, entry: float):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod

        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=3000,
            entry=entry,
            tp=entry + 0.0018 if side == "LONG" else entry - 0.0018,
            sl=entry - 0.0012 if side == "LONG" else entry + 0.0012,
            thesis="failed-break retest timing test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="FAILURE_RISK",
                narrative="breakout failure lane",
                chart_story="failed break and retest map",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="sl trades",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "nearest_support": 1.16000,
                        "nearest_resistance": 1.16400,
                        "nearest_support_distance_pips": -20.0,
                        "nearest_resistance_distance_pips": 20.0,
                    },
                    "M15": {
                        "nearest_support": 1.15900,
                        "nearest_resistance": 1.16500,
                        "nearest_support_distance_pips": -30.0,
                        "nearest_resistance_distance_pips": 30.0,
                    },
                },
            },
        )

    def test_short_stop_entry_on_lower_half_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="SHORT", order_type="STOP-ENTRY", entry=1.16080)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)

    def test_long_stop_entry_on_upper_half_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="LONG", order_type="STOP-ENTRY", entry=1.16320)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)

    def test_short_limit_at_resistance_does_not_get_stop_chase_block(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="SHORT", order_type="LIMIT", entry=1.16360)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)


class PatternReversalChaseTest(unittest.TestCase):
    def _intent(
        self,
        *,
        side: str = "SHORT",
        order_type: str = "STOP-ENTRY",
        method: str = "TREND_CONTINUATION",
        metadata_extra: dict | None = None,
    ):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod

        entry = 1.16080
        metadata = {
            "pattern_reversal_dominant_side": "LONG",
            "pattern_reversal_weight_long": 22.5,
            "pattern_reversal_weight_short": 6.0,
            "pattern_signals": [
                {
                    "name": "failed_breakout",
                    "timeframe": "M15",
                    "direction": "UP",
                    "side": "LONG",
                    "weight": 11.25,
                    "chase_block_evidence": True,
                    "rationale": "M15 BOS_DOWN wick-only (close_confirmed=False) -> trap fade UP",
                },
                {
                    "name": "hammer",
                    "timeframe": "M5",
                    "direction": "UP",
                    "side": "LONG",
                    "weight": 11.25,
                    "chase_block_evidence": True,
                    "rationale": "M5 hammer at lower rail -> UP",
                },
                {
                    "name": "aroon_strong_down",
                    "timeframe": "M15",
                    "direction": "DOWN",
                    "side": "SHORT",
                    "weight": 6.0,
                    "chase_block_evidence": False,
                    "rationale": "M15 aroon momentum DOWN",
                },
            ],
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=3000,
            entry=entry,
            tp=entry + 0.0018 if side == "LONG" else entry - 0.0018,
            sl=entry - 0.0012 if side == "LONG" else entry + 0.0012,
            thesis="pattern reversal chase test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_DOWN",
                narrative="trend lane",
                chart_story="failed break / candle-shape evidence",
                method=TradeMethod.parse(method),
                invalidation="sl trades",
            ),
            metadata=metadata,
        )

    def test_short_stop_entry_against_failed_break_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent()
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("PATTERN_REVERSAL_CHASE", codes)

    def test_breakout_failure_market_against_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(order_type="MARKET", method="BREAKOUT_FAILURE")
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("PATTERN_REVERSAL_CHASE", codes)

    def test_retest_limit_does_not_get_pattern_chase_block(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(order_type="LIMIT", method="BREAKOUT_FAILURE")
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)

    def test_close_confirmed_operating_break_allows_continuation(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            metadata_extra={
                "chart_story_structural": (
                    "EUR_USD TREND_DOWN; "
                    "M5(TREND_DOWN struct=BOS_DOWN@1.1580); "
                    "M15(TREND_DOWN struct=CHOCH_DOWN@1.1575)"
                ),
                "m5_long_bias": 0.30,
                "m5_short_bias": 0.70,
                "m15_long_bias": 0.35,
                "m15_short_bias": 0.65,
            }
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)

    def test_recovery_hedge_against_reversal_warns(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True})
        issue = next(issue for issue in _method_context_issues(intent) if issue["code"] == "PATTERN_REVERSAL_CHASE")

        self.assertEqual(issue["severity"], "WARN")

    def test_recovery_hedge_market_against_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            order_type="MARKET",
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(issue for issue in _method_context_issues(intent) if issue["code"] == "PATTERN_REVERSAL_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_recovery_hedge_market_buy_into_short_reversal_and_exhaustion_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            order_type="MARKET",
            metadata_extra={
                "position_intent": "HEDGE",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "pattern_reversal_dominant_side": "SHORT",
                "pattern_reversal_weight_long": 33.75,
                "pattern_reversal_weight_short": 39.11,
                "pattern_signals": [
                    {
                        "name": "rsi_extreme_top",
                        "timeframe": "M5",
                        "direction": "DOWN",
                        "side": "SHORT",
                        "weight": 11.25,
                        "chase_block_evidence": True,
                        "rationale": "M5 top exhaustion -> DOWN",
                    },
                ],
                "range_24h_sigma_multiple": 8.657,
                "price_percentile_24h": 0.83,
            },
        )
        issues = {issue["code"]: issue for issue in _method_context_issues(intent)}

        self.assertEqual(issues["PATTERN_REVERSAL_CHASE"]["severity"], "BLOCK")
        self.assertEqual(issues["EXHAUSTION_RANGE_CHASE"]["severity"], "BLOCK")

    def test_aligned_pattern_dominance_passes(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            metadata_extra={
                "pattern_reversal_dominant_side": "SHORT",
                "pattern_reversal_weight_long": 6.0,
                "pattern_reversal_weight_short": 22.5,
                "pattern_signals": [
                    {
                        "name": "shooting_star",
                        "timeframe": "M15",
                        "direction": "DOWN",
                        "side": "SHORT",
                        "weight": 11.25,
                        "chase_block_evidence": True,
                        "rationale": "M15 shooting star -> DOWN",
                    }
                ],
            }
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)


if __name__ == "__main__":
    unittest.main()
