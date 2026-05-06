from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.gpt_trader import GPTTraderBrain, StaticTraderProvider


LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"


class GPTTraderBrainTest(unittest.TestCase):
    def test_accepts_schema_valid_evidence_cited_live_ready_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(summary.selected_lane_id, LANE_ID)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertIn("GPT Trader Decision Report", (root / "gpt_decision.md").read_text())

    def test_accepts_batch_trade_when_selected_lanes_are_live_ready_and_cited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_lane = f"{LANE_ID}:MARKET"
            files["intents"].write_text(
                json.dumps({"results": [_result(), _result(lane_id=market_lane)]})
            )
            brain = _brain(root, files, _batch_trade_decision([LANE_ID, market_lane]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_ids, (LANE_ID, market_lane))
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_batch_trade_when_selected_lane_is_not_cited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_lane = f"{LANE_ID}:MARKET"
            files["intents"].write_text(
                json.dumps({"results": [_result(), _result(lane_id=market_lane)]})
            )
            decision = _batch_trade_decision([LANE_ID, market_lane])
            decision["evidence_refs"].remove(f"intent:{market_lane}")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELECTED_LANE_EVIDENCE_MISSING", codes)

    def test_accepts_trade_when_existing_position_is_protected_and_trader_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_operator_manual_position_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-470201",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "units": 25000,
                        "entry_price": 155.962,
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                    }
                ],
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_operator_manual_pending_order_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[{**_pending_order(), "order_id": "manual-pending", "owner": "unknown"}])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_trader_pending_entry_for_gateway_basket_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_rejects_trade_when_broker_exposure_is_not_layerable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position(stop_loss=None)])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("EXPOSURE_BLOCKS_TRADE", codes)

    def test_rejects_hallucinated_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["evidence_refs"] = ["broker:snapshot", "legacy:invented-row"]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_EVIDENCE_REF", codes)

    def test_accepts_missing_option_skew_evidence_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["evidence_refs"].append("option:skew:unknown")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_rejects_stale_request_evidence_when_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _request_evidence_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("REQUEST_EVIDENCE_WITH_LIVE_READY_LANES", codes)

    def test_rejects_wait_when_flat_target_open_and_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            self.assertFalse(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)

    def test_accepts_wait_when_trader_exposure_is_already_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_cancel_pending_without_order_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _cancel_pending_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MISSING_CANCEL_ORDER_IDS", codes)

    def test_accepts_cancel_pending_with_current_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertEqual(payload["decision"]["cancel_order_ids"], ["pending-1"])

    def test_cli_uses_external_decision_response_without_model_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_decision_response.json"
            decision_response.write_text(json.dumps(_trade_decision()))

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "gpt-trader-decision",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--intents",
                        str(files["intents"]),
                        "--campaign-plan",
                        str(files["campaign"]),
                        "--strategy-profile",
                        str(files["strategy"]),
                        "--market-story-profile",
                        str(files["story"]),
                        "--target-state",
                        str(files["target"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--decision-response",
                        str(decision_response),
                        "--output",
                        str(root / "cli_decision.json"),
                        "--report",
                        str(root / "cli_decision.md"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads((root / "cli_decision.json").read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["decision"]["selected_lane_id"], LANE_ID)

    def test_default_packet_includes_range_lane_beyond_first_eight_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            range_lane = "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            filler = [
                _result(lane_id=f"candidate_{idx}:EUR_USD:LONG:TREND_CONTINUATION")
                for idx in range(9)
            ]
            files["intents"].write_text(
                json.dumps({"results": [*filler, _result(lane_id=range_lane, method="RANGE_ROTATION")]})
            )
            brain = _brain(root, files, _trade_decision(lane_id=range_lane, method="RANGE_ROTATION"))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, range_lane)

    def test_live_ready_lane_beyond_packet_cap_is_still_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            filler = [
                _result(lane_id=f"candidate_{idx}:EUR_USD:LONG:TREND_CONTINUATION")
                for idx in range(13)
            ]
            files["intents"].write_text(json.dumps({"results": [*filler, _result()]}))
            brain = _brain(root, files, _trade_decision(), max_lanes=12)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, LANE_ID)
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet_lane_ids = {lane["lane_id"] for lane in payload["input_packet"]["lanes"]}
            self.assertIn(LANE_ID, packet_lane_ids)
            self.assertGreater(len(packet_lane_ids), 12)

    def test_packet_includes_market_context_payloads_not_only_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            market_context = payload["input_packet"]["market_context"]
            eur = market_context["pairs"]["EUR_USD"]

            self.assertEqual(eur["chart"]["dominant_regime"], "TREND_UP")
            self.assertEqual(eur["chart"]["views"]["M5"]["atr_pips"], 5.3)
            self.assertEqual(eur["chart"]["views"]["M5"]["regime_state"], "TREND_STRONG")
            self.assertEqual(eur["flow"]["spread"]["stress_flag"], "NORMAL")
            self.assertEqual(eur["levels"]["pdh"], 1.18)
            self.assertFalse(eur["calendar"]["in_window"])
            self.assertEqual(market_context["currency_strength"]["USD"]["rank"], 2)
            self.assertEqual(market_context["cot"]["USD"]["leveraged_net"], 1234)

    def test_accepts_attack_advice_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "coverage_pct": 49.0,
                        "recommended_now_lane_ids": [LANE_ID],
                        "recommended_now_reward_jpy": 900.0,
                        "recommended_now_risk_jpy": 300.0,
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["input_packet"]["ai_attack_advice"]["recommended_now_lane_ids"], [LANE_ID])
            self.assertFalse(payload["input_packet"]["ai_attack_advice"]["live_permission"])

    def test_rejects_wait_when_attack_advice_recommends_lane_even_with_trader_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)

    def test_rejects_trade_that_ignores_attack_advice_recommended_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            alternative_lane = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(),
                            _result(lane_id=alternative_lane, method="BREAKOUT_FAILURE"),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            decision = _trade_decision(lane_id=alternative_lane, method="BREAKOUT_FAILURE")
            decision["evidence_refs"].append("attack:advice")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_IGNORED", codes)

    def test_rejects_trade_that_skips_first_attack_priority_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            priority_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=priority_lane),
                            _result(),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [priority_lane, LANE_ID],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_PRIORITY_SKIPPED", codes)

    def test_accepts_attack_priority_lane_in_selected_basket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            priority_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=priority_lane),
                            _result(),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [priority_lane, LANE_ID],
                    }
                )
            )
            decision = _batch_trade_decision([priority_lane, LANE_ID])
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{priority_lane}", f"attack:lane:{LANE_ID}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_recommended_trade_without_attack_advice_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_EVIDENCE_MISSING", codes)
            self.assertIn("ATTACK_ADVICE_LANE_EVIDENCE_MISSING", codes)

    def test_rejects_strategy_review_that_uses_wrong_method_for_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["strategy_reviews"] = [
                {
                    "lane_id": LANE_ID,
                    "method": "RANGE_ROTATION",
                    "verdict": "SUPPORTS",
                    "summary": "wrong review method should not authorize the selected trend lane",
                }
            ]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("STRATEGY_REVIEW_METHOD_MISMATCH", codes)


def _brain(root: Path, files: dict[str, Path], decision: dict, *, max_lanes: int | None = None) -> GPTTraderBrain:
    return GPTTraderBrain(
        provider=StaticTraderProvider(decision),
        intents_path=files["intents"],
        campaign_plan_path=files["campaign"],
        strategy_profile_path=files["strategy"],
        market_story_profile_path=files["story"],
        target_state_path=files["target"],
        output_path=root / "gpt_decision.json",
        report_path=root / "gpt_decision.md",
        pair_charts_path=files["pair_charts"],
        cross_asset_path=files["cross_asset"],
        flow_path=files["flow"],
        currency_strength_path=files["currency_strength"],
        levels_path=files["levels"],
        calendar_path=files["calendar"],
        cot_path=files["cot"],
        option_skew_path=files["option_skew"],
        attack_advice_path=files["attack_advice"],
        **({"max_lanes": max_lanes} if max_lanes is not None else {}),
    )


def _fixtures(root: Path, *, positions: list[dict] | None = None, orders: list[dict] | None = None) -> dict[str, Path]:
    files = {
        "snapshot": root / "snapshot.json",
        "intents": root / "intents.json",
        "campaign": root / "campaign.json",
        "strategy": root / "strategy.json",
        "story": root / "story.json",
        "target": root / "target.json",
        "pair_charts": root / "pair_charts.json",
        "cross_asset": root / "cross_asset.json",
        "flow": root / "flow.json",
        "currency_strength": root / "currency_strength.json",
        "levels": root / "levels.json",
        "calendar": root / "calendar.json",
        "cot": root / "cot.json",
        "option_skew": root / "option_skew.json",
        "attack_advice": root / "attack_advice.json",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": positions or [],
                "orders": orders or [],
                "quotes": {"EUR_USD": {"bid": 1.172, "ask": 1.1721, "timestamp_utc": now}},
            }
        )
    )
    files["intents"].write_text(json.dumps({"results": [_result()]}))
    files["campaign"].write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW_IF_CLEAN",
                        "required_receipt": "live-ready continuation receipt",
                    }
                ]
            }
        )
    )
    files["strategy"].write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5200,
                        "live_net_jpy": 1800,
                        "live_worst_jpy": -350,
                    }
                ]
            }
        )
    )
    files["story"].write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 12},
                        "themes": {"momentum": 4},
                        "examples": ["EUR_USD trend-bull staircase continuation"],
                    }
                ]
            }
        )
    )
    files["target"].write_text(
        json.dumps(
            {
                "status": "PURSUE_TARGET",
                "target_jpy": 22278.1,
                "progress_jpy": 0.0,
                "remaining_target_jpy": 22278.1,
                "remaining_risk_budget_jpy": 500.0,
            }
        )
    )
    files["pair_charts"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "chart_story": "EUR_USD trend-up test story",
                        "long_score": 0.8,
                        "short_score": 0.2,
                        "session": {
                            "current_tag": "NY_AM_KILLZONE",
                            "jp_holiday": False,
                            "judas_armed": False,
                            "ny_midnight_open_price": 1.17,
                        },
                        "views": [
                            {
                                "granularity": "M5",
                                "indicators": {
                                    "atr_pips": 5.3,
                                    "adx_14": 42.0,
                                    "rsi_14": 61.0,
                                    "choppiness_14": 39.0,
                                    "bb_width_percentile_100": 0.62,
                                    "atr_percentile_100": 0.71,
                                },
                                "regime_reading": {
                                    "state": "TREND_STRONG",
                                    "confidence": 0.82,
                                    "hurst": 0.58,
                                },
                                "family_scores": {
                                    "trend_score": 1.2,
                                    "mean_rev_score": -0.4,
                                    "breakout_score": 0.3,
                                    "disagreement": 0.35,
                                },
                                "stat_filters": {
                                    "last_jump_bars_ago": 8,
                                    "lag1_autocorr": 0.12,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )
    files["cross_asset"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "synthetic_dxy": {"last_value": 98.1, "change_pct_24h": -0.2},
                "yield_spreads": [{"name": "US10Y_minus_US2Y", "spread_last": 7.4}],
                "assets": [{"instrument": "USB10Y_USD", "trend_label": "UP", "last_price": 110.5}],
                "correlations": {"EUR_USD": {"USB10Y_USD": 0.15}},
                "issues": [],
            }
        )
    )
    files["flow"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "spreads": [
                    {
                        "instrument": "EUR_USD",
                        "current_pips": 0.8,
                        "median_pips": 1.2,
                        "p90_pips": 1.7,
                        "stress_flag": "NORMAL",
                    }
                ],
                "issues": [],
            }
        )
    )
    files["currency_strength"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "scores": [
                    {"currency": "EUR", "rank": 1, "score_pct": 0.4},
                    {"currency": "USD", "rank": 2, "score_pct": 0.2},
                ],
                "strongest_pair_suggestion": "EUR_USD",
                "issues": [],
            }
        )
    )
    files["levels"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "pairs": [
                    {
                        "pair": "EUR_USD",
                        "pdh": 1.18,
                        "pdl": 1.16,
                        "pdc": 1.17,
                        "daily_open": 1.171,
                        "pivots": [{"style": "STANDARD", "pp": 1.17, "r1": 1.18, "s1": 1.16}],
                        "round_numbers": [{"price": 1.18, "distance_pips": 8.0}],
                    }
                ],
                "issues": [],
            }
        )
    )
    files["calendar"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "pair_windows": [
                    {
                        "pair": "EUR_USD",
                        "in_window": False,
                        "reason": "next event outside window",
                        "next_event": {"currency": "USD", "impact": "Medium", "title": "ADP"},
                    }
                ],
                "issues": [],
            }
        )
    )
    files["cot"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "reports": [
                    {"currency": "USD", "leveraged_net": 1234, "week_change_leveraged_net": 56},
                    {"currency": "EUR", "leveraged_net": -789, "week_change_leveraged_net": -12},
                ],
                "issues": [],
            }
        )
    )
    files["option_skew"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "readings": [
                    {"pair": "EUR_USD", "tenor": "1W", "rr_25d": None, "issue": "MISSING_OPTION_SKEW_FEED"}
                ],
                "issues": ["MISSING_OPTION_SKEW_FEED"],
            }
        )
    )
    files["attack_advice"].write_text(json.dumps({}))
    return files


def _result(*, lane_id: str = LANE_ID, method: str = "TREND_CONTINUATION") -> dict:
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_allowed": True,
        "risk_issues": [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": "EUR_USD",
            "side": "LONG",
            "order_type": "STOP-ENTRY",
            "units": 1000,
            "entry": 1.1725,
            "tp": 1.1737,
            "sl": 1.1717,
            "thesis": "EUR_USD continuation can pay before daily target window closes.",
            "owner": "trader",
            "market_context": {
                "regime": f"{method} campaign lane",
                "narrative": "Dollar pressure and momentum theme favor EUR_USD continuation.",
                "chart_story": "Higher lows are pressing into the trigger shelf.",
                "method": method,
                "invalidation": "Invalid if the shelf breaks before entry.",
                "event_risk": "",
                "session": "test",
            },
        },
    }


def _trade_decision(*, lane_id: str = LANE_ID, method: str = "TREND_CONTINUATION") -> dict:
    return {
        "action": "TRADE",
        "selected_lane_id": lane_id,
        "confidence": "HIGH",
        "thesis": "The live-ready EUR_USD continuation lane has current story and positive mined evidence.",
        "method": method,
        "narrative": "Momentum and campaign role align with a controlled stop-entry.",
        "chart_story": "Higher lows press into the trigger shelf.",
        "invalidation": "Do not trade if the shelf fails before entry or the SL level trades.",
        "rejected_alternatives": ["WAIT rejected because the target gap remains open and lane is clean."],
        "risk_notes": ["Use only the lane units, TP, and SL already verified by the dry-run receipt."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            f"intent:{lane_id}",
            f"campaign:{lane_id}",
            "strategy:EUR_USD:LONG",
            "story:EUR_USD",
        ],
        "operator_summary": "Accept the verified EUR_USD continuation lane.",
    }


def _batch_trade_decision(lane_ids: list[str]) -> dict:
    decision = _trade_decision(lane_id=lane_ids[0])
    decision["selected_lane_ids"] = lane_ids
    refs = list(decision["evidence_refs"])
    for lane_id in lane_ids[1:]:
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}"])
    decision["evidence_refs"] = refs
    decision["operator_summary"] = "Accept the verified EUR_USD continuation basket."
    return decision


def _request_evidence_decision() -> dict:
    return {
        "action": "REQUEST_EVIDENCE",
        "selected_lane_id": None,
        "confidence": "HIGH",
        "thesis": "Request more evidence because the packet appears to have no executable lanes.",
        "method": "EVENT_RISK",
        "narrative": "The daily target is open, but the operator believes live-ready coverage is absent.",
        "chart_story": "No clean chart story is accepted yet.",
        "invalidation": "Refresh when current broker truth produces a live-ready lane.",
        "rejected_alternatives": ["TRADE rejected because no current lane appears executable."],
        "risk_notes": ["Stay flat until executable evidence appears."],
        "evidence_refs": ["broker:snapshot", "target:daily"],
        "operator_summary": "Do not trade from this stale evidence request.",
    }


def _wait_decision() -> dict:
    return {
        "action": "WAIT",
        "selected_lane_id": None,
        "confidence": "MEDIUM",
        "thesis": "Wait despite the live-ready lane because discretionary timing is not clean enough.",
        "method": "EVENT_RISK",
        "narrative": "The lane is executable, but event timing argues for patience this cycle.",
        "chart_story": "The trigger shelf exists, but confirmation has not printed yet.",
        "invalidation": "Reconsider if the shelf holds and spread remains inside the receipt.",
        "rejected_alternatives": [f"{LANE_ID} rejected for this cycle because timing confirmation is incomplete."],
        "risk_notes": ["No exposure is open; waiting adds no risk."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            f"intent:{LANE_ID}",
            f"campaign:{LANE_ID}",
            "strategy:EUR_USD:LONG",
            "story:EUR_USD",
        ],
        "operator_summary": "Wait with an explicit rejection of the current executable lane.",
    }


def _cancel_pending_decision(*, cancel_order_ids: list[str] | None = None) -> dict:
    return {
        "action": "CANCEL_PENDING",
        "selected_lane_id": None,
        "cancel_order_ids": cancel_order_ids or [],
        "confidence": "HIGH",
        "thesis": "The pending entry is stale relative to current broker truth and should be cleared before new risk.",
        "method": "POSITION_MANAGEMENT",
        "narrative": "A pending order blocks clean discretionary comparison.",
        "chart_story": "The original trigger has drifted away from the current executable lane.",
        "invalidation": "Do not cancel if the order id is not present in current broker truth.",
        "rejected_alternatives": ["TRADE rejected until pending exposure is resolved."],
        "risk_notes": ["Canceling a pending entry reduces possible future exposure."],
        "evidence_refs": ["broker:snapshot", "target:daily"],
        "operator_summary": "Clear the stale pending order before considering another entry.",
    }


def _pending_order() -> dict:
    return {
        "order_id": "pending-1",
        "pair": "EUR_USD",
        "order_type": "STOP",
        "price": 1.173,
        "state": "PENDING",
        "units": 1000,
        "owner": "trader",
    }


def _position(*, stop_loss: float | None = 1.17) -> dict:
    return {
        "trade_id": "101",
        "pair": "EUR_USD",
        "side": "LONG",
        "units": 1000,
        "entry_price": 1.171,
        "unrealized_pl_jpy": 120.0,
        "take_profit": 1.173,
        "stop_loss": stop_loss,
        "owner": "trader",
    }


if __name__ == "__main__":
    unittest.main()
