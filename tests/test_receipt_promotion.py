from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.profile import StrategyProfile
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter


class ReceiptPromotionTest(unittest.TestCase):
    def test_promotes_risk_repair_and_pending_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(root)
            intents = _intents(root)
            report = root / "promotion.md"

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=report).run()

            self.assertEqual(summary.promoted, 2)
            payload = json.loads(profile.read_text())
            statuses = {(item["pair"], item["direction"]): item["status"] for item in payload["profiles"]}
            methods = {(item["pair"], item["direction"]): item.get("method") for item in payload["profiles"]}
            self.assertEqual(statuses[("EUR_USD", "LONG")], "CANDIDATE")
            self.assertEqual(statuses[("EUR_JPY", "LONG")], "CANDIDATE")
            self.assertEqual(statuses[("USD_JPY", "SHORT")], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertEqual(methods[("EUR_USD", "LONG")], "TREND_CONTINUATION")
            self.assertEqual(methods[("EUR_JPY", "LONG")], "RANGE_ROTATION")
            loaded = StrategyProfile.load(profile)
            self.assertEqual(loaded.validate(_intent("EUR_USD", "LONG"), for_live_send=True), ())
            self.assertIn("RISK_REPAIR_CANDIDATE -> CANDIDATE", report.read_text())

    def test_does_not_promote_blocking_risk_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(root)
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "snapshot_path": "snapshot.json",
                        "results": [
                            _receipt(
                                "risk:EUR_USD:LONG",
                                "EUR_USD",
                                "LONG",
                                "STOP-ENTRY",
                                risk_issues=[
                                    {
                                        "code": "LOSS_CAP_EXCEEDED",
                                        "message": "too much risk",
                                        "severity": "BLOCK",
                                    }
                                ],
                            )
                        ],
                    }
                )
            )

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 0)
            payload = json.loads(profile.read_text())
            statuses = {(item["pair"], item["direction"]): item["status"] for item in payload["profiles"]}
            self.assertEqual(statuses[("EUR_USD", "LONG")], "RISK_REPAIR_CANDIDATE")

    def test_creates_missing_method_profile_from_pair_side_repair_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "existing failure edge",
                            },
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "cap risk",
                                "target_reward_risk": 2.6,
                                "live_net_jpy": 5738.5,
                                "live_n": 103,
                                "pretrade_net_jpy": 1605.1,
                                "pretrade_n": 34,
                            },
                        ]
                    }
                )
            )
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "snapshot_path": "snapshot.json",
                        "results": [
                            _method_missing_receipt(
                                fallback_status="RISK_REPAIR_CANDIDATE",
                                order_type="LIMIT",
                            )
                        ],
                    }
                )
            )

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 1)
            payload = json.loads(profile.read_text())
            by_key = {
                (item["pair"], item["direction"], item.get("method")): item
                for item in payload["profiles"]
            }
            self.assertEqual(by_key[("EUR_USD", "SHORT", None)]["status"], "RISK_REPAIR_CANDIDATE")
            range_profile = by_key[("EUR_USD", "SHORT", "RANGE_ROTATION")]
            self.assertEqual(range_profile["status"], "CANDIDATE")
            self.assertEqual(range_profile["receipt_promotion"]["from_status"], "RISK_REPAIR_CANDIDATE")
            self.assertEqual(range_profile["target_reward_risk"], 2.6)
            loaded = StrategyProfile.load(profile)
            self.assertEqual(
                loaded.validate(_intent("EUR_USD", "SHORT", method="RANGE_ROTATION"), for_live_send=True),
                (),
            )

    def test_does_not_create_missing_method_profile_from_blocked_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "existing failure edge",
                            },
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "bad live history",
                            },
                        ]
                    }
                )
            )
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "snapshot_path": "snapshot.json",
                        "results": [
                            _method_missing_receipt(
                                fallback_status="BLOCK_UNTIL_NEW_EVIDENCE",
                                order_type="LIMIT",
                            )
                        ],
                    }
                )
            )

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 0)
            payload = json.loads(profile.read_text())
            keys = {(item["pair"], item["direction"], item.get("method")) for item in payload["profiles"]}
            self.assertNotIn(("EUR_USD", "SHORT", "RANGE_ROTATION"), keys)

    def test_does_not_mutate_pair_side_profile_into_existing_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_JPY",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "existing failure edge",
                            },
                            {
                                "pair": "EUR_JPY",
                                "direction": "LONG",
                                "status": "MINE_MISSED_EDGE",
                                "required_fix": "arm pending receipt",
                            },
                        ]
                    }
                )
            )
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "snapshot_path": "snapshot.json",
                        "results": [
                            _breakout_limit_receipt()
                        ],
                    }
                )
            )

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 0)
            payload = json.loads(profile.read_text())
            by_key = {
                (item["pair"], item["direction"], item.get("method")): item
                for item in payload["profiles"]
            }
            self.assertEqual(by_key[("EUR_JPY", "LONG", None)]["status"], "MINE_MISSED_EDGE")
            self.assertEqual(by_key[("EUR_JPY", "LONG", "BREAKOUT_FAILURE")]["status"], "CANDIDATE")
            self.assertEqual(len(by_key), len(payload["profiles"]))

    def test_dedupes_existing_exact_method_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_JPY",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "first",
                            },
                            {
                                "pair": "EUR_JPY",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "receipt",
                                "receipt_promotion": {"lane_id": "preferred"},
                            },
                        ]
                    }
                )
            )
            intents = root / "intents.json"
            intents.write_text(json.dumps({"snapshot_path": "snapshot.json", "results": []}))

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 0)
            payload = json.loads(profile.read_text())
            self.assertEqual(len(payload["profiles"]), 1)
            self.assertEqual(payload["profiles"][0]["required_fix"], "receipt")


def _profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "RISK_REPAIR_CANDIDATE",
                        "required_fix": "cap risk",
                    },
                    {
                        "pair": "EUR_JPY",
                        "direction": "LONG",
                        "status": "MINE_MISSED_EDGE",
                        "required_fix": "arm pending receipt",
                    },
                    {
                        "pair": "USD_JPY",
                        "direction": "SHORT",
                        "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                        "required_fix": "bad live history",
                    },
                ]
            }
        )
    )
    return path


def _intents(root: Path) -> Path:
    path = root / "intents.json"
    path.write_text(
        json.dumps(
            {
                "snapshot_path": "snapshot.json",
                "results": [
                    _receipt("risk:EUR_USD:LONG", "EUR_USD", "LONG", "STOP-ENTRY"),
                    _receipt("trigger:EUR_JPY:LONG", "EUR_JPY", "LONG", "LIMIT"),
                    _receipt("blocked:USD_JPY:SHORT", "USD_JPY", "SHORT", "STOP-ENTRY"),
                ],
            }
        )
    )
    return path


def _receipt(
    lane_id: str,
    pair: str,
    direction: str,
    order_type: str,
    *,
    risk_issues: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "lane_id": lane_id,
        "status": "DRY_RUN_PASSED",
        "risk_allowed": True,
        "risk_issues": risk_issues or [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": pair,
            "side": direction,
            "order_type": order_type,
            "units": 1000,
            "entry": 1.1,
            "tp": 1.2,
            "sl": 1.0,
            "thesis": "test",
            "owner": "trader",
            "market_context": {
                "method": "RANGE_ROTATION" if order_type == "LIMIT" else "TREND_CONTINUATION",
            },
        },
    }


def _method_missing_receipt(*, fallback_status: str, order_type: str) -> dict:
    receipt = _receipt(
        "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
        "EUR_USD",
        "SHORT",
        order_type,
    )
    receipt["intent"]["market_context"]["method"] = "RANGE_ROTATION"
    evidence = {
        "available_methods": ["BREAKOUT_FAILURE"],
        "fallback_pair_side_profile": {
            "profile_pair": "EUR_USD",
            "profile_direction": "SHORT",
            "profile_method": None,
            "profile_match": "pair_side_fallback_not_used",
            "profile_status": fallback_status,
            "requested_method": "RANGE_ROTATION",
            "required_fix": "cap risk",
            "target_reward_risk": 2.6,
            "live_net_jpy": 5738.5,
            "live_n": 103,
            "pretrade_net_jpy": 1605.1,
            "pretrade_n": 34,
        },
        "profile_direction": "SHORT",
        "profile_match": "method_specific_missing",
        "profile_pair": "EUR_USD",
        "requested_method": "RANGE_ROTATION",
    }
    issue = {
        "code": "STRATEGY_METHOD_PROFILE_MISSING",
        "message": "missing method profile",
        "severity": "BLOCK",
        "strategy_profile_evidence": evidence,
    }
    receipt["strategy_issues"] = [{**issue, "severity": "WARN"}]
    receipt["live_strategy_issues"] = [issue]
    receipt["live_blockers"] = ["missing method profile"]
    return receipt


def _breakout_limit_receipt() -> dict:
    receipt = _receipt(
        "failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
        "EUR_JPY",
        "LONG",
        "LIMIT",
    )
    receipt["intent"]["market_context"]["method"] = "BREAKOUT_FAILURE"
    return receipt


def _intent(pair: str, direction: str, *, method: str = "TREND_CONTINUATION"):
    from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod

    return OrderIntent(
        pair=pair,
        side=Side.parse(direction),
        order_type=OrderType.STOP_ENTRY,
        units=1000,
        entry=1.1,
        tp=1.2,
        sl=1.0,
        thesis="test",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime="TREND_CONTINUATION test",
            narrative="test",
            chart_story="test",
            method=TradeMethod.parse(method),
            invalidation="test",
        ),
    )


if __name__ == "__main__":
    unittest.main()
