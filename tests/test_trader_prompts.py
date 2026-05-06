from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.trader_prompts import (
    BRANCH_ENTRY,
    BRANCH_POSITION,
    BRANCH_REFRESH,
    BRANCH_VERIFY,
    route_trader_prompts,
)


class TraderPromptRouteTest(unittest.TestCase):
    def test_routes_missing_artifacts_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            route = route_trader_prompts(
                snapshot_path=root / "missing_snapshot.json",
                target_state_path=root / "missing_target.json",
                intents_path=root / "missing_intents.json",
                decision_response_path=None,
            )

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertIn("10_precheck_refresh.md", _read_paths(route)[-1])

    def test_routes_flat_open_target_with_live_ready_lanes_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        read_paths = _read_paths(route)
        self.assertTrue(any(path.endswith("30_entry_decision.md") for path in read_paths))
        self.assertTrue(any(path.endswith("90_decision_receipt_schema.md") for path in read_paths))

    def test_routes_unprotected_trader_position_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("needs protection repair", route.reasons[0])

    def test_routes_existing_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "WAIT"}))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertTrue(any(path.endswith("40_verify_execute.md") for path in _read_paths(route)))

    def test_routes_rejected_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [{"code": "UNKNOWN_LANE", "severity": "BLOCK"}],
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already verified as REJECTED", route.reasons[0])

    def test_routes_consumed_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["live_order"].write_text(json.dumps({"status": "SENT", "sent": True}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["live_order"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already consumed by live order gateway receipt", route.reasons[0])

    def test_routes_decision_older_than_refreshed_broker_truth_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 101.0)
            _set_mtime(files["intents"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("predates refreshed broker snapshot", route.reasons[0])

    def test_routes_same_timestamp_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)

    def test_cli_prints_prompt_route_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-prompt-route",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--target-state",
                        str(files["target"]),
                        "--intents",
                        str(files["intents"]),
                        "--pair-charts",
                        str(files["pair_charts"]),
                        "--cross-asset",
                        str(files["cross_asset"]),
                        "--flow",
                        str(files["flow"]),
                        "--currency-strength",
                        str(files["currency_strength"]),
                        "--levels",
                        str(files["levels"]),
                        "--calendar",
                        str(files["calendar"]),
                        "--cot",
                        str(files["cot"]),
                        "--option-skew",
                        str(files["option_skew"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--decision-response",
                        str(root / "missing_decision_response.json"),
                        "--live-order",
                        str(files["live_order"]),
                        "--position-execution",
                        str(files["position_execution"]),
                        "--autotrade-report",
                        str(files["autotrade_report"]),
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["branch"], BRANCH_ENTRY)
        self.assertTrue(any(item["path"].endswith("30_entry_decision.md") for item in payload["read_order"]))


def _route_paths(files: dict[str, Path]) -> dict[str, Path]:
    return {
        "snapshot_path": files["snapshot"],
        "target_state_path": files["target"],
        "intents_path": files["intents"],
        "pair_charts_path": files["pair_charts"],
        "cross_asset_path": files["cross_asset"],
        "flow_path": files["flow"],
        "currency_strength_path": files["currency_strength"],
        "levels_path": files["levels"],
        "calendar_path": files["calendar"],
        "cot_path": files["cot"],
        "option_skew_path": files["option_skew"],
        "attack_advice_path": files["attack_advice"],
        "gpt_decision_path": files["gpt_decision"],
        "live_order_path": files["live_order"],
        "position_execution_path": files["position_execution"],
        "autotrade_report_path": files["autotrade_report"],
    }


def _read_paths(route) -> list[str]:
    return [str(doc.path) for doc in route.read_order]


def _fixtures(root: Path, *, positions: list[dict] | None = None) -> dict[str, Path]:
    files = {
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "pair_charts": root / "pair_charts.json",
        "cross_asset": root / "cross_asset_snapshot.json",
        "flow": root / "flow_snapshot.json",
        "currency_strength": root / "currency_strength.json",
        "levels": root / "levels_snapshot.json",
        "calendar": root / "economic_calendar.json",
        "cot": root / "cot_snapshot.json",
        "option_skew": root / "option_skew_snapshot.json",
        "attack_advice": root / "ai_attack_advice.json",
        "gpt_decision": root / "gpt_trader_decision.json",
        "live_order": root / "live_order_request.json",
        "position_execution": root / "position_execution.json",
        "autotrade_report": root / "autotrade_cycle_report.md",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": positions or [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701, "timestamp_utc": now}},
            }
        )
    )
    files["target"].write_text(
        json.dumps(
            {
                "status": "PURSUE_TARGET",
                "remaining_target_jpy": 1000.0,
                "progress_jpy": 0.0,
            }
        )
    )
    files["intents"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
    )
    for key in (
        "pair_charts",
        "cross_asset",
        "flow",
        "currency_strength",
        "levels",
        "calendar",
        "cot",
        "option_skew",
        "attack_advice",
    ):
        files[key].write_text(json.dumps({}))
    return files


def _set_mtime(path: Path, value: float) -> None:
    os.utime(path, (value, value))


if __name__ == "__main__":
    unittest.main()
