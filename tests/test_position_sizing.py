from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.risk import RiskEngine, RiskPolicy


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass


position_sizing = _load_tool("position_sizing")
place_trader_order = _load_tool("place_trader_order")


class PositionSizingToolTest(unittest.TestCase):
    def test_a_grade_sizes_from_target_and_risk_cap(self) -> None:
        result = position_sizing.size_position(
            position_sizing.PositionSizingInput(
                pair="USD_JPY",
                side="LONG",
                entry=150.0,
                tp=151.0,
                sl=149.0,
                conviction_grade="A",
                day_start_nav=100_000.0,
                current_nav=101_000.0,
                remaining_to_5pct=4_000.0,
                mode="ATTACK",
                remaining_risk_budget_yen=1_000.0,
                target_path_role="HERO",
                path_board_available=True,
                attack_stack_available=True,
                maps_to_attack_stack=True,
            )
        )

        self.assertEqual(result.suggested_units, 1000)
        self.assertEqual(result.risk_yen, 1000.0)
        self.assertEqual(result.target_yen, 1000.0)
        self.assertEqual(result.valid_as_target_path, "YES")

    def test_b0_main_path_is_blocked_under_five_pct(self) -> None:
        result = position_sizing.size_position(
            position_sizing.PositionSizingInput(
                pair="USD_JPY",
                side="LONG",
                entry=150.0,
                tp=151.0,
                sl=149.0,
                conviction_grade="B0",
                day_start_nav=100_000.0,
                current_nav=101_000.0,
                remaining_to_5pct=4_000.0,
                mode="ATTACK",
                remaining_risk_budget_yen=1_000.0,
                target_path_role="HERO",
            )
        )

        self.assertEqual(result.suggested_units, 0)
        self.assertEqual(result.valid_as_target_path, "NO")
        self.assertIn("TARGET_PATH_GRADE_TOO_LOW", {issue["code"] for issue in result.issues})

    def test_extension_gate_requires_all_conditions(self) -> None:
        yes = position_sizing.evaluate_extension_gate(
            position_sizing.ExtensionGateInput(
                progress_pct=4.0,
                hero_thesis_paying=True,
                theme_confirmations=3,
                spread_stable=True,
                last_a_s_trade_state="protected",
                reload_level_exists=True,
            )
        )
        no = position_sizing.evaluate_extension_gate(
            position_sizing.ExtensionGateInput(
                progress_pct=4.0,
                hero_thesis_paying=True,
                theme_confirmations=3,
                spread_stable=True,
                last_a_s_trade_state="protected",
                reload_level_exists=False,
            )
        )

        self.assertEqual(yes.gate, "YES")
        self.assertEqual(no.gate, "NO")
        self.assertIn("no real reload/second-shot level; would be chase", no.blockers)

    def test_place_trader_order_is_dry_run_only(self) -> None:
        parser = place_trader_order._parser()
        args = parser.parse_args(
            [
                "--pair",
                "USD_JPY",
                "--side",
                "LONG",
                "--entry",
                "150",
                "--tp",
                "151",
                "--sl",
                "149",
                "--grade",
                "A",
                "--day-start-nav",
                "100000",
                "--current-nav",
                "101000",
                "--remaining-to-5pct",
                "4000",
                "--mode",
                "ATTACK",
                "--remaining-risk-budget-yen",
                "1000",
                "--target-path-role",
                "HERO",
                "--send",
            ]
        )

        result = place_trader_order.evaluate_order(args)

        self.assertEqual(result.status, "DRY_RUN_BLOCKED")
        self.assertFalse(result.live_order_sent)
        self.assertIn("LIVE_SEND_DISABLED", {issue["code"] for issue in result.issues})


class TargetPathRiskEngineTest(unittest.TestCase):
    def _snapshot(self) -> BrokerSnapshot:
        now = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)
        return BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "EUR_USD": Quote("EUR_USD", 1.1000, 1.1001, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", 150.0, 150.01, timestamp_utc=now),
            },
        )

    def _intent(self, metadata: dict) -> OrderIntent:
        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.1001,
            tp=1.1021,
            sl=1.0991,
            thesis="target path test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="test",
                chart_story="test",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="test",
            ),
            metadata={"max_loss_jpy": 1000.0, **metadata},
        )

    def test_risk_engine_blocks_extend_b_grade_metadata(self) -> None:
        decision = RiskEngine(
            policy=RiskPolicy(require_live_enabled_for_send=False),
            live_enabled=True,
            validation_time_utc=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
        ).validate(
            self._intent(
                {
                    "daily_target_mode": "EXTEND",
                    "conviction_grade": "B+",
                    "remaining_minimum_jpy": 0.0,
                    "extension_gate": "YES",
                    "target_path_role": "SUPPORT",
                }
            ),
            self._snapshot(),
            for_live_send=True,
        )

        self.assertIn("EXTEND_REQUIRES_A_GRADE", {issue.code for issue in decision.issues})

    def test_risk_engine_blocks_missing_path_attack_mapping_when_available(self) -> None:
        decision = RiskEngine(
            policy=RiskPolicy(require_live_enabled_for_send=False),
            live_enabled=True,
            validation_time_utc=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
        ).validate(
            self._intent(
                {
                    "daily_target_mode": "ATTACK",
                    "conviction_grade": "A",
                    "remaining_minimum_jpy": 3000.0,
                    "path_board_available": True,
                    "attack_stack_available": True,
                    "maps_to_attack_stack": False,
                    "target_path_role": "HERO",
                }
            ),
            self._snapshot(),
            for_live_send=True,
        )

        self.assertIn("PATH_ATTACK_STACK_MAPPING_MISSING", {issue.code for issue in decision.issues})


if __name__ == "__main__":
    unittest.main()
