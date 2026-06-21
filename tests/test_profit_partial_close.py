from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.profit_partial_close import (
    ProfitPartialCloseAction,
    apply_profit_partial_closes,
    compute_profit_partial_close,
    save_profit_partial_state_from_results,
)


class ComputeProfitPartialCloseTest(unittest.TestCase):
    def _ctx(self) -> dict:
        return {
            "h1_adx": 30.0,
            "session_current_tag": "LONDON_NY_OVERLAP",
            "confluence": {"higher_tf_alignment": "LONG_LEAN"},
        }

    def test_no_action_if_not_profitable(self) -> None:
        action = compute_profit_partial_close(
            trade_id="t1",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.2980,
            atr_pips=20,
            chart_context=self._ctx(),
        )

        self.assertIsNone(action)

    def test_profit_milestone_closes_part_and_keeps_runner(self) -> None:
        action = compute_profit_partial_close(
            trade_id="t2",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.3030,
            atr_pips=20,
            chart_context=self._ctx(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.milestone, 3)
        self.assertEqual(action.close_units, 2500)
        self.assertEqual(action.remaining_units, 7500)
        self.assertIn("runner 7500", action.rationale)

    def test_manual_profit_milestone_closes_part_and_keeps_runner(self) -> None:
        action = compute_profit_partial_close(
            trade_id="manual-t2",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.3030,
            atr_pips=20,
            owner="manual",
            chart_context=self._ctx(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.close_units, 2500)
        self.assertEqual(action.remaining_units, 7500)

    def test_external_position_is_not_profit_partially_closed(self) -> None:
        action = compute_profit_partial_close(
            trade_id="external-t2",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.3030,
            atr_pips=20,
            owner="external",
            chart_context=self._ctx(),
        )

        self.assertIsNone(action)

    def test_same_milestone_does_not_repeat(self) -> None:
        action = compute_profit_partial_close(
            trade_id="t3",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.3030,
            atr_pips=20,
            chart_context=self._ctx(),
            last_milestone=3,
        )

        self.assertIsNone(action)


class ApplyProfitPartialCloseTest(unittest.TestCase):
    def _action(self) -> ProfitPartialCloseAction:
        return ProfitPartialCloseAction(
            trade_id="t4",
            pair="EUR_USD",
            side="LONG",
            original_units=10000,
            close_units=2500,
            remaining_units=7500,
            profit_pips=30.0,
            atr_pips=20.0,
            trigger_mult=0.5,
            fraction=0.25,
            milestone=3,
            prior_milestone=0,
            rationale="test",
        )

    def test_send_requires_live_enabled_and_confirmation(self) -> None:
        results = apply_profit_partial_closes(
            [self._action()],
            broker_client=None,
            send=True,
            live_enabled=False,
            confirm_live=False,
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("LIVE_DISABLED", results[0]["error"])

    def test_live_send_calls_broker_with_partial_units(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = []

            def close_trade(self, trade_id, units):
                self.calls.append((trade_id, units))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [self._action()],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertTrue(results[0]["sent"])
        self.assertEqual(client.calls, [("t4", "2500")])
        self.assertEqual(results[0]["provenance"], "profit_partial_close")

    def test_live_send_uses_provenance_method_when_supported(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = []

            def close_trade(self, trade_id, units):
                raise AssertionError("profit partial close must use provenance-aware close when available")

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [self._action()],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertTrue(results[0]["sent"])
        self.assertEqual(client.calls, [("t4", "2500", "profit_partial_close")])

    def test_state_updates_only_after_successful_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = save_profit_partial_state_from_results(
                [{"trade_id": "t4", "milestone": 3, "sent": True}],
                path=path,
                state={"trade_milestones": {}},
            )

            self.assertEqual(state["trade_milestones"]["t4"], 3)
            self.assertIn('"t4": 3', path.read_text())


if __name__ == "__main__":
    unittest.main()
