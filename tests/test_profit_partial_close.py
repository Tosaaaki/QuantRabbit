from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from quant_rabbit.models import BrokerPosition, Owner, Side
from quant_rabbit.strategy.profit_partial_close import (
    ProfitPartialCloseAction,
    apply_profit_partial_closes,
    compute_all_profit_partial_closes,
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
            owner="trader",
            chart_context=self._ctx(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.milestone, 3)
        self.assertEqual(action.close_units, 2500)
        self.assertEqual(action.remaining_units, 7500)
        self.assertIn("runner 7500", action.rationale)

    def test_manual_profit_milestone_is_monitor_only(self) -> None:
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

        self.assertIsNone(action)

    def test_operator_manual_position_is_excluded_before_profit_partial_computation(
        self,
    ) -> None:
        position = BrokerPosition(
            trade_id="operator-manual-profit",
            pair="EUR_USD",
            side=Side.LONG,
            units=10000,
            entry_price=1.3000,
            owner=Owner.OPERATOR_MANUAL,
            raw={
                "operator_manual_position": {
                    "operator_confirmed": True,
                    "management_intent": "KEEP",
                    "auto_tp_modify_allowed": False,
                }
            },
        )

        actions = compute_all_profit_partial_closes(
            positions=[position],
            quotes={"EUR_USD": {"bid": 1.3030, "ask": 1.3031}},
            pair_charts={
                "EUR_USD": {
                    "confluence": {"h4_atr_pips": 20.0},
                    "views": [],
                }
            },
        )

        self.assertEqual(actions, [])

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
            owner="trader",
            chart_context=self._ctx(),
            last_milestone=3,
        )

        self.assertIsNone(action)

    def test_missing_owner_defaults_fail_closed(self) -> None:
        action = compute_profit_partial_close(
            trade_id="owner-missing",
            pair="EUR_USD",
            side="LONG",
            units=10000,
            entry_price=1.3000,
            current_price=1.3030,
            atr_pips=20,
            chart_context=self._ctx(),
        )

        self.assertIsNone(action)

    def test_predictive_scout_is_excluded_before_profit_partial_computation(self) -> None:
        position = BrokerPosition(
            trade_id="scout-profit",
            pair="EUR_USD",
            side=Side.LONG,
            units=10000,
            entry_price=1.3000,
            owner=Owner.TRADER,
            raw={
                "tradeClientExtensions": {
                    "comment": "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT vehicle=psv-test"
                }
            },
        )

        actions = compute_all_profit_partial_closes(
            positions=[position],
            quotes={"EUR_USD": {"bid": 1.3030, "ask": 1.3031}},
            pair_charts={
                "EUR_USD": {
                    "confluence": {"h4_atr_pips": 20.0},
                    "views": [],
                }
            },
        )

        self.assertEqual(actions, [])


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
            owner="trader",
        )

    def _position(
        self,
        *,
        owner: Owner = Owner.TRADER,
        trade_id: str = "t4",
        side: Side = Side.LONG,
        units: int = 10000,
        entry_price: float = 1.3000,
        unrealized_pl_jpy: float = 100.0,
        raw: dict | None = None,
    ) -> BrokerPosition:
        return BrokerPosition(
            trade_id=trade_id,
            pair="EUR_USD",
            side=side,
            units=units,
            entry_price=entry_price,
            unrealized_pl_jpy=unrealized_pl_jpy,
            owner=owner,
            raw=raw or {},
        )

    @staticmethod
    def _short_action() -> ProfitPartialCloseAction:
        return ProfitPartialCloseAction(
            trade_id="short-t4",
            pair="EUR_USD",
            side="SHORT",
            original_units=22500,
            close_units=7500,
            remaining_units=15000,
            profit_pips=30.0,
            atr_pips=20.0,
            trigger_mult=0.5,
            fraction=1 / 3,
            milestone=3,
            prior_milestone=0,
            rationale="stale short action",
            owner="trader",
        )

    @staticmethod
    def _snapshot(position: object, *, bid: float = 1.3030, ask: float = 1.3031):
        return SimpleNamespace(
            positions=(position,),
            quotes={"EUR_USD": SimpleNamespace(bid=bid, ask=ask)},
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
        self.assertFalse(results[0]["broker_post_attempted"])
        self.assertIn("LIVE_DISABLED", results[0]["error"])

    def test_direct_send_rechecks_stale_local_trader_against_fresh_operator_owner(self) -> None:
        fresh_position = self._position(owner=Owner.OPERATOR_MANUAL)
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

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

        self.assertFalse(results[0]["sent"])
        self.assertIn("NON_TRADER_PROFIT_PARTIAL_CLOSE_FORBIDDEN", results[0]["error"])
        self.assertEqual(results[0]["owner"], "operator_manual")
        self.assertEqual(results[0]["action_owner"], "trader")
        self.assertEqual(client.calls, [])

    def test_direct_send_rejects_missing_fresh_owner_as_unknown(self) -> None:
        fresh_position = SimpleNamespace(
            trade_id="t4",
            pair="EUR_USD",
            side=Side.LONG,
            units=10000,
            raw={},
        )
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

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

        self.assertFalse(results[0]["sent"])
        self.assertEqual(results[0]["owner"], "unknown")
        self.assertIn("NON_TRADER_PROFIT_PARTIAL_CLOSE_FORBIDDEN", results[0]["error"])
        self.assertEqual(client.calls, [])

    def test_missing_action_owner_cannot_use_fresh_trader_as_implicit_default(self) -> None:
        action = ProfitPartialCloseAction(
            **{key: value for key, value in self._action().__dict__.items() if key != "owner"}
        )
        fresh_position = self._position()
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [action],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertEqual(action.owner, "unknown")
        self.assertFalse(results[0]["sent"])
        self.assertEqual(results[0]["owner"], "trader")
        self.assertEqual(results[0]["action_owner"], "unknown")
        self.assertIn("action owner is not explicitly trader-owned", results[0]["error"])
        self.assertEqual(client.calls, [])

    def test_fresh_losing_quote_blocks_stale_profit_partial_action(self) -> None:
        fresh_position = self._position(
            trade_id="short-t4",
            side=Side.SHORT,
            units=22500,
            entry_price=1.14048,
            unrealized_pl_jpy=-100.0,
        )
        fresh_snapshot = self._snapshot(fresh_position, bid=1.1409, ask=1.1410)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [self._short_action()],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("FRESH_PROFIT_PARTIAL_CLOSE_NOT_PROFITABLE", results[0]["error"])
        self.assertEqual(results[0]["fresh_broker_evidence"]["position_units"], 22500.0)
        self.assertLess(
            results[0]["fresh_broker_evidence"]["executable_profit_pips"],
            0,
        )
        self.assertEqual(client.calls, [])

    def test_fresh_unit_reduction_cannot_turn_partial_action_into_full_close(self) -> None:
        fresh_position = self._position(
            trade_id="short-t4",
            side=Side.SHORT,
            units=7500,
            entry_price=1.14048,
            unrealized_pl_jpy=100.0,
        )
        fresh_snapshot = self._snapshot(fresh_position, bid=1.1369, ask=1.1370)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [self._short_action()],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("FRESH_BROKER_POSITION_UNITS_CHANGED", results[0]["error"])
        self.assertEqual(results[0]["fresh_broker_evidence"]["position_units"], 7500.0)
        self.assertEqual(client.calls, [])

    def test_partial_close_action_can_never_request_all_fresh_units(self) -> None:
        action = ProfitPartialCloseAction(
            **{
                **self._short_action().__dict__,
                "original_units": 7500,
                "close_units": 7500,
                "remaining_units": 0,
            }
        )
        fresh_position = self._position(
            trade_id="short-t4",
            side=Side.SHORT,
            units=7500,
            entry_price=1.14048,
            unrealized_pl_jpy=100.0,
        )
        fresh_snapshot = self._snapshot(fresh_position, bid=1.1369, ask=1.1370)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [action],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("PROFIT_PARTIAL_CLOSE_MUST_LEAVE_RUNNER", results[0]["error"])
        self.assertEqual(client.calls, [])

    def test_fresh_profit_must_still_reach_action_milestone(self) -> None:
        fresh_position = self._position(
            trade_id="short-t4",
            side=Side.SHORT,
            units=22500,
            entry_price=1.14048,
            unrealized_pl_jpy=100.0,
        )
        fresh_snapshot = self._snapshot(fresh_position, bid=1.1388, ask=1.13898)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = Client()
        results = apply_profit_partial_closes(
            [self._short_action()],
            broker_client=client,
            send=True,
            live_enabled=True,
            confirm_live=True,
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("FRESH_PROFIT_MILESTONE_NO_LONGER_REACHED", results[0]["error"])
        self.assertEqual(results[0]["fresh_broker_evidence"]["fresh_milestone"], 1)
        self.assertEqual(client.calls, [])

    def test_send_boundary_blocks_fresh_broker_scout_trade(self) -> None:
        fresh_position = self._position()
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

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
            forbidden_trade_reasons={
                "t4": "PREDICTIVE_SCOUT_EXIT_GEOMETRY_FROZEN: fresh broker truth"
            },
        )

        self.assertFalse(results[0]["sent"])
        self.assertIn("PREDICTIVE_SCOUT_EXIT_GEOMETRY_FROZEN", results[0]["error"])
        self.assertEqual(client.calls, [])

    def test_live_send_blocks_raw_close_fallback(self) -> None:
        fresh_position = self._position()
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            @staticmethod
            def snapshot(pairs):
                return fresh_snapshot

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

        self.assertFalse(results[0]["sent"])
        self.assertTrue(results[0]["broker_post_attempted"])
        self.assertEqual(client.calls, [])
        self.assertEqual(results[0]["provenance"], "profit_partial_close")
        self.assertIn("close_trade_with_provenance", results[0]["error"])

    def test_live_send_uses_provenance_method_when_supported(self) -> None:
        fresh_position = self._position()
        fresh_snapshot = self._snapshot(fresh_position)

        class Client:
            def __init__(self) -> None:
                self.calls = []
                self.snapshot_calls = []

            def snapshot(self, pairs):
                self.snapshot_calls.append(pairs)
                return fresh_snapshot

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
        self.assertTrue(results[0]["broker_post_attempted"])
        self.assertEqual(client.snapshot_calls, [("EUR_USD",)])
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
