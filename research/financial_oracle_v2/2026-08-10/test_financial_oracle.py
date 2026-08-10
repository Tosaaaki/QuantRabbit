#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("financial_oracle_builder", HERE / "build_financial_oracle.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sqlite_row(payload: dict, *, event_type: str, trade_id: str, realized: str, financing: str = "0") -> sqlite3.Row:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE x(event_uid,ts_utc,event_type,trade_id,units,realized_pl_jpy,financing_jpy,oanda_transaction_id,raw_json)"
    )
    connection.execute(
        "INSERT INTO x VALUES(?,?,?,?,?,?,?,?,?)",
        ("e", payload["time"], event_type, trade_id, 1000, realized, financing, payload["id"], json.dumps(payload)),
    )
    return connection.execute("SELECT * FROM x").fetchone()


class FinancialOracleTest(unittest.TestCase):
    def test_multi_leg_close_selects_exact_trade_without_double_count(self) -> None:
        payload = {
            "id": "10",
            "time": "2026-01-01T00:00:00Z",
            "type": "ORDER_FILL",
            "pl": "-30",
            "financing": "-3",
            "commission": "0",
            "guaranteedExecutionFee": "0",
            "tradesClosed": [
                {"tradeID": "A", "units": "-1000", "realizedPL": "-10", "financing": "-1", "guaranteedExecutionFee": "0"},
                {"tradeID": "B", "units": "-1000", "realizedPL": "-20", "financing": "-2", "guaranteedExecutionFee": "0"},
            ],
        }
        row = sqlite_row(payload, event_type="TRADE_CLOSED", trade_id="B", realized="-20", financing="-2")
        components, issues = MODULE.close_component(row, {"A", "B"})
        self.assertEqual(issues, [])
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["trade_id"], "B")
        self.assertEqual(components[0]["amount_jpy"], -22.0)

    def test_nonzero_multi_leg_commission_fails_closed(self) -> None:
        payload = {
            "id": "11",
            "time": "2026-01-01T00:00:00Z",
            "type": "ORDER_FILL",
            "pl": "3",
            "financing": "0",
            "commission": "-1",
            "guaranteedExecutionFee": "0",
            "tradesClosed": [
                {"tradeID": "A", "units": "-1", "realizedPL": "1", "financing": "0", "guaranteedExecutionFee": "0"},
                {"tradeID": "B", "units": "-1", "realizedPL": "2", "financing": "0", "guaranteedExecutionFee": "0"},
            ],
        }
        row = sqlite_row(payload, event_type="TRADE_CLOSED", trade_id="A", realized="1")
        _, issues = MODULE.close_component(row, {"A", "B"})
        self.assertIn("MULTI_LEG_COMMISSION_UNALLOCATED:11", issues)

    def test_daily_financing_allocates_explicit_trade_ids_only(self) -> None:
        payload = {
            "id": "12",
            "time": "2026-01-01T21:00:00Z",
            "type": "DAILY_FINANCING",
            "financing": "-3",
            "positionFinancings": [{
                "financing": "-3",
                "openTradeFinancings": [
                    {"tradeID": "A", "financing": "-1"},
                    {"tradeID": "OUT", "financing": "-2"},
                ],
            }],
        }
        row = sqlite_row(payload, event_type="OANDA_TRANSACTION", trade_id="", realized="0")
        components, audit, issues = MODULE.daily_components(row, {"A"})
        self.assertEqual(issues, [])
        self.assertEqual([item["trade_id"] for item in components], ["A"])
        self.assertEqual(audit["cohort_financing_jpy"], -1.0)
        self.assertEqual(audit["out_of_cohort_financing_jpy"], -2.0)

    def test_daily_top_residual_is_detected(self) -> None:
        payload = {
            "id": "13",
            "time": "2026-01-01T21:00:00Z",
            "type": "DAILY_FINANCING",
            "financing": "-4",
            "positionFinancings": [{
                "financing": "-3",
                "openTradeFinancings": [{"tradeID": "A", "financing": "-3"}],
            }],
        }
        row = sqlite_row(payload, event_type="OANDA_TRANSACTION", trade_id="", realized="0")
        _, _, issues = MODULE.daily_components(row, {"A"})
        self.assertIn("DAILY_FINANCING_TOP_RESIDUAL:13", issues)

    def test_generated_component_conservation_and_uniqueness(self) -> None:
        components = MODULE.read_jsonl(HERE / "cashflow_components_v2.jsonl")
        trades = MODULE.read_jsonl(HERE / "trade_cashflows_v2.jsonl")
        ids = [row["component_id"] for row in components]
        self.assertEqual(len(ids), len(set(ids)))
        by_trade: dict[str, Decimal] = {}
        for component in components:
            trade_id = str(component["trade_id"])
            by_trade[trade_id] = by_trade.get(trade_id, Decimal("0")) + Decimal(str(component["amount_jpy"]))
        self.assertEqual(len(trades), 251)
        for trade in trades:
            self.assertEqual(trade["allocation_status"], "PASS")
            self.assertLessEqual(
                abs(by_trade[str(trade["trade_id"])] - Decimal(str(trade["corrected_net_jpy"]))),
                Decimal("0.00011"),
            )

    def test_paired_identity(self) -> None:
        for row in MODULE.read_jsonl(HERE / "trade_cashflows_v2.jsonl"):
            left = Decimal(str(row["original_v1_net_jpy"])) + Decimal(str(row["paired_delta_vs_v1_jpy"]))
            right = Decimal(str(row["corrected_net_jpy"]))
            self.assertLessEqual(abs(left - right), Decimal("0.00011"))

    def test_source_hash_guard(self) -> None:
        prereg = json.loads((HERE / "preregister_v2.json").read_text(encoding="utf-8"))
        for binding in prereg["source_bindings"].values():
            self.assertEqual(MODULE.sha256_path(MODULE.REPO / binding["path"]), binding["sha256"])

    def test_drawdown_fixture(self) -> None:
        values = [Decimal("10"), Decimal("-3"), Decimal("-12"), Decimal("4")]
        self.assertEqual(MODULE.max_drawdown(values), Decimal("15"))


if __name__ == "__main__":
    unittest.main()
