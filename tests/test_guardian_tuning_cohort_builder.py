from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.capture_economics import RealizedOutcome
from quant_rabbit import guardian_tuning_cohort as cohort
from quant_rabbit.strategy.entry_thesis_ledger import (
    record_entry_thesis_from_response_result,
)


LANE = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
REVIEWED_AT = datetime(2026, 7, 9, tzinfo=timezone.utc)


def _review() -> dict:
    return {
        "review_status": "TEST_REQUIRED",
        "affected_pairs": ["EUR_USD"],
        "affected_bot_families": ["forecast"],
        "hypothesis": "a stricter recorded forecast floor improves forward capture",
        "falsifiable_experiment": "evaluate only entries opened after this review",
        "proposed_adjustments": [
            {
                "pair": "EUR_USD",
                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                "bot_family": "forecast",
                "parameter": "forecast_confidence_floor",
                "current_value": 0.65,
                "candidate_value": 0.70,
                "rationale": "one precommitted confidence-floor tightening",
            }
        ],
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
    }


class GuardianTuningCohortBuilderTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        missing_thesis_trade: str | None = None,
        pre_review_trade: str | None = None,
        at_review_trade: str | None = None,
        unresolved_first_with_later_resolved: bool = False,
        null_lane_first_with_gateway: bool = False,
        late_append_first_entry: bool = False,
        record_theses_from_gateway_response: bool = False,
    ) -> tuple[Path, Path, Path, list[RealizedOutcome], dict]:
        ledger = root / "data" / "execution_ledger.db"
        ledger.parent.mkdir(parents=True)
        thesis_path = root / "data" / "entry_thesis_ledger.jsonl"
        forecast_path = root / "data" / "forecast_history.jsonl"
        theses: list[str] = []
        forecasts: list[str] = []
        outcomes: list[RealizedOutcome] = []
        with sqlite3.connect(ledger) as conn:
            conn.execute("CREATE TABLE sync_state(key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany(
                "INSERT INTO sync_state VALUES (?, ?)",
                (
                    ("oanda_transaction_coverage_start_utc", "2026-05-06T16:52:01Z"),
                    ("last_oanda_transaction_id", "473024"),
                ),
            )
            conn.execute(
                """
                CREATE TABLE execution_events(
                    event_type TEXT, trade_id TEXT, order_id TEXT,
                    lane_id TEXT, pair TEXT, side TEXT, units REAL, ts_utc TEXT,
                    raw_json TEXT, exit_reason TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "ACTIVATION_ANCHOR",
                    "",
                    "",
                    "",
                    "",
                    "",
                    0,
                    REVIEWED_AT.isoformat(),
                    "",
                    "",
                ),
            )
            conn.commit()
            activation_anchor = cohort.current_execution_ledger_anchor(
                ledger_path=ledger,
            )
            entry_count = (
                21
                if unresolved_first_with_later_resolved or late_append_first_entry
                else 20
            )
            entry_indexes = list(range(entry_count))
            if late_append_first_entry:
                entry_indexes = [*range(1, entry_count), 0]
            for index in entry_indexes:
                trade_id = f"trade-{index}"
                entry_at = REVIEWED_AT + timedelta(hours=index + 1)
                if trade_id == pre_review_trade:
                    entry_at = REVIEWED_AT - timedelta(seconds=1)
                if trade_id == at_review_trade:
                    entry_at = REVIEWED_AT
                signal_at = entry_at - timedelta(minutes=5)
                order_id = f"order-{index}"
                units = 1000 * (1 + index % 2)
                if null_lane_first_with_gateway and index == 0:
                    conn.execute(
                        "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            "GATEWAY_ORDER_SENT",
                            "",
                            order_id,
                            LANE,
                            "EUR_USD",
                            "LONG",
                            units,
                            (entry_at - timedelta(seconds=1)).isoformat(),
                            json.dumps({"reason": "LIMIT_ORDER"}),
                            "LIMIT_ORDER",
                        ),
                    )
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "ORDER_FILLED",
                        trade_id,
                        order_id,
                        None if null_lane_first_with_gateway and index == 0 else LANE,
                        "EUR_USD",
                        "LONG",
                        units,
                        entry_at.isoformat(),
                        json.dumps(
                            {
                                "id": f"fill-transaction-{index}",
                                "time": entry_at.isoformat(),
                                "type": "ORDER_FILL",
                                "orderID": order_id,
                                "instrument": "EUR_USD",
                                "units": str(units),
                                "reason": "LIMIT_ORDER",
                                "tradeOpened": {
                                    "tradeID": trade_id,
                                    "units": str(units),
                                }
                            }
                        ),
                        "LIMIT_ORDER",
                    ),
                )
                cycle_id = f"cycle-{index}"
                confidence = 0.65 if index < 4 else 0.75
                forecasts.append(
                    json.dumps(
                        {
                            "timestamp_utc": signal_at.isoformat(),
                            "cycle_id": cycle_id,
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": confidence,
                        },
                        sort_keys=True,
                    )
                )
                if record_theses_from_gateway_response:
                    with forecast_path.open("a", encoding="utf-8") as handle:
                        handle.write(forecasts[-1] + "\n")

                    class GatewayIntent:
                        pair = "EUR_USD"
                        side = "LONG"
                        order_type = "LIMIT"
                        thesis = "EUR_USD LONG canonical forward tuning sample"
                        entry = 1.1000
                        metadata = {
                            "desk": "trend_trader",
                            "parent_lane_id": (
                                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
                            ),
                            "regime_state": "TREND_UP",
                            "m5_trend_score": 0.8,
                        }

                    response = {
                        "orderFillTransaction": {
                            "id": f"fill-transaction-{index}",
                            "time": entry_at.isoformat(),
                            "type": "ORDER_FILL",
                            "orderID": order_id,
                            "instrument": "EUR_USD",
                            "units": str(units),
                            "reason": "LIMIT_ORDER",
                            "price": "1.1000",
                            "tradeOpened": {
                                "tradeID": trade_id,
                                "units": str(units),
                            },
                        }
                    }
                    result = record_entry_thesis_from_response_result(
                        response=response,
                        intent=GatewayIntent(),
                        data_root=ledger.parent,
                    )
                    self.assertEqual(result.status, "RECORDED")
                elif trade_id != missing_thesis_trade:
                    theses.append(
                        json.dumps(
                            {
                                "timestamp_utc": entry_at.isoformat(),
                                "trade_id": trade_id,
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "forecast_confidence": confidence,
                                "context_evidence": {
                                    "order_id": order_id,
                                    "lane_id": LANE,
                                    "forecast_timestamp_utc": signal_at.isoformat(),
                                    "forecast_cycle_id": cycle_id,
                                },
                            },
                            sort_keys=True,
                        )
                    )
                if not (unresolved_first_with_later_resolved and index == 0):
                    outcomes.append(
                        RealizedOutcome(
                            ts_utc=(entry_at + timedelta(minutes=30)).isoformat(),
                            trade_id=trade_id,
                            pair="EUR_USD",
                            side="LONG",
                            lane_id=LANE,
                            method="TREND_CONTINUATION",
                            exit_reason="TAKE_PROFIT_ORDER",
                            realized_pl_jpy=-100.0 if index < 4 else 100.0,
                            entry_vehicle="LIMIT",
                            entry_truth_consistent=True,
                            broker_close_ts_utc=(
                                entry_at + timedelta(minutes=30)
                            ).isoformat(),
                            broker_time_consistent=True,
                        )
                    )
        if not record_theses_from_gateway_response:
            thesis_path.write_text("\n".join(theses) + "\n")
            forecast_path.write_text("\n".join(forecasts) + "\n")
        return ledger, thesis_path, forecast_path, outcomes, activation_anchor

    def test_builds_complete_post_review_post_cost_normalized_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

            self.assertEqual(payload["schema_version"], 5)
            self.assertEqual(len(payload["samples"]), 20)
            self.assertEqual(payload["validation_contract"]["mode"], "FORWARD_POST_REVIEW")
            self.assertEqual(payload["samples"][0]["net_jpy_per_1000_units"], -100.0)
            self.assertEqual(payload["samples"][1]["net_jpy_per_1000_units"], -50.0)
            self.assertNotIn("signal_evidence_ref", payload["source_watermark"])

    def test_gateway_response_writer_builds_complete_forward_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                record_theses_from_gateway_response=True,
            )
            written_theses = [
                json.loads(line)
                for line in theses.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(written_theses), 20)
            first_context = written_theses[0]["context_evidence"]
            self.assertEqual(written_theses[0]["timestamp_utc"], "2026-07-09T01:00:00+00:00")
            self.assertEqual(first_context["order_id"], "order-0")
            self.assertEqual(first_context["lane_id"], LANE)
            self.assertEqual(
                first_context["forecast_timestamp_utc"],
                "2026-07-09T00:55:00+00:00",
            )
            self.assertEqual(first_context["forecast_cycle_id"], "cycle-0")
            self.assertEqual(
                first_context["guardian_tuning_signal_state"]["regime_state"],
                "TREND_UP",
            )

            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

            self.assertEqual(len(payload["samples"]), 20)
            self.assertEqual(
                [sample["trade_id"] for sample in payload["samples"]],
                [f"trade-{index}" for index in range(20)],
            )
            self.assertEqual(
                [sample["order_id"] for sample in payload["samples"]],
                [f"order-{index}" for index in range(20)],
            )

    def test_post_activation_monitor_uses_exact_first_twenty_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(payload["status"], "POST_ACTIVATION_COHORT_COMPLETE")
            self.assertEqual(payload["first_trade_ids"], [f"trade-{i}" for i in range(20)])
            self.assertEqual(payload["sample_count"], 20)
            self.assertEqual(payload["primary_metric_value"], 45.0)
            self.assertEqual(validation["status"], "VALID")

    def test_current_truth_allows_unrelated_ledger_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "UNRELATED_APPEND",
                        "",
                        "",
                        "",
                        "USD_JPY",
                        "SHORT",
                        0,
                        (REVIEWED_AT + timedelta(days=2)).isoformat(),
                        "",
                        "",
                    ),
                )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(validation["status"], "VALID")

    def test_current_truth_rejects_late_earlier_unresolved_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )
            late_trade_id = "late-unresolved-loss"
            late_order_id = "late-unresolved-order"
            late_entry_at = REVIEWED_AT + timedelta(minutes=30)
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "ORDER_FILLED",
                        late_trade_id,
                        late_order_id,
                        LANE,
                        "EUR_USD",
                        "LONG",
                        1000,
                        late_entry_at.isoformat(),
                        json.dumps(
                            {
                                "id": "late-fill-transaction",
                                "time": late_entry_at.isoformat(),
                                "type": "ORDER_FILL",
                                "orderID": late_order_id,
                                "instrument": "EUR_USD",
                                "units": "1000",
                                "reason": "LIMIT_ORDER",
                                "tradeOpened": {
                                    "tradeID": late_trade_id,
                                    "units": "1000",
                                },
                            }
                        ),
                        "LIMIT_ORDER",
                    ),
                )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(
                validation["status"],
                "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            )
            self.assertEqual(
                validation["current_status"],
                "WAITING_FOR_FIRST_20_RESOLUTIONS",
            )
            self.assertEqual(
                validation["current_first_trade_ids"][0],
                late_trade_id,
            )

    def test_current_truth_rejects_late_earlier_resolved_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )
            late_trade_id = "late-resolved-loss"
            late_order_id = "late-resolved-order"
            late_entry_at = REVIEWED_AT + timedelta(minutes=30)
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "ORDER_FILLED",
                        late_trade_id,
                        late_order_id,
                        LANE,
                        "EUR_USD",
                        "LONG",
                        1000,
                        late_entry_at.isoformat(),
                        json.dumps(
                            {
                                "id": "late-resolved-fill-transaction",
                                "time": late_entry_at.isoformat(),
                                "type": "ORDER_FILL",
                                "orderID": late_order_id,
                                "instrument": "EUR_USD",
                                "units": "1000",
                                "reason": "LIMIT_ORDER",
                                "tradeOpened": {
                                    "tradeID": late_trade_id,
                                    "units": "1000",
                                },
                            }
                        ),
                        "LIMIT_ORDER",
                    ),
                )
            late_outcome = RealizedOutcome(
                ts_utc=(late_entry_at + timedelta(minutes=30)).isoformat(),
                trade_id=late_trade_id,
                pair="EUR_USD",
                side="LONG",
                lane_id=LANE,
                method="TREND_CONTINUATION",
                exit_reason="TAKE_PROFIT_ORDER",
                realized_pl_jpy=-1000.0,
                entry_vehicle="LIMIT",
                entry_truth_consistent=True,
                broker_close_ts_utc=(
                    late_entry_at + timedelta(minutes=30)
                ).isoformat(),
                broker_time_consistent=True,
            )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=[*outcomes, late_outcome],
            ):
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(
                validation["status"],
                "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            )
            self.assertEqual(
                validation["current_status"],
                "POST_ACTIVATION_COHORT_COMPLETE",
            )
            self.assertEqual(
                validation["current_first_trade_ids"][0],
                late_trade_id,
            )
            self.assertIn("samples", validation["changed_fields"])

    def test_current_truth_preserves_nanoseconds_for_late_first_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            first_entry_time = "2026-07-09T01:00:00.000000900Z"
            with sqlite3.connect(ledger) as conn:
                raw = json.loads(
                    conn.execute(
                        "SELECT raw_json FROM execution_events WHERE trade_id='trade-0'"
                    ).fetchone()[0]
                )
                raw["time"] = first_entry_time
                conn.execute(
                    """
                    UPDATE execution_events SET ts_utc=?, raw_json=?
                    WHERE trade_id='trade-0'
                    """,
                    (first_entry_time, json.dumps(raw)),
                )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

            late_trade_id = "late-nanosecond-first"
            late_order_id = "late-nanosecond-order"
            late_entry_time = "2026-07-09T01:00:00.000000100Z"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "ORDER_FILLED",
                        late_trade_id,
                        late_order_id,
                        LANE,
                        "EUR_USD",
                        "LONG",
                        1000,
                        late_entry_time,
                        json.dumps(
                            {
                                "id": "late-nanosecond-fill",
                                "time": late_entry_time,
                                "type": "ORDER_FILL",
                                "orderID": late_order_id,
                                "instrument": "EUR_USD",
                                "units": "1000",
                                "reason": "LIMIT_ORDER",
                                "tradeOpened": {
                                    "tradeID": late_trade_id,
                                    "units": "1000",
                                },
                            }
                        ),
                        "LIMIT_ORDER",
                    ),
                )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(
                validation["status"],
                "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            )
            self.assertEqual(validation["current_first_trade_ids"][0], late_trade_id)

    def test_current_truth_rejects_late_financing_economics_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )
            sealed_rowid = int(payload["ledger_rowid_watermark"])
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "LATE_FINANCING_MARKER",
                        "trade-0",
                        "",
                        "",
                        "EUR_USD",
                        "LONG",
                        0,
                        (REVIEWED_AT + timedelta(days=2)).isoformat(),
                        "",
                        "",
                    ),
                )
            financed = list(outcomes)
            financed[0] = RealizedOutcome(
                **{
                    **financed[0].__dict__,
                    "realized_pl_jpy": financed[0].realized_pl_jpy - 50.0,
                }
            )

            def outcomes_for_snapshot(path: Path) -> list[RealizedOutcome]:
                with sqlite3.connect(path) as conn:
                    rowid = int(
                        conn.execute(
                            "SELECT COALESCE(MAX(rowid), 0) FROM execution_events"
                        ).fetchone()[0]
                    )
                return financed if rowid > sealed_rowid else outcomes

            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                side_effect=outcomes_for_snapshot,
            ):
                validation = cohort.validate_post_activation_monitor_cohort(
                    payload,
                    ledger_path=ledger,
                )

            self.assertEqual(
                validation["status"],
                "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            )
            self.assertIn("samples", validation["changed_fields"])
            self.assertIn("primary_metric_value", validation["changed_fields"])

    def test_late_appended_earlier_broker_entry_stays_in_first_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(
                root,
                late_append_first_entry=True,
            )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

            self.assertEqual(payload["status"], "POST_ACTIVATION_COHORT_COMPLETE")
            self.assertEqual(
                payload["first_trade_ids"],
                [f"trade-{index}" for index in range(20)],
            )
            self.assertNotIn("trade-20", payload["first_trade_ids"])

    def test_activation_ledger_anchor_revalidates_exact_historical_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, _, activation_anchor = self._fixture(root)

            self.assertEqual(
                cohort.validate_execution_ledger_anchor(
                    ledger_path=ledger,
                    anchor=activation_anchor,
                ),
                activation_anchor,
            )
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    "UPDATE execution_events SET ts_utc=? WHERE rowid=1",
                    ((REVIEWED_AT - timedelta(seconds=1)).isoformat(),),
                )
            with self.assertRaisesRegex(ValueError, "prefix no longer matches"):
                cohort.validate_execution_ledger_anchor(
                    ledger_path=ledger,
                    anchor=activation_anchor,
                )

    def test_post_anchor_backfill_with_pre_activation_broker_time_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(
                root,
                pre_review_trade="trade-0",
            )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

            self.assertEqual(payload["status"], "WAITING_FOR_FIRST_20_ENTRIES")
            self.assertEqual(payload["entry_count"], 19)
            self.assertNotIn("trade-0", payload["first_trade_ids"])

    def test_post_activation_monitor_rejects_normalized_entry_time_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            with sqlite3.connect(ledger) as conn:
                raw = json.loads(
                    conn.execute(
                        """
                        SELECT raw_json FROM execution_events
                        WHERE event_type='ORDER_FILLED'
                        ORDER BY rowid LIMIT 1
                        """
                    ).fetchone()[0]
                )
                raw["time"] = (
                    REVIEWED_AT + timedelta(hours=1, microseconds=1)
                ).isoformat()
                conn.execute(
                    """
                    UPDATE execution_events SET raw_json=?
                    WHERE event_type='ORDER_FILLED'
                      AND rowid=(
                          SELECT MIN(rowid) FROM execution_events
                          WHERE event_type='ORDER_FILLED'
                      )
                    """,
                    (json.dumps(raw),),
                )
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(ValueError, "canonical post-activation ledger is unreadable"),
            ):
                cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

    def test_post_activation_monitor_rejects_unverified_raw_close_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(root)
            forged = list(outcomes)
            forged[0] = RealizedOutcome(
                **{
                    **forged[0].__dict__,
                    "broker_time_consistent": False,
                }
            )
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=forged),
                self.assertRaisesRegex(ValueError, "broker timestamps are unverified"),
            ):
                cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

    def test_post_activation_monitor_cannot_substitute_a_later_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, _, _, outcomes, activation_anchor = self._fixture(
                root,
                unresolved_first_with_later_resolved=True,
            )
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_post_activation_monitor_cohort(
                    ledger_path=ledger,
                    lane_id=LANE,
                    activated_at_utc=REVIEWED_AT.isoformat(),
                    activation_ledger_anchor=activation_anchor,
                )

            self.assertEqual(
                payload["status"],
                "WAITING_FOR_FIRST_20_RESOLUTIONS",
            )
            self.assertIn("trade-0", payload["unresolved_trade_ids"])
            self.assertNotIn("trade-20", payload["first_trade_ids"])

    def test_missing_losing_thesis_fails_instead_of_cherry_picking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                missing_thesis_trade="trade-0",
            )
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(ValueError, "exactly one entry-time thesis"),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

    def test_pre_review_trade_cannot_fill_the_forward_sample_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                pre_review_trade="trade-0",
            )
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(ValueError, "found 19"),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

    def test_entry_at_exact_review_timestamp_is_not_forward_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                at_review_trade="trade-0",
            )
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(ValueError, "found 19"),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

    def test_reviewed_lane_cannot_be_reselected_after_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(root)
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(ValueError, "review identity is unsupported"),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:LIMIT",
                )

    def test_unresolved_early_entry_cannot_be_skipped_for_later_winners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                unresolved_first_with_later_resolved=True,
            )
            self.assertEqual(len(outcomes), 20)
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(
                    ValueError,
                    "first forward entry cohort is unresolved",
                ),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

    def test_inherited_gateway_lane_entry_cannot_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(
                root,
                unresolved_first_with_later_resolved=True,
                null_lane_first_with_gateway=True,
            )
            self.assertEqual(len(outcomes), 20)
            with (
                patch.object(cohort, "read_attributed_net_outcomes", return_value=outcomes),
                self.assertRaisesRegex(
                    ValueError,
                    "first forward entry cohort is unresolved",
                ),
            ):
                cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )

    def test_canonical_revalidation_rejects_tampered_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, theses, forecasts, outcomes, _ = self._fixture(root)
            with patch.object(
                cohort,
                "read_attributed_net_outcomes",
                return_value=outcomes,
            ):
                payload = cohort.build_canonical_forward_cohort(
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                    review_completed_at_utc=REVIEWED_AT.isoformat(),
                    lane_id=LANE,
                )
                forged = json.loads(json.dumps(payload))
                forged["samples"][0]["realized_net_jpy"] = 9999.0
                forged["samples"][0]["net_jpy_per_1000_units"] = 9999.0
                validation = cohort.validate_canonical_forward_cohort(
                    forged,
                    ledger_path=ledger,
                    entry_thesis_path=theses,
                    forecast_history_path=forecasts,
                    review=_review(),
                )

            self.assertEqual(validation["status"], "CANONICAL_COHORT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
