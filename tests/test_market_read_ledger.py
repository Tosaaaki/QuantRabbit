from __future__ import annotations

import json
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.market_read_ledger import (
    _exact_order_fill_evidence,
    market_read_feedback_summary,
    record_market_read_prediction,
    refresh_market_read_measurements,
)
from quant_rabbit.decision_execution_lineage import (
    DecisionExecutionLineage,
    append_execution_link,
    build_execution_link,
)


UTC = timezone.utc
PAIR = "EUR_USD"


def _decision(
    predicted_at: datetime,
    *,
    direction: str = "LONG",
    target: float = 1.1020,
    invalidation: float = 1.0980,
    applied_at: datetime | None = None,
) -> dict:
    decision = {
        "generated_at_utc": predicted_at.isoformat(),
        "action": "WAIT",
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "market_read_first": {
            "naked_read": {
                "cleanest_pair_expression": PAIR,
                "tape_state": "TREND",
                "location_24h": "MID",
                "thesis_state": "BUILDING",
            },
            "next_30m_prediction": {
                "pair": PAIR,
                "direction": direction,
                "target_zone": str(target),
                "invalidation": str(invalidation),
            },
            "next_2h_prediction": {
                "pair": PAIR,
                "direction": direction,
                "target_zone": str(target),
                "invalidation": str(invalidation),
            },
            "best_trade_if_forced": {
                "pair": PAIR,
                "direction": direction,
                "entry": "1.1000",
                "tp": str(target),
                "sl": str(invalidation),
                "vehicle": "MARKET",
            },
        },
    }
    if applied_at is not None:
        decision["decision_provenance"] = {
            "author_kind": "CODEX_MARKET_READ",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "authored_at_utc": applied_at.isoformat(),
            "applied_at_utc": applied_at.isoformat(),
        }
    return decision


def _packet(snapshot_at: datetime) -> dict:
    timestamp = snapshot_at.isoformat()
    return {
        "broker_snapshot": {
            "fetched_at_utc": timestamp,
            "quotes": {
                PAIR: {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "timestamp_utc": timestamp,
                }
            },
        },
        "lanes": [],
    }


def _candles(
    predicted_at: datetime,
    *,
    count: int = 24,
    missing_indices: set[int] | None = None,
    overrides: dict[int, dict[str, float]] | None = None,
) -> list[dict]:
    missing_indices = missing_indices or set()
    overrides = overrides or {}
    rows: list[dict] = []
    for index in range(count):
        if index in missing_indices:
            continue
        close = 1.1001 + (index * 0.00002)
        row = {
            "t": (predicted_at + timedelta(minutes=5 * index)).isoformat(),
            "o": close - 0.00002,
            "h": close + 0.00005,
            "l": close - 0.00005,
            "c": close,
            "complete": True,
        }
        row.update(overrides.get(index, {}))
        rows.append(row)
    return rows


def _write_pair_charts(path: Path, candles: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": PAIR,
                        "views": [
                            {
                                "granularity": "M5",
                                "recent_candles": candles,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class MarketReadLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predicted_at = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
        self.resolved_at = self.predicted_at + timedelta(hours=2, minutes=5)

    def _record(
        self,
        root: Path,
        *,
        decision: dict | None = None,
        packet: dict | None = None,
        candles: list[dict] | None = None,
        execution_ledger_path: Path | None = None,
        now: datetime | None = None,
    ) -> dict:
        charts_path = root / "pair_charts.json"
        _write_pair_charts(charts_path, candles if candles is not None else _candles(self.predicted_at))
        return record_market_read_prediction(
            decision or _decision(self.predicted_at),
            packet or _packet(self.predicted_at),
            status="ACCEPTED",
            issues=(),
            predictions_path=root / "market_read_predictions.jsonl",
            report_path=root / "market_read_score_report.md",
            pair_charts_path=charts_path,
            execution_ledger_path=execution_ledger_path,
            now=now or self.resolved_at,
        )

    def test_v1_bytes_are_preserved_and_v2_is_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "market_read_predictions.jsonl"
            legacy = '{  "prediction_id": "legacy-1", "verdict": "CORRECT" }\n'
            path.write_text(legacy, encoding="utf-8")

            result = self._record(root)

            self.assertEqual(result["status"], "RECORDED")
            self.assertTrue(path.read_text(encoding="utf-8").startswith(legacy))
            rows = _rows(path)
            self.assertEqual(len(rows), 2)
            self.assertNotIn("schema_version", rows[0])
            self.assertEqual(rows[1]["schema_version"], 2)
            self.assertEqual(
                rows[1]["originating_decision_receipt_id"],
                result["decision_receipt_id"],
            )
            self.assertTrue(rows[1]["originating_decision_receipt_id"].startswith("gptd:"))
            self.assertEqual(
                rows[1]["direct_execution_attribution"]["market_read_prediction_id"],
                result["prediction_id"],
            )
            self.assertFalse(
                rows[1]["direct_execution_attribution"]["pair_or_time_inference_used"]
            )
            self.assertEqual(rows[1]["direct_realized_outcome"]["status"], "UNRESOLVED")
            report = (root / "market_read_score_report.md").read_text(encoding="utf-8")
            self.assertIn("stored verdicts preserved; not direction accuracy", report)

    def test_direction_correct_is_separate_from_target_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self._record(root)

            row = _rows(root / "market_read_predictions.jsonl")[0]
            for horizon in ("30m", "2h"):
                scored = row["horizon_results"][horizon]
                self.assertEqual(scored["direction_status"], "CORRECT")
                self.assertEqual(scored["target_completion_status"], "NOT_TOUCHED")
                self.assertEqual(scored["first_touch_status"], "NEITHER_TOUCHED")
                self.assertEqual(
                    scored["full_read_status"],
                    "DIRECTION_CORRECT_TARGET_INCOMPLETE",
                )
                self.assertEqual(scored["truth_source"], "MID_CANDLE_DIAGNOSTIC")
                self.assertFalse(scored["live_permission"])
            two_hour = row["horizon_results"]["2h"]
            self.assertEqual(two_hour["endpoint_observed_at_utc"], "2026-07-10T02:00:00Z")
            self.assertEqual(two_hour["endpoint_offset_from_due_seconds"], 0.0)
            self.assertEqual(two_hour["resolution_lag_seconds"], 300.0)

    def test_hyphenated_target_zone_keeps_second_price_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(self.predicted_at, target="1.1020-1.1030")

            self._record(root, decision=decision)

            row = _rows(root / "market_read_predictions.jsonl")[0]
            self.assertTrue(row["score_eligible"])
            self.assertEqual(row["horizon_results"]["30m"]["target_price"], 1.102)

    def test_straddling_directional_rails_are_ineligible_and_never_scored_as_touched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(self.predicted_at, target="1.0990-1.1020")

            self._record(root, decision=decision)

            row = _rows(root / "market_read_predictions.jsonl")[0]
            self.assertFalse(row["score_eligible"])
            self.assertIn(
                "NEXT_30M_PREDICTION_TARGET_GEOMETRY_CONFLICT",
                row["score_ineligible_reasons"],
            )
            thirty = row["horizon_results"]["30m"]
            self.assertEqual(thirty["target_completion_status"], "UNSCORABLE_GEOMETRY")
            self.assertNotEqual(thirty["target_completion_status"], "TOUCHED")

    def test_codex_prediction_starts_at_applied_time_not_baseline_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applied_at = self.predicted_at + timedelta(minutes=10)
            decision = _decision(self.predicted_at, applied_at=applied_at)
            candles = _candles(
                self.predicted_at,
                overrides={
                    # This target touch was visible before Codex published the
                    # overlay and must never count as forward performance.
                    1: {"h": 1.1022},
                },
            )

            self._record(root, decision=decision, candles=candles)

            row = _rows(root / "market_read_predictions.jsonl")[0]
            self.assertEqual(row["generated_at_utc"], "2026-07-10T00:10:00Z")
            self.assertEqual(row["baseline_generated_at_utc"], "2026-07-10T00:00:00Z")
            self.assertEqual(row["prediction_time_basis"], "CODEX_APPLIED_AT")
            self.assertEqual(row["source_quote_lag_seconds"], 600.0)
            self.assertFalse(row["score_eligible"])
            self.assertIn(
                "SOURCE_QUOTE_LAG_EXCEEDS_WINDOW",
                row["score_ineligible_reasons"],
            )
            self.assertEqual(
                row["horizon_results"]["30m"]["target_completion_status"],
                "NOT_TOUCHED",
            )

    def test_codex_source_lag_inside_five_minutes_remains_score_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applied_at = self.predicted_at + timedelta(seconds=30)

            self._record(
                root,
                decision=_decision(self.predicted_at, applied_at=applied_at),
            )

            row = _rows(root / "market_read_predictions.jsonl")[0]
            self.assertEqual(row["source_quote_lag_seconds"], 30.0)
            self.assertTrue(row["score_eligible"])

    def test_ineligible_unresolved_prediction_remains_in_lifecycle_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applied_at = self.predicted_at + timedelta(minutes=10)

            self._record(
                root,
                decision=_decision(self.predicted_at, applied_at=applied_at),
                now=applied_at + timedelta(minutes=1),
            )
            feedback = market_read_feedback_summary(
                root / "market_read_predictions.jsonl"
            )

            self.assertEqual(feedback["status"], "V2_EVIDENCE_UNRESOLVED")
            self.assertEqual(feedback["metrics"]["rows"], 1)
            self.assertEqual(feedback["metrics"]["score_eligible"], 0)
            self.assertEqual(feedback["metrics"]["score_ineligible"], 1)
            self.assertEqual(
                feedback["metrics"]["score_ineligible_reason_counts"][
                    "SOURCE_QUOTE_LAG_EXCEEDS_WINDOW"
                ],
                1,
            )
            for horizon in ("30m", "2h"):
                metrics = feedback["metrics"]["horizons"][horizon]
                self.assertEqual(metrics["resolved"], 0)
                self.assertEqual(metrics["unresolved"], 1)
                self.assertEqual(metrics["eligible_resolved"], 0)
                self.assertEqual(metrics["eligible_unresolved"], 0)
                self.assertEqual(metrics["ineligible_resolved"], 0)
                self.assertEqual(metrics["ineligible_unresolved"], 1)

            report = (root / "market_read_score_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "Lifecycle resolved/unresolved (all v2): `0` / `1`",
                report,
            )
            self.assertIn(
                "Score-eligible resolved/unresolved: `0` / `0`",
                report,
            )
            self.assertIn(
                "Score-ineligible resolved/unresolved: `0` / `1`",
                report,
            )

    def test_malformed_ineligible_reasons_are_fail_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applied_at = self.predicted_at + timedelta(minutes=10)
            path = root / "market_read_predictions.jsonl"
            self._record(
                root,
                decision=_decision(self.predicted_at, applied_at=applied_at),
                now=applied_at + timedelta(minutes=1),
            )
            row = _rows(path)[0]
            row["score_ineligible_reasons"] = 1
            row["duplicate_observation_count"] = {"invalid": True}
            surrogate_row = dict(row)
            surrogate_row["prediction_id"] = "mr2:surrogate-reason"
            surrogate_row["score_ineligible_reasons"] = ["bad\ud800reason"]
            surrogate_row["duplicate_observation_count"] = "not-an-int"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(surrogate_row) + "\n",
                encoding="utf-8",
            )

            feedback = market_read_feedback_summary(path)

            self.assertEqual(feedback["metrics"]["score_ineligible"], 2)
            self.assertEqual(
                feedback["metrics"]["score_ineligible_reason_counts"],
                {"MALFORMED_SCORE_INELIGIBLE_REASONS": 2},
            )
            self.assertEqual(
                feedback["metrics"]["coalesced_duplicate_observations"],
                0,
            )

    def test_first_touch_requires_path_and_same_candle_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candles = _candles(
                self.predicted_at,
                overrides={
                    2: {
                        "o": 1.1000,
                        "h": 1.1022,
                        "l": 1.0978,
                        "c": 1.1002,
                    }
                },
            )

            self._record(root, candles=candles)

            row = _rows(root / "market_read_predictions.jsonl")[0]
            thirty = row["horizon_results"]["30m"]
            self.assertEqual(thirty["target_completion_status"], "TOUCHED")
            self.assertEqual(thirty["invalidation_status"], "TOUCHED")
            self.assertEqual(thirty["first_touch_status"], "AMBIGUOUS_SAME_CANDLE")
            self.assertEqual(thirty["full_read_status"], "AMBIGUOUS_SAME_CANDLE")
            self.assertNotIn("INVALIDATED_FIRST", json.dumps(thirty))

    def test_missing_m5_truth_remains_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self._record(root, candles=_candles(self.predicted_at, missing_indices={3}))

            row = _rows(root / "market_read_predictions.jsonl")[0]
            for horizon in ("30m", "2h"):
                scored = row["horizon_results"][horizon]
                self.assertEqual(scored["resolution_status"], "UNRESOLVED")
                self.assertEqual(scored["unresolved_reason"], "M5_WINDOW_INCOMPLETE")
                self.assertEqual(scored["full_read_status"], "UNRESOLVED")

    def test_exact_source_snapshot_repeat_coalesces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            first = self._record(root, now=self.predicted_at + timedelta(minutes=1))
            second_decision = _decision(self.predicted_at + timedelta(seconds=20))
            second = self._record(
                root,
                decision=second_decision,
                now=self.predicted_at + timedelta(minutes=2),
            )

            self.assertEqual(first["status"], "RECORDED")
            self.assertEqual(second["status"], "COALESCED_EXACT_DUPLICATE")
            rows = _rows(root / "market_read_predictions.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["duplicate_observation_count"], 1)
            self.assertEqual(len(rows[0]["decision_observations"]), 2)

    def test_conflicting_read_from_same_snapshot_is_score_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            conflicting = _decision(
                self.predicted_at + timedelta(seconds=20),
                direction="SHORT",
                target=1.0980,
                invalidation=1.1020,
            )

            result = self._record(
                root,
                decision=conflicting,
                now=self.predicted_at + timedelta(minutes=2),
            )

            self.assertEqual(result["status"], "RECORDED_SOURCE_SNAPSHOT_CONFLICT")
            rows = _rows(root / "market_read_predictions.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["source_snapshot_conflict"] for row in rows))
            self.assertTrue(all(row["score_eligible"] is False for row in rows))
            self.assertTrue(
                all(
                    "SOURCE_SNAPSHOT_PREDICTION_CONFLICT" in row["score_ineligible_reasons"]
                    for row in rows
                )
            )

    def test_feedback_is_bounded_advisory_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root)

            feedback = market_read_feedback_summary(root / "market_read_predictions.jsonl")

            self.assertEqual(feedback["status"], "OK")
            self.assertTrue(feedback["advisory_only"])
            self.assertTrue(feedback["read_only"])
            self.assertFalse(feedback["live_permission"])
            self.assertFalse(feedback["may_change_execution_permission"])
            self.assertEqual(feedback["metrics"]["horizons"]["30m"]["direction_accuracy_pct"], 100.0)
            self.assertEqual(feedback["metrics"]["horizons"]["30m"]["target_completion_pct"], 0.0)
            self.assertEqual(
                feedback["metrics"]["reaction_chains"]["first_subsequent_decision_unresolved"],
                1,
            )
            self.assertEqual(len(feedback["latest_resolved"]), 1)

    def test_predecision_refresh_resolves_due_truth_before_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))

            refresh = refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                now=self.resolved_at,
            )
            feedback = market_read_feedback_summary(root / "market_read_predictions.jsonl")

            self.assertEqual(refresh["status"], "REFRESHED")
            self.assertTrue(refresh["read_only_measurement"])
            self.assertFalse(refresh["live_permission"])
            self.assertFalse(refresh["may_change_execution_permission"])
            self.assertEqual(feedback["status"], "OK")
            self.assertEqual(feedback["metrics"]["horizons"]["2h"]["resolved"], 1)

            unchanged = refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                now=self.resolved_at + timedelta(minutes=1),
            )
            self.assertEqual(unchanged["status"], "NO_CHANGE")

    def test_first_subsequent_decision_is_linked_once_by_strict_record_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            next_at = self.predicted_at + timedelta(hours=1)

            second_result = self._record(
                root,
                decision=_decision(next_at),
                packet=_packet(next_at),
                now=next_at + timedelta(minutes=1),
            )

            rows = _rows(root / "market_read_predictions.jsonl")
            first_chain = rows[0]["reaction_chain"]
            reaction = first_chain["first_subsequent_decision"]
            self.assertEqual(reaction["status"], "RESOLVED")
            self.assertTrue(reaction["decision_receipt_id"].startswith("gptd:"))
            self.assertEqual(reaction["decision_receipt_id"], second_result["decision_receipt_id"])
            self.assertEqual(reaction["decision_recorded_at_utc"], "2026-07-10T01:01:00Z")
            self.assertEqual(reaction["action"], "WAIT")
            self.assertEqual(
                reaction["reaction_link_basis"],
                "NEXT_MARKET_READ_LEDGER_RECEIPT_IN_STRICT_RECORD_ORDER",
            )
            self.assertEqual(first_chain["execution_attribution"]["status"], "UNATTRIBUTED")
            self.assertEqual(first_chain["realized_outcome"]["status"], "UNRESOLVED")
            receipt_id = reaction["decision_receipt_id"]

            later_at = self.predicted_at + timedelta(hours=1, minutes=10)
            self._record(
                root,
                decision=_decision(later_at),
                packet=_packet(later_at),
                now=later_at + timedelta(minutes=1),
            )
            first_after_third = _rows(root / "market_read_predictions.jsonl")[0]
            self.assertEqual(
                first_after_third["reaction_chain"]["first_subsequent_decision"]["decision_receipt_id"],
                receipt_id,
            )

    def test_explicit_close_trade_id_joins_to_realized_pl_without_time_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            ledger_path = root / "execution_ledger.db"
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events(
                        event_uid TEXT,
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        realized_pl_jpy REAL,
                        financing_jpy REAL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "close:t-explicit",
                        "2026-07-10T01:00:30Z",
                        "TRADE_CLOSED",
                        "t-explicit",
                        123.4,
                        -3.0,
                    ),
                )
            close_at = self.predicted_at + timedelta(hours=1)
            close_decision = _decision(close_at)
            close_decision["action"] = "CLOSE"
            close_decision["close_trade_ids"] = ["t-explicit"]

            self._record(
                root,
                decision=close_decision,
                packet=_packet(close_at),
                execution_ledger_path=ledger_path,
                now=close_at + timedelta(minutes=1),
            )

            chain = _rows(root / "market_read_predictions.jsonl")[0]["reaction_chain"]
            attribution = chain["execution_attribution"]
            self.assertEqual(attribution["status"], "PARTIALLY_ATTRIBUTED")
            self.assertEqual(attribution["trade_ids"]["ids"], ["t-explicit"])
            self.assertEqual(attribution["fill_ids"]["status"], "UNATTRIBUTED")
            realized = chain["realized_outcome"]
            self.assertEqual(realized["status"], "RESOLVED")
            self.assertEqual(realized["realized_pl_jpy"], 123.4)
            self.assertEqual(realized["financing_jpy"], -3.0)
            self.assertEqual(realized["net_realized_jpy"], 120.4)
            self.assertEqual(realized["trade_outcomes"][0]["execution_event_ids"], ["close:t-explicit"])

    def test_actual_gateway_link_reconciles_reaction_and_realized_pl_by_two_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            next_at = self.predicted_at + timedelta(hours=1)
            trade_decision = _decision(next_at)
            trade_decision["action"] = "TRADE"
            trade_decision["selected_lane_id"] = "lane:EUR_USD:LONG"
            trade_decision["selected_lane_ids"] = ["lane:EUR_USD:LONG"]
            second_result = self._record(
                root,
                decision=trade_decision,
                packet=_packet(next_at),
                now=next_at + timedelta(minutes=1),
            )
            decision_id = second_result["decision_receipt_id"]
            prediction_id = second_result["prediction_id"]
            token = "mdl-" + hashlib.sha256(
                f"{decision_id}\0{prediction_id}".encode("utf-8")
            ).hexdigest()[:20]
            lineage = DecisionExecutionLineage(
                decision_receipt_id=decision_id,
                market_read_prediction_id=prediction_id,
                lineage_token=token,
                decision_generated_at_utc=next_at.isoformat(),
            )
            links_path = root / "market_read_execution_links.jsonl"
            link = build_execution_link(
                lineage=lineage,
                gateway_response={
                    "orderCreateTransaction": {"id": "order-101"},
                    "orderFillTransaction": {
                        "id": "fill-102",
                        "orderID": "order-101",
                        "tradeOpened": {"tradeID": "trade-200"},
                    },
                    "relatedTransactionIDs": ["order-101", "fill-102"],
                },
                lane_id="lane:EUR_USD:LONG",
                parent_lane_id="lane:EUR_USD:LONG",
                forecast_cycle_id="forecast-1",
                claim_id="claim-1",
                order_request_sha256="c" * 64,
                client_extension_id=f"qrv1-EURUSD-L-test-{token}",
            )
            append_execution_link(links_path, link)

            ledger_path = root / "execution_ledger.db"
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events(
                        event_uid TEXT,
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        realized_pl_jpy REAL,
                        financing_jpy REAL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "close:trade-200",
                        "2026-07-10T02:30:00Z",
                        "TRADE_CLOSED",
                        "trade-200",
                        321.0,
                        -4.0,
                    ),
                )

            refresh = refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                execution_ledger_path=ledger_path,
                execution_links_path=links_path,
                now=self.resolved_at + timedelta(hours=1),
            )

            self.assertEqual(refresh["execution_links_status"], "VALID")
            rows = _rows(root / "market_read_predictions.jsonl")
            chain = rows[0]["reaction_chain"]
            attribution = chain["execution_attribution"]
            self.assertEqual(attribution["attribution_basis"], "EXPLICIT_ACTUAL_GATEWAY_RESPONSE_IDS_ONLY")
            self.assertEqual(attribution["decision_receipt_id"], decision_id)
            self.assertEqual(attribution["market_read_prediction_id"], prediction_id)
            self.assertEqual(attribution["order_ids"]["ids"], ["order-101"])
            self.assertEqual(attribution["fill_ids"]["ids"], ["fill-102"])
            self.assertEqual(attribution["trade_ids"]["ids"], ["trade-200"])
            self.assertFalse(attribution["pair_or_time_inference_used"])
            realized = chain["realized_outcome"]
            self.assertEqual(realized["status"], "RESOLVED")
            self.assertEqual(realized["net_realized_jpy"], 317.0)

            # The same execution must also be visible on the prediction that
            # actually originated this decision. The prior row's reaction
            # chain remains a separate prior-read -> next-decision view.
            current = rows[1]
            self.assertEqual(current["originating_decision_receipt_id"], decision_id)
            direct = current["direct_execution_attribution"]
            self.assertEqual(
                direct["attribution_basis"],
                "EXPLICIT_ACTUAL_GATEWAY_RESPONSE_IDS_ONLY",
            )
            self.assertEqual(direct["decision_receipt_id"], decision_id)
            self.assertEqual(direct["market_read_prediction_id"], prediction_id)
            self.assertEqual(direct["order_ids"]["ids"], ["order-101"])
            self.assertEqual(direct["fill_ids"]["ids"], ["fill-102"])
            self.assertEqual(direct["trade_ids"]["ids"], ["trade-200"])
            self.assertFalse(direct["pair_or_time_inference_used"])
            direct_realized = current["direct_realized_outcome"]
            self.assertEqual(direct_realized["status"], "RESOLVED")
            self.assertEqual(direct_realized["net_realized_jpy"], 317.0)
            report = (root / "market_read_score_report.md").read_text()
            self.assertIn("Direct execution partially attributed/unattributed", report)
            self.assertIn("Reaction execution partially attributed/unattributed", report)

    def test_pending_gateway_order_reconciles_later_exact_fill_and_close_pl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            next_at = self.predicted_at + timedelta(hours=1)
            trade_decision = _decision(next_at)
            trade_decision["action"] = "TRADE"
            trade_decision["selected_lane_id"] = "lane:EUR_USD:LONG"
            trade_decision["selected_lane_ids"] = ["lane:EUR_USD:LONG"]
            current = self._record(
                root,
                decision=trade_decision,
                packet=_packet(next_at),
                now=next_at + timedelta(minutes=1),
            )
            decision_id = current["decision_receipt_id"]
            prediction_id = current["prediction_id"]
            token = "mdl-" + hashlib.sha256(
                f"{decision_id}\0{prediction_id}".encode("utf-8")
            ).hexdigest()[:20]
            links_path = root / "market_read_execution_links.jsonl"
            append_execution_link(
                links_path,
                build_execution_link(
                    lineage=DecisionExecutionLineage(
                        decision_receipt_id=decision_id,
                        market_read_prediction_id=prediction_id,
                        lineage_token=token,
                        decision_generated_at_utc=next_at.isoformat(),
                    ),
                    # A pending LIMIT/STOP POST commonly returns only this
                    # exact order-create transaction; no fill/trade id exists
                    # yet in the gateway response.
                    gateway_response={
                        "orderCreateTransaction": {"id": "pending-order-501"}
                    },
                    lane_id="lane:EUR_USD:LONG",
                    parent_lane_id="lane:EUR_USD:LONG",
                    forecast_cycle_id="forecast-pending",
                    claim_id="claim-pending",
                    order_request_sha256="d" * 64,
                    client_extension_id=f"qrv1-pending-{token}",
                ),
            )

            ledger_path = root / "execution_ledger.db"
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events(
                        event_uid TEXT,
                        ts_utc TEXT,
                        event_type TEXT,
                        order_id TEXT,
                        trade_id TEXT,
                        realized_pl_jpy REAL,
                        financing_jpy REAL,
                        oanda_transaction_id TEXT
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "fill:pending-order-501",
                            "2026-07-10T01:10:00Z",
                            "ORDER_FILLED",
                            "pending-order-501",
                            "trade-900",
                            None,
                            None,
                            "fill-transaction-502",
                        ),
                        (
                            "close:trade-900",
                            "2026-07-10T02:30:00Z",
                            "TRADE_CLOSED",
                            "close-order-901",
                            "trade-900",
                            250.0,
                            -2.0,
                            "close-transaction-902",
                        ),
                    ],
                )

            refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                execution_ledger_path=ledger_path,
                execution_links_path=links_path,
                now=self.resolved_at + timedelta(hours=1),
            )

            rows = _rows(root / "market_read_predictions.jsonl")
            reaction_attribution = rows[0]["reaction_chain"]["execution_attribution"]
            reaction_outcome = rows[0]["reaction_chain"]["realized_outcome"]
            direct_attribution = rows[1]["direct_execution_attribution"]
            direct_outcome = rows[1]["direct_realized_outcome"]
            for attribution, outcome in (
                (reaction_attribution, reaction_outcome),
                (direct_attribution, direct_outcome),
            ):
                self.assertEqual(
                    attribution["attribution_basis"],
                    "EXPLICIT_ACTUAL_GATEWAY_ORDER_ID_THEN_EXACT_EXECUTION_LEDGER_ORDER_FILLED",
                )
                self.assertEqual(
                    attribution["order_ids"]["ids"], ["pending-order-501"]
                )
                self.assertEqual(
                    attribution["fill_ids"]["ids"], ["fill-transaction-502"]
                )
                self.assertEqual(attribution["trade_ids"]["ids"], ["trade-900"])
                self.assertIn(
                    "fill-transaction-502", attribution["transaction_ids"]["ids"]
                )
                reconciliation = attribution["order_fill_reconciliation"]
                self.assertEqual(reconciliation["status"], "RESOLVED")
                self.assertEqual(
                    reconciliation["attribution_basis"],
                    "EXACT_ORDER_ID_EQUALITY_ON_ORDER_FILLED",
                )
                self.assertFalse(reconciliation["pair_or_time_inference_used"])
                self.assertFalse(attribution["pair_or_time_inference_used"])
                self.assertEqual(outcome["status"], "RESOLVED")
                self.assertEqual(outcome["net_realized_jpy"], 248.0)

    def test_pending_gateway_order_never_joins_a_different_filled_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            next_at = self.predicted_at + timedelta(hours=1)
            trade_decision = _decision(next_at)
            trade_decision["action"] = "TRADE"
            trade_decision["selected_lane_id"] = "lane:EUR_USD:LONG"
            trade_decision["selected_lane_ids"] = ["lane:EUR_USD:LONG"]
            current = self._record(
                root,
                decision=trade_decision,
                packet=_packet(next_at),
                now=next_at + timedelta(minutes=1),
            )
            decision_id = current["decision_receipt_id"]
            prediction_id = current["prediction_id"]
            token = "mdl-" + hashlib.sha256(
                f"{decision_id}\0{prediction_id}".encode("utf-8")
            ).hexdigest()[:20]
            links_path = root / "market_read_execution_links.jsonl"
            append_execution_link(
                links_path,
                build_execution_link(
                    lineage=DecisionExecutionLineage(
                        decision_receipt_id=decision_id,
                        market_read_prediction_id=prediction_id,
                        lineage_token=token,
                        decision_generated_at_utc=next_at.isoformat(),
                    ),
                    gateway_response={
                        "orderCreateTransaction": {"id": "pending-order-501"}
                    },
                    lane_id="lane:EUR_USD:LONG",
                    parent_lane_id=None,
                    forecast_cycle_id=None,
                    claim_id=None,
                    order_request_sha256=None,
                    client_extension_id=f"qrv1-pending-{token}",
                ),
            )

            ledger_path = root / "execution_ledger.db"
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events(
                        event_uid TEXT,
                        ts_utc TEXT,
                        event_type TEXT,
                        order_id TEXT,
                        trade_id TEXT,
                        realized_pl_jpy REAL,
                        financing_jpy REAL,
                        oanda_transaction_id TEXT
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "fill:other-order",
                            "2026-07-10T01:10:00Z",
                            "ORDER_FILLED",
                            "other-order",
                            "trade-900",
                            None,
                            None,
                            "fill-other",
                        ),
                        (
                            "close:trade-900",
                            "2026-07-10T02:30:00Z",
                            "TRADE_CLOSED",
                            "close-order-901",
                            "trade-900",
                            999.0,
                            0.0,
                            "close-transaction-902",
                        ),
                    ],
                )

            refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                execution_ledger_path=ledger_path,
                execution_links_path=links_path,
                now=self.resolved_at + timedelta(hours=1),
            )

            rows = _rows(root / "market_read_predictions.jsonl")
            for attribution, outcome in (
                (
                    rows[0]["reaction_chain"]["execution_attribution"],
                    rows[0]["reaction_chain"]["realized_outcome"],
                ),
                (rows[1]["direct_execution_attribution"], rows[1]["direct_realized_outcome"]),
            ):
                self.assertEqual(
                    attribution["order_ids"]["ids"], ["pending-order-501"]
                )
                self.assertEqual(attribution["trade_ids"]["ids"], [])
                self.assertEqual(attribution["fill_ids"]["ids"], [])
                reconciliation = attribution["order_fill_reconciliation"]
                self.assertEqual(reconciliation["status"], "UNRESOLVED")
                self.assertEqual(
                    reconciliation["error"],
                    "EXACT_GATEWAY_ORDER_ID_NOT_FILLED",
                )
                self.assertFalse(reconciliation["pair_or_time_inference_used"])
                self.assertFalse(attribution["pair_or_time_inference_used"])
                self.assertEqual(outcome["status"], "UNRESOLVED")
                self.assertEqual(
                    outcome["unresolved_reason"],
                    "EXACT_GATEWAY_ORDER_ID_NOT_FILLED",
                )
                self.assertIsNone(outcome["net_realized_jpy"])

    def test_exact_order_fill_reconciliation_fails_closed_on_missing_or_malformed_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_evidence, missing_error = _exact_order_fill_evidence(
                root / "missing.db",
                ["pending-order-501"],
            )
            self.assertIsNone(missing_evidence)
            self.assertEqual(missing_error, "EXECUTION_LEDGER_MISSING")

            malformed = root / "malformed.db"
            with closing(sqlite3.connect(malformed)) as conn, conn:
                conn.execute("CREATE TABLE execution_events(order_id TEXT)")
                conn.execute(
                    "INSERT INTO execution_events VALUES (?)",
                    ("pending-order-501",),
                )
            malformed_evidence, malformed_error = _exact_order_fill_evidence(
                malformed,
                ["pending-order-501"],
            )
            self.assertIsNone(malformed_evidence)
            self.assertEqual(malformed_error, "EXECUTION_LEDGER_SCHEMA_INVALID")

    def test_gateway_link_with_wrong_prediction_id_never_joins_by_pair_or_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, now=self.predicted_at + timedelta(minutes=1))
            next_at = self.predicted_at + timedelta(hours=1)
            second_result = self._record(
                root,
                decision=_decision(next_at),
                packet=_packet(next_at),
                now=next_at + timedelta(minutes=1),
            )
            decision_id = second_result["decision_receipt_id"]
            wrong_prediction_id = "mr2:" + "f" * 64
            token = "mdl-" + hashlib.sha256(
                f"{decision_id}\0{wrong_prediction_id}".encode("utf-8")
            ).hexdigest()[:20]
            links_path = root / "market_read_execution_links.jsonl"
            append_execution_link(
                links_path,
                build_execution_link(
                    lineage=DecisionExecutionLineage(
                        decision_receipt_id=decision_id,
                        market_read_prediction_id=wrong_prediction_id,
                        lineage_token=token,
                        decision_generated_at_utc=next_at.isoformat(),
                    ),
                    gateway_response={"orderCreateTransaction": {"id": "wrong-order"}},
                    lane_id="lane:EUR_USD:LONG",
                    parent_lane_id=None,
                    forecast_cycle_id=None,
                    claim_id=None,
                    order_request_sha256=None,
                    client_extension_id=f"qrv1-test-{token}",
                ),
            )

            refresh_market_read_measurements(
                predictions_path=root / "market_read_predictions.jsonl",
                report_path=root / "market_read_score_report.md",
                pair_charts_path=root / "pair_charts.json",
                execution_links_path=links_path,
                now=self.resolved_at + timedelta(hours=1),
            )

            rows = _rows(root / "market_read_predictions.jsonl")
            attribution = rows[0]["reaction_chain"]["execution_attribution"]
            self.assertEqual(attribution["status"], "UNATTRIBUTED")
            self.assertEqual(
                attribution["unattributed_reason"],
                "EXPLICIT_EXECUTION_LINK_PREDICTION_ID_MISMATCH",
            )
            self.assertEqual(attribution["order_ids"]["ids"], [])
            self.assertFalse(attribution["pair_or_time_inference_used"])
            direct = rows[1]["direct_execution_attribution"]
            self.assertEqual(direct["status"], "UNATTRIBUTED")
            self.assertEqual(
                direct["unattributed_reason"],
                "EXPLICIT_EXECUTION_LINK_PREDICTION_ID_MISMATCH",
            )
            self.assertEqual(direct["order_ids"]["ids"], [])
            self.assertFalse(direct["pair_or_time_inference_used"])

    def test_malformed_ledger_fails_closed_without_modifying_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "market_read_predictions.jsonl"
            original = b'{"prediction_id":"ok"}\nnot-json\n'
            path.write_bytes(original)

            result = self._record(root)

            self.assertEqual(result["status"], "MARKET_READ_LEDGER_INVALID")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
