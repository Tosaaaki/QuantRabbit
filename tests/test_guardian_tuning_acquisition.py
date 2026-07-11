from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.capture_economics import AttributedEntry, RealizedOutcome
from quant_rabbit import guardian_tuning_acquisition as acquisition


REVIEWED_AT = datetime(2026, 7, 11, tzinfo=timezone.utc)
LANE = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"


def _work_order(
    *,
    exact_lane: bool = True,
    event_type: str = "TECHNICAL_STATE_CHANGE",
    family: str = "trend",
) -> dict:
    selected_event = {
        "event_type": event_type,
        "pair": "EUR_USD",
        "direction": "LONG",
        "details": {},
    }
    if exact_lane:
        selected_event["details"]["lane_id"] = LANE
    return {
        "work_order_id": "guardian-tuning-test",
        "status": "PENDING_HOURLY_AI_REVIEW",
        "latest_observation_id": "observation-current",
        "latest_reviewed_observation_id": "observation-current",
        "structured_review_completed_at_utc": REVIEWED_AT.isoformat(),
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
        "selected_event": selected_event,
        "bot_tuning_review_validation": {"status": "VALID", "issues": []},
        "bot_tuning_review": {
            "review_status": "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            "affected_pairs": ["EUR_USD"],
            "affected_bot_families": [family],
            "hypothesis": "collect immutable forward signal state",
            "falsifiable_experiment": "freeze the first twenty attributed entries",
            "proposed_adjustments": [],
            "evidence_acquisition": {
                "action_kind": "ADD_PREENTRY_SIGNAL_LOG",
                "source_ref": "data/entry_thesis_ledger.jsonl",
                "required_new_samples": 20,
                "success_condition": "store one exact pre-entry state and resolve the same first twenty entries",
            },
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        },
    }


def _entries(count: int, *, lane: str = LANE) -> list[AttributedEntry]:
    result: list[AttributedEntry] = []
    for index in range(count):
        opened_at = REVIEWED_AT + timedelta(minutes=index + 1)
        result.append(
            AttributedEntry(
                trade_id=f"trade-{index}",
                order_id=f"order-{index}",
                pair="EUR_USD",
                side="LONG",
                lane_id=lane.rsplit(":", 1)[0],
                canonical_lane_id=lane,
                method="TREND_CONTINUATION",
                entry_vehicle="LIMIT",
                entry_ts_utc=opened_at.isoformat(),
                entry_units=1000.0,
                ledger_rowid=index + 1,
                broker_entry_ts_utc=opened_at.isoformat(),
                broker_time_consistent=True,
            )
        )
    return result


def _outcomes(entries: list[AttributedEntry]) -> list[RealizedOutcome]:
    result: list[RealizedOutcome] = []
    for entry in entries:
        result.append(
            RealizedOutcome(
                ts_utc=(
                    datetime.fromisoformat(entry.entry_ts_utc) + timedelta(minutes=30)
                ).isoformat(),
                trade_id=entry.trade_id,
                pair=entry.pair,
                side=entry.side,
                lane_id=entry.canonical_lane_id,
                method=entry.method,
                exit_reason="TAKE_PROFIT_ORDER",
                realized_pl_jpy=100.0,
                entry_vehicle=entry.entry_vehicle,
                entry_truth_consistent=True,
                broker_close_ts_utc=(
                    datetime.fromisoformat(entry.entry_ts_utc) + timedelta(minutes=30)
                ).isoformat(),
                broker_time_consistent=True,
            )
        )
    return result


def _thesis(
    entry: AttributedEntry,
    *,
    context: dict | None = None,
) -> dict:
    return {
        "timestamp_utc": entry.broker_entry_ts_utc,
        "trade_id": entry.trade_id,
        "pair": entry.pair,
        "side": entry.side,
        "context_evidence": (
            context
            if context is not None
            else {
                "order_id": entry.order_id,
                "lane_id": entry.canonical_lane_id,
                "guardian_tuning_signal_state": {
                    "m5_regime": "TREND",
                    "m5_trend_score": 0.8,
                },
            }
        ),
    }


def _write_theses(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GuardianTuningAcquisitionProgressTest(unittest.TestCase):
    def test_progress_moves_from_zero_to_nineteen_to_ready_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            ledger.touch()
            thesis_path = root / "entry_thesis_ledger.jsonl"

            cases = (
                (0, "WAITING_FOR_ENTRIES", 20),
                (19, "WAITING_FOR_ENTRIES", 1),
                (20, "READY_FOR_GPT_REVIEW_UPGRADE", 0),
            )
            for count, expected_status, expected_remaining in cases:
                with self.subTest(count=count):
                    entries = _entries(count)
                    _write_theses(thesis_path, [_thesis(entry) for entry in entries])
                    with (
                        patch.object(
                            acquisition,
                            "read_attributed_system_entries",
                            return_value=entries,
                        ),
                        patch.object(
                            acquisition,
                            "read_attributed_net_outcomes",
                            return_value=_outcomes(entries),
                        ),
                    ):
                        payload = acquisition.build_guardian_tuning_acquisition_progress(
                            [_work_order()],
                            entry_thesis_path=thesis_path,
                            ledger_path=ledger,
                        )

                    progress = payload["work_orders"][0]
                    self.assertEqual(progress["status"], expected_status)
                    self.assertEqual(progress["entry_count"], count)
                    self.assertEqual(progress["complete_count"], count)
                    self.assertEqual(
                        progress["remaining_complete_count"],
                        expected_remaining,
                    )
                    self.assertNotIn("realized_pl_jpy", json.dumps(payload))
                    self.assertNotIn("permission", json.dumps(payload).lower())

    def test_first_twenty_are_frozen_and_later_resolution_cannot_replace_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            ledger.touch()
            thesis_path = root / "entry_thesis_ledger.jsonl"
            entries = _entries(21)
            _write_theses(thesis_path, [_thesis(entry) for entry in entries])
            resolved = _outcomes(entries[1:])

            with (
                patch.object(
                    acquisition,
                    "read_attributed_system_entries",
                    return_value=entries,
                ),
                patch.object(
                    acquisition,
                    "read_attributed_net_outcomes",
                    return_value=resolved,
                ),
            ):
                payload = acquisition.build_guardian_tuning_acquisition_progress(
                    [_work_order(exact_lane=False)],
                    entry_thesis_path=thesis_path,
                    ledger_path=ledger,
                )

            progress = payload["work_orders"][0]
            self.assertEqual(progress["status"], "WAITING_FOR_RESOLUTION")
            self.assertEqual(progress["entry_count"], 20)
            self.assertEqual(progress["resolved_count"], 19)
            self.assertEqual(progress["remaining_resolution_count"], 1)
            self.assertEqual(
                progress["first_trade_ids"],
                [f"trade-{index}" for index in range(20)],
            )
            self.assertNotIn("trade-20", progress["first_trade_ids"])

    def test_missing_exact_links_and_failed_acceptance_state_are_defects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            ledger.touch()
            thesis_path = root / "entry_thesis_ledger.jsonl"
            failure_lane = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            entries = _entries(20, lane=failure_lane)
            rows = []
            for index, entry in enumerate(entries):
                context = {
                    "order_id": entry.order_id,
                    "lane_id": entry.canonical_lane_id,
                    "guardian_tuning_signal_state": {
                        "failed_acceptance": True,
                        "acceptance_zone": 1.14,
                    },
                }
                if index == 0:
                    context.pop("order_id")
                elif index == 1:
                    context.pop("guardian_tuning_signal_state")
                elif index == 2:
                    context["guardian_tuning_signal_state"] = {
                        "failed_acceptance": False,
                    }
                rows.append(_thesis(entry, context=context))
            _write_theses(thesis_path, rows)
            work_order = _work_order(event_type="FAILED_ACCEPTANCE", family="breakout")
            work_order["selected_event"]["details"]["lane_id"] = failure_lane

            with (
                patch.object(
                    acquisition,
                    "read_attributed_system_entries",
                    return_value=entries,
                ),
                patch.object(
                    acquisition,
                    "read_attributed_net_outcomes",
                    return_value=_outcomes(entries),
                ),
            ):
                payload = acquisition.build_guardian_tuning_acquisition_progress(
                    [work_order],
                    entry_thesis_path=thesis_path,
                    ledger_path=ledger,
                )

            progress = payload["work_orders"][0]
            self.assertEqual(progress["status"], "ACQUISITION_SOURCE_DEFECT")
            self.assertEqual(progress["preentry_complete_count"], 17)
            defects = {
                item["trade_id"]: set(item["codes"])
                for item in progress["signal_defects"]
            }
            self.assertIn("ENTRY_THESIS_ORDER_ID_MISMATCH", defects["trade-0"])
            self.assertIn(
                "GUARDIAN_TUNING_SIGNAL_STATE_MISSING",
                defects["trade-1"],
            )
            self.assertEqual(
                defects["trade-2"],
                {
                    "FAILED_ACCEPTANCE_PREDICATE_MISSING",
                    "FAILED_ACCEPTANCE_ZONE_MISSING",
                },
            )

    def test_technical_state_rejects_nontechnical_nonfamily_signal_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            ledger.touch()
            thesis_path = root / "entry_thesis_ledger.jsonl"
            entries = _entries(1)
            _write_theses(
                thesis_path,
                [
                    _thesis(
                        entries[0],
                        context={
                            "order_id": entries[0].order_id,
                            "lane_id": entries[0].canonical_lane_id,
                            "guardian_tuning_signal_state": {
                                "desk": "trend_trader"
                            },
                        },
                    )
                ],
            )

            with (
                patch.object(
                    acquisition,
                    "read_attributed_system_entries",
                    return_value=entries,
                ),
                patch.object(
                    acquisition,
                    "read_attributed_net_outcomes",
                    return_value=_outcomes(entries),
                ),
            ):
                payload = acquisition.build_guardian_tuning_acquisition_progress(
                    [_work_order()],
                    entry_thesis_path=thesis_path,
                    ledger_path=ledger,
                )

            progress = payload["work_orders"][0]
            self.assertEqual(progress["status"], "COLLECTING_WITH_SIGNAL_DEFECT")
            self.assertEqual(progress["preentry_complete_count"], 0)
            self.assertEqual(
                set(progress["signal_defects"][0]["codes"]),
                {
                    "TECHNICAL_REGIME_STATE_MISSING",
                    "TREND_FAMILY_SIGNAL_MISSING",
                },
            )

    def test_required_temp_paths_are_used_and_malformed_temp_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "only-this-ledger.db"
            ledger.touch()
            thesis_path = root / "only-this-thesis.jsonl"
            thesis_path.write_text("{not-json}\n", encoding="utf-8")

            with (
                patch.object(
                    acquisition,
                    "read_attributed_system_entries",
                    return_value=[],
                ) as entry_reader,
                patch.object(
                    acquisition,
                    "read_attributed_net_outcomes",
                    return_value=[],
                ) as outcome_reader,
            ):
                payload = acquisition.build_guardian_tuning_acquisition_progress(
                    [_work_order()],
                    entry_thesis_path=thesis_path,
                    ledger_path=ledger,
                )

            entry_reader.assert_called_once_with(ledger)
            outcome_reader.assert_called_once_with(ledger)
            self.assertEqual(payload["status"], "SOURCE_DEFECT")
            self.assertEqual(
                payload["work_orders"][0]["source_issue_codes"],
                ["ENTRY_THESIS_SOURCE_UNREADABLE"],
            )
            with self.assertRaises(TypeError):
                acquisition.build_guardian_tuning_acquisition_progress(
                    [_work_order()]
                )


if __name__ == "__main__":
    unittest.main()
