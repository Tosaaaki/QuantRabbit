from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tools.guardian_wake_dispatcher as dispatcher_module
import tools.guardian_tuning_metric_evaluator as tuning_evaluator_module
import tools.guardian_tuning_review_enrich as tuning_review_enrich_tool
from quant_rabbit.guardian_tuning_overrides import (
    resolve_forecast_confidence_floor_state,
)
from tools.guardian_wake_dispatcher import (
    DispatcherPaths,
    _resolve_codex_bin,
    _select_dispatch_event,
    run_dispatcher,
)


NOW = datetime(2026, 6, 30, 4, 0, tzinfo=timezone.utc)


class GuardianWakeDispatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        # Lifecycle unit tests exercise queue/evidence binding. Canonical
        # ledger/log reconstruction is covered by test_guardian_tuning_cohort_builder.
        canonical = patch.object(
            dispatcher_module,
            "validate_canonical_forward_cohort",
            return_value={"status": "VALID"},
        )
        canonical.start()
        self.addCleanup(canonical.stop)
        current_tip = patch.object(
            dispatcher_module,
            "current_canonical_forward_source_tip",
            return_value={
                "last_oanda_transaction_id": "473024",
                "ledger_rowid_watermark": 500,
                "ledger_prefix_sha256": "a" * 64,
                "entry_thesis_prefix_bytes": 100,
                "entry_thesis_prefix_sha256": "c" * 64,
                "forecast_history_prefix_bytes": 200,
                "forecast_history_prefix_sha256": "d" * 64,
            },
        )
        current_tip.start()
        self.addCleanup(current_tip.stop)
        activation_anchor = patch.object(
            dispatcher_module,
            "current_execution_ledger_anchor",
            return_value={
                "ledger_rowid_watermark": 500,
                "ledger_prefix_sha256": "a" * 64,
                "execution_ledger_coverage_start_utc": "2026-05-06T16:52:01+00:00",
                "last_oanda_transaction_id": "473024",
                "captured_at_utc": NOW.isoformat(),
            },
        )
        activation_anchor.start()
        self.addCleanup(activation_anchor.stop)
        monitor_gate = patch(
            "quant_rabbit.guardian_tuning_cohort.build_post_activation_monitor_cohort",
            return_value={
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_ENTRIES",
                "entry_count": 0,
                "required_entry_count": 20,
            },
        )
        monitor_gate.start()
        self.addCleanup(monitor_gate.stop)

    def test_default_codex_bin_prefers_current_desktop_app_over_path_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_bin = Path(tmp) / "ChatGPT.app/Contents/Resources/codex"
            app_bin.parent.mkdir(parents=True)
            app_bin.write_text("desktop cli\n")
            with (
                patch("tools.guardian_wake_dispatcher.DEFAULT_CODEX_APP_BIN", app_bin),
                patch(
                    "tools.guardian_wake_dispatcher.LEGACY_CODEX_APP_BIN",
                    Path(tmp) / "missing-codex-app",
                ),
                patch(
                    "tools.guardian_wake_dispatcher.shutil.which",
                    return_value="/opt/homebrew/bin/codex",
                ),
            ):
                resolved = _resolve_codex_bin({})

            self.assertEqual(resolved, str(app_bin))

    def test_gpt_json_only_prompt_contains_required_schema(self) -> None:
        prompt = (Path(__file__).resolve().parents[1] / "docs" / "guardian_wake_prompt.md").read_text()

        self.assertIn("Return JSON only", prompt)
        self.assertIn('"action": "TRADE|ADD|HOLD|HARVEST|REDUCE|CANCEL_PENDING|NO_ACTION"', prompt)
        self.assertIn('"new_information": true', prompt)
        self.assertIn('"thesis_state": "ALIVE|WOUNDED|INVALIDATED|EMERGENCY"', prompt)
        self.assertIn('"ownership": "SYSTEM|OPERATOR_MANUAL|UNKNOWN"', prompt)
        self.assertIn('"gateway_required": true', prompt)
        self.assertIn('"no_direct_oanda": true', prompt)
        self.assertIn('"bot_tuning_review"', prompt)
        self.assertIn('"falsifiable_experiment"', prompt)
        self.assertIn('"evidence_acquisition"', prompt)
        self.assertIn('"action_kind"', prompt)
        self.assertIn('"source_ref"', prompt)
        self.assertIn('"required_new_samples"', prompt)
        self.assertIn('"success_condition"', prompt)
        for action_kind in dispatcher_module._TUNING_ACQUISITION_ACTION_KINDS:
            self.assertIn(action_kind, prompt)
        self.assertIn("data/entry_thesis_ledger.jsonl", prompt)
        for retired_action_kind in (
            "BUILD_BID_ASK_REPLAY",
            "COLLECT_FORWARD_ENTRIES",
            "REFRESH_CLOSED_CANDLES",
            "RESOLVE_ATTRIBUTED_OUTCOMES",
        ):
            self.assertNotIn(retired_action_kind, prompt)
        self.assertIn('"live_permission_allowed": false', prompt)

    def test_prompt_no_change_tuning_review_matches_validator_contract(self) -> None:
        selected_event = _technical_tuning_event(pair="EUR_USD")
        prompt_no_change_review = {
            "review_status": "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            "affected_pairs": ["EUR_USD"],
            "affected_bot_families": ["trend"],
            "hypothesis": (
                "the current state change lacks enough pre-entry observations "
                "to precommit a tighter forecast floor"
            ),
            "falsifiable_experiment": (
                "collect one bounded forward cohort and compare its recorded "
                "entry-time forecast confidence with canonical outcomes"
            ),
            "evidence_acquisition": {
                "action_kind": "ADD_PREENTRY_SIGNAL_LOG",
                "source_ref": "data/entry_thesis_ledger.jsonl",
                "required_new_samples": 20,
                "success_condition": (
                    "resolve the first 20 canonical attributed post-review entries"
                ),
            },
            "proposed_adjustments": [],
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }

        result = dispatcher_module._validate_bot_tuning_review(
            prompt_no_change_review,
            selected_event=selected_event,
        )

        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["review"], prompt_no_change_review)

    def test_wake_false_does_not_start_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), wake=False)
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "NO_WAKE")
            self.assertEqual(calls, [])
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())
            self.assertIn("Receipt exists: `no`", paths.action_review.read_text())

    def test_accepted_hold_receipt_survives_later_no_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )
            paths.escalation.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "wake_gpt": False}))

            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="TRADE")),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(second["status"], "NO_WAKE")
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `NO_WAKE`", review)
            self.assertIn("Latest Accepted Receipt", review)
            self.assertIn("Action: `HOLD`", review)

    def test_accepted_no_action_receipt_survives_later_suppressed_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="NO_ACTION")),
            )

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="TRADE")),
            )

            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(len(calls), 1)
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "NO_ACTION")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `SUPPRESSED`", review)
            self.assertIn("Action: `NO_ACTION`", review)

    def test_wake_true_starts_codex_exec_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][0]).name, "codex")
            self.assertEqual(calls[0][1:3], ["exec", "-"])
            self.assertTrue(paths.action_receipt.exists())

    def test_no_action_tuning_event_waits_for_hourly_ai_without_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            event = _technical_tuning_event(pair="CAD_CHF")
            paths.events.write_text(json.dumps({"events": [event]}))
            paths.event_state.write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "events": {
                            event["dedupe_key"]: {
                                **event,
                                "last_seen_at_utc": NOW.isoformat(),
                            }
                        },
                    }
                )
            )
            paths.escalation.write_text(
                json.dumps(
                    {
                        "wake_gpt": True,
                        "events_to_review": [event],
                    }
                )
            )
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_TUNING_MODE": "HOURLY"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "QUEUED_FOR_HOURLY_TUNING")
            self.assertFalse(result["wake_gpt"])
            self.assertTrue(result["queued_for_hourly_tuning"])
            self.assertEqual(calls, [])
            self.assertFalse(paths.action_receipt.exists())
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertIn(event["dedupe_key"], state["hourly_routed_events"])
            self.assertEqual(state.get("pending_dispatches"), {})

            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env={"QR_GUARDIAN_WAKE_TUNING_MODE": "HOURLY"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(second["status"], "SUPPRESSED")
            self.assertEqual(calls, [])

    def test_hourly_tuning_row_does_not_starve_entry_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            tuning = _technical_tuning_event(pair="CAD_CHF")
            entry = _event(
                severity="P2",
                event_id="entry-after-tuning",
                pair="EUR_USD",
                direction="LONG",
            )
            paths.events.write_text(
                json.dumps({"events": [tuning, entry]})
            )
            _write_technical_state(paths, "CAD_CHF", "EUR_USD")
            paths.escalation.write_text(
                json.dumps(
                    {
                        "wake_gpt": True,
                        "events_to_review": [
                            tuning,
                            {**entry, "wake_reason_codes": ["NEW_EVENT"]},
                        ],
                    }
                )
            )
            calls: list[list[str]] = []
            response = _valid_receipt(
                event_id=entry["event_id"],
                pair=entry["pair"],
                side=entry["direction"],
                dedupe_key=entry["dedupe_key"],
            )

            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_TUNING_MODE": "HOURLY"},
                subprocess_run=_fake_codex(calls, response),
            )
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env={"QR_GUARDIAN_WAKE_TUNING_MODE": "HOURLY"},
                subprocess_run=_fake_codex(calls, response),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(first["selected_event"]["event_id"], entry["event_id"])
            self.assertEqual(second["status"], "QUEUED_FOR_HOURLY_TUNING")
            self.assertEqual(second["selected_event"]["event_id"], tuning["event_id"])
            self.assertEqual(len(calls), 1)

    def test_runtime_disk_p0_queues_without_starting_codex_or_marking_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            gib = 1024**3

            with patch(
                "tools.guardian_wake_dispatcher.shutil.disk_usage",
                return_value=SimpleNamespace(total=100 * gib, used=99 * gib, free=1 * gib),
            ):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={},
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
                )

            self.assertEqual(result["status"], "RUNTIME_DISK_P0")
            self.assertTrue(result["queued_for_active_trader"])
            self.assertEqual(calls, [])
            self.assertFalse(paths.codex_home.exists())
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertNotIn("reviewed_events", state)
            attempt = state["dispatch_attempts"][_event(severity="P1")["dedupe_key"]]
            self.assertEqual(attempt["last_status"], "RUNTIME_DISK_P0")
            self.assertEqual(attempt["runtime_disk"]["free_bytes"], gib)

    def test_runtime_disk_warning_is_in_result_and_gpt_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            prompts: list[str] = []
            gib = 1024**3

            with patch(
                "tools.guardian_wake_dispatcher.shutil.disk_usage",
                return_value=SimpleNamespace(total=100 * gib, used=97 * gib, free=3 * gib),
            ):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={},
                    subprocess_run=_fake_codex_with_prompt(
                        calls,
                        prompts,
                        _valid_receipt(),
                    ),
                )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["runtime_disk"]["status"], "RUNTIME_DISK_WARNING")
            self.assertEqual(len(prompts), 1)
            self.assertIn('"status": "RUNTIME_DISK_WARNING"', prompts[0])

    def test_prompt_contains_only_authoritative_selected_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            selected = _event(severity="P1")
            selected["details"] = {
                "source_event": {
                    "event_id": "nested-source-event-must-not-leak",
                    "dedupe_key": "EUR_USD|nested_source|FAILED_ACCEPTANCE|HOLD",
                    "pair": "EUR_USD",
                    "event_type": "FAILED_ACCEPTANCE",
                    "price_zone": "prior observation retained as evidence",
                }
            }
            selected["direction"] = None
            sibling = _event(
                severity="P2",
                event_id="event-sibling-must-not-leak",
                pair="NZD_CHF",
                dedupe_key="NZD_CHF|other_pending|FAILED_ACCEPTANCE|TRADE",
            )
            paths.events.write_text(json.dumps({"events": [selected, sibling]}))
            paths.event_state.write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "events": {
                            selected["dedupe_key"]: {
                                "event_id": "state-event-must-not-leak",
                                "event_type": "TECHNICAL_STATE_CHANGE",
                                "pair": "EUR_USD",
                                "last_seen_at_utc": NOW.isoformat(),
                                "details": {"mid": 1.1422},
                            },
                            sibling["dedupe_key"]: {"event_id": sibling["event_id"]},
                        }
                    }
                )
            )
            paths.events.write_text(
                json.dumps(
                    {
                        "events": [selected, sibling],
                        "trigger_contract": {
                            "status": "INVALID",
                            "issues": [
                                {
                                    "code": "CONTRACT_ENTRY_DEADLINE_EXPIRED",
                                    "severity": "BLOCK",
                                    "message": "NZD_CHF sibling contract must not leak",
                                }
                            ],
                        },
                    }
                )
            )
            paths.daily_target_state.write_text(
                json.dumps(
                    {
                        "status": "OPEN",
                        "current_equity_jpy": 280_000,
                        "positions": [
                            {
                                "pair": "NZD_CHF",
                                "side": "LONG",
                                "trade_id": "sibling-position-must-not-leak",
                            }
                        ],
                        "unprotected_positions": 1,
                    }
                )
            )
            paths.event_report.write_text(
                "\n".join(
                    [
                        "# Guardian Event Report",
                        "",
                        "## Events",
                        "",
                        "- `P1` `FAILED_ACCEPTANCE` `EUR_USD` `LONG`",
                        f"  - review: `ENTRY_REVIEW` dedupe: `{selected['dedupe_key']}`",
                        "  - report_mid: `1.142200` must not replace immutable retry observation",
                        "- `P2` `FAILED_ACCEPTANCE` `NZD_CHF` `LONG`",
                        f"  - review: `ENTRY_REVIEW` dedupe: `{sibling['dedupe_key']}`",
                    ]
                )
                + "\n"
            )
            paths.escalation.write_text(
                json.dumps(
                    {
                        "wake_gpt": True,
                        "wake_reason_codes": ["NEW_EVENT", "SIBLING_HARVEST_MUST_NOT_LEAK"],
                        "events_to_review": [
                            {**selected, "wake_reason_codes": ["NEW_EVENT"]},
                            {**sibling, "wake_reason_codes": ["NEW_EVENT"]},
                        ],
                    }
                )
            )
            paths.dispatcher_state.write_text(
                json.dumps(
                    {
                        "dispatch_attempts": {
                            selected["dedupe_key"]: {
                                "event_id": selected["event_id"],
                                "attempt_count": 0,
                                "max_attempts": 3,
                                "last_status": "CODEX_USAGE_LIMIT_RECOVERED",
                                "usage_limit_recovery_source_dedupe_key": (
                                    "USD_CAD|sibling_quota_recovery|SPREAD_ANOMALY|HOLD"
                                ),
                            }
                        },
                        "pending_dispatches": {
                            sibling["dedupe_key"]: {
                                "event": sibling,
                                "queued_at_utc": NOW.isoformat(),
                                "expires_at_utc": (NOW + timedelta(hours=1)).isoformat(),
                            }
                        },
                        "last_result": {"selected_event": sibling},
                    }
                )
            )
            broker_snapshot = json.loads(paths.broker_snapshot.read_text())
            broker_snapshot["quotes"]["USD_JPY"] = {"bid": 160.0, "ask": 160.01}
            paths.broker_snapshot.write_text(json.dumps(broker_snapshot))
            calls: list[list[str]] = []
            prompts: list[str] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_with_prompt(
                    calls,
                    prompts,
                    _valid_receipt(action="HOLD", side="NONE"),
                ),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(prompts), 1)
            self.assertIn("event_id=event-P1 pair=EUR_USD", prompts[0])
            self.assertIn("Set side exactly to NONE", prompts[0])
            self.assertNotIn("event-sibling-must-not-leak", prompts[0])
            self.assertNotIn("NZD_CHF", prompts[0])
            self.assertNotIn("USD_CAD", prompts[0])
            self.assertNotIn("USD_JPY", prompts[0])
            self.assertNotIn("SIBLING_HARVEST_MUST_NOT_LEAK", prompts[0])
            self.assertNotIn("nested-source-event-must-not-leak", prompts[0])
            self.assertNotIn("state-event-must-not-leak", prompts[0])
            self.assertNotIn("report_mid", prompts[0])

    def test_dispatcher_lock_prevents_shared_output_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            with patch(
                "tools.guardian_wake_dispatcher.fcntl.flock",
                side_effect=BlockingIOError(),
            ):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={},
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
                )

            self.assertEqual(result["status"], "DISPATCHER_LOCKED")
            self.assertEqual(calls, [])
            self.assertFalse(paths.codex_output.exists())
            self.assertFalse(paths.action_receipt.exists())
            self.assertFalse(paths.dispatcher_state.exists())

    def test_runtime_disk_recovery_retries_same_event_without_waiting_for_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            gib = 1024**3
            disk_checks = 0

            def fake_disk_usage(_path):
                nonlocal disk_checks
                disk_checks += 1
                free = 1 * gib if disk_checks == 1 else 8 * gib
                return SimpleNamespace(total=100 * gib, used=100 * gib - free, free=free)

            with patch(
                "tools.guardian_wake_dispatcher.shutil.disk_usage",
                side_effect=fake_disk_usage,
            ):
                first = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={},
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
                )
                recovered = run_dispatcher(
                    paths=paths,
                    now=NOW + timedelta(seconds=10),
                    env={},
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
                )

            self.assertEqual(first["status"], "RUNTIME_DISK_P0")
            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 1)
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(state["dispatch_attempts"], {})

    def test_active_qr_trader_lock_queues_instead_of_starting_second_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.live_lock.mkdir(parents=True)
            (paths.live_lock / "pid").write_text(str(os.getpid()))
            (paths.live_lock / "command").write_text("run-autotrade-live")
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "QUEUED_FOR_ACTIVE_TRADER")
            self.assertEqual(calls, [])
            escalation = json.loads(paths.escalation.read_text())
            self.assertTrue(escalation["queued_for_active_trader"])
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertNotIn("reviewed_events", state)
            self.assertNotIn("dispatch_attempts", state)
            self.assertIn(_event(severity="P1")["dedupe_key"], state["pending_dispatches"])

            for child in paths.live_lock.iterdir():
                child.unlink()
            paths.live_lock.rmdir()
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            paths.events.write_text(json.dumps({"events": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 1)
            recovered_state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(recovered_state["pending_dispatches"], {})

    def test_duplicate_wake_is_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(len(calls), 1)
            reasons = {item["reason"] for item in result["selection"]["suppressed"]}
            self.assertIn("THROTTLED", reasons)
            self.assertTrue(paths.action_receipt.exists())

    def test_legacy_failed_review_record_does_not_suppress_same_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            event = _event(severity="P1")
            paths.dispatcher_state.write_text(
                json.dumps(
                    {
                        "reviewed_events": {
                            event["dedupe_key"]: {
                                "event_id": event["event_id"],
                                "severity": "P1",
                                "last_reviewed_at_utc": NOW.isoformat(),
                                "last_status": "CODEX_MODEL_UNSUPPORTED",
                                "receipt_written": False,
                            }
                        }
                    }
                )
            )
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=10),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertTrue(state["reviewed_events"][event["dedupe_key"]]["receipt_written"])

    def test_p0_severity_increase_can_bypass_throttle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root, severity="P1", reasons=("NEW_EVENT",))
            calls: list[list[str]] = []
            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))
            _write_wake(paths, severity="P0", reasons=("SEVERITY_INCREASE",))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P0")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)

    def test_material_equal_severity_event_dispatches_after_success_throttle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            env = {"QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS": "60"}
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env=env,
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )
            moved = {
                **_event(severity="P1", event_id="event-moved"),
                "price_zone": "EUR_USD 1.1715 rejection",
                "wake_reason_codes": [
                    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
                    "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
                ],
            }
            paths.events.write_text(json.dumps({"events": [moved]}))
            paths.escalation.write_text(
                json.dumps({"wake_gpt": True, "events_to_review": [moved]})
            )

            throttled = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env=env,
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id="event-moved", tuning_review=True),
                ),
            )
            retried = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=61),
                env=env,
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id="event-moved", tuning_review=True),
                ),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(throttled["status"], "SUPPRESSED")
            self.assertEqual(retried["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)
            state = json.loads(paths.dispatcher_state.read_text())
            baseline = state["reviewed_events"][moved["dedupe_key"]]
            self.assertEqual(baseline["event_id"], "event-moved")
            self.assertEqual(baseline["price_zone"], moved["price_zone"])
            self.assertIn("material_fingerprint", baseline)

    def test_state_change_reason_families_are_dispatchable(self) -> None:
        reasons = (
            "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
            "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
            "THESIS_ALIVE_TO_WOUNDED",
            "PRICE_ENTERED_HARVEST_ZONE",
            "TECHNICAL_STATE_CHANGE_TREND_FLIP",
            "REGIME_SHIFT_TO_RANGE",
            "VOLATILITY_BUCKET_CHANGE",
            "FAMILY_DISAGREEMENT_STATE_CHANGE",
            "CLOSED_CANDLE_STRUCTURE_CHANGE",
        )
        for reason in reasons:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), reasons=(reason,))
                calls: list[list[str]] = []

                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={},
                    subprocess_run=_fake_codex(
                        calls,
                        _valid_receipt(
                            tuning_review=dispatcher_module._is_tuning_handoff_reason(reason)
                        ),
                    ),
                )

                self.assertEqual(result["status"], "RECEIPT_WRITTEN")
                self.assertEqual(len(calls), 1)

    def test_priority_prefers_open_exposure_then_p0_then_material(self) -> None:
        open_event = {
            **_event(
                severity="P1",
                event_id="open-event",
                pair="ZZZ_USD",
                dedupe_key="open-key",
            ),
            "event_type": "HARVEST_ZONE",
            "action_hint": "HARVEST",
            "wake_reason_codes": ["PRICE_ENTERED_HARVEST_ZONE"],
            "details": {"trade_id": "123", "units": 1000},
        }
        p0_event = {
            **_event(
                severity="P0",
                event_id="p0-event",
                pair="AAA_USD",
                dedupe_key="p0-key",
            ),
            "wake_reason_codes": ["NEW_EVENT"],
        }
        material_event = {
            **_event(
                severity="P1",
                event_id="material-event",
                pair="BBB_USD",
                dedupe_key="material-key",
            ),
            "wake_reason_codes": ["TECHNICAL_STATE_CHANGE_TREND_FLIP"],
        }

        selected, selection = _select_dispatch_event(
            escalation={"events_to_review": [material_event, p0_event, open_event]},
            events_payload={"events": [material_event, p0_event, open_event]},
            dispatcher_state={},
            now=NOW,
            env={},
        )

        self.assertEqual(selected["event_id"], "open-event")
        self.assertTrue(selection["selected_priority"]["open_exposure"])

    def test_candidate_harvest_is_not_classified_as_open_exposure(self) -> None:
        candidate = {
            **_event(
                severity="P1",
                event_id="candidate-harvest",
                pair="EUR_USD",
                dedupe_key="candidate-harvest-key",
            ),
            "event_type": "HARVEST_ZONE",
            "action_hint": "HARVEST",
            "details": {
                "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                "status": "LIVE_READY",
            },
        }
        open_position = {
            **candidate,
            "event_id": "open-position-harvest",
            "details": {**candidate["details"], "trade_id": "t1"},
        }

        self.assertFalse(dispatcher_module._event_has_open_exposure(candidate))
        self.assertFalse(
            dispatcher_module._dispatch_priority_evidence(candidate)[
                "open_exposure"
            ]
        )
        self.assertTrue(dispatcher_module._event_has_open_exposure(open_position))
        self.assertTrue(
            dispatcher_module._dispatch_priority_evidence(open_position)[
                "open_exposure"
            ]
        )

    def test_retained_stale_pair_suppresses_pending_entry_dispatch(self) -> None:
        entry = {
            **_event(
                severity="P1",
                event_id="pending-entry",
                pair="EUR_USD",
                dedupe_key="pending-entry-key",
            ),
            "wake_reason_codes": ["DURABLE_PENDING_DISPATCH"],
        }

        selected, selection = _select_dispatch_event(
            escalation={"events_to_review": [entry]},
            events_payload={
                "generated_at_utc": NOW.isoformat(),
                "events": [entry],
            },
            dispatcher_state={},
            now=NOW,
            env={},
            event_state={
                "generated_at_utc": NOW.isoformat(),
                "events": {
                    "EUR_USD|stale": {
                        "event_type": "TECHNICAL_INPUT_STALE",
                        "pair": "EUR_USD",
                        "last_seen_at_utc": (NOW - timedelta(minutes=20)).isoformat(),
                    }
                },
            },
        )

        self.assertIsNone(selected)
        self.assertEqual(selection["status"], "NO_DISPATCHABLE_EVENT")
        self.assertEqual(selection["suppressed"][0]["reason"], "TECHNICAL_INPUT_BLOCKED")

    def test_missing_or_old_technical_baseline_suppresses_entry_dispatch(self) -> None:
        entry = {
            **_event(
                severity="P1",
                event_id="entry-without-state",
                pair="EUR_USD",
                dedupe_key="entry-without-state-key",
            ),
            "wake_reason_codes": ["NEW_EVENT"],
        }
        for events_generated_at, event_state in (
            (None, {}),
            (
                (NOW - timedelta(minutes=10)).isoformat(),
                {
                    "generated_at_utc": NOW.isoformat(),
                    "events": {
                        "EUR_USD|technical": {
                            "event_type": "TECHNICAL_STATE_CHANGE",
                            "pair": "EUR_USD",
                            "last_seen_at_utc": (NOW - timedelta(minutes=10)).isoformat(),
                        }
                    },
                },
            ),
        ):
            with self.subTest(events_generated_at=events_generated_at):
                events_payload = {"events": [entry]}
                if events_generated_at is not None:
                    events_payload["generated_at_utc"] = events_generated_at
                selected, selection = _select_dispatch_event(
                    escalation={"events_to_review": [entry]},
                    events_payload=events_payload,
                    dispatcher_state={},
                    now=NOW,
                    env={},
                    event_state=event_state,
                )

                self.assertIsNone(selected)
                self.assertEqual(
                    selection["suppressed"][0]["reason"],
                    "GUARDIAN_TECHNICAL_STATE_UNAVAILABLE",
                )

    def test_multiple_review_events_are_durably_drained_across_router_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            first_event = {
                **_event(
                    severity="P0",
                    event_id="event-first",
                    pair="AAA_USD",
                    dedupe_key="AAA_USD|first|FAILED_ACCEPTANCE|TRADE",
                ),
                "wake_reason_codes": ["NEW_EVENT"],
            }
            second_event = {
                **_event(
                    severity="P1",
                    event_id="event-second",
                    pair="BBB_USD",
                    dedupe_key="BBB_USD|second|FAILED_ACCEPTANCE|TRADE",
                ),
                "wake_reason_codes": ["NEW_EVENT"],
            }
            paths.events.write_text(json.dumps({"events": [first_event, second_event]}))
            _write_technical_state(paths, "AAA_USD", "BBB_USD")
            paths.escalation.write_text(
                json.dumps(
                    {
                        "wake_gpt": True,
                        "events_to_review": [second_event, first_event],
                    }
                )
            )
            calls: list[list[str]] = []

            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(
                        event_id="event-first",
                        pair="AAA_USD",
                        dedupe_key=first_event["dedupe_key"],
                    ),
                ),
            )
            state_after_first = json.loads(paths.dispatcher_state.read_text())
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            paths.events.write_text(json.dumps({"events": []}))
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(
                        event_id="event-second",
                        pair="BBB_USD",
                        dedupe_key=second_event["dedupe_key"],
                    ),
                ),
            )

            self.assertEqual(first["selected_event"]["event_id"], "event-first")
            self.assertIn(second_event["dedupe_key"], state_after_first["pending_dispatches"])
            self.assertEqual(second["selected_event"]["event_id"], "event-second")
            self.assertEqual(json.loads(paths.dispatcher_state.read_text())["pending_dispatches"], {})
            self.assertEqual(len(calls), 2)

    def test_codex_command_uses_gpt55_and_read_only_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            cmd = calls[0]
            self.assertEqual(cmd[cmd.index("-m") + 1], "gpt-5.5")
            self.assertEqual(cmd[cmd.index("-s") + 1], "read-only")
            self.assertEqual(cmd[cmd.index("-C") + 1], str(paths.live_root))
            self.assertIn('model_reasoning_effort = "high"', (paths.codex_home / "config.toml").read_text())

    def test_dispatcher_never_calls_oanda_or_gateway_cli_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            command_text = " ".join(calls[0]).lower()
            self.assertNotIn("oanda", command_text)
            self.assertNotIn("stage-live-order", command_text)
            self.assertNotIn("autotrade-cycle", command_text)
            self.assertNotIn("position-execution", command_text)

    def test_dispatcher_source_has_no_direct_oanda_order_cancel_close_path(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "tools" / "guardian_wake_dispatcher.py").read_text()

        for forbidden in (
            "requests.",
            "urllib.request",
            "api-fxtrade.oanda.com",
            "OANDA_TOKEN",
            "openTrades",
            "stage-live-order",
            "position-execution",
            "autotrade-cycle --send",
        ):
            self.assertNotIn(forbidden, source)

    def test_invalid_gpt_output_does_not_create_executable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, "not json"))

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())

    def test_empty_last_message_falls_back_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex_stdout(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["codex"]["stdout_fallback_used"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(paths.action_receipt.exists())

    def test_jsonl_stdout_assistant_message_beats_trailing_status_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            stdout = "\n".join(
                [
                    json.dumps({"type": "session_started", "session_id": "s1"}),
                    json.dumps({"type": "agent_message", "message": _valid_receipt()}),
                    json.dumps({"type": "task_complete", "status": "success"}),
                ]
            )

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex_stdout(calls, stdout))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["codex"]["stdout_fallback_used"])
            self.assertEqual(result["codex"]["last_message_source"], "stdout-assistant")
            self.assertTrue(paths.action_receipt.exists())

    def test_output_last_message_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "output-last-message")
            self.assertTrue(paths.action_receipt.exists())

    def test_explicit_output_file_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_explicit(paths, calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "explicit-output-file")
            self.assertTrue(paths.action_receipt.exists())

    def test_session_jsonl_assistant_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session(paths, calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "session-jsonl-assistant")
            self.assertTrue(result["codex"]["session_jsonl_fallback_used"])
            self.assertTrue(paths.action_receipt.exists())

    def test_empty_output_becomes_codex_empty_last_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, ""))

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_EMPTY_LAST_MESSAGE")
            self.assertEqual(len(calls), 2)
            self.assertFalse(paths.action_receipt.exists())

    def test_codex_banner_without_json_becomes_no_json_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, "OpenAI Codex\nworking in read-only mode\n"),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_JSON_RECEIPT")
            self.assertEqual(len(calls), 2)
            self.assertFalse(paths.action_receipt.exists())

    def test_banner_only_output_retries_once_and_can_recover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_stdout_sequence(
                    calls,
                    ["OpenAI Codex\nworking in read-only mode\n", _valid_receipt()],
                ),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_assistant_text_without_json_becomes_no_json_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session(paths, calls, "I cannot recommend action."),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_JSON_RECEIPT")

    def test_session_jsonl_without_assistant_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session_without_assistant(paths, calls),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_ASSISTANT_MESSAGE")

    def test_codex_timeout_is_classified_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            def fake_timeout(cmd, *, input, capture_output, text, timeout, env):
                calls.append(list(cmd))
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="still running")

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=fake_timeout)

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_TIMEOUT")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_codex_auth_or_sandbox_failure_is_classified_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(calls, stderr="not authenticated; login required"),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_AUTH_OR_SANDBOX_FAILURE")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_codex_usage_limit_is_classified_without_schema_repair_and_delays_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            message = json.dumps(
                {
                    "message": (
                        "You've hit your usage limit. Visit settings to purchase more credits "
                        "or try again at 9:42 PM."
                    )
                }
            )

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_with_diagnostics(
                    calls,
                    message,
                    stdout=message,
                    returncode=1,
                ),
            )

            self.assertEqual(result["status"], "CODEX_USAGE_LIMIT")
            self.assertEqual(result["parse"]["error"], "CODEX_USAGE_LIMIT")
            self.assertFalse(result["repair_attempted"])
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())
            state = json.loads(paths.dispatcher_state.read_text())
            attempt = next(iter(state["dispatch_attempts"].values()))
            self.assertEqual(attempt["last_error"], "CODEX_USAGE_LIMIT")
            self.assertEqual(attempt["usage_limit_retry_seconds"], 90 * 60)
            self.assertEqual(
                datetime.fromisoformat(attempt["retry_after_utc"]),
                NOW + timedelta(minutes=90),
            )
            self.assertNotIn("parse_failure", state["last_result"])

    def test_legacy_usage_limit_schema_failure_is_reclassified_before_another_gpt_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            event = {**_event(severity="P1"), "wake_reason_codes": ["NEW_EVENT"]}
            source_time = (NOW - timedelta(seconds=5)).isoformat()
            paths.dispatcher_state.write_text(
                json.dumps(
                    {
                        "last_result": {
                            "generated_at_utc": source_time,
                            "parse": {
                                "error": "SCHEMA_INVALID",
                                "receipt": {
                                    "message": "You've hit your usage limit; try again at 9:42 PM."
                                },
                            },
                        },
                        "dispatch_attempts": {
                            event["dedupe_key"]: {
                                "event_id": event["event_id"],
                                "dedupe_key": event["dedupe_key"],
                                "event": event,
                                "attempt_count": 1,
                                "last_error": "SCHEMA_INVALID",
                                "last_status": "PARSE_FAILED",
                                "retry_after_utc": (NOW - timedelta(seconds=1)).isoformat(),
                                "expires_at_utc": (NOW + timedelta(hours=1)).isoformat(),
                            }
                        },
                        "pending_dispatches": {
                            event["dedupe_key"]: {
                                "event": event,
                                "queued_at_utc": (NOW - timedelta(minutes=1)).isoformat(),
                                "expires_at_utc": (NOW + timedelta(hours=1)).isoformat(),
                            }
                        },
                    }
                )
            )
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(calls, [])
            self.assertEqual(
                result["usage_limit_reclassification"]["status"],
                "LEGACY_USAGE_LIMIT_RECLASSIFIED",
            )
            state = json.loads(paths.dispatcher_state.read_text())
            attempt = state["dispatch_attempts"][event["dedupe_key"]]
            self.assertEqual(attempt["last_error"], "CODEX_USAGE_LIMIT")
            self.assertEqual(
                datetime.fromisoformat(attempt["retry_after_utc"]),
                NOW + timedelta(minutes=90),
            )

    def test_later_accepted_receipt_releases_only_older_usage_limit_backoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), wake=False)
            selected = _event(severity="P1")
            sibling = _event(
                severity="P2",
                event_id="event-P2-sibling",
                pair="AUD_JPY",
                dedupe_key="AUD_JPY|technical_state|TECHNICAL_STATE_CHANGE|NO_ACTION",
            )
            ordinary_failure = _event(
                severity="P2",
                event_id="event-P2-parse",
                pair="CAD_CHF",
                dedupe_key="CAD_CHF|technical_state|TECHNICAL_STATE_CHANGE|NO_ACTION",
            )
            failed_at = NOW - timedelta(minutes=10)
            recovered_at = NOW - timedelta(minutes=5)

            def attempt(event: dict, *, error: str) -> dict:
                return {
                    "event_id": event["event_id"],
                    "dedupe_key": event["dedupe_key"],
                    "event": event,
                    "attempt_count": 3,
                    "max_attempts": 3,
                    "last_error": error,
                    "last_status": error,
                    "last_failed_at_utc": failed_at.isoformat(),
                    "retry_after_utc": (NOW + timedelta(minutes=80)).isoformat(),
                    "expires_at_utc": (NOW + timedelta(hours=3)).isoformat(),
                    "retry_budget_exhausted": True,
                }

            paths.dispatcher_state.write_text(
                json.dumps(
                    {
                        "reviewed_events": {
                            "USD_CAD|spread|SPREAD_ANOMALY|HOLD": {
                                "receipt_written": True,
                                "last_status": "RECEIPT_WRITTEN",
                                "last_reviewed_at_utc": recovered_at.isoformat(),
                            }
                        },
                        "dispatch_attempts": {
                            selected["dedupe_key"]: attempt(selected, error="CODEX_USAGE_LIMIT"),
                            sibling["dedupe_key"]: attempt(sibling, error="CODEX_USAGE_LIMIT"),
                            ordinary_failure["dedupe_key"]: attempt(
                                ordinary_failure,
                                error="SCHEMA_INVALID",
                            ),
                        },
                        "pending_dispatches": {
                            event["dedupe_key"]: {
                                "event": event,
                                "queued_at_utc": failed_at.isoformat(),
                                "expires_at_utc": (NOW + timedelta(hours=3)).isoformat(),
                            }
                            for event in (selected, sibling)
                        },
                    }
                )
            )
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["usage_limit_recovery"]["released_count"], 2)
            self.assertEqual(len(calls), 1)
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertNotIn(selected["dedupe_key"], state["dispatch_attempts"])
            sibling_attempt = state["dispatch_attempts"][sibling["dedupe_key"]]
            self.assertEqual(sibling_attempt["attempt_count"], 0)
            self.assertFalse(sibling_attempt["retry_budget_exhausted"])
            self.assertEqual(sibling_attempt["last_status"], "CODEX_USAGE_LIMIT_RECOVERED")
            self.assertEqual(
                datetime.fromisoformat(sibling_attempt["retry_after_utc"]),
                NOW,
            )
            ordinary_attempt = state["dispatch_attempts"][ordinary_failure["dedupe_key"]]
            self.assertEqual(ordinary_attempt["attempt_count"], 3)
            self.assertTrue(ordinary_attempt["retry_budget_exhausted"])
            self.assertEqual(ordinary_attempt["last_status"], "SCHEMA_INVALID")

    def test_unsupported_codex_preflight_queues_without_full_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/opt/homebrew/bin/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.25.0\n"),
            )

            self.assertEqual(result["status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(codex_calls, [])
            self.assertEqual(preflight_calls, [["/opt/homebrew/bin/codex", "--version"]])
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())
            self.assertTrue(result["queued_for_active_trader"])
            preflight = result["codex_preflight"]
            self.assertEqual(preflight["codex_binary_path"], "/opt/homebrew/bin/codex")
            self.assertEqual(preflight["codex_version"], "0.25.0")
            self.assertEqual(preflight["requested_model"], "gpt-5.5")
            self.assertIn("QR_GUARDIAN_WAKE_CODEX_BIN", preflight["remediation_hint"])

            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(state["last_status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(state["last_result"]["codex_preflight"]["codex_version"], "0.25.0")
            self.assertNotIn("reviewed_events", state)
            self.assertEqual(
                state["dispatch_attempts"][_event(severity="P1")["dedupe_key"]]["attempt_count"],
                1,
            )
            escalation = json.loads(paths.escalation.read_text())
            self.assertTrue(escalation["queued_for_active_trader"])

    def test_binary_change_retries_same_event_during_failure_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            first_preflight_calls: list[list[str]] = []
            second_preflight_calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/Codex.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(first_preflight_calls, stdout="codex-cli 0.25.0\n"),
            )
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=10),
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/ChatGPT.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(second_preflight_calls, stdout="codex-cli 0.144.0\n"),
            )

            self.assertEqual(first["status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(second["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(codex_calls), 1)
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(state["dispatch_attempts"], {})
            self.assertIn(_event(severity="P1")["dedupe_key"], state["reviewed_events"])

    def test_unsupported_codex_model_error_is_not_schema_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(
                    calls,
                    stderr="gpt-5.5 model requires a newer version of Codex",
                ),
            )

            self.assertEqual(result["status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_MODEL_UNSUPPORTED")
            self.assertNotEqual(result["parse"]["error"], "SCHEMA_INVALID")
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["codex"]["requested_model"], "gpt-5.5")
            self.assertIn("newer version", result["codex"]["raw_stderr_excerpt"])
            self.assertIn("QR_GUARDIAN_WAKE_CODEX_BIN", result["codex"]["remediation_hint"])
            self.assertFalse(paths.action_receipt.exists())

    def test_supported_preflight_records_codex_version_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/ChatGPT.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.142.4\n"),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertNotIn(result["status"], {"CODEX_MODEL_UNSUPPORTED", "CODEX_CLI_VERSION_UNSUPPORTED"})
            self.assertEqual(preflight_calls, [["/Applications/ChatGPT.app/Contents/Resources/codex", "--version"]])
            self.assertEqual(len(codex_calls), 1)
            self.assertEqual(codex_calls[0][0], "/Applications/ChatGPT.app/Contents/Resources/codex")
            self.assertEqual(result["codex_preflight"]["codex_version"], "0.142.4")
            self.assertEqual(result["codex"]["codex_version"], "0.142.4")
            self.assertTrue(paths.action_receipt.exists())

    def test_supported_preflight_valid_json_ignores_stale_unsupported_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/ChatGPT.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex_with_diagnostics(
                    codex_calls,
                    _valid_receipt(),
                    stderr="old session diagnostic: gpt-5.5 model requires a newer version of Codex",
                    returncode=0,
                ),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.142.4\n"),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["status"], "OK")
            self.assertTrue(paths.action_receipt.exists())

    def test_old_session_unsupported_text_does_not_poison_new_valid_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale_dir = paths.codex_home / "sessions" / "2026" / "06" / "29"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "rollout-old.jsonl").write_text(
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "gpt-5.5 model requires a newer version of Codex",
                    }
                )
                + "\n"
            )
            codex_calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(codex_calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "output-last-message")
            self.assertTrue(paths.action_receipt.exists())

    def test_current_cli_version_failure_still_classifies_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(
                    calls,
                    stderr="codex cli version unsupported; too old, please upgrade",
                ),
            )

            self.assertEqual(result["status"], "CODEX_CLI_VERSION_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_CLI_VERSION_UNSUPPORTED")
            self.assertFalse(paths.action_receipt.exists())

    def test_one_repair_retry_occurs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertTrue(result["repair_attempted"])
            self.assertEqual([attempt["attempt"] for attempt in result["codex_attempts"]], ["initial", "repair"])
            self.assertEqual(len(calls), 2)

    def test_invalid_after_retry_does_not_create_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertFalse(paths.action_receipt.exists())

    def test_parse_failure_for_new_event_keeps_unrelated_accepted_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )
            _write_wake(paths, severity="P0", reasons=("SEVERITY_INCREASE",))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            archive_dir = paths.action_receipt.parent / "guardian_action_receipts"
            self.assertTrue(any(archive_dir.glob("*.json")))
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `PARSE_FAILED`", review)
            self.assertIn("Action: `HOLD`", review)

    def test_valid_retry_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", _valid_receipt()]),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["repair_attempted"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_empty_output_retries_once_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["", _valid_receipt()]),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_stale_broker_snapshot_refreshes_before_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale = json.loads(paths.broker_snapshot.read_text())
            stale["fetched_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths.broker_snapshot.write_text(json.dumps(stale))
            codex_calls: list[list[str]] = []
            refresh_calls: list[list[str]] = []

            def fake_refresh(cmd, *, cwd, capture_output, text, timeout, env):
                refresh_calls.append(list(cmd))
                fresh = dict(stale)
                fresh["fetched_at_utc"] = NOW.isoformat()
                paths.broker_snapshot.write_text(json.dumps(fresh))
                return SimpleNamespace(returncode=0, stdout="refreshed", stderr="")

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                snapshot_refresh_run=fake_refresh,
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(refresh_calls), 1)
            self.assertEqual(len(codex_calls), 1)
            self.assertTrue(result["broker_snapshot_freshness"]["fresh"])

    def test_stale_broker_snapshot_queues_without_prompt_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale = json.loads(paths.broker_snapshot.read_text())
            stale["fetched_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths.broker_snapshot.write_text(json.dumps(stale))
            codex_calls: list[list[str]] = []

            def fake_refresh(cmd, *, cwd, capture_output, text, timeout, env):
                return SimpleNamespace(returncode=1, stdout="", stderr="missing read credentials")

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                snapshot_refresh_run=fake_refresh,
            )

            self.assertEqual(result["status"], "BROKER_SNAPSHOT_STALE")
            self.assertTrue(result["queued_for_active_trader"])
            self.assertEqual(codex_calls, [])
            self.assertFalse(paths.action_receipt.exists())

            stale["fetched_at_utc"] = (NOW + timedelta(seconds=1)).isoformat()
            paths.broker_snapshot.write_text(json.dumps(stale))
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            paths.events.write_text(json.dumps({"events": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={},
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
            )

            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(codex_calls), 1)

    def test_active_lock_after_invalid_output_queues_instead_of_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            def fake(cmd, *, input, capture_output, text, timeout, env):
                calls.append(list(cmd))
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("not json")
                paths.live_lock.mkdir(parents=True)
                (paths.live_lock / "pid").write_text(str(os.getpid()))
                (paths.live_lock / "command").write_text("run-autotrade-live")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=fake)

            self.assertEqual(result["status"], "QUEUED_FOR_ACTIVE_TRADER")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_parse_failure_uses_bounded_backoff_then_retries_same_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            third = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=61),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(first["status"], "PARSE_FAILED")
            self.assertEqual(second["status"], "SUPPRESSED")
            self.assertEqual(second["selection"]["suppressed"][0]["reason"], "RETRY_BACKOFF")
            self.assertEqual(third["status"], "RECEIPT_WRITTEN")
            self.assertEqual(third["retry_injection"]["status"], "DUE_FAILED_ATTEMPTS_INJECTED")
            self.assertEqual(len(calls), 3)
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(state["dispatch_attempts"], {})
            self.assertEqual(state["parse_failures"], {})
            self.assertIn(_event(severity="P1")["dedupe_key"], state["reviewed_events"])

    def test_retry_budget_waits_for_ttl_before_opening_fresh_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            env = {
                "QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1",
                "QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS": "2",
                "QR_GUARDIAN_WAKE_RETRY_TTL_SECONDS": "10",
            }
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env=env,
                subprocess_run=_fake_codex_sequence(calls, ["bad", "bad"]),
            )
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env=env,
                subprocess_run=_fake_codex_sequence(calls, ["bad", "bad"]),
            )
            blocked = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=2),
                env=env,
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=11),
                env=env,
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(first["status"], "PARSE_FAILED")
            self.assertEqual(second["status"], "PARSE_FAILED")
            self.assertEqual(blocked["status"], "SUPPRESSED")
            self.assertEqual(blocked["selection"]["suppressed"][0]["reason"], "RETRY_BUDGET_EXHAUSTED")
            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 5)

    def test_quote_tick_does_not_reset_failed_dispatch_backoff_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["bad", "bad"]),
            )
            original = _event(severity="P1")
            ticked = {
                **original,
                "event_id": "event-tick-only",
                "price_zone": "EUR_USD 1.17001 rejection",
            }
            paths.events.write_text(json.dumps({"events": [ticked]}))
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))

            blocked = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(first["status"], "PARSE_FAILED")
            self.assertEqual(blocked["status"], "SUPPRESSED")
            self.assertEqual(blocked["selection"]["suppressed"][0]["reason"], "RETRY_BACKOFF")
            state = json.loads(paths.dispatcher_state.read_text())
            attempt = state["dispatch_attempts"][original["dedupe_key"]]
            self.assertEqual(attempt["attempt_count"], 1)
            self.assertEqual(attempt["event_id"], original["event_id"])
            self.assertEqual(len(calls), 2)

    def test_valid_gpt_output_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["receipt_status"], "ACCEPTED")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            self.assertIn("expires_at_utc", payload)
            self.assertFalse(payload["consumed_by_trader"])
            self.assertEqual(payload["receipt"]["action"], "TRADE")
            self.assertEqual(payload["receipt"]["dedupe_key"], payload["selected_event"]["dedupe_key"])
            self.assertTrue(payload["receipt"]["gateway_required"])
            self.assertTrue(payload["receipt"]["no_direct_oanda"])
            state = json.loads(paths.dispatcher_state.read_text())
            baseline = state["reviewed_events"][payload["selected_event"]["dedupe_key"]]
            self.assertEqual(baseline["event_id"], payload["selected_event_id"])
            self.assertEqual(baseline["price_zone"], payload["selected_event"]["price_zone"])
            self.assertEqual(baseline["details"], payload["selected_event"]["details"])
            self.assertIn("material_fingerprint", baseline)

    def test_accepted_material_event_writes_idempotent_safe_tuning_work_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            first_work_order_text = paths.tuning_work_order.read_text()
            work_order = json.loads(first_work_order_text)

            # Simulate dispatcher-state restoration without deleting the
            # durable work order; the same event must not churn the file.
            paths.dispatcher_state.write_text("{}\n")
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=10),
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(tuning_review=True),
                ),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(first["tuning_handoff"]["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(work_order["status"], "PENDING_HOURLY_AI_REVIEW")
            self.assertFalse(work_order["live_permission_allowed"])
            self.assertTrue(work_order["no_direct_oanda"])
            self.assertTrue(work_order["preserve_blockers"])
            self.assertEqual(work_order["bot_tuning_review_validation"]["status"], "VALID")
            self.assertEqual(second["tuning_handoff"]["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(paths.tuning_work_order.read_text(), first_work_order_text)

    def test_margin_pressure_cannot_inherit_a_bot_tuning_queue_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            margin_event = {
                **_event(
                    severity="P1",
                    event_id="portfolio-margin-pressure",
                    pair="PORTFOLIO",
                    direction="",
                    dedupe_key="PORTFOLIO|PORTFOLIO_MARGIN_CAPACITY|MARGIN_PRESSURE|HOLD",
                ),
                "event_type": "MARGIN_PRESSURE",
                "action_hint": "HOLD",
                "recommended_review_type": "RISK_REVIEW",
                "thesis": "portfolio margin capacity",
                "thesis_state": "WOUNDED",
                "price_zone": "margin_used/nav=0.850; available/nav=0.151; cap=0.920",
                # This material reason belongs to a price observation that
                # woke the successor.  It must not turn the risk event into a
                # bot/lane tuning obligation.
                "wake_reason_codes": [
                    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
                    "DURABLE_PENDING_SUCCESSOR",
                ],
                "details": {
                    "nav_jpy": 291704.3565,
                    "margin_used_jpy": 247870.52,
                    "margin_available_jpy": 44072.7198,
                    "max_margin_utilization_pct": 92.0,
                },
            }

            result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=margin_event,
                receipt={"bot_tuning_review": _valid_tuning_review("PORTFOLIO")},
                now=NOW,
            )

            self.assertEqual(result["status"], "SKIPPED_NON_TUNING_EVENT")
            self.assertFalse(path.exists())
            self.assertEqual(dispatcher_module._tuning_handoff_reasons(margin_event), [])

    def test_supported_tuning_event_surfaces_still_admit_material_reasons(self) -> None:
        for event_type in (
            "TECHNICAL_STATE_CHANGE",
            "FAILED_ACCEPTANCE",
        ):
            with self.subTest(event_type=event_type):
                event = {
                    **_event(severity="P1"),
                    "event_type": event_type,
                    "wake_reason_codes": ["LARGE_PRICE_DISPLACEMENT_STATE_CHANGE"],
                }
                self.assertEqual(
                    dispatcher_module._tuning_handoff_reasons(event),
                    ["LARGE_PRICE_DISPLACEMENT_STATE_CHANGE"],
                )

    def test_revision_five_writer_rejects_nonadmissible_pending_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(pair="EUR_USD"),
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            before = path.read_bytes()
            source_sha256 = hashlib.sha256(before).hexdigest()
            for surface in ("top", "child"):
                with self.subTest(surface=surface):
                    payload = json.loads(before)
                    target = (
                        payload
                        if surface == "top"
                        else payload["work_orders"][0]
                    )
                    target["selected_event"]["event_type"] = "MARGIN_PRESSURE"

                    with self.assertRaisesRegex(OSError, "not tuning-admissible"):
                        dispatcher_module._write_tuning_queue_json(
                            path,
                            payload,
                            expected_source_sha256=source_sha256,
                        )

            self.assertEqual(path.read_bytes(), before)

    def test_legacy_non_tuning_admissions_are_audited_and_release_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            legacy_event_types = frozenset(
                {
                    *dispatcher_module._TUNING_HANDOFF_EVENT_TYPES,
                    "MARGIN_PRESSURE",
                    "CONTRACT_ADD_TRIGGER",
                }
            )
            legacy_work_order_ids: set[str] = set()
            with (
                patch.object(
                    dispatcher_module,
                    "_TUNING_HANDOFF_EVENT_TYPES",
                    legacy_event_types,
                ),
                patch.object(
                    dispatcher_module,
                    "TUNING_QUEUE_SCHEMA_REVISION",
                    4,
                ),
            ):
                baseline_event = _technical_tuning_event(
                    pair="EUR_USD",
                    event_id="baseline-technical",
                )
                baseline = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=baseline_event,
                    receipt={
                        "bot_tuning_review": _valid_tuning_review("EUR_USD")
                    },
                    now=NOW,
                )
                self.assertEqual(baseline["status"], "WORK_ORDER_WRITTEN")
                for index in range(5):
                    margin_event = {
                        **_event(
                            severity="P1",
                            event_id=f"legacy-margin-{index}",
                            pair="PORTFOLIO",
                            direction="",
                            dedupe_key=(
                                "PORTFOLIO|PORTFOLIO_MARGIN_CAPACITY|"
                                "MARGIN_PRESSURE|HOLD"
                            ),
                        ),
                        "event_type": "MARGIN_PRESSURE",
                        "action_hint": "HOLD",
                        "recommended_review_type": "RISK_REVIEW",
                        "thesis": "portfolio margin capacity",
                        "thesis_state": "WOUNDED",
                        "price_zone": f"margin_used/nav=0.84{index}",
                        "wake_reason_codes": [
                            "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE"
                        ],
                    }
                    written = dispatcher_module._maybe_write_tuning_work_order(
                        path=path,
                        selected_event=margin_event,
                        receipt={
                            "bot_tuning_review": _valid_tuning_review("PORTFOLIO")
                        },
                        now=NOW + timedelta(seconds=index + 1),
                    )
                    self.assertEqual(written["status"], "WORK_ORDER_WRITTEN")
                    legacy_work_order_ids.add(written["work_order_id"])

                for index in range(3):
                    contract_event = {
                        **_event(
                            severity="P1",
                            event_id=f"legacy-contract-add-{index}",
                            pair="EUR_USD",
                            direction="LONG",
                            dedupe_key=f"EUR_USD|range-rail-{index}|CONTRACT_ADD_TRIGGER|ADD",
                        ),
                        "event_type": "CONTRACT_ADD_TRIGGER",
                        "action_hint": "ADD",
                        "recommended_review_type": "ADD_REVIEW",
                        "thesis": f"range rail recheck {index}",
                        "price_zone": f"mid <= 1.140{index} fired",
                        "wake_reason_codes": [
                            "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE"
                        ],
                        "details": {
                            "contract_trigger": {
                                "kind": "range_rail_recheck",
                                "lane_id": (
                                    "failure_trader:EUR_USD:LONG:"
                                    "BREAKOUT_FAILURE:LIMIT"
                                ),
                            }
                        },
                    }
                    written = dispatcher_module._maybe_write_tuning_work_order(
                        path=path,
                        selected_event=contract_event,
                        receipt={
                            "bot_tuning_review": _valid_tuning_review("EUR_USD")
                        },
                        now=NOW + timedelta(seconds=10 + index),
                    )
                    self.assertEqual(written["status"], "WORK_ORDER_WRITTEN")
                    legacy_work_order_ids.add(written["work_order_id"])

            legacy_source = json.loads(path.read_text())
            self.assertEqual(legacy_source["pending_count"], 9)
            expected_rejected_originals = {
                str(item.get("work_order_id")): (
                    dispatcher_module._strip_tuning_envelope(item)
                )
                for item in legacy_source["work_orders"]
                if dispatcher_module._tuning_queue_entry_is_non_admissible(item)
            }
            recovery_event = _technical_tuning_event(
                pair="GBP_USD",
                event_id="post-repair-technical",
            )
            recovered = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=recovery_event,
                receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(
                recovered["status"],
                "WORK_ORDER_WRITTEN",
                recovered,
            )
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["queue_schema_revision"],
                dispatcher_module.TUNING_QUEUE_SCHEMA_REVISION,
            )
            self.assertEqual(payload["pending_count"], 2)
            self.assertEqual(payload["terminal_history_count"], 0)
            ledger_path = dispatcher_module._tuning_admission_rejection_path(path)
            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(ledger["record_count"], 8)
            rejected = ledger["records"]
            commitment = payload["admission_rejection_ledger"]
            self.assertEqual(commitment["record_count"], 8)
            self.assertEqual(
                commitment["sha256"],
                hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                commitment["rejection_ids"],
                [str(item["rejection_id"]) for item in rejected],
            )
            self.assertEqual(
                {str(item.get("work_order_id")) for item in rejected},
                legacy_work_order_ids,
            )
            self.assertEqual(
                {
                    str(item.get("work_order_id")): item["original_entry"]
                    for item in rejected
                },
                expected_rejected_originals,
            )
            for item in rejected:
                self.assertIn(
                    item["rejected_event_type"],
                    {"MARGIN_PRESSURE", "CONTRACT_ADD_TRIGGER"},
                )
                self.assertEqual(
                    item["validator"],
                    dispatcher_module.TUNING_ADMISSION_REJECTION_VALIDATOR,
                )
                self.assertNotIn("status", item)
                self.assertNotIn("experiment_id", item)
                self.assertNotIn("experiment_evidence_ref", item)
                self.assertEqual(
                    item["original_entry"]["selected_event"]["event_type"],
                    item["rejected_event_type"],
                )

            loaded = dispatcher_module._load_tuning_work_order(path)
            self.assertNotIn("_read_error", loaded)
            loaded_ledger = dispatcher_module._load_tuning_admission_rejections(
                ledger_path
            )
            self.assertNotIn("_read_error", loaded_ledger)
            migrated_bytes = path.read_bytes()
            ledger_bytes = ledger_path.read_bytes()
            repeated = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=recovery_event,
                receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                now=NOW + timedelta(minutes=2),
            )
            self.assertEqual(repeated["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(path.read_bytes(), migrated_bytes)
            self.assertEqual(ledger_path.read_bytes(), ledger_bytes)

            ledger_path.unlink()
            missing_ledger = dispatcher_module._load_tuning_work_order(path)
            self.assertIn("_read_error", missing_ledger)
            self.assertIn("ledger is missing", missing_ledger["_read_error"])

            ledger_path.write_bytes(ledger_bytes)
            tampered = json.loads(ledger_path.read_text())
            tampered["records"][0]["rejection_code"] = "TAMPERED"
            ledger_path.write_text(json.dumps(tampered))
            tampered_ledger = dispatcher_module._load_tuning_work_order(path)
            self.assertIn("_read_error", tampered_ledger)
            self.assertIn(
                "failed strict validation",
                tampered_ledger["_read_error"],
            )

    def test_all_invalid_revision_four_queue_repairs_to_empty_revision_five(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            with patch.object(
                dispatcher_module,
                "TUNING_QUEUE_SCHEMA_REVISION",
                4,
            ):
                created = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(pair="EUR_USD"),
                    receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                    now=NOW,
                )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            legacy = json.loads(path.read_text())
            for entry in (legacy, legacy["work_orders"][0]):
                entry["selected_event"]["wake_reason_codes"] = [
                    "DURABLE_PENDING_SUCCESSOR"
                ]
            path.write_text(json.dumps(legacy))

            repaired = dispatcher_module.repair_tuning_queue_admissions(
                path=path,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(repaired["status"], "ADMISSION_REPAIR_WRITTEN")
            self.assertEqual(repaired["rejected_count"], 1)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["queue_schema_revision"], 5)
            self.assertEqual(payload["work_orders"], [])
            self.assertEqual(payload["pending_count"], 0)
            self.assertNotIn("work_order_id", payload)
            self.assertNotIn(
                "_read_error",
                dispatcher_module._load_tuning_work_order(path),
            )
            ledger = json.loads(
                dispatcher_module._tuning_admission_rejection_path(path).read_text()
            )
            self.assertEqual(ledger["record_count"], 1)
            self.assertEqual(
                ledger["records"][0]["rejection_code"],
                "MATERIAL_TUNING_REASON_MISSING",
            )

    def test_admission_repair_preserves_exact_invalid_child_review_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            with patch.object(
                dispatcher_module,
                "TUNING_QUEUE_SCHEMA_REVISION",
                4,
            ):
                created = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(pair="EUR_USD"),
                    receipt={
                        "bot_tuning_review": _valid_tuning_review("EUR_USD")
                    },
                    now=NOW,
                )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            legacy = json.loads(path.read_text())
            child = legacy["work_orders"][0]
            for entry in (legacy, child):
                entry["selected_event"]["event_type"] = "MARGIN_PRESSURE"

            invalid_child_review = {
                "review_status": "TEST_REQUIRED",
                "affected_pairs": ["EUR_USD", "GBP_USD"],
                "legacy_evidence_acquisition": {
                    "action_kind": "COLLECT_FORWARD_ENTRIES",
                    "source_ref": "logs/legacy-forward.jsonl",
                    "nested": [
                        {"keep": "every legacy field"},
                        ["including", "nested", "values"],
                    ],
                },
                "legacy_notes": "invalid review must remain verbatim",
            }
            child["bot_tuning_review"] = invalid_child_review
            # The top compatibility mirror has the same identity but divergent
            # content.  It must not overwrite the durable child source kept by
            # the rejection ledger.
            legacy["bot_tuning_review"] = {
                "review_status": "NO_CHANGE_INSUFFICIENT_EVIDENCE",
                "top_mirror_only": True,
            }
            expected_original = dispatcher_module._strip_tuning_envelope(
                json.loads(json.dumps(child))
            )
            path.write_text(json.dumps(legacy))

            repaired = dispatcher_module.repair_tuning_queue_admissions(
                path=path,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(repaired["status"], "ADMISSION_REPAIR_WRITTEN")
            ledger_path = dispatcher_module._tuning_admission_rejection_path(path)
            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(ledger["record_count"], 1)
            original = ledger["records"][0]["original_entry"]
            self.assertEqual(original, expected_original)
            self.assertEqual(original["bot_tuning_review"], invalid_child_review)
            self.assertNotIn("top_mirror_only", original["bot_tuning_review"])
            self.assertNotIn(
                "_read_error",
                dispatcher_module._load_tuning_admission_rejections(ledger_path),
            )

    def test_orphan_rejection_ledger_is_bound_before_revision_five_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            with patch.object(
                dispatcher_module,
                "TUNING_QUEUE_SCHEMA_REVISION",
                4,
            ):
                created = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(pair="EUR_USD"),
                    receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                    now=NOW,
                )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            loaded = dispatcher_module._load_tuning_work_order(path)
            source_sha256 = loaded["_queue_source_sha256"]
            invalid_original = dict(loaded["work_orders"][0])
            invalid_original["selected_event"] = {
                **invalid_original["selected_event"],
                "event_type": "MARGIN_PRESSURE",
            }
            ledger_path = dispatcher_module._tuning_admission_rejection_path(path)
            persisted = dispatcher_module._persist_tuning_admission_rejections(
                path=ledger_path,
                entries=[invalid_original],
                source_queue_sha256=source_sha256,
                source_queue_schema_revision=4,
                now=NOW + timedelta(seconds=1),
            )
            self.assertEqual(persisted["status"], "ADMISSION_REJECTIONS_WRITTEN")
            before = path.read_bytes()

            repaired = dispatcher_module.repair_tuning_queue_admissions(
                path=path,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(repaired["status"], "ADMISSION_REPAIR_WRITTEN")
            self.assertTrue(repaired["ledger_bound_only"])
            self.assertEqual(repaired["rejected_count"], 0)
            self.assertNotEqual(path.read_bytes(), before)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["queue_schema_revision"], 5)
            self.assertEqual(payload["pending_count"], 1)
            self.assertEqual(
                payload["admission_rejection_ledger"]["record_count"],
                1,
            )
            self.assertNotIn(
                "_read_error",
                dispatcher_module._load_tuning_work_order(path),
            )

    def test_tuning_queue_groups_same_closed_candle_semantics_across_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            first_event = {
                **_event(
                    severity="P2",
                    event_id="technical-observation-1",
                    pair="CAD_CHF",
                    direction="SHORT",
                    dedupe_key="CAD_CHF|technical|TECHNICAL_STATE_CHANGE|NO_ACTION",
                ),
                "event_type": "TECHNICAL_STATE_CHANGE",
                "action_hint": "NO_ACTION",
                "recommended_review_type": "TUNING_REVIEW",
                "price_zone": "mid=0.571025 spread_pips=6.9",
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                "details": {
                    "mid": 0.571025,
                    "live_spread_pips": 6.9,
                    "chart_generated_at_utc": NOW.isoformat(),
                    "material_fingerprint": {
                        "dominant_regime": "TREND_DOWN",
                        "volatility_bucket": "QUIET",
                        "family_consensus": {
                            "trend": "DOWN",
                            "mean_reversion": "UP",
                            "breakout": "MIXED",
                        },
                        "closed_structure": "M1:CHOCH_DOWN:2026-06-30T03:20:00+00:00",
                    },
                    "closed_candle_watermarks": {
                        "M1": "2026-06-30T03:29:00+00:00",
                        "M5": "2026-06-30T03:25:00+00:00",
                        "M15": "2026-06-30T03:15:00+00:00",
                    },
                },
            }
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={
                    "receipt_id": "receipt-1",
                    "bot_tuning_review": _valid_tuning_review("CAD_CHF"),
                },
                now=NOW,
            )
            first_text = path.read_text()
            second_event = {
                **first_event,
                "event_id": "technical-observation-2",
                "price_zone": "mid=0.571025 spread_pips=13.8",
                "details": {
                    **first_event["details"],
                    "live_spread_pips": 13.8,
                    "chart_generated_at_utc": (NOW + timedelta(hours=3)).isoformat(),
                },
            }
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=second_event,
                receipt={
                    "receipt_id": "receipt-2",
                    "bot_tuning_review": _valid_tuning_review("CAD_CHF"),
                },
                now=NOW + timedelta(hours=3),
            )

            self.assertEqual(first["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(second["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(first["semantic_state_id"], second["semantic_state_id"])
            self.assertNotEqual(first["observation_id"], second["observation_id"])
            self.assertEqual(path.read_text(), first_text)

    def test_technical_pair_scope_does_not_split_when_position_action_hint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            candidate_event = _technical_tuning_event(
                pair="CAD_CHF",
                event_id="candidate-technical",
            )
            candidate_event.update(
                {
                    "action_hint": "NO_ACTION",
                    "recommended_review_type": "TUNING_REVIEW",
                    "dedupe_key": "CAD_CHF|technical|TECHNICAL_STATE_CHANGE|NO_ACTION",
                }
            )
            open_position_event = {
                **candidate_event,
                "event_id": "open-position-technical",
                "action_hint": "HOLD",
                "recommended_review_type": "THESIS_REVIEW",
                "dedupe_key": "CAD_CHF|technical|TECHNICAL_STATE_CHANGE|HOLD",
            }

            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=candidate_event,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW,
            )
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=open_position_event,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            self.assertEqual(first["semantic_state_id"], second["semantic_state_id"])
            self.assertEqual(second["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(payload["pending_count"], 1)

    def test_major_figure_semantic_identity_ignores_ask_only_rollover_widening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            first_event = {
                **_event(
                    severity="P2",
                    event_id="major-1",
                    pair="EUR_NZD",
                    direction="",
                    dedupe_key="EUR_NZD|major|FAILED_ACCEPTANCE|HOLD",
                ),
                "action_hint": "HOLD",
                "price_zone": "major figure 1.98000 distance_pips=2.00",
                "wake_reason_codes": ["FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE"],
                "details": {"bid": 1.97986, "ask": 1.98054},
            }
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_NZD")},
                now=NOW,
            )
            second_event = {
                **first_event,
                "event_id": "major-2",
                "price_zone": "major figure 1.98000 distance_pips=6.25",
                "details": {"bid": 1.97986, "ask": 1.98139},
            }
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=second_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_NZD")},
                now=NOW + timedelta(hours=1),
            )
            third_event = {
                **first_event,
                "event_id": "major-3",
                "price_zone": "major figure 1.98000 distance_pips=4.00",
                "details": {"bid": 1.97966, "ask": 1.98046},
            }
            third = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=third_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_NZD")},
                now=NOW + timedelta(hours=2),
            )

            self.assertEqual(first["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(second["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(third["status"], "WORK_ORDER_OBSERVATION_APPENDED")
            self.assertEqual(first["semantic_state_id"], second["semantic_state_id"])
            self.assertEqual(first["semantic_state_id"], third["semantic_state_id"])
            self.assertNotEqual(first["observation_id"], second["observation_id"])
            self.assertEqual(json.loads(path.read_text())["observation_count"], 2)

    def test_material_price_displacement_appends_bounded_observation_to_same_work_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            first_event = _technical_tuning_event(
                event_id="technical-price-1",
                mid=0.571000,
                reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",),
            )
            second_event = _technical_tuning_event(
                event_id="technical-price-2",
                mid=0.573500,
                reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",),
            )

            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW,
            )
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=second_event,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            self.assertEqual(first["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(second["status"], "WORK_ORDER_OBSERVATION_APPENDED")
            self.assertEqual(payload["pending_count"], 1)
            self.assertEqual(payload["observation_count"], 2)
            self.assertEqual(
                [item["selected_event"]["details"]["mid"] for item in payload["observations"]],
                [0.571000, 0.573500],
            )

    def test_same_lane_status_retains_new_price_trigger_and_blocker_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            lane_id = "trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION:LIMIT"
            first_event = {
                **_event(
                    severity="P2",
                    event_id="lane-1",
                    pair="AUD_JPY",
                    dedupe_key=f"AUD_JPY|{lane_id}|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE"],
                "details": {
                    "lane_id": lane_id,
                    "status": "DRY_RUN_BLOCKED",
                    "trigger": 104.10,
                    "blocker": "SPREAD_TOO_WIDE",
                },
            }
            second_event = {
                **first_event,
                "event_id": "lane-2",
                "price_zone": "AUD_JPY 104.250 rejection",
                "details": {
                    **first_event["details"],
                    "trigger": 104.25,
                    "blocker": "LOCATION_CHASE",
                },
            }

            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("AUD_JPY", lane_id=lane_id)},
                now=NOW,
            )
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=second_event,
                receipt={"bot_tuning_review": _valid_tuning_review("AUD_JPY", lane_id=lane_id)},
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            observations = payload["work_orders"][0]["observations"]
            self.assertEqual(second["status"], "WORK_ORDER_OBSERVATION_APPENDED")
            self.assertEqual(payload["pending_count"], 1)
            self.assertEqual(
                [item["selected_event"]["details"]["blocker"] for item in observations],
                ["SPREAD_TOO_WIDE", "LOCATION_CHASE"],
            )

    def test_tuning_reason_without_structured_review_is_persisted_but_not_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []

            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )
            first_state = json.loads(paths.dispatcher_state.read_text())

            self.assertEqual(first["status"], "TUNING_HANDOFF_FAILED")
            self.assertFalse(first["receipt_written"])
            self.assertEqual(first["tuning_handoff"]["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertEqual(first["gateway_handoff"]["status"], "SKIPPED_TUNING_HANDOFF_FAILED")
            self.assertNotIn(_event(severity="P1")["dedupe_key"], first_state.get("reviewed_events", {}))
            self.assertEqual(
                json.loads(paths.tuning_work_order.read_text())["bot_tuning_review_validation"]["status"],
                "MISSING",
            )

            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(recovered["tuning_handoff"]["status"], "WORK_ORDER_REVIEW_ENRICHED")
            self.assertEqual(
                json.loads(paths.tuning_work_order.read_text())["bot_tuning_review_validation"]["status"],
                "VALID",
            )

    def test_failed_tuning_handoff_does_not_publish_or_supersede_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            prior_receipt = {
                "status": "ACCEPTED",
                "receipt_status": "ACCEPTED",
                "receipt_lifecycle": "ACTIVE",
                "dispatcher_status": "RECEIPT_WRITTEN",
                "selected_event_id": "urgent-reduce",
                "receipt": {"action": "REDUCE", "event_id": "urgent-reduce"},
            }
            paths.action_receipt.write_text(json.dumps(prior_receipt))
            before = paths.action_receipt.read_text()
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            archive_dir = paths.action_receipt.parent / "guardian_action_receipts"
            self.assertEqual(result["status"], "TUNING_HANDOFF_FAILED")
            self.assertFalse(result["receipt_written"])
            self.assertEqual(paths.action_receipt.read_text(), before)
            self.assertFalse(archive_dir.exists())

    def test_nested_cross_pair_or_execution_adjustment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            receipt = json.loads(_valid_receipt())
            receipt["bot_tuning_review"] = {
                **_valid_tuning_review("EUR_USD"),
                "affected_bot_families": ["oanda_gateway"],
                "proposed_adjustments": [
                    {
                        "pair": "USD_JPY",
                        "bot_family": "oanda_gateway",
                        "parameter": "send_order_now",
                        "current_value": 0,
                        "candidate_value": 1,
                        "rationale": "send MARKET now",
                        "live_permission_allowed": True,
                        "no_direct_oanda": False,
                    }
                ],
            }

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, json.dumps(receipt)),
            )

            payload = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(result["status"], "TUNING_HANDOFF_FAILED")
            self.assertEqual(payload["bot_tuning_review_validation"]["status"], "INVALID_UNSAFE_BOUNDARY")
            self.assertNotIn("bot_tuning_review", payload)

    def test_empty_family_and_risk_sizing_adjustments_are_not_valid_tuning_reviews(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")
        empty_family = {
            **_valid_tuning_review("EUR_USD"),
            "affected_bot_families": [""],
        }
        risk_adjustment = {
            **_valid_tuning_review("EUR_USD"),
            "affected_bot_families": ["trend"],
            "proposed_adjustments": [
                {
                    "pair": "EUR_USD",
                    "bot_family": "trend",
                    "parameter": "daily_risk_pct",
                    "current_value": 1.0,
                    "candidate_value": 10.0,
                    "rationale": "increase risk",
                }
            ],
        }
        allocation_adjustment = {
            **_valid_tuning_review("EUR_USD"),
            "affected_bot_families": ["trend"],
            "proposed_adjustments": [
                {
                    "pair": "EUR_USD",
                    "bot_family": "trend",
                    "parameter": "allocation_fraction",
                    "current_value": 0.1,
                    "candidate_value": 1.0,
                    "rationale": "increase allocation",
                }
            ],
        }
        instruction = {
            **_valid_tuning_review("EUR_USD"),
            "hypothesis": "bypass gateway and send MARKET now",
        }

        empty_result = dispatcher_module._validate_bot_tuning_review(
            empty_family,
            selected_event=selected,
        )
        risk_result = dispatcher_module._validate_bot_tuning_review(
            risk_adjustment,
            selected_event=selected,
        )
        instruction_result = dispatcher_module._validate_bot_tuning_review(
            instruction,
            selected_event=selected,
        )
        allocation_result = dispatcher_module._validate_bot_tuning_review(
            allocation_adjustment,
            selected_event=selected,
        )

        self.assertNotEqual(empty_result["status"], "VALID")
        self.assertEqual(risk_result["status"], "INVALID_UNSAFE_BOUNDARY")
        self.assertEqual(allocation_result["status"], "INVALID_UNSAFE_BOUNDARY")
        self.assertEqual(instruction_result["status"], "INVALID_UNSAFE_BOUNDARY")

        for unsafe_text in (
            "invoke the broker client to liquidate all holdings",
            "execute a short transaction immediately",
            "submit a position through the trading API",
            "OANDAへ直接注文を送れば利益になる",
            "今すぐ実口座で成行注文を発注して確認する",
            "リスクゲートを無視して実注文を送る",
        ):
            with self.subTest(unsafe_text=unsafe_text):
                unsafe_result = dispatcher_module._validate_bot_tuning_review(
                    {
                        **_valid_tuning_review("EUR_USD"),
                        "hypothesis": unsafe_text,
                    },
                    selected_event=selected,
                )
                self.assertEqual(unsafe_result["status"], "INVALID_UNSAFE_BOUNDARY")

        for safe_offline_text in (
            "注文の実行コストを履歴リプレイで比較する",
            "ポジション実行の過去データを比較する",
        ):
            with self.subTest(safe_offline_text=safe_offline_text):
                self.assertFalse(dispatcher_module._unsafe_tuning_instruction(safe_offline_text))

    def test_quantity_weight_and_unrecognized_parameters_are_rejected(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")

        def review(parameter: str) -> dict:
            return {
                **_valid_tuning_review("EUR_USD"),
                "affected_bot_families": ["trend"],
                "proposed_adjustments": [
                    {
                        "pair": "EUR_USD",
                        "bot_family": "trend",
                        "parameter": parameter,
                        "current_value": 0.1,
                        "candidate_value": 0.2,
                        "rationale": "frozen-cohort technical comparison",
                    }
                ],
            }

        for parameter in (
            "order_quantity",
            "trade_quantity",
            "account_weight",
            "portfolio_share",
            "trade_amount",
            "daily_loss_limit_threshold",
            "drawdown_threshold",
            "max_open_positions_threshold",
            "broker_execute_threshold",
            "spread_threshold",
            "slippage_threshold",
            "latency_staleness_threshold",
            "mystery_knob",
        ):
            with self.subTest(parameter=parameter):
                result = dispatcher_module._validate_bot_tuning_review(
                    review(parameter),
                    selected_event=selected,
                )
                self.assertEqual(result["status"], "INVALID_UNSAFE_BOUNDARY")

    def test_test_required_rejects_allowlisted_parameters_the_evaluator_cannot_run(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")
        unsupported = (
            ("trend", "confirmation_bars"),
            ("trend", "trend_lookback_bars"),
            ("mean_reversion", "band_score_weight"),
            ("execution", "spread_score_weight"),
        )

        for family, parameter in unsupported:
            with self.subTest(family=family, parameter=parameter):
                review = {
                    **_valid_tuning_review("EUR_USD"),
                    "affected_bot_families": [family],
                    "proposed_adjustments": [
                        {
                            "pair": "EUR_USD",
                            "bot_family": family,
                            "parameter": parameter,
                            "current_value": 0.4,
                            "candidate_value": 0.5,
                            "rationale": "compare one frozen offline cohort",
                        }
                    ],
                }

                result = dispatcher_module._validate_bot_tuning_review(
                    review,
                    selected_event=selected,
                )

                self.assertEqual(result["status"], "INVALID_INCOMPLETE_REVIEW")
                self.assertNotIn("review", result)
                self.assertTrue(
                    any("current frozen evaluator" in issue for issue in result["issues"])
                )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            execution_review = {
                **_valid_tuning_review("EUR_USD"),
                "affected_bot_families": ["execution"],
                "proposed_adjustments": [
                    {
                        "pair": "EUR_USD",
                        "bot_family": "execution",
                        "parameter": "spread_score_weight",
                        "current_value": 0.4,
                        "candidate_value": 0.5,
                        "rationale": "compare one frozen offline cohort",
                    }
                ],
            }

            written = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=selected,
                receipt={"bot_tuning_review": execution_review},
                now=NOW,
            )

            payload = json.loads(path.read_text())
            self.assertEqual(written["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertEqual(
                payload["bot_tuning_review_validation"]["status"],
                "INVALID_INCOMPLETE_REVIEW",
            )
            self.assertNotIn("bot_tuning_review", payload)

    def test_test_required_floor_values_are_finite_unit_interval_and_tighten_only(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")

        def review(current_value: object, candidate_value: object) -> dict:
            payload = _valid_tuning_review("EUR_USD")
            payload["proposed_adjustments"][0]["current_value"] = current_value
            payload["proposed_adjustments"][0]["candidate_value"] = candidate_value
            return payload

        invalid = (
            (-0.01, 0.2, "within 0..1"),
            (0.2, 1.01, "within 0..1"),
            (float("nan"), 0.3, "finite numbers"),
            (0.3, float("inf"), "finite numbers"),
            (0.5, 0.5, "greater than current_value"),
            (0.6, 0.5, "greater than current_value"),
        )
        for current_value, candidate_value, issue_text in invalid:
            with self.subTest(
                current_value=current_value,
                candidate_value=candidate_value,
            ):
                result = dispatcher_module._validate_bot_tuning_review(
                    review(current_value, candidate_value),
                    selected_event=selected,
                )
                self.assertEqual(result["status"], "INVALID_INCOMPLETE_REVIEW")
                self.assertNotIn("review", result)
                self.assertTrue(any(issue_text in issue for issue in result["issues"]))

        for current_value, candidate_value in ((0.0, 0.1), (0.9, 1.0)):
            with self.subTest(valid=(current_value, candidate_value)):
                result = dispatcher_module._validate_bot_tuning_review(
                    review(current_value, candidate_value),
                    selected_event=selected,
                )
                self.assertEqual(result["status"], "VALID")

        no_change = _valid_no_change_tuning_review("EUR_USD")
        validated_no_change = dispatcher_module._validate_bot_tuning_review(
            no_change,
            selected_event=selected,
        )
        self.assertEqual(validated_no_change["status"], "VALID")
        self.assertEqual(
            validated_no_change["review"]["evidence_acquisition"]["required_new_samples"],
            20,
        )

        vague_no_change = {
            **no_change,
            "evidence_acquisition": {
                "action_kind": "WAIT",
                "source_ref": "later",
                "required_new_samples": 0,
                "success_condition": "wait",
            },
        }
        vague_result = dispatcher_module._validate_bot_tuning_review(
            vague_no_change,
            selected_event=selected,
        )
        self.assertEqual(vague_result["status"], "INVALID_INCOMPLETE_REVIEW")
        self.assertNotIn("review", vague_result)

        vague_phrase = {
            **no_change,
            "evidence_acquisition": {
                **no_change["evidence_acquisition"],
                "success_condition": (
                    "wait and monitor later until enough evidence appears"
                ),
            },
        }
        vague_phrase_result = dispatcher_module._validate_bot_tuning_review(
            vague_phrase,
            selected_event=selected,
        )
        self.assertEqual(
            vague_phrase_result["status"],
            "INVALID_INCOMPLETE_REVIEW",
        )

        concrete_count = {
            **no_change,
            "evidence_acquisition": {
                **no_change["evidence_acquisition"],
                "success_condition": (
                    "pass when sample count reaches exactly 20 canonical attributed entries"
                ),
            },
        }
        concrete_result = dispatcher_module._validate_bot_tuning_review(
            concrete_count,
            selected_event=selected,
        )
        self.assertEqual(concrete_result["status"], "VALID")

    def test_test_required_review_precommits_exact_lane(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")
        missing = _valid_tuning_review("EUR_USD")
        missing["proposed_adjustments"][0].pop("lane_id")
        missing_result = dispatcher_module._validate_bot_tuning_review(
            missing,
            selected_event=selected,
        )
        self.assertEqual(missing_result["status"], "INVALID_INCOMPLETE_REVIEW")

        event_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:LIMIT"
        selected_with_lane = {
            **selected,
            "details": {**selected.get("details", {}), "lane_id": event_lane},
        }
        mismatched_result = dispatcher_module._validate_bot_tuning_review(
            _valid_tuning_review("EUR_USD"),
            selected_event=selected_with_lane,
        )
        self.assertEqual(mismatched_result["status"], "INVALID_UNSAFE_BOUNDARY")
        matching_result = dispatcher_module._validate_bot_tuning_review(
            _valid_tuning_review("EUR_USD", lane_id=event_lane),
            selected_event=selected_with_lane,
        )
        self.assertEqual(matching_result["status"], "VALID")

    def test_more_than_twelve_adjustments_is_rejected_without_unvalidated_tail(self) -> None:
        selected = _technical_tuning_event(pair="EUR_USD")
        family, parameter = _supported_tuning_parameter()
        review = {
            **_valid_tuning_review("EUR_USD"),
            "affected_bot_families": [family],
            "proposed_adjustments": [
                {
                    "pair": "EUR_USD",
                    "bot_family": family,
                    "parameter": parameter,
                    "current_value": 0.6,
                    "candidate_value": 0.61,
                    "rationale": "frozen-cohort score comparison",
                }
                for index in range(13)
            ],
        }

        result = dispatcher_module._validate_bot_tuning_review(
            review,
            selected_event=selected,
        )

        self.assertEqual(result["status"], "INVALID_INCOMPLETE_REVIEW")
        self.assertNotIn("review", result)

    def test_safe_structured_bot_tuning_review_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            receipt = json.loads(_valid_receipt())
            receipt["bot_tuning_review"] = _valid_tuning_review("EUR_USD")

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, json.dumps(receipt)),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            work_order = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(work_order["bot_tuning_review_validation"]["status"], "VALID")
            self.assertEqual(work_order["bot_tuning_review"]["affected_pairs"], ["EUR_USD"])
            self.assertEqual(
                work_order["bot_tuning_review"]["proposed_adjustments"][0]["parameter"],
                _supported_tuning_parameter()[1],
            )
            self.assertFalse(work_order["bot_tuning_review"]["live_permission_allowed"])

        execution_review = {
            **_valid_tuning_review("EUR_USD"),
            "affected_bot_families": ["execution"],
            "proposed_adjustments": [
                {
                    "pair": "EUR_USD",
                    "bot_family": "execution",
                    "parameter": "spread_score_weight",
                    "current_value": 0.4,
                    "candidate_value": 0.5,
                    "rationale": "compare cost-score ranking on frozen quotes",
                }
            ],
        }
        validated = dispatcher_module._validate_bot_tuning_review(
            execution_review,
            selected_event=_technical_tuning_event(pair="EUR_USD"),
        )
        self.assertEqual(validated["status"], "INVALID_INCOMPLETE_REVIEW")

    def test_same_observation_can_upgrade_no_change_review_but_never_ack_unsafe_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="review-upgrade")
            no_change = _valid_no_change_tuning_review("EUR_USD")
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": no_change},
                now=NOW,
            )
            self.assertEqual(first["status"], "WORK_ORDER_WRITTEN")

            upgraded = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(upgraded["status"], "WORK_ORDER_REVIEW_ENRICHED")
            self.assertEqual(
                json.loads(path.read_text())["bot_tuning_review"]["review_status"],
                "TEST_REQUIRED",
            )

            unsafe = {
                **_valid_tuning_review("EUR_USD"),
                "live_permission_allowed": True,
            }
            rejected = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": unsafe},
                now=NOW + timedelta(minutes=2),
            )
            self.assertEqual(rejected["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertEqual(
                json.loads(path.read_text())["bot_tuning_review"]["review_status"],
                "TEST_REQUIRED",
            )

            downgrade = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": no_change},
                now=NOW + timedelta(minutes=3),
            )
            self.assertEqual(downgrade["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertEqual(
                downgrade["bot_tuning_review_status"],
                "REVIEW_DOWNGRADE_FORBIDDEN",
            )
            self.assertEqual(
                json.loads(path.read_text())["bot_tuning_review"]["review_status"],
                "TEST_REQUIRED",
            )

    def test_new_observation_may_replace_old_test_with_current_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            first_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="test-required-old-observation",
            )
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            self.assertEqual(first["status"], "WORK_ORDER_WRITTEN")

            no_change = _valid_no_change_tuning_review("EUR_USD")
            new_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="no-change-new-observation",
                mid=0.581025,
                reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",),
            )
            refreshed = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=new_event,
                receipt={"bot_tuning_review": no_change},
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(refreshed["status"], "WORK_ORDER_OBSERVATION_APPENDED")
            payload = json.loads(path.read_text())
            self.assertNotEqual(payload["latest_observation_id"], first["observation_id"])
            self.assertEqual(
                payload["latest_reviewed_observation_id"],
                payload["latest_observation_id"],
            )
            self.assertEqual(
                payload["bot_tuning_review"]["review_status"],
                "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            )

    def test_hourly_enrichment_persists_idempotent_no_change_then_only_upgrades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="hourly-no-change-review",
            )
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={},
                now=NOW,
            )
            self.assertEqual(created["status"], "STRUCTURED_REVIEW_REQUIRED")
            no_change = _valid_no_change_tuning_review("EUR_USD")

            unsafe = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review={**no_change, "live_permission_allowed": True},
                reviewed_by="qr-trader-hourly",
                now=NOW + timedelta(seconds=10),
            )
            self.assertEqual(unsafe["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertNotIn("bot_tuning_review", json.loads(path.read_text()))

            bound_at = NOW + timedelta(minutes=1)
            bound = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=no_change,
                reviewed_by="qr-trader-hourly",
                now=bound_at,
            )
            self.assertEqual(bound["status"], "WORK_ORDER_REVIEW_ENRICHED")
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["bot_tuning_review"]["review_status"],
                "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            )
            self.assertEqual(
                payload["latest_reviewed_observation_id"],
                created["observation_id"],
            )
            self.assertEqual(
                payload["structured_review_completed_at_utc"],
                bound_at.isoformat(),
            )
            self.assertEqual(
                payload["structured_review_completed_by"],
                "qr-trader-hourly",
            )

            bound_bytes = path.read_bytes()
            repeated = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=no_change,
                reviewed_by="another-hourly-retry",
                now=NOW + timedelta(minutes=2),
            )
            self.assertEqual(repeated["status"], "WORK_ORDER_REVIEW_ALREADY_BOUND")
            self.assertEqual(path.read_bytes(), bound_bytes)

            upgraded_at = NOW + timedelta(minutes=3)
            upgraded = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=_valid_tuning_review("EUR_USD"),
                reviewed_by="qr-trader-hourly",
                now=upgraded_at,
            )
            self.assertEqual(upgraded["status"], "WORK_ORDER_REVIEW_ENRICHED")
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["bot_tuning_review"]["review_status"],
                "TEST_REQUIRED",
            )
            self.assertEqual(
                payload["structured_review_completed_at_utc"],
                upgraded_at.isoformat(),
            )

            upgraded_bytes = path.read_bytes()
            downgrade = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=no_change,
                reviewed_by="qr-trader-hourly",
                now=NOW + timedelta(minutes=4),
            )
            self.assertEqual(downgrade["status"], "WORK_ORDER_REVIEW_CONFLICT")
            self.assertEqual(downgrade["current_review_status"], "TEST_REQUIRED")
            self.assertEqual(
                downgrade["incoming_review_status"],
                "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            )
            self.assertEqual(path.read_bytes(), upgraded_bytes)

            new_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="hourly-no-change-new-observation",
                mid=0.581025,
                reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",),
            )
            appended = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=new_event,
                receipt={},
                now=NOW + timedelta(minutes=5),
            )
            self.assertEqual(appended["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertNotEqual(appended["observation_id"], created["observation_id"])

            rebound = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=appended["observation_id"],
                review=no_change,
                reviewed_by="qr-trader-hourly",
                now=NOW + timedelta(minutes=6),
            )
            self.assertEqual(rebound["status"], "WORK_ORDER_REVIEW_ENRICHED")
            rebound_payload = json.loads(path.read_text())
            self.assertEqual(
                rebound_payload["latest_reviewed_observation_id"],
                appended["observation_id"],
            )
            self.assertEqual(
                rebound_payload["bot_tuning_review"]["review_status"],
                "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            )

    def test_hourly_review_enrich_cli_accepts_safe_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="hourly-no-change-cli",
            )
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={},
                now=NOW,
            )
            review_path = root / "review.json"
            review_path.write_text(
                json.dumps(
                    _valid_no_change_tuning_review("EUR_USD")
                )
            )
            output = io.StringIO()
            real_write_queue = dispatcher_module._write_tuning_queue_json

            with (
                patch.object(
                    dispatcher_module,
                    "_write_tuning_queue_json",
                    wraps=real_write_queue,
                ) as write_queue,
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(path),
                        "--work-order-id",
                        created["work_order_id"],
                        "--expected-observation-id",
                        created["observation_id"],
                        "--review-json",
                        str(review_path),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.getvalue())["status"],
                "WORK_ORDER_REVIEW_ENRICHED",
            )
            self.assertEqual(
                json.loads(path.read_text())["bot_tuning_review"]["review_status"],
                "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            )
            self.assertEqual(write_queue.call_count, 1)

    def test_hourly_review_enrich_batch_prevalidates_then_binds_every_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created: list[tuple[str, dict]] = []
            for index, pair in enumerate(("EUR_USD", "USD_JPY")):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(
                        pair=pair,
                        event_id=f"hourly-batch-{index}",
                    ),
                    receipt={},
                    now=NOW + timedelta(seconds=index),
                )
                self.assertEqual(result["status"], "STRUCTURED_REVIEW_REQUIRED")
                created.append((pair, result))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "work_order_id": result["work_order_id"],
                                "expected_observation_id": result["observation_id"],
                                "review": _valid_no_change_tuning_review(pair),
                            }
                            for pair, result in created
                        ]
                    }
                )
            )
            output = io.StringIO()
            real_write_queue = dispatcher_module._write_tuning_queue_json

            with (
                patch.object(
                    dispatcher_module,
                    "_write_tuning_queue_json",
                    wraps=real_write_queue,
                ) as write_queue,
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(path),
                        "--manifest-json",
                        str(manifest_path),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            summary = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "BATCH_REVIEW_ENRICHED")
            self.assertEqual(summary["requested_count"], 2)
            self.assertEqual(summary["prevalidated_count"], 2)
            self.assertEqual(summary["success_count"], 2)
            self.assertEqual(summary["enriched_count"], 2)
            self.assertEqual(summary["failure_count"], 0)
            self.assertEqual(summary["queue_write_count"], 1)
            self.assertEqual(write_queue.call_count, 1)
            payload = json.loads(path.read_text())
            self.assertEqual(
                [
                    item["bot_tuning_review_validation"]["status"]
                    for item in payload["work_orders"]
                ],
                ["VALID", "VALID"],
            )
            self.assertEqual(
                {
                    item["latest_reviewed_observation_id"]
                    for item in payload["work_orders"]
                },
                {result["observation_id"] for _, result in created},
            )

    def test_hourly_review_enrich_batch_invalid_review_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created: list[tuple[str, dict]] = []
            for index, pair in enumerate(("EUR_USD", "USD_JPY")):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(
                        pair=pair,
                        event_id=f"hourly-batch-invalid-{index}",
                    ),
                    receipt={},
                    now=NOW + timedelta(seconds=index),
                )
                created.append((pair, result))
            valid_review = _valid_no_change_tuning_review("EUR_USD")
            invalid_review = _valid_no_change_tuning_review("USD_JPY")
            invalid_review.pop("evidence_acquisition")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "work_order_id": created[0][1]["work_order_id"],
                                "expected_observation_id": created[0][1]["observation_id"],
                                "review": valid_review,
                            },
                            {
                                "work_order_id": created[1][1]["work_order_id"],
                                "expected_observation_id": created[1][1]["observation_id"],
                                "review": invalid_review,
                            },
                        ]
                    }
                )
            )
            before = path.read_bytes()
            output = io.StringIO()

            with (
                patch.object(dispatcher_module, "_write_tuning_queue_json") as write_queue,
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(path),
                        "--manifest-json",
                        str(manifest_path),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            summary = json.loads(output.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "BATCH_MANIFEST_VALIDATION_FAILED")
            self.assertEqual(summary["written_count"], 0)
            self.assertIn(
                "STRUCTURED_REVIEW_REQUIRED",
                {failure["code"] for failure in summary["failures"]},
            )
            self.assertEqual(path.read_bytes(), before)
            write_queue.assert_not_called()

    def test_hourly_review_enrich_batch_rejects_duplicate_or_stale_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="hourly-batch-identity",
                ),
                receipt={},
                now=NOW,
            )
            review = _valid_no_change_tuning_review("EUR_USD")
            item = {
                "work_order_id": created["work_order_id"],
                "expected_observation_id": created["observation_id"],
                "review": review,
            }
            before = path.read_bytes()

            duplicate = tuning_review_enrich_tool._prevalidate_batch_manifest(
                path=path,
                manifest={"reviews": [item, item]},
            )
            stale = tuning_review_enrich_tool._prevalidate_batch_manifest(
                path=path,
                manifest={
                    "reviews": [
                        {
                            **item,
                            "expected_observation_id": "0" * 64,
                        }
                    ]
                },
            )

            self.assertEqual(duplicate["status"], "BATCH_MANIFEST_VALIDATION_FAILED")
            self.assertIn(
                "DUPLICATE_WORK_ORDER_ID",
                {failure["code"] for failure in duplicate["failures"]},
            )
            self.assertEqual(stale["status"], "BATCH_MANIFEST_VALIDATION_FAILED")
            self.assertIn(
                "WORK_ORDER_OBSERVATION_STALE",
                {failure["code"] for failure in stale["failures"]},
            )
            self.assertEqual(path.read_bytes(), before)

    def test_hourly_review_enrich_batch_mixes_already_bound_and_new_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created: list[tuple[str, dict, dict]] = []
            for index, pair in enumerate(("EUR_USD", "USD_JPY")):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(
                        pair=pair,
                        event_id=f"hourly-batch-mixed-{index}",
                    ),
                    receipt={},
                    now=NOW + timedelta(seconds=index),
                )
                review = _valid_no_change_tuning_review(pair)
                created.append((pair, result, review))
            first = dispatcher_module.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created[0][1]["work_order_id"],
                expected_observation_id=created[0][1]["observation_id"],
                review=created[0][2],
                reviewed_by="qr-trader-hourly",
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(first["status"], "WORK_ORDER_REVIEW_ENRICHED")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "work_order_id": result["work_order_id"],
                                "expected_observation_id": result["observation_id"],
                                "review": review,
                            }
                            for _, result, review in created
                        ]
                    }
                )
            )
            before = path.read_bytes()
            output = io.StringIO()
            real_write_queue = dispatcher_module._write_tuning_queue_json

            with (
                patch.object(
                    dispatcher_module,
                    "_write_tuning_queue_json",
                    wraps=real_write_queue,
                ) as write_queue,
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(path),
                        "--manifest-json",
                        str(manifest_path),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            summary = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "BATCH_REVIEW_ENRICHED")
            self.assertEqual(summary["enriched_count"], 1)
            self.assertEqual(summary["already_bound_count"], 1)
            self.assertEqual(summary["queue_write_count"], 1)
            self.assertEqual(write_queue.call_count, 1)
            self.assertNotEqual(path.read_bytes(), before)
            payload = json.loads(path.read_text())
            self.assertEqual(
                {
                    item["latest_reviewed_observation_id"]
                    for item in payload["work_orders"]
                },
                {result["observation_id"] for _, result, _ in created},
            )

    def test_hourly_review_enrich_batch_revalidates_stale_item_without_partial_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created: list[tuple[str, dict, dict]] = []
            for index, pair in enumerate(("USD_JPY", "EUR_USD")):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(
                        pair=pair,
                        event_id=f"hourly-batch-revalidate-{index}",
                    ),
                    receipt={},
                    now=NOW + timedelta(seconds=index),
                )
                review = _valid_no_change_tuning_review(pair)
                created.append((pair, result, review))
            manifest = {
                "reviews": [
                    {
                        "work_order_id": result["work_order_id"],
                        "expected_observation_id": result["observation_id"],
                        "review": review,
                    }
                    for _, result, review in created
                ]
            }
            preview = tuning_review_enrich_tool._prevalidate_batch_manifest(
                path=path,
                manifest=manifest,
            )
            self.assertEqual(preview["status"], "VALID")
            advanced = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="hourly-batch-revalidate-new-observation",
                    mid=0.581025,
                    reasons=("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",),
                ),
                receipt={},
                now=NOW + timedelta(minutes=5),
            )
            self.assertNotEqual(
                advanced["observation_id"],
                created[1][1]["observation_id"],
            )
            before = path.read_bytes()

            with patch.object(
                dispatcher_module, "_write_tuning_queue_json"
            ) as write_queue:
                result = dispatcher_module.enrich_tuning_work_order_reviews_batch(
                    path=path,
                    reviews=preview["reviews"],
                    reviewed_by="qr-trader-hourly",
                    now=NOW + timedelta(minutes=6),
                )

            self.assertEqual(result["status"], "BATCH_MANIFEST_VALIDATION_FAILED")
            self.assertEqual(result["written_count"], 0)
            self.assertIn(
                "WORK_ORDER_OBSERVATION_STALE",
                {failure["status"] for failure in result["failures"]},
            )
            write_queue.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_hourly_review_enrich_batch_write_failure_preserves_queue_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            manifest_path = root / "manifest.json"
            reviews = []
            for index, pair in enumerate(("EUR_USD", "USD_JPY")):
                created = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=_technical_tuning_event(
                        pair=pair,
                        event_id=f"hourly-batch-write-failure-{index}",
                    ),
                    receipt={},
                    now=NOW + timedelta(seconds=index),
                )
                reviews.append(
                    {
                        "work_order_id": created["work_order_id"],
                        "expected_observation_id": created["observation_id"],
                        "review": _valid_no_change_tuning_review(pair),
                    }
                )
            manifest_path.write_text(json.dumps({"reviews": reviews}))
            before = path.read_bytes()
            output = io.StringIO()
            real_write_queue = dispatcher_module._write_tuning_queue_json

            with (
                patch.object(
                    dispatcher_module,
                    "_write_tuning_queue_json",
                    wraps=real_write_queue,
                ) as write_queue,
                patch.object(
                    dispatcher_module.os,
                    "replace",
                    side_effect=OSError("ENOSPC"),
                ),
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(path),
                        "--manifest-json",
                        str(manifest_path),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            summary = json.loads(output.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "WORK_ORDER_WRITE_FAILED")
            self.assertEqual(summary["written_count"], 0)
            self.assertEqual(summary["queue_write_count"], 0)
            self.assertEqual(write_queue.call_count, 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tuning.tmp")),
                [],
            )

    def test_hourly_review_enrich_batch_rejects_post_merge_oversize_without_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="hourly-batch-post-merge-size",
                ),
                receipt={},
                now=NOW,
            )
            review = _valid_no_change_tuning_review("EUR_USD")
            before = path.read_bytes()

            with patch.object(
                dispatcher_module,
                "MAX_TUNING_QUEUE_BYTES",
                len(before) + 1,
            ):
                result = dispatcher_module.enrich_tuning_work_order_reviews_batch(
                    path=path,
                    reviews=[
                        {
                            "work_order_id": created["work_order_id"],
                            "expected_observation_id": created["observation_id"],
                            "review": review,
                        }
                    ],
                    reviewed_by="qr-trader-hourly",
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(result["status"], "WORK_ORDER_WRITE_FAILED")
            self.assertEqual(result["written_count"], 0)
            self.assertEqual(result["queue_write_count"], 0)
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn(
                "_read_error",
                dispatcher_module._load_tuning_work_order(path),
            )
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tuning.tmp")),
                [],
            )

    def test_hourly_review_enrich_batch_lock_contention_preserves_queue_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "guardian_tuning_work_order.json"
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="hourly-batch-lock-contention",
                ),
                receipt={},
                now=NOW,
            )
            review = _valid_no_change_tuning_review("EUR_USD")
            before = path.read_bytes()
            lock_path = path.with_name(f"{path.name}.lock")

            with lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(
                    dispatcher_module, "_write_tuning_queue_json"
                ) as write_queue:
                    result = dispatcher_module.enrich_tuning_work_order_reviews_batch(
                        path=path,
                        reviews=[
                            {
                                "work_order_id": created["work_order_id"],
                                "expected_observation_id": created["observation_id"],
                                "review": review,
                            }
                        ],
                        reviewed_by="qr-trader-hourly",
                        now=NOW + timedelta(minutes=1),
                    )

            self.assertEqual(result["status"], "WORK_ORDER_CONCURRENT_UPDATE")
            self.assertEqual(result["written_count"], 0)
            write_queue.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_hourly_review_enrich_manifest_reader_enforces_size_and_shape_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * (dispatcher_module.MAX_TUNING_QUEUE_BYTES + 1))

            with patch.object(tuning_review_enrich_tool.json, "loads") as loads:
                manifest, error = tuning_review_enrich_tool._read_batch_manifest_json(
                    oversized
                )

            self.assertIsNone(manifest)
            self.assertEqual(error["status"], "MANIFEST_JSON_TOO_LARGE")
            loads.assert_not_called()

            output = io.StringIO()
            with (
                patch.object(
                    tuning_review_enrich_tool,
                    "_prevalidate_batch_manifest",
                ) as prevalidate,
                redirect_stdout(output),
            ):
                code = tuning_review_enrich_tool.main(
                    [
                        "--path",
                        str(root / "queue.json"),
                        "--manifest-json",
                        str(oversized),
                        "--reviewed-by",
                        "qr-trader-hourly",
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(
                json.loads(output.getvalue())["status"],
                "MANIFEST_JSON_TOO_LARGE",
            )
            prevalidate.assert_not_called()

            too_deep = root / "too-deep.json"
            nested: dict = {"leaf": True}
            for _ in range(dispatcher_module.MAX_TUNING_QUEUE_JSON_DEPTH + 1):
                nested = {"nested": nested}
            too_deep.write_text(json.dumps({"reviews": [{"review": nested}]}))

            manifest, error = tuning_review_enrich_tool._read_batch_manifest_json(
                too_deep
            )

            self.assertIsNone(manifest)
            self.assertEqual(error["status"], "MANIFEST_JSON_SHAPE_INVALID")

            lone_surrogate = root / "lone-surrogate.json"
            lone_surrogate.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "review": {
                                    "hypothesis": "bad\ud800text",
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest, error = tuning_review_enrich_tool._read_batch_manifest_json(
                lone_surrogate
            )

            self.assertIsNone(manifest)
            self.assertEqual(error["status"], "MANIFEST_JSON_SHAPE_INVALID")

            oversized_string = root / "oversized-string.json"
            oversized_string.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "review": {
                                    "hypothesis": "x"
                                    * (
                                        dispatcher_module.MAX_TUNING_QUEUE_STRING_CHARS
                                        + 1
                                    )
                                }
                            }
                        ]
                    }
                )
            )

            manifest, error = tuning_review_enrich_tool._read_batch_manifest_json(
                oversized_string
            )

            self.assertIsNone(manifest)
            self.assertEqual(error["status"], "MANIFEST_JSON_SHAPE_INVALID")

    def test_tuning_work_order_write_failure_is_retried_without_accepted_ack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            real_write_queue = dispatcher_module._write_tuning_queue_json

            def fail_only_work_order(path, payload, **kwargs):
                if path == paths.tuning_work_order:
                    raise OSError("ENOSPC")
                return real_write_queue(path, payload, **kwargs)

            with patch.object(
                dispatcher_module,
                "_write_tuning_queue_json",
                side_effect=fail_only_work_order,
            ):
                first = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                    subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
                )

            first_state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(first["status"], "TUNING_HANDOFF_FAILED")
            self.assertFalse(first["receipt_written"])
            self.assertFalse(paths.tuning_work_order.exists())
            self.assertNotIn(_event(severity="P1")["dedupe_key"], first_state.get("reviewed_events", {}))
            self.assertIn(_event(severity="P1")["dedupe_key"], first_state["dispatch_attempts"])

            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.tuning_work_order.exists())
            self.assertIn(
                _event(severity="P1")["dedupe_key"],
                json.loads(paths.dispatcher_state.read_text())["reviewed_events"],
            )

    def test_full_tuning_queue_is_retried_without_dropping_pending_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            for index in range(dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS):
                queued_event = {
                    **_event(
                        severity="P2",
                        event_id=f"queued-{index}",
                        pair=f"PAIR_{index:02d}",
                        dedupe_key=f"PAIR_{index:02d}|queued|FAILED_ACCEPTANCE|HOLD",
                    ),
                    "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                }
                written = dispatcher_module._maybe_write_tuning_work_order(
                    path=paths.tuning_work_order,
                    selected_event=queued_event,
                    receipt={
                        "bot_tuning_review": _valid_tuning_review(f"PAIR_{index:02d}")
                    },
                    now=NOW + timedelta(seconds=index),
                )
                self.assertEqual(written["status"], "WORK_ORDER_WRITTEN")
            before = paths.tuning_work_order.read_text()
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            self.assertEqual(result["status"], "TUNING_HANDOFF_FAILED")
            self.assertEqual(result["tuning_handoff"]["status"], "WORK_ORDER_QUEUE_FULL")
            self.assertEqual(paths.tuning_work_order.read_text(), before)
            state = json.loads(paths.dispatcher_state.read_text())
            selected_key = _event(severity="P1")["dedupe_key"]
            self.assertNotIn(selected_key, state.get("reviewed_events", {}))
            self.assertIn(selected_key, state["dispatch_attempts"])

    def test_queue_full_newer_same_dedupe_preserves_exact_pending_and_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            _fill_tuning_queue(paths.tuning_work_order)
            calls: list[list[str]] = []
            original = _event(severity="P1")

            first = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "60"},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            newer = {
                **original,
                "event_id": "event-new-tick",
                "price_zone": "EUR_USD 1.1725 rejection",
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }
            paths.events.write_text(json.dumps({"events": [newer]}))
            paths.escalation.write_text(
                json.dumps({"wake_gpt": True, "events_to_review": [newer]})
            )

            blocked = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=31),
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "60"},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id="event-new-tick", tuning_review=True),
                ),
            )

            state = json.loads(paths.dispatcher_state.read_text())
            key = original["dedupe_key"]
            self.assertEqual(first["status"], "TUNING_HANDOFF_FAILED")
            self.assertEqual(blocked["status"], "SUPPRESSED")
            self.assertEqual(blocked["selection"]["suppressed"][0]["reason"], "RETRY_BACKOFF")
            self.assertEqual(len(calls), 1)
            self.assertEqual(state["dispatch_attempts"][key]["attempt_count"], 1)
            self.assertEqual(state["dispatch_attempts"][key]["event_id"], original["event_id"])
            self.assertEqual(state["pending_dispatches"][key]["event"]["event_id"], original["event_id"])
            successors = state["pending_dispatches"][key]["successors"]
            self.assertEqual(len(successors), 1)
            self.assertEqual(successors[0]["event_id"], newer["event_id"])

    def test_new_material_same_dedupe_promotes_after_old_retry_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            original = _event(severity="P1")
            env = {"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "60"}

            failed = run_dispatcher(
                paths=paths,
                now=NOW,
                env=env,
                subprocess_run=_fake_codex_sequence(calls, ["bad", "bad"]),
            )
            successor = {
                **original,
                "event_id": "event-material-successor",
                "price_zone": "EUR_USD 1.1725 rejection",
                "wake_reason_codes": ["NEW_EVENT"],
            }
            paths.events.write_text(json.dumps({"events": [successor]}))
            paths.escalation.write_text(
                json.dumps({"wake_gpt": True, "events_to_review": [successor]})
            )

            blocked = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env=env,
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id=successor["event_id"]),
                ),
            )
            blocked_state = json.loads(paths.dispatcher_state.read_text())
            key = original["dedupe_key"]
            self.assertEqual(failed["status"], "PARSE_FAILED")
            self.assertEqual(blocked["status"], "SUPPRESSED")
            self.assertEqual(blocked_state["dispatch_attempts"][key]["attempt_count"], 1)
            self.assertEqual(blocked_state["pending_dispatches"][key]["event_id"], original["event_id"])
            self.assertEqual(
                [item["event_id"] for item in blocked_state["pending_dispatches"][key]["successors"]],
                [successor["event_id"]],
            )

            old_recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=61),
                env=env,
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id=original["event_id"]),
                ),
            )
            promoted_state = json.loads(paths.dispatcher_state.read_text())
            promoted = promoted_state["pending_dispatches"][key]
            self.assertEqual(old_recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(old_recovered["selected_event"]["event_id"], original["event_id"])
            self.assertEqual(promoted["event_id"], successor["event_id"])
            self.assertTrue(promoted["promoted_successor"])
            self.assertEqual(promoted_state["dispatch_attempts"], {})

            paths.events.write_text(json.dumps({"events": []}))
            paths.escalation.write_text(
                json.dumps({"wake_gpt": False, "events_to_review": []})
            )
            successor_recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=62),
                env=env,
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(event_id=successor["event_id"]),
                ),
            )
            final_state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(successor_recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(
                successor_recovered["selected_event"]["event_id"],
                successor["event_id"],
            )
            self.assertEqual(final_state["pending_dispatches"], {})
            self.assertEqual(len(calls), 4)

    def test_pending_successor_queue_is_bounded_and_keeps_latest_material_events(self) -> None:
        original = _event(severity="P1")
        record = {
            "event": original,
            "event_id": original["event_id"],
            "material_fingerprint": dispatcher_module._event_material_fingerprint(original),
            "queued_at_utc": NOW.isoformat(),
            "last_seen_at_utc": NOW.isoformat(),
            "expires_at_utc": (NOW + timedelta(hours=2)).isoformat(),
        }
        successor_ids: list[str] = []
        for index in range(dispatcher_module.MAX_PENDING_SUCCESSORS_PER_DEDUPE + 3):
            successor = {
                **original,
                "event_id": f"successor-{index}",
                "price_zone": f"EUR_USD {1.17 + index * 0.0001:.4f} rejection",
                "wake_reason_codes": ["NEW_EVENT"],
            }
            successor_ids.append(successor["event_id"])
            record, _ = dispatcher_module._enqueue_pending_dispatch_successor(
                record,
                event=successor,
                now=NOW + timedelta(seconds=index + 1),
                ttl_seconds=3600,
            )

        self.assertEqual(
            len(record["successors"]),
            dispatcher_module.MAX_PENDING_SUCCESSORS_PER_DEDUPE,
        )
        self.assertEqual(
            [item["event_id"] for item in record["successors"]],
            successor_ids[-dispatcher_module.MAX_PENDING_SUCCESSORS_PER_DEDUPE :],
        )
        self.assertEqual(record["successor_overflow_count"], 3)

    def test_queue_slot_recovery_releases_exhausted_retry_and_preserves_terminal_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            _fill_tuning_queue(paths.tuning_work_order)
            calls: list[list[str]] = []
            failed = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=30),
                env={
                    "QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1",
                    "QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS": "1",
                },
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            failed_state = json.loads(paths.dispatcher_state.read_text())
            selected_key = _event(severity="P1")["dedupe_key"]
            self.assertTrue(failed_state["dispatch_attempts"][selected_key]["retry_budget_exhausted"])

            queue = json.loads(paths.tuning_work_order.read_text())
            terminal = queue["work_orders"][-1]
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=paths.tuning_work_order,
                work_order_id=terminal["work_order_id"],
                observation_id=terminal["latest_observation_id"],
                experiment_id="slot-release-experiment",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            terminal_result = dispatcher_module.transition_tuning_work_order(
                path=paths.tuning_work_order,
                work_order_id=terminal["work_order_id"],
                expected_observation_id=terminal["latest_observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly-ai",
                experiment_id="slot-release-experiment",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(seconds=31),
            )
            self.assertEqual(terminal_result["status"], "WORK_ORDER_TERMINAL_WRITTEN")
            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))

            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=32),
                env={
                    "QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1",
                    "QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS": "1",
                },
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            payload = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(failed["tuning_handoff"]["status"], "WORK_ORDER_QUEUE_FULL")
            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertEqual(
                recovered["tuning_queue_recovery"]["status"],
                "WORK_ORDER_QUEUE_CAPACITY_RECOVERED",
            )
            self.assertEqual(payload["pending_count"], dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS)
            self.assertTrue(
                any(
                    item.get("experiment_id") == "slot-release-experiment"
                    for item in payload["terminal_history"]
                )
            )

    def test_queue_full_skips_enabled_gateway_action_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            _fill_tuning_queue(paths.tuning_work_order)
            codex_calls: list[list[str]] = []
            action_calls: list[list[str]] = []

            def fake_action_cycle(cmd, **kwargs):
                action_calls.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="{}", stderr="")

            with patch("tools.guardian_wake_dispatcher.subprocess.run", side_effect=fake_action_cycle):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW + timedelta(seconds=30),
                    env={
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        "QR_LIVE_ENABLED": "1",
                    },
                    subprocess_run=_fake_codex(codex_calls, _valid_receipt(tuning_review=True)),
                )

            self.assertEqual(result["status"], "TUNING_HANDOFF_FAILED")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_TUNING_HANDOFF_FAILED")
            self.assertEqual(action_calls, [])

    def test_corrupt_tuning_queue_never_releases_budget_or_overwrites_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            corrupt = b"{partial full queue"
            paths.tuning_work_order.write_bytes(corrupt)
            event = _event(severity="P1")
            paths.dispatcher_state.write_text(
                json.dumps(
                    {
                        "dispatch_attempts": {
                            event["dedupe_key"]: {
                                "event_id": event["event_id"],
                                "dedupe_key": event["dedupe_key"],
                                "event": event,
                                "material_fingerprint": dispatcher_module._event_material_fingerprint(event),
                                "attempt_count": 3,
                                "max_attempts": 3,
                                "last_error": "WORK_ORDER_QUEUE_FULL",
                                "last_status": "TUNING_HANDOFF_FAILED",
                                "retry_after_utc": (NOW + timedelta(minutes=30)).isoformat(),
                                "expires_at_utc": (NOW + timedelta(hours=1)).isoformat(),
                                "retry_budget_exhausted": True,
                            }
                        }
                    }
                )
            )
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            direct = dispatcher_module._maybe_write_tuning_work_order(
                path=paths.tuning_work_order,
                selected_event={**event, "wake_reason_codes": ["REGIME_STATE_CHANGE"]},
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )

            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(result["tuning_queue_recovery"]["status"], "WORK_ORDER_QUEUE_READ_FAILED")
            self.assertEqual(direct["status"], "WORK_ORDER_READ_FAILED")
            self.assertEqual(state["dispatch_attempts"][event["dedupe_key"]]["attempt_count"], 3)
            self.assertEqual(calls, [])
            self.assertEqual(paths.tuning_work_order.read_bytes(), corrupt)

    def test_structurally_invalid_queue_is_not_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            malformed = {
                "schema_version": 2,
                "queue_schema_revision": 3,
                "work_orders": "not-a-list",
                "pending_count": 20,
            }
            path.write_text(json.dumps(malformed))
            event = _technical_tuning_event(pair="EUR_USD", event_id="structural-corrupt")

            result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )

            self.assertEqual(result["status"], "WORK_ORDER_READ_FAILED")
            self.assertEqual(json.loads(path.read_text()), malformed)

    def test_revision4_queue_cannot_be_downgraded_by_deleting_revision_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="revision-strip")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            payload = json.loads(path.read_text())
            payload.pop("queue_schema_revision")
            path.write_text(json.dumps(payload))
            before = path.read_bytes()

            loaded = dispatcher_module._load_tuning_work_order(path)
            write_result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="GBP_USD",
                    event_id="revision-strip-followup",
                ),
                receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                now=NOW + timedelta(minutes=1),
            )

            self.assertIn("_read_error", loaded)
            self.assertIn("downgrade", loaded["_read_error"])
            self.assertEqual(write_result["status"], "WORK_ORDER_READ_FAILED")
            self.assertEqual(path.read_bytes(), before)

    def test_tuning_queue_raw_and_collection_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="bounded-queue")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            valid = json.loads(path.read_text())

            def too_many_work_orders(payload: dict) -> None:
                payload["work_orders"] = [
                    {**payload["work_orders"][0], "work_order_id": f"work-{index}"}
                    for index in range(dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS + 1)
                ]
                payload["pending_count"] = len(payload["work_orders"])

            def too_much_terminal_history(payload: dict) -> None:
                payload["terminal_history"] = [
                    {"work_order_id": f"terminal-{index}"}
                    for index in range(dispatcher_module.MAX_TUNING_TERMINAL_HISTORY + 1)
                ]
                payload["terminal_history_count"] = len(payload["terminal_history"])

            def too_many_observations(payload: dict) -> None:
                observations = [
                    {"observation_id": f"observation-{index}"}
                    for index in range(
                        dispatcher_module.MAX_TUNING_OBSERVATIONS_PER_WORK_ORDER + 1
                    )
                ]
                payload["work_orders"][0]["observations"] = observations
                payload["work_orders"][0]["observation_count"] = len(observations)
                payload["work_orders"][0]["observation_count_total"] = len(observations)

            def too_many_stale_contracts(payload: dict) -> None:
                payload["work_orders"][0]["stale_prepared_experiment_contracts"] = [
                    {"experiment_id": f"stale-{index}"}
                    for index in range(
                        dispatcher_module.MAX_TUNING_STALE_PREPARED_CONTRACTS + 1
                    )
                ]

            def too_many_aborted_contracts(payload: dict) -> None:
                payload["work_orders"][0]["aborted_experiment_contracts"] = [
                    {"experiment_id": f"aborted-{index}"}
                    for index in range(
                        dispatcher_module.MAX_TUNING_ABORTED_EXPERIMENT_CONTRACTS + 1
                    )
                ]

            mutations = (
                ("work_orders", too_many_work_orders),
                ("terminal_history", too_much_terminal_history),
                ("observations", too_many_observations),
                ("stale_prepared_experiment_contracts", too_many_stale_contracts),
                ("aborted_experiment_contracts", too_many_aborted_contracts),
            )
            for expected, mutate in mutations:
                with self.subTest(expected=expected):
                    payload = json.loads(json.dumps(valid))
                    mutate(payload)
                    path.write_text(json.dumps(payload))
                    before = path.read_bytes()

                    loaded = dispatcher_module._load_tuning_work_order(path)

                    self.assertIn("_read_error", loaded)
                    self.assertIn(expected, loaded["_read_error"])
                    self.assertEqual(path.read_bytes(), before)

            oversized = json.dumps(
                {"padding": "x" * dispatcher_module.MAX_TUNING_QUEUE_BYTES}
            ).encode("utf-8")
            path.write_bytes(oversized)
            loaded = dispatcher_module._load_tuning_work_order(path)
            self.assertIn("_read_error", loaded)
            self.assertIn("raw bytes", loaded["_read_error"])
            self.assertEqual(path.stat().st_size, len(oversized))

    def test_tuning_queue_depth_and_string_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="shape-bounds")
            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            valid = json.loads(path.read_text())

            oversized_string = json.loads(json.dumps(valid))
            oversized_string["work_orders"][0]["oversized"] = (
                "x" * (dispatcher_module.MAX_TUNING_QUEUE_STRING_CHARS + 1)
            )
            path.write_text(json.dumps(oversized_string))
            self.assertIn(
                "oversized string",
                dispatcher_module._load_tuning_work_order(path)["_read_error"],
            )

            nested: dict = {"value": "leaf"}
            for _ in range(dispatcher_module.MAX_TUNING_QUEUE_JSON_DEPTH):
                nested = {"child": nested}
            excessive_depth = json.loads(json.dumps(valid))
            excessive_depth["work_orders"][0]["nested"] = nested
            path.write_text(json.dumps(excessive_depth))
            self.assertIn(
                "nesting-depth",
                dispatcher_module._load_tuning_work_order(path)["_read_error"],
            )

    def test_tuning_queue_strict_json_rejects_nonfinite_duplicates_and_bad_commitment_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="strict-json-queue",
                ),
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            valid_raw = path.read_bytes()
            valid = json.loads(valid_raw)

            malformed_documents = {
                "nan": b'{"value":NaN}',
                "positive_infinity": b'{"value":Infinity}',
                "negative_infinity": b'{"value":-Infinity}',
                "float_overflow": b'{"value":1e9999}',
                "duplicate_key": b'{"value":1,"value":2}',
                "lone_surrogate": b'{"value":"\\ud800"}',
                "recursive_depth": (
                    b'{"value":'
                    + (b"[" * 2_000)
                    + b"0"
                    + (b"]" * 2_000)
                    + b"}"
                ),
            }
            for label, raw in malformed_documents.items():
                with self.subTest(label=label):
                    path.write_bytes(raw)
                    loaded = dispatcher_module._load_tuning_work_order(path)
                    self.assertIn("_read_error", loaded)

            unhashable_commitment = json.loads(valid_raw)
            unhashable_commitment["admission_rejection_ledger"].update(
                {
                    "sha256": "0" * 64,
                    "record_count": 1,
                    "rejection_ids": [{}],
                }
            )
            path.write_text(json.dumps(unhashable_commitment))
            loaded = dispatcher_module._load_tuning_work_order(path)
            self.assertIn("_read_error", loaded)
            self.assertIn("rejection_ids", loaded["_read_error"])

            path.write_bytes(valid_raw)
            nonfinite_payload = json.loads(valid_raw)
            nonfinite_payload["work_orders"][0]["nonfinite"] = float("nan")
            with self.assertRaisesRegex(OSError, "non-finite"):
                dispatcher_module._write_tuning_queue_json(
                    path,
                    nonfinite_payload,
                    expected_source_sha256=hashlib.sha256(valid_raw).hexdigest(),
                )
            self.assertEqual(path.read_bytes(), valid_raw)

            oversized_current = b"x" * (
                dispatcher_module.MAX_TUNING_QUEUE_BYTES + 1
            )
            path.write_bytes(oversized_current)
            with self.assertRaisesRegex(OSError, "raw-byte bound"):
                dispatcher_module._write_tuning_queue_json(
                    path,
                    valid,
                    expected_source_sha256=hashlib.sha256(
                        oversized_current
                    ).hexdigest(),
                )
            self.assertEqual(path.stat().st_size, len(oversized_current))

    def test_admission_rejection_ledger_strict_json_is_bounded_and_exception_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_admission_rejections.json"
            original = {
                "work_order_id": "legacy-margin",
                "event_fingerprint": "legacy-margin",
                "status": "PENDING_HOURLY_AI_REVIEW",
                "selected_event": {
                    "event_id": "legacy-margin-event",
                    "event_type": "MARGIN_PRESSURE",
                    "pair": "PORTFOLIO",
                    "wake_reason_codes": [
                        "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE"
                    ],
                },
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            }
            record = dispatcher_module._tuning_admission_rejection_record(
                original,
                source_queue_sha256="1" * 64,
                source_queue_schema_revision=4,
            )
            valid = {
                "schema_version": (
                    dispatcher_module.TUNING_ADMISSION_REJECTION_SCHEMA_VERSION
                ),
                "generated_at_utc": NOW.isoformat(),
                "validator": dispatcher_module.TUNING_ADMISSION_REJECTION_VALIDATOR,
                "max_record_count": (
                    dispatcher_module.MAX_TUNING_ADMISSION_REJECTIONS
                ),
                "record_count": 1,
                "records": [record],
            }
            dispatcher_module._write_tuning_admission_rejections(
                path,
                valid,
                expected_source_sha256=None,
            )
            valid_raw = path.read_bytes()
            self.assertNotIn(
                "_read_error",
                dispatcher_module._load_tuning_admission_rejections(path),
            )

            malformed_documents = {
                "oversized": (
                    b"x"
                    * (
                        dispatcher_module.MAX_TUNING_ADMISSION_REJECTION_BYTES
                        + 1
                    )
                ),
                "nan": b'{"value":NaN}',
                "infinity": b'{"value":Infinity}',
                "float_overflow": b'{"value":1e9999}',
                "duplicate_key": b'{"value":1,"value":2}',
                "lone_surrogate": b'{"value":"\\ud800"}',
                "recursive_depth": (
                    b'{"value":'
                    + (b"[" * 2_000)
                    + b"0"
                    + (b"]" * 2_000)
                    + b"}"
                ),
            }
            for label, raw in malformed_documents.items():
                with self.subTest(label=label):
                    path.write_bytes(raw)
                    loaded = dispatcher_module._load_tuning_admission_rejections(
                        path
                    )
                    self.assertIn("_read_error", loaded)

            path.write_bytes(valid_raw)
            nonfinite_payload = json.loads(valid_raw)
            nonfinite_payload["records"][0]["original_entry"][
                "nonfinite"
            ] = float("inf")
            with self.assertRaisesRegex(OSError, "non-finite"):
                dispatcher_module._write_tuning_admission_rejections(
                    path,
                    nonfinite_payload,
                    expected_source_sha256=hashlib.sha256(valid_raw).hexdigest(),
                )
            self.assertEqual(path.read_bytes(), valid_raw)

    def test_invalid_queue_counters_observations_and_terminal_history_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            base = {
                "work_order_id": "counter-corrupt",
                "event_fingerprint": "counter-corrupt",
                "status": "PENDING_HOURLY_AI_REVIEW",
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            }
            mutations = (
                ({"observation_count_total": "corrupt"}, []),
                ({"observation_count": -1}, []),
                ({"reopened_count": True}, []),
                ({"observations": "not-a-list"}, []),
                ({"observations": [{}]}, []),
                (
                    {
                        "observations": [{"observation_id": "obs-1"}],
                        "observation_count": 0,
                        "observation_count_total": 1,
                    },
                    [],
                ),
                (
                    {},
                    [
                        {
                            "work_order_id": "old-terminal",
                            "event_fingerprint": "old-terminal",
                            "status": "ARCHIVED",
                            "consumed_at_utc": NOW.isoformat(),
                            "consumed_by": "hourly-ai",
                            "experiment_id": "exp1",
                            "experiment_result": "NO_EDGE",
                        }
                    ],
                ),
            )
            for mutation, history in mutations:
                with self.subTest(mutation=mutation, history=history):
                    payload = {
                        **base,
                        **mutation,
                        "schema_version": 2,
                        "queue_schema_revision": 3,
                        "work_orders": [{**base, **mutation}],
                        "pending_count": 1,
                        "terminal_history": history,
                        "terminal_history_count": len(history),
                    }
                    path.write_text(json.dumps(payload))
                    before = path.read_bytes()

                    loaded = dispatcher_module._load_tuning_work_order(path)

                    self.assertIn("_read_error", loaded)
                    self.assertEqual(path.read_bytes(), before)

            mismatched_count = {
                **base,
                "schema_version": 2,
                "queue_schema_revision": 3,
                "work_orders": [base],
                "pending_count": 0,
                "terminal_history": [],
                "terminal_history_count": 1,
            }
            path.write_text(json.dumps(mismatched_count))
            self.assertIn("_read_error", dispatcher_module._load_tuning_work_order(path))

    def test_concurrent_hourly_queue_update_wins_over_dispatcher_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            first_event = _technical_tuning_event(pair="EUR_USD", event_id="race-first")
            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            real_load = dispatcher_module._load_tuning_work_order
            hourly_payload = {"status": "CONSUMED", "experiment_id": "hourly-won"}

            def load_then_hourly_write(queue_path):
                loaded = real_load(queue_path)
                queue_path.write_text(json.dumps(hourly_payload))
                return loaded

            second_event = _technical_tuning_event(pair="GBP_USD", event_id="race-second")
            with patch.object(
                dispatcher_module,
                "_load_tuning_work_order",
                side_effect=load_then_hourly_write,
            ):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=second_event,
                    receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(result["status"], "WORK_ORDER_CONCURRENT_UPDATE")
            self.assertEqual(json.loads(path.read_text()), hourly_payload)

    def test_tuning_queue_lock_contention_retries_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="lock-held")

            with patch.object(dispatcher_module.fcntl, "flock", side_effect=BlockingIOError):
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=event,
                    receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                    now=NOW,
                )

            self.assertEqual(result["status"], "WORK_ORDER_CONCURRENT_UPDATE")
            self.assertFalse(path.exists())

    def test_hourly_lifecycle_writer_atomically_moves_reviewed_item_to_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="terminal-write")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="exp-frozen-eurusd-1",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )

            result = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-frozen-eurusd-1",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            self.assertEqual(result["status"], "WORK_ORDER_TERMINAL_WRITTEN")
            self.assertEqual(payload["pending_count"], 0)
            self.assertEqual(payload["terminal_history_count"], 1)
            terminal = payload["terminal_history"][0]
            self.assertEqual(terminal["status"], "CONSUMED")
            self.assertEqual(terminal["experiment_id"], "exp-frozen-eurusd-1")
            self.assertEqual(
                terminal["terminal_transition_source"],
                "guardian_tuning_work_order_lifecycle",
            )
            fresh_load = dispatcher_module._load_tuning_work_order(path)
            self.assertNotIn("_read_error", fresh_load)

            idempotent = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-frozen-eurusd-1",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=2),
            )
            self.assertEqual(idempotent["status"], "WORK_ORDER_ALREADY_TERMINAL")

            conflict = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id="wrong-observation",
                status="SUPERSEDED",
                consumed_by="different-consumer",
                experiment_id="exp-frozen-eurusd-1",
                experiment_result="CONTRADICTORY_RESULT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=3),
            )
            self.assertEqual(conflict["status"], "WORK_ORDER_TERMINAL_CONFLICT")
            self.assertEqual(
                conflict["recorded_terminal"]["experiment_result"],
                "REJECTED_NO_IMPROVEMENT",
            )

    def test_terminal_confirm_crash_blocks_then_idempotently_recovers_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="confirm-crash")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="exp-confirm-crash",
                result="ACCEPTED_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )

            with patch.object(
                dispatcher_module,
                "confirm_accepted_override",
                side_effect=OSError("simulated crash after terminal commit"),
            ):
                interrupted = dispatcher_module.transition_tuning_work_order(
                    path=path,
                    work_order_id=created["work_order_id"],
                    expected_observation_id=created["observation_id"],
                    status="CONSUMED",
                    consumed_by="qr-trader-hourly",
                    experiment_id="exp-confirm-crash",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    experiment_evidence_ref=evidence_ref,
                    now=NOW + timedelta(minutes=1),
                )

            override_path = path.with_name("guardian_tuning_overrides.json")
            pending = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=override_path,
                queue_path=path,
            )
            self.assertEqual(
                interrupted["status"],
                "TUNING_OVERRIDE_CONFIRMATION_PENDING",
            )
            self.assertEqual(pending["status"], "OVERRIDE_CONFIRMATION_PENDING")

            recovered = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-confirm-crash",
                experiment_result="ACCEPTED_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=2),
            )
            active = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=override_path,
                queue_path=path,
            )

            self.assertEqual(recovered["status"], "WORK_ORDER_ALREADY_TERMINAL")
            self.assertEqual(
                recovered["tuning_override_confirmation"]["status"],
                "OVERRIDE_ACTIVATED",
            )
            self.assertEqual(active["status"], "ACTIVE_OVERRIDE")
            self.assertEqual(active["resolved_value"], 0.70)

            successor_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="confirm-crash-successor",
            )
            successor = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=successor_event,
                receipt={
                    "bot_tuning_review": _valid_tuning_review(
                        "EUR_USD",
                        current_value=0.70,
                        candidate_value=0.75,
                    )
                },
                now=NOW + timedelta(minutes=3),
            )
            successor_evidence = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=successor["work_order_id"],
                observation_id=successor["observation_id"],
                experiment_id="exp-confirm-crash-successor",
                result="ACCEPTED_IMPROVEMENT",
                generated_at=NOW + timedelta(minutes=3, seconds=30),
            )
            rejected_successor = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=successor["work_order_id"],
                expected_observation_id=successor["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-confirm-crash-successor",
                experiment_result="ACCEPTED_IMPROVEMENT",
                experiment_evidence_ref=successor_evidence,
                now=NOW + timedelta(minutes=4),
            )
            self.assertEqual(
                rejected_successor["status"],
                "TUNING_OVERRIDE_PRIOR_MONITOR_NOT_KEPT",
            )

    def test_successor_terminal_recheck_blocks_stale_proof_and_recovers_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            first_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="successor-prior",
            )
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            first_evidence = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=first["work_order_id"],
                observation_id=first["observation_id"],
                experiment_id="exp-successor-prior",
                result="ACCEPTED_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            first_terminal = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=first["work_order_id"],
                expected_observation_id=first["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-successor-prior",
                experiment_result="ACCEPTED_IMPROVEMENT",
                experiment_evidence_ref=first_evidence,
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(first_terminal["status"], "WORK_ORDER_TERMINAL_WRITTEN")

            override_path = path.with_name("guardian_tuning_overrides.json")
            prior_record = json.loads(override_path.read_text())["active_overrides"][0]
            monitor_ref = (
                "data/guardian_tuning_monitor_evidence/"
                + "9" * 64
                + ".json#sha256="
                + "9" * 64
            )
            monitor_validation = {
                "status": "VALID",
                "decision": "KEEP",
                "primary_metric_value": 1.0,
            }
            with (
                patch.object(
                    dispatcher_module,
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
                patch(
                    "quant_rabbit.guardian_tuning_monitor."
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
            ):
                monitored = dispatcher_module.commit_tuning_override_monitor(
                    path=path,
                    override_path=override_path,
                    override_key=str(prior_record["override_key"]),
                    experiment_id="exp-successor-prior",
                    monitor_evidence_ref=monitor_ref,
                    decision="KEEP",
                    primary_metric_value=1.0,
                    now=NOW + timedelta(minutes=2),
                )
            self.assertEqual(monitored["status"], "POST_ACTIVATION_MONITOR_COMMITTED")

            successor_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="successor-after-keep",
            )
            successor = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=successor_event,
                receipt={
                    "bot_tuning_review": _valid_tuning_review(
                        "EUR_USD",
                        current_value=0.70,
                        candidate_value=0.75,
                    )
                },
                now=NOW + timedelta(minutes=3),
            )
            with patch(
                "quant_rabbit.guardian_tuning_monitor."
                "validate_post_activation_monitor_evidence",
                return_value=monitor_validation,
            ):
                successor_evidence = _write_completed_tuning_evidence(
                    queue_path=path,
                    work_order_id=successor["work_order_id"],
                    observation_id=successor["observation_id"],
                    experiment_id="exp-successor-after-keep",
                    result="ACCEPTED_IMPROVEMENT",
                    generated_at=NOW + timedelta(minutes=3, seconds=30),
                )

            real_verifier = dispatcher_module.read_validated_kept_predecessor_record
            verifier_calls = 0

            def stale_at_terminal_commit(**kwargs: object) -> dict:
                nonlocal verifier_calls
                verifier_calls += 1
                if verifier_calls == 2:
                    raise ValueError("simulated current-ledger change after stage")
                return real_verifier(**kwargs)

            with (
                patch.object(
                    dispatcher_module,
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
                patch(
                    "quant_rabbit.guardian_tuning_monitor."
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
                patch.object(
                    dispatcher_module,
                    "read_validated_kept_predecessor_record",
                    side_effect=stale_at_terminal_commit,
                ),
            ):
                interrupted = dispatcher_module.transition_tuning_work_order(
                    path=path,
                    work_order_id=successor["work_order_id"],
                    expected_observation_id=successor["observation_id"],
                    status="CONSUMED",
                    consumed_by="qr-trader-hourly",
                    experiment_id="exp-successor-after-keep",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    experiment_evidence_ref=successor_evidence,
                    now=NOW + timedelta(minutes=4),
                )

            staged_state = json.loads(override_path.read_text())
            self.assertEqual(
                interrupted["status"],
                "TUNING_OVERRIDE_PRIOR_PROVENANCE_INVALID",
            )
            self.assertEqual(verifier_calls, 2)
            self.assertEqual(len(staged_state["pending_overrides"]), 1)
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

            with (
                patch.object(
                    dispatcher_module,
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
                patch(
                    "quant_rabbit.guardian_tuning_monitor."
                    "validate_post_activation_monitor_evidence",
                    return_value=monitor_validation,
                ),
                patch.object(
                    dispatcher_module,
                    "read_validated_kept_predecessor_record",
                    wraps=real_verifier,
                ) as retry_verifier,
            ):
                recovered = dispatcher_module.transition_tuning_work_order(
                    path=path,
                    work_order_id=successor["work_order_id"],
                    expected_observation_id=successor["observation_id"],
                    status="CONSUMED",
                    consumed_by="qr-trader-hourly",
                    experiment_id="exp-successor-after-keep",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    experiment_evidence_ref=successor_evidence,
                    now=NOW + timedelta(minutes=5),
                )

            active_state = json.loads(override_path.read_text())
            self.assertEqual(
                recovered["status"],
                "WORK_ORDER_TERMINAL_WRITTEN",
                recovered,
            )
            self.assertEqual(
                recovered["tuning_override_application"]["status"],
                "OVERRIDE_ALREADY_STAGED",
            )
            self.assertEqual(
                recovered["tuning_override_confirmation"]["status"],
                "OVERRIDE_ACTIVATED",
            )
            self.assertEqual(retry_verifier.call_count, 2)
            self.assertEqual(
                active_state["active_overrides"][0]["experiment_id"],
                "exp-successor-after-keep",
            )

    def test_trimmed_terminal_experiment_id_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            reused_id = "exp-monotonic-id"
            first_event = _technical_tuning_event(
                pair="EUR_USD",
                event_id="id-history-first",
            )
            first = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=first_event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            first_evidence = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=first["work_order_id"],
                observation_id=first["observation_id"],
                experiment_id=reused_id,
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            completed = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=first["work_order_id"],
                expected_observation_id=first["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id=reused_id,
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=first_evidence,
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(completed["status"], "WORK_ORDER_TERMINAL_WRITTEN")

            second_event = _technical_tuning_event(
                pair="GBP_USD",
                event_id="id-history-second",
            )
            second = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=second_event,
                receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                now=NOW + timedelta(minutes=2),
            )
            queue = json.loads(path.read_text())
            self.assertIn(
                dispatcher_module._experiment_id_digest(reused_id),
                queue["experiment_id_digest_history"],
            )
            queue["terminal_history"] = []
            queue["terminal_history_count"] = 0
            path.write_text(json.dumps(queue))
            second_evidence = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=second["work_order_id"],
                observation_id=second["observation_id"],
                experiment_id=reused_id,
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(minutes=2, seconds=30),
            )

            rejected = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=second["work_order_id"],
                expected_observation_id=second["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id=reused_id,
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=second_evidence,
                now=NOW + timedelta(minutes=3),
            )

            self.assertEqual(rejected["status"], "EXPERIMENT_ID_ALREADY_USED")
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

    def test_hourly_lifecycle_writer_rejects_stale_observation_and_shared_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="terminal-stale")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="exp-stale-1",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            kwargs = {
                "path": path,
                "work_order_id": created["work_order_id"],
                "expected_observation_id": "stale-observation",
                "status": "SUPERSEDED",
                "consumed_by": "qr-trader-hourly",
                "experiment_id": "exp-stale-1",
                "experiment_result": "REJECTED_NO_IMPROVEMENT",
                "experiment_evidence_ref": evidence_ref,
                "now": NOW + timedelta(minutes=1),
            }

            stale = dispatcher_module.transition_tuning_work_order(**kwargs)
            self.assertEqual(stale["status"], "WORK_ORDER_OBSERVATION_STALE")
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

            lock_path = path.with_name(f"{path.name}.lock")
            with lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = dispatcher_module.transition_tuning_work_order(
                    **{**kwargs, "expected_observation_id": created["observation_id"]}
                )
            self.assertEqual(locked["status"], "WORK_ORDER_CONCURRENT_UPDATE")
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

    def test_hourly_lifecycle_writer_keeps_insufficient_evidence_review_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/not-run.json#sha256=" + ("0" * 64)
            review = _valid_no_change_tuning_review("EUR_USD")
            event = _technical_tuning_event(pair="EUR_USD", event_id="needs-evidence")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": review},
                now=NOW,
            )

            result = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="exp-not-run",
                experiment_result="NO_EVIDENCE",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(result["status"], "EVIDENCE_ACQUISITION_REQUIRED")
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

    def test_hourly_lifecycle_writer_rejects_unbound_plain_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="plain-evidence")
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW,
            )
            _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="plain-evidence",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            evidence_path = root / "data" / "guardian_tuning_evidence" / "plain.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text("NOT RUN\n")
            evidence_ref = (
                "data/guardian_tuning_evidence/plain.json#sha256="
                + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            )

            result = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly",
                experiment_id="plain-evidence",
                experiment_result="NOT_RUN",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(result["status"], "EXPERIMENT_EVIDENCE_INVALID")
            self.assertEqual(json.loads(path.read_text())["pending_count"], 1)

    def test_distinct_pending_tuning_work_orders_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            second_event = {
                **_event(
                    severity="P1",
                    event_id="event-gbp",
                    pair="GBP_USD",
                    dedupe_key="GBP_USD|regime|FAILED_ACCEPTANCE|TRADE",
                ),
                "wake_reason_codes": ["VOLATILITY_BUCKET_CHANGE"],
            }
            paths.events.write_text(json.dumps({"events": [second_event]}))
            _write_technical_state(paths, "EUR_USD", "GBP_USD")
            paths.escalation.write_text(
                json.dumps({"wake_gpt": True, "events_to_review": [second_event]})
            )
            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(
                        event_id="event-gbp",
                        pair="GBP_USD",
                        dedupe_key=second_event["dedupe_key"],
                        tuning_review=True,
                    ),
                ),
            )

            work_order = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(second["status"], "RECEIPT_WRITTEN")
            self.assertEqual(work_order["schema_version"], 2)
            self.assertEqual(work_order["pending_count"], 2)
            self.assertEqual(
                {item["selected_event"]["pair"] for item in work_order["work_orders"]},
                {"EUR_USD", "GBP_USD"},
            )

    def test_schema_v2_tuning_queue_stays_flat_and_fails_closed_when_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"

            def json_depth(value) -> int:
                if isinstance(value, dict):
                    return 1 + max((json_depth(item) for item in value.values()), default=0)
                if isinstance(value, list):
                    return 1 + max((json_depth(item) for item in value), default=0)
                return 0

            first_size = 0
            first_depth = 0
            bounded_text = ""
            for index in range(25):
                event = {
                    **_event(
                        severity="P1",
                        event_id=f"event-tuning-{index}",
                        pair=f"PAIR_{index:02d}",
                        dedupe_key=f"PAIR_{index:02d}|regime|FAILED_ACCEPTANCE|HOLD",
                    ),
                    "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                }
                result = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=event,
                    receipt={
                        "receipt_id": f"receipt-{index}",
                        "bot_tuning_review": _valid_tuning_review(f"PAIR_{index:02d}"),
                    },
                    now=NOW + timedelta(seconds=index),
                )
                payload = json.loads(path.read_text())
                text_size = path.stat().st_size
                depth = json_depth(payload)

                expected_count = min(index + 1, dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS)
                if index < dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS:
                    self.assertEqual(result["status"], "WORK_ORDER_WRITTEN")
                    bounded_text = path.read_text()
                else:
                    self.assertEqual(result["status"], "WORK_ORDER_QUEUE_FULL")
                    self.assertTrue(result["retry_required"])
                    self.assertEqual(path.read_text(), bounded_text)
                self.assertEqual(payload["pending_count"], expected_count)
                self.assertEqual(len(payload["work_orders"]), expected_count)
                self.assertTrue(
                    all(
                        "work_orders" not in item
                        and "schema_version" not in item
                        and "pending_count" not in item
                        for item in payload["work_orders"]
                    )
                )
                if index == 0:
                    first_size = text_size
                    first_depth = depth
                else:
                    # A flat queue grows at most linearly until its configured
                    # cap.  The historical recursive-envelope bug exceeds
                    # this bound within a few rewrites, before producing a
                    # dangerously large test artifact.
                    self.assertLessEqual(text_size, first_size * (expected_count + 1))
                    self.assertLessEqual(depth, first_depth)

            newest_event = {
                **_event(
                    severity="P1",
                    event_id="event-tuning-19",
                    pair="PAIR_19",
                    dedupe_key="PAIR_19|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }
            for offset in range(3):
                unchanged = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=newest_event,
                    receipt={
                        "receipt_id": f"duplicate-{offset}",
                        "bot_tuning_review": _valid_tuning_review("PAIR_19"),
                    },
                    now=NOW + timedelta(minutes=1, seconds=offset),
                )
                self.assertEqual(unchanged["status"], "UNCHANGED_IDEMPOTENT")
                self.assertEqual(path.read_text(), bounded_text)

    def test_legacy_twenty_rows_migrate_to_semantic_groups_without_losing_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            raw_entries: list[dict] = []
            for index in range(20):
                if index < 4:
                    event = _technical_tuning_event(
                        pair="CAD_CHF",
                        event_id=f"legacy-duplicate-{index}",
                        mid=0.571000 + index * 0.0001,
                    )
                    event["details"].pop("closed_candle_watermarks")
                else:
                    pair = f"LEGACY_{index:02d}"
                    event = {
                        **_event(
                            severity="P2",
                            event_id=f"legacy-{index}",
                            pair=pair,
                            dedupe_key=f"{pair}|lane|FAILED_ACCEPTANCE|HOLD",
                        ),
                        "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                        "details": {"lane_id": f"lane-{index}", "status": "DRY_RUN_BLOCKED"},
                    }
                observation_id = dispatcher_module._event_material_fingerprint(event)
                raw_entries.append(
                    {
                        "generated_at_utc": (NOW + timedelta(seconds=index)).isoformat(),
                        "work_order_id": f"legacy-work-{index}",
                        "status": "PENDING_HOURLY_AI_REVIEW",
                        "event_fingerprint": observation_id,
                        "observation_id": observation_id,
                        "selected_event": event,
                        "material_reason_codes": event["wake_reason_codes"],
                        "bot_tuning_review_validation": {"status": "VALID", "issues": []},
                        "bot_tuning_review": _valid_tuning_review(event["pair"]),
                        "live_permission_allowed": False,
                        "no_direct_oanda": True,
                        "preserve_blockers": True,
                    }
                )
            path.write_text(
                json.dumps(
                    {
                        **raw_entries[0],
                        "schema_version": 2,
                        "work_orders": raw_entries,
                        "pending_count": 20,
                    }
                )
            )
            self.assertNotIn("_read_error", dispatcher_module._load_tuning_work_order(path))
            new_event = {
                **_event(
                    severity="P2",
                    event_id="post-migration",
                    pair="NZD_USD",
                    dedupe_key="NZD_USD|new|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }

            result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=new_event,
                receipt={"bot_tuning_review": _valid_tuning_review("NZD_USD")},
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            cad = next(item for item in payload["work_orders"] if item["selected_event"]["pair"] == "CAD_CHF")
            self.assertEqual(result["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(payload["queue_schema_revision"], dispatcher_module.TUNING_QUEUE_SCHEMA_REVISION)
            self.assertEqual(payload["pending_count"], 18)
            self.assertEqual(len(cad["observations"]), 4)
            self.assertEqual(
                sum(len(item.get("observations") or []) for item in payload["work_orders"]),
                21,
            )

    def test_revision_four_per_candle_technical_ids_migrate_to_one_pair_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"

            def legacy_entry(index: int) -> dict:
                event = _technical_tuning_event(
                    pair="CAD_CHF",
                    event_id=f"revision-four-candle-{index}",
                    mid=0.5710 + index * 0.0001,
                )
                event["details"]["closed_candle_watermarks"] = {
                    "M1": f"2026-06-30T03:{29 + index:02d}:00+00:00",
                    "M5": "2026-06-30T03:25:00+00:00",
                    "M15": "2026-06-30T03:15:00+00:00",
                }
                observation_id = dispatcher_module._event_material_fingerprint(event)
                return {
                    "generated_at_utc": (NOW + timedelta(minutes=index)).isoformat(),
                    "work_order_id": f"legacy-revision-four-{index}",
                    "status": "PENDING_HOURLY_AI_REVIEW",
                    "event_fingerprint": observation_id,
                    "semantic_state_id": f"old-watermark-bound-semantic-{index}",
                    "observation_id": observation_id,
                    "latest_observation_id": observation_id,
                    "latest_reviewed_observation_id": observation_id,
                    "selected_event": event,
                    "selected_event_id": event["event_id"],
                    "selected_event_dedupe_key": event["dedupe_key"],
                    "material_reason_codes": event["wake_reason_codes"],
                    "bot_tuning_review_validation": {"status": "VALID", "issues": []},
                    "bot_tuning_review": _valid_tuning_review("CAD_CHF"),
                    "live_permission_allowed": False,
                    "no_direct_oanda": True,
                    "preserve_blockers": True,
                }

            entries = [legacy_entry(0), legacy_entry(1)]
            path.write_text(
                json.dumps(
                    {
                        **entries[0],
                        "schema_version": 2,
                        "queue_schema_revision": 4,
                        "work_orders": entries,
                        "pending_count": 2,
                        "terminal_history": [],
                        "terminal_history_count": 0,
                        "experiment_semantic_digest_history": [],
                        "experiment_id_digest_history": [],
                        "override_lifecycle_heads": [],
                    }
                )
            )
            incoming = _technical_tuning_event(
                pair="CAD_CHF",
                event_id="revision-four-candle-2",
                mid=0.5712,
            )
            incoming["details"]["closed_candle_watermarks"] = {
                "M1": "2026-06-30T03:31:00+00:00",
                "M5": "2026-06-30T03:30:00+00:00",
                "M15": "2026-06-30T03:15:00+00:00",
            }

            result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=incoming,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW + timedelta(minutes=2),
            )

            payload = json.loads(path.read_text())
            self.assertEqual(result["status"], "WORK_ORDER_OBSERVATION_APPENDED")
            self.assertEqual(payload["pending_count"], 1)
            self.assertEqual(len(payload["work_orders"]), 1)
            work_order = payload["work_orders"][0]
            self.assertEqual(
                work_order["semantic_identity_version"],
                dispatcher_module.TUNING_SEMANTIC_IDENTITY_VERSION,
            )
            self.assertEqual(work_order["observation_count"], 3)
            self.assertEqual(work_order["selected_event_id"], incoming["event_id"])
            self.assertEqual(
                work_order["latest_observation_id"],
                work_order["latest_reviewed_observation_id"],
            )

    def test_malformed_technical_semantic_identity_version_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(
                pair="CAD_CHF",
                event_id="malformed-semantic-version",
            )
            created = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={"bot_tuning_review": _valid_tuning_review("CAD_CHF")},
                now=NOW,
            )
            self.assertEqual(created["status"], "WORK_ORDER_WRITTEN")
            payload = json.loads(path.read_text())
            payload["work_orders"][0]["semantic_identity_version"] = "not-an-integer"
            path.write_text(json.dumps(payload))
            before = path.read_bytes()

            loaded = dispatcher_module._load_tuning_work_order(path)
            write_result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=_technical_tuning_event(
                    pair="EUR_USD",
                    event_id="after-malformed-semantic-version",
                ),
                receipt={"bot_tuning_review": _valid_tuning_review("EUR_USD")},
                now=NOW + timedelta(minutes=1),
            )

            self.assertIn("_read_error", loaded)
            self.assertIn("semantic_identity_version", loaded["_read_error"])
            self.assertEqual(write_result["status"], "WORK_ORDER_READ_FAILED")
            self.assertEqual(path.read_bytes(), before)

    def test_migrated_pair_scope_never_presents_older_review_as_latest(self) -> None:
        old_event = _technical_tuning_event(
            pair="CAD_CHF",
            event_id="reviewed-old-candle",
            mid=0.5710,
        )
        new_event = _technical_tuning_event(
            pair="CAD_CHF",
            event_id="unreviewed-new-candle",
            mid=0.5712,
        )
        old_id = dispatcher_module._event_material_fingerprint(old_event)
        new_id = dispatcher_module._event_material_fingerprint(new_event)
        reviewed = {
            "generated_at_utc": NOW.isoformat(),
            "work_order_id": "legacy-reviewed-old",
            "status": "PENDING_HOURLY_AI_REVIEW",
            "event_fingerprint": old_id,
            "semantic_state_id": "old-candle-semantic-1",
            "observation_id": old_id,
            "latest_observation_id": old_id,
            "latest_reviewed_observation_id": old_id,
            "selected_event": old_event,
            "bot_tuning_review_validation": {"status": "VALID", "issues": []},
            "bot_tuning_review": _valid_tuning_review("CAD_CHF"),
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        unreviewed = {
            "generated_at_utc": (NOW + timedelta(minutes=1)).isoformat(),
            "work_order_id": "legacy-unreviewed-new",
            "status": "PENDING_HOURLY_AI_REVIEW",
            "event_fingerprint": new_id,
            "semantic_state_id": "old-candle-semantic-2",
            "observation_id": new_id,
            "latest_observation_id": new_id,
            "selected_event": new_event,
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        payload = {
            "schema_version": 2,
            "queue_schema_revision": dispatcher_module.TUNING_QUEUE_SCHEMA_REVISION,
            "work_orders": [reviewed, unreviewed],
            "pending_count": 2,
            "terminal_history": [],
            "terminal_history_count": 0,
            "experiment_semantic_digest_history": [],
            "experiment_id_digest_history": [],
            "override_lifecycle_heads": [],
        }

        pending, _ = dispatcher_module._normalized_tuning_work_order_queue(payload)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["latest_observation_id"], new_id)
        self.assertNotIn("latest_reviewed_observation_id", pending[0])
        self.assertNotIn("bot_tuning_review", pending[0])
        self.assertEqual(
            pending[0]["bot_tuning_review_validation"]["status"],
            "MISSING",
        )

    def test_legacy_lax_valid_review_is_revalidated_before_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event = _technical_tuning_event(pair="EUR_USD", event_id="legacy-lax")
            observation_id = dispatcher_module._event_material_fingerprint(event)
            legacy = {
                "generated_at_utc": NOW.isoformat(),
                "work_order_id": "legacy-lax-review",
                "status": "PENDING_HOURLY_AI_REVIEW",
                "event_fingerprint": observation_id,
                "selected_event": event,
                "bot_tuning_review_validation": {"status": "VALID", "issues": []},
                "bot_tuning_review": {
                    "proposal": "lower every blocker",
                    "live_permission_allowed": False,
                    "no_direct_oanda": True,
                    "preserve_blockers": True,
                },
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            }
            path.write_text(
                json.dumps(
                    {
                        **legacy,
                        "schema_version": 2,
                        "work_orders": [legacy],
                        "pending_count": 1,
                    }
                )
            )

            result = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event,
                receipt={},
                now=NOW + timedelta(minutes=1),
            )

            payload = json.loads(path.read_text())
            self.assertEqual(result["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertNotEqual(payload["bot_tuning_review_validation"]["status"], "VALID")
            self.assertNotIn("bot_tuning_review", payload)

    def test_incomplete_terminal_transition_keeps_pending_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event_a = {
                **_event(
                    severity="P2",
                    event_id="incomplete-a",
                    pair="AUD_USD",
                    dedupe_key="AUD_USD|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }
            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event_a,
                receipt={"bot_tuning_review": _valid_tuning_review("AUD_USD")},
                now=NOW,
            )
            payload = json.loads(path.read_text())
            payload["status"] = "CONSUMED"
            payload["work_orders"][0]["status"] = "CONSUMED"
            path.write_text(json.dumps(payload))
            event_b = {
                **_event(
                    severity="P2",
                    event_id="incomplete-b",
                    pair="NZD_USD",
                    dedupe_key="NZD_USD|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }

            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event_b,
                receipt={"bot_tuning_review": _valid_tuning_review("NZD_USD")},
                now=NOW + timedelta(minutes=1),
            )

            migrated = json.loads(path.read_text())
            restored = next(
                item for item in migrated["work_orders"] if item["selected_event"]["pair"] == "AUD_USD"
            )
            self.assertEqual(migrated["pending_count"], 2)
            self.assertEqual(restored["status"], "PENDING_HOURLY_AI_REVIEW")
            self.assertEqual(
                restored["rejected_terminal_transition"]["reason"],
                "TERMINAL_FIELDS_INCOMPLETE",
            )
            self.assertEqual(migrated["terminal_history_count"], 0)

    def test_boundaryless_incomplete_terminal_is_restored_fail_closed(self) -> None:
        payload = {
            "schema_version": 2,
            "queue_schema_revision": 3,
            "work_orders": [
                {
                    "work_order_id": "old-unfinished",
                    "event_fingerprint": "old-unfinished",
                    "status": "CONSUMED",
                }
            ],
            "pending_count": 1,
            "terminal_history": [],
            "terminal_history_count": 0,
        }

        pending, terminal = dispatcher_module._normalized_tuning_work_order_queue(payload)

        self.assertEqual(len(pending), 1)
        self.assertEqual(terminal, [])
        self.assertEqual(pending[0]["status"], "PENDING_HOURLY_AI_REVIEW")
        self.assertIs(pending[0]["live_permission_allowed"], False)
        self.assertIs(pending[0]["no_direct_oanda"], True)
        self.assertIs(pending[0]["preserve_blockers"], True)

    def test_complementary_incomplete_top_and_child_do_not_form_terminal(self) -> None:
        base = {
            "work_order_id": "split-terminal",
            "event_fingerprint": "split-terminal",
            "semantic_state_id": "split-terminal-state",
            "status": "CONSUMED",
        }
        payload = {
            **base,
            "consumed_at_utc": NOW.isoformat(),
            "consumed_by": "hourly-ai",
            "schema_version": 2,
            "queue_schema_revision": 3,
            "work_orders": [
                {
                    **base,
                    "experiment_id": "split-experiment",
                    "experiment_result": "NO_EDGE",
                }
            ],
            "pending_count": 1,
            "terminal_history": [],
            "terminal_history_count": 0,
        }

        pending, terminal = dispatcher_module._normalized_tuning_work_order_queue(payload)

        self.assertEqual(len(pending), 1)
        self.assertEqual(terminal, [])
        self.assertEqual(
            pending[0]["rejected_terminal_transition"]["reason"],
            "TERMINAL_FIELDS_INCOMPLETE",
        )

    def test_conflicting_complete_top_and_child_terminals_restore_pending_with_both_records(self) -> None:
        base = {
            "work_order_id": "conflicting-terminal",
            "event_fingerprint": "conflicting-terminal",
            "semantic_state_id": "conflicting-terminal-state",
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        top = {
            **base,
            "status": "CONSUMED",
            "consumed_at_utc": NOW.isoformat(),
            "consumed_by": "top-consumer",
            "experiment_id": "top-experiment",
            "experiment_result": "PASS",
        }
        child = {
            **base,
            "status": "SUPERSEDED",
            "consumed_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
            "consumed_by": "child-consumer",
            "experiment_id": "child-experiment",
            "experiment_result": "FAIL",
        }
        payload = {
            **top,
            "schema_version": 2,
            "queue_schema_revision": 3,
            "work_orders": [child],
            "pending_count": 1,
            "terminal_history": [],
            "terminal_history_count": 0,
        }

        pending, terminal = dispatcher_module._normalized_tuning_work_order_queue(payload)

        self.assertEqual(len(pending), 1)
        self.assertEqual(terminal, [])
        conflict = pending[0]["rejected_terminal_transition"]
        self.assertEqual(conflict["reason"], "TOP_CHILD_TERMINAL_CONFLICT")
        self.assertEqual(conflict["top_terminal"]["experiment_id"], "top-experiment")
        self.assertEqual(conflict["child_terminal"]["experiment_id"], "child-experiment")

    def test_pending_status_with_partial_terminal_fields_is_not_dropped(self) -> None:
        entry = {
            "work_order_id": "partial-pending",
            "event_fingerprint": "partial-pending",
            "semantic_state_id": "partial-pending-state",
            "status": "PENDING_HOURLY_AI_REVIEW",
            "consumed_at_utc": NOW.isoformat(),
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        payload = {
            **entry,
            "schema_version": 2,
            "queue_schema_revision": 3,
            "work_orders": [entry],
            "pending_count": 1,
            "terminal_history": [],
            "terminal_history_count": 0,
        }

        pending, terminal = dispatcher_module._normalized_tuning_work_order_queue(payload)

        self.assertEqual(len(pending), 1)
        self.assertEqual(terminal, [])
        self.assertNotIn("consumed_at_utc", pending[0])
        self.assertEqual(
            pending[0]["rejected_terminal_transition"]["reason"],
            "TERMINAL_FIELDS_INCOMPLETE",
        )

    def test_true_unversioned_legacy_child_terminal_wins_over_stale_top_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"
            event_a = {
                **_event(
                    severity="P2",
                    event_id="mirror-a",
                    pair="AUD_USD",
                    dedupe_key="AUD_USD|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }
            fingerprint = dispatcher_module._event_material_fingerprint(event_a)
            legacy_pending = {
                "generated_at_utc": NOW.isoformat(),
                "work_order_id": "legacy-child-terminal",
                "status": "PENDING_HOURLY_AI_REVIEW",
                "event_fingerprint": fingerprint,
                "selected_event": event_a,
                "material_reason_codes": ["REGIME_STATE_CHANGE"],
                "bot_tuning_review_validation": {"status": "VALID", "issues": []},
                "bot_tuning_review": _valid_tuning_review("AUD_USD"),
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            }
            legacy_terminal = {
                **legacy_pending,
                "status": "CONSUMED",
                "consumed_at_utc": (NOW + timedelta(minutes=1)).isoformat(),
                "consumed_by": "qr-trader-hourly-ai",
                "experiment_id": "child-exp",
                "experiment_result": "REJECTED_HYPOTHESIS",
            }
            path.write_text(
                json.dumps(
                    {
                        **legacy_pending,
                        "schema_version": 2,
                        "work_orders": [legacy_terminal],
                        "pending_count": 1,
                    }
                )
            )
            self.assertNotIn("_read_error", dispatcher_module._load_tuning_work_order(path))
            event_b = {
                **_event(
                    severity="P2",
                    event_id="mirror-b",
                    pair="GBP_USD",
                    dedupe_key="GBP_USD|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }

            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event_b,
                receipt={"bot_tuning_review": _valid_tuning_review("GBP_USD")},
                now=NOW + timedelta(minutes=2),
            )

            migrated = json.loads(path.read_text())
            self.assertEqual(migrated["pending_count"], 1)
            self.assertEqual(migrated["work_orders"][0]["selected_event"]["pair"], "GBP_USD")
            self.assertEqual(migrated["terminal_history"][0]["experiment_id"], "child-exp")

    def test_non_top_terminal_experiment_history_survives_write_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_work_order.json"

            def event(pair: str) -> dict:
                return {
                    **_event(
                        severity="P2",
                        event_id=f"event-{pair}",
                        pair=pair,
                        dedupe_key=f"{pair}|regime|FAILED_ACCEPTANCE|HOLD",
                    ),
                    "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                }

            event_a = event("AUD_USD")
            for pair in ("AUD_USD", "GBP_USD"):
                dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=event(pair),
                    receipt={"bot_tuning_review": _valid_tuning_review(pair)},
                    now=NOW,
                )
            payload = json.loads(path.read_text())
            terminal = next(
                item for item in payload["work_orders"] if item["selected_event"]["pair"] == "AUD_USD"
            )
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=path,
                work_order_id=terminal["work_order_id"],
                observation_id=terminal["latest_observation_id"],
                experiment_id="exp-a",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            terminal_result = dispatcher_module.transition_tuning_work_order(
                path=path,
                work_order_id=terminal["work_order_id"],
                expected_observation_id=terminal["latest_observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly-ai",
                experiment_id="exp-a",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(terminal_result["status"], "WORK_ORDER_TERMINAL_WRITTEN")

            dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event("NZD_USD"),
                receipt={"bot_tuning_review": _valid_tuning_review("NZD_USD")},
                now=NOW + timedelta(minutes=2),
            )
            reopened = dispatcher_module._maybe_write_tuning_work_order(
                path=path,
                selected_event=event_a,
                receipt={"bot_tuning_review": _valid_tuning_review("AUD_USD")},
                now=NOW + timedelta(minutes=3),
            )

            final = json.loads(path.read_text())
            self.assertEqual(reopened["status"], "WORK_ORDER_WRITTEN")
            self.assertTrue(any(item.get("experiment_id") == "exp-a" for item in final["terminal_history"]))
            self.assertEqual(final["reopened_from_terminal"]["experiment_id"], "exp-a")
            self.assertEqual(final["reopened_count"], 1)

    def test_consumed_tuning_fingerprint_reopens_on_later_accepted_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )
            consumed = json.loads(paths.tuning_work_order.read_text())
            evidence_ref = _write_completed_tuning_evidence(
                queue_path=paths.tuning_work_order,
                work_order_id=consumed["work_order_id"],
                observation_id=consumed["latest_observation_id"],
                experiment_id="experiment-1",
                result="REJECTED_NO_IMPROVEMENT",
                generated_at=NOW + timedelta(seconds=30),
            )
            terminal_result = dispatcher_module.transition_tuning_work_order(
                path=paths.tuning_work_order,
                work_order_id=consumed["work_order_id"],
                expected_observation_id=consumed["latest_observation_id"],
                status="CONSUMED",
                consumed_by="qr-trader-hourly-ai",
                experiment_id="experiment-1",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                experiment_evidence_ref=evidence_ref,
                now=NOW + timedelta(minutes=1),
            )
            self.assertEqual(terminal_result["status"], "WORK_ORDER_TERMINAL_WRITTEN")
            paths.dispatcher_state.write_text("{}\n")

            reopened = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=2),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(tuning_review=True)),
            )

            work_order = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(reopened["tuning_handoff"]["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(work_order["status"], "PENDING_HOURLY_AI_REVIEW")
            self.assertEqual(work_order["reopened_count"], 1)
            self.assertEqual(work_order["reopened_from_terminal"]["status"], "CONSUMED")

    def test_unsafe_optional_bot_tuning_review_cannot_change_work_order_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_SHIFT_TO_RANGE",))
            calls: list[list[str]] = []
            receipt = json.loads(_valid_receipt())
            receipt["bot_tuning_review"] = {
                "proposal": "lower all blockers",
                "live_permission_allowed": True,
                "no_direct_oanda": False,
                "preserve_blockers": False,
            }

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, json.dumps(receipt)),
            )

            self.assertEqual(result["status"], "TUNING_HANDOFF_FAILED")
            self.assertEqual(result["tuning_handoff"]["status"], "STRUCTURED_REVIEW_REQUIRED")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_TUNING_HANDOFF_FAILED")
            work_order = json.loads(paths.tuning_work_order.read_text())
            self.assertEqual(
                work_order["bot_tuning_review_validation"]["status"],
                "INVALID_UNSAFE_BOUNDARY",
            )
            self.assertNotIn("bot_tuning_review", work_order)
            self.assertFalse(work_order["live_permission_allowed"])
            self.assertTrue(work_order["no_direct_oanda"])
            self.assertTrue(work_order["preserve_blockers"])

    def test_selected_event_aud_jpy_receipt_usd_cad_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            aud = _event(
                severity="P1",
                event_id="event-aud",
                pair="AUD_JPY",
                direction="LONG",
                dedupe_key="AUD_JPY|thesis|FAILED_ACCEPTANCE|TRADE",
            )
            usd = _event(
                severity="P1",
                event_id="event-usd",
                pair="USD_CAD",
                direction="SHORT",
                dedupe_key="USD_CAD|thesis|FAILED_ACCEPTANCE|TRADE",
            )
            paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": [aud, usd]}))
            _write_technical_state(paths, "AUD_JPY", "USD_CAD")
            paths.escalation.write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "wake_gpt": True,
                        "events_to_review": [
                            {**aud, "wake_reason_codes": ["NEW_EVENT"]},
                            {**usd, "wake_reason_codes": ["NEW_EVENT"]},
                        ],
                    }
                )
            )
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(
                        event_id="event-usd",
                        pair="USD_CAD",
                        side="SHORT",
                        dedupe_key="USD_CAD|thesis|FAILED_ACCEPTANCE|TRADE",
                    ),
                ),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(result["queued_for_active_trader"])
            self.assertIn("RECEIPT_EVENT_MISMATCH", paths.action_review.read_text())
            state = json.loads(paths.dispatcher_state.read_text())
            self.assertNotIn("reviewed_events", state)
            self.assertIn(aud["dedupe_key"], state["dispatch_attempts"])

    def test_selected_event_event_id_mismatch_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-other")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_selected_event_pair_and_event_id_match_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P1", pair="EUR_USD", side="LONG")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.action_receipt.exists())

    def test_selected_event_side_mismatch_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P1", pair="EUR_USD", side="SHORT")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_selected_event_dedupe_mismatch_rejects_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(dedupe_key="WRONG|DEDUPE")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_accepted_receipt_writes_matching_review_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            payload = json.loads(paths.action_receipt.read_text())
            review = paths.action_review.read_text()
            selected = payload["selected_event"]
            self.assertEqual(payload["dispatcher_status"], result["status"])
            self.assertEqual(payload["selected_event_id"], selected["event_id"])
            self.assertEqual(payload["selected_event_dedupe_key"], selected["dedupe_key"])
            self.assertEqual(payload["receipt"]["event_id"], selected["event_id"])
            self.assertEqual(payload["receipt"]["dedupe_key"], selected["dedupe_key"])
            self.assertIn(f"Generated at: `{payload['generated_at_utc']}`", review)
            self.assertIn(f"Event id: `{selected['event_id']}`", review)
            self.assertIn(f"Dedupe key: `{selected['dedupe_key']}`", review)
            self.assertIn("Receipt exists: `yes`", review)
            self.assertIn("Dispatcher status: `RECEIPT_WRITTEN`", review)

    def test_valid_no_action_output_preserves_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="NO_ACTION")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            payload = json.loads(paths.action_receipt.read_text())
            receipt = payload["receipt"]
            self.assertEqual(receipt["action"], "NO_ACTION")
            for field in (
                "event_id",
                "pair",
                "side",
                "thesis_state",
                "reason",
                "invalidation_evidence",
                "harvest_trigger",
                "margin_state",
                "ownership",
            ):
                self.assertIn(field, receipt)
                self.assertNotEqual(receipt[field], "")
            self.assertTrue(receipt["gateway_required"])
            self.assertTrue(receipt["no_direct_oanda"])

    def test_valid_hold_output_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.action_receipt.exists())
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_DEFAULT_OFF")

    def test_default_gateway_handoff_off_prevents_immediate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_DEFAULT_OFF")
            self.assertEqual(len(calls), 1)

    def test_enabled_gateway_handoff_requires_action_execute_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1", "QR_LIVE_ENABLED": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["gateway_handoff"]["required_gateway"], "LiveOrderGateway")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_ACTION_EXECUTE_DISABLED")
            self.assertEqual(len(calls), 1)

    def test_all_guardian_execute_flags_call_action_cycle_cli_after_accepted_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            action_calls: list[list[str]] = []

            def fake_action_cycle(cmd, **kwargs):
                action_calls.append(list(cmd))
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"status":"VERIFIED_NO_SEND","executed":false}\n',
                    stderr="",
                )

            with patch("tools.guardian_wake_dispatcher.subprocess.run", side_effect=fake_action_cycle):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        "QR_LIVE_ENABLED": "1",
                    },
                    subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                )

            self.assertEqual(result["gateway_handoff"]["status"], "ACTION_CYCLE_CALLED")
            self.assertEqual(action_calls, [["python3", "-m", "quant_rabbit.cli", "guardian-action-cycle"]])
            self.assertEqual(len(codex_calls), 1)


def _fixture(
    root: Path,
    *,
    wake: bool = True,
    severity: str = "P1",
    reasons: tuple[str, ...] = ("NEW_EVENT",),
) -> DispatcherPaths:
    live_root = root / "live"
    live_root.mkdir(parents=True)
    paths = DispatcherPaths.from_root(root, live_root=live_root)
    for directory in (root / "data", root / "docs", root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    paths.prompt_template.write_text("Return JSON only.\n")
    paths.daily_target_state.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "mode": "PURSUE_TARGET"}))
    paths.event_report.write_text("# Guardian Event Report\n")
    paths.event_state.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "events": {
                    "EUR_USD|technical": {
                        "event_type": "TECHNICAL_STATE_CHANGE",
                        "pair": "EUR_USD",
                        "last_seen_at_utc": NOW.isoformat(),
                    }
                },
            }
        )
    )
    paths.broker_snapshot.write_text(
        json.dumps(
            {
                "fetched_at_utc": NOW.isoformat(),
                "account": {"nav_jpy": 200000, "margin_available_jpy": 150000},
                "positions": [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701}},
            }
        )
    )
    if wake:
        _write_wake(paths, severity=severity, reasons=reasons)
    else:
        paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": []}))
        paths.escalation.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "wake_gpt": False}))
    return paths


def _write_wake(paths: DispatcherPaths, *, severity: str, reasons: tuple[str, ...]) -> None:
    event = _event(severity=severity)
    review_event = {**event, "wake_reason_codes": list(reasons)}
    paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": [event]}))
    paths.escalation.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "wake_gpt": True,
                "model_target": "GPT-5.5",
                "wake_reason_codes": list(reasons),
                "events_to_review": [review_event],
            }
        )
    )


def _write_technical_state(
    paths: DispatcherPaths,
    *pairs: str,
    now: datetime = NOW,
) -> None:
    paths.event_state.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "events": {
                    f"{pair}|technical": {
                        "event_type": "TECHNICAL_STATE_CHANGE",
                        "pair": pair,
                        "last_seen_at_utc": now.isoformat(),
                    }
                    for pair in pairs
                },
            }
        )
    )


def _event(
    *,
    severity: str,
    event_id: str | None = None,
    pair: str = "EUR_USD",
    direction: str = "LONG",
    dedupe_key: str | None = None,
) -> dict:
    return _event_payload(
        severity=severity,
        event_id=event_id,
        pair=pair,
        direction=direction,
        dedupe_key=dedupe_key,
    )


def _event_payload(
    *,
    severity: str,
    event_id: str | None = None,
    pair: str = "EUR_USD",
    direction: str = "LONG",
    dedupe_key: str | None = None,
) -> dict:
    return {
        "event_id": event_id or f"event-{severity}",
        "event_type": "FAILED_ACCEPTANCE",
        "pair": pair,
        "direction": direction,
        "thesis": "major figure rejection",
        "price_zone": f"{pair} 1.1700 rejection",
        "severity": severity,
        "recommended_review_type": "ENTRY_REVIEW",
        "dedupe_key": dedupe_key or f"{pair}|major_figure_rejection|FAILED_ACCEPTANCE|TRADE",
        "action_hint": "TRADE",
        "thesis_state": "ALIVE",
        "detected_at_utc": NOW.isoformat(),
        "details": {},
    }


def _technical_tuning_event(
    *,
    pair: str = "CAD_CHF",
    event_id: str = "technical-observation",
    mid: float = 0.571025,
    reasons: tuple[str, ...] = ("REGIME_STATE_CHANGE",),
) -> dict:
    return {
        **_event(
            severity="P2",
            event_id=event_id,
            pair=pair,
            direction="SHORT",
            dedupe_key=f"{pair}|technical|TECHNICAL_STATE_CHANGE|NO_ACTION",
        ),
        "event_type": "TECHNICAL_STATE_CHANGE",
        "action_hint": "NO_ACTION",
        "recommended_review_type": "TUNING_REVIEW",
        "price_zone": f"mid={mid:.6f} spread_pips=1.0",
        "wake_reason_codes": list(reasons),
        "details": {
            "mid": mid,
            "live_spread_pips": 1.0,
            "chart_generated_at_utc": NOW.isoformat(),
            "material_fingerprint": {
                "dominant_regime": "TREND_DOWN",
                "volatility_bucket": "QUIET",
                "family_consensus": {
                    "trend": "DOWN",
                    "mean_reversion": "UP",
                    "breakout": "MIXED",
                },
                "closed_structure": "M1:CHOCH_DOWN:2026-06-30T03:20:00+00:00",
            },
            "closed_candle_watermarks": {
                "M1": "2026-06-30T03:29:00+00:00",
                "M5": "2026-06-30T03:25:00+00:00",
                "M15": "2026-06-30T03:15:00+00:00",
            },
        },
    }


def _fill_tuning_queue(path: Path) -> None:
    for index in range(dispatcher_module.MAX_PENDING_TUNING_WORK_ORDERS):
        pair = f"PAIR_{index:02d}"
        event = {
            **_event(
                severity="P2",
                event_id=f"queued-{index}",
                pair=pair,
                dedupe_key=f"{pair}|queued|FAILED_ACCEPTANCE|HOLD",
            ),
            "wake_reason_codes": ["REGIME_STATE_CHANGE"],
        }
        result = dispatcher_module._maybe_write_tuning_work_order(
            path=path,
            selected_event=event,
            receipt={"bot_tuning_review": _valid_tuning_review(pair)},
            now=NOW + timedelta(seconds=index),
        )
        if result["status"] != "WORK_ORDER_WRITTEN":
            raise AssertionError(result)


def _valid_receipt(
    *,
    event_id: str = "event-P1",
    action: str = "TRADE",
    pair: str = "EUR_USD",
    side: str = "LONG",
    dedupe_key: str | None = None,
    tuning_review: bool = False,
) -> str:
    payload = {
        "action": action,
        "event_id": event_id,
        "new_information": True,
        "pair": pair,
        "side": side,
        "thesis_state": "ALIVE",
        "reason": "fresh failed acceptance at the major figure",
        "invalidation_evidence": "accepted trade back below failed-acceptance support",
        "invalidation": "accepted trade back below failed-acceptance support",
        "harvest_trigger": "upper range rail",
        "margin_state": "margin available and no active position",
        "ownership": "system",
        "gateway_required": True,
        "no_direct_oanda": True,
    }
    if dedupe_key is not None:
        payload["dedupe_key"] = dedupe_key
    if tuning_review:
        payload["bot_tuning_review"] = _valid_tuning_review(pair)
    return json.dumps(payload)


def _valid_tuning_review(
    pair: str,
    *,
    lane_id: str | None = None,
    current_value: float = 0.65,
    candidate_value: float = 0.70,
) -> dict:
    family, parameter = _supported_tuning_parameter()
    return {
        "review_status": "TEST_REQUIRED",
        "affected_pairs": [pair],
        "affected_bot_families": [family],
        "hypothesis": "the selected closed-candle state changed lane ranking",
        "falsifiable_experiment": "freeze the source packet and compare before/after lane scores",
        "evidence_acquisition": {
            "action_kind": "ADD_PREENTRY_SIGNAL_LOG",
            "source_ref": "data/entry_thesis_ledger.jsonl",
            "required_new_samples": 20,
            "success_condition": (
                "resolve the first 20 canonical attributed post-review entries "
                "for the selected pair and bot family"
            ),
        },
        "proposed_adjustments": [
            {
                "pair": pair,
                "lane_id": lane_id or f"trend_trader:{pair}:LONG:TREND_CONTINUATION:LIMIT",
                "bot_family": family,
                "parameter": parameter,
                "current_value": current_value,
                "candidate_value": candidate_value,
                "rationale": "compare lane ranking on the frozen source packet",
            }
        ],
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
    }


def _valid_no_change_tuning_review(pair: str) -> dict:
    review = _valid_tuning_review(pair)
    review.update(
        {
            "review_status": "NO_CHANGE_INSUFFICIENT_EVIDENCE",
            "affected_bot_families": ["trend"],
            "proposed_adjustments": [],
        }
    )
    return review


def _supported_tuning_parameter() -> tuple[str, str]:
    supported = set(dispatcher_module.SUPPORTED_THRESHOLD_PARAMETERS)
    for family, parameters in dispatcher_module._TUNING_SAFE_PARAMETERS_BY_FAMILY.items():
        matches = sorted(supported.intersection(parameters))
        if matches:
            return family, matches[0]
    common = sorted(supported.intersection(dispatcher_module._TUNING_COMMON_SAFE_PARAMETERS))
    if common:
        return "trend", common[0]
    raise AssertionError("the frozen evaluator has no structured-review parameter")


def _write_completed_tuning_evidence(
    *,
    queue_path: Path,
    work_order_id: str,
    observation_id: str,
    experiment_id: str,
    result: str,
    generated_at: datetime,
) -> str:
    queue = json.loads(queue_path.read_text())
    work_order = next(
        item
        for item in queue["work_orders"]
        if item["work_order_id"] == work_order_id
    )
    review = work_order["bot_tuning_review"]
    adjustment = review["proposed_adjustments"][0]
    root = queue_path.parent.parent
    lane_id = f"trend_trader:{adjustment['pair']}:LONG:TREND_CONTINUATION:LIMIT"
    reviewed_at = datetime.fromisoformat(work_order["structured_review_completed_at_utc"])
    cutoff = reviewed_at + timedelta(microseconds=40)
    source_payload = {
        "schema_version": 5,
        "cohort_id": "frozen-cohort-1",
        "source_watermark": {
            "selection_cutoff_utc": cutoff.isoformat(),
            "last_oanda_transaction_id": "473024",
            "ledger_rowid_watermark": 500,
            "ledger_prefix_sha256": "a" * 64,
            "canonical_outcome_set_sha256": "b" * 64,
            "entry_thesis_prefix_bytes": 100,
            "entry_thesis_prefix_sha256": "c" * 64,
            "forecast_history_prefix_bytes": 200,
            "forecast_history_prefix_sha256": "d" * 64,
        },
        "selection_cutoff_utc": cutoff.isoformat(),
        "pair": adjustment["pair"],
        "bot_family": adjustment["bot_family"],
        "lane_id": lane_id,
        "parameter": adjustment["parameter"],
        "validation_contract": {
            "mode": "FORWARD_POST_REVIEW",
            "review_digest_sha256": dispatcher_module._tuning_review_digest(review),
            "review_completed_at_utc": reviewed_at.isoformat(),
            "minimum_sample_count": 20,
        },
        "provenance": {
            "generator": "guardian_tuning_cohort_builder_v3",
            "execution_ledger_coverage_start_utc": "2026-05-06T16:52:01+00:00",
            "last_oanda_transaction_id": "473024",
            "post_cost_financing_included": True,
        },
        "samples": [
            {
                "sample_id": f"sample-{index}",
                "pair": adjustment["pair"],
                "bot_family": adjustment["bot_family"],
                "lane_id": lane_id,
                "trade_id": f"trade-{index}",
                "order_id": f"order-{index}",
                "entry_at_utc": (reviewed_at + timedelta(microseconds=index + 1)).isoformat(),
                "closed_at_utc": (reviewed_at + timedelta(microseconds=index + 21)).isoformat(),
                "signal_observed_at_utc": (reviewed_at + timedelta(microseconds=index)).isoformat(),
                "signal_record_sha256": f"{index:064x}",
                "signal_value": (
                    float(adjustment["current_value"])
                    if result == "ACCEPTED_IMPROVEMENT" and index < 4
                    else float(adjustment["candidate_value"])
                    if result == "ACCEPTED_IMPROVEMENT"
                    else 0.80
                ),
                "realized_net_jpy": (
                    -10.0
                    if result == "ACCEPTED_IMPROVEMENT" and index < 4
                    else 10.0
                    if result == "ACCEPTED_IMPROVEMENT"
                    else 10.0 if index % 2 == 0 else -10.0
                ),
                "entry_units": 1000.0,
                "net_jpy_per_1000_units": (
                    -10.0
                    if result == "ACCEPTED_IMPROVEMENT" and index < 4
                    else 10.0
                    if result == "ACCEPTED_IMPROVEMENT"
                    else 10.0 if index % 2 == 0 else -10.0
                ),
            }
            for index in range(20)
        ],
    }
    input_bytes = (
        json.dumps(source_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    input_digest = hashlib.sha256(input_bytes).hexdigest()
    input_relative = (
        Path("data")
        / "guardian_tuning_experiment_inputs"
        / "data"
        / f"{input_digest}.json"
    )
    input_path = root / input_relative
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(input_bytes)
    input_ref = f"{input_relative}#sha256={input_digest}"

    evaluator_bytes = (
        Path(dispatcher_module.__file__).resolve().with_name(
            "guardian_tuning_metric_evaluator.py"
        ).read_bytes()
    )
    evaluator_digest = hashlib.sha256(evaluator_bytes).hexdigest()
    evaluator_relative = (
        Path("data")
        / "guardian_tuning_experiment_inputs"
        / "evaluators"
        / f"{evaluator_digest}.py"
    )
    evaluator_path = root / evaluator_relative
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_bytes(evaluator_bytes)
    evaluator_ref = f"{evaluator_relative}#sha256={evaluator_digest}"

    prepared = dispatcher_module.prepare_tuning_experiment_contract(
        path=queue_path,
        work_order_id=work_order_id,
        expected_observation_id=observation_id,
        experiment_id=experiment_id,
        cohort_id="frozen-cohort-1",
        source_watermark=source_payload["source_watermark"],
        sample_count=20,
        evaluator=dispatcher_module.TUNING_EVALUATOR_NAME,
        source_data_ref=input_ref,
        evaluator_artifact_ref=evaluator_ref,
        primary_metric=dispatcher_module.TUNING_EVALUATOR_PRIMARY_METRIC,
        objective=dispatcher_module.TUNING_EVALUATOR_OBJECTIVE,
        acceptance_threshold=dispatcher_module.TUNING_FIXED_ACCEPTANCE_THRESHOLD,
        metric_names=list(dispatcher_module.TUNING_EVALUATOR_METRIC_NAMES),
        prepared_by="test-hourly-ai",
        now=generated_at - timedelta(seconds=1),
    )
    if prepared["status"] not in {
        "EXPERIMENT_CONTRACT_PREPARED",
        "EXPERIMENT_CONTRACT_ALREADY_PREPARED",
    }:
        raise AssertionError(prepared)
    prepared_contract = prepared["prepared_experiment_contract"]
    evaluation = tuning_evaluator_module.evaluate(
        source_payload,
        parameter=adjustment["parameter"],
        current_value=adjustment["current_value"],
        candidate_value=adjustment["candidate_value"],
        primary_metric=dispatcher_module.TUNING_EVALUATOR_PRIMARY_METRIC,
        objective=dispatcher_module.TUNING_EVALUATOR_OBJECTIVE,
        acceptance_threshold=dispatcher_module.TUNING_FIXED_ACCEPTANCE_THRESHOLD,
    )
    if evaluation["derived_result"] != result:
        raise AssertionError(
            f"fixture requested {result}, evaluator derived {evaluation['derived_result']}"
        )
    source_payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "exit_status": (
            "COMPLETED_SUCCESS"
            if result == "ACCEPTED_IMPROVEMENT"
            else "COMPLETED_NO_EDGE"
        ),
        "work_order_id": work_order_id,
        "observation_id": observation_id,
        "experiment_id": experiment_id,
        "experiment_contract_digest": prepared_contract["experiment_contract_digest"],
        "review_digest_sha256": dispatcher_module._tuning_review_digest(review),
        "pair": adjustment["pair"],
        "bot_family": adjustment["bot_family"],
        "parameter": adjustment["parameter"],
        "current_value": adjustment["current_value"],
        "candidate_value": adjustment["candidate_value"],
        "cohort_id": evaluation["cohort_id"],
        "source_watermark": evaluation["source_watermark"],
        "sample_count": evaluation["sample_count"],
        "baseline_metrics": evaluation["baseline_metrics"],
        "candidate_metrics": evaluation["candidate_metrics"],
        "acceptance_constraints": evaluation["acceptance_constraints"],
        "evaluator": dispatcher_module.TUNING_EVALUATOR_NAME,
        "source_data_ref": input_ref,
        "evaluator_artifact_ref": evaluator_ref,
        "primary_metric": dispatcher_module.TUNING_EVALUATOR_PRIMARY_METRIC,
        "objective": dispatcher_module.TUNING_EVALUATOR_OBJECTIVE,
        "acceptance_threshold": dispatcher_module.TUNING_FIXED_ACCEPTANCE_THRESHOLD,
        "result": result,
        "generated_at_utc": generated_at.isoformat(),
        "evaluator_execution": {
            "runner": dispatcher_module.TUNING_EVALUATOR_RUNNER,
            "exit_code": 0,
            "stdout_sha256": hashlib.sha256(
                (
                    json.dumps(evaluation, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "source_data_sha256": input_digest,
            "evaluator_artifact_sha256": evaluator_digest,
            "executed_at_utc": generated_at.isoformat(),
        },
        "no_live_side_effects": True,
    }
    source_raw = (
        json.dumps(source_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    source_digest = hashlib.sha256(source_raw).hexdigest()
    source_relative = (
        Path("data") / "guardian_tuning_experiment_runs" / f"{source_digest}.json"
    )
    source_path = root / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_raw)
    source_ref = f"{source_relative}#sha256={source_digest}"
    source_validation = dispatcher_module._validate_tuning_experiment_run_ref(
        queue_path=queue_path,
        source_artifact_ref=source_ref,
        work_order_id=work_order_id,
        observation_id=observation_id,
        experiment_id=experiment_id,
        experiment_result=result,
        review=review,
        semantic_state_id=work_order["semantic_state_id"],
        prepared_contract=prepared_contract,
        work_order_generated_at=work_order["generated_at_utc"],
        review_completed_at_utc=work_order[
            "structured_review_completed_at_utc"
        ],
        now=generated_at + timedelta(seconds=1),
    )
    assert source_validation["status"] == "VALID"
    payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "work_order_id": work_order_id,
        "observation_id": observation_id,
        "experiment_id": experiment_id,
        "review_digest_sha256": dispatcher_module._tuning_review_digest(review),
        "hypothesis": review["hypothesis"],
        "falsifiable_experiment": review["falsifiable_experiment"],
        "pair": adjustment["pair"],
        "bot_family": adjustment["bot_family"],
        "parameter": adjustment["parameter"],
        "current_value": adjustment["current_value"],
        "candidate_value": adjustment["candidate_value"],
        "result": result,
        "source_artifact_ref": source_ref,
        "experiment_contract_digest": source_validation["experiment_contract_digest"],
        "generated_at_utc": generated_at.isoformat(),
        "no_live_side_effects": True,
    }
    evidence_raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(evidence_raw).hexdigest()
    relative = Path("data") / "guardian_tuning_evidence" / f"{digest}.json"
    evidence_path = root / relative
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return f"{relative}#sha256={digest}"


def _fake_codex(calls: list[list[str]], output: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_with_prompt(calls: list[list[str]], prompts: list[str], output: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        prompts.append(input)
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_stdout(calls: list[list[str]], stdout: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


def _fake_codex_explicit(paths: DispatcherPaths, calls: list[list[str]], output: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        paths.codex_explicit_output.parent.mkdir(parents=True, exist_ok=True)
        paths.codex_explicit_output.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_session(paths: DispatcherPaths, calls: list[list[str]], assistant_text: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        session_dir = paths.codex_home / "sessions" / "2026" / "06" / "30"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"rollout-{len(calls)}.jsonl"
        session_path.write_text(
            json.dumps({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})
            + "\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_session_without_assistant(paths: DispatcherPaths, calls: list[list[str]]):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        session_dir = paths.codex_home / "sessions" / "2026" / "06" / "30"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"rollout-no-assistant-{len(calls)}.jsonl"
        session_path.write_text(json.dumps({"type": "message", "role": "user", "content": "prompt"}) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_failed(calls: list[list[str]], *, stderr: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=1, stdout="", stderr=stderr)

    return run


def _fake_codex_with_diagnostics(
    calls: list[list[str]],
    output: str,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def _fake_preflight(calls: list[list[str]], *, stdout: str = "", stderr: str = "", returncode: int = 0):
    def run(cmd, *, capture_output, text, timeout, env):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def _fake_codex_sequence(calls: list[list[str]], outputs: list[str]):
    remaining = list(outputs)

    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        output = remaining.pop(0) if remaining else outputs[-1]
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_stdout_sequence(calls: list[list[str]], outputs: list[str]):
    remaining = list(outputs)

    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        output = remaining.pop(0) if remaining else outputs[-1]
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    return run


if __name__ == "__main__":
    unittest.main()
