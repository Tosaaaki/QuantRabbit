from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tools.guardian_wake_dispatcher as dispatcher_module
from tools.guardian_wake_dispatcher import (
    DispatcherPaths,
    _resolve_codex_bin,
    _select_dispatch_event,
    run_dispatcher,
)


NOW = datetime(2026, 6, 30, 4, 0, tzinfo=timezone.utc)


class GuardianWakeDispatcherTest(unittest.TestCase):
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
                        "events": {
                            selected["dedupe_key"]: {
                                "event_id": "state-event-must-not-leak",
                                "pair": "EUR_USD",
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
                subprocess_run=_fake_codex(calls, _valid_receipt()),
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
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-moved")),
            )
            retried = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=61),
                env=env,
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-moved")),
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
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
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
                subprocess_run=_fake_codex(calls, _valid_receipt()),
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
                subprocess_run=_fake_codex(calls, _valid_receipt()),
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
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(first["tuning_handoff"]["status"], "WORK_ORDER_WRITTEN")
            self.assertEqual(work_order["status"], "PENDING_HOURLY_AI_REVIEW")
            self.assertFalse(work_order["live_permission_allowed"])
            self.assertTrue(work_order["no_direct_oanda"])
            self.assertTrue(work_order["preserve_blockers"])
            self.assertEqual(work_order["bot_tuning_review_validation"]["status"], "MISSING")
            self.assertEqual(second["tuning_handoff"]["status"], "UNCHANGED_IDEMPOTENT")
            self.assertEqual(paths.tuning_work_order.read_text(), first_work_order_text)

    def test_tuning_work_order_write_failure_is_retried_without_accepted_ack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            real_write_json = dispatcher_module._write_json

            def fail_only_work_order(path, payload):
                if path == paths.tuning_work_order:
                    raise OSError("ENOSPC")
                return real_write_json(path, payload)

            with patch.object(dispatcher_module, "_write_json", side_effect=fail_only_work_order):
                first = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                    subprocess_run=_fake_codex(calls, _valid_receipt()),
                )

            first_state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(first["status"], "TUNING_HANDOFF_FAILED")
            self.assertTrue(first["receipt_written"])
            self.assertFalse(paths.tuning_work_order.exists())
            self.assertNotIn(_event(severity="P1")["dedupe_key"], first_state.get("reviewed_events", {}))
            self.assertIn(_event(severity="P1")["dedupe_key"], first_state["dispatch_attempts"])

            paths.escalation.write_text(json.dumps({"wake_gpt": False, "events_to_review": []}))
            recovered = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(seconds=1),
                env={"QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(recovered["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.tuning_work_order.exists())
            self.assertIn(
                _event(severity="P1")["dedupe_key"],
                json.loads(paths.dispatcher_state.read_text())["reviewed_events"],
            )

    def test_distinct_pending_tuning_work_orders_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
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

    def test_schema_v2_tuning_queue_stays_flat_and_bounded_across_rewrites(self) -> None:
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
                    receipt={"receipt_id": f"receipt-{index}"},
                    now=NOW + timedelta(seconds=index),
                )
                payload = json.loads(path.read_text())
                text_size = path.stat().st_size
                depth = json_depth(payload)

                self.assertEqual(result["status"], "WORK_ORDER_WRITTEN")
                self.assertEqual(payload["pending_count"], min(index + 1, 20))
                self.assertEqual(len(payload["work_orders"]), min(index + 1, 20))
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
                    # A flat queue grows at most linearly until its 20-entry
                    # cap.  The historical recursive-envelope bug exceeds
                    # this bound within a few rewrites, before producing a
                    # dangerously large test artifact.
                    self.assertLessEqual(
                        text_size,
                        first_size * (min(index + 1, 20) + 1),
                    )
                    self.assertLessEqual(depth, first_depth)

            bounded_text = path.read_text()
            newest_event = {
                **_event(
                    severity="P1",
                    event_id="event-tuning-24",
                    pair="PAIR_24",
                    dedupe_key="PAIR_24|regime|FAILED_ACCEPTANCE|HOLD",
                ),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
            }
            for offset in range(3):
                unchanged = dispatcher_module._maybe_write_tuning_work_order(
                    path=path,
                    selected_event=newest_event,
                    receipt={"receipt_id": f"duplicate-{offset}"},
                    now=NOW + timedelta(minutes=1, seconds=offset),
                )
                self.assertEqual(unchanged["status"], "UNCHANGED_IDEMPOTENT")
                self.assertEqual(path.read_text(), bounded_text)

    def test_consumed_tuning_fingerprint_reopens_on_later_accepted_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), reasons=("REGIME_STATE_CHANGE",))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )
            consumed = json.loads(paths.tuning_work_order.read_text())
            consumed["status"] = "CONSUMED"
            consumed["consumed_at_utc"] = (NOW + timedelta(minutes=1)).isoformat()
            consumed["consumed_by"] = "qr-trader-hourly-ai"
            consumed["experiment_id"] = "experiment-1"
            consumed["experiment_result"] = "NO_CHANGE_INSUFFICIENT_EDGE"
            paths.tuning_work_order.write_text(json.dumps(consumed))
            paths.dispatcher_state.write_text("{}\n")

            reopened = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=2),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
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

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
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
    paths.event_state.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": {}}))
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


def _valid_receipt(
    *,
    event_id: str = "event-P1",
    action: str = "TRADE",
    pair: str = "EUR_USD",
    side: str = "LONG",
    dedupe_key: str | None = None,
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
    return json.dumps(payload)


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
