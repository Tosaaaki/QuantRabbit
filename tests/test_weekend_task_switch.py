from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.weekend_task_switch import switch_tasks


ROOT = Path(__file__).resolve().parents[1]


class WeekendTaskSwitchTest(unittest.TestCase):
    def test_pause_snapshots_and_pauses_related_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _codex_task(codex_root, "qr-hole-audit", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _codex_task(codex_root, "qr-weekend-market-off", "ACTIVE")
            _codex_task(codex_root, "qr-weekend-market-on", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)

            result = _run_switch("pause", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-news-digest"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-hole-audit"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-weekend-market-off"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-weekend-market-on"), "ACTIVE")
            self.assertFalse(_claude_enabled(claude_root, "trader"))
            state = json.loads(state_file.read_text())
            self.assertEqual(state["mode"], "paused")
            self.assertEqual(state["tasks"]["codex:qr-trader"]["status"], "ACTIVE")
            self.assertEqual(state["tasks"]["codex:qr-hole-audit"]["status"], "ACTIVE")
            self.assertNotIn("codex:qr-weekend-market-off", state["tasks"])
            refresh = json.loads(result.stdout)["codex_scheduler_refresh_required"]
            self.assertEqual(refresh[0]["task_id"], "qr-trader")
            self.assertEqual(
                {row["task_id"]: row["status"] for row in refresh},
                {
                    "qr-hole-audit": "PAUSED",
                    "qr-news-digest": "PAUSED",
                    "qr-self-improvement-watch": "PAUSED",
                    "qr-trader": "PAUSED",
                },
            )

    def test_restore_uses_snapshot_without_enabling_disabled_claude_traders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _codex_task(codex_root, "qr-hole-audit", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _codex_task(codex_root, "qr-weekend-market-off", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(paused.returncode, 0)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-news-digest"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-hole-audit"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-weekend-market-off"), "ACTIVE")
            self.assertFalse(_claude_enabled(claude_root, "trader"))
            self.assertFalse(_claude_enabled(claude_root, "trader_v2"))
            refresh = json.loads(result.stdout)["codex_scheduler_refresh_required"]
            self.assertEqual(refresh[-1]["task_id"], "qr-trader")
            self.assertEqual(
                {row["task_id"]: row["status"] for row in refresh},
                {
                    "qr-hole-audit": "ACTIVE",
                    "qr-news-digest": "ACTIVE",
                    "qr-self-improvement-watch": "PAUSED",
                    "qr-trader": "ACTIVE",
                },
            )
            changed_by_task = {
                row["task_id"]: row["config_file_changed"] for row in refresh
            }
            self.assertTrue(changed_by_task["qr-trader"])
            self.assertTrue(changed_by_task["qr-news-digest"])
            self.assertTrue(changed_by_task["qr-hole-audit"])
            self.assertFalse(changed_by_task["qr-self-improvement-watch"])

    def test_restore_never_enables_self_improvement_watch_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(paused.returncode, 0)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")

    def test_scheduled_restore_waits_for_dst_aware_market_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused_result = _run_switch("pause", env)
            self.assertEqual(paused_result.returncode, 0)
            self.assertEqual(_ack_refresh(paused_result, env).returncode, 0)

            with mock.patch.dict(os.environ, env, clear=True):
                waiting = switch_tasks(
                    "restore",
                    require_market_open=True,
                    now=datetime(2026, 1, 11, 21, 0, tzinfo=timezone.utc),
                )
                mode_after_wait = json.loads(state_file.read_text())["mode"]
                restored = switch_tasks(
                    "restore",
                    require_market_open=True,
                    now=datetime(2026, 1, 11, 22, 0, tzinfo=timezone.utc),
                )
                acknowledged = switch_tasks(
                    "ack-codex-scheduler-refresh",
                    operation_id=restored["codex_scheduler_refresh_operation_id"],
                    updated_tasks=_updated_tasks(restored),
                    now=datetime(2026, 1, 11, 22, 1, tzinfo=timezone.utc),
                )

            self.assertEqual(waiting["status"], "WAITING_FOR_MARKET_OPEN")
            self.assertEqual(waiting["changed_count"], 0)
            self.assertFalse(waiting["market_status"]["is_fx_open"])
            self.assertEqual(mode_after_wait, "paused")
            self.assertEqual(json.loads(state_file.read_text())["mode"], "restored")
            self.assertEqual(restored["status"], "PENDING_CODEX_SCHEDULER_REFRESH")
            self.assertEqual(acknowledged["status"], "OK")
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")

    def test_scheduled_pause_keeps_the_last_winter_market_hour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)

            with mock.patch.dict(os.environ, env, clear=True):
                waiting = switch_tasks(
                    "pause",
                    require_market_closed=True,
                    now=datetime(2026, 1, 9, 21, 0, tzinfo=timezone.utc),
                )
                state_existed_before_close = state_file.exists()
                paused = switch_tasks(
                    "pause",
                    require_market_closed=True,
                    now=datetime(2026, 1, 9, 22, 0, tzinfo=timezone.utc),
                )

            self.assertEqual(waiting["status"], "WAITING_FOR_MARKET_CLOSE")
            self.assertEqual(waiting["changed_count"], 0)
            self.assertTrue(waiting["market_status"]["is_fx_open"])
            self.assertFalse(state_existed_before_close)
            self.assertEqual(paused["status"], "PENDING_CODEX_SCHEDULER_REFRESH")
            self.assertEqual(json.loads(state_file.read_text())["mode"], "paused")
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "PAUSED")

    def test_restore_reconciles_drift_after_snapshot_was_already_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _codex_task(codex_root, "qr-hole-audit", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(paused.returncode, 0)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)
            first_restore = _run_switch("restore", env)
            self.assertEqual(first_restore.returncode, 0)
            self.assertEqual(_ack_refresh(first_restore, env).returncode, 0)

            _write_codex_status(codex_root, "qr-trader", "PAUSED")
            _write_codex_status(codex_root, "qr-hole-audit", "PAUSED")
            _write_codex_status(codex_root, "qr-self-improvement-watch", "ACTIVE")
            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["changed_count"], 3)
            self.assertIn("weekend snapshot already restored", payload["warnings"])
            self.assertIn("restored snapshot drift reconciled", payload["warnings"])
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-hole-audit"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")
            self.assertEqual(_ack_refresh(result, env).returncode, 0)
            state = json.loads(state_file.read_text())
            self.assertEqual(state["mode"], "restored")
            self.assertIn("last_restore_reconciled_at_utc", state)

            retry = _run_switch("restore", env)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            retry_refresh = json.loads(retry.stdout)["codex_scheduler_refresh_required"]
            self.assertTrue(retry_refresh)
            self.assertTrue(all(not row["config_file_changed"] for row in retry_refresh))

    def test_restore_requires_exact_scheduler_ack_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(paused.returncode, 0)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)

            restored = _run_switch("restore", env)

            self.assertEqual(restored.returncode, 0, restored.stderr)
            payload = json.loads(restored.stdout)
            self.assertEqual(payload["status"], "PENDING_CODEX_SCHEDULER_REFRESH")
            state = json.loads(state_file.read_text())
            self.assertEqual(state["mode"], "paused")
            self.assertEqual(
                state["codex_scheduler_refresh_pending"]["operation_id"],
                payload["codex_scheduler_refresh_operation_id"],
            )
            bad_ack = _run_ack("codex-refresh-wrong", _updated_tasks(payload), env)
            self.assertEqual(bad_ack.returncode, 1)
            self.assertEqual(json.loads(state_file.read_text())["mode"], "paused")
            incomplete_ack = _run_ack(
                payload["codex_scheduler_refresh_operation_id"],
                _updated_tasks(payload)[:-1],
                env,
            )
            self.assertEqual(incomplete_ack.returncode, 1)
            (codex_root / "qr-news-digest" / "automation.toml").write_text("not valid toml")

            acknowledged = _ack_refresh(restored, env)

            self.assertEqual(acknowledged.returncode, 0, acknowledged.stderr)
            state = json.loads(state_file.read_text())
            self.assertEqual(state["mode"], "restored")
            self.assertNotIn("codex_scheduler_refresh_pending", state)
            replayed = _ack_refresh(restored, env)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)
            self.assertTrue(json.loads(replayed.stdout)["idempotent_replay"])

    def test_dry_run_never_requests_codex_scheduler_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            result = subprocess.run(
                [sys.executable, "-m", "quant_rabbit.weekend_task_switch", "pause", "--dry-run"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout)["codex_scheduler_refresh_required"],
                [],
            )

    def test_opposite_transition_cannot_replace_pending_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)
            restored = _run_switch("restore", env)
            pending_before = json.loads(state_file.read_text())[
                "codex_scheduler_refresh_pending"
            ]

            opposite = _run_switch("pause", env)

            self.assertEqual(opposite.returncode, 1)
            state = json.loads(state_file.read_text())
            self.assertEqual(state["codex_scheduler_refresh_pending"], pending_before)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(
                json.loads(restored.stdout)["codex_scheduler_refresh_operation_id"],
                pending_before["operation_id"],
            )

    def test_pause_cannot_replace_restored_reconciliation_pending_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            paused = _run_switch("pause", env)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)
            restored = _run_switch("restore", env)
            self.assertEqual(_ack_refresh(restored, env).returncode, 0)
            _write_codex_status(codex_root, "qr-news-digest", "PAUSED")
            reconciliation = _run_switch("restore", env)
            self.assertEqual(reconciliation.returncode, 0)
            state_before = json.loads(state_file.read_text())
            self.assertEqual(state_before["mode"], "restored")

            opposite = _run_switch("pause", env)

            self.assertEqual(opposite.returncode, 1)
            state_after = json.loads(state_file.read_text())
            self.assertEqual(
                state_after["codex_scheduler_refresh_pending"],
                state_before["codex_scheduler_refresh_pending"],
            )
            self.assertEqual(state_after["mode"], "restored")
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")

    def test_claude_quant_rabbit_weekday_tasks_are_snapshot_managed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _claude_task(claude_root, "trader", False, project="/Users/tossaki/App/QuantRabbit-live")
            _claude_task(claude_root, "daily-review", True, project="/Users/tossaki/App/QuantRabbit")
            _claude_task(claude_root, "daily-slack-summary", True, project="/Users/tossaki/App/QuantRabbit")
            _claude_task(claude_root, "other-project", True, project="/tmp/not-qr")
            paused = _run_switch("pause", env)
            self.assertEqual(paused.returncode, 0)
            self.assertEqual(_ack_refresh(paused, env).returncode, 0)

            self.assertFalse(_claude_enabled(claude_root, "daily-review"))
            self.assertFalse(_claude_enabled(claude_root, "daily-slack-summary"))
            self.assertTrue(_claude_enabled(claude_root, "other-project"))

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(_claude_enabled(claude_root, "daily-review"))
            self.assertTrue(_claude_enabled(claude_root, "daily-slack-summary"))
            self.assertTrue(_claude_enabled(claude_root, "other-project"))

    def test_pause_is_idempotent_and_keeps_original_active_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            first_pause = _run_switch("pause", env)
            second_pause = _run_switch("pause", env)
            self.assertEqual(first_pause.returncode, 0)
            self.assertEqual(second_pause.returncode, 0)
            self.assertEqual(
                json.loads(first_pause.stdout)["codex_scheduler_refresh_operation_id"],
                json.loads(second_pause.stdout)["codex_scheduler_refresh_operation_id"],
            )
            self.assertEqual(_ack_refresh(second_pause, env).returncode, 0)

            state = json.loads(state_file.read_text())
            self.assertEqual(state["tasks"]["codex:qr-trader"]["status"], "ACTIVE")
            self.assertEqual(_run_switch("restore", env).returncode, 0)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")

    def test_restore_refuses_multiple_prepaused_trader_schedulers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _claude_task(claude_root, "trader", True)
            _claude_task(claude_root, "trader_v2", False)
            self.assertEqual(_run_switch("pause", env).returncode, 0)

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 1)
            self.assertIn("refusing to restore multiple trader schedulers", result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "PAUSED")
            self.assertFalse(_claude_enabled(claude_root, "trader"))

    def test_decabot_launchd_agents_are_paused_and_restored_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _codex_root, _claude_root, state_file = _env(root)
            _install_fake_launchctl(root, env, {"com.decabot.ai", "com.decabot.monitor"})
            launch_agent_root = Path(env["QR_WEEKEND_DECABOT_LAUNCH_AGENT_ROOT"])
            for label in ("com.decabot.ai", "com.decabot.monitor", "com.decabot.review"):
                _launch_agent_plist(launch_agent_root, label)

            result = _run_switch("pause", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_fake_launchd_loaded(env), set())
            state = json.loads(state_file.read_text())
            self.assertTrue(state["tasks"]["decabot:com.decabot.ai"]["loaded"])
            self.assertTrue(state["tasks"]["decabot:com.decabot.monitor"]["loaded"])
            self.assertFalse(state["tasks"]["decabot:com.decabot.review"]["loaded"])

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_fake_launchd_loaded(env), {"com.decabot.ai", "com.decabot.monitor"})


def _env(root: Path) -> tuple[dict[str, str], Path, Path, Path]:
    codex_root = root / "codex" / "automations"
    claude_root = root / "claude" / "scheduled-tasks"
    state_file = root / "state.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "QR_WEEKEND_CODEX_AUTOMATION_ROOT": str(codex_root),
            "QR_WEEKEND_CLAUDE_TASK_ROOT": str(claude_root),
            "QR_WEEKEND_DECABOT_LAUNCH_AGENT_ROOT": str(root / "LaunchAgents"),
            "QR_WEEKEND_LAUNCHD_DOMAIN": "gui/501",
            "QR_WEEKEND_TASK_STATE_FILE": str(state_file),
        }
    )
    return env, codex_root, claude_root, state_file


def _run_switch(action: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "quant_rabbit.weekend_task_switch", action],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_ack(
    operation_id: str,
    updated_tasks: tuple[str, ...],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    task_args = [item for task in updated_tasks for item in ("--updated-task", task)]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_rabbit.weekend_task_switch",
            "ack-codex-scheduler-refresh",
            "--operation-id",
            operation_id,
            *task_args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ack_refresh(
    result: subprocess.CompletedProcess[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    payload = json.loads(result.stdout)
    operation_id = payload["codex_scheduler_refresh_operation_id"]
    return _run_ack(operation_id, _updated_tasks(payload), env)


def _updated_tasks(payload: dict[str, object]) -> tuple[str, ...]:
    rows = payload["codex_scheduler_refresh_required"]
    assert isinstance(rows, list)
    return tuple(
        f"{row['task_id']}={row['status']}"
        for row in rows
        if isinstance(row, dict)
    )


def _codex_task(root: Path, task_id: str, status: str) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "automation.toml").write_text(
        "\n".join(
            [
                "version = 1",
                f'id = "{task_id}"',
                'kind = "cron"',
                f'name = "{task_id}"',
                'prompt = "test"',
                f'status = "{status}"',
                'rrule = "FREQ=WEEKLY;BYDAY=MO;BYHOUR=7;BYMINUTE=0"',
                'model = "gpt-5.5"',
                'reasoning_effort = "low"',
                'execution_environment = "local"',
                'cwds = ["/tmp/live"]',
            ]
        )
        + "\n"
    )


def _claude_task(root: Path, task_id: str, enabled: bool, *, project: str = "/tmp/live") -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "schedule.json").write_text(
        json.dumps(
            {
                "taskId": task_id,
                "project": project,
                "description": "test",
                "cronExpression": "*/20 * * * 1-6",
                "enabled": enabled,
                "model": "claude-sonnet-4-5",
                "jitterSeconds": 0,
            },
            indent=2,
        )
        + "\n"
    )


def _codex_status(root: Path, task_id: str) -> str:
    text = (root / task_id / "automation.toml").read_text()
    for line in text.splitlines():
        if line.startswith("status = "):
            return line.split('"')[1]
    raise AssertionError("missing status")


def _write_codex_status(root: Path, task_id: str, status: str) -> None:
    path = root / task_id / "automation.toml"
    lines = []
    changed = False
    for line in path.read_text().splitlines():
        if line.startswith("status = "):
            lines.append(f'status = "{status}"')
            changed = True
        else:
            lines.append(line)
    if not changed:
        raise AssertionError("missing status")
    path.write_text("\n".join(lines) + "\n")


def _claude_enabled(root: Path, task_id: str) -> bool:
    return bool(json.loads((root / task_id / "schedule.json").read_text())["enabled"])


def _launch_agent_plist(root: Path, label: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{label}.plist").write_text("<plist></plist>\n")


def _install_fake_launchctl(root: Path, env: dict[str, str], loaded: set[str]) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    state_file = root / "fake_launchctl_state.json"
    state_file.write_text(json.dumps(sorted(loaded)) + "\n")
    script = bin_dir / "launchctl"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["FAKE_LAUNCHCTL_STATE"])

def read_state():
    return set(json.loads(state_path.read_text()))

def write_state(state):
    state_path.write_text(json.dumps(sorted(state)) + "\\n")

def label_from_plist(path):
    name = Path(path).name
    return name[:-6] if name.endswith(".plist") else name

cmd = sys.argv[1]
state = read_state()
if cmd == "print":
    label = sys.argv[2].rsplit("/", 1)[-1]
    if label in state:
        print(f"service = {label}")
        raise SystemExit(0)
    print(f'Could not find service "{label}" in domain for user gui: 501', file=sys.stderr)
    raise SystemExit(113)
if cmd == "bootout":
    state.discard(label_from_plist(sys.argv[3]))
    write_state(state)
    raise SystemExit(0)
if cmd == "bootstrap":
    state.add(label_from_plist(sys.argv[3]))
    write_state(state)
    raise SystemExit(0)
if cmd == "enable":
    raise SystemExit(0)
print(f"unsupported launchctl command: {cmd}", file=sys.stderr)
raise SystemExit(64)
"""
    )
    script.chmod(0o755)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_LAUNCHCTL_STATE"] = str(state_file)


def _fake_launchd_loaded(env: dict[str, str]) -> set[str]:
    return set(json.loads(Path(env["FAKE_LAUNCHCTL_STATE"]).read_text()))


if __name__ == "__main__":
    unittest.main()
