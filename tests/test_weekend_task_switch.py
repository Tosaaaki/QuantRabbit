from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
            self.assertEqual(_run_switch("pause", env).returncode, 0)

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-news-digest"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-hole-audit"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")
            self.assertEqual(_codex_status(codex_root, "qr-weekend-market-off"), "ACTIVE")
            self.assertFalse(_claude_enabled(claude_root, "trader"))
            self.assertFalse(_claude_enabled(claude_root, "trader_v2"))

    def test_restore_never_enables_self_improvement_watch_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            self.assertEqual(_run_switch("pause", env).returncode, 0)

            result = _run_switch("restore", env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_codex_status(codex_root, "qr-trader"), "ACTIVE")
            self.assertEqual(_codex_status(codex_root, "qr-self-improvement-watch"), "PAUSED")

    def test_restore_reconciles_drift_after_snapshot_was_already_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _codex_task(codex_root, "qr-news-digest", "ACTIVE")
            _codex_task(codex_root, "qr-hole-audit", "ACTIVE")
            _codex_task(codex_root, "qr-self-improvement-watch", "ACTIVE")
            _claude_task(claude_root, "trader", False)
            _claude_task(claude_root, "trader_v2", False)
            self.assertEqual(_run_switch("pause", env).returncode, 0)
            self.assertEqual(_run_switch("restore", env).returncode, 0)

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
            state = json.loads(state_file.read_text())
            self.assertEqual(state["mode"], "restored")
            self.assertIn("last_restore_reconciled_at_utc", state)

    def test_claude_quant_rabbit_weekday_tasks_are_snapshot_managed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, codex_root, claude_root, _state_file = _env(Path(tmp))
            _codex_task(codex_root, "qr-trader", "ACTIVE")
            _claude_task(claude_root, "trader", False, project="/Users/tossaki/App/QuantRabbit-live")
            _claude_task(claude_root, "daily-review", True, project="/Users/tossaki/App/QuantRabbit")
            _claude_task(claude_root, "daily-slack-summary", True, project="/Users/tossaki/App/QuantRabbit")
            _claude_task(claude_root, "other-project", True, project="/tmp/not-qr")
            self.assertEqual(_run_switch("pause", env).returncode, 0)

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
            self.assertEqual(_run_switch("pause", env).returncode, 0)
            self.assertEqual(_run_switch("pause", env).returncode, 0)

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
