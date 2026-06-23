from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.automation import _acquire_autotrade_lock


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run-autotrade-live.sh"
GUARDIAN_WRAPPER = ROOT / "scripts" / "run-position-guardian-live.sh"


class LiveWrapperTest(unittest.TestCase):
    def test_unset_live_enabled_stays_dry_run_even_with_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture)
            env.pop("QR_LIVE_ENABLED", None)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=0\n", payload)
            self.assertIn("QR_AUTOTRADE_LOCK_HELD=1\n", payload)
            self.assertIn("<--send>", payload)
            self.assertIn("<--use-gpt-trader>", payload)
            self.assertIn("<--reuse-market-artifacts>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertEqual((root / "sync.args").read_text(), "--live-only --skip-tests\n")

    def test_env_file_live_enabled_allows_live_send_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env.pop("QR_LIVE_ENABLED", None)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=1\n", payload)
            self.assertIn("QR_POSITION_GUARDIAN_ACTIVE=1\n", payload)
            self.assertNotIn("forcing dry-run mode", result.stderr)
            self.assertEqual((root / "sync.args").read_text(), "--live-only --skip-tests\n")

    def test_live_send_marks_inactive_position_guardian_for_gateway_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", guardian_loaded=False)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_POSITION_GUARDIAN_ACTIVE=0\n", payload)
            self.assertIn("LiveOrderGateway will block fresh entry sends", result.stderr)

    def test_live_gpt_handoff_adds_missing_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=1\n", payload)
            self.assertIn("<--gpt-decision-response><data/codex_trader_decision_response.json>", payload)
            self.assertIn("<--send>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("adding --send to avoid a stage-only live trader cycle", result.stderr)

    def test_stage_only_live_gpt_handoff_requires_explicit_escape_hatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env["QR_ALLOW_LIVE_STAGE_ONLY"] = "1"

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertNotIn("<--send>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("QR_ALLOW_LIVE_STAGE_ONLY=1; keeping GPT handoff stage-only", result.stderr)

    def test_successful_cycle_refreshes_post_gateway_sidecars_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("<-m><quant_rabbit.cli><autotrade-cycle>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("refreshing post-gateway sidecars under live lock", result.stderr)

    def test_failed_cycle_runs_projection_position_and_audit_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", python_exit=37)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 37)
            payload = capture.read_text()
            self.assertIn("<-m><quant_rabbit.cli><autotrade-cycle>", payload)
            self.assertNotIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("<-m><quant_rabbit.cli><post-autotrade-failure-sidecars>", payload)
            self.assertIn("refreshing failure-repair sidecars under live lock", result.stderr)

    def test_sync_failure_continues_when_runtime_is_current_with_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", sync_exit=37)
            _init_git(root)
            (root / "docs").mkdir(exist_ok=True)
            (root / "docs" / "cycle_report.md").write_text("tracked report\n")
            _run(["git", "add", "."], cwd=root)
            _run(["git", "commit", "-m", "initial"], cwd=root)
            _run(["git", "branch", "-m", "main"], cwd=root)
            (root / "docs" / "cycle_report.md").write_text("runtime drift\n")
            env["QR_SYNC_DEV_ROOT"] = str(root)
            env["QR_SYNC_MAIN_BRANCH"] = "main"
            env["QR_SYNC_MARKER_PATH"] = str(root / "docs" / "sync_report.md")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(capture.exists())
            self.assertIn("live sync failed with status=37", result.stderr)

    def test_empty_verdict_marker_is_removed_before_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            _init_git(root)
            (root / "EXTEND").write_text("")
            _run(["git", "add", "."], cwd=root)
            _run(["git", "commit", "-m", "initial"], cwd=root)
            _run(["git", "branch", "-m", "main"], cwd=root)
            _run(["git", "rm", "--cached", "EXTEND"], cwd=root)
            _run(["git", "commit", "-m", "untrack marker"], cwd=root)
            (root / "EXTEND").write_text("")
            env["QR_SYNC_DEV_ROOT"] = str(root)
            env["QR_SYNC_MAIN_BRANCH"] = "main"

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "EXTEND").exists())
            self.assertIn("removed empty verdict marker: EXTEND", result.stderr)

    def test_existing_live_lock_blocks_overlapping_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 75)
            self.assertFalse(capture.exists())
            self.assertIn("another autotrade cycle is already running", result.stderr)

    def test_live_send_waits_for_position_guardian_lock_then_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env["QR_RUN_POST_GATEWAY_SIDECARS"] = "0"
            env["QR_AUTOTRADE_LOCK_WAIT_SECONDS"] = "5"
            env["QR_AUTOTRADE_LOCK_POLL_SECONDS"] = "0.1"
            holder = root / "run-position-guardian-live.sh"
            holder.write_text("#!/usr/bin/env bash\nsleep 1\n")
            holder.chmod(0o755)
            proc = subprocess.Popen([str(holder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                lock_dir = root / "lock"
                lock_dir.mkdir()
                (lock_dir / "pid").write_text(f"{proc.pid}\n")

                result = subprocess.run(
                    ["bash", str(WRAPPER), "--send"],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            finally:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=2)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(capture.exists())
            self.assertIn("waiting up to 5s", result.stderr)
            self.assertIn("removing defunct lock holder", result.stderr)

    def test_position_guardian_skips_when_full_trader_lock_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")
            (lock_dir / "command").write_text("run-autotrade-live\n")

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(capture.exists())
            self.assertIn("live runtime lock busy", result.stderr)
            self.assertIn("skipped guardian cycle", result.stderr)

    def test_live_lock_release_preserves_reacquired_lock_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        "source \"$QR_HELPER\"; "
                        "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 0 '' 0.1; "
                        "printf '%s\\n' other-token > \"$QR_LOCK_DIR/token\"; "
                        "qr_live_lock_release; "
                        "test -d \"$QR_LOCK_DIR\""
                    ),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(lock_dir.exists())

    def test_direct_send_lock_blocks_overlapping_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")
            original_lock_dir = os.environ.get("QR_AUTOTRADE_LOCK_DIR")
            original_lock_held = os.environ.get("QR_AUTOTRADE_LOCK_HELD")
            os.environ["QR_AUTOTRADE_LOCK_DIR"] = str(lock_dir)
            os.environ.pop("QR_AUTOTRADE_LOCK_HELD", None)
            try:
                with self.assertRaisesRegex(RuntimeError, "another autotrade cycle is already running"):
                    _acquire_autotrade_lock(send=True)
            finally:
                _restore_env("QR_AUTOTRADE_LOCK_DIR", original_lock_dir)
                _restore_env("QR_AUTOTRADE_LOCK_HELD", original_lock_held)


def _wrapper_env(
    root: Path,
    capture: Path,
    *,
    live_enabled: str | None = None,
    sync_exit: int = 0,
    python_exit: int = 0,
    guardian_loaded: bool = True,
) -> dict[str, str]:
    env_file = root / "oanda.env"
    lines = [
        "QR_OANDA_ACCOUNT_ID=acct-test",
        "QR_OANDA_TOKEN=token-test",
        "QR_OANDA_BASE_URL=https://example.test",
    ]
    if live_enabled is not None:
        lines.append(f"QR_LIVE_ENABLED={live_enabled}")
    env_file.write_text(
        "\n".join(lines)
        + "\n"
    )
    fake_scripts = root / "scripts"
    fake_scripts.mkdir()
    fake_sync = fake_scripts / "sync-live-runtime.sh"
    fake_sync.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "printf '%s\\n' \"$*\" > \"$QR_SYNC_MARKER_PATH\"",
                f"exit {sync_exit}",
            ]
        )
        + "\n"
    )
    fake_sync.chmod(0o755)
    fake_guardian_check = fake_scripts / "install-position-guardian.sh"
    fake_guardian_check.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "if [[ \"${1:-}\" != \"--require-loaded\" ]]; then",
                "  printf 'unexpected guardian check args: %s\\n' \"$*\" >&2",
                "  exit 64",
                "fi",
                "if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-1}\" == \"1\" ]]; then",
                "  printf '[install-position-guardian] active OK: label=com.quantrabbit.position-guardian\\n'",
                "  exit 0",
                "fi",
                "printf '[install-position-guardian] position guardian launchd label is not loaded: com.quantrabbit.position-guardian\\n' >&2",
                "exit 6",
            ]
        )
        + "\n"
    )
    fake_guardian_check.chmod(0o755)
    fake_bin = root / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "{",
                "  printf 'QR_LIVE_ENABLED=%s\\n' \"${QR_LIVE_ENABLED:-}\"",
                "  printf 'QR_REQUIRE_POSITION_GUARDIAN_ACTIVE=%s\\n' \"${QR_REQUIRE_POSITION_GUARDIAN_ACTIVE:-}\"",
                "  printf 'QR_POSITION_GUARDIAN_ACTIVE=%s\\n' \"${QR_POSITION_GUARDIAN_ACTIVE:-}\"",
                "  printf 'QR_AUTOTRADE_LOCK_HELD=%s\\n' \"${QR_AUTOTRADE_LOCK_HELD:-}\"",
                "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH:-}\"",
                "  printf 'ARGV='",
                "  for arg in \"$@\"; do printf '<%s>' \"$arg\"; done",
                "  printf '\\n'",
                "} >> \"$QR_CAPTURE_PATH\"",
                "for arg in \"$@\"; do",
                "  if [[ \"$arg\" == \"autotrade-cycle\" ]]; then",
                f"    exit {python_exit}",
                "  fi",
                "done",
                "exit 0",
            ]
        )
        + "\n"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "QR_CAPTURE_PATH": str(capture),
            "QR_PYTHON": str(fake_python),
            "QR_OANDA_ENV_FILE": str(env_file),
            "QR_AUTOTRADE_LOCK_DIR": str(root / "lock"),
            "QR_TRADER_ROOT_DIR": str(root),
            "QR_LIVE_SYNC_ENABLED": "1",
            "QR_SYNC_MARKER_PATH": str(root / "sync.args"),
            "QR_FAKE_POSITION_GUARDIAN_LOADED": "1" if guardian_loaded else "0",
        }
    )
    return env


def _init_git(root: Path) -> None:
    _run(["git", "init"], cwd=root)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=root)
    _run(["git", "config", "user.name", "Test User"], cwd=root)


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
