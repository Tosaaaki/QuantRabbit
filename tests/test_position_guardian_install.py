from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "install-position-guardian.sh"


class PositionGuardianInstallTest(unittest.TestCase):
    def test_check_preflight_does_not_install_launch_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)

            result = subprocess.run(
                ["bash", str(INSTALL), "--check"],
                env=_install_env(live=live, home=home),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preflight OK", result.stdout)
            self.assertFalse((home / "Library" / "LaunchAgents").exists())

    def test_check_preflight_allows_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            (live / "docs" / "position_guardian_management_report.md").write_text("runtime drift\n")
            (live / "docs" / "new_report.md").write_text("new runtime report\n")

            result = subprocess.run(
                ["bash", str(INSTALL), "--check"],
                env=_install_env(live=live, home=home),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preflight OK", result.stdout)

    def test_check_preflight_allows_guardian_trigger_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            (live / "data").mkdir(exist_ok=True)
            (live / "data" / "guardian_trigger_contract.json").write_text('{"generated_at_utc":"runtime"}\n')

            result = subprocess.run(
                ["bash", str(INSTALL), "--check"],
                env=_install_env(live=live, home=home),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preflight OK", result.stdout)

    def test_check_preflight_allows_proof_evidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            (live / "data").mkdir(exist_ok=True)
            (live / "docs").mkdir(exist_ok=True)
            (live / "data" / "as_proof_pack_queue.json").write_text('{"generated_at":"runtime"}\n')
            (live / "docs" / "as_proof_pack_queue.md").write_text("runtime proof report\n")

            result = subprocess.run(
                ["bash", str(INSTALL), "--check"],
                env=_install_env(live=live, home=home),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preflight OK", result.stdout)

    def test_check_preflight_blocks_non_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            (live / "data").mkdir(exist_ok=True)
            (live / "data" / "codex_trader_decision_response.json").write_text("{}\n")

            result = subprocess.run(
                ["bash", str(INSTALL), "--check"],
                env=_install_env(live=live, home=home),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 3)
            self.assertIn("blocking dirty live path", result.stderr)
            self.assertIn("data/codex_trader_decision_response.json", result.stderr)

    def test_require_loaded_fails_when_launchd_label_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            _write_guardian_plist(home)
            env = _install_env(live=live, home=home)
            _install_fake_launchctl(root, env, loaded=False)

            result = subprocess.run(
                ["bash", str(INSTALL), "--require-loaded"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("position guardian launchd label is not loaded", result.stderr)

    def test_require_loaded_passes_when_launchd_label_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            _write_guardian_plist(home)
            _write_guardian_heartbeat(live)
            env = _install_env(live=live, home=home)
            _install_fake_launchctl(root, env, loaded=True)

            result = subprocess.run(
                ["bash", str(INSTALL), "--require-loaded"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("heartbeat OK", result.stdout)
            self.assertIn("active OK", result.stdout)

    def test_require_loaded_fails_when_launchd_loaded_but_heartbeat_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            _write_guardian_plist(home)
            env = _install_env(live=live, home=home)
            _install_fake_launchctl(root, env, loaded=True)

            result = subprocess.run(
                ["bash", str(INSTALL), "--require-loaded"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("position guardian heartbeat is missing or stale", result.stderr)
            self.assertIn("fresh entry sends remain blocked", result.stderr)

    def test_require_loaded_fails_when_launchd_loaded_but_heartbeat_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            _write_guardian_plist(home)
            _write_guardian_heartbeat(live, generated_at=datetime.now(timezone.utc) - timedelta(minutes=10))
            env = _install_env(live=live, home=home)
            _install_fake_launchctl(root, env, loaded=True)

            result = subprocess.run(
                ["bash", str(INSTALL), "--require-loaded"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("position guardian heartbeat is missing or stale", result.stderr)
            self.assertIn("position_guardian_execution.json", result.stderr)

    def test_require_loaded_heartbeat_check_has_explicit_operator_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            home = root / "home"
            _create_live_repo(live)
            _write_guardian_plist(home)
            env = _install_env(live=live, home=home)
            env["QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT"] = "0"
            _install_fake_launchctl(root, env, loaded=True)

            result = subprocess.run(
                ["bash", str(INSTALL), "--require-loaded"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("heartbeat check skipped", result.stdout)
            self.assertIn("active OK", result.stdout)


def _create_live_repo(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / "run-position-guardian-live.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (root / "scripts" / "run-position-guardian-live.sh").chmod(0o755)
    (root / ".env.local").write_text(
        "\n".join(
            [
                "QR_OANDA_ACCOUNT_ID=acct-test",
                "QR_OANDA_TOKEN=token-test",
                "QR_OANDA_BASE_URL=https://example.test",
                "QR_LIVE_ENABLED=0",
            ]
        )
        + "\n"
    )
    (root / ".gitignore").write_text(".env.local\n")
    _run(["git", "init"], cwd=root)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=root)
    _run(["git", "config", "user.name", "Test User"], cwd=root)
    _run(["git", "add", "."], cwd=root)
    _run(["git", "commit", "-m", "initial"], cwd=root)
    _run(["git", "branch", "-m", "main"], cwd=root)


def _install_env(*, live: Path, home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "QR_SYNC_LIVE_ROOT": str(live),
            "QR_SYNC_DEV_ROOT": str(live),
            "QR_SYNC_MAIN_BRANCH": "main",
            "QR_POSITION_GUARDIAN_INTERVAL": "30",
        }
    )
    return env


def _write_guardian_plist(home: Path) -> Path:
    plist = home / "Library" / "LaunchAgents" / "com.quantrabbit.position-guardian.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text("<plist><dict><key>Label</key><string>com.quantrabbit.position-guardian</string></dict></plist>\n")
    return plist


def _write_guardian_heartbeat(live: Path, *, generated_at: datetime | None = None) -> Path:
    path = live / "data" / "position_guardian_execution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": (generated_at or datetime.now(timezone.utc)).isoformat(),
                "status": "NO_ACTION",
                "sent": False,
            }
        )
        + "\n"
    )
    return path


def _install_fake_launchctl(root: Path, env: dict[str, str], *, loaded: bool) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "launchctl"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cmd=\"${1:-}\"",
                "label=\"${2:-}\"",
                "if [[ \"$cmd\" == \"list\" && \"$label\" == \"com.quantrabbit.position-guardian\" ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf '123\\t0\\tcom.quantrabbit.position-guardian\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "if [[ \"$cmd\" == \"print\" && \"$label\" == gui/*/com.quantrabbit.position-guardian ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf 'com.quantrabbit.position-guardian = { active = 1 }\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "printf 'unsupported fake launchctl command: %s %s\\n' \"$cmd\" \"$label\" >&2",
                "exit 64",
            ]
        )
        + "\n"
    )
    script.chmod(0o755)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["QR_FAKE_POSITION_GUARDIAN_LOADED"] = "1" if loaded else "0"


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
