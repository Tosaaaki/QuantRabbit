from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
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


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
