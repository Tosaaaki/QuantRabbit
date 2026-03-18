from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_local_v2_launchd.sh"
STATUS_SCRIPT = REPO_ROOT / "scripts" / "status_local_v2_launchd.sh"


def _write_launchctl_stub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "launchctl"
    stub.write_text(
        """#!/bin/bash
set -euo pipefail

cmd="${1:-}"
shift || true
case "${cmd}" in
  print)
    printf '%s\\n' "${LAUNCHCTL_PRINT_OUTPUT:-stub launchctl print}"
    ;;
  *)
    ;;
esac
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir


def _env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LAUNCHCTL_PRINT_OUTPUT"] = "stub launchctl print"
    return env


def test_install_local_v2_launchd_defaults_to_trade_min(tmp_path: Path) -> None:
    bin_dir = _write_launchctl_stub(tmp_path)
    env = _env(tmp_path, bin_dir)

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    plist_path = (
        Path(env["HOME"])
        / "Library"
        / "LaunchAgents"
        / "com.quantrabbit.local-v2-autorecover.plist"
    )
    plist = plist_path.read_text(encoding="utf-8")

    assert "[ok] installed launchd agent" in completed.stdout
    assert "watchdog --once --profile 'trade_min'" in plist
    assert "watchdog --once --profile 'trade_cover'" not in plist


def test_status_local_v2_launchd_warns_on_profile_drift(tmp_path: Path) -> None:
    bin_dir = _write_launchctl_stub(tmp_path)
    env = _env(tmp_path, bin_dir)
    plist_path = (
        Path(env["HOME"])
        / "Library"
        / "LaunchAgents"
        / "com.quantrabbit.local-v2-autorecover.plist"
    )
    plist_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>exec '/repo/scripts/local_v2_stack.sh' watchdog --once --profile 'trade_cover' --env '/repo/ops/env/local-v2-stack.env'</string>
  </array>
</dict>
</plist>
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(STATUS_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configured_profile=trade_cover" in completed.stdout
    assert "launchd autorecover profile drift" in completed.stdout
    assert "--profile trade_min" in completed.stdout
