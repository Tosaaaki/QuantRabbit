from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.automation import _acquire_autotrade_lock


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run-autotrade-live.sh"


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


def _wrapper_env(root: Path, capture: Path) -> dict[str, str]:
    env_file = root / "oanda.env"
    env_file.write_text(
        "\n".join(
            [
                "QR_OANDA_ACCOUNT_ID=acct-test",
                "QR_OANDA_TOKEN=token-test",
                "QR_OANDA_BASE_URL=https://example.test",
            ]
        )
        + "\n"
    )
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
                "  printf 'QR_AUTOTRADE_LOCK_HELD=%s\\n' \"${QR_AUTOTRADE_LOCK_HELD:-}\"",
                "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH:-}\"",
                "  printf 'ARGV='",
                "  for arg in \"$@\"; do printf '<%s>' \"$arg\"; done",
                "  printf '\\n'",
                "} > \"$QR_CAPTURE_PATH\"",
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
            "QR_OANDA_ENV_FILE": str(env_file),
            "QR_AUTOTRADE_LOCK_DIR": str(root / "lock"),
        }
    )
    return env


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
