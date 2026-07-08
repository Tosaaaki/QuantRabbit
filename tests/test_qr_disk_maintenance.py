from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qr_disk_maintenance.py"


class QrDiskMaintenanceTest(unittest.TestCase):
    def test_help_runs_without_pythonpath(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Bound QuantRabbit runtime disk growth", result.stdout)

    def test_dry_run_reports_candidates_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history" / "run" / "EUR_USD"
            history.mkdir(parents=True)
            source = history / "EUR_USD_S5_BA_20260601T000000Z_20260701T000000Z.jsonl"
            source.write_text('{"time":"2026-06-01T00:00:00Z"}\n', encoding="utf-8")
            _make_old(source)
            report = root / "logs" / "disk_maintenance_report.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--min-size-mb",
                    "0",
                    "--min-age-minutes",
                    "0",
                    "--report-path",
                    str(report.relative_to(root)),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(source.exists())
            self.assertFalse(source.with_name(f"{source.name}.gz").exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["compression_candidates"], 1)
            self.assertEqual(payload["summary"]["compressed_files"], 0)

    def test_apply_compresses_history_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history" / "run" / "EUR_USD"
            history.mkdir(parents=True)
            source = history / "EUR_USD_S5_BA_20260601T000000Z_20260701T000000Z.jsonl"
            source.write_text('{"time":"2026-06-01T00:00:00Z"}\n', encoding="utf-8")
            _make_old(source)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--apply",
                    "--min-size-mb",
                    "0",
                    "--min-age-minutes",
                    "0",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

            gz_path = source.with_name(f"{source.name}.gz")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(source.exists())
            self.assertTrue(gz_path.exists())
            with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), '{"time":"2026-06-01T00:00:00Z"}\n')

    def test_apply_removes_stale_history_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history" / "run" / "EUR_USD"
            history.mkdir(parents=True)
            stale = history / "EUR_USD_S5_BA.jsonl.gz.partial"
            fresh = history / "EUR_USD_S5_BA.jsonl.gz.tmp"
            stale.write_text("partial", encoding="utf-8")
            fresh.write_text("tmp", encoding="utf-8")
            _make_old(stale, days=3)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--apply",
                    "--prune-temp-days",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())


def _make_old(path: Path, *, days: int = 1) -> None:
    timestamp = time.time() - days * 24 * 60 * 60
    os.utime(path, (timestamp, timestamp))


if __name__ == "__main__":
    unittest.main()
