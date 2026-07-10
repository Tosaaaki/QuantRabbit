from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qr_disk_maintenance.py"

SPEC = importlib.util.spec_from_file_location("qr_disk_maintenance_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DISK_MAINTENANCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DISK_MAINTENANCE
SPEC.loader.exec_module(DISK_MAINTENANCE)


class _ClosedInspector:
    def state(self, path: Path) -> str:
        del path
        return "CLOSED"


class _OpenInspector:
    def state(self, path: Path) -> str:
        del path
        return "OPEN"


class _MutateSourceOnSecondCheckInspector:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.source_checks = 0

    def state(self, path: Path) -> str:
        if path == self.source:
            self.source_checks += 1
            if self.source_checks == 2:
                with path.open("ab") as handle:
                    handle.write(b'{"late":true}\n')
        return "CLOSED"


class _OpenOnlyInspector:
    def __init__(self, open_path: Path) -> None:
        self.open_path = open_path

    def state(self, path: Path) -> str:
        return "OPEN" if path == self.open_path else "CLOSED"


class _MutateLogOnSecondCheckInspector:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.source_checks = 0

    def state(self, path: Path) -> str:
        if path == self.source:
            self.source_checks += 1
            if self.source_checks == 2:
                with path.open("ab") as handle:
                    handle.write(b"late append\n")
        return "CLOSED"


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
            source = _history_file(root)
            source.write_text('{"time":"2026-06-01T00:00:00Z"}\n' * 20, encoding="utf-8")
            _make_old(source)

            result, payload = _run(root, "--min-size-mb", "0", "--min-age-minutes", "0")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(source.exists())
            self.assertFalse(source.with_name(f"{source.name}.gz").exists())
            self.assertEqual(payload["summary"]["compression_candidates"], 1)
            self.assertEqual(payload["summary"]["compressed_files"], 0)

    def test_apply_compresses_and_verifies_history_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _history_file(root)
            expected = ''.join(f'{{"row":{index}}}\n' for index in range(200))
            source.write_text(expected, encoding="utf-8")
            _make_old(source)

            result, payload = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )

            gz_path = source.with_name(f"{source.name}.gz")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(source.exists())
            self.assertTrue(gz_path.exists())
            with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), expected)
            row = payload["compressed"][0]
            self.assertEqual(row["status"], "compressed_verified")
            self.assertEqual(row["source_sha256"], row["verified_sha256"])
            self.assertEqual(row["row_count"], 200)

    def test_existing_corrupt_gzip_never_causes_plain_source_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _history_file(root)
            source.write_text('{"row":1}\n', encoding="utf-8")
            source.with_name(f"{source.name}.gz").write_bytes(b"not-gzip")
            _make_old(source)

            result, payload = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(source.exists())
            reasons = [item.get("reason", "") for item in payload["skipped_safety"]]
            self.assertTrue(any(reason.startswith("EXISTING_GZIP_INVALID") for reason in reasons))

    def test_existing_mismatched_gzip_never_causes_plain_source_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _history_file(root)
            source.write_text('{"row":1}\n', encoding="utf-8")
            with gzip.open(source.with_name(f"{source.name}.gz"), "wt", encoding="utf-8") as handle:
                handle.write('{"row":2}\n')
            _make_old(source)

            result, payload = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(source.exists())
            self.assertIn(
                "EXISTING_GZIP_CONTENT_MISMATCH",
                [item.get("reason") for item in payload["skipped_safety"]],
            )

    def test_post_publish_source_race_removes_only_new_stale_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _history_file(root)
            source.write_text('{"row":1}\n' * 100, encoding="utf-8")
            plan = DISK_MAINTENANCE._plan(
                source,
                source.parent,
                source.lstat(),
                10_000,
                "replay_jsonl",
            )

            with self.assertRaisesRegex(DISK_MAINTENANCE.MaintenanceSkip, "FILE_CHANGED_SINCE_SCAN"):
                DISK_MAINTENANCE._compress_jsonl(
                    plan,
                    _MutateSourceOnSecondCheckInspector(source),
                )

            self.assertTrue(source.exists())
            self.assertIn("late", source.read_text(encoding="utf-8"))
            self.assertFalse(source.with_name(f"{source.name}.gz").exists())

    def test_apply_removes_stale_history_temp_but_preserves_young_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = _history_file(root).parent
            stale = history / "EUR_USD_S5_BA.jsonl.gz.partial"
            fresh = history / "EUR_USD_S5_BA.jsonl.gz.tmp"
            stale.write_text("partial", encoding="utf-8")
            fresh.write_text("tmp", encoding="utf-8")
            _make_old(stale, days=3)

            result, _ = _run(root, "--apply", "--prune-temp-days", "2")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())

    def test_project_atomic_cleanup_requires_newer_canonical_and_protects_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir(parents=True)
            canonical = data / "runtime_state.json"
            debris = data / "runtime_state.json.tmp"
            canonical.write_text("new", encoding="utf-8")
            debris.write_text("old", encoding="utf-8")
            _make_old(debris, days=3)

            missing_target = data / "orphan_state.json.partial"
            missing_target.write_text("keep", encoding="utf-8")
            _make_old(missing_target, days=3)

            ledger = data / "execution_ledger.db"
            ledger_debris = data / "execution_ledger.db.tmp"
            ledger.write_bytes(b"ledger")
            ledger_debris.write_bytes(b"old-ledger")
            _make_old(ledger_debris, days=3)

            forecast = data / "forecast_history.jsonl"
            forecast_debris = data / "forecast_history.jsonl.bak"
            forecast.write_text("current", encoding="utf-8")
            forecast_debris.write_text("old", encoding="utf-8")
            _make_old(forecast_debris, days=3)

            result, payload = _run(root, "--apply", "--prune-atomic-hours", "48")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(debris.exists())
            self.assertTrue(canonical.exists())
            self.assertTrue(missing_target.exists())
            self.assertTrue(ledger.exists())
            self.assertTrue(ledger_debris.exists())
            self.assertTrue(forecast.exists())
            self.assertTrue(forecast_debris.exists())
            self.assertEqual(payload["summary"]["removed_project_atomic_files"], 1)

    def test_project_atomic_equal_mtime_does_not_prove_supersession(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir(parents=True)
            canonical = data / "state.json"
            debris = data / "state.json.tmp"
            canonical.write_text("possibly broken", encoding="utf-8")
            debris.write_text("recoverable", encoding="utf-8")
            timestamp = time.time() - 10 * 24 * 60 * 60
            os.utime(canonical, (timestamp, timestamp))
            os.utime(debris, (timestamp, timestamp))

            result, payload = _run(root, "--apply", "--prune-atomic-hours", "48")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(debris.exists())
            self.assertIn(
                "NO_NEWER_CANONICAL_TARGET",
                {item.get("reason") for item in payload["skipped_safety"]},
            )

    def test_project_backup_honors_longer_configured_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir(parents=True)
            canonical = data / "state.json"
            backup = data / "state.json.bak"
            backup.write_text("rollback", encoding="utf-8")
            _make_old(backup, days=10)
            canonical.write_text("current", encoding="utf-8")

            result, _ = _run(root, "--apply", "--prune-atomic-hours", "720")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(backup.exists())

    def test_project_atomic_revalidates_canonical_before_debris_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir(parents=True)
            canonical = data / "state.json"
            debris = data / "state.json.tmp"
            debris.write_text("recoverable", encoding="utf-8")
            _make_old(debris, days=3)
            canonical.write_text("current", encoding="utf-8")
            plans = DISK_MAINTENANCE._project_atomic_candidates(
                root,
                now=datetime.now(timezone.utc),
                min_age=timedelta(hours=48),
                report={},
            )
            self.assertEqual(len(plans), 1)
            replacement = data / "replacement.json"
            replacement.write_text("changed", encoding="utf-8")
            replacement.replace(canonical)

            with self.assertRaisesRegex(
                DISK_MAINTENANCE.MaintenanceSkip,
                "CANONICAL_TARGET_CHANGED_SINCE_SCAN",
            ):
                DISK_MAINTENANCE._safe_unlink(plans[0], _ClosedInspector())

            self.assertTrue(debris.exists())

    def test_project_atomic_young_file_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir(parents=True)
            (data / "state.json").write_text("current", encoding="utf-8")
            young = data / "state.json.tmp"
            young.write_text("in-progress", encoding="utf-8")

            result, _ = _run(root, "--apply", "--prune-atomic-hours", "48")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(young.exists())

    def test_external_diagnostic_allowlist_is_strict_and_test_root_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            allowed = temp_root / "qr_execution_ledger_profit_profile.db"
            not_diagnostic = temp_root / "qr_execution_ledger.db"
            wrong_prefix = temp_root / "other_profit_profile.db"
            substring_only = temp_root / "qr_unprofiled_trade_evidence.db"
            plural_only = temp_root / "qr_customer_profiles.db"
            concatenated_only = temp_root / "qr_customerprofile.db"
            immediate_token = temp_root / "qr_benchmark.db"
            allowed.write_bytes(b"profile")
            not_diagnostic.write_bytes(b"evidence")
            wrong_prefix.write_bytes(b"other")
            substring_only.write_bytes(b"evidence")
            plural_only.write_bytes(b"customer")
            concatenated_only.write_bytes(b"customer")
            immediate_token.write_bytes(b"benchmark")
            for path in (
                allowed,
                not_diagnostic,
                wrong_prefix,
                substring_only,
                plural_only,
                concatenated_only,
                immediate_token,
            ):
                _make_old(path, days=2)
            report: dict[str, object] = {}

            plans = DISK_MAINTENANCE._external_diagnostic_candidates(
                temp_root,
                now=datetime.now(timezone.utc),
                min_age=timedelta(hours=1),
                trusted_temp_root=temp_root,
                report=report,
            )

            self.assertEqual(
                [plan.path for plan in plans],
                sorted([allowed.resolve(), immediate_token.resolve()]),
            )
            for plan in plans:
                DISK_MAINTENANCE._safe_unlink(plan, _ClosedInspector())
            self.assertFalse(allowed.exists())
            self.assertFalse(immediate_token.exists())
            self.assertTrue(not_diagnostic.exists())
            self.assertTrue(wrong_prefix.exists())
            self.assertTrue(substring_only.exists())
            self.assertTrue(plural_only.exists())
            self.assertTrue(concatenated_only.exists())

    def test_external_diagnostic_root_cannot_escape_trusted_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            with self.assertRaises(ValueError):
                DISK_MAINTENANCE._external_diagnostic_candidates(
                    Path(tmp),
                    now=datetime.now(timezone.utc),
                    min_age=timedelta(),
                    trusted_temp_root=Path(other),
                    report={},
                )

    def test_guardian_cache_expires_but_sessions_and_state_db_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guardian = root / "data" / "codex_guardian_home"
            cache = guardian / "cache" / "codex_apps_tools" / "old.json"
            session = guardian / "sessions" / "2026" / "07" / "wake.jsonl"
            state_db = guardian / "state_5.sqlite"
            cache.parent.mkdir(parents=True)
            session.parent.mkdir(parents=True)
            cache.write_text("cache", encoding="utf-8")
            session.write_text("session evidence", encoding="utf-8")
            state_db.write_bytes(b"state")
            for path in (cache, session, state_db):
                _make_old(path, days=10)

            result, payload = _run(root, "--apply", "--guardian-cache-hours", "48")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(cache.exists())
            self.assertTrue(session.exists())
            self.assertTrue(state_db.exists())
            self.assertEqual(
                payload["effective_policy"]["guardian_sessions"],
                "PRESERVED_AUDIT_EVIDENCE",
            )

    def test_only_known_launchd_log_is_rotated_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir(parents=True)
            known = logs / "guardian_wake_dispatcher.launchd.log"
            unknown = logs / "trader_journal.jsonl"
            expected = ("known launchd line\n" * 150_000).encode("utf-8")
            known.write_bytes(expected)
            unknown.write_text("execution evidence\n", encoding="utf-8")

            result, payload = _run(root, "--apply", "--log-max-mb", "0.001")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(known.exists())
            self.assertEqual(known.stat().st_size, 0)
            generation = known.with_name(f"{known.name}.1.gz")
            self.assertTrue(generation.exists())
            with gzip.open(generation, "rb") as handle:
                self.assertEqual(handle.read(), expected)
            self.assertEqual(unknown.read_text(encoding="utf-8"), "execution evidence\n")
            self.assertEqual(payload["summary"]["rotated_logs"], 1)

    def test_open_existing_log_generation_blocks_rotation_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir(parents=True)
            source = logs / "guardian_wake_dispatcher.launchd.log"
            generation = source.with_name(f"{source.name}.1.gz")
            source.write_bytes(b"new log\n" * 100)
            with gzip.open(generation, "wb") as handle:
                handle.write(b"prior generation\n")
            plan = DISK_MAINTENANCE._plan(
                source,
                logs,
                source.lstat(),
                0.0,
                "launchd_log",
            )

            with self.assertRaisesRegex(DISK_MAINTENANCE.MaintenanceSkip, "OPEN_FILE_SKIPPED"):
                DISK_MAINTENANCE._rotate_log(plan, _OpenOnlyInspector(generation))

            self.assertGreater(source.stat().st_size, 0)
            with gzip.open(generation, "rb") as handle:
                self.assertEqual(handle.read(), b"prior generation\n")

    def test_log_rotation_recovers_verified_crash_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir(parents=True)
            source = logs / "guardian_wake_dispatcher.launchd.log"
            expected = b"recoverable live log\n" * 100
            source.write_bytes(expected)
            temp_generation = source.with_name(f"{source.name}.1.gz.tmp")
            with gzip.open(temp_generation, "wb") as handle:
                handle.write(expected)
            plan = DISK_MAINTENANCE._plan(
                source,
                logs,
                source.lstat(),
                0.0,
                "launchd_log",
            )

            result = DISK_MAINTENANCE._rotate_log(plan, _ClosedInspector())

            self.assertEqual(result["status"], "rotated_verified_gzip")
            self.assertEqual(source.stat().st_size, 0)
            self.assertFalse(temp_generation.exists())
            with gzip.open(source.with_name(f"{source.name}.1.gz"), "rb") as handle:
                self.assertEqual(handle.read(), expected)

    def test_log_append_race_preserves_source_and_prior_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir(parents=True)
            source = logs / "guardian_wake_dispatcher.launchd.log"
            expected = b"current log\n" * 100
            source.write_bytes(expected)
            generation = source.with_name(f"{source.name}.1.gz")
            with gzip.open(generation, "wb") as handle:
                handle.write(b"prior generation\n")
            plan = DISK_MAINTENANCE._plan(
                source,
                logs,
                source.lstat(),
                0.0,
                "launchd_log",
            )

            with self.assertRaisesRegex(DISK_MAINTENANCE.MaintenanceSkip, "FILE_CHANGED_SINCE_SCAN"):
                DISK_MAINTENANCE._rotate_log(
                    plan,
                    _MutateLogOnSecondCheckInspector(source),
                )

            self.assertEqual(source.read_bytes(), expected + b"late append\n")
            with gzip.open(source.with_name(f"{source.name}.1.gz"), "rb") as handle:
                self.assertEqual(handle.read(), expected)
            with gzip.open(source.with_name(f"{source.name}.2.gz"), "rb") as handle:
                self.assertEqual(handle.read(), b"prior generation\n")

    def test_apply_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _history_file(root)
            source.write_text('{"same":true}\n' * 100, encoding="utf-8")
            _make_old(source)

            first, _ = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )
            second, payload = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(payload["summary"]["compressed_files"], 0)
            self.assertEqual(payload["summary"]["reclaimed_bytes_estimate"], 0)

    def test_single_lock_rejects_overlapping_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "logs" / ".qr_disk_maintenance.lock"
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--root",
                        str(root),
                        "--allow-p0-no-reclaim",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn("already running", result.stdout)

    def test_apply_returns_nonzero_when_p0_has_no_safe_reclaim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                apply=True,
                allow_p0_no_reclaim=False,
                guardian_cache_hours=168.0,
                guardian_temp_hours=24.0,
                history_dir=[DISK_MAINTENANCE.DEFAULT_HISTORY_DIR],
                log_max_mb=8.0,
                min_age_minutes=30.0,
                min_size_mb=1.0,
                prune_atomic_hours=48.0,
                prune_diagnostic_hours=6.0,
                prune_external_diagnostics=False,
                prune_temp_days=2.0,
                report_path=DISK_MAINTENANCE.DEFAULT_REPORT_PATH,
                root=root,
            )
            fake_disk = {
                "total_bytes": 100 * 1024**3,
                "used_bytes": 99 * 1024**3,
                "free_bytes": 1 * 1024**3,
            }

            with (
                patch.object(DISK_MAINTENANCE, "_disk_usage", return_value=fake_disk),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = DISK_MAINTENANCE._run_locked(args, root)

            payload = json.loads(
                (root / DISK_MAINTENANCE.DEFAULT_REPORT_PATH).read_text(encoding="utf-8")
            )
            self.assertEqual(result, 3)
            self.assertEqual(payload["summary"]["status"], "P0_NO_SAFE_RECLAIM")
            self.assertEqual(payload["summary"]["reclaimed_bytes_estimate"], 0)

    def test_apply_still_returns_nonzero_when_reclaim_does_not_clear_p0_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "data" / "stale.tmp"
            canonical = root / "data" / "stale"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"x" * 1024)
            canonical.write_text("current", encoding="utf-8")
            _make_old(stale, days=3)
            args = argparse.Namespace(
                apply=True,
                allow_p0_no_reclaim=False,
                guardian_cache_hours=168.0,
                guardian_temp_hours=24.0,
                history_dir=[DISK_MAINTENANCE.DEFAULT_HISTORY_DIR],
                log_max_mb=8.0,
                min_age_minutes=30.0,
                min_size_mb=1.0,
                prune_atomic_hours=48.0,
                prune_diagnostic_hours=6.0,
                prune_external_diagnostics=False,
                prune_temp_days=2.0,
                report_path=DISK_MAINTENANCE.DEFAULT_REPORT_PATH,
                root=root,
            )
            fake_disk = {
                "total_bytes": 100 * 1024**3,
                "used_bytes": 99 * 1024**3,
                "free_bytes": 1 * 1024**3,
            }

            with (
                patch.object(DISK_MAINTENANCE, "_disk_usage", return_value=fake_disk),
                patch.object(DISK_MAINTENANCE.OpenFileInspector, "state", return_value="CLOSED"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = DISK_MAINTENANCE._run_locked(args, root)

            payload = json.loads(
                (root / DISK_MAINTENANCE.DEFAULT_REPORT_PATH).read_text(encoding="utf-8")
            )
            self.assertEqual(result, 3)
            self.assertFalse(stale.exists())
            self.assertEqual(payload["summary"]["status"], "P0_LOW_SPACE")
            self.assertGreater(payload["summary"]["reclaimed_bytes_estimate"], 0)

    def test_p0_shortens_replay_temp_retention_to_six_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = _history_file(root).with_suffix(".jsonl.tmp")
            stale.write_bytes(b"orphaned replay temp")
            seven_hours_ago = time.time() - 7 * 60 * 60
            os.utime(stale, (seven_hours_ago, seven_hours_ago))
            args = argparse.Namespace(
                apply=True,
                allow_p0_no_reclaim=True,
                guardian_cache_hours=168.0,
                guardian_temp_hours=24.0,
                history_dir=[DISK_MAINTENANCE.DEFAULT_HISTORY_DIR],
                log_max_mb=8.0,
                min_age_minutes=30.0,
                min_size_mb=1.0,
                prune_atomic_hours=48.0,
                prune_diagnostic_hours=6.0,
                prune_external_diagnostics=False,
                prune_temp_days=2.0,
                report_path=DISK_MAINTENANCE.DEFAULT_REPORT_PATH,
                root=root,
            )
            fake_disk = {
                "total_bytes": 100 * 1024**3,
                "used_bytes": 99 * 1024**3,
                "free_bytes": 1 * 1024**3,
            }

            with (
                patch.object(DISK_MAINTENANCE, "_disk_usage", return_value=fake_disk),
                patch.object(DISK_MAINTENANCE.OpenFileInspector, "state", return_value="CLOSED"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = DISK_MAINTENANCE._run_locked(args, root)

            payload = json.loads(
                (root / DISK_MAINTENANCE.DEFAULT_REPORT_PATH).read_text(encoding="utf-8")
            )
            self.assertEqual(result, 0)
            self.assertFalse(stale.exists())
            self.assertEqual(payload["effective_policy"]["replay_temp_age_seconds"], 6 * 60 * 60)
            self.assertEqual(payload["summary"]["removed_stale_temp_files"], 1)

    def test_history_path_traversal_fails_without_touching_outside_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp)
            outside = Path(other) / "outside.jsonl"
            outside.write_text("keep\n", encoding="utf-8")
            _make_old(outside)

            result, _ = _run(
                root,
                "--apply",
                "--history-dir",
                str(Path(other)),
                expect_report=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep\n")

    def test_report_path_traversal_fails_without_writing_outside(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp)
            outside = Path(other) / "report.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--report-path",
                    str(outside),
                    "--allow-p0-no-reclaim",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(outside.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unavailable")
    def test_symlink_candidate_is_ignored_and_target_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp)
            history = _history_file(root).parent
            outside = Path(other) / "outside.jsonl"
            outside.write_text("keep\n", encoding="utf-8")
            link = history / "linked.jsonl"
            link.symlink_to(outside)

            result, _ = _run(
                root,
                "--apply",
                "--min-size-mb",
                "0",
                "--min-age-minutes",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(link.is_symlink())
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep\n")

    def test_open_file_plan_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "old.tmp"
            path.write_text("open", encoding="utf-8")
            file_stat = path.lstat()
            plan = DISK_MAINTENANCE._plan(path, root, file_stat, 10_000, "test")

            with self.assertRaisesRegex(DISK_MAINTENANCE.MaintenanceSkip, "OPEN_FILE_SKIPPED"):
                DISK_MAINTENANCE._safe_unlink(plan, _OpenInspector())

            self.assertTrue(path.exists())

    def test_inode_or_mtime_change_after_scan_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "old.tmp"
            path.write_text("old", encoding="utf-8")
            plan = DISK_MAINTENANCE._plan(path, root, path.lstat(), 10_000, "test")
            replacement = root / "replacement"
            replacement.write_text("replacement", encoding="utf-8")
            replacement.replace(path)

            with self.assertRaisesRegex(DISK_MAINTENANCE.MaintenanceSkip, "FILE_CHANGED_SINCE_SCAN"):
                DISK_MAINTENANCE._safe_unlink(plan, _ClosedInspector())

            self.assertEqual(path.read_text(encoding="utf-8"), "replacement")

    def test_report_contains_two_and_five_gib_operating_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result, payload = _run(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(payload["operating_floor_bytes"], 5 * 1024**3)
            self.assertEqual(payload["p0_free_bytes"], 2 * 1024**3)
            self.assertIn(payload["pressure_before"], {"OK", "BELOW_OPERATING_FLOOR", "P0"})
            self.assertIn(payload["pressure_after"], {"OK", "BELOW_OPERATING_FLOOR", "P0"})
            self.assertEqual(
                payload["summary"]["bytes_to_operating_floor"],
                payload["bytes_to_operating_floor"],
            )


def _history_file(root: Path) -> Path:
    history = root / "logs" / "replay" / "oanda_history" / "run" / "EUR_USD"
    history.mkdir(parents=True, exist_ok=True)
    return history / "EUR_USD_S5_BA_20260601T000000Z_20260701T000000Z.jsonl"


def _run(
    root: Path,
    *args: str,
    expect_report: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    report = root / "logs" / "disk_maintenance_report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(root),
        "--report-path",
        str(report.relative_to(root)),
        "--allow-p0-no-reclaim",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_report and report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
    else:
        payload = {}
    return result, payload


def _make_old(path: Path, *, days: int = 1) -> None:
    timestamp = time.time() - days * 24 * 60 * 60
    os.utime(path, (timestamp, timestamp))


if __name__ == "__main__":
    unittest.main()
