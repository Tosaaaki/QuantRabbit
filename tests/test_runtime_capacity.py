from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from quant_rabbit.runtime_capacity import (
    CapacityPolicy,
    CapacityReceiptError,
    CapacityStatus,
    RootQuota,
    build_capacity_receipt,
    capture_size_snapshot,
    content_digest_unchanged,
    evaluate_capacity,
    measure_cycle_size_delta,
    measure_tree_size,
    read_capacity_receipt,
    read_capacity_receipt_fail_closed,
    write_capacity_receipt,
)


class RuntimeCapacityTest(unittest.TestCase):
    def test_filesystem_watermark_thresholds(self) -> None:
        policy = CapacityPolicy(Path("/runtime"), low_free_bytes=100, high_free_bytes=200)

        self.assertEqual(_evaluate(policy, free=250).status, CapacityStatus.OK)
        self.assertEqual(_evaluate(policy, free=200).status, CapacityStatus.OK)
        self.assertEqual(_evaluate(policy, free=199).status, CapacityStatus.PRESSURE)
        self.assertEqual(_evaluate(policy, free=100).status, CapacityStatus.PRESSURE)
        self.assertEqual(_evaluate(policy, free=99).status, CapacityStatus.BLOCK)

    def test_root_quota_can_raise_pressure_or_block(self) -> None:
        quota = RootQuota("reports", Path("/reports"), pressure_bytes=400, block_bytes=500)
        policy = CapacityPolicy(
            Path("/runtime"),
            low_free_bytes=100,
            high_free_bytes=200,
            root_quotas=(quota,),
        )

        pressure = evaluate_capacity(
            policy,
            disk_usage_reader=lambda _: _usage(250),
            root_size_reader=lambda _: 450,
        )
        blocked = evaluate_capacity(
            policy,
            disk_usage_reader=lambda _: _usage(250),
            root_size_reader=lambda _: 500,
        )

        self.assertEqual(pressure.status, CapacityStatus.PRESSURE)
        self.assertIn("reports:ROOT_QUOTA_PRESSURE", pressure.issues)
        self.assertEqual(blocked.status, CapacityStatus.BLOCK)
        self.assertIn("reports:ROOT_QUOTA_REACHED", blocked.issues)

    def test_missing_or_unreadable_measurements_fail_closed(self) -> None:
        quota = RootQuota("cache", Path("/missing"), pressure_bytes=10, block_bytes=20)
        policy = CapacityPolicy(
            Path("/runtime"),
            low_free_bytes=100,
            high_free_bytes=200,
            root_quotas=(quota,),
        )

        assessment = evaluate_capacity(
            policy,
            disk_usage_reader=lambda _: (_ for _ in ()).throw(OSError("unreadable")),
            root_size_reader=lambda _: (_ for _ in ()).throw(PermissionError("denied")),
        )

        self.assertEqual(assessment.status, CapacityStatus.BLOCK)
        self.assertIsNone(assessment.free_bytes)
        self.assertTrue(any(issue.startswith("FILESYSTEM_STATS_UNAVAILABLE") for issue in assessment.issues))
        self.assertTrue(any(issue.startswith("cache:ROOT_STATS_UNAVAILABLE") for issue in assessment.issues))

    def test_latest_receipt_is_atomic_stable_and_skips_unchanged_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity.json"
            assessment = _evaluate(
                CapacityPolicy(Path(tmp), low_free_bytes=100, high_free_bytes=200),
                free=250,
            )
            first_time = datetime(2026, 9, 4, 1, 0, tzinfo=timezone.utc)
            second_time = datetime(2026, 9, 4, 1, 10, tzinfo=timezone.utc)

            self.assertTrue(write_capacity_receipt(path, assessment, observed_at=first_time))
            first_bytes = path.read_bytes()
            first_stat = path.stat()
            self.assertFalse(write_capacity_receipt(path, assessment, observed_at=second_time))

            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertEqual(path.stat().st_mtime_ns, first_stat.st_mtime_ns)
            self.assertEqual(read_capacity_receipt(path)["status"], "OK")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_tampered_receipt_fails_closed_and_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity.json"
            assessment = _evaluate(
                CapacityPolicy(Path(tmp), low_free_bytes=100, high_free_bytes=200),
                free=250,
            )
            payload = build_capacity_receipt(assessment)
            payload["status"] = "BLOCK"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(CapacityReceiptError) as caught:
                read_capacity_receipt(path)
            self.assertEqual(caught.exception.code, "RECEIPT_TAMPERED")
            self.assertEqual(read_capacity_receipt_fail_closed(path)["status"], "BLOCK")
            with self.assertRaises(CapacityReceiptError):
                write_capacity_receipt(path, assessment)

    def test_receipt_timestamp_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity.json"
            assessment = _evaluate(
                CapacityPolicy(Path(tmp), low_free_bytes=100, high_free_bytes=200),
                free=250,
            )
            payload = build_capacity_receipt(assessment)
            payload["observed_at_utc"] = "2099-01-01T00:00:00Z"
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                read_capacity_receipt_fail_closed(path)["issues"],
                ["RECEIPT_TAMPERED"],
            )

    def test_content_digest_helper_detects_exact_no_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"stable")

            self.assertTrue(content_digest_unchanged(path, b"stable"))
            self.assertFalse(content_digest_unchanged(path, b"changed"))
            self.assertFalse(content_digest_unchanged(path.parent / "missing", b"stable"))

    def test_cycle_delta_and_metadata_only_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = capture_size_snapshot({"data": root}, root_size_reader=lambda _: 10)
            after = capture_size_snapshot({"data": root}, root_size_reader=lambda _: 25)

            delta = measure_cycle_size_delta(before, after)

            self.assertEqual(delta["total_delta_bytes"], 15)
            self.assertEqual(delta["roots"][0]["delta_bytes"], 15)

    def test_tree_measurement_does_not_follow_symlinks(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unsupported")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured = root / "measured"
            outside = root / "outside"
            measured.mkdir()
            outside.mkdir()
            (measured / "local.bin").write_bytes(b"local")
            (outside / "secret.bin").write_bytes(b"not-counted")
            (measured / "outside-link").symlink_to(outside, target_is_directory=True)

            self.assertEqual(measure_tree_size(measured), len(b"local"))
            with self.assertRaises(OSError):
                measure_tree_size(measured / "outside-link")

    def test_latest_only_receipt_has_bounded_growth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "capacity.json"
            policy = CapacityPolicy(root, low_free_bytes=100, high_free_bytes=200)

            for index in range(80):
                assessment = _evaluate(policy, free=250 if index % 2 == 0 else 150)
                write_capacity_receipt(
                    path,
                    assessment,
                    observed_at=datetime(2026, 9, 4, 1, index % 60, tzinfo=timezone.utc),
                )

            files = [candidate for candidate in root.iterdir() if candidate.is_file()]
            self.assertEqual(files, [path])
            self.assertLess(path.stat().st_size, 16 * 1024)
            self.assertIn(read_capacity_receipt(path)["status"], {"OK", "PRESSURE"})

    def test_symlink_receipt_path_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unsupported")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "capacity.json"
            link.symlink_to(target)

            self.assertEqual(
                read_capacity_receipt_fail_closed(link)["issues"],
                ["RECEIPT_PATH_SYMLINK"],
            )


def _usage(free: int) -> SimpleNamespace:
    total = 1_000
    return SimpleNamespace(total=total, used=total - free, free=free)


def _evaluate(policy: CapacityPolicy, *, free: int):
    return evaluate_capacity(policy, disk_usage_reader=lambda _: _usage(free))


if __name__ == "__main__":
    unittest.main()
