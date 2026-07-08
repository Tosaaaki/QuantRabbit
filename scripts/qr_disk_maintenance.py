#!/usr/bin/env python3
"""Bound QuantRabbit runtime disk growth.

The main production offender is OANDA replay history under
``logs/replay/oanda_history``. Replay readers already support ``.jsonl.gz``,
so this command safely converts old plain ``.jsonl`` candle files to gzip and
removes stale temp files. It never touches broker state or tracked reports.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_DIR = Path("logs/replay/oanda_history")
DEFAULT_REPORT_PATH = Path("logs/disk_maintenance_report.json")


@dataclass(frozen=True)
class FilePlan:
    path: Path
    size_bytes: int
    age_seconds: float


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    now = datetime.now(timezone.utc)
    before = _disk_usage(root)
    report: dict[str, Any] = {
        "generated_at_utc": _iso(now),
        "root": str(root),
        "apply": bool(args.apply),
        "min_age_minutes": float(args.min_age_minutes),
        "min_size_mb": float(args.min_size_mb),
        "history_dirs": [str(item) for item in args.history_dir],
        "disk_before": before,
        "compressed": [],
        "compression_candidates": [],
        "removed_stale_temp": [],
        "stale_temp_candidates": [],
        "errors": [],
    }

    candidates = _compression_candidates(
        root,
        args.history_dir,
        now=now,
        min_age=timedelta(minutes=float(args.min_age_minutes)),
        min_size_bytes=int(float(args.min_size_mb) * 1024 * 1024),
    )
    temp_candidates = _stale_temp_candidates(
        root,
        args.history_dir,
        now=now,
        min_age=timedelta(days=float(args.prune_temp_days)),
    )

    for plan in candidates:
        rel = _rel(plan.path, root)
        item: dict[str, Any] = {
            "path": rel,
            "size_bytes": plan.size_bytes,
            "age_seconds": round(plan.age_seconds, 3),
        }
        if args.apply:
            try:
                compressed = _compress_jsonl(plan.path)
                item.update(compressed)
                report["compressed"].append(item)
            except Exception as exc:  # noqa: BLE001 - maintenance reports and continues.
                item["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(item)
        else:
            report["compression_candidates"].append(item)

    for plan in temp_candidates:
        rel = _rel(plan.path, root)
        item = {
            "path": rel,
            "size_bytes": plan.size_bytes,
            "age_seconds": round(plan.age_seconds, 3),
        }
        if args.apply:
            try:
                plan.path.unlink()
                report["removed_stale_temp"].append(item)
            except OSError as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(item)
        else:
            report["stale_temp_candidates"].append(item)

    after = _disk_usage(root)
    report["disk_after"] = after
    report["summary"] = _summary(report)
    report_path = _resolve_under_root(root, args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(f"wrote {report_path}")
    return 1 if report["errors"] else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--history-dir",
        type=Path,
        action="append",
        default=[DEFAULT_HISTORY_DIR],
        help="history directory relative to --root unless absolute",
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-age-minutes", type=float, default=30.0)
    parser.add_argument("--min-size-mb", type=float, default=1.0)
    parser.add_argument("--prune-temp-days", type=float, default=2.0)
    parser.add_argument("--apply", action="store_true", help="perform compression/deletion; otherwise dry-run")
    return parser.parse_args()


def _compression_candidates(
    root: Path,
    history_dirs: list[Path],
    *,
    now: datetime,
    min_age: timedelta,
    min_size_bytes: int,
) -> list[FilePlan]:
    out: list[FilePlan] = []
    for history_dir in history_dirs:
        base = _resolve_under_root(root, history_dir)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.jsonl")):
            if not path.is_file() or _is_temp_path(path):
                continue
            gz_path = path.with_name(f"{path.name}.gz")
            if gz_path.exists():
                # The compressed copy is already present; removing the plain
                # duplicate still frees disk without changing replay coverage.
                stat = path.stat()
                age_seconds = _age_seconds(stat.st_mtime, now)
                if age_seconds >= min_age.total_seconds() and stat.st_size >= min_size_bytes:
                    out.append(FilePlan(path=path, size_bytes=stat.st_size, age_seconds=age_seconds))
                continue
            stat = path.stat()
            age_seconds = _age_seconds(stat.st_mtime, now)
            if age_seconds < min_age.total_seconds() or stat.st_size < min_size_bytes:
                continue
            out.append(FilePlan(path=path, size_bytes=stat.st_size, age_seconds=age_seconds))
    return out


def _stale_temp_candidates(
    root: Path,
    history_dirs: list[Path],
    *,
    now: datetime,
    min_age: timedelta,
) -> list[FilePlan]:
    out: list[FilePlan] = []
    suffixes = (".tmp", ".partial")
    for history_dir in history_dirs:
        base = _resolve_under_root(root, history_dir)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or not path.name.endswith(suffixes):
                continue
            stat = path.stat()
            age_seconds = _age_seconds(stat.st_mtime, now)
            if age_seconds < min_age.total_seconds():
                continue
            out.append(FilePlan(path=path, size_bytes=stat.st_size, age_seconds=age_seconds))
    return out


def _compress_jsonl(path: Path) -> dict[str, Any]:
    gz_path = path.with_name(f"{path.name}.gz")
    if gz_path.exists():
        original_size = path.stat().st_size
        gz_size = gz_path.stat().st_size
        path.unlink()
        return {
            "status": "removed_plain_duplicate",
            "gz_path": str(gz_path),
            "gz_size_bytes": gz_size,
            "reclaimed_bytes": original_size,
        }

    tmp_path = path.with_name(f"{path.name}.gz.tmp")
    tmp_path.unlink(missing_ok=True)
    original_stat = path.stat()
    try:
        with path.open("rb") as src, gzip.open(tmp_path, mode="wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.utime(tmp_path, (original_stat.st_atime, original_stat.st_mtime))
        tmp_path.replace(gz_path)
        gz_size = gz_path.stat().st_size
        path.unlink()
        return {
            "status": "compressed",
            "gz_path": str(gz_path),
            "gz_size_bytes": gz_size,
            "reclaimed_bytes": max(0, original_stat.st_size - gz_size),
        }
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    compressed = report.get("compressed") or []
    removed = report.get("removed_stale_temp") or []
    candidates = report.get("compression_candidates") or []
    temp_candidates = report.get("stale_temp_candidates") or []
    reclaimed = sum(
        int(item.get("reclaimed_bytes") or 0)
        for item in compressed
    )
    candidate_bytes = sum(int(item.get("size_bytes") or 0) for item in candidates)
    temp_bytes = sum(int(item.get("size_bytes") or 0) for item in removed)
    temp_candidate_bytes = sum(int(item.get("size_bytes") or 0) for item in temp_candidates)
    disk_after = report.get("disk_after") or {}
    return {
        "status": "ERROR" if report.get("errors") else "OK",
        "compressed_files": len(compressed),
        "compression_candidates": len(candidates),
        "removed_stale_temp_files": len(removed),
        "stale_temp_candidates": len(temp_candidates),
        "reclaimed_bytes_estimate": reclaimed + temp_bytes,
        "candidate_bytes": candidate_bytes + temp_candidate_bytes,
        "free_bytes_after": disk_after.get("free_bytes"),
    }


def _disk_usage(root: Path) -> dict[str, int]:
    usage = shutil.disk_usage(root)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _resolve_under_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _age_seconds(mtime: float, now: datetime) -> float:
    return max(0.0, now.timestamp() - mtime)


def _is_temp_path(path: Path) -> bool:
    return path.name.endswith(".tmp") or path.name.endswith(".partial")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
