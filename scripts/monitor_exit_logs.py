#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Iterable

LOG_DIR = pathlib.Path("logs")
DEFAULT_LOG = LOG_DIR / "cloudrun_fx_trader_stderr.log"
MARKERS = ("[exit_manager]", "[EXIT_MANAGER]", "GPT adjust", "GPT close")


def _iter_lines(path: pathlib.Path, follow: bool) -> Iterable[str]:
    if follow and shutil.which("tail"):
        proc = subprocess.Popen(["tail", "-n", "200", "-f", str(path)], stdout=subprocess.PIPE)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                yield line.decode("utf-8", errors="ignore")
        finally:
            proc.terminate()
        return

    # manual follow implementation
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        if follow:
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                yield line
        else:
            for line in fh.readlines()[-500:]:
                yield line


def filter_lines(lines: Iterable[str], keywords: Iterable[str]) -> Iterable[str]:
    lowered = [kw.lower() for kw in keywords]
    for line in lines:
        text = line.strip()
        ltext = text.lower()
        if any(kw in ltext for kw in lowered):
            yield text


def run_local(path: pathlib.Path, follow: bool) -> None:
    if not path.exists():
        print(f"Log file not found: {path}", file=sys.stderr)
        sys.exit(1)
    for line in filter_lines(_iter_lines(path, follow), MARKERS):
        print(line)


def run_cloud(project: str, service: str, limit: int) -> None:
    cmd = [
        "gcloud",
        "logging",
        "read",
        f"resource.type=cloud_run_revision AND resource.labels.service_name={service} AND textPayload:\"exit_manager\"",
        f"--limit={limit}",
        "--format=value(textPayload)",
        "--project",
        project,
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.output or exc, file=sys.stderr)
        sys.exit(exc.returncode)
    for line in out.splitlines():
        if line.strip():
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor GPT exit manager logs (local or Cloud Logging)")
    ap.add_argument("--cloud", action="store_true", help="Use gcloud logging instead of local file")
    ap.add_argument("--project", default="quantrabbit")
    ap.add_argument("--service", default="fx-trader")
    ap.add_argument("--limit", type=int, default=200, help="Cloud log row limit")
    ap.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    ap.add_argument("--follow", action="store_true", help="Follow local log (tail -f)")
    args = ap.parse_args()

    if args.cloud:
        run_cloud(args.project, args.service, args.limit)
    else:
        run_local(args.log, args.follow)


if __name__ == "__main__":
    main()
