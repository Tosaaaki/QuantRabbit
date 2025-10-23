#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuous_backtest.py
- 1 回の起動で「直近 N 日のローソクを拾って backtest + tuning」を実行
- systemd timer（hourly/daily）で定期実行する想定。重複起動を避けるためロックファイルを使用。
"""
import argparse, os, pathlib, sys, subprocess

from autotune.database import AUTOTUNE_BQ_TABLE, get_settings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = REPO_ROOT / "logs" / ".autotune.lock"
CANDLES_DIR = REPO_ROOT / "logs"
DEFAULT_GLOB = "candles_M1_*.json"

def acquire_lock() -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False

def release_lock():
    try:
        os.remove(LOCK)
    except FileNotFoundError:
        pass

def list_candles(pattern: str = DEFAULT_GLOB):
    files = sorted(CANDLES_DIR.glob(pattern))
    return files[-14:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="all", help="tuning profile (scalp|micro|macro|all)")
    ap.add_argument("--trials", type=int, default=0, help="override trials-per-strategy (0 = profile default)")
    ap.add_argument("--strategies", default="", help="explicit strategy list (single profile only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-best", action="store_true")
    args = ap.parse_args()

    if not acquire_lock():
        print("[WARN] another autotune instance is running. exit.")
        sys.exit(0)

    try:
        settings = get_settings()
        if not settings.get("enabled", True):
            print("[INFO] autotune disabled via settings. skipping.")
            return

        files = list_candles()
        if not files:
            print("[WARN] no candles found. nothing to do.")
            return

        tune_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tune_scalp.py"),
            "--profile",
            args.profile,
        ]
        if args.trials:
            tune_cmd.extend(["--trials-per-strategy", str(args.trials)])
        if args.strategies:
            tune_cmd.extend(["--strategies", args.strategies])
        bq_table = AUTOTUNE_BQ_TABLE.strip()
        if bq_table:
            tune_cmd.extend(["--bq-table", bq_table])
        if args.dry_run:
            tune_cmd.append("--dry-run")
        if args.write_best:
            tune_cmd.append("--write-best")
        print("[INFO] run:", " ".join(tune_cmd))
        r = subprocess.run(tune_cmd, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise SystemExit(r.returncode)
    finally:
        release_lock()

if __name__ == "__main__":
    main()
