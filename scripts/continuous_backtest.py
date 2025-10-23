#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuous_backtest.py
- 1 回の起動で「直近 N 日のローソクを拾って backtest + tuning」を実行
- systemd timer（hourly/daily）で定期実行する想定。重複起動を避けるためロックファイルを使用。
"""
import argparse, os, pathlib, sys, subprocess

from autotune.database import get_settings

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

def list_candles():
    files = sorted(CANDLES_DIR.glob(DEFAULT_GLOB))
    return files[-14:]  # 直近 2 週間ぶん程度

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--strategies", default="M1Scalper,PulseBreak,RangeFader")
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
            "--trials-per-strategy",
            str(args.trials),
            "--strategies",
            args.strategies,
        ]
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
