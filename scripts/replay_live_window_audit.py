#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit replay coverage for live trade windows and optionally run standard replay.

This wrapper bridges live trade history (`trades.db`) and local replay assets.
For each requested worker, it:

1. resolves the canonical `strategy_tag`
2. loads live trades from `trades.db`
3. expands replay windows around each trade (with configurable pre/post minutes)
4. finds overlapping tick files from one or more globs
5. writes clipped tick JSONL per covered window
6. optionally runs the standard grouped replay command

The output is a JSON report that can be used as the single audit artifact for
coverage status and replay execution results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


UTC = timezone.utc
WORKER_TAGS = {
    "impulse_break_s5": "impulse_break_s5",
    "impulse_retest_s5": "impulse_retest_s5",
    "impulse_momentum_s5": "impulse_momentum_s5",
    "pullback_s5": "pullback_s5",
    "vwap_magnet_s5": "vwap_magnet_s5",
    "stop_run_reversal": "stop_run_reversal",
    "session_open": "session_open_breakout",
    "trend_breakout": "TrendBreakout",
    "pullback_continuation": "PullbackContinuation",
    "failed_break_reverse": "FailedBreakReverse",
}


@dataclass(frozen=True)
class LiveTrade:
    strategy_tag: str
    entry_time: Optional[datetime]
    open_time: datetime
    close_time: datetime


@dataclass(frozen=True)
class ReplayWindow:
    start: datetime
    end: datetime
    trade_count: int


@dataclass(frozen=True)
class TickFileSpan:
    path: Path
    start: datetime
    end: datetime


def _parse_iso(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        stamp = float(value)
        if stamp > 1.0e12:
            stamp = stamp / 1000.0
        return datetime.fromtimestamp(stamp, tz=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _split_csv(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        for token in str(raw or "").split(","):
            item = token.strip()
            if item and item not in out:
                out.append(item)
    return out


def _default_trades_db() -> Path:
    return REPO_ROOT / "logs" / "trades.db"


def _default_tick_patterns() -> List[str]:
    return [
        "logs/replay/USD_JPY/USD_JPY_ticks_*.jsonl",
        "logs/archive/replay.*.dir/USD_JPY/USD_JPY_ticks_*.jsonl",
        "tmp/USD_JPY_ticks_*.jsonl",
        "tmp/vm_ticks/logs/replay/USD_JPY/USD_JPY_ticks_*.jsonl",
        "tmp/vm_ticks/logs/archive/replay.*.dir/USD_JPY/USD_JPY_ticks_*.jsonl",
    ]


def _load_live_trades(trades_db: Path, workers: Sequence[str]) -> Dict[str, List[LiveTrade]]:
    worker_tags: Dict[str, str] = {}
    for worker in workers:
        strategy_tag = WORKER_TAGS.get(worker)
        if strategy_tag:
            worker_tags[worker] = strategy_tag
    results: Dict[str, List[LiveTrade]] = {worker: [] for worker in workers}
    if not worker_tags or not trades_db.exists():
        return results

    tags = list(dict.fromkeys(worker_tags.values()))
    placeholders = ",".join("?" for _ in tags)
    sql = (
        "SELECT strategy_tag, entry_time, open_time, close_time "
        f"FROM trades WHERE strategy_tag IN ({placeholders}) "
        "ORDER BY COALESCE(open_time, entry_time) ASC"
    )
    con = sqlite3.connect(str(trades_db))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql, tags).fetchall()
    finally:
        con.close()

    tag_to_worker = {tag: worker for worker, tag in worker_tags.items()}
    for row in rows:
        strategy_tag = str(row["strategy_tag"] or "").strip()
        worker = tag_to_worker.get(strategy_tag)
        if not worker:
            continue
        opened_at = _parse_iso(row["open_time"]) or _parse_iso(row["entry_time"])
        closed_at = _parse_iso(row["close_time"]) or opened_at
        if opened_at is None or closed_at is None:
            continue
        results[worker].append(
            LiveTrade(
                strategy_tag=strategy_tag,
                entry_time=_parse_iso(row["entry_time"]),
                open_time=opened_at,
                close_time=closed_at,
            )
        )
    return results


def _build_trade_windows(
    trades: Sequence[LiveTrade],
    *,
    pre_minutes: float,
    post_minutes: float,
) -> List[ReplayWindow]:
    if not trades:
        return []
    pre_delta = timedelta(minutes=max(0.0, float(pre_minutes)))
    post_delta = timedelta(minutes=max(0.0, float(post_minutes)))
    spans = sorted(
        (
            (trade.open_time - pre_delta, trade.close_time + post_delta)
            for trade in trades
        ),
        key=lambda item: item[0],
    )
    windows: List[ReplayWindow] = []
    current_start, current_end = spans[0]
    current_count = 1
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            current_count += 1
            continue
        windows.append(ReplayWindow(start=current_start, end=current_end, trade_count=current_count))
        current_start, current_end, current_count = start, end, 1
    windows.append(ReplayWindow(start=current_start, end=current_end, trade_count=current_count))
    return windows


def _expand_tick_globs(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in _split_csv(patterns):
        for match in glob(pattern, recursive=True):
            path = Path(match)
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
    files.sort()
    return files


def _tick_span(path: Path) -> Optional[TickFileSpan]:
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            ts = _parse_iso(payload.get("ts") or payload.get("timestamp"))
            if ts is None:
                continue
            if first_dt is None:
                first_dt = ts
            last_dt = ts
    if first_dt is None or last_dt is None:
        return None
    return TickFileSpan(path=path, start=first_dt, end=last_dt)


def _tick_spans(paths: Sequence[Path]) -> List[TickFileSpan]:
    spans: List[TickFileSpan] = []
    for path in paths:
        span = _tick_span(path)
        if span is not None:
            spans.append(span)
    return spans


def _overlapping_tick_files(window: ReplayWindow, spans: Sequence[TickFileSpan]) -> List[TickFileSpan]:
    return [
        span
        for span in spans
        if span.start <= window.end and span.end >= window.start
    ]


def _clip_ticks_to_window(
    spans: Sequence[TickFileSpan],
    *,
    window: ReplayWindow,
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen: set[str] = set()
    with out_path.open("w", encoding="utf-8") as out:
        for span in sorted(spans, key=lambda item: (item.start, item.path.name)):
            with span.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    ts = _parse_iso(payload.get("ts") or payload.get("timestamp"))
                    if ts is None or ts < window.start or ts > window.end:
                        continue
                    dedupe_key = json.dumps(
                        {
                            "ts": payload.get("ts") or payload.get("timestamp"),
                            "bid": payload.get("bid"),
                            "ask": payload.get("ask"),
                        },
                        sort_keys=True,
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    out.write(raw + "\n")
                    written += 1
    return written


def _run_replay(
    *,
    worker: str,
    ticks_path: Path,
    out_dir: Path,
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_exit_workers_groups.py"),
        "--ticks",
        str(ticks_path),
        "--workers",
        worker,
        "--no-hard-sl",
        "--exclude-end-of-replay",
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    summary_path = out_dir / "summary_all.json"
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "summary_all_path": str(summary_path) if summary_path.exists() else None,
    }


def _window_label(index: int, window: ReplayWindow) -> str:
    return f"{index:02d}_{window.start.strftime('%Y%m%dT%H%M%SZ')}_{window.end.strftime('%Y%m%dT%H%M%SZ')}"


def _required_tick_basenames(window: ReplayWindow, *, instrument: str = "USD_JPY") -> List[str]:
    basenames: List[str] = []
    cursor = window.start.date()
    end_date = window.end.date()
    while cursor <= end_date:
        basenames.append(f"{instrument}_ticks_{cursor.strftime('%Y%m%d')}.jsonl")
        cursor = cursor + timedelta(days=1)
    return basenames


def _with_warmup(window: ReplayWindow, *, replay_warmup_minutes: float) -> ReplayWindow:
    warmup_delta = timedelta(minutes=max(0.0, float(replay_warmup_minutes)))
    if warmup_delta <= timedelta(0):
        return window
    return ReplayWindow(
        start=window.start - warmup_delta,
        end=window.end,
        trade_count=window.trade_count,
    )


def _iso_utc_z(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _align_window_to_s5(window: ReplayWindow) -> tuple[datetime, datetime]:
    start_ts = math.floor(window.start.timestamp() / 5.0) * 5
    end_ts = math.ceil(window.end.timestamp() / 5.0) * 5
    if end_ts <= start_ts:
        end_ts = start_ts + 5
    return (
        datetime.fromtimestamp(start_ts, tz=UTC),
        datetime.fromtimestamp(end_ts, tz=UTC),
    )


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _fetch_candles_to_json(
    *,
    instrument: str,
    start: datetime,
    end: datetime,
    out_path: Path,
) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "fetch_candles.py"),
        "--instrument",
        instrument,
        "--granularity",
        "S5",
        "--start",
        _iso_utc_z(start),
        "--end",
        _iso_utc_z(end),
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(REPO_ROOT)
    )
    env.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    candle_count = 0
    if proc.returncode == 0 and out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            candles = payload.get("candles") or []
            if isinstance(candles, list):
                candle_count = len(candles)
        except Exception:
            candle_count = 0
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "candles_path": str(out_path) if out_path.exists() else None,
        "candle_count": candle_count,
    }


def _synth_candles_to_ticks(
    *,
    candles_path: Path,
    out_path: Path,
) -> Dict[str, object]:
    from sim.pseudo_cfg import SimCfg
    from sim.pseudo_ticks import synth_from_candles

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_path, density_info = synth_from_candles(str(candles_path), str(out_path), SimCfg())
    tick_count = _count_nonempty_lines(sim_path)
    return {
        "status": "completed" if tick_count > 0 else "empty_ticks",
        "ticks_path": str(sim_path),
        "tick_count": tick_count,
        "density": density_info,
    }


def _run_candle_sim_fallback(
    *,
    instrument: str,
    window: ReplayWindow,
    worker_out_dir: Path,
    label: str,
) -> Dict[str, object]:
    aligned_start, aligned_end = _align_window_to_s5(window)
    candles_path = worker_out_dir / f"{label}_candles_s5.json"
    ticks_path = worker_out_dir / f"{label}_candles_sim_ticks.jsonl"
    fallback: Dict[str, object] = {
        "enabled": True,
        "used": False,
        "status": "failed",
        "source": "s5_candles_pseudo_ticks",
        "window_start": aligned_start.isoformat(),
        "window_end": aligned_end.isoformat(),
        "candles_path": None,
        "generated_ticks_path": None,
        "candle_count": 0,
        "generated_tick_count": 0,
        "fetch": None,
        "simulation": None,
    }
    fetch_result = _fetch_candles_to_json(
        instrument=instrument,
        start=aligned_start,
        end=aligned_end,
        out_path=candles_path,
    )
    fallback["fetch"] = fetch_result
    fallback["candles_path"] = fetch_result.get("candles_path")
    fallback["candle_count"] = int(fetch_result.get("candle_count") or 0)
    if fetch_result.get("status") != "completed":
        fallback["status"] = "fetch_failed"
        return fallback
    if int(fetch_result.get("candle_count") or 0) <= 0:
        fallback["status"] = "empty_candles"
        return fallback
    try:
        simulation_result = _synth_candles_to_ticks(
            candles_path=candles_path,
            out_path=ticks_path,
        )
    except Exception as exc:
        fallback["status"] = "simulation_failed"
        fallback["simulation"] = {"status": "failed", "error": str(exc)}
        return fallback
    fallback["simulation"] = simulation_result
    fallback["generated_ticks_path"] = simulation_result.get("ticks_path")
    fallback["generated_tick_count"] = int(simulation_result.get("tick_count") or 0)
    if int(simulation_result.get("tick_count") or 0) <= 0:
        fallback["status"] = "empty_ticks"
        return fallback
    fallback["status"] = "completed"
    fallback["used"] = True
    return fallback


def build_report(
    *,
    workers: Sequence[str],
    trades_db: Path,
    tick_patterns: Sequence[str],
    pre_minutes: float,
    post_minutes: float,
    out_dir: Path,
    run_replay: bool,
    allow_candle_sim_fallback: bool = False,
    replay_warmup_minutes: float = 0.0,
) -> Dict[str, object]:
    tick_files = _expand_tick_globs(tick_patterns)
    spans = _tick_spans(tick_files)
    live_trades = _load_live_trades(trades_db, workers)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "trades_db": str(trades_db),
        "tick_patterns": _split_csv(tick_patterns),
        "tick_file_count": len(spans),
        "allow_candle_sim_fallback": bool(allow_candle_sim_fallback),
        "replay_warmup_minutes": max(0.0, float(replay_warmup_minutes)),
        "workers": {},
    }

    workers_payload: Dict[str, object] = {}
    for worker in workers:
        strategy_tag = WORKER_TAGS.get(worker)
        worker_trades = live_trades.get(worker, [])
        windows = _build_trade_windows(
            worker_trades,
            pre_minutes=pre_minutes,
            post_minutes=post_minutes,
        )
        window_payloads: List[Dict[str, object]] = []
        for index, window in enumerate(windows, start=1):
            label = _window_label(index, window)
            replay_window = _with_warmup(window, replay_warmup_minutes=replay_warmup_minutes)
            matched_spans = _overlapping_tick_files(window, spans)
            replay_spans = _overlapping_tick_files(replay_window, spans)
            coverage_status = "covered" if matched_spans else "missing"
            clipped_path = out_dir / worker / f"{label}_ticks.jsonl"
            replay_clip_path = out_dir / worker / f"{label}_replay_ticks.jsonl"
            clipped_tick_count = 0
            replay_ticks_path: Optional[Path] = None
            replay_tick_count = 0
            replay_result: Dict[str, object] = {"status": "skipped"}
            required_basenames = _required_tick_basenames(window)
            replay_required_basenames = _required_tick_basenames(replay_window)
            fallback_payload: Dict[str, object] = {
                "enabled": bool(allow_candle_sim_fallback),
                "used": False,
                "status": "not_needed" if matched_spans else "disabled",
                "source": "s5_candles_pseudo_ticks",
                "candles_path": None,
                "generated_ticks_path": None,
                "candle_count": 0,
                "generated_tick_count": 0,
                "fetch": None,
                "simulation": None,
            }
            if matched_spans:
                clipped_tick_count = _clip_ticks_to_window(
                    matched_spans,
                    window=window,
                    out_path=clipped_path,
                )
                if clipped_tick_count <= 0:
                    coverage_status = "missing"
                    replay_result = {"status": "skipped_no_ticks"}
                    try:
                        clipped_path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    replay_ticks_path = clipped_path
                    replay_tick_count = clipped_tick_count
                    if replay_window.start < window.start:
                        replay_tick_count = _clip_ticks_to_window(
                            replay_spans,
                            window=replay_window,
                            out_path=replay_clip_path,
                        )
                        if replay_tick_count > 0:
                            replay_ticks_path = replay_clip_path
                        else:
                            replay_ticks_path = clipped_path
                            replay_tick_count = clipped_tick_count
            if coverage_status == "missing" and allow_candle_sim_fallback:
                fallback_payload = _run_candle_sim_fallback(
                    instrument="USD_JPY",
                    window=replay_window,
                    worker_out_dir=out_dir / worker,
                    label=label,
                )
                if bool(fallback_payload.get("used")):
                    coverage_status = "candle_simulated"
                    ticks_path_text = str(fallback_payload.get("generated_ticks_path") or "").strip()
                    if ticks_path_text:
                        replay_ticks_path = Path(ticks_path_text)
                        replay_tick_count = int(fallback_payload.get("generated_tick_count") or 0)
            elif not allow_candle_sim_fallback:
                fallback_payload["status"] = "disabled"
            if replay_ticks_path is not None and run_replay:
                replay_result = _run_replay(
                    worker=worker,
                    ticks_path=replay_ticks_path,
                    out_dir=out_dir / worker / label,
                )
            window_payloads.append(
                {
                    "label": label,
                    "window_start": window.start.isoformat(),
                    "window_end": window.end.isoformat(),
                    "replay_window_start": replay_window.start.isoformat(),
                    "replay_window_end": replay_window.end.isoformat(),
                    "trade_count": window.trade_count,
                    "required_tick_basenames": required_basenames,
                    "replay_required_tick_basenames": replay_required_basenames,
                    "coverage": {
                        "status": coverage_status,
                        "tick_file_count": len(matched_spans),
                        "tick_files": [str(span.path) for span in matched_spans],
                        "replay_tick_file_count": len(replay_spans),
                        "replay_tick_files": [str(span.path) for span in replay_spans],
                        "fallback_enabled": bool(allow_candle_sim_fallback),
                        "fallback_used": bool(fallback_payload.get("used")),
                        "fallback": fallback_payload,
                    },
                    "clipped_ticks_path": str(clipped_path) if clipped_tick_count > 0 else None,
                    "clipped_tick_count": clipped_tick_count,
                    "replay_ticks_path": str(replay_ticks_path) if replay_ticks_path is not None else None,
                    "replay_tick_count": replay_tick_count,
                    "replay": replay_result,
                }
            )
        workers_payload[worker] = {
            "strategy_tag": strategy_tag,
            "live_trade_count": len(worker_trades),
            "window_count": len(window_payloads),
            "windows": window_payloads,
        }
    report["workers"] = workers_payload
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit exact replay coverage for live trade windows.")
    ap.add_argument("--workers", required=True, help="Comma-separated replay worker names")
    ap.add_argument(
        "--ticks-glob",
        action="append",
        default=None,
        help="Tick JSONL glob(s). Repeatable and comma-separated values are both accepted.",
    )
    ap.add_argument("--trades-db", type=Path, default=_default_trades_db())
    ap.add_argument("--pre-minutes", type=float, default=5.0)
    ap.add_argument("--post-minutes", type=float, default=15.0)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/replay_live_window_audit"))
    ap.add_argument("--run-replay", action="store_true")
    ap.add_argument("--allow-candle-sim-fallback", action="store_true")
    ap.add_argument("--replay-warmup-minutes", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        workers=_split_csv([args.workers]),
        trades_db=args.trades_db,
        tick_patterns=args.ticks_glob or _default_tick_patterns(),
        pre_minutes=args.pre_minutes,
        post_minutes=args.post_minutes,
        out_dir=args.out_dir,
        run_replay=bool(args.run_replay),
        allow_candle_sim_fallback=bool(args.allow_candle_sim_fallback),
        replay_warmup_minutes=float(args.replay_warmup_minutes or 0.0),
    )
    report_path = args.out_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(report_path))


if __name__ == "__main__":
    main()
