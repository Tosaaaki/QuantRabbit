#!/usr/bin/env python3
"""Adaptive cycle runner for market playbook generation.

Cadence policy (default):
- normal: every 15m
- event pre-window (<=60m): every 5m
- event active window (-10m..+30m): every 1m
- post window (-120m..-10m): every 5m

The script is designed to be called frequently (e.g., every 1 minute by systemd timer)
and self-throttle via `state_path`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import fetch_market_snapshot
from scripts import gpt_ops_report

UTC = timezone.utc


@dataclass(frozen=True)
class CadenceConfig:
    normal_interval_sec: int = 15 * 60
    event_interval_sec: int = 5 * 60
    active_interval_sec: int = 60
    post_interval_sec: int = 5 * 60
    pre_window_min: int = 60
    active_before_min: int = 10
    active_after_min: int = 30
    post_window_min: int = 120


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_iso(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _event_minutes(event: dict[str, Any], *, now_utc: datetime) -> Optional[int]:
    if not isinstance(event, dict):
        return None
    if "minutes_to_event" in event:
        try:
            return int(float(event.get("minutes_to_event")))
        except Exception:
            pass
    dt = _parse_iso(event.get("time_utc"))
    if dt is None:
        return None
    return int(round((dt - now_utc).total_seconds() / 60.0))


def select_interval_sec(
    *,
    events: list[dict[str, Any]],
    now_utc: datetime,
    cfg: CadenceConfig,
) -> tuple[int, str, Optional[dict[str, Any]]]:
    minutes_list: list[tuple[int, dict[str, Any]]] = []
    for event in events:
        mins = _event_minutes(event, now_utc=now_utc)
        if mins is None:
            continue
        minutes_list.append((mins, event))

    # Active window has highest priority.
    for mins, event in minutes_list:
        if -cfg.active_before_min <= mins <= cfg.active_after_min:
            return cfg.active_interval_sec, "event_active", event

    # Pre-window before upcoming event.
    future = [(mins, event) for mins, event in minutes_list if 0 <= mins <= cfg.pre_window_min]
    if future:
        mins, event = min(future, key=lambda x: x[0])
        return cfg.event_interval_sec, "event_pre", event

    # Post-window after event.
    post = [(mins, event) for mins, event in minutes_list if -cfg.post_window_min <= mins < -cfg.active_before_min]
    if post:
        mins, event = max(post, key=lambda x: x[0])
        return cfg.post_interval_sec, "event_post", event

    return cfg.normal_interval_sec, "normal", (min(minutes_list, key=lambda x: abs(x[0]))[1] if minutes_list else None)


def _next_run_from_state(state_path: Path) -> Optional[datetime]:
    state = _read_json(state_path)
    return _parse_iso(state.get("next_run_utc"))


def _run_gpt_report(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/gpt_ops_report.py",
        "--hours",
        str(float(args.hours)),
        "--output",
        str(args.report_output),
        "--markdown-output",
        str(args.markdown_output),
        "--events-path",
        str(args.events_path),
        "--market-external-path",
        str(args.external_output),
        "--market-context-path",
        str(args.market_context_path),
        "--overlay-path",
        str(args.overlay_path),
        "--policy-output",
        str(args.policy_output),
        "--trades-db",
        str(args.trades_db),
        "--orders-db",
        str(args.orders_db),
    ]
    if args.policy:
        cmd.append("--policy")
    if args.apply_policy:
        cmd.append("--apply-policy")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _run_cycle(args: argparse.Namespace, now_utc: datetime) -> tuple[bool, str, list[dict[str, Any]]]:
    fetch_ok = True
    events_payload: dict[str, Any] = {}

    try:
        external, events_payload = fetch_market_snapshot.build_snapshot(now_utc)
        _write_json(Path(args.external_output), external)
        _write_json(Path(args.events_path), events_payload)
        logging.info("[PLAYBOOK_CYCLE] snapshot updated external=%s events=%s", args.external_output, args.events_path)
    except Exception as exc:
        fetch_ok = False
        logging.warning("[PLAYBOOK_CYCLE] market snapshot fetch failed: %s", exc)

    report_ok = True
    try:
        _run_gpt_report(args)
    except Exception as exc:
        report_ok = False
        logging.error("[PLAYBOOK_CYCLE] gpt_ops_report failed: %s", exc)

    events = gpt_ops_report._load_events(Path(args.events_path), now_utc=now_utc)

    status = "ok"
    if not fetch_ok and not report_ok:
        status = "fetch_and_report_failed"
    elif not fetch_ok:
        status = "fetch_failed"
    elif not report_ok:
        status = "report_failed"

    return report_ok, status, events


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptive market playbook cycle runner")
    ap.add_argument("--state-path", default="logs/ops_playbook_cycle_state.json")
    ap.add_argument("--external-output", default="logs/market_external_snapshot.json")
    ap.add_argument("--events-path", default="logs/market_events.json")
    ap.add_argument("--market-context-path", default="logs/market_context_latest.json")
    ap.add_argument("--report-output", default="logs/gpt_ops_report.json")
    ap.add_argument("--markdown-output", default="logs/gpt_ops_report.md")
    ap.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    ap.add_argument("--overlay-path", default="logs/policy_overlay.json")
    ap.add_argument("--trades-db", default="logs/trades.db")
    ap.add_argument("--orders-db", default="logs/orders.db")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--policy", action="store_true", default=False)
    ap.add_argument("--apply-policy", action="store_true", default=False)
    ap.add_argument("--normal-interval-sec", type=int, default=15 * 60)
    ap.add_argument("--event-interval-sec", type=int, default=5 * 60)
    ap.add_argument("--active-interval-sec", type=int, default=60)
    ap.add_argument("--post-interval-sec", type=int, default=5 * 60)
    ap.add_argument("--pre-window-min", type=int, default=60)
    ap.add_argument("--active-before-min", type=int, default=10)
    ap.add_argument("--active-after-min", type=int, default=30)
    ap.add_argument("--post-window-min", type=int, default=120)
    ap.add_argument("--force", action="store_true", default=False)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    now_utc = datetime.now(UTC)
    state_path = Path(args.state_path)

    next_run = _next_run_from_state(state_path)
    if not args.force and next_run is not None and now_utc < next_run:
        logging.info(
            "[PLAYBOOK_CYCLE] skip: next_run_utc=%s now_utc=%s",
            next_run.replace(microsecond=0).isoformat(),
            now_utc.replace(microsecond=0).isoformat(),
        )
        return 0

    ok, status, events = _run_cycle(args, now_utc)

    cfg = CadenceConfig(
        normal_interval_sec=max(60, int(args.normal_interval_sec)),
        event_interval_sec=max(60, int(args.event_interval_sec)),
        active_interval_sec=max(30, int(args.active_interval_sec)),
        post_interval_sec=max(60, int(args.post_interval_sec)),
        pre_window_min=max(1, int(args.pre_window_min)),
        active_before_min=max(0, int(args.active_before_min)),
        active_after_min=max(0, int(args.active_after_min)),
        post_window_min=max(1, int(args.post_window_min)),
    )
    interval_sec, phase, next_event = select_interval_sec(events=events, now_utc=now_utc, cfg=cfg)

    # If run failed, retry sooner.
    if not ok:
        interval_sec = min(interval_sec, cfg.event_interval_sec)

    next_run_utc = now_utc + timedelta(seconds=int(interval_sec))
    state = {
        "last_run_utc": now_utc.replace(microsecond=0).isoformat(),
        "next_run_utc": next_run_utc.replace(microsecond=0).isoformat(),
        "selected_interval_sec": int(interval_sec),
        "phase": phase,
        "status": status,
        "next_event": next_event,
    }
    _write_json(state_path, state)
    logging.info(
        "[PLAYBOOK_CYCLE] status=%s phase=%s next_interval_sec=%s next_run_utc=%s",
        status,
        phase,
        interval_sec,
        next_run_utc.replace(microsecond=0).isoformat(),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
