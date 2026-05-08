#!/usr/bin/env python3
"""Post cycle anomalies (BLOCKED status / new gap reports) to Slack `#qr-commands`.

Reads:
- `docs/autotrade_cycle_report.md` — parses the front-matter bullets to
  extract `Status`, daily target, and GPT verdict. Posts when status
  signals a hard block (e.g. anything other than the success / WAIT /
  HOLD_PROTECTED / SENT / GPT_* family) or when the cycle ended without
  taking action while the daily target is open and `LIVE_READY` lanes
  exist (campaign-exposure occupancy hint).
- `docs/gap_report_*.md` — posts when a new gap report appears since the
  marker, citing its severity and 1-line summary.

State markers:
- `logs/.slack_cycle_last_status` (last cycle generated_at + status hash)
- `logs/.slack_cycle_seen_gaps` (list of gap report filenames already posted)

Designed to be called from `scripts/run-autotrade-live.sh` after the
cycle and ledger sync. Idempotent.

Usage:
    python3 tools/slack_cycle_alert.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slack_post import load_slack_config, post_message  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cycle_report_path() -> Path:
    return _repo_root() / "docs" / "autotrade_cycle_report.md"


def _docs_dir() -> Path:
    return _repo_root() / "docs"


def _status_marker_path() -> Path:
    return _repo_root() / "logs" / ".slack_cycle_last_status"


def _gap_marker_path() -> Path:
    return _repo_root() / "logs" / ".slack_cycle_seen_gaps"


# Status values that mean the cycle worked as intended (ok to stay quiet).
QUIET_STATUSES = {
    "SENT",
    "STAGED",
    "WAIT",
    "REQUEST_EVIDENCE",
    "HOLD_PROTECTED",
    "NO_ACTION",
    "TARGET_REACHED",
    "TARGET_HIT",
    "GPT_CLOSE",
    "GPT_CANCEL_PENDING",
    "GPT_WAIT",
    "CLOSED_GPT_TRADES",
    "CANCELED_GPT_PENDING",
    "OK",
}


def _parse_bullet(text: str, key: str) -> str | None:
    pattern = rf"-\s+{re.escape(key)}:\s+`([^`]*)`"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _parse_cycle_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return {
        "generated_at": _parse_bullet(text, "Generated at UTC"),
        "status": _parse_bullet(text, "Status"),
        "positions": _parse_bullet(text, "Positions"),
        "orders": _parse_bullet(text, "Orders"),
        "live_ready": _parse_bullet(text, "Live-ready intents"),
        "deterministic_lane": _parse_bullet(text, "Deterministic lane"),
        "selected_lane": _parse_bullet(text, "Selected lane"),
        "sent": _parse_bullet(text, "Sent"),
        "position_management": _parse_bullet(text, "Position management"),
        "daily_target": _parse_bullet(text, "Daily target"),
        "gpt_action": _extract_gpt_action(text),
        "gpt_status": _extract_gpt_status(text),
    }


def _extract_gpt_action(text: str) -> str | None:
    match = re.search(r"GPT trader:[^\n]*action=`([^`]*)`", text)
    return match.group(1) if match else None


def _extract_gpt_status(text: str) -> str | None:
    match = re.search(r"GPT trader:\s*status=`([^`]*)`", text)
    return match.group(1) if match else None


def _is_alert_status(status: str | None) -> bool:
    if not status:
        return False
    upper = status.upper()
    if upper in QUIET_STATUSES:
        return False
    # Anything containing BLOCK / FAIL / ERROR / REJECT / STALE always alerts.
    for needle in ("BLOCK", "FAIL", "ERROR", "REJECT", "STALE", "MISSING"):
        if needle in upper:
            return True
    # CYCLE_HALT / GATEWAY_HALT / RISK_HALT etc.
    if "HALT" in upper:
        return True
    # Otherwise: unrecognized status → alert (better safe than silent).
    return True


def _format_cycle_alert(report: dict[str, Any]) -> str:
    status = report.get("status") or "UNKNOWN"
    when = report.get("generated_at") or "?"
    lines = [f"🚧 *Cycle alert: {status}*  [{when}]"]
    if report.get("deterministic_lane"):
        lines.append(f"  lane: {report['deterministic_lane']}")
    if report.get("daily_target"):
        lines.append(f"  daily target: {report['daily_target']}")
    if report.get("gpt_action") or report.get("gpt_status"):
        lines.append(f"  GPT: status={report.get('gpt_status')} action={report.get('gpt_action')}")
    if report.get("positions") or report.get("orders") or report.get("live_ready"):
        lines.append(
            f"  positions={report.get('positions')} orders={report.get('orders')} live_ready={report.get('live_ready')}"
        )
    return "\n".join(lines)


def _format_gap_alert(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    severity_match = re.search(r"\*\*Severity\*\*:\s*([^\n]+)", text)
    summary = ""
    summary_match = re.search(r"##\s+Summary\s+(.+?)(?=\n##|\n\Z)", text, re.DOTALL)
    if summary_match:
        first_para = summary_match.group(1).strip().split("\n\n", 1)[0]
        summary = first_para.replace("\n", " ").strip()
        if len(summary) > 280:
            summary = summary[:277] + "…"
    title = title_match.group(1) if title_match else path.name
    severity = severity_match.group(1).strip() if severity_match else "?"
    return f"⚠️ *Gap report: {title}*  ({severity})\n  {summary}\n  file: `{path.name}`"


def _read_status_marker() -> str:
    p = _status_marker_path()
    return p.read_text().strip() if p.exists() else ""


def _write_status_marker(value: str) -> None:
    p = _status_marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value)


def _read_gap_marker() -> set[str]:
    p = _gap_marker_path()
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError):
        return set()


def _write_gap_marker(seen: set[str]) -> None:
    p = _gap_marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(seen)))


def _new_gap_reports(seen: set[str]) -> list[Path]:
    docs = _docs_dir()
    if not docs.exists():
        return []
    out: list[Path] = []
    for path in sorted(docs.glob("gap_report_*.md")):
        if path.name not in seen:
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of posting")
    args = parser.parse_args()

    slack = load_slack_config()
    token = slack.get("QR_SLACK_BOT_TOKEN")
    channel = slack.get("QR_SLACK_CHANNEL_COMMANDS") or slack.get("QR_SLACK_CHANNEL_ID")
    if not args.dry_run and (not token or not channel):
        print("ERROR: QR_SLACK_BOT_TOKEN and QR_SLACK_CHANNEL_COMMANDS required", file=sys.stderr)
        sys.exit(2)

    posted = 0
    would_post = 0

    # 1) Cycle status alert.
    report = _parse_cycle_report(_cycle_report_path())
    if report and report.get("status"):
        last_marker = _read_status_marker()
        marker_value = f"{report.get('generated_at')}|{report.get('status')}"
        if marker_value != last_marker and _is_alert_status(report["status"]):
            text = _format_cycle_alert(report)
            if args.dry_run:
                print(text)
                print("---")
                would_post += 1
            else:
                post_message(text, channel, token)
                posted += 1
        if not args.dry_run:
            _write_status_marker(marker_value)

    # 2) New gap reports.
    seen = _read_gap_marker()
    new_reports = _new_gap_reports(seen)
    for path in new_reports:
        text = _format_gap_alert(path)
        if args.dry_run:
            print(text)
            print("---")
            would_post += 1
        else:
            post_message(text, channel, token)
            posted += 1
        seen.add(path.name)
    if not args.dry_run and new_reports:
        _write_gap_marker(seen)

    if args.dry_run:
        print(f"[dry-run] would post {would_post} alert(s)")
    elif posted:
        print(f"OK: posted {posted} cycle alert(s)")


if __name__ == "__main__":
    main()
