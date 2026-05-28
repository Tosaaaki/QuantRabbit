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
# POSITION_ACTION_SENT covers TP-replace / SL-trail / harvest cycles — the
# action already shows up in #qr-trades when material (or is suppressed as a
# TP nudge). Posting it here too was producing the lane=None / GPT=not used
# noise wall.
QUIET_STATUSES = {
    "SENT",
    "STAGED",
    "WAIT",
    "REQUEST_EVIDENCE",
    "HOLD_PROTECTED",
    "NO_ACTION",
    "NO_LIVE_READY_INTENT",
    "NO_TRADE",
    "TARGET_REACHED",
    "TARGET_HIT",
    "TARGET_REACHED_PROTECT",
    "GPT_CLOSE",
    "GPT_CANCEL_PENDING",
    "GPT_PROTECT",
    "GPT_TIGHTEN_SL",
    "GPT_WAIT",
    # The verifier can reject WAIT / request-evidence receipts as normal
    # audit feedback; autotrade exits 0 and reroutes on the next cycle.
    "GPT_REQUEST_EVIDENCE",
    "GPT_REJECTED",
    "CLOSED_GPT_TRADES",
    "CANCELED_GPT_PENDING",
    # Stale/self-contradicting trader pending cleanup is a successful
    # maintenance action. Reposting it every cycle to #qr-commands is noise.
    "CANCELED_CONTAMINATED_PENDING",
    "CANCELED_TARGET_REACHED_PENDING",
    "MONITOR_ONLY_EXPOSURE_OPEN",
    "POSITION_ACTION_SENT",
    "POSITION_ACTION_STAGED",
    "OK",
    # GPT picked outside the prefiltered basket — next cycle re-prefilters
    # and naturally recovers. cli.py treats it as exit 0. No human action.
    "GPT_DECISION_NOT_PREFILTERED",
}


def _parse_bullet(text: str, key: str) -> str | None:
    pattern = rf"-\s+{re.escape(key)}:\s+`([^`]*)`"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _line_after(text: str, label: str) -> str | None:
    """Return everything after `- {label}:` on its line, or None."""
    match = re.search(rf"^-\s+{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _first_backtick(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"`([^`]*)`", value)
    return match.group(1) if match else None


def _subfield(value: str | None, key: str) -> str | None:
    """Pull `key=`X`` from a parsed line."""
    if not value:
        return None
    match = re.search(rf"{re.escape(key)}=`([^`]*)`", value)
    return match.group(1) if match else None


def _parse_cycle_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    target_line = _line_after(text, "Daily target")
    gpt_line = _line_after(text, "GPT trader")
    return {
        "generated_at": _parse_bullet(text, "Generated at UTC"),
        "status": _parse_bullet(text, "Status"),
        "positions": _parse_bullet(text, "Positions"),
        "orders": _parse_bullet(text, "Orders"),
        "live_ready": _parse_bullet(text, "Live-ready intents"),
        "deterministic_lane": _parse_bullet(text, "Deterministic lane"),
        "selected_lane": _parse_bullet(text, "Selected lane"),
        "basket_lanes": _parse_bullet(text, "Selected basket lanes"),
        "sent": _parse_bullet(text, "Sent"),
        "canceled_orders": _parse_bullet(text, "Canceled orders"),
        "position_management": _parse_bullet(text, "Position management"),
        "target_status": _first_backtick(target_line),
        "target_remaining_jpy": _subfield(target_line, "remaining"),
        "target_progress_pct": _subfield(target_line, "progress_pct"),
        "gpt_status": _subfield(gpt_line, "status"),
        "gpt_action": _subfield(gpt_line, "action"),
        "gpt_allowed": _subfield(gpt_line, "allowed"),
        "gpt_issues": _subfield(gpt_line, "issues"),
        "gpt_error": _parse_bullet(text, "GPT error"),
        "gpt_recovery_source": _parse_bullet(text, "GPT recovery source"),
    }


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


STATUS_DIAGNOSIS: dict[str, str] = {
    "NO_LIVE_READY_INTENT": "発注可能レーンが0本 — prefilter 全弾き or 市況フィルター",
    "STALE_QUOTE_BLOCKED": "broker quote が古い — 接続 or 週末/祝日",
    "PRE_FLIGHT_GATE": "事前ガード発動 — 未コミット worktree や stale artifact",
    "GPT_REQUIRED_FOR_LIVE_SEND": "ライブ送信に `--use-gpt-trader` が必須",
    "BASKET_PAIR_COVERAGE_INCOMPLETE": "basket validator がペア網羅不足で停止",
    "PER_TRADE_RISK_BLOCKED": "1トレード当たり損失上限に引っかかった",
    "PORTFOLIO_RISK_BLOCKED": "ポートフォリオ risk 上限に引っかかった",
    "OPEN_POSITION_EXISTS": "重複玉ガード発動",
    "UNPROTECTED_POSITION": "SL/TP 未設定の玉が残っている",
}


def _format_cycle_alert(report: dict[str, Any]) -> str:
    status = report.get("status") or "UNKNOWN"
    when = report.get("generated_at") or "?"
    lines = [f"🚧 *Cycle alert: {status}*  [{when}]"]

    hint = STATUS_DIAGNOSIS.get(status.upper())
    if hint:
        lines.append(f"  → {hint}")

    target_pct = report.get("target_progress_pct")
    target_remaining = report.get("target_remaining_jpy")
    target_status = report.get("target_status")
    if target_status or target_pct:
        bits = []
        if target_status:
            bits.append(target_status)
        if target_pct:
            try:
                bits.append(f"{float(target_pct):.1f}%")
            except (TypeError, ValueError):
                bits.append(f"{target_pct}%")
        if target_remaining:
            try:
                bits.append(f"remaining {float(target_remaining):,.0f} JPY")
            except (TypeError, ValueError):
                bits.append(f"remaining {target_remaining} JPY")
        lines.append("  target: " + " · ".join(bits))

    gpt_bits = []
    if report.get("gpt_status"):
        gpt_bits.append(f"status={report['gpt_status']}")
    if report.get("gpt_action"):
        gpt_bits.append(f"action={report['gpt_action']}")
    issues = report.get("gpt_issues")
    if issues and issues not in {"0", "None"}:
        gpt_bits.append(f"issues={issues}")
    error = report.get("gpt_error")
    if error and error != "none":
        gpt_bits.append(f"error={error}")
    if gpt_bits:
        lines.append("  GPT: " + " ".join(gpt_bits))

    recovery = report.get("gpt_recovery_source")
    if recovery and recovery != "none":
        lines.append(f"  recovery: {recovery}")

    pos_action = report.get("position_management")
    if pos_action and pos_action != "none":
        lines.append(f"  position: {pos_action}")

    canceled = report.get("canceled_orders")
    if canceled and canceled != "none":
        lines.append(f"  canceled: {canceled}")

    basket = report.get("basket_lanes")
    selected = report.get("selected_lane")
    if selected and selected != "None":
        lines.append(f"  selected: {selected}")
    elif basket and basket != "none":
        lines.append(f"  basket: {basket}")

    lines.append(
        "  capacity: "
        f"positions={report.get('positions')} "
        f"orders={report.get('orders')} "
        f"live_ready={report.get('live_ready')}"
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
