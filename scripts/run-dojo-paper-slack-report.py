#!/usr/bin/env python3
"""Send one verified PAPER hourly reply through the approved Irori helper only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_paired_model_queue import (
    canonical_json_bytes,
    canonical_sha256,
)

HELPER: Final = Path(
    "/Users/tossaki/.claude/scheduled-tasks/_shared/post_slack.sh"
)
PARENT_STATE: Final = "slack-paper-parent.json"
EXPECTED_TEAM: Final = "T0ANC64KY4V"
EXPECTED_BOT: Final = "U0AQ00RG709"
MENTION: Final = "<@U0AN9749B1R>"


class PaperSlackReportError(RuntimeError):
    """The approved Slack helper or its verified result is invalid."""


def _read(path: Path) -> dict[str, Any]:
    if path == Path("-"):
        value = json.load(sys.stdin)
        if not isinstance(value, dict):
            raise PaperSlackReportError("stdin JSON root must be object")
        return value
    with path.resolve(strict=True).open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PaperSlackReportError(f"JSON root must be object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verified_parent_state(root: Path) -> dict[str, Any] | None:
    path = root / PARENT_STATE
    if not path.exists():
        return None
    state = _read(path)
    unsigned = {key: item for key, item in state.items() if key != "state_sha256"}
    if (
        state.get("contract") != "QR_DOJO_PAPER_SLACK_PARENT_V1"
        or state.get("route") != "qr"
        or not isinstance(state.get("parent_ts"), str)
        or not state["parent_ts"]
        or state.get("authority") != "NONE"
        or state.get("state_sha256") != canonical_sha256(unsigned)
    ):
        raise PaperSlackReportError("PAPER Slack parent state is invalid")
    return state


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "未計測"
    return f"{float(value):,.2f}"


def _pct(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "未計測"
    return f"{float(value) * 100:.2f}%"


def _detail(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    decision = report["decision"]
    rooms = report["rooms"]
    room_lines = "\n".join(
        (
            f"{room['room_id']}: {room['position_summary']} / "
            f"uPL {_fmt(room['unrealized_pl_jpy'])} / "
            f"AI {room['last_ai_action']}"
        )
        for room in rooms
    )
    return (
        f"{MENTION}\n"
        "実行主体: Codex\n"
        "*:memo: 結論*\n"
        f"data_at JST: {report['data_at_jst']}\n"
        f"全室 net P/L {_fmt(totals['net_pl_jpy'])} JPY / "
        f"unrealized {_fmt(totals['unrealized_pl_jpy'])} / "
        f"realized {_fmt(totals['realized_pl_jpy'])}\n"
        f"TP gross {_fmt(totals['tp_gross_jpy'])} / "
        f"通常損失 {_fmt(totals['normal_loss_jpy'])} / "
        f"margin pressure {_pct(totals['margin_pressure'])}\n\n"
        "*:hammer_and_wrench: 実施*\n"
        f"{room_lines}\n\n"
        "*:white_check_mark: 確認*\n"
        f"AI action: {decision['action']}\n"
        f"reason: {decision['reason']}\n"
        f"前回差: {_fmt(totals['previous_balance_delta_jpy'])} JPY\n"
        "authority NONE / shadow only / broker・order mutationなし\n\n"
        "*:warning: 要観察 / リスク*\n"
        "未計測値は推測していません。economic resultがUNDETERMINEDなら"
        "黒字判定は未確定です。\n\n"
        "*:dart: 次アクション*\n"
        f"next review: {decision['next_review']}"
    )


def send_report(*, root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Invoke only the approved helper and verify its returned end state."""

    if report.get("authority") != "NONE" or report.get("room_count") != 4:
        raise PaperSlackReportError("PAPER report authority/room count is invalid")
    report_sha = str(report.get("report_sha256") or "")
    if report_sha != canonical_sha256(
        {
            key: item
            for key, item in report.items()
            if key not in {"report_sha256", "inventory_report_markdown"}
        }
    ):
        raise PaperSlackReportError("PAPER report content seal is invalid")
    operation_id = f"qr-paper-hourly:{report_sha}"
    detail = _detail(report)
    parent_state = _verified_parent_state(root)
    environment = os.environ.copy()
    environment.update(
        {
            "IRORI_REPORT_OPERATION_ID": operation_id,
            "IRORI_REPORT_JSON_OUTPUT": "1",
        }
    )
    if parent_state is not None:
        environment["IRORI_REPORT_PARENT_TS"] = parent_state["parent_ts"]
    result = subprocess.run(
        [
            "/bin/bash",
            str(HELPER),
            "qr",
            "-",
            "QR DOJO PAPER｜AI inventory supervisor",
        ],
        input=detail,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=90,
    )
    if result.returncode != 0:
        raise PaperSlackReportError(
            f"approved Slack helper failed: {result.stderr.strip()[:500]}"
        )
    try:
        verified = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PaperSlackReportError("approved Slack helper returned invalid JSON") from exc
    if (
        not isinstance(verified, dict)
        or verified.get("ok") is not True
        or verified.get("verified") is not True
        or verified.get("identity_team_id") != EXPECTED_TEAM
        or verified.get("identity_user_id") != EXPECTED_BOT
        or verified.get("channel_name") != "qr"
        or verified.get("operation_id") != operation_id
        or not verified.get("channel")
        or not verified.get("parent_ts")
        or not verified.get("reply_ts")
        or not verified.get("permalink")
        or detail not in str(verified.get("verified_reply_text") or "")
    ):
        raise PaperSlackReportError("approved Slack helper verification is incomplete")
    if parent_state is None:
        body = {
            "contract": "QR_DOJO_PAPER_SLACK_PARENT_V1",
            "schema_version": 1,
            "route": "qr",
            "channel": verified["channel"],
            "parent_ts": verified["parent_ts"],
            "permalink": verified["permalink"],
            "created_by_operation_id": operation_id,
            "authority": "NONE",
        }
        _write_exclusive(
            root / PARENT_STATE,
            {**body, "state_sha256": canonical_sha256(body)},
        )
    elif verified["parent_ts"] != parent_state["parent_ts"]:
        raise PaperSlackReportError("Slack helper returned a different PAPER parent")
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = send_report(root=args.root, report=_read(args.report))
    except (OSError, PaperSlackReportError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "verified": False,
                    "error": str(exc),
                    "blind_retry_allowed": False,
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
