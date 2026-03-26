#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_repo_history_lane_index import (
    build_repo_history_lane_payload,
)  # noqa: E402
from trade_findings_review import _entry_field, _parse_findings  # noqa: E402

DEFAULT_FINDINGS_PATH = REPO_ROOT / "docs" / "TRADE_FINDINGS.md"
DEFAULT_OUT_JSON = REPO_ROOT / "logs" / "trade_findings_index_latest.json"
DEFAULT_OUT_MD = REPO_ROOT / "logs" / "trade_findings_index_latest.md"
HEADING_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a derived index from TRADE_FINDINGS."
    )
    parser.add_argument("--path", default=str(DEFAULT_FINDINGS_PATH))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _entry_dt(heading: str) -> datetime | None:
    match = HEADING_DT_RE.match(heading)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _compact(text: str, max_chars: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    match = re.search(r"`([^`]+)`", text)
    if match:
        return match.group(1).strip()
    return text.lstrip("- ").strip()


def _lane_enrichment(
    lane_lookup: dict[str, dict[str, object]], key: str
) -> dict[str, object]:
    lane = lane_lookup.get(key) or {}
    if not isinstance(lane, dict):
        return {}
    return {
        "lane_family": lane.get("family_key") or "",
        "family_entries": lane.get("family_entries"),
        "family_distinct_hypothesis_keys": lane.get("family_distinct_hypothesis_keys"),
        "history_commit_count": lane.get("history_commit_count"),
        "repeat_risk": lane.get("repeat_risk") or "n/a",
        "repeat_risk_reasons": lane.get("repeat_risk_reasons") or [],
        "strategies": lane.get("strategies") or [],
    }


def main() -> int:
    args = _parse_args()
    path = Path(args.path).resolve()
    entries = _parse_findings(path)
    cutoff = datetime.now() - timedelta(days=max(args.days, 1))
    lane_payload = build_repo_history_lane_payload(path, max(args.limit, 50))
    lane_lookup = (
        lane_payload.get("lanes") if isinstance(lane_payload.get("lanes"), dict) else {}
    )
    current_open_trading = (
        lane_payload.get("current_open_trading_lanes")
        if isinstance(lane_payload.get("current_open_trading_lanes"), list)
        else []
    )
    recommended_focus = lane_payload.get("recommended_single_focus_lane")

    latest_by_key: dict[str, dict[str, object]] = {}
    recent_unresolved: list[dict[str, object]] = []
    driver_counts: Counter[str] = Counter()
    missing_key: list[dict[str, str]] = []

    for entry in entries:
        dt = _entry_dt(entry.heading)
        key = _normalize(_entry_field(entry, "Hypothesis Key"))
        verdict = _normalize(_entry_field(entry, "Verdict")).lower()
        status = _normalize(_entry_field(entry, "Status")).lower()
        driver = _compact(_entry_field(entry, "Primary Loss Driver"), 120)
        if key and key not in latest_by_key:
            latest_by_key[key] = {
                "heading": entry.heading,
                "verdict": verdict,
                "status": status,
                "primary_loss_driver": driver,
                "next_action": _compact(_entry_field(entry, "Next Action"), 160),
            }
            latest_by_key[key].update(_lane_enrichment(lane_lookup, key))
        if dt and dt >= cutoff and driver:
            driver_counts[driver] += 1
        if (
            dt
            and dt >= cutoff
            and (verdict in {"pending", "mixed"} or status in {"open", "in_progress"})
        ):
            unresolved_item: dict[str, object] = {
                "heading": entry.heading,
                "hypothesis_key": key,
                "verdict": verdict,
                "status": status,
                "primary_loss_driver": driver,
            }
            if key:
                unresolved_item.update(_lane_enrichment(lane_lookup, key))
            recent_unresolved.append(unresolved_item)
        if dt and dt >= cutoff and not key:
            missing_key.append({"heading": entry.heading})

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": (
            str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path)
        ),
        "days": args.days,
        "latest_by_hypothesis_key": latest_by_key,
        "recent_unresolved": recent_unresolved[: args.limit],
        "top_primary_loss_drivers": [
            {"primary_loss_driver": driver, "count": count}
            for driver, count in driver_counts.most_common(args.limit)
        ],
        "top_repeat_risk_open_lanes": current_open_trading[: args.limit],
        "recommended_single_focus_lane": recommended_focus,
        "recent_entries_missing_hypothesis_key": missing_key[: args.limit],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# TRADE_FINDINGS Index",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- source: `{payload['source']}`",
        f"- days: `{args.days}`",
        "",
        "## Recommended Single Focus Lane",
    ]
    if isinstance(recommended_focus, dict):
        md_lines.extend(
            [
                f"- `{recommended_focus['hypothesis_key']}` / strategy={', '.join(recommended_focus['strategies'])} / repeat_risk={recommended_focus['repeat_risk']} / history_commits={recommended_focus['history_commit_count']}",
                f"- reason: {recommended_focus['selection_reason']}",
            ]
        )
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## Top Repeat-Risk Open Lanes",
        ]
    )
    if current_open_trading:
        for item in current_open_trading[: args.limit]:
            md_lines.append(
                f"- `{item['hypothesis_key']}` / strategy={', '.join(item['strategies'])} / repeat_risk={item['repeat_risk']} / history_commits={item['history_commit_count']} / family={item['family_key']}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## Latest By Hypothesis Key",
        ]
    )
    if latest_by_key:
        for key, item in list(latest_by_key.items())[: args.limit]:
            md_lines.append(
                f"- `{key}`: {item['heading']} / verdict={item['verdict'] or 'n/a'} / status={item['status'] or 'n/a'} / driver={item['primary_loss_driver'] or 'n/a'} / repeat_risk={item.get('repeat_risk') or 'n/a'} / history_commits={item.get('history_commit_count') or 'n/a'} / family={item.get('lane_family') or 'n/a'}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(["", "## Recent Unresolved"])
    if recent_unresolved:
        for item in recent_unresolved[: args.limit]:
            md_lines.append(
                f"- {item['heading']} / key={item['hypothesis_key'] or 'n/a'} / verdict={item['verdict'] or 'n/a'} / status={item['status'] or 'n/a'} / driver={item['primary_loss_driver'] or 'n/a'} / repeat_risk={item.get('repeat_risk') or 'n/a'} / history_commits={item.get('history_commit_count') or 'n/a'} / family={item.get('lane_family') or 'n/a'}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(["", "## Top Primary Loss Drivers"])
    if driver_counts:
        for driver, count in driver_counts.most_common(args.limit):
            md_lines.append(f"- {driver}: `{count}`")
    else:
        md_lines.append("- none")

    md_lines.extend(["", "## Recent Entries Missing Hypothesis Key"])
    if missing_key:
        for item in missing_key[: args.limit]:
            md_lines.append(f"- {item['heading']}")
    else:
        md_lines.append("- none")

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("TRADE_FINDINGS index")
    print(f"source: {payload['source']}")
    print(f"out_json: {out_json}")
    print(f"out_md: {out_md}")
    print(f"latest_keys: {len(latest_by_key)}")
    print(f"recent_unresolved: {len(recent_unresolved)}")
    if isinstance(recommended_focus, dict):
        print(
            "recommended_single_focus_lane: "
            f"{recommended_focus['hypothesis_key']} "
            f"(repeat_risk={recommended_focus['repeat_risk']}, history_commits={recommended_focus['history_commit_count']})"
        )
    print(f"recent_missing_hypothesis_key: {len(missing_key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
