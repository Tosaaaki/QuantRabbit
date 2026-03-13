#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from trade_findings_review import _entry_field, _parse_findings  # noqa: E402


DEFAULT_FINDINGS_PATH = REPO_ROOT / "docs" / "TRADE_FINDINGS.md"
DEFAULT_STRICT_SINCE = "2026-03-13 20:00"
HEADING_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})")
KEY_RE = re.compile(r"^[a-z0-9_]+$")

REQUIRED_FIELDS = {
    "Hypothesis Key": ("Hypothesis Key",),
    "Primary Loss Driver": ("Primary Loss Driver",),
    "Mechanism Fired": ("Mechanism Fired",),
    "Do Not Repeat Unless": ("Do Not Repeat Unless",),
    "Change": ("Change",),
    "Why": ("Why", "Why/Hypothesis"),
    "Hypothesis": ("Hypothesis", "Why/Hypothesis"),
    "Expected Good": ("Expected Good",),
    "Expected Bad": ("Expected Bad",),
    "Period": ("Period",),
    "Fact": ("Fact", "Observed/Fact", "Observed"),
    "Failure Cause": ("Failure Cause",),
    "Improvement": ("Improvement",),
    "Verification": ("Verification",),
    "Verdict": ("Verdict",),
    "Next Action": ("Next Action",),
    "Status": ("Status",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint TRADE_FINDINGS entries for required fields.")
    parser.add_argument("--path", default=str(DEFAULT_FINDINGS_PATH))
    parser.add_argument("--strict-since", default=DEFAULT_STRICT_SINCE)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _entry_dt(heading: str) -> datetime | None:
    match = HEADING_DT_RE.match(heading)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _normalize_key(raw: str) -> str:
    text = " ".join(raw.split())
    if not text:
        return ""
    tick_match = re.search(r"`([^`]+)`", text)
    if tick_match:
        return tick_match.group(1).strip()
    text = text.lstrip("- ").strip()
    return text


def _issue(heading: str, kind: str, detail: str) -> dict[str, str]:
    return {"heading": heading, "kind": kind, "detail": detail}


def main() -> int:
    args = _parse_args()
    path = Path(args.path).resolve()
    strict_since = datetime.strptime(args.strict_since, "%Y-%m-%d %H:%M")
    entries = _parse_findings(path)

    strict_entries = []
    issues: list[dict[str, str]] = []
    for entry in entries:
        entry_dt = _entry_dt(entry.heading)
        if entry_dt is None or entry_dt < strict_since:
            continue
        strict_entries.append(entry)
        for label, aliases in REQUIRED_FIELDS.items():
            value = _entry_field(entry, *aliases).strip()
            if not value:
                issues.append(_issue(entry.heading, "missing_field", label))
        hypothesis_key = _normalize_key(_entry_field(entry, "Hypothesis Key"))
        if hypothesis_key and not KEY_RE.fullmatch(hypothesis_key):
            issues.append(_issue(entry.heading, "invalid_hypothesis_key", hypothesis_key))

    payload = {
        "source": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "strict_since": args.strict_since,
        "checked_entries": len(entries),
        "strict_entries": len(strict_entries),
        "issues": issues,
        "ok": not issues,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("TRADE_FINDINGS lint")
        print(f"source: {payload['source']}")
        print(f"strict_since: {payload['strict_since']}")
        print(f"checked_entries: {payload['checked_entries']}")
        print(f"strict_entries: {payload['strict_entries']}")
        print(f"ok: {'yes' if payload['ok'] else 'no'}")
        if issues:
            print("issues:")
            for issue in issues:
                print(f"- {issue['heading']}: {issue['kind']} {issue['detail']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
