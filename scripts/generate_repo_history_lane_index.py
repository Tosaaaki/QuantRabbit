#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from trade_findings_review import _entry_field, _parse_findings  # noqa: E402


DEFAULT_FINDINGS_PATH = REPO_ROOT / "docs" / "TRADE_FINDINGS.md"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "REPO_HISTORY_LANE_INDEX.md"
DEFAULT_OUT_JSON = REPO_ROOT / "logs" / "repo_history_lane_index_latest.json"

ENTRY_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})")
BACKTICK_RE = re.compile(r"`([^`]+)`")
CAMEL_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+")

SEARCH_STOP_WORDS = {
    "after",
    "anti",
    "block",
    "change",
    "close",
    "commit",
    "current",
    "docs",
    "entry",
    "escalation",
    "family",
    "findings",
    "gate",
    "guard",
    "history",
    "hypothesis",
    "improvement",
    "index",
    "lane",
    "local",
    "loop",
    "loss",
    "next",
    "offline",
    "pending",
    "preflight",
    "proceed",
    "protocol",
    "pullback",
    "refine",
    "repeat",
    "review",
    "same",
    "script",
    "setup",
    "stop",
    "surface",
    "task",
    "time",
    "trade",
    "trigger",
    "worker",
    "wrapper",
}
NON_STRATEGY_TOKENS = {
    "TRADE_FINDINGS",
    "Hypothesis Key",
    "Primary Loss Driver",
    "Mechanism Fired",
    "Why Not Same As Last Time",
    "Promotion Gate",
    "Escalation Trigger",
    "STOP_LOSS_ORDER",
    "MARKET_ORDER_TRADE_CLOSE",
    "TAKE_PROFIT_ORDER",
    "entry_probability_reject",
    "entry_probability_below_min_units",
    "close_reject_profit_buffer",
    "local-v2",
    "trade_findings",
    "orders.db",
    "trades.db",
    "change_preflight.sh",
    "generate_repo_history_lane_index.py",
}
GOVERNANCE_KEY_PREFIXES = (
    "anti_loop",
    "trade_findings",
    "change_preflight",
    "preflight",
    "improvement_memory",
)
GOVERNANCE_SEARCH_PREFIXES: dict[str, tuple[str, ...]] = {
    "anti_loop": ("anti_loop", "anti-loop"),
    "trade_findings": (),
    "change_preflight": ("change_preflight", "change-preflight", "preflight"),
    "preflight": ("preflight_guard", "preflight", "change_preflight"),
    "improvement_memory": ("improvement_memory", "preflight", "dominant_loss_driver"),
}


@dataclass(frozen=True)
class CommitEntry:
    date: str
    short_hash: str
    subject: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class LaneEntry:
    key: str
    heading: str
    heading_dt: dt.datetime | None
    strategies: tuple[str, ...]
    primary_loss_driver: str
    verdict: str
    status: str
    next_action: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a repo-history lane cross-index from TRADE_FINDINGS."
    )
    parser.add_argument("--findings-path", default=str(DEFAULT_FINDINGS_PATH))
    parser.add_argument("--out-doc", default=str(DEFAULT_DOC_PATH))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def _is_history_maintenance_path(path: str) -> bool:
    return (
        path == "docs/INDEX.md"
        or path == "scripts/generate_repo_history_minutes.py"
        or path.startswith("docs/REPO_HISTORY_")
    )


def _git_history() -> list[CommitEntry]:
    out = subprocess.check_output(
        [
            "git",
            "log",
            "--reverse",
            "--date=short",
            "--pretty=format:@@@%ad\t%h\t%s",
            "--name-only",
        ],
        text=True,
        cwd=REPO_ROOT,
    )
    entries: list[CommitEntry] = []
    current: tuple[str, str, str] | None = None
    files: list[str] = []

    for line in out.splitlines() + ["@@@END\t\t"]:
        if line.startswith("@@@"):
            if current is not None:
                date, short_hash, subject = current
                if not files or not all(
                    _is_history_maintenance_path(path) for path in files
                ):
                    entries.append(
                        CommitEntry(
                            date=date,
                            short_hash=short_hash,
                            subject=subject,
                            files=tuple(files),
                        )
                    )
            if line.startswith("@@@END"):
                break
            current = tuple(line[3:].split("\t", 2))  # type: ignore[assignment]
            files = []
            continue
        if line.strip():
            files.append(line.strip())

    return entries


def _normalize(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    match = re.search(r"`([^`]+)`", text)
    if match:
        return match.group(1).strip()
    return text.lstrip("- ").strip()


def _plain_text(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*-\s*", "", line).replace("`", "").strip()
        if stripped:
            cleaned.append(stripped)
    return " ".join(cleaned).strip()


def _compact(text: str, max_chars: int = 96) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _heading_dt(heading: str) -> dt.datetime | None:
    match = ENTRY_DT_RE.match(heading)
    if not match:
        return None
    try:
        return dt.datetime.strptime(
            f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M"
        )
    except ValueError:
        return None


def _normalize_driver(raw: str) -> str:
    value = _plain_text(raw)
    lowered = value.lower()
    if "stop_loss_order" in lowered:
        return "STOP_LOSS_ORDER"
    if "entry_probability_reject" in lowered:
        return "entry_probability_reject"
    if "entry_probability_below_min_units" in lowered:
        return "entry_probability_below_min_units"
    if "close_reject_profit_buffer" in lowered:
        return "close_reject_profit_buffer"
    if "market_order_trade_close" in lowered:
        return "MARKET_ORDER_TRADE_CLOSE"
    if "take_profit_order" in lowered:
        return "TAKE_PROFIT_ORDER"
    return _compact(value or "n/a", 80)


def _canonical_strategy(token: str) -> str:
    token = token.strip().strip("`")
    token = re.sub(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_:-]+$", "", token)
    token = re.sub(r"[-_](long|short)$", "", token, flags=re.IGNORECASE)
    if token.endswith("-live"):
        token = token[:-5]
    if token.endswith("_live"):
        token = token[:-5]
    return token


def _is_strategy_like(token: str) -> bool:
    if not token or token in NON_STRATEGY_TOKENS:
        return False
    if " " in token:
        return False
    if "/" in token or token.endswith((".py", ".md", ".json", ".db")):
        return False
    if token.isdigit():
        return False
    if token[0].isdigit():
        return False
    lowered = token.lower()
    if lowered in {"pending", "done", "open", "mixed", "good", "bad"}:
        return False
    if lowered.startswith(("logs", "docs", "scripts")):
        return False
    if re.fullmatch(r"[A-Z0-9_]+", token) and token.endswith("_ORDER"):
        return False
    if lowered.startswith(("scalp_", "micro_", "session_open")):
        return True
    if any(part.isupper() for part in CAMEL_TOKEN_RE.findall(token) if part):
        return True
    return bool(re.search(r"[A-Z][a-z]", token))


def _extract_strategies(heading: str, raw_text: str, key: str) -> tuple[str, ...]:
    if "trade_findings:" in heading.lower():
        return ("trade_findings_governance",)

    found: list[str] = []

    def add(token: str) -> None:
        strategy = _canonical_strategy(token)
        if not strategy or not _is_strategy_like(strategy):
            return
        if strategy not in found:
            found.append(strategy)

    for token in BACKTICK_RE.findall(heading):
        add(token)
    if not found:
        for token in BACKTICK_RE.findall(raw_text):
            add(token)

    if not found:
        lowered_key = key.lower()
        if any(lowered_key.startswith(prefix) for prefix in GOVERNANCE_KEY_PREFIXES):
            found.append("trade_findings_governance")

    return tuple(found[:3])


def _term_variants(text: str) -> set[str]:
    lowered = text.lower()
    base = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    variants: set[str] = set()
    if base:
        variants.update(
            {
                base,
                base.replace(" ", "_"),
                base.replace(" ", "-"),
                base.replace(" ", ""),
            }
        )
    camel_parts = [part.lower() for part in CAMEL_TOKEN_RE.findall(text) if part]
    if camel_parts:
        joined = " ".join(camel_parts)
        variants.update(
            {
                joined,
                joined.replace(" ", "_"),
                joined.replace(" ", "-"),
                joined.replace(" ", ""),
            }
        )

    for suffix in ("_live", "-live", "_long", "-long", "_short", "-short", "-m1"):
        extra: set[str] = set()
        for item in list(variants):
            if item.endswith(suffix):
                extra.add(item[: -len(suffix)])
        variants.update(extra)

    return {item for item in variants if len(item.replace(" ", "")) >= 4}


def _key_terms(key: str) -> set[str]:
    parts = [part for part in key.lower().split("_") if part and not part.isdigit()]
    filtered = [
        part for part in parts if len(part) >= 4 and part not in SEARCH_STOP_WORDS
    ]
    combos: list[str] = []
    for idx in range(len(filtered) - 1):
        combo = f"{filtered[idx]}_{filtered[idx + 1]}"
        if len(combo.replace("_", "")) >= 8:
            combos.append(combo)
    return set(filtered + combos)


def _lane_search_terms(key: str, strategies: tuple[str, ...]) -> list[str]:
    terms: set[str] = set()
    non_governance = [
        strategy for strategy in strategies if strategy != "trade_findings_governance"
    ]
    for strategy in non_governance:
        terms.update(_term_variants(strategy))
    if not terms:
        lowered_key = key.lower()
        key_terms = _key_terms(key)
        for prefix, prefix_terms in GOVERNANCE_SEARCH_PREFIXES.items():
            if lowered_key.startswith(prefix):
                seed_terms = prefix_terms
                if prefix == "trade_findings" and key_terms:
                    seed_terms = tuple(sorted(key_terms))
                for term in prefix_terms:
                    terms.update(_term_variants(term))
                for term in seed_terms:
                    terms.update(_term_variants(term))
                break
        if not terms:
            for term in key_terms:
                terms.update(_term_variants(term))
    return sorted(terms, key=lambda item: (-len(item.replace(" ", "")), item))


def _history_matches(
    history: list[CommitEntry],
    terms: list[str],
    *,
    subject_only: bool = False,
) -> list[CommitEntry]:
    if not terms:
        return []
    matches: list[CommitEntry] = []
    for commit in history:
        haystack = commit.subject.lower()
        if not subject_only:
            haystack = " ".join((commit.subject, *commit.files)).lower()
        if any(term in haystack for term in terms):
            matches.append(commit)
    return matches


def _strategy_history_terms(strategy: str) -> tuple[list[str], bool]:
    if strategy == "trade_findings_governance":
        return (["trade_findings:", "anti-loop", "preflight", "lint"], True)
    return (_lane_search_terms(strategy, (strategy,)), False)


def _lane_entries(findings_path: Path) -> dict[str, list[LaneEntry]]:
    lanes: dict[str, list[LaneEntry]] = collections.defaultdict(list)
    for entry in _parse_findings(findings_path):
        key = _normalize(_entry_field(entry, "Hypothesis Key"))
        if not key:
            continue
        lanes[key].append(
            LaneEntry(
                key=key,
                heading=entry.heading,
                heading_dt=_heading_dt(entry.heading),
                strategies=_extract_strategies(entry.heading, entry.raw_text, key),
                primary_loss_driver=_normalize_driver(
                    _entry_field(entry, "Primary Loss Driver")
                ),
                verdict=_normalize(_entry_field(entry, "Verdict")).lower(),
                status=_normalize(_entry_field(entry, "Status")).lower(),
                next_action=_compact(_entry_field(entry, "Next Action"), 160),
            )
        )

    for key, items in lanes.items():
        lanes[key] = sorted(
            items,
            key=lambda item: (item.heading_dt or dt.datetime.min, item.heading),
        )
    return lanes


def _is_current_lane(entry: LaneEntry) -> bool:
    return entry.verdict in {"pending", "mixed"} or entry.status in {
        "open",
        "in_progress",
        "pending",
    }


def _build_payload(findings_path: Path, limit: int) -> dict[str, object]:
    history = _git_history()
    lanes = _lane_entries(findings_path)
    lane_payload: dict[str, dict[str, object]] = {}
    family_groups: dict[str, list[str]] = collections.defaultdict(list)
    strategy_groups: dict[str, list[str]] = collections.defaultdict(list)

    current_lanes: list[dict[str, object]] = []

    for key, items in lanes.items():
        latest = items[-1]
        strategies = list(latest.strategies) or ["trade_findings_governance"]
        terms = _lane_search_terms(key, latest.strategies)
        history_matches = _history_matches(
            history,
            terms,
            subject_only=strategies == ["trade_findings_governance"],
        )
        recent_history = [
            {
                "date": commit.date,
                "short_hash": commit.short_hash,
                "subject": commit.subject,
            }
            for commit in history_matches[-2:]
        ]
        family_key = f"{' + '.join(strategies)} :: {latest.primary_loss_driver}"
        family_groups[family_key].append(key)
        for strategy in strategies:
            strategy_groups[strategy].append(key)

        lane_info = {
            "hypothesis_key": key,
            "strategies": strategies,
            "primary_loss_driver": latest.primary_loss_driver,
            "entries": len(items),
            "first_heading": items[0].heading,
            "latest_heading": latest.heading,
            "latest_verdict": latest.verdict or "n/a",
            "latest_status": latest.status or "n/a",
            "next_action": latest.next_action,
            "history_commit_count": len(history_matches),
            "history_match_terms": terms[:8],
            "recent_history_matches": recent_history,
            "family_key": family_key,
        }
        lane_payload[key] = lane_info
        if _is_current_lane(latest):
            current_lanes.append(lane_info)

    family_payload: list[dict[str, object]] = []
    for family_key, keys in family_groups.items():
        latest_key = max(
            keys,
            key=lambda item: (
                lanes[item][-1].heading_dt or dt.datetime.min,
                lanes[item][-1].heading,
            ),
        )
        latest_lane = lane_payload[latest_key]
        family_payload.append(
            {
                "family_key": family_key,
                "hypothesis_keys": sorted(keys),
                "entries": sum(len(lanes[key]) for key in keys),
                "latest_heading": latest_lane["latest_heading"],
                "history_commit_count": max(
                    int(lane_payload[key]["history_commit_count"]) for key in keys
                ),
            }
        )

    strategy_payload: list[dict[str, object]] = []
    for strategy, keys in strategy_groups.items():
        unique_keys = sorted(set(keys))
        strategy_terms, subject_only = _strategy_history_terms(strategy)
        history_matches = _history_matches(
            history, strategy_terms, subject_only=subject_only
        )
        driver_counts: collections.Counter[str] = collections.Counter(
            str(lane_payload[key]["primary_loss_driver"]) for key in unique_keys
        )
        strategy_payload.append(
            {
                "strategy": strategy,
                "findings_entries": sum(len(lanes[key]) for key in unique_keys),
                "distinct_hypothesis_keys": len(unique_keys),
                "open_lanes": sum(
                    1 for key in unique_keys if _is_current_lane(lanes[key][-1])
                ),
                "history_commit_count": len(history_matches),
                "dominant_drivers": [
                    {"primary_loss_driver": driver, "count": count}
                    for driver, count in driver_counts.most_common(3)
                ],
            }
        )

    current_lanes.sort(
        key=lambda item: _heading_dt(str(item["latest_heading"])) or dt.datetime.min,
        reverse=True,
    )
    family_payload.sort(
        key=lambda item: (-int(item["entries"]), str(item["family_key"]))
    )
    strategy_payload.sort(
        key=lambda item: (-int(item["findings_entries"]), str(item["strategy"]))
    )

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(findings_path.relative_to(REPO_ROOT)),
        "history_commit_count": len(history),
        "lane_count": len(lane_payload),
        "lane_family_count": len(family_payload),
        "strategy_count": len(strategy_payload),
        "distinct_hypothesis_keys": len(lane_payload),
        "distinct_lane_families": len(family_payload),
        "current_open_lanes": current_lanes[:limit],
        "lanes": lane_payload,
        "lane_families": family_payload[:limit],
        "strategies": strategy_payload[:limit],
    }
    return payload


def _md(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ").strip()


def _render_markdown(payload: dict[str, object]) -> str:
    current_lanes = payload["current_open_lanes"]
    lane_families = payload["lane_families"]
    strategies = payload["strategies"]

    lines = [
        "# QuantRabbit Repo History Lane Index",
        "",
        f"- Generated by `scripts/{Path(__file__).name}`. 手修正より再生成を優先してください。",
        "- `Lane` は `TRADE_FINDINGS.md` の `Hypothesis Key`、`Lane Family` は `strategy × Primary Loss Driver` として扱います。",
        "- `History Commits` は `git log` の subject / changed paths に対する heuristic match です。厳密な因果証明ではなく、同じ論点をどれだけ触ってきたかを見るための索引です。",
        f"- 履歴 commit 数（history-maintenance docs-only commit 除外後）: `{payload['history_commit_count']}`",
        "",
        "## Summary",
        "",
        f"- distinct hypothesis keys: `{payload['distinct_hypothesis_keys']}`",
        f"- distinct lane families: `{payload['distinct_lane_families']}`",
        f"- current open/pending lanes: `{len(current_lanes)}`",
        "",
        "## Current Open Lanes",
        "",
        "| Hypothesis Key | Strategy | Driver | Entries | Latest Entry | History Commits |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]

    if current_lanes:
        for item in current_lanes:
            lines.append(
                "| "
                + f"`{_md(item['hypothesis_key'])}` | {_md(', '.join(item['strategies']))} | {_md(item['primary_loss_driver'])} | "
                + f"{item['entries']} | {_md(_compact(str(item['latest_heading']), 72))} | {item['history_commit_count']} |"
            )
    else:
        lines.append("| - | - | - | 0 | - | 0 |")

    lines.extend(
        [
            "",
            "## Repeated Lane Families",
            "",
            "| Lane Family | Entries | Hypothesis Keys | Latest Entry | History Commits |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )

    if lane_families:
        for item in lane_families:
            lines.append(
                "| "
                + f"`{_md(item['family_key'])}` | {item['entries']} | {len(item['hypothesis_keys'])} | "
                + f"{_md(_compact(str(item['latest_heading']), 72))} | {item['history_commit_count']} |"
            )
    else:
        lines.append("| - | 0 | 0 | - | 0 |")

    lines.extend(
        [
            "",
            "## Strategy Crosswalk",
            "",
            "| Strategy | Findings Entries | Distinct Keys | Open Lanes | History Commits | Dominant Drivers |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    if strategies:
        for item in strategies:
            dominant = (
                ", ".join(
                    f"{driver['primary_loss_driver']} ({driver['count']})"
                    for driver in item["dominant_drivers"]
                )
                or "-"
            )
            lines.append(
                "| "
                + f"`{_md(item['strategy'])}` | {item['findings_entries']} | {item['distinct_hypothesis_keys']} | "
                + f"{item['open_lanes']} | {item['history_commit_count']} | {_md(dominant)} |"
            )
    else:
        lines.append("| - | 0 | 0 | 0 | 0 | - |")

    lines.extend(["", "## Current Lane Samples", ""])

    if current_lanes:
        for item in current_lanes[:8]:
            lines.append(
                f"- `{item['hypothesis_key']}`: strategy={', '.join(item['strategies']) or 'trade_findings_governance'} / "
                f"driver={item['primary_loss_driver']} / history_commits={item['history_commit_count']}"
            )
            recent = item["recent_history_matches"]
            if recent:
                lines.append(
                    "  recent_history: "
                    + " / ".join(
                        f"{match['date']} {match['short_hash']} {match['subject']}"
                        for match in recent
                    )
                )
            if item["next_action"]:
                lines.append(
                    f"  next_action: {_md(_compact(str(item['next_action']), 160))}"
                )
    else:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    findings_path = Path(args.findings_path).resolve()
    payload = _build_payload(findings_path, max(args.limit, 1))
    doc = _render_markdown(payload)

    if args.write:
        Path(args.out_doc).write_text(doc, encoding="utf-8")
        Path(args.out_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0

    print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
