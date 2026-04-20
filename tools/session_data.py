#!/usr/bin/env python3
"""
Session Data — fetch all data needed at trader session start in a single command

Consolidates Bash steps ②③④ into one. This single script covers:
1. Technical refresh (refresh_factor_cache)
2. OANDA: prices, positions, account
3. Macro view (macro_view)
4. Adaptive Technicals
5. Slack: user messages
6. Memory recall: lessons for held, pending, and scanner candidate pairs
7. Learning-weighted edge board + deployment cues for current seats
8. Today's performance

Usage: python3 tools/session_data.py [--state-ts LAST_SLACK_TS]
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_loader import get_oanda_config

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
CHART_DIR = ROOT / "logs" / "charts"
LESSON_REGISTRY_PATH = ROOT / "collab_trade" / "memory" / "lesson_registry.json"
CHART_TIMEFRAMES = ("M1", "M5", "H1")
CHART_STALE_MIN = 40
QUALITY_AUDIT_STALE_MIN = 70
DIRECT_USD_PAIRS = {"EUR_USD", "GBP_USD", "AUD_USD"}
JPY_CROSS_PAIRS = {"EUR_JPY", "GBP_JPY", "AUD_JPY"}
NO_EDGE_PATTERNS = (
    "no-edge",
    "no proven edge",
    "pair remains no-edge",
    "hard cap remains b-size",
    "b-size max",
    "statistically weak",
    "execution-sensitive",
    "pair expectancy is negative",
)
POSITIVE_EDGE_PATTERNS = (
    "core directional edge",
    "real edge",
    "proven",
    "cleaner structural edge",
    "must be traded",
    "best core edge",
)
CAP_LABELS = {
    "pass": "pass unless exceptional",
    "b_only": "B-only / pass unless exceptional",
    "b_scout": "B scout only",
    "ba_max": "B/A max",
    "a_max": "A max",
    "as_confirmed": "A/S when theme confirmed",
}
SCANNER_MEMORY_TARGET_LIMIT = 6
MAX_MEMORY_TARGETS = 10
SOURCE_PRIORITY = {
    "pending": 2,
    "state": 1,
    "scanner": 0,
}
_TRADE_CONTEXT_STATS = None


def load_config():
    return get_oanda_config()


def _expected_chart_paths() -> list[Path]:
    return [CHART_DIR / f"{pair}_{tf}.png" for pair in PAIRS for tf in CHART_TIMEFRAMES]


def _chart_age_stats(paths: list[Path]) -> tuple[float | None, float | None]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None, None
    ages = [(time.time() - p.stat().st_mtime) / 60 for p in existing]
    return max(ages), min(ages)


def ensure_chart_snapshots_fresh() -> dict:
    """Refresh the full chart set when the trader's local view is missing or stale."""
    paths = _expected_chart_paths()
    missing = [p.name for p in paths if not p.exists()]
    oldest_age_min, newest_age_min = _chart_age_stats(paths)
    refresh_reason = None

    if missing:
        refresh_reason = f"missing {len(missing)} chart(s)"
    elif oldest_age_min is None:
        refresh_reason = "chart set unavailable"
    elif oldest_age_min > CHART_STALE_MIN:
        refresh_reason = f"oldest chart {oldest_age_min:.0f}min old"

    refreshed = False
    refresh_output = ""
    if refresh_reason:
        refresh_output = run_script(
            [VENV_PYTHON, "tools/chart_snapshot.py", "--all", "--with-m1"],
            timeout=90,
        )
        refreshed = True
        oldest_age_min, newest_age_min = _chart_age_stats(paths)

    return {
        "refreshed": refreshed,
        "refresh_reason": refresh_reason,
        "missing": missing,
        "oldest_age_min": oldest_age_min,
        "newest_age_min": newest_age_min,
        "refresh_output": refresh_output,
    }


def oanda_api(path, cfg):
    url = f"{cfg['oanda_base_url']}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def run_script(args, timeout=30):
    """Run a script in a subprocess. Does not abort on failure."""
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT)
        )
        return r.stdout.strip()
    except Exception as e:
        return f"(skip: {e})"


def section(title):
    print(f"\n=== {title} ===")


PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"),
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}

CORRELATED_PAIRS = [
    ("EUR_USD", "GBP_USD", "anti-USD"),
    ("EUR_JPY", "GBP_JPY", "anti-JPY"),
    ("AUD_USD", "AUD_JPY", "pro-AUD"),
]

_MEMORY_HYBRID_SEARCH = None


def _load_memory_search():
    global _MEMORY_HYBRID_SEARCH
    if _MEMORY_HYBRID_SEARCH is None:
        memory_dir = ROOT / "collab_trade" / "memory"
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from recall import hybrid_search  # type: ignore
        _MEMORY_HYBRID_SEARCH = hybrid_search
    return _MEMORY_HYBRID_SEARCH


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    match = re.search(pattern, text, re.M)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_hot_updates(state_text: str) -> list[str]:
    section_text = _extract_markdown_section(state_text, "Hot Updates")
    if not section_text:
        return []
    updates = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            updates.append(line[2:].strip())
    return updates[:5]


def _units_to_direction(units) -> str | None:
    try:
        return "LONG" if float(units) > 0 else "SHORT"
    except Exception:
        return None


def _parse_state_positions(state_text: str) -> list[dict]:
    positions = []
    seen = set()
    lines = state_text.splitlines()

    in_positions = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("## "):
            in_positions = line.lower().startswith("## positions")
            continue
        if not in_positions or not line:
            continue
        match = re.match(r"^([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\s+[\d,]+u\b", line)
        if not match or "LIMIT" in line.upper():
            continue
        key = (match.group(1), match.group(2))
        if key in seen:
            continue
        seen.add(key)
        positions.append({
            "pair": match.group(1),
            "direction": match.group(2),
            "line": line,
        })

    if positions:
        return positions

    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("###") or "LIMIT" in line.upper():
            continue
        match = re.match(r"^###\s+(?:Position:\s+)?([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b", line)
        if not match:
            continue
        key = (match.group(1), match.group(2))
        if key in seen:
            continue
        seen.add(key)
        positions.append({
            "pair": match.group(1),
            "direction": match.group(2),
            "line": line,
        })

    return positions


def _build_state_position_sync(state_text: str, trades_data: dict) -> dict:
    state_positions = _parse_state_positions(state_text)
    state_by_key = {
        (p["pair"], p["direction"]): p
        for p in state_positions
    }

    live_by_key = {}
    for trade in trades_data.get("trades", []):
        pair = trade.get("instrument")
        direction = _units_to_direction(trade.get("currentUnits"))
        if not pair or not direction:
            continue
        key = (pair, direction)
        entry = live_by_key.setdefault(key, {
            "pair": pair,
            "direction": direction,
            "units": 0,
            "trade_ids": [],
        })
        entry["units"] += abs(int(float(trade.get("currentUnits", 0))))
        entry["trade_ids"].append(str(trade.get("id", "")))

    live_keys = set(live_by_key)
    state_keys = set(state_by_key)
    return {
        "match": live_keys == state_keys,
        "live_not_in_state": [live_by_key[key] for key in sorted(live_keys - state_keys)],
        "state_not_live": [state_by_key[key] for key in sorted(state_keys - live_keys)],
    }


def _parse_scanner_candidates(scanner_output: str) -> list[dict]:
    candidates = []
    for line in scanner_output.splitlines():
        match = re.match(r"🎯\s+(\w+_\w+)\s+(LONG|SHORT)\s+([^\[]+)", line.strip())
        if not match:
            continue
        candidates.append({
            "pair": match.group(1),
            "direction": match.group(2),
            "recipe": match.group(3).strip(),
        })
    return candidates


def _extract_pair_direction(text: str) -> tuple[str | None, str | None]:
    pair_match = re.search(r"\b(" + "|".join(PAIRS) + r")\b", text)
    if not pair_match:
        return None, None

    pair = pair_match.group(1)
    direction_match = re.search(
        rf"\b{re.escape(pair)}\b\s+(LONG|SHORT)\b",
        text,
        flags=re.IGNORECASE,
    )
    if direction_match:
        return pair, direction_match.group(1).upper()

    tail = text[pair_match.end():]
    direction_match = re.search(r"\b(LONG|SHORT)\b", tail, flags=re.IGNORECASE)
    direction = direction_match.group(1).upper() if direction_match else None
    if direction is None:
        lowered = tail.lower()
        if re.search(r"\bbuy\b", lowered):
            direction = "LONG"
        elif re.search(r"\bsell\b", lowered):
            direction = "SHORT"
    return pair, direction


def _parse_state_carry_targets(state_text: str) -> list[dict]:
    if not state_text:
        return []

    targets: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_target(line: str, origin: str) -> None:
        pair, direction = _extract_pair_direction(line)
        if not pair or not direction:
            return
        key = (pair, direction)
        if key in seen:
            return
        targets.append({
            "pair": pair,
            "direction": direction,
            "source": "state",
            "recipe": _clip_text(f"{origin}: {line}", 140),
        })
        seen.add(key)

    prefixes = {
        "Backup vehicle:": "backup vehicle",
        "Second-best expression:": "second-best expression",
        "Next fresh risk allowed NOW:": "next fresh risk",
        "Best direct-USD seat NOW:": "best direct-USD seat",
        "Best rotation candidate:": "rotation candidate",
        "Best expression NOW:": "best expression",
    }

    for raw_line in state_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for prefix, origin in prefixes.items():
            if not line.startswith(prefix):
                continue
            if prefix == "Next fresh risk allowed NOW:" and re.search(r":\s*none\b", line, flags=re.IGNORECASE):
                continue
            add_target(line, origin)
            break

        if re.match(r"^Podium #\d+:", line):
            add_target(line, "excavation podium")
        elif re.match(r"^(Short-term|Medium-term|Long-term) S", line):
            add_target(line, "s-hunt horizon")
        elif re.match(r"^Lane [23] /", line):
            add_target(line, "multi-vehicle lane")

    return targets


def _build_memory_targets(
    trades_data: dict,
    pending_orders: list[dict],
    scanner_output: str,
    carry_targets: list[dict] | None = None,
) -> list[dict]:
    targets = []
    seen = set()

    def add_target(pair: str | None, direction: str | None, source: str, recipe: str | None = None):
        if not pair or not direction:
            return
        key = (pair, direction)
        if key in seen:
            return
        targets.append({
            "pair": pair,
            "direction": direction,
            "source": source,
            "recipe": recipe or "",
        })
        seen.add(key)

    for trade in trades_data.get("trades", []):
        add_target(trade.get("instrument"), _units_to_direction(trade.get("currentUnits")), "held")

    for order in pending_orders:
        add_target(order.get("instrument"), _units_to_direction(order.get("units")), "pending")

    for candidate in carry_targets or []:
        add_target(candidate.get("pair"), candidate.get("direction"), "state", candidate.get("recipe"))

    for candidate in _parse_scanner_candidates(scanner_output)[:SCANNER_MEMORY_TARGET_LIMIT]:
        add_target(candidate.get("pair"), candidate.get("direction"), "scanner", candidate.get("recipe"))

    return targets[:MAX_MEMORY_TARGETS]


def _memory_query_for_target(target: dict) -> str:
    parts = [target["pair"], target["direction"]]
    if target["source"] == "held":
        parts.extend(["hold", "exit", "lesson", "failure", "success"])
    elif target["source"] == "pending":
        parts.extend(["entry", "limit", "lesson", "failure", "success"])
    elif target["source"] == "state":
        parts.extend(["trigger", "backup", "rotation", "retest", "lesson"])
    else:
        parts.extend(["setup", "lesson", "failure", "success"])
    if target.get("recipe"):
        parts.append(target["recipe"])
    return " ".join(parts)


def _load_lesson_registry() -> dict:
    if not LESSON_REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(LESSON_REGISTRY_PATH.read_text())
    except Exception:
        return {}


def _registry_rank(lesson: dict) -> tuple[int, int, str, str]:
    return (
        int(lesson.get("trust_score", 0)),
        int(lesson.get("state_rank", 0)),
        str(lesson.get("lesson_date", "")),
        str(lesson.get("id", "")),
    )


def _target_lessons_for_profile(target: dict, registry: dict) -> tuple[list[dict], list[dict]]:
    lessons = registry.get("lessons") or []
    pair = target.get("pair")
    direction = target.get("direction")
    if not pair or not direction or not lessons:
        return [], []

    exact = sorted(
        [
            lesson for lesson in lessons
            if lesson.get("pair") == pair
            and lesson.get("direction") == direction
            and lesson.get("state") != "deprecated"
        ],
        key=_registry_rank,
        reverse=True,
    )
    pair_only = sorted(
        [
            lesson for lesson in lessons
            if lesson.get("pair") == pair
            and not lesson.get("direction")
            and lesson.get("state") != "deprecated"
        ],
        key=_registry_rank,
        reverse=True,
    )
    return exact, pair_only


def _lesson_text_blob(lessons: list[dict]) -> str:
    return " ".join(
        (
            f"{lesson.get('title', '')} {lesson.get('text', '')}".strip().lower()
            for lesson in lessons
        )
    )


def _clip_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _ledger_safe_text(text: str, limit: int = 120) -> str:
    return _clip_text(str(text or "").replace("|", "/"), limit)


def _load_audit_narrative_context() -> dict:
    audit_path = ROOT / "logs" / "quality_audit.md"
    if not audit_path.exists():
        return {"available": False, "stale": False, "age_min": None, "pairs": {}, "strongest": None}

    age_min = (time.time() - audit_path.stat().st_mtime) / 60
    stale = age_min > QUALITY_AUDIT_STALE_MIN
    if stale:
        return {"available": True, "stale": True, "age_min": age_min, "pairs": {}, "strongest": None}

    try:
        from record_audit_narrative import build_entry as build_audit_narrative_entry
        entry = build_audit_narrative_entry(audit_path.read_text())
    except Exception:
        return {"available": True, "stale": False, "age_min": age_min, "pairs": {}, "strongest": None}

    pair_map: dict[tuple[str, str], list[str]] = {}
    strongest = None
    strongest_raw = entry.get("strongest_unheld") or {}
    if strongest_raw.get("pair") and strongest_raw.get("direction"):
        strongest = (strongest_raw["pair"], strongest_raw["direction"])
        pair_map.setdefault(strongest, []).append("audit strongest-unheld")

    for pick in entry.get("narrative_picks") or []:
        pair = pick.get("pair")
        direction = pick.get("direction")
        edge = str(pick.get("edge") or "")
        if not pair or not direction or edge not in {"S", "A"}:
            continue
        label = f"audit narrative Edge {edge}"
        key = (pair, direction)
        if label not in pair_map.setdefault(key, []):
            pair_map[key].append(label)

    return {
        "available": True,
        "stale": False,
        "age_min": age_min,
        "pairs": pair_map,
        "strongest": strongest,
    }


def _tokyo_open_start_utc(now_utc: datetime) -> datetime:
    anchor = datetime(now_utc.year, now_utc.month, now_utc.day, 21, tzinfo=timezone.utc)
    if now_utc < anchor:
        anchor -= timedelta(days=1)
    return anchor


def _fetch_window_move(pair: str, start_utc: datetime, cfg: dict) -> dict | None:
    params = urllib.parse.urlencode(
        {
            "price": "M",
            "granularity": "M5",
            "from": start_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
    )
    url = f"{cfg['oanda_base_url']}/v3/instruments/{pair}/candles?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )
    data = json.loads(urllib.request.urlopen(req, timeout=10).read())
    candles = data.get("candles") or []
    values = []
    for candle in candles:
        mid = candle.get("mid") or {}
        try:
            values.append(
                (
                    candle.get("time"),
                    float(mid["o"]),
                    float(mid["h"]),
                    float(mid["l"]),
                    float(mid["c"]),
                )
            )
        except Exception:
            continue

    if not values:
        return None

    pip_factor = 100 if "JPY" in pair else 10000
    first = values[0]
    last = values[-1]
    high = max(values, key=lambda item: item[2])
    low = min(values, key=lambda item: item[3])
    open_px = first[1]
    return {
        "pair": pair,
        "open_time": first[0],
        "last_time": last[0],
        "net_pips": (last[4] - open_px) * pip_factor,
        "range_pips": (high[2] - low[3]) * pip_factor,
        "max_up_pips": (high[2] - open_px) * pip_factor,
        "max_down_pips": (open_px - low[3]) * pip_factor,
        "high_time": high[0],
        "low_time": low[0],
    }


def _print_tokyo_open_breadth(cfg: dict, now_utc: datetime, spread_data: dict[str, float]) -> None:
    if now_utc.hour >= 8:
        return

    start_utc = _tokyo_open_start_utc(now_utc)
    section("TOKYO-OPEN BREADTH")
    print(
        "Window: "
        f"{start_utc.strftime('%Y-%m-%d %H:%M UTC')} -> {now_utc.strftime('%Y-%m-%d %H:%M UTC')} "
        f"({(start_utc + timedelta(hours=9)).strftime('%m/%d %H:%M')} JST -> now)"
    )

    rows = []
    for pair in PAIRS:
        try:
            row = _fetch_window_move(pair, start_utc, cfg)
        except Exception:
            row = None
        if row:
            rows.append(row)

    if not rows:
        print("(tokyo-open breadth unavailable)")
        return

    positive = sum(1 for row in rows if row["net_pips"] > 0.5)
    negative = sum(1 for row in rows if row["net_pips"] < -0.5)
    active = [
        row for row in rows
        if abs(float(row["net_pips"])) >= 15.0 or float(row["range_pips"]) >= 30.0
    ]
    leaders = sorted(rows, key=lambda item: abs(float(item["net_pips"])), reverse=True)[:3]
    leader_text = ", ".join(f"{row['pair']} {float(row['net_pips']):+.1f}pip" for row in leaders)
    print(
        f"Breadth: up {positive}/{len(rows)} | down {negative}/{len(rows)} "
        f"| active movers {len(active)}/{len(rows)} | leaders: {leader_text}"
    )
    for row in rows:
        spread_note = ""
        if row["pair"] in spread_data:
            spread_note = f" | spread {float(spread_data[row['pair']]):.1f}pip"
        print(
            f"{row['pair']}: net {float(row['net_pips']):+.1f}pip | range {float(row['range_pips']):.1f}pip "
            f"| max up {float(row['max_up_pips']):+.1f} | max down {float(row['max_down_pips']):+.1f}"
            f"{spread_note}"
        )
    if len(active) >= 3:
        print(
            "Deployment implication: Tokyo breadth is real. If you still hold only one live receipt, "
            "carry at least one BACKUP trigger/limit across the next 20-minute cadence or state exactly "
            "why each second lane lost orderability."
        )


def _pair_bucket(pair: str | None) -> str:
    if pair in DIRECT_USD_PAIRS:
        return "direct_usd"
    if pair in JPY_CROSS_PAIRS:
        return "jpy_cross"
    if pair == "USD_JPY":
        return "usd_jpy"
    return "other"


def _execution_style_rank(style: str | None) -> int:
    mapping = {
        "MARKET": 3,
        "STOP-ENTRY": 2,
        "LIMIT": 1,
        "PASS": 0,
    }
    return mapping.get(str(style or "").upper(), 0)


def _select_multi_vehicle_lanes(fresh_profiles: list[dict], max_lanes: int = 3) -> list[dict]:
    candidates = [
        profile for profile in fresh_profiles
        if profile.get("execution_style") != "PASS"
    ]
    candidates.sort(
        key=lambda item: (
            _execution_style_rank(item.get("execution_style")),
            int(item.get("learning_score", 0)),
            _cap_rank(str(item.get("allocation_cap", ""))),
            int(item.get("trade_count", 0)),
            float(item.get("trade_ev", 0.0)),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    seen_pairs: set[str] = set()
    seen_bases: set[str] = set()
    seen_buckets: set[str] = set()

    for profile in candidates:
        pair = str(profile.get("pair", ""))
        if not pair or pair in seen_pairs:
            continue
        base, _quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        bucket = str(profile.get("bucket", "other"))
        if base in seen_bases and bucket in seen_buckets:
            continue
        selected.append(profile)
        seen_pairs.add(pair)
        seen_bases.add(base)
        seen_buckets.add(bucket)
        if len(selected) >= max_lanes:
            return selected

    for profile in candidates:
        if len(selected) >= max_lanes:
            break
        pair = str(profile.get("pair", ""))
        if not pair or pair in seen_pairs:
            continue
        selected.append(profile)
        seen_pairs.add(pair)

    return selected


def _multi_vehicle_role(index: int) -> str:
    if index == 1:
        return "PRIMARY"
    if index == 2:
        return "BACKUP"
    if index == 3:
        return "THIRD CURRENCY"
    return f"LANE {index}"


def _multi_vehicle_reason(profile: dict, prior: list[dict]) -> str:
    pair = str(profile.get("pair", ""))
    base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
    if not prior:
        return f"first live lane: {base}/{quote} is currently the cleanest expression"

    parts = []
    if all(PAIR_CURRENCIES.get(str(item.get("pair", "")), ("?", "?"))[0] != base for item in prior):
        parts.append(f"new base currency {base}")
    if all(str(item.get("bucket", "other")) != str(profile.get("bucket", "other")) for item in prior):
        parts.append(f"new vehicle bucket {profile.get('bucket')}")
    if not parts:
        other_pairs = ", ".join(str(item.get("pair", "")) for item in prior[:2])
        parts.append(f"separate expression from {other_pairs}")
    return " + ".join(parts)


def _build_s_excavation_podium_seeds(
    fresh_profiles: list[dict],
    best_direct: dict | None,
    best_cross: dict | None,
    best_usdjpy: dict | None,
    audit_context: dict,
) -> list[dict]:
    seeds: dict[tuple[str, str], dict] = {}

    def ensure_seed(pair: str, direction: str) -> dict:
        return seeds.setdefault(
            (pair, direction),
            {
                "pair": pair,
                "direction": direction,
                "priority": 0,
                "why_bits": [],
                "blocker": None,
                "upgrade_action": None,
            },
        )

    for idx, profile in enumerate(fresh_profiles[:6], start=1):
        pair = str(profile.get("pair", ""))
        direction = str(profile.get("direction", ""))
        if not pair or not direction:
            continue
        style = str(profile.get("execution_style") or "PASS").upper()
        if style == "PASS":
            continue

        seed = ensure_seed(pair, direction)
        priority = int(profile.get("learning_score", 0))
        if style == "STOP-ENTRY":
            priority += 35
        elif style == "LIMIT":
            priority += 28
        elif style == "MARKET":
            priority += 10
        if idx == 1:
            priority += 12
        elif idx == 2:
            priority += 8
        elif idx == 3:
            priority += 4
        seed["priority"] = max(seed["priority"], priority)

        why_bits = seed["why_bits"]
        tag = f"tournament #{idx}"
        if tag not in why_bits:
            why_bits.append(tag)
        if best_direct and pair == best_direct.get("pair") and direction == best_direct.get("direction"):
            why_bits.append("best direct-USD")
        if best_cross and pair == best_cross.get("pair") and direction == best_cross.get("direction"):
            why_bits.append("best cross")
        if best_usdjpy and pair == best_usdjpy.get("pair") and direction == best_usdjpy.get("direction"):
            why_bits.append("best USD_JPY")

        for label in audit_context.get("pairs", {}).get((pair, direction), []):
            if label not in why_bits:
                why_bits.append(label)
            seed["priority"] += 18 if "strongest-unheld" in label else 10

        summary = _ledger_safe_text(
            f"learning {int(profile.get('learning_score', 0))}/100 {profile.get('verdict')}",
            120,
        )
        if summary not in why_bits:
            why_bits.append(summary)

        blocker = profile.get("execution_note")
        if blocker:
            seed["blocker"] = _ledger_safe_text(blocker, 120)
        elif style == "MARKET":
            seed["blocker"] = "no memory blocker remains; only a live chart contradiction should keep this out of S Hunt"

        if style in {"STOP-ENTRY", "LIMIT", "MARKET"}:
            seed["upgrade_action"] = style

    ordered = sorted(
        seeds.values(),
        key=lambda item: (
            int(item.get("priority", 0)),
            1 if str(item.get("upgrade_action", "")) == "STOP-ENTRY" else 0,
            1 if str(item.get("upgrade_action", "")) == "LIMIT" else 0,
        ),
        reverse=True,
    )

    result = []
    seen_pairs: set[str] = set()
    for seed in ordered:
        if seed["pair"] in seen_pairs:
            continue
        if not seed.get("blocker") or not seed.get("upgrade_action"):
            continue
        result.append(
            {
                "pair": seed["pair"],
                "direction": seed["direction"],
                "closest_to_s_because": _ledger_safe_text(" + ".join(seed["why_bits"]), 150),
                "still_blocked_by": seed["blocker"],
                "upgrade_action": seed["upgrade_action"],
            }
        )
        seen_pairs.add(seed["pair"])
        if len(result) >= 3:
            break

    return result


def _current_session_bucket(now_utc) -> str:
    h = now_utc.hour
    if 0 <= h < 7:
        return "tokyo"
    if 7 <= h < 15:
        return "london"
    if 15 <= h < 22:
        return "newyork"
    return "late"


def _infer_current_pair_regime(pair: str) -> str | None:
    tech = _load_technicals(ROOT, pair)
    m5 = tech.get("M5", {})
    if not m5:
        return None
    adx = float(m5.get("adx", 0) or 0)
    plus_di = float(m5.get("plus_di", 0) or 0)
    minus_di = float(m5.get("minus_di", 0) or 0)
    bbw = float(m5.get("bbw", 0) or 0)
    kc_width = float(m5.get("kc_width", 0) or 0)
    di_gap = abs(plus_di - minus_di)

    if kc_width > 0 and bbw > 0 and bbw < kc_width * 0.9:
        return "squeeze"
    if adx >= 25 and di_gap >= 8:
        return "trending"
    if adx < 18:
        return "range"
    if bbw < 0.0015:
        return "quiet"
    return "transition"


def _cap_rank(label: str) -> int:
    mapping = {
        CAP_LABELS["pass"]: 0,
        CAP_LABELS["b_only"]: 1,
        CAP_LABELS["b_scout"]: 2,
        CAP_LABELS["ba_max"]: 3,
        CAP_LABELS["a_max"]: 4,
        CAP_LABELS["as_confirmed"]: 5,
    }
    return mapping.get(label, 2)


def _cap_label_from_rank(rank: int) -> str:
    reverse = {
        0: CAP_LABELS["pass"],
        1: CAP_LABELS["b_only"],
        2: CAP_LABELS["b_scout"],
        3: CAP_LABELS["ba_max"],
        4: CAP_LABELS["a_max"],
        5: CAP_LABELS["as_confirmed"],
    }
    return reverse[max(0, min(5, rank))]


def _format_context_stat(stats: dict | None, label: str) -> str:
    if not stats or int(stats.get("count", 0) or 0) <= 0:
        return f"{label}: no sample"
    return (
        f"{label}: WR {float(stats.get('win_rate', 0.0))*100:.0f}% "
        f"EV {float(stats.get('ev', 0.0)):+.0f} n={int(stats.get('count', 0))}"
    )


def _context_signal(stats: dict | None, *, min_count: int = 3) -> tuple[int, int, str]:
    if not stats:
        return 0, 0, "no sample"

    count = int(stats.get("count", 0) or 0)
    if count < min_count:
        return 0, 0, "thin"

    ev = float(stats.get("ev", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)

    if ev > 0 and win_rate >= 0.55:
        score = 4 + (2 if count >= 5 else 0) + (2 if win_rate >= 0.60 else 0)
        cap = 1 if count >= 5 else 0
        return score, cap, "tailwind"

    if ev < 0 or win_rate <= 0.40:
        score = -6 - (3 if count >= 5 else 0) - (2 if ev < 0 and win_rate <= 0.35 else 0)
        cap = -2 if count >= 5 and ev < 0 and win_rate <= 0.40 else -1
        return score, cap, "headwind"

    return 0, 0, "mixed"


def _load_trade_context_stats() -> dict:
    global _TRADE_CONTEXT_STATS
    if _TRADE_CONTEXT_STATS is not None:
        return _TRADE_CONTEXT_STATS

    memory_dir = ROOT / "collab_trade" / "memory"
    db_path = memory_dir / "memory.db"
    if str(memory_dir) not in sys.path:
        sys.path.insert(0, str(memory_dir))
    try:
        from schema import get_conn  # type: ignore

        conn = get_conn()
    except ModuleNotFoundError as exc:
        if exc.name != "apsw":
            raise
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("PRAGMA query_only=ON")

    def pack(rows, key_builder):
        out = {}
        for row in rows:
            *head, cnt, ev, total_pl, wins = row
            cnt = int(cnt or 0)
            out[key_builder(head)] = {
                "count": cnt,
                "ev": float(ev or 0.0),
                "total_pl": float(total_pl or 0.0),
                "wins": int(wins or 0),
                "win_rate": (float(wins or 0) / float(cnt)) if cnt else 0.0,
            }
        return out

    try:
        pair_rows = conn.execute(
            """SELECT pair, direction,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL
               GROUP BY pair, direction"""
        ).fetchall()
        session_rows = conn.execute(
            """SELECT pair, direction,
                      CASE
                        WHEN session_hour BETWEEN 0 AND 6 THEN 'tokyo'
                        WHEN session_hour BETWEEN 7 AND 14 THEN 'london'
                        WHEN session_hour BETWEEN 15 AND 21 THEN 'newyork'
                        ELSE 'late'
                      END AS session_bucket,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL AND session_hour IS NOT NULL
               GROUP BY pair, direction, session_bucket"""
        ).fetchall()
        regime_rows = conn.execute(
            """SELECT pair, direction, regime,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL AND regime IS NOT NULL AND TRIM(regime) <> ''
               GROUP BY pair, direction, regime"""
        ).fetchall()
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()

    _TRADE_CONTEXT_STATS = {
        "pair": pack(pair_rows, lambda head: tuple(head)),
        "session": pack(session_rows, lambda head: tuple(head)),
        "regime": pack(regime_rows, lambda head: tuple(head)),
    }
    return _TRADE_CONTEXT_STATS


def _build_learning_profile(target: dict, registry: dict) -> dict:
    from datetime import datetime, timezone

    exact, pair_only = _target_lessons_for_profile(target, registry)
    relevant = exact[:3] + pair_only[:2]
    top = exact[0] if exact else (pair_only[0] if pair_only else None)
    now_utc = datetime.now(timezone.utc)
    current_session = _current_session_bucket(now_utc)
    current_regime = _infer_current_pair_regime(target.get("pair", ""))
    context_stats = _load_trade_context_stats()
    pair_stat = context_stats.get("pair", {}).get((target.get("pair"), target.get("direction")))
    session_stat = context_stats.get("session", {}).get(
        (target.get("pair"), target.get("direction"), current_session)
    )
    regime_stat = None
    if current_regime:
        regime_stat = context_stats.get("regime", {}).get(
            (target.get("pair"), target.get("direction"), current_regime)
        )

    display_stats = pair_stat or {}

    profile = {
        "pair": target.get("pair"),
        "direction": target.get("direction"),
        "source": target.get("source"),
        "recipe": target.get("recipe", ""),
        "bucket": _pair_bucket(target.get("pair")),
        "learning_score": 18,
        "verdict": "limited history",
        "allocation_cap": "B scout only",
        "evidence": "No strong pair-specific lesson yet. Respect the tape more than memory.",
        "state": None,
        "trust_score": 0,
        "trade_count": 0,
        "trade_ev": 0.0,
        "trade_wr": 0.0,
        "has_exact": False,
        "current_session": current_session,
        "current_regime": current_regime,
        "session_stat": session_stat,
        "regime_stat": regime_stat,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "context_bias": "neutral",
    }
    if not top:
        return profile

    stats = {}
    for lesson in exact + pair_only:
        trade_stats = lesson.get("trade_stats") or {}
        if trade_stats.get("count"):
            stats = trade_stats
            break

    if not display_stats and stats:
        display_stats = stats

    trade_count = int(display_stats.get("count", 0) or 0)
    trade_ev = float(display_stats.get("ev", 0.0) or 0.0)
    trade_wr = float(display_stats.get("win_rate", 0.0) or 0.0)
    trust = int(top.get("trust_score", 0) or 0)
    text_blob = _lesson_text_blob(relevant)
    no_edge = any(pattern in text_blob for pattern in NO_EDGE_PATTERNS)
    positive = any(pattern in text_blob for pattern in POSITIVE_EDGE_PATTERNS)
    session_score_delta, session_cap_delta, session_bias = _context_signal(session_stat)
    regime_score_delta, regime_cap_delta, regime_bias = _context_signal(regime_stat)

    score = trust
    if exact:
        score += 12
    elif pair_only:
        score += 5
    if top.get("state") == "confirmed":
        score += 8
    elif top.get("state") == "watch":
        score += 3
    if pair_stat and trade_count >= 5 and trade_ev > 0:
        score += 8
    elif trade_count >= 5 and trade_ev > 0:
        score += 8
    if trade_count >= 10 and trade_wr >= 0.55:
        score += 5
    if trade_count >= 5 and trade_ev < 0:
        score -= 14
    if no_edge:
        score -= 28
    elif trade_count >= 5 and trade_wr < 0.40:
        score -= 8
    if pair_only and not exact:
        score -= 12
        score = min(score, 74)
    score += session_score_delta + regime_score_delta
    score = max(0, min(99, score))

    if no_edge:
        verdict = "no-edge / restricted"
        allocation_cap = CAP_LABELS["b_only"]
    elif exact and top.get("state") == "confirmed" and trade_count >= 5 and trade_ev > 0 and trade_wr >= 0.50:
        verdict = "confirmed edge"
        allocation_cap = CAP_LABELS["as_confirmed"]
    elif exact and top.get("state") == "confirmed":
        verdict = "confirmed but session-dependent"
        allocation_cap = CAP_LABELS["a_max"]
    elif exact and top.get("state") == "watch" and trade_count >= 3 and trade_ev > 0:
        verdict = "watch edge"
        allocation_cap = CAP_LABELS["a_max"]
    elif exact and top.get("state") == "watch":
        verdict = "watch / unproven"
        allocation_cap = CAP_LABELS["b_only"]
    elif pair_only and trade_count >= 5 and trade_ev > 0:
        verdict = "pair memory positive, direction unproven"
        allocation_cap = CAP_LABELS["ba_max"]
    elif pair_only:
        verdict = "pair memory only"
        allocation_cap = CAP_LABELS["b_scout"]
    elif positive:
        verdict = "positive lesson, thin sample"
        allocation_cap = CAP_LABELS["ba_max"]
    else:
        verdict = "limited history"
        allocation_cap = CAP_LABELS["b_scout"]

    cap_rank = _cap_rank(allocation_cap)
    cap_rank += session_cap_delta + regime_cap_delta
    if no_edge:
        cap_rank = min(cap_rank, 1)
    if not exact:
        cap_rank = min(cap_rank, 3)
    if trade_count < 3:
        cap_rank = min(cap_rank, 2)
    allocation_cap = _cap_label_from_rank(cap_rank)

    context_flags = []
    if session_bias == "headwind":
        context_flags.append(f"{current_session} headwind")
    elif session_bias == "tailwind":
        context_flags.append(f"{current_session} tailwind")
    if current_regime:
        if regime_bias == "headwind":
            context_flags.append(f"{current_regime} headwind")
        elif regime_bias == "tailwind":
            context_flags.append(f"{current_regime} tailwind")
    if context_flags:
        verdict = f"{verdict} | {' + '.join(context_flags)}"

    evidence = _clip_text(top.get("title") or top.get("text") or profile["evidence"])
    context_bias = "neutral"
    if "headwind" in context_flags:
        context_bias = "headwind"
    elif "tailwind" in context_flags:
        context_bias = "tailwind"
    profile.update({
        "learning_score": score,
        "verdict": verdict,
        "allocation_cap": allocation_cap,
        "evidence": evidence,
        "state": top.get("state"),
        "trust_score": trust,
        "trade_count": trade_count,
        "trade_ev": trade_ev,
        "trade_wr": trade_wr,
        "has_exact": bool(exact),
        "session_stat": session_stat,
        "regime_stat": regime_stat,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "context_bias": context_bias,
    })
    return profile


def _build_learning_edge_profiles(targets: list[dict], registry: dict) -> list[dict]:
    profiles = [_build_learning_profile(target, registry) for target in targets]
    profiles.sort(
        key=lambda item: (
            item.get("learning_score", 0),
            item.get("has_exact", False),
            item.get("trade_count", 0),
            item.get("trade_ev", 0.0),
            item.get("pair", ""),
        ),
        reverse=True,
    )
    return profiles


def _bayesian_update_hint(profile: dict) -> str:
    verdict = str(profile.get("verdict", ""))
    if "no-edge" in verdict:
        return "One clean win is not enough to expand size. Require repeated clean evidence before promoting above B."
    if "confirmed edge" in verdict:
        return "One loss does not kill the prior. Only demote if the tape breaks the pattern or losses start repeating."
    if "watch" in verdict or "pair memory positive" in verdict:
        return "A clean trigger win upgrades live confidence today; a failed trigger keeps this in watch/B lane."
    return "Treat the next outcome as candidate evidence only. Do not rewrite the pair story from one print."


def _execution_target_pips(profile: dict) -> float:
    regime = profile.get("current_regime")
    if regime == "trending":
        return 20.0
    if regime == "range":
        return 12.0
    if regime == "quiet":
        return 10.0
    if regime == "squeeze":
        return 10.0
    if regime == "transition":
        return 12.0
    return 12.0


def _market_spread_bands(pair: str, spread_pips: float | None, spread_ratio: float | None) -> tuple[bool, bool]:
    if spread_pips is None and spread_ratio is None:
        return True, True

    is_jpy = str(pair).endswith("JPY")
    tight_abs = 2.5 if is_jpy else 1.5
    normal_abs = 3.0 if is_jpy else 2.0
    tight_ok = (
        (spread_ratio is None or spread_ratio <= 0.18)
        or (spread_pips is not None and spread_pips <= tight_abs)
    )
    normal_ok = (
        (spread_ratio is None or spread_ratio <= 0.24)
        or (spread_pips is not None and spread_pips <= normal_abs)
    )
    return tight_ok, normal_ok


def _headwind_participation_plan(
    regime: str | None,
    cap_rank: int,
    learning_score: int,
    trade_count: int,
    has_exact: bool,
    confirmed_state: bool,
    tight_market_spread: bool,
    normal_market_spread: bool,
) -> tuple[str | None, str | None]:
    if regime not in {"trending", "transition", "squeeze"}:
        return None, None

    support = trade_count >= 3 or has_exact or confirmed_state
    if tight_market_spread and support:
        if cap_rank >= 2 and learning_score >= 50:
            return (
                "MARKET",
                "historical headwind caps size, not all participation; take a small live scout now and leave one reload LIMIT",
            )
        if cap_rank == 1 and learning_score >= 58:
            return (
                "MARKET",
                "even with B-only memory, the live tape is paying enough at tight spread to justify a scout instead of full flatness",
            )

    if normal_market_spread and learning_score >= 45 and (support or cap_rank >= 2):
        return (
            "STOP-ENTRY",
            "historical headwind blocks blind chase, but the tape is alive enough to arm a trigger instead of passing",
        )

    return None, None


def _recommend_profile_execution(profile: dict, spread_pips: float | None) -> dict:
    regime = profile.get("current_regime")
    context_bias = profile.get("context_bias")
    cap = profile.get("allocation_cap")
    source = profile.get("source")
    learning_score = int(profile.get("learning_score", 0) or 0)
    trade_count = int(profile.get("trade_count", 0) or 0)
    target_pips = _execution_target_pips(profile)
    spread_ratio = (spread_pips / target_pips) if spread_pips is not None and target_pips > 0 else None
    cap_rank = _cap_rank(cap)
    has_exact = bool(profile.get("has_exact"))
    confirmed_state = profile.get("state") == "confirmed"
    tight_market_spread, normal_market_spread = _market_spread_bands(
        str(profile.get("pair", "")),
        spread_pips,
        spread_ratio,
    )
    reasons: list[str] = []

    if source == "pending":
        reasons.append("already armed; leave it alone unless the thesis is dead")
        return {
            "style": "LIMIT",
            "orderability": "LIMIT",
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if cap == CAP_LABELS["pass"]:
        reasons.append("learning cap already says pass unless exceptional")
        return {
            "style": "PASS",
            "orderability": "PASS",
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if spread_ratio is not None and spread_ratio > 0.30 and not normal_market_spread:
        if regime in {"trending", "transition", "squeeze"}:
            reasons.append(
                f"spread {spread_pips:.1f}pip is too expensive for immediate execution while {regime} still needs proof"
            )
            style = "STOP-ENTRY"
        else:
            reasons.append(f"spread {spread_pips:.1f}pip is too large for a market fill")
            style = "LIMIT"
        return {
            "style": style,
            "orderability": "ENTER NOW" if style == "MARKET" else style,
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if regime in {"range", "quiet"}:
        reasons.append(f"{regime} pays better from structural price improvement than from chasing")
        return {
            "style": "LIMIT",
            "orderability": "LIMIT",
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if context_bias == "headwind":
        style, note = _headwind_participation_plan(
            regime,
            cap_rank,
            learning_score,
            trade_count,
            has_exact,
            confirmed_state,
            tight_market_spread,
            normal_market_spread,
        )
        if style and note:
            reasons.append(note)
            return {
                "style": style,
                "orderability": "ENTER NOW" if style == "MARKET" else style,
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }
        reasons.append("learning context is headwind, so require confirmation instead of a blind market fill")
        style = "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"
        return {
            "style": style,
            "orderability": style,
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if regime in {"squeeze", "transition"}:
        if (
            cap_rank >= 4
            and learning_score >= 64
            and normal_market_spread
            and (trade_count >= 3 or has_exact or confirmed_state or context_bias == "tailwind")
        ):
            reasons.append(
                f"{regime} is already leaning one way and the seat has earned immediate participation"
            )
            return {
                "style": "MARKET",
                "orderability": "ENTER NOW",
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }
        if (
            cap_rank in {1, 2, 3}
            and learning_score >= 53
            and normal_market_spread
            and (context_bias == "tailwind" or trade_count >= 3 or has_exact or confirmed_state)
        ):
            reasons.append(
                f"{regime} is leaning enough for a market scout; participate now and still leave one reload LIMIT"
            )
            return {
                "style": "MARKET",
                "orderability": "ENTER NOW",
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }
        reasons.append(f"{regime} tape should be closed as a trigger, not as prose")
        return {
            "style": "STOP-ENTRY",
            "orderability": "STOP-ENTRY",
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    if regime == "trending":
        if (
            cap_rank >= 4
            and learning_score >= 58
            and normal_market_spread
            and (trade_count >= 3 or has_exact or confirmed_state or context_bias == "tailwind")
        ):
            if tight_market_spread or context_bias == "tailwind" or cap_rank >= 5:
                reasons.append(
                    "trend is already paying and the learning cap is strong enough to justify immediate execution"
                )
            else:
                reasons.append(
                    "trend is live enough now; enter the market print and still keep one reload LIMIT in the plan"
                )
            return {
                "style": "MARKET",
                "orderability": "ENTER NOW",
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }
        if (
            cap_rank in {1, 2, 3}
            and learning_score >= 50
            and normal_market_spread
            and (context_bias == "tailwind" or trade_count >= 3 or has_exact or confirmed_state)
        ):
            reasons.append(
                "B-conviction seat with live tape leaning one way; take a market scout now and keep one reload LIMIT"
            )
            return {
                "style": "MARKET",
                "orderability": "ENTER NOW",
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }

    if regime == "trending":
        reasons.append("the trend is alive enough to keep a trigger armed even when the market scout lane is not justified")
        return {
            "style": "STOP-ENTRY",
            "orderability": "STOP-ENTRY",
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
        }

    reasons.append("the seat is alive, but better closed as an armed order than as a market chase")
    return {
        "style": "LIMIT",
        "orderability": "LIMIT",
        "note": "; ".join(reasons),
        "spread_ratio": spread_ratio,
        "target_pips": target_pips,
    }


def _registry_hits_for_target(target: dict, registry: dict) -> list[dict]:
    lessons = registry.get("lessons") or []
    if not lessons:
        return []

    pair = target.get("pair")
    direction = target.get("direction")
    if not pair or not direction:
        return []

    exact = [
        lesson for lesson in lessons
        if lesson.get("pair") == pair
        and lesson.get("direction") == direction
        and lesson.get("state") != "deprecated"
    ]
    pair_only = [
        lesson for lesson in lessons
        if lesson.get("pair") == pair
        and not lesson.get("direction")
        and lesson.get("state") != "deprecated"
    ]
    generic = [
        lesson for lesson in lessons
        if lesson.get("lesson_type") in {"playbook", "hygiene"}
        and lesson.get("state") == "confirmed"
    ]

    def registry_rank(lesson: dict) -> tuple[int, int, str, str]:
        return (
            int(lesson.get("trust_score", 0)),
            int(lesson.get("state_rank", 0)),
            str(lesson.get("lesson_date", "")),
            str(lesson.get("id", "")),
        )

    selected: list[dict] = []
    seen_ids = set()
    for pool in (sorted(exact, key=registry_rank, reverse=True),
                 sorted(pair_only, key=registry_rank, reverse=True),
                 sorted(generic, key=registry_rank, reverse=True)):
        for lesson in pool:
            lesson_id = lesson.get("id")
            if lesson_id in seen_ids:
                continue
            seen_ids.add(lesson_id)
            trade_stats = lesson.get("trade_stats") or {}
            stats_text = ""
            if trade_stats.get("count"):
                stats_text = (
                    f" EV {float(trade_stats.get('ev', 0.0)):+.0f}"
                    f" WR {float(trade_stats.get('win_rate', 0.0))*100:.0f}%"
                )
            summary = (
                f"trusted lesson {lesson.get('trust_score', 0)}/100 "
                f"[{lesson.get('state', '?')}/{lesson.get('lesson_type', '?')}] "
                f"{lesson.get('title', '').strip()}{stats_text}"
            ).strip()
            selected.append({
                "id": f"registry:{lesson_id}",
                "session_date": registry.get("strategy_updated") or lesson.get("lesson_date"),
                "chunk_type": "lesson",
                "question": summary,
                "content": lesson.get("text", ""),
                "pair": lesson.get("pair"),
                "direction": lesson.get("direction"),
                "tags": ",".join(
                    filter(
                        None,
                        [
                            "registry",
                            "lesson",
                            str(lesson.get("state", "")),
                            str(lesson.get("lesson_type", "")),
                            pair.lower() if pair else "",
                            direction.lower() if direction else "",
                        ],
                    )
                ),
                "source_file": "lesson_registry.json",
                "match_type": "registry",
                "trust_score": lesson.get("trust_score"),
                "state": lesson.get("state"),
                "lesson_type": lesson.get("lesson_type"),
            })
            if len(selected) >= 2:
                return selected
    return selected


def _summarize_memory_hit(hit: dict) -> str:
    question = (hit.get("question") or "").strip()
    if question:
        return question
    for raw_line in hit.get("content", "").splitlines():
        line = raw_line.strip().strip("-")
        if not line or line.startswith("|") or line.startswith("##") or line.startswith("###"):
            continue
        return line[:160]
    return "(no summary)"


def _compact_memory_hits(hits: list[dict]) -> list[dict]:
    preferred = {"trade": 0, "lesson": 1, "user_call": 2, "thesis": 8, "summary": 9}

    def date_rank(hit: dict) -> int:
        date_text = str(hit.get("session_date", "")).replace("-", "")
        return int(date_text) if date_text.isdigit() else 0

    def id_rank(hit: dict) -> int:
        raw = str(hit.get("id", "0"))
        return int(raw) if raw.isdigit() else 0

    def score(hit: dict) -> tuple[int, int, int, int]:
        chunk_type = hit.get("chunk_type", "")
        source_file = hit.get("source_file", "")
        tags = {tag.strip() for tag in str(hit.get("tags", "")).split(",") if tag.strip()}
        source_penalty = 3 if source_file == "state.md" else 0
        tag_penalty = 0
        if "lesson" in tags:
            tag_penalty -= 1
        if "cancel" in tags or "carry" in tags or "position" in tags:
            tag_penalty += 3
        elif "pending" in tags:
            tag_penalty += 1
        return (
            preferred.get(chunk_type, 9) + source_penalty + tag_penalty,
            0 if source_file == "trades.md" else 1,
            -date_rank(hit),
            -id_rank(hit),
        )

    ranked = sorted(hits, key=score)
    unique = []
    seen = set()
    for hit in ranked:
        key = (hit.get("session_date"), _summarize_memory_hit(hit))
        if key in seen:
            continue
        unique.append(hit)
        seen.add(key)
    if unique:
        selected = [unique[0]]
        if len(unique) > 1:
            top_is_lesson = unique[0].get("chunk_type") == "lesson"
            lesson_candidate = next(
                (
                    hit for hit in unique[1:]
                    if hit.get("chunk_type") == "lesson"
                    or hit.get("source_file") == "strategy_memory.md"
                ),
                None,
            )
            if lesson_candidate and not top_is_lesson:
                selected.append(lesson_candidate)
        if len(selected) < 2:
            for hit in unique[1:]:
                if hit in selected:
                    continue
                selected.append(hit)
                if len(selected) >= 2:
                    break
        if selected:
            return selected

    ranked_fallback = sorted(hits, key=lambda hit: (-date_rank(hit), -id_rank(hit)))
    unique = []
    seen = set()
    for hit in ranked_fallback:
        key = (hit.get("session_date"), _summarize_memory_hit(hit))
        if key in seen:
            continue
        unique.append(hit)
        seen.add(key)
        if len(unique) >= 2:
            break
    return unique


def _run_actionable_memory_recall(targets: list[dict]) -> list[tuple[dict, list[dict], str | None]]:
    if not targets:
        return []
    try:
        hybrid_search = _load_memory_search()
    except Exception as exc:
        return [(target, [], f"(skip: {exc})") for target in targets]

    registry = _load_lesson_registry()
    results = []
    for target in targets:
        try:
            registry_hits = _registry_hits_for_target(target, registry)
            hits = hybrid_search(
                _memory_query_for_target(target),
                top_k=8,
                pair=target["pair"],
                direction=target["direction"],
            )
            if registry_hits:
                hits = registry_hits + [
                    hit for hit in hits
                    if hit.get("source_file") != "strategy_memory.md"
                ]
            results.append((target, _compact_memory_hits(hits), None))
        except Exception as exc:
            results.append((target, [], f"(skip: {exc})"))
    return results


def _load_technicals(root, pair):
    f = root / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def _print_currency_pulse(root):
    """Cross-currency triangulation at H4, M15, M1."""
    # Gather DI signals per currency per TF
    ccy_signals = {c: {tf: [] for tf in ["H4", "M15", "M1"]} for c in ["USD", "JPY", "EUR", "GBP", "AUD"]}

    all_tech = {}
    for pair in PAIRS:
        all_tech[pair] = _load_technicals(root, pair)

    for pair, (base, quote) in PAIR_CURRENCIES.items():
        tech = all_tech.get(pair, {})
        for tf in ["H4", "M15", "M1"]:
            d = tech.get(tf, {})
            di_p = d.get("plus_di", 0)
            di_m = d.get("minus_di", 0)
            adx = d.get("adx", 0)
            if adx < 5:
                continue
            gap = (di_p - di_m) * min(adx / 25.0, 1.5)
            ccy_signals[base][tf].append((pair, gap))
            ccy_signals[quote][tf].append((pair, -gap))

    # Print per currency
    strongest_h4 = ("", -999)
    weakest_h4 = ("", 999)

    for ccy in ["USD", "JPY", "EUR", "GBP", "AUD"]:
        parts = []
        for tf in ["H4", "M15", "M1"]:
            sigs = ccy_signals[ccy][tf]
            if not sigs:
                parts.append(f"{tf}=?")
                continue
            avg = sum(g for _, g in sigs) / len(sigs)
            if avg > 3:
                label = f"BID(+{avg:.0f})"
            elif avg < -3:
                label = f"offered({avg:.0f})"
            else:
                label = f"neutral({avg:+.0f})"
            parts.append(f"{tf}={label}")

            if tf == "H4":
                if avg > strongest_h4[1]:
                    strongest_h4 = (ccy, avg)
                if avg < weakest_h4[1]:
                    weakest_h4 = (ccy, avg)

        # Detect H4↔M15 conflict
        h4_sigs = ccy_signals[ccy]["H4"]
        m15_sigs = ccy_signals[ccy]["M15"]
        h4_avg = sum(g for _, g in h4_sigs) / len(h4_sigs) if h4_sigs else 0
        m15_avg = sum(g for _, g in m15_sigs) / len(m15_sigs) if m15_sigs else 0

        conflict = ""
        if h4_avg > 3 and m15_avg < -3:
            conflict = " → H4↔M15 CONFLICT (H4 bid but M15 selling)"
        elif h4_avg < -3 and m15_avg > 3:
            conflict = " → H4↔M15 CONFLICT (H4 offered but M15 buying)"
        elif abs(h4_avg) <= 3 and abs(m15_avg) <= 3:
            conflict = " → No direction"
        else:
            conflict = ""

        print(f"{ccy}: {' | '.join(parts)}{conflict}")

    # M1 synchrony detection
    for ccy in ["USD", "JPY", "EUR", "GBP", "AUD"]:
        m1_sigs = ccy_signals[ccy]["M1"]
        if len(m1_sigs) >= 3:
            all_pos = all(g > 0 for _, g in m1_sigs)
            all_neg = all(g < 0 for _, g in m1_sigs)
            if all_pos:
                print(f"⚠ M1 synchrony: {ccy} BID across {len(m1_sigs)}/{len(m1_sigs)} crosses")
            elif all_neg:
                print(f"⚠ M1 synchrony: {ccy} OFFERED across {len(m1_sigs)}/{len(m1_sigs)} crosses")

    # Correlation break detection
    for pair_a, pair_b, label in CORRELATED_PAIRS:
        tech_a = all_tech.get(pair_a, {}).get("M15", {})
        tech_b = all_tech.get(pair_b, {}).get("M15", {})
        di_a = tech_a.get("plus_di", 0) - tech_a.get("minus_di", 0)
        di_b = tech_b.get("plus_di", 0) - tech_b.get("minus_di", 0)
        adx_a = tech_a.get("adx", 0)
        adx_b = tech_b.get("adx", 0)
        # Break = one clearly directional (ADX>15, |DI gap|>5) and other flat or opposite
        if adx_a > 15 and abs(di_a) > 5 and (adx_b < 12 or di_a * di_b < 0):
            dir_a = "bull" if di_a > 0 else "bear"
            dir_b = "flat" if adx_b < 12 else ("bull" if di_b > 0 else "bear")
            print(f"⚠ Correlation break ({label}): {pair_a} M15={dir_a}(ADX={adx_a:.0f}) vs {pair_b} M15={dir_b}(ADX={adx_b:.0f})")
        elif adx_b > 15 and abs(di_b) > 5 and (adx_a < 12 or di_a * di_b < 0):
            dir_a = "flat" if adx_a < 12 else ("bull" if di_a > 0 else "bear")
            dir_b = "bull" if di_b > 0 else "bear"
            print(f"⚠ Correlation break ({label}): {pair_a} M15={dir_a}(ADX={adx_a:.0f}) vs {pair_b} M15={dir_b}(ADX={adx_b:.0f})")

    # Best vehicle
    if strongest_h4[0] and weakest_h4[0] and strongest_h4[0] != weakest_h4[0]:
        # Find the pair that matches strongest vs weakest
        best_pair = None
        best_dir = None
        for pair, (base, quote) in PAIR_CURRENCIES.items():
            if base == strongest_h4[0] and quote == weakest_h4[0]:
                best_pair = pair
                best_dir = "LONG"
            elif base == weakest_h4[0] and quote == strongest_h4[0]:
                best_pair = pair
                best_dir = "SHORT"
        if best_pair:
            print(f"Best vehicle: {strongest_h4[0]}(+{strongest_h4[1]:.0f}) vs {weakest_h4[0]}({weakest_h4[1]:.0f}) → {best_pair} {best_dir}")
        else:
            print(f"Best vehicle: {strongest_h4[0]}(+{strongest_h4[1]:.0f}) vs {weakest_h4[0]}({weakest_h4[1]:.0f}) → no direct pair")


def _print_h4_position(root):
    """Where in the H4 wave — lifecycle label for each pair."""
    for pair in PAIRS:
        tech = _load_technicals(root, pair)
        h4 = tech.get("H4", {})
        if not h4:
            print(f"{pair}: NO H4 DATA")
            continue

        stoch_rsi = h4.get("stoch_rsi", 0.5)
        s5 = h4.get("ema_slope_5", 0)
        s10 = h4.get("ema_slope_10", 0)
        adx = h4.get("adx", 0)
        di_p = h4.get("plus_di", 0)
        di_m = h4.get("minus_di", 0)
        vwap_gap = h4.get("vwap_gap", 0)
        cci = h4.get("cci", 0)

        # Direction
        bull = di_p > di_m
        direction = "BULL" if bull else "BEAR"
        if adx < 20:
            direction = "NEUTRAL"

        # StRSI zone
        if stoch_rsi >= 0.9:
            zone = "ceiling!"
        elif stoch_rsi >= 0.7:
            zone = "upper"
        elif stoch_rsi >= 0.4:
            zone = "mid"
        elif stoch_rsi >= 0.15:
            zone = "lower"
        else:
            zone = "floor"

        # Momentum quality (slope comparison)
        abs_s5 = abs(s5)
        abs_s10 = abs(s10)
        if abs_s10 > 1e-8:
            ratio = abs_s5 / abs_s10
        else:
            ratio = 1.0

        if ratio > 1.15:
            accel = "accelerating"
        elif ratio < 0.75:
            accel = "decelerating"
        else:
            accel = "steady"

        # Lifecycle label
        if direction == "NEUTRAL":
            label = "NO TREND"
        elif stoch_rsi >= 0.9 and bull:
            label = "EXHAUSTING BULL" if accel != "accelerating" else "LATE BULL (accel)"
        elif stoch_rsi <= 0.1 and not bull:
            label = "EXHAUSTING BEAR" if accel != "accelerating" else "LATE BEAR (accel)"
        elif stoch_rsi >= 0.9 and not bull:
            label = "EARLY BEAR" if adx > 25 else "BEAR SETUP"
        elif stoch_rsi <= 0.1 and bull:
            label = "EARLY BULL" if adx > 25 else "BULL SETUP"
        elif stoch_rsi >= 0.6:
            label = f"LATE {direction}" if accel == "decelerating" else f"MID {direction}"
        elif stoch_rsi <= 0.3:
            label = f"EARLY {direction}" if accel != "decelerating" else f"BOTTOM {direction}"
        else:
            label = f"MID {direction}"

        room = "room" if 0.15 < stoch_rsi < 0.85 else "limited" if 0.85 <= stoch_rsi or stoch_rsi <= 0.15 else "mid"

        print(
            f"{pair}: StRSI={stoch_rsi:.2f}({zone}) "
            f"slope={accel} "
            f"VWAP={vwap_gap:+.0f}pip "
            f"→ {label} [{room}]"
        )


def main():
    t0 = time.time()

    # Parse args — last_slack_ts from CLI or auto-read from file
    last_slack_ts = ""
    if "--state-ts" in sys.argv:
        idx = sys.argv.index("--state-ts")
        if idx + 1 < len(sys.argv):
            last_slack_ts = sys.argv[idx + 1]
    if not last_slack_ts:
        ts_file = ROOT / "logs" / ".slack_last_read_ts"
        if ts_file.exists():
            last_slack_ts = ts_file.read_text().strip()

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # === Session time marker (first thing in output) ===
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if 0 <= hour < 3:
        session_label = "Tokyo"
    elif 3 <= hour < 6:
        session_label = "Tokyo (pre-London positioning)"
    elif 6 <= hour < 8:
        session_label = "Tokyo-London overlap"
    elif 8 <= hour < 12:
        session_label = "London"
    elif 12 <= hour < 16:
        session_label = "London-NY overlap"
    elif 16 <= hour < 21:
        session_label = "NY"
    else:
        session_label = "Late NY (rollover zone)"
    utc_stamp = now_utc.strftime('%Y-%m-%d %H:%M UTC')
    print(f"=== SESSION: {utc_stamp} | {session_label} ===")
    state_path = ROOT / "collab_trade" / "state.md"
    if state_path.exists():
        state_mtime = datetime.fromtimestamp(state_path.stat().st_mtime, tz=timezone.utc)
        state_age_min = max(0.0, (now_utc - state_mtime).total_seconds() / 60)
        print(
            f"state.md last modified: {state_mtime.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({state_age_min:.0f}min ago)"
        )
    else:
        print("state.md last modified: missing")
    state_text = state_path.read_text() if state_path.exists() else ""
    carry_targets = _parse_state_carry_targets(state_text)
    hot_updates = _parse_hot_updates(state_text)
    if hot_updates:
        section("HOT UPDATES FROM LAST SESSION")
        for update in hot_updates:
            print(f"- {update}")
    if carry_targets:
        section("STATE CARRY-FORWARD WATCHLIST")
        for item in carry_targets:
            print(f"{item['pair']} {item['direction']} [{str(item.get('source', '?')).upper()}] | {item.get('recipe')}")

    # === PARALLEL BLOCK: Start heavy I/O tasks concurrently ===
    executor = ThreadPoolExecutor(max_workers=4)

    def _run_tech_refresh():
        return run_script(
            [VENV_PYTHON, "tools/refresh_factor_cache.py", "--all", "--quiet"],
            timeout=45,
        )

    def _run_m5_candles():
        results = {}
        for pair in PAIRS:
            try:
                candles_resp = oanda_api(
                    f"/v3/instruments/{pair}/candles?granularity=M5&count=20&price=M",
                    cfg,
                )
                results[pair] = candles_resp.get("candles", [])
            except Exception as e:
                results[pair] = []
        return results

    # Submit heavy tasks in parallel
    future_tech = executor.submit(_run_tech_refresh)
    future_m5 = executor.submit(_run_m5_candles)

    trades_data = {}
    pending_orders = []
    try:
        trades_data = oanda_api(f"/v3/accounts/{acct}/openTrades", cfg)
    except Exception:
        pass

    try:
        pending = oanda_api(f"/v3/accounts/{acct}/pendingOrders", cfg)
        pending_orders = pending.get("orders", [])
    except Exception:
        pending_orders = []

    # Wait for tech refresh (needed before adaptive_technicals)
    tech_out = future_tech.result()
    section("TECH REFRESH")
    print(tech_out[:200] if tech_out else "done")

    chart_status = ensure_chart_snapshots_fresh()
    section("CHART SNAPSHOTS")
    oldest_chart_age = chart_status["oldest_age_min"]
    newest_chart_age = chart_status["newest_age_min"]
    if chart_status["refreshed"]:
        print(f"Refreshed chart PNGs incl. M1 execution set ({chart_status['refresh_reason']}).")
        if chart_status["refresh_output"]:
            lines = [line for line in chart_status["refresh_output"].splitlines() if line.strip()]
            print("\n".join(lines[:6]))
    elif oldest_chart_age is None:
        print("⚠️ chart PNGs unavailable")
    else:
        print(
            f"Fresh local chart set ready (M1/M5/H1) "
            f"(oldest={oldest_chart_age:.0f}min newest={newest_chart_age:.0f}min)."
        )
    if chart_status["missing"]:
        print(f"Missing before refresh: {', '.join(chart_status['missing'][:6])}")

    # Wait for M5 candles and display
    m5_candles = future_m5.result()
    section("M5 PRICE ACTION (read this FIRST — before indicators)")
    try:
        for pair in PAIRS:
            candles = m5_candles.get(pair, [])
            if not candles:
                continue
            # Analyze candle shape
            bodies = []
            upper_wicks = []
            lower_wicks = []
            pip_factor = 100 if "JPY" in pair else 10000
            directions = []  # 1=bull, -1=bear, 0=doji
            for c in candles:
                mid = c.get("mid", {})
                o, h, l, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
                body = abs(cl - o) * pip_factor
                bodies.append(body)
                if cl >= o:  # bull
                    upper_wicks.append((h - cl) * pip_factor)
                    lower_wicks.append((o - l) * pip_factor)
                    directions.append(1)
                else:  # bear
                    upper_wicks.append((h - o) * pip_factor)
                    lower_wicks.append((cl - l) * pip_factor)
                    directions.append(-1)
                if body < 0.3:
                    directions[-1] = 0

            # First half vs second half — momentum change detection
            half = len(bodies) // 2
            first_avg = sum(bodies[:half]) / max(half, 1)
            second_avg = sum(bodies[half:]) / max(len(bodies) - half, 1)
            if second_avg > first_avg * 1.3:
                momentum = "accelerating (bodies growing)"
            elif second_avg < first_avg * 0.7:
                momentum = "exhausting (bodies shrinking)"
            else:
                momentum = "steady"

            # Recent direction (last 5 candles)
            recent = directions[-5:]
            bulls = sum(1 for d in recent if d > 0)
            bears = sum(1 for d in recent if d < 0)
            if bulls >= 4:
                bias = "buyers dominant"
            elif bears >= 4:
                bias = "sellers dominant"
            elif bulls >= 3:
                bias = "buyers leaning"
            elif bears >= 3:
                bias = "sellers leaning"
            else:
                bias = "contested"

            # Wick analysis (reversal pressure)
            avg_upper = sum(upper_wicks[-5:]) / 5
            avg_lower = sum(lower_wicks[-5:]) / 5
            wick_note = ""
            if avg_upper > second_avg * 0.5 and avg_upper > avg_lower * 1.5:
                wick_note = " | upper wicks expanding (selling pressure)"
            elif avg_lower > second_avg * 0.5 and avg_lower > avg_upper * 1.5:
                wick_note = " | lower wicks expanding (buying pressure)"

            # High/low update pattern
            last_c = candles[-1]["mid"]
            prev_c = candles[-2]["mid"]
            hh = float(last_c["h"]) > float(prev_c["h"])
            ll = float(last_c["l"]) < float(prev_c["l"])
            hl_note = ""
            if hh and not ll:
                hl_note = " | making higher highs"
            elif ll and not hh:
                hl_note = " | making lower lows"
            elif hh and ll:
                hl_note = " | range expanding"

            last_price = float(candles[-1]["mid"]["c"])
            print(f"{pair} @{last_price:.5g}: {bias}, {momentum}{wick_note}{hl_note}")
    except Exception as e:
        print(f"(skip: {e})")

    # 2. OANDA: prices, positions, account
    section("PRICES")
    spread_data = {}  # pair -> spread_pips (referenced in other sections)
    try:
        prices = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={','.join(PAIRS)}", cfg)
        for p in prices.get("prices", []):
            pair = p["instrument"]
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
            pip_factor = 100 if "JPY" in pair else 10000
            spread_pip = (ask - bid) * pip_factor
            spread_data[pair] = spread_pip
            warn = " ⚠️ spread wide" if spread_pip > 2.0 else ""
            print(
                f"{pair} bid={p['bids'][0]['price']} ask={p['asks'][0]['price']} Sp={spread_pip:.1f}pip{warn}"
            )
    except Exception as e:
        print(f"ERROR: {e}")

    _print_tokyo_open_breadth(cfg, now_utc, spread_data)

    # Load pair edge stats from strategy_feedback.json (written by trade_performance.py)
    pair_edge = {}
    feedback_path = ROOT / "logs" / "strategy_feedback.json"
    if feedback_path.exists():
        try:
            fb = json.loads(feedback_path.read_text())
            bp = fb.get("by_pair", {})
            if isinstance(bp, dict):
                pair_edge = bp  # keys are pair names, values are stat dicts
            elif isinstance(bp, list):
                for entry in bp:
                    pair_edge[entry.get("pair", "")] = entry
        except Exception:
            pass

    section("TRADES")
    try:
        trades = trades_data  # already fetched above for held_pairs
        for t in trades.get("trades", []):
            pair = t['instrument']
            units = int(t.get('currentUnits', 0))
            side = "LONG" if units > 0 else "SHORT"
            # Inline pair edge
            edge_str = ""
            pe = pair_edge.get(pair)
            if pe:
                wr = pe.get("win_rate", 0)
                total = pe.get("total_pl_jpy", pe.get("total_pl", 0))
                edge_str = f" | edge: {wr:.0%} WR, {total:+.0f}JPY total"
            print(
                f"{pair} {t['currentUnits']}u @{t['price']} PL={t.get('unrealizedPL', 0)} id={t['id']}{edge_str}"
            )
        if not trades.get("trades"):
            print("(no open trades)")
    except Exception as e:
        print(f"ERROR: {e}")

    section("ACCOUNT")
    try:
        summary = oanda_api(f"/v3/accounts/{acct}/summary", cfg).get(
            "account", {}
        )
        nav = float(summary.get("NAV", 0))
        margin_used = float(summary.get("marginUsed", 0))
        margin_pct = (margin_used / nav * 100) if nav > 0 else 0
        margin_warn = ""
        if margin_pct >= 95:
            margin_warn = " 🚨 CRITICAL — force half-close now (rule: 95%+)"
        elif margin_pct >= 90:
            margin_warn = " 🚨 DANGER — no new entries (rule: 90%+)"
        print(
            f"NAV:{summary.get('NAV')} Bal:{summary.get('balance')} "
            f"Margin:{summary.get('marginUsed')}/{summary.get('marginAvailable')} "
            f"({margin_pct:.1f}%){margin_warn}"
        )
    except Exception as e:
        print(f"ERROR: {e}")

    # Churn detection: today's entry count per pair
    log_path = ROOT / "logs" / "live_trade_log.txt"
    if log_path.exists():
        today_str = now_utc.strftime("%Y-%m-%d")
        from collections import Counter
        entry_counts = Counter()
        for line in log_path.read_text().strip().split("\n"):
            if "ENTRY" in line and today_str in line:
                for p in PAIRS:
                    if p in line:
                        entry_counts[p] += 1
                        break
        if entry_counts:
            parts = [f"{p}×{c}" for p, c in entry_counts.most_common()]
            total = sum(entry_counts.values())
            churn_warn = ""
            max_pair_count = entry_counts.most_common(1)[0][1] if entry_counts else 0
            if max_pair_count >= 4:
                churn_warn = " ⚠️ churn risk — same pair 4+ times"
            print(f"Today's entries: {' '.join(parts)} | total {total}{churn_warn}")

    # 2b. Pending Orders (limit orders, TP/SL check)
    section("PENDING ORDERS")
    try:
        if pending_orders:
            for o in pending_orders:
                otype = o.get("type", "?")
                pair = o.get("instrument", "?")
                units = o.get("units", "?")
                price = o.get("price", "?")
                gtd = o.get("gtdTime", "GTC")[:16] if o.get("gtdTime") else "GTC"
                print(f"{otype} {pair} {units}u @{price} exp={gtd} id={o.get('id', '?')}")
        else:
            print("(no pending orders)")
    except Exception as e:
        print(f"ERROR: {e}")

    # 2c. Trade attached orders (TP/SL/Trailing)
    section("TRADE PROTECTIONS")
    try:
        for t in trades.get("trades", []):
            protections = []
            if t.get("takeProfitOrder"):
                protections.append(f"TP={t['takeProfitOrder'].get('price', '?')}")
            if t.get("stopLossOrder"):
                protections.append(f"SL={t['stopLossOrder'].get('price', '?')}")
            if t.get("trailingStopLossOrder"):
                protections.append(f"Trail={t['trailingStopLossOrder'].get('distance', '?')}")
            if protections:
                print(f"{t['instrument']} id={t['id']}: {' | '.join(protections)}")
            else:
                print(f"{t['instrument']} id={t['id']}: ⚠️ NO PROTECTION")
    except Exception:
        pass

    section("STATE POSITION SYNC")
    if not state_path.exists():
        print("state.md missing — create/update it before analysis.")
    else:
        sync = _build_state_position_sync(state_text, trades_data)
        if sync["match"]:
            print("state.md Positions section matches live OANDA.")
        else:
            print(
                "Mismatch — fix `## Positions (Current)` before analysis if this is not intentional."
            )
            for p in sync["live_not_in_state"]:
                ids = ",".join(p["trade_ids"])
                print(
                    f"OANDA only: {p['pair']} {p['direction']} {p['units']}u id={ids}"
                )
            for p in sync["state_not_live"]:
                print(
                    f"state.md only: {p['pair']} {p['direction']} | {p['line']}"
                )

    # 2d. News digest (created by Cowork on 1-hour interval)
    section("NEWS DIGEST")
    news_digest = ROOT / "logs" / "news_digest.md"
    if news_digest.exists():
        digest_text = news_digest.read_text().strip()
        # Freshness check: file modification time
        age_min = (time.time() - news_digest.stat().st_mtime) / 60
        if age_min > 90:
            print(f"⚠️ news stale ({age_min:.0f}min ago)")
        print(digest_text[:2000])  # max 2000 chars
    else:
        print("(news_digest.md not found — Cowork qr-news-digest task not run)")

    # 2d2. Economic calendar (upcoming events from news_cache)
    section("UPCOMING EVENTS (next 4h)")
    try:
        nc = ROOT / "logs" / "news_cache.json"
        if nc.exists():
            ncdata = json.loads(nc.read_text())
            cal = ncdata.get("calendar", [])
            if cal:
                shown = 0
                for ev in cal[:10]:
                    title = ev.get("event", "")
                    impact = ev.get("impact", "")
                    country = ev.get("country", "")
                    ev_time = ev.get("time", "")
                    if title:
                        impact_str = f" ({impact} impact)" if impact else ""
                        ccy_str = f" — {country}" if country else ""
                        print(f"{ev_time} {title}{impact_str}{ccy_str}")
                        shown += 1
                if shown == 0:
                    print("(no upcoming events)")
            else:
                print("(no calendar data)")
        else:
            print("(news_cache.json not found)")
    except Exception as e:
        print(f"(skip: {e})")

    # 2e. API parser structured data (re-fetch if cache is stale)
    out = run_script(
        [VENV_PYTHON, "tools/news_fetcher.py", "--if-stale", "60"],
        timeout=20,
    )
    if out and "skip" not in out:
        print(out[:200])
    # Show summary if cache exists
    news_cache = ROOT / "logs" / "news_cache.json"
    if news_cache.exists():
        out = run_script([VENV_PYTHON, "tools/news_fetcher.py", "--summary"])
        if out and "no cache" not in out:
            section("NEWS DATA (structured)")
            print(out)

    # 3. Macro view
    section("MACRO VIEW")
    out = run_script([VENV_PYTHON, "tools/macro_view.py"])
    print(out)

    # 3b. Currency Pulse (cross-currency × 3 TFs)
    section("CURRENCY PULSE (cross-currency × H4/M15/M1)")
    try:
        _print_currency_pulse(ROOT)
    except Exception as e:
        print(f"(currency pulse error: {e})")

    # 3c. H4 Position (where in the wave)
    section("H4 POSITION (where in the wave)")
    try:
        _print_h4_position(ROOT)
    except Exception as e:
        print(f"(h4 position error: {e})")

    # 4. Adaptive technicals
    section("ADAPTIVE TECHNICALS")
    out = run_script([VENV_PYTHON, "tools/adaptive_technicals.py"])
    print(out)

    # 4a. S-Conviction Scan (TF × indicator pattern detection)
    section("S-CONVICTION CANDIDATES")
    scanner_out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    print(scanner_out)

    # 4b. Fib Wave Analysis (M5)
    section("FIB WAVE ANALYSIS (M5)")
    out = run_script([VENV_PYTHON, "tools/fib_wave.py", "--all"])
    if out:
        for line in out.strip().split("\n")[:50]:
            print(line)

    # 4c. Fib Wave Analysis (H1) — multi-TF confluence
    section("FIB WAVE ANALYSIS (H1)")
    out = run_script([VENV_PYTHON, "tools/fib_wave.py", "--all", "H1", "100"])
    if out:
        for line in out.strip().split("\n")[:50]:
            print(line)

    # 5. Slack: user messages
    section("SLACK (user messages)")
    slack_args = [
        VENV_PYTHON,
        "tools/slack_read.py",
        "--channel",
        "C0APAELAQDN",
        "--user-only",
        "--no-update-ts",
    ]
    if last_slack_ts:
        slack_args += ["--after", last_slack_ts, "--limit", "20"]
    else:
        slack_args += ["--limit", "5"]
    out = run_script(slack_args)
    print(out if out else "(no user messages)")

    # 6. Memory recall (held + pending + top scanner candidates)
    memory_targets = _build_memory_targets(trades_data, pending_orders, scanner_out, carry_targets)
    registry = _load_lesson_registry()
    learning_profiles = _build_learning_edge_profiles(memory_targets, registry)
    memory_results = _run_actionable_memory_recall(memory_targets)
    for profile in learning_profiles:
        plan = _recommend_profile_execution(profile, spread_data.get(profile.get("pair", "")))
        profile["execution_style"] = plan.get("style")
        profile["orderability"] = plan.get("orderability")
        profile["execution_note"] = plan.get("note")
        profile["execution_target_pips"] = plan.get("target_pips")
        profile["execution_spread_ratio"] = plan.get("spread_ratio")
    best_direct = None
    best_cross = None
    best_usdjpy = None
    audit_context = {"available": False, "stale": False, "age_min": None, "pairs": {}, "strongest": None}
    podium_seeds: list[dict] = []
    if learning_profiles:
        section("LEARNING EDGE BOARD")
        for profile in learning_profiles:
            recipe = f" | {profile['recipe']}" if profile.get("recipe") else ""
            stats = ""
            if profile.get("trade_count"):
                stats = (
                    f" | WR {float(profile.get('trade_wr', 0.0))*100:.0f}%"
                    f" EV {float(profile.get('trade_ev', 0.0)):+.0f}"
                    f" n={int(profile.get('trade_count', 0))}"
                )
            print(
                f"{profile['pair']} {profile['direction']} [{str(profile.get('source', '?')).upper()}]{recipe} "
                f"| learning {int(profile.get('learning_score', 0))}/100 "
                f"| {profile.get('verdict')} | cap {profile.get('allocation_cap')}{stats}"
            )
            print(f"  Why: {profile.get('evidence')}")
            print(
                f"  Context: {profile.get('session_context')} | "
                f"{profile.get('regime_context')}"
            )
            spread_note = ""
            if profile.get("execution_spread_ratio") is not None:
                spread_note = (
                    f" | spread {spread_data.get(profile['pair'], 0):.1f}pip"
                    f" ({float(profile.get('execution_spread_ratio', 0.0))*100:.0f}% of "
                    f"{float(profile.get('execution_target_pips', 0.0)):.0f}pip path)"
                )
            print(
                f"  Deployment cue: close as {profile.get('execution_style')} "
                f"({profile.get('orderability')}){spread_note}"
            )
            print(f"  Why now: {profile.get('execution_note')}")

        fresh_profiles = [profile for profile in learning_profiles if profile.get("source") != "held"]
        if fresh_profiles:
            fresh_profiles = sorted(
                fresh_profiles,
                key=lambda item: (
                    int(item.get("learning_score", 0)),
                    SOURCE_PRIORITY.get(str(item.get("source", "")), 0),
                    int(item.get("trade_count", 0)),
                    float(item.get("trade_ev", 0.0)),
                ),
                reverse=True,
            )
            section("FRESH-RISK TOURNAMENT (learning-weighted)")
            for idx, profile in enumerate(fresh_profiles[:4], start=1):
                print(
                    f"{idx}. {profile['pair']} {profile['direction']} [{str(profile.get('source', '?')).upper()}] "
                    f"| learning {int(profile.get('learning_score', 0))}/100 "
                    f"| cap {profile.get('allocation_cap')} | {profile.get('verdict')} "
                    f"| exec {profile.get('execution_style')} "
                    f"| {profile.get('current_session')}/{profile.get('current_regime') or 'n/a'}"
                )

            best_direct = next((p for p in fresh_profiles if p.get("bucket") == "direct_usd"), None)
            best_cross = next((p for p in fresh_profiles if p.get("bucket") == "jpy_cross"), None)
            best_usdjpy = next((p for p in fresh_profiles if p.get("bucket") == "usd_jpy"), None)
            if best_direct:
                print(
                    f"Best direct-USD learning seat: {best_direct['pair']} {best_direct['direction']} "
                    f"| cap {best_direct.get('allocation_cap')} | {best_direct.get('verdict')}"
                )
            if best_cross:
                print(
                    f"Best cross learning seat: {best_cross['pair']} {best_cross['direction']} "
                    f"| cap {best_cross.get('allocation_cap')} | {best_cross.get('verdict')}"
                )
            if best_usdjpy:
                print(
                    f"Best USD_JPY learning seat: {best_usdjpy['pair']} {best_usdjpy['direction']} "
                    f"| cap {best_usdjpy.get('allocation_cap')} | {best_usdjpy.get('verdict')}"
                )
            audit_context = _load_audit_narrative_context()
            podium_seeds = _build_s_excavation_podium_seeds(
                fresh_profiles,
                best_direct,
                best_cross,
                best_usdjpy,
                audit_context,
            )
            section("S EXCAVATION SEEDS (copy unless live chart disproves)")
            audit_age = audit_context.get("age_min")
            if audit_context.get("available") and not audit_context.get("stale"):
                age_note = f"{audit_age:.0f}min" if audit_age is not None else "fresh"
                print(
                    "Source mix: fresh-risk tournament + fresh audit strongest-unheld / narrative A-S "
                    f"({age_note} old)."
                )
            elif audit_context.get("stale"):
                age_note = f"{audit_age:.0f}min" if audit_age is not None else "stale"
                print(
                    f"Audit narrative is stale ({age_note}); seeds fall back to the live fresh-risk board only."
                )
            else:
                print("Audit narrative unavailable; seeds fall back to the live fresh-risk board only.")
            if podium_seeds:
                for idx, seed in enumerate(podium_seeds, start=1):
                    print(
                        f"Podium #{idx}: {seed['pair']} {seed['direction']} "
                        f"| Closest-to-S because {seed['closest_to_s_because']} "
                        f"| Still blocked by {seed['still_blocked_by']} "
                        f"| If it upgrades: {seed['upgrade_action']}"
                    )
                if len(podium_seeds) < 3:
                    print(
                        f"Only {len(podium_seeds)} podium seed(s) survived the live execution gate. "
                        "Fill the remaining slot(s) manually only if the chart gives a concrete better seat."
                    )
            else:
                print(
                    "No auto-seeded podium survived the execution gate. Fill Podium #1-#3 manually from the "
                    "live chart and name the concrete contradiction that beat the board."
                )
            section("S-HUNT DEPLOYMENT CUES (always close the seat as an order, not prose)")
            for label, profile in (
                ("Best direct-USD seat", best_direct),
                ("Best cross seat", best_cross),
                ("Best USD_JPY seat", best_usdjpy),
            ):
                if not profile:
                    continue
                spread_note = ""
                if profile.get("execution_spread_ratio") is not None:
                    spread_note = (
                        f" | spread {spread_data.get(profile['pair'], 0):.1f}pip"
                        f" vs {float(profile.get('execution_target_pips', 0.0)):.0f}pip path"
                    )
                print(
                    f"{label}: {profile['pair']} {profile['direction']} "
                    f"[{str(profile.get('source', '?')).upper()}] "
                    f"→ {profile.get('execution_style')} ({profile.get('orderability')}){spread_note}"
                )
                print(f"  Closure rule: {profile.get('execution_note')}")
                print(
                    "  If this is still your best live seat after the chart read, "
                    "do not leave S Hunt as prose. Close it as an order or kill it explicitly."
                )
            multi_vehicle_lanes = _select_multi_vehicle_lanes(fresh_profiles)
            if multi_vehicle_lanes:
                section("MULTI-VEHICLE DEPLOYMENT LANES (when several currencies are alive)")
                for idx, profile in enumerate(multi_vehicle_lanes, start=1):
                    role = _multi_vehicle_role(idx)
                    pair = str(profile.get("pair", ""))
                    base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
                    spread_note = ""
                    if profile.get("execution_spread_ratio") is not None:
                        spread_note = (
                            f" | spread {spread_data.get(pair, 0):.1f}pip"
                            f" vs {float(profile.get('execution_target_pips', 0.0)):.0f}pip path"
                        )
                    print(
                        f"{role}: {pair} {profile['direction']} [{str(profile.get('source', '?')).upper()}] "
                        f"→ {profile.get('execution_style')} ({profile.get('orderability')})"
                        f" | {base}/{quote}{spread_note}"
                    )
                    print(
                        f"  Why this can coexist: {_multi_vehicle_reason(profile, multi_vehicle_lanes[:idx-1])}"
                    )
                    print(f"  Execution: {profile.get('execution_note')}")
                    print(
                        "  Book rule: valid to carry with the other lanes if this is not same-pair averaging "
                        "and worst-case margin after all pending fills stays below 90%."
                    )
            section("INTRADAY LEARNING LOOP (OODA + DECISION JOURNAL)")
            for idx, profile in enumerate(fresh_profiles[:2], start=1):
                print(
                    f"### Seat #{idx}: {profile['pair']} {profile['direction']} "
                    f"[{str(profile.get('source', '?')).upper()}]"
                )
                print(
                    f"  Learning context: {profile.get('session_context')} | "
                    f"{profile.get('regime_context')}"
                )
                print("  Observe: What changed in the live tape since the last read?")
                print(
                    f"  Orient: learning {int(profile.get('learning_score', 0))}/100 | "
                    f"{profile.get('verdict')} | cap {profile.get('allocation_cap')}"
                )
                print(f"    Memory says: {profile.get('evidence')}")
                print(
                    f"  Decide: default close state = {profile.get('orderability')} "
                    f"because {profile.get('execution_note')}"
                )
                print("  Decide override only if the live chart clearly proves a better closure state.")
                print("  Act: [exact order id / exact pass reason]")
                print(f"  Bayesian update: {_bayesian_update_hint(profile)}")
    if memory_results:
        section("ACTIONABLE MEMORY")
        for target, hits, error in memory_results:
            recipe = f" | {target['recipe']}" if target.get("recipe") else ""
            print(f"--- {target['pair']} {target['direction']} [{target['source'].upper()}]{recipe} ---")
            if error:
                print(error)
                continue
            if not hits:
                print("(no comparable memory)")
                continue
            for hit in hits:
                summary = _summarize_memory_hit(hit)
                match_type = hit.get("match_type", "?")
                print(f"{hit.get('session_date', '?')} [{match_type}/{hit.get('chunk_type', '?')}] {summary}")

    # 6b. Quality Audit status + latest preserved analysis
    audit_path = ROOT / "logs" / "quality_audit.md"
    section("QUALITY AUDIT STATUS")
    if audit_path.exists():
        age_min = (time.time() - audit_path.stat().st_mtime) / 60
        print(f"Audit memo age: {age_min:.0f}min")
        if age_min > QUALITY_AUDIT_STALE_MIN:
            print("⚠️ audit narrative is stale. Use fresh PNGs + raw data first; treat old audit prose as context only.")
        if age_min <= QUALITY_AUDIT_STALE_MIN:
            audit_text = audit_path.read_text().strip()
            if audit_text:
                section(f"QUALITY AUDIT ({age_min:.0f}min ago — facts + latest preserved analysis)")
                print(audit_text[:5000])
    else:
        print("quality_audit.md missing")

    # 7. Today's performance
    section("PERFORMANCE (today)")
    out = run_script([VENV_PYTHON, "tools/trade_performance.py", "--days", "1"])
    if out:
        lines = out.split("\n")[:20]
        print("\n".join(lines))

    # 8. Pre-filled templates (v8.4 — model fills blanks, can't skip fields)
    section("⚠ MANDATORY TEMPLATES — write ALL of these to state.md FIRST, before any analysis")

    # Self-check template with pre-filled entry count
    entry_count_str = ""
    try:
        today_str = time.strftime("%Y-%m-%d", time.gmtime())
        log_path = ROOT / "logs" / "live_trade_log.txt"
        if log_path.exists():
            from collections import Counter as _C
            _ec = _C()
            for _line in log_path.read_text().strip().split("\n"):
                if "ENTRY" in _line and today_str in _line:
                    for _p in PAIRS:
                        if _p in _line:
                            _ec[_p] += 1
                            break
            if _ec:
                _top = _ec.most_common(1)[0]
                entry_count_str = f"{sum(_ec.values())} total. Most: {_top[0]} ×{_top[1]}"
            else:
                entry_count_str = "0 today"
    except Exception:
        entry_count_str = "? (check log)"

    print(f"""
## Self-check ← WRITE THIS FIRST (before Market Narrative)
Entries today: {entry_count_str}. Fixated on one pair? [YES: why / NO: justified because ___]
Last 3 closed trades: [W/L/W]. Streak: [hot/cold/neutral]. If 2+L → B max only, don't chase
Bias: Am I holding ___ because thesis is alive, or because cutting = admitting I was wrong?
Macro chain: How does the current theme affect EACH currency?
  USD: ___ | EUR: ___ | GBP: ___ | JPY: ___ | AUD: ___""")

    print("""
## Hot Updates (carry into next session)
- [UTC time] Pair / direction / trigger quality / what changed / next-seat correction
- [UTC time] Example: GBP_JPY SHORT rollover hold only. Valid late-NY breakdown became hold-only management; no fresh execution before spreads normalize.
- [UTC time] Example: EUR_USD LONG missed because reclaim never printed. Next seat must stay trigger-honest, not a higher rewrite.
""")

    # Position management template per held trade
    held_trades = trades.get("trades", []) if trades else []
    if held_trades:
        print("\n## Position Management (fill ALL 4 for each C)")
        for t in held_trades:
            pair = t.get("instrument", "?").replace("/", "_")
            tid = t.get("id", "?")
            units = t.get("currentUnits", "?")
            upl = t.get("unrealizedPL", "0")
            # Load M15 data for this pair
            m15_hint = ""
            m1_hint = ""
            h4_hint = ""
            try:
                tech = _load_technicals(ROOT, pair)
                m15 = tech.get("M15", {})
                m1_data = tech.get("M1", {})
                h4 = tech.get("H4", {})
                if m15:
                    di_gap = m15.get("plus_di", 0) - m15.get("minus_di", 0)
                    hist = m15.get("macd_hist", 0)
                    hist_dir = "expanding" if abs(hist) > 0.0001 else "shrinking"
                    m15_hint = f"DI gap={di_gap:+.0f} MACD hist={hist_dir}"
                if h4:
                    sr = h4.get("stoch_rsi", 0.5)
                    zone = "ceiling!" if sr >= 0.9 else "upper" if sr >= 0.7 else "mid" if sr >= 0.4 else "lower" if sr >= 0.15 else "floor"
                    h4_hint = f"StRSI={sr:.2f}({zone})"
                # M1 currency pulse for base currency
                base_ccy = pair[:3]
                m1_sigs = []
                for p2, (b2, q2) in PAIR_CURRENCIES.items():
                    m1d = _load_technicals(ROOT, p2).get("M1", {})
                    if m1d:
                        gap = m1d.get("plus_di", 0) - m1d.get("minus_di", 0)
                        if b2 == base_ccy:
                            m1_sigs.append(gap)
                        elif q2 == base_ccy:
                            m1_sigs.append(-gap)
                if m1_sigs:
                    avg = sum(m1_sigs) / len(m1_sigs)
                    m1_label = "BID" if avg > 3 else "offered" if avg < -3 else "neutral"
                    m1_hint = f"{base_ccy} M1={m1_label}({avg:+.0f}) across {len(m1_sigs)} crosses"
            except Exception:
                pass

            print(f"""
### {pair} {units}u id={tid} UPL={upl}
  A — Close now: {upl} JPY
  B — Half TP: close ___u, trail ___pip
  C — Hold REQUIRES all 4:
    (1) Changed since last session: ___
    (2) Entry TF: ___
        M15: {m15_hint} → [with/against] ___
        M1: {m1_hint} → [supports/threatens] ___
    (3) H4: {h4_hint} → Room? [YES/NO]
    (4) Enter NOW @current? [YES/NO]
  → Chosen: [A/B/C]""")

    # Conviction template (for new entries)
    print(f"""
## New Entry Template (v8.4 — fill ALL fields)
  Thesis: [CHART story, not indicators]
  Last 5 M5: bodies [___] × [___], wicks [___] → Buyers defending? [YES @___ / NO → PASS]
  Regime: [___] | Type: [___] | Expected: [___] → Zombie: [___Z]
  Theme confidence: [proving/confirmed/late] → allocation lane: [B/A/S] (does NOT change edge)
  Learning verdict: [confirmed edge / watch edge / no-edge / limited history] ← copy from LEARNING EDGE BOARD
  Learning cap: [A/S when theme confirmed / A max / B-only / pass unless exceptional]
  Execution cue from session_data: [MARKET / LIMIT / STOP-ENTRY / PASS] because ___
  If not MARKET: exact structural level / exact trigger = ___ | if PASS: dead thesis because ___
  Tournament rank: [#1 fresh-risk seat / #2 / #3 / unranked]
  Multi-vehicle lane: [PRIMARY / BACKUP / THIRD CURRENCY / NONE]
  AGAINST: ___
  If wrong: ___
  H4 position: {h4_hint if held_trades else "StRSI=___ →"} [early/mid/late/exhausting]
  Cross-currency: [currency] M15 [bid/offered] across [N] pairs → [currency-wide/pair-specific] → conviction [UP/DOWN]
  Event asymmetry: [event] at [time]. Positioned for [___]. [favorable/unfavorable]
  Margin: ___% → worst case ___% | → Edge: [S/A/B/C] Allocation: [S/A/B/C] Size: ___u""")

    print("\n## S Excavation Matrix (write after 7-Pair Scan, before S Hunt)")
    print("Default podium source: copy `S EXCAVATION SEEDS` unless the live chart disproves them.")
    for pair in PAIRS:
        print(
            f"{pair}: Best expression ___ | Why not S now ___ | "
            "Upgrade to S only if ___ | Dead if ___"
        )
    for idx in range(3):
        if idx < len(podium_seeds):
            seed = podium_seeds[idx]
            print(
                f"Podium #{idx + 1}: {seed['pair']} {seed['direction']} "
                f"| Closest-to-S because {seed['closest_to_s_because']} "
                f"| Still blocked by {seed['still_blocked_by']} "
                f"| If it upgrades: {seed['upgrade_action']}"
            )
        else:
            print(
                f"Podium #{idx + 1}: [PAIR LONG/SHORT] | Closest-to-S because ___ | "
                "Still blocked by ___ | If it upgrades: [MARKET / LIMIT / STOP-ENTRY]"
            )
    print("\n## Multi-Vehicle Deployment (when several currencies are alive)")
    print("Lane 1 / PRIMARY: ___ [pair + dir + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 2 / BACKUP: ___ [pair + dir + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 3 / THIRD CURRENCY: ___ [pair + dir + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Book rule: not same-pair averaging | distinct currency expression if possible | worst-case margin after all pending fills < 90%")

    print("""
## Micro AAR (fill immediately after each entry / exit / miss)
  Planned: ___
  Actual: ___
  Gap: [direction / timing / vehicle / size / execution]
  Hot update for next seat: ___
  Bayesian note: [supports prior / contradicts prior / still noise]
""")

    elapsed = time.time() - t0
    print(f"\n[session_data: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
