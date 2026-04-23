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

Usage: python3 tools/session_data.py [--state-ts LAST_SLACK_TS] [--emit-templates]
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_loader import get_oanda_config
from pricing_probe import probe_market
from runtime_history import live_history_scope_label, live_history_start
from technicals_json import load_technicals_timeframes, parse_cache_timestamp, timeframe_age_minutes
from trader_order_guard import (
    exact_pretrade_advisories,
    exact_pretrade_hard_blockers,
    exact_execution_note,
    exact_execution_style,
    exact_pretrade_label,
    run_exact_pretrade,
)

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")
TRADER_LOCK_PATH = ROOT / "logs" / ".trader_lock"
ACTION_BOARD_PATH = ROOT / "logs" / "session_action_board.json"
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
CHART_DIR = ROOT / "logs" / "charts"
LESSON_REGISTRY_PATH = ROOT / "collab_trade" / "memory" / "lesson_registry.json"
CHART_TIMEFRAMES = ("M1", "M5", "H1")
CHART_STALE_MIN = 40
QUALITY_AUDIT_STALE_MIN = 70
JST = timezone(timedelta(hours=9))
SESSION_CADENCE_MIN = 20
TECHNICAL_STALE_LIMITS = {
    "M1": 4.0,
    "M5": 10.0,
    "M15": 30.0,
    "H1": 120.0,
    "H4": 480.0,
}
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
ALLOCATION_BAND_SIZE_TEXT = {
    "B+": "3000-4500u (full-B pressure)",
    "B0": "2000-3000u (normal B scout)",
    "B-": "1000-2000u (probe only / often pass)",
}
ORDER_COMMAND_UNIT_FLOORS = {
    "S": 8000,
    "A": 4000,
    "B+": 3000,
    "B0": 2000,
    "B-": 1000,
    "C": 1000,
}
SCANNER_MEMORY_TARGET_LIMIT = 10
AUDIT_MEMORY_TARGET_LIMIT = 8
AUDIT_RANGE_TARGET_LIMIT = 6
MAX_MEMORY_TARGETS = 18
MAX_DEPLOYMENT_LANES = 10
MAX_PODIUM_SEEDS = 5
MAX_SAME_PAIR_LANES = 4
MAX_SAME_PAIR_DIRECTION_LANES = 3
SOURCE_PRIORITY = {
    "audit_range": 4,
    "audit": 3,
    "pending": 2,
    "state": 1,
    "scanner": 0,
}
RECENT_PRETRADE_LOOKBACK_DAYS = 14
RECENT_TRADE_LOOKBACK_DAYS = 21
RECENT_FEEDBACK_MIN_COUNT = 3
AUDIT_PRESSURE_LOOKBACK_HOURS = 6
AUDIT_REPEAT_TRIGGER_COUNT = 3
MISSED_SEAT_LOOKBACK_HOURS = 12
MISSED_SEAT_MIN_PIPS = 20.0
_TRADE_CONTEXT_STATS = None
_RECENT_REGRET_PAYLOAD = None
_RECENT_AUDIT_PRESSURE = None
_RECENT_MISSED_SEAT_PRESSURE = None
_SAME_DAY_STYLE_CONTEXT = None


def load_config():
    return get_oanda_config()


def _profile_source_label(profile: dict | None) -> str:
    primary = str((profile or {}).get("source") or "?").upper()
    merged_sources: list[str] = []
    for raw in (profile or {}).get("merged_sources") or []:
        text = str(raw or "").upper()
        if text and text not in merged_sources:
            merged_sources.append(text)
    if not merged_sources:
        return primary
    if primary not in merged_sources:
        merged_sources.insert(0, primary)
    extra = len([source for source in merged_sources if source != primary])
    return primary if extra <= 0 else f"{primary}+{extra}"


def _action_snapshot_profile(profile: dict | None) -> dict | None:
    if not profile:
        return None
    default_expression = _profile_default_expression(profile)
    default_orderability = _profile_default_orderability(profile)
    return {
        "pair": str(profile.get("pair") or ""),
        "direction": str(profile.get("direction") or ""),
        "source": str(profile.get("source") or ""),
        "source_label": _profile_source_label(profile),
        "seat_key": str(profile.get("seat_key") or ""),
        "seat_family": str(profile.get("seat_family") or ""),
        "recipe": _ledger_safe_text(str(profile.get("recipe") or ""), 120),
        "default_expression": default_expression,
        "default_orderability": default_orderability,
        "execution_style": default_expression,
        "orderability": default_orderability,
        "allocation_cap": str(profile.get("allocation_cap") or ""),
        "allocation_band": str(profile.get("allocation_band") or ""),
        "verdict": str(profile.get("verdict") or ""),
        "live_tape": _profile_live_tape_label(profile),
        "shelf_life_min": profile.get("shelf_life_min"),
        "shelf_life_sessions": profile.get("shelf_life_sessions"),
        "expires_at_utc": str(profile.get("expires_at_utc") or ""),
        "shelf_life_label": str(profile.get("shelf_life_label") or ""),
        "carry_rule": _ledger_safe_text(str(profile.get("carry_rule") or ""), 180),
        "pressure": _ledger_safe_text(str(profile.get("promotion_pressure_note") or ""), 160),
        "corroboration": _ledger_safe_text(str(profile.get("cross_source_corroboration") or ""), 180),
        "merged_sources": [str(item) for item in (profile.get("merged_sources") or [])[:6] if item],
        "execution_note": _ledger_safe_text(str(profile.get("execution_note") or ""), 220),
        "hard_guardrails": [
            _ledger_safe_text(str(item or ""), 180)
            for item in (profile.get("exact_pretrade_hard_blockers") or [])[:4]
            if str(item or "").strip()
        ],
        "execution_advisories": [
            _ledger_safe_text(str(item or ""), 180)
            for item in (profile.get("exact_pretrade_advisories") or [])[:4]
            if str(item or "").strip()
        ],
    }


def _write_action_board_snapshot(
    *,
    session_intent: dict,
    market_now_profiles: list[dict],
    multi_vehicle_lanes: list[dict],
    best_direct: dict | None,
    best_cross: dict | None,
    best_usdjpy: dict | None,
) -> None:
    if not TRADER_LOCK_PATH.exists():
        return

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_intent": {
            "mode": str(session_intent.get("mode") or ""),
            "reasons": [str(item) for item in (session_intent.get("reasons") or [])[:6]],
        },
        "market_now": [
            item
            for item in (
                _action_snapshot_profile(profile)
                for profile in market_now_profiles[:MAX_DEPLOYMENT_LANES]
            )
            if item
        ],
        "multi_vehicle_lanes": [
            item
            for item in (
                _action_snapshot_profile(profile)
                for profile in multi_vehicle_lanes[:MAX_DEPLOYMENT_LANES]
            )
            if item
        ],
        "best_direct": _action_snapshot_profile(best_direct),
        "best_cross": _action_snapshot_profile(best_cross),
        "best_usdjpy": _action_snapshot_profile(best_usdjpy),
    }
    try:
        ACTION_BOARD_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    except Exception:
        pass


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


def run_trade_event_sync():
    """Sync broker-side TP/SL closes before the trader reads the live book."""
    try:
        r = subprocess.run(
            [VENV_PYTHON, "tools/trade_event_sync.py", "--notify-slack"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ROOT),
        )
    except Exception as exc:
        return f"TRADE_EVENT_SYNC_ERROR {exc}"
    out = r.stdout.strip()
    err = r.stderr.strip()
    if r.returncode != 0:
        return "\n".join(part for part in (out, f"TRADE_EVENT_SYNC_ERROR {err}") if part)
    return out


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


def _parse_jst_minute(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M JST")
    except Exception:
        return None


def _load_recent_regret_payload() -> dict:
    global _RECENT_REGRET_PAYLOAD
    if _RECENT_REGRET_PAYLOAD is None:
        try:
            from post_close_regret import build_regret_payload  # type: ignore
        except Exception:
            _RECENT_REGRET_PAYLOAD = {}
            return _RECENT_REGRET_PAYLOAD
        try:
            session_date_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            _RECENT_REGRET_PAYLOAD = build_regret_payload(
                session_date_from=session_date_from,
                hours=6,
            )
        except Exception:
            _RECENT_REGRET_PAYLOAD = {}
    return _RECENT_REGRET_PAYLOAD


def _summarize_regret_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    recovered = sum(1 for row in rows if row.get("recovered"))
    avg_loss = sum(float(row.get("loss_pips") or 0.0) for row in rows) / len(rows)
    avg_fav = sum(float(row.get("fav_pips") or 0.0) for row in rows) / len(rows)
    lags = []
    for row in rows:
        if not row.get("recovered"):
            continue
        close_time = _parse_jst_minute(row.get("close_time_jst"))
        recovered_time = _parse_jst_minute(row.get("recovered_at"))
        if close_time is None or recovered_time is None:
            continue
        lags.append((recovered_time - close_time).total_seconds() / 60.0)
    return {
        "count": len(rows),
        "recovered": recovered,
        "recovery_rate": recovered / len(rows) * 100.0,
        "avg_loss_pips": avg_loss,
        "avg_fav_pips": avg_fav,
        "median_lag_min": statistics.median(lags) if lags else None,
    }


def _recent_regret_for_pair(pair: str) -> dict | None:
    payload = _load_recent_regret_payload()
    rows = [row for row in payload.get("results", []) if row.get("pair") == pair]
    return _summarize_regret_rows(rows)


def _today_jst_date() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _today_session_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_audit_history_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    try:
        if text.endswith("UTC"):
            return datetime.strptime(text, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _open_memory_read_conn():
    memory_dir = ROOT / "collab_trade" / "memory"
    db_path = memory_dir / "memory.db"
    if str(memory_dir) not in sys.path:
        sys.path.insert(0, str(memory_dir))
    try:
        from schema import get_conn  # type: ignore

        return get_conn()
    except ModuleNotFoundError as exc:
        if exc.name != "apsw":
            raise
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("PRAGMA query_only=ON")
        return conn


def _load_recent_audit_pressure(window_hours: int = AUDIT_PRESSURE_LOOKBACK_HOURS) -> dict:
    global _RECENT_AUDIT_PRESSURE
    if _RECENT_AUDIT_PRESSURE is not None:
        return _RECENT_AUDIT_PRESSURE

    history_path = ROOT / "logs" / "audit_history.jsonl"
    payload = {
        "window_hours": window_hours,
        "repeat_trigger": AUDIT_REPEAT_TRIGGER_COUNT,
        "pairs": {},
        "hottest": None,
    }
    if not history_path.exists():
        _RECENT_AUDIT_PRESSURE = payload
        return payload

    now_jst = datetime.now(JST)
    cutoff = now_jst - timedelta(hours=window_hours)
    strongest_counts: Counter[tuple[str, str]] = Counter()
    narrative_counts: Counter[tuple[str, str]] = Counter()
    latest_reason: dict[tuple[str, str], str] = {}

    for line in history_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = _parse_audit_history_timestamp(entry.get("timestamp"))
        if timestamp is None:
            continue
        ts_jst = timestamp.astimezone(JST)
        if ts_jst < cutoff:
            continue

        strongest = entry.get("strongest_unheld") or {}
        pair = strongest.get("pair")
        direction = strongest.get("direction")
        if pair and direction:
            key = (pair, direction)
            strongest_counts[key] += 1
            why = _ledger_safe_text(strongest.get("why") or "", 120)
            if why:
                latest_reason[key] = why

        for pick in entry.get("narrative_picks") or []:
            pair = pick.get("pair")
            direction = pick.get("direction")
            edge = str(pick.get("edge") or "")
            if not pair or not direction or edge not in {"S", "A"}:
                continue
            key = (pair, direction)
            narrative_counts[key] += 1
            why = _ledger_safe_text(str(pick.get("why") or ""), 120)
            if why:
                latest_reason.setdefault(key, why)

    pairs: dict[tuple[str, str], dict] = {}
    for key in set(strongest_counts) | set(narrative_counts):
        total = int(strongest_counts.get(key, 0) + narrative_counts.get(key, 0))
        pairs[key] = {
            "count": total,
            "strongest_count": int(strongest_counts.get(key, 0)),
            "narrative_count": int(narrative_counts.get(key, 0)),
            "why": latest_reason.get(key),
            "window_hours": window_hours,
            "repeat_trigger": AUDIT_REPEAT_TRIGGER_COUNT,
        }

    hottest = None
    if pairs:
        hottest = max(
            pairs.items(),
            key=lambda item: (
                int(item[1].get("count", 0)),
                int(item[1].get("strongest_count", 0)),
                item[0][0],
                item[0][1],
            ),
        )[0]

    payload["pairs"] = pairs
    payload["hottest"] = hottest
    _RECENT_AUDIT_PRESSURE = payload
    return payload


def _load_recent_missed_seat_pressure(
    *,
    window_hours: int = MISSED_SEAT_LOOKBACK_HOURS,
    min_pips: float = MISSED_SEAT_MIN_PIPS,
) -> dict:
    global _RECENT_MISSED_SEAT_PRESSURE
    if _RECENT_MISSED_SEAT_PRESSURE is not None:
        return _RECENT_MISSED_SEAT_PRESSURE

    payload = {
        "window_hours": window_hours,
        "min_pips": min_pips,
        "pairs": {},
    }
    cutoff = (datetime.now(JST) - timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = _open_memory_read_conn()
    try:
        rows = conn.execute(
            """
            SELECT pair, direction,
                   COUNT(*) AS cnt,
                   MAX(ABS(COALESCE(pip_move, 0))) AS max_pip_move,
                   GROUP_CONCAT(horizon, ' | ') AS horizons
            FROM seat_outcomes
            WHERE source = 's_hunt'
              AND missed = 1
              AND ABS(COALESCE(pip_move, 0)) >= ?
              AND COALESCE(created_at, '') >= ?
            GROUP BY pair, direction
            """,
            (min_pips, cutoff),
        ).fetchall()
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()

    payload["pairs"] = {
        (row[0], row[1]): {
            "count": int(row[2] or 0),
            "max_pip_move": float(row[3] or 0.0),
            "horizons": str(row[4] or ""),
            "window_hours": window_hours,
        }
        for row in rows
        if row[0] and row[1]
    }
    _RECENT_MISSED_SEAT_PRESSURE = payload
    return payload


def _load_same_day_execution_style_context() -> dict:
    global _SAME_DAY_STYLE_CONTEXT
    if _SAME_DAY_STYLE_CONTEXT is not None:
        return _SAME_DAY_STYLE_CONTEXT

    today = _today_session_date()
    conn = _open_memory_read_conn()
    try:
        exact_rows = conn.execute(
            """
            SELECT pair, direction, execution_style,
                   COUNT(*) AS cnt,
                   AVG(pl) AS ev,
                   SUM(pl) AS total_pl,
                   SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
            FROM pretrade_outcomes
            WHERE pl IS NOT NULL
              AND COALESCE(execution_style, '') <> ''
              AND session_date = ?
            GROUP BY pair, direction, execution_style
            """,
            (today,),
        ).fetchall()
        global_rows = conn.execute(
            """
            SELECT execution_style,
                   COUNT(*) AS cnt,
                   AVG(pl) AS ev,
                   SUM(pl) AS total_pl,
                   SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
            FROM pretrade_outcomes
            WHERE pl IS NOT NULL
              AND COALESCE(execution_style, '') <> ''
              AND session_date = ?
            GROUP BY execution_style
            """,
            (today,),
        ).fetchall()
        seat_rows = conn.execute(
            """
            SELECT pair, direction, SUM(captured) AS captured
            FROM seat_outcomes
            WHERE source = 's_hunt'
              AND session_date = ?
            GROUP BY pair, direction
            """,
            (today,),
        ).fetchall()
        global_captured = conn.execute(
            """
            SELECT COALESCE(SUM(captured), 0)
            FROM seat_outcomes
            WHERE source = 's_hunt'
              AND session_date = ?
            """,
            (today,),
        ).fetchone()
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()

    captured_by_pair = {
        (row[0], row[1]): int(row[2] or 0)
        for row in seat_rows
        if row[0] and row[1]
    }
    exact = {}
    for row in exact_rows:
        count = int(row[3] or 0)
        wins = int(row[6] or 0)
        exact[(row[0], row[1], row[2])] = {
            "count": count,
            "ev": float(row[4] or 0.0),
            "total_pl": float(row[5] or 0.0),
            "wins": wins,
            "win_rate": (wins / count) if count else 0.0,
            "captured": captured_by_pair.get((row[0], row[1]), 0),
            "label": f"{today} UTC session",
        }

    global_style = {}
    total_captured = int((global_captured[0] if global_captured else 0) or 0)
    for row in global_rows:
        count = int(row[1] or 0)
        wins = int(row[4] or 0)
        global_style[row[0]] = {
            "count": count,
            "ev": float(row[2] or 0.0),
            "total_pl": float(row[3] or 0.0),
            "wins": wins,
            "win_rate": (wins / count) if count else 0.0,
            "captured": total_captured,
            "label": f"{today} UTC session",
        }

    _SAME_DAY_STYLE_CONTEXT = {
        "today": today,
        "exact": exact,
        "global": global_style,
    }
    return _SAME_DAY_STYLE_CONTEXT


def _format_day_style_guard(style: str, stat: dict, scope: str) -> str:
    return (
        f"{style} {scope} {stat.get('label', 'today')} "
        f"(n={int(stat.get('count', 0) or 0)}, "
        f"EV {float(stat.get('ev', 0.0) or 0.0):+.0f}, "
        f"total {float(stat.get('total_pl', 0.0) or 0.0):+.0f}, "
        f"captured S {int(stat.get('captured', 0) or 0)})"
    )


def _same_day_style_kill_stat(pair: str | None, direction: str | None, style: str | None) -> dict | None:
    style_key = str(style or "").upper()
    if style_key not in {"MARKET", "STOP-ENTRY"}:
        return None

    context = _load_same_day_execution_style_context()
    exact = context.get("exact", {}).get((pair, direction, style_key))
    if exact and int(exact.get("count", 0) or 0) >= 2:
        if float(exact.get("total_pl", 0.0) or 0.0) < 0 and int(exact.get("captured", 0) or 0) == 0:
            return {"scope": "exact lane", **exact}

    global_style = context.get("global", {}).get(style_key)
    if global_style and int(global_style.get("count", 0) or 0) >= 2:
        if float(global_style.get("total_pl", 0.0) or 0.0) < 0 and int(global_style.get("captured", 0) or 0) == 0:
            return {"scope": "global lane", **global_style}
    return None


def _units_to_direction(units) -> str | None:
    try:
        return "LONG" if float(units) > 0 else "SHORT"
    except Exception:
        return None


def _pip_factor(pair: str) -> int:
    return 100 if str(pair).endswith("JPY") else 10000


def _parse_oanda_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    match = re.match(r"^(.*?\.\d{6})\d+([+-]\d\d:\d\d)$", text)
    if match:
        text = f"{match.group(1)}{match.group(2)}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _pending_order_metrics(
    order: dict,
    price_map: dict[str, dict[str, float]],
    spread_data: dict[str, float],
    now_utc: datetime,
) -> dict:
    pair = str(order.get("instrument", ""))
    market = price_map.get(pair) or {}
    direction = _units_to_direction(order.get("units")) or "?"

    try:
        order_price = float(order.get("price"))
    except Exception:
        order_price = None

    mid = market.get("mid")
    gap_pips = None
    relation = None
    if order_price is not None and mid is not None:
        gap_pips = abs(mid - order_price) * _pip_factor(pair)
        relation = "below live" if order_price < mid else "above live"

    create_dt = _parse_oanda_time(order.get("createTime"))
    gtd_dt = _parse_oanda_time(order.get("gtdTime"))
    age_min = (now_utc - create_dt).total_seconds() / 60 if create_dt else None
    ttl_min = (gtd_dt - now_utc).total_seconds() / 60 if gtd_dt else None

    tech = _load_technicals(ROOT, pair) if pair else {}
    m5 = tech.get("M5", {}) if tech else {}
    atr_pips = float(m5.get("atr_pips", 0) or 0) or None
    spread_pips = spread_data.get(pair)
    wish_threshold = None
    if atr_pips is not None or spread_pips is not None:
        wish_threshold = max(atr_pips or 0.0, (spread_pips or 0.0) * 4)

    flags = []
    if gap_pips is not None and wish_threshold and gap_pips > wish_threshold:
        flags.append("wish-distance")
    if ttl_min is not None and ttl_min <= 45:
        flags.append("expiring-soon")
    if age_min is not None and age_min >= 60:
        flags.append("aged")

    return {
        "pair": pair,
        "direction": direction,
        "gap_pips": gap_pips,
        "gap_relation": relation,
        "atr_pips": atr_pips,
        "spread_pips": spread_pips,
        "wish_threshold": wish_threshold,
        "age_min": age_min,
        "ttl_min": ttl_min,
        "gtd_dt": gtd_dt,
        "status": ",".join(flags) if flags else "fresh",
    }


def _format_expiry_utc(dt: datetime | None) -> str:
    if not dt:
        return ""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M UTC")


def _profile_default_expression(profile: dict | None) -> str:
    if not profile:
        return "PASS"
    return str(profile.get("default_expression") or profile.get("execution_style") or "PASS").upper()


def _profile_default_orderability(profile: dict | None) -> str:
    if not profile:
        return "PASS"
    explicit = str(profile.get("default_orderability") or profile.get("orderability") or "").upper()
    if explicit:
        return explicit
    style = _profile_default_expression(profile)
    return "ENTER NOW" if style == "MARKET" else style


def _lane_shelf_life(profile: dict, now_utc: datetime) -> dict:
    style = _profile_default_expression(profile)
    source = str(profile.get("source") or "").lower()
    regime = str(profile.get("current_regime") or "").lower()
    pending_metrics = profile.get("pending_metrics") or {}

    ttl_min: float | None = None
    expires_at: datetime | None = None
    carry_rule = ""

    if source == "held":
        ttl_min = float(SESSION_CADENCE_MIN)
        expires_at = now_utc + timedelta(minutes=ttl_min)
        carry_rule = "live position already owns risk; the next session must re-underwrite hold / trim / exit instead of rediscovering it as a fresh lane"
    elif source == "pending":
        ttl_min = pending_metrics.get("ttl_min")
        gtd_dt = pending_metrics.get("gtd_dt")
        expires_at = gtd_dt if isinstance(gtd_dt, datetime) else None
        if ttl_min is None:
            carry_rule = "armed order without GTD; keep only while the structural level is still clean, otherwise cancel explicitly"
        else:
            carry_rule = "carry the armed order until GTD; next session must choose LEAVE / EXTEND GTD / CANCEL explicitly, not rediscover it as fresh prose"
    elif style == "MARKET":
        ttl_min = float(SESSION_CADENCE_MIN)
        expires_at = now_utc + timedelta(minutes=ttl_min)
        carry_rule = "this is payable now only; if untouched by the next 20-minute session, rewrite from fresh tape instead of carrying it blindly"
    elif style == "STOP-ENTRY":
        ttl_min = float(SESSION_CADENCE_MIN * 2)
        expires_at = now_utc + timedelta(minutes=ttl_min)
        carry_rule = "carry across the next 2 sessions only while the trigger stays untraded; if price already printed and failed, kill or rewrite it"
    elif style == "LIMIT":
        ttl_min = float(SESSION_CADENCE_MIN * (6 if regime in {"range", "quiet"} else 4))
        expires_at = now_utc + timedelta(minutes=ttl_min)
        carry_rule = "carry across the next sessions only while the structural price-improvement level is still intact; refresh or kill it once that shelf expires"
    else:
        ttl_min = float(SESSION_CADENCE_MIN)
        expires_at = now_utc + timedelta(minutes=ttl_min)
        carry_rule = "not a live lane; only re-open next session if the same pressure and tape survive a fresh read"

    sessions = None
    if ttl_min is not None and ttl_min > 0:
        sessions = max(1, math.ceil(ttl_min / SESSION_CADENCE_MIN))

    if source == "held":
        label = f"re-underwrite by next session ({SESSION_CADENCE_MIN}m) until {_format_expiry_utc(expires_at)}"
    elif ttl_min is None:
        label = "GTC / until explicit cancel"
    elif sessions == 1:
        label = f"{ttl_min:.0f}m (~1 session) until {_format_expiry_utc(expires_at)}"
    else:
        label = f"{ttl_min:.0f}m (~{sessions} sessions) until {_format_expiry_utc(expires_at)}"

    return {
        "shelf_life_min": ttl_min,
        "shelf_life_sessions": sessions,
        "expires_at_utc": _format_expiry_utc(expires_at),
        "shelf_life_label": label,
        "carry_rule": carry_rule,
    }


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
        stripped = line.strip()
        match = re.match(r"🎯\s+(\w+_\w+)\s+(LONG|SHORT)\s+(.+)$", stripped)
        if not match:
            continue
        candidates.append({
            "pair": match.group(1),
            "direction": match.group(2),
            "recipe": _clip_text(match.group(3).strip(), 180),
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
    seen: set[str] = set()
    collapsed_focus_targets: dict[tuple[str, str, str], dict] = {}

    origin_priority = {
        "primary vehicle": 7,
        "best expression": 6,
        "best direct-USD seat": 5,
        "next fresh risk": 4,
        "backup vehicle": 3,
        "second-best expression": 2,
        "rotation candidate": 1,
    }

    def state_focus_bucket(line: str) -> str:
        upper = str(line or "").upper()
        if "LIMIT" in upper:
            return "LIMIT"
        if "STOP-ENTRY" in upper or "STOP ENTRY" in upper or "TRIGGER-ONLY" in upper or "TRIGGER ONLY" in upper:
            return "STOP-ENTRY"
        return "ACTIVE"

    def add_target(line: str, origin: str) -> None:
        pair, direction = _extract_pair_direction(line)
        if not pair or not direction:
            return
        lowered = line.lower()
        if "dead thesis because" in lowered or lowered.startswith("none because"):
            return
        if "id=board" in lowered:
            # `id=board` is only a handoff echo of the previous ranked board, not a live receipt.
            # Carry-forward memory should preserve unresolved real risk, not re-inject old board prose.
            return
        if re.search(r"\bid=\s*`?\d+", line, flags=re.IGNORECASE):
            # Live receipts and armed orders already enter via OANDA held/pending targets.
            # Keep state carry for unresolved / contradictory seats, not duplicate receipt echoes.
            return

        if origin != "s-hunt horizon":
            bucket = state_focus_bucket(line)
            group_key = (pair, direction, bucket)
            body = re.sub(r"^[^:]+:\s*", "", line).strip()
            entry = collapsed_focus_targets.get(group_key)
            if entry is None:
                collapsed_focus_targets[group_key] = {
                    "pair": pair,
                    "direction": direction,
                    "bucket": bucket,
                    "origins": [origin],
                    "representative_line": line,
                    "representative_body": body,
                    "priority": origin_priority.get(origin, 0),
                }
                return

            if origin not in entry["origins"]:
                entry["origins"].append(origin)
            current_priority = int(entry.get("priority", 0) or 0)
            new_priority = origin_priority.get(origin, 0)
            current_line = str(entry.get("representative_line") or "")
            if (
                new_priority > current_priority
                or (new_priority == current_priority and len(line) > len(current_line))
                or (bucket == "ACTIVE" and "ENTER NOW" in line.upper() and "ENTER NOW" not in current_line.upper())
            ):
                entry["representative_line"] = line
                entry["representative_body"] = body
                entry["priority"] = new_priority
            return

        recipe = _clip_text(f"{origin}: {line}", 140)
        key = _normalize_identity_fragment(f"{pair}|{direction}|{origin}|{line}", limit=160)
        if key in seen:
            return
        targets.append({
            "pair": pair,
            "direction": direction,
            "source": "state",
            "recipe": recipe,
            "seat_family": origin,
            "seat_key": _build_seat_key(
                pair,
                direction,
                "state",
                recipe,
                seat_family=origin,
            ),
        })
        seen.add(key)

    prefixes = {
        "Primary vehicle:": "primary vehicle",
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

        if re.match(r"^(Short-term|Medium-term|Long-term) S", line):
            add_target(line, "s-hunt horizon")

    for entry in collapsed_focus_targets.values():
        origins = list(entry.get("origins") or [])
        representative_body = str(entry.get("representative_body") or "").strip()
        if representative_body.endswith("."):
            representative_body = representative_body[:-1].strip()
        origin_label = " / ".join(origins)
        recipe = _clip_text(f"state focus ({origin_label}): {representative_body}", 140)
        seat_family = f"state-focus-{str(entry.get('bucket') or 'active').lower()}"
        key = _normalize_identity_fragment(
            f"{entry['pair']}|{entry['direction']}|{seat_family}|{representative_body}",
            limit=160,
        )
        if key in seen:
            continue
        targets.append({
            "pair": entry["pair"],
            "direction": entry["direction"],
            "source": "state",
            "recipe": recipe,
            "seat_family": seat_family,
            "seat_key": _build_seat_key(
                entry["pair"],
                entry["direction"],
                "state",
                recipe,
                seat_family=seat_family,
            ),
        })
        seen.add(key)

    return targets


def _parse_state_focus_snapshot(state_text: str) -> dict[str, str]:
    if not state_text:
        return {}

    prefixes = (
        "Best expression NOW:",
        "Primary vehicle:",
        "Primary vehicle shelf-life now:",
        "Backup vehicle:",
        "Backup vehicle shelf-life now:",
        "Second-best expression:",
        "Next fresh risk allowed NOW:",
        "Next fresh risk shelf-life now:",
        "20-minute backup trigger armed NOW:",
        "15-minute backup trigger armed NOW:",
        "Best direct-USD seat NOW:",
    )
    snapshot: dict[str, str] = {}

    for raw_line in state_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for prefix in prefixes:
            if line.startswith(prefix) and prefix not in snapshot:
                snapshot[prefix] = line[len(prefix):].strip()
                break

    return snapshot


def _build_memory_targets(
    trades_data: dict,
    pending_orders: list[dict],
    scanner_output: str,
    price_map: dict[str, dict[str, float]],
    spread_data: dict[str, float],
    now_utc: datetime,
    carry_targets: list[dict] | None = None,
    audit_range_targets: list[dict] | None = None,
    audit_targets: list[dict] | None = None,
) -> list[dict]:
    targets = []
    seen = set()

    def add_target(
        pair: str | None,
        direction: str | None,
        source: str,
        recipe: str | None = None,
        **extra: object,
    ):
        if not pair or not direction:
            return
        target = {
            "pair": pair,
            "direction": direction,
            "source": source,
            "recipe": recipe or "",
        }
        target.update(extra)
        target["seat_family"] = target.get("seat_family") or _seat_family_from_target(target)
        identity_extra = {
            key: value
            for key, value in target.items()
            if key not in {"pair", "direction", "source", "recipe", "seat_key"}
        }
        target["seat_key"] = target.get("seat_key") or _build_seat_key(
            pair,
            direction,
            source,
            recipe,
            **identity_extra,
        )
        key = target["seat_key"]
        if key in seen:
            return
        targets.append(target)
        seen.add(key)

    def pending_style(order: dict) -> str:
        order_type = str(order.get("type") or "").upper()
        if order_type == "LIMIT":
            return "LIMIT"
        if order_type in {"STOP", "MARKET_IF_TOUCHED"}:
            return "STOP-ENTRY"
        return "LIMIT"

    for trade in trades_data.get("trades", []):
        add_target(
            trade.get("instrument"),
            _units_to_direction(trade.get("currentUnits")),
            "held",
            trade_id=str(trade.get("id") or ""),
        )

    for order in pending_orders:
        pending_metrics = _pending_order_metrics(order, price_map, spread_data, now_utc)
        add_target(
            order.get("instrument"),
            _units_to_direction(order.get("units")),
            "pending",
            armed_style=pending_style(order),
            armed_id=str(order.get("id") or ""),
            pending_metrics=pending_metrics,
        )

    for candidate in audit_range_targets or []:
        extra = {
            key: value
            for key, value in candidate.items()
            if key not in {"pair", "direction", "source", "recipe"}
        }
        add_target(
            candidate.get("pair"),
            candidate.get("direction"),
            "audit_range",
            candidate.get("recipe"),
            **extra,
        )

    for candidate in audit_targets or []:
        extra = {
            key: value
            for key, value in candidate.items()
            if key not in {"pair", "direction", "source", "recipe"}
        }
        add_target(
            candidate.get("pair"),
            candidate.get("direction"),
            "audit",
            candidate.get("recipe"),
            **extra,
        )

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
        if target.get("armed_style"):
            parts.append(str(target.get("armed_style")))
    elif target["source"] == "audit_range":
        parts.extend(["audit", "range", "rotation", "opposite band", "limit", "lesson", "failure", "success"])
    elif target["source"] == "audit":
        parts.extend(["audit", "strongest-unheld", "missed", "setup", "lesson", "failure", "success"])
    elif target["source"] == "state":
        parts.extend(["trigger", "backup", "rotation", "retest", "lesson"])
    else:
        parts.extend(["setup", "lesson", "failure", "success"])
    if target.get("recipe"):
        parts.append(target["recipe"])
    return " ".join(parts)


def _build_audit_targets(
    audit_context: dict,
    missed_pressure: dict,
    *,
    limit: int = AUDIT_MEMORY_TARGET_LIMIT,
) -> list[dict]:
    pair_map = audit_context.get("pairs") or {}
    candidates = audit_context.get("candidates") or []
    if not pair_map and not candidates:
        return []

    recent_pressure_pairs = (audit_context.get("recent_pressure") or {}).get("pairs", {})
    narrative_details = audit_context.get("narrative_details") or {}
    hottest = audit_context.get("strongest") or (audit_context.get("recent_pressure") or {}).get("hottest")
    missed_pairs = missed_pressure.get("pairs") or {}
    ranked: list[dict] = []

    def is_pair_shared_label(label: str) -> bool:
        lower = str(label or "").strip().lower()
        return lower.startswith("audit repeated ")

    def add_ranked_target(
        pair: str,
        direction: str,
        labels: list[str],
        detail: dict,
        *,
        seat_family: str | None = None,
    ) -> None:
        if not pair or not direction:
            return
        stats = recent_pressure_pairs.get((pair, direction)) or {}
        missed_stats = missed_pairs.get((pair, direction)) or {}
        score = 0
        recipe_bits: list[str] = []

        if (pair, direction) == hottest:
            score += 30
            recipe_bits.append("audit hottest")

        for label in labels:
            text = str(label or "").strip()
            if not text or text in recipe_bits:
                continue
            recipe_bits.append(text)
            lower = text.lower()
            if "strongest-unheld" in lower:
                score += 24
            elif "inventory lead" in lower:
                score += 10
            elif "edge s" in lower:
                score += 18
            elif "edge a" in lower:
                score += 12
            elif "edge b" in lower:
                score += 7
            elif "audit repeated" in lower:
                score += 16

        edge = str(detail.get("edge") or detail.get("audit_edge") or "").upper()
        allocation = str(detail.get("allocation") or detail.get("audit_allocation") or edge).upper()
        if edge == "S":
            score += 12
        elif edge == "A":
            score += 8
        elif edge == "B":
            score += 4
        if allocation == "A":
            score += 4
        elif allocation == "B":
            score += 2

        count = int(stats.get("count", 0) or 0)
        strongest_count = int(stats.get("strongest_count", 0) or 0)
        if count:
            score += min(22, count * 4 + strongest_count * 2)
            repeat_hours = int(
                stats.get("window_hours", AUDIT_PRESSURE_LOOKBACK_HOURS) or AUDIT_PRESSURE_LOOKBACK_HOURS
            )
            repeat_bit = (
                f"repeat {count}x/"
                f"{repeat_hours}h"
            )
            repeat_signature = f"{count}x/{repeat_hours}h"
            if not any("repeat" in bit.lower() and repeat_signature in bit for bit in recipe_bits):
                recipe_bits.append(repeat_bit)
        why = _ledger_safe_text(str(stats.get("why") or "").strip(), 80)
        if why and why not in recipe_bits:
            recipe_bits.append(why)
        detail_why = _ledger_safe_text(
            str(detail.get("why") or detail.get("audit_why") or "").strip(),
            90,
        )
        if detail_why and detail_why not in recipe_bits:
            recipe_bits.append(detail_why)
        entry_price = detail.get("entry_price")
        if entry_price is None:
            entry_price = detail.get("audit_entry_price")
        tp_price = detail.get("tp_price")
        if tp_price is None:
            tp_price = detail.get("audit_tp_price")
        if entry_price is not None and tp_price is not None:
            recipe_bits.append(f"@{entry_price} -> {tp_price}")

        missed_count = int(missed_stats.get("count", 0) or 0)
        max_pip_move = float(missed_stats.get("max_pip_move", 0.0) or 0.0)
        if missed_count > 0 and max_pip_move > 0:
            score += min(20, 6 + int(max_pip_move // 3))
            recipe_bits.append(f"missed {missed_count}x/{max_pip_move:.1f}pip")

        ranked.append(
            {
                "pair": pair,
                "direction": direction,
                "score": score,
                "missed_pips": max_pip_move,
                "recipe": _ledger_safe_text(" | ".join(recipe_bits[:4]), 120),
                "seat_family": seat_family
                or str(detail.get("seat_family") or "")
                or _normalize_identity_fragment(
                    f"audit-{pair}-{direction}-{edge or 'seat'}-{entry_price or 'na'}-{tp_price or 'na'}-{detail_why or '-'.join(labels[:2])}",
                    limit=80,
                ),
                "audit_entry_price": entry_price,
                "audit_tp_price": tp_price,
                "audit_why": detail_why,
                "audit_edge": edge or None,
                "audit_allocation": allocation or None,
                "audit_held_status": detail.get("held_status") or detail.get("audit_held_status"),
                "audit_opposite_entry_price": detail.get("opposite_entry_price")
                or detail.get("audit_opposite_entry_price"),
            }
        )

    covered_pair_directions: set[tuple[str, str]] = set()

    for candidate in candidates:
        pair = str(candidate.get("pair") or "").strip()
        direction = str(candidate.get("direction") or "").strip().upper()
        if not pair or direction not in {"LONG", "SHORT"}:
            continue
        labels: list[str] = []
        for label in candidate.get("labels") or []:
            text = str(label or "").strip()
            if text and text not in labels:
                labels.append(text)
        for label in pair_map.get((pair, direction), []):
            text = str(label or "").strip()
            if text and is_pair_shared_label(text) and text not in labels:
                labels.append(text)
        add_ranked_target(
            pair,
            direction,
            labels,
            candidate,
            seat_family=str(candidate.get("seat_family") or "").strip() or None,
        )
        covered_pair_directions.add((pair, direction))

    for (pair, direction), labels in pair_map.items():
        if candidates and (pair, direction) in covered_pair_directions:
            continue
        detail = narrative_details.get((pair, direction)) or {}
        add_ranked_target(pair, direction, labels, detail)

    ranked.sort(
        key=lambda item: (
            int(item.get("score", 0) or 0),
            float(item.get("missed_pips", 0.0) or 0.0),
            item.get("pair", ""),
            item.get("direction", ""),
            item.get("seat_family", ""),
        ),
        reverse=True,
    )
    return [
        {
            "pair": item["pair"],
            "direction": item["direction"],
            "source": "audit",
            "recipe": item["recipe"],
            "seat_family": item.get("seat_family"),
            "audit_entry_price": item.get("audit_entry_price"),
            "audit_tp_price": item.get("audit_tp_price"),
            "audit_why": item.get("audit_why"),
            "audit_edge": item.get("audit_edge"),
            "audit_allocation": item.get("audit_allocation"),
            "audit_held_status": item.get("audit_held_status"),
            "audit_opposite_entry_price": item.get("audit_opposite_entry_price"),
        }
        for item in ranked[:limit]
    ]


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


def _normalize_identity_fragment(value: object, *, limit: int = 64) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    if not compact:
        return "na"
    return compact[:limit]


def _seat_family_from_target(target: dict) -> str:
    source = str(target.get("source") or "seat")
    if target.get("seat_family"):
        return _normalize_identity_fragment(target.get("seat_family"))
    if source == "held" and target.get("trade_id"):
        return f"live-{_normalize_identity_fragment(target.get('trade_id'))}"
    if source == "pending" and target.get("armed_id"):
        armed_style = _normalize_identity_fragment(target.get("armed_style") or "pending")
        return f"{armed_style}-{_normalize_identity_fragment(target.get('armed_id'))}"
    if source == "audit_range":
        side = _normalize_identity_fragment(target.get("range_side") or "range")
        entry = _normalize_identity_fragment(target.get("range_entry_price") or "entry")
        return f"{side}-{entry}"
    recipe = str(target.get("recipe") or "").strip()
    if recipe:
        return _normalize_identity_fragment(recipe, limit=80)
    return source


def _build_seat_key(
    pair: str | None,
    direction: str | None,
    source: str,
    recipe: str | None = None,
    **extra: object,
) -> str:
    return "|".join(
        (
            str(pair or ""),
            str(direction or ""),
            str(source or ""),
            _seat_family_from_target(
                {
                    "source": source,
                    "recipe": recipe or "",
                    **extra,
                }
            ),
        )
    )


def _today_entry_activity(now_utc: datetime) -> dict:
    log_path = ROOT / "logs" / "live_trade_log.txt"
    counts = Counter()
    pair_counts = Counter()
    if not log_path.exists():
        return {"counts": counts, "pair_counts": pair_counts}

    day_prefix = f"[{now_utc.strftime('%Y-%m-%d')} "
    for raw in log_path.read_text().splitlines():
        if not raw.startswith(day_prefix):
            continue
        pair = next((candidate for candidate in PAIRS if candidate in raw), None)
        if " ENTRY_ORDER " in raw:
            counts["entry_orders"] += 1
            if pair:
                pair_counts[pair] += 1
        elif "] ENTRY " in raw:
            counts["entries"] += 1
            if pair:
                pair_counts[pair] += 1
        elif " ORDER_REJECT " in raw:
            counts["rejects"] += 1
        elif " CANCEL_ORDER " in raw:
            counts["cancels"] += 1
        elif "] CLOSE " in raw:
            counts["closes"] += 1
    return {"counts": counts, "pair_counts": pair_counts}


def _entry_activity_summary(activity: dict, *, include_pair: bool = False) -> str:
    counts = activity.get("counts") or {}
    pair_counts = activity.get("pair_counts") or {}
    summary = (
        f"{int(counts.get('entries', 0))} fills / "
        f"{int(counts.get('entry_orders', 0))} new entry orders / "
        f"{int(counts.get('rejects', 0))} rejects"
    )
    if include_pair and pair_counts:
        top_pair, top_count = pair_counts.most_common(1)[0]
        summary = f"{summary}. Most active: {top_pair} ×{top_count}"
    return summary


def _load_audit_narrative_context() -> dict:
    audit_path = ROOT / "logs" / "quality_audit.md"
    recent_pressure = _load_recent_audit_pressure()
    if not audit_path.exists():
        return {
            "available": False,
            "stale": False,
            "age_min": None,
            "pairs": {},
            "candidates": [],
            "strongest": None,
            "recent_pressure": recent_pressure,
            "range_targets": [],
            "narrative_details": {},
        }

    age_min = (time.time() - audit_path.stat().st_mtime) / 60
    stale = age_min > QUALITY_AUDIT_STALE_MIN
    if stale:
        pair_map: dict[tuple[str, str], list[str]] = {}
        for key, stats in recent_pressure.get("pairs", {}).items():
            label = (
                f"audit repeated {int(stats.get('count', 0))}x/"
                f"{int(stats.get('window_hours', AUDIT_PRESSURE_LOOKBACK_HOURS))}h"
            )
            pair_map.setdefault(key, []).append(label)
        return {
            "available": True,
            "stale": True,
            "age_min": age_min,
            "pairs": pair_map,
            "candidates": [],
            "strongest": recent_pressure.get("hottest"),
            "recent_pressure": recent_pressure,
            "range_targets": [],
            "narrative_details": {},
        }

    try:
        from record_audit_narrative import build_entry as build_audit_narrative_entry
        entry = build_audit_narrative_entry(audit_path.read_text())
    except Exception:
        pair_map: dict[tuple[str, str], list[str]] = {}
        for key, stats in recent_pressure.get("pairs", {}).items():
            label = (
                f"audit repeated {int(stats.get('count', 0))}x/"
                f"{int(stats.get('window_hours', AUDIT_PRESSURE_LOOKBACK_HOURS))}h"
            )
            pair_map.setdefault(key, []).append(label)
        return {
            "available": True,
            "stale": False,
            "age_min": age_min,
            "pairs": pair_map,
            "candidates": [],
            "strongest": recent_pressure.get("hottest"),
            "recent_pressure": recent_pressure,
            "range_targets": [],
            "narrative_details": {},
        }

    pair_map: dict[tuple[str, str], list[str]] = {}
    strongest = None
    range_targets: list[dict] = []
    narrative_details: dict[tuple[str, str], dict] = {}
    narrative_candidates: list[dict] = []
    narrative_candidate_index: dict[tuple[object, ...], int] = {}
    edge_rank = {"S": 4, "A": 3, "B": 2, "C": 1}

    def candidate_rank(payload: dict) -> tuple[int, int, float]:
        return (
            edge_rank.get(str(payload.get("edge") or "").upper(), 0),
            edge_rank.get(str(payload.get("allocation") or "").upper(), 0),
            float(payload.get("entry_price") or 0.0),
        )

    def register_narrative_candidate(label: str, candidate: dict) -> None:
        pair = str(candidate.get("pair") or "").strip()
        direction = str(candidate.get("direction") or "").strip().upper()
        if not pair or direction not in {"LONG", "SHORT"}:
            return
        payload = dict(candidate)
        payload["pair"] = pair
        payload["direction"] = direction
        key = (pair, direction)
        if label not in pair_map.setdefault(key, []):
            pair_map[key].append(label)
        current = narrative_details.get(key) or {}
        if not current or candidate_rank(payload) > candidate_rank(current):
            narrative_details[key] = payload

        signature = (
            pair,
            direction,
            str(payload.get("edge") or "").upper(),
            str(payload.get("allocation") or "").upper(),
            float(payload.get("entry_price") or 0.0),
            float(payload.get("tp_price") or 0.0),
            _normalize_identity_fragment(payload.get("why") or "", limit=96),
        )
        existing_index = narrative_candidate_index.get(signature)
        if existing_index is not None:
            existing = narrative_candidates[existing_index]
            labels = existing.setdefault("labels", [])
            if label not in labels:
                labels.append(label)
            for field in (
                "edge",
                "allocation",
                "entry_price",
                "tp_price",
                "why",
                "held_status",
            ):
                if existing.get(field) in {None, ""} and payload.get(field) not in {None, ""}:
                    existing[field] = payload.get(field)
            return

        seat_family = str(payload.get("seat_family") or "").strip() or _normalize_identity_fragment(
            f"audit-{pair}-{direction}-{label}-{payload.get('edge') or 'seat'}-{payload.get('entry_price') or 'na'}-{payload.get('tp_price') or 'na'}-{payload.get('why') or ''}",
            limit=80,
        )
        narrative_candidate_index[signature] = len(narrative_candidates)
        narrative_candidates.append(
            {
                **payload,
                "seat_family": seat_family,
                "labels": [label],
            }
        )

    strongest_raw = entry.get("strongest_unheld") or {}
    if strongest_raw.get("pair") and strongest_raw.get("direction"):
        strongest = (strongest_raw["pair"], strongest_raw["direction"])

    for pick in entry.get("narrative_picks") or []:
        pair = pick.get("pair")
        direction = pick.get("direction")
        edge = str(pick.get("edge") or "").upper()
        allocation = str(pick.get("allocation") or edge).upper()
        if not pair or not direction or edge not in {"S", "A", "B"}:
            continue
        register_narrative_candidate(
            f"audit narrative Edge {edge}",
            {
                "pair": pair,
                "direction": direction,
                "edge": edge,
                "allocation": allocation,
                "entry_price": pick.get("entry_price"),
                "tp_price": pick.get("tp_price"),
                "why": _ledger_safe_text(str(pick.get("why") or ""), 120),
                "held_status": pick.get("held_status"),
            },
        )

    inventory_lead = entry.get("inventory_lead") or {}
    if inventory_lead.get("pair") and inventory_lead.get("direction"):
        edge = str(inventory_lead.get("edge") or "").upper()
        allocation = str(inventory_lead.get("allocation") or edge).upper()
        if edge in {"S", "A", "B"}:
            register_narrative_candidate(
                "audit inventory lead",
                {
                    "pair": inventory_lead.get("pair"),
                    "direction": inventory_lead.get("direction"),
                    "edge": edge,
                    "allocation": allocation,
                    "entry_price": inventory_lead.get("entry_price"),
                    "tp_price": inventory_lead.get("tp_price"),
                    "why": _ledger_safe_text(str(inventory_lead.get("why") or ""), 120),
                    "held_status": inventory_lead.get("held_status"),
                },
            )

    if strongest:
        strongest_label = "audit strongest-unheld"
        if strongest_label not in pair_map.setdefault(strongest, []):
            pair_map[strongest].append(strongest_label)
        matching_indexes = [
            idx
            for idx, candidate in enumerate(narrative_candidates)
            if (candidate.get("pair"), candidate.get("direction")) == strongest
        ]
        if matching_indexes:
            best_index = max(matching_indexes, key=lambda idx: candidate_rank(narrative_candidates[idx]))
            labels = narrative_candidates[best_index].setdefault("labels", [])
            if strongest_label not in labels:
                labels.append(strongest_label)
        else:
            register_narrative_candidate(
                strongest_label,
                {
                    "pair": strongest_raw.get("pair"),
                    "direction": strongest_raw.get("direction"),
                    "edge": str(strongest_raw.get("edge") or "").upper(),
                    "allocation": str(strongest_raw.get("allocation") or strongest_raw.get("edge") or "").upper(),
                    "entry_price": strongest_raw.get("entry_price"),
                    "tp_price": strongest_raw.get("tp_price"),
                    "why": _ledger_safe_text(str(strongest_raw.get("why") or ""), 120),
                    "held_status": strongest_raw.get("held_status"),
                },
            )

    if strongest and strongest not in narrative_details:
        narrative_details[strongest] = {
            "pair": strongest[0],
            "direction": strongest[1],
            "edge": str(strongest_raw.get("edge") or "").upper(),
            "allocation": str(strongest_raw.get("allocation") or strongest_raw.get("edge") or "").upper(),
            "entry_price": strongest_raw.get("entry_price"),
            "tp_price": strongest_raw.get("tp_price"),
            "why": _ledger_safe_text(str(strongest_raw.get("why") or ""), 120),
            "held_status": strongest_raw.get("held_status"),
            "opposite_entry_price": None,
        }

    opposite_by_pair: dict[str, dict[str, float]] = {}
    for (pair, direction), detail in narrative_details.items():
        entry_price = detail.get("entry_price")
        if entry_price is None:
            continue
        opposite_by_pair.setdefault(str(pair), {})[str(direction)] = float(entry_price)
    for (pair, direction), detail in list(narrative_details.items()):
        opposite_direction = "SHORT" if str(direction) == "LONG" else "LONG"
        detail["opposite_entry_price"] = opposite_by_pair.get(str(pair), {}).get(opposite_direction)
    for candidate in narrative_candidates:
        pair = str(candidate.get("pair") or "")
        direction = str(candidate.get("direction") or "")
        opposite_direction = "SHORT" if direction == "LONG" else "LONG"
        candidate["opposite_entry_price"] = opposite_by_pair.get(pair, {}).get(opposite_direction)

    for key, stats in recent_pressure.get("pairs", {}).items():
        label = (
            f"audit repeated {int(stats.get('count', 0))}x/"
            f"{int(stats.get('window_hours', AUDIT_PRESSURE_LOOKBACK_HOURS))}h"
        )
        if label not in pair_map.setdefault(key, []):
            pair_map[key].append(label)

    for item in entry.get("range_opportunities") or []:
        pair = str(item.get("pair") or "").strip()
        direction = str(item.get("direction") or "").strip().upper()
        if not pair or direction not in {"LONG", "SHORT"}:
            continue
        entry_price = item.get("entry_price")
        tp_price = item.get("tp_price")
        if entry_price is None or tp_price is None:
            continue
        note = _ledger_safe_text(str(item.get("note") or ""), 90)
        visual = _ledger_safe_text(str(item.get("visual") or ""), 120)
        risk = _ledger_safe_text(str(item.get("risk") or ""), 120)
        spread_multiple = float(item.get("spread_multiple") or 0.0)
        target_pips = float(item.get("target_pips") or 0.0)
        recipe_bits = [
            "audit range",
            f"{str(item.get('range_side') or direction).upper()} @{entry_price}",
            f"TP {tp_price}",
        ]
        if note:
            recipe_bits.append(note)
        range_targets.append(
            {
                "pair": pair,
                "direction": direction,
                "source": "audit_range",
                "recipe": _ledger_safe_text(" | ".join(recipe_bits), 140),
                "regime_hint": "range",
                "range_entry_price": float(entry_price),
                "range_tp_price": float(tp_price),
                "range_opposite_entry_price": item.get("opposite_entry_price"),
                "range_visual": visual,
                "range_risk": risk,
                "range_note": note,
                "range_side": str(item.get("range_side") or "").upper(),
                "range_payoff_ratio": spread_multiple,
                "range_target_pips": target_pips,
            }
        )

    range_targets.sort(
        key=lambda item: (
            float(item.get("range_payoff_ratio") or 0.0),
            float(item.get("range_target_pips") or 0.0),
            item.get("pair", ""),
            item.get("direction", ""),
        ),
        reverse=True,
    )

    return {
        "available": True,
        "stale": False,
        "age_min": age_min,
        "pairs": pair_map,
        "candidates": narrative_candidates,
        "strongest": strongest,
        "recent_pressure": recent_pressure,
        "range_targets": range_targets[:AUDIT_RANGE_TARGET_LIMIT],
        "narrative_details": narrative_details,
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
        "LIMIT": 2,
        "STOP-ENTRY": 1,
        "PASS": 0,
    }
    return mapping.get(str(style or "").upper(), 0)


def _tape_supports_direction(direction: str | None, bias: str | None) -> bool:
    side = str(direction or "").upper()
    if side == "LONG":
        return bias == "buyers pressing"
    if side == "SHORT":
        return bias == "sellers pressing"
    return False


def _tape_opposes_direction(direction: str | None, bias: str | None) -> bool:
    side = str(direction or "").upper()
    if side == "LONG":
        return bias == "sellers pressing"
    if side == "SHORT":
        return bias == "buyers pressing"
    return False


def _tape_fallback_style(regime: str | None) -> str:
    if regime in {"trending", "transition", "squeeze"}:
        return "STOP-ENTRY"
    return "LIMIT"


def _live_tape_brief(summary: dict | None) -> str:
    if not summary:
        return "live tape unavailable"
    if summary.get("tape") == "unavailable":
        mode = summary.get("probe_mode") or "?"
        error = summary.get("error") or "no data"
        return f"unavailable ({error}; mode={mode})"

    mode = summary.get("probe_mode") or "?"
    return (
        f"{summary.get('bias')} / {summary.get('tape')} "
        f"(move {float(summary.get('delta_pips', 0.0)):+.1f}pip, "
        f"range {float(summary.get('range_pips', 0.0)):.1f}pip, "
        f"spread {float(summary.get('avg_spread_pips', 0.0)):.1f}/"
        f"{float(summary.get('max_spread_pips', 0.0)):.1f}pip, mode={mode})"
    )


def _live_tape_bucket(direction: str, summary: dict | None) -> str:
    if not summary:
        return "unavailable"
    tape = str(summary.get("tape") or "unavailable")
    bias = str(summary.get("bias") or "unknown")
    if tape == "unavailable":
        return "unavailable"
    if tape == "spread unstable":
        return "spread_unstable"
    if tape == "quiet / stable":
        return "quiet_stable"
    if tape == "friction-dominated":
        return "friction"
    if tape == "whipsaw / two-way" or bias == "two-way":
        return "two_way"
    if _tape_opposes_direction(direction, bias):
        return "opposed"
    if _tape_supports_direction(direction, bias):
        return "aligned"
    return "mixed"


def _live_tape_bucket_label(bucket: str | None) -> str:
    mapping = {
        "aligned": "aligned tape",
        "mixed": "mixed tape",
        "quiet_stable": "quiet-stable tape",
        "two_way": "two-way tape",
        "friction": "friction tape",
        "opposed": "opposed tape",
        "spread_unstable": "spread-unstable tape",
        "unavailable": "unavailable tape",
    }
    key = str(bucket or "").strip().lower()
    return mapping.get(key, f"{key or 'unknown'} tape")


def _execution_style_context_label(style: str | None) -> str:
    return f"{str(style or 'unknown').upper()} execution"


def _profile_live_tape_rank(profile: dict) -> int:
    summary = profile.get("live_tape") or {}
    tape = str(summary.get("tape") or "unavailable")
    bias = str(summary.get("bias") or "unknown")
    direction = str(profile.get("direction") or "")

    if tape == "spread unstable":
        return -3
    if tape == "unavailable":
        return -2
    if _tape_opposes_direction(direction, bias) and tape in {"clean one-way", "tradeable"}:
        return -2
    if tape == "quiet / stable":
        return 1 if not _tape_opposes_direction(direction, bias) else 0
    if tape == "friction-dominated":
        return -1
    if tape == "whipsaw / two-way" or bias in {"two-way", "mixed"}:
        return 0
    if _tape_supports_direction(direction, bias) and tape == "clean one-way":
        return 3
    if _tape_supports_direction(direction, bias):
        return 2
    if tape == "tradeable":
        return 1
    return 0


def _profile_live_tape_label(profile: dict) -> str:
    summary = profile.get("live_tape") or {}
    tape = str(summary.get("tape") or "unavailable")
    bias = str(summary.get("bias") or "unknown")
    rank = _profile_live_tape_rank(profile)
    if rank >= 3:
        return f"aligned {bias} / {tape}"
    if rank == 2:
        return f"leaning {bias} / {tape}"
    if rank == 1:
        if tape == "quiet / stable":
            return f"quiet-stable {bias}"
        return f"usable {bias} / {tape}"
    if rank == 0:
        return f"mixed {bias} / {tape}"
    if rank == -1:
        return f"friction {bias} / {tape}"
    if rank == -2:
        return f"against {bias} / {tape}"
    return f"unstable {bias} / {tape}"


def _quiet_stable_market_scout_reason(profile: dict, plan: dict, summary: dict | None) -> str | None:
    style = str((plan or {}).get("style") or "PASS").upper()
    if style != "STOP-ENTRY":
        return None

    pair = str(profile.get("pair") or "")
    if pair not in DIRECT_USD_PAIRS:
        return None

    regime = str(profile.get("current_regime") or "").lower()
    if regime not in {"trending", "transition", "squeeze"}:
        return None

    if str((summary or {}).get("tape") or "") != "quiet / stable":
        return None

    direction = str(profile.get("direction") or "")
    bias = str((summary or {}).get("bias") or "unknown")
    if _tape_opposes_direction(direction, bias):
        return None

    if str(profile.get("context_bias") or "") == "headwind":
        pressure_rank = int(profile.get("promotion_pressure_rank", 0) or 0)
        trade_count = int(profile.get("trade_count", 0) or 0)
        trade_ev = float(profile.get("trade_ev", 0.0) or 0.0)
        if pressure_rank < 14 or trade_count < 5 or trade_ev <= 0:
            return None

    learning_score = int(profile.get("learning_score", 0) or 0)
    cap_rank = _cap_rank(str(profile.get("allocation_cap") or ""))
    pressure_rank = int(profile.get("promotion_pressure_rank", 0) or 0)
    trade_count = int(profile.get("trade_count", 0) or 0)
    trade_ev = float(profile.get("trade_ev", 0.0) or 0.0)
    spread_ratio = plan.get("spread_ratio")
    if learning_score < 45:
        return None
    if cap_rank < 1:
        if pressure_rank < 14 or trade_count < 5 or trade_ev <= 0:
            return None
    elif pressure_rank < 8:
        return None
    if spread_ratio is not None and float(spread_ratio) > 0.16:
        return None

    return (
        "quiet-stable direct-USD seat with repeat pressure can pay as a small market scout "
        "instead of another trigger-only rewrite"
    )


def _pass_cap_pressure_participation_plan(
    profile: dict,
    *,
    normal_market_spread: bool,
) -> tuple[str | None, str | None]:
    regime = str(profile.get("current_regime") or "").lower()
    if regime not in {"trending", "transition", "squeeze"}:
        return None, None
    if not normal_market_spread:
        return None, None

    learning_score = int(profile.get("learning_score", 0) or 0)
    trade_count = int(profile.get("trade_count", 0) or 0)
    trade_ev = float(profile.get("trade_ev", 0.0) or 0.0)
    pressure_rank = int(profile.get("promotion_pressure_rank", 0) or 0)
    source = str(profile.get("source") or "")
    context_bias = str(profile.get("context_bias") or "neutral")
    support = (
        trade_count >= 5
        or trade_ev > 0
        or source in {"audit", "audit_range"}
        or bool(profile.get("has_exact"))
        or profile.get("state") == "confirmed"
    )

    if learning_score < 45 or pressure_rank < 12 or not support:
        return None, None
    if context_bias == "headwind" and pressure_rank < 16 and trade_ev <= 0:
        return None, None

    pressure_note = str(profile.get("promotion_pressure_note") or "").strip()
    note = (
        "repeat audit / missed-seat pressure is too strong to leave this pass-cap seat as prose; "
        "re-open it as a thin trigger-first scout"
    )
    if pressure_note:
        note = f"{note} ({pressure_note})"
    return "STOP-ENTRY", note


def _fresh_profile_sort_key(profile: dict) -> tuple[int, int, int, int, int, float]:
    return (
        _execution_style_rank(_profile_default_expression(profile)),
        int(profile.get("promotion_pressure_rank", 0) or 0),
        _profile_live_tape_rank(profile),
        int(profile.get("learning_score", 0)),
        _cap_rank(str(profile.get("allocation_cap", ""))),
        SOURCE_PRIORITY.get(str(profile.get("source", "")), 0),
        float(profile.get("trade_ev", 0.0)),
    )


def _profile_trigger_bucket(profile: dict) -> str:
    blob = " ".join(
        str(profile.get(field) or "")
        for field in (
            "recipe",
            "seat_family",
            "audit_why",
            "range_note",
            "range_visual",
            "execution_note",
        )
    ).lower()
    if any(token in blob for token in ("range", "rail", "opposite band", "mean revert", "mean-revert", "fade")):
        return "range"
    if any(token in blob for token in ("pullback", "dip", "retest", "reload", "limit")):
        return "pullback"
    if any(
        token in blob
        for token in (
            "breakout",
            "trigger",
            "stop-entry",
            "stop entry",
            "momentum",
            "squeeze",
            "trend",
            "continuation",
            "enter now",
        )
    ):
        return "continuation"
    return "generic"


def _profile_price_signature(pair: str, price: object) -> str:
    if price in {None, ""}:
        return "na"
    try:
        value = float(price)
    except (TypeError, ValueError):
        return _normalize_identity_fragment(price, limit=24)
    precision = 3 if str(pair).endswith("JPY") else 5
    return f"{value:.{precision}f}"


def _profile_anchor_signature(profile: dict) -> tuple[object, ...] | None:
    source = str(profile.get("source") or "")
    pair = str(profile.get("pair") or "")
    direction = str(profile.get("direction") or "")
    style = _profile_default_expression(profile)

    if source == "held":
        return ("held", str(profile.get("seat_key") or ""))
    if source == "pending":
        return ("pending", str(profile.get("armed_id") or profile.get("seat_key") or ""))
    if source == "audit_range":
        return (
            "audit_range",
            pair,
            direction,
            style,
            str(profile.get("range_side") or ""),
            _profile_price_signature(pair, profile.get("range_entry_price")),
            _profile_price_signature(pair, profile.get("range_tp_price")),
        )

    audit_entry = profile.get("audit_entry_price")
    audit_tp = profile.get("audit_tp_price")
    if audit_entry is not None or audit_tp is not None:
        return (
            "audit",
            pair,
            direction,
            style,
            _profile_trigger_bucket(profile),
            _profile_price_signature(pair, audit_entry),
            _profile_price_signature(pair, audit_tp),
        )
    return None


def _profile_generic_signature(profile: dict) -> tuple[object, ...]:
    pair = str(profile.get("pair") or "")
    direction = str(profile.get("direction") or "")
    style = _profile_default_expression(profile)
    if style == "MARKET":
        trigger_bucket = "live"
    else:
        trigger_bucket = _profile_trigger_bucket(profile)
    return (pair, direction, style, trigger_bucket)


def _profile_pair_style_signature(profile: dict) -> tuple[str, str, str]:
    return (
        str(profile.get("pair") or ""),
        str(profile.get("direction") or ""),
        _profile_default_expression(profile),
    )


def _profile_consolidation_rank(profile: dict) -> tuple[int, int, int, int, int, float]:
    return (
        _execution_style_rank(_profile_default_expression(profile)),
        _cap_rank(str(profile.get("allocation_cap") or "")),
        int(profile.get("learning_score", 0) or 0),
        int(profile.get("promotion_pressure_rank", 0) or 0),
        SOURCE_PRIORITY.get(str(profile.get("source") or ""), 0),
        float(profile.get("trade_ev", 0.0) or 0.0),
    )


def _merge_profile_cluster(cluster: list[dict]) -> dict:
    leader = max(cluster, key=_profile_consolidation_rank)
    merged = dict(leader)
    if len(cluster) == 1:
        return merged

    ordered = sorted(cluster, key=_profile_consolidation_rank, reverse=True)
    merged_sources: list[str] = []
    merged_families: list[str] = []
    merged_keys: list[str] = []
    for item in ordered:
        source = str(item.get("source") or "")
        if source and source not in merged_sources:
            merged_sources.append(source)
        family = str(item.get("seat_family") or "")
        if family and family not in merged_families:
            merged_families.append(family)
        key = str(item.get("seat_key") or "")
        if key and key not in merged_keys:
            merged_keys.append(key)

    merged["merged_sources"] = merged_sources
    merged["merged_seat_families"] = merged_families
    merged["merged_seat_keys"] = merged_keys
    merged["merged_source_count"] = len(cluster)
    primary_label = _profile_source_label({"source": merged.get("source"), "merged_sources": merged_sources})
    if len(merged_sources) == 1:
        corroboration = f"{primary_label} duplicate seat lines collapsed into one live seat"
    else:
        corroboration = (
            f"{primary_label} corroboration: "
            + " + ".join(source.upper() for source in merged_sources)
            + " describe the same live seat"
        )
    merged["cross_source_corroboration"] = _ledger_safe_text(corroboration, 180)
    return merged


def _consolidate_cross_source_profiles(profiles: list[dict]) -> list[dict]:
    if len(profiles) < 2:
        return profiles

    groups: dict[tuple[object, ...], list[dict]] = {}
    group_order: list[tuple[object, ...]] = []
    anchored_by_generic: dict[tuple[object, ...], list[tuple[object, ...]]] = {}
    anchored_by_pair_style: dict[tuple[str, str, str], list[tuple[object, ...]]] = {}
    generic_profiles: list[dict] = []

    def ensure_group(key: tuple[object, ...]) -> list[dict]:
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        return groups[key]

    for profile in profiles:
        anchor_key = _profile_anchor_signature(profile)
        if anchor_key is None:
            generic_profiles.append(profile)
            continue
        group_key = ("anchor",) + anchor_key
        ensure_group(group_key).append(profile)
        if str(profile.get("source") or "") in {"held", "pending"}:
            continue
        generic_key = _profile_generic_signature(profile)
        if group_key not in anchored_by_generic.setdefault(generic_key, []):
            anchored_by_generic[generic_key].append(group_key)
        pair_style_key = _profile_pair_style_signature(profile)
        if group_key not in anchored_by_pair_style.setdefault(pair_style_key, []):
            anchored_by_pair_style[pair_style_key].append(group_key)

    for profile in generic_profiles:
        generic_key = _profile_generic_signature(profile)
        pair_style_key = _profile_pair_style_signature(profile)
        candidate_group_keys = anchored_by_generic.get(generic_key) or anchored_by_pair_style.get(pair_style_key) or []
        if candidate_group_keys:
            best_group_key = max(
                candidate_group_keys,
                key=lambda key: max(_profile_consolidation_rank(item) for item in groups.get(key, [])),
            )
            groups[best_group_key].append(profile)
            continue
        ensure_group(("generic",) + generic_key).append(profile)

    merged = [_merge_profile_cluster(groups[key]) for key in group_order]
    merged.sort(key=_fresh_profile_sort_key, reverse=True)
    return merged


def _apply_live_tape_profile_guard(profile: dict, plan: dict, live_tape: dict | None) -> dict:
    guarded = dict(plan or {})
    summary = dict(live_tape or {})
    preserve_armed_style = bool(guarded.get("preserve_armed_style"))
    if not summary:
        summary = {
            "pair": profile.get("pair"),
            "samples": 0,
            "bias": "unknown",
            "tape": "unavailable",
            "error": "no current pricing read",
        }
    guarded["live_tape"] = summary
    guarded["live_tape_note"] = _live_tape_brief(summary)

    style = str(guarded.get("style") or "PASS").upper()
    if style == "PASS":
        return guarded

    scout_reason = _quiet_stable_market_scout_reason(profile, guarded, summary)
    if scout_reason and not preserve_armed_style and guarded.get("exact_pretrade_status") != "ok":
        guarded["style"] = "MARKET"
        guarded["orderability"] = "ENTER NOW"
        guarded["note"] = (
            f"{guarded.get('note', '').strip()}; {scout_reason}; "
            f"{guarded['live_tape_note']}"
        ).strip("; ")
        return guarded

    regime = str(profile.get("current_regime") or "").lower() or None
    direction = str(profile.get("direction") or "")
    tape = str(summary.get("tape") or "unavailable")
    bias = str(summary.get("bias") or "unknown")
    fallback_style = _tape_fallback_style(regime)
    guard_note = None

    if style == "MARKET":
        if preserve_armed_style:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape now reads {guarded['live_tape_note']}; "
                "existing armed order stays as-is unless the thesis is dead"
            ).strip("; ")
            return guarded
        if tape == "unavailable":
            guard_note = "live tape is unavailable, so do not pay market without a current read"
        elif tape in {"spread unstable", "friction-dominated"}:
            guard_note = f"live tape is {tape}, so the board seat stays armed but not payable at market"
        elif _tape_opposes_direction(direction, bias):
            guard_note = f"live tape is paying the other side ({guarded['live_tape_note']}), so do not chase into it"
        elif tape == "whipsaw / two-way" or (tape != "quiet / stable" and bias in {"two-way", "mixed"}):
            guard_note = f"live tape is still two-way ({guarded['live_tape_note']}), so require trigger proof first"
        if guard_note:
            guarded["style"] = fallback_style
            guarded["orderability"] = fallback_style
            guarded["note"] = f"{guard_note}; {guarded['live_tape_note']}"
            guarded["live_tape_guard"] = guard_note
            return guarded
        if tape == "quiet / stable":
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape is quiet but stable "
                f"({guarded['live_tape_note']}), so the chart/timeframe edge must justify paying market"
            ).strip("; ")
            return guarded
        if _tape_supports_direction(direction, bias):
            guarded["note"] = f"{guarded.get('note', '').strip()}; live tape confirms {guarded['live_tape_note']}".strip("; ")
        return guarded

    if style == "STOP-ENTRY":
        if preserve_armed_style:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape now reads {guarded['live_tape_note']}; "
                "existing armed stop stays live unless the thesis is dead"
            ).strip("; ")
            return guarded
        if _tape_opposes_direction(direction, bias) and tape in {"clean one-way", "tradeable"}:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape still pays the other side "
                f"({guarded['live_tape_note']}), so the reclaim/break print must do the work"
            ).strip("; ")
        elif tape == "quiet / stable":
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape is calm/stable "
                f"({guarded['live_tape_note']}), so trigger proof still matters more than urgency"
            ).strip("; ")
        elif tape in {"spread unstable", "friction-dominated", "whipsaw / two-way"} or bias in {"two-way", "mixed"}:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape is messy ({guarded['live_tape_note']}), "
                "so trigger proof remains mandatory"
            ).strip("; ")
        return guarded

    if style == "LIMIT":
        if preserve_armed_style:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape now reads {guarded['live_tape_note']}; "
                "existing armed limit stays live unless the thesis is dead"
            ).strip("; ")
            return guarded
        if _tape_supports_direction(direction, bias) and tape == "clean one-way":
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape is already one-way "
                f"({guarded['live_tape_note']}), so keep this passive only if price improvement still matters"
            ).strip("; ")
        elif _tape_opposes_direction(direction, bias) and tape in {"clean one-way", "tradeable"}:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; live tape pays the other side now "
                f"({guarded['live_tape_note']}), so do not upgrade this without a real contradiction"
            ).strip("; ")
    return guarded


def _profile_allocation_band(profile: dict) -> tuple[str, str]:
    cap = str(profile.get("allocation_cap") or "")
    style = _profile_default_expression(profile)
    verdict = str(profile.get("verdict") or "").lower()
    score = int(profile.get("learning_score", 0) or 0)
    context_bias = str(profile.get("context_bias") or "neutral")

    positive_signals = 0
    negative_signals = 0
    reasons: list[str] = []

    if cap in {CAP_LABELS["as_confirmed"], CAP_LABELS["a_max"], CAP_LABELS["ba_max"]}:
        positive_signals += 1
        reasons.append(f"cap {cap}")
    if score >= 75:
        positive_signals += 1
        reasons.append(f"learning {score}/99")
    if style in {"LIMIT", "STOP-ENTRY", "MARKET"}:
        positive_signals += 1
        reasons.append(f"{style} is order-honest")
    if context_bias == "tailwind":
        positive_signals += 1
        reasons.append("context tailwind")
    if any(tag in verdict for tag in ("confirmed", "watch edge", "pair memory positive")):
        positive_signals += 1

    if cap in {CAP_LABELS["pass"], CAP_LABELS["b_only"]}:
        negative_signals += 1
    if style == "PASS":
        negative_signals += 2
    if "no-edge" in verdict or "restricted" in verdict:
        negative_signals += 2
    if context_bias == "headwind":
        negative_signals += 1

    if negative_signals >= 3 and positive_signals <= 2:
        return "B-", "passive or headwind B"
    if positive_signals >= 3 and negative_signals <= 2:
        return "B+", "promotable B once price/trigger is honest"
    if reasons:
        return "B0", reasons[0]
    return "B0", "mixed evidence"


def _default_upgrade_action_for_profile(profile: dict) -> str:
    style = _profile_default_expression(profile)
    if style in {"MARKET", "STOP-ENTRY", "LIMIT"}:
        return style
    regime = str(profile.get("current_regime") or "").lower()
    if regime in {"trending", "transition", "squeeze"}:
        return "STOP-ENTRY"
    return "LIMIT"


def _is_excavatable_pass_profile(profile: dict) -> bool:
    style = _profile_default_expression(profile)
    if style != "PASS":
        return False

    verdict = str(profile.get("verdict") or "").lower()
    note = str(profile.get("execution_note") or "").lower()
    cap_rank = _cap_rank(str(profile.get("allocation_cap") or ""))
    learning_score = int(profile.get("learning_score", 0) or 0)

    if "no-edge" in verdict or cap_rank <= 0:
        return False

    hard_block_notes = (
        "not worth real risk",
        "learning cap already says pass unless exceptional",
        "pass unless exceptional",
    )
    if any(blocker in note for blocker in hard_block_notes):
        return False

    if learning_score >= 62 and cap_rank >= 2:
        return True
    if (
        learning_score >= 54
        and cap_rank >= 3
        and any(tag in verdict for tag in ("confirmed", "watch", "pair memory positive"))
    ):
        return True
    return False


def _select_bucket_board_profile(fresh_profiles: list[dict], bucket: str) -> dict | None:
    bucket_profiles = [profile for profile in fresh_profiles if profile.get("bucket") == bucket]
    if not bucket_profiles:
        return None

    actionable = [
        profile for profile in bucket_profiles
        if _profile_default_expression(profile) != "PASS"
    ]
    excavatable = [profile for profile in bucket_profiles if _is_excavatable_pass_profile(profile)]
    pool = actionable or excavatable
    if not pool:
        return None
    return max(
        pool,
        key=_fresh_profile_sort_key,
    )


def _profile_lane_signature(profile: dict) -> tuple[str, str, str, str]:
    return (
        str(profile.get("pair") or ""),
        str(profile.get("direction") or ""),
        str(profile.get("seat_family") or ""),
        _profile_default_expression(profile),
    )


def _select_multi_vehicle_lanes(
    fresh_profiles: list[dict],
    max_lanes: int = MAX_DEPLOYMENT_LANES,
) -> list[dict]:
    candidates = [
        profile for profile in fresh_profiles
        if _profile_default_expression(profile) != "PASS"
    ]
    candidates.sort(key=_fresh_profile_sort_key, reverse=True)

    selected: list[dict] = []
    seen_signatures: set[tuple[str, str, str, str]] = set()
    pair_counts: Counter[str] = Counter()
    pair_direction_counts: Counter[tuple[str, str]] = Counter()

    def can_add(profile: dict) -> bool:
        pair = str(profile.get("pair", ""))
        direction = str(profile.get("direction", ""))
        signature = _profile_lane_signature(profile)
        if not pair or signature in seen_signatures:
            return False
        if pair_counts[pair] >= MAX_SAME_PAIR_LANES:
            return False
        if pair_direction_counts[(pair, direction)] >= MAX_SAME_PAIR_DIRECTION_LANES:
            return False
        return True

    def add(profile: dict) -> None:
        pair = str(profile.get("pair", ""))
        direction = str(profile.get("direction", ""))
        selected.append(profile)
        seen_signatures.add(_profile_lane_signature(profile))
        pair_counts[pair] += 1
        pair_direction_counts[(pair, direction)] += 1

    for profile in candidates:
        if len(selected) >= max_lanes:
            break
        if can_add(profile):
            add(profile)

    return selected


def _multi_vehicle_role(index: int) -> str:
    mapping = {
        1: "PRIMARY",
        2: "BACKUP",
        3: "THIRD CURRENCY",
        4: "FOURTH SEAT",
        5: "FIFTH SEAT",
    }
    role = mapping.get(index, f"SEAT {index}")
    return f"Lane {index} / {role}"


def _multi_vehicle_reason(profile: dict, prior: list[dict]) -> str:
    pair = str(profile.get("pair", ""))
    base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
    seat_family = str(profile.get("seat_family") or "seat")
    if not prior:
        return f"first live lane: {base}/{quote} is currently the cleanest expression"

    parts = []
    prior_same_pair = [item for item in prior if str(item.get("pair", "")) == pair]
    if prior_same_pair:
        parts.append(
            f"same pair is allowed here because `{seat_family}` is a separate trigger/vehicle, not blind averaging"
        )
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
    seeds: dict[str, dict] = {}

    def ensure_seed(profile: dict, pair: str, direction: str) -> dict:
        key = str(
            profile.get("seat_key")
            or f"{pair}|{direction}|{profile.get('seat_family') or profile.get('source')}"
        )
        return seeds.setdefault(
            key,
            {
                "pair": pair,
                "direction": direction,
                "seat_family": str(profile.get("seat_family") or ""),
                "priority": 0,
                "why_bits": [],
                "blocker": None,
                "upgrade_action": None,
            },
        )

    for idx, profile in enumerate(fresh_profiles[:MAX_DEPLOYMENT_LANES], start=1):
        pair = str(profile.get("pair", ""))
        direction = str(profile.get("direction", ""))
        if not pair or not direction:
            continue
        style = _profile_default_expression(profile)
        excavatable_pass = _is_excavatable_pass_profile(profile)
        if style == "PASS" and not excavatable_pass:
            continue

        seed = ensure_seed(profile, pair, direction)
        priority = int(profile.get("learning_score", 0))
        if style == "LIMIT":
            priority += 35
        elif style == "STOP-ENTRY":
            priority += 24
        elif style == "MARKET":
            priority += 10
        elif excavatable_pass:
            priority += 18
        if idx == 1:
            priority += 12
        elif idx == 2:
            priority += 8
        elif idx == 3:
            priority += 4
        tape_rank = _profile_live_tape_rank(profile)
        priority += max(-6, min(12, tape_rank * 4))
        seed["priority"] = max(seed["priority"], priority)

        why_bits = seed["why_bits"]
        tag = f"tournament #{idx}"
        if tag not in why_bits:
            why_bits.append(tag)
        tape_tag = f"tape {_profile_live_tape_label(profile)}"
        if tape_tag not in why_bits:
            why_bits.append(tape_tag)
        family_tag = f"seat {profile.get('seat_family')}"
        if family_tag not in why_bits:
            why_bits.append(family_tag)
        if best_direct and pair == best_direct.get("pair") and direction == best_direct.get("direction"):
            why_bits.append("best direct-USD")
        if best_cross and pair == best_cross.get("pair") and direction == best_cross.get("direction"):
            why_bits.append("best cross")
        if best_usdjpy and pair == best_usdjpy.get("pair") and direction == best_usdjpy.get("direction"):
            why_bits.append("best USD_JPY")

        for label in audit_context.get("pairs", {}).get((pair, direction), []):
            if label not in why_bits:
                why_bits.append(label)
            if "strongest-unheld" in label:
                seed["priority"] += 18
            elif "audit repeated" in label:
                seed["priority"] += 22
            else:
                seed["priority"] += 10

        pressure_note = str(profile.get("promotion_pressure_note") or "").strip()
        if pressure_note and pressure_note not in why_bits:
            why_bits.append(pressure_note)
        seed["priority"] += int(profile.get("promotion_pressure_rank", 0) or 0)

        summary = _ledger_safe_text(
            f"learning {int(profile.get('learning_score', 0))}/100 {profile.get('verdict')}",
            120,
        )
        if summary not in why_bits:
            why_bits.append(summary)
        if excavatable_pass:
            pass_tag = "near-S one print away"
            if pass_tag not in why_bits:
                why_bits.append(pass_tag)

        blocker = profile.get("execution_note")
        if blocker:
            seed["blocker"] = _ledger_safe_text(blocker, 120)
        elif style == "MARKET":
            seed["blocker"] = "no memory blocker remains; only a live chart contradiction should keep this out of S Hunt"
        elif excavatable_pass:
            seed["blocker"] = (
                "live chart still contradicts immediate entry; keep it on the podium until the missing print completes"
            )
        if pressure_note and seed.get("blocker"):
            seed["blocker"] = _ledger_safe_text(
                f"{seed['blocker']}; repeated pressure means either arm it now or write the exact contradiction",
                120,
            )

        upgrade_action = _default_upgrade_action_for_profile(profile)
        if upgrade_action:
            seed["upgrade_action"] = upgrade_action

    ordered = sorted(
        seeds.values(),
        key=lambda item: (
            int(item.get("priority", 0)),
            1 if str(item.get("upgrade_action", "")) == "LIMIT" else 0,
            1 if str(item.get("upgrade_action", "")) == "STOP-ENTRY" else 0,
        ),
        reverse=True,
    )

    result = []
    pair_counts: Counter[str] = Counter()
    for seed in ordered:
        if pair_counts[str(seed["pair"])] >= 2:
            continue
        if not seed.get("blocker") or not seed.get("upgrade_action"):
            continue
        result.append(
            {
                "pair": seed["pair"],
                "direction": seed["direction"],
                "seat_family": seed.get("seat_family"),
                "closest_to_s_because": _ledger_safe_text(" + ".join(seed["why_bits"]), 150),
                "still_blocked_by": seed["blocker"],
                "upgrade_action": seed["upgrade_action"],
            }
        )
        pair_counts[str(seed["pair"])] += 1
        if len(result) >= MAX_PODIUM_SEEDS:
            break

    return result


ENTRY_PENDING_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED"}


def _iter_pending_entry_orders(pending_orders: list[dict]):
    for order in pending_orders or []:
        order_type = str(order.get("type") or "").upper()
        if order_type not in ENTRY_PENDING_TYPES:
            continue
        client_ext = order.get("clientExtensions") or {}
        if order.get("tradeID") or client_ext.get("tag") in {"tp", "sl"}:
            continue
        yield order


def _count_pending_entry_orders(pending_orders: list[dict]) -> int:
    return sum(1 for _ in _iter_pending_entry_orders(pending_orders))


def _pending_entry_receipt_line(order: dict | None) -> str:
    if not order:
        return "none because no fresh pending entry order is live"
    pair = str(order.get("instrument") or "?").strip()
    direction = _units_to_direction(order.get("units")) or "?"
    order_type = str(order.get("type") or "").upper()
    style = "LIMIT" if order_type == "LIMIT" else "STOP-ENTRY"
    return f"{pair} {direction} {style} id=`{order.get('id', '?')}`"


def _apply_profile_promotion_pressure(
    profile: dict,
    audit_context: dict,
    missed_pressure: dict,
) -> None:
    key = (profile.get("pair"), profile.get("direction"))
    bits: list[str] = []
    rank = 0

    audit_stats = (audit_context.get("recent_pressure") or {}).get("pairs", {}).get(key)
    if audit_stats:
        count = int(audit_stats.get("count", 0) or 0)
        hours = int(audit_stats.get("window_hours", AUDIT_PRESSURE_LOOKBACK_HOURS) or AUDIT_PRESSURE_LOOKBACK_HOURS)
        bits.append(f"audit repeated {count}x/{hours}h")
        rank += min(20, 6 + count * 3)
        why = str(audit_stats.get("why") or "").strip()
        if why:
            bits.append(_ledger_safe_text(why, 80))

    missed_stats = (missed_pressure.get("pairs") or {}).get(key)
    if missed_stats:
        count = int(missed_stats.get("count", 0) or 0)
        max_pip_move = float(missed_stats.get("max_pip_move", 0.0) or 0.0)
        bits.append(f"missed {count}x today; best {max_pip_move:.1f}pip worked")
        rank += min(24, 8 + int(max_pip_move // 4))

    profile["promotion_pressure_note"] = _ledger_safe_text(" + ".join(bits), 150) if bits else ""
    profile["promotion_pressure_rank"] = rank


def _compute_session_intent_gate(
    trades_data: dict,
    pending_orders: list[dict],
    audit_context: dict,
    missed_pressure: dict,
    range_profiles: list[dict] | None = None,
) -> dict:
    open_trades = len(trades_data.get("trades", []) or [])
    pending_entries = _count_pending_entry_orders(pending_orders)
    reasons: list[str] = []

    if open_trades:
        reasons.append(f"open trades={open_trades}")
    if pending_entries:
        reasons.append(f"pending entries={pending_entries}")

    repeated = [
        (pair, direction, stats)
        for (pair, direction), stats in (audit_context.get("recent_pressure") or {}).get("pairs", {}).items()
        if int(stats.get("count", 0) or 0) >= AUDIT_REPEAT_TRIGGER_COUNT
    ]
    if repeated:
        pair, direction, stats = sorted(
            repeated,
            key=lambda item: (
                int(item[2].get("count", 0) or 0),
                int(item[2].get("strongest_count", 0) or 0),
                item[0],
                item[1],
            ),
            reverse=True,
        )[0]
        reasons.append(
            f"repeat audit pressure {pair} {direction} {int(stats.get('count', 0) or 0)}x/"
            f"{int(stats.get('window_hours', AUDIT_PRESSURE_LOOKBACK_HOURS) or AUDIT_PRESSURE_LOOKBACK_HOURS)}h"
        )

    missed = list((missed_pressure.get("pairs") or {}).items())
    if missed:
        (pair, direction), stats = sorted(
            missed,
            key=lambda item: (
                float(item[1].get("max_pip_move", 0.0) or 0.0),
                int(item[1].get("count", 0) or 0),
                item[0][0],
                item[0][1],
            ),
            reverse=True,
        )[0]
        reasons.append(
            f"fresh missed-S pressure {pair} {direction} "
            f"{float(stats.get('max_pip_move', 0.0) or 0.0):.1f}pip"
        )

    fresh_range_profiles = [
        item for item in (range_profiles or [])
        if str(item.get("source") or "") == "audit_range"
        and _profile_default_expression(item) != "PASS"
        and float(item.get("range_payoff_ratio", 0.0) or 0.0) >= 4.0
        and float(item.get("range_target_pips", 0.0) or 0.0) >= 10.0
    ]
    if fresh_range_profiles:
        top = max(
            fresh_range_profiles,
            key=lambda item: (
                int(item.get("learning_score", 0) or 0),
                _cap_rank(str(item.get("allocation_cap") or "")),
                float(item.get("range_payoff_ratio", 0.0) or 0.0),
                float(item.get("range_target_pips", 0.0) or 0.0),
                str(item.get("pair") or ""),
                str(item.get("direction") or ""),
            ),
        )
        reasons.append(
            f"fresh audit range {top.get('pair')} {top.get('direction')} "
            f"@{top.get('range_entry_price')} -> {top.get('range_tp_price')}"
        )

    if reasons:
        return {"mode": "FULL_TRADER", "reasons": reasons}
    return {
        "mode": "WATCH-ONLY",
        "reasons": [
            "flat live book",
            "no repeated audit pressure in the last 6h",
            f"no missed S >= {MISSED_SEAT_MIN_PIPS:.0f}pip in the last {MISSED_SEAT_LOOKBACK_HOURS}h",
        ],
    }


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


def _stat_source_label(
    source: str,
    lookback_days: int | None,
    *,
    context_label: str | None = None,
) -> str:
    if source == "pretrade_recent":
        label = f"recent pretrade feedback ({live_history_scope_label(lookback_days)})"
    elif source == "trades_recent":
        label = f"recent closed trades ({live_history_scope_label(lookback_days)})"
    else:
        label = f"discretionary closed trades ({live_history_scope_label(None)})"
    if context_label:
        return f"{label}, {context_label}"
    return label


def _format_feedback_stat(stats: dict | None, label: str = "pair") -> str:
    if not stats or int(stats.get("count", 0) or 0) <= 0:
        return f"{label}: no sample"
    return (
        f"{stats.get('source_label', label)}: WR {float(stats.get('win_rate', 0.0))*100:.0f}% "
        f"EV {float(stats.get('ev', 0.0)):+.0f} n={int(stats.get('count', 0))}"
    )


def _recent_feedback_override(stats: dict | None) -> dict | None:
    if not stats or stats.get("source") != "pretrade_recent":
        return None

    count = int(stats.get("count", 0) or 0)
    if count < 4:
        return None

    ev = float(stats.get("ev", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    summary = _format_feedback_stat(stats, "pair")

    if ev < 0 and win_rate <= 0.40:
        return {"tier": "hard_headwind", "reason": summary}
    if ev <= 0 or win_rate <= 0.35:
        return {"tier": "soft_headwind", "reason": summary}
    return None


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
    conn = _open_memory_read_conn()

    def pack(
        rows,
        key_builder,
        source: str,
        lookback_days: int | None,
        *,
        context_label_builder=None,
    ):
        out = {}
        for row in rows:
            *head, cnt, ev, total_pl, wins = row
            cnt = int(cnt or 0)
            context_label = context_label_builder(head) if context_label_builder else None
            out[key_builder(head)] = {
                "count": cnt,
                "ev": float(ev or 0.0),
                "total_pl": float(total_pl or 0.0),
                "wins": int(wins or 0),
                "win_rate": (float(wins or 0) / float(cnt)) if cnt else 0.0,
                "source": source,
                "lookback_days": lookback_days,
                "context_label": context_label,
                "source_label": _stat_source_label(
                    source,
                    lookback_days,
                    context_label=context_label,
                ),
            }
        return out

    def merge_missing(primary: dict, fallback: dict) -> dict:
        merged = dict(primary)
        for key, value in fallback.items():
            merged.setdefault(key, value)
        return merged

    try:
        recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
        recent_trade_cutoff = live_history_start(RECENT_TRADE_LOOKBACK_DAYS)
        all_trade_cutoff = live_history_start(None)

        pair_pretrade_rows = conn.execute(
            """SELECT pair, direction,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM pretrade_outcomes
               WHERE pl IS NOT NULL
                 AND session_date >= ?
               GROUP BY pair, direction""",
            (recent_pretrade_cutoff,),
        ).fetchall()
        session_pretrade_rows = conn.execute(
            """SELECT pair, direction, session_bucket,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM pretrade_outcomes
               WHERE pl IS NOT NULL
                 AND COALESCE(session_bucket, '') <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, session_bucket""",
            (recent_pretrade_cutoff,),
        ).fetchall()
        regime_pretrade_rows = conn.execute(
            """SELECT pair, direction, regime_snapshot,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM pretrade_outcomes
               WHERE pl IS NOT NULL
                 AND COALESCE(regime_snapshot, '') <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, regime_snapshot""",
            (recent_pretrade_cutoff,),
        ).fetchall()
        tape_pretrade_rows = conn.execute(
            """SELECT pair, direction, live_tape_bucket,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM pretrade_outcomes
               WHERE pl IS NOT NULL
                 AND COALESCE(live_tape_bucket, '') <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, live_tape_bucket""",
            (recent_pretrade_cutoff,),
        ).fetchall()
        style_pretrade_rows = conn.execute(
            """SELECT pair, direction, execution_style,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM pretrade_outcomes
               WHERE pl IS NOT NULL
                 AND COALESCE(execution_style, '') <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, execution_style""",
            (recent_pretrade_cutoff,),
        ).fetchall()

        pair_trade_recent_rows = conn.execute(
            """SELECT pair, direction,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL
                 AND session_date >= ?
               GROUP BY pair, direction""",
            (recent_trade_cutoff,),
        ).fetchall()
        session_trade_recent_rows = conn.execute(
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
               WHERE pl IS NOT NULL
                 AND session_hour IS NOT NULL
                 AND session_date >= ?
               GROUP BY pair, direction, session_bucket""",
            (recent_trade_cutoff,),
        ).fetchall()
        regime_trade_recent_rows = conn.execute(
            """SELECT pair, direction, regime,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL
                 AND regime IS NOT NULL
                 AND TRIM(regime) <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, regime""",
            (recent_trade_cutoff,),
        ).fetchall()

        pair_trade_all_rows = conn.execute(
            """SELECT pair, direction,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL
                 AND session_date >= ?
               GROUP BY pair, direction"""
            ,
            (all_trade_cutoff,),
        ).fetchall()
        session_trade_all_rows = conn.execute(
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
                 AND session_date >= ?
               GROUP BY pair, direction, session_bucket"""
            ,
            (all_trade_cutoff,),
        ).fetchall()
        regime_trade_all_rows = conn.execute(
            """SELECT pair, direction, regime,
                      COUNT(*) AS cnt,
                      AVG(pl) AS ev,
                      SUM(pl) AS total_pl,
                      SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades
               WHERE pl IS NOT NULL
                 AND regime IS NOT NULL
                 AND TRIM(regime) <> ''
                 AND session_date >= ?
               GROUP BY pair, direction, regime"""
            ,
            (all_trade_cutoff,),
        ).fetchall()
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()

    pair_stats = merge_missing(
        pack(pair_pretrade_rows, lambda head: tuple(head), "pretrade_recent", RECENT_PRETRADE_LOOKBACK_DAYS),
        pack(pair_trade_recent_rows, lambda head: tuple(head), "trades_recent", RECENT_TRADE_LOOKBACK_DAYS),
    )
    pair_stats = merge_missing(
        pair_stats,
        pack(pair_trade_all_rows, lambda head: tuple(head), "trades_all", None),
    )

    session_stats = merge_missing(
        pack(session_pretrade_rows, lambda head: tuple(head), "pretrade_recent", RECENT_PRETRADE_LOOKBACK_DAYS),
        pack(session_trade_recent_rows, lambda head: tuple(head), "trades_recent", RECENT_TRADE_LOOKBACK_DAYS),
    )
    session_stats = merge_missing(
        session_stats,
        pack(session_trade_all_rows, lambda head: tuple(head), "trades_all", None),
    )

    regime_stats = merge_missing(
        pack(regime_pretrade_rows, lambda head: tuple(head), "pretrade_recent", RECENT_PRETRADE_LOOKBACK_DAYS),
        pack(regime_trade_recent_rows, lambda head: tuple(head), "trades_recent", RECENT_TRADE_LOOKBACK_DAYS),
    )
    regime_stats = merge_missing(
        regime_stats,
        pack(regime_trade_all_rows, lambda head: tuple(head), "trades_all", None),
    )

    tape_stats = pack(
        tape_pretrade_rows,
        lambda head: tuple(head),
        "pretrade_recent",
        RECENT_PRETRADE_LOOKBACK_DAYS,
        context_label_builder=lambda head: _live_tape_bucket_label(head[2]),
    )
    style_stats = pack(
        style_pretrade_rows,
        lambda head: tuple(head),
        "pretrade_recent",
        RECENT_PRETRADE_LOOKBACK_DAYS,
        context_label_builder=lambda head: _execution_style_context_label(head[2]),
    )

    _TRADE_CONTEXT_STATS = {
        "pair": pair_stats,
        "session": session_stats,
        "regime": regime_stats,
        "tape": tape_stats,
        "style": style_stats,
    }
    return _TRADE_CONTEXT_STATS


def _build_learning_profile(
    target: dict,
    registry: dict,
    live_tape: dict | None = None,
) -> dict:
    from datetime import datetime, timezone

    source = str(target.get("source") or "")
    exact, pair_only = _target_lessons_for_profile(target, registry)
    relevant = exact[:3] + pair_only[:2]
    top = exact[0] if exact else (pair_only[0] if pair_only else None)
    now_utc = datetime.now(timezone.utc)
    current_session = _current_session_bucket(now_utc)
    current_regime = str(target.get("regime_hint") or _infer_current_pair_regime(target.get("pair", "")) or "")
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
    current_tape_bucket = _live_tape_bucket(str(target.get("direction") or ""), live_tape)
    tape_stat = context_stats.get("tape", {}).get(
        (target.get("pair"), target.get("direction"), current_tape_bucket)
    )
    preferred_style = None
    if source == "audit_range":
        preferred_style = "LIMIT"
    elif source == "pending":
        armed_style = str(target.get("armed_style") or "").upper()
        if armed_style in {"LIMIT", "STOP-ENTRY"}:
            preferred_style = armed_style
    style_stat = None
    if preferred_style:
        style_stat = context_stats.get("style", {}).get(
            (target.get("pair"), target.get("direction"), preferred_style)
        )
    range_payoff_ratio = float(target.get("range_payoff_ratio") or 0.0)
    range_target_pips = float(target.get("range_target_pips") or 0.0)
    display_stats = pair_stat or {}
    feedback_stats = display_stats
    if (
        preferred_style
        and style_stat
        and int(style_stat.get("count", 0) or 0) >= RECENT_FEEDBACK_MIN_COUNT
        and source == "audit_range"
    ):
        feedback_stats = style_stat
    if tape_stat and int(tape_stat.get("count", 0) or 0) >= RECENT_FEEDBACK_MIN_COUNT:
        feedback_stats = tape_stat

    profile = {
        "pair": target.get("pair"),
        "direction": target.get("direction"),
        "source": source,
        "recipe": target.get("recipe", ""),
        "seat_family": target.get("seat_family") or _seat_family_from_target(target),
        "seat_key": target.get("seat_key") or _build_seat_key(
            target.get("pair"),
            target.get("direction"),
            source,
            target.get("recipe"),
            **{
                key: value
                for key, value in target.items()
                if key not in {"pair", "direction", "source", "recipe", "seat_key"}
            },
        ),
        "armed_style": target.get("armed_style"),
        "armed_id": target.get("armed_id"),
        "pending_metrics": target.get("pending_metrics"),
        "bucket": _pair_bucket(target.get("pair")),
        "learning_score": 18,
        "verdict": "limited history",
        "allocation_cap": "B scout only",
        "evidence": "No strong pair-specific lesson yet. Respect the tape more than memory.",
        "state": None,
        "trust_score": 0,
        "trade_count": int(display_stats.get("count", 0) or 0),
        "trade_ev": float(display_stats.get("ev", 0.0) or 0.0),
        "trade_wr": float(display_stats.get("win_rate", 0.0) or 0.0),
        "has_exact": False,
        "current_session": current_session,
        "current_regime": current_regime,
        "current_tape_bucket": current_tape_bucket,
        "pair_context": _format_feedback_stat(display_stats),
        "tape_context": _format_feedback_stat(
            tape_stat,
            _live_tape_bucket_label(current_tape_bucket),
        ),
        "recent_feedback_note": None,
        "session_stat": session_stat,
        "regime_stat": regime_stat,
        "tape_stat": tape_stat,
        "preferred_execution_style": preferred_style,
        "preferred_execution_style_stat": style_stat,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "preferred_execution_style_context": _format_feedback_stat(
            style_stat,
            _execution_style_context_label(preferred_style),
        ) if preferred_style else None,
        "context_bias": "neutral",
        "range_entry_price": target.get("range_entry_price"),
        "range_tp_price": target.get("range_tp_price"),
        "range_opposite_entry_price": target.get("range_opposite_entry_price"),
        "range_visual": target.get("range_visual"),
        "range_risk": target.get("range_risk"),
        "range_note": target.get("range_note"),
        "range_side": target.get("range_side"),
        "range_payoff_ratio": range_payoff_ratio,
        "range_target_pips": range_target_pips,
        "audit_entry_price": target.get("audit_entry_price"),
        "audit_tp_price": target.get("audit_tp_price"),
        "audit_why": target.get("audit_why"),
        "audit_edge": target.get("audit_edge"),
        "audit_allocation": target.get("audit_allocation"),
        "audit_held_status": target.get("audit_held_status"),
        "audit_opposite_entry_price": target.get("audit_opposite_entry_price"),
    }
    if not top and int(display_stats.get("count", 0) or 0) == 0 and source not in {"audit", "audit_range"}:
        return profile

    stats = {}
    for lesson in exact + pair_only:
        trade_stats = lesson.get("trade_stats") or {}
        if trade_stats.get("count"):
            stats = trade_stats
            break

    if not display_stats and stats:
        display_stats = stats
        if not feedback_stats:
            feedback_stats = stats

    trade_count = int(display_stats.get("count", 0) or 0)
    trade_ev = float(display_stats.get("ev", 0.0) or 0.0)
    trade_wr = float(display_stats.get("win_rate", 0.0) or 0.0)
    trust = int(top.get("trust_score", 0) or 0) if top else 0
    text_blob = _lesson_text_blob(relevant)
    no_edge = any(pattern in text_blob for pattern in NO_EDGE_PATTERNS)
    positive = any(pattern in text_blob for pattern in POSITIVE_EDGE_PATTERNS)
    pair_feedback = _recent_feedback_override(feedback_stats)
    session_score_delta, session_cap_delta, session_bias = _context_signal(session_stat)
    regime_score_delta, regime_cap_delta, regime_bias = _context_signal(regime_stat)
    tape_score_delta, tape_cap_delta, tape_bias = _context_signal(
        tape_stat,
        min_count=RECENT_FEEDBACK_MIN_COUNT,
    )
    style_score_delta, style_cap_delta, style_bias = _context_signal(
        style_stat,
        min_count=RECENT_FEEDBACK_MIN_COUNT,
    ) if preferred_style else (0, 0, "no sample")

    score = trust
    if exact:
        score += 12
    elif pair_only:
        score += 5
    if top and top.get("state") == "confirmed":
        score += 8
    elif top and top.get("state") == "watch":
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
    if pair_feedback:
        if pair_feedback["tier"] == "hard_headwind":
            score -= 24
        elif pair_feedback["tier"] == "soft_headwind":
            score -= 12
    if source == "audit_range":
        score += 14
        if current_regime in {"range", "quiet"}:
            score += 8
        if range_payoff_ratio >= 6.0:
            score += min(10, int(range_payoff_ratio))
        elif range_payoff_ratio >= 4.0:
            score += 4
        if range_target_pips >= 20.0:
            score += 3
    elif source == "audit":
        audit_edge = str(target.get("audit_edge") or "").upper()
        if audit_edge == "S":
            score += 14
        elif audit_edge == "A":
            score += 10
        elif audit_edge == "B":
            score += 6
        if target.get("audit_entry_price") is not None and target.get("audit_tp_price") is not None:
            score += 4
    score += session_score_delta + regime_score_delta + tape_score_delta + style_score_delta
    score = max(0, min(99, score))

    if pair_feedback and pair_feedback["tier"] == "hard_headwind":
        verdict = "no-edge / restricted"
        allocation_cap = CAP_LABELS["pass"]
    elif pair_feedback and pair_feedback["tier"] == "soft_headwind":
        verdict = "watch / unproven"
        allocation_cap = CAP_LABELS["b_only"]
    elif no_edge:
        verdict = "no-edge / restricted"
        allocation_cap = CAP_LABELS["b_only"]
    elif exact and top and top.get("state") == "confirmed" and trade_count >= 5 and trade_ev > 0 and trade_wr >= 0.50:
        verdict = "confirmed edge"
        allocation_cap = CAP_LABELS["as_confirmed"]
    elif exact and top and top.get("state") == "confirmed":
        verdict = "confirmed but session-dependent"
        allocation_cap = CAP_LABELS["a_max"]
    elif exact and top and top.get("state") == "watch" and trade_count >= 3 and trade_ev > 0:
        verdict = "watch edge"
        allocation_cap = CAP_LABELS["a_max"]
    elif exact and top and top.get("state") == "watch":
        verdict = "watch / unproven"
        allocation_cap = CAP_LABELS["b_only"]
    elif pair_only and trade_count >= 5 and trade_ev > 0:
        verdict = "pair memory positive, direction unproven"
        allocation_cap = CAP_LABELS["ba_max"]
    elif pair_only:
        verdict = "pair memory only"
        allocation_cap = CAP_LABELS["b_scout"]
    elif trade_count >= 5 and trade_ev > 0:
        verdict = "watch edge"
        allocation_cap = CAP_LABELS["ba_max"]
    elif positive:
        verdict = "positive lesson, thin sample"
        allocation_cap = CAP_LABELS["ba_max"]
    else:
        verdict = "limited history"
        allocation_cap = CAP_LABELS["b_scout"]

    if source == "audit_range" and not no_edge:
        if verdict in {"limited history", "pair memory only"}:
            verdict = "audit range seat"
    elif source == "audit" and not no_edge:
        if verdict in {"limited history", "pair memory only"}:
            verdict = "audit inventory seat"

    cap_rank = _cap_rank(allocation_cap)
    cap_rank += session_cap_delta + regime_cap_delta + tape_cap_delta + style_cap_delta
    if no_edge:
        cap_rank = min(cap_rank, 1)
    if pair_feedback:
        if pair_feedback["tier"] == "hard_headwind":
            cap_rank = min(cap_rank, 0)
        elif pair_feedback["tier"] == "soft_headwind":
            cap_rank = min(cap_rank, 1)
    if source == "audit_range" and not no_edge:
        cap_rank = max(cap_rank, 2)
        if range_payoff_ratio >= 6.0:
            cap_rank = max(cap_rank, 3)
    elif source == "audit" and not no_edge:
        audit_allocation = str(target.get("audit_allocation") or target.get("audit_edge") or "").upper()
        if audit_allocation in {"S", "A"}:
            cap_rank = max(cap_rank, 3)
        elif audit_allocation == "B":
            cap_rank = max(cap_rank, 2)
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
    if tape_bias == "headwind":
        context_flags.append(f"{_live_tape_bucket_label(current_tape_bucket)} headwind")
    elif tape_bias == "tailwind":
        context_flags.append(f"{_live_tape_bucket_label(current_tape_bucket)} tailwind")
    if context_flags:
        verdict = f"{verdict} | {' + '.join(context_flags)}"
    if preferred_style and style_bias == "tailwind":
        verdict = f"{verdict} | {_execution_style_context_label(preferred_style)} tailwind"
    elif preferred_style and style_bias == "headwind":
        verdict = f"{verdict} | {_execution_style_context_label(preferred_style)} headwind"

    evidence = _clip_text((top.get("title") or top.get("text")) if top else profile["evidence"])
    if source == "audit_range":
        audit_visual = _clip_text(str(target.get("range_visual") or ""), 180)
        audit_risk = _clip_text(str(target.get("range_risk") or ""), 120)
        evidence = _clip_text(
            " | ".join(
                bit for bit in (
                    audit_visual or "auditor flagged a fresh paid range rotation",
                    f"risk {audit_risk}" if audit_risk else "",
                )
                if bit
            ),
            180,
        )
    elif source == "audit":
        audit_why = _clip_text(str(target.get("audit_why") or ""), 180)
        if audit_why:
            evidence = audit_why
    context_bias = "neutral"
    if "headwind" in context_flags:
        context_bias = "headwind"
    elif "tailwind" in context_flags:
        context_bias = "tailwind"
    final_score = score
    if pair_feedback and pair_feedback["tier"] == "hard_headwind":
        final_score = min(final_score, 24)
    elif pair_feedback and pair_feedback["tier"] == "soft_headwind":
        final_score = min(final_score, 48)

    profile.update({
        "learning_score": final_score,
        "verdict": verdict,
        "allocation_cap": allocation_cap,
        "evidence": evidence,
        "state": top.get("state") if top else None,
        "trust_score": trust,
        "trade_count": trade_count,
        "trade_ev": trade_ev,
        "trade_wr": trade_wr,
        "has_exact": bool(exact),
        "pair_context": _format_feedback_stat(display_stats),
        "tape_context": _format_feedback_stat(
            tape_stat,
            _live_tape_bucket_label(current_tape_bucket),
        ),
        "recent_feedback_note": pair_feedback.get("reason") if pair_feedback else None,
        "session_stat": session_stat,
        "regime_stat": regime_stat,
        "tape_stat": tape_stat,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "context_bias": context_bias,
    })
    return profile


def _build_learning_edge_profiles(
    targets: list[dict],
    registry: dict,
    live_tape_map: dict[str, dict] | None = None,
) -> list[dict]:
    profiles = [
        _build_learning_profile(
            target,
            registry,
            (live_tape_map or {}).get(str(target.get("pair") or "")),
        )
        for target in targets
    ]
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


def _apply_recent_execution_style_guard(profile: dict, plan: dict) -> dict:
    guarded = dict(plan or {})
    style = str(guarded.get("style") or "PASS").upper()
    preserve_armed_style = bool(guarded.get("preserve_armed_style"))
    style_stat = _load_trade_context_stats().get("style", {}).get(
        (profile.get("pair"), profile.get("direction"), style)
    )
    guarded["execution_style_stat"] = style_stat
    guarded["execution_style_context"] = _format_feedback_stat(
        style_stat,
        _execution_style_context_label(style),
    )
    day_kill_stat = _same_day_style_kill_stat(profile.get("pair"), profile.get("direction"), style)
    guarded["execution_style_day_kill"] = day_kill_stat

    if day_kill_stat:
        if style == "MARKET":
            fallback = _tape_fallback_style(str(profile.get("current_regime") or "").lower() or None)
            reason = (
                f"{_format_day_style_guard(style, day_kill_stat, str(day_kill_stat.get('scope') or 'today'))}, "
                "so the market-chase lane is closed until it captures a seat again"
            )
        else:
            fallback = "LIMIT"
            reason = (
                f"{_format_day_style_guard(style, day_kill_stat, str(day_kill_stat.get('scope') or 'today'))}, "
                "so keep the thesis alive only with a better-price LIMIT for the rest of today"
            )
        if preserve_armed_style:
            guarded["note"] = (
                f"{guarded.get('note', '').strip()}; {reason}; existing armed order stays review-only until you "
                "explicitly cancel or replace it"
            ).strip("; ")
            guarded["execution_style_feedback_note"] = reason
            return guarded
        guarded["style"] = fallback
        guarded["orderability"] = fallback
        guarded["note"] = f"{guarded.get('note', '').strip()}; {reason}".strip("; ")
        guarded["execution_style_feedback_note"] = reason
        return guarded

    if style not in {"MARKET", "STOP-ENTRY"} or not style_stat or style_stat.get("source") != "pretrade_recent":
        return guarded

    count = int(style_stat.get("count", 0) or 0)
    min_count = 2 if style == "MARKET" else 3
    if count < min_count:
        return guarded

    ev = float(style_stat.get("ev", 0.0) or 0.0)
    win_rate = float(style_stat.get("win_rate", 0.0) or 0.0)
    if ev >= 0 or win_rate > 0.40:
        return guarded

    if style == "MARKET":
        fallback = _tape_fallback_style(str(profile.get("current_regime") or "").lower() or None)
        reason = (
            f"{_execution_style_context_label(style)} is still losing here "
            f"({_format_feedback_stat(style_stat, style)}), so do not pay market until the lane repairs"
        )
    else:
        fallback = "LIMIT"
        reason = (
            f"{_execution_style_context_label(style)} is still losing here "
            f"({_format_feedback_stat(style_stat, style)}), so keep the lane alive with a better-price LIMIT "
            "instead of a proof-chase trigger"
        )
    if preserve_armed_style:
        guarded["note"] = (
            f"{guarded.get('note', '').strip()}; {reason}; existing armed order stays review-only until you "
            "explicitly cancel or replace it"
        ).strip("; ")
        guarded["execution_style_feedback_note"] = reason
        return guarded
    guarded["style"] = fallback
    guarded["orderability"] = fallback
    guarded["note"] = f"{guarded.get('note', '').strip()}; {reason}".strip("; ")
    guarded["execution_style_feedback_note"] = reason
    return guarded


def _bayesian_update_hint(profile: dict) -> str:
    verdict = str(profile.get("verdict", ""))
    if "no-edge" in verdict:
        return "One clean win is not enough to expand size. Require repeated clean evidence before promoting above B."
    if "confirmed edge" in verdict:
        return "One loss does not kill the market-state prior. But a failed exact trigger/vehicle stays blocked until the tape prints a materially new state change."
    if "watch" in verdict or "pair memory positive" in verdict:
        return "A clean trigger win upgrades live confidence today; a failed trigger keeps this in watch/B lane and blocks the same exact trigger until the tape changes."
    return "Treat the next outcome as candidate evidence only. Do not rewrite the market-state story from one print."


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


def _price_delta_from_pips(pair: str, pips: float) -> float:
    return float(pips) / float(_pip_factor(pair))


def _profile_counter_hint(profile: dict) -> bool:
    blob = " ".join(
        part
        for part in (
            str(profile.get("recipe") or ""),
            str(profile.get("seat_family") or ""),
            str(profile.get("range_note") or ""),
        )
        if part
    ).lower()
    return any(token in blob for token in ("counter", "reversal", "mean revert", "mean_revert", "fade"))


def _profile_command_order_type(profile: dict) -> str | None:
    style = _profile_default_orderability(profile)
    if style == "ENTER NOW":
        return "MARKET"
    if style in {"MARKET", "LIMIT", "STOP-ENTRY"}:
        return style
    return None


def _profile_command_allocation(profile: dict) -> tuple[str, str | None]:
    label = str(profile.get("exact_pretrade_label") or "").upper()
    matches = re.findall(r"(B\+|B0|B-|A|S|C)", label)
    band = matches[-1] if matches else str(profile.get("allocation_band") or "").upper()
    if band in {"B+", "B0", "B-"}:
        return "B", band
    if band in {"A", "S", "C"}:
        return band, None
    allocation = str(profile.get("audit_allocation") or "").upper()
    if allocation in {"A", "S", "B", "C"}:
        return allocation, band if band in {"B+", "B0", "B-"} else None
    if band.startswith("B"):
        return "B", band
    return "C", None


def _profile_command_units(profile: dict) -> int:
    label = str(profile.get("exact_pretrade_label") or "").upper()
    matches = re.findall(r"(B\+|B0|B-|A|S|C)", label)
    band = matches[-1] if matches else str(profile.get("allocation_band") or "").upper()
    if band in ORDER_COMMAND_UNIT_FLOORS:
        return ORDER_COMMAND_UNIT_FLOORS[band]
    allocation, band_override = _profile_command_allocation(profile)
    if band_override in ORDER_COMMAND_UNIT_FLOORS:
        return ORDER_COMMAND_UNIT_FLOORS[band_override]
    return ORDER_COMMAND_UNIT_FLOORS.get(allocation, 1000)


def _profile_command_receipt_hint(profile: dict) -> str:
    order_type = _profile_command_order_type(profile)
    if order_type == "MARKET":
        return "ENTER NOW already filled as trade id=___"
    if order_type == "STOP-ENTRY":
        return "armed STOP id=___"
    if order_type == "LIMIT":
        return "armed LIMIT id=___"
    return "dead thesis because ___"


def _profile_order_command(profile: dict) -> str | None:
    if str(profile.get("source") or "").lower() == "pending":
        return None
    if str(profile.get("exact_pretrade_status") or "").lower() not in {"ok", "advisory"}:
        return None

    order_type = _profile_command_order_type(profile)
    if order_type not in {"MARKET", "LIMIT", "STOP-ENTRY"}:
        return None

    pair = str(profile.get("pair") or "").upper()
    direction = str(profile.get("direction") or "").upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return None

    tp_price = profile.get("exact_tp_price")
    sl_price = profile.get("exact_sl_price")
    if tp_price in {None, ""} or sl_price in {None, ""}:
        return None
    entry_price = profile.get("exact_entry_price")
    if order_type != "MARKET" and entry_price in {None, ""}:
        return None

    precision = 3 if pair.endswith("JPY") else 5
    units = _profile_command_units(profile)
    allocation, band = _profile_command_allocation(profile)
    thesis = _normalize_identity_fragment(
        f"{pair}_{direction}_{profile.get('seat_family') or profile.get('recipe') or order_type.lower()}",
        limit=96,
    )
    pretrade = str(profile.get("exact_pretrade_label") or profile.get("exact_pretrade_style") or "B(0/10)").strip()

    command_parts = [
        "python3",
        "tools/place_trader_order.py",
        order_type,
        pair,
        direction,
        str(units),
    ]
    if order_type != "MARKET":
        command_parts.extend(["--entry", f"{float(entry_price):.{precision}f}"])
    command_parts.extend(
        [
            "--tp",
            f"{float(tp_price):.{precision}f}",
            "--sl",
            f"{float(sl_price):.{precision}f}",
            "--thesis",
            thesis,
            "--pretrade",
            f"'{pretrade}'",
            "--allocation",
            allocation,
        ]
    )
    if band:
        command_parts.extend(["--allocation-band", band])
    if _profile_counter_hint(profile):
        command_parts.append("--counter")
    return " ".join(command_parts)


def _profile_proxy_stop_pips(profile: dict, target_pips: float, spread_pips: float | None) -> float:
    pair = str(profile.get("pair") or "")
    regime = str(profile.get("current_regime") or "").lower()
    tech = _load_technicals(ROOT, pair) if pair else {}
    m5 = tech.get("M5", {}) if tech else {}
    atr_pips = float(m5.get("atr_pips", 0.0) or 0.0)
    spread_floor = max(float(spread_pips or 0.0) * 5.0, 4.0)

    if str(profile.get("source") or "") == "audit_range":
        structural = max(target_pips * 0.45, atr_pips * 0.55)
    elif _profile_counter_hint(profile):
        structural = max(target_pips * 0.70, atr_pips * 0.70)
    elif regime == "trending":
        structural = max(target_pips * 0.60, atr_pips * 0.80)
    elif regime in {"transition", "squeeze"}:
        structural = max(target_pips * 0.65, atr_pips * 0.75)
    elif regime in {"range", "quiet"}:
        structural = max(target_pips * 0.50, atr_pips * 0.60)
    else:
        structural = max(target_pips * 0.60, atr_pips * 0.70)

    return round(max(spread_floor, structural), 1)


def _profile_proxy_geometry(
    profile: dict,
    style: str | None,
    market: dict | None,
    spread_pips: float | None,
) -> dict | None:
    pair = str(profile.get("pair") or "")
    direction = str(profile.get("direction") or "").upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return None

    normalized_style = str(style or _profile_default_expression(profile) or "PASS").upper()
    if normalized_style == "PASS":
        return None

    target_pips = (
        float(profile.get("range_target_pips") or 0.0)
        if str(profile.get("source") or "") == "audit_range"
        else 0.0
    )
    if target_pips <= 0:
        target_pips = _execution_target_pips(profile)
    if _profile_counter_hint(profile):
        target_pips = min(target_pips, 10.0)

    stop_pips = _profile_proxy_stop_pips(profile, target_pips, spread_pips)
    target_delta = _price_delta_from_pips(pair, target_pips)
    stop_delta = _price_delta_from_pips(pair, stop_pips)
    bid = float((market or {}).get("bid")) if (market or {}).get("bid") is not None else None
    ask = float((market or {}).get("ask")) if (market or {}).get("ask") is not None else None
    mid = float((market or {}).get("mid")) if (market or {}).get("mid") is not None else None
    anchor = mid if mid is not None else (ask if direction == "LONG" else bid)

    if str(profile.get("source") or "") == "audit_range":
        entry_price = profile.get("range_entry_price")
        tp_price = profile.get("range_tp_price")
        if entry_price is None or tp_price is None:
            return None
        entry = float(entry_price)
        tp = float(tp_price)
        sl = entry - stop_delta if direction == "LONG" else entry + stop_delta
        return {
            "entry_price": entry,
            "tp_price": tp,
            "sl_price": sl,
            "target_pips": target_pips,
            "stop_pips": stop_pips,
            "anchor": "audit_range",
        }

    if str(profile.get("source") or "") == "audit":
        entry_price = profile.get("audit_entry_price")
        tp_price = profile.get("audit_tp_price")
        if entry_price is not None and tp_price is not None:
            entry = float(entry_price)
            tp = float(tp_price)
            sl = entry - stop_delta if direction == "LONG" else entry + stop_delta
            return {
                "entry_price": entry,
                "tp_price": tp,
                "sl_price": sl,
                "target_pips": abs(tp - entry) * _pip_factor(pair),
                "stop_pips": stop_pips,
                "anchor": "audit_narrative",
            }

    if anchor is None:
        return None

    if normalized_style == "MARKET":
        entry = ask if direction == "LONG" else bid
        if entry is None:
            entry = anchor
    elif normalized_style == "LIMIT":
        offset_pips = max(target_pips * 0.30, float(spread_pips or 0.0) * 2.0, 2.0)
        offset = _price_delta_from_pips(pair, offset_pips)
        entry = anchor - offset if direction == "LONG" else anchor + offset
    elif normalized_style == "STOP-ENTRY":
        offset_pips = max(target_pips * 0.20, float(spread_pips or 0.0) * 2.0, 1.5)
        offset = _price_delta_from_pips(pair, offset_pips)
        entry = anchor + offset if direction == "LONG" else anchor - offset
    else:
        entry = anchor

    tp = entry + target_delta if direction == "LONG" else entry - target_delta
    sl = entry - stop_delta if direction == "LONG" else entry + stop_delta
    return {
        "entry_price": entry,
        "tp_price": tp,
        "sl_price": sl,
        "target_pips": target_pips,
        "stop_pips": stop_pips,
        "anchor": "live_price",
    }


def _reprice_proxy_geometry(
    *,
    pair: str,
    direction: str,
    entry_price: float,
    target_pips: float,
    stop_pips: float,
    anchor: str,
    repair_note: str | None = None,
) -> dict:
    target_delta = _price_delta_from_pips(pair, target_pips)
    stop_delta = _price_delta_from_pips(pair, stop_pips)
    tp = entry_price + target_delta if direction == "LONG" else entry_price - target_delta
    sl = entry_price - stop_delta if direction == "LONG" else entry_price + stop_delta
    return {
        "entry_price": float(entry_price),
        "tp_price": float(tp),
        "sl_price": float(sl),
        "target_pips": float(target_pips),
        "stop_pips": float(stop_pips),
        "anchor": anchor,
        "repair_note": repair_note or "",
    }


def _run_exact_pretrade_for_proxy(
    profile: dict,
    proxy: dict,
    spread_pips: float | None,
    live_tape: dict | None,
) -> dict:
    return run_exact_pretrade(
        pair=str(profile.get("pair") or ""),
        direction=str(profile.get("direction") or ""),
        entry_price=float(proxy["entry_price"]),
        tp_price=float(proxy["tp_price"]),
        sl_price=float(proxy["sl_price"]),
        counter=_profile_counter_hint(profile),
        regime=str(profile.get("current_regime") or "") or None,
        spread_pips=spread_pips,
        live_tape=live_tape,
    )


def _repair_proxy_geometry(
    profile: dict,
    proxy: dict,
    exact_result: dict | None,
    spread_pips: float | None,
) -> dict | None:
    plan = (exact_result or {}).get("execution_plan") or {}
    setup = (exact_result or {}).get("setup_quality") or {}
    pair = str(profile.get("pair") or "")
    direction = str(profile.get("direction") or "").upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return None

    current_target = float(proxy.get("target_pips") or 0.0)
    current_stop = float(proxy.get("stop_pips") or 0.0)
    repaired_target = current_target
    repaired_stop = current_stop
    repairs: list[str] = []

    if plan.get("requires_stop_widening"):
        required_stop = max(
            current_stop,
            float(setup.get("noise_stop_floor_pips") or 0.0),
            float(spread_pips or 0.0) * 4.0,
        )
        required_stop = round(required_stop, 1)
        if required_stop > repaired_stop + 0.05:
            repairs.append(f"SL {current_stop:.1f}->{required_stop:.1f}pip to clear noise floor")
            repaired_stop = required_stop

    explicit_target = (
        str(profile.get("source") or "") in {"audit", "audit_range"}
        and (
            profile.get("audit_tp_price") is not None
            or profile.get("range_tp_price") is not None
        )
    )
    if plan.get("requires_better_payoff") and not explicit_target:
        min_target = max(
            current_target,
            float(spread_pips or 0.0) * 4.25,
            4.0 if pair.endswith("JPY") else 2.5,
        )
        min_target = round(min_target, 1)
        if min_target > repaired_target + 0.05:
            repairs.append(f"TP {current_target:.1f}->{min_target:.1f}pip to clear friction floor")
            repaired_target = min_target

    if not repairs:
        return None

    return _reprice_proxy_geometry(
        pair=pair,
        direction=direction,
        entry_price=float(proxy["entry_price"]),
        target_pips=repaired_target,
        stop_pips=repaired_stop,
        anchor=str(proxy.get("anchor") or "repaired_proxy"),
        repair_note="; ".join(repairs),
    )


def _merge_execution_notes(*notes: str | None) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for note in notes:
        text = " ".join(str(note or "").split()).strip()
        if not text or text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return "; ".join(merged)


def _apply_exact_pretrade_profile_guard(
    profile: dict,
    plan: dict,
    market: dict | None,
    spread_pips: float | None,
    live_tape: dict | None,
) -> dict:
    guarded = dict(plan or {})
    if bool(guarded.get("preserve_armed_style")):
        return guarded

    proxy = _profile_proxy_geometry(profile, guarded.get("style"), market, spread_pips)
    if not proxy:
        guarded["exact_pretrade_status"] = "skipped"
        guarded["exact_pretrade_note"] = "proxy geometry unavailable"
        return guarded

    try:
        exact_result = _run_exact_pretrade_for_proxy(profile, proxy, spread_pips, live_tape)
    except Exception as exc:
        guarded["exact_pretrade_status"] = "error"
        guarded["exact_pretrade_note"] = f"exact pretrade proxy failed: {exc}"
        return guarded

    exact_style = exact_execution_style(exact_result)
    requested_style = str(guarded.get("style") or "PASS").upper()
    hard_blockers = exact_pretrade_hard_blockers(exact_result)
    repair_note = ""
    if hard_blockers:
        repaired_proxy = _repair_proxy_geometry(profile, proxy, exact_result, spread_pips)
        if repaired_proxy:
            try:
                repaired_result = _run_exact_pretrade_for_proxy(
                    profile,
                    repaired_proxy,
                    spread_pips,
                    live_tape,
                )
            except Exception:
                repaired_result = None
            if repaired_result is not None:
                repaired_hard_blockers = exact_pretrade_hard_blockers(repaired_result)
                if not repaired_hard_blockers or len(repaired_hard_blockers) < len(hard_blockers):
                    proxy = repaired_proxy
                    exact_result = repaired_result
                    exact_style = exact_execution_style(exact_result)
                    hard_blockers = repaired_hard_blockers
                    repair_note = str(repaired_proxy.get("repair_note") or "")
    exact_advisories = exact_pretrade_advisories(
        requested_style=requested_style if requested_style in {"MARKET", "STOP-ENTRY", "LIMIT"} else None,
        result=exact_result,
    )
    final_style = requested_style
    final_note = _merge_execution_notes(
        guarded.get("note"),
        repair_note,
        f"exact {exact_pretrade_label(exact_result)}",
        exact_execution_note(exact_result),
    )

    if hard_blockers:
        final_style = "PASS"
        final_note = _merge_execution_notes(final_note, *hard_blockers)

    if str(profile.get("source") or "") == "audit_range" and final_style != "PASS":
        final_style = "LIMIT"
        final_note = _merge_execution_notes(
            final_note,
            "range rail stays passive even when exact pretrade says the box is live",
        )
    elif str(profile.get("source") or "") == "audit" and profile.get("audit_entry_price") is not None and market:
        anchor_mid = (market or {}).get("mid")
        if anchor_mid is not None:
            anchor_mid = float(anchor_mid)
            planned_entry = float(profile.get("audit_entry_price"))
            threshold = _price_delta_from_pips(
                str(profile.get("pair") or ""),
                max(float(spread_pips or 0.0) * 1.5, 1.0),
            )
            relation_style = None
            if str(profile.get("direction") or "").upper() == "LONG":
                if planned_entry < anchor_mid - threshold:
                    relation_style = "LIMIT"
                elif planned_entry > anchor_mid + threshold:
                    relation_style = "STOP-ENTRY"
            else:
                if planned_entry > anchor_mid + threshold:
                    relation_style = "LIMIT"
                elif planned_entry < anchor_mid - threshold:
                    relation_style = "STOP-ENTRY"
            if relation_style and final_style != "PASS":
                final_style = relation_style
                final_note = _merge_execution_notes(
                    final_note,
                    f"audit inventory entry {planned_entry} is still a {relation_style} seat versus live price",
                )

    guarded["style"] = final_style
    guarded["orderability"] = "ENTER NOW" if final_style == "MARKET" else final_style
    guarded["note"] = final_note
    guarded["target_pips"] = float(proxy["target_pips"])
    guarded["exact_pretrade_status"] = (
        "hard_blocked" if hard_blockers else ("advisory" if exact_advisories else "ok")
    )
    guarded["exact_pretrade_style"] = exact_style
    guarded["exact_pretrade_label"] = exact_pretrade_label(exact_result)
    guarded["exact_pretrade_note"] = exact_execution_note(exact_result)
    guarded["exact_pretrade_hard_blockers"] = hard_blockers
    guarded["exact_pretrade_advisories"] = exact_advisories
    guarded["exact_entry_price"] = float(proxy["entry_price"])
    guarded["exact_tp_price"] = float(proxy["tp_price"])
    guarded["exact_sl_price"] = float(proxy["sl_price"])
    guarded["exact_stop_pips"] = float(proxy["stop_pips"])
    guarded["exact_proxy_anchor"] = str(proxy["anchor"])
    guarded["exact_proxy_repair_note"] = repair_note
    return guarded


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
                "STOP-ENTRY",
                "historical headwind plus sub-ideal conviction does not deserve fresh market risk; arm the trigger first and leave one reload LIMIT",
            )
        if cap_rank == 1 and learning_score >= 58:
            return (
                "STOP-ENTRY",
                "even with tight spread, B-only memory is not enough for a blind market scout; require trigger proof first",
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
    market_allowed = cap_rank >= 4
    reasons: list[str] = []

    if source == "pending":
        armed_style = str(profile.get("armed_style") or "LIMIT").upper()
        reasons.append(f"already armed as {armed_style}; leave it alone unless the thesis is dead")
        return {
            "style": armed_style,
            "orderability": armed_style,
            "note": "; ".join(reasons),
            "spread_ratio": spread_ratio,
            "target_pips": target_pips,
            "preserve_armed_style": True,
        }

    if cap == CAP_LABELS["pass"]:
        style, note = _pass_cap_pressure_participation_plan(
            profile,
            normal_market_spread=normal_market_spread,
        )
        if style and note:
            reasons.append(note)
            return {
                "style": style,
                "orderability": style,
                "note": "; ".join(reasons),
                "spread_ratio": spread_ratio,
                "target_pips": target_pips,
            }
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
        if source == "audit_range":
            entry_price = profile.get("range_entry_price")
            tp_price = profile.get("range_tp_price")
            target_pips = float(profile.get("range_target_pips", 0.0) or 0.0)
            payoff = float(profile.get("range_payoff_ratio", 0.0) or 0.0)
            if entry_price is not None and tp_price is not None:
                reasons.append(
                    f"audit range level {entry_price} -> TP {tp_price}"
                    + (
                        f" ({target_pips:.0f}pip, {payoff:.0f}x spread)"
                        if target_pips > 0 and payoff > 0
                        else ""
                    )
                )
            risk = str(profile.get("range_risk") or "").strip()
            if risk:
                reasons.append(f"range risk: {risk}")
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
                f"{regime} is leaning enough to keep a trigger alive, but fresh market risk stays reserved for A/S-cap seats"
            )
            return {
                "style": "STOP-ENTRY",
                "orderability": "STOP-ENTRY",
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
                "live tape is leaning, but B-cap seats still need trigger proof instead of a blind market scout"
            )
            return {
                "style": "STOP-ENTRY",
                "orderability": "STOP-ENTRY",
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

    if not market_allowed:
        reasons.append("fresh market risk is reserved for A/S-cap seats; this lane must stay trigger- or price-improvement-only")
        style = "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"
        return {
            "style": style,
            "orderability": style,
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
    return load_technicals_timeframes(f)


def _pair_technical_age_map(pair: str, now_utc: datetime) -> dict[str, float | None]:
    timeframes = _load_technicals(ROOT, pair)
    return {
        tf: timeframe_age_minutes(timeframes, tf, now_utc)
        for tf in TECHNICAL_STALE_LIMITS
    }


def _technical_stale_reasons(age_map: dict[str, float | None]) -> list[str]:
    reasons = []
    for tf, limit in TECHNICAL_STALE_LIMITS.items():
        age = age_map.get(tf)
        if age is None:
            reasons.append(f"{tf}:missing")
        elif age > limit:
            reasons.append(f"{tf}:{age:.0f}m>{limit:.0f}m")
    return reasons


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
    emit_templates = "--emit-templates" in sys.argv

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
    section("TRADE EVENT SYNC")
    sync_out = run_trade_event_sync()
    if sync_out:
        print(sync_out)
    else:
        print("TRADE_EVENT_SYNC no output")
    state_text = state_path.read_text() if state_path.exists() else ""
    carry_targets = _parse_state_carry_targets(state_text)
    focus_snapshot = _parse_state_focus_snapshot(state_text)
    hot_updates = _parse_hot_updates(state_text)
    if hot_updates:
        section("HOT UPDATES FROM LAST SESSION")
        for update in hot_updates:
            print(f"- {update}")
    if carry_targets:
        section("STATE CARRY-FORWARD WATCHLIST")
        for item in carry_targets:
            print(f"{item['pair']} {item['direction']} [{str(item.get('source', '?')).upper()}] | {item.get('recipe')}")
    section("CARRY-FORWARD FOCUS (protect concentration before scanning new seats)")
    if focus_snapshot:
        print(
            "Copy this ladder into Market Narrative before you re-rank fresh risk. "
            "A prettier board seat is not enough to replace the old primary."
        )
        for prefix in (
            "Best expression NOW:",
            "Primary vehicle:",
            "Primary vehicle shelf-life now:",
            "Backup vehicle:",
            "Backup vehicle shelf-life now:",
            "Second-best expression:",
            "Next fresh risk allowed NOW:",
            "Next fresh risk shelf-life now:",
            "20-minute backup trigger armed NOW:",
            "15-minute backup trigger armed NOW:",
            "Best direct-USD seat NOW:",
        ):
            value = focus_snapshot.get(prefix)
            if value:
                display_prefix = "20-minute backup trigger armed NOW:" if prefix == "15-minute backup trigger armed NOW:" else prefix
                print(f"{display_prefix} {value}")
        print("Primary continuity verdict: [KEEP / ROTATE / DEAD]")
        print("  If KEEP: next unit of risk stays with ___ because ___")
        print("  If ROTATE: exact contradiction that killed the old primary = ___")
        print("  Replacement primary: ___ because ___")
        print("Backup continuity: [still armed / arm now / dead] because ___")
    else:
        print("No carry-forward focus found in state.md. Build a fresh primary / backup / next-risk ladder this session.")

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

    section("TECHNICAL CACHE FRESHNESS")
    now_for_tech = datetime.now(timezone.utc)
    tech_age_map = {pair: _pair_technical_age_map(pair, now_for_tech) for pair in PAIRS}
    stale_pairs = []
    for pair in PAIRS:
        age_map = tech_age_map[pair]
        bits = []
        for tf in ("M1", "M5", "M15", "H1", "H4"):
            age = age_map.get(tf)
            bits.append(f"{tf}={age:.0f}m" if age is not None else f"{tf}=missing")
        reasons = _technical_stale_reasons(age_map)
        if reasons:
            stale_pairs.append(pair)
            print(f"{pair}: {' | '.join(bits)} ⚠️ stale ({', '.join(reasons)})")
        else:
            print(f"{pair}: {' | '.join(bits)}")
    if stale_pairs:
        retry_out = run_script(
            [VENV_PYTHON, "tools/refresh_factor_cache.py", *stale_pairs, "--quiet"],
            timeout=30,
        )
        print(f"Retried stale technical caches for: {', '.join(stale_pairs)}")
        if retry_out:
            print(retry_out[:200])
        now_for_tech = datetime.now(timezone.utc)
        still_stale = []
        for pair in stale_pairs:
            reasons = _technical_stale_reasons(_pair_technical_age_map(pair, now_for_tech))
            if reasons:
                still_stale.append(f"{pair} ({', '.join(reasons)})")
        if still_stale:
            print("⚠️ technical cache still stale after retry: " + "; ".join(still_stale))

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
    price_map = {}  # pair -> bid/ask/mid
    spread_data = {}  # pair -> spread_pips (referenced in other sections)
    live_tape_map: dict[str, dict] = {}
    try:
        prices = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={','.join(PAIRS)}", cfg)
        for p in prices.get("prices", []):
            pair = p["instrument"]
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
            price_map[pair] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
            pip_factor = _pip_factor(pair)
            spread_pip = (ask - bid) * pip_factor
            spread_data[pair] = spread_pip
            warn = " ⚠️ spread wide" if spread_pip > 2.0 else ""
            print(
                f"{pair} bid={p['bids'][0]['price']} ask={p['asks'][0]['price']} Sp={spread_pip:.1f}pip{warn}"
            )
    except Exception as e:
        print(f"ERROR: {e}")

    section("LIVE TAPE PROBE (pricing microstructure)")
    try:
        probe = probe_market(cfg, pairs=PAIRS, samples=10, interval_sec=0.45, write_cache=True)
        print(
            f"probe mode: {probe.get('mode_used')} "
            f"(duration {float(probe.get('duration_sec', 0.0)):.2f}s)"
        )
        for pair in PAIRS:
            summary = dict((probe["pairs"].get(pair) or {}))
            summary["probe_mode"] = probe.get("mode_used") or probe.get("mode_requested") or "?"
            live_tape_map[pair] = summary
            if summary.get("tape") == "unavailable":
                print(f"{pair}: unavailable ({summary.get('error', 'no data')})")
                continue
            print(
                f"{pair}: {summary.get('bias')} | tape={summary.get('tape')} | "
                f"move={float(summary.get('delta_pips', 0.0)):+.1f}pip | "
                f"range={float(summary.get('range_pips', 0.0)):.1f}pip | "
                f"spread avg/max={float(summary.get('avg_spread_pips', 0.0)):.1f}/"
                f"{float(summary.get('max_spread_pips', 0.0)):.1f}pip | "
                f"ticks {summary.get('upticks', 0)}/{summary.get('downticks', 0)}/{summary.get('flats', 0)}"
            )
        if probe.get("errors"):
            print("probe warnings:")
            for error in probe["errors"][:5]:
                print(f"  - {error}")
    except Exception as e:
        print(f"(pricing probe error: {e})")

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

    # Churn detection: today's actual fills + entry orders per pair
    entry_activity = _today_entry_activity(now_utc)
    entry_counts = entry_activity.get("pair_counts") or Counter()
    if entry_counts:
        parts = [f"{p}×{c}" for p, c in entry_counts.most_common()]
        total = sum(entry_counts.values())
        churn_warn = ""
        max_pair_count = entry_counts.most_common(1)[0][1] if entry_counts else 0
        if max_pair_count >= 4:
            churn_warn = " ⚠️ churn risk — same pair 4+ times"
        print(
            f"Today's entry activity: {' '.join(parts)} | total {total}"
            f" | {_entry_activity_summary(entry_activity)}{churn_warn}"
        )

    # 2b. Pending Orders (limit orders, TP/SL check)
    section("PENDING ORDERS")
    try:
        if pending_orders:
            for o in pending_orders:
                otype = o.get("type", "?")
                pair = o.get("instrument", "?")
                direction = _units_to_direction(o.get("units")) or "?"
                units = o.get("units", "?")
                price = o.get("price", "?")
                gtd = o.get("gtdTime", "GTC")[:16] if o.get("gtdTime") else "GTC"
                metrics = _pending_order_metrics(o, price_map, spread_data, now_utc)
                bits = [f"{otype} {pair} {direction} {units}u @{price} exp={gtd} id={o.get('id', '?')}"]
                if metrics["age_min"] is not None:
                    bits.append(f"age={metrics['age_min']:.0f}m")
                if metrics["ttl_min"] is not None:
                    bits.append(f"ttl={metrics['ttl_min']:.0f}m")
                if metrics["gap_pips"] is not None:
                    gap_bits = [f"gap={metrics['gap_pips']:.1f}pip {metrics['gap_relation']}"]
                    atr_pips = metrics["atr_pips"]
                    spread_pips = metrics["spread_pips"]
                    if atr_pips:
                        gap_bits.append(f"{metrics['gap_pips']/atr_pips:.1f}x ATR")
                    if spread_pips:
                        gap_bits.append(f"{metrics['gap_pips']/spread_pips:.1f}x spread")
                    gap_text = gap_bits[0]
                    if len(gap_bits) > 1:
                        gap_text += f" ({' / '.join(gap_bits[1:])})"
                    bits.append(gap_text)
                bits.append(metrics["status"])
                print(" | ".join(bits))
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
                window_end = now_utc + timedelta(hours=4)
                upcoming = []
                for ev in cal:
                    ev_dt = parse_cache_timestamp(ev.get("time"))
                    if ev_dt is None:
                        continue
                    if now_utc <= ev_dt <= window_end:
                        upcoming.append((ev_dt, ev))
                upcoming.sort(key=lambda item: item[0])
                if upcoming:
                    for ev_dt, ev in upcoming[:10]:
                        title = ev.get("event", "")
                        impact = ev.get("impact", "")
                        country = ev.get("country", "")
                        if not title:
                            continue
                        impact_str = f" ({impact} impact)" if impact else ""
                        ccy_str = f" — {country}" if country else ""
                        delta_min = max(0, int((ev_dt - now_utc).total_seconds() // 60))
                        print(
                            f"{ev_dt.astimezone(JST).strftime('%m/%d %H:%M JST')} "
                            f"{title}{impact_str}{ccy_str} [in {delta_min}m]"
                        )
                else:
                    print("(no scheduled events in next 4h)")
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

    # 6. Memory recall (held + pending + audit pressure + top scanner candidates)
    audit_context = _load_audit_narrative_context()
    missed_pressure = _load_recent_missed_seat_pressure()
    audit_targets = _build_audit_targets(audit_context, missed_pressure)
    memory_targets = _build_memory_targets(
        trades_data,
        pending_orders,
        scanner_out,
        price_map,
        spread_data,
        now_utc,
        carry_targets,
        audit_context.get("range_targets"),
        audit_targets,
    )
    registry = _load_lesson_registry()
    learning_profiles = _build_learning_edge_profiles(memory_targets, registry, live_tape_map)
    for profile in learning_profiles:
        if profile.get("source") != "held":
            _apply_profile_promotion_pressure(profile, audit_context, missed_pressure)
    memory_results = _run_actionable_memory_recall(memory_targets)
    for profile in learning_profiles:
        pair = str(profile.get("pair", ""))
        live_tape = live_tape_map.get(pair) or {
            "pair": pair,
            "samples": 0,
            "bias": "unknown",
            "tape": "unavailable",
            "error": "probe unavailable in this session",
            "probe_mode": "missing",
        }
        plan = _recommend_profile_execution(profile, spread_data.get(pair))
        plan = _apply_exact_pretrade_profile_guard(
            profile,
            plan,
            price_map.get(pair),
            spread_data.get(pair),
            live_tape,
        )
        plan = _apply_live_tape_profile_guard(profile, plan, live_tape)
        plan = _apply_recent_execution_style_guard(profile, plan)
        profile["default_expression"] = plan.get("style")
        profile["default_orderability"] = plan.get("orderability")
        profile["execution_style"] = plan.get("style")
        profile["orderability"] = plan.get("orderability")
        profile["execution_note"] = plan.get("note")
        profile["execution_target_pips"] = plan.get("target_pips")
        profile["execution_spread_ratio"] = plan.get("spread_ratio")
        profile["live_tape"] = plan.get("live_tape") or live_tape
        profile["live_tape_note"] = plan.get("live_tape_note") or _live_tape_brief(live_tape)
        profile["live_tape_rank"] = _profile_live_tape_rank(profile)
        profile["execution_style_stat"] = plan.get("execution_style_stat")
        profile["execution_style_context"] = plan.get("execution_style_context")
        profile["execution_style_feedback_note"] = plan.get("execution_style_feedback_note")
        profile["exact_pretrade_status"] = plan.get("exact_pretrade_status")
        profile["exact_pretrade_style"] = plan.get("exact_pretrade_style")
        profile["exact_pretrade_label"] = plan.get("exact_pretrade_label")
        profile["exact_pretrade_note"] = plan.get("exact_pretrade_note")
        profile["exact_pretrade_hard_blockers"] = plan.get("exact_pretrade_hard_blockers") or []
        profile["exact_pretrade_advisories"] = plan.get("exact_pretrade_advisories") or []
        profile["exact_entry_price"] = plan.get("exact_entry_price")
        profile["exact_tp_price"] = plan.get("exact_tp_price")
        profile["exact_sl_price"] = plan.get("exact_sl_price")
        profile["exact_stop_pips"] = plan.get("exact_stop_pips")
        profile["exact_proxy_anchor"] = plan.get("exact_proxy_anchor")
        allocation_band, band_reason = _profile_allocation_band(profile)
        profile["allocation_band"] = allocation_band
        profile["allocation_band_reason"] = band_reason
        profile["band_size_note"] = ALLOCATION_BAND_SIZE_TEXT.get(allocation_band, "size by grade")
        profile.update(_lane_shelf_life(profile, now_utc))
    learning_profiles = _consolidate_cross_source_profiles(learning_profiles)
    best_direct = None
    best_cross = None
    best_usdjpy = None
    market_now_profiles: list[dict] = []
    multi_vehicle_lanes: list[dict] = []
    session_intent = {"mode": "FULL_TRADER", "reasons": []}
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
                f"{profile['pair']} {profile['direction']} [{_profile_source_label(profile)}]{recipe} "
                f"| learning {int(profile.get('learning_score', 0))}/100 "
                f"| {profile.get('verdict')} | cap {profile.get('allocation_cap')} "
                f"| B band {profile.get('allocation_band')} | seat {profile.get('seat_family')}{stats}"
            )
            print(f"  Why: {profile.get('evidence')}")
            if profile.get("cross_source_corroboration"):
                print(f"  Corroboration: {profile.get('cross_source_corroboration')}")
            if profile.get("source") == "audit" and profile.get("audit_entry_price") is not None:
                print(
                    f"  Audit inventory: entry {profile.get('audit_entry_price')} -> TP {profile.get('audit_tp_price')} "
                    f"| edge {profile.get('audit_edge') or '?'} / alloc {profile.get('audit_allocation') or '?'} "
                    f"| opposite {profile.get('audit_opposite_entry_price') or 'n/a'}"
                )
            if profile.get("source") == "audit_range":
                print(
                    f"  Audit range: entry {profile.get('range_entry_price')} -> TP {profile.get('range_tp_price')} "
                    f"| opposite rail {profile.get('range_opposite_entry_price') or 'n/a'}"
                )
            print(
                f"  Context: {profile.get('session_context')} | "
                f"{profile.get('regime_context')}"
            )
            print(f"  Pair feedback: {profile.get('pair_context')}")
            if profile.get("tape_stat"):
                print(f"  Tape-matched feedback: {profile.get('tape_context')}")
            if profile.get("execution_style_feedback_note"):
                print(f"  Vehicle feedback: {profile.get('execution_style_feedback_note')}")
                print(f"  Vehicle sample: {profile.get('execution_style_context')}")
                if profile.get("exact_pretrade_label"):
                    pair_precision = 3 if str(profile.get("pair") or "").endswith("JPY") else 5
                    print(
                        f"  Exact pretrade: {profile.get('exact_pretrade_label')} "
                        f"-> {profile.get('exact_pretrade_style')} "
                        f"({profile.get('exact_pretrade_status')}) "
                        f"| proxy {profile.get('exact_proxy_anchor')} "
                        f"{float(profile.get('exact_entry_price')):.{pair_precision}f}"
                        f" -> TP {float(profile.get('exact_tp_price')):.{pair_precision}f}"
                        f" / SL {float(profile.get('exact_sl_price')):.{pair_precision}f}"
                    )
                if profile.get("exact_pretrade_note"):
                    print(f"  Exact note: {profile.get('exact_pretrade_note')}")
                if profile.get("exact_pretrade_hard_blockers"):
                    print(
                        "  Hard guardrails: "
                        + " | ".join(str(item) for item in (profile.get("exact_pretrade_hard_blockers") or []))
                    )
                elif profile.get("exact_pretrade_advisories"):
                    print(
                        "  Trader override room: "
                        + " | ".join(str(item) for item in (profile.get("exact_pretrade_advisories") or []))
                    )
                if profile.get("recent_feedback_note"):
                    print(f"  Recent feedback override: {profile.get('recent_feedback_note')}")
                if profile.get("promotion_pressure_note"):
                    print(f"  Promotion pressure: {profile.get('promotion_pressure_note')}")
                print(
                    f"  Sizing lane: {profile.get('band_size_note')} | "
                    f"{profile.get('allocation_band_reason')}"
                )
            spread_note = ""
            if profile.get("execution_spread_ratio") is not None:
                spread_note = (
                    f" | spread {spread_data.get(profile['pair'], 0):.1f}pip"
                    f" ({float(profile.get('execution_spread_ratio', 0.0))*100:.0f}% of "
                    f"{float(profile.get('execution_target_pips', 0.0)):.0f}pip path)"
                )
                print(
                    f"  Default expression: {_profile_default_expression(profile)} "
                    f"({_profile_default_orderability(profile)}){spread_note}"
                )
                print(f"  Why AI likes this default: {profile.get('execution_note')}")
                print(f"  Live tape: {profile.get('live_tape_note')}")

        fresh_profiles = [profile for profile in learning_profiles if profile.get("source") != "held"]
        if fresh_profiles:
            for profile in fresh_profiles:
                _apply_profile_promotion_pressure(profile, audit_context, missed_pressure)
            fresh_profiles = sorted(fresh_profiles, key=_fresh_profile_sort_key, reverse=True)
            best_direct = _select_bucket_board_profile(fresh_profiles, "direct_usd")
            best_cross = _select_bucket_board_profile(fresh_profiles, "jpy_cross")
            best_usdjpy = _select_bucket_board_profile(fresh_profiles, "usd_jpy")
            session_intent = _compute_session_intent_gate(
                trades_data,
                pending_orders,
                audit_context,
                missed_pressure,
                fresh_profiles,
            )

            section("SESSION INTENT GATE")
            print(f"Mode: {session_intent.get('mode')}")
            for reason in session_intent.get("reasons", []):
                print(f"- {reason}")
            if session_intent.get("mode") == "WATCH-ONLY":
                print(
                    "Carry the prior map forward. Update management, pending freshness, and the exact re-arm / kill line "
                    "instead of re-ranking every pair from scratch."
                )
            else:
                print(
                    "A live deployment reason exists. Do not leave this session as fresh prose only: arm the seat or "
                    "write the exact contradiction that still blocks it."
                )

            section("FRESH-RISK TOURNAMENT (learning + live-tape weighted)")
            for idx, profile in enumerate(fresh_profiles[:8], start=1):
                pressure = ""
                if profile.get("promotion_pressure_note"):
                    pressure = f" | pressure {profile.get('promotion_pressure_note')}"
                print(
                    f"{idx}. {profile['pair']} {profile['direction']} [{_profile_source_label(profile)}] "
                    f"| learning {int(profile.get('learning_score', 0))}/100 "
                    f"| cap {profile.get('allocation_cap')} | band {profile.get('allocation_band')} "
                    f"| {profile.get('verdict')} "
                    f"| default {_profile_default_expression(profile)} "
                    f"| tape {_profile_live_tape_label(profile)} "
                    f"| {profile.get('current_session')}/{profile.get('current_regime') or 'n/a'} "
                    f"| life {profile.get('shelf_life_label')} "
                    f"| seat {profile.get('seat_family')}{pressure}"
                )

            actionable_profiles = [
                profile for profile in fresh_profiles
                if _profile_default_expression(profile) != "PASS"
            ]
            if actionable_profiles:
                section("SEAT INVENTORY BOARD (copy this before you compress)")
                print(
                    "The market can carry more than one seat per pair when the trigger/vehicle differs. "
                    "Do not compress these into one story unless the chart proves they are the same seat."
                )
                for idx, profile in enumerate(actionable_profiles[:MAX_DEPLOYMENT_LANES], start=1):
                    print(
                        f"{_multi_vehicle_role(idx)}: {profile['pair']} {profile['direction']} "
                        f"[{_profile_source_label(profile)}] "
                        f"| seat {profile.get('seat_family')} "
                        f"| default {_profile_default_expression(profile)} ({_profile_default_orderability(profile)}) "
                        f"| cap {profile.get('allocation_cap')} "
                        f"| life {profile.get('shelf_life_label')}"
                    )
                    print(f"  Why AI likes this default: {profile.get('execution_note')}")
                    if profile.get("exact_pretrade_hard_blockers"):
                        print("  Hard guardrails: " + " | ".join(profile.get("exact_pretrade_hard_blockers") or []))
                    elif profile.get("exact_pretrade_advisories"):
                        print("  Trader override room: " + " | ".join(profile.get("exact_pretrade_advisories") or []))
                section("GOLD MINE INVENTORY (top seams to monetize, not narrate)")
                print(
                    "In FULL_TRADER, copy at least Gold #1-#5 before you compress into one hero seat. "
                    "Each mine must close as a real receipt or an exact contradiction."
                )
                for idx, profile in enumerate(actionable_profiles[:MAX_DEPLOYMENT_LANES], start=1):
                    default_action = _profile_default_expression(profile)
                    receipt_hint = _profile_command_receipt_hint(profile)
                    print(
                        f"Gold #{idx}: {profile['pair']} {profile['direction']} "
                        f"[{_profile_source_label(profile)}] | seat {profile.get('seat_family')} "
                        f"| Arm if alive now as {default_action}"
                    )
                    print(f"  Why mineable now: {profile.get('execution_note')}")
                    if profile.get("cross_source_corroboration"):
                        print(f"  Corroboration: {profile.get('cross_source_corroboration')}")
                    print(
                        f"  If not armed by session end: {receipt_hint} or exact chart/tape contradiction "
                        f"(not `no pending id`)."
                    )
            if actionable_profiles:
                mix = Counter(_profile_default_expression(profile) for profile in actionable_profiles)
                section("LANE SHELF-LIFE BOARD")
                print(
                    "Actionable mix: "
                    + " | ".join(
                        f"{style} {mix.get(style, 0)}"
                        for style in ("MARKET", "STOP-ENTRY", "LIMIT")
                        if mix.get(style, 0)
                    )
                )
                for profile in actionable_profiles[:MAX_DEPLOYMENT_LANES]:
                    print(
                        f"{profile['pair']} {profile['direction']} [{_profile_source_label(profile)}] "
                        f"| seat {profile.get('seat_family')} "
                        f"| {_profile_default_expression(profile)} "
                        f"| shelf {profile.get('shelf_life_label')}"
                    )
                    print(f"  Carry rule: {profile.get('carry_rule')}")

            if best_direct:
                print(
                    f"Best direct-USD board seat: {best_direct['pair']} {best_direct['direction']} "
                    f"| cap {best_direct.get('allocation_cap')} | {best_direct.get('verdict')} "
                    f"| default {_profile_default_expression(best_direct)} "
                    f"| tape {_profile_live_tape_label(best_direct)}"
                )
            if best_cross:
                print(
                    f"Best cross board seat: {best_cross['pair']} {best_cross['direction']} "
                    f"| cap {best_cross.get('allocation_cap')} | {best_cross.get('verdict')} "
                    f"| default {_profile_default_expression(best_cross)} "
                    f"| tape {_profile_live_tape_label(best_cross)}"
                )
            if best_usdjpy:
                print(
                    f"Best USD_JPY board seat: {best_usdjpy['pair']} {best_usdjpy['direction']} "
                    f"| cap {best_usdjpy.get('allocation_cap')} | {best_usdjpy.get('verdict')} "
                    f"| default {_profile_default_expression(best_usdjpy)} "
                    f"| tape {_profile_live_tape_label(best_usdjpy)}"
                )
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
                        f"| seat {seed.get('seat_family') or 'seat'} "
                        f"| Closest-to-S because {seed['closest_to_s_because']} "
                        f"| Still blocked by {seed['still_blocked_by']} "
                        f"| If it upgrades: {seed['upgrade_action']}"
                    )
                if len(podium_seeds) < MAX_PODIUM_SEEDS:
                    print(
                        f"Only {len(podium_seeds)} podium seed(s) survived the live execution gate. "
                        "Fill the remaining slot(s) manually only if the chart gives a concrete better seat."
                    )
            else:
                print(
                    f"No auto-seeded podium survived the execution gate. Fill Podium #1-#{MAX_PODIUM_SEEDS} manually from the "
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
                    f"[{_profile_source_label(profile)}] "
                    f"→ default {_profile_default_expression(profile)} ({_profile_default_orderability(profile)}){spread_note}"
                )
                if profile.get("promotion_pressure_note"):
                    print(f"  Pressure: {profile.get('promotion_pressure_note')}")
                if profile.get("cross_source_corroboration"):
                    print(f"  Corroboration: {profile.get('cross_source_corroboration')}")
                print(f"  Shelf-life: {profile.get('shelf_life_label')}")
                print(f"  Carry rule: {profile.get('carry_rule')}")
                print(f"  Live tape: {profile.get('live_tape_note')}")
                print(f"  Profit-first rationale: {profile.get('execution_note')}")
                if profile.get("exact_pretrade_hard_blockers"):
                    print("  Hard guardrails: " + " | ".join(profile.get("exact_pretrade_hard_blockers") or []))
                elif profile.get("exact_pretrade_advisories"):
                    print("  Trader override room: " + " | ".join(profile.get("exact_pretrade_advisories") or []))
                print(
                    "  If this is still your best live seat after the chart read, "
                    "do not leave S Hunt as prose. Close it as an order or kill it explicitly."
                )
            section("PAYABLE-NOW MARKET CHECK")
            market_now_profiles = [
                profile for profile in fresh_profiles
                if _profile_default_expression(profile) == "MARKET"
            ]
            if market_now_profiles:
                for profile in market_now_profiles[:MAX_DEPLOYMENT_LANES]:
                    print(
                        f"{profile['pair']} {profile['direction']} [{_profile_source_label(profile)}] "
                        f"| seat {profile.get('seat_family')} | ENTER NOW | tape {_profile_live_tape_label(profile)}"
                    )
                    print(f"  Shelf-life: {profile.get('shelf_life_label')}")
                    print(f"  Carry rule: {profile.get('carry_rule')}")
                    if profile.get("cross_source_corroboration"):
                        print(f"  Corroboration: {profile.get('cross_source_corroboration')}")
                    print(f"  Why payable now: {profile.get('execution_note')}")
            else:
                print("No payable market now.")
                for profile in fresh_profiles[:5]:
                    print(
                        f"- {profile['pair']} {profile['direction']} [{_profile_source_label(profile)}] "
                        f"| seat {profile.get('seat_family')} "
                        f"default stays {_profile_default_expression(profile)} because "
                        f"{_ledger_safe_text(profile.get('execution_note'), 110)}"
                    )
                    if profile.get("exact_pretrade_hard_blockers"):
                        print("  Hard guardrails: " + " | ".join(profile.get("exact_pretrade_hard_blockers") or []))
                    elif profile.get("exact_pretrade_advisories"):
                        print("  Trader override room: " + " | ".join(profile.get("exact_pretrade_advisories") or []))
                print(
                    "If you still want MARKET here, use trader judgment. Only hard guardrails bind."
                )
            multi_vehicle_lanes = _select_multi_vehicle_lanes(fresh_profiles)
            _write_action_board_snapshot(
                session_intent=session_intent,
                market_now_profiles=market_now_profiles,
                multi_vehicle_lanes=multi_vehicle_lanes,
                best_direct=best_direct,
                best_cross=best_cross,
                best_usdjpy=best_usdjpy,
            )
            if multi_vehicle_lanes:
                section("MULTI-VEHICLE DEPLOYMENT LANES (broad tape or multi-seat same-pair deployment)")
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
                        f"{role}: {pair} {profile['direction']} [{_profile_source_label(profile)}] "
                        f"| seat {profile.get('seat_family')} "
                        f"→ {_profile_default_expression(profile)} ({_profile_default_orderability(profile)})"
                        f" | {base}/{quote}{spread_note}"
                    )
                    print(f"  Shelf-life: {profile.get('shelf_life_label')}")
                    print(f"  Carry rule: {profile.get('carry_rule')}")
                    print(f"  Live tape: {profile.get('live_tape_note')}")
                    if profile.get("cross_source_corroboration"):
                        print(f"  Corroboration: {profile.get('cross_source_corroboration')}")
                    print(
                        f"  Why this can coexist: {_multi_vehicle_reason(profile, multi_vehicle_lanes[:idx-1])}"
                    )
                    print(f"  Execution: {profile.get('execution_note')}")
                    print(
                        "  Book rule: valid to carry with the other lanes when this is a distinct trigger/vehicle "
                        "seat. Blind same-pair averaging is still banned. Worst-case margin after all pending fills "
                        "must stay below 90%."
                    )
                armable_commands = [
                    (idx, profile, _profile_order_command(profile))
                    for idx, profile in enumerate(multi_vehicle_lanes, start=1)
                ]
                armable_commands = [
                    (idx, profile, command)
                    for idx, profile, command in armable_commands
                    if command
                ]
                if armable_commands:
                    section("CANDIDATE ORDER RECIPES FOR ARMABLE LANES (default expression; trader decides)")
                    for idx, profile, command in armable_commands:
                        role = _multi_vehicle_role(idx)
                        print(
                            f"{role}: {profile['pair']} {profile['direction']} "
                            f"[{_profile_source_label(profile)}] -> {_profile_command_order_type(profile)}"
                        )
                        print(
                            f"  Expected receipt: {_profile_command_receipt_hint(profile)} "
                            f"| size {_profile_command_units(profile)}u"
                        )
                        print(f"  Command: {command}")
                        print(
                            "  Invalid dead close: `dead thesis because no live pending entry order exists` "
                            "is execution drift, not a market contradiction."
                        )
                        print(
                            "  Trader override is valid if the live tape changed. Only hard guards "
                            "(live-book drift, payoff/stop geometry, unpaid-unprotected same-pair stacking) bind."
                        )
            section("INTRADAY LEARNING LOOP (OODA + DECISION JOURNAL)")
            for idx, profile in enumerate(fresh_profiles[:4], start=1):
                print(
                    f"### Seat #{idx}: {profile['pair']} {profile['direction']} "
                    f"[{_profile_source_label(profile)}] seat {profile.get('seat_family')}"
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
                    f"  Decide: default close state = {_profile_default_orderability(profile)} "
                    f"({_profile_default_expression(profile)}) "
                    f"because {profile.get('execution_note')}"
                )
                print(f"    Live tape now: {profile.get('live_tape_note')}")
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

    if not emit_templates:
        section("HANDOFF REFRESH (copy these live facts into state.md; do not reuse stale carry-forward focus)")
        print(f"Entries today: {_entry_activity_summary(entry_activity, include_pair=True)}")
        live_trades = trades.get("trades", []) if trades else []
        live_keys = {
            (
                str(trade.get("instrument") or "").replace("/", "_"),
                _units_to_direction(trade.get("currentUnits")) or "?",
            )
            for trade in live_trades
        }
        if live_trades:
            primary_trade = live_trades[0]
            primary_pair = str(primary_trade.get("instrument") or "?").replace("/", "_")
            primary_dir = _units_to_direction(primary_trade.get("currentUnits")) or "?"
            primary_line = f"{primary_pair} {primary_dir} live trade id=`{primary_trade.get('id', '?')}`"
        elif multi_vehicle_lanes:
            lane = multi_vehicle_lanes[0]
            primary_line = (
                f"{lane.get('pair')} {lane.get('direction')} "
                f"{_profile_default_orderability(lane)}"
            )
        else:
            primary_line = "none"

        backup_lane = next(
            (
                lane for lane in multi_vehicle_lanes
                if (str(lane.get("pair") or ""), str(lane.get("direction") or "")) not in live_keys
            ),
            None,
        )
        backup_line = (
            f"{backup_lane.get('pair')} {backup_lane.get('direction')} "
            f"{_profile_default_orderability(backup_lane)}"
            if backup_lane
            else "none"
        )
        armed_backup_order = next(
            (
                order
                for order in _iter_pending_entry_orders(pending_orders)
                if (
                    str(order.get("instrument") or "").strip(),
                    _units_to_direction(order.get("units")) or "",
                ) not in live_keys
            ),
            None,
        )
        armed_backup_line = _pending_entry_receipt_line(armed_backup_order)

        next_profile = next(
            (
                lane for lane in market_now_profiles
                if (str(lane.get("pair") or ""), str(lane.get("direction") or "")) not in live_keys
            ),
            None,
        ) or backup_lane
        next_risk_line = (
            f"{next_profile.get('pair')} {next_profile.get('direction')} "
            f"{_profile_default_orderability(next_profile)}"
            if next_profile
            else "none"
        )

        print(f"Primary vehicle now: {primary_line}")
        print(f"Backup vehicle now: {backup_line}")
        print(f"Next fresh risk allowed NOW: {next_risk_line}")
        print(f"20-minute backup trigger armed NOW: {armed_backup_line}")
        primary_profile = None
        if live_trades:
            primary_profile = next(
                (
                    profile for profile in learning_profiles
                    if str(profile.get("pair") or "") == primary_pair
                    and str(profile.get("direction") or "") == primary_dir
                ),
                None,
            )
        elif multi_vehicle_lanes:
            primary_profile = multi_vehicle_lanes[0]
        if primary_profile:
            print(f"Primary vehicle shelf-life now: {primary_profile.get('shelf_life_label')}")
        if backup_lane:
            print(f"Backup vehicle shelf-life now: {backup_lane.get('shelf_life_label')}")
        if next_profile:
            print(f"Next fresh risk shelf-life now: {next_profile.get('shelf_life_label')}")
        if next_profile:
            print(
                "Lane-two discipline: do not write `none while the live trade is active` if the current board still says "
                f"{next_risk_line}. Either arm it or write the exact pretrade contradiction."
            )
            print(f"Current lane-two gate: {_ledger_safe_text(str(next_profile.get('execution_note') or ''), 160)}")
            print(f"Next-session carry rule: {next_profile.get('carry_rule')}")
        else:
            print("Lane-two gate: no second lane survived the current payoff / stop-floor / live-tape checks.")
        if multi_vehicle_lanes:
            print("Lane inventory now:")
            for idx, lane in enumerate(multi_vehicle_lanes[:MAX_DEPLOYMENT_LANES], start=1):
                print(
                    f"  {_multi_vehicle_role(idx)}: {lane.get('pair')} {lane.get('direction')} "
                    f"| seat {lane.get('seat_family')} "
                    f"| {_profile_default_orderability(lane)}"
                )
            print("Gold Mine Inventory lines now:")
            for idx, lane in enumerate(multi_vehicle_lanes[:MAX_DEPLOYMENT_LANES], start=1):
                print(
                    f"  Gold #{idx}: {lane.get('pair')} {lane.get('direction')} "
                    f"[{_profile_source_label(lane)}] | seat {lane.get('seat_family')} "
                    f"| Arm if alive now as {_profile_default_expression(lane)} "
                    f"| If not armed by session end: {_profile_command_receipt_hint(lane)} "
                    "or exact contradiction ___"
                )
            print("Capital Deployment lane lines now:")
            for idx, lane in enumerate(multi_vehicle_lanes[:MAX_DEPLOYMENT_LANES], start=1):
                print(
                    f"  {_multi_vehicle_role(idx)}: {lane.get('pair')} {lane.get('direction')} "
                    f"{_profile_default_orderability(lane)} [{_profile_source_label(lane)}] "
                    f"-> {_profile_command_receipt_hint(lane)} or dead thesis because ___"
                )

        prior_backup = str(focus_snapshot.get("Backup vehicle:") or "").strip()
        prior_next = str(focus_snapshot.get("Next fresh risk allowed NOW:") or "").strip()
        if backup_line != "none" and re.search(r"\bnone\b", prior_backup, flags=re.IGNORECASE):
            print(
                f"Carry-forward contradiction: prior `Backup vehicle` was `{prior_backup or 'none'}`, "
                f"but the live board now says `{backup_line}`."
            )
        if next_risk_line != "none" and re.search(r"\bnone\b", prior_next, flags=re.IGNORECASE):
            print(
                f"Carry-forward contradiction: prior `Next fresh risk allowed NOW` was `{prior_next or 'none'}`, "
                f"but the live board now says `{next_risk_line}`."
            )

        section("TEMPLATES")
        print(
            "Static state.md templates omitted in normal runtime mode. "
            "Use docs/SKILL_trader.md for the canonical blocks, or rerun session_data.py with --emit-templates."
        )
        elapsed = time.time() - t0
        print(f"\n[session_data: {elapsed:.1f}s]")
        return

    # 8. Pre-filled templates (v8.4 — model fills blanks, can't skip fields)
    section("⚠ MANDATORY TEMPLATES — write ALL of these to state.md FIRST, before any analysis")

    # Self-check template with pre-filled entry count
    entry_count_str = _entry_activity_summary(entry_activity, include_pair=True)

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
        print("\n## Position Management (fill ALL 5 for each C)")
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
            regret_hint = "No recent sample."
            regret_stats = _recent_regret_for_pair(pair)
            if regret_stats:
                regret_hint = (
                    f"{regret_stats['recovered']}/{regret_stats['count']} recovered in 6h "
                    f"({regret_stats['recovery_rate']:.0f}%) | avg loss {regret_stats['avg_loss_pips']:.1f}pip "
                    f"-> avg later favorable {regret_stats['avg_fav_pips']:.1f}pip"
                )
                if regret_stats.get("median_lag_min") is not None:
                    regret_hint += f" | median return {regret_stats['median_lag_min']:.0f}min"

            print(f"""
### {pair} {units}u id={tid} UPL={upl}
  A — Close now: {upl} JPY
  B — Half TP: close ___u, trail ___pip
  C — Hold REQUIRES all 5:
    (1) Changed since last session: ___
    (2) Entry TF: ___
        M15: {m15_hint} → [with/against] ___
        M1: {m1_hint} → [supports/threatens] ___
    (3) H4: {h4_hint} → Room? [YES/NO]
    (4) Enter NOW @current? [YES/NO]
    (5) Range monetization: [n/a because not range / TAKE PROFIT NOW / HOLD ONLY FOR BREAKOUT / ROTATE THE BOX]
        Regime transition: [same / TREND->RANGE / SQUEEZE->RANGE / other]
        Box now: lower ___ | upper ___
        Paid path from here: [breakout through ___ / sell upper rail ___ / buy lower rail ___ / TP ___]
        Why holding the old direction still beats boxing it: ___
    Recent regret on losing closes: {regret_hint}
    Dead layer if closing now: [market / structure / trigger / vehicle / aging]
    Surviving layers if closing now: ___
    If dead layer = trigger / vehicle, why full close beats hold / half / reload: ___
  → Chosen: [A/B/C]""")

    # Conviction template (for new entries)
    print(f"""
## New Entry Template (v8.5 — fill ALL fields)
  pretrade_check command: python3 collab_trade/memory/pretrade_check.py {{PAIR}} {{LONG|SHORT}} [--counter] --entry {{ENTRY_PRICE}} --tp {{TP_PRICE}} --sl {{SL_PRICE}}
  place_trader_order command: python3 tools/place_trader_order.py [MARKET / LIMIT / STOP-ENTRY] {{PAIR}} {{LONG|SHORT}} {{UNITS}} [--entry {{ENTRY_PRICE}}] --tp {{TP_PRICE}} --sl {{SL_PRICE}} --thesis {{THESIS}} --pretrade {{GRADE}} --allocation {{LANE}} [--allocation-band {{BAND}}] [--counter]
  Thesis: [CHART story, not indicators. If the indicators disappeared, this line must still stand]
  Last 5 M5: bodies [___] × [___], wicks [___] → Buyers defending? [YES @___ / NO → PASS]
  What the market is paying RIGHT NOW on this pair: ___
  Why the opposite side is wrong RIGHT NOW: ___
  Regime: [___] | Type: [___] | Expected: [___] → Zombie: [___Z]
  MTF chain: H4/H1 ___ | M15 ___ | M5 ___ | M1 ___
  Theme confidence: [proving/confirmed/late] → allocation lane: [B/A/S] (does NOT change edge)
  B band if allocation lane = B: [B+ / B0 / B- / n/a] because ___
  Learning verdict: [confirmed edge / watch edge / no-edge / limited history] ← copy from LEARNING EDGE BOARD
  Learning cap: [A/S when theme confirmed / A max / B-only / pass unless exceptional]
  Tape-first orderability: [ENTER NOW / NEED STOP-ENTRY / NEED LIMIT / PASS] because ___
  Primary continuity: [same-theme add / same-theme reload / rotation away from old primary]
  If rotation away from last primary: exact contradiction = ___
  If same theme: why this deserves the next unit instead of inventing a new lane = ___
  pretrade_check execution style: [MARKET / LIMIT / STOP-ENTRY / PASS] | [agrees/disagrees] because ___
  A/S proof combo: ___ [cross-TF combo that makes this deserve market / trigger / price-improvement treatment]
  Why this is not just B: ___
  If not MARKET: exact structural level / exact trigger = ___ | if PASS: dead thesis because ___
  Tournament rank: [#1-#10 fresh-risk seat / unranked]
  Multi-vehicle lane: [Lane 1 / PRIMARY ... Lane 10 / SEAT 10 / NONE]
  Same-pair live inventory right now: [NONE / id=___ side=___ UPL=___]
  Second fill allowed only if: [existing leg already paid / risk already reduced / different trigger/vehicle / NO because ___]
  Paid lane on this thesis today already? [NO / YES trade id=___ + why it proved]
  If YES, why is this same-size / +1 lane / full-size? ___
  If this wins, the next honest size on the same clean theme = ___u because ___
  AGAINST: ___
  If wrong: ___
  H4 position: {h4_hint if held_trades else "StRSI=___ →"} [early/mid/late/exhausting]
  Cross-currency: [currency] M15 [bid/offered] across [N] pairs → [currency-wide/pair-specific] → conviction [UP/DOWN]
  Event asymmetry: [event] at [time]. Positioned for [___]. [favorable/unfavorable]
  Margin: ___% → worst case ___% | → Edge: [S/A/B/C] Allocation: [S/A/B/C] B band: [B+ / B0 / B- / n/a] Size: ___u""")

    print("\n## S Excavation Matrix (write after 7-Pair Scan, before S Hunt)")
    print("Default podium source: copy `S EXCAVATION SEEDS` unless the live chart disproves them.")
    for pair in PAIRS:
        print(
            f"{pair}: Best expression ___ | Best A/S path ___ [MTF + indicator combo] | Why not S now ___ | "
            "Upgrade to S only if ___ | Dead if ___"
        )
    for idx in range(MAX_PODIUM_SEEDS):
        if idx < len(podium_seeds):
            seed = podium_seeds[idx]
            print(
                f"Podium #{idx + 1}: {seed['pair']} {seed['direction']} "
                f"| Seat family {seed.get('seat_family') or '___'} "
                f"| Closest-to-S because {seed['closest_to_s_because']} "
                "| MTF/indicator combo ___ "
                f"| Still blocked by {seed['still_blocked_by']} "
                f"| If it upgrades: {seed['upgrade_action']}"
            )
        else:
            print(
                f"Podium #{idx + 1}: [PAIR LONG/SHORT] | Closest-to-S because ___ | "
                "MTF/indicator combo ___ | "
                "Still blocked by ___ | If it upgrades: [MARKET / LIMIT / STOP-ENTRY]"
            )
    print("""

## A/S Excavation Mandate (write after S Excavation Matrix)
Best A/S live now: [PAIR LONG/SHORT / none only if every live candidate is explicitly contradicted]
  Why this is A/S: ___
  MTF/indicator combo: ___
  Why this is not just B: ___
  Order now: [ENTER NOW already filled as trade id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]

Best A/S one print away: [PAIR LONG/SHORT]
  Missing print: ___
  MTF/indicator combo waiting to complete: ___
  Arm now as: [armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]

Best A/S I am explicitly rejecting: [PAIR LONG/SHORT / none]
  Exact contradiction: ___
""")
    print("\n## Multi-Vehicle Deployment (copy every surviving lane from the runtime board)")
    print("Lane 1 / PRIMARY: ___ [pair + dir + seat family + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 2 / BACKUP: ___ [pair + dir + seat family + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 3 / THIRD CURRENCY: ___ [pair + dir + seat family + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 4 / FOURTH SEAT: ___ [pair + dir + seat family + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Lane 5 / FIFTH SEAT: ___ [pair + dir + seat family + entered id=___ / armed STOP id=___ / armed LIMIT id=___ / dead thesis because ___]")
    print("Additional lanes: continue `Lane N / ...` lines for every surviving seat on the action board.")
    print("Execution count this session: ___ live receipts | ___ armed receipts")
    print("If broad tape but fewer than 2 live/armed lanes survived: exact blocker to second lane = ___")
    print("Size asymmetry audit: today's biggest winner used ___u; best live seat now uses ___u; if not bigger, exact blocker = ___")
    print("Book rule: blind same-pair averaging is banned | same-pair multi-seat is allowed only when trigger/vehicle/invalidation differs | distinct currency expression is still preferred | worst-case margin after all pending fills < 90%")

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
