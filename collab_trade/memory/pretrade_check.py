"""
QuantRabbit Trading Memory — Pre-Trade Check
Cross-references 3 memory layers (risk) immediately before entry + evaluates current setup quality (conviction)

Output:
  - RISK: HIGH/MEDIUM/LOW — risk warnings from historical data (backward-looking)
  - CONFIDENCE: S/A/B/C — quality of current setup (forward-looking) → basis for sizing decisions
  - B-BAND: B+/B0/B- — separates promotable B from noisy B without rewriting the core grade ladder
  - RECOMMENDED SIZE: unit count based on conviction
  - HISTORICAL PAYOFF: realized expectancy / avg win-loss / break-even WR

Usage:
  python3 pretrade_check.py GBP_USD SHORT [--adx 38] [--headline "Iran"] [--entry 1.35160 --tp 1.35210 --sl 1.35074]
"""
from __future__ import annotations

import sys
import json
import re
from collections import Counter
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from schema import get_conn, init_db, fetchall_dict, fetchone_val, fetchone_dict, serialize_f32

ROOT = Path(__file__).resolve().parent.parent.parent  # memory/ → collab_trade/ → QuantRabbit/
sys.path.insert(0, str(ROOT / "tools"))
from technicals_json import load_technicals_timeframes
from pricing_probe import MIN_TAPE_SAMPLES, PROBE_CACHE_PATH, probe_market
from runtime_history import live_history_scope_label, live_history_start

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
DIRECT_USD_PAIRS = {"EUR_USD", "GBP_USD", "AUD_USD"}
PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"), "EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"), "EUR_JPY": ("EUR", "JPY"), "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}
LESSON_REGISTRY_PATH = ROOT / "collab_trade" / "memory" / "lesson_registry.json"
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
GRADE_RANK = {"C": 0, "B": 1, "A": 2, "S": 3}
EXECUTION_STYLE_RANK = {"PASS": 0, "STOP-ENTRY": 1, "LIMIT": 2, "MARKET": 3}
THESIS_HISTORY_LIMIT = 5
THESIS_LOOKBACK_DAYS = 10
THESIS_EXACT_BLOCK_HOURS = 6
THESIS_FAMILY_HEADWIND_STREAK = 2
RECENT_PRETRADE_LOOKBACK_DAYS = 14
RECENT_TRADE_LOOKBACK_DAYS = 21
RECENT_FEEDBACK_MIN_COUNT = 3
PRICING_PROBE_MAX_AGE_SEC = 90.0
PRICING_PROBE_SAMPLES = 10
PRICING_PROBE_INTERVAL_SEC = 0.40
PRICING_PROBE_DURATION_SEC = 4.0
JST = timezone(timedelta(hours=9))
AUDIT_PRESSURE_LOOKBACK_HOURS = 6
AUDIT_REPEAT_TRIGGER_COUNT = 3
MISSED_SEAT_LOOKBACK_HOURS = 12
MISSED_SEAT_MIN_PIPS = 20.0
_RECENT_REGRET_PAYLOAD: dict | None = None
_RECENT_AUDIT_PRESSURE: dict | None = None
_RECENT_MISSED_SEAT_PRESSURE: dict | None = None


def _load_lesson_registry() -> dict:
    try:
        return json.loads(LESSON_REGISTRY_PATH.read_text())
    except Exception:
        return {}


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
    return {
        "count": len(rows),
        "recovered": recovered,
        "recovery_rate": recovered / len(rows) * 100.0,
        "avg_loss_pips": avg_loss,
        "avg_fav_pips": avg_fav,
    }


def _recent_regret_for_pair(pair: str, *, stop_loss_only: bool = False) -> dict | None:
    payload = _load_recent_regret_payload()
    rows = [
        row
        for row in payload.get("results", [])
        if row.get("pair") == pair
        and (not stop_loss_only or row.get("close_reason") == "STOP_LOSS_ORDER")
    ]
    return _summarize_regret_rows(rows)


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
            why = _clip_text(str(strongest.get("why") or ""), limit=120)
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
            why = _clip_text(str(pick.get("why") or ""), limit=120)
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
    conn = get_conn()
    try:
        rows = fetchall_dict(
            conn,
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
        )
    except Exception:
        rows = []

    payload["pairs"] = {
        (row["pair"], row["direction"]): {
            "count": int(row.get("cnt") or 0),
            "max_pip_move": float(row.get("max_pip_move") or 0.0),
            "horizons": str(row.get("horizons") or ""),
            "window_hours": window_hours,
        }
        for row in rows
        if row.get("pair") and row.get("direction")
    }
    _RECENT_MISSED_SEAT_PRESSURE = payload
    return payload


def _recent_promotion_pressure(pair: str, direction: str) -> dict:
    audit_stats = (_load_recent_audit_pressure().get("pairs") or {}).get((pair, direction))
    missed_stats = (_load_recent_missed_seat_pressure().get("pairs") or {}).get((pair, direction))
    bits: list[str] = []
    rank = 0

    if audit_stats:
        count = int(audit_stats.get("count", 0) or 0)
        hours = int(audit_stats.get("window_hours", AUDIT_PRESSURE_LOOKBACK_HOURS) or AUDIT_PRESSURE_LOOKBACK_HOURS)
        bits.append(f"audit repeated {count}x/{hours}h")
        rank += min(20, 6 + count * 3)
        why = str(audit_stats.get("why") or "").strip()
        if why:
            bits.append(_clip_text(why, limit=80))

    if missed_stats:
        count = int(missed_stats.get("count", 0) or 0)
        max_pip_move = float(missed_stats.get("max_pip_move", 0.0) or 0.0)
        hours = int(missed_stats.get("window_hours", MISSED_SEAT_LOOKBACK_HOURS) or MISSED_SEAT_LOOKBACK_HOURS)
        bits.append(f"missed {count}x/{hours}h; best {max_pip_move:.1f}pip worked")
        rank += min(24, 8 + int(max_pip_move // 4))

    return {
        "rank": rank,
        "note": " + ".join(bits),
        "audit_stats": audit_stats,
        "missed_stats": missed_stats,
    }


def _recent_noise_stop_floor(pair: str, spread_pips: float | None) -> dict | None:
    stop_loss_stats = _recent_regret_for_pair(pair, stop_loss_only=True)
    pair_stats = _recent_regret_for_pair(pair, stop_loss_only=False)

    source = None
    stats = None
    if (
        stop_loss_stats
        and int(stop_loss_stats.get("count", 0) or 0) >= 3
        and float(stop_loss_stats.get("recovery_rate", 0.0) or 0.0) >= 65.0
    ):
        source = "recent stop-loss regret"
        stats = stop_loss_stats
    elif (
        pair_stats
        and int(pair_stats.get("count", 0) or 0) >= 5
        and float(pair_stats.get("recovery_rate", 0.0) or 0.0) >= 75.0
    ):
        source = "recent pair regret"
        stats = pair_stats

    if not stats:
        return None

    required_stop_pips = float(stats.get("avg_loss_pips", 0.0) or 0.0)
    if spread_pips is not None:
        required_stop_pips = max(required_stop_pips, float(spread_pips) * 5.0)
    if required_stop_pips <= 0:
        return None

    return {
        "source": source,
        "count": int(stats.get("count", 0) or 0),
        "recovered": int(stats.get("recovered", 0) or 0),
        "recovery_rate": float(stats.get("recovery_rate", 0.0) or 0.0),
        "avg_loss_pips": float(stats.get("avg_loss_pips", 0.0) or 0.0),
        "avg_fav_pips": float(stats.get("avg_fav_pips", 0.0) or 0.0),
        "required_stop_pips": required_stop_pips,
    }


def _noise_stop_floor_reason(pair: str, planned_stop_pips: float, floor: dict) -> str:
    return (
        f"{pair} planned SL {planned_stop_pips:.1f}pip is still inside the recent noise floor "
        f"{float(floor.get('required_stop_pips', 0.0) or 0.0):.1f}pip "
        f"({floor.get('source')}: {int(floor.get('recovered', 0) or 0)}/{int(floor.get('count', 0) or 0)} "
        f"recovered in 6h, avg loss {float(floor.get('avg_loss_pips', 0.0) or 0.0):.1f}pip "
        f"-> avg later favorable {float(floor.get('avg_fav_pips', 0.0) or 0.0):.1f}pip)"
    )


def _registry_rank(lesson: dict) -> tuple[int, int, str, str]:
    return (
        int(lesson.get("trust_score", 0) or 0),
        int(lesson.get("state_rank", 0) or 0),
        str(lesson.get("lesson_date", "")),
        str(lesson.get("id", "")),
    )


def _target_lessons_for_profile(pair: str, direction: str, registry: dict) -> tuple[list[dict], list[dict]]:
    lessons = registry.get("lessons") or []
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


def _compact_token(value: str | None) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().upper()).strip("_")
    return token or "UNKNOWN"


def _setup_has_detail(setup: dict, *needles: str) -> bool:
    details = [str(detail).lower() for detail in (setup.get("details") or []) if detail]
    return any(any(needle in detail for needle in needles) for detail in details)


def _setup_archetype(setup: dict) -> str:
    if setup.get("is_counter"):
        return "COUNTER_REVERSAL"

    wave = str(setup.get("wave") or "auto").lower()
    mtf = int(setup.get("mtf_aligned", 0) or 0)
    if wave == "big":
        return "TREND_SWING" if mtf >= 2 else "SWING_PROBE"
    if wave == "mid":
        return "TREND_PULLBACK" if mtf >= 2 else "MID_ROTATION"
    if wave == "small":
        return "MICRO_TRIGGER"
    return "STRUCTURE_TEST"


def _setup_flags(setup: dict) -> list[str]:
    flags: list[str] = []
    if _setup_has_detail(setup, "divergence", "reversal confirmed"):
        flags.append("REVERSAL")
    if _setup_has_detail(setup, "h4 extreme", "overbought", "oversold"):
        flags.append("EXTREME")
    if _setup_has_detail(setup, "macro aligned"):
        flags.append("MACRO_OK")
    elif _setup_has_detail(setup, "macro opposing"):
        flags.append("MACRO_AGAINST")
    if _setup_has_detail(setup, "m5 not aligned", "no upper tf alignment"):
        flags.append("THIN_ALIGN")
    return flags


def _macro_flow_state(pair: str, direction: str) -> tuple[str, str]:
    try:
        strengths = _calc_currency_strength()
        base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        base_strength = float(strengths.get(base, 0.0) or 0.0)
        quote_strength = float(strengths.get(quote, 0.0) or 0.0)
    except Exception:
        return "FLOW_UNKNOWN", "macro flow unavailable"

    gap = base_strength - quote_strength
    if direction == "LONG":
        aligned = gap >= 0.30
        against = gap <= -0.30
    else:
        aligned = gap <= -0.30
        against = gap >= 0.30

    if aligned:
        return "FLOW_ALIGNED", f"macro flow aligned ({base}{base_strength:+.2f} vs {quote}{quote_strength:+.2f})"
    if against:
        return "FLOW_AGAINST", f"macro flow against ({base}{base_strength:+.2f} vs {quote}{quote_strength:+.2f})"
    return "FLOW_NEUTRAL", f"macro flow mixed ({base}{base_strength:+.2f} vs {quote}{quote_strength:+.2f})"


def _structure_state(setup: dict) -> tuple[str, str]:
    archetype = _setup_archetype(setup)
    wave = str(setup.get("wave") or ("counter" if setup.get("is_counter") else "auto"))
    align_count = int(setup.get("mtf_aligned", 0) or 0)
    if align_count >= 3:
        align_label = "ALIGN_3PLUS"
        align_text = "3TF aligned"
    elif align_count == 2:
        align_label = "ALIGN_2TF"
        align_text = "2TF aligned"
    elif align_count == 1:
        align_label = "ALIGN_1TF"
        align_text = "1TF aligned"
    else:
        align_label = "ALIGN_0TF"
        align_text = "no upper-TF alignment"

    if _setup_has_detail(setup, "h4 extreme", "bb lower band", "bb upper band"):
        structure_zone = "EDGE_ZONE"
        zone_text = "edge zone / extreme"
    elif _setup_has_detail(setup, "divergence", "cci extreme"):
        structure_zone = "REVERSAL_ZONE"
        zone_text = "reversal zone"
    else:
        structure_zone = "PLAIN_ZONE"
        zone_text = "plain structure"

    return (
        "|".join(
            (
                _compact_token(archetype),
                _compact_token(f"WAVE_{wave}"),
                align_label,
                structure_zone,
            )
        ),
        f"{archetype.lower()} | {wave} | {align_text} | {zone_text}",
    )


def _trigger_state(setup: dict, execution_plan: dict) -> tuple[str, str]:
    style = str((execution_plan or {}).get("style") or "PASS").upper()
    if _setup_has_detail(setup, "reversal confirmed"):
        trigger_shape = "REVERSAL_CONFIRMED"
        trigger_text = "reversal confirmed"
    elif _setup_has_detail(setup, "partial signal", "divergence", "technical confluence"):
        trigger_shape = "REVERSAL_BUILDING"
        trigger_text = "reversal building"
    elif _setup_has_detail(setup, "wait for timing", "no reversal yet", "m5 not aligned"):
        trigger_shape = "WAITING_FOR_PRINT"
        trigger_text = "waiting for the print"
    elif int(setup.get("mtf_aligned", 0) or 0) >= 2:
        trigger_shape = "CONTINUATION_READY"
        trigger_text = "continuation timing"
    else:
        trigger_shape = "STRUCTURE_TEST"
        trigger_text = "structure test only"

    proof = {
        "MARKET": "PRINT_LIVE",
        "STOP-ENTRY": "TRIGGER_ARMED",
        "LIMIT": "PRICE_ARMED",
        "PASS": "BLOCKED",
    }.get(style, "BLOCKED")
    proof_text = {
        "PRINT_LIVE": "proof already live",
        "TRIGGER_ARMED": "needs breakout / reclaim print",
        "PRICE_ARMED": "needs first-defense price",
        "BLOCKED": "no live trigger",
    }[proof]
    return (
        "|".join((trigger_shape, proof)),
        f"{trigger_text} | {proof_text}",
    )


def _vehicle_state(pair: str, setup: dict, execution_plan: dict) -> tuple[str, str]:
    style = str((execution_plan or {}).get("style") or "PASS").upper()
    spread_pips = setup.get("spread_pips")
    spread_ratio = setup.get("spread_ratio")
    tight_spread, normal_spread = _market_spread_bands(pair, spread_pips, spread_ratio)
    if tight_spread:
        spread_state = "SPREAD_TIGHT"
        spread_text = "tight spread"
    elif normal_spread:
        spread_state = "SPREAD_NORMAL"
        spread_text = "normal spread"
    else:
        spread_state = "SPREAD_WIDE"
        spread_text = "wide spread"

    allocation_band = str(setup.get("allocation_band") or setup.get("allocation_grade") or "UNKNOWN")
    return (
        "|".join((_compact_token(style), spread_state, _compact_token(f"BAND_{allocation_band}"))),
        f"{style} | {spread_text} | band {allocation_band}",
    )


def _thesis_age_label(feedback: dict | None) -> tuple[str, str]:
    if not feedback:
        return "FRESH", "fresh thesis"

    status = str(feedback.get("status") or "").lower()
    if status == "blocked":
        return "STALE_BLOCKED", "stale / blocked after fresh loss"
    if status == "family_headwind":
        return "PRESSURED", "family under pressure"
    if status == "layer_headwind":
        layer = str(feedback.get("layer") or "layer").lower()
        return "RECYCLE_LAYER", f"recycled {layer} layer"

    exact_rows = feedback.get("exact_rows") or []
    family_rows = feedback.get("family_rows") or []
    if exact_rows:
        return "RECYCLE_EXACT", "recycled exact thesis"
    if family_rows:
        return "FAMILIAR", "familiar family"
    return "FRESH", "fresh thesis"


def _build_thesis_context(
    pair: str,
    direction: str,
    setup: dict,
    learning_profile: dict,
    execution_plan: dict,
) -> dict:
    session_bucket = str(learning_profile.get("current_session") or "unknown")
    regime = str(learning_profile.get("current_regime") or "unknown")
    market_state, market_label = _macro_flow_state(pair, direction)
    structure_state, structure_label = _structure_state(setup)
    trigger_state, trigger_label = _trigger_state(setup, execution_plan)
    vehicle_state, vehicle_label = _vehicle_state(pair, setup, execution_plan)
    flags = _setup_flags(setup)

    family_parts = [
        f"session_{session_bucket}",
        f"regime_{regime}",
        market_state,
        structure_state,
    ]
    key_parts = family_parts + [
        trigger_state,
        vehicle_state,
    ]
    key_parts.extend(_compact_token(flag) for flag in flags)

    return {
        "family": "|".join(_compact_token(part) for part in family_parts),
        "key": "|".join(_compact_token(part) for part in key_parts),
        "family_label": f"{session_bucket}/{regime} | {market_label} | {structure_label}",
        "key_label": f"{trigger_label} | {vehicle_label}",
        "market": market_state,
        "market_label": f"{session_bucket}/{regime} | {market_label}",
        "structure": structure_state,
        "structure_label": structure_label,
        "trigger": trigger_state,
        "trigger_label": trigger_label,
        "vehicle": vehicle_state,
        "vehicle_label": vehicle_label,
        "age": "FRESH",
        "age_label": "fresh thesis",
        "flags": flags,
        "raw_execution_style": str(execution_plan.get("style") or "PASS"),
    }


def _parse_local_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _merge_note(existing: str | None, extra: str) -> str:
    existing_text = " ".join(str(existing or "").split()).strip()
    extra_text = " ".join(str(extra or "").split()).strip()
    if not existing_text:
        return extra_text
    if not extra_text or extra_text in existing_text:
        return existing_text
    return f"{existing_text}; {extra_text}"


def _append_hard_execution_blocker(execution_plan: dict, reason: str | None) -> None:
    text = " ".join(str(reason or "").split()).strip()
    if not text:
        return
    blockers = list(execution_plan.get("hard_blockers") or [])
    if text not in blockers:
        blockers.append(text)
    execution_plan["hard_blockers"] = blockers


def _collect_hard_execution_blockers(execution_plan: dict | None) -> list[str]:
    blockers = []
    for item in list((execution_plan or {}).get("hard_blockers") or []):
        text = " ".join(str(item or "").split()).strip()
        if text and text not in blockers:
            blockers.append(text)
    return blockers


def _loss_streak(rows: list[dict]) -> tuple[int, float]:
    streak = 0
    total = 0.0
    for row in rows:
        pl = float(row.get("pl") or 0.0)
        if pl >= 0:
            break
        streak += 1
        total += pl
    return streak, total


def _recent_outcome_rows(
    conn,
    pair: str,
    direction: str,
    *,
    thesis_key: str | None = None,
    thesis_family: str | None = None,
    layer_filters: dict[str, str] | None = None,
    level: str | None = None,
    exact_only: bool = False,
    family_only: bool = False,
    limit: int = THESIS_HISTORY_LIMIT,
) -> list[dict]:
    lookback = live_history_start(THESIS_LOOKBACK_DAYS)
    conditions = ["pair = ?", "direction = ?", "pl IS NOT NULL", "session_date >= ?"]
    params: list[object] = [pair, direction, lookback]

    if exact_only and thesis_key:
        conditions.append("COALESCE(thesis_key, '') = ?")
        params.append(thesis_key)
    elif family_only and thesis_family:
        conditions.append("COALESCE(thesis_family, '') = ?")
        params.append(thesis_family)
        if thesis_key:
            conditions.append("(thesis_key IS NULL OR thesis_key != ?)")
            params.append(thesis_key)
    else:
        scoped = []
        if thesis_key:
            scoped.append("COALESCE(thesis_key, '') = ?")
            params.append(thesis_key)
        if thesis_family:
            scoped.append("COALESCE(thesis_family, '') = ?")
            params.append(thesis_family)
        for column, value in (layer_filters or {}).items():
            if value:
                scoped.append(f"COALESCE({column}, '') = ?")
                params.append(value)
        if not scoped and level:
            scoped.append("pretrade_level = ?")
            params.append(level)
        if scoped:
            conditions.append("(" + " OR ".join(scoped) + ")")

    params.append(limit)
    return fetchall_dict(
        conn,
        f"""SELECT id, session_date, created_at, pl, lesson_from_review, thesis,
                  thesis_key, thesis_family, pretrade_level, execution_style,
                  thesis_market, thesis_structure, thesis_trigger, thesis_vehicle,
                  thesis_age, collapse_layer, collapse_note
           FROM pretrade_outcomes
           WHERE {' AND '.join(conditions)}
           ORDER BY id DESC
           LIMIT ?""",
        tuple(params),
    )


def _collapse_layer_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"market": 0, "structure": 0, "trigger": 0, "vehicle": 0, "aging": 0}
    for row in rows:
        layer = str(row.get("collapse_layer") or "").strip().lower()
        if layer in counts:
            counts[layer] += 1
    return counts


def _dominant_layer(rows: list[dict], candidates: tuple[str, ...]) -> str | None:
    counts = _collapse_layer_counts(rows)
    winner = None
    winner_count = 0
    for candidate in candidates:
        count = counts.get(candidate, 0)
        if count > winner_count:
            winner = candidate
            winner_count = count
    return winner if winner_count > 0 else None


def _review_feedback_text(row: dict) -> str:
    bits = [str(row.get("collapse_note") or "").strip(), str(row.get("lesson_from_review") or "").strip()]
    return " ".join(bit for bit in bits if bit).strip()


def _review_supports_exact_recycle(row: dict) -> bool:
    text = _review_feedback_text(row).lower()
    if not text:
        return False
    patterns = (
        "market/structure later recovered",
        "keep the family alive",
        "trigger wobble",
        "first wobble was misread",
        "require a new print",
        "wait for a better vehicle",
        "vehicle was wrong",
    )
    return any(pattern in text for pattern in patterns)


def _soft_exact_thesis_feedback(
    exact_rows: list[dict],
    family_rows: list[dict],
    trigger_rows: list[dict],
    vehicle_rows: list[dict],
    *,
    exact_streak: int,
    exact_total: float,
) -> dict | None:
    if not exact_rows:
        return None

    last_exact = exact_rows[0]
    layer = str(last_exact.get("collapse_layer") or "").strip().lower()
    if layer not in {"trigger", "vehicle"}:
        return None

    dominant_deep = _dominant_layer(exact_rows, ("market", "structure", "aging"))
    if dominant_deep:
        return None

    review_supports_recycle = _review_supports_exact_recycle(last_exact)
    dominant_shallow = _dominant_layer(exact_rows, ("trigger", "vehicle"))
    if not review_supports_recycle and (dominant_shallow != layer or len(exact_rows) < 2):
        return None

    last_pl = float(last_exact.get("pl") or 0.0)
    session_date = str(last_exact.get("session_date") or "?")
    if exact_streak >= 2:
        reason = (
            f"same thesis lost {exact_streak} straight times ({exact_total:+,.0f} JPY), "
            f"but the dominant failure layer is still {layer}; "
        )
    else:
        reason = (
            f"same thesis just lost {last_pl:+,.0f} JPY on {session_date}, "
            f"but the failure was tagged as {layer}-layer damage; "
        )

    if layer == "vehicle":
        reason += "keep the direction alive only with a different vehicle, not another paid repeat of the same seat"
    else:
        reason += "keep the direction alive only after a materially new print, not the same trigger recycle"

    review_note = _review_feedback_text(last_exact)
    if review_note:
        reason = f"{reason} | last review: {_clip_text(review_note, limit=90)}"

    return {
        "status": "layer_headwind",
        "layer": layer,
        "reason": reason,
        "exact_rows": exact_rows,
        "family_rows": family_rows,
        "trigger_rows": trigger_rows,
        "vehicle_rows": vehicle_rows,
    }


def _recent_thesis_feedback(
    conn,
    pair: str,
    direction: str,
    *,
    level: str,
    thesis_context: dict,
) -> dict | None:
    thesis_key = thesis_context.get("key")
    thesis_family = thesis_context.get("family")
    exact_rows = _recent_outcome_rows(
        conn,
        pair,
        direction,
        thesis_key=thesis_key,
        exact_only=True,
        limit=4,
    ) if thesis_key else []
    family_rows = _recent_outcome_rows(
        conn,
        pair,
        direction,
        thesis_key=thesis_key,
        thesis_family=thesis_family,
        family_only=True,
        limit=4,
    ) if thesis_family else []
    trigger_rows = _recent_outcome_rows(
        conn,
        pair,
        direction,
        layer_filters={"thesis_trigger": thesis_context.get("trigger")},
        limit=4,
    ) if thesis_context.get("trigger") else []
    vehicle_rows = _recent_outcome_rows(
        conn,
        pair,
        direction,
        layer_filters={"thesis_vehicle": thesis_context.get("vehicle")},
        limit=4,
    ) if thesis_context.get("vehicle") else []

    if exact_rows:
        last_exact = exact_rows[0]
        exact_streak, exact_total = _loss_streak(exact_rows)
        soft_exact_feedback = _soft_exact_thesis_feedback(
            exact_rows,
            family_rows,
            trigger_rows,
            vehicle_rows,
            exact_streak=exact_streak,
            exact_total=exact_total,
        )
        last_exact_pl = float(last_exact.get("pl") or 0.0)
        last_created = _parse_local_created_at(last_exact.get("created_at"))
        exact_is_fresh_loss = (
            last_exact_pl < 0
            and (
                (
                    last_created is not None
                    and datetime.now() - last_created <= timedelta(hours=THESIS_EXACT_BLOCK_HOURS)
                )
                or (
                    last_created is None
                    and last_exact.get("session_date") == str(date.today())
                )
            )
        )
        if exact_is_fresh_loss:
            if soft_exact_feedback:
                return soft_exact_feedback
            reason = (
                f"same thesis just lost {last_exact_pl:+,.0f} JPY and is still inside the "
                f"{THESIS_EXACT_BLOCK_HOURS}h cool-off window; write the materially new state change before re-entry"
            )
            if last_exact.get("lesson_from_review"):
                reason = f"{reason} | last review: {_clip_text(last_exact['lesson_from_review'], limit=90)}"
            return {
                "status": "blocked",
                "reason": reason,
                "exact_rows": exact_rows,
                "family_rows": family_rows,
                "trigger_rows": trigger_rows,
                "vehicle_rows": vehicle_rows,
            }
        if exact_streak >= 2:
            if soft_exact_feedback:
                return soft_exact_feedback
            reason = (
                f"same thesis lost {exact_streak} straight times ({exact_total:+,.0f} JPY); "
                "stop recycling the same trigger"
            )
            return {
                "status": "blocked",
                "reason": reason,
                "exact_rows": exact_rows,
                "family_rows": family_rows,
                "trigger_rows": trigger_rows,
                "vehicle_rows": vehicle_rows,
            }

    if family_rows:
        family_streak, family_total = _loss_streak(family_rows)
        recent_family_wins = sum(1 for row in family_rows[:3] if float(row.get("pl") or 0.0) > 0)
        if family_streak >= THESIS_FAMILY_HEADWIND_STREAK and recent_family_wins == 0:
            dominant_deep = _dominant_layer(family_rows, ("market", "structure", "aging"))
            dominant_shallow = _dominant_layer(family_rows, ("trigger", "vehicle"))
            if dominant_shallow and not dominant_deep:
                return {
                    "status": "layer_headwind",
                    "layer": dominant_shallow,
                    "reason": (
                        f"same market/structure family is losing through the {dominant_shallow} layer; "
                        "keep the direction alive, but require a new trigger/vehicle instead of repeating the old seat"
                    ),
                    "exact_rows": exact_rows,
                    "family_rows": family_rows,
                    "trigger_rows": trigger_rows,
                    "vehicle_rows": vehicle_rows,
                }
            return {
                "status": "family_headwind",
                "reason": (
                    f"same market/structure family lost {family_streak} straight times ({family_total:+,.0f} JPY); "
                    "keep it watch/B-only until the market-state or structure materially changes"
                ),
                "exact_rows": exact_rows,
                "family_rows": family_rows,
                "trigger_rows": trigger_rows,
                "vehicle_rows": vehicle_rows,
            }

    if trigger_rows:
        streak, total = _loss_streak(trigger_rows)
        if streak >= 3:
            return {
                "status": "layer_headwind",
                "layer": "trigger",
                "reason": (
                    f"same trigger layer lost {streak} straight times ({total:+,.0f} JPY); "
                    "do not recycle the same trigger even if the broader direction still looks alive"
                ),
                "exact_rows": exact_rows,
                "family_rows": family_rows,
                "trigger_rows": trigger_rows,
                "vehicle_rows": vehicle_rows,
            }

    if vehicle_rows:
        streak, total = _loss_streak(vehicle_rows)
        if streak >= 3:
            return {
                "status": "layer_headwind",
                "layer": "vehicle",
                "reason": (
                    f"same vehicle layer lost {streak} straight times ({total:+,.0f} JPY); "
                    "switch the execution vehicle before taking fresh risk again"
                ),
                "exact_rows": exact_rows,
                "family_rows": family_rows,
                "trigger_rows": trigger_rows,
                "vehicle_rows": vehicle_rows,
            }

    return None


def _apply_thesis_feedback_guard(
    warnings: list[str],
    setup: dict,
    learning_profile: dict,
    execution_plan: dict,
    feedback: dict | None,
) -> tuple[dict, dict, dict]:
    if not feedback or not feedback.get("status"):
        return setup, learning_profile, execution_plan

    setup = _clone_setup_for_contest(setup)
    learning_profile = dict(learning_profile or {})
    execution_plan = dict(execution_plan or {})
    guard_reason = str(feedback.get("reason") or "").strip()
    if guard_reason:
        warnings.append(f"⚠️ THESIS GUARD: {guard_reason}")
        setup["details"].append(f"Thesis guard: {guard_reason}")
        execution_plan["note"] = _merge_note(execution_plan.get("note"), guard_reason)
        learning_profile["thesis_guard"] = guard_reason
        execution_plan["thesis_guard"] = feedback.get("status")

    current_score = int(learning_profile.get("learning_score", 0) or 0)
    if feedback.get("status") == "blocked":
        execution_plan["style"] = "PASS"
        learning_profile["learning_score"] = max(0, current_score - 28)
        learning_profile["allocation_cap"] = CAP_LABELS["pass"]
        learning_profile["verdict"] = (
            f"{learning_profile.get('verdict', 'limited history')} | exact thesis blocked"
        )
        return setup, learning_profile, execution_plan

    current_style = str(execution_plan.get("style", "PASS"))
    regime = str(learning_profile.get("current_regime") or "").lower()
    demoted_style = "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"
    if feedback.get("status") == "layer_headwind":
        layer = str(feedback.get("layer") or "trigger")
        if layer == "vehicle":
            execution_plan["style"] = "LIMIT"
            learning_profile["learning_score"] = max(0, current_score - 8)
            learning_profile["allocation_cap"] = CAP_LABELS["b_only"]
            learning_profile["verdict"] = (
                f"{learning_profile.get('verdict', 'limited history')} | vehicle headwind"
            )
            return setup, learning_profile, execution_plan
        if EXECUTION_STYLE_RANK.get(current_style, 0) > EXECUTION_STYLE_RANK.get(demoted_style, 0):
            execution_plan["style"] = demoted_style
        learning_profile["learning_score"] = max(0, current_score - 10)
        learning_profile["allocation_cap"] = CAP_LABELS["b_only"]
        learning_profile["verdict"] = (
            f"{learning_profile.get('verdict', 'limited history')} | {layer} layer headwind"
        )
        return setup, learning_profile, execution_plan

    if EXECUTION_STYLE_RANK.get(current_style, 0) > EXECUTION_STYLE_RANK.get(demoted_style, 0):
        execution_plan["style"] = demoted_style
    learning_profile["learning_score"] = max(0, current_score - 12)
    learning_profile["allocation_cap"] = CAP_LABELS["b_only"]
    learning_profile["verdict"] = (
        f"{learning_profile.get('verdict', 'limited history')} | market/structure family headwind"
    )
    return setup, learning_profile, execution_plan


def _current_session_bucket(now_utc: datetime) -> str:
    hour = now_utc.hour
    if 0 <= hour < 7:
        return "tokyo"
    if 7 <= hour < 15:
        return "london"
    if 15 <= hour < 22:
        return "newyork"
    return "late"


def _infer_current_pair_regime(pair: str) -> str | None:
    tfs = _load_technicals(pair)
    m5 = tfs.get("M5", {})
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


def _format_context_stat(stats: dict | None, label: str) -> str:
    if not stats or int(stats.get("count", 0) or 0) <= 0:
        return f"{label}: no sample"
    return (
        f"{label}: WR {float(stats.get('win_rate', 0.0))*100:.0f}% "
        f"EV {float(stats.get('expectancy', 0.0)):+.0f} n={int(stats.get('count', 0))}"
    )


def _context_signal(stats: dict | None, *, min_count: int = 3) -> tuple[int, int, str]:
    if not stats:
        return 0, 0, "no sample"

    count = int(stats.get("count", 0) or 0)
    if count < min_count:
        return 0, 0, "thin"

    ev = float(stats.get("expectancy", 0.0) or 0.0)
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


def _learning_cap_to_grade(label: str) -> str:
    if label == CAP_LABELS["pass"]:
        return "C"
    if label in {CAP_LABELS["b_only"], CAP_LABELS["b_scout"]}:
        return "B"
    return "A"


def _pip_factor(pair: str) -> int:
    return 100 if str(pair).endswith("JPY") else 10000


def _parse_iso_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    match = re.match(r"^(.*?\.\d{6})\d+([+-]\d\d:\d\d)$", text)
    if match:
        text = f"{match.group(1)}{match.group(2)}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _probe_age_seconds(payload: dict) -> float | None:
    fetched_at = _parse_iso_utc(payload.get("fetched_at"))
    if not fetched_at:
        return None
    age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
    return max(0.0, age)


def _cached_live_tape_summary(pair: str) -> dict | None:
    try:
        payload = json.loads(PROBE_CACHE_PATH.read_text())
    except Exception:
        return None
    summary = ((payload.get("pairs") or {}) or {}).get(pair)
    if not isinstance(summary, dict):
        return None
    age_sec = _probe_age_seconds(payload)
    if age_sec is None or age_sec > PRICING_PROBE_MAX_AGE_SEC:
        return None
    if int(summary.get("samples", 0) or 0) < MIN_TAPE_SAMPLES:
        return None
    enriched = dict(summary)
    enriched["probe_mode"] = payload.get("mode_used") or payload.get("mode_requested") or "cache"
    enriched["probe_age_sec"] = round(age_sec, 1)
    return enriched


def _fresh_live_tape_summary(pair: str) -> dict:
    cached = _cached_live_tape_summary(pair)
    if cached:
        return cached
    try:
        payload = probe_market(
            pairs=[pair],
            mode="stream",
            samples=PRICING_PROBE_SAMPLES,
            interval_sec=PRICING_PROBE_INTERVAL_SEC,
            duration_sec=PRICING_PROBE_DURATION_SEC,
            write_cache=False,
        )
        summary = dict((((payload.get("pairs") or {}) or {}).get(pair) or {}))
    except Exception as exc:
        summary = {
            "pair": pair,
            "samples": 0,
            "bias": "unknown",
            "tape": "unavailable",
            "error": str(exc),
        }
        payload = {}
    if not summary:
        summary = {
            "pair": pair,
            "samples": 0,
            "bias": "unknown",
            "tape": "unavailable",
            "error": "missing pair summary",
        }
    summary["probe_mode"] = payload.get("mode_used") or payload.get("mode_requested") or "stream"
    summary["probe_age_sec"] = 0.0
    return summary


def _tape_supports_direction(direction: str, bias: str | None) -> bool:
    if direction == "LONG":
        return bias == "buyers pressing"
    return bias == "sellers pressing"


def _tape_opposes_direction(direction: str, bias: str | None) -> bool:
    if direction == "LONG":
        return bias == "sellers pressing"
    return bias == "buyers pressing"


def _tape_fallback_style(regime: str | None) -> str:
    return "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"


def _live_tape_brief(summary: dict | None) -> str:
    if not summary:
        return "live tape unavailable"
    if summary.get("tape") == "unavailable":
        mode = summary.get("probe_mode") or "?"
        error = summary.get("error") or "no current pricing read"
        return f"unavailable ({error}; mode={mode})"

    mode = summary.get("probe_mode") or "?"
    age = summary.get("probe_age_sec")
    age_text = ""
    if age is not None:
        age_text = f", age {float(age):.0f}s"
    return (
        f"{summary.get('bias')} / {summary.get('tape')} "
        f"(move {float(summary.get('delta_pips', 0.0)):+.1f}pip, "
        f"range {float(summary.get('range_pips', 0.0)):.1f}pip, "
        f"spread {float(summary.get('avg_spread_pips', 0.0)):.1f}/"
        f"{float(summary.get('max_spread_pips', 0.0)):.1f}pip, "
        f"mode={mode}{age_text})"
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


def _quiet_stable_market_scout_reason(
    pair: str,
    direction: str,
    learning_profile: dict,
    execution_plan: dict,
    summary: dict | None,
    *,
    risk_level: str | None = None,
    promotion_pressure: dict | None = None,
) -> str | None:
    style = str((execution_plan or {}).get("style") or "PASS").upper()
    if style != "STOP-ENTRY":
        return None

    regime = str(learning_profile.get("current_regime") or "").lower()
    if regime not in {"trending", "transition", "squeeze"}:
        return None
    if str((summary or {}).get("tape") or "") != "quiet / stable":
        return None
    if str(risk_level or "").upper() == "HIGH":
        return None

    bias = str((summary or {}).get("bias") or "unknown")
    if _tape_opposes_direction(direction, bias):
        return None

    learning_score = int(learning_profile.get("learning_score", 0) or 0)
    cap_rank = _cap_rank(learning_profile.get("allocation_cap", CAP_LABELS["b_scout"]))
    trade_count = int(learning_profile.get("trade_count", 0) or 0)
    trade_ev = float(learning_profile.get("trade_ev", 0.0) or 0.0)
    context_bias = str(learning_profile.get("context_bias") or "neutral")
    pressure_rank = int((promotion_pressure or {}).get("rank", 0) or 0)
    pressure_note = str((promotion_pressure or {}).get("note") or "").strip()
    has_support = trade_count >= 5 or trade_ev > 0 or bool(learning_profile.get("has_exact")) or learning_profile.get("state") == "confirmed"
    if learning_score < 45:
        return None
    if context_bias == "headwind" and trade_count < 5 and trade_ev <= 0:
        return None

    if pair in DIRECT_USD_PAIRS:
        if cap_rank < 1 and not has_support:
            return None
        note = (
            "quiet-stable direct-USD seat can pay as a small market scout "
            "instead of another trigger-only rewrite"
        )
        if pressure_note:
            note = f"{note} ({pressure_note})"
        return note

    if cap_rank < 1:
        return None
    if learning_score < 70:
        return None
    if pressure_rank < 18:
        return None
    if not has_support and pressure_rank < 24:
        return None

    note = (
        "quiet-stable repeat-pressure seat can pay as a small market scout "
        "instead of another trigger-only rewrite"
    )
    if pressure_note:
        note = f"{note} ({pressure_note})"
    return note


def _pass_cap_scout_participation_plan(
    pair: str,
    learning_profile: dict,
    *,
    risk_level: str,
    normal_market_spread: bool,
    promotion_pressure: dict | None = None,
) -> tuple[str | None, str | None]:
    if str(risk_level or "").upper() == "HIGH":
        return None, None
    if not normal_market_spread:
        return None, None

    regime = str(learning_profile.get("current_regime") or "").lower()
    if regime not in {"trending", "transition", "squeeze"}:
        return None, None

    learning_score = int(learning_profile.get("learning_score", 0) or 0)
    trade_count = int(learning_profile.get("trade_count", 0) or 0)
    trade_ev = float(learning_profile.get("trade_ev", 0.0) or 0.0)
    context_bias = str(learning_profile.get("context_bias") or "neutral")
    pressure_rank = int((promotion_pressure or {}).get("rank", 0) or 0)
    pressure_note = str((promotion_pressure or {}).get("note") or "").strip()
    has_support = trade_count >= 5 or trade_ev > 0 or bool(learning_profile.get("has_exact")) or learning_profile.get("state") == "confirmed"
    if pair in DIRECT_USD_PAIRS:
        if learning_score < 45 or not has_support:
            return None, None
        if context_bias == "headwind" and trade_count < 5 and trade_ev <= 0 and pressure_rank < 16:
            return None, None
        note = "pass-cap / C-grade direct-USD seat still deserves a thin scout; arm the trigger instead of leaving it as prose"
        if pressure_note:
            note = f"{note} ({pressure_note})"
        return "STOP-ENTRY", note

    if pressure_rank < 12 or learning_score < 40 or not has_support:
        return None, None
    if context_bias == "headwind" and trade_ev <= 0 and pressure_rank < 16:
        return None, None

    note = (
        "repeat audit / missed-seat pressure is too strong to leave this pass-cap seat as prose; "
        "re-open it as a thin trigger-first scout"
    )
    if pressure_note:
        note = f"{note} ({pressure_note})"
    return "STOP-ENTRY", note


def _execution_style_context_label(style: str | None) -> str:
    return f"{str(style or 'unknown').upper()} execution"


def _apply_live_tape_execution_guard(
    pair: str,
    direction: str,
    learning_profile: dict,
    execution_plan: dict,
    live_tape: dict | None,
    *,
    risk_level: str | None = None,
    warnings: list[str] | None = None,
) -> dict:
    execution_plan = dict(execution_plan or {})
    style = str(execution_plan.get("style", "PASS"))
    if style == "PASS":
        if live_tape:
            execution_plan["live_tape"] = live_tape
            execution_plan["live_tape_note"] = _live_tape_brief(live_tape)
        return execution_plan

    regime = str(learning_profile.get("current_regime") or "").lower() or None
    fallback_style = _tape_fallback_style(regime)
    summary = live_tape or {
        "pair": pair,
        "samples": 0,
        "bias": "unknown",
        "tape": "unavailable",
        "error": "no current pricing read",
    }
    execution_plan["live_tape"] = summary
    brief = _live_tape_brief(summary)
    execution_plan["live_tape_note"] = brief

    scout_reason = _quiet_stable_market_scout_reason(
        pair,
        direction,
        learning_profile,
        execution_plan,
        summary,
        risk_level=risk_level,
        promotion_pressure=learning_profile.get("promotion_pressure"),
    )
    if scout_reason:
        execution_plan["style"] = "MARKET"
        execution_plan["note"] = (
            f"{execution_plan.get('note', '').strip()}; {scout_reason}; {brief}"
        ).strip("; ")
        if warnings is not None:
            warnings.append(
                f"⚙️ MARKET SCOUT: {pair} {direction} upgraded from STOP-ENTRY to MARKET. {scout_reason}."
            )
        return execution_plan

    tape = str(summary.get("tape") or "unavailable")
    bias = str(summary.get("bias") or "unknown")
    samples = int(summary.get("samples", 0) or 0)
    demotion_note = None

    if style == "MARKET":
        if tape == "unavailable" or samples < MIN_TAPE_SAMPLES:
            demotion_note = "live tape is unavailable right now, so do not pay market without current confirmation"
        elif tape in {"spread unstable", "friction-dominated"}:
            demotion_note = f"live tape is {tape}, so market friction is too high to pay now"
        elif _tape_opposes_direction(direction, bias):
            demotion_note = (
                f"live tape is paying the other side ({brief}), so do not force a market fill into counter-pressure"
            )
        elif tape == "whipsaw / two-way" or (tape != "quiet / stable" and bias in {"two-way", "mixed"}):
            demotion_note = f"live tape is two-way ({brief}), so require trigger proof instead of a blind market fill"

        if demotion_note:
            execution_plan["style"] = fallback_style
            execution_plan["note"] = f"{demotion_note}; {brief}"
            execution_plan["live_tape_guard"] = demotion_note
            if warnings is not None:
                warnings.append(
                    f"⚠️ LIVE TAPE GUARD: {pair} {direction} MARKET demoted to {fallback_style}. "
                    f"{demotion_note}."
                )
            return execution_plan

        if tape == "quiet / stable":
            execution_plan["note"] = (
                f"{execution_plan.get('note', '').strip()}; live tape is quiet but stable ({brief}), "
                "so the chart/timeframe edge must justify paying market"
            ).strip("; ")
            return execution_plan

        if _tape_supports_direction(direction, bias):
            execution_plan["note"] = (
                f"{execution_plan.get('note', '').strip()}; live tape confirms {brief}".strip("; ")
            )
        return execution_plan

    if style == "STOP-ENTRY":
        if _tape_opposes_direction(direction, bias) and tape == "clean one-way":
            execution_plan["note"] = (
                f"{execution_plan.get('note', '').strip()}; live tape is still paying the other side ({brief}), "
                "so do not arm this without a real reclaim/break trigger"
            ).strip("; ")
        elif tape == "quiet / stable":
            execution_plan["note"] = (
                f"{execution_plan.get('note', '').strip()}; live tape is calm/stable ({brief}), "
                "so trigger proof still matters more than urgency"
            ).strip("; ")
        elif tape in {"spread unstable", "friction-dominated", "whipsaw / two-way"}:
            execution_plan["note"] = (
                f"{execution_plan.get('note', '').strip()}; live tape stays messy ({brief}), "
                "so trigger proof remains mandatory"
            ).strip("; ")
        return execution_plan

    if style == "LIMIT" and _tape_supports_direction(direction, bias) and tape == "clean one-way":
        execution_plan["note"] = (
            f"{execution_plan.get('note', '').strip()}; live tape is already one-way ({brief}), "
            "so this stays passive only if you still want price improvement"
        ).strip("; ")
    return execution_plan


def _apply_recent_execution_style_guard(
    conn,
    pair: str,
    direction: str,
    learning_profile: dict,
    execution_plan: dict,
    *,
    warnings: list[str] | None = None,
) -> dict:
    execution_plan = dict(execution_plan or {})
    style = str(execution_plan.get("style") or "PASS").upper()
    style_stat = execution_style_stats(conn, pair, direction, style)
    execution_plan["execution_style_stat"] = style_stat
    execution_plan["execution_style_context"] = _format_feedback_stat(
        style_stat,
        _execution_style_context_label(style),
    )
    day_kill_stat = _same_day_execution_style_stat(conn, pair, direction, style)
    execution_plan["execution_style_day_kill"] = day_kill_stat

    if day_kill_stat:
        if style == "MARKET":
            fallback_style = _tape_fallback_style(str(learning_profile.get("current_regime") or "").lower() or None)
            reason = (
                f"{_format_same_day_style_guard(style, day_kill_stat)}, "
                "so the market-chase lane is closed until it captures a seat again"
            )
        else:
            fallback_style = "LIMIT"
            reason = (
                f"{_format_same_day_style_guard(style, day_kill_stat)}, "
                "so keep the thesis alive only with a better-price LIMIT for the rest of today"
            )
        execution_plan["style"] = fallback_style
        execution_plan["note"] = _merge_note(execution_plan.get("note"), reason)
        execution_plan["execution_style_feedback_note"] = reason
        if warnings is not None:
            warnings.append(f"⚠️ VEHICLE FEEDBACK: {reason}")
        return execution_plan

    if style not in {"MARKET", "STOP-ENTRY"} or style_stat.get("stat_source") != "pretrade_recent":
        return execution_plan

    count = int(style_stat.get("count", 0) or 0)
    min_count = 2 if style == "MARKET" else 3
    if count < min_count:
        return execution_plan

    ev = float(style_stat.get("expectancy", style_stat.get("avg_pl", 0.0)) or 0.0)
    win_rate = float(style_stat.get("win_rate", 0.0) or 0.0)
    if ev >= 0 or win_rate > 0.40:
        return execution_plan

    if style == "MARKET":
        fallback_style = _tape_fallback_style(str(learning_profile.get("current_regime") or "").lower() or None)
        reason = (
            f"{_execution_style_context_label(style)} is still losing here "
            f"({_format_feedback_stat(style_stat, style)}), so do not pay market until the lane repairs"
        )
    else:
        fallback_style = "LIMIT"
        reason = (
            f"{_execution_style_context_label(style)} is still losing here "
            f"({_format_feedback_stat(style_stat, style)}), so keep the lane alive with a better-price LIMIT "
            "instead of a proof-chase trigger"
        )
    execution_plan["style"] = fallback_style
    execution_plan["note"] = _merge_note(execution_plan.get("note"), reason)
    execution_plan["execution_style_feedback_note"] = reason
    if warnings is not None:
        warnings.append(f"⚠️ VEHICLE FEEDBACK: {reason}")
    return execution_plan


def _planned_pips(
    pair: str,
    *,
    entry_price: float | None = None,
    tp_price: float | None = None,
    sl_price: float | None = None,
    tp_pips: float | None = None,
    sl_pips: float | None = None,
) -> tuple[float | None, float | None]:
    pip_factor = _pip_factor(pair)
    planned_tp = float(tp_pips) if tp_pips is not None else None
    planned_sl = float(sl_pips) if sl_pips is not None else None
    if planned_tp is None and entry_price is not None and tp_price is not None:
        planned_tp = abs(float(tp_price) - float(entry_price)) * pip_factor
    if planned_sl is None and entry_price is not None and sl_price is not None:
        planned_sl = abs(float(entry_price) - float(sl_price)) * pip_factor
    return planned_tp, planned_sl

# --- SQL Layer: Statistics from structured data ---

def payoff_metrics(pls: list[float]) -> dict:
    """Realized payoff quality from closed-trade P&L.

    R:R alone is not enough. What matters is whether the realized win rate and
    realized payout shape produce positive expectancy after repeated trades.
    """
    if not pls:
        return {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_pl": 0.0,
            "total_pl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "rr_ratio": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "break_even_win_rate": None,
        }

    wins = [pl for pl in pls if pl > 0]
    losses = [pl for pl in pls if pl < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss < 0:
        rr_ratio = avg_win / abs(avg_loss)
    elif avg_win > 0 and not losses:
        rr_ratio = float("inf")
    else:
        rr_ratio = 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    break_even_win_rate = None
    if avg_win > 0 and avg_loss < 0:
        break_even_win_rate = abs(avg_loss) / (avg_win + abs(avg_loss))
    elif avg_win > 0 and not losses:
        break_even_win_rate = 0.0

    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float("inf")

    return {
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pls),
        "avg_pl": sum(pls) / len(pls),
        "total_pl": sum(pls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio,
        "profit_factor": profit_factor,
        "expectancy": sum(pls) / len(pls),
        "break_even_win_rate": break_even_win_rate,
    }


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


def _stat_payload(
    rows: list[dict],
    *,
    source: str,
    lookback_days: int | None,
    lesson_key: str = "lesson",
    include_no_sl: bool = False,
    context_label: str | None = None,
) -> dict:
    if not rows:
        return {
            "count": 0,
            "stat_source": source,
            "lookback_days": lookback_days,
            "context_label": context_label,
            "stat_source_label": _stat_source_label(
                source,
                lookback_days,
                context_label=context_label,
            ),
        }

    pls = [float(row["pl"]) for row in rows if row.get("pl") is not None]
    metrics = payoff_metrics(pls)
    payload = {
        "count": len(pls),
        **metrics,
        "worst": min(pls),
        "best": max(pls),
        "lessons": [row.get(lesson_key) for row in rows if row.get(lesson_key)],
        "no_sl_count": 0,
        "stat_source": source,
        "lookback_days": lookback_days,
        "context_label": context_label,
        "stat_source_label": _stat_source_label(
            source,
            lookback_days,
            context_label=context_label,
        ),
    }
    if include_no_sl:
        payload["no_sl_count"] = sum(1 for row in rows if int(row.get("had_sl", 0) or 0) == 0)
    return payload


def _format_feedback_stat(stats: dict | None, label: str = "pair") -> str:
    if not stats or int(stats.get("count", 0) or 0) <= 0:
        return f"{label}: no sample"
    return (
        f"{stats.get('stat_source_label', label)}: WR {float(stats.get('win_rate', 0.0))*100:.0f}% "
        f"EV {float(stats.get('expectancy', stats.get('avg_pl', 0.0))):+.0f} "
        f"n={int(stats.get('count', 0))}"
    )


def _recent_feedback_override(stats: dict | None) -> dict | None:
    if not stats or stats.get("stat_source") != "pretrade_recent":
        return None

    count = int(stats.get("count", 0) or 0)
    if count < 4:
        return None

    ev = float(stats.get("expectancy", stats.get("avg_pl", 0.0)) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    summary = _format_feedback_stat(stats, "pair")

    if ev < 0 and win_rate <= 0.40:
        return {"tier": "hard_headwind", "reason": summary}
    if ev <= 0 or win_rate <= 0.35:
        return {"tier": "soft_headwind", "reason": summary}
    return None


def trade_stats(conn, pair: str, direction: str) -> dict:
    """Win rate and average P&L by pair x direction"""
    recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
    recent_pretrade = fetchall_dict(
        conn,
        """SELECT pl, lesson_from_review AS lesson
           FROM pretrade_outcomes
           WHERE pair = ?
             AND direction = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, recent_pretrade_cutoff),
    )
    if len(recent_pretrade) >= RECENT_FEEDBACK_MIN_COUNT:
        return _stat_payload(
            recent_pretrade,
            source="pretrade_recent",
            lookback_days=RECENT_PRETRADE_LOOKBACK_DAYS,
        )

    recent_trade_cutoff = live_history_start(RECENT_TRADE_LOOKBACK_DAYS)
    recent_trades = fetchall_dict(
        conn,
        """SELECT pl, regime, had_sl, entry_type, lesson
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, recent_trade_cutoff),
    )
    if recent_trades:
        return _stat_payload(
            recent_trades,
            source="trades_recent",
            lookback_days=RECENT_TRADE_LOOKBACK_DAYS,
            include_no_sl=True,
        )

    all_trades = fetchall_dict(
        conn,
        """SELECT pl, regime, had_sl, entry_type, lesson
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, live_history_start(None)),
    )
    return _stat_payload(
        all_trades,
        source="trades_all",
        lookback_days=None,
        include_no_sl=True,
    )


def tape_stats(conn, pair: str, direction: str, tape_bucket: str) -> dict:
    """Recent matched results for the current live-tape bucket."""
    context_label = _live_tape_bucket_label(tape_bucket)
    recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
    recent_pretrade = fetchall_dict(
        conn,
        """SELECT pl, lesson_from_review AS lesson
           FROM pretrade_outcomes
           WHERE pair = ?
             AND direction = ?
             AND COALESCE(live_tape_bucket, '') = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, tape_bucket, recent_pretrade_cutoff),
    )
    return _stat_payload(
        recent_pretrade,
        source="pretrade_recent",
        lookback_days=RECENT_PRETRADE_LOOKBACK_DAYS,
        context_label=context_label,
    )


def execution_style_stats(conn, pair: str, direction: str, style: str) -> dict:
    """Recent matched results for the planned execution vehicle."""
    context_label = _execution_style_context_label(style)
    recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
    recent_pretrade = fetchall_dict(
        conn,
        """SELECT pl, lesson_from_review AS lesson
           FROM pretrade_outcomes
           WHERE pair = ?
             AND direction = ?
             AND COALESCE(execution_style, '') = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, style, recent_pretrade_cutoff),
    )
    return _stat_payload(
        recent_pretrade,
        source="pretrade_recent",
        lookback_days=RECENT_PRETRADE_LOOKBACK_DAYS,
        context_label=context_label,
    )


def _today_jst_date() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _today_session_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _same_day_execution_style_stat(conn, pair: str, direction: str, style: str) -> dict | None:
    style_key = str(style or "").upper()
    if style_key not in {"MARKET", "STOP-ENTRY"}:
        return None

    today = _today_session_date()
    pair_row = fetchone_dict(
        conn,
        """
        SELECT COUNT(*) AS cnt,
               AVG(pl) AS ev,
               SUM(pl) AS total_pl,
               SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
        FROM pretrade_outcomes
        WHERE pl IS NOT NULL
          AND pair = ?
          AND direction = ?
          AND COALESCE(execution_style, '') = ?
          AND session_date = ?
        """,
        (pair, direction, style_key, today),
    )
    pair_captured = fetchone_val(
        conn,
        """
        SELECT COALESCE(SUM(captured), 0)
        FROM seat_outcomes
        WHERE source = 's_hunt'
          AND pair = ?
          AND direction = ?
          AND session_date = ?
        """,
        (pair, direction, today),
    )
    pair_count = int((pair_row or {}).get("cnt") or 0)
    pair_total = float((pair_row or {}).get("total_pl") or 0.0)
    if pair_count >= 2 and pair_total < 0 and int(pair_captured or 0) == 0:
        wins = int((pair_row or {}).get("wins") or 0)
        return {
            "scope": "exact lane",
            "count": pair_count,
            "ev": float((pair_row or {}).get("ev") or 0.0),
            "total_pl": pair_total,
            "wins": wins,
            "win_rate": (wins / pair_count) if pair_count else 0.0,
            "captured": int(pair_captured or 0),
            "label": f"{today} UTC session",
        }

    global_row = fetchone_dict(
        conn,
        """
        SELECT COUNT(*) AS cnt,
               AVG(pl) AS ev,
               SUM(pl) AS total_pl,
               SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
        FROM pretrade_outcomes
        WHERE pl IS NOT NULL
          AND COALESCE(execution_style, '') = ?
          AND session_date = ?
        """,
        (style_key, today),
    )
    global_captured = fetchone_val(
        conn,
        """
        SELECT COALESCE(SUM(captured), 0)
        FROM seat_outcomes
        WHERE source = 's_hunt'
          AND session_date = ?
        """,
        (today,),
    )
    global_count = int((global_row or {}).get("cnt") or 0)
    global_total = float((global_row or {}).get("total_pl") or 0.0)
    if global_count >= 2 and global_total < 0 and int(global_captured or 0) == 0:
        wins = int((global_row or {}).get("wins") or 0)
        return {
            "scope": "global lane",
            "count": global_count,
            "ev": float((global_row or {}).get("ev") or 0.0),
            "total_pl": global_total,
            "wins": wins,
            "win_rate": (wins / global_count) if global_count else 0.0,
            "captured": int(global_captured or 0),
            "label": f"{today} UTC session",
        }
    return None


def _format_same_day_style_guard(style: str, stat: dict) -> str:
    return (
        f"{style} {stat.get('scope', 'today')} {stat.get('label', 'today')} "
        f"(n={int(stat.get('count', 0) or 0)}, "
        f"EV {float(stat.get('ev', 0.0) or 0.0):+.0f}, "
        f"total {float(stat.get('total_pl', 0.0) or 0.0):+.0f}, "
        f"captured S {int(stat.get('captured', 0) or 0)})"
    )


def regime_stats(conn, pair: str, direction: str, regime: str) -> dict:
    """Win rate under a specific regime"""
    recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
    recent_pretrade = fetchall_dict(
        conn,
        """SELECT pl
           FROM pretrade_outcomes
           WHERE pair = ?
             AND direction = ?
             AND COALESCE(regime_snapshot, '') = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, regime, recent_pretrade_cutoff),
    )
    if len(recent_pretrade) >= RECENT_FEEDBACK_MIN_COUNT:
        return _stat_payload(
            recent_pretrade,
            source="pretrade_recent",
            lookback_days=RECENT_PRETRADE_LOOKBACK_DAYS,
        )

    recent_trade_cutoff = live_history_start(RECENT_TRADE_LOOKBACK_DAYS)
    recent_trades = fetchall_dict(
        conn,
        """SELECT pl
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND regime = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, regime, recent_trade_cutoff),
    )
    if recent_trades:
        return _stat_payload(
            recent_trades,
            source="trades_recent",
            lookback_days=RECENT_TRADE_LOOKBACK_DAYS,
        )

    trades = fetchall_dict(
        conn,
        """SELECT pl
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND regime = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, regime, live_history_start(None)),
    )
    return _stat_payload(
        trades,
        source="trades_all",
        lookback_days=None,
    )


def session_stats(conn, pair: str, direction: str, session_bucket: str) -> dict:
    """Win rate by UTC session bucket for the same pair/direction."""
    recent_pretrade_cutoff = live_history_start(RECENT_PRETRADE_LOOKBACK_DAYS)
    recent_pretrade = fetchall_dict(
        conn,
        """SELECT pl
           FROM pretrade_outcomes
           WHERE pair = ?
             AND direction = ?
             AND COALESCE(session_bucket, '') = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, session_bucket, recent_pretrade_cutoff),
    )
    if len(recent_pretrade) >= RECENT_FEEDBACK_MIN_COUNT:
        return _stat_payload(
            recent_pretrade,
            source="pretrade_recent",
            lookback_days=RECENT_PRETRADE_LOOKBACK_DAYS,
        )

    recent_trade_cutoff = live_history_start(RECENT_TRADE_LOOKBACK_DAYS)
    recent_trades = fetchall_dict(
        conn,
        """SELECT pl
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND session_hour IS NOT NULL
             AND CASE
                   WHEN session_hour BETWEEN 0 AND 6 THEN 'tokyo'
                   WHEN session_hour BETWEEN 7 AND 14 THEN 'london'
                   WHEN session_hour BETWEEN 15 AND 21 THEN 'newyork'
                   ELSE 'late'
                 END = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, session_bucket, recent_trade_cutoff),
    )
    if recent_trades:
        return _stat_payload(
            recent_trades,
            source="trades_recent",
            lookback_days=RECENT_TRADE_LOOKBACK_DAYS,
        )

    trades = fetchall_dict(
        conn,
        """SELECT pl
           FROM trades
           WHERE pair = ?
             AND direction = ?
             AND session_hour IS NOT NULL
             AND CASE
                   WHEN session_hour BETWEEN 0 AND 6 THEN 'tokyo'
                   WHEN session_hour BETWEEN 7 AND 14 THEN 'london'
                   WHEN session_hour BETWEEN 15 AND 21 THEN 'newyork'
                   ELSE 'late'
                 END = ?
             AND pl IS NOT NULL
             AND session_date >= ?""",
        (pair, direction, session_bucket, live_history_start(None)),
    )
    return _stat_payload(
        trades,
        source="trades_all",
        lookback_days=None,
    )


def headline_risk(conn, pair: str) -> list[dict]:
    """Past market events related to this pair"""
    return fetchall_dict(conn,
        """SELECT event_type, headline, spike_pips, spike_direction, impact, session_date
           FROM market_events
           WHERE pairs_affected = ? OR pairs_affected IS NULL
           ORDER BY spike_pips DESC""",
        (pair,))


def active_headlines_history(conn, headline_keyword: str) -> list[dict]:
    """Historical trade results when a specific headline was active"""
    return fetchall_dict(conn,
        """SELECT pair, direction, pl, regime, lesson
           FROM trades
           WHERE active_headlines LIKE ?
           AND pl IS NOT NULL""",
        (f"%{headline_keyword}%",))


# --- User Call Layer ---

def user_call_stats(conn, pair: str = None, direction: str = None) -> dict:
    """Accuracy rate of user's market reads"""
    where = ["outcome IS NOT NULL"]
    params = []
    if pair:
        where.append("pair = ?")
        params.append(pair)
    if direction:
        where.append("direction = ?")
        params.append(direction)

    calls = fetchall_dict(conn,
        f"SELECT outcome, pl_after_30m, conditions, call_text FROM user_calls WHERE {' AND '.join(where)}",
        tuple(params))

    if not calls:
        return {"count": 0}

    correct = [c for c in calls if c["outcome"] == "correct"]
    incorrect = [c for c in calls if c["outcome"] == "incorrect"]

    return {
        "count": len(calls),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": len(correct) / len(calls) if calls else 0,
        "recent_calls": [c["call_text"] for c in calls[-3:]],
    }


def latest_user_call(conn, pair: str = None, max_age_days: int = 3) -> dict | None:
    """Most recent user call within max_age_days. Market conditions change — stale calls are noise."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    if pair:
        return fetchone_dict(conn,
            "SELECT * FROM user_calls WHERE pair = ? AND session_date >= ? ORDER BY id DESC LIMIT 1",
            (pair, cutoff))
    return fetchone_dict(conn,
        "SELECT * FROM user_calls WHERE session_date >= ? ORDER BY id DESC LIMIT 1",
        (cutoff,))


# --- Vector Layer ---

def similar_trades_narrative(query: str, pair: str | None = None, direction: str | None = None, top_k: int = 3) -> list[dict]:
    """Retrieve narratives of similar situations via vector search"""
    try:
        from recall import hybrid_search
        hits = hybrid_search(query, top_k=max(top_k * 4, 6), pair=pair, direction=direction)
        preferred = []
        fallback = []
        for hit in hits:
            if hit.get("source_file") == "state.md" or hit.get("chunk_type") not in {"trade", "lesson"}:
                fallback.append(hit)
            else:
                preferred.append(hit)
        chosen = preferred[:top_k]
        if len(chosen) < top_k:
            chosen.extend(fallback[:top_k - len(chosen)])
        return chosen
    except Exception:
        return []


# --- Setup Quality Assessment (forward-looking: quality of current setup) ---

def _load_technicals(pair: str) -> dict:
    """Read technical data from logs/technicals_{PAIR}.json"""
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    try:
        return load_technicals_timeframes(f)
    except Exception:
        return {}


def _calc_currency_strength() -> dict[str, float]:
    """Calculate per-currency scores from H1 ADX x DI direction"""
    scores: dict[str, list[float]] = {c: [] for c in ["USD", "EUR", "GBP", "AUD", "JPY"]}
    for pair in PAIRS:
        tfs = _load_technicals(pair)
        h1 = tfs.get("H1", {})
        if not h1:
            continue
        adx = h1.get("adx", 0)
        di_plus = h1.get("plus_di", 0)
        di_minus = h1.get("minus_di", 0)
        base, quote = PAIR_CURRENCIES[pair]
        direction = (di_plus - di_minus) / max(di_plus + di_minus, 1)
        weight = min(adx / 30, 1.5)
        signal = direction * weight
        scores[base].append(signal)
        scores[quote].append(-signal)
    return {ccy: sum(vals) / len(vals) if vals else 0 for ccy, vals in scores.items()}


def _get_current_spread(pair: str) -> float | None:
    """Fetch current spread in pips from OANDA pricing API"""
    try:
        cfg = {}
        for line in open(ROOT / "config" / "env.toml"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/pricing?instruments={pair}"
        import urllib.request
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        p = data["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        pip_factor = 100 if "JPY" in pair else 10000
        return (ask - bid) * pip_factor
    except Exception:
        return None


def _same_direction_inventory_summary(pair: str, direction: str) -> dict | None:
    """Summarize current same-pair same-direction live inventory from OANDA."""
    try:
        cfg = {}
        for line in open(ROOT / "config" / "env.toml"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/openTrades"
        import urllib.request

        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        same_direction = []
        for trade in data.get("trades", []):
            if trade.get("instrument") != pair:
                continue
            units = float(trade.get("currentUnits", 0) or 0)
            if direction == "LONG" and units <= 0:
                continue
            if direction == "SHORT" and units >= 0:
                continue
            same_direction.append(trade)

        if not same_direction:
            return None

        def trade_has_risk_reduced_stop(trade: dict) -> bool:
            entry_price = float(trade.get("price", 0) or 0)
            if entry_price <= 0:
                return False
            stop_order = trade.get("stopLossOrder") or {}
            stop_price = stop_order.get("price")
            if stop_price in {None, ""}:
                return False
            try:
                stop = float(stop_price)
            except (TypeError, ValueError):
                return False
            if direction == "LONG":
                return stop >= entry_price
            return stop <= entry_price

        return {
            "count": len(same_direction),
            "units": sum(abs(int(float(trade.get("currentUnits", 0) or 0))) for trade in same_direction),
            "upl": sum(float(trade.get("unrealizedPL", 0) or 0) for trade in same_direction),
            "has_protection": any(
                trade.get("stopLossOrder") or trade.get("trailingStopLossOrder") or trade.get("takeProfitOrder")
                for trade in same_direction
            ),
            "has_stop_protection": any(
                trade.get("stopLossOrder") or trade.get("trailingStopLossOrder")
                for trade in same_direction
            ),
            "risk_reduced": any(trade_has_risk_reduced_stop(trade) for trade in same_direction),
            "trade_ids": [str(trade.get("id", "?")) for trade in same_direction],
        }
    except Exception:
        return None


GRADE_ORDER = ["C", "B", "A", "S"]
GRADE_TO_UNITS = {
    "S": "8000-10000u (full pressure. strongest deployment)",
    "A": "5000-8000u (core size. real edge)",
    "B": "2000-3000u (scout / limited deployment)",
    "C": "1000u or less (weak basis — data suggests caution, you decide)",
}
ALLOCATION_BAND_TO_UNITS = {
    "S": GRADE_TO_UNITS["S"],
    "A": GRADE_TO_UNITS["A"],
    "B+": "3000-4500u (full-B pressure. A/S path, but price/trigger still decides timing)",
    "B0": GRADE_TO_UNITS["B"],
    "B-": "1000-2000u (thin probe only. often better as PASS)",
    "C": GRADE_TO_UNITS["C"],
}


def _score_to_grade(score: int, thresholds: tuple[tuple[int, str], ...]) -> str:
    for cutoff, grade in thresholds:
        if score >= cutoff:
            return grade
    return thresholds[-1][1]


def _cap_grade(grade: str, cap: str) -> str:
    return GRADE_ORDER[min(GRADE_ORDER.index(grade), GRADE_ORDER.index(cap))]


def _grade_icon(grade: str) -> str:
    return {"S": "🔥", "A": "✅", "B": "⚠️", "C": "❌"}.get(grade, "?")


def _sizing_text_for_allocation(allocation_grade: str, allocation_band: str | None = None) -> str:
    band = str(allocation_band or allocation_grade or "C")
    return ALLOCATION_BAND_TO_UNITS.get(band, GRADE_TO_UNITS.get(allocation_grade, GRADE_TO_UNITS["C"]))


def _opposite_direction(direction: str) -> str:
    return "SHORT" if str(direction).upper() == "LONG" else "LONG"


def _clone_setup_for_contest(setup: dict) -> dict:
    clone = dict(setup or {})
    clone["details"] = list((setup or {}).get("details") or [])
    return clone


def _setup_live_score(setup: dict, learning_profile: dict) -> float:
    quality = float(
        setup.get("quality_score", setup.get("score", 0)) or 0
    )
    learning = float(learning_profile.get("learning_score", 0) or 0) / 10.0
    edge = float(GRADE_RANK.get(setup.get("edge_grade", setup.get("grade", "C")), 0)) * 2.0
    alloc = float(GRADE_RANK.get(setup.get("allocation_grade", setup.get("edge_grade", "C")), 0))
    return quality + learning + edge + alloc


def _same_pair_direction_contest(
    conn,
    pair: str,
    direction: str,
    *,
    regime: str | None,
    wave: str,
    counter: bool,
    risk_level: str,
    current_setup: dict,
    current_learning: dict,
    current_execution: dict,
    live_tape: dict | None = None,
) -> dict | None:
    current_regime = str(current_learning.get("current_regime") or regime or "").lower() or None
    current_style = str(current_execution.get("style", "PASS"))
    if current_regime in {"range", "quiet"} and current_style == "LIMIT":
        return None

    opposite = _opposite_direction(direction)
    opposite_setup = (
        assess_counter_trade(pair, opposite)
        if counter else assess_setup_quality(pair, opposite, wave=wave)
    )
    opposite_setup["is_counter"] = bool(counter)
    opposite_learning = _build_learning_profile(conn, pair, opposite, regime_hint=regime)
    opposite_setup = _cap_setup_allocation(opposite_setup, opposite_learning)
    opposite_execution = _recommend_execution_style(pair, opposite, opposite_setup, opposite_learning, risk_level)
    opposite_execution = _apply_live_tape_execution_guard(
        pair,
        opposite,
        opposite_learning,
        opposite_execution,
        live_tape,
        risk_level=risk_level,
    )

    current_rank = EXECUTION_STYLE_RANK.get(current_style, 0)
    opposite_style = str(opposite_execution.get("style", "PASS"))
    opposite_rank = EXECUTION_STYLE_RANK.get(opposite_style, 0)
    current_score = _setup_live_score(current_setup, current_learning)
    opposite_score = _setup_live_score(opposite_setup, opposite_learning)
    current_alloc_rank = GRADE_RANK.get(current_setup.get("allocation_grade", current_setup.get("edge_grade", "C")), 0)
    opposite_alloc_rank = GRADE_RANK.get(opposite_setup.get("allocation_grade", opposite_setup.get("edge_grade", "C")), 0)

    if current_regime in {"range", "quiet"} and current_rank == opposite_rank:
        return None

    cleaner = (
        opposite_rank >= max(current_rank + 1, 2)
        and opposite_score >= current_score + 3.0
        and opposite_alloc_rank >= current_alloc_rank + 1
    )
    if not cleaner:
        return None

    return {
        "current_direction": direction,
        "current_style": current_style,
        "current_score": current_score,
        "opposite_direction": opposite,
        "opposite_style": opposite_style,
        "opposite_score": opposite_score,
        "opposite_edge_grade": opposite_setup.get("edge_grade", opposite_setup.get("grade", "?")),
        "opposite_allocation_grade": opposite_setup.get("allocation_grade", "?"),
        "opposite_learning_verdict": opposite_learning.get("verdict"),
        "opposite_learning_score": int(opposite_learning.get("learning_score", 0) or 0),
        "reason": (
            f"{pair} {opposite} is cleaner live now "
            f"({opposite_style}, edge {opposite_setup.get('edge_grade', opposite_setup.get('grade', '?'))}, "
            f"learning {int(opposite_learning.get('learning_score', 0) or 0)}/99) "
            f"than {direction} ({current_style}, edge {current_setup.get('edge_grade', current_setup.get('grade', '?'))}, "
            f"learning {int(current_learning.get('learning_score', 0) or 0)}/99)"
        ),
    }


def _apply_same_pair_direction_contest(
    pair: str,
    warnings: list[str],
    setup: dict,
    execution_plan: dict,
    contest: dict | None,
) -> tuple[dict, dict]:
    if not contest:
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})

    contest_reason = contest["reason"]
    warnings.append(f"⚠️ SAME-PAIR CONTEST: {contest_reason}")
    setup["details"].append(f"Same-pair contest: {contest_reason}")

    current_style = str(execution_plan.get("style", "PASS"))
    current_alloc_rank = GRADE_RANK.get(setup.get("allocation_grade", setup.get("edge_grade", "C")), 0)
    if current_style == "MARKET":
        execution_plan["style"] = "PASS"
        execution_plan["note"] = (
            f"same-pair cleaner opposite side: {pair} {contest['opposite_direction']} is the live payer now; "
            "do not pay market into the weaker side"
        )
    elif current_style in {"LIMIT", "STOP-ENTRY"} and current_alloc_rank <= 1:
        execution_plan["style"] = "PASS"
        execution_plan["note"] = (
            f"same-pair cleaner opposite side: {pair} {contest['opposite_direction']} is the better live lane; "
            "leave this side flat until the contradiction clears"
        )

    execution_plan["same_pair_contest"] = contest_reason
    return setup, execution_plan


def _apply_same_direction_inventory_guard(
    pair: str,
    direction: str,
    warnings: list[str],
    setup: dict,
    execution_plan: dict,
    inventory: dict | None,
    learning_profile: dict | None = None,
    promotion_pressure: dict | None = None,
) -> tuple[dict, dict]:
    if not inventory or int(inventory.get("count", 0) or 0) <= 0:
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})
    trade_ids = ",".join(inventory.get("trade_ids") or [])
    summary = (
        f"{pair} {direction} already open: {int(inventory.get('count', 0))} leg(s), "
        f"{int(inventory.get('units', 0))}u, UPL {float(inventory.get('upl', 0) or 0):+,.0f} JPY"
    )
    if trade_ids:
        summary += f" (id={trade_ids})"

    execution_plan["same_direction_inventory"] = summary
    if float(inventory.get("upl", 0) or 0) <= 0:
        current_style = str(execution_plan.get("style") or "PASS").upper()
        if not bool(inventory.get("has_protection")):
            warnings.append(
                f"⚠️ SAME-PAIR RELOAD: {summary}. Existing leg is not paid yet → no second fill."
            )
            setup["details"].append(
                "Same-pair reload blocked: existing leg is still unpaid and unprotected, so a second fill would just stack friction"
            )
            execution_plan["style"] = "PASS"
            execution_plan["note"] = (
                "same-pair reload blocked until the existing leg is paid or risk is reduced; "
                "do not average into unpaid inventory"
            )
            _append_hard_execution_blocker(execution_plan, execution_plan.get("note"))
        else:
            learning_profile = dict(learning_profile or {})
            cap_rank = _cap_rank(str(learning_profile.get("allocation_cap") or CAP_LABELS["b_scout"]))
            learning_score = int(learning_profile.get("learning_score", 0) or 0)
            trade_count = int(learning_profile.get("trade_count", 0) or 0)
            trade_ev = float(learning_profile.get("trade_ev", 0.0) or 0.0)
            pressure_rank = int((promotion_pressure or {}).get("rank", 0) or 0)
            pressure_note = str((promotion_pressure or {}).get("note") or "").strip()
            regime = str(learning_profile.get("current_regime") or "").lower()
            exact_ok = bool(learning_profile.get("has_exact"))
            tailwind = str(learning_profile.get("context_bias") or "neutral") == "tailwind"
            risk_reduced = bool(inventory.get("risk_reduced"))
            allow_managed_market_reload = (
                current_style == "MARKET"
                and risk_reduced
                and int(inventory.get("count", 0) or 0) == 1
                and regime in {"trending", "transition", "squeeze"}
                and cap_rank >= 4
                and learning_score >= 78
                and (pressure_rank >= 18 or (exact_ok and trade_count >= 5 and trade_ev > 0 and tailwind))
            )
            if allow_managed_market_reload:
                reload_note = (
                    "same-pair managed reload is allowed because the existing leg is already risk-reduced "
                    "and this seat still has A/S-quality live pressure"
                )
                if pressure_note:
                    reload_note = f"{reload_note}; {pressure_note}"
                warnings.append(f"ℹ️ SAME-PAIR MANAGED RELOAD: {summary}. {reload_note}.")
                setup["details"].append(
                    "Same-pair managed reload allowed: the live leg is already risk-reduced, so a small add is cleaner than another trigger-only rewrite"
                )
                execution_plan["note"] = reload_note
                execution_plan["same_direction_reload_allowed"] = reload_note
                return setup, execution_plan

            warnings.append(
                f"⚠️ SAME-PAIR RELOAD: {summary}. Existing leg is still unpaid, but protection is already on "
                "→ only thinner trigger/passive reloads are allowed."
            )
            setup["details"].append(
                "Same-pair protected reload: the live leg is still unpaid, so keep any add-on trigger/passive only and demand a cleaner price/trigger than the current seat"
            )
            if current_style == "MARKET":
                execution_plan["style"] = "STOP-ENTRY"
            execution_plan["note"] = (
                "same-pair reload stays trigger/passive only while the existing leg is unpaid but protected; "
                "require a cleaner price/trigger than the current inventory"
            )
    else:
        warnings.append(
            f"ℹ️ SAME-PAIR RELOAD: {summary}. Only add if this is clearly a better price/trigger than the paid leg."
        )
    return setup, execution_plan


def _apply_negative_expectancy_guard(
    pair: str,
    direction: str,
    warnings: list[str],
    setup: dict,
    learning_profile: dict,
    execution_plan: dict,
    stats: dict | None,
    promotion_pressure: dict | None = None,
) -> tuple[dict, dict]:
    if not stats or int(stats.get("count", 0) or 0) < 4:
        return setup, execution_plan

    if str((execution_plan or {}).get("style") or "PASS").upper() == "PASS":
        return setup, execution_plan

    count = int(stats.get("count", 0) or 0)
    ev = float(stats.get("expectancy", stats.get("avg_pl", 0.0)) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    break_even = stats.get("break_even_win_rate")
    cap = str((learning_profile or {}).get("allocation_cap") or "")
    confirmed_exact = (
        bool((learning_profile or {}).get("has_exact"))
        and (learning_profile or {}).get("state") == "confirmed"
        and cap in {CAP_LABELS["a_max"], CAP_LABELS["as_confirmed"]}
    )
    below_break_even = (
        break_even is not None and win_rate + 0.03 < float(break_even)
    )
    severe_headwind = ev < 0 and (
        below_break_even
        or win_rate <= 0.25
        or (count >= 6 and ev <= -75)
        or ev <= -100
    )
    if not severe_headwind or confirmed_exact:
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})
    summary = _format_feedback_stat(stats, f"{pair} {direction}")
    pressure_rank = int((promotion_pressure or {}).get("rank", 0) or 0)
    pressure_note = str((promotion_pressure or {}).get("note") or "").strip()
    current_style = str(execution_plan.get("style") or "PASS").upper()
    if pressure_rank >= 12:
        fallback_style = current_style
        if current_style == "MARKET":
            fallback_style = _tape_fallback_style(str(learning_profile.get("current_regime") or "").lower())
        guard_reason = (
            f"{pair} {direction} is still negative expectancy ({summary}); keep it only as a thin {fallback_style} scout "
            "until the base rate repairs"
        )
        warnings.append(f"⚠️ NEGATIVE EV HEADWIND: {guard_reason}")
        setup["details"].append(f"Negative expectancy headwind: {guard_reason}")
        execution_plan["style"] = fallback_style
        execution_plan["negative_expectancy_guard"] = guard_reason
        execution_plan["note"] = f"{guard_reason}; {pressure_note}".rstrip("; ")
        return setup, execution_plan

    guard_reason = (
        f"{pair} {direction} is still negative expectancy ({summary}); "
        "stop arming this lane until the base rate repairs"
    )
    warnings.append(f"⚠️ NEGATIVE EV GUARD: {guard_reason}")
    setup["details"].append(f"Negative expectancy guard: {guard_reason}")
    execution_plan["style"] = "PASS"
    execution_plan["note"] = _merge_note(execution_plan.get("note"), guard_reason)
    return setup, execution_plan


def _apply_planned_stop_floor_guard(
    pair: str,
    warnings: list[str],
    setup: dict,
    execution_plan: dict,
) -> tuple[dict, dict]:
    stop_multiple = setup.get("stop_spread_multiple")
    planned_stop = setup.get("planned_stop_pips")
    spread_pips = setup.get("spread_pips")
    if stop_multiple is None or planned_stop is None or spread_pips is None:
        return setup, execution_plan

    regret_floor = _recent_noise_stop_floor(pair, float(spread_pips))
    static_floor_pips = float(spread_pips) * 4.0
    required_stop_pips = max(
        static_floor_pips,
        float((regret_floor or {}).get("required_stop_pips", 0.0) or 0.0),
    )
    if float(planned_stop) >= required_stop_pips:
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})
    if regret_floor and float(planned_stop) < float(regret_floor.get("required_stop_pips", 0.0) or 0.0):
        floor_reason = (
            f"{_noise_stop_floor_reason(pair, float(planned_stop), regret_floor)}; "
            "widen the stop or improve the entry first"
        )
    else:
        floor_reason = (
            f"{pair} planned SL {float(planned_stop):.1f}pip is only {float(stop_multiple):.1f}x "
            f"spread ({float(spread_pips):.1f}pip); widen the stop or improve the entry first"
        )
    warnings.append(f"⚠️ STOP FLOOR: {floor_reason}. No live fill.")
    setup["details"].append(f"Stop floor guard: {floor_reason}")
    execution_plan["style"] = "PASS"
    execution_plan["note"] = floor_reason
    execution_plan["requires_stop_widening"] = True
    _append_hard_execution_blocker(execution_plan, floor_reason)
    setup["noise_stop_floor_pips"] = required_stop_pips
    setup["noise_stop_floor_reason"] = floor_reason
    return setup, execution_plan


def _apply_planned_target_floor_guard(
    pair: str,
    warnings: list[str],
    setup: dict,
    execution_plan: dict,
) -> tuple[dict, dict]:
    target_multiple = setup.get("target_spread_multiple")
    planned_target = setup.get("planned_target_pips")
    spread_pips = setup.get("spread_pips")
    if target_multiple is None or planned_target is None or spread_pips is None:
        return setup, execution_plan
    if float(target_multiple) > 4.0:
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})
    floor_reason = (
        f"{pair} planned TP {float(planned_target):.1f}pip is only {float(target_multiple):.1f}x "
        f"spread ({float(spread_pips):.1f}pip); the move is too thin to pay live friction"
    )
    warnings.append(f"⚠️ TARGET FLOOR: {floor_reason}. No live fill.")
    setup["details"].append(f"Target floor guard: {floor_reason}")
    execution_plan["style"] = "PASS"
    execution_plan["note"] = floor_reason
    execution_plan["requires_better_payoff"] = True
    _append_hard_execution_blocker(execution_plan, floor_reason)
    return setup, execution_plan


def _apply_counter_reversal_guard(
    pair: str,
    warnings: list[str],
    setup: dict,
    execution_plan: dict,
) -> tuple[dict, dict]:
    if not setup.get("is_counter"):
        return setup, execution_plan

    setup = _clone_setup_for_contest(setup)
    execution_plan = dict(execution_plan or {})
    allocation_grade = str(setup.get("allocation_grade", setup.get("edge_grade", "C")) or "C")
    if GRADE_RANK.get(allocation_grade, 0) > GRADE_RANK["B"]:
        warnings.append(
            f"⚠️ COUNTER SIZE GUARD: {pair} counter/reversal stays B-max until it pays. "
            f"Allocation {allocation_grade} → B."
        )
        setup["details"].append(
            "Counter-size guard: counter/reversal can keep edge quality, but deployment stays B-max until the reversal actually pays."
        )
        setup["allocation_grade"] = "B"

    if str(execution_plan.get("style", "PASS")) == "MARKET":
        execution_plan["style"] = "STOP-ENTRY"
        execution_plan["note"] = (
            "counter/reversal guard: do not pay market into the reversal; require the trigger to print first"
        )
        setup["details"].append(
            "Counter execution guard: MARKET demoted to STOP-ENTRY until the reversal proves itself."
        )

    return setup, execution_plan


def assess_setup_quality(
    pair: str,
    direction: str,
    wave: str = "auto",
    *,
    planned_target_pips: float | None = None,
    planned_stop_pips: float | None = None,
    spread_pips: float | None = None,
) -> dict:
    """
    Score current setup quality 0-12 → output S/A/B/C conviction grade

    wave: "big" (H4/H1 swing), "mid" (M5 trade), "small" (M1 scalp), "auto" (auto-detect)

    Evaluation axes (weights vary by wave size):
    1. MTF direction alignment (0-4 pts): TF agreement based on wave size
    2. ADX trend strength (0-2 pts): ADX strength of the reference TF
    3. Macro currency strength alignment (0-2 pts): does currency pair strength direction agree?
    4. Technical confluence (0-2 pts): divergence, StochRSI extremes, BB position
    5. Wave position penalty (-2 to +1 pts): penalty for same-direction entry at H4 extremes
    6. Spread penalty (-2 to 0 pts): penalty if spread is too large relative to target
    """
    tfs = _load_technicals(pair)
    h4 = tfs.get("H4", {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})

    quality_score = 0
    details = []
    is_long = direction == "LONG"

    # --- TF alignment helper ---
    def tf_aligned(tf_data: dict) -> bool:
        if not tf_data:
            return False
        di_plus = tf_data.get("plus_di", 0)
        di_minus = tf_data.get("minus_di", 0)
        return (di_plus > di_minus) if is_long else (di_minus > di_plus)

    h4_aligned = tf_aligned(h4)
    h1_aligned = tf_aligned(h1)
    m5_aligned = tf_aligned(m5)

    # --- Auto-detect wave size ---
    if wave == "auto":
        # H4+H1 aligned → big wave (swing)
        # H4+M5 aligned (H1 transitioning) → mid wave (H1 hasn't flipped yet but higher TF supports)
        # H1+M5 aligned but H4 opposite → mid wave (early trend reversal)
        # M5 only → small wave (rotation / scalp)
        if h4_aligned and h1_aligned:
            wave = "big"
        elif h4_aligned and m5_aligned:
            wave = "mid"  # H4 supports direction, H1 still transitioning
        elif h1_aligned and m5_aligned:
            wave = "mid"
        elif m5_aligned:
            wave = "small"
        else:
            wave = "big"  # default: evaluate against swing standard

    # --- 1. MTF Direction Agreement (0-4) --- evaluation criteria vary by wave size
    aligned_count = sum([h4_aligned, h1_aligned, m5_aligned])

    if wave == "big":
        # Big wave: H4+H1 are important. M5 is timing, not setup quality
        if h4_aligned and h1_aligned and m5_aligned:
            quality_score += 4
            details.append(f"[big wave] MTF full alignment (H4+H1+M5) +4")
        elif h4_aligned and h1_aligned:
            quality_score += 3
            details.append(f"[big wave] H4+H1 aligned +3 (M5 awaiting timing)")
        elif h1_aligned and m5_aligned:
            quality_score += 3
            details.append(f"[big wave] H1+M5 aligned +3 (may be early H4 reversal)")
        elif h1_aligned:
            quality_score += 1
            details.append(f"[big wave] H1 only aligned +1")
        else:
            details.append(f"[big wave] no upper TF alignment +0")

    elif wave == "mid":
        # Mid wave: H1+M5 or H4+M5 are important
        if h1_aligned and m5_aligned:
            quality_score += 4 if h4_aligned else 3
            h4_note = " + H4 aligned" if h4_aligned else ""
            details.append(f"[mid wave] H1+M5 aligned{h4_note} +{4 if h4_aligned else 3}")
        elif m5_aligned:
            pts = 3 if h4_aligned else 2
            h4_note = " + H4 supports" if h4_aligned else " (H1 opposite)"
            quality_score += pts
            details.append(f"[mid wave] M5 aligned{h4_note} +{pts}")
        else:
            quality_score += 1 if h1_aligned else 0
            details.append(f"[mid wave] M5 not aligned +{1 if h1_aligned else 0}")

    elif wave == "small":
        # Small wave: M5 is important. H1 is directional reference. H4 irrelevant
        if m5_aligned:
            quality_score += 3 if h1_aligned else 2
            h1_note = " + H1 aligned" if h1_aligned else " (H1 opposing - caution)"
            details.append(f"[small wave] M5 aligned{h1_note} +{3 if h1_aligned else 2}")
        else:
            details.append(f"[small wave] M5 not aligned +0 ← don't enter")

    # --- 2. ADX Trend Strength (0-2) --- ADX of the reference TF
    if wave == "big":
        ref_adx = h1.get("adx", 0) if h1 else 0
        ref_aligned = h1_aligned
        ref_label = "H1"
    elif wave == "mid":
        ref_adx = m5.get("adx", 0) if m5 else 0
        ref_aligned = m5_aligned
        ref_label = "M5"
    else:  # small
        ref_adx = m5.get("adx", 0) if m5 else 0
        ref_aligned = m5_aligned
        ref_label = "M5"

    if ref_aligned and ref_adx > 30:
        quality_score += 2
        details.append(f"{ref_label} ADX={ref_adx:.0f}(>30) strong +2")
    elif ref_aligned and ref_adx > 25:
        quality_score += 1
        details.append(f"{ref_label} ADX={ref_adx:.0f}(>25) +1")
    else:
        details.append(f"{ref_label} ADX={ref_adx:.0f} weak +0")

    # --- 3. Macro Currency Strength Alignment (0-2) ---
    try:
        ccy_strength = _calc_currency_strength()
        base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        base_str = ccy_strength.get(base, 0)
        quote_str = ccy_strength.get(quote, 0)

        if is_long:
            # LONG = base strong + quote weak is ideal
            macro_aligned = base_str > 0.1 and quote_str < -0.1
            macro_neutral = base_str > quote_str
        else:
            # SHORT = base weak + quote strong is ideal
            macro_aligned = base_str < -0.1 and quote_str > 0.1
            macro_neutral = base_str < quote_str

        if macro_aligned:
            quality_score += 2
            details.append(f"Macro aligned ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +2")
        elif macro_neutral:
            quality_score += 1
            details.append(f"Macro neutral ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +1")
        else:
            details.append(f"Macro opposing ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +0")
    except Exception:
        details.append("Macro calculation unavailable +0")

    # --- 4. Technical Confluence (0-2) ---
    confluence = 0
    confluence_notes = []

    # Divergence in trade direction
    if h1:
        div_rsi = h1.get("div_rsi_score", 0)
        div_macd = h1.get("div_macd_score", 0)
        div_rsi_kind = h1.get("div_rsi_kind", 0)
        # Regular bullish div (kind=1) supports LONG, regular bearish (kind=-1) supports SHORT
        # Hidden bullish div (kind=2) supports LONG, hidden bearish (kind=-2) supports SHORT
        div_supports = False
        if is_long and div_rsi_kind in (1, 2) and div_rsi > 0:
            div_supports = True
        elif not is_long and div_rsi_kind in (-1, -2) and div_rsi > 0:
            div_supports = True
        if div_supports:
            confluence += 1
            confluence_notes.append(f"H1 Div confirmed (score={div_rsi:.1f})")

    # StochRSI extreme in entry direction
    if m5:
        stoch = m5.get("stoch_rsi", 0.5)
        if is_long and stoch < 0.15:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f} (extreme oversold)")
        elif not is_long and stoch > 0.85:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f} (extreme overbought)")

    # BB position
    if m5:
        bb_pos = m5.get("bb", 0.5)
        if is_long and bb_pos < 0.1:
            confluence_notes.append(f"M5 BB lower band ({bb_pos:.2f})")
        elif not is_long and bb_pos > 0.9:
            confluence_notes.append(f"M5 BB upper band ({bb_pos:.2f})")

    conf_points = min(confluence, 2)
    quality_score += conf_points
    if confluence_notes:
        details.append(f"Technical confluence: {', '.join(confluence_notes)} +{conf_points}")
    else:
        details.append("No technical confluence +0")

    # --- 5. Wave Position Penalty/Bonus (-2 to +1) ---
    if h4:
        h4_cci = h4.get("cci", 0)
        h4_rsi = h4.get("rsi", 50)

        h4_extreme_long = h4_rsi > 70 or h4_cci > 200
        h4_extreme_short = h4_rsi < 30 or h4_cci < -200

        if is_long and h4_extreme_long:
            quality_score -= 2
            details.append(f"⚠ H4 overbought LONG (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← don't enter after the move is done")
        elif not is_long and h4_extreme_short:
            quality_score -= 2
            details.append(f"⚠ H4 extreme oversold SHORT (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← don't enter after the move is done")
        elif is_long and h4_extreme_short:
            quality_score += 1
            details.append(f"H4 oversold LONG (CCI={h4_cci:.0f}) +1 ← counter-trend opportunity")
        elif not is_long and h4_extreme_long:
            quality_score += 1
            details.append(f"H4 overbought SHORT (CCI={h4_cci:.0f}) +1 ← counter-trend opportunity")
        else:
            details.append(f"H4 neutral (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) +0")
    else:
        details.append("H4 data unavailable +0")

    # --- 6. Spread Penalty (0 to -2) --- is spread too large relative to target
    spread_pip = float(spread_pips) if spread_pips is not None else _get_current_spread(pair)
    target = None
    spread_ratio = None
    target_spread_multiple = None
    stop_spread_multiple = None
    if spread_pip is not None:
        # Acceptable spread varies by wave size
        # Big wave (targeting 15-30 pip): 2.0 pip spread = 6-13% → acceptable
        # Mid wave (targeting 10-15 pip): 2.0 pip spread = 13-20% → caution
        # Small wave (targeting 5-10 pip): 2.0 pip spread = 20-40% → fatal
        pip_targets = {"big": 20, "mid": 12, "small": 7}
        target = float(planned_target_pips) if planned_target_pips is not None else pip_targets.get(wave, 12)
        spread_ratio = spread_pip / target if target > 0 else None
        target_spread_multiple = target / spread_pip if spread_pip > 0 and target > 0 else None

        if planned_target_pips is not None and target_spread_multiple is not None:
            if target_spread_multiple <= 4.0:
                quality_score -= 2
                details.append(
                    f"⚠️ Planned TP={target:.1f}pip is only {target_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -2 ← payoff too thin. Pass."
                )
            elif target_spread_multiple <= 6.0:
                quality_score -= 1
                details.append(
                    f"⚠️ Planned TP={target:.1f}pip is only {target_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -1 ← fast edge is paying too much friction"
                )
            else:
                details.append(
                    f"Planned TP={target:.1f}pip ({target_spread_multiple:.1f}x spread {spread_pip:.1f}pip) OK"
                )
        elif spread_ratio is not None and spread_ratio > 0.30:  # spread > 30% of target profit → fatal
            quality_score -= 2
            details.append(f"⚠️ Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) -2 ← R:R broken. Pass.")
        elif spread_ratio is not None and spread_ratio > 0.20:  # > 20% → caution
            quality_score -= 1
            details.append(f"⚠️ Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) -1 ← reduce size")
        else:
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) OK")

        if planned_stop_pips is not None and spread_pip > 0:
            stop_spread_multiple = float(planned_stop_pips) / spread_pip
            if stop_spread_multiple <= 3.0:
                quality_score -= 2
                details.append(
                    f"⚠️ Planned SL={float(planned_stop_pips):.1f}pip is only {stop_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -2 ← noise stop, not protection"
                )
            elif stop_spread_multiple <= 4.0:
                quality_score -= 1
                details.append(
                    f"⚠️ Planned SL={float(planned_stop_pips):.1f}pip is only {stop_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -1 ← stop is too close to execution noise"
                )
            else:
                details.append(
                    f"Planned SL={float(planned_stop_pips):.1f}pip ({stop_spread_multiple:.1f}x spread {spread_pip:.1f}pip) OK"
                )
            regret_floor = _recent_noise_stop_floor(pair, spread_pip)
            if regret_floor and float(planned_stop_pips) < float(regret_floor.get("required_stop_pips", 0.0) or 0.0):
                floor_gap = float(regret_floor.get("required_stop_pips", 0.0) or 0.0) - float(planned_stop_pips)
                if floor_gap >= 1.0:
                    quality_score -= 2
                else:
                    quality_score -= 1
                details.append(
                    "⚠️ "
                    + _noise_stop_floor_reason(pair, float(planned_stop_pips), regret_floor)
                    + " ← stop is still inside historical noise"
                )
    else:
        details.append("Spread unavailable +0")

    # --- Grade mapping ---
    quality_score = max(0, quality_score)  # floor at 0
    edge_grade = _score_to_grade(
        quality_score,
        (
            (8, "S"),
            (6, "A"),
            (4, "B"),
            (0, "C"),
        ),
    )
    allocation_grade = edge_grade

    return {
        "grade": edge_grade,  # backward-compatible alias
        "edge_grade": edge_grade,
        "allocation_grade": allocation_grade,
        "quality_score": quality_score,
        "score_max": 10,
        "wave": wave,
        "details": details,
        "sizing": GRADE_TO_UNITS[allocation_grade],
        "counter_note": None,
        "mtf_aligned": aligned_count,
        "spread_pips": spread_pip,
        "spread_ratio": spread_ratio if spread_pip is not None else None,
        "target_pips": target if spread_pip is not None else None,
        "target_spread_multiple": target_spread_multiple,
        "planned_target_pips": planned_target_pips,
        "planned_stop_pips": planned_stop_pips,
        "stop_spread_multiple": stop_spread_multiple,
        "is_counter": False,
    }


def assess_counter_trade(
    pair: str,
    direction: str,
    *,
    planned_target_pips: float | None = None,
    planned_stop_pips: float | None = None,
    spread_pips: float | None = None,
) -> dict:
    """
    Evaluate a COUNTER-TRADE: M5 trade against H4/H1 direction.

    Counter-trades have fundamentally different evaluation axes:
    - H4 extreme is FOR (not against) — the more extreme, the better
    - MTF alignment is irrelevant (counter = by definition against upper TF)
    - M5 reversal signal is the timing trigger
    - Macro opposition is expected (counter = against macro)

    Counter-trades can have HIGH EDGE even when allocation stays smaller.
    This separates:
    - edge_grade: how strong the reversal read is
    - allocation_grade: how much size the book deserves right now
    """
    tfs = _load_technicals(pair)
    h4 = tfs.get("H4", {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})

    score = 0
    details = []
    is_long = direction == "LONG"

    # --- 1. H4 Extreme (0-3) — the core of a counter-trade ---
    if h4:
        h4_stoch = h4.get("stoch_rsi", 0.5)
        h4_cci = h4.get("cci", 0)
        h4_rsi = h4.get("rsi", 50)

        # Counter-LONG needs H4 oversold; Counter-SHORT needs H4 overbought
        if is_long:
            extreme = (h4_stoch <= 0.05) or (h4_cci < -200) or (h4_rsi < 30)
            moderate = (h4_stoch <= 0.15) or (h4_cci < -100) or (h4_rsi < 35)
        else:
            extreme = (h4_stoch >= 0.95) or (h4_cci > 200) or (h4_rsi > 70)
            moderate = (h4_stoch >= 0.85) or (h4_cci > 100) or (h4_rsi > 65)

        if extreme:
            score += 3
            details.append(f"H4 extreme (StRSI={h4_stoch:.2f} CCI={h4_cci:.0f} RSI={h4_rsi:.0f}) +3 ← strong counter zone")
        elif moderate:
            score += 1
            details.append(f"H4 moderate (StRSI={h4_stoch:.2f} CCI={h4_cci:.0f}) +1")
        else:
            details.append(f"H4 not extreme (StRSI={h4_stoch:.2f}) +0 ← no counter-trade basis")
    else:
        details.append("H4 data unavailable +0")

    # --- 2. H1 Divergence or Fatigue (0-2) — confirmation ---
    if h1:
        div_rsi = h1.get("div_rsi_score", 0)
        div_macd = h1.get("div_macd_score", 0)
        h1_cci = h1.get("cci", 0)
        has_div = div_rsi > 0 or div_macd > 0
        has_cci_extreme = (is_long and h1_cci < -200) or (not is_long and h1_cci > 200)

        if has_div and has_cci_extreme:
            score += 2
            details.append(f"H1 divergence + CCI extreme ({h1_cci:.0f}) +2 ← reversal confirmed")
        elif has_div or has_cci_extreme:
            score += 1
            details.append(f"H1 {'divergence' if has_div else f'CCI={h1_cci:.0f}'} +1")
        else:
            details.append(f"H1 no reversal signal +0")

    # --- 3. M5 Reversal Signal (0-2) — timing ---
    if m5:
        m5_stoch = m5.get("stoch_rsi", 0.5)
        m5_macd_h = m5.get("macd_hist", 0)

        # Counter-LONG: M5 was oversold and MACD turning up
        # Counter-SHORT: M5 was overbought and MACD turning down
        if is_long:
            reversal = m5_stoch < 0.2 or m5_macd_h > 0
            timing = m5_stoch < 0.3
        else:
            reversal = m5_stoch > 0.8 or m5_macd_h < 0
            timing = m5_stoch > 0.7

        if reversal and timing:
            score += 2
            details.append(f"M5 reversal confirmed (StRSI={m5_stoch:.2f} MACD_H={m5_macd_h:.5f}) +2")
        elif reversal or timing:
            score += 1
            details.append(f"M5 partial signal (StRSI={m5_stoch:.2f}) +1")
        else:
            details.append(f"M5 no reversal yet (StRSI={m5_stoch:.2f}) +0 ← wait for timing")

    # --- 4. Spread vs counter-trade target (0 to -1) ---
    spread_pip = float(spread_pips) if spread_pips is not None else _get_current_spread(pair)
    counter_target = float(planned_target_pips) if planned_target_pips is not None else 8
    spread_ratio = None
    target_spread_multiple = None
    stop_spread_multiple = None
    if spread_pip is not None:
        spread_ratio = spread_pip / counter_target if counter_target > 0 else None
        target_spread_multiple = counter_target / spread_pip if spread_pip > 0 and counter_target > 0 else None
        if planned_target_pips is not None and target_spread_multiple is not None:
            if target_spread_multiple <= 4.0:
                score -= 2
                details.append(
                    f"⚠️ Planned TP={counter_target:.1f}pip is only {target_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -2 ← counter edge is too thin"
                )
            elif target_spread_multiple <= 6.0:
                score -= 1
                details.append(
                    f"⚠️ Planned TP={counter_target:.1f}pip is only {target_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -1 ← counter edge is paying too much friction"
                )
            else:
                details.append(
                    f"Planned TP={counter_target:.1f}pip ({target_spread_multiple:.1f}x spread {spread_pip:.1f}pip) OK"
                )
        elif spread_ratio is not None and spread_ratio > 0.25:
            score -= 1
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {counter_target}pip) -1 ← too wide for counter")
        else:
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {counter_target}pip) OK")

        if planned_stop_pips is not None and spread_pip > 0:
            stop_spread_multiple = float(planned_stop_pips) / spread_pip
            if stop_spread_multiple <= 3.0:
                score -= 2
                details.append(
                    f"⚠️ Planned SL={float(planned_stop_pips):.1f}pip is only {stop_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -2 ← counter stop is inside noise"
                )
            elif stop_spread_multiple <= 4.0:
                score -= 1
                details.append(
                    f"⚠️ Planned SL={float(planned_stop_pips):.1f}pip is only {stop_spread_multiple:.1f}x spread "
                    f"({spread_pip:.1f}pip) -1 ← counter stop is too tight"
                )
            else:
                details.append(
                    f"Planned SL={float(planned_stop_pips):.1f}pip ({stop_spread_multiple:.1f}x spread {spread_pip:.1f}pip) OK"
                )
            regret_floor = _recent_noise_stop_floor(pair, spread_pip)
            if regret_floor and float(planned_stop_pips) < float(regret_floor.get("required_stop_pips", 0.0) or 0.0):
                floor_gap = float(regret_floor.get("required_stop_pips", 0.0) or 0.0) - float(planned_stop_pips)
                if floor_gap >= 1.0:
                    score -= 2
                else:
                    score -= 1
                details.append(
                    "⚠️ "
                    + _noise_stop_floor_reason(pair, float(planned_stop_pips), regret_floor)
                    + " ← counter stop is still inside historical noise"
                )

    # --- Edge vs allocation (counter-trades can be S/A edge, smaller deployment) ---
    score = max(0, score)
    edge_grade = _score_to_grade(
        score,
        (
            (6, "S"),
            (4, "A"),
            (2, "B"),
            (0, "C"),
        ),
    )
    allocation_grade = _cap_grade(edge_grade, "A")
    counter_note = None
    if edge_grade != allocation_grade:
        counter_note = (
            "Execution edge is stronger than deployment size because this is still a "
            "counter-trade against the upper-TF flow."
        )
    if allocation_grade == "C":
        sizing = "skip — counter conditions not met"
    else:
        sizing = GRADE_TO_UNITS[allocation_grade]

    return {
        "grade": edge_grade,  # backward-compatible alias
        "edge_grade": edge_grade,
        "allocation_grade": allocation_grade,
        "score": score,
        "score_max": 7,
        "details": details,
        "sizing": sizing,
        "counter_note": counter_note,
        "spread_pips": spread_pip,
        "spread_ratio": (spread_pip / counter_target) if spread_pip is not None else None,
        "target_pips": counter_target,
        "target_spread_multiple": target_spread_multiple,
        "planned_target_pips": planned_target_pips,
        "planned_stop_pips": planned_stop_pips,
        "stop_spread_multiple": stop_spread_multiple,
        "is_counter": True,
    }


# --- Risk Assessment (backward-looking: risk from historical data) ---

def _build_learning_profile(
    conn,
    pair: str,
    direction: str,
    *,
    regime_hint: str | None = None,
    live_tape: dict | None = None,
) -> dict:
    registry = _load_lesson_registry()
    exact, pair_only = _target_lessons_for_profile(pair, direction, registry)
    relevant = exact[:3] + pair_only[:2]
    top = exact[0] if exact else (pair_only[0] if pair_only else None)
    now_utc = datetime.now(timezone.utc)
    current_session = _current_session_bucket(now_utc)
    current_regime = regime_hint or _infer_current_pair_regime(pair)
    pair_stat = trade_stats(conn, pair, direction)
    current_tape_bucket = _live_tape_bucket(direction, live_tape)
    tape_stat = tape_stats(conn, pair, direction, current_tape_bucket)
    session_stat = session_stats(conn, pair, direction, current_session)
    regime_stat = regime_stats(conn, pair, direction, current_regime) if current_regime else {"count": 0}
    trade_count = int(pair_stat.get("count", 0) or 0)
    trade_ev = float(pair_stat.get("expectancy", 0.0) or 0.0)
    trade_wr = float(pair_stat.get("win_rate", 0.0) or 0.0)
    feedback_stats = pair_stat
    if int(tape_stat.get("count", 0) or 0) >= RECENT_FEEDBACK_MIN_COUNT:
        feedback_stats = tape_stat
    pair_feedback = _recent_feedback_override(feedback_stats)

    profile = {
        "learning_score": 18,
        "verdict": "limited history",
        "allocation_cap": CAP_LABELS["b_scout"],
        "evidence": "No strong pair-direction lesson yet. Respect the chart more than memory.",
        "state": None,
        "trust_score": 0,
        "trade_count": trade_count,
        "trade_ev": trade_ev,
        "trade_wr": trade_wr,
        "has_exact": False,
        "current_session": current_session,
        "current_regime": current_regime,
        "current_tape_bucket": current_tape_bucket,
        "pair_context": _format_feedback_stat(pair_stat),
        "tape_context": _format_feedback_stat(tape_stat, _live_tape_bucket_label(current_tape_bucket)),
        "recent_feedback_note": pair_feedback.get("reason") if pair_feedback else None,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "tape_stat": tape_stat,
        "context_bias": "neutral",
    }
    if not top and trade_count == 0:
        return profile

    trust = int(top.get("trust_score", 0) or 0) if top else 0
    text_blob = _lesson_text_blob(relevant)
    no_edge = any(pattern in text_blob for pattern in NO_EDGE_PATTERNS)
    positive = any(pattern in text_blob for pattern in POSITIVE_EDGE_PATTERNS)
    session_score_delta, session_cap_delta, session_bias = _context_signal(session_stat)
    regime_score_delta, regime_cap_delta, regime_bias = _context_signal(regime_stat)
    tape_score_delta, tape_cap_delta, tape_bias = _context_signal(
        tape_stat,
        min_count=RECENT_FEEDBACK_MIN_COUNT,
    )

    score = trust
    if exact:
        score += 12
    elif pair_only:
        score += 5
    if top and top.get("state") == "confirmed":
        score += 8
    elif top and top.get("state") == "watch":
        score += 3
    if trade_count >= 5 and trade_ev > 0:
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
    score += session_score_delta + regime_score_delta + tape_score_delta
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

    cap_rank = _cap_rank(allocation_cap)
    cap_rank += session_cap_delta + regime_cap_delta + tape_cap_delta
    if no_edge:
        cap_rank = min(cap_rank, 1)
    if pair_feedback:
        if pair_feedback["tier"] == "hard_headwind":
            cap_rank = min(cap_rank, 0)
        elif pair_feedback["tier"] == "soft_headwind":
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
    if tape_bias == "headwind":
        context_flags.append(f"{_live_tape_bucket_label(current_tape_bucket)} headwind")
    elif tape_bias == "tailwind":
        context_flags.append(f"{_live_tape_bucket_label(current_tape_bucket)} tailwind")
    if context_flags:
        verdict = f"{verdict} | {' + '.join(context_flags)}"

    context_bias = "neutral"
    if "headwind" in context_flags:
        context_bias = "headwind"
    elif "tailwind" in context_flags:
        context_bias = "tailwind"

    profile.update({
        "learning_score": min(score, 24) if pair_feedback and pair_feedback["tier"] == "hard_headwind" else (
            min(score, 48) if pair_feedback and pair_feedback["tier"] == "soft_headwind" else score
        ),
        "verdict": verdict,
        "allocation_cap": allocation_cap,
        "evidence": _clip_text(
            (top.get("title") or top.get("text")) if top else profile["evidence"]
        ),
        "state": top.get("state") if top else None,
        "trust_score": trust,
        "trade_count": trade_count,
        "trade_ev": trade_ev,
        "trade_wr": trade_wr,
        "has_exact": bool(exact),
        "pair_context": _format_feedback_stat(pair_stat),
        "tape_context": _format_feedback_stat(tape_stat, _live_tape_bucket_label(current_tape_bucket)),
        "recent_feedback_note": pair_feedback.get("reason") if pair_feedback else None,
        "session_context": _format_context_stat(session_stat, current_session),
        "regime_context": _format_context_stat(regime_stat, current_regime or "regime"),
        "tape_stat": tape_stat,
        "context_bias": context_bias,
    })
    return profile


def _cap_setup_allocation(setup: dict, learning_profile: dict) -> dict:
    raw_grade = setup.get("allocation_grade", setup.get("edge_grade", "C"))
    capped_grade = _cap_grade(
        raw_grade,
        _learning_cap_to_grade(learning_profile.get("allocation_cap", CAP_LABELS["b_scout"])),
    )
    setup["raw_allocation_grade"] = raw_grade
    setup["allocation_grade"] = capped_grade
    setup["learning_score"] = learning_profile.get("learning_score")
    setup["learning_verdict"] = learning_profile.get("verdict")
    setup["learning_cap"] = learning_profile.get("allocation_cap")
    setup["learning_context"] = (
        f"{learning_profile.get('session_context')} | {learning_profile.get('regime_context')}"
    )
    setup["learning_evidence"] = learning_profile.get("evidence")
    if capped_grade != raw_grade:
        setup["details"].append(
            f"Learning cap: {learning_profile.get('allocation_cap')} → allocation {raw_grade}→{capped_grade}"
        )
    setup["allocation_band"] = capped_grade
    setup["allocation_band_reason"] = f"base allocation {capped_grade}"
    setup["sizing"] = _sizing_text_for_allocation(capped_grade, capped_grade)
    return setup


def _derive_allocation_band(
    setup: dict,
    learning_profile: dict,
    execution_plan: dict,
    risk_level: str,
) -> tuple[str, str]:
    allocation_grade = str(setup.get("allocation_grade", setup.get("edge_grade", "C")) or "C")
    if allocation_grade != "B":
        return allocation_grade, f"base allocation stays {allocation_grade}"

    edge_grade = str(setup.get("edge_grade", allocation_grade) or allocation_grade)
    raw_alloc = str(setup.get("raw_allocation_grade", allocation_grade) or allocation_grade)
    learning_score = int(learning_profile.get("learning_score", 0) or 0)
    allocation_cap = str(learning_profile.get("allocation_cap") or CAP_LABELS["b_scout"])
    verdict = str(learning_profile.get("verdict") or "").lower()
    context_bias = str(learning_profile.get("context_bias") or "neutral")
    exec_style = str(execution_plan.get("style") or "PASS").upper()

    positive_signals: list[str] = []
    negative_signals: list[str] = []

    if raw_alloc in {"A", "S"} or edge_grade in {"A", "S"}:
        positive_signals.append("raw seat quality was A/S before capping")
    if learning_score >= 80:
        positive_signals.append(f"learning score {learning_score}/99 is promotable")
    if allocation_cap in {CAP_LABELS["as_confirmed"], CAP_LABELS["a_max"], CAP_LABELS["ba_max"]}:
        positive_signals.append(f"learning cap is {allocation_cap}")
    if exec_style in {"LIMIT", "STOP-ENTRY"}:
        positive_signals.append(f"{exec_style} keeps the seat trigger-honest")
    if context_bias == "tailwind":
        positive_signals.append("current context is a tailwind")

    if exec_style == "PASS":
        negative_signals.append("execution still closes as PASS")
    if allocation_cap in {CAP_LABELS["pass"], CAP_LABELS["b_only"]}:
        negative_signals.append(f"learning cap is restrictive ({allocation_cap})")
    if "no-edge" in verdict or "restricted" in verdict:
        negative_signals.append("learning verdict is no-edge / restricted")
    if context_bias == "headwind":
        negative_signals.append("current context is a headwind")
    if risk_level == "HIGH":
        negative_signals.append("risk stack is HIGH")

    if len(negative_signals) >= 3 and len(positive_signals) <= 2:
        reason_bits = negative_signals[:2] or ["seat is mostly caution-driven"]
        return "B-", " | ".join(reason_bits)
    if len(positive_signals) >= 3 and len(negative_signals) <= 2:
        reason_bits = positive_signals[:2] or ["seat is promotable once timing resolves"]
        return "B+", " | ".join(reason_bits)

    reason_bits = []
    if positive_signals:
        reason_bits.append(positive_signals[0])
    if negative_signals:
        reason_bits.append(negative_signals[0])
    if not reason_bits:
        reason_bits.append("mixed evidence keeps this in the middle B bucket")
    return "B0", " | ".join(reason_bits)


def _recommend_execution_style(
    pair: str,
    direction: str,
    setup: dict,
    learning_profile: dict,
    risk_level: str,
    promotion_pressure: dict | None = None,
) -> dict:
    session_bucket = learning_profile.get("current_session")
    regime = learning_profile.get("current_regime")
    pair = str(pair)
    spread_pips = setup.get("spread_pips")
    spread_ratio = setup.get("spread_ratio")
    wave = setup.get("wave")
    learning_score = int(learning_profile.get("learning_score", 0) or 0)
    trade_count = int(learning_profile.get("trade_count", 0) or 0)
    cap_rank = _cap_rank(learning_profile.get("allocation_cap", CAP_LABELS["b_scout"]))
    has_exact = bool(learning_profile.get("has_exact"))
    confirmed_state = learning_profile.get("state") == "confirmed"
    tight_market_spread, normal_market_spread = _market_spread_bands(pair, spread_pips, spread_ratio)
    market_allowed = str(setup.get("allocation_grade", setup.get("edge_grade", "C"))) in {"A", "S"}
    reasons: list[str] = []

    if learning_profile.get("allocation_cap") == CAP_LABELS["pass"] or setup.get("allocation_grade") == "C":
        style, note = _pass_cap_scout_participation_plan(
            pair,
            learning_profile,
            risk_level=risk_level,
            normal_market_spread=normal_market_spread,
            promotion_pressure=promotion_pressure,
        )
        if style and note:
            reasons.append(note)
            return {"style": style, "note": "; ".join(reasons)}
        reasons.append("learning cap or final allocation says this seat is not worth real risk")
        return {"style": "PASS", "note": "; ".join(reasons)}

    if learning_profile.get("context_bias") == "headwind" and risk_level == "HIGH":
        reasons.append("historical learning and current risk stack are both against this seat")
        return {"style": "PASS", "note": "; ".join(reasons)}

    if spread_ratio is not None and spread_ratio > 0.30 and not normal_market_spread:
        if regime in {"squeeze", "transition"}:
            reasons.append(
                f"spread {spread_pips:.1f}pip is too expensive for immediate entry while {regime} still needs proof"
            )
            return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}
        reasons.append(f"spread {spread_pips:.1f}pip destroys the payoff shape for a market fill")
        return {"style": "LIMIT", "note": "; ".join(reasons)}

    if regime in {"range", "quiet"} and wave in {"big", "mid"}:
        reasons.append(f"{regime} regime rewards price improvement more than chasing")
        return {"style": "LIMIT", "note": "; ".join(reasons)}

    if session_bucket in {"newyork", "late"} and spread_ratio is not None and spread_ratio > 0.24 and not normal_market_spread:
        reasons.append(
            f"{session_bucket} bucket with {spread_pips:.1f}pip spread is execution-sensitive"
        )
        return {"style": "LIMIT", "note": "; ".join(reasons)}

    if learning_profile.get("context_bias") == "headwind":
        support = trade_count >= 3 or has_exact or confirmed_state
        if regime in {"trending", "transition", "squeeze"} and risk_level != "HIGH":
            if tight_market_spread and support:
                if cap_rank >= 2 and learning_score >= 50:
                    reasons.append(
                        "historical headwind plus sub-ideal conviction does not deserve fresh market risk; arm the trigger first and leave one reload LIMIT"
                    )
                    return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}
                if cap_rank == 1 and learning_score >= 58:
                    reasons.append(
                        "even with tight spread, B-only memory is not enough for a blind market scout; require trigger proof first"
                    )
                    return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}
            if normal_market_spread and learning_score >= 45 and (support or cap_rank >= 2):
                reasons.append(
                    "historical headwind blocks blind chase, but the tape is alive enough to arm a trigger instead of passing"
                )
                return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}
        reasons.append("learning context is headwind, so require either price improvement or trigger proof")
        style = "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"
        return {"style": style, "note": "; ".join(reasons)}

    if regime in {"squeeze", "transition"} and not setup.get("is_counter"):
        if (
            market_allowed
            and
            cap_rank >= 4
            and learning_score >= 64
            and normal_market_spread
            and (trade_count >= 3 or has_exact or confirmed_state or learning_profile.get("context_bias") == "tailwind")
            and risk_level != "HIGH"
        ):
            reasons.append(
                f"{regime} is already leaning one way and the seat has earned immediate participation"
            )
            return {"style": "MARKET", "note": "; ".join(reasons)}
        if (
            cap_rank in {1, 2, 3}
            and learning_score >= 53
            and normal_market_spread
            and (
                learning_profile.get("context_bias") == "tailwind"
                or trade_count >= 3
                or has_exact
                or confirmed_state
            )
            and risk_level != "HIGH"
        ):
            reasons.append(
                f"{regime} is leaning enough to keep a trigger alive, but fresh market risk stays reserved for A/S seats"
            )
            return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}
        reasons.append(f"{regime} pays better after the reclaim/breakout proves itself")
        return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}

    if regime == "trending":
        if (
            market_allowed
            and
            cap_rank >= 4
            and learning_score >= 58
            and normal_market_spread
            and (trade_count >= 3 or has_exact or confirmed_state or learning_profile.get("context_bias") == "tailwind")
            and risk_level != "HIGH"
        ):
            if tight_market_spread or learning_profile.get("context_bias") == "tailwind" or cap_rank >= 5:
                reasons.append("trend is already paying and the learning cap is strong enough for immediate execution")
            else:
                reasons.append("trend is live enough now; enter the market print and still keep one reload LIMIT")
            return {"style": "MARKET", "note": "; ".join(reasons)}
        if (
            cap_rank in {1, 2, 3}
            and learning_score >= 50
            and normal_market_spread
            and (
                learning_profile.get("context_bias") == "tailwind"
                or trade_count >= 3
                or has_exact
                or confirmed_state
            )
            and risk_level != "HIGH"
        ):
            reasons.append(
                "live tape is leaning, but B seats still need trigger proof instead of a blind market scout"
            )
            return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}

    if risk_level == "HIGH":
        reasons.append("setup exists, but the risk stack is elevated enough to avoid paying market")
        return {"style": "LIMIT", "note": "; ".join(reasons)}

    if regime == "trending":
        reasons.append("trend is alive enough to keep a trigger armed even when the market scout lane is not justified")
        return {"style": "STOP-ENTRY", "note": "; ".join(reasons)}

    if not market_allowed:
        reasons.append("fresh market risk is reserved for A/S seats; this seat must prove itself as a trigger or better price first")
        style = "STOP-ENTRY" if regime in {"trending", "transition", "squeeze"} else "LIMIT"
        return {"style": style, "note": "; ".join(reasons)}

    reasons.append("edge, learning cap, session, and spread all support immediate execution")
    return {"style": "MARKET", "note": "; ".join(reasons)}

def assess_risk(
    pair: str,
    direction: str,
    adx: float = None,
    headline: str = None,
    regime: str = None,
    wave: str = "auto",
    counter: bool = False,
    entry_price: float | None = None,
    tp_price: float | None = None,
    sl_price: float | None = None,
    tp_pips: float | None = None,
    sl_pips: float | None = None,
    spread_pips: float | None = None,
    live_tape: dict | None = None,
    log_result: bool = True,
) -> dict:
    """Overall risk + setup quality assessment"""
    init_db(quiet=True)
    conn = get_conn()
    if regime:
        regime = str(regime).strip().lower()
    warnings = []
    risk_score = 0  # 0-10

    # 1. Trade statistics
    stats = trade_stats(conn, pair, direction)
    if stats["count"] > 0:
        if stats["expectancy"] < 0:
            warnings.append(
                f"⚠️ {pair} {direction} expectancy: {stats['expectancy']:+,.0f} JPY/trade "
                f"(avg win {stats['avg_win']:+,.0f} vs avg loss {stats['avg_loss']:+,.0f})"
            )
            risk_score += 2
        be_wr = stats.get("break_even_win_rate")
        if (
            be_wr is not None
            and stats["count"] >= 5
            and stats["win_rate"] + 0.05 < be_wr
        ):
            warnings.append(
                f"⚠️ Break-even WR is {be_wr:.0%}, actual WR is {stats['win_rate']:.0%} "
                f"→ payoff shape is not carrying the misses yet"
            )
            risk_score += 1
        if stats["win_rate"] < 0.5:
            warnings.append(f"⚠️ {pair} {direction} win rate: {stats['win_rate']:.0%} ({stats['wins']}/{stats['count']})")
            risk_score += 2
        if stats["worst"] < -1000:
            warnings.append(f"🚨 Historical max loss: {stats['worst']:+,.0f} JPY")
            risk_score += 3

    # 2. By regime
    if regime:
        rs = regime_stats(conn, pair, direction, regime)
        if rs["count"] > 0 and rs["win_rate"] < 0.4:
            warnings.append(f"⚠️ Win rate in {regime} regime: {rs['win_rate']:.0%} ({rs['count']} trades)")
            risk_score += 2

    # 3. Headline / event risk
    if headline:
        hl_trades = active_headlines_history(conn, headline)
        if hl_trades:
            hl_pls = [t["pl"] for t in hl_trades]
            avg = sum(hl_pls) / len(hl_pls)
            if avg < 0:
                warnings.append(f"🚨 Avg P&L during {headline} headline: {avg:+,.0f} JPY ({len(hl_trades)} trades)")
                risk_score += 3
            losses = [t for t in hl_trades if t["pl"] < -500]
            for lt in losses:
                warnings.append(f"  → {lt['pair']} {lt['direction']} {lt['pl']:+,.0f} JPY: {lt['lesson']}")

    events = headline_risk(conn, pair)
    if events:
        high_events = [e for e in events if e["impact"] == "high"]
        if high_events:
            for e in high_events[:2]:
                warnings.append(f"🚨 Historical spike: {e['headline']} → {pair} {e['spike_pips']:.0f}pip ({e['session_date']})")
                risk_score += 2

    # 4. User call (only recent — market conditions change fast)
    uc = latest_user_call(conn, pair, max_age_days=3)
    if uc:
        user_dir = uc.get("direction")
        verified = uc.get("outcome") is not None
        call_date = uc.get("session_date", "?")
        if user_dir and user_dir != ("UP" if direction == "LONG" else "DOWN"):
            uc_stats = user_call_stats(conn, pair)
            if verified and uc_stats["count"] > 0:
                warnings.append(
                    f"⚠️ User call ({call_date}): \"{uc['call_text']}\" ({user_dir}) "
                    f"= opposite to {direction} (verified {uc_stats['accuracy']:.0%}, n={uc_stats['count']})"
                )
                risk_score += 2
            else:
                # Unverified call — note it but don't add risk score
                warnings.append(
                    f"ℹ️ User call ({call_date}): \"{uc['call_text']}\" ({user_dir}) "
                    f"= opposite to {direction} (unverified — info only, no score impact)"
                )

    # 5. Historical results without SL
    if stats["count"] > 0 and stats["no_sl_count"] == stats["count"]:
        losses_no_sl = [t for t in fetchall_dict(
            conn,
            """SELECT pl
               FROM trades
               WHERE pair = ?
                 AND had_sl = 0
                 AND pl IS NOT NULL
                 AND pl < 0
                 AND session_date >= ?""",
            (pair, live_history_start(None)),
        )]
        if losses_no_sl:
            worst = min(t["pl"] for t in losses_no_sl)
            warnings.append(f"⚠️ Max loss without SL: {worst:+,.0f} JPY → consider setting SL")
            risk_score += 1

    # Risk level
    if risk_score >= 7:
        level = "HIGH"
    elif risk_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    planned_target_pips, planned_stop_pips = _planned_pips(
        pair,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
    )

    # 6b. Setup Quality Assessment (forward-looking)
    if counter:
        setup = assess_counter_trade(
            pair,
            direction,
            planned_target_pips=planned_target_pips,
            planned_stop_pips=planned_stop_pips,
            spread_pips=spread_pips,
        )
        setup["is_counter"] = True
    else:
        setup = assess_setup_quality(
            pair,
            direction,
            wave=wave,
            planned_target_pips=planned_target_pips,
            planned_stop_pips=planned_stop_pips,
            spread_pips=spread_pips,
        )
        setup["is_counter"] = False

    # 6c. Pair-level history context (data, not grade override — recording.md: "you make the call")
    if stats["count"] >= 5 and not counter:
        if stats["win_rate"] < 0.40:
            # Low WR warning — data only, no grade cap
            # Stats are regime-dependent (4/9 lesson: bullish-period SHORTs inflate loss counts)
            warnings.append(
                f"⚠ PAIR HISTORY: {pair} {direction} all-time WR={stats['win_rate']:.0%} "
                f"({stats['count']} trades, total {stats['total_pl']:+,.0f}JPY). "
                f"Stats are from sample period — check if current H4 structure matches."
            )
            setup["details"].append(
                f"Low historical WR={stats['win_rate']:.0%} ({stats['count']} trades) — data point, not cap"
            )
        elif stats["win_rate"] >= 0.60:
            # Trending pair with proven edge: bonus for high-conviction
            tfs = _load_technicals(pair)
            h1 = tfs.get("H1", {})
            h1_adx = h1.get("adx", 0) if h1 else 0
            cs = _calc_currency_strength()
            base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
            cs_aligned = (
                (direction == "LONG" and cs.get(base, 0) > cs.get(quote, 0))
                or (direction == "SHORT" and cs.get(base, 0) < cs.get(quote, 0))
            )
            if h1_adx > 35 and cs_aligned:
                old_score = setup.get("quality_score", 0)
                setup["quality_score"] = old_score + 2
                setup["details"].append(
                    f"TRENDING BONUS +2: WR={stats['win_rate']:.0%}, H1 ADX={h1_adx:.0f}, macro aligned"
                )
                # Re-grade
                qs = setup["quality_score"]
                if qs >= 8:
                    setup["grade"] = "S"
                    setup["edge_grade"] = "S"
                    setup["allocation_grade"] = "S"
                    setup["sizing"] = "8000-10000u (iron-clad. size up)"
                elif qs >= 6:
                    setup["grade"] = "A"
                    setup["edge_grade"] = "A"
                    setup["allocation_grade"] = "A"
                    setup["sizing"] = "5000-8000u (high conviction. trade it properly)"

    # 6d. Macro regime conflict warning
    try:
        cs = _calc_currency_strength()
        base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        base_str = cs.get(base, 0)
        quote_str = cs.get(quote, 0)
        gap = base_str - quote_str
        if direction == "LONG" and gap < -0.3:
            warnings.append(
                f"⚠ AGAINST macro flow: {base}({base_str:+.2f}) vs {quote}({quote_str:+.2f}). "
                f"CS gap={gap:+.2f} AGAINST LONG."
            )
        elif direction == "SHORT" and gap > 0.3:
            warnings.append(
                f"⚠ AGAINST macro flow: {base}({base_str:+.2f}) vs {quote}({quote_str:+.2f}). "
                f"CS gap={gap:+.2f} AGAINST SHORT."
            )
    except Exception:
        pass

    promotion_pressure = _recent_promotion_pressure(pair, direction)
    effective_live_tape = dict(live_tape or {}) if live_tape else None
    if effective_live_tape is None and (
        setup.get("allocation_grade") != "C" or int(promotion_pressure.get("rank", 0) or 0) >= 12
    ):
        effective_live_tape = _fresh_live_tape_summary(pair)
    learning_profile = _build_learning_profile(
        conn,
        pair,
        direction,
        regime_hint=regime,
        live_tape=effective_live_tape,
    )
    learning_profile["promotion_pressure"] = promotion_pressure
    setup = _cap_setup_allocation(setup, learning_profile)
    execution_plan = _recommend_execution_style(
        pair,
        direction,
        setup,
        learning_profile,
        level,
        promotion_pressure=promotion_pressure,
    )
    execution_plan = _apply_live_tape_execution_guard(
        pair,
        direction,
        learning_profile,
        execution_plan,
        effective_live_tape,
        risk_level=level,
        warnings=warnings,
    )
    execution_plan = _apply_recent_execution_style_guard(
        conn,
        pair,
        direction,
        learning_profile,
        execution_plan,
        warnings=warnings,
    )
    same_pair_contest = _same_pair_direction_contest(
        conn,
        pair,
        direction,
        regime=regime,
        wave=wave,
        counter=counter,
        risk_level=level,
        current_setup=setup,
        current_learning=learning_profile,
        current_execution=execution_plan,
        live_tape=live_tape,
    )
    setup, execution_plan = _apply_same_pair_direction_contest(
        pair,
        warnings,
        setup,
        execution_plan,
        same_pair_contest,
    )
    setup, execution_plan = _apply_planned_target_floor_guard(
        pair,
        warnings,
        setup,
        execution_plan,
    )
    setup, execution_plan = _apply_planned_stop_floor_guard(
        pair,
        warnings,
        setup,
        execution_plan,
    )
    setup, execution_plan = _apply_counter_reversal_guard(
        pair,
        warnings,
        setup,
        execution_plan,
    )
    same_direction_inventory = _same_direction_inventory_summary(pair, direction)
    setup, execution_plan = _apply_same_direction_inventory_guard(
        pair,
        direction,
        warnings,
        setup,
        execution_plan,
        same_direction_inventory,
        learning_profile=learning_profile,
        promotion_pressure=promotion_pressure,
    )
    setup, execution_plan = _apply_negative_expectancy_guard(
        pair,
        direction,
        warnings,
        setup,
        learning_profile,
        execution_plan,
        stats,
        promotion_pressure=promotion_pressure,
    )
    thesis_context = _build_thesis_context(pair, direction, setup, learning_profile, execution_plan)
    past_outcomes = _past_pretrade_outcomes(
        conn,
        pair,
        direction,
        level,
        thesis_key=thesis_context.get("key"),
        thesis_family=thesis_context.get("family"),
        thesis_market=thesis_context.get("market"),
        thesis_trigger=thesis_context.get("trigger"),
        thesis_vehicle=thesis_context.get("vehicle"),
    )
    thesis_feedback = _recent_thesis_feedback(
        conn,
        pair,
        direction,
        level=level,
        thesis_context=thesis_context,
    )
    setup, learning_profile, execution_plan = _apply_thesis_feedback_guard(
        warnings,
        setup,
        learning_profile,
        execution_plan,
        thesis_feedback,
    )
    thesis_context["age"], thesis_context["age_label"] = _thesis_age_label(thesis_feedback)
    allocation_band, allocation_band_reason = _derive_allocation_band(
        setup,
        learning_profile,
        execution_plan,
        level,
    )
    setup["allocation_band"] = allocation_band
    setup["allocation_band_reason"] = allocation_band_reason
    setup["sizing"] = _sizing_text_for_allocation(setup.get("allocation_grade", "C"), allocation_band)
    thesis_context = _build_thesis_context(pair, direction, setup, learning_profile, execution_plan)
    thesis_context["age"], thesis_context["age_label"] = _thesis_age_label(thesis_feedback)

    result = {
        "pair": pair,
        "direction": direction,
        "risk_level": level,
        "risk_score": risk_score,
        "warnings": warnings,
        "trade_stats": stats if stats["count"] > 0 else None,
        "past_outcomes": past_outcomes,
        "setup_quality": setup,
        "learning_profile": learning_profile,
        "execution_plan": execution_plan,
        "live_tape": effective_live_tape,
        "same_pair_contest": same_pair_contest,
        "same_direction_inventory": same_direction_inventory,
        "hard_execution_blockers": _collect_hard_execution_blockers(execution_plan),
        "promotion_pressure": promotion_pressure,
        "thesis_context": thesis_context,
        "thesis_feedback": thesis_feedback,
        "planned_trade": {
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_pips": planned_target_pips,
            "sl_pips": planned_stop_pips,
        },
    }

    # 7. Similar situations via vector search
    query_parts = [pair, direction]
    if headline:
        query_parts.append(headline)
    if adx:
        query_parts.append(f"ADX={adx}")
    if setup.get("wave"):
        query_parts.append(f"{setup['wave']} wave")
    query_parts.extend(["lesson", "failure", "success"])
    query_text = " ".join(query_parts)
    narratives = similar_trades_narrative(query_text, pair=pair, direction=direction, top_k=2)
    if narratives:
        result["similar_memories"] = [
            {"date": n.get("session_date", "?"), "content": n.get("content", "")[:200]}
            for n in narratives
        ]

    # 8. Log this check result to pretrade_outcomes (daily_review will fill in pl later)
    if log_result:
        thesis = _build_pretrade_thesis(result, headline)
        _log_pretrade(
            conn,
            pair,
            direction,
            level,
            risk_score,
            warnings,
            thesis,
            metadata={
                "learning_score": learning_profile.get("learning_score"),
                "learning_verdict": learning_profile.get("verdict"),
                "learning_cap": learning_profile.get("allocation_cap"),
                "session_bucket": learning_profile.get("current_session"),
                "regime_snapshot": learning_profile.get("current_regime"),
                "execution_style": execution_plan.get("style"),
                "execution_note": execution_plan.get("note"),
                "allocation_band": setup.get("allocation_band"),
                "thesis_key": thesis_context.get("key"),
                "thesis_family": thesis_context.get("family"),
                "thesis_market": thesis_context.get("market"),
                "thesis_structure": thesis_context.get("structure"),
                "thesis_trigger": thesis_context.get("trigger"),
                "thesis_vehicle": thesis_context.get("vehicle"),
                "thesis_age": thesis_context.get("age"),
                "live_tape_bias": (effective_live_tape or {}).get("bias"),
                "live_tape_state": (effective_live_tape or {}).get("tape"),
                "live_tape_bucket": _live_tape_bucket(direction, effective_live_tape),
                "live_tape_samples": (effective_live_tape or {}).get("samples"),
                "live_tape_mode": (effective_live_tape or {}).get("probe_mode"),
            },
        )

    return result


def _past_pretrade_outcomes(
    conn,
    pair: str,
    direction: str,
    level: str,
    *,
    thesis_key: str | None = None,
    thesis_family: str | None = None,
    thesis_market: str | None = None,
    thesis_trigger: str | None = None,
    thesis_vehicle: str | None = None,
) -> list[dict]:
    """Outcomes for the exact thesis first, then the thesis layers."""
    seen_ids: set[int] = set()
    results: list[dict] = []

    for scope, rows in (
        ("same thesis", _recent_outcome_rows(conn, pair, direction, thesis_key=thesis_key, exact_only=True)),
        (
            "same family",
            _recent_outcome_rows(
                conn,
                pair,
                direction,
                thesis_key=thesis_key,
                thesis_family=thesis_family,
                family_only=True,
            ),
        ),
        (
            "same market",
            _recent_outcome_rows(
                conn,
                pair,
                direction,
                layer_filters={"thesis_market": thesis_market},
            ),
        ),
        (
            "same trigger",
            _recent_outcome_rows(
                conn,
                pair,
                direction,
                layer_filters={"thesis_trigger": thesis_trigger},
            ),
        ),
        (
            "same vehicle",
            _recent_outcome_rows(
                conn,
                pair,
                direction,
                layer_filters={"thesis_vehicle": thesis_vehicle},
            ),
        ),
    ):
        for row in rows:
            row_id = int(row.get("id") or 0)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            enriched = dict(row)
            enriched["match_scope"] = scope
            results.append(enriched)
            if len(results) >= THESIS_HISTORY_LIMIT:
                return results

    return results


def _build_pretrade_thesis(result: dict, headline: str | None = None) -> str:
    setup = result.get("setup_quality") or {}
    learning = result.get("learning_profile") or {}
    execution = result.get("execution_plan") or {}
    live_tape = result.get("live_tape") or {}
    edge = setup.get("edge_grade") or setup.get("grade") or "?"
    allocation = setup.get("allocation_grade") or edge
    allocation_band = setup.get("allocation_band") or allocation
    wave = setup.get("wave") or "?"
    details = setup.get("details") or []
    detail_text = " | ".join(str(detail).strip() for detail in details[:2] if detail)
    detail_text = detail_text[:220]
    headline_text = f" headline={headline}" if headline else ""
    learn_cap = learning.get("allocation_cap") or "?"
    exec_style = execution.get("style") or "?"
    thesis = (
        f"edge={edge} alloc={allocation} alloc_band={allocation_band} "
        f"learn={learn_cap} exec={exec_style} wave={wave}{headline_text}"
    )
    if live_tape and live_tape.get("tape") and live_tape.get("tape") != "unavailable":
        tape_bias = str(live_tape.get("bias") or "unknown").replace(" ", "_")
        tape_state = str(live_tape.get("tape") or "unknown").replace(" ", "_").replace("/", "_")
        thesis += f" tape={tape_bias}:{tape_state}"
    if detail_text:
        thesis += f" | {detail_text}"
    return thesis[:280]


def _recent_duplicate_pretrade(
    conn,
    pair: str,
    direction: str,
    level: str,
    score: int,
    thesis: str,
    lookback_minutes: int = 15,
) -> bool:
    recent = fetchone_val(
        conn,
        f"""SELECT 1
            FROM pretrade_outcomes
            WHERE pair = ?
              AND direction = ?
              AND pretrade_level = ?
              AND pretrade_score = ?
              AND thesis = ?
              AND trade_id IS NULL
              AND pl IS NULL
              AND datetime(created_at) >= datetime('now', 'localtime', '-{lookback_minutes} minutes')
            ORDER BY id DESC
            LIMIT 1""",
        (pair, direction, level, score, thesis),
    )
    return recent is not None


def _log_pretrade(
    conn,
    pair: str,
    direction: str,
    level: str,
    score: int,
    warnings: list,
    thesis: str,
    metadata: dict | None = None,
):
    """Log this check result to pretrade_outcomes"""
    try:
        if _recent_duplicate_pretrade(conn, pair, direction, level, score, thesis):
            return
        metadata = metadata or {}
        conn.execute(
            """INSERT INTO pretrade_outcomes
               (
                 session_date, pair, direction, pretrade_level, pretrade_score,
                 pretrade_warnings, thesis,
                 learning_score, learning_verdict, learning_cap,
                 session_bucket, regime_snapshot, execution_style, execution_note, allocation_band,
                 thesis_key, thesis_family,
                 thesis_market, thesis_structure, thesis_trigger, thesis_vehicle, thesis_age,
                 live_tape_bias, live_tape_state, live_tape_bucket, live_tape_samples, live_tape_mode
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(date.today()),
                pair,
                direction,
                level,
                score,
                json.dumps(warnings, ensure_ascii=False),
                thesis,
                metadata.get("learning_score"),
                metadata.get("learning_verdict"),
                metadata.get("learning_cap"),
                metadata.get("session_bucket"),
                metadata.get("regime_snapshot"),
                metadata.get("execution_style"),
                metadata.get("execution_note"),
                metadata.get("allocation_band"),
                metadata.get("thesis_key"),
                metadata.get("thesis_family"),
                metadata.get("thesis_market"),
                metadata.get("thesis_structure"),
                metadata.get("thesis_trigger"),
                metadata.get("thesis_vehicle"),
                metadata.get("thesis_age"),
                metadata.get("live_tape_bias"),
                metadata.get("live_tape_state"),
                metadata.get("live_tape_bucket"),
                metadata.get("live_tape_samples"),
                metadata.get("live_tape_mode"),
            )
        )
    except Exception:
        pass  # ignore if table does not yet exist


def _summarize_as_proof_combo(setup: dict) -> str | None:
    details = [str(detail).strip() for detail in (setup.get("details") or []) if detail]
    if not details:
        return None

    picks: list[str] = []

    def take(*needles: str) -> None:
        for detail in details:
            lowered = detail.lower()
            if any(needle in lowered for needle in needles) and detail not in picks:
                picks.append(detail)
                return

    take("mtf")
    take("technical confluence", "reversal confirmed", "h1 divergence", "h4 extreme")
    take("macro")

    for detail in details:
        if detail not in picks:
            picks.append(detail)
        if len(picks) >= 3:
            break

    compact = []
    for detail in picks[:3]:
        compact.append(
            detail.replace("← don't enter", "")
            .replace("← no counter-trade basis", "")
            .replace("← wait for timing", "")
            .strip()
        )

    summary = " + ".join(compact).strip()
    return summary[:220] if summary else None


def format_check(result: dict) -> str:
    """Format check result for readable display"""
    lines = []
    level = result["risk_level"]
    setup = result.get("setup_quality", {})
    learning = result.get("learning_profile", {})
    execution = result.get("execution_plan", {})
    grade = setup.get("grade", "?")
    is_counter = setup.get("is_counter", False)

    if is_counter:
        q_score = setup.get("score", 0)
        edge_grade = setup.get("edge_grade", grade)
        allocation_grade = setup.get("allocation_grade", edge_grade)
        allocation_band = setup.get("allocation_band", allocation_grade)
        score_max = setup.get("score_max", 7)

        lines.append(
            f"🔄 COUNTER-TRADE | {_grade_icon(edge_grade)} Edge: {edge_grade} "
            f"(score={q_score}/{score_max}) | Allocation: {allocation_grade}"
            f"{'' if allocation_band == allocation_grade else f' ({allocation_band})'} | "
            f"Risk: {level} (score={result['risk_score']})"
        )
        lines.append(
            f"   {result['pair']} {result['direction']} (M5 against H4/H1) "
            f"→ {setup.get('sizing', '?')}"
        )
        lines.append("")
        lines.append("📐 Counter-trade evaluation (inverted axes — H4 extreme = FOR):")
        for detail in setup.get("details", []):
            lines.append(f"  {detail}")
        if setup.get("counter_note"):
            lines.append(f"  Note: {setup['counter_note']}")
    else:
        q_score = setup.get("quality_score", 0)
        edge_grade = setup.get("edge_grade", grade)
        allocation_grade = setup.get("allocation_grade", edge_grade)
        allocation_band = setup.get("allocation_band", allocation_grade)
        score_max = setup.get("score_max", 10)

        # Main header: conviction is most important (determines sizing)
        wave_label = {"big": "big wave", "mid": "mid wave", "small": "small wave"}.get(setup.get("wave", "?"), "?")
        lines.append(
            f"{_grade_icon(edge_grade)} Edge: {edge_grade} (score={q_score}/{score_max}) "
            f"| Allocation: {allocation_grade}"
            f"{'' if allocation_band == allocation_grade else f' ({allocation_band})'} | Risk: {level} "
            f"(score={result['risk_score']}) | Wave: {wave_label}"
        )
        lines.append(f"   {result['pair']} {result['direction']} → {setup.get('sizing', '?')}")
        lines.append("")

        # Setup quality details
        lines.append("📐 Setup quality:")
        for detail in setup.get("details", []):
            lines.append(f"  {detail}")

    raw_alloc = setup.get("raw_allocation_grade")
    if raw_alloc and raw_alloc != setup.get("allocation_grade"):
        lines.append(f"  Allocation capped by learning: {raw_alloc} → {setup.get('allocation_grade')}")
    allocation_band = setup.get("allocation_band")
    allocation_band_reason = setup.get("allocation_band_reason")
    if allocation_band and allocation_band != setup.get("allocation_grade"):
        lines.append(f"  B-band split: {allocation_band} → {allocation_band_reason}")
    combo_summary = _summarize_as_proof_combo(setup)
    if combo_summary:
        lines.append(f"🧩 Strongest A/S combo: {combo_summary}")
    planned_trade = result.get("planned_trade") or {}
    if planned_trade.get("tp_pips") or planned_trade.get("sl_pips"):
        tp_pips = planned_trade.get("tp_pips")
        sl_pips = planned_trade.get("sl_pips")
        spread = setup.get("spread_pips")
        tp_ratio = setup.get("target_spread_multiple")
        sl_ratio = setup.get("stop_spread_multiple")
        parts = []
        if tp_pips is not None:
            tp_text = f"TP {tp_pips:.1f}pip"
            if tp_ratio is not None:
                tp_text += f" ({tp_ratio:.1f}x spread)"
            parts.append(tp_text)
        if sl_pips is not None:
            sl_text = f"SL {sl_pips:.1f}pip"
            if sl_ratio is not None:
                sl_text += f" ({sl_ratio:.1f}x spread)"
            parts.append(sl_text)
        if spread is not None:
            parts.append(f"spread {spread:.1f}pip")
        lines.append(f"📏 Planned payoff: {' | '.join(parts)}")
        if setup.get("noise_stop_floor_pips"):
            lines.append(
                f"   Historical noise floor: stop wants at least {float(setup.get('noise_stop_floor_pips') or 0.0):.1f}pip"
            )
    lines.append("")

    if learning:
        lines.append(
            f"🧠 Learning gate: {int(learning.get('learning_score', 0))}/99 "
            f"| {learning.get('verdict', '?')} | cap {learning.get('allocation_cap', '?')}"
        )
        lines.append(f"   Evidence: {learning.get('evidence', '?')}")
        lines.append(f"   Pair feedback: {learning.get('pair_context', '?')}")
        if learning.get("tape_stat") and int(learning["tape_stat"].get("count", 0) or 0) > 0:
            lines.append(f"   Tape-matched feedback: {learning.get('tape_context', '?')}")
        lines.append(
            f"   Context: {learning.get('session_context', '?')} | "
            f"{learning.get('regime_context', '?')}"
        )
        if learning.get("recent_feedback_note"):
            lines.append(f"   Recent feedback override: {learning.get('recent_feedback_note')}")
        if learning.get("thesis_guard"):
            lines.append(f"   Thesis guard: {learning.get('thesis_guard')}")
        lines.append("")

    promotion_pressure = result.get("promotion_pressure") or {}
    if int(promotion_pressure.get("rank", 0) or 0) > 0:
        lines.append(
            f"🔥 Promotion pressure: {int(promotion_pressure.get('rank', 0) or 0)}/44 "
            f"| {promotion_pressure.get('note', '')}"
        )
        lines.append("")

    live_tape = result.get("live_tape")
    if live_tape:
        lines.append(f"📡 Live tape: {_live_tape_brief(live_tape)}")
        lines.append("")

    if execution:
        lines.append(
            f"⚙️ Execution plan: {execution.get('style', '?')} "
            f"| {execution.get('note', 'no note')}"
        )
        if execution.get("execution_style_feedback_note"):
            lines.append(f"   Vehicle feedback: {execution.get('execution_style_feedback_note')}")
            lines.append(f"   Vehicle sample: {execution.get('execution_style_context', '?')}")
        hard_blockers = result.get("hard_execution_blockers") or []
        if hard_blockers:
            lines.append("   Hard blockers:")
            for blocker in hard_blockers:
                lines.append(f"     - {blocker}")
        lines.append("")

    thesis_context = result.get("thesis_context") or {}
    if thesis_context:
        lines.append("🎯 Thesis stack:")
        lines.append(f"   Market: {thesis_context.get('market_label', '?')}")
        lines.append(f"   Structure: {thesis_context.get('structure_label', '?')}")
        lines.append(f"   Trigger: {thesis_context.get('trigger_label', '?')}")
        lines.append(f"   Vehicle: {thesis_context.get('vehicle_label', '?')}")
        lines.append(f"   Aging: {thesis_context.get('age_label', '?')}")
        lines.append(f"   Family: {thesis_context.get('family_label', '?')}")
        lines.append(f"   Key: {thesis_context.get('key_label', '?')}")
        lines.append("")

    contest = result.get("same_pair_contest")
    if contest:
        lines.append(
            f"↔ Same-pair contest: cleaner live side = {result['pair']} {contest['opposite_direction']} "
            f"({contest['opposite_style']}, edge {contest['opposite_edge_grade']}, "
            f"alloc {contest['opposite_allocation_grade']}, learning {contest['opposite_learning_score']}/99)"
        )
        lines.append("")

    same_direction_inventory = result.get("same_direction_inventory")
    if same_direction_inventory:
        protection = " | protected" if same_direction_inventory.get("has_protection") else " | unprotected"
        lines.append(
            f"📦 Same-direction live inventory: {same_direction_inventory['count']} leg(s) / "
            f"{same_direction_inventory['units']}u / UPL {same_direction_inventory['upl']:+,.0f} JPY"
            f"{protection}"
        )
        lines.append("")

    # Trade statistics
    stats = result.get("trade_stats")
    if stats:
        be_wr = stats.get("break_even_win_rate")
        pf = stats.get("profit_factor")
        be_text = f" | BE WR {be_wr:.0%}" if be_wr is not None else ""
        if pf is None:
            pf_text = ""
        elif pf == float("inf"):
            pf_text = " | PF inf"
        else:
            pf_text = f" | PF {pf:.2f}"
        lines.append(
            f"📊 Historical record: {stats['wins']}W {stats['losses']}L "
            f"(WR {stats['win_rate']:.0%}) | EV {stats['expectancy']:+,.0f} JPY/trade"
            f"{be_text}{pf_text}"
        )
        lines.append(
            f"   avg win {stats['avg_win']:+,.0f} | avg loss {stats['avg_loss']:+,.0f} "
            f"| R:R {stats['rr_ratio']:.2f} | total {stats['total_pl']:+,.0f} JPY"
        )

    # Warnings
    if result["warnings"]:
        lines.append("")
        for w in result["warnings"]:
            lines.append(w)

    # Historical lessons
    if stats and stats.get("lessons"):
        lines.append("")
        lines.append("📝 Historical lessons:")
        for lesson in stats["lessons"][:3]:
            lines.append(f"  - {lesson}")

    # Outcomes from past same-condition entries (core of feedback loop)
    past = result.get("past_outcomes", [])
    if past:
        lines.append("")
        lines.append(f"📖 Recent outcomes for this thesis/family ({result['pair']} {result['direction']}):")
        for p in past:
            outcome = "WIN" if p['pl'] and p['pl'] > 0 else "LOSS"
            pl_str = f"{p['pl']:+,.0f} JPY" if p['pl'] else "?"
            scope = f" [{p.get('match_scope')}]" if p.get("match_scope") else ""
            lines.append(f"  [{p['session_date']}] {outcome} {pl_str}{scope}")
            if p.get('lesson_from_review'):
                lines.append(f"    → {p['lesson_from_review']}")
            if p.get("collapse_layer"):
                lines.append(f"    collapse={p['collapse_layer']}: {p.get('collapse_note') or 'no note'}")

    # Similar memories
    if result.get("similar_memories"):
        lines.append("")
        lines.append("🧠 Memories of similar situations:")
        for mem in result["similar_memories"]:
            lines.append(f"  [{mem['date']}] {mem['content'][:150]}...")

    return "\n".join(lines)


# --- CLI ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 pretrade_check.py <PAIR> <LONG|SHORT> [--counter] [--wave big|mid|small] [--adx N] [--headline TEXT] [--regime TYPE] [--entry PRICE --tp PRICE --sl PRICE]")
        print("Example: python3 pretrade_check.py GBP_USD SHORT --wave big --entry 1.35160 --tp 1.35210 --sl 1.35074")
        print("Example: python3 pretrade_check.py EUR_JPY SHORT --counter")
        print("  --counter: evaluate as counter-trade (M5 against H4/H1 direction)")
        print("  --wave: big=H4/H1 swing, mid=M5 trade, small=M1 scalp, auto=auto-detect (default)")
        sys.exit(1)

    pair = sys.argv[1]
    direction = sys.argv[2]

    adx = None
    headline = None
    regime = None
    wave = "auto"
    counter_mode = False
    entry_price = None
    tp_price = None
    sl_price = None
    tp_pips = None
    sl_pips = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--counter":
            counter_mode = True; i += 1
        elif sys.argv[i] == "--adx" and i + 1 < len(sys.argv):
            adx = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--headline" and i + 1 < len(sys.argv):
            headline = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--regime" and i + 1 < len(sys.argv):
            regime = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--wave" and i + 1 < len(sys.argv):
            wave = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--entry" and i + 1 < len(sys.argv):
            entry_price = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--tp" and i + 1 < len(sys.argv):
            tp_price = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--sl" and i + 1 < len(sys.argv):
            sl_price = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--tp-pips" and i + 1 < len(sys.argv):
            tp_pips = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--sl-pips" and i + 1 < len(sys.argv):
            sl_pips = float(sys.argv[i + 1]); i += 2
        else:
            i += 1

    if counter_mode:
        result = assess_risk(
            pair,
            direction,
            adx=adx,
            headline=headline,
            regime=regime,
            wave=wave,
            counter=True,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
    else:
        result = assess_risk(
            pair,
            direction,
            adx=adx,
            headline=headline,
            regime=regime,
            wave=wave,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
    print(format_check(result))
