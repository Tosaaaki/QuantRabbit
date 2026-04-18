"""
QuantRabbit lesson registry

Builds a structured registry from strategy_memory.md so the trader runtime can
separate:
  - what was learned
  - what state the lesson is in
  - how much the lesson should be trusted right now
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

try:
    from schema import get_conn
except ModuleNotFoundError:
    from .schema import get_conn

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_MD = ROOT / "collab_trade" / "strategy_memory.md"
REGISTRY_JSON = Path(__file__).resolve().parent / "lesson_registry.json"
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
STATE_LABELS = ("candidate", "watch", "confirmed", "deprecated")


def extract_pair(text: str) -> str | None:
    for pair in PAIRS:
        if pair in text:
            return pair
    return None


def extract_direction(text: str) -> str | None:
    matches = {m.upper() for m in re.findall(r"\b(LONG|SHORT)\b", text, re.I)}
    if len(matches) == 1:
        return next(iter(matches))
    return None


def parse_state_marker(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    match = re.match(r"^-?\s*\[(CANDIDATE|WATCH|CONFIRMED|DEPRECATED)\]\s*(.*)$", stripped, re.I)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    if stripped.startswith("- "):
        return None, stripped[2:].strip()
    if stripped.startswith("-"):
        return None, stripped[1:].strip()
    return None, stripped


def strip_state_marker(text: str) -> str:
    _, core = parse_state_marker(text)
    return core


def split_markdown_sections(text: str, header_levels: tuple[str, ...]) -> list[str]:
    escaped = "|".join(re.escape(level) for level in header_levels)
    pattern = rf"\n(?=(?:{escaped})\s)"
    return [section.strip() for section in re.split(pattern, text) if section.strip()]


def extract_header(section: str) -> tuple[str, str] | None:
    match = re.match(r"(#{2,3})\s+(.+)", section)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def normalize_heading_text(text: str) -> str:
    cleaned = re.sub(r"（.*?）", "", text)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = cleaned.replace("##", "").replace("###", "")
    return " ".join(cleaned.strip().split())


def section_tag(label: str) -> str:
    normalized = normalize_heading_text(label).lower()
    replacements = {
        "⚡ read this first opportunity cost matters, but forced action kills expectancy too": "read_first",
        "confirmed patterns": "confirmed_pattern",
        "active observations": "active_observation",
        "deprecated": "deprecated",
        "per-pair learnings": "pair_learning",
        "pretrade feedback": "pretrade_feedback",
        "指標組み合わせの学び": "indicator_combo",
        "s-scan recipe scorecard": "s_scan_scorecard",
        "event day + thin market rules": "event_rule",
        "メンタル・行動": "mental",
    }
    return replacements.get(normalized, re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "strategy_memory")


def bullet_title(text: str) -> str:
    line = strip_state_marker(text)
    line = re.sub(r"^\[\d+(?:/\d+)?(?:[^\]]*)\]\s*", "", line)
    for sep in (":", "—", "->"):
        if sep in line:
            return line.split(sep, 1)[0].strip()
    return line[:120].strip()


def extract_strategy_update_date(text: str) -> date:
    match = re.search(r"最終更新:\s*(\d{4}-\d{2}-\d{2})", text)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return date.today()


def collect_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped)
    return bullets


def parse_verified_count(text: str, fallback_state: str) -> int:
    match = re.search(r"Verified:\s*(\d+)x", strip_state_marker(text), re.I)
    if match:
        return int(match.group(1))
    if fallback_state == "confirmed":
        return 3
    if fallback_state == "watch":
        return 2
    if fallback_state == "candidate":
        return 1
    return 0


def parse_observation_date(text: str, fallback: date) -> date:
    matches = re.findall(r"(\d{1,2})/(\d{1,2})", strip_state_marker(text))
    if not matches:
        return fallback
    resolved = []
    for month, day_value in matches:
        try:
            resolved.append(date(fallback.year, int(month), int(day_value)))
        except ValueError:
            continue
    return max(resolved) if resolved else fallback


def infer_state(top_tag: str) -> str:
    if top_tag in {"confirmed_pattern", "pair_learning", "event_rule", "mental"}:
        return "confirmed"
    if top_tag == "active_observation":
        return "candidate"
    if top_tag == "deprecated":
        return "deprecated"
    return "watch"


def infer_lesson_type(top_tag: str, pair: str | None) -> str:
    if top_tag in {"pretrade_feedback", "indicator_combo", "s_scan_scorecard"}:
        return "model_feedback"
    if top_tag in {"event_rule", "mental", "read_first"}:
        return "hygiene"
    if pair or top_tag == "pair_learning":
        return "pair_edge"
    if top_tag == "active_observation":
        return "hygiene"
    return "playbook"


def state_rank(state: str) -> int:
    return {
        "confirmed": 4,
        "watch": 3,
        "candidate": 2,
        "deprecated": 1,
    }.get(state, 0)


def trust_score(
    *,
    state: str,
    lesson_type: str,
    verified_count: int,
    lesson_date: date,
    pair: str | None,
    direction: str | None,
    trade_stats: dict | None,
) -> int:
    base = {
        "confirmed": 68,
        "watch": 52,
        "candidate": 34,
        "deprecated": 8,
    }.get(state, 0)
    type_boost = {
        "pair_edge": 8,
        "playbook": 6,
        "hygiene": 4,
        "model_feedback": 7,
    }.get(lesson_type, 0)
    score = base + type_boost + min(max(verified_count, 0), 6) * 4
    if pair and direction:
        score += 3
    elif pair:
        score += 1
    if trade_stats:
        trade_count = int(trade_stats.get("count", 0) or 0)
        pair_ev = float(trade_stats.get("ev", 0.0) or 0.0)
        win_rate = float(trade_stats.get("win_rate", 0.0) or 0.0)
        if trade_count >= 5 and pair_ev > 0:
            score += 4
        if trade_count >= 10 and pair_ev > 0:
            score += 4
        if trade_count >= 5 and pair_ev < 0:
            score -= 6
        if trade_count >= 10 and pair_ev < 0:
            score -= 4
        if trade_count >= 5 and win_rate >= 0.60:
            score += 3
        if trade_count >= 5 and win_rate < 0.40:
            score -= 4
    age_days = max((date.today() - lesson_date).days, 0)
    score -= min(age_days // 3, 12)
    return max(0, min(100, score))


def lesson_age_days(lesson_date: date) -> int:
    return max((date.today() - lesson_date).days, 0)


def suggest_state(
    *,
    state: str,
    lesson_type: str,
    verified_count: int,
    trust: int,
    lesson_date: date,
    trade_stats: dict | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    age_days = lesson_age_days(lesson_date)
    trade_count = int((trade_stats or {}).get("count", 0) or 0)
    pair_ev = float((trade_stats or {}).get("ev", 0.0) or 0.0)
    win_rate = float((trade_stats or {}).get("win_rate", 0.0) or 0.0)

    if state == "deprecated":
        return "deprecated", reasons

    if state == "candidate":
        if verified_count >= 3 and trust >= 68:
            reasons.append("verified>=3 with trust>=68")
            return "confirmed", reasons
        if verified_count >= 2 or trust >= 56:
            reasons.append("candidate matured past first observation")
            return "watch", reasons
        if age_days >= 10 and trust < 45:
            reasons.append("candidate stayed weak and stale for 10d+")
            return "deprecated", reasons
        return "candidate", reasons

    if state == "watch":
        if verified_count >= 3 and trust >= 70:
            reasons.append("watch lesson now has enough proof")
            return "confirmed", reasons
        if age_days >= 14 and trust < 48:
            reasons.append("watch lesson went stale without enough trust")
            return "deprecated", reasons
        return "watch", reasons

    if lesson_type == "pair_edge" and trade_count >= 10 and pair_ev < 0:
        reasons.append("pair-edge empirical EV is negative")
        return "watch", reasons
    if lesson_type == "pair_edge" and trade_count >= 10 and win_rate < 0.45:
        reasons.append("pair-edge empirical win rate is weak")
        return "watch", reasons
    if age_days >= 30 and trust < 55:
        reasons.append("confirmed lesson aged into low trust")
        return "watch", reasons
    return state, reasons


def load_trade_stats() -> dict[tuple[str | None, str | None], dict]:
    conn = get_conn()
    stats: dict[tuple[str | None, str | None], dict] = {}

    exact_rows = conn.execute(
        """SELECT pair, direction,
                  COUNT(*) AS cnt,
                  AVG(pl) AS ev,
                  SUM(pl) AS total_pl,
                  SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
           FROM trades
           WHERE pl IS NOT NULL
           GROUP BY pair, direction"""
    ).fetchall()
    for pair, direction, cnt, ev, total_pl, wins in exact_rows:
        stats[(pair, direction)] = {
            "count": int(cnt or 0),
            "ev": float(ev or 0.0),
            "total_pl": float(total_pl or 0.0),
            "wins": int(wins or 0),
            "win_rate": (float(wins or 0) / float(cnt or 1)) if cnt else 0.0,
        }

    pair_rows = conn.execute(
        """SELECT pair,
                  COUNT(*) AS cnt,
                  AVG(pl) AS ev,
                  SUM(pl) AS total_pl,
                  SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) AS wins
           FROM trades
           WHERE pl IS NOT NULL
           GROUP BY pair"""
    ).fetchall()
    for pair, cnt, ev, total_pl, wins in pair_rows:
        stats[(pair, None)] = {
            "count": int(cnt or 0),
            "ev": float(ev or 0.0),
            "total_pl": float(total_pl or 0.0),
            "wins": int(wins or 0),
            "win_rate": (float(wins or 0) / float(cnt or 1)) if cnt else 0.0,
        }
    return stats


def split_strategy_subsections(section_text: str) -> list[tuple[str | None, str]]:
    sections = split_markdown_sections(section_text, ("###",))
    if len(sections) == 1 and not sections[0].startswith("###"):
        return [(None, sections[0])]
    out: list[tuple[str | None, str]] = []
    for subsection in sections:
        header_info = extract_header(subsection)
        if header_info and header_info[0] == "###":
            out.append((header_info[1], subsection))
        else:
            out.append((None, subsection))
    return out


def build_registry() -> dict:
    if not STRATEGY_MD.exists():
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "strategy_updated": None,
            "lesson_count": 0,
            "by_state": {},
            "by_type": {},
            "lessons": [],
        }

    text = STRATEGY_MD.read_text()
    strategy_updated = extract_strategy_update_date(text)
    trade_stats = load_trade_stats()
    lessons: list[dict] = []

    sections = split_markdown_sections(text, ("##",))
    for section in sections:
        header_info = extract_header(section)
        if not header_info or header_info[0] != "##":
            continue
        section_name = header_info[1]
        top_tag = section_tag(section_name)
        state = infer_state(top_tag)

        for subsection_name, subsection_text in split_strategy_subsections(section):
            bullets = collect_bullets(subsection_text)
            if not bullets:
                continue
            for bullet in bullets:
                explicit_state, core_text = parse_state_marker(bullet)
                pair = extract_pair(bullet) or extract_pair(subsection_name or "")
                direction = extract_direction(bullet)
                lesson_type = infer_lesson_type(top_tag, pair)
                lesson_state = explicit_state or state
                verified_count = parse_verified_count(bullet, lesson_state)
                lesson_date = parse_observation_date(bullet, strategy_updated)
                stats_key = (pair, direction) if pair and direction else (pair, None)
                pair_stats = trade_stats.get(stats_key) if pair else None
                trust = trust_score(
                    state=lesson_state,
                    lesson_type=lesson_type,
                    verified_count=verified_count,
                    lesson_date=lesson_date,
                    pair=pair,
                    direction=direction,
                    trade_stats=pair_stats,
                )
                suggested_state, suggestion_reasons = suggest_state(
                    state=lesson_state,
                    lesson_type=lesson_type,
                    verified_count=verified_count,
                    trust=trust,
                    lesson_date=lesson_date,
                    trade_stats=pair_stats,
                )
                lessons.append({
                    "id": f"{top_tag}:{len(lessons) + 1}",
                    "title": bullet_title(bullet),
                    "text": f"- {core_text}",
                    "raw_text": bullet,
                    "core_text": core_text,
                    "state": lesson_state,
                    "state_source": "marker" if explicit_state else "section",
                    "state_rank": state_rank(lesson_state),
                    "lesson_type": lesson_type,
                    "section": normalize_heading_text(section_name),
                    "subsection": normalize_heading_text(subsection_name or ""),
                    "pair": pair,
                    "direction": direction,
                    "verified_count": verified_count,
                    "lesson_date": lesson_date.isoformat(),
                    "age_days": lesson_age_days(lesson_date),
                    "trust_score": trust,
                    "suggested_state": suggested_state,
                    "suggestion_reasons": suggestion_reasons,
                    "trade_stats": pair_stats,
                    "supports_pair_promotion": lesson_type == "pair_edge" and lesson_state != "deprecated",
                })

    by_state = Counter(item["state"] for item in lessons)
    by_type = Counter(item["lesson_type"] for item in lessons)
    by_suggested_state = Counter(item["suggested_state"] for item in lessons)
    review_queue = [
        item for item in lessons
        if item["suggested_state"] != item["state"]
    ]
    review_queue.sort(
        key=lambda item: (
            item["suggested_state"] == "confirmed",
            item["state"] == "confirmed",
            item["trust_score"],
            -item["age_days"],
            item["verified_count"],
        ),
        reverse=True,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "strategy_updated": strategy_updated.isoformat(),
        "lesson_count": len(lessons),
        "by_state": dict(sorted(by_state.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_suggested_state": dict(sorted(by_suggested_state.items())),
        "review_queue": review_queue[:20],
        "lessons": lessons,
    }


def refresh_registry() -> dict:
    registry = build_registry()
    REGISTRY_JSON.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    return registry


def load_registry() -> dict:
    if REGISTRY_JSON.exists():
        return json.loads(REGISTRY_JSON.read_text())
    return refresh_registry()


def stats() -> str:
    registry = load_registry()
    lines = [
        f"strategy_updated: {registry.get('strategy_updated')}",
        f"generated_at: {registry.get('generated_at')}",
        f"lesson_count: {registry.get('lesson_count', 0)}",
        "",
        "By state:",
    ]
    for key, value in sorted((registry.get("by_state") or {}).items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("By type:")
    for key, value in sorted((registry.get("by_type") or {}).items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("By suggested state:")
    for key, value in sorted((registry.get("by_suggested_state") or {}).items()):
        lines.append(f"  {key}: {value}")
    queue = registry.get("review_queue") or []
    if queue:
        lines.append("")
        lines.append("Top review queue:")
        for item in queue[:5]:
            lines.append(
                f"  {item['state']} -> {item['suggested_state']} | trust {item['trust_score']} | {item['title']}"
            )
    return "\n".join(lines)


def sync_strategy_memory_states() -> int:
    registry = refresh_registry()
    lessons = registry.get("lessons") or []
    pending = [item for item in lessons if item.get("suggested_state") != item.get("state")]
    if not pending:
        return 0

    review_map_by_raw = {item.get("raw_text"): item for item in pending if item.get("raw_text")}
    review_map_by_core = {item.get("core_text"): item for item in pending if item.get("core_text")}
    if not review_map_by_raw and not review_map_by_core:
        return 0

    lines = STRATEGY_MD.read_text().splitlines()
    changed = 0
    current_section_state: str | None = None

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            current_section_state = infer_state(section_tag(stripped[3:]))
            continue
        if not stripped.startswith("- "):
            continue
        explicit_state, core_text = parse_state_marker(stripped)
        review = review_map_by_raw.get(stripped) or review_map_by_core.get(core_text)
        if not review:
            continue
        target_state = review.get("suggested_state")
        if target_state not in STATE_LABELS:
            continue
        fallback_state = current_section_state or "candidate"
        if target_state == fallback_state:
            new_line = f"- {core_text}"
        else:
            new_line = f"- [{target_state.upper()}] {core_text}"
        if raw_line != new_line:
            lines[idx] = new_line
            changed += 1

    if changed:
        STRATEGY_MD.write_text("\n".join(lines) + "\n")
    return changed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print(stats())
    elif len(sys.argv) > 1 and sys.argv[1] == "sync":
        changed = sync_strategy_memory_states()
        print(f"updated {changed} bullet state markers in {STRATEGY_MD}")
    else:
        registry = refresh_registry()
        print(f"wrote {REGISTRY_JSON} ({registry.get('lesson_count', 0)} lessons)")
