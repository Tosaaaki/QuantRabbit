from __future__ import annotations

import json
import re
from typing import Any, Iterable

_EPHEMERAL_SUFFIX_PATTERNS = (
    re.compile(r"^(?P<base>.+?)-l[0-9a-f]{8,}$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+?)-[0-9a-f]{8,}$", re.IGNORECASE),
)

_PREFIX_ALIAS_PATTERNS = (
    (re.compile(r"^micropul[0-9a-f]{8,}$", re.IGNORECASE), "MicroPullbackEMA"),
    (re.compile(r"^microran[0-9a-f]{8,}$", re.IGNORECASE), "MicroRangeBreak"),
    (re.compile(r"^microtre[0-9a-f]{8,}$", re.IGNORECASE), "MicroTrendRetest-long"),
    (
        re.compile(r"^scalpmacdrsi[0-9a-f]{8,}$", re.IGNORECASE),
        "scalp_macd_rsi_div_b_live",
    ),
)

_EXACT_ALIASES = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
    "trendbreakout": "TrendBreakout",
    "m1scalper": "M1Scalper",
    "m1scalper_m1": "M1Scalper-M1",
    "m1scalper-m1": "M1Scalper-M1",
    "m1scalperm1": "M1Scalper-M1",
    "microlevelreactor": "MicroLevelReactor",
    "micropullbackema": "MicroPullbackEMA",
    "microrangebreak": "MicroRangeBreak",
    "microvwapbound": "MicroVWAPBound",
    "momentumburst": "MomentumBurst",
}


def normalize_strategy_lookup_key(raw: str | None) -> str:
    return "".join(ch.lower() for ch in str(raw or "") if ch.isalnum())


def strip_ephemeral_strategy_suffix(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    for pattern in _EPHEMERAL_SUFFIX_PATTERNS:
        matched = pattern.match(text)
        if matched:
            base = str(matched.group("base") or "").strip()
            if base:
                return base
    return text


def _known_values(known_keys: Iterable[str] | None) -> tuple[str, ...]:
    if not known_keys:
        return tuple()
    values: list[str] = []
    for item in known_keys:
        text = str(item or "").strip()
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _resolve_alias(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    alias = _EXACT_ALIASES.get(lower)
    if alias:
        return alias
    lookup = normalize_strategy_lookup_key(text)
    alias = _EXACT_ALIASES.get(lookup)
    if alias:
        return alias
    for pattern, target in _PREFIX_ALIAS_PATTERNS:
        if pattern.match(text):
            return target
    return text


def _match_known(text: str, known_keys: Iterable[str] | None) -> str:
    known = _known_values(known_keys)
    if not known:
        return text

    lowered = text.lower()
    lookup = normalize_strategy_lookup_key(text)
    exact_match = ""
    exact_lookup_match = ""
    prefix_match = ""
    prefix_lookup_match = ""
    prefix_lookup_len = -1

    for candidate in known:
        cand_lower = candidate.lower()
        cand_lookup = normalize_strategy_lookup_key(candidate)
        if lowered == cand_lower:
            exact_match = candidate
            break
        if lookup and lookup == cand_lookup:
            exact_lookup_match = candidate
        if lowered.startswith(cand_lower):
            next_char = lowered[len(cand_lower) : len(cand_lower) + 1]
            if not next_char or next_char in {"-", "_", "/", " "}:
                if len(candidate) > len(prefix_match):
                    prefix_match = candidate
        if lookup and cand_lookup and lookup.startswith(cand_lookup):
            if len(cand_lookup) > prefix_lookup_len:
                prefix_lookup_len = len(cand_lookup)
                prefix_lookup_match = candidate

    if exact_match:
        return exact_match
    if exact_lookup_match:
        return exact_lookup_match
    if prefix_match:
        return prefix_match
    if prefix_lookup_match:
        return prefix_lookup_match
    return text


def resolve_strategy_tag(
    raw: str | None, *, known_keys: Iterable[str] | None = None
) -> str:
    text = strip_ephemeral_strategy_suffix(raw)
    if not text:
        return ""
    aliased = _resolve_alias(text)
    return _match_known(aliased, known_keys)


def _parse_entry_thesis(entry_thesis: Any) -> dict[str, Any]:
    if isinstance(entry_thesis, dict):
        return entry_thesis
    if not isinstance(entry_thesis, str):
        return {}
    text = entry_thesis.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _prefer_lane_tag(raw: str) -> str:
    text = strip_ephemeral_strategy_suffix(raw)
    if not text:
        return ""
    resolved = resolve_strategy_tag(text)
    if not resolved:
        return text
    if resolved == text:
        return resolved
    if any(sep in text for sep in ("-", ":", "/", " ")):
        return text
    return resolved


def extract_strategy_tags(
    *,
    strategy_tag: str | None = None,
    strategy: str | None = None,
    entry_thesis: Any = None,
    known_keys: Iterable[str] | None = None,
) -> tuple[str, str]:
    thesis = _parse_entry_thesis(entry_thesis)
    raw_tag = ""
    for candidate in (
        thesis.get("strategy_tag_raw"),
        thesis.get("strategy_tag"),
        strategy_tag,
        thesis.get("strategy"),
        strategy,
    ):
        text = _prefer_lane_tag(str(candidate or "").strip())
        if text:
            raw_tag = text
            break

    canonical_tag = ""
    for candidate in (
        thesis.get("strategy"),
        strategy,
        thesis.get("strategy_tag"),
        strategy_tag,
        raw_tag,
    ):
        resolved = resolve_strategy_tag(
            str(candidate or "").strip(), known_keys=known_keys
        )
        if resolved:
            canonical_tag = resolved
            break

    if canonical_tag and not raw_tag:
        raw_tag = canonical_tag
    if raw_tag and not canonical_tag:
        canonical_tag = resolve_strategy_tag(raw_tag, known_keys=known_keys) or raw_tag
    return raw_tag, canonical_tag


def strategy_like_matches(
    strategy_tag: str | None,
    like_pattern: str | None,
    *,
    known_keys: Iterable[str] | None = None,
) -> bool:
    text = str(like_pattern or "").strip()
    if not text:
        return True
    escaped = re.escape(text)
    regex = re.compile(
        "^" + escaped.replace("%", ".*").replace("_", ".") + "$", re.IGNORECASE
    )
    candidates = (
        str(strategy_tag or "").strip(),
        strip_ephemeral_strategy_suffix(strategy_tag),
        resolve_strategy_tag(strategy_tag, known_keys=known_keys),
    )
    seen: set[str] = set()
    for candidate in candidates:
        current = str(candidate or "").strip()
        if not current or current in seen:
            continue
        seen.add(current)
        if regex.match(current):
            return True
    return False
