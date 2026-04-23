#!/usr/bin/env python3
"""
Validate trader state handoff before SESSION_END.

The goal is not to force specific market opinions. The goal is to block
prose-only flat books such as:
  - "armed mentally only"
  - "retest only"
  - "none placed"
  - flat-book summaries with zero real order receipts

Usage:
  python3 tools/validate_trader_state.py
  python3 tools/validate_trader_state.py /path/to/state.md
  python3 tools/validate_trader_state.py --require-live-oanda /path/to/state.md
  python3 tools/validate_trader_state.py --require-live-book-coverage /path/to/state.md
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_loader import get_oanda_config
from record_s_hunt_ledger import STATE_PATH, _extract_section, _parse_simple_line, build_entry
from slack_read import BOT_USER_ID, read_messages
from trader_order_guard import (
    ENTRY_PENDING_TYPES,
    exact_pretrade_issues,
    requested_style_from_order_type,
    run_exact_pretrade,
)


REQUIRED_HORIZONS = ("Short-term S", "Medium-term S", "Long-term S")
REQUIRED_EXCAVATION_PAIRS = (
    "USD_JPY",
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
)
EXCAVATION_REQUIRED_FROM = "2026-04-18"
PLACEHOLDER_TOKEN = "___"
PAIR_DIRECTION_RE = re.compile(r"\b([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b")
ID_RE = re.compile(r"id=\s*`?(\d+)`?")
BAD_PROSE = (
    "armed mentally",
    "retest only",
    "breakout only",
    "none placed",
)
EXECUTION_ONLY_BLOCKER_PHRASES = (
    "no live pending entry order exists",
    "no pending entry order exists",
    "no live pending id",
    "board lane, not a live",
    "never earned a real stop-entry id",
    "never earned a stop receipt",
    "never earned a real pending id",
)
PENDING_WAITING_BLOCKER_PHRASES = (
    "has not broken",
    "hasn't broken",
    "not broken yet",
    "has not printed",
    "not printed yet",
    "not triggered",
    "trigger has not",
    "still needs trigger",
    "waiting for trigger",
    "needs price improvement",
    "price-improvement only",
    "price improvement only",
    "has not traded there yet",
    "not there yet",
    "one print away",
    "waiting for retest",
)
PENDING_REAL_CONTRADICTION_TOKENS = (
    "acceptance",
    "accepted",
    "body close above",
    "body close below",
    "invalidation",
    "contradict",
    "rotated",
    "not worth real risk",
    "quality bar",
    "pmi risk",
    "macro risk",
    "theme flipped",
    "edge collapsed",
)
PROMOTION_GATE_PHRASE = "no seat cleared promotion gate"
RECEIPT_CLOSED_RE = re.compile(r"receipt\s+id=\s*`?\d+`?\s+is\s+closed\b", re.I)
MANDATE_LABELS = (
    "Best A/S live now",
    "Best A/S one print away",
    "Best A/S I am explicitly rejecting",
)
ARMABLE_ACTIONS = {"LIMIT", "STOP-ENTRY"}
LIVE_ACTIONS = {"ENTER NOW", "MARKET", "LIMIT", "STOP-ENTRY"}
LIVE_RECEIPT_SECTIONS = (
    "A/S Excavation Mandate",
    "S Hunt",
    "Multi-Vehicle Deployment",
    "Multi-Vehicle / Pending",
    "Pending LIMITs",
    "Capital Deployment",
    "Positions (Current)",
)
MIN_PODIUM_ROWS = 3
MIN_GOLD_ROWS = 5
LANE_LINE_RE = re.compile(r"^(Lane\s+(\d+)\s*/\s*[^:]+):\s*(.+)$")
PODIUM_LINE_RE = re.compile(r"^Podium #(\d+):\s*(.+)$", re.I)
GOLD_LINE_RE = re.compile(r"^Gold #(\d+):\s*(.+)$", re.I)
LANE_BAD_PROSE = (
    "trigger-only",
    "watch lane",
    "watch-only",
)
TRADER_LOCK_PATH = STATE_PATH.parent.parent / "logs" / ".trader_lock"
ACTION_BOARD_PATH = STATE_PATH.parent.parent / "logs" / "session_action_board.json"
ACTION_BOARD_MAX_AGE_MIN = 45
SLACK_LAST_READ_TS_PATH = STATE_PATH.parent.parent / "logs" / ".slack_last_read_ts"
SLACK_COMMANDS_CHANNEL = "C0APAELAQDN"
SLACK_REQUIRED_FIELDS = (
    "Pending user ts",
    "Latest handled user ts",
    "Message class",
    "Trade consequence",
    "Reply receipt",
)
ENTRY_ACTIVITY_REQUIRED_TOKENS = ("fills /", "new entry orders", "rejects")
COUNTER_ORDER_COMMENT_TOKENS = ("counter", "reversal", "mean_revert", "mean-revert")
UNIQUE_HANDOFF_SECTIONS = (
    "Self-check",
    "Slack Response",
    "Market Narrative",
    "Positions (Current)",
    "OODA / Decision Journal",
    "Deepening Pass",
    "Directional Mix",
    "7-Pair Scan",
    "S Excavation Matrix",
    "A/S Excavation Mandate",
    "S Hunt",
    "Multi-Vehicle Deployment",
    "Pending LIMITs (freshness review)",
    "Capital Deployment",
    "Action Tracking",
    "Lessons (Recent)",
)


def _has_id(value: str | None) -> bool:
    return bool(value and "id=" in value.lower())


def _is_dead(value: str | None) -> bool:
    return bool(value and "dead" in value.lower())


def _contains_bad_prose(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    for phrase in BAD_PROSE:
        if phrase in lowered:
            return phrase
    return None


def _contains_lane_bad_prose(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    for phrase in LANE_BAD_PROSE:
        if phrase in lowered:
            if (
                _is_dead(value)
                and "because" in lowered
                and any(
                    token in lowered
                    for token in (
                        "acceptance",
                        "accepted",
                        "failed",
                        "closed",
                        "no-edge",
                        "friction",
                        "spread",
                        "exact guard",
                    )
                )
            ):
                continue
            return phrase
    return _contains_bad_prose(value)


def _execution_only_blocker_phrase(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    lowered = normalized.lower().strip(" .")
    if not lowered:
        return None

    candidates: list[str] = []
    if lowered == PROMOTION_GATE_PHRASE:
        return PROMOTION_GATE_PHRASE
    if lowered.startswith(PROMOTION_GATE_PHRASE):
        remainder = normalized[len(PROMOTION_GATE_PHRASE):].strip(" .:;,-—")
        if not remainder:
            return PROMOTION_GATE_PHRASE
        candidates.append(_normalize_text(remainder).lower())
    candidates.append(lowered)

    for candidate in candidates:
        for phrase in EXECUTION_ONLY_BLOCKER_PHRASES:
            if phrase in candidate:
                return phrase
        if RECEIPT_CLOSED_RE.search(candidate):
            return "receipt id=... is closed"
    return None


def _dead_thesis_reason(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    lowered = normalized.lower()
    marker = "dead thesis because"
    if marker not in lowered:
        return None
    start = lowered.index(marker) + len(marker)
    return normalized[start:].strip(" .")


def _tautological_dead_closure_phrase(value: str | None) -> str | None:
    reason = _dead_thesis_reason(value)
    if not reason:
        return None
    return _execution_only_blocker_phrase(reason)


def _pending_lane_waiting_phrase(value: str | None, lane: dict | None) -> str | None:
    if _snapshot_lane_action(lane) not in ARMABLE_ACTIONS:
        return None
    reason = _dead_thesis_reason(value)
    if not reason:
        return None
    lowered = _normalize_text(reason).lower()
    if any(token in lowered for token in PENDING_REAL_CONTRADICTION_TOKENS):
        return None
    for phrase in PENDING_WAITING_BLOCKER_PHRASES:
        if phrase in lowered:
            return phrase
    if "trigger" in lowered and "yet" in lowered:
        return "trigger not printed yet"
    if "price" in lowered and "yet" in lowered:
        return "price improvement not traded yet"
    return None


def _extract_pipe_field(line: str, label: str) -> str | None:
    for part in line.split("|"):
        stripped = part.strip()
        for candidate in (stripped, stripped.split(":", 1)[1].strip() if ":" in stripped else None):
            if not candidate:
                continue
            if candidate.lower().startswith(label.lower()):
                value = candidate[len(label):].strip(" :")
                return value or None
    return None


def _extract_block_field(text: str, label: str, stop_labels: tuple[str, ...]) -> str | None:
    if stop_labels:
        stop_expr = "|".join(re.escape(item) for item in stop_labels)
        pattern = rf"{re.escape(label)}\s*(.*?)(?=(?:{stop_expr})|$)"
    else:
        pattern = rf"{re.escape(label)}\s*(.*)$"
    match = re.search(pattern, text, re.S)
    return match.group(1).strip(" .") if match else None


def _extract_labeled_block(section: str, label: str, stop_labels: tuple[str, ...]) -> str | None:
    if not section:
        return None

    lines = section.splitlines()
    capture = False
    block_lines: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if capture:
                block_lines.append("")
            continue

        if any(stripped.startswith(f"{candidate}:") for candidate in stop_labels):
            if capture:
                break

        if stripped.startswith(f"{label}:"):
            capture = True
            block_lines = [stripped]
            continue

        if capture:
            block_lines.append(stripped)

    cleaned = "\n".join(line for line in block_lines if line)
    return cleaned or None


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("`", "").split())


def _parse_mandate_seat(block: str | None, action_label: str) -> dict[str, str | None]:
    if not block:
        return {"pair": None, "direction": None, "action": None, "raw": None}
    pair_match = PAIR_DIRECTION_RE.search(block)
    action = _extract_block_field(block, f"{action_label}:", tuple())
    return {
        "pair": pair_match.group(1) if pair_match else None,
        "direction": pair_match.group(2) if pair_match else None,
        "action": action.strip() if action else None,
        "raw": block,
    }


def _parse_execution_count(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    match = re.search(r"(\d+)\s+live receipts?\s*\|\s*(\d+)\s+armed receipts?", value, re.I)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _is_noneish(value: str | None) -> bool:
    normalized = _normalize_text(value).lower()
    if not normalized:
        return True
    return (
        normalized in {"none", "(none)"}
        or normalized.startswith("none ")
        or normalized.startswith("none.")
        or normalized.startswith("none;")
        or normalized.startswith("none because")
    )


def _pair_direction_key(value: str | None) -> tuple[str, str] | None:
    match = PAIR_DIRECTION_RE.search(value or "")
    if not match:
        return None
    return match.group(1), match.group(2)


def _dead_line_allows_carry(value: str | None) -> bool:
    normalized = _normalize_text(value).lower()
    return (
        "this cadence" in normalized
        or "next-session board seat" in normalized
        or "next session board seat" in normalized
        or "no exact pending id survived" in normalized
    )


def _validate_unique_sections(text: str) -> list[str]:
    errors: list[str] = []
    for name in UNIQUE_HANDOFF_SECTIONS:
        count = len(re.findall(rf"^## {re.escape(name)}\s*$", text, re.M))
        if count > 1:
            errors.append(
                f"`state.md` contains {count} `## {name}` sections. Rewrite the existing block in place; "
                "the parser only reads one of them, so later fixes can stay invisible."
            )
    return errors


def _load_last_handled_slack_ts() -> str | None:
    if not SLACK_LAST_READ_TS_PATH.exists():
        return None
    value = SLACK_LAST_READ_TS_PATH.read_text().strip()
    return value or None


def _load_pending_slack_user_messages(limit: int = 20) -> list[dict]:
    after = _load_last_handled_slack_ts()
    messages = read_messages(channel_id=SLACK_COMMANDS_CHANNEL, limit=limit, after=after)
    pending = [
        msg
        for msg in messages
        if msg.get("user") != BOT_USER_ID
        and "bot_id" not in msg
        and msg.get("subtype") != "bot_message"
    ]
    pending.sort(key=lambda msg: float(str(msg.get("ts") or "0") or "0"))
    return pending


def _validate_live_slack_response(text: str) -> list[str]:
    section = _extract_section(text, "Slack Response")
    if not section:
        return [
            "Missing `## Slack Response` block. Close the latest Slack handling explicitly before `SESSION_END`."
        ]

    values = {label: _parse_simple_line(section, label) for label in SLACK_REQUIRED_FIELDS}
    errors: list[str] = []
    for label, value in values.items():
        if not value:
            errors.append(f"`Slack Response` is missing `{label}`.")
        elif PLACEHOLDER_TOKEN in value:
            errors.append(f"`Slack Response` `{label}` still contains placeholder blanks.")

    pending_line = values.get("Pending user ts")
    handled_line = values.get("Latest handled user ts")
    reply_receipt = values.get("Reply receipt")

    try:
        runtime_last_handled = _load_last_handled_slack_ts()
        pending_messages = _load_pending_slack_user_messages()
    except Exception as exc:
        return [f"Live Slack reply verification failed: {exc}"]

    if runtime_last_handled:
        if not handled_line or runtime_last_handled not in handled_line:
            errors.append(
                "`Slack Response` must mirror the runtime-handled user ts "
                f"`{runtime_last_handled}` from `logs/.slack_last_read_ts`."
            )
    elif handled_line and not _is_noneish(handled_line):
        errors.append(
            "`Slack Response` names a handled user ts, but the runtime has no handled Slack ts yet. "
            "Use `none` until a real `--reply-to` receipt exists."
        )

    if pending_messages:
        latest = pending_messages[-1]
        latest_ts = str(latest.get("ts") or "").strip()
        latest_text = _normalize_text(str(latest.get("text") or ""))[:120]
        if not pending_line or latest_ts not in pending_line:
            errors.append(
                f"`Slack Response` must name the latest pending user ts=`{latest_ts}` until it is replied."
            )
        errors.append(
            "Unread Slack user message still exists: "
            f"ts=`{latest_ts}` text=`{latest_text}`. Reply before trade decisions and before `SESSION_END`."
        )
        if reply_receipt and "--reply-to" in reply_receipt and latest_ts not in reply_receipt:
            errors.append(
                f"`Slack Response` `Reply receipt` must use `--reply-to {latest_ts}` for the pending user message."
            )
    else:
        if pending_line and not _is_noneish(pending_line):
            errors.append(
                "`Slack Response` still names a pending user ts even though Slack has no unread human messages. "
                "Reset it to `none` after the reply is sent."
            )
        if runtime_last_handled and reply_receipt and not _is_noneish(reply_receipt):
            if "--reply-to" not in reply_receipt or runtime_last_handled not in reply_receipt:
                errors.append(
                    "`Slack Response` `Reply receipt` must either say "
                    "`none because no unread user message` or cite the real "
                    f"`slack_post.py --reply-to {runtime_last_handled}` receipt."
                )

    return errors


def _load_recent_action_board_snapshot() -> dict | None:
    if not TRADER_LOCK_PATH.exists() or not ACTION_BOARD_PATH.exists():
        return None
    try:
        payload = json.loads(ACTION_BOARD_PATH.read_text())
    except Exception:
        return None
    generated_at = str(payload.get("generated_at_utc") or "").strip()
    if not generated_at:
        return None
    try:
        generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - generated.astimezone(timezone.utc)
    if age > timedelta(minutes=ACTION_BOARD_MAX_AGE_MIN):
        return None
    payload["_generated_at"] = generated
    return payload


def _snapshot_lane_action(lane: dict | None) -> str | None:
    if not lane:
        return None
    return _normalized_action(
        str(
            lane.get("default_orderability")
            or lane.get("orderability")
            or lane.get("default_expression")
            or lane.get("execution_style")
            or ""
        ).strip()
    )


def _state_execution_counts(text: str) -> tuple[int, int]:
    capital_section = _extract_section(text, "Capital Deployment")
    execution_line = _parse_simple_line(capital_section, "Execution count this session")
    live_count, armed_count = _parse_execution_count(execution_line)
    return int(live_count or 0), int(armed_count or 0)


def _snapshot_lane_has_matching_receipt(text: str, lane: dict | None) -> bool:
    if not lane:
        return False
    live_sections = [
        _extract_section(text, section_name)
        for section_name in LIVE_RECEIPT_SECTIONS
    ]
    live_text = "\n".join(section for section in live_sections if section)
    seat = {
        "pair": str(lane.get("pair") or "") or None,
        "direction": str(lane.get("direction") or "") or None,
        "raw": f"{lane.get('pair', '')} {lane.get('direction', '')}",
    }
    for action in ("ENTER NOW", "MARKET", "LIMIT", "STOP-ENTRY"):
        if _has_matching_receipt(live_text, seat, action):
            return True
    return False


def _snapshot_lane_has_matching_dead_closure(text: str, lane: dict | None) -> bool:
    if not lane:
        return False
    pair = str(lane.get("pair") or "").strip().upper()
    direction = str(lane.get("direction") or "").strip().upper()
    if not pair or not direction:
        return False
    for raw in text.splitlines():
        line = _normalize_text(raw).upper()
        if pair in line and direction in line and "DEAD THESIS BECAUSE" in line:
            if _tautological_dead_closure_phrase(raw) or _pending_lane_waiting_phrase(raw, lane):
                return False
            return True
    return False


def _snapshot_lane_label(lane: dict | None) -> str:
    if not lane:
        return "UNKNOWN"
    pair = str(lane.get("pair") or "UNKNOWN").strip()
    direction = str(lane.get("direction") or "UNKNOWN").strip()
    action = (
        _snapshot_lane_action(lane)
        or str(lane.get("default_expression") or lane.get("execution_style") or "UNKNOWN").strip()
        or "UNKNOWN"
    )
    source = str(lane.get("source") or "").strip()
    if source:
        return f"{pair} {direction} {action} [{source}]"
    return f"{pair} {direction} {action}"


def _line_mentions_lane(line: str | None, lane: dict | None) -> bool:
    if not line or not lane:
        return False
    pair = str(lane.get("pair") or "").strip().upper()
    direction = str(lane.get("direction") or "").strip().upper()
    normalized = _normalize_text(line).upper()
    return bool(pair and direction and pair in normalized and direction in normalized)


def _validate_focus_ladder_line(
    line: str | None,
    lane: dict | None,
    *,
    label: str,
) -> list[str]:
    if not lane:
        return []
    if not line:
        return [f"`Market Narrative` is missing `{label}` while the latest action board still has `{_snapshot_lane_label(lane)}`."]
    normalized = _normalize_text(line).lower()
    if _line_mentions_lane(line, lane):
        tautology = _tautological_dead_closure_phrase(line)
        if tautology:
            return [
                f"`{label}` closes `{_snapshot_lane_label(lane)}` only because `{tautology}`. "
                "Receipt absence is not a market contradiction; either arm the lane or write the actual tape / structure blocker."
            ]
        waiting_phrase = _pending_lane_waiting_phrase(line, lane)
        if waiting_phrase:
            return [
                f"`{label}` kills `{_snapshot_lane_label(lane)}` only because `{waiting_phrase}`. "
                "For `LIMIT` / `STOP-ENTRY`, an untraded trigger or unfilled price is why the order should be armed, not why the thesis is dead."
            ]
        return []
    if "none" in normalized:
        return [
            f"`{label}` still says `none` while the latest action board has `{_snapshot_lane_label(lane)}`. "
            "Name the lane directly there or close it there as `dead thesis because ...`."
        ]
    return [
        f"`{label}` does not close the latest action-board lane `{_snapshot_lane_label(lane)}`. "
        "Do not leave the focus ladder stale while only the lane lines are updated."
    ]


def _iter_snapshot_actionable_lanes(snapshot: dict | None, key: str) -> list[dict]:
    if not snapshot:
        return []
    lanes: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for raw_lane in snapshot.get(key) or []:
        if not isinstance(raw_lane, dict):
            continue
        lane = dict(raw_lane)
        pair = str(lane.get("pair") or "").strip().upper()
        direction = str(lane.get("direction") or "").strip().upper()
        action = _snapshot_lane_action(lane)
        if not pair or not direction or action not in LIVE_ACTIONS:
            continue
        dedupe_key = _snapshot_lane_identity(lane)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        lane["pair"] = pair
        lane["direction"] = direction
        lanes.append(lane)
    return lanes


def _snapshot_lane_identity(lane: dict | None) -> tuple[str, ...]:
    if not lane:
        return ("UNKNOWN",)
    seat_key = str(lane.get("seat_key") or "").strip()
    if seat_key:
        return ("seat_key", seat_key)
    return (
        "lane",
        str(lane.get("pair") or "").strip().upper(),
        str(lane.get("direction") or "").strip().upper(),
        str(_snapshot_lane_action(lane) or "").strip().upper(),
        str(lane.get("source") or "").strip().lower(),
    )


def _merge_snapshot_lanes(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        for lane in group:
            identity = _snapshot_lane_identity(lane)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(lane)
    return merged


def _validate_recent_action_board(text: str) -> list[str]:
    snapshot = _load_recent_action_board_snapshot()
    if not snapshot:
        return []

    session_intent = snapshot.get("session_intent") or {}
    session_mode = str(session_intent.get("mode") or "").strip().upper()
    live_count, armed_count = _state_execution_counts(text)
    market_now_lanes = _iter_snapshot_actionable_lanes(snapshot, "market_now")
    multi_vehicle_lanes = _iter_snapshot_actionable_lanes(snapshot, "multi_vehicle_lanes")

    def unresolved(lanes: list[dict]) -> list[dict]:
        return [
            lane
            for lane in lanes
            if not _snapshot_lane_has_matching_receipt(text, lane)
            and not _snapshot_lane_has_matching_dead_closure(text, lane)
        ]

    def tautological_dead_closures(lanes: list[dict]) -> list[tuple[dict, str]]:
        matches: list[tuple[dict, str]] = []
        for lane in lanes:
            pair = str(lane.get("pair") or "").strip().upper()
            direction = str(lane.get("direction") or "").strip().upper()
            if not pair or not direction:
                continue
            for raw in text.splitlines():
                normalized = _normalize_text(raw).upper()
                if pair not in normalized or direction not in normalized or "DEAD THESIS BECAUSE" not in normalized:
                    continue
                phrase = _tautological_dead_closure_phrase(raw)
                if phrase:
                    matches.append((lane, phrase))
                    break
        return matches

    def pending_wait_dead_closures(lanes: list[dict]) -> list[tuple[dict, str]]:
        matches: list[tuple[dict, str]] = []
        for lane in lanes:
            pair = str(lane.get("pair") or "").strip().upper()
            direction = str(lane.get("direction") or "").strip().upper()
            if not pair or not direction:
                continue
            for raw in text.splitlines():
                normalized = _normalize_text(raw).upper()
                if pair not in normalized or direction not in normalized or "DEAD THESIS BECAUSE" not in normalized:
                    continue
                phrase = _pending_lane_waiting_phrase(raw, lane)
                if phrase:
                    matches.append((lane, phrase))
                    break
        return matches

    errors: list[str] = []
    invalid_market = tautological_dead_closures(market_now_lanes)
    invalid_multi = tautological_dead_closures(multi_vehicle_lanes)
    invalid_pending_wait = pending_wait_dead_closures(multi_vehicle_lanes)
    unresolved_market = unresolved(market_now_lanes)
    unresolved_multi = unresolved(multi_vehicle_lanes)
    actionable = market_now_lanes or (multi_vehicle_lanes if session_mode != "WATCH-ONLY" else [])
    lane_lines = _extract_lane_lines(text)
    market_section = _extract_section(text, "Market Narrative")
    backup_focus = _parse_simple_line(market_section, "Backup vehicle")
    next_risk_focus = _parse_simple_line(market_section, "Next fresh risk allowed NOW")

    if invalid_market:
        lane_list = ", ".join(
            f"{_snapshot_lane_label(lane)} (`{phrase}`)"
            for lane, phrase in invalid_market[:3]
        )
        errors.append(
            "Latest session action board still had payable lane(s) "
            f"{lane_list}, but the handoff killed them with receipt-free dead-thesis prose. "
            "That is execution drift, not a real contradiction."
        )

    if invalid_multi:
        lane_list = ", ".join(
            f"{_snapshot_lane_label(lane)} (`{phrase}`)"
            for lane, phrase in invalid_multi[:3]
        )
        errors.append(
            "Latest session action board still had armable lane(s) "
            f"{lane_list}, but the handoff closed them only because no receipt existed yet. "
            "An armable lane must be armed or contradicted by tape / structure, not by missing execution."
        )

    if invalid_pending_wait:
        lane_list = ", ".join(
            f"{_snapshot_lane_label(lane)} (`{phrase}`)"
            for lane, phrase in invalid_pending_wait[:3]
        )
        errors.append(
            "Latest session action board still had pending-style lane(s) "
            f"{lane_list}, but the handoff killed them only because the trigger / better price had not printed yet. "
            "For `LIMIT` / `STOP-ENTRY`, waiting conditions are the reason to arm the order, not thesis death."
        )

    if unresolved_market:
        lane_list = ", ".join(_snapshot_lane_label(lane) for lane in unresolved_market[:3])
        errors.append(
            "Latest session action board still had payable lane(s) "
            f"{lane_list}, but the handoff never resolved them as a real `id=` receipt or explicit dead thesis."
        )

    if session_mode != "WATCH-ONLY" and armed_count == 0 and unresolved_multi:
        lane_list = ", ".join(_snapshot_lane_label(lane) for lane in unresolved_multi[:3])
        errors.append(
            "Latest session action board still had armable lane(s) "
            f"{lane_list}, but `Capital Deployment` closed with `0 armed receipts` and no explicit dead-thesis close."
        )

    if actionable and (live_count + armed_count) == 0:
        unresolved_all = unresolved(actionable)
        if unresolved_all:
            lane_list = ", ".join(_snapshot_lane_label(lane) for lane in unresolved_all[:3])
            errors.append(
                "Latest session action board named actionable lane(s) "
                f"{lane_list}, but `state.md` still closed flat with `0 live receipts | 0 armed receipts`."
            )

    ordered_lane_items = list(lane_lines.items())
    for idx, lane in enumerate(multi_vehicle_lanes):
        if idx >= len(ordered_lane_items):
            errors.append(
                f"`Capital Deployment` is missing a lane line for the latest action-board lane "
                f"`{_snapshot_lane_label(lane)}`. Add `Lane {idx + 1} / ...` and close it there."
            )
            continue
        label, line = ordered_lane_items[idx]
        if not line:
            continue
        tautology = _tautological_dead_closure_phrase(line)
        if _line_mentions_lane(line, lane):
            if tautology:
                errors.append(
                    f"`{label}` closes the latest action-board lane `{_snapshot_lane_label(lane)}` only because "
                    f"`{tautology}`. Receipt absence is not valid lane death."
                )
            waiting_phrase = _pending_lane_waiting_phrase(line, lane)
            if waiting_phrase:
                errors.append(
                    f"`{label}` kills the latest action-board lane `{_snapshot_lane_label(lane)}` only because "
                    f"`{waiting_phrase}`. For `LIMIT` / `STOP-ENTRY`, that is why the order should be armed."
                )
            continue
        if "none" in _normalize_text(line).lower() or _contains_lane_bad_prose(line):
            errors.append(
                f"`{label}` still hides the latest action-board lane `{_snapshot_lane_label(lane)}` behind prose "
                "or `none`. Close that lane explicitly in the lane line as a receipt or `dead thesis because ...`."
            )
        elif _snapshot_lane_has_matching_receipt(text, lane) or _snapshot_lane_has_matching_dead_closure(text, lane):
            errors.append(
                f"`{label}` does not mention the current action-board lane `{_snapshot_lane_label(lane)}`. "
                "Do not resolve the lane only elsewhere in the handoff; close it in the lane line itself."
            )

    next_risk_lane = market_now_lanes[0] if market_now_lanes else None
    if next_risk_lane:
        errors.extend(
            _validate_focus_ladder_line(
                next_risk_focus,
                next_risk_lane,
                label="Next fresh risk allowed NOW",
            )
        )

    backup_lane = next(
        (
            lane for lane in multi_vehicle_lanes
            if not next_risk_lane
            or (
                str(lane.get("pair") or "").upper(),
                str(lane.get("direction") or "").upper(),
            ) != (
                str(next_risk_lane.get("pair") or "").upper(),
                str(next_risk_lane.get("direction") or "").upper(),
            )
        ),
        None,
    )
    if backup_lane:
        errors.extend(
            _validate_focus_ladder_line(
                backup_focus,
                backup_lane,
                label="Backup vehicle",
            )
        )

    return errors


def _lane_label_order(label: str) -> int:
    match = re.match(r"Lane\s+(\d+)\s*/", label, re.I)
    return int(match.group(1)) if match else 999


def _extract_lane_lines(text: str) -> dict[str, str]:
    lines: dict[str, str] = {}
    for section_name in ("Capital Deployment", "Multi-Vehicle / Pending", "Multi-Vehicle Deployment"):
        section = _extract_section(text, section_name)
        if not section:
            continue
        for raw in section.splitlines():
            stripped = raw.strip()
            match = LANE_LINE_RE.match(stripped)
            if not match:
                continue
            label = match.group(1)
            if label not in lines:
                lines[label] = match.group(3).strip()
    return dict(sorted(lines.items(), key=lambda item: _lane_label_order(item[0])))


def _normalized_action(action: str | None) -> str | None:
    if not action:
        return None
    upper = _normalize_text(action).upper()
    if "DEAD THESIS" in upper:
        return "DEAD"
    for candidate in ("STOP-ENTRY", "LIMIT", "ENTER NOW", "MARKET"):
        if candidate in upper:
            return candidate
    return None


def _line_has_receipt(line: str, pair: str, direction: str, action: str) -> bool:
    normalized = _normalize_text(line).upper()
    if pair not in normalized or direction not in normalized or "ID=" not in normalized:
        return False

    if action == "LIMIT":
        return "LIMIT" in normalized
    if action == "STOP-ENTRY":
        return "STOP-ENTRY" in normalized or "ARMED STOP" in normalized or " STOP " in normalized
    if action in {"ENTER NOW", "MARKET"}:
        if "ARMED LIMIT" in normalized or "ARMED STOP" in normalized or "STOP-ENTRY" in normalized:
            return False
        return True
    return False


def _line_has_pending_receipt(line: str, pair: str, action: str) -> bool:
    normalized = _normalize_text(line).upper()
    if pair not in normalized or "ID=" not in normalized:
        return False
    if action == "LIMIT":
        return "LIMIT" in normalized
    if action == "STOP-ENTRY":
        return "STOP-ENTRY" in normalized or " STOP " in normalized
    return False


def _has_matching_receipt(text: str, seat: dict[str, str | None], action: str) -> bool:
    pair = seat.get("pair")
    direction = seat.get("direction")
    raw = seat.get("raw") or ""
    if not pair or not direction:
        return False

    raw_normalized = _normalize_text(raw).upper()
    if "ID=" in raw_normalized:
        if action == "LIMIT" and "LIMIT" in raw_normalized:
            return True
        if action == "STOP-ENTRY" and ("STOP-ENTRY" in raw_normalized or "ARMED STOP" in raw_normalized):
            return True
        if action in {"ENTER NOW", "MARKET"} and "DEAD THESIS" not in raw_normalized:
            return True

    for line in text.splitlines():
        if _line_has_receipt(line, pair, direction, action):
            return True
        if action in ARMABLE_ACTIONS and _line_has_pending_receipt(line, pair, action):
            return True
    return False


def _pending_section_says_none(section: str) -> bool:
    lowered = section.lower()
    return (
        "no discretionary entry limit" in lowered
        or "no fresh entry orders are live" in lowered
        or "none." in lowered
        or lowered.strip() == "none"
    )


def _oanda_api(path: str, cfg: dict[str, object]) -> dict:
    url = f"{cfg['oanda_base_url']}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _collect_live_receipt_claims(text: str) -> dict[tuple[str, str], set[str]]:
    claims: dict[tuple[str, str], set[str]] = {}
    for section_name in LIVE_RECEIPT_SECTIONS:
        section = _extract_section(text, section_name)
        if not section:
            continue
        for raw in section.splitlines():
            line = raw.strip()
            if "id=" not in line:
                continue
            ids = ID_RE.findall(line)
            if not ids:
                continue

            normalized = _normalize_text(line).upper()
            if any(
                token in normalized
                for token in (
                    "CANCELLED",
                    "CANCELED",
                    "CANCEL_ORDER",
                    "CLOSED",
                    " CLOSE ",
                    "TAKE_PROFIT",
                    "TAKE PROFIT",
                    "STOP_LOSS",
                    "TP CAPTURED",
                    "CAPTURED TP",
                    "NO LIVE RECEIPT",
                )
            ):
                continue
            kind = None
            if "ARMED LIMIT" in normalized or "ARMED STOP" in normalized:
                kind = "pending"
            elif "TRADE ID=" in normalized or "ENTERED ID=" in normalized:
                kind = "trade"
            elif section_name == "Pending LIMITs":
                kind = "pending"
            elif line.startswith("### "):
                kind = "trade"
            elif normalized.startswith("LIVE NOW:"):
                kind = "trade"
            elif normalized.startswith("RELOAD:") or normalized.startswith("SECOND SHOT / OTHER SIDE:"):
                kind = "pending"

            if kind is None:
                continue

            for receipt_id in ids:
                claims.setdefault((kind, receipt_id), set()).add(section_name)
    return claims


def _validate_handoff_consistency(text: str) -> list[str]:
    market_section = _extract_section(text, "Market Narrative")
    positions_section = _extract_section(text, "Positions (Current)")
    capital_section = _extract_section(text, "Capital Deployment")

    pending_orders_line = _parse_simple_line(positions_section, "Pending orders")
    armed_backup_line = _parse_simple_line(capital_section, "Armed backup lane for this cadence")
    backup_trigger_line = _parse_simple_line(market_section, "20-minute backup trigger armed NOW")
    execution_line = _parse_simple_line(capital_section, "Execution count this session")
    _live_count, armed_count = _parse_execution_count(execution_line)

    claims = _collect_live_receipt_claims(text)
    pending_ids = sorted({receipt_id for (kind, receipt_id), _sections in claims.items() if kind == "pending"})
    pending_preview = ", ".join(f"id=`{receipt_id}`" for receipt_id in pending_ids[:3])

    errors: list[str] = []
    if armed_count is not None:
        if armed_count == 0 and pending_ids:
            errors.append(
                "`Capital Deployment` says `0 armed receipts`, but the handoff still claims armed entry receipt(s) "
                f"{pending_preview}."
            )
        if armed_count > 0 and not pending_ids:
            errors.append(
                "`Capital Deployment` says armed receipts survived, but no armed entry receipt with `id=` appears "
                "anywhere in the handoff."
            )

    if pending_orders_line and _is_noneish(pending_orders_line):
        if armed_count and armed_count > 0:
            errors.append(
                "`Positions (Current)` says `Pending orders: none`, but `Capital Deployment` still claims armed receipts."
            )
        if backup_trigger_line and not _is_noneish(backup_trigger_line) and not _is_dead(backup_trigger_line):
            errors.append(
                "`20-minute backup trigger armed NOW` names a live lane even though `Pending orders` says `none`. "
                "Use `none because ...` until a real pending entry `id=` exists."
            )

    if armed_backup_line and _is_noneish(armed_backup_line) and armed_count and armed_count > 0:
        errors.append(
            "`Armed backup lane for this cadence` says `none`, but `Execution count this session` still claims armed receipts."
        )

    lane_statuses: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for label, line in _extract_lane_lines(text).items():
        matches = PAIR_DIRECTION_RE.findall(line)
        if len(matches) > 1:
            errors.append(
                f"`{label}` mentions multiple pair/direction seats. Close exactly one lane per line; "
                "do not hide another board seat inside the same lane."
            )
        if not matches:
            continue
        key = matches[0]
        lane_statuses.setdefault(key, []).append(("dead" if _is_dead(line) else "live", line))

    focus_lines = {
        "Backup vehicle": _parse_simple_line(market_section, "Backup vehicle"),
        "Next fresh risk allowed NOW": _parse_simple_line(market_section, "Next fresh risk allowed NOW"),
        "20-minute backup trigger armed NOW": backup_trigger_line,
        "SECOND SHOT / OTHER SIDE": _parse_simple_line(capital_section, "SECOND SHOT / OTHER SIDE"),
    }
    for label, line in focus_lines.items():
        key = _pair_direction_key(line)
        if not key or _is_dead(line):
            continue
        lane_states = lane_statuses.get(key) or []
        live_lane_exists = any(state != "dead" for state, _line in lane_states)
        blocking_dead_lines = [
            line
            for state, line in lane_states
            if state == "dead" and not _dead_line_allows_carry(line)
        ]
        if lane_states and not live_lane_exists and blocking_dead_lines:
            errors.append(
                f"`{label}` still names `{key[0]} {key[1]}` as active, but its lane line already closes that seat "
                "as `dead thesis because ...`."
            )

    return errors


def _validate_live_oanda_receipts(text: str) -> list[str]:
    try:
        cfg = get_oanda_config()
        acct = str(cfg["oanda_account_id"])
        open_trades = _oanda_api(f"/v3/accounts/{acct}/openTrades", cfg).get("trades", [])
        pending_orders = _oanda_api(f"/v3/accounts/{acct}/pendingOrders", cfg).get("orders", [])
    except Exception as exc:
        return [f"Live OANDA receipt verification failed: {exc}"]

    open_trade_ids = {str(trade.get("id")) for trade in open_trades if trade.get("id")}
    pending_order_ids = {str(order.get("id")) for order in pending_orders if order.get("id")}
    claims = _collect_live_receipt_claims(text)

    errors: list[str] = []
    for (kind, receipt_id), sections in sorted(claims.items()):
        if kind == "trade" and receipt_id not in open_trade_ids:
            errors.append(
                f"Live OANDA book is missing trade id=`{receipt_id}` referenced in {', '.join(sorted(sections))}."
            )
        if kind == "pending" and receipt_id not in pending_order_ids:
            errors.append(
                f"Live OANDA book is missing pending order id=`{receipt_id}` referenced in {', '.join(sorted(sections))}."
            )
    return errors


def _validate_live_book_coverage(text: str, *, log_path: Path | None = None) -> list[str]:
    try:
        cfg = get_oanda_config()
        acct = str(cfg["oanda_account_id"])
        open_trades = _oanda_api(f"/v3/accounts/{acct}/openTrades", cfg).get("trades", [])
        pending_orders = _oanda_api(f"/v3/accounts/{acct}/pendingOrders", cfg).get("orders", [])
    except Exception as exc:
        return [f"Live book coverage verification failed: {exc}"]

    state_ids = set(ID_RE.findall(text))
    try:
        log_text = (log_path or (STATE_PATH.parent.parent / "logs" / "live_trade_log.txt")).read_text()
    except Exception:
        log_text = ""
    log_ids = set(ID_RE.findall(log_text))

    errors: list[str] = []
    for trade in open_trades:
        trade_id = str(trade.get("id") or "").strip()
        if not trade_id:
            continue
        pair = str(trade.get("instrument") or "UNKNOWN")
        if trade_id not in state_ids:
            errors.append(
                f"Live open trade id=`{trade_id}` ({pair}) exists in OANDA but is missing from `collab_trade/state.md`."
            )
        if trade_id not in log_ids:
            errors.append(
                f"Live open trade id=`{trade_id}` ({pair}) exists in OANDA but is missing from `logs/live_trade_log.txt`."
            )

    for order in pending_orders:
        order_type = str(order.get("type") or "").upper()
        if order_type not in ENTRY_PENDING_TYPES:
            continue
        order_id = str(order.get("id") or "").strip()
        if not order_id:
            continue
        pair = str(order.get("instrument") or "UNKNOWN")
        if order_id not in state_ids:
            errors.append(
                f"Live pending entry order id=`{order_id}` ({pair} {order_type}) exists in OANDA but is missing from `collab_trade/state.md`."
            )
        if order_id not in log_ids:
            errors.append(
                f"Live pending entry order id=`{order_id}` ({pair} {order_type}) exists in OANDA but is missing from `logs/live_trade_log.txt`."
            )

    return errors


def validate_state_for_entry(
    state_path: Path,
    *,
    verify_live_oanda: bool = True,
    verify_live_book_coverage: bool = True,
) -> list[str]:
    text = state_path.read_text()
    errors: list[str] = []
    if verify_live_oanda:
        errors.extend(_validate_live_oanda_receipts(text))
    if verify_live_book_coverage:
        errors.extend(_validate_live_book_coverage(text))
    return errors


def _validate_live_pending_entry_orderability() -> list[str]:
    try:
        cfg = get_oanda_config()
        acct = str(cfg["oanda_account_id"])
        pending_orders = _oanda_api(f"/v3/accounts/{acct}/pendingOrders", cfg).get("orders", [])
    except Exception as exc:
        return [f"Live pending orderability verification failed: {exc}"]

    errors: list[str] = []
    for order in pending_orders:
        order_type = str(order.get("type") or "").strip().upper()
        if order_type not in ENTRY_PENDING_TYPES:
            continue
        ext = order.get("clientExtensions") or {}
        tag = str(ext.get("tag") or "").strip().lower()
        if tag and tag != "trader":
            continue
        trade_ext = order.get("tradeClientExtensions") or {}

        order_id = str(order.get("id") or "").strip() or "?"
        pair = str(order.get("instrument") or "").strip().upper()
        units_raw = str(order.get("units") or "0").strip()
        try:
            units = float(units_raw)
        except ValueError:
            units = 0.0
        direction = "LONG" if units > 0 else "SHORT"
        requested_style = requested_style_from_order_type(order_type)

        try:
            entry_price = float(order.get("price")) if order.get("price") is not None else None
            tp_price = float(((order.get("takeProfitOnFill") or {}).get("price")))
            sl_price = float(((order.get("stopLossOnFill") or {}).get("price")))
        except (TypeError, ValueError):
            errors.append(
                f"Live pending trader order id=`{order_id}` ({pair}) is missing a clean entry/TP/SL payload for exact orderability review."
            )
            continue

        comment_blob = " ".join(
            str(value or "")
            for value in (ext.get("comment"), trade_ext.get("comment"))
        ).lower()
        counter = any(token in comment_blob for token in COUNTER_ORDER_COMMENT_TOKENS)

        result = run_exact_pretrade(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            counter=counter,
        )
        issues = exact_pretrade_issues(requested_style=requested_style, result=result)
        for issue in issues:
            errors.append(
                f"Live pending trader order id=`{order_id}` ({pair} {direction} {order_type}) no longer clears exact pretrade: {issue}"
            )
    return errors


def _validate_as_excavation_mandate(text: str) -> list[str]:
    section = _extract_section(text, "A/S Excavation Mandate")
    if not section:
        return ["Missing `## A/S Excavation Mandate` block."]

    live_block = _extract_labeled_block(section, "Best A/S live now", MANDATE_LABELS)
    one_print_block = _extract_labeled_block(section, "Best A/S one print away", MANDATE_LABELS)
    reject_block = _extract_labeled_block(section, "Best A/S I am explicitly rejecting", MANDATE_LABELS)

    errors: list[str] = []
    if not live_block:
        errors.append("`A/S Excavation Mandate` is missing `Best A/S live now`.")
    if not one_print_block:
        errors.append("`A/S Excavation Mandate` is missing `Best A/S one print away`.")
    if not reject_block:
        errors.append("`A/S Excavation Mandate` is missing `Best A/S I am explicitly rejecting`.")

    live_header = next((line.strip() for line in (live_block or "").splitlines() if line.strip()), None)
    one_print_header = next((line.strip() for line in (one_print_block or "").splitlines() if line.strip()), None)

    live_seat = _parse_mandate_seat(live_block, "Order now")
    one_print_seat = _parse_mandate_seat(one_print_block, "Arm now as")
    pending_section = _extract_section(text, "Pending LIMITs")

    live_action = _normalized_action(live_seat.get("action"))
    one_print_action = _normalized_action(one_print_seat.get("action"))

    if live_block and live_seat.get("action") is None:
        errors.append("`Best A/S live now` is missing `Order now:`.")
    if one_print_block and one_print_seat.get("action") is None:
        errors.append("`Best A/S one print away` is missing `Arm now as:`.")

    live_header_dead_phrase = _tautological_dead_closure_phrase(live_header)
    if live_header_dead_phrase:
        errors.append(
            "`Best A/S live now` header closes the seat only because "
            f"`{live_header_dead_phrase}`. Keep the header as seat identity; put the real closure only in `Order now:`."
        )
    one_print_header_dead_phrase = _tautological_dead_closure_phrase(one_print_header)
    if one_print_header_dead_phrase:
        errors.append(
            "`Best A/S one print away` header closes the seat only because "
            f"`{one_print_header_dead_phrase}`. Keep the header as seat identity; put the real closure only in `Arm now as:`."
        )

    live_dead_phrase = _tautological_dead_closure_phrase(live_seat.get("action"))
    if live_dead_phrase:
        errors.append(
            "`Best A/S live now` closes the seat only because "
            f"`{live_dead_phrase}`. That is missing execution, not a chart contradiction."
        )
    one_print_dead_phrase = _tautological_dead_closure_phrase(one_print_seat.get("action"))
    if one_print_dead_phrase:
        errors.append(
            "`Best A/S one print away` closes the seat only because "
            f"`{one_print_dead_phrase}`. If the print is honest, arm it or state the actual blocker."
        )

    if live_action in LIVE_ACTIONS and not _has_matching_receipt(text, live_seat, live_action):
        errors.append(
            "`Best A/S live now` names a live/orderable seat but no real receipt with `id=` appears anywhere in the handoff."
        )

    if one_print_action in ARMABLE_ACTIONS:
        if not _has_matching_receipt(text, one_print_seat, one_print_action):
            errors.append(
                "`Best A/S one print away` says to arm the seat, but no armed receipt with `id=` appears anywhere in the handoff."
            )
        if pending_section and _pending_section_says_none(pending_section):
            errors.append(
                "`Best A/S one print away` still says `Arm now`, but `## Pending LIMITs` says no fresh entry orders are live."
            )

    if pending_section and _pending_section_says_none(pending_section):
        normalized_text = _normalize_text(text).upper()
        if "ARMED LIMIT ID=" in normalized_text or "ARMED STOP ID=" in normalized_text:
            errors.append(
                "`## Pending LIMITs` says no fresh entry orders are live, but another section still claims an armed entry order."
            )

    return errors


def _validate_self_check_summary(text: str) -> list[str]:
    section = _extract_section(text, "Self-check")
    if not section:
        return ["Missing `## Self-check` block."]

    entries_line = _parse_simple_line(section, "Entries today")
    if not entries_line:
        return ["`Self-check` is missing `Entries today`."]

    lowered = _normalize_text(entries_line).lower()
    if " total" in lowered or not all(token in lowered for token in ENTRY_ACTIVITY_REQUIRED_TOKENS):
        return [
            "`Self-check` `Entries today` must use the live structured format "
            "(`X fills / Y new entry orders / Z rejects`), not stale `N total` prose."
        ]

    return []


def _validate_hot_updates(text: str) -> list[str]:
    section = _extract_section(text, "Hot Updates")
    if not section:
        return []

    errors: list[str] = []
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        note = line.split("|", 2)[-1].strip() if "|" in line else line
        phrase = _execution_only_blocker_phrase(note)
        if phrase:
            errors.append(
                "`Hot Updates` still explains a carry-forward seat only with missing execution "
                f"(`{phrase}`). Write the tape / trigger blocker that kept the seat unpromoted."
            )
    return errors


def _validate_multi_vehicle_closure(text: str) -> list[str]:
    capital_section = _extract_section(text, "Capital Deployment")
    if not capital_section:
        return ["Missing `## Capital Deployment` block."]

    errors: list[str] = []
    lane_lines = _extract_lane_lines(text)
    if not lane_lines:
        errors.append("`Capital Deployment` handoff is missing any `Lane N / ...` lines.")
    if not any(label.lower().startswith("lane 1 /") for label in lane_lines):
        errors.append("`Capital Deployment` handoff must include `Lane 1 / ...`.")

    for label, line in lane_lines.items():
        pair_mentions = PAIR_DIRECTION_RE.findall(line)
        if len(pair_mentions) > 1:
            errors.append(
                f"`{label}` mentions multiple pair/direction seats. Close exactly one lane per line."
            )
        bad_phrase = _contains_lane_bad_prose(line)
        if bad_phrase:
            errors.append(f"`{label}` still uses prose-only lane closure (`{bad_phrase}`).")
        tautology = _tautological_dead_closure_phrase(line)
        if tautology:
            errors.append(
                f"`{label}` closes a lane only because `{tautology}`. "
                "Missing receipt is not valid thesis death."
            )
        pair_match = PAIR_DIRECTION_RE.search(line)
        if pair_match and not _has_id(line) and not _is_dead(line):
            errors.append(
                f"`{label}` names a live lane candidate but does not close as `id=...` or `dead thesis because ...`."
            )

    execution_line = _parse_simple_line(capital_section, "Execution count this session")
    if not execution_line:
        errors.append("`Capital Deployment` is missing `Execution count this session`.")
        live_count = armed_count = None
    else:
        live_count, armed_count = _parse_execution_count(execution_line)
        if live_count is None or armed_count is None:
            errors.append(
                "`Capital Deployment` `Execution count this session` must say `X live receipts | Y armed receipts`."
            )

    blocker_line = _parse_simple_line(
        capital_section,
        "If broad tape but fewer than 2 live/armed lanes survived",
    )
    if live_count is not None and armed_count is not None and (live_count + armed_count) < 2:
        if not blocker_line:
            errors.append(
                "`Capital Deployment` must explain the exact blocker to lane two when fewer than 2 live/armed lanes survived."
            )
        elif PLACEHOLDER_TOKEN in blocker_line or not blocker_line.strip():
            errors.append(
                "`Capital Deployment` second-lane blocker line still contains blanks while the book stayed underdeployed."
            )
        elif _execution_only_blocker_phrase(blocker_line):
            errors.append(
                "`Capital Deployment` explains underdeployment only with missing execution "
                f"(`{_execution_only_blocker_phrase(blocker_line)}`). Name the tape / structure / risk blocker instead."
            )

    return errors


def _validate_gold_mine_inventory(text: str) -> list[str]:
    snapshot = _load_recent_action_board_snapshot()
    if not snapshot:
        return []
    session_mode = str((snapshot.get("session_intent") or {}).get("mode") or "").strip().upper()
    actionable_lanes = _merge_snapshot_lanes(
        _iter_snapshot_actionable_lanes(snapshot, "market_now"),
        _iter_snapshot_actionable_lanes(snapshot, "multi_vehicle_lanes"),
    )
    if session_mode != "FULL_TRADER" or not actionable_lanes:
        return []

    section = _extract_section(text, "Gold Mine Inventory")
    if not section:
        return [
            "`Gold Mine Inventory` is missing while the latest action board is `FULL_TRADER` "
            "with actionable lanes. Copy at least Gold #1-#5 or write exact contradictions."
        ]

    lines = [line.strip() for line in section.splitlines() if line.strip()]
    gold_lines: dict[int, str] = {}
    errors: list[str] = []
    for line in lines:
        match = GOLD_LINE_RE.match(line)
        if match:
            gold_lines[int(match.group(1))] = match.group(2).strip()

    required = min(MIN_GOLD_ROWS, len(actionable_lanes))
    for idx in range(1, required + 1):
        line = gold_lines.get(idx)
        if not line:
            errors.append(f"`Gold Mine Inventory` is missing `Gold #{idx}`.")
            continue
        if PLACEHOLDER_TOKEN in line:
            errors.append(f"`Gold Mine Inventory` `Gold #{idx}` still contains placeholder blanks.")
        if not PAIR_DIRECTION_RE.search(line):
            errors.append(f"`Gold Mine Inventory` `Gold #{idx}` must name a concrete `PAIR LONG/SHORT` seat.")
        normalized = _normalize_text(line).upper()
        if not any(action in normalized for action in LIVE_ACTIONS) and "LIVE" not in normalized:
            errors.append(
                f"`Gold Mine Inventory` `Gold #{idx}` must name an executable expression "
                "(`MARKET / LIMIT / STOP-ENTRY`)."
            )
        if "ID=" not in normalized and "DEAD THESIS BECAUSE" not in normalized and "EXACT CONTRADICTION" not in normalized:
            errors.append(
                f"`Gold Mine Inventory` `Gold #{idx}` must close with a receipt `id=...` "
                "or an exact contradiction, not only a board label."
            )
        phrase = _tautological_dead_closure_phrase(line)
        if phrase:
            errors.append(
                f"`Gold Mine Inventory` `Gold #{idx}` closes only because `{phrase}`. "
                "Receipt absence is not a market contradiction."
            )
        lane = actionable_lanes[idx - 1]
        if not _line_mentions_lane(line, lane):
            errors.append(
                f"`Gold Mine Inventory` `Gold #{idx}` does not match latest action-board lane "
                f"`{_snapshot_lane_label(lane)}`."
            )
    return errors


def _validate_s_excavation(text: str, session_date: str | None) -> list[str]:
    if not session_date or session_date < EXCAVATION_REQUIRED_FROM:
        return []

    section = _extract_section(text, "S Excavation Matrix")
    if not section:
        return [f"Missing `## S Excavation Matrix` for {EXCAVATION_REQUIRED_FROM}+ state handoffs."]

    lines = [line.strip() for line in section.splitlines() if line.strip()]
    errors: list[str] = []
    for pair in REQUIRED_EXCAVATION_PAIRS:
        pair_line = next((line for line in lines if line.startswith(f"{pair}:")), None)
        if not pair_line:
            errors.append(f"`S Excavation Matrix` is missing the `{pair}` line.")
            continue
        if PLACEHOLDER_TOKEN in pair_line:
            errors.append(f"`S Excavation Matrix` `{pair}` line still contains placeholder blanks.")
        lowered = pair_line.lower()
        best_expression = _extract_pipe_field(pair_line, "Best expression")
        why_not_s_now = _extract_pipe_field(pair_line, "Why not S now")
        upgrade_only_if = _extract_pipe_field(pair_line, "Upgrade to S only if") or _extract_pipe_field(
            pair_line, "Upgrade to S only after"
        )
        dead_if = _extract_pipe_field(pair_line, "Dead if") or _extract_pipe_field(pair_line, "Dead while")
        if "best expression" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Best expression`.")
        elif not best_expression:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Best expression`.")
        if "why not s now" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Why not S now`.")
        elif not why_not_s_now:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Why not S now`.")
        if "upgrade to s" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Upgrade to S`.")
        elif not upgrade_only_if:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Upgrade to S only if`.")
        if "dead if" not in lowered and "dead while" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Dead if`.")
        elif not dead_if:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Dead if`.")

    podium_lines = [
        (int(match.group(1)), line)
        for line in lines
        for match in [PODIUM_LINE_RE.match(line)]
        if match
    ]
    podium_lines.sort(key=lambda item: item[0])

    for idx in range(1, MIN_PODIUM_ROWS + 1):
        podium_line = next((line for rank, line in podium_lines if rank == idx), None)
        if not podium_line:
            errors.append(f"`S Excavation Matrix` is missing `Podium #{idx}`.")
            continue
    for idx, podium_line in podium_lines:
        if PLACEHOLDER_TOKEN in podium_line:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` still contains placeholder blanks.")
        if not PAIR_DIRECTION_RE.search(podium_line):
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must name a concrete `PAIR LONG/SHORT` seat.")
        reason = _extract_pipe_field(podium_line, "Closest-to-S because")
        blocker = _extract_pipe_field(podium_line, "Still blocked by")
        upgrade_action = _extract_pipe_field(podium_line, "If it upgrades")
        if "closest-to-s because" not in podium_line.lower():
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` is missing `Closest-to-S because`.")
        elif not reason:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `Closest-to-S because`.")
        if "still blocked by" not in podium_line.lower():
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` is missing `Still blocked by`.")
        elif not blocker:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `Still blocked by`.")
        if not upgrade_action:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `If it upgrades`.")
        elif upgrade_action.upper() not in {"MARKET", "LIMIT", "STOP-ENTRY"} and not (
            ("HOLD" in upgrade_action.upper() or "LIVE" in upgrade_action.upper())
            and "live" in podium_line.lower()
        ):
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must name the upgrade action (`MARKET / LIMIT / STOP-ENTRY`).")

    return errors


def validate_state(
    state_path: Path,
    *,
    verify_live_oanda: bool = False,
    verify_live_book_coverage: bool = False,
    check_action_board: bool = True,
    verify_live_entry_orderability: bool = False,
    verify_live_slack_replies: bool = False,
) -> list[str]:
    entry = build_entry(state_path)
    if entry is None:
        return [f"Could not parse state handoff from {state_path}"]

    text = state_path.read_text()
    capital_section = _extract_section(text, "Capital Deployment")
    flat_status = _parse_simple_line(capital_section, "Flat-book status") or ""
    live_now = _parse_simple_line(capital_section, "LIVE NOW") or ""
    reload_line = _parse_simple_line(capital_section, "RELOAD") or ""
    second_line = _parse_simple_line(capital_section, "SECOND SHOT / OTHER SIDE") or ""

    errors: list[str] = []
    errors.extend(_validate_unique_sections(text))
    errors.extend(_validate_self_check_summary(text))
    errors.extend(_validate_hot_updates(text))
    errors.extend(_validate_handoff_consistency(text))
    errors.extend(_validate_s_excavation(text, entry.get("session_date")))
    errors.extend(_validate_as_excavation_mandate(text))
    errors.extend(_validate_multi_vehicle_closure(text))
    errors.extend(_validate_gold_mine_inventory(text))
    if check_action_board:
        errors.extend(_validate_recent_action_board(text))
    if verify_live_oanda:
        errors.extend(_validate_live_oanda_receipts(text))
    if verify_live_book_coverage:
        errors.extend(_validate_live_book_coverage(text))
    if verify_live_entry_orderability:
        errors.extend(_validate_live_pending_entry_orderability())
    if verify_live_slack_replies:
        errors.extend(_validate_live_slack_response(text))
    horizons_by_name = {h.get("horizon"): h for h in entry.get("horizons", [])}
    horizon_deployment = entry.get("horizon_deployment", {})

    deployed_horizons = 0
    for horizon_name in REQUIRED_HORIZONS:
        horizon = horizons_by_name.get(horizon_name)
        if not horizon:
            errors.append(f"Missing `{horizon_name}` block in `## S Hunt`.")
            continue

        orderability = (horizon.get("orderability") or "").strip()
        deployment_result = (horizon.get("deployment_result") or "").strip()
        summary_line = (horizon_deployment.get(horizon_name.replace(" S", "")) or "").strip()
        raw_text = (horizon.get("raw") or "").strip()
        promotion_proof = _extract_block_field(
            raw_text,
            "Promotion proof:",
            ("MTF chain:", "Payout path:", "Orderability:", "If not live:", "Deployment result:"),
        )

        if not orderability:
            errors.append(f"`{horizon_name}` is missing `Orderability:`.")
        if not deployment_result:
            errors.append(f"`{horizon_name}` is missing `Deployment result:`.")
        if not promotion_proof:
            errors.append(f"`{horizon_name}` is missing `Promotion proof:`.")

        bad_phrase = _contains_bad_prose(deployment_result)
        if bad_phrase:
            errors.append(f"`{horizon_name}` deployment result still uses prose-only closure (`{bad_phrase}`).")
        tautology = _tautological_dead_closure_phrase(deployment_result)
        if tautology:
            errors.append(
                f"`{horizon_name}` deployment result closes only because `{tautology}`. "
                "Missing execution is not a market contradiction."
            )

        if orderability:
            lowered = orderability.lower()
            non_live_watch = "watch" in lowered or "not live" in lowered
            if "still pass" in lowered or non_live_watch:
                if not _is_dead(deployment_result):
                    errors.append(f"`{horizon_name}` is `STILL PASS` but does not close as a dead thesis.")
                if "still pass" in lowered and "no seat cleared promotion gate" not in (promotion_proof or "").lower():
                    errors.append(
                        f"`{horizon_name}` is `STILL PASS` so `Promotion proof` must say `none — no seat cleared promotion gate`."
                    )
                if "still pass" in lowered and "no seat cleared promotion gate" not in deployment_result.lower():
                    errors.append(
                        f"`{horizon_name}` is `STILL PASS` so `Deployment result` must close as `dead thesis because no seat cleared promotion gate: ...`."
                    )
            else:
                if not _has_id(deployment_result):
                    errors.append(f"`{horizon_name}` is live/orderable but has no real `id=` in `Deployment result`.")
                lowered_proof = (promotion_proof or "").lower()
                if lowered_proof.startswith("none") or "no seat cleared promotion gate" in lowered_proof:
                    errors.append(f"`{horizon_name}` is promoted live/orderable but `Promotion proof` still says no seat cleared.")
                if "blocker was" not in lowered_proof or "cleared by" not in lowered_proof:
                    errors.append(
                        f"`{horizon_name}` `Promotion proof` must state `blocker was ... -> cleared by ...`."
                    )

        if _has_id(deployment_result):
            deployed_horizons += 1

        if not summary_line:
            errors.append(f"`Capital Deployment` is missing the `{horizon_name.replace(' S', '')}` summary line.")
        else:
            bad_phrase = _contains_bad_prose(summary_line)
            if bad_phrase:
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line still uses prose-only closure (`{bad_phrase}`).")
            tautology = _tautological_dead_closure_phrase(summary_line)
            if tautology:
                errors.append(
                    f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line closes only because `{tautology}`. "
                    "That line must mirror the real market blocker, not the missing receipt."
                )

            if _has_id(deployment_result) and not _has_id(summary_line):
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line drifted away from the real order receipt.")
            summary_lowered = summary_line.lower()
            if _is_dead(deployment_result) and "dead" not in summary_lowered and not (
                "no live receipt" in summary_lowered and "close" in summary_lowered
            ):
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line must also close as dead thesis.")

    capital_has_id = any(_has_id(value) for value in (live_now, reload_line, second_line))
    is_flat = "flat" in flat_status.lower()
    if is_flat and deployed_horizons == 0 and not capital_has_id:
        errors.append("Flat book has zero real order IDs across `S Hunt` and `Capital Deployment`.")
    if is_flat and "acceptable flat book" in flat_status.lower() and deployed_horizons == 0 and not capital_has_id:
        errors.append("`acceptable flat book` is invalid when every horizon is still prose-only.")

    return errors


def main(argv: list[str]) -> int:
    require_live = False
    require_live_book_coverage = False
    skip_action_board = False
    path_arg = None
    for arg in argv[1:]:
        if arg == "--require-live-oanda":
            require_live = True
        elif arg == "--require-live-book-coverage":
            require_live_book_coverage = True
        elif arg == "--skip-action-board":
            skip_action_board = True
        else:
            path_arg = arg

    state_path = Path(path_arg).expanduser().resolve() if path_arg else STATE_PATH
    verify_live = require_live or state_path == STATE_PATH.resolve()
    verify_coverage = require_live_book_coverage or state_path == STATE_PATH.resolve()
    errors = validate_state(
        state_path,
        verify_live_oanda=verify_live,
        verify_live_book_coverage=verify_coverage,
        check_action_board=not skip_action_board,
        verify_live_entry_orderability=(state_path == STATE_PATH.resolve()),
        verify_live_slack_replies=(state_path == STATE_PATH.resolve() and TRADER_LOCK_PATH.exists()),
    )
    if errors:
        print("STATE_VALIDATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("STATE_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
