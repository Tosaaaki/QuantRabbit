#!/usr/bin/env python3
"""Helpers for the bot inventory policy written by trader or backup LLM."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_MD = ROOT / "logs" / "bot_inventory_policy.md"
POLICY_JSON = ROOT / "logs" / "bot_inventory_policy.json"
POLICY_TIME_FMT = "%Y-%m-%d %H:%M UTC"
PAIR_ORDER = (
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
    "NZD_JPY", "CAD_JPY",
    "EUR_CHF", "AUD_NZD", "AUD_CAD",
)
DEFAULT_PROJECTED_MARGIN_CAP = 0.82
DEFAULT_PANIC_MARGIN_CAP = 0.90
DEFAULT_RELEASE_MARGIN_CAP = 0.78
DEFAULT_POLICY_TTL_MIN = 30
DEFAULT_POLICY_GRACE_MIN = 120
DEFAULT_MAX_PENDING_AGE_MIN = 18
DEFAULT_TARGET_ACTIVE_WORKER_PAIRS = 1
OWNERSHIP_VALUES = {"TRADER_ONLY", "SHARED_PASSIVE", "SHARED_MARKET"}
TEMPO_VALUES = {"BALANCED", "FAST", "MICRO"}
ENTRY_BIAS_VALUES = {"PASSIVE", "BALANCED", "EARLY"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_policy_time(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime(POLICY_TIME_FMT)


def parse_policy_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), POLICY_TIME_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def default_pair_policy() -> dict:
    return {
        "mode": "BOTH",
        "allow_market": True,
        "pending": "KEEP",
        "max_pending": 1,
        "ownership": "TRADER_ONLY",
        "tempo": "BALANCED",
        "entry_bias": "BALANCED",
        "note": "",
    }


def default_policy(now: datetime | None = None, ttl_min: int = DEFAULT_POLICY_TTL_MIN) -> dict:
    now = now or utc_now()
    expires = now + timedelta(minutes=ttl_min)
    return {
        "schema": 1,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_human": format_policy_time(now),
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at_human": format_policy_time(expires),
        "global_status": "ACTIVE",
        "projected_margin_cap": DEFAULT_PROJECTED_MARGIN_CAP,
        "panic_margin_cap": DEFAULT_PANIC_MARGIN_CAP,
        "release_margin_cap": DEFAULT_RELEASE_MARGIN_CAP,
        "max_pending_age_min": DEFAULT_MAX_PENDING_AGE_MIN,
        "target_active_worker_pairs": DEFAULT_TARGET_ACTIVE_WORKER_PAIRS,
        "notes": "Default permissive policy. Replace with trader or inventory-director output.",
        "pairs": {},
    }


def clamp_float(value: object, fallback: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, parsed))


def normalize_ownership(value: object) -> str:
    ownership = str(value or "TRADER_ONLY").upper()
    if ownership not in OWNERSHIP_VALUES:
        return "TRADER_ONLY"
    return ownership


def normalize_tempo(value: object) -> str:
    tempo = str(value or "BALANCED").upper()
    if tempo not in TEMPO_VALUES:
        return "BALANCED"
    return tempo


def normalize_entry_bias(value: object) -> str:
    entry_bias = str(value or "BALANCED").upper()
    if entry_bias not in ENTRY_BIAS_VALUES:
        return "BALANCED"
    return entry_bias


def ownership_allows_worker(ownership: str, worker_kind: str) -> bool:
    ownership = normalize_ownership(ownership)
    worker_kind = str(worker_kind or "").upper()
    if ownership == "SHARED_MARKET":
        return worker_kind in {"PASSIVE", "MARKET"}
    if ownership == "SHARED_PASSIVE":
        return worker_kind == "PASSIVE"
    return False


def normalize_pair_policy(raw: dict | None) -> dict:
    base = default_pair_policy()
    if not isinstance(raw, dict):
        return base
    mode = str(raw.get("mode", base["mode"])).upper()
    if mode not in {"BOTH", "LONG_ONLY", "SHORT_ONLY", "PAUSE"}:
        mode = base["mode"]
    pending = str(raw.get("pending", base["pending"])).upper()
    if pending not in {"KEEP", "CANCEL"}:
        pending = base["pending"]
    try:
        max_pending = max(0, int(raw.get("max_pending", base["max_pending"])))
    except (TypeError, ValueError):
        max_pending = base["max_pending"]
    return {
        "mode": mode,
        "allow_market": bool(raw.get("allow_market", base["allow_market"])),
        "pending": pending,
        "max_pending": max_pending,
        "ownership": normalize_ownership(raw.get("ownership", base["ownership"])),
        "tempo": normalize_tempo(raw.get("tempo", base["tempo"])),
        "entry_bias": normalize_entry_bias(raw.get("entry_bias", base["entry_bias"])),
        "note": str(raw.get("note", base["note"])),
    }


def _parse_policy_iso(text: object) -> datetime | None:
    try:
        if not text:
            return None
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_policy_dict(raw: dict | None, now: datetime | None = None) -> dict:
    now = now or utc_now()
    loaded = raw if isinstance(raw, dict) else {}

    updated = parse_policy_time(loaded.get("generated_at_human")) or parse_policy_time(loaded.get("updated"))
    if updated is None:
        updated = _parse_policy_iso(loaded.get("generated_at"))
    if updated is None:
        updated = now

    expires = parse_policy_time(loaded.get("expires_at_human")) or parse_policy_time(loaded.get("expires"))
    if expires is None:
        expires = _parse_policy_iso(loaded.get("expires_at"))
    if expires is None or expires <= updated:
        expires = updated + timedelta(minutes=DEFAULT_POLICY_TTL_MIN)

    ttl_min = max(1, int((expires - updated).total_seconds() / 60))
    policy = default_policy(updated, ttl_min=ttl_min)
    policy["generated_at_human"] = format_policy_time(updated)
    policy["generated_at"] = updated.strftime("%Y-%m-%dT%H:%M:%SZ")
    policy["expires_at_human"] = format_policy_time(expires)
    policy["expires_at"] = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    policy["global_status"] = str(loaded.get("global_status", policy["global_status"])).upper()
    policy["projected_margin_cap"] = clamp_float(
        loaded.get("projected_margin_cap"), DEFAULT_PROJECTED_MARGIN_CAP, 0.50, 0.90
    )
    policy["panic_margin_cap"] = clamp_float(
        loaded.get("panic_margin_cap"), DEFAULT_PANIC_MARGIN_CAP, 0.75, 0.95
    )
    policy["release_margin_cap"] = clamp_float(
        loaded.get("release_margin_cap"), DEFAULT_RELEASE_MARGIN_CAP, 0.60, 0.90
    )
    policy["max_pending_age_min"] = max(
        5,
        int(clamp_float(loaded.get("max_pending_age_min"), DEFAULT_MAX_PENDING_AGE_MIN, 5, 120)),
    )
    policy["target_active_worker_pairs"] = max(
        0,
        int(
            clamp_float(
                loaded.get("target_active_worker_pairs"),
                DEFAULT_TARGET_ACTIVE_WORKER_PAIRS,
                0,
                7,
            )
        ),
    )
    policy["notes"] = str(loaded.get("notes", policy["notes"]))

    raw_pairs = loaded.get("pairs", {})
    if not isinstance(raw_pairs, dict):
        raw_pairs = {}
    extra_pairs = sorted(pair for pair in raw_pairs.keys() if pair not in PAIR_ORDER)
    ordered_pairs = list(PAIR_ORDER) + extra_pairs
    policy["pairs"] = {pair: normalize_pair_policy(raw_pairs.get(pair)) for pair in ordered_pairs}
    return policy


def format_policy_markdown(policy: dict | None) -> str:
    normalized = normalize_policy_dict(policy)
    lines = [
        "# Bot Inventory Policy",
        "",
        f"Updated: {normalized['generated_at_human']}",
        f"Expires: {normalized['expires_at_human']}",
        f"Global Status: {normalized['global_status']}",
        f"Projected Margin Cap: {normalized['projected_margin_cap']:.2f}",
        f"Panic Margin Cap: {normalized['panic_margin_cap']:.2f}",
        f"Release Margin Cap: {normalized['release_margin_cap']:.2f}",
        f"Max Pending Age Min: {normalized['max_pending_age_min']}",
        f"Target Active Worker Pairs: {normalized['target_active_worker_pairs']}",
        f"Notes: {str(normalized['notes']).replace('|', '/')}",
        "",
        "| Pair | Mode | Market | Pending | MaxPending | Ownership | Tempo | EntryBias | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair, pair_policy in normalized["pairs"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    pair,
                    pair_policy["mode"],
                    "YES" if pair_policy["allow_market"] else "NO",
                    pair_policy["pending"],
                    str(pair_policy["max_pending"]),
                    pair_policy["ownership"],
                    pair_policy["tempo"],
                    pair_policy["entry_bias"],
                    str(pair_policy["note"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def load_policy() -> tuple[dict, list[str]]:
    notes: list[str] = []
    policy = default_policy()
    if not POLICY_JSON.exists():
        notes.append("policy json missing; reduce-only fallback")
        policy["global_status"] = "REDUCE_ONLY"
        return policy, notes

    try:
        loaded = json.loads(POLICY_JSON.read_text())
    except json.JSONDecodeError:
        notes.append("policy json corrupt; reduce-only fallback")
        policy["global_status"] = "REDUCE_ONLY"
        return policy, notes

    policy = normalize_policy_dict(loaded)
    if policy["global_status"] not in {"ACTIVE", "REDUCE_ONLY", "PAUSE_ALL"}:
        notes.append(f"unknown global_status={policy['global_status']}; reduce-only fallback")
        policy["global_status"] = "REDUCE_ONLY"

    expires_at = parse_policy_time(policy.get("expires_at_human")) or parse_policy_time(
        loaded.get("expires_at_human")
    )
    if expires_at is None:
        try:
            expires_at = datetime.fromisoformat(str(policy["expires_at"]).replace("Z", "+00:00"))
        except ValueError:
            expires_at = None
    now = utc_now()
    if expires_at is not None and now > expires_at:
        grace_deadline = expires_at + timedelta(minutes=DEFAULT_POLICY_GRACE_MIN)
        if policy["global_status"] == "ACTIVE" and now <= grace_deadline:
            notes.append(
                f"policy expired at {format_policy_time(expires_at)}; "
                f"ACTIVE grace until {format_policy_time(grace_deadline)}"
            )
        else:
            notes.append(f"policy expired at {format_policy_time(expires_at)}; reduce-only fallback")
            policy["global_status"] = "REDUCE_ONLY"
    return policy, notes


def get_pair_policy(policy: dict, pair: str) -> dict:
    return normalize_pair_policy((policy.get("pairs") or {}).get(pair))


def global_status_allows_new_entries(policy: dict) -> bool:
    return str(policy.get("global_status", "")).upper() == "ACTIVE"


def mode_allows_direction(mode: str, direction: str) -> bool:
    mode = mode.upper()
    direction = direction.upper()
    if mode == "PAUSE":
        return False
    if mode == "BOTH":
        return True
    if mode == "LONG_ONLY":
        return direction == "BUY"
    if mode == "SHORT_ONLY":
        return direction == "SELL"
    return False


def pair_policy_worker_block_reason(pair_policy: dict, direction: str, worker_kind: str) -> str | None:
    pair_policy = normalize_pair_policy(pair_policy)
    direction = str(direction or "").upper()
    worker_kind = str(worker_kind or "").upper()
    if not mode_allows_direction(pair_policy["mode"], direction):
        return f"policy {pair_policy['mode']} blocks {direction}"
    if worker_kind == "MARKET" and not pair_policy["allow_market"]:
        return "policy market disabled"
    if worker_kind == "PASSIVE":
        if pair_policy["pending"] == "CANCEL":
            return "policy pending disabled"
        if int(pair_policy["max_pending"]) <= 0:
            return "policy max_pending=0"
    return None


def pair_policy_allows_worker_entry(pair_policy: dict, direction: str, worker_kind: str) -> bool:
    return pair_policy_worker_block_reason(pair_policy, direction, worker_kind) is None
