#!/usr/bin/env python3
"""Deterministic guardrail against flat-book worker starvation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from bot_policy import (
    DEFAULT_POLICY_TTL_MIN,
    POLICY_JSON,
    POLICY_MD,
    default_policy,
    format_policy_markdown,
    get_pair_policy,
    mode_allows_direction,
    normalize_policy_dict,
    utc_now,
)
from bot_trade_manager import BOT_TAGS, fetch_open_trades, fetch_pending_orders, get_tag, load_config
from range_bot import (
    FAST_MIN_MARKET_RR,
    MICRO_MIN_MARKET_RR,
    MIN_MARKET_RR,
    choose_order_plan,
    fetch_prices,
    is_entry_pending_order,
    scan_ranges,
)
from render_bot_inventory_policy import parse_markdown
from trend_bot import scan_trends


CONVICTION_SCORE = {"S": 400, "A": 300, "B": 200, "C": 100}
RANGE_TF_SCORE = {"H1": 20, "M5": 10}


def load_policy_from_paths(policy_md: Path, policy_json: Path) -> tuple[dict, list[str], str]:
    notes: list[str] = []
    raw: dict | None = None
    source = "default"
    now = utc_now()

    if policy_json.exists():
        try:
            raw = json.loads(policy_json.read_text())
            source = "json"
        except json.JSONDecodeError:
            notes.append("policy json corrupt; trying markdown recovery")

    if raw is None and policy_md.exists():
        try:
            raw = parse_markdown(policy_md)
            source = "markdown"
            notes.append("policy recovered from markdown")
        except Exception as exc:  # pragma: no cover - defensive recovery path
            notes.append(f"policy markdown unreadable: {exc}")

    if raw is None:
        raw = default_policy(now)
        raw["global_status"] = "REDUCE_ONLY"
        raw["notes"] = "Policy guard fallback: no valid trader or backup policy was available."

    return normalize_policy_dict(raw, now=now), notes, source


def min_market_rr_for_tempo(tempo: str) -> float:
    tempo = str(tempo or "BALANCED").upper()
    if tempo == "MICRO":
        return MICRO_MIN_MARKET_RR
    if tempo == "FAST":
        return FAST_MIN_MARKET_RR
    return MIN_MARKET_RR


def range_direction(opp: dict) -> str | None:
    signal = str(opp.get("active_signal", "")).upper()
    if "BUY" in signal:
        return "BUY"
    if "SELL" in signal:
        return "SELL"
    return None


def range_plan_for_policy(opp: dict, pair_policy: dict, prices: dict) -> dict | None:
    direction = range_direction(opp)
    if direction is None:
        return None
    if direction == "BUY":
        entry = opp["buy_entry"]
        tp = opp["bb_mid"]
        sl = opp["buy_sl"]
    else:
        entry = opp["sell_entry"]
        tp = opp["bb_mid"]
        sl = opp["sell_sl"]
    return choose_order_plan(
        opp,
        opp["pair"],
        direction,
        entry,
        tp,
        sl,
        prices,
        min_market_rr_for_tempo(pair_policy.get("tempo", "BALANCED")),
        pair_policy.get("tempo", "BALANCED"),
        pair_policy.get("entry_bias", "BALANCED"),
        True,
    )


def is_range_allowed(policy: dict, opp: dict, prices: dict) -> bool:
    direction = range_direction(opp)
    if direction is None:
        return False
    pair_policy = get_pair_policy(policy, opp["pair"])
    if not mode_allows_direction(pair_policy["mode"], direction):
        return False
    if pair_policy["pending"] != "KEEP" or int(pair_policy["max_pending"]) <= 0:
        return False
    plan = range_plan_for_policy(opp, pair_policy, prices)
    return plan is not None and plan.get("order_type") != "SKIP"


def range_repair_candidate(policy: dict, opp: dict, prices: dict) -> dict | None:
    direction = range_direction(opp)
    if direction is None:
        return None
    pair_policy = get_pair_policy(policy, opp["pair"])
    blocked_by_policy = []
    if not mode_allows_direction(pair_policy["mode"], direction):
        blocked_by_policy.append(f"mode {pair_policy['mode']} blocks {direction}")
    if pair_policy["pending"] != "KEEP" or int(pair_policy["max_pending"]) <= 0:
        blocked_by_policy.append(
            f"pending {pair_policy['pending']} / max_pending {pair_policy['max_pending']}"
        )
    if not blocked_by_policy:
        return None

    repaired_policy = deepcopy(pair_policy)
    repaired_policy["mode"] = "LONG_ONLY" if direction == "BUY" else "SHORT_ONLY"
    repaired_policy["allow_market"] = False
    repaired_policy["pending"] = "KEEP"
    repaired_policy["max_pending"] = max(1, int(pair_policy.get("max_pending", 0)))
    repaired_policy["entry_bias"] = "EARLY"
    plan = range_plan_for_policy(opp, repaired_policy, prices)
    if plan is None or plan.get("order_type") == "SKIP":
        return None

    score = (
        CONVICTION_SCORE.get(str(opp.get("conviction", "C")).upper(), 0)
        + int(opp.get("signal_strength", 0)) * 10
        + RANGE_TF_SCORE.get(str(opp.get("setup_tf", "M5")).upper(), 0)
    )
    side = "LONG" if direction == "BUY" else "SHORT"
    repair_note = (
        f"Auto-guard repair: flat-book coverage was starved, so keep one passive {side.lower()} "
        f"scout alive on the cleanest current range seat."
    )
    return {
        "kind": "range",
        "pair": opp["pair"],
        "direction": direction,
        "score": score,
        "reason": "; ".join(blocked_by_policy),
        "row": repaired_policy | {"note": repair_note},
        "header_note": (
            f"Auto-guard reopened {opp['pair']} {direction} as a passive flat-book repair seat "
            f"because coverage target was live but every current range lane was blocked only by policy."
        ),
    }


def normalize_trend_tempo(current_tempo: str, m1_state: str) -> str:
    tempo = str(current_tempo or "BALANCED").upper()
    m1_state = str(m1_state or "").lower()
    if tempo == "FAST" and m1_state != "aligned":
        return "MICRO" if m1_state == "reload" else "BALANCED"
    if tempo == "MICRO" and m1_state not in {"aligned", "reload"}:
        return "BALANCED"
    return tempo


def is_trend_allowed(policy: dict, opp: dict) -> bool:
    pair_policy = get_pair_policy(policy, opp["pair"])
    direction = str(opp.get("direction", "")).upper()
    return mode_allows_direction(pair_policy["mode"], direction) and bool(pair_policy["allow_market"])


def trend_repair_candidate(policy: dict, opp: dict) -> dict | None:
    pair_policy = get_pair_policy(policy, opp["pair"])
    direction = str(opp.get("direction", "")).upper()
    blocked_by_policy = []
    if not mode_allows_direction(pair_policy["mode"], direction):
        blocked_by_policy.append(f"mode {pair_policy['mode']} blocks {direction}")
    if not pair_policy["allow_market"]:
        blocked_by_policy.append("market disabled")
    if not blocked_by_policy:
        return None

    repaired_policy = deepcopy(pair_policy)
    repaired_policy["mode"] = "LONG_ONLY" if direction == "BUY" else "SHORT_ONLY"
    repaired_policy["allow_market"] = True
    repaired_policy["pending"] = "KEEP"
    repaired_policy["max_pending"] = max(1, int(pair_policy.get("max_pending", 0)))
    repaired_policy["tempo"] = normalize_trend_tempo(pair_policy.get("tempo", "BALANCED"), opp.get("m1_state", ""))
    repaired_policy["entry_bias"] = "EARLY"

    score = (
        CONVICTION_SCORE.get(str(opp.get("conviction", "C")).upper(), 0)
        + int(opp.get("signal_strength", 0)) * 10
    )
    side = "LONG" if direction == "BUY" else "SHORT"
    repair_note = (
        f"Auto-guard repair: flat-book coverage was starved, so reopen the cleanest continuation "
        f"{side.lower()} lane instead of staying empty behind stale policy."
    )
    return {
        "kind": "trend",
        "pair": opp["pair"],
        "direction": direction,
        "score": score,
        "reason": "; ".join(blocked_by_policy),
        "row": repaired_policy | {"note": repair_note},
        "header_note": (
            f"Auto-guard reopened {opp['pair']} {direction} as a continuation repair seat "
            f"because coverage target was live but every current trend lane was blocked only by policy."
        ),
    }


def worker_pending_pairs(token: str, acct: str) -> set[str]:
    pending_orders = fetch_pending_orders(token, acct)
    return {
        str(order.get("instrument", "?"))
        for order in pending_orders
        if get_tag(order) in BOT_TAGS and is_entry_pending_order(order)
    }


def apply_repair(policy: dict, candidate: dict, ttl_min: int) -> dict:
    updated = normalize_policy_dict(policy, now=utc_now())
    now = utc_now()
    expires = now + timedelta(minutes=max(5, ttl_min))
    updated["generated_at_human"] = now.strftime("%Y-%m-%d %H:%M UTC")
    updated["generated_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    updated["expires_at_human"] = expires.strftime("%Y-%m-%d %H:%M UTC")
    updated["expires_at"] = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    updated["pairs"][candidate["pair"]] = candidate["row"]
    base_note = str(updated.get("notes", "")).strip()
    repair_note = candidate["header_note"]
    updated["notes"] = f"{base_note} {repair_note}".strip() if base_note else repair_note
    return normalize_policy_dict(updated, now=now)


def write_policy(policy: dict, policy_md: Path, policy_json: Path) -> None:
    normalized = normalize_policy_dict(policy)
    policy_md.write_text(format_policy_markdown(normalized))
    policy_json.write_text(json.dumps(normalized, indent=2) + "\n")


def force_aggression_override(policy: dict) -> tuple[dict, bool]:
    """Level 3 mode: when account margin is NORMAL (<60%), force ALL pairs to
    BOTH SHARED_MARKET FAST EARLY. This overrides trader's PAUSE writes — the
    brake_gate / inventory_brake / stranded_drain layer is the new safety net,
    so the policy file should NOT be the entry filter.

    Returns (mutated_policy, changed). When margin is past CAUTION, leaves
    policy alone so the trader's defensive writes still apply.

    CRITICAL: If the trader wrote a fresh policy that has not yet expired,
    respect it unconditionally. The aggression override only applies to stale
    (expired) policies — it must never fight a live trader write.
    """
    now = utc_now()
    expires_str = policy.get("expires_at", "")
    if expires_str:
        try:
            expires_at = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
            if expires_at > now:
                return policy, False  # Policy is fresh — skip aggression override
        except Exception:
            pass

    brake_path = Path(__file__).resolve().parent.parent / "logs" / "bot_brake_state.json"
    if not brake_path.exists():
        return policy, False
    try:
        brake = json.loads(brake_path.read_text())
    except Exception:
        return policy, False
    stage = brake.get("margin_stage", "UNKNOWN")
    if stage not in ("NORMAL",):
        return policy, False  # past NORMAL = let trader's conservative policy stand

    pairs = policy.get("pairs", {}) or {}
    changed = False
    aggressive_pair = {
        "mode": "BOTH",
        "allow_market": True,
        "pending": "KEEP",
        "max_pending": 2,
        "ownership": "SHARED_MARKET",
        "tempo": "FAST",
        "entry_bias": "EARLY",
        "note": "Level 3 aggression (auto-forced by policy_guard while margin NORMAL)",
    }
    for pair_name in (
        "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD",
        "EUR_JPY", "GBP_JPY", "AUD_JPY",
        "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
        "NZD_JPY", "CAD_JPY",
        "EUR_CHF", "AUD_NZD", "AUD_CAD",
    ):
        cur = pairs.get(pair_name, {}) or {}
        # Only override if PAUSE or LONG_ONLY/SHORT_ONLY (defensive single-side)
        if cur.get("mode") in ("PAUSE", "LONG_ONLY", "SHORT_ONLY") or not cur:
            pairs[pair_name] = dict(aggressive_pair)
            changed = True
    if changed:
        policy["pairs"] = pairs
        # Bump target so policy_guard isn't satisfied with covered=1
        policy["target_active_worker_pairs"] = max(
            int(policy.get("target_active_worker_pairs", 1)), 4
        )
    return policy, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the repair decision without writing files")
    parser.add_argument("--assume-flat-book", action="store_true", help="Ignore live worker coverage when evaluating starvation")
    parser.add_argument("--ttl-min", type=int, default=DEFAULT_POLICY_TTL_MIN, help="TTL for auto-repaired policy")
    parser.add_argument("--policy-md", type=Path, default=POLICY_MD)
    parser.add_argument("--policy-json", type=Path, default=POLICY_JSON)
    args = parser.parse_args()

    policy, policy_notes, source = load_policy_from_paths(args.policy_md, args.policy_json)
    # Level 3 aggression override — runs BEFORE all other repair logic.
    # When margin is NORMAL, force PAUSE/LONG_ONLY/SHORT_ONLY -> BOTH SHARED_MARKET FAST EARLY.
    policy, aggression_overrode = force_aggression_override(policy)
    if aggression_overrode and not args.dry_run:
        write_policy(policy, args.policy_md, args.policy_json)
        print("AGGRESSION_OVERRIDE: PAUSE/single-side pairs forced to BOTH SHARED_MARKET FAST EARLY (margin NORMAL).")
    token, acct = load_config()
    prices = fetch_prices(token, acct)
    open_pairs = set() if args.assume_flat_book else {
        str(trade.get("instrument", "?"))
        for trade in fetch_open_trades(token, acct)
        if get_tag(trade) in BOT_TAGS
    }
    pending_pairs = set() if args.assume_flat_book else worker_pending_pairs(token, acct)
    covered_pairs = open_pairs | pending_pairs
    target = max(0, int(policy.get("target_active_worker_pairs", 0)))

    existing_md = args.policy_md.read_text() if args.policy_md.exists() else ""
    existing_json = args.policy_json.read_text() if args.policy_json.exists() else ""
    normalized = normalize_policy_dict(policy)
    canonical_md = format_policy_markdown(normalized)
    canonical_json = json.dumps(normalized, indent=2) + "\n"
    needs_normalize = canonical_md != existing_md or canonical_json != existing_json

    print(
        f"BOT POLICY GUARD | source={source} | status={policy['global_status']} | "
        f"target={target} | covered={len(covered_pairs)}"
    )
    if policy_notes:
        print(f"Policy notes: {'; '.join(policy_notes)}")

    if policy["global_status"] != "ACTIVE" or target <= len(covered_pairs):
        if policy["global_status"] != "ACTIVE":
            print("No repair: policy is not ACTIVE.")
        else:
            print("No repair: coverage target already satisfied.")
        if needs_normalize:
            if args.dry_run:
                print("Would normalize policy files to canonical format.")
            else:
                write_policy(normalized, args.policy_md, args.policy_json)
                print("Normalized policy files to canonical format.")
        return 0

    range_opps = scan_ranges(prices)
    trend_opps = scan_trends(prices)
    allowed_range = [opp for opp in range_opps if is_range_allowed(policy, opp, prices)]
    allowed_trend = [opp for opp in trend_opps if is_trend_allowed(policy, opp)]
    print(
        f"Candidates | range={len(range_opps)} allowed_range={len(allowed_range)} | "
        f"trend={len(trend_opps)} allowed_trend={len(allowed_trend)}"
    )

    if allowed_range or allowed_trend:
        print("No repair: at least one current live lane is already permitted by policy.")
        if needs_normalize:
            if args.dry_run:
                print("Would normalize policy files to canonical format.")
            else:
                write_policy(normalized, args.policy_md, args.policy_json)
                print("Normalized policy files to canonical format.")
        return 0

    range_candidates = [
        candidate
        for opp in range_opps
        if (candidate := range_repair_candidate(policy, opp, prices)) is not None
    ]
    trend_candidates = [
        candidate for opp in trend_opps if (candidate := trend_repair_candidate(policy, opp)) is not None
    ]

    best_candidate = None
    if range_candidates:
        best_candidate = max(range_candidates, key=lambda item: item["score"])
    elif trend_candidates:
        best_candidate = max(trend_candidates, key=lambda item: item["score"])

    if best_candidate is None:
        print("No repair: starvation is not caused by policy-only blocking.")
        if needs_normalize:
            if args.dry_run:
                print("Would normalize policy files to canonical format.")
            else:
                write_policy(normalized, args.policy_md, args.policy_json)
                print("Normalized policy files to canonical format.")
        return 0

    repaired = apply_repair(policy, best_candidate, args.ttl_min)
    print(
        f"Repairing {best_candidate['pair']} {best_candidate['direction']} via {best_candidate['kind']} "
        f"lane ({best_candidate['reason']})."
    )
    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    write_policy(repaired, args.policy_md, args.policy_json)
    print(f"Wrote repaired policy: {args.policy_md}")
    print(f"Wrote repaired policy json: {args.policy_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
