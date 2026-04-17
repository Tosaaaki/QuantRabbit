#!/usr/bin/env python3
"""
Inventory Brake — first guardrail in the bot loop.

Two jobs every minute:
  1. Per-pair imbalance: if one bot side accumulates >3x the other,
     block new adds on the heavy side and flag drain_mode.
  2. Account-wide margin: stages NORMAL / CAUTION / EMERGENCY / PANIC,
     halts new entries past CAUTION, force-closes 50% of the heavy
     stranded book at PANIC.

Writes logs/bot_brake_state.json for range_bot / trend_bot / stranded_drain
to consume. This is the kill switch for the "make 15% then give it all back
in a closeout" pattern.

Usage:
    python3 tools/inventory_brake.py
    python3 tools/inventory_brake.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Worktree-aware: resolve to main repo for config/logs writes
_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    _git_common = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip())
    _MAIN_ROOT = _git_common.resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

from range_bot import (
    BOT_TAG, BOT_MARKET_TAG, fetch_account, fetch_open_trades,
    fetch_pending_orders, fetch_prices, get_tag, load_config, signed_units,
    is_entry_pending_order, oanda_api,
)

PAIRS = (
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
    "NZD_JPY", "CAD_JPY",
    "EUR_CHF", "AUD_NZD", "AUD_CAD",
)
BOT_TAGS = {BOT_TAG, BOT_MARKET_TAG, "trend_bot_market"}

BRAKE_STATE_PATH = _MAIN_ROOT / "logs" / "bot_brake_state.json"
LOG_FILE = _MAIN_ROOT / "logs" / "live_trade_log.txt"

# Imbalance thresholds (units of stranded vs winning side)
IMBALANCE_BLOCK_RATIO = 3.0      # stranded:winning >= 3:1 -> block adds on heavy side
IMBALANCE_DRAIN_RATIO = 4.0      # >=4:1 -> drain mode (stranded_drain takes over)
IMBALANCE_PANIC_RATIO = 6.0      # >=6:1 -> panic stage entered if margin also high

# Margin stages (pct of NAV used)
MARGIN_NORMAL_MAX = 0.60
MARGIN_CAUTION_MAX = 0.75
MARGIN_EMERGENCY_MAX = 0.85
# Above EMERGENCY = PANIC

# Panic action: close X% of heaviest stranded side at market
PANIC_DRAIN_FRACTION = 0.50

# Min absolute units to consider "real" inventory (filter dust)
MIN_INVENTORY_UNITS = 500
# Min units + loss on heavy side to flag "stranded" (a fresh single trade isn't a bag)
MIN_STRANDED_UNITS = 6000
MIN_STRANDED_LOSS_JPY = -150


def append_log(line: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(f"[{now}] {line}\n")


def classify_margin_stage(margin_pct: float) -> str:
    if margin_pct < MARGIN_NORMAL_MAX:
        return "NORMAL"
    if margin_pct < MARGIN_CAUTION_MAX:
        return "CAUTION"
    if margin_pct < MARGIN_EMERGENCY_MAX:
        return "EMERGENCY"
    return "PANIC"


def aggregate_pair_inventory(bot_trades: list[dict]) -> dict[str, dict]:
    """Sum bot-tagged units + UPL per pair per side."""
    out: dict[str, dict] = {p: {
        "long_units": 0, "short_units": 0,
        "long_upl": 0.0, "short_upl": 0.0,
        "long_trades": [], "short_trades": [],
    } for p in PAIRS}
    for t in bot_trades:
        pair = t.get("instrument")
        if pair not in out:
            continue
        units = signed_units(t)
        upl = float(t.get("unrealizedPL", 0))
        if units > 0:
            out[pair]["long_units"] += units
            out[pair]["long_upl"] += upl
            out[pair]["long_trades"].append(t)
        elif units < 0:
            out[pair]["short_units"] += abs(units)
            out[pair]["short_upl"] += upl
            out[pair]["short_trades"].append(t)
    return out


def compute_pair_brake(pair: str, inv: dict, margin_stage: str) -> dict:
    long_u = inv["long_units"]
    short_u = inv["short_units"]
    long_upl = inv["long_upl"]
    short_upl = inv["short_upl"]

    block_long = False
    block_short = False
    drain_mode = False
    stranded_side = None
    ratio: float | None = None
    reason = ""

    # Skip dust
    if max(long_u, short_u) < MIN_INVENTORY_UNITS:
        return {
            "long_units": long_u, "short_units": short_u,
            "long_upl": round(long_upl, 0), "short_upl": round(short_upl, 0),
            "imbalance_ratio": None, "stranded_side": None,
            "block_long_new": False, "block_short_new": False,
            "drain_mode": False, "reason": "dust",
        }

    # Identify stranded side: must be heavy AND losing meaningfully (not fresh single trade)
    def _is_stranded(units: int, upl: float) -> bool:
        return units >= MIN_STRANDED_UNITS and upl <= MIN_STRANDED_LOSS_JPY

    if long_u == 0 and short_u >= MIN_INVENTORY_UNITS:
        stranded_side = "SHORT" if _is_stranded(short_u, short_upl) else None
        ratio = float("inf")
    elif short_u == 0 and long_u >= MIN_INVENTORY_UNITS:
        stranded_side = "LONG" if _is_stranded(long_u, long_upl) else None
        ratio = float("inf")
    else:
        if long_u > short_u:
            ratio = long_u / max(short_u, 1)
            heavy = "LONG"
            heavy_upl = long_upl
            heavy_units = long_u
        else:
            ratio = short_u / max(long_u, 1)
            heavy = "SHORT"
            heavy_upl = short_upl
            heavy_units = short_u
        if _is_stranded(heavy_units, heavy_upl):
            stranded_side = heavy

    # Apply blocks
    if ratio is not None and stranded_side and ratio >= IMBALANCE_BLOCK_RATIO:
        if stranded_side == "LONG":
            block_long = True
            reason = f"imbalance LONG/SHORT={ratio:.1f} stranded LONG upl={long_upl:+.0f}"
        else:
            block_short = True
            reason = f"imbalance SHORT/LONG={ratio:.1f} stranded SHORT upl={short_upl:+.0f}"

    if ratio is not None and stranded_side and ratio >= IMBALANCE_DRAIN_RATIO:
        drain_mode = True

    # Margin overlay: CAUTION+ blocks all new on the heavy side regardless of ratio
    if margin_stage in ("CAUTION", "EMERGENCY", "PANIC") and stranded_side:
        if stranded_side == "LONG":
            block_long = True
        else:
            block_short = True
        if margin_stage in ("EMERGENCY", "PANIC"):
            drain_mode = True

    return {
        "long_units": long_u, "short_units": short_u,
        "long_upl": round(long_upl, 0), "short_upl": round(short_upl, 0),
        "imbalance_ratio": (round(ratio, 2) if ratio not in (None, float("inf")) else ratio),
        "stranded_side": stranded_side,
        "block_long_new": block_long, "block_short_new": block_short,
        "drain_mode": drain_mode, "reason": reason,
    }


def panic_drain(token: str, acct: str, brake_pairs: dict, prices: dict, dry_run: bool) -> list[str]:
    """At PANIC margin: close PANIC_DRAIN_FRACTION of largest stranded inventory at market.

    This is the 'don't get closeout-killed' last-resort. Better to take a -X loss
    we control than have OANDA pick the worst price.
    """
    actions = []
    # Find pair with largest stranded units * |upl|
    candidates = []
    for pair, b in brake_pairs.items():
        if not b.get("stranded_side"):
            continue
        side = b["stranded_side"]
        units = b["long_units"] if side == "LONG" else b["short_units"]
        upl = b["long_upl"] if side == "LONG" else b["short_upl"]
        if units < MIN_INVENTORY_UNITS:
            continue
        candidates.append((pair, side, units, upl, abs(upl) * units))

    if not candidates:
        return ["panic_drain: no stranded inventory found"]

    candidates.sort(key=lambda c: -c[4])  # largest pain first
    pair, side, units, upl, _ = candidates[0]

    # Get the trades for that side, close PANIC_DRAIN_FRACTION proportionally
    open_trades = fetch_open_trades(token, acct)
    bot_trades = [t for t in open_trades
                  if t.get("instrument") == pair and get_tag(t) in BOT_TAGS]
    target_trades = [t for t in bot_trades
                     if (signed_units(t) > 0 and side == "LONG")
                     or (signed_units(t) < 0 and side == "SHORT")]
    if not target_trades:
        return [f"panic_drain: {pair} {side} no bot trades found"]

    target_units_close = int(units * PANIC_DRAIN_FRACTION)
    closed_so_far = 0
    for t in sorted(target_trades, key=lambda x: float(x.get("unrealizedPL", 0))):  # worst loss first
        if closed_so_far >= target_units_close:
            break
        trade_id = t.get("id")
        trade_units = abs(signed_units(t))
        close_units = min(trade_units, target_units_close - closed_so_far)
        cmd = [
            sys.executable, str(_MAIN_ROOT / "tools" / "close_trade.py"),
            str(trade_id), str(close_units),
            "--reason", "panic_margin",
            "--auto-log", "--auto-slack",
            "--force-worker-close",
        ]
        if dry_run:
            actions.append(f"[DRY] PANIC close {pair} {side} {close_units}u (id={trade_id})")
            closed_so_far += close_units
        else:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                ok = proc.returncode == 0
                actions.append(f"PANIC close {pair} {side} {close_units}u id={trade_id} ok={ok}")
                if ok:
                    closed_so_far += close_units
                    append_log(f"INVENTORY_BRAKE PANIC closed {pair} {side} {close_units}u id={trade_id}")
            except Exception as e:
                actions.append(f"PANIC close FAILED {pair} {side} id={trade_id}: {e}")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory Brake")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== INVENTORY BRAKE === {now_utc.strftime('%Y-%m-%d %H:%M:%SZ')}")

    token, acct = load_config()
    account = fetch_account(token, acct)
    prices = fetch_prices(token, acct)

    nav = account.get("nav", 0)
    margin_used = account.get("margin_used", 0)
    margin_pct = (margin_used / nav) if nav > 0 else 0
    margin_stage = classify_margin_stage(margin_pct)

    open_trades = fetch_open_trades(token, acct)
    bot_trades = [t for t in open_trades if get_tag(t) in BOT_TAGS]

    inventory = aggregate_pair_inventory(bot_trades)
    brake_pairs = {p: compute_pair_brake(p, inv, margin_stage)
                   for p, inv in inventory.items()}

    # Global flags
    global_halt_new = margin_stage in ("CAUTION", "EMERGENCY", "PANIC")
    panic_drain_active = margin_stage == "PANIC"

    panic_actions: list[str] = []
    if panic_drain_active:
        panic_actions = panic_drain(token, acct, brake_pairs, prices, args.dry_run)

    state = {
        "ts": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nav": round(nav, 0),
        "margin_used": round(margin_used, 0),
        "margin_pct": round(margin_pct, 4),
        "margin_stage": margin_stage,
        "global_halt_new": global_halt_new,
        "panic_drain_active": panic_drain_active,
        "panic_actions": panic_actions,
        "thresholds": {
            "imbalance_block_ratio": IMBALANCE_BLOCK_RATIO,
            "imbalance_drain_ratio": IMBALANCE_DRAIN_RATIO,
            "margin_normal_max": MARGIN_NORMAL_MAX,
            "margin_caution_max": MARGIN_CAUTION_MAX,
            "margin_emergency_max": MARGIN_EMERGENCY_MAX,
        },
        "pairs": brake_pairs,
    }

    if not args.dry_run:
        BRAKE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BRAKE_STATE_PATH.write_text(json.dumps(state, indent=2))

    print(f"Margin: {margin_pct*100:.1f}% ({margin_stage}) | NAV={nav:.0f} used={margin_used:.0f}")
    if global_halt_new:
        print(f"  -> GLOBAL HALT_NEW (stage {margin_stage})")
    if panic_drain_active:
        print(f"  -> PANIC DRAIN active. Actions:")
        for a in panic_actions:
            print(f"     {a}")

    flagged = [p for p, b in brake_pairs.items() if b.get("block_long_new") or b.get("block_short_new") or b.get("drain_mode")]
    if flagged:
        print(f"Pair brakes: {len(flagged)} flagged")
        for p in flagged:
            b = brake_pairs[p]
            tags = []
            if b.get("block_long_new"): tags.append("BLK_LONG")
            if b.get("block_short_new"): tags.append("BLK_SHORT")
            if b.get("drain_mode"): tags.append("DRAIN")
            print(f"  {p:8} L={b['long_units']:>6} S={b['short_units']:>6} "
                  f"upl(L/S)={b['long_upl']:+.0f}/{b['short_upl']:+.0f} "
                  f"ratio={b.get('imbalance_ratio')} {' '.join(tags)} {b.get('reason','')}")
    else:
        print("Pair brakes: clean")

    print(f"State -> {BRAKE_STATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
