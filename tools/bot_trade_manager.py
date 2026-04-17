#!/usr/bin/env python3
"""
Bot Trade Manager — fast protection layer for bot-tagged orders and trades.

Purpose:
- Keep fast bot entry possible without letting deadlocked books drift into closeout
- Manage pending bot orders and bot-tagged open trades every minute
- Leave higher-order discretion to the trader task

Usage:
    python3 tools/bot_trade_manager.py
    python3 tools/bot_trade_manager.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tools"))

from market_state import get_market_state
from bot_policy import get_pair_policy, load_policy
from range_bot import (
    BOT_MARKET_TAG,
    BOT_TAG,
    GTD_HOURS,
    LEVERAGE,
    cancel_order,
    estimate_margin,
    fetch_prices,
    format_price,
    is_entry_pending_order,
    load_config,
    load_technicals,
    oanda_api,
    parse_oanda_time,
    to_pips,
)
from worker_target_race import (
    get_trade_state,
    plan_from_payload,
    prune_state,
    remember_trade_plan,
    update_trade_state,
)
from oanda_trade_tags import enrich_open_trades

BOT_TAGS = {BOT_TAG, BOT_MARKET_TAG, "trend_bot_market"}
LOG_FILE = ROOT / "logs" / "live_trade_log.txt"

HARD_PROJECTED_MARGIN_PCT = 0.90
RELEASE_MARGIN_PCT = 0.78
DEADLOCK_GROSS_MARGIN_PCT = 0.28
DEADLOCK_BALANCE_MAX_GAP = 0.35
STALE_PENDING_MIN = 45
TRAP_HOLD_MIN = 25
FULL_CLOSE_HOLD_MIN = 60
TRAP_BREAK_ATR_FRACTION = 0.25
TRAP_PROGRESS_MIN = 0.15
RELIEF_PARTIAL_FRACTION = 0.50
MAX_RELIEF_ACTIONS = 3

MICRO_MARKET_STALE_MIN = 4    # was 6 — resolve MICRO faster
MICRO_MARKET_FULL_CLOSE_MIN = 7  # was 10
FAST_MARKET_STALE_MIN = 8    # was 10
FAST_MARKET_FULL_CLOSE_MIN = 13  # was 16
BALANCED_MARKET_STALE_MIN = 16
BALANCED_MARKET_FULL_CLOSE_MIN = 26
MICRO_PASSIVE_STALE_MIN = 8   # was 10
MICRO_PASSIVE_FULL_CLOSE_MIN = 13  # was 16
FAST_PASSIVE_STALE_MIN = 12   # was 14
FAST_PASSIVE_FULL_CLOSE_MIN = 19  # was 22
BALANCED_PASSIVE_STALE_MIN = 20
BALANCED_PASSIVE_FULL_CLOSE_MIN = 32
MICRO_MIN_PROGRESS = 0.20
FAST_MIN_PROGRESS = 0.30
BALANCED_MIN_PROGRESS = 0.25

# range_bot-specific profile: TP = BB mid. H1 range = 60-120 min to fill.
# Generic PASSIVE timeout (32 min) force-closes before TP reachable. Diagnosis
# 2026-04-17: 5/5 range_bot trades closed in 1.4-14.9 min, TP reached 0 times.
# R:R dropped to 0.04 (wins +6 JPY, losses -137 JPY) because wins never realized.
RANGE_BOT_STALE_MIN = 75
RANGE_BOT_FULL_CLOSE_MIN = 150
RANGE_BOT_MIN_PROGRESS = 0.35


def append_log(line: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with open(LOG_FILE, "a") as fh:
        fh.write(f"[{now}] {line}\n")


def fetch_account_summary(token: str, acct: str) -> dict:
    data = oanda_api(f"/v3/accounts/{acct}/summary", token, acct)
    account = data.get("account", {})
    nav = float(account.get("NAV", 0))
    margin_used = float(account.get("marginUsed", 0))
    margin_closeout_pct = float(account.get("marginCloseoutPercent", 0))
    return {
        "nav": nav,
        "balance": float(account.get("balance", 0)),
        "margin_used": margin_used,
        "margin_available": float(account.get("marginAvailable", 0)),
        "margin_pct": (margin_used / nav) if nav > 0 else 1.0,
        "margin_closeout_pct": margin_closeout_pct,
        "health_buffer": max(0.0, 1.0 - margin_closeout_pct) if margin_closeout_pct > 0 else None,
    }


def get_tag(payload: dict) -> str:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = payload.get(key, {}) or {}
        tag = ext.get("tag")
        if tag:
            return str(tag)
    return ""


def fetch_open_trades(token: str, acct: str) -> list[dict]:
    data = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
    trades = data.get("trades", [])
    return enrich_open_trades(
        trades,
        lambda path: oanda_api(path.format(account_id=acct), token, acct),
    )


def fetch_pending_orders(token: str, acct: str) -> list[dict]:
    data = oanda_api(f"/v3/accounts/{acct}/pendingOrders", token, acct)
    return data.get("orders", [])


def trade_side(trade: dict) -> str:
    return "LONG" if int(trade.get("currentUnits", 0)) > 0 else "SHORT"


def trade_direction(trade: dict) -> str:
    return "BUY" if trade_side(trade) == "LONG" else "SELL"


def current_exit_price(trade: dict, prices: dict) -> float:
    pair = trade.get("instrument", "")
    quote = prices.get(pair, {})
    if trade_side(trade) == "LONG":
        return float(quote.get("bid", 0))
    return float(quote.get("ask", 0))


def pending_margin(order: dict, prices: dict) -> float:
    pair = order.get("instrument", "")
    units = abs(int(order.get("units", 0)))
    price = float(order.get("price", 0))
    if price <= 0:
        quote = prices.get(pair, {})
        price = float(quote.get("ask") or quote.get("bid") or quote.get("mid") or 0)
    return estimate_margin(units, price, pair) if price > 0 and units > 0 else 0.0


def trade_margin(trade: dict, prices: dict) -> float:
    pair = trade.get("instrument", "")
    units = abs(int(trade.get("currentUnits", 0)))
    price = current_exit_price(trade, prices) or float(trade.get("price", 0))
    return estimate_margin(units, price, pair) if price > 0 and units > 0 else 0.0


def scalp_timeout_profile(tag: str, tempo: str) -> dict:
    tempo = str(tempo or "BALANCED").upper()
    # range_bot: slow mean-reversion to BB mid. Never use scalp timeouts.
    if tag in {"range_bot", "range_bot_market"}:
        return {
            "stale_min": RANGE_BOT_STALE_MIN,
            "full_close_min": RANGE_BOT_FULL_CLOSE_MIN,
            "min_progress": RANGE_BOT_MIN_PROGRESS,
        }
    market_tag = tag in {BOT_MARKET_TAG, "trend_bot_market"}
    if market_tag:
        if tempo == "MICRO":
            return {
                "stale_min": MICRO_MARKET_STALE_MIN,
                "full_close_min": MICRO_MARKET_FULL_CLOSE_MIN,
                "min_progress": MICRO_MIN_PROGRESS,
            }
        if tempo == "FAST":
            return {
                "stale_min": FAST_MARKET_STALE_MIN,
                "full_close_min": FAST_MARKET_FULL_CLOSE_MIN,
                "min_progress": FAST_MIN_PROGRESS,
            }
        return {
            "stale_min": BALANCED_MARKET_STALE_MIN,
            "full_close_min": BALANCED_MARKET_FULL_CLOSE_MIN,
            "min_progress": BALANCED_MIN_PROGRESS,
        }
    if tempo == "MICRO":
        return {
            "stale_min": MICRO_PASSIVE_STALE_MIN,
            "full_close_min": MICRO_PASSIVE_FULL_CLOSE_MIN,
            "min_progress": MICRO_MIN_PROGRESS,
        }
    if tempo == "FAST":
        return {
            "stale_min": FAST_PASSIVE_STALE_MIN,
            "full_close_min": FAST_PASSIVE_FULL_CLOSE_MIN,
            "min_progress": FAST_MIN_PROGRESS,
        }
    return {
        "stale_min": BALANCED_PASSIVE_STALE_MIN,
        "full_close_min": BALANCED_PASSIVE_FULL_CLOSE_MIN,
        "min_progress": BALANCED_MIN_PROGRESS,
    }


def tp_price(trade: dict) -> float | None:
    for key in ("takeProfitOrder", "takeProfitOnFill"):
        payload = trade.get(key, {}) or {}
        value = payload.get("price")
        if value is not None:
            return float(value)
    return None


def hold_minutes(trade: dict, now_utc: datetime) -> float:
    opened_at = parse_oanda_time(trade.get("openTime"))
    if opened_at is None:
        return 0.0
    return max(0.0, (now_utc - opened_at).total_seconds() / 60)


def tp_progress(trade: dict, prices: dict) -> float:
    target = tp_price(trade)
    if not target:
        return 0.0

    entry = float(trade.get("price", 0))
    current = current_exit_price(trade, prices)
    if entry <= 0 or current <= 0:
        return 0.0

    if trade_side(trade) == "LONG":
        distance = target - entry
        covered = current - entry
    else:
        distance = entry - target
        covered = entry - current

    if distance <= 0:
        return 0.0
    return max(0.0, min(1.5, covered / distance))


def target_progress(trade: dict, prices: dict, target: float | None) -> float:
    if not target:
        return 0.0

    entry = float(trade.get("price", 0))
    current = current_exit_price(trade, prices)
    if entry <= 0 or current <= 0:
        return 0.0

    if trade_side(trade) == "LONG":
        distance = target - entry
        covered = current - entry
    else:
        distance = entry - target
        covered = entry - current
    if distance <= 0:
        return 0.0
    return max(0.0, min(2.0, covered / distance))


def target_reached(trade: dict, prices: dict, target: float | None) -> bool:
    if not target:
        return False
    current = current_exit_price(trade, prices)
    if current <= 0:
        return False
    if trade_side(trade) == "LONG":
        return current >= target
    return current <= target


def apply_trade_orders(
    token: str,
    acct: str,
    trade_id: str,
    pair: str,
    *,
    take_profit: float | None,
    stop_loss: float | None,
    dry_run: bool,
) -> tuple[bool, str]:
    payload: dict[str, dict[str, str]] = {}
    if take_profit is not None:
        payload["takeProfit"] = {"price": format_price(float(take_profit), pair), "timeInForce": "GTC"}
    if stop_loss is not None:
        payload["stopLoss"] = {"price": format_price(float(stop_loss), pair), "timeInForce": "GTC"}
    if not payload:
        return True, "no order update"
    if dry_run:
        return True, f"[DRY] trade#{trade_id} orders -> {payload}"
    try:
        oanda_api(
            f"/v3/accounts/{acct}/trades/{trade_id}/orders",
            token,
            acct,
            method="PUT",
            data=payload,
        )
        return True, "updated"
    except Exception as exc:
        return False, str(exc)


def notify_runner_update(pair: str, action: str, units: int, price: float, note: str, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "slack_trade_notify.py"),
        "modify",
        "--pair",
        pair,
        "--action",
        action,
        "--units",
        str(units),
        "--price",
        str(price),
        "--pl",
        "runner",
        "--note",
        note,
    ]
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass


def evaluate_trade(
    trade: dict,
    prices: dict,
    now_utc: datetime,
    policy_row: dict | None = None,
) -> dict:
    pair = trade.get("instrument", "")
    side = trade_side(trade)
    direction = trade_direction(trade)
    tag = get_tag(trade)
    tempo = str((policy_row or {}).get("tempo", "BALANCED")).upper()
    profile = scalp_timeout_profile(tag, tempo)
    current_price = current_exit_price(trade, prices)
    entry = float(trade.get("price", 0))
    upl = float(trade.get("unrealizedPL", 0))
    hold_min = hold_minutes(trade, now_utc)
    trade_id = str(trade.get("id", "?"))
    runner_plan = plan_from_payload(trade, pair, direction)
    runner_state = get_trade_state(trade_id)
    if runner_plan:
        runner_state = remember_trade_plan(
            trade_id,
            {**runner_plan, "entry": entry, "stop": tp_price(trade) or entry},
            abs(int(trade.get("currentUnits", 0))),
        )
    stored_plan = runner_state.get("plan") if runner_state else None
    runner_plan = stored_plan or runner_plan
    tp1_done = bool(runner_state.get("tp1_done", False))
    runner_orders_set = bool(runner_state.get("runner_orders_set", False))
    active_target = (
        (runner_plan or {}).get("tp2")
        if tp1_done
        else (runner_plan or {}).get("tp1")
    )
    progress = target_progress(trade, prices, active_target) if active_target else tp_progress(trade, prices)

    tfs = load_technicals(pair)
    m5 = tfs.get("M5", {})
    atr_pips = max(float(m5.get("atr_pips", 0)), 1.0)
    adx = float(m5.get("adx", 0))
    ema_slope = float(m5.get("ema_slope_5", 0))
    macd_hist = float(m5.get("macd_hist", 0))
    bb_lower = float(m5.get("bb_lower", 0))
    bb_upper = float(m5.get("bb_upper", 0))

    adverse_trend = False
    break_buffer = 0.0
    trap_reason = ""
    if pair:
        break_buffer = atr_pips * TRAP_BREAK_ATR_FRACTION
        if side == "LONG":
            adverse_trend = ema_slope < 0 and macd_hist < 0
            if bb_lower > 0 and current_price > 0 and to_pips(bb_lower - current_price, pair) > break_buffer and adx >= 28:
                trap_reason = f"range broke below lower band by {to_pips(bb_lower - current_price, pair):.1f}pip"
        else:
            adverse_trend = ema_slope > 0 and macd_hist > 0
            if bb_upper > 0 and current_price > 0 and to_pips(current_price - bb_upper, pair) > break_buffer and adx >= 28:
                trap_reason = f"range broke above upper band by {to_pips(current_price - bb_upper, pair):.1f}pip"

    runner_active = bool(runner_plan) and tp1_done
    stale = (
        hold_min >= profile["stale_min"]
        and progress < profile["min_progress"]
        and upl <= 0
    )
    # Vol-aware timeout: don't force-close just because time elapsed.
    # Require either meaningful adverse move (>= 0.5 ATR against entry)
    # or non-trivial loss in JPY. Otherwise the trade still has a chance
    # in a low-volatility tape — closing it is just paying spread for nothing.
    adverse_pips = 0.0
    if entry > 0 and current_price > 0:
        if side == "LONG":
            adverse_pips = max(0.0, to_pips(entry - current_price, pair))
        else:
            adverse_pips = max(0.0, to_pips(current_price - entry, pair))
    meaningful_adverse = adverse_pips >= atr_pips * 0.5
    real_loss_jpy = upl <= -100  # ~10 pip loss on small lot
    timed_out_raw = hold_min >= profile["full_close_min"]
    timed_out = timed_out_raw and (meaningful_adverse or real_loss_jpy or bool(trap_reason))
    trapped = bool(trap_reason) or (stale and adverse_trend)
    if runner_active:
        stale = False
        timed_out = False
        trapped = False
    needs_cleanup = timed_out or trapped

    priority = 0.0
    if timed_out:
        priority += 160
    if trapped:
        priority += 120
    if stale:
        priority += 40
    if upl < 0:
        priority += min(abs(upl) / 25.0, 40.0)
    priority += min(hold_min / 10.0, 20.0)

    return {
        "trade": trade,
        "pair": pair,
        "tag": tag,
        "tempo": tempo,
        "side": side,
        "direction": direction,
        "entry": entry,
        "current_price": current_price,
        "upl": upl,
        "hold_min": hold_min,
        "progress": progress,
        "trade_id": trade_id,
        "runner_plan": runner_plan,
        "runner_state": runner_state,
        "runner_tp1_done": tp1_done,
        "runner_orders_set": runner_orders_set,
        "runner_active": runner_active,
        "tp1_reached": target_reached(trade, prices, (runner_plan or {}).get("tp1")),
        "timed_out": timed_out,
        "trapped": trapped,
        "stale": stale,
        "needs_cleanup": needs_cleanup,
        "trap_reason": trap_reason or ("stale loser in adverse momentum" if trapped else ""),
        "priority": priority,
        "margin": trade_margin(trade, prices),
        "scalp_profile": profile,
        "scalp_state": (
            "RUNNER"
            if runner_active
            else "TIMED_OUT"
            if timed_out
            else "TRAPPED"
            if trapped
            else "STALE"
            if stale
            else "LIVE"
        ),
    }


def deadlock_profile(trades: list[dict], prices: dict, nav: float) -> dict:
    long_margin = 0.0
    short_margin = 0.0
    for trade in trades:
        margin = trade_margin(trade, prices)
        if trade_side(trade) == "LONG":
            long_margin += margin
        else:
            short_margin += margin
    gross = long_margin + short_margin
    gross_pct = (gross / nav) if nav > 0 else 0.0
    balance_gap = abs(long_margin - short_margin) / gross if gross > 0 else 1.0
    return {
        "long_margin": long_margin,
        "short_margin": short_margin,
        "gross_margin": gross,
        "gross_pct": gross_pct,
        "balance_gap": balance_gap,
        "deadlocked": gross_pct >= DEADLOCK_GROSS_MARGIN_PCT and balance_gap <= DEADLOCK_BALANCE_MAX_GAP,
    }


def close_via_script(
    trade_id: str,
    units: int,
    reason: str,
    dry_run: bool,
    *,
    force_worker_close: bool = False,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "close_trade.py"),
        str(trade_id),
        str(units),
        "--reason",
        reason,
        "--auto-log",
        "--auto-slack",
    ]
    if force_worker_close:
        cmd.append("--force-worker-close")
    if dry_run:
        return True, f"[DRY] {' '.join(cmd)}"

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot Trade Manager")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying trades/orders")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    state, reason = get_market_state(now_utc)
    print(f"=== BOT TRADE MANAGER === {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Market: {state} ({reason})")
    if state == "CLOSED":
        return 1

    token, acct = load_config()
    policy, policy_notes = load_policy()
    account = fetch_account_summary(token, acct)
    prices = fetch_prices(token, acct)
    open_trades = fetch_open_trades(token, acct)
    pending_orders = fetch_pending_orders(token, acct)

    bot_trades = [trade for trade in open_trades if get_tag(trade) in BOT_TAGS]
    bot_pending = [
        order for order in pending_orders
        if get_tag(order) == BOT_TAG and is_entry_pending_order(order)
    ]
    prune_state({str(trade.get("id")) for trade in bot_trades if trade.get("id")})
    projected_margin = account["margin_used"] + sum(
        pending_margin(order, prices) for order in pending_orders if is_entry_pending_order(order)
    )
    projected_pct = (projected_margin / account["nav"]) if account["nav"] > 0 else 1.0
    deadlock = deadlock_profile(bot_trades, prices, account["nav"])
    projected_cap = float(policy["projected_margin_cap"])
    panic_cap = float(policy["panic_margin_cap"])
    release_cap = float(policy["release_margin_cap"])
    stale_pending_min = max(STALE_PENDING_MIN, int(policy.get("max_pending_age_min", STALE_PENDING_MIN)))

    print(
        f"NAV: {account['nav']:,.0f} JPY | Margin: {account['margin_pct']*100:.1f}% "
        f"| Projected: {projected_pct*100:.1f}% | Bot trades: {len(bot_trades)} | Bot pending: {len(bot_pending)}"
    )
    print(
        f"Policy: {policy['global_status']} | projected_cap={projected_cap*100:.1f}% "
        f"| panic_cap={panic_cap*100:.1f}% | release_cap={release_cap*100:.1f}%"
    )
    if policy_notes:
        print(f"Policy notes: {'; '.join(policy_notes)}")
    if deadlock["gross_margin"] > 0:
        print(
            f"Bot gross margin: {deadlock['gross_margin']:,.0f} JPY ({deadlock['gross_pct']*100:.1f}%) "
            f"| balance gap={deadlock['balance_gap']:.2f}"
        )

    actions = 0
    cancelled = 0
    closed = 0
    skipped: list[str] = []

    # --- Pending cleanup ---
    for order in bot_pending:
        oid = str(order.get("id", "?"))
        pair = order.get("instrument", "?")
        pair_policy = get_pair_policy(policy, pair)
        created = parse_oanda_time(order.get("createTime"))
        age_min = (now_utc - created).total_seconds() / 60 if created else 999
        reason_note = ""
        if state == "ROLLOVER":
            reason_note = "rollover pending cleanup"
        elif policy["global_status"] == "PAUSE_ALL":
            reason_note = "policy pause-all"
        elif pair_policy["pending"] == "CANCEL" or pair_policy["mode"] == "PAUSE":
            reason_note = f"policy {pair_policy['mode']} pending={pair_policy['pending']}"
        elif age_min > stale_pending_min:
            reason_note = f"stale backup {age_min:.0f}m"
        elif projected_pct > panic_cap:
            reason_note = f"projected margin {projected_pct*100:.1f}% > {panic_cap*100:.0f}%"
        elif pair in {trade.get('instrument') for trade in bot_trades}:
            reason_note = "pair already filled by bot"

        if not reason_note:
            continue

        print(f"Pending cancel: {pair} id={oid} | {reason_note}")
        append_log(f"BOT_MANAGER_CANCEL_PENDING {pair} id={oid} tag={BOT_TAG} reason={reason_note}")
        if args.dry_run:
            cancelled += 1
        else:
            if cancel_order(token, acct, oid):
                cancelled += 1
                projected_margin -= pending_margin(order, prices)
                projected_pct = (projected_margin / account["nav"]) if account["nav"] > 0 else 1.0
        actions += 1

    # --- Trade relief / trap cleanup ---
    reviews = sorted(
        (
            evaluate_trade(
                trade,
                prices,
                now_utc,
                get_pair_policy(policy, trade.get("instrument", "")),
            )
            for trade in bot_trades
        ),
        key=lambda item: item["priority"],
        reverse=True,
    )
    cleanup_trade_ids: set[str] = set()
    runner_trade_ids: set[str] = set()

    # --- TP1 partial -> runner promotion ---
    for review in reviews:
        runner_plan = review.get("runner_plan") or {}
        if not runner_plan:
            continue

        trade = review["trade"]
        trade_id = review["trade_id"]
        pair = review["pair"]
        units = abs(int(trade.get("currentUnits", 0)))

        tp1_done = bool(review.get("runner_tp1_done"))
        runner_orders_set = bool(review.get("runner_orders_set"))
        if not tp1_done and not review.get("tp1_reached"):
            continue
        if tp1_done and runner_orders_set:
            continue

        if not tp1_done:
            if units <= 1:
                continue
            partial_units = max(1, int(round(units * float(runner_plan.get("partial_fraction", RELIEF_PARTIAL_FRACTION)))))
            if partial_units >= units:
                partial_units = units - 1
            if partial_units <= 0:
                continue
            print(
                f"Runner promote: {pair} {review['side']} id={trade_id} close {partial_units}/{units}u "
                f"@{format_price(review['current_price'], pair)} | TP1={runner_plan.get('tp1')} reached"
            )
            ok, output = close_via_script(trade_id, partial_units, "bot_half_tp", args.dry_run)
            if not ok:
                skipped.append(f"{pair}: half-tp failed ({output[:80]})")
                continue
            closed += 1
            actions += 1
            update_trade_state(
                trade_id,
                tp1_done=True,
                runner_orders_set=False,
                partial_units=partial_units,
            )
            remaining_units = units - partial_units
            note = (
                f"remaining {remaining_units}u | TP2={runner_plan.get('tp2')} "
                f"| hold>{runner_plan.get('hold_boundary')}"
            )
            append_log(
                f"BOT_MANAGER_HALF_TP {pair} id={trade_id} close={partial_units}u "
                f"remaining={remaining_units}u tp1={runner_plan.get('tp1')} "
                f"tp2={runner_plan.get('tp2')} hold_boundary={runner_plan.get('hold_boundary')}"
            )
            notify_runner_update(pair, "HALF TP -> RUNNER", partial_units, review["current_price"], note, args.dry_run)
            runner_trade_ids.add(trade_id)

        ok, output = apply_trade_orders(
            token,
            acct,
            trade_id,
            pair,
            take_profit=runner_plan.get("tp2"),
            stop_loss=runner_plan.get("hold_boundary"),
            dry_run=args.dry_run,
        )
        if ok:
            update_trade_state(trade_id, runner_orders_set=True)
            append_log(
                f"BOT_MANAGER_RUNNER_SET {pair} id={trade_id} tp2={runner_plan.get('tp2')} "
                f"sl={runner_plan.get('hold_boundary')}"
            )
            runner_trade_ids.add(trade_id)
        else:
            skipped.append(f"{pair}: runner orders failed ({output[:80]})")

    for review in reviews:
        if actions >= MAX_RELIEF_ACTIONS:
            break
        if not review["needs_cleanup"]:
            continue
        if review["trade_id"] in runner_trade_ids:
            continue

        trade = review["trade"]
        trade_id = str(trade.get("id", "?"))
        pair = review["pair"]
        units = abs(int(trade.get("currentUnits", 0)))
        if units <= 0:
            continue

        if review["timed_out"]:
            reason_note = "bot_scalp_timeout"
        elif review["trapped"]:
            reason_note = "bot_trap_cleanup"
        else:
            reason_note = "bot_scalp_timeout"

        profile = review["scalp_profile"]
        print(
            f"Scalp cleanup: {pair} {review['side']} id={trade_id} close {units}u "
            f"@{format_price(review['current_price'], pair)} | state={review['scalp_state']} "
            f"| hold={review['hold_min']:.0f}m | progress={review['progress']:.0%} "
            f"| timeout={profile['stale_min']}/{profile['full_close_min']}m"
        )
        ok, output = close_via_script(
            trade_id,
            units,
            reason_note,
            args.dry_run,
        )
        if ok:
            closed += 1
            actions += 1
            cleanup_trade_ids.add(trade_id)
            projected_margin -= estimate_margin(units, review["current_price"], pair)
            projected_pct = (projected_margin / account["nav"]) if account["nav"] > 0 else 1.0
        else:
            skipped.append(f"{pair}: cleanup failed ({output[:80]})")

    forced_relief = (
        projected_pct > panic_cap
        or account["margin_closeout_pct"] >= 0.90
        or account["health_buffer"] is not None and account["health_buffer"] <= 0.10
    )
    deadlock_relief = deadlock["deadlocked"] and projected_pct > max(projected_cap + 0.03, 0.84)
    relief_needed = forced_relief or deadlock_relief
    relief_target = release_cap if forced_relief else max(projected_cap - 0.02, release_cap)
    deadlock_relief_done = False

    for review in reviews:
        if actions >= MAX_RELIEF_ACTIONS and forced_relief:
            break

        if not relief_needed:
            break
        if projected_pct <= relief_target and not (deadlock_relief and review["upl"] < 0 and not deadlock_relief_done):
            break

        if not forced_relief and not (deadlock_relief and review["upl"] < 0):
            continue

        trade = review["trade"]
        trade_id = str(trade.get("id", "?"))
        if trade_id in cleanup_trade_ids:
            continue
        pair = review["pair"]
        units = abs(int(trade.get("currentUnits", 0)))
        if units <= 0:
            continue

        close_units = units
        reason_note = "deadlock_relief" if deadlock_relief else "panic_margin"
        if not forced_relief and review["hold_min"] < FULL_CLOSE_HOLD_MIN:
            close_units = max(1, int(round(units * RELIEF_PARTIAL_FRACTION)))
            reason_note = "deadlock_relief"
        elif forced_relief and projected_pct <= panic_cap + 0.03 and units > 1:
            close_units = max(1, int(round(units * RELIEF_PARTIAL_FRACTION)))
            reason_note = "panic_margin"

        print(
            f"Trade relief: {pair} {review['side']} id={trade.get('id')} close {close_units}/{units}u "
            f"@{format_price(review['current_price'], pair)} | UPL={review['upl']:+.0f} | {reason_note}"
        )
        ok, output = close_via_script(
            str(trade.get("id")),
            close_units,
            reason_note,
            args.dry_run,
            force_worker_close=True,
        )
        if ok:
            closed += 1
            actions += 1
            projected_margin -= estimate_margin(close_units, review["current_price"], pair)
            projected_pct = (projected_margin / account["nav"]) if account["nav"] > 0 else 1.0
            if deadlock["deadlocked"]:
                deadlock_relief_done = True
        else:
            skipped.append(f"{pair}: close failed ({output[:80]})")

    print(f"\n{'='*50}")
    print("BOT TRADE MANAGER SUMMARY")
    print(f"  Margin now: {account['margin_pct']*100:.1f}%")
    print(f"  Projected after actions: {projected_pct*100:.1f}%")
    print(f"  Pending cancelled: {cancelled}")
    print(f"  Trades reduced/closed: {closed}")
    if deadlock["deadlocked"]:
        print(
            f"  Deadlock: yes (gross {deadlock['gross_pct']*100:.1f}% / gap {deadlock['balance_gap']:.2f})"
        )
    else:
        print("  Deadlock: no")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    if not cancelled and not closed:
        print("  (no action)")
    print(f"{'='*50}")

    return 0 if cancelled or closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
