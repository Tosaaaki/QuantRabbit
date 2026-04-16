#!/usr/bin/env python3
"""
Range Bot — Automated range scalp entry bot.

Detects ranges via range_scalp_scanner, evaluates both M5 and H1 range views,
uses MARKET for live A/S edge plus strong B-edge MICRO bites,
otherwise places LIMIT orders at BB extremes.
Exits are handled by the trader task (profit_check.py + discretionary judgment).

The bot is Claude's tool — an extension of the trader's arm for range setups only.

Usage:
    python3 tools/range_bot.py              # live mode (places real orders)
    python3 tools/range_bot.py --dry-run    # print plans, no orders
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Find the real repo root (worktrees don't have config/)
_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    # We're in a worktree — resolve to main repo
    _git_common = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip())
    _MAIN_ROOT = _git_common.resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

from market_state import get_market_state
from bot_policy import (
    global_status_allows_new_entries,
    get_pair_policy,
    load_policy,
    mode_allows_direction,
    ownership_allows_worker,
    pair_policy_allows_worker_entry,
    pair_policy_worker_block_reason,
)
from range_scalp_scanner import (
    PAIRS,
    TYPICAL_SPREADS,
    analyze_range,
    pip_size,
    to_pips,
)
from worker_target_race import (
    build_plan as build_target_race_plan,
    encode_comment as encode_target_race_comment,
    extract_trade_id_from_order_result,
    remember_trade_plan,
)
from oanda_trade_tags import enrich_open_trades
import brake_gate

# === Constants ===
BOT_TAG = "range_bot"
BOT_MARKET_TAG = "range_bot_market"
WORKER_ORDER_TAGS = {BOT_TAG, BOT_MARKET_TAG, "trend_bot_market"}
ENTRY_ORDER_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED"}
MIN_BOT_MARGIN_PCT = 0.20
BASE_BOT_MARGIN_PCT = 0.28
HIGH_BOT_MARGIN_PCT = 0.36
MAX_BOT_MARGIN_PCT = 0.42  # smaller per-shot sizing, but allow more simultaneous worker seats
LATE_SESSION_LIMIT_ONLY_HOURS_UTC = set(range(19, 24))  # 19:00-23:59 UTC = passive only
POISON_HOURS_UTC = LATE_SESSION_LIMIT_ONLY_HOURS_UTC  # trend_bot remains market-only in this window
GTD_HOURS = 4
OANDA_TIME_FMT = "%Y-%m-%dT%H:%M:%S.000000000Z"
CONVICTION_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3}
CONVICTION_LADDER = ["C", "B", "A", "S"]
PENDING_KEEP_MAX_AGE_MIN = 12
PENDING_REPRICE_ATR_FRACTION = 0.20
PENDING_REPRICE_SPREAD_MULTIPLE = 2.0
MARKET_ELIGIBLE_CONVICTIONS = {"S", "A"}
MICRO_MARKET_ELIGIBLE_CONVICTIONS = {"S", "A", "B"}
EARLY_MARKET_ELIGIBLE_CONVICTIONS = {"S", "A", "B"}
MARKET_SIZE_FACTOR = 0.80  # pay the spread, carry less size
MICRO_MARKET_SIZE_FACTOR = 0.65  # smaller size for ultra-short bites
MARKET_MAX_SPREAD_MULTIPLE = 1.40
MARKET_NEAR_ENTRY_PIPS = 1.5
MARKET_MAX_PROGRESS_TO_TP = 0.35
MARKET_MAX_DRIFT_SPREAD_MULTIPLE = 3.0
MARKET_MAX_DRIFT_ATR_FRACTION = 0.35
MIN_MARKET_RR = 1.0
FAST_MIN_MARKET_RR = 0.85
MICRO_MIN_MARKET_RR = 1.00
MICRO_MARKET_MAX_SPREAD_MULTIPLE = 1.25  # was 1.20 — consistent with trend_bot MICRO
PASSIVE_LIMIT_MAX_SPREAD_MULTIPLE = 4.0
PASSIVE_LIMIT_MAX_ATR_FRACTION = 0.80
H1_BREAKOUT_ADX = 28
H1_BREAKOUT_DI_GAP = 8
M15_BREAKOUT_ADX = 24
M15_BREAKOUT_DI_GAP = 7
M1_MARKET_READY_SCORE = 2
MICRO_M1_MARKET_READY_SCORE = 3
M1_AGAINST_SCORE = -2
M1_WICK_MIN_PIPS = 0.3
PULSE_TFS = ("H1", "M15", "M1")
PULSE_WEIGHTS = {"H1": 0.45, "M15": 0.35, "M1": 0.20}
PULSE_BLOCK_SCORE = 6.0
PULSE_SUPPORT_SCORE = 3.0
# Sizing: small lots per shot so the worker can keep more seats alive.
CONVICTION_MARGIN_PCT = {"S": 0.10, "A": 0.07, "B": 0.05}
BOT_UNIT_BOUNDS = {
    "LIMIT": {
        "BALANCED": (3000, 4500),
        "FAST": (2000, 3250),
        "MICRO": (1500, 2500),
    },
    "MARKET": {
        "BALANCED": (2250, 3250),
        "FAST": (1500, 2500),
        "MICRO": (1000, 1800),
    },
}
CONVICTION_UNIT_SCALE = {"S": 1.00, "A": 0.90, "B": 0.82, "C": 0.72}
PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"),
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}
# Price decimal places per pair
PRICE_DECIMALS = {
    "USD_JPY": 3, "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "EUR_JPY": 3, "GBP_JPY": 3, "AUD_JPY": 3,
}
LEVERAGE = {
    "USD_JPY": 25, "EUR_USD": 25, "GBP_USD": 25, "AUD_USD": 20,
    "EUR_JPY": 25, "GBP_JPY": 25, "AUD_JPY": 25,
}
LOG_FILE = _MAIN_ROOT / "logs" / "live_trade_log.txt"
REENTRY_COOLDOWN_BY_TEMPO = {"BALANCED": 8, "FAST": 4, "MICRO": 1}  # MICRO was 2 — faster re-entry
STOPLOSS_REENTRY_COOLDOWN_MIN = 6
RECENT_CLOSE_REASONS = ("TRADER_WORKER_INVENTORY", "SL_AUTO_FIRED", "SL_FIRED", "STOP_LOSS")
LOG_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S UTC",
    "%Y-%m-%d %H:%M UTC",
    "%Y-%m-%d %H:%M:%SZ",
)
CLOSE_LOG_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+CLOSE\s+(?P<pair>[A-Z_]+)\s+(?P<side>LONG|SHORT|BUY|SELL)\b.*?\breason=(?P<reason>\S+)"
)
FAST_RANGE_TP_FACTOR = 0.60
FAST_RANGE_MIN_SPREAD_MULTIPLE = 2.5
FAST_RANGE_MIN_ATR_FRACTION = 0.35
MICRO_RANGE_TP_FACTOR = 0.55
MICRO_RANGE_MIN_SPREAD_MULTIPLE = 2.5  # was 3.0 — more MICRO range shots
MICRO_RANGE_MIN_ATR_FRACTION = 0.40
WORKER_DISASTER_STOP_FACTOR = {
    "range": {"BALANCED": 2.4, "FAST": 2.8, "MICRO": 3.2},
    "trend": {"BALANCED": 1.8, "FAST": 2.1, "MICRO": 2.4},
}
WORKER_DISASTER_SPREAD_MULT = {"BALANCED": 6.0, "FAST": 7.0, "MICRO": 8.0}
WORKER_DISASTER_ATR_MULT = {"BALANCED": 1.00, "FAST": 1.15, "MICRO": 1.30}
RECENT_CLOSE_TX_REASONS = {"STOP_LOSS_ORDER", "MARKET_ORDER_TRADE_CLOSE"}
RANGE_SETUP_TFS = ("M5", "H1")
RANGE_TF_PRIORITY = {"H1": 0, "M5": 1}
H1_PROMOTABLE_C_SIGNAL_STRENGTH = 7


def load_config() -> tuple[str, str]:
    """Load OANDA credentials from config/env.toml (main repo)."""
    text = (_MAIN_ROOT / "config" / "env.toml").read_text()
    token = [l.split("=")[1].strip().strip('"') for l in text.split("\n")
             if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in text.split("\n")
            if l.startswith("oanda_account_id")][0]
    return token, acct


def load_technicals(pair: str) -> dict:
    """Load cached technicals from main repo logs/."""
    f = _MAIN_ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def fetch_prices(token: str, acct: str) -> dict:
    """Fetch current bid/ask for all pairs."""
    pairs_str = ",".join(PAIRS)
    data = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={pairs_str}", token, acct)
    prices = {}
    for p in data.get("prices", []):
        pair = p["instrument"]
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        prices[pair] = {
            "bid": float(bids[0].get("price", 0)) if bids else 0,
            "ask": float(asks[0].get("price", 0)) if asks else 0,
            "mid": (float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))) / 2
            if bids and asks else 0,
        }
    return prices


def fetch_account(token: str, acct: str) -> dict:
    """Fetch NAV and margin info."""
    data = oanda_api(f"/v3/accounts/{acct}/summary", token, acct)
    a = data.get("account", {})
    return {
        "nav": float(a.get("NAV", 0)),
        "margin_used": float(a.get("marginUsed", 0)),
        "margin_available": float(a.get("marginAvailable", 0)),
    }


def fetch_last_transaction_id(token: str, acct: str) -> int | None:
    data = oanda_api(f"/v3/accounts/{acct}/summary", token, acct)
    try:
        return int(data.get("lastTransactionID", 0))
    except (TypeError, ValueError):
        return None


def oanda_api(path: str, token: str, acct: str,
              method: str = "GET", data: dict | None = None) -> dict:
    """Hit OANDA REST API. Same pattern as rollover_guard.py."""
    url = f"https://api-fxtrade.oanda.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method=method)
    if data is not None:
        req.data = json.dumps(data).encode()
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


def fetch_open_trades(token: str, acct: str) -> list[dict]:
    """Get open trades."""
    data = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
    trades = data.get("trades", [])
    return enrich_open_trades(
        trades,
        lambda path: oanda_api(path.format(account_id=acct), token, acct),
    )


def get_tag(payload: dict) -> str:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = payload.get(key, {}) or {}
        tag = ext.get("tag")
        if tag:
            return str(tag)
    return ""


def fetch_pending_orders(token: str, acct: str) -> list[dict]:
    data = oanda_api(f"/v3/accounts/{acct}/pendingOrders", token, acct)
    return data.get("orders", [])


def pending_margin(order: dict, prices: dict) -> float:
    pair = order.get("instrument", "")
    units = abs(int(order.get("units", 0)))
    price = float(order.get("price", 0))
    if price <= 0:
        quote = prices.get(pair, {})
        price = float(quote.get("ask") or quote.get("bid") or quote.get("mid") or 0)
    return estimate_margin(units, price, pair) if price > 0 and units > 0 else 0.0


def fetch_recent_worker_close_events(token: str, acct: str, lookback_ids: int = 80) -> list[dict]:
    last_tx = fetch_last_transaction_id(token, acct)
    if not last_tx:
        return []
    start_tx = max(1, last_tx - lookback_ids)
    data = oanda_api(f"/v3/accounts/{acct}/transactions/idrange?from={start_tx}&to={last_tx}", token, acct)
    txs = data.get("transactions", [])
    worker_opens: dict[str, dict] = {}
    events: list[dict] = []

    for tx in txs:
        if tx.get("type") != "ORDER_FILL":
            continue
        trade_opened = tx.get("tradeOpened") or {}
        trade_id = trade_opened.get("tradeID")
        tag = ((trade_opened.get("clientExtensions") or {}).get("tag") or "").strip()
        if trade_id and tag in WORKER_ORDER_TAGS:
            units = int(trade_opened.get("units", tx.get("units", 0)))
            worker_opens[str(trade_id)] = {
                "pair": tx.get("instrument", ""),
                "direction": "BUY" if units > 0 else "SELL",
                "tag": tag,
            }

        reason = str(tx.get("reason", "")).upper()
        if reason not in RECENT_CLOSE_TX_REASONS:
            continue
        closed_at = parse_oanda_time(tx.get("time"))
        if closed_at is None:
            continue
        for closed in tx.get("tradesClosed") or []:
            trade_id = str(closed.get("tradeID", ""))
            meta = worker_opens.get(trade_id)
            if not meta:
                continue
            events.append({
                "pair": meta["pair"] or tx.get("instrument", ""),
                "direction": meta["direction"],
                "reason": reason,
                "closed_at": closed_at,
                "trade_id": trade_id,
                "tag": meta["tag"],
            })
    return events


def is_entry_pending_order(order: dict) -> bool:
    order_type = str(order.get("type", "")).upper()
    if order_type not in ENTRY_ORDER_TYPES:
        return False
    if not order.get("instrument"):
        return False
    try:
        return int(order.get("units", 0)) != 0
    except (TypeError, ValueError):
        return False


def fetch_pending_bot_orders(token: str, acct: str) -> list[dict]:
    """Get pending orders tagged as range_bot."""
    bot_orders = []
    for order in fetch_pending_orders(token, acct):
        if not is_entry_pending_order(order):
            continue
        if get_tag(order) == BOT_TAG:
            bot_orders.append(order)
    return bot_orders


def parse_oanda_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    normalized = ts
    if ts.endswith("Z"):
        core = ts[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            normalized = f"{head}.{frac[:6]}+00:00"
        else:
            normalized = f"{core}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def signed_units(payload: dict) -> int:
    for key in ("currentUnits", "units"):
        try:
            return int(payload.get(key, 0))
        except (TypeError, ValueError):
            continue
    return 0


def payload_direction(payload: dict) -> str:
    return "BUY" if signed_units(payload) > 0 else "SELL"


def order_direction(order: dict) -> str:
    return payload_direction(order)


def is_worker_payload(payload: dict) -> bool:
    return get_tag(payload) in WORKER_ORDER_TAGS


def is_worker_pending_order(order: dict) -> bool:
    return is_worker_payload(order)


def worker_kind_from_order_type(order_type: str) -> str:
    return "PASSIVE" if str(order_type).upper() == "LIMIT" else "MARKET"


def discretionary_open_trade_conflicts(
    open_trades: list[dict],
    pair: str,
    direction: str,
    worker_kind: str,
    ownership: str,
) -> list[tuple[str, dict]]:
    conflicts: list[tuple[str, dict]] = []
    for trade in open_trades:
        if trade.get("instrument") != pair:
            continue
        if is_worker_payload(trade):
            continue
        trade_direction = payload_direction(trade)
        if trade_direction != direction:
            conflicts.append(("opposite", trade))
            continue
        if not ownership_allows_worker(ownership, worker_kind):
            conflicts.append(("owned", trade))
    return conflicts


def describe_trade_conflicts(conflicts: list[tuple[str, dict]]) -> str:
    refs = []
    for reason, trade in conflicts:
        trade_id = trade.get("id", "?")
        if reason == "opposite":
            refs.append(f"opposite trade#{trade_id}")
        else:
            refs.append(f"trader-owned seat#{trade_id}")
    return ", ".join(refs)


def worker_trade_conflicts(open_trades: list[dict], pair: str) -> list[dict]:
    return [
        trade for trade in open_trades
        if trade.get("instrument") == pair and is_worker_payload(trade)
    ]


def describe_worker_trades(trades: list[dict]) -> str:
    return ", ".join(f"trade#{trade.get('id', '?')}" for trade in trades)


def discretionary_pending_conflicts(pending_orders: list[dict], pair: str, direction: str) -> list[dict]:
    conflicts = []
    for order in pending_orders:
        if not is_entry_pending_order(order):
            continue
        if order.get("instrument") != pair:
            continue
        if is_worker_pending_order(order):
            continue
        if order_direction(order) != direction:
            continue
        conflicts.append(order)
    return conflicts


def discretionary_pending_conflicts_for_worker(
    pending_orders: list[dict],
    pair: str,
    direction: str,
    worker_kind: str,
    ownership: str,
) -> list[dict]:
    conflicts = []
    for order in pending_orders:
        if not is_entry_pending_order(order):
            continue
        if order.get("instrument") != pair:
            continue
        if is_worker_pending_order(order):
            continue
        if order_direction(order) != direction:
            continue
        if ownership_allows_worker(ownership, worker_kind):
            continue
        conflicts.append(order)
    return conflicts


def describe_pending_conflicts(orders: list[dict]) -> str:
    refs = []
    for order in orders:
        refs.append(f"{order.get('type', '?')}#{order.get('id', '?')}")
    return ", ".join(refs)


def pending_order_sort_key(order: dict) -> datetime:
    created_at = parse_oanda_time(order.get("createTime"))
    if created_at is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return created_at


def normalize_side(side: str) -> str:
    side = str(side or "").upper()
    if side in {"BUY", "LONG"}:
        return "BUY"
    if side in {"SELL", "SHORT"}:
        return "SELL"
    return side


def parse_log_time(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in LOG_TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def recent_close_cooldown(
    pair: str,
    direction: str,
    now_utc: datetime,
    tempo: str = "BALANCED",
    recent_worker_closes: list[dict] | None = None,
) -> str | None:
    if not LOG_FILE.exists():
        lines = []
    else:
        try:
            lines = LOG_FILE.read_text(errors="ignore").splitlines()[-500:]
        except OSError:
            lines = []

    wanted_side = normalize_side(direction)
    for line in reversed(lines):
        match = CLOSE_LOG_RE.match(line.strip())
        if not match:
            continue
        if match.group("pair") != pair:
            continue
        if normalize_side(match.group("side")) != wanted_side:
            continue
        reason = match.group("reason").upper()
        if not any(token in reason for token in RECENT_CLOSE_REASONS):
            continue
        closed_at = parse_log_time(match.group("ts"))
        if closed_at is None:
            continue
        age_min = max(0.0, (now_utc - closed_at).total_seconds() / 60)
        cooldown_min = reentry_cooldown_minutes(tempo, reason)
        if age_min <= cooldown_min:
            return f"recent {match.group('reason')} close {age_min:.1f}m ago"
        break

    for event in sorted(recent_worker_closes or [], key=lambda item: item["closed_at"], reverse=True):
        if event["pair"] != pair:
            continue
        if normalize_side(event["direction"]) != wanted_side:
            continue
        age_min = max(0.0, (now_utc - event["closed_at"]).total_seconds() / 60)
        cooldown_min = reentry_cooldown_minutes(tempo, event["reason"])
        if age_min <= cooldown_min:
            return f"recent {event['reason']} close {age_min:.1f}m ago"
        break
    return None


def cancel_order(token: str, acct: str, order_id: str) -> bool:
    """Cancel a pending order. Returns True on success."""
    try:
        oanda_api(
            f"/v3/accounts/{acct}/orders/{order_id}/cancel",
            token, acct, method="PUT"
        )
        return True
    except urllib.error.HTTPError as e:
        print(f"  WARN: cancel order {order_id} failed: {e.code}")
        return False


def should_keep_pending_limit(order: dict, pair: str, direction: str,
                              desired_entry: float, opp: dict, prices: dict,
                              now_utc: datetime, max_age_min: int) -> tuple[bool, str]:
    """Decide whether an existing pending LIMIT is still good enough to keep."""
    if order_direction(order) != direction:
        return False, "direction changed"

    created_at = parse_oanda_time(order.get("createTime"))
    if created_at is not None:
        age_min = (now_utc - created_at).total_seconds() / 60
        if age_min > max_age_min:
            return False, f"age {age_min:.0f}m > {max_age_min}m"

    existing_entry = float(order.get("price", 0))
    if existing_entry <= 0:
        return False, "missing order price"

    spread = max(float(opp.get("spread", 0)), 0.5)
    atr_pips = max(float(opp.get("atr_pips", 0)), 1.0)
    tolerance_pips = max(1.0, spread * PENDING_REPRICE_SPREAD_MULTIPLE, atr_pips * PENDING_REPRICE_ATR_FRACTION)
    drift_pips = abs(to_pips(existing_entry - desired_entry, pair))
    if drift_pips > tolerance_pips:
        return False, f"entry drift {drift_pips:.1f}pip > {tolerance_pips:.1f}"

    current_price = current_execution_price(pair, direction, prices)
    if current_price > 0:
        max_gap_pips = passive_limit_gap_cap_pips(spread, atr_pips)
        live_gap_pips = abs(to_pips(current_price - existing_entry, pair))
        if live_gap_pips > max_gap_pips:
            return False, f"live gap {live_gap_pips:.1f}pip > {max_gap_pips:.1f}"

    return True, f"keep existing LIMIT @{format_price(existing_entry, pair)}"


def format_price(price: float, pair: str) -> str:
    """Format price to correct decimal places for OANDA API."""
    decimals = PRICE_DECIMALS.get(pair, 5)
    return f"{price:.{decimals}f}"


def shift_conviction(conviction: str, delta: int) -> str:
    if conviction not in CONVICTION_LADDER:
        return conviction
    idx = CONVICTION_LADDER.index(conviction)
    next_idx = max(0, min(len(CONVICTION_LADDER) - 1, idx + delta))
    return CONVICTION_LADDER[next_idx]


def infer_direction(signal: str) -> str | None:
    if "BUY" in signal:
        return "BUY"
    if "SELL" in signal:
        return "SELL"
    return None


def summarize_range_view(opp: dict) -> str:
    setup_tf = opp.get("setup_tf", "?")
    signal = str(opp.get("active_signal", ""))
    signal_label = signal.replace("_", " ").strip() or "NO SIGNAL"
    conviction = opp.get("conviction", "?")
    range_type = opp.get("range_type", "?")
    return f"{setup_tf} {conviction} {signal_label} {range_type}"


def range_candidate_sort_key(opp: dict) -> tuple:
    return (
        CONVICTION_ORDER.get(opp.get("conviction", "C"), 9),
        0 if opp.get("market_ready") else 1,
        -int(opp.get("signal_strength", 0)),
        RANGE_TF_PRIORITY.get(opp.get("setup_tf", "M5"), 9),
    )


def divergence_bias(tf_data: dict) -> str:
    rsi_kind = int(tf_data.get("div_rsi_kind", 0) or 0)
    macd_kind = int(tf_data.get("div_macd_kind", 0) or 0)
    bull = rsi_kind in (1, 3) or macd_kind in (1, 3)
    bear = rsi_kind in (2, 4) or macd_kind in (2, 4)
    if bull and not bear:
        return "BULL"
    if bear and not bull:
        return "BEAR"
    return "NONE"


def detect_band_walk_risk(tf_label: str, direction: str, tf_data: dict | None) -> dict:
    result = {
        "blocked": False,
        "notes": [],
    }
    if not tf_data:
        return result

    adx = float(tf_data.get("adx", 0))
    plus_di = float(tf_data.get("plus_di", 0))
    minus_di = float(tf_data.get("minus_di", 0))
    ema_slope_fast = float(tf_data.get("ema_slope_5", 0))
    ema_slope_slow = float(tf_data.get("ema_slope_10", tf_data.get("ema_slope_20", 0)))
    macd_hist = float(tf_data.get("macd_hist", 0))
    bb_upper = float(tf_data.get("bb_upper", 0))
    bb_lower = float(tf_data.get("bb_lower", 0))
    bb_mid = float(tf_data.get("bb_mid", 0))
    close = float(tf_data.get("close", 0))
    upper_wick = float(tf_data.get("upper_wick_avg_pips", 0))
    lower_wick = float(tf_data.get("lower_wick_avg_pips", 0))
    high_hits = float(tf_data.get("high_hits", 0))
    low_hits = float(tf_data.get("low_hits", 0))

    if bb_upper <= bb_lower or close <= 0:
        return result

    bb_pos = (close - bb_lower) / max(bb_upper - bb_lower, 1e-9)
    if direction == "SELL":
        blocked = (
            bb_pos >= 0.82
            and adx >= M15_BREAKOUT_ADX
            and plus_di >= minus_di + M15_BREAKOUT_DI_GAP
            and ema_slope_fast > 0
            and ema_slope_slow > 0
            and macd_hist > 0
            and close >= bb_mid
            and upper_wick <= max(M1_WICK_MIN_PIPS, lower_wick * 1.15)
            and high_hits >= low_hits
        )
        if blocked:
            result["blocked"] = True
            result["notes"] = [f"{tf_label} band walk up — no fade"]
    else:
        blocked = (
            bb_pos <= 0.18
            and adx >= M15_BREAKOUT_ADX
            and minus_di >= plus_di + M15_BREAKOUT_DI_GAP
            and ema_slope_fast < 0
            and ema_slope_slow < 0
            and macd_hist < 0
            and close <= bb_mid
            and lower_wick <= max(M1_WICK_MIN_PIPS, upper_wick * 1.15)
            and low_hits >= high_hits
        )
        if blocked:
            result["blocked"] = True
            result["notes"] = [f"{tf_label} band walk down — no fade"]
    return result


def assess_tf_context(tf_label: str, pair: str, direction: str, tf_data: dict | None,
                      prices: dict, breakout_adx: float, breakout_di_gap: float) -> dict:
    result = {
        "status": "missing",
        "confirmed": False,
        "blocked": False,
        "notes": [],
    }
    if not tf_data:
        result["notes"].append(f"{tf_label} missing")
        return result

    band_walk = detect_band_walk_risk(tf_label, direction, tf_data)
    if band_walk["blocked"]:
        result["status"] = "band_walk"
        result["blocked"] = True
        result["notes"] = band_walk["notes"]
        return result

    tf_opp = analyze_range(pair, tf_data, prices)
    adx = float(tf_data.get("adx", 0))
    plus_di = float(tf_data.get("plus_di", 0))
    minus_di = float(tf_data.get("minus_di", 0))
    ema_slope = float(tf_data.get("ema_slope_20", tf_data.get("ema_slope_10", 0)))
    macd_hist = float(tf_data.get("macd_hist", 0))
    bbw = float(tf_data.get("bbw", 0))
    kc_width = float(tf_data.get("kc_width", 0))

    if tf_opp and tf_opp.get("tradeable"):
        result["status"] = "confirmed"
        result["confirmed"] = True
        result["notes"].append(f"{tf_label} {tf_opp['range_type']}")
        if tf_opp.get("is_symmetric"):
            result["notes"].append(f"{tf_label} symmetric")
        tf_signal = infer_direction(tf_opp.get("active_signal", ""))
        if tf_signal == direction:
            result["notes"].append(f"{tf_label} edge aligned")
    else:
        result["status"] = "neutral"

    if direction == "BUY":
        trend_break = (
            adx >= breakout_adx
            and minus_di >= plus_di + breakout_di_gap
            and ema_slope < 0
            and macd_hist < 0
        )
        squeeze_break = kc_width > 0 and bbw < kc_width * 0.85 and minus_di > plus_di and ema_slope < 0
    else:
        trend_break = (
            adx >= breakout_adx
            and plus_di >= minus_di + breakout_di_gap
            and ema_slope > 0
            and macd_hist > 0
        )
        squeeze_break = kc_width > 0 and bbw < kc_width * 0.85 and plus_di > minus_di and ema_slope > 0

    if trend_break:
        result["status"] = "breakout_risk"
        result["confirmed"] = False
        result["blocked"] = True
        result["notes"] = [f"{tf_label} breakout risk ADX={adx:.0f} DI gap={abs(plus_di - minus_di):.0f}"]
    elif squeeze_break:
        result["status"] = "squeeze_risk"
        result["confirmed"] = False
        result["blocked"] = True
        result["notes"] = [f"{tf_label} squeeze leaning into breakout"]

    return result


def assess_h1_context(pair: str, direction: str, h1_data: dict | None, prices: dict) -> dict:
    return assess_tf_context("H1", pair, direction, h1_data, prices, H1_BREAKOUT_ADX, H1_BREAKOUT_DI_GAP)


def assess_m15_context(pair: str, direction: str, m15_data: dict | None, prices: dict) -> dict:
    return assess_tf_context("M15", pair, direction, m15_data, prices, M15_BREAKOUT_ADX, M15_BREAKOUT_DI_GAP)


def assess_m1_micro_context(direction: str, m1_data: dict | None) -> dict:
    result = {
        "score": 0,
        "state": "missing",
        "market_ready": False,
        "notes": [],
    }
    if not m1_data:
        result["notes"].append("M1 missing")
        return result

    score = 0
    notes: list[str] = []
    stoch_rsi = float(m1_data.get("stoch_rsi", 0.5))
    cci = float(m1_data.get("cci", 0))
    rsi = float(m1_data.get("rsi", 50))
    upper_wick = float(m1_data.get("upper_wick_avg_pips", 0))
    lower_wick = float(m1_data.get("lower_wick_avg_pips", 0))
    ema_slope = float(m1_data.get("ema_slope_5", 0))
    macd_hist = float(m1_data.get("macd_hist", 0))
    plus_di = float(m1_data.get("plus_di", 0))
    minus_di = float(m1_data.get("minus_di", 0))
    div_bias = divergence_bias(m1_data)

    if direction == "BUY":
        if stoch_rsi <= 0.25:
            score += 1
            notes.append(f"M1 StRSI={stoch_rsi:.2f}")
        elif stoch_rsi >= 0.75:
            score -= 1
        if cci <= -100:
            score += 1
            notes.append(f"M1 CCI={cci:.0f}")
        elif cci >= 100:
            score -= 1
        if rsi <= 38:
            score += 1
        elif rsi >= 62:
            score -= 1
        if lower_wick >= max(M1_WICK_MIN_PIPS, upper_wick * 1.2):
            score += 1
            notes.append(f"M1 lower wick {lower_wick:.1f}pip")
        elif upper_wick >= max(M1_WICK_MIN_PIPS, lower_wick * 1.2):
            score -= 1
        if div_bias == "BULL":
            score += 1
            notes.append("M1 bull div")
        elif div_bias == "BEAR":
            score -= 1
        if ema_slope > 0 and macd_hist > 0:
            score += 1
        elif ema_slope < 0 and macd_hist < 0 and minus_di > plus_di + 5:
            score -= 1
    else:
        if stoch_rsi >= 0.75:
            score += 1
            notes.append(f"M1 StRSI={stoch_rsi:.2f}")
        elif stoch_rsi <= 0.25:
            score -= 1
        if cci >= 100:
            score += 1
            notes.append(f"M1 CCI={cci:.0f}")
        elif cci <= -100:
            score -= 1
        if rsi >= 62:
            score += 1
        elif rsi <= 38:
            score -= 1
        if upper_wick >= max(M1_WICK_MIN_PIPS, lower_wick * 1.2):
            score += 1
            notes.append(f"M1 upper wick {upper_wick:.1f}pip")
        elif lower_wick >= max(M1_WICK_MIN_PIPS, upper_wick * 1.2):
            score -= 1
        if div_bias == "BEAR":
            score += 1
            notes.append("M1 bear div")
        elif div_bias == "BULL":
            score -= 1
        if ema_slope < 0 and macd_hist < 0:
            score += 1
        elif ema_slope > 0 and macd_hist > 0 and plus_di > minus_di + 5:
            score -= 1

    if score >= M1_MARKET_READY_SCORE:
        state = "aligned"
    elif score <= M1_AGAINST_SCORE:
        state = "against"
    else:
        state = "neutral"
    if not notes:
        notes.append(f"M1 {state}")

    result.update(
        score=score,
        state=state,
        market_ready=score >= M1_MARKET_READY_SCORE,
        notes=notes,
    )
    return result


def build_currency_pulse(all_technicals: dict[str, dict]) -> dict[str, dict[str, float]]:
    currencies = ["USD", "JPY", "EUR", "GBP", "AUD"]
    signals = {ccy: {tf: [] for tf in PULSE_TFS} for ccy in currencies}

    for pair, (base, quote) in PAIR_CURRENCIES.items():
        tfs = all_technicals.get(pair, {})
        for tf in PULSE_TFS:
            d = tfs.get(tf, {})
            adx = float(d.get("adx", 0))
            if adx < 5:
                continue
            plus_di = float(d.get("plus_di", 0))
            minus_di = float(d.get("minus_di", 0))
            gap = (plus_di - minus_di) * min(adx / 25.0, 1.5)
            signals[base][tf].append(gap)
            signals[quote][tf].append(-gap)

    pulse: dict[str, dict[str, float]] = {}
    for ccy in currencies:
        pulse[ccy] = {}
        for tf in PULSE_TFS:
            vals = signals[ccy][tf]
            pulse[ccy][tf] = sum(vals) / len(vals) if vals else 0.0
    return pulse


def assess_currency_context(pair: str, direction: str, pulse: dict[str, dict[str, float]]) -> dict:
    base, quote = PAIR_CURRENCIES[pair]
    tf_scores = {
        tf: pulse.get(base, {}).get(tf, 0.0) - pulse.get(quote, {}).get(tf, 0.0)
        for tf in PULSE_TFS
    }
    composite = sum(PULSE_WEIGHTS[tf] * tf_scores[tf] for tf in PULSE_TFS)
    if direction == "BUY":
        aligned = composite >= PULSE_SUPPORT_SCORE
        blocked = composite <= -PULSE_BLOCK_SCORE
    else:
        aligned = composite <= -PULSE_SUPPORT_SCORE
        blocked = composite >= PULSE_BLOCK_SCORE

    notes = []
    if blocked:
        notes.append(
            f"currency pulse favors breakout "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )
    elif aligned:
        notes.append(
            f"currency pulse supports fade "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )
    elif abs(composite) >= PULSE_SUPPORT_SCORE:
        notes.append(
            f"currency pulse mixed "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )

    return {
        "blocked": blocked,
        "aligned": aligned,
        "score": composite,
        "notes": notes,
    }


def unit_bounds(order_type: str, tempo: str, conviction: str) -> tuple[int, int]:
    order_type = str(order_type or "LIMIT").upper()
    tempo = str(tempo or "BALANCED").upper()
    bounds_by_order = BOT_UNIT_BOUNDS.get(order_type, BOT_UNIT_BOUNDS["LIMIT"])
    min_units, max_units = bounds_by_order.get(tempo, bounds_by_order["BALANCED"])
    scaled_max = int(round(max_units * CONVICTION_UNIT_SCALE.get(str(conviction or "B").upper(), 0.82)))
    return min_units, max(min_units, scaled_max)


def calculate_units(conviction: str, nav: float, price: float,
                    pair: str, order_type: str = "LIMIT",
                    tempo: str = "BALANCED") -> int:
    """Calculate units based on conviction and NAV, then clamp to the worker's small-lot ladder."""
    margin_pct = CONVICTION_MARGIN_PCT.get(conviction, 0.05)
    margin_budget = nav * margin_pct
    leverage = LEVERAGE.get(pair, 25)
    units = int(margin_budget / (price / leverage))
    order_type = str(order_type or "LIMIT").upper()
    tempo = str(tempo or "BALANCED").upper()
    if order_type == "MARKET":
        if tempo == "MICRO":
            units = int(units * MICRO_MARKET_SIZE_FACTOR)
        else:
            units = int(units * MARKET_SIZE_FACTOR)
    min_units, max_units = unit_bounds(order_type, tempo, conviction)
    return max(min_units, min(max_units, units))


def reentry_cooldown_minutes(tempo: str, reason: str | None = None) -> int:
    tempo = str(tempo or "BALANCED").upper()
    cooldown = REENTRY_COOLDOWN_BY_TEMPO.get(tempo, REENTRY_COOLDOWN_BY_TEMPO["BALANCED"])
    reason_text = str(reason or "").upper()
    if any(token in reason_text for token in ("STOP_LOSS", "SL_AUTO_FIRED", "SL_FIRED")):
        cooldown = max(cooldown, STOPLOSS_REENTRY_COOLDOWN_MIN)
    return cooldown


def opportunity_is_live(opp: dict) -> bool:
    if "market_ready" in opp:
        return bool(opp.get("market_ready"))
    return str(opp.get("m1_state", "")).lower() == "aligned"


def dynamic_bot_budget_profile(policy: dict, opportunities: list[dict]) -> tuple[float, list[str]]:
    pct = MIN_BOT_MARGIN_PCT
    reasons: list[str] = ["base scalp budget"]
    if not global_status_allows_new_entries(policy):
        return pct, reasons

    pairs = policy.get("pairs") or {}
    active_lanes = [
        row for row in pairs.values()
        if any(
            pair_policy_allows_worker_entry(row, direction, worker_kind)
            for direction in ("BUY", "SELL")
            for worker_kind in ("PASSIVE", "MARKET")
        )
    ]
    if len(active_lanes) >= 2:
        pct = max(pct, BASE_BOT_MARGIN_PCT)
        reasons.append(f"{len(active_lanes)} active worker lanes")

    qualified = [opp for opp in opportunities if opp.get("conviction") in {"S", "A"}]
    live_qualified = [opp for opp in qualified if opportunity_is_live(opp)]
    live_s = [opp for opp in live_qualified if opp.get("conviction") == "S"]

    if qualified:
        pct = max(pct, BASE_BOT_MARGIN_PCT)
        reasons.append(f"{len(qualified)} A/S candidate(s)")
    if live_qualified or len(qualified) >= 2:
        pct = max(pct, HIGH_BOT_MARGIN_PCT)
        reasons.append(f"{len(live_qualified) or len(qualified)} high-conviction lane(s) live/stacked")
    if len(live_s) >= 1 and (len(live_qualified) >= 2 or len(live_s) >= 2):
        pct = MAX_BOT_MARGIN_PCT
        reasons.append("S-conviction tape with multiple live lanes")

    return min(MAX_BOT_MARGIN_PCT, pct), reasons


def estimate_margin(units: int, price: float, pair: str) -> float:
    """Estimate margin required for a position."""
    leverage = LEVERAGE.get(pair, 25)
    return units * price / leverage


def worker_disaster_stop(
    pair: str,
    direction: str,
    entry: float,
    structural_stop: float,
    *,
    tempo: str,
    style: str,
    atr_pips: float,
    spread_pips: float,
) -> tuple[float, float]:
    tempo = str(tempo or "BALANCED").upper()
    style = str(style or "range").lower()
    structural_stop_pips = abs(to_pips(structural_stop - entry, pair))
    disaster_pips = max(
        structural_stop_pips,
        structural_stop_pips * WORKER_DISASTER_STOP_FACTOR.get(style, WORKER_DISASTER_STOP_FACTOR["range"]).get(tempo, 2.0),
        max(float(atr_pips), 0.5) * WORKER_DISASTER_ATR_MULT.get(tempo, 1.0),
        max(float(spread_pips), 0.1) * WORKER_DISASTER_SPREAD_MULT.get(tempo, 6.0),
    )
    ps = pip_size(pair)
    if direction == "BUY":
        return entry - disaster_pips * ps, disaster_pips
    return entry + disaster_pips * ps, disaster_pips


def place_limit(token: str, acct: str, pair: str, direction: str,
                units: int, entry: float, tp: float, sl: float | None,
                gtd_time: str, comment: str) -> dict:
    """Place a LIMIT order with TP on fill and bot tag. SL is optional — pass None for MICRO/FAST no-SL mode."""
    signed_units = str(units) if direction == "BUY" else str(-units)
    order: dict = {
        "type": "LIMIT",
        "instrument": pair,
        "units": signed_units,
        "price": format_price(entry, pair),
        "timeInForce": "GTD",
        "gtdTime": gtd_time,
        "takeProfitOnFill": {
            "price": format_price(tp, pair),
            "timeInForce": "GTC",
        },
        "clientExtensions": {
            "tag": BOT_TAG,
            "comment": comment,
        },
        "tradeClientExtensions": {
            "tag": BOT_TAG,
            "comment": comment,
        },
    }
    if sl is not None:
        order["stopLossOnFill"] = {
            "price": format_price(sl, pair),
            "timeInForce": "GTC",
        }
    payload = {"order": order}
    try:
        return oanda_api(
            f"/v3/accounts/{acct}/orders", token, acct,
            method="POST", data=payload
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else str(e)
        return {"error": body, "code": e.code}


def place_market(token: str, acct: str, pair: str, direction: str,
                 units: int, tp: float, sl: float | None, comment: str) -> dict:
    """Place a MARKET order with TP on fill and market tag. SL is optional — pass None for MICRO/FAST no-SL mode."""
    signed_units = str(units) if direction == "BUY" else str(-units)
    order: dict = {
        "type": "MARKET",
        "instrument": pair,
        "units": signed_units,
        "timeInForce": "FOK",
        "takeProfitOnFill": {
            "price": format_price(tp, pair),
            "timeInForce": "GTC",
        },
        "clientExtensions": {
            "tag": BOT_MARKET_TAG,
            "comment": comment,
        },
        "tradeClientExtensions": {
            "tag": BOT_MARKET_TAG,
            "comment": comment,
        },
    }
    if sl is not None:
        order["stopLossOnFill"] = {
            "price": format_price(sl, pair),
            "timeInForce": "GTC",
        }
    payload = {"order": order}
    try:
        return oanda_api(
            f"/v3/accounts/{acct}/orders", token, acct,
            method="POST", data=payload
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else str(e)
        return {"error": body, "code": e.code}


def trade_metrics(direction: str, entry: float, tp: float, sl: float, pair: str) -> tuple[float, float, float]:
    """Return TP pips, SL pips, and R:R for a planned or live entry."""
    if direction == "BUY":
        tp_pips = to_pips(tp - entry, pair)
        sl_pips = to_pips(entry - sl, pair)
    else:
        tp_pips = to_pips(entry - tp, pair)
        sl_pips = to_pips(sl - entry, pair)
    rr = tp_pips / sl_pips if sl_pips > 0 else 0
    return tp_pips, sl_pips, rr


def market_stop_floor_pips(tempo: str, spread_pips: float, atr_pips: float) -> float:
    """Minimum live-entry stop width so a market fill is not just spread/noise bait."""
    tempo = str(tempo or "BALANCED").upper()
    if tempo == "MICRO":
        return max(spread_pips * 3.5, atr_pips * 0.45)
    if tempo == "FAST":
        return max(spread_pips * 4.0, atr_pips * 0.55)
    return max(spread_pips * 4.5, atr_pips * 0.65)


def widened_market_stop(pair: str, direction: str, entry: float, sl: float, min_sl_pips: float) -> float:
    """Only widen the live-entry stop when the inherited stop is obviously too tight."""
    current_sl_pips = abs(to_pips(entry - sl, pair))
    if current_sl_pips >= min_sl_pips:
        return sl
    ps = pip_size(pair)
    if direction == "BUY":
        return entry - min_sl_pips * ps
    return entry + min_sl_pips * ps


def passive_limit_gap_cap_pips(spread_pips: float, atr_pips: float) -> float:
    """Maximum distance a passive LIMIT may sit away from the live price."""
    return max(spread_pips * PASSIVE_LIMIT_MAX_SPREAD_MULTIPLE, atr_pips * PASSIVE_LIMIT_MAX_ATR_FRACTION)


def finalize_passive_limit_plan(plan: dict, pair: str, direction: str, tp: float, sl: float,
                                opp: dict, prices: dict) -> dict:
    """Cap passive LIMIT distance so the bot does not park unrealistic far-away orders."""
    if plan.get("order_type") != "LIMIT":
        return plan

    current_price = current_execution_price(pair, direction, prices)
    if current_price <= 0:
        return plan

    spread_pips = max(float(opp.get("spread", 0)), 0.5)
    atr_pips = max(float(opp.get("atr_pips", 0)), 1.0)
    max_gap_pips = passive_limit_gap_cap_pips(spread_pips, atr_pips)
    desired_entry = float(plan.get("entry", 0))
    if desired_entry <= 0:
        return plan

    gap_pips = abs(to_pips(current_price - desired_entry, pair))
    if gap_pips <= max_gap_pips:
        return plan

    ps = pip_size(pair)
    adjusted_entry = current_price - max_gap_pips * ps if direction == "BUY" else current_price + max_gap_pips * ps
    tp_pips, sl_pips, rr = trade_metrics(direction, adjusted_entry, tp, sl, pair)
    if tp_pips <= 0 or sl_pips <= 0 or rr < 1.0:
        plan["order_type"] = "SKIP"
        plan["reason"] = (
            f"passive limit {gap_pips:.1f}pip from live; closer reload would kill R:R ({rr:.1f})"
        )
        return plan

    plan["entry"] = adjusted_entry
    plan["tp_pips"] = tp_pips
    plan["sl_pips"] = sl_pips
    plan["rr"] = rr
    plan["reason"] = f"{plan['reason']} | capped reload {gap_pips:.1f}->{max_gap_pips:.1f}pip from live"
    return plan


def current_execution_price(pair: str, direction: str, prices: dict) -> float:
    """Get the price a market order would hit right now."""
    quote = prices.get(pair, {})
    if direction == "BUY":
        return float(quote.get("ask", 0))
    return float(quote.get("bid", 0))


def current_stop_trigger_price(pair: str, direction: str, prices: dict) -> float:
    """Get the live quote that would trigger the protective stop right now."""
    quote = prices.get(pair, {})
    if direction == "BUY":
        return float(quote.get("bid", 0))
    return float(quote.get("ask", 0))


def is_marketable_limit(pair: str, direction: str, entry: float, prices: dict) -> bool:
    """Return True when a LIMIT would execute immediately at current quotes."""
    quote = prices.get(pair, {})
    bid = float(quote.get("bid", 0))
    ask = float(quote.get("ask", 0))
    if direction == "BUY":
        return ask > 0 and entry >= ask
    return bid > 0 and entry <= bid


def stop_is_already_broken(pair: str, direction: str, sl: float, prices: dict) -> bool:
    """Return True when the live stop-trigger side is already through the stop."""
    trigger_price = current_stop_trigger_price(pair, direction, prices)
    if trigger_price <= 0:
        return False
    if direction == "BUY":
        return trigger_price <= sl
    return trigger_price >= sl


def choose_order_plan(opp: dict, pair: str, direction: str, limit_entry: float,
                      tp: float, sl: float, prices: dict, min_market_rr: float,
                      tempo: str = "BALANCED", entry_bias: str = "BALANCED",
                      coverage_repair: bool = False,
                      late_session_limit_only: bool = False) -> dict:
    """Choose LIMIT vs MARKET based on live edge quality."""
    def finish(plan: dict) -> dict:
        return finalize_passive_limit_plan(plan, pair, direction, tp, sl, opp, prices)

    current_price = current_execution_price(pair, direction, prices)
    current_spread = float(opp.get("spread", 0))
    typical_spread = TYPICAL_SPREADS.get(pair, max(current_spread, 1.0))
    conviction = opp.get("conviction", "C")
    market_ready = bool(opp.get("market_ready", False))
    market_note = str(opp.get("market_note", "")).strip()
    tempo = str(tempo or "BALANCED").upper()
    entry_bias = str(entry_bias or "BALANCED").upper()
    if entry_bias == "PASSIVE":
        market_eligible_convictions: set[str] = set()
    elif entry_bias == "EARLY" and coverage_repair:
        market_eligible_convictions = (
            MICRO_MARKET_ELIGIBLE_CONVICTIONS if tempo == "MICRO" else EARLY_MARKET_ELIGIBLE_CONVICTIONS
        )
    else:
        market_eligible_convictions = (
            MICRO_MARKET_ELIGIBLE_CONVICTIONS if tempo == "MICRO" else MARKET_ELIGIBLE_CONVICTIONS
        )

    limit_tp_pips, limit_sl_pips, limit_rr = trade_metrics(direction, limit_entry, tp, sl, pair)
    plan = {
        "order_type": "LIMIT",
        "entry": limit_entry,
        "sl": sl,
        "tag": BOT_TAG,
        "reason": "wait at BB edge",
        "tp_pips": limit_tp_pips,
        "sl_pips": limit_sl_pips,
        "rr": limit_rr,
        "current_price": current_price,
        "progress_to_tp": 0.0,
    }

    if conviction not in market_eligible_convictions:
        if is_marketable_limit(pair, direction, limit_entry, prices):
            plan["order_type"] = "SKIP"
            if stop_is_already_broken(pair, direction, sl, prices):
                plan["reason"] = f"{conviction} limit crossed and stop already invalid"
            else:
                plan["reason"] = f"{conviction} conviction stays passive; crossed limit skipped"
        else:
            plan["reason"] = f"{conviction} conviction stays passive"
        return finish(plan)
    if not market_ready:
        if is_marketable_limit(pair, direction, limit_entry, prices):
            plan["order_type"] = "SKIP"
            plan["reason"] = f"crossed limit but {market_note or 'M1 trigger not ready'}"
            return plan
        plan["reason"] = f"wait at BB edge | {market_note or 'M1 trigger not ready'}"
        return finish(plan)
    if tempo == "MICRO":
        if float(opp.get("m1_score", 0)) < MICRO_M1_MARKET_READY_SCORE:
            if is_marketable_limit(pair, direction, limit_entry, prices):
                plan["order_type"] = "SKIP"
                plan["reason"] = (
                    f"crossed limit but M1 micro score {opp.get('m1_score', 0)} < {MICRO_M1_MARKET_READY_SCORE}"
                )
                return plan
            plan["reason"] = (
                f"wait at BB edge | M1 micro score {opp.get('m1_score', 0)} < {MICRO_M1_MARKET_READY_SCORE}"
            )
            return finish(plan)
        if current_spread > typical_spread * MICRO_MARKET_MAX_SPREAD_MULTIPLE:
            if is_marketable_limit(pair, direction, limit_entry, prices):
                plan["order_type"] = "SKIP"
                plan["reason"] = (
                    f"crossed limit but spread {current_spread:.1f}pip is too wide for MICRO"
                )
                return plan
            plan["reason"] = f"spread {current_spread:.1f}pip too wide for MICRO"
            return finish(plan)
    if current_price <= 0:
        plan["reason"] = "live price unavailable"
        return finish(plan)
    if stop_is_already_broken(pair, direction, sl, prices):
        plan["order_type"] = "SKIP"
        plan["reason"] = "live stop trigger already through SL"
        return plan
    if current_spread > typical_spread * MARKET_MAX_SPREAD_MULTIPLE:
        if is_marketable_limit(pair, direction, limit_entry, prices):
            plan["order_type"] = "SKIP"
            plan["reason"] = (
                f"crossed limit but spread {current_spread:.1f}pip is too wide"
            )
            return plan
        plan["reason"] = (
            f"spread {current_spread:.1f}pip > {MARKET_MAX_SPREAD_MULTIPLE:.1f}x normal"
        )
        return finish(plan)

    atr_pips = max(float(opp.get("atr_pips", 0)), 1.0)
    market_sl = widened_market_stop(
        pair,
        direction,
        current_price,
        sl,
        market_stop_floor_pips(tempo, current_spread, atr_pips),
    )
    market_tp_pips, market_sl_pips, market_rr = trade_metrics(direction, current_price, tp, market_sl, pair)
    if market_rr < min_market_rr or market_tp_pips <= 0 or market_sl_pips <= 0:
        plan["reason"] = f"market R:R {market_rr:.1f} too weak"
        return finish(plan)

    gap_pips = abs(to_pips(current_price - limit_entry, pair))
    tp_distance = abs(tp - limit_entry)
    moved_toward_tp = (
        current_price - limit_entry
        if direction == "BUY"
        else limit_entry - current_price
    )
    progress_to_tp = max(0.0, moved_toward_tp / tp_distance) if tp_distance > 0 else 0.0
    if gap_pips <= max(MARKET_NEAR_ENTRY_PIPS, current_spread * 1.5):
        reason = "live extreme now"
        if coverage_repair and entry_bias == "EARLY" and conviction == "B":
            reason = "flat-book repair scout off live edge"
        if late_session_limit_only:
            plan["reason"] = f"late NY / pre-Tokyo: passive only | {reason}"
            return plan
        return {
            "order_type": "MARKET",
            "entry": current_price,
            "sl": market_sl,
            "tag": BOT_MARKET_TAG,
            "reason": reason,
            "tp_pips": market_tp_pips,
            "sl_pips": market_sl_pips,
            "rr": market_rr,
            "current_price": current_price,
            "progress_to_tp": progress_to_tp,
        }

    max_drift_pips = max(
        current_spread * MARKET_MAX_DRIFT_SPREAD_MULTIPLE,
        atr_pips * MARKET_MAX_DRIFT_ATR_FRACTION,
    )
    if 0 < progress_to_tp <= MARKET_MAX_PROGRESS_TO_TP and gap_pips <= max_drift_pips:
        reason = f"rescue bounce {progress_to_tp:.0%} off edge"
        if coverage_repair and entry_bias == "EARLY" and conviction == "B":
            reason = f"flat-book repair rescue {progress_to_tp:.0%} off edge"
        if late_session_limit_only:
            plan["reason"] = f"late NY / pre-Tokyo: passive only | {reason}"
            return plan
        return {
            "order_type": "MARKET",
            "entry": current_price,
            "sl": market_sl,
            "tag": BOT_MARKET_TAG,
            "reason": reason,
            "tp_pips": market_tp_pips,
            "sl_pips": market_sl_pips,
            "rr": market_rr,
            "current_price": current_price,
            "progress_to_tp": progress_to_tp,
        }

    if is_marketable_limit(pair, direction, limit_entry, prices):
        plan["order_type"] = "SKIP"
        plan["reason"] = f"crossed limit after {progress_to_tp:.0%} TP progress — no chase"
        return plan

    plan["reason"] = f"price drifted {progress_to_tp:.0%} to TP — wait for retest"
    return finish(plan)


def tempo_tp_target(pair: str, direction: str, entry: float, base_tp: float,
                    spread_pips: float, atr_pips: float, tempo: str) -> float:
    base_tp_pips = abs(to_pips(base_tp - entry, pair))
    if base_tp_pips <= 0:
        return base_tp

    tempo = str(tempo or "BALANCED").upper()
    if tempo == "MICRO":
        target_pips = max(
            base_tp_pips * MICRO_RANGE_TP_FACTOR,
            spread_pips * MICRO_RANGE_MIN_SPREAD_MULTIPLE,
            atr_pips * MICRO_RANGE_MIN_ATR_FRACTION,
        )
    else:
        target_pips = max(
            base_tp_pips * FAST_RANGE_TP_FACTOR,
            spread_pips * FAST_RANGE_MIN_SPREAD_MULTIPLE,
            atr_pips * FAST_RANGE_MIN_ATR_FRACTION,
        )
    target_pips = min(base_tp_pips, target_pips)
    ps = 0.001 if "JPY" in pair else 0.0001
    if direction == "BUY":
        return entry + target_pips * ps
    return entry - target_pips * ps


def move_price(pair: str, direction: str, start: float, pips: float) -> float:
    ps = 0.001 if "JPY" in pair else 0.0001
    if direction == "BUY":
        return start + pips * ps
    return start - pips * ps


def build_range_target_race(
    pair: str,
    direction: str,
    entry: float,
    tp1: float,
    sl: float,
    opp: dict,
    tempo: str,
) -> dict:
    atr_pips = max(float(opp.get("atr_pips", 0)), 1.0)
    spread = max(float(opp.get("spread", 0)), 0.5)
    setup_tf = str(opp.get("setup_tf", "M5")).upper()
    if direction == "BUY":
        far_target = float(opp.get("bb_upper", tp1))
    else:
        far_target = float(opp.get("bb_lower", tp1))

    if tempo == "MICRO":
        tp2 = float(opp.get("bb_mid", tp1))
    elif tempo == "FAST":
        tp2 = float(opp.get("bb_mid", far_target))
    else:
        tp2 = far_target

    tp1_pips = max(abs(to_pips(tp1 - entry, pair)), spread * 2.0)
    if tempo == "MICRO":
        tp1_pips = max(tp1_pips, spread * 3.0, atr_pips * 0.40)
    lock_pips = max(spread * 1.5, min(tp1_pips * 0.25, atr_pips * 0.55))
    hold_boundary = move_price(pair, direction, entry, lock_pips)
    pace_fast = max(atr_pips * 0.55, spread * 2.0)
    pace_slow = max(atr_pips * (0.35 if setup_tf == "H1" else 0.25), spread * 1.5)
    return build_target_race_plan(
        style="range",
        pair=pair,
        direction=direction,
        entry=entry,
        stop=sl,
        tp1=tp1,
        tp2=tp2,
        hold_boundary=hold_boundary,
        pace_fast_pips=pace_fast,
        pace_slow_pips=pace_slow,
    )


def log_entry(order_type: str, pair: str, direction: str, units: int,
              entry: float, tp: float, sl: float | None, order_id: str, tag: str) -> None:
    """Append entry to live_trade_log.txt."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    tif = f"GTD={GTD_HOURS}h " if order_type == "LIMIT" else ""
    sl_str = str(sl) if sl is not None else "NO_SL"
    line = (
        f"[{now}] RANGE_BOT_{order_type} {pair} {direction} {units}u "
        f"@{entry} TP={tp} SL={sl_str} {tif}id={order_id} tag={tag}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)


def log_cancel(order_type: str, pair: str, direction: str, units: int,
               entry: float, order_id: str, tag: str, reason: str) -> None:
    """Append immediate OANDA cancel/reject outcome to live_trade_log.txt."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = (
        f"[{now}] RANGE_BOT_{order_type}_CANCEL {pair} {direction} {units}u "
        f"@{entry} id={order_id} tag={tag} reason={reason}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)


def slack_notify(order_type: str, pair: str, direction: str, units: int,
                 entry: float, tp: float, sl: float | None, note: str) -> None:
    """Post entry notification to Slack via slack_trade_notify.py."""
    side = "LONG" if direction == "BUY" else "SHORT"
    cmd = [
        sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
        "entry",
        "--pair", pair,
        "--side", side,
        "--units", str(units),
        "--price", str(entry),
        *(["--sl", str(sl)] if sl is not None else []),
        "--thesis",
        (
            f"Range bot {order_type.lower()}: "
            f"BB {'lower' if direction == 'BUY' else 'upper'} {note}"
        ),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass  # Slack failure must not abort the bot


def scan_ranges(prices: dict) -> list[dict]:
    """Scan all pairs for range opportunities. Returns filtered + sorted list."""
    results = []
    all_technicals = {pair: load_technicals(pair) for pair in PAIRS}
    currency_pulse = build_currency_pulse(all_technicals)

    for pair in PAIRS:
        tfs = all_technicals.get(pair, {})
        pair_candidates: list[dict] = []
        for setup_tf in RANGE_SETUP_TFS:
            tf_data = tfs.get(setup_tf)
            if not tf_data:
                continue
            opp = analyze_range(pair, tf_data, prices)
            if not opp or not opp.get("tradeable"):
                continue

            conv = opp.get("conviction", "C")
            promotable_h1_c = (
                setup_tf == "H1"
                and conv == "C"
                and int(opp.get("signal_strength", 0)) >= H1_PROMOTABLE_C_SIGNAL_STRENGTH
            )
            if conv not in ("S", "A", "B") and not promotable_h1_c:
                continue

            signal = opp.get("active_signal", "")
            if signal == "MID_ZONE":
                continue
            direction = infer_direction(signal)
            if not direction:
                continue

            setup_band_walk = detect_band_walk_risk(setup_tf, direction, tf_data)
            if setup_band_walk["blocked"]:
                continue

            m15_context = assess_m15_context(pair, direction, tfs.get("M15"), prices)
            h1_context = assess_h1_context(pair, direction, tfs.get("H1"), prices)
            currency_context = assess_currency_context(pair, direction, currency_pulse)
            if h1_context["blocked"]:
                continue

            soft_breakout_pressure = setup_tf == "H1" and (
                m15_context["blocked"] or currency_context["blocked"]
            )
            if (m15_context["blocked"] or currency_context["blocked"]) and not soft_breakout_pressure:
                continue

            m1_context = assess_m1_micro_context(direction, tfs.get("M1"))
            conviction_delta = 0
            confirmed_frames = int(m15_context["confirmed"]) + int(h1_context["confirmed"])

            if confirmed_frames >= 1 and conv in ("A", "B"):
                conviction_delta += 1
            if confirmed_frames == 2 and currency_context["aligned"] and conv == "B":
                conviction_delta += 1
            elif currency_context["aligned"] and conv == "B":
                conviction_delta += 1
            if promotable_h1_c and (
                confirmed_frames >= 1
                or currency_context["aligned"]
                or m1_context["market_ready"]
            ):
                conviction_delta += 1

            if (
                m1_context["state"] == "against"
                and not confirmed_frames
                and not currency_context["aligned"]
                and conv == "B"
            ):
                conviction_delta -= 1
            if soft_breakout_pressure:
                conviction_delta -= int(m15_context["blocked"]) + int(currency_context["blocked"])
            if conv == "C":
                conviction_delta = min(conviction_delta, 1)

            enriched = dict(opp)
            enriched["setup_tf"] = setup_tf
            enriched["base_conviction"] = conv
            enriched["conviction"] = shift_conviction(conv, conviction_delta)
            if enriched["conviction"] not in ("S", "A", "B"):
                continue

            context_notes = (
                [f"{setup_tf} source"]
                + setup_band_walk["notes"][:1]
                + m15_context["notes"][:2]
                + h1_context["notes"][:2]
                + currency_context["notes"][:1]
                + m1_context["notes"][:2]
            )
            if promotable_h1_c and conviction_delta > 0:
                context_notes.append("H1 clean range promoted from C by structure + live trigger")
            context_notes = [note for note in context_notes if note]
            enriched["signal_strength"] = (
                opp.get("signal_strength", 0)
                + int(m15_context["confirmed"])
                + int(h1_context["confirmed"])
                + int(currency_context["aligned"])
            )
            enriched["triggers"] = context_notes + list(opp.get("triggers", []))
            enriched["m15_context"] = m15_context["status"]
            enriched["h1_context"] = h1_context["status"]
            enriched["currency_pulse_score"] = round(currency_context["score"], 2)
            enriched["currency_context"] = "aligned" if currency_context["aligned"] else "neutral"
            enriched["m1_context"] = m1_context["state"]
            enriched["m1_score"] = m1_context["score"]
            market_structure_ready = (
                confirmed_frames >= 1
                or currency_context["aligned"]
                or conv == "S"
            )
            enriched["market_ready"] = (
                m1_context["market_ready"]
                and market_structure_ready
                and not soft_breakout_pressure
            )
            enriched["market_note"] = " / ".join(context_notes) if context_notes else "MTF trigger not ready"
            pair_candidates.append(enriched)

        if not pair_candidates:
            continue

        pair_candidates.sort(key=range_candidate_sort_key)
        primary = dict(pair_candidates[0])
        primary["alternate_views"] = [summarize_range_view(candidate) for candidate in pair_candidates[1:]]
        results.append(primary)

    # Sort: better conviction first, then market-ready views, then stronger signal.
    results.sort(key=range_candidate_sort_key)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Range Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plans without placing orders")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== RANGE BOT === {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    # --- PRE-FLIGHT ---
    state, reason = get_market_state(now_utc)
    if state != "OPEN":
        print(f"Market {state}: {reason}")
        return 1

    late_session_limit_only = now_utc.hour in LATE_SESSION_LIMIT_ONLY_HOURS_UTC
    if late_session_limit_only:
        print(f"Late NY / pre-Tokyo hour {now_utc.hour} UTC: LIMIT-only mode.")

    # --- FETCH STATE ---
    token, acct = load_config()
    account = fetch_account(token, acct)
    nav = account["nav"]
    margin_used = account["margin_used"]
    margin_pct = margin_used / nav * 100 if nav > 0 else 100
    policy, policy_notes = load_policy()
    allow_new_entries = global_status_allows_new_entries(policy)
    max_pending_age_min = int(policy.get("max_pending_age_min", PENDING_KEEP_MAX_AGE_MIN))

    print(f"NAV: {nav:,.0f} JPY | Margin: {margin_pct:.1f}%")
    print(
        f"Policy: {policy['global_status']} | projected_cap={policy['projected_margin_cap']:.2f} "
        f"| panic={policy['panic_margin_cap']:.2f} | pending_age={max_pending_age_min}m"
    )
    if policy_notes:
        print(f"Policy notes: {'; '.join(policy_notes)}")
    if not allow_new_entries:
        print("Policy blocks new entries: reduce-only mode")

    open_trades = fetch_open_trades(token, acct)
    open_pairs = {trade.get("instrument", "?") for trade in open_trades}
    if open_pairs:
        print(f"Open positions: {', '.join(sorted(open_pairs))}")

    prices = fetch_prices(token, acct)
    recent_worker_closes = fetch_recent_worker_close_events(token, acct)

    # --- EXISTING PENDING ENTRIES ---
    pending_orders = fetch_pending_orders(token, acct)
    bot_orders = [
        order for order in pending_orders
        if is_entry_pending_order(order) and get_tag(order) == BOT_TAG
    ]
    worker_covered_pairs = {
        trade.get("instrument", "?") for trade in open_trades if get_tag(trade) in WORKER_ORDER_TAGS
    } | {
        order.get("instrument", "?")
        for order in pending_orders
        if is_entry_pending_order(order) and get_tag(order) in WORKER_ORDER_TAGS
    }
    bot_orders_by_pair: dict[str, list[dict]] = {}
    for order in bot_orders:
        bot_orders_by_pair.setdefault(order.get("instrument", "?"), []).append(order)
    for orders in bot_orders_by_pair.values():
        orders.sort(key=pending_order_sort_key, reverse=True)

    handled_order_ids: set[str] = set()
    pending_slots_used: dict[str, int] = {}
    cancelled = 0

    # --- SCAN RANGES ---
    opportunities = scan_ranges(prices)
    if opportunities:
        print(f"\nRanges found: {len(opportunities)}")
    else:
        print("\nRanges found: 0")

    # --- MARGIN BUDGET ---
    budget_pct, budget_reasons = dynamic_bot_budget_profile(policy, opportunities)
    bot_budget = nav * budget_pct
    target_active_pairs = max(0, int(policy.get("target_active_worker_pairs", 0)))
    coverage_repair = len(worker_covered_pairs) < target_active_pairs
    projected_margin = margin_used + sum(
        pending_margin(order, prices) for order in pending_orders if is_entry_pending_order(order)
    )
    projected_headroom = max(0.0, nav * float(policy["projected_margin_cap"]) - projected_margin)
    budget_remaining = min(bot_budget, projected_headroom)
    print(
        f"Bot margin budget: {budget_remaining:,.0f} JPY "
        f"(dynamic_cap={budget_pct:.2f}, projected_headroom={projected_headroom:,.0f}, max_cap={MAX_BOT_MARGIN_PCT:.2f})"
    )
    if budget_reasons:
        print(f"  Budget reason: {', '.join(budget_reasons)}")
    if target_active_pairs > 0:
        print(
            f"  Coverage target: {len(worker_covered_pairs)}/{target_active_pairs} worker pair(s)"
            + (" → repair flat book" if coverage_repair else "")
        )

    # --- PLACE ORDERS ---
    placed = []
    kept = []
    skipped = []
    gtd_time = (now_utc + timedelta(hours=GTD_HOURS)).strftime(OANDA_TIME_FMT)

    for opp in opportunities:
        pair = opp["pair"]
        conv = opp["conviction"]
        signal = opp["active_signal"]
        pair_policy = get_pair_policy(policy, pair)

        # Determine direction
        if "BUY" in signal:
            direction = "BUY"
            entry = opp["buy_entry"]
            base_tp = opp["bb_mid"]
            sl = opp["buy_sl"]
        elif "SELL" in signal:
            direction = "SELL"
            entry = opp["sell_entry"]
            base_tp = opp["bb_mid"]
            sl = opp["sell_sl"]
        else:
            continue

        if not mode_allows_direction(pair_policy["mode"], direction):
            skipped.append(f"{pair}: policy {pair_policy['mode']} blocks {direction}")
            continue

        if not allow_new_entries:
            skipped.append(f"{pair}: policy reduce-only")
            continue

        gate_blocked, gate_reason = brake_gate.check(pair, direction)
        if gate_blocked:
            skipped.append(f"{pair}: brake_gate {gate_reason}")
            continue

        tempo = str(pair_policy.get("tempo", "BALANCED")).upper()
        entry_bias = str(pair_policy.get("entry_bias", "BALANCED")).upper()
        tp1 = base_tp
        if tempo in {"FAST", "MICRO"}:
            tp1 = tempo_tp_target(
                pair,
                direction,
                entry,
                base_tp,
                float(opp.get("spread", 0)),
                max(float(opp.get("atr_pips", 0)), 1.0),
                tempo,
            )

        if tempo == "MICRO":
            min_market_rr = MICRO_MIN_MARKET_RR
        elif tempo == "FAST":
            min_market_rr = FAST_MIN_MARKET_RR
        else:
            min_market_rr = MIN_MARKET_RR
        order_plan = choose_order_plan(
            opp,
            pair,
            direction,
            entry,
            tp1,
            sl,
            prices,
            min_market_rr,
            tempo,
            entry_bias,
            coverage_repair,
            late_session_limit_only,
        )
        order_type = order_plan["order_type"]
        execution_entry = order_plan["entry"]
        structural_sl = float(order_plan.get("sl", sl))
        sl = structural_sl

        if order_type == "MARKET" and not pair_policy["allow_market"]:
            fallback_tp_pips, fallback_sl_pips, fallback_rr = trade_metrics(direction, entry, tp1, sl, pair)
            order_plan = {
                "order_type": "LIMIT",
                "entry": entry,
                "tag": BOT_TAG,
                "reason": f"policy blocks market; wait at BB edge | {order_plan['reason']}",
                "tp_pips": fallback_tp_pips,
                "sl_pips": fallback_sl_pips,
                "rr": fallback_rr,
                "current_price": order_plan["current_price"],
                "progress_to_tp": order_plan["progress_to_tp"],
            }
            order_type = "LIMIT"
            execution_entry = entry

        race_plan = build_range_target_race(pair, direction, execution_entry, tp1, structural_sl, opp, tempo)
        broker_tp = float(race_plan["tp2"])
        # MICRO/FAST: no broker SL — timeout mechanism in bot_trade_manager is the primary protection.
        # BALANCED: keep wide disaster backstop to catch gap events.
        if tempo in ("MICRO", "FAST"):
            broker_sl = None
            broker_sl_pips = 0.0
        else:
            broker_sl, broker_sl_pips = worker_disaster_stop(
                pair,
                direction,
                execution_entry,
                structural_sl,
                tempo=tempo,
                style="range",
                atr_pips=max(float(opp.get("atr_pips", 0)), 1.0),
                spread_pips=max(float(opp.get("spread", 0)), 0.1),
            )
        runner_comment = encode_target_race_comment(race_plan)

        if order_type == "SKIP":
            skipped.append(f"{pair}: {order_plan['reason']}")
            print(f"\n  {pair} {conv}-{('LONG' if direction == 'BUY' else 'SHORT')} SKIP")
            print(f"    {order_plan['reason']}")
            continue

        worker_trades = worker_trade_conflicts(open_trades, pair)
        if worker_trades:
            skipped.append(f"{pair}: worker trade already live ({describe_worker_trades(worker_trades)})")
            continue

        worker_kind = worker_kind_from_order_type(order_type)
        block_reason = pair_policy_worker_block_reason(pair_policy, direction, worker_kind)
        if block_reason:
            skipped.append(f"{pair}: {block_reason}")
            continue

        trade_conflicts = discretionary_open_trade_conflicts(
            open_trades, pair, direction, worker_kind, pair_policy["ownership"]
        )
        if trade_conflicts:
            skipped.append(
                f"{pair}: discretionary open trade blocks ({describe_trade_conflicts(trade_conflicts)})"
            )
            continue

        pending_conflicts = discretionary_pending_conflicts_for_worker(
            pending_orders, pair, direction, worker_kind, pair_policy["ownership"]
        )
        if pending_conflicts:
            skipped.append(
                f"{pair}: discretionary pending entry exists ({describe_pending_conflicts(pending_conflicts)})"
            )
            continue

        cooldown_reason = recent_close_cooldown(
            pair,
            direction,
            now_utc,
            tempo=tempo,
            recent_worker_closes=recent_worker_closes,
        )
        if cooldown_reason:
            skipped.append(f"{pair}: {cooldown_reason}")
            continue

        existing_orders = list(bot_orders_by_pair.get(pair, []))
        same_direction_orders = [order for order in existing_orders if order_direction(order) == direction]
        opposite_direction_orders = [order for order in existing_orders if order_direction(order) != direction]
        pending_cap = max(0, int(pair_policy["max_pending"]))

        for order in opposite_direction_orders:
            oid = str(order.get("id", "?"))
            handled_order_ids.add(oid)
            if args.dry_run:
                print(f"  [DRY] Would cancel: {pair} id={oid} (direction changed)")
            else:
                if cancel_order(token, acct, oid):
                    print(f"  Cancelled: {pair} id={oid} (direction changed)")
                    cancelled += 1

        same_direction_orders = [order for order in same_direction_orders if str(order.get("id", "?")) not in handled_order_ids]
        if pair_policy["pending"] == "CANCEL" or (order_type == "LIMIT" and pending_cap == 0):
            cancel_reason = (
                f"policy {pair_policy['mode']} pending={pair_policy['pending']}"
                if pair_policy["pending"] == "CANCEL"
                else "max_pending=0"
            )
            for order in same_direction_orders:
                oid = str(order.get("id", "?"))
                handled_order_ids.add(oid)
                if args.dry_run:
                    print(f"  [DRY] Would cancel: {pair} id={oid} ({cancel_reason})")
                else:
                    if cancel_order(token, acct, oid):
                        print(f"  Cancelled: {pair} id={oid} ({cancel_reason})")
                        cancelled += 1
            same_direction_orders = []
        passive_block_reason = pair_policy_worker_block_reason(pair_policy, direction, "PASSIVE")
        if same_direction_orders and passive_block_reason:
            for order in same_direction_orders:
                oid = str(order.get("id", "?"))
                handled_order_ids.add(oid)
                if args.dry_run:
                    print(f"  [DRY] Would cancel: {pair} id={oid} ({passive_block_reason})")
                else:
                    if cancel_order(token, acct, oid):
                        print(f"  Cancelled: {pair} id={oid} ({passive_block_reason})")
                        cancelled += 1
            same_direction_orders = []

        if order_type == "LIMIT":
            kept_existing_limit = False
            for order in same_direction_orders:
                oid = str(order.get("id", "?"))
                if pending_slots_used.get(pair, 0) >= pending_cap:
                    keep_reason = f"over max_pending {pending_cap}"
                    handled_order_ids.add(oid)
                    if args.dry_run:
                        print(f"  [DRY] Would cancel: {pair} id={oid} ({keep_reason})")
                    else:
                        if cancel_order(token, acct, oid):
                            print(f"  Cancelled: {pair} id={oid} ({keep_reason})")
                            cancelled += 1
                    continue

                keep_order, keep_reason = should_keep_pending_limit(
                    order, pair, direction, execution_entry, opp, prices, now_utc, max_pending_age_min
                )
                if not keep_order:
                    handled_order_ids.add(oid)
                    if args.dry_run:
                        print(f"  [DRY] Would cancel: {pair} id={oid} ({keep_reason})")
                    else:
                        if cancel_order(token, acct, oid):
                            print(f"  Cancelled: {pair} id={oid} ({keep_reason})")
                            cancelled += 1
                    continue

                existing_entry = float(order.get("price", entry))
                existing_units = abs(int(order.get("units", 0))) or calculate_units(
                    conv, nav, existing_entry, pair, order_type="LIMIT"
                )
                existing_margin = estimate_margin(existing_units, existing_entry, pair)
                if existing_margin > budget_remaining:
                    keep_reason = "keep-order margin budget exhausted"
                    handled_order_ids.add(oid)
                    if args.dry_run:
                        print(f"  [DRY] Would cancel: {pair} id={oid} ({keep_reason})")
                    else:
                        if cancel_order(token, acct, oid):
                            print(f"  Cancelled: {pair} id={oid} ({keep_reason})")
                            cancelled += 1
                    continue

                handled_order_ids.add(oid)
                pending_slots_used[pair] = pending_slots_used.get(pair, 0) + 1
                kept.append({
                    "pair": pair,
                    "direction": direction,
                    "units": existing_units,
                    "entry": existing_entry,
                    "conviction": conv,
                    "order_type": "LIMIT",
                    "order_id": oid,
                })
                budget_remaining -= existing_margin
                side_label = "LONG" if direction == "BUY" else "SHORT"
                print(
                    f"\n  {pair} {conv}-{side_label} KEEP LIMIT "
                    f"{existing_units}u @{format_price(existing_entry, pair)}"
                )
                print(f"    id={oid} | {keep_reason}")
                kept_existing_limit = True
                break

            if kept_existing_limit:
                continue
            if pending_slots_used.get(pair, 0) >= pending_cap:
                skipped.append(f"{pair}: max_pending {pending_cap} reached")
                continue

        elif same_direction_orders:
            for order in same_direction_orders:
                oid = str(order.get("id", "?"))
                handled_order_ids.add(oid)
                if args.dry_run:
                    print(f"  [DRY] Would cancel: {pair} id={oid} (upgrade to MARKET)")
                else:
                    if cancel_order(token, acct, oid):
                        print(f"  Cancelled: {pair} id={oid} (upgrade to MARKET)")
                        cancelled += 1

        # Size
        units = calculate_units(conv, nav, execution_entry, pair, order_type=order_type, tempo=tempo)
        est_margin = estimate_margin(units, execution_entry, pair)

        # Budget check
        if est_margin > budget_remaining:
            skipped.append(f"{pair}: margin budget exhausted")
            break

        # R:R minimum check
        rr_floor = min_market_rr if order_type == "MARKET" else 1.0
        if order_plan["rr"] < rr_floor:
            skipped.append(f"{pair}: {order_type} R:R {order_plan['rr']:.1f} < {rr_floor:.2f}")
            continue

        # Print plan
        side_label = "LONG" if direction == "BUY" else "SHORT"
        print(
            f"\n  {pair} {conv}-{side_label} {order_type} "
            f"{units}u @{format_price(execution_entry, pair)}"
        )
        print(
            f"    TP1={format_price(tp1, pair)} (+{race_plan['tp1_pips']:.1f}pip)"
            f" | TP2={format_price(broker_tp, pair)} (+{race_plan['tp2_pips']:.1f}pip)"
        )
        print(
            f"    Thesis line={format_price(structural_sl, pair)} (-{order_plan['sl_pips']:.1f}pip)"
            + (f" | Disaster SL={format_price(broker_sl, pair)} (-{broker_sl_pips:.1f}pip)" if broker_sl is not None else " | No-SL (timeout mode)")
        )
        print(
            f"    R:R={order_plan['rr']:.1f} | {opp.get('setup_tf', 'M5')} {opp['range_type']} "
            f"| Str={opp['signal_strength']} | {order_plan['reason']}"
        )
        print(f"    Margin ~{est_margin:,.0f} JPY | Triggers: {', '.join(opp['triggers'][:3])}")
        if race_plan.get("eta_fast_bars") is not None or race_plan.get("eta_slow_bars") is not None:
            print(
                f"    Race: TP1~{race_plan.get('eta_fast_bars', '?')} bars | "
                f"TP2~{race_plan.get('eta_slow_bars', '?')} bars | "
                f"hold>{format_price(race_plan['hold_boundary'], pair)} after TP1"
            )
        if opp.get("alternate_views"):
            print(f"    Alt views: {', '.join(opp['alternate_views'])}")

        if args.dry_run:
            print(f"    [DRY RUN — not placed]")
            placed.append({
                "pair": pair, "direction": direction, "units": units,
                "entry": execution_entry, "tp": broker_tp, "sl": broker_sl, "conviction": conv,
                "order_type": order_type,
            })
            if order_type == "LIMIT":
                pending_slots_used[pair] = pending_slots_used.get(pair, 0) + 1
            budget_remaining -= est_margin
            continue

        live_policy, live_policy_notes = load_policy()
        if not global_status_allows_new_entries(live_policy):
            note = f"policy switched to {live_policy['global_status']} before submit"
            if live_policy_notes:
                note = f"{note} ({'; '.join(live_policy_notes)})"
            print(f"    BLOCKED before submit: {note}")
            skipped.append(f"{pair}: {note}")
            continue
        live_pair_policy = get_pair_policy(live_policy, pair)
        live_block_reason = pair_policy_worker_block_reason(live_pair_policy, direction, worker_kind)
        if live_block_reason:
            print(f"    BLOCKED before submit: {live_block_reason}")
            skipped.append(f"{pair}: {live_block_reason} before submit")
            continue

        # Place the order
        if order_type == "MARKET":
            result = place_market(token, acct, pair, direction, units, broker_tp, broker_sl, runner_comment)
        else:
            result = place_limit(token, acct, pair, direction, units, entry, broker_tp, broker_sl, gtd_time, runner_comment)

        if "error" in result:
            print(f"    ERROR: {result['error']}")
            skipped.append(f"{pair}: {order_type} API error")
            continue

        cancel_resp = result.get("orderCancelTransaction", {})
        if cancel_resp:
            reason = cancel_resp.get("reason", "CANCELLED")
            order_id = cancel_resp.get("orderID") or result.get("orderCreateTransaction", {}).get("id", "?")
            print(f"    CANCELLED by OANDA: {reason}")
            log_cancel(order_type, pair, direction, units, execution_entry, order_id, order_plan["tag"], reason)
            skipped.append(f"{pair}: {order_type} cancelled ({reason})")
            continue

        reject_resp = result.get("orderRejectTransaction", {})
        if reject_resp:
            reason = reject_resp.get("rejectReason", "REJECTED")
            order_id = reject_resp.get("id", "?")
            print(f"    REJECTED by OANDA: {reason}")
            log_cancel(order_type, pair, direction, units, execution_entry, order_id, order_plan["tag"], reason)
            skipped.append(f"{pair}: {order_type} rejected ({reason})")
            continue

        # Extract order ID
        order_resp = result.get("orderCreateTransaction", {})
        fill_resp = result.get("orderFillTransaction", {})
        order_id = order_resp.get("id") or fill_resp.get("orderID") or "?"
        fill_price = float(fill_resp.get("price", execution_entry)) if fill_resp else execution_entry
        print(f"    PLACED id={order_id}")
        trade_id = extract_trade_id_from_order_result(result)
        if trade_id:
            remember_trade_plan(trade_id, race_plan, units)

        # Log + Slack
        log_entry(order_type, pair, direction, units, fill_price, broker_tp, broker_sl, order_id, order_plan["tag"])
        slack_notify(order_type, pair, direction, units, fill_price, broker_tp, broker_sl, order_plan["reason"])

        placed.append({
            "pair": pair, "direction": direction, "units": units,
            "entry": fill_price, "tp": broker_tp, "sl": broker_sl, "conviction": conv,
            "order_id": order_id,
            "order_type": order_type,
        })
        if order_type == "LIMIT":
            pending_slots_used[pair] = pending_slots_used.get(pair, 0) + 1
        budget_remaining -= est_margin

    # --- CLEAN UP UNTENDED BOT LIMITS ---
    for order in sorted(bot_orders, key=pending_order_sort_key, reverse=True):
        oid = str(order.get("id", "?"))
        if oid in handled_order_ids:
            continue

        pair = order.get("instrument", "?")
        pair_policy = get_pair_policy(policy, pair)
        direction = order_direction(order)
        pending_cap = max(0, int(pair_policy["max_pending"]))
        worker_trades = worker_trade_conflicts(open_trades, pair)
        passive_block_reason = pair_policy_worker_block_reason(pair_policy, direction, "PASSIVE")
        trade_conflicts = discretionary_open_trade_conflicts(
            open_trades, pair, direction, "PASSIVE", pair_policy["ownership"]
        )
        pending_conflicts = discretionary_pending_conflicts(pending_orders, pair, direction)
        if passive_block_reason:
            reason = passive_block_reason
            should_cancel = True
        elif worker_trades:
            reason = f"worker trade already live ({describe_worker_trades(worker_trades)})"
            should_cancel = True
        elif trade_conflicts:
            reason = f"discretionary open trade blocks ({describe_trade_conflicts(trade_conflicts)})"
            should_cancel = True
        elif pending_conflicts:
            reason = f"discretionary pending entry exists ({describe_pending_conflicts(pending_conflicts)})"
            should_cancel = True
        elif not mode_allows_direction(pair_policy["mode"], direction):
            reason = f"policy {pair_policy['mode']} blocks {direction}"
            should_cancel = True
        elif pair_policy["pending"] == "CANCEL" or pair_policy["mode"] == "PAUSE":
            reason = f"policy {pair_policy['mode']} pending={pair_policy['pending']}"
            should_cancel = True
        elif pending_slots_used.get(pair, 0) >= pending_cap:
            reason = f"over max_pending {pending_cap}"
            should_cancel = True
        else:
            created_at = parse_oanda_time(order.get("createTime"))
            age_min = (now_utc - created_at).total_seconds() / 60 if created_at is not None else 999
            should_cancel = age_min > max_pending_age_min
            reason = f"orphan age {age_min:.0f}m" if should_cancel else "fresh orphan kept"

        if not should_cancel:
            existing_entry = float(order.get("price", 0))
            existing_units = abs(int(order.get("units", 0)))
            if existing_entry > 0 and existing_units > 0:
                budget_remaining -= estimate_margin(existing_units, existing_entry, pair)
                kept.append({
                    "pair": pair,
                    "direction": direction,
                    "units": existing_units,
                    "entry": existing_entry,
                    "conviction": "?",
                    "order_type": "LIMIT",
                    "order_id": oid,
                })
                pending_slots_used[pair] = pending_slots_used.get(pair, 0) + 1
            print(f"  Keeping: {pair} id={oid} ({reason})")
            continue

        if args.dry_run:
            print(f"  [DRY] Would cancel: {pair} id={oid} ({reason})")
        else:
            if cancel_order(token, acct, oid):
                print(f"  Cancelled: {pair} id={oid} ({reason})")
                cancelled += 1

    # --- SUMMARY ---
    print(f"\n{'='*50}")
    print(f"RANGE BOT SUMMARY")
    print(f"  Scanned: {len(PAIRS)} pairs")
    print(f"  Ranges: {len(opportunities)}")
    print(f"  Placed: {len(placed)}")
    print(f"  Kept: {len(kept)}")
    print(f"  Cancelled: {cancelled}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    if kept:
        for p in kept:
            side = "LONG" if p["direction"] == "BUY" else "SHORT"
            print(
                f"    KEEP {p['pair']} {p['conviction']}-{side} {p['order_type']} "
                f"{p['units']}u @{format_price(p['entry'], p['pair'])} id={p['order_id']}"
            )
    if placed:
        for p in placed:
            side = "LONG" if p["direction"] == "BUY" else "SHORT"
            oid = p.get("order_id", "dry")
            print(
                f"    {p['pair']} {p['conviction']}-{side} {p['order_type']} "
                f"{p['units']}u @{format_price(p['entry'], p['pair'])} id={oid}"
            )
    if not placed and not kept:
        print(f"  (no orders placed)")
    print(f"{'='*50}")

    return 0 if placed or kept or cancelled else 1


if __name__ == "__main__":
    sys.exit(main())
