#!/usr/bin/env python3
"""
Live Trading Monitor v3 — data + signals + mechanical position management.

Runs every 30s via launchd. No LLM cost. Pure Python.

What it does:
  1. Data collection: pricing, S5/M1/M5 candles + indicators
  2. Signal scoring: 7 pairs × 2 directions → pre-computed scores
  3. Mechanical position management: trail/partial/close based on trade_registry rules
  4. Risk checks: margin, drawdown, circuit breaker, exposure
  5. Session detection + currency strength

Output: logs/live_monitor.json
Registry: logs/trade_registry.json (read by this script, written by Claude tasks)

Usage:
    python3 scripts/trader_tools/live_monitor.py          # one-shot
    python3 scripts/trader_tools/live_monitor.py --loop 30 # loop every 30s
"""

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tomli
import pandas as pd
from indicators.calc_core import IndicatorEngine

OANDA_BASE = "https://api-fxtrade.oanda.com"
ALL_PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
OUTPUT_PATH = ROOT / "logs" / "live_monitor.json"
REGISTRY_PATH = ROOT / "logs" / "trade_registry.json"
TRADE_LOG_PATH = ROOT / "logs" / "live_trade_log.txt"

SCALP_KEYS = [
    "rsi", "adx", "plus_di", "minus_di", "stoch_rsi", "macd_hist",
    "ema_slope_5", "ema_slope_10", "cci", "bb_upper", "bb_lower", "bb_mid",
    "close", "atr_pips", "regime", "vwap_gap", "bbw",
]

# Default management rules (used when trade not in registry)
DEFAULT_SCALP_RULES = {"trail_at_pip": 5, "partial_at_pip": 8, "max_hold_min": 30, "cut_at_pip": -5, "cut_age_min": 10}
DEFAULT_SWING_RULES = {"trail_at_pip": 8, "partial_at_pip": 15, "max_hold_min": 480, "cut_at_pip": -15, "cut_age_min": 60}

# Risk limits
MAX_MARGIN_USAGE_PCT = 80
MAX_DAILY_DRAWDOWN_PCT = 3.0  # from session start NAV


def _load_config():
    with open(ROOT / "config" / "env.toml", "rb") as f:
        cfg = tomli.load(f)
    return cfg["oanda_token"], cfg["oanda_account_id"]


def _api_get(token: str, path: str, timeout: int = 8):
    req = urllib.request.Request(
        f"{OANDA_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _api_put(token: str, path: str, body: dict, timeout: int = 8):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OANDA_BASE}{path}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="PUT",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _log_action(msg: str):
    """Append to live trade log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(f"[{ts}] MONITOR: {msg}\n")


# ─────────────────────────────────────────────
# Data Fetchers
# ─────────────────────────────────────────────

def fetch_pricing(token: str, acc: str) -> dict:
    instruments = ",".join(ALL_PAIRS)
    data = _api_get(token, f"/v3/accounts/{acc}/pricing?instruments={instruments}")
    result = {}
    for p in data.get("prices", []):
        pair = p["instrument"]
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        bid = float(bids[0]["price"]) if bids else 0
        ask = float(asks[0]["price"]) if asks else 0
        pip = _pip_size(pair)
        spread = round((ask - bid) / pip, 1)
        result[pair] = {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 5), "spread_pips": spread}
    return result


def fetch_s5_candles(token: str, pair: str, count: int = 24) -> list:
    try:
        data = _api_get(token, f"/v3/instruments/{pair}/candles?granularity=S5&count={count}&price=M", timeout=5)
        return [{"time": c["time"], "close": float(c["mid"]["c"]),
                 "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])}
                for c in data.get("candles", [])]
    except Exception:
        return []


def compute_micro_momentum(s5_candles: list, pip_size: float) -> dict:
    if len(s5_candles) < 6:
        return {"direction": "FLAT", "velocity": 0, "range_pips": 0, "diff_pips": 0}
    closes = [c["close"] for c in s5_candles]
    highs = [c["high"] for c in s5_candles]
    lows = [c["low"] for c in s5_candles]
    half = len(closes) // 2
    first_avg = sum(closes[:half]) / half
    last_avg = sum(closes[half:]) / (len(closes) - half)
    diff_pips = round((last_avg - first_avg) / pip_size, 1)
    direction = "UP" if diff_pips > 0.5 else ("DOWN" if diff_pips < -0.5 else "FLAT")
    velocity = round(diff_pips / (len(s5_candles) * 5 / 60), 2)
    range_pips = round((max(highs) - min(lows)) / pip_size, 1)
    return {"direction": direction, "velocity": velocity, "range_pips": range_pips, "diff_pips": diff_pips}


def fetch_candles_and_compute(token: str, pair: str, granularity: str, count: int) -> dict:
    try:
        data = _api_get(token, f"/v3/instruments/{pair}/candles?granularity={granularity}&count={count}&price=M", timeout=8)
        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", False) and granularity not in ("M1", "S5"):
                continue
            mid = c["mid"]
            candles.append({"time": c["time"], "open": float(mid["o"]),
                            "high": float(mid["h"]), "low": float(mid["l"]), "close": float(mid["c"])})
        if len(candles) < 20:
            return {}
        df = pd.DataFrame(candles).rename(columns={"time": "timestamp"})
        factors = IndicatorEngine.compute(df)
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in factors.items() if k in SCALP_KEYS and v is not None}
    except Exception as e:
        return {"error": str(e)}


def fetch_positions(token: str, acc: str) -> tuple:
    trades_data = _api_get(token, f"/v3/accounts/{acc}/openTrades")
    summary_data = _api_get(token, f"/v3/accounts/{acc}/summary")
    positions = []
    for t in trades_data.get("trades", []):
        units = int(t["currentUnits"])
        pair = t["instrument"]
        pip = _pip_size(pair)
        entry = float(t["price"])
        upl = float(t.get("unrealizedPL", "0"))
        mid_price = entry  # approximation

        sl_order = t.get("stopLossOrder", {})
        tp_order = t.get("takeProfitOrder", {})
        trail_order = t.get("trailingStopLossOrder", {})
        sl = float(sl_order["price"]) if sl_order.get("price") else None
        tp = float(tp_order["price"]) if tp_order.get("price") else None
        trail_dist = float(trail_order["distance"]) if trail_order.get("distance") else None

        open_time = t["openTime"][:19]
        try:
            opened = datetime(int(open_time[:4]), int(open_time[5:7]), int(open_time[8:10]),
                              int(open_time[11:13]), int(open_time[14:16]), int(open_time[17:19]),
                              tzinfo=timezone.utc)
            age_min = round((datetime.now(timezone.utc) - opened).total_seconds() / 60, 1)
        except Exception:
            age_min = 0

        # UPL in pips: use entry vs current mid price
        abs_units = abs(units)
        upl_pips = 0
        if abs_units > 0 and pip > 0:
            # Simple: UPL(JPY) / (abs_units * pip_value_in_JPY_per_pip)
            # For JPY pairs: pip_value = abs_units * 0.01 → upl_pips = upl / (abs_units * 0.01)
            # For USD pairs: pip_value = abs_units * 0.0001 * JPY_per_USD
            # Approximate: just use upl / (abs_units * pip) for JPY pairs
            # For non-JPY, upl is in account currency (JPY), need conversion
            if "JPY" in pair and pair.endswith("JPY"):
                # e.g., USD_JPY: 1pip = 0.01, pip_value = units * 0.01
                upl_pips = round(upl / (abs_units * 0.01), 1)
            else:
                # e.g., EUR_USD: 1pip = 0.0001, pip_value in USD = units * 0.0001
                # but UPL is in JPY, so need ~159 conversion
                upl_pips = round(upl / (abs_units * 0.0001 * 159), 1)  # rough JPY conversion

        positions.append({
            "id": t["id"], "pair": pair, "units": units, "entry": entry,
            "upl": upl, "upl_pips": upl_pips,
            "sl": sl, "tp": tp, "trail": trail_dist,
            "has_trail": trail_dist is not None,
            "is_be": sl is not None and abs(sl - entry) < pip * 0.5,
            "age_min": age_min, "opened": open_time,
        })

    s = summary_data["account"]
    account = {
        "nav": round(float(s["NAV"])),
        "upl": round(float(s["unrealizedPL"])),
        "margin_used": round(float(s["marginUsed"])),
        "margin_avail": round(float(s["marginAvailable"])),
        "open_trades": int(s["openTradeCount"]),
        "balance": round(float(s["balance"])),
        "pl": round(float(s.get("pl", "0"))),
    }
    return positions, account


def load_h1_bias() -> dict:
    bias = {}
    for pair in ALL_PAIRS:
        path = ROOT / "logs" / f"technicals_{pair}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            h1 = data.get("timeframes", {}).get("H1", {})
            h4 = data.get("timeframes", {}).get("H4", {})
            bias[pair] = {
                "h1_regime": h1.get("regime", "unknown"), "h1_adx": h1.get("adx"),
                "h1_rsi": h1.get("rsi"), "h1_plus_di": h1.get("plus_di"), "h1_minus_di": h1.get("minus_di"),
                "h4_regime": h4.get("regime", "unknown"), "h4_adx": h4.get("adx"),
                "h4_ema_slope_5": h4.get("ema_slope_5"),
            }
        except Exception:
            continue
    return bias


# ─────────────────────────────────────────────
# Signal Scoring
# ─────────────────────────────────────────────

def score_pair(pair_data: dict, direction: str) -> tuple:
    """Score a pair for entry. Returns (score, reasons)."""
    m5 = pair_data.get("m5", {})
    m1 = pair_data.get("m1", {})
    micro = pair_data.get("micro", {})
    bias = pair_data.get("bias", {})
    price = pair_data.get("price", {})
    score = 0
    reasons = []

    # 1. M5 ADX > 20 (trending)
    if m5.get("adx", 0) and m5["adx"] > 20:
        score += 1
        reasons.append("M5_ADX>" + str(round(m5["adx"])))

    # 2. M1+M5 RSI alignment
    m5_rsi = m5.get("rsi", 50)
    m1_rsi = m1.get("rsi", 50)
    if direction == "LONG" and m5_rsi > 50 and m1_rsi > 50:
        score += 1
        reasons.append("RSI_aligned_bull")
    elif direction == "SHORT" and m5_rsi < 50 and m1_rsi < 50:
        score += 1
        reasons.append("RSI_aligned_bear")

    # 3. Micro-momentum aligned
    micro_dir = micro.get("direction", "FLAT")
    if (direction == "LONG" and micro_dir == "UP") or (direction == "SHORT" and micro_dir == "DOWN"):
        score += 1
        reasons.append(f"micro_{micro_dir}")

    # 4. H1 bias supports
    h1_regime = bias.get("h1_regime", "")
    h1_plus = bias.get("h1_plus_di", 0) or 0
    h1_minus = bias.get("h1_minus_di", 0) or 0
    if direction == "LONG" and h1_plus > h1_minus:
        score += 1
        reasons.append("H1_bull")
    elif direction == "SHORT" and h1_minus > h1_plus:
        score += 1
        reasons.append("H1_bear")

    # 5. Spread OK (<2pip)
    spread = price.get("spread_pips", 99)
    if spread < 2.0:
        score += 1
        reasons.append(f"spread={spread}")

    return score, reasons


# ─────────────────────────────────────────────
# Mechanical Position Management
# ─────────────────────────────────────────────

def load_registry() -> dict:
    """Load trade registry. Returns {trade_id: {owner, type, rules, ...}}"""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH) as f:
            data = json.load(f)
        return {str(t["trade_id"]): t for t in data if "trade_id" in t}
    except Exception:
        return {}


def save_registry(registry: dict):
    """Save trade registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(list(registry.values()), f, indent=2)


def get_rules_for_trade(trade_id: str, registry: dict) -> dict:
    """Get management rules for a trade. Falls back to default scalp rules."""
    entry = registry.get(trade_id, {})
    rules = entry.get("rules", {})
    if rules:
        return rules
    trade_type = entry.get("type", "scalp")
    return DEFAULT_SWING_RULES if trade_type == "swing" else DEFAULT_SCALP_RULES


def manage_positions(token: str, acc: str, positions: list, pricing: dict) -> list:
    """Apply mechanical management rules. Returns list of actions taken."""
    registry = load_registry()
    actions = []

    for pos in positions:
        tid = pos["id"]
        pair = pos["pair"]
        upl_pips = pos.get("upl_pips", 0)
        age_min = pos.get("age_min", 0)
        has_trail = pos.get("has_trail", False)
        units = pos["units"]
        abs_units = abs(units)
        pip = _pip_size(pair)

        rules = get_rules_for_trade(tid, registry)
        trail_at = rules.get("trail_at_pip", 5)
        partial_at = rules.get("partial_at_pip", 8)
        max_hold = rules.get("max_hold_min", 30)
        cut_at = rules.get("cut_at_pip", -5)
        cut_age = rules.get("cut_age_min", 10)

        action = None

        # Rule 1: Set trailing stop if profit exceeds threshold and no trail yet
        if upl_pips >= trail_at and not has_trail:
            trail_distance = str(round(trail_at * pip, 5))
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/orders", {
                    "trailingStopLoss": {"distance": trail_distance}
                })
                action = f"TRAIL_SET {pair} {units}u trail={trail_at}pip (UPL={upl_pips}pip)"
            except Exception as e:
                action = f"TRAIL_FAIL {pair}: {e}"

        # Rule 2: Partial close if profit exceeds threshold
        elif upl_pips >= partial_at:
            half = max(abs_units // 2, 1)
            close_units = str(half)  # positive number for close
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {"units": close_units})
                action = f"PARTIAL {pair} closed {half}u of {units}u (UPL={upl_pips}pip)"
            except Exception as e:
                action = f"PARTIAL_FAIL {pair}: {e}"

        # Rule 3: Cut loss if negative and old enough
        elif upl_pips <= cut_at and age_min >= cut_age:
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {})
                action = f"CUT {pair} {units}u (UPL={upl_pips}pip, age={age_min}min)"
            except Exception as e:
                action = f"CUT_FAIL {pair}: {e}"

        # Rule 4: Close if held too long with insufficient profit
        elif age_min >= max_hold and upl_pips < trail_at * 0.6:
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {})
                action = f"TIMEOUT {pair} {units}u (UPL={upl_pips}pip, age={age_min}min>{max_hold}min)"
            except Exception as e:
                action = f"TIMEOUT_FAIL {pair}: {e}"

        if action:
            actions.append(action)
            _log_action(action)

    # Clean registry of closed trades
    open_ids = {pos["id"] for pos in positions}
    cleaned = {k: v for k, v in registry.items() if k in open_ids}
    if len(cleaned) != len(registry):
        save_registry(cleaned)

    return actions


# ─────────────────────────────────────────────
# Risk & Session
# ─────────────────────────────────────────────

def compute_risk(account: dict, positions: list) -> dict:
    """Compute risk metrics."""
    nav = account.get("nav", 0)
    margin_used = account.get("margin_used", 0)
    margin_usage_pct = round(margin_used / nav * 100, 1) if nav > 0 else 0

    # Currency exposure (net units per currency)
    exposure = {}
    for pos in positions:
        pair = pos["pair"]
        units = pos["units"]
        base, quote = pair.split("_")
        exposure[base] = exposure.get(base, 0) + units
        exposure[quote] = exposure.get(quote, 0) - units

    # Circuit breaker: daily drawdown
    balance = account.get("balance", nav)
    daily_pnl = nav - balance  # approximate intraday P/L
    daily_drawdown_pct = round(abs(min(daily_pnl, 0)) / balance * 100, 2) if balance > 0 else 0
    circuit_breaker = (daily_drawdown_pct >= MAX_DAILY_DRAWDOWN_PCT) or (margin_usage_pct >= MAX_MARGIN_USAGE_PCT)

    return {
        "margin_usage_pct": margin_usage_pct,
        "daily_drawdown_pct": daily_drawdown_pct,
        "circuit_breaker": circuit_breaker,
        "circuit_reason": ("drawdown" if daily_drawdown_pct >= MAX_DAILY_DRAWDOWN_PCT
                           else "margin" if margin_usage_pct >= MAX_MARGIN_USAGE_PCT else None),
        "exposure": exposure,
    }


def detect_session() -> dict:
    """Detect current trading session."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 6:
        session = "TOKYO"
        volatility = "low"
    elif 6 <= hour < 8:
        session = "TOKYO_LONDON_OVERLAP"
        volatility = "rising"
    elif 8 <= hour < 12:
        session = "LONDON"
        volatility = "high"
    elif 12 <= hour < 16:
        session = "LONDON_NY_OVERLAP"
        volatility = "highest"
    elif 16 <= hour < 21:
        session = "NEW_YORK"
        volatility = "medium"
    else:
        session = "LATE_NY"
        volatility = "low"
    return {"session": session, "volatility": volatility, "utc_hour": hour}


def compute_currency_strength(pricing: dict) -> dict:
    """Simple currency strength from price changes vs reference."""
    # We use micro-momentum direction across pairs as a proxy
    # USD appears in: USD_JPY(base), EUR_USD(quote), GBP_USD(quote), AUD_USD(quote)
    # If USD_JPY up AND EUR_USD down AND GBP_USD down → USD strong
    strength = {"USD": 0, "EUR": 0, "GBP": 0, "AUD": 0, "JPY": 0}

    for pair, data in pricing.items():
        spread = data.get("spread_pips", 99)
        if spread > 5:
            continue
        base, quote = pair.split("_")
        mid = data.get("mid", 0)
        if mid == 0:
            continue
        # We'd need price change for real strength, but we can approximate
        # from the existing micro data in pair_data. For now, just return the structure.

    return strength


# ─────────────────────────────────────────────
# Main Builder
# ─────────────────────────────────────────────

def build_monitor() -> dict:
    token, acc = _load_config()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch data
    pricing = fetch_pricing(token, acc)
    positions, account = fetch_positions(token, acc)
    h1_bias = load_h1_bias()

    # Mechanical position management (EXECUTES TRADES)
    actions = manage_positions(token, acc, positions, pricing)

    # Re-fetch positions if actions were taken (positions may have changed)
    if actions:
        positions, account = fetch_positions(token, acc)

    # Per-pair data
    pairs = {}
    for pair in ALL_PAIRS:
        pip = _pip_size(pair)
        price_data = pricing.get(pair, {})
        s5 = fetch_s5_candles(token, pair, count=24)
        micro = compute_micro_momentum(s5, pip)
        m1 = fetch_candles_and_compute(token, pair, "M1", 100)
        m5 = fetch_candles_and_compute(token, pair, "M5", 100)
        bias = h1_bias.get(pair, {})

        pair_data = {"price": price_data, "micro": micro, "m1": m1, "m5": m5, "bias": bias}

        # Signal scoring
        long_score, long_reasons = score_pair(pair_data, "LONG")
        short_score, short_reasons = score_pair(pair_data, "SHORT")
        best_dir = "LONG" if long_score > short_score else "SHORT"
        best_score = max(long_score, short_score)
        pair_data["signal"] = {
            "long_score": long_score, "long_reasons": long_reasons,
            "short_score": short_score, "short_reasons": short_reasons,
            "best": best_dir, "best_score": best_score,
        }
        pairs[pair] = pair_data

    # Risk & session
    risk = compute_risk(account, positions)
    session = detect_session()

    return {
        "timestamp": now,
        "pairs": pairs,
        "positions": positions,
        "account": account,
        "risk": risk,
        "session": session,
        "actions_taken": actions,
    }


def main():
    loop_sec = None
    for arg in sys.argv[1:]:
        if arg == "--loop":
            idx = sys.argv.index("--loop")
            if idx + 1 < len(sys.argv):
                loop_sec = int(sys.argv[idx + 1])

    while True:
        try:
            monitor = build_monitor()
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(monitor, f, indent=2)
            ts = monitor["timestamp"]
            n_pos = len(monitor["positions"])
            nav = monitor["account"]["nav"]
            n_act = len(monitor.get("actions_taken", []))
            risk = monitor.get("risk", {})
            cb = " CIRCUIT_BREAKER!" if risk.get("circuit_breaker") else ""
            print(f"[{ts}] Monitor: {n_pos} pos, NAV={nav}, actions={n_act}{cb}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        if loop_sec is None:
            break
        time.sleep(loop_sec)


if __name__ == "__main__":
    main()
