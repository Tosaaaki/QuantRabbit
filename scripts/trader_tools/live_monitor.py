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
RECENTLY_CLOSED_PATH = ROOT / "logs" / "recently_closed.json"

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

# Leverage for JPY account (OANDA Japan)
LEVERAGE = 25

# Updated dynamically in build_monitor() from pricing data
_USDJPY_RATE = 159  # fallback


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

        # Read clientExtensions (tag/comment set at order time)
        client_ext = t.get("clientExtensions", {})
        trade_tag = client_ext.get("tag", "")  # "scalp" or "swing"
        trade_comment = client_ext.get("comment", "")

        open_time = t["openTime"][:19]
        try:
            opened = datetime(int(open_time[:4]), int(open_time[5:7]), int(open_time[8:10]),
                              int(open_time[11:13]), int(open_time[14:16]), int(open_time[17:19]),
                              tzinfo=timezone.utc)
            age_min = round((datetime.now(timezone.utc) - opened).total_seconds() / 60, 1)
        except Exception:
            age_min = 0

        # SL distance in pips (used for trade type inference)
        sl_pips = None
        if sl is not None:
            sl_pips = round(abs(sl - entry) / pip, 1)

        # UPL in pips: use entry vs current mid price
        abs_units = abs(units)
        upl_pips = 0
        if abs_units > 0 and pip > 0:
            if "JPY" in pair and pair.endswith("JPY"):
                upl_pips = round(upl / (abs_units * 0.01), 1)
            else:
                upl_pips = round(upl / (abs_units * 0.0001 * _USDJPY_RATE), 1)  # JPY conversion

        positions.append({
            "id": t["id"], "pair": pair, "units": units, "entry": entry,
            "upl": upl, "upl_pips": upl_pips,
            "sl": sl, "tp": tp, "sl_pips": sl_pips, "trail": trail_dist,
            "has_trail": trail_dist is not None,
            "is_be": sl is not None and abs(sl - entry) < pip * 0.5,
            "age_min": age_min, "opened": open_time,
            "tag": trade_tag, "comment": trade_comment,
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
# Recently Closed Tracking (prevents duplicate close)
# ─────────────────────────────────────────────

def load_recently_closed() -> dict:
    """Load recently closed trade IDs with timestamps."""
    if not RECENTLY_CLOSED_PATH.exists():
        return {}
    try:
        with open(RECENTLY_CLOSED_PATH) as f:
            data = json.load(f)
        # Expire entries older than 10 minutes
        now = datetime.now(timezone.utc)
        return {k: v for k, v in data.items()
                if (now - datetime.fromisoformat(v["closed_at"])).total_seconds() < 600}
    except Exception:
        return {}


def mark_closed(trade_id: str, pair: str, reason: str):
    """Record a trade as recently closed."""
    closed = load_recently_closed()
    closed[str(trade_id)] = {
        "pair": pair,
        "reason": reason,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    RECENTLY_CLOSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECENTLY_CLOSED_PATH, "w") as f:
        json.dump(closed, f, indent=2)


# ─────────────────────────────────────────────
# Position Size Calculator
# ─────────────────────────────────────────────

def compute_max_units(pair: str, nav: float, margin_avail: float,
                      trade_type: str = "scalp", atr_pips: float = None,
                      pair_price: float = 0) -> dict:
    """Calculate max position size based on margin and risk.

    Returns dict with max_units, recommended_units, and reasoning.
    pair_price: mid price of the pair (used to compute margin per unit in JPY).
    """
    # Margin-based limit: margin_used must stay below (1 - free_target) * NAV
    margin_free_target = 0.60 if trade_type == "scalp" else 0.70
    current_margin_used = nav - margin_avail
    max_margin_allowed = nav * (1 - margin_free_target)
    margin_budget = max(max_margin_allowed - current_margin_used, 0)

    # Margin per unit: base currency value in JPY / leverage
    # For XXX_JPY: 1 unit of base = price JPY → margin = price / 25
    # For XXX_USD (e.g. EUR_USD): 1 unit of base ≈ pair_price * USDJPY / leverage
    # We read USDJPY from pricing if available, else fallback to 159
    if pair.endswith("JPY"):
        margin_per_unit = pair_price / LEVERAGE if pair_price > 0 else 8.0
    else:
        usdjpy_rate = _USDJPY_RATE or 159
        margin_per_unit = (pair_price * usdjpy_rate) / LEVERAGE if pair_price > 0 else 8.0
    max_by_margin = int(margin_budget / margin_per_unit) if margin_per_unit > 0 else 0

    # Risk-based limit: max 1% of NAV per trade loss at SL
    max_loss_jpy = nav * 0.01
    pip = _pip_size(pair)
    sl_pips = 5 if trade_type == "scalp" else 15  # default SL

    if "JPY" in pair and pair.endswith("JPY"):
        loss_per_unit_at_sl = sl_pips * pip  # JPY per unit per pip
    else:
        usdjpy_rate = _USDJPY_RATE or 159
        loss_per_unit_at_sl = sl_pips * pip * usdjpy_rate  # convert to JPY

    max_by_risk = int(max_loss_jpy / loss_per_unit_at_sl) if loss_per_unit_at_sl > 0 else 0

    # ATR-based adjustment: reduce size if volatility is high
    if atr_pips and atr_pips > 0:
        normal_atr = 5.0 if trade_type == "scalp" else 10.0
        vol_ratio = min(atr_pips / normal_atr, 3.0)
        max_by_risk = int(max_by_risk / vol_ratio)

    max_units = min(max_by_margin, max_by_risk)

    # Round to nearest 100 (OANDA accepts any integer, but cleaner)
    recommended = (max_units // 100) * 100
    recommended = max(recommended, 0)

    return {
        "max_units": max_units,
        "recommended_units": recommended,
        "max_by_margin": max_by_margin,
        "max_by_risk": max_by_risk,
        "margin_budget_jpy": round(margin_budget),
        "margin_free_target_pct": int(margin_free_target * 100),
        "trade_type": trade_type,
        "can_trade": recommended >= 100,
    }


# ─────────────────────────────────────────────
# Signal Scoring v2 — Direction + Timing + Macro
# ─────────────────────────────────────────────

def load_macro_bias() -> dict:
    """Load macro bias from shared_state.json (written by macro-intel)."""
    try:
        path = ROOT / "logs" / "shared_state.json"
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        return data.get("macro_bias", {})
    except Exception:
        return {}


def score_pair(pair_data: dict, direction: str, pair: str = "", macro_bias: dict = None) -> tuple:
    """Score a pair for scalp entry. Returns (score, reasons).

    Measures "should I enter NOW?" not just "is there a trend?"
    Range: -3 to +7. Minimum 4 recommended for entry.

    A. Direction (trend on my side?)  -> up to +3
    B. Timing (good entry point?)    -> up to +2
    C. Macro alignment               -> +1 or PENALTY -2
    D. Penalties                     -> down to -2
    Spread is gate (pass/fail), not scored.
    """
    m5 = pair_data.get("m5", {})
    m1 = pair_data.get("m1", {})
    bias = pair_data.get("bias", {})
    price = pair_data.get("price", {})
    score = 0
    reasons = []

    # --- GATE: Spread must be < 2pip ---
    spread = price.get("spread_pips", 99)
    if spread >= 2.0:
        return 0, [f"GATE_FAIL:spread={spread}"]

    # === A. DIRECTION (up to +3) ===

    # A1. H1 bias alignment (+1)
    h1_plus = bias.get("h1_plus_di", 0) or 0
    h1_minus = bias.get("h1_minus_di", 0) or 0
    if direction == "LONG" and h1_plus > h1_minus:
        score += 1
        reasons.append("H1_bull")
    elif direction == "SHORT" and h1_minus > h1_plus:
        score += 1
        reasons.append("H1_bear")

    # A2. M5 trend: ADX > 20 AND DI aligned (+1)
    m5_adx = m5.get("adx", 0) or 0
    m5_plus = m5.get("plus_di", 0) or 0
    m5_minus = m5.get("minus_di", 0) or 0
    if m5_adx > 20:
        if (direction == "LONG" and m5_plus > m5_minus) or \
           (direction == "SHORT" and m5_minus > m5_plus):
            score += 1
            reasons.append(f"M5_trend(ADX={round(m5_adx)},DI_ok)")

    # A3. M1+M5 RSI alignment (+1)
    m5_rsi = m5.get("rsi", 50) or 50
    m1_rsi = m1.get("rsi", 50) or 50
    if direction == "LONG" and m5_rsi > 50 and m1_rsi > 50:
        score += 1
        reasons.append(f"RSI_aligned(M5={round(m5_rsi)},M1={round(m1_rsi)})")
    elif direction == "SHORT" and m5_rsi < 50 and m1_rsi < 50:
        score += 1
        reasons.append(f"RSI_aligned(M5={round(m5_rsi)},M1={round(m1_rsi)})")

    # === B. TIMING (up to +2) ===

    # B1. M1 Stoch RSI extreme (+1)
    m1_stoch = m1.get("stoch_rsi")
    if m1_stoch is not None:
        if direction == "LONG" and m1_stoch < 0.2:
            score += 1
            reasons.append(f"TIMING:M1_stoch_oversold({round(m1_stoch, 2)})")
        elif direction == "SHORT" and m1_stoch > 0.8:
            score += 1
            reasons.append(f"TIMING:M1_stoch_overbought({round(m1_stoch, 2)})")

    # B2. M1 BB band edge — price near band in trade direction (+1)
    m1_close = m1.get("close", 0) or 0
    m1_bb_upper = m1.get("bb_upper", 0) or 0
    m1_bb_lower = m1.get("bb_lower", 0) or 0
    if m1_bb_upper and m1_bb_lower and m1_close:
        bb_range = m1_bb_upper - m1_bb_lower
        if bb_range > 0:
            if direction == "LONG" and (m1_close - m1_bb_lower) < bb_range * 0.2:
                score += 1
                reasons.append("TIMING:M1_BB_lower_bounce")
            elif direction == "SHORT" and (m1_bb_upper - m1_close) < bb_range * 0.2:
                score += 1
                reasons.append("TIMING:M1_BB_upper_reject")

    # === C. MACRO ALIGNMENT (+1 or -2) ===
    if macro_bias and pair in macro_bias:
        mb = macro_bias[pair]
        macro_score = mb.get("score", 0) or 0
        macro_label = mb.get("bias", "NEUTRAL")
        if (direction == "LONG" and macro_score > 0) or \
           (direction == "SHORT" and macro_score < 0):
            score += 1
            reasons.append(f"MACRO_OK({macro_label})")
        elif (direction == "LONG" and macro_score < 0) or \
             (direction == "SHORT" and macro_score > 0):
            score -= 2
            reasons.append(f"!MACRO_CONFLICT({macro_label})")

    # === D. PENALTIES ===

    # D1. M5 choppy (ADX < 15) — noise kills scalps (-1)
    if m5_adx and m5_adx < 15:
        score -= 1
        reasons.append(f"PENALTY:choppy(M5_ADX={round(m5_adx)})")

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


def infer_trade_type(pos: dict) -> str:
    """Infer trade type from OANDA clientExtensions tag or SL distance.

    Priority:
      1. clientExtensions.tag (set at order time by Claude agents)
      2. SL distance heuristic (SL >= 8pip → swing, else scalp)
      3. Default: scalp
    """
    tag = pos.get("tag", "").lower()
    if tag in ("scalp", "swing"):
        return tag

    # Fallback: SL distance heuristic
    sl_pips = pos.get("sl_pips")
    if sl_pips is not None:
        return "swing" if sl_pips >= 8 else "scalp"

    return "scalp"


def get_rules_for_trade(trade_id: str, registry: dict, pos: dict = None) -> dict:
    """Get management rules for a trade.

    Priority:
      1. Registry custom rules (explicit override by Claude agent)
      2. OANDA tag / SL-distance inference → default rules for that type
      3. Default scalp rules
    """
    # Layer 1: Registry has explicit custom rules
    entry = registry.get(trade_id, {})
    rules = entry.get("rules", {})
    if rules:
        return rules

    # Layer 2: Infer type from OANDA data (tag or SL distance)
    if pos:
        trade_type = infer_trade_type(pos)
    else:
        trade_type = entry.get("type", "scalp")

    return DEFAULT_SWING_RULES.copy() if trade_type == "swing" else DEFAULT_SCALP_RULES.copy()


def manage_positions(token: str, acc: str, positions: list, pricing: dict) -> list:
    """Apply mechanical management rules. Returns list of actions taken.

    Features:
    - recently_closed tracking to prevent duplicate close attempts
    - Catches OANDA 404 (trade already closed) gracefully
    - Logs all actions with rules_source for audit
    """
    registry = load_registry()
    recently_closed = load_recently_closed()
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

        # Skip if recently closed by another process
        if tid in recently_closed:
            continue

        inferred_type = infer_trade_type(pos)
        rules = get_rules_for_trade(tid, registry, pos)
        rules_source = "registry" if tid in registry and registry[tid].get("rules") else f"inferred:{inferred_type}"
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
                action = f"TRAIL_SET {pair} {units}u trail={trail_at}pip (UPL={upl_pips}pip) [{rules_source}]"
            except Exception as e:
                if "404" in str(e) or "TRADE_DOESNT_EXIST" in str(e):
                    mark_closed(tid, pair, "gone_during_trail")
                    action = f"TRAIL_SKIP {pair}: trade already closed"
                else:
                    action = f"TRAIL_FAIL {pair}: {e}"

        # Rule 2: Partial close if profit exceeds threshold
        elif upl_pips >= partial_at:
            half = max(abs_units // 2, 1)
            close_units = str(half)  # positive number for close
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {"units": close_units})
                action = f"PARTIAL {pair} closed {half}u of {units}u (UPL={upl_pips}pip) [{rules_source}]"
            except Exception as e:
                if "404" in str(e) or "TRADE_DOESNT_EXIST" in str(e):
                    mark_closed(tid, pair, "gone_during_partial")
                    action = f"PARTIAL_SKIP {pair}: trade already closed"
                else:
                    action = f"PARTIAL_FAIL {pair}: {e}"

        # Rule 3: Cut loss if negative and old enough
        elif upl_pips <= cut_at and age_min >= cut_age:
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {})
                mark_closed(tid, pair, "cut")
                action = f"CUT {pair} {units}u (UPL={upl_pips}pip, age={age_min}min) [{rules_source}]"
            except Exception as e:
                if "404" in str(e) or "TRADE_DOESNT_EXIST" in str(e):
                    mark_closed(tid, pair, "gone_during_cut")
                    action = f"CUT_SKIP {pair}: trade already closed"
                else:
                    action = f"CUT_FAIL {pair}: {e}"

        # Rule 4: Close if held too long with insufficient profit
        elif age_min >= max_hold and upl_pips < trail_at * 0.6:
            try:
                _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {})
                mark_closed(tid, pair, "timeout")
                action = f"TIMEOUT {pair} {units}u (UPL={upl_pips}pip, age={age_min}min>{max_hold}min) [{rules_source}]"
            except Exception as e:
                if "404" in str(e) or "TRADE_DOESNT_EXIST" in str(e):
                    mark_closed(tid, pair, "gone_during_timeout")
                    action = f"TIMEOUT_SKIP {pair}: trade already closed"
                else:
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

    # Update global USD/JPY rate for margin calculations
    global _USDJPY_RATE
    usdjpy_price = pricing.get("USD_JPY", {}).get("mid", 0)
    if usdjpy_price > 0:
        _USDJPY_RATE = usdjpy_price

    # Mechanical position management (EXECUTES TRADES)
    actions = manage_positions(token, acc, positions, pricing)

    # Re-fetch positions if actions were taken (positions may have changed)
    if actions:
        positions, account = fetch_positions(token, acc)

    # Annotate positions with inferred type + rules source for visibility
    registry = load_registry()
    for pos in positions:
        tid = pos["id"]
        pos["inferred_type"] = infer_trade_type(pos)
        pos["rules_source"] = "registry" if tid in registry and registry[tid].get("rules") else f"inferred:{pos['inferred_type']}"

    # Load macro bias for scoring
    macro_bias = load_macro_bias()

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

        # Signal scoring v2 (direction + timing + macro)
        long_score, long_reasons = score_pair(pair_data, "LONG", pair, macro_bias)
        short_score, short_reasons = score_pair(pair_data, "SHORT", pair, macro_bias)
        best_dir = "LONG" if long_score > short_score else "SHORT"
        best_score = max(long_score, short_score)
        pair_data["signal"] = {
            "long_score": long_score, "long_reasons": long_reasons,
            "short_score": short_score, "short_reasons": short_reasons,
            "best": best_dir, "best_score": best_score,
        }

        # Position sizing: pre-compute max units for this pair
        m5_atr = m5.get("atr_pips") if isinstance(m5, dict) else None
        mid_price = price_data.get("mid", 0)
        pair_data["sizing"] = {
            "scalp": compute_max_units(pair, account["nav"], account["margin_avail"], "scalp", m5_atr, mid_price),
            "swing": compute_max_units(pair, account["nav"], account["margin_avail"], "swing", m5_atr, mid_price),
        }

        pairs[pair] = pair_data

    # Risk & session
    risk = compute_risk(account, positions)
    session = detect_session()

    # Recently closed trades (for Claude to check before closing)
    recently_closed = load_recently_closed()

    return {
        "timestamp": now,
        "pairs": pairs,
        "positions": positions,
        "account": account,
        "risk": risk,
        "session": session,
        "actions_taken": actions,
        "recently_closed": recently_closed,
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
