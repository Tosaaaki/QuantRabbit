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
import urllib.error
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
SUMMARY_PATH = ROOT / "logs" / "live_monitor_summary.json"
REGISTRY_PATH = ROOT / "logs" / "trade_registry.json"
TRADE_LOG_PATH = ROOT / "logs" / "live_trade_log.txt"
CRITICAL_LOG_PATH = ROOT / "logs" / "critical_events.log"
RECENTLY_CLOSED_PATH = ROOT / "logs" / "recently_closed.json"
PREDICTION_TRACKER_PATH = ROOT / "logs" / "prediction_tracker.json"

SCALP_KEYS = [
    "rsi", "adx", "plus_di", "minus_di", "stoch_rsi", "macd_hist",
    "ema_slope_5", "ema_slope_10", "cci", "bb_upper", "bb_lower", "bb_mid",
    "close", "atr_pips", "regime", "vwap_gap", "bbw",
    # v4: expose divergence, Ichimoku, structure to Claude
    "div_rsi_kind", "div_rsi_score", "div_rsi_age",
    "div_macd_kind", "div_macd_score", "div_macd_age", "div_score",
    "ichimoku_span_a_gap", "ichimoku_span_b_gap", "ichimoku_cloud_pos",
    "swing_dist_high", "swing_dist_low",
    "donchian_width", "kc_width", "chaikin_vol",
    "bb_span_pips", "vol_5m",
    "macd", "macd_signal",
]

# ─────────────────────────────────────────────
# Pair Profiles — pair-specific characteristics
# ─────────────────────────────────────────────

PAIR_PROFILES = {
    "USD_JPY": {
        "nickname": "UJ",
        "spread_gate": 1.5,    # tight spread pair
        "scalp_sl_range": (4, 7),
        "scalp_tp_range": (3, 5),
        "swing_sl_range": (8, 20),
        "swing_tp_range": (10, 30),
        "character": "Most liquid JPY pair. Driven by BOJ/Fed policy divergence. Tight spreads allow aggressive scalps. Sensitive to US yields and risk sentiment. Intervention risk above 160/below 140.",
        "best_sessions": ["TOKYO", "LONDON_NY_OVERLAP"],
        "session_notes": {
            "TOKYO": "Steady flow from Japanese institutions, good for range scalps",
            "LONDON_NY_OVERLAP": "Highest volatility, breakout opportunities",
            "LATE_NY": "Thin liquidity, avoid new entries",
        },
        "stoch_rsi_entry": (0.2, 0.8),  # standard
        "adx_trend_min": 18,  # slightly lower — UJ trends smoothly
        "atr_normal_m5": 4.0,
        "cooldown_after_sl_min": 10,  # shorter — UJ recovers fast
    },
    "EUR_USD": {
        "nickname": "EU",
        "spread_gate": 1.5,
        "scalp_sl_range": (4, 7),
        "scalp_tp_range": (3, 5),
        "swing_sl_range": (8, 20),
        "swing_tp_range": (10, 25),
        "character": "World's most traded pair. Macro-driven (FOMC vs ECB). Trends well on H1/H4 during London. Range-bound in Tokyo. Reacts to DXY inversely.",
        "best_sessions": ["LONDON", "LONDON_NY_OVERLAP"],
        "session_notes": {
            "TOKYO": "Low volume, range-bound — avoid or scalp tight range",
            "LONDON": "Best trending period, follow H1 direction",
            "LONDON_NY_OVERLAP": "News-driven spikes, wide SL needed",
        },
        "stoch_rsi_entry": (0.2, 0.8),
        "adx_trend_min": 20,
        "atr_normal_m5": 4.0,
        "cooldown_after_sl_min": 15,
    },
    "GBP_USD": {
        "nickname": "GU",
        "spread_gate": 2.0,  # slightly wider spread
        "scalp_sl_range": (5, 9),
        "scalp_tp_range": (4, 7),
        "swing_sl_range": (10, 25),
        "swing_tp_range": (15, 40),
        "character": "Cable — volatile, fast moves. Wider spreads but bigger TP possible. UK data/BOE sensitive. Fakeouts common at London open. Trends aggressively when it moves.",
        "best_sessions": ["LONDON", "LONDON_NY_OVERLAP"],
        "session_notes": {
            "TOKYO": "Avoid — thin liquidity, erratic moves",
            "LONDON": "Strongest moves. Watch for fakeout at 07:00-08:00 UTC then trend",
            "NEW_YORK": "Follow-through from London trends or mean reversion",
        },
        "stoch_rsi_entry": (0.15, 0.85),  # wider extremes needed — noisier
        "adx_trend_min": 22,  # higher bar — noisy trends
        "atr_normal_m5": 6.0,
        "cooldown_after_sl_min": 15,
    },
    "AUD_USD": {
        "nickname": "AU",
        "spread_gate": 2.0,
        "scalp_sl_range": (4, 7),
        "scalp_tp_range": (3, 5),
        "swing_sl_range": (8, 20),
        "swing_tp_range": (10, 25),
        "character": "Risk-on/risk-off barometer. Commodity-linked (iron ore, China). RBA sensitive. Trends well on H4 during risk-on. Mean-reverts during risk-off. Correlated with S&P and copper.",
        "best_sessions": ["TOKYO", "LONDON"],
        "session_notes": {
            "TOKYO": "Active — Australian session overlap. Good for RBA-related moves",
            "LONDON": "Follows risk sentiment, good trend continuation",
            "NEW_YORK": "Follows US equities direction",
        },
        "stoch_rsi_entry": (0.2, 0.8),
        "adx_trend_min": 20,
        "atr_normal_m5": 3.5,
        "cooldown_after_sl_min": 15,
    },
    "EUR_JPY": {
        "nickname": "EJ",
        "spread_gate": 2.5,  # cross pair — wider spread
        "scalp_sl_range": (5, 9),
        "scalp_tp_range": (4, 7),
        "swing_sl_range": (10, 25),
        "swing_tp_range": (15, 35),
        "character": "Cross pair — driven by both EUR and JPY flows. Risk-on = up, risk-off = down. Higher ATR than majors. Avoid during conflicting EUR/JPY events. Good for trend continuation when EUR and JPY are aligned.",
        "best_sessions": ["LONDON", "LONDON_NY_OVERLAP"],
        "session_notes": {
            "TOKYO": "JPY-driven flow. Watch for BOJ/Japan data",
            "LONDON": "EUR+JPY flow combined — can be explosive",
        },
        "stoch_rsi_entry": (0.18, 0.82),
        "adx_trend_min": 20,
        "atr_normal_m5": 6.0,
        "cooldown_after_sl_min": 15,
    },
    "GBP_JPY": {
        "nickname": "GJ",
        "spread_gate": 3.5,  # widest spread — expensive to scalp
        "scalp_sl_range": (7, 12),
        "scalp_tp_range": (5, 10),
        "swing_sl_range": (12, 30),
        "swing_tp_range": (20, 50),
        "character": "Beast pair — extreme volatility, wide spreads (3-5pip). NOT ideal for scalps. Best as swing trade when GBP and JPY are aligned. Whipsaws violently around news. Carry trade flow dominant.",
        "best_sessions": ["LONDON"],
        "session_notes": {
            "TOKYO": "DANGEROUS — thin liquidity + JPY flow = random spikes",
            "LONDON": "Best window — GBP flow creates clean trends",
            "NEW_YORK": "Follow-through or violent reversals",
        },
        "stoch_rsi_entry": (0.15, 0.85),
        "adx_trend_min": 25,  # needs strong trend to overcome spread
        "atr_normal_m5": 9.0,
        "cooldown_after_sl_min": 20,  # longer — high cost per SL
    },
    "AUD_JPY": {
        "nickname": "AJ",
        "spread_gate": 2.5,
        "scalp_sl_range": (5, 9),
        "scalp_tp_range": (4, 7),
        "swing_sl_range": (10, 25),
        "swing_tp_range": (15, 35),
        "character": "Pure risk barometer. Risk-on = up (AUD strong, JPY weak), risk-off = sharp dump. Correlated with Nikkei/ASX. Commodity-influenced. Trends cleanly during Tokyo session if RBA/China news.",
        "best_sessions": ["TOKYO", "LONDON"],
        "session_notes": {
            "TOKYO": "Active — both AUD and JPY home session",
            "LONDON": "Follows global risk tone",
            "LATE_NY": "Thin — avoid",
        },
        "stoch_rsi_entry": (0.18, 0.82),
        "adx_trend_min": 20,
        "atr_normal_m5": 5.5,
        "cooldown_after_sl_min": 15,
    },
}

# Default management rules (used when trade not in registry)
# v4.4: SL-free discretionary approach — CUT is disaster-only (-20pip scalp, -25pip swing)
# Claude does discretionary 棚卸し (position housekeeping) every 2-3min cycle.
# Monitor only cuts at catastrophic levels to prevent blowup.
DEFAULT_SCALP_RULES = {"trail_at_pip": 5, "partial_at_pip": 8, "max_hold_min": 30, "cut_at_pip": -20, "cut_age_min": 10, "be_at_pip": 2}
DEFAULT_SWING_RULES = {"trail_at_pip": 8, "partial_at_pip": 15, "max_hold_min": 480, "cut_at_pip": -25, "cut_age_min": 60, "be_at_pip": 5}

# ATR adaptive threshold (SL widening on volatility spike)
ATR_CHANGE_THRESHOLD = 0.30  # 30% ATR change triggers SL adjustment

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
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Read error response body for detailed OANDA error info
        error_body = ""
        try:
            error_body = e.read().decode()
        except Exception:
            pass
        raise Exception(f"HTTP Error {e.code}: {error_body or e.reason}") from e


def _api_post(token: str, path: str, body: dict, timeout: int = 8):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OANDA_BASE}{path}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode()
        except Exception:
            pass
        raise Exception(f"HTTP Error {e.code}: {error_body or e.reason}") from e


def _api_delete(token: str, path: str, timeout: int = 8):
    req = urllib.request.Request(
        f"{OANDA_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="DELETE",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _log_action(msg: str):
    """Append to live trade log. Falls back to critical_events.log if disk is full."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] MONITOR: {msg}\n"
    try:
        with open(TRADE_LOG_PATH, "a") as f:
            f.write(line)
    except OSError:
        # Disk full or write error — write to small fallback log so close events are never lost
        try:
            with open(CRITICAL_LOG_PATH, "a") as f:
                f.write(line)
        except Exception:
            pass  # last resort: silently discard rather than crash the monitor


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
        sl_order_id = sl_order.get("id")  # needed for atomic cancel+TSL
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
            "sl": sl, "sl_order_id": sl_order_id, "tp": tp, "sl_pips": sl_pips, "trail": trail_dist,
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
    margin_free_target = 0.20 if trade_type == "scalp" else 0.35
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


def detect_event_risk(pair: str, macro_bias: dict) -> str:
    """Detect event risk level for a pair from macro_bias.event_risk.

    Returns: "extreme", "high", or "normal"
    Checks each event entry independently — only flags if the entry
    mentions BOTH this pair's currency AND a severity keyword.
    """
    if not macro_bias:
        return "normal"
    event_risk = macro_bias.get("event_risk", {})
    if not event_risk:
        return "normal"

    base, quote = pair.split("_")
    ccy_map = {
        "JPY": {"BOJ", "JPY", "YEN"}, "USD": {"FOMC", "FED", "NFP", "USD"},
        "GBP": {"BOE", "GBP"}, "EUR": {"ECB", "EUR"}, "AUD": {"RBA", "AUD"},
    }
    pair_keywords = set()
    for ccy in (base, quote):
        pair_keywords.update(ccy_map.get(ccy, {ccy}))

    max_level = "normal"
    for key, text in event_risk.items():
        # Check relevance: key name OR text must mention this pair's currency
        # Use key name first (more precise: "BOJ_decision" → JPY only)
        key_upper = str(key).upper()
        text_upper = str(text).upper()

        # Check if event KEY maps to a currency in this pair
        key_relevant = any(kw in key_upper for kw in pair_keywords)

        # If key doesn't match, check if text explicitly names this pair (e.g. "EUR/USD")
        pair_slash = f"{base}/{quote}"
        pair_under = pair
        text_relevant = pair_slash in text_upper or pair_under in text_upper

        if not key_relevant and not text_relevant:
            continue

        if "EXTREME" in text_upper or ("TODAY" in text_upper and "RISK" in text_upper):
            return "extreme"
        if "HIGH" in text_upper or "SURPRISE" in text_upper:
            max_level = "high"

    return "normal"


def compute_scalp_params(pair: str, m5_atr: float, event_risk_level: str, live_spread: float = None) -> dict:
    """Compute ATR-adaptive TP/SL/trail parameters for scalping.

    Instead of fixed 3-5pip TP / 5-8pip SL, use ATR-relative values.
    SL must be > 1.0x ATR to survive normal M5 noise.
    TP should be 0.5-0.8x ATR for realistic scalp targets.
    live_spread: real-time spread in pips (falls back to spread_gate if None).
    """
    if not m5_atr or m5_atr <= 0:
        m5_atr = 5.0  # fallback

    pip = 0.01 if m5_atr > 1 else 0.0001  # rough: JPY pairs have ATR > 1

    # Base ATR-relative ratios
    tp_ratio = 0.6   # TP = 0.6x ATR
    sl_ratio = 1.2   # SL = 1.2x ATR (survives 1 candle noise)
    trail_ratio = 0.5  # trail at 0.5x ATR profit

    # Event risk adjustments
    if event_risk_level == "extreme":
        sl_ratio = 1.5    # wider SL for spikes
        tp_ratio = 0.8    # wider TP to capture event moves
        trail_ratio = 0.6
    elif event_risk_level == "high":
        sl_ratio = 1.3
        tp_ratio = 0.7
        trail_ratio = 0.55

    tp_pips = round(m5_atr * tp_ratio, 1)
    sl_pips = round(m5_atr * sl_ratio, 1)
    trail_pips = round(m5_atr * trail_ratio, 1)

    # Clamp to pair-specific ranges
    profile = PAIR_PROFILES.get(pair, {})
    sl_range = profile.get("scalp_sl_range", (3.0, 15.0))
    tp_range = profile.get("scalp_tp_range", (2.0, 10.0))
    tp_pips = max(tp_range[0], min(tp_pips, tp_range[1]))
    sl_pips = max(sl_range[0], min(sl_pips, sl_range[1]))
    trail_pips = max(2.0, min(trail_pips, 8.0))

    # Is this pair scalpable? SL > max of pair's range = too wide
    scalpable = sl_pips <= sl_range[1]

    # Spread-aware R:R calculation — use live spread, fall back to gate
    spread_gate = profile.get("spread_gate", 2.0)
    spread = live_spread if live_spread is not None else spread_gate
    # For LONG: BID travels TP+spread up, SL-spread down
    # For SHORT: ASK travels TP+spread down, SL-spread up
    bid_tp_dist = tp_pips + spread   # distance BID/ASK must travel for TP
    bid_sl_dist = sl_pips - spread   # distance BID/ASK must travel for SL
    if bid_sl_dist <= 0:
        bid_sl_dist = 0.1  # avoid division by zero
    true_rr = round(bid_tp_dist / bid_sl_dist, 2)  # >1 means TP harder to reach
    spread_pct_of_tp = round(spread / tp_pips * 100, 1)

    # Recommend minimum TP that gives fair R:R (TP >= SL + 2*spread)
    fair_tp_pips = round(sl_pips + 2 * spread, 1)

    return {
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "trail_pips": trail_pips,
        "m5_atr": round(m5_atr, 1),
        "sl_atr_ratio": round(sl_pips / m5_atr, 2),
        "event_risk": event_risk_level,
        "scalpable": scalpable,
        "spread_info": {
            "live_spread": round(spread, 1),
            "spread_gate": spread_gate,
            "bid_tp_distance": bid_tp_dist,
            "bid_sl_distance": round(bid_sl_dist, 1),
            "true_distance_rr": true_rr,
            "spread_pct_of_tp": spread_pct_of_tp,
            "fair_tp_minimum": fair_tp_pips,
        },
        "note": f"ATR={round(m5_atr,1)} TP={tp_pips}({tp_ratio}x) SL={sl_pips}({sl_ratio}x) spread={spread}pip({spread_pct_of_tp}%) true_RR={true_rr} event={event_risk_level}"
    }


def score_pair(pair_data: dict, direction: str, pair: str = "", macro_bias: dict = None,
               session: dict = None) -> tuple:
    """Score a pair for entry. Returns (score, reasons, confluence_detail).

    v4: Uses ALL available indicators + pair profiles + session awareness.
    Range: -4 to +10. Guidelines (not hard thresholds):
      5+ = strong setup, 4 = solid, 3 = marginal (context-dependent), ≤2 = weak.

    Categories:
    A. Direction (trend alignment)    -> up to +3
    B. Timing (entry precision)       -> up to +2
    C. Confluence (extra confirmation)-> up to +3  [NEW: divergence, Ichimoku, VWAP]
    D. Macro alignment                -> +1 or -2
    E. Penalties/bonuses              -> -3 to +1  [NEW: session, volatility, pair-specific]
    """
    m5 = pair_data.get("m5", {})
    m1 = pair_data.get("m1", {})
    bias = pair_data.get("bias", {})
    price = pair_data.get("price", {})
    profile = PAIR_PROFILES.get(pair, {})
    score = 0
    reasons = []
    confluence = {}  # detailed breakdown for Claude's discretionary use

    # --- GATE: Pair-specific spread gate ---
    spread = price.get("spread_pips", 99)
    spread_gate = profile.get("spread_gate", 2.0)
    if spread >= spread_gate:
        return 0, [f"GATE_FAIL:spread={spread}>{spread_gate}"], {}

    # === A. DIRECTION (up to +3) ===

    # A1. H1 bias alignment (+1)
    h1_plus = bias.get("h1_plus_di", 0) or 0
    h1_minus = bias.get("h1_minus_di", 0) or 0
    h1_adx = bias.get("h1_adx", 0) or 0
    if direction == "LONG" and h1_plus > h1_minus:
        score += 1
        reasons.append(f"H1_bull(DI+={round(h1_plus)}>DI-={round(h1_minus)},ADX={round(h1_adx)})")
    elif direction == "SHORT" and h1_minus > h1_plus:
        score += 1
        reasons.append(f"H1_bear(DI-={round(h1_minus)}>DI+={round(h1_plus)},ADX={round(h1_adx)})")

    # A2. M5 trend: pair-specific ADX threshold AND DI aligned (+1)
    m5_adx = m5.get("adx", 0) or 0
    m5_plus = m5.get("plus_di", 0) or 0
    m5_minus = m5.get("minus_di", 0) or 0
    adx_min = profile.get("adx_trend_min", 20)
    if m5_adx > adx_min:
        if (direction == "LONG" and m5_plus > m5_minus) or \
           (direction == "SHORT" and m5_minus > m5_plus):
            score += 1
            reasons.append(f"M5_trend(ADX={round(m5_adx)}>{adx_min},DI_ok)")

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

    # B1. M1 Stoch RSI extreme — pair-specific thresholds (+1)
    m1_stoch = m1.get("stoch_rsi")
    stoch_lo, stoch_hi = profile.get("stoch_rsi_entry", (0.2, 0.8))
    if m1_stoch is not None:
        if direction == "LONG" and m1_stoch < stoch_lo:
            score += 1
            reasons.append(f"TIMING:M1_stoch_oversold({round(m1_stoch, 2)}<{stoch_lo})")
        elif direction == "SHORT" and m1_stoch > stoch_hi:
            score += 1
            reasons.append(f"TIMING:M1_stoch_overbought({round(m1_stoch, 2)}>{stoch_hi})")

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

    # === C. CONFLUENCE — NEW in v4 (up to +3) ===

    # C1. M5 Divergence alignment (+1)
    m5_div_score = m5.get("div_score", 0) or 0
    m5_div_rsi_kind = m5.get("div_rsi_kind", 0) or 0
    m5_div_macd_kind = m5.get("div_macd_kind", 0) or 0
    # kind: +1=bullish, -1=bearish, +2=hidden bullish, -2=hidden bearish
    if m5_div_score != 0:
        div_bullish = m5_div_rsi_kind > 0 or m5_div_macd_kind > 0
        div_bearish = m5_div_rsi_kind < 0 or m5_div_macd_kind < 0
        if (direction == "LONG" and div_bullish) or (direction == "SHORT" and div_bearish):
            score += 1
            div_type = "bull" if div_bullish else "bear"
            reasons.append(f"CONFLUENCE:M5_div_{div_type}(score={round(m5_div_score, 1)},RSI_k={m5_div_rsi_kind},MACD_k={m5_div_macd_kind})")
            confluence["divergence"] = {"type": div_type, "score": m5_div_score, "rsi_kind": m5_div_rsi_kind, "macd_kind": m5_div_macd_kind}
        elif (direction == "LONG" and div_bearish) or (direction == "SHORT" and div_bullish):
            # Divergence AGAINST trade direction — warning, not penalty
            reasons.append(f"WARN:M5_div_against(score={round(m5_div_score, 1)})")
            confluence["divergence_against"] = True

    # C2. M5 Ichimoku cloud position (+1)
    m5_cloud_pos = m5.get("ichimoku_cloud_pos", 0) or 0
    if m5_cloud_pos != 0:
        if (direction == "LONG" and m5_cloud_pos > 0) or \
           (direction == "SHORT" and m5_cloud_pos < 0):
            score += 1
            reasons.append(f"CONFLUENCE:M5_ichimoku_{'above' if m5_cloud_pos > 0 else 'below'}(gap={round(m5_cloud_pos, 1)}pip)")
            confluence["ichimoku"] = {"position": "above" if m5_cloud_pos > 0 else "below", "gap_pips": round(m5_cloud_pos, 1)}

    # C3. VWAP alignment (+1)
    m5_vwap = m5.get("vwap_gap", 0) or 0
    if abs(m5_vwap) > 1.0:
        if (direction == "LONG" and m5_vwap > 0) or \
           (direction == "SHORT" and m5_vwap < 0):
            score += 1
            reasons.append(f"CONFLUENCE:M5_vwap_{'above' if m5_vwap > 0 else 'below'}({round(m5_vwap, 1)}pip)")
            confluence["vwap"] = {"gap_pips": round(m5_vwap, 1)}

    # === D. MACRO ALIGNMENT (+1 or -2) ===
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
            reasons.append(f"MACRO_CONFLICT({macro_label})")

    # === E. PENALTIES & BONUSES ===

    # E1. M5 choppy — pair-specific ADX threshold (-1, or -2 if very weak)
    choppy_threshold = max(adx_min - 5, 12)  # pair-specific
    if m5_adx and m5_adx < choppy_threshold:
        score -= 1
        reasons.append(f"PENALTY:choppy(M5_ADX={round(m5_adx)}<{choppy_threshold})")
    # E1b. M5 ADX < 15 = no directional conviction — extra penalty (data: 100% loss rate)
    if m5_adx and m5_adx < 15:
        score -= 1
        reasons.append(f"PENALTY:ADX_DEAD(M5_ADX={round(m5_adx)}<15,trend_trades_lose)")
        confluence["adx_quality"] = "dead"
    elif m5_adx and m5_adx >= 25:
        confluence["adx_quality"] = "strong"
    else:
        confluence["adx_quality"] = "moderate"

    # E2. Event risk (-1 for extreme)
    if macro_bias:
        ev_risk = detect_event_risk(pair, macro_bias)
        if ev_risk == "extreme":
            score -= 1
            reasons.append(f"PENALTY:EVENT_EXTREME({pair})")
        elif ev_risk == "high":
            reasons.append(f"WARN:EVENT_HIGH({pair})")

    # E3. Session suitability (+1 or -1) — NEW
    if session:
        current_session = session.get("session", "")
        best = profile.get("best_sessions", [])
        if current_session in best:
            score += 1
            reasons.append(f"SESSION_BONUS({current_session})")
        elif current_session in ("LATE_NY",):
            score -= 1
            reasons.append(f"PENALTY:SESSION_THIN({current_session})")
        # Add session note for Claude
        session_note = profile.get("session_notes", {}).get(current_session, "")
        if session_note:
            confluence["session_note"] = session_note

    # E4. Volatility context — is current ATR normal for this pair? (-1 if extreme)
    m5_atr = m5.get("atr_pips", 0) or 0
    atr_normal = profile.get("atr_normal_m5", 5.0)
    if m5_atr > 0:
        vol_ratio = m5_atr / atr_normal
        confluence["vol_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 2.5:
            score -= 1
            reasons.append(f"PENALTY:VOL_EXTREME(ATR={round(m5_atr, 1)},normal={atr_normal},ratio={round(vol_ratio, 1)}x)")
        elif vol_ratio < 0.4:
            score -= 1
            reasons.append(f"PENALTY:VOL_DEAD(ATR={round(m5_atr, 1)},normal={atr_normal},ratio={round(vol_ratio, 1)}x)")

    # E5. MTF alignment (+1 if aligned, -1 if conflicted, -2 if H1 turning) — NEW
    mtf = compute_mtf_alignment(pair, m5, bias)
    mtf_info = mtf.get(direction.lower(), {})
    mtf_status = mtf_info.get("status", "unknown")
    mtf_detail = mtf_info.get("detail", "")
    confluence["mtf"] = mtf_info

    if mtf_status == "aligned":
        score += 1
        reasons.append(f"MTF_ALIGNED({mtf_detail[:60]})")
    elif mtf_status == "h4_counter":
        # H4 against but H1+M5 ok — not a penalty for scalps, but note it
        reasons.append(f"MTF:H4_COUNTER({mtf_detail[:60]})")
    elif mtf_status in ("h1_conflict", "conflict"):
        score -= 1
        reasons.append(f"PENALTY:MTF_CONFLICT({mtf_detail[:60]})")
    elif mtf_status == "h1_turning":
        score -= 2
        reasons.append(f"PENALTY:H1_TURNING({mtf_detail[:60]})")
    elif mtf_status == "m5_counter":
        score -= 1
        reasons.append(f"PENALTY:M5_COUNTER({mtf_detail[:60]})")

    # Add pair character for Claude
    confluence["pair_character"] = profile.get("character", "")
    confluence["cooldown_min"] = profile.get("cooldown_after_sl_min", 15)

    return score, reasons, confluence


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
        # Support both dict format {"464498": {...}} and list format [{...}, ...]
        if isinstance(data, dict):
            # Dict format: keys are trade IDs, values are trade objects
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        elif isinstance(data, list):
            return {str(t["trade_id"]): t for t in data if isinstance(t, dict) and "trade_id" in t}
        return {}
    except Exception:
        return {}


def save_registry(registry: dict):
    """Save trade registry. Logs a critical warning if disk is full."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(list(registry.values()), f, indent=2)
    except OSError as e:
        # Disk full: registry write failed — log to critical events so trade IDs are preserved
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = list(registry.keys())
        try:
            with open(CRITICAL_LOG_PATH, "a") as f:
                f.write(f"[{ts}] MONITOR: REGISTRY_WRITE_FAILED disk_full trade_ids={ids} err={e}\n")
        except Exception:
            pass


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


def fetch_current_atr(token: str, pairs: set) -> dict:
    """Fetch current M5 ATR for a set of pairs. Lightweight — only called for open position pairs."""
    result = {}
    for pair in pairs:
        try:
            m5 = fetch_candles_and_compute(token, pair, "M5", 30)
            if isinstance(m5, dict) and "atr_pips" in m5:
                result[pair] = m5["atr_pips"]
        except Exception:
            pass
    return result


def update_sl(token: str, acc: str, tid: str, new_sl: float, pos: dict) -> str:
    """Move SL to a new price via OANDA API. Returns action string or None on failure."""
    body = {"stopLoss": {"price": str(round(new_sl, 5))}}
    # Preserve existing TP if present
    if pos.get("tp") is not None:
        body["takeProfit"] = {"price": str(round(pos["tp"], 5))}
    try:
        _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/orders", body)
        return None  # success
    except Exception as e:
        return str(e)


def manage_positions(token: str, acc: str, positions: list, pricing: dict,
                     micro_data: dict = None) -> list:
    """Apply mechanical management rules. Returns list of actions taken.

    Features:
    - recently_closed tracking to prevent duplicate close attempts
    - Catches OANDA 404 (trade already closed) gracefully
    - Logs all actions with rules_source for audit
    - Dynamic SL: breakeven move + ATR-adaptive widening/tightening
    - Momentum close: close when profit + momentum reversal detected

    Rule execution order:
      0a. BE_MOVE (breakeven SL)
      0b. ATR_ADJUST (volatility-adaptive SL)
      1.  TRAIL_SET (trailing stop)
      2.  PARTIAL (partial close)
      3.  CUT (cut loss)
      4.  TIMEOUT (max hold)
      5.  MOMENTUM_CLOSE (profit + momentum death)
      (Custom rules can be added after Rule 5 by the trader agent)

    micro_data: dict of {pair: {"direction": str, "velocity": float}} from compute_micro_momentum.
    Trader can modify rules in registry to control behavior. See TRADER_PROMPT.md for registry format.
    """
    registry = load_registry()
    recently_closed = load_recently_closed()
    actions = []
    if micro_data is None:
        micro_data = {}

    # Fetch current ATR for pairs with open positions (for dynamic SL adjustment)
    open_pairs = {pos["pair"] for pos in positions if pos["id"] not in recently_closed}
    current_atr = fetch_current_atr(token, open_pairs) if open_pairs else {}

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
        # be_at_pip: if registry has rules (even without be_at_pip), infer from trail_at_pip
        if "be_at_pip" in rules:
            be_at = rules["be_at_pip"]
        elif rules_source == "registry":
            # Registry rules exist but no be_at_pip — derive from trade style
            # trail_at_pip > 6 suggests swing-like → wider BE threshold
            be_at = 5 if rules.get("trail_at_pip", 5) > 6 else 2
        else:
            be_at = 2 if inferred_type == "scalp" else 5

        action = None
        entry_data = registry.get(tid, {})
        entry_price = pos.get("entry", entry_data.get("entry_price", 0))
        is_long = units > 0
        current_sl = pos.get("sl")

        # ─── Dynamic SL Rules (run before trail/partial/cut) ───

        # Rule 0a: Breakeven SL move — protect capital once in profit
        # Only if: in profit >= be_at_pip, SL not already at/past breakeven, no trailing stop yet
        if (upl_pips >= be_at and not has_trail and current_sl is not None
                and not pos.get("is_be", False)):
            # Move SL to entry + 0.5pip buffer (cover spread)
            spread_buf = 0.5 * pip
            if is_long:
                new_sl = entry_price + spread_buf
                should_move = current_sl < new_sl  # only move SL up (tighter)
            else:
                new_sl = entry_price - spread_buf
                should_move = current_sl > new_sl  # only move SL down (tighter)

            if should_move:
                err = update_sl(token, acc, tid, new_sl, pos)
                if err:
                    if "404" in err or "TRADE_DOESNT_EXIST" in err:
                        mark_closed(tid, pair, "gone_during_be")
                        action = f"BE_SKIP {pair}: trade already closed"
                    else:
                        action = f"BE_FAIL {pair}: {err}"
                else:
                    action = f"BE_MOVE {pair} SL→{round(new_sl, 5)} (was {current_sl}, UPL={upl_pips}pip) [{rules_source}]"
                    pos["sl"] = new_sl  # update local state for subsequent rules
                    pos["is_be"] = True

        # Rule 0b: ATR-adaptive SL — widen SL if volatility spiked since entry
        # Only if: registry has entry_atr, atr_adjust not disabled, no trailing stop, SL exists
        if (action is None and not has_trail and current_sl is not None
                and entry_data.get("entry_atr") and entry_data.get("atr_adjust", True)):
            entry_atr = entry_data["entry_atr"]
            now_atr = current_atr.get(pair)
            if now_atr and entry_atr > 0:
                atr_ratio = now_atr / entry_atr
                if atr_ratio > (1 + ATR_CHANGE_THRESHOLD):
                    # Volatility spiked — widen SL proportionally to avoid premature stop
                    if is_long:
                        sl_dist = entry_price - current_sl  # positive
                        new_sl_dist = sl_dist * atr_ratio
                        new_sl = entry_price - new_sl_dist
                        should_move = new_sl < current_sl  # only widen (move SL further away)
                    else:
                        sl_dist = current_sl - entry_price  # positive
                        new_sl_dist = sl_dist * atr_ratio
                        new_sl = entry_price + new_sl_dist
                        should_move = new_sl > current_sl  # only widen

                    if should_move:
                        err = update_sl(token, acc, tid, new_sl, pos)
                        if err:
                            action = f"ATR_ADJUST_FAIL {pair}: {err}"
                        else:
                            action = f"ATR_WIDEN {pair} SL→{round(new_sl, 5)} (ATR {entry_atr}→{now_atr}, ratio={round(atr_ratio, 2)}) [{rules_source}]"
                            pos["sl"] = new_sl

                elif atr_ratio < (1 - ATR_CHANGE_THRESHOLD) and upl_pips > 0:
                    # Volatility dropped AND in profit — tighten SL to lock more profit
                    if is_long:
                        sl_dist = entry_price - current_sl
                        new_sl_dist = sl_dist * atr_ratio
                        new_sl = entry_price - new_sl_dist
                        should_move = new_sl > current_sl  # only tighten (move SL closer to price)
                    else:
                        sl_dist = current_sl - entry_price
                        new_sl_dist = sl_dist * atr_ratio
                        new_sl = entry_price + new_sl_dist
                        should_move = new_sl < current_sl

                    if should_move:
                        err = update_sl(token, acc, tid, new_sl, pos)
                        if err:
                            action = f"ATR_ADJUST_FAIL {pair}: {err}"
                        else:
                            action = f"ATR_TIGHTEN {pair} SL→{round(new_sl, 5)} (ATR {entry_atr}→{now_atr}, ratio={round(atr_ratio, 2)}) [{rules_source}]"
                            pos["sl"] = new_sl

        # ─── Existing Rules (trail/partial/cut/timeout) ───
        if action is not None:
            actions.append(action)
            _log_action(action)
            continue  # dynamic SL action taken, skip trail/partial/cut this cycle

        # Rule 1: Set trailing stop if profit exceeds threshold and no trail yet
        if upl_pips >= trail_at and not has_trail:
            trail_dist_pips = trail_at
            # OANDA rejects TSL if distance < current spread; ensure distance >= spread + buffer
            pair_spread_pips = 0
            if pair in pricing:
                pair_spread_pips = pricing[pair].get("spread_pips", 0)
            if trail_dist_pips < pair_spread_pips + 0.5:
                trail_dist_pips = round(pair_spread_pips + 0.5, 1)
            trail_distance = str(round(trail_dist_pips * pip, 5))
            try:
                # OANDA v20: PUT /trades/{id}/orders to set TSL
                # If a fixed SL exists, cancel it atomically using its order ID
                # OANDA requires "cancelledOrderID" to cancel an existing dependent order
                pos_sl = pos.get("sl")
                pos_sl_order_id = pos.get("sl_order_id")
                if pos_sl is not None and pos_sl_order_id:
                    # Atomic: cancel SL by ID + set TSL in one request
                    order_body = {
                        "stopLoss": {"cancelledOrderID": pos_sl_order_id},
                        "trailingStopLoss": {"distance": trail_distance}
                    }
                else:
                    order_body = {
                        "trailingStopLoss": {"distance": trail_distance}
                    }
                try:
                    _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/orders", order_body)
                except Exception as e1:
                    # Fallback: if atomic call fails, try 2-step (cancel SL first, then set TSL)
                    if pos_sl is not None and "400" in str(e1):
                        pos_sl_order_id = pos.get("sl_order_id")
                        if pos_sl_order_id:
                            # Step 1: cancel SL via its order ID (reliable OANDA approach)
                            try:
                                _api_delete(token, f"/v3/accounts/{acc}/orders/{pos_sl_order_id}")
                            except Exception:
                                pass  # may already be gone
                        # Step 2: set TSL now that SL is removed
                        _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/orders", {
                            "trailingStopLoss": {"distance": trail_distance}
                        })
                    else:
                        raise e1
                action = f"TRAIL_SET {pair} {units}u trail={trail_dist_pips}pip (UPL={upl_pips}pip, spread={pair_spread_pips}pip) [{rules_source}]"
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

                # Auto-reverse: if enabled, open opposite position
                if rules.get("auto_reverse"):
                    try:
                        rev_units = str(-int(units))  # flip direction
                        rev_pip = _pip_size(pair)
                        price_data = pricing.get(pair, {})
                        rev_mid = price_data.get("mid", 0)
                        # Reverse TP = original SL distance from new entry
                        rev_tp_dist = abs(cut_at) * rev_pip
                        # Reverse SL = same distance as original TP target
                        rev_sl_dist = trail_at * rev_pip
                        if int(units) > 0:  # was LONG, now SHORT
                            rev_tp = round(rev_mid - rev_tp_dist, 5)
                            rev_sl = round(rev_mid + rev_sl_dist, 5)
                        else:  # was SHORT, now LONG
                            rev_tp = round(rev_mid + rev_tp_dist, 5)
                            rev_sl = round(rev_mid - rev_sl_dist, 5)

                        rev_body = {
                            "order": {
                                "type": "MARKET", "instrument": pair,
                                "units": rev_units, "timeInForce": "FOK",
                                "stopLossOnFill": {"price": str(rev_sl)},
                                "takeProfitOnFill": {"price": str(rev_tp)},
                                "clientExtensions": {"tag": "scalp", "comment": "auto-reverse"}
                            }
                        }
                        rev_resp = _api_post(token, f"/v3/accounts/{acc}/orders", rev_body)
                        rev_fill = rev_resp.get("orderFillTransaction", {})
                        rev_tid = str(rev_fill.get("tradeOpened", {}).get("tradeID", ""))
                        if rev_tid:
                            # Register reverse trade
                            registry[rev_tid] = {
                                "trade_id": rev_tid, "owner": "auto-reverse",
                                "type": "scalp", "pair": pair, "units": int(rev_units),
                                "rules": {"trail_at_pip": abs(cut_at), "partial_at_pip": abs(cut_at) * 1.5,
                                          "max_hold_min": 15, "cut_at_pip": -trail_at, "cut_age_min": 5,
                                          "auto_reverse": False}
                            }
                            save_registry(registry)
                        action += f" → REVERSED {rev_units}u TP={rev_tp} SL={rev_sl}"
                        _log_action(f"AUTO_REVERSE {pair} {rev_units}u @{rev_mid} TP={rev_tp} SL={rev_sl}")
                    except Exception as re:
                        action += f" → REVERSE_FAIL: {re}"
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

        # Rule 5: Momentum close — take profit when momentum reverses
        # Trader enables this per-trade via registry rules:
        #   "momentum_close": {"enabled": true, "min_profit_pip": 3, "vel_threshold": 0.2}
        # Default: disabled. Trader decides which trades get this behavior.
        #
        # Logic: if in profit >= min_profit_pip AND micro_vel has reversed direction
        # (was going with the trade, now going against), close the position.
        # This catches "move is dying" before price fully reverses.
        #
        # Trader can customize by changing:
        #   - min_profit_pip: minimum profit before momentum check kicks in
        #   - vel_threshold: how much velocity reversal triggers close (0 = any reversal)
        elif action is None:
            mc = rules.get("momentum_close", {})
            if isinstance(mc, dict) and mc.get("enabled"):
                mc_min = mc.get("min_profit_pip", 3)
                mc_vel_thresh = mc.get("vel_threshold", 0.0)
                micro = micro_data.get(pair, {})
                micro_vel = micro.get("velocity", 0)

                if upl_pips >= mc_min and micro_vel != 0:
                    # Check if momentum is going AGAINST our trade direction
                    # LONG: we want vel > 0 (price going up). vel < -threshold = reversal
                    # SHORT: we want vel < 0 (price going down). vel > +threshold = reversal
                    vel_against = (is_long and micro_vel < -mc_vel_thresh) or \
                                  (not is_long and micro_vel > mc_vel_thresh)

                    if vel_against:
                        try:
                            _api_put(token, f"/v3/accounts/{acc}/trades/{tid}/close", {})
                            mark_closed(tid, pair, "momentum_close")
                            action = (f"MOMENTUM_CLOSE {pair} {units}u +{upl_pips}pip "
                                      f"(vel={micro_vel}, reversed) [{rules_source}]")
                        except Exception as e:
                            if "404" in str(e) or "TRADE_DOESNT_EXIST" in str(e):
                                mark_closed(tid, pair, "gone_during_momentum")
                                action = f"MOMENTUM_SKIP {pair}: trade already closed"
                            else:
                                action = f"MOMENTUM_FAIL {pair}: {e}"

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


def compute_currency_strength(pairs_data: dict) -> dict:
    """Compute currency strength from M5 momentum across all pairs.

    Uses M5 RSI, EMA slope, ADX-weighted DI to score each currency.
    Positive = strong, negative = weak. Range roughly -3 to +3.
    """
    strength = {"USD": 0.0, "EUR": 0.0, "GBP": 0.0, "AUD": 0.0, "JPY": 0.0}
    counts = {"USD": 0, "EUR": 0, "GBP": 0, "AUD": 0, "JPY": 0}

    for pair, pdata in pairs_data.items():
        m5 = pdata.get("m5", {})
        if not m5 or isinstance(m5, str):
            continue
        base, quote = pair.split("_")

        rsi = m5.get("rsi", 50) or 50
        rsi_signal = (rsi - 50) / 25  # normalized -2 to +2

        slope = m5.get("ema_slope_5", 0) or 0
        pip = 0.01 if "JPY" in pair else 0.0001
        slope_signal = max(-1.5, min(1.5, slope / pip))

        adx = m5.get("adx", 0) or 0
        plus_di = m5.get("plus_di", 0) or 0
        minus_di = m5.get("minus_di", 0) or 0
        di_signal = ((plus_di - minus_di) / (plus_di + minus_di)) if adx > 15 and (plus_di + minus_di) > 0 else 0

        combined = rsi_signal * 0.4 + slope_signal * 0.3 + di_signal * 0.3

        strength[base] += combined
        strength[quote] -= combined
        counts[base] += 1
        counts[quote] += 1

    for ccy in strength:
        if counts[ccy] > 0:
            strength[ccy] = round(strength[ccy] / counts[ccy], 2)

    return strength


def compute_market_regime(pairs_data: dict, session: dict, currency_strength: dict) -> dict:
    """Detect overall market regime from cross-pair analysis.

    Returns regime, risk_tone, tradeable flag, and human-readable note for Claude.
    """
    scores = []
    adx_values = []
    vol_ratios = []
    active_pairs = []

    for pair, pdata in pairs_data.items():
        sig = pdata.get("signal", {})
        m5 = pdata.get("m5", {})
        best_score = sig.get("best_score", 0)
        scores.append(best_score)
        if best_score >= 3:
            active_pairs.append(f"{PAIR_PROFILES.get(pair, {}).get('nickname', pair)}({best_score})")

        adx_values.append(m5.get("adx", 0) or 0)

        best_dir = sig.get("best", "LONG")
        conf = sig.get(f"{best_dir.lower()}_confluence", {})
        vol_ratios.append(conf.get("vol_ratio", 1.0))

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    avg_adx = round(sum(adx_values) / len(adx_values), 1) if adx_values else 0
    avg_vol = round(sum(vol_ratios) / len(vol_ratios), 2) if vol_ratios else 1.0
    high_score_count = sum(1 for s in scores if s >= 4)

    # Regime
    if avg_adx < 15 and avg_vol < 0.5:
        regime = "dead"
    elif avg_adx < 15:
        regime = "choppy"
    elif avg_adx > 25 and high_score_count >= 3:
        regime = "trending"
    elif avg_vol > 2.0:
        regime = "event_driven"
    else:
        regime = "range"

    # Risk tone from AUD vs JPY
    aud = currency_strength.get("AUD", 0)
    jpy = currency_strength.get("JPY", 0)
    if aud > 0.3 and jpy < -0.3:
        risk_tone = "risk_on"
    elif aud < -0.3 and jpy > 0.3:
        risk_tone = "risk_off"
    elif abs(aud) < 0.2 and abs(jpy) < 0.2:
        risk_tone = "neutral"
    else:
        risk_tone = "mixed"

    # Dominant driver
    sorted_str = sorted(currency_strength.items(), key=lambda x: abs(x[1]), reverse=True)
    dominant = sorted_str[0] if sorted_str else ("", 0)
    dominant_driver = f"{dominant[0]}({'strong' if dominant[1] > 0 else 'weak'},{dominant[1]})"

    # Tradeable?
    tradeable = True
    note_parts = []

    if regime == "dead":
        tradeable = False
        note_parts.append("Market dead — ATR collapsed. Sit out.")
    elif regime == "choppy":
        note_parts.append("Choppy — low ADX. Reduce size, tighten TP.")
    elif regime == "event_driven":
        note_parts.append("Event-driven vol — wider SL, reduce size.")
    elif regime == "trending":
        note_parts.append(f"Trending — {high_score_count} pairs score≥4.")

    if avg_score < 2.0:
        tradeable = False
        note_parts.append(f"No setups — avg score {avg_score}. Wait.")

    if risk_tone == "risk_off":
        note_parts.append("Risk-off: JPY strong, AUD weak → favor JPY longs.")
    elif risk_tone == "risk_on":
        note_parts.append("Risk-on: AUD strong, JPY weak → favor AUD/crosses up.")

    note_parts.append(f"Driver: {dominant_driver}")

    return {
        "regime": regime, "risk_tone": risk_tone,
        "dominant_driver": dominant_driver,
        "currency_strength": currency_strength,
        "tradeable": tradeable,
        "avg_score": avg_score, "avg_adx": avg_adx, "avg_vol_ratio": avg_vol,
        "high_score_pairs": high_score_count,
        "active_pairs": active_pairs,
        "note": " | ".join(note_parts),
    }


# ─────────────────────────────────────────────
# MTF Alignment Detection
# ─────────────────────────────────────────────

def compute_mtf_alignment(pair: str, m5: dict, bias: dict) -> dict:
    """Detect multi-timeframe alignment or conflict for a pair.

    Checks H4, H1, M5 direction consistency and flags:
    - "aligned": all TFs agree → high probability
    - "h1_m5_aligned": H1+M5 agree but H4 differs → pullback or reversal?
    - "m5_counter": M5 against H1/H4 → short-term counter-trend, risky scalp
    - "conflict": TFs disagree significantly → avoid or reduce size
    - "h1_turning": H1 ADX dropping + DI narrowing → regime change imminent

    Returns dict with alignment status and detail for both LONG and SHORT.
    """
    result = {"long": {}, "short": {}}

    # Extract directions
    h4_regime = bias.get("h4_regime", "unknown")
    h1_adx = bias.get("h1_adx", 0) or 0
    h1_plus = bias.get("h1_plus_di", 0) or 0
    h1_minus = bias.get("h1_minus_di", 0) or 0
    m5_adx = m5.get("adx", 0) or 0
    m5_plus = m5.get("plus_di", 0) or 0
    m5_minus = m5.get("minus_di", 0) or 0
    m5_rsi = m5.get("rsi", 50) or 50

    # Direction per TF: +1=bull, -1=bear, 0=neutral
    def tf_dir(plus_di, minus_di, adx, threshold=15):
        if adx < threshold:
            return 0
        return 1 if plus_di > minus_di else -1

    h4_dir = 1 if "bull" in h4_regime.lower() else (-1 if "bear" in h4_regime.lower() else 0)
    h1_dir = tf_dir(h1_plus, h1_minus, h1_adx)
    m5_dir = tf_dir(m5_plus, m5_minus, m5_adx)

    # H1 turning detection: ADX dropping + DI converging
    h1_turning = False
    if h1_adx > 0 and h1_adx < 20:
        di_gap = abs(h1_plus - h1_minus)
        if di_gap < 5:
            h1_turning = True

    for direction_name, dir_val in [("long", 1), ("short", -1)]:
        h4_ok = (h4_dir == dir_val) or (h4_dir == 0)
        h1_ok = (h1_dir == dir_val) or (h1_dir == 0)
        m5_ok = (m5_dir == dir_val) or (m5_dir == 0)

        h4_against = (h4_dir == -dir_val)
        h1_against = (h1_dir == -dir_val)
        m5_against = (m5_dir == -dir_val)

        if h4_ok and h1_ok and m5_ok and not (h4_dir == 0 and h1_dir == 0 and m5_dir == 0):
            status = "aligned"
            detail = f"H4={'bull' if dir_val==1 else 'bear'}/H1=ok/M5=ok — full alignment"
        elif h1_ok and m5_ok and h4_against:
            status = "h4_counter"
            detail = f"H4 AGAINST but H1+M5 aligned — pullback trade, shorter hold"
        elif h4_ok and h1_ok and m5_against:
            status = "m5_counter"
            detail = f"M5 against H1/H4 — wait for M5 to turn or skip"
        elif h4_ok and m5_ok and h1_against:
            status = "h1_conflict"
            detail = f"H1 against — dangerous, H1 is your anchor. Avoid."
        else:
            status = "conflict"
            detail = f"TFs disagree (H4={h4_dir},H1={h1_dir},M5={m5_dir}) — avoid"

        if h1_turning:
            status = "h1_turning"
            detail = f"H1 regime changing (ADX={round(h1_adx)},DI_gap={round(abs(h1_plus-h1_minus))}) — wait for clarity"

        result[direction_name] = {
            "status": status,
            "detail": detail,
            "h4_dir": h4_dir, "h1_dir": h1_dir, "m5_dir": m5_dir,
            "h1_turning": h1_turning,
        }

    return result


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

    # Pre-compute micro momentum for open position pairs (needed for momentum_close)
    micro_for_positions = {}
    if positions:
        for pos in positions:
            p = pos["pair"]
            if p not in micro_for_positions:
                pip = _pip_size(p)
                s5 = fetch_s5_candles(token, p, count=24)
                micro_for_positions[p] = compute_micro_momentum(s5, pip)

    # Mechanical position management (EXECUTES TRADES)
    actions = manage_positions(token, acc, positions, pricing, micro_data=micro_for_positions)

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

    # Risk & session (computed early for scoring)
    session = detect_session()

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

        # Signal scoring v4 (direction + timing + confluence + macro + session)
        long_score, long_reasons, long_confluence = score_pair(pair_data, "LONG", pair, macro_bias, session)
        short_score, short_reasons, short_confluence = score_pair(pair_data, "SHORT", pair, macro_bias, session)
        best_dir = "LONG" if long_score > short_score else "SHORT"
        best_score = max(long_score, short_score)
        pair_data["signal"] = {
            "long_score": long_score, "long_reasons": long_reasons, "long_confluence": long_confluence,
            "short_score": short_score, "short_reasons": short_reasons, "short_confluence": short_confluence,
            "best": best_dir, "best_score": best_score,
        }

        # Pair profile for Claude's discretionary use
        profile = PAIR_PROFILES.get(pair, {})
        pair_data["profile"] = {
            "nickname": profile.get("nickname", pair),
            "character": profile.get("character", ""),
            "scalp_sl_range": profile.get("scalp_sl_range", (5, 8)),
            "scalp_tp_range": profile.get("scalp_tp_range", (3, 5)),
            "swing_sl_range": profile.get("swing_sl_range", (10, 20)),
            "swing_tp_range": profile.get("swing_tp_range", (10, 30)),
            "cooldown_after_sl_min": profile.get("cooldown_after_sl_min", 15),
            "best_sessions": profile.get("best_sessions", []),
            "session_note": profile.get("session_notes", {}).get(session.get("session", ""), ""),
        }

        # Position sizing: pre-compute max units for this pair
        m5_atr = m5.get("atr_pips") if isinstance(m5, dict) else None
        mid_price = price_data.get("mid", 0)
        pair_data["sizing"] = {
            "scalp": compute_max_units(pair, account["nav"], account["margin_avail"], "scalp", m5_atr, mid_price),
            "swing": compute_max_units(pair, account["nav"], account["margin_avail"], "swing", m5_atr, mid_price),
        }

        # ATR-adaptive scalp parameters (TP/SL/trail) with live spread
        ev_risk = detect_event_risk(pair, macro_bias)
        live_spread = price_data.get("spread_pips")
        pair_data["scalp_params"] = compute_scalp_params(pair, m5_atr or 5.0, ev_risk, live_spread)

        # M5 regime for strategy selection
        m5_regime = m5.get("regime", "unknown") if isinstance(m5, dict) else "unknown"
        pair_data["regime"] = m5_regime

        pairs[pair] = pair_data

    # Risk (session already computed above)
    risk = compute_risk(account, positions)

    # Currency strength and market regime (cross-pair analysis)
    ccy_strength = compute_currency_strength(pairs)
    market = compute_market_regime(pairs, session, ccy_strength)

    # Post-scoring: inject currency strength differential into confluence
    for pair in ALL_PAIRS:
        if pair not in pairs:
            continue
        base, quote = pair.split("_")
        base_cs = ccy_strength.get(base, 0)
        quote_cs = ccy_strength.get(quote, 0)
        cs_diff = round(abs(base_cs - quote_cs), 2)
        sig = pairs[pair].get("signal", {})
        for d in ("long", "short"):
            conf = sig.get(f"{d}_confluence", {})
            if isinstance(conf, dict):
                conf["cs_diff"] = cs_diff
                conf["cs_base"] = round(base_cs, 2)
                conf["cs_quote"] = round(quote_cs, 2)
                # High CS divergence = high conviction (data: 100% win when cs_diff>0.5)
                if cs_diff >= 0.5:
                    conf["cs_quality"] = "strong"
                elif cs_diff >= 0.3:
                    conf["cs_quality"] = "moderate"
                else:
                    conf["cs_quality"] = "weak"

    # Recently closed trades (for Claude to check before closing)
    recently_closed = load_recently_closed()

    return {
        "timestamp": now,
        "pairs": pairs,
        "positions": positions,
        "account": account,
        "risk": risk,
        "session": session,
        "market": market,
        "actions_taken": actions,
        "recently_closed": recently_closed,
    }


def _verify_predictions(monitor: dict) -> None:
    """Auto-verify open predictions against current prices.

    Predictions are written by scalp-fast/swing-trader as:
    {
      "id": "pred_20260319_114500_EU",
      "timestamp": "2026-03-19T11:45:00Z",
      "agent": "scalp-fast",
      "pair": "EUR_USD",
      "direction": "SHORT",
      "target": 1.1460,
      "invalidation": 1.1485,
      "entry_price": 1.14770,
      "reason": "London selling into NY open, M5 MACD hist shrinking",
      "score_at_entry": 8,
      "score_agreed": false,
      "indicators_at_entry": {"m5_adx": 25, "m1_rsi": 62, "m5_bbw": 0.0003},
      "session": "LONDON",
      "status": "open",
      "result": null,
      "resolved_at": null,
      "pips": null
    }
    """
    try:
        with open(PREDICTION_TRACKER_PATH) as f:
            predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    changed = False
    prices = {}
    for pair, data in monitor.get("pairs", {}).items():
        p = data.get("price", {})
        prices[pair] = {"bid": p.get("bid", 0), "ask": p.get("ask", 0), "mid": p.get("mid", 0)}

    now = monitor.get("timestamp", "")

    for pred in predictions:
        if pred.get("status") != "open":
            continue
        pair = pred.get("pair", "")
        if pair not in prices:
            continue

        mid = prices[pair]["mid"]
        direction = pred.get("direction", "")
        target = pred.get("target")
        invalidation = pred.get("invalidation")
        entry = pred.get("entry_price", 0)

        if not target or not invalidation or not entry:
            continue

        # Check if target reached
        if direction == "LONG" and mid >= target:
            pred["status"] = "correct"
            pred["result"] = "target_reached"
            pred["resolved_at"] = now
            pred["pips"] = round((mid - entry) * (100 if pair.endswith("JPY") else 10000), 1)
            changed = True
        elif direction == "SHORT" and mid <= target:
            pred["status"] = "correct"
            pred["result"] = "target_reached"
            pred["resolved_at"] = now
            pred["pips"] = round((entry - mid) * (100 if pair.endswith("JPY") else 10000), 1)
            changed = True
        # Check if invalidated
        elif direction == "LONG" and mid <= invalidation:
            pred["status"] = "wrong"
            pred["result"] = "invalidated"
            pred["resolved_at"] = now
            pred["pips"] = round((mid - entry) * (100 if pair.endswith("JPY") else 10000), 1)
            changed = True
        elif direction == "SHORT" and mid >= invalidation:
            pred["status"] = "wrong"
            pred["result"] = "invalidated"
            pred["resolved_at"] = now
            pred["pips"] = round((entry - mid) * (100 if pair.endswith("JPY") else 10000), 1)
            changed = True
        # Check if stale (>30min for scalp, >8hr for swing)
        else:
            try:
                from datetime import datetime, timedelta
                pred_time = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))
                now_time = datetime.fromisoformat(now.replace("Z", "+00:00"))
                max_age = timedelta(minutes=30) if pred.get("agent") == "scalp-fast" else timedelta(hours=8)
                if now_time - pred_time > max_age:
                    # Expired — check if direction was at least right
                    if (direction == "LONG" and mid > entry) or (direction == "SHORT" and mid < entry):
                        pred["status"] = "partial"
                        pred["result"] = "direction_correct_timeout"
                    else:
                        pred["status"] = "wrong"
                        pred["result"] = "direction_wrong_timeout"
                    pred["resolved_at"] = now
                    pips_mult = 100 if pair.endswith("JPY") else 10000
                    pred["pips"] = round((mid - entry) * pips_mult, 1) if direction == "LONG" \
                        else round((entry - mid) * pips_mult, 1)
                    changed = True
            except Exception:
                pass

    if changed:
        with open(PREDICTION_TRACKER_PATH, "w") as f:
            json.dump(predictions, f, indent=2)


def _build_summary(monitor: dict) -> dict:
    """Build compact summary for scalp-fast. ~2KB instead of ~25KB."""
    pairs_summary = {}
    for pair, data in monitor.get("pairs", {}).items():
        price = data.get("price", {})
        micro = data.get("micro", {})
        m1 = data.get("m1", {})
        m5 = data.get("m5", {})
        bias = data.get("bias", {})
        signal = data.get("signal", {})
        sizing = data.get("sizing", {}).get("scalp", {})
        sp = data.get("scalp_params", {})

        # H1 direction as human-readable string
        h1_plus = bias.get("h1_plus_di", 0) or 0
        h1_minus = bias.get("h1_minus_di", 0) or 0
        h1_adx_val = bias.get("h1_adx", 0) or 0
        if h1_adx_val < 15:
            h1_dir_str = "DEAD"
        elif h1_plus > h1_minus:
            h1_dir_str = "BULL"
        else:
            h1_dir_str = "BEAR"

        pairs_summary[pair] = {
            "bid": price.get("bid"), "ask": price.get("ask"),
            "spread": price.get("spread_pips"),
            "micro_dir": micro.get("direction"), "micro_vel": micro.get("velocity"),
            # H1/H4 — MTF context (direction king)
            "h1_bias": h1_dir_str,
            "h1_adx": round(h1_adx_val, 1),
            "h1_rsi": round(bias.get("h1_rsi", 50) or 50, 1),
            "h1_di_plus": round(h1_plus, 1),
            "h1_di_minus": round(h1_minus, 1),
            "h4_regime": bias.get("h4_regime", "unknown"),
            # Key predictive indicators
            "m1_rsi": round(m1.get("rsi", 50), 1),
            "m1_stoch": round(m1.get("stoch_rsi", 0.5), 2),
            "m5_adx": round(m5.get("adx", 0), 1),
            "m5_rsi": round(m5.get("rsi", 50), 1),
            "m5_macd_hist": round(m5.get("macd_hist", 0), 5),
            "m5_bbw": round(m5.get("bbw", 0), 5),
            "m5_bb_pos": round((price.get("mid", 0) - m5.get("bb_lower", 0)) /
                               max(m5.get("bb_upper", 1) - m5.get("bb_lower", 1), 0.0001), 2)
                         if m5.get("bb_upper") and m5.get("bb_lower") else None,
            "m5_vwap_gap": round(m5.get("vwap_gap", 0), 1),
            "m5_ichimoku_cloud": round(m5.get("ichimoku_cloud_pos", 0), 1),
            "m5_div_rsi": m5.get("div_rsi_kind", 0),
            "m5_div_macd": m5.get("div_macd_kind", 0),
            "swing_dist_high": m5.get("swing_dist_high"),
            "swing_dist_low": m5.get("swing_dist_low"),
            # Scores
            "long_score": signal.get("long_score", 0),
            "short_score": signal.get("short_score", 0),
            # Sizing
            "can_trade": sizing.get("can_trade", False),
            "rec_units": sizing.get("recommended_units", 0),
            # Scalp params
            "tp_pips": sp.get("tp_pips"), "sl_pips": sp.get("sl_pips"),
        }

    # Currency strength flow per pair — makes flow conflict instantly visible
    cs = monitor.get("market", {}).get("currency_strength", {})
    for pair, psum in pairs_summary.items():
        base, quote = pair.split("_")
        base_cs = cs.get(base, 0)
        quote_cs = cs.get(quote, 0)
        psum["cs_base"] = round(base_cs, 2)
        psum["cs_quote"] = round(quote_cs, 2)
        cs_diff = base_cs - quote_cs  # positive = base stronger = LONG flow
        if abs(cs_diff) >= 0.5:
            psum["cs_flow"] = "STRONG_LONG" if cs_diff > 0 else "STRONG_SHORT"
        elif abs(cs_diff) >= 0.3:
            psum["cs_flow"] = "LEAN_LONG" if cs_diff > 0 else "LEAN_SHORT"
        else:
            psum["cs_flow"] = "NEUTRAL"

    # Recent predictions for agent context (so agents know what they predicted last)
    recent_preds = []
    try:
        with open(PREDICTION_TRACKER_PATH) as f:
            all_preds = json.load(f)
        # Last 5 predictions (any agent)
        for p in all_preds[-5:]:
            recent_preds.append({
                "agent": p.get("agent"),
                "pair": p.get("pair"),
                "direction": p.get("direction"),
                "reason": p.get("reason", "")[:80],
                "status": p.get("status"),
                "pips": p.get("pips"),
                "age_min": None,
            })
            # Calculate age
            try:
                from datetime import datetime
                pt = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
                nt = datetime.fromisoformat(monitor["timestamp"].replace("Z", "+00:00"))
                recent_preds[-1]["age_min"] = round((nt - pt).total_seconds() / 60, 1)
            except Exception:
                pass
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Prediction accuracy stats
    pred_stats = {}
    try:
        with open(PREDICTION_TRACKER_PATH) as f:
            preds = json.load(f)
        for agent in ["scalp-fast", "swing-trader"]:
            resolved = [p for p in preds if p.get("agent") == agent and p.get("status") != "open"]
            last_10 = resolved[-10:] if len(resolved) > 10 else resolved
            if last_10:
                correct = sum(1 for p in last_10 if p["status"] in ("correct", "partial"))
                wrong = sum(1 for p in last_10 if p["status"] == "wrong")
                total = len(last_10)
                # Find best/worst patterns
                by_pair = {}
                for p in resolved[-20:]:
                    pair = p.get("pair", "?")
                    if pair not in by_pair:
                        by_pair[pair] = {"correct": 0, "wrong": 0}
                    if p["status"] in ("correct", "partial"):
                        by_pair[pair]["correct"] += 1
                    elif p["status"] == "wrong":
                        by_pair[pair]["wrong"] += 1
                pred_stats[agent] = {
                    "last_10": f"{correct}/{total} correct",
                    "accuracy_pct": round(correct / total * 100) if total > 0 else 0,
                    "by_pair": {k: f"{v['correct']}/{v['correct']+v['wrong']}" for k, v in by_pair.items()},
                    "open": sum(1 for p in preds if p.get("agent") == agent and p.get("status") == "open"),
                }
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return {
        "timestamp": monitor.get("timestamp"),
        "pairs": pairs_summary,
        "market": monitor.get("market", {}),
        "positions": monitor.get("positions", []),
        "account": monitor.get("account", {}),
        "risk": monitor.get("risk", {}),
        "session": monitor.get("session", {}),
        "actions_taken": monitor.get("actions_taken", []),
        "recently_closed": monitor.get("recently_closed", {}),
        "recent_predictions": recent_preds,
        "prediction_accuracy": pred_stats,
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
            # Write compact summary for scalp-fast (much smaller, faster to read)
            try:
                summary = _build_summary(monitor)
                with open(SUMMARY_PATH, "w") as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"WARN: summary build failed: {e}", file=sys.stderr)
            # Verify open predictions against current prices
            try:
                _verify_predictions(monitor)
            except Exception as e:
                print(f"WARN: prediction verify failed: {e}", file=sys.stderr)
            ts = monitor["timestamp"]
            n_pos = len(monitor["positions"])
            nav = monitor["account"]["nav"]
            n_act = len(monitor.get("actions_taken", []))
            risk = monitor.get("risk", {})
            cb = " CIRCUIT_BREAKER!" if risk.get("circuit_breaker") else ""
            print(f"[{ts}] Monitor: {n_pos} pos, NAV={nav}, actions={n_act}{cb}", file=sys.stderr)
            # Disk space check — warn if < 500MB free
            try:
                import os as _os
                st = _os.statvfs(str(ROOT))
                free_mb = st.f_bavail * st.f_frsize / 1024 / 1024
                if free_mb < 500:
                    warn_line = f"[{ts}] MONITOR: DISK_LOW free={free_mb:.0f}MB — registry/log writes may fail\n"
                    print(warn_line.strip(), file=sys.stderr)
                    with open(CRITICAL_LOG_PATH, "a") as _f:
                        _f.write(warn_line)
            except Exception:
                pass
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        if loop_sec is None:
            break
        time.sleep(loop_sec)


if __name__ == "__main__":
    main()
