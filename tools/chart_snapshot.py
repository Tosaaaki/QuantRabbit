#!/usr/bin/env python3
"""Chart Snapshot — visual candlestick charts + regime detection (Trend/Range/Squeeze).

Generates PNG chart images from OANDA candle data. Claude reads the image
via the Read tool to get VISUAL chart perception — not just indicator numbers.

Also outputs regime detection: TREND / RANGE / SQUEEZE per pair.

Usage:
    python3 tools/chart_snapshot.py EUR_USD M5          # single pair
    python3 tools/chart_snapshot.py EUR_USD H1           # H1 chart
    python3 tools/chart_snapshot.py --all                 # all 7 pairs, M5 + H1
    python3 tools/chart_snapshot.py --all --regime-only   # regime detection only (no charts)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "logs" / "charts"
CHART_DIR.mkdir(exist_ok=True)

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_config():
    text = (ROOT / "config" / "env.toml").read_text()
    token = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_account_id")][0]
    return token, acct


def fetch_candles(pair: str, granularity: str, count: int, token: str, acct: str) -> list[dict]:
    """Fetch candle data from OANDA."""
    url = f"https://api-fxtrade.oanda.com/v3/instruments/{pair}/candles?granularity={granularity}&count={count}&price=MBA"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    return data.get("candles", [])


def fetch_positions(token: str, acct: str) -> dict:
    """Fetch open positions. Returns {pair: {units, price, pl}}."""
    url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/openTrades"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req, timeout=10).read())
    positions = {}
    for t in data.get("trades", []):
        pair = t["instrument"]
        if pair not in positions:
            positions[pair] = []
        positions[pair].append({
            "units": int(t.get("currentUnits", 0)),
            "price": float(t["price"]),
            "pl": float(t.get("unrealizedPL", 0)),
        })
    return positions


def compute_indicators(candles: list[dict]) -> dict:
    """Compute EMA12, EMA20, BB(20,2), Keltner(20,1.5), ATR(14) from candle data."""
    closes = []
    highs = []
    lows = []
    for c in candles:
        mid = c.get("mid", {})
        closes.append(float(mid["c"]))
        highs.append(float(mid["h"]))
        lows.append(float(mid["l"]))

    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)

    # EMA
    def ema(data, period):
        result = np.zeros_like(data)
        result[0] = data[0]
        k = 2 / (period + 1)
        for i in range(1, len(data)):
            result[i] = data[i] * k + result[i - 1] * (1 - k)
        return result

    ema12 = ema(closes, 12)
    ema20 = ema(closes, 20)

    # Bollinger Bands (SMA20, 2σ)
    bb_mid = np.convolve(closes, np.ones(20) / 20, mode="valid")
    pad = len(closes) - len(bb_mid)
    bb_mid_full = np.concatenate([np.full(pad, np.nan), bb_mid])
    bb_std = np.array([np.std(closes[max(0, i - 19):i + 1]) for i in range(len(closes))])
    bb_std[:19] = np.nan
    bb_upper = bb_mid_full + 2 * bb_std
    bb_lower = bb_mid_full - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid_full  # normalized width

    # ATR(14)
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    tr[0] = highs[0] - lows[0]
    atr = ema(tr, 14)

    # Keltner Channel (EMA20 ± 1.5*ATR)
    kc_upper = ema20 + 1.5 * atr
    kc_lower = ema20 - 1.5 * atr

    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "ema12": ema12,
        "ema20": ema20,
        "bb_mid": bb_mid_full,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "atr": atr,
    }


def detect_regime(indicators: dict, pair: str) -> dict:
    """Detect market regime: TREND / RANGE / SQUEEZE.

    Logic:
    - BB inside KC = SQUEEZE (volatility compression, breakout pending)
    - ADX > 25 or EMA12/20 slope steep = TREND
    - ADX < 20, BB width stable, price oscillating between BB bands = RANGE
    """
    closes = indicators["closes"]
    ema12 = indicators["ema12"]
    ema20 = indicators["ema20"]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_width = indicators["bb_width"]
    kc_upper = indicators["kc_upper"]
    kc_lower = indicators["kc_lower"]
    atr = indicators["atr"]

    pip_factor = 100 if "JPY" in pair else 10000

    # Recent values (last 5 candles)
    recent_bb_width = bb_width[-5:]
    recent_bb_width = recent_bb_width[~np.isnan(recent_bb_width)]

    # Squeeze detection: BB inside KC (must be genuine compression, not just low-vol session noise)
    squeeze = False
    if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]):
        bb_inside_kc = bb_upper[-1] < kc_upper[-1] and bb_lower[-1] > kc_lower[-1]
        # Also require BB width to be narrowing (current < average of last 20)
        recent_widths = bb_width[-20:]
        recent_widths = recent_widths[~np.isnan(recent_widths)]
        if len(recent_widths) > 5:
            bb_narrowing = bb_width[-1] < np.mean(recent_widths) * 0.85
        else:
            bb_narrowing = False
        squeeze = bb_inside_kc and bb_narrowing

    # Trend detection: EMA slope — use fixed 10-period lookback regardless of candle count
    lookback = min(10, len(ema20) - 1)
    ema_slope = (ema20[-1] - ema20[-1 - lookback]) * pip_factor / lookback * 10 if lookback > 0 else 0
    ema_sep = abs(ema12[-1] - ema20[-1]) * pip_factor

    # Price position relative to BB
    price = closes[-1]
    bb_pos = (price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if not np.isnan(bb_upper[-1]) and bb_upper[-1] != bb_lower[-1] else 0.5

    # Range detection: price oscillating between bands
    # Count how many times price touched upper/lower 20% of BB range in last 20 candles
    upper_touches = 0
    lower_touches = 0
    mid_time = 0
    for i in range(-min(20, len(closes)), 0):
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range <= 0:
            continue
        pos = (closes[i] - bb_lower[i]) / bb_range
        if pos > 0.8:
            upper_touches += 1
        elif pos < 0.2:
            lower_touches += 1
        else:
            mid_time += 1

    # Determine regime
    if squeeze:
        regime = "SQUEEZE"
        detail = "BB inside KC — volatility compressed. Breakout imminent."
        trade_approach = "Wait for breakout direction. First candle outside BB = entry signal."
    elif abs(ema_slope) > 3.0 and ema_sep > 2.0:
        direction = "BULL" if ema_slope > 0 else "BEAR"
        regime = f"TREND-{direction}"
        detail = f"EMA slope={ema_slope:+.1f}pip, EMA12/20 gap={ema_sep:.1f}pip. Clear trend."
        trade_approach = f"Trade WITH trend ({direction}). Buy dips at EMA20 / BB mid. TP at structure."
    elif upper_touches >= 2 and lower_touches >= 2:
        bb_range_pips = (bb_upper[-1] - bb_lower[-1]) * pip_factor if not np.isnan(bb_upper[-1]) else 0
        # Distinguish real RANGE from low-vol drift: range must be tradeable (> ATR×1.5)
        atr_pips = atr[-1] * pip_factor
        if bb_range_pips > atr_pips * 1.2:
            regime = "RANGE"
            bb_mid_val = (bb_upper[-1] + bb_lower[-1]) / 2
            touch_symmetry = min(upper_touches, lower_touches) / max(upper_touches, lower_touches)
            zone = "LOWER" if bb_pos < 0.25 else "UPPER" if bb_pos > 0.75 else "MID"
            detail = (f"Price bouncing between BB bands. Range={bb_range_pips:.1f}pip (ATR={atr_pips:.1f}). "
                      f"Touches: upper={upper_touches}, lower={lower_touches}. "
                      f"BB pos={bb_pos:.0%} ({zone}). Symmetry={touch_symmetry:.0%}.")
            trade_approach = (f"RANGE SCALP: Buy @{bb_lower[-1]:.5g} → TP1 @{bb_mid_val:.5g} (mid, +{bb_range_pips/2:.1f}pip) "
                              f"→ TP2 @{bb_upper[-1]:.5g} (+{bb_range_pips:.1f}pip). "
                              f"Sell @{bb_upper[-1]:.5g} → TP1 @{bb_mid_val:.5g} (mid). "
                              f"SL=outside range. Rotation: BUY low→TP mid→SELL high→TP mid→repeat.")
        else:
            regime = "LOW-VOL"
            detail = f"Low volatility. BB range={bb_range_pips:.1f}pip, ATR={atr_pips:.1f}pip. Not enough range to trade."
            trade_approach = "Skip or wait for volatility expansion. Spread eats most of the range."
    elif len(recent_bb_width) > 0 and np.mean(recent_bb_width) < 0.005:
        # Only flag as squeeze if BB width is EXTREMELY narrow (not just normal low-vol session)
        regime = "SQUEEZE"
        detail = "BB width extremely narrow. Breakout pending."
        trade_approach = "Wait for breakout direction. First candle outside BB = entry signal."
    else:
        # Mild trend or transition
        direction = "BULL" if ema_slope > 0 else "BEAR"
        regime = f"MILD-{direction}"
        detail = f"Weak trend or transition. EMA slope={ema_slope:+.1f}pip. Not strongly directional."
        trade_approach = f"Cautious {direction} bias. Small size. Quick TP."

    # BB width trend (narrowing → squeeze forming)
    bb_widths_20 = bb_width[-20:]
    bb_widths_20 = bb_widths_20[~np.isnan(bb_widths_20)]
    bb_width_trend = 0.0
    if len(bb_widths_20) > 5 and np.mean(bb_widths_20) > 0:
        bb_width_trend = (bb_width[-1] - np.mean(bb_widths_20)) / np.mean(bb_widths_20) * 100

    result = {
        "regime": regime,
        "detail": detail,
        "trade_approach": trade_approach,
        "bb_position": round(bb_pos, 2),
        "ema_slope_pip": round(ema_slope, 1),
        "ema_separation_pip": round(ema_sep, 1),
        "squeeze": squeeze,
        "atr_pip": round(atr[-1] * pip_factor, 1),
        "bb_range_pip": round((bb_upper[-1] - bb_lower[-1]) * pip_factor, 1) if not np.isnan(bb_upper[-1]) else 0,
        "upper_touches": upper_touches,
        "lower_touches": lower_touches,
        "bb_width_trend_pct": round(bb_width_trend, 1),
    }

    # Add range scalp levels when RANGE detected
    if regime == "RANGE":
        bb_mid_val = (bb_upper[-1] + bb_lower[-1]) / 2
        result["range_scalp"] = {
            "bb_upper": round(float(bb_upper[-1]), 5),
            "bb_mid": round(float(bb_mid_val), 5),
            "bb_lower": round(float(bb_lower[-1]), 5),
            "range_pips": round(bb_range_pips, 1),
            "zone": "LOWER" if bb_pos < 0.25 else "UPPER" if bb_pos > 0.75 else "MID",
            "touch_symmetry": round(min(upper_touches, lower_touches) / max(upper_touches, lower_touches, 1), 2),
        }

    return result


def generate_chart(pair: str, candles: list[dict], indicators: dict,
                   regime: dict, positions: list[dict] | None, tf: str) -> str:
    """Generate candlestick chart with BB, EMA, and regime info. Returns PNG path."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.05})

    # Parse candle data
    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for c in candles:
        mid = c.get("mid", {})
        times.append(datetime.fromisoformat(c["time"][:19]))
        opens.append(float(mid["o"]))
        highs.append(float(mid["h"]))
        lows.append(float(mid["l"]))
        closes.append(float(mid["c"]))
        volumes.append(int(c.get("volume", 0)))

    n = len(times)
    x = np.arange(n)

    # Candlesticks
    for i in range(n):
        color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
        ax1.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.8)
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_height = max(body_high - body_low, (highs[i] - lows[i]) * 0.01)
        ax1.bar(x[i], body_height, bottom=body_low, width=0.6, color=color, edgecolor=color)

    # Bollinger Bands
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_mid = indicators["bb_mid"]
    valid = ~np.isnan(bb_upper)
    if np.any(valid):
        ax1.fill_between(x[valid], bb_lower[valid], bb_upper[valid], alpha=0.08, color="#2196F3")
        ax1.plot(x[valid], bb_upper[valid], color="#2196F3", linewidth=0.7, alpha=0.5)
        ax1.plot(x[valid], bb_lower[valid], color="#2196F3", linewidth=0.7, alpha=0.5)
        ax1.plot(x[valid], bb_mid[valid], color="#2196F3", linewidth=0.5, alpha=0.3, linestyle="--")

    # EMA 12/20
    ax1.plot(x, indicators["ema12"], color="#FF9800", linewidth=1.0, label="EMA12", alpha=0.8)
    ax1.plot(x, indicators["ema20"], color="#9C27B0", linewidth=1.0, label="EMA20", alpha=0.8)

    # Keltner Channel (dotted, for squeeze visual)
    ax1.plot(x, indicators["kc_upper"], color="#4CAF50", linewidth=0.5, linestyle=":", alpha=0.4)
    ax1.plot(x, indicators["kc_lower"], color="#4CAF50", linewidth=0.5, linestyle=":", alpha=0.4)

    # Position entry lines
    if positions:
        for pos in positions:
            side = "LONG" if pos["units"] > 0 else "SHORT"
            color = "#26a69a" if side == "LONG" else "#ef5350"
            ax1.axhline(pos["price"], color=color, linewidth=1.2, linestyle="--", alpha=0.7)
            ax1.text(n - 1, pos["price"], f" {side} {abs(pos['units'])}u @{pos['price']:.5g}",
                     fontsize=8, color=color, va="bottom" if side == "LONG" else "top")

    # Current price
    ax1.axhline(closes[-1], color="white", linewidth=0.5, alpha=0.3)

    # Regime label
    regime_color = {"TREND-BULL": "#26a69a", "TREND-BEAR": "#ef5350",
                    "RANGE": "#2196F3", "SQUEEZE": "#FF9800",
                    "MILD-BULL": "#81C784", "MILD-BEAR": "#E57373"}.get(regime["regime"], "#9E9E9E")
    ax1.text(0.02, 0.95, f"{regime['regime']}  ATR={regime['atr_pip']}pip  BB={regime['bb_range_pip']}pip",
             transform=ax1.transAxes, fontsize=11, fontweight="bold", color=regime_color,
             va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Approach text
    ax1.text(0.02, 0.88, regime["trade_approach"],
             transform=ax1.transAxes, fontsize=8, color="#BDBDBD",
             va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # Volume bars
    vol_colors = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(n)]
    ax2.bar(x, volumes, color=vol_colors, alpha=0.6, width=0.6)
    ax2.set_ylabel("Vol", fontsize=8, color="#9E9E9E")

    # X-axis: time labels
    tick_step = max(1, n // 10)
    ax2.set_xticks(x[::tick_step])
    ax2.set_xticklabels([t.strftime("%H:%M") for t in times[::tick_step]], fontsize=7, color="#9E9E9E")
    ax1.set_xticks([])

    # Styling
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#9E9E9E", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.grid(True, alpha=0.1, color="#555")

    fig.patch.set_facecolor("#0f0f1a")
    ax1.set_title(f"{pair} {tf}  |  {times[-1].strftime('%Y-%m-%d %H:%M')} UTC",
                  fontsize=12, color="#E0E0E0", pad=10)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.5)

    # Save
    path = CHART_DIR / f"{pair}_{tf}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Chart snapshot + regime detection")
    parser.add_argument("pair", nargs="?", help="Pair (e.g., EUR_USD)")
    parser.add_argument("tf", nargs="?", default="M5", help="Timeframe (M5, M15, H1, H4)")
    parser.add_argument("--all", action="store_true", help="All 7 pairs, M5 + H1")
    parser.add_argument("--regime-only", action="store_true", help="Regime detection only, no chart images")
    parser.add_argument("--count", type=int, default=60, help="Number of candles (default: 60)")
    args = parser.parse_args()

    token, acct = load_config()

    if args.all:
        pairs = PAIRS
        timeframes = ["M5", "H1"]
    elif args.pair:
        pairs = [args.pair.upper()]
        timeframes = [args.tf.upper()]
    else:
        parser.print_help()
        return

    # Fetch positions once
    try:
        positions = fetch_positions(token, acct)
    except Exception:
        positions = {}

    # Load previous regimes for transition detection
    regime_history_path = ROOT / "logs" / "regime_history.json"
    prev_regimes = {}
    if regime_history_path.exists():
        try:
            prev_regimes = json.loads(regime_history_path.read_text())
        except Exception:
            pass

    print("=== REGIME DETECTION ===")
    chart_paths = []
    current_regimes = {}

    for pair in pairs:
        for tf in timeframes:
            try:
                candles = fetch_candles(pair, tf, args.count, token, acct)
                if len(candles) < 25:
                    print(f"{pair} {tf}: insufficient data ({len(candles)} candles)")
                    continue

                indicators = compute_indicators(candles)
                regime = detect_regime(indicators, pair)

                # Track regime for history
                key = f"{pair}_{tf}"
                r = regime["regime"]
                current_regimes[key] = r

                # Detect transition
                prev_r = prev_regimes.get(key, "")
                transition = ""
                if prev_r and prev_r != r:
                    transition = f" [was {prev_r}]"

                # Print regime
                emoji = {"TREND-BULL": "📈", "TREND-BEAR": "📉", "RANGE": "↔️",
                         "SQUEEZE": "🔸", "MILD-BULL": "↗️", "MILD-BEAR": "↘️"}.get(r, "❓")
                print(f"{emoji} {pair} {tf}: {r}{transition} | {regime['detail']}")
                print(f"   → {regime['trade_approach']}")
                if regime["regime"] == "RANGE":
                    print(f"   BB pos={regime['bb_position']} | upper_touches={regime['upper_touches']} lower_touches={regime['lower_touches']}")

                # Generate chart
                if not args.regime_only:
                    pos = positions.get(pair)
                    path = generate_chart(pair, candles, indicators, regime, pos, tf)
                    chart_paths.append(path)
                    print(f"   Chart: {path}")

            except Exception as e:
                print(f"{pair} {tf}: ERROR {e}")

    # Save current regimes for next run's transition detection
    try:
        regime_history_path.write_text(json.dumps(current_regimes, indent=2))
    except Exception:
        pass

    if chart_paths:
        print(f"\n=== {len(chart_paths)} charts saved to {CHART_DIR}/ ===")
        print("Read charts with: Read tool → file_path")


if __name__ == "__main__":
    main()
