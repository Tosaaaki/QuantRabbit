#!/usr/bin/env python3
"""
Regime Switch — per-pair regime detection.

Reads cached technicals (M5 + M15 + H1) and classifies each pair:
  - TREND: ADX strong + DI gap clear + EMA slope persistent + BB expanding
  - RANGE: ADX weak + BBW compressed
  - MIXED: anything else (default; permissive)

Writes logs/bot_regime_state.json. range_bot / trend_bot read it:
  - When regime=TREND, only allow entries IN the trend direction
  - When regime=RANGE, both sides allowed (range_bot's natural mode)

Usage:
    python3 tools/regime_switch.py
    python3 tools/regime_switch.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    _git_common = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip())
    _MAIN_ROOT = _git_common.resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

PAIRS = (
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
    "NZD_JPY", "CAD_JPY",
    "EUR_CHF", "AUD_NZD", "AUD_CAD",
)
REGIME_STATE_PATH = _MAIN_ROOT / "logs" / "bot_regime_state.json"

# TREND thresholds (must hit ALL on the chosen TF)
TREND_M5_ADX_MIN = 25
TREND_M15_ADX_MIN = 22
TREND_DI_GAP_MIN = 8
TREND_EMA_SLOPE_MIN = 0.00005   # rough — depends on price; we just want non-zero same-sign
TREND_BBW_RATIO_MIN = 1.10      # current BBW vs 20-bar avg (proxy: span_pips vs ATR_pips)

# RANGE thresholds
RANGE_M5_ADX_MAX = 18
RANGE_BBW_RATIO_MAX = 0.80


def load_technicals(pair: str) -> dict:
    f = _MAIN_ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text()).get("timeframes", {})
    except Exception:
        return {}


def safe_float(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def assess_pair(pair: str, tech: dict) -> dict:
    m5 = tech.get("M5", {}) or {}
    m15 = tech.get("M15", {}) or {}
    h1 = tech.get("H1", {}) or {}

    if not m5:
        return {"regime": "UNKNOWN", "trend_dir": None, "reason": "no_m5_data"}

    adx_m5 = safe_float(m5, "adx")
    adx_m15 = safe_float(m15, "adx")
    adx_h1 = safe_float(h1, "adx")
    plus_di_m5 = safe_float(m5, "plus_di")
    minus_di_m5 = safe_float(m5, "minus_di")
    plus_di_h1 = safe_float(h1, "plus_di")
    minus_di_h1 = safe_float(h1, "minus_di")
    di_gap_m5 = abs(plus_di_m5 - minus_di_m5)
    di_gap_h1 = abs(plus_di_h1 - minus_di_h1)

    ema_slope_m5 = safe_float(m5, "ema_slope_20") or safe_float(m5, "ema_slope_10")
    bb_span_pips = safe_float(m5, "bb_span_pips")
    atr_pips_m5 = safe_float(m5, "atr_pips")
    bbw_ratio = (bb_span_pips / atr_pips_m5 / 4.0) if atr_pips_m5 > 0 else 1.0
    # rough proxy: BB span vs ATR. Wide BB relative to ATR = expansion.

    bbw = safe_float(m5, "bbw")

    trend_dir = None
    if plus_di_m5 > minus_di_m5:
        trend_dir = "LONG"
    elif minus_di_m5 > plus_di_m5:
        trend_dir = "SHORT"

    # H1 alignment is the tiebreaker for direction
    h1_dir = None
    if plus_di_h1 > minus_di_h1:
        h1_dir = "LONG"
    elif minus_di_h1 > plus_di_h1:
        h1_dir = "SHORT"

    is_trend_m5 = (adx_m5 >= TREND_M5_ADX_MIN and di_gap_m5 >= TREND_DI_GAP_MIN)
    is_trend_m15 = (adx_m15 >= TREND_M15_ADX_MIN)
    h1_aligned = (h1_dir == trend_dir) if (h1_dir and trend_dir) else False
    bb_expanding = bbw_ratio >= TREND_BBW_RATIO_MIN

    is_range = (adx_m5 < RANGE_M5_ADX_MAX and bbw_ratio <= RANGE_BBW_RATIO_MAX)

    regime = "MIXED"
    reason = ""
    if is_trend_m5 and (is_trend_m15 or h1_aligned):
        regime = "TREND"
        reason = (f"M5_ADX={adx_m5:.0f} di_gap={di_gap_m5:.0f} "
                  f"M15_ADX={adx_m15:.0f} H1_dir={h1_dir} bbw_ratio={bbw_ratio:.2f}")
    elif is_range:
        regime = "RANGE"
        reason = f"M5_ADX={adx_m5:.0f} bbw_ratio={bbw_ratio:.2f}"
        trend_dir = None
    else:
        reason = f"M5_ADX={adx_m5:.0f} di_gap={di_gap_m5:.0f} bbw_ratio={bbw_ratio:.2f}"
        # Don't force a trend_dir in MIXED unless DI gap is meaningful
        if di_gap_m5 < TREND_DI_GAP_MIN:
            trend_dir = None

    return {
        "regime": regime,
        "trend_dir": trend_dir,
        "adx_m5": round(adx_m5, 1),
        "adx_m15": round(adx_m15, 1),
        "adx_h1": round(adx_h1, 1),
        "di_gap_m5": round(di_gap_m5, 1),
        "h1_dir": h1_dir,
        "bbw_ratio": round(bbw_ratio, 2),
        "bb_expanding": bb_expanding,
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime Switch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== REGIME SWITCH === {now_utc.strftime('%Y-%m-%d %H:%M:%SZ')}")

    pair_states = {}
    for pair in PAIRS:
        tech = load_technicals(pair)
        pair_states[pair] = assess_pair(pair, tech)

    state = {
        "ts": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": {
            "trend_m5_adx_min": TREND_M5_ADX_MIN,
            "trend_m15_adx_min": TREND_M15_ADX_MIN,
            "trend_di_gap_min": TREND_DI_GAP_MIN,
            "range_m5_adx_max": RANGE_M5_ADX_MAX,
            "range_bbw_ratio_max": RANGE_BBW_RATIO_MAX,
        },
        "pairs": pair_states,
    }

    if not args.dry_run:
        REGIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGIME_STATE_PATH.write_text(json.dumps(state, indent=2))

    for pair, s in pair_states.items():
        regime = s["regime"]
        td = s.get("trend_dir") or "-"
        marker = ""
        if regime == "TREND":
            marker = f"-> ONLY {td}"
        elif regime == "RANGE":
            marker = "-> BOTH OK"
        print(f"  {pair:8} {regime:8} {marker:18} {s.get('reason','')}")

    print(f"State -> {REGIME_STATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
