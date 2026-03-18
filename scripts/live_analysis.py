#!/usr/bin/env python3
"""USD/JPY Real-Time Scalping Analysis - 15 cycles, 12s apart. RESEARCH ONLY."""

import urllib.request
import json
import time
import math
from datetime import datetime, timezone

TOKEN = "97535b26b8ce757470c8996155ed764d-c8efe747b14c4e3c50fe765c59a7a378"
ACCOUNT = "001-009-13679149-002"
BASE = "https://api-fxtrade.oanda.com/v3"
INSTRUMENT = "USD_JPY"

def fetch_candles(granularity, count):
    url = f"{BASE}/instruments/{INSTRUMENT}/candles?granularity={granularity}&count={count}&price=MAB"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    candles = []
    for c in data.get("candles", []):
        if c["complete"] or granularity == "M1":
            mid = c["mid"]
            candles.append({
                "time": c["time"],
                "o": float(mid["o"]), "h": float(mid["h"]),
                "l": float(mid["l"]), "c": float(mid["c"]),
                "vol": int(c["volume"]),
                "ba_spread": float(c["ask"]["c"]) - float(c["bid"]["c"])
            })
    # Always include last candle even if incomplete
    if not candles or candles[-1]["time"] != data["candles"][-1]["time"]:
        c = data["candles"][-1]
        mid = c["mid"]
        candles.append({
            "time": c["time"],
            "o": float(mid["o"]), "h": float(mid["h"]),
            "l": float(mid["l"]), "c": float(mid["c"]),
            "vol": int(c["volume"]),
            "ba_spread": float(c["ask"]["c"]) - float(c["bid"]["c"])
        })
    return candles

def ema(values, period):
    if len(values) < period:
        return [None] * len(values)
    result = [None] * (period - 1)
    k = 2.0 / (period + 1)
    sma = sum(values[:period]) / period
    result.append(sma)
    for i in range(period, len(values)):
        val = values[i] * k + result[-1] * (1 - k)
        result.append(val)
    return result

def sma(values, period):
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(values[i-period+1:i+1]) / period)
    return result

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    if len(gains) < period:
        return None
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))

def stochastic(candles, k_period=14, d_period=3):
    if len(candles) < k_period:
        return None, None
    k_vals = []
    for i in range(k_period - 1, len(candles)):
        window = candles[i - k_period + 1:i + 1]
        highest = max(c["h"] for c in window)
        lowest = min(c["l"] for c in window)
        if highest == lowest:
            k_vals.append(50.0)
        else:
            k_vals.append(100.0 * (candles[i]["c"] - lowest) / (highest - lowest))
    if len(k_vals) < d_period:
        return k_vals[-1] if k_vals else None, None
    d_val = sum(k_vals[-d_period:]) / d_period
    return k_vals[-1], d_val

def macd(closes, fast=12, slow=26, signal=9):
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = []
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    valid = [v for v in macd_line if v is not None]
    if len(valid) < signal:
        return None, None, None
    sig = ema(valid, signal)
    macd_val = valid[-1]
    sig_val = sig[-1] if sig else None
    hist = macd_val - sig_val if sig_val is not None else None
    return macd_val, sig_val, hist

def atr(candles, period=14):
    if len(candles) < 2:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else None
    atr_val = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
    return atr_val

def bollinger(closes, period=20, std_mult=2.0):
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return mid - std_mult * std, mid, mid + std_mult * std

def adx(candles, period=14):
    if len(candles) < period + 1:
        return None
    plus_dm_list, minus_dm_list, tr_list = [], [], []
    for i in range(1, len(candles)):
        h, l = candles[i]["h"], candles[i]["l"]
        ph, pl = candles[i-1]["h"], candles[i-1]["l"]
        pc = candles[i-1]["c"]
        plus_dm = max(h - ph, 0) if (h - ph) > (pl - l) else 0
        minus_dm = max(pl - l, 0) if (pl - l) > (h - ph) else 0
        tr = max(h - l, abs(h - pc), abs(l - pc))
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(tr)
    if len(tr_list) < period:
        return None
    atr_v = sum(tr_list[:period])
    plus_di_sum = sum(plus_dm_list[:period])
    minus_di_sum = sum(minus_dm_list[:period])
    for i in range(period, len(tr_list)):
        atr_v = atr_v - atr_v / period + tr_list[i]
        plus_di_sum = plus_di_sum - plus_di_sum / period + plus_dm_list[i]
        minus_di_sum = minus_di_sum - minus_di_sum / period + minus_dm_list[i]
    if atr_v == 0:
        return None
    plus_di = 100 * plus_di_sum / atr_v
    minus_di = 100 * minus_di_sum / atr_v
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0
    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx

def momentum(closes, period=5):
    if len(closes) < period + 1:
        return None
    return closes[-1] - closes[-period - 1]

def find_swing_points(candles, lookback=3):
    """Find swing highs and lows using N-bar pivots."""
    swings = []
    for i in range(lookback, len(candles) - lookback):
        is_high = all(candles[i]["h"] >= candles[i+j]["h"] and candles[i]["h"] >= candles[i-j]["h"]
                       for j in range(1, lookback + 1))
        is_low = all(candles[i]["l"] <= candles[i+j]["l"] and candles[i]["l"] <= candles[i-j]["l"]
                      for j in range(1, lookback + 1))
        if is_high:
            swings.append(("H", candles[i]["h"], i))
        if is_low:
            swings.append(("L", candles[i]["l"], i))
    return swings

def analyze_nwave(swings):
    if len(swings) < 4:
        return "INSUFFICIENT DATA"
    last4 = swings[-4:]
    types = "".join(s[0] for s in last4)
    vals = [s[1] for s in last4]
    if types == "LHLH":
        if vals[2] > vals[0] and vals[3] > vals[1]:
            return "UPTREND N-WAVE (HL+HH)"
        elif vals[2] < vals[0] and vals[3] < vals[1]:
            return "DOWNTREND (LL+LH)"
        else:
            return "RANGE"
    elif types == "HLHL":
        if vals[1] > vals[3] and vals[0] > vals[2]:
            return "DOWNTREND N-WAVE (LH+LL)"
        elif vals[1] < vals[3] and vals[0] < vals[2]:
            return "UPTREND (HH+HL)"
        else:
            return "RANGE"
    else:
        return f"MIXED({types})"

def score_h1(candles):
    closes = [c["c"] for c in candles]
    e5 = ema(closes, 5)
    e9 = ema(closes, 9)
    e21 = ema(closes, min(21, len(closes)))
    adx_val = adx(candles)
    score = 0
    if e5[-1] and e9[-1] and e21[-1]:
        if e5[-1] > e9[-1] > e21[-1]:
            score = 1
            if adx_val and adx_val > 25:
                score = 2
        elif e5[-1] < e9[-1] < e21[-1]:
            score = -1
            if adx_val and adx_val > 25:
                score = -2
    return score, adx_val

def score_m5(candles, h1_bias):
    closes = [c["c"] for c in candles]
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    rsi_val = rsi(closes)
    price = closes[-1]
    score = 0
    if e9[-1] and e21[-1]:
        dist_e9 = (price - e9[-1]) * 100  # in pips
        dist_e21 = (price - e21[-1]) * 100
        if h1_bias > 0:  # bullish
            if -3 <= dist_e9 <= 1:  # pullback to EMA9 in uptrend
                score = 2
            elif -3 <= dist_e21 <= 1:
                score = 1
        elif h1_bias < 0:  # bearish
            if -1 <= dist_e9 <= 3:  # pullback to EMA9 in downtrend
                score = -2
            elif -1 <= dist_e21 <= 3:
                score = -1
        else:
            if rsi_val and rsi_val < 35:
                score = 1
            elif rsi_val and rsi_val > 65:
                score = -1
    return score

def score_m1(candles):
    closes = [c["c"] for c in candles]
    rsi_val = rsi(closes)
    mom = momentum(closes)
    score = 0
    if rsi_val is not None and mom is not None:
        if rsi_val < 35 and mom > 0:
            score = 2
        elif rsi_val < 40 and mom > 0:
            score = 1
        elif rsi_val > 65 and mom < 0:
            score = -2
        elif rsi_val > 60 and mom < 0:
            score = -1
    return score

def get_signal(total, h1, m5, m1, stoch_m5, rsi_m5, near_ema, vol_spike):
    if total >= 4 and near_ema and (stoch_m5 is not None and stoch_m5 < 20):
        return "STRONG LONG", "H1 bull + M5 pullback to EMA + Stoch oversold"
    elif total >= 3 and near_ema:
        return "MOD LONG", "Multi-TF alignment bull + near EMA"
    elif total >= 2 and m1 >= 1:
        return "MOD LONG", "Decent alignment + M1 momentum up"
    elif total <= -4 and near_ema and (stoch_m5 is not None and stoch_m5 > 80):
        return "STRONG SHORT", "H1 bear + M5 pullback to EMA + Stoch overbought"
    elif total <= -3 and near_ema:
        return "MOD SHORT", "Multi-TF alignment bear + near EMA"
    elif total <= -2 and m1 <= -1:
        return "MOD SHORT", "Decent alignment + M1 momentum down"
    else:
        reasons = []
        if abs(total) < 2:
            reasons.append("weak alignment")
        if not near_ema:
            reasons.append("not near EMA")
        if stoch_m5 and 30 < stoch_m5 < 70:
            reasons.append("stoch neutral")
        return "WAIT", " + ".join(reasons) if reasons else "no clear setup"

def f(val, fmt=".1f", mul=1):
    """Safe format: return formatted val or '-'."""
    if val is None:
        return "-"
    return format(val * mul, fmt)

def run_cycle(cycle_num):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%H:%M:%S")

    try:
        h1 = fetch_candles("H1", 10)
        m5 = fetch_candles("M5", 30)
        m1 = fetch_candles("M1", 20)
    except Exception as e:
        print(f"[{ts}] FETCH ERROR: {e}")
        return

    price = m1[-1]["c"]

    # === H1 Analysis ===
    h1_score, h1_adx = score_h1(h1)
    h1_closes = [c["c"] for c in h1]
    h1_rsi = rsi(h1_closes)
    h1_atr = atr(h1)

    # === M5 Analysis ===
    m5_closes = [c["c"] for c in m5]
    m5_e5 = ema(m5_closes, 5)
    m5_e9 = ema(m5_closes, 9)
    m5_e21 = ema(m5_closes, 21)
    m5_rsi = rsi(m5_closes)
    m5_stoch_k, m5_stoch_d = stochastic(m5)
    m5_macd_val, m5_macd_sig, m5_macd_hist = macd(m5_closes)
    m5_atr = atr(m5)
    m5_bb_lo, m5_bb_mid, m5_bb_hi = bollinger(m5_closes)
    m5_score = score_m5(m5, h1_score)

    # Distance to EMAs
    dist_e9 = (price - m5_e9[-1]) * 100 if m5_e9[-1] else None  # pips
    dist_e21 = (price - m5_e21[-1]) * 100 if m5_e21[-1] else None
    near_ema = False
    if dist_e9 is not None and abs(dist_e9) <= 3:
        near_ema = True
    elif dist_e21 is not None and abs(dist_e21) <= 3:
        near_ema = True

    # === M1 Analysis ===
    m1_closes = [c["c"] for c in m1]
    m1_rsi = rsi(m1_closes)
    m1_stoch_k, m1_stoch_d = stochastic(m1)
    m1_mom = momentum(m1_closes)
    m1_score = score_m1(m1)

    # Volume analysis
    m1_vols = [c["vol"] for c in m1]
    avg_vol = sum(m1_vols[:-1]) / max(len(m1_vols) - 1, 1)
    vol_ratio = m1_vols[-1] / avg_vol if avg_vol > 0 else 1.0
    vol_spike = vol_ratio > 2.0

    # Total score
    total = h1_score + m5_score + m1_score

    # Regime
    if h1_adx and h1_adx > 25:
        regime = "TREND"
    else:
        regime = "RANGE"

    # N-Wave
    swings = find_swing_points(m5)
    nwave = analyze_nwave(swings)

    # Signal
    signal, reason = get_signal(total, h1_score, m5_score, m1_score,
                                 m5_stoch_k, m5_rsi, near_ema, vol_spike)

    # === Output ===
    ema_tag = ""
    if near_ema:
        if dist_e9 is not None and abs(dist_e9) <= 3:
            ema_tag = " *** SETUP ZONE (near EMA9) ***"
        else:
            ema_tag = " *** SETUP ZONE (near EMA21) ***"

    stoch_tag = ""
    if m5_stoch_k is not None:
        if m5_stoch_k > 80:
            stoch_tag = " [OVERBOUGHT]"
        elif m5_stoch_k < 20:
            stoch_tag = " [OVERSOLD]"

    vol_tag = " !!SPIKE!!" if vol_spike else ""

    print(f"\n{'='*72}")
    print(f"[{ts}] Cycle {cycle_num}/15 | Price:{price:.3f} | Score:{total:+d} (H1:{h1_score:+d} M5:{m5_score:+d} M1:{m1_score:+d}) | Regime:{regime}")
    print(f"  H1: ADX:{f(h1_adx)} RSI:{f(h1_rsi)} ATR:{f(h1_atr, '.1f', 100)}p")
    print(f"  M5: RSI:{f(m5_rsi)} Stoch:{f(m5_stoch_k, '.0f')}/{f(m5_stoch_d, '.0f')}{stoch_tag} EMA9:{f(m5_e9[-1], '.3f')} dist:{f(dist_e9)}p | BB:{f(m5_bb_lo, '.3f')}-{f(m5_bb_hi, '.3f')}")
    print(f"      MACD:{f(m5_macd_val, '.4f')} Hist:{f(m5_macd_hist, '.4f')} ATR:{f(m5_atr, '.1f', 100)}p")
    print(f"  M1: RSI:{f(m1_rsi)} Stoch:{f(m1_stoch_k, '.0f')}/{f(m1_stoch_d, '.0f')} Mom:{f(m1_mom, '.1f', 100)}p Vol:{vol_ratio:.1f}x{vol_tag}")
    print(f"  N-Wave: {nwave}{ema_tag}")
    print(f"  >>> SIGNAL: {signal} - {reason}")
    if signal != "WAIT":
        if "SHORT" in signal:
            sl_dist = 10
            tp_dist = 15
            print(f"      Entry zone: {price:.3f} | SL: {price + sl_dist/100:.3f} (+{sl_dist}p) | TP: {price - tp_dist/100:.3f} (-{tp_dist}p)")
        else:
            sl_dist = 10
            tp_dist = 15
            print(f"      Entry zone: {price:.3f} | SL: {price - sl_dist/100:.3f} (-{sl_dist}p) | TP: {price + tp_dist/100:.3f} (+{tp_dist}p)")
        print(f"      NOTE: FOMC+BOJ today - SIZE 50% REDUCED")
    print(f"{'='*72}")

def main():
    print("=" * 72)
    print("USD/JPY REAL-TIME SCALP ANALYSIS - RESEARCH ONLY (NO ORDERS)")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("15 cycles, 12 seconds apart")
    print("RISK: FOMC + BOJ today = HIGH event risk. Size 50% reduced.")
    print("=" * 72)

    for i in range(1, 16):
        run_cycle(i)
        if i < 15:
            time.sleep(12)

    print("\n\n=== ANALYSIS COMPLETE (15 cycles) ===")

if __name__ == "__main__":
    main()
