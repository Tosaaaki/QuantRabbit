#!/usr/bin/env python3
"""
Fib/Wave — Fibonacci retracement/extension + N-wave structure detection

Usage:
    python3 tools/fib_wave.py USD_JPY M5 100   # single pair
    python3 tools/fib_wave.py --all              # all 7 pairs at once (M5 100)
    python3 tools/fib_wave.py --all H1 200       # all pairs + specify TF/count
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_PAIRS = [
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY",
]

FIB_RET = [0.236, 0.382, 0.500, 0.618, 0.786]
FIB_EXT = [1.272, 1.618, 2.000]


# ── Config / API ─────────────────────────────────────────────

def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def fetch_candles(pair, tf, count, token):
    url = (
        f"https://api-fxtrade.oanda.com/v3/instruments/{pair}/candles"
        f"?granularity={tf}&count={count}&price=MBA"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    candles = data.get("candles", [])
    # Normalise to simple dict with h/l/o/c
    out = []
    for c in candles:
        if not c.get("complete", True) and c is not candles[-1]:
            continue
        m = c.get("mid", {})
        out.append({
            "high": float(m.get("h", 0)),
            "low": float(m.get("l", 0)),
            "open": float(m.get("o", 0)),
            "close": float(m.get("c", 0)),
            "time": c.get("time", ""),
        })
    return out


# ── Pip helper ───────────────────────────────────────────────

def pip_size(pair):
    """JPY pairs = 0.01, others = 0.0001"""
    return 0.01 if "JPY" in pair else 0.0001


def to_pips(diff, pair):
    return round(diff / pip_size(pair), 1)


# ── Pivot extraction (adapted from archived n_wave.py) ───────

def extract_pivots(candles, pair, *, window=3, min_pips=1.2, max_points=120):
    buf = list(candles or [])
    if not buf:
        return []
    if max_points and len(buf) > max_points:
        buf = buf[-max_points:]

    pip = pip_size(pair)
    pivots = []
    last_price = None
    last_kind = None
    size = len(buf)

    for idx in range(window, size - window):
        high = buf[idx]["high"]
        low = buf[idx]["low"]
        is_hi = True
        is_lo = True

        for off in range(idx - window, idx + window + 1):
            if off == idx:
                continue
            if buf[off]["high"] > high:
                is_hi = False
            if buf[off]["low"] < low:
                is_lo = False
            if not is_hi and not is_lo:
                break

        if not is_hi and not is_lo:
            continue

        kind = "high" if is_hi else "low"
        price = high if kind == "high" else low

        # Filter tiny swings
        if last_price is not None and kind != last_kind:
            if abs(price - last_price) / pip < min_pips:
                continue

        # Merge consecutive same-direction pivots (keep more extreme)
        if pivots and pivots[-1][2] == kind:
            prev = pivots[-1]
            if (kind == "high" and price > prev[1]) or (kind == "low" and price < prev[1]):
                pivots[-1] = (idx, price, kind)
                last_price = price
                last_kind = kind
            continue

        pivots.append((idx, price, kind))
        last_price = price
        last_kind = kind

    return pivots


# ── Fibonacci levels ─────────────────────────────────────────

def fib_retracement(swing_h, swing_l, direction):
    """Return list of (ratio, price) for retracement levels."""
    rng = swing_h - swing_l
    levels = []
    for r in FIB_RET:
        if direction == "BULL":
            # Retracing down from high
            price = swing_h - rng * r
        else:
            # Retracing up from low
            price = swing_l + rng * r
        levels.append((r, price))
    return levels


def fib_extension(a, b, c):
    """Extension from C using AB range. Returns list of (ratio, price)."""
    ab = abs(b - a)
    direction = 1 if b > a else -1
    levels = []
    for r in FIB_EXT:
        price = c + direction * ab * r
        levels.append((r, price))
    return levels


# ── N-wave detection ─────────────────────────────────────────

def detect_nwaves(pivots, pair):
    """Find 4-pivot N-wave sequences (A→B→C→D).
    BULL: low→high→low→high   BEAR: high→low→high→low
    Returns list of dicts with wave info.
    """
    if len(pivots) < 4:
        return []

    pip = pip_size(pair)
    waves = []

    for i in range(len(pivots) - 3):
        a, b, c, d = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3]
        # BULL: L→H→L→H
        if a[2] == "low" and b[2] == "high" and c[2] == "low" and d[2] == "high":
            if c[1] > a[1] and d[1] > b[1]:
                ab = (b[1] - a[1]) / pip
                cd = (d[1] - c[1]) / pip
                q = cd / ab if ab else 0
                waves.append({
                    "dir": "BULL", "A": a, "B": b, "C": c, "D": d,
                    "ab_pips": round(ab, 1), "cd_pips": round(cd, 1),
                    "quality": round(q, 2),
                })
        # BEAR: H→L→H→L
        elif a[2] == "high" and b[2] == "low" and c[2] == "high" and d[2] == "low":
            if c[1] < a[1] and d[1] < b[1]:
                ab = (a[1] - b[1]) / pip
                cd = (c[1] - d[1]) / pip
                q = cd / ab if ab else 0
                waves.append({
                    "dir": "BEAR", "A": a, "B": b, "C": c, "D": d,
                    "ab_pips": round(ab, 1), "cd_pips": round(cd, 1),
                    "quality": round(q, 2),
                })

    return waves


# ── Formatting helpers ───────────────────────────────────────

def fmt_price(price, pair):
    dec = 3 if "JPY" in pair else 5
    return f"{price:.{dec}f}"


def nearest_fib_label(current, levels, pair):
    """Which fib level is closest to current price?"""
    pip = pip_size(pair)
    best = None
    best_dist = float("inf")
    for ratio, price in levels:
        dist = abs(current - price) / pip
        if dist < best_dist:
            best_dist = dist
            best = (ratio, price, dist)
    if best and best[2] < 5:
        return f"Fib {best[0]*100:.1f}% = pullback zone" if best[2] < 3 else f"near Fib {best[0]*100:.1f}%"
    return None


def format_detail(pair, tf, candles, pivots, current):
    """Full detail output for a single pair."""
    pip = pip_size(pair)
    lines = []
    lines.append(f"=== {pair} {tf} Fib/Wave ===")

    if len(pivots) < 2:
        lines.append("Insufficient pivots (not enough data)")
        return "\n".join(lines)

    # Find most recent swing high and swing low
    recent_high = None
    recent_low = None
    for p in reversed(pivots):
        if p[2] == "high" and recent_high is None:
            recent_high = p
        elif p[2] == "low" and recent_low is None:
            recent_low = p
        if recent_high and recent_low:
            break

    if not recent_high or not recent_low:
        lines.append("Swing H/L not detected")
        return "\n".join(lines)

    h_price, l_price = recent_high[1], recent_low[1]
    h_idx, l_idx = recent_high[0], recent_low[0]
    rng_pips = to_pips(h_price - l_price, pair)
    direction = "BULL" if l_idx < h_idx else "BEAR"

    lines.append(
        f"Swing: H={fmt_price(h_price, pair)}(@idx{h_idx}) "
        f"L={fmt_price(l_price, pair)}(@idx{l_idx}) "
        f"range={rng_pips}pip dir={direction}"
    )

    # Current price context
    fib_note = nearest_fib_label(
        current, fib_retracement(h_price, l_price, direction), pair
    )
    cur_str = f"Current: {fmt_price(current, pair)}"
    if fib_note:
        cur_str += f" ({fib_note})"
    lines.append(cur_str)
    lines.append("")

    # ── Fib retracement ──
    ret_levels = fib_retracement(h_price, l_price, direction)
    lines.append(f"Fib Retracement ({'H→L' if direction == 'BULL' else 'L→H'}):")
    for ratio, price in ret_levels:
        dist = to_pips(price - current, pair)
        mark = " <- near" if abs(dist) < 3 else ""
        lines.append(
            f"  {ratio*100:.1f}%: {fmt_price(price, pair)}  [{dist:+.0f}pip]{mark}"
        )
    lines.append("")

    # ── Fib extension (from most recent C point) ──
    # Find A, B, C for extension: use last 3 pivots
    if len(pivots) >= 3:
        a, b, c = pivots[-3], pivots[-2], pivots[-1]
        ext_levels = fib_extension(a[1], b[1], c[1])
        ext_dir = "BULL" if b[1] > a[1] else "BEAR"
        lines.append(
            f"Fib Extension (from C={fmt_price(c[1], pair)}, AB={ext_dir}):"
        )
        for ratio, price in ext_levels:
            dist = to_pips(price - current, pair)
            lines.append(
                f"  {ratio*100:.1f}%: {fmt_price(price, pair)}  [{dist:+.0f}pip]"
            )
        lines.append("")

    # ── N-wave ──
    waves = detect_nwaves(pivots, pair)
    if waves:
        w = waves[-1]  # most recent
        a, b, c, d = w["A"], w["B"], w["C"], w["D"]
        lines.append(
            f"N-Wave: {w['dir']} q={w['quality']} "
            f"A={fmt_price(a[1], pair)}->"
            f"B={fmt_price(b[1], pair)}->"
            f"C={fmt_price(c[1], pair)}->"
            f"D={fmt_price(d[1], pair)}"
        )
        # Re-entry zone: fib 61.8% of AB applied to C
        ab = abs(b[1] - a[1])
        if w["dir"] == "BULL":
            re_entry_lo = c[1]
            re_entry_hi = c[1] + ab * 0.382
            invalidation = a[1] - pip * 2
            tp = c[1] + ab * 1.272
        else:
            re_entry_hi = c[1]
            re_entry_lo = c[1] - ab * 0.382
            invalidation = a[1] + pip * 2
            tp = c[1] - ab * 1.272
        lines.append(
            f"  Re-entry: {fmt_price(min(re_entry_lo, re_entry_hi), pair)}"
            f"-{fmt_price(max(re_entry_lo, re_entry_hi), pair)} "
            f"(Fib 38.2-61.8% of AB)"
        )
        lines.append(
            f"  Invalidation: {fmt_price(invalidation, pair)} "
            f"({'below A' if w['dir'] == 'BULL' else 'above A'})"
        )
        lines.append(f"  TP target: {fmt_price(tp, pair)} (127.2% ext)")
    else:
        lines.append("N-Wave: not detected")

    return "\n".join(lines)


def format_summary(pair, tf, candles, pivots, current):
    """Condensed one-line summary for --all mode."""
    pip = pip_size(pair)
    if len(pivots) < 2:
        return f"{pair}: insufficient pivots"

    recent_high = None
    recent_low = None
    for p in reversed(pivots):
        if p[2] == "high" and recent_high is None:
            recent_high = p
        elif p[2] == "low" and recent_low is None:
            recent_low = p
        if recent_high and recent_low:
            break

    if not recent_high or not recent_low:
        return f"{pair}: Swing not detected"

    h_price, l_price = recent_high[1], recent_low[1]
    h_idx, l_idx = recent_high[0], recent_low[0]
    rng = to_pips(h_price - l_price, pair)
    direction = "BULL" if l_idx < h_idx else "BEAR"

    # Where is current in fib terms?
    if h_price != l_price:
        if direction == "BULL":
            fib_pos = (h_price - current) / (h_price - l_price)
        else:
            fib_pos = (current - l_price) / (h_price - l_price)
    else:
        fib_pos = 0.5

    fib_pct = f"{fib_pos*100:.0f}%"

    waves = detect_nwaves(pivots, pair)
    wave_str = ""
    if waves:
        w = waves[-1]
        wave_str = f" N={w['dir']}(q={w['quality']})"

    return (
        f"{pair}: {direction} range={rng}pip "
        f"now@Fib{fib_pct} H={fmt_price(h_price, pair)} L={fmt_price(l_price, pair)}"
        f"{wave_str}"
    )


# ── Main ─────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    all_mode = "--all" in sys.argv

    if all_mode:
        # --all [TF] [COUNT]
        tf = args[0] if len(args) >= 1 else "M5"
        count = int(args[1]) if len(args) >= 2 else 100
        pairs = ALL_PAIRS
    elif args:
        # PAIR [TF] [COUNT]
        pairs = [args[0]]
        tf = args[1] if len(args) >= 2 else "M5"
        count = int(args[2]) if len(args) >= 3 else 100
    else:
        pairs = ALL_PAIRS
        tf = "M5"
        count = 100

    cfg = load_config()
    token = cfg.get("oanda_token", "")

    for pair in pairs:
        try:
            candles = fetch_candles(pair, tf, count, token)
        except Exception as e:
            print(f"{pair}: ERROR fetching candles: {e}")
            continue

        if not candles:
            print(f"{pair}: No candle data")
            continue

        current = candles[-1]["close"]
        pivots = extract_pivots(candles, pair)

        if len(pairs) > 1:
            print(format_summary(pair, tf, candles, pivots, current))
        else:
            print(format_detail(pair, tf, candles, pivots, current))

    if len(pairs) > 1:
        print(f"\n(TF={tf} count={count})")


if __name__ == "__main__":
    main()
