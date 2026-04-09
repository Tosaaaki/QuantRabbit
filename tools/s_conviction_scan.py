#!/usr/bin/env python3
"""
S-Conviction Scanner — detects TF × indicator patterns that indicate S-conviction setups.

The trader sees individual indicators. This script sees PATTERNS across timeframes.
Each recipe is a proven combination that, when all conditions fire simultaneously, = S-conviction.

Output: one-line summary per detected S-candidate, or "(no S-candidates)" if none found.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def load_macro() -> dict:
    """Load currency strength from macro_view cache."""
    try:
        out = {}
        cache = ROOT / "logs" / "news_cache.json"
        if cache.exists():
            data = json.loads(cache.read_text())
            out = data.get("currency_strength", {})
        return out
    except Exception:
        return {}


def g(d: dict, key: str, default=0.0) -> float:
    return float(d.get(key, default))


def scan_pair(pair: str, tfs: dict, cs: dict) -> list[str]:
    """Scan a single pair for S-conviction recipes. Returns list of match descriptions."""
    h4 = tfs.get("H4", {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    m1 = tfs.get("M1", {})

    if not h1 or not m5:
        return []

    matches = []

    # ── Recipe 1: Multi-TF Extreme Counter ──
    # H4 extreme + H1 extreme + M5 opposite extreme = counter-trade S
    # Example: EUR_JPY right now — H4+H1 StRSI=1.0, M5 StRSI=0.0
    h4_ob = g(h4, "stoch_rsi") >= 0.95
    h4_os = g(h4, "stoch_rsi") <= 0.05
    h1_ob = g(h1, "stoch_rsi") >= 0.95
    h1_os = g(h1, "stoch_rsi") <= 0.05
    m5_ob = g(m5, "stoch_rsi") >= 0.95
    m5_os = g(m5, "stoch_rsi") <= 0.05

    if h4_ob and h1_ob and m5_os:
        extra = []
        if g(h4, "div_rsi_score") > 0 or g(h4, "div_macd_score") > 0:
            extra.append("H4 div")
        if g(h1, "div_rsi_score") > 0 or g(h1, "div_macd_score") > 0:
            extra.append("H1 div")
        if abs(g(h1, "cci")) >= 200:
            extra.append(f"H1 CCI={g(h1, 'cci'):.0f}")
        if g(h1, "rsi") >= 75:
            extra.append(f"H1 RSI={g(h1, 'rsi'):.0f}")
        div_str = f" + {'+'.join(extra)}" if extra else ""
        matches.append(f"🎯 {pair} SHORT Counter-S: H4+H1 StRSI=1.0 + M5 StRSI=0.0{div_str}")

    if h4_os and h1_os and m5_ob:
        extra = []
        if g(h4, "div_rsi_score") > 0 or g(h4, "div_macd_score") > 0:
            extra.append("H4 div")
        if g(h1, "div_rsi_score") > 0 or g(h1, "div_macd_score") > 0:
            extra.append("H1 div")
        if abs(g(h1, "cci")) >= 200:
            extra.append(f"H1 CCI={g(h1, 'cci'):.0f}")
        div_str = f" + {'+'.join(extra)}" if extra else ""
        matches.append(f"🎯 {pair} LONG Counter-S: H4+H1 StRSI=0.0 + M5 StRSI=1.0{div_str}")

    # ── Recipe 2: H1 Trend + M5 Dip (Confirmed Pattern) ──
    # H1 ADX>25 + DI aligned + M5 StRSI extreme = dip buy/sell in trend
    h1_adx = g(h1, "adx")
    h1_dip = g(h1, "plus_di")
    h1_dim = g(h1, "minus_di")

    if h1_adx >= 25 and h1_dip > h1_dim and m5_os:
        extra = []
        if g(h4, "plus_di") > g(h4, "minus_di"):
            extra.append("H4 aligned")
        if abs(g(m5, "cci")) >= 100:
            extra.append(f"M5 CCI={g(m5, 'cci'):.0f}")
        al_str = f" + {'+'.join(extra)}" if extra else ""
        matches.append(f"🎯 {pair} LONG Trend-Dip-S: H1 ADX={h1_adx:.0f} BULL + M5 StRSI=0.0{al_str}")

    if h1_adx >= 25 and h1_dim > h1_dip and m5_ob:
        extra = []
        if g(h4, "minus_di") > g(h4, "plus_di"):
            extra.append("H4 aligned")
        if abs(g(m5, "cci")) >= 100:
            extra.append(f"M5 CCI={g(m5, 'cci'):.0f}")
        al_str = f" + {'+'.join(extra)}" if extra else ""
        matches.append(f"🎯 {pair} SHORT Trend-Dip-S: H1 ADX={h1_adx:.0f} BEAR + M5 StRSI=1.0{al_str}")

    # ── Recipe 3: Multi-TF Divergence + Extreme ──
    # H4 div + H1 div + (H1 or M5 extreme) = reversal S
    h4_div = g(h4, "div_rsi_score") > 0 or g(h4, "div_macd_score") > 0
    h1_div = g(h1, "div_rsi_score") > 0 or g(h1, "div_macd_score") > 0
    h1_extreme = g(h1, "stoch_rsi") >= 0.95 or g(h1, "stoch_rsi") <= 0.05 or abs(g(h1, "cci")) >= 200

    if h4_div and h1_div and h1_extreme:
        # Determine direction from divergence kind
        h4_bear_div = g(h4, "div_rsi_kind") in [2, 4] or g(h4, "div_macd_kind") in [2, 4]
        h4_bull_div = g(h4, "div_rsi_kind") in [1, 3] or g(h4, "div_macd_kind") in [1, 3]
        if h4_bear_div and h1_ob:
            matches.append(f"🎯 {pair} SHORT Div-Reversal-S: H4+H1 bear div + H1 extreme OB")
        elif h4_bull_div and h1_os:
            matches.append(f"🎯 {pair} LONG Div-Reversal-S: H4+H1 bull div + H1 extreme OS")

    # ── Recipe 4: Currency Strength Gap + MTF Alignment ──
    # CS gap > 0.5 between base and quote + H4+H1+M5 aligned = momentum S
    base, quote = pair[:3], pair[4:]
    base_cs = cs.get(base, 0)
    quote_cs = cs.get(quote, 0)
    cs_gap = base_cs - quote_cs

    if abs(cs_gap) >= 0.5:
        h4_bull = g(h4, "plus_di") > g(h4, "minus_di")
        h1_bull = g(h1, "plus_di") > g(h1, "minus_di")
        m5_bull = g(m5, "plus_di") > g(m5, "minus_di")

        if cs_gap >= 0.5 and h4_bull and h1_bull and m5_bull:
            matches.append(
                f"🎯 {pair} LONG Momentum-S: CS {base}({base_cs:+.2f}) vs {quote}({quote_cs:+.2f}) gap={cs_gap:.2f} + H4+H1+M5 BULL"
            )
        elif cs_gap <= -0.5 and not h4_bull and not h1_bull and not m5_bull:
            matches.append(
                f"🎯 {pair} SHORT Momentum-S: CS {base}({base_cs:+.2f}) vs {quote}({quote_cs:+.2f}) gap={cs_gap:.2f} + H4+H1+M5 BEAR"
            )

    # ── Recipe 5: Structural Confluence + Timing ──
    # M5 at BB lower/upper + M5 extreme + H1 trend aligned = structural entry S
    m5_close = g(m5, "close")
    m5_bb_lower = g(m5, "bb_lower")
    m5_bb_upper = g(m5, "bb_upper")
    m5_bbw = g(m5, "bbw", 999)

    # At BB lower + oversold + H1 bull = bounce S
    if m5_close > 0 and m5_bb_lower > 0:
        at_bb_lower = (m5_close - m5_bb_lower) / max(m5_close, 0.001) < 0.0003  # within 3 pips for JPY pairs
        at_bb_upper = (m5_bb_upper - m5_close) / max(m5_close, 0.001) < 0.0003

        if at_bb_lower and m5_os and h1_dip > h1_dim:
            ichi = g(m5, "ichimoku_cloud_pos")
            ichi_str = f" + Ichi={ichi:.0f}pip above cloud" if ichi > 0 else ""
            matches.append(f"🎯 {pair} LONG Structural-S: M5 at BB lower + StRSI=0.0 + H1 BULL{ichi_str}")

        if at_bb_upper and m5_ob and h1_dim > h1_dip:
            matches.append(f"🎯 {pair} SHORT Structural-S: M5 at BB upper + StRSI=1.0 + H1 BEAR")

    # ── Recipe 6: Squeeze Breakout + Trend ──
    # M5 squeeze + H1 strong trend (ADX>30) + M1 directional = breakout S
    m5_squeeze = m5_bbw < 0.002
    h1_strong = h1_adx >= 30
    m1_dir_bull = g(m1, "plus_di") > g(m1, "minus_di") + 10  # clear directional
    m1_dir_bear = g(m1, "minus_di") > g(m1, "plus_di") + 10

    if m5_squeeze and h1_strong:
        if h1_dip > h1_dim and m1_dir_bull:
            matches.append(f"🎯 {pair} LONG Squeeze-S: M5 squeeze + H1 ADX={h1_adx:.0f} BULL + M1 buyers confirmed")
        elif h1_dim > h1_dip and m1_dir_bear:
            matches.append(f"🎯 {pair} SHORT Squeeze-S: M5 squeeze + H1 ADX={h1_adx:.0f} BEAR + M1 sellers confirmed")

    return matches


def deduplicate(matches: list[str]) -> list[str]:
    """Keep only the strongest recipe per pair+direction."""
    import re as _re
    best = {}
    for line in matches:
        m = _re.match(r"🎯 (\w+) (LONG|SHORT) (\S+): (.+)", line)
        if not m:
            continue
        key = (m.group(1), m.group(2))  # (pair, direction)
        if key not in best:
            best[key] = line
        else:
            # More detail = more evidence = stronger
            existing_detail = best[key].split(": ", 1)[-1] if ": " in best[key] else ""
            new_detail = m.group(4)
            if len(new_detail) > len(existing_detail):
                best[key] = line
    return list(best.values())


def main():
    # Load currency strength
    cs = {}
    try:
        mv_out = ROOT / "logs" / "news_cache.json"
        if mv_out.exists():
            data = json.loads(mv_out.read_text())
            cs = data.get("currency_strength", {})
    except Exception:
        pass

    # Fallback: parse macro_view output if no cache
    if not cs:
        try:
            import subprocess
            venv = str(ROOT / ".venv" / "bin" / "python")
            r = subprocess.run(
                [venv, "tools/macro_view.py"], capture_output=True, text=True, timeout=10, cwd=str(ROOT)
            )
            # Parse "EUR(+0.57) > AUD(+0.33) ..." format
            for token in r.stdout.split():
                if "(" in token and ")" in token:
                    ccy = token.split("(")[0]
                    val = token.split("(")[1].rstrip(")")
                    try:
                        cs[ccy] = float(val)
                    except ValueError:
                        pass
        except Exception:
            pass

    all_matches = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        if not tfs:
            continue
        matches = scan_pair(pair, tfs, cs)
        all_matches.extend(matches)

    if all_matches:
        # Deduplicate: same pair + same direction → keep strongest recipe
        deduped = deduplicate(all_matches)
        for m in deduped:
            # Append current price for outcome tracking
            pair_m = m.split()[1] if len(m.split()) > 1 else ""
            tfs = load_technicals(pair_m)
            m5 = tfs.get("M5", {})
            price = m5.get("close", 0)
            price_str = f" @{price:.5f}" if price else ""
            print(f"{m}{price_str}")
        raw_count = len(all_matches)
        dup_note = f" (raw {raw_count}, deduplicated)" if raw_count != len(deduped) else ""
        print(f"\n→ {len(deduped)} S-candidate(s) found{dup_note}.")
    else:
        print("(no S-candidates detected across 7 pairs)")


if __name__ == "__main__":
    main()
