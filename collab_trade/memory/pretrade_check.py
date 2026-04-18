"""
QuantRabbit Trading Memory — Pre-Trade Check
Cross-references 3 memory layers (risk) immediately before entry + evaluates current setup quality (conviction)

Output:
  - RISK: HIGH/MEDIUM/LOW — risk warnings from historical data (backward-looking)
  - CONFIDENCE: S/A/B/C — quality of current setup (forward-looking) → basis for sizing decisions
  - RECOMMENDED SIZE: unit count based on conviction
  - HISTORICAL PAYOFF: realized expectancy / avg win-loss / break-even WR

Usage:
  python3 pretrade_check.py GBP_USD SHORT [--adx 38] [--headline "Iran"]
"""
from __future__ import annotations

import sys
import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from schema import get_conn, init_db, fetchall_dict, fetchone_val, fetchone_dict, serialize_f32

ROOT = Path(__file__).resolve().parent.parent.parent  # memory/ → collab_trade/ → QuantRabbit/
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"), "EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"), "EUR_JPY": ("EUR", "JPY"), "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}

# --- SQL Layer: Statistics from structured data ---

def payoff_metrics(pls: list[float]) -> dict:
    """Realized payoff quality from closed-trade P&L.

    R:R alone is not enough. What matters is whether the realized win rate and
    realized payout shape produce positive expectancy after repeated trades.
    """
    if not pls:
        return {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_pl": 0.0,
            "total_pl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "rr_ratio": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "break_even_win_rate": None,
        }

    wins = [pl for pl in pls if pl > 0]
    losses = [pl for pl in pls if pl < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss < 0:
        rr_ratio = avg_win / abs(avg_loss)
    elif avg_win > 0 and not losses:
        rr_ratio = float("inf")
    else:
        rr_ratio = 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    break_even_win_rate = None
    if avg_win > 0 and avg_loss < 0:
        break_even_win_rate = abs(avg_loss) / (avg_win + abs(avg_loss))
    elif avg_win > 0 and not losses:
        break_even_win_rate = 0.0

    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float("inf")

    return {
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pls),
        "avg_pl": sum(pls) / len(pls),
        "total_pl": sum(pls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio,
        "profit_factor": profit_factor,
        "expectancy": sum(pls) / len(pls),
        "break_even_win_rate": break_even_win_rate,
    }

def trade_stats(conn, pair: str, direction: str) -> dict:
    """Win rate and average P&L by pair x direction"""
    all_trades = fetchall_dict(conn,
        "SELECT pl, regime, had_sl, entry_type, lesson FROM trades WHERE pair = ? AND direction = ? AND pl IS NOT NULL",
        (pair, direction))

    if not all_trades:
        return {"count": 0}

    wins = [t for t in all_trades if t["pl"] > 0]
    pls = [t["pl"] for t in all_trades]
    metrics = payoff_metrics(pls)

    return {
        "count": len(all_trades),
        **metrics,
        "worst": min(pls),
        "best": max(pls),
        "no_sl_count": sum(1 for t in all_trades if t["had_sl"] == 0),
        "lessons": [t["lesson"] for t in all_trades if t["lesson"]],
    }


def regime_stats(conn, pair: str, direction: str, regime: str) -> dict:
    """Win rate under a specific regime"""
    trades = fetchall_dict(conn,
        "SELECT pl FROM trades WHERE pair = ? AND direction = ? AND regime = ? AND pl IS NOT NULL",
        (pair, direction, regime))

    if not trades:
        return {"count": 0}

    pls = [t["pl"] for t in trades]
    return {"count": len(trades), **payoff_metrics(pls)}


def headline_risk(conn, pair: str) -> list[dict]:
    """Past market events related to this pair"""
    return fetchall_dict(conn,
        """SELECT event_type, headline, spike_pips, spike_direction, impact, session_date
           FROM market_events
           WHERE pairs_affected = ? OR pairs_affected IS NULL
           ORDER BY spike_pips DESC""",
        (pair,))


def active_headlines_history(conn, headline_keyword: str) -> list[dict]:
    """Historical trade results when a specific headline was active"""
    return fetchall_dict(conn,
        """SELECT pair, direction, pl, regime, lesson
           FROM trades
           WHERE active_headlines LIKE ?
           AND pl IS NOT NULL""",
        (f"%{headline_keyword}%",))


# --- User Call Layer ---

def user_call_stats(conn, pair: str = None, direction: str = None) -> dict:
    """Accuracy rate of user's market reads"""
    where = ["outcome IS NOT NULL"]
    params = []
    if pair:
        where.append("pair = ?")
        params.append(pair)
    if direction:
        where.append("direction = ?")
        params.append(direction)

    calls = fetchall_dict(conn,
        f"SELECT outcome, pl_after_30m, conditions, call_text FROM user_calls WHERE {' AND '.join(where)}",
        tuple(params))

    if not calls:
        return {"count": 0}

    correct = [c for c in calls if c["outcome"] == "correct"]
    incorrect = [c for c in calls if c["outcome"] == "incorrect"]

    return {
        "count": len(calls),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": len(correct) / len(calls) if calls else 0,
        "recent_calls": [c["call_text"] for c in calls[-3:]],
    }


def latest_user_call(conn, pair: str = None, max_age_days: int = 3) -> dict | None:
    """Most recent user call within max_age_days. Market conditions change — stale calls are noise."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    if pair:
        return fetchone_dict(conn,
            "SELECT * FROM user_calls WHERE pair = ? AND session_date >= ? ORDER BY id DESC LIMIT 1",
            (pair, cutoff))
    return fetchone_dict(conn,
        "SELECT * FROM user_calls WHERE session_date >= ? ORDER BY id DESC LIMIT 1",
        (cutoff,))


# --- Vector Layer ---

def similar_trades_narrative(query: str, pair: str | None = None, direction: str | None = None, top_k: int = 3) -> list[dict]:
    """Retrieve narratives of similar situations via vector search"""
    try:
        from recall import hybrid_search
        hits = hybrid_search(query, top_k=max(top_k * 4, 6), pair=pair, direction=direction)
        preferred = []
        fallback = []
        for hit in hits:
            if hit.get("source_file") == "state.md" or hit.get("chunk_type") not in {"trade", "lesson"}:
                fallback.append(hit)
            else:
                preferred.append(hit)
        chosen = preferred[:top_k]
        if len(chosen) < top_k:
            chosen.extend(fallback[:top_k - len(chosen)])
        return chosen
    except Exception:
        return []


# --- Setup Quality Assessment (forward-looking: quality of current setup) ---

def _load_technicals(pair: str) -> dict:
    """Read technical data from logs/technicals_{PAIR}.json"""
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text()).get("timeframes", {})
    except Exception:
        return {}


def _calc_currency_strength() -> dict[str, float]:
    """Calculate per-currency scores from H1 ADX x DI direction"""
    scores: dict[str, list[float]] = {c: [] for c in ["USD", "EUR", "GBP", "AUD", "JPY"]}
    for pair in PAIRS:
        tfs = _load_technicals(pair)
        h1 = tfs.get("H1", {})
        if not h1:
            continue
        adx = h1.get("adx", 0)
        di_plus = h1.get("plus_di", 0)
        di_minus = h1.get("minus_di", 0)
        base, quote = PAIR_CURRENCIES[pair]
        direction = (di_plus - di_minus) / max(di_plus + di_minus, 1)
        weight = min(adx / 30, 1.5)
        signal = direction * weight
        scores[base].append(signal)
        scores[quote].append(-signal)
    return {ccy: sum(vals) / len(vals) if vals else 0 for ccy, vals in scores.items()}


def _get_current_spread(pair: str) -> float | None:
    """Fetch current spread in pips from OANDA pricing API"""
    try:
        cfg = {}
        for line in open(ROOT / "config" / "env.toml"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/pricing?instruments={pair}"
        import urllib.request
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        p = data["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        pip_factor = 100 if "JPY" in pair else 10000
        return (ask - bid) * pip_factor
    except Exception:
        return None


GRADE_ORDER = ["C", "B", "A", "S"]
GRADE_TO_UNITS = {
    "S": "8000-10000u (full pressure. strongest deployment)",
    "A": "5000-8000u (core size. real edge)",
    "B": "2000-3000u (scout / limited deployment)",
    "C": "1000u or less (weak basis — data suggests caution, you decide)",
}


def _score_to_grade(score: int, thresholds: tuple[tuple[int, str], ...]) -> str:
    for cutoff, grade in thresholds:
        if score >= cutoff:
            return grade
    return thresholds[-1][1]


def _cap_grade(grade: str, cap: str) -> str:
    return GRADE_ORDER[min(GRADE_ORDER.index(grade), GRADE_ORDER.index(cap))]


def _grade_icon(grade: str) -> str:
    return {"S": "🔥", "A": "✅", "B": "⚠️", "C": "❌"}.get(grade, "?")


def assess_setup_quality(pair: str, direction: str, wave: str = "auto") -> dict:
    """
    Score current setup quality 0-12 → output S/A/B/C conviction grade

    wave: "big" (H4/H1 swing), "mid" (M5 trade), "small" (M1 scalp), "auto" (auto-detect)

    Evaluation axes (weights vary by wave size):
    1. MTF direction alignment (0-4 pts): TF agreement based on wave size
    2. ADX trend strength (0-2 pts): ADX strength of the reference TF
    3. Macro currency strength alignment (0-2 pts): does currency pair strength direction agree?
    4. Technical confluence (0-2 pts): divergence, StochRSI extremes, BB position
    5. Wave position penalty (-2 to +1 pts): penalty for same-direction entry at H4 extremes
    6. Spread penalty (-2 to 0 pts): penalty if spread is too large relative to target
    """
    tfs = _load_technicals(pair)
    h4 = tfs.get("H4", {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})

    quality_score = 0
    details = []
    is_long = direction == "LONG"

    # --- TF alignment helper ---
    def tf_aligned(tf_data: dict) -> bool:
        if not tf_data:
            return False
        di_plus = tf_data.get("plus_di", 0)
        di_minus = tf_data.get("minus_di", 0)
        return (di_plus > di_minus) if is_long else (di_minus > di_plus)

    h4_aligned = tf_aligned(h4)
    h1_aligned = tf_aligned(h1)
    m5_aligned = tf_aligned(m5)

    # --- Auto-detect wave size ---
    if wave == "auto":
        # H4+H1 aligned → big wave (swing)
        # H4+M5 aligned (H1 transitioning) → mid wave (H1 hasn't flipped yet but higher TF supports)
        # H1+M5 aligned but H4 opposite → mid wave (early trend reversal)
        # M5 only → small wave (rotation / scalp)
        if h4_aligned and h1_aligned:
            wave = "big"
        elif h4_aligned and m5_aligned:
            wave = "mid"  # H4 supports direction, H1 still transitioning
        elif h1_aligned and m5_aligned:
            wave = "mid"
        elif m5_aligned:
            wave = "small"
        else:
            wave = "big"  # default: evaluate against swing standard

    # --- 1. MTF Direction Agreement (0-4) --- evaluation criteria vary by wave size
    aligned_count = sum([h4_aligned, h1_aligned, m5_aligned])

    if wave == "big":
        # Big wave: H4+H1 are important. M5 is timing, not setup quality
        if h4_aligned and h1_aligned and m5_aligned:
            quality_score += 4
            details.append(f"[big wave] MTF full alignment (H4+H1+M5) +4")
        elif h4_aligned and h1_aligned:
            quality_score += 3
            details.append(f"[big wave] H4+H1 aligned +3 (M5 awaiting timing)")
        elif h1_aligned and m5_aligned:
            quality_score += 3
            details.append(f"[big wave] H1+M5 aligned +3 (may be early H4 reversal)")
        elif h1_aligned:
            quality_score += 1
            details.append(f"[big wave] H1 only aligned +1")
        else:
            details.append(f"[big wave] no upper TF alignment +0")

    elif wave == "mid":
        # Mid wave: H1+M5 or H4+M5 are important
        if h1_aligned and m5_aligned:
            quality_score += 4 if h4_aligned else 3
            h4_note = " + H4 aligned" if h4_aligned else ""
            details.append(f"[mid wave] H1+M5 aligned{h4_note} +{4 if h4_aligned else 3}")
        elif m5_aligned:
            pts = 3 if h4_aligned else 2
            h4_note = " + H4 supports" if h4_aligned else " (H1 opposite)"
            quality_score += pts
            details.append(f"[mid wave] M5 aligned{h4_note} +{pts}")
        else:
            quality_score += 1 if h1_aligned else 0
            details.append(f"[mid wave] M5 not aligned +{1 if h1_aligned else 0}")

    elif wave == "small":
        # Small wave: M5 is important. H1 is directional reference. H4 irrelevant
        if m5_aligned:
            quality_score += 3 if h1_aligned else 2
            h1_note = " + H1 aligned" if h1_aligned else " (H1 opposing - caution)"
            details.append(f"[small wave] M5 aligned{h1_note} +{3 if h1_aligned else 2}")
        else:
            details.append(f"[small wave] M5 not aligned +0 ← don't enter")

    # --- 2. ADX Trend Strength (0-2) --- ADX of the reference TF
    if wave == "big":
        ref_adx = h1.get("adx", 0) if h1 else 0
        ref_aligned = h1_aligned
        ref_label = "H1"
    elif wave == "mid":
        ref_adx = m5.get("adx", 0) if m5 else 0
        ref_aligned = m5_aligned
        ref_label = "M5"
    else:  # small
        ref_adx = m5.get("adx", 0) if m5 else 0
        ref_aligned = m5_aligned
        ref_label = "M5"

    if ref_aligned and ref_adx > 30:
        quality_score += 2
        details.append(f"{ref_label} ADX={ref_adx:.0f}(>30) strong +2")
    elif ref_aligned and ref_adx > 25:
        quality_score += 1
        details.append(f"{ref_label} ADX={ref_adx:.0f}(>25) +1")
    else:
        details.append(f"{ref_label} ADX={ref_adx:.0f} weak +0")

    # --- 3. Macro Currency Strength Alignment (0-2) ---
    try:
        ccy_strength = _calc_currency_strength()
        base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        base_str = ccy_strength.get(base, 0)
        quote_str = ccy_strength.get(quote, 0)

        if is_long:
            # LONG = base strong + quote weak is ideal
            macro_aligned = base_str > 0.1 and quote_str < -0.1
            macro_neutral = base_str > quote_str
        else:
            # SHORT = base weak + quote strong is ideal
            macro_aligned = base_str < -0.1 and quote_str > 0.1
            macro_neutral = base_str < quote_str

        if macro_aligned:
            quality_score += 2
            details.append(f"Macro aligned ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +2")
        elif macro_neutral:
            quality_score += 1
            details.append(f"Macro neutral ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +1")
        else:
            details.append(f"Macro opposing ({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +0")
    except Exception:
        details.append("Macro calculation unavailable +0")

    # --- 4. Technical Confluence (0-2) ---
    confluence = 0
    confluence_notes = []

    # Divergence in trade direction
    if h1:
        div_rsi = h1.get("div_rsi_score", 0)
        div_macd = h1.get("div_macd_score", 0)
        div_rsi_kind = h1.get("div_rsi_kind", 0)
        # Regular bullish div (kind=1) supports LONG, regular bearish (kind=-1) supports SHORT
        # Hidden bullish div (kind=2) supports LONG, hidden bearish (kind=-2) supports SHORT
        div_supports = False
        if is_long and div_rsi_kind in (1, 2) and div_rsi > 0:
            div_supports = True
        elif not is_long and div_rsi_kind in (-1, -2) and div_rsi > 0:
            div_supports = True
        if div_supports:
            confluence += 1
            confluence_notes.append(f"H1 Div confirmed (score={div_rsi:.1f})")

    # StochRSI extreme in entry direction
    if m5:
        stoch = m5.get("stoch_rsi", 0.5)
        if is_long and stoch < 0.15:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f} (extreme oversold)")
        elif not is_long and stoch > 0.85:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f} (extreme overbought)")

    # BB position
    if m5:
        bb_pos = m5.get("bb", 0.5)
        if is_long and bb_pos < 0.1:
            confluence_notes.append(f"M5 BB lower band ({bb_pos:.2f})")
        elif not is_long and bb_pos > 0.9:
            confluence_notes.append(f"M5 BB upper band ({bb_pos:.2f})")

    conf_points = min(confluence, 2)
    quality_score += conf_points
    if confluence_notes:
        details.append(f"Technical confluence: {', '.join(confluence_notes)} +{conf_points}")
    else:
        details.append("No technical confluence +0")

    # --- 5. Wave Position Penalty/Bonus (-2 to +1) ---
    if h4:
        h4_cci = h4.get("cci", 0)
        h4_rsi = h4.get("rsi", 50)

        h4_extreme_long = h4_rsi > 70 or h4_cci > 200
        h4_extreme_short = h4_rsi < 30 or h4_cci < -200

        if is_long and h4_extreme_long:
            quality_score -= 2
            details.append(f"⚠ H4 overbought LONG (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← don't enter after the move is done")
        elif not is_long and h4_extreme_short:
            quality_score -= 2
            details.append(f"⚠ H4 extreme oversold SHORT (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← don't enter after the move is done")
        elif is_long and h4_extreme_short:
            quality_score += 1
            details.append(f"H4 oversold LONG (CCI={h4_cci:.0f}) +1 ← counter-trend opportunity")
        elif not is_long and h4_extreme_long:
            quality_score += 1
            details.append(f"H4 overbought SHORT (CCI={h4_cci:.0f}) +1 ← counter-trend opportunity")
        else:
            details.append(f"H4 neutral (CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) +0")
    else:
        details.append("H4 data unavailable +0")

    # --- 6. Spread Penalty (0 to -2) --- is spread too large relative to target
    spread_pip = _get_current_spread(pair)
    if spread_pip is not None:
        # Acceptable spread varies by wave size
        # Big wave (targeting 15-30 pip): 2.0 pip spread = 6-13% → acceptable
        # Mid wave (targeting 10-15 pip): 2.0 pip spread = 13-20% → caution
        # Small wave (targeting 5-10 pip): 2.0 pip spread = 20-40% → fatal
        pip_targets = {"big": 20, "mid": 12, "small": 7}
        target = pip_targets.get(wave, 12)
        spread_ratio = spread_pip / target

        if spread_ratio > 0.30:  # spread > 30% of target profit → fatal
            quality_score -= 2
            details.append(f"⚠️ Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) -2 ← R:R broken. Pass.")
        elif spread_ratio > 0.20:  # > 20% → caution
            quality_score -= 1
            details.append(f"⚠️ Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) -1 ← reduce size")
        else:
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {target}pip target) OK")
    else:
        details.append("Spread unavailable +0")

    # --- Grade mapping ---
    quality_score = max(0, quality_score)  # floor at 0
    edge_grade = _score_to_grade(
        quality_score,
        (
            (8, "S"),
            (6, "A"),
            (4, "B"),
            (0, "C"),
        ),
    )
    allocation_grade = edge_grade

    return {
        "grade": edge_grade,  # backward-compatible alias
        "edge_grade": edge_grade,
        "allocation_grade": allocation_grade,
        "quality_score": quality_score,
        "score_max": 10,
        "wave": wave,
        "details": details,
        "sizing": GRADE_TO_UNITS[allocation_grade],
        "counter_note": None,
        "mtf_aligned": aligned_count,
    }


def assess_counter_trade(pair: str, direction: str) -> dict:
    """
    Evaluate a COUNTER-TRADE: M5 trade against H4/H1 direction.

    Counter-trades have fundamentally different evaluation axes:
    - H4 extreme is FOR (not against) — the more extreme, the better
    - MTF alignment is irrelevant (counter = by definition against upper TF)
    - M5 reversal signal is the timing trigger
    - Macro opposition is expected (counter = against macro)

    Counter-trades can have HIGH EDGE even when allocation stays smaller.
    This separates:
    - edge_grade: how strong the reversal read is
    - allocation_grade: how much size the book deserves right now
    """
    tfs = _load_technicals(pair)
    h4 = tfs.get("H4", {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})

    score = 0
    details = []
    is_long = direction == "LONG"

    # --- 1. H4 Extreme (0-3) — the core of a counter-trade ---
    if h4:
        h4_stoch = h4.get("stoch_rsi", 0.5)
        h4_cci = h4.get("cci", 0)
        h4_rsi = h4.get("rsi", 50)

        # Counter-LONG needs H4 oversold; Counter-SHORT needs H4 overbought
        if is_long:
            extreme = (h4_stoch <= 0.05) or (h4_cci < -200) or (h4_rsi < 30)
            moderate = (h4_stoch <= 0.15) or (h4_cci < -100) or (h4_rsi < 35)
        else:
            extreme = (h4_stoch >= 0.95) or (h4_cci > 200) or (h4_rsi > 70)
            moderate = (h4_stoch >= 0.85) or (h4_cci > 100) or (h4_rsi > 65)

        if extreme:
            score += 3
            details.append(f"H4 extreme (StRSI={h4_stoch:.2f} CCI={h4_cci:.0f} RSI={h4_rsi:.0f}) +3 ← strong counter zone")
        elif moderate:
            score += 1
            details.append(f"H4 moderate (StRSI={h4_stoch:.2f} CCI={h4_cci:.0f}) +1")
        else:
            details.append(f"H4 not extreme (StRSI={h4_stoch:.2f}) +0 ← no counter-trade basis")
    else:
        details.append("H4 data unavailable +0")

    # --- 2. H1 Divergence or Fatigue (0-2) — confirmation ---
    if h1:
        div_rsi = h1.get("div_rsi_score", 0)
        div_macd = h1.get("div_macd_score", 0)
        h1_cci = h1.get("cci", 0)
        has_div = div_rsi > 0 or div_macd > 0
        has_cci_extreme = (is_long and h1_cci < -200) or (not is_long and h1_cci > 200)

        if has_div and has_cci_extreme:
            score += 2
            details.append(f"H1 divergence + CCI extreme ({h1_cci:.0f}) +2 ← reversal confirmed")
        elif has_div or has_cci_extreme:
            score += 1
            details.append(f"H1 {'divergence' if has_div else f'CCI={h1_cci:.0f}'} +1")
        else:
            details.append(f"H1 no reversal signal +0")

    # --- 3. M5 Reversal Signal (0-2) — timing ---
    if m5:
        m5_stoch = m5.get("stoch_rsi", 0.5)
        m5_macd_h = m5.get("macd_hist", 0)

        # Counter-LONG: M5 was oversold and MACD turning up
        # Counter-SHORT: M5 was overbought and MACD turning down
        if is_long:
            reversal = m5_stoch < 0.2 or m5_macd_h > 0
            timing = m5_stoch < 0.3
        else:
            reversal = m5_stoch > 0.8 or m5_macd_h < 0
            timing = m5_stoch > 0.7

        if reversal and timing:
            score += 2
            details.append(f"M5 reversal confirmed (StRSI={m5_stoch:.2f} MACD_H={m5_macd_h:.5f}) +2")
        elif reversal or timing:
            score += 1
            details.append(f"M5 partial signal (StRSI={m5_stoch:.2f}) +1")
        else:
            details.append(f"M5 no reversal yet (StRSI={m5_stoch:.2f}) +0 ← wait for timing")

    # --- 4. Spread vs counter-trade target (0 to -1) ---
    spread_pip = _get_current_spread(pair)
    counter_target = 8  # counter-trades target 5-10pip, use 8 as reference
    if spread_pip is not None:
        spread_ratio = spread_pip / counter_target
        if spread_ratio > 0.25:
            score -= 1
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {counter_target}pip) -1 ← too wide for counter")
        else:
            details.append(f"Spread={spread_pip:.1f}pip ({spread_ratio:.0%} of {counter_target}pip) OK")

    # --- Edge vs allocation (counter-trades can be S/A edge, smaller deployment) ---
    score = max(0, score)
    edge_grade = _score_to_grade(
        score,
        (
            (6, "S"),
            (4, "A"),
            (2, "B"),
            (0, "C"),
        ),
    )
    allocation_grade = _cap_grade(edge_grade, "A")
    counter_note = None
    if edge_grade != allocation_grade:
        counter_note = (
            "Execution edge is stronger than deployment size because this is still a "
            "counter-trade against the upper-TF flow."
        )
    if allocation_grade == "C":
        sizing = "skip — counter conditions not met"
    else:
        sizing = GRADE_TO_UNITS[allocation_grade]

    return {
        "grade": edge_grade,  # backward-compatible alias
        "edge_grade": edge_grade,
        "allocation_grade": allocation_grade,
        "score": score,
        "score_max": 7,
        "details": details,
        "sizing": sizing,
        "counter_note": counter_note,
    }


# --- Risk Assessment (backward-looking: risk from historical data) ---

def assess_risk(
    pair: str,
    direction: str,
    adx: float = None,
    headline: str = None,
    regime: str = None,
    wave: str = "auto",
    counter: bool = False,
) -> dict:
    """Overall risk + setup quality assessment"""
    conn = get_conn()
    warnings = []
    risk_score = 0  # 0-10

    # 1. Trade statistics
    stats = trade_stats(conn, pair, direction)
    if stats["count"] > 0:
        if stats["expectancy"] < 0:
            warnings.append(
                f"⚠️ {pair} {direction} expectancy: {stats['expectancy']:+,.0f} JPY/trade "
                f"(avg win {stats['avg_win']:+,.0f} vs avg loss {stats['avg_loss']:+,.0f})"
            )
            risk_score += 2
        be_wr = stats.get("break_even_win_rate")
        if (
            be_wr is not None
            and stats["count"] >= 5
            and stats["win_rate"] + 0.05 < be_wr
        ):
            warnings.append(
                f"⚠️ Break-even WR is {be_wr:.0%}, actual WR is {stats['win_rate']:.0%} "
                f"→ payoff shape is not carrying the misses yet"
            )
            risk_score += 1
        if stats["win_rate"] < 0.5:
            warnings.append(f"⚠️ {pair} {direction} win rate: {stats['win_rate']:.0%} ({stats['wins']}/{stats['count']})")
            risk_score += 2
        if stats["worst"] < -1000:
            warnings.append(f"🚨 Historical max loss: {stats['worst']:+,.0f} JPY")
            risk_score += 3

    # 2. By regime
    if regime:
        rs = regime_stats(conn, pair, direction, regime)
        if rs["count"] > 0 and rs["win_rate"] < 0.4:
            warnings.append(f"⚠️ Win rate in {regime} regime: {rs['win_rate']:.0%} ({rs['count']} trades)")
            risk_score += 2

    # 3. Headline / event risk
    if headline:
        hl_trades = active_headlines_history(conn, headline)
        if hl_trades:
            hl_pls = [t["pl"] for t in hl_trades]
            avg = sum(hl_pls) / len(hl_pls)
            if avg < 0:
                warnings.append(f"🚨 Avg P&L during {headline} headline: {avg:+,.0f} JPY ({len(hl_trades)} trades)")
                risk_score += 3
            losses = [t for t in hl_trades if t["pl"] < -500]
            for lt in losses:
                warnings.append(f"  → {lt['pair']} {lt['direction']} {lt['pl']:+,.0f} JPY: {lt['lesson']}")

    events = headline_risk(conn, pair)
    if events:
        high_events = [e for e in events if e["impact"] == "high"]
        if high_events:
            for e in high_events[:2]:
                warnings.append(f"🚨 Historical spike: {e['headline']} → {pair} {e['spike_pips']:.0f}pip ({e['session_date']})")
                risk_score += 2

    # 4. User call (only recent — market conditions change fast)
    uc = latest_user_call(conn, pair, max_age_days=3)
    if uc:
        user_dir = uc.get("direction")
        verified = uc.get("outcome") is not None
        call_date = uc.get("session_date", "?")
        if user_dir and user_dir != ("UP" if direction == "LONG" else "DOWN"):
            uc_stats = user_call_stats(conn, pair)
            if verified and uc_stats["count"] > 0:
                warnings.append(
                    f"⚠️ User call ({call_date}): \"{uc['call_text']}\" ({user_dir}) "
                    f"= opposite to {direction} (verified {uc_stats['accuracy']:.0%}, n={uc_stats['count']})"
                )
                risk_score += 2
            else:
                # Unverified call — note it but don't add risk score
                warnings.append(
                    f"ℹ️ User call ({call_date}): \"{uc['call_text']}\" ({user_dir}) "
                    f"= opposite to {direction} (unverified — info only, no score impact)"
                )

    # 5. Historical results without SL
    if stats["count"] > 0 and stats["no_sl_count"] == stats["count"]:
        losses_no_sl = [t for t in fetchall_dict(conn,
            "SELECT pl FROM trades WHERE pair = ? AND had_sl = 0 AND pl IS NOT NULL AND pl < 0",
            (pair,))]
        if losses_no_sl:
            worst = min(t["pl"] for t in losses_no_sl)
            warnings.append(f"⚠️ Max loss without SL: {worst:+,.0f} JPY → consider setting SL")
            risk_score += 1

    # Risk level
    if risk_score >= 7:
        level = "HIGH"
    elif risk_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    # 6. pretrade_outcomes: outcomes when same conditions were entered in the past
    past_outcomes = _past_pretrade_outcomes(conn, pair, direction, level)

    # 6b. Setup Quality Assessment (forward-looking)
    if counter:
        setup = assess_counter_trade(pair, direction)
        setup["is_counter"] = True
    else:
        setup = assess_setup_quality(pair, direction, wave=wave)
        setup["is_counter"] = False

    # 6c. Pair-level history context (data, not grade override — recording.md: "you make the call")
    if stats["count"] >= 5 and not counter:
        if stats["win_rate"] < 0.40:
            # Low WR warning — data only, no grade cap
            # Stats are regime-dependent (4/9 lesson: bullish-period SHORTs inflate loss counts)
            warnings.append(
                f"⚠ PAIR HISTORY: {pair} {direction} all-time WR={stats['win_rate']:.0%} "
                f"({stats['count']} trades, total {stats['total_pl']:+,.0f}JPY). "
                f"Stats are from sample period — check if current H4 structure matches."
            )
            setup["details"].append(
                f"Low historical WR={stats['win_rate']:.0%} ({stats['count']} trades) — data point, not cap"
            )
        elif stats["win_rate"] >= 0.60:
            # Trending pair with proven edge: bonus for high-conviction
            tfs = _load_technicals(pair)
            h1 = tfs.get("H1", {})
            h1_adx = h1.get("adx", 0) if h1 else 0
            cs = _calc_currency_strength()
            base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
            cs_aligned = (
                (direction == "LONG" and cs.get(base, 0) > cs.get(quote, 0))
                or (direction == "SHORT" and cs.get(base, 0) < cs.get(quote, 0))
            )
            if h1_adx > 35 and cs_aligned:
                old_score = setup.get("quality_score", 0)
                setup["quality_score"] = old_score + 2
                setup["details"].append(
                    f"TRENDING BONUS +2: WR={stats['win_rate']:.0%}, H1 ADX={h1_adx:.0f}, macro aligned"
                )
                # Re-grade
                qs = setup["quality_score"]
                if qs >= 8:
                    setup["grade"] = "S"
                    setup["sizing"] = "8000-10000u (iron-clad. size up)"
                elif qs >= 6:
                    setup["grade"] = "A"
                    setup["sizing"] = "5000-8000u (high conviction. trade it properly)"

    # 6d. Macro regime conflict warning
    try:
        cs = _calc_currency_strength()
        base, quote = PAIR_CURRENCIES.get(pair, ("?", "?"))
        base_str = cs.get(base, 0)
        quote_str = cs.get(quote, 0)
        gap = base_str - quote_str
        if direction == "LONG" and gap < -0.3:
            warnings.append(
                f"⚠ AGAINST macro flow: {base}({base_str:+.2f}) vs {quote}({quote_str:+.2f}). "
                f"CS gap={gap:+.2f} AGAINST LONG."
            )
        elif direction == "SHORT" and gap > 0.3:
            warnings.append(
                f"⚠ AGAINST macro flow: {base}({base_str:+.2f}) vs {quote}({quote_str:+.2f}). "
                f"CS gap={gap:+.2f} AGAINST SHORT."
            )
    except Exception:
        pass

    result = {
        "pair": pair,
        "direction": direction,
        "risk_level": level,
        "risk_score": risk_score,
        "warnings": warnings,
        "trade_stats": stats if stats["count"] > 0 else None,
        "past_outcomes": past_outcomes,
        "setup_quality": setup,
    }

    # 7. Similar situations via vector search
    query_parts = [pair, direction]
    if headline:
        query_parts.append(headline)
    if adx:
        query_parts.append(f"ADX={adx}")
    if setup.get("wave"):
        query_parts.append(f"{setup['wave']} wave")
    query_parts.extend(["lesson", "failure", "success"])
    query_text = " ".join(query_parts)
    narratives = similar_trades_narrative(query_text, pair=pair, direction=direction, top_k=2)
    if narratives:
        result["similar_memories"] = [
            {"date": n.get("session_date", "?"), "content": n.get("content", "")[:200]}
            for n in narratives
        ]

    # 8. Log this check result to pretrade_outcomes (daily_review will fill in pl later)
    thesis = _build_pretrade_thesis(result, headline)
    _log_pretrade(conn, pair, direction, level, risk_score, warnings, thesis)

    return result


def _past_pretrade_outcomes(conn, pair: str, direction: str, level: str) -> list[dict]:
    """Outcomes when the same pair/direction/level was entered in the past"""
    return fetchall_dict(conn,
        """SELECT session_date, pl, lesson_from_review, thesis
           FROM pretrade_outcomes
           WHERE pair = ? AND direction = ? AND pretrade_level = ? AND pl IS NOT NULL
           ORDER BY id DESC LIMIT 5""",
        (pair, direction, level))


def _build_pretrade_thesis(result: dict, headline: str | None = None) -> str:
    setup = result.get("setup_quality") or {}
    edge = setup.get("edge_grade") or setup.get("grade") or "?"
    allocation = setup.get("allocation_grade") or edge
    wave = setup.get("wave") or "?"
    details = setup.get("details") or []
    detail_text = " | ".join(str(detail).strip() for detail in details[:2] if detail)
    detail_text = detail_text[:220]
    headline_text = f" headline={headline}" if headline else ""
    thesis = f"edge={edge} alloc={allocation} wave={wave}{headline_text}"
    if detail_text:
        thesis += f" | {detail_text}"
    return thesis[:280]


def _recent_duplicate_pretrade(
    conn,
    pair: str,
    direction: str,
    level: str,
    score: int,
    thesis: str,
    lookback_minutes: int = 15,
) -> bool:
    recent = fetchone_val(
        conn,
        f"""SELECT 1
            FROM pretrade_outcomes
            WHERE pair = ?
              AND direction = ?
              AND pretrade_level = ?
              AND pretrade_score = ?
              AND thesis = ?
              AND trade_id IS NULL
              AND pl IS NULL
              AND datetime(created_at) >= datetime('now', 'localtime', '-{lookback_minutes} minutes')
            ORDER BY id DESC
            LIMIT 1""",
        (pair, direction, level, score, thesis),
    )
    return recent is not None


def _log_pretrade(conn, pair: str, direction: str, level: str, score: int, warnings: list, thesis: str):
    """Log this check result to pretrade_outcomes"""
    try:
        if _recent_duplicate_pretrade(conn, pair, direction, level, score, thesis):
            return
        conn.execute(
            """INSERT INTO pretrade_outcomes
               (session_date, pair, direction, pretrade_level, pretrade_score, pretrade_warnings, thesis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(date.today()), pair, direction, level, score,
             json.dumps(warnings, ensure_ascii=False), thesis)
        )
    except Exception:
        pass  # ignore if table does not yet exist


def format_check(result: dict) -> str:
    """Format check result for readable display"""
    lines = []
    level = result["risk_level"]
    setup = result.get("setup_quality", {})
    grade = setup.get("grade", "?")
    is_counter = setup.get("is_counter", False)

    if is_counter:
        q_score = setup.get("score", 0)
        edge_grade = setup.get("edge_grade", grade)
        allocation_grade = setup.get("allocation_grade", edge_grade)
        score_max = setup.get("score_max", 7)

        lines.append(
            f"🔄 COUNTER-TRADE | {_grade_icon(edge_grade)} Edge: {edge_grade} "
            f"(score={q_score}/{score_max}) | Allocation: {allocation_grade} | "
            f"Risk: {level} (score={result['risk_score']})"
        )
        lines.append(
            f"   {result['pair']} {result['direction']} (M5 against H4/H1) "
            f"→ {setup.get('sizing', '?')}"
        )
        lines.append("")
        lines.append("📐 Counter-trade evaluation (inverted axes — H4 extreme = FOR):")
        for detail in setup.get("details", []):
            lines.append(f"  {detail}")
        if setup.get("counter_note"):
            lines.append(f"  Note: {setup['counter_note']}")
    else:
        q_score = setup.get("quality_score", 0)
        edge_grade = setup.get("edge_grade", grade)
        allocation_grade = setup.get("allocation_grade", edge_grade)
        score_max = setup.get("score_max", 10)

        # Main header: conviction is most important (determines sizing)
        wave_label = {"big": "big wave", "mid": "mid wave", "small": "small wave"}.get(setup.get("wave", "?"), "?")
        lines.append(
            f"{_grade_icon(edge_grade)} Edge: {edge_grade} (score={q_score}/{score_max}) "
            f"| Allocation: {allocation_grade} | Risk: {level} "
            f"(score={result['risk_score']}) | Wave: {wave_label}"
        )
        lines.append(f"   {result['pair']} {result['direction']} → {setup.get('sizing', '?')}")
        lines.append("")

        # Setup quality details
        lines.append("📐 Setup quality:")
        for detail in setup.get("details", []):
            lines.append(f"  {detail}")
    lines.append("")

    # Trade statistics
    stats = result.get("trade_stats")
    if stats:
        be_wr = stats.get("break_even_win_rate")
        pf = stats.get("profit_factor")
        be_text = f" | BE WR {be_wr:.0%}" if be_wr is not None else ""
        if pf is None:
            pf_text = ""
        elif pf == float("inf"):
            pf_text = " | PF inf"
        else:
            pf_text = f" | PF {pf:.2f}"
        lines.append(
            f"📊 Historical record: {stats['wins']}W {stats['losses']}L "
            f"(WR {stats['win_rate']:.0%}) | EV {stats['expectancy']:+,.0f} JPY/trade"
            f"{be_text}{pf_text}"
        )
        lines.append(
            f"   avg win {stats['avg_win']:+,.0f} | avg loss {stats['avg_loss']:+,.0f} "
            f"| R:R {stats['rr_ratio']:.2f} | total {stats['total_pl']:+,.0f} JPY"
        )

    # Warnings
    if result["warnings"]:
        lines.append("")
        for w in result["warnings"]:
            lines.append(w)

    # Historical lessons
    if stats and stats.get("lessons"):
        lines.append("")
        lines.append("📝 Historical lessons:")
        for lesson in stats["lessons"][:3]:
            lines.append(f"  - {lesson}")

    # Outcomes from past same-condition entries (core of feedback loop)
    past = result.get("past_outcomes", [])
    if past:
        lines.append("")
        lines.append(f"📖 Last time {result['pair']} {result['direction']} was entered at pretrade {result['risk_level']}:")
        for p in past:
            outcome = "WIN" if p['pl'] and p['pl'] > 0 else "LOSS"
            pl_str = f"{p['pl']:+,.0f} JPY" if p['pl'] else "?"
            lines.append(f"  [{p['session_date']}] {outcome} {pl_str}")
            if p.get('lesson_from_review'):
                lines.append(f"    → {p['lesson_from_review']}")

    # Similar memories
    if result.get("similar_memories"):
        lines.append("")
        lines.append("🧠 Memories of similar situations:")
        for mem in result["similar_memories"]:
            lines.append(f"  [{mem['date']}] {mem['content'][:150]}...")

    return "\n".join(lines)


# --- CLI ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 pretrade_check.py <PAIR> <LONG|SHORT> [--counter] [--wave big|mid|small] [--adx N] [--headline TEXT] [--regime TYPE]")
        print("Example: python3 pretrade_check.py GBP_USD SHORT --wave big")
        print("Example: python3 pretrade_check.py EUR_JPY SHORT --counter")
        print("  --counter: evaluate as counter-trade (M5 against H4/H1 direction)")
        print("  --wave: big=H4/H1 swing, mid=M5 trade, small=M1 scalp, auto=auto-detect (default)")
        sys.exit(1)

    pair = sys.argv[1]
    direction = sys.argv[2]

    adx = None
    headline = None
    regime = None
    wave = "auto"
    counter_mode = False
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--counter":
            counter_mode = True; i += 1
        elif sys.argv[i] == "--adx" and i + 1 < len(sys.argv):
            adx = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--headline" and i + 1 < len(sys.argv):
            headline = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--regime" and i + 1 < len(sys.argv):
            regime = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--wave" and i + 1 < len(sys.argv):
            wave = sys.argv[i + 1]; i += 2
        else:
            i += 1

    if counter_mode:
        result = assess_risk(pair, direction, adx=adx, headline=headline, regime=regime, wave=wave, counter=True)
    else:
        result = assess_risk(pair, direction, adx=adx, headline=headline, regime=regime, wave=wave)
    print(format_check(result))
