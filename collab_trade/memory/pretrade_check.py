"""
QuantRabbit Trading Memory — Pre-Trade Check
Cross-references 3 memory layers (risk) immediately before entry + evaluates current setup quality (conviction)

Output:
  - RISK: HIGH/MEDIUM/LOW — risk warnings from historical data (backward-looking)
  - CONFIDENCE: S/A/B/C — quality of current setup (forward-looking) → basis for sizing decisions
  - RECOMMENDED SIZE: unit count based on conviction

Usage:
  python3 pretrade_check.py GBP_USD SHORT [--adx 38] [--headline "Iran"]
"""
from __future__ import annotations

import sys
import json
from datetime import date
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

def trade_stats(conn, pair: str, direction: str) -> dict:
    """Win rate and average P&L by pair x direction"""
    all_trades = fetchall_dict(conn,
        "SELECT pl, regime, had_sl, entry_type, lesson FROM trades WHERE pair = ? AND direction = ? AND pl IS NOT NULL",
        (pair, direction))

    if not all_trades:
        return {"count": 0}

    wins = [t for t in all_trades if t["pl"] > 0]
    losses = [t for t in all_trades if t["pl"] < 0]
    pls = [t["pl"] for t in all_trades]

    return {
        "count": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(all_trades) if all_trades else 0,
        "avg_pl": sum(pls) / len(pls),
        "total_pl": sum(pls),
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
    wins = [p for p in pls if p > 0]
    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_pl": sum(pls) / len(pls),
    }


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


def latest_user_call(conn, pair: str = None) -> dict | None:
    """Most recent user call"""
    if pair:
        return fetchone_dict(conn,
            "SELECT * FROM user_calls WHERE pair = ? ORDER BY id DESC LIMIT 1", (pair,))
    return fetchone_dict(conn,
        "SELECT * FROM user_calls ORDER BY id DESC LIMIT 1")


# --- Vector Layer ---

def similar_trades_narrative(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve narratives of similar situations via vector search"""
    try:
        from recall import hybrid_search
        return hybrid_search(query, top_k=top_k)
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
        # H1+M5 aligned but H4 opposite → mid wave (early trend reversal)
        # M5 only → small wave
        if h4_aligned and h1_aligned:
            wave = "big"
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
        # Mid wave: H1+M5 are important. H4 is reference
        if h1_aligned and m5_aligned:
            quality_score += 4 if h4_aligned else 3
            h4_note = " + H4 aligned" if h4_aligned else ""
            details.append(f"[mid wave] H1+M5 aligned{h4_note} +{4 if h4_aligned else 3}")
        elif m5_aligned:
            quality_score += 2
            details.append(f"[mid wave] M5 aligned (H1 opposite) +2")
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
    if quality_score >= 8:
        grade = "S"
    elif quality_score >= 6:
        grade = "A"
    elif quality_score >= 4:
        grade = "B"
    else:
        grade = "C"

    # Sizing recommendation — conviction determines size, not wave size.
    # Even on a small wave, if conviction is S, size up. Compensate for smaller profit target with size.
    sizing = {
        "S": "8000-10000u (iron-clad. size up)",
        "A": "5000-8000u (high conviction. trade it properly)",
        "B": "2000-3000u (conservative)",
        "C": "1000u or less (weak basis. pass recommended)",
    }

    return {
        "grade": grade,
        "quality_score": quality_score,
        "wave": wave,
        "details": details,
        "sizing": sizing[grade],
        "mtf_aligned": aligned_count,
    }


# --- Risk Assessment (backward-looking: risk from historical data) ---

def assess_risk(
    pair: str,
    direction: str,
    adx: float = None,
    headline: str = None,
    regime: str = None,
    wave: str = "auto",
) -> dict:
    """Overall risk + setup quality assessment"""
    conn = get_conn()
    warnings = []
    risk_score = 0  # 0-10

    # 1. Trade statistics
    stats = trade_stats(conn, pair, direction)
    if stats["count"] > 0:
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

    # 4. User call
    uc = latest_user_call(conn, pair)
    if uc:
        user_dir = uc.get("direction")
        if user_dir and user_dir != ("UP" if direction == "LONG" else "DOWN"):
            uc_stats = user_call_stats(conn, pair)
            if uc_stats["count"] > 0:
                warnings.append(
                    f"⚠️ Latest user call: \"{uc['call_text']}\" ({user_dir}) "
                    f"= opposite to {direction} (user accuracy: {uc_stats['accuracy']:.0%})"
                )
                risk_score += 2

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
    setup = assess_setup_quality(pair, direction, wave=wave)

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
    query_text = f"{pair} {direction}"
    if headline:
        query_text += f" {headline}"
    if adx:
        query_text += f" ADX={adx}"
    narratives = similar_trades_narrative(query_text, top_k=2)
    if narratives:
        result["similar_memories"] = [
            {"date": n.get("session_date", "?"), "content": n.get("content", "")[:200]}
            for n in narratives
        ]

    # 8. Log this check result to pretrade_outcomes (daily_review will fill in pl later)
    _log_pretrade(conn, pair, direction, level, risk_score, warnings)

    return result


def _past_pretrade_outcomes(conn, pair: str, direction: str, level: str) -> list[dict]:
    """Outcomes when the same pair/direction/level was entered in the past"""
    return fetchall_dict(conn,
        """SELECT session_date, pl, lesson_from_review, thesis
           FROM pretrade_outcomes
           WHERE pair = ? AND direction = ? AND pretrade_level = ? AND pl IS NOT NULL
           ORDER BY id DESC LIMIT 5""",
        (pair, direction, level))


def _log_pretrade(conn, pair: str, direction: str, level: str, score: int, warnings: list):
    """Log this check result to pretrade_outcomes"""
    try:
        conn.execute(
            """INSERT INTO pretrade_outcomes
               (session_date, pair, direction, pretrade_level, pretrade_score, pretrade_warnings)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(date.today()), pair, direction, level, score,
             json.dumps(warnings, ensure_ascii=False))
        )
    except Exception:
        pass  # ignore if table does not yet exist


def format_check(result: dict) -> str:
    """Format check result for readable display"""
    lines = []
    level = result["risk_level"]
    setup = result.get("setup_quality", {})
    grade = setup.get("grade", "?")
    q_score = setup.get("quality_score", 0)

    grade_icon = {"S": "🔥", "A": "✅", "B": "⚠️", "C": "❌"}.get(grade, "?")
    risk_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[level]

    # Main header: conviction is most important (determines sizing)
    wave_label = {"big": "big wave", "mid": "mid wave", "small": "small wave"}.get(setup.get("wave", "?"), "?")
    lines.append(f"{grade_icon} Conviction: {grade} (score={q_score}) | Risk: {level} (score={result['risk_score']}) | Wave: {wave_label}")
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
        lines.append(f"📊 Historical record: {stats['wins']}W {stats['losses']}L (win rate {stats['win_rate']:.0%}) avg P&L={stats['avg_pl']:+,.0f} JPY total={stats['total_pl']:+,.0f} JPY")

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
        print("Usage: python3 pretrade_check.py <PAIR> <LONG|SHORT> [--wave big|mid|small] [--adx N] [--headline TEXT] [--regime TYPE]")
        print("Example: python3 pretrade_check.py GBP_USD SHORT --wave big")
        print("  --wave: big=H4/H1 swing, mid=M5 trade, small=M1 scalp, auto=auto-detect (default)")
        sys.exit(1)

    pair = sys.argv[1]
    direction = sys.argv[2]

    adx = None
    headline = None
    regime = None
    wave = "auto"
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--adx" and i + 1 < len(sys.argv):
            adx = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--headline" and i + 1 < len(sys.argv):
            headline = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--regime" and i + 1 < len(sys.argv):
            regime = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--wave" and i + 1 < len(sys.argv):
            wave = sys.argv[i + 1]; i += 2
        else:
            i += 1

    result = assess_risk(pair, direction, adx=adx, headline=headline, regime=regime, wave=wave)
    print(format_check(result))
