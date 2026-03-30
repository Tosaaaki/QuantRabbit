"""
QuantRabbit Trading Memory — Pre-Trade Check
エントリー直前に過去の記憶を3層照合(リスク) + 今のセットアップ品質(確度)を評価

出力:
  - RISK: HIGH/MEDIUM/LOW — 過去データからのリスク警告（後ろ向き）
  - CONFIDENCE: S/A/B/C — 今のセットアップの質（前向き） → サイジング決定の根拠
  - RECOMMENDED SIZE: 確度に基づくunit数

使い方:
  python3 pretrade_check.py GBP_USD SHORT [--adx 38] [--headline "Iran"]
"""
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

# --- SQL Layer: 構造化データからの統計 ---

def trade_stats(conn, pair: str, direction: str) -> dict:
    """ペア×方向の勝率・平均PL"""
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
    """特定レジームでの勝率"""
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
    """このペアに関連する過去のマーケットイベント"""
    return fetchall_dict(conn,
        """SELECT event_type, headline, spike_pips, spike_direction, impact, session_date
           FROM market_events
           WHERE pairs_affected = ? OR pairs_affected IS NULL
           ORDER BY spike_pips DESC""",
        (pair,))


def active_headlines_history(conn, headline_keyword: str) -> list[dict]:
    """特定ヘッドラインが出た時の過去のトレード結果"""
    return fetchall_dict(conn,
        """SELECT pair, direction, pl, regime, lesson
           FROM trades
           WHERE active_headlines LIKE ?
           AND pl IS NOT NULL""",
        (f"%{headline_keyword}%",))


# --- User Call Layer ---

def user_call_stats(conn, pair: str = None, direction: str = None) -> dict:
    """ユーザーの相場読みの的中率"""
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
    """直近のユーザーコール"""
    if pair:
        return fetchone_dict(conn,
            "SELECT * FROM user_calls WHERE pair = ? ORDER BY id DESC LIMIT 1", (pair,))
    return fetchone_dict(conn,
        "SELECT * FROM user_calls ORDER BY id DESC LIMIT 1")


# --- Vector Layer ---

def similar_trades_narrative(query: str, top_k: int = 3) -> list[dict]:
    """ベクトル検索で類似状況のナラティブを引く"""
    try:
        from recall import hybrid_search
        return hybrid_search(query, top_k=top_k)
    except Exception:
        return []


# --- Setup Quality Assessment (前向き: 今のセットアップの質) ---

def _load_technicals(pair: str) -> dict:
    """logs/technicals_{PAIR}.json からテクニカルデータを読む"""
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text()).get("timeframes", {})
    except Exception:
        return {}


def _calc_currency_strength() -> dict[str, float]:
    """H1のADX×DI方向から通貨別スコアを算出"""
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
    """OANDA pricing APIから現在のスプレッドを取得(pip単位)"""
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
    今のセットアップの質を0-12で評価 → S/A/B/C確度を出す

    wave: "big" (H4/H1スイング), "mid" (M5トレード), "small" (M1スキャルプ), "auto" (自動判定)

    評価軸（波の大きさで重みが変わる）:
    1. MTF方向一致 (0-4点): 波の大きさに応じたTFの一致
    2. ADXトレンド強度 (0-2点): 注目TFのADXの強さ
    3. マクロ通貨強弱一致 (0-2点): 通貨ペアの強弱方向と一致してるか
    4. テクニカル複合確認 (0-2点): ダイバージェンス、StochRSI極限、BB位置
    5. 波の位置ペナルティ (-2〜+1点): H4極端で同方向エントリーはペナルティ
    6. スプレッドペナルティ (-2〜0点): 狙いに対してスプが大きすぎたらペナルティ
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
        # H4+H1一致 → 大波（スイング）
        # H1+M5一致だがH4逆 → 中波（トレンド転換の初動）
        # M5のみ → 小波
        if h4_aligned and h1_aligned:
            wave = "big"
        elif h1_aligned and m5_aligned:
            wave = "mid"
        elif m5_aligned:
            wave = "small"
        else:
            wave = "big"  # デフォルトはスイング基準で評価

    # --- 1. MTF Direction Agreement (0-4) --- 波の大きさで評価基準が変わる
    aligned_count = sum([h4_aligned, h1_aligned, m5_aligned])

    if wave == "big":
        # 大波: H4+H1が重要。M5はタイミングであってセットアップの質ではない
        if h4_aligned and h1_aligned and m5_aligned:
            quality_score += 4
            details.append(f"[大波] MTF全一致(H4+H1+M5) +4")
        elif h4_aligned and h1_aligned:
            quality_score += 3
            details.append(f"[大波] H4+H1一致 +3 (M5はタイミング待ち)")
        elif h1_aligned and m5_aligned:
            quality_score += 3
            details.append(f"[大波] H1+M5一致 +3 (H4転換の初動かも)")
        elif h1_aligned:
            quality_score += 1
            details.append(f"[大波] H1のみ一致 +1")
        else:
            details.append(f"[大波] 上位TF一致なし +0")

    elif wave == "mid":
        # 中波: H1+M5が重要。H4は参考
        if h1_aligned and m5_aligned:
            quality_score += 4 if h4_aligned else 3
            h4_note = " + H4も一致" if h4_aligned else ""
            details.append(f"[中波] H1+M5一致{h4_note} +{4 if h4_aligned else 3}")
        elif m5_aligned:
            quality_score += 2
            details.append(f"[中波] M5一致(H1逆) +2")
        else:
            quality_score += 1 if h1_aligned else 0
            details.append(f"[中波] M5未一致 +{1 if h1_aligned else 0}")

    elif wave == "small":
        # 小波: M5が重要。H1は方向感の参考。H4は関係ない
        if m5_aligned:
            quality_score += 3 if h1_aligned else 2
            h1_note = " + H1順行" if h1_aligned else " (H1逆行注意)"
            details.append(f"[小波] M5一致{h1_note} +{3 if h1_aligned else 2}")
        else:
            details.append(f"[小波] M5未一致 +0 ← 入るな")

    # --- 2. ADX Trend Strength (0-2) --- 注目TFのADX
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
            # LONG = base強 + quote弱 が理想
            macro_aligned = base_str > 0.1 and quote_str < -0.1
            macro_neutral = base_str > quote_str
        else:
            # SHORT = base弱 + quote強 が理想
            macro_aligned = base_str < -0.1 and quote_str > 0.1
            macro_neutral = base_str < quote_str

        if macro_aligned:
            quality_score += 2
            details.append(f"マクロ一致({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +2")
        elif macro_neutral:
            quality_score += 1
            details.append(f"マクロ中立({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +1")
        else:
            details.append(f"マクロ逆行({base}={base_str:+.2f},{quote}={quote_str:+.2f}) +0")
    except Exception:
        details.append("マクロ計算不可 +0")

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
            confluence_notes.append(f"H1 Div確認(score={div_rsi:.1f})")

    # StochRSI extreme in entry direction
    if m5:
        stoch = m5.get("stoch_rsi", 0.5)
        if is_long and stoch < 0.15:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f}(極端売られすぎ)")
        elif not is_long and stoch > 0.85:
            confluence += 1
            confluence_notes.append(f"M5 StRSI={stoch:.2f}(極端買われすぎ)")

    # BB position
    if m5:
        bb_pos = m5.get("bb", 0.5)
        if is_long and bb_pos < 0.1:
            confluence_notes.append(f"M5 BB下限({bb_pos:.2f})")
        elif not is_long and bb_pos > 0.9:
            confluence_notes.append(f"M5 BB上限({bb_pos:.2f})")

    conf_points = min(confluence, 2)
    quality_score += conf_points
    if confluence_notes:
        details.append(f"テクニカル複合: {', '.join(confluence_notes)} +{conf_points}")
    else:
        details.append("テクニカル複合なし +0")

    # --- 5. Wave Position Penalty/Bonus (-2 to +1) ---
    if h4:
        h4_cci = h4.get("cci", 0)
        h4_rsi = h4.get("rsi", 50)

        h4_extreme_long = h4_rsi > 70 or h4_cci > 200
        h4_extreme_short = h4_rsi < 30 or h4_cci < -200

        if is_long and h4_extreme_long:
            quality_score -= 2
            details.append(f"⚠ H4過熱圏でLONG(CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← 動き切った後に入るな")
        elif not is_long and h4_extreme_short:
            quality_score -= 2
            details.append(f"⚠ H4極端売られすぎでSHORT(CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) -2 ← 動き切った後に入るな")
        elif is_long and h4_extreme_short:
            quality_score += 1
            details.append(f"H4売られすぎでLONG(CCI={h4_cci:.0f}) +1 ← 逆張りチャンス")
        elif not is_long and h4_extreme_long:
            quality_score += 1
            details.append(f"H4過熱圏でSHORT(CCI={h4_cci:.0f}) +1 ← 逆張りチャンス")
        else:
            details.append(f"H4ニュートラル(CCI={h4_cci:.0f},RSI={h4_rsi:.0f}) +0")
    else:
        details.append("H4データなし +0")

    # --- 6. Spread Penalty (0 to -2) --- スプレッドが狙いに対して大きすぎるか
    spread_pip = _get_current_spread(pair)
    if spread_pip is not None:
        # 波の大きさで許容スプレッドが変わる
        # 大波(15-30pip狙い): スプ2.0pipでも6-13%→許容
        # 中波(10-15pip狙い): スプ2.0pipで13-20%→注意
        # 小波(5-10pip狙い): スプ2.0pipで20-40%→致命的
        pip_targets = {"big": 20, "mid": 12, "small": 7}
        target = pip_targets.get(wave, 12)
        spread_ratio = spread_pip / target

        if spread_ratio > 0.30:  # スプが利幅の30%超 → 致命的
            quality_score -= 2
            details.append(f"⚠️ スプレッド={spread_pip:.1f}pip (利幅{target}pipの{spread_ratio:.0%}) -2 ← RR崩壊。見送れ")
        elif spread_ratio > 0.20:  # 20%超 → 要注意
            quality_score -= 1
            details.append(f"⚠️ スプレッド={spread_pip:.1f}pip (利幅{target}pipの{spread_ratio:.0%}) -1 ← サイズ控えめに")
        else:
            details.append(f"スプレッド={spread_pip:.1f}pip (利幅{target}pipの{spread_ratio:.0%}) OK")
    else:
        details.append("スプレッド取得不可 +0")

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

    # Sizing recommendation — 確度がサイズを決める。波の大きさではない。
    # 小波でも確度Sなら大きく張れ。利幅が小さい分、サイズで稼ぐ。
    sizing = {
        "S": "8000-10000u (鉄板。大きく張れ)",
        "A": "5000-8000u (高確度。しっかり張れ)",
        "B": "2000-3000u (控えめに)",
        "C": "1000u以下 (根拠弱い。見送り推奨)",
    }

    return {
        "grade": grade,
        "quality_score": quality_score,
        "wave": wave,
        "details": details,
        "sizing": sizing[grade],
        "mtf_aligned": aligned_count,
    }


# --- Risk Assessment (後ろ向き: 過去データからのリスク) ---

def assess_risk(
    pair: str,
    direction: str,
    adx: float = None,
    headline: str = None,
    regime: str = None,
    wave: str = "auto",
) -> dict:
    """総合リスク+セットアップ品質判定"""
    conn = get_conn()
    warnings = []
    risk_score = 0  # 0-10

    # 1. トレード統計
    stats = trade_stats(conn, pair, direction)
    if stats["count"] > 0:
        if stats["win_rate"] < 0.5:
            warnings.append(f"⚠️ {pair} {direction} 勝率: {stats['win_rate']:.0%} ({stats['wins']}/{stats['count']})")
            risk_score += 2
        if stats["worst"] < -1000:
            warnings.append(f"🚨 過去最大損失: {stats['worst']:+,.0f}円")
            risk_score += 3

    # 2. レジーム別
    if regime:
        rs = regime_stats(conn, pair, direction, regime)
        if rs["count"] > 0 and rs["win_rate"] < 0.4:
            warnings.append(f"⚠️ {regime}レジームでの勝率: {rs['win_rate']:.0%} ({rs['count']}件)")
            risk_score += 2

    # 3. ヘッドライン/イベントリスク
    if headline:
        hl_trades = active_headlines_history(conn, headline)
        if hl_trades:
            hl_pls = [t["pl"] for t in hl_trades]
            avg = sum(hl_pls) / len(hl_pls)
            if avg < 0:
                warnings.append(f"🚨 {headline}ヘッドライン中の平均PL: {avg:+,.0f}円 ({len(hl_trades)}件)")
                risk_score += 3
            losses = [t for t in hl_trades if t["pl"] < -500]
            for lt in losses:
                warnings.append(f"  → {lt['pair']} {lt['direction']} {lt['pl']:+,.0f}円: {lt['lesson']}")

    events = headline_risk(conn, pair)
    if events:
        high_events = [e for e in events if e["impact"] == "high"]
        if high_events:
            for e in high_events[:2]:
                warnings.append(f"🚨 過去スパイク: {e['headline']} → {pair} {e['spike_pips']:.0f}pip ({e['session_date']})")
                risk_score += 2

    # 4. ユーザーコール
    uc = latest_user_call(conn, pair)
    if uc:
        user_dir = uc.get("direction")
        if user_dir and user_dir != ("UP" if direction == "LONG" else "DOWN"):
            uc_stats = user_call_stats(conn, pair)
            if uc_stats["count"] > 0:
                warnings.append(
                    f"⚠️ ユーザー直近コール: 「{uc['call_text']}」({user_dir}) "
                    f"= {direction}と逆方向 (ユーザー的中率: {uc_stats['accuracy']:.0%})"
                )
                risk_score += 2

    # 5. SLなしの過去の結果
    if stats["count"] > 0 and stats["no_sl_count"] == stats["count"]:
        losses_no_sl = [t for t in fetchall_dict(conn,
            "SELECT pl FROM trades WHERE pair = ? AND had_sl = 0 AND pl IS NOT NULL AND pl < 0",
            (pair,))]
        if losses_no_sl:
            worst = min(t["pl"] for t in losses_no_sl)
            warnings.append(f"⚠️ SLなしでの最大損失: {worst:+,.0f}円 → SL設定を検討")
            risk_score += 1

    # リスクレベル
    if risk_score >= 7:
        level = "HIGH"
    elif risk_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    # 6. pretrade_outcomes: 過去に同条件で入った時の結末
    past_outcomes = _past_pretrade_outcomes(conn, pair, direction, level)

    # 6b. Setup Quality Assessment (前向き)
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

    # 7. ベクトル検索で類似状況
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

    # 8. このチェック結果を pretrade_outcomes に記録（後でdaily_reviewがplを埋める）
    _log_pretrade(conn, pair, direction, level, risk_score, warnings)

    return result


def _past_pretrade_outcomes(conn, pair: str, direction: str, level: str) -> list[dict]:
    """過去に同じpair/direction/levelで入った時の結末"""
    return fetchall_dict(conn,
        """SELECT session_date, pl, lesson_from_review, thesis
           FROM pretrade_outcomes
           WHERE pair = ? AND direction = ? AND pretrade_level = ? AND pl IS NOT NULL
           ORDER BY id DESC LIMIT 5""",
        (pair, direction, level))


def _log_pretrade(conn, pair: str, direction: str, level: str, score: int, warnings: list):
    """このチェック結果をpretrade_outcomesに記録"""
    try:
        conn.execute(
            """INSERT INTO pretrade_outcomes
               (session_date, pair, direction, pretrade_level, pretrade_score, pretrade_warnings)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(date.today()), pair, direction, level, score,
             json.dumps(warnings, ensure_ascii=False))
        )
    except Exception:
        pass  # テーブルがまだない場合は無視


def format_check(result: dict) -> str:
    """チェック結果を読みやすく表示"""
    lines = []
    level = result["risk_level"]
    setup = result.get("setup_quality", {})
    grade = setup.get("grade", "?")
    q_score = setup.get("quality_score", 0)

    grade_icon = {"S": "🔥", "A": "✅", "B": "⚠️", "C": "❌"}.get(grade, "?")
    risk_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[level]

    # Main header: 確度が最重要（サイジングを決める）
    wave_label = {"big": "大波", "mid": "中波", "small": "小波"}.get(setup.get("wave", "?"), "?")
    lines.append(f"{grade_icon} 確度: {grade} (score={q_score}) | リスク: {level} (score={result['risk_score']}) | 波: {wave_label}")
    lines.append(f"   {result['pair']} {result['direction']} → {setup.get('sizing', '?')}")
    lines.append("")

    # Setup quality details
    lines.append("📐 セットアップ品質:")
    for detail in setup.get("details", []):
        lines.append(f"  {detail}")
    lines.append("")

    # トレード統計
    stats = result.get("trade_stats")
    if stats:
        lines.append(f"📊 過去実績: {stats['wins']}勝{stats['losses']}敗 (勝率{stats['win_rate']:.0%}) 平均PL={stats['avg_pl']:+,.0f}円 合計={stats['total_pl']:+,.0f}円")

    # 警告
    if result["warnings"]:
        lines.append("")
        for w in result["warnings"]:
            lines.append(w)

    # 過去の教訓
    if stats and stats.get("lessons"):
        lines.append("")
        lines.append("📝 過去の教訓:")
        for lesson in stats["lessons"][:3]:
            lines.append(f"  - {lesson}")

    # 過去の同条件エントリー結末（フィードバックループの核心）
    past = result.get("past_outcomes", [])
    if past:
        lines.append("")
        lines.append(f"📖 前回 {result['pair']} {result['direction']} を pretrade {result['risk_level']} で入った時:")
        for p in past:
            outcome = "WIN" if p['pl'] and p['pl'] > 0 else "LOSS"
            pl_str = f"{p['pl']:+,.0f}円" if p['pl'] else "?"
            lines.append(f"  [{p['session_date']}] {outcome} {pl_str}")
            if p.get('lesson_from_review'):
                lines.append(f"    → {p['lesson_from_review']}")

    # 類似記憶
    if result.get("similar_memories"):
        lines.append("")
        lines.append("🧠 類似状況の記憶:")
        for mem in result["similar_memories"]:
            lines.append(f"  [{mem['date']}] {mem['content'][:150]}...")

    return "\n".join(lines)


# --- CLI ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 pretrade_check.py <PAIR> <LONG|SHORT> [--wave big|mid|small] [--adx N] [--headline TEXT] [--regime TYPE]")
        print("Example: python3 pretrade_check.py GBP_USD SHORT --wave big")
        print("  --wave: big=H4/H1スイング, mid=M5トレード, small=M1スキャルプ, auto=自動判定(デフォルト)")
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
