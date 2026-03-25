"""
QuantRabbit Trading Memory — Pre-Trade Check
エントリー直前に過去の記憶を3層で照合し、リスク判定する

使い方:
  python3 pretrade_check.py GBP_USD SHORT [--adx 38] [--headline "Iran"]
"""
import sys
import json
from schema import get_conn, fetchall_dict, fetchone_val, fetchone_dict, serialize_f32

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


# --- Risk Assessment ---

def assess_risk(
    pair: str,
    direction: str,
    adx: float = None,
    headline: str = None,
    regime: str = None,
) -> dict:
    """総合リスク判定"""
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

    result = {
        "pair": pair,
        "direction": direction,
        "risk_level": level,
        "risk_score": risk_score,
        "warnings": warnings,
        "trade_stats": stats if stats["count"] > 0 else None,
    }

    # 6. ベクトル検索で類似状況
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

    return result


def format_check(result: dict) -> str:
    """チェック結果を読みやすく表示"""
    lines = []
    level = result["risk_level"]
    icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[level]
    lines.append(f"{icon} PRE-TRADE CHECK: {result['pair']} {result['direction']} → RISK: {level} (score={result['risk_score']})")
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
        print("Usage: python3 pretrade_check.py <PAIR> <LONG|SHORT> [--adx N] [--headline TEXT] [--regime TYPE]")
        print("Example: python3 pretrade_check.py GBP_USD SHORT --headline Iran")
        sys.exit(1)

    pair = sys.argv[1]
    direction = sys.argv[2]

    adx = None
    headline = None
    regime = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--adx" and i + 1 < len(sys.argv):
            adx = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--headline" and i + 1 < len(sys.argv):
            headline = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--regime" and i + 1 < len(sys.argv):
            regime = sys.argv[i + 1]; i += 2
        else:
            i += 1

    result = assess_risk(pair, direction, adx=adx, headline=headline, regime=regime)
    print(format_check(result))
