"""
QuantRabbit — Daily Review Data Gatherer
日次振り返りのための事実収集。判断はしない、事実を並べる。

daily-review scheduled taskが呼ぶ。Claudeはこの出力を読んで
strategy_memory.mdに振り返りを書く。

使い方:
  python3 tools/daily_review.py --date 2026-03-27
  python3 tools/daily_review.py  # 今日
"""
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "collab_trade" / "memory"))
from schema import get_conn, init_db, fetchall_dict, fetchone_val

ENV_TOML = ROOT / "config" / "env.toml"
OANDA_PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}


def _load_oanda_config():
    text = ENV_TOML.read_text()
    cfg = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def fetch_closed_trades(session_date: str) -> list[dict]:
    """OANDA APIから指定日の決済済みトレードを取得"""
    try:
        cfg = _load_oanda_config()
    except Exception:
        return []

    token = cfg.get('oanda_token', '')
    acct = cfg.get('oanda_account_id', '')
    base = 'https://api-fxtrade.oanda.com'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    since = f"{session_date}T00:00:00.000000000Z"
    dt = datetime.strptime(session_date, '%Y-%m-%d')
    to = (dt + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00.000000000Z')

    try:
        import urllib.parse
        params = urllib.parse.urlencode({
            'from': since, 'to': to, 'type': 'ORDER_FILL', 'pageSize': 1000,
        })
        url = f"{base}/v3/accounts/{acct}/transactions?{params}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            id_range = json.loads(resp.read())

        all_txns = []
        for page_url in id_range.get('pages', []):
            req = urllib.request.Request(page_url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                all_txns.extend(data.get('transactions', []))
    except Exception as e:
        print(f"OANDA API error: {e}")
        return []

    entries = {}
    closes = {}
    for txn in all_txns:
        if txn.get('type') != 'ORDER_FILL':
            continue
        instrument = txn.get('instrument', '')
        if instrument not in OANDA_PAIRS:
            continue

        trade_opened = txn.get('tradeOpened')
        if trade_opened:
            tid = trade_opened.get('tradeID', '')
            units = int(float(trade_opened.get('units', 0)))
            entries[tid] = {
                'price': float(txn.get('price', 0)),
                'units': abs(units),
                'direction': 'LONG' if units > 0 else 'SHORT',
                'pair': instrument,
                'time': txn.get('time', ''),
            }

        for tc in txn.get('tradesClosed', []) + txn.get('tradesReduced', []):
            tid = tc.get('tradeID', '')
            pl = float(tc.get('realizedPL', 0))
            units_c = abs(int(float(tc.get('units', 0))))
            close_time = txn.get('time', '')
            if tid not in closes:
                closes[tid] = {
                    'price': float(txn.get('price', 0)),
                    'pl': pl, 'units': units_c,
                    'instrument': instrument, 'time': close_time,
                }
            else:
                closes[tid]['pl'] += pl
                closes[tid]['units'] += units_c

    trades = []
    for tid in set(list(entries.keys()) + list(closes.keys())):
        entry = entries.get(tid)
        close = closes.get(tid)
        if not close:
            continue
        pair = entry['pair'] if entry else close.get('instrument', 'UNKNOWN')
        direction = entry['direction'] if entry else ('LONG' if close['pl'] >= 0 else 'SHORT')

        # 保持時間計算
        hold_min = None
        if entry and close:
            try:
                t1 = datetime.fromisoformat(entry['time'].replace('Z', '+00:00').split('.')[0] + '+00:00')
                t2 = datetime.fromisoformat(close['time'].replace('Z', '+00:00').split('.')[0] + '+00:00')
                hold_min = int((t2 - t1).total_seconds() / 60)
            except Exception:
                pass

        trades.append({
            'trade_id': tid,
            'pair': pair,
            'direction': direction,
            'units': entry['units'] if entry else close['units'],
            'entry_price': entry['price'] if entry else None,
            'exit_price': close['price'],
            'pl': close['pl'],
            'hold_minutes': hold_min,
        })

    return trades


def match_pretrade_outcomes(conn, session_date: str, oanda_trades: list[dict]):
    """pretrade_outcomesテーブルのplを埋める（予測と結果の紐付け）"""
    # この日のpretrade_outcomes（plがまだ空のもの）
    outcomes = fetchall_dict(conn,
        """SELECT id, pair, direction, pretrade_level
           FROM pretrade_outcomes
           WHERE session_date = ? AND pl IS NULL""",
        (session_date,))

    if not outcomes:
        return 0

    # ペア×方向でグループ化
    from collections import defaultdict
    trade_by_key = defaultdict(list)
    for t in oanda_trades:
        trade_by_key[(t['pair'], t['direction'])].append(t)

    matched = 0
    for outcome in outcomes:
        key = (outcome['pair'], outcome['direction'])
        trades_for_key = trade_by_key.get(key, [])
        if not trades_for_key:
            continue

        # 最初の未マッチトレードを紐付け
        trade = trades_for_key.pop(0)
        conn.execute(
            "UPDATE pretrade_outcomes SET pl = ?, trade_id = ? WHERE id = ?",
            (trade['pl'], trade['trade_id'], outcome['id'])
        )
        matched += 1

    return matched


def parse_pretrade_from_trades_md(trades_md_path: Path) -> list[dict]:
    """trades.mdからpretrade結果付きエントリーを抽出"""
    if not trades_md_path.exists():
        return []

    text = trades_md_path.read_text()
    entries = []

    # テーブル行からpretrade結果を抽出
    for line in text.split('\n'):
        if 'pretrade' not in line.lower() and 'LOW' not in line and 'MEDIUM' not in line and 'HIGH' not in line:
            continue
        # ペアと方向を抽出
        pair_match = re.search(r'(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)', line)
        dir_match = re.search(r'(LONG|SHORT)', line)
        level_match = re.search(r'(?:pretrade|PRETRADE)[:\s]*(\w+)', line)
        if not level_match:
            level_match = re.search(r'(LOW|MEDIUM|HIGH)', line)

        if pair_match and dir_match and level_match:
            entries.append({
                'pair': pair_match.group(1),
                'direction': dir_match.group(1),
                'pretrade_level': level_match.group(1).upper(),
                'line': line.strip(),
            })

    return entries


def generate_report(session_date: str) -> str:
    """日次レビュー用の構造化レポートを生成"""
    conn = get_conn()
    lines = []

    lines.append(f"# Daily Review Report: {session_date}")
    lines.append("")

    # 1. OANDA決済トレード
    oanda_trades = fetch_closed_trades(session_date)
    lines.append(f"## 決済トレード ({len(oanda_trades)}件)")
    lines.append("")

    if not oanda_trades:
        lines.append("決済トレードなし")
    else:
        total_pl = sum(t['pl'] for t in oanda_trades)
        wins = [t for t in oanda_trades if t['pl'] > 0]
        losses = [t for t in oanda_trades if t['pl'] <= 0]
        lines.append(f"合計P&L: {total_pl:+,.0f}円 | 勝: {len(wins)} 敗: {len(losses)} | 勝率: {len(wins)/len(oanda_trades)*100:.0f}%")
        lines.append("")

        # ペア別集計
        from collections import defaultdict
        by_pair = defaultdict(list)
        for t in oanda_trades:
            by_pair[t['pair']].append(t)

        for pair, trades in sorted(by_pair.items()):
            pair_pl = sum(t['pl'] for t in trades)
            pair_wins = sum(1 for t in trades if t['pl'] > 0)
            lines.append(f"### {pair}: {pair_pl:+,.0f}円 ({pair_wins}/{len(trades)}勝)")
            for t in trades:
                hold = f" ({t['hold_minutes']}min)" if t['hold_minutes'] is not None else ""
                lines.append(f"  {t['direction']} {t['units']}u → {t['pl']:+,.0f}円{hold}")
            lines.append("")

    # 2. pretrade_outcomes マッチング
    matched = match_pretrade_outcomes(conn, session_date, oanda_trades)
    if matched:
        lines.append(f"## Pretrade Outcomes: {matched}件紐付け済み")
        lines.append("")

    # 3. pretrade結果 vs 実際のP&L（この日のフィードバック）
    outcomes = fetchall_dict(conn,
        """SELECT pair, direction, pretrade_level, pretrade_score, pl, pretrade_warnings
           FROM pretrade_outcomes
           WHERE session_date = ? AND pl IS NOT NULL
           ORDER BY pl""",
        (session_date,))

    if outcomes:
        lines.append("## Pretrade予測 vs 結果")
        lines.append("")
        for o in outcomes:
            result = "WIN" if o['pl'] > 0 else "LOSS"
            lines.append(f"  {o['pair']} {o['direction']} pretrade={o['pretrade_level']}(score={o['pretrade_score']}) → {result} {o['pl']:+,.0f}円")
        lines.append("")

        # LOW無視の分析
        low_entries = [o for o in outcomes if o['pretrade_level'] == 'LOW']
        if low_entries:
            low_wins = sum(1 for o in low_entries if o['pl'] > 0)
            low_total_pl = sum(o['pl'] for o in low_entries)
            lines.append(f"### LOW無視エントリー: {len(low_entries)}件")
            lines.append(f"  勝率: {low_wins}/{len(low_entries)} | 合計: {low_total_pl:+,.0f}円")
            lines.append("")

    # 4. trades.mdからpretrade注記のあるエントリー（DB未登録のものも拾う）
    trades_md_path = ROOT / "collab_trade" / "daily" / session_date / "trades.md"
    md_entries = parse_pretrade_from_trades_md(trades_md_path)
    if md_entries:
        lines.append("## trades.md内のpretrade注記エントリー")
        lines.append("")
        for e in md_entries:
            lines.append(f"  {e['pair']} {e['direction']} pretrade={e['pretrade_level']}")
        lines.append("")

    # 5. 勝ちパターン vs 負けパターン
    if oanda_trades:
        lines.append("## パターン分析")
        lines.append("")

        # 勝ちの平均保持時間 vs 負けの平均保持時間
        win_holds = [t['hold_minutes'] for t in wins if t['hold_minutes'] is not None]
        loss_holds = [t['hold_minutes'] for t in losses if t['hold_minutes'] is not None]
        if win_holds:
            lines.append(f"  勝ちの平均保持: {sum(win_holds)/len(win_holds):.0f}分")
        if loss_holds:
            lines.append(f"  負けの平均保持: {sum(loss_holds)/len(loss_holds):.0f}分")

        # 最大勝ち/負け
        if wins:
            best = max(wins, key=lambda t: t['pl'])
            lines.append(f"  最大勝ち: {best['pair']} {best['direction']} {best['pl']:+,.0f}円")
        if losses:
            worst = min(losses, key=lambda t: t['pl'])
            lines.append(f"  最大負け: {worst['pair']} {worst['direction']} {worst['pl']:+,.0f}円")

        lines.append("")

    # 6. DBのtrades統計（全期間との比較）
    db_stats = fetchall_dict(conn,
        """SELECT pair, direction,
           COUNT(*) as cnt,
           SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) as wins,
           AVG(pl) as avg_pl,
           SUM(pl) as total_pl
           FROM trades WHERE pl IS NOT NULL
           GROUP BY pair, direction
           ORDER BY total_pl""")

    if db_stats:
        lines.append("## 全期間ペア×方向別成績（参考）")
        lines.append("")
        for s in db_stats:
            wr = s['wins'] / s['cnt'] * 100 if s['cnt'] > 0 else 0
            lines.append(f"  {s['pair']} {s['direction']}: {s['wins']}/{s['cnt']}勝({wr:.0f}%) 平均{s['avg_pl']:+,.0f}円 合計{s['total_pl']:+,.0f}円")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    init_db()

    target_date = str(date.today())
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--date" and i < len(sys.argv) - 1:
            target_date = sys.argv[i + 1]

    report = generate_report(target_date)
    print(report)
