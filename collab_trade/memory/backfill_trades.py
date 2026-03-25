#!/usr/bin/env python3
"""
QuantRabbit — OANDAトランザクション履歴からtradesテーブルをバックフィル

OANDA REST API v20 の /v3/accounts/{id}/transactions を使用して
全ORDER_FILL（エントリー＋決済）を取得し、トレードごとにペアリングして
memory.db の trades テーブルに挿入する。

Usage:
    python3 backfill_trades.py              # 全期間
    python3 backfill_trades.py 2026-03-20   # 指定日以降
    python3 backfill_trades.py --dry-run    # 実行せず件数だけ表示
"""
import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

from schema import get_conn, init_db, fetchone_val, fetchall_dict

# --- Config ---
ENV_TOML = Path(__file__).parent.parent.parent / "config" / "env.toml"

def load_config():
    text = ENV_TOML.read_text()
    cfg = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

CFG = load_config()
TOKEN = CFG['oanda_token']
ACCT = CFG['oanda_account_id']
BASE = 'https://api-fxtrade.oanda.com'
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'Content-Type': 'application/json',
}

PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}


def api_get(path: str, params: dict = None) -> dict:
    url = BASE + path
    if params:
        url += '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_transactions(since: str, to: str = None) -> list[dict]:
    """OANDA transactions API (ページネーション対応)"""
    all_txns = []
    params = {
        'from': since,
        'type': 'ORDER_FILL',
        'pageSize': 1000,
    }
    if to:
        params['to'] = to

    # まずtransaction ID範囲を取得
    id_range = api_get(f'/v3/accounts/{ACCT}/transactions', params)
    pages = id_range.get('pages', [])

    for page_url in pages:
        req = urllib.request.Request(page_url, headers=HEADERS)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            txns = data.get('transactions', [])
            all_txns.extend(txns)

    return all_txns


def parse_fills(txns: list[dict]) -> list[dict]:
    """ORDER_FILLトランザクションからトレード記録を生成

    各ORDER_FILLには tradeOpened / tradesClosed / tradesReduced がある。
    tradesClosed に P/L が含まれる（=決済完了トレード）。
    tradeOpened は新規エントリー（この時点ではP/L未確定）。

    戦略: tradesClosed のある ORDER_FILL からペア・方向・P/L・価格を取得する。
    """
    trades = []

    for txn in txns:
        if txn.get('type') != 'ORDER_FILL':
            continue

        instrument = txn.get('instrument', '')
        if instrument not in PAIRS:
            continue

        # 決済トレードを処理
        trades_closed = txn.get('tradesClosed', [])
        trades_reduced = txn.get('tradesReduced', [])

        for tc in trades_closed + trades_reduced:
            trade_id = tc.get('tradeID', '')
            units = int(float(tc.get('units', 0)))
            pl = float(tc.get('realizedPL', 0))
            price = float(txn.get('price', 0))
            ts = txn.get('time', '')  # RFC3339

            # 方向: 閉じたunitsが正 = LONGを閉じた, 負 = SHORTを閉じた
            # ただしtransaction自体のunitsで判断する方が正確
            # tradesClosed.units は「閉じた分」で符号はエントリーと逆
            direction = 'LONG' if units > 0 else 'SHORT'
            # 実際は逆: 売りで閉じた = 元はLONG
            direction = 'SHORT' if units > 0 else 'LONG'

            # タイムスタンプからセッション日と時間を取得
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00').split('.')[0] + '+00:00')
            except:
                dt = datetime.now(timezone.utc)

            session_date = dt.strftime('%Y-%m-%d')
            session_hour = dt.hour

            # エントリー理由の推測
            order_type = txn.get('type', '')
            reason = txn.get('reason', '')

            trades.append({
                'session_date': session_date,
                'trade_id': trade_id,
                'pair': instrument,
                'direction': direction,
                'units': abs(units),
                'entry_price': None,  # ORDER_FILLからは取得困難
                'exit_price': price,
                'pl': pl,
                'session_hour': session_hour,
                'reason': reason,
                'timestamp': ts,
            })

        # 新規エントリーも記録（P/Lなし）
        trade_opened = txn.get('tradeOpened', None)
        if trade_opened and not trades_closed and not trades_reduced:
            trade_id = trade_opened.get('tradeID', '')
            units = int(float(trade_opened.get('units', 0)))
            price = float(txn.get('price', 0))
            ts = txn.get('time', '')

            direction = 'LONG' if units > 0 else 'SHORT'

            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00').split('.')[0] + '+00:00')
            except:
                dt = datetime.now(timezone.utc)

            session_date = dt.strftime('%Y-%m-%d')
            session_hour = dt.hour

            trades.append({
                'session_date': session_date,
                'trade_id': trade_id,
                'pair': instrument,
                'direction': direction,
                'units': abs(units),
                'entry_price': price,
                'exit_price': None,
                'pl': None,  # まだ決済してない
                'session_hour': session_hour,
                'reason': txn.get('reason', ''),
                'timestamp': ts,
            })

    return trades


def merge_trades(raw_trades: list[dict]) -> list[dict]:
    """エントリーと決済をtrade_idでマージ"""
    entries = {}  # trade_id -> entry record
    closes = {}   # trade_id -> close record(s)

    for t in raw_trades:
        tid = t['trade_id']
        if t['entry_price'] is not None and t['pl'] is None:
            entries[tid] = t
        elif t['pl'] is not None:
            if tid not in closes:
                closes[tid] = t
            else:
                # 部分決済: P/Lを合算
                closes[tid]['pl'] += t['pl']
                closes[tid]['units'] += t['units']

    # マージ
    merged = []
    all_ids = set(list(entries.keys()) + list(closes.keys()))

    for tid in all_ids:
        entry = entries.get(tid)
        close = closes.get(tid)

        if entry and close:
            merged.append({
                'session_date': close['session_date'],  # 決済日
                'trade_id': tid,
                'pair': entry['pair'],
                'direction': entry['direction'],
                'units': entry['units'],
                'entry_price': entry['entry_price'],
                'exit_price': close['exit_price'],
                'pl': close['pl'],
                'session_hour': entry['session_hour'],
                'reason': entry['reason'],
            })
        elif close:
            # エントリーが見つからない（古いトレード等）
            merged.append({
                'session_date': close['session_date'],
                'trade_id': tid,
                'pair': close['pair'],
                'direction': close['direction'],
                'units': close['units'],
                'entry_price': None,
                'exit_price': close['exit_price'],
                'pl': close['pl'],
                'session_hour': close['session_hour'],
                'reason': close['reason'],
            })
        # entry only = まだオープン中 → スキップ

    return merged


def backfill(since_date: str = None, dry_run: bool = False):
    init_db()
    conn = get_conn()

    # 既存のtrade_id取得（重複防止）
    existing_ids = set()
    for row in conn.execute("SELECT trade_id FROM trades WHERE trade_id IS NOT NULL"):
        existing_ids.add(row[0])
    print(f"既存trades: {len(existing_ids)}件")

    # OANDA APIから取得
    if since_date:
        since = f"{since_date}T00:00:00.000000000Z"
    else:
        since = "2026-03-01T00:00:00.000000000Z"  # プロジェクト開始日

    print(f"OANDA APIからトランザクション取得中 (since={since})...")
    txns = fetch_transactions(since)
    print(f"取得: {len(txns)}件のORDER_FILL")

    raw_trades = parse_fills(txns)
    print(f"パース: {len(raw_trades)}件の生トレード（エントリー+決済）")

    merged = merge_trades(raw_trades)
    print(f"マージ: {len(merged)}件の完了トレード")

    # 重複除外
    new_trades = [t for t in merged if t['trade_id'] not in existing_ids]
    print(f"新規: {new_trades.__len__()}件（{len(merged) - len(new_trades)}件は既存）")

    if dry_run:
        print("\n--- DRY RUN ---")
        for t in new_trades[:10]:
            print(f"  {t['session_date']} {t['pair']} {t['direction']} {t['units']}u PL={t['pl']}")
        if len(new_trades) > 10:
            print(f"  ... 他{len(new_trades)-10}件")
        return

    # DB挿入
    inserted = 0
    for t in new_trades:
        try:
            conn.execute(
                """INSERT INTO trades (session_date, trade_id, pair, direction, units,
                   entry_price, exit_price, pl, session_hour, reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (t['session_date'], t['trade_id'], t['pair'], t['direction'], t['units'],
                 t['entry_price'], t['exit_price'], t['pl'], t['session_hour'], t['reason'])
            )
            inserted += 1
        except Exception as e:
            print(f"  ERROR inserting trade {t['trade_id']}: {e}")

    print(f"\n完了: {inserted}件挿入")

    # サマリー
    stats = fetchall_dict(conn, """
        SELECT pair, direction, COUNT(*) as cnt,
               SUM(CASE WHEN pl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pl < 0 THEN 1 ELSE 0 END) as losses,
               ROUND(SUM(pl), 1) as total_pl
        FROM trades
        WHERE pl IS NOT NULL
        GROUP BY pair, direction
        ORDER BY pair, direction
    """)
    print("\n--- トレードサマリー ---")
    for s in stats:
        wr = round(s['wins'] / s['cnt'] * 100, 1) if s['cnt'] > 0 else 0
        print(f"  {s['pair']} {s['direction']}: {s['cnt']}件 WR={wr}% PL={s['total_pl']}円")


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    since = args[0] if args else None
    backfill(since_date=since, dry_run=dry_run)
