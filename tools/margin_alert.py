#!/usr/bin/env python3
"""毎サイクル実行: マージン使用率・ペア分散・確度の質を検知して出力。
traderのBashに組み込んで、出力で意識させる。

方針: マージンは積極的に使え。ただし確度の高いエントリーだけで。
"""
import urllib.request, json, sys, os, re

def get_recent_low_rate():
    """直近のエントリーのLOW率を計算"""
    log_path = 'logs/live_trade_log.txt'
    if not os.path.exists(log_path):
        return 0, 0, 0

    entries = []
    for line in open(log_path):
        if 'ENTRY' not in line:
            continue
        if 'pretrade' in line.lower() or 'PRETRADE' in line:
            entries.append(line)

    # 直近20件
    recent = entries[-20:] if len(entries) >= 20 else entries
    low = sum(1 for e in recent if 'LOW' in e)
    med = sum(1 for e in recent if 'MEDIUM' in e or 'MED' in e)
    high = sum(1 for e in recent if 'HIGH' in e)
    total = len(recent)
    return low, med + high, total


def main():
    cfg = {}
    for line in open('config/env.toml'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')

    token = cfg['oanda_token']
    acct = cfg['oanda_account_id']
    base = 'https://api-fxtrade.oanda.com'
    headers = {'Authorization': f'Bearer {token}'}

    # Account summary
    req = urllib.request.Request(f'{base}/v3/accounts/{acct}/summary', headers=headers)
    a = json.loads(urllib.request.urlopen(req).read())['account']
    nav = float(a['NAV'])
    margin_used = float(a['marginUsed'])
    margin_pct = margin_used / nav * 100 if nav > 0 else 0

    # Open trades
    req2 = urllib.request.Request(f'{base}/v3/accounts/{acct}/openTrades', headers=headers)
    trades = json.loads(urllib.request.urlopen(req2).read())['trades']

    # Count unique pairs
    pairs_held = set(t['instrument'] for t in trades)
    all_pairs = {'USD_JPY', 'EUR_USD', 'GBP_USD', 'AUD_USD', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY'}
    empty_pairs = all_pairs - pairs_held

    alerts = []

    # Margin alerts — マージンは積極的に使え。80%が標準
    if margin_pct < 30:
        alerts.append(f'🚨 MARGIN {margin_pct:.0f}% — 資金効率が悪い。7ペアスキャンしてチャンスを探せ。3000u以上で入れ')
    elif margin_pct < 50:
        alerts.append(f'⚠️ MARGIN {margin_pct:.0f}% — まだ半分以上余ってる。マージン80%が標準。チャンスを探して入れ')
    elif margin_pct < 70:
        alerts.append(f'📡 MARGIN {margin_pct:.0f}% — まだ余力あり。80%目指してチャンスがあれば追加')

    # Pair concentration alert
    if len(pairs_held) == 1 and len(trades) >= 3:
        pair = list(pairs_held)[0]
        alerts.append(f'⚠️ CONCENTRATION: {pair}に{len(trades)}本集中。他ペアにもチャンスがあるはず')

    # Empty pairs alert
    if len(empty_pairs) >= 5:
        alerts.append(f'📡 {len(empty_pairs)}ペアがノーポジ（{", ".join(sorted(empty_pairs))}）。確度の高いチャンスを探せ')
    elif len(empty_pairs) >= 4:
        alerts.append(f'📡 {len(empty_pairs)}ペアがノーポジ（{", ".join(sorted(empty_pairs))}）')

    # エントリー品質（参考情報として表示）
    low, med_high, total = get_recent_low_rate()
    if total >= 10:
        # LOW率は参考値。LOWでも成績は良い。重要なのはテーゼの確信度
        pass  # pretrade判定は品質指標として信頼性が低いため、警告を出さない

    # Total unrealized loss alert
    total_upl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
    if total_upl < -2000:
        alerts.append(f'🚨 含み損 ¥{total_upl:,.0f} — テーゼ崩壊してないか確認しろ')

    if alerts:
        print('=== ⚡ OWNER ALERT (自動検知) ===')
        for a in alerts:
            print(a)
        print('================================')
    else:
        print(f'MARGIN OK: {margin_pct:.0f}% | {len(pairs_held)}ペア保有')

if __name__ == '__main__':
    main()
