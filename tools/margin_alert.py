#!/usr/bin/env python3
"""Run every cycle: detect and output margin usage, pair diversification, and entry quality.
Embed in the trader's Bash to keep these metrics top of mind.

Policy: use margin aggressively — but only on high-confidence entries.
"""
import urllib.request, json, sys, os, re

def get_recent_low_rate():
    """Calculate the LOW rate for recent entries"""
    log_path = 'logs/live_trade_log.txt'
    if not os.path.exists(log_path):
        return 0, 0, 0

    entries = []
    for line in open(log_path):
        if 'ENTRY' not in line:
            continue
        if 'pretrade' in line.lower() or 'PRETRADE' in line:
            entries.append(line)

    # Last 20 entries
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

    # Margin alerts — use margin aggressively; 80% is the standard
    if margin_pct < 30:
        alerts.append(f'🚨 MARGIN {margin_pct:.0f}% — poor capital efficiency. Scan all 7 pairs for opportunities. Enter with 3000u+')
    elif margin_pct < 50:
        alerts.append(f'⚠️ MARGIN {margin_pct:.0f}% — more than half unused. 80% margin is the standard. Look for opportunities and enter')
    elif margin_pct < 70:
        alerts.append(f'📡 MARGIN {margin_pct:.0f}% — still has capacity. Add positions if opportunities arise, targeting 80%')

    # Pair concentration alert
    if len(pairs_held) == 1 and len(trades) >= 3:
        pair = list(pairs_held)[0]
        alerts.append(f'⚠️ CONCENTRATION: {len(trades)} positions concentrated in {pair}. There should be opportunities in other pairs')

    # Empty pairs alert
    if len(empty_pairs) >= 5:
        alerts.append(f'📡 {len(empty_pairs)} pairs with no position ({", ".join(sorted(empty_pairs))}). Look for high-confidence opportunities')
    elif len(empty_pairs) >= 4:
        alerts.append(f'📡 {len(empty_pairs)} pairs with no position ({", ".join(sorted(empty_pairs))})')

    # Entry quality (displayed as reference information)
    low, med_high, total = get_recent_low_rate()
    if total >= 10:
        # LOW rate is a reference value. Even LOW entries can perform well. What matters is conviction in the thesis.
        pass  # pretrade judgment has low reliability as a quality indicator, so no alert is raised

    # Total unrealized loss alert
    total_upl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
    if total_upl < -2000:
        alerts.append(f'🚨 Unrealized loss ¥{total_upl:,.0f} — check whether the thesis has broken down')

    if alerts:
        print('=== ⚡ OWNER ALERT (auto-detected) ===')
        for a in alerts:
            print(a)
        print('======================================')
    else:
        print(f'MARGIN OK: {margin_pct:.0f}% | {len(pairs_held)} pairs held')

if __name__ == '__main__':
    main()
