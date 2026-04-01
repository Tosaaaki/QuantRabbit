#!/usr/bin/env python3
"""Auto-run each cycle: fetch current positions and account from OANDA and save snapshot"""
import json, time, urllib.request, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = open(os.path.join(ROOT, 'config/env.toml')).read()
token = [l.split('"')[1] for l in env.split('\n') if 'oanda_token' in l][0]
acct = [l.split('"')[1] for l in env.split('\n') if 'oanda_account_id' in l][0]
base = 'https://api-fxtrade.oanda.com'
headers = {'Authorization': f'Bearer {token}'}

def api(path):
    req = urllib.request.Request(f'{base}{path}', headers=headers)
    return json.loads(urllib.request.urlopen(req).read())

trades = api(f'/v3/accounts/{acct}/openTrades').get('trades', [])
summary = api(f'/v3/accounts/{acct}/summary').get('account', {})

snapshot = {
    'ts': int(time.time()),
    'time': time.strftime('%Y-%m-%d %H:%M JST', time.localtime()),
    'trades': [{
        'instrument': t['instrument'],
        'units': t['currentUnits'],
        'price': t['price'],
        'unrealizedPL': t.get('unrealizedPL', '0'),
        'id': t['id']
    } for t in trades],
    'nav': summary.get('NAV', ''),
    'balance': summary.get('balance', ''),
    'marginUsed': summary.get('marginUsed', ''),
    'marginAvailable': summary.get('marginAvailable', ''),
}

out = os.path.join(ROOT, 'logs', '.trader_snapshot.json')
with open(out, 'w') as f:
    json.dump(snapshot, f, ensure_ascii=False, indent=2)

# Brief output
for t in snapshot['trades']:
    print(f"{t['instrument']} {t['units']}u @{t['price']} PL={t['unrealizedPL']}")
print(f"NAV:{snapshot['nav']} Bal:{snapshot['balance']} Margin:{snapshot['marginUsed']}/{snapshot['marginAvailable']}")
