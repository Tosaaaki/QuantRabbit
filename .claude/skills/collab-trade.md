---
name: collab-trade
description: "「共同トレード」でユーザー×Claude共同裁量トレードセッションを開始。タスク停止→monitor停止→口座確認→市況チェック→トレード開始。"
trigger: "Use when the user says '共同トレード', 'collab trade', or wants to start a collaborative discretionary trading session."
---

# 共同トレード起動スキル

ユーザーが「共同トレード」と言ったら、このスキルが起動する。

## 実行手順

### Step 1: 行動規範を読む

`collab_trade/CLAUDE.md` を読む。ここに全ての原則・ルール・手順が書いてある。

### Step 2: 外部記憶を読む

`collab_trade/state.md` を読む。前回セッションの状態が残っていれば即復帰。

### Step 3: 定期タスク・monitor停止

```bash
# 定期タスク停止
launchctl stop com.quantrabbit.trader 2>/dev/null
launchctl stop com.quantrabbit.analyst 2>/dev/null
launchctl stop com.quantrabbit.secretary 2>/dev/null

# live_monitor停止
launchctl stop com.quantrabbit.live-monitor 2>/dev/null
```

### Step 4: 口座確認 + 全ペアプライス取得

config/env.tomlからOANDA認証情報を読み取り、口座残高・オープンポジション・主要ペアの現在価格を取得する。

```bash
python3 -c "
import urllib.request, json

token = open('config/env.toml').read()
token = [l.split('=')[1].strip().strip('\"') for l in token.split('\n') if l.startswith('oanda_token')][0]
acct = [l.split('=')[1].strip().strip('\"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_account_id')][0]
base = 'https://api-fxtrade.oanda.com'

# Account summary
req = urllib.request.Request(f'{base}/v3/accounts/{acct}/summary', headers={'Authorization': f'Bearer {token}'})
a = json.loads(urllib.request.urlopen(req).read())['account']
print(f'=== ACCOUNT ===')
print(f'Balance: {float(a[\"balance\"]):,.0f} JPY | NAV: {float(a[\"NAV\"]):,.0f} JPY | UPL: {float(a[\"unrealizedPL\"]):,.0f} JPY')
print(f'Margin Used: {float(a[\"marginUsed\"]):,.0f} | Available: {float(a[\"marginAvailable\"]):,.0f}')
print(f'Open Trades: {a[\"openTradeCount\"]}')
print()

# Open trades
req = urllib.request.Request(f'{base}/v3/accounts/{acct}/trades?state=OPEN', headers={'Authorization': f'Bearer {token}'})
trades = json.loads(urllib.request.urlopen(req).read())['trades']
if trades:
    print('=== OPEN POSITIONS ===')
    for t in trades:
        print(f'{t[\"instrument\"]} {int(t[\"currentUnits\"]):+d}u @ {t[\"price\"]} | UPL: {float(t[\"unrealizedPL\"]):+.0f} JPY')
    print()
else:
    print('=== NO OPEN POSITIONS (FLAT) ===')
    print()

# Pricing
pairs = 'USD_JPY,EUR_USD,GBP_USD,AUD_USD,EUR_JPY,GBP_JPY,AUD_JPY,NZD_USD,USD_CHF,USD_CAD'
req = urllib.request.Request(f'{base}/v3/accounts/{acct}/pricing?instruments={pairs}', headers={'Authorization': f'Bearer {token}'})
prices = json.loads(urllib.request.urlopen(req).read())['prices']
print('=== PRICES ===')
for p in prices:
    bid = float(p['bids'][0]['price'])
    ask = float(p['asks'][0]['price'])
    spread = (ask - bid) * (100 if 'JPY' in p['instrument'] else 10000)
    print(f'{p[\"instrument\"]}: {bid:.5g}/{ask:.5g} (sp:{spread:.1f})')
"
```

### Step 5: state.md を初期化

新しいセッション情報で `collab_trade/state.md` を更新する。

### Step 6: ユーザーに報告してトレード開始

口座状態・市況を簡潔に報告し、即トレード開始。

---

## トレード中の行動

- **全ての原則は `collab_trade/CLAUDE.md` に従う**
- **state.md に随時書き込む**（外部記憶）
- **live_trade_log.txt に全トレードを記録**
- **バックグラウンドタスク禁止** — 対話の中でその場でAPI叩く
- **気づいたことは即書く。ToDoは達成する**
