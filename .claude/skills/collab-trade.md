---
name: collab-trade
description: "「共同トレード」でユーザー×Claude共同裁量トレードセッションを開始。タスク停止→monitor停止→口座確認→市況チェック→トレード開始。"
trigger: "Use when the user says '共同トレード', 'collab trade', or wants to start a collaborative discretionary trading session."
---

# 共同トレード起動スキル

ユーザーが「共同トレード」と言ったら、このスキルが起動する。

## 実行手順

### Step 1: 行動規範を読む

`collab_trade/CLAUDE.md` を**全部**読む。行動規範・手法・テクニカル・失敗パターン・ペア別ノートが全て入っている。

### Step 2: 外部記憶 + 統括を読む

- `collab_trade/state.md` — 前回セッションの状態。即復帰用
- `collab_trade/summary.md` — 全日の成績・傾向

### Step 3: 定期タスク・monitor停止

```bash
launchctl stop com.quantrabbit.trader 2>/dev/null
launchctl stop com.quantrabbit.analyst 2>/dev/null
launchctl stop com.quantrabbit.secretary 2>/dev/null
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

req = urllib.request.Request(f'{base}/v3/accounts/{acct}/summary', headers={'Authorization': f'Bearer {token}'})
a = json.loads(urllib.request.urlopen(req).read())['account']
print(f'=== ACCOUNT ===')
print(f'Balance: {float(a[\"balance\"]):,.0f} JPY | NAV: {float(a[\"NAV\"]):,.0f} JPY | UPL: {float(a[\"unrealizedPL\"]):,.0f} JPY')
print(f'Margin Used: {float(a[\"marginUsed\"]):,.0f} | Available: {float(a[\"marginAvailable\"]):,.0f}')
print(f'Open Trades: {a[\"openTradeCount\"]}')
print()

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

### Step 5: 今日の日次ディレクトリ作成

```bash
TODAY=$(date -u +%Y-%m-%d)
mkdir -p collab_trade/daily/$TODAY
```

### Step 6: state.md を初期化

今日の日付・口座状態・テーゼで `collab_trade/state.md` を更新する。

### Step 7: ユーザーに報告してトレード開始

口座状態・市況を簡潔に報告し、即トレード開始。

---

## トレード中の記録先

| ファイル | いつ書く | 何を書く |
|----------|---------|---------|
| `collab_trade/state.md` | 随時 | 現在のポジション・テーゼ・確定益（外部記憶） |
| `collab_trade/daily/YYYY-MM-DD/trades.md` | トレード毎 | エントリー・決済の詳細テーブル |
| `collab_trade/daily/YYYY-MM-DD/notes.md` | 随時 | ユーザー発言・気づき・発見 |
| `collab_trade/summary.md` | セッション終了時 | 日次統括の更新 |
| `collab_trade/CLAUDE.md` | 重要発見時 | notes→手法・ルールへの昇格 |

## 絶対ルール

- **バックグラウンドタスク禁止** — 対話の中でその場でAPI叩く
- **気づいたことは即書く** — ToDoは達成する。「次回やる」は禁止
- **全ての詳細は `collab_trade/CLAUDE.md` を参照**
