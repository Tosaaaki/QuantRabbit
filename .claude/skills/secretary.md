---
name: secretary
description: "「秘書」でOANDA APIライブデータを取得し、口座・ポジション・アラート・ロック状態を即レポート。裁量FXトレーディング秘書。"
trigger: "Use when the user says '秘書', 'ポジション', '口座', 'status', or asks about current trading status/positions."
---

# トレーディング秘書スキル

ユーザーが「秘書」と言ったら、即座にトレーディング秘書として応答する。

## 実行手順

### Step 1: OANDA APIライブデータ取得（最優先）

**shared_state.jsonを鵜呑みにするな。必ずOANDA APIからライブデータを最初に取得する。**

config/env.tomlからOANDA認証情報を読み取り、以下を並列実行:

```bash
python3 -c "
import urllib.request, json
token = open('config/env.toml').read()
token = [l.split('=')[1].strip().strip('\"') for l in token.split('\n') if l.startswith('oanda_token')][0]
acct = [l.split('=')[1].strip().strip('\"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_account_id')][0]
base = 'https://api-fxtrade.oanda.com'

req = urllib.request.Request(f'{base}/v3/accounts/{acct}/summary', headers={'Authorization': f'Bearer {token}'})
resp = json.loads(urllib.request.urlopen(req).read())
a = resp['account']
print('=== ACCOUNT ===')
print(f'NAV: {a[\"NAV\"]}  Balance: {a[\"balance\"]}  UPL: {a[\"unrealizedPL\"]}  MarginUsed: {a[\"marginUsed\"]}  MarginAvail: {a[\"marginAvailable\"]}  OpenTrades: {a[\"openTradeCount\"]}')

req2 = urllib.request.Request(f'{base}/v3/accounts/{acct}/openTrades', headers={'Authorization': f'Bearer {token}'})
resp2 = json.loads(urllib.request.urlopen(req2).read())
print('=== OPEN TRADES ===')
for t in resp2.get('trades', []):
    print(f'{t[\"instrument\"]} units={t[\"currentUnits\"]} entry={t[\"price\"]} UPL={t[\"unrealizedPL\"]}  id={t[\"id\"]}')
    if 'stopLossOrder' in t: print(f'  SL={t[\"stopLossOrder\"][\"price\"]}')
    if 'takeProfitOrder' in t: print(f'  TP={t[\"takeProfitOrder\"][\"price\"]}')
    if 'trailingStopLossOrder' in t: print(f'  TRAIL={t[\"trailingStopLossOrder\"][\"distance\"]}')
"
```

### Step 2: 補助情報取得（並列実行）

- `logs/shared_state.json` — レジーム・アラート・テクニカル（updated_atが10分以上古ければ「データが古い」と明示）
- `tail -30 logs/live_trade_log.txt` — 直近のトレーダー判断
- `python3 scripts/trader_tools/task_lock.py status` — ロック状態

### Step 3: レポート出力

以下のフォーマットで簡潔にレポート:

```
**秘書レポート** (OANDA APIライブ取得)

## 口座
| 項目 | 値 |
|------|-----|
| NAV | xxx JPY |
| Balance | xxx JPY |
| UPL | +/-xxx JPY |
| Margin Used | xxx (xx%) |
| Margin Avail | xxx |

## オープンポジション (N件)
| # | ペア | 方向 | 数量 | Entry | 現UPL | SL | TP |
（各ポジション）

## 直近アクション
（live_trade_log.txtから最新の判断を要約）

## ステータス
- ロック: xxx
- レジーム: xxx
- アラート: xxx
- shared_state鮮度: xxxZ更新 (N分前)

## 所感
（ポジションの健全性・リスク・注意点を1-2行）
```

### Step 4: ユーザーの指示に応じてスキルを自由に使う

ユーザーが何か言ったら、以下のスキル群から必要なものを自分の判断で呼び出して対応する。
明示的に「/スキル名」と言われなくても、文脈から最適なスキルを選んで実行する。

**注文・ポジション操作:**
- `/market-order` — 成行注文
- `/limit-order` — 指値注文
- `/close-all` — 全ポジ決済
- `/close-pair` — ペア指定決済
- `/partial-close` — 部分決済
- `/move-sl` — SL変更
- `/move-tp` — TP変更
- `/set-trail` — トレーリングストップ設定

**分析・判断支援:**
- `/mtf-dashboard` — マルチタイムフレーム一覧
- `/regime-check` — レジーム判定
- `/key-levels` — 主要レベル算出
- `/correlation-matrix` — 通貨相関
- `/volatility-scan` — ボラティリティスキャン
- `/setup-scanner` — セットアップスキャン
- `/exit-advisor` — 決済タイミング助言
- `/risk-score` — リスクスコア算出
- `/position-sizer` — ポジションサイズ計算

**管理・レポート:**
- `/daily-report` — 日次レポート
- `/weekly-review` — 週次レビュー
- `/trade-journal` — トレード日誌記録
- `/drawdown-alert` — ドローダウンチェック
- `/agent-health` — エージェント稼働状態
- `/force-unlock` — ロック強制解除
- `/backtest` — バックテスト

**ユーティリティ:**
- `/memo` — メモ記録
- `/timer` — タイマー
- `/relay-message` — エージェント間メッセージ
- `/add-indicator` — インジケーター追加
- `/playbook-update` — プレイブック更新
- `/translate-summary` — 翻訳サマリー
- `/devils-advocate` — 反対意見生成
- `/event-calendar` — 経済イベント確認

**使い方の例:**
- 「ドル円閉じて」→ `/close-pair` 実行
- 「SLもうちょい広げて」→ `/move-sl` 実行
- 「今日どうだった？」→ `/daily-report` 実行
- 「エージェント大丈夫？」→ `/agent-health` 実行
- 「ユーロドル買いたい」→ `/position-sizer` → `/market-order` 連携実行
- 「リスク高くない？」→ `/risk-score` + `/drawdown-alert` 併用

## トーン

簡潔・的確。データベース。余計な前置きなし。
