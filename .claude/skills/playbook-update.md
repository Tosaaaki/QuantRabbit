---
name: playbook-update
description: "成功/失敗トレードからPlay集を自動更新。戦略の学習ループを回す。"
trigger: "Use when the user says 'プレイブック', 'playbook', 'play更新', '戦略更新', or asks to update trading plays/strategies."
---

# プレイブック更新スキル

## 実行手順

### Step 1: データ収集

1. OANDA API: 直近のクローズ済みトレード
2. logs/live_trade_log.txt: エントリー判断根拠とPlay名
3. docs/trade_journal/: 日記・振り返り
4. memory/project_trading_strategy.md: 現在の教訓

### Step 2: Play成績集計

各Playの成績を集計:

```
## 📚 プレイブック更新

### Play成績ランキング
| Play | トレード数 | 勝率 | 平均R:R | PF | ステータス |
|------|-----------|------|---------|-----|-----------|
| H1 Bear + M5 Peak | 12 | 67% | 1.4:1 | 2.8 | ✅ACTIVE |
| VWAP Reversion | 8 | 63% | 1.2:1 | 2.0 | ✅ACTIVE |
| RSI Extreme Counter | 6 | 33% | 0.8:1 | 0.4 | ❌RETIRE |
| Breakout Chase | 4 | 25% | 0.6:1 | 0.2 | ❌RETIRE |
```

### Step 3: 更新アクション

#### 成功Play → 強化
- 条件を精緻化（どのセッション、どのボラ環境で最も効くか）
- scalp-traderプロンプトでの優先度を上げる

#### 失敗Play → 改善 or 廃止
- PF < 1.0 が5トレード以上 → 廃止候補
- 条件追加で改善できるか検討（セッションフィルター等）

#### 新Play → 追加
- トレード日記から成功パターンを抽出
- 条件を言語化してプレイブックに追加

### Step 4: 反映

1. `docs/SCALP_TRADER_PROMPT.md` のPlay集セクションを更新
2. `memory/project_trading_strategy.md` に教訓追記
3. shared_state.jsonに更新通知

```
✅ プレイブック更新完了

変更:
- RETIRED: RSI Extreme Counter (勝率33%, PF=0.4)
- UPDATED: H1 Bear + M5 Peak — ロンドン時間限定に変更 (勝率67%→推定75%)
- NEW: "Session Open Momentum" — ロンドン開場30分の方向にフォロー
```
