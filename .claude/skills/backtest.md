---
name: backtest
description: "指定戦略をヒストリカルデータで簡易バックテスト。勝率・R:R・PF等を算出。"
trigger: "Use when the user says 'バックテスト', 'backtest', '検証', or asks to test a strategy on historical data."
---

# バックテストスキル

## 使い方

- 「UJ H1ベア+M5バウンスピークのバックテスト、直近1週間」
- 「RSI<25逆張りLONGの検証、EU過去2週間」
- 「今のPlay全部、勝率検証して」

## 実行手順

### Step 1: ヒストリカルデータ取得

OANDA APIからキャンドルデータ:
```
GET /v3/instruments/{pair}/candles?granularity={tf}&from={start}&to={end}&count=500
```

### Step 2: 戦略ルール定義

ユーザー指定のエントリー/イグジットルールをコード化:
```python
# 例: H1 Bear + M5 Bounce Peak
entry_condition = (
    h1_adx > 20 and h1_minus_di > h1_plus_di  # H1 Bear
    and m5_stoch_rsi > 0.9  # M5 overbought
    and m5_last_candle_bearish  # M5 反転
)
direction = "SHORT"
sl_pips = 25
tp_pips = 35
```

### Step 3: シミュレーション実行

各エントリーポイントでSL/TPどちらに先に到達したか判定。
スリッページ1pip、スプレッド考慮。

### Step 4: 出力

```
## 🔬 バックテスト結果

### 条件
| 項目 | 値 |
|------|-----|
| 戦略 | H1 Bear + M5 Bounce Peak SHORT |
| ペア | USD/JPY |
| 期間 | 2026-03-12 〜 03-19 (5営業日) |
| SL | 25pip | TP | 35pip |

### 結果
| 指標 | 値 |
|------|-----|
| シグナル数 | 23 |
| 勝率 | 61% (14W/9L) |
| 平均勝ち | +33.2pip |
| 平均負け | -24.1pip |
| Profit Factor | 2.13 |
| 期待値 | +11.5pip/trade |
| 最大連敗 | 3 |

### トレード一覧
| # | 日時 | Entry | Exit | 結果 | P&L |
...

### 📊 エクイティカーブ
(テキストグラフ)

### 所感
- 勝率61%は十分だがサンプルが少ない
- ロンドン時間帯の勝率が75%と高い
- 東京時間帯は40%と低い → セッションフィルター追加を推奨
```
