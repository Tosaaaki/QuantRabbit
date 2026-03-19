---
name: setup-scanner
description: "全通貨ペアをスキャンし、今セットアップが揃っているペア・方向・Play名を抽出。チャンス見逃し防止ツール。"
trigger: "Use when the user says 'スキャン', 'scan', 'セットアップ', 'チャンス', '何かある？', or asks what trades are available."
---

# セットアップスキャナースキル

全ペアを巡回して「今エントリーセットアップが揃ってるペア」だけ抽出する。

## 対象ペア

USD/JPY, EUR/USD, GBP/USD, AUD/USD, EUR/JPY, GBP/JPY, AUD/JPY

## 実行手順

### Step 1: データ取得（並列）

1. OANDA API: 全ペア最新価格
2. shared_state.json: レジーム・テクニカル・アラート
3. factor_cacheが利用可能なら指標データも取得

### Step 2: 各ペアのセットアップ判定

各ペアについて、以下のPlayに該当するか判定:

#### Trend Continuation Plays
| Play名 | 条件 |
|---------|------|
| **H1 Bear + M5 Bounce Peak** | H1=trending_bear + M5 StochRSI≥0.9→下落開始 |
| **H1 Bull + M5 Dip Bottom** | H1=trending_bull + M5 StochRSI≤0.1→上昇開始 |
| **ADX Surge** | H1 ADX>25かつ上昇中 + M5同方向 |
| **VWAP Reversion** | 価格がH1 VWAPに戻ってきて反発 |

#### Reversal Plays
| Play名 | 条件 |
|---------|------|
| **RSI Extreme Bounce** | H1 RSI<25 or >75 + M5で反転シグナル |
| **Fib 61.8 Touch** | 直近スイングのFib 61.8%に到達+反発 |
| **Session Reversal** | セッション切替時の方向転換パターン |

#### Momentum Plays
| Play名 | 条件 |
|---------|------|
| **Breakout** | レンジ上限/下限ブレイク + 出来高増加 |
| **JPY Cross Sweep** | JPYクロス全体が同方向に急伸 |
| **Correlation Divergence** | 通常相関ペアの乖離→収斂狙い |

### Step 3: コンフルエンス採点

各セットアップに対して:
- H4方向一致: +2
- H1方向一致: +2
- M5エントリーシグナル: +2
- RSI/StochRSI適正: +1
- ADX>20: +1
- キーレベル近接: +1
- マクロ追い風: +1

最大10点。**7点以上を推奨**、5-6点は条件付き。

### Step 4: 出力

```
## 📡 セットアップスキャン結果

### 🟢 推奨 (スコア7+)
| ペア | 方向 | Play | スコア | Entry目安 | SL | TP | R:R |
|------|------|------|--------|----------|-----|-----|-----|
| EUR/USD | SHORT | H1 Bear + M5 Bounce Peak | 8/10 | 1.1475 | 1.1510 | 1.1435 | 1.1:1 |

### 🟡 条件付き (スコア5-6)
| ペア | 方向 | Play | スコア | 条件 |
|------|------|------|--------|------|
| GBP/JPY | LONG | RSI Extreme Bounce | 6/10 | M5で反転確認待ち |

### ⚪ セットアップなし
USD/JPY (レンジ中, 方向不明), AUD/JPY (テクニカル混在)

### 現在のレジーム概要
| ペア | H1 | M5 | ADX |
|------|-----|-----|-----|
（全ペア一覧）
```

## ルール

- データがないのに「セットアップあり」と言うな。確認できた根拠だけで判定。
- R:R 1:1未満のセットアップは除外（スキャルプでも最低1:1）。
- 既にポジション保有中のペアは「保有中」と明記。
