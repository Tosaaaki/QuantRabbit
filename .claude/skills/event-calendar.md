---
name: event-calendar
description: "今日〜明日の経済指標・中銀イベントをWeb検索で取得。ポジション保有中のリスク管理に。"
trigger: "Use when the user says 'イベント', 'event', '指標', '経済カレンダー', 'calendar', 'ニュース', or asks about upcoming economic events."
---

# イベントカレンダースキル

## 実行手順

### Step 1: Web検索

以下のソースから経済指標・イベント情報を取得:
- Web検索: "forex economic calendar today [date]"
- 主要ソース: ForexFactory, Investing.com, DailyFX

### Step 2: フィルタリング

対象通貨: USD, EUR, GBP, AUD, JPY（トレード対象ペアに関連するもの）
重要度: 🔴高インパクト を優先。🟡中も表示。🟢低は省略。

### Step 3: 出力

```
## 📅 経済カレンダー

### 今日 (2026-03-19)
| 時間(JST) | 通貨 | イベント | 重要度 | 前回 | 予想 | ポジ影響 |
|-----------|------|---------|--------|------|------|---------|
| 12:00 | GBP | BOE金利決定 | 🔴 | 4.50% | 4.50% | GU/GJ注意 |
| 21:30 | USD | 新規失業保険 | 🟡 | 220K | 218K | UJ/EU |
| 23:00 | USD | 中古住宅販売 | 🟡 | 4.08M | 3.95M | UJ |

### 明日 (2026-03-20)
| 時間(JST) | 通貨 | イベント | 重要度 |
...

### ⚠️ 現在ポジションへの影響
- AUD_USD SHORT 1000u:
  - 直近AUD関連指標なし → 影響低
  - USD指標21:30 → ボラ注意、SL確認推奨

### 推奨アクション
- BOE 12:00前にGBP/USD, GBP/JPYポジションがあれば要判断
- 21:30前にUSD関連ポジのSL確認
```
