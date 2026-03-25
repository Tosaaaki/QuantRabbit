---
name: memory-recall
description: "memory.dbからトレード記憶をベクトル検索。過去の類似トレード・ユーザー助言・教訓を引き出す。"
trigger: "Use when user says '/memory-recall', '記憶検索', 'recall', or when you need to search past trading sessions for similar patterns, lessons, or user calls."
---

# メモリ検索

memory.db（SQLite + sqlite-vec）から過去のトレード記憶を検索する。

## ハイブリッド検索（ベクトル + キーワード）

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 recall.py search '<自然言語クエリ>' --top 5
```

### オプション

- `--pair USD_JPY` — 通貨ペアでフィルタ
- `--type trade` — チャンクタイプでフィルタ（trade / user_call / thesis / lesson / summary / market_read）
- `--top 5` — 結果数

### 使用例

```bash
# GBPで損した記憶
python3 recall.py search 'GBPのスパイクで大損' --pair GBP_USD

# ユーザーの相場読みパターン
python3 recall.py search 'ユーザーがドルスト上がると言った' --type user_call

# 損切りの教訓
python3 recall.py search '損切り遅れた教訓' --type lesson

# キーワード検索
python3 recall.py keyword 'スパイク' --pair GBP_USD
```

## 共同トレード開始時の自動検索

セッション開始時に、保有ペアや今日の状況に関連する過去の記憶を自動で引くこと。
例: EUR_USD LONG を持っている → `search 'EUR_USD ロング 教訓' --pair EUR_USD`
