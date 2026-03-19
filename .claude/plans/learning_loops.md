# 学習ループ整備プラン

## 現状分析

Python側の配管は完備:
- `live_monitor.py` — `_verify_predictions()` が30秒ごとにprediction_tracker.jsonを検証
- `_build_summary()` — 予測精度統計をsummaryに含める
- `trade_performance.py` — `parse_prediction_accuracy()` がログからright/wrong集計

**欠けてるもの:**
1. `logs/prediction_tracker.json` ファイルが存在しない（初期化されてない）
2. `trade_performance.py` がprediction_tracker.jsonを読んでない（ログテキストのみ）
3. shared_stateのアラートpruning機構なし（蓄積する一方）

## 実装

### 1. prediction_tracker.json 初期化
- `logs/prediction_tracker.json` を `[]` で作成

### 2. trade_performance.py にprediction_tracker.json解析を追加
- prediction_tracker.jsonを直接読む新関数 `parse_prediction_tracker()`
- ペア別精度、セッション別精度、score_agree vs disagree の精度比較
- strategy_feedback.jsonに詳細な予測精度データを出力
- 既存の `parse_prediction_accuracy()` はログベースのまま維持（補完的に使う）

### 3. shared_state アラートpruning ユーティリティ
- `scripts/trader_tools/prune_alerts.py` — 1時間以上前のアラートを除去
- secretaryやmacro-intelがbashで呼べるように

### 変更しないもの
- プロンプト — Step 4B (SCALP_FAST) / Step 5B (SWING_TRADER) は既に正しく書かれてる
- live_monitor.py — 予測検証ロジックは完成済み
