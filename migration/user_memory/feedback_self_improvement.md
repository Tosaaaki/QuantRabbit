# エージェント自問・自己改善・思考の広がり設計 (2026-03-19)

## 設計原則
- 全エージェントに自問セクションをMandatoryで組み込む
- 分析だけでなく「行動」まで含める（shared_state更新、プロンプト編集、ツール開発）
- メタ自問で「改善が改善になっているか」を検証する再帰構造

## エージェント別の自問設計

### scalp-fast (Step 6)
- 5つの自問ローテーション（2個/サイクル）
- Post-Trade Reflection: 勝敗理由+調整を毎回記録
- 3連敗でPATTERN CHECK義務化

### swing-trader (Step 7)
- Pre-Analysis Check: thesis先出し→データで検証（確証バイアス防止）
- 4つの自問（アンカリング・反対論・ペア偏り・クロスペア物語）
- SWING REVIEW: thesis正否+H1読み精度
- 時間毎Deep Reflection

### market-radar (Self-Check 4層)
- Layer 1: 基本業務 / Layer 2: 全体像 / Layer 3: トレーダー貢献 / Layer 4: アラート品質

### macro-intel (Section 5)
- 5つの必須質問（Q1-Q5）を毎サイクル
- 即行動テーブル（分析→アクション直結）
- メタ自問（過剰修正・ルール膨張・自己採点）

### secretary (Section 5-7)
- Accountability Audit: 各エージェントのREFLECTION/REVIEW/Q1-Q5実施を検証
- Cross-Agent Feedback Relay: エージェント間の学びをshared_stateで中継
- Complexity Pruning: 1時間毎にルール膨張・矛盾チェック。アラート5個超→古いの削除

## クロスエージェント学習ループ
```
scalp-fast → REFLECTION: → live_trade_log.txt
swing-trader → SWING REVIEW: → live_trade_log.txt
macro-intel → reads reflections → identifies patterns → updates shared_state/prompts
secretary → audits compliance → relays findings → prunes complexity
```

## 重要原則
- 「何もしない」は選択肢にない。7ペアあれば常にチャンスがある
- 問題は「トレードするか否か」ではなく「最高の機会を選んでいるか」
- ルールは増やすだけでなく定期的に削る（secretary Complexity Pruning）

## trade_performance.py
- v3ネイティブ: live_trade_log.txt → パフォーマンス統計
- 旧strategy_feedback_worker.py（trades.db依存）を置換
- macro-intelが毎サイクル実行
