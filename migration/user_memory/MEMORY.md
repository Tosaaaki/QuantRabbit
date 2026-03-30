# Memory Index

## アーキテクチャ・戦略
- [project_architecture_v2.md](project_architecture_v2.md) - **v8アーキテクチャ**: trader一本の裁量トレードシステム。旧遺産全アーカイブ済 (2026-03-26)
- [project_trading_strategy.md](project_trading_strategy.md) - Claude裁量スキャルプ手法ルール・成功/失敗パターン (2026-03-17~)
- [project_sl_free_strategy.md](project_sl_free_strategy.md) - SL-free戦略の根拠: ボット実績と裁量棚卸しの理論射程

## トレード実行フィードバック
- [feedback_trading_approach.md](feedback_trading_approach.md) - 常時監視・ファイルログ・裁量判断・ボット化禁止
- [feedback_trend_switch.md](feedback_trend_switch.md) - H1転換時に即バイアス切替。データと矛盾したら立ち止まる
- [feedback_prediction_first.md](feedback_prediction_first.md) - 予測ファースト原則。Claudeの裁量予測がエッジ。スプレッド考慮必須
- [feedback_scalp_execution.md](feedback_scalp_execution.md) - スキャルプ3大失敗: 利確遅い・サイズ過大・リベンジ
- [feedback_scalp_rotation.md](feedback_scalp_rotation.md) - スキャルプ高速回転設計。HOLD禁止
- [feedback_hedging.md](feedback_hedging.md) - OANDA両建て: 反対ポジションは追加マージン0。MTF階層分離でH1テーゼ維持+M5回転
- [feedback_adaptive_indicators.md](feedback_adaptive_indicators.md) - 指標は固定セットでなく状況で適宜選択。組み合わせの効果を記録して自律成長

## ユーザーインターフェース
- [feedback_trade_session.md](feedback_trade_session.md) - 「トレード開始」で即セッション開始
- [feedback_secretary.md](feedback_secretary.md) - 「秘書」で秘書モード即応答
- [feedback_secretary_live_first.md](feedback_secretary_live_first.md) - 秘書モードではOANDA APIライブデータを最初に取得
- [feedback_collab_trade_trigger.md](feedback_collab_trade_trigger.md) - 「共同トレード」→ `collab_trade/CLAUDE.md` を読んで即開始

## 運用・プロセス
- [feedback_no_bot_trading.md](feedback_no_bot_trading.md) - 常駐ボット禁止。Claude自作の道具(スクリプト/Scheduled Task)は自由
- [feedback_deploy_immediately.md](feedback_deploy_immediately.md) - コード変更したら即デプロイ。聞くな
- [feedback_memory_discipline.md](feedback_memory_discipline.md) - 変更時は必ずメモリ更新+CHANGELOG.md記録
- [feedback_keep_strategy_updated.md](feedback_keep_strategy_updated.md) - 戦略変更のたびにproject_trading_strategy.mdを即更新
- [feedback_analysis_breadth.md](feedback_analysis_breadth.md) - 分析を広く。Ichimoku/VWAP/Fib/相関/マクロ全部使う

## 設計思想
- [feedback_prompt_design.md](feedback_prompt_design.md) - プロンプト改善パターン
- [feedback_prompt_design_v2.md](feedback_prompt_design_v2.md) - ルール→手法集転換、裁量重視設計原則
- [feedback_self_improvement.md](feedback_self_improvement.md) - 自問・自己改善・思考の広がり設計
- [feedback_task_reliability.md](feedback_task_reliability.md) - タスク信頼性: ロック/データ鮮度/並列プロセス管理

## 過去のセッション記録
- [project_trade_session_20260318.md](project_trade_session_20260318.md) - 2026-03-18セッション記録・失敗分析

## 構想（未実装）
- [project_alert_system_idea.md](project_alert_system_idea.md) - アラート駆動トレード構想
