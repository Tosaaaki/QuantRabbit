# Changelog

## 2026-03-19

- **10:30 v3アーキテクチャ移行**
  - `live_monitor.py` 拡張: シグナルスコアリング、機械的ポジ管理(trail/partial/close)、リスクチェック(circuit breaker)、セッション判定、通貨エクスポージャー
  - `trade_registry.json` 新設: ポジション所有権(scalp-fast/swing-trader)と管理ルール
  - `scalp-fast` タスク新設 (Sonnet, 2分, ロック無し): モニター読む→オーバーライド→エントリー
  - `swing-trader` タスク新設 (Opus, 10分, global lock): H1/H4深い分析→中期エントリー
  - 旧 `scalp-trader` 無効化
  - launchd設定: `com.quantrabbit.live-monitor.plist` (30秒間隔)

- **09:00 ポジション管理ワークフロー改善**
  - `SCALP_TRADER_PROMPT.md` にステップ2「Manage Open Positions」新設
  - 毎サイクル全ポジに"fresh entry test"を強制
  - trailing stop APIの具体例追加
  - ログテンプレートにポジション管理アクション欄追加

- **08:45 高速回転方針転換**
  - `SCALP_TRADER_PROMPT.md` 冒頭: 「Precision > Frequency」→「Speed > Perfection」
  - トレードスタイル: TP短縮、30分以上HOLD禁止
  - 教訓#11(+770→+91)、#12(trailing stop怠慢) 追加

- **10:40 CLAUDE.mdにドキュメントマップ追加**
  - 必読/運用/レガシーの分類、ランタイムファイル一覧、スクリプト一覧

## 2026-03-18

- 初回トレードセッション。教訓#1-10を追加
- 20 plays体系のプロンプト設計
[2026-03-19T10:45Z] update: 秘書スキルにスキル自由活用セクション追加。ユーザー指示に応じて33スキルを自律的に選択・実行する設計
