# Changelog

## 2026-03-19

- **11:10 スコアリングv2 + エントリー判断フレームワーク**
  - `live_monitor.py` score_pair()完全改修:
    - 旧: trend(ADX) + RSI + micro + H1 + spread(常にtrue) = 最大5点 → スコア3で入れてノイズトレード量産
    - 新: Direction(H1+M5_DI+RSI)=+3 + Timing(stoch_RSI+BB_band)=+2 + Macro(aligned/conflict)=+1/-2 + Penalty(choppy)=-1
    - マクロconflict時は-2点 → 事実上エントリー不可能に（EUR LONG + LEAN_SHORT = 自動ブロック）
    - spread<2pipはスコアではなくgate（pass/fail）に変更
    - micro momentum（S5ノイズ）をスコアから除外
  - `live_monitor.py`: load_macro_bias()新設、shared_state.jsonのmacro_biasをスコアリングに統合
  - `SCALP_FAST_PROMPT.md` 根本改修:
    - 冒頭: 「10+ラウンドトリップ」→「3-8 quality trades」、量→質
    - Step 3: スコア読み方ガイド追加（Direction/Timing/Macro/Penaltyの解説）
    - エントリー判断Decision Tree追加（score<4, MACRO_CONFLICT, TIMING無し, cooldown, SL>8pip → SKIP）
    - 最低スコア: 3→4に引き上げ
    - Pre-Entry Checklist: MACRO_CONFLICT禁止、TIMING信号必須、同一ペアcooldown追加

- **11:00 trade type推定3レイヤー防御**
  - `live_monitor.py`: 1)registry custom rules → 2)OANDA clientExtensions.tag → 3)SL距離推定 の3段階でtrade type決定
  - `live_monitor.py`: positions出力に `inferred_type`, `rules_source`, `sl_pips`, `tag`, `comment` 追加（デバッグ可視化）
  - `live_monitor.py`: actions_takenに `[rules_source]` 表示追加
  - `SCALP_FAST_PROMPT.md`: 注文に `clientExtensions: {tag: "scalp"}` 必須化、registry登録をMANDATORYに格上げ
  - `SWING_TRADER_PROMPT.md`: 注文に `clientExtensions: {tag: "swing"}` 必須化、3レイヤー管理の説明追加
  - `SECRETARY_PROMPT.md`: scalp-trader→scalp-fast/swing-trader更新、position tagging監視項目追加

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
[2026-03-19T10:50Z] update: 秘書スキルにスキル自作機能追加。既存スキルで対応不可なら/skill-creatorでその場で新スキル作成→即実行
[2026-03-19T11:00Z] fix: live_monitor.py — 6つの重大改善
  1. compute_max_units(): NAV+マージン+ATR+リスクベースの動的ポジションサイズ計算。sizing出力をmonitor JSONに追加
  2. recently_closed追跡: logs/recently_closed.json で10分間の閉じたトレードIDを記録。重複クローズ防止
  3. manage_positions(): OANDA 404エラー（既に閉じたトレード）をgracefulにキャッチ→mark_closed
  4. USD/JPYレート動的取得: ハードコード159を廃止、pricingから取得
  5. SCALP_FAST_PROMPT.md全面改訂: Pre-Entry Checklist追加、sizing.recommended_units必須化、recently_closedチェック必須化
  6. SWING_TRADER_PROMPT.md改訂: Pre-Entry Checklist追加、sizing必須化、重複クローズ防止追加
