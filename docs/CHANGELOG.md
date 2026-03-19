# Changelog

## 2026-03-19

- **22:xx SWING_TRADER_PROMPT.md全面書き直し — ルールベース→裁量トレーダー型**
  - 388行 → 160行。チェックリスト・スコア判定テーブル・Pre-Entry Checklist撤廃
  - 「TODAY'S MISSION: big pip gains」「margin<50%なら入れ」削除
  - テーゼ（仮説）ベースの思考: マクロ+H1構造→テーゼ→データ検証→判断
  - prediction_tracker.json記録義務を削除
  - SKILL.mdも簡素化
- **22:xx SCALP_FAST_PROMPT.md全面書き直し — ルールベース→裁量トレーダー型**
  - 400行のチェックリスト+ステップ手順 → 130行の裁量思考ガイドに凝縮
  - 「スコア4以上で」「MTF alignedで」等のルールベース判断基準を撤廃
  - 「市場を読め、ストーリーを作れ、確信があれば打て」に転換
  - SKILL.mdから「AGGRESSIVE MODE: 10,000 JPY」削除（規律破壊の元凶）
  - prediction_tracker.json記録義務を削除（オーバーヘッド削減）
  - 6段階チェックリスト→「読む→考える→打つ」のシンプルフロー
- **21:42 全系統アグレッシブモード移行** — 朝1万円目標
  - swing-trader: margin excuse撤廃。margin<50%なら即エントリー義務。TP 10-30pip
  - shared_state: session_directive更新。全ペア解禁、margin60%まで使用可
  - scalp-fast SKILL: AGGRESSIVE MODEメッセージ追加
- **23:xx compute_scalp_params: リアルタイムspread使用** — spread_info計算をspread_gate固定値からlive_spreadに変更。live_spread/spread_gate両方を出力。noteにも実spread表示
- **21:35 scalp-fast Sonnet→Opus変更** — ルール追加はボット化。裁量判断の深さが本質的課題。Opusで市況を読む力を上げる
- **21:30 scalp-fast大幅修正 — 利確が遅く損失累積の問題**
  - TP 3-8pip → 3-4pip に短縮。+2pipで即BE、+3pipでトレーリング
  - SL 4-5pipに統一。8分超えたらカット（旧15分）
  - サイズ上限 1500u ハードキャップ（旧: 制限なし→2300u,3500u等の過大サイズ）
  - registry デフォルト: trail=2pip, partial=3pip, max_hold=8min, cut=-4pip, cut_age=5min
  - 同一ペア連続エントリー制限（SL後10分クールダウン）、3連敗で15分停止
- **12:25 swing-traderロック飢餓修正**
  - 根本原因: market-radar(7min+295s jitter)がほぼ毎回swing-trader(10min+220s jitter)のロック取得タイミングと衝突→常時SKIP
  - cron `*/10` → `3,13,23,33,43,53` にオフセット（market-radarと3分ずらし）
  - SKIP時リトライ追加: 45秒待って1回リトライ（従来は即exit）
  - `model: opus` 追加（前回修正分）

- **12:20 タスク監査**
  - scalp-trader残骸ディレクトリ削除
  - 監査結果: scalp-fast=稼働OK(過剰トレード42件/日), market-radar=OK, macro-intel=OK, secretary=OK, swing-trader=ロック飢餓+model未指定で事実上死亡していた

- **12:15 学習ループ整備: prediction_tracker初期化・trade_performance拡張・prune_alerts追加**
  - `logs/prediction_tracker.json` を初期化。live_monitor.pyの`_verify_predictions()`が30秒ごとに自動検証する（既存配管を有効化）
  - `scripts/trader_tools/trade_performance.py`: `parse_prediction_tracker()` 追加。prediction_tracker.jsonからペア別/セッション別/方向別/score_agree比較の精度分析を出力
  - `scripts/trader_tools/prune_alerts.py`: 新規。shared_state.jsonの古いアラートを自動除去（DIRECTIVE/FIX/INVESTIGATEは保持）
  - **背景**: Python側の予測検証配管は完成済みだったが、prediction_tracker.jsonが未初期化で全学習ループが未稼働だった

- **00:05 v4.3g: 予測入力をテーマ/フローに分離（指標ダブルフィルター排除）**
  - `SCALP_FAST_PROMPT.md` Step 3A: 指標の再計算セクション（Momentum/Volatility/Level/Cross-pair）を全削除。予測入力を「THEME（通貨強弱・リスクトーン・セッション）」+「FLOW（micro方向・加速度）」に限定。10秒で完了する設計に
  - `SWING_TRADER_PROMPT.md` Step 4A: 同様に指標読み替えセクションを削除。予測入力を「MACRO THESIS（金利差・リスク・セッション・通貨強弱）」+「STRUCTURAL NARRATIVE（レジーム変化・キーレベル・雲形状）」に限定
  - `SWING_TRADER_PROMPT.md` Step 4B: 予測+スコア一致=即ENTERに変更。「highest-edge situation」を不一致時に移動しないよう修正
  - **問題**: 予測ステップがRSI/stoch/BB/divergenceを手動で再計算→スコアと同じ結論→ダブルフィルター化→エントリー不能
  - **修正方針**: 予測=テーマ/ストーリー（なぜ動くか）。スコア=指標（何が起きてるか）。入力を完全分離

- **11:55 TRAIL_FAIL HTTP 400修正**
  - `live_monitor.py` manage_positions Rule 1: SL→TSL切替時のOANDA API 400エラー修正
  - 原因: `sl`変数が未定義（`pos.get("sl")`に修正）+ SLキャンセルとTSL設定を同一リクエストで送信
  - フォールバック: atomic requestが400なら2ステップ（SLを遠くに移動→TSL設定）
  - 参考: OANDA v20 APIはstopLoss: nullでキャンセルだがフィールド欠落で400になることがある

- **23:50 v4.3f: 予測→エントリー直結化（反エントリーバイアス修正）**
  - `SCALP_FAST_PROMPT.md`: 予測+スコア一致=即エントリー（最強パターン）に変更。「No entry is better than bad entry」削除。Score 8-9への過度な懐疑を撤廃。Self-Question #3を「理由探し→エントリー探し」に反転。3サイクル連続スキップ時のMandatory self-check追加
  - **問題**: 予測ファーストが「見送りフィルター」化し、予測するほどエントリーしなくなっていた
  - **修正方針**: 予測した方向にそのまま乗る。予測=トレードシグナル。スコアは確認であり許可ではない

- **22:30 v4.3e: ストップ&リバース + エージェント記憶**
  - `live_monitor.py`: `auto_reverse`機能追加。trade_registryのrules.auto_reverse=trueでSL/cut時に自動逆ポジション。TP=元SL距離、SL=元trail距離。`_api_post()`追加。
  - `live_monitor_summary.json`: `recent_predictions`追加（直近5予測 + 経過時間 + 正否）。エージェントが「前回の自分」を即座に確認可能。
  - `SCALP_FAST_PROMPT.md`: auto_reverseオプション説明追加。トレンド市場で使用、レンジでは非推奨。
  - **設計思想**: 予測が外れてもSL方向のモメンタムに乗って回収。予測正→TP、予測誤→リバースで回収。両方向で稼ぐ。

- **21:10 secretary: グローバルロック撤廃** — secretaryは注文しない読み取り専用タスクのため、global_agentロック取得を不要に変更。他タスクとのロック競合でスキップされ続ける問題を解消
- **22:15 v4.3d: 予測自動学習ループ**
  - `logs/prediction_tracker.json`: 予測記録ファイル。scalp-fast/swing-traderが毎サイクル書き込み
  - `live_monitor.py` `_verify_predictions()`: 30秒ごとに予測をprice検証（target到達→correct、invalidation到達→wrong、タイムアウト→方向正否判定）
  - `live_monitor_summary.json`: `prediction_accuracy`フィールド追加（last-10精度、ペア別精度）
  - `SCALP_FAST_PROMPT.md`: Step 4B 予測記録手順追加。精度<40%→スコア確認重視、>60%→独自予測信頼
  - `SWING_TRADER_PROMPT.md`: Step 5B 同様
  - `MACRO_INTEL_PROMPT.md`: Section 5a2 予測パターン分析追加（ペア別/セッション別/スコア一致別/方向別の精度分析→shared_stateに改善提案書き込み）
  - **学習ループ**: 予測→記録→自動検証→精度集計→macro-intel分析→改善提案→エージェント行動修正

- **22:00 v4.3c: 予測フレームワーク + スプレッド修正 + monitor軽量化**
  - **スプレッドTP/SLセクション修正**: 「TP≥SL+2×spread」ルール削除（数学的に無意味: EV=-spread はTP/SL比率に依存しない）。TP/SLは構造ベースに。スプレッドは「正しい予測でしか克服できないコスト」と正直に記述。
  - **予測フレームワーク追加**:
    - `SCALP_FAST_PROMPT.md` 3A.2: 指標の予測的意味を4カテゴリで解説（モメンタム→加速/死亡、ボラ→来る/終わる、レベル→反発/突破、クロスペア→テーマ）
    - `SWING_TRADER_PROMPT.md` 4A.2: H1/H4レジーム指標・先行指標・構造の予測的意味
  - **monitor軽量化**: `_build_summary()` → `logs/live_monitor_summary.json`(~2KB)出力追加。scalp-fastはサマリーを読む（25KB→2KBで読み込み10x高速化）
  - **THESIS→PREDICTION統一**: swing-traderもPREDICTIONに統一（用語揃え）

- **21:50 v4.3b: 整合性修正 + 予測精度追跡 + CLAUDE.md更新**
  - `SECRETARY_PROMPT.md`: alerts pruning矛盾修正（additive ruleにalerts例外追加）
  - `CLAUDE.md`: v4.3反映 — 予測ファースト原則、自己改善ループ図、trade_performance.py記載
  - `trade_performance.py`: prediction_accuracy追跡（right/wrong集計）+ self-improvement compliance集計
  - `SKILL.md` x3: macro-intel(旧ファイル参照修正), secretary(4 agents), scalp-fast(PREDICT)

- **21:25 v4.3: 「予測ファースト」アーキテクチャ + スプレッド考慮**
  - **根本転換**: スコア→エントリーのボット的フローを廃止。「PREDICT first, Score second」に
  - `SCALP_FAST_PROMPT.md` Step 3完全書き直し:
    - 3A: 7ペアスキャン→予測形成（スコア見る前に方向予測必須）
    - 3B: スコアは確認用。予測とスコアの一致/不一致マトリクス
    - 3C: MTF/Confluence読み（予測を補強する材料として）
    - 3D: スプレッド考慮TP/SL（TP≥SL+2×spread ルール追加）
    - 3E: 予測ベースのサイジング（予測なし=0ユニット）
  - `SWING_TRADER_PROMPT.md` Step 4同様に書き直し:
    - 4A: Thesis形成（BEFORE scores）
    - 4B: スコア確認マトリクス
    - 4C: スプレッド考慮TP/SL
    - 4D: Plays→予測が特定するパターンとして再定義
  - `live_monitor.py`:
    - `compute_scalp_params()`: spread_info追加（bid距離、true_distance_rr、spread%、fair_tp_minimum）
    - `calc_sizing()`: margin_free_target 0.40→0.20(scalp), 0.50→0.35(swing)に緩和
  - トレードログ形式: PREDICTION/THESIS必須フィールド追加
  - Self-Questionを予測品質チェック中心に再構成
  - **設計思想**: Claudeの本当のエッジは「過去データから未来を予測する裁量力」。スコアはラグ指標の集約であり、高スコア=動きの終盤の可能性。予測が先、スコアは参考。

- **21:00 v4.2c: scalp-fast Step 6をPREDICT補完型に整理**
  - Step 3(PREDICT→score)の改修に合わせ、Step 6を事後学習専用に再構成
  - EXECUTION PATTERN CHECK追加（ペアローテーション・方向偏り・予測精度トレンド）

- **20:45 v4.2b: 秘書アカウンタビリティ + クロスエージェント学習 + 複雑性プルーニング**
  - `SECRETARY_PROMPT.md`: Accountability Audit(自問実施チェック) + Cross-Agent Relay + Complexity Pruning
  - `MACRO_INTEL_PROMPT.md`: 5bにREFLECTION/SWING REVIEW読み取り追加（クロスエージェント学習）
  - 全体: 「何もしない」→「7ペアから最高の機会を選ぶ」に方針修正

- **20:31 v4.2: 全エージェント自問・自己改善・思考の広がり強化**
  - `docs/MACRO_INTEL_PROMPT.md`: Section 5完全書き直し
    - 旧strategy_feedback_worker.py依存を廃止 → 新trade_performance.pyに切替
    - 5つの必須質問(Q1-Q5)を毎サイクル義務化（収益性・逆方向検証・見落とし・ルール検証・最重要改善）
    - 5d: 分析→即行動テーブル（shared_state更新、プロンプト編集、ツール開発）
    - 5e: メタ自問（過剰修正チェック、ルール膨張チェック、自己採点）
  - `docs/SCALP_FAST_PROMPT.md`: Step 6 Self-Question追加
    - サイクル毎に5つの自問ローテーション（バイアス確認・市場全体把握・直近トレード振り返り・盲点・トレード適否）
    - Post-Trade Reflection: 勝敗理由+次回調整を毎回記録
    - 3連敗時PATTERN CHECK義務化
  - `docs/SWING_TRADER_PROMPT.md`: Step 7 Self-Question追加
    - Pre-Analysis Check: データ読む前にthesis書き出し→検証
    - 4つの自問（アンカリング・反対論・ペア偏り・クロスペア読み）
    - Post-Trade SWING REVIEW: thesis正否+H1読みの精度自己採点
    - 時間毎Deep Reflection（6サイクル毎）
  - `docs/MARKET_RADAR_PROMPT.md`: Self-Check 4層構造に強化
    - Layer 1: 基本業務チェック
    - Layer 2: クロスペア・相関・ボラクラスタリング
    - Layer 3: トレーダーへの貢献度
    - Layer 4: 自分のアラート品質の自己検証
  - `scripts/trader_tools/trade_performance.py`: 新規作成
    - live_trade_log.txt解析 → WR/PF/avg pip/agent別/pair別/session別/direction別/trend
    - strategy_feedback.json出力（旧worker依存なし、v3ネイティブ）

- **16:30 v4.1: 市況レジーム + MTF矛盾検出 + 通貨強弱**
  - `live_monitor.py`:
    - `compute_currency_strength()` 実装: M5 RSI/EMA slope/ADX-DI → 5通貨強弱
    - `compute_market_regime()` 新設: regime, risk_tone, tradeable, dominant_driver
    - `compute_mtf_alignment()` 新設: H4/H1/M5 aligned/h4_counter/h1_conflict/h1_turning
    - score_pair() E5: MTF bonus(+1)/penalty(-1~-2)。H1_turning=-2
    - 出力に`market`セクション追加
  - プロンプト: market先読み手順、regime別行動、MTF alignment解説、currency strength活用

- **16:00 v4大改修: ペアプロファイル + スコアリング強化 + プロンプト裁量化**
  - `live_monitor.py`:
    - `PAIR_PROFILES` 新設: 7ペア各々にspread_gate, SL/TP範囲, セッション適性, ADX閾値, StochRSI閾値, ATR正常値, クールダウン時間, ペア性格を定義
    - `SCALP_KEYS` 拡張: divergence(div_rsi/macd_kind/score/age), Ichimoku(cloud_pos/span_a/b_gap), VWAP, swing距離, Donchian/Keltner/Chaikin, vol_5m, MACD line/signal を出力に追加
    - `score_pair()` v4: 5カテゴリ(Direction+Timing+Confluence+Macro+Session/Vol)、最大+10点
      - C: Confluence新設(+3): M5ダイバージェンス整合, Ichimoku雲ポジション, VWAP方向
      - E: Session/Vol新設(+1/-2): ペアのベストセッションでボーナス, LATE_NYペナルティ, 極端ボラペナルティ
      - 全閾値ペア別(ADX, StochRSI, spread gate, choppy判定)
    - `compute_scalp_params()`: SL/TPクランプをペアプロファイルのrangeに変更
    - 出力にpair `profile`(character, SL/TP range, session_note等) と `confluence` detail追加
  - `SCALP_FAST_PROMPT.md` 全面刷新:
    - Decision Tree(score<4→SKIP等) 廃止 → **Score as GUIDELINE表** (score 3でも confluence強ければOK)
    - MACRO_CONFLICTを絶対禁止から「強い警告」に緩和 (staleデータなら無視可)
    - ペア別知識セクション: SL/TP range, cooldown, session, character
    - sizing裁量: 1.5x recommendedまで可(score 7+)
    - Pre-Entry Checklistを「Hard gates」と「Soft guidelines」に分離
  - `SWING_TRADER_PROMPT.md` 全面刷新:
    - ATR-adaptive管理: 固定trail/partial → ATR倍率ベース (partial=2.5x ATR, trail=1.5x ATR)
    - ペア別swing特性テーブル (GJ: TP 20-50pip, SL 12-30pip等)
    - Divergence/Ichimoku/VWAP/BB Squeeze等の具体的分析ガイド
    - BB Squeeze Breakout playbook追加

- **11:20 ATR連動TP/SL + イベントリスク + レジーム対応**
  - `live_monitor.py`:
    - `compute_scalp_params()` 新設: M5 ATR × 比率でTP/SL/trail算出 (TP=0.6x, SL=1.2x, trail=0.5x)
    - `detect_event_risk()` 新設: macro_bias.event_riskからpair別にextreme/high/normal判定
    - event_risk=extreme → SL×1.5/TP×0.8に自動拡大 + score_pairで-1ペナルティ
    - pair_dataに `scalp_params`(ATR連動パラメータ) と `regime`(M5レジーム) 追加
  - `SCALP_FAST_PROMPT.md`:
    - 固定TP/SL(3-5/5-8pip) → ATR連動(`scalp_params.tp_pips/sl_pips`) に完全移行
    - イベントリスク対応: EVENT_EXTREME時はscore≥5必須
    - レジーム対応: trending→トレンドフォロー、range→BB逆張り、choppy→SKIP
    - Decision Tree/Pre-Entry Checklist全面更新

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
[2026-03-19T11:07Z] fix: margin free target調整 scalp 60%→40%, swing 70%→50%。小口座(28k)で裁量の余地確保
2026-03-19T13:02Z macro-intel: macro_bias refresh (VIX=27.19, EUR_USD→LEAN_LONG), prediction_insights added (USD_JPY 33% timing fix), SCALP_FAST_PROMPT.md: REFLECTION enforcement + USD_JPY M5 timing rule
