# TASKS – Repository Task Board

本リポジトリの全タスクを一元管理する台帳。タスクが発生したら本ファイルへ逐次追記し、作業中は本ファイルを参照しながら進め、完了後はアーカイブへ移してください。詳細ポリシーは `AGENTS.md` の「11. タスク運用ルール（共通）」を参照。  
※ `micro_core` / `macro_core` / `scalp_core` と `core_executor` は廃止済み。本文に残るコア関連タスク/記述はレガシーとして扱ってください。  
※ 2025-12-26 時点で共通 `execution/exit_manager.py` はスタブ化（自動EXITなし）。Archive に残る ExitManager 改修タスクは履歴としてのみ参照してください。

## 運用ルール（要約）

- 発生: 本ファイル「Open Tasks」にテンプレートで追加。
- 進行: ステータス/Plan/Notes を更新し、PR/コミットには `[Task:<ID>]` を含める。
- 完了: 「Archive」へ移動し、完了日・PR/コミットリンク・要約を追記。
- 例外: オンライン自動チューニング関連は `docs/autotune_taskboard.md` を主に使用（必要に応じて相互リンク）。

## テンプレート

以下をコピーして使用:

```md
- [ ] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: todo | in-progress | review | done
  Priority: P1 | P2 | P3
  Owner: <担当>
  Scope/Paths: <例> AGENTS.md, docs/TASKS.md
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <受入条件1>
    - <受入条件2>
  Plan:
    - <主要ステップ1>
    - <主要ステップ2>
  Notes:
    - <補足>
```

## Open Tasks
- [ ] ID: T-20260122-007
  Title: EXIT コンテキストの保存（ローソク/指標との照合用）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: execution/order_manager.py, docs/TASKS.md
  Context: 損切り妥当性の検証で exit_reason とローソクの照合が困難。
  Acceptance:
    - close_request/close_ok に exit_context が保存される
    - M1/M5/H1/H4 の指標スナップショットが残る
  Plan:
    - order_manager で exit_context を付与
    - デプロイ後に orders.db で確認
  Notes:
    - 「ローソクと照らし合わせて」の検証に使う

- [ ] ID: T-20260122-006
  Title: テクニカル損切りの総合判定強化（reversal確認条件追加）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: analysis/technique_engine.py, docs/TASKS.md
  Context: `tech_candle_reversal` の不一致が多く、単一シグナルで損切りになっている可能性がある。
  Acceptance:
    - reversal 判定は combo または return_score で確認される
    - `tech_*` の損切りが単発判定にならない
  Plan:
    - technique_engine の reversal 判定に確認条件を追加
    - デプロイ後に orders.db の `tech_*` を監視
  Notes:
    - 「損切りは総合判断」を優先

- [ ] ID: T-20260122-005
  Title: テクニカル反転は損切り許可（tech_candle_reversal を遮らない）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: analysis/technique_engine.py, docs/TASKS.md
  Context: `close_reject_no_negative` に `tech_candle_reversal` が残り、テクニカル損切りが通らないケースがある。
  Acceptance:
    - 反転シグナル時にマイナスなら allow_negative が True になる
    - `close_reject_no_negative` の tech_* 起因が減る
  Plan:
    - technique_engine の反転判定で allow_negative を明示的に許可
    - デプロイ後に orders.db で tech_* の reject を確認
  Notes:
    - 「損切りはテクニカル」を優先

- [ ] ID: T-20260122-004
  Title: 損切りの総合判定強化（exit_emergency のマージン/DD反映）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: workers/common/exit_emergency.py, execution/order_manager.py, AGENTS.md, docs/TASKS.md
  Context: negative close が health buffer だけで詰まり、損切り判断が総合的になっていない。損益報告の取り違えも再発防止する。
  Acceptance:
    - health buffer / free margin / margin usage / unrealized DD のいずれかで negative close を許可できる
    - `exit_emergency_allow_negative` に理由タグが記録される
    - AGENTS に損益報告ルールを追記する
  Plan:
    - exit_emergency に複合判定を追加する
    - AGENTS の損益/マージン報告ルールを更新する
    - デプロイ後に `orders.db` の `close_reject_no_negative` を監視する
  Notes:
    - 「停止なし」条件を維持する

- [ ] ID: T-20260122-003
  Title: TP距離の異常拡大を抑止（signal gate/注文整合）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: execution/order_manager.py, docs/TASKS.md
  Context: signal gate 経由の min_rr 補正で TP が 70〜80p に拡大している。妥当なTP距離でクランプし、fast_scalpはSLヒントを送らない。
  Acceptance:
    - TP距離がポケット上限を超えない
    - fast_scalp の TP が min_rr で過大化しない
    - logs/orders.db で tp_cap_adjust が確認できる
  Plan:
    - order_manager に TP cap と fast_scalp の SLヒント除外を追加
    - デプロイ後に orders.db で TP 距離を再確認
  Notes:
    - 「停止なし」条件を維持

- [ ] ID: T-20260122-002
  Title: 非停止前提でのエントリー密度/勝ち幅改善（scalp中心）
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: workers/common/perf_guard.py, workers/scalp_m1scalper/config.py, workers/scalp_multistrat/config.py, workers/fast_scalp/config.py, execution/risk_guard.py, config/worker_reentry.yaml, docs/TASKS.md
  Context: 停止なしの条件で、勝ち幅とエントリー数の改善が必要。M1Scalperの再入場間隔が長く、fast_scalpの回転が足りない。
  Acceptance:
    - perf_guardはwarn運用で停止しない
    - M1Scalper/ScalpMultiのマージン上限を引き上げ、回転とサイズを改善
    - fast_scalpの回転制限を緩め、scalp_fast配分を増やす
  Plan:
    - M1Scalper/ScalpMultiのMAX_MARGIN_USAGEとreentry cooldownを調整
    - fast_scalpのオーダー回転制限とshare_hintを調整
    - 反映後にorders.db/trades.dbで改善兆候を確認
  Notes:
    - 「停止等はなし」を優先するため、perf_guardはwarnを既定化

- [ ] ID: T-20260122-001
  Title: Perf guard block化とscalp/s5系のマージン上限を攻め設定に統一
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: workers/common/perf_guard.py, workers/fast_scalp/config.py, workers/pullback_s5/config.py, workers/pullback_runner_s5/config.py, workers/mirror_spike_s5/config.py, docs/TASKS.md
  Context: 低勝率ストラテジがブロックされず損失が拡大し、fast_scalp/鏡反転/PB系が低いMAX_MARGIN_USAGEでほぼ停止している。
  Acceptance:
    - perf_guardがwarnではなくblockで低PF/低勝率戦略を停止できる
    - fast_scalp/鏡反転/PB系がマージン85-92%帯でもエントリー可能
    - 変更後のログでreentry_block偏重が緩和される
  Plan:
    - デフォルトMAX_MARGIN_USAGEの引き上げとperf_guardデフォルトの修正
    - デプロイ後にorders.db/journalctlで挙動を確認
  Notes:
    - 低勝率のM1Scalper/TrendMA等はperf_guardで自動停止させる

- [ ] ID: T-20260118-001
  Title: Factor cache stale ガードの監視と閾値調整（市場オープン後）
  Status: todo
  Priority: P1
  Owner: codex
  Scope/Paths: execution/order_manager.py, main.py, docs/weekly/2026-01-19_week.md, docs/TASKS.md
  Context: factor cache stale ガード/自動再シードを有効化済み。市場オープン後の過剰ブロック有無を確認し、必要なら閾値を調整する。
  Acceptance:
    - 市場オープン後に `orders.db` と `journalctl` で `factor_stale` / `FACTOR_CACHE` を確認し、過剰ブロックがない
    - stale が出る場合は `ENTRY_FACTOR_MAX_AGE_SEC` / `FACTOR_CACHE_STALE_SEC` / `FACTOR_CACHE_REFRESH_MIN_INTERVAL_SEC` を調整して再デプロイ
    - 週次ドキュメントに結果を記録する
  Plan:
    - 市場オープン後に `orders.db` と `journalctl` を確認
    - しきい値調整と再デプロイ（必要時のみ）
    - 週次ドキュメントに結果を追記
  Notes:
    - 週次ドキュメント: `docs/weekly/2026-01-19_week.md`

- [ ] ID: T-20260118-002
  Title: ワーカー別 return-wait の最終確定と再エントリー条件更新
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: analytics/worker_return_wait_report.py, config/worker_reentry.yaml, docs/hedge_plan.md, docs/weekly/2026-01-19_week.md, docs/TASKS.md
  Context: 取り残し対策として「戻り待ちが有利/不利」をワーカー別に確定し、再エントリー条件を更新する。
  Acceptance:
    - ワーカー別に保有時間分布/勝率/平均損益を算出し、`return_wait_bias` を最終確定
    - `config/worker_reentry.yaml` にクールダウン/同方向再入場閾値を反映
    - 週次ドキュメントに結果を記録する
  Plan:
    - `analytics/worker_return_wait_report.py` を使ってワーカー別の return-wait を集計
    - `config/worker_reentry.yaml` を更新し、挙動を監視
    - 週次ドキュメントに結果を追記
  Notes:
    - 取り残し/戻り待ちの判断は pocket ではなく worker 単位で扱う
    - same_dir_mode を追加して戻り待ち/追随の両対応を開始

- [ ] ID: T-20260118-003
  Title: entry_thesis フラグ保存と MFE/MAE/BE 時間分析
  Status: todo
  Priority: P2
  Owner: codex
  Scope/Paths: execution/order_manager.py, execution/position_manager.py, analytics/*, docs/hedge_plan.md, docs/weekly/2026-01-19_week.md, docs/TASKS.md
  Context: entry_thesis の `entry_guard_*` / `trend_bias` が履歴に残らず差異分析が難しい。MFE/MAE と BE までの時間も未分析。
  Acceptance:
    - `entry_guard_*` / `trend_bias` などのフラグが `entry_thesis` に確実に残り、trades.db 解析で利用できる
    - MFE/MAE と BE 到達時間をワーカー別に出力できる（分析スクリプト）
    - 週次ドキュメントに結果を記録する
  Plan:
    - entry_thesis の保存/復元の欠落箇所を特定し修正
    - MFE/MAE/BE 時間を集計するスクリプトを追加
    - 週次ドキュメントに結果を追記
  Notes:
    - 差異分析のボトルネック解消が主目的

- [ ] ID: T-20251209-001
  Title: BQ strategy scores → Firestore snapshot (read-only反映)
  Status: todo
  Priority: P1
  Owner: codex
  Scope/Paths: analytics/*, scripts/*, config/*, docs/TASKS.md
  Context: リアルタイム売買に影響を与えずに利益直結のロット/SLTP調整を回すため、BQ 集計結果を Firestore に小さく保存し、VM 側は短 TTL キャッシュで読むだけの仕組みを入れる。
  Acceptance:
    - BQ 上の strategy×pocket×regime 集計からスコア/ロット係数/SLTPプリセットを JSON 化し、Firestore `strategy_scores/current` に上書きできる（1MB 未満）
    - VM 側で Firestore 読み取りが失敗しても既定値/最終キャッシュでトレード継続し、適用有無がログ/metrics に残る
    - 適用対象はロット係数と SLTP プリセットのみで、エントリー頻度/クールダウン/時間帯ゲートを変更しないことが確認できる
  Plan:
    - BQ 集計ビュー/クエリを用意し、strategy×pocket×regime の PF/Sharpe/score と SLTP 推奨値を算出するスクリプトを作成
    - Firestore へサマリ JSON を書き出すバッチ（Cloud Scheduler/cron 想定）を実装し、1 ドキュメント 1MB 未満で履歴は GCS/BQ に残す
    - VM 側に TTL キャッシュ付きリーダーを追加し、ロット係数/SLTP プリセットのみ反映するフラグを設定ファイル経由で切替できるようにする
  Notes:
    - まずは読み取りのみでログ確認し、安全性が確認できたら適用フラグを ON にする二段階ロールアウトにする

- [ ] ID: T-20251127-004
  Title: Pattern-based lot boost from candle/tech win-rate
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: analysis/pattern_stats.py, main.py, docs/TASKS.md
  Context: Scale lots up when historically strongローソク/テクニカルパターンが出現した際に信頼度を反映したい。trades.db のエントリー履歴からパターン別勝率を学習し、rangeモードでは抑制する。
  Acceptance:
    - 新規トレードの entry_thesis に candle/indicator 由来の `pattern_tag` / `pattern_meta` が保存される
    - trades.db を集計した pattern stats からサンプル数閾値以上でのみブースト係数を返し、rangeモード時は上限を縮小する
    - main ループで係数が適用され pocket 配分を超えずにロットが拡張され、metric/log で factor・pattern・samples が確認できる
  Plan:
    - pattern signature builder と trades.db 集計キャッシュを実装し、lookback/サンプル数ガードを付ける
    - main のロット計算に pattern boost を組み込み、range 上限を守りつつ entry_thesis にメタ情報を保存する
    - ドライラン/ログで係数と clamp を確認し、しきい値を微調整 or 無効化できるようにする
  Notes:
    - refresh 間隔は perf 更新と同じ 5 分で、ホットパスに負荷を掛けない

- [ ] ID: T-20251212-002
  Title: Expand technical composite for lot/entry/exit (scalp→micro→macro)
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: indicators/*, main.py, strategies/scalping/*, execution/exit_manager.py, docs/TASKS.md
  Context: テクニカル/パターンをlot調整・エントリー・EXITに広く使い、信頼度に応じてサイズ/粘りを動的に変える。まずscalpで導入し、検証後にmicro/macroへ拡大する。
  Acceptance:
    - facにMACD/ROC/CCI/StochRSI/DMI/EMAスロープ/Ichimoku/Keltner/Donchian/ChaikinVol/VWAP乖離/高安クラスタ距離が供給される
    - scalp戦略でコンポジットスコアによりconfidence/lot/TPを0.7〜1.3倍スケールし、range/トレンドで挙動が変わる
    - EXITでMACD/DMI/ボラ/パターン(N波/ローソク)を用い、順行でデファー・逆行で部分利確/BE強化が働く（ログ/metrics確認可）
    - 短時間リプレイ/バックテストでエントリー数・lot・EXIT挙動が想定通りであることを確認し、micro/macro適用前に記録する
  Plan:
    - indicatorsに不足指標を実装し、facに追加（nullセーフ）
    - scalpの信頼度/TP/lotスケールにコンポジットスコアを組み込み、ログで倍率を出す
    - EXITにMACD/DMI/ボラ/パターンのデファー・部分利確ロジックを追加し、micro/macro展開前にリプレイ検証
    - 検証後にmicro/macroへ適用し、回帰チェックを行う
  Notes:
    - 最初はscalpのみ有効化し、影響を限定した状態で観察する

- [ ] ID: T-20251117-002
  Title: TrendMomentumMicro gating fix & range-to-trend handoff
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: (core 廃止済み) workers/micro_core/worker.py, analysis/range_guard.py, docs/TASKS.md
  Context: 2025-11-17 JST 05:55〜06:05 のローソクで MA 乖離≥0.45pips & ADX≥20 でも TrendMomentumMicro が発火せず、BB_RSI がショートを連発している。
  Acceptance:
    - MA 乖離≥0.45pips & ADX≥20 時は range_score が高くても regime_mode が trend_* へ遷移し、同時間帯のログで確認できる
    - 強トレンド中は BB_RSI/MicroRangeBreak が抑止され、代わりに TrendMomentumMicro または MicroPullbackEMA のエントリーが trades.db に残る
    - デプロイ後 06:08Z 以降の `journalctl` / `sqlite3` 監視結果を Notes に反映し、改善状況を記録する
  Plan:
    - VM から trades.db / logs を取得し、regime_mode・range_score と損失クラスター (ID 4278-4282 付近) を把握する
    - `analysis.range_guard` / `_classify_regime` を調整し、強トレンド検出時に trend モードへ切り替わるよう修正してデプロイする
    - デプロイ後に `journalctl` と `sqlite3` で 06:08Z 以降のエントリーを監視し、TrendMomentum 系の稼働を確認する
  Notes:
    - 直近損失: 2025-11-17 06:52 JST 前後で BB_RSI ショートが 3 連敗 (-5.7pips)、06:27/06:37 に bot 再起動ログあり

- [ ] ID: T-20251123-001
  Title: 11/21 急変・レンジ損失抑止とチャンス捕捉の強化
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: main.py, execution/exit_manager.py, execution/stage_tracker.py, analysis/range_guard.py, docs/TASKS.md
  Context: 2025-11-21 の日次 -17pips（-1,716 JPY）。15:03Z の pullback ショート -10.9pips が致命傷、04–05Z に micro 逆行小負けが積み上がり、07:00Z の大幅下落(58pipsレンジ)は未参戦。レンジ追随ショートと急変後の逆行に弱い。
  Acceptance:
    - 30分±20pips超の片道を検知したら同方向エントリーをブロックし、保有同方向ポジは即トレイル/部分利確を実行したログが残る（理由タグ付き）
    - ブレイク/プルバック系で初期SLを5–7pipsに自動設定し、MFE+4〜5pips未達はBE付近へトレイル、MFE+8pipsで部分利確が実行され trades.db に残る
    - range_guard の判定を micro/pullback/mirror に適用し、レンジ中はTrend/Mirror系の発火が抑制され、代わりにタイトSL/TP設定が付くことをログで確認できる
    - 07–08Z のような大きなトレンド開始を time-band優先で許可し、07:00Z帯のバックテスト/リプレイでエントリーが1件以上生じている
  Plan:
    - 11/21 OANDAログ/ローソクをもとに急変閾値・MFE/MAE分布を抽出し、各戦略への適用設計をまとめる
    - `exit_manager` に MFEトレイル＋部分利確、`main.py/stage_tracker` に急変クールダウンと時間帯優先ゲート、`range_guard` に適用範囲拡大を実装
    - 11/21データでリプレイ/回帰を流し、15:03Zの損失縮小と07:00Z帯のエントリー発生を確認して Notes に記録
  Notes:
    - ソースデータ: logs/oanda/transactions_20251121T20251123T090935Z.jsonl, logs/oanda_candles/candles_M1_20251121.json（M1 134pips 日レンジ）
    - 進捗(2025-11-23): main.py に急変検知クールダウンを追加（10分±0.30円 + 30分±0.20円で方向別に約15分ブロック）。range_active 時に TrendMA/MomentumBurst を抑制し、非レンジでも macro/micro のSL下限を6p/5pに統一。exit_manager に MFE 部分利確（macro 8p / micro 5p）と簡易トレイルを追加。range_soft_active でも軽いSL/TPタイト化・信頼度減衰を適用。残: range_guard適用範囲の精緻化/リプレイ検証/ログ記録。
    - 進捗(2025-11-24): micro の最短保有/損失ガードを 180s/-2.5p へ延長して即損切りを抑制。spread_guard デフォルトを緩和（max 1.2p / release 0.9p / min_high 4 / release_samples 6 / cooldown 15s）し、ブロックからの解除を早めた。
    - 進捗(2025-11-24夕): macro snapshot stale 検知時に即リフレッシュするよう main.py を修正し、13:48Z 再起動後は stale ログなし。スプレッドガードのブロックは未発生で、直近は 11:58Z の TrendMomentumMicro ロング (-1.3p) のみ約定。
    - 進捗(2025-11-26): 11/21 M1 ローソクと OANDA 取引 17 件を突合。日次 -19.5p、保有中央値 31s、14:31Z〜16:23Z の ±40〜63p 片道ムーブを未捕捉。MFE10>8p なのに早期決済/逆行損切りが 6 件（例: 01:50Z +12.9p 潜在/2.3p 決済, 04:45Z +7.9p 潜在/-2.6p, 15:03Z -10.9p 即死）。ADX/EMA 乖離ゲートの抑止がチャンス欠落要因と推定。次対応: (1) 急変帯は EMA/ADX スキップしてエントリー優先、(2) micro/scalp は最短保有 120s と BE トレイル強化、(3) 30–60s での逆行ストップ幅を ATR*1.2+5p へ拡大し、(4) MFE>6p で部分利確+トレイルを必須化。
    - 追加検証: `scripts/backtest_exit_manager.py --candles logs/oanda_candles/candles_M1_20251121.json` を実行し、macroシグナルのエグジット挙動を簡易確認（closed 99/ under5min 85 / under2pips 37 / profit_pips -32.1）。tickリプレイ/real trades が無いため、詳細な MFE/MAE 検証は未実施。

- [ ] ID: T-20251117-003
  Title: Micro adaptive tuning & momentum stacking strategy
  Status: in-progress
  Priority: P1
  Owner: codex
  Scope/Paths: (core 廃止済み) workers/micro_core/worker.py, strategies/micro/*, docs/TASKS.md
  Context: micro pocket が固定パラメータのままでボラ変化に追随できず、強トレンド中も BB_RSI ばかりが約定している。小刻みな順張り追加・学習的な可変調整を導入したい。
  Acceptance:
    - ATR/vol/range_score に応じて TrendMomentum/BB_RSI のしきい値が自動で上下し、ログに動的パラメータが残る
    - 新戦略（Momentum Stack など）が追加され、strong trend 中に 2 つ以上の micro 戦略が並走できる
    - 変更を VM へデプロイ後、`trades.db` に新戦略の取引と BB_RSI 抑制の両方が確認できるかモニタリング手順を Notes に残す
  Plan:
    - fac_m1 を基にした動的チューニング関数を実装し、TrendMomentum/BB_RSI/MicroPullback と regime 判定で利用
    - 新しい順張り積み増し戦略 (仮称 MicroMomentumStack) を実装し worker へ登録
    - VM へデプロイして journalctl / sqlite3 で挙動を監視、結果を記録
  Notes:
    - 11/17 06:50 前後は ATR≈1.7、MA 乖離≈2.4pips でも Range 判定で抑止されていたため、ATR連動の gap/ADX 緩和が必要

- [ ] ID: T-20251208-001
  Title: M1Scalper fast_cut緩和に合わせたMFEリトレース部分カット
  Status: todo
  Priority: P1
  Owner: codex
  Scope/Paths: strategies/scalping/m1_scalper.py, execution/exit_manager.py, docs/TASKS.md
  Context: fast_cut緩和で保有が長くなるため、MFEリトレース時に早めの部分カットを入れて逆行ダメージを抑えたい（scalp/microのみ）。
  Acceptance:
    - MFE>しきい値後のリトレースで部分カットが発火し、logs/orders.db / trades.db に部分決済が残る
    - エントリー回数は大きく減らず、fast_cut/hard_cutを残したまま部分カットが先行する
  Plan:
    - exit_manager に MFE リトレース判定を追加し、scalp/microはまず50%近辺を部分クローズ
    - M1Scalper のエントリーにタグ/メタを付与し、exit 側で選択的に適用
    - VM でログ確認し、過剰発火があれば閾値/比率を調整

- [ ] ID: T-20251208-002
  Title: BB_RSI トレンド時のサイズ縮小とTP寄せ
  Status: in-progress
  Priority: P2
  Owner: codex
  Scope/Paths: strategies/mean_reversion/bb_rsi.py, docs/TASKS.md
  Context: fast_cut緩和で損切りが遅れがち。トレンド検知時にフルサイズで逆張りしないよう抑えたい。
  Acceptance:
    - ADX/MA傾きが強いときは BB_RSI の confidence/size を下げ、TPを近めに設定したシグナルがログに残る
    - レンジ時のエントリー数は従来と同程度
  Plan:
    - トレンド判定(ADX+MA傾き)を組み込み、サイズ係数とTP短縮を適用
    - VM でログ/約定を確認し、抑制しすぎないよう閾値を調整
  Notes:
    - 2025-12-08: BB_RSI に trend_score (ADX+MA gap) を追加し、confidence/downsize・TP寄せを適用。`trend_bias`/`size_factor_hint` がログ/シグナルに出る。要VMデプロイと挙動確認。

- [ ] ID: T-20251208-003
  Title: TrendMA 短時間クールダウンと逆行初動の軽い部分クローズ
  Status: todo
  Priority: P3
  Owner: codex
  Scope/Paths: strategies/trend/ma_cross.py, execution/exit_manager.py, docs/TASKS.md
  Context: TrendMA の連続エントリーを少し間引き、逆行初動で小さく逃がすオプションを入れたい。
  Acceptance:
    - 約定直後に2–3分の再エントリークールダウンがかかり、ログに理由付きで出力される
    - モメンタム大回転時に小さな部分クローズが実行され trades.db に記録される
  Plan:
    - ma_cross にクールダウンロジックを追加
    - exit_manager に TrendMA 用の軽い部分クローズ条件を追加（任意で無効化可）
    - VM でログ/約定を確認して閾値を調整

- [ ] ID: T-20251208-004
  Title: MomentumBurst/MTF Breakout 同方向重複のサイズ上限と反転トレイル強化
  Status: todo
  Priority: P2
  Owner: codex
  Scope/Paths: strategies/micro/momentum_burst.py, strategies/micro/mtf_breakout.py, execution/exit_manager.py, docs/TASKS.md
  Context: ImpulseRetraceが順張り化したことで同方向ポジが重なる可能性。重複時のサイズ抑制と反転時のトレイル強化が必要。
  Acceptance:
    - 同方向ポジ保有中は新規サイズが縮小され、ログに理由が残る
    - モメンタム急反転時にトレイル/早期クローズが動作し trades.db に反映される
  Plan:
    - 戦略側で同方向オープン数を参照しサイズ係数を適用
    - exit_manager にモメンタム反転トレイル（micro向け）を追加
    - VM で挙動を確認し、サイズ係数/トレイル閾値を調整

- [ ] ID: T-20251208-005
  Title: Macro_core / Trend_H1 の軽量エクスポージャ制御（core 廃止済みメモ）
  Status: todo
  Priority: P3
  Owner: codex
  Scope/Paths: main.py, execution/risk_guard.py, docs/TASKS.md
  Context: 下位足の逆向きヘッジが長持ちしやすくなったため、同方向総エクスポージャが閾値超なら新規サイズを小さくする軽い制御を入れたい。
  Acceptance:
    - 同方向エクスポージャが設定閾値を超えた場合、新規オーダーのサイズが縮小されるログが残る
    - エクスポージャ閾値は環境変数で調整可能
  Plan:
    - risk_guard か main のロット計算で同方向オープン量を参照し縮小係数を適用
    - VM でログ/約定を確認し、閾値と係数を微調整

## Archive
- [x] ID: T-20251121-004
  Title: ボラ急変時の並列戦略トリガとキャンドル機会の補完
  Status: done
  Priority: P2
  Owner: codex
  Scope/Paths: main.py, strategies/micro/momentum_burst.py, strategies/scalping/impulse_retrace.py
  Context: 11/20〜21 の ±0.30 円級ムーブでエントリーゼロ。spread gate・macro snapshot stale・戦略不足が重なってチャンスを逃していた。
  Completed: 2025-11-21
  Summary: 10 分窓の価格急変 (≥0.30円) を検出する deque バッファを logic_loop に追加し、発火時に MomentumBurst を micro hint へ、ImpulseRetrace を scalp ランキングへ強制追加して M1Scalper 以外の並列エントリーを許可した。spread override フラグと plan snapshot の notes も公開し、ログで surge 検出理由を確認できるようにした。

- [x] ID: T-20251121-003
  Title: ATR 連動 partial exit・SL 圧縮・微益回収の実装
  Status: done
  Priority: P1
  Owner: codex
  Scope/Paths: execution/exit_manager.py, execution/order_manager.py
  Completed: 2025-11-21
  Summary: ExitManager に ATR/vol_5m を用いた loss_guard 圧縮と TrendMA/Vol partial exit 強化を加え、2.5〜3.0p で 2/3 クローズ→ EMA gap 1.0p で残りを解放するフローを実装。注文生成側でも pocket 最小単位を clamp し、TAKE_PROFIT_ON_FILL_LOSS の再発時はログに理由が残るよう修正した。（現行コードでは ExitManager はスタブ化され、専用 exit_worker 側へ移行済み）

- [x] ID: T-20251121-002
  Title: Macro lot 下限 & exposure 推定の是正
  Status: done
  Priority: P1
  Owner: codex
  Scope/Paths: execution/risk_guard.py, execution/order_manager.py, workers/common/core_executor.py
  Completed: 2025-11-21
  Summary: risk_guard.allowed_lot に pocket 別最小ロット (macro=0.1) と手動玉の除外リストを追加、order_manager/StageExecutor が 10k 未満の注文を clamp & ログ出力するようにした。ExposureState は `manual/unknown` を無視するため margin 使用率が実態どおり 0.8 以上まで伸びる。

- [x] ID: T-20251121-001
  Title: main.py 再適用と spread gate / macro snapshot エラー潰し
  Status: done
  Priority: P1
  Owner: codex
  Scope/Paths: main.py, scripts/deploy_to_vm.sh
  Completed: 2025-11-21
  Summary: spread gate オーバーライド引数をすべての PocketPlan/Snapshot 経路に伝播させ、VM へ push & `deploy_to_vm.sh` で再デプロイ。journalctl で NameError/TypeError の連鎖が消え、`policy_bus` notes に relax フラグが出力されることを確認済み。

- [x] ID: T-20251114-002
  Title: Micro order guard hardening & margin scaling
  Status: done
  Priority: P1
  Owner: codex
  Scope/Paths: docs/TASKS.md, execution/order_manager.py, workers/micro_core/worker.py, main.py
  Context: 2025-11-13〜14 の VM ログで micro 戦略の STOP_LOSS_ON_FILL_LOSS / POSITION_TO_REDUCE_TOO_SMALL が多発し margin health も 3% 前後まで低下。
  Acceptance:
    - STOP_LOSS_ON_FILL_LOSS を自動補正/再試行する仕組みが導入され、再発時は安全側の SL/TP でリトライされる
    - 部分クローズ要求が trade size を超えないよう検証し、必要に応じて実サイズで再送 or ALL クローズする
    - micro pocket の lot 配分が spread>0.9p または margin buffer<4% で段階的に縮小し、ログでスケール理由を確認できる
  Completed: 2025-11-17
  PR: <pending>
  Summary: workers/micro_core/worker.py now scales lots using spread/margin telemetry, enforces per-direction/strategy limits, and feeds entry_price metadata back into execution/order_manager.py’s SL/TP normalization so STOP_LOSS_ON_FILL issues retry safely; main.py already handles fallback retries and partial-close resubmits through order_manager.

- [x] ID: T-20251117-001
  Title: Micro strategy expansion & multi-entry support
  Status: done
  Priority: P1
  Owner: codex
  Scope/Paths: strategies/micro/trend_momentum.py, strategies/mean_reversion/bb_rsi.py, workers/micro_core/worker.py, docs/TASKS.md
  Context: 11/17 JST 6:00〜 のローソクで BB_RSI が逆張り連敗し TrendMomentum が発火しておらず、1 pocket 1 trade 制限でエントリー数も不足。
  Acceptance:
    - TrendMomentumMicro がレンジ→トレンド切替後に実際にエントリーし、当日のローソク（05:55〜06:05 JST）で順張り約定がログに残る
    - BB_RSI が強トレンド条件（|MA10−MA20|≥0.5pips & ADX≥21）では自動停止し、逆方向シグナルを出さない
    - micro pocket で同ループ内に複数戦略のエントリーが許可され、1時間当たりトレード数が現在の2倍以上となる
  Completed: 2025-11-17
  PR: <pending>
  Summary: workers/micro_core/worker.py removes the one-trade guard, tracks per-strategy concurrency, and logs range→trend transitions so TrendMomentum can follow through, while strategies/mean_reversion/bb_rsi.py now aborts when |MA10−MA20|≥0.5pips & ADX≥21 to avoid counter-trend entries.

- [x] ID: T-20251110-001
  Title: Restore macro snapshot freshness & widen entry coverage
  Status: done
  Priority: P1
  Owner: maint
  Scope/Paths: analysis/macro_snapshot_builder.py, main.py, scripts/run_sync_pipeline.py, docs/TASKS.md
  Context: Macro snapshot has been stale for >24h causing macro pocket shutdown and reducing trade opportunities; data sync also stops at 00:22 so battle report is incomplete.
  Acceptance:
    - Macro snapshot refresh automatically updates when file asof is stale, and VM loads the latest snapshot (verified via logs without `[MACRO] Snapshot stale` spam)
    - `run_sync_pipeline.py` can sync trades/candles to remote_logs even without BigQuery credentials (lot insights optional)
    - Manual evaluation identifies missed trade windows and documents next strategy additions (momentum/London session)
  Completed: 2025-11-17
  PR: <pending>
  Summary: main.py now refreshes macro snapshots on stale detection and scripts/run_sync_pipeline.py gained `--remote-dir` mirroring trades/candles into remote_logs when BigQuery is unavailable; docs/TRADE_FINDINGS.md captures the manual evaluation and follow-up windows.

- [x] ID: T-20251103-001
  Title: H1トレンドワーカーの追加実装
  Status: done
  Priority: P1
  Owner: tossaki
  Scope/Paths: market_data/candle_fetcher.py, indicators/factor_cache.py, workers/trend_h1/, main.py, docs/TASKS.md
  Context: 中期トレンド用の常時稼働ワーカー不足（AGENT.me 仕様 §3.5, 提案メモ 2025-11-03）
  Acceptance:
    - H1足の因子が取得・キャッシュされ、ワーカーが利用できる
    - TrendH1ワーカーがMovingAverageCrossを用いて売買指示を生成し、Risk/Exit連携済み
    - main.py からワーカーの起動制御が可能で、環境変数でON/OFFできる
    - リプレイ/紙上テスト手順を README or コメントで案内
  Completed: 2025-11-17
  PR: <pending>
  Summary: factors now include H1 caches and workers/trend_h1/worker.py consumes them, with main.py exposing enable flags and scripts/replay_trendma.py documenting replay/testing, fulfilling the Trend H1 worker rollout.

- [x] ID: T-20251104-001
  Title: Impulse Momentum S5 派生ワーカーの実装
  Status: done
  Priority: P1
  Owner: tossaki
  Scope/Paths: workers/impulse_momentum_s5/, main.py, scripts/replay_workers.py, docs/TASKS.md
  Context: impulse_break_s5 の好成績部分（強トレンド＋瞬間変動）を抽出した派生ワーカーで期待値を引き上げる
  Acceptance:
    - impulse_momentum_s5 の config/worker が追加され、環境変数で ON/OFF 可能
    - main.py から起動され、PositionManager/Risk/Exit 連携が整合
    - 瞬間変動・スプレッド・方向整合ガードが実装され、RR>1 の利確/損切りが設定されている
    - scripts/replay_workers.py で実ティックリプレイが可能になり、10月データで検証ログを取得
  Completed: 2025-11-17
  PR: <pending>
  Summary: workers/impulse_momentum_s5/ (config+worker) now mirrors the impulse_break core with stricter trend/volatility gates, is wired into main.py’s worker launcher, and scripts/replay_workers.py can replay it, achieving the targeted S5 derivative.

- [x] ID: T-20251104-002
  Title: Mirror Spike Tight 派生ワーカーの実装
  Status: done
  Priority: P2
  Owner: tossaki
  Scope/Paths: workers/mirror_spike_tight/, main.py, scripts/replay_workers.py, docs/TASKS.md
  Context: mirror_spike_s5 の薄利傾向を、トレンド整合＋高ボラ条件のみで狙う派生を作り平均損益を改善
  Acceptance:
    - mirror_spike_tight の config/worker が追加され、stage=1 固定で RR>1 設定
    - H1/M1 トレンド整合と瞬間変動・スプレッド閾値ガードが組み込まれている
    - main.py から起動制御でき、env で切替可能
    - リプレイスクリプトで再生でき、10月実ティックで検証結果を得る
  Completed: 2025-11-17
  PR: <pending>
  Summary: workers/mirror_spike_tight/ brings the tighter config/worker pair online with trend/spread guards, is registered in main.py’s worker list, and scripts/replay_workers.py includes it for October tick replays.

- [x] ID: T-20251105-003
  Title: Macro/Scalp コア実行の切り出し
  Status: done
  Priority: P1
  Owner: tossaki
  Scope/Paths: main.py, execution/, workers/common/, analysis/plan_bus.py, docs/TASKS.md
  Context: AGENT.me §3.5 / 3.5.1 に沿って macro/scalp の発注・Exit をワーカーへ委譲する前段として、main.py の依存ヘルパーを共通化してプラン参照を標準化する
  Acceptance:
    - client_order_id やステージ計算など macro/scalp 固有ヘルパーが共通モジュール化され、main とワーカーの両方から利用できる
    - PocketPlan/plan_bus がワーカー向けに最新プランの取得・鮮度判定を提供し、Plan 生成側から必要メタを埋め込む
    - docs/TASKS.md で移行計画が追跡され、影響パスが明示されている
  Completed: 2025-11-17
  PR: <pending>
  Summary: main.py now publishes PocketPlans (plan_bus) for macro/scalp with spread-aware factors, while workers/common/core_executor.py consumes them using shared StageTracker/ExitManager helpers, completing the extraction prerequisites。（現在は core_executor/ExitManager は廃止・スタブ化済みで、本項は履歴のみ）

- [x] ID: T-20251108-001
  Title: Macro 戦略のスプレッド考慮 SL/TP 反映
  Status: done
  Priority: P1
  Owner: maint
  Scope/Paths: main.py, strategies/trend/ma_cross.py, strategies/breakout/donchian55.py, workers/common/
  Context: micro/scalp は spread-aware SL/TP を導入済みだが、macro pocket (TrendMA, Donchian55) は未対応のため矛盾している。PocketPlan 経由で最新スプレッド値を全 executor に渡し、macro 戦略でもコスト込みの最低 SL/TP を保証する。
  Acceptance:
    - main から生成される PocketPlan/factors に `spread_pips` が含まれ、macro/scalp executor から参照できる
    - TrendMA / Donchian55 の SL/TP 算出がスプレッドを加味し、RR >= 1.2 + spread buffer を満たす
    - 既存の spread gate ログと矛盾しない
  Completed: 2025-11-17
  PR: <pending>
  Summary: main.py now injects live `spread_pips` into both M1/H4 factors that feed PocketPlans so strategies/trend/ma_cross.py and strategies/breakout/donchian55.py keep their spread-aware target math aligned with executor expectations.

- [x] ID: T-20251105-004
  Title: Macro/Scalp コアワーカー実装
  Status: done
  Priority: P1
  Owner: tossaki
  Scope/Paths: workers/macro_core/, workers/scalp_core/, execution/, main.py, docs/TASKS.md
  Context: macro/scalp Plan を消費する専用ワーカーを追加し、ステージ制御→発注→Exit までを main から切り出す（AGENT.me §3.5, range 強化仕様）
  Acceptance:
    - workers/macro_core/worker.py および workers/scalp_core/worker.py にプランポーリング〜Stage/Exit/Order の実処理ループが実装されている（ログ・メトリクス含む）（※ core 系は現在廃止済み）
    - StageTracker / ExitManager / PositionManager をワーカー内で保有し、main 側との競合なく PocketPlan を消費できる（※ ExitManager はスタブ化済み）
    - main.py 側から新ワーカーを起動できる設定項目が用意され（既定は安全側で無効）、実装メモが docs/TASKS.md に反映されている
  Completed: 2025-11-17
  PR: <pending>
  Summary: workers/macro_core/worker.py and workers/scalp_core/worker.py already host the StageTracker/Exit/Order loops and now receive live PocketPlans from main.py, so macro/scalp execution runs fully inside the dedicated workers（現在は core 系・共通 EXIT を廃止しており履歴のみ）。

- [x] ID: T-20251104-003
  Title: Maker型1pipスカルパーワーカーの実装
  Status: done
  Priority: P1
  Owner: tossaki
  Scope/Paths: market_data/orderbook_state.py, analytics/cost_guard.py, workers/onepip_maker_s1/, main.py, docs/TASKS.md
  Context: 1pip利確多数回の実装方針（板主導メイカー戦略）を別ワーカーとして実装する（AGENT.me 仕様 §3.5.1、提案メモ 2025-11-04）
  Acceptance:
    - L2板情報キャッシュとレイテンシ推定を扱う orderbook_state モジュールが追加され、最新スナップショットを参照できる
    - 取引コスト c を推定するコストガードが実装され、latest fill から c を更新・参照できる
    - workers/onepip_maker_s1 worker が shadow モードで稼働し、post-only・TTLロジックの骨組みとメトリクス出力が揃う
    - main.py にワーカー起動導線が追加され、環境変数で有効化/シャドウ切替が可能
  Completed: 2025-11-17
  PR: <pending>
  Summary: market_data/orderbook_state.py maintains the L2 snapshot, analytics/cost_guard.py tracks realised costs, and workers/onepip_maker_s1/worker.py (wired via main.py flags) runs the one-pip maker logic in shadow/post-only mode as specified.


完了タスクを以下形式で移動・記録:

```md
- [x] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: done
  Priority: P2
  Owner: <担当>
  Scope/Paths: <対象>
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <満たした条件1>
  Completed: YYYY-MM-DD
  PR: <PRリンク or コミットSHA>
  Summary: <1〜3 行で要約>
```

---

- [x] ID: T-20251102-001
  Title: gcloud doctor/installer とデプロイ前診断導入
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/install_gcloud.sh, scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, README.md
  Context: デプロイ不能・gcloud 未導入の再発防止（IAP/OS Login を含む）
  Acceptance:
    - gcloud 未導入時に install スクリプトで導入可能
    - doctor で project/API/OS Login/インスタンス/SSH を検査できる
    - deploy スクリプトが前提不備で早期失敗・誘導
    - セットアップ手順ドキュメントがある
  Completed: 2025-11-02
  PR: <pending>
  Summary: gcloud インストーラ・プリフライト（doctor）とドキュメントを追加し、deploy スクリプトに前提チェックを組み込みました。

- [x] ID: T-20251102-002
  Title: ヘッドレス運用（サービスアカウント）対応
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, AGENTS.md
  Context: アクティブアカウント不在でも GCP/VM を操作できるようにする
  Acceptance:
    - SA キー指定で doctor/deploy が動作（-K）
    - 必要時 SA インパーソネートにも対応（-A）
    - ドキュメントに手順とロール要件を明記
  Completed: 2025-11-02
  PR: <pending>
  Summary: doctor/deploy に SA キー自動有効化とインパーソネートを追加し、ヘッドレス運用手順を docs に追記しました。

- [x] ID: T-20251102-003
  Title: AGENTS.md に quantrabbit クイックコマンドを追加
  Status: done
  Priority: P3
  Owner: maint
  Scope/Paths: AGENTS.md
  Context: 実運用の即時参照用に、Doctor/Deploy/IAP/SA の実値例を集約
  Acceptance:
    - AGENTS.md 10.3 に quantrabbit 固定のコマンド例がある
    - 旧来の冗長/重複手順は削除済み
  Completed: 2025-11-02
  PR: <pending>
  Summary: プロジェクト/ゾーン/インスタンス実値のクイックコマンドを AGENTS.md に追記。

- [x] ID: T-20251111-001
  Title: 戦略ごとの最低ホールド時間とATR連動Exitを実装して超短期クローズを是正
  Status: done
  Priority: P0
  Owner: tossaki
  Scope/Paths: strategies/*, main.py, execution/exit_manager.py, execution/order_manager.py, execution/stage_tracker.py, docs/TASKS.md
  Context: `<60s` クローズ偏重を是正して利確レンジを確保する。
  Acceptance:
    - 戦略シグナルに `profile/min_hold_sec/target_tp_pips/loss_guard` を追加し、exit/partial ロジックがメタを参照する
    - exit_manager が保持期間ガードを適用し、StageTracker が違反を記録してクールダウンと `exit_hold_violation` メトリクスを出す
    - range_mode 時の分割利確が min_hold 経過までは抑制され、`partial_hold_guard` をログへ記録
  Completed: 2025-11-13
  Summary: TrendMA/Donchian/BB_RSI 各ストラテジーに profile/hold メタを付与し、main→order_manager→exit_manager がエントリー thesis に埋め込んだ値で判断するよう更新。StageTracker・hold monitor で違反を記録し、range partial も `partial_hold_guard` により早期利確をブロックする。ユニットテスト（tests/execution/test_exit_manager.py, test_risk_guard.py）を追加し、pytest 実行はローカルに pytest が無いため未実施。

- [x] ID: T-20251112-001
  Title: Micro/Scalpのエントリー・SLTP・EXIT精度を改善し manual sentinel を実装
  Status: done
  Priority: P0
  Owner: tossaki
  Scope/Paths: main.py, workers/micro_core/worker.py, strategies/mean_reversion/bb_rsi.py, execution/exit_manager.py, docs/TASKS.md
  Context: manual 玉の巻き添えと BB_RSI の低品質エントリーを抑制。
  Acceptance:
    - manual/unknown エクスポージャを検知して micro/scalp を自動停止し、解除は 2 サイクル遅延で行う
    - BB_RSI が環境フィルタ＋ATR ベース SL/TP/target を提供し、新メタを exit/partial/metrics へ連携
    - ExitManager/微コアが hold ratio ガード・manual sentinel・ヒステリシスをメトリクス付きで実装（※ ExitManager/微コアは現在廃止・スタブ化）
  Completed: 2025-11-13
  Summary: main/micro_core と HoldMonitor で manual sentinel＋hold ratio guard を導入し、2 サイクル待機後に自動解除する仕組みを追加。BB_RSI に profile/target/min_hold を付与して環境依存の SLTP を生成、micro worker の policy 出力と metrics (`manual_halt_active`, `hold_ratio_guard`) を更新した（現在の運用は専用 micro_* ワーカー＋スタブ ExitManager）。

- [x] ID: T-20251112-002
  Title: 戦略プロファイル化（ATR/ADX/RSIコンボ）でエントリー〜EXITを一貫化
  Status: done
  Priority: P0
  Owner: tossaki
  Scope/Paths: strategies/mean_reversion/bb_rsi.py, strategies/breakout/donchian55.py, strategies/trend/ma_cross.py, main.py, execution/exit_manager.py
  Context: 戦略メタを entry→exit まで引き回し、レンジ/トレンド応じたガードを実装。
  Acceptance:
    - `strategy_profile` と target/hold/loss メタが entry thesis に保存され、range ガードや exit_manager が参照
    - profile 情報を基に main が SL/TP を再調整し、exit_manager が逆シグナル無視のヒステリシスを適用
  Completed: 2025-11-13
  Summary: 各コア戦略に profile 名と target/loss/min_hold の算出ロジックを追加し、main が thesis へ書き込んで exit_manager・order_manager がそれらを尊重するよう接続。range 圧縮時の SL/TP 補正と partial/exit の profile 分岐が揃い、worker_breakdown/metrics に profile ベースのログが出るようになった。

- [x] ID: T-20251112-003
  Title: 手動ポジションと bot の合計使用率を 92% 以下に保つリスク配分
  Status: done
  Priority: P0
  Owner: tossaki
  Scope/Paths: main.py, workers/micro_core/worker.py, execution/risk_guard.py, execution/position_manager.py, tests/execution/test_risk_guard.py
  Context: manual 玉と bot 玉を合わせた総エクスポージャの上限を 92% で統制する。
  Acceptance:
    - PositionManager が manual exposure スナップショットを提供し、risk_guard が cap 比率とメトリクスを計算
    - main/micro_core が exposure state を参照し、lot/units を動的に縮退させる（cap 超過時は新規停止）
    - `exposure_ratio` メトリクスと単体テストで 92% 超過がブロックされることを確認
  Completed: 2025-11-13
  Summary: PositionManager に manual exposure API を追加し、risk_guard が `ExposureState` を算出して `exposure_ratio` を記録するように変更。main/micro_core の lot 配分と発注前チェックで cap を超える場合はスキップし、成功時は state を consume。tests/execution/test_risk_guard.py で割当ロジックを検証（pytest は環境に未導入のため実行不可）。

- [ ] ID: T-20251209-001
  Title: fast_cut タグレス化＋テクニカル必須化の仕上げとトレイル強化
  Status: in-progress
  Priority: P0
  Owner: tossaki
  Scope/Paths: main.py, execution/exit_manager.py, execution/order_manager.py, strategies/*
  Context: fast_cut/kill がタグ漏れで外れる問題を解消し、エントリー時に ATR/RSI/ADX/レジームを必須メタとして付与。レンジ/トレンド別の部分利確・トレイルとレジーム連動ロット縮退も合わせて仕上げる。
  Acceptance:
    - 全エントリーが ATR/RSI/ADX/fast_cut_pips/time/hard_mult を thesis に保持し、欠損時は発注しない
    - exit_manager がタグレスでもテクニカル揃いのポジに fast_cut/hard_cut を適用し、manual 口座は除外
    - レンジ判定時は部分利確/トレイル/fast_cut がタイト化、トレンド強時は緩和される
    - レジーム連動のロット縮退（低ADX・BBW収縮時に初手/追撃を抑制）を main/risk_guard で反映
    - 手動ポジは常に除外され、メトリクスに tech-on/skip が記録される
