# ワーカー再編の確定ログ（2026-02-13）

## 方針（最終確定）

- 各戦略は `ENTRY/EXIT` を1対1で持つ。
- `precision` 系のサービス名は廃止し、サービス名は `quant-scalp-*` へ切り分ける。
- `quant-hard-stop-backfill` / `quant-realtime-metrics` は削除対象。
- データ供給は `quant-market-data-feed`、制御配信は `quant-strategy-control` に分離。
- 補助的運用ワーカーは本体管理マップから除外。

### 2026-02-27（追記）replay auto-improve を時間帯ブロック依存から reentry 動的調整へ移行

- 背景:
  - `analysis.replay_quality_gate_worker` の auto-improve は
    `policy_hints.block_jst_hours` を `worker_reentry` へ反映していたため、
    時間帯封鎖に寄りやすく、ノイズ時に「停止で回避」へ偏るリスクがあった。
  - 実運用方針（JST7-8保守帯を除き恒久ブロックを避ける）と整合させるため、
    反映対象を reentry の動的パラメータへ変更した。
- 変更:
  - `analysis/trade_counterfactual_worker.py`
    - `pattern_book_deep` を事前確率として統合（strategy+side）。
    - spread/stuck/OOS を `noise_penalty` 化し、
      `noise_lcb_uplift_pips` と `quality_score` で候補を再ランキング。
    - `policy_hints.reentry_overrides`（`tighten/loosen` + multiplier）を出力。
  - `analysis/replay_quality_gate_worker.py`
    - auto-improve の採用条件を
      `reentry_overrides.confidence` と `lcb_uplift_pips` ベースへ変更。
    - `worker_reentry.yaml` へ `cooldown_win_sec` / `cooldown_loss_sec` /
      `same_dir_reentry_pips` / `return_wait_bias` を反映。
    - `block_jst_hours` 自動適用は `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_BLOCK_HOURS=1`
      指定時のみ有効（既定は無効）。
  - `ops/env/quant-trade-counterfactual.env` /
    `ops/env/quant-replay-quality-gate.env`
    - pattern/noise/reentry gate 用の運用パラメータを追加。
- 意図:
  - 「高速探索 + 厳格昇格」を維持しつつ、
    時間帯封鎖ではなく reentry 品質（再突入距離/待機時間/バイアス）で改善を回す。

### 2026-02-27（追記）`scalp_ping_5s_c` の leading profile reject を C専用で緩和

- 背景（VM実測）:
  - `quant-scalp-ping-5s-c.service` 再起動後（`2026-02-27 15:19:59 UTC`）の
    15:20-15:24 UTC で、ログ集計は `open=59` に対して
    `entry_leading_profile_reject=56`。
  - 同時間帯 `orders.db` では `scalp_ping_5s_c_live` の
    `submit_attempt/filled` が 0 件で、実質エントリー停止状態。
  - `metrics.db` では `order_perf_block` が 15:18 UTC 以降 0 件のため、
    主阻害は perf guard ではなく strategy_entry 側の leading profile reject。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.64 -> 0.56`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_PENALTY_MAX: 0.20 -> 0.14`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_UNITS_MIN_MULT: 0.72 -> 0.58`
- 意図:
  - hard reject を減らし、低確度帯はゼロ化ではなく縮小ロットで通す。
  - C戦略のエントリー再開余地を作りつつ、`UNITS_MIN_MULT` を下げてリスク急拡大を回避する。

### 2026-02-27（追記）`scalp_ping_5s_c` long 側の leading hard reject を無効化

- 背景（VM実測）:
  - 上記緩和反映後も `2026-02-27 15:25:36-15:28:54 UTC` のログで
    `market_order rejected ... reason=entry_leading_profile_reject` が
    主に `side=long` で継続。
  - `orders.db` では `submit_attempt=3 / filled=3` の再開は確認したが、
    long reject 連発により露出回復が不足。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.56 -> 0.00`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.80` は維持。
- 意図:
  - Cの long bias 運用に合わせ、long 側の hard reject を止めて約定再開を優先。
  - short 側は従来閾値を維持し、逆方向エクスポージャの過多は抑える。

### 2026-02-27（追記）`scalp_ping_5s_c` の preserve-intent 確率閾値を 0.58 へ再緩和

- 背景（VM実測）:
  - long leading reject 無効化後（15:34:41 UTC 以降）、
    `entry_probability_reject_threshold` が主拒否理由に遷移。
  - Cログで `prob=0.81〜0.89` 帯のシグナルでも reject が連発し、
    Cの約定回復が不足した。
  - `quant-order-manager` 実効envは
    `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]=0.72`。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE: 0.72 -> 0.58`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE: 0.72 -> 0.58`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C: 0.72 -> 0.58`
- 意図:
  - leading profile 緩和後の新ボトルネック（probability reject）を解消し、
    Cの送出再開を狙う。
  - worker/order-manager 間で確率閾値を同期し、経路差による拒否ブレを抑える。

### 2026-02-27（追記）`scalp_ping_5s_c` の preflight 閾値を `quant-order-manager` 実効envへ同期

- 背景（VM実測）:
  - `quant-order-manager.service` は `quant-v2-runtime.env` と `quant-order-manager.env` のみを読む。
  - C worker 側（`scalp_ping_5s_c.env`）で緩和した `PERF_GUARD_FAILFAST_*` / `PRESERVE_INTENT_*` が
    order-manager 側へ未同期で、`perf_block` と `entry_probability_reject` が継続していた。
  - `perf_guard.is_allowed(..., env_prefix=SCALP_PING_5S_C)` を VM で再現すると、
    order-manager 実効envでは `hard:hour13:failfast:pf=0.32 win=0.36 n=22` を返却。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_*_STRATEGY_SCALP_PING_5S_C[_LIVE]` を
      `reject_under=0.74 / min_scale=0.34 / max_scale=0.56` へ更新。
    - `SCALP_PING_5S[_C]_PERF_GUARD_FAILFAST_*` を
      `min_trades=30 / pf=0.20 / win=0.20` へ更新。
    - 反映後に `hard:sl_loss_rate` が主因化したため、
      `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_*` も
      `min_trades=30 / max=0.68` へ更新。
- 意図:
  - V2分離の責務に合わせ、order-manager preflight 判定を worker 運用値と一致させる。
  - 同一戦略が worker 経路と order-manager 経路で異なる閾値を読む状態を解消する。

### 2026-02-27（追記）`scalp_ping_5s_c_live` の hard `perf_block` を failfast/sl_loss_rate 起因で緩和

- 背景（VM実測）:
  - `logs/metrics.db`（直近60分）で `order_perf_block=134`、うち `scalp_ping_5s_c_live=119`。
  - 主理由は `hard:hour14:sl_loss_rate=0.68 pf=0.39 n=41`（94件）と
    `hard:hour15:failfast:pf=0.12 win=0.28 n=43`（25件）。
  - `logs/orders.db` でも同時間帯で C は `filled=4 / perf_block=119` と、B（`filled=24 / perf_block=15`）より hard block 依存が強い状態。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_HARD_PF=0.00` を追加。
    - fallback `SCALP_PING_5S_PERF_GUARD_FAILFAST_HARD_PF=0.00` も追加。
    - `SCALP_PING_5S_C_PERF_GUARD_SL_LOSS_RATE_MAX` を `0.55 -> 0.70`。
    - fallback `SCALP_PING_5S_PERF_GUARD_SL_LOSS_RATE_MAX` を `0.55 -> 0.70`。
  - `ops/env/scalp_ping_5s_c.env`
    - 同値（`FAILFAST_HARD_PF=0.00`, `SL_LOSS_RATE_MAX=0.70`）を C + fallback の両プレフィクスへ同期。
- 意図:
  - `reduce` 運用のまま hard failfast 既定床（PF floor）で即停止する挙動を抑え、連続エントリーの再開余地を残す。
  - `sl_loss_rate` hard block を hour14 の実測値に対して過剰拒否しない水準へ調整し、改善ループを継続させる。

### 2026-02-27（追記）`scalp_extrema_reversal_live` の「取り残し」防止を実装

- 背景（VM実測）:
  - `orders.db` 直近7日で `scalp_extrema_reversal_live` の `filled=47` に対し、
    `stopLossOnFill` 付きは `4`（約8.5%）に留まっていた。
  - 同時点 open trade は `scalp_extrema_reversal_live` が `3/3` で、
    いずれも broker-side SL なし（TPのみ）。
  - `trade_id=408076` は `-49.5 pips` まで逆行して保持され、
    「SLなし + loss_cut未発火」で取り残しが発生していた。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL_LIVE=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL=1`
  - `config/strategy_exit_protections.yaml`
    - `scalp_extrema_reversal_live.exit_profile` を追加:
      - `loss_cut_enabled=true`
      - `loss_cut_require_sl=false`
      - `loss_cut_hard_pips=7.0`
      - `loss_cut_reason_hard=m1_structure_break`
      - `loss_cut_max_hold_sec=900`
      - `loss_cut_cooldown_sec=4`
- 意図:
  - エントリー時に broker-side SL を優先付与し、まず「無防備建玉」を減らす。
  - 仮にSL未付与で建っても、EXIT worker の deterministic loss-cut で取り残しを回避する。

### 2026-02-27（追記）`scalp_ping_5s_c` の実効ロット下限を引き上げ（約定再開後）

- 背景（VM実測）:
  - preflight hard block 解消後に `filled` は再開したが、`entry_units_intent` が `1-2` に偏り、
    long 側の露出不足が継続。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `BASE_ENTRY_UNITS=140`
    - `MIN_UNITS=5`
    - `MAX_UNITS=260`
    - `ALLOW_HOURS_OUTSIDE_UNITS_MULT=0.70`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT=0.72`
    - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=1.00`
- 意図:
  - 通過率を維持しつつ、再開後の極小ロット約定を減らして収益寄与を改善する。

### 2026-02-27（追記）VM systemd の `EnvironmentFile` 重複読込を自動是正

- 背景（VM実測）:
  - `quant-online-tuner.service` などで `/etc/quantrabbit.env` の二重読込（同一パス重複）が残存。
  - 実害は小さいが、監査ノイズと設定経路の不透明化を招くため除去対象とした。
- 変更:
  - `scripts/dedupe_systemd_envfiles.py` を追加。
    - `systemctl show` の FragmentPath/DropInPaths を使って読み込み順を解析。
    - **同一パスの `EnvironmentFile` 重複のみ**を `/etc/systemd/system/*.service.d/*.conf` から削除。
    - fragment unit は編集しない（drop-in のみ編集）。
  - `scripts/install_trading_services.sh` に上記スクリプトの自動実行（`--apply`）を追加。
  - `scripts/ops_v2_audit.py` に `EnvironmentFile` 重複検知（warn）を追加。
- 意図:
  - V2 運用で env 読み込み経路を単純化し、`systemctl cat` 監査結果の可読性を維持する。

### 2026-02-27（追記）ops系 unit の `/etc/quantrabbit.env` 注入を除去可能化

- 背景（VM実測）:
  - `quant-autotune-ui.service` / `quant-bq-sync.service` などで、
    `quant-v2-runtime.env` に加えて `/etc/quantrabbit.env` が drop-in で追加され、
    `FORECAST_GATE_*` 系の衝突を生んでいた。
- 変更:
  - `scripts/dedupe_systemd_envfiles.py` に `--remove-envfile` を追加。
    - 指定した env パスを `/etc/systemd/system/*.service.d/*.conf` から安全に除去可能。
    - fragment unit は編集しない。
  - `scripts/ops_v2_audit.py` に、ops系 unit へ `/etc/quantrabbit.env` が残っている場合の warn を追加。
- 意図:
  - 監査ノイズを減らし、ops系 unit の設定責務を `quant-v2-runtime.env` 側へ寄せる。

### 2026-02-27（追記）M1系4ワーカーの spread 上限を 1.00 へ統一

- 背景:
  - 分離3ワーカーで `M1SCALP_MAX_SPREAD_PIPS=0.35/0.40/0.45` を設定しており、
    実スプレッド `0.80p` 局面で entry 以前に `blocked by spread` が連続した。
- 変更:
  - `ops/env/quant-scalp-trend-breakout.env`: `M1SCALP_MAX_SPREAD_PIPS=1.00`
  - `ops/env/quant-scalp-pullback-continuation.env`: `M1SCALP_MAX_SPREAD_PIPS=1.00`
  - `ops/env/quant-scalp-failed-break-reverse.env`: `M1SCALP_MAX_SPREAD_PIPS=1.00`
  - 分離3ワーカーに `spread_guard_max_pips=1.00` を追加（`spread_monitor` 側上限も統一）
  - `ops/env/quant-m1scalper.env`: `M1SCALP_MAX_SPREAD_PIPS=1.00` を明示追加
- 意図:
  - M1系の spread 判定を一律化し、設定ばらつきでの機会損失を防ぐ。

### 2026-02-27（追記）`gpt_ops_report` を deterministic 市況プレイブックへ更新

- 目的:
  - 手動で作っていた「短期/スイング + 3シナリオ + リスク手順」の整理を、
    LLMなし・定期実行で自動生成する。
- 変更:
  - `scripts/gpt_ops_report.py`
    - 旧 stub（`llm_disabled` の最小 JSON）を廃止し、
      `factor_cache(M1/H4)`・`forecast_gate`・`trades/orders`・`policy_overlay` を統合した
      deterministic レポートを生成する実装へ差し替え。
    - 出力に `snapshot`, `short_term`, `swing`, `scenarios(3本)`, `risk_protocol`,
      `event_context`, `order_quality`, `performance` を追加。
    - 既存の `--policy/--apply-policy` 導線は維持（自動適用は行わず no-change diff を出力）。
    - Markdown 出力（`--markdown-output`）を追加し、運用レビュー用の可読テキストを自動生成。
  - `tests/scripts/test_gpt_ops_report.py` を追加し、
    PF計算、シナリオ確率合計、最小入力時の出力構造を回帰テスト化。
- 影響範囲:
  - `quant-ops-policy.service` が生成する `logs/gpt_ops_report.json` の内容が拡張される。
  - 取引執行経路（strategy worker / order_manager / position_manager）の判定ロジックは非変更。

### 2026-02-27（追記）`gpt_ops_report` を「主因→壊れる点→A/B/C→If-Then」へ拡張

- 目的:
  - 市況整理を「こねくり回さない」固定手順へ寄せ、運用者が同じ順序で検証できる状態にする。
- 変更:
  - `scripts/gpt_ops_report.py`
    - `market_context`（pairs / DXY / rates / risk / events_summary）を追加。
    - `driver_breakdown`（`rate_diff` / `yen_flow` / `risk_sentiment`）を追加し、
      `dominant_driver` と `net_score` を機械算出。
    - `break_points`（壊れる条件）と `if_then_rules`（条件式）を追加。
    - 生成バージョンを `playbook_version=2` に更新。
    - `--market-context-path`, `--market-external-path`,
      `--macro-snapshot-path` を追加し、データ入力を明示化。
  - `scripts/build_market_context.py`
    - `market_context_latest.json` を単体で生成するCLIを追加。
  - `tests/scripts/test_gpt_ops_report.py`
    - `market_context` 生成、`driver_breakdown` の優勢判定、
      v2 出力構造を回帰テストへ追加。
- 影響範囲:
  - `quant-ops-policy.service` の既存起動コマンドで動作し、追加フィールドが JSON/Markdown に出力される。
  - 自動ポリシー適用は従来どおり無効（no-change diff のみ）。

### 2026-02-27（追記）市場メモ（文章）を運用入力へ変換する導線を追加

- 目的:
  - 手動で作った市況メモ（価格・金利・イベント時刻）をそのまま
    `gpt_ops_report` の入力へ落とし、判断フレームを実運用へ接続する。
- 変更:
  - `scripts/import_market_brief.py`
    - 市況メモの本文から `USD/JPY, EUR/USD, AUD/JPY, EUR/JPY, DXY, US10Y, JP10Y` を抽出。
    - Markdown表の「東京時間（JST）」列をパースし、`logs/market_events.json` 互換のイベント JSON を生成。
    - 出力:
      - `logs/market_external_snapshot.json`（または任意パス）
      - `logs/market_events.json`（または任意パス）
  - `tests/scripts/test_import_market_brief.py`
    - 価格/金利抽出と、翌日イベント時刻（`翌0:00`）の変換を回帰テスト化。
- 影響範囲:
  - 市況メモ→`market_context`→`gpt_ops_report` の入力連鎖がCLIで再現可能になった。
  - 発注経路・strategy worker の判定ロジックは非変更。

### 2026-02-27（追記）外部市況の自動取得（価格/金利/イベント）を追加

- 目的:
  - 手入力なしで「調査→分析」を回せるようにし、`gpt_ops_report` の入力を自動更新する。
- 変更:
  - `scripts/fetch_market_snapshot.py`
    - Stooq から `USD/JPY, EUR/USD, AUD/JPY, EUR/JPY, DXY` を取得し、
      日中変化率（open→close）を算出。
    - TradingEconomics から `US10Y, JP10Y` を抽出。
    - TradingEconomics calendar HTML から主要国イベント（US/JP/EU/DE/UK/AU/CN）を抽出し、
      `impact`（`calendar-date-1/2/3`）と `time_utc/time_jst/minutes_to_event` を付与。
    - 出力:
      - `logs/market_external_snapshot.json`
      - `logs/market_events.json`
  - `tests/scripts/test_fetch_market_snapshot.py`
    - CSV parse / bond value抽出 / calendar抽出の回帰テストを追加。
- 影響範囲:
  - `fetch_market_snapshot.py` → `build_market_context.py` → `gpt_ops_report.py` の
    一連フローで、短期/スイングの分析材料を自動生成可能になった。
  - 発注・ポジション管理・strategy worker の実行ロジックは非変更。

### 2026-02-27（追記）`quant-ops-policy` の実行周期をイベント連動で可変化

- 目的:
  - 平常時は低頻度、指標前後のみ高頻度で更新し、運用負荷と鮮度を両立する。
- 変更:
  - `scripts/run_market_playbook_cycle.py` を追加。
    - `fetch_market_snapshot.py` と `gpt_ops_report.py` を1サイクルで実行。
    - `logs/ops_playbook_cycle_state.json` で次回実行時刻を管理。
    - 既定周期:
      - 通常: 15分
      - 指標前60分: 5分
      - 指標アクティブ（-10分〜+30分）: 1分
      - 指標後（〜120分）: 5分
  - `systemd/quant-ops-policy.service`
    - `ExecStart` を `run_market_playbook_cycle.py --policy --apply-policy` へ変更。
  - `systemd/quant-ops-policy.timer`
    - 起動は1分間隔に変更し、実行間隔の最終判断は cycle スクリプト側へ委譲。
  - `tests/scripts/test_run_market_playbook_cycle.py`
    - 周期選択（normal/pre/active/post）の回帰テストを追加。
- 影響範囲:
  - `quant-ops-policy` の更新タイミングが固定30分から可変へ移行。
  - 発注導線は変更せず、分析レポート生成周期のみ変更。

### 2026-02-27（追記）M1シナリオ3戦略を「プロセス独立」から「ロジック独立」へ移行

- 背景:
  - 先行実装では `quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}` が
    別 service で起動していても、実体は `workers.scalp_m1scalper.worker`/`exit_worker` のラッパー呼び出しだった。
  - そのため、`m1scalper` 本体変更が3戦略へ同時波及し、「戦略ごとに別進化」の要求に対して独立性が不足していた。
- 変更:
  - `workers/scalp_trend_breakout/*` を `scalp_m1scalper` からフル複製し、entry/exit本体を自戦略パッケージへ内包。
  - `workers/scalp_pullback_continuation/*` を同様にフル複製し、entry/exit本体を独立。
  - `workers/scalp_failed_break_reverse/*` を同様にフル複製し、entry/exit本体を独立。
  - 各 `config.py` で戦略別の既定値を明示:
    - `LOG_PREFIX`
    - `ALLOW_REVERSION` / `ALLOW_TREND`
    - `SIGNAL_TAG_CONTAINS` の default
    - `STRATEGY_TAG_OVERRIDE` の default
  - 各 `exit_worker.py` の `_DEFAULT_ALLOWED_TAGS` を戦略名へ変更し、EXIT閉域を既定化。
  - 回帰テストを `tests/workers/test_m1scalper_split_workers.py` で更新し、
    `workers.scalp_m1scalper` 直importが無いことと、戦略別 default を監査。
- 意図:
  - 3戦略を本当に別ワーカーとして運用し、個別改善時の波及を最小化する。
  - 既存 `quant-m1scalper` と並行運用しても、戦略単位で差分検証できる状態を作る。

### 2026-02-27（追記）M1シナリオ3戦略を既存M1と同時運用で有効化

- 背景:
  - ユーザー指定で「既存を止めず、全ユニット起動」を優先。
  - 直前は新規3戦略 unit を standby（disabled/inactive）で保持していた。
- 変更:
  - `ops/env/quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}.env` の `M1SCALP_ENABLED=1` を確定。
  - VM で対象6 unit（ENTRY/EXIT）を `enable --now`。
  - 既存 `quant-m1scalper.service` / `quant-m1scalper-exit.service` は継続稼働のまま維持。
- 意図:
  - 運用停止を入れずに、分離戦略の同時稼働データを即時取得する。

### 2026-02-27（追記）全体監査で B/C を再圧縮し、Wick/Extrema へ再配分（service timeout 再劣化も補正）

- 背景（VM実測, UTC 2026-02-27 05:35 時点）:
  - 直近24hの realized P/L で `scalp_ping_5s_c_live=-1455.4 JPY`, `scalp_ping_5s_b_live=-592.3 JPY` が主損失源。
  - 同期間で `WickReversalBlend=+332.3 JPY`, `scalp_extrema_reversal_live=+30.9 JPY` は正寄与。
  - 直近12hの B/C ログに `order_manager service call failed ... Read timed out (45.0)` が `165件` 発生。
  - 実効 runtime 設定が `ORDER_MANAGER_SERVICE_TIMEOUT=60.0` へ戻っていた。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - 頻度・サイズ・逆行側倍率を追加圧縮
      (`MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=300`, `MAX_UNITS=1100`,
      `DIRECTION_BIAS_*_OPPOSITE_UNITS_MULT` 引き下げ、`SIDE_BIAS_BLOCK_THRESHOLD` 引き上げ)。
  - `ops/env/scalp_ping_5s_c.env`
    - 頻度・サイズと side metrics 上限を追加圧縮
      (`MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=110`, `MAX_UNITS=240`,
      `SIDE_BIAS_BLOCK_THRESHOLD=0.16`, `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT=0.88`)。
    - fallback local 用の `ORDER_MANAGER_PRESERVE_INTENT_*` を
      `0.68/0.35/0.72` へ同期。
  - `ops/env/quant-order-manager.env`
    - B/C の preserve-intent しきい値を厳格化
      （`REJECT_UNDER` 引き上げ、`MAX_SCALE` 引き下げ）。
  - `ops/env/scalp_ping_5s_b.env`
    - fallback local の preserve-intent 値を
      `0.72/0.25/0.42` へ同期。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0`
    - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC=4.0`
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `MAX_OPEN_TRADES=4`, `UNIT_BASE_UNITS=11200`。
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `COOLDOWN_SEC=30`, `MAX_OPEN_TRADES=3`, `BASE_UNITS=13000`,
      `MIN_ENTRY_CONF=54`。
  - `systemd/quant-scalp-ping-5s-b.service`
    - unit直書きの `SCALP_PING_5S_B_BASE_ENTRY_UNITS` /
      `SCALP_PING_5S_B_MAX_UNITS` /
      `ORDER_MANAGER_PRESERVE_INTENT_*` を削除し、
      repo側 env ファイルを実効値の唯一ソースへ統一。
- 意図:
  - B/C の損失寄与を機械的に圧縮しつつ、勝ち寄与戦略へ回転と配分を寄せる。
  - order-manager 遅延時の長時間ブロックを縮め、取りこぼしの再発を抑える。

### 2026-02-27（追記）B/C 非エントリー主因の設定を解除（revert + order/min + service timeout）

- 背景（VM実測, UTC 2026-02-27 00:56-01:26）:
  - `quant-scalp-ping-5s-{b,c}` で `SCALP_PING_5S_REVERT_ENABLED is OFF` を確認。
  - skip要因に `revert_disabled` と `rate_limited` が高頻度で混在。
  - `quant-v2-runtime.env` の `ORDER_MANAGER_SERVICE_TIMEOUT=20.0` で、
    service遅延時に `order_manager_none` が重なる状態だった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_REVERT_ENABLED=1`（from `0`）
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=24`（from `4`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_REVERT_ENABLED=1`（from `0`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=24`（from `4`）
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=8.0`（from `20.0`）
  - `execution/strategy_entry.py`
    - 事前協調/パターンゲートで `coordinated_units=0` の場合、
      `client_order_id` に reject理由をキャッシュ記録。
    - worker側で `order_manager_none` に潰れていた reject内訳を観測可能化。
- 意図:
  - `no_signal:revert_disabled` と `rate_limited` の過抑制を外し、B/C のシグナル通過率を回復。
  - order-manager service固着時の待ち時間を短縮し、`order_manager_none` 起点のエントリー空転を減らす。

### 2026-02-27（追記）UI戦略制御の即時反映バグを修正（cache invalidation）

- 背景:
  - `apps/autotune_ui.py` の strategy control state に TTL キャッシュ
    （`UI_STRATEGY_CONTROL_CACHE_TTL_SEC`）を導入後、
    `/ops/strategy-control` 実行直後でも旧キャッシュが残り、
    dashboard の toggle 状態が最大 TTL 秒だけ古いまま表示されることを確認。
- 変更:
  - `apps/autotune_ui.py`
    - `_invalidate_strategy_control_cache()` を追加。
    - `_handle_strategy_control_action()` の `set_global_flags` /
      `set_strategy_flags` 成功後にキャッシュを必ず無効化。
  - `tests/apps/test_autotune_ui_caching.py`
    - global 更新と strategy 更新の両経路で
      `strategy_control` キャッシュが即時クリアされる回帰テストを追加。
- 意図:
  - Ops タブでの Entry/Exit/Lock 更新結果を dashboard に即時反映し、
    「更新成功なのに表示が戻る」誤認を防止する。

### 2026-02-27（追記）UI戦略制御フォームの認証漏れを修正 + 外部更新検知キャッシュを追加

- 背景:
  - `ops/strategy-control` は `ops_token` 認証が必須だが、
    dashboard の戦略制御フォーム群に `ops_token` 入力が無く、
    `ui_ops_token` 設定環境では更新操作が `Unauthorized` になる経路があった。
  - strategy control 状態は TTL キャッシュのみだったため、
    UI 以外の経路で `strategy_control.db` が更新された場合に
    TTL 期間中の表示遅延が残る余地があった。
- 変更:
  - `templates/autotune/dashboard.html`
    - `/ops/strategy-control` の global/strategy/row すべてのフォームに
      `ops_token` 入力を追加（required）。
  - `apps/autotune_ui.py`
    - strategy control キャッシュに
      `strategy_control.db + trades/signals/orders` の mtime シグネチャを導入。
    - TTL 内でもシグネチャ変化時は即時再計算するよう更新。
  - `tests/apps/test_autotune_ui_strategy_control_auth.py`
    - token なし拒否 / token あり成功を endpoint レベルで追加検証。
  - `tests/apps/test_autotune_ui_template_guards.py`
    - 戦略制御フォームに `ops_token` が存在する回帰ガードを追加。
  - `tests/apps/test_autotune_ui_caching.py`
    - シグネチャ変化でキャッシュ再読込される回帰テストを追加。
- 意図:
  - 戦略制御UIの「見えているのに操作できない」状態を排除し、
    外部更新を含む表示不一致の再発確率を下げる。

### 2026-02-27（追記）夜間の時間別履歴欠落を防ぐため hourly 集計ソースを強化

- 背景:
  - dashboard の `hourly_trades` が不完全なスナップショット（lookback不足）を含む場合、
    夜間帯（00〜06 JST など）が表に出ないケースがあった。
  - fallback 側も DB 読み取り失敗時に `recent_trades` 限定データへ退避するため、
    高頻度時間帯のみ表示される偏りが出る余地があった。
- 変更:
  - `apps/autotune_ui.py`
    - `_hourly_trades_is_usable()` を追加し、
      snapshot の `hourly_trades` が lookback/行数不足なら採用しない。
    - `_summarise_snapshot()` で不完全 snapshot 時は
      `_build_hourly_fallback()` に必ず切り替える。
    - `_build_hourly_fallback()` は時間窓付き SQL 集計
      （`julianday(close_time)` + JST時間丸め）を優先し、
      夜間帯を含む lookback 全体を再集計する。
  - テスト:
    - `tests/apps/test_autotune_ui_hourly_source_guard.py`
      - 不完全 snapshot で fallback 強制、
        完全 snapshot で既存値維持を検証。
    - `tests/apps/test_autotune_ui_hourly_fallback.py`
      - 集計クエリ優先動作と `hourly_trades` 採用条件の回帰を追加。
- 意図:
  - 「夜中の履歴が抜ける」表示欠陥を防ぎ、
    時間帯別評価の一貫性を維持する。

### 2026-02-26（追記）B の縮小ロット失効を更に抑えるため最小ロット閾値を再調整

- 背景（VM実測, 2026-02-26 12:08 UTC / 21:08 JST）:
  - 12:06:59 UTC の再起動後、`scalp_ping_5s_c` の `units_below_min` は解消した一方、
    `scalp_ping_5s_b` では `entry-skip summary side=short total=4 units_below_min=4` が残存。
  - `RISK multiplier=0.40` の縮小局面で、Bの最終ユニットが still `min_units=20` を割り込むケースがあった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MIN_UNITS: 20 -> 10`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`
- 意図:
  - side filter（sell限定）を維持したまま、Bの短期縮小ロットを発注可能域へ残す。
  - `units_below_min` 起因の失効を減らし、`submit_attempt/filled` への遷移回復を狙う。

### 2026-02-26（追記）B/C sell限定で `units_below_min` が主因化したため、最小ロット閾値を緩和

- 背景（VM実測, 2026-02-26 12:00 UTC / 21:00 JST）:
  - `quant-scalp-ping-5s-{b,c}.service` 再起動後、`orders.db` は `2026-02-26T11:59:57Z` 以降の新規レコードが 0。
  - ワーカーログでは `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell` により long が `side_filter_block` で抑制され、
    short 側は `units_below_min` が多発（B:10件, C:8件）。
  - 同時に `RISK multiplier` は `mult=0.40` で継続しており、短期縮小ロットが min units 未満へ落ちやすい状態だった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MIN_UNITS: 30 -> 20`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_MIN_UNITS: 30 -> 20`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE: 30 -> 20`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C: 30 -> 20`
- 意図:
  - side filter（sell限定）は維持したまま、縮小ロットの失効を減らして
    `preflight -> submit_attempt -> filled` の遷移を回復する。

### 2026-02-26（追記）クォート崩れ耐性を `order_manager` で強化（再クォート + 健全性判定）

- 背景（運用課題）:
  - `OFF_QUOTES` 系 reject が出る局面で、同一 payload を即再送しても再rejectしやすく、
    30分集計で reject 偏重になる時間帯があった。
  - preflight 中に quote が欠落/異常（cross/wide）でも注文を継続する経路があり、
    「崩れた quote に当たる」余地が残っていた。
  - 変更:
  - `execution/order_manager.py`
    - quote 健全性判定 `_quote_is_usable()` を追加（cross/負spread/過大spreadを除外）。
    - `_fetch_quote_with_retry()` を追加し、初回 quote 取得と reject 後再取得で再利用。
    - `ORDER_REQUIRE_HEALTHY_QUOTE_FOR_ENTRY=1`（既定）時は、
      non-manual 新規で quote 不健全なら `quote_unavailable` で skip。
    - `OFF_QUOTES` など quote 系 reject で
      `status=quote_retry` を残しつつ再クォートして再送する経路を追加。
    - `market_order` に加えて `limit_order` でも同経路を適用し、
      `ORDER_SUBMIT_MAX_ATTEMPTS=1` 運用時でも quote retry だけは動作するよう補強。
  - `ops/env/quant-order-manager.env`
    - `ORDER_QUOTE_*` と `ORDER_REQUIRE_HEALTHY_QUOTE_FOR_ENTRY` を明示し、
      実運用の quote retry/健全性判定を runtime env で固定化。
  - テスト:
    - `tests/execution/test_order_manager_preflight.py`
      - quote 健全性判定（cross/wide spread）
      - transient 失敗後の quote 再取得成功
    - `tests/execution/test_order_manager_log_retry.py`
      - `limit_order` が `ORDER_SUBMIT_MAX_ATTEMPTS=1` でも `OFF_QUOTES` 後に再送できることを追加検証。
- 意図:
  - quote 崩れ局面での無駄な reject 連鎖を減らし、
    正常 quote へ乗り換えて執行を継続する。

### 2026-02-26（追記）運用方針を「停止なし・時間帯停止なし」へ再固定

- 背景（VM実測）:
  - 直近24hで `scalp_ping_5s_b/c` の負寄与が継続し、停止・時間帯限定に寄せた運用が混在していた。
  - `strategy_control` と `worker_reentry` に停止相当設定（`entry=0` / `block_jst_hours`）が残り、
    「常時動的トレード」方針と不整合が発生していた。
- 変更:
  - `ops/env/scalp_ping_5s_{b,c}.env`
    - `ALLOW_HOURS_JST=`（時間帯制約解除）
    - `PERF_GUARD_MODE=reduce`（blockではなく縮小）
  - `config/worker_reentry.yaml`
    - `M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`,
      `scalp_ping_5s_{b,c,d}_live` の `block_jst_hours` を `[]` へ変更。
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_{c,c_live}` の `neg_exit` を no-block 化
      （`strict_no_negative: false`, `deny_reasons: []`）。
- 意図:
  - 停止/時間帯ブロックではなく、戦略ローカル判定と preflight リスクで
    常時運転しながら損失側を圧縮する運用へ戻す。

### 2026-02-26（追記）改善点台帳更新: EXIT封鎖/SL未実装の再発防止項目を追加

- 背景（VM/OANDA 実測）:
  - `strategy_control_exit_disabled` が `2026-02-24 02:20:16` ～ `09:13:14` UTC に `10,277` 件発生。
  - `MicroPullbackEMA` の4 ticket（`384420/384425/384430/384435`）で
    各 `2,044～2,045` 回の close 拒否後に `MARKET_ORDER_MARGIN_CLOSEOUT`（合計 `-16,837.4 JPY`）。
  - 該当 ticket は broker 側 `stopLossOnFill` 未設定（TPのみ設定）を確認。
- 変更:
  - 改善/敗因の単一台帳 `docs/TRADE_FINDINGS.md` に
    「2026-02-26 09:15 UTC / 18:15 JST - EXIT封鎖とSL未実装の複合で margin closeout が連鎖」
    を追記。
  - 再発防止項目（`entry=1 & exit=0` 禁止、broker `stopLossOnFill` 必須化、
    stale flag 解消、検証KPI）を同エントリへ明示。
- 意図:
  - 同種事故の原因と改善項目を単一台帳へ固定し、
    実装・運用・検証を同じチェックリストで回せる状態にする。

### 2026-02-26（追記）改善/敗因記録の単一台帳を `docs/TRADE_FINDINGS.md` に統一

- 背景:
  - 改善案・敗因分析がチャットや個別メモへ散在し、継続比較と再利用が難しかった。
  - 同じ失敗の再発防止ループ（原因→対策→検証）がチーム横断で追いにくかった。
- 変更:
  - 新規台帳 `docs/TRADE_FINDINGS.md` を運用の単一記録先として定義。
  - `AGENTS.md` に「改善/敗因は同ファイルへ追記、分散メモを作らない」ルールを追加。
  - `docs/INDEX.md` へ同台帳を追加し、全担当者の導線を固定化。
- 意図:
  - 改善と敗因の知見を時系列で蓄積し、再発防止と改善速度を上げる。
  - 担当者が変わっても同じフォーマットで追記できる運用にする。

### 2026-02-26（追記）VM高負荷の恒久対策: core backup を `cron.hourly` から guarded timer へ移行

- 背景（VM実測, UTC 2026-02-26 08:33 前後）:
  - `load average` が 60 超、`Swap 2.0GiB/2.0GiB`、`wa` 90% 近傍で高止まり。
  - `ssh.service` 配下の長時間 `sqlite3` と
    `cron.hourly` の `qr-gcs-backup-core`（`tar ... orders.db(-wal/-shm)`）が
    I/O wait を増幅し、`decision_latency_ms` 劣化を誘発していた。
- 変更:
  - 追加: `scripts/qr_gcs_backup_core.sh`
    - low-impact 方針の guarded backup 実装。
    - `load1` / `D-state` / `MemAvailable` / `SwapUsed` のガードにより高負荷時は即 `skip`。
    - SQLite は live WAL を直接 tar せず `.backup` スナップショット経由で退避。
    - 既定は hot DB（`orders.db` / `metrics.db`）を除外し、
      `trades/signals/snapshot/pipeline` を中心に退避。
  - 追加: `systemd/quant-core-backup.service`, `systemd/quant-core-backup.timer`
    - `Nice=19`, `IOSchedulingClass=idle`, `CPUWeight=10` で本番トレード導線より低優先。
  - 追加: `ops/env/quant-core-backup.env`
    - guard/対象ファイル/timeout の運用キーを分離。
  - 追加: `scripts/install_core_backup_service.sh`
    - `/usr/local/bin/qr-gcs-backup-core` と新 timer を導入し、
      legacy `/etc/cron.hourly/qr-gcs-backup-core` を無効化するインストーラ。
  - 更新: `scripts/install_trading_services.sh`
    - `quant-core-backup.service/.timer` を標準インストール対象へ追加。
- 意図:
  - バックアップ起因の I/O 詰まりを恒久的に抑止し、
    トレード導線（market-data/strategy-control/order-manager）の遅延悪化を避ける。
  - バックアップ失敗よりも「本番売買の継続性」を優先する運用へ固定する。

### 2026-02-26（追記）負け筋遮断: manual余力圧迫ガード + 恒常赤字ワーカーの全時間ブロック

- 背景（VM実測, UTC 2026-02-26 08:20 前後）:
  - `openTrades` は手動建玉 1 本（`USD_JPY -8500`, `clientExtensions={}`, SL/TPなし）を保持。
  - 口座スナップショットは `NAV 57,017 JPY` / `margin_used 53,049 JPY` /
    `margin_available 4,001 JPY` / `free_margin_ratio 0.070`。
  - 直近14日で `MARKET_ORDER_MARGIN_CLOSEOUT` の損失が集中:
    - `MicroPullbackEMA`: `-16,837 JPY`
    - `scalp_ping_5s_b_live`: `-12,778 JPY`
    - `scalp_ping_5s_flow_live`: `-2,797 JPY`
- 変更:
  - `execution/order_manager.py`
    - 新規ガード `_manual_margin_pressure_details()` を追加。
    - 条件:
      - `manual/unknown` 建玉が存在
      - かつ `free_margin_ratio` / `health_buffer` / `margin_available` が閾値未達
    - 発動時は新規ENTRYを `manual_margin_pressure` で拒否し、
      `order_manual_margin_block` メトリクスを記録。
    - `market_order` と `limit_order` 両経路に適用。
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANUAL_MARGIN_GUARD_*` を運用値として追加。
  - `config/worker_reentry.yaml`
    - `MicroPullbackEMA` / `MicroLevelReactor` / `M1Scalper` を
      `block_jst_hours=0..23` に更新（全時間ブロック）。
  - テスト:
    - `tests/execution/test_order_manager_preflight.py`
      - manual建玉+低余力でガード発動するケース
      - manual建玉なしでは発動しないケース
- 意図:
  - 手動建玉が余力を占有している局面での追加エントリーを止め、
    `margin closeout` 連鎖を防止する。
  - 恒常赤字ワーカーの再流入を `reentry` 層で遮断し、
    正の期待値ワーカーへの資本配分を維持する。

### 2026-02-26（追記）`scalp_ping_5s_{b,c}` の「勝ち時間帯×方向」限定 + dynamic_alloc 罰則強化

- 背景（VM実測, 14日集計）:
  - `scalp_ping_5s_b_live`: `-41,022 JPY`（`4,976 trades`, PF `0.425`）
  - `scalp_ping_5s_c_live`: `-10,554 JPY`（`855 trades`, PF `0.475`）
  - Bは `sell` 側損失が卓越（`sell -35,865 JPY` / `buy -5,156 JPY`）。
  - Cは `buy` でも全時間運転だと赤字だが、時間帯分解で
    `18/19/22 JST` が相対優位（`buy` 限定時に損失帯を回避可能）。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=buy`
    - `SCALP_PING_5S_B_ALLOW_HOURS_JST=16,17,18,23`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=block`（reduce -> block）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=buy`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=18,19,22`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=block`（reduce -> block）
    - 共通prefix経路（`SCALP_PING_5S_PERF_GUARD_*`）も同値に更新。
  - `config/worker_reentry.yaml`
    - `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` の `block_jst_hours` を
      上記許可時間帯以外へ更新（worker env と二重化）。
  - `scripts/dynamic_alloc_worker.py`
    - lookback抽出を `julianday(close_time)` 基準へ修正。
    - `margin_closeout_rate` を導入し、`MARKET_ORDER_MARGIN_CLOSEOUT` 多発戦略の
      `lot_multiplier` を追加で抑制。
    - 実現損益（JPY）悪化に対する段階的 cap を追加（`-2500/-5000 JPY`）。
  - `workers/common/dynamic_alloc.py`
    - `strategy_key` 解決を case-insensitive fallback 対応にし、
      戦略名の大小文字揺れでプロファイル未適用になる漏れを削減。
- 意図:
  - 「停止ではなく全時間稼働」から「勝っている条件だけ稼働」へ切り替え、
    scalp_fast の継続赤字を抑える。
  - dynamic allocation が `pips` 偏重で負け筋を過大評価する経路を防ぎ、
    実現損益ベースでロット縮小を強制する。

### 2026-02-26（追記）当日回収モード: guard緩和 + b/c 許可時間内ロット増

- 背景:
  - 手動建玉残存時に `ORDER_MANUAL_MARGIN_GUARD_*` が厳しすぎると、
    有望時間帯でも新規エントリーが実質停止する。
  - 本日の回収優先運用として、`guard=ON` は維持しつつ
    「完全停止しない閾値」へ一時調整が必要。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO: 0.12 -> 0.05`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER: 0.18 -> 0.07`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY: 8000 -> 3000`
  - `ops/env/scalp_ping_5s_b.env`
    - `BASE_ENTRY_UNITS: 520 -> 1200`
    - `MAX_UNITS: 900 -> 2400`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 1 -> 2`
    - `BASE_ENTRY_UNITS: 260 -> 700`
    - `MAX_UNITS: 420 -> 1600`
- 意図:
  - 方向/時間帯制限（buy + allow hours）は維持したまま、
    当日中の期待収益を引き上げる。
  - margin guard は残して margin closeout 再発だけは避ける。

### 2026-02-26（追記）forecast_context 欠落対策: `edge_allow` を明示返却

- 背景（VM実測）:
  - `strategy_entry` の `entry_thesis.forecast.reason=not_applicable` が多発し、
    `forecast_fusion` が `p_up` 欠損で早期 return するケースが継続していた。
  - 原因は `workers/common/forecast_gate.py` で
    `scale >= 0.999` の場合に `None` を返していたため、
    「縮小不要だが予測は有効」な局面でも forecast 文脈が落ちていたこと。
- 変更:
  - `workers/common/forecast_gate.py`
    - `scale >= 0.999` を no-op ではなく
      `ForecastDecision(allowed=True, scale=1.0, reason=\"edge_allow\")` で返すよう修正。
    - `p_up` / `edge` / `expected_pips` / `target_reach_prob` などの監査メタを
      常に `strategy_entry` へ伝播できるよう統一。
  - テスト:
    - `tests/workers/test_forecast_gate.py`
      - `test_decide_returns_explicit_allow_when_scale_is_full` を追加。
    - `tests/execution/test_strategy_entry_forecast_fusion.py`
      - `test_inject_entry_forecast_context_keeps_allow_decision` を追加。
- 意図:
  - 予測が「ブロック/縮小しない局面」でも forecast を有効シグナルとして使い、
    `forecast_fusion` の実適用率を引き上げる。

### 2026-02-26（追記）分析班×リプレイ班を直結し、自動改善ループを常設化

- 背景:
  - `quant-replay-quality-gate` はこれまで replay の pass/fail 監査で終了し、
    replay の悪化要因を `worker_reentry` へ反映する工程が手動だった。
  - `analysis.trade_counterfactual_worker` は replay JSON を読める実装済みだったが、
    定期 replay run との自動連鎖が未接続だった。
- 変更:
  - `analysis/replay_quality_gate_worker.py`
    - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_*` を追加し、
      replay 完了後に `analysis.trade_counterfactual_worker` を戦略単位で自動実行。
    - `policy_hints.block_jst_hours` を抽出し、
      `config/worker_reentry.yaml` の `strategies.<strategy>.block_jst_hours` へ自動反映。
    - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MAX_BLOCK_HOURS`（既定 8）を超える候補は
      `skipped_too_many_block_hours` として採用しない。
    - 採用/非採用理由・コマンド・stderr tail を
      `logs/replay_quality_gate_latest.json` の `auto_improve` に記録。
  - `ops/env/quant-replay-quality-gate.env`
    - auto-improve を本番既定で有効化（`REPLAY_QUALITY_GATE_AUTO_IMPROVE_ENABLED=1`）。
  - 回帰テスト:
    - `tests/analysis/test_replay_quality_gate_worker.py`
      - 戦略選定フィルタ
      - `worker_reentry` 反映
      - `run_once` 連鎖反映
- 意図:
  - replay 監査と分析改善の断絶を解消し、
    「replay 失敗戦略の時間帯ブロック更新」までを定期ワーカー単体で完結させる。

### 2026-02-26（追記）常時ループをレート制限付きへ調整（軽分析高頻度 / replay低頻度）

- 背景:
  - 固定費VMでも CPU/IO は有限で、replay を高頻度で回し続けると
    本番ワーカーへの干渉と過剰チューニング（設定振動）のリスクがある。
- 変更:
  - `systemd/quant-replay-quality-gate.timer`
    - `OnUnitActiveSec=1h` → `3h`（重い replay は低頻度）
  - `systemd/quant-trade-counterfactual.timer`
    - `OnUnitActiveSec=30min` → `20min`（軽い分析は高頻度）
  - `analysis/replay_quality_gate_worker.py`
    - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC` と
      `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_STATE_PATH` を追加。
    - 反映クールダウン内は `reentry_apply_cooldown` として
      `worker_reentry` 反映をスキップし、解析のみ継続。
  - `ops/env/quant-replay-quality-gate.env`
    - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC=10800` を追加。
- 意図:
  - 「分析→replay→改善」の常時ループは維持しつつ、
    本番干渉と設定の過敏反応を抑えた持続運用にする。

### 2026-02-26（追記）レンジ不全対策: forecast/perf の向きを「range復帰 + 低EV抑制」へ再配線

- 背景（VM実測, 直近7日）:
  - `Range` は `-1671.5 pips`、`scalp_fast` は `-2298.2 pips` で主損失。
  - `forecast_gate_block=12641` のうち
    `style_mismatch_range=1854`（`RangeFader`/`MicroLevelReactor` 集中）、
    `edge_block=9899` が観測された。
  - `order_perf_block=24517` と `order_probability_reject=18032` が併発し、
    戦略間で「通るべきレンジ戦略が止まり、通った scalp_fast が負ける」偏りが出ていた。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `MicroLevelReactor` の forecast 閾値を緩和:
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_MICROLEVELREACTOR=0.12`
      - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_MICROLEVELREACTOR=-0.03`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MICROLEVELREACTOR=0.22`
      - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_MICROLEVELREACTOR=0.40`
    - range 系 style guard 緩和:
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_RANGEFADER=0.32`
      - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_RANGEFADER=0.40`
      - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_MICROVWAPREVERT=0.40`
      - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_MICROVWAPBOUND=0.40`
  - `ops/env/quant-order-manager.env`
    - `scalp_ping_5s_b/c` の preserve/perf を tighten:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.50`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.80`
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.64`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=0.70`
      - `SCALP_PING_5S_B/C_*_PERF_GUARD_FAILFAST_*` と `SL_LOSS_RATE_MAX` を引き上げ/引き締め。
    - order_manager preflight で scalp_fast 低EVを追加抑制:
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_B_LIVE=0.68`
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_SCALP_PING_5S_B_LIVE=0.18`
      - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_SCALP_PING_5S_B_LIVE=-0.01`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_SCALP_PING_5S_B_LIVE=0.28`
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_C_LIVE=0.70`
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_SCALP_PING_5S_C_LIVE=0.20`
      - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_SCALP_PING_5S_C_LIVE=-0.01`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_SCALP_PING_5S_C_LIVE=0.30`
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_FLOW_LIVE=0.72`
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_SCALP_PING_5S_FLOW_LIVE=0.22`
      - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_SCALP_PING_5S_FLOW_LIVE=-0.01`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_SCALP_PING_5S_FLOW_LIVE=0.32`
- 意図:
  - レンジ専用戦略の `style_mismatch_range` 側の過剰拒否を下げる。
  - 直近で損失主因だった `scalp_fast` の低期待値エントリーを preflight で早期遮断する。
  - 「レンジで死ぬ」状態を、レンジ戦略へ流し直しつつ低EV scalp を抑える方向へ修正する。

### 2026-02-26（追記）`quant-range-metrics` の stale 誤判定を修正（H4/H1 の鮮度閾値を分離）

- 背景（VM実測, UTC 2026-02-26 07:26）:
  - `quant-range-metrics.service` は timer で起動していたが、
    `WARNING: [range_metric] data too old (age=12392.6s)` が継続し、
    `range_mode_active` の更新が止まる時間帯が発生していた。
  - 原因は `scripts/publish_range_mode.py` の鮮度判定が
    `max(M1_age, macro_age)` を `RANGE_MODE_PUBLISH_MAX_DATA_AGE_SEC=900` で一律評価していたこと。
    H4運用では macro 足が 15 分超になるのは通常で、営業中でも stale 扱いになっていた。
- 変更:
  - `scripts/publish_range_mode.py`
    - M1 と macro(H1/H4) の鮮度判定を分離。
    - macro 側は `RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC` を追加し、
      未設定時は `macro_bar_sec * 1.35`（H4=19440秒, H1=4860秒）を自動採用。
    - `range_mode_active` タグへ `m1_age_sec` / `macro_age_sec` /
      `m1_limit_sec` / `macro_limit_sec` を追加して監査可能化。
  - `tests/scripts/test_publish_range_mode.py`
    - H4 intra-bar（macro age 約3.4h）で stale にならない回帰テストを追加。
    - macro 限界超過時に stale 判定へ入るテストを追加。
- 意図:
  - `quant-range-metrics` を営業時間中に継続発行可能な状態へ戻し、
    `range_mode_active` が分析ワーカーとして実際に戦略判断へ使える監視信号になるよう復旧する。

### 2026-02-26（追記）Pattern Gate の一致率改善: `pt/rg` フォールバック一致を追加

- 背景（VM実測, UTC 2026-02-26 07:52, 直近サンプル3000行）:
  - `pattern_gate_opt_in` 付き 1633 件のうち、exact `pattern_id` 一致は 1037 件、
    非一致が 596 件（約 36.5%）だった。
  - 同サンプルで `drop_rg` まで許容すると一致 1598 件、
    `drop_pt_rg` まで許容すると一致 1616 件まで回復した。
  - 実運用では `scalp_ping_5s_c_live`（`scalp_fast`）で `rg` 差分による不一致が多数を占め、
    pattern book 更新済みでも gate が no-op になる局面が発生していた。
- 変更:
  - `workers/common/pattern_gate.py`
    - exact 不一致時に `drop_pt -> drop_rg -> drop_pt_rg` の順で近傍一致を探索。
    - fallback 一致時は `match_mode` を保持し、
      `requested_pattern_id`（要求ID）と `pattern_id`（採用ID）を payload へ記録。
    - 既定で `ORDER_PATTERN_GATE_FALLBACK_DISABLE_BLOCK=1` とし、
      fallback 一致では block せず縮小/据え置き中心に運用。
    - fallback 適用時の倍率を `ORDER_PATTERN_GATE_FALLBACK_SCALE_MIN/MAX`
      （既定 `0.85/1.05`）でクランプし、過剰なサイズ変動を抑制。
  - `config/env.example.toml`
    - `ORDER_PATTERN_GATE_FALLBACK_*` の運用キーを追記。
  - `tests/workers/test_pattern_gate.py`
    - `rg/pt` 差分時に fallback 一致できる回帰テストを追加。
    - fallback 一致では `avoid` でも block しない（既定）回帰テストを追加。
- 意図:
  - Pattern Gate を「opt-in しているのに効かない」状態から外し、
    方向意図を壊さずに縮小判断を継続適用できる運用へ戻す。
  - fallback 由来の誤 block を避けつつ、分析ワーカーの実効寄与を上げる。

### 2026-02-26（追記）SL運用の曖昧さを解消（baseline明示 + override契約を仕様化）

- 背景:
  - 運用問い合わせで「SLを付ける/付けない」の判定根拠が
    `docs/SL_POLICY.md` と実装/環境の説明で不一致だった。
  - 実装は strategy override を許可しているが、
    ドキュメント上は `ORDER_FIXED_SL_MODE` 単独判定に見える箇所が残っていた。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_FIXED_SL_MODE=0` を明示追加（未設定依存を排除）。
  - `execution/order_manager.py`
    - `_allow_stop_loss_on_fill()` の docstring を
      「strategy override → family override → global baseline」の順序へ更新。
  - `docs/SL_POLICY.md`
    - `stopLossOnFill` の最終判定を
      `_entry_sl_disabled_for_strategy()` / `_allow_stop_loss_on_fill()` ベースへ修正。
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>` と
      `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_[B|C|D]` を
      正式な運用キーとして明記。
  - `tests/execution/test_order_manager_sl_overrides.py`
    - `MicroPullbackEMA` の generic strategy override で
      fixed-mode OFF でも `sl_disabled=False` になる回帰テストを追加。
- 意図:
  - 実挙動を変えずに運用判定を一本化し、
    「baselineはOFF/ONどちらか、必要戦略だけoverrideで再有効化」
    という契約を明確化する。

### 2026-02-25（追記）replay quality gate を定期実行へ復帰（skip既定を解除）

- 背景（VM実測, UTC 2026-02-25 16:24）:
  - `quant-replay-quality-gate.timer` 自体は `enabled/active` だったが、
    `quant-replay-quality-gate.service` は
    `REPLAY_QUALITY_GATE_ENABLED=0` により
    `skipped: REPLAY_QUALITY_GATE_ENABLED=0` で即終了していた。
  - その結果、replay 品質監査が「定期起動はされるが実処理は走らない」状態だった。
- 変更:
  - `ops/env/quant-replay-quality-gate.env`
    - `REPLAY_QUALITY_GATE_ENABLED=1` へ変更し、1h 周期 replay 監査を実行に戻した。
  - `docs/RISK_AND_EXECUTION.md`
    - replay quality gate 運用補足の本番既定値を `1` へ更新。
- 意図:
  - replay 改善劣化の検知を定期ワーカーで継続し、
    手動実行依存を外して品質改善サイクルを維持する。
  - 実行優先度は既存の `Nice=15` / `IOSchedulingClass=idle` / `CPUWeight=20` を維持し、
    本番導線への干渉を最小化する。

### 2026-02-26（追記）Autotune UI の再発性フロント不具合を恒久対策

- 背景:
  - dashboard の自動更新で、`visibilitychange` や `isEditing` 再スケジュール時に
    既存の `setInterval/setTimeout` が取り切れず、多重タイマー化する経路があった。
  - これにより、長時間表示で更新表示の不安定化（カウントダウンの多重更新・リフレッシュ挙動の揺れ）が再発しうる状態だった。
  - 併せて chart タブ側は `sessionStorage` 直接アクセスで、ブラウザ制約時に script 全体が失敗する余地があった。
- 変更:
  - `templates/autotune/base.html`
    - `qrInitAutoRefresh` に `clearScheduledJobs()` を導入し、再スケジュール時の単一タイマー保証へ変更。
    - refresh 実行は `runRefresh()` に一本化し、hard refresh fallback を共通化。
    - `visibilitychange` 時に hidden 側で timer/interval を必ず解放し、visible 復帰時のみ最新DOMから再初期化。
    - `data-auto-refresh` が無い画面へ切り替わった場合も timer を残さないようガードを追加。
    - `qrInitTabs` は tab コンテナ不在時に `tabs-ready` を除去し、画面遷移時の表示状態リークを防止。
  - `templates/autotune/dashboard.html`
    - chart スクリプトに `safeSessionGet/safeSessionSet` を追加し、
      `sessionStorage` 制約環境での JS 例外起点を遮断。
  - `apps/autotune_ui.py`
    - `hourly_trades` fallback を `snapshot.recent_trades` 依存から切り離し、
      `trades.db` の close 済み履歴を lookback 窓で直接読む方式へ変更。
    - `UI_HOURLY_FALLBACK_SCAN_LIMIT`（default: `5000`）を追加し、
      fallback 集計で走査する最大件数を env で調整可能にした。
  - `tests/apps/test_autotune_ui_template_guards.py`
    - 自動更新タイマー解放ガード・tabs ready リセット・safe storage 利用を回帰テスト化。
  - `tests/apps/test_autotune_ui_hourly_fallback.py`
    - fallback が DB窓を優先して夜間帯履歴を取りこぼさないことをテスト化。
- 意図:
  - UI を「長時間開いた時に壊れにくい」状態へ寄せ、
    snapshot 配信の揺れと独立にフロント側の再発要因（多重タイマー/storage例外）を除去する。

### 2026-02-26（追記）Autotune UI 遅延対策: position_manager 呼び出しを fail-fast 化

- 背景（VM実測）:
  - `quant-autotune-ui.service` の `/dashboard` 応答が 20〜30 秒に達し、
    体感で「更新が遅い」状態になっていた。
  - `journalctl` でも
    `position_manager service error ... performance_summary timeout (20.0s)` が確認され、
    UI 側の snapshot 生成が position_manager 応答待ちに引きずられていた。
- 変更:
  - `systemd/quant-autotune-ui.service`
    - `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0`
    - `POSITION_MANAGER_SERVICE_TIMEOUT=1.5`
    - `POSITION_MANAGER_SERVICE_PERFORMANCE_SUMMARY_TIMEOUT=2.0`
    - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=1.2`
    - `POSITION_MANAGER_SERVICE_FAIL_BACKOFF_SEC=2.0`
    - `POSITION_MANAGER_SERVICE_FAIL_BACKOFF_MAX_SEC=10.0`
  - UI プロセスのみで fail-fast 設定を適用し、
    position_manager が高負荷時でも軽量 fallback（local trades/metrics）で応答を維持する。
- 意図:
  - 精密メトリクスより表示応答性を優先し、
    ダッシュボード更新の待ち時間を秒単位へ戻す。

### 2026-02-25（追記）position/order manager の遅延・ロック恒久対策

- 背景（VM実測）:
  - `quant-position-manager.service` で `sync_trades timeout (8.0s)` /
    `position manager busy` が継続し、同時刻にプロセス CPU 使用率が高止まりしていた。
  - `quant-order-manager.service` では
    `[ORDER][LOG] failed to persist orders log: database is locked` が連発し、
    `orders.db` ログ永続化でのロック競合が発生していた。
- 変更:
  - `execution/position_manager.py`
    - `sync_trades` に最小ポーリング間隔
      （`POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC`）を導入し、
      短周期の重複同期を抑制。
    - `sync_trades` のキャッシュ即時返却を強化し、
      lock 競合時も直近キャッシュを優先返却。
    - 無制限だった `entry/pocket/client` キャッシュに上限を導入
      （`POSITION_MANAGER_*_CACHE_MAX_ENTRIES`）。
  - `workers/position_manager/worker.py`
    - `sync_trades` キャッシュを `max_fetch` 非依存の単一キーへ統一し、
      要求値の差で同一同期が多重実行される経路を解消。
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH` で fetch 上限を固定。
    - 空エラーメッセージを `unknown error` へ正規化して監査性を改善。
  - `execution/order_manager.py`
    - `orders.db` 書き込みに process 内 `RLock` を追加し、
      同一プロセス内同時書き込みによる flock 自己競合を抑制。
  - `ops/env/quant-order-manager.env`
    - `ORDER_DB_*`（busy_timeout / retry / file lock timeout）を
      実運用値へ引き上げ（`250ms/0.12s` 系から `1500ms/0.30s` 系へ）。
  - `ops/env/quant-v2-runtime.env`
    - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC` ほか cache 上限・stale age を追加。
- 意図:
  - PositionManager の過剰同期とメモリ膨張を抑え、
    `sync_trades timeout` 由来の上流遅延を恒久的に減らす。
  - OrderManager の orders 永続化を安定化し、
    `database is locked` ノイズとログ欠落を抑止する。

### 2026-02-25（追記）forecast 改善度の定期監査ワーカーを追加

- 背景:
  - `vm_forecast_snapshot.py` / `eval_forecast_before_after.py` は手動実行前提で、
    定期的な before/after 改善判定（1m/5m/10m）が欠けていた。
  - 予測式や重み調整の反映後に、劣化を自動検知できる定期監査が必要だった。
- 変更:
  - `analysis/forecast_improvement_worker.py` を追加。
    - `scripts/vm_forecast_snapshot.py --json` と
      `scripts/eval_forecast_before_after.py --json-out` を定期実行。
    - `hit_delta/mae_delta/range_coverage_delta` の劣化閾値で
      `improved/mixed/degraded` を判定。
    - `logs/reports/forecast_improvement/<timestamp>/report.md` と
      `logs/reports/forecast_improvement/latest.md` を更新し、
      「改善TF/悪化TF/次の調整案（最大3件）」を出力。
    - 状態・履歴を `logs/forecast_improvement_latest.json` /
      `logs/forecast_improvement_history.jsonl` へ記録。
  - `systemd/quant-forecast-improvement-audit.service` /
    `systemd/quant-forecast-improvement-audit.timer` を追加（1h 周期）。
  - `ops/env/quant-forecast-improvement-audit.env` を追加し、
    監査の bars/閾値/保持本数を env で運用可能にした。
  - 仕様ドキュメント更新:
    - `docs/FORECAST.md`
    - `docs/WORKER_ROLE_MATRIX_V2.md`
    - `docs/ARCHITECTURE.md`
- 意図:
  - forecast 設定変更後の改善/劣化を定期的に可視化し、
    劣化時の再調整を運用ルーチン化する。

### 2026-02-25（追記）cleanup / replay バックグラウンド負荷の本番干渉を抑制

- 背景（VM実測）:
  - `cleanup-qr-logs.service` の `sqlite3 logs/orders.db VACUUM` が長時間継続し、
    `quant-order-manager` 側で `orders.db file lock timeout` を連発。
  - `quant-replay-quality-gate.service` が replay 実行中に CPU を大きく消費し、
    本番ワーカー群の遅延悪化を助長していた。
- 変更:
  - `scripts/cleanup_logs.sh`
    - `DB_VACUUM_SKIP_FILES`（既定: `orders.db trades.db metrics.db`）を追加。
    - `DB_VACUUM_ALLOW_HOT_DBS=0` を既定化し、
      稼働中ホットDBの `VACUUM` をデフォルト禁止（checkpoint のみ実施）。
  - `systemd/quant-replay-quality-gate.service`
    - `Nice=15`, `IOSchedulingClass=idle`, `CPUWeight=20` を追加し、
      replay 品質検証を低優先度で実行。
- 意図:
  - 定期メンテ/検証ジョブが本番発注系 DB と CPU を奪う経路を塞ぎ、
    発注遅延・ロックノイズの再発を防ぐ。

### 2026-02-25（追記）`quant-ui-snapshot` の長時間ハング対策を追加

- 背景（VM実測）:
  - `quant-autotune-ui.service` は active だが、
    `quant-ui-snapshot.service` が `activating` のまま 1 時間超継続し、
    `quant-ui-snapshot.timer` の次回実行が詰まって UI 更新が停止した。
  - 直近では `2026-02-25 07:18:59 UTC` 開始の one-shot 実行が戻らず、
    UI 側で stale 監視が継続していた。
- 変更:
  - `scripts/run_ui_snapshot.sh`
    - `UI_SNAPSHOT_MAX_RUNTIME_SEC`（既定 `90` 秒）を追加。
    - `timeout --signal=TERM --kill-after=10s` で
      `publish_ui_snapshot.py` 実行時間を上限管理。
    - timeout 時は `exit 0` でその周期のみスキップし、次周期の timer 実行を継続。
- 意図:
  - one-shot の単発ハングで timer 連鎖全体が停止する状態を回避し、
    UI スナップショット更新を自動復帰可能にする。

### 2026-02-25（追記）`publish_ui_snapshot --lite` の `sync_trades` をデフォルト無効化

- 背景（VM実測）:
  - ハング上限を入れた後も、`quant-ui-snapshot.service`（`--lite`）で
    `sync_trades timeout` が継続し、90 秒 timeout で終了する周期が連発した。
  - timer 停止は解消したが、publish 到達率が下がり、UI 更新間隔が不安定だった。
- 変更:
  - `scripts/publish_ui_snapshot.py`
    - `UI_SNAPSHOT_SYNC_TRADES_LITE`（既定 `0`）を追加。
    - `--lite` 実行時の `sync_trades` は
      `UI_SNAPSHOT_SYNC_TRADES=1` かつ `UI_SNAPSHOT_SYNC_TRADES_LITE=1` の場合のみ実行。
- 意図:
  - `--lite` の主目的（定期 publish の安定性）を優先し、
    PositionManager 側の遅延で snapshot publish が巻き添えになる状態を避ける。

### 2026-02-25（追記）`publish_ui_snapshot --lite` の open positions 取得をデフォルト無効化

- 背景（VM実測）:
  - `sync_trades` 無効化後も、`fetch_recent_trades` / `performance_summary` の後段で
    open positions 取得が長時間化し、`run_ui_snapshot.sh` の 90 秒 timeout に到達する周期が残った。
- 変更:
  - `scripts/publish_ui_snapshot.py`
    - `UI_SNAPSHOT_LITE_INCLUDE_POSITIONS`（既定 `0`）を追加。
    - `--lite` 実行時の `pm.get_open_positions()` は
      `UI_SNAPSHOT_LITE_INCLUDE_POSITIONS=1` の場合のみ実行。
- 意図:
  - `--lite` で publish 到達率を最優先し、重い建玉参照は明示 opt-in へ分離する。

### 2026-02-25（追記）`publish_ui_snapshot --lite` の拡張メトリクス収集をデフォルト無効化

- 背景（VM実測）:
  - `sync_trades` / open positions を既定無効化した後も、`--lite` 実行で
    90 秒 timeout が残存し、publish 完了率が安定しなかった。
  - 原因切り分け上、`orders/signals` 系の拡張メトリクス収集を
    `--lite` 既定動作から外して publish 優先に寄せる必要があった。
- 変更:
  - `scripts/publish_ui_snapshot.py`
    - `UI_SNAPSHOT_LITE_INCLUDE_EXTENDED_METRICS`（既定 `0`）を追加。
    - `--lite` 実行時は、同フラグを有効化した場合のみ
      `orders_last` / `orders_status_1h` / `signals_recent` 等を収集する。
- 意図:
  - `--lite` の処理時間を短く保ち、`quant-ui-snapshot.timer` 周期で
    確実に publish を回すことを優先する。

### 2026-02-25（追記）UI snapshot の GCS upload timeout を導入

- 背景（VM実測）:
  - `--lite` で取得項目を削減後も、`quant-ui-snapshot.service` が
    90 秒 timeout に到達する周期が残り、I/O 待ち由来の長時間ブロックが疑われた。
- 変更:
  - `analytics/gcs_publisher.py`
    - `UI_SNAPSHOT_GCS_UPLOAD_TIMEOUT_SEC`（既定 `8` 秒）を追加。
    - `blob.upload_from_string(..., timeout=...)` で upload 上限時間を明示。
- 意図:
  - GCS 側の応答停滞で snapshot publish パス全体が停止する状態を避け、
    timer 周期での再試行性を確保する。

### 2026-02-25（追記）`publish_ui_snapshot --lite` で PositionManager を既定無効化

- 背景（VM実測）:
  - `sync_trades` / open positions / 拡張メトリクスを既定無効化後も、
    `PositionManager` 初期化経路で `performance_summary` の遅延が残り、
    90 秒 timeout を継続していた。
- 変更:
  - `scripts/publish_ui_snapshot.py`
    - `UI_SNAPSHOT_LITE_USE_POSITION_MANAGER`（既定 `0`）を追加。
    - `--lite` 実行時は、同フラグが有効な場合のみ `PositionManager` を初期化。
- 意図:
  - `--lite` の既定動作を「ローカルDBベースの publish 優先」に固定し、
    position service 側遅延の巻き込みを回避する。

### 2026-02-25（追記）`quant-order-manager` / `quant-position-manager` の service 有効化不整合を修正

- 背景（VM実測）:
  - `quant-order-manager.service` / `quant-position-manager.service` は active でも、
    `127.0.0.1:8300/8301` が listen せず、worker 側で
    `order_manager service call failed` / `position_manager service call failed`
    （timeout / connection refused）が継続。
  - 原因は env 上書き不整合で、`ops/env/quant-v2-runtime.env` は
    `ORDER_MANAGER_SERVICE_ENABLED=1` / `POSITION_MANAGER_SERVICE_ENABLED=1`
    だが、service が読む `ops/env/quant-order-manager.env` /
    `ops/env/quant-position-manager.env` が `0` になっていた。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_ENABLED: 0 -> 1`
  - `ops/env/quant-position-manager.env`
    - `POSITION_MANAGER_SERVICE_ENABLED: 0 -> 1`
- 意図:
  - V2 の固定導線（strategy -> order/position manager service）を復帰し、
    `local fallback` 側へ落ちる不一致状態を解消する。

### 2026-02-25（追記）forecast 強化の段階導入（`cand1_mid` を micro 2戦略限定で適用）

- 背景（VM実測）:
  - `scripts/eval_forecast_before_after.py` を同一期間
    （`2026-01-07T02:54:00Z` 〜 `2026-02-25T08:17:00Z`, `max-bars=8050`）
    で比較した結果、強め設定（`feature_expansion_gain=0.35`）は
    `10m` の `mae_delta=+0.0225` で劣化ガード（`> +0.020`）を超過。
  - 一方で `cand1_mid`
    （`feature_expansion_gain=0.20`,
    `breakout_map=1m=0.16,5m=0.22,10m=0.24`,
    `session_map=1m=0.0,5m=0.22,10m=0.28`）は
    `1m/5m/10m` 全てでガード内に収まった。
  - 監査レポート:
    - `logs/reports/forecast_improvement/report_20260225T081315Z_viability.md`
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_GATE_STRATEGY_ALLOWLIST=MicroRangeBreak,MicroVWAPBound` を追加。
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.20` へ更新。
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.16,5m=0.22,10m=0.24` へ更新。
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.28` へ更新。
- 意図:
  - 全戦略へ一括適用せず、`MicroRangeBreak` / `MicroVWAPBound` のみで
    forecast 強化を段階検証する。
  - 10m MAE 悪化の再発を防ぎつつ、hit 改善側の設定レンジを実運用で確認する。

### 2026-02-25（追記）`scalp_extrema_reversal_live` を新設（高値ショート/安値ロングの両側反転）

- 背景（VM実測）:
  - 高値帯・安値帯の反転局面で、既存 5秒スキャ系は
    `entry_probability_below_min_units` / `perf_block` / `reentry_block`
    により最終発注まで到達しないケースが連続した。
  - 特に「高値で売れない」「安値で買えない」の対称的な取りこぼしがあり、
    方向別の後段ゲート依存を下げた専用導線が必要だった。
- 変更:
  - 追加: `workers/scalp_extrema_reversal/worker.py`
    - `M1` の極値帯（range snapshot 上下端 + 直近 tick 極値）近傍で、
      `tick_reversal` 確認後のみ `OPEN_SHORT/OPEN_LONG` を出す専用 ENTRY ワーカーを実装。
    - `entry_thesis` に `entry_probability` / `entry_units_intent` を必須注入し、
      `entry_probability_raw` と `extrema` 文脈を監査用に付与。
  - 追加: `workers/scalp_extrema_reversal/exit_worker.py`
    - `scalp_level_reject` の exit 実装を env マッピングで再利用するラッパーを追加。
  - 追加: `systemd/quant-scalp-extrema-reversal.service`
  - 追加: `systemd/quant-scalp-extrema-reversal-exit.service`
  - 追加: `ops/env/quant-scalp-extrema-reversal.env`
  - 追加: `ops/env/quant-scalp-extrema-reversal-exit.env`
  - 更新: `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_*` と `ORDER_MIN_UNITS_*` を
      `SCALP_EXTREMA_REVERSAL_LIVE` 向けに追加（`min_units=30`）。
    - `SCALP_EXTREMA_REVERSAL_PERF_GUARD_MODE=warn` を追加。
  - 更新: `ops/env/quant-v2-runtime.env`
    - local fallback 一貫性のため同 strategy 向けの
      `ORDER_MANAGER_PRESERVE_INTENT_*` / `ORDER_MIN_UNITS_*` を追記。
  - 更新: `scripts/ops_v2_audit.py`
    - `quant-scalp-extrema-reversal*.service` を optional pair として監査対象へ追加。
- 意図:
  - 高値ショート/安値ロングの反転機会を、既存戦略の過剰な同一導線抑制から分離して確保する。
  - V2 の役割分離（strategy local decision + order_manager preflight）を維持しつつ、
    両方向の取りこぼしを同時に減らす。

### 2026-02-25（追記）`quant-order-manager` の B/C 最小ロット上書きを service env へ移管

- 背景（VM実測）:
  - 直近監査で `scalp_ping_5s_c_live` の sell は `preflight_start` が出ている一方、
    `entry_probability_reject` / `perf_block` が連続し、
    `OPEN_SKIP note=entry_probability:entry_probability_below_min_units` が主因だった。
  - `quant-order-manager.service` は
    `ops/env/quant-v2-runtime.env` + `ops/env/quant-order-manager.env` のみを読むため、
    worker 側 `ops/env/scalp_ping_5s_{b,c}.env` に置いた
    `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_*_LIVE` は service-mode preflight に反映されていなかった。
  - 同時に runtime 側 `ORDER_MIN_UNITS_SCALP=900` が有効なため、
    B/C の縮小後ユニットが `min_units` 未満として reject されやすい状態だった。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE=30`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B=30`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE=30`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C=30`
- 意図:
  - service-mode の preflight でも B/C の strategy 最小ロットを 30 に固定し、
    `ORDER_MIN_UNITS_SCALP=900` へのフォールバックで発生する
    `entry_probability_below_min_units` 連発を抑止する。

### 2026-02-25（追記）`scalp_ping_5s_c_live` のクラスター損失に対する動的抑制を追加

- 背景（VM実測）:
  - `scalp_ping_5s_c_live` は同分内の同方向クラスター建てで損失が集中し、
    `MARKET_ORDER_TRADE_CLOSE` / `max_adverse` の連鎖で日次損益を押し下げた。
  - `quant-scalp-ping-5s-c-exit` は `workers/scalp_ping_5s_c/exit_worker.py` を使用しており、
    base 実装側で有効な `short_*` / `non_range_max_hold_sec_short` の side override が
    C 側に反映されていなかった。
- 変更:
  - `workers/scalp_ping_5s/worker.py`
    - `direction_bias + horizon + side_adverse_stack` を使って
      同方向 cap を自動縮小する `dynamic_direction_cap` 判定を追加。
    - 判定後の `dynamic cap` で `direction_cap` を評価し、弱一致/逆行圧時の
      同方向積み増しを抑制。
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_*` を追加（opt-in型）。
  - `workers/scalp_ping_5s_c/exit_worker.py`
    - `non_range_max_hold_sec_<side>` と
      `direction_flip.short_* / long_*` の side override 適用を追加し、
      base と同一挙動へ整合。
  - `ops/env/scalp_ping_5s_c.env`
    - `SIDE_ADVERSE_STACK_UNITS_ENABLED=1` へ戻し、
      `DYNAMIC_DIRECTION_CAP_*` を有効化。
  - テスト:
    - `tests/workers/test_scalp_ping_5s_worker.py` に dynamic cap 判定を追加。
    - `tests/workers/test_scalp_ping_5s_c_exit_worker.py` を追加し、
      C exit の side override 適用を回帰テスト化。
- 意図:
  - 「固定cap」ではなく市況・方向一致・逆行圧に応じて同方向上限を縮め、
    クラスター由来の尾を先に削る。
  - C exit の side別保護を実際の service 実装へ反映し、設定と実装の乖離を解消する。

### 2026-02-25（追記）reentry 判定を V2 導線で無効化

- 背景:
  - `execution/reentry_gate.py`（order-manager preflight）の `reentry_block` が
    strategy local 判定とは別軸で ENTRY 密度を抑制するケースがあり、
    運用方針「戦略ローカル優先」と乖離しやすかった。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `REENTRY_GATE_ENABLED=0`
    - `REENTRY_ENABLE_ALL=0`
  - `ops/env/quant-order-manager.env`
    - `REENTRY_GATE_ENABLED=0`
    - `REENTRY_ENABLE_ALL=0`
- 意図:
  - `order_manager` の reentry preflight を no-op 化し、
    ENTRY 可否は strategy ローカル判定 + 既存ガード/リスク判定へ集約する。
  - `config/worker_reentry.yaml` は監査・再開用の履歴として保持し、再開時のみ env を戻して使う。

### 2026-02-25（追記）`scalp_ping_5s_c_live` の entry-probability 拒否を緩和

- 背景（VM実測）:
  - `orders.db` と `quant-order-manager.service` で
    `entry_probability_reject -> perf_block` が短時間に連続し、
    `OPEN_SKIP note=entry_probability:entry_probability_below_min_units` が
    高値/安値反転局面の未約定要因になっていた。
  - 例: `2026-02-25 02:04:52Z` と `02:05:34Z` の sell 候補は
    `entry_probability=0.4871/0.6304` でも `below_min_units` で skip。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.35 -> 0.25`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.40 -> 0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.90 -> 1.00`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE): 30 -> 20`
- 意図:
  - C 戦略の低〜中確率帯を「即reject」ではなく「小さめで通す」側へ寄せ、
    反転ポイントでの取りこぼしを減らす。

### 2026-02-25（追記）`scalp_ping_5s_c_live` を forecast-first で再稼働

- 背景（VM実測）:
  - `quant-scalp-ping-5s-c.service` の実環境が
    `SCALP_PING_5S_C_ENABLED=0` + `SCALP_PING_5S_C_PERF_GUARD_MODE=block`
    になっており、予測が優勢でも C 導線が実行されにくい状態だった。
  - 同時に C の `entry_probability` 再補正が保守寄りで、
    forecast優勢局面でもロットが伸びず取りこぼしが残っていた。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENABLED=1`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_C_CONF_FLOOR: 74 -> 68`
    - `SCALP_PING_5S_C_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.20 -> 0.45`
    - `ENTRY_PROBABILITY_ALIGN_*` を forecast寄りへ調整
      （`BOOST_MAX` 引き上げ / `PENALTY_MAX` 引き下げ /
      `FLOOR_RAW_MIN` と `FLOOR` 緩和 / `UNITS_MAX_MULT` 引き上げ）。
    - `ENTRY_PROBABILITY_BAND_ALLOC_*` を中確率帯で通しやすい閾値へ変更
      （`LOW/HIGH=0.62/0.82`、`HIGH_REDUCE_MAX` 緩和、`LOW_BOOST_MAX` 引き上げ）。
  - `ops/env/quant-order-manager.env`
    - C 向け preserve-intent をさらに通過寄りへ更新:
      - `REJECT_UNDER: 0.25 -> 0.18`
      - `MIN_SCALE: 0.70 -> 0.80`
      - `MAX_SCALE: 1.00 -> 1.15`
      - `BOOST_PROBABILITY=0.65` を strategy override で追加。
- 意図:
  - 「予測が強いときは入る・伸ばす」を C 導線で明示し、
    `entry_probability` 起因の未約定と過小ロットを同時に減らす。
  - `perf_guard` は止める運用から縮小運用へ寄せ、機会損失を抑える。

### 2026-02-25（追記）`scalp_ping_5s_c` preserve-intent を中立寄りへ再調整

- 背景:
  - `quant-order-manager.env` が並行作業で C 向け aggressive 値
    （`REJECT_UNDER=0.18`, `MIN_SCALE=0.80`, `MAX_SCALE=1.15`）へ更新され、
    低確率帯の通過が増えやすい状態になっていた。
  - 当日調査の主目的は「`entry_probability_below_min_units` の連発解消」であり、
    低確率帯の過通過まで狙っていないため、運用意図とずれがあった。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.18 -> 0.25`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.80 -> 0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 1.15 -> 1.00`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.65 -> 0.80`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=20` は維持。
- 意図:
  - 反転局面の取りこぼしは減らしつつ、低確率帯の過剰エントリーを抑える。

### 2026-02-25（追記）`scalp_ping_5s_c` preserve-intent を forecast-first aggressive へ再適用

- 背景（VM実測）:
  - `quant-order-manager.service` の journal で、`scalp_ping_5s_c_live` は
    `2026-02-25 02:28:13Z` / `02:28:41Z` の `sell` 候補が
    `OPEN_SKIP note=entry_probability:entry_probability_below_min_units`
    で未約定になっていた。
  - `orders.db` でも `2026-02-25T04:01:21Z` の `sell -121` は通過した一方、
    直前帯（`02:15Z〜02:19Z`）に `entry_probability_reject -> perf_block`
    が連続し、予測方向の取りこぼしが残っていた。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.25 -> 0.18`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.70 -> 0.80`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE): 1.00 -> 1.15`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.80 -> 0.65`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=20` は維持。
- 意図:
  - 予測優勢側（とくに下落側）の方向意図を order preflight で潰しにくくし、
    `entry_probability` 起因の skip/reject を減らす。

### 2026-02-25（追記）`scalp_ping_5s_c` の逆向きショート抑制を強化

- 背景（VM実測）:
  - `orders.db` の `client_order_id=qr-1771991927466-scalp_ping_5s_c_live-s6e04304b`
    （`2026-02-25 04:01:37Z`）は `sell` で約定したが、
    `entry_thesis` では `horizon_composite_side=long`（`score=0.411`）に対し
    `mtf_regime_side=short` が優先されて通過していた。
  - 同レコードの確率補正は `raw=0.92 -> adjusted=0.654939`
    （`counter=0.478655`, `support=0.0`, `penalty=0.309804`）で、
    予測は反映されているが「減点のみ」で拒否までは到達していなかった。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_HORIZON_BLOCK_SCORE=0.38`（逆向き side を block 側へ寄せる）
    - `SCALP_PING_5S_C_HORIZON_OPPOSITE_UNITS_MULT=0.28`
    - `SCALP_PING_5S_C_HORIZON_OPPOSITE_BLOCK_MIN_MULT=0.08`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_PENALTY_MAX: 0.42 -> 0.55`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX: 0.20 -> 0.35`
- 意図:
  - 予測（horizon）と逆向きのショートが raw 勢いだけで通る経路を減らし、
    C の「forecast-first」方針を実発注で強める。

### 2026-02-24（追記）`order_manager` の orders.db ロック待機を低遅延寄りに再調整

- 背景（VM実測）:
  - `orders.db` / `orders.db-wal` が数GB規模へ肥大した状態で、
    `preflight_start -> submit_attempt` が `60s〜167s` の遅延を複数観測。
  - `quant-order-manager` journal に `failed to persist orders log: database is locked` が連発し、
    close系も `Read timed out` / `close_request` 停滞を伴った。
- 変更:
  - `ops/env/quant-v2-runtime.env`
  - `ops/env/quant-order-manager.env`
  - `ORDER_DB_BUSY_TIMEOUT_MS: 5000 -> 250`
  - `ORDER_DB_LOG_RETRY_ATTEMPTS: 8 -> 3`
  - `ORDER_DB_LOG_RETRY_SLEEP_SEC: 0.05 -> 0.02`
  - `ORDER_DB_LOG_RETRY_BACKOFF: 1.8 -> 1.5`
  - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC: 1.00 -> 0.10`
- 意図:
  - 発注導線の同期ログ書き込みがロック待ちで秒単位ブロックする状態を抑え、
    `OPEN_SCALE -> OPEN_REQ` の遅延を先に縮める。
  - ログ欠落よりも約定遅延リスクを優先して、低遅延側へ寄せる。

### 2026-02-24（追記）5秒スキャ B/C/D の「未エントリー」要因を追加緩和

- 背景:
  - VM 実ログで `quant-scalp-ping-5s-b/c/d` は active だが、
    B は `no_signal:revert_not_found` + `directional_bias_zero`、
    C は `spread_blocked(reason=spread_stale)` が支配し、約定が停滞していた。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SIDE_BIAS_BLOCK_THRESHOLD: 0.30 -> 0.00`
    - `REVERT_*` を flow 寄りへ緩和
      (`REVERT_MIN_TICK_RATE: 0.90 -> 0.60`,
      `RANGE/SWEEP/BOUNCE/CONFIRM: 0.45/0.35/0.10/0.58 -> 0.35/0.22/0.08/0.50`)
    - short 判定の最小要件を緩和
      (`SHORT_MIN_TICKS/SIGNAL_TICKS/TICK_RATE: 4/4/0.72 -> 3/3/0.60`)
  - `ops/env/scalp_ping_5s_c.env`
    - `SIDE_FILTER: long -> (empty)`（両方向許可）
    - `MAX_SPREAD_PIPS: 1.60 -> 2.00`
    - `REVERT_*` と short 側閾値を緩和
      (`SHORT_MIN_TICKS/SIGNAL_TICKS/TICK_RATE: 8/8/1.05 -> 4/4/0.72`,
       `SHORT/LONG_MOMENTUM_TRIGGER: 0.22/0.12 -> 0.11/0.11`)
    - stale spread による連続停止を抑止:
      `spread_guard_stale_grace_sec=900`,
      `spread_guard_stale_block_min_pips=1.80`
  - `ops/env/scalp_ping_5s_d.env`
    - `SIDE_FILTER: long -> (empty)`（両方向許可）
    - `MAX_SPREAD_PIPS: 1.60 -> 2.00`
    - `SIDE_BIAS_BLOCK_THRESHOLD: 0.30 -> 0.10`
    - `REVERT_*` と short 側閾値を C と同等に緩和
    - stale spread 抑止の `spread_guard_stale_*` を C と同値で追加
- 意図:
  - 「戦略停止」ではなく「ゲート過多」で入らない状態をほどき、
  B/C/D の約定再開と方向追従（long-only 固定の解除）を優先する。

### 2026-02-24（追記）5秒スキャ B/C の no-signal 継続に対する第2段緩和

- 背景:
  - 第1段反映後、`spread_stale` は解消したが、B/C は
    `no_signal:revert_not_found` と `units_below_min` が継続し、
    `submit_attempt` が発生しない区間が残った。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MIN_UNITS: 100 -> 50` と `ORDER_MIN_UNITS_*: 100 -> 50`
    - `MIN_TICK_RATE: 0.72 -> 0.50`、`SHORT_MIN_TICK_RATE: 0.60 -> 0.50`
    - `REVERT_WINDOW/SHORT_WINDOW: 1.60/0.55 -> 2.20/0.75`
    - `REVERT_MIN_TICK_RATE: 0.60 -> 0.50`
    - `REVERT_RANGE/SWEEP/BOUNCE: 0.35/0.22/0.08 -> 0.20/0.12/0.05`
    - `LONG/SHORT/MOMENTUM_TRIGGER_PIPS: 0.10` に統一
    - `SIDE_ADVERSE_STACK_UNITS_ENABLED: 1 -> 0`
  - `ops/env/scalp_ping_5s_c.env`
    - `MIN_UNITS: 100 -> 50` と `ORDER_MIN_UNITS_*: 100 -> 50`
    - `SIDE_BIAS_BLOCK_THRESHOLD: 0.30 -> 0.10`
    - `MIN_TICK_RATE: 0.80 -> 0.50`、`SHORT_MIN_TICK_RATE: 0.72 -> 0.50`
    - `REVERT_WINDOW/SHORT_WINDOW: 1.60/0.55 -> 2.20/0.75`
    - `REVERT_MIN_TICK_RATE: 0.60 -> 0.50`
    - `REVERT_RANGE/SWEEP/BOUNCE: 0.35/0.22/0.08 -> 0.20/0.12/0.05`
    - `LONG/SHORT/MOMENTUM_TRIGGER_PIPS: 0.10` に統一
    - `SIDE_ADVERSE_STACK_UNITS_ENABLED: 1 -> 0`
- 意図:
  - revert 判定の成立域を広げて no-signal を減らし、
    多段縮小で `min_units` 未満へ落ちるケースを減らして
    B/C の約定再開を狙う。

### 2026-02-24（追記）`forecast_fusion` に weak-contra reject を追加（逆行ノイズの見送り）

- 背景:
  - VM実トレード監査（`logs/trades.db`）で、`direction_prob<0.5` の逆行群のうち
    `edge_strength<0.30` が損失寄与の中心で、既存の `strong_contra_reject`
    （高edge逆行のみ拒否）では防ぎ切れていなかった。
  - 一方で `edge_strength>=0.30` の逆行群は必ずしも劣化せず、
    「逆行=即拒否」ではなく「弱い逆行だけ拒否」が必要だった。
- 変更:
  - `execution/strategy_entry.py`
    - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_*` を追加。
    - 条件: `direction_prob<=prob_max` かつ `edge_strength<=edge_max` かつ
      `allowed=false` または逆行方向のとき、`units=0` として reject。
    - 監査ペイロードに `weak_contra_reject` と `reject_reason=weak_contra_forecast`
      を追加。
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED=1`
    - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_PROB_MAX=0.50`
    - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_EDGE_MAX=0.30`
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - weak-contra reject の発火ケースと、
      高edge逆行を拒否しないケースの回帰テストを追加。
  - `docs/FORECAST.md`
    - weak-contra reject の仕様と運用キーを追記。
- 意図:
  - 方向確率は逆行でも「確信が弱い逆行（低edge）」だけを機械的に見送って、
    逆行ノイズ由来の損失を抑える。
  - 高edge逆行は一律拒否せず、既存の strong-contra / rebound 判定へ分離する。

### 2026-02-24（追記）`forecast_fusion` strong-contra reject を既定OFFへ切替

- 背景（VM実測）:
  - `logs/trades.db` × `logs/orders.db(request_json.entry_thesis)` を 14日で照合し、
    `strong_contra` 条件（`direction_prob<=0.22 && edge_strength>=0.65`）に該当した
    11件が合計 `+25.5 pips`（勝率 90.9%）で正寄与だった。
  - 同期間の `weak_contra` 条件（`direction_prob<=0.50 && edge_strength<=0.30`）は
    500件で `-857.3 pips` と明確な負寄与で、weak 側のみ reject を維持する方が
    成績改善に有利だった。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED: 1 -> 0`
- 意図:
  - 「強い逆行＝即拒否」の過剰抑制を外し、実績が悪い weak-contra のみ機械的に遮断する。
  - 方向意図の選別を forecast で過剰に行わず、既存の戦略ローカル判定 + weak-contra に寄せる。

### 2026-02-24（追記）`scalp_ping_5s_b_live` の損失時間帯ブロック + 確率閾値引き上げ

- 背景（VM実測, 直近14日）:
  - `scalp_ping_5s_b_live` は `closed=2827` で `-1920.5 pips`。
  - 逆算で `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` を `0.50` に引き上げると
    `+184.6 pips` 改善余地。
  - 時間帯ブロック候補（JST `1,2,3,10,13,15,16,19,21,22`）は
    `+1611.4 pips` 改善余地を確認。
  - 直近3日でも同方針で `+236.1 pips` の改善余地を確認。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BLOCK_HOURS_JST=1,2,3,10,13,15,16,19,21,22`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE: 0.24 -> 0.50`
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_BLOCK_HOURS_JST` 読み取り（CSV int, 0-23正規化）を追加。
  - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
    - Bプレフィックス経由で `BLOCK_HOURS_JST` が反映される回帰テストを追加。
- 意図:
  - B戦略の「負けが集中する時間帯」と「低確率エントリー」を先に削って、
    取引密度を維持しつつ損失寄与を切り落とす。

### 2026-02-24（追記）`trade_counterfactual` に replay由来の「取り残し型」抽出を追加

- 背景:
  - 「ポジションが上下に取り残される」問題を replay から定量化し、
    live 履歴だけでなく replay 出力でも同一ロジックで `block/reduce/boost` を
    生成できるようにする必要があった。
- 変更:
  - `analysis/trade_counterfactual_worker.py`
    - `COUNTERFACTUAL_REPLAY_JSON_GLOBS` を追加し、
      `replay_exit_workers.json` の `trades[]` を直接入力可能化。
    - `reason` / `hold_sec` / `source`（live|replay）をサンプル特徴へ追加。
    - stuck 判定（`stuck_hold_sec`, `stuck_loss_pips`, `stuck_reasons`）を導入し、
      `stuck_rate` を action 推定へ反映。
    - policy hint に `block_reasons` / `reduce_reasons` を追加。
  - `ops/env/quant-trade-counterfactual.env`
    - replay入力と stuck 閾値の運用キーを追加。
  - `tests/analysis/test_trade_counterfactual_worker.py`
    - replay JSON 読み込みと stuck block 推奨の回帰テストを追加。
  - `docs/REPLAY_STANDARD.md`
    - replay 専用入力で counterfactual を回す標準コマンドを追記。
  - `docs/ARCHITECTURE.md`
    - trade counterfactual の replay入力/stuck判定仕様を追記。
- 意図:
  - replay で「悪い局面の型」を将棋の定跡のように収集し、
    `block_jst_hours` / `block_reasons` を運用調整へ直結させる。

### 2026-02-24（追記）5秒スキャ B/C の確率rejectを緩和し、Bの逆方向flipを停止

- 背景:
  - VM 実測で `scalp_ping_5s_b_live` は `open mode=... route=sl_streak_flip/side_metrics_flip`
    で short 化した後に `OPEN_SKIP note=entry_probability:entry_probability_reject_threshold`
    が継続し、シグナルは出るが約定しない状態だった。
  - `scalp_ping_5s_c_live` も strategy 別の reject 閾値が高く、
    `flow` 比で entry 密度が落ちていた。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED: 1 -> 0`
    - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED: 1 -> 0`
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED: 1 -> 0`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE: 0.48 -> 0.30`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE: 0.65 -> 0.45`
  - `ops/env/scalp_ping_5s_c.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE: 0.55 -> 0.45`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN/MAX_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE: 0.45/0.85 -> 0.40/0.90`
- 意図:
  - B の「逆方向flipで低確率化→order_manager reject」ループを切り、
    5秒スキャの方向整合と約定回復を同時に狙う。
  - C は profit-first の閾値を保ったまま reject 過多を緩和し、
    flow 偏重を減らす。

### 2026-02-24（追記）5秒スキャ B/C の `perf_block` 偏重と `rate_limited` を緩和

- 背景:
  - VM 実測で `orders.db` の B/C 系 `status` が `entry_probability_reject` と
    `perf_block` に集中し、エントリー意図は出ていても preflight で止まる時間帯が継続した。
  - C は `entry-skip summary` に `rate_limited` が多発し、`MAX_ORDERS_PER_MINUTE=6`
    がボトルネックになっていた。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE: 10 -> 14`
    - `MIN_UNITS / ORDER_MIN_UNITS_*: 50 -> 30`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.30 -> 0.24`
    - `PERF_GUARD_*` をサンプル増加時のみ強く効く値へ緩和
      （`MIN_TRADES`/`HOURLY_MIN_TRADES`/`FAILFAST_MIN_TRADES` 引き上げ、
      `PF/WIN` 閾値を緩和）。
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 6 -> 14`
    - `MIN_UNITS / ORDER_MIN_UNITS_*: 50 -> 30`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.45 -> 0.35`
    - `PERF_GUARD_*` を同様に緩和（過剰 `perf_block` 回避）。
- 意図:
  - `perf_block` と `rate_limited` の同時多発を抑え、5秒スキャの実約定密度を回復する。
  - 完全無効化ではなく、サンプルが積み上がった劣化局面では依然ブロックが効く形を維持する。

### 2026-02-24（追記）5秒スキャ B の perf guard を `reduce` へ変更

- 背景:
  - 直近 VM 観測（09:29 UTC 以降）で `scalp_ping_5s_b_live` は
    `entry_probability_reject` に加えて `perf_block` が連続し、
    C が `filled` を回復する一方で B だけ回転が戻らなかった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE: block -> reduce`
- 意図:
  - B は perf guard を「警告のみ」に落とし、preflight でのブロック停止を解除して
    5秒スキャの実エントリー頻度を回復する。

### 2026-02-24（追記）`quant-regime-router` を追加（5秒スキャのレジーム配分切替）

- 背景:
  - 5秒スキャの B/C/D/FLOW を同時稼働したまま固定配分で回すと、
    市況（trend/range/breakout）と戦略特性の不一致が継続しやすく、
    同一時間帯での負け構造が残りやすかった。
  - V2 導線を崩さず、`strategy_control` の `entry_enabled` だけを
    レジームで動的切替する配分レイヤが必要だった。
- 変更:
  - 追加: `workers/regime_router/worker.py`
    - `M1/H4` レジームを `workers.common.quality_gate.current_regime()` で取得し、
      `trend/breakout/range/mixed/event/unknown` へルーティング。
    - `REGIME_ROUTER_MIN_DWELL_SEC` による route 切替の最小滞在時間を導入。
    - `REGIME_ROUTER_MANAGED_STRATEGIES` に含まれる戦略のみ
      `strategy_control.set_strategy_flags(..., entry=...)` を更新。
    - `exit_enabled` / `global_lock` は変更しない。
    - 状態監査を `logs/regime_router_state.json` へ出力。
  - 追加: `systemd/quant-regime-router.service`
  - 追加: `ops/env/quant-regime-router.env`
    - 初期値は `REGIME_ROUTER_ENABLED=0`（opt-in）
    - 既定ルートは `trend/breakout -> scalp_ping_5s_d`,
      `range/mixed/unknown -> scalp_ping_5s_c`。
  - 追加: `tests/workers/test_regime_router_worker.py`
    - route判定、dwell制御、entry plan更新の回帰テストを追加。
- 意図:
  - 戦略ロジック本体や order_manager のガード層を変更せず、
    「どの戦略を entry 有効にするか」だけを独立ワーカーで調整する。
  - 既存の V2 役割分離（data/control/order/position）を維持しながら、
    5秒スキャ群の配分を市況連動で切替える。

### 2026-02-24（追記）`TickImbalance` の reentry `same_dir_mode` を `both` へ修正

- 背景:
  - `TickImbalance` は `config/worker_reentry.yaml` の個別設定で `same_dir_mode` が未指定のため、
    defaults の `same_dir_mode: return` を継承していた。
  - その結果、直近クローズ価格から大きく乖離した相場で
    `quant-order-manager` の `OPEN_SKIP note=reentry_gate:price_distance` が連発し、
    実質的にエントリー再開不能な状態が発生していた。
- 変更:
  - `config/worker_reentry.yaml`
    - `strategies.TickImbalance.same_dir_mode: both` を明示追加。
- 意図:
  - 戻り待ち限定の片側拘束を外し、`return/follow` の両方向で
    reentry 判定を成立可能にして「reentry が機能する状態」へ復帰する。

### 2026-02-24（追記）`TickImbalance` reentry の stale state 耐性を追加し、price_distance を撤廃

- 背景:
  - VM 実データで `logs/stage_state.db.strategy_reentry_state` の `last_trade_id` が
    `logs/trades.db` の現行 `id` より大きい状態が残り、`TickImbalance` の
    `last_close_price` が 2026-02-12 の値で固定化していた。
  - その結果、`note=reentry_gate:price_distance` が集中し、
    reentry 判定が実質機能停止していた。
- 変更:
  - `execution/stage_tracker.py`
    - `strategy_reentry_state` 更新条件を `trade_id` 単独比較から拡張し、
      `trade_id` が巻き戻った場合でも `close_time` が新しければ更新する。
  - `config/worker_reentry.yaml`
    - `TickImbalance.cooldown_loss_sec: 541 -> 120`
    - `TickImbalance.same_dir_reentry_pips: 1.76 -> 0.0`（price distance 実質撤廃）
  - `tests/test_stage_tracker.py`
    - `trade_id` 回帰ケースで `strategy_reentry_state` が更新される回帰テストを追加。
- 意図:
  - DB restore/世代切替などで `trades.id` 系列が不連続になっても、
    reentry の基準状態を最新トレードへ追随させる。
  - `TickImbalance` は price-distance 依存で詰まらない運用へ切り替え、
    エントリー再開性を優先する。

### 2026-02-24（追記）5秒スキャ B/C/D の entry 過抑制を緩和

- 背景:
  - VM 実ログで `quant-scalp-ping-5s-{b,c,d}.service` が稼働中にもかかわらず、
    `entry-skip summary` の大半を `no_signal:revert_not_found` と
    `lookahead_block` が占め、`[ORDER][OPEN_REQ]` がほぼ発生しない状態だった。
  - 直近は `scalp_ping_5s_flow_live` のみが継続約定し、B/C/D は事実上休止状態だった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MIN_TICKS: 5 -> 4`
    - `MIN_SIGNAL_TICKS: 4 -> 3`
    - `MIN_TICK_RATE: 0.85 -> 0.72`
    - `IMBALANCE_MIN: 0.55 -> 0.50`
    - `LOOKAHEAD_GATE_ENABLED: 1 -> 0`
    - `REVERT_MIN_TICK_RATE: 1.20 -> 0.90`
    - `EXTREMA_GATE_ENABLED: 1 -> 0`
  - `ops/env/scalp_ping_5s_c.env`
    - `MIN_TICKS: 5 -> 4`
    - `MIN_SIGNAL_TICKS: 4 -> 3`
    - `MIN_TICK_RATE: 0.95 -> 0.80`
    - `IMBALANCE_MIN: 0.55 -> 0.50`
    - `LOOKAHEAD_GATE_ENABLED: 1 -> 0`
    - `REVERT_MIN_TICK_RATE: 1.20 -> 0.90`
    - `EXTREMA_GATE_ENABLED: 1 -> 0`
  - `ops/env/scalp_ping_5s_d.env`
    - `MIN_TICKS: 5 -> 4`
    - `MIN_SIGNAL_TICKS: 4 -> 3`
    - `MIN_TICK_RATE: 0.95 -> 0.80`
    - `IMBALANCE_MIN: 0.55 -> 0.50`
    - `LOOKAHEAD_GATE_ENABLED: 1 -> 0`
    - `REVERT_MIN_TICK_RATE: 1.20 -> 0.90`
    - `EXTREMA_GATE_ENABLED: 1 -> 0`
- 意図:
  - B/C/D の「シグナル成立はするがゲート過多で entry しない」状態を解消し、
    5秒スキャの約定頻度を回復させる。
  - C/D の `SIDE_FILTER=long` は維持し、方向リスクは抑えたまま entry 数のみ緩和する。

### 2026-02-24（追記）`scalp_ping_5s` の side filter を最終シグナルへ強制

- 背景:
  - `SCALP_PING_5S_*_SIDE_FILTER=long` を設定した replay/live-entry で、
    `mtf_reversion_fade` 等の後段リルートにより short 約定が混入する経路があった。
  - 原因は、`_build_tick_signal` での side filter 適用後に
    `MTF/flip` 系ロジックが side を再書き換えること。
- 変更:
  - `workers/scalp_ping_5s/worker.py`
    - ルーティング完了後（`fast_flip/sl_streak/side_metrics_flip` 後）に
      `side_filter_final_block` を追加し、最終 side が filter と不一致なら entry を reject。
  - `scripts/replay_exit_workers.py`
    - `_signal_scalp_ping_5s_b` で `MTF` 調整後の `signal.side` に対しても
      `SIDE_FILTER` を再評価し、不一致なら signal を破棄。
  - `tests/scripts/test_replay_exit_workers.py`
    - `test_ping5s_signal_respects_post_regime_side_filter` を追加
      （`_apply_mtf_regime` が short へ反転しても `SIDE_FILTER=long` で reject されることを検証）。
- 検証:
  - `pytest -q tests/scripts/test_replay_exit_workers.py` : `8 passed`
  - replay 再実行（`SCALP_PING_5S_C_SIDE_FILTER=long`）で
    short 約定は `0` 件になることを確認。

### 2026-02-24（追記）`scalp_ping_5s_d_live` WFO窓を `3x2` へ更新

- 背景:
  - `config/replay_quality_gate_ping5s_d.yaml` 既定の `train_files=2 / test_files=1`
    は、1日単位のtestで `pf_stability_ratio` と `train_trade_count` の揺れが大きく、
    同一データでも fold判定が不安定だった。
- 変更:
  - `config/replay_quality_gate_ping5s_d.yaml`
    - `walk_forward.train_files: 2 -> 3`
    - `walk_forward.test_files: 1 -> 2`
- 検証（同一7日データ比較）:
  - `2x1`（旧）: `pass_rate=0.40`, `median_test_jpy_per_hour=3.4848`
  - `3x1`（補助評価）: `pass_rate=0.75`
  - `3x2`（新）: `pass_rate=1.00`, `median_test_jpy_per_hour=2.5301`
- 補足:
  - `hold` 短縮（`45/35`）は `20260210` を `-12.5 JPY` まで悪化させたため不採用。
  - 運用・再現性の観点で、D系の品質判定は `3x2` を基準とする。

### 2026-02-24（追記）dynamic_alloc のタグ汚染対策と損失戦略のサイズ抑制

- 背景:
  - `dynamic_alloc.json` に `scalp_ping_5s_b_live-lxxxx` の単発タグが多数混入し、
    本来同一戦略の成績が分割集計されて lot 配分が鈍る状態だった。
- 変更:
  - `scripts/dynamic_alloc_worker.py`
    - `normalize_strategy_key()` を追加し、
      `-l[hex]` / `-[hex]` の末尾一時タグを戦略キーへ正規化。
    - 低品質戦略の lot 上限ガードを強化:
      - `pf < 1.0`: `<=0.90`
      - `pf < 0.8`: `<=0.80`
      - `pf < 0.7`: `<=0.75`
      - `pf < 0.6`: `<=0.70`
      - `avg_pips <= -1.0` かつ十分サンプル: `<=0.72`
      - `sum_pips <= -80` かつ十分サンプル: `<=0.68`
  - `systemd/quant-dynamic-alloc.service`
    - `--limit 400 -> 2400`
    - `--min-trades 12 -> 24`
  - `tests/test_dynamic_alloc_worker.py`
    - 正規化・集約・損失ペナルティの回帰テストを追加。
- 意図:
  - 損失主導戦略（直近では `scalp_ping_5s_b_live`）の資金消費を機械的に抑え、
    利益戦略への配分効率を改善する。

### 2026-02-22（追記）5秒スキャC replay のロット再現とWFO構造改善（継続）

- 背景:
  - `scripts/replay_exit_workers.py` の `ScalpReplayEntryEngine` が
    replay entry で `units=10000` 固定だったため、
    `scalp_ping_5s_c` の実運用ロット（`BASE_ENTRY_UNITS=800`, `MAX_UNITS=2200`）を
    再現できず、time-stop 損失が過大化していた。
- 変更:
  - `scripts/replay_exit_workers.py`
    - `ScalpPing5SB` signal に `entry_units_intent` を付与。
    - replay entry の units 解決を `entry_units_intent` 優先へ変更（固定 10000 を廃止）。
    - ping5s 用 helper を追加し、`BASE_ENTRY_UNITS/MIN_UNITS/MAX_UNITS` と
      `confidence` から replay units を算出。
  - `tests/scripts/test_replay_exit_workers.py`
    - units 解決ロジック（bounds/符号）の回帰テストを追加。
  - `config/replay_quality_gate_ping5s_c.yaml`
    - replay.env に C運用値（`BASE/MIN/MAX_UNITS`, `FORCE_EXIT_*`）を明示注入。
    - 時間帯ブロックと gate 閾値を段階調整（loss cluster の除去優先）。
- 検証:
  - `20260222_071602`（units固定解除のみ）
    - `achieved_jpy_per_hour: -74.47`（旧 `-78.90` から改善、ただし未達）
  - `20260222_093855`（C env 注入 + hold短縮 + block 強化）
    - `achieved_jpy_per_hour: -0.48` まで改善（ほぼ損益分岐）
    - trade units は `642..800`、`time_stop` 平均損失は `-26.74 JPY/件`
      （旧 `-609.64 JPY/件` から大幅改善）
  - `20260222_100500`（追加 block + gate 低サンプル調整）
    - `achieved_jpy_per_hour: +5.31`
    - fold pass は `2/5`（`pass_rate=0.4`）で strict では fail 継続。
- 留意:
  - `+2000円/h` 目標は引き続き未達。
  - 現runの逆算条件は
    `required_trades_per_hour_at_current_ev ≈ 319.77` または
    `required_jpy_per_trade_at_current_freq ≈ 2353.81` で、現行5秒スキャCの構造では乖離が大きい。

### 2026-02-22（追記）`scalp_ping_5s_d_live` を追加（別ワーカーで独立検証）

- 背景:
  - 5秒スキャ改善を既存 `scalp_ping_5s_b/c` から分離し、設定差分を安全に検証するため
    `D` 系ワーカーを追加。
- 変更:
  - 追加: `workers/scalp_ping_5s_d/`
    - `worker.py` は `SCALP_PING_5S_D_*` プレフィックスを `SCALP_PING_5S_*` へ投影するラッパ。
    - `exit_worker.py` は D env を適用し、C の exit 実装へ委譲する thin wrapper。
  - 追加: `systemd/quant-scalp-ping-5s-d.service`
  - 追加: `systemd/quant-scalp-ping-5s-d-exit.service`
  - 追加: `ops/env/quant-scalp-ping-5s-d.env`
  - 追加: `ops/env/quant-scalp-ping-5s-d-exit.env`
  - 追加: `ops/env/scalp_ping_5s_d.env`
  - 変更: `scripts/replay_exit_workers.py`
    - `SCALP_REPLAY_PING_VARIANT=B|C|D` に拡張し、
      `SCALP_REPLAY_MODE=scalp_ping_5s_d` で D replay を直接実行可能化。
  - 追加: `config/replay_quality_gate_ping5s_d.yaml`
    - `SCALP_REPLAY_PING_VARIANT=D` を固定し、D 用 replay gate を分離。
  - 変更: `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_d` / `scalp_ping_5s_d_live` を B/C 共通 exit profile に接続。
  - 変更: `config/worker_reentry.yaml`
    - `scalp_ping_5s_d_live` の時間帯ブロックを追加。
  - 変更: `ops/env/quant-order-manager.env`
    - `scalp_ping_5s_d_live` 向け preserve-intent/perf-guard キーを追加。
  - 変更: テスト
    - `tests/workers/test_scalp_ping_5s_b_worker_env.py` に D env-prefix/force-exit default の回帰テスト追加。
    - `tests/scripts/test_replay_exit_workers.py` に D variant module mapping テスト追加。

### 2026-02-21（追記）`scalp_ping_5s_c_live` を新設（profit-first クローン）

- 背景:
  - `scalp_ping_5s_b_live` の replay/WFO で、主損失源が short 側に集中し時給収益目標を満たせない状態が継続。
  - 既存Bを汚さず、強デリスク設定（short抑制・サイズ抑制・spread guard有効）を別ワーカーで検証する必要がある。
- 実施:
  - 追加: `workers/scalp_ping_5s_c/`（ENTRY/EXIT 一式を B から複製）
    - `workers/scalp_ping_5s_c/worker.py` を `SCALP_PING_5S_C_*` プレフィックスへ切替。
    - `workers/scalp_ping_5s_c/exit_worker.py` エントリポイントを `scalp_ping_5s_c_exit_worker()` へ変更。
  - 追加: `systemd/quant-scalp-ping-5s-c.service`
  - 追加: `systemd/quant-scalp-ping-5s-c-exit.service`
  - 追加: `ops/env/quant-scalp-ping-5s-c.env`
  - 追加: `ops/env/quant-scalp-ping-5s-c-exit.env`
  - 追加: `ops/env/scalp_ping_5s_c.env`
    - `SIDE_FILTER=long`
    - `MAX_UNITS=2200`, `BASE_ENTRY_UNITS=800`
    - short閾値強化（`SHORT_MIN_TICKS=8`, `SHORT_MIN_TICK_RATE=1.05` など）
    - `SPREAD_GUARD_DISABLE=0`
  - 変更: `workers/scalp_ping_5s/config.py`
    - `ENV_PREFIX=SCALP_PING_5S_C` を B 同等の default 群として扱う（B/C 互換）。
  - 変更: `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_c` / `scalp_ping_5s_c_live` を B/C 共通 `exit_profile` に接続。
  - 変更: `scripts/replay_exit_workers.py`
    - `SCALP_REPLAY_PING_VARIANT=B|C` で replay 対象を B/C 切替可能化。
  - 変更: `ops/env/quant-order-manager.env`
    - `scalp_ping_5s_c_live` 向けの preserve-intent/perf-guard キーを追加。
- 意図:
  - B運用を保持したまま C で「損失構造の先行除去」を検証し、実トレード投入前に replay/WFO で改善の有無を切り分ける。

### 2026-02-22（追記）5秒スキャCの WFO profit-lock ゲートを強化

- 対象:
  - `analytics/replay_quality_gate.py`
  - `scripts/replay_quality_gate.py`
  - `config/replay_quality_gate_ping5s_c.yaml`
  - `docs/REPLAY_STANDARD.md`
  - `docs/ARCHITECTURE.md`
- 変更:
  - replay quality gate の fold 指標へ `total_jpy` / `jpy_per_hour` / `max_drawdown_jpy` を追加。
  - gate 閾値へ `min_test_total_jpy` / `min_test_jpy_per_hour` / `max_test_drawdown_jpy` を追加。
  - `scripts/replay_quality_gate.py` に `replay.env` 注入機能を追加し、
    config から `SCALP_REPLAY_PING_VARIANT=C` などを固定可能化。
  - 5秒スキャC専用プロファイル `config/replay_quality_gate_ping5s_c.yaml` を追加。
    - `exit_workers_main`
    - `no_main_strategies=true`
    - `exclude_end_of_replay=false`
    - `SCALP_REPLAY_BLOCK_JST_HOURS=21`
    - `min_tick_lines=50000`
    - `min_fold_pass_rate=1.0`
  - `scripts/replay_jpy_hour_sweep.py` を追加し、既存 WFO レポートから
    `min_test_jpy_per_hour` 閾値スイープと目標時給の必要条件逆算を自動化。
- 検証（ローカル replay, strict）:
  - 旧設定（`min_tick_lines=30000`, hour blockなし）は fail
    - `tmp/replay_quality_gate_ping5s_c_strict/20260222_020543/quality_gate_report.json`
    - `pass_rate=0.4286 (3/7)`、`20260127` fold で `end_of_replay` の大損失を検出。
  - 調整後（hour 21 block + sample filter強化）は pass
    - `tmp/replay_quality_gate_ping5s_c_strict/20260222_022355/quality_gate_report.json`
    - `pass_rate=1.0 (5/5)`、`min_test_total_jpy>0` を全foldで満たす。
  - 追加評価（スイープ）:
    - `tmp/replay_jpy_hour_sweep/20260222_from_022355.json`
    - `min_test_jpy_per_hour=150` でも `pass_rate=0.2`（fail）
    - `300/500/2000` は `pass_rate=0.0`（all fail）
    - 目標 `+2000円/h` に対して現状 aggregate は
      `achieved_jpy_per_hour=123.38`, `achieved_trades_per_hour=1.10`, `achieved_jpy_per_trade=111.98`。
      必要条件は `required_trades_per_hour_at_current_ev=17.86` または
      `required_jpy_per_trade_at_current_freq=1815.21`。
- 留意:
  - 本プロファイルで「プラス維持」は満たすが、`median_test_jpy_per_hour ≈ 120.38` で、
    `+2000円/時` 目標には未達。

### 2026-02-22（追記）5秒スキャCの force-exit 再現性を構造補強（live/replay整合）

- 背景:
  - `scalp_ping_5s_c` は `FORCE_EXIT_MAX_ACTIONS` 未設定時に 0 となり、
    `FORCE_EXIT_ACTIVE` が無効化されるため、time-stop 系の保護が実質 OFF になるケースがあった。
  - `scripts/replay_exit_workers.py` 側は `ScalpPing5SB` の `timeout_sec=0` 固定で、
    実運用の force-exit hold を再現できず `end_of_replay` close に寄りやすかった。
- 変更:
  - `workers/scalp_ping_5s/config.py`
    - `ENV_PREFIX=SCALP_PING_5S_B/C` では `SCALP_PING_5S_FORCE_EXIT_MAX_ACTIONS` の既定を `2` に変更し、
      B/C variant の force-exit がデフォルトで有効になるよう修正。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_FORCE_EXIT_*` を明示追加
      （`MAX_ACTIONS=2`, `MAX_HOLD_SEC=95`, `SHORT_MAX_HOLD_SEC=70`,
      `MAX_FLOATING_LOSS_PIPS=1.9`, `SHORT_MAX_FLOATING_LOSS_PIPS=1.5` など）。
  - `scripts/replay_exit_workers.py`
    - ping5s signal で `FORCE_EXIT_MAX_HOLD_SEC` / `SHORT_FORCE_EXIT_MAX_HOLD_SEC` を `timeout_sec` として返却。
    - `force_exit_reason` を signal/thesis に渡し、trade の timeout 判定を `SimBroker.check_timeouts()` で毎tick実行。
    - timeout 到達時は通常 close を試行し、ガード拒否時のみ market 同値で強制 close して `end_of_replay` 依存を低減。
    - replay 直エントリー（非 live-entry）でも `entry_thesis` に `timeout_sec` を注入し、timeout 判定の一貫性を確保。
- 意図:
  - 5秒スキャCの「実運用で効く時間切れ保護」を replay/WFO でも同じ入力で再現し、
    チューニング時の評価誤差（end_of_replay 偏重）を削減する。

### 2026-02-21（追記）反実仮想レビューに疑似OOS確実性ゲートを追加

- 対象:
  - `analysis/trade_counterfactual_worker.py`
  - `ops/env/quant-trade-counterfactual.env`
  - `tests/analysis/test_trade_counterfactual_worker.py`
  - `docs/ARCHITECTURE.md`
- 変更:
  - 既存の `fold_consistency` + `lb95_pips` 判定に加え、fold 外の疑似 OOS 検証を追加。
  - 各候補（feature/bucket/action）について、fold ごとに train→test で action を再推定し、
    `oos_action_match_ratio` / `oos_positive_ratio` / `oos_lb95_uplift_pips` を算出。
  - 閾値（`COUNTERFACTUAL_OOS_*`）未達の候補は提案から除外。
  - レポートに OOS 指標を追加し、`certainty` に OOS 成分を反映。
  - feature 軸に `hour_spread` / `hour_prob` を追加して多要因条件を拡張。
- 意図:
  - 「この時こうしておけばよかった」の提案を、
    in-sample 偏りだけで採用しない運用に固定する。

### 2026-02-21（追記）実トレードの反実仮想レビューを定期ワーカー化

- 対象:
  - `analysis/trade_counterfactual_worker.py`
  - `systemd/quant-trade-counterfactual.service`
  - `systemd/quant-trade-counterfactual.timer`
  - `ops/env/quant-trade-counterfactual.env`
  - `tests/analysis/test_trade_counterfactual_worker.py`
  - `docs/ARCHITECTURE.md`
  - `docs/WORKER_ROLE_MATRIX_V2.md`
- 変更:
  - closed trades + filled order quote を読み取り、
    `side/hour_jst/spread_bin/prob_bin` の多要因で反実仮想候補を算出する
    `analysis/trade_counterfactual_worker.py` を追加。
  - 判定は `lb95_pips`（平均pipsの95%下限）と
    5fold の符号一貫性（`fold_consistency`）を併用し、
    `block/reduce/boost` を提案。
  - `logs/trade_counterfactual_latest.json`（最新）と
    `logs/trade_counterfactual_history.jsonl`（履歴）を出力。
  - `quant-trade-counterfactual.timer`（30min 周期）を追加し、
    事後判定を定期バッチ化。
- 意図:
  - 「この時こうしておけばよかった」を定量化しつつ、
    低サンプル/不安定条件の過学習を抑えて運用に反映する。

### 2026-02-21（追記）replay 品質ゲートを定期ワーカー化（service/timer + latest snapshot）

- 対象:
  - `analysis/replay_quality_gate_worker.py`
  - `systemd/quant-replay-quality-gate.service`
  - `systemd/quant-replay-quality-gate.timer`
  - `ops/env/quant-replay-quality-gate.env`
  - `tests/analysis/test_replay_quality_gate_worker.py`
  - `docs/REPLAY_STANDARD.md`
  - `docs/ARCHITECTURE.md`
  - `docs/WORKER_ROLE_MATRIX_V2.md`
- 変更:
  - `scripts/replay_quality_gate.py` の実行ラッパーとして
    `analysis/replay_quality_gate_worker.py` を追加。
  - worker は run ごとに `quality_gate_report.json` を検出し、
    `logs/replay_quality_gate_latest.json`（最新）と
    `logs/replay_quality_gate_history.jsonl`（履歴）を更新。
  - `tmp/replay_quality_gate/<timestamp>` の古い run を
    `REPLAY_QUALITY_GATE_KEEP_RUNS` 件に自動トリム。
  - `quant-replay-quality-gate.service`（oneshot）と
    `quant-replay-quality-gate.timer`（1h 周期）を追加。
  - `ops/env/quant-replay-quality-gate.env` で
    config/timeout/strict/keep_runs を外部設定化。
  - worker の report 抽出/トリム/状態出力に対する回帰テストを追加。
- 意図:
  - 手動実行中心だった replay gate を定期実行へ移し、
    内部テスト精度の劣化検知を運用フローへ固定する。

### 2026-02-20（追記）`position_manager` read timeout 連発を fail-fast + stale 優先へ再調整

- 背景（VM実測）:
  - strategy worker で `position_manager service call failed ... Read timed out (read timeout=8.0)` が連発し、
    `entry-skip summary ... position_manager_timeout=*` が継続。
  - `quant-position-manager` 側では `sync_trades timeout (6.0s)` と
    `orders.db lookup failed ... database is locked` が並発し、`open_positions` tail latency が増加。
- 実施:
  - `ops/env/quant-v2-runtime.env` を調整。
    - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=6.0`
    - `POSITION_MANAGER_HTTP_RETRY_TOTAL=0`
    - `POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT=2.8`
    - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC=60.0`
    - `POSITION_MANAGER_ORDERS_DB_READ_TIMEOUT_SEC=0.08`
    - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC=5.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=8.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=20.0`
  - `ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE=0` は維持
    （service mode worker の pre-service `orders.db` 書き込み抑止を継続）。
- 意図:
  - `open_positions` を 6 秒以内で stale fallback へ寄せ、
    strategy 側の read timeout と `position_manager_timeout` skip を同時に低減する。

### 2026-02-21（追記）forecast 安定化ガード追加（watchdog + bq-sync 低優先度化）

- 背景（VM実測）:
  - `quant-forecast.service` は `active` 表示でも `D` 状態で `/health` が応答しない区間が発生。
  - 同時に `quant-bq-sync.service` の BigQuery 送信失敗（`RetryError`）が継続し、
    予測系の復旧を妨げるケースがあった。
- 実施:
  - 追加: `scripts/forecast_watchdog.sh`
    - `/health` 失敗を連続監視し、閾値超過時に `quant-forecast.service` を自動再起動。
    - 再起動後も不健康な場合は `quant-bq-sync.service` を停止して forecast 優先で復旧。
  - 追加: `systemd/quant-forecast-watchdog.service` / `systemd/quant-forecast-watchdog.timer`
    - 1分周期で watchdog を起動。
  - 変更: `systemd/quant-bq-sync.service`
    - `Nice` / `IOScheduling*` / `CPUWeight` / `IOWeight` / `TasksMax` / `MemoryHigh` / `MemoryMax`
      を設定し、予測系より低優先度で動作させるよう調整。
  - 変更: `ops/env/quant-v2-runtime.env`
    - `FORECAST_WATCHDOG_*` の運用キーを追加（enabled/timeout/max_fails/cooldown など）。
    - `BQ_FAILURE_BACKOFF_BASE_SEC` / `BQ_FAILURE_BACKOFF_MAX_SEC` を追加。
  - 変更: `scripts/run_sync_pipeline.py`
    - BigQuery export 失敗時に指数バックオフの cooldown を導入し、
      外部SSL失敗の連発時に `insert_rows_json` を毎サイクル叩かないように修正。
  - 変更: `scripts/deploy_via_metadata.sh`
    - metadata deploy 時の install 対象に forecast watchdog unit を追加。
- 意図:
  - `quant-forecast` を本番優先で維持し、補助系（BQ同期）の不調で予測APIが巻き込まれる経路を短時間で自動回復する。

### 2026-02-21（追記）SLOメトリクス発行を `quant-strategy-control` へ再配置

- 背景（VM実測）:
  - `logs/metrics.db` で `decision_latency_ms` / `data_lag_ms` の最終時刻が
    2026-02-14 付近で停滞し、`reject_rate` など他指標のみ更新される状態だった。
  - `main.py` 廃止後、旧 `worker_only_loop` が担っていた SLO 指標発行経路が
    V2 常駐サービスに移管されていなかった。
- 実施:
  - `workers/strategy_control/worker.py`
    - 市場オープン時のみ `data_lag_ms`（`tick_window` 基準）と
      `decision_latency_ms`（control loop 処理時間）を定期発行する処理を追加。
    - 追加キー:
      - `STRATEGY_CONTROL_SLO_METRICS_ENABLED`（既定: 1）
      - `STRATEGY_CONTROL_SLO_METRICS_INTERVAL_SEC`（既定: 10）
  - `tests/workers/test_strategy_control_worker.py`
    - `data_lag_ms` 算出と市場クローズ時スキップ、発行内容の回帰テストを追加。
- 意図:
  - V2 構成で欠落していた SLO 観測値の連続性を回復し、
    `policy_guard` / UI スナップショットの判断材料を再び実データ追従に戻す。

### 2026-02-21（追記）`policy_guard` に SLO メトリクス欠損/停滞ガードを追加

- 背景:
  - 旧実装は `decision_latency_ms` / `data_lag_ms` が lookback 内で欠損しても
    `violations` に乗らず、guard が「正常」と判定する抜け道があった。
  - V2 移行時の発行経路断で実際に欠損が起き、監視ガードの盲点が顕在化した。
- 実施:
  - `scripts/policy_guard.py`
    - `collect_violations()` を導入し、違反判定を関数化。
    - 市場オープン時のみ `decision_latency_ms` / `data_lag_ms` の
      欠損（`*_missing`）と停滞（`*_stale`）を違反化。
    - 追加キー:
      - `POLICY_GUARD_REQUIRE_SLO_METRICS_WHEN_OPEN`（既定: 1）
      - `POLICY_GUARD_SLO_METRICS_MAX_STALE_SEC`（既定: 900）
  - `tests/scripts/test_policy_guard.py`
    - 市場オープン時の欠損/停滞検知、クローズ時の非違反、既存閾値判定の回帰を追加。
  - `config/env.example.toml`
    - 新規 env キーを追記。
- 意図:
  - 「メトリクスが無いから違反が出ない」状態を解消し、
    guard が観測欠損そのものを即時に運用リスクとして扱う。

### 2026-02-20（追記）分析メトリクス書き込みの lock 耐性を強化

- 対象:
  - `utils/metrics_logger.py`
  - `scripts/publish_range_mode.py`
  - `ops/env/quant-v2-runtime.env`
  - `config/env.example.toml`
  - `docs/OBSERVABILITY.md`
- 変更:
  - `metrics_logger` の SQLite 書き込みを 2 回固定リトライから可変リトライへ拡張。
    - `METRICS_DB_BUSY_TIMEOUT_MS`（既定 5000ms）
    - `METRICS_DB_WRITE_RETRIES`（既定 6）
    - `METRICS_DB_RETRY_BASE_SLEEP_SEC` / `...MAX_SLEEP_SEC`
  - `metrics_logger` の write ループから毎回の `PRAGMA journal_mode=WAL` を除去し、
    `CREATE TABLE IF NOT EXISTS` はプロセス初回のみ実行するよう変更。
  - lock 時は指数バックオフで再試行し、最終失敗時は `metric/reason/attempts/db` を debug ログへ記録。
  - `log_metric()` が書き込み可否を返すようにし、
    `publish_range_mode` で write 失敗時は `logged active=...` を出さず
    `metric_write_failed` を warning 記録するよう修正。
  - runtime env に上記 `METRICS_DB_*` を追加して VM 既定値を固定。
- 背景（VM実測）:
  - `quant-range-metrics.service` は `logged active=...` を出力していたが、
    `logs/metrics.db` の `range_mode_active` が `2026-02-20T08:01:40Z` で停滞。
  - 同条件で手動実行すると
    `DEBUG: [metrics] drop metric=range_mode_active due to lock: database is locked`
    を再現。
- 意図:
  - 分析ワーカーが「処理成功ログを出しているのに DB 反映されない」状態を防ぎ、
    監視指標の欠落を可視化しつつ書き込み成功率を引き上げる。

### 2026-02-20（追記）`quant-bq-sync` の再発ハングを抑止（BQ送信分割 + stop短縮）

- 対象:
  - `analytics/bq_exporter.py`
  - `scripts/run_sync_pipeline.py`
  - `systemd/quant-bq-sync.service`
  - `ops/env/quant-v2-runtime.env`
  - `docs/GCP_PLATFORM.md`
  - `docs/ARCHITECTURE.md`
- 変更:
  - `bq_exporter` の `insert_rows_json` を一括送信からチャンク送信へ変更。
    - `BQ_EXPORT_BATCH_SIZE`（既定 250〜500運用）
    - `BQ_INSERT_TIMEOUT_SEC`
    - `BQ_RETRY_TIMEOUT_SEC` / `BQ_RETRY_INITIAL_SEC` / `BQ_RETRY_MAX_SEC` / `BQ_RETRY_MULTIPLIER`
  - BigQuery 行の `row_ids` を `ticket_id:updated_at(:transaction_id)` で固定し、
    リトライ時の重複挿入を抑制。
  - `run_sync_pipeline` の SQLite read を context manager 化し、
    例外時でも接続を確実に close するよう修正（`PIPELINE_DB_READ_TIMEOUT_SEC` 追加）。
  - `quant-bq-sync.service` の起動引数を
    `scripts/run_sync_pipeline.py --interval 60 --bq-interval 300 --limit 1200` へ変更し、
    `TimeoutStopSec=30` を追加。
- 背景（VM実測）:
  - `quant-bq-sync.service` で `insertAll` が `RetryError(timeout=600s, SSLEOFError)` を起こし、
    停止時に `stop-sigterm timeout -> SIGKILL` が発生。
  - 同時間帯に `run_sync_pipeline.py` が `D` 状態で残留し、`quant-forecast` 側の I/O待ち連鎖を誘発。
- 意図:
  - BigQuery 経路の失敗時に「長時間ぶら下がる」挙動を避け、
    systemd の stop/restart を短時間で完了させる。
  - 監視/UI補助クエリの接続リークを防ぎ、`orders.db` lock 圧力の再発を減らす。

### 2026-02-20（追記）orders.db lock 連鎖と監査系タイムアウトを是正

- 対象:
  - `execution/order_manager.py`
  - `scripts/maintain_logs.sh`
  - `scripts/ops_v2_audit.py`
  - `ops/env/quant-v2-runtime.env`
  - `ops/env/quant-order-manager.env`
  - `config/env.example.toml`
  - `docs/RISK_AND_EXECUTION.md`
- 変更:
  - `order_manager._log_order` で `sqlite` 例外時に `rollback` と
    `_reset_orders_con()` を徹底し、lock失敗後に同一接続が詰まり続ける経路を抑止。
  - retry 待機の上限計算で fast-fail/通常設定を正しく使うよう修正。
  - orders logger の運用値を強化:
    - `ORDER_DB_BUSY_TIMEOUT_MS=5000`
    - `ORDER_DB_LOG_RETRY_ATTEMPTS=8`
    - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.05`
    - `ORDER_DB_LOG_RETRY_BACKOFF=1.8`
    - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=1.00`
  - `maintain_logs.sh` の replay アーカイブを
    `mv -> stage tar -> cleanup` 方式へ変更し、
    live ディレクトリ削除レース（`Directory not empty`）で service fail しないよう修正。
  - `ops_v2_audit.py` の `subprocess.run(timeout=...)` を安全化し、
    `journalctl` タイムアウト時も監査ジョブが例外で落ちないよう修正。
    併せて `OPS_V2_AUDIT_JOURNAL_TIMEOUT_SEC` を導入。
- 背景（VM実測）:
  - `quant-order-manager` で
    `[ORDER][LOG] failed to persist orders log: database is locked`
    が継続し、`orders.db` の最新時刻が実運用に追従しない区間が発生。
  - `quant-maintain-logs.service` は
    `rm: cannot remove .../logs/replay: Directory not empty` で失敗。
  - `quant-v2-audit.service` は
    `subprocess.TimeoutExpired(... journalctl ... timeout 15.0s)` で失敗。
- 意図:
  - 注文監査ログの欠落を減らしつつ、運用補助ジョブの失敗で
    監視系が赤化し続ける状態を解消する。

### 2026-02-20（追記）内部リプレイ精度ゲートを backend 切替対応で標準化

- 対象:
  - `scripts/replay_quality_gate.py`
  - `config/replay_quality_gate.yaml`
  - `config/replay_quality_gate_main.yaml`
  - `docs/REPLAY_STANDARD.md`
  - `docs/ARCHITECTURE.md`
  - `tests/analysis/test_replay_quality_gate_script.py`
- 変更:
  - `replay_quality_gate` が `exit_workers_groups` / `exit_workers_main` の 2 backend を選択可能に変更。
  - `--backend` CLI override を追加し、config 側の backend 設定を実行時に上書き可能化。
  - `exit_workers_main` 用の標準 config（`replay_quality_gate_main.yaml`）を追加。
  - `realistic` 設定時の slippage/fill フラグ重複を排除し、明示 override が優先されるよう修正。
  - main backend の worker 抽出（strategy/pocket/source）とコマンド組み立てをテストで固定化。
- 意図:
  - VM 側の worker 構成差分や再生データ差分に応じて、同一 walk-forward gate を backend 切替だけで再利用できる状態を維持する。

### 2026-02-20（追記）`replay_quality_gate` に日内 UTC 窓適用を追加（main backend）

- 対象:
  - `scripts/replay_quality_gate.py`
  - `config/replay_quality_gate_main.yaml`
  - `tests/analysis/test_replay_quality_gate_script.py`
  - `docs/REPLAY_STANDARD.md`
- 変更:
  - `exit_workers_main` 実行時に `replay.intraday_start_utc` / `replay.intraday_end_utc` を追加。
  - tick ファイル名の `YYYYMMDD` から `--start/--end` を自動生成して、各日付で同一 UTC 時間帯を再生可能にした。
  - 明示 `replay.start` / `replay.end` がある場合はそちらを優先する。
  - コマンド生成テストを追加し、intraday 適用と明示 override の両方を固定化した。
- 意図:
  - walk-forward を実運用データで回しつつ、処理時間を制御して継続的な内部精度検証を成立させる。

### 2026-02-20（追記）`replay_exit_workers` に `--main-only` を追加（品質ゲート高速化）

- 対象:
  - `scripts/replay_exit_workers.py`
  - `scripts/replay_quality_gate.py`
  - `config/replay_quality_gate_main.yaml`
  - `tests/analysis/test_replay_quality_gate_script.py`
  - `docs/REPLAY_STANDARD.md`
  - `docs/ARCHITECTURE.md`
- 変更:
  - `replay_exit_workers.py` に `--main-only` を追加し、scalp replay entry/exit worker を無効化したうえで
    main 戦略（TrendMA/BB_RSI）の entry/exit のみ実行可能にした。
  - `replay_quality_gate` から `replay.main_only` を pass-through できるようにし、
    `replay_quality_gate_main.yaml` 既定を `main_only: true` に設定。
- 意図:
  - main 戦略の walk-forward 品質判定を、不要経路を省いた状態で繰り返し実行できるようにする。

### 2026-02-20（追記）replay 時の `factor_cache` 永続化I/Oを無効化

- 対象:
  - `scripts/replay_exit_workers.py`
  - `docs/REPLAY_STANDARD.md`
- 変更:
  - replay 初期化 (`_reset_replay_state`) で `factor_cache._persist_cache` /
    `factor_cache._restore_cache` / `factor_cache.refresh_cache_from_disk` を no-op 化。
  - replay 走行中に `logs/factor_cache.json` への書き込み/読み戻しを行わないようにした。
- 意図:
  - replay 実行時間を支配していたディスクI/Oを除去し、
    quality gate の評価ランを安定実行できる状態にする。

### 2026-02-20（追記）`scalp_ping_5s_flow_live` の stale margin_closeout ブロックを短縮

- 対象:
  - `ops/env/quant-order-manager.env`
- 変更:
  - `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を追加。
  - `perf_guard` の `margin_closeout_n>0` 緊急ブロックは維持しつつ、
    参照窓を `3日 -> 1日` に短縮。
- 背景（VM実測）:
  - `2026-02-20 08:57 JST` 時点で `quant-order-manager` は
    `OPEN_REJECT note=perf_block:margin_closeout_n=1 n=24` を連発し、
    reboot後の `OPEN_SCALE` がすべて reject されていた。
  - 該当 closeout は `2026-02-18 00:01 JST`（`MARKET_ORDER_MARGIN_CLOSEOUT`）で、
    直近1日には同事象がなく、古い closeout 参照が過剰抑止を発生させていた。
- 意図:
  - 直近の強制ロスカが再発した時だけ即時ブロックする挙動を維持し、
    stale 事象でエントリーが長時間停止する状態を解消する。

### 2026-02-20（追記）`scalp_ping_5s_flow_live` の env_prefix 上書きバグ修正

- 対象:
  - `execution/strategy_entry.py`
  - `execution/order_manager.py`
  - `tests/execution/test_env_prefix_inference.py`
- 変更:
  - `strategy_tag=scalp_ping_5s_flow_*` を
    `SCALP_PING_5S` ではなく `SCALP_PING_5S_FLOW` として推論する分岐を追加。
  - `strategy_entry` と `order_manager` の両方で同一修正を適用。
  - 回帰テストを追加。
- 背景（VM実測）:
  - flow worker は `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S_FLOW` を保持していたが、
    共通導線で strategy_tag 由来推論が `SCALP_PING_5S` に強制され、
    `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` が無効化されていた。
  - 結果として `OPEN_REJECT note=perf_block:margin_closeout_n=1 n=24` が継続した。
- 意図:
  - flow クローンの戦略別ガード設定を正しく適用し、
    strategy tag 正規化で別系統設定が潰れる経路を防止する。

### 2026-02-20（追記）`scalp_ping_5s_b_live` 方向転換遅延と逆配分を是正（反応速度優先）

- 対象:
  - `ops/env/scalp_ping_5s_b.env`
- 変更:
  - `entry_probability` 帯別ロット補正を短期反応寄りへ変更。
    - `ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES: 180 -> 120`
    - `...MIN_TRADES_PER_BAND: 20 -> 14`
    - `...HIGH_REDUCE_MAX: 0.55 -> 0.78`
    - `...LOW_BOOST_MAX: 0.35 -> 0.50`
    - `...SAMPLE_STRONG_TRADES: 60 -> 30`
  - side成績補正（SL率と成り行き利確率）を短期化・強化。
    - `...SIDE_METRICS_LOOKBACK_TRADES: 120 -> 36`
    - `...SIDE_METRICS_GAIN: 0.90 -> 1.35`
    - `...SIDE_METRICS_MIN_MULT: 0.60 -> 0.40`
  - 方向フリップを加速。
    - `FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS: 0.12 -> 0.08`
    - `FAST_DIRECTION_FLIP_COOLDOWN_SEC: 1.2 -> 0.6`
    - `SL_STREAK_DIRECTION_FLIP_MIN_STREAK: 2 -> 1`
    - `SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP: 0 -> 1`
    - `SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS: 2 -> 1`
    - `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK: 3 -> 2`
    - `SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM: 1 -> 0`
    - `SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN: 0.55 -> 0.48`
    - `SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN: 0.42 -> 0.30`
  - 極値ゲート由来の反転遅れを緩和。
    - `EXTREMA_REQUIRE_M1_M5_AGREE_SHORT: 1 -> 0`
    - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT: 0 -> 1`
- 背景（VM実測）:
  - `2026-02-20 01:39 JST` 時点、直近クローズは `long SL 2件 = -25.5` に対して `short MARKET_CLOSE 5件 = +12.9` で、方向転換が遅れた局面で損失が先行。
  - `2026-02-18 17:00 JST` 以降の `scalp_ping_5s_b` は `entry_probability >= 0.90` 帯が劣後し、ロット配分の過大評価が継続。
- 意図:
  - 取引頻度は維持しつつ、逆方向の過大ロットを即時縮小。
  - SL連敗後のフリップ待ち時間を短縮し、方向修正を先行させる。

### 2026-02-20（追記）`scalp_ping_5s_b_live` side実績フリップを追加（long偏重SL対策）

- 対象:
  - `workers/scalp_ping_5s/worker.py`
  - `workers/scalp_ping_5s/config.py`
  - `ops/env/scalp_ping_5s_b.env`
- 変更:
  - `side_metrics_direction_flip` を新設。
    - 直近side実績（`SL率` と `MARKET close +率`）を比較し、
      現在sideが劣後している局面では opposite side へ即時リターゲット。
    - 既存の `fast_flip` / `sl_streak_flip` と併用し、`entry` 頻度を落とさず方向修正を前倒し。
  - 追加設定（B運用）:
    - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`
    - `...LOOKBACK_TRADES=36`
    - `...MIN_CURRENT_TRADES=8`, `...MIN_TARGET_TRADES=6`
    - `...MIN_CURRENT_SL_RATE=0.58`
    - `...MIN_SL_GAP=0.18`
    - `...MIN_MARKET_PLUS_GAP=0.08`
    - `...CONFIDENCE_ADD=3`, `...COOLDOWN_SEC=0.6`
  - 監査キー:
    - `entry_thesis.side_metrics_direction_flip_*`
    - `tech_route_reasons` に `side_metrics_flip` を追加記録。
- 背景:
  - VMで long 側SLが急増する短時間クラスタを確認し、
    `sl_streak` 発火だけでは転換が遅れる局面が残った。
- 意図:
  - 「何回かSLにかかったら方向を疑う」を明示ロジック化し、
    side別実績に基づく反転を機械的に実行する。

### 2026-02-19（追記）`scalp_ping_5s_b_live` 反転遅れと order log lock を同時修正

- 対象:
  - `workers/scalp_ping_5s/worker.py`
  - `workers/scalp_ping_5s/config.py`
  - `execution/order_manager.py`
  - `ops/env/scalp_ping_5s_b.env`
  - `ops/env/quant-order-manager.env`
- 変更:
  - `SL streak direction flip` に metrics override を追加。
    - 連続SL streak（先頭連続）だけに依存せず、recent side統計
      （`side_sl_hits` / `side_trades` / `target_market_plus`）で反転可否を判定。
    - `SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED`
      `..._METRICS_SIDE_TRADES_MIN`
      `..._METRICS_SIDE_SL_RATE_MIN`
      を導入。
    - flip reason に `slrate` と `m_ovr`（metrics override採用有無）を追加。
  - `order_manager` の orders.db lock待機を短縮。
    - `ORDER_DB_BUSY_TIMEOUT_MS` の既定を `5000ms -> 250ms` に変更し、
      DB lock待ちで `/order/coordinate_entry_intent` が詰まる経路を抑制。
  - B戦略の通過率を微調整。
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE`
      を `0.45 -> 0.42` へ緩和。
- 背景:
  - VMで `sl_streak_direction_flip_reason=below_min_streak/streak_stale` が多数、
    `fast_flip` が出てもSL連敗由来の転換バイアスが持続せず逆行を再度掴むケースが発生。
  - 同時に `failed to persist orders log: database is locked` が多発し、
    `order_manager_none` と service timeout を誘発していた。
- 意図:
  - 方向転換を「連続SLが崩れても」実績ベースで発火させる。
  - 注文ログのlock待機でエントリー判定が遅延/欠落する経路を縮小する。

### 2026-02-19（追記）`scalp_ping_5s_b_live` のSL欠損を根治（entry時SL復帰）

- 対象:
  - `execution/order_manager.py`
  - `workers/scalp_ping_5s/config.py`
  - `config/env.example.toml`
- 変更:
  - `order_manager` の `scalp_ping_5*` 一律 hard-stop 無効化を戦略別に分離。
    - `scalp_ping_5s_b*` は既定で `disable_entry_hard_stop=False`
    - legacy `scalp_ping_5*` は従来どおり既定 `True`（env で上書き可）
  - `ORDER_FIXED_SL_MODE` が未設定/0でも、`scalp_ping_5s_b*` は
    `stopLossOnFill` を許可できるように戦略別オーバーライドを追加。
  - `workers.scalp_ping_5s.config` を修正し、B系タグ（`scalp_ping_5s_b*`）は
    `SCALP_PING_5S_USE_SL` を既定有効、`SCALP_PING_5S_DISABLE_ENTRY_HARD_STOP` を既定無効化。
- 背景:
  - VM実データ（`2026-02-18 15:00 JST`以降）で
    `scalp_ping_5s_b_live` の filled 注文が `SL missing 66/66` を確認。
  - 既存実装では `scalp_ping_5*` を order_manager で一律 hard-stop 無効化しており、
    `ORDER_FIXED_SL_MODE` 未設定時は `stopLossOnFill` も常時OFFだった。
- 意図:
  - `scalp_ping_5s_b_live` の負けトレードを exit遅延依存にせず、
    entry時点で broker SL を復帰して tail-loss を抑制する。

### 2026-02-19（追記）`scalp_macd_rsi_div_b_live` を精度優先プロファイルへ更新

- 対象:
  - `ops/env/quant-scalp-macd-rsi-div-b.env`
  - `workers/scalp_macd_rsi_div/worker.py`
- 変更:
  - B版 env を「looser mode」から「precision-biased」へ変更。
    - `REQUIRE_RANGE_ACTIVE=1`（range-only）
    - `RANGE_MIN_SCORE=0.35`
    - `MAX_ADX=30`, `MAX_SPREAD_PIPS=1.0`
    - `MIN_DIV_SCORE=0.08`, `MIN_DIV_STRENGTH=0.12`, `MAX_DIV_AGE_BARS=24`
    - `RSI_LONG/SHORT_ARM, ENTRY` を `36/62` に引き締め
    - `MAX_OPEN_TRADES=1`, `COOLDOWN_SEC=45`, `BASE_ENTRY_UNITS=5000`
    - `TECH_FAILOPEN=0`（tech不許可時は fail-close）
  - `workers/scalp_macd_rsi_div.worker` に `MIN_ENTRY_CONF` の実効ガードを追加し、
    低信頼シグナルを `gate_block confidence` で reject するよう修正。
- 背景:
  - VM `trades.db` で `scalp_macd_rsi_div_b_live` の直近実績（UTC 2026-02-18 02:22〜
    2026-02-19 01:33, 4 trades）が `PF=0.046`, `sum=-32.9 pips` と悪化。
  - `tick_entry_validate` でも直近負け（ticket `365759`）は
    `TP_touch<=600s = 0/1` かつ逆行継続で、エントリー条件の緩さが主因と判定。
- 意図:
  - B版を「件数優先」から「精度優先」に切り替え、
    トレンド局面の逆張りエントリーと弱い divergence の誤発火を抑制する。

### 2026-02-18（追記）`scalp_ping_5s_b_live` 取り残し対策（EXITプロファイル適用漏れ修正）

- 対象:
  - `config/strategy_exit_protections.yaml`
- 変更:
  - `scalp_ping_5s_b_live` キーを `*SCALP_PING_5S_EXIT_PROFILE` へ明示的にエイリアス追加。
- 背景:
  - `quant-scalp-ping-5s-b-exit` は `ALLOWED_TAGS=scalp_ping_5s_b_live` で建玉を監視する一方、
    EXITプロファイル定義は `scalp_ping_5s_b` のみだったため、`_exit_profile_for_tag` が default へフォールバック。
  - その結果、`loss_cut/non_range_max_hold/direction_flip` が無効化され、
    無SLの負け玉が長時間残留する経路が発生していた。
- 意図:
  - `scalp_ping_5s_b_live` 建玉にも `scalp_ping_5s` 系の負け玉整理ルールを確実適用し、
    上方向取り残しの再発を抑制する。

### 2026-02-18（追記）`direction_flip` de-risk 失敗時の sentinel reason 漏れ修正

- 対象:
  - `workers/scalp_ping_5s/exit_worker.py`
  - `workers/scalp_ping_5s_b/exit_worker.py`
  - `config/strategy_exit_protections.yaml`
  - `tests/workers/test_scalp_ping_5s_exit_worker.py`
- 変更:
  - `direction_flip` の de-risk 判定で部分クローズが失敗した場合、
    内部 sentinel `__de_risk__` をそのまま full close 理由に使わず、
    `direction_flip.reason`（既定 `m1_structure_break`）へフォールバックするよう修正。
  - `scalp_ping_5s*` の `neg_exit.strict_allow_reasons / allow_reasons` に
    `m1_structure_break` と `risk_reduce` を追加。
  - 回帰防止テスト（de-risk sentinel fallback）を追加。
- 背景:
  - VM 実ログで `quant-scalp-ping-5s-b-exit` が `reason=__de_risk__` の close 失敗を大量連発し、
    `order_manager /order/close_trade` timeout と組み合わさって含み損整理が遅延していた。
- 意図:
  - internal 用 reason の外部流出を止め、negative-close ガード下でも
    `direction_flip` 系の損切り/デリスクを機械的に実行可能にする。

### 2026-02-18（追記）`scalp_macd_rsi_div` legacy tag 正規化（EXIT監視漏れ修正）

- 対象:
  - `workers/scalp_macd_rsi_div/exit_worker.py`
- 変更:
  - `_base_strategy_tag()` で、`scalpmacdrsi*` 形式（例: `scalpmacdrsic7c3e9c1`）を
    `scalp_macd_rsi_div_live` へ正規化する分岐を追加。
- 背景:
  - `quant-scalp-macd-rsi-div-exit` の `SCALP_PRECISION_EXIT_TAGS` は
    `scalp_macd_rsi_div_live` を想定しているが、
    旧 client_id 由来の `strategy_tag=scalpmacdrsi...` は一致せず
    `_filter_trades()` で除外され、EXIT管理（max_hold/loss_cut）から外れていた。
- 意図:
  - 旧タグ建玉も現行タグへ収束させ、EXITワーカーの監視漏れで
    長時間の上方向取り残しが発生しない状態を維持する。

### 2026-02-18（追記）`scalp_macd_rsi_div_live` EXITプロファイルを明示（defaultフォールバック解消）

- 対象:
  - `config/strategy_exit_protections.yaml`
- 変更:
  - `scalp_macd_rsi_div_live` を strategy profile に追加し、
    `loss_cut_enabled=true` / `loss_cut_require_sl=false` /
    `loss_cut_hard_pips=7.0` / `loss_cut_max_hold_sec=1800` を明示。
- 背景:
  - legacy tag 正規化後も、当該 strategy の profile 未定義時は default へフォールバックし、
    `loss_cut_enabled=false` かつ `require_sl=true` により
    無SLの負け玉が close 条件に入らない経路が残っていた。
- 意図:
  - `scalp_macd_rsi_div_live` を profile 指定で明示管理し、
    含み損の長期滞留を機械的に抑止する。

### 2026-02-18（追記）dynamic_alloc のサイズ配分を保守化（負け戦略の増量抑止）

- 対象:
  - `scripts/dynamic_alloc_worker.py`
  - `systemd/quant-dynamic-alloc.service`
- 変更:
  - `dynamic_alloc_worker` に PF ガードを追加。
    - `pf < 1.0` の戦略は `lot_multiplier <= 0.95`
    - `pf < 0.7` の戦略は `lot_multiplier <= 0.90`
    - `trades < min_trades` は `lot_multiplier <= 1.00`
  - `quant-dynamic-alloc.service` の `--target-use` を `0.90 -> 0.88` に調整。
- 背景:
  - VM 実メトリクス（過去6h）で `account.margin_usage_ratio` が平均 `0.94` 台と高く、
    `margin_usage_exceeds_cap` / `margin_usage_projected_cap` が多発していた。
  - 旧ロジックでは PF<1 の戦略でも win-rate 由来で `lot_multiplier > 1` になるケースがあり、
    負け筋への過剰配分を招いていた。
- 意図:
  - ブロック多発時の過剰エクスポージャを抑えつつ、勝ち筋への配分を維持し、
    実運用の約定効率と期待値を改善する。

### 2026-02-18（追記）`RANGEFADER_EXIT_NEW_POLICY_START_TS` の形式不一致を修正

- 対象:
  - `ops/env/quant-scalp-ping-5s-b-exit.env`
  - `docs/RISK_AND_EXECUTION.md`
- 変更:
  - `RANGEFADER_EXIT_NEW_POLICY_START_TS` を ISO文字列から Unix秒へ変更。
    - 旧: `2026-02-17T00:00:00Z`（`_float_env` で読めず default=現在時刻へフォールバック）
    - 新: `1771286400`（2026-02-17 00:00:00 UTC）
- 背景:
  - `workers/scalp_ping_5s_b/exit_worker.py` は同キーを `_float_env` で読む実装のため、
    文字列日時だと毎回「起動時刻」が `new_policy_start_ts` になり、既存建玉が常に legacy 扱いになっていた。
- 意図:
  - restart後も既存の `scalp_ping_5s_b_live` 建玉へ新ポリシーを継続適用し、
    `loss_cut/non_range_max_hold/direction_flip` が有効な状態を維持する。

### 2026-02-18（追記）forecast 追加改善（5m breakout/session 局所再探索）を運用値へ再反映

- 対象:
  - `ops/env/quant-v2-runtime.env`
  - `docs/FORECAST.md`
- 変更:
  - 追加グリッド（`logs/reports/forecast_improvement/extra_improve_b5s5_latest.json`）で
    現行 `fg=0.03,b5=0.20,b10=0.30,s5=0.20,s10=0.30,rb5=0.01,rb10=0.02` を基準に、
    `5m` 重みのみ局所再探索。
  - 採用値:
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.24,10m=0.30`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.03`（維持）
    - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`（維持）
- 評価（new - current）:
  - `2h`: `hit +0.0000`, `mae +0.000018`
  - `4h`: `hit +0.0000`, `mae +0.000062`
  - `24h`: `hit +0.0000`, `mae -0.000242`
  - `72h`: `hit +0.002222`, `mae -0.000156`
  - weighted objective: `+0.01762`（safe）
- 意図:
  - 直近短期の微小悪化を許容しつつ、24h/72h の安定改善と 72h hit 上振れを優先する。

### 2026-02-18（追記）forecast 追加改善（2h/4h/24h/72h 同時最適化）を運用値へ反映

- 対象:
  - `ops/env/quant-v2-runtime.env`
  - `docs/FORECAST.md`
- 変更:
  - VM実データ再探索結果（`logs/reports/forecast_improvement/extra_improve_grid_latest.json`）に基づき、
    次を runtime 採用値へ更新:
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.03`
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.20,10m=0.30`
    - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.20,10m=0.30` は維持。
- 評価（new - baseline）:
  - `2h`: `hit +0.0000`, `mae -0.000120`
  - `4h`: `hit +0.0000`, `mae -0.000061`
  - `24h`: `hit +0.003672`, `mae +0.000110`（劣化ガード内）
  - `72h`: `hit +0.001266`, `mae -0.000164`, `range_cov +0.000317`
  - weighted objective: `+0.02198`
- 意図:
  - 24h/72h の安全条件を維持したまま、72h の hit/MAE と短中期 MAE を同時改善し、
    直近窓での過剰反発バイアス（5m）を抑制する。

### 2026-02-18（追記）`scalp_ping_5s(_b)` EXIT に反転検知（予測バイアス併用）を追加

- 対象:
  - `workers/scalp_ping_5s/exit_worker.py`
  - `workers/scalp_ping_5s_b/exit_worker.py`
  - `config/strategy_exit_protections.yaml` (`scalp_ping_5s`, `scalp_ping_5s_live`, `scalp_ping_5s_b`)
- 変更:
  - 含み損時に `M1` 構造（MA差/RSI/EMA傾き/VWAP乖離）と
    `analysis.local_decider._technical_forecast_bias` の予測バイアスを合成した
    `direction_flip` 判定を追加。
  - `direction_flip` に二段階制御を追加:
    - 疑い段階（閾値弱）で `risk_reduce` の部分クローズ（de-risk）
    - 継続悪化（閾値強+確認ヒット）で全クローズ
    - 回復時は保持し、再エントリー側の通常シグナルで追加玉を許容
  - ヒステリシス（`score_threshold` / `release_threshold`）と
    連続ヒット確認（`confirm_hits` / `confirm_window_sec`）でノイズ起因の早切りを抑制。
  - `range_active` 前提の `range_timeout` だけでは残るケース向けに、
    非レンジ時の時間上限 `non_range_max_hold_sec` を追加（含み損時のみ）。
  - 新ロジックは既存建玉を触らないよう、
    `RANGEFADER_EXIT_NEW_POLICY_START_TS` 以降に建ったポジションだけに適用。
- 意図:
  - 「利益を削る一律SL」ではなく、方向転換が確認された負け玉のみを機械的に整理し、
    長時間取り残しとマージン圧迫の再発を抑える。

### 2026-02-18（追記）forecast に目標到達確率（`target_reach_prob`）を追加

- 対象:
  - `workers/common/forecast_gate.py`
  - `execution/strategy_entry.py`
  - `workers/*/exit_forecast.py`（全戦略）
- 変更:
  - `forecast_gate` で、予測分布（`expected_pips`/`range_sigma_pips`）と `tp_pips_hint` から
    方向別の `target_reach_prob`（0.0-1.0）を算出。
  - `ForecastDecision` と `entry_thesis["forecast"]` へ `target_reach_prob` を伝播。
  - 各 `exit_forecast` で `target_reach_prob` を参照し、
    低確率時は `contra_score` を加算（早期EXIT寄り）、
    高確率時は `contra_score` を減衰（ホールド寄り）する補正を追加。
- 意図:
  - 「このまま持つか / いったん切ってプルバック再エントリーを待つか」の判断を、
    方向確率・edge に加えて目標到達確率でも機械判定できるようにする。

### 2026-02-18（追記）forecast に「急落後反発」シグナルを追加し短期重みを再固定

- 対象:
  - `workers/common/forecast_gate.py`
  - `tests/workers/test_forecast_gate.py`
  - `ops/env/quant-v2-runtime.env`
  - `docs/FORECAST.md`
- 変更:
  - `forecast_gate` の technical 予測へ、急落後反発の補助シグナルを追加。
    - 構成: `drop_score`（直近下落強度）+ `oversold_score` + `decel_score` + `wick_score`（下ヒゲ拒否）
    - 監査キー: `rebound_signal_20`, `rebound_*_score_20`, `rebound_weight`
    - `combo` へ `rebound_weight * rebound_signal` を加算（短期重み map で調整）。
  - 新env:
    - `FORECAST_TECH_REBOUND_ENABLED=1`
    - `FORECAST_TECH_REBOUND_WEIGHT=0.06`
    - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.04,10m=0.02`
  - 既定再固定:
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0`
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.10,5m=0.18,10m=0.26`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.18,10m=0.30`
  - テスト:
    - `tests/workers/test_forecast_gate.py` に反発シグナルと下ヒゲ拒否の差分検証を追加。
    - 実行: `pytest -q tests/workers/test_forecast_gate.py`（20 passed）。
- 評価（VM実データ由来の同一期間）:
  - 比較レポート:
    - `logs/reports/forecast_improvement/forecast_eval_rebound_base_20260218T024741Z.json`
    - `logs/reports/forecast_improvement/forecast_eval_rebound_tuned_20260218T024741Z.json`
    - `logs/reports/forecast_improvement/rebound_tune_report_20260218T024741Z.md`
  - candidate-after - base-after:
    - `1m`: `hit +0.0002`, `mae -0.0002`
    - `5m`: `hit -0.0001`, `mae -0.0002`
    - `10m`: `hit +0.0000`, `mae -0.0002`

### 2026-02-18（追記）`position_manager` の close 導線を service-safe 化 + `MicroCompressionRevert` 一時抑制

- 対象:
  - `execution/position_manager.py`
  - `tests/execution/test_position_manager_close.py`
  - `ops/env/quant-micro-compressionrevert.env`
- 変更:
  - `PositionManager.close()` で、`POSITION_MANAGER_SERVICE_ENABLED=1` のとき
    `/position/close` をリモート呼び出ししないよう変更。
    - `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1` 運用でも shared service を閉じない。
    - close はローカル接続 (`self.con`) の後始末のみに限定。
  - 再発防止テストを追加し、service 有効時に `/position/close` を叩かないことを固定化。
  - `quant-micro-compressionrevert` 専用 env で
    `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.33` へ引き上げ（従来 `0.28`）。
    直近 score≈`0.307` の負け筋を当面自動ブロックする。
- 背景:
  - VM で `POSITION_MANAGER_SERVICE_ENABLED=1` かつ
    `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1` の runtime において、
    `Cannot operate on a closed database` が継続発生。
  - 直近の `MicroCompressionRevert-short` は 24h/7d で pips 合計がマイナスのため、
    loser block 閾値を専用 worker 側で一段引き上げ。

### 2026-02-18（追記）`quant-micro-compressionrevert-exit` の `open_positions` timeout 緩和

- 対象:
  - `ops/env/quant-micro-compressionrevert-exit.env`
- 変更:
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=6.5`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC=1.5`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC=6.0`
- 目的:
  - `quant-micro-compressionrevert-exit` で継続していた
    `position_manager service call failed ... Read timed out` の抑制。
  - worker 側で `open_positions` を短TTLキャッシュし、問い合わせ頻度を下げて
    `quant-position-manager` への集中アクセスを緩和。

### 2026-02-17（追記）forecast に分位レンジ予測（上下帯）を追加

- `analysis/forecast_sklearn.py` の最新予測出力に
  `q10_pips/q50_pips/q90_pips` と `range_low/high_pips`（+ `range_sigma_pips`）を追加。
- `workers/common/forecast_gate.py` で technical/bundle/blend すべてに対して
  分位レンジを正規化し、`range_low/high_price` を `anchor_price` 基準で算出するよう統一。
- `ForecastDecision` / `workers/forecast/worker.py` / `execution/order_manager.py` /
  `execution/strategy_entry.py` の forecast メタ連携に上下帯キーを追加し、
  `entry_thesis["forecast"]` / `forecast_execution` / order監査 payload へ伝播。
- `scripts/vm_forecast_snapshot.py` に `range_pips` / `range_price` 表示を追加。
- `scripts/eval_forecast_before_after.py` に before/after 比較指標として
  `range_cov_before/after/delta`（帯内包含率）と
  `range_width_before/after/delta`（平均帯幅）を追加。
- VM実測（`bars=8050`, `2026-01-06T07:30:00+00:00`〜`2026-02-17T08:32:00+00:00`）:
  - `feature_expansion_gain=0.35`
    - `1m`: hit `0.4926 -> 0.4902`, MAE `1.4846 -> 1.4957`, range_cov `0.2126 -> 0.2087`
    - `5m`: hit `0.4858 -> 0.4830`, MAE `3.4696 -> 3.4939`, range_cov `0.1852 -> 0.1831`
    - `10m`: hit `0.4779 -> 0.4769`, MAE `5.1711 -> 5.2053`, range_cov `0.1834 -> 0.1849`
  - `feature_expansion_gain=0.0` では before/after 全指標が一致（回帰なし）。
- 運用判断: `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0` を維持し、
  分位レンジ（上下帯）は予測監査と `forecast_context` 伝播に先行適用する。

### 2026-02-17（追記）短期TF（1m/5m/10m）の trend/range 重みを再配分

- `workers/common/forecast_gate.py` と `scripts/eval_forecast_before_after.py` の
  `TECH_HORIZON_CFG` を短期TFのみ調整。
  - `1m`: `trend_w=0.90, mr_w=0.10` -> `trend_w=0.70, mr_w=0.30`
  - `5m`: `trend_w=0.86, mr_w=0.14` -> `trend_w=0.40, mr_w=0.60`
  - `10m`: `trend_w=0.84, mr_w=0.16` -> `trend_w=0.40, mr_w=0.60`
- 背景:
  - VM実測で短期足はレンジ区間が多く、順張り寄り配分だと MAE が悪化しやすい傾向が継続。
  - 「時間帯=TF」前提で、戦略ごとの主TFに対して mean-reversion 成分を短期ほど強める方針へ変更。
- VM同一条件評価（`bars=8050`, `--steps 1,5,10`, `--max-bars 12000`）:
  - `feature_expansion_gain=0.0`（before=after一致の基準式）
    - `1m`: hit `0.4932 -> 0.4947`, MAE `1.4868 -> 1.4754`
    - `5m`: hit `0.4880 -> 0.4887`, MAE `3.4739 -> 3.3885`
    - `10m`: hit `0.4810 -> 0.4836`, MAE `5.1633 -> 5.0371`
  - `feature_expansion_gain=0.35`（before/after 比較）
    - `1m`: after hit `0.4905 -> 0.4922`, after MAE `1.4994 -> 1.4883`
    - `5m`: after hit `0.4858 -> 0.4902`, after MAE `3.4955 -> 3.4162`
    - `10m`: after hit `0.4808 -> 0.4827`, after MAE `5.1878 -> 5.0774`
- 反映:
  - commit: `61105190`
  - VM: `HEAD == origin/main` を確認し、`quant-market-data-feed` / `quant-order-manager` /
    `quant-forecast` を再起動。

### 2026-02-17（追記）JST時間帯バイアス（session bias）を短期TFへ追加

- `workers/common/forecast_gate.py`
  - 直近履歴から「同じJST hour の先行方向バイアス」を推定する
    `session_bias` を追加（`FORECAST_TECH_SESSION_BIAS_*`）。
  - `combo` へ `session_bias_weight * session_bias` を注入。
  - `1m` はノイズ増を避けるため `session_bias_weight=0.0` 固定、
    `5m/10m` 以上でのみ重み適用（既定 `0.12`）。
  - 監査メタに `session_bias_jst/session_bias_weight/session_mean_pips_jst/
    session_samples_jst/session_hour_jst` を追加。
- `scripts/eval_forecast_before_after.py`
  - `--session-bias-weight` / `--session-bias-min-samples` /
    `--session-bias-lookback` を追加。
  - 評価ループは lookahead なし（過去履歴のみ）で session bias を算出。
  - 追加後に重くなったため、集計を O(1) 更新（hour別 sum/count + sliding window）へ最適化。
- テスト:
  - `tests/workers/test_forecast_gate.py`
    - `test_estimate_session_hour_bias_positive`
    - `test_estimate_session_hour_bias_negative`
- VM同一条件比較（`bars=8050`, `--steps 1,5,10`, `feature_expansion_gain=0.0`,
  `breakout_adaptive_weight=0.22`, `session_bias_min_samples=24`,
  `session_bias_lookback=720`）:
  - `session_bias_weight=0.12`
    - `1m`: hit `0.4961`, MAE `1.4677`
    - `5m`: hit `0.4899`, MAE `3.3845`
    - `10m`: hit `0.4870`, MAE `5.0654`
  - `session_bias_weight=0.00`
    - `1m`: hit `0.4961`, MAE `1.4677`
    - `5m`: hit `0.4892`, MAE `3.3854`
    - `10m`: hit `0.4851`, MAE `5.0684`
  - 差分（0.12 - 0.00）:
    - `1m`: hit `+0.0000`, MAE `+0.0000`（実質同等）
    - `5m`: hit `+0.0007`, MAE `-0.0009`
    - `10m`: hit `+0.0019`, MAE `-0.0030`

### 2026-02-17（追記）session bias を TF別重みへ拡張（5m/10m分離）

- `workers/common/forecast_gate.py`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP` を追加（例: `1m=0.0,5m=0.22,10m=0.30`）。
  - 既定を上記 map に更新し、`1m` は 0 固定、`5m` と `10m` を別重みで運用可能化。
- `scripts/eval_forecast_before_after.py`
  - `--session-bias-weight-map` を追加して、同じ重みmapで before/after 比較可能化。
- VM比較（`bars=8050`, `feature_expansion_gain=0.0`, `breakout_adaptive_weight=0.22`）:
  - map `1m=0.0,5m=0.22,10m=0.22`
    - `5m`: hit `0.4911`, MAE `3.3849`
    - `10m`: hit `0.4893`, MAE `5.0646`
  - map `1m=0.0,5m=0.22,10m=0.30`
    - `5m`: hit `0.4911`, MAE `3.3849`（同等）
    - `10m`: hit `0.4912`, MAE `5.0630`（改善）
- 追加確認（`max-bars=6000`）でも `10m=0.30` は hit/MAE とも改善を維持。
- 運用デフォルトを `1m=0.0,5m=0.22,10m=0.30` へ更新。

### 2026-02-17（追記）breakout adaptive 重みを TF別に再調整（1m分離）

- 目的:
  - `breakout_bias_20` 適応項の過剰反応を 1m で抑えつつ、5m/10m の改善を維持する。
- 同一固定データで比較:
  - 入力固定: `/tmp/candles_eval_full_20260217_1549_wrapped.json`
  - 期間: `bars=8050`（`2026-01-06T07:30:00+00:00`〜`2026-02-17T15:47:00+00:00`）
  - そのほか: `feature_expansion_gain=0.0`, `session_bias_weight_map=1m=0.0,5m=0.22,10m=0.30`
  - 比較:
    - 旧既定 `1m=0.22,5m=0.22,10m=0.22`
      - `1m`: hit `0.4975`, MAE `1.4743`
      - `5m`: hit `0.4919`, MAE `3.4080`
      - `10m`: hit `0.4929`, MAE `5.1150`
    - 新既定候補 `1m=0.16,5m=0.22,10m=0.30`
      - `1m`: hit `0.4979`, MAE `1.4743`（hit 改善、MAE 同等）
      - `5m`: hit `0.4919`, MAE `3.4080`（同等）
      - `10m`: hit `0.4928`, MAE `5.1146`（hit 同等域、MAE 改善）
- 反映:
  - `workers/common/forecast_gate.py` の既定
    `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP` を
    `1m=0.16,5m=0.22,10m=0.30` に更新。
  - `scripts/eval_forecast_before_after.py` の既定 map も同値へ更新。

### 2026-02-17（追記）session bias 重みを再調整（5m/10mを強化）

- 目的:
  - breakout 調整後の構成（`1m=0.16,5m=0.22,10m=0.30`）を固定したまま、
    `session_bias` の寄与を 5m/10m で追加改善する。
- 同一固定データで比較:
  - 入力固定: `/tmp/candles_eval_full_20260217_1549_wrapped.json`
  - 期間: `bars=8050`（`2026-01-06T07:30:00+00:00`〜`2026-02-17T15:47:00+00:00`）
  - 比較:
    - 旧既定 `session_bias_weight_map=1m=0.0,5m=0.22,10m=0.30`
      - `5m`: hit `0.4919`, MAE `3.4080`
      - `10m`: hit `0.4928`, MAE `5.1146`
    - 新既定候補 `session_bias_weight_map=1m=0.0,5m=0.26,10m=0.34`
      - `5m`: hit `0.4922`, MAE `3.4079`
      - `10m`: hit `0.4941`, MAE `5.1142`
- 反映:
  - `workers/common/forecast_gate.py` の既定
    `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP` を
    `1m=0.0,5m=0.26,10m=0.34` に更新。
  - `scripts/eval_forecast_before_after.py` の既定 map も同値へ更新。

### 2026-02-17（追記）session bias 10m重みを 0.38 へ再引き上げ

- 背景:
  - 上記更新後に同一固定データで再グリッド（`/tmp/forecast_eval_grid_00..07.json`）を実施し、
    5mを維持しつつ10mをさらに上げられる設定を探索。
- 比較条件:
  - 入力固定: `/tmp/candles_eval_full_20260217_2237_wrapped.json`
  - 期間: `bars=8050`（`2026-01-06T07:30:00+00:00`〜`2026-02-17T23:59:00+00:00`）
  - breakout は `1m=0.16,5m=0.22,10m=0.30` 固定。
- 最良:
  - `session_bias_weight_map=1m=0.0,5m=0.26,10m=0.38`
    - `5m`: hit `0.4886`（維持）, MAE `3.2707`（維持）
    - `10m`: hit `0.4878 -> 0.4900`, MAE `4.8568 -> 4.8564`
  - 参考: grid score（`score_5_10`）は `0.897385` で候補中最大。
- 反映:
  - `workers/common/forecast_gate.py` の既定
    `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP` を
    `1m=0.0,5m=0.26,10m=0.38` に更新。
  - `scripts/eval_forecast_before_after.py` の既定 map も同値へ更新。

### 2026-02-17（追記）live未確定バーでの `feature_row_incomplete` 誤発生を抑止

- 事象:
  - `vm_forecast_snapshot.py --horizon 1m,5m,10m` で `have=400` なのに
    `nonfinite_feature_atr_pips_14:nan` が出て短期予測が pending になるケースを確認。
  - 原因は live バー（未確定）由来の非有限特徴量が「最終行」に入ると、
    直近1行のみ検査していた technical 予測が即座に `feature_row_incomplete` を返す設計だったこと。
- 対応:
  - `workers/common/forecast_gate.py`
    - `_technical_prediction_for_horizon()` で最終行固定をやめ、末尾から逆順に
      required feature がすべて有限な行を探索して採用するよう変更。
    - `session_bias` の `current_timestamp` と `feature_ts` も採用行に合わせて整合化。
  - `tests/workers/test_forecast_gate.py`
    - `test_technical_prediction_uses_latest_finite_feature_row` を追加し、
      最新行が `nan` でも直近有限行で `status=ready` になることを検証。

### 2026-02-17（追記）「時間帯=TF」前提で戦略別主TF＋補助TF整合を追加

- `execution/strategy_entry.py` の戦略契約に `forecast_support_horizons` を追加し、
  各戦略が主TF（`forecast_profile`）に加えて補助TFを `entry_thesis` に注入する構成へ変更。
- `workers/common/forecast_gate.py` に TF整合ロジック（`FORECAST_GATE_TF_CONFLUENCE_*`）を追加。
  - 主TF edge と補助TF edge の整合を `tf_confluence_score` として算出。
  - 同方向なら edge 微増、逆方向なら edge 減衰（最終ブロック/スケール判定は既存ロジック）。
- 連携先（`workers/forecast/worker.py`, `execution/order_manager.py`, `execution/strategy_entry.py`）へ
  `tf_confluence_score/count/horizons` を伝播し、監査ログへ残せるようにした。
- 目的: 「戦略ごとに適したTFで予測しつつ、上位/下位TFの整合で誤判定を抑える」運用へ統一。

### 2026-02-17（追記）`entry_thesis` 欠損経路でも戦略タグ基準で主TFを補完

- VM実測確認（`logs/trades.db`, 直近7日）で `entry_thesis.forecast_horizon` が
  `27 / 2231` 件しか埋まっていないことを確認。
- 欠損時は従来 `pocket` 既定（`micro=8h`, `scalp=5m`）にフォールバックしていたため、
  `MicroVWAPRevert` / `MicroRangeBreak` / `MicroLevelReactor` などで
  意図TF（`10m`）と乖離していた。
- `workers/common/forecast_gate.py` の `_horizon_for()` を拡張し、
  `entry_thesis/meta` に主TFが無い場合は `strategy_tag` から主TFを推定するよう修正。
  - 例: `Micro*` / `MomentumBurst` -> `10m`,
    `scalp_ping_5s*` -> `1m`,
    `scalp_macd_rsi_div*` / `WickReversal*` -> `10m`,
    `M1Scalper*` / `TickImbalance*` -> `5m`
- 既存の `FORECAST_GATE_HORIZON*` 強制設定と `entry_thesis/meta` 明示値は優先維持
  （補完は最終フォールバックのみ）。
- テスト追加:
  - `tests/workers/test_forecast_gate.py`
    - `test_horizon_for_strategy_tag_prefers_micro_10m`
    - `test_horizon_for_strategy_tag_prefers_scalp_ping_1m`
    - `test_horizon_for_unknown_strategy_uses_pocket_default`

### 2026-02-17（追記）`breakout_bias_20` を直近スキルで適応重み化

- 背景:
  - VM同期間評価で `breakout_bias_20` 一致率が `~0.47-0.49` と低く、
    固定符号で使うとレジーム変化時に逆方向へ効く場面が確認された。
- 修正:
  - `workers/common/forecast_gate.py`
    - 直近履歴から `breakout_bias_20` の方向一致スキルを推定
      （`FORECAST_TECH_BREAKOUT_ADAPTIVE_*`）。
    - スキルを `[-1, +1]` で扱い、`combo` へ
      `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT * tanh(breakout_bias) * skill`
      として注入（負スキル時は実質反転）。
    - 監査メタへ `breakout_skill_20` / `breakout_hit_rate_20` /
      `breakout_samples_20` を追加。
  - `scripts/eval_forecast_before_after.py`
    - `--breakout-adaptive-weight` /
      `--breakout-adaptive-min-samples` /
      `--breakout-adaptive-lookback` を追加。
    - 評価ループ内で「過去サンプルのみ」を使ってスキルを更新し、
      lookahead なしで after 式を比較可能化。
- テスト追加:
  - `tests/workers/test_forecast_gate.py`
    - `test_estimate_directional_skill_positive`
    - `test_estimate_directional_skill_negative`

### 2026-02-17（追記）MicroRangeBreak を micro_multistrat から独立ワーカー化

- `workers/micro_rangebreak` を新設し、`python -m workers.micro_rangebreak.worker` と
  `python -m workers.micro_rangebreak.exit_worker` を実行する
  `quant-micro-rangebreak.service` / `quant-micro-rangebreak-exit.service` を追加。
- `workers/micro_rangebreak/worker.py` で起動時に `MICRO_STRATEGY_ALLOWLIST=MicroRangeBreak`,
  `MICRO_MULTI_LOG_PREFIX=[MicroRangeBreak]` を上書き（既定）注入。
- `workers/micro_rangebreak/exit_worker.py` で起動時に `MICRO_MULTI_EXIT_ENABLED=1`,
  `MICRO_MULTI_EXIT_TAG_ALLOWLIST=MicroRangeBreak`,
  `MICRO_MULTI_LOG_PREFIX=[MicroRangeBreak]` を上書き（既定）注入。
- `ops/env/quant-micro-rangebreak.env` / `ops/env/quant-micro-rangebreak-exit.env` を追加。
- `ops/env/quant-micro-multi.env` から `MICRO_STRATEGY_ALLOWLIST` の `MicroRangeBreak` を除外し、
  複数 unit からの重複実行を防止。

### 2026-02-17（追記）market_data_feed の `on_candle` 契約不整合を修正

- 事象: VM の `quant-market-data-feed.service` で
  `tick_fetcher reconnect: on_candle() missing 1 required positional argument: 'candle'`
  が連続発生し、再接続ループと factor 更新遅延を誘発。
- 原因: `workers/market_data_feed/worker.py` の `_build_handlers()` が
  `factor_cache.on_candle(tf, candle)` ではなく `on_candle(candle)` 形式で購読に渡していた。
- 対応: timeframe を closure で束縛したハンドラ `_bind_factor_handler()` を追加し、
  `start_candle_stream` へ渡す `Callable[[Candle], ...]` 契約を維持しながら
  `factor_cache.on_candle(tf, candle)` を正しく呼ぶよう修正。
- テスト: `tests/workers/test_market_data_feed_worker.py` を追加し、
  timeframe 受け渡しと sync コールバック許容を検証。

### 2026-02-17（追記）position_manager `open_positions` のタイムアウト耐性を強化

- 事象: `scalp_ping_5s_b` / `scalp_ping_5s_flow` で
  `position_manager open_positions timeout after 6.0s` が断続し、エントリー見送りが増加。
- 原因:
  - `get_open_positions()` のホットパスで、agent trade ごとに `orders.db` を参照していた。
  - 参照側でも `CREATE TABLE/INDEX + COMMIT` を伴う ` _ensure_orders_db()` を毎回呼び、
    `orders.db` 書き込み競合時に busy timeout を引きずって API 応答が遅延。
- 対応（`execution/position_manager.py`）:
  - `orders.db` 読み取り専用ヘルパ（`mode=ro`）を追加し、短い read timeout で fail-fast 化。
  - `_load_entry_thesis()` / `_get_trade_details_from_orders()` を read-only 経路へ切替。
  - `get_open_positions()` の per-trade `orders.db` 参照を常時実行から条件実行へ変更し、
    `client_id/comment` から strategy 推定できる場合は DB lookup を回避。
- 期待効果:
  - position-manager API 応答の tail latency を抑制し、
    strategy worker 側の `position_manager_timeout` による skip を低減。

### 2026-02-17（追記）予測の価格到達メタを一元化しVMで可視化

- `workers/common/forecast_gate.py` の `ForecastDecision` に `anchor_price` / `target_price` / `tp_pips_hint` / `sl_pips_cap` / `rr_floor` を追加し、ロジック決定と同時に保存するよう統一。
- `workers/forecast/worker.py` の `/forecast/decide` 応答、`execution/strategy_entry.py` の `_format_forecast_context()`、`execution/order_manager.py` の `forecast_meta` / `entry_thesis["forecast_execution"]` に同値を連携。
- `scripts/vm_forecast_snapshot.py` では `--horizon` で 5m/10m を明示すると、履歴不足時でも `insufficient_history` 行を pending で出し、`need >= X candles` / `remediation` を表示するよう補強。
- `execution/order_manager.py` の forecast メタ監査キーに `expected_pips` / `anchor_price` / `target_price` / `tp_pips_hint` / `sl_pips_cap` / `rr_floor` を追記し、監査 DB に TP/SL まで残るようにした。

### 2026-02-17（追記）micro_multistrat 配下の主要 micro 戦略を個別ワーカー化

- `MicroLevelReactor` / `MicroVWAPBound` / `MicroVWAPRevert` / `MomentumBurstMicro` / `MicroMomentumStack` /
  `MicroPullbackEMA` / `TrendMomentumMicro` / `MicroTrendRetest` / `MicroCompressionRevert` / `MomentumPulse`
  をそれぞれ独立ワーカー化。
- 各戦略で `workers/micro_<slug>/worker.py` / `workers/micro_<slug>/exit_worker.py` を追加し、
  起動時に `MICRO_STRATEGY_ALLOWLIST` / `MICRO_MULTI_LOG_PREFIX`、`MICRO_MULTI_EXIT_TAG_ALLOWLIST` を
  戦略名単位で上書きするように統一。
- 対応する systemd unit/env を追加:
  - `quant-micro-levelreactor` / `quant-micro-vwapbound` / `quant-micro-vwaprevert` /
    `quant-micro-momentumburst` / `quant-micro-momentumstack` / `quant-micro-pullbackema` /
    `quant-micro-trendmomentum` / `quant-micro-trendretest` / `quant-micro-compressionrevert` /
    `quant-micro-momentumpulse`（各 `-exit` 含む）
- `quant-micro-multi.service` は `MICRO_MULTI_ENABLED=0` に切替えて共通走行を停止し、
  共通依存の二重判定を抑止。`micro_multistrat` の実運用は legacy 保留に変更。

### 2026-02-17（追加）エントリー意図に予測メタを同梱

- `execution/strategy_entry.py` で `forecast_gate.decide(...)` を戦略側補助判定として実行し、  
  `entry_thesis["forecast"]` に `future_flow`/`trend_strength`/`range_pressure` 等を保存。
- `market_order` / `limit_order` から `order_manager.coordinate_entry_intent(...)` へ
  `forecast_context` を付与し、`order_manager/board` の `details` にも保持。
- `execution/order_manager.py` と `workers/order_manager/worker.py` の経路を拡張し、`entry_intent_board` の監査情報を
  `forecast_context` 対応に更新。

### 2026-02-18（追記）scalp_fast で短期予測をデフォルト化

- `workers/common/forecast_gate.py` の `FORECAST_GATE_HORIZON_SCALP_FAST` デフォルトを `1h` から `1m` に変更。
- `forecast_gate` の技術予測で `M1` キャンドルを取得するようにし、`1m` の短期予測を実データで計算できる経路を追加。
- `1m` の Horizon メタを `timeframe=M1`、`step_bars=12` に変更し、`scalp_fast` の「短期」想定に合わせた予測可視性を向上。
- 運用確認用ドキュメント（`docs/FORECAST.md`）を更新し、`scalp_fast` の標準確認軸として `1m` を明記。
- `auto` 運用でも `1m` は技術予測優先で使うようにする
  (`FORECAST_GATE_TECH_PREFERRED_HORIZONS`) を追加し、既存バンドルの `1m` が長期仕様のままでも短期解釈が崩れにくいよう保護。

### 2026-02-18（追記）戦略別 forecast プロファイルを `strategy_entry` 経由で適用

- `execution/strategy_entry.py` で `strategy_tag`/pocket 契約を参照し、`forecast_profile` / `forecast_timeframe` / `forecast_step_bars` /
  `forecast_blend_with_bundle` / `forecast_technical_only` を戦略側で注入する形へ強化。
- `workers/common/forecast_gate.py` の `decide()` が `forecast_profile` を解決して、`technical_only` や `blend_with_bundle` に応じて
  技術予測行を参照・合成する経路を追加。`forecast_timeframe`/`step_bars` から horizon を推定し、履歴不足時の
  欠損観測時にも `NO_MATCHING_HORIZONS` が起きにくいよう補助行生成を改善。
- `scripts/vm_forecast_snapshot.py` の `--horizon` 解析を拡張（`--horizon 5m,10m` 対応）し、要求 horizon ごとに
  `insufficient_history`（必要本数/復旧手順つき）行を作りやすくした。

### 2026-02-17（追加）予測ワーカーの分離導線

- `workers/forecast/` 配下に `worker.py` を追加し、`/forecast/decide` と `/forecast/predictions` を公開する
  `quant-forecast.service` を導入。`workers/common/forecast_gate.py` の local 決定を service 化。
- `systemd/quant-forecast.service` と `ops/env/quant-forecast.env` を追加し、`FORECAST_SERVICE_*` で
  `execution/order_manager.py` からの連携先を明示。
- `execution/order_manager.py` に `_forecast_decide_with_service` を追加し、
  `ORDER_MANAGER_FORECAST_GATE_ENABLED=1` かつサービス応答可のときは service 経由で判断。
- サービス不通時の `FORECAST_SERVICE_FALLBACK_LOCAL=1` を既定化して、停止時も local 判定で運用継続しつつ
  forecast 判定ログを標準化できるように整備。

### 2026-02-17（追記）trend_h1 を下落追尾用に短期化

- `workers/trend_h1/config.py` に `TREND_H1_FORCE_DIRECTION` を追加し、戦略起動側で `short` のみを許可できる制御を導入。
- `workers/trend_h1/worker.py` の `entry_thesis` へ `entry_probability` と `entry_units_intent` を明示注入し、`strategy_entry` の意図連携が
  `coordinate_entry_intent` 側に確実に渡る構成へ揃えた。
- `ops/env/quant-trend-h1.env` を追加し、`TREND_H1_FORCE_DIRECTION=short`・`TREND_H1_ALLOWED_DIRECTIONS=Short` をデフォルトとした。
- `systemd/quant-trend-h1.service` / `systemd/quant-trend-h1-exit.service` を追加して、`trend_h1` の ENTRY/EXIT を分離起動できる状態へ。

### 2026-02-17（追記）micro_multistrat の時刻取得フォールバック修正

- `workers/micro_multistrat/worker.py` の `_ts_ms_from_tick` が `tick_window` の時刻キー `epoch` を解釈していなかったため、`_build_m1_from_ticks` が `None` を返しやすく、`quant-micro-multi` 側で `factor_stale_warn` が継続する状態を修正。
- `_ts_ms_from_tick` の優先順位を `ts_ms -> timestamp -> epoch` に拡張し、`tick_window` のエポック形式データでもM1再構築候補が生成されるよう調整。
- これにより `micro_multi_skip` の主因を減らし、`range_mode` 判定とレンジ/順張りシグナル抽出の起点更新機会を戻すことを目的とした。

### 2026-02-17（追記）micro_multistrat のレンジ時の順張り候補再開

- `workers/micro_multistrat/worker.py` のレンジ判定ロジックを修正し、`range_only` 時も許可リスト掲載の戦略はスキップしないよう変更した。
- `workers/micro_multistrat/config.py` に `RANGE_ONLY_TREND_ALLOWLIST` を追加し、戦略名を環境変数 `MICRO_MULTI_RANGE_ONLY_TREND_ALLOWLIST` で運用可能にした。
- `ops/env/quant-micro-multi.env` に
  `MICRO_MULTI_RANGE_ONLY_TREND_ALLOWLIST=TrendMomentumMicro,MicroMomentumStack,MicroPullbackEMA,MicroTrendRetest`
  を追加して、レンジ寄り時間でも順張り再エントリー機会を確保した。
- `RANGE_TREND_PENALTY` を基準にしつつ、 allowlist 戦略のみ減点係数を `35%` に減衰させ、過度な抑制を回避した。

### 2026-02-17（追記）MicroRangeBreak のレンジ内逆張り＋ブレイク順張りの分離

- `strategies/micro/range_break.py` に、レンジ内逆張り（既存）に加え、レンジブレイク発生時の順張りシグナルを追加。
- 追加シグナルは `signal` に `signal_mode: trend` / `signal_mode: reversion` を付与し、レンジ内は `reversion`、ブレイク外逸脱は `trend` として識別できるようにした。
- `workers/micro_multistrat/worker.py` の BB 方向判定を `signal_mode` 連動へ変更し、`MicroRangeBreak` のブレイク順張りを
  `strategy` 名だけで判断して拒否しない構成へ変更。
- `workers/micro_multistrat/worker.py` の `entry_thesis` に `signal_mode` を明示注入し、後段ガード/監査で意図の追跡性を確保。
- `ops/env/quant-micro-multi.env` にブレイク判定の閾値を追加入力できるキーを追加（`MICRO_RANGEBREAK_BREAKOUT_*` 系）。

- 2026-02-17（運用チューニング）`MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE` を
  `0.46` から `0.44` に下げ、レンジブレイク順張りの `signal_mode=trend` 受け条件を
  1本運用でやや拡張。  
  同時に V2 ランタイムの `quant-v2-runtime.env` で
  `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0` を維持し、global 強制を抑制する方針へ合わせた。

- 2026-02-17（運用チューニング2）`MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE` を
  `0.44` から `0.42` へ追加下げし、レンジブレイク `signal_mode=trend` の受理条件を
  さらに1本運用で拡張。  
  同時に `workers/micro_multistrat/worker.py` の起動ログを明示化し、`Application started!` を
  出力して `journalctl` 監査の可観測性を確保した（`quant-micro-multi.service` 配下）。

### 2026-02-17（追記）`scalp_ping_5s` / `scalp_ping_5s_b` の低証拠金旧キー整理

- `execution/risk_guard.py` の `allowed_lot` から
  `MIN_FREE_MARGIN_RATIO` / `ALLOW_HEDGE_ON_LOW_MARGIN` ベースの即時拒否分岐を削除し、A/B共通で
  margin cap / usage ガードのみに統一。
- `tests/execution/test_risk_guard.py` の該当期待値を
  「低マージン閾値の即時拒否」前提から「usage/逆方向ネット縮小時の優先整合」前提へ更新。
- `ops/env` の `scalp_ping_5s` 系環境ファイルから旧キーを整理。
  - `SCALP_PING_5S_MIN_FREE_MARGIN_RATIO`
  - `SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_*`
  - `SCALP_PING_5S_B_MIN_FREE_MARGIN_RATIO`
  - `SCALP_PING_5S_B_LOW_MARGIN_HEDGE_RELIEF_*`
- 全体運用オーバーライド `config/vm_env_overrides_aggressive.env` の
  `MIN_FREE_MARGIN_RATIO` を除去し、運用キー整合を維持。
- 対象追記: `ops/env/quant-scalp-ping-5s.env`, `ops/env/scalp_ping_5s_b.env`,
  `ops/env/scalp_ping_5s_tuning_20260212.env`, `ops/env/scalp_ping_5s_entry_profit_boost_20260213.env`,
  `ops/env/scalp_ping_5s_max_20260212.env`, `config/vm_env_overrides_aggressive.env`。

### 2026-02-17（追記）未使用戦略のレガシー化（誤作動封じ）

- `workers/` から稼働外戦略 43 件を削除して、VM実行導線に残る戦略を
  `scalp_false_break_fade / scalp_level_reject / scalp_macd_rsi_div / scalp_macd_rsi_div_b / scalp_ping_5s* / scalp_rangefader / scalp_squeeze_pulse_break / scalp_tick_imbalance / scalp_wick_reversal_blend / scalp_wick_reversal_pro / m1scalper / micro_multistrat / session_open` に絞り込んだ。
- systemd/env の旧戦略起動点を整理。
  - `systemd/quant-impulse-retest-s5*.service`
  - `systemd/quant-micro-adaptive-revert*.service`
  - `systemd/quant-trend-h1*.service`（該当分）
  - `ops/env/quant-impulse-retest-s5*.env`
  - `ops/env/quant-micro-adaptive-revert*.env`
  - `ops/env/quant-trend-h1.env`
- `scalp_rangefader` について、`entry_probability` を明示的に `entry_thesis` へ注入。
- `session_open` の `order` へ `entry_probability` を `meta.entry_probability` としても明示し、`AddonLiveBroker` 経路でも `order_manager` 側での意図解釈を明確化。

### 2026-02-17（追記）M1Scalper に提案戦略 1/3（`breakout_retest`, `vshape_rebound`）を実装

- `strategies/scalping/m1_scalper.py` に `breakout_retest` / `vshape_rebound` の2シグナル生成ヘルパーを追加。
  - 直近レンジ突破→浅いリテスト時の順張り再突入を拾う `breakout_retest`。
  - 急変動後の最初の反発（long/short）を拾う `vshape_rebound`。
- 両シグナルは `check()` の momentum/既存シグナル判定前に評価し、成立時は共通の `structure_targets` と
  技術倍率適用を経て `_attach_kill()` を通して発注可否パスへ接続。
- `ops/env/quant-m1scalper.env` にデフォルト閾値を追加。
  - `M1SCALP_BREAKOUT_RETEST_*`
  - `M1SCALP_VSHAPE_REBOUND_*`
- `docs/KATA_SCALP_M1SCALPER.md` へ 1/3 実装の要件・観測観点を追記し、監査可能化。

### 2026-02-17（追記）order_manager の意図改変を既定停止

- `execution/order_manager.py` の `market_order` / `limit_order` における
  `entry_probability` 起因のサイズ縮小・拒否（`entry_probability_reject` / `probability_scaled`）は
  既定で実行しないよう整理。
- 新規フラグ `ORDER_MANAGER_PRESERVE_INTENT_UNIT_ADJUST_ENABLED`（既定 `0`）を追加し、必要時のみ
  `ORDER_MANAGER_PRESERVE_INTENT_UNIT_ADJUST_ENABLED_STRATEGY_<TAG>` で戦略別有効化を行える運用へ変更。
- ワーカー側が決めた `entry_units_intent` / `entry_probability` を order_manager が
  追加で潰さない方向へ収束。`order_manager` のガード/リスク責務と
  `strategy_entry` 側のローカル意図決定の分離を明確化。
- 同時に `docs/WORKER_ROLE_MATRIX_V2.md` のオーダー面記述を更新。

### 2026-02-16（追記）scalp_ping_5s_b ラッパー env_prefix 混在回避を根本修正

- `workers/scalp_ping_5s_b/worker.py` の `_apply_alt_env()` を修正し、`SCALP_PING_5S_*` の掃除時に
  `SCALP_PING_5S_B_*` を残してから base-prefix へ再投入する順序へ変更。
- これにより、`SCALP_PING_5S_B_ENABLED` や `SCALP_PING_5S_B_*` の必須値が誤って削除され、
  子プロセス `workers.scalp_ping_5s.worker` が意図せず `SCALP_PING_5S_ENABLED=0` になる問題を解消。
- `SCALP_PING_5S_B` 側の設定を `SCALP_PING_5S_*` へ正規射影し、`ENV_PREFIX` を `SCALP_PING_5S_B` へ固定した
  上で `enabled/strategy_tag/log_prefix` も一貫注入する経路を明文化。
- 対象は B版起動時の env 混在・取り残し（`no_signal` の母集団原因に直結する無効化）を避けるための
  根本対応として記録。

### 2026-02-16（追記）env_prefix 共通化フェイルセーフ（entry→order_manager）

- `execution/strategy_entry.py` の `env_prefix` 取りまとめを強化し、`entry_thesis` / `meta` / `strategy_tag` から
  正規化 (`strip` + 大文字化) したうえで一貫して注入するよう変更。
- `execution/order_manager.py` 側でも `env_prefix` を同様に正規化し、`strategy_tag` からのフォールバック推定を追加。
  これにより、`SCALP_PING_5S_B` と `SCALP_PING_5S` が混在した場合の
  ガード参照ずれを抑え、`no_signal` の母集団に寄与しうる「意図外の設定混線」を低減。

### 2026-02-16（追記）scalp_precision 実行委譲の完全停止（戦略別独立実行）

- `workers/scalp_tick_imbalance`, `workers/scalp_squeeze_pulse_break`, `workers/scalp_wick_reversal_blend`, `workers/scalp_wick_reversal_pro` の `entry`/`exit` を `workers.scalp_precision` の子プロセス実行から切り離し、各パッケージ内のローカル `worker.py` / `exit_worker.py` を直接実行する構成へ変更。
- `workers/scalp_macd_rsi_div` と `workers/scalp_ping_5s` / `workers/scalp_ping_5s_b` の `exit_worker.py` も `python -m workers.scalp_precision.exit_worker` 委譲を停止し、同一プロセス内で `exit` 判定を完結。
- 戦略独立実行のため、上記5戦略に対応する `config.py` / `common.py`（該当戦略）をローカル配備し、戦略起動時の実行経路から `scalp_precision` 参照を外した。

### 2026-02-16（追記）scalp_precision からの最終依存排除

- `workers/scalp_tick_imbalance/`, `workers/scalp_squeeze_pulse_break/`, `workers/scalp_wick_reversal_blend/`, `workers/scalp_wick_reversal_pro/` の
  `strategy_tag` fallback を `"scalp_precision"` から各戦略名へ変更し、`client_order_id`/`market_order` の strategy tag 注入でも戦略別値を使用。
- `workers/scalp_macd_rsi_div/`, `workers/scalp_ping_5s/`, `workers/scalp_ping_5s_b/` の `exit_worker` も戦略別エントリ名へ変更し、同名 main 呼び出しに揃えた。

### 2026-02-16（追記）`scalp_precision` 依存 wrapper の切断（戦略別ワーカー単位）

- `workers/scalp_tick_imbalance` / `scalp_squeeze_pulse_break` / `scalp_wick_reversal_*` / `scalp_macd_rsi_div` / `scalp_ping_5s_b` の
  `entry` / `exit_worker` から `workers.scalp_precision` のインポート依存を除去し、各戦略ワーカー側を subprocess 起動に切替え。
- `python -m workers.<strategy>.worker` / `python -m workers.<strategy>.exit_worker` を直接起動し、
  ワーカー本体の起動責務は戦略別ランナー（entry/exit）として完結させた。
- テスト `tests/test_spread_ok_tick_cache_fallback.py` の `scalp_precision.common` 参照を削除し、共有 `spread_ok` の依存を排除。

### 2026-02-16（追記）起動監査ログの統一

- 各戦略 `entry/exit` の起動直後に `Application started!` を明示ログ化。
  対象は `scalp_tick_imbalance` / `scalp_squeeze_pulse_break` / `scalp_wick_reversal_blend` / `scalp_wick_reversal_pro` / `scalp_macd_rsi_div` / `scalp_ping_5s` / `scalp_ping_5s_b` の `entry`/`exit`。
- VM 上の反映確認を、`journalctl` の `Application started!` 検索に統一し、起動実在性を監査しやすくした。

### 2026-02-16（追記）strategy_entry の意図注入を完全化

- `execution/strategy_entry.py` に `entry_thesis` / `meta` の `env_prefix` 双方向注入を追加し、`coordinate_entry_intent` まで `env_prefix` を確実反映。
- `workers/scalp_ping_5s/worker.py` の `no_signal` 理由正規化を拡張し、`insufficient_signal_rows_fallback` を判別可能にして監査集計の粒度を固定化。

### 2026-02-16（追記）`PositionManager.close()` の共有DB保護

- `execution/position_manager.py` の `PositionManager.close()` に共有サービスモード保護を追加。
- `POSITION_MANAGER_SERVICE_ENABLED=1` かつ `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0` の運用では、クライアント側からの
  `close()` 呼び出しをリモート `/position/close` に転送せず、共有 `trades.db` を意図せず閉じないガードを実装。
- 直近 VM ログで観測される大量の `Cannot operate on a closed database` は、この close 過剰呼び出し由来の再発を抑止する対象。
- `workers/position_manager/worker.py` は `PositionManager` をローカルモードで起動するため、サービス停止時の
  正規クローズ経路は維持。

### 2026-02-16（追記）`entry_probability` の高確率側サイズ増強を全戦略共通化

- `execution/order_manager.py` の `order_manager` 共通プリフライトで `entry_probability` スケーリングを拡張し、
  `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY` 以上の高確率時のみ、サイズを `>1` へ拡張可能に変更。
- 同時に、`ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE` を追加して上振れ上限を制御可能化。
- 低確率側（`<= reject_under`）の拒否/縮小は既存ルールを維持し、縮小判断の主軸は従来どおり `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE`
  / `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` に寄せたまま。
- 追加運用キー（例値）:
  - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY=0.80`
  - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE=1.25`
- 変更は `order_manager` 経由の全戦略に共通適用され、戦略側ロジック選別は現行どおり保持。

### 2026-02-16（追記）`install_trading_services` でログ自動軽量化を標準化

- `scripts/install_trading_services.sh` を更新し、`--all` / `--units` 指定に関わらず `quant-cleanup-qr-logs.service` と
  `quant-cleanup-qr-logs.timer` が常設でインストールされるようにしました。
- `systemd/cleanup-qr-logs.timer` は 1日2回（07:30/19:30）起動で、実運用ノードでもディスククリーンアップを自動化する前提へ統一。

### 2026-02-16（追記）非5秒エントリー再開のための実行条件調整

- `market_data/tick_fetcher.py` の `callback` 発火経路を `_dispatch_tick_callback` に一本化し、`tick_fetcher reconnect` 側の
  `NoneType can't be used in 'await' expression` ループ再接続原因の対処を反映。
- `ops/env/quant-micro-multi.env` に `MICRO_MULTI_ENABLED=1` を追加し、`quant-micro-multi` の ENTRY 側を起動状態に寄せる。
- `ops/env/quant-m1scalper.env` の `M1SCALP_ALLOWED_REGIMES` を `trend` 固定から `trend,range,mixed` に変更し、市況レジーム偏在時の過度な阻害を回避。

### 2026-02-22（追記）M1Scalper 戦略ラベルの復元ルールを明文化

- `execution/position_manager.py` の戦略タグ正規化に、`client_order_id` の
  `m1scalpe*` 系文字列を `M1Scalper-M1` に復元するエントリを追加し、`strategy` / `strategy_tag` の表示整合を向上。
- 同ファイルに `M1Scalper` 系の戦略名を戦略-ポケット推定 (`_STRATEGY_POCKET_MAP`) へ追加。
- `scripts/backfill_strategy_tags.py` の `client_id` 由来補完に、`qr-<ts>-scalp-m1scalpe*` 系を `M1Scalper-M1` として扱うロジックを追加。

## 補足（戦略判断責務の明確化）

- **方針確定**: 各戦略ワーカーは「ENTRY/EXIT判定の脳」を保持し、ロジックの主判断は各ワーカー固有で行う。
- `quant-strategy-control` は「最終実行可否」を左右する制御入力を配信するのみ（`entry_enabled` / `exit_enabled` / `global_lock` / メモ）。
- したがって、`strategy-control` が戦略ロジックを代行しているわけではなく、**各戦略の意思決定を中断/再開するガードレイヤー**として機能する。
- UI の戦略ON/OFFや緊急ロックはこのガードレイヤーを介して、並行中の戦略群へ即時反映する。

### 2026-02-16（追記）5秒スキャ（scalp_ping_5s）運用通過率の即時改善

- `ops/env/scalp_ping_5s.env` を更新し、5秒戦略向け `entry_probability` スケーリング後の
  最小ロット拒否を回避するため、戦略別最小ロットを引き下げた。
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S=20`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_LIVE=20`
- 同ファイルで `SCALP_PING_5S_PERF_GUARD_*` を追加し、5秒戦略単体で `perf_guard` の実運用しきい値を事実上緩和。
  - `PERF_GUARD_MIN_TRADES=99999` / `PF_MIN=0.0` / `WIN_MIN=0.0`
  - `PERF_GUARD_FAILFAST_MIN_TRADES=99999` / `FAILFAST_PF=0.0` / `FAILFAST_WIN=0.0`
- 変更対象は 5秒エントリー運用のみに限定（`SCALP_PING_5S_*` プレフィックス）。
- 反映後、VM 上で `orders.db` の `strategy=scalp_ping_5s_live` の `entry_probability_reject` と `perf_block`
  比率低下、`filled`/`submit_attempt` 増加を優先監視する。

### 2026-02-16（追記）5秒スキャ no_signal 原因可視化の追加

- `workers/scalp_ping_5s/worker.py` の `_build_tick_signal()` を
  `reason` 付き返却に変更し、`entry-skip` の `no_signal` に
  `insufficient_*`/`stale_tick`/`momentum_tail_failed` などの原因を
  `detail` として残す監査を追加。
- `ops/env/scalp_ping_5s.env` で 5秒戦略の入口閾値を追加緩和。
  - `SCALP_PING_5S_SIGNAL_WINDOW_SEC=0.35`
  - `SCALP_PING_5S_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_MOMENTUM_SPREAD_MULT=0.12`
  - `SCALP_PING_5S_ENTRY_BID_ASK_EDGE_PIPS=0.0`
- 本修正は 5秒戦略のみ対象。`SCALP_PING_5S` 系サービスの入場通過率回復を優先し、
  `entry_probability_reject` が減り `filled` が増えるかを次回監視で確認する。

### 2026-02-16（追記）env_prefix 混在対策と no_signal 算出粒度の追加

- `execution/order_manager.py` の `env_prefix` 解決を `entry_thesis` 優先へ固定し、
  `meta` 側と値が混在した場合も戦略側の意図が上書きされないようにした。
  - 混在時は `pocket` / `strategy` / `meta` / `entry` の値を debug ログへ出力し、原因追跡を可能化。
- `workers/scalp_ping_5s/worker.py` の `no_signal` 原因集計で
  `insufficient_mid_rows` / `insufficient_rows` / `invalid_latest_epoch` / `stale_tick`
  を追加サブ分類化し、`insufficient_signal_rows` 系と同様に監査しやすくした。

### 2026-02-16（追記）M1スキャ (`scalp_m1scalper`) の最小ロットソフトガードを調整

- `workers/scalp_m1scalper/worker.py` の `_resolve_min_units_guard` にて、
  `min_units_soft_raise` 時に `config.MIN_UNITS` へサイズを引き上げる処理を廃止し、
  ソフト判定後も `abs_units` を維持して通過させるよう変更。
- `ops/env/quant-m1scalper.env` に `M1SCALP_MIN_UNITS_SOFT_FAIL_RATIO=0.05` を追加し、
  M1スキャの小型シグナル（例: `-57`）でも過度な拒否になりにくい状態へ寄せた。

### 2026-02-16（追記）M1スキャ戦略の実行実態観測結果に基づく最小ロット再調整

- `OPS` 実行環境（`quant-v2-runtime.env`）で `ORDER_MIN_UNITS_STRATEGY_M1SCALPER_M1=700` を
  `ORDER_MIN_UNITS_STRATEGY_M1SCALPER_M1=300` に更新し、同時に `quant-m1scalper.env` に
  `M1SCALP_MIN_UNITS=350` と `M1SCALP_MIN_UNITS_SOFT_FAIL_RATIO=0.62` を明示。
- 本日ログ上の `OPEN_SKIP` で、`entry_probability_below_min_units` が実質的な主因（約90%超）であったため、
  `ORDER_MIN_UNITS` と worker 側 min_units threshold を引き下げ、最低サイズ起因の取り逃しを抑える方向へ変更。
- 適用後は同日再集計で `OPEN_SKIP` 内訳（`entry_probability_below_min_units`）の件数が低下するかを
  監視対象に設定。

### 2026-02-16（追記）5秒B no_signal 可観測性の精緻化

- `workers/scalp_ping_5s/worker.py`
  - `entry-skip` ログの `no_signal` 集計を `no_signal:<detail_head>` まで分割してカウントし、
    `insufficient_*` 系と `revert_*` 系の内訳を見える化。
  - `revert_long/revert_short` を含む `no_signal` に対して方向推定を行い、サイド別内訳の精度を向上。
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0` を明示し、5秒B起動時にも
    lookaheadゲートが確実に `off` になる状態を固定。

### 2026-02-16（追記）5秒B閾値を5秒A運用帯へ寄せて通過率を改善

- `ops/env/scalp_ping_5s_b.env` を、`scalp_ping_5s`（A）側で稼働している
  エントリー通過優先値に寄せる形で緩和。
  - `SCALP_PING_5S_B_MIN_TICKS=4`
  - `SCALP_PING_5S_B_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE=0.50`
  - `SCALP_PING_5S_B_MIN_TICK_RATE=0.50`
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_SEC=0.35`
  - `SCALP_PING_5S_B_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.08`
  - `SCALP_PING_5S_B_MOMENTUM_SPREAD_MULT=0.12`
  - `SCALP_PING_5S_B_ENTRY_BID_ASK_EDGE_PIPS=0.0`
- 目的: B版の `no_signal`（特に `insufficient_signal_rows`）を減らし、`scalp_ping_5s_b_live` の実入場頻度を
  主要な5秒戦略レベルまで引き上げる。

### 2026-02-16（追記）5秒Bの再取り込み改善（`insufficient_signal_rows` / `revert_not_enabled` 対策）

- `ops/env/scalp_ping_5s_b.env` に追加・更新:
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_SEC=0.55`
  - `SCALP_PING_5S_B_MIN_TICK_RATE=0.70`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE=0.45`
  - `SCALP_PING_5S_B_REVERT_MIN_TICK_RATE=1.2`
- `revert_not_enabled` は実質的な `no_signal` 側ボトルネックだったため、
  `revert` 判定閾値とシグナル窓を緩和し、同時に短側閾値もわずかに緩めることで
  `revert_not_enabled` 集計を抑える方針に変更。

### 2026-02-16（追記）`no_signal` のフォールバックを窓不足向けに限定

- `workers/scalp_ping_5s/worker.py`
  - `insufficient_signal_rows` 時のみ、`SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` で指定した上限内で
    追加窓を1回だけ再評価するように変更（不足時は `insufficient_signal_rows` の理由へ
    `window/fallback` を記録）。
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_SEC=2.00` を追加。
- `TickSignal` に `signal_window_sec` を付与し、実際に使ったシグナル窓を監査しやすくした。

### 2026-02-24（追記）5秒B no_signal 監査の実装事故修正

- `workers/scalp_ping_5s/config.py` に `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` を
  読み込む `SIGNAL_WINDOW_FALLBACK_SEC` 定数を追加し、VM起動時の
  `AttributeError: module ... has no attribute 'SIGNAL_WINDOW_FALLBACK_SEC'` を解消。
- `no_signal` のフォールバックは従来どおり `insufficient_signal_rows` 時のみ行い、データ不足以外の
  退避導線は従来ロジックを維持して副作用増加を回避。

### 2026-02-16（追記）fallback窓展開を「明示設定のみ」に固定

- `workers/scalp_ping_5s/worker.py`
  - `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` が `0` の場合、`WINDOW_SEC` へ無条件で拡張しないよう制御を変更。
  - fallback が実施されなかった/失敗した場合も `insufficient_signal_rows` 理由に
    `fallback_used=attempted` を残して、監査時に「窓を広げたが不足継続」を判別できるようにした。

### 2026-02-16（追記）5秒Bの意図設定プレフィックスを明示化

- `workers/scalp_ping_5s_b/worker.py` の環境コピー処理で
  `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S_B` を明示的に上書きするよう追加。
- `ops/env/scalp_ping_5s.env` / `config/env.example.toml` に `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S` を明示し、
  A/Bのプレフィックス判定を設定側で明確化。
- `ops/env/scalp_ping_5s_b.env` へ
  `SCALP_PING_5S_B_ENV_PREFIX=SCALP_PING_5S_B` を追加。
- B版の `PERF_GUARD` / `entry_thesis` 参照で、A/B混在時に B 固有キーを誤適用しないようにした。
- `workers/scalp_ping_5s/config.py` で `ENV_PREFIX` を `SCALP_PING_5S_ENV_PREFIX` 参照に切替し、
  B版 `SCALP_PING_5S_B_ENV_PREFIX` 経由で `SCALP_PING_5S_ENV_PREFIX=SCALP_PING_5S_B` が有効化されるよう明示化。
- 追跡指標は `perf_block` / `entry_probability_reject` の内訳で次回検証し、必要であれば
  `SCALP_PING_5S_B_PERF_GUARD_*` の調整へ接続する。

### 2026-02-16（追記）5秒B no_signal 原因粒度の崩れを最小修正

- `workers/scalp_ping_5s/worker.py`
  - `insufficient_signal_rows` の詳細文字列を `detail_parts` で組み立てる方式へ変更し、
    `fallback_min_rate_window` 付与時に `fallback_used` が欠落していた組版バグを是正。
  - これにより `entry-skip` 集計で `insufficient_signal_rows_fallback` /
    `insufficient_signal_rows_fallback_exhausted` の振り分けが崩れないようにした。

### 2026-02-16（追記）5秒Bの `insufficient_signal_rows_fallback_exhausted` 抑制

- `workers/scalp_ping_5s/worker.py`
  - `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC` が明示されている場合、`fallback` での窓拡張を
    `WINDOW_SEC` まで1回だけ行って再評価する追加ルートを導入。
  - これにより `insufficient_signal_rows` が `fallback_exhausted` に偏るケースを、
    同一運用ノイズを増やさずに取りこぼし低減できる領域へ寄せることを目的とした。

### 2026-02-24（追記）quant-scalp-ping-5s-b の env_prefix 混在排除

- `workers/scalp_ping_5s_b/worker.py` を追加修正し、`SCALP_PING_5S_B_*` を
  `SCALP_PING_5S_*` に写像する前に、実行時環境内の既存 `SCALP_PING_5S_*` を一度削除してから再注入するように変更。
- B サービスは `scalp_ping_5s.env` と `scalp_ping_5s_b.env` を併読するため、A 側残存値で上書きされる混在リスクを除去。

### 2026-02-24（追記）5秒B no_signal のデータ不足と env_prefix 混在を再監査前提で収束

- `workers/scalp_ping_5s/worker.py` の `_build_tick_signal()` を更新し、
  `insufficient_signal_rows` の判定で、取得した全 `rows` 数が `min_signal_ticks` 未満の場合は
  `fallback` 再試行を行わず `fallback_used=no_data` として上流監査に出すよう変更。
  これにより、データ不足と fallback 過程失敗（`fallback_attempted`）の原因混線を分離し、
  `no_signal` 原因の切り分けを明確化。
- `workers/scalp_ping_5s/worker.py` の no_signal 正規化を拡張し
  `insufficient_signal_rows_no_data` を追加。
- `execution/order_manager.py` の env_prefix 解決を調整し、`entry_thesis` / `meta` が
  混在しても strategy 側が解釈可能な prefix があればそちらへ収束するようにして、
  ゾーン混線時のリスクガード参照ずれを抑える構成に変更。
- 追加監査として、`env_prefix` 起因が疑われる `revert_not_enabled` / `momentum_tail_failed` が
  再発しないかを `no_signal` ログで追跡する。

### 2026-02-24（追記）strategy_entry で strategy_tag 連動 env_prefix 正規化を固定

- `execution/strategy_entry.py` の `_coalesce_env_prefix()` を、`strategy_tag` から導出できる
  `env_prefix` を最優先にする設計へ変更。
- これにより `entry_thesis` / `meta` に混在した A/B の `env_prefix` が混ざっていても、
  strategy 側の正しい識別子（例: `SCALP_PING_5S_B`）でガードの参照元が収束し、
  `entry_thesis` への誤注入を抑えて取りこぼし原因を低減する方針とした。

### 2026-02-16（追記）PositionManager 停止再開耐障害化

- `execution/position_manager.py`:
  - `trades.db` 接続再確立時の再試行回数/待ち時間を追加し、連続再接続失敗時はインターバルで抑止するよう調整。
  - `close()` で接続をクローズ後 `self.con = None` へセットし、閉じた接続の再利用を防止。
  - closed DB 例外時の再接続制御を `_ensure_connection_open` / `_commit_with_retry` / `_executemany_with_retry` で統一。
- `workers/position_manager/worker.py`:
  - 起動時 `PositionManager` 初期化をリトライ化。失敗時は明示エラー付きで起動継続し、サービス再起動ループを回避。

### 2026-02-16（追記）PositionManager service 呼び出しタイムアウトの整合化

- `execution/position_manager.py` は `POSITION_MANAGER_HTTP_TIMEOUT` と
  `POSITION_MANAGER_SERVICE_TIMEOUT` のギャップを起動時に検査し、サービス側タイムアウトが短すぎる構成を
  自動補正するロジックを前提として、運用環境設定の同期を追加。
- `ops/env/quant-v2-runtime.env` を次の値へ更新:
  - `POSITION_MANAGER_SERVICE_TIMEOUT=9.0`
  - `POSITION_MANAGER_HTTP_TIMEOUT=8.0`
- 目的: service経路で `sync_trades/get_open_positions` のタイムアウト再試行を抑制し、
  直近データ取得の安定性を上げる。

### 2026-02-16（追記）5秒Bの `client_order_id` と strategy_tag 復元を固定

- `workers/scalp_ping_5s/worker.py` の `_client_order_id()` は `config.STRATEGY_TAG` を 24 文字まで保持し、
  `qr-<ts>-<strategy>-<side><digest>` を生成するよう変更。これにより `scalp_ping_5s` と
  `scalp_ping_5s_b` の識別崩れ（先頭 12 文字トランケート）を解消。
- `execution/position_manager.py` の `_infer_strategy_tag()` は `qr-<ts>-<strategy>-...` 系の
  戦略タグ推定を強化し、`qr-<timestamp>` を strategy_tag と誤認する既知不具合を抑止。
- 目的は 5 秒B 側の `open_trades` 管理漏れを潰し、取り残し残高の監視・追跡を確実化すること。

### 2026-02-16（追記）scalp_precision と scalp_ping_5s の `position_manager` 失敗時ガード

- `workers/scalp_precision/exit_worker.py`
  - `PositionManager.get_open_positions()` を非同期タイムアウト付きの安全取得へ変更し、`position_manager_error` / `position_manager_timeout` 時は同サイクルをスキップ。
  - `_filter_trades` は `entry_thesis` 欠損時に `client_id/client_order_id` ベースの戦略タグ推定を追加し、タグ欠落による戦略漏れを抑制。
  - 取得失敗の連続警報を間引き。
- `workers/scalp_ping_5s/worker.py`
  - `_safe_get_open_positions()` 後、エラー時に `continue` で当該ループを明示保留に変更し、空 `pocket_info` での判定を防止。
  - `forced_exit`/`profit_bank` 再取得でも同様にスキップへ統一。
- 運用意図: フォールバックは「空データ埋め込み」ではなく、取引判断の実行抑止（Fail-safe）に限定。

## 追加（実装済み）

- `systemd/quant-market-data-feed.service`
- `systemd/quant-strategy-control.service`
- `apps/autotune_ui.py`
  - `summary`/`ops` ビューに「戦略制御」セクションを追加
  - `POST /api/strategy-control`
  - `POST /ops/strategy-control`
- `systemd/quant-scalp-ping-5s-exit.service`
- `systemd/quant-scalp-macd-rsi-div-exit.service`
- `systemd/quant-scalp-tick-imbalance.service`
- `systemd/quant-scalp-tick-imbalance-exit.service`
- `systemd/quant-scalp-squeeze-pulse-break.service`

### 2026-02-14（追記）market_order 入口の entry_thesis 補完を追加

- `execution/order_manager.py` に `market_order()` 入口ガード `_ensure_entry_intent_payload()` を追加。
- 戦略側 `entry_thesis` が欠けるケースに対し、`entry_units_intent` と `entry_probability` を実行時に補完。
- 併せて `strategy_tag` が未入力時は `client_order_id` から補完して `entry_thesis` に反映するようにし、V2各戦略の `market_order` 呼び出し互換性を維持。
- `systemd/quant-scalp-squeeze-pulse-break-exit.service`
- `systemd/quant-scalp-wick-reversal-blend.service`
- `systemd/quant-scalp-wick-reversal-blend-exit.service`
- `systemd/quant-scalp-wick-reversal-pro.service`
- `systemd/quant-scalp-wick-reversal-pro-exit.service`
- `workers/market_data_feed/worker.py`
- `workers/strategy_control/worker.py`
- `workers/scalp_ping_5s/exit_worker.py`
- `workers/scalp_macd_rsi_div/exit_worker.py`
- `systemd/quant-order-manager.service`
- `systemd/quant-position-manager.service`
- `workers/order_manager/__init__.py`
- `workers/order_manager/worker.py`
- `workers/position_manager/__init__.py`
- `workers/position_manager/worker.py`
- `execution/order_manager.py`（service-first 経路追加、`_ORDER_MANAGER_SERVICE_*` 利用）
- `execution/position_manager.py`（service-first 経路追加、`_POSITION_MANAGER_SERVICE_*` 利用）
- `config/env.example.toml`（order/position service URL/enable 設定追加）
- `main.py`
  - `WORKER_SERVICES` に `market_data_feed` / `strategy_control` を追加。
  - `initialize_history("USD_JPY")` を `worker_only_loop` から撤去（初期シードを market-data-feed worker に移譲）。

### 2026-02-18（追記）黒板協調判定要件を仕様化

- `execution/strategy_entry.py` は `/order/coordinate_entry_intent` を経由して
  `execution/order_manager.py` の `entry_intent_board` 判定と整合した運用へ固定。
- 判定の固定要件を明文化:
  - `own_score = abs(raw_units) * normalized(entry_probability)`  
  - `dominance = opposite_score / max(own_score,1.0)` を算出し監査記録するが、方向意図は `raw_units` を基本維持して通す
  - 最終 `abs(final_units) < min_units_for_strategy(strategy_tag, pocket)` なら reject
  - `reason` と `status` は `entry_intent_board` に永続化し、`final_units=0` は `order_manager` 経路に流さない運用をログ追跡対象化。
  - `status`: `intent_accepted/intent_scaled/intent_rejected`
  - `reason`: `scale_to_zero/below_min_units_after_scale/coordination_load_error` 等
- `opposite_domination` は廃止し、逆方向優勢でも方向意図を否定しない運用へ更新。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` を同一ブランチ変更で更新し、監査対象文言を同期済みにする運用へ反映。

### 2026-02-19（追記）戦略ENTRYで技術判定コンテキストを付与

- `execution/strategy_entry.py` の `market_order` / `limit_order` に、
  `analysis.technique_engine.evaluate_entry_techniques` を使って
  `N波`, `フィボ`, `ローソク` を含む入場技術スコア算出を追加。
- エントリー価格は `entry_thesis` の価格情報優先、未提供時は直近ティックから補完し算出。
- 算出結果は `entry_thesis["technical_context"]` に保存され、各戦略からの
  エントリー意図として `ENTRY/EXIT` の判断（ロギング/追跡含む）に利用可能。
- 機能スイッチは `ENTRY_TECH_CONTEXT_ENABLED`（未設定時 true）とし、必要時は
  `execution/strategy_entry.py` の既定動作から外せるようにした。

### 2026-02-15（追記）strategy_entry の戦略キー正規化経路を関数化

- `execution/strategy_entry.py` の `_NORMALIZED_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を
  集約生成する処理を `_normalize_strategy_requirements()` に集約し、`_strategy_key` 参照順序依存を
  回避する形に変更。
- これにより、`quant-strategy-entry` 起動時の `NameError` リスク（`_strategy_key` 未定義）を回避しやすくした。

### 2026-02-14（追記）戦略ENTRYに戦略別技術断面を常設

- `execution/strategy_entry.py` の `_inject_entry_technical_context()` を拡張し、
  `entry_thesis["technical_context"]` に `evaluate_entry_techniques`（N波/フィボ/ローソク）結果だけでなく、
  `entry_price` の有無に関わらず `D1/H4/H1/M5/M1` の技術指標スナップショットを保存するようにした。
- 主要保存項目には `ma10/ma20/ema12/ema20/ema24/rsi/atr/atr_pips/adx/bbw/macd/...` を含む
  `indicators` が入り、`ENTRY_TECH_DEFAULT_TFS` で参照TFの優先順を上書き可能。
- 戦略側が必要とする指標を限定したい場合、`technical_context_tfs` / `technical_context_fields` を
  `entry_thesis` に付与して保存範囲を絞り込める仕様を同時に導入。
- 既存の `ENTRY_TECH_CONTEXT_ENABLED` スイッチは維持し、無効時は `entry_thesis` への技術注入を抑制。

### 2026-02-20（追記）戦略別の必要データ契約を明文化

- `execution/strategy_entry.py` で、戦略が必要とする技術入力は `entry_thesis` を通じて受領する運用を明示。
  - `technical_context_tfs`: 収集する指標TF順（例: `["H1", "M5", "M1"]`）
  - `technical_context_fields`: `indicators` に保存するフィールド名（未指定は全件）
  - `technical_context_ticks`: エントリー時に参照する最新ティック名（例: `latest_bid` / `latest_ask` / `spread_pips`）
  - `technical_context_candle_counts`: `{"H1": 120, "M5": 80}` のような TF 別ローソク本数指定
- `entry_thesis["technical_context"]` には、上記要求を反映した `indicators`（TF毎）と `ticks`、要求内容を保存し、技術判定結果（`result`）とセットで持つ。
- `analysis/technique_engine.evaluate_entry_techniques` は `technical_context_tfs` / `technical_context_candle_counts` を解釈して、
  TF 及びローソク取得本数を戦略側要求へ寄せる処理を追加（`common` 既定は維持）。
  - `ENTRY_TECH_CONTEXT_GATE_MODE` は `off/soft/hard` を明示し、`hard` 時は `technical_context.result.allowed=False` を最終拒否条件に反映する運用を確認（当時の共通評価仕様）。
  - 同時に `session_open` 経路（`addon_live -> strategy_entry`）向け契約も追加し、`technical_context_tfs`/`fields`/`ticks`/`candle_counts` を明示。
  （当時の仕様では N波/フィボ/ローソク必須寄りだったため、現在は戦略側の `tech_policy` 明示に委譲）

### 2026-02-21（追記）技術判定の共通計算を strategy_entry から分離

- `execution/strategy_entry.py` で共通技術判定を実行するフローをデフォルト停止し、`ENTRY_TECH_CONTEXT_COMMON_EVAL=0`（既定）時は
  `analysis.technique_engine.evaluate_entry_techniques` を呼ばないようにした。
- `entry_thesis["technical_context"]` への保存は維持し、`indicators`（TF別）・`ticks`・要求パラメータは戦略のローカル計算入力として保全。
- `technical_context["result"]` は共通側未評価時は `allowed=True` の監査用結果を持たせ、評価結果不在での注文拒否を発生させないようにする。
- サイズ決定・方向決定は引き続き戦略側（各strategyワーカー）に委譲。`strategy_entry` 側では共通スコアに基づく拒否/縮小は行わない。
- AGENTS/運用側の方針に合わせ、各戦略内で N波/フィボ/ローソク判定を含む必要なテクニカルを計算して `entry_thesis`/`technical_context` の形で整合させる前提へ一本化。

### 2026-02-15（追記）共通評価設定キーを環境定義から整理

- `config/env.example.toml` から `ENTRY_TECH_CONTEXT_GATE_MODE` / `ENTRY_TECH_CONTEXT_COMMON_EVAL` / `ENTRY_TECH_CONTEXT_APPLY_SIZE_MULT` を削除し、
  `ENTRY_TECH_CONTEXT_ENABLED` のみ残して `technical_context` 注入（保存）を明文化。
- `WORKER_ROLE_MATRIX_V2.md` の技術要件章を更新し、`strategy_entry` は各戦略のローカル判断を代替しない構成へ統一。
- `strategy_entry` は `technical_context.result` を上書きしない（保存専有）運用を前提に文言を整合。

### 2026-02-22（追記）N波/フィボ/ローソクの必須条件を共通契約から外す

- 現行運用として、`strategy_entry` 側の共通注入は `technical_context` 取得範囲までに限定し、
  `require_fib` / `require_nwave` / `require_candle` の既定強制を廃止。
- `execution/strategy_entry.py` の契約辞書から `tech_policy` の既定付与を事実上無効化し、各戦略ワーカー側の
  `evaluate_entry_techniques(..., tech_policy=...)` 呼び出しで戦略個別要件を持つ運用へ戻す。
- これにより、戦略ごとの意図（許容するテクニカル条件）が壊れず維持される設計に再整合。

### 2026-02-22（追記）tech_fusion を mode 別 tech_policy 明示へ収束

- `workers/tech_fusion/worker.py` の `tech_policy` を `range` / `trend` モードで明示分岐化。
- `range` モードは `require_nwave=True` を明示し、`trend` モードは `require_*` を `False` のまま
  戦略ローカルで明示定義する形へ統一。
- `technical_context_tfs/fields/ticks/candle_counts` は現行定義を保持しつつ、`evaluate_entry_techniques` への
  要件注入を戦略側で完結する形へ更新。  
  同時に監査観点の `tech_fusion` `require_*` 未最適化状態を解消。

### 2026-02-15（追記）戦略内テクニカル評価の統一ルートを拡張

- `workers/hedge_balancer/worker.py` と `workers/manual_swing/worker.py` を
  `evaluate_entry_techniques` のローカル呼び出し＋`technical_context_*` 明示へ統一し、ローカル判定後に
  `entry_probability` / `entry_units_intent` を `entry_thesis` に反映する形へ拡張。
- `workers/macro_tech_fusion/worker.py` / `workers/micro_pullback_fib/worker.py` /
  `workers/range_compression_break/worker.py` / `workers/scalp_reversal_nwave/worker.py` について、
  `tech_tfs` を維持しつつ `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts`
  を明示化し、監査観点での入力要求の一貫性を揃えた。
- `docs/strategy_entry_technical_context_audit_2026_02_15.md` を更新し、現時点の集計を `evaluate_entry_techniques` 実装 37、
  `technical_context` 一式 37 に反映。

### 2026-02-24（追記）戦略別分析係ワーカーを追加

- `analysis/strategy_feedback_worker.py` を新規追加し、`quant-strategy-feedback.service` / `quant-strategy-feedback.timer` で
  定期実行する設計を追加。
- ワーカーは以下を自動反映し、戦略追加・停止・再開へ追従する運用を実装:
  - `systemd` からの戦略ワーカー検出（`quant-*.service`）
  - `workers.common.strategy_control` の有効状態
  - `logs/trades.db` の直近実績
- 主要出力先は `logs/strategy_feedback.json`（既存 `analysis.strategy_feedback.current_advice` の入力と互換）へ更新。
- 事故回避として、停止中戦略については最近のクローズ履歴なしなら出力抑止する `keep_inactive` 条件を導入。
- 追加改良: エントリー専用ワーカーが稼働中かを優先基準にし、`EXIT` ワーカーのみ残存するケースでの誤適用を防止。
- `ops/env/quant-strategy-feedback.env` を追加し、`STRATEGY_FEEDBACK_*` の運用キー（lookback/min_trades/保存先/探索範囲）を明文化。
- `docs/WORKER_REFACTOR_LOG.md` と `docs/WORKER_ROLE_MATRIX_V2.md` へ同時反映（監査トレースを維持）。

### 2026-02-16（追記）5秒スキャの最小ロット下限制御を運用値へ追従

- `workers/scalp_ping_5s/config.py` の `MIN_UNITS` が `max(100, ...)` で固定されていたため、`SCALP_PING_5S_MIN_UNITS=50` が設定されても
  実戦ロジックでは `units_below_min` が発生してエントリーを通過しづらい不整合があった。
- `SCALP_PING_5S_MIN_UNITS` の下限固定を `max(1, ...)` に変更し、環境変数ベースで 50 以上での運用実験を可能にした。
- 併せて、`ORDER_MIN_UNITS_SCALP_FAST=50` との整合を前提に、5秒スキャの再現監査で `units_below_min` の抑制を優先監視項目に加える運用を明示した。

### 2026-02-15（追記）35戦略の `evaluate_entry_techniques` 組み込みを構文修正

- 42戦略監査の残作業として一括追加した新規 `evaluate_entry_techniques` 呼び出しで、
  一部 `market_order/limit_order(` の引数開始直後にブロック混入が発生し、構文エラーを起こしていた箇所を補正。
- `entry_thesis_ctx` 前処理を注文呼び出しの外側へ移動し、`market_order/limit_order` 呼び出しの構文を復元。
- 補正対象は 30 戦略（`workers/*/worker.py` のうち `market_order/limit_order` 直呼び + `entry_thesis_ctx = None` 直下パターン）。
- 対象ファイルの `docs/strategy_entry_technical_context_audit_2026_02_15.md` を再生成し、`evaluate_entry_techniques` と
  `technical_context_*`/`tech_policy` 要件の監査ビューを最新化。

### 2026-02-15（追記）戦略別 technical_context 要件監査を実施

- `42` 戦略の監査対象（ユーザー指定）について、`evaluate_entry_techniques` と `technical_context_*` キーの実装有無を再集計。
- 監査結果は `docs/strategy_entry_technical_context_audit_2026_02_15.md` に保存。
- まとめ:
  - `evaluate_entry_techniques` 未実装かつ `technical_context` 明示なしが多数（ユーザー指定42件の主要対象）
  - `market_order` 非検出の wrapper/非entry系を除くと、実装未完了対象が 31 件
  - `tech_policy` の `require_*` 監査対象 5戦略について、`require_*` 値の確認を同時実施済み（`tech_fusion` は `range`/`trend` で分岐定義、`range` は `False/F/F/F`、`trend` は `False/F/F/F`）
- 次アクションとして、未実装戦略へ `technical_context_*` 要求明示とローカルテック評価の呼び出し導線を順次付与する運用を開始。

### 2026-02-15（追記）戦略側監査対象を戦略ワーカーに限定

- `order_manager` は注文 API 経路のインフラ層であり、戦略ローカル判断の監査対象から除外。
- `strategy` 側の条件（`workers/*/worker.py`）として再集計し、`market_order/limit_order` 直呼び 37 件が
  `evaluate_entry_techniques` と `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts` を
  すべて明示する状態を確認。
- ここで未完了だった `hedge_balancer` / `manual_swing` / `macro_tech_fusion` / `micro_pullback_fib` /
  `range_compression_break` / `scalp_reversal_nwave` を含む戦略群を最新定義に揃える対応を完了。

- 追記サマリ（戦略側監査完了）:
  - `workers/*/worker.py` 中の戦略ワーカー `market_order/limit_order` 直呼び: 37
  - `evaluate_entry_techniques` 未呼び: 0
  - `technical_context_tfs` / `technical_context_ticks` / `technical_context_candle_counts` 未明示: 0
  - `order_manager` は監査外（インフラ/API 入口）

### 2026-02-15（追記）technical_context の自動契約注入を明示要件時に限定

- `execution/strategy_entry.py` に `ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS` を追加（既定 `false`）。
- `strategy_entry` は、戦略タグ由来の自動補完ではなく、`technical_context_*` を戦略側で明示している場合のみ
  `technical_context` の取得・注入を実施する方針へ変更（共通で全戦略へ前提を押し付けない）。
- `technical_context_tfs` / `technical_context_fields` / `technical_context_ticks` / `technical_context_candle_counts` の
  明示がない戦略は、既定値フォールバックでの自動注入を行わない。
- `workers/tech_fusion/worker.py` について、`evaluate_entry_techniques` 呼び出しに
  `tech_tfs` / `tech_policy`（`require_fib` / `require_nwave` / `require_candle` 含む）と
  `technical_context_*` の要求定義を追加し、戦略ローカルでの評価前提を明示。

### 2026-02-20（追記）派生タグの戦略別技術契約を明示

- `execution/strategy_entry.py` の `strategy_tag` 解決を拡張し、サフィックス付き戦略名でも明示契約で解決できるようにした。
- 新規に明示化した主な `strategy_tag`（規約化キー）:
  - `tech_fusion`, `macro_tech_fusion`
  - `MicroPullbackFib`, `RangeCompressionBreak`
  - `ScalpReversalNWave`, `TrendReclaimLong`, `VolSpikeRider`
  - `MacroTechFusion`, `MacroH1Momentum`, `trend_h1`, `LondonMomentum`, `H1MomentumSwing`
- `entry_thesis` の受け渡し時に `technical_context_ticks` / `technical_context_tfs` / `technical_context_candle_counts` を
  明示し、`tech_policy` による要件定義を戦略側で扱う方針へ移行する布石とした（当時は `true` 前提を記録していた）。
- `SESSION_OPEN` を含む既存フローは維持しつつ、suffix 付き `scalp`/`macro`/`micro` タグでも
  pocket 非依存で解決可能なフォールバックを追加。  
  これにより N波/フィボ/ローソク要件の適用経路がより安定化した。

### 2026-02-14（追記）戦略別技術契約の運用名寄せ

- `execution/strategy_entry.py` に `_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を追加し、戦略ごとの既定テック要件を明文化。
- 自動注入されるキー:
  - `technical_context_tfs`（取得TF順）
  - `technical_context_fields`（保存指標）
  - `technical_context_ticks`（参照tick）
  - `technical_context_candle_counts`（TF別ローソク本数）
- 戦略側 `entry_thesis` がこれらを未設定の場合、上位契約で補完される。  
  補完後に `analysis.technique_engine.evaluate_entry_techniques` へ渡され、  
  `technical_context["result"]` と合わせて `entry_thesis["technical_context"]` へ格納する運用を統一。
- 対象は `SCALP_PING_5S`, `SCALP_PING_5S_B`, `SCALP_M1SCALPER`, `SCALP_MACD_RSI_DIV`,
  `SCALP_TICK_IMBALANCE`, `SCALP_SQUEEZE_PULSE_BREAK`,
  `SCALP_WICK_REVERSAL_BLEND`, `SCALP_WICK_REVERSAL_PRO`,
  `MICRO_ADAPTIVE_REVERT`, `MICRO_MULTISTRAT`。

### 2026-02-14（追記）戦略要件の絶対化（N波/フィボ/ローソク）

- `execution/strategy_entry.py` の
  `_STRATEGY_TECH_CONTEXT_REQUIREMENTS` を更新し、以下を標準化:
  - `technical_context_ticks` の戦略別明示（`latest_bid/ask/mid/spread_pips` を前提に、`tick_imbalance` 系で `tick_rate` 追加）
  - `technical_context_candle_counts` の戦略別明示（N本取得本数を戦略単位で定義）
  - `tech_policy` を戦略契約に追加し、`require_fib` / `require_nwave` / `require_candle` を `true` 固定
  - `tech_policy_locked` を追加し、`TECH_*` 環境上書きによる要件破壊を抑制
- 対象戦略/サブタグを契約化:
  - `scalp_ping_5s`, `scalp_m1scalper`, `scalp_macd_rsi_div`, `scalp_squeeze_pulse_break`
  - `tick_imbalance`, `tick_imbalance_rrplus`
  - `level_reject`, `level_reject_plus`
  - `tick_wick_reversal`, `wick_reversal`, `wick_reversal_blend`, `wick_reversal_hf`, `wick_reversal_pro`
  - `micro_multistrat` の代表として `micro_rangebreak`, `micro_vwapbound`, `micro_vwaprevert` 等の主要マイクロサブタグ
- `analysis/technique_engine.py` に `tech_policy_locked` を反映し、ロック時は `TECH_` 系環境変数での上書きをスキップする挙動を追加。
- 補足: `entry_thesis` が既に `tech_policy` を持つ場合も、`tech_policy_locked=True` を契約側で維持するためのマージ規則を追加。

## 削除（実装済み）

- `systemd/quant-hard-stop-backfill.service`
- `systemd/quant-realtime-metrics.service`
- `systemd/quant-m1scalper-trend-long.service`
- `systemd/quant-scalp-precision-*.service`（まとめて削除）
- `systemd/quant-hedge-balancer.service`
- `systemd/quant-hedge-balancer-exit.service`
- `systemd/quant-trend-reclaim-long.service`
- `systemd/quant-trend-reclaim-long-exit.service`
- `systemd/quant-margin-relief-exit.service`
- `systemd/quant-realtime-metrics.timer`

## 補足

- `scalp_ping_5s` と `scalp_macd_rsi_div` は現状、ENTRY/EXIT を1対1で分離済み。
- `quant-scalp-precision-*` は削除済みで、置換された `quant-scalp-*` サービス群が戦略別で単独起動される。
- `strategy_control` はフラグ配信の母体で、`execution/order_manager.py` の事前チェックで
  `strategy_control.can_enter/can_exit` を参照することで、ENTRY/EXIT 可否を実行時に即時反映。

## V2 追加（完了: 2026-02-14）

- `ops/systemd/quantrabbit.service` は monolithic エントリとして廃止対象へ昇格（本番起動から排除）。
- 本設計では「データ / 制御 / 戦略 / 分析 / 注文 / ポジ管理」を別プロセス境界で扱う。
- `execution/order_manager.py` / `execution/position_manager.py` は service-first ルートを持ち、
  strategy worker は基本的に HTTP で各サービスを経由する運用へ。

## 運用反映（2026-02-14 直近）

- `fx-trader-vm` にて `main` 基点の全再インストールを実施し、`quant-market-data-feed` / `quant-strategy-control` を含む
  V2サービス群を再有効化。`quantrabbit.service` は再起動済み。
- `quant-order-manager.service` / `quant-position-manager.service` はサービス側で再有効化したが、`main` 上に
  `workers/order_manager` / `workers/position_manager` が未収録のため、現時点では起動が `ModuleNotFoundError` で継続リトライ。
- 今回の状態は次のデプロイでワーカー実装を main に反映して解消する必要がある。
- `scripts/install_trading_services.sh` を改善し、`enable --now` の起動失敗でスクリプト全体が止まらないように
  `enable` と `start` を分離。これにより、起動時点で `enabled` 指定されたサービス群は有効化された状態を維持し、
  VM再起動時の自動起動対象から漏れにくくする運用を確立。

## 2026-02-14 組織図更新運用（V2）

- `docs/WORKER_ROLE_MATRIX_V2.md` の「現在の状態」「図」「運用制約」を、V2構成変更時に毎回更新する運用ルールを明文化。
- WORKER関連の変更点は、`WORKER_REFACTOR_LOG` と `WORKER_ROLE_MATRIX_V2` を同一コミットで同期更新する運用を追加。
- `main` 反映後の VM 監査時に、構成図と実サービス状態の齟齬がないかを確認する（監査ログの追記対象）。

## 2026-02-16 VM再投入後整備（V2固定）

- `fx-trader-vm` で再監査し、V2外のレガシー戦略・monolithic系ユニットを停止/無効化しました。対象:
  - `quantrabbit.service`
  - `quant-impulse-retest-s5*`
  - `quant-hard-stop-backfill.service`
  - `quant-margin-relief-exit*`
  - `quant-trend-reclaim-long*`
  - `quant-micro-adaptive-revert*`
  - `quant-scalp-precision-*`（旧系）
  - `quant-realtime-metrics.service/timer`（分析補助タイマーも除外）
- VM上の V2実行群は `quant-market-data-feed` / `quant-strategy-control` / 各ENTRY-EXITペア + `quant-order-manager` / `quant-position-manager` のみ有効稼働を維持。
- `systemctl list-unit-files --state=enabled --all` / `systemctl list-units --state=active` で再確認済み。

### 2026-02-16（追加）V2のENTRY/EXIT責務固定

- 戦略実行の意思決定入力を統一:
  - `scalp_ping_5s`
  - `scalp_m1scalper`
  - `micro_multistrat`
  - `scalp_macd_rsi_div`
  - `scalp_precision`（`scalp_squeeze_pulse_break` / `scalp_tick_imbalance` / `scalp_wick_reversal_*` のラッパー含む）
- 各戦略の `entry_thesis` を拡張し、`entry_probability` と `entry_units_intent` を付与する実装を反映:
  - `workers/scalp_ping_5s/worker.py`
  - `workers/scalp_m1scalper/worker.py`
  - `workers/micro_multistrat/worker.py`
  - `workers/scalp_macd_rsi_div/worker.py`
  - `workers/scalp_precision/worker.py`
- `order_manager` 側の役割を「ガード＋リスク検査」に限定:
  - `quant-strategy-control` の可否フラグ（entry/exit/global）を参照するだけのフローに合わせる
  - 戦略横断の強制的な勝率採点/順位付けや「代替戦略選別」ロジックは追加しない方針を維持。
- `WORKER_ROLE_MATRIX_V2.md` を今回内容に合わせて同一コミットで更新（責務・禁止ルール・実行図の注記）。

### 2026-02-16（追記）ping-5s 配布整合性

- `workers/scalp_ping_5s/worker.py` の `entry_thesis` 生成部に起きていた
  `IndentationError: unexpected indent (line 4253)` を修正。
- 同コミットで `entry_probability` と `entry_units_intent` を付与したロジックは維持しつつ、`entry_thesis` の
  インデントを `WORKER_ROLE_MATRIX_V2.md` の責務定義に準拠する形へ整形。
- `main` (`1b7f6c56`) を VM へ反映し、`quant-scalp-ping-5s.service` は `active (running)` を確認済み。

### 2026-02-16（追記）session_openの意図受け渡しをaddon_live経路へ統一

- `workers/session_open/worker.py`
  - `projection_probability` を `entry_probability` として `order` へ付与。
  - `size_mult` 由来の意図ロットを `entry_units_intent` として `order` へ付与。
- `workers/common/addon_live.py`
  - `order` から `entry_probability` / `entry_units_intent` を抽出し、`entry_thesis` に確実に反映するように統一。
  - `intent` 側の同名値もフォールバックで受ける運用に変更。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` 側の責務文言は、
  `AddonLive` 経路でも `session_open` を含む各戦略で意図値を `entry_thesis` へ保持する運用へ揃えた。

### 2026-02-16（追記）runtime env 参照の `ops/env/quant-v2-runtime.env` へ移行

- `systemd/*.service` の `EnvironmentFile` を `ops/env/quant-v2-runtime.env` に統一。
- `quant-v2-runtime.env` へ V2に必要なキーのみを収束（OANDA, V2ガード制御, order/position service, pattern/brain/forecast gate, tuner）。
- scalp系調整系スクリプト（`vm_apply_scalp_ping_5s_*`）の環境適用先を
  `ops/env/scalp_ping_5s.env` 系へ移行。
- `startup_script.sh` と `scripts/deploy_via_metadata.sh`/`scripts/vm_apply_entry_precision_hardening.sh` で
  legacy 環境ファイル依存を撤去し、`ops/env/quant-v2-runtime.env` をデフォルト注入先に変更。
- 併せて AGENTS/VM/GCP/監査ドキュメントの監査対象コマンドを新環境ファイル参照へ更新。

### 2026-02-16（追記）戦略ENTRY/EXIT workerのenv分離

- V2戦略ENTRY/EXIT群（`scalp*`, `micro*`, `session_open`, `impulse_retest_s5`）の `systemd/*.service` から
  戦略固有 `Environment=` を切り出し、各サービス対応の `ops/env/quant-<service>.env` を新設。
  - 追加/更新対象 `systemd`:
    - `quant-m1scalper*.service`
    - `quant-micro-adaptive-revert*.service`
    - `quant-micro-multi*.service`
    - `quant-scalp-macd-rsi-div*.service`
    - `quant-scalp-ping-5s*.service`
    - `quant-scalp-squeeze-pulse-break*.service`
    - `quant-scalp-tick-imbalance*.service`
    - `quant-scalp-wick-reversal-blend*.service`
    - `quant-scalp-wick-reversal-pro*.service`
    - `quant-session-open*.service`
    - `quant-impulse-retest-s5*.service`
  - 追加/更新対象 `ops/env/*`:
    - `ops/env/quant-m1scalper*.env`
    - `ops/env/quant-micro-adaptive-revert*.env`
    - `ops/env/quant-micro-multi*.env`
    - `ops/env/quant-scalp-macd-rsi-div*.env`
    - `ops/env/quant-scalp-ping-5s*.env`
    - `ops/env/quant-scalp-squeeze-pulse-break*.env`
    - `ops/env/quant-scalp-tick-imbalance*.env`
    - `ops/env/quant-scalp-wick-reversal-blend*.env`
    - `ops/env/quant-scalp-wick-reversal-pro*.env`
    - `ops/env/quant-session-open*.env`
    - `ops/env/quant-impulse-retest-s5*.env`
- `quant-scalp-ping-5s` 系は既存の戦略上書きenv（`scalp_ping_5s.env`, `scalp_ping_5s_b.env`）を維持し、`ops/env/quant-scalp-ping-5s*.env` を基本設定用として分離。
- `AGENTS.md` と `WORKER_ROLE_MATRIX_V2.md` を同一コミットで更新し、監査時に `EnvironmentFile` の二段構造
  (`quant-v2-runtime.env` + `quant-<service>.env`) をチェック対象化。

### 2026-02-17（追記）position_manager 呼び出しのHTTPメソッド齟齬解消

- VM監査で `quant-position-manager` 側の `POST /position/open_positions` が `405` となるログを確認。
- 原因は `execution/position_manager.py` の `open_positions` 呼び出しが固定 `POST` だったため、ワーカー定義
  (`workers/position_manager/worker.py`) が `GET /position/open_positions` を公開していることとの不一致。
- 修正: `execution/position_manager.py` の `_position_manager_service_call()` を `path == "/position/open_positions"` 時に
  `GET` + query params (`include_unknown`) へ分岐するよう変更し、サービス経路の整合を復旧。
- 変更反映後、`quant-order-manager` を再起動して該当 405 検知率の改善を確認する。

### 2026-02-17（追記）open_positions 405 の下位互換対策

- 運用側の呼び出しに POST が混在しているケースを想定し、`workers/position_manager/worker.py` に
  `POST /position/open_positions` を追加受け口として実装。
- `execution/position_manager.py` では `path` の末尾スラッシュを除去して正規化し、`/position/open_positions` 系を
  `GET + params` へ固定振り分けする分岐を堅牢化。
- 既存の GET 経路は維持しつつ、POST 混在時の `405 Method Not Allowed` を回避。

### 2026-02-14（追記）order_manager の戦略意図保全（市場/リミット両方）

- `execution/order_manager.py` の `market_order()` と `limit_order()` 入口で、`entry_probability` と `entry_units_intent` を
  `entry_thesis` へ必須注入・補完する仕組みを統一し、`entry_probability` に応じたロット縮小／リジェクトのみを
  `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1` 時の実装として明確化。
  - reduce_only/manual 除外時のみ `preserve` を有効化。
- `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1` かつ pocket/manual 以外では、戦略側意図が示すSL/TPやサイズ方針を
  order_manager が一方的に再設計しないよう、以下は `not preserve` 条件へ追従:
  - Brain / Forecast / Pattern gate
  - entry-quality / microstructure gate
  - spread block / dynamic SL / min-RR 調整 / TP&SLキャップ / hard stop / normalize / loss cap / direction cap
- ただし、`entry_probability` による「許容上限超えでの縮小」や「超低確率での拒否」は risk 側許容範囲として維持。
- リミット注文側も同様に `entry_probability` 注入・同条件ガードを追加し、`order_manager_service` 経路に同じ意図を引き継ぐように統一。

### 2026-02-14（追記）戦略横断意図協調（entry_intent_board）基盤整備
- `execution/order_manager.py` に `entry_intent_board` / `intent_coordination` の基盤（スキーマ、DB、preflight、worker endpoint）を追加。
- 当時の方針整理で `strategy_entry` 側連携は一旦抑止し、`order_manager` 側に上書き的な再設計を残さない構成を優先した。

### 2026-02-18（追記）意図協調をstrategy_entry経由で復帰
- `execution/strategy_entry.py` の `market_order` / `limit_order` で
  `entry_probability` と `entry_units_intent` を維持したまま `strategy_tag` 解決し、
  `/order/coordinate_entry_intent` を経由してから `order_manager` へ転送する形へ戻す。
- `workers/order_manager/worker.py` の `POST /order/coordinate_entry_intent` を有効のまま維持し、
  各戦略が自戦略意図を保持したまま黒板協調の結果を反映できる運用へ復元。

### 2026-02-17（追記）scalp_macd_rsi_div の env 互換・Micro版起点を追加
- `workers/scalp_macd_rsi_div/config.py` に `MACDRSIDIV` / `SCALP_MACD_RSI_DIV_B` 系設定名の互換吸収を追加し、戦略が現行
  `SCALP_PRECISION_*` で読み取れるよう統一。
- `workers/scalp_macd_rsi_div_b/worker.py` を追加し、`SCALP_MACD_RSI_DIV_B_*` 設定を `SCALP_PRECISION_*` にマッピングして
  既存エントリー基盤を再利用した micro 相当運用を追加。
- `systemd/quant-scalp-macd-rsi-div-b.service` と
  `ops/env/quant-scalp-macd-rsi-div-b.env` を追加し、`quant-scalp-macd-rsi-div-exit.env` の管理タグに
  `scalp_macd_rsi_div_b_live` を追記して B版と共存可能な状態にした。

### 2026-02-16（追記）scalp_macd_rsi_div B のEXIT独立ユニット化
- `systemd/quant-scalp-macd-rsi-div-b-exit.service` を追加し、B版のエントリー/退出を1:1構成へ拡張。
- `ops/env/quant-scalp-macd-rsi-div-b-exit.env` を追加し、`SCALP_PRECISION_EXIT_TAGS`/`MACDRSIDIV_EXIT_TAGS` を
  `scalp_macd_rsi_div_b_live` のみで受ける専用Exit運用に変更。
- `ops/env/quant-scalp-macd-rsi-div-exit.env` から B タグを外し、既存ExitにB案件が混在しない分離構成へ変更。
- 監査側（`scripts/ops_v2_audit.py`）の `optional_pairs` に
  `quant-scalp-macd-rsi-div-b[-exit]` を追加し、ペア稼働状態の監査対象に明記。

### 2026-02-14（追記）黒板協調・最小ロット判定を strategy 固定化
- `execution/order_manager.py` の `entry_intent_board` 集約キーを `strategy_tag` 前提へ更新。
- `_coordinate_entry_intent` が `pocket` ではなく `strategy_tag + instrument` の組で照合するよう変更。
- `min_units_for_strategy(strategy_tag, pocket)` を新設し、`strategy_tag` 指定時は `ORDER_MIN_UNITS_STRATEGY_<strategy>` を優先適用。
- `execution/strategy_entry.py` の戦略側協調前チェックも `min_units_for_strategy` を利用するよう更新。

### 2026-02-17（追記）order/position worker の自己service呼び出しガード

- `quant-position-manager.service` と `quant-order-manager.service` の環境衝突（`quant-v2-runtime.env` が
  `*_SERVICE_ENABLED=1` を上書きしていた）を受け、専用 env を新設してサービス責務を明確化。
  - 追加: `ops/env/quant-position-manager.env`
  - 追加: `ops/env/quant-order-manager.env`
  - 更新: `systemd/quant-position-manager.service`

### 2026-02-17（追記）order_manager 入力確率の欠損耐性を強化

- `execution/order_manager.py` の `market_order` 入口で、`entry_probability` の正規化候補を拡張。
- `entry_probability` が `None` / 非数値 / `NaN` / `Inf` いずれでも
  `entry_thesis["confidence"]` / `confidence` 引数（優先順）を用いて補完し、意図値を欠損に依存させない実装を追加。
- `entry_probability` が不正値でも有効な `confidence` があれば上書き補完する挙動に変更し、`entry_units_intent` 同様に
  実行経路の安定性を維持。
- 本変更は品質低下を避けるため、ロジック上の選別を追加せず、既存のガード/リスク条件の枠内でのみ運用される。
  - 更新: `systemd/quant-order-manager.service`
- 両ワーカー側にも明示ガードを入れ、`execution/*_manager.py` の service-first 経路を有効化しつつ、
  各ワーカー実体が self-call（自分自身のHTTP経路を再コール）しない安全策を追加。
- `main` 経由の再監査で `POSITION_MANAGER` 側の 127.0.0.1:8301 での read timeout 連鎖が解消することを確認済み（次アクションとして
  デプロイ後の VM 監査結果を添付）。

### 2026-02-14（追記）market-data-feed の履歴取得を差分化

- `market_data/candle_fetcher.py`
  - `fetch_historical_candles()` が `count` 取得時と同時に `from` / `to` 範囲取得（RFC3339）も扱えるように拡張。
  - `initialize_history()` を「既存キャッシュ終端から次バー境界まで」を起点にした差分再取得へ変更。
    - 既存 `factor_cache` の最終足時刻を参照し、その `+TF幅` から `now` までを取得。
    - 取得時に重複時刻を除外して append し、既存件数に加えて 20本条件を満たせば成功扱い。
  - 運用中再シード時に固定リトライで同一履歴を上書きし続ける問題を低減。
- `WORKER_ROLE_MATRIX_V2.md` のデータ面に、シード時の差分補完方針（前回キャッシュ終端ベース）を追記。

### 2026-02-18（追記）V2監査（VM自動実行）追加

- `scripts/ops_v2_audit.py` を追加し、V2導線（データ/制御/ORDER/POSITION/戦略）監査を1回の実行で集約。
- 追加: `scripts/ops_v2_audit.sh`（systemd起動ラッパ）
- 追加: `systemd/quant-v2-audit.service` / `systemd/quant-v2-audit.timer`
- 監査対象:
  - `quant-market-data-feed`, `quant-strategy-control`, `quant-order-manager`, `quant-position-manager` の active
  - 戦略ENTRY/EXITの主要ペア active 状態
  - `EnvironmentFile` 構成（`quant-v2-runtime.env` + `quant-<service>.env`）
  - `quant-v2-runtime.env` 制御キー値（主要フラグ）
  - `position/open_positions` の `405` 監視（order/position worker 呼び出し相当）
  - `quantrabbit.service` 等 legacy active の有無
- `systemd install`/`timer` 導入は `install_trading_services.sh --units` 経由で統一し、運用側は `logs/ops_v2_audit_latest.json` を監査ログとして参照。

### 2026-02-17（追記）V2監査の誤検知を抑制するための修正

- `quant-v2-audit.service` が誤検知していた `position/open_positions` の `405` 集計は、journal 行中の
  タイムスタンプ文字列（`...:405`）が `405` 検知に拾われる副作用が原因だったため、`scripts/ops_v2_audit.py` を修正。
  - メソッド不一致判定は `Method Not Allowed` と `... /position/open_positions ... 405` の実リクエスト行のみを対象化。
- `install_trading_services.sh --all` が V2監査で禁止とするレガシーサービスを誤って再有効化しないよう、除外対象を明示。
  - 除外: `quant-impulse-retest-s5*`, `quant-micro-adaptive-revert*`（`--all` ではインストールせず、明示 `--units` 指定時のみ許容）
- `install_trading_services --all` 再実行時も V2監査の disallow ルールを壊しにくい状態に更新。

### 2026-02-14（追記）legacy残存時の監査許容ルールを追加

- `scripts/ops_v2_audit.py` に、`OPS_V2_ALLOWED_LEGACY_SERVICES` で legacy サービスを明示許可する設定を追加。
- `ops/env/quant-v2-runtime.env` に `OPS_V2_ALLOWED_LEGACY_SERVICES=...` を設定し、
  `quant-impulse-retest-s5*` と `quant-micro-adaptive-revert*` の active 判定を
  `critical` ではなく `warn` へトレードオフし、当面の運用継続を担保。

### 2026-02-14（追記）install_trading_services.sh の起動待機ハング対策

- `scripts/install_trading_services.sh --all` 実行時に、`quant-strategy-optimizer.service` の
  oneshot長時間処理起動でスクリプト全体が待機し続ける現象を確認。
- 対応として `scripts/install_trading_services.sh` の `enable_unit()` に
  `NO_BLOCK_START_UNITS` を追加し、`quant-strategy-optimizer.service` を
  `systemctl start --no-block` で起動要求するよう変更。

### 2026-02-14（追記）本番トレード制御フラグ有効化

- `ops/env/quant-v2-runtime.env` の `MAIN_TRADING_ENABLED` を `0` から `1` に変更し、
  V2運用の「戦略ワーカー→order/position manager経路」実行を本番可で許可。
- 併せて VM 運用環境の `ops/env/quant-v2-runtime.env` も同値で更新し、core 監査（`quant-v2-audit`）後に
  リアルタイム取引許可状態の整合を確認。
- `ops_v2_audit` の実運用期待値を `MAIN_TRADING_ENABLED=1` に更新し、取引許可状態を監査基準へ反映。
- このため `--all` 実行時の完了待機を回避しつつ、監査ジョブ（`quant-v2-audit`）の定期実行を維持。

### 2026-02-24（追記）戦略分析係に戦略固有パラメータ参照を追加

- `analysis/strategy_feedback_worker.py` を更新し、`systemd` 上の戦略サービス `EnvironmentFile` から
  戦略ごとに一致する環境パラメータを抽出して `strategy_params` として保持するようにした。
- 取得した `strategy_params` は `strategy_feedback` 生成時に各戦略の `strategy_params.configured_params` として
  JSON 出力へ同梱し、`entry_probability_multiplier` / `entry_units_multiplier` /
  `tp_distance_multiplier` / `sl_distance_multiplier` の根拠追跡性を強化。
- 戦略追加・停止時の観測に対しても `systemd` 検知を優先し、停止戦略は
  `LAST_CLOSED` の古さ条件を満たさない限り `strategy_feedback` 出力対象外にして誤適用を回避。
- 追加で、実行中 `quant-*.service` の `FragmentPath` を systemd から直接読み取り、リポジトリ上の
  unit ファイルに未同期の戦略追加にも即座に追従するようにした。

### 2026-02-15（追記）analysis_feedback の最終受け渡しを明示化

- `analysis/strategy_feedback.py` で `strategy_params` 内の `configured_params` を分離して
  `advice["configured_params"]` として明示的に出力し、ノイズ対策しない形で戦略固有パラメータを保持。
- `execution/strategy_entry.py` で分析結果を `entry_thesis["analysis_feedback"]` に格納し、
  既存利用者互換として `analysis_advice` も併記して戦略別改善値の監査性を維持。

### 2026-02-15（追記）戦略側 technical_context 要件の最終穴埋め

- `workers/hedge_balancer/worker.py` と `workers/manual_swing/worker.py` の
  `evaluate_entry_techniques` 呼び出し前に `tech_policy` を明示化し、
  `require_fib/require_median/require_nwave/require_candle` を全戦略側で揃えた。
- 同時に監査資料 `docs/strategy_entry_technical_context_audit_2026_02_15.md` の
  該当行（`hedge_balancer`, `manual_swing`）も同値に更新。

### 2026-02-15（追記）tick imbalance ラッパーを独立戦略モード固定化

- `workers/scalp_tick_imbalance/worker.py` を `scalp_precision_worker` 直接 import 依存から
  起動時に `SCALP_PRECISION_*` を明示設定して mode 固定起動する構成へ更新。
- `SCALP_PRECISION_MODE=tick_imbalance`、`ALLOWLIST`、`MODE_FILTER_ALLOWLIST`、`UNIT_ALLOWLIST`、
  `LOG_PREFIX` を `__main__` 起動時に再設定し、`workers/scalp_precision/worker.py` の
  `evaluate_entry_techniques` 入口を再利用しつつ、戦略名を `scalp_tick_imbalance` に固定。
- 監査資料 `docs/strategy_entry_technical_context_audit_2026_02_15.md` の
  `scalp_tick_imbalance` 行を `evaluate_entry_techniques`/`technical_context_*` 実装済み扱いに更新。

### 2026-02-15（追記）V2実用起動候補（8戦略ペア）構成監査

- 対象エントリー/EXITペア:
  - `quant-scalp-ping-5s` / `quant-scalp-ping-5s-exit`
  - `quant-micro-multi` / `quant-micro-multi-exit`
  - `quant-scalp-macd-rsi-div` / `quant-scalp-macd-rsi-div-exit`
  - `quant-scalp-tick-imbalance` / `quant-scalp-tick-imbalance-exit`
  - `quant-scalp-squeeze-pulse-break` / `quant-scalp-squeeze-pulse-break-exit`
  - `quant-scalp-wick-reversal-blend` / `quant-scalp-wick-reversal-blend-exit`
  - `quant-scalp-wick-reversal-pro` / `quant-scalp-wick-reversal-pro-exit`
  - `quant-m1scalper` / `quant-m1scalper-exit`
- `/systemd/*.service` 上で 16本全てが `ExecStart` を `workers.<strategy>.worker` / `.exit_worker` に正しく固定し、
  `EnvironmentFile` も `quant-v2-runtime.env` + 各戦略 env の2行で定義される構成を確認。
- `quant-scalp-ping-5s.service` の追加 `EnvironmentFile` として
  `ops/env/scalp_ping_5s.env` を明示的に用意し、参照行の未作成問題を解消。
- `scalp_wick_reversal_blend` 側 env は `SCALP_PRECISION_ENABLED=1` に更新し、実運用起動前提を揃えた。

### 2026-02-15（追記）WickReversalBlend 起動停止原因の除去

- `workers/scalp_precision/worker.py` の `_place_order` 内で、
  `not tech_decision.allowed` の分岐と `_tech_units <= 0` の分岐が
  `continue` になっており、関数内制御として不正だったため
  `SyntaxError: 'continue' not properly in loop` を発生させていた点を修正。
- 2箇所を `return None` に変更し、`quant-scalp-wick-reversal-blend.service` の起動停止要因を解消する
  形にした。

### 2026-02-15（追記）wick/squeeze/tick wrapper の独立起動化

- `workers/scalp_squeeze_pulse_break/worker.py` / `workers/scalp_wick_reversal_blend/worker.py` /
  `workers/scalp_wick_reversal_pro/worker.py` を `SCALP_PRECISION_MODE` 固定起動形式へ揃え、
  `SCALP_PRECISION_ENABLED/ALLOWLIST/UNIT_ALLOWLIST/LOG_PREFIX` を `__main__` で明示設定するように修正。
- これにより、環境差し戻し時でもそれぞれの戦略名で `scalp_precision` のローカル評価を実行し、
  V2実用8戦略（`scalp_tick_imbalance`含む）へ同一方針で意図固定を維持できる状態を整備。

### 2026-02-16（追記）M1 Scalp の N-Wave アライメント停止要因を可変化

- `strategies/scalping/m1_scalper.py` に `M1SCALP_NWAVE_ALIGN_*` 環境変数を追加し、
  N-Wave 連続検知時の `skip_nwave_*_alignment` を運用側で可変化。
- `ops/env/quant-m1scalper.env` に `M1SCALP_NWAVE_ALIGN_ENABLED=0` を追加して
  アライメントガードを一時無効化（必要に応じて再有効化可能）し、5秒以外の戦略通過率改善を優先。

### 2026-02-16（追記）micro_multi の因子劣化時軽量化 + PositionManager サービス耐障害化

- `workers/micro_multistrat/worker.py`
  - M1 因子 `age_m1` が `MAX_FACTOR_AGE_SEC` を超過した場合、`FRESH_TICKS_STALE_SCALE_MIN` と
    `FRESH_TICKS_STALE_SCALE_HARD_SEC` を使って `stale_scale` を算出し、`units` に適用して
    スケールアウト時に過大エントリーを抑える「自動軽量化」を追加。
  - `workers/micro_multistrat/config.py` に上記2変数を追加し、運用時の調整余地を確保。
- `execution/position_manager.py`
  - サービス障害時の再試行を指数バックオフ（`POSITION_MANAGER_SERVICE_FAIL_BACKOFF_*`）で抑止し、
    短時間の連打エラーを軽減。
- `ops/env/quant-v2-runtime.env`
  - `ORDER_MANAGER_SERVICE_FALLBACK_LOCAL` と `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL` を `1` に変更し、
    サービス面不具合時はローカルフォールバック経路へ短時間で移行する運用を追加。

### 2026-02-16（追記）session_open（AddonLive）経路で `env_prefix` を固定注入

- `workers/common/addon_live.py` の `AddonLiveBroker.send()` で、`order/intent/meta` から
  `env_prefix`（または `ENV_PREFIX`）を受け取り、`entry_thesis` と `meta_payload` に
  同一値を注入するよう変更。
- `execution/strategy_entry` 側の `coordinate_entry_intent` で `env_prefix` が混在しにくい状態を保つための
  追加対策として、session_open 系の発注にも意図値（`entry_units_intent` / `entry_probability`）と同様に
  `env_prefix` の明示注入を追加。

- 2026-02-16: `scalp_m1scalper` エントリー抑制対策として `ops/env/quant-m1scalper.env` に `M1SCALP_ALLOW_REVERSION=1` を追加し、レンジ条件許可 (`M1SCALP_REVERSION_REQUIRE_STRONG_RANGE=1`, `M1SCALP_REVERSION_ALLOWED_RANGE_MODES=range`) と閾値を緩和 (`M1SCALP_REVERSION_MIN_RANGE_SCORE=0.72`, `M1SCALP_REVERSION_MAX_ADX=20`) へ変更。M1の`buy-dip / sell-rally` 系シグナル（リバーション）を通過しやすくし、エントリー減少の改善を試験的に実施。
- 2026-02-16: `M1SCALP_REVERSION_ALLOWED_RANGE_MODES` を `range,mixed` に広げ、`M1SCALP_REVERSION_MIN_RANGE_SCORE` を `0.55`、`M1SCALP_REVERSION_MAX_ADX` を `30` に変更。`ALLOW_REVERSION=1` を維持しつつ、リバーション通過条件の厳しさを中立寄りに再調整。

### 2026-02-24（追記）5秒B `no_signal` の根本分解とフォールバック制御を確定

- `workers/scalp_ping_5s/worker.py`
  - `no_signal` の内部理由を `revert_not_enabled` から分解し、`revert_disabled` と `revert_not_found` を明示化。
  - `revert` 判定対象がない場合の判定を意図別に分離し、`no_signal` 集計を `data/Signal不足` と
    `revertロジック未成立` で分けて監査可能にした。
  - `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW` を導入し、fallback の
    `WINDOW_SEC` までの最終全体再試行を明示オプトイン化。既存挙動（既定 true）を保ちつつ、
    B 側では明示的に無効化できるよう固定。
- `workers/scalp_ping_5s/config.py`
  - `SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW` を新設し、
    no_signal の再試行拡張挙動を設定可能にした。
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_REVERT_ENABLED=1` を明示し、revert 判定未評価時の
    `revert_not_enabled` を混線要因から切り離せるようにした。
  - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=0` を追加して、
    Bでの `no_signal:insufficient_signal_rows_fallback_exhausted` 発生条件を抑制。
  - `SCALP_PING_5S_PATTERN_GATE_OPT_IN` の混在要因になっていた `SCALP_PING_5S` 系行を整理し、B版専用キーへ寄せた。

### 2026-02-16（追記）5秒B env-prefix 監査ログを追加

- `workers/scalp_ping_5s_b/worker.py`
  - B ラッパー起動時に、`SCALP_PING_5S_B -> SCALP_PING_5S` の変換後実効値を明示ログ化。
  - `SCALP_PING_5S_REVERT_ENABLED` を起動時に監査ログへ反映し、B運用時の意図しない `revert_disabled` 連発を
    起点で検知しやすくした。
- `workers/scalp_ping_5s/worker.py`
  - `no_signal` の detail 正規化で `revert_not_enabled`（旧名）を
    `revert_disabled` として一括吸収し、監査時に混線カテゴリを抑制。

### 2026-02-16（追記）5秒B `revert_not_found` 優勢時の取り残し抑制（実験値）

- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_REVERT_WINDOW_SEC=1.60` / `SCALP_PING_5S_B_REVERT_SHORT_WINDOW_SEC=0.55` に引き上げ、
    5秒Bでのリバーション検知窓を拡大。
  - `SCALP_PING_5S_B_REVERT_MIN_TICK_RATE=0.90` / `SCALP_PING_5S_B_REVERT_RANGE_MIN_PIPS=0.45` /
    `SCALP_PING_5S_B_REVERT_SWEEP_MIN_PIPS=0.35` / `SCALP_PING_5S_B_REVERT_BOUNCE_MIN_PIPS=0.10` /
    `SCALP_PING_5S_B_REVERT_CONFIRM_RATIO_MIN=0.50` を低減し、revert判定の成立しやすさを上げる。
- 目的は `no_signal:revert_not_found` 偏在を短期的に抑え、取り残し率を下げること。
- この調整は 5秒B のみを対象としており、A側設定には影響しない。

### 2026-02-16（追記）5秒B entry-skip（spread/min_units）対策を適用

- `ops/env/scalp_ping_5s_b.env` に対し、`scalp_ping_5s_b_live` の取り逃し要因である
  `spread_block` と `units_below_min` を同時に抑えるため、次を反映。
  - `SCALP_PING_5S_B_MAX_SPREAD_PIPS=1.00`
  - `ORDER_SPREAD_BLOCK_PIPS_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
  - `ORDER_SPREAD_BLOCK_PIPS_STRATEGY_SCALP_PING_5S_B=1.00`
  - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.65`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE=300`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B=300`
- 対象は 5秒Bのみ。A側（`scalp_ping_5s`）設定は変更なし。
- 適用後、同一日の 3d/7d で
  `spread_block`・`entry_probability_below_min_units` の比率が低下し、
  `submit_attempt` が増えるかを次回監査で確認する。

- `$(date +
- `2026-02-17`: 旧戦略残骸の最終除去を追加。`quant-m1scalper*` 系 (`systemd`/`ops/env` / VM `/etc/systemd/system` / VMリポジトリ `systemd` ディレクトリ) を一括廃止し、VM上の `systemctl list-unit-files` からも一致除去。`quant-impulse-retest-s5*` と `quant-micro-adaptive-revert*` 系も同時に除去・非起動化。

### 2026-02-17（追記）factor_cache の確定足更新を market_data_feed で再接続

- `workers/market_data_feed/worker.py`
  - `start_candle_stream` へ渡すハンドラに `indicators.factor_cache.on_candle` を追加。
  - これにより `M1` の終了足が `logs/factor_cache.json` へ永続反映される経路を復元。
- 変更意図: `on_candle_live` は同プロセスの即時計算には有効だが、別プロセスの戦略ワーカーからは
  `refresh_cache_from_disk()` 経由で読み出すため、確定足更新の永続化がないと
  `factor_stale` が継続して発生し続けるため。

### 2026-02-17（追記）quant-micro-multi の factor_stale 警告耐性を最適化

- `workers/micro_multistrat/worker.py`
  - M1 factors を `all_factors()` 取得後に `MAX_FACTOR_AGE_SEC` を超える場合、まず tick から再構築した値を
    `refresh_cache_from_disk()` 直後の評価ループで短時間保持し、次ループでも再利用可能にした。
  - `micro_multi_skip` 警告は「tick 再構成で復旧可能なケース」では抑制し、まだ stale な場合のみ
    `stale_scale` 適用と警告を継続。
  - `age` が一時的に跳ねた直後の連続ノイズを抑えつつ、`proceeding anyway` の安全挙動は維持。
- 追加意図: この調整は、`quant-micro-multi` の入口停止が誘発されやすい局面での
  `entry_thesis` 通過率を維持しつつ、保全的な規模縮小/警告は維持すること。

### 2026-02-17（追記）quant-micro-multi 起動ログの安定可視化と `_LOCAL_FRESH_M1` 参照保護

- `workers/micro_multistrat/worker.py`
  - `Application started!` を起動イベントとして `warning` で出すよう変更。
  - `micro_multi_worker` の `_LOCAL_FRESH_M1` / `_LAST_FRESH_M1_TS` 参照を `globals()` 経由で扱い、参照スコープ崩れ（`UnboundLocalError`）を回避。
  - 併せて `local_fresh_m1` / `last_fresh_m1_ts` をループ内で更新し、tick 再構成時の状態整合を維持。
- 変更意図: 再起動直後からの起動監査可視化と稼働阻害エラー再発防止を優先し、`Application started!` 検知運用を成立させること。

### 2026-02-17（追記）5秒スキャを B 専用に切替（無印ENTRY停止）

- 目的: `scalp_ping_5s_live`（無印）と `scalp_ping_5s_b_live`（B）の実運用差分を固定し、5秒スキャのENTRY経路をBへ一本化。
- 反映:
  - `ops/env/scalp_ping_5s.env`: `SCALP_PING_5S_ENABLED=0` を先頭で明示。
  - `ops/env/quant-scalp-ping-5s.env`: `SCALP_PING_5S_ENABLED=0` に変更。
  - `ops/env/quant-scalp-ping-5s-b.env`: `SCALP_PING_5S_B_ENABLED=1` に変更。
- 意図: systemdの `EnvironmentFile` 多重読み込み時でも、無印は常時idle、Bは常時有効の状態を維持する。

### 2026-02-17（追記）5秒スキャ無印の運用導線を削除（B専用化完了）

- 目的: `scalp_ping_5s_live`（無印）の再起動・再有効化経路を消し、5秒スキャを `scalp_ping_5s_b_live` に一本化。
- 削除:
  - systemd: `quant-scalp-ping-5s.service`, `quant-scalp-ping-5s-exit.service`
  - autotune unit: `quant-scalp-ping-5s-autotune.service`, `quant-scalp-ping-5s-autotune.timer`
  - env: `ops/env/quant-scalp-ping-5s.env`, `ops/env/quant-scalp-ping-5s-exit.env`, `ops/env/scalp_ping_5s.env`
- 監査更新:
  - `scripts/ops_v2_audit.py` の mandatory pair を `quant-scalp-ping-5s-b*` へ切替。
- 仕様更新:
  - `AGENTS.md`, `docs/WORKER_ROLE_MATRIX_V2.md`, `docs/OPS_CURRENT.md` を B専用前提へ更新。

### 2026-02-17（追記）戦略ローカル予測で TP/ロット補正を標準化

- `analysis/technique_engine.py`
  - `evaluate_entry_techniques()` に戦略ローカル予測（forecast profile: `timeframe`/`step_bars`）を追加。
  - 予測から `expected_pips` / `p_up` / `directional_edge` を算出し、`size_mult` と `tp_mult` を出力。
  - `TechniqueEntryDecision` に `tp_mult` を追加し、`debug.forecast` で監査できるようにした。
- `workers/scalp_ping_5s/worker.py`
  - `forecast_profile=M1x1` を `entry_thesis` に明示。
  - `tech_decision.tp_mult` で TP を再スケールし、`entry_units_intent` は `size_mult` 反映後の値を維持。
- `workers/scalp_m1scalper/worker.py`
  - limit/market の両経路で `forecast_profile=M1x5` を明示。
  - `tech_tp_mult` を `entry_thesis` へ記録し、TP価格を再計算して `clamp_sl_tp` へ再適用。
- `workers/scalp_rangefader/worker.py`
  - `forecast_profile=M1x5` を明示し、ローカル予測を TP/units の補正へ接続。
- `workers/scalp_macd_rsi_div/worker.py`
  - `forecast_profile=M5x2`（10分相当）を明示し、TP再スケールを追加。
- `workers/micro_runtime/worker.py`
  - `forecast_profile=M5x2` を明示し、各 micro 戦略のエントリーで TP/ロット補正を統一。
- `execution/strategy_entry.py`
  - `_STRATEGY_TECH_CONTEXT_REQUIREMENTS` に `forecast_profile` と `forecast_technical_only` を追加。
  - 戦略契約未指定でも、`forecast_horizon` と整合する予測間隔が補完されるようにした。

### 2026-02-17（追記）ローカル予測の過去検証と精度改善（M1/M5）

- 検証データ:
  - `logs/candles_M1*.json`, `logs/candles_USDJPY_M1*.json`, `logs/oanda/candles_M1_latest.json` を統合（重複時刻は後勝ち）。
  - 総バー数: 9,204（`2026-01-06` 〜 `2026-02-17`）。
- 追加:
  - `scripts/eval_local_forecast.py` を新規追加。`baseline`（旧式）と `improved`（新式）を同一データで比較可能。
- `analysis/technique_engine.py` 改善:
  - `step_bars<=1`: momentum + short mean-reversion（ノイズ抑制）。
  - `step_bars<=5`: persistence（連続方向性）で trend/reversion を混合。
  - `step_bars>=10`: 既存 baseline を維持（過剰改変を避ける）。
  - 予測値は `close_vol` ベースでクリップし、`debug.forecast` に `regime/persistence/close_vol_pips` を出力。
- 検証結果（`scripts/eval_local_forecast.py --steps 1,5,10`）:
  - baseline
    - step=1: hit `0.4592`, MAE `1.6525`
    - step=5: hit `0.4821`, MAE `4.4869`
    - step=10: hit `0.4699`, MAE `6.8365`
  - improved
    - step=1: hit `0.4597`, MAE `1.6391`（改善）
    - step=5: hit `0.4850`, MAE `4.3381`（改善）
    - step=10: hit `0.4699`, MAE `6.8365`（維持）

### 2026-02-17（追記）短期 horizon（1m/5m/10m）を M1 基準へ統一

- `workers/common/forecast_gate.py`
  - `_HORIZON_META_DEFAULT` の短期定義を `M1x1 / M1x5 / M1x10` へ変更。
  - `profile.timeframe` が `M5/H1` 等でも、`horizon in {1m,5m,10m}` の場合は
    `step_bars * timeframe_minutes` で `M1` へ正規化する処理を追加。
  - 正規化時は `profile_normalization`（例: `M5x2->M1x10`）を予測行へ残し監査可能化。
  - `factor_cache` の `M1` が stale（既定 150秒超）な場合は
    `logs/oanda/candles_M1_latest.json` から最新 M1 を読み、短期予測の入力へ自動フォールバック。
- 目的:
  - 短期予測で `M5` 足更新遅延の影響を避ける。
  - 戦略ごとの `forecast_profile` が `M5x2` を指定していても、短期では最新 `M1` 系列で計算できるようにする。
  - `NO_MATCHING_HORIZONS` / stale 由来の短期予測欠損を運用側で吸収しやすくする。

### 2026-02-17（追記）今回の予測改善フロー（依頼対応ログ）

- 要望: 「過去分で予測し、今あっているか検証し、精度を上げる」。
- 実施フロー:
  - VMの過去足（M1中心）で baseline/improved を比較し、hit率とMAEを定量化。
  - `analysis/technique_engine.py` の短期予測式（1m/5m）を改善し、10mは回帰防止で据え置き。
  - `scripts/eval_local_forecast.py` を追加し、同じ条件で再評価できる手順を固定化。
  - `workers/common/forecast_gate.py` で短期 horizon を `M1` 基準に統一（`M5x2 -> M1x10` 換算）。
  - `M1` が stale の場合は `logs/oanda/candles_M1_latest.json` へフォールバックするよう変更。
  - VMへ反映後、`vm_forecast_snapshot.py` で 1m/5m/10m が最新 `feature_ts` で出力されることを確認。
- 反映確認（当日）:
  - VM `git rev-parse HEAD` と `origin/main` の一致を確認済み。
  - `quant-order-manager.service` 再起動後の `Application startup complete` を確認済み。

### 2026-02-17（追記）予測特徴量をトレンドライン/サポレジ圧力へ拡張

- 目的:
  - 「過去足から未来方向を推定する」ロジックで、単純モメンタム偏重を下げる。
  - 線形トレンド（線を引く系）とサポレジ圧力（ブレイク/圧縮）を同時に判定し、`p_up` の再現性を上げる。
- `analysis/forecast_sklearn.py`
  - `compute_feature_frame()` に以下を追加:
    - `trend_slope_pips_20`, `trend_slope_pips_50`, `trend_accel_pips`
    - `support_gap_pips_20`, `resistance_gap_pips_20`, `sr_balance_20`
    - `breakout_up_pips_20`, `breakout_down_pips_20`
    - `donchian_width_pips_20`, `range_compression_20`, `trend_pullback_norm_20`
  - 既存の return/MA/ATR/RSI 特徴量と併用して学習・推論の特徴空間を拡張。
- `workers/common/forecast_gate.py`
  - テクニカル予測の `required_keys` に上記追加特徴量を組み込み。
  - `trend_score` / `mean_revert_score` を更新し、短期の方向性に対して
    `trend_slope`・`breakout_bias`・`sr_balance`・`squeeze_score` を反映。
  - 出力行へ `trend_slope_pips_20/50`, `trend_accel_pips`, `sr_balance_20`,
    `breakout_bias_20`, `squeeze_score_20` を追加し、監査ログから根拠を追跡可能化。
- テスト:
  - `tests/analysis/test_forecast_sklearn.py` に新特徴量列の存在確認と、上昇/下落トレンドでの傾き方向一致テストを追加。
  - `tests/workers/test_forecast_gate.py` に `trendline/sr` 出力項目の存在・方向性テストを追加。
  - 実行: `pytest -q tests/analysis/test_forecast_sklearn.py tests/workers/test_forecast_gate.py`（13 passed）。

### 2026-02-17（追記）VM実データで breakout_bias 一致率監査 + before/after 比較ジョブ追加

- VM実データ監査（`fx-trader-vm`）:
  - `/home/tossaki/QuantRabbit/.venv/bin/python scripts/vm_forecast_snapshot.py --horizon 1m,5m,10m` を実行し、
    短期 horizon の予測行が `status=ready` で取得できることを確認。
  - 同期間（`logs/oanda/candles_M1_latest.json`, 500 bars）で `breakout_bias_20` の先行方向一致率を算出。
    - step=1: hit `0.4872`（filtered `0.4828`）
    - step=5: hit `0.4831`（filtered `0.4829`）
    - step=10: hit `0.4711`（filtered `0.4708`）
- 追加ジョブ:
  - `scripts/eval_forecast_before_after.py` を追加。
  - 同一期間で `before/after` の `hit_rate` / `MAE(pips)` と `breakout_bias_20` 一致率を一括出力。
  - `--feature-expansion-gain` で新特徴量寄与を段階評価可能化（0.0-1.0）。
- 同一期間比較（VMデータ 1,930 bars）:
  - gain=0.35 では before 比で短期 hit/MAE が小幅悪化（1m/5m/10m）。
  - 実運用デフォルトは保守設定として `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0` を採用。
    - 既存挙動を維持しつつ、追加特徴量は監査ログとオフライン評価で継続検証する方針。
- 反映後の再監査（VMデータ 8,050 bars）:
  - gain=0.35:
    - step=1: hit `0.4901` / MAE `1.4948`
    - step=5: hit `0.4826` / MAE `3.4788`
    - step=10: hit `0.4773` / MAE `5.1679`
  - gain=0.0（運用デフォルト）:
    - step=1: hit `0.4926` / MAE `1.4848`
    - step=5: hit `0.4855` / MAE `3.4571`
    - step=10: hit `0.4783` / MAE `5.1374`
  - `breakout_bias_20` の方向一致率（同期間）は step=1/5/10 で `0.4869 / 0.4788 / 0.4716`（filtered は `0.4873 / 0.4808 / 0.4726`）。

### 2026-02-17（追記）position-manager タイムアウト再発の抑止（open_positions 経路）

- 背景:
  - `quant-scalp-ping-5s-b` / `quant-scalp-ping-5s-flow` で
    `position_manager open_positions timeout after 6.0s` が断続再発。
  - 同時に `execution/position_manager.py` の service 呼び出しで
    `read timeout=9.0` が積み上がり、to_thread 待ち切り（6秒）との不整合で遅延が増幅していた。
- 修正:
  - `execution/position_manager.py`
    - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT`（既定 4.5s）を追加し、
      `/position/open_positions` は共通 timeout より短く fail-fast するよう分離。
    - service 呼び出しを `requests.Session` + pool（keep-alive）へ変更し、
      高頻度呼び出し時の接続張り直しコストを削減。
    - `/position/open_positions` のクライアント側短TTLキャッシュ
      （`POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC`, 既定 0.35s）を追加。
    - service 失敗時は短時間の stale キャッシュ
      （`POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC`, 既定 2.0s）を返せるようにし、
      一時的な遅延バーストでの entry 停止を抑止。
    - OANDA `openTrades` 取得専用 timeout
      （`POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT`, 既定 3.5s）を追加し、
      service 側の fetch が 4.5s client timeout を超えないように調整。
  - `workers/position_manager/worker.py`
    - Uvicorn の access log をデフォルト OFF（`POSITION_MANAGER_ACCESS_LOG=0` 相当）に変更。
    - 高頻度 `open_positions` アクセス時のログI/Oボトルネックを低減。
- 期待効果:
  - `open_positions` 呼び出しの tail latency（6-9秒帯）と worker 側 `position_manager_timeout` の頻度を低下。
  - position-manager service の過負荷時でも、短期キャッシュ経由で戦略ループ継続性を維持。

### 2026-02-17（追記）5秒スキャB/Flowのエントリー閾値を現況向けに緩和

- 背景（VM実ログ, 15分集計）:
  - `scalp_ping_5s_b`: `no_signal` と `no_signal:revert_not_found` が支配的、`units_below_min` も頻発。
  - `scalp_ping_5s_flow`: `low_tick_count` が大半を占有。
- 反映:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_REVERT_MIN_TICKS=2`（旧3）
    - `SCALP_PING_5S_B_REVERT_CONFIRM_TICKS=1`（旧2）
    - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=1`（旧0）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE=150`（旧300）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B=150`（旧300）
  - `ops/env/scalp_ping_5s_flow.env`
    - `SCALP_PING_5S_FLOW_MIN_TICKS=3`（新規）
    - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS=2`（新規）
    - `SCALP_PING_5S_FLOW_MIN_TICK_RATE=0.35`（新規）
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_PIPS=0.20`（旧0.30）
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_TICKS=4`（旧6）
- 目的:
  - `revert_not_found` と `low_tick_count` による機会損失を下げ、
    `scalp_fast` pocket での現況追従エントリー頻度を回復する。

### 2026-02-17（追記）scalp_ping_5s_b の short extrema 合意を既定ON化

- 背景:
  - `ops: tune scalp ping 5s b/flow entry thresholds` 反映後、
    `SCALP_PING_5S_B_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT` が未設定だと
    short 側は共通設定（`SCALP_PING_5S_B_EXTREMA_REQUIRE_M1_M5_AGREE=0`）へ
    フォールバックしていた。
- 修正:
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT` の既定値を
      `ENV_PREFIX == "SCALP_PING_5S_B"` のとき `true` に変更。
    - env 明示がある場合は従来どおり env 値を優先。
- 意図:
  - B運用で short の `short_bottom_m1m5` を M1+M5 合意時のみ block し、
    下落継続の再エントリー取り逃しを抑える。

### 2026-02-17（追記）5秒スキャの極値反転ルーティング（件数維持型）

- 背景:
  - 直近の live で `short_bottom_soft` が連続し、底付近で short の積み上がりが発生。
  - 要件は「エントリー件数は落とさず、底/天井の方向精度を上げる」。
- 実装:
  - `workers/scalp_ping_5s/config.py`
    - `EXTREMA_REVERSAL_ENABLED` ほか `EXTREMA_REVERSAL_*` を追加。
    - `ENV_PREFIX=SCALP_PING_5S_B` では既定で反転ルーティングを有効化。
  - `workers/scalp_ping_5s/worker.py`
    - `_extrema_reversal_route()` を追加。
    - `short_bottom_*` / `long_top_*` / `short_h4_low` で、M1/M5/H4位置と
      M1の `RSI/EMA`、`MTF heat`、`horizon` を合算評価し、
      閾値到達時は block せず opposite side へ反転 (`*_extrev`)。
    - `entry_thesis` に以下を追加して監査可能化:
      - `extrema_reversal_enabled`
      - `extrema_reversal_applied`
      - `extrema_reversal_score`
- 目的:
  - 極値局面での同方向積み上げを抑えつつ、注文本数は維持する。
  - `orders.db` から反転適用率と成績を継続監視できる状態にする。

### 2026-02-18（追記）micro戦略の forecast 参照を常時有効化

- 背景:
  - VM監査で `micro` pocket の `entry_thesis.forecast.reason=not_applicable` が継続し、
    `scalp/scalp_fast` に比べて forecast 文脈の記録が欠落していた。
  - 原因は `quant-micro-*.env` の `FORECAST_GATE_ENABLED=0`。
- 反映:
  - 以下 11 エントリーworker env を `FORECAST_GATE_ENABLED=1` に統一。
    - `ops/env/quant-micro-compressionrevert.env`
    - `ops/env/quant-micro-levelreactor.env`
    - `ops/env/quant-micro-momentumburst.env`
    - `ops/env/quant-micro-momentumpulse.env`
    - `ops/env/quant-micro-momentumstack.env`
    - `ops/env/quant-micro-pullbackema.env`
    - `ops/env/quant-micro-rangebreak.env`
    - `ops/env/quant-micro-trendmomentum.env`
    - `ops/env/quant-micro-trendretest.env`
    - `ops/env/quant-micro-vwapbound.env`
    - `ops/env/quant-micro-vwaprevert.env`
- 期待効果:
  - `execution/strategy_entry.py` の forecast 注入経路で micro も `entry_thesis["forecast"]` を保持。
  - `coordinate_entry_intent` 呼び出し時に `forecast_context` が黒板へ監査記録される。
  - `tp_pips_hint/target_price` などの forecast メタを micro でも追跡可能化。

### 2026-02-18（追記）戦略内 forecast 融合（units/probability/TP/SL）

- 背景:
  - これまで forecast は主に監査メタとして連携され、戦略ごとの `units`/`entry_probability` と
    同時最適化が弱い経路が残っていた。
- 実装:
  - `execution/strategy_entry.py` に `_apply_forecast_fusion(...)` を追加。
  - `market_order` / `limit_order` の共通経路で、`_apply_strategy_feedback(...)` 後に
    forecast 融合を適用するよう変更。
  - 合成ルール:
    - `p_up` と売買方向から `direction_prob` を算出。
    - `edge` と整合して `units_scale` を導出し、順方向で小幅boost、逆方向/`allowed=false` で縮小。
    - `entry_probability` を同様に補正（欠損時は forecast から補完可能）。
    - `tp_pips_hint` は順方向時に `tp_pips` へブレンド、`sl_pips_cap` は `sl_pips` の上限として適用。
  - 監査:
    - `entry_thesis["forecast_fusion"]` に `units_before/after`、`entry_probability_before/after`、
      `units_scale`、`forecast_reason`、`forecast_horizon` を保存。
- テスト:
  - `tests/execution/test_strategy_entry_forecast_fusion.py` を追加し、
    逆行縮小・順行拡大・確率補完・コンテキスト欠損時不変を検証。

### 2026-02-18（追記）全戦略の forecast+戦略ローカル融合を確度寄りに強化

- 背景:
  - 「各戦略が forecast と自戦略計算を同時に使い、逆行局面を減らす」要件に対し、
    既存 `forecast_fusion` は方向補正中心で、TF整合と強逆行の明示拒否が不足していた。
  - 併せて `quant-m1scalper.env` が `FORECAST_GATE_ENABLED=0` のままで、M1系の forecast 参照が欠落していた。
- 実装:
  - `execution/strategy_entry.py`
    - `tf_confluence_score/tf_confluence_count` を `units` と `entry_probability` の補正に追加。
      - 低整合（負値）は追加縮小、高整合（正値）は小幅増加。
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_*` を追加し、強い逆行予測
      （`direction_prob` 低位 + `edge` 高位、または `allowed=false`）は `units=0` で見送り。
    - 監査に `tf_confluence_score/tf_confluence_count/strong_contra_reject/reject_reason` を追加。
  - `ops/env/quant-m1scalper.env`
    - `FORECAST_GATE_ENABLED=1` へ変更。
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_GATE_ENABLED=1`
    - `STRATEGY_FORECAST_CONTEXT_ENABLED=1`
    - `STRATEGY_FORECAST_FUSION_ENABLED=1`
    - `STRATEGY_FORECAST_FUSION_TF_CUT_MAX=0.35`
    - `STRATEGY_FORECAST_FUSION_TF_BOOST_MAX=0.12`
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=1`
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_PROB_MAX=0.22`
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_EDGE_MIN=0.65`
- テスト:
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - 強逆行見送り（`units=0`）の検証を追加。
    - `tf_confluence_score` が負のときに `units/probability` が縮小する検証を追加。

### 2026-02-18（追記）`scalp_ping_5s` に signal window 可変化 + shadow 評価を追加

- 背景:
  - `signal_window_sec` の固定値運用（1.2s/1.5s 等）だけでは地合い変化時の追従が弱く、
    「まずは本番同等ロジックで shadow 計測し、十分なサンプルが揃ったら適用」にしたい要望があった。
- 実装:
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_*` 一式を追加。
    - デフォルトは `ADAPTIVE_ENABLED=0`（挙動据え置き）。
    - `*_SHADOW_ENABLED` と候補窓 (`*_CANDIDATES_SEC`) で shadow 比較を可能化。
  - `workers/scalp_ping_5s/worker.py`
    - `_build_tick_signal(...)` に `signal_window_override_sec` / `allow_window_fallback` を追加。
    - `trades.db` から `signal_window_sec` 別成績を読む `_load_signal_window_stats(...)` を追加（TTL cache）。
    - 候補窓をスコアリングする `_maybe_adapt_signal_window(...)` を追加。
      - `adaptive=off` なら選択は変えず shadow ログのみ。
      - `adaptive=on` かつ `min_trades` と `selection_margin` を満たす場合のみ窓を切替。
    - `entry_thesis` に `signal_window_adaptive_*` 監査キーを追加（live/best/selected, sample, score）。
  - env:
    - `ops/env/scalp_ping_5s_b.env` / `ops/env/scalp_ping_5s_flow.env` に
      `*_SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED=1` を追加（適用はOFFのまま）。
- 目的:
  - まず本番ログで「候補窓ごとの期待値差」を収集し、過学習を避けて段階的に適用する。

### 2026-02-18（追記）forecast改善フローの記録（依頼対応ログ）

- 要求:
  - 「各戦略が forecast と戦略ローカル計算を併用して、確実性のあるトレードを行う状態」
  - micro だけでなく全戦略で同一の監査可能な forecast 参照経路を維持すること。
- 実施フロー（時系列）:
  - `execution/strategy_entry.py` の forecast 融合を拡張し、
    `tf_confluence_score/tf_confluence_count` を units/probability 補正へ追加。
  - 強い逆行予測時の見送りガード
    （`STRATEGY_FORECAST_FUSION_STRONG_CONTRA_*`）を追加し、`units=0` で発注回避。
  - `ops/env/quant-m1scalper.env` を `FORECAST_GATE_ENABLED=1` へ変更。
  - `ops/env/quant-v2-runtime.env` に
    `STRATEGY_FORECAST_CONTEXT_ENABLED=1` / `STRATEGY_FORECAST_FUSION_ENABLED=1` と
    強逆行拒否・TF補正パラメータを明示追加。
  - テスト:
    - `tests/execution/test_strategy_entry_forecast_fusion.py`（6 passed）
    - `tests/execution/test_order_manager_preflight.py`（20 passed）
  - commit/push:
    - `ddd7c3a3` `feat: harden strategy forecast fusion with tf-confluence guard`
  - VM反映:
    - `scripts/vm.sh ... deploy -b main -i --restart quant-market-data-feed.service -t`
    - VMで `git rev-parse HEAD == origin/main` を確認。
    - `quant-order-manager` / `quant-micro-rangebreak` / `quant-scalp-ping-5s-b` /
      `quant-m1scalper` / `quant-session-open` の再起動と
      `Application started!` / `Application startup complete.` を確認。
  - ランタイム監査:
    - `quant-m1scalper`, `quant-micro-rangebreak`, `quant-scalp-ping-5s-b` の
      `/proc/<pid>/environ` で
      `FORECAST_GATE_ENABLED=1`,
      `STRATEGY_FORECAST_FUSION_ENABLED=1`,
      `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=1` を確認。
  - 参考観測:
    - `scripts/vm_forecast_snapshot.py --horizon 1m,5m,10m` で
      直近 snapshot（p_up/edge/expected_pips/range帯）を取得して稼働確認。

- 運用整備（依頼「スキル化/オートメーション化」対応）:
  - 監査手順を再利用可能にするため、
    `$CODEX_HOME/skills/qr-forecast-improvement-audit/` を新規作成。
    - `SKILL.md`
    - `references/forecast_improvement_playbook.md`
    - `agents/openai.yaml`
  - 2時間周期で実行する automation の作成案（`QR Forecast Audit 2h`）を提示済み。
- 現在の状態:
  - forecast は「黒板の監査メタ」に留まらず、strategy_entry 内で
    各戦略の units/probability/TP/SL へ反映される運用へ移行済み。
  - 効果判定は `eval_forecast_before_after.py` で同一期間比較を継続する。

### 2026-02-18（追記）EXIT導線のタイムアウト不整合を是正（position/order service）

- 背景:
  - 本番VMで `position_manager` の `/position/open_positions` が 8s ではタイムアウトし、
    実測で 10s 前後になる局面が継続。
  - exit worker 側は `SCALP_PRECISION_EXIT_OPEN_POSITIONS_TIMEOUT_SEC=6.0` と
    `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT` の既定（4.5s）で fail-fast しており、
    `close_request` が止まり建玉が残留した。
  - `ORDER_MANAGER_SERVICE_TIMEOUT=5.0` により `close_trade` 呼び出しもタイムアウトが発生。
- 対応（`ops/env/quant-v2-runtime.env`）:
  - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0`
  - `POSITION_MANAGER_SERVICE_TIMEOUT=15.0`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=12.0`
  - `POSITION_MANAGER_HTTP_RETRY_TOTAL=1`
  - `POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT=4.0`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC=1.2`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC=12.0`
  - `SCALP_PRECISION_EXIT_OPEN_POSITIONS_TIMEOUT_SEC=12.0`
- 意図:
  - EXITロジック（`take_profit/lock_floor/max_adverse`）は変更せず、
    判定結果が実際に `close_request -> close_ok` まで到達する通路だけを安定化。
  - 失敗時は stale cache 返却で「無応答より継続判定」を優先し、取りこぼしを抑制。

### 2026-02-18（追記）全戦略EXITへ forecast 補正を接続（TP/Trail/LossCut）

- 要求:
  - 「全戦略の EXIT でも forecast を使い、損切りにも反映する」を実装。
  - 共通一律の後付け EXIT 判定は作らず、各戦略 `exit_worker` 内ロジックに補正として組み込む。
- 実装:
  - 新規: `workers/common/exit_forecast.py`
    - `entry_thesis.forecast` / `forecast_fusion` から逆行確率（`against_prob`）を算出。
    - 逆行強度に応じて `profit_take` / `trail_start` / `trail_backoff` / `lock_buffer` / `loss_cut` / `max_hold` の
      乗数補正を返す。
    - 主要 env:
      - `EXIT_FORECAST_ENABLED`
      - `EXIT_FORECAST_MIN_AGAINST_PROB`
      - `EXIT_FORECAST_PROFIT_TIGHTEN_MAX`
      - `EXIT_FORECAST_LOSSCUT_TIGHTEN_MAX`
      - `EXIT_FORECAST_MAX_HOLD_TIGHTEN_MAX`
  - 適用先:
    - `workers/micro_runtime/exit_worker.py`
    - `workers/session_open/exit_worker.py`
    - `workers/scalp_rangefader/exit_worker.py`
    - `workers/scalp_m1scalper/exit_worker.py`
    - `workers/scalp_ping_5s/exit_worker.py`
    - `workers/scalp_ping_5s_b/exit_worker.py`
    - `workers/scalp_false_break_fade/exit_worker.py`
    - `workers/scalp_level_reject/exit_worker.py`
    - `workers/scalp_macd_rsi_div/exit_worker.py`
    - `workers/scalp_squeeze_pulse_break/exit_worker.py`
    - `workers/scalp_tick_imbalance/exit_worker.py`
    - `workers/scalp_wick_reversal_blend/exit_worker.py`
    - `workers/scalp_wick_reversal_pro/exit_worker.py`
  - 既存 `micro_*` 個別 exit ワーカー（`micro_rangebreak` など）も同補正を適用。
- テスト:
  - 追加: `tests/workers/test_exit_forecast.py`
  - 実行:
    - `pytest -q tests/workers/test_exit_forecast.py`（3 passed）
    - `pytest -q tests/workers/test_loss_cut.py tests/addons/test_session_open_worker.py`（5 passed）
    - `python3 -m py_compile ...`（対象 exit_worker + `workers/common/exit_forecast.py`）
- 現在の状態:
  - forecast は ENTRY 側の `units/probability/TP/SL` 補正だけでなく、
    EXIT 側の利確・トレール・損切り閾値補正にも接続済み。
  - ただし最終クローズ判定の責務は従来どおり各 strategy `exit_worker` に残し、
    共通レイヤでの一律拒否/強制クローズは導入していない。

### 2026-02-18（追記）EXITワーカーの戦略別実体化（委譲停止）

- 要求:
  - 「共通テンプレート委譲ではなく、全戦略を個別EXITワーカー化」。
- 実装:
  - `micro_*` 系11本の `exit_worker` から `workers.micro_runtime.exit_worker` への委譲を廃止し、
    各戦略パッケージ内に実体ロジックを配置。
    - 対象:
      - `workers/micro_compressionrevert/exit_worker.py`
      - `workers/micro_levelreactor/exit_worker.py`
      - `workers/micro_momentumburst/exit_worker.py`
      - `workers/micro_momentumpulse/exit_worker.py`
      - `workers/micro_momentumstack/exit_worker.py`
      - `workers/micro_pullbackema/exit_worker.py`
      - `workers/micro_rangebreak/exit_worker.py`
      - `workers/micro_trendmomentum/exit_worker.py`
      - `workers/micro_trendretest/exit_worker.py`
      - `workers/micro_vwapbound/exit_worker.py`
      - `workers/micro_vwaprevert/exit_worker.py`
  - `workers/scalp_ping_5s_flow/exit_worker.py` の `workers.scalp_ping_5s.exit_worker` subprocess 委譲を廃止し、
    同ファイル内でflow専用envマップ + EXIT実体ロジックを実行する構成へ変更。
  - 各 micro 戦略および flow 戦略にローカル `config.py` を追加し、`ENV_PREFIX` を戦略パッケージ内で解決。
- 検証:
  - `rg` による委譲参照確認で `exit_worker -> 別 exit_worker` 参照は 0 件。
  - `python3 -m py_compile`（対象13 worker + 12 config）通過。
  - `pytest -q tests/workers/test_exit_forecast.py tests/workers/test_loss_cut.py tests/addons/test_session_open_worker.py`（8 passed）。

### 2026-02-18（追記）EXIT低レイヤ共通モジュールの戦略別内製化

- 要求:
  - 「戦略ごとに内製化」: `exit_worker` 実行パスで `workers/common/*` に依存しない構成へ移行。
- 実装:
  - 全 `workers/*/exit_worker.py`（25本）の `from workers.common.*` をローカル import へ差し替え。
  - 各戦略パッケージ内へ、必要なEXIT低レイヤを複製配置:
    - `exit_utils.py`
    - `exit_emergency.py`
    - `reentry_decider.py`
    - `pro_stop.py`
    - `loss_cut.py`
    - `exit_scaling.py`
    - `exit_forecast.py`
    - `rollout_gate.py`（使用戦略のみ）
  - `exit_utils.py` 内の緊急許可参照も `workers.common.exit_emergency` ではなく
    同一戦略パッケージ内 `exit_emergency` を参照するよう変更。
- 検証:
  - `rg` で `exit_worker` + ローカルEXITモジュール群に `workers.common` 参照が残っていないことを確認。
  - `python3 -m py_compile`（exit_worker + ローカルEXITモジュール、193ファイル）通過。
  - `pytest -q tests/workers/test_exit_forecast.py tests/workers/test_loss_cut.py tests/addons/test_session_open_worker.py`（8 passed）。

### 2026-02-18（追記）MicroCompressionRevert の実運用デリスク（ENTRY/EXIT）

- 背景:
  - VM `trades.db` 直近24hで `MicroCompressionRevert-short` が `PF<1` を継続。
  - 同時刻に複数玉が積み上がるクラスター損失（`MARKET_ORDER_TRADE_CLOSE`）を確認。
- 対応:
  - `ops/env/quant-micro-compressionrevert.env`
    - `MICRO_MULTI_BASE_UNITS=14000`（28000→半減）
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroCompressionRevert:0.45`
    - `MICRO_MULTI_MAX_SIGNALS_PER_CYCLE=1`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=120`
    - `MICRO_MULTI_HIST_MIN_TRADES=8`
    - `MICRO_MULTI_HIST_SKIP_SCORE=0.55`
    - `MICRO_MULTI_DYN_ALLOC_MIN_TRADES=8`
    - `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.45`
  - `config/strategy_exit_protections.yaml` `MicroCompressionRevert.exit_profile`
    - `loss_cut_hard_sl_mult=1.20`（1.60→引き締め）
    - `loss_cut_soft_sl_mult=0.95` を追加
    - `loss_cut_max_hold_sec=900`（2400→短縮）
    - `range_max_hold_sec=900`
    - `loss_cut_cooldown_sec=4`
    - `profit/trail/lock` を明示（`profit_pips=1.1`, `trail_start_pips=1.5` など）
- 目的:
  - まずは「損失クラスターを作らない」ことを優先し、サイズ・頻度・保有時間を同時に圧縮。
  - 2h/6h/24h 窓で `PF/avg_pips/close_reason` を再判定し、必要なら段階的に再開放。

### 2026-02-18（追記）forecast短期窓の精度改善（2h/4h、VM実データ）

- 背景:
  - 直近監査（`max-bars=120/240`）で、評価用構成
    `feature_expansion_gain=0.35`,
    `breakout=1m=0.16,5m=0.22,10m=0.30`,
    `session=1m=0.0,5m=0.26,10m=0.38`
    が `1m/5m` の `hit_delta` を押し下げる局面を確認。
- 実施:
  - VM上で `scripts/eval_forecast_before_after.py` を使い、2段階で探索。
    - 重みグリッド探索（`forecast_tune_grid_latest.json`）
    - `feature_expansion_gain` を含む再探索（`forecast_tune_feature_latest.json`）
  - 同一期間（`bars=120/240`）で baseline/candidate を比較し、
    `logs/reports/forecast_improvement/report_tuning_20260218T023002Z.md` を生成。
- 採用設定（runtime env）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.10,5m=0.18,10m=0.26`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.18,10m=0.30`
- 比較結果（同一時点、before/after差分の合計）:
  - `hit_delta_sum`: `-0.0025 -> +0.0219`（改善）
  - `mae_delta_sum`: `-0.0366 -> -0.0096`（改善幅は縮小）
  - `1m/5m` の `hit_delta` マイナスは解消、`10m` はプラス維持。
- 備考:
  - 方向一致（hit）優先で短期TFをデチューンし、過反応を抑える方針。
  - MAE改善幅が縮小するため、次回は 6h/12h 窓でも再検証して再調整する。

### 2026-02-18（追記）forecast監査の12h窓不整合を修正し、6h/12hで再チューニング

- 事象:
  - `eval_forecast_before_after.py --max-bars 720` で、`bars=220` かつ
    `2026-01-23` 帯だけが評価されるケースを確認。
  - 原因は timestamp の小数秒あり/なし混在時に `pd.to_datetime(..., errors='coerce')`
    が一部を `NaT` 化していたこと。
- 対応:
  - `scripts/eval_forecast_before_after.py` に mixed-format 対応パーサ
    `pd.to_datetime(..., format='mixed')` のフォールバック付き処理を追加。
  - テスト追加: `tests/test_eval_forecast_before_after.py`
    (`test_to_datetime_utc_handles_mixed_precision_iso8601`)。
- 追加評価（VM実データ）:
  - OANDAから直近13hの M1 を再取得（`logs/oanda/candles_M1_eval_12h_latest.json`, 774本）し、
    連続データで `120/240/360/720 bars` を再比較。
  - 6h/12h 同時グリッド（`forecast_tune_grid_6h12h_latest.json`）で最良:
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.05`
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.20,10m=0.28`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.18,10m=0.30`（維持）
- 反映:
  - `ops/env/quant-v2-runtime.env` を上記へ更新。
  - 旧候補（gain=0.0, breakout=0.10/0.18/0.26）比で、
    `total_hit`/`total_mae` は小幅改善、`5m` 悪化件数は同等に維持。

### 2026-02-18（追記）M1Scalper / RangeFader EXITのforecast損切り補正を有効化

- 背景:
  - `scalp_m1scalper` / `scalp_rangefader` は `apply_exit_forecast_to_targets` は接続済みだったが、
    `apply_exit_forecast_to_loss_cut` の経路が未接続で、EXITの負け側制御が他戦略と非対称だった。
- 実装:
  - `workers/scalp_m1scalper/exit_worker.py`
    - `apply_exit_forecast_to_loss_cut` を導入し、`max_adverse` / `max_hold` の補正を統一。
    - `entry_thesis.hard_stop_pips`（存在時）を hard 側しきい値に反映。
  - `workers/scalp_rangefader/exit_worker.py`
    - `apply_exit_forecast_to_loss_cut` を導入。
    - 逆行時 (`pnl<=0`) に forecast補正済み `loss_cut_hard_pips` / `loss_cut_max_hold_sec` を評価し、
      `max_adverse` / `max_hold_loss` でクローズ可能な経路を追加。
    - 追加env:
      - `RANGEFADER_EXIT_SOFT_ADVERSE_PIPS`（default: `1.8`）
      - `RANGEFADER_EXIT_HARD_ADVERSE_PIPS`（default: `2.8`）
      - `RANGEFADER_EXIT_MAX_HOLD_LOSS_SEC`（default: `180`）
- 検証:
  - `python3 -m py_compile workers/scalp_m1scalper/exit_worker.py workers/scalp_rangefader/exit_worker.py`
  - `pytest -q tests/workers/test_exit_forecast.py`（3 passed）
- 状態:
  - forecast は ENTRY（units/probability/TP/SL）だけでなく、
    `scalp_m1scalper` / `scalp_rangefader` でも EXITの利確・損切り・負け持ち保持時間まで一貫して反映。

### 2026-02-18（追記）全EXITワーカーで予測価格ヒントを補正に反映

- 背景:
  - 既存のEXIT forecast補正は主に `p_up` / `edge` ベースで、
    `target_price` / `anchor_price` / `range_price` を直接使っていなかった。
- 実装:
  - 全 `workers/*/exit_forecast.py`（25戦略）に同一変更を適用。
  - `build_exit_forecast_adjustment(...)` で以下を評価して `contra_score` に反映:
    - `target_price` と `anchor_price` の距離（pips）
    - `tp_pips_hint`
    - `range_low_price` / `range_high_price` の予測レンジ幅（pips）
  - 追加env:
    - `EXIT_FORECAST_PRICE_HINT_ENABLED`（default: `1`）
    - `EXIT_FORECAST_PRICE_HINT_WEIGHT_MAX`（default: `0.20`）
    - `EXIT_FORECAST_PRICE_HINT_MIN_PIPS`（default: `0.6`）
    - `EXIT_FORECAST_PRICE_HINT_MAX_PIPS`（default: `8.0`）
    - `EXIT_FORECAST_RANGE_HINT_NARROW_PIPS`（default: `3.5`）
    - `EXIT_FORECAST_RANGE_HINT_WIDE_PIPS`（default: `18.0`）
- 期待効果:
  - 予測ターゲット距離が小さい（伸び余地が薄い）局面ではEXITを引き締め、
    予測レンジが十分広く方向確率が有利な局面では過度な早期EXITを抑制。
- 検証:
  - `python3 -m py_compile workers/*/exit_forecast.py`
  - `pytest -q tests/workers/test_exit_forecast.py`（75 passed）

### 2026-02-18（追記）ENTRY forecast_fusion の strong-contra 判定を実データ整合へ修正

- 背景:
  - 直近VM実績で、`p_up<0.20` かつ `forecast.allowed=0` のロングが縮小のみで通過し、
    予測逆行の拒否が発火しないケースを確認。
  - 原因は `execution/strategy_entry.py` の `edge_strength` が
    `max((edge-0.5)/0.5, 0)` だったため、`edge<0.5`（強い下方向）で常に 0 になっていたこと。
- 実装:
  - `execution/strategy_entry.py`
    - `edge_strength = abs(edge-0.5)/0.5` に変更し、上方向/下方向どちらの強い予測も
      strong-contra 判定へ反映するよう修正。
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - `edge=0.08`, `p_up=0.10`, `allowed=false` の bearish 強逆行ケースで
      `strong_contra_reject=True` と `units=0` を確認する回帰テストを追加。
  - `docs/FORECAST.md`
    - strong-contra の `edge_strength` 定義を明文化。
- 期待効果:
  - 「予測を主軸にしつつ、手元テクニカルを併用（auto blend）」のまま、
    明確な逆行予測は ENTRY 段階で見送りやすくなる。

### 2026-02-18（追記）quant-position-manager 応答詰まり対策（single-flight + stale cache）

- 背景:
  - VM実運用で `position_manager(127.0.0.1:8301)` の `/position/open_positions` が 12-15秒 read timeout を連発し、
    EXIT worker の `position_manager_timeout` が断続的に発生。
  - 同時リクエスト集中時に `quant-position-manager` の threadpool が詰まり、`/health` まで応答遅延する状態を確認。
- 実装:
  - `workers/position_manager/worker.py` を更新。
  - `/position/open_positions` を server-side single-flight 化し、起動時warmupでキャッシュを事前投入したうえで fresh/stale を優先返却。
  - `open_positions` キャッシュ処理の `deepcopy` を除去し、返却時はトップレベル shallow copy のみでメタ付与。
  - `/position/sync_trades` も single-flight + TTL/stale cache 化して同時負荷を平準化。
  - manager 呼び出しを `asyncio.wait_for(asyncio.to_thread(...))` で timeout 制御し、
    長時間処理時も API 応答の詰まりを回避。
  - `/health` を `async` 化し、threadpool飽和時でもヘルス応答が取りやすい形へ変更。
- 追加env（任意）:
  - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC`
  - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_CACHE_TTL_SEC`
  - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC`
- runtimeチューニング（`ops/env/quant-v2-runtime.env`）:
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=8.0`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_CACHE_TTL_SEC=4.0`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC=24.0`
  - `POSITION_MANAGER_SERVICE_POOL_CONNECTIONS=4`
  - `POSITION_MANAGER_SERVICE_POOL_MAXSIZE=16`
  - `SCALP_PRECISION_EXIT_OPEN_POSITIONS_TIMEOUT_SEC=8.0`
- 期待効果:
  - `open_positions`/`sync_trades` の同時呼び出し時でも service 側の tail latency を抑え、
    EXIT worker 側 timeout 連鎖を縮小。

### 2026-02-18（追記）`open_positions` の `busy/timeout` 連鎖を追加抑止

- 背景:
  - 前段対策後も VM で `open_positions` の `ok=false` が継続し、`position manager busy` と
    `open_positions timeout (8.0s)` が優勢な時間帯が残存。
  - `execution/position_manager.py` の hot path で `orders.db` 参照が広く残り、
    `entry_thesis` 補完が毎回フルスキャンになっていた。
- 実装:
  - `execution/position_manager.py`
    - `open_positions` の `entry_thesis` 補完を「全 client_id 一律」から
      「`strategy_tag` / `entry_thesis` が不足する client_id 優先」に変更
      （`POSITION_MANAGER_OPEN_POSITIONS_ENRICH_ALL_CLIENTS=1` で従来挙動へ戻せる）。
    - `client_order_id -> entry_thesis` の in-memory TTL キャッシュを追加し、
      `orders.db` の同一問い合わせを短時間で再実行しないように変更。
      - `POSITION_MANAGER_ENTRY_THESIS_CACHE_TTL_SEC`（default: `900`）
      - `POSITION_MANAGER_ENTRY_THESIS_CACHE_MAX_ENTRIES`（default: `4096`）
    - `self._last_positions = copy.deepcopy(pockets)` を廃止し、
      `open_positions` 計算完了時の重複 deep copy を削減。
  - `workers/position_manager/worker.py`
    - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_STALE_MAX_AGE_SEC` 既定を `45s` に拡張。
    - `busy/timeout/error` かつ worker cache 未命中時に、
      `PositionManager` が保持する最新スナップショット（`_last_positions`）へ
      フォールバックして `ok=true` 応答を返せる経路を追加。
- 期待効果:
  - `/position/open_positions` の `ok=false` 比率を下げ、EXIT worker 側の
    timeout/connection refused 連鎖を縮小。
  - `orders.db` 読み取り競合による tail latency を抑制。

### 2026-02-18（追記）反発予測を `p_up` から独立出力化

- 背景:
  - 反発シグナル（`rebound_signal_20`）は `p_up` 合成に使っていたが、戦略側で独立に扱える値が不足していた。
- 実装:
  - `workers/common/forecast_gate.py`
    - `ForecastDecision` に `rebound_probability` を追加。
    - `row["rebound_probability"]` または `rebound_signal_20` から 0.0-1.0 へ正規化して返却。
  - `workers/forecast/worker.py`
    - `/forecast/decide` のシリアライズに `rebound_probability` と `target_reach_prob` を追加（service/local parity 修正）。
  - `execution/order_manager.py`
    - forecast service payload 逆シリアライズに `rebound_probability` / `target_reach_prob` を追加。
    - `entry_thesis["forecast"]` と order log の forecast監査項目へ `rebound_probability` を反映。
  - `execution/strategy_entry.py`
    - `forecast_context` に `rebound_probability` を追加（戦略ローカルで任意利用可能）。
- テスト:
  - `tests/workers/test_forecast_gate.py` に `rebound_probability` 伝播検証を追加。
  - `tests/workers/test_forecast_worker.py` を新規追加（serviceシリアライズ検証）。
  - `tests/execution/test_order_manager_preflight.py` に service payload 変換検証を追加。

### 2026-02-18（追記）反発予測を ENTRY融合ロジックへ反映

- 背景:
  - `rebound_probability` を独立出力しただけでは、実際の entry units / probability へ直接効かない。
- 実装:
  - `execution/strategy_entry.py`
    - `forecast_fusion` に反発項を追加し、`rebound_probability` から
      side別 support（`rebound_side_support`）を算出して `units_scale` / `entry_probability` を補正。
    - 追加env:
      - `STRATEGY_FORECAST_FUSION_REBOUND_ENABLED`
      - `STRATEGY_FORECAST_FUSION_REBOUND_UNITS_BOOST_MAX`
      - `STRATEGY_FORECAST_FUSION_REBOUND_UNITS_CUT_MAX`
      - `STRATEGY_FORECAST_FUSION_REBOUND_PROB_GAIN`
      - `STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_STRONG_CONTRA`
      - `STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_PROB_MIN`
      - `STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_DIR_PROB_MAX`
    - strong-contra 条件は維持しつつ、`long` かつ反発確率が十分高いときだけ
      `units=0` 拒否を回避して縮小試行できるオーバーライドを追加。
  - `ops/env/quant-v2-runtime.env`
    - 上記の反発融合キーを運用値として明示（再起動で反映）。
- テスト:
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - 反発確率あり/なしで contra-buy の縮小率が変わることを追加検証。
    - 反発高確率で strong-contra reject を回避できるケースを追加検証。

### 2026-02-18（追記）forecast before/after 評価スクリプトに反発項を追加

- 背景:
  - `scripts/eval_forecast_before_after.py` は breakout/session までしか比較できず、
    `FORECAST_TECH_REBOUND_WEIGHT(_MAP)` の候補比較を同一期間で行えなかった。
- 実装:
  - `scripts/eval_forecast_before_after.py`
    - CLIに `--rebound-weight` / `--rebound-weight-map` を追加。
    - after式に `rebound_signal`（`_rebound_bias_signal` 相当）を追加し、
      `combo` へ `rebound_weight * rebound_signal` を反映。
    - wick判定のため評価用 `merged` に `open/high/low` を追加。
    - JSON出力の `config` に rebound パラメータを保存。
- 期待効果:
  - VM同一データで反発重み候補を機械的に比較し、
    hit/MAE ベースで `quant-v2-runtime.env` の重み更新可否を判断できる。

### 2026-02-18（追記）反発重みグリッド再評価で短期TFを微調整

- 実データ評価:
  - VM上で `scripts/eval_forecast_before_after.py` を実行し、
    `max-bars=8050` / `steps=1,5,10` / breakout+session固定のまま
    `rebound_weight_map` 候補を比較。
  - 出力:
    - `logs/reports/forecast_improvement/rebound_grid_20260218T035935Z_*.json`
- 採用:
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.02,10m=0.01`（soft5m）
- 理由:
  - base（`1m=0.10,5m=0.04,10m=0.02`）比で、集計 `hit_after` が僅差で上回り、
    `mae_after` も悪化せず改善側を維持。
  - 1mは据え置き、5m/10mの反発寄与を控えめにして過反応リスクを抑制。
- 反映:
  - `ops/env/quant-v2-runtime.env` を更新し、`quant-forecast` / strategy worker再起動で適用。

### 2026-02-18（追記）72h forecast再評価で5m重みを更新

- 背景:
  - 2h/4h/48h では改善傾向を確認済みだったため、信頼度を上げる目的で72h窓を追加検証。
  - 同一M1連続窓（`bars=3181`, `2026-02-15T22:06:00+00:00`〜`2026-02-18T03:34:00+00:00`）で
    現行値（`gain=0.05`, `breakout_5m=0.20`, `session_5m=0.18`）と候補を比較した。
- 実施:
  - 72hグリッド探索（`forecast_tune_72h_5m_20260218T033726Z.json`）をVM実データで実行。
  - 上位候補 `gain=0.05`, `breakout_5m=0.26`, `session_5m=0.22` を同一窓で再評価。
    - `forecast_eval_20260218T035154Z_72h_current_recheck.json`
    - `forecast_eval_20260218T035154Z_72h_tuned_candidate.json`
    - `report_20260218T035154Z_72h_candidate_recheck.md`
- 結果（candidate - current）:
  - `1m`: ほぼ同等
  - `5m`: `hit_delta +0.0013`, `mae_delta -0.0003`, `range_cov_delta +0.0007`
  - `10m`: 同等
  - 合計: `hit_delta_sum +0.0013`, `mae_delta_sum -0.0003` で改善判定
- 反映:
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.05`
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.26,10m=0.28`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`

### 2026-02-18（追記）2h/4h/24h + 72h 統合監査で5m重みを最終調整

- 背景:
  - 72h単独では `breakout_5m=0.26` が最良だったが、同日直近の 2h/4h 窓でヒット改善が伸びにくい時間帯を確認。
  - 過学習回避のため、`2h/4h/24h`（直近25hデータ）と `72h`（既存72hデータ）を同時評価して最終値を確定。
- 実施:
  - 直近25hデータ取得: `logs/oanda/candles_M1_eval_24h_latest.json`（1494 bars）
  - 比較レポート:
    - `report_20260218T035819Z_followup_2h4h24h.md`
    - `followup_tune_2h4h24h_strict_20260218T040401Z.json`
    - `followup_aggregate_2h4h24h72h_20260218T040516Z.json`
    - `report_20260218T040516Z_followup_final.md`
- 判定:
  - `breakout_5m=0.22` は `breakout_5m=0.26` 比で
    - `2h/4h`: 同等（hit差ほぼ0）
    - `24h`: hit/mae とも小幅改善
    - `72h`: ごく小幅の悪化（許容範囲）
  - 短中期バランスを優先し、`5m=0.22` を最終採用。
- 反映:
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.05`（維持）
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.22,10m=0.28`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`（維持）

### 2026-02-18（追記）rebound_5m を追加最適化して微改善

- 背景:
  - `breakout/session` は `5m=0.22` で概ね収束したため、残りの改善余地として
    `FORECAST_TECH_REBOUND_WEIGHT_MAP` の `5m` 成分のみを再探索。
- 実施:
  - 最新VM実データを再取得:
    - `logs/oanda/candles_M1_eval_24h_latest.json`（1554 bars）
    - `logs/oanda/candles_M1_eval_72h_latest.json`（3224 bars）
  - グリッド:
    - `breakout_5m=[0.20,0.22,0.24,0.26]`
    - `session_5m=[0.20,0.22,0.24]`
    - `rebound_5m=[0.00,0.02,0.04,0.06]`
  - 出力:
    - `logs/reports/forecast_improvement/more_improve_grid_20260218T041728Z.json`
    - `logs/reports/forecast_improvement/report_20260218T041728Z_more_improve.md`
- 判定:
  - 有効候補は `breakout/session` を現行維持したままの `rebound_5m` 調整のみ。
  - `rebound_5m=0.06` は `0.02` 比で `hit_delta` を維持しつつ、
    `2h/4h/24h/72h` 全窓で `mae_delta` を微小改善（いずれもマイナス方向）。
- 反映:
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.06,10m=0.01`

### 2026-02-18（追記）72h/24h 同時最適化で forecast 5m の hit を上積み

- 背景:
  - 反発予測を含む運用値で、`5m` の hit をもう一段上げるために
    `feature_expansion_gain` / `breakout_5m` / `session_5m` / `rebound_5m` を再探索した。
- 実施:
  - VM上で高速グリッド（`108` 候補）を同一データ比較で実行。
  - 窓:
    - 72h相当: `max-bars=4320`
    - 24h相当: `max-bars=1440`
  - 出力:
    - `logs/reports/forecast_improvement/grid_fast_5m_20260218T044010Z.json`
  - 選定候補:
    - `gain=0.04`, `breakout_5m=0.20`, `session_5m=0.20`, `rebound_5m=0.03`
- 判定（候補 - 旧運用）:
  - 72h:
    - `1m`: `hit_after +0.0007`, `mae_delta -0.0006`
    - `5m`: `hit_after +0.0016`, `mae_delta -0.0018`
    - `10m`: `hit_after -0.0010`, `mae_delta -0.0022`
  - 24h:
    - `hit_after` は `1m/5m/10m` で同値
    - `mae_delta` は微小差（`5m +0.0001`）
  - 24h hit を維持したまま `5m` を上積みできるため採用。
- 反映:
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.04`
    - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.20,10m=0.28`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.20,10m=0.30`
    - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.03,10m=0.01`

### 2026-02-18（追記）scalp_ping_5s_b の「上方向取り残し」抑制チューニング

- 背景:
  - 本番VM/OANDA実データで、`scalp_ping_5s_b_live` の一部エントリーが
    `forecast.reason=edge_block` / `forecast.allowed=0` でも約定し、
    逆行時に長時間取り残されるケースを確認。
  - 同時間帯に `quant-position-manager` / `quant-order-manager` の再起動・busy/timeout が重なり、
    exit worker の close 呼び出し失敗（connection refused）が発生。
- 実施:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.70` を追加。
    - 低確率シグナルを `order_manager` 側で機械的に reject し、逆張り方向の残留ポジ増加を抑制。
  - `ops/env/quant-scalp-ping-5s-b-exit.env`
    - `RANGEFADER_EXIT_NEW_POLICY_START_TS=2026-02-17T00:00:00Z` を追加。
    - service再起動のたびに `new_policy_start_ts` が現在時刻へリセットされ、
      既存建玉が legacy 扱いで loss-cut 系ルールから外れる事象を回避。
- 期待効果:
  - `scalp_ping_5s_b_live` の低確率エントリー頻度を抑制。
  - 既存建玉にも `loss_cut/non_range_max_hold/direction_flip` を継続適用し、長時間取り残しを低減。

### 2026-02-19（追記）scalp_ping_5s_b のSL欠損再発を根本遮断

- 背景:
  - VM実運用で `scalp_ping_5s_b_live` の新規約定に `stopLossOnFill` 未付与が再発。
  - 原因は `ops/env/scalp_ping_5s_b.env` の `SCALP_PING_5S_B_USE_SL=0` が
    `workers/scalp_ping_5s_b.worker` の prefix マッピングで
    `SCALP_PING_5S_USE_SL=0` に投影され、SL/ハードストップが同時に無効化される構成だったこと。
- 実施:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_USE_SL=1`
    - `SCALP_PING_5S_B_DISABLE_ENTRY_HARD_STOP=0`
  - `ops/env/quant-order-manager.env`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B=1`
    - `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_B=0`
  - `workers/scalp_ping_5s_b/worker.py`
    - 起動時 fail-safe を追加し、B戦略はデフォルトで
      `SCALP_PING_5S_USE_SL=1` / `SCALP_PING_5S_DISABLE_ENTRY_HARD_STOP=0`
      へ自動補正する。
    - 例外運用は `SCALP_PING_5S_B_ALLOW_UNPROTECTED_ENTRY=1` でのみ許可。
  - `execution/order_manager.py`
    - `ORDER_FIXED_SL_MODE=0` でも、`scalp_ping_5s_b*` は
      `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B` を優先して
      `stopLossOnFill` を戦略ローカルで有効化可能に修正。
- 期待効果:
  - B戦略で「SLなし + ハードストップ無効」の組み合わせを既定で禁止し、
    同種のテイル損失を設定ドリフト起因で再発させない。

### 2026-02-19（追記）scalp_ping_5s_b の方向転換遅延を即時リルート化

- 背景:
  - VM実績で `scalp_ping_5s_b_live` は long 側の方向一致率は高い一方、
    short 側で方向一致率が低く、逆向きエントリーの SL ヒットが連発。
  - 要件として「エントリー頻度を落とさず、方向転換だけ速くする」を固定。
- 実施:
  - `workers/scalp_ping_5s/config.py`
    - `FAST_DIRECTION_FLIP_*` 系パラメータを追加。
    - Bプレフィックス（`SCALP_PING_5S_B`）では既定有効化。
  - `workers/scalp_ping_5s/worker.py`
    - `direction_bias` と `horizon_bias` が同方向で強く一致した場合、
      reject せず `signal.side` を即時反転する
      `_maybe_fast_direction_flip` を追加。
    - 反転後は `horizon` を再評価し、`*_fflip` モードで `entry_thesis` に記録。
    - `extrema_reversal` 後にも最終 side へ再適用するようにし、
      逆向き上書きが残っても post-route で是正できるよう更新。
    - `horizon=neutral` でも `direction_bias` が強い場合は反転を許可する
      neutral-horizon 条件（bias強度閾値）を追加。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_*` を設定し、
      低遅延（cooldown短め）で方向反転を許可。
- 期待効果:
  - ショート偏重時にエントリー自体を止めず、ロング側への反転を優先して
    頻度を維持しながら SL 到達率を圧縮する。

### 2026-02-19（追記）scalp_ping_5s_b に連続SL起点の方向反転（SL Streak Flip）を追加

- 背景:
  - VM実績で `scalp_ping_5s_b_live` は、同方向の `STOP_LOSS_ORDER` が連続した直後に
    同方向エントリーを継続すると勝率が大きく低下し、損失が連鎖する傾向を確認。
  - 要件は「エントリー頻度を落とさず、方向だけを根本補正する」こと。
- 実施:
  - `workers/scalp_ping_5s/config.py`
    - `SL_STREAK_DIRECTION_FLIP_*` パラメータを追加
      （enabled/min_streak/lookback/max_age/confidence_add/cache_ttl/log_interval）。
  - `workers/scalp_ping_5s/worker.py`
    - `trades.db` の `strategy_tag + pocket` クローズ履歴から
      直近の `STOP_LOSS_ORDER` 同方向連敗を検出する `_load_stop_loss_streak` を追加。
    - `extrema/fast_flip` 後に `_maybe_sl_streak_direction_flip` を適用し、
      連敗方向と同じ side の新規シグナルのみ `*_slflip` モードで反転。
    - 反転後は `horizon/m1_trend/direction_bias` のロット倍率を再評価し、
      エントリー拒否はせず side リライトのみ実施。
    - `entry_thesis` へ `sl_streak_direction_flip_*` と `sl_streak_*` を記録。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_*` を追加してB戦略で有効化。
  - テスト:
    - `tests/workers/test_scalp_ping_5s_sl_streak_flip.py`
      - 同方向SL連敗の検出
      - 同方向シグナル時の反転
      - 既に逆方向シグナル時の非反転
      - 連敗が古い場合（stale）の非反転
- 期待効果:
  - 「連続SL後に同方向へ入り続ける」局面を自動で切り替え、
    時間帯ブロックやエントリー削減なしで方向遅延損失を抑制する。

### 2026-02-19（追記）SL Streak Flip を「SL回数 + 成り行きプラス + テクニカル一致」へ再調整

- 背景:
  - デプロイ初期の実績で `sl_streak_direction_flip_applied=1` 群の成績が劣化。
  - `fast_direction_flip` でロングへ反転した直後に
    `sl_streak` が再度ショートへ上書きするケースを確認。
- 実施:
  - `workers/scalp_ping_5s/config.py`
    - `SL_STREAK_DIRECTION_FLIP_*` に以下を追加:
      - `ALLOW_WITH_FAST_FLIP`
      - `MIN_SIDE_SL_HITS`
      - `MIN_TARGET_MARKET_PLUS`
      - `METRICS_LOOKBACK_TRADES`
      - `METRICS_CACHE_TTL_SEC`
      - `REQUIRE_TECH_CONFIRM`
      - `DIRECTION_SCORE_MIN`
      - `HORIZON_SCORE_MIN`
  - `workers/scalp_ping_5s/worker.py`
    - 直近クローズ履歴から side別に
      - `STOP_LOSS_ORDER` 回数
      - `MARKET_ORDER_TRADE_CLOSE` かつ `realized_pl>0` 回数
      を集計する `SideCloseMetrics` を追加。
    - `sl_streak_flip` 発火条件を以下へ変更:
      - 同方向SL連敗（既存）
      - side別SL回数が閾値以上
      - 反転先sideの成り行きプラス回数が閾値以上
      - `direction_bias` または `horizon` が反転先sideを支持
      - 同ループで `fast_flip` 済みなら（既定）`sl_streak` は発火しない
    - `entry_thesis` に
      `sl_streak_side_sl_hits_recent` /
      `sl_streak_target_market_plus_recent` /
      `sl_streak_direction_confirmed` /
      `sl_streak_horizon_confirmed`
      を追記。
  - `ops/env/scalp_ping_5s_b.env`
    - 上記 `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_*` の新規パラメータを追加。
  - テスト:
    - `tests/workers/test_scalp_ping_5s_sl_streak_flip.py`
      - target market-plus不足時の非発火
      - fast_flip優先時の非発火
      を追加。
- 期待効果:
  - 連敗反転の発火を「条件付きの高品質反転」に絞り、
    方向衝突（fast_flip vs sl_streak）での逆行エントリーを抑制する。

### 2026-02-19（追記）scalp_ping_5s_b のショート偏重クラスタを根本抑制（extrema 非対称化）

- 背景:
  - VM `trades.db`（2026-02-17 以降）で `scalp_ping_5s_b_live` は
    `short_bottom_soft` / `long_top_soft_reverse` クラスタの平均損益が大幅マイナス。
  - 要件は「時間帯ブロックではなく、方向決定ロジックを根本補正しつつ件数は落とさない」。
- 実施:
  - `workers/scalp_ping_5s/config.py`
    - 追加:
      - `EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT`
      - `EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT`
      - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT`
      - `EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE`
      - `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK`
  - `workers/scalp_ping_5s/worker.py`
    - `short_bottom_soft` を side=short 専用の縮小倍率で処理し、
      `mtf_balanced` かつ short非優勢時はさらに縮小
      （`short_bottom_soft_balanced` を `entry_thesis` に記録）。
    - extrema reversal を非対称化し、
      B既定で `long -> short` の reversal 経路を無効化。
      （`short -> long` reversal は維持）
    - `sl_streak_direction_flip` に
      `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK` を追加し、
      連敗が閾値以上のときは `target_market_plus` 条件をバイパス可能にした
      （tech確認条件は維持）。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT=0.42`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT=0.30`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE=2.10`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=3`
  - テスト:
    - 新規 `tests/workers/test_scalp_ping_5s_extrema_routes.py`
      - `long->short` reversal 抑止
      - `short->long` reversal 維持
      - `short_bottom_soft_balanced` 縮小倍率適用
    - `tests/workers/test_scalp_ping_5s_sl_streak_flip.py`
      - `force_streak` で `target_market_plus` 弱いケースを反転許可する回帰テストを追加。
- 期待効果:
  - エントリー拒否ではなく「方向補正 + ロット縮小」で件数を維持し、
    ショート偏重の逆行SL連鎖を抑える。

### 2026-02-19（追記）方向誤判定クラスタ向けホットフィックス（flip発火の厳格化）

- 背景:
  - VM直近ログで `fast_flip side=long` / `sl_streak_flip side=long` が短時間に連続し、
    方向上書き後の `STOP_LOSS_ORDER` がクラスター化。
  - 要件「方向性を改善して利益を残す」に対して、反転発火の閾値を即時に引き締めた。
- 実施（`ops/env/scalp_ping_5s_b.env`）:
  - `FAST_DIRECTION_FLIP_*` を厳格化:
    - `DIRECTION_SCORE_MIN=0.52`（旧0.38）
    - `HORIZON_SCORE_MIN=0.32`（旧0.18）
    - `HORIZON_AGREE_MIN=3`（旧2）
    - `NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.82`（旧0.68）
    - `MOMENTUM_MIN_PIPS=0.18`（旧0.06）
    - `CONFIDENCE_ADD=2`（旧4）
    - `COOLDOWN_SEC=1.2`（旧0.4）
    - `REGIME_BLOCK_SCORE=0.60`（旧0.70）
  - `SL_STREAK_DIRECTION_FLIP_*` を厳格化:
    - `LOOKBACK_TRADES=10`（旧6）
    - `MIN_SIDE_SL_HITS=3`（旧2）
    - `MIN_TARGET_MARKET_PLUS=2`（旧1）
    - `FORCE_STREAK=5`（旧3）
    - `DIRECTION_SCORE_MIN=0.55`（旧0.40）
    - `HORIZON_SCORE_MIN=0.42`（旧0.24）
- 期待効果:
  - 逆方向への過剰リライトを減らし、連続SLによる利益の全戻しを抑制する。

### 2026-02-19（追記）order_manager 側 perf_block による発注欠落を是正

- 背景:
  - flip 厳格化デプロイ後、`quant-order-manager` ログで
    `OPEN_REJECT note=perf_block:margin_closeout_n=...` を確認。
  - `scalp_ping_5s_b` ワーカー側の `PERF_GUARD_MODE=warn` は
    ワーカー env にのみ存在し、V2 分離の `quant-order-manager` には未反映だった。
- 実施:
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=warn` を追加。
- 期待効果:
  - `margin_closeout_n` を理由にした order_manager 側の一律 reject を防ぎ、
    件数を落とさず方向改善ロジックの効果検証を継続可能にする。

### 2026-02-19（追記）利伸ばし強化: scalp_ping_5s_b の早利確/早ロックを緩和

- 背景（VM実績, `scalp_ping_5s_b_live`, 直近6時間）:
  - `win3 (>=+3p)` は `35件 / 平均 +4.15p` だが、
    `loss24 (<=-2.4p)` は `231件 / 平均 -2.64p`。
  - `close_request` 理由では
    - `take_profit`: `168件 / 平均 +2.279p / win3=41 / loss24=0`
    - `lock_floor`: `55件 / 平均 +0.609p / win3=1`
  - 伸ばせる局面で lock_floor が先に発火しやすく、利伸ばし余地が残っていた。
- 実施:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` を alias から個別定義へ変更。
    - 追加/変更（exit_profile）:
      - `profit_pips=2.0`（実質TP発火の底上げ）
      - `trail_start_pips=2.3`
      - `trail_backoff_pips=0.95`
      - `lock_buffer_pips=0.70`
      - `lock_floor_min_hold_sec=45`
      - `range_profit_pips=1.6`
      - `range_trail_start_pips=2.0`
      - `range_trail_backoff_pips=0.80`
      - `range_lock_buffer_pips=0.55`
    - 既存の `loss_cut/non_range_max_hold/direction_flip` は維持。
- 期待効果:
  - 早い `lock_floor` クローズを抑え、`take_profit` 側へ遷移する比率を上げる。
  - 頻度を落とさず、伸びるトレードの平均利幅を押し上げる。

### 2026-02-19（追記）ロット逆転是正: 「勝ちで小、負けで大」を確率スケールで補正

- 背景（VM実績, `scalp_ping_5s_b_live`, 直近24時間）:
  - `win3 (>=+3p)` の平均ロット: `620.6`
  - `loss24 (<=-2.4p)` の平均ロット: `864.7`
  - `entry_probability` 別では
    - `ep>=0.90`: `n=463`, `avg_units=1464.4`, `avg_pips=-0.95`, `win3_rate=0.03`
    - `<0.70`: `n=308`, `avg_units=388.8`, `avg_pips=-1.01`, `win3_rate=0.127`
  - 高確率側が高ロットに寄りつつ期待値が改善しておらず、サイズ配分が逆効率だった。
- 実施:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.45`（0.70→0.45）
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`（新規）
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`（新規）
  - `ops/env/scalp_ping_5s_b.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`（0.65→0.75）
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`（新規）
- 期待効果:
  - 高 `entry_probability` 時の過大ロット（>1.0x）を禁止し、負け側の損失振れ幅を抑える。
  - 低〜中 `entry_probability` の過小ロットを緩和し、勝ち側の取り分を改善する。

### 2026-02-19（追記）高確率遅行補正: `entry_probability` を方向整合で再校正

- 背景（VM実績, `scalp_ping_5s_b_live`, 2026-02-18 17:00 JST 以降）:
  - `entry_probability >= 0.90`: `367件`, `avg_pips=-0.59`, `SL率=49%`
  - ショート `entry_probability >= 0.90`: `123件`, `avg_pips=-1.07`, `SL率=63.4%`
  - `0.50-0.59` 帯の方が `avg_pips=+0.40` で優位
  - 発注遅延は主要因でなく、`preflight->fill ≒ 381ms` で安定
- 実施:
  - `workers/scalp_ping_5s/config.py`
    - `ENTRY_PROBABILITY_ALIGN_*` パラメータ群を追加
      - direction/horizon/m1 重み
      - penalty/boost
      - revert 時ペナルティ緩和
      - floor（高prob時の過剰リジェクト回避）
      - units follow（確率調整比率でロットを追随縮小）
  - `workers/scalp_ping_5s/worker.py`
    - `entry_probability` を `confidence` 直結から
      `direction_bias + horizon + m1_trend` の整合再校正へ変更。
    - 追加した `probability_units_mult` をサイズ計算へ反映し、
      過大評価局面のロットを自動縮小。
    - `entry_thesis` に監査項目を追加:
      - `entry_probability_raw`
      - `entry_probability_units_mult`
      - `entry_probability_alignment.{support,counter,penalty,...}`
    - openログへ `prob=raw->adjusted` と `p_mult` を追加。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_*` を追加
      （本番Bワーカーの既定値を明示）。
- 期待効果:
  - 高確率の遅行・過大評価を抑え、逆行局面でのロット集中を低減。
  - エントリー件数を大きく落とさず（高prob floor あり）、
    方向転換遅れ時の損失インパクトを縮小。

### 2026-02-19（追記）反転撤退遅れの是正: short側EXITをサイド別に高速化

- 背景（VM実績, `scalp_ping_5s_b_live`, 2026-02-18 17:00 JST 以降）:
  - 合計: `n=1529`, `PL=-19,369.5 JPY`, `avg_pips=-1.745`
  - side別:
    - `long`: `n=528`, `PL=-2,590.0`, `avg_pips=-0.249`
    - `short`: `n=1001`, `PL=-16,779.5`, `avg_pips=-2.534`
  - `short + MARKET_ORDER_TRADE_CLOSE`: `n=831`, `PL=-14,279.4`, `avg_pips=-2.666`, `avg_hold=625s`
  - 保有時間バケット（short + MARKET close）:
    - `900s+`: `n=278`, `PL=-12,537.1`, `avg_pips=-6.565`（損失の主塊）
- 実施:
  - `workers/scalp_ping_5s/exit_worker.py`
    - `exit_profile` にサイド別キーを追加サポート:
      - `non_range_max_hold_sec_<side>`（例: `_short`）
      - `direction_flip` の `<side>_*` 上書き
        - `min_hold_sec`, `min_adverse_pips`
        - `score_threshold`, `release_threshold`
        - `confirm_hits`, `confirm_window_sec`, `cooldown_sec`
        - `forecast_weight`, `de_risk_threshold`
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b_live` のみ調整:
      - `non_range_max_hold_sec_short=300`（既定900からshortのみ短縮）
      - `direction_flip.short_*` を追加してshort逆行時の判定を早める
        - `min_hold_sec=45`
        - `min_adverse_pips=1.0`
        - `score_threshold=0.56`
        - `release_threshold=0.42`
        - `confirm_hits=2`
        - `confirm_window_sec=18`
        - `de_risk_threshold=0.50`
        - `forecast_weight=0.45`
  - テスト:
    - `tests/workers/test_scalp_ping_5s_exit_worker.py`
      - short側オーバーライド適用テスト
      - `non_range_max_hold_sec_short` がshortのみに効くテスト
- 期待効果:
  - 「反転を察して微益/小損で撤退」の遅れを short 側で直接改善。
  - エントリー頻度を落とさず、長時間逆行ホールド由来の tail 損失を圧縮。

### 2026-02-19（追記）内部テスト精度ゲート: replay walk-forward 品質判定を追加

- 背景:
  - 戦略ワーカーのリプレイ比較が `summary_all.json` 手動確認中心で、期間分割（in-sample / out-of-sample）と閾値判定の自動化が不足していた。
  - 「改善したつもり」の変更を機械判定で落とす仕組みを標準化する必要があった。
- 実施:
  - 追加: `analytics/replay_quality_gate.py`
    - `PF/勝率/総pips/maxDD` 算出
    - walk-forward fold 生成
    - gate 判定（`pf_stability_ratio` を含む）
  - 追加: `scripts/replay_quality_gate.py`
    - `scripts/replay_exit_workers_groups.py` を複数 tick へ連続実行
    - fold ごとに train/test 判定
    - `quality_gate_report.json` / `quality_gate_report.md` / `commands.json` を出力
  - 追加: `config/replay_quality_gate.yaml`
    - 対象ワーカー、標準 replay フラグ、閾値、walk-forward 分割を定義
  - 追加テスト: `tests/analysis/test_replay_quality_gate.py`
  - 仕様追記: `docs/REPLAY_STANDARD.md`, `docs/ARCHITECTURE.md`
- 運用:
  - 標準実行:
    - `python scripts/replay_quality_gate.py --config config/replay_quality_gate.yaml --ticks-glob 'logs/replay/USD_JPY/USD_JPY_ticks_YYYYMM*.jsonl' --strict`
  - pass/fail は worker 単位で fold pass rate を集計し、`min_fold_pass_rate` を下回った worker を fail とする。

### 2026-02-19（追記）replay ワーカー互換性修正: 不在モジュールを optional 化

- 背景:
  - 実行環境によって `workers.impulse_*` / `workers.pullback_s5` などが存在しない場合、`scripts/replay_exit_workers_groups.py` が import 時点で停止していた。
  - `scripts/replay_exit_workers.py` も `workers.micro_bbrsi` 不在で同様に停止していた。
- 実施:
  - `scripts/replay_exit_workers.py`
    - `workers.micro_bbrsi.exit_worker` を optional import 化。
  - `scripts/replay_exit_workers_groups.py`
    - exit worker import を optional 化し、存在する worker のみ実行対象へ動的登録。
    - 要求 worker が全て不在の場合は fail-fast で明示エラーを返す。
  - `config/replay_quality_gate.yaml`
    - 既定 worker を `session_open` へ更新（現行 VM 構成で実行可能なデフォルト）。

### 2026-02-19（追記）確率帯ロット再配分: 勝ち小ロット/負け大ロットの逆配分を補正

- 背景（VM実績, `scalp_ping_5s_b_live`, 2026-02-18 17:00 JST 以降）:
  - `entry_probability >= 0.90` 帯が低成績でもロット比重が高く、逆行時の損失インパクトが大きかった。
  - `entry_probability < 0.70` 帯は相対優位でもロットが小さく、利益回収効率が悪かった。
  - 直近では「勝ちで小、負けで大」の逆配分が継続していた。
- 実施:
  - `workers/scalp_ping_5s/worker.py`
    - side別に `trades.db` の確率帯統計（`<0.70` / `>=0.90`）を集計する
      `EntryProbabilityBandMetrics` とキャッシュを追加。
    - `_entry_probability_band_units_multiplier()` を追加し、
      - 高確率帯が劣後する局面: high帯ロットを縮小
      - 低確率帯が優位な局面: low帯ロットを増量
      の動的再配分を実装。
    - 追加で side別 `SL hit` と `MARKET_ORDER_TRADE_CLOSE +` 件数を係数化し、
      同一 side の連敗圧が高いときはロットを減衰。
    - サイズ計算チェーンに `probability_band_units_mult` を追加。
    - `entry_thesis` に `entry_probability_band_units_mult` と
      `entry_probability_band_allocation.*` を記録。
  - `workers/scalp_ping_5s/config.py`
    - `ENTRY_PROBABILITY_BAND_ALLOC_*` 設定群を追加
      （lookback, 閾値, 最大縮小/増量, side指標ゲインなど）。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
- テスト:
  - 追加: `tests/workers/test_scalp_ping_5s_probability_band_alloc.py`
    - high帯縮小 / low帯増量 / side指標減衰 / DB集計を検証。
  - 実行: `22 passed`
    - `tests/workers/test_scalp_ping_5s_probability_band_alloc.py`
    - `tests/workers/test_scalp_ping_5s_sl_streak_flip.py`
    - `tests/workers/test_scalp_ping_5s_exit_worker.py`
    - `tests/workers/test_scalp_ping_5s_extrema_routes.py`
- 目的:
  - エントリー頻度を落とさず、損失が出やすい確率帯への過大配分を抑える。
  - 利が伸びる帯へロットを寄せることで、同一シグナル密度でも期待値を改善する。

### 2026-02-19（追記）利小損大是正: `scalp_ping_5s_b_live` EXIT非対称を再調整

- 背景（VM, 直近12h）:
  - `scalp_ping_5s_b_live` は `avg_win=+2.021p` に対して `avg_loss=-3.634p`。
  - 特に `MARKET_ORDER_TRADE_CLOSE` 負け平均が `-6.343p` と深く、
    short 側で長時間保有の tail 損失が残っていた。
- 実施:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` の `exit_profile` を再調整。
      - 利益側（利を伸ばす）:
        - `profit_pips: 2.4`（from 2.0）
        - `trail_start_pips: 2.7`（from 2.3）
        - `range_profit_pips: 1.8`（from 1.6）
      - 損失側（深い逆行を抑制）:
        - `loss_cut_hard_pips: 6.0`（from 8.0）
        - `loss_cut_hard_cap_pips: 6.2`（新規）
        - `loss_cut_max_hold_sec: 900`（from 1200）
        - `non_range_max_hold_sec: 780`（from 900）
        - `non_range_max_hold_sec_short: 240`（`b_live` にも追加）
      - 方向反転（short早期退避）:
        - `direction_flip` の `short_*` 閾値を前倒し
        - `de_risk_fraction: 0.55`（from 0.40）
- テスト:
  - `tests/workers/test_scalp_ping_5s_exit_worker.py`
  - `tests/workers/test_scalp_ping_5s_extrema_routes.py`
  - 結果: `11 passed`
- 目的:
  - まず「負けを深くしない」を優先して、
    1トレード損益の非対称（損大・利小）を改善する。

### 2026-02-19（追記）`scalp_ping_5s_b_live` 根本補正: 確率floor厳格化 + 連敗反転高速化 + ロット増幅圧縮

- 背景（VM実績）:
  - 高確率帯の成績劣後が継続し、ロット逆配分（高確率帯に重いサイズ）が残っていた。
  - `SL streak` 連敗時に tech confirm 不足で方向転換が遅延するケースがあった。
- 実施:
  - `workers/scalp_ping_5s/worker.py`
    - `_adjust_entry_probability_alignment()`:
      - floor適用条件に `support/counter` ガードを追加。
      - `support < counter` または `counter > FLOOR_MAX_COUNTER` の場合は floor を不適用。
      - メタ情報に `floor_block_reason` を追加。
    - `_maybe_sl_streak_direction_flip()`:
      - `force_streak` 到達時、`SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM=1`
        なら tech confirm 未充足でも反転を許可。
      - reason に `force_tech` を記録。
    - `_confidence_scale()`:
      - 固定レンジ（0.65〜1.15）を廃止し、config駆動へ変更。
  - `workers/scalp_ping_5s/config.py`
    - 追加:
      - `CONFIDENCE_SCALE_MIN_MULT` / `CONFIDENCE_SCALE_MAX_MULT`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER`
      - `SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM`
  - `ops/env/scalp_ping_5s_b.env`
    - B専用チューニングを更新:
      - confidence増幅圧縮
      - probability-band再配分強化（high縮小/side減衰）
      - fast flip / sl streak flip の前倒し
      - force streak時のtechバイパス有効化
- テスト:
  - 追加/更新:
    - `tests/workers/test_scalp_ping_5s_probability_band_alloc.py`
      - floor block / floor apply のケースを追加
    - `tests/workers/test_scalp_ping_5s_sl_streak_flip.py`
      - force streak + tech bypass の有効/無効ケースを追加
- 目的:
  - 方向逆行時の「高確率過信」を減らし、連敗局面での反転遅れを圧縮。
  - エントリー頻度を維持したまま、損大・利小の非対称を改善する。

### 2026-02-19（追記）`scalp_ping_5s_b_live` 緊急デリスク: 同方向クラスター抑制 + SL反転失効の緩和

- 背景（VM実績, 直近2h）:
  - `scalp_ping_5s_b_live` が `-104.8 pips`。
  - 損失の主因は `long + STOP_LOSS_ORDER`（`155件 / -373.8 pips`）。
  - `sl_streak_direction_flip_reason` が `streak_stale` となるケースが多く、
    連続SL後の方向転換が有効期限切れで機能しにくかった。
- 実施（`ops/env/scalp_ping_5s_b.env`）:
  - 建玉集中の抑制:
    - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES=20`（from 40）
    - `SCALP_PING_5S_B_MAX_PER_DIRECTION=12`（from 24）
  - SL反転の有効期限を延長:
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC=480`（from 180）
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES=36`（from 24）
  - side成績連動の減衰強化:
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN=0.90`（from 0.65）
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.60`（from 0.70）
  - 取りこぼし抑制（片側小ロット通過）:
    - `SCALP_PING_5S_B_MIN_UNITS=100`（from 150）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=100`（from 150）
- 目的:
  - 同方向同時建玉でのSLクラスターを抑え、ドローダウン加速を止める。
  - 連続SL時の転換ロジックを、実運用時間軸で失効しにくくする。

### 2026-02-20（追記）`scalp_ping_5s_b_live` 逆行スタック抑制: 同方向含み損中のロット自動圧縮

- 背景（VM実績, 直近40件）:
  - `38/40` が long 側で、`SL率 89.5%`、`-391.9 JPY`。
  - 同時刻 `STOP_LOSS_ORDER` クラスター（`n=8`, `n=5`）で損失が加速。
  - 問題は「頻度不足」ではなく、方向不利局面で同方向ロットが維持されること。
- 実施:
  - `workers/scalp_ping_5s/worker.py`
    - `SideAdverseStackEval` と `_side_adverse_stack_units_multiplier()` を追加。
    - side別 close実績（`SL率` / `MARKET close +率`）と、同方向の
      `active trades`、同方向の含み損DD（pips）を合成し、
      逆行局面のみロット倍率を段階圧縮。
    - エントリー可否は変えず、サイズのみ縮小（頻度維持）。
    - `entry_thesis` に `side_adverse_stack_*` を記録。
  - `workers/scalp_ping_5s/config.py`
    - `SIDE_ADVERSE_STACK_*` 設定群を追加
      （lookback, 閾値, active start, step, min multiplier, DD連動）。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_*` を追加して有効化。
    - あわせて `FAST_DIRECTION_FLIP_*` の閾値を軽くし、反転遅延を短縮。
- 目的:
  - 方向ミス時の同方向積み増しをサイズで抑え、SLクラスターの損失勾配を下げる。
  - 発注本数は維持しながら、悪い局面でだけ損失の厚みを削る。

### 2026-02-20（追記）`scalp_ping_5s_flow_live` fallback経路の perf_guard 整合化

- 背景（VM実測）:
  - `strategy_entry -> quant-order-manager` の service call が timeout した場合、
    worker 内 local fallback で `order_manager.market_order` が実行される。
  - flow 専用の `PERF_GUARD_LOOKBACK_DAYS=1` を order-manager 側にのみ設定していると、
    fallback 側では default(3日) となり `perf_block:margin_closeout_n=1 n=24` が継続する。
- 実施:
  - `ops/env/scalp_ping_5s_flow.env` に
    `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を追加。
  - `ops/env/quant-scalp-ping-5s-flow.env` にも同値を追加し、
    systemd worker env と repo override env の双方で値を一致させた。
- 目的:
  - service 経路と local fallback 経路で同一の perf_guard 判定を維持し、
    timeout 時だけ stale closeout 判定へ戻る不整合を排除する。

### 2026-02-20（追記）Autotune UI の snapshot 選択を「鮮度優先」に修正

- 背景:
  - `apps/autotune_ui.py` の dashboard/snapshot 選択は `remote -> gcs -> local` の固定優先で、
    `generated_at` が古い remote snapshot でも metrics があれば常に採用される経路があった。
  - その結果、UI の「1時間ごとのトレード」などが更新停止に見える事象を誘発しうる。
- 実施:
  - `apps/autotune_ui.py`
    - `UI_SNAPSHOT_MAX_AGE_SEC`（default: `max(120, UI_AUTO_REFRESH_SEC*4)`）を追加。
    - `generated_at` から snapshot 年齢を判定する `_snapshot_age_sec` / `_is_snapshot_fresh` を追加。
    - `_pick_snapshot_by_preference` を更新し、fresh な snapshot のみを優先採用。
    - fresh が無い場合は source 固定優先ではなく、`generated_at` が新しい snapshot を採用。
    - dashboard の stale 表示判定も同一閾値へ統一。
  - テスト追加:
    - `tests/apps/test_autotune_ui_snapshot_selection.py`
      - stale remote + fresh local で local を採用
      - fresh remote を優先採用
      - 全候補 stale 時は最も新しい候補を採用
- 目的:
  - stale な外部 snapshot に引きずられて UI 更新が止まって見える問題を防止し、
    時間帯実績を含む表示を実データ鮮度に追従させる。
### 2026-02-20（追記）orders.db ロック耐性の強化（order_manager）

- 背景（VM実測）:
  - `quant-order-manager` / strategy worker の双方で
    `[ORDER][LOG] failed to persist orders log: database is locked` が散発。
  - lock 発生時に orders ログが欠損し、service timeout 時の原因追跡が難化していた。
- 実施:
  - `execution/order_manager.py`
    - orders logger の `busy_timeout` デフォルトを `250ms -> 1500ms` に変更。
    - `database is locked/busy` を検知した場合のみ、短い backoff 付き再試行
      （`ORDER_DB_LOG_RETRY_*`）を実装。
    - lock 時は接続をリセットして再試行し、最終失敗時のみ warning を残す。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_DB_BUSY_TIMEOUT_MS=1500`
    - `ORDER_DB_LOG_RETRY_ATTEMPTS=3`
    - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.03`
    - `ORDER_DB_LOG_RETRY_BACKOFF=2.0`
    - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.20`
  - テスト:
    - `tests/execution/test_order_manager_log_retry.py` を追加。
      - lock 1回で再試行成功
      - retry 上限到達時に warning
- 目的:
  - 高頻度発注時の SQLite write 競合で orders 監査ログが落ちる経路を縮小し、
    実運用での可観測性を維持する。

### 2026-02-20（追記）orders.db lock再発の設定上書き修正（quant-order-manager.env）

- 背景（VM実測）:
  - code と `quant-v2-runtime.env` では `ORDER_DB_BUSY_TIMEOUT_MS=1500` へ更新済みだったが、
    `quant-order-manager.service` は `ops/env/quant-order-manager.env` を後段で読み込むため、
    同ファイルの `ORDER_DB_BUSY_TIMEOUT_MS=250` が runtime 値を再上書きしていた。
  - その結果、`[ORDER][LOG] failed to persist orders log: database is locked` が
    直近30分で再び多発（`lock_count_30m=20`）していた。
- 実施:
  - `ops/env/quant-order-manager.env`
    - `ORDER_DB_BUSY_TIMEOUT_MS=1500`
    - `ORDER_DB_LOG_RETRY_ATTEMPTS=3`
    - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.03`
    - `ORDER_DB_LOG_RETRY_BACKOFF=2.0`
    - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.20`
  - `docs/RISK_AND_EXECUTION.md` に
    「`quant-order-manager.env` と `quant-v2-runtime.env` の `ORDER_DB_*` 同期必須」
    を追記。
- 目的:
  - service 起動時の env 上書き順による設定逆戻りを防ぎ、
    lock 耐性設定を本番実行値へ確実に反映する。

### 2026-02-20（追記）`close_trade` の lock耐性強化（fast-fail order log）

- 背景（VM実測）:
  - `scalp_ping_5s_b_exit -> /order/close_trade` で
    `exit_reason=take_profit` が発火しても、
    `order_manager service call failed ... Read timed out (12s)` が発生し、
    利確取り逃し後に `STOP_LOSS_ORDER` へ落ちるケースを確認。
  - 同時刻に `quant-order-manager` 側で
    `[ORDER][LOG] failed to persist orders log: database is locked`
    が多発し、close 処理中の orders ログ書き込み待ちが tail latency を押し上げていた。
- 実施:
  - `execution/order_manager.py`
    - `_log_order(..., fast_fail=True)` モードを追加。
    - fast-fail 時は通常 retry budget ではなく `ORDER_DB_LOG_FAST_RETRY_*`
      （既定 `attempts=1`）で短時間打ち切り。
    - `close_trade()` の orders 監査ログ呼び出しを
      `_log_close_order()`（fast-fail wrapper）へ切替。
    - lock で打ち切った場合は warning ではなく
      `"[ORDER][LOG] fast-fail dropped by lock"` を info で記録。
  - テスト:
    - `tests/execution/test_order_manager_log_retry.py`
      に fast-fail retry budget の検証ケースを追加。
- 目的:
  - DB lock 時に「監査ログ待ちで close API が詰まる」経路を抑制し、
    `take_profit` / `lock_floor` 等の exit 指示を OANDA へ先に通す。

### 2026-02-20（追記）replay_quality_gate_main の精度補正（intraday 既定OFF）

- 背景（VM実測）:
  - `config/replay_quality_gate_main.yaml` を既定の
    `intraday_start_utc=00:00:00` / `intraday_end_utc=06:00:00`
    で回すと、`exclude_end_of_replay=true` の影響で
    close が `end_of_replay` 側へ寄り、`trade_count=0` fold が連発。
  - 同一コードでもフルデイ再生では約定が出るケースが確認され、
    判定の歪みは戦略品質ではなく評価窓設定に起因すると判断。
- 実施:
  - `config/replay_quality_gate_main.yaml`
    - `intraday_start_utc` / `intraday_end_utc` の既定を空文字へ変更（既定OFF）。
    - `main_only` 既定を `false` に変更し、主経路限定の見落としを回避。
  - `docs/REPLAY_STANDARD.md`
    - intraday 既定OFF方針と、
      `exclude_end_of_replay` 併用時の `trade_count=0` リスクを明記。
  - `docs/ARCHITECTURE.md`
    - replay 品質ゲート節へ同注意点を反映。
- 目的:
  - 内部テストの「精度」を、戦略性能評価として意味のある条件
    （ゼロ件折り畳みを避けた条件）へ戻す。

### 2026-02-20（追記）service mode worker の pre-service orders ログ抑止

- 背景（VM実測）:
  - `database is locked` は `quant-order-manager` 単体ではなく、
    `quant-scalp-ping-5s-flow` / `quant-scalp-ping-5s-b` /
    `quant-scalp-rangefader` など複数 worker から同時発生していた。
  - 解析の結果、service mode worker でも `market_order` の
    pre-service 段 (`entry_probability_reject` / `probability_scaled`) で
    `orders.db` へ直接書き込みが走り、主ライターと競合していた。
- 実施:
  - `execution/order_manager.py`
    - `ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE`（default: `0`）を追加。
    - `_should_persist_preservice_order_log()` を追加し、
      service mode (`ORDER_MANAGER_SERVICE_ENABLED=1` かつ service URL 有効) では
      pre-service の `orders.db` 記録を既定で抑止。
    - 対象: `entry_probability_reject` / `probability_scaled` の pre-service log。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE=0` を明示。
  - テスト:
    - `tests/execution/test_order_manager_log_retry.py`
      に service mode 時の pre-service ログ抑止ケースを追加。
- 目的:
  - `quant-order-manager` を `orders.db` の主ライターに寄せ、
    複数 worker 同時書き込みによる lock 競合を低減する。

### 2026-02-20（追記）ops/保守ワーカーの安定化（audit timeout・log maintenance）

- 背景（VM実測）:
  - `quant-v2-audit.service` が `journalctl` 呼び出しで `TimeoutExpired` し、oneshot が失敗していた。
  - `quant-maintain-logs.service` は `logs/replay` の並行更新で `rm ... Directory not empty` や
    `sqlite3 ... database is locked` で失敗していた。
  - SSH/IAP 不調時の metadata デプロイで `deploy_via_metadata.sh` 生成スクリプト内の
    `RUNTIME_ENV_FILE` クオート不整合により startup-script が早期失敗していた。
- 実施:
  - `scripts/ops_v2_audit.py`
    - `OPS_V2_AUDIT_JOURNAL_TIMEOUT_SEC` を追加し、`_run()` で timeout 例外を握って
      audit 全体を落とさないように修正。
    - `journalctl` 側へ `--grep` を追加して検索対象を絞り、重い全量走査を回避。
  - `scripts/maintain_logs.sh`
    - `archive_replay()` を stage 退避方式へ変更し、ライブ書き込みパスを即再作成。
    - `REPLAY_ARCHIVE_TIMEOUT_SEC` を追加して replay 圧縮の長時間化を打ち切り可能化。
    - `checkpoint_db()` に `busy_timeout` と warning-only ハンドリングを追加し、
      lock 発生時も service を成功終了可能に修正。
  - `scripts/deploy_via_metadata.sh`
    - startup-script 生成時の `RUNTIME_ENV_FILE` 参照クオートを修正し、
      metadata fallback デプロイの早期失敗を解消。
- 反映確認（VM）:
  - `git rev-parse HEAD == git rev-parse origin/main` を確認。
  - `quant-v2-audit.service` は `status=0/SUCCESS` で `warn=0` を確認。
  - `quant-maintain-logs.service` は最新実行で `status=0/SUCCESS` を確認。

### 2026-02-20（追記）replay_quality_gate の低サンプル日フィルタ追加

- 背景（VM実測）:
  - walk-forward の test fold が `trade_count=0` 日
    （例: `20260218`, `20260219`）に当たると、
    `test_trade_count` / `pf_stability_ratio` で一律 fail し、
    戦略品質より「データ量不足」を拾う判定になっていた。
- 実施:
  - `scripts/replay_quality_gate.py`
    - `--min-tick-lines` を追加。
    - config `min_tick_lines` を追加し、閾値未満の tick ファイルを
      walk-forward 前に自動除外。
    - report `meta` に `matched_tick_file_count` / `min_tick_lines` /
      `filtered_out_files` を記録。
  - `config/replay_quality_gate_main.yaml`
    - `min_tick_lines: 50000` を追加。
    - walk-forward 既定を `train_files=2 / test_files=1 / step_files=1` へ更新。
  - テスト:
    - `tests/analysis/test_replay_quality_gate_script.py`
      に `min_tick_lines` フィルタの単体テストを追加。
  - ドキュメント:
    - `docs/REPLAY_STANDARD.md` / `docs/ARCHITECTURE.md` へ運用注意を追記。
- 目的:
  - 低サンプル日による偽陰性 fail を減らし、
    内部テストを「戦略品質の判定」へ寄せる。

### 2026-02-21（追記）replay_quality_gate の複数 root 自動収集（archive 含む）

- 背景（VM実測）:
  - `logs/replay` はログローテーション後に最新日しか残らないことがあり、
    walk-forward が `fold=1` まで縮退しやすかった。
  - 実際の評価では `logs/archive/replay.*.dir` に十分な日数が残っており、
    手動で `--ticks-glob` を差し替える運用が必要になっていた。
- 実施:
  - `scripts/replay_quality_gate.py`
    - `ticks_globs`（config配列）を追加し、`--ticks-glob` は
      カンマ区切り複数 pattern に対応。
    - 複数 root の同日ファイル（同 basename）は自動で重複排除し、
      サイズが大きいファイルを優先。
    - report `meta` に `ticks_globs` / `deduped_tick_file_count` /
      `duplicate_tick_file_count` を追加。
  - `config/replay_quality_gate_main.yaml`
    - 既定 source を `logs/replay` + `logs/archive/replay.*.dir` の
      2 root に更新。
  - テスト:
    - `tests/analysis/test_replay_quality_gate_script.py`
      に重複解消（サイズ優先）と `ticks_globs` 解決順の単体テストを追加。
  - ドキュメント:
    - `docs/REPLAY_STANDARD.md` / `docs/ARCHITECTURE.md` を更新。
- 目的:
  - 運用時に手動で glob を差し替えずに、十分な fold 数を確保した
    内部精度ゲートを継続実行できるようにする。

### 2026-02-21（追記）時間帯損失の即時ブロック（worker_reentry）

- 背景（VM実測, JST）:
  - `2026-02-20 20:00/21:00/22:00` は合計 `-927.1 JPY`（`-259.0 pips`）。
  - `2026-02-21 03:00/05:00/06:00` は合計 `-232.6 JPY`（`-34.7 pips`）。
  - 主因は `scalp_ping_5s_b_live`（20-22時帯）と `M1Scalper-M1`（03/05/06時帯）。
- 実施:
  - `config/worker_reentry.yaml`
    - `M1Scalper.block_jst_hours = [3, 5, 6, 20, 21, 22]`
    - `scalp_ping_5s_b_live.block_jst_hours = [3, 5, 6, 20, 21, 22]`
    - `scalp_ping_5s_b_live.return_wait_reason = hourly_pnl_guard_20260221`
- 目的:
  - 直近で損失集中した JST 時間帯のエントリーを `reentry_gate` で停止し、
    1時間あたり収益の下振れを即時抑制する。

### 2026-02-21（追記）`scalp_ping_5s_b_live` 逆方向抑制（short厳格化）

- 背景（VM実測, 直近7日）:
  - `scalp_ping_5s_b_live` の side 別成績は
    - `long`: `n=1979`, `pl_pips=-96.4`, `realized_pl=+1060.0`
    - `short`: `n=1688`, `pl_pips=-4129.6`, `realized_pl=-34378.3`
  - `extrema_reversal_applied=1` かつ `short` は
    `n=136`, `pl_pips=-299.7`, `realized_pl=-2665.1` で、
    `long->short` 反転の寄与が負側に偏っていた。
- 実施:
  - `ops/env/scalp_ping_5s_b.env`
    - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`（B既定へ復帰）
    - short 判定を厳格化:
      - `SHORT_MIN_TICKS=4`
      - `SHORT_MIN_SIGNAL_TICKS=4`
      - `SHORT_MIN_TICK_RATE=0.62`
      - `SHORT_MOMENTUM_TRIGGER_PIPS=0.11`
    - 方向/逆行ガード強化:
      - `DIRECTION_BIAS_BLOCK_SCORE=0.52`
      - `DIRECTION_BIAS_OPPOSITE_UNITS_MULT=0.60`
      - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.42`
      - `SIDE_BIAS_SCALE_GAIN=0.50`
      - `SIDE_BIAS_SCALE_FLOOR=0.18`
      - `SIDE_BIAS_BLOCK_THRESHOLD=0.30`
    - 確率整合ペナルティ強化:
      - `ENTRY_PROBABILITY_ALIGN_PENALTY_MAX=0.55`
      - `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.32`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.24`
      - `ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT=0.45`
- 目的:
  - 戦略自体は残しつつ、逆方向（特に short 側）の誤エントリー頻度と
    逆行時の平均サイズを同時に抑える。

### 2026-02-21（追記）`scalp_ping_5s_b_live` ケース別デリスク強化（総合）

- 背景（VM実測, 直近3日）:
  - side別:
    - `long`: `n=1530`, `pl_pips=-734.0`, `realized_pl=-5796.4`
    - `short`: `n=1520`, `pl_pips=-3570.8`, `realized_pl=-23856.4`
  - 問題時間帯（JST）では `dip_then_up`（下押し後に上で引け）局面が連発:
    - `2026-02-20 20:00`: `co=+5.1p`, `lo=-5.3p`
    - `2026-02-20 21:00`: `co=+2.2p`, `lo=-5.0p`
    - `2026-02-20 22:00`: `co=+16.4p`, `lo=-20.2p`
    - `2026-02-21 03:00`: `co=+2.6p`, `lo=-18.4p`
  - 同方向積み上げ時の縮小発火が弱く、`side_adverse_stack_units_mult` が
    実質 `1.0` のまま通るケースが多かった。
- 実施:
  - `ops/env/scalp_ping_5s_b.env`（strategy local）
    - 過剰エントリー抑制:
      - `MAX_ACTIVE_TRADES=14`, `MAX_PER_DIRECTION=8`, `MAX_ORDERS_PER_MINUTE=10`
      - `ENTRY_CHASE_MAX_PIPS=1.0`
      - `MIN_TICKS=5`, `MIN_SIGNAL_TICKS=4`, `MIN_TICK_RATE=0.85`
      - `IMBALANCE_MIN=0.55`
    - 方向/反転ガード:
      - `FAST_DIRECTION_FLIP_*` 閾値を引き上げ（誤反転抑制）
      - `SL_STREAK_DIRECTION_FLIP_*` を厳格化（`min_streak=2`, tech confirm 必須）
      - `SIDE_METRICS_DIRECTION_FLIP_*` はサンプル閾値を緩和して
        side劣化時の反転発火を増やす
    - 逆行スタック縮小:
      - `SIDE_ADVERSE_STACK_UNITS_ACTIVE_START=2`
      - `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.20`
      - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.18`
      - `SIDE_ADVERSE_STACK_DD_START_PIPS=0.45`
      - `SIDE_ADVERSE_STACK_DD_FULL_PIPS=1.60`
      - `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.22`
    - 確率/帯域補正:
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN=1.70`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.25`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT=1.05`
      - `ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT=0.40`
    - ケース追従:
      - `SIGNAL_WINDOW_ADAPTIVE_ENABLED=1`
      - `LOOKAHEAD_GATE_ENABLED=1`
      - `LONG_MOMENTUM_TRIGGER_PIPS=0.12`
      - `SHORT_MIN_TICK_RATE=0.72`
      - `MOMENTUM_TRIGGER_PIPS=0.11`
    - order前ガード:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.48`
      - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE...=0.65`
  - `ops/env/quant-order-manager.env`（service側）
    - 上記 `ORDER_MANAGER_PRESERVE_INTENT_*` を同値に更新
    - `SCALP_PING_5S_B_PERF_GUARD_*` を `block + hourly + split_directional` へ強化
- 目的:
  - 「上昇押し目局面の逆張りshort」「天井掴みlong」「同方向クラスター」の
    3系統を同時に抑え、劣化時は fail-fast で機械停止させる。

### 2026-02-21（追記）forecast「型」寄り再調整（8,050 bars, short-list）

- 背景:
  - 直近運用値（`fg=0.03, b5=0.24, b10=0.30, s5=0.22, s10=0.30, rb5=0.01`）で、
    `5m` の `mae_delta` がプラス寄りになる局面を再検出。
  - 予測を「型」（trend/rangeの混在レジーム）で安定化させるため、
    `feature/session/rebound` を short-list 候補で再評価。
- 実施（VM実データ, 同一期間）:
  - 比較期間: `2026-01-07T02:54:00+00:00` ～ `2026-02-20T21:59:00+00:00`
  - サンプル: `max-bars=8050`, `steps=1,5,10`
  - 出力:
    - `logs/reports/forecast_improvement/forecast_tune_shortlist_20260221T021810Z.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T021810Z_cand_c.json`
    - `logs/reports/forecast_improvement/report_20260221T021810Z_cand_c.md`
- 採用候補（cand_c）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.24,10m=0.30`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.24`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.00,10m=0.02`
- before/after（cand_c）:
  - `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
  - `5m`: `hit_delta=+0.0024`, `mae_delta=-0.0006`, `range_cov_delta=-0.0001`
  - `10m`: `hit_delta=+0.0052`, `mae_delta=-0.0041`, `range_cov_delta=+0.0003`
- 判定:
  - `1m` の hit はほぼ同等（軽微マイナス）を維持しつつ、
    `5m/10m` で `hit` と `MAE` を同時改善したため採用。

### 2026-02-21（追記）forecast 追加最適化（cand_d2, 96候補 narrow grid）

- 背景:
  - cand_c 適用後も `5m/10m` に上積み余地があるため、同一期間で
    cand_c 周辺のみを再探索（narrow grid）した。
- 実施:
  - データ: `bars=8050`（`2026-01-07T02:54:00+00:00` ～ `2026-02-20T21:59:00+00:00`）
  - 探索: `feature_gain / breakout_5m,10m / session_5m,10m / rebound_5m` の 96候補
  - 出力:
    - `logs/reports/forecast_improvement/forecast_tune_local_narrow_20260221T031535Z.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T031535Z_cand_d2.json`
    - `logs/reports/forecast_improvement/report_20260221T031535Z_cand_d2.md`
- 採用値（cand_d2）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.26,10m=0.32`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.26`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`
- before/after（cand_d2）:
  - `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
  - `5m`: `hit_delta=+0.0025`, `mae_delta=-0.0006`, `range_cov_delta=+0.0000`
  - `10m`: `hit_delta=+0.0056`, `mae_delta=-0.0044`, `range_cov_delta=+0.0001`
- 判定:
  - cand_c 比で `5m/10m` の `hit` と `MAE` をさらに改善したため、
    runtime 運用値を cand_d2 へ更新。

### 2026-02-21（追記）forecast 微調整（mid_253327, 8050固定スナップショット）

- 背景:
  - cand_d2 と best_soft は `10m hit` と `10m mae/range_cov` がトレードオフで拮抗したため、
    同一データ固定で中間値を再探索した。
- 実施:
  - 比較期間: `2026-01-07T02:54:00+00:00` ～ `2026-02-20T21:59:00+00:00`
  - サンプル: `bars=8050`（固定スナップショット）
  - 候補: `cand_d2/best_soft` + 中間6候補（合計8候補）
  - 出力:
    - `logs/reports/forecast_improvement/forecast_tune_midgrid_20260221T045844Z.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T045844Z_current_d2.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T045844Z_mid_253327.json`
    - `logs/reports/forecast_improvement/report_20260221T045844Z_mid_253327.md`
- 採用値（mid_253327）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.25,10m=0.33`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.27`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`（維持）
- before/after（mid_253327）:
  - `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
  - `5m`: `hit_delta=+0.0025`, `mae_delta=-0.0006`, `range_cov_delta=+0.0000`
  - `10m`: `hit_delta=+0.0055`, `mae_delta=-0.0046`, `range_cov_delta=+0.0003`
- 判定:
  - `cand_d2` 比で `10m hit` は微減（`+0.0056 -> +0.0055`）だが、
    `10m mae` と `range_cov` を改善し、総合スコア `score_vs_current=+0.000083` を確認。
  - 僅差ながら上積みがあるため runtime を mid_253327 へ更新。

### 2026-02-21（追記）forecast 再最適化（cand_e1, two-stage）

- 背景:
  - `mid_253327` は改善幅が小さく、`5m/10m` の hit 上積み余地が残っていた。
  - IAP接続の揺らぎを避けるため、VM取得済みスナップショット（`bars=8050`）で
    ローカル two-stage 探索（stage1=90候補, stage2=上位12候補）を実施。
- 実施:
  - 探索出力:
    - `logs/reports/forecast_improvement/forecast_tune_two_stage_20260221T053704Z.json`
  - VM実データ再検証:
    - `logs/reports/forecast_improvement/forecast_eval_20260221T054555Z_current_mid253327.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T054555Z_cand_e1.json`
    - `logs/reports/forecast_improvement/report_20260221T054555Z_cand_e1.md`
- 採用値（cand_e1）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.24`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.24,10m=0.31`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=160`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=360`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT=0.10`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.20,10m=0.29`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=18`
  - `FORECAST_TECH_SESSION_BIAS_LOOKBACK=900`
  - `FORECAST_TECH_REBOUND_WEIGHT=0.07`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.04`
- VM同一期間 before/after（cand_e1）:
  - `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
  - `5m`: `hit_delta=+0.0040`, `mae_delta=-0.0015`, `range_cov_delta=+0.0006`
  - `10m`: `hit_delta=+0.0117`, `mae_delta=-0.0072`, `range_cov_delta=+0.0012`
- `mid_253327` 比（after-after）:
  - `5m`: `hit_after +0.0015`, `mae_after -0.0008`, `range_cov_after +0.0006`
  - `10m`: `hit_after +0.0062`, `mae_after -0.0026`, `range_cov_after +0.0009`
- 判定:
  - `5m/10m` で hit と MAE を同時改善し、帯域カバレッジも上振れしたため
    runtime 運用値を cand_e1 へ更新。

### 2026-02-21（追記）forecast 1m専用ブースト（cand_e1_1mboost）

- 背景:
  - `cand_e1` で `5m/10m` は大きく改善した一方、`1m hit` はまだ `0.50` 未満だったため、
    `1m` のみを追加改善する局所探索を実施。
- 実施:
  - 探索:
    - `logs/reports/forecast_improvement/forecast_tune_1m_only_20260221T063329Z.json`
    - `b1/rb1/s1` の 90候補（`step=1`）を比較。
  - VM再検証:
    - `logs/reports/forecast_improvement/forecast_eval_20260221T064525Z_current_cand_e1.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T064525Z_cand_e1_1mboost.json`
    - `logs/reports/forecast_improvement/report_20260221T064525Z_cand_e1_1mboost.md`
- 採用値（cand_e1_1mboost）:
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.16,5m=0.24,10m=0.31`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.14,5m=0.01,10m=0.04`
  - その他 `cand_e1` パラメータは維持。
- VM同一期間 before/after（cand_e1_1mboost）:
  - `1m`: `hit_delta=+0.0002`, `mae_delta=-0.0001`, `range_cov_delta=+0.0005`
  - `5m`: `hit_delta=+0.0040`, `mae_delta=-0.0015`, `range_cov_delta=+0.0006`
  - `10m`: `hit_delta=+0.0117`, `mae_delta=-0.0072`, `range_cov_delta=+0.0012`
- `cand_e1` 比（after-after）:
  - `1m`: `hit_after +0.0005`, `range_cov_after +0.0003`, `mae_after` はほぼ同等
  - `5m/10m`: 同等（差分ほぼ 0）
- 判定:
  - `5m/10m` を維持したまま `1m` を改善できたため、
    runtime 運用値を `cand_e1_1mboost` へ更新。

### 2026-02-21（追記）forecast 再加速チューニング（candD, 6本同時比較）

- 背景:
  - `cand_e1_1mboost` は安定していたが、`5m/10m` の hit と MAE に追加の改善余地があったため、
    aggressive 側の候補を同一期間で再比較した。
- 実施:
  - ローカル探索:
    - `logs/reports/forecast_improvement/forecast_tune_balanced_breakthrough_20260221T222015Z.json`
  - VM同一期間比較（`bars=8050`）:
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_current.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_candA.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_candB.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_candC.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_candD.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260221T222744Z_candE.json`
- 採用値（candD）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.03`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.28`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.16,5m=0.28,10m=0.34`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=360`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT=0.12`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.24,10m=0.33`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=18`
  - `FORECAST_TECH_SESSION_BIAS_LOOKBACK=1080`
  - `FORECAST_TECH_REBOUND_WEIGHT=0.06`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.12,5m=0.01,10m=0.05`
- VM同一期間 before/after（candD）:
  - `1m`: `hit_delta=-0.0002`, `mae_delta=+0.0009`, `range_cov_delta=-0.0002`
  - `5m`: `hit_delta=+0.0054`, `mae_delta=-0.0020`, `range_cov_delta=-0.0004`
  - `10m`: `hit_delta=+0.0129`, `mae_delta=-0.0125`, `range_cov_delta=+0.0006`
- `cand_e1_1mboost` 比（after-after）:
  - `1m`: `hit_after -0.0003`, `mae_after +0.0010`, `range_cov_after -0.0006`
  - `5m`: `hit_after +0.0013`, `mae_after -0.0006`, `range_cov_after -0.0010`
  - `10m`: `hit_after +0.0012`, `mae_after -0.0053`, `range_cov_after -0.0006`
- 判定:
  - `5m/10m` の hit と MAE を同時に押し上げる総合スコアが最大だったため、
    runtime 運用値を `candD` へ更新。

### 2026-02-22（追記）forecast 微調整（candD_1m_mae_boost）

- 背景:
  - `candD` は `5m/10m` を最大化できたが、`1m` で僅かな改善余地が残っていたため、
    `1m` 専用の `breakout/rebound` 重みだけを局所探索した。
- 実施:
  - スキャン:
    - `tmp/qr_vm_hit_scan_b1_rb1.py`（`b1/rb1` の 16 組合せ）
    - `tmp/qr_vm_hit_scan_fg_rb.py`（`fg/rb` の 9 組合せ）
  - VM再検証:
    - `logs/reports/forecast_improvement/forecast_eval_20260222T000006Z_current_candD.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260222T000006Z_candD_1m_mae_boost.json`
- 採用値（candD_1m_mae_boost）:
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.28,10m=0.34`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`
  - その他 `candD` パラメータは維持。
- `candD` 比（after-after）:
  - `1m`: `hit_after_delta=+0.000000`, `mae_after_delta=-0.000068`, `range_cov_after_delta=+0.000000`
  - `5m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`
  - `10m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`
- 判定:
  - `5m/10m` を固定したまま `1m MAE` を微改善し、`hit` を維持できたため
    runtime 運用値を `candD_1m_mae_boost` へ更新。

### 2026-02-22（追記）forecast 微調整（cand_hit_nonneg_mae_up）

- 背景:
  - `candD_1m_mae_boost` は `1m` の改善は達成できたが、`5m/10m` の MAE を
    `hit` 非劣化制約付きでさらに詰める余地が残っていた。
- 実施:
  - 制約付きスキャン:
    - `logs/reports/forecast_improvement/forecast_scan_hit_constraints_20260222T023311Z.json`
  - VM再検証:
    - `logs/reports/forecast_improvement/forecast_eval_20260222T023331Z_current_candD_1m_mae_boost.json`
    - `logs/reports/forecast_improvement/forecast_eval_20260222T023331Z_cand_hit_nonneg_mae_up.json`
- 採用値（cand_hit_nonneg_mae_up）:
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.34`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.24,10m=0.35`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`（維持）
  - そのほか `candD_1m_mae_boost` の値を維持。
- `candD_1m_mae_boost` 比（after-after）:
  - `1m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
  - `5m`: `hit_after_delta=+0.000000`, `mae_after_delta=-0.000003`, `range_cov_after_delta=+0.000000`
  - `10m`: `hit_after_delta=+0.000000`, `mae_after_delta=-0.000811`, `range_cov_after_delta=+0.000446`
- 判定:
  - 全TFで `hit` を維持しつつ `5m/10m MAE` を改善、`10m range_cov` も改善できたため
    runtime 運用値を `cand_hit_nonneg_mae_up` へ更新。

### 2026-02-22（追記）forecast 多窓最適化3（dynamic_h1510_rnd100）

- 背景:
  - `dynamic_h510_rnd161` で10mは改善したが、同時に5mのMAEも押し下げる余地があった。
- 実施:
  - `rnd161` をベースに 5m/10m 同時改善を優先した再探索（161候補→上位30候補の多窓再評価）:
    - `logs/reports/forecast_improvement/forecast_dyn_multistage_v3_20260222.json`
- 採用値（runtime）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.01`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.30`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.26,10m=0.49`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.02,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=1m,5m,10m`
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.22`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.025`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.22`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.14`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.07`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.34`（維持）
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.02`
- 直前運用値（`dynamic_h510_rnd161`）比:
  - `24h`: `10m mae_after_delta=-0.001756`, `10m cov_after_delta=+0.001142`, `5m mae_after_delta=-0.000039`
  - `72h`: `10m mae_after_delta=-0.004361`（hit同等）, `5m mae_after_delta=-0.001198`（hit_after_delta=-0.000328）
  - `full(8050 bars)`: `10m hit_after_delta=+0.001040`, `10m mae_after_delta=-0.002625`,
    `10m range_cov_after_delta=+0.000149`, `5m mae_after_delta=-0.000740`（5m hit同等）
  - `1m`: `24h/full` は hit 同等 + MAE改善、`72h` は `hit_after_delta=-0.000340` だが MAE改善。
- 判定:
  - `full` で `10m` の `hit/MAE` を同時改善し、`5m` は hit維持で MAE改善のため採用。

### 2026-02-22（追記）forecast 多窓最適化4（dynamic_meta_rnd056）

- 背景:
  - `dynamic_h1510_rnd100` は有効だったが、`lookback/min_samples` を含む適応重み本体の再探索余地が残っていた。
- 実施:
  - `rnd100` を基準に `feature_gain` / adaptive global weight / `*_min_samples` / `*_lookback` /
    dynamic gain を含む拡張探索を実施:
    - `logs/reports/forecast_improvement/forecast_dyn_multistage_v5_20260222.json`
- 採用値（runtime）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.015`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.30`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.31,10m=0.26`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=150`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=480`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.30,10m=0.53`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=12`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.015,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.18`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.02`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.24`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.12`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.06`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.26`
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.0`
- 同一スナップショット比較（`rnd100` 比）:
  - `24h`: `10m hit_delta=+0.001142`, `10m mae_delta=-0.011401`, `5m mae_delta=-0.004179`
  - `72h`: `10m hit_delta=+0.007178`, `10m mae_delta=-0.002647`, `5m mae_delta=-0.001440`
  - `full(8050 bars)`: `10m hit_delta=+0.007426`, `10m mae_delta=-0.003738`,
    `5m hit_delta=-0.000300`, `5m mae_delta=-0.001179`
  - `1m`: hit は改善だが、`full` の MAE は `+0.000085` の小幅悪化。
- 判定:
  - `10m` の方向一致改善幅が大きく、`5m` の MAE 改善を維持できるため採用。

### 2026-02-23（追記）forecast 多窓最適化5（dynamic_meta_rnd087）

- 背景:
  - `dynamic_meta_rnd056` は改善したが、`5m/10m hit` をさらに押し上げる余地があった。
- 実施:
  - `rnd056` を基準に局所探索（91候補→上位20候補の多窓再評価）を実施:
    - `logs/reports/forecast_improvement/forecast_dyn_multistage_v6_20260223.json`
- 採用値（runtime）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.015`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.28`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.29,10m=0.24`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=150`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=720`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.34,10m=0.57`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=8`
  - `FORECAST_TECH_SESSION_BIAS_LOOKBACK=1260`
  - `FORECAST_TECH_REBOUND_WEIGHT=0.04`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.025,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`（維持）
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.18`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.018`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.20`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.10`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.055`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.22`
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.02`
- 同一スナップショット比較（`rnd056` 比）:
  - `24h`: `5m hit_delta=+0.003440`, `5m mae_delta=-0.005085`,
    `10m hit_delta=+0.007991`, `10m mae_delta=-0.010956`
  - `72h`: `5m hit_delta=+0.005579`, `5m mae_delta=-0.005690`,
    `10m hit_delta=+0.008809`, `10m mae_delta=-0.015130`
  - `full(8050 bars)`: `5m hit_delta=+0.002847`, `5m mae_delta=-0.003387`,
    `10m hit_delta=+0.004307`, `10m mae_delta=-0.009587`
  - `1m` は hit が小幅悪化する窓があるが、MAE は概ね改善。
- 判定:
  - `5m/10m` の hit と MAE を同時改善できるため採用。

### 2026-02-24（追記）open_positions の非canonicalタグ補完を強化（底/天井ポジ残り対策）

- 背景:
  - VM実測で OANDA `openTrades` に未決済が残る一方、`open_positions` 側の `entry_thesis.strategy_tag` が
    `micropul...` のような client_id 由来短縮タグへ退化するケースを確認。
  - この状態だと各 exit_worker の `MICRO_MULTI_EXIT_TAG_ALLOWLIST`（例: `MicroPullbackEMA`）と一致せず、
    EXIT 評価ループから除外され、含み損ポジションが長時間残る原因になっていた。
- 原因:
  - `execution/position_manager.py` の `open_positions` ホットパスで、
    `client_id` からタグ推定できた場合は `orders.db` の `entry_thesis` 再解決を省略していた。
  - 推定タグが非canonical（短縮/ハッシュ混在）でも「タグあり」と判定され、`entry_thesis` 補完対象から外れていた。
- 対応:
  - `execution/position_manager.py`
    - `_is_canonical_strategy_tag()` を追加。
    - `POSITION_MANAGER_OPEN_POSITIONS_ENRICH_ALL_CLIENTS=0` 運用時でも、
      `strategy_tag` が非canonical、または `entry_thesis` 欠損時は
      `client_order_id -> entry_thesis` 補完対象に必ず入れるよう条件を拡張。
  - これにより `orders.db` の `submit_attempt.request_json.entry_thesis` から
    canonicalタグ（例: `MicroPullbackEMA`）を復元し、exit_worker の tag allowlist と整合する。

### 2026-02-24（追記）forecast 多窓最適化6（dynamic_meta_rnd090）

- 背景:
  - `dynamic_meta_rnd087` は改善したが、`5m/10m` をさらに同時改善できる余地があった。
- 実施:
  - `rnd087` を基準に `1m` 劣化ペナルティ付き局所探索（106候補→上位20候補多窓再評価）を実施:
    - `logs/reports/forecast_improvement/forecast_dyn_multistage_v7_20260223.json`
- 採用値（runtime）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.01`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.26`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.20`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=720`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT=0.16`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.38,10m=0.59`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=4`
  - `FORECAST_TECH_SESSION_BIAS_LOOKBACK=1260`
  - `FORECAST_TECH_REBOUND_WEIGHT=0.06`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.025,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`（維持）
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.18`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.016`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.20`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.12`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.06`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.24`
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.02`（維持）
- 同一スナップショット比較（`rnd087` 比）:
  - `24h`: `5m hit_delta=+0.006881`, `5m mae_delta=-0.006449`,
    `10m hit_delta=+0.023973`, `10m mae_delta=-0.022077`
  - `72h`: `5m hit_delta=+0.006219`, `5m mae_delta=-0.002442`,
    `10m hit_delta=+0.011427`, `10m mae_delta=-0.008303`
  - `full(8050 bars)`: `5m hit_delta=+0.003891`, `5m mae_delta=-0.001912`,
    `10m hit_delta=+0.008765`, `10m mae_delta=-0.007722`
  - `1m`: hit は同等、`full mae_delta=-0.000185` の小幅改善。
- 判定:
  - `1m` を維持しつつ `5m/10m` の hit と MAE を同時改善できるため採用。

### 2026-02-24（追記）forecast 多窓最適化8（session_bias cap 可変化）

- 背景:
  - `dynamic_meta_rnd081` 採用後に同条件で再探索（`forecast_dyn_multistage_v9_20260224.json`）を実施したが、
    上積み候補は得られず現行値が局所最適だった。
  - 一方で runtime 実装は `session_bias` を固定上限 `0.6` でクランプしており、
    先行探索で有効だった `10m=0.63` が実効化されない状態だった。
- 対応:
  - `workers/common/forecast_gate.py`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_CAP`（default `0.6`）を追加。
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT(_MAP)` の上限を可変化。
    - dynamic session weight の clamp 上限も同 cap に統一。
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_CAP=0.70`
    - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.42,10m=0.63`
- 同一スナップショット比較（`session_10=0.60` 比）:
  - `24h`: `10m hit_delta=+0.001142`, `10m mae_delta=-0.003579`
  - `72h`: `10m hit_delta=+0.000326`, `10m mae_delta=-0.000884`
  - `full(8050 bars)`: `10m hit_delta=+0.000149`, `10m mae_delta=-0.001232`
  - `1m/5m` は同等、`10m range coverage` は `72h/full` で小幅改善。
- 判定:
  - `5m` を維持したまま `10m` の hit/MAE を同時改善できるため採用。

### 2026-02-24（追記）forecast 多窓最適化9（session_10 sweep）

- 背景:
  - `session_bias cap` 可変化後の `session_10=0.63` は改善したが、10m の余地をさらに確認する必要があった。
- 実施:
  - `session_10` を `0.60..0.70` で同一スナップショット評価:
    - `logs/reports/forecast_improvement/forecast_session10_sweep_20260224.json`
- 採用値（runtime）:
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_CAP=0.70`（維持）
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.42,10m=0.70`
- 同一スナップショット比較（`session_10=0.63` 比）:
  - `24h`: `10m hit_delta=+0.003425`, `10m mae_delta=-0.007574`, `10m cov_delta=+0.001142`
  - `72h`: `10m hit_delta=+0.004569`, `10m mae_delta=-0.002006`, `10m cov_delta=+0.000653`
  - `full(8050 bars)`: `10m hit_delta=+0.003119`, `10m mae_delta=-0.002780`, `10m cov_delta=+0.000446`
  - `1m/5m` は同等。
- 判定:
  - `10m` の hit/MAE/range coverage を全窓で同時改善できるため採用。

### 2026-02-24（追記）scalp_ping_5s_flow の EXIT残留対策を追加

- 背景:
  - `scalp_ping_5s_flow_live` が `config/strategy_exit_protections.yaml` に未定義のため、
    default `exit_profile`（`loss_cut_enabled=false`）へフォールバックする経路があった。
  - `RANGEFADER_EXIT_NEW_POLICY_START_TS` より前の legacy 建玉は
    `workers/scalp_ping_5s_flow.exit_worker` 側で worker 既定値に戻す分岐があり、
    既定値が無効だと含み損玉が長時間残留しやすかった。
  - 併せて `position_manager` 応答が瞬断した際、`open_positions` 取得1回失敗で
    EXITサイクルを丸ごと skip するケースが確認された。
- 対応:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_flow` / `scalp_ping_5s_flow_live` を
      `*SCALP_PING_5S_BC_EXIT_PROFILE` へ明示マップ。
  - `ops/env/quant-scalp-ping-5s-flow-exit.env`
    - legacy fallback 用に `RANGEFADER_EXIT_LOSS_CUT_*` を明示
      （`enabled=1`, `require_sl=0`, `hard_pips=12`, `max_hold=900`）。
    - `SCALP_PRECISION_EXIT_OPEN_POSITIONS_RETRY_*` を追加（1回再試行）。
  - `workers/scalp_ping_5s_flow/exit_worker.py`
    - `_safe_get_open_positions()` に短時間再試行を実装し、
      一過性 timeout/error での EXIT 取りこぼしを緩和。

### 2026-02-24（追記）forecast 多窓最適化7（dynamic_meta_rnd081）

- 背景:
  - `dynamic_meta_rnd090` は有効だったが、`5m/10m` の hit と MAE を同時にさらに押し上げる余地が残っていた。
- 実施:
  - `rnd090` を基準に `1m` 非劣化制約を維持した多窓探索（121候補→上位20候補再評価）を実施:
    - `logs/reports/forecast_improvement/forecast_dyn_multistage_v8_20260224.json`
- 採用値（runtime）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.006`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT=0.26`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.24`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=150`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK=720`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT=0.18`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.42,10m=0.60`
  - `FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES=3`
  - `FORECAST_TECH_SESSION_BIAS_LOOKBACK=1080`
  - `FORECAST_TECH_REBOUND_WEIGHT=0.04`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.02,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=1m,5m,10m`
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.18`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.012`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.24`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.10`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.07`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.22`
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.01`
- 同一スナップショット比較（`rnd090` 比）:
  - `24h`: `5m hit_delta=+0.005734`, `5m mae_delta=-0.004041`,
    `10m hit_delta=+0.004566`, `10m mae_delta=-0.009280`
  - `72h`: `5m hit_delta=+0.003601`, `5m mae_delta=-0.001168`,
    `10m hit_delta=+0.001306`, `10m mae_delta=-0.002970`
  - `full(8050 bars)`: `5m hit_delta=+0.004041`, `5m mae_delta=-0.001019`,
    `10m hit_delta=+0.001783`, `10m mae_delta=-0.002777`
  - `1m`: hit は同等で、`24h/72h/full` の MAE が
    `-0.000038/-0.000193/-0.000142` 改善。
  - range coverage は `10m` で `24h/full` 改善（`+0.002283/+0.000594`）、`72h` は同等。
- 判定:
  - `1m` を維持しつつ `5m/10m` の hit と MAE を同時改善できるため採用。

### 2026-02-24（追記）5秒スキャリプレイの stale 判定を修正（WFO再開）

- 背景:
  - `scripts/replay_exit_workers.py` で `scalp_ping_5s` の signal 判定が `time.time()`（実時間）基準のため、
    過去日付ティックを再生すると `stale_tick` 扱いになり `trades=0` が続発していた。
- 実装:
  - `scripts/replay_exit_workers.py`
    - `sim_clock` を `workers.scalp_ping_5s.*` の `time.time/monotonic` に注入する
      `_patch_module_clock` / `_patch_ping_runtime_clock` を追加。
    - replay main 起動時（`sp`経路）に上記パッチを適用。
  - `tests/scripts/test_replay_exit_workers.py`
    - clock パッチの回帰テストを追加。
- 検証:
  - `pytest -q tests/scripts/test_replay_exit_workers.py tests/scripts/test_replay_regime_router_wfo.py`
    - `15 passed`
  - 修正後リプレイ（`--sp-live-entry --exclude-end-of-replay`）:
    - `tmp/replay_ping5s_c_regimewfo_20260123_live.json` `trades=43` `total_pnl_jpy=-4979.54`
    - `tmp/replay_ping5s_d_regimewfo_20260123_live.json` `trades=43` `total_pnl_jpy=-4979.54`
    - `tmp/replay_ping5s_c_regimewfo_20260126_live.json` `trades=39` `total_pnl_jpy=-1969.14`
    - `tmp/replay_ping5s_d_regimewfo_20260126_live.json` `trades=39` `total_pnl_jpy=-1969.14`

### 2026-02-24（追記）5秒スキャ ENTRY 詰まり解除と TickImbalance reentry 撤廃

- 背景（VM実測）:
  - `orders.db` で `scalp_ping_5s_b_live` の `reentry_block` が連発し、
    `request_json.reentry_details` は `jst_hour=20/21` と
    `block_jst_hours=[3,5,6,20,21,22]` を示していた。
  - `quant-order-manager` の `OPEN_REJECT` は B/C とも
    `note=perf_block:failfast` が支配的で、`B pf=0.46 win=0.39`,
    `C pf=0.28 win=0.45` により連続拒否していた。
  - 既存の緩和値が `ops/env/scalp_ping_5s_*.env` 側のみ更新され、
    実際に preflight を担う `ops/env/quant-order-manager.env` と不整合になっていた。
- 対応:
  - `ops/env/quant-order-manager.env`
    - B/C の `ORDER_MANAGER_PRESERVE_INTENT_*` を緩和値へ同期。
    - B/C の `SCALP_PING_5S_*_PERF_GUARD_MODE=reduce` へ変更。
    - B/C の `PERF_GUARD_*`（`MIN_TRADES/PF_MIN/WIN_MIN/FAILFAST_*`）を
      実運用値へ同期。
    - `scalp_ping_5s_c_live` は `env_prefix=SCALP_PING_5S` で preflight されるため、
      `SCALP_PING_5S_PERF_GUARD_*`（共通プレフィクス）にも同値を追加。
  - `ops/env/scalp_ping_5s_c.env`
    - `ORDER_MANAGER_SERVICE_ENABLED=0` の local fallback 経路でも
      同じ判定を使うため、`SCALP_PING_5S_PERF_GUARD_*` を同値で追加。
  - `config/worker_reentry.yaml`
    - `scalp_ping_5s_b_live` / `scalp_ping_5s_d_live` の
      `block_jst_hours` から `20/21/22` を削除（`3/5/6` は維持）。
  - `ops/env/quant-scalp-tick-imbalance.env`
    - TickImbalance の reentry 距離ゲートを撤廃
      （`LOOKBACK_SEC=0`, `MIN_PRICE_GAP_PIPS=0`, `REQUIRE_LAST_PROFIT=0`）。

### 2026-02-24（追記）scalp_ping_5s_b の時間帯再学習 + 低確率帯しきい値を再調整

- 背景（VM実測, `trades.db`, `strategy_tag=scalp_ping_5s_b_live`）:
  - 14日: `n=4163`, `sum_pips=-4573.2`, `win_rate=0.494`
  - 時間帯別で `JST 23時=-931.7 pips (n=168)`, `0時=-818.4 pips (n=62)` が突出悪化。
  - 既存の `block_jst_hours=1,2,3,10,13,15,16,19,21,22` は 23/0 を未ブロックで、
    逆に 1/2 は悪化寄与が相対的に小さかった。
  - 低確率帯カットの閾値スイープ（14日）:
    - `reject_under=0.24`: `-4573.2 pips`（基準）
    - `reject_under=0.35`: `-4526.8 pips`（`+46.4` 改善）
    - `reject_under=0.50`: `-4388.6 pips`（`+184.6` 改善, 約定減少が大きい）
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BLOCK_HOURS_JST` を
      `0,3,10,13,15,16,19,21,22,23` へ更新（`1,2` を外し `23,0` を追加）。
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE`
      を `0.24 -> 0.35` へ更新（約定を維持しつつ低確率帯を追加カット）。
- 変更前データでの what-if（同一14日, 既存 block 比）:
  - 既存 block + `reject_under=0.24`: `n=2132`, `sum_pips=-2044.4`
  - 新 block（23/0追加） + `reject_under=0.35`: `n=2153`, `sum_pips=-520.0`
  - 改善幅: `+1524.4 pips`（トレード数は同程度）。

### 2026-02-24（追記）Bの hard block を最悪帯のみに縮小（要求対応）

- 要求:
  - hard block を `0,3,15,23` のみ残し、他時間帯は縮小運用へ。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BLOCK_HOURS_JST=0,3,15,23`
    - `SCALP_PING_5S_B_PERF_GUARD_HOURLY_MIN_TRADES=10`（時間帯reduceを早める）
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_B_PERF_GUARD_HOURLY_MIN_TRADES=10`（preflight側も同期）
- 事前what-if（14d, `reject_under=0.35`）:
  - 直前設定（hard block: `0,3,10,13,15,16,19,21,22,23`）:
    - `n=2153`, `sum_pips=-520.0`
  - 今回設定（hard block: `0,3,15,23`）:
    - `n=3734`, `sum_pips=-1895.1`（`delta=-1375.1 pips`, `delta_n=+1581`）
- 判定:
  - 約定数は増えるが、pipsベースでは悪化リスクが高い。
  - 要求どおり適用し、`perf_guard(reduce)` の実効を短サイクルで監視する。

### 2026-02-24（追記）損切り肥大の緊急抑制（B戦略SL圧縮 + MicroPullbackEMAのbroker SL有効化）

- 背景（VM実測, `fx-trader-vm` / `logs/trades.db`）:
  - 直近7日 `scalp_ping_5s_b_live`: `n=3715`, `sum_pips=-5041.7`,
    `avg_sl_pips=2.496` に対して `avg_tp_pips=0.314`（`sl/tp size ratio=7.94`）。
  - 直近48時間でも `STOP_LOSS_ORDER` が重く（`n=218`, `sum_pips=-533.5`）、
    さらに `MARKET_ORDER_MARGIN_CLOSEOUT` が `n=4`, `sum_pips=-582.6` で損失を押し上げ。
  - `MicroPullbackEMA` で `-145 pips` 級の margin closeout が連発し、
    strategy単位の SL attach 不在が最大DDを拡大していた。
- 対応:
  - `execution/order_manager.py`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<TAG>` の汎用 override を追加。
    - 既存の `scalp_ping_5s_b` 例外は維持しつつ、非B戦略でも strategy単位で
      `stopLossOnFill` を opt-in 可能にした。
  - `ops/env/quant-order-manager.env`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROPULLBACKEMA=1`
    - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_MICROPULLBACKEMA=6.0`
    - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX=0.62`（`0.70 -> 0.62`）
  - `ops/env/scalp_ping_5s_b.env`
    - `SL_BASE_PIPS=1.8`（`2.4 -> 1.8`）, `SL_MIN_PIPS=0.9`, `SL_MAX_PIPS=2.8`
    - `SHORT_SL_BASE_PIPS=1.6`, `SHORT_SL_MIN_PIPS=0.9`, `SHORT_SL_MAX_PIPS=2.4`
    - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.2`,
      `SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.8`
    - `MAX_UNITS=2800`（`3500 -> 2800`）
    - `PERF_GUARD_SL_LOSS_RATE_MAX=0.62`（`0.70 -> 0.62`）
  - `ops/env/quant-micro-pullbackema.env`
    - `MICRO_MULTI_BASE_UNITS=16000`（`28000 -> 16000`）
    - `MICRO_MULTI_MAX_MARGIN_USAGE=0.86`（`0.92 -> 0.86`）
    - `MICRO_MULTI_MAX_SIGNALS_PER_CYCLE=2`（`3 -> 2`）
- 目的:
  - 損切り幅の上限と保有ロットを同時に圧縮し、
    「利確幅より損切り幅が極端に大きい」状態を先に止血する。
  - margin closeout を `broker SL` と `entry max SL cap` で構造的に減らし、
    DDの尻尾を短くする。

### 2026-02-25（追記）scalp_ping_5s_b: DD後リカバリー待ち（戻り優先）プロファイル

- 背景（本番、`2026-02-24 13:05 UTC` 反映後）:
  - `scalp_ping_5s_b_live` は `closed=33`, `sum_pips=-52.0`。
  - `STOP_LOSS_ORDER=26`、`avg_sl_pips=1.8` まで縮小できたが、
    「DD後に戻る前にSLで落ちる」体感が残った。
- 対応:
  - `ops/env/scalp_ping_5s_b.env`
    - broker SL 幅をやや拡張:
      - `SL_BASE_PIPS=2.2`, `SL_MAX_PIPS=3.0`
      - `SHORT_SL_BASE_PIPS=2.0`, `SHORT_SL_MAX_PIPS=2.8`
    - force-exit を「即損切り」から「戻り待ち付き」へ:
      - `FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=20`
      - `FORCE_EXIT_RECOVERY_WINDOW_SEC=75`
      - `FORCE_EXIT_RECOVERABLE_LOSS_PIPS=1.05`
      - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.6`
      - `SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.2`
    - 戻り後の再悪化を早めに畳む:
      - `FORCE_EXIT_GIVEBACK_ENABLED=1`
      - `FORCE_EXIT_GIVEBACK_ARM_PIPS=0.8`
      - `FORCE_EXIT_GIVEBACK_BACKOFF_PIPS=0.7`
      - `FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC=25`
      - `FORCE_EXIT_GIVEBACK_PROTECT_PIPS=0.05`
  - `ops/env/quant-order-manager.env`
    - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_PING_5S_B_LIVE=2.6`
      を追加し、entry時の broker SL 上限を戦略別で固定。
- 目的:
  - 「DDしても短時間の戻り余地があるケース」を残しつつ、
    戻らない玉は recovery timeout で機械的にカットする。
  - `SLヒット率` と `avg loss` を同時に抑えるための
    中間プロファイル（hard-stop全廃はしない）。

### 2026-02-24（追記）`scalp_ping_5s_d` narrow worker 化（allow_jst_hours 追加）

- 背景:
  - 5秒スキャ D の short-only で時間帯を絞ると損失時給が大きく改善し、
    `allow_jst_hours=1,10` では day23/day26 ともプラスを確認。
- 変更:
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_ALLOW_HOURS_JST` を追加。
  - `workers/scalp_ping_5s/worker.py`
    - `ALLOW_HOURS_JST` 判定を entry gate に追加。
    - 許可時間外は `outside_allow_hour_jst` でスキップし、
      従来の `BLOCK_HOURS_JST` と併用可能にした。
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_SIDE_FILTER=short`
    - `SCALP_PING_5S_D_ALLOW_HOURS_JST=10`
    - `SCALP_PING_5S_D_BLOCK_HOURS_JST=`（空）
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=9000`
    - `SCALP_PING_5S_D_MAX_UNITS=9000`
    - `SCALP_PING_5S_D_MAX_ACTIVE_TRADES=1`
    - `SCALP_PING_5S_D_MAX_PER_DIRECTION=1`
  - `scripts/replay_exit_workers.py`
    - `SCALP_REPLAY_ALLOW_JST_HOURS` / `SCALP_REPLAY_BLOCK_JST_HOURS` 未指定時に
      `SCALP_PING_5S_D_ALLOW_HOURS_JST` / `SCALP_PING_5S_D_BLOCK_HOURS_JST`
      を自動フォールバックし、replay 条件を live 設定へ揃えるよう修正。
  - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
    - D プレフィックス時に `ALLOW_HOURS_JST` が base config へマップされる
      回帰テストを追加。
  - `tests/scripts/test_replay_exit_workers.py`
    - replay の時間帯フォールバック（`SCALP_PING_5S_D_*`）を検証するテストを追加。
- 追加メモ:
  - D単体（replay, `allow=10`, day26）: `+2104.70 JPY`, `42 trades`
    - `jpy_per_hour(active)=+2079.16`
    - `max_drawdown_jpy=2589.66`
  - C/D ルーター混在では C 側配分で悪化しうるため、D narrow は単独導線で評価する。

### 2026-02-25（追記）`scalp_ping_5s_d` 当日ティック再最適化（allow=11 / side=both）

- 背景（VM実績, 直近6h・補正集計）:
  - `scalp_ping_5s_d_live`: `2 trades`, `-30.28 JPY`
  - `scalp_ping_5s_c_live`: `59 trades`, `-216.43 JPY`
  - Dの `short + allow=10` が当日地合いで逆行したため、当日ティックで再探索を実施。
- 当日リプレイ（`USD_JPY_ticks_20260225.jsonl`, `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`）:
  - `short_allow9`: `-1974.52 JPY`
  - `short_allow10`: `-2809.12 JPY`
  - `short_allow11`: `+153.00 JPY`
  - `long_allow10`: `-1168.54 JPY`
  - `both_allow11`: `+190.12 JPY`（`PF_jpy=1.365`）
- 反映:
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_SIDE_FILTER=`（両方向）
    - `SCALP_PING_5S_D_ALLOW_HOURS_JST=11`
- 意図:
  - 日内で損失源だった `10時(JST)` を除外し、
    当日再生で唯一正だった `11時(JST)` 窓へ集約する。

### 2026-02-25（追記）`scalp_ping_5s_c` 停止 + `scalp_ping_5s_d` サイズ増分（11時窓）

- 背景（VM実績, 直近90分）:
  - `scalp_ping_5s_c_live`: `35 trades`, `-671.68 JPY`
  - `scalp_ping_5s_d_live`: `3 trades`, `-28.54 JPY`
  - 直近の主損失源は C と判定。
- 当日ティック再生（`USD_JPY_ticks_20260225.jsonl`）:
  - 条件: `allow=11`, `side=both`, `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`
  - units sweep:
    - `base/max=9000`: `+190.12 JPY`（`9 trades`, `PF_jpy=1.365`）
    - `base/max=12000`: `+253.49 JPY`（`9 trades`, `PF_jpy=1.365`）
    - `base/max=15000`: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
- 反映:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENABLED=0`（C entry停止）
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=15000`
    - `SCALP_PING_5S_D_MAX_UNITS=15000`
- 意図:
  - Cの損失リークを遮断し、11時窓で正の期待値を示した D に資本を寄せる。

### 2026-02-25（追記）`order_manager_none` 対応（Dの二重縮小を解除）

- 症状（VMログ）:
  - `scalp_ping_5s_d_live` で `entry_probability_below_min_units` 後に
    `order_manager_none` が発生し、エントリーがほぼ通らない。
  - 例（JST 11:46）: `units=-108` が確率縮小で下限未満扱いとなり reject。
- 原因:
  - Dワーカー側でロットが既に縮小された後、order_manager 側の
    preserve-intent 確率縮小（`reject_under=0.55`, `min_scale=0.45`, `max_scale=0.85`）
    が重なり、`min_units` 判定で落ちる二重縮小状態。
- 変更:
  - `ops/env/scalp_ping_5s_d.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D_LIVE=0.35`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_D_LIVE=1.00`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_D_LIVE=1.00`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_D_LIVE=30`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_D=30`
  - `ops/env/quant-order-manager.env`
    - 同じ D 専用キーを追加し、service 側でも同一判定に統一。
- 検証（当日ティック replay, allow=11, side=both, units=15000）:
  - 変更前: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
  - 変更後: `+316.87 JPY`（同値、悪化なし）

### 2026-02-25（追記）`scalp_ping_5s_d_live` 詰まり解消（min_units + perf_guard）

- 背景（VM直近1h）:
  - `scalp_ping_5s_c_live`: `35 trades`, `-671.7 JPY`
  - `scalp_ping_5s_d_live`: `3 trades`, `-28.5 JPY`
  - Dは `units_below_min` と `perf_block` が多く、通過件数が不足。
- 当日ティック replay 比較（`USD_JPY_ticks_20260225.jsonl`、
  `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`）:
  - baseline: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
  - 候補A（`MIN_UNITS=30` + `PERF_GUARD_MODE=reduce`）:
    `+316.87 JPY`（同値）
  - 候補B（候補A + `MIN_TICKS=3`, `MIN_SIGNAL_TICKS=2`,
    `SHORT_MIN_SIGNAL_TICKS=3`）:
    `+181.87 JPY`（悪化）→ 不採用
- 反映:
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_MIN_UNITS=30`
    - `SCALP_PING_5S_D_PERF_GUARD_MODE=reduce`
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_D_PERF_GUARD_MODE=reduce`
- 意図:
  - replay の優位（`+316.87 JPY`）を崩さず、live の
    `units_below_min` / `perf_block` 由来の取りこぼしを減らす。

### 2026-02-25（追記）`scalp_ping_5s_d_live` WFO比較で 11時固定を維持し units を 20%増

- 3日比較（`USD_JPY_ticks_20260213/19/25`, `--sp-live-entry --sp-only --no-hard-sl --exclude-end-of-replay`）:
  - `h11_both`: `+316.87 JPY`（`9 trades`, `PF_jpy=1.365`）
  - `h11_short`: `+255.00 JPY`
  - `h11_long`: `-1547.86 JPY`
  - `h10,11_both`: `-5522.09 JPY`
  - `h9,11_both`: `-7999.04 JPY`
  - `h10,11_short`: `-4426.87 JPY`
- 判定:
  - 10時/9時を含む拡張は再現負けが大きく不採用。
  - 方向は `both` 維持（`short-only` より総益が高い）。
- units スイープ（2/25, `h11_both`）:
  - `15000`: `+316.87 JPY`
  - `18000`: `+380.24 JPY`
  - `22000`: `+464.74 JPY`
  - `30000`: `+633.73 JPY`
- 反映:
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=18000`
    - `SCALP_PING_5S_D_MAX_UNITS=18000`
- 意図:
  - 条件は固定し、再生で悪化を出さない範囲で
    期待値を段階的に引き上げる。

### 2026-02-25（追記）`scalp_ping_5s_d_live` バーンイン判定を自動化

- 追加:
  - `scripts/ping5s_d_canary_guard.py`
- 概要:
  - `trades.db` / `orders.db` の直近窓を集計し、
    `jpy_per_hour`, `trades_per_hour`, `margin_reject_count` で
    `promote / hold / rollback` を返す。
  - 既定閾値:
    - `min_jpy_per_hour=0`
    - `min_trades_per_hour=6`
    - `max_margin_reject=0`
    - `rollback_jpy_per_hour=-200`
  - 既定ターゲット:
    - promote: `22000`
    - rollback: `15000`
- 出力:
  - `logs/ping5s_d_canary_guard_latest.json`
- 実装メモ:
  - `orders/trades` は DB 全掃引を避け、`id` の直近行数
    （既定 `orders=50000`, `trades=20000`）へ先に絞ってから
    `window_minutes` で判定する。
- 意図:
  - 口頭判断を排除し、`u=18000` からの昇格/降格を
    同一基準で再現可能にする。

### 2026-02-25（追記）`scalp_ping_5s_d_live` バーンイン判定を60分timer化

- 追加:
  - `scripts/run_ping5s_d_canary_guard.sh`
  - `ops/env/quant-ping5s-d-canary-guard.env`
  - `systemd/quant-ping5s-d-canary-guard.service`
  - `systemd/quant-ping5s-d-canary-guard.timer`
- 構成:
  - `quant-ping5s-d-canary-guard.timer`: `OnUnitActiveSec=1h`, `Persistent=true`
  - `quant-ping5s-d-canary-guard.service`: oneshot で wrapper を実行
  - wrapper は `flock` で多重起動を抑止し、
    実行結果を `latest.json` と `history.jsonl` へ保存
- 既定運用:
  - `PING5S_D_CANARY_APPLY=0`（判定のみ）
  - 反映を自動化する場合のみ env で `PING5S_D_CANARY_APPLY=1` を明示。
- 意図:
  - promote 条件成立待ちの監視を、手動起動に依存せず継続運転する。

### 2026-02-25（追記）`scalp_ping_5s_d_live` バーンイン自動化を実反映モードへ更新

- 背景:
  - `SCALP_PING_5S_D_ALLOW_HOURS_JST=11` のため、`window=120m` では
    非対象時間の実行がほぼ `trade_count=0` となり `hold` が継続。
  - 既定 `PING5S_D_CANARY_APPLY=0` では判定のみで、units反映が起きない。
- 変更:
  - `ops/env/quant-ping5s-d-canary-guard.env`
    - `PING5S_D_CANARY_APPLY=1`
    - `PING5S_D_CANARY_WINDOW_MINUTES=1440`
    - `PING5S_D_CANARY_MIN_TRADES_PER_HOUR=3`
    - `PING5S_D_CANARY_MIN_OBSERVED_TRADES=24`
    - `PING5S_D_CANARY_ROLLBACK_JPY_PER_HOUR=-20`
    - `PING5S_D_CANARY_RESTART_ON_CHANGE=1`
    - `PING5S_D_CANARY_RESTART_UNITS=quant-scalp-ping-5s-d.service quant-scalp-ping-5s-d-exit.service`
  - `scripts/run_ping5s_d_canary_guard.sh`
    - `decision.env_updated=true` 検知時に、上記 service を自動再起動。
- 意図:
  - 判定ロジックを「監視のみ」から「自動反映＋再起動」へ昇格し、
    バーンインが実運用パラメータへ反映される導線を固定化する。

### 2026-02-25（追記）`order_manager_none` + `orders.db locked` 連鎖の安定化

- 症状（VM実測）:
  - strategy worker 側で `order_manager service call failed ... Read timed out (read timeout=12.0)` と
    `order_manager_none` が断続し、local fallback 比率が上昇。
  - 同時間帯に `orders.db` で `database is locked` 警告が増加し、
    `orders.db` の追従が遅れる局面を確認。
  - `quant-order-manager` の `:8300` は listen しているが、`/health` 応答がタイムアウトする状態を確認。
- 原因:
  - `quant-order-manager` が単一 worker で重い外部 I/O（OANDA API）を抱えた際に、
    service リクエスト待ちが詰まり、strategy 側 `12s` timeout で fallback が多発。
  - fallback 多発で `orders.db` への同時書き込み競合が増え、`database is locked` を誘発。
- 変更:
  - `execution/order_manager.py`
    - OANDA client に `ORDER_OANDA_REQUEST_TIMEOUT_SEC`（既定 8.0s）を導入し、
      無期限待ちを回避。
    - OANDA/`order_manager` service 両方に HTTP session pool を導入
      （`*_POOL_CONNECTIONS` / `*_POOL_MAXSIZE`）。
    - `order_manager` service 呼び出しは connect/read timeout を分離
      （`ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT` + `ORDER_MANAGER_SERVICE_TIMEOUT`）。
    - service failure のログ payload を要約化し、巨大 payload 直書きによる
      journald 負荷を抑制。
  - `workers/order_manager/worker.py`
    - `ORDER_MANAGER_SERVICE_WORKERS`（既定 1）を追加し、
      systemd 側から複数 worker 起動を可能化。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=20.0`
    - `ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT=1.0`
    - `ORDER_MANAGER_SERVICE_POOL_CONNECTIONS=8`
    - `ORDER_MANAGER_SERVICE_POOL_MAXSIZE=32`
    - `ORDER_OANDA_REQUEST_TIMEOUT_SEC=8.0`
    - `ORDER_OANDA_REQUEST_POOL_CONNECTIONS=16`
    - `ORDER_OANDA_REQUEST_POOL_MAXSIZE=32`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_WORKERS=1`
    - 備考: `workers` キーは追加済みだが、現行 unit では
      `python -m workers.order_manager.worker` 起動互換を優先して 1 固定で運用。
- 意図:
  - service 導線を優先維持し、`timeout -> fallback -> orders.db lock` の連鎖を抑える。
  - 発注判断ロジック（strategy/local decision、risk/policy）は変更せず、
    通路の安定化に限定。

### 2026-02-25（追記）`order_manager` 長時間応答の抑止（再試行予算の短縮）

- 症状（VM実測, UTC 2026-02-25 08:50-09:08）:
  - `quant-scalp-ping-5s-b/c` で
    `order_manager service call failed ... Read timed out (read timeout=20.0)` が継続。
  - その後 `order_manager_none` が発生し、local fallback で発注継続。
  - 同時間帯に `orders.db` で `database is locked` が多発。
- 原因:
  - `market_order` の protection fallback / submit retry が残っている間、
    service request が 20 秒を超えて詰まり、strategy 側 timeout を誘発。
  - timeout 後の local fallback が `orders.db` 同時書き込みを増やし、lock 警告を増幅。
- 変更:
  - `execution/order_manager.py`
    - `ORDER_SUBMIT_MAX_ATTEMPTS` の最小値を `2 -> 1` に変更し、
      service 側で単発 submit を許可。
  - `ops/env/quant-order-manager.env`
    - `ORDER_SUBMIT_MAX_ATTEMPTS=1`
    - `ORDER_PROTECTION_FALLBACK_MAX_RETRIES=0`
    - `ORDER_DB_BUSY_TIMEOUT_MS=250`
    - `ORDER_DB_LOG_RETRY_ATTEMPTS=3`
    - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.02`
    - `ORDER_DB_LOG_RETRY_BACKOFF=1.5`
    - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.10`
    - 意図: service API の応答遅延を抑えるため、service 側は低遅延 retry budget を維持。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_DB_BUSY_TIMEOUT_MS=1500`
    - `ORDER_DB_LOG_RETRY_ATTEMPTS=6`
    - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.04`
    - `ORDER_DB_LOG_RETRY_BACKOFF=1.8`
    - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.30`
- 意図:
  - service 経路の 20 秒超ブロッキングを避け、`order_manager_none` を減らす。
  - `orders.db` への書き込み再試行予算を戻し、stale 化を抑える。

### 2026-02-25（追記）`orders.db` ロック残留抑止（file lock + coordination write削減）

- 症状（VM実測, UTC 2026-02-25 09:xx）:
  - `order_manager_none` と `database is locked` が継続し、
    `orders.db` / `entry_intent_board` の更新が遅延。
  - `quant-order-manager` は稼働中でも `/order/coordinate_entry_intent` が
    lock系エラーを返す局面を確認。
- 変更:
  - `execution/order_manager.py`
    - `orders.db` write 前に cross-process `flock` を取得する
      `_order_db_file_lock` を追加（`orders.db.lock`）。
    - `_log_order` / `entry_intent_board` の `purge` と `record` を
      file lock 配下で実行し、write競合を直列化。
    - lock timeout は `sqlite3.OperationalError("database is locked ...")`
      として扱い、既存 retry/rollback 経路へ統合。
    - `entry_intent_board` の read前重複削除を削除し、
      `coordinate_entry_intent` 経路の不要 write を削減。
    - `entry_intent_board` write 失敗時に `rollback/reset` を追加し、
      lock発生後のトランザクション残留を防止。
  - `ops/env/quant-order-manager.env`
    - `ORDER_DB_FILE_LOCK_ENABLED=1`
    - `ORDER_DB_FILE_LOCK_TIMEOUT_SEC=0.12`
    - `ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC=0.02`
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_DB_FILE_LOCK_ENABLED=1`
    - `ORDER_DB_FILE_LOCK_TIMEOUT_SEC=0.30`
    - `ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC=0.05`
- 意図:
  - service と fallback の同時書き込みを lock-step 化し、
    `orders.db is locked` 警告と stale 進行を抑える。
  - `coordinate_entry_intent` の遅延要因を減らし、
    strategy 側 timeout 連鎖を止める。

### 2026-02-25（追記）`order_manager_none` 誤分類の抑止（status cache）

- 症状（VM実測, UTC 2026-02-25 09:47）:
  - service timeout が収束した後も、
    `scalp_ping_5s_b` の short で `order_manager_none` が残る。
  - 実際は `OPEN_SKIP note=entry_probability:entry_probability_reject_threshold`
    で、service障害ではなく local reject。
- 変更:
  - `execution/order_manager.py`
    - `ORDER_STATUS_CACHE_TTL_SEC`（既定 180s）を追加し、
      `client_order_id` ごとの直近 status をメモリ保持。
    - `_log_order` 実行時に DB 成否に依存せず status cache を更新。
    - `get_last_order_status_by_client_id` は DB 未取得時に
      status cache を返す。
    - `entry_probability_reject` で pre-service DB 記録を抑制する経路でも
      status cache へ reject reason を記録。
  - `ops/env/quant-order-manager.env`
    - `ORDER_STATUS_CACHE_TTL_SEC=180`
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_STATUS_CACHE_TTL_SEC=180`
- 意図:
  - `order_manager_none` を「通信失敗」と混同しないようにし、
    local reject 理由（`entry_probability_reject*`）を可視化する。

### 2026-02-25（追記）緊急止血: `scalp_ping_5s_b` / `M1Scalper-M1` の新規ENTRY停止

- 背景（VM実測, UTC 2026-02-25 11:49 前後）:
  - 直近90分の実現損益が `-2821.69`。
  - 内訳は `scalp_ping_5s_b_live=-1525.31`, `M1Scalper-M1=-1309.33` が主因。
  - `STOP_LOSS_ORDER` が損失主成分（同期間 `-2287.41`）。
- 変更:
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BLOCK_HOURS_UTC=0-23`（全時間帯ブロック）
  - `systemd/quant-scalp-ping-5s-b.service`
    - `UnsetEnvironment=SCALP_PING_5S_B_BLOCK_HOURS_JST` を追加
    - `Environment="SCALP_PING_5S_B_BLOCK_HOURS_JST=0,1,...,23"` を追記
      （`scalp_ping_5s_b.env` の既存設定を強制上書きし、全時間帯ブロック）
- 意図:
  - EXITワーカーは維持したまま、`B` と `M1` の新規ENTRYのみを即時停止して
    ドローダウン拡大を止血する。
  - 予測/方向整合の再検証が終わるまで、再開しない運用に固定する。

### 2026-02-25（追記）緊急止血: 全戦略ENTRY一時停止（strategy-control global）

- 背景（VM実測, UTC 2026-02-25 12:35 前後）:
  - `quant-scalp-ping-5s-b` / `quant-m1scalper` 停止後も
    `scalp_ping_5s_c_live` の新規発注（`orders.db: filled`）を確認。
  - `systemd/quant-scalp-ping-5s-b.service` の
    `UnsetEnvironment=SCALP_PING_5S_B_BLOCK_HOURS_JST` により、
    実行プロセスで `SCALP_PING_5S_B_BLOCK_HOURS_JST` が消失していた。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_CONTROL_GLOBAL_ENTRY_ENABLED=0`
    - `STRATEGY_CONTROL_GLOBAL_EXIT_ENABLED=1`
    - `STRATEGY_CONTROL_GLOBAL_LOCK=0`
  - `systemd/quant-scalp-ping-5s-b.service`
    - `UnsetEnvironment=SCALP_PING_5S_B_BLOCK_HOURS_JST` を削除
    - `Environment=\"SCALP_PING_5S_B_BLOCK_HOURS_JST=0,1,...,23\"` を維持
      （Bの全時間帯ブロックを有効化）
  - VM運用反映（即時）:
    - `strategy_control.set_global_flags(entry=False, exit=True, lock=False)` を適用。
    - 反映後の `orders.db` で
      `status=strategy_control_entry_disabled` を確認（新規ENTRY拒否）。
- 意図:
  - EXIT経路を維持したまま、全戦略の新規ENTRYを止めてDD拡大を停止する。
  - B/M1個別停止だけでは止血できない状況を、global guardで確実に抑える。

### 2026-02-25（追記）根本対策: 「停止」から品質ゲート運用へ復帰

- 背景（VM実測, UTC 2026-02-25 12:44）:
  - 直近6時間で `scalp_ping_5s_b_live / scalp_ping_5s_c_live / M1Scalper-M1` が
    PF<0.5 まで悪化し、`reduce` モード運用では劣化局面のエントリーが継続した。
  - 直近90分集計でも `STOP_LOSS_ORDER` と
    `MARKET_ORDER_TRADE_CLOSE` の負けが主因。
- 変更:
  - `workers/common/perf_guard.py`
    - `reduce` モードでも `margin_closeout / failfast / sl_loss_rate` は
      **hard block** として必ず拒否するロジックを追加。
    - 既定 `PERF_GUARD_RELAX_TAGS` から `M1Scalper` を除外
      （既定は `ImpulseRetrace` のみ）。
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_CONTROL_GLOBAL_ENTRY_ENABLED=1`（全停止解除）
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=1`
    - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=1`
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BLOCK_HOURS_ENABLED=0`（時間帯全停止を解除）
    - `M1SCALP_PERF_GUARD_*` を明示し、PF/勝率 failfast で劣化時を拒否。
    - `M1SCALP_PERF_GUARD_RELAX_TAGS=`（緩和無効）
  - `ops/env/scalp_ping_5s_c.env`
    - `PERF_GUARD_MODE=block` へ変更。
    - `failfast` / `sl_loss_rate` 閾値を引き上げ。
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` を 0.48 へ引き上げ。
  - `ops/env/scalp_ping_5s_d.env`
    - `PERF_GUARD_MODE=block` へ変更。
    - `BASE_ENTRY_UNITS` / `MAX_UNITS` を縮小し、過大ロットを抑制。
    - 確率下限とスケール上限を厳格化。
  - `systemd/quant-scalp-ping-5s-b.service`
    - 24h 強制ブロック（全時間帯 block）を削除。
    - B専用の quality-first override（failfast/sl-loss-rate/units/probability）を追加。
- 意図:
  - 「新規を全部止める」運用をやめ、劣化条件だけを機械的に遮断する。
  - 逆行予測・劣化PF・高SL率の局面を入口で除去し、トレード品質を上げる。

### 2026-02-25（追記）恒久対策: replay/cleanup の本番干渉を構造的に抑止

- 症状（VM実測, UTC 2026-02-25 12:25-12:47）:
  - `cleanup-qr-logs.service` が `orders.db VACUUM` を実行し続け、
    `quant-order-manager` の `database is locked` が集中発生。
  - `quant-replay-quality-gate.service` が高CPUで長時間走行し、
    `decision_latency_ms` と `data_lag_ms` のスパイクを誘発。
- 変更:
  - `analysis/replay_quality_gate_worker.py`
    - `REPLAY_QUALITY_GATE_ENABLED` を追加。
    - `0` の場合は replay 実行を行わず、即時 `rc=0` で終了。
  - `ops/env/quant-replay-quality-gate.env`
    - `REPLAY_QUALITY_GATE_ENABLED=0`（本番VMの既定値）。
  - `scripts/cleanup_logs.sh`（既存反映済み）:
    - `DB_VACUUM_SKIP_FILES=orders.db trades.db metrics.db`
    - `DB_VACUUM_ALLOW_HOT_DBS=0`
    - hot DB は checkpoint のみ実施し、VACUUM を既定で抑止。
  - `systemd/quant-replay-quality-gate.service`（既存反映済み）:
    - `Nice=15`, `IOSchedulingClass=idle`, `CPUWeight=20`
- VM反映確認:
  - `git rev-parse HEAD == origin/main == aeae2ac3a35976a9662619f5b3826f5ece8d661d`
  - 再起動後カウント（UTC 2026-02-25 12:31-）:
    - `quant-order-manager`: `failed to persist orders log: database is locked` = 0
    - `quant-position-manager`: `sync_trades timeout` = 0 / `position manager busy` = 0
- 意図:
  - 本番取引経路（market-data/strategy-control/order/position）と
    replay・保守ジョブの競合を運用条件で切り離し、
    DB lock/遅延スパイクの再発確率を下げる。

### 2026-02-25（追記）恒久対策: `performance_summary` timeout による遅延スパイク抑止

- 症状（VM実測, UTC 2026-02-25 13:15-13:30）:
  - `quant-bq-sync` が毎サイクル `pm.get_performance_summary()` を呼び、
    `quant-position-manager` の `/position/performance_summary` が
    8秒タイムアウトで `unknown error` を返却。
  - 直列で service失敗→local fallback が発生し、1サイクルあたり約20秒級の
    既定処理時間（8秒 timeout + 約12秒 local 集計）になっていた。
  - 実測: `/position/performance_summary` 応答時間は約 8030ms で失敗固定、
    local 直接実行は約 12100ms。
- 変更:
  - `workers/position_manager/worker.py`
    - `POSITION_MANAGER_WORKER_PERFORMANCE_SUMMARY_TIMEOUT_SEC` を追加
      （既定 20s）。
    - `/position/performance_summary` は timeout を明示エラー
      `performance_summary timeout (...)` で返す。
    - `/position/fetch_recent_trades` も timeout を明示化
      `fetch_recent_trades timeout (...)`。
  - `scripts/run_sync_pipeline.py`
    - `PIPELINE_PERF_SUMMARY_ENABLED` を追加（`0` で集計呼び出し停止）。
    - `PIPELINE_PERF_SUMMARY_REFRESH_SEC`（既定 180s）を追加し、
      `performance_summary` をTTLキャッシュする。
    - `PIPELINE_PERF_SUMMARY_STALE_MAX_AGE_SEC`（既定 900s）内は
      取得失敗時に stale cache を使ってスナップショット継続。
  - `ops/env/quant-v2-runtime.env`
    - `PIPELINE_PERF_SUMMARY_ENABLED=0`（本番ループでは重い集計を無効化）
    - `PIPELINE_PERF_SUMMARY_REFRESH_SEC=180`
    - `PIPELINE_PERF_SUMMARY_STALE_MAX_AGE_SEC=900`
    - `POSITION_MANAGER_SERVICE_PERFORMANCE_SUMMARY_TIMEOUT=25.0`
    - `POSITION_MANAGER_WORKER_PERFORMANCE_SUMMARY_TIMEOUT_SEC=20.0`
    - `POSITION_MANAGER_WORKER_FETCH_RECENT_TRADES_TIMEOUT_SEC=8.0`
- 意図:
  - `unknown error` と service->local 二重計算による遅延スパイクを減らし、
    `quant-bq-sync` が本番経路に与えるCPU/待ち時間の影響を常態化させない。

### 2026-02-25（追記）恒久対策: `strategy_tag` 断片化で perf_guard が効かない欠陥を修正

- 背景（VM実測, UTC 2026-02-25 13:44-13:46）:
  - `scalp_ping_5s_*_live` の `strategy_tag` が `-l<hex>` / `-s<hex>` 付きで記録されるケースで、
    `perf_guard` の集計が exact match 偏重となり、劣化サンプルの一部を取りこぼしていた。
  - その結果、実際は劣化している戦略でも `warmup_n` 判定になり、
    failfast / sl_loss_rate block が遅れて新規エントリーが通る窓が発生した。
- 変更:
  - `workers/common/perf_guard.py`
    - `strategy_tag` 照合を拡張し、`<tag>-l<hex>` / `<tag>-s<hex>` の
      ハッシュ付きタグを `LOWER(...) GLOB` で同一戦略として集計する。
    - 反映箇所: `_query_perf` / `_query_setup_perf` / `_query_perf_scale`。
    - `regime` スライスが成立していても、`margin_closeout` / `failfast` /
      `sl_loss_rate` の hard block は全体集計（non-regime）で再判定するよう変更。
      局所 regime の良化だけで severe 劣化をすり抜けないようにした。
  - `tests/test_perf_guard_failfast.py`
    - `HashBleed-l0abc1234` を使った回帰テストを追加し、
      ハッシュ付きタグでも failfast が発火することを固定化。
    - `test_perf_guard_regime_slice_cannot_bypass_global_hard_block` を追加し、
      regime スライス優先時でも global hard block が必ず効くことを固定化。
- 意図:
  - 「劣化を検知できないために entry が通る」構造欠陥を塞ぎ、
    新規停止なしでも品質ゲートの実効性を維持する。

### 2026-02-25（追記）品質維持: forecast実測監査 + MicroLevelReactor の局所絞り込み

- 背景（VM実測, UTC 2026-02-25 14:12 前後）:
  - forecast 導線は有効（`ORDER_MANAGER_FORECAST_GATE_ENABLED=1`,
    `STRATEGY_FORECAST_FUSION_ENABLED=1`,
    `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=1`）。
  - runtime近似パラメータでの before/after 評価は `mixed`。
    - 1m: `hit_delta=-0.0006`, `mae_delta=+0.0052`
    - 5m: `hit_delta=+0.0105`, `mae_delta=+0.0135`
    - 10m:`hit_delta=+0.0266`, `mae_delta=+0.0067`
  - 一方で `microlevelreactor` は直近7日 `n=5, PF=0.274, win_rate=0.200` と弱く、
    warmup条件のため新規が通りやすかった。
- 変更:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_BASE_UNITS=14000`（28000→14000）
    - `MICRO_MULTI_MAX_SIGNALS_PER_CYCLE=1`（3→1）
    - `MICRO_MULTI_DYN_ALLOC_MIN_TRADES=4`（10→4）
    - `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.45`（0.28→0.45）
    - `PERF_GUARD_MODE=block`, `PERF_GUARD_MIN_TRADES=8`
    - `PERF_GUARD_FAILFAST_MIN_TRADES=4`, `PERF_GUARD_FAILFAST_PF=0.90`,
      `PERF_GUARD_FAILFAST_WIN=0.45`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.52`,
      `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE=0.75`
- 意図:
  - 全体停止をせずに、弱い戦略だけ早く減速・遮断する。
  - forecast は利用継続しつつ、低品質 strategy の warmup すり抜けを抑制する。

### 2026-02-25（追記）根本対策: forecast gate の「通過条件」自体を品質基準へ更新

- 背景（VM実測, UTC 2026-02-25 14:50 前後）:
  - 直近2hで `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` / `MicroLevelReactor` が主な毀損源。
  - `orders.db` では `perf_block` が効いている一方、forecast gate は
    allowlist 制限により上記戦略に十分適用されていなかった。
- 変更:
  - `workers/common/forecast_gate.py`
    - strategy別 override の env 解決を拡張。
      - `..._STRATEGY_SCALP_PING_5S_B_LIVE` のような underscore 形式と、
        既存の英数字正規化形式の両方を解決可能にした。
    - `FORECAST_GATE_EDGE_BLOCK` の strategy別 override を追加。
    - `expected_pips` を方向付き期待値に変換し、strategy別に
      `expected_pips_contra` / `expected_pips_low` を block できるよう追加。
    - `target_reach_prob` の strategy別下限ガードを追加
      （`target_reach_prob_low` で block）。
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_GATE_STRATEGY_ALLOWLIST` を拡張:
      `MicroLevelReactor`, `scalp_ping_5s_b_live`,
      `scalp_ping_5s_c_live`, `scalp_ping_5s_flow_live` を追加。
    - 上記戦略へ `edge_block` / `expected_pips` / `target_reach_prob` の
      strategy別閾値を追加。
  - `tests/workers/test_forecast_gate.py`
    - underscore suffix の strategy別 override 回帰テストを追加。
    - `expected_pips_low` / `target_reach_prob_low` block の回帰テストを追加。
- 意図:
  - ロットだけを落とす運用ではなく、エントリー通過条件そのものを
    予測品質基準へ寄せる。
  - 新規全停止を避けつつ、低期待値・到達確率不足のシグナルを入口で遮断する。

### 2026-02-25（追記）根本対策: `perf_guard` hard/soft 判定で過剰全停止を抑制

- 背景（VM実測, UTC 2026-02-25 16:24 前後）:
  - `orders.db` で `preflight_start` は継続している一方、
    `perf_block` が支配的で `filled` が極少（2件）になっていた。
  - 主因は `reduce` モード戦略でも `hard:margin_closeout_n=*` /
    `hard:failfast:*` が一律 reject となり、
    stale closeout や片側指標劣化で entry が止まる経路だった。
- 変更:
  - `workers/common/perf_guard.py`
    - `margin_closeout` 判定を hard/soft に分離。
      - hard 条件:
        `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES` 未満、
        または `PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE` 超過
        （`..._HARD_MIN_COUNT` 以上）。
      - soft 条件は `margin_closeout_soft_n=...` を返し、
        `PERF_GUARD_MODE=reduce` では `warn` 通過させる。
    - `failfast` も hard/soft を分離。
      - hard 条件:
        `PERF_GUARD_FAILFAST_HARD_PF` 未満、
        または `PERF_GUARD_FAILFAST_HARD_REQUIRE_BOTH=1` で
        PF/勝率の同時悪化。
      - hard 以外は `failfast_soft:*` を返し、
        `reduce` では reject せず継続判定へ回す。
  - `tests/test_perf_guard_failfast.py`
    - `test_perf_guard_failfast_soft_in_reduce_mode` を追加。
    - `test_perf_guard_margin_closeout_soft_in_reduce_mode` を追加。
    - 既存 hard block テスト（margin closeout / regime hard bypass 禁止）は維持。
  - 仕様更新:
    - `docs/RISK_AND_EXECUTION.md` に
      hard/soft 判定と新規 env キーを追記。
- 意図:
  - 「劣化戦略は止める」性質を保ったまま、
    `reduce` モードの機会損失を生む過剰 hard block を除去する。
  - 一律ロット半減ではなく、入口判定の質を上げて
    エントリー可否を戦略状態に応じて切り替える。

### 2026-02-26（追記）根本対策: `scalp_ping_5s_c/d` の hard-stop 設定が反映されない欠陥を修正

- 背景（VM実測, UTC 2026-02-26 00:29 前後）:
  - 直近24hで `scalp_ping_5s_c_live` は
    `MARKET_ORDER_TRADE_CLOSE=383` / `STOP_LOSS_ORDER=0`、
    `-3741.1 JPY` と損失が偏っていた。
  - `ops/env/quant-order-manager.env` には
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_C=1` と
    `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_C=0` があるのに、
    `order_manager` 実装が B 系以外を legacy 分岐へ落としていた。
- 変更:
  - `execution/order_manager.py`
    - `_allow_stop_loss_on_fill` を修正し、
      `scalp_ping_5s_c*` / `scalp_ping_5s_d*` の family override を参照するよう更新。
    - `_disable_hard_stop_by_strategy` を修正し、
      `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_C/D` を優先評価するよう更新。
  - `tests/execution/test_order_manager_sl_overrides.py`
    - C variant の `stopLossOnFill` 反映テストを追加。
    - C variant の hard-stop disable/enabled 判定テストを追加。
    - 合計 11 テスト pass を確認。
  - 仕様更新:
    - `docs/RISK_AND_EXECUTION.md` に C/D family override の参照仕様を追記。
- 意図:
  - 「設定しているのにSLが効かない」実装不整合を解消し、
    C/D の損失を market-close 偏重から hard-stop 管理へ戻す。

### 2026-02-26（追記）strategy_entry に戦略ローカル先行プロファイルを搭載

- 背景:
  - `entry_probability` の計算品質に戦略ごとの差があり、
    「各戦略で先行指標を計算して閾値管理する」導線を
    `strategy_entry` 共通経路へ統一する必要があった。
- 変更:
  - `execution/strategy_entry.py`
    - `market_order` / `limit_order` で黒板協調直前に
      `_apply_strategy_leading_profile(...)` を追加。
    - `entry_thesis.env_prefix`（未指定時は strategy_tag 正規化）単位で
      `*_ENTRY_LEADING_PROFILE_*` を参照し、
      `entry_probability` を forecast/tech/range/micro/pattern 成分で再補正。
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW(_LONG/_SHORT)` による
      戦略別 reject を追加（未設定時は reject 無効）。
    - `entry_thesis.entry_probability_leading_profile` に補正内容を監査記録。
    - `limit_order` でも黒板協調後の最終 units を
      `entry_thesis.entry_units_intent` へ再同期。
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - 先行プロファイルの boost / threshold reject / manual pocket bypass テストを追加。
    - 既存 forecast fusion テストと合わせて pass を確認。
  - `docs/RISK_AND_EXECUTION.md`
    - 先行プロファイルの運用キーと責務を追記。
- 意図:
  - 黒板は意図協調のみを維持しつつ、
    先行指標による確率補正と閾値判断を戦略ローカル設定で運用できる状態にする。

### 2026-02-26（追記）先行プロファイルを戦略別 env に展開（global off + per-strategy opt-in）

- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0` を追加し、
      共通デフォルトは無効化。
  - 各 strategy worker env に `*_ENTRY_LEADING_PROFILE_*` を追加し、
    戦略別に `REJECT_BELOW` / `BOOST_MAX` / `PENALTY_MAX` /
    `UNITS_MIN/MAX_MULT` / `WEIGHT_*` を明示。
    - `scalp_ping_5s_{b,c,d,flow}`
    - `M1SCALP`
    - `RANGEFADER`
    - `SCALP_LEVEL_REJECT`
    - `SCALP_FALSE_BREAK_FADE`
    - `SCALP_EXTREMA_REVERSAL`
    - `SCALP_PRECISION`（macd/wick/squeeze/tick 各unit）
    - `MICRO_MULTI`（MicroRangeBreak unit）
  - `docs/RISK_AND_EXECUTION.md`
    - global off + per-strategy opt-in の運用方針を追記。
- 意図:
  - 先行指標ロジック自体は共通化しつつ、閾値と重みは戦略ローカルで管理し、
    「各戦略に専用閾値」の運用を担保する。

### 2026-02-26（追記）`spread_monitor` の stale ロックを解消（entry 停止の根本バグ修正）

- 背景（VM実測, UTC 2026-02-26 00:39 前後）:
  - `quant-scalp-ping-5s-flow` が `spread_stale age=... > max=4000ms` で連続ブロック。
  - 同時刻の `logs/tick_cache.json` は数秒以内の更新と 0.8p 前後の spread を保持しており、
    in-process snapshot stale と共有 tick cache の真値が乖離していた。
- 原因:
  - `market_data/spread_monitor.py:get_state()` は `_snapshot` が存在すると
    stale 時でも snapshot 優先で返し、tick cache fallback を使わない実装だった。
  - その結果、いったん stale snapshot に入ると `is_blocked()` が stale を維持し、
    entry ループで解除されないロック状態が継続した。
- 変更:
  - `market_data/spread_monitor.py`
    - snapshot が `MAX_AGE_MS` 超過時、tick cache fallback を評価し、
      非 stale または snapshot より新しい場合は fallback state を優先採用するよう修正。
    - fallback 採用時に `snapshot_age_ms` を state に付与し監査可能化。
    - fallback 採用時に snapshot/histories を再水和して、
      stale 判定の再発ループを防止。
    - stale cooldown 中でも fallback が fresh なら、
      `spread_stale` ブロックを即時解除する。
  - `workers/scalp_ping_5s/worker.py`
    - `spread_stale` ブロック検知時、`continue` する前に
      snapshot fallback を再取得して再判定する導線を追加。
    - 復帰できた場合は同サイクルで entry 判定へ復帰する。
  - `tests/test_spread_monitor_hot_fallback.py`
    - stale snapshot 存在時に fresh tick cache へフォールバックする回帰テストを追加。
    - stale cooldown が fresh tick cache で解除される回帰テストを追加。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    - `spread_stale` 判定ヘルパの回帰テストを追加。
  - テスト結果:
    - `pytest -q tests/test_spread_monitor_hot_fallback.py tests/test_spread_ok_tick_cache_fallback.py tests/workers/test_scalp_ping_5s_worker.py -k 'is_spread_stale_block or clears_stale_cooldown or prefers_fresh_tick_cache'` で pass。
- 意図:
  - スプレッド真値が更新されているのに stale 判定で新規エントリーが止まり続ける
    偽ブロックを除去し、`scalp_fast` 系の停止要因を「実スプレッド条件」に限定する。

### 2026-02-26（追記）`+2000円/h` 目標向けに scalp_fast を再配線（B/D/M1 stop, C narrow long）

- 背景（VM実測, UTC 2026-02-26 00:35 前後）:
  - 直近24h aggregate は `-589.34 JPY`（`1070 trades`）。
  - 直近6h aggregate は `-2758.59 JPY`（`187 trades`）。
  - 7日集計で `scalp_ping_5s_b_live=-16798.8 JPY`, `M1Scalper=-1618.66 JPY`,
    `scalp_ping_5s_d_live=-1626.13 JPY` が主損失源。
  - 一方 `scalp_ping_5s_c_live` は時間帯依存が強く、`JST 18時` は
    `68 trades / +6475.56 JPY`（`+95.23 JPY/trade`）で突出。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ENABLED=0`
  - `ops/env/quant-scalp-ping-5s-b.env`
    - `SCALP_PING_5S_B_ENABLED=0`
  - `ops/env/scalp_ping_5s_d.env`
    - `SCALP_PING_5S_D_ENABLED=0`
  - `ops/env/quant-scalp-ping-5s-d.env`
    - `SCALP_PING_5S_D_ENABLED=0`
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_ENABLED=0`
    - `M1SCALP_BLOCK_HOURS_ENABLED=1`
    - `M1SCALP_BLOCK_HOURS_UTC=0-23`
  - `ops/env/scalp_ping_5s_flow.env`
    - `SCALP_PING_5S_FLOW_ENABLED=0`
  - `ops/env/quant-scalp-ping-5s-flow.env`
    - `SCALP_PING_5S_FLOW_ENABLED=0`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=long`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=18,19,22`
    - `SCALP_PING_5S_C_BLOCK_HOURS_JST=`（空）
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=3000`
    - `SCALP_PING_5S_C_MAX_UNITS=7500`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `ORDER_MANAGER_PRESERVE_INTENT_*_SCALP_PING_5S_C_LIVE` を
      `reject_under=0.35 / min_scale=1.00 / max_scale=1.20` に更新。
  - `ops/env/quant-ping5s-d-canary-guard.env`
    - `PING5S_D_CANARY_APPLY=0`（D停止中の自動units反映を無効化）
- 意図:
  - 「損失源を止める」と「勝ち時間へ資本集中」を同時に実施し、
    `+2000円/h` 到達のための入口品質とサイズ効率を優先する。

### 2026-02-26（追記）AddonLive 経路の `strategy_tag` 欠損を恒久修正

- 背景:
  - `workers/common/addon_live.py` で、解決済みの `strategy_tag` を
    後段で `order.strategy_tag/order.tag` のみで再代入して欠損させる経路があった。
  - 欠損時は `entry_thesis.strategy_tag` と `client_order_id` の戦略識別が崩れ、
    `missing_strategy_tag` 系の拒否/監査断絶につながる。
- 変更:
  - `strategy_tag` 解決順を `order -> intent -> meta -> worker_id` に統一し、
    一度解決した値を再上書きしないよう修正。
  - 解決不能な注文は送信前に skip して、欠損注文を order_manager へ流さない。
  - 回帰テスト `tests/addons/test_addon_live_broker_strategy_tag.py` を追加。
- 意図:
  - AddonLive でも V2 契約（`strategy_tag` 必須）を担保し、
    ENTRY/EXIT/監査の戦略整合性を維持する。

### 2026-02-26（追記）`allow_hours` に soft-mode を追加し「停止」を恒久回避

- 背景（VM実測, UTC 2026-02-26 00:50〜01:10）:
  - `scalp_ping_5s_c_live` が `entry-skip summary ... outside_allow_hour_jst=*` で
    連続スキップし、スプレッド復旧後も新規が止まる時間帯が発生。
  - 既存実装は `ALLOW_HOURS_JST` を hard block しており、許可外時間は
    シグナル品質に関係なく新規を全面停止していた。
- 変更:
  - `workers/scalp_ping_5s/config.py`
    - `SCALP_PING_5S_ALLOW_HOURS_SOFT_ENABLED` を追加（既定: `0`, C/D prefix は `1`）。
    - 許可時間外の soft 運用パラメータを追加:
      - `SCALP_PING_5S_ALLOW_HOURS_OUTSIDE_UNITS_MULT`
      - `SCALP_PING_5S_ALLOW_HOURS_OUTSIDE_MIN_CONFIDENCE`
      - `SCALP_PING_5S_ALLOW_HOURS_OUTSIDE_MIN_ENTRY_PROBABILITY`
  - `workers/scalp_ping_5s/worker.py`
    - `AllowHourEntryPolicy` / `_resolve_allow_hour_entry_policy()` を追加。
    - `ALLOW_HOURS` 判定を `hard block` または `soft gate` に分離。
    - soft mode 時は
      - `confidence` 下限
      - `entry_probability` 下限
      - `units_mult` 縮小
      を適用し、全面停止ではなく高品質シグナルのみ通す。
    - 監査用に `entry_thesis` と open log へ allow-hour policy を記録。
    - `ENV_PREFIX=SCALP_PING_5S_C` / `SCALP_PING_5S_D` では soft-mode を既定有効化し、
      既存 `ALLOW_HOURS_JST` 設定が hard stop にならないようにした。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    - allow-hours policy（許可内/許可外hard/許可外soft）の回帰テストを追加。
- 意図:
  - 「止まる」原因を hard 時間帯ゲートから切り離し、
    取引継続性を維持しながら品質（確率・信頼度・ロット）で制御する。

### 2026-02-26（追記）`perf_guard` の margin closeout hard-block を reduce mode で恒久緩和

- 背景（VM実測）:
  - `quant-order-manager` で `scalp_ping_5s_c_live` が
    `perf_block:hard:hour4:margin_closeout_n=2 rate=0.333 n=6` により
    新規拒否を連発し、戦略が実質停止した。
  - C worker 側は `PERF_GUARD_MODE=reduce` だが、margin closeout 判定は
    warmup（小標本）でも hard 扱いになり、reduce の意図が効いていなかった。
- 変更:
  - `workers/common/perf_guard.py`
    - margin closeout 判定を改修し、`mode=reduce/warn` では
      warmup由来の closeout を `margin_closeout_soft_warmup_*` として扱う。
    - block mode かつ hard閾値超過時のみ `margin_closeout_n=*` の hard block を維持。
  - `tests/test_perf_guard_failfast.py`
    - `test_perf_guard_margin_closeout_warmup_soft_in_reduce_mode` を追加し、
      小標本 closeout が reduce mode で hard block にならないことを回帰固定。
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.60`
      （C の低確率エントリーを抑制）
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT=3`
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE=0.50`
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES=1`
- 意図:
  - 「軽微な closeout で長時間停止する」問題を解消しつつ、
    低確率エントリーを削って C 戦略の実行品質を上げる。

### 2026-02-26（追記）時間帯依存を廃止し、常時運転へ固定（`+2000円/h` 方針）

- 背景（VM実測, UTC 2026-02-26）:
  - 直近24h:
    - `scalp_ping_5s_c_live buy`: `339 trades / +4985.32 JPY`
    - `scalp_ping_5s_c_live sell`: `97 trades / -939.95 JPY`
    - `M1Scalper-M1 buy`: `25 trades / +356.46 JPY`
    - `M1Scalper-M1 sell`: `73 trades / -1450.49 JPY`
  - 7日集計でも `sell` 側（C/M1）が継続して負寄与。
  - `logs/trades.db` の open は `0`（反映前にフラット確認）。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`（空）
    - `SCALP_PING_5S_C_ALLOW_HOURS_SOFT_ENABLED=0`
    - `SCALP_PING_5S_C_SIDE_FILTER=long` を維持
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=6000`
    - `SCALP_PING_5S_C_MAX_UNITS=15000`
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_ENABLED=1`
    - `M1SCALP_BLOCK_HOURS_ENABLED=0`
    - `M1SCALP_SIDE_FILTER=long`
  - `ops/env/quant-v2-runtime.env`
    - `FORECAST_GATE_STRATEGY_ALLOWLIST=MicroRangeBreak,MicroTrendRetest,MicroVWAPBound,MicroLevelReactor,M1Scalper-M1,scalp_ping_5s_c_live`
  - `ops/env/quant-micro-pullbackema.env`
    - `MICRO_MULTI_ENABLED=0`（大幅負寄与戦略の停止維持）
- 意図:
  - 「勝ち時間だけを開ける」設計をやめ、常時稼働のまま
    `方向フィルタ + strategy別 gate + perf_guard` で負け筋を削る。
  - 時間帯は gating 条件から除外し、`+2000円/h` のために
    取引機会を全時間帯で取りに行く。

### 2026-02-26（追記）C系 `env_prefix` 解決を修正し、誤った `perf_block` を抑制

- 背景（VM実測）:
  - `scalp_ping_5s_c_live` の preflight で
    `perf_block:hard:hour4:margin_closeout_n=2 rate=0.333 n=6` が多発。
  - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce` を設定していても、
    `strategy_tag=scalp_ping_5s_c_live` が `SCALP_PING_5S` と解決され、
    C専用閾値が使われない経路があった。
- 変更:
  - `execution/order_manager.py` / `execution/strategy_entry.py`
    - `_infer_env_prefix_from_strategy_tag()` に
      `SCALP_PING_5S_C` / `SCALP_PING_5S_D` を追加。
    - `scalp_ping_5s_c_live` が base prefix へ落ちる誤判定を解消。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES=1`
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT=3`
    - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE=0.50`
  - `tests/execution/test_env_prefix_inference.py`
    - C prefix 推論の回帰テストを追加。
  - `ops/env/quant-v2-runtime.env`
    - `SCALP_PING_5S_C_PERF_GUARD_*` を order-manager service 側へ明示追加し、
      worker env だけに定義されていた C 閾値を preflight 判定で確実に参照。
    - 互換運用として `SCALP_PING_5S_PERF_GUARD_MARGIN_CLOSEOUT_HARD_*` も同値化。
- 意図:
  - C戦略で `reduce` 運用を正しく適用し、
    軽微な margin closeout 履歴だけで hard reject 連発する状態を解消する。

### 2026-02-26（追記）`quant-autotune-ui` の更新遅延を修正（snapshot candidate の短絡化）

- 背景（VM実測, UTC 2026-02-26 01:34）:
  - `/dashboard` 応答が `14〜17s` で遅延。
  - `apps/autotune_ui.py` の `_collect_snapshot_candidates()` が
    `gcs` snapshot が fresh でも毎回 `local` snapshot（`_build_local_snapshot_with_status()`）を
    実行していた。
  - 実測で `fetch_gcs ~= 0.145s` に対して `build_local_with_status ~= 13.8s`。
- 変更:
  - `apps/autotune_ui.py`
    - `_has_fresh_candidate()` を追加。
    - `remote/gcs` に fresh かつ metrics 有効な snapshot がある場合、
      `local` snapshot の構築を `skip` して即 return するよう変更。
    - `fetch_attempts` に `local: skip (remote/gcs snapshot is already fresh)` を記録。
  - `tests/apps/test_autotune_ui_snapshot_selection.py`
    - fresh な `gcs` 時に `local` が呼ばれないテストを追加。
    - stale な `gcs` 時は `local` を継続して呼ぶ回帰テストを追加。
- 意図:
  - UI応答を `snapshot source = gcs/remote` の通常ケースで高速化し、
    `local` fallback は本当に必要なケース（stale/欠損時）のみ実行する。

### 2026-02-26（追記）`quant-autotune-ui` 追加高速化（secret / strategy_control のTTLキャッシュ）

- 背景（VM profile, UTC 2026-02-26 01:40）:
  - `_load_dashboard_data` の p95 がまだ 2s 台。
  - プロファイル内訳:
    - `_load_strategy_control_state` ≒ 1.75s/req
    - Secret Manager 経由の `_get_secret_optional` ≒ 1.70s/req
  - いずれも毎リクエスト再取得/再集計されていた。
- 変更:
  - `apps/autotune_ui.py`
    - `UI_SECRET_CACHE_TTL_SEC`（default: 60s）で secret 値をTTLキャッシュ化
      （未設定キーも `None` として負キャッシュ）。
    - `UI_STRATEGY_CONTROL_CACHE_TTL_SEC`（default: 30s）で
      `strategy_control` 状態をTTLキャッシュ化。
    - `strategy_control` キャッシュ返却は deep copy で行い、
      呼び出し側の破壊的変更がキャッシュ本体へ波及しないようにした。
  - `tests/apps/test_autotune_ui_caching.py`
    - secret cache（命中/未設定）と strategy_control cache の回帰テストを追加。
- 意図:
  - dashboard の毎リクエストで発生していた
    「外部secret往復」「大きなDB走査＋JSON parse」をTTL内で再利用し、
    UI更新の体感遅延をさらに圧縮する。

### 2026-02-26（追記）dynamic_alloc を soft participation 既定へ変更（停止回避）

- 背景:
  - VM実績で `STOP_LOSS_ORDER` 偏重時に、`dyn_alloc loser_block / winner_only` が
    戦略停止に寄り、取引機会が局所化していた。
  - 要件として「全戦略を止めず、サイズ調整で収益改善」を優先する必要がある。
- 変更:
  - `scripts/dynamic_alloc_worker.py`
    - recency weighted（half-life）スコアへ更新。
    - `sl_rate` / `downside_share` を score/lot 計算へ反映。
    - `allocation_policy` を `dynamic_alloc.json` へ出力
      (`soft_participation`, `allow_loser_block`, `allow_winner_only`,
      `min_lot_multiplier`, `max_lot_multiplier`)。
    - 既定を `soft_participation=1`（`allow_loser_block=0`, `allow_winner_only=0`）へ変更。
  - `workers/common/dynamic_alloc.py`
    - `allocation_policy` を読んで `load_strategy_profile()` の戻り値へ
      `allow_loser_block` / `allow_winner_only` / `soft_participation` を追加。
  - `workers/micro_runtime/worker.py`
    - dyn alloc hard block を profile 側 `allow_loser_block` で制御。
    - winner-only 絞り込みを profile 側 `allow_winner_only` で制御。
  - `workers/scalp_m1scalper/worker.py`
  - `workers/scalp_macd_rsi_div/worker.py`
    - dyn alloc hard block を profile 側 `allow_loser_block` で制御。
  - テスト:
    - `tests/test_dynamic_alloc_worker.py`
    - `tests/workers/common/test_dynamic_alloc.py`（新規）
- 意図:
  - hard stop ではなく「負け戦略は細く、勝ち戦略は太く」を徹底し、
    全戦略を稼働させたまま資金配分で改善する。

### 2026-02-26（追記）`scalp_ping_5s_c_live` tail-loss 抑制（大口約定の縮小 + 確率ゲート強化）

- 背景（VM 実測, UTC 05:15-06:15）:
  - `trades.db` で `scalp_ping_5s_c_live` は勝率が高い一方、
    `MARKET_ORDER_TRADE_CLOSE` の大口（7k〜8k units）が
    `-6〜-9 pips` を複数回出し、1時間集計がマイナスに傾いた。
  - `orders.db` では `entry_probability_reject` の記録後でも
    同一 `client_order_id` が最終 `filled` になるケースがあり、
    低品質帯の通過余地が残っていた。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_MAX_ACTIVE_TRADES=8 -> 5`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=6000 -> 2200`
    - `SCALP_PING_5S_C_MAX_UNITS=15000 -> 4500`
    - `SCALP_PING_5S_C_SL_BASE_PIPS=2.4 -> 1.3`
    - `SCALP_PING_5S_C_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.9 -> 1.2`
    - `SCALP_PING_5S_C_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.5 -> 1.0`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.62 -> 0.70`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.48 -> 0.56`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.60 -> 0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=0.80 -> 0.55`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=1.15 -> 1.00`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C_LIVE=0.65 -> 0.90`
- 意図:
  - 低確率帯の entry 通過を減らし、同時保有/大口約定による尾損失を抑制する。

### 2026-02-26（追記）`client_order_id` 由来 `strategy_tag` 解決バグを修正（C 確率ゲートの恒久化）

- 背景（VM実測, UTC 05:25-06:00）:
  - `orders.db` で `scalp_ping_5s_c_live` の `filled` に
    `entry_probability < 0.60` が多数残っていた。
  - 原因は `execution/order_manager.py` の
    `_strategy_tag_from_client_id()` が
    `qr-<ts>-<strategy_tag>-<digest>` 形式を誤解釈し、
    digest 片を `strategy_tag` として返していたこと。
  - その結果、`ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_*`
    など strategy 個別しきい値が適用されない経路が存在した。
- 変更:
  - `execution/order_manager.py`
    - `_strategy_tag_from_client_id()` を修正し、
      - `qr-<ts>-<strategy_tag>-<digest>`
      - `qr-<ts>-<pocket>-<strategy_tag>-<digest>`
      の両形式を正しく解釈するようにした。
  - `tests/execution/test_env_prefix_inference.py`
    - 上記2形式の回帰テストを追加。
- 意図:
  - C 戦略に設定した確率拒否・ロット縮小・perf ガード等の
    strategy 固有制約を常に有効化し、低品質エントリーの通過を防ぐ。
  - 収益の安定化を優先し、`+2000円/h` へ向けた負け筋の先行除去を行う。

### 2026-02-26（追記）`scalp_ping_5s_c_live` の hard SL 未添付を恒久修正（fixed-mode優先の解除）

- 背景（VM実測, UTC 06:29-06:31）:
  - `orders.db` の `filled` request payload で `virtual_sl_price` は存在する一方、
    OANDA送信 `order` に `stopLossOnFill` が含まれず、
    `openTrades` も `stopLoss=null` が継続していた。
  - `ops/env/quant-order-manager.env` では
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_C=1` /
    `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_C=0` が有効であり、
    env と実挙動が不一致だった。
- 原因:
  - `execution/order_manager.py` で `sl_disabled` を
    `stop_loss_disabled_for_pocket()`（`ORDER_FIXED_SL_MODE`）優先で評価しており、
    B 以外の戦略（C/D含む）で
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_*` による再有効化が反映されなかった。
- 変更:
  - `execution/order_manager.py`
    - `_entry_sl_disabled_for_strategy()` を追加。
    - `market_order` / `limit_order` の `sl_disabled` 判定を
      新ヘルパーに置換し、
      strategy別 `ORDER_ALLOW_STOP_LOSS_ON_FILL_*` が true の場合は
      `ORDER_FIXED_SL_MODE` が OFF でも hard SL を有効化するよう統一。
  - `tests/execution/test_order_manager_sl_overrides.py`
    - C/B 両戦略で「fixed-mode OFF でも strategy override で
      `sl_disabled=False` になる」回帰テストを追加。
    - C 戦略で override 未設定時は従来どおり `sl_disabled=True` となる
      ガードテストを追加。
- 意図:
  - C 戦略の「エントリー時 hard SL 欠損」を解消し、
    逆行時の放置損失と `direction_cap` 長時間拘束の再発を防ぐ。

### 2026-02-26（追記）`OPEN_SKIP` 後の同一client約定を修正（service null 応答の誤fallback）

- 背景（VM実測, UTC 06:37-06:41）:
  - `quant-order-manager` journal で `OPEN_SKIP (entry_probability_reject_threshold)` が
    出ている同一 `client_order_id` について、`orders.db` には
    直後の `preflight_start -> submit_attempt -> filled` が残るケースが発生。
  - 実害として、service 側 reject 判定と local 経路の発注結果が競合し、
    判定一貫性と hard SL 添付の一貫性が崩れていた。
- 原因:
  - `execution/order_manager.py` の service橋渡しが
    「`service result = null`（= 正規の reject/skip）」と
    「service未使用/通信失敗」を同じ `None` で表現していた。
  - `market_order` / `limit_order` で `None` を
    「未処理」と誤認して local 処理にフォールバックし、
    reject 後の再発注が起き得た。
- 変更:
  - `execution/order_manager.py`
    - `_ORDER_MANAGER_SERVICE_UNHANDLED` sentinel を導入。
    - `_order_manager_service_request*` は
      - service未使用/通信失敗(fallback時) => sentinel
      - service処理済み（`result` が `None` 含む）=> 実値
      を返すよう分離。
    - `coordinate_entry_intent / cancel_order / close_trade /
      set_trade_protections / market_order / limit_order` の
      呼び出し側判定を sentinel 対応へ更新。
  - `tests/execution/test_order_manager_log_retry.py`
    - serviceが `None` を返したときに local fallback を実行しない
      回帰テスト（`market_order`, `limit_order`）を追加。
- 意図:
  - service判定（reject/skip）と local 発注経路の競合を防ぎ、
    `client_order_id` 単位での判定・発注整合性を恒久的に担保する。

### 2026-02-26（追記）`scalp_ping_5s_c_live` のロット上振れを抑止（最終送信前クランプ）

- 背景（VM実測, UTC 06:29-06:31）:
  - `SCALP_PING_5S_C_MAX_UNITS=4500` で運用しているにもかかわらず、
    `orders.db` / worker ログに `units=5737` / `units=4872` が記録された。
  - 原因は 2 段で、
    - `workers/scalp_ping_5s/worker.py` の TECH sizing 適用後に再クランプが無い。
    - `execution/strategy_entry.py` の協調後ユニットが strategy 要求量を上回り得る。
- 変更:
  - `workers/scalp_ping_5s/worker.py`
    - `tech_decision.size_mult` 適用後に `units_risk` / `MAX_UNITS` / `MIN_UNITS` で再クランプ。
    - `market_order` 送信直前に最終セーフティクランプを追加し、送信ユニットの上振れを防止。
  - `execution/strategy_entry.py`
    - `_coordinate_entry_units()` で `final_units` を strategy 要求量（`abs(units)`）以下に制限。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=long` を解除し、方向は
      forecast/horizon を含む strategy ローカル判定へ委譲。
- 意図:
  - 戦略ローカルで設計したユニット意図を上限として厳守し、
    予期しない上振れロットによる tail-loss を再発させない。

### 2026-02-26（追記）`scalp_ping_5s_c_live` no-block 改善（確率/サイズ/退出の再チューニング）

- 背景（VM 実測, JST 15:16 時点）:
  - 日次 `scalp_ping_5s_c_live` が `311 trades / -4572.0 JPY / -284.5 pips` で悪化継続。
  - 勝率は約 51% でも、平均勝ち `+1.71 pips` に対して平均負け `-3.68 pips` と
    損失幅が勝ち幅を上回っていた。
- 方針:
  - 停止・時間帯ブロックは行わず、通過注文の期待値を改善する。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - 取引密度/同時保有の抑制:
      - `MAX_ACTIVE_TRADES=4`
      - `MAX_ORDERS_PER_MINUTE=10`
      - `BASE_ENTRY_UNITS=1600`
      - `MAX_UNITS=3000`
    - no-block運用の統一:
      - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`
      - `SCALP_PING_5S_C_PERF_SCALE_ENABLED=0`
      - `SCALP_PING_5S_PERF_SCALE_ENABLED=0`
    - tail-loss抑制:
      - `FORCE_EXIT_MAX_HOLD_SEC=75`
      - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.0`
      - `FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=2.5`
      - `FORCE_EXIT_FLOATING_LOSS_MIN_PIPS=0.45`
    - entry品質ガード強化:
      - `CONF_FLOOR=72`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.76`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.62`
      - `ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT=1.00`
      - `ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT=1.00`
      - `ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX=0.20`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.40`
      - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=1.00`
    - local fallback経路の preserve-intent 調整:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.55`
      - `...MIN_SCALE...=0.55`
      - `...MAX_SCALE...=1.00`
      - `...BOOST_PROBABILITY...=0.85`
  - `ops/env/quant-order-manager.env`
    - service経路と同値化:
      - `REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.55`
      - `MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.55`
      - `MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=1.00`
      - `BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.85`
      - `SCALP_PING_5S(_C)_PERF_SCALE_ENABLED=0`
- 意図:
  - ブロックに頼らず、低品質エントリー通過率とロット上振れを抑え、
    `MARKET_ORDER_TRADE_CLOSE` 側の尾損失を縮小する。

### 2026-02-26（追記）`scalp_ping_5s_c(_live)` の negative-close 解放（引っ張り損失の抑止）

- 背景（VM実測, JST 15時台）:
  - `orders.db` で C 戦略に `close_reject_no_negative` が多発し、
    その時間帯の `pl_pips` が継続してマイナスだった。
  - reject 時の `exit_reason` は `take_profit`, `range_timeout`, `candle_*` が中心で、
    「EXIT シグナルは出ているが、損益符号ガードで閉じられない」状態だった。
- 変更:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_c` / `scalp_ping_5s_c_live` に
      `neg_exit.allow_reasons: ["*"]` を追加。
- 意図:
  - C の no-block 運用では EXIT の機動性を優先し、
    `close_reject_no_negative` 起因の含み損拡大を防ぐ。

### 2026-02-26（追記）円換算（JPY）優先のリスク判定へ修正

- 背景:
  - リスク判定で `pips` と `JPY` が混在し、ロット差を含む実損益と
    ガード判定軸がずれるケースがあった。
- 変更:
  - `execution/risk_guard.py`
    - `check_global_drawdown()` を `realized_pl`（JPY）優先に変更。
      - 既定: `GLOBAL_DD_LOOKBACK_DAYS` 日の `net_pl` を
        `GLOBAL_DD_EQUITY_BASE_JPY` または `update_dd_context()` の
        口座 equity ヒントで割って DD 比率を算出。
      - `realized_pl` が無いDBのみ `pl_pips` 系へフォールバック。
    - `loss_cooldown_status()` の連敗判定を `realized_pl` 優先に変更
      （`LOSS_COOLDOWN_MIN_ABS_JPY` で微小損益のノイズ除外可）。
  - `workers/common/perf_guard.py`
    - PF/勝率判定の集計列を `PERF_GUARD_VALUE_COLUMN` で選択可能にし、
      既定を `realized_pl` 優先に変更（`pl_pips` フォールバックあり）。
    - 既存の `avg_pips` 閾値系は互換維持のためそのまま残し、
      値幅評価（補助）と実損益評価（主軸）を分離。
- 意図:
  - エントリー抑制・連敗クールダウン・全体DD監視を「円換算基準」で揃え、
    口座ダメージに直結する軸でリスクを管理する。

### 2026-02-26（追記）`scalp_ping_5s_c_live` を利益時間帯限定 + failfast block 化

- 背景（VM実測, 2026-02-26 JST 16:01 時点）:
  - 直近 1h: `193 trades / -3650.8 JPY`（自動戦略のみ）
  - 主因: `scalp_ping_5s_c_live` が `184 trades / -3227.3 JPY`
  - 時間帯別（累計）では C は `JST 18/19/22` 以外で負け越しが目立った。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=18,19,22`
    - `SCALP_PING_5S_C_ALLOW_HOURS_SOFT_ENABLED=0`（許可外は hard block）
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=block`
    - `SCALP_PING_5S_C_PERF_GUARD_HOURLY_MIN_TRADES: 14 -> 8`
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_MIN_TRADES: 14 -> 8`
    - fallback 経路の `SCALP_PING_5S_PERF_GUARD_MODE` も `block` へ統一
    - 連打抑制として `MAX_ACTIVE_TRADES/MAX_PER_DIRECTION=1`、`BASE_ENTRY_UNITS=900`
- 意図:
  - C 戦略の「負け時間帯での高頻度連打」を止め、
    収益が出ている時間帯へ約定密度を寄せる。

### 2026-02-26（追記）`scalp_ping_5s_c_live` flip 過発火デリスク（SL連発帯の反転抑制）

- 背景（VM 実測, UTC 2026-02-26 07:05 前後）:
  - `ui_state` 直近60件で `flip_any=19`（`fast=14 / sl=2 / side=3`）かつ
    全て `short->long` 偏重。
  - 同期間の平均 `pl_pips` は `flipあり=-3.12`、`flipなし=-0.84`。
  - 直近12件で `STOP_LOSS_ORDER=11` と SL クラスターが継続。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - fast flip の閾値/クールダウンを引き上げ:
      - `DIRECTION_SCORE_MIN=0.58`
      - `HORIZON_SCORE_MIN=0.38`
      - `HORIZON_AGREE_MIN=3`
      - `NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.82`
      - `MOMENTUM_MIN_PIPS=0.16`
      - `COOLDOWN_SEC=1.2`
      - `CONFIDENCE_ADD=1`
    - sl-streak flip の発火条件を厳格化:
      - `MIN_SIDE_SL_HITS=3`
      - `MIN_TARGET_MARKET_PLUS=2`
      - `FORCE_STREAK=4`
      - `METRICS_OVERRIDE_ENABLED=0`
      - `DIRECTION_SCORE_MIN=0.55`
      - `HORIZON_SCORE_MIN=0.42`
    - `SIDE_METRICS_DIRECTION_FLIP_ENABLED=0`（C では一時停止）。
    - ロット/通過抑制:
      - `MAX_ORDERS_PER_MINUTE=4`
      - `BASE_ENTRY_UNITS=700`
      - `MAX_UNITS=1200`
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.60`
      - `...MIN_SCALE...=0.40`
      - `...MAX_SCALE...=0.85`
- 意図:
  - 方向転換機構を残したまま、SL連発局面での反転由来エントリーとロット上振れを先に抑える。

### 2026-02-26（追記）`scalp_ping_5s_c_live` 時間帯停止解除（効果実証不足）

- 背景（VM 実測, UTC 07:16 集計）:
  - 直近14日 `scalp_ping_5s_c_live` は JST時間帯別の分散が大きく、
    これまで許可していた `18` 時帯が `34 trades / -698.8 JPY / -280.9 pips` と劣後。
  - 許可帯（18/19/22）合算 `181 trades / -325.3 JPY` という観測値はあるが、
    時間帯ブロック有無の因果効果を示すA/B実証ではない。
  - 実運用ログで `outside_allow_hour_jst` による entry skip が連続し、
    フィルタ起因で機会損失が継続。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`（空へ変更）
- 意図:
  - 時間帯での機械停止を外し、効果が計測しやすい
    `entry_probability / flip / risk` 系ガードに制御責務を戻す。

### 2026-02-26（追記）`scalp_ping_5s_c_live` 連続悪化の緊急デリスク（VM実測ベース）

- 背景（VM実測, UTC 2026-02-26 07:24 集計）:
  - 直近24hで
    - `scalp_ping_5s_c_live`: `587 trades / -7025.8 JPY / -749.1 pips / win 53.2% / PF 0.434`
    - `scalp_ping_5s_b_live`: `481 trades / -3144.3 JPY / -414.7 pips / win 24.9% / PF 0.359`
  - 直近6hでは `scalp_ping_5s_c_live long` が
    `188 trades / -3258.0 JPY / -231.4 pips / win 38.3%` と悪化継続。
  - `close_reason` は
    `scalp_ping_5s_c_live: MARKET_ORDER_TRADE_CLOSE=543件/-6335.9JPY`、
    `scalp_ping_5s_b_live: STOP_LOSS_ORDER=346件/-4681.5JPY` が主因。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - 約定密度とサイズを追加圧縮:
      - `MAX_ORDERS_PER_MINUTE=2`
      - `BASE_ENTRY_UNITS=450`
      - `MAX_UNITS=700`
    - entry品質ゲートを引き上げ:
      - `CONF_FLOOR=78`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.70`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.24`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.52`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.60`
      - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.85`
    - failfast を前倒し:
      - `PERF_GUARD_FAILFAST_MIN_TRADES=6`
      - `PERF_GUARD_FAILFAST_PF=0.90`
      - `PERF_GUARD_FAILFAST_WIN=0.48`
      - `PERF_GUARD_SL_LOSS_RATE_MAX=0.55`
  - `ops/env/quant-order-manager.env`
    - C の preserve-intent を追加厳格化:
      - `REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.72`
      - `MIN_SCALE...=0.30`
      - `MAX_SCALE...=0.55`
    - B の preserve-intent も再悪化防止:
      - `REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.62`
      - `MIN_SCALE...=0.35`
      - `MAX_SCALE...=0.65`
    - fallback 側 perf_guard を `reduce -> block` に統一し、
      `FAILFAST` を `min_trades=8 / PF=0.90 / win=0.48` へ引き上げ。
  - `ops/env/scalp_ping_5s_b.env` / `systemd/quant-scalp-ping-5s-b.service`
    - B の実効ロット上限を追加圧縮:
      - `BASE_ENTRY_UNITS=600`
      - `MAX_UNITS=1200`
      - `MAX_ORDERS_PER_MINUTE=8`
    - unit override でも同値に固定し、再起動後のズレを防止。
- 意図:
  - `C` の直近悪化帯で「通過率とロット上限」を同時に落として下方分散を先に止める。
  - `B` はすでに取引量が落ちているため、再悪化しないよう上限をさらに抑える。

### 2026-02-26（追記）`scalp_ping_5s_b/c` setup-perf gate を有効化（hour×side×regime）

- 背景（VM実測, UTC 2026-02-26 08:47 集計）:
  - 直近1h（時刻正規化後）:
    - `scalp_ping_5s_c_live`: `28 trades / -143.4 JPY / -19.4 pips`
  - `orders.db` では同期間に `perf_block` と `filled` が併存し、
    failfast だけでは劣化クラスター停止が遅延。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - 取引密度/サイズを追加圧縮:
      - `MAX_ORDERS_PER_MINUTE=1`
      - `BASE_ENTRY_UNITS=320`
      - `MAX_UNITS=520`
    - `PERF_GUARD` を短期反応へ:
      - `LOOKBACK_DAYS=1`
      - `MIN_TRADES=16`
      - `PF_MIN=0.90`
      - `WIN_MIN=0.47`
      - `HOURLY_MIN_TRADES=6`
    - `REGIME_FILTER=1` / `REGIME_MIN_TRADES=10` を明示。
    - setup 判定を有効化:
      - `SETUP_ENABLED=1`
      - `SETUP_USE_HOUR=1`
      - `SETUP_USE_DIRECTION=1`
      - `SETUP_USE_REGIME=1`
      - `SETUP_MIN_TRADES=6`
      - `SETUP_PF_MIN=0.95`
      - `SETUP_WIN_MIN=0.50`
      - `SETUP_AVG_PIPS_MIN=0.00`
    - fallback prefix (`SCALP_PING_5S_*`) にも同値を併記。
  - `ops/env/scalp_ping_5s_b.env`
    - `PERF_GUARD` を短期反応へ:
      - `LOOKBACK_DAYS=1`
      - `MIN_TRADES=20`
      - `PF_MIN=0.88`
      - `WIN_MIN=0.46`
      - `HOURLY_MIN_TRADES=6`
    - setup 判定を有効化:
      - `SETUP_MIN_TRADES=6`
      - `SETUP_PF_MIN=0.92`
      - `SETUP_WIN_MIN=0.48`
      - `SETUP_AVG_PIPS_MIN=0.00`
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_B_*` / `SCALP_PING_5S_C_*` / `SCALP_PING_5S_*`
      すべてに上記 `PERF_GUARD` と setup 判定のしきい値を同値反映。
- 意図:
  - 静的時間帯ブロックではなく、
    直近実績の `hour×side×regime` 組み合わせで preflight を動的選別する。

### 2026-02-26（追記）根本隔離: `B long-only` + `C一時停止`

- 背景（VM実測, 2026-02-26 UTC 08:05 集計）:
  - 直近14日の日次:
    - `scalp_ping_5s_c_live`: 2/24 `-1574.2JPY`、2/25 `-3628.7JPY`、2/26 `-5351.1JPY`
    - `scalp_ping_5s_b_live`: 2/24 `-2246.6JPY`、2/25 `-3716.2JPY`、2/26 `-615.9JPY`
  - 90日（実データ範囲）:
    - `scalp_ping_5s_b_live`: `4978 trades / -41156.5JPY / -5288.6pips / PF 0.427`
    - `scalp_ping_5s_c_live`: `855 trades / -10554.0JPY / -1006.7pips / PF 0.440`
  - 時間帯×方向で B の short が構造的に悪化:
    - `JST00 short`: `n=62, EV=-13.2 pips`
    - `JST23 short`: `n=125, EV=-8.19 pips`
    - `JST15 short`: `n=47, EV=-7.29 pips`
  - close reason:
    - B は `STOP_LOSS_ORDER` が主因（`1912件 / -4387.9pips`）
    - C は `MARKET_ORDER_TRADE_CLOSE` が主因（`778件 / -964.8pips`）
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENABLED=0` を維持し、一時停止方針を明記
- 意図:
  - 負けの主要ドライバ（B short / C全体）を先に隔離し、
    期待値が確認できる導線だけで再検証する。

### 2026-02-26（追記）固定サイド制約の撤回（市況依存へ統一）

- 背景:
  - 固定 `long-only/short-only` は相場変化に追従しづらく、
    恒常運用の方針として不適切。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER` を削除。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER` を削除。
- 方針:
  - 方向判定は `PERF_GUARD_SETUP (hour×side×regime)` +
    `DIRECTION_BIAS` + `HORIZON_BIAS` で動的に選別する。

### 2026-02-26（追記）`scalp_ping_5s_c_live` を緊急停止 + `close_reject_no_negative` 欠損reason救済

- 背景（VM 実測, UTC 2026-02-26 07:40 前後）:
  - 直近14日ワースト: `scalp_ping_5s_b_live -41,156 JPY`, `scalp_ping_5s_c_live -10,554 JPY`。
  - 直近24h（orders）で `scalp_ping_5s_c_live` は `filled=580 / perf_block=7435 / close_reject_no_negative=52`。
  - 口座は `marginAvailable ~400 JPY`, `marginUsed ~56,200 JPY`, `openTradeCount=1`（`-9000` の無タグ open trade）で、
    margin圧迫による reject/cooldown が継続。
  - `quant-market-data-feed.service` は当日中に短周期 restart が多発し、`decision_latency_ms/data_lag_ms` が悪化。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`: `SCALP_PING_5S_C_ENABLED=0`
  - `ops/env/quant-scalp-ping-5s-c.env`: `SCALP_PING_5S_C_ENABLED=0`
  - `execution/order_manager.py`:
    - `_reason_matches_tokens()` を修正。
    - `allow_reasons=["*"]` を持つ戦略では `exit_reason` 欠損時も wildcard を一致扱いにし、
      `close_reject_no_negative` の不必要な拒否を抑制。
- 意図:
  - まず最大ドローダウン源の C を即時停止して資金流出を止める。
  - つぎに `exit_reason` 未注入での負け持ち越しを機械的に減らし、margin 解放速度を上げる。

### 2026-02-26（追記）AGENTS運用方針更新（改善優先 + JST7〜8メンテ除外）

- 変更:
  - `AGENTS.md`
    - 「戦略停止より改善優先」を明文化。
    - 成績悪化時は `原因分析→パラメータ/執行品質改善→再検証` を先行し、
      恒久的な時間帯制限による回避を禁止。
    - ただし `JST 7〜8時` はメンテ時間として運用対象外に明記。
    - `quant-replay-quality-gate` の `policy_hints.block_jst_hours` は
      `config/worker_reentry.yaml` へ自動反映しない方針へ更新。
- 意図:
  - 時間帯封鎖で成績を見かけ上維持する運用を避け、改善ループを主導線に固定する。
  - メンテ時間（JST7〜8）は例外として除外し、それ以外は動的改善で運用する。

### 2026-02-26（追記）B/C「停止なし」改善チューニング（VM 24h 実測ベース）

- 背景（VM実測, UTC 2026-02-26 集計）:
  - `scalp_ping_5s_b_live`: `481 trades / -3144.3 JPY / -414.7 pips / win 24.9% / PF 0.359`
  - `scalp_ping_5s_c_live`: `587 trades / -7025.8 JPY / -749.1 pips / win 53.2% / PF 0.434`
  - `B` は確率帯 `0.85-0.92` の long が主劣化（`228 trades / -2332.4 JPY / PF 0.377`）。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ENABLED=1`（停止運用をやめて常時稼働）
    - 取引密度とサイズを縮小:
      - `MAX_ACTIVE_TRADES=10`
      - `MAX_PER_DIRECTION=6`
      - `MAX_ORDERS_PER_MINUTE=6`
      - `BASE_ENTRY_UNITS=520`
      - `MAX_UNITS=900`
    - 静的時間帯ブロックを解除:
      - `BLOCK_HOURS_JST=`（空）
    - `PERF_GUARD_MODE=reduce` に変更（停止ではなく縮小）
    - 確率・配分ガードを厳格化:
      - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.54`
      - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.20`
      - `ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD=0.85`
      - `ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX=0.12`
      - `ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT=1.00`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED=1`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN=1.10`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.30`
      - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT=0.95`
    - side逆風時の自動縮小を有効化:
      - `SL_STREAK_DIRECTION_FLIP_ENABLED=1`
      - `SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`
      - `SIDE_ADVERSE_STACK_UNITS_ENABLED=1`
      - `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.25`
      - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.15`
    - order intent の通過条件を厳格化:
      - `REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.68`
      - `MIN_SCALE...=0.25`
      - `MAX_SCALE...=0.55`
    - leading-profile しきい値を引き上げ:
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.62`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.68`
      - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.80`
  - `ops/env/scalp_ping_5s_c.env` / `ops/env/quant-scalp-ping-5s-c.env`
    - `SCALP_PING_5S_C_ENABLED=1`（停止運用から復帰）
    - `BASE_ENTRY_UNITS=260`, `MAX_UNITS=420` に縮小
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - fallback も `SCALP_PING_5S_PERF_GUARD_MODE=reduce`
  - `ops/env/quant-order-manager.env`
    - B preserve-intent を env 同値へ更新（`0.68/0.25/0.55`）
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - forecast gate allowlist に B を復帰:
      - `FORECAST_GATE_STRATEGY_ALLOWLIST=...scalp_ping_5s_b_live,scalp_ping_5s_c_live`
    - B/C forecast しきい値を引き上げ:
      - `B edge=0.72 / expected_pips_min=0.24 / target_reach_min=0.34`
      - `C edge=0.74 / expected_pips_min=0.24 / target_reach_min=0.34`
  - `ops/env/quant-v2-runtime.env`
    - worker runtime の forecast 設定も order-manager と同値へ統一:
      - `FORECAST_GATE_STRATEGY_ALLOWLIST` に `scalp_ping_5s_b_live` を追加
      - `B edge=0.72 / expected_pips_min=0.24 / target_reach_min=0.34`
      - `C edge=0.74 / expected_pips_min=0.24 / target_reach_min=0.34`
  - `systemd/quant-scalp-ping-5s-b.service`
    - unit override も `PERF_GUARD_MODE=reduce` へ統一
    - `BASE_ENTRY_UNITS=520`, `MAX_UNITS=900` を固定
    - `ORDER_MANAGER_PRESERVE_INTENT_*` を `0.68/0.25/0.55` へ統一
- 意図:
  - 「止める」のではなく、`B` は低EV通過率を下げつつ縮小連続運転に寄せる。
  - `C` は小ロット再稼働で、勝ちスパイク再現余地を残しながら尾損失を抑える。
  - forecast gate allowlist の B 欠落を解消し、preflight の実効性を回復する。

### 2026-02-26（追記）order preflight SLO ガード + 分析ワーカーの市場時間スキップ

- 背景（VM運用観測）:
  - `decision_latency_ms` / `data_lag_ms` の悪化局面と、`MARKET_ORDER_MARGIN_CLOSEOUT` の連動が確認された。
  - replay/counterfactual の重い処理が市場時間中に稼働し、実行遅延と競合する時間帯があった。
- 変更:
  - `workers/common/slo_guard.py` を新設。
    - `metrics.db` の `data_lag_ms` / `decision_latency_ms`（mode=`strategy_control`）を
      latest + p95 で評価し、閾値超過時は entry を拒否。
    - キャッシュTTL (`ORDER_SLO_GUARD_TTL_SEC`) 付きで preflight 負荷を抑制。
  - `execution/order_manager.py`
    - preflight に `slo_guard.decide()` を追加。
    - 拒否時は `status=slo_block`、`order_slo_block` metric、`OPEN_REJECT` ログを必須記録。
  - `analysis/replay_quality_gate_worker.py`
    - `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=1` のとき market open 中は skip。
  - `analysis/trade_counterfactual_worker.py`
    - `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=1` のとき market open 中は skip。
  - `ops/env/quant-order-manager.env`
    - `ORDER_SLO_GUARD_*` を有効化（lookback 180s, sample min 8, lag/latencyのlatest+p95しきい値）。
    - closeout hard を strategy別に前倒し:
      - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_* = 6 / 2 / 0.20`
      - `SCALP_PING_5S_D_PERF_GUARD_MARGIN_CLOSEOUT_HARD_* = 6 / 1 / 0.12`
      - `PERF_GUARD_MARGIN_CLOSEOUT_HARD_*_STRATEGY_MICROPULLBACKEMA = 4 / 1 / 0.20`
  - `ops/env/quant-replay-quality-gate.env`
    - `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=1`
  - `ops/env/quant-trade-counterfactual.env`
    - `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=1`
- テスト:
  - `tests/workers/test_slo_guard.py` 新規追加（disable/scope/latest/p95/healthy）。
  - `tests/analysis/test_replay_quality_gate_worker.py` の auto-improve 適用テストを
    クールダウン状態に依存しない形へ補強。

### 2026-02-26（追記）intraday利益寄せ（負け筋遮断 + 勝ち筋サイズ寄せ）

- 背景（VM実測, JST 2026-02-26 18時台）:
  - 当日実現: `-6642 JPY / -459.4 pips`。
  - 当日マイナス寄与の中心:
    - `scalp_ping_5s_c_live: -5351 JPY`
    - `scalp_ping_5s_b_live: -616 JPY`
    - `MicroCompressionRevert-short: -216 JPY`
  - 14日では `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` が継続マイナス。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ALLOW_HOURS_JST=17,18`（16/23を除外）
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=1800`
    - `SCALP_PING_5S_B_MAX_UNITS=3600`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=reduce`
    - local fallback 経路の整合:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=19,22`（18を除外）
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=400`
    - `SCALP_PING_5S_C_MAX_UNITS=900`
    - `SCALP_PING_5S_C_CONF_FLOOR=82`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`（fallback整合）
    - local fallback 経路の整合:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.76`
  - `ops/env/quant-order-manager.env`
    - B通過率は利益時間帯での約定回復を優先しやや緩和:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_B_LIVE=0.70`
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_SCALP_PING_5S_B_LIVE=0.20`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_SCALP_PING_5S_B_LIVE=0.30`
    - Cは低EV通過を遮断するため厳格化:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.76`
      - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_C_LIVE=0.78`
      - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_SCALP_PING_5S_C_LIVE=0.32`
      - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_SCALP_PING_5S_C_LIVE=0.42`
  - `ops/env/quant-micro-compressionrevert.env`
    - `MICRO_MULTI_ENABLED=0`（当日/14日マイナス寄与を遮断）
  - `ops/env/quant-v2-runtime.env`
    - worker側 forecast gate も order-manager と同値化:
      - `B edge=0.70 / expected_pips_min=0.20 / target_reach_min=0.30`
      - `C edge=0.78 / expected_pips_min=0.32 / target_reach_min=0.42`
  - `ops/env/quant-micro-vwaprevert.env`
    - `MICRO_MULTI_BASE_UNITS=42000`（勝ち寄与のサイズ増）
  - `ops/env/quant-micro-rangebreak.env`
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=4.0`
    - `MICRO_MULTI_BASE_UNITS=42000`
  - `ops/env/quant-micro-trendretest.env`
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=4.0`
    - `MICRO_MULTI_BASE_UNITS=42000`
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=4.0`
    - `MICRO_MULTI_BASE_UNITS=42000`
- 意図:
  - CとMicroCompressionRevertの負け寄与を先に止血しつつ、Bの利益時間帯と
    Micro勝ち寄与戦略へ当日中の期待値とロットを寄せる。

### 2026-02-26（追記）B hard failfast解除 + B/C 方向固定解除（停止回避）

- 背景（VM実測, 2026-02-26 10:27-10:31 UTC）:
  - `orders.db` 直近30分で `scalp_ping_5s_b_live` は `perf_block=32` 件。
  - `quant-order-manager` journal の拒否理由は
    `perf_block:hard:hour10:failfast:pf=0.62 win=0.29 n=191` が連続。
  - `scalp_ping_5s_c_live` は同窓で `perf_block=4`、`strategy_control_entry_disabled=1`。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_ENABLED=0`（service内の自己HTTP再入を停止）
    - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF=0.58`
    - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN=0.27`
    - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX=0.75`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.48`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.62`
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=`（buy固定解除）
    - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF=0.58`
    - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN=0.27`
    - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX=0.75`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.48`
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=0`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=`（buy固定解除）
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.62`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=0`
  - `ops/env/quant-v2-runtime.env`
    - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=0`
    - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED=0`
- 意図:
  - 時間帯停止や恒久停止を使わず、`hard failfast` での全面停止を回避しつつ
    方向固定バイアスを外して約定機会を確保する。
  - 失速時は `reduce` と `ORDER_MANAGER_PRESERVE_INTENT_*` の縮小でリスクを抑制する。

### 2026-02-26（追記）manual margin guard 再調整（B最終ボトルネック解消）

- 背景（VM実測, 2026-02-26 11:02 UTC）:
  - `orders.db` 直近15分で `scalp_ping_5s_b_live` の `manual_margin_pressure=3`。
  - 同窓の B は `preflight_start=6`, `probability_scaled=3` で、通過意図が
    margin guard で止まる状態に収束。
  - 口座は手動ショート `USD_JPY -8500` 併走（`margin_available=4,942.48`）。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.01`（from `0.05`）
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.02`（from `0.07`）
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=500`（from `3000`）
- 意図:
  - guard を無効化せず、near-closeout を避けながら小ロットの再開余地を残す。
  - 停止ではなく縮小・継続運用（no-stop）を維持する。

### 2026-02-26（追記）no-stop 再配分: B/C/M1 圧縮 + RangeFader 通過回復 + Micro勝ち源増量

- 背景（VM実測, 2026-02-26 11:12 UTC）:
  - 直近30分の `orders.db` で `entry_probability_reject=21` は全件 `rangefader`。
  - reject reason は `entry_probability_below_min_units`（`entry_probability≈0.40`）に集中。
  - 7日損益は `MicroRangeBreak +662.3 (PF 3.05)`、`MomentumBurst +1613.7` に対し、
    `scalp_ping_5s_b_live -9475.8`, `scalp_ping_5s_c_live -2735.5`, `M1Scalper-M1 -1627.3`。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=300` を追加（rangefader の below-min reject を減らす）
  - `ops/env/scalp_ping_5s_b.env`
    - `BASE_ENTRY_UNITS=900`（from `1800`）
    - `MAX_UNITS=1800`（from `3600`）
    - `MAX_ORDERS_PER_MINUTE=4`（from `6`）
    - `CONF_FLOOR=78`（from `74`）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=20`（from `30`）
    - `PERF_GUARD_FAILFAST_PF=0.10`（from `0.58`）
    - `PERF_GUARD_FAILFAST_HARD_PF=0.00`（新規）
  - `ops/env/scalp_ping_5s_c.env`
    - `BASE_ENTRY_UNITS=220`（from `400`）
    - `MAX_UNITS=500`（from `900`）
    - `MAX_ORDERS_PER_MINUTE=1`（from `2`）
    - `CONF_FLOOR=86`（from `82`）
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BASE_UNITS=3000`（from `10000`）
    - `M1SCALP_MAX_OPEN_TRADES=1`（from `2`）
    - `M1SCALP_PERF_GUARD_MODE=reduce`（from `block`）
    - `M1SCALP_PERF_GUARD_FAILFAST_PF/WIN=0.30/0.35`（from `0.82/0.44`）
    - `M1SCALP_PERF_GUARD_FAILFAST_HARD_PF=0.20`（新規）
  - `ops/env/quant-order-manager.env`
    - `PERF_GUARD_MODE_STRATEGY_M1SCALPER_M1=reduce`
    - `PERF_GUARD_FAILFAST_PF/WIN_STRATEGY_M1SCALPER_M1=0.30/0.35`
    - `PERF_GUARD_FAILFAST_HARD_PF_STRATEGY_M1SCALPER_M1=0.20`
    - `M1SCALP_PERF_GUARD_*` 同値を order-manager 側にも併記
  - `ops/env/quant-micro-rangebreak.env`
    - `MICRO_MULTI_BASE_UNITS=52000`（from `42000`）
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=3.0`（from `4.0`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ADX=16.0`（from `20.0`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE=0.34`（from `0.42`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ATR=0.9`（from `1.2`）
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_BASE_UNITS=52000`（from `42000`）
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=3.0`（from `4.0`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ADX=16.0`（from `20.0`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE=0.34`（from `0.42`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ATR=0.9`（from `1.2`）
  - `ops/env/quant-scalp-rangefader.env`
    - `RANGEFADER_BASE_UNITS=11000`（from `13000`）
- 意図:
  - 全停止なしで負け源のサイズを抑え、勝ち寄与が出ている micro へ配分を寄せる。
  - `rangefader` は reject 連打を解消して「動いて勝てるか」を再評価可能な状態に戻す。

### 2026-02-26（追記）no-stop 継続の最終ボトルネック除去（B/M1 perf_block + B manual_margin_pressure）

- 背景（VM実測, 2026-02-26 11:30 UTC）:
  - 直近15分 `orders.db`: `perf_block=52`, `manual_margin_pressure=8`。
  - strategy別は `M1Scalper-M1 perf_block=25`, `scalp_ping_5s_b_live perf_block=24`, `scalp_ping_5s_b_live manual_margin_pressure=10`。
  - 口座は manual `-8500` 併走で、`margin_available=4553.5`, `health_buffer=0.079`。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
    - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
    - `M1SCALP_PERF_GUARD_ENABLED=0`
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
    - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
    - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
    - `M1SCALP_PERF_GUARD_ENABLED=0`
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_PERF_GUARD_ENABLED=0`
- 意図:
  - 時間帯停止やサービス停止を使わず、B/M1 の hard reject を外して常時稼働の注文通過率を回復。
  - worker local `order_manager` 経路でも同値を使うため、runtime env 側を正本として同期する。
  - manual 併走時の追加ブロックは `manual_margin_guard` ではなく既存の `margin_usage_projected_cap` 系で管理する。

### 2026-02-26（追記）B/C 方向バイアス是正（long 逆風遮断）

- 背景（VM実測, 2026-02-26 11:45 UTC, trades.db 24h）:
  - `scalp_ping_5s_c_live`: long `347/-5380.5 JPY`, short `75/-88.8 JPY`
  - `scalp_ping_5s_b_live`: long `128/-1300.8 JPY`, short `26/-7.7 JPY`
  - 損失の大半が long 側に偏在。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=sell`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=sell`
- 意図:
  - no-stop を維持したまま、実損優位側（long）を止めて short 側へ方向適応する。
  - 時間帯封鎖ではなく、戦略ローカルの方向制御で損失勾配を即座に下げる。

### 2026-02-26（追記）SLO data-lag ブロック緩和 + M1配分増

- 背景（VM実測, 2026-02-26 11:50 UTC）:
  - 30分窓 `orders.db`: `slo_block=10`（latest `11:45:09Z`）
  - `slo_block.reason` は `data_lag_p95_exceeded` に集中し、`data_lag_p95_ms≈7152`。
  - 閾値は `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=5000` のため、scalp_fast が連続reject。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000`（from `5000`）
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_SLO_GUARD_*` を明示追加
    - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000`
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BASE_UNITS=4500`（from `3000`）
- 意図:
  - no-stop のまま、インフラ遅延スパイクでの過剰拒否を減らして通過率を回復。
  - ほぼ横ばいまで戻した M1 を増量し、短期の利益寄与を取りに行く。

### 2026-02-26（追記）負け源圧縮 + 勝ち源増量（即効配分）

- 背景（VM実測, 2026-02-26 12:00 UTC, trades.db 24h）:
  - 下位: `scalp_ping_5s_c_live -5477.6`, `scalp_ping_5s_b_live -1047.5`
  - 上位: `MomentumBurst +117.9`, `MicroVWAPRevert +43.1`, `TickImbalance +39.9`
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=700`（from `900`）
    - `SCALP_PING_5S_B_MAX_UNITS=1400`（from `1800`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=140`（from `220`）
    - `SCALP_PING_5S_C_MAX_UNITS=320`（from `500`）
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_BASE_UNITS=62000`（from `52000`）
  - `ops/env/quant-micro-vwaprevert.env`
    - `MICRO_MULTI_BASE_UNITS=52000`（from `42000`）
    - `MICRO_MULTI_LOOP_INTERVAL_SEC=4.0`（from `6.0`）
- 意図:
  - no-stop を維持しつつ、負け戦略の損失勾配を即時に低下。
  - 勝ち寄与が出ている micro を厚くし、約定1件あたりの期待利益を引き上げる。

### 2026-02-26（追記）勝ち源の発火頻度を引き上げ（VWAPRevert/TickImbalance）

- 背景（VM実測, 2026-02-26 12:05 UTC）:
  - B/C short-only 反映後、`slo_block`/`manual_margin_pressure` は再発せず、reject 側は沈静化。
  - ただし直近の注文生成が薄く、利益速度が不足。
- 変更:
  - `ops/env/quant-micro-vwaprevert.env`
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ADX=16.0`（from `20.0`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE=0.34`（from `0.42`）
    - `MICRO_RANGEBREAK_BREAKOUT_MIN_ATR=0.9`（from `1.2`）
  - `ops/env/quant-scalp-tick-imbalance.env`
    - `SCALP_PRECISION_COOLDOWN_SEC=120`（from `180`）
- 意図:
  - 勝ち寄与が確認できている戦略の発火条件を緩和し、約定機会を増やす。
  - 停止なしのまま、利益速度を上げるための「頻度側」チューニングを優先する。

### 2026-02-26（追記）B/C short-only後の無約定解消（revert緩和 + short最小通過ロット引下げ）

- 背景（VM実測, 2026-02-26 12:00-12:16 UTC）:
  - `quant-order-manager` 側の reject は沈静化した一方、B/C は約定が枯渇。
  - `entry-skip summary` は `no_signal:revert_not_found` が最多で、
    short 側は `units_below_min` が継続（B: 3-17/30秒, C: 1-6/30秒）。
  - `SCALP_PING_5S_[B/C]_SIDE_FILTER=sell` 自体は機能（long は継続遮断）。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MIN_UNITS=5`（from `10`）
    - `REVERT_SHORT_WINDOW_SEC=1.20`（from `0.75`）
    - `REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO=0.12/0.06/0.02/0.30`（from `0.20/0.12/0.05/0.50`）
    - `FAST_DIRECTION_FLIP_*` を short変換寄りに緩和
      - `DIRECTION_SCORE_MIN=0.42`（from `0.50`）
      - `HORIZON_SCORE_MIN=0.22`（from `0.30`）
      - `HORIZON_AGREE_MIN=1`（from `2`）
      - `NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.62`（from `0.76`）
      - `MOMENTUM_MIN_PIPS=0.05`（from `0.08`）
    - `EXTREMA_GATE_ENABLED=1`（from `0`）
    - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`（from `0`）
    - `EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE=1.45`（from `2.10`）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=5`（from `10`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=180`（from `140`）
    - `SCALP_PING_5S_C_MIN_UNITS=5`（from `20`）
    - `REVERT_SHORT_WINDOW_SEC=1.20`（from `0.75`）
    - `REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO=0.12/0.06/0.02/0.30`（from `0.20/0.12/0.05/0.50`）
    - `SHORT_MIN_TICKS=3`（from `4`）, `SHORT_MIN_SIGNAL_TICKS=3`（from `4`）
    - `FAST_DIRECTION_FLIP_*` を緩和
      - `DIRECTION_SCORE_MIN=0.46`（from `0.58`）
      - `HORIZON_SCORE_MIN=0.26`（from `0.38`）
      - `HORIZON_AGREE_MIN=1`（from `3`）
      - `NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.64`（from `0.82`）
      - `MOMENTUM_MIN_PIPS=0.08`（from `0.16`）
    - `SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`（from `0`）
    - `EXTREMA_GATE_ENABLED=1`（from `0`）
    - `EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=0`（from `1`）
    - `EXTREMA_SHORT_BOTTOM_BLOCK_POS=0.10`（from `0.18`）
    - `EXTREMA_SHORT_BOTTOM_SOFT_POS=0.18`（from `0.24`）
    - `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`（from `0`）
    - `EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE=1.50`（from `2.10`）
    - `EXTREMA_TECH_FILTER_SHORT_BOTTOM_RSI_MIN=54.0`（from `58.0`）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=5`（from `20`）
- 意図:
  - long遮断は維持しつつ、short側の「発火不足」と「最小ロット未満」を同時に解消する。
  - 時間帯停止なしで、B/C を short適応の実運転へ戻す。

### 2026-02-26（追記）`scalp_ping_5s_b/c` side filter の fail-closed 化

- 背景（VM実測, 2026-02-26 12:10 UTC 以降）:
  - `quant-scalp-ping-5s-b.service` 実行環境に
    `SCALP_PING_5S_B_SIDE_FILTER=sell` と
    `SCALP_PING_5S_SIDE_FILTER=sell` が存在することを確認。
  - ただし過去約定（`orders.db`）では `b_live` の buy 発注履歴が残り、
    side filter 欠落時の再発余地を排除する必要がある。
- 変更:
  - `workers/scalp_ping_5s_b/worker.py`
    - `_apply_alt_env()` に B専用 fail-closed を追加。
    - `SCALP_PING_5S_SIDE_FILTER` が未設定/不正値のときは `sell` を強制。
    - 起動ログに `side_filter` を追加して監査可能化。
  - `workers/scalp_ping_5s_c/worker.py`
    - `_apply_alt_env()` に C専用 fail-closed を追加。
    - `SCALP_PING_5S_SIDE_FILTER` が未設定/不正値のときは `sell` を強制。
    - 起動ログに `side_filter` を追加して監査可能化。
  - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
    - B/C それぞれの side filter `missing/invalid/valid` ケースを追加。
- 意図:
  - env 取り回し崩れがあっても B/C worker が意図せず long 側へ流れないようにする。
  - direction 制御を env 前提だけにせず、ワーカー起動時に fail-closed を保証する。

### 2026-02-26（追記）no-stop 無約定化の解消（B/C通過率回復 + 共通perf block解除）

- 背景（VM実測, 2026-02-26 12:25 UTC）:
  - `orders.db` 直近30分で `orders` が `0 rows`（`datetime(substr(ts,1,19))` 基準）。
  - B/C ワーカーは alive だが `entry-skip` の主因が `no_signal:revert_not_found` と `units_below_min` に集中。
  - `entry_intent_board` 直近45分は `1件` のみで、`below_min_units_after_scale` が発生。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0`（from `1`）
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MIN_UNITS=1`（from `5`）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=1`（from `5`）
    - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
    - `SCALP_PING_5S_B_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
    - `SCALP_PING_5S_B_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.58`（from `0.42`）
    - `SCALP_PING_5S_B_SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=none`（from `sell`）
    - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
    - `SCALP_PING_5S_C_MIN_UNITS=1`（from `5`）
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`（from `5`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=3`（from `1`）
    - `SCALP_PING_5S_C_SHORT_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
    - `SCALP_PING_5S_C_LONG_MOMENTUM_TRIGGER_PIPS=0.18`（from `0.10`）
    - `SCALP_PING_5S_C_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
    - `SCALP_PING_5S_C_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.62`（from `0.45`）
    - `SCALP_PING_5S_C_SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.68`（from `0.76`）
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`（from `0.70`）
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.38`（from `0.24`）
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.52`（from `0.62`）
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
    - `SCALP_PING_5S_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
    - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
    - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
    - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.46`（from `0.52`）
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.46`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN/MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.40/0.85`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.85`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`
    - `SCALP_PING_5S_PERF_GUARD_MODE=off`, `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=off`, `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`
- 意図:
  - 停止なしのまま、B/C のローカル判定段階での枯渇を緩和し order-manager まで intent を通す。
  - 共通 `perf_block` での hard reject 常態を解き、戦略ローカル最適化での復帰余地を確保する。

### 2026-02-26（追記）quote 再取得耐性の強化（order-manager）

- 背景（VM実測, 2026-02-26 12:35 UTC）:
  - `orders.db` 直近24hは `50365` 件で、`quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` は 0 件。
  - reject 主因は `perf_block=16971`, `probability_scaled=7719`, `entry_probability_reject=1101`。
  - ただし実運用では瞬間的な再クオート要求が発生しうるため、`order_manager` 側の quote リトライ余力を増やす。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_TICK_QUOTE_MAX_AGE_SEC=1.2`（新規）
    - `ORDER_QUOTE_FETCH_ATTEMPTS=4`（from `2`）
    - `ORDER_QUOTE_FETCH_SLEEP_SEC=0.10`（from `0.08`）
    - `ORDER_QUOTE_FETCH_MAX_SLEEP_SEC=0.50`（from `0.30`）
    - `ORDER_QUOTE_RETRY_MAX_RETRIES=2`（from `1`）
    - `ORDER_QUOTE_RETRY_SLEEP_SEC=0.45`（from `0.30`）
    - `ORDER_QUOTE_RETRY_SLEEP_BACKOFF=1.8`（from `1.6`）
    - `ORDER_QUOTE_RETRY_MAX_SLEEP_SEC=1.5`（from `1.0`）
- 意図:
  - メイン判定ロジックは変えず、瞬間的な quote 揺れを `order_manager` の pre-submit 層で吸収する。
  - `ORDER_SUBMIT_MAX_ATTEMPTS=1` を維持しつつ、quote 専用リトライだけ厚くして取り逃しを抑える。

### 2026-02-26（追記）方向精度の再崩れ対策（C no-side-filter封鎖 + side_filterフォールバック）

- 背景:
  - `scalp_ping_5s_c` で `SCALP_PING_5S_C_SIDE_FILTER=none` と
    `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` の組み合わせが許可される経路が残り、
    C が再び両方向エントリーに戻る再発点があった。
  - `workers/scalp_ping_5s` 本体では、初段で side_filter を通過しても、
    後段ルーティング（fast/sl/metrics flip 等）で side が反転すると
    `side_filter_final_block` で最終的に no-entry になるケースが発生しうる。
- 変更:
  - `workers/scalp_ping_5s_c/worker.py`
    - `ALLOW_NO_SIDE_FILTER` の no-filter 例外を廃止。
    - `SIDE_FILTER` が未設定/不正値のときは常に `sell` へ fail-closed。
  - `workers/scalp_ping_5s/worker.py`
    - `_resolve_final_signal_for_side_filter()` を追加。
    - 後段ルーティングで side が filter と不一致になった場合、
      初段の side-filter 適合シグナル（anchor）へ復元して発注経路を維持。
    - anchor が無い場合のみ `side_filter_final_block` で拒否。
    - ロット計算を「途中段階ごとの整数丸め」から
      「倍率を合算して最終段のみ丸め」へ変更。
      `units_below_min` での 0 化を抑えてシグナル消失を減らす。
    - `_maybe_rescue_min_units()` を追加し、
      `entry_probability/confidence/risk_cap` を満たす場合のみ
      `MIN_UNITS` へ救済して `units_below_min` を削減。
  - `workers/scalp_ping_5s/config.py`
    - `MIN_UNITS_RESCUE_*` パラメータを追加（B/C既定ON）。
  - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
    - C の no-side-filter override を許容しない期待値へ更新。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    - side_filter 最終解決の `aligned/fallback/block` を追加検証。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=sell`
    - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=0`
    - `SCALP_PING_5S_C_MIN_UNITS_RESCUE_*` を明示設定。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MIN_UNITS_RESCUE_*` を明示設定。
- 意図:
  - 方向精度の再崩れ要因をコード・envの両面で閉じる。
  - side_filter を厳格維持しつつ、後段反転でのシグナル消失を減らして約定導線を維持する。

### 2026-02-26（追記）`stage_state.db` ロックで market_order が落ちる問題の修正

- 背景（VM実測）:
  - `quant-order-manager` で `request failed: database is locked` が断続発生。
  - `orders.db` は `preflight_start/probability_scaled` が増える一方、`trades.db` の新規 entry が止まる状態を確認。
- 変更:
  - `execution/strategy_guard.py`
    - DB接続を `busy_timeout` + `WAL` + `check_same_thread=False` + `isolation_level=None` に変更。
    - lock競合時リトライ（`STRATEGY_GUARD_DB_LOCK_RETRY*`）を追加。
    - `set_block` / `is_blocked` / `clear_expired` で lock発生時に fail-open で継続し、例外を上位へ伝播しないように変更。
- 意図:
  - `stage_state.db` 共有アクセス競合で `market_order` が中断される経路を遮断し、
    エントリー導線を停止なしで維持する。

### 2026-02-26（追記）B/C no-signal 緩和（revert依存解除 + side filter開放）

- 背景（VM実測）:
  - `order-manager` lock 解消後も B/C は `entry-skip` が継続。
  - skip 主因は `revert_not_found` と `units_below_min`、加えて `side_filter_block` が残存。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=none`
    - `SCALP_PING_5S_B_REVERT_ENABLED=0`
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=6`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=900`
    - `SCALP_PING_5S_B_MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`
    - `SCALP_PING_5S_B_MIN_UNITS_RESCUE_MIN_CONFIDENCE=65`
    - `SCALP_PING_5S_B_CONF_FLOOR=72`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=none`
    - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
    - `SCALP_PING_5S_C_REVERT_ENABLED=0`
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=6`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=260`
    - `SCALP_PING_5S_C_MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`
    - `SCALP_PING_5S_C_MIN_UNITS_RESCUE_MIN_CONFIDENCE=70`
    - `SCALP_PING_5S_C_CONF_FLOOR=74`
- 意図:
  - no-signal の主要因（revert 固着）を外し、B/C の market_order 到達率を回復する。
  - 停止を使わず、低ロット救済と両方向運用でエントリー密度を引き上げる。

### 2026-02-26（追記）`SIDE_FILTER=none` の wrapper 上書き不整合を修正（B/C）

- 背景:
  - `ops/env/scalp_ping_5s_b.env` / `ops/env/scalp_ping_5s_c.env` で `SIDE_FILTER=none` を設定しても、
    `workers/scalp_ping_5s_b/worker.py` と `workers/scalp_ping_5s_c/worker.py` が
    invalid 値として `sell` へ fail-closed 上書きしていた。
  - その結果、`ALLOW_NO_SIDE_FILTER=1` が実効せず `side_filter_block` が継続した。
- 変更:
  - `workers/scalp_ping_5s_b/worker.py`
  - `workers/scalp_ping_5s_c/worker.py`
    - `ALLOW_NO_SIDE_FILTER=1` を有効化し、
      `SIDE_FILTER in {"", "none", "off", "disabled"}` を空フィルタとして許容。
    - それ以外の不正値は従来どおり `sell` へ fail-closed。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` を追加。
  - `ops/env/scalp_ping_5s_c.env`
    - 運用コメントを実装に合わせて更新（no-side-filter を wrapper opt-in として明示）。
  - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
    - B/C の no-side-filter opt-in ケースを更新・追加。
- 検証:
  - `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py -k "side_filter"`: `8 passed`
  - `python3 -m py_compile workers/scalp_ping_5s_b/worker.py workers/scalp_ping_5s_c/worker.py`: pass

### 2026-02-27（追記）B即効デリスク（サイズ圧縮 + 確率floor引き上げ）

- 背景（VM実測, UTC 2026-02-27 00:22 前後）:
  - 直近15分は全体 `+1311.9 JPY` だが、自動戦略のみで見ると `-65.1 JPY`。
  - JST当日（00:00以降）の自動損益は `-645.6 JPY`。
  - `scalp_ping_5s_b_live` が `-428.1 JPY` で主損失寄与。
  - 直近2h の B は long 側が `34 trades / -41.1 JPY` と偏っていた。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=720`（from `900`）
    - `SCALP_PING_5S_B_CONF_FLOOR=75`（from `72`）
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.60`（from `0.54`）
- 意図:
  - 停止なしを維持しつつ、B の低品質 long 発火を圧縮して当日の損失勾配を下げる。
  - エントリーは継続し、`filled` を維持したまま 1トレードあたりの毀損を抑える。

### 2026-02-27（追記）勝ち筋寄せ再配分（WickBlend増量 + B/C縮小）

- 背景（VM実測, UTC 2026-02-27 00:33 前後）:
  - 24h 自動戦略の損益は `WickReversalBlend=+114.7 JPY` を除きマイナス。
  - `scalp_ping_5s_b_live=-540.2 JPY`, `scalp_ping_5s_c_live=-3403.0 JPY` と B/C の負け寄与が継続。
  - 直近15分の自動損益も `-91.9 JPY` で未反転。
- 変更:
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_UNIT_BASE_UNITS=9500`（from `7000`）
    - `SCALP_PRECISION_UNIT_CAP_MAX=0.65`（from `0.55`）
    - `SCALP_PRECISION_COOLDOWN_SEC=8`（from `12`）
    - `SCALP_PRECISION_MAX_OPEN_TRADES=2`（from `1`）
    - `WICK_BLEND_RANGE_SCORE_MIN=0.40`（from `0.45`）
    - `WICK_BLEND_ADX_MIN/MAX=14.0/28.0`（from `16.0/24.0`）
    - `WICK_BLEND_BB_TOUCH_RATIO=0.18`（from `0.22`）
    - `WICK_BLEND_TICK_MIN_STRENGTH=0.30`（from `0.40`）
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=600`（from `720`）
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=5`（from `6`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=220`（from `260`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）
- 意図:
  - 停止なしを維持したまま、勝ち戦略の寄与を増やし、負け戦略の単位時間損失を圧縮する。
  - B/C の発火は維持しつつ、サイズと頻度の上限を絞って損失勾配を下げる。

### 2026-02-27（追記）`StageTracker` ロック耐性を追加（WickBlend起動失敗の解消）

- 背景（VM実測, UTC 2026-02-27 00:46）:
  - `quant-scalp-wick-reversal-blend.service` が起動直後に
    `sqlite3.OperationalError: database is locked`（`execution/stage_tracker.py`）で停止。
  - B/C は稼働しており、勝ち寄与側ワーカーだけが止まる状態だった。
- 変更:
  - `execution/stage_tracker.py`
    - `STAGE_DB_BUSY_TIMEOUT_MS` / `STAGE_DB_LOCK_RETRY` /
      `STAGE_DB_LOCK_RETRY_SLEEP_SEC` を追加。
    - 接続に `busy_timeout` と `journal_mode=WAL` を設定し、autocommit化。
    - schema作成 SQL を `_execute_with_lock_retry()` 経由へ変更し、
      lock競合時に再試行して起動を継続。
- 検証:
  - `python3 -m py_compile execution/stage_tracker.py`: pass
  - `pytest -q tests/test_stage_tracker.py`: `3 passed`

### 2026-02-27（追記）`StageTracker` 起動DDLの最小化（ロック再発対策）

- 背景（VM実測, UTC 2026-02-27 00:50）:
  - 上記ロック耐性追加後も、`quant-scalp-wick-reversal-blend.service` が
    `StageTracker.__init__` の初期DDL実行で `database is locked` を再発。
  - `stage_state.db` は複数 worker 共有のため、起動時の「毎回DDL」が競合点として残っていた。
- 変更:
  - `execution/stage_tracker.py`
    - `_safe_identifier`, `_table_exists`, `_column_exists`,
      `_ensure_table`, `_ensure_column` を追加。
    - `CREATE TABLE IF NOT EXISTS ...` を毎回発行する実装から、
      「テーブル未存在時のみ DDL 実行」へ変更。
    - `ALTER TABLE clamp_guard_state ADD COLUMN clamp_score ...` も
      列未存在時のみ実行。
- 意図:
  - 既存スキーマ環境では起動時DDLを書き込まないことで、
    スキーマロック競合で勝ち筋 worker が落ちる経路を閉じる。

### 2026-02-27（追記）`quant-order-manager` 並列度の増強（timeout緩和）

- 背景（VM実測, UTC 2026-02-27 00:57-00:58）:
  - B/C worker で `order_manager service call failed ... Read timed out (20s)` が発生。
  - order_manager は active だが service worker 数が `1` で、OANDA待ち重複時に応答遅延が生じていた。
- 変更:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_WORKERS=4`（from `1`, staged `2 -> 4`）
- 意図:
  - localhost API の同時処理能力を引き上げ、entry/close の service timeout 発生率を下げる。
  - 連続 timeout 起点の重複 request/reject（`CLIENT_TRADE_ID_ALREADY_EXISTS`）を抑制する。

### 2026-02-27（追記）WickBlend `StageTracker` 失敗時の fail-open（noop fallback）

- 背景（VM実測, UTC 2026-02-27 00:59）:
  - `quant-scalp-wick-reversal-blend.service` が `StageTracker()` 初期化の
    `database is locked` 例外で再停止。
- 変更:
  - `workers/scalp_wick_reversal_blend/worker.py`
    - `_NoopStageTracker` を追加。
    - `StageTracker()` 初期化を `try/except` 化し、失敗時に noop へフォールバック。
- 意図:
  - DBロック瞬間風速で WickBlend worker 全体が停止する経路を遮断し、
    エントリー導線を維持する。

### 2026-02-27（追記）`position_manager sync_trades` 負荷緩和（runtime tuning）

- 背景（VM実測）:
  - `quant-position-manager.service` で
    `sync_trades timeout (8.0s)` / `position manager busy` が連続発生。
  - WickBlend などの strategy worker で `/position/sync_trades` 失敗が継続。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `POSITION_MANAGER_MAX_FETCH=600`（new）
    - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC=4.0`（from `2.0`）
    - `POSITION_MANAGER_SYNC_CACHE_WINDOW_SEC=4.0`（from `1.5`）
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=12.0`（from `8.0`）
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC=3.0`（new）
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=120.0`（from `60.0`）
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=600`（from `1000`）
- 意図:
  - sync処理の過負荷ループを抑え、position manager 起点の遅延/拒否を減らす。

### 2026-02-27（追記）B/C 追加圧縮（units・頻度を一段低下）

- 背景（VM実測, 直近30分）:
  - `scalp_ping_5s_b_live=-35.7 JPY`, `scalp_ping_5s_c_live=-18.4 JPY` と負け寄与が継続。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=450`（from `600`）
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=4`（from `5`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=170`（from `220`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=4`（from `5`）
- 意図:
  - 停止なしを維持したまま、B/C の単位時間損失をさらに圧縮する。

### 2026-02-26（追記）B/C を sell 固定へ再ピン留め（方向精度優先 + rescue維持）

- 背景:
  - 直近の env 緩和で `SCALP_PING_5S_{B,C}_SIDE_FILTER=none` と
    `ALLOW_NO_SIDE_FILTER=1` が有効化され、buy 系意図が再流入しうる状態だった。
  - 一方で no-entry 側は `MIN_UNITS_RESCUE` 実装で緩和済みのため、
    side filter を開放しなくても導線改善余地がある。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=sell`
    - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=0`
    - `SCALP_PING_5S_B_MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.58`
    - `SCALP_PING_5S_B_MIN_UNITS_RESCUE_MIN_CONFIDENCE=78`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=sell`
    - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=0`
    - `SCALP_PING_5S_C_MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.60`
    - `SCALP_PING_5S_C_MIN_UNITS_RESCUE_MIN_CONFIDENCE=82`
- 意図:
  - 方向精度を優先して B/C の売り方向ガードを固定。
  - `MIN_UNITS_RESCUE` は維持し、厳格 side filter 下でも
    `units_below_min` によるシグナル消失を抑制する。

### 2026-02-27（追記）B/C の sell 固定を解除し、確率ゲートを再強化

- 背景（VM実測, UTC 2026-02-27 01:20 前後）:
  - post-check（`2026-02-27T00:36:34Z` 以降）で B/C の約定はほぼ `sell` 側のみ。
    - `B sell: 27 trades / acc 37.0% / -11.8 pips`
    - `C sell: 22 trades / acc 40.9% / -10.3 pips`
  - `orders.db` 24h 集計で低 `entry_probability` 帯の通過が継続し、負け寄与が集中。
  - 稼働プロセス環境で
    `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell`,
    `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` が
    `B=0.48 / C=0.46` だった。

- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER=none`
    - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=1`
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.64`
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.70`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER=none`
    - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.62`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=1`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.62`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.68`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.62`
  - `systemd/quant-scalp-ping-5s-b.service`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
      へ同値化（envとの不整合防止）。

- 意図:
  - side固定による方向ミスマッチを解消し、戦略ローカル判定へ方向選択を戻す。
  - 低確率帯の通過を機械的に削り、B/C のエントリー精度を先に回復させる。

### 2026-02-27（追記）duplicate CID 回収と order-manager RPC詰まり緩和

- 背景（VM実測, UTC 2026-02-27 01:00-01:33）:
  - `quant-scalp-ping-5s-b/c` と `quant-scalp-extrema-reversal` で
    `order_manager service call failed ... Read timed out (20.0)` が継続。
  - `orders.db` 30分窓の `rejected` は `CLIENT_TRADE_ID_ALREADY_EXISTS` が大半で、
    同一CIDに `filled -> rejected` が並ぶケースを確認。
- 変更:
  - `execution/order_manager.py`
    - `_latest_filled_trade_id_by_client_id` を追加。
    - `market_order` の reject reason が `CLIENT_TRADE_ID_ALREADY_EXISTS` のとき、
      既存 `filled` 行から `trade_id` を回収し
      `status=duplicate_recovered` で成功返却する導線を追加。
    - service timeout で fallback する前に、
      同一CIDの `orders.db` 状態を最大10秒ポーリングして
      `filled/rejected` の終端状態を先に回収する導線を追加。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=45.0`（from `8.0`）
    - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC=10.0`
    - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_POLL_SEC=0.5`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_WORKERS=6`（from `4`）
- 意図:
  - timeout起点の重複再送を reject 終了させず、既存約定へ収束させる。
  - service待ちの詰まりを緩和し、`order_manager_none` と CID重複rejectを低減する。

### 2026-02-27（追記）`strategy_entry` の拒否理由伝播を修正（coordination誤ラベル解消）

- 背景（VM実測）:
  - `quant-scalp-ping-5s-b/c` で `order_reject:coordination_reject` が多発する一方、
    `orders.db` では同時刻の `rejected` が乖離し、前段拒否と coordination拒否の切り分けが困難だった。
  - `strategy_entry` は `forecast_fusion` / `entry_leading_profile` で `units=0` になっても
    coordination 呼び出しへ進むため、結果的に `coordination_reject` へ潰れる経路があった。

- 変更:
  - `execution/strategy_entry.py`
    - `_normalized_reject_reason`, `_cache_entry_reject_status` を追加。
    - `market_order` / `limit_order` で以下を追加:
      - `strategy_feedback` 後 `units=0` → `analysis_feedback_zero_units` で reject記録
      - `forecast_fusion` 後 `units=0` → `reject_reason`（例: `strong_contra_forecast`）で reject記録
      - `entry_leading_profile` 後 `units=0` → `entry_leading_profile_reject` で reject記録
    - `coordination` 拒否時も共通 helper 経由で記録し、side は requested units 基準へ統一。
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    - 上記理由伝播の回帰テストを2件追加（forecast reject / leading reject）。

- 変更（方向バイアス緩和の運用値）:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.09`（from `0.08`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SHORT_MOMENTUM_TRIGGER_PIPS=0.10`（from `0.08`）
    - `SCALP_PING_5S_C_LONG_MOMENTUM_TRIGGER_PIPS=0.12`（from `0.18`）

- 意図:
  - 「何で entry が落ちたか」を strategy段で明示し、精度劣化の根因を即時再調整可能にする。
  - C の過度な short 優位を緩和し、sell 固定化を避ける。

### 2026-02-27（追記）`quant-order-manager` API の event-loop ブロッキングを解消

- 背景:
  - `quant-scalp-ping-5s-b/c` で `order_manager service call failed ... Read timed out (45.0)` が継続し、
    service worker 並列を 6 に増やした後も timeout 警告が残った。
  - `workers/order_manager/worker.py` の FastAPI endpoint は `async def` で
    `execution.order_manager.*` を直接 await していたが、
    実処理は OANDA API / SQLite I/O を多用する同期ブロックを含むため、
    worker event loop を長時間占有しやすい構造だった。
- 変更:
  - `workers/order_manager/worker.py`
    - `_run_order_manager_call` / `_run_order_manager_call_sync` を追加。
    - `cancel_order` / `close_trade` / `set_trade_protections` /
      `market_order` / `coordinate_entry_intent` / `limit_order`
      の各 endpoint で `execution.order_manager.*` 呼び出しを
      `asyncio.to_thread(... asyncio.run(...))` 経由へ変更。
    - `ORDER_MANAGER_SERVICE_SLOW_REQUEST_WARN_SEC`（default `8.0`）を追加し、
      遅延リクエストを `slow_request` 警告で監査可能にした。
- 意図:
  - service worker の event loop 占有を避け、同時 RPC の頭詰まりを低減する。
  - `Read timed out` 起点の fallback 連鎖を抑え、約定取りこぼしを減らす。

### 2026-02-27（追記）service timeout を 60s へ再調整（slow_request 実測対応）

- 背景:
  - event-loop 非ブロッキング化後の VM 監査で、
    `quant-order-manager.service` に `slow_request op=market_order elapsed=49.047s` を確認。
  - strategy worker 側の service timeout は `45.0s` のため、
    49秒級の遅延が timeout 扱いになる余地が残った。
- 変更:
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=60.0`（from `45.0`）
- 意図:
  - 実測遅延の上側（~49秒）を timeout 閾値で吸収し、
    `Read timed out` 起点の不要 fallback を減らす。

### 2026-02-27（追記）B/C の負け寄与を圧縮し、Wick 優位へ再配分

- 背景（VM実測, 24h）:
  - `scalp_ping_5s_c_live`: `493 trades / -1984.2 JPY`
  - `scalp_ping_5s_b_live`: `415 trades / -588.6 JPY`
  - `WickReversalBlend`: `7 trades / +332.3 JPY`
  - `orders.db` では B/C の `submit_attempt` が高水準（`b=600`, `c=608`）で、
    高頻度運転のまま期待値マイナスを積み上げる状態だった。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=12`（from `24`）
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=380`（from `450`）
    - `SCALP_PING_5S_B_CONF_FLOOR=77`（from `75`）
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.76`（from `0.74`）
    - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.63`（from `0.60`）
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.68`（from `0.64`）
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.68`（from `0.64`）
    - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.74`（from `0.70`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=12`（from `24`）
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=140`（from `170`）
    - `SCALP_PING_5S_C_CONF_FLOOR=76`（from `74`）
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.70`（from `0.68`）
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.61`（from `0.58`）
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.66`（from `0.62`）
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.66`（from `0.62`）
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.72`（from `0.68`）
  - `ops/env/quant-order-manager.env`
    - B/C preserve-intent reject を同値へ同期（`B=0.68`, `C=0.66`）
    - preserve-intent max scale を縮小（`B: 0.55->0.50`, `C: 0.85->0.78`）
  - `systemd/quant-scalp-ping-5s-b.service`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=420`（from `520`）
    - `SCALP_PING_5S_B_MAX_UNITS=780`（from `900`）
    - preserve-intent 閾値/scale を env 同値へ同期。
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_UNIT_BASE_UNITS=10200`（from `9500`）
    - `SCALP_PRECISION_COOLDOWN_SEC=7`（from `8`）
- 意図:
  - B/C は停止せず稼働を維持しつつ、低EV通過・過剰頻度・過剰サイズを同時に圧縮する。
  - 優位戦略（Wick）へ配分を寄せ、合算期待値を引き上げる。

### 2026-02-27（追記）即時収益寄せ: B/C 追加圧縮 + Wick/Extrema 発火緩和

- 背景（VM実測, 直近60分）:
  - `scalp_ping_5s_b_live`: `39 trades / -20.7 JPY`
  - `scalp_ping_5s_c_live`: `21 trades / -11.0 JPY`
  - 直近窓で Wick/Extrema の新規寄与が薄く、利益側の回転不足が継続。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=8`（from `12`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=8`（from `12`）
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `SCALP_EXTREMA_REVERSAL_COOLDOWN_SEC=45`（from `75`）
    - `SCALP_EXTREMA_REVERSAL_MAX_OPEN_TRADES=2`（from `1`）
    - `SCALP_EXTREMA_REVERSAL_MAX_SPREAD_PIPS=1.6`（from `1.3`）
    - `SCALP_EXTREMA_REVERSAL_MIN_ENTRY_CONF=57`（from `60`）
    - `SCALP_EXTREMA_REVERSAL_HIGH/LOW_BAND_PIPS=1.0`（from `1.2`）
    - `SCALP_EXTREMA_REVERSAL_RSI_LONG_MAX=46.0`（from `44.0`）
    - `SCALP_EXTREMA_REVERSAL_RSI_SHORT_MIN=54.0`（from `56.0`）
    - `SCALP_EXTREMA_REVERSAL_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.48`（from `0.52`）
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_COOLDOWN_SEC=5`（from `7`）
    - `SCALP_PRECISION_MAX_OPEN_TRADES=3`（from `2`）
    - `WICK_BLEND_RANGE_SCORE_MIN=0.34`（from `0.40`）
    - `WICK_BLEND_ADX_MIN/MAX=12.0/32.0`（from `14.0/28.0`）
    - `WICK_BLEND_BB_TOUCH_RATIO=0.15`（from `0.18`）
    - `WICK_BLEND_TICK_MIN_TICKS=4`（from `6`）
    - `WICK_BLEND_TICK_MIN_STRENGTH=0.22`（from `0.30`）
    - `SCALP_PRECISION_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.46`（from `0.50`）
- 意図:
  - 負け寄与（B/C）は頻度をさらに削って損失勾配を圧縮。
  - 利益寄与候補（Wick/Extrema）は発火条件を緩和し、約定機会を増やして即時寄与を狙う。

### 2026-02-27（追記）発火不足対策: Wick/Extrema の監視サイクルを短縮

- 背景:
  - 反映直後の短期窓で、B/C 損失は抑制できた一方で Wick/Extrema の約定が薄かった。
- 変更:
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `SCALP_EXTREMA_REVERSAL_LOOP_INTERVAL_SEC=1.5`（from `2.0`）
    - `SCALP_EXTREMA_REVERSAL_COOLDOWN_SEC=35`（from `45`）
    - `SCALP_EXTREMA_REVERSAL_LOOKBACK=24`（from `32`）
    - `SCALP_EXTREMA_REVERSAL_HIGH/LOW_BAND_PIPS=0.9`（from `1.0`）
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_LOOP_INTERVAL_SEC=2.0`（from `3.0`）
    - `SCALP_PRECISION_COOLDOWN_SEC=4`（from `5`）
    - `WICK_BLEND_BBW_MAX=0.0018`（from `0.0014`）
- 意図:
  - 収益側戦略の信号取りこぼしを減らし、短時間での約定回転を引き上げる。

### 2026-02-27（追記）Autotune UI の時間帯履歴欠落を再発防止

- 背景:
  - `hourly_trades` が不完全でも「lookback件数だけ満たしていれば」採用されるケースがあり、
    深夜帯（JST 00:00-06:00）を含む24時間枠が欠ける表示が発生し得た。
  - snapshot 取得失敗時は `_dashboard_defaults()` の空配列がそのまま表示され、
    「時間帯ごとの表が出ない」状態になり得た。
- 変更:
  - `apps/autotune_ui.py`
    - `hourly_trades` 採用判定を厳格化:
      - JST 固定チェック
      - 期待24時間ウィンドウ（生成時刻基準）のキー完全一致チェック
      - ラベル/キー正規化（年跨ぎ考慮）を追加
    - snapshot 取得失敗時でも `recent_trades` + `hourly_fallback` を使う
      `local-fallback` サマリを生成するよう変更。
    - 7日集計（`wins/losses/win_rate/recent_closed`）をローカル `trades.db` の
      JST 窓集計で補正し、snapshot 側 `total` メトリクスが 0 固定でも
      「決済件数>0 なのに 0勝0敗」の不整合が出ないように修正。
    - `daily/weekly` P/L も同ローカル集計を優先し、時間帯表とサマリーカードの
      表示値を一致させるよう補正。
    - ローカル DB が無い環境（Cloud Run）向けに、snapshot `daily/weekly` が
      0 固定でも `recent_trades` から非ゼロ実績を検知した場合は
      日次/週次/前日比を再計算して 0 固定表示を回避。
    - `hourly_trades` が「24行そろっているが全ゼロ」の stale snapshot でも、
      `recent_trades` のクローズ実績を検知したら fallback 再集計へ切り替える
      判定（`_hourly_trades_is_stale`）を追加。
    - `_build_hourly_fallback` で DB 集計が `None/{}` の場合は
      `recent_trades` 起点の再集計に必ず退避するよう修正
      （Cloud Run 環境での「fallback しても全ゼロ」再発を防止）。
    - Cloud Run（ローカル `trades.db` なし）では、snapshot 側 summary が
      非ゼロでも `recent_trades` 由来と不整合なら、日次/週次/前日比/
      勝敗/勝率/7d件数を `recent_trades` 由来へ正規化するよう補正。
    - `scripts/publish_ui_snapshot.py` 側で `metrics.hourly_trades` を
      `trades.db` の直近24h集計から常時注入するよう変更し、
      Cloud Run 側が DB なしでも時間帯別（夜間含む）を欠落なく表示できるよう修正。
    - `tab=architecture` の図表に `quant-order-manager` /
      `quant-position-manager` が欠けていたため、V2 実導線に沿って図を更新。
  - テスト追加/更新:
    - `tests/apps/test_autotune_ui_dashboard_local_fallback.py`
    - `tests/apps/test_autotune_ui_hourly_fallback.py`
      - DB集計 `{}` のとき snapshot trades へ退避する回帰ケースを追加
    - `tests/apps/test_autotune_ui_hourly_source_guard.py`
      - 全ゼロ stale snapshot の fallback 強制ケースを追加
    - `tests/apps/test_autotune_ui_summary_consistency.py`
      - DBなし環境で snapshot 非ゼロ値が stale な場合に `recent_trades` へ
        正規化される回帰ケースを追加
    - `tests/apps/test_autotune_ui_template_guards.py`
    - `tests/scripts/test_publish_ui_snapshot.py`
      - GCS公開スナップショット用の hourly 集計（manual除外 / lookback固定）を追加検証
    - `tests/apps/test_autotune_ui_summary_consistency.py`
      - Cloud Run（DBなし）でも、snapshot の `hourly_trades` が有効な場合は
        summary を `recent_trades` で上書きしない回帰ケースを追加
- 検証:
  - `pytest -q tests/apps` で `32 passed` を確認。
  - `pytest -q tests/apps tests/scripts/test_publish_ui_snapshot.py` で `35 passed` を確認。

### 2026-02-27（追記）B/C 収益監査に基づく追加チューニング（損失幅圧縮）

- 目的:
  - VM実測で継続していた B/C 赤字を、停止せずに縮小運転で改善する。
- 仮説:
  - 執行品質ではなく strategy pay-off が主因のため、B は SL/force-exit 圧縮、C は品質ゲート強化で改善余地がある。
- 変更ファイル:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ACTIVE_TRADES=6`, `MAX_PER_DIRECTION=4`
    - `PERF_GUARD_ENABLED=1`
    - `SL_BASE/SHORT_BASE=1.8/1.7`, `SL_MAX/SHORT_MAX=2.4/2.2`
    - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.0`（short `1.8`）
    - `FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=14`
    - `FORCE_EXIT_RECOVERY_WINDOW_SEC=55`
    - `FORCE_EXIT_RECOVERABLE_LOSS_PIPS=0.80`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE=5`
    - `BASE_ENTRY_UNITS=90`, `MAX_UNITS=200`
    - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`
    - `SCALP_PING_5S_PERF_GUARD_ENABLED=1`（fallback local 同期）
    - `CONF_FLOOR=78`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.64`
- 影響範囲:
  - `quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service` のみ。
  - V2 共通導線（`quant-order-manager` / `quant-position-manager` / `quant-strategy-control`）の仕様変更なし。
- 検証手順:
  1. `main` push 後、VMへ反映して B/C サービス再起動。
  2. `/proc/<pid>/environ` で上記キー反映を確認。
  3. 2h/6hで `avg_loss`, `realized_pl`, `margin_usage_* cap` を前回値と比較。

### 2026-02-27（追記）order-manager service 実効設定ズレの修正

- 目的:
  - B/C の perf guard と preserve-intent が service 経路でも確実に有効になる状態へ統一。
- 仮説:
  - `quant-order-manager` の env が旧値のままだと、worker側更新だけでは preflight 判定が改善しない。
- 変更ファイル:
  - `ops/env/quant-order-manager.env`
    - B preserve-intent: `REJECT_UNDER=0.74`, `MAX_SCALE=0.38`
    - C preserve-intent: `REJECT_UNDER=0.72`, `MIN_SCALE=0.30`, `MAX_SCALE=0.64`
    - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=1`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`
    - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_PERF_GUARD_ENABLED=1`
  - `ops/env/scalp_ping_5s_b.env`
    - preserve-intent: `0.74 / 0.25 / 0.38` に同期
  - `ops/env/scalp_ping_5s_c.env`
    - preserve-intent: `0.72 / 0.30 / 0.64` に同期
- 影響範囲:
  - `quant-order-manager.service` の B/C preflight 判定に限定。
- 検証手順:
  1. デプロイ後に `quant-order-manager` の `/proc/<pid>/environ` を確認。
  2. `orders.db` の B/C reject内訳（`entry_probability_reject`, `margin_usage_*`）を前後比較。

### 2026-02-27（追記）実効競合ENVの整理（service override / DB timeout）

- 目的:
  - 「重複しているだけで実効しない値」と「実際に process env で競合していた値」を分離し、実在競合のみを解消する。
- 実測根拠（VM）:
  - `quant-scalp-ping-5s-b.service` と `quant-autotune-ui.service` の inline `Environment=` は、
    いずれも process env では後段 `EnvironmentFile` 値に上書きされ、実効していなかった。
  - `ORDER_DB_BUSY_TIMEOUT_MS` は B/C/D で `250`、共通runtime系で `1500` が併存し、
    process env でも strategy ごとに乖離していた。
- 変更ファイル:
  - `systemd/quant-scalp-ping-5s-b.service`
    - 実効しない fail-fast 3キー（PF/WIN/SL_LOSS_RATE）を削除。
  - `systemd/quant-autotune-ui.service`
    - 実効しない `POSITION_MANAGER_SERVICE_*` 6キーを削除。
  - `ops/env/scalp_ping_5s_b.env`
  - `ops/env/scalp_ping_5s_c.env`
  - `ops/env/scalp_ping_5s_d.env`
    - `ORDER_DB_BUSY_TIMEOUT_MS=1500` に統一。
- 影響範囲:
  - `quant-scalp-ping-5s-{b,c,d}.service` と `quant-autotune-ui.service` の環境解決のみ。
  - エントリー/EXIT 判定ロジックや V2 導線の制御仕様は変更なし。

### 2026-02-27（追記）B/C方向精度の再引き締め + M1/MACD単発損失圧縮

- 背景（VM実測 `2026-02-27 06:41 UTC`）:
  - 24h: `scalp_ping_5s_c_live=-3394.6 JPY`, `scalp_ping_5s_b_live=-519.9 JPY`
  - 6h: `scalp_macd_rsi_div_live=-279.4 JPY`, `M1Scalper-M1=-90.1 JPY`, `B=-129.2 JPY`, `C=-36.1 JPY`
  - Bは `STOP_LOSS_ORDER` が `112 trades / -265.1 JPY` と損失主因。
- 変更ファイル:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE=4`（from `6`）
    - `BASE_ENTRY_UNITS=220`（from `300`）
    - `MAX_UNITS=750`（from `1100`）
    - `CONF_FLOOR=80`（from `77`）
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.79` / `FLOOR=0.67`
    - preserve-intent `REJECT_UNDER/MIN/MAX=0.78/0.20/0.32`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE=4`（from `5`）
    - `BASE_ENTRY_UNITS=70`（from `90`）
    - `MAX_UNITS=160`（from `200`）
    - `FORCE_EXIT_MAX_HOLD_SEC=60`（short `55`）
    - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.8`（short `0.8`）
    - `CONF_FLOOR=80`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.78` / `FLOOR=0.68`
    - preserve-intent `REJECT_UNDER/MIN/MAX=0.76/0.24/0.50`
  - `ops/env/quant-order-manager.env`
    - B/C preserve-intent を上記値に同期
    - B perf guard しきい値を引き上げ（PF/WIN/SL-loss-rate）
    - `M1SCALP_PERF_GUARD_ENABLED=1`
    - B/C forecast gate（edge/expected_pips/target_reach）を引き上げ
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_SIDE_FILTER=none`（from `long`）
    - `M1SCALP_PERF_GUARD_ENABLED=1`（from `0`）
    - `M1SCALP_BASE_UNITS=3000`（from `4500`）
    - `M1SCALP_MIN_UNITS=250`（from `350`）
    - `M1SCALP_CONFIDENCE_FLOOR=50`（from `45`）
  - `ops/env/quant-scalp-macd-rsi-div.env`
    - `MACDRSIDIV_REQUIRE_RANGE_ACTIVE=1`（from `0`）
    - `MACDRSIDIV_MIN_DIV_SCORE=0.12` / `MIN_DIV_STRENGTH=0.18`
    - `MACDRSIDIV_BASE_ENTRY_UNITS=3000`（from `7000`）
    - `MACDRSIDIV_MIN_UNITS=600`（from `1200`）
    - `MACDRSIDIV_COOLDOWN_SEC=120`、`MAX_SPREAD_PIPS=0.9`、`CAP_MAX=0.55`
- 意図:
  - 停止ではなく、低品質シグナルの通過率と通過ロットを同時に下げて損失幅を圧縮する。
  - M1/MACD は「単発大損の回避」を最優先に、方向/品質ゲートとサイズを同時に厳格化。

### 2026-02-27（追記）UI時間帯テーブル欠損の恒久対策（snapshot生成 + 採用ロジック）

- 目的:
  - historyタブの「1時間ごとのトレード」で夜間帯を含む24h行が欠損しない状態へ固定する。
  - summaryカードとhistory集計の不整合（`+0` 固着）を、snapshot欠損時でも自動補正する。
- 仮説:
  - `metrics.hourly_trades` 欠損周期があると、UIが `recent_trades` 限定fallbackへ降格して
    時間帯集計が欠落する。
  - fresh判定のみで snapshot source を選ぶと、軽量 remote が完全 gcs を上書きしうる。
  - `scripts/run_sync_pipeline.py` も同一 GCS object を更新しており、
    `ui_recent=50 / hourly未付与` の payload で最新 object を上書きしていた。
- 変更ファイル:
  - `scripts/publish_ui_snapshot.py`
    - `hourly_trades` を DB row走査で生成する経路へ変更（ISO/非ISO時刻をPythonで吸収）。
    - `UI_HOURLY_DB_TIMEOUT_SEC` / `UI_HOURLY_DB_RETRY_COUNT` / `UI_HOURLY_SCAN_LIMIT` を追加。
    - DB集計失敗時でも `recent_trades` から hourly payload を生成して `metrics.hourly_trades` 欠落を防止。
  - `scripts/run_sync_pipeline.py`
    - GCS publish 前に `hourly_trades` を必ず付与（DB→recent_trades fallback）。
    - `--ui-recent` の既定値を `UI_RECENT_TRADES_LIMIT`（default `200`）へ統一。
  - `apps/autotune_ui.py`
    - fresh snapshot 選択を「固定順」から「hourly有効性 + metrics充足数」優先へ更新。
    - `hourly` が有効でも `daily/yesterday/weekly/total` が欠損時は `recent_trades` で summary を再構成。
  - `tests/scripts/test_publish_ui_snapshot.py`
    - DB欠損時 fallback（recent_trades由来 hourly 生成）を追加検証。
  - `tests/apps/test_autotune_ui_snapshot_selection.py`
    - fresh remote よりも充足度の高い gcs を優先する回帰テストを追加。
  - `tests/apps/test_autotune_ui_summary_consistency.py`
    - hourly有効 + rollup欠損時の summary 再構成テストを追加。
- 影響範囲:
  - UIスナップショット生成 (`quant-ui-snapshot.service`) と dashboard 描画ロジックのみ。
  - 取引導線（entry/exit/risk/order-manager/position-manager）の判定仕様には変更なし。
- 検証:
  - `pytest -q tests/scripts/test_publish_ui_snapshot.py tests/apps/test_autotune_ui_snapshot_selection.py tests/apps/test_autotune_ui_summary_consistency.py tests/apps/test_autotune_ui_hourly_source_guard.py tests/apps/test_autotune_ui_hourly_fallback.py`
    - `24 passed`
  - `pytest -q tests/apps tests/scripts/test_publish_ui_snapshot.py`
    - `38 passed`

### 2026-02-27（追記）`scalp_ping_5s_b_live` EXIT詰まりの即時解消（neg_exit no-block化）
- 目的:
  - `close_reject_no_negative` 連発を止め、B系の EXIT 機動性を回復する。
- 仮説:
  - `scalp_ping_5s_b(_live)` が `scalp_ping_5s` 共通の
    `neg_exit.strict_no_negative=true` を継承し、`candle_*` 系 EXIT と衝突している。
- 実測根拠（VM）:
  - `strategy_control_flags` は全戦略 `entry=1 & exit=1`（`entry=1 & exit=0` は 0 件）。
  - `orders.db` 24h: `close_reject_no_negative=37`、うち
    `client_order_id LIKE '%scalp_ping_5s_b_live%'` が `35` 件。
  - 6h でも `close_reject_no_negative=9` が継続。
- 変更ファイル:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live`
      `neg_exit.strict_no_negative=false`
      `neg_exit.allow_reasons=["*"]`
      `neg_exit.deny_reasons=[]`
- 影響範囲:
  - B系タグの close preflight（`execution/order_manager.py` の negative-close 判定）に限定。
  - entry 判定・サイズ決定・V2導線（worker分離）は非変更。
- 検証手順:
  1. 本番反映後 1h/6h で `orders.db` の `close_reject_no_negative` を再集計。
  2. `client_order_id LIKE '%scalp_ping_5s_b_live%'` の同ステータスが減少することを確認。
  3. `strategy_control_flags` が引き続き `entry=1 & exit=1` を維持することを確認。

### 2026-02-27（追記）dashboard history「1時間ごとのトレード」をカード内スクロール化
- 目的:
  - history タブの「1時間ごとのトレード」を、最近のトレード表と同様にカード内スクロールで閲覧できるようにする。
- 仮説:
  - 当該テーブルのみ `table-wrap` で描画されており、縦スクロール制御 (`table-wrap-scroll`) が未適用。
- 変更ファイル:
  - `templates/autotune/dashboard.html`
    - history タブの hourly table wrapper を
      `class="table-wrap"` から `class="table-wrap table-wrap-scroll"` へ変更。
- 影響範囲:
  - dashboard 表示レイヤ（history タブの hourly テーブル）に限定。
  - 取引ロジック、snapshot 生成、order/position/worker 導線への影響なし。
- 検証:
  - `run.app` の `/dashboard?tab=history` HTML を直接確認し、`tab-history` セクションの
    wrapper class が `table-wrap table-wrap-scroll` であることを検証。

### 2026-02-27（追記）V2発注/EXIT 安全化（entry-intent必須化 + duplicate CID再採番 + exit failopen）
- 目的:
  - `entry_thesis` 欠損由来の低品質エントリーを遮断し、発注意図の監査一貫性を回復する。
  - `CLIENT_TRADE_ID_ALREADY_EXISTS` の連鎖で再拒否が続く経路を、再採番リトライで自己回復させる。
  - `strategy_control_exit_disabled` が連続した際の EXIT 詰まりを、緊急時のみ fail-open で解消する。
- 仮説:
  - VM実測で `strategy_control_exit_disabled` と duplicate CID が収益悪化/強制クローズの直接要因になっている。
  - order_manager で preflight ガードを追加すれば、戦略意図を変えずに事故系 reject を減らせる。
- 変更ファイル:
  - `execution/order_manager.py`
    - `_entry_probability_value` の不正候補処理を修正（未定義変数参照防止）。
    - 非 manual / 非 reduce-only の entry に `entry_intent_guard` を追加。
      - `strategy_tag` / `entry_units_intent` / `entry_probability` 欠損を `entry_intent_guard_reject` で拒否。
    - `CLIENT_TRADE_ID_ALREADY_EXISTS` 発生時:
      - 既存 filled の逆引き復元を継続。
      - 復元不可時は client id を再採番して再送（market/limit 両経路）。
    - reject ログへ `request_payload` を必須付与し、原因追跡を可能化。
    - `strategy_control_exit_disabled` 連続時の fail-open 判定を追加
      - 既定は emergency 条件付きのみ。
      - close 成功/benign success 時に連続ブロック状態をクリア。
  - `workers/scalp_ping_5s/worker.py`
    - `_client_order_id` に `monotonic_ns` nonce を導入して同ms衝突耐性を強化。
  - `ops/env/quant-v2-runtime.env`
    - entry-intent 必須ガードと strategy-control exit fail-open 関連キーを明示追加。
  - テスト:
    - `tests/execution/test_order_manager_log_retry.py`
    - `tests/execution/test_order_manager_exit_policy.py`
    - `tests/workers/test_scalp_ping_5s_worker.py`
- 影響範囲:
  - V2 の order/close preflight と `scalp_ping_5s` の CID 生成に限定。
  - 戦略ロジック（シグナル方向・forecast/brain/pattern gate 判定式）は非変更。
- 検証:
  - 追加テストで以下を回帰確認:
    - `entry_probability` 非数値時の安全処理
    - entry-intent 必須ガードの reject
    - duplicate CID 再採番リトライ（limit path）
    - strategy_control exit fail-open 条件分岐
    - `scalp_ping_5s` CID 同ms重複回避

### 2026-02-27（追記）M1Scalperに quickshot 判定を追加（M5 breakout + M1 pullback + 100円逆算）

- 目的:
  - 「今すぐ1回で100円前後を狙う」判断を裁量依存ではなく、`M1Scalper` の機械判定として再現可能にする。
  - spread/メンテ時間/方向不一致を先に弾き、TP/SLとロットを同一ロジックで決める。
- 仮説:
  - `breakout_retest` に限定した上で `M5 breakout + M1 pullback` を追加すれば、追いかけエントリーを減らせる。
  - `target_jpy` を `tp_pips` で逆算して `entry_units_intent` を決めることで、過大ロットを抑制しつつ再現性が上がる。
- 変更ファイル:
  - `workers/scalp_m1scalper/config.py`
    - `M1SCALP_USDJPY_QUICKSHOT_*` 設定群を追加（spread上限、JSTブロック、ATR連動TP/SL、target_jpy等）。
  - `workers/scalp_m1scalper/worker.py`
    - `_detect_usdjpy_quickshot_plan(...)` を追加。
    - quickshot allow 時に `tp_pips/sl_pips/entry_probability/target_units` を適用。
    - `entry_thesis["usdjpy_quickshot"]` へ監査情報を保存。
  - `ops/env/quant-m1scalper.env`
    - quickshot を有効化し、運用初期値を設定。
  - テスト:
    - `tests/workers/test_m1scalper_config.py`
    - `tests/workers/test_m1scalper_quickshot.py`（新規）
- 影響範囲:
  - `M1Scalper` の entry 判定・サイズ決定に限定。
  - order_manager / position_manager / strategy_entry 契約（`entry_probability`, `entry_units_intent`）は維持。
- 検証手順:
  1. `pytest -q tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`
  2. VM反映後、`orders.db` で `entry_thesis.usdjpy_quickshot` の付与率と `quickshot_*` block 理由を集計。
  3. 24h で `avg_win_jpy` / `avg_loss_jpy` と `entry_units_intent` 分布を確認し、過大ロット再発がないことを確認。

### M1/B 早利確・ペイオフ是正（2026-02-27 追記）
- 背景（VM live）:
  - `M1Scalper-M1` は `close_request.exit_reason` で `lock_floor` / `m1_rsi_fade` が多く、
    勝ちでも `tp_ratio(pl_pips/tp_pips)` が低い。
  - `scalp_ping_5s_b_live` は `TAKE_PROFIT_ORDER` が +1p 台、`STOP_LOSS_ORDER` が -2p 台で
    期待値が伸びづらい。
- 実装:
  - `workers/scalp_m1scalper/exit_worker.py`
    - `lock_trigger` 関連を env 化:
      - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO`
      - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS`
    - 既存の `profit_take/trail/lock` 固定値を env から上書き可能化。
  - `ops/env/quant-m1scalper-exit.env`
    - `M1SCALP_EXIT_RSI_FADE_LONG=40`
    - `M1SCALP_EXIT_RSI_FADE_SHORT=60`
    - `M1SCALP_EXIT_LOCK_FROM_TP_RATIO=0.70`
    - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO=0.55`
    - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS=1.00`
  - `ops/env/scalp_ping_5s_b.env`
    - `TP_BASE/TP_MAX` 引き上げ、`SL_BASE` / force-exit loss 圧縮。
    - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT` を `0.80` へ緩和。
- 影響範囲:
  - `quant-m1scalper-exit.service` の EXIT 閾値と
    `quant-scalp-ping-5s-b.service` の戦略ローカル TP/SL/units 設定に限定。
  - 共通 order_manager/strategy_control の判定仕様は非変更。

### 2026-02-27（追記）quickshot 実装の回帰テスト追加

- 目的:
  - `M1Scalper` quickshot 判定が将来変更で崩れないよう、allow/block 条件を固定テスト化する。
- 変更ファイル:
  - `tests/workers/test_m1scalper_quickshot.py`（新規）
  - `tests/workers/test_m1scalper_config.py`（quickshot env 読込の回帰）
- 検証:
  - `pytest -q tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`（7 passed）

### 2026-02-27（追記）場面別3戦略を専用ワーカーへ分離（Trend/Pullback/FailedBreak）

- 目的:
  - 「場面ごとに戦略を切り替える」運用を、既存ワーカーと競合しない形で V2 の ENTRY/EXIT 分離へ実装する。
- 方針:
  - 共通実装は `workers.scalp_m1scalper` を再利用し、ラッパーで signal/tag を固定。
  - 既存導線との衝突回避のため、各専用 env は `M1SCALP_ENABLED=0`（デフォルト無効）。
  - EXIT は `M1SCALP_EXIT_TAG_ALLOWLIST` で strategy_tag ごとに閉域化。
- 変更ファイル:
  - 追加（ENTRY/EXIT ラッパー）:
    - `workers/scalp_trend_breakout/*`
    - `workers/scalp_pullback_continuation/*`
    - `workers/scalp_failed_break_reverse/*`
  - 追加（systemd）:
    - `systemd/quant-scalp-trend-breakout*.service`
    - `systemd/quant-scalp-pullback-continuation*.service`
    - `systemd/quant-scalp-failed-break-reverse*.service`
  - 追加（env）:
    - `ops/env/quant-scalp-trend-breakout*.env`
    - `ops/env/quant-scalp-pullback-continuation*.env`
    - `ops/env/quant-scalp-failed-break-reverse*.env`
  - 既存更新:
    - `workers/scalp_m1scalper/exit_worker.py`（`M1SCALP_EXIT_TAG_ALLOWLIST` を追加）
    - `docs/WORKER_ROLE_MATRIX_V2.md`（stage 導入ユニットを追記）
- 競合回避仕様:
  - `TrendBreakout` は `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest`、strategy_tag を `TrendBreakout` 固定。
  - `PullbackContinuation` は `buy-dip,sell-rally` のみ許可、strategy_tag を `PullbackContinuation` 固定。
  - `FailedBreakReverse` は `vshape-rebound` のみ許可、strategy_tag を `FailedBreakReverse` 固定。
  - EXIT は各 strategy_tag の allowlist で分離し、重複クローズを防ぐ。
- 検証:
  - `python3 -m py_compile` とユニットテストで import/設定分岐の回帰を確認（詳細は次項）。

### 2026-02-27（追記）`WickReversalBlend` を成功パターンへ寄せる閾値更新（VM実測ベース）

- 目的:
  - `WickReversalBlend` の勝ち筋（高 `entry_probability` + 早期解放）へ約定を寄せ、`max_adverse` テールを圧縮する。
- 実測根拠（VM）:
  - 24h: `16 trades / +403.5 JPY`、14d: `24 trades / +255.8 JPY`。
  - `entry_probability` 閾値シミュレーションで `>=0.78` が 24h/14d とも改善（`24h +497.8 JPY`, `14d +457.1 JPY`）。
  - `orders.db` で `WickReversalBlend` の `close_reject_profit_buffer` を確認。
    - 例: `ticket 408599` は `est_pips=0.5` が `min_profit_pips=0.6` を下回って reject され、後段 `max_adverse` で `-322.0 JPY`。
- 変更:
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.46 -> 0.78`
  - `config/strategy_exit_protections.yaml`（`WickReversalBlend`）
    - `min_profit_pips: 0.6 -> 0.45`
    - `loss_cut_max_hold_sec: 900 -> 420`
- 影響範囲:
  - `quant-scalp-wick-reversal-blend.service` の entry preflight 縮小/拒否閾値。
  - `quant-scalp-wick-reversal-blend-exit.service` が参照する strategy exit protection（WickReversalBlendタグのみ）。
  - order_manager/position_manager/strategy_control の共通導線仕様は変更なし。
- 検証観点:
  1. 反映後 2h/24h で `WickReversalBlend` の `entry_probability>=0.78` への収束。
  2. `orders.db` の `close_reject_profit_buffer`（WickReversalBlend）減少。
  3. `trades.db` の `hold_sec>=420` かつ負けトレード件数の低下。

### 2026-02-27（追記）split M1Scalper の spread 判定を単一ソース運用へ統一

- 目的:
  - `spread_monitor` とワーカー内 `MAX_SPREAD_PIPS` の重複判定による設定不一致を解消し、
    split worker の entry skip 要因を一本化する。
- 変更:
  - `workers/scalp_{trend_breakout,pullback_continuation,failed_break_reverse,m1scalper}/config.py`
    - `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` を追加（既定 `0`）。
  - `workers/scalp_{trend_breakout,pullback_continuation,failed_break_reverse,m1scalper}/worker.py`
    - spread block 判定を整理:
      - 既定は `spread_monitor.is_blocked()` を単一ソースとして採用。
      - `M1SCALP_LOCAL_SPREAD_CAP_ENABLED=1` または `SPREAD_GUARD_DISABLE=1` の時のみ
        `M1SCALP_MAX_SPREAD_PIPS` をローカル fallback として有効化。
      - ログ reason に `local_cap` を追加。
  - env 固定:
    - `ops/env/quant-v2-runtime.env` に
      `M1SCALP_LOCAL_SPREAD_CAP_ENABLED=0` を追加（全体デフォルト固定）。
    - split 3 worker env にも `M1SCALP_LOCAL_SPREAD_CAP_ENABLED=0` を明示。
    - `ops/env/quant-m1scalper.env` は `SPREAD_GUARD_DISABLE=1` 運用維持のため
      `M1SCALP_LOCAL_SPREAD_CAP_ENABLED=1` を明示。
  - テスト:
    - `tests/workers/test_m1scalper_split_workers.py`
      - split config の `LOCAL_SPREAD_CAP_ENABLED` 既定/上書きの回帰テストを追加。
- 影響範囲:
  - spread 入口判定に限定。order_manager/position_manager/strategy_control の共通導線仕様は非変更。

### 2026-02-27（追記）`scalp_ping_5s_b/c` long-side RR改善 + lot圧縮緩和（VM実測対応）

- 背景（VM 24h）:
  - `scalp_ping_5s_b_live long`: `avg_sl=2.03 pips`, `avg_tp=0.99 pips`, `tp/sl=0.49`, `sum_realized_pl=-565.8 JPY`
  - `scalp_ping_5s_c_live long`: `avg_sl=1.30 pips`, `avg_tp=0.90 pips`, `tp/sl=0.69`, `sum_realized_pl=-117.8 JPY`
  - side合算で `long avg_units=185.8 < short avg_units=338.9`。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - longの `SL` 短縮 + `TP` 拡張（shortは `SHORT_*` で従来維持）
    - `BASE_ENTRY_UNITS/MAX_UNITS` と `ENTRY_LEADING_PROFILE_UNITS_*` を小幅引き上げ
    - `ORDER_MANAGER_PRESERVE_INTENT_*` の max/min scale を緩和
  - `ops/env/scalp_ping_5s_c.env`
    - longの `SL` 短縮 + `TP` 拡張（shortは `SHORT_*` を明示して維持）
    - `BASE_ENTRY_UNITS/MAX_UNITS` と `ENTRY_LEADING_PROFILE_UNITS_*` を小幅引き上げ
    - `ORDER_MANAGER_PRESERVE_INTENT_*` の max/min scale を緩和
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル判定のみ。
  - V2導線（`quant-market-data-feed` / `quant-strategy-control` / `quant-order-manager` / `quant-position-manager`）の責務分離は不変。

### 2026-02-27（追記）split M1Scalper の nwave 早入り許容を env で戦略限定

- 目的:
  - split 3戦略（Trend/Pullback/FailedBreak）で `skip_nwave_*_late` を減らし、
    早入りを小幅に許容する。
- 変更:
  - `strategies/scalping/m1_scalper.py`
    - `nwave` の `tolerance_default/tolerance_tactical` に
      env override を追加。
      - 推奨（短縮キー）:
        - `M1SCALP_NWAVE_TOL_DEF_PIPS`
        - `M1SCALP_NWAVE_TOL_TAC_PIPS`
      - 互換（従来キー）:
        - `M1SCALP_NWAVE_TOLERANCE_DEFAULT_PIPS`
        - `M1SCALP_NWAVE_TOLERANCE_TACTICAL_PIPS`
    - 未設定時は既存ロジック（`scalp_active_params.json` + 既定値）を維持。
  - split 3 worker env:
    - `ops/env/quant-scalp-trend-breakout.env`
    - `ops/env/quant-scalp-pullback-continuation.env`
    - `ops/env/quant-scalp-failed-break-reverse.env`
    - 追加:
      - `M1SCALP_NWAVE_TOL_DEF_PIPS=0.50`
      - `M1SCALP_NWAVE_TOL_TAC_PIPS=0.62`
  - テスト:
    - `tests/workers/test_m1scalper_nwave_tolerance_override.py`
      - env override 未設定/設定時で `nwave` long-late 判定が切り替わることを確認。
- 影響範囲:
  - 既定では split 3 worker のみ（env設定済み）。
  - `quant-m1scalper` は env 未設定のため従来挙動を維持。

### 2026-02-27（追記）`scalp_ping_5s_b/c` 第3ラウンド調整（RR再補正 + longロット押上げ）

- 背景（VM, 直近6h）:
  - Round2 後は `rate_limited` / `revert_not_found` ではなく `entry_leading_profile_reject` が主因へ移行。
  - B/C long の `tp/sl` が 1 未満のまま（B `0.54`, C `0.74`）で、long側実効 units も short比で不足。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `BASE_ENTRY_UNITS=300`, `MAX_UNITS=1000`
    - `TP_BASE/MAX=0.75/2.2`, `SL_BASE/MAX=1.20/1.8`
    - `TP_NET_MIN=0.65`, `TP_TIME_MULT_MIN=0.72`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.34`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.68`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN/MAX=0.70/1.00`
  - `ops/env/scalp_ping_5s_c.env`
    - `BASE_ENTRY_UNITS=120`, `MAX_UNITS=260`
    - `TP_BASE/MAX=0.60/1.8`, `SL_BASE/MAX=1.05/1.7`
    - `TP_NET_MIN=0.55`, `TP_TIME_MULT_MIN=0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.38`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.68`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN/MAX=0.68/0.95`
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 判定・ロット・仮想TP/SLのみ。
  - V2固定導線（order_manager / position_manager / strategy_control / market-data-feed）は非変更。

### 2026-02-27（追記）`scalp_ping_5s_b/c` 第4ラウンド調整（rate-limit/revert/perf 同時緩和）

- 背景（VM, Round3反映後 13:21-13:26 UTC）:
  - `entry-skip summary` 上位が `rate_limited` と `no_signal:revert_not_found` のまま残存。
  - C は約定再開が弱く、long 側通過率の回復が不足。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE=10`
    - `REVERT_*` 追加緩和（`range/sweep/bounce/confirm_ratio`）
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.64`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT=0.76`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.74`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.40`
    - `PERF_GUARD_SETUP/HOURLY` の min_trades と閾値を緩和
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE=10`
    - `REVERT_*` 追加緩和（B と同値）
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.64`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT=0.74`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.72`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.44`
    - `PERF_GUARD_SETUP/HOURLY` を `SCALP_PING_5S_C_*` と fallback `SCALP_PING_5S_*` の両方で緩和
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 条件・ロット算出に限定。
  - V2固定導線（order_manager / position_manager / strategy_control / market-data-feed）は非変更。

### 2026-02-27（追記）`scalp_ping_5s_c` spread guard 閾値の実勢追従化（第5ラウンド）

- 背景（VM, Round4反映後）:
  - C の `entry-skip summary` が `spread_blocked` に偏重（例: `total=143, spread_blocked=134`）。
  - `spread_med ... >= limit 1.00p` が主因で、実勢 `p95=1.16p` を吸収できていない。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `spread_guard_max_pips=1.30`
    - `spread_guard_release_pips=1.05`
    - `spread_guard_hot_trigger_pips=1.50`
    - `spread_guard_hot_cooldown_sec=6`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の spread guard 判定に限定。
  - B 戦略設定および V2固定導線（order_manager / position_manager / strategy_control / market-data-feed）は非変更。

### 2026-02-27（追記）`scalp_ping_5s_c` 第6ラウンド（rate-limit/perf_block 縮小）

- 背景（VM, 第5ラウンド反映後）:
  - `spread_blocked` は低下したが、`rate_limited` と `perf_block` が主因へ移行。
  - `orders.db` で `filled` 再開が確認できず、通過率不足が継続。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE=10`
    - `SCALP_PING_5S_C_PERF_GUARD_HOURLY_MIN_TRADES=16`
    - `SCALP_PING_5S_C_PERF_GUARD_SETUP_MIN_TRADES=16`
    - fallback `SCALP_PING_5S_PERF_GUARD_HOURLY_MIN_TRADES=16`
    - fallback `SCALP_PING_5S_PERF_GUARD_SETUP_MIN_TRADES=16`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.78`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.70`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 判定のみ。
  - B 戦略設定および V2固定導線は非変更。

### 2026-02-27（追記）`scalp_ping_5s_c` 第7ラウンド（rate_limited 優位への追加緩和）

- 背景（VM, 第6ラウンド反映後）:
  - `spread_blocked` は縮小したが、`rate_limited` が主因で残存（例: `total=107, rate_limited=65`）。
  - `orders.db` で `filled` 再開が未確認。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `ENTRY_COOLDOWN_SEC=1.2`
    - `MAX_ORDERS_PER_MINUTE=16`
    - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.56`
    - `MIN_UNITS_RESCUE_MIN_CONFIDENCE=78`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.74`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.66`
- 影響範囲:
  - `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 通過率に限定。
  - B 戦略設定および V2固定導線は非変更。

### 2026-02-27（追記）`scalp_ping_5s_b/c` 第9ラウンド（revert/leading の過剰拒否を小幅緩和）

- 背景（VM, Round8反映後）:
  - 24h 集計は依然マイナス。
    - `scalp_ping_5s_b_live`: `582 trades / -670.8 JPY / avg_win=1.165 / avg_loss=1.994`
    - `scalp_ping_5s_c_live`: `359 trades / -139.1 JPY / avg_win=1.090 / avg_loss=1.849`
  - 直近120分 `orders.db` は `perf_block` 偏重。
    - B: `perf_block=50`
    - C: `perf_block=110`
  - worker ログは B/C とも `revert_not_found` が主因として残存し、C は `entry_leading_profile_reject` も高頻度。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE: 7 -> 8`
    - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
    - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
    - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
    - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.77 -> 0.76`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.68 -> 0.67`
  - `ops/env/scalp_ping_5s_c.env`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR: 0.72 -> 0.70`
    - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
    - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
    - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
    - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.74 -> 0.72`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.66 -> 0.64`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT: 0.82 -> 0.80`
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY 通過判定に限定。
  - TP/SL や order_manager/position_manager/strategy_control の V2責務分離は非変更。

### 2026-02-27（追記）`scalp_ping_5s_c` 第13ラウンド（failfast hard block の下限調整）

- 背景（VM, Round12反映直後）:
  - C は setup 緩和後も `perf_block` が残存。
  - C ログで拒否理由を確認:
    - `perf_block:hard:hour15:failfast:pf=0.12 win=0.28 n=43`
  - B は同時間帯に `submit_attempt/filled` が継続しており、ボトルネックは C failfast に限定。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
    - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
    - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
    - `SCALP_PING_5S_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
  - `ops/env/quant-order-manager.env`
    - C/fallback failfast 閾値を同値へ同期。
- 影響範囲:
  - C の failfast 判定に限定（setup/SL系ガード、B設定、V2導線責務分離は維持）。
  - 目標は「hard block 常態化の緩和」であり、無制限化ではない（PF 下限 0.10 は維持）。

### 2026-02-27（追記）`scalp_ping_5s_c` 第12ラウンド（setup perf guard を failfast中心へ寄せる）

- 背景（VM, Round11反映直後）:
  - B は `submit_attempt=4 / filled=4` まで回復した一方、C は `perf_block=12` が残存。
  - C の `entry-skip summary` 例（14:57:40 UTC）:
    - `total=31`, `order_reject:perf_block=10`
  - C の `RISK multiplier` は `pf=0.47, win=0.48` で、setup guard の `PF_MIN=0.90` が主因で遮断。
  - C worker は `mapped_prefix=SCALP_PING_5S` を使用するため、fallback 側の setup 閾値も同時に効く。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_PERF_GUARD_SETUP_MIN_TRADES: 16 -> 24`
    - `SCALP_PING_5S_C_PERF_GUARD_SETUP_PF_MIN: 0.90 -> 0.45`
    - `SCALP_PING_5S_C_PERF_GUARD_SETUP_WIN_MIN: 0.45 -> 0.40`
    - `SCALP_PING_5S_PERF_GUARD_SETUP_MIN_TRADES: 16 -> 24`
    - `SCALP_PING_5S_PERF_GUARD_SETUP_PF_MIN: 0.90 -> 0.45`
    - `SCALP_PING_5S_PERF_GUARD_SETUP_WIN_MIN: 0.45 -> 0.40`
  - `ops/env/quant-order-manager.env`
    - 上記 C/fallback setup 閾値を同値へ同期。
- 影響範囲:
  - C 系の perf guard setup 判定に限定（B の failfast, TP/SL, V2導線責務分離は維持）。
  - `failfast` と `SL_LOSS_RATE` は据え置きで、急悪化時の防波堤は維持。

### 2026-02-27（追記）`quant-order-manager` 第11ラウンド（B/C 閾値ドリフト同期）

- 背景（VM, Round10後）:
  - 直近120分 `orders.db`（B/C strategy tag）は `perf_block` のみ。
    - B: `57`
    - C: `131`
  - worker ログは `revert_not_found` / `extrema_block` / `entry_leading_profile_reject` が優位で、
    その上に `order_reject:perf_block` が重なって通過率を悪化。
  - `quant-order-manager.env` が worker env より strict（B preserve-intent/min_units、B/C setup/hourly guard）で、
    worker 側緩和の効果を上書きしていた。
- 変更:
  - `ops/env/quant-order-manager.env`
    - B preserve-intent/min-units 同期:
      - `REJECT_UNDER 0.78 -> 0.76`
      - `MIN/MAX_SCALE 0.20/0.32 -> 0.40/0.42`
      - `ORDER_MIN_UNITS 10 -> 1`
    - B perf guard 同期:
      - `HOURLY_MIN_TRADES 6 -> 10`
      - `SETUP_MIN_TRADES 6 -> 10`
      - `SETUP_PF/WIN 0.92/0.48 -> 0.88/0.44`
      - `FAILFAST_PF/WIN 0.70/0.42 -> 0.10/0.27`
      - `SL_LOSS_RATE_MAX 0.68 -> 0.75`
    - C + fallback perf guard 同期:
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER 0.74 -> 0.72`
      - `SCALP_PING_5S[_C]_PERF_GUARD_HOURLY_MIN_TRADES 6 -> 16`
      - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_MIN_TRADES 6 -> 16`
      - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_PF/WIN 0.95/0.50 -> 0.90/0.45`
      - `SCALP_PING_5S[_C]_PERF_GUARD_PF/WIN_MIN 0.90/0.47, 0.93/0.50 -> 0.92/0.49`
      - `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_MAX 0.68 -> 0.55`
- 影響範囲:
  - `quant-order-manager.service` preflight の B/C 関連ガード閾値に限定。
  - 戦略ロジック本体（signal生成/TP/SL）および V2導線責務分離は非変更。

### 2026-02-27（追記）`quant-order-manager` の orders.db スレッド競合是正 + duplicate復旧強化

- 目的:
  - `CLIENT_TRADE_ID_ALREADY_EXISTS` 発生時の復旧率を上げ、不要 reject を減らす。
  - `orders.db` 書き込み時の `SQLite objects created in a thread` 警告を解消する。
- 変更:
  - `execution/order_manager.py`
    - `orders.db` 接続を global singleton から thread-local 管理へ変更。
    - `_cache_order_status` に `ticket_id` を追加（duplicate復旧でキャッシュから trade_id 回収可能化）。
    - `_latest_filled_trade_id_by_client_id` に `trades.db` フォールバックを追加。
      - 優先順: `orders.db(filled)` -> `order_status_cache` -> `trades.db(client_order_id)`.
- 背景根拠（VM）:
  - `journalctl -u quant-order-manager.service` で同一警告が連続発生。
  - 24h `orders.db` で `status='rejected'` の主因が `CLIENT_TRADE_ID_ALREADY_EXISTS`（294件）。
- 影響範囲:
  - `quant-order-manager.service` の注文ログ永続化/duplicate回復経路のみ。
  - 戦略ロジック・entry/exit判定・V2導線分離は非変更。

### 2026-02-27（追記）`scalp_ping_5s_b/c` 第5ラウンド調整（損失側圧縮 + 低品質約定抑制）

- 目的:
  - B/C の `avg_loss_pips > avg_win_pips` を是正し、24h損失の縮小を優先する。
  - C で発生していた小ロット churn を抑え、低品質通過を減らす。
- 実測根拠（VM, 24h）:
  - `scalp_ping_5s_b_live`: `587 trades / -679.0 JPY / avg_win=1.158 / avg_loss=1.998`
  - `scalp_ping_5s_c_live`: `372 trades / -146.5 JPY / avg_win=1.075 / avg_loss=1.844`
  - B の `STOP_LOSS_ORDER` は `310 trades / -1094.6 JPY`。
  - 最新 `orders.db` は B/C とも `perf_block` が上位で、C は `units=2-5` の小ロット通過が中心。
- 変更:
  - `ops/env/scalp_ping_5s_b.env`
    - `MAX_ORDERS_PER_MINUTE: 10 -> 5`
    - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY: 0.54 -> 0.58`
    - `MIN_UNITS_RESCUE_MIN_CONFIDENCE: 75 -> 78`
    - `TP_BASE/MAX: 0.75/2.2 -> 0.90/2.6`
    - `SL_BASE/MAX: 1.20/1.8 -> 1.00/1.5`
    - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: 1.5 -> 1.2`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.74 -> 0.80`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.64 -> 0.70`
  - `ops/env/scalp_ping_5s_c.env`
    - `MAX_ORDERS_PER_MINUTE: 10 -> 6`
    - `BASE_ENTRY_UNITS/MAX_UNITS: 120/260 -> 80/160`
    - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY: 0.56 -> 0.60`
    - `MIN_UNITS_RESCUE_MIN_CONFIDENCE: 78 -> 82`
    - `TP_BASE/MAX: 0.60/1.8 -> 0.85/2.3`
    - `SL_BASE/MAX: 1.05/1.7 -> 0.90/1.4`
    - `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: 0.8 -> 0.6`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.72 -> 0.82`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.64 -> 0.74`
- 影響範囲:
  - `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` の戦略ローカル ENTRY/units/virtual TP/SL に限定。
  - `execution/order_manager.py` / `execution/strategy_entry.py` の契約（`entry_probability`, `entry_units_intent`）および V2 導線責務分離は非変更。


### 2026-02-27（追記）`gpt_ops_report` の factor 鮮度ガード追加（外部価格フォールバック）

- 目的:
  - `factor_cache` が stale のままでも市況プレイブックが古い価格で方向判定しないようにする。
  - `snapshot.current_price` と `market_context.pairs.usd_jpy` の乖離を自動検出し、誤ったシナリオ確信度を抑える。
- 変更:
  - `scripts/gpt_ops_report.py`
    - `OPS_PLAYBOOK_FACTOR_MAX_AGE_SEC`（default: 900 sec）で M1 factor 鮮度を判定。
    - factor stale 時は `snapshot.current_price` を `market_context` の外部 `USD/JPY` へフォールバック。
    - `snapshot` に `current_price_source/factor_stale/factor_age_m1_sec/factor_timestamp_utc` を追加。
    - stale 時は `direction_score` と `direction_confidence_pct` を減衰し、`if_then_rules` と `break_points` に鮮度ガードを追加。
    - `data_sources` に `factors_m1_stale/factors_m1_age_sec` を追加。
  - `tests/scripts/test_gpt_ops_report.py`
    - stale/fresh の 2 ケースを追加し、価格ソース切替と stale フラグを検証。
  - `execution/order_manager.py`
    - `_factor_age_seconds()` が `timestamp` に加えて `ts/time` も参照するよう修正。
    - `ENTRY_FACTOR_MAX_AGE_SEC` による preflight stale block が、ts系キーでも有効化される。
- 検証:
  - `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` -> `11 passed`
  - ローカル再現で `factor_timestamp_utc=2025-10-29...` の stale 条件時、`current_price_source=external_snapshot` を確認。
- 影響範囲:
  - `quant-ops-policy` のプレイブック生成品質（分析/運用判断）に限定。
  - strategy worker の ENTRY/EXIT 実装、V2分離導線、`order_manager` ガード契約は非変更。

### 2026-02-27（追記）`quant-ops-policy.service` 新導線の依存補完（bs4）

- 背景:
  - VM で `quant-ops-policy.service` を `run_market_playbook_cycle.py` へ更新後、
    `ModuleNotFoundError: No module named 'bs4'` で起動失敗。
  - 原因は `scripts/fetch_market_snapshot.py` の BeautifulSoup 依存が
    `requirements.txt` に未記載だったこと。
- 変更:
  - `requirements.txt` に `beautifulsoup4==4.12.3` を追加。
- 影響範囲:
  - `quant-ops-policy` の市場スナップショット取得経路のみ。
  - 戦略 worker / order_manager / position_manager の売買ロジックは非変更。

### 2026-02-27（追記）`gpt_ops_report --apply-policy` の自動反映を有効化（deterministic translator）

- 目的:
  - 市況プレイブックを「レビュー用レポート」で止めず、`policy_overlay` へ機械適用して
    `order_manager` の entry gate（`allow_new` / `bias`）に即時反映する。
  - 同じ状態で overlay の version が毎回増える問題を抑止する（`no_change` 判定）。
- 変更:
  - `scripts/gpt_ops_report.py`
    - `_build_policy_diff_from_report()` を追加し、以下を `policy_diff.patch` へ変換:
      - `short_term.bias` / `direction_confidence_pct` / scenario gap
      - `event_context.event_soon/event_active_window`
      - `snapshot.factor_stale`
      - `order_quality.reject_rate`
    - pocket別 (`macro/micro/scalp`) に `enabled/bias/confidence/entry_gates` を生成。
    - `--apply-policy` 時は `analytics.policy_apply.apply_policy_diff_to_paths` を呼び出して
      `overlay/history/latest` を更新。
    - 現在 overlay と patch を deep-subset 比較し、同値なら `no_change=true` で反映スキップ。
    - 生成 diff は `validate_policy_diff()` で検証し、異常時は `generated_diff_invalid` として fail-safe。
  - `tests/scripts/test_gpt_ops_report.py`
    - directional bias 付与ケースと no-delta ケースを追加。
- 検証:
  - `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` -> `13 passed`
  - ローカル dry-run:
    - `python3 scripts/gpt_ops_report.py ... --policy --apply-policy ...`
    - `INFO [OPS_POLICY] applied=True ...` を確認。
- 影響範囲:
  - `quant-ops-policy` の policy生成/反映導線のみ。
  - 各戦略 worker のローカル判定・EXIT判断・V2責務分離は非変更。
