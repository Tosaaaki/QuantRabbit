# ワーカー再編の確定ログ（2026-02-13）

## 方針（最終確定）

- 各戦略は `ENTRY/EXIT` を1対1で持つ。
- `precision` 系のサービス名は廃止し、サービス名は `quant-scalp-*` へ切り分ける。
- `quant-hard-stop-backfill` / `quant-realtime-metrics` は削除対象。
- データ供給は `quant-market-data-feed`、制御配信は `quant-strategy-control` に分離。
- 補助的運用ワーカーは本体管理マップから除外。

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
