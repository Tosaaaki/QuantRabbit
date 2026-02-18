# Forecast (scikit-learn)

このリポジトリには、USD/JPY のローソク足から「方向の確率」を推定するオフライン予測（scikit-learn）があります。

用途は、点予測ではなく「エントリーをブロックするか」「ロットを縮小するか」の判断材料です。

## 構成
- 予測モデル: `analysis/forecast_sklearn.py`
- 学習スクリプト: `scripts/train_forecast_bundle.py`
- エントリーゲート: `workers/common/forecast_gate.py`（既定で有効）
- 入力ローソク: ランタイムは `indicators/factor_cache.py`（M5/H1/D1）、学習は `logs/replay/...` を主に利用

`forecast_gate` は以下の順に予測を使います。
- bundle がある場合: sklearn 予測
- bundle がない場合: テクニカル指標ベースの deterministic 予測（M5/H1/D1）
- `FORECAST_GATE_SOURCE=auto` の場合: bundle と technical をブレンド
- 戦略スタイル（trend/range）と予測コンテキスト（trend_strength/range_pressure）が不一致ならブロック

## 予測特徴量（2026-02-17 更新）
- 回帰系: `ret_pips_*`, `ma_gap_pips_10_20`, `close_ma*`, `atr_pips_14`, `vol_pips_20`, `rsi_14`
- 線形トレンド系（線を引く系）: `trend_slope_pips_20`, `trend_slope_pips_50`, `trend_accel_pips`
- サポレジ/ブレイク系: `support_gap_pips_20`, `resistance_gap_pips_20`, `sr_balance_20`,
  `breakout_up_pips_20`, `breakout_down_pips_20`, `donchian_width_pips_20`, `range_compression_20`
- 予測行には監査用に `breakout_bias_20` / `squeeze_score_20` も出力され、`vm_forecast_snapshot.py` で確認可能
- 分位レンジ（上下帯）として `range_low_pips` / `range_high_pips` / `range_sigma_pips` と
  `range_low_price` / `range_high_price` を出力し、`q10_pips` / `q50_pips` / `q90_pips` も監査可能

## 学習に必要な足の履歴
`scripts/train_forecast_bundle.py` の既定は以下です（概ね「学習や予測が成立する」目安）。
- M5: 120 日
- H1: 540 日
- D1: 2000 日

不足すると `scripts/train_forecast_bundle.py` が `missing_replay_candles` または `insufficient_samples` で失敗します。

## 足の履歴を backfill する
`logs/replay/<instrument>/...jsonl` は通常ランタイムが追記していきますが、過去分は OANDA REST から backfill できます。

コマンド（既定値で M5/H1/D1 を backfill）:
```bash
python scripts/backfill_replay_candles.py --instrument USD_JPY --timeframes M5,H1,D1
```

書き込み先:
- `logs/replay/USD_JPY/USD_JPY_M5_YYYYMMDD.jsonl`
- `logs/replay/USD_JPY/USD_JPY_H1_YYYYMMDD.jsonl`
- `logs/replay/USD_JPY/USD_JPY_D1_YYYYMMDD.jsonl`

dry-run（書き込まずに取得件数だけ確認）:
```bash
python scripts/backfill_replay_candles.py --instrument USD_JPY --timeframes M5,H1,D1 --dry-run
```

## 学習（bundle 作成）
```bash
python scripts/train_forecast_bundle.py --instrument USD_JPY --out config/forecast_models/USD_JPY_bundle.joblib
```

## 運用でゲートを有効化
既定で有効ですが、主要パラメータは以下です（例は `config/env.example.toml` 参照）。
- `FORECAST_BUNDLE_PATH=config/forecast_models/USD_JPY_bundle.joblib`
- `FORECAST_GATE_SOURCE=auto|bundle|technical`
- `FORECAST_TECH_ENABLED=1`
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0`（新特徴量の寄与ゲイン。`0.0` で無効、`0.0-1.0` で段階適用）
- `FORECAST_RANGE_BAND_LOWER_Q=0.20`（予測帯の下限分位）
- `FORECAST_RANGE_BAND_UPPER_Q=0.80`（予測帯の上限分位）
- `FORECAST_RANGE_SIGMA_FLOOR_PIPS=0.35`（予測帯の最小分散）
- `FORECAST_GATE_HORIZON_SCALP_FAST=1m`（scalp_fast 向け）
- `FORECAST_GATE_HORIZON_SCALP=5m`（scalp 向け）
- `FORECAST_GATE_TECH_PREFERRED_HORIZONS=1m,5m,10m`（`auto` 時に短期3軸をテック優先で利用）
- `FORECAST_GATE_STYLE_GUARD_ENABLED=1`
- `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH=0.52`
- `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE=0.52`
- 戦略別 override:
  - `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH_STRATEGY_TRENDMA=...`
  - `FORECAST_GATE_EDGE_BLOCK_TREND_STRATEGY_TRENDMA=...`
  - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_BBRSI=...`
  - `FORECAST_GATE_EDGE_BLOCK_RANGE_STRATEGY_BBRSI=...`

## VMで最新予測を直接確認

実行中VM上で `forecast_gate` の最新推定を直接確認するには、次を実行します（環境ファイルを読み込んで、`PRED_GATE` に準拠した値で表示します）。

```bash
python3 scripts/vm_forecast_snapshot.py \
  --env-file /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env
```

短期予測を短絡的に確認する際は、必要 horizon を明示できます。  
`5m/10m` が未生成でも pending 行（原因 + 取得推奨）が表示されます。

```bash
python3 scripts/vm_forecast_snapshot.py \
  --env-file /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env \
  --horizon 5m \
  --horizon 10m
```

出力例:
 - `p_up` が 0.55 超: 上振れ寄り
 - `p_up` が 0.45 未満: 下振れ寄り
 - `trend_strength` と `range_pressure` が分裂し `天井警戒` / `底警戒` が付いた場合は、逆方向の追随が弱めと解釈して慎重判定
 - `range_pips=[low,high]` と `range_price=[low,high]` は「予測上下帯」。実値が帯内に収まるかを同一期間で監査

`order_manager` は予測判定に以下を渡すよう統一されています。
- `expected_pips`: 期待進行（pip）
- `anchor_price`: 直近クローズ基準価格
- `target_price`: 到達想定価格（`anchor_price + expected_pips * pip`）
- `range_low_pips` / `range_high_pips`: 分位レンジの上下帯（pip）
- `range_low_price` / `range_high_price`: 分位レンジの上下帯（price）
- `range_sigma_pips`: 帯幅の内部推定分散（pip）
- `tf_confluence_score`: 補助TF整合スコア（-1.0〜+1.0）
- `tf_confluence_count`: 実際に参照できた補助TF本数
- `tf_confluence_horizons`: 参照した補助TF（例: `5m,10m`）
- `breakout_skill_20`: `breakout_bias_20` の直近方向一致スキル（-1.0〜+1.0）
- `breakout_hit_rate_20`: 直近サンプルでの `breakout_bias_20` 一致率
- `breakout_samples_20`: スキル推定に使った直近サンプル数
- `tp_pips_hint`: TP 方向ヒント（pips）
- `target_reach_prob`: 現在ポジ方向で `tp_pips_hint` 到達を見込む確率（0.0-1.0）
- `sl_pips_cap`: SL 上限ヒント（pips）
- `rr_floor`: TP/SL の下限 R:R 値

`order_manager` が通過判定を通過した注文は、`entry_thesis["forecast"]` に `future_flow` を残します。
`future_flow` は `horizon:style:state:strength` 形式の文字列で、実運用の可視化（ログ・監査）で「今後の流れ」を素早く拾えます。
例: `5m:trend:上昇トレンド継続:強い`

`entry_thesis["forecast"]` は監査経路として `entry_intent_board` の `details.forecast_context` へも反映され、同一決定の意図と合わせてトレードログに残せます。

JSONで回収する場合:
```bash
python3 scripts/vm_forecast_snapshot.py --json
```

## before/after 比較ジョブ（同一期間）
短期予測式の変更を入れたときは、同一データで `before/after` の hit率・MAE を比較します。

```bash
python3 scripts/eval_forecast_before_after.py \
  --patterns logs/candles_M1*.json,logs/candles_USDJPY_M1*.json,logs/oanda/candles_M1_latest.json \
  --steps 1,5,10 \
  --feature-expansion-gain 0.35
```

`breakout_bias_20` の方向一致率（filtered/unfiltered）も同時に出るため、
「線形トレンド＋サポレジ圧力」の有効性を同一期間で監査できます。

加えて `range_cov_before/after`（帯内包含率）と `range_width_before/after`（平均帯幅）を出力するため、
方向一致率だけでなく「予測帯が実値をどれだけ覆えているか」も before/after で比較できます。

`after` 側は `breakout_bias_20` の符号を固定解釈せず、過去サンプルだけで推定した方向スキル
（`--breakout-adaptive-*`）で強弱/反転するため、レジーム変化時の逆噴射を抑える設計です。
- TF別重みは `breakout_adaptive_weight_map`（例: `1m=0.16,5m=0.22,10m=0.30`）で上書きできます。
  現在の既定は `1m=0.16,5m=0.22,10m=0.30` です。
- 同一固定データ（`bars=8050`, `2026-01-06T07:30:00+00:00`〜`2026-02-17T15:47:00+00:00`）では、
  旧既定 `1m=0.22,5m=0.22,10m=0.22` 比で新既定は
  `1m hit 0.4975 -> 0.4979`、`10m MAE 5.1150 -> 5.1146` を確認（`5m` は同等）。
- 最新バーが未確定で `atr_pips_14` などが `nan` の場合は、直近の有限特徴行へ自動フォールバックして
  `feature_row_incomplete` の誤発生を抑制します（予測停止を回避）。

加えて 2026-02-17 以降は JST 時間帯バイアス（`--session-bias-*`）を導入し、
同じ時間帯の先行方向ドリフトを after 式へ反映できます。
- 既定は `session_bias_weight=0.12`
- TF別重みは `session_bias_weight_map`（例: `1m=0.0,5m=0.26,10m=0.38`）で上書きできます。
  現在の既定は `1m=0.0,5m=0.26,10m=0.38` です。
- `1m` は過学習回避のため適用重みを 0.0 に固定しています。
- 同一期間VM評価（`bars=8050`）では `session_bias_weight=0.12` が
  `5m/10m` で hit と MAE の双方を小幅改善しました（`1m` は同等）。
- 同一固定データ（`bars=8050`, `2026-01-06T07:30:00+00:00`〜`2026-02-17T23:59:00+00:00`）では、
  旧既定 `1m=0.0,5m=0.26,10m=0.34` 比で新既定候補 `1m=0.0,5m=0.26,10m=0.38` は
  `5m hit 0.4886 -> 0.4886`（同等）, `5m MAE 3.2707 -> 3.2707`（同等）,
  `10m hit 0.4878 -> 0.4900`, `10m MAE 4.8568 -> 4.8564` を確認。

2026-02-18 の短期窓（2h/4h）では、`feature_expansion_gain=0.35` 構成で
`1m/5m` の `hit_delta` がマイナスに寄る局面を確認したため、運用側の runtime env は次を明示設定:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.10,5m=0.18,10m=0.26`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.18,10m=0.30`

同一時点の before/after 比較（`max-bars=120/240`）では、上記設定で
`1m/5m` の `hit_delta` マイナスを解消し、`10m` はプラスを維持することを確認済みです。

2026-02-17 時点では、短期TFの `TECH_HORIZON_CFG` を次に調整しています（`forecast_gate`/評価ジョブで同値）。
- `1m`: `trend_w=0.70`, `mr_w=0.30`
- `5m`: `trend_w=0.40`, `mr_w=0.60`
- `10m`: `trend_w=0.40`, `mr_w=0.60`
VM同一期間評価（`bars=8050`）では、`feature_expansion_gain=0.0` 基準で
`1m/5m/10m` の hit がそれぞれ `+0.0015/+0.0007/+0.0026`、MAE は
`-0.0114/-0.0854/-0.1262` 改善しました。

運用上はまず `scalp_fast` なら `1m`、`scalp` なら `5m` / `10m` を短期軸として見る前提にして、`8h` / `1d` を中期〜長期補完として確認します。  
`1h` と `8h` が同方向で `trend_strength` が高いほど「現在の順張り解釈」が強くなります。`range_pressure` 優勢で中立寄りの場合は「レンジ中の天井/底付近」に寄るケースが増えます。

## 戦略ごとのTF運用（2026-02-17更新）
- 各戦略は `execution/strategy_entry.py` の契約で `forecast_profile`（主TF）を持ち、`forecast_support_horizons`（補助TF）を併せて注入します。
- `forecast_gate` は主TF予測に加えて補助TFとの整合を評価し、同方向なら `edge` を微補正、逆方向なら `edge` を減衰します（`FORECAST_GATE_TF_CONFLUENCE_*`）。
- `entry_thesis` の `forecast_horizon/forecast_profile` が欠ける経路でも、`forecast_gate` 側で `strategy_tag` から主TFを補完します（例: `Micro*` は `10m`、`scalp_ping_5s*` は `1m`、`scalp_macd_rsi_div*` は `10m`）。
- micro系は各 `quant-micro-*.env` で `FORECAST_GATE_ENABLED=1` を維持し、`entry_thesis.forecast` の欠損（`not_applicable`）を防止します。
- `quant-m1scalper.env` も `FORECAST_GATE_ENABLED=1` を維持し、M1系の forecast 欠損を防止します。
- 例:
  - `SCALP_PING_5S*`: 主TF `1m` + 補助TF `5m,10m`
  - `SCALP_M1SCALPER`: 主TF `5m` + 補助TF `1m,10m`
  - `SCALP_MACD_RSI_DIV` / `MICRO_*`: 主TF `10m` + 補助TF `5m,1h`

## 黒板（entry_intent_board）での扱い
- 黒板は forecast を「方向決定ロジック」としては使わず、`details.forecast_context` の監査メタとして保持します。
- 協調の主判定は `raw_units * entry_probability` の score 集計で、forecast は記録・追跡用途です。
- 実際の `allow/scale/block` と TP/SL ヒント反映は `strategy_entry` / `order_manager` の preflight で行います。

## 戦略内融合（forecast_fusion）
- `execution/strategy_entry.py` で、戦略の既存計算（`units`, `entry_probability`）に対して
  forecast の向き（`p_up`）と強さ（`edge`）を合成する `forecast_fusion` を適用します。
- 方向一致時はロット/確率を小幅に押し上げ、逆行時や `allowed=false` はロット/確率を縮小します。
- `tf_confluence_score`（上位/下位TF整合）も同時に反映し、整合が弱いときは追加縮小、整合が強いときのみ軽微に押し上げます。
- `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_*` で、強い逆行予測（例: `direction_prob<=0.22` かつ `edge>=0.65`）は
  `units=0` として見送りできます。
- 反映結果は `entry_thesis["forecast_fusion"]` に保存し、監査時に
  `units_before/after`, `entry_probability_before/after`, `units_scale`, `forecast_reason`,
  `tf_confluence_score`, `strong_contra_reject` を追跡できます。
- TPは `tp_pips_hint` がある場合に `tp_pips` へブレンドし（順方向時のみ）、
  SLは `sl_pips_cap` がある場合に `sl_pips` を上限でクリップします。

## EXITでのforecast補正
- 各 `exit_worker` は `entry_thesis.forecast` / `forecast_fusion` を参照し、
  `build_exit_forecast_adjustment` で逆行確率を評価して EXIT 閾値へ反映します。
- 2026-02-18 時点で `scalp_m1scalper` / `scalp_rangefader` を含む全EXIT系で
  `apply_exit_forecast_to_targets`（利確/トレール）と
  `apply_exit_forecast_to_loss_cut`（損切り/最大保持）の補正が有効です。
- 2026-02-18 追記: `target_price` / `anchor_price` / `tp_pips_hint` / `range_low_price` / `range_high_price` を
  EXIT補正に取り込み、予測レンジが狭い局面では早めの利確・損切り、予測ターゲット距離が大きい局面では
  過度な早期クローズを抑える方向へ補正するようにしました。
- 2026-02-18 追記: `target_reach_prob`（目標到達確率）を EXIT 補正へ追加し、
  低確率なら早期クローズ寄り、高確率ならホールド寄りへ補正します（`allowed=false` 時は緩和しない）。
- 主なenv:
  - `EXIT_FORECAST_PRICE_HINT_ENABLED`
  - `EXIT_FORECAST_PRICE_HINT_WEIGHT_MAX`
  - `EXIT_FORECAST_PRICE_HINT_MIN_PIPS`
  - `EXIT_FORECAST_PRICE_HINT_MAX_PIPS`
  - `EXIT_FORECAST_RANGE_HINT_NARROW_PIPS`
  - `EXIT_FORECAST_RANGE_HINT_WIDE_PIPS`
  - `EXIT_FORECAST_TARGET_REACH_ENABLED`
  - `EXIT_FORECAST_TARGET_REACH_WEIGHT_MAX`
  - `EXIT_FORECAST_TARGET_REACH_LOW`
  - `EXIT_FORECAST_TARGET_REACH_HIGH`
