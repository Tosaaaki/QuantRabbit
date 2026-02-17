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
- `tp_pips_hint`: TP 方向ヒント（pips）
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

運用上はまず `scalp_fast` なら `1m`、`scalp` なら `5m` / `10m` を短期軸として見る前提にして、`8h` / `1d` を中期〜長期補完として確認します。  
`1h` と `8h` が同方向で `trend_strength` が高いほど「現在の順張り解釈」が強くなります。`range_pressure` 優勢で中立寄りの場合は「レンジ中の天井/底付近」に寄るケースが増えます。
