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
- 反発監査キー: `rebound_signal_20` / `rebound_drop_score_20` / `rebound_oversold_score_20` /
  `rebound_decel_score_20` / `rebound_wick_score_20` / `rebound_weight`
- `forecast_gate.decide()` は `rebound_signal_20` から独立値 `rebound_probability`（0.0-1.0）も返し、
  `p_up` とは別軸で「急落後反発の強さ」を戦略側に渡せるようにします。
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
- `FORECAST_TECH_REBOUND_ENABLED=1`（急落後反発シグナルを有効化）
- `FORECAST_TECH_REBOUND_WEIGHT=0.06`（horizon map 未指定時の既定重み）
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.02,10m=0.01`（短期TFの反発重み）
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
- `rebound_probability`: 反発確率ヒント（0.0-1.0, `p_up` と独立）
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

反発項を同一期間で比較する場合は `--rebound-weight` / `--rebound-weight-map` を併用します
（例: `--rebound-weight 0.06 --rebound-weight-map 1m=0.10,5m=0.06,10m=0.01`）。

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

2026-02-18 フォローアップ（反発シグナル追加）では、VM実データ（M1連続窓）を同一期間で再評価し、
短期TFの採用値を次で固定:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.0`（再固定）
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.10,5m=0.18,10m=0.26`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.18,10m=0.30`
- `FORECAST_TECH_REBOUND_ENABLED=1`
- `FORECAST_TECH_REBOUND_WEIGHT=0.06`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.06,10m=0.01`

同一データ比較（`logs/reports/forecast_improvement/rebound_tune_report_20260218T024741Z.md`）では、
候補適用時の `after` 指標差分（candidate-after - base-after）は次:
- `1m`: `hit +0.0002`, `mae -0.0002`
- `5m`: `hit -0.0001`, `mae -0.0002`
- `10m`: `hit +0.0000`, `mae -0.0002`

2026-02-18 の72h窓（VM実データ, `bars=3181`, `2026-02-15T22:06:00+00:00`〜`2026-02-18T03:34:00+00:00`）で
`5m` 重視の再探索を実施し、短期TFの採用値を次に更新:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.05`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.26,10m=0.28`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`

同一72h窓の再チェック（`logs/reports/forecast_improvement/report_20260218T035154Z_72h_candidate_recheck.md`）では、
旧運用値（`gain=0.05`, `breakout_5m=0.20`, `session_5m=0.18`）比で
`5m hit_delta +0.0026 -> +0.0039`, `5m mae_delta -0.0013 -> -0.0016` を確認。
`1m/10m` は同等で、合計は `hit_delta_sum +0.0117 -> +0.0131`,
`mae_delta_sum -0.0061 -> -0.0064` と改善しました。

同日フォローアップとして、直近VM実データで `2h/4h/24h` と `72h` を同時監査
（`report_20260218T040516Z_followup_final.md`）し、`5m` 重みを最終調整:
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.22,10m=0.28`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`（維持）
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.05`（維持）

`5m=0.26` 比で、`2h/4h` は同等、`24h` は `hit/mae` とも小幅改善、
`72h` はごく小幅の悪化に留まるため、短中期バランス優先で `5m=0.22` を採用しました。

同日追加の微調整（`report_20260218T041728Z_more_improve.md`）では、
`breakout/session` は現行（`5m=0.22`）が最適のまま、`rebound_5m` のみ再探索。
`rebound_5m=0.06` は `0.02` 比で hit は同等のまま `2h/4h/24h/72h` 全窓で MAE を
微小改善したため、運用値を `1m=0.10,5m=0.06,10m=0.01` に更新しました。

さらに 2026-02-18 04:40 UTC のVM実データ再探索（`logs/reports/forecast_improvement/grid_fast_5m_20260218T044010Z.json`）で、
`72h(4320 bars)` と `24h(1440 bars)` を同時評価し、次を採用値に更新:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.04`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.20,10m=0.28`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.20,10m=0.30`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.03,10m=0.01`

比較結果（新 - 旧）:
- 72h: `1m hit +0.0007 / mae_delta -0.0006`, `5m hit +0.0016 / mae_delta -0.0018`, `10m hit -0.0010 / mae_delta -0.0022`
- 24h: `hit_after` は `1m/5m/10m` で同値、`mae_delta` 変化は微小（`5m +0.0001` を含む誤差レンジ）

`5m` の確率改善を優先しつつ直近24hの hit を落とさないため、上記を運用値として反映しています。

同日 2026-02-18 05:09 UTC の追加再探索（VM実データ, `logs/reports/forecast_improvement/extra_improve_grid_latest.json`）で、
`2h/4h/24h/72h` を同時最適化し、劣化ガード（24h/72h の hit/MAE 下限）を満たした候補として次を採用:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.03`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.20,10m=0.30`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.20,10m=0.30`（維持）
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`

最終差分（new - baseline, baseline=`fg=0.04,b10=0.28,rb5=0.03,rb10=0.01`）:
- 2h: `hit +0.0000`, `mae -0.000120`
- 4h: `hit +0.0000`, `mae -0.000061`
- 24h: `hit +0.003672`, `mae +0.000110`（許容ガード内）
- 72h: `hit +0.001266`, `mae -0.000164`, `range_cov +0.000317`

重み付き objective は `+0.02198` で正、かつ 24h/72h の安全条件を満たすため、
上記値を runtime env の運用値へ更新しました。

同日 2026-02-18 05:23 UTC の追加探索（`logs/reports/forecast_improvement/extra_improve_b5s5_latest.json`）で、
現行（`fg=0.03,b5=0.20,b10=0.30,s5=0.20,s10=0.30,rb5=0.01,rb10=0.02`）を基準に
`5m` 系重みを局所再探索し、劣化ガードを満たす最上位候補を次で採用:
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.24,10m=0.30`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.30`
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.03`（維持）
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`（維持）

差分（new - current）:
- 2h: `hit +0.0000`, `mae +0.000018`
- 4h: `hit +0.0000`, `mae +0.000062`
- 24h: `hit +0.0000`, `mae -0.000242`
- 72h: `hit +0.002222`, `mae -0.000156`

短期窓（2h/4h）の MAE は微小悪化する一方、24h/72h で MAE 改善、
72h の hit が上振れし、重み付き objective は `+0.01762` を確認しました。

2026-02-21 の再評価（VM実データ, `bars=8050`）では、`型`（短期の trend/range 混在）で
`5m` の MAE 劣化を抑えるため、short-list 候補比較を実施しました。
採用値（cand_c）は次です。
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.24,10m=0.30`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.24`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.00,10m=0.02`

同一期間比較（`2026-01-07T02:54:00+00:00` ～ `2026-02-20T21:59:00+00:00`）:
- `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
- `5m`: `hit_delta=+0.0024`, `mae_delta=-0.0006`, `range_cov_delta=-0.0001`
- `10m`: `hit_delta=+0.0052`, `mae_delta=-0.0041`, `range_cov_delta=+0.0003`

`1m` はほぼ同等を維持しつつ、`5m/10m` で `hit` と `MAE` を同時改善するため、
運用値は上記に更新しています。

同日 2026-02-21 の追加最適化（narrow grid, 96候補）で、上記 cand_c からさらに
`5m/10m` を押し上げる候補（cand_d2）を採用しました。
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`（維持）
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`（維持）
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.26,10m=0.32`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.26`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`

同一期間比較（`bars=8050`）:
- `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
- `5m`: `hit_delta=+0.0025`, `mae_delta=-0.0006`, `range_cov_delta=+0.0000`
- `10m`: `hit_delta=+0.0056`, `mae_delta=-0.0044`, `range_cov_delta=+0.0001`

同日 2026-02-21 のフォローアップ（mid grid, 同一 `bars=8050` 固定スナップショット）で、
`cand_d2` と周辺7候補を再比較し、最小差分の上積み候補 `mid_253327` を採用:
- `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.00`（維持）
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.12,5m=0.25,10m=0.33`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.27`
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES=120`（維持）
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.10,5m=0.01,10m=0.02`（維持）

同一期間の before/after（mid_253327）:
- `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
- `5m`: `hit_delta=+0.0025`, `mae_delta=-0.0006`, `range_cov_delta=+0.0000`
- `10m`: `hit_delta=+0.0055`, `mae_delta=-0.0046`, `range_cov_delta=+0.0003`

`cand_d2` 比では `10m hit` が微減（`+0.0056 -> +0.0055`）する一方、
`10m mae`（`-0.0044 -> -0.0046`）と `10m range_cov`（`+0.0001 -> +0.0003`）が改善し、
総合スコア（`score_vs_current=+0.000083`）で優位だったため runtime を更新。

同日 2026-02-21 の再最適化（two-stage search, VM snapshot `bars=8050`）で、
`mid_253327` から `5m/10m` の hit を優先して再探索し、`cand_e1` を採用しました。
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

VM同一期間比較（`2026-01-07T02:54:00+00:00` ～ `2026-02-20T21:59:00+00:00`）:
- `1m`: `hit_delta=-0.0003`, `mae_delta=-0.0001`, `range_cov_delta=+0.0002`
- `5m`: `hit_delta=+0.0040`, `mae_delta=-0.0015`, `range_cov_delta=+0.0006`
- `10m`: `hit_delta=+0.0117`, `mae_delta=-0.0072`, `range_cov_delta=+0.0012`

`mid_253327` 比の after-after 差分:
- `1m`: ほぼ同等
- `5m`: `hit_after +0.0015`, `mae_after -0.0008`, `range_cov_after +0.0006`
- `10m`: `hit_after +0.0062`, `mae_after -0.0026`, `range_cov_after +0.0009`

同日 2026-02-21 の追加最適化（1m-only grid, 90候補）で、
`cand_e1` の `5m/10m` を固定したまま `1m` 専用パラメータだけを再探索し、
`cand_e1_1mboost` を採用しました。
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.16,5m=0.24,10m=0.31`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.14,5m=0.01,10m=0.04`
- その他 `cand_e1` パラメータは維持

VM同一期間比較（`bars=8050`）:
- `1m`: `hit_delta=+0.0002`, `mae_delta=-0.0001`, `range_cov_delta=+0.0005`
- `5m`: `hit_delta=+0.0040`, `mae_delta=-0.0015`, `range_cov_delta=+0.0006`
- `10m`: `hit_delta=+0.0117`, `mae_delta=-0.0072`, `range_cov_delta=+0.0012`

`cand_e1` 比の after-after 差分:
- `1m`: `hit_after +0.0005`, `range_cov_after +0.0003`（`mae_after` はほぼ同等）
- `5m`: 同等
- `10m`: 同等

同日 2026-02-21 の追加最適化（6本同時比較, `bars=8050`）で、
`cand_e1_1mboost` を基準に `feature/session/rebound` を再加速探索し、
`candD` を採用しました。
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

VM同一期間 before/after（`candD`）:
- `1m`: `hit_delta=-0.0002`, `mae_delta=+0.0009`, `range_cov_delta=-0.0002`
- `5m`: `hit_delta=+0.0054`, `mae_delta=-0.0020`, `range_cov_delta=-0.0004`
- `10m`: `hit_delta=+0.0129`, `mae_delta=-0.0125`, `range_cov_delta=+0.0006`

`cand_e1_1mboost` 比の after-after 差分:
- `1m`: `hit_after -0.0003`, `mae_after +0.0010`, `range_cov_after -0.0006`
- `5m`: `hit_after +0.0013`, `mae_after -0.0006`, `range_cov_after -0.0010`
- `10m`: `hit_after +0.0012`, `mae_after -0.0053`, `range_cov_after -0.0006`

判定:
- `5m/10m` で `hit` と `MAE` の同時改善幅が最も大きく、総合スコアが最大だったため採用。
- `1m` は軽微に悪化するが、`5m/10m` への寄与を優先する運用方針で許容。

同日 2026-02-22 の微調整（`candD_1m_mae_boost`）では、
`candD` の `5m/10m` を固定し、`1m` 専用重みのみを再探索しました。
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.28,10m=0.34`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`
- その他 `candD` パラメータは維持

VM同一期間比較（`candD` 比, `bars=8050`）:
- `1m`: `hit_after_delta=+0.0000`, `mae_after_delta=-0.000068`, `range_cov_after_delta=+0.0000`
- `5m`: 差分なし
- `10m`: 差分なし

判定:
- 方向精度（hit）を維持したまま `1m MAE` を僅かに改善できたため採用。

同日 2026-02-22 の追加微調整（`cand_hit_nonneg_mae_up`）では、
`candD_1m_mae_boost` のまま `hit` を落とさず `5m/10m MAE` をさらに削る制約で、
`breakout/session` の 2 点だけを再探索しました。
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.34`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.24,10m=0.35`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`（維持）

VM同一期間比較（`candD_1m_mae_boost` 比, `bars=8050`）:
- `1m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
- `5m`: `hit_after_delta=+0.000000`, `mae_after_delta=-0.000003`, `range_cov_after_delta=+0.000000`
- `10m`: `hit_after_delta=+0.000000`, `mae_after_delta=-0.000811`, `range_cov_after_delta=+0.000446`

判定:
- `hit` を全TFで維持したまま `5m/10m MAE` と `10m range_cov` を改善できたため採用。

同日 2026-02-22 の追加微調整（`cand_10m_hit_mae_boost`）では、
`10m` の hit と MAE を同時改善する目的で `b10/s10` のみを再探索しました。
- `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.27,10m=0.32`
- `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.24,10m=0.37`
- `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`（維持）

VM同一期間比較（`cand_hit_nonneg_mae_up` 比, `bars=8050`）:
- `1m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
- `5m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
- `10m`: `hit_after_delta=+0.000594`, `mae_after_delta=-0.000728`, `range_cov_after_delta=+0.000149`

判定:
- `5m` を維持したまま `10m` の `hit/MAE/range_cov` を同時改善できたため採用。

同日 2026-02-22 の追加改善（`dynamic_10m_weighting`）では、
`10m` の breakout/session 重みを「固定値」から「相場状態連動（可変）」へ切り替えました。
- `FORECAST_TECH_DYNAMIC_WEIGHT_ENABLED=1`
- `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=10m`
- `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.14`
- `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.02`
- `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.16`
- `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.08`
- `FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER=0.06`
- `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.18`
- `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.0`

VM同一期間比較（`dynamic_off` 比, `bars=8050`）:
- `1m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
- `5m`: `hit_after_delta=+0.000000`, `mae_after_delta=+0.000000`, `range_cov_after_delta=+0.000000`
- `10m`: `hit_after_delta=+0.000446`, `mae_after_delta=-0.002131`, `range_cov_after_delta=-0.000297`

判定:
- `10m` で `hit` と `MAE` を同時改善でき、coverage 低下は軽微なため採用。

同日 2026-02-22 の追加微調整（`dynamic_10m_weighting_stronger`）では、
可変化を維持したまま `10m` の改善幅をもう一段押し上げるため、
dynamic ゲインのみを局所比較しました（`bars=8050`）。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_eval_20260222T063520Z_dynamic_off.json`
  - `logs/reports/forecast_improvement/forecast_eval_20260222T063520Z_dynamic_on.json`
  - `logs/reports/forecast_improvement/forecast_dynamic_targeted_compare_20260222.json`
- 運用反映値:
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=10m`（維持）
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.16`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.20`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.12`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.22`
- 直前運用値（`max_delta=0.14,b_skill=0.16,b_regime=0.08,s_bias=0.18`）比:
  - `1m`: 変化なし
  - `5m`: 変化なし
  - `10m`: `hit_after +0.000446`, `mae_after -0.000327`, `range_cov_after +0.000149`

判定:
- `5m` を維持したまま `10m` の `hit/MAE/range_cov` を同時改善できたため、
  dynamic 強化版を runtime 採用。

同日 2026-02-22 の追加改善（`dynamic_h510_aggr3`）では、
`10m` 強化値を維持したまま `5m` にも可変重みを適用し、
`5m/10m` を同時に改善する候補を採用しました。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_dynamic_candidates_20260222T2.json`
  - `logs/reports/forecast_improvement/forecast_dynamic_h510_check_20260222.json`
- 運用反映値:
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.20`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.20`（維持）
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.16`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.26`
- 直前運用値（`horizons=10m,max_delta=0.16,b_regime=0.12,s_bias=0.22`）比:
  - `1m`: 変化なし
  - `5m`: `hit_after +0.000898`, `mae_after -0.000743`, `range_cov_after +0.000749`
  - `10m`: `hit_after +0.000743`, `mae_after -0.000587`, `range_cov_after +0.000000`

判定:
- `5m/10m` で `hit` と `MAE` を同時改善できるため、`5m,10m` 可変化を採用。

同日 2026-02-22 の追加改善（`dynamic_h510_rand028`）では、
`5m,10m` 可変化を維持しつつ、固定mapとdynamicゲインを同時に多窓探索して
実運用向けの改善幅を拡大しました。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_dyn_multistage_20260222.json`
  - `logs/reports/forecast_improvement/forecast_dynamic_h510_check_20260222.json`
- 運用反映値:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.02`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.29,10m=0.30`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.26,10m=0.41`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.00,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.18`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.18`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.10`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.26`（維持）
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.04`
- 直前運用値（`dynamic_h510_aggr3`）比:
  - `24h`: `5m hit +0.001147 / mae -0.000471`, `10m hit +0.000000 / mae -0.001657`
  - `72h`: `5m hit +0.000983 / mae -0.001357`, `10m hit +0.001306 / mae -0.003911`
  - `full`: `5m hit +0.000599 / mae -0.000955`, `10m hit +0.001486 / mae -0.002445`
  - `1m` は各窓で非劣化（hit維持、MAEは小幅改善）。

判定:
- `24h/72h/full` の全窓で `5m/10m` の `hit` と `MAE` を同時改善できたため採用。

同日 2026-02-22 の追加改善（`dynamic_h510_rnd161`）では、
多窓ランダム探索を拡張（`261候補→上位24候補を再評価`）し、
`10m` の方向一致と誤差をさらに押し上げる設定を採用しました。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_dyn_multistage_v2_20260222.json`
- 運用反映値:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.02`（維持）
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.14,5m=0.29,10m=0.34`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.26,10m=0.45`
  - `FORECAST_TECH_REBOUND_WEIGHT_MAP=1m=0.16,5m=0.01,10m=0.05`
  - `FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS=5m,10m`（維持）
  - `FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA=0.22`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER=0.015`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN=0.22`
  - `FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN=0.16`
  - `FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN=0.34`
  - `FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN=0.0`
- 直前運用値（`dynamic_h510_rand028`）比:
  - `24h`: `10m hit +0.003425 / mae -0.003373`, `5m mae -0.000260`
  - `72h`: `10m hit +0.002286 / mae -0.004327`, `5m hit +0.000655 / mae -0.000243`
  - `full`: `10m hit +0.001634 / mae -0.002883`, `5m mae -0.000217`
  - `1m` は非劣化（同等）。

判定:
- 3窓すべてで `10m` の `hit/MAE` を同時改善し、`5m` も非劣化〜改善のため採用。

同日 2026-02-22 の追加改善（`dynamic_h1510_rnd100`）では、
`rnd161` をベースに `5m/10m` 同時改善を優先した再探索（`161候補→上位30候補を多窓再評価`）を実施し、
`10m` の方向一致と誤差をさらに改善しつつ、`5m` の MAE も押し下げる構成へ更新しました。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_dyn_multistage_v3_20260222.json`
- 運用反映値:
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
  - `24h`: `10m mae -0.001756 / cov +0.001142`, `5m mae -0.000039`（hitは同等）
  - `72h`: `10m mae -0.004361`（hit同等）、`5m mae -0.001198`（hit -0.000328）
  - `full`: `10m hit +0.001040 / mae -0.002625 / cov +0.000149`, `5m mae -0.000740`（hit同等）
  - `1m`: `24h/full` は hit同等 + MAE改善、`72h` は `hit -0.000340` だが MAE改善。

判定:
- `full` で `10m` の `hit/MAE` を同時改善し、`5m` は hit維持で MAE 改善のため採用。

同日 2026-02-22 の追加改善（`dynamic_meta_rnd056`）では、
`rnd100` を基準に `lookback/min_samples` と適応重み本体まで探索対象を拡張し、
`10m` の方向一致をさらに押し上げる構成を採用しました。
- 比較ファイル:
  - `logs/reports/forecast_improvement/forecast_dyn_multistage_v5_20260222.json`
- 運用反映値:
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
- 同一スナップショット（最新 bars）での `rnd100` 比:
  - `24h`: `10m hit +0.001142 / mae -0.011401`, `5m mae -0.004179`
  - `72h`: `10m hit +0.007178 / mae -0.002647`, `5m mae -0.001440`
  - `full`: `10m hit +0.007426 / mae -0.003738`, `5m hit -0.000300 / mae -0.001179`
  - `1m` は hit 改善だが、`full` で MAE が `+0.000085` とごく小幅悪化。

判定:
- `10m` の hit 改善幅が大きく、`5m` も MAE 改善を維持できるため採用。

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
- `rebound_probability` がある場合は、`p_up` と独立に反発バイアスとして合成し、
  逆行局面の過度な縮小/拒否を緩和できます（long側中心）。
- 方向一致時はロット/確率を小幅に押し上げ、逆行時や `allowed=false` はロット/確率を縮小します。
- `tf_confluence_score`（上位/下位TF整合）も同時に反映し、整合が弱いときは追加縮小、整合が強いときのみ軽微に押し上げます。
- `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_*` で、強い逆行予測は `units=0` として見送りできます。
  - 逆行強度は `edge_strength = abs(edge - 0.5) / 0.5` で評価します。
  - 例: `direction_prob<=0.22` かつ `edge_strength>=0.65`（`allowed=false` も条件）で reject。
- 反発オーバーライド（`STRATEGY_FORECAST_FUSION_REBOUND_OVERRIDE_*`）を有効にすると、
  `rebound_probability` が十分高い long については strong-contra reject を回避し、
  縮小サイズでの試行に切り替えます。
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
