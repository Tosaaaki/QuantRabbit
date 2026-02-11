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
- `FORECAST_GATE_STYLE_GUARD_ENABLED=1`
- `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH=0.52`
- `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE=0.52`
- 戦略別 override:
  - `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH_STRATEGY_TRENDMA=...`
  - `FORECAST_GATE_EDGE_BLOCK_TREND_STRATEGY_TRENDMA=...`
  - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_BBRSI=...`
  - `FORECAST_GATE_EDGE_BLOCK_RANGE_STRATEGY_BBRSI=...`
