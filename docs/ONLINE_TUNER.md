# Online Tuner (minimal)

目的：直近 N 分の実績ログから **“ゆっくり動くノブ”**（入口閾値・Exit感度・macro:micro配分）を**小刻みに**調整し、
安全に継続最適化するための最小実装。

## 使い方（最短）

1. 依存: `pandas`, `pyyaml`
2. プリセット確認: `config/tuning_presets.yaml`
3. 一回実行:
```bash
python3 scripts/run_online_tuner.py --logs-glob "tmp/exit_eval*.csv"   --presets config/tuning_presets.yaml   --overrides-out logs/tuning/tuning_overrides.yaml   --history-dir logs/tuning/history   --minutes 15 --shadow
```
  - 実績ログが DB のみの場合は先にエクスポート:
```bash
PYTHONPATH=. python3 scripts/export_exit_eval.py --db logs/trades.db --out tmp/exit_eval_live.csv
```
4. 本番（定期実行）: `snippets/main_hook.pyfrag` を `main.py` の周期処理へ貼付

- 生成物：`logs/tuning/tuning_overrides.yaml`（本適用）、`logs/tuning/history/tuning_*.yaml`（履歴）
- 生成物の既定出力は `logs/tuning/` 配下（`TUNING_*_PATH` / `TUNING_RUNTIME_DIR` で上書き可）
- 本適用は `scripts/apply_override.py` で `logs/tuning/tuning_overlay.yaml` にマージし、
  ランタイムの設定読み込みで **オーバーレイ**します。

## 何を動かすか（小幅）
- Exit（lowvol）: `upper_bound_max_sec`, `hazard_debounce_ticks`, `hazard_cost_*`, `min_grace_before_scratch_ms`, `scratch_requires_events`
- Strategies: `MomentumPulse.min_confidence`, `MicroVWAPRevert.vwap_z_min`, `VolCompressionBreak.accel_pctile`, `BB_RSI_Fast.reentry_block_s`
- Alloc: `alloc.regime.quiet_low_vol.micro_share`

## ガード
- すべてクランプ：Exit, Gate, Alloc の範囲はチューナ内部で固定
- 1回の更新幅は小さく、履歴が常に出ます
- `--shadow` なら `overrides.yaml` は書かず、履歴だけ

## ログ前提（柔軟）
- 最低限：`timestamp, reason, pips[, strategy, hazard_ticks, regime]` を読む
- 列が無ければスキップします

## 次の拡張
- Regime別ルール、戦略別 acceptance のターゲット制御
- 影運用の PNL 差分計測、GPT の影評価マージ
