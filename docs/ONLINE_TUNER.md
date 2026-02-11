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
4. 本番（定期実行）: systemd timer で実行（VM）
   - `systemd/quant-online-tuner.timer` → `systemd/quant-online-tuner.service` → `scripts/run_online_tuner_live.sh`
   - 状態確認:
```bash
systemctl status quant-online-tuner.timer
journalctl -u quant-online-tuner.service -n 200 --no-pager
```

- 生成物：`logs/tuning/tuning_overrides.yaml`（本適用）、`logs/tuning/history/tuning_*.yaml`（履歴）
- 生成物の既定出力は `logs/tuning/` 配下（`TUNING_*_PATH` / `TUNING_RUNTIME_DIR` で上書き可）
- 本適用は `scripts/apply_override.py` で `logs/tuning/tuning_overlay.yaml` にマージし、
  ランタイムの設定読み込みで **オーバーレイ**します。
- 安全弁の state: `logs/tuning/online_tuner_state.yaml`（LKG/rollback の状態）
- LKG: `logs/tuning/tuning_overrides.lkg.yaml`（最後に良かった overrides）

## 何を動かすか（小幅）
- Exit（lowvol）: `upper_bound_max_sec`, `hazard_debounce_ticks`, `hazard_cost_*`, `min_grace_before_scratch_ms`, `scratch_requires_events`
- Strategies: `MomentumPulse.min_confidence`, `MicroVWAPRevert.vwap_z_min`, `VolCompressionBreak.accel_pctile`, `BB_RSI_Fast.reentry_block_s`
- Alloc: `alloc.regime.quiet_low_vol.micro_share`

## ガード
- すべてクランプ：Exit, Gate, Alloc の範囲はチューナ内部で固定
- **データ不足/偏り時は freeze**（min trades / target regime カバレッジ / reason 欠落）
- **bad window が継続したら LKG に自動ロールバック**（cooldown 付き）
- `--shadow` なら `overrides.yaml` は書かず、履歴だけ

### 主要 Env（運用で調整）
- `TUNER_MIN_TRADES` / `TUNER_MIN_TARGET_REGIME_PCT` / `TUNER_MIN_REASON_NONEMPTY_PCT`
- `TUNER_MIN_MICRO_TRADES`（alloc drift 防止）
- `TUNER_ROLLBACK_AFTER_BAD_RUNS` / `TUNER_ROLLBACK_COOLDOWN_MIN`
- `TUNER_GOOD_*` / `TUNER_BAD_*`（EV/PF/勝率のしきい値）

## ログ前提（柔軟）
- 最低限：`timestamp, reason, pips[, strategy, hazard_ticks, regime]` を読む
- 列が無ければスキップします

## 次の拡張
- Regime別ルール、戦略別 acceptance のターゲット制御
- 影運用の PNL 差分計測、GPT の影評価マージ
