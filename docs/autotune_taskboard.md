# オンラインチューナ影運用タスクボード

更新日時: 2025-11-17 07:10Z

## 進捗サマリ
- **本番モード**（`TUNER_SHADOW_MODE=false`）に切り替え済み。ループが直接 `config/tuning_overrides.yaml` を更新する。
- 直近の昇格パッチ: `config/tuning_history/tuning_20251102_021731.yaml` → `config/tuning_overrides.yaml` → `config/tuning_overlay.yaml` を生成。
- 適用内容は Exit（lowvol）のタイムアウト・ハザード感度、quiet_low_vol の `micro_share=0.30`。今後は本番ループで継続更新される。
- M1Scalper の自動調整ループをワーカー内で起動する準備を追加（`SCALP_AUTOTUNE_ENABLED=1` で有効化）。
- M1Scalper の自動調整ガードは `SCALP_AUTOTUNE_MIN_WIN_RATE` で調整可能。

## 現在検証中
- [ ] 初回本番ループの監視  
 `logs/metrics.db` の `decision_latency` と `hazard_exit` 閾値が想定範囲 (p95<2s, hazard 0.05–0.30) に収まるかを 3 サイクル確認。
- [ ] オーバーレイ読込の検証  
 `config/tuning_overlay.yaml` が起動時に読み込まれているか、ログ（`[TUNER] run`）と `config/tuning_overrides.yaml` の更新時刻で確認。

## 次に実行すべきアクション
1. 本番モードの定常化  
 - [ ] `scripts/run_online_tuner.py` の 2 回目以降の実行で `config/tuning_overrides.yaml` が追記更新されるかを確認。  
 - [ ] `scripts/apply_override.py` を nightly で実行する仕組み (cron/systemd) を検討し、自動的に `config/tuning_overlay.yaml` を更新。
2. ガードレンジの再評価  
 - [ ] 本番パラメータが上下限へ寄っていないか `docs/ONLINE_TUNER.md` の基準に照らしてレビュー。  
 - [ ] 逸脱時は `tuning_presets.yaml` 側のクランプを調整。
3. 運用モニタリング  
 - [ ] `PYTHONPATH=.` 付きで `scripts/run_online_tuner.py` を 5–15 分間隔で回す cron/systemd タイマーを有効化。  
 - [ ] 実行ログとエラー通知の経路（Cloud Logging / Slack）を確定し、失敗時に自動リトライする。

## 参考コマンド
```bash
PYTHONPATH=. python3 scripts/run_online_tuner.py \
  --logs-glob "tmp/exit_eval_*_v2.csv" \
  --presets config/tuning_presets.yaml \
  --overrides-out config/tuning_overrides.yaml \
  --minutes 15
```

```bash
PYTHONPATH=. python3 scripts/apply_override.py \
  --base config/tuning_presets.yaml \
  --over config/tuning_overrides.yaml \
  --out  config/tuning_overlay.yaml
```

## メモ
- 本番モードでは `config/tuning_overrides.yaml` がそのまま上書きされるため、履歴との差分は `config/tuning_history/` で追跡する。
- `decision_latency`・`fallback率` がガードライン外になった場合は即座に `TUNER_SHADOW_MODE=true` へ戻し、原因を切り分ける。
- 提案が急変した場合は `tmp/exit_eval_*.csv` の入力内容（特に `reason`, `pips`, `strategy` 列）と `alloc.regime` のフェーズを確認する。
