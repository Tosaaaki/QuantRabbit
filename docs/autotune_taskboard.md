# オンラインチューナ影運用タスクボード

更新日時: 2025-11-02 02:18Z

## 進捗サマリ
- 現在は **シャドウ運用**でオンラインチューナを稼働中 (`TUNER_SHADOW_MODE=true`)。
- 直近の実行履歴: `config/tuning_history/tuning_20251102_021731.yaml`
- 生成された提案は Exit（lowvol）と quiet_low_vol 配分の微調整。まだ本番適用は行っていない。

## 現在検証中
- [ ] シャドウ履歴の妥当性確認  
 チューナ出力の `upper_bound_max_sec`, `hazard_debounce_ticks`, `micro_share` が想定レンジ内で推移するかレビューする。
- [ ] 入力ログの鮮度チェック  
 `tmp/exit_eval_*_v2.csv` の更新が 15 分以内か、欠損がないか確認する。

## 次に実行すべきアクション
1. シャドウ期間の評価  
 - [ ] `config/tuning_history/` を1–2セッション分レビューし、指標（EV、hazard比率）の改善方向か確認。  
 - [ ] 必要なら `docs/ONLINE_TUNER.md` のガイドに沿ってパラメータのクランプ範囲を調整。
2. 本適用への移行判断  
 - [ ] 影運用で問題なければ `TUNER_SHADOW_MODE=false` に切り替える。  
 - [ ] `scripts/apply_override.py` を実行して `config/tuning_overlay.yaml` を生成し、ロジックで読み込むか検討。
3. 定期実行の自動化  
 - [ ] `PYTHONPATH=.` を付与した上で cron / systemd タイマーへ登録。  
 - [ ] 実行ログとエラー通知の経路（例: Cloud Logging / Slack）を決める。

## 参考コマンド
```bash
PYTHONPATH=. python3 scripts/run_online_tuner.py \
  --logs-glob "tmp/exit_eval_*_v2.csv" \
  --presets config/tuning_presets.yaml \
  --overrides-out config/tuning_overrides.yaml \
  --minutes 15 --shadow
```

```bash
python3 scripts/apply_override.py \
  --base config/tuning_presets.yaml \
  --over config/tuning_overrides.yaml \
  --out  config/tuning_overlay.yaml
```

## メモ
- シャドウ運用中は `config/tuning_overrides.yaml` が書き換わらないため、履歴だけで差分を追跡できる。
- 本適用前に `decision_latency`・`fallback率` がガードライン内かを必ずダッシュボードで確認する。
- 影運用の提案が急変した場合は `tmp/exit_eval_*.csv` の入力内容（特に `reason`, `pips`, `strategy` 列）を確認する。
