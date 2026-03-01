# Replay Standard (実運用寄せ)

このドキュメントは、実運用に近い条件でリプレイするための標準手順をまとめたものです。

## 標準ルール
- 実運用寄せの既定は `scripts/replay_exit_workers_groups.py` を使用する。
- **ハードTPは有効 / ハードSLは無効** にする。
  - `--no-hard-sl` を付ける。
  - `--no-hard-tp` は付けない（TPはデフォ有効）。
- `end_of_replay` 強制決済は除外する（`--exclude-end-of-replay`）。
- ワーカーは **毎回 `--workers` で選択**する。
- 出力は `summary_all.json` を採用する。
- シナリオ同時再生を行う場合は `--scenarios` を追加する。既定は `all` で、既存運用と同一。
- `summary_all.json` は `base_scenarios` / `tuned_scenarios` にシナリオ別要約を持つ（後述）。

## 例

```bash
python scripts/replay_exit_workers_groups.py \
  --ticks tmp/ticks_USDJPY_YYYYMM_all.jsonl \
  --workers session_open \
  --no-hard-sl \
  --exclude-end-of-replay \
  --out-dir tmp/replay_exit_workers_groups_YYYYMM_all
```

```bash
python scripts/replay_exit_workers_groups.py \
  --ticks tmp/ticks_USDJPY_YYYYMM_all.jsonl \
  --workers scalp_ping_5s_b,session_open \
  --scenarios all,wide_spread,high_vol,trend \
  --no-hard-sl \
  --exclude-end-of-replay \
  --out-dir tmp/replay_exit_workers_groups_YYYYMM_scenarios
```

### `summary_all.json` のシナリオ追跡

- 追加されるキー:
  - `base_scenarios[scenario].summary`
  - `base_scenarios[scenario].selection` (`requested` / `applied` / `excluded`)
  - `base_scenarios[scenario].tick_count` / `tick_meta`
  - `tuned_scenarios[...]`（`--tune` 時）
- 出力ファイル命名:
  - `replay_exit_<worker>_base_<scenario>.json`
  - `replay_exit_<worker>_tuned_<scenario>.json`
- `scenario == all` は従来互換として、`replay_exit_<worker>_base.json` / `replay_exit_<worker>_tuned.json` も併存する。

## 補足
- `--ticks` は bid/ask を含む JSONL を前提にする。
- 窓を変える場合は `--ticks` の入力だけ差し替える。
- ルールを変えない限り、比較は `summary_all.json` 同士で行う。
- `replay_exit_workers_groups.py` は存在するワーカーのみ実行する。環境差分でモジュールが無いワーカーはスキップされるため、実行時は `--workers` を現行 VM のワーカー構成に合わせる。

## 内部精度ゲート（walk-forward）

- リプレイ出力に対して in-sample / out-of-sample を自動判定する場合は `scripts/replay_quality_gate.py` を使う。
- 実行バックエンドは 2 種類:
  - `exit_workers_groups`（`scripts/replay_exit_workers_groups.py`）
  - `exit_workers_main`（`scripts/replay_exit_workers.py`）
- 標準フラグ（`--no-hard-sl` / `--exclude-end-of-replay`）を既定で適用する。
- 閾値は `config/replay_quality_gate*.yaml` の `gates.default` と `gates.workers` で管理する。
- `replay.env`（config）で replay 実行時の環境変数を固定化できる。
  - 例: `SCALP_REPLAY_PING_VARIANT=C|D`,
    `SCALP_PING_5S_C_MAX_TICK_AGE_MS=9999999999`,
    `SCALP_PING_5S_D_MAX_TICK_AGE_MS=9999999999`
- `exit_workers_main` は `replay.intraday_start_utc` / `replay.intraday_end_utc` を指定すると、
  tick ファイル名の日付（`YYYYMMDD`）に対して日内 UTC 時間帯を自動適用できる。
- `config/replay_quality_gate_main.yaml` の既定は intraday 無効（空文字）として扱い、
  フルデイ再生で品質ゲートを判定する。
- `replay_quality_gate.py` は `ticks_globs`（config 配列）または
  `--ticks-glob` のカンマ区切り複数指定を受け付ける。
  複数 root を使う場合は basename（`USD_JPY_ticks_YYYYMMDD.jsonl`）で重複を解消し、
  よりサイズが大きいファイルを優先する。
- `replay_quality_gate_main.yaml` 既定は
  `logs/replay` と `logs/archive/replay.*.dir` の両方を参照する。
- `min_tick_lines`（main config 既定: `50000`）未満の tick ファイルは
  walk-forward 対象から自動除外される。
- `exclude_end_of_replay=true` のまま短い intraday 窓を使うと、
  close が `end_of_replay` のみになって `trade_count=0` に寄る場合がある。
  窓を使う場合はこの挙動を前提に評価する。
- `exit_workers_main` で `replay.main_only=true` を使うと、
  TrendMA/BB_RSI の main 経路だけを再生し、scalp replay 経路を省略できる（高速化向け）。
- `replay_exit_workers` は replay 実行時に `factor_cache` のディスク永続化を無効化する。
  これにより長時間リプレイ時の I/O 競合を避け、再生時間のばらつきを抑える。
- `replay_quality_gate_main.yaml` の walk-forward 既定は
  `train_files=2 / test_files=1 / step_files=1`。
- `replay_quality_gate_ping5s_d.yaml` の walk-forward 既定は
  `train_files=3 / test_files=2 / step_files=1`（2026-02-24更新）。
- ゲート指標は `pips` 系に加え `JPY` 系にも対応している。
  - `min_test_total_jpy`
  - `min_test_jpy_per_hour`
  - `max_test_drawdown_jpy`

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate.yaml \
  --ticks-glob "logs/replay/USD_JPY/USD_JPY_ticks_202602*.jsonl" \
  --strict
```

`replay_quality_gate.py` でもシナリオ指定は可能です。  
`--scenarios` で直接指定するか、`config/*.yaml` の `replay.scenarios` に同等のリスト/CSVを記載します。  
`backend=exit_workers_groups` 時のみ `replay_exit_workers_groups.py` のシナリオ出力を使い、レポート `worker_results[worker]["scenarios"][<scenario>]` にシナリオ別要約が入り、`summary` は `all` シナリオを軸に既存互換で保持されます。

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate_main.yaml \
  --ticks-glob "logs/replay/USD_JPY/USD_JPY_ticks_202602*.jsonl,logs/archive/replay.*.dir/USD_JPY/USD_JPY_ticks_202602*.jsonl" \
  --strict
```

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate_ping5s_c.yaml \
  --strict
```

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate_ping5s_d.yaml \
  --strict
```

```bash
python scripts/replay_jpy_hour_sweep.py \
  --report tmp/replay_quality_gate_ping5s_c_strict/<run>/quality_gate_report.json \
  --thresholds "150,300,500,2000" \
  --target-jpy-per-hour 2000
```

- 出力:
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.json`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.md`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/commands.json`

## 取り残し（stuck）パターン抽出

- replay の約定履歴から「取り残されやすい局面（hour/side/reason）」を抽出する場合は
  `analysis/trade_counterfactual_worker.py` を replay 入力モードで使う。
- `--include-live-trades 0` で replay 専用評価に固定できる。

```bash
python -m analysis.trade_counterfactual_worker \
  --strategy-like "scalp_ping_5s_b_live%" \
  --include-live-trades 0 \
  --replay-json-globs "tmp/replay_quality_gate/*/runs/*/replay_exit_workers.json" \
  --stuck-hold-sec 120 \
  --stuck-loss-pips -0.30 \
  --stuck-reasons "time_stop,no_recovery,max_floating_loss,end_of_replay"
```

- 出力:
  - `logs/trade_counterfactual_latest.json` の `summary.stuck_trade_ratio`
  - `policy_hints.block_jst_hours` / `policy_hints.block_reasons`
  - `recommendations[*].stuck_rate`

## ワーカー運用（systemd）

- 定期実行は `quant-replay-quality-gate.service` + `quant-replay-quality-gate.timer` を使う。
- worker 本体は `analysis/replay_quality_gate_worker.py`。
  - `scripts/replay_quality_gate.py` を呼び出して walk-forward を実行。
  - 最新スナップショットを `logs/replay_quality_gate_latest.json` に保存。
  - 実行履歴を `logs/replay_quality_gate_history.jsonl` に追記。
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_ENABLED=1` のときは、
    replay 完了後に `analysis.trade_counterfactual_worker` を戦略ごとに連鎖実行し、
    `policy_hints.block_jst_hours` を `config/worker_reentry.yaml` へ自動反映する。
    - 対象戦略は既定で `failing_workers`（`REPLAY_QUALITY_GATE_AUTO_IMPROVE_SCOPE=failing`）。
    - replay入力は既定で当該 run の `runs/*/replay_exit_workers.json` と
      `runs/*/replay_exit_*_base.json` を自動探索する。
    - 反映内容と採用/非採用理由は `logs/replay_quality_gate_latest.json`
      の `auto_improve` に監査記録される。
  - `tmp/replay_quality_gate/<timestamp>` は `REPLAY_QUALITY_GATE_KEEP_RUNS` 件だけ保持。
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC`（既定 10800秒）で
    `worker_reentry` 反映の最小間隔を制御する。間隔内は
    `reentry_apply_cooldown` として解析のみ実施し、反映はスキップする。
- 主要設定は `ops/env/quant-replay-quality-gate.env` で管理する。
  - 既定は `config/replay_quality_gate_main.yaml`（archive 併用の walk-forward）を使う。
  - `REPLAY_QUALITY_GATE_CONFIG`
  - `REPLAY_QUALITY_GATE_TIMEOUT_SEC`
  - `REPLAY_QUALITY_GATE_STRICT`
  - `REPLAY_QUALITY_GATE_KEEP_RUNS`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_ENABLED`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_SCOPE`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_INCLUDE_LIVE_TRADES`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_TRADES`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MAX_BLOCK_HOURS`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_STATE_PATH`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_REENTRY`
- 導入例:

```bash
sudo bash scripts/install_trading_services.sh \
  --repo /home/tossaki/QuantRabbit \
  --units "quant-replay-quality-gate.service quant-replay-quality-gate.timer"
sudo systemctl enable --now quant-replay-quality-gate.timer
```
