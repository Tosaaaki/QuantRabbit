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

## 例

```bash
python scripts/replay_exit_workers_groups.py \
  --ticks tmp/ticks_USDJPY_YYYYMM_all.jsonl \
  --workers session_open \
  --no-hard-sl \
  --exclude-end-of-replay \
  --out-dir tmp/replay_exit_workers_groups_YYYYMM_all
```

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
- `exit_workers_main` は `replay.intraday_start_utc` / `replay.intraday_end_utc` を指定すると、
  tick ファイル名の日付（`YYYYMMDD`）に対して日内 UTC 時間帯を自動適用できる。
- `config/replay_quality_gate_main.yaml` の既定は intraday 無効（空文字）として扱い、
  フルデイ再生で品質ゲートを判定する。
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

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate.yaml \
  --ticks-glob "logs/replay/USD_JPY/USD_JPY_ticks_202602*.jsonl" \
  --strict
```

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate_main.yaml \
  --ticks-glob "logs/replay/USD_JPY/USD_JPY_ticks_202602*.jsonl" \
  --strict
```

- 出力:
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.json`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.md`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/commands.json`
