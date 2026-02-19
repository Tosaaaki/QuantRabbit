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
  --workers impulse_break_s5,impulse_momentum_s5,impulse_retest_s5,pullback_s5 \
  --no-hard-sl \
  --exclude-end-of-replay \
  --out-dir tmp/replay_exit_workers_groups_YYYYMM_all
```

## 補足
- `--ticks` は bid/ask を含む JSONL を前提にする。
- 窓を変える場合は `--ticks` の入力だけ差し替える。
- ルールを変えない限り、比較は `summary_all.json` 同士で行う。

## 内部精度ゲート（walk-forward）

- リプレイ出力に対して in-sample / out-of-sample を自動判定する場合は `scripts/replay_quality_gate.py` を使う。
- 実行バックエンドは `scripts/replay_exit_workers_groups.py` で、標準フラグ（`--no-hard-sl` / `--exclude-end-of-replay`）を既定で適用する。
- 閾値は `config/replay_quality_gate.yaml` の `gates.default` と `gates.workers` で管理する。

```bash
python scripts/replay_quality_gate.py \
  --config config/replay_quality_gate.yaml \
  --ticks-glob "logs/replay/USD_JPY/USD_JPY_ticks_202602*.jsonl" \
  --strict
```

- 出力:
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.json`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/quality_gate_report.md`
  - `tmp/replay_quality_gate/<UTC_TIMESTAMP>/commands.json`
