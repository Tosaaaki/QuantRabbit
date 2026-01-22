#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/tossaki/QuantRabbit}"
PY_BIN="${PY_BIN:-$REPO_DIR/.venv/bin/python}"
WINDOW_MIN="${TUNER_WINDOW_MINUTES:-15}"
EXPORT_LIMIT="${TUNER_EXPORT_LIMIT:-4000}"
LOGS_CSV="${TUNER_LOGS_CSV:-tmp/exit_eval_live.csv}"

cd "$REPO_DIR"

if [[ ! -f "logs/trades.db" ]]; then
  echo "[online_tuner] missing logs/trades.db" >&2
  exit 0
fi

if [[ -n "${TUNER_ENABLE-}" ]]; then
  case "${TUNER_ENABLE,,}" in
    0|false|no|off) echo "[online_tuner] disabled (TUNER_ENABLE=$TUNER_ENABLE)"; exit 0 ;;
  esac
fi

mkdir -p tmp

PYTHONPATH=. "$PY_BIN" scripts/export_exit_eval.py \
  --db logs/trades.db \
  --out "$LOGS_CSV" \
  --limit "$EXPORT_LIMIT"

PYTHONPATH=. "$PY_BIN" scripts/run_online_tuner.py \
  --logs-glob "$LOGS_CSV" \
  --presets config/tuning_presets.yaml \
  --overrides-out config/tuning_overrides.yaml \
  --minutes "$WINDOW_MIN"

PYTHONPATH=. "$PY_BIN" scripts/apply_override.py \
  --base config/tuning_presets.yaml \
  --over config/tuning_overrides.yaml \
  --out config/tuning_overlay.yaml
