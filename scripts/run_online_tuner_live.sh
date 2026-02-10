#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/tossaki/QuantRabbit}"
PY_BIN="${PY_BIN:-$REPO_DIR/.venv/bin/python}"
WINDOW_MIN="${TUNER_WINDOW_MINUTES:-15}"
EXPORT_LIMIT="${TUNER_EXPORT_LIMIT:-4000}"

# NOTE: These outputs are written by online tuning. Keep them outside tracked paths
# to avoid dirty git worktrees on the VM (which can silently break deploy pulls).
PRESETS_PATH="${TUNING_PRESETS_PATH:-config/tuning_presets.yaml}"
OVERRIDES_OUT="${TUNING_OVERRIDES_PATH:-config/tuning_overrides.yaml}"
OVERLAY_OUT="${TUNING_OVERLAY_PATH:-config/tuning_overlay.yaml}"
HISTORY_DIR="${TUNING_HISTORY_DIR:-config/tuning_history}"
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

mkdir -p "$(dirname "$LOGS_CSV")"
mkdir -p "$(dirname "$OVERRIDES_OUT")"
mkdir -p "$(dirname "$OVERLAY_OUT")"
mkdir -p "$HISTORY_DIR"

if [[ ! -f "$PRESETS_PATH" ]]; then
  echo "[online_tuner] missing presets: $PRESETS_PATH" >&2
  exit 0
fi

PYTHONPATH=. "$PY_BIN" scripts/export_exit_eval.py \
  --db logs/trades.db \
  --out "$LOGS_CSV" \
  --limit "$EXPORT_LIMIT"

PYTHONPATH=. "$PY_BIN" scripts/run_online_tuner.py \
  --logs-glob "$LOGS_CSV" \
  --presets "$PRESETS_PATH" \
  --overrides-out "$OVERRIDES_OUT" \
  --history-dir "$HISTORY_DIR" \
  --minutes "$WINDOW_MIN"

PYTHONPATH=. "$PY_BIN" scripts/apply_override.py \
  --base "$PRESETS_PATH" \
  --over "$OVERRIDES_OUT" \
  --out "$OVERLAY_OUT"
