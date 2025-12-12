#!/usr/bin/env bash
set -euo pipefail

# Run export + online tuner in shadow mode.

cd "$(dirname "$0")/.."

LOG=${TUNER_LOGS_GLOB:-tmp/exit_eval_live.csv}
DB=${TUNER_DB_PATH:-logs/trades.db}
LIMIT=${TUNER_EXPORT_LIMIT:-4000}
WINDOW=${TUNER_WINDOW_MINUTES:-15}

export PYTHONPATH=.

if [ ! -f "$DB" ]; then
  echo "[tuner-shadow] missing DB: $DB" >&2
  exit 1
fi

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 scripts/export_exit_eval.py --db "$DB" --out "$LOG" --limit "$LIMIT"
.venv/bin/python scripts/run_online_tuner.py \
  --logs-glob "$LOG" \
  --presets config/tuning_presets.yaml \
  --overrides-out config/tuning_overrides.yaml \
  --minutes "$WINDOW" \
  --shadow
