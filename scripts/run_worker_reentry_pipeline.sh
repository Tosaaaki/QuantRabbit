#!/usr/bin/env bash
set -euo pipefail

DAYS="${DAYS:-14}"
MIN_TRADES="${MIN_TRADES:-30}"
BLOCK_HOUR_TRADES="${BLOCK_HOUR_TRADES:-20}"
BLOCK_HOUR_MFE_MAX="${BLOCK_HOUR_MFE_MAX:-1.2}"
BLOCK_HOUR_MAE_MAX="${BLOCK_HOUR_MAE_MAX:-1.2}"
BLOCK_HOUR_TOP="${BLOCK_HOUR_TOP:-4}"
BLOCK_HOUR_WINDOW="${BLOCK_HOUR_WINDOW:-}"
BLOCK_HOURS_SCOPE="${BLOCK_HOURS_SCOPE:-global}"
OUT_DIR="${OUT_DIR:-logs/reports/worker_return_wait}"
PY="${PYTHON:-python3}"
APPLY=0
REQUIRE_BLOCK_HOURS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)
      DAYS="$2"
      shift 2
      ;;
    --min-trades)
      MIN_TRADES="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --block-hour-trades)
      BLOCK_HOUR_TRADES="$2"
      shift 2
      ;;
    --block-hour-mfe-max)
      BLOCK_HOUR_MFE_MAX="$2"
      shift 2
      ;;
    --block-hour-mae-max)
      BLOCK_HOUR_MAE_MAX="$2"
      shift 2
      ;;
    --block-hour-top)
      BLOCK_HOUR_TOP="$2"
      shift 2
      ;;
    --block-hour-window)
      BLOCK_HOUR_WINDOW="$2"
      shift 2
      ;;
    --block-hours-scope)
      BLOCK_HOURS_SCOPE="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift 1
      ;;
    --require-block-hours)
      REQUIRE_BLOCK_HOURS=1
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR"

$PY -m analytics.worker_return_wait_report \
  --days "$DAYS" \
  --min-trades "$MIN_TRADES" \
  --block-hour-trades "$BLOCK_HOUR_TRADES" \
  --block-hour-mfe-max "$BLOCK_HOUR_MFE_MAX" \
  --block-hour-mae-max "$BLOCK_HOUR_MAE_MAX" \
  --block-hour-top "$BLOCK_HOUR_TOP" \
  --block-hour-window "$BLOCK_HOUR_WINDOW" \
  --block-hours-scope "$BLOCK_HOURS_SCOPE" \
  --out-json "$OUT_DIR/latest.json" \
  --out-yaml "$OUT_DIR/worker_reentry.yaml"

if [[ "$APPLY" -eq 1 ]]; then
  $PY utils/yaml_merge.py \
    --base config/worker_reentry.yaml \
    --over "$OUT_DIR/worker_reentry.yaml" \
    --out config/worker_reentry.yaml
fi

if [[ "$REQUIRE_BLOCK_HOURS" -eq 1 ]]; then
  if ! awk '
    /^defaults:/ {in_defaults=1; next}
    /^strategies:/ {in_defaults=0}
    in_defaults && /block_jst_hours:/ {
      if ($0 ~ /\[[^]]*[0-9]/) ok=1
    }
    END {
      if (ok) exit 0
      exit 1
    }
  ' "$OUT_DIR/worker_reentry.yaml"; then
    echo "[worker_reentry] block_jst_hours is empty in defaults; aborting (--require-block-hours)." >&2
    exit 2
  fi
fi
