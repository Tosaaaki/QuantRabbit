#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${DB_PATH:-logs/trades.db}"
INSTRUMENT="${INSTRUMENT:-USD_JPY}"
DAYS="${DAYS:-180}"
TRAIN_DAYS="${TRAIN_DAYS:-28}"
TEST_DAYS="${TEST_DAYS:-7}"
STEP_DAYS="${STEP_DAYS:-7}"
MIN_TRAIN_TRADES="${MIN_TRAIN_TRADES:-15}"
MIN_TEST_TRADES="${MIN_TEST_TRADES:-8}"
METRIC="${METRIC:-pf}"
OUT_DIR="${OUT_DIR:-logs/reports/wfo_overfit}"
PY="${PYTHON:-python3}"

mkdir -p "$OUT_DIR"

$PY -m analytics.wfo_overfit_report \
  --db "$DB_PATH" \
  --instrument "$INSTRUMENT" \
  --days "$DAYS" \
  --train-days "$TRAIN_DAYS" \
  --test-days "$TEST_DAYS" \
  --step-days "$STEP_DAYS" \
  --min-train-trades "$MIN_TRAIN_TRADES" \
  --min-test-trades "$MIN_TEST_TRADES" \
  --metric "$METRIC" \
  --out-json "$OUT_DIR/latest.json" \
  --out-md "$OUT_DIR/latest.md"
