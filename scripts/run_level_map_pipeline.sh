#!/usr/bin/env bash
# Batch pipeline: upload local candles to BigQuery and generate level_map.json to GCS.
# Intended for cron/systemd use. Relies on python scripts/upload_candles_to_bq.py and generate_level_map.py.

set -euo pipefail

PROJECT="${PROJECT:-${BQ_PROJECT:-${GOOGLE_CLOUD_PROJECT:-quantrabbit}}}"
DATASET="${DATASET:-${BQ_DATASET:-quantrabbit}}"
TABLE="${TABLE:-candles_m1}"
TIMEFRAME="${TIMEFRAME:-M1}"
INSTRUMENT="${INSTRUMENT:-USD_JPY}"
CANDLE_COUNT="${CANDLE_COUNT:-500}"
CANDLE_PRICE="${CANDLE_PRICE:-M}"
CANDLE_OUTPUT="${CANDLE_OUTPUT:-logs/oanda/candles_${TIMEFRAME}_latest.json}"
REFRESH_CANDLES="${REFRESH_CANDLES:-1}"
INPUTS="${INPUTS:-${CANDLE_OUTPUT}}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-90}"
GCS_BUCKET="${GCS_BUCKET:-${LEVEL_MAP_BUCKET:-}}"
GCS_OBJECT="${GCS_OBJECT:-analytics/level_map.json}"
JSON_OUT="${JSON_OUT:-/tmp/level_map.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

if [[ -z "${PROJECT}" ]]; then
  echo "PROJECT is required (set PROJECT or BQ_PROJECT or GOOGLE_CLOUD_PROJECT)" >&2
  exit 1
fi

log "Uploading candles -> BQ ${PROJECT}.${DATASET}.${TABLE} from ${INPUTS}"
if [[ "${REFRESH_CANDLES}" == "1" ]]; then
  log "Refreshing candles -> ${CANDLE_OUTPUT}"
  ${PYTHON_BIN} scripts/refresh_latest_candles.py \
    --instrument "${INSTRUMENT}" \
    --granularity "${TIMEFRAME}" \
    --count "${CANDLE_COUNT}" \
    --price "${CANDLE_PRICE}" \
    --output "${CANDLE_OUTPUT}"
fi

${PYTHON_BIN} scripts/upload_candles_to_bq.py \
  --project "${PROJECT}" \
  --dataset "${DATASET}" \
  --table "${TABLE}" \
  --inputs ${INPUTS} \
  --timeframe "${TIMEFRAME}" \
  --instrument "${INSTRUMENT}"

log "Generating level_map (lookback ${LOOKBACK_DAYS}d) -> ${JSON_OUT}"
GEN_CMD=(
  ${PYTHON_BIN} scripts/generate_level_map.py
  --project "${PROJECT}"
  --dataset "${DATASET}"
  --table "${TABLE}"
  --lookback-days "${LOOKBACK_DAYS}"
  --json-out "${JSON_OUT}"
)

if [[ -n "${GCS_BUCKET}" ]]; then
  GEN_CMD+=(--gcs-bucket "${GCS_BUCKET}" --gcs-object "${GCS_OBJECT}")
fi

"${GEN_CMD[@]}"
log "Done."
