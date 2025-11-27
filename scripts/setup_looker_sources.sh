#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit - Looker Studio data sources bootstrap
# - Creates service account for Looker Studio
# - Grants GCS (read) and BigQuery (viewer + jobUser)
# - Ensures GCS bucket/object for realtime UI JSON
# - Ensures BigQuery dataset and optional views
#
# Usage (env-first):
#   GCP_PROJECT=quantrabbit \
#   UI_BUCKET=fx-ui-realtime \
#   BQ_DATASET=quantrabbit \
#   BQ_LOCATION=asia-northeast1 \
#   UI_SA_EMAIL=ui-dashboard-sa@quantrabbit.iam.gserviceaccount.com \
#   ./scripts/setup_looker_sources.sh
#
# Optional:
#   KEY_OUT=./ui-dashboard-sa.json  # Defaults to ./ui-dashboard-sa.json

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

# Ensure gcloud/bq are on PATH for non-interactive shells
if ! command -v gcloud >/dev/null 2>&1; then
  SDK_BIN="$HOME/google-cloud-sdk/bin"
  if [[ -d "$SDK_BIN" ]]; then
    export PATH="$SDK_BIN:$PATH"
  fi
fi

command -v gcloud >/dev/null 2>&1 || { echo "gcloud not found"; exit 1; }
command -v bq >/dev/null 2>&1 || { echo "bq not found"; exit 1; }

# ------------ helpers ------------
function _toml() {
  local key="$1"; local default="${2:-}"
  local path="${ROOT}/config/env.toml"
  if [[ ! -f "$path" ]]; then path="${ROOT}/config/env.example.toml"; fi
  python3 - <<PY 2>/dev/null || true
import os
try:
  try:
    import tomllib
  except Exception:
    import tomli as tomllib  # type: ignore
  with open("$path","rb") as f:
    cfg=tomllib.load(f)
  print(cfg.get("${key}", "${default}"))
except Exception:
  print("${default}")
PY
}

PROJECT="${GCP_PROJECT:-$(_toml gcp_project_id)}"
LOCATION="${BQ_LOCATION:-$(_toml gcp_location "asia-northeast1")}"
DATASET="${BQ_DATASET:-$(_toml BQ_DATASET "quantrabbit")}"
BUCKET="${UI_BUCKET:-$(_toml ui_bucket_name "fx-ui-realtime")}"
OBJECT_PATH="$(_toml ui_state_object_path "realtime/ui_state.json")"

# Fallback to gcloud default when tomllib isn't available or placeholder remains
if [[ -z "$PROJECT" || "$PROJECT" == "your-project-id" ]]; then
  GCLOUD_PROJ=$(gcloud config get-value project 2>/dev/null || true)
  if [[ -n "$GCLOUD_PROJ" ]]; then PROJECT="$GCLOUD_PROJ"; fi
fi

if [[ -z "${UI_SA_EMAIL:-}" ]]; then
  UI_SA_EMAIL="ui-dashboard-sa@${PROJECT}.iam.gserviceaccount.com"
fi
KEY_OUT="${KEY_OUT:-${ROOT}/ui-dashboard-sa.json}"

echo "[Plan] project=${PROJECT} dataset=${DATASET} location=${LOCATION} bucket=gs://${BUCKET}/${OBJECT_PATH}"
echo "[Plan] serviceAccount=${UI_SA_EMAIL} key=${KEY_OUT}"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: GCP project is empty. Set GCP_PROJECT or config/env(.example).toml" >&2
  exit 2
fi

# ------------ Service Account ------------
SA_NAME="${UI_SA_EMAIL%@*}"
SA_SHORT="${SA_NAME##*/}"

if ! gcloud iam service-accounts describe "$UI_SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  echo "[IAM] Creating service account: $UI_SA_EMAIL"
  gcloud iam service-accounts create "$SA_SHORT" \
    --project "$PROJECT" \
    --display-name "Looker Studio dashboard SA"
else
  echo "[IAM] Service account exists: $UI_SA_EMAIL"
fi

# ------------ IAM Bindings ------------
echo "[IAM] Grant BigQuery roles (dataViewer, jobUser)"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:${UI_SA_EMAIL}" \
  --role roles/bigquery.dataViewer >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:${UI_SA_EMAIL}" \
  --role roles/bigquery.jobUser >/dev/null

# ------------ Bucket ensure + permissions ------------
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "[GCS] Creating bucket gs://${BUCKET} (location=${LOCATION})"
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="$PROJECT" \
    --location="$LOCATION" \
    --uniform-bucket-level-access >/dev/null
else
  echo "[GCS] Bucket exists: gs://${BUCKET}"
fi

echo "[GCS] Grant objectViewer to ${UI_SA_EMAIL}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${UI_SA_EMAIL}" \
  --role roles/storage.objectViewer >/dev/null

# Ensure placeholder JSON
echo "{}" | gcloud storage cp - "gs://${BUCKET}/${OBJECT_PATH}" >/dev/null || true
echo "[GCS] Ensured placeholder at gs://${BUCKET}/${OBJECT_PATH}"

# ------------ BigQuery dataset ------------
DATASET_LOC=""
if DATASET_JSON=$(bq --project_id "$PROJECT" --format=prettyjson show "${PROJECT}:${DATASET}" 2>/dev/null); then
  DATASET_LOC=$(python3 - <<PY 2>/dev/null
import json,sys
try:
  j=json.loads(sys.stdin.read()); print(j.get('location',''))
except Exception:
  print('')
PY
  <<<"$DATASET_JSON")
  echo "[BQ] Dataset exists: ${PROJECT}:${DATASET} (location=${DATASET_LOC:-unknown})"
else
  echo "[BQ] Creating dataset ${PROJECT}:${DATASET} (${LOCATION})"
  bq --location="$LOCATION" --project_id "$PROJECT" mk -d "$DATASET" >/dev/null
  DATASET_LOC="$LOCATION"
fi

# Views (only if base table exists)
if bq --project_id "$PROJECT" show "${PROJECT}:${DATASET}.trades_raw" >/dev/null 2>&1; then
  echo "[BQ] Creating views trades_recent_view / trades_daily_features"
  # Use dataset location if known; otherwise omit --location to avoid mismatch
  BQ_LOC_ARG=""
  if [[ -n "${DATASET_LOC:-}" ]]; then BQ_LOC_ARG="--location=${DATASET_LOC}"; fi
  bq --project_id "$PROJECT" ${BQ_LOC_ARG} query --use_legacy_sql=false <<SQL >/dev/null
CREATE OR REPLACE VIEW ${PROJECT}.${DATASET}.trades_recent_view AS
SELECT
  ticket_id,
  entry_time,
  close_time,
  pocket,
  instrument,
  units,
  entry_price,
  close_price,
  pl_pips,
  realized_pl,
  state,
  close_reason
FROM ${PROJECT}.${DATASET}.trades_raw
WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY close_time DESC;
SQL

  bq --project_id "$PROJECT" ${BQ_LOC_ARG} query --use_legacy_sql=false <<SQL >/dev/null
CREATE OR REPLACE VIEW ${PROJECT}.${DATASET}.trades_daily_features AS
SELECT
  DATE(close_time) AS day,
  COUNTIF(close_time IS NOT NULL) AS trades,
  SUM(pl_pips) AS pl_pips,
  AVG(pl_pips) AS avg_pl_pips,
  SAFE_DIVIDE(SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END), COUNT(*)) AS win_rate,
  SUM(CASE WHEN pocket='micro' THEN pl_pips ELSE 0 END) AS pl_pips_micro,
  SUM(CASE WHEN pocket='macro' THEN pl_pips ELSE 0 END) AS pl_pips_macro
FROM ${PROJECT}.${DATASET}.trades_raw
WHERE close_time IS NOT NULL
GROUP BY day
ORDER BY day DESC;
SQL
else
  echo "[BQ] Base table ${PROJECT}:${DATASET}.trades_raw not found. Views skipped (run scripts/run_sync_pipeline.py first)."
fi

# ------------ Service account key (local file) ------------
if [[ -f "$KEY_OUT" ]]; then
  echo "[IAM] Key file already exists: $KEY_OUT (skipping create)"
else
  echo "[IAM] Creating key file: $KEY_OUT"
  gcloud iam service-accounts keys create "$KEY_OUT" \
    --iam-account "$UI_SA_EMAIL" \
    --project "$PROJECT" >/dev/null
  echo "[SECURITY] Keep this key safe and DO NOT commit it. (.gitignore updated to ignore common key patterns)"
fi

echo "[Done] Looker Studio data sources are ready."
echo "       - GCS: gs://${BUCKET}/${OBJECT_PATH}"
echo "       - BigQuery dataset: ${PROJECT}:${DATASET} (views may require data sync)"
