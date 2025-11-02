#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value core/project)}
REGION=${REGION:-asia-northeast1}
SERVICE=${SERVICE:-bq-exporter}

echo "Deploying Cloud Run service: ${SERVICE} in ${PROJECT}/${REGION}"
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --source . \
  --no-allow-unauthenticated \
  --set-env-vars=BQ_PROJECT=${BQ_PROJECT:-${PROJECT}},BQ_DATASET=${BQ_DATASET:-quantrabbit},BQ_TRADES_TABLE=${BQ_TRADES_TABLE:-trades_raw},BQ_FEATURE_TABLE=${BQ_FEATURE_TABLE:-trades_daily_features},BQ_LOCATION=${BQ_LOCATION:-US},BQ_MAX_EXPORT=${BQ_MAX_EXPORT:-5000} \
  --set-env-vars=SQLITE_PATH=${SQLITE_PATH:-logs/trades.db},BQ_SYNC_STATE=${BQ_SYNC_STATE:-logs/bq_sync_state.json},GUNICORN_APP=cloudrun.bq_export_service:app

URL=$(gcloud run services describe "${SERVICE}" --project="${PROJECT}" --region="${REGION}" --format='value(status.url)')
echo "Deployed: ${URL}"

if [[ "${CREATE_SCHEDULER:-1}" == "1" ]]; then
  JOB_NAME=${JOB_NAME:-bq-exporter-schedule}
  SVC_EMAIL=${SVC_EMAIL:-$(gcloud iam service-accounts list --format='value(email)' --filter='compute@developer.gserviceaccount.com' | head -n1)}
  echo "Creating/Updating Cloud Scheduler job: ${JOB_NAME} (every 10 minutes)"
  gcloud scheduler jobs create http "${JOB_NAME}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --schedule="*/10 * * * *" \
    --uri="${URL}/export" \
    --http-method=GET \
    --oidc-service-account-email="${SVC_EMAIL}" \
    --oidc-token-audience="${URL}" || \
  gcloud scheduler jobs update http "${JOB_NAME}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --schedule="*/10 * * * *" \
    --uri="${URL}/export" \
    --http-method=GET \
    --oidc-service-account-email="${SVC_EMAIL}" \
    --oidc-token-audience="${URL}"
fi

echo "Done."
