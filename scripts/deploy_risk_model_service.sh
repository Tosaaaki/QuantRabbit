#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value core/project)}
REGION=${REGION:-asia-northeast1}
SERVICE=${SERVICE:-risk-model}

echo "Deploying Cloud Run service: ${SERVICE} in ${PROJECT}/${REGION}"
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --source . \
  --no-allow-unauthenticated \
  --set-env-vars=RISK_MODEL_PROJECT=${RISK_MODEL_PROJECT:-${PROJECT}},RISK_MODEL_DATASET=${RISK_MODEL_DATASET:-quantrabbit},RISK_MODEL_FEATURE_TABLE=${RISK_MODEL_FEATURE_TABLE:-trades_daily_features},RISK_MODEL_ID=${RISK_MODEL_ID:-strategy_risk_model},RISK_MODEL_MIN_TRADES=${RISK_MODEL_MIN_TRADES:-5},RISK_MODEL_LOOKBACK_DAYS=${RISK_MODEL_LOOKBACK_DAYS:-120},RISK_MODEL_STATE=${RISK_MODEL_STATE:-logs/risk_scores.json},RISK_PUBSUB_TOPIC=${RISK_PUBSUB_TOPIC:-risk-model-updates},GUNICORN_APP=cloudrun.risk_model_service:app

URL=$(gcloud run services describe "${SERVICE}" --project="${PROJECT}" --region="${REGION}" --format='value(status.url)')
echo "Deployed: ${URL}"

if [[ "${CREATE_SCHEDULER:-1}" == "1" ]]; then
  JOB_NAME=${JOB_NAME:-risk-model-run}
  SCHEDULE_CRON=${SCHEDULE_CRON:-"*/30 * * * *"}
  METHOD=${METHOD:-POST}
  SVC_EMAIL=${SVC_EMAIL:-$(gcloud iam service-accounts list --format='value(email)' --filter='compute@developer.gserviceaccount.com' | head -n1)}
  echo "Creating/Updating Cloud Scheduler job: ${JOB_NAME} (${SCHEDULE_CRON})"
  gcloud scheduler jobs create http "${JOB_NAME}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --schedule="${SCHEDULE_CRON}" \
    --uri="${URL}/run?train=false" \
    --http-method="${METHOD}" \
    --oidc-service-account-email="${SVC_EMAIL}" \
    --oidc-token-audience="${URL}" || \
  gcloud scheduler jobs update http "${JOB_NAME}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --schedule="${SCHEDULE_CRON}" \
    --uri="${URL}/run?train=false" \
    --http-method="${METHOD}" \
    --oidc-service-account-email="${SVC_EMAIL}" \
    --oidc-token-audience="${URL}"
fi

echo "Done."
