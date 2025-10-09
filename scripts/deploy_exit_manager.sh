#!/usr/bin/env bash
set -euo pipefail

PROJECT="quantrabbit"
REGION="asia-northeast1"
IMAGE="gcr.io/${PROJECT}/exit-manager"
SERVICE="fx-exit-manager"
CLOUD_BUILD_CONFIG="cloudrun/cloudbuild.exit_manager.yaml"
GCLOUD_BIN="${GCLOUD_BIN:-}"

if [ -z "$GCLOUD_BIN" ]; then
  if command -v gcloud >/dev/null 2>&1; then
    GCLOUD_BIN="$(command -v gcloud)"
  elif [ -x "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD_BIN="$HOME/google-cloud-sdk/bin/gcloud"
  else
    echo "gcloud CLI が見つかりません。PATHを設定するかGCLOUD_BINを指定してください" >&2
    exit 1
  fi
fi

current_project="$($GCLOUD_BIN config get-value core/project 2>/dev/null || true)"
if [ "$current_project" != "$PROJECT" ]; then
  echo "[deploy_exit_manager] gcloud config project を ${PROJECT} に切り替えます"
  $GCLOUD_BIN config set project "$PROJECT" >/dev/null
fi

# Build image
$GCLOUD_BIN builds submit \
  --config "$CLOUD_BUILD_CONFIG" \
  --substitutions _IMAGE="$IMAGE"

# Deploy Cloud Run service
$GCLOUD_BIN run deploy "$SERVICE" \
  --region "$REGION" \
  --image "$IMAGE" \
  --platform managed \
  --allow-unauthenticated \
  --concurrency 1 \
  --memory 512Mi \
  --timeout 300 \
  --set-env-vars "INSTRUMENT=USD_JPY" \
  --set-env-vars "EXIT_GPT_TRIGGER_PIPS=${EXIT_GPT_TRIGGER_PIPS:-8.0}" \
  --set-env-vars "EXIT_GPT_TRIGGER_MULT_MICRO=${EXIT_GPT_TRIGGER_MULT_MICRO:-0.8}" \
  --set-env-vars "EXIT_GPT_TRIGGER_MULT_MACRO=${EXIT_GPT_TRIGGER_MULT_MACRO:-1.0}" \
  --set-env-vars "EXIT_GPT_MAX_WAIT_MIN=${EXIT_GPT_MAX_WAIT_MIN:-45}" \
  --set-env-vars "EXIT_BE_BUFFER_PIPS=${EXIT_BE_BUFFER_PIPS:-2.0}"

echo "[deploy_exit_manager] Service deployed -> https://${SERVICE}-${PROJECT}.asia-northeast1.run.app"
