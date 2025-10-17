#!/usr/bin/env bash
set -euo pipefail

PROJECT="quantrabbit"
ACCOUNT="www.tosakiweb.net@gmail.com"
IMAGE="gcr.io/${PROJECT}/news-summarizer"
REGION="asia-northeast1"
SERVICE="fx-trader"
CLOUD_BUILD_CONFIG="cloudrun/cloudbuild.yaml"

# Resolve gcloud path for non-interactive shells (PATH may not include SDK)
GCLOUD_BIN="${GCLOUD_BIN:-}"
if [ -z "$GCLOUD_BIN" ]; then
  if command -v gcloud >/dev/null 2>&1; then
    GCLOUD_BIN="$(command -v gcloud)"
  else
    for p in \
      "$HOME/google-cloud-sdk/bin/gcloud" \
      "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud" \
      "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud"; do
      [ -x "$p" ] && GCLOUD_BIN="$p" && break
    done
  fi
fi

if [ -z "$GCLOUD_BIN" ]; then
  echo "[deploy_fx_trader] gcloud が見つかりません。Cloud SDK をインストールし PATH を通すか、GCLOUD_BIN でパスを指定してください。" >&2
  exit 127
fi

current_account="$($GCLOUD_BIN config get-value core/account 2>/dev/null || true)"
current_project="$($GCLOUD_BIN config get-value core/project 2>/dev/null || true)"

use_active_context=false
if [ "$current_account" = "$ACCOUNT" ] && [ "$current_project" = "$PROJECT" ]; then
  use_active_context=true
fi

impersonate_args=()
if [ -n "${DEPLOY_IMPERSONATE_SA:-}" ]; then
  impersonate_args=(--impersonate-service-account "$DEPLOY_IMPERSONATE_SA")
fi

if $use_active_context; then
  echo "[deploy_fx_trader] Active gcloud context matches ${ACCOUNT}/${PROJECT}."
  echo "[deploy_fx_trader] Cloud Build 実行 (image=$IMAGE)"
  "$GCLOUD_BIN" builds submit \
    --project "$PROJECT" \
    --account "$ACCOUNT" \
    --config "$CLOUD_BUILD_CONFIG" \
    --substitutions _IMAGE="$IMAGE"

  echo "[deploy_fx_trader] Cloud Run デプロイ (service=$SERVICE, region=$REGION)"
  "$GCLOUD_BIN" run deploy "$SERVICE" \
    --account "$ACCOUNT" \
    --project "$PROJECT" \
    --region "$REGION" \
    --image "$IMAGE" \
    --platform managed \
    --set-env-vars GUNICORN_APP=cloudrun.trader_service:app
else
  echo "[deploy_fx_trader] 現在の gcloud アカウント/プロジェクトは ${current_account:-<unset>} / ${current_project:-<unset>} です。"
  echo "[deploy_fx_trader] Cloud Build + Cloud Run を --project=$PROJECT で実行します。"

  cmd=("$GCLOUD_BIN")
  if [ ${#impersonate_args[@]} -gt 0 ]; then
    cmd+=("${impersonate_args[@]}")
  fi
  cmd+=(--account "$ACCOUNT" --project "$PROJECT" builds submit --config "$CLOUD_BUILD_CONFIG" --substitutions _IMAGE="$IMAGE")
  "${cmd[@]}"

  cmd=("$GCLOUD_BIN")
  if [ ${#impersonate_args[@]} -gt 0 ]; then
    cmd+=("${impersonate_args[@]}")
  fi
  cmd+=(--account "$ACCOUNT" --project "$PROJECT" run deploy "$SERVICE" --region "$REGION" --image "$IMAGE" --platform managed --set-env-vars GUNICORN_APP=cloudrun.trader_service:app)
  "${cmd[@]}"
fi

echo "[deploy_fx_trader] 完了"
