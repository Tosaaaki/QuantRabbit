#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL=${EXIT_MANAGER_URL:-}
PROJECT=${GCP_PROJECT:-quantrabbit}
REGION=${GCP_REGION:-asia-northeast1}
GCLOUD_BIN="${GCLOUD_BIN:-}"

if [ -z "$SERVICE_URL" ]; then
  if [ -z "$GCLOUD_BIN" ]; then
    if command -v gcloud >/dev/null 2>&1; then
      GCLOUD_BIN="$(command -v gcloud)"
    elif [ -x "$HOME/google-cloud-sdk/bin/gcloud" ]; then
      GCLOUD_BIN="$HOME/google-cloud-sdk/bin/gcloud"
    fi
  fi
  if [ -n "$GCLOUD_BIN" ]; then
    SERVICE_URL=$($GCLOUD_BIN run services describe fx-exit-manager --project "$PROJECT" --region "$REGION" --format="value(status.url)" 2>/dev/null || true)
  fi
fi

if [ -z "$SERVICE_URL" ]; then
  echo "exit-manager の URL が分かりません。EXIT_MANAGER_URL で指定してください" >&2
  exit 1
fi

curl -sSL "$SERVICE_URL" || {
  echo "exit manager call failed" >&2
  exit 1
}
