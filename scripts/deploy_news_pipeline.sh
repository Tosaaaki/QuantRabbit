#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-quantrabbit}"
ACCOUNT="${ACCOUNT:-www.tosakiweb.net@gmail.com}"
REGIONS=("asia-northeast1" "us-central1")

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
  echo "[deploy_news_pipeline] gcloud が見つかりません。Cloud SDK をインストールし PATH を通すか、GCLOUD_BIN でパスを指定してください。" >&2
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

build_cmd=("$GCLOUD_BIN" "${impersonate_args[@]}")
deploy_cmd=("$GCLOUD_BIN" "${impersonate_args[@]}")

if $use_active_context; then
  echo "[deploy_news_pipeline] Active gcloud context matches ${ACCOUNT}/${PROJECT}."
else
  echo "[deploy_news_pipeline] 現在の gcloud アカウント/プロジェクトは ${current_account:-<unset>} / ${current_project:-<unset>} です。"
  echo "[deploy_news_pipeline] Cloud Build + Cloud Run を --project=$PROJECT で実行します。"
  build_cmd+=(--project "$PROJECT")
  deploy_cmd+=(--project "$PROJECT")
fi

echo "[deploy_news_pipeline] Cloud Build (fetch-news-runner)"
"${build_cmd[@]}" builds submit \
  --config=cloudrun/cloudbuild.fetch.yaml \
  --substitutions=_IMAGE=gcr.io/${PROJECT}/fetch-news-runner .

echo "[deploy_news_pipeline] Cloud Build (news-summarizer)"
"${build_cmd[@]}" builds submit \
  --config=cloudrun/cloudbuild.yaml \
  --substitutions=_IMAGE=gcr.io/${PROJECT}/news-summarizer .

echo "[deploy_news_pipeline] Deploying to Cloud Run regions: ${REGIONS[*]}"
for r in "${REGIONS[@]}"; do
  echo "  -> fetch-news-runner (${r})"
  "${deploy_cmd[@]}" run deploy fetch-news-runner \
    --image gcr.io/${PROJECT}/fetch-news-runner \
    --region=${r} --platform=managed \
    --service-account=fetch-news-runner-sa@${PROJECT}.iam.gserviceaccount.com \
    --set-env-vars=BUCKET=quantrabbit-fx-news --quiet

  echo "  -> news-summarizer (${r})"
  "${deploy_cmd[@]}" run deploy news-summarizer \
    --image gcr.io/${PROJECT}/news-summarizer \
    --region=${r} --platform=managed \
    --service-account=news-summarizer-sa@${PROJECT}.iam.gserviceaccount.com \
    --set-env-vars=BUCKET=quantrabbit-fx-news --quiet
done

echo "[deploy_news_pipeline] 完了"
