#!/usr/bin/env bash
set -euo pipefail

# Usage: PROJECT_ID=quantrabbit REGION=asia-northeast1 ./scripts/update_cloud_run_env.sh fx-trader fx-exit-manager

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-asia-northeast1}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: PROJECT_ID=... REGION=... $0 <service> [<service> ...]" >&2
  exit 2
fi

for SVC in "$@"; do
  echo "Updating env for Cloud Run service: ${SVC}"
  gcloud run services update "${SVC}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --update-env-vars=TRADE_REPOSITORY_PROJECT=${PROJECT_ID},TRADE_REPOSITORY_COLLECTION=trades,TRADE_REPOSITORY_CACHE_TTL=45,TRADE_REPOSITORY_MAX_FETCH=2000
done

echo "Done. Verify with: gcloud run services describe <service> --project=${PROJECT_ID} --region=${REGION} --format='get(spec.template.spec.containers[0].env)'"

