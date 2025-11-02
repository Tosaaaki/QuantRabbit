#!/usr/bin/env bash
set -euo pipefail

# Usage: PROJECT_ID=quantrabbit ./scripts/setup_firestore_indexes.sh

: "${PROJECT_ID:?Set PROJECT_ID}"

echo "Creating composite index: trades(state ASC, pocket ASC, close_time DESC)"
gcloud firestore indexes composite create \
  --project="${PROJECT_ID}" \
  --collection-group=trades \
  --field-config="fieldPath=state,order=ASCENDING" \
  --field-config="fieldPath=pocket,order=ASCENDING" \
  --field-config="fieldPath=close_time,order=DESCENDING" || true

echo "Creating composite index: trades(state ASC, pocket ASC, entry_time DESC)"
gcloud firestore indexes composite create \
  --project="${PROJECT_ID}" \
  --collection-group=trades \
  --field-config="fieldPath=state,order=ASCENDING" \
  --field-config="fieldPath=pocket,order=ASCENDING" \
  --field-config="fieldPath=entry_time,order=DESCENDING" || true

echo "Done. You can list with: gcloud firestore indexes composite list --project=${PROJECT_ID}"

