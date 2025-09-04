#!/usr/bin/env bash
set -euo pipefail

# Usage: PROJECT_ID=your-proj REGION=asia-northeast1 ./scripts/deploy_all.sh

PROJECT_ID="${PROJECT_ID:-${TF_VAR_project_id:-}}"
REGION="${REGION:-${TF_VAR_region:-asia-northeast1}}"
IMAGE="gcr.io/${PROJECT_ID}/news-summarizer"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "[ERR] PROJECT_ID env is required" >&2
  exit 1
fi

check_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERR] '$1' not found" >&2; exit 1; }; }
check_cmd gcloud
check_cmd terraform

echo "[INFO] Building image via Cloud Build: ${IMAGE}"
gcloud builds submit --config cloudrun/cloudbuild.yaml --substitutions=_IMAGE="${IMAGE}"

pushd infra/terraform >/dev/null
  echo "[INFO] Terraform init"
  terraform init -input=false
  echo "[INFO] Terraform apply (project=${PROJECT_ID}, region=${REGION})"
  TF_VAR_project_id="${PROJECT_ID}" TF_VAR_region="${REGION}" terraform apply -auto-approve
popd >/dev/null

echo "[INFO] Deployment completed"

