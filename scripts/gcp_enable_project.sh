#!/usr/bin/env bash
set -euo pipefail

# Enable a GCP project for Quantrabbit workflows in one go.
# - Links billing (optional)
# - Grants IAM to a user (Owner + OS Login + IAP)
# - Enables required APIs
# - Turns on OS Login at project metadata
#
# Usage (dry-run by default):
#   scripts/gcp_enable_project.sh -p quantrabbit -u www.tosakiweb.net@gmail.com -b BILLING_ID
#   scripts/gcp_enable_project.sh -p quantrabbit -u ... -b BILLING_ID --apply
#
# Notes
# - You must be authenticated with gcloud and have sufficient permissions to edit the project and billing.
# - Billing linking requires Billing Account Admin or Project Billing Manager on the billing account.

PROJECT=""
USER_EMAIL=""
BILLING=""
APPLY=""
DISABLE_OSLOGIN=""

die() { echo "[gcp-enable] $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required binary: $1"; }

print_usage() {
  cat <<USAGE >&2
Usage: $0 -p <PROJECT_ID> -u <USER_EMAIL> [-b <BILLING_ID>] [--apply] [--no-oslogin]

Options:
  -p <PROJECT_ID>   Target GCP project (e.g., quantrabbit)
  -u <USER_EMAIL>   User to grant access to (e.g., www.tosakiweb.net@gmail.com)
  -b <BILLING_ID>   Billing account ID (e.g., 0000-AAAAAA-BBBBBB)
  --apply           Actually run changes (dry-run by default)
  --no-oslogin      Do not set enable-oslogin=TRUE metadata
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PROJECT="$2"; shift 2 ;;
    -u) USER_EMAIL="$2"; shift 2 ;;
    -b) BILLING="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --no-oslogin) DISABLE_OSLOGIN=1; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$PROJECT" && -n "$USER_EMAIL" ]] || { print_usage; die "-p and -u are required"; }

need gcloud

run() {
  if [[ -n "$APPLY" ]]; then
    printf '+ ';
    printf '%q ' "$@"; echo
    "$@"
  else
    printf '(dry-run) ';
    printf '%q ' "$@"; echo
  fi
}

echo "[gcp-enable] Checking gcloud auth..."
if ! gcloud auth list --format='get(account)' >/dev/null 2>&1; then
  die "gcloud not authenticated. Run: gcloud auth login"
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' || true)
echo "[gcp-enable] Active account: ${ACTIVE_ACCOUNT:-<none>}"

echo "[gcp-enable] Verifying project: $PROJECT"
if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  die "Project not found or no access: $PROJECT"
fi

echo "[gcp-enable] Setting default project"
run gcloud config set project "$PROJECT"

if [[ -n "$BILLING" ]]; then
  echo "[gcp-enable] Linking billing account: $BILLING"
  run gcloud beta billing projects link "$PROJECT" --billing-account="$BILLING"
else
  echo "[gcp-enable] Billing linking skipped (no -b)."
fi

echo "[gcp-enable] Granting IAM roles to user: $USER_EMAIL"
run gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="user:$USER_EMAIL" --role="roles/owner"
run gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="user:$USER_EMAIL" --role="roles/compute.osAdminLogin"
run gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="user:$USER_EMAIL" --role="roles/iap.tunnelResourceAccessor"

echo "[gcp-enable] Enabling required APIs"
run gcloud services enable \
  compute.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  iap.googleapis.com \
  oslogin.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com

if [[ -z "$DISABLE_OSLOGIN" ]]; then
  echo "[gcp-enable] Setting project metadata: enable-oslogin=TRUE"
  run gcloud compute project-info add-metadata --project="$PROJECT" --metadata enable-oslogin=TRUE
else
  echo "[gcp-enable] Skipping OS Login project metadata"
fi

echo "[gcp-enable] Verifying (read-only checks)"
run gcloud beta billing projects describe "$PROJECT"
run gcloud projects get-iam-policy "$PROJECT" --filter="bindings.members:user:$USER_EMAIL"
run gcloud services list --enabled --project "$PROJECT" --filter='name~(compute|run|cloudbuild|artifactregistry|iap|oslogin|logging|monitoring|pubsub|storage)'

cat <<NEXT

[gcp-enable] Next steps
- 初回のみ Cloud Console で利用規約に同意が必要な場合があります: https://console.cloud.google.com/
- OS Login 鍵登録: ssh-keygen -t ed25519 -f ~/.ssh/gcp_oslogin_quantrabbit -N '' -C 'oslogin-quantrabbit'
- OS Login 鍵を登録: gcloud compute os-login ssh-keys add --key-file ~/.ssh/gcp_oslogin_quantrabbit.pub --ttl 30d
- IAP 経由で VM に疎通確認 (例):
  gcloud compute ssh fx-trader-vm --project=$PROJECT --zone=asia-northeast1-a \
    --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
    --command "sudo -n true && echo SUDO_OK || echo NO_SUDO"
NEXT

echo "[gcp-enable] Done. Use --apply to execute if this was a dry-run."
