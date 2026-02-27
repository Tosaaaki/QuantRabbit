#!/usr/bin/env bash
# Deploy helper: local git push -> VM git sync -> service restart
#
# Usage:
#   scripts/deploy_to_vm.sh [-b BRANCH] [-i] [-p PROJECT] [-z ZONE] [-m INSTANCE] [-d REPO_DIR] [-s SERVICE] \
#                           [-k SSH_KEYFILE] [-t] [-K SA_KEYFILE] [-A SA_ACCOUNT]
#
# Options:
#   -b BRANCH    Git branch to deploy (default: current branch)
#   -i           Install/upgrade requirements in the VM venv
#   -p PROJECT   GCP project (default: gcloud config get-value project)
#   -z ZONE      GCE zone (default: asia-northeast1-a)
#   -m INSTANCE  VM instance name (default: fx-trader-vm)
#   -d REPO_DIR  Repo path on VM (default: /home/tossaki/QuantRabbit)
#   -s SERVICE   systemd service name (default: quant-market-data-feed.service)
#   -k SSH_KEYFILE  SSH private key for OS Login (optional)
#   -t           Use IAP tunnel
#   -K SA_KEYFILE Service Account JSON key; auto-activate if no active account
#   -A SA_ACCOUNT Service Account email to impersonate for gcloud commands
#
# Requirements:
#   - gcloud CLI logged in and able to SSH to the VM (OS Login or SSH keys)
#   - VM repo remote 'origin' is reachable (public GitHub OK)

set -euo pipefail

# Ensure gcloud is reachable even in non-interactive shells
if ! command -v gcloud >/dev/null 2>&1; then
  SDK_BIN="$HOME/google-cloud-sdk/bin"
  if [[ -d "$SDK_BIN" ]]; then
    export PATH="$SDK_BIN:$PATH"
  fi
fi

# Preflight: ensure gcloud exists with basic config
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[deploy] gcloud が見つかりません。まず 'scripts/install_gcloud.sh' を実行してください。" >&2
  exit 2
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
INSTALL_DEPS=0
PROJECT="$(gcloud config get-value project 2>/dev/null || echo "")"
ZONE="asia-northeast1-a"
INSTANCE="fx-trader-vm"
REPO_DIR="/home/tossaki/QuantRabbit"
SERVICE="quant-market-data-feed.service"

USE_IAP=0
SSH_KEYFILE=""
SA_KEYFILE=""
SA_IMPERSONATE=""
while getopts ":b:ip:z:m:d:s:k:tK:A:" opt; do
  case "$opt" in
    b) BRANCH="$OPTARG" ;;
    i) INSTALL_DEPS=1 ;;
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    d) REPO_DIR="$OPTARG" ;;
    s) SERVICE="$OPTARG" ;;
    k) SSH_KEYFILE="$OPTARG" ;;
    t) USE_IAP=1 ;;
    K) SA_KEYFILE="$OPTARG" ;;
    A) SA_IMPERSONATE="$OPTARG" ;;
    *)
      echo "Usage: $0 [-b BRANCH] [-i] [-p PROJECT] [-z ZONE] [-m INSTANCE] [-d REPO_DIR] [-s SERVICE]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "[deploy] GCP project が未設定です。-p で指定するか 'gcloud config set project <PROJECT>' を実行してください。" >&2
  exit 2
fi

# If no active account and SA keyfile provided, activate
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' || true)
if [[ -z "$ACTIVE_ACCOUNT" && -n "$SA_KEYFILE" ]]; then
  echo "[deploy] No active account; activating service account from key: $SA_KEYFILE"
  gcloud auth activate-service-account --key-file "$SA_KEYFILE"
fi

# Run doctor for early failure (non-destructive, no SSH attempt here)
if [[ -x scripts/gcloud_doctor.sh ]]; then
  doctor_args=(-p "$PROJECT" -z "$ZONE" -m "$INSTANCE")
  if [[ $USE_IAP -eq 1 ]]; then
    doctor_args+=(-t)
  fi
  if [[ -n "$SSH_KEYFILE" ]]; then
    doctor_args+=(-k "$SSH_KEYFILE")
  fi
  if [[ -n "$SA_KEYFILE" ]]; then
    doctor_args+=(-K "$SA_KEYFILE")
  fi
  if [[ -n "$SA_IMPERSONATE" ]]; then
    doctor_args+=(-A "$SA_IMPERSONATE")
  fi
  scripts/gcloud_doctor.sh "${doctor_args[@]}" || {
    echo "[deploy] gcloud preflight failed. See messages above." >&2
    exit 2
  }
fi

echo "[deploy] Project=$PROJECT Zone=$ZONE Instance=$INSTANCE Branch=$BRANCH Service=$SERVICE"
echo "[deploy] Pushing local branch to origin..."

# Ensure there is a remote
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "[deploy] No 'origin' remote configured in this repo." >&2
  exit 2
fi

# Push current commits
git push origin "$BRANCH"

echo "[deploy] Connecting to VM and updating repo..."

# derive repository owner from REPO_DIR (e.g. /home/<owner>/QuantRabbit)
REPO_OWNER="$(basename "$(dirname "$REPO_DIR")")"

ssh_args=("--project" "$PROJECT" "--zone" "$ZONE")
if [[ -n "$SA_IMPERSONATE" ]]; then
  ssh_args+=("--impersonate-service-account=$SA_IMPERSONATE")
fi
if [[ -n "$SSH_KEYFILE" ]]; then
  ssh_args+=("--ssh-key-file" "$SSH_KEYFILE")
fi
if [[ $USE_IAP -eq 1 ]]; then
  ssh_args+=("--tunnel-through-iap")
fi

# 1) Update repo
gcloud compute ssh "$INSTANCE" "${ssh_args[@]}" \
  --command "sudo -u '$REPO_OWNER' -H bash -lc 'cd \"$REPO_DIR\" && echo \"[vm] PWD: \$(pwd)\" && git fetch --all --prune && git checkout -B \"$BRANCH\" \"origin/$BRANCH\"'"

# 2) Install deps (optional)
if [[ $INSTALL_DEPS -eq 1 ]]; then
  gcloud compute ssh "$INSTANCE" "${ssh_args[@]}" \
    --command "sudo -u '$REPO_OWNER' -H bash -lc 'cd \"$REPO_DIR\" && if [ -f .venv/bin/activate ]; then . .venv/bin/activate; pip install -U pip && pip install -r requirements.txt; else echo \"[vm] venv not found, skipping pip install\"; fi'"
fi

# 3) Restart service
gcloud compute ssh "$INSTANCE" "${ssh_args[@]}" \
  --command "sudo systemctl daemon-reload && sudo systemctl restart '$SERVICE' && sleep 2 && systemctl is-active '$SERVICE' && echo '[vm] service restarted OK'"

echo "[deploy] Done. Tail logs with:"
echo "  gcloud compute ssh $INSTANCE --project $PROJECT --zone $ZONE --command 'journalctl -u $SERVICE -f'"
