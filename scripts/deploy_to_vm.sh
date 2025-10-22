#!/usr/bin/env bash
# Deploy helper: Local git push -> VM git pull -> service restart
#
# Usage:
#   scripts/deploy_to_vm.sh [-b BRANCH] [-i] [-p PROJECT] [-z ZONE] [-m INSTANCE] [-d REPO_DIR] [-s SERVICE]
#
# Options:
#   -b BRANCH    Git branch to deploy (default: current branch)
#   -i           Install/upgrade requirements in the VM venv
#   -p PROJECT   GCP project (default: gcloud config get-value project)
#   -z ZONE      GCE zone (default: asia-northeast1-a)
#   -m INSTANCE  VM instance name (default: fx-trader-vm)
#   -d REPO_DIR  Repo path on VM (default: /home/tossaki/QuantRabbit)
#   -s SERVICE   systemd service name (default: quantrabbit.service)
#
# Requirements:
#   - gcloud CLI logged in and able to SSH to the VM (OS Login or SSH keys)
#   - VM repo remote 'origin' is reachable (public GitHub OK)

set -euo pipefail

BRANCH=""
INSTALL_DEPS=0
PROJECT="$(gcloud config get-value project 2>/dev/null || echo "")"
ZONE="asia-northeast1-a"
INSTANCE="fx-trader-vm"
REPO_DIR="/home/tossaki/QuantRabbit"
SERVICE="quantrabbit.service"

while getopts ":b:ip:z:m:d:s:" opt; do
  case "$opt" in
    b) BRANCH="$OPTARG" ;;
    i) INSTALL_DEPS=1 ;;
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    d) REPO_DIR="$OPTARG" ;;
    s) SERVICE="$OPTARG" ;;
    *) echo "Usage: $0 [-b BRANCH] [-i] [-p PROJECT] [-z ZONE] [-m INSTANCE] [-d REPO_DIR] [-s SERVICE]" >&2; exit 2 ;;
  esac
done

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
fi

if [[ -z "$PROJECT" ]]; then
  echo "[deploy] GCP project is not set. Use -p or 'gcloud config set project ...'" >&2
  exit 2
fi

echo "[deploy] Project=$PROJECT Zone=$ZONE Instance=$INSTANCE Branch=$BRANCH"
echo "[deploy] Pushing local branch to origin..."

# Ensure there is a remote
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "[deploy] No 'origin' remote configured in this repo." >&2
  exit 2
fi

# Push current commits
git push origin "$BRANCH"

echo "[deploy] Connecting to VM and updating repo..."

remote_cmd=(
  "set -euo pipefail"
  "cd '$REPO_DIR'"
  "echo '[vm] PWD:' \"\$(pwd)\""
  "git fetch --all --prune"
  "git checkout -B '$BRANCH' 'origin/$BRANCH'"
)

if [[ $INSTALL_DEPS -eq 1 ]]; then
  remote_cmd+=(
    "if [ -f .venv/bin/activate ]; then . .venv/bin/activate; pip install -U pip; pip install -r requirements.txt; else echo '[vm] venv not found, skipping pip install'; fi"
  )
fi

remote_cmd+=(
  "sudo systemctl restart '$SERVICE'"
  "sleep 2"
  "systemctl is-active '$SERVICE' || (echo '[vm] service failed to start' >&2; exit 1)"
  "echo '[vm] service restarted OK'"
)

gcloud compute ssh "$INSTANCE" \
  --project "$PROJECT" --zone "$ZONE" \
  --command "bash -lc \"$(printf '%s; ' "${remote_cmd[@]}")\""

echo "[deploy] Done. Tail logs with:"
echo "  gcloud compute ssh $INSTANCE --project $PROJECT --zone $ZONE --command 'journalctl -u $SERVICE -f'"

