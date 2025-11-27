#!/usr/bin/env bash
# Tail QuantRabbit VM logs via gcloud SSH (IAP + OS Login friendly)
set -euo pipefail
PROJECT="${PROJECT:-quantrabbit}"
ZONE="${ZONE:-asia-northeast1-a}"
INSTANCE="${INSTANCE:-fx-trader-vm}"
KEYFILE="${VM_SSH_KEY:-}" # optional override
TAIL_CMD="sudo journalctl -u quantrabbit.service -f"
usage() {
  cat <<USAGE
Usage: $0 [-p project] [-z zone] [-m instance] [-k keyfile] [-c command]
  -p   GCP project (default: ${PROJECT})
  -z   Compute zone (default: ${ZONE})
  -m   VM instance name (default: ${INSTANCE})
  -k   SSH key file for OS Login/IAP (optional)
  -c   Command to execute remotely (default: ${TAIL_CMD})
Example:
  VM_SSH_KEY=~/.ssh/gcp_oslogin_qr $0 -c "sudo tail -n 200 /var/log/syslog"
USAGE
}
while getopts "hp:z:m:k:c:" opt; do
  case "$opt" in
    h) usage; exit 0 ;;
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    c) TAIL_CMD="$OPTARG" ;;
    *) usage; exit 1 ;;
  esac
done
cmd=(gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap)
if [[ -n "$KEYFILE" ]]; then
  cmd+=(--ssh-key-file "$KEYFILE")
fi
cmd+=(--command "$TAIL_CMD")
exec "${cmd[@]}"
