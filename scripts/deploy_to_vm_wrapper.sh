#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for legacy deploy_to_vm.sh
# Forwards to scripts/vm.sh deploy with equivalent flags.

HERE="$(cd "$(dirname "$0")" && pwd)"
VM_SH="$HERE/vm.sh"
[ -x "$VM_SH" ] || { echo "vm.sh not found at $VM_SH" >&2; exit 1; }

PROJECT=""; ZONE=""; INSTANCE=""; ACCOUNT=""; KEYFILE=""; USE_IAP=""; REMOTE_DIR="";
BRANCH=""; INSTALL=""; SERVICE="";

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]
  -p <PROJECT>       GCP project
  -z <ZONE>          GCE zone
  -m <INSTANCE>      VM instance
  -A <ACCOUNT>       gcloud account (optional)
  -k <KEYFILE>       SSH key file (optional)
  -t                 Use IAP tunnel
  -d <REMOTE_DIR>    Remote repo dir (default: ~/QuantRabbit)
  -b <BRANCH>        Branch to deploy (default: current local branch)
  -i                 Install requirements in remote .venv
  -s <SERVICE>       systemd service to restart (e.g., quantrabbit.service)
USAGE
}

while getopts ":p:z:m:A:k:td:b:is:" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    A) ACCOUNT="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    t) USE_IAP=1 ;;
    d) REMOTE_DIR="$OPTARG" ;;
    b) BRANCH="$OPTARG" ;;
    i) INSTALL=1 ;;
    s) SERVICE="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

FLAGS=( -p "${PROJECT}" -z "${ZONE}" -m "${INSTANCE}" )
[[ -n "$ACCOUNT" ]] && FLAGS+=( -A "$ACCOUNT" )
[[ -n "$KEYFILE" ]] && FLAGS+=( -k "$KEYFILE" )
[[ -n "$USE_IAP" ]] && FLAGS+=( -t )
[[ -n "$REMOTE_DIR" ]] && FLAGS+=( -d "$REMOTE_DIR" )

DEPLOY_ARGS=( deploy )
[[ -n "$BRANCH" ]] && DEPLOY_ARGS+=( -b "$BRANCH" )
[[ -n "$INSTALL" ]] && DEPLOY_ARGS+=( -i )
[[ -n "$SERVICE" ]] && DEPLOY_ARGS+=( --restart "$SERVICE" )

exec "$VM_SH" "${FLAGS[@]}" "${DEPLOY_ARGS[@]}"

