#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/deploy_bundle_to_gcs.sh -b <bucket> [-o <object_path>]
  -b <bucket>       GCS bucket (required)
  -o <object_path>  Object path (default: deploy/qr_bundle_<timestamp>.tar.gz)
USAGE
}

BUCKET=""
OBJECT_PATH=""

while getopts ":b:o:h" opt; do
  case "$opt" in
    b) BUCKET="$OPTARG" ;;
    o) OBJECT_PATH="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "[bundle] bucket is required" >&2
  usage
  exit 2
fi

if [[ -z "$OBJECT_PATH" ]]; then
  OBJECT_PATH="deploy/qr_bundle_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "[bundle] gcloud not found" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_BUNDLE="$(mktemp -t qr_bundle.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_BUNDLE"' EXIT

tar -czf "$TMP_BUNDLE" -C "$ROOT_DIR" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".cache" \
  --exclude "__pycache__" \
  --exclude "logs" \
  --exclude "local" \
  --exclude "remote_logs" \
  --exclude "remote_logs_current" \
  --exclude "remote_logs_long" \
  --exclude "remote_logs_vm" \
  --exclude "tmp" \
  --exclude "remote_tmp" \
  .

DEST="gs://${BUCKET}/${OBJECT_PATH}"
echo "[bundle] uploading ${TMP_BUNDLE} -> ${DEST}"
gcloud storage cp "$TMP_BUNDLE" "$DEST"
echo "[bundle] done: ${DEST}"
