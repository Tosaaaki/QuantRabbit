#!/usr/bin/env bash
# 毎日深夜に SQLite とログを GCS へ同期

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BACKUP_DIR="$PROJECT_ROOT/logs"
BUCKET="gs://${GCS_BACKUP_BUCKET:-fx-backups}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
gsutil -m rsync -r "${BACKUP_DIR}" "${BUCKET}/${TIMESTAMP}/"

echo "Backup finished: ${BUCKET}/${TIMESTAMP}/"