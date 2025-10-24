#!/usr/bin/env bash
set -euo pipefail

# Run excursion report hourly and save artifacts under logs/reports/excursion
# - Uses repo-local Python if available (.venv), else system python3

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY=python3
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

OUT_DIR="logs/reports/excursion/hourly"
mkdir -p "$OUT_DIR"

# JST timestamp for filename
TS_JST="$(TZ=Asia/Tokyo date '+%Y%m%d_%H')"
OUT_FILE="$OUT_DIR/${TS_JST}.txt"

{
  echo "=== Excursion Report @ $(date -u '+%Y-%m-%d %H:%M:%SZ') (JST $TS_JST) ==="
  echo "-- 15min window --"
  "$PY" -m analytics.excursion_report --days 1 --post-min 15 --thresholds 0.6 1.0 1.6 2.0 || true
  echo
  echo "-- 30min window --"
  "$PY" -m analytics.excursion_report --days 1 --post-min 30 --thresholds 1.6 2.0 || true
} > "$OUT_FILE"

cp -f "$OUT_FILE" "logs/reports/excursion/latest.txt"
echo "[excursion_report] wrote $OUT_FILE"

