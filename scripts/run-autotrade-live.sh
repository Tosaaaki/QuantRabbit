#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="/Users/tossaki/App/QuantRabbit"
cd "$ROOT_DIR"

export PYTHONPATH="src"
export QR_LIVE_ENABLED="${QR_LIVE_ENABLED:-1}"
export QR_OANDA_ENV_FILE="${QR_OANDA_ENV_FILE:-.env.local}"

if [[ ! -f "$QR_OANDA_ENV_FILE" ]]; then
  echo "[run-autotrade-live] missing OANDA env file: $QR_OANDA_ENV_FILE" >&2
  exit 2
fi

if [[ "$QR_LIVE_ENABLED" != "1" ]]; then
  echo "[run-autotrade-live] QR_LIVE_ENABLED=$QR_LIVE_ENABLED; forcing dry-run mode." >&2
fi

missing_keys=0
for required_key in QR_OANDA_ACCOUNT_ID QR_OANDA_TOKEN QR_OANDA_BASE_URL; do
  if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${required_key}[[:space:]]*=" "$QR_OANDA_ENV_FILE"; then
    echo "[run-autotrade-live] missing ${required_key} in ${QR_OANDA_ENV_FILE}" >&2
    missing_keys=1
  fi
done

if [[ "$missing_keys" -ne 0 ]]; then
  echo "[run-autotrade-live] OANDA env file is invalid for live operations." >&2
  exit 2
fi

readonly LIVE_ARG="$*"
echo "[run-autotrade-live] running from ${ROOT_DIR} with env_file=${QR_OANDA_ENV_FILE}, QR_LIVE_ENABLED=${QR_LIVE_ENABLED}, args=${LIVE_ARG:-<none>}" >&2

python3 -m quant_rabbit.cli autotrade-cycle "$@"
