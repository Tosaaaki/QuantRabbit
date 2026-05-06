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

args=("$@")
has_arg() {
  local needle="$1"
  local item
  for item in "${args[@]}"; do
    if [[ "$item" == "$needle" || "$item" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

if has_arg "--send" && ! has_arg "--use-gpt-trader"; then
  gpt_args=("--use-gpt-trader")
  if ! has_arg "--reuse-market-artifacts"; then
    gpt_args=("--reuse-market-artifacts" "${gpt_args[@]}")
  fi
  if ! has_arg "--gpt-decision-response"; then
    gpt_args=("${gpt_args[@]}" "--gpt-decision-response" "data/codex_trader_decision_response.json")
  fi
  args=("${gpt_args[@]}" "${args[@]}")
  echo "[run-autotrade-live] live send requires trader decision handoff; using args=${args[*]}" >&2
fi

python3 -m quant_rabbit.cli autotrade-cycle "${args[@]}"
