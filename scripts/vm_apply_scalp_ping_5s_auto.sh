#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT="${SCALP_PING_5S_AUTO_PROJECT:-quantrabbit}"
ZONE="${SCALP_PING_5S_AUTO_ZONE:-asia-northeast1-a}"
INSTANCE="${SCALP_PING_5S_AUTO_INSTANCE:-fx-trader-vm}"
WINDOW_MIN="${SCALP_PING_5S_AUTO_WINDOW_MIN:-10}"
USE_IAP="${SCALP_PING_5S_AUTO_USE_IAP:-1}"

AUTOTUNE_SCRIPT="$SCRIPT_DIR/vm_apply_scalp_ping_5s_rapid_mode.sh"
LOCAL="${SCALP_PING_5S_AUTO_LOCAL:-1}"

if [[ ! -x "$AUTOTUNE_SCRIPT" ]]; then
  echo "[scalp5 auto] autotune script not executable: $AUTOTUNE_SCRIPT" >&2
  exit 1
fi

args=(-p "$PROJECT" -z "$ZONE" -m "$INSTANCE" --auto --window-min "$WINDOW_MIN")
if [[ "$LOCAL" == "1" ]]; then
  args+=(--local)
fi
if [[ "$USE_IAP" == "1" ]]; then
  args+=(-t)
fi

echo "[scalp5 auto] $(date -u +'%Y-%m-%dT%H:%M:%SZ') project=$PROJECT zone=$ZONE instance=$INSTANCE window=${WINDOW_MIN}m iap=$USE_IAP"
(
  cd "$ROOT_DIR"
  bash "$AUTOTUNE_SCRIPT" "${args[@]}"
)
