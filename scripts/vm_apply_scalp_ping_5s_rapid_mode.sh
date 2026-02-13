#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  scripts/vm_apply_scalp_ping_5s_rapid_mode.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t --mode base|safe
#  scripts/vm_apply_scalp_ping_5s_rapid_mode.sh ... --mode auto [--window-min 15]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="${SCRIPT_DIR}/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
ACCOUNT=""
KEYFILE=""
USE_IAP=""
MODE="base"
DRY_RUN=""
WINDOW_MIN="15"
AUTO="0"

BASE_ENV="ops/env/scalp_ping_5s_rapid_mode_base_20260213.env"
SAFE_ENV="ops/env/scalp_ping_5s_rapid_mode_safe_20260213.env"

SL_RATE_SAFE=0.06
SL_RATE_RETURN=0.04
SHORT_AVG_SAFE=-3.0
SHORT_AVG_RETURN=-1.8

# Keep hysteresis: SAFE->BASE requires 2 consecutive good windows.
BASE_STREAK_TARGET=2

die() { echo "[scalp5 rapid mode] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2 ;;
    -z|--zone) ZONE="$2"; shift 2 ;;
    -m|--instance) INSTANCE="$2"; shift 2 ;;
    -A|--account) ACCOUNT="$2"; shift 2 ;;
    -k|--keyfile) KEYFILE="$2"; shift 2 ;;
    -t|--iap) USE_IAP=1; shift ;;
    --mode) MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --window-min) WINDOW_MIN="$2"; shift 2 ;;
    --auto) AUTO=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p/--project, -z/--zone, -m/--instance are required"
[[ -x "$VM_SH" ]] || die "vm.sh not found or not executable: $VM_SH"
[[ -f "$BASE_ENV" ]] || die "Base env file missing: $BASE_ENV"
[[ -f "$SAFE_ENV" ]] || die "Safe env file missing: $SAFE_ENV"

case "$MODE" in
  base|safe|auto) ;;
  *) die "--mode must be base|safe|auto" ;;
esac

run_vm() {
  local -a args=("$VM_SH" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE")
  if [[ -n "$ACCOUNT" ]]; then args+=(-A "$ACCOUNT"); fi
  if [[ -n "$KEYFILE" ]]; then args+=(-k "$KEYFILE"); fi
  if [[ -n "$USE_IAP" ]]; then args+=(-t); fi
  args+=("$@")

  if [[ -n "$DRY_RUN" ]]; then
    printf '[dry-run]'
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi
  "${args[@]}"
}

query_remote() {
  local query="$1"
  run_vm exec -- "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/trades.db \"$query\""
}

get_current_mode() {
  run_vm exec -- "grep -E '^SCALP_PING_5S_RAPID_MODE=' /etc/quantrabbit/scalp_ping_5s.env 2>/dev/null | sed -n '1p' | cut -d= -f2"
}

apply_mode() {
  local target_mode="$1"
  local env_file
  if [[ "$target_mode" == "safe" ]]; then
    env_file="$SAFE_ENV"
  else
    env_file="$BASE_ENV"
  fi

  echo "[INFO] Applying rapid mode '$target_mode' from $env_file"
  bash "$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE" ${USE_IAP:+-t} ${KEYFILE:+-k "$KEYFILE"} ${ACCOUNT:+-A "$ACCOUNT"} --env-file "$env_file"
}

normalize_mode() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

if [[ "$AUTO" == "1" ]]; then
  echo "[INFO] auto mode enabled. window=${WINDOW_MIN}m"
  QUERY="SELECT CAST(SUM(CASE WHEN units < 0 AND close_reason='STOP_LOSS_ORDER' THEN 1 ELSE 0 END) AS REAL) / MAX(CAST(SUM(CASE WHEN units < 0 THEN 1 ELSE 0 END) AS REAL), 0.0001) AS short_sl_rate, AVG(CASE WHEN units < 0 AND close_reason IS NOT NULL THEN pl_pips END) AS short_avg FROM trades WHERE strategy_tag='scalp_ping_5s_live' AND close_time >= datetime('now','-${WINDOW_MIN} minutes') AND close_time IS NOT NULL;"
  SHORT_STATS="$(query_remote "$QUERY")"
  SHORT_SL_RATE="$(echo "$SHORT_STATS" | awk -F'|' '{print $1}')"
  SHORT_AVG="$(echo "$SHORT_STATS" | awk -F'|' '{print $2}')"
  SHORT_SL_RATE="${SHORT_SL_RATE:-0}"
  SHORT_AVG="${SHORT_AVG:-0}"

  if [[ "$SHORT_SL_RATE" == "0" && "$SHORT_AVG" == "0" ]]; then
    echo "[WARN] auto stats empty; fallback to base"
    TARGET_MODE="base"
  else
    CURRENT_MODE="$(normalize_mode "$(get_current_mode || true)")"
    BASE_STREAK_FILE="/tmp/scalp_ping_5s_base_streak"

    if [[ "${CURRENT_MODE}" == "base" ]]; then
      if awk -v sl="$SHORT_SL_RATE" -v avg="$SHORT_AVG" -v safe_sl="$SL_RATE_SAFE" -v safe_avg="$SHORT_AVG_SAFE" 'BEGIN { exit !((sl > safe_sl) || (avg < safe_avg)) }'; then
        TARGET_MODE="safe"
      else
        TARGET_MODE="base"
      fi
    else
      STREAK="$(cat "$BASE_STREAK_FILE" 2>/dev/null || echo 0)"
      if awk -v sl="$SHORT_SL_RATE" -v avg="$SHORT_AVG" -v ret_sl="$SL_RATE_RETURN" -v ret_avg="$SHORT_AVG_RETURN" 'BEGIN { exit !((sl < ret_sl) && (avg > ret_avg)) }'; then
        STREAK=$(awk -v s="$STREAK" 'BEGIN { print (s+1) }')
      else
        STREAK=0
      fi
      if [[ "$STREAK" -ge "$BASE_STREAK_TARGET" ]]; then
        TARGET_MODE="base"
      else
        TARGET_MODE="safe"
      fi
      echo "$STREAK" > "$BASE_STREAK_FILE"
    fi
  fi

  echo "[INFO] short_sl_rate=${SHORT_SL_RATE} short_avg=${SHORT_AVG} -> target_mode=${TARGET_MODE}"
else
  TARGET_MODE="$MODE"
fi

if [[ "${TARGET_MODE}" == "base" || "${TARGET_MODE}" == "safe" ]]; then
  apply_mode "$TARGET_MODE"
else
  die "Target mode invalid: ${TARGET_MODE}"
fi

echo "[INFO] current rapid mode:" && get_current_mode
