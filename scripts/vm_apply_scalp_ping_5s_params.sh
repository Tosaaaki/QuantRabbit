#!/usr/bin/env bash
set -euo pipefail

# Apply ad-hoc SCALP_PING_5S_* environment overrides to /etc/quantrabbit/scalp_ping_5s.env
# on the VM and restart quant-scalp-ping-5s.service.
#
# Usage:
#   scripts/vm_apply_scalp_ping_5s_params.sh \
#     -p <PROJECT> -z <ZONE> -m <INSTANCE> -t \
#     --set SCALP_PING_5S_MIN_UNITS=1000 \
#     --set SCALP_PING_5S_BASE_ENTRY_UNITS=3000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="$SCRIPT_DIR/vm.sh"
TUNING_SCRIPT="$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh"

PROJECT=""
ZONE=""
INSTANCE=""
USE_IAP=""
KEYFILE=""
ACCOUNT=""
DRY_RUN=""

declare -a KV_PAIRS=()

die() { echo "[vm_apply_scalp_ping_5s_params] $*" >&2; exit 1; }

normalize_key() {
  local key="$1"
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"
  echo "$key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2 ;;
    -z|--zone) ZONE="$2"; shift 2 ;;
    -m|--instance) INSTANCE="$2"; shift 2 ;;
    -A|--account) ACCOUNT="$2"; shift 2 ;;
    -k|--keyfile) KEYFILE="$2"; shift 2 ;;
    -t|--iap) USE_IAP=1; shift ;;
    --set)
      [[ $# -ge 2 ]] || die "--set requires KEY=VALUE"
      KV_PAIRS+=("$2")
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

[[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p/--project, -z/--zone, -m/--instance are required"
[[ -x "$VM_SH" ]] || die "vm.sh not found or not executable: $VM_SH"
[[ -x "$TUNING_SCRIPT" ]] || die "vm_apply_scalp_ping_5s_tuning.sh not found or not executable: $TUNING_SCRIPT"
[[ ${#KV_PAIRS[@]} -gt 0 ]] || die "--set is required at least once"

TMP_ENV="$(mktemp /tmp/qr_scalp_ping_5s_params.XXXXXX.env)"
trap 'rm -f "$TMP_ENV"' EXIT

for pair in "${KV_PAIRS[@]}"; do
  if [[ "$pair" != *"="* ]]; then
    die "Invalid --set format: $pair (expected KEY=VALUE)"
  fi

  key="$(normalize_key "${pair%%=*}")"
  val="${pair#*=}"

  if [[ -z "$key" ]]; then
    die "Empty key in --set: $pair"
  fi
  if [[ "$key" != SCALP_PING_5S_* ]]; then
    die "Only SCALP_PING_5S_* keys are allowed. Invalid: $key"
  fi

  echo "${key}=${val}" >> "$TMP_ENV"
done

sort -u "$TMP_ENV" -o "$TMP_ENV"

echo "[INFO] Applying ${#KV_PAIRS[@]} ad-hoc params from $TMP_ENV to $INSTANCE ($PROJECT/$ZONE)"

cmd=(bash "$TUNING_SCRIPT" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE")
if [[ -n "$USE_IAP" ]]; then
  cmd+=(-t)
fi
if [[ -n "$ACCOUNT" ]]; then
  cmd+=(-A "$ACCOUNT")
fi
if [[ -n "$KEYFILE" ]]; then
  cmd+=(-k "$KEYFILE")
fi
if [[ -n "$DRY_RUN" ]]; then
  cmd+=(--dry-run)
fi
cmd+=(--env-file "$TMP_ENV")

"${cmd[@]}"
