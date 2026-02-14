#!/usr/bin/env bash
set -euo pipefail

# Apply scalp_ping_5s entry-quality env overrides on VM and restart only
# quant-scalp-ping-5s.service.
#
# Usage:
#   scripts/vm_apply_scalp_ping_5s_entry_quality.sh \
#     -p <PROJECT> -z <ZONE> -m <INSTANCE> [-t] [-k <KEYFILE>] [-A <ACCOUNT>] \
#     [--env-file ops/env/scalp_ping_5s_entry_quality_on.env] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="$SCRIPT_DIR/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
USE_IAP=""
KEYFILE=""
ACCOUNT=""
ENV_FILE="ops/env/scalp_ping_5s_entry_quality_on.env"
TARGET_ENV_FILE="/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s.env"
SERVICE_NAME="quant-scalp-ping-5s.service"
DRY_RUN=""

die() { echo "[vm_apply_scalp_ping_5s_entry_quality] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2;;
    -z|--zone) ZONE="$2"; shift 2;;
    -m|--instance) INSTANCE="$2"; shift 2;;
    -A|--account) ACCOUNT="$2"; shift 2;;
    -k|--keyfile) KEYFILE="$2"; shift 2;;
    -t|--iap) USE_IAP=1; shift;;
    --env-file) ENV_FILE="$2"; shift 2;;
    --target-env-file) TARGET_ENV_FILE="$2"; shift 2;;
    --service) SERVICE_NAME="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p/--project, -z/--zone, -m/--instance are required"
[[ -x "$VM_SH" ]] || die "vm.sh not found or not executable: $VM_SH"
[[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE"

ENV_OVERRIDES_CONTENT="$(sed -e 's/\\/\\\\/g' -e 's/`/\\`/g' -e 's/\$/\\$/g' "$ENV_FILE")"

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

read -r -d '' REMOTE <<EOF_REMOTE || true
sudo bash -lc '
  set -euo pipefail
  TS=\$(date -u +%Y%m%dT%H%M%SZ)
  TMP=/tmp/qr_scalp_ping_5s_entry_quality.env
  cat > "\$TMP" <<'"'"'EOC'"'"'
${ENV_OVERRIDES_CONTENT}
EOC

  mkdir -p "\$(dirname "$TARGET_ENV_FILE")"
  touch "$TARGET_ENV_FILE"
  cp "$TARGET_ENV_FILE" "$TARGET_ENV_FILE.bak.\$TS" || true

  while IFS= read -r line; do
    trimmed="\${line%%#*}"
    trimmed="\$(echo "\$trimmed" | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*\$//")"
    [[ -z "\$trimmed" ]] && continue
    key="\${trimmed%%=*}"
    if grep -q "^\${key}=" "$TARGET_ENV_FILE"; then
      sed -i "s|^\${key}=.*|\${trimmed}|" "$TARGET_ENV_FILE"
    else
      echo "\$trimmed" >> "$TARGET_ENV_FILE"
    fi
  done < "\$TMP"

  systemctl daemon-reload
  systemctl restart "$SERVICE_NAME"

  echo "[OK] Applied env overrides from $ENV_FILE"
  echo "[OK] Target env file: $TARGET_ENV_FILE"
  echo "--- effective env keys ---"
  grep -E "^(ORDER_ENTRY_QUALITY_MICROSTRUCTURE_ENABLED|ORDER_ENTRY_QUALITY_REGIME_PENALTY_ENABLED|ORDER_ENTRY_QUALITY_REGIME_MISMATCH_MIN_CONF_SCALP_FAST|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY_SCALP_FAST)=" "$TARGET_ENV_FILE" || true
'
EOF_REMOTE

echo "[INFO] Applying scalp_ping_5s entry-quality overrides to $INSTANCE ($PROJECT/$ZONE) ..."
run_vm exec -- "$REMOTE"

echo "[INFO] $SERVICE_NAME status (short) ..."
run_vm exec -- "sudo systemctl is-active $SERVICE_NAME && sudo systemctl status --no-pager $SERVICE_NAME | sed -n '1,20p'"

echo "[INFO] Recent scalp_ping_5s orders (20m) ..."
run_vm exec -- "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/orders.db \"SELECT status, COUNT(*) FROM orders WHERE julianday(ts)>=julianday('now','-20 minutes') AND json_valid(request_json) AND json_extract(request_json,'$.entry_thesis.strategy_tag')='scalp_ping_5s_live' GROUP BY status ORDER BY COUNT(*) DESC;\""

echo "[DONE]"
