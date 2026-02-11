#!/usr/bin/env bash
set -euo pipefail

# Apply entry-precision env hardening on VM and restart active quant services.
#
# Usage:
#   scripts/vm_apply_entry_precision_hardening.sh \
#     -p <PROJECT> -z <ZONE> -m <INSTANCE> [-t] [-k <KEYFILE>] [-A <ACCOUNT>] \
#     [--env-file ops/env/entry_precision_hardening.env] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="$SCRIPT_DIR/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
USE_IAP=""
KEYFILE=""
ACCOUNT=""
ENV_FILE="ops/env/entry_precision_hardening.env"
DRY_RUN=""

die() { echo "[vm_apply_entry_precision_hardening] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2;;
    -z|--zone) ZONE="$2"; shift 2;;
    -m|--instance) INSTANCE="$2"; shift 2;;
    -A|--account) ACCOUNT="$2"; shift 2;;
    -k|--keyfile) KEYFILE="$2"; shift 2;;
    -t|--iap) USE_IAP=1; shift;;
    --env-file) ENV_FILE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help)
      sed -n '1,60p' "$0"
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

read -r -d '' REMOTE <<EOF || true
sudo bash -lc '
  set -euo pipefail
  TS=\$(date -u +%Y%m%dT%H%M%SZ)
  TMP=/tmp/qr_entry_precision_hardening.env
  cat > "\$TMP" <<'"'"'EOC'"'"'
${ENV_OVERRIDES_CONTENT}
EOC

  touch /etc/quantrabbit.env
  cp /etc/quantrabbit.env "/etc/quantrabbit.env.bak.\$TS" || true

  while IFS= read -r line; do
    trimmed="\${line%%#*}"
    trimmed="\$(echo "\$trimmed" | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$//")"
    [[ -z "\$trimmed" ]] && continue
    key="\${trimmed%%=*}"
    if grep -q "^\${key}=" /etc/quantrabbit.env; then
      sed -i "s|^\${key}=.*|\${trimmed}|" /etc/quantrabbit.env
    else
      echo "\$trimmed" >> /etc/quantrabbit.env
    fi
  done < "\$TMP"

  systemctl daemon-reload

  mapfile -t active_quant < <(systemctl list-units --type=service --state=active --no-legend "quant-*.service" | awk "{print \\\$1}")
  if [[ \${#active_quant[@]} -gt 0 ]]; then
    systemctl restart "\${active_quant[@]}"
  fi
  systemctl restart quantrabbit.service || true

  echo "[OK] Applied env overrides from $ENV_FILE"
  echo "[OK] Active quant services restarted: \${#active_quant[@]}"
  echo "--- effective env keys ---"
  grep -E "^(ENTRY_FACTOR_MAX_AGE_SEC|ENTRY_FACTOR_STALE_ALLOW_POCKETS|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_ENABLED|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_WINDOW_SEC|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MAX_AGE_MS|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_SPAN_RATIO|ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY_MICRO)=" /etc/quantrabbit.env || true
'
EOF

echo "[INFO] Applying entry-precision env overrides to $INSTANCE ($PROJECT/$ZONE) ..."
run_vm exec -- "$REMOTE"

echo "[INFO] quantrabbit.service status (short) ..."
run_vm exec -- "sudo systemctl is-active quantrabbit.service && sudo systemctl status --no-pager quantrabbit.service | sed -n '1,20p'"

echo "[INFO] Recent order blocks (entry quality/factor stale) ..."
run_vm exec -- "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/orders.db \"WITH r AS (SELECT * FROM orders WHERE ts>=datetime('now','-30 minutes')) SELECT status, COUNT(*) c FROM r WHERE status LIKE 'entry_quality_%' OR status IN('factor_stale','forecast_scale_below_min','spread_block') GROUP BY status ORDER BY c DESC, status;\""

echo "[DONE]"
