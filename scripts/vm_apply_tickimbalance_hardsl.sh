#!/usr/bin/env bash
set -euo pipefail

# Apply TickImbalance (scalp_precision) entry hard SL + SL-gated loss-cut EXIT via systemd drop-ins.
#
# This script runs remote commands via scripts/vm.sh (gcloud compute ssh / OS Login / IAP).
#
# Usage:
#   scripts/vm_apply_tickimbalance_hardsl.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> [-t] [-k <KEYFILE>] [--monitor-sec N] [--dry-run]
#
# Example:
#   scripts/vm_apply_tickimbalance_hardsl.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t -k ~/.ssh/gcp_oslogin_quantrabbit --monitor-sec 300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="$SCRIPT_DIR/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
USE_IAP=""
KEYFILE=""
ACCOUNT=""
MONITOR_SEC="180"
DRY_RUN=""

die() { echo "[vm_apply_tickimbalance_hardsl] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2;;
    -z|--zone) ZONE="$2"; shift 2;;
    -m|--instance) INSTANCE="$2"; shift 2;;
    -A|--account) ACCOUNT="$2"; shift 2;;
    -k|--keyfile) KEYFILE="$2"; shift 2;;
    -t|--iap) USE_IAP=1; shift;;
    --monitor-sec) MONITOR_SEC="$2"; shift 2;;
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

ENTRY_SVC="quant-scalp-precision-tick-imbalance.service"
EXIT_SVC="quant-scalp-precision-tick-imbalance-exit.service"

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

read -r -d '' REMOTE <<'EOF' || true
sudo bash -lc '
  set -euo pipefail

  ENTRY_SVC="quant-scalp-precision-tick-imbalance.service"
  EXIT_SVC="quant-scalp-precision-tick-imbalance-exit.service"

  mkdir -p "/etc/systemd/system/${ENTRY_SVC}.d"
  cat > "/etc/systemd/system/${ENTRY_SVC}.d/override.conf" <<EOC
[Service]
Environment="ORDER_DISABLE_STOP_LOSS_SCALP=false"
Environment="ORDER_ALLOW_STOP_LOSS_WITH_EXIT_NO_NEGATIVE_CLOSE_SCALP=true"
Environment="ORDER_ENTRY_HARD_STOP_PIPS_STRATEGY_TICKIMBALANCE=25"
EOC

  mkdir -p "/etc/systemd/system/${EXIT_SVC}.d"
  cat > "/etc/systemd/system/${EXIT_SVC}.d/override.conf" <<EOC
[Service]
Environment="RANGEFADER_EXIT_LOSS_CUT_ENABLED=true"
Environment="RANGEFADER_EXIT_LOSS_CUT_REQUIRE_SL=true"
Environment="RANGEFADER_EXIT_LOSS_CUT_SOFT_PIPS=12"
Environment="RANGEFADER_EXIT_LOSS_CUT_HARD_PIPS=25"
Environment="RANGEFADER_EXIT_LOSS_CUT_MAX_HOLD_SEC=1200"
EOC

  systemctl daemon-reload
  systemctl restart "$EXIT_SVC"
  systemctl restart "$ENTRY_SVC"

  systemctl --no-pager -l status "$ENTRY_SVC" "$EXIT_SVC" || true
  echo "[OK] drop-ins applied + services restarted"
'
EOF

echo "[INFO] Applying systemd drop-ins + restarting services on $INSTANCE ($PROJECT/$ZONE) ..."
run_vm exec -- "$REMOTE"

echo "[INFO] Effective Environment (systemctl show) ..."
run_vm exec -- "sudo systemctl show -p Environment '$ENTRY_SVC' | sed 's/^Environment=//'"
run_vm exec -- "sudo systemctl show -p Environment '$EXIT_SVC' | sed 's/^Environment=//'"

echo "[INFO] Recent logs (last 120 lines each) ..."
run_vm exec -- "sudo journalctl -u '$ENTRY_SVC' -n 120 --output=short-iso --no-pager || true"
run_vm exec -- "sudo journalctl -u '$EXIT_SVC' -n 120 --output=short-iso --no-pager || true"

if [[ "${MONITOR_SEC:-0}" != "0" ]]; then
  echo "[INFO] Monitoring journald for ${MONITOR_SEC}s ..."
  run_vm exec -- "sudo timeout ${MONITOR_SEC} journalctl -u '$ENTRY_SVC' -u '$EXIT_SVC' -n 30 -f --output=short-iso || true"
fi

echo "[DONE]"
