#!/usr/bin/env bash
set -euo pipefail

# Apply 5s ping scalp env tuning on VM and restart quant-scalp-ping-5s.service.
#
# Usage:
#   scripts/vm_apply_scalp_ping_5s_tuning.sh \
#     -p <PROJECT> -z <ZONE> -m <INSTANCE> [-t] [-k <KEYFILE>] [-A <ACCOUNT>] \
#     [--env-file ops/env/scalp_ping_5s_tuning_20260212.env] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="$SCRIPT_DIR/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
USE_IAP=""
KEYFILE=""
ACCOUNT=""
ENV_FILE="ops/env/scalp_ping_5s_tuning_20260212.env"
DRY_RUN=""
LOCAL_MODE="0"

die() { echo "[vm_apply_scalp_ping_5s_tuning] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2 ;;
    -z|--zone) ZONE="$2"; shift 2 ;;
    -m|--instance) INSTANCE="$2"; shift 2 ;;
    -A|--account) ACCOUNT="$2"; shift 2 ;;
    -k|--keyfile) KEYFILE="$2"; shift 2 ;;
    -t|--iap) USE_IAP=1; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --local) LOCAL_MODE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) die "Unknown arg: $1" ;;
  esac
done

if [[ "$LOCAL_MODE" == "1" ]]; then
  PROJECT="local"
  ZONE="local"
  INSTANCE="local"
else
  [[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p/--project, -z/--zone, -m/--instance are required"
  [[ -x "$VM_SH" ]] || die "vm.sh not found or not executable: $VM_SH"
fi
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
set -euo pipefail
TS=\$(date -u +%Y%m%dT%H%M%SZ)
TARGET=/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s.env
TMP=\$(mktemp /tmp/qr_scalp_ping_5s_tuning.XXXXXX).env
cat > "\$TMP" <<'EOC'
${ENV_OVERRIDES_CONTENT}
EOC

sudo mkdir -p /home/tossaki/QuantRabbit/ops/env
sudo touch "\$TARGET"
sudo cp "\$TARGET" "\${TARGET}.bak.\$TS" || true

while IFS= read -r line; do
  trimmed="\${line%%#*}"
  trimmed="\$(echo "\$trimmed" | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$//")"
  [[ -z "\$trimmed" ]] && continue
  key="\${trimmed%%=*}"
  if sudo grep -q "^\${key}=" "\$TARGET"; then
    sudo sed -i "s|^\${key}=.*|\${trimmed}|g" "\$TARGET"
  else
    echo "\$trimmed" | sudo tee -a "\$TARGET" >/dev/null
  fi
done < "\$TMP"

sudo awk -F= "!/^[[:space:]]*#/ && !/^[[:space:]]*$/ && index(\\\$0, \"=\") > 0 { key=\\\$1; gsub(/^[[:space:]]+|[[:space:]]+$/, \"\", key); keys[++n]=key; vals[key]=\\\$0 } END { for (i=1; i<=n; i++) { k=keys[i]; if (!(k in printed)) { print vals[k]; printed[k]=1 } } }" "\$TARGET" | sudo tee "\${TARGET}.normalized" >/dev/null
sudo mv "\${TARGET}.normalized" "\$TARGET"

sudo systemctl daemon-reload
sudo systemctl restart quant-scalp-ping-5s.service
sudo systemctl is-active quant-scalp-ping-5s.service

PID=\$(sudo systemctl show -p MainPID --value quant-scalp-ping-5s.service)
echo "--- quant-scalp-ping-5s MainPID=\$PID ---"
sudo tr "\\0" "\\n" < "/proc/\$PID/environ" | grep "^SCALP_PING_5S_" | sort || true
rm -f "\$TMP"
EOF

run_local() {
  local cmd="$1"
  if [[ -n "$DRY_RUN" ]]; then
    echo "[dry-run] $cmd"
    return 0
  fi
  bash -lc "$cmd"
}

apply_local() {
  local tmp
  tmp="$(mktemp /tmp/qr_scalp_ping_5s_tuning.XXXXXX).env"
  cp "$ENV_FILE" "$tmp"

  local cmd
  read -r -d '' cmd <<EOF || true
set -euo pipefail
TS=\$(date -u +%Y%m%dT%H%M%SZ)
TARGET=/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s.env
TMP=\$(mktemp /tmp/qr_scalp_ping_5s_tuning.XXXXXX).env
cp "$tmp" "\$TMP"

sudo mkdir -p /home/tossaki/QuantRabbit/ops/env
sudo touch "\$TARGET"
sudo cp "\$TARGET" "\${TARGET}.bak.\$TS" || true

while IFS= read -r line; do
  trimmed="\${line%%#*}"
  trimmed="\$(echo "\$trimmed" | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$//")"
  [[ -z "\$trimmed" ]] && continue
  key="\${trimmed%%=*}"
  if sudo grep -q "^\${key}=" "\$TARGET"; then
    sudo sed -i "s|^\${key}=.*|\${trimmed}|g" "\$TARGET"
  else
    echo "\$trimmed" | sudo tee -a "\$TARGET" >/dev/null
  fi
done < "\$TMP"

sudo awk -F= "!/^[[:space:]]*#/ && !/^[[:space:]]*$/ && index(\\\$0, \"=\") > 0 { key=\\\$1; gsub(/^[[:space:]]+|[[:space:]]+$/, \"\", key); keys[++n]=key; vals[key]=\\\$0 } END { for (i=1; i<=n; i++) { k=keys[i]; if (!(k in printed)) { print vals[k]; printed[k]=1 } } }" "\$TARGET" | sudo tee "\${TARGET}.normalized" >/dev/null
sudo mv "\${TARGET}.normalized" "\$TARGET"

sudo systemctl daemon-reload
sudo systemctl restart quant-scalp-ping-5s.service
sudo systemctl is-active quant-scalp-ping-5s.service

PID=\$(sudo systemctl show -p MainPID --value quant-scalp-ping-5s.service)
echo "--- quant-scalp-ping-5s MainPID=\$PID ---"
sudo tr "\\0" "\\n" < "/proc/\$PID/environ" | grep "^SCALP_PING_5S_" | sort || true
rm -f "\$TMP"
EOF

  run_local "$cmd"
  rm -f "$tmp"
}

echo "[INFO] Applying scalp_ping_5s tuning env to $INSTANCE ($PROJECT/$ZONE) ..."
if [[ "$LOCAL_MODE" == "1" ]]; then
  apply_local
else
  run_vm exec -- "$REMOTE"
fi

echo "[INFO] Recent scalp_ping_5s_live closes (last 30m) ..."
if [[ "$LOCAL_MODE" == "1" ]]; then
  run_local "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/trades.db \"SELECT close_reason, COUNT(*) FROM trades WHERE strategy_tag='scalp_ping_5s_live' AND close_time >= datetime('now','-30 minutes') GROUP BY close_reason ORDER BY 2 DESC;\""
else
  run_vm exec -- "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/trades.db \"SELECT close_reason, COUNT(*) FROM trades WHERE strategy_tag='scalp_ping_5s_live' AND close_time >= datetime('now','-30 minutes') GROUP BY close_reason ORDER BY 2 DESC;\""
fi

echo "[INFO] Recent scalp_fast perf blocks (last 30m) ..."
if [[ "$LOCAL_MODE" == "1" ]]; then
  run_local "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/orders.db \"SELECT status, COUNT(*) FROM orders WHERE pocket='scalp_fast' AND ts >= datetime('now','-30 minutes') AND status LIKE 'perf_block%' GROUP BY status ORDER BY 2 DESC;\""
else
  run_vm exec -- "sqlite3 -readonly /home/tossaki/QuantRabbit/logs/orders.db \"SELECT status, COUNT(*) FROM orders WHERE pocket='scalp_fast' AND ts >= datetime('now','-30 minutes') AND status LIKE 'perf_block%' GROUP BY status ORDER BY 2 DESC;\""
fi

echo "[DONE]"
