#!/usr/bin/env bash
set -euo pipefail

# Install and enable "practice mirror" systemd units on the VM.
#
# Goal:
# - Run the same *worker* services as prod, but against an OANDA practice account.
# - Keep prod config as baseline, override only via /etc/quantrabbit_practice.env.
# - Isolate state/log DBs by forcing WorkingDirectory to a separate repo checkout.
#
# What it does:
# - Detects running `quant-*.service` units whose ExecStart contains `-m workers.`
# - Copies each unit file + its drop-ins to a new `*-practice.service` unit
# - Adds a final drop-in `zz-practice.conf` that:
#   - sets WorkingDirectory to PRACTICE_REPO
#   - loads PRACTICE_ENV (must contain OANDA_PRACTICE=true + practice token/account)
#   - adds ExecStartPre guards + creates logs/
#
# Usage (on VM, as root):
#   sudo bash scripts/install_practice_mirror_units.sh
#   sudo bash scripts/install_practice_mirror_units.sh --units "quant-micro-multi.service quant-trendma.service"
#

LIVE_REPO="/home/tossaki/QuantRabbit"
PRACTICE_REPO="/home/tossaki/QuantRabbit_practice"
PRACTICE_ENV="/etc/quantrabbit_practice.env"
SYSTEMD_DEST="/etc/systemd/system"

ENABLE_NOW=1
DRY_RUN=0
FROM_RUNNING=1
EXPLICIT_UNITS=()

usage() {
  cat <<'USAGE'
Usage: sudo bash scripts/install_practice_mirror_units.sh [options]

Options:
  --live-repo PATH        Live repo dir (default: /home/tossaki/QuantRabbit)
  --practice-repo PATH    Practice repo dir (default: /home/tossaki/QuantRabbit_practice)
  --practice-env PATH     Practice env file (default: /etc/quantrabbit_practice.env)
  --units "A B C"         Explicit unit names to mirror (disables --from-running)
  --from-running          Mirror currently running quant-*.service worker units (default)
  --no-enable             Only install unit files (do not enable/start)
  --dry-run               Print actions without writing or restarting services
  -h, --help              Show this help
USAGE
}

die() { echo "[practice-mirror] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live-repo) LIVE_REPO="$2"; shift 2 ;;
    --practice-repo) PRACTICE_REPO="$2"; shift 2 ;;
    --practice-env) PRACTICE_ENV="$2"; shift 2 ;;
    --units)
      IFS=' ' read -r -a EXPLICIT_UNITS <<< "${2:-}"
      FROM_RUNNING=0
      shift 2
      ;;
    --from-running) FROM_RUNNING=1; shift 1 ;;
    --no-enable) ENABLE_NOW=0; shift 1 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ $EUID -eq 0 ]] || die "Please run as root (sudo)."
command -v systemctl >/dev/null 2>&1 || die "systemctl not found."

[[ -d "$LIVE_REPO" ]] || die "Live repo not found: $LIVE_REPO"
[[ -d "$PRACTICE_REPO" ]] || die "Practice repo not found: $PRACTICE_REPO"
[[ -f "$PRACTICE_ENV" ]] || die "Practice env file not found: $PRACTICE_ENV"

if ! grep -Eq '^[[:space:]]*OANDA_PRACTICE[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$PRACTICE_ENV"; then
  die "Practice env must contain OANDA_PRACTICE=true: $PRACTICE_ENV"
fi

is_worker_unit() {
  local unit="$1"
  local execstart
  execstart="$(systemctl show -p ExecStart --value "$unit" 2>/dev/null || true)"
  [[ "$execstart" == *" -m workers."* ]]
}

list_running_units() {
  systemctl list-units --type=service --state=running --no-legend \
    | awk '{print $1}' \
    | grep -E '^quant-.*\.service$' \
    | grep -Ev -- '-practice\.service$' \
    || true
}

units=()
if [[ ${#EXPLICIT_UNITS[@]} -gt 0 ]]; then
  units=("${EXPLICIT_UNITS[@]}")
elif [[ $FROM_RUNNING -eq 1 ]]; then
  mapfile -t units < <(list_running_units)
else
  die "No units specified."
fi

selected=()
for unit in "${units[@]}"; do
  if is_worker_unit "$unit"; then
    selected+=("$unit")
  else
    echo "[practice-mirror] Skip (non-worker): $unit"
  fi
done

[[ ${#selected[@]} -gt 0 ]] || die "No worker units selected."

install_file() {
  local src="$1" dest="$2" mode="$3"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] install -m $mode $src $dest"
    return
  fi
  install -m "$mode" "$src" "$dest"
}

write_dropin() {
  local dest="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] write $dest"
    return
  fi
  cat >"$dest" <<EOF
[Service]
WorkingDirectory=${PRACTICE_REPO}
EnvironmentFile=${PRACTICE_ENV}
Environment=QR_PRACTICE_MIRROR=1
ExecStartPre=/bin/bash -lc '[[ -d "${PRACTICE_REPO}" ]]'
ExecStartPre=/bin/bash -lc 'mkdir -p logs tmp'
ExecStartPre=/bin/bash -lc '[[ "\${OANDA_PRACTICE:-}" == "true" ]]'
ExecStartPre=/bin/bash -lc '[[ "\${OANDA_ACCOUNT:-}" == 101-* ]]'
ExecStartPre=/bin/bash -lc '[[ -n "\${OANDA_TOKEN:-}" && -n "\${OANDA_ACCOUNT:-}" ]]'
EOF
}

created_units=()
for unit in "${selected[@]}"; do
  frag="$(systemctl show -p FragmentPath --value "$unit" 2>/dev/null || true)"
  [[ -n "$frag" && -f "$frag" ]] || { echo "[practice-mirror] Skip (no fragment): $unit"; continue; }

  practice_unit="${unit%.service}-practice.service"
  dest_unit="${SYSTEMD_DEST}/${practice_unit}"

  echo "[practice-mirror] Mirror $unit -> $practice_unit"
  install_file "$frag" "$dest_unit" 0644

  dropins="$(systemctl show -p DropInPaths --value "$unit" 2>/dev/null || true)"
  dest_dropin_dir="${SYSTEMD_DEST}/${practice_unit}.d"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] mkdir -p $dest_dropin_dir"
  else
    mkdir -p "$dest_dropin_dir"
  fi

  if [[ -n "$dropins" ]]; then
    # DropInPaths is space-separated.
    for path in $dropins; do
      [[ -f "$path" ]] || continue
      install_file "$path" "${dest_dropin_dir}/$(basename "$path")" 0644
    done
  fi

  write_dropin "${dest_dropin_dir}/zz-practice.conf"
  created_units+=("$practice_unit")
done

[[ ${#created_units[@]} -gt 0 ]] || die "No practice units created."

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] systemctl daemon-reload"
else
  systemctl daemon-reload
fi

if [[ $ENABLE_NOW -eq 1 ]]; then
  for unit in "${created_units[@]}"; do
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] systemctl enable --now $unit"
    else
      systemctl enable --now "$unit"
    fi
  done
fi

echo "[practice-mirror] Done. Practice units:"
for unit in "${created_units[@]}"; do
  echo "  - $unit"
done
