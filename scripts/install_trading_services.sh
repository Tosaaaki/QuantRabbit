#!/usr/bin/env bash
set -euo pipefail

# Install and enable QuantRabbit systemd units so they survive VM reboots.
# Usage:
#   sudo bash scripts/install_trading_services.sh [--repo /home/tossaki/QuantRabbit] [--all] [--units "quant-impulse-break-s5.service quant-impulse-break-s5-exit.service"]
# Defaults: only installs/enables the main gate `quantrabbit.service`.

REPO_DIR="/home/tossaki/QuantRabbit"
INSTALL_ALL=0
EXPLICIT_UNITS=()
V2_DISALLOWED_UNITS=(
  "quant-impulse-retest-s5.service"
  "quant-impulse-retest-s5-exit.service"
  "quant-micro-adaptive-revert.service"
  "quant-micro-adaptive-revert-exit.service"
  "quant-trend-reclaim-long.service"
  "quant-trend-reclaim-long-exit.service"
)
NO_BLOCK_START_UNITS=(
  "quant-strategy-optimizer.service"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_DIR="$2"
      shift 2
      ;;
    --all)
      INSTALL_ALL=1
      shift 1
      ;;
    --units)
      IFS=' ' read -r -a EXPLICIT_UNITS <<< "$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: sudo bash scripts/install_trading_services.sh [--repo /path/to/QuantRabbit] [--all] [--units \"quant-impulse-break-s5.service ...\"]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

SYSTEMD_DEST="/etc/systemd/system"
MAIN_UNIT_SRC="ops/systemd/quantrabbit.service"
MAIN_UNIT_DEST="$SYSTEMD_DEST/quantrabbit.service"

install_unit() {
  local src="$1"
  local dest="$SYSTEMD_DEST/$(basename "$src")"
  local base="$(basename "$src")"
  local skip="0"
  if [[ ! -f "$src" ]]; then
    echo "Skip (not found): $src"
    return
  fi
  if [[ $INSTALL_ALL -eq 1 ]]; then
    for svc in "${V2_DISALLOWED_UNITS[@]}"; do
      if [[ "$base" == "$svc" ]]; then
        skip="1"
        break
      fi
    done
    if [[ "$skip" == "1" ]]; then
      echo "Skip disabled legacy unit (by V2 policy): $base"
      return
    fi
  fi
  install -m 0644 "$src" "$dest"
  echo "Installed: $dest"
  ENABLE_QUEUE+=("$(basename "$dest")")
}

enable_unit() {
  local unit="$1"
  if systemctl enable "$unit"; then
    local no_block="0"
    for skip in "${NO_BLOCK_START_UNITS[@]}"; do
      if [[ "$unit" == "$skip" ]]; then
        no_block="1"
        break
      fi
    done
    if [[ "$no_block" == "1" ]]; then
      if systemctl start --no-block "$unit"; then
        echo "Enabled and start requested (non-blocking): $unit"
      else
        echo "Enabled for boot (start request failed/deferred): $unit"
      fi
    else
      if systemctl start "$unit"; then
        echo "Enabled and started: $unit"
      else
        echo "Enabled for boot (start deferred/failing now): $unit"
      fi
    fi
  else
    echo "Failed to enable: $unit" >&2
  fi
}

remove_legacy_qr_units() {
  local -a legacy_units=()
  local -a legacy_dirs=()
  for path in "$SYSTEMD_DEST"/qr-*.service "$SYSTEMD_DEST"/qr-*.timer; do
    [[ -e "$path" ]] || continue
    legacy_units+=("$(basename "$path")")
  done
  for dir in "$SYSTEMD_DEST"/qr-*.service.d; do
    [[ -e "$dir" ]] || continue
    legacy_dirs+=("$dir")
  done
  if [[ ${#legacy_units[@]} -eq 0 && ${#legacy_dirs[@]} -eq 0 ]]; then
    return
  fi
  echo "Removing legacy qr units..."
  for unit in "${legacy_units[@]}"; do
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    rm -f "$SYSTEMD_DEST/$unit"
  done
  for dir in "${legacy_dirs[@]}"; do
    rm -rf "$dir"
  done
}

ENABLE_QUEUE=()

# Always install the main gate service
install_unit "$MAIN_UNIT_SRC"

if [[ $INSTALL_ALL -eq 1 ]]; then
  for unit in systemd/*.service systemd/*.timer; do
    [[ -e "$unit" ]] || continue
    install_unit "$unit"
  done
fi

if [[ ${#EXPLICIT_UNITS[@]} -gt 0 ]]; then
  for name in "${EXPLICIT_UNITS[@]}"; do
    if [[ -f "systemd/$name" ]]; then
      install_unit "systemd/$name"
    elif [[ -f "ops/systemd/$name" ]]; then
      install_unit "ops/systemd/$name"
    elif [[ -f "$name" ]]; then
      install_unit "$name"
    else
      echo "Warning: unit file not found for $name (looked in systemd/, ops/systemd/, .)" >&2
    fi
  done
fi

remove_legacy_qr_units

systemctl daemon-reload

for unit in "${ENABLE_QUEUE[@]}"; do
  enable_unit "$unit"
done

echo "Done. Enabled units:"
for unit in "${ENABLE_QUEUE[@]}"; do
  echo "  - $unit"
done
