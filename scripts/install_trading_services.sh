#!/usr/bin/env bash
set -euo pipefail

# Install and enable QuantRabbit systemd units so they survive VM reboots.
# Usage:
#   sudo bash scripts/install_trading_services.sh [--repo /home/tossaki/QuantRabbit] [--all] [--units "quant-impulse-break-s5.service quant-impulse-break-s5-exit.service"]
# Defaults: only installs/enables the main gate `quantrabbit.service`.

REPO_DIR="/home/tossaki/QuantRabbit"
INSTALL_ALL=0
EXPLICIT_UNITS=()

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
  if [[ ! -f "$src" ]]; then
    echo "Skip (not found): $src"
    return
  fi
  install -m 0644 "$src" "$dest"
  echo "Installed: $dest"
  ENABLE_QUEUE+=("$(basename "$dest")")
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
  systemctl enable --now "$unit"
  echo "Enabled and started: $unit"
done

echo "Done. Enabled units:"
for unit in "${ENABLE_QUEUE[@]}"; do
  echo "  - $unit"
done
