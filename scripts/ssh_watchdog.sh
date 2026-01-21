#!/usr/bin/env bash
set -euo pipefail

TAG="qr-ssh-watchdog"

SSH_UNIT=""
if systemctl list-unit-files --type=service | grep -q '^ssh\.service'; then
  SSH_UNIT="ssh"
elif systemctl list-unit-files --type=service | grep -q '^sshd\.service'; then
  SSH_UNIT="sshd"
fi

if [[ -n "$SSH_UNIT" ]]; then
  if ! systemctl is-active --quiet "$SSH_UNIT"; then
    logger -t "$TAG" "$SSH_UNIT inactive; restarting"
    systemctl restart "$SSH_UNIT" || true
  fi
fi

GUEST_UNIT="google-guest-agent"
if systemctl list-unit-files --type=service | grep -q "^${GUEST_UNIT}\\.service"; then
  if ! systemctl is-active --quiet "$GUEST_UNIT"; then
    logger -t "$TAG" "$GUEST_UNIT inactive; restarting"
    systemctl restart "$GUEST_UNIT" || true
  fi
fi

PORT_LISTEN_OK=0
if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq '(:22)$'; then
    PORT_LISTEN_OK=1
  fi
elif command -v netstat >/dev/null 2>&1; then
  if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(:22)$'; then
    PORT_LISTEN_OK=1
  fi
fi

if [[ -n "$SSH_UNIT" && "$PORT_LISTEN_OK" -eq 0 ]]; then
  logger -t "$TAG" "port 22 not listening; restarting $SSH_UNIT"
  systemctl restart "$SSH_UNIT" || true
fi
