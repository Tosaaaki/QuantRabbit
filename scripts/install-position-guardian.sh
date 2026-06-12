#!/usr/bin/env bash
# Install (or reinstall) the QuantRabbit position guardian as a launchd agent.
# It runs the live position-only guard frequently so open trades can be managed
# without waiting for a full new-entry trader cycle.

set -euo pipefail

LIVE_ROOT="${QR_SYNC_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
SCRIPT="$LIVE_ROOT/scripts/run-position-guardian-live.sh"
LABEL="com.quantrabbit.position-guardian"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL="${QR_POSITION_GUARDIAN_INTERVAL:-30}"

if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 15 ]]; then
  echo "[install-position-guardian] QR_POSITION_GUARDIAN_INTERVAL must be an integer >= 15 seconds." >&2
  exit 2
fi
if [[ ! -x "$SCRIPT" ]]; then
  echo "[install-position-guardian] missing executable live guardian script: $SCRIPT" >&2
  echo "[install-position-guardian] run scripts/sync-live-runtime.sh after committing the guardian change." >&2
  exit 2
fi

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/tmp/$LABEL.out.log</string>
  <key>StandardErrorPath</key><string>/tmp/$LABEL.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed: $PLIST (every ${INTERVAL}s)"
