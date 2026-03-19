#!/bin/bash
# Setup live_monitor.py as a launchd service (runs every 30 seconds)
# Usage: bash scripts/trader_tools/setup_live_monitor.sh

set -e
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.quantrabbit.live-monitor.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.quantrabbit.live-monitor.plist"

# Stop if already loaded
launchctl unload "$PLIST_DST" 2>/dev/null || true

# Copy and load
cp "$PLIST_SRC" "$PLIST_DST"
launchctl load "$PLIST_DST"

echo "[OK] live_monitor loaded. Runs every 30s."
echo "  Output: logs/live_monitor.json"
echo "  Errors: logs/live_monitor_err.log"
echo ""
echo "  Stop:   launchctl unload ~/Library/LaunchAgents/com.quantrabbit.live-monitor.plist"
echo "  Start:  launchctl load ~/Library/LaunchAgents/com.quantrabbit.live-monitor.plist"
