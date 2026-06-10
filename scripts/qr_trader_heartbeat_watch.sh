#!/bin/bash
# qr_trader_heartbeat_watch.sh — dead-scheduler detector for the live trader.
#
# Why this exists (2026-06-10): the Codex scheduler kept firing qr-trader every
# 20 minutes after 2026-06-09 02:00 JST, but every run exited in ~2s with zero
# tokens because the plan's credits were exhausted. The automation looked
# ACTIVE while no cycle wrote a single artifact for ~36 hours and nobody was
# notified. This watchdog checks the live runtime's artifact heartbeat and
# raises a local notification when the trader has been silent for longer than
# three scheduled cycles during an open FX market window.
#
# It is notify-only: it never trades, never touches OANDA, never edits state.
# Install via scripts/install-trader-heartbeat-watch.sh (launchd, 10-min tick).

set -u

LIVE_ROOT="${QR_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
STATE_FILE="${QR_WEEKEND_STATE:-$HOME/.codex/quant_rabbit_weekend_task_state.json}"
ALERT_LOG="${QR_HEARTBEAT_ALERT_LOG:-$LIVE_ROOT/logs/heartbeat_alerts.log}"
STAMP_FILE="${QR_HEARTBEAT_STAMP:-$HOME/.cache/qr_trader_heartbeat_last_alert}"
# 3 × 20-minute cadence + 5 min slack. The cadence is the qr-trader RRULE
# interval; change both together.
MAX_SILENCE_SECONDS="${QR_HEARTBEAT_MAX_SILENCE:-3900}"
ALERT_THROTTLE_SECONDS="${QR_HEARTBEAT_THROTTLE:-3600}"

now_epoch=$(date +%s)

# --- 1. Skip when the weekend switcher has intentionally paused the fleet ---
if [ -f "$STATE_FILE" ]; then
  mode=$(/usr/bin/python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('mode',''))" "$STATE_FILE" 2>/dev/null)
  if [ "$mode" = "paused" ]; then
    exit 0
  fi
fi

# --- 2. Skip the FX weekend close window (JST), mirroring the weekend
#        switcher schedule: off Saturday 06:00 JST, on Monday 07:00 JST. ---
jst_dow=$(TZ=Asia/Tokyo date +%u)   # 1=Mon … 6=Sat 7=Sun
jst_hm=$(TZ=Asia/Tokyo date +%H%M)
if [ "$jst_dow" = "6" ] && [ "$jst_hm" -ge 0600 ]; then exit 0; fi
if [ "$jst_dow" = "7" ]; then exit 0; fi
if [ "$jst_dow" = "1" ] && [ "$jst_hm" -lt 0700 ]; then exit 0; fi

# --- 3. Heartbeat: newest artifact the trader cycle always writes ---
newest_mtime=0
for f in \
  "$LIVE_ROOT/data/codex_trader_decision_response.json" \
  "$LIVE_ROOT/data/broker_snapshot.json" \
  "$LIVE_ROOT/logs/trader_journal.jsonl"; do
  if [ -f "$f" ]; then
    m=$(stat -f %m "$f" 2>/dev/null || echo 0)
    if [ "$m" -gt "$newest_mtime" ]; then newest_mtime=$m; fi
  fi
done

if [ "$newest_mtime" -eq 0 ]; then
  silence="unknown (no heartbeat artifacts found)"
  silent_for=$MAX_SILENCE_SECONDS
else
  silent_for=$(( now_epoch - newest_mtime ))
  silence="$(( silent_for / 60 )) min"
fi

if [ "$silent_for" -lt "$MAX_SILENCE_SECONDS" ]; then
  exit 0
fi

# --- 4. Throttled alert ---
if [ -f "$STAMP_FILE" ]; then
  last_alert=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
  if [ $(( now_epoch - last_alert )) -lt "$ALERT_THROTTLE_SECONDS" ]; then
    exit 0
  fi
fi
mkdir -p "$(dirname "$STAMP_FILE")" "$(dirname "$ALERT_LOG")"
echo "$now_epoch" > "$STAMP_FILE"

msg="QuantRabbit trader silent for ${silence} during open market. Check Codex credits / scheduler."
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $msg" >> "$ALERT_LOG"
/usr/bin/osascript -e "display notification \"$msg\" with title \"QR trader heartbeat\" sound name \"Basso\"" 2>/dev/null

exit 0
