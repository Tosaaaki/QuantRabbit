#!/bin/bash
# reap_stale_agents.sh — launchd-driven trader supervisor (runs every 60s)
#
# Roles:
#   1. REAP:  Kill zombie scheduled-task processes (>480s old)
#   2. CLEAN: Remove stale lock files (PID dead)
#   3. ALERT: Slack notify if trader has been dead for >10 minutes
#
# Reliability:
#   - Runs outside Claude Code (can't zombie itself)
#   - macOS kernel guarantees execution via launchd
#   - Pure bash + python3 for Slack, no model inference, no API cost

PROJECT="/Users/tossaki/App/QuantRabbit"
LOCK="$PROJECT/logs/.trader_lock"
STATE="$PROJECT/collab_trade/state.md"
ALERT_FLAG="/tmp/quantrabbit-trader-dead-alerted"
MAX_AGE=480       # Kill processes older than 8 minutes
DEAD_THRESHOLD=600  # Alert if trader dead for >10 minutes
COUNT=0

# macOS etime format: [[DD-]HH:]MM:SS → seconds
# Uses 10# prefix to avoid octal interpretation of 08, 09
etime_to_sec() {
    local t="$1" days=0 hours=0 mins=0 secs=0
    if [[ "$t" == *-* ]]; then
        days=${t%%-*}
        t=${t#*-}
    fi
    IFS=':' read -ra parts <<< "$t"
    local n=${#parts[@]}
    secs=$((10#${parts[$((n-1))]}))
    [ "$n" -ge 2 ] && mins=$((10#${parts[$((n-2))]}))
    [ "$n" -ge 3 ] && hours=$((10#${parts[$((n-3))]}))
    echo $(( days*86400 + hours*3600 + mins*60 + secs ))
}

# ── PHASE 1: REAP zombie scheduled-task processes ──
for pid in $(pgrep -f "claude" 2>/dev/null); do
    ETIME=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$ETIME" ] && continue

    ELAPSED=$(etime_to_sec "$ETIME")
    [ -z "$ELAPSED" ] && continue
    [ "$ELAPSED" -lt "$MAX_AGE" ] && continue

    CMDLINE=$(ps -o args= -p "$pid" 2>/dev/null)

    # Only kill scheduled-task sessions (bypassPermissions = all scheduled tasks)
    if echo "$CMDLINE" | grep -q "bypassPermissions"; then
        kill "$pid" 2>/dev/null
        # Give 5s for graceful shutdown, then force kill
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        COUNT=$((COUNT + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') REAPED pid=$pid elapsed=${ELAPSED}s"
    fi
done

# ── PHASE 2: CLEAN stale lock ──
if [ -f "$LOCK" ]; then
    OLD_PID=$(awk '{print $2}' "$LOCK")
    if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
        rm -f "$LOCK" "$PROJECT/logs/.trader_start"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Removed stale lock (pid=$OLD_PID dead)"
    fi
fi

# ── PHASE 3: ALERT if trader is dead ──
# Check state.md age — if not updated in DEAD_THRESHOLD seconds, trader is stuck/dead
if [ -f "$STATE" ]; then
    NOW=$(date +%s)
    STATE_MTIME=$(stat -f %m "$STATE" 2>/dev/null || echo "$NOW")
    STATE_AGE=$(( NOW - STATE_MTIME ))

    if [ "$STATE_AGE" -gt "$DEAD_THRESHOLD" ]; then
        # Only alert once (don't spam every 60s)
        if [ ! -f "$ALERT_FLAG" ]; then
            touch "$ALERT_FLAG"
            MINS=$(( STATE_AGE / 60 ))
            echo "$(date '+%Y-%m-%d %H:%M:%S') ALERT: trader dead for ${MINS}min (state.md age=${STATE_AGE}s)"
            # Post to Slack
            cd "$PROJECT" && python3 tools/slack_post.py \
                "⚠️ Trader が ${MINS}分間停止中。state.md 更新なし。ゾンビ ${COUNT}個 kill済。自動復旧を試みます。" \
                --channel C0APAELAQDN 2>/dev/null
        fi
    else
        # Trader is alive — clear alert flag
        [ -f "$ALERT_FLAG" ] && rm -f "$ALERT_FLAG"
    fi
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING: state.md not found"
fi

# ── Summary ──
if [ "$COUNT" -gt 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Reaped $COUNT zombie(s)"
fi
