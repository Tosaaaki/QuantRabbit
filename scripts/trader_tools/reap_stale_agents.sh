#!/bin/bash
# reap_stale_agents.sh — launchd-driven zombie killer (runs every 60s)
#
# This is the ONLY reliable zombie defense because:
# - Runs outside Claude Code sessions (can't zombie itself)
# - macOS kernel guarantees execution via launchd
# - Pure bash, no model inference, no API calls
#
# Kills Claude Code scheduled-task sessions older than 8 minutes (480s).
# The trader session should complete within 5 minutes. 8 min = generous buffer.

MAX_AGE=480
COUNT=0

# macOS ps uses etime (not etimes). Format: [[DD-]HH:]MM:SS → convert to seconds
etime_to_sec() {
    local t="$1"
    local days=0 hours=0 mins=0 secs=0
    # DD-HH:MM:SS
    if echo "$t" | grep -q '-'; then
        days=$(echo "$t" | cut -d'-' -f1)
        t=$(echo "$t" | cut -d'-' -f2)
    fi
    local parts
    IFS=':' read -ra parts <<< "$t"
    local n=${#parts[@]}
    secs=${parts[$((n-1))]}
    [ "$n" -ge 2 ] && mins=${parts[$((n-2))]}
    [ "$n" -ge 3 ] && hours=${parts[$((n-3))]}
    echo $(( days*86400 + hours*3600 + mins*60 + secs ))
}

for pid in $(pgrep -f "claude" 2>/dev/null); do
    ETIME=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$ETIME" ] && continue

    ELAPSED=$(etime_to_sec "$ETIME")
    [ "$ELAPSED" -lt "$MAX_AGE" ] && continue

    CMDLINE=$(ps -o args= -p "$pid" 2>/dev/null)

    # Only kill scheduled-task sessions (have disallowedTools or scheduled-tasks in args)
    if echo "$CMDLINE" | grep -q "disallowedTools\|scheduled-tasks"; then
        kill -9 "$pid" 2>/dev/null
        COUNT=$((COUNT + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') KILLED pid=$pid elapsed=${ELAPSED}s"
    fi
done

if [ "$COUNT" -gt 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Reaped $COUNT zombie(s)"

    # Clean stale lock if it exists and its PID is dead
    LOCK="/Users/tossaki/App/QuantRabbit/logs/.trader_lock"
    if [ -f "$LOCK" ]; then
        OLD_PID=$(awk '{print $2}' "$LOCK")
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            rm -f "$LOCK"
            echo "$(date '+%Y-%m-%d %H:%M:%S') Removed stale lock (pid=$OLD_PID dead)"
        fi
    fi
fi
