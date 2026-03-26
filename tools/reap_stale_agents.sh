#!/bin/bash
# Reap stale claude-code agent processes that didn't exit cleanly.
# Kills any claude-code process older than 4 minutes, EXCEPT:
#   - The process whose PID matches the current global_agent lock holder
#   - Processes with --resume flag (interactive sessions)
# Also releases stale locks whose PID was reaped.
#
# Intended to run every 2 minutes via launchd.

LOCK_META="/Users/tossaki/App/QuantRabbit/logs/locks/global_agent.d/meta.json"
LOCK_SCRIPT="/Users/tossaki/App/QuantRabbit/tools/task_lock.py"
PROTECTED_PID=""

# Get the PID of the currently active lock holder
if [ -f "$LOCK_META" ]; then
    PROTECTED_PID=$(python3 -c "import json; print(json.load(open('$LOCK_META')).get('pid',''))" 2>/dev/null)
fi

KILLED=0
KILLED_PIDS=""
for pid in $(pgrep -f "claude-code.*claude.*--permission-mode"); do
    # Skip if this is the protected (actively running) process
    if [ "$pid" = "$PROTECTED_PID" ]; then
        continue
    fi

    # Skip interactive sessions (--resume flag = user conversation)
    if ps -p "$pid" -o command= 2>/dev/null | grep -q "\-\-resume"; then
        continue
    fi

    # Check age: kill if older than 4 minutes (240 seconds)
    ELAPSED=$(ps -p "$pid" -o etime= 2>/dev/null | awk '{
        gsub(/^[ \t]+/, "");
        n = split($0, a, ":");
        if (n == 2) print a[1]*60 + a[2];
        else if (n == 3) print a[1]*3600 + a[2]*60 + a[3];
        else print 0
    }')

    if [ -n "$ELAPSED" ] && [ "$ELAPSED" -gt 240 ]; then
        kill "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
        KILLED_PIDS="$KILLED_PIDS $pid"
    fi
done

# If we killed the lock holder, release the lock
if [ -n "$PROTECTED_PID" ] && echo "$KILLED_PIDS" | grep -qw "$PROTECTED_PID"; then
    python3 "$LOCK_SCRIPT" release global_agent 2>/dev/null
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') released lock (holder pid=$PROTECTED_PID was reaped)" >> /Users/tossaki/App/QuantRabbit/logs/reaper.log
fi

# Also check: if lock holder PID is dead (not just reaped), release lock
if [ -n "$PROTECTED_PID" ] && ! kill -0 "$PROTECTED_PID" 2>/dev/null; then
    python3 "$LOCK_SCRIPT" release global_agent 2>/dev/null
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') released orphan lock (pid=$PROTECTED_PID dead)" >> /Users/tossaki/App/QuantRabbit/logs/reaper.log
fi

if [ "$KILLED" -gt 0 ]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') reaped $KILLED stale agent(s)" >> /Users/tossaki/App/QuantRabbit/logs/reaper.log
fi
