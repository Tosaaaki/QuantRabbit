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
# It never trades, never touches OANDA, never edits trading state. Escalation
# tiers when the trader is silent past the threshold (2026-06-11, after the
# 55.6h silent stop 06-09 02:22 JST -> 06-11 10:00 JST went unnoticed through
# 19 hours of local-notification-only alerts):
#   1. local macOS notification + append-only alert log (always)
#   2. relaunch Codex.app when its process is gone (QR_HEARTBEAT_OPEN_CODEX=0
#      to disable) — recovers the app-crash/app-quit cause; a credits-exhausted
#      scheduler keeps its process alive, so the relaunch is a no-op there
#   3. Slack push via the repo .env.local bot token (opt-in:
#      QR_HEARTBEAT_SLACK_ENABLE=1) — disabled by default to honor the
#      2026-05-30 operator directive that the live trader must not post
#      routine cycle traffic to Slack; a dead-trader page is the operator's
#      call to re-enable
# Install via scripts/install-trader-heartbeat-watch.sh (launchd, 10-min tick).

set -u

LIVE_ROOT="${QR_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
STATE_FILE="${QR_WEEKEND_STATE:-$HOME/.codex/quant_rabbit_weekend_task_state.json}"
ALERT_LOG="${QR_HEARTBEAT_ALERT_LOG:-$LIVE_ROOT/logs/heartbeat_alerts.log}"
STAMP_FILE="${QR_HEARTBEAT_STAMP:-$HOME/.cache/qr_trader_heartbeat_last_alert}"
# 3 × 360-minute supervision cadence + 5 min slack. The cadence is the qr-trader RRULE
# interval; change both together.
MAX_SILENCE_SECONDS="${QR_HEARTBEAT_MAX_SILENCE:-65100}"
ALERT_THROTTLE_SECONDS="${QR_HEARTBEAT_THROTTLE:-3600}"

now_epoch=$(date +%s)

# --- 1. Skip when the weekend switcher has intentionally paused the fleet ---
if [ -f "$STATE_FILE" ]; then
  mode=$(/usr/bin/python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('mode',''))" "$STATE_FILE" 2>/dev/null)
  if [ "$mode" = "paused" ]; then
    exit 0
  fi
fi

# --- 2. Skip only while the deterministic New York weekly calendar says the
#        FX market is closed. UNKNOWN/import failure continues the heartbeat so
#        a broken market-status dependency cannot hide a dead scheduler. ---
market_state=$(PYTHONPATH="$LIVE_ROOT/src" /usr/bin/python3 -c \
  'from quant_rabbit.analysis.market_status import compute_market_status; print("OPEN" if compute_market_status().is_fx_open else "CLOSED")' \
  2>/dev/null || echo UNKNOWN)
if [ "$market_state" = "CLOSED" ]; then exit 0; fi

# --- 3. Heartbeat: prefer the sealed, tuning-only AI supervision artifact.
# Legacy trader-cycle artifacts remain a fallback during cutover, but they may
# not mask a stale supervisor once a valid authority-NONE artifact exists. ---
newest_mtime=0
supervision_artifact="$LIVE_ROOT/data/ai_regime_supervision.json"
supervision_present=0
if [ -f "$supervision_artifact" ]; then
  supervision_present=1
  supervision_epoch=$(/usr/bin/python3 -c 'import datetime,hashlib,json,sys;p=json.load(open(sys.argv[1]));s=p.pop("contract_sha256","");v=p.get("schema_version");ts=datetime.datetime.fromisoformat(str(p.get("generated_at_utc","")).replace("Z","+00:00"));raw=json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode();ok=p.get("contract")=="QR_AI_REGIME_SUPERVISION_V1" and type(v) is int and v==1 and ts.tzinfo is not None and int(ts.timestamp())<=int(sys.argv[2]) and p.get("ai_role")=="REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY" and p.get("ai_order_authority")=="NONE" and p.get("live_permission") is False and p.get("broker_mutation_allowed") is False and isinstance(p.get("pairs"),dict) and len(s)==64 and s==hashlib.sha256(raw).hexdigest();print(int(ts.timestamp())) if ok else sys.exit(1)' "$supervision_artifact" "$now_epoch" 2>/dev/null || echo 0)
  if [ "$supervision_epoch" -gt 0 ]; then
    newest_mtime=$supervision_epoch
  fi
fi

if [ "$supervision_present" -eq 0 ]; then
  for f in \
    "$LIVE_ROOT/data/codex_trader_decision_response.json" \
    "$LIVE_ROOT/data/broker_snapshot.json" \
    "$LIVE_ROOT/logs/trader_journal.jsonl"; do
    if [ -f "$f" ]; then
      m=$(stat -f %m "$f" 2>/dev/null || echo 0)
      if [ "$m" -gt "$newest_mtime" ]; then newest_mtime=$m; fi
    fi
  done
fi

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

# --- 5. Self-heal: relaunch Codex.app when its process is gone. The six-hour
#        scheduler lives inside the app, so an app crash/quit IS a dead
#        trader; relaunching restores the already-enabled automation without
#        touching task state. When the process is alive (e.g. the 2026-06-09
#        credits exhaustion) this is a no-op.
if [ "${QR_HEARTBEAT_OPEN_CODEX:-1}" = "1" ]; then
  if ! /usr/bin/pgrep -xq Codex; then
    /usr/bin/open -a Codex 2>/dev/null \
      && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) self-heal: relaunched Codex.app" >> "$ALERT_LOG" \
      || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) self-heal: failed to relaunch Codex.app" >> "$ALERT_LOG"
  fi
fi

# --- 6. Slack escalation (opt-in). Reads the untracked repo .env.local
#        (QR_SLACK_BOT_TOKEN + QR_SLACK_CHANNEL_COMMANDS / QR_SLACK_CHANNEL_ID);
#        never store the token in this script. Honors the 2026-05-30
#        no-routine-Slack directive by staying off unless the operator
#        explicitly enables dead-trader paging.
if [ "${QR_HEARTBEAT_SLACK_ENABLE:-0}" = "1" ] && [ -f "$LIVE_ROOT/.env.local" ]; then
  slack_token=$(/usr/bin/awk -F= '$1=="QR_SLACK_BOT_TOKEN"{print $2}' "$LIVE_ROOT/.env.local" | tr -d '"' | tr -d "'")
  slack_channel=$(/usr/bin/awk -F= '$1=="QR_SLACK_CHANNEL_COMMANDS"{print $2}' "$LIVE_ROOT/.env.local" | tr -d '"' | tr -d "'")
  if [ -z "$slack_channel" ]; then
    slack_channel=$(/usr/bin/awk -F= '$1=="QR_SLACK_CHANNEL_ID"{print $2}' "$LIVE_ROOT/.env.local" | tr -d '"' | tr -d "'")
  fi
  if [ -n "$slack_token" ] && [ -n "$slack_channel" ]; then
    /usr/bin/curl -sf -m 15 -X POST "https://slack.com/api/chat.postMessage" \
      -H "Authorization: Bearer ${slack_token}" \
      -H "Content-Type: application/json; charset=utf-8" \
      -d "{\"channel\":\"${slack_channel}\",\"text\":\":rotating_light: ${msg}\"}" \
      > /dev/null 2>&1 \
      && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) escalation: Slack alert posted" >> "$ALERT_LOG" \
      || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) escalation: Slack alert failed" >> "$ALERT_LOG"
  fi
fi

exit 0
