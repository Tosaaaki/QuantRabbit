#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit -> DecaBot operational bridge.
#
# DecaBot is a QuantRabbit-derived experiment, but it runs from a separate
# repository/root and a separate OANDA account. This script is deliberately an
# operator bridge only: it never imports QuantRabbit trading code and never
# routes DecaBot orders through the QuantRabbit live gateway.

readonly DECABOT_ROOT="${DECABOT_ROOT:-/Users/tossaki/App/DecaBot}"
readonly DECABOT_PYTHON="${DECABOT_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
readonly USER_DOMAIN="gui/$(id -u)"
readonly LABELS=(com.decabot.ai com.decabot.monitor com.decabot.review)

usage() {
  cat <<'EOF'
Usage: scripts/qr-decabot.sh <command>

Commands:
  status        Show DecaBot launchd state, current account summary, open trades, config, and latest AI cycle.
  logs [name]   Tail logs. name: ai | monitor | review | live | all. Default: all.
  cycle         Run one DecaBot-AI cycle now (live if DecaBot dry_run=false).
  monitor       Run one DecaBot monitor poll now.
  start         Load ai/monitor/review LaunchAgents. Does not load com.decabot.live.
  stop          Unload ai/monitor/review LaunchAgents. Does not touch com.decabot.live.
  print         Print launchd details for DecaBot LaunchAgents.
  root          Print the DecaBot root path.
  shell         Open an interactive shell in the DecaBot root.

Environment:
  DECABOT_ROOT    Override DecaBot root. Default: /Users/tossaki/App/DecaBot
  DECABOT_PYTHON  Override Python used for module commands.
EOF
}

require_root() {
  if [[ ! -d "$DECABOT_ROOT" ]]; then
    echo "[qr-decabot] DecaBot root not found: $DECABOT_ROOT" >&2
    exit 2
  fi
}

run_decabot_python() {
  require_root
  cd "$DECABOT_ROOT"
  "$DECABOT_PYTHON" "$@"
}

status() {
  require_root
  echo "DecaBot root: $DECABOT_ROOT"
  echo
  echo "launchd:"
  launchctl list | awk '/com\.decabot/ {print "  " $0}' || true
  echo
  run_decabot_python - <<'PY'
import json
from pathlib import Path

from decabot.ai_cycle import _load_state
from decabot.oanda import OandaClient

root = Path.cwd()
cfg_path = root / "data" / "ai_config.json"
state_path = root / "data" / "monitor_state.json"
ai_log = root / "logs" / "ai_cycle.jsonl"

print("ai_config:")
if cfg_path.exists():
    print(json.dumps(json.loads(cfg_path.read_text()), ensure_ascii=False, indent=2))
else:
    print("  missing")
print()

client = OandaClient.from_env(root / ".env")
summary = client.summary()
print("account:")
for key in ("NAV", "balance", "marginUsed", "marginAvailable", "lastTransactionID"):
    print(f"  {key}: {summary.get(key)}")
print()

trades = client.open_trades()
print(f"open_trades: {len(trades)}")
for trade in trades:
    sl = (trade.get("stopLossOrder") or {}).get("price")
    tp = (trade.get("takeProfitOrder") or {}).get("price")
    print(
        "  "
        f"{trade.get('id')} {trade.get('instrument')} {trade.get('currentUnits')}u "
        f"@{trade.get('price')} uPnL={trade.get('unrealizedPL')} "
        f"SL={sl} TP={tp}"
    )
print()

print("ai_state:")
print(json.dumps(_load_state(), ensure_ascii=False, indent=2)[:1800])
print()

print("monitor_state:")
if state_path.exists():
    print(state_path.read_text().strip())
else:
    print("  missing")
print()

print("latest_ai_cycle:")
if ai_log.exists():
    lines = [line for line in ai_log.read_text().splitlines() if line.strip()]
    print(lines[-1] if lines else "  empty")
else:
    print("  missing")
PY
}

tail_logs() {
  require_root
  local name="${1:-all}"
  case "$name" in
    ai)
      tail -n 80 "$DECABOT_ROOT/logs/launchd_ai.log" "$DECABOT_ROOT/logs/launchd_ai_err.log"
      ;;
    monitor)
      tail -n 80 "$DECABOT_ROOT/logs/launchd_monitor.log" "$DECABOT_ROOT/logs/launchd_monitor_err.log"
      ;;
    review)
      tail -n 80 "$DECABOT_ROOT/logs/launchd_review.log" "$DECABOT_ROOT/logs/launchd_review_err.log"
      ;;
    live)
      tail -n 80 "$DECABOT_ROOT/logs/launchd_live.log" "$DECABOT_ROOT/logs/launchd_live_err.log"
      ;;
    all)
      tail -n 40 \
        "$DECABOT_ROOT/logs/launchd_ai.log" \
        "$DECABOT_ROOT/logs/launchd_ai_err.log" \
        "$DECABOT_ROOT/logs/launchd_monitor.log" \
        "$DECABOT_ROOT/logs/launchd_monitor_err.log" \
        "$DECABOT_ROOT/logs/launchd_review.log" \
        "$DECABOT_ROOT/logs/launchd_review_err.log"
      ;;
    *)
      echo "[qr-decabot] unknown log name: $name" >&2
      usage >&2
      exit 2
      ;;
  esac
}

start_agents() {
  require_root
  local label
  for label in "${LABELS[@]}"; do
    launchctl bootout "$USER_DOMAIN" "$HOME/Library/LaunchAgents/${label}.plist" >/dev/null 2>&1 || true
    launchctl bootstrap "$USER_DOMAIN" "$HOME/Library/LaunchAgents/${label}.plist"
    launchctl enable "${USER_DOMAIN}/${label}" || true
  done
  launchctl list | awk '/com\.decabot/ {print $0}' || true
}

stop_agents() {
  require_root
  local label
  for label in "${LABELS[@]}"; do
    launchctl bootout "$USER_DOMAIN" "$HOME/Library/LaunchAgents/${label}.plist" >/dev/null 2>&1 || true
  done
  launchctl list | awk '/com\.decabot/ {print $0}' || true
}

print_agents() {
  local label
  for label in com.decabot.ai com.decabot.monitor com.decabot.review com.decabot.live; do
    echo "== ${label} =="
    launchctl print "${USER_DOMAIN}/${label}" 2>&1 | sed -n '1,120p'
  done
}

cmd="${1:-}"
case "$cmd" in
  status)
    status
    ;;
  logs)
    tail_logs "${2:-all}"
    ;;
  cycle)
    run_decabot_python -m decabot.ai_cycle --force
    ;;
  monitor)
    run_decabot_python -m decabot.monitor
    ;;
  start)
    start_agents
    ;;
  stop)
    stop_agents
    ;;
  print)
    print_agents
    ;;
  root)
    require_root
    printf '%s\n' "$DECABOT_ROOT"
    ;;
  shell)
    require_root
    cd "$DECABOT_ROOT"
    exec "${SHELL:-/bin/zsh}" -l
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "[qr-decabot] unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
