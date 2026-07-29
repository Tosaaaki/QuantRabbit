#!/bin/bash
# Install isolated bitbank sibling-strategy Paper Shadow launchd agents.

set -euo pipefail

ACTION="${1:-install}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${QR_CRYPTO_SHADOW_PYTHON:-}" ]]; then
  PYTHON="$QR_CRYPTO_SHADOW_PYTHON"
elif [[ -x /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 ]]; then
  PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
else
  PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
fi
RUNTIME_ROOT="$ROOT/data/crypto/strategy-lab"
LOG_ROOT="$ROOT/logs/crypto-strategy-lab"
DOMAIN="gui/$(id -u)"
EVALUATOR_LABEL="com.quantrabbit.crypto-strategy-lab-evaluator"
STRATEGIES=(
  "RANGE_MAKER_REVERSION"
  "BREAKOUT_CONFIRMATION"
  "TREND_PULLBACK_MAKER"
  "ORDER_BOOK_FADE"
  "ORDER_BOOK_FADE_COOLDOWN_5S"
  "ORDER_BOOK_FADE_MAKER_EXIT"
  "ORDER_BOOK_FADE_SL_FIXED_CONTROL"
  "ORDER_BOOK_FADE_SL_VOLATILITY"
  "ORDER_BOOK_FADE_SL_TIME"
)

slug_for() {
  echo "$1" | tr '[:upper:]_' '[:lower:]-'
}

label_for() {
  local strategy_slug
  strategy_slug="$(slug_for "$1")"
  echo "com.quantrabbit.crypto-strategy-${strategy_slug}-$2"
}

plist_for() {
  echo "$HOME/Library/LaunchAgents/$1.plist"
}

write_plist() {
  local target="$1"
  local label="$2"
  local strategy="$3"
  local mode="$4"
  local strategy_slug
  strategy_slug="$(slug_for "$strategy")"
  "$PYTHON" - "$target" "$label" "$strategy" "$mode" "$PYTHON" \
    "$ROOT" "$RUNTIME_ROOT/$strategy_slug" "$LOG_ROOT" <<'PY'
import os
import plistlib
import sys

(
    target,
    label,
    strategy,
    mode,
    python,
    root,
    runtime_root,
    log_root,
) = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [
        python,
        "-m",
        "quant_rabbit.crypto.cli",
        "shadow-service",
        "--mode",
        mode,
        "--runtime-root",
        runtime_root,
        "--strategy",
        strategy,
        "--entry-control",
        os.path.join(root, "config", "crypto_paper_entry_control_v1.json"),
        "--initial-cash-jpy",
        "10000",
        "--pair-limit",
        "2",
    ],
    "WorkingDirectory": root,
    "EnvironmentVariables": {
        "PYTHONPATH": os.path.join(root, "src"),
        "NO_EXECUTE": "true",
        "CRYPTO_LIVE_READY": "false",
        "WITHDRAWAL_ENABLED": "false",
        "CRYPTO_ORDER_AUTHORITY": "NONE",
        "QR_CRYPTO_FAST_TELEMETRY_EVERY_EVENTS": "25",
    },
    "KeepAlive": True,
    "RunAtLoad": True,
    "ProcessType": "Background",
    "StandardOutPath": os.path.join(log_root, f"{label}.out.log"),
    "StandardErrorPath": os.path.join(log_root, f"{label}.err.log"),
}
with open(target, "wb") as handle:
    plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=False)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

write_evaluator_plist() {
  local target="$1"
  "$PYTHON" - "$target" "$EVALUATOR_LABEL" "$PYTHON" "$ROOT" \
    "$RUNTIME_ROOT" "$LOG_ROOT" <<'PY'
import os
import plistlib
import sys

target, label, python, root, runtime_root, log_root = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [
        python,
        "-m",
        "quant_rabbit.crypto.cli",
        "strategy-lab-evaluate",
        "--runtime-root",
        runtime_root,
    ],
    "WorkingDirectory": root,
    "EnvironmentVariables": {
        "PYTHONPATH": os.path.join(root, "src"),
        "NO_EXECUTE": "true",
        "CRYPTO_LIVE_READY": "false",
        "WITHDRAWAL_ENABLED": "false",
        "CRYPTO_ORDER_AUTHORITY": "NONE",
    },
    "StartInterval": 300,
    "RunAtLoad": True,
    "ProcessType": "Background",
    "StandardOutPath": os.path.join(log_root, f"{label}.out.log"),
    "StandardErrorPath": os.path.join(log_root, f"{label}.err.log"),
}
with open(target, "wb") as handle:
    plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=False)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

bootstrap_agent() {
  local plist="$1"
  local attempt
  for attempt in {1..20}; do
    if launchctl bootstrap "$DOMAIN" "$plist" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done
  echo "[crypto-strategy-lab] bootstrap failed plist=$plist" >&2
  return 1
}

all_agents() {
  local strategy
  local mode
  for strategy in "${STRATEGIES[@]}"; do
    for mode in spot margin; do
      echo "$strategy $mode $(label_for "$strategy" "$mode")"
    done
  done
}

case "$ACTION" in
  --check|check)
    while read -r strategy mode label; do
      temporary="$(mktemp "${TMPDIR:-/tmp}/crypto-strategy.XXXXXX")"
      write_plist "$temporary" "$label" "$strategy" "$mode"
      plutil -lint "$temporary" >/dev/null
      rm -f "$temporary"
    done < <(all_agents)
    temporary="$(mktemp "${TMPDIR:-/tmp}/crypto-strategy.XXXXXX")"
    write_evaluator_plist "$temporary"
    plutil -lint "$temporary" >/dev/null
    rm -f "$temporary"
    echo "[crypto-strategy-lab] preflight OK root=$ROOT python=$PYTHON"
    ;;
  status)
    while read -r _strategy _mode label; do
      launchctl print "$DOMAIN/$label" 2>/dev/null \
        | sed -n '1,16p' || echo "$label not-loaded"
    done < <(all_agents)
    launchctl print "$DOMAIN/$EVALUATOR_LABEL" 2>/dev/null \
      | sed -n '1,16p' || echo "$EVALUATOR_LABEL not-loaded"
    ;;
  uninstall|stop)
    while read -r _strategy _mode label; do
      plist="$(plist_for "$label")"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      if [[ "$ACTION" == "uninstall" ]]; then
        rm -f "$plist"
      fi
    done < <(all_agents)
    evaluator_plist="$(plist_for "$EVALUATOR_LABEL")"
    launchctl bootout "$DOMAIN/$EVALUATOR_LABEL" 2>/dev/null || true
    if [[ "$ACTION" == "uninstall" ]]; then
      rm -f "$evaluator_plist"
    fi
    echo "[crypto-strategy-lab] $ACTION complete"
    ;;
  install|start)
    mkdir -p "$HOME/Library/LaunchAgents" "$RUNTIME_ROOT" "$LOG_ROOT"
    while read -r strategy mode label; do
      plist="$(plist_for "$label")"
      temporary="$(mktemp "$HOME/Library/LaunchAgents/.$label.XXXXXX")"
      write_plist "$temporary" "$label" "$strategy" "$mode"
      plutil -lint "$temporary" >/dev/null
      chmod 600 "$temporary"
      mv -f "$temporary" "$plist"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      bootstrap_agent "$plist"
    done < <(all_agents)
    evaluator_plist="$(plist_for "$EVALUATOR_LABEL")"
    temporary="$(
      mktemp "$HOME/Library/LaunchAgents/.$EVALUATOR_LABEL.XXXXXX"
    )"
    write_evaluator_plist "$temporary"
    plutil -lint "$temporary" >/dev/null
    chmod 600 "$temporary"
    mv -f "$temporary" "$evaluator_plist"
    launchctl bootout "$DOMAIN/$EVALUATOR_LABEL" 2>/dev/null || true
    bootstrap_agent "$evaluator_plist"
    echo "[crypto-strategy-lab] started ${#STRATEGIES[@]} strategies x Spot/Margin"
    ;;
  install-one|start-one)
    requested_strategy="${2:-}"
    allowed=false
    for strategy in "${STRATEGIES[@]}"; do
      if [[ "$strategy" == "$requested_strategy" ]]; then
        allowed=true
        break
      fi
    done
    if [[ "$allowed" != true ]]; then
      echo "unknown configured strategy: $requested_strategy" >&2
      exit 2
    fi
    mkdir -p "$HOME/Library/LaunchAgents" "$RUNTIME_ROOT" "$LOG_ROOT"
    for mode in spot margin; do
      label="$(label_for "$requested_strategy" "$mode")"
      plist="$(plist_for "$label")"
      temporary="$(mktemp "$HOME/Library/LaunchAgents/.$label.XXXXXX")"
      write_plist "$temporary" "$label" "$requested_strategy" "$mode"
      plutil -lint "$temporary" >/dev/null
      chmod 600 "$temporary"
      mv -f "$temporary" "$plist"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
      bootstrap_agent "$plist"
    done
    echo "[crypto-strategy-lab] started $requested_strategy x Spot/Margin"
    ;;
  *)
    echo "usage: $0 [install|start|install-one STRATEGY|start-one STRATEGY|stop|status|uninstall|--check]" >&2
    exit 2
    ;;
esac
