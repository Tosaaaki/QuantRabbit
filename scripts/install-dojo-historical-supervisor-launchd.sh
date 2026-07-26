#!/bin/zsh
set -euo pipefail

LABEL="com.quantrabbit.dojo-historical-supervisor-r12"
REPO_ROOT="/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
SUPERVISOR_WRAPPER="$REPO_ROOT/scripts/run-dojo-historical-supervisor-launchd.py"
RUN_CONTROL="$REPO_ROOT/config/dojo_g2_parallel_rooms_run_control_v6.json"
OUTPUT_ROOT="/Users/tossaki/App/QuantRabbit-live/logs/dojo-historical/g2-parallel-rooms-20260723-r12/supervisor"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL_SECONDS=3600
TIMEOUT_SECONDS=180

usage() {
  print "usage: $0 [--check|--status|--uninstall]"
}

mode="install"
if (( $# > 1 )); then
  usage >&2
  exit 64
fi
if (( $# == 1 )); then
  case "$1" in
    --check) mode="check" ;;
    --status) mode="status" ;;
    --uninstall) mode="uninstall" ;;
    *) usage >&2; exit 64 ;;
  esac
fi

if [[ "$mode" == "status" ]]; then
  [[ -f "$PLIST" ]] && plist_state="present" || plist_state="missing"
  if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
    launchd_state="loaded"
  else
    launchd_state="not-loaded"
  fi
  print "label=$LABEL plist=$plist_state launchd=$launchd_state interval_seconds=$INTERVAL_SECONDS timeout_seconds=$TIMEOUT_SECONDS"
  exit 0
fi

if [[ "$mode" == "uninstall" ]]; then
  launchctl bootout "gui/$UID" "$PLIST" >/dev/null 2>&1 || true
  if [[ -f "$PLIST" ]]; then
    backup_dir="$HOME/.codex/backups/dojo-historical-supervisor-r12-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$backup_dir"
    mv "$PLIST" "$backup_dir/"
    print "archived=$backup_dir/$LABEL.plist"
  fi
  exit 0
fi

[[ -x "$PYTHON" ]] || { print "missing python: $PYTHON" >&2; exit 2; }
[[ -f "$SUPERVISOR_WRAPPER" ]] || { print "missing supervisor wrapper: $SUPERVISOR_WRAPPER" >&2; exit 2; }
[[ -f "$RUN_CONTROL" ]] || { print "missing run control: $RUN_CONTROL" >&2; exit 2; }
"$PYTHON" -m py_compile "$SUPERVISOR_WRAPPER"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$HOME/Library/LaunchAgents"
temp_plist="$(mktemp "$HOME/Library/LaunchAgents/.$LABEL.XXXXXX")"
trap 'rm -f "$temp_plist"' EXIT

plutil -create xml1 "$temp_plist"
plutil -insert Label -string "$LABEL" "$temp_plist"
plutil -insert ProgramArguments -json '[]' "$temp_plist"
plutil -insert ProgramArguments.0 -string "$PYTHON" "$temp_plist"
plutil -insert ProgramArguments.1 -string "$SUPERVISOR_WRAPPER" "$temp_plist"
plutil -insert ProgramArguments.2 -string "--run-control" "$temp_plist"
plutil -insert ProgramArguments.3 -string "$RUN_CONTROL" "$temp_plist"
plutil -insert ProgramArguments.4 -string "--timeout-seconds" "$temp_plist"
plutil -insert ProgramArguments.5 -string "$TIMEOUT_SECONDS" "$temp_plist"
plutil -insert WorkingDirectory -string "$REPO_ROOT" "$temp_plist"
plutil -insert EnvironmentVariables -json '{}' "$temp_plist"
plutil -insert EnvironmentVariables.PYTHONPATH -string "$REPO_ROOT/src" "$temp_plist"
plutil -insert RunAtLoad -bool true "$temp_plist"
plutil -insert StartInterval -integer "$INTERVAL_SECONDS" "$temp_plist"
plutil -insert ProcessType -string "Background" "$temp_plist"
plutil -insert LowPriorityIO -bool true "$temp_plist"
plutil -insert StandardOutPath -string "$OUTPUT_ROOT/launchd.stdout.log" "$temp_plist"
plutil -insert StandardErrorPath -string "$OUTPUT_ROOT/launchd.stderr.log" "$temp_plist"
plutil -lint "$temp_plist" >/dev/null

if [[ "$mode" == "check" ]]; then
  print "check=pass label=$LABEL interval_seconds=$INTERVAL_SECONDS timeout_seconds=$TIMEOUT_SECONDS paper_only=true order_authority=NONE"
  exit 0
fi

if [[ -f "$PLIST" ]]; then
  backup_dir="$HOME/.codex/backups/dojo-historical-supervisor-r12-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$backup_dir"
  cp -p "$PLIST" "$backup_dir/"
fi
chmod 600 "$temp_plist"
mv "$temp_plist" "$PLIST"
trap - EXIT
launchctl bootout "gui/$UID" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID" "$PLIST"
launchctl kickstart -k "gui/$UID/$LABEL"
launchctl print "gui/$UID/$LABEL" >/dev/null
print "installed=$PLIST label=$LABEL interval_seconds=$INTERVAL_SECONDS timeout_seconds=$TIMEOUT_SECONDS paper_only=true order_authority=NONE"
