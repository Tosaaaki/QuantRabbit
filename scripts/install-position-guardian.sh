#!/usr/bin/env bash
# Install (or reinstall) the QuantRabbit position guardian as a launchd agent.
# It runs the live position-only guard frequently so open trades can be managed
# without waiting for a full new-entry trader cycle.

set -euo pipefail

LIVE_ROOT="${QR_SYNC_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
DEV_ROOT="${QR_SYNC_DEV_ROOT:-/Users/tossaki/App/QuantRabbit}"
MAIN_BRANCH="${QR_SYNC_MAIN_BRANCH:-main}"
SCRIPT="$LIVE_ROOT/scripts/run-position-guardian-live.sh"
LABEL="com.quantrabbit.position-guardian"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL="${QR_POSITION_GUARDIAN_INTERVAL:-30}"
ENV_FILE="${QR_OANDA_ENV_FILE:-.env.local}"
CHECK_ONLY=0

usage() {
  cat <<'USAGE'
Usage: scripts/install-position-guardian.sh [--check]

Install the QuantRabbit position guardian launchd agent. --check runs the same
activation preflight without writing a plist or loading launchd.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install-position-guardian] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

die() {
  echo "[install-position-guardian] $1" >&2
  exit "${2:-2}"
}

status_path() {
  local line="$1"
  printf '%s' "${line:3}"
}

is_report_path() {
  local path="$1"
  [[ "$path" == docs/*_report.md || "$path" == docs/*_report.close_reentry.md ]]
}

assert_only_report_drift() {
  local dirty=0 line path
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="$(status_path "$line")"
    if ! is_report_path "$path"; then
      echo "[install-position-guardian] blocking dirty live path: $line" >&2
      dirty=1
    fi
  done < <(git -C "$LIVE_ROOT" status --short --untracked-files=all)
  if [[ "$dirty" -ne 0 ]]; then
    die "live worktree must be clean except docs/*_report.md runtime drift." 3
  fi
}

env_file_path() {
  if [[ "$ENV_FILE" = /* ]]; then
    printf '%s\n' "$ENV_FILE"
  else
    printf '%s/%s\n' "$LIVE_ROOT" "$ENV_FILE"
  fi
}

validate_env_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing OANDA env file: $path" 2
  local required_key line value
  for required_key in QR_OANDA_ACCOUNT_ID QR_OANDA_TOKEN QR_OANDA_BASE_URL; do
    if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${required_key}[[:space:]]*=" "$path"; then
      die "missing ${required_key} in $path" 2
    fi
  done
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?QR_LIVE_ENABLED[[:space:]]*=" "$path" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    value="${line#*=}"
    value="${value%%#*}"
    value="$(printf '%s' "$value" | tr -d "[:space:]\"'")"
    case "$value" in
      0|1) ;;
      *) die "invalid QR_LIVE_ENABLED in $path; expected 0 or 1." 2 ;;
    esac
  fi
}

preflight() {
  [[ "$INTERVAL" =~ ^[0-9]+$ ]] || die "QR_POSITION_GUARDIAN_INTERVAL must be an integer >= 15 seconds." 2
  [[ "$INTERVAL" -ge 15 ]] || die "QR_POSITION_GUARDIAN_INTERVAL must be an integer >= 15 seconds." 2
  [[ -x "$SCRIPT" ]] || {
    echo "[install-position-guardian] missing executable live guardian script: $SCRIPT" >&2
    die "run scripts/sync-live-runtime.sh after committing the guardian change." 2
  }
  git -C "$LIVE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "missing live git worktree: $LIVE_ROOT" 2
  git -C "$DEV_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "missing development git repo: $DEV_ROOT" 2

  local live_head main_head
  live_head="$(git -C "$LIVE_ROOT" rev-parse HEAD)"
  main_head="$(git -C "$DEV_ROOT" rev-parse "$MAIN_BRANCH")"
  if [[ "$live_head" != "$main_head" ]]; then
    die "live HEAD $live_head does not match $MAIN_BRANCH $main_head; run scripts/sync-live-runtime.sh first." 4
  fi

  assert_only_report_drift
  validate_env_file "$(env_file_path)"
}

preflight

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "[install-position-guardian] preflight OK: live_root=$LIVE_ROOT interval=${INTERVAL}s plist=$PLIST"
  exit 0
fi

command -v launchctl >/dev/null 2>&1 || die "launchctl is not available." 2
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
