#!/usr/bin/env bash
set -euo pipefail

readonly SERVICE_PREFIX="${QR_WEBULL_KEYCHAIN_PREFIX:-QuantRabbit.Webull}"
readonly KEYCHAIN_ACCOUNT="${QR_WEBULL_KEYCHAIN_ACCOUNT:-${USER:-quant_rabbit}}"
readonly SECURITY_BIN="${QR_SECURITY_BIN:-/usr/bin/security}"

usage() {
  cat <<'EOF'
Usage:
  scripts/webull-keychain.sh store-login
  scripts/webull-keychain.sh store-openapi
  scripts/webull-keychain.sh status
  scripts/webull-keychain.sh run -- <command> [args...]

Stores Webull secrets in macOS Keychain without putting secret values in shell
history, command arguments, repo files, or stdout.

store-login stores the Webull login phone/password for operator reference only.
QuantRabbit OpenAPI trading does not use phone/password login.

store-openapi stores the official OpenAPI App Key, App Secret, and Account ID.

run loads stored OpenAPI values into the child process as QR_WEBULL_* env vars.
EOF
}

require_security_bin() {
  if [[ ! -x "$SECURITY_BIN" ]]; then
    echo "[webull-keychain] macOS security tool not found: ${SECURITY_BIN}" >&2
    exit 2
  fi
}

service_name() {
  printf '%s.%s' "$SERVICE_PREFIX" "$1"
}

store_prompted() {
  local key="$1"
  local label="$2"
  local service
  service="$(service_name "$key")"
  echo "[webull-keychain] Enter ${label} for service=${service}; value will not be echoed."
  # Keep -w as the final argument so security prompts for the secret instead of
  # receiving it through argv, stdin, repo files, or shell history.
  "$SECURITY_BIN" add-generic-password \
    -U \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$service" \
    -l "${SERVICE_PREFIX} ${label}" \
    -w >/dev/null
}

has_key() {
  "$SECURITY_BIN" find-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$(service_name "$1")" >/dev/null 2>&1
}

read_key() {
  "$SECURITY_BIN" find-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$(service_name "$1")" \
    -w
}

print_status_line() {
  local key="$1"
  local label="$2"
  if has_key "$key"; then
    printf '[webull-keychain] %-24s present=true\n' "$label"
  else
    printf '[webull-keychain] %-24s present=false\n' "$label"
  fi
}

store_login() {
  require_security_bin
  store_prompted "login_phone" "Webull login phone"
  store_prompted "login_password" "Webull login password"
  echo "[webull-keychain] stored login reference. QuantRabbit OpenAPI trading will not use login phone/password."
}

store_openapi() {
  require_security_bin
  store_prompted "openapi_app_key" "Webull OpenAPI App Key"
  store_prompted "openapi_app_secret" "Webull OpenAPI App Secret"
  store_prompted "openapi_account_id" "Webull OpenAPI Account ID"
  echo "[webull-keychain] stored OpenAPI credentials."
}

status() {
  require_security_bin
  print_status_line "login_phone" "login phone"
  print_status_line "login_password" "login password"
  print_status_line "openapi_app_key" "OpenAPI App Key"
  print_status_line "openapi_app_secret" "OpenAPI App Secret"
  print_status_line "openapi_account_id" "OpenAPI Account ID"
}

run_with_openapi_env() {
  require_security_bin
  if [[ "${1:-}" != "--" ]]; then
    usage >&2
    exit 2
  fi
  shift
  if [[ "$#" -eq 0 ]]; then
    usage >&2
    exit 2
  fi
  if ! has_key "openapi_app_key" || ! has_key "openapi_app_secret"; then
    echo "[webull-keychain] missing OpenAPI App Key/App Secret. Run: scripts/webull-keychain.sh store-openapi" >&2
    exit 2
  fi

  export QR_WEBULL_APP_KEY
  export QR_WEBULL_APP_SECRET
  export QR_WEBULL_ACCOUNT_ID
  QR_WEBULL_APP_KEY="$(read_key "openapi_app_key")"
  QR_WEBULL_APP_SECRET="$(read_key "openapi_app_secret")"
  if has_key "openapi_account_id"; then
    QR_WEBULL_ACCOUNT_ID="$(read_key "openapi_account_id")"
  fi

  exec "$@"
}

case "${1:-}" in
  store-login)
    store_login
    ;;
  store-openapi)
    store_openapi
    ;;
  status)
    status
    ;;
  run)
    shift
    run_with_openapi_env "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
