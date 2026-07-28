#!/usr/bin/env bash
set -euo pipefail

readonly SERVICE_PREFIX="${QR_BITBANK_KEYCHAIN_PREFIX:-QuantRabbit.Bitbank}"
readonly KEYCHAIN_ACCOUNT="${QR_BITBANK_KEYCHAIN_ACCOUNT:-${USER:-quant_rabbit}}"
readonly SECURITY_BIN="${QR_SECURITY_BIN:-/usr/bin/security}"

usage() {
  cat <<'EOF'
Usage:
  scripts/bitbank-keychain.sh store-readonly
  scripts/bitbank-keychain.sh status
  scripts/bitbank-keychain.sh registry
  scripts/bitbank-keychain.sh run-readonly -- <command> [args...]

Stores a newly issued bitbank key and secret in macOS Keychain. The key must
have read-only/minimum permissions and no withdrawal permission. Secret values
are never put in shell history, command arguments, repo files, or stdout.

run-readonly exposes the two values only to the child process as
QR_BITBANK_API_KEY and QR_BITBANK_API_SECRET.
EOF
}

require_security_bin() {
  if [[ ! -x "$SECURITY_BIN" ]]; then
    echo "[bitbank-keychain] macOS security tool not found: ${SECURITY_BIN}" >&2
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
  echo "[bitbank-keychain] Enter ${label} for service=${service}; value will not be echoed."
  # Keep -w as the final argument so macOS prompts securely instead of receiving
  # the value through argv, stdin, a repo file, or shell history.
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

present_json() {
  if has_key "$1"; then
    printf 'true'
  else
    printf 'false'
  fi
}

store_readonly() {
  require_security_bin
  cat <<'EOF'
[bitbank-keychain] Store only a newly issued minimum-permission key.
[bitbank-keychain] Confirm in bitbank that trading and withdrawal are disabled.
EOF
  store_prompted "readonly_api_key" "bitbank read-only API key"
  store_prompted "readonly_api_secret" "bitbank read-only API secret"
  echo "[bitbank-keychain] stored read-only credential pair."
  status
}

status() {
  require_security_bin
  printf '[bitbank-keychain] service=%s account=%s present=%s\n' \
    "$(service_name "readonly_api_key")" "$KEYCHAIN_ACCOUNT" \
    "$(present_json "readonly_api_key")"
  printf '[bitbank-keychain] service=%s account=%s present=%s\n' \
    "$(service_name "readonly_api_secret")" "$KEYCHAIN_ACCOUNT" \
    "$(present_json "readonly_api_secret")"
}

registry() {
  require_security_bin
  local checked_at
  checked_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '{\n'
  printf '  "schema": "QR_BITBANK_CREDENTIAL_REGISTRY_V1",\n'
  printf '  "account": "%s",\n' "$KEYCHAIN_ACCOUNT"
  printf '  "api_type": "bitbank Private REST read-only",\n'
  printf '  "declared_permissions": ["ASSET_READ"],\n'
  printf '  "trading_enabled": false,\n'
  printf '  "withdrawal_enabled": false,\n'
  printf '  "checked_at_utc": "%s",\n' "$checked_at"
  printf '  "entries": [\n'
  printf '    {"service": "%s", "present": %s},\n' \
    "$(service_name "readonly_api_key")" "$(present_json "readonly_api_key")"
  printf '    {"service": "%s", "present": %s}\n' \
    "$(service_name "readonly_api_secret")" "$(present_json "readonly_api_secret")"
  printf '  ]\n'
  printf '}\n'
}

run_readonly() {
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
  if ! has_key "readonly_api_key" || ! has_key "readonly_api_secret"; then
    echo "[bitbank-keychain] read-only credential pair is not present." >&2
    echo "[bitbank-keychain] Run store-readonly after key rotation." >&2
    exit 2
  fi

  export QR_BITBANK_API_KEY
  export QR_BITBANK_API_SECRET
  QR_BITBANK_API_KEY="$(read_key "readonly_api_key")"
  QR_BITBANK_API_SECRET="$(read_key "readonly_api_secret")"
  exec "$@"
}

case "${1:-}" in
  store-readonly)
    store_readonly
    ;;
  status)
    status
    ;;
  registry)
    registry
    ;;
  run-readonly)
    shift
    run_readonly "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
