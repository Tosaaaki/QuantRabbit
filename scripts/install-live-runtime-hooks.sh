#!/usr/bin/env bash
set -euo pipefail

readonly DEV_ROOT="${QR_SYNC_DEV_ROOT:-/Users/tossaki/App/QuantRabbit}"
readonly HOOK_DIR="$(git -C "$DEV_ROOT" rev-parse --git-path hooks)"
readonly HOOK_PATH="${HOOK_DIR}/post-commit"

mkdir -p "$HOOK_DIR"

if [[ -f "$HOOK_PATH" ]] && ! grep -Fq "QuantRabbit live runtime sync" "$HOOK_PATH"; then
  backup="${HOOK_PATH}.backup.$(date +%Y%m%d%H%M%S)"
  cp "$HOOK_PATH" "$backup"
  echo "[install-live-runtime-hooks] existing post-commit hook backed up to $backup"
fi

cat > "$HOOK_PATH" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit live runtime sync
repo_root="$(git rev-parse --show-toplevel)"
"${repo_root}/scripts/sync-live-runtime.sh"
HOOK

chmod +x "$HOOK_PATH"
echo "[install-live-runtime-hooks] installed $HOOK_PATH"
