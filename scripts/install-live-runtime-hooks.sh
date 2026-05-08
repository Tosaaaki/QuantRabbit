#!/usr/bin/env bash
set -euo pipefail

readonly DEV_ROOT="${QR_SYNC_DEV_ROOT:-/Users/tossaki/App/QuantRabbit}"
readonly LIVE_ROOT="${QR_SYNC_LIVE_ROOT:-/Users/tossaki/App/QuantRabbit-live}"
readonly HOOK_DIR="$(git -C "$DEV_ROOT" rev-parse --git-path hooks)"
readonly POST_COMMIT_PATH="${HOOK_DIR}/post-commit"
readonly PRE_COMMIT_PATH="${HOOK_DIR}/pre-commit"

mkdir -p "$HOOK_DIR"

# ---------------------------------------------------------------------------
# post-commit: run sync-live-runtime.sh after every dev-side commit.
# ---------------------------------------------------------------------------
if [[ -f "$POST_COMMIT_PATH" ]] && ! grep -Fq "QuantRabbit live runtime sync" "$POST_COMMIT_PATH"; then
  backup="${POST_COMMIT_PATH}.backup.$(date +%Y%m%d%H%M%S)"
  cp "$POST_COMMIT_PATH" "$backup"
  echo "[install-live-runtime-hooks] existing post-commit hook backed up to $backup"
fi

cat > "$POST_COMMIT_PATH" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit live runtime sync
repo_root="$(git rev-parse --show-toplevel)"
"${repo_root}/scripts/sync-live-runtime.sh"
HOOK

chmod +x "$POST_COMMIT_PATH"
echo "[install-live-runtime-hooks] installed $POST_COMMIT_PATH"

# ---------------------------------------------------------------------------
# pre-commit: refuse non-doc commits in /Users/tossaki/App/QuantRabbit-live.
# Worktrees share hooks with the dev root; the script self-detects whether it
# was invoked from the live worktree and only enforces there.
#
# Why: sync-live-runtime.sh does `merge --ff-only main` from dev to live. A
# live-only commit is non-fast-forward → sync refuses → wrapper exits →
# routine cycles silently fail their pre-flight gate. (2026-05-08 incident:
# 8 cycles BLOCKED for ~80 min before the operator noticed.)
# ---------------------------------------------------------------------------
if [[ -f "$PRE_COMMIT_PATH" ]] && ! grep -Fq "QuantRabbit live pre-commit guard" "$PRE_COMMIT_PATH"; then
  backup="${PRE_COMMIT_PATH}.backup.$(date +%Y%m%d%H%M%S)"
  cp "$PRE_COMMIT_PATH" "$backup"
  echo "[install-live-runtime-hooks] existing pre-commit hook backed up to $backup"
fi

cat > "$PRE_COMMIT_PATH" <<HOOK
#!/usr/bin/env bash
# QuantRabbit live pre-commit guard (shared across worktrees).
set -euo pipefail

LIVE_ROOT="${LIVE_ROOT}"
REPO_ROOT="\$(git rev-parse --show-toplevel 2>/dev/null || true)"

# Only enforce when committing from the live worktree.
if [[ "\$REPO_ROOT" != "\$LIVE_ROOT" ]]; then
    exit 0
fi

disallowed=()
while IFS= read -r path; do
    [[ -z "\$path" ]] && continue
    if [[ "\$path" == docs/*_report.md || "\$path" == docs/*_report ]]; then
        continue
    fi
    disallowed+=("\$path")
done < <(git diff --cached --name-only)

if [[ \${#disallowed[@]} -eq 0 ]]; then
    exit 0
fi

cat >&2 <<MSG
[live-pre-commit] BLOCKED: live root only accepts docs/*_report.md commits.

The trader cycle wrapper runs sync-live-runtime.sh which does
\\\`merge --ff-only main\\\`. A live-only commit is non-fast-forward from
main → sync refuses → wrapper bails → routine cycles silently fail
their pre-flight gate. (2026-05-08: 8 cycles blocked for ~80 min.)

Files this commit attempts:
MSG
for path in "\${disallowed[@]}"; do
    printf '  - %s\\n' "\$path" >&2
done
cat >&2 <<MSG

Commit them on dev main instead:
  cd /Users/tossaki/App/QuantRabbit
  # apply your change in dev main, then:
  git add <files> && git commit -m "..."
  # the next routine cycle's sync propagates to live automatically.

To bypass (rare, e.g. live-only bootstrap or hot-fix when sync is
unreachable): \\\`git commit --no-verify\\\`. The next sync may still
rewrite this commit's effect.
MSG
exit 1
HOOK

chmod +x "$PRE_COMMIT_PATH"
echo "[install-live-runtime-hooks] installed $PRE_COMMIT_PATH"
