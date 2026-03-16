#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p .githooks
chmod +x \
  .githooks/pre-commit \
  scripts/improvement_preflight.sh \
  scripts/preflight_guard.py \
  scripts/trade_findings_lint.py \
  scripts/trade_findings_index.py
git config core.hooksPath .githooks

echo "installed git hooks"
echo "core.hooksPath=$(git config --get core.hooksPath)"
