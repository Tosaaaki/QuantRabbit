#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUERY="${1:-}"
CANDIDATES="${2:-}"
LIMIT="${3:-5}"

if [[ -z "${QUERY}" || -z "${CANDIDATES}" ]]; then
  echo "usage: scripts/improvement_preflight.sh \"<query>\" \"<strategy::surface::primary_loss_driver::idea||...>\" [limit]" >&2
  exit 2
fi

"${ROOT_DIR}/scripts/change_preflight.sh" "${QUERY}" "${LIMIT}"
echo
echo "== Improvement Proposal Gate =="
exec python3 "${ROOT_DIR}/scripts/improvement_gate.py" \
  --query "${QUERY}" \
  --candidates "${CANDIDATES}" \
  --limit "${LIMIT}"
