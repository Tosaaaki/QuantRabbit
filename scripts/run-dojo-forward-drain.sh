#!/usr/bin/env bash
set -euo pipefail

# One-way boundary for the 2026-07-22 legacy mixed diagnostic session.
# This launcher has no bot, no agent inbox, and no code path that can create
# an entry.  It only lets the virtual broker resolve already-open exposure by
# TP/SL/margin or the original eight-hour position ceiling.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
session_dir="${repo_root}/research/data/dojo_forward_20260720"

export QR_OANDA_ENV_FILE="/Users/tossaki/App/QuantRabbit-live/.env.local"
export PYTHONPATH="${repo_root}/src"

exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  "${repo_root}/scripts/run-virtual-market-session.py" \
  --feed live \
  --session-dir "${session_dir}" \
  --pairs USD_JPY \
  --minutes 720 \
  --balance 200000 \
  --drain-only \
  --drain-ceiling-min 480 \
  --allow-legacy-untagged \
  --slippage-pips 0 \
  --financing-pips-day 0 \
  --leverage 25 \
  --runtime-dependency "${repo_root}/scripts/run-dojo-forward-drain.sh" \
  >> "${session_dir}/forward-drain.log" 2>&1
