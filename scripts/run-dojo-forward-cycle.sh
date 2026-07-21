#!/usr/bin/env bash
set -euo pipefail

# Permanent paper-only launcher for the forward W_FADE + W_SPIKE_FADE floor.
# Run it inside a detached supervisor (currently GNU screen).  This script
# has read-only OANDA pricing access and the virtual broker has no order API.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
session_dir="${repo_root}/research/data/dojo_forward_20260720"

export QR_OANDA_ENV_FILE="/Users/tossaki/App/QuantRabbit-live/.env.local"
export DOJO_BOT_COMBO='[{"strategy_tag":"W_FADE","signal":"range_fade_limit","pairs":["USD_JPY"],"tp_pips":6,"sl_pips":null,"ceiling_min":480,"max_concurrent":3,"per_pos_lev":4.3,"atr_floor_pips":1.0,"fade_atr":1.2,"eff_max":0.2},{"strategy_tag":"W_SPIKE","signal":"spike_fade","pairs":["USD_JPY"],"tp_atr":3.0,"sl_pips":null,"ceiling_min":480,"max_concurrent":2,"per_pos_lev":4.3,"atr_floor_pips":1.0,"fade_atr":1.2,"eff_max":0.2}]'
export PYTHONPATH="${repo_root}/src"

exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  "${repo_root}/scripts/run-virtual-market-session.py" \
  --feed live \
  --session-dir "${session_dir}" \
  --pairs USD_JPY \
  --minutes 720 \
  --balance 200000 \
  --bot-module "${repo_root}/bots/combo_bot.py:Bot" \
  --seed-m1-root "${repo_root}/research/data/m1_seed" \
  --seed-hours 200 \
  >> "${session_dir}/forward-supervisor.log" 2>&1
