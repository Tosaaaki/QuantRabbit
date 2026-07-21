#!/bin/zsh

set -u

readonly DOJO_WORKTREE="/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval"
readonly HEARTBEAT_CLI="scripts/run-dojo-continuous-heartbeat.py"
readonly HEARTBEAT_POLICY="config/dojo_continuous_heartbeat_policy_v1.json"
readonly HEARTBEAT_PROBE="config/dojo_continuous_heartbeat_local_probe_v1.json"
readonly HEARTBEAT_STATE="/Users/tossaki/.codex/automations/dojo-historical-heartbeat/runtime"
readonly HEARTBEAT_INTERVAL_SECONDS=60

while true; do
  event_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  output="$(
    cd "${DOJO_WORKTREE}" &&
      PYTHONPATH=src python3 "${HEARTBEAT_CLI}" tick-local \
        --policy "${HEARTBEAT_POLICY}" \
        --state-dir "${HEARTBEAT_STATE}" \
        --probe "${HEARTBEAT_PROBE}" \
        --event-at-utc "${event_at_utc}" 2>&1
  )"
  exit_code=$?

  if (( exit_code != 0 && exit_code != 75 )); then
    print -u2 -r -- "${event_at_utc} ${output}"
    exit "${exit_code}"
  fi

  sleep "${HEARTBEAT_INTERVAL_SECONDS}"
done
