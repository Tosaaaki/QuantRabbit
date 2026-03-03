# Local Lane Split Runbook

## 1. Goal
- Keep VM production (`main` repo) and local real-trading lane in separate git repos.
- Run both lanes in parallel and compare outcome with one comparator (`scripts/compare_live_lanes.py`).

## 2. Bootstrap Local Repo

```bash
cd /Users/tossaki/Documents/App/QuantRabbit
bash scripts/bootstrap_local_lane_repo.sh \
  --target /Users/tossaki/Documents/App/QuantRabbitLocalLane \
  --mode init \
  --init-git
```

Notes:
- Source bot defaults to `/tmp/codex_long_autotrade.py`.
- Generated local repo files are managed. If a managed file already differs, a
  `*.bootstrap.new` file is written unless `--force` is specified.

## 3. Start Local Lane (Real Trading)

```bash
cd /Users/tossaki/Documents/App/QuantRabbitLocalLane
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
bash scripts/run_local_lane.sh
```

Output log:
- `/Users/tossaki/Documents/App/QuantRabbitLocalLane/state/codex_long_autotrade.log`

## 4. Compare VM vs Local Lane

1. Create bridge env:

```bash
cd /Users/tossaki/Documents/App/QuantRabbit
cp ops/env/local-lane-bridge.env.example ops/env/local-lane-bridge.env
```

2. Run comparator watcher:

```bash
cd /Users/tossaki/Documents/App/QuantRabbit
bash scripts/watch_lane_winner_external.sh
```

Outputs:
- `logs/lane_winner_latest.json`
- `logs/lane_winner_history.jsonl`

## 5. JSONL Contract (Local -> Main)

Required events in local log:
- `trade_opened`: should include `trade_id` and `client_id`.
- `trade_closed`: should include `trade_id` or `client_id`, and `pl` + `pnl_pips` (or `pl_pips`).

Prefix requirement:
- Local lane should use `client_id` prefix `codexlhf_` (default in local bot), or adjust `LOCAL_PREFIX`.

## 6. Safety / Ops Checks
- Keep manual positions untouched (bot identifiers only).
- Verify OANDA credentials are set in local `.env`.
- Before deploying VM changes, keep this split unchanged:
  - VM repo for production services and deploy scripts.
  - Local repo for local LLM lane runtime.
