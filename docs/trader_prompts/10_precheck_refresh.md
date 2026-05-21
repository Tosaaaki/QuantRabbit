# Precheck And Refresh

## Precheck

- Run before any report-writing command.
- `git status --short` may contain only tracked `docs/*_report.md` runtime drift from the previous cycle.
- Source, config, data, decision, or prompt diffs block the scheduled trader cycle until resolved.
- Confirm exactly one trader scheduled task is enabled.
- Do not run report-writing refresh commands from a dirty development tree.
- Do not stop because refreshed broker truth disagrees with stale journal, stale local state, or an older decision receipt. Broker truth wins; re-route and rewrite the receipt when needed.
- A locally remembered pending order that is absent from the refreshed OANDA snapshot is stale local memory, not a send blocker.

## Refresh Evidence

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew
```

**News is produced out-of-band** by the dedicated `qr-news-digest`
Claude Desktop routine (hourly at :23 JST). That routine runs in the
dev worktree (`/Users/tossaki/App/QuantRabbit/`) and writes
WebSearch-curated trader-perspective content to
`logs/news_digest.md` + `logs/news_flow_log.md`. The live worktree's
`logs/news_digest.md` is a symlink pointing to the dev file so this
cycle automatically sees the curated digest. **Do not call
`news-snapshot` from the trader cycle** — that would write raw RSS
output through the symlink and clobber the curated content. If the
digest goes stale (the routine fails or is paused), `market_story.py`
surfaces missing-evidence rationale on lanes; it does not crash. The
2026-05-13 incident exposed a different bug: live's `logs/news_digest.md`
was a standalone file, 7 days stale (last refresh 2026-05-06), while
the dev routine was writing fresh hourly to its own file the trader
never read. Symlink resolves the bridge.

Refresh the derived live market-story profile from that curated digest before
intent pricing. `logs/news_digest.md` being fresh is not sufficient by itself:
`trader_brain` and `gpt_trader` read `data/market_story_profile.json`.
Write the side report under `data/` so the precheck path does not create new
tracked `docs/*_report.md` diffs.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories \
  --news-dir logs \
  --profile data/market_story_profile.json \
  --report data/market_story_report.md
```

**daily-review** runs every cycle (idempotent, no network) to refresh
`data/trader_overrides.json` from execution_ledger.db. trader_brain's
Module C reads that file for direction-bias overrides + blocked-lane
hints derived from the last 24h of realized P&L. Running every cycle
keeps the override rolling without a separate scheduled task and lets
new closed trades immediately influence the next cycle's scoring.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli daily-review
```

## Reprice Intents

Context fetches can outlive the quote freshness window. Refresh broker truth again immediately before intent pricing.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli tp-rebalance
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice
PYTHONPATH=src python3 -m quant_rabbit.cli generate-predictive-limits
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route
```

Do not stop after evidence refresh. Re-run the router with the refreshed
snapshot/intents, read the returned branch, write one current decision receipt,
then continue to `gpt-trader-decision` and exactly one gateway cycle. A refresh
that ends at `generate-predictive-limits` is an incomplete cycle.

`tp-rebalance` is part of the refresh/reprice path, not only a TRADE aftercare
step. Run it even if the later receipt becomes WAIT, because existing broker
TPs are position protection and stale profitable TPs must not wait for a fresh
entry to be managed. Refresh broker truth after the TP pass so the decision
packet and order intents cite the actual dependent-order price.

## Refresh Strategy Evidence

Run only when strategy artifacts are missing, stale, or a branch explicitly routes to evidence repair.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
```

## Stop Conditions

- Any `MISSING_*` artifact that is required for the active branch and cannot be refreshed.
- Dirty source/config/data/decision files before report writes.
- More than one trader scheduler enabled.
- Missing OANDA read credentials for broker-truth refresh.
- Active OANDA broker-truth exposure/risk gates that the current gateway cannot reconcile.
