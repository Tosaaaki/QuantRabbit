# Precheck And Refresh

## Precheck

- Run before any report-writing command.
- `git status --short` may contain only tracked `docs/*_report.md` runtime drift from the previous cycle.
- Source, config, data, decision, or prompt diffs block the scheduled trader cycle until resolved.
- Confirm exactly one trader scheduled task is enabled.
- Do not run report-writing refresh commands from a dirty development tree.
- Do not stop because refreshed broker truth disagrees with stale journal, stale local state, or an older decision receipt. Broker truth wins; re-route and rewrite the receipt when needed.
- A locally remembered pending order that is absent from the refreshed OANDA snapshot is stale local memory, not a send blocker.

## Refresh Evidence — ONE command

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10
```

`cycle-refresh` runs the full refresh step list in one process — broker
snapshot, daily target state, ledger sync, strategy mining, pair charts,
all market-context layers, market-story mining, `news-health --strict`,
daily-review, tp-rebalance, projection verification, intent generation,
coverage/attack/learning/capture/operator-precedent/verification audits,
predictive limits, position sidecars, memory health — in the same order and
with the same
arguments the per-step skeleton used (`cli._cycle_refresh_steps` is the
canonical list), then prints one compact digest that already includes the
re-routed prompt branch (`route`).

**Token discipline (2026-06-09 credit-exhaustion incident).** Running these
steps as separate shell turns burned ~3M tokens per former 20-minute cycle and
silently stopped live trading for ~36 hours. Do NOT run the refresh steps
individually; do NOT cat multi-megabyte artifacts. Read the digest, then
drill into `data/order_intents.json`, `data/pair_charts.json`,
`data/market_context_matrix.json` etc. with targeted `jq` / `python -c`
queries only where the digest flags something.

**Wait long, poll never (2026-06-11).** `cycle-refresh`, the live gateway
wrapper, and `cycle-sidecars` run for minutes. Invoke each with ONE long
shell wait (yield/timeout ≥ 300000 ms, generous max_output_tokens) so the
call returns once when done. Re-polling a yielded command every ~10s
re-sends the entire conversation context per poll — telemetry on
2026-06-11 measured ~25 such empty turns per cycle, which kept the cycle
at ~3.9M tokens despite the consolidated commands.

Digest semantics:

- `steps_failed` lists failed steps with stderr tails. A failed REQUIRED
  step aborts the rest and exits 2. Treat `order_intents.json` and every
  downstream decision/gateway artifact as prior-cycle state until a complete
  refresh publishes new intents; never combine the failed refresh's newer
  broker/market artifacts with those old intents. Name the producer error,
  repair the named step, then rerun `cycle-refresh` once. Optional-step failures (for example `news-health
  --strict` during a stale-news window) do not stop evidence generation;
  treat them as named blockers in the decision receipt exactly as before.
- `tp-rebalance` already ran inside the refresh, even when the later
  receipt becomes WAIT — existing broker TPs are position protection and
  must not wait for a fresh entry to be managed.
- `target`, `intents`, `attack_advice`, `capture_economics`,
  `operator_precedent`, `manual_market_context`,
  `memory_health`, `news_health`, `thesis_evolution`,
  `forecast_persistence`, `position_thesis` summarize the artifacts the
  decision receipt must cite.

**News stays consumer-only.** The digest's `news_health` reflects the
artifacts written by the dedicated `qr-news-digest` Codex Desktop routine
(hourly, same live runtime worktree). Do not call `news-snapshot` from the
trader cycle — that would replace the curated digest with raw RSS output.
During the open FX week a stale digest fails loud through `news-health
--strict`; during the contract weekend pause it accepts the paused
scheduler when the weekend snapshot says `mode=paused`.

Do not stop after evidence refresh. The digest's `route` field is the
re-route result — read the returned branch, write one current decision
receipt, then continue to `gpt-trader-decision` and exactly one gateway
cycle. A cycle that ends right after `cycle-refresh` leaves fresh evidence
unused and is incomplete. Only run `trader-prompt-route` again if you
changed an artifact after the digest was printed.

## Refresh Strategy Evidence (repair only)

`cycle-refresh` already runs `import-legacy` + `mine-strategy` +
`mine-market-stories` every cycle. Run the explicit repair block only when a
branch routes to deep evidence repair (for example a corrupt history DB or a
campaign plan that must be rebuilt outside the normal generate-intents
refresh):

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli plan-campaign --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
```

Never paste a JPY literal into `--start-balance` (§3.5); derive it from the
current daily target state as above.

## Stop Conditions

- Any `MISSING_*` artifact that is required for the active branch and cannot be refreshed.
- Dirty source/config/data/decision files before report writes.
- More than one trader scheduler enabled.
- Missing OANDA read credentials for broker-truth refresh.
- Active OANDA broker-truth exposure/risk gates that the current gateway cannot reconcile.
