# QuantRabbit vNext

This is the clean rebuild workspace.

The legacy system was archived before this repository was initialized. Do not copy old runtime behavior forward by default. Pull code back only when it passes the new execution contract.

Previous system location: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`.

See `ARCHIVE_POINTER.md` for the legacy snapshot location and `SYSTEM_REBUILD_CHARTER.md` for the rebuild rules.

See `docs/COMPLETION_DESIGN.md` for the completion architecture, GPT trader requirements, daily 10% campaign gates, and roadmap.

## Runtime Branching

`main` is the stable integration branch. The active trader automation runs from `/Users/tossaki/App/QuantRabbit-live` on `codex/live-trader-runtime`, which mirrors `main` for broker reads, receipts, and gateway execution. Development happens in `/Users/tossaki/App/QuantRabbit` or another worktree.

Use `scripts/sync-live-runtime.sh` to promote committed code: source branch -> `main` -> `codex/live-trader-runtime`, fast-forward only. The live worktree does not create its own commits; it mirrors `main` and clears runtime `docs/*_report.md` drift during live-only sync. The live runner also calls `scripts/sync-live-runtime.sh --live-only --skip-tests` before each cycle, and `scripts/install-live-runtime-hooks.sh` installs a post-commit hook so development commits are automatically tested and promoted.

## Local Credentials

OANDA credentials are stored in `.env.local` at the repository root as `QR_OANDA_ACCOUNT_ID`, `QR_OANDA_TOKEN`, and `QR_OANDA_BASE_URL`. The client loads this file automatically when process environment variables are absent. Do not print the secret values; use `QR_OANDA_ENV_FILE=/path/to/file` to override the lookup in tests or alternate environments.

## Current vNext Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli replay-backtest --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --start-balance 222781 --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision --snapshot data/broker_snapshot.json --decision-response data/codex_trader_decision_response.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --reuse-market-artifacts --use-gpt-trader --gpt-decision-response data/codex_trader_decision_response.json
PYTHONPATH=src python3 -m quant_rabbit.cli replay-execution --prices data/quote_path.json --target-jpy 22278
PYTHONPATH=src python3 -m quant_rabbit.cli learn-post-trade --outcome outcome.json
PYTHONPATH=src python3 -m quant_rabbit.cli certify-dry-run
./scripts/run-autotrade-live.sh --reuse-market-artifacts --use-gpt-trader --gpt-decision-response data/codex_trader_decision_response.json --send
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
```

`risk-dry-run` reads `data/strategy_profile.json` when present, so mined legacy evidence is enforced alongside current risk geometry. Order intents also carry `market_context` (`regime`, `narrative`, `chart_story`, `method`, `invalidation`) so the system can reject method-vs-regime mismatches before live use.

`plan-campaign` builds the multi-desk daily 10% campaign: trend, range, failure, event-risk, and position-management desks all propose or veto lanes, then the Portfolio Director reports what can become a receipt and what is still missing.

`replay-backtest` replays imported legacy days against the 10% target, current loss cap, captured profit, pretrade evidence, and missed-seat evidence. It reports whether misses were caused by insufficient coverage, missed edge capture, or risk-repair failures.

`daily-target-state` records the day-start equity, 10% target, realized/unrealized PnL, remaining target, open risk, and remaining risk budget. USD-quoted open risk uses the current `USD_JPY` snapshot quote; missing conversion data blocks fresh risk budget instead of falling back to a static rate. Position-management reports use the same snapshot conversion and mark risk/reward as unknown when the conversion quote is absent. `plan-campaign` initializes this ledger, and `autotrade-cycle` refreshes it when the state file exists.

`generate-intents` turns campaign lanes into priced dry-run order intents when a read-only broker snapshot is available. Each receipt carries broker-truth risk metrics (`risk_jpy`, `reward_jpy`, reward/risk, spread) from `RiskEngine`. Without a snapshot it reports `NEEDS_BROKER_SNAPSHOT`; this is a hard stop, not a prompt problem.

`gpt-trader-decision` builds a broker-truth input packet and verifies a Codex-created GPT decision receipt from `--decision-response`. Automation GPT means the Codex model itself, not an API-key model call from QuantRabbit. The verifier rejects any output that invents evidence, selects an unknown lane, trades while exposure is open, or uses a non-`LIVE_READY` lane. It writes `data/gpt_trader_decision.json` and `docs/gpt_trader_decision_report.md`; this command does not send broker orders.

`promote-receipts` feeds successful dry-run receipts back into `data/strategy_profile.json`. It can reopen `RISK_REPAIR_CANDIDATE` only when the current receipt passes risk geometry, and can reopen `MINE_MISSED_EDGE` only when the receipt is a LIMIT or STOP-ENTRY trigger. It never auto-promotes `BLOCK_UNTIL_NEW_EVIDENCE`.

`optimize-coverage` measures whether current `LIVE_READY` receipts can cover the remaining daily target, separates potential coverage from `DRY_RUN_PASSED` lanes that still need promotion, and emits explicit gap tasks.

`stage-live-order` turns one live-ready intent into an OANDA order request after fetching fresh broker truth and rerunning live validation. It stages by default. A real send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, and an explicit `--lane-id`.

`autotrade-cycle` is the automation entrypoint. It fetches fresh broker truth first; if any position or pending order exists, it manages existing exposure before considering fresh risk and sends no new entry. If a pending order came from a lane the current TraderBrain now vetoes, the cycle can cancel that pending order before waiting for the next cycle. If a position is open, PositionManager writes the TP/SL, remaining risk/reward, same-vs-opposite thesis score, and protection/exit-review action. The protection gateway can then close a contradicted trader-owned trade, create missing TP/SL, or tighten SL to break-even/profit protection; it refuses to widen SL or move existing TP. If flat and the daily target state is already `TARGET_REACHED_PROTECT`, the cycle records a no-send protection-first receipt and adds no fresh risk. Otherwise, if flat, it regenerates intents, asks TraderBrain to compare the live-ready lanes against mined history, market story, campaign role, narrative risk, and current broker state, then sends only the selected lane when `QR_LIVE_ENABLED=1` and `--send` are present. With `--use-gpt-trader --gpt-decision-response ...`, the deterministic TraderBrain becomes the prefilter and the gateway receives a lane only when the Codex-created GPT receipt verifies as an ACCEPTED `TRADE` for one of those prefiltered lanes; rejected, hallucinated, or non-prefiltered GPT decisions become no-send cycle receipts.

`replay-execution` replays live-ready order receipts over a supplied quote path, checking fill, TP/SL, and target coverage without broker side effects.

`learn-post-trade` converts fills/closes/protection receipts into advisory learning candidates. It does not mutate the strategy profile; losses beyond the risk cap become blockers.

`certify-dry-run` verifies coverage, execution replay, learning receipts, order-intent contracts, GPT status, and no-send dry-run artifacts before any live expansion.

Live execution is guarded behind this gateway: strategy evidence, market story, campaign role, fresh broker truth, risk geometry, and explicit live enablement must all pass before any OANDA write occurs.
