# Rolling 30d 4x Recovery Plan

Generated: 2026-07-04T11:53:12Z  
Mode: inspection, refactor, and evidence generation only. No broker-side action.

## Evidence Sources

- Live root: `/Users/tossaki/App/QuantRabbit-live`
- Fresh read-only broker snapshot: `tmp/read_only_broker_snapshot_20260704.json`, fetched `2026-07-04T11:44:50.791825+00:00`
- Target state: `data/daily_target_state.json`
- Capital flows: `data/capital_flows.json`
- Broker truth: `data/broker_snapshot.json` plus the fresh read-only snapshot above
- Profitability: `data/capture_economics.json`, `data/profitability_acceptance.json`, `data/execution_timing_audit.json`, `data/profit_capture_bot.json`
- Routing: `data/order_intents.json`, `data/trader_support_bot.json`, `data/self_improvement_audit.json`

## 30d 4x Funding-Adjusted Board

Performance basis is `funding_adjusted_equity`. Raw NAV is risk, margin, and sizing context only.

| Field | Value |
| --- | ---: |
| current_equity_raw | 270740.4982 JPY |
| 30d capital flows excluded from P/L | 100000.0000 JPY |
| funding_adjusted_equity | 170740.4982 JPY |
| rolling_30d_start_equity | 175305.0552 JPY |
| 4x target funding-adjusted equity | 701220.2208 JPY |
| rolling_30d_multiplier_funding_adjusted | 0.973962x |
| rolling_30d_multiplier_raw | 1.544396x |
| remaining_to_4x_funding_adjusted | 530479.7226 JPY |
| rolling_30d_elapsed_calendar_days | 4.4737 |
| rolling_30d_remaining_calendar_days | 25.5263 |
| rolling_30d_remaining_active_days | 18.7193 |
| required_calendar_daily_return_funding_adjusted | 5.690202% |
| required_active_day_return_funding_adjusted | 7.838697% |
| required weekly progress | 1.473137x, or 47.313689% |
| current pace_state | BEHIND |

The 100000 JPY operator deposit at `2026-07-02T08:33:11Z` is capital flow, not trading P/L. It is included in raw NAV and excluded from funding-adjusted performance.

## Manual EUR_USD Protection

EUR_USD trade `472987` remains `OPERATOR_MANUAL / KEEP`.

- It is not system P/L.
- It is not system occupancy.
- It must not be auto-closed.
- It must not receive auto SL/TP modification.
- It must not receive same-theme add exposure.
- Existing pending TP/order state is broker/operator state and must not be modified by this task.

## Trading Capability Target

Fresh entries can resume only when all of the following are true:

- `profitability_acceptance` is refreshed from current `capture_economics` and current `order_intents`, and remains blocking unless the lane is an exact narrow repair exception with positive proof.
- `self_improvement_audit` has no P0, especially no memory-health P0 and no target-open/no-live-ready P0.
- `order_intents` is aligned to a fresh broker snapshot and contains at least one current `LIVE_READY` lane.
- The lane has a fresh GPT-5.5 `TRADE` or `ADD` receipt, RiskEngine pass, and LiveOrderGateway pass.
- Manual EUR_USD 472987 remains protected and excluded.
- No target-path execution flag is enabled by this plan.

An A/S lane must prove:

- Pair/side/method-specific positive expectancy after spread, not broad market or stale aggregate support.
- Attached TP geometry with exact method scope or spread-included bid/ask replay proof.
- Forecast support that is executable, not watch-only or REQUEST_EVIDENCE.
- Clean geometry: RR acceptable, target not too thin for spread, no broad premium-long or discount-short chase, no market-close leak pattern.
- Current broker/risk fit: margin, unit floor, loss budget, and no manual exposure conflict.
- Durable close/profit-capture discipline: no loss-side market-close leakage used as the edge source.

One lane becomes `LIVE_READY` only when global P0s are clear or explicitly not applicable to that exact lane, all local blockers are gone, GPT/Risk/Gateway pass on the same fresh snapshot, and the lane is A/S quality. B/C churn is not a valid path to rolling 30d 4x.

+10% extension day is eligible only after the +5% floor is actually protected by A/S-grade evidence, the last A/S winner is green/protected/alive, the thesis still pays across current structure, spreads are stable, the next 30 minutes are not whipsaw, and replay/GPT/Risk/Gateway all agree. Current state is not eligible: there are zero `LIVE_READY` lanes and no protected system winner.

## Support Freshness Diagnosis

`PROFITABILITY_ACCEPTANCE_STALE_AGAINST_INPUTS`

| Artifact | generated_at_utc |
| --- | --- |
| capture_economics.json | 2026-07-03T20:39:53.779849+00:00 |
| order_intents.json | 2026-07-03T20:16:34.043087+00:00 |
| profitability_acceptance.json | 2026-07-03T20:16:44.476825+00:00 |
| trader_support_bot.json | 2026-07-03T20:16:56.340170+00:00 |

The current on-disk `capture_economics.json` is newer than both `profitability_acceptance.json` and `order_intents.json`. Recommendation: `REFRESH_PROFITABILITY_ACCEPTANCE_FROM_CURRENT_INPUTS`. The stale acceptance artifact must not be used to allow routing. Its blockers are still valid as a conservative stop signal, but not as permission.

## Blocker Decomposition

| Blocker | Current evidence | Source artifact | Source module/file | Current vs stale | Clearing condition | Required replay/evidence/code change | Ordinary fresh entries blocked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NEGATIVE_EXPECTANCY_ACTIVE | Overall capture is `NEGATIVE_EXPECTANCY`: 234 trades, win rate 58.6%, avg win 418.0 JPY, avg loss 1015.3 JPY, payoff 0.412, expectancy -176.2 JPY/trade, net -41225.9 JPY. | `data/capture_economics.json`, `data/profitability_acceptance.json` | `src/quant_rabbit/capture_economics.py`, `src/quant_rabbit/profitability_acceptance.py`, `src/quant_rabbit/strategy/intent_generator.py` | Current hard P0, though acceptance needs refresh against newer capture. | Realized expectancy non-negative, or only exact TP-proven HARVEST repair shapes pass while global routing stays blocked. | Refresh capture and profitability acceptance; build exact method TP proof with positive spread-included edge. | Yes. |
| MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE | EUR_USD LONG BREAKOUT_FAILURE is TP-proven but net-damaged by system gateway market closes: TP expectancy 591.5 JPY, market close net -15091.7 JPY, market close expectancy -2156.0 JPY. Loss trade ids: 470730, 471089, 470353, 471174, 470356. Operator manual rows are excluded; these rows count against system edge. | `data/profitability_acceptance.json`, `data/capture_economics.json` | `src/quant_rabbit/profitability_acceptance.py`, `src/quant_rabbit/capture_economics.py`, `src/quant_rabbit/strategy/trader_brain.py` | Current hard P0. | No TP-proven segment remains net-damaged by `MARKET_ORDER_TRADE_CLOSE`; preserve broker TP/profit capture instead of scaling loss-close paths. | Refresh capture; rerun execution timing; keep market-close leak families out of LIVE_READY. | Yes. |
| RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK | Not present in current live `profitability_acceptance.json`; stale dev report still mentions it. Execution timing labels 19 contained-risk and 13 may-have-been-premature loss market closes, but latest acceptance did not raise this 7-day P0. | Current: `data/profitability_acceptance.json`, `data/execution_timing_audit.json`; stale: `docs/profitability_acceptance_report.md` | `src/quant_rabbit/profitability_acceptance.py` | Stale artifact drift / currently cleared, while broader market-close leak remains active. | If re-raised: recent leak loss closes must be zero, or every loss-side market close must have contained-risk timing plus durable gateway/GPT close proof. | Refresh acceptance from current inputs; do not synthesize contained-risk proof. | Yes, because other P0s remain. |
| LOSS_CLOSE_GATE_EVIDENCE_MISSING | Not present in current live `profitability_acceptance.json`. Code and tests now read `verification_observations` `check_name=close_gate_evidence`; historical verification rows include BLOCK/PASS evidence, not a current missing-evidence P0. | `data/profitability_acceptance.json`, `data/execution_ledger.sqlite3`, tests under `tests/test_cli.py`, `tests/test_execution_ledger.py`, `tests/test_autotrade_cycle.py` | `src/quant_rabbit/gpt_trader.py`, `src/quant_rabbit/execution_ledger.py`, `src/quant_rabbit/profitability_acceptance.py`, `src/quant_rabbit/automation.py` | Stale artifact drift / currently cleared. | Every recent GPT loss-side close has durable PASS `close_gate_evidence`, or failed/missing historical rows age out without new leaks. | No code change required now. Keep schema: trade id, pair/side, UPL, loss-side flag, Gate A, Gate B, P0 citation, timing citation, hard timing requirement, same-direction support conflict. | Yes, because other P0s remain. |
| MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE | 744h replay: actual loss-close P/L -39275.3429 JPY; repair replay P/L -20504.5826 JPY; delta +18770.7603 JPY, still negative. Residuals include GBP_USD LONG BREAKOUT_FAILURE trade 472071, AUD_USD LONG RANGE_ROTATION 472952, EUR_USD LONG RANGE_ROTATION 471817, EUR_USD SHORT RANGE_ROTATION 471711, NZD_CAD SHORT RANGE_ROTATION 472380/472312, AUD_USD SHORT RANGE_ROTATION 472834. | `data/execution_timing_audit.json`, `data/profitability_acceptance.json` | `src/quant_rabbit/profitability_acceptance.py`, `src/quant_rabbit/strategy/intent_generator.py`, `src/quant_rabbit/trader_support_bot.py` | Current hard P0 for matching residual groups. | Rerun 744h execution-timing and acceptance until month-scale replay is non-negative or matching residual groups disappear. | Improve close-gate evidence, entry selection, or exit geometry; rerun `execution-timing-audit --lookback-hours 744 --post-close-hours 6`. | Yes for matching groups, and practically yes globally because no LIVE_READY lanes exist. |
| HISTORICAL_PROFIT_CAPTURE_MISSED | 14 recent loss closes missed executable profit capture; production-gate replay triggers 13, delta 18770.76 JPY. Post-repair live evidence currently shows 0 misses and 0 repair triggers, so this is mostly historical repair evidence, not permission. | `data/profit_capture_bot.json`, `data/execution_timing_audit.json` | `src/quant_rabbit/profit_capture_bot.py`, `src/quant_rabbit/execution_timing_audit.py`, `src/quant_rabbit/position_manager.py` | Current P0 in support/profit-capture context; historical root cause remains. | Post-repair live evidence must stay clean and replay residuals must become non-negative or age out. | Rerun profit capture bot and execution timing; inspect residual families before adding turnover. | Yes. |
| BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN | Price truth is loaded, but all-currency high-turn validation is blocked by 53 under-sampled pair-directions, 59 pending future truth samples, and 1343 no-market-session unscorable samples across 56 pair-directions. | `data/profitability_acceptance.json`, `logs/reports/forecast_improvement/oanda_history_replay_validate_20260703T145218Z.json` | `scripts/oanda_history_replay_validate.py`, `scripts/package_bidask_replay_precision_rules.py`, `src/quant_rabbit/profitability_acceptance.py` | Current P1 evidence gap. | All-currency sample coverage no longer `UNDER_SAMPLED`; pair-direction replay meets minimum 30 directional samples, 3 active days, positive day rate >= 0.6667, max daily share <= 0.7. | Read-only regenerate: `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 ...`; then package rules and run focused tests. | Yes in practice because no LIVE_READY lanes and P0s remain. |
| NO_LIVE_READY_TARGET_COVERAGE | Daily target is open; `order_intents` has 84 lanes, zero `LIVE_READY`. Support reports `NO_LIVE_READY_LANES`. | `data/order_intents.json`, `data/trader_support_bot.json`, `data/self_improvement_audit.json` | `src/quant_rabbit/strategy/intent_generator.py`, `src/quant_rabbit/trader_support_bot.py`, `src/quant_rabbit/trader_prompts.py` | Current operational blocker. | At least one current A/S lane becomes `LIVE_READY` on a fresh broker snapshot. | Refresh market context/order intents after evidence repairs. | Yes. |
| REPAIR_FRONTIER_BLOCKED | 12 repair-mode candidates exist; support says none clear live gates. Current repair frontier is EUR_USD-heavy and includes manual same-theme/add conflicts or forecast/replay blocks. | `data/trader_support_bot.json`, `data/order_intents.json` | `src/quant_rabbit/trader_support_bot.py`, `src/quant_rabbit/strategy/intent_generator.py`, `src/quant_rabbit/trader_repair_orchestrator.py` | Current P1 evidence/route blocker. | At least one repair candidate clears global P0s, local blockers, manual exposure conflict, and becomes A/S `LIVE_READY`. | Do not add into manual EUR_USD; repair candidate evidence must be exact and non-market-close-leak. | Yes. |

## Close-Gate Evidence Repair Decision

No code change is required in this pass. The current acceptance artifact does not raise `LOSS_CLOSE_GATE_EVIDENCE_MISSING`, and the source path already requires durable evidence:

- `profitability_acceptance` queries `verification_observations` for `check_name='close_gate_evidence'`.
- It distinguishes missing, not-passing, and PASS evidence.
- It does not turn missing evidence into permission.

If the blocker reappears, the durable artifact must include the schema in `docs/AGENT_CONTRACT.md`: trade id, pair/side, unrealized P/L, loss-side flag, Gate A verdict/reason, Gate B standing/explicit authorization state, profitability-P0 citation state, timing-audit citation state, hard-timing requirement, and same-direction support conflict. The missing event must be identified from `recent_close_gate_missing_loss_examples` in the refreshed acceptance file before any code change.

## Market-Close Leak Diagnostic

The current hard leak is system-gateway attributed, not operator manual:

- Lane family: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Trades: 470730, 471089, 470353, 471174, 470356
- Scope: `SYSTEM_GATEWAY_ATTRIBUTED_ONLY`
- Operator manual excluded: true
- Counts against system edge: true

Manual EUR_USD 472987 is not part of the system leak sample and must remain excluded. The market-close blocker stays active until the leak segment is repaired or no longer net-damages the TP-proven edge.

## TP Progress Replay Path

Replay window: 744h, post-close 6h, contract `TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1`.

- Actual loss-close P/L: -39275.3429 JPY
- Raw profit-capture counterfactual P/L: -20547.4015 JPY
- Repair replay counterfactual P/L: -20504.5826 JPY
- Repair delta: +18770.7603 JPY
- Verdict: improvement is real but insufficient; replay remains negative.

The filter or exit discipline must remove residual groups where replay is still `BELOW_TP_PROGRESS_GATE` or `NO_PROFIT_CANDIDATE`, especially AUD_USD RANGE_ROTATION, GBP_USD BREAKOUT_FAILURE, EUR_USD RANGE_ROTATION, NZD_CAD RANGE_ROTATION, and EUR_CHF TREND_CONTINUATION. No exact shape is currently proven TP-positive enough to override the global gate without overfitting.

## Bid/Ask Replay Coverage Expansion

Current coverage has 28 pairs and 56 pair-directions. Pair-level evaluated samples are healthy, but all-currency high-turn readiness is blocked by under-sampled or unscorable pair-directions. Priority pairs for this recovery plan:

- EUR_JPY: DOWN 1147, UP 1566 evaluated samples; both still marked under-sampled due no-market-session unscorable samples.
- USD_JPY: DOWN 525, UP 843 evaluated samples; USD_JPY DOWN contrarian-to-UP rule is locally live-grade, but all-currency coverage is still thin.
- AUD_USD: DOWN 972, UP 395 evaluated samples; both current replay statuses are negative expectancy.

Read-only regeneration command published by acceptance:

```bash
python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667
```

Expected output artifact: `logs/reports/forecast_improvement/oanda_history_replay_validate_<timestamp>.json`, followed by packaged `src/quant_rabbit/bidask_replay_precision_rules.json` only if the replay improves evidence without weakening blockers.

## Staged Path Back

P0 before any fresh entry:

- Refresh `capture_economics`, `order_intents`, `profitability_acceptance`, and `trader_support_bot` in sequence from one fresh broker snapshot.
- Clear or keep blocking `NEGATIVE_EXPECTANCY_ACTIVE`.
- Keep `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` active until EUR_USD LONG BREAKOUT_FAILURE leakage is repaired.
- Keep month-scale residual blockers active until replay is non-negative or matching residual groups disappear.
- Resolve self-improvement P0s: memory health and target-open/no-live-ready diagnostics.

P1 to restore A/S candidates:

- Generate exact local TP proof for EUR_JPY SHORT and USD_JPY candidates.
- Repair USD_JPY strategy-profile status from `BLOCK_UNTIL_NEW_EVIDENCE` using risk-resized dry-run receipts and same pair/side/method evidence.
- Expand S5 bid/ask replay coverage and package only evidence that passes daily-stable criteria.
- Rebuild order intents and require at least one current A/S `LIVE_READY` lane.

P2 cleanup:

- Remove stale report drift after refreshed artifacts are produced.
- Keep stale `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` and `LOSS_CLOSE_GATE_EVIDENCE_MISSING` references classified as drift unless fresh acceptance re-raises them.
- Keep docs/data boards synced after every evidence refresh.

Exact files/modules:

- `src/quant_rabbit/profitability_acceptance.py`
- `src/quant_rabbit/capture_economics.py`
- `src/quant_rabbit/profit_capture_bot.py`
- `src/quant_rabbit/execution_timing_audit.py`
- `src/quant_rabbit/strategy/intent_generator.py`
- `src/quant_rabbit/trader_support_bot.py`
- `src/quant_rabbit/gpt_trader.py`
- `src/quant_rabbit/execution_ledger.py`
- `scripts/oanda_history_replay_validate.py`
- `scripts/package_bidask_replay_precision_rules.py`
- `tools/check_task_sync.py`

Shortest path to one legitimate A/S `LIVE_READY` lane:

1. Refresh stale profitability support from current inputs.
2. Target USD_JPY LONG passive LIMIT / bidask precision lane first because it has spread-included S5 live-grade evidence: 91 samples, PF 3.2194, hit rate 0.7473, positive day rate 0.75, TP10/SL7.
3. Clear its hard local blockers: exact TP scope is missing and USD_JPY LONG strategy profile is `BLOCK_UNTIL_NEW_EVIDENCE`.
4. Reject it if geometry remains range-chase/exhaustion, forecast support stays non-executable, or strategy profile cannot be repaired with risk-resized evidence.
5. Rebuild order intents and accept only if it becomes A/S `LIVE_READY` with GPT/Risk/Gateway pass.

Fallback path: EUR_JPY SHORT local TP proof is a priority family, but current S5 replay is explicitly negative (`EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY`, 1147 samples, PF 0.0, positive day rate 0.0), so it needs new exact local TP-positive evidence before it can be considered A/S.

Shortest path to +10% extension eligibility:

There is no current +10% extension path. First produce a +5% protected A/S winner from a legitimate `LIVE_READY` lane, then prove extension criteria on fresh evidence. With zero `LIVE_READY` lanes and only manual EUR_USD exposure, normal routing remains blocked.
