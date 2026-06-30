# Strategy Memory

## 2026-06-30 Guardian Wake Dispatch

- Guardian wake is now event-driven through `tools/guardian_wake_dispatcher.py`: the router still detects deterministic events, while the dispatcher starts read-only `codex exec` for GPT-5.5 only when the event is new/severity-increased, broker truth is fresh, no live lock is active, and the dedupe key has not already been reviewed.
- The wake path is review/receipt only by default. `QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0` prevents immediate execution; even an enabled handoff must go through the existing `RiskEngine` / `LiveOrderGateway` path and must not create direct OANDA writes.
- If the full trader is already active, the dispatcher marks `data/guardian_escalation.json` as `queued_for_active_trader`; the next trader cycle must resolve guardian action receipts before normal new entries.
- `data/guardian_trigger_contract.json` is the trader-owned market-trigger surface. The router may fire only explicit fired/triggered trigger entries or machine-readable predicates; missing contract data becomes `CONTRACT_STALE` when exposure exists, not invented harvest/add/invalidation logic.
- Empty or non-JSON Codex wake output is not silent failure. The dispatcher retries once read-only, writes diagnostics to `docs/guardian_action_review.md`, emits `WAKE_PARSE_FAILURE` through dispatcher state, and queues repeated same-event failures for the active trader.

## 2026-06-30 Target Cadence Policy

- Optimize the system toward rolling 30-day 4x account growth, not a hard forced +5% every UTC day. +5% is a pace marker / review trigger / protection milestone; it must not force bad-day trades.
- +10% remains extension-only behind the favorable-market gate. Red/no-edge days are allowed inside the rolling plan, but margin closeout, unattended carry, and counting operator manual precedent as system edge remain blocked.
- Runtime target state now reports the rolling policy fields every cycle: `rolling_30d_start_equity`, `current_equity`, `current_30d_multiplier`, `remaining_to_4x`, `required_calendar_daily_return`, `required_active_day_return`, and `pace_state`. Verifier pace pressure applies only to A/S or attack-recommended lanes; B/C lanes are not forced to repair a +5% pace miss.

## 2026-06-30 Generalized Discretionary Trade Shape

- The 2025 USD_JPY manual history is not a USD_JPY-only rule. It is operator precedent for a reusable trade shape: read theme, build only when thesis is alive, prefer bounded adverse add over with-move pyramid, avoid tight SL in noise, harvest actively, and forbid margin closeout / unattended carry.
- Apply the common shape across current candidate pairs first; pair-specific overlays adjust ranking and sizing caution only. They do not replace risk geometry and they do not grant live permission.

## 2026-06-30 SL_LINT / THESIS_INVALIDATION_EXIT

- A stop is not a market read. A red position is not thesis invalidation. Broker SL must represent true invalidation or emergency protection. If the thesis is alive, the system should manage size, time, and exposure, not blindly cut.
- Broker SLs near major figures, spread/ATR noise, recent wick/stop-run shelves, event/intervention zones, or duplicated same-direction JPY themes must be blocked unless they are explicitly emergency-only and documented as such.
- Loss-side CLOSE needs real thesis invalidation. Negative P/L, NEGATIVE_EXPECTANCY, duplicate blockers, low LIVE_READY, and stale SL templates are execution-quality evidence, not exit authorization.
- After a stop-out, review whether the thesis failed or the broker SL failed. If price later moves in the intended direction and the SL was inside a noise/battle zone, the next cycle should consider re-entry/scout instead of treating the stop as proof that the read was wrong.

## 2026-06-30 Operator Manual USD_JPY 162 Fade

- The current confirmed USD_JPY SHORT 22,000u manual exposure is `operator_manual` / `OPERATOR_ALPHA_CANDIDATE`, not unknown system risk. It is observed, TP-assisted, and reported only.
- Do not count this exposure in system profitability or trader risk-budget progress. Account-level NAV/UPL may still report it as broker truth, but system P/L must not claim it.
- Red UPL is not invalidation for the 162.00 historical/intervention-risk fade. Thesis state requires exact evidence: `ALIVE` below the figure without accepted break, `WOUNDED` on wick/stop-run or touch without acceptance, `INVALIDATED` only after accepted trade above 162.00, and `EMERGENCY` only on true margin/protection emergency.
- While this operator_manual USD_JPY exposure exists, fresh bot USD_JPY and JPY-cross adds are blocked unless the operator explicitly authorizes overlap. The system must not attach SL, loss-close, or average into the manual exposure by itself.
- Confirmation is tranche-scoped. If broker truth shows more no-receipt USD_JPY short units than the operator-confirmed 22,000u, classify only the oldest exact 22,000u tranche as `operator_manual`; leave surplus unknown exposure visible for operator review rather than silently adopting it.

## 2026-06-28 UTC Daily Target Engine

- Base operating target is +5% from UTC day-start NAV.
- +10% is extension-only after an explicit favorable-market gate; default mode after +5% is PROTECT.
- Day-start NAV is persisted under `logs/day_start_nav/YYYY-MM-DD.json`.
- Intraday P/L selection uses UTC 00:00 boundaries. Slack/operator displays may show JST, but the trading-day math stays UTC.
- This change is read-only for live ordering: no gateway, send, cancel, or close behavior was changed.

## 2026-06-28 FULL_TRADER 5% Pace Discipline

- Every FULL_TRADER session must fill the `5% PACE BOARD` and map it to `ATTACK STACK` before reporting a decision.
- Under +5%, Path A / HERO must be a concrete route to the +5% floor or an exact blocker with next trigger and shelf-life.
- B/C second-shot or no-honest-path churn cannot be presented as the +5% pace path, and one distant pending order is not enough.
- This change is read-only for live ordering: it changes session reporting and validation only.

## 2026-06-28 Target-Aware Dry-Run Sizing

- Added `tools/position_sizing.py` and dry-run-only `tools/place_trader_order.py` so fresh target-path orders can be sized against target contribution, explicit risk capacity, optional margin capacity, and conviction grade before any gateway path.
- S/A can be the main +5% pace path. B+ is scout/reload support only. B0/B-/C are not valid +5% pace trades.
- The 10% Extension Gate defaults to NO; YES requires strong progress or protected S/A carry, paying hero thesis, theme/trend confirmation, stable spread, no near whipsaw event, healthy last A/S trade, and a real reload/second-shot level.
- RiskEngine now blocks target-path metadata that violates the grade, extension, recent same-thesis loss, or 5% PACE BOARD / ATTACK STACK mapping contract. This is still dry-run/gateway-safe: no live OANDA order helper was added.

## 2026-06-29 Controlled Target-Path Live Learning

- Target-path live sends are still off by default. They require the normal `QR_LIVE_ENABLED=1 --send --confirm-live` path plus `QR_TARGET_PATH_LIVE_ENABLED=1`.
- `tools/place_trader_order.py` remains dry-run only. It can emit a LiveOrderGateway intent with `--gateway-intent-output`, but all OANDA entry posts still happen only inside `LiveOrderGateway`.
- Live target-path sends must carry `LIVE_LEARNING` receipts with daily target mode, remaining-to-5%, path role, attack-stack slot, grade, suggested/final units, risk, target contribution, and gateway receipt id.
- `daily-review` now classifies sent target-path trades as `discovery failure`, `deployment failure`, `sizing failure`, `vehicle failure`, `management failure`, or `good execution` so the next cycle can learn from the exact failure mode instead of treating all losses as generic lane bias.

## 2026-06-29 USER_ALPHA Continuation

- A profitable manual/operator-discovered winner is `USER_ALPHA` / `OPERATOR_ALPHA`, not proof that the system discovered the edge.
- The EUR_USD LONG operator-discovered winner exposed the gap: the broker/system could manage TP, but the trader did not convert the green outcome into RELOAD / SECOND_SHOT / +5% continuation.
- `daily-review` now publishes user-led winners separately under `user_alpha_trades` and `user_alpha_continuation`; GPT trader receipts must cite `user_alpha:continuation` and either continue the same pair/side or name an exact blocker and next trigger.
- Stale pending replacements must carry ignored pending ids through preflight and final send so `BASKET_DUPLICATE_PARENT_LANE` cannot block cancel/replace of the pending order being replaced.

## 2026-06-29 MARKET_READ_FIRST

- A blocker is not a market read. `LIVE_READY=0` is not a market read. Negative expectancy is not a market read.
- Codex must first predict price, then filter execution.
- A blocked but correct read is discovery success / execution miss.
- A wrong read that passes filters is market-read failure.
- Do not build direction-biased rules from biased samples; read the current chart for both directions first, then let execution gates decide whether anything can be traded.
