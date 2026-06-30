# Changelog

## 2026-06-30

- Added the Guardian Trigger Contract: hourly cycles now refresh `data/guardian_trigger_contract.json` / `docs/guardian_trigger_contract_report.md`, preserve trader-defined trigger arrays, and let `guardian-event-router` wake GPT-5.5 for explicit contract harvest/add/no-add/wounded/invalidation/emergency triggers or `CONTRACT_STALE` when exposure exists without a fresh valid contract.
- Hardened Guardian GPT-5.5 wake receipts: prompt requires JSON-only schema, dispatcher falls back from empty last-message to stdout, classifies empty output as `CODEX_EMPTY_LAST_MESSAGE` and banner/no-JSON output as `CODEX_NO_JSON_RECEIPT`, retries once read-only, records parse failures as `WAKE_PARSE_FAILURE`, and queues repeated same-event parse failures for the active trader instead of endless Codex wakes.
- Changed active `qr-trader` runtime policy to gpt-5.5 high every 60 minutes, with guardian probe/router remaining deterministic and frequent for risk monitoring.
- Installed guardian wake dispatcher activation requirements for `com.quantrabbit.guardian-wake-dispatcher`: launchd stdout/stderr use dedicated `.launchd.*` logs, `QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0`, `QR_GUARDIAN_ACTION_EXECUTE=0`, and `CODEX_DISABLE_UPDATE_CHECK=1` are explicit, preserving read-only GPT wake behavior by default.
- Added `guardian-action-cycle` safe CLI handoff for GPT-5.5 guardian action receipts. Defaults remain no-send unless `QR_LIVE_ENABLED=1`, `QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=1`, and `QR_GUARDIAN_ACTION_EXECUTE=1`; receipt, broker-truth, lock, duplicate, thesis-state, manual-exposure, RiskEngine, and gateway checks run before any execution.
- Added `tools/guardian_wake_dispatcher.py`, `docs/guardian_wake_prompt.md`, and a launchd plist for event-driven GPT-5.5 guardian wakes through read-only `codex exec`; default behavior writes review/receipt artifacts only, never executes broker actions, and treats generated guardian action reviews as live runtime drift.
- Activated `ROLLING_30D_4X` in runtime target reporting: trader cycles now surface rolling 30d start equity, current equity, multiplier, remaining-to-4x, required calendar/active-day returns, and pace state; +5% is treated as pace/review/protection rather than forced daily churn.
- Updated verifier pace pressure so WAIT/REQUEST_EVIDENCE is forced only when A/S or attack-recommended `LIVE_READY` lanes exist; B/C lanes are not forced solely to hit the +5% pace marker.
- Added `tools/target_cadence_analysis.py` plus target-cadence and trade-shape precedent artifacts comparing hard daily +5% cadence with rolling 30-day 4x growth; recommendation is rolling 30-day 4x, with +5% as a pace marker and +10% only behind the extension gate.
- Added a pair-agnostic discretionary trade-shape engine: the 2025 USD_JPY manual history is now advisory source evidence for a reusable shape across current pairs, with pair-specific overlays limited to score adjustments and no live-permission authority.
- Added `SL_LINT`, `THESIS_INVALIDATION_EXIT_REQUIRED`, and `POST_STOP_THESIS_REVIEW`: broker SLs now publish invalidation evidence and block major-figure/noise/wick/event/JPY-theme stop placement; loss-side CLOSE receipts cannot use red P/L, negative expectancy, duplicate blockers, low LIVE_READY, or old SL templates as standalone exit reasons.
- Added explicit `operator_manual` classification for confirmed USD_JPY manual exposure: emits `OPERATOR_MANUAL_POSITION`, excludes the exposure from system P/L, blocks fresh USD_JPY/Jpy-cross bot adds while active unless operator-authorized, and treats the 162.00 fade thesis as alive until an accepted break above the figure is proven.
- Made operator-manual confirmation tranche-aware: if broker truth has additional no-receipt USD_JPY shorts beyond the confirmed unit count, only the oldest exact confirmed tranche is classified and surplus exposure remains unknown for operator review.

## 2026-06-29

- Added MARKET_READ_FIRST receipt discipline: GPT trader receipts must include naked read, 30m/2h prediction, and forced-trade read before blockers; verifier rejects blocker-first, LIVE_READY=0-only, and NEGATIVE_EXPECTANCY-only reasoning; market-read predictions are recorded/scored separately from trade P/L.
- Added USER_ALPHA / OPERATOR_ALPHA continuation handling: daily-review publishes profitable user-led winners separately, GPT trader receipts must answer continuation or exact blocker, and stale pending replacement sends retain ignored pending ids through final gateway validation.
- Added controlled target-path live routing behind `QR_TARGET_PATH_LIVE_ENABLED=1`, with LiveOrderGateway receipts and LIVE-LEARNING daily-review classification.

## 2026-06-28

- Added `tools/position_sizing.py` and dry-run-only `tools/place_trader_order.py` for target-aware, risk-based order sizing.
- Added the `10% EXTENSION GATE` checklist and RiskEngine target-path metadata guards.
- Added `docs/SKILL_daily-review.md` Daily Target Review with +5%/+10% miss classification.
- Added the required FULL_TRADER `5% PACE BOARD` and `ATTACK STACK` session contract.
- Added read-only trader-state/task-sync validators so B/C churn cannot be documented as a +5% path.
- Added `tools/daily_target.py` as a read-only UTC 00:00 daily target engine.
- Added `tools/session_data.py` to print the session-start daily target block with +5% base target and +10% extension target.
- Added `tools/intraday_pl_update.py` so intraday P/L uses UTC trading-day boundaries while display surfaces may still show JST.
- Documented that +10% is extension-only and does not change live ordering behavior in this change.
