# Trading Improvement Report

- Generated at JST: `2026-05-05`
- Scope: QuantRabbit vNext profitability RCA, research synthesis, risk/coverage fixes
- Execution: read-only analysis and dry-run verification only; no live order was sent
- Deployment model: local scheduled-task/workspace operation only. VM deploy scripts, SSH deploys, and cloud instance restarts are not used for QuantRabbit.

## Position

「1日の開始残高の10%以上」を利益保証として扱うことはできない。今回の修正では、10%を日次KPIとして扱い、過大評価されたカバレッジや誤ったリスク換算を潰して、現実のブローカー真実と再現検証に近づけた。

現状は `COVERAGE_REQUIRES_REPLAY_EVIDENCE`。現在のライブ可能候補は日次残ターゲットを紙上では超えるが、50日リプレイでターゲットを覆えた日は4日だけなので、「確実に達成」とは判定しない。

## Current State

- Daily target state: target `20,894.58 JPY`, progress `1,550.2960 JPY`, remaining `19,344.2840 JPY`.
- Risk budget: remaining `4,202.1337 JPY`, per-trade cap `1,050.5334 JPY`, target pace `4` trades/day.
- Broker truth: one protected trader-owned `EUR_USD SHORT` remains open with SL at breakeven and TP attached.
- Order intents: `9` lanes are `LIVE_READY`; opposing `EUR_USD LONG` lanes are now blocked by `OPPOSING_POSITION_EXISTS`; `STALE_CONVERSION_QUOTE` is `0`.
- Coverage optimizer: live-ready reward `23,881.2039 JPY`; sequential ladder reward `19,457.4399 JPY` over `6` steps.
- Replay: `50` days, historical target hits `0`, evidence target covered `4`, total historical net `-41,482.8194 JPY`, risk-capped net `-3,002.4926 JPY`.

## Root Causes Found

1. Fixed 500 JPY loss cap leaked into strategy mining, campaign planning, replay, and position repair. This understated/overstated reward-risk and made the 10% campaign look more executable than the current account budget allowed. Fixed by resolving the per-trade cap from `daily_target_state.json` and replay equity inputs.

2. USD/EUR/GBP quote-currency conversion was inferred from conversion pairs and could be marked stale even when OANDA already returned account-home conversion factors. Fixed by requesting and persisting broker `homeConversions`, then using those factors before quote-pair fallback.

3. Previous coverage promotion could pass current dry-run receipts but still lacked enough historical replay support. The system now keeps coverage blocked when replay evidence covers only `4/50` days.

4. Market data gaps remain material: OANDA order/position book calls returned `401 Unauthorized`, and option skew has no configured feed. These are product blockers for flow/skew-driven filters, not reasons to invent signals.

5. The deterministic trader could select a fresh `EUR_USD LONG` while a protected `EUR_USD SHORT` was open. That is not a clean portfolio add; on OANDA it can become unintended netting/position management. Fixed by blocking same-pair opposing entries in `RiskEngine` and keeping them out of `LIVE_READY`.

## Research Synthesis

- OANDA official v20 pricing defines `HomeConversions` for converting currency gains, losses, and position value into account home currency. That supports using broker-provided conversion factors for risk/P&L instead of treating a stale conversion pair as the source of truth: https://developer.oanda.com/rest-live-v20/pricing-df/
- CFTC COT reports are weekly Tuesday open-interest snapshots, typically released Friday. They are useful as slow macro positioning context, not an intraday entry trigger: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- BIS 2025 Triennial FX data is the structural source for FX turnover/currency mix. BIS says preliminary results were released on September 30, 2025 and final data with the December 2025 Quarterly Review. This supports treating retail candle volume as a weak proxy rather than actual decentralized FX volume: https://www.bis.org/statistics/rpfx25.htm
- FRED API and series such as `VIXCLS` and `BAMLH0A0HYM2` are suitable for risk-regime features around vol and credit stress: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- Bailey and Lopez de Prado's Deflated Sharpe Ratio work warns that selection bias, backtest overfitting, and non-normal returns inflate apparent strategy quality. QuantRabbit should track trial count and deflated/validated performance before promoting mined edges: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Lee and Mykland's jump-test work supports adding high-frequency jump/event filters around macro releases and sudden gap risk instead of assuming smooth intraday diffusion: https://galton.uchicago.edu/~mykland/paperlinks/leemykland_103106.pdf

## Implemented Changes

- Dynamic replay loss cap:
  - `src/quant_rabbit/replay.py`
  - `src/quant_rabbit/cli.py`
  - replay reports now show cap source, daily risk pct, and target trade pace.

- Dynamic strategy/campaign cap:
  - `src/quant_rabbit/strategy/miner.py`
  - `src/quant_rabbit/strategy/ensemble.py`
  - strategy reports and campaign lanes now use the daily target cap instead of a hardcoded `500 JPY`.

- Broker home conversions:
  - `src/quant_rabbit/broker/oanda.py`
  - `src/quant_rabbit/models.py`
  - `src/quant_rabbit/risk.py`
  - `src/quant_rabbit/strategy/intent_generator.py`
  - `src/quant_rabbit/target.py`
  - `src/quant_rabbit/strategy/position_manager.py`

- Position repair cap:
  - `src/quant_rabbit/strategy/position_manager.py`
  - repair SL sizing now reads `per_trade_risk_budget_jpy` from daily target state before falling back to policy defaults.

- TraderBrain dynamic cap scoring:
  - `src/quant_rabbit/strategy/trader_brain.py`
  - historical pretrade/live P&L scaling, repaired-worst-loss warnings, and risk geometry ranking now use the current per-trade cap rather than fixed `500/900/300/450 JPY` thresholds.

- Opposing-position protection:
  - `src/quant_rabbit/risk.py`
  - protected same-pair positions may be layered only in the same direction; opposite-direction fresh entries now emit `OPPOSING_POSITION_EXISTS` and must route through position management.

- Test coverage:
  - OANDA home conversion parsing.
  - Risk engine false stale-conversion prevention.
  - Intent, target, position, strategy, replay, trader-brain, and autotrade cycles.
  - Opposing same-pair portfolio entries are blocked before live staging.
  - Full suite: `280` tests passed.

## Remaining Blockers

- 10% daily KPI is not replay-certified. Latest replay covers target on `4/50` days only.
- Current ladder is potential reward, not guaranteed realized P/L. It must be tick-replayed against entry fill, spread, slippage, TP/SL order priority, and news/jump windows.
- Flow data is incomplete until OANDA orderBook/positionBook authorization is fixed or the feature is explicitly disabled.
- Option skew is unavailable until a provider is configured.
- Backtest promotion still needs explicit trial-count accounting and deflated/validated edge reporting before increasing risk.
- The current external GPT decision response is stale and was rejected (`BAD_METHOD`); it cited 15 live-ready lanes while the current post-fix packet has 9. A fresh decision receipt is required before any send path.

## Next Engineering Actions

1. Add a tick-level replay gate for the current `6`-step ladder and block promotion unless fill path, spread, TP/SL sequence, and conversion all pass.
2. Add per-setup trial counters and deflated Sharpe / PBO-style metadata to `strategy_profile.json`.
3. Wire or disable order/position book feeds so flow blockers are explicit and not repeatedly rediscovered.
4. Configure an option-skew provider or keep skew features hard-disabled in scoring.
5. Add event/jump filters using calendar tier, Lee-Mykland-style jump flags, VIX/credit stress, and JPY intervention narrative as size-down or no-new-risk constraints depending on contract severity.
