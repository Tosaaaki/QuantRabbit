# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T17:18:16.794093+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: Selected EUR_USD SHORT (failure_trader BREAKOUT_FAILURE lane) from 15 LIVE_READY candidates. Strongest confluence: USD rank 1 (G8 strongest) vs EUR rank 4, pair chart TREND_DOWN with 0.975 SHORT score (97.5% indicator agreement), MTF cascade of BOS_DOWN structure (M5/M15/H1), DXY climbing +0.38%, COT showing EUR long-reduction (-8.7k week_change). Entry at 1.16864 STOP-ENTRY (S1 pivot area), SL 1.16913 (49 pips), TP 1.16571 (S3 target, 293 pips), 5.99R reward/risk. Risk 1,001 JPY < per_trade_cap 1,050 JPY (compliant). Spread 0.8pip NORMAL. No calendar event risk. Campaign progress 8.64% (behind pace), need high-R shots. Rejected AUD_JPY LONG 8.00R (counter-trend, AUD rank 8 weakest), EUR_USD LONG (against regime), GBP_USD LONG (weaker confluence). Decision verified against contract §9 (fresh broker truth), §3.5 (ATR-derived geometry, equity-derived units, no hardcoded caps), SKILL_trader anti-patterns (no invented thresholds, explicit gate citations). Proceeding to gpt-trader-decision verification then autotrade-cycle gateway send.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.
