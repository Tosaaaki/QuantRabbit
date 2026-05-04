# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T19:46:22.444887+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: EUR_USD SHORT via judas_armed reversal setup: Asian high swept, price now 36 pips below midnight open in NY PM killzone. H1 bearish structure (BOS_DOWN, RSI=43, %R=-65, AroonOsc=-71) supports downside despite M5 bullish family scores. STOP entry 1.16969 (2p below current) with tight 4.8p SL and 28.8p TP for 6:1 R:R. Risk 980 JPY (within 1050 budget). All JPY-cross lanes rejected for directional conflict with TREND_DOWN regimes.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.
