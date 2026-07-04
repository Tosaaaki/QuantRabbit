# A/S Lane Candidate Board

Generated: 2026-07-04T11:53:12Z  
Mode: evidence board only. No orders, cancels, closes, SL/TP changes, or execution flag changes.

Source: `/Users/tossaki/App/QuantRabbit-live/data/order_intents.json` generated `2026-07-03T20:16:34.043087+00:00`, with support artifacts from the same live runtime. `profitability_acceptance` must be refreshed before any routing decision because current on-disk `capture_economics.json` is newer than acceptance.

## Summary

- `LIVE_READY` lanes: 0 / 84
- A/S-current lanes: 0
- Best repair family: USD_JPY LONG passive LIMIT / bidask precision, but it is not live-ready.
- Best priority family from the user request: EUR_JPY SHORT local TP proof, but current S5 bid/ask evidence is negative.
- AUD_USD remains excluded from A/S path until expectancy and geometry are repaired.
- B/C churn, stale WAIT/REQUEST_EVIDENCE recovery, manual EUR_USD same-theme add, broad premium LONG / discount SHORT chase, and market-close leak patterns are not valid paths to 30d 4x.

## Candidate Lanes

### 1. failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT

- Pair/side/strategy: USD_JPY LONG BREAKOUT_FAILURE
- Current blockers: `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `STRATEGY_NOT_ELIGIBLE`
- Market read direction: USD_JPY confluence `LONG_LEAN`; forecast direction is DOWN with 0.1714 confidence, used as contrarian fade evidence rather than trend chase.
- Trade shape: LIMIT entry 161.355, TP 161.455, SL 161.285, 4000 units, RR 1.4286, risk 280.0 JPY, reward 400.0 JPY.
- 24h/7d location: 24h percentile 0.8381 inside 160.480-161.524; 7d percentile 0.2126 inside 160.984-162.729.
- Spread state: 0.8 pip.
- Forecast confidence: low, market support not executable.
- Bid/ask replay status: `LIVE_GRADE_DAILY_STABLE`; rule `USD_JPY_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7`; 91 samples, PF 3.2194, hit 0.7473, positive day rate 0.75, TP10/SL7.
- TP proof status: target source is `BIDASK_REPLAY_PRECISION`, but capture TP method scope is missing for `USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER`.
- Negative expectancy status: global capture is `NEGATIVE_EXPECTANCY`, payoff 0.412, market-close expectancy -756.7 JPY.
- Invalidation: invalid if SL 161.285 trades or campaign overlay vetoes.
- Harvest trigger: passive LIMIT retest with attached broker TP matching audited replay grid.
- No-add trigger: no add if strategy profile stays `BLOCK_UNTIL_NEW_EVIDENCE`, forecast support remains non-executable, or RR/geometry deviates from TP10/SL7 proof.
- Could become A/S: yes, but only after exact TP proof and USD_JPY LONG strategy-profile repair.
- Missing evidence: risk-resized dry-run receipts for same pair/side/method, exact capture TP scope, fresh GPT-5.5 receipt, refreshed profitability acceptance.

### 2. range_trader:USD_JPY:LONG:RANGE_ROTATION

- Pair/side/strategy: USD_JPY LONG RANGE_ROTATION
- Current blockers: `OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED`, `RANGE_PHASE_NOT_ROTATION`, `RANGE_ROTATION_BROADER_LOCATION_CHASE`, `EXHAUSTION_RANGE_CHASE`
- Market read direction: USD_JPY `LONG_LEAN`; forecast DOWN with 0.1714 confidence.
- Trade shape: LIMIT entry 161.363, TP 161.463, SL 161.293, 4000 units, RR 1.4286, risk 280.0 JPY, reward 400.0 JPY.
- 24h/7d location: 24h percentile 0.8458; 7d percentile 0.2172.
- Spread state: 0.8 pip.
- Forecast confidence: low and not executable for live.
- Bid/ask replay status: same USD_JPY contrarian live-grade TP10/SL7 rule as above.
- TP proof status: `MISSING_METHOD_EXIT` for `USD_JPY|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER`; OANDA vehicle `USD_JPY|LONG|range_reversion|tp1_sl0.75` is audit-only and requires local TP proof.
- Negative expectancy status: global capture remains negative.
- Invalidation: invalid if SL 161.293 trades or campaign overlay vetoes.
- Harvest trigger: only if range phase becomes real rotation and the TP10/SL7 shape remains exact.
- No-add trigger: current `BREAKOUT_UP`, broader location chase, or exhaustion chase stays active.
- Could become A/S: possible but less direct than the LIMIT breakout-failure lane because range phase is currently wrong.
- Missing evidence: local TP proof, phase repair, non-chase location, fresh acceptance.

### 3. trend_trader:EUR_JPY:SHORT:TREND_CONTINUATION

- Pair/side/strategy: EUR_JPY SHORT TREND_CONTINUATION
- Current blockers: `OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
- Market read direction: EUR_JPY short score 0.462 > long score 0.448; forecast DOWN with 0.2791 confidence.
- Trade shape: STOP-ENTRY 184.522, TP 184.339, SL 184.725, 2000 units, RR 0.9015, risk 406.0 JPY, reward 366.0 JPY.
- 24h/7d location: 24h percentile 0.8715 inside 183.925-184.610; 7d percentile 0.4371 inside 183.602-185.707.
- Spread state: 1.3 pips.
- Forecast confidence: low, market support not executable.
- Bid/ask replay status: `LIVE_BLOCK_NEGATIVE_EXPECTANCY`; `EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY`, 1147 samples, hit 0.2947, positive day rate 0.0, optimized PF 0.0.
- TP proof status: `MISSING_METHOD_SCOPE` for `EUR_JPY|SHORT|TREND_CONTINUATION|TAKE_PROFIT_ORDER`; OANDA vehicle `EUR_JPY|SHORT|trend_continuation|tp1_sl1` is historical audit-only.
- Negative expectancy status: global capture negative and local bid/ask negative.
- Invalidation: invalid if SL 184.725 trades or campaign overlay vetoes.
- Harvest trigger: none until local spread-included TP-positive proof replaces the negative S5 replay.
- No-add trigger: any attempt to use OANDA audit-only firepower as live permission.
- Could become A/S: not currently; requires new exact local TP-positive evidence.
- Missing evidence: non-negative bid/ask replay for EUR_JPY SHORT, local TP scope, improved RR above current 0.90, fresh forecast support.

### 4. range_trader:EUR_JPY:SHORT:RANGE_ROTATION

- Pair/side/strategy: EUR_JPY SHORT RANGE_ROTATION
- Current blockers: `OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED`, `RANGE_PHASE_NOT_ROTATION`, `RANGE_ROTATION_BROADER_LOCATION_CHASE`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
- Market read direction: same EUR_JPY short lean, but range phase is `BREAKOUT_PENDING`.
- Trade shape: LIMIT 184.580, TP 184.439, SL 184.736, 2000 units, RR 0.9038, risk 312.0 JPY, reward 282.0 JPY.
- 24h/7d location: 24h percentile 0.9562; 7d percentile 0.4646.
- Spread state: 1.3 pips.
- Forecast confidence: 0.2791 and not executable.
- Bid/ask replay status: same EUR_JPY DOWN negative rule; not live-grade.
- TP proof status: `MISSING_METHOD_EXIT`; OANDA vehicle `EUR_JPY|SHORT|range_reversion|tp1_sl1` is audit-only.
- Negative expectancy status: global capture negative and local bid/ask negative.
- Invalidation: invalid if SL 184.736 trades or campaign overlay vetoes.
- Harvest trigger: only if range becomes real rotation and spread-included replay turns TP-positive.
- No-add trigger: current high 24h location plus negative replay.
- Could become A/S: no in current geometry.
- Missing evidence: local TP proof, real rotation phase, non-negative S5 replay, RR repair.

### 5. failure_trader:USD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT

- Pair/side/strategy: USD_JPY SHORT BREAKOUT_FAILURE
- Current blockers: `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `STRATEGY_NOT_ELIGIBLE`
- Market read direction: USD_JPY dominant regime TREND_DOWN, but current range phase is `BREAKOUT_UP`.
- Trade shape: LIMIT 161.395, TP 161.298, SL 161.491, 4000 units, RR 1.0104, risk 384.0 JPY, reward 388.0 JPY.
- 24h/7d location: 24h percentile 0.8764; 7d percentile 0.2355.
- Spread state: 0.8 pip.
- Forecast confidence: 0.1714 and not executable.
- Bid/ask replay status: `USD_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY`, 525 samples, optimized PF 0.1116, positive day rate 0.0333.
- TP proof status: `MISSING_METHOD_SCOPE`.
- Negative expectancy status: global and local negative.
- Invalidation: invalid if SL 161.491 trades.
- Harvest trigger: none.
- No-add trigger: negative replay and strategy profile block.
- Could become A/S: no without a new vehicle.
- Missing evidence: exact TP-positive proof, new strategy-profile evidence, better RR.

### 6. range_trader:AUD_USD:LONG:RANGE_ROTATION

- Pair/side/strategy: AUD_USD LONG RANGE_ROTATION
- Current blockers: `RANGE_COUNTERTREND_RR_TOO_LOW`, `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `RANGE_PHASE_NOT_ROTATION`, `RANGE_ROTATION_BROADER_LOCATION_CHASE`, `EXHAUSTION_RANGE_CHASE`, `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE`, `FORECAST_NOT_EXECUTABLE_FOR_LIVE`
- Trade shape: LIMIT 0.69331, TP 0.69458, SL 0.69163, 1000 units, RR 0.7560, risk 271.65 JPY, reward 205.36 JPY.
- 24h/7d location: 24h percentile 0.5718; 7d percentile 0.8880.
- Spread state: 1.4 pips.
- Forecast confidence: UNCLEAR, 0.0032.
- Bid/ask replay status: `AUD_USD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY`, 395 samples, optimized PF 0.1414, positive day rate 0.0.
- TP proof status: `MISSING_METHOD_EXIT`.
- Negative expectancy status: global and local negative.
- Could become A/S: no until expectancy, forecast, RR, range phase, and residual geometry are fixed.

### 7. range_trader:AUD_USD:SHORT:RANGE_ROTATION

- Pair/side/strategy: AUD_USD SHORT RANGE_ROTATION
- Current blockers: `REWARD_RISK_TOO_LOW`, `RANGE_COUNTERTREND_RR_TOO_LOW`, `TARGET_TOO_THIN_FOR_SPREAD`, `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `RANGE_PHASE_NOT_ROTATION`, `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE`, `FORECAST_NOT_EXECUTABLE_FOR_LIVE`
- Trade shape: LIMIT 0.69373, TP 0.69336, SL 0.69541, 1000 units, RR 0.2202, risk 271.65 JPY, reward 59.83 JPY.
- 24h/7d location: 24h percentile 0.6795; 7d percentile 0.9461.
- Spread state: 1.4 pips and target too thin for spread.
- Forecast confidence: UNCLEAR, 0.0032.
- Bid/ask replay status: `AUD_USD_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY`, 972 samples, optimized PF 0.6855, positive day rate 0.4242.
- TP proof status: `MISSING_METHOD_EXIT`.
- Negative expectancy status: global and local negative.
- Could become A/S: no. Current RR/target/spread geometry disqualifies it even before profitability.

## Shortest A/S Path

The shortest realistic path is not an immediate trade. It is evidence repair around USD_JPY LONG passive LIMIT / bidask precision:

1. Refresh profitability acceptance from the current capture and order intent inputs.
2. Produce exact `USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER` TP proof or equivalent same-shape spread-included acceptance support.
3. Repair `USD_JPY LONG` strategy profile from `BLOCK_UNTIL_NEW_EVIDENCE` using risk-resized dry-run receipts.
4. Rebuild order intents from a fresh broker snapshot.
5. Allow only if the same lane becomes A/S `LIVE_READY` and then receives fresh GPT-5.5 TRADE/ADD, RiskEngine pass, and LiveOrderGateway pass.

EUR_JPY SHORT remains a priority family for local TP proof generation, not for live routing today. AUD_USD is diagnostic only until expectancy and geometry are fixed.
