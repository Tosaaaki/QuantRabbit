# Receipt Promotion Report

- Generated at UTC: `2026-04-30T16:32:15.138144+00:00`
- Strategy profile: `/Users/tossaki/App/QuantRabbit/data/strategy_profile.json`
- Order intents: `/Users/tossaki/App/QuantRabbit/data/order_intents.json`
- Intent snapshot: `data/broker_snapshot.json`
- Profiles seen: `21`
- Promoted: `5`
- Still blocked: `12`

## Promotions

- `EUR_USD LONG` RISK_REPAIR_CANDIDATE -> CANDIDATE via `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
  - reason: loss-cap geometry repaired by current dry-run receipt
- `AUD_JPY LONG` RISK_REPAIR_CANDIDATE -> CANDIDATE via `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE`
  - reason: loss-cap geometry repaired by current dry-run receipt
- `EUR_USD SHORT` RISK_REPAIR_CANDIDATE -> CANDIDATE via `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
  - reason: loss-cap geometry repaired by current dry-run receipt
- `EUR_JPY LONG` MINE_MISSED_EDGE -> CANDIDATE via `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE`
  - reason: missed edge converted into STOP-ENTRY trigger receipt
- `GBP_USD LONG` MINE_MISSED_EDGE -> CANDIDATE via `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE`
  - reason: missed edge converted into STOP-ENTRY trigger receipt

## Promotion Contract

- `RISK_REPAIR_CANDIDATE` can promote only from a risk-allowed dry-run receipt with no blocking risk issue.
- `MINE_MISSED_EDGE` can promote only from a risk-allowed LIMIT or STOP-ENTRY receipt.
- `BLOCK_UNTIL_NEW_EVIDENCE` is never auto-promoted by this command.
