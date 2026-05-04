# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T11:18:21.285812+00:00`
- Status: `ACCEPTED`
- Action: `TIGHTEN_SL`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: Protection cycle. EUR_USD SHORT moved from -75 JPY to +245 JPY in 5 minutes, daily progress 8.64%→9.8%. Tighten SL from 1.17068 to 1.17010 (BE+1pip) to lock profit while structure supports continuation to TP 1.16702. Near-target management per contract §5.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.
