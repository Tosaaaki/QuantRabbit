# Position Management Report

- Generated at UTC: `2026-06-26T06:40:23.451215+00:00`
- Broker snapshot fetched at UTC: `2026-06-26T06:40:02.668056+00:00`
- Action: `NO_POSITION`
- Positions: `0`

## Positions

- none

## Management Contract

- Existing positions are managed before any new entry is considered.
- Operator-managed manual/tagless positions are eligible for TP-only profit management.
- Manual/tagless positions must never receive SL repair, SL tightening, or loss-close actions.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit clears remaining stop risk plus current session noise.
- SL-free break-even/profit-lock is allowed only after executable profit clears M5 ATR/spread micro-noise.
- Profit-only TAKE_PROFIT_MARKET is separate from loss-side REVIEW_EXIT Gate A/B discipline.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
- With QR_DISABLE_AUTO_CLOSE=1, deterministic REVIEW_EXIT stays advisory unless QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1 explicitly opts into structural auto-close; GPT CLOSE Gate A+B remains executable.
