# Position Management Report

- Generated at UTC: `2026-05-06T11:43:48.356141+00:00`
- Action: `NO_POSITION`
- Positions: `0`

## Positions

- none

## Management Contract

- Existing positions are managed before any new entry is considered.
- Operator-managed manual/tagless positions are observed in broker truth but ignored by this gateway.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
