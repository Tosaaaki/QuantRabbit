# Position Management Report

- Generated at UTC: `2026-04-30T17:01:46.516465+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470024` `EUR_USD LONG` units=`1000` action=`HOLD_PROTECTED` upl=`6.2`
  - scores: same=`164.85` opposite=`150.14`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 126 JPY
  - reason: remaining reward about 188 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
