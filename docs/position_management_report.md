# Position Management Report

- Generated at UTC: `2026-04-30T17:09:48.285203+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470024` `EUR_USD LONG` units=`1000` action=`HOLD_PROTECTED` upl=`-56.5`
  - scores: same=`164.85` opposite=`150.14`
  - protection plan: sl=`None` tp=`None`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 126 JPY
  - reason: remaining reward about 188 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
