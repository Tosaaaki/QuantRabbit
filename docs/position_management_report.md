# Position Management Report

- Generated at UTC: `2026-05-04T04:16:47.207889+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470122` `EUR_USD LONG` units=`3000` action=`HOLD_PROTECTED` upl=`145.3`
  - scores: same=`-257.15` opposite=`-274.56`
  - protection plan: sl=`None` tp=`None`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 0 JPY
  - reason: remaining reward about 1784 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
