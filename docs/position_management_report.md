# Position Management Report

- Generated at UTC: `2026-05-04T21:21:39.517946+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470188` `EUR_USD SHORT` units=`13000` action=`HOLD_PROTECTED` upl=`244.8`
  - scores: same=`-270.56` opposite=`-253.15`
  - protection plan: sl=`None` tp=`None`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 0 JPY
  - reason: remaining reward about 5887 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
