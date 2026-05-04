# Position Management Report

- Generated at UTC: `2026-05-04T15:00:21.610717+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470154` `EUR_USD SHORT` units=`13000` action=`HOLD_PROTECTED` upl=`1345.4`
  - scores: same=`186.44` opposite=`200.85`
  - protection plan: sl=`None` tp=`None`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 0 JPY
  - reason: remaining reward about 5862 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
