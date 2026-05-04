# Position Management Report

- Generated at UTC: `2026-05-04T20:15:14.436282+00:00`
- Action: `PROFIT_PROTECT_REQUIRED`
- Positions: `1`

## Positions

- `470188` `EUR_USD SHORT` units=`13000` action=`PROFIT_PROTECT_REQUIRED` upl=`856.6`
  - scores: same=`186.44` opposite=`200.85`
  - protection plan: sl=`1.16956` tp=`None`
  - reason: profit is large enough to require break-even/trailing review
  - reason: break-even SL candidate 1.16956
  - reason: remaining risk about 981 JPY
  - reason: remaining reward about 5886 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
