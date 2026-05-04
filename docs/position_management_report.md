# Position Management Report

- Generated at UTC: `2026-05-04T11:54:21.459287+00:00`
- Action: `PROFIT_PROTECT_REQUIRED`
- Positions: `1`

## Positions

- `470140` `EUR_USD SHORT` units=`12000` action=`PROFIT_PROTECT_REQUIRED` upl=`978.7`
  - scores: same=`-266.56` opposite=`-249.15`
  - protection plan: sl=`1.17016` tp=`None`
  - reason: profit is large enough to require break-even/trailing review
  - reason: break-even SL candidate 1.17016
  - reason: remaining risk about 981 JPY
  - reason: remaining reward about 5922 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
