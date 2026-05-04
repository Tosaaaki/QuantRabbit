# Position Management Report

- Generated at UTC: `2026-05-04T14:24:01.282420+00:00`
- Action: `HOLD_PROTECTED`
- Positions: `1`

## Positions

- `470154` `EUR_USD SHORT` units=`13000` action=`HOLD_PROTECTED` upl=`-429.6`
  - scores: same=`-266.56` opposite=`-249.15`
  - protection plan: sl=`None` tp=`None`
  - reason: TP/SL present and current thesis is not contradicted enough to force exit
  - reason: remaining risk about 1001 JPY
  - reason: remaining reward about 5860 JPY

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
