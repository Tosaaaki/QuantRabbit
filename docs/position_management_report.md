# Position Management Report

- Generated at UTC: `2026-05-04T06:15:09.632045+00:00`
- Action: `REPAIR_PROTECTION_REQUIRED`
- Positions: `1`

## Positions

- `470130` `EUR_USD LONG` units=`20000` action=`REPAIR_PROTECTION_REQUIRED` upl=`1972.0`
  - scores: same=`-257.15` opposite=`-274.56`
  - protection plan: sl=`1.17274` tp=`None`
  - reason: missing TP/SL
  - reason: repair SL candidate 1.17274

## Management Contract

- Existing positions are managed before any new entry is considered.
- Missing TP/SL is a repair requirement, not a passive monitor state.
- Profit protection is required once open profit is large relative to remaining stop risk.
- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.
