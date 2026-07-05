# Month-Scale Residual Family Table

- Generated: `2026-07-05T16:24:00Z`
- Execution timing generated: `2026-07-03T20:08:53.084075+00:00`
- Harvest evidence generated: `2026-07-05T16:21:04.081371+00:00`
- Harvest source matches timing: `True`
- Family count: `23`
- Priority families: `7`
- Tail families: `16`
- All negative families can create live permission: `False`

## Replay

- Before filters baseline P/L JPY: `-39246.4662`
- Before filters improved/residual P/L JPY: `-20863.5316`
- After residual-family filters residual P/L JPY: `2984.1927`
- MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE clears after filters: `True`
- Excluded family count: `23`
- Remaining residual groups: `[]`

## Families

| family | trades | realized P/L | counterfactual P/L | residual P/L | cause | blocker | action | A/S now | can ever A/S | priority |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| GBP_USD LONG BREAKOUT_FAILURE | 472071 | -2981.8961 | -2981.8961 | -2981.8961 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `True` |
| AUD_USD LONG RANGE_ROTATION | 472952 | -2690.6967 | -2690.6967 | -2690.6967 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| EUR_USD LONG RANGE_ROTATION | 471817 | -2333.8215 | -2333.8215 | -2333.8215 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| EUR_USD SHORT RANGE_ROTATION | 471711 | -2181.1565 | -2181.1565 | -2181.1565 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| NZD_CAD SHORT RANGE_ROTATION | 472312,472380 | -2044.4543 | -2044.4543 | -2044.4543 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| AUD_USD SHORT RANGE_ROTATION | 472834 | -1705.6738 | -1705.6738 | -1705.6738 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| NZD_USD LONG RANGE_ROTATION | 472743 | -1380.8008 | -1380.8008 | -1380.8008 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| EUR_CHF LONG TREND_CONTINUATION | 472174,472445 | -1272.0771 | -1272.0771 | -1272.0771 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |
| EUR_JPY LONG RANGE_ROTATION | 472094 | -1071.9 | -1071.9 | -1071.9 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `True` |
| GBP_USD SHORT RANGE_ROTATION | 472837 | -971.0121 | -971.0121 | -971.0121 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| GBP_CHF LONG TREND_CONTINUATION | 472190 | -955.691 | -955.691 | -955.691 | `FORECAST_NOT_EXECUTABLE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_FORECAST_EXECUTABLE` | `False` | `True` | `False` |
| EUR_GBP SHORT BREAKOUT_FAILURE | 472208 | -891.0833 | -891.0833 | -891.0833 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |
| EUR_JPY SHORT RANGE_ROTATION | 472903 | -844.2 | -844.2 | -844.2 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| USD_JPY SHORT RANGE_ROTATION | 472775,472900 | -744.8 | -744.8 | -744.8 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| CHF_JPY SHORT RANGE_ROTATION | 472497 | -661.5 | -661.5 | -661.5 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| USD_CAD LONG TREND_CONTINUATION | 472427 | -620.0993 | -620.0993 | -620.0993 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |
| AUD_NZD SHORT RANGE_ROTATION | 472632 | -239.4791 | -239.4791 | -239.4791 | `NEGATIVE_BIDASK_REPLAY` | `MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED` | `REQUIRE_BIDASK_NON_NEGATIVE` | `False` | `True` | `False` |
| NZD_CAD LONG RANGE_ROTATION | 472088 | -218.1407 | -218.1407 | -218.1407 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| EUR_JPY LONG BREAKOUT_FAILURE | 472156 | -170.1 | -170.1 | -170.1 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |
| EUR_GBP SHORT TREND_CONTINUATION | 472125 | -169.1591 | -169.1591 | -169.1591 | `FORECAST_NOT_EXECUTABLE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_FORECAST_EXECUTABLE` | `False` | `True` | `False` |
| NZD_CHF LONG RANGE_ROTATION | 472530 | -96.6085 | -96.6085 | -96.6085 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` | `False` |
| GBP_AUD LONG TREND_CONTINUATION | 472233 | -48.3235 | -48.3235 | -48.3235 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |
| EUR_CHF LONG BREAKOUT_FAILURE | 472252 | -38.054 | -38.054 | -38.054 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` | `False` |

## Priority Repair Requirements

### GBP_USD LONG BREAKOUT_FAILURE
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `BAD_EXIT` / `REQUIRE_LOCAL_TP_PROOF`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where GBP_USD LONG BREAKOUT_FAILURE no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- spread-included local TP proof for the exact pair/side/method/order-type shape
- close-gate proof if any MARKET_ORDER_TRADE_CLOSE path is retained
- loss-side close path proves thesis invalidation and contained risk; otherwise use attached TP/HARVEST only

### AUD_USD LONG RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where AUD_USD LONG RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

### EUR_USD LONG RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where EUR_USD LONG RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

### EUR_USD SHORT RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where EUR_USD SHORT RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

### NZD_CAD SHORT RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where NZD_CAD SHORT RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

### AUD_USD SHORT RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where AUD_USD SHORT RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

### EUR_JPY LONG RANGE_ROTATION
- Decision: `BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE`
- Cause/action: `RANGE_CHASE` / `REQUIRE_GEOMETRY_REPAIR`
- A/S status: `NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration`
- fresh 744h execution-timing-audit where EUR_JPY LONG RANGE_ROTATION no longer has negative residual replay P/L
- profitability-acceptance regenerated from the same timing artifact and not stale against inputs
- order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family
- RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear
- fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY
- RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half
- TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion

## Residual Tail

- Tail family count: `16`
- Tail residual P/L JPY: `-9321.1285`
- `NZD_USD LONG RANGE_ROTATION`: `-1380.8008` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_CHF LONG TREND_CONTINUATION`: `-1272.0771` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `GBP_USD SHORT RANGE_ROTATION`: `-971.0121` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `GBP_CHF LONG TREND_CONTINUATION`: `-955.691` JPY, `FORECAST_NOT_EXECUTABLE`, `REQUIRE_FORECAST_EXECUTABLE`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_GBP SHORT BREAKOUT_FAILURE`: `-891.0833` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_JPY SHORT RANGE_ROTATION`: `-844.2` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `USD_JPY SHORT RANGE_ROTATION`: `-744.8` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `CHF_JPY SHORT RANGE_ROTATION`: `-661.5` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `USD_CAD LONG TREND_CONTINUATION`: `-620.0993` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `AUD_NZD SHORT RANGE_ROTATION`: `-239.4791` JPY, `NEGATIVE_BIDASK_REPLAY`, `REQUIRE_BIDASK_NON_NEGATIVE`, blocker `MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED`
- `NZD_CAD LONG RANGE_ROTATION`: `-218.1407` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_JPY LONG BREAKOUT_FAILURE`: `-170.1` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_GBP SHORT TREND_CONTINUATION`: `-169.1591` JPY, `FORECAST_NOT_EXECUTABLE`, `REQUIRE_FORECAST_EXECUTABLE`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `NZD_CHF LONG RANGE_ROTATION`: `-96.6085` JPY, `RANGE_CHASE`, `REQUIRE_GEOMETRY_REPAIR`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `GBP_AUD LONG TREND_CONTINUATION`: `-48.3235` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`
- `EUR_CHF LONG BREAKOUT_FAILURE`: `-38.054` JPY, `BAD_EXIT`, `REQUIRE_LOCAL_TP_PROOF`, blocker `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`

## Gate Definitions

- `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED`: The matching family disappears from a refreshed 744h replay, or exact local TP/geometry/forecast proof makes the filtered replay non-negative and order_intents no longer carry residual metadata. Can create A/S permission: `False`
- `MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED`: Spread-included non-negative replay plus current production-gate evidence removes the matching residual family. Can create A/S permission: `False`
