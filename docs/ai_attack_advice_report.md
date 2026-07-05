# AI Attack Advice Report

- Generated at UTC: `2026-07-05T18:09:03.833132+00:00`
- Status: `NO_ATTACK_ADVICE`
- Read only: `True`
- Live permission: `False`
- Live-ready lanes: `0`
- Live-ready reward: `0 JPY` (`0.0%`)
- Recommended now: `0` lanes, reward=`0 JPY`, risk=`0 JPY`
- Required additional reward: `28719 JPY`
- Required additional live-ready lanes: `None`
- Projection economic precision edges: `10`
- Capture segment priority edges: `1`

## Recommended Now

- none

## Watchlist

- none

## Matrix-Supported Repair Queue

- `EUR_USD LONG` state=`SURFACED_BUT_BLOCKED` profile=`BLOCK_UNTIL_NEW_EVIDENCE` matrix_support=`8` managed_net=`16698.5786` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `AUD_JPY SHORT` state=`SURFACED_BUT_BLOCKED` profile=`BLOCK_UNTIL_NEW_EVIDENCE` matrix_support=`9` managed_net=`7196.8676` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `GBP_USD LONG` state=`SURFACED_BUT_BLOCKED` profile=`BLOCK_UNTIL_NEW_EVIDENCE` matrix_support=`8` managed_net=`5771.1754` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `EUR_JPY LONG` state=`SURFACED_BUT_BLOCKED` profile=`MINE_MISSED_EDGE` matrix_support=`7` managed_net=`3515.8` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `GBP_JPY LONG` state=`SURFACED_BUT_BLOCKED` profile=`BLOCK_UNTIL_NEW_EVIDENCE` matrix_support=`7` managed_net=`571.7416` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `USD_JPY LONG` state=`SURFACED_BUT_BLOCKED` profile=`BLOCK_UNTIL_NEW_EVIDENCE` matrix_support=`7` managed_net=`131.0` blocker=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`

## Projection Edge Activation Queue

- `session_expansion_london` bucket=`EUR_USD:UNCLEAR` direction=`EITHER` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.963` matched_lanes=`0` blocker=``
- `liquidity_sweep_high_up` bucket=`AUD_JPY:UNCLEAR` direction=`UP` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.9191` matched_lanes=`0` blocker=``
- `bb_squeeze_expansion_imminent` bucket=`_all_pairs:TREND` direction=`EITHER` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.9455` matched_lanes=`0` blocker=``
- `bb_squeeze_expansion_imminent` bucket=`_all_pairs:RANGE` direction=`EITHER` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.9155` matched_lanes=`0` blocker=``
- `bb_squeeze_expansion_imminent` bucket=`_all_pairs:UNCLEAR` direction=`EITHER` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.9155` matched_lanes=`0` blocker=``
- `session_expansion_ny` bucket=`_all_pairs:TREND` direction=`EITHER` status=`EDGE_READY_NO_CURRENT_SIGNAL` repair=`DETECTOR_REFRESH_WAIT` economic_Wilson95_lower=`0.9071` matched_lanes=`0` blocker=``

## Precision Filtered

- none

## Blockers

- no LIVE_READY lanes are available for attack advice

## Action Items

- build additional LIVE_READY receipts for 28719 JPY of target coverage
- repair matrix-supported profitable edges before broad exploration: EUR_USD LONG (GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED); AUD_JPY SHORT (GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED); GBP_USD LONG (GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED)
- activate projection economic precision edges only after current blockers clear: session_expansion_london EUR_USD:UNCLEAR (DETECTOR_REFRESH_WAIT); liquidity_sweep_high_up AUD_JPY:UNCLEAR (DETECTOR_REFRESH_WAIT); bb_squeeze_expansion_imminent _all_pairs:TREND (DETECTOR_REFRESH_WAIT)
- resolve coverage optimizer status COVERAGE_GAP before treating attack advice as certified

## Contract

- This advice is read-only and never places, stages, or resizes broker orders.
- `LiveOrderGateway` remains the final broker-truth and risk authority.
- Do not raise loss caps from this report; fix coverage by adding validated lanes or improving execution evidence.
