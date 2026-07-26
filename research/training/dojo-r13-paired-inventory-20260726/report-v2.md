# r13 2025-01 OHLC paired inventory counterfactual

- Status: `EXPERIMENTAL_UNRANKED`
- Fixed denominator: 12 coordinates × 7 cadences = 84 measured cells
- Experimental best cadence: `FIXED_5M`
- Actual provider-model checkpoint cells: 0/84
- OLHC/paper-shadow candidates: none

## Twelve-account comparison

| family | cost | bot full-run net | policy full-run net | paired OOS effect | bot DD | policy DD | policy OOS+terminal PF | policy expectancy | evals | provider calls | interventions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | BASE | -56580.98 | -17929.81 | 38651.17 | 0.2851 | 0.0959 | 0.000 | -126.02 | 4320 | 0 | 4320 |
| burst | STRESS | -67554.85 | -21955.13 | 45599.72 | 0.3396 | 0.1144 | 0.000 | -137.81 | 4320 | 0 | 4320 |
| mean_revert_24h | BASE | -28245.65 | -14541.15 | 13704.50 | 0.1482 | 0.0727 | N/A | N/A | 4320 | 0 | 4320 |
| mean_revert_24h | STRESS | -36799.98 | -17281.68 | 19518.30 | 0.1892 | 0.0864 | N/A | N/A | 4320 | 0 | 4320 |
| prev_day_extreme_fade | BASE | 5784.01 | 2730.04 | -3053.97 | 0.0241 | 0.0241 | 1.655 | 17.26 | 4320 | 0 | 237 |
| prev_day_extreme_fade | STRESS | 2536.90 | -323.45 | -2860.34 | 0.0299 | 0.0331 | 1.380 | 11.38 | 4320 | 0 | 234 |
| pullback_limit | BASE | -8303.58 | -3594.77 | 4708.81 | 0.0582 | 0.0313 | 0.215 | -17.52 | 4320 | 0 | 4077 |
| pullback_limit | STRESS | -16167.54 | -4974.37 | 11193.18 | 0.0905 | 0.0349 | N/A | N/A | 4320 | 0 | 4320 |
| round_number_fade | BASE | 5441.42 | 4766.04 | -675.39 | 0.0104 | 0.0096 | 3.255 | 53.51 | 4320 | 0 | 51 |
| round_number_fade | STRESS | 4783.72 | 4102.73 | -680.99 | 0.0109 | 0.0101 | 2.756 | 46.80 | 4320 | 0 | 51 |
| spike_fade | BASE | -477.83 | 470.59 | 948.42 | 0.0110 | 0.0061 | 48.354 | 52.29 | 4320 | 0 | 8 |
| spike_fade | STRESS | -519.47 | 431.69 | 951.16 | 0.0110 | 0.0062 | 30.765 | 48.03 | 4320 | 0 | 8 |

Both arms have an identical calibration prefix and the policy only activates in OOS. Therefore the final-settlement full-run net difference is the paired OOS policy effect; absolute net and DD retain the shared calibration prefix. The eight block balance proxies are diagnostic only because they exclude the terminal flat-settlement boundary. Bot-only PF/win rate and TP-profit-retained are N/A because the immutable r13 baseline does not expose OOS trade-level gross wins/losses or TP-attributed cash. Account rows are independent and must not be summed.

The `policy evals` column is deterministic frozen-policy evaluation, not an AI provider call. Provider calls and provider cost are zero.

## Economic and safety detail

| family | cost | bot realized | policy realized | bot expectancy/trade | policy win rate | margin peak bot/policy | margin calls bot/policy | ruin bot/policy | execution cost bot/policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | BASE | -56580.98 | -17929.81 | -106.76 | 0.000 | 0.0810/0.0806 | 0/0 | 0/0 | 43638.20/13556.39 |
| burst | STRESS | -66129.99 | -21465.78 | -127.46 | 0.000 | 0.0810/0.0806 | 0/0 | 0/0 | 53754.96/17113.10 |
| mean_revert_24h | BASE | -28245.65 | -14541.15 | -68.23 | N/A | 0.0807/0.0806 | 0/0 | 0/0 | 31042.34/8895.31 |
| mean_revert_24h | STRESS | -35862.94 | -16970.72 | -88.89 | N/A | 0.0807/0.0806 | 0/0 | 0/0 | 38366.74/11353.98 |
| prev_day_extreme_fade | BASE | 5784.01 | 2730.04 | 49.86 | 0.472 | 0.0810/0.0817 | 0/0 | 0/0 | 12267.61/12267.61 |
| prev_day_extreme_fade | STRESS | 2865.32 | -139.07 | 21.87 | 0.410 | 0.0810/0.0816 | 0/0 | 0/0 | 14998.86/15098.98 |
| pullback_limit | BASE | -8303.58 | -3594.77 | -25.63 | 0.474 | 0.0806/0.0805 | 0/0 | 0/0 | 32979.73/11254.42 |
| pullback_limit | STRESS | -15234.52 | -4693.80 | -49.90 | N/A | 0.0806/0.0805 | 0/0 | 0/0 | 39416.73/11383.14 |
| round_number_fade | BASE | 5441.42 | 4766.04 | 201.53 | 0.614 | 0.0806/0.0808 | 0/0 | 0/0 | 3291.84/3291.84 |
| round_number_fade | STRESS | 4842.03 | 4136.15 | 177.17 | 0.571 | 0.0806/0.0808 | 0/0 | 0/0 | 3879.08/3910.56 |
| spike_fade | BASE | -477.83 | 470.59 | -477.83 | 0.556 | 0.0803/0.0803 | 0/0 | 0/0 | 89.55/89.55 |
| spike_fade | STRESS | -516.05 | 432.28 | -519.47 | 0.444 | 0.0803/0.0803 | 0/0 | 0/0 | 127.85/127.89 |

Bot-only win rate and PF remain N/A. All intervention audit entries passed the no-future/no-terminal/no-wall-clock checks and are retained in the hash-manifested raw result files.

## Promotion decision

None. Source quote coverage is unproved, the month is worn TRAIN with prior aggregate outcome exposure, and no actual provider model was called at checkpoints. The paired deltas are experimental only.
