# Fast-bot 2026-08-28 shock causal audit — 2026-08-30

## 結論

- 今回の損失原因は単なるLONGではなく、下降継続ショック中にも `RANGE_ROTATION` の逆張りLONGを出し続けたレジーム遷移誤認です。実signal ledgerのmethod/strategy_idを使用し、価格から戦略名を捏造していません。
- 14:03のEUR/USDは `RANGE_ROTATION/LONG`、regime_score `-2.0`、spread `0.8` pips、結果 `STOP_LOSS_AMBIGUOUS_FILL_S5` / `-3.2` pipsです。
- 14:03のUSD/JPYは `BREAKOUT_FAILURE/LONG` signal自体は存在しました。非参加理由は `PASSIVE_LIMIT_NOT_TOUCHED_WITHIN_TTL` で、veto・spread・gap・quarantineではありません。実注文は全proposalがshadow-onlyかつexecution authority NONEのため0件です。
- 過去同型shockは raw 3,000件、比較可能 2,970件です。分類は continuation 1,159、V-reversal 318、whipsaw 1,493件です。
- 利益化は未達です。PF<1の案を改善とは呼ばず、損失回避と利益創出を分離します。

## 14:00–14:20 UTC actual proposal stream

| Arm | Filled / net pips / PF | Loss avoidance vs baseline | Profit creating |
|---|---:|---:|---:|
| `baseline` | 18 / -56.400 / 0.146747 | 0.000 | false |
| `shock_freeze_5m` | 13 / -44.400 / 0.124260 | 12.000 | false |
| `side_relative_regime_transition_veto` | 8 / -19.800 / 0.226562 | 36.600 | false |
| `trend_aligned_continuation_after_5m_half_size` | 0 / 0.000 / n/a | 56.400 | false |
| `v_reversal_confirmed_only` | 0 / 0.000 / n/a | 56.400 | false |
| `whipsaw_freeze` | 13 / -44.400 / 0.124260 | 12.000 | false |
| `bot_owned_50pct_staged_drain_proxy` | 18 / -28.200 / 0.146747 | 28.200 | false |

| Pair / method / side | Proposals | Filled | Net pips | PF |
|---|---:|---:|---:|---:|
| `EUR_USD/BREAKOUT_FAILURE/LONG` | 3 | 3 | -14.700 | 0.000000 |
| `EUR_USD/BREAKOUT_FAILURE/SHORT` | 3 | 2 | -1.100 | 0.755556 |
| `EUR_USD/RANGE_ROTATION/LONG` | 14 | 12 | -43.000 | 0.083156 |
| `EUR_USD/RANGE_ROTATION/SHORT` | 1 | 1 | 2.400 | n/a |
| `USD_JPY/BREAKOUT_FAILURE/LONG` | 7 | 3 | -7.200 | 0.250000 |
| `USD_JPY/RANGE_ROTATION/LONG` | 1 | 1 | 2.400 | n/a |

`catastrophic_stop_plus_structure_exit` はretained proposal outcomeだけでは新しいS5退出経路を再採点できないため、同一proposal streamでは未確認とし、価格proxyで埋めていません。historical M1 bid/ask cohortの別表でのみ比較します。

## Historical EUR/USD M1 bid/ask shock cohort

| Horizon | Continuation | 50% retrace | Mean MFE | Mean MAE |
|---|---:|---:|---:|---:|
| 5m | 44.04% | 7.14% | 3.377p | 3.606p |
| 15m | 45.45% | 24.78% | 6.748p | 6.953p |
| 30m | 46.13% | 38.48% | 9.825p | 9.851p |
| 60m | 48.08% | 52.19% | 13.845p | 13.811p |

| Bounded arm | Trades / net pips / PF |
|---|---:|
| `baseline_immediate_continuation` | 2970 / -4923.100 / 0.800386 |
| `new_shock_guard` | 0 / 0.000 / n/a |
| `new_shock_guard_plus_50pct_drain_proxy` | 2970 / -2461.550 / 0.800386 |
| `trend_aligned_continuation_after_5m_half_size` | 1171 / -1013.450 / 0.777310 |
| `v_reversal_after_failed_continuation` | 750 / -1409.100 / 0.752277 |
| `whipsaw_freeze` | 1921 / -2422.550 / 0.763403 |
| `catastrophic_stop_plus_structure_exit` | 2970 / -5992.675 / 0.386862 |

ATRはonset triggerに使っていません。volatility bandはshock前60分のraw range、cross-pair confirmationはhistorical inputにUSD/JPY truthがないため unavailable です。

## Profitability frontierとの統合

- requested 224-signal corrective snapshot best PF: 0.488
- latest retained corrective scorecard: 292 signals / best PF 0.406226
- shock continuation validation PF: 0.531544 / net -346.938839 pips
- nonshock hourly PF: 0.864926 / net -178.5 pips
- profitability frontier trade-eligible candidates: 0
- 採用はzero-authority shadow観測のみ。live昇格条件は独立holdoutでafter-cost PF>1、上下両方向、cost stress、十分な日数/件数を同時に満たすこと。停止条件はstale/gap、seal drift、PF<=1、tail悪化、片方向集中、実行権限の非NONE化です。

## Authority

`execution_authority=NONE`, `Gateway invocation=0`, `external_order_attempts=0`, `external_orders=0`, manual/tagless `NO_TOUCH`。
