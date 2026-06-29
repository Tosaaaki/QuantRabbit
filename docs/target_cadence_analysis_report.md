# Target Cadence Analysis

- Generated at UTC: `2026-06-29T23:25:37+00:00`
- Scope: analysis/report generation only; no live orders, cancels, or position changes.
- Recommendation: **Optimize for Model B: rolling 30-day 4x account growth.**

## Target Model Comparison

| model | avg daily % | avg trading-day % | 30d multiplier | red days allowed | forced overtrading risk |
| --- | --- | --- | --- | --- | --- |
| A fixed daily +5% | 5.0 | 6.88 | 4.3219 | 0 | HIGH: a hard same-day +5% obligation turns normal red/no-edge days into incident pressure and can force churn after the edge is gone. |
| B rolling 30d 4x | 4.73 | 6.5 | 4.0 | 8 | LOWER: the model keeps the 4x arithmetic target but lets the trader skip bad days and treat +5% as pace/diagnostic evidence instead of a forced execution trigger. |

Model A is arithmetically stricter than the requested rolling 4x plan: +5% every calendar day compounds to 4.3219x over 30 days. Model B requires 4.73% average calendar-day return.

## Drawdown Sensitivity

Required remaining daily return after a drawdown on day 10:

Model A:

| drawdown % | remaining days | required daily % |
| --- | --- | --- |
| 5.0 | 20 | 7.87 |
| 10.0 | 20 | 8.16 |
| 20.0 | 20 | 8.8 |

Model B:

| drawdown % | remaining days | required daily % |
| --- | --- | --- |
| 5.0 | 20 | 7.45 |
| 10.0 | 20 | 7.74 |
| 20.0 | 20 | 8.38 |

## 2025 Precedent

- Active/best 30d evidence: `2025-06-13T00:02:29.467604+00:00` to `2025-07-10T04:56:44.898653+00:00`.
- Best funding-adjusted 30d return: `319.72`% = `4.1972`x.
- End funding-adjusted P/L: `269208.7038` JPY (`134.6`%).
- Exits: `411`, realized P/L: `266815.9` JPY, win rate: `0.511`, profit factor: `1.4497`.
- Best/worst day: `['2025-07-08', 276012.5]` / `['2025-07-11', -183158.0]`.
- Best 30d active UTC days: `12`, red UTC days: `5`, +5% days: `4`, +10% days: `3`.

## Precedent Breakdowns

Pair:

| bucket | exits | net JPY | win rate | payoff | median hold h |
| --- | --- | --- | --- | --- | --- |
| USD_JPY | 411 | 266815.9 | 0.511 | 1.3 | 0.48 |

Side:

| bucket | exits | net JPY | win rate | payoff | median hold h |
| --- | --- | --- | --- | --- | --- |
| LONG | 240 | 351347.9 | 0.596 | 1.78 | 0.72 |
| SHORT | 171 | -84532.0 | 0.392 | 1.14 | 0.25 |

Session:

| bucket | exits | net JPY | win rate | payoff | median hold h |
| --- | --- | --- | --- | --- | --- |
| LONDON_AM | 86 | 185804.8 | 0.651 | 1.9 | 1.35 |
| NY_OVERLAP | 188 | 88430.6 | 0.473 | 1.52 | 0.16 |
| OFF_HOURS | 34 | 14468.0 | 0.471 | 1.4 | 0.28 |
| TOKYO | 103 | -21887.5 | 0.476 | 0.9 | 0.43 |

Exit reason:

| bucket | exits | net JPY | win rate | payoff | median hold h |
| --- | --- | --- | --- | --- | --- |
| MARKET_ORDER_POSITION_CLOSEOUT | 143 | 391581.0 | 0.65 | 1.81 | 2.33 |
| TAKE_PROFIT_ORDER | 69 | 177659.0 | 1.0 |  | 0.1 |
| MARKET_ORDER_TRADE_CLOSE | 72 | 32508.7 | 0.653 | 0.79 | 1.05 |
| STOP_LOSS_ORDER | 103 | -117605.0 | 0.0 |  | 0.04 |
| MARKET_ORDER_MARGIN_CLOSEOUT | 24 | -217327.8 | 0.042 | 2.97 | 12.38 |

## Tail And Build Risks

- Margin closeout: `trades=24, net=-217327.8, win_rate=0.042, median_hold_hours=12.38, expectancy=-9055.3`.
- Long-hold excluded tail: `trades=88, net_jpy=161194.7, win_rate=0.648, median_hold_hours=108.13, expectancy_jpy=1831.8`.
- With-move pyramid bounded replay: `clusters=2, entries=10, net_jpy=-25195.0, win_rate=0.5, expectancy_jpy=-12597.5`.
- Bounded adverse-add summary: `clusters=8, entries=24, net_jpy=102564.0, win_rate=0.875, expectancy_jpy=12820.5, max_entries=4`.

## Daily Return Evidence

- Active UTC days reconstructed: `25`.
- Red days: `13`; green days: `12`.
- Days hitting +5%: `7`; days hitting +10%: `3`.
- Median red day return: `-7.62`%; worst day return: `-21.29`%.

## Origin Separation

- 2025 manual history is classified as `OPERATOR_MANUAL` and **not** counted as system profitability.
- 2025 manual realized P/L: `266815.858` JPY.
- Current execution-ledger origin buckets are included in the JSON artifact with conservative qrv1/lane-id attribution.

## Policy Answer

- Optimize for: **Optimize for Model B: rolling 30-day 4x account growth.**
- Use +5% as: **PACE_MARKER: use +5% to measure whether the rolling plan is on schedule, to trigger review when missed, and to unlock protection-first behavior when reached; do not force bad-day trades solely to print +5%.**
- Allow +10% when: Only when the existing 10% Extension Gate is YES: strong progress or protected S/A carry, hero thesis still paying, broad theme/trend confirmation, stable spread, no near whipsaw event, and real reload/second-shot levels.
- Bad/red days: With +10% good days, the 30-day 4x plan can mathematically absorb several red/flat days; the exact allowance must be recomputed from realized red-day severity, not assumed as permission to take weak trades.

## Behavior To Block

- Counting OPERATOR_MANUAL / USER_ALPHA results as system profitability.
- Forcing entries on no-edge days to satisfy a same-day hard target.
- Margin-closeout tolerance or any unbounded long unattended hold.
- With-move pyramiding without current basket/portfolio risk validation.
- Adverse adds that are not bounded by thesis invalidation, current location, and margin budget.
- USD_JPY-only rules; convert precedent into pair-agnostic theme/location/session/shape checks.
- Broker SL or close logic that does not represent thesis invalidation.

## Missing Data

- Intraday account-equity curve around deposits/withdrawals, so day-start equity remains estimated.
- MFE/MAE path per 2025 manual exit for exact drawdown and red-day severity modeling.
- Pair-agnostic H4/D context for every 2025 manual trade; current manual-context audit is H1/M5-centered.
- Explicit origin tags in 2025 history distinguishing operator discovery, system management, and user-alpha reloads.
- Current execution ledger close rows do not always preserve lane/client id on the closing event; attribution is conservative.
- No Notion page was found that directly defines the rolling 30-day 4x policy; this report derives it from the user objective plus local precedent.

## Sources

- `/Users/tossaki/App/QuantRabbit/data/manual_history_2025_mining.json`
- `/Users/tossaki/App/QuantRabbit/data/operator_precedent_audit.json`
- `/Users/tossaki/App/QuantRabbit/data/manual_market_context_audit.json`
- `/Users/tossaki/App/QuantRabbit/data/daily_target_state.json`
- `/Users/tossaki/App/QuantRabbit/data/execution_ledger.db`
- `docs/trade_case_studies/`
- `collab_trade/strategy_memory.md`
- `docs/CHANGELOG.md`
- Notion: [quantrabbit trade reference](https://app.notion.com/p/38ef1c8e53a781eb8243ee2342891f2c)
- Notion: [NAV percent for asset-relative numbers](https://app.notion.com/p/38ef1c8e53a7818499b4ef739905e1a5)
- Notion: [basket execution + pace cap](https://app.notion.com/p/38ef1c8e53a7816eb2fff7e443a76524)
- Notion: [routine owns trading](https://app.notion.com/p/38ef1c8e53a7815ab946d33a299600b5)
