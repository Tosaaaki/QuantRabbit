# 2026-06-29 USD/JPY 162 Fade and JPY-Cross SL Failure

## Summary

The operator correctly identified the USD/JPY 162.00 area as a major
psychological and intervention-risk zone. The bot treated the same area as a
normal `RANGE_ROTATION` setup with a tight broker SL, then stopped out just
before the thesis played out.

The same failure shape appeared in EUR/JPY: JPY-cross short thesis, attached
SL, and loss before the broader JPY reversal could be managed discretionarily.

## Broker Truth

Source artifacts:

- Live execution ledger: `/Users/tossaki/App/QuantRabbit-live/data/execution_ledger.db`
- Broker snapshot after operator add: `/tmp/qr_broker_snapshot_after_5k_add.json`
- News digest: `/Users/tossaki/App/QuantRabbit-live/logs/news_digest.md`
- Market context matrix: `/Users/tossaki/App/QuantRabbit-live/data/market_context_matrix.json`

## Timeline

| UTC | JST | Actor | Pair | Action | Units | Price | TP | SL | Result |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-06-29 12:08:17 | 2026-06-29 21:08:17 | bot | USD/JPY | placed SHORT limit | 8,000 | 161.884 | 161.826 | 161.941 | pending |
| 2026-06-29 12:26:43 | 2026-06-29 21:26:43 | bot | USD/JPY | filled SHORT | 8,000 | 161.884 | 161.826 | 161.941 | open |
| 2026-06-29 13:36:19 | 2026-06-29 22:36:19 | bot | USD/JPY | SL close | 8,000 | 161.942 |  |  | -464.0 JPY |
| 2026-06-29 13:24:09 | 2026-06-29 22:24:09 | bot | EUR/JPY | filled SHORT | 6,300 | 184.649 | 184.504 | 184.779 | open |
| 2026-06-29 13:56:50 | 2026-06-29 22:56:50 | bot | EUR/JPY | SL close | 6,300 | 184.783 |  |  | -844.2 JPY |
| 2026-06-29 13:47:34 | 2026-06-29 22:47:34 | operator | USD/JPY | market SHORT | 15,000 | 161.938 | none | none | open |
| 2026-06-29 14:03:20 | 2026-06-29 23:03:20 | operator | USD/JPY | added SHORT | 5,000 | 161.892 | none | none | open |

At `2026-06-29 14:03:20 UTC / 23:03:20 JST`, broker truth showed:

- `472909`: USD/JPY SHORT 15,000u from `161.938`, UPL `+555 JPY`.
- `472913`: USD/JPY SHORT 5,000u from `161.892`, UPL `-45 JPY`.
- Combined USD/JPY SHORT exposure: 20,000u, weighted average about `161.9265`.
- Combined UPL: `+490 JPY`.
- Quote: bid `161.894`, ask `161.902`.
- Orders: none.
- TP/SL: none.
- Margin used: `129,518.4 JPY`.
- Margin available: `46,719.2 JPY`.

## Operator Thesis

The operator thesis was not a generic range fade. It was a specific macro and
market-location thesis:

- `162.00` is a psychological figure.
- The market is near multi-decade yen weakness.
- Intervention risk can cap USD/JPY upside.
- A move into `161.94-162.00` is a short-building area if price fails to accept
  above the figure.
- A tight broker SL just below/around the figure is likely to be harvested before
  the thesis resolves.

The operator's 15,000u short and subsequent 5,000u add validated the near-term
read: the bot stopped at `161.942`, while the operator short from `161.938`
moved positive shortly after.

## Bot/System Thesis

The bot selected:

- Lane: `range_trader:USD_JPY:SHORT:RANGE_ROTATION`
- Role: `OANDA_FIREPOWER_ROUTE`
- Entry: `161.884`
- TP: `161.826`
- SL: `161.941`
- Planned risk: about `456 JPY`
- Planned reward: about `464 JPY`

The decision packet did include news and context references:

- `news:health`
- `news:items`
- `news:USD_JPY`
- `calendar:USD_JPY`
- `cross:dxy`
- `cross:USB10Y_USD`
- `matrix:USD_JPY:SHORT`

But the system weighted the context incorrectly. It saw chart/macro LONG
pressure and still placed a small SL short, instead of recognizing the special
case: a major round-number/intervention-risk fade needs either no tight SL, a
confirmed acceptance-above-figure invalidation, or much smaller scout sizing.

## What The Bot Missed

1. Major figure handling:
   `162.00` is not an ordinary local resistance. It is a psychological and
   narrative level. Stop placement just below/inside that battle zone is weak.

2. Intervention-risk interpretation:
   News saying intervention fears cap USD/JPY upside should not be treated only
   as "do not fade without chart confirmation." Near the figure it is also a
   reason to watch for failed acceptance and short build.

3. Post-stop thesis continuity:
   The bot's SL was hit at `161.942`; the operator then shorted `161.938` and
   was soon positive. The thesis did not fail. The broker SL failed.

4. JPY-cross theme grouping:
   USD/JPY and EUR/JPY were managed as separate small protected trades, but both
   expressed the same JPY reversal theme. Two separate attached SL losses turned
   a thematic read into repeated stop-outs.

5. Exit economics:
   `capture_economics` was already `NEGATIVE_EXPECTANCY`: average win `418.0`
   JPY vs average loss `999.5` JPY. The system keeps letting stop/close losses
   erase TP wins.

## Required Improvements

1. Add a `major_figure_intervention_fade` case model for USD/JPY and JPY
   crosses near major round numbers.

2. For that model, separate:
   - scout entry,
   - adverse add zone,
   - invalidation by acceptance above the figure,
   - hard no-add zone,
   - profit capture zone.

3. Do not attach a tight broker SL inside the major-figure battle zone. If a
   broker SL is required, it must represent true thesis invalidation, not normal
   figure probing noise.

4. Make `news_digest` intervention-cap language actionable:
   intervention cap + major figure + overextended yen weakness should become a
   specific fade/watch state, not just a generic warning.

5. Group same-theme JPY cross exposure before placing separate attached-SL
   trades. A USD/JPY short and EUR/JPY short should share a JPY-theme risk
   budget and management plan.

6. Preserve operator alpha as a case-study input. When the operator's manual
   trade succeeds immediately after a bot stop-out, the case must be promoted
   into strategy repair evidence rather than treated as unrelated manual
   exposure.

## Implemented Repair 2026-06-30

- `LiveOrderGateway` now writes `sl_lint` for every staged broker SL and blocks
  non-emergency stops inside major-figure, spread/ATR noise, wick/stop-run,
  event/intervention, and duplicated JPY-theme zones.
- `gpt-trader-decision` now emits `THESIS_INVALIDATION_EXIT_REQUIRED` when a
  loss-side close cites red P/L, `NEGATIVE_EXPECTANCY`, duplicate blockers, low
  `LIVE_READY`, or old SL templates without Gate A thesis invalidation.
- `POST_STOP_THESIS_REVIEW` answers whether the thesis failed or the broker SL
  failed, so a 162.00 stop-out followed by intended-direction movement becomes
  re-entry/scout evidence instead of being misread as a bad market thesis.

## Review Questions

1. Should major-figure fade trades be SL-free by default, with explicit
   acceptance-above-figure invalidation instead of broker SL?

2. How should the system detect "accepted above 162.00" versus "wick/stop run
   above 162.00"?

3. Should same-theme JPY-cross trades be limited to one protected bot position
   plus operator-managed manual adds?

4. Should `LONG_LEAN` near a major figure be interpreted as breakout pressure or
   late-stage squeeze risk depending on location and news?

5. How should operator manual success be converted into future live gating
   without overfitting or enabling revenge trades?
