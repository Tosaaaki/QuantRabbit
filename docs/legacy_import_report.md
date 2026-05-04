# Legacy Import Report

- Archive: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`
- History DB: `/Users/tossaki/App/QuantRabbit/data/legacy_history.db`
- Imported at UTC: `2026-05-04T01:12:13.592235+00:00`

## Coverage

- Source files indexed: `11998`
- Live trade log events: `10389`
- Trader journal events: `37`
- `trades` rows: `6728`
- `pretrade_outcomes` rows: `704`
- `seat_outcomes` rows: `2145`
- `chunks` rows: `768`
- `user_calls` rows: `11`
- `market_events` rows: `13`

## Source Kinds

- `md`: `10514`
- `json`: `1099`
- `txt`: `294`
- `db`: `62`
- `jsonl`: `29`

## Live Log Action Coverage

- `NOTE`: `6581`
- `CLOSE`: `695`
- `POSITION_SNAPSHOT`: `501`
- `ENTRY`: `439`
- `ACCOUNT_SNAPSHOT`: `321`
- `LIMIT`: `244`
- `LEGACY_ENTRY`: `225`
- `SESSION_MARKER`: `193`
- `ORDER_REJECT`: `171`
- `MODIFY`: `161`
- `RANGE_BOT_LIMIT`: `145`
- `SIGNAL_NOTE`: `92`
- `CANCEL_ORDER`: `70`
- `CANCEL`: `62`
- `PARTIAL_CLOSE`: `55`
- `ENTRY_ORDER`: `53`
- `TREND_BOT_MARKET`: `39`
- `LIMIT_PLACED`: `36`
- `LIMIT_FILL`: `35`
- `MARKET_SNAPSHOT`: `24`

## Strongest Historical Pair/Direction Edges

- `EUR_USD LONG`: n=49 net=16174.0 JPY avg=330.1 JPY
- `EUR_JPY LONG`: n=23 net=3158.5 JPY avg=137.3 JPY
- `AUD_JPY LONG`: n=19 net=3152.2 JPY avg=165.9 JPY
- `EUR_USD SHORT`: n=34 net=1605.1 JPY avg=47.2 JPY
- `GBP_USD LONG`: n=21 net=1341.7 JPY avg=63.9 JPY
- `AUD_USD LONG`: n=9 net=-18.1 JPY avg=-2.0 JPY
- `USD_JPY LONG`: n=13 net=-29.0 JPY avg=-2.2 JPY
- `EUR_JPY SHORT`: n=8 net=-569.0 JPY avg=-71.1 JPY
- `GBP_USD SHORT`: n=14 net=-837.8 JPY avg=-59.8 JPY
- `USD_JPY SHORT`: n=17 net=-2544.0 JPY avg=-149.6 JPY
- `GBP_JPY SHORT`: n=5 net=-2590.0 JPY avg=-518.0 JPY
- `AUD_USD SHORT`: n=3 net=-3016.0 JPY avg=-1005.3 JPY

## Worst Live Losses To Design Against

- `2026-04-30 15:15:17 UTC` `USD_JPY LONG` 20000u P/L=-3360.0 reason=`structure_break`
- `2026-04-30 12:11:11 UTC` `USD_JPY LONG` 20000u P/L=-2740.0 reason=`TRAILING_STOP_LOSS_ORDER`
- `2026-04-14 18:17 UTC` `GBP_USD LONG` 8000u P/L=-2583.1 reason=`post-BoE_rule(not_above_1.35900_at_18:15Z)`
- `2026-04-23 15:06:56 UTC` `EUR_USD SHORT` 5000u P/L=-2076.6831 reason=`structure_break`
- `2026-04-28 15:26:42 UTC` `EUR_USD SHORT` 6000u P/L=-1439.2457 reason=`STOP_LOSS_ORDER`
- `2026-04-29 16:37:23 UTC` `EUR_JPY LONG` 8000u P/L=-1272.0 reason=`STOP_LOSS_ORDER`
- `2026-03-31T08:03:42Z` `AUD_JPY SHORT` 5000u P/L=-1190.0 reason=`SL_hit_bounce_extended_Fib170pct Sp=1.6pip`
- `2026-03-27 UTC` `EUR_USD SHORT` 3000u P/L=-1027.0 reason=`None`
- `2026-04-08 17:50 UTC` `AUD_USD LONG` 4000u P/L=-1016.0 reason=`MARGIN_EMERGENCY_98pct H1_ADX50_DI+intact_but_forced_close Sp=1.4pip`
- `2026-03-31 11:31:16 UTC` `EUR_JPY SHORT` 5000u P/L=-930.0 reason=`SL_triggered Sp=1.9pip`
- `2026-04-29 14:54:45 UTC` `EUR_USD SHORT` 8000u P/L=-898.0501 reason=`STOP_LOSS_ORDER`
- `2026-04-14 01:35Z` `GBP_JPY LONG` 4000u P/L=-876.0 reason=`SL_hit_Tokyo_thin Sp=3.2pip`

## Live Log Net By Pair/Direction

- `USD_JPY LONG`: n=21 net=-6954.8 JPY avg=-331.2 JPY
- `GBP_USD LONG`: n=25 net=-3935.3 JPY avg=-157.4 JPY
- `AUD_JPY SHORT`: n=28 net=-2660.0 JPY avg=-95.0 JPY
- `EUR_JPY SHORT`: n=18 net=-2284.5 JPY avg=-126.9 JPY
- `EUR_JPY LONG`: n=19 net=-1757.5 JPY avg=-92.5 JPY
- `AUD_USD LONG`: n=9 net=-1668.2 JPY avg=-185.4 JPY
- `GBP_USD SHORT`: n=17 net=-1501.6 JPY avg=-88.3 JPY
- `GBP_JPY SHORT`: n=6 net=-1269.5 JPY avg=-211.6 JPY
- `AUD_USD SHORT`: n=10 net=-1158.7 JPY avg=-115.9 JPY
- `EUR_GBP LONG`: n=5 net=-908.9 JPY avg=-181.8 JPY
- `USD_JPY SHORT`: n=12 net=-554.0 JPY avg=-46.2 JPY
- `NZD_USD SHORT`: n=8 net=-503.6 JPY avg=-62.9 JPY
- `GBP_JPY LONG`: n=15 net=-474.0 JPY avg=-31.6 JPY
- `NZD_JPY LONG`: n=2 net=-281.3 JPY avg=-140.6 JPY
- `EUR_CHF SHORT`: n=3 net=-209.1 JPY avg=-69.7 JPY
- `AUD_CAD SHORT`: n=1 net=-204.7 JPY avg=-204.7 JPY
- `AUD_NZD LONG`: n=2 net=-107.8 JPY avg=-53.9 JPY
- `GBP_USD None`: n=3 net=-105.9 JPY avg=-35.3 JPY
- `USD_JPY None`: n=1 net=159.7 JPY avg=159.7 JPY
- `CAD_JPY LONG`: n=10 net=417.7 JPY avg=41.8 JPY

## Rejection Reasons Feeding vNext Guards

- n=55 `exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk`
- n=11 `C size cap is 1000u but the order asks for 2000u`
- n=7 `A size floor is 4000u but the order asks for 3000u`
- n=4 `EUR_USD live tape probe is friction-dominated (range 0.0pip vs avg spread 0.8pip)`
- n=3 `OANDA immediate cancel`
- n=3 `B0 size floor is 2000u but the order asks for 1000u`
- n=3 `B0 size cap is 3000u but the order asks for 4000u`
- n=2 `size asymmetry guard: dirty seat B0 at 3000u is larger than the recent paid winner size 2000u`
- n=2 `exact pretrade only allows LIMIT here; the seat still needs price improvement: same-pair reload stays trigger/passive only while the existing leg is unpaid but protected; require a cleaner price/trigger than the current inventory`
- n=2 `exact pretrade blocks this order: learning cap or final allocation says this seat is not worth real risk`
- n=2 `exact pretrade blocks this order: EUR_USD planned SL 4.6pip is still inside the recent noise floor 6.1pip (recent stop-loss regret: 7/10 recovered in 6h, avg loss 6.1pip -> avg later favorable 14.1pip); widen the stop or improve the entry first`
- n=2 `exact pretrade blocks this order: EUR_JPY planned SL 6.4pip is still inside the recent noise floor 9.5pip (recent stop-loss regret: 4/4 recovered in 6h, avg loss 6.2pip -> avg later favorable 19.9pip); widen the stop or improve the entry first`
- n=2 `EUR_USD technical cache is stale at order time (M15:24m>20m, H1:84m>75m)`
- n=2 `EUR_USD planned TP 1.0pip is only 1.2x spread (0.8pip); the move is too thin to pay live friction`
- n=2 `C size cap is 1000u but the order asks for 4000u`

## Mandatory vNext Implications

- Broker-synced or manual/tagless exposure must block fresh entries until adopted or closed.
- Any live order path must compute JPY loss before send; risk above 500 JPY is not an execution detail.
- Reward/risk below 1.2x and targets/stops inside live spread friction are hard rejects.
- The importer intentionally excludes secrets, Python environments, Git internals, and large replay/tick caches from the source index; the legacy archive still contains them.
