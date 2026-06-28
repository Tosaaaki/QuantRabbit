# AI Test Bot Backtest Report

- Generated at UTC: `2026-06-17T07:47:36.451904+00:00`
- Status: `RESEARCH_PROFITABLE_NOT_CERTIFIED`
- Live permission: `False`
- History DB: `/Users/tossaki/App/QuantRabbit/data/legacy_history.db`
- Execution ledger DB: `/var/folders/64/3384w89n47v0hwlmw3ytr6j80000gn/T/qr-cycle-check-ww9ek2dm/execution_ledger.db`
- Source tables: `trades, pretrade_outcomes, seat_outcomes`
- Promotable source tables: `trades, pretrade_outcomes, seat_outcomes`
- Execution ledger selection: `not_requested`
- Opportunity dedupe: `True` (raw=`7896`, deduped=`6762`)
- Start balance: `174199 JPY`
- Target: `17420 JPY` (`10.0%`)
- Per-trade loss cap: `581 JPY` (`equity-derived 10% daily risk / 30 target trades per day`)
- Training days: `6`
- Min training trades: `10`
- Min training win rate: `55.0%`
- Max active buckets: `6`
- Context theme overlay: max=`1` min_trades=`20` min_win_rate=`65.0%`
- Validation days: `44`
- Traded days: `21`
- Target-hit days: `0`
- Total managed net: `40295 JPY`
- Profit factor: `2.7628`
- Max drawdown: `1795 JPY`

## Root Cause

- Best selected day: `9305 JPY`
- Average selected day: `916 JPY`
- Average selected trade: `191.0 JPY`
- Required trades/day at observed expectancy: `92`
- Oracle top-6 target-hit days: `0`
- Train-eligible oracle target-hit days: `0`
- Oracle all-positive target-hit days: `0`
- Selected/oracle capture: `52.5%`
- All-positive oracle ceiling: `68.5%`

## Target Band

- Contract floor: `5.0%`
- Stretch target: `10.0%`
- Selected-policy attainable: `5.0%`
- Selected-policy best return: `5.34%`
- Train-eligible oracle attainable: `5.0%`
- Train-eligible oracle best return: `5.34%`
- All-positive oracle attainable: `6.0%`
- All-positive oracle best return: `6.85%`
- Status: `SELECTED_POLICY_REACHES_FLOOR_BELOW_STRETCH`
- `5.0%` target=`8710` selected_hits=`1/44` top_n_oracle_hits=`2` train_eligible_hits=`1` all_positive_hits=`2` best_selected_coverage=`106.8%` required_trades_day=`46`
- `6.0%` target=`10452` selected_hits=`0/44` top_n_oracle_hits=`1` train_eligible_hits=`0` all_positive_hits=`1` best_selected_coverage=`89.0%` required_trades_day=`55`
- `7.0%` target=`12194` selected_hits=`0/44` top_n_oracle_hits=`0` train_eligible_hits=`0` all_positive_hits=`0` best_selected_coverage=`76.3%` required_trades_day=`64`
- `8.0%` target=`13936` selected_hits=`0/44` top_n_oracle_hits=`0` train_eligible_hits=`0` all_positive_hits=`0` best_selected_coverage=`66.8%` required_trades_day=`73`
- `9.0%` target=`15678` selected_hits=`0/44` top_n_oracle_hits=`0` train_eligible_hits=`0` all_positive_hits=`0` best_selected_coverage=`59.4%` required_trades_day=`83`
- `10.0%` target=`17420` selected_hits=`0/44` top_n_oracle_hits=`0` train_eligible_hits=`0` all_positive_hits=`0` best_selected_coverage=`53.4%` required_trades_day=`92`

## Mechanism Ablations

### RiskEngine Loss Cap

- Scope: `selected_validation_trades`
- Raw selected net: `30009 JPY`
- Managed selected net: `40295 JPY`
- Managed-minus-raw effect: `10286 JPY`
- Raw PF: `1.9054`
- Managed PF: `2.7628`
- Interpretation: `LOSS_CAP_HELPED_SELECTED_POLICY`

### Target-Aware Sizing Diagnostics

- Status: `FLOOR_ALREADY_HIT`
- Diagnostic only: `True`
- Best selected day: `9305 JPY` (`2026-04-13`)
- `5.0%` target=`8710` required_size_multiplier=`0.9361` scaled_loss_cap=`544` scaled_target_hits=`1` scaled_max_dd=`1680` status=`ALREADY_HIT`
- `6.0%` target=`10452` required_size_multiplier=`1.1233` scaled_loss_cap=`652` scaled_target_hits=`1` scaled_max_dd=`2016` status=`MODERATE_SIZE_UP_REQUIRED`
- `7.0%` target=`12194` required_size_multiplier=`1.3105` scaled_loss_cap=`761` scaled_target_hits=`1` scaled_max_dd=`2352` status=`MODERATE_SIZE_UP_REQUIRED`
- `8.0%` target=`13936` required_size_multiplier=`1.4977` scaled_loss_cap=`870` scaled_target_hits=`1` scaled_max_dd=`2688` status=`MODERATE_SIZE_UP_REQUIRED`
- `9.0%` target=`15678` required_size_multiplier=`1.6850` scaled_loss_cap=`978` scaled_target_hits=`1` scaled_max_dd=`3024` status=`MATERIAL_SIZE_UP_REQUIRED`
- `10.0%` target=`17420` required_size_multiplier=`1.8722` scaled_loss_cap=`1087` scaled_target_hits=`1` scaled_max_dd=`3360` status=`MATERIAL_SIZE_UP_REQUIRED`

### CLOSE Gate A/B Diagnostics

- Status: `MEASURED`
- Reason: `broker-truth execution ledger close diagnostics; not a live permission override`
- Close events: `213`
- Close net: `-19065 JPY`
- Bot-attributed close events: `207`
- Gateway close sent events: `0`
- Gateway close reconciled events: `85`
- Broker accepted TRADE_CLOSE events: `88`
- Broker accepted TRADE_CLOSE sources: `{"DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1, "TRADER_ENTRY_LANE_ID": 87}`
- Loss-side market closes: `74` net=`-86837 JPY`
- Gateway loss-side market closes: `73` net=`-81570 JPY`
- Gateway GPT_CLOSE loss-side market closes: `0` net=`0 JPY`
- Stale GPT_CLOSE satisfied loss-side market closes: `0` net=`0 JPY`
- Gateway REVIEW_EXIT loss-side market closes: `0` net=`0 JPY`
- Broker accepted loss-side market closes: `74` net=`-86837 JPY`
- Broker accepted without gateway close receipt: `1` net=`-5268 JPY`
- Broker accepted without gateway sources: `{"DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1}`
- Broker accepted without gateway evidence: `{"NO_CLIENT_EXTENSION": 1, "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 1}`
- No close-order provenance loss-side market closes: `0` net=`0 JPY`
- Take-profit closes: `93` net=`46875 JPY`
- Exit segments:
  - `TRADE_CLOSED:MARKET_ORDER_TRADE_CLOSE` count=`85` net=`-80222` bot_attributed=`84` gateway_close_sent=`0` gateway_close_reconciled=`84` broker_trade_close=`85`
  - `TRADE_CLOSED:MARKET_ORDER_MARGIN_CLOSEOUT` count=`4` net=`-5641` bot_attributed=`4` gateway_close_sent=`0` gateway_close_reconciled=`1` broker_trade_close=`1`
  - `TRADE_REDUCED:MARKET_ORDER_TRADE_CLOSE` count=`2` net=`-3792` bot_attributed=`2` gateway_close_sent=`0` gateway_close_reconciled=`2` broker_trade_close=`2`
  - `TRADE_CLOSED:STOP_LOSS_ORDER` count=`28` net=`991` bot_attributed=`24` gateway_close_sent=`0` gateway_close_reconciled=`0` broker_trade_close=`0`
  - `TRADE_CLOSED:MARKET_ORDER_POSITION_CLOSEOUT` count=`1` net=`22725` bot_attributed=`0` gateway_close_sent=`0` gateway_close_reconciled=`0` broker_trade_close=`0`
  - `TRADE_CLOSED:TAKE_PROFIT_ORDER` count=`93` net=`46875` bot_attributed=`93` gateway_close_sent=`0` gateway_close_reconciled=`0` broker_trade_close=`1`
- Close source segments:
  - `GATEWAY_RECONCILED:BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` count=`87` net=`-78902` pf=`0.035` expectancy=`-907` loss_market=`73`
  - `BROKER_ACCEPT:DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE` count=`1` net=`-5268` pf=`0.000` expectancy=`-5268` loss_market=`1`
  - `BROKER_ACCEPT:TRADER_ENTRY_LANE_ID` count=`1` net=`320` pf=`n/a` expectancy=`320` loss_market=`0`
  - `BROKER:STOP_LOSS_ORDER` count=`28` net=`991` pf=`1.467` expectancy=`35` loss_market=`0`
  - `BROKER:MARGIN_OR_POSITION_CLOSEOUT` count=`4` net=`17239` pf=`4.142` expectancy=`4310` loss_market=`0`
  - `BROKER:TAKE_PROFIT_ORDER` count=`92` net=`46555` pf=`n/a` expectancy=`506` loss_market=`0`
- Loss-side market close daily:
  - `2026-06-16` count=`1` net=`-662` gateway_close_sent=`0` gateway_close_reconciled=`1` broker_trade_close=`1` bot_attributed=`1`
  - `2026-06-15` count=`3` net=`-2189` gateway_close_sent=`0` gateway_close_reconciled=`3` broker_trade_close=`3` bot_attributed=`3`
  - `2026-06-12` count=`2` net=`-1661` gateway_close_sent=`0` gateway_close_reconciled=`2` broker_trade_close=`2` bot_attributed=`2`
  - `2026-06-11` count=`10` net=`-3489` gateway_close_sent=`0` gateway_close_reconciled=`10` broker_trade_close=`10` bot_attributed=`10`
  - `2026-06-08` count=`2` net=`-342` gateway_close_sent=`0` gateway_close_reconciled=`2` broker_trade_close=`2` bot_attributed=`2`
  - `2026-06-05` count=`4` net=`-4734` gateway_close_sent=`0` gateway_close_reconciled=`4` broker_trade_close=`4` bot_attributed=`4`
  - `2026-06-04` count=`5` net=`-8610` gateway_close_sent=`0` gateway_close_reconciled=`5` broker_trade_close=`5` bot_attributed=`5`
  - `2026-05-29` count=`2` net=`-8575` gateway_close_sent=`0` gateway_close_reconciled=`1` broker_trade_close=`2` bot_attributed=`1`
- Worst loss-side market close examples:
  - `2026-05-12T15:33:19.430010978Z` `GBP_USD SHORT` trade=`470854` pl=`-11987` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-05-12T15:33:20.090528634Z` `EUR_USD SHORT` trade=`470730` pl=`-8379` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-05-29T02:34:16.747617953Z` `EUR_USD LONG` trade=`471240` pl=`-5268` gateway_close=`False` gateway_reason=`` broker_trade_close=`True` bot_attributed=`False`
  - `2026-05-14T14:44:38.121219113Z` `GBP_USD SHORT` trade=`471008` pl=`-3689` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-05-29T02:18:24.867826983Z` `EUR_USD LONG` trade=`471232` pl=`-3307` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-05-13T01:23:50.028353009Z` `AUD_JPY LONG` trade=`470898` pl=`-3171` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-06-05T12:41:09.564409966Z` `GBP_USD LONG` trade=`472071` pl=`-2982` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`
  - `2026-05-08T07:23:10.799861871Z` `EUR_USD LONG` trade=`470427` pl=`-2891` gateway_close=`False` gateway_reason=`BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED` broker_trade_close=`True` bot_attributed=`True`

## Context Coverage

- Rows with context features: `6650`/`6762`
- Rows with context theme buckets: `515`
- `entry_type` rows=`157`
- `event_risk` rows=`28`
- `h1_trend` rows=`64`
- `m5_trend` rows=`29`
- `news_headlines` rows=`26`
- `regime` rows=`102`
- `session_hour` rows=`6648`
- `vix` rows=`1`

## Source Contributions

- `pretrade_outcomes` validation_net=`-348` selected_net=`0` validation_rows=`65` selected_rows=`0` deduped_rows=`65` days=`15` win_rate=`27.7%`
- `seat_outcomes` validation_net=`-3450` selected_net=`0` validation_rows=`47` selected_rows=`0` deduped_rows=`47` days=`10` win_rate=`27.7%`
- `trades` validation_net=`28351` selected_net=`40295` validation_rows=`2138` selected_rows=`211` deduped_rows=`6650` days=`43` win_rate=`39.1%`

## Blockers

- 10% target was missed on 44/44 validation days

## Action Items

- RiskEngine loss-cap ablation is currently positive (managed-minus-raw 10286 JPY); any cap removal/weakening candidate must beat the raw selected baseline, not just move faster
- 10% stretch cannot be treated as a small sizing tweak; selected best day needs 1.87x size, so coverage/reward geometry must improve before real-time target-chasing is credible
- external/direct broker TRADE_CLOSE orders are separated from CLOSE Gate A-B attribution (1 loss-side close(s), net -5268 JPY); sources={"DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1}; treat them as operator/broker-sync exit drag, not as autonomous gate evidence
- worst close-source segment is negative (GATEWAY_RECONCILED:BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED, count 87, net -78902 JPY, PF 0.0345); tune exits by measured close source before expanding discovery or news weighting
- exit split shows take-profit closes positive while loss-side market closes are negative; focus on CLOSE trigger attribution/timing rather than blanket TP removal
- seat_outcomes discovery universe is negative and selected no validation trades (net -3450 JPY across 47 receipts); repair discovery filters before increasing live frequency
- archive opportunity ceiling misses 10% target even with an all-positive oracle (gap 5486 JPY); expand verified opportunity universe/receipt coverage before more prediction tuning
- selected policy currently reaches 5% of the 5-10% band, not 10%; tune the next loop against 6% coverage while preserving the 5% floor
- observed expectancy requires about 92 selected trades/day versus 4.8; increase current LIVE_READY opportunity count or per-receipt reward geometry without raising loss caps
- legacy rows do not persist market-context-matrix/news/non-FX context as a dense pre-entry feature set; store matrix refs, news refs, and gold/oil/context-asset readings on each live entry receipt before treating non-FX/news prediction as backtest-certified

## Bucket Contributions

- `trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED` net=`16739` raw=`12972` trades=`48` days=`11` win_rate=`64.6%` worst=`-3436` best=`2376`
- `trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED` net=`7224` raw=`6393` trades=`52` days=`6` win_rate=`71.2%` worst=`-1200` best=`1256`
- `trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED` net=`5812` raw=`2932` trades=`18` days=`8` win_rate=`72.2%` worst=`-2583` best=`1110`
- `trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`4716` raw=`4441` trades=`8` days=`4` win_rate=`37.5%` worst=`-856` best=`4020`
- `trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`3516` raw=`3516` trades=`15` days=`7` win_rate=`73.3%` worst=`-384` best=`976`
- `trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED` net=`3144` raw=`1024` trades=`35` days=`6` win_rate=`54.3%` worst=`-1439` best=`2992`
- `trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`599` raw=`200` trades=`18` days=`5` win_rate=`55.6%` worst=`-876` best=`903`
- `trades:USD_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`131` raw=`131` trades=`5` days=`2` win_rate=`80.0%` worst=`-24` best=`88`
- `trades:AUD_USD:SHORT:UNSPECIFIED:UNSPECIFIED` net=`-768` raw=`-783` trades=`3` days=`2` win_rate=`0.0%` worst=`-595` best=`-6`
- `trades:EUR_JPY:SHORT:UNSPECIFIED:UNSPECIFIED` net=`-817` raw=`-817` trades=`9` days=`1` win_rate=`44.4%` worst=`-296` best=`77`

## Evidence Bucket Attributions

- `trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL` net=`22551` raw=`15904` trades=`66` days=`12` win_rate=`66.7%` worst=`-3436` best=`2376`
- `trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED` net=`16739` raw=`12972` trades=`48` days=`11` win_rate=`64.6%` worst=`-3436` best=`2376`
- `trades_theme:RISK_ON_JPY_CROSS:LONG:FX_RISK_THEME:ALL` net=`8831` raw=`8157` trades=`41` days=`8` win_rate=`58.5%` worst=`-876` best=`4020`
- `pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED` net=`8509` raw=`8509` trades=`19` days=`4` win_rate=`89.5%` worst=`-419` best=`1876`
- `trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED` net=`7224` raw=`6393` trades=`52` days=`6` win_rate=`71.2%` worst=`-1200` best=`1256`
- `pretrade_outcomes:EUR_USD:LONG:MEDIUM:UNSPECIFIED` net=`6832` raw=`6832` trades=`11` days=`5` win_rate=`54.5%` worst=`-459` best=`2376`
- `trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED` net=`5812` raw=`2932` trades=`18` days=`8` win_rate=`72.2%` worst=`-2583` best=`1110`
- `trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`4716` raw=`4441` trades=`8` days=`4` win_rate=`37.5%` worst=`-856` best=`4020`
- `trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED` net=`3516` raw=`3516` trades=`15` days=`7` win_rate=`73.3%` worst=`-384` best=`976`
- `trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED` net=`3144` raw=`1024` trades=`35` days=`6` win_rate=`54.3%` worst=`-1439` best=`2992`
- `trades_theme:USD_STRENGTH_AGAINST_RISK:SHORT:FX_RISK_THEME:ALL` net=`2376` raw=`241` trades=`38` days=`7` win_rate=`50.0%` worst=`-1439` best=`2992`
- `pretrade_outcomes:AUD_JPY:LONG:LOW:UNSPECIFIED` net=`1408` raw=`1408` trades=`3` days=`1` win_rate=`66.7%` worst=`-288` best=`1120`

## Missed Best Buckets

- `2026-04-07` selected=`3309` best=`trades:AUD_USD:LONG:UNSPECIFIED:UNSPECIFIED` best_net=`4886`
- `2026-04-06` selected=`459` best=`pretrade_outcomes:AUD_USD:LONG:MEDIUM:UNSPECIFIED` best_net=`3366`
- `2026-04-27` selected=`0` best=`pretrade_outcomes:EUR_USD:SHORT:HIGH:C` best_net=`2992`
- `2026-03-31` selected=`2229` best=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED` best_net=`2885`
- `2026-04-23` selected=`0` best=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED` best_net=`1681`
- `2026-04-10` selected=`1743` best=`trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED` best_net=`1474`
- `2026-04-17` selected=`1554` best=`trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED` best_net=`1408`
- `2026-03-25` selected=`424` best=`trades:GBP_JPY:SHORT:UNSPECIFIED:UNSPECIFIED` best_net=`1064`
- `2026-03-20` selected=`0` best=`trades:AUD_USD:SHORT:UNSPECIFIED:UNSPECIFIED` best_net=`1042`
- `2026-03-18` selected=`0` best=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED` best_net=`1007`
- `2026-04-29` selected=`726` best=`seat_outcomes:EUR_USD:SHORT:LIMIT:S_EXCAVATION` best_net=`990`
- `2026-03-24` selected=`0` best=`trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED` best_net=`972`

## Backtest Contract

- This is an offline research bot. It never places or stages broker orders.
- Bucket selection uses only prior training-window days; validation-day winners cannot select themselves.
- Buckets must also prove majority capped win rate in the training window before promotion.
- Seat outcome buckets use only observable setup/orderability/source fields, not future `CAPTURED/FAILED/MISSED` labels.
- Execution-ledger outcomes require attribution to a sent gateway entry; manual/tagless or otherwise unattributed closes are ignored.
- Execution-ledger buckets use gateway lane desk/strategy as pre-entry evidence; exit reason remains post-trade evidence and is not used as a selection bucket.
- Execution-ledger buckets require raw-positive training net, because hypothetical caps cannot certify old real-exit losses as fixed evidence.
- In mixed-source runtime backtests, execution-ledger rows stay diagnostic-only until actual exits prove a raw-positive promotable policy.
- Cross-pair context theme buckets are a strict one-bucket overlay selected only from pre-entry FX exposure, not from validation-day outcomes.
- Opportunity dedupe counts repeated seat receipts and cross-source same-trade-id rows as one candidate before training or validation scoring.
- Losses are capped by the equity-derived per-trade cap; wins are not enlarged.
- `live_permission=false` means this receipt can support research, not live execution.

## Validation Days

- `2026-03-09` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-10` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-11` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-12` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-13` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-15` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-16` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-17` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-18` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-19` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-03-20` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-22` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-23` net=`489` raw=`489` trades=`6` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-24` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:AUD_USD:SHORT:UNSPECIFIED:UNSPECIFIED, trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-25` net=`424` raw=`424` trades=`10` target_hit=`False` buckets=`trades:AUD_USD:SHORT:UNSPECIFIED:UNSPECIFIED, trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-26` net=`1989` raw=`1543` trades=`60` target_hit=`False` buckets=`trades:EUR_JPY:SHORT:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED, trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED, trades:USD_JPY:LONG:UNSPECIFIED:UNSPECIFIED`
- `2026-03-27` net=`500` raw=`274` trades=`6` target_hit=`False` buckets=`trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED, trades:AUD_USD:SHORT:UNSPECIFIED:UNSPECIFIED, trades:USD_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED`
- `2026-03-30` net=`-270` raw=`-270` trades=`5` target_hit=`False` buckets=`trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-03-31` net=`2229` raw=`1610` trades=`5` target_hit=`False` buckets=`trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-01` net=`4479` raw=`4479` trades=`8` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-02` net=`-205` raw=`-3061` trades=`4` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-03` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-06` net=`459` raw=`459` trades=`4` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-07` net=`3309` raw=`3309` trades=`7` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED`
- `2026-04-08` net=`4883` raw=`3357` trades=`7` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-09` net=`3919` raw=`3919` trades=`7` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-10` net=`1743` raw=`1481` trades=`8` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-13` net=`9305` raw=`9305` trades=`10` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-14` net=`3656` raw=`1083` trades=`16` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:AUD_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-15` net=`-793` raw=`-897` trades=`14` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-16` net=`-1001` raw=`-1001` trades=`12` target_hit=`False` buckets=`trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:EUROPE_FX_USD_WEAK:LONG:FX_RISK_THEME:ALL`
- `2026-04-17` net=`1554` raw=`1554` trades=`8` target_hit=`False` buckets=`trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades_theme:RISK_ON_JPY_CROSS:LONG:FX_RISK_THEME:ALL`
- `2026-04-18` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED`
- `2026-04-19` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED, trades:GBP_JPY:LONG:UNSPECIFIED:UNSPECIFIED`
- `2026-04-20` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-21` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-22` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-23` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-24` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-26` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-27` net=`0` raw=`0` trades=`0` target_hit=`False` buckets=`none`
- `2026-04-28` net=`2491` raw=`1417` trades=`5` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-29` net=`726` raw=`126` trades=`5` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
- `2026-04-30` net=`409` raw=`409` trades=`4` target_hit=`False` buckets=`trades:EUR_USD:SHORT:UNSPECIFIED:UNSPECIFIED`
