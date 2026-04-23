# Strategy Memory — プロトレーダーの経験知

**daily-reviewが毎日更新。traderが毎セッション冒頭で読む。**

最終更新: 2026-04-23 (daily-review for 2026-04-22 UTC)

---

## ⚡ READ THIS FIRST — Opportunity cost matters, but forced action kills expectancy too

- Bad B-trade avg loss: about -354 JPY. Recoverable in one trade.
- Drought day opportunity cost: about -12,569 JPY. Not recoverable by being "careful."
- 4/15-4/17 showed both edges of the same mistake: fear keeps the book flat, but forced late-session churn also kills expectancy.
- Correct standard when flat: name the best seat, exact trigger, and invalidation. Flat with a real trigger is allowed. Forced action without edge is not.
- 4/18 reminder: strongest-unheld `EUR_USD LONG` Edge A / Allocation B @1.1700 moved +63.8 pip without deployment. Missing the core edge costs more than scratching a bad probe.
- 4/17 reminder: negative expectancy came from repeating wrong-direction/high-score ideas, not from lack of opportunity. The cleanest opportunity was `AUD_JPY LONG`, and pretrade called it LOW.
- 4/17 S-hunt reminder: medium-term `EUR_JPY SHORT` was directionally right often enough to matter even though the book never deployed it. Missed capture is evidence, not zero.
- Benchmark deficit is not a setup. It may tighten the quality bar or force patience, but it never tells you which pair deserves the next unit of risk.
- 4/21 reminder: honest `dead thesis because no seat cleared promotion gate` closes beat fake promotion, but once the first `EUR_USD LONG` shelf squeeze has already stalled, repeating the same shelf-hold is idea recycling, not fresh S.
- 4/23 reminder: 4/22 was not a no-opportunity day. Filled `EUR_USD LONG` / `AUD_JPY LONG` / `GBP_JPY LONG` triggers lost, while undeployed `EUR_JPY SHORT`, `GBP_USD LONG`, and late `EUR_USD SHORT` windows kept paying. Separate pair edge from trigger choice.
- 4/23 reminder: once the runtime board prints a payable / armable lane, the handoff must resolve that same lane as a real receipt or an explicit dead thesis. Flat prose after a live board call is hidden inaction, not patience.
- 4/23 reminder: `dead thesis because no live pending entry order exists` is not a real contradiction. It only says execution never happened. If the lane was still armable, the bug is deployment, not the market.
- 4/23 reminder: unresolved action-board lanes are a `SESSION_END` problem, not an entry-preflight problem. If fresh risk is blocked before the order helper can place the resolving receipt, the runtime is self-sabotaging.
- 4/23 reminder: rebroadcasting giant static templates every cycle is runtime waste, not deeper analysis. Keep live board / tape / orderability foregrounded.
- 4/24 reminder: pending LIMIT management is discretionary. If the thesis is alive but the original price is now a wish, reprice only when the tighter entry still preserves spread-adjusted payout and noise-floor-safe SL/TP; otherwise leave or cancel.
- 4/24 reminder: same-pair opposite-side exposure must carry a role map. Hero / hedge-or-rotation / collapse condition must be written, or the weaker side should be closed or cancelled.
- 4/24 reminder: close notifications must be transaction-synced, not screen-memory synced. TP/SL fills can happen while the trader is away; `trade_event_sync.py --notify-slack` is the broker truth path and should batch multiple closes into one clean receipt.
- 4/24 reminder: handoff hygiene is execution capacity. `Gold Mine Inventory` must mirror the latest action board order, `## Positions (Current)` must parse to the live OANDA book, and bare logged fill / broker `STOP_FILL` receipts count as trader-owned fills. Do not let parser drift turn valid receipts into fake manual/no-log alarms.

---

## Confirmed Patterns（3回以上検証済み — ルールとして扱え）

### ✦ 勝ちパターン

- **Good day recipe**: one hero pair, repeated rotation on that pair, size escalation after proof, and session WR above 67%. 4/7 remains the model day.
- **Size is the #1 lever**: 1-2k units lose money in aggregate; 3-5k units make money; 9k+ only works when the story is truly S-conviction.
- **30-120min hold is where the edge lives**: <30min is noise, 2h+ becomes zombie inventory unless deliberately converted to swing.
- **Late NY is poison**: 21:00-00:00 UTC destroys expectancy. If the setup is real, it usually survives to Tokyo/London.
- **Momentum, not scalp speed, is the real edge**: the system makes money by reading the next 30m-2h directional move, not by micromanaging 5-minute noise.
- **S-conviction is narrative, not scanner-only**: scanner helps, but S emerges when macro, structure, and chart all tell the same story.
- [WATCH] **EUR_USD LONG is still the system's core directional edge**: when EUR is strong, exact H1/H4 support reloads remain the default high-quality seat, but 4/20 proved the edge pays from the precise first-defense / re-arm price, not from repeating the same shelf-hold after the first squeeze stalls.
- [WATCH] **AUD_JPY LONG in Tokyo band-walk conditions is real edge**: AUD strength + JPY offered + M5/H1 continuation produces clean 30-120min wins.
- **EUR_JPY LONG in strong EUR / weak JPY trend overrides LOW pretrade**: trending conditions on this pair systematically under-score.
- **Counter-S is now proven**: 5/6 correct. H4/H1 extreme plus chart agreement is entry-worthy, not just interesting.
- [WATCH] **LIMIT -> TP auto-fire is the best passive pattern**: structural LIMITs with realistic ATR-based TP keep making money without emotional churn; 4/21 `EUR_USD SHORT` retest LIMIT -> TP +315 JPY and 4/22 late `EUR_USD SHORT` retrace LIMIT -> TP +270 JPY reinforced it, while spread-heavy `GBP_JPY LONG` breakout stops kept proving the worse vehicle.
- **Momentum-S multi-pair alignment must be traded**: when 3+ pairs fire the same directional regime, enter the hero pair with real size.
- **Rotation SHORT inside a LONG thesis is standard pullback trading**: if M5 flips while H1 thesis remains alive, short the pullback instead of only tightening TP on the long.

### ✦ 負けパターン

- **3-loss circuit breaker**: after 3 losses in the same direction, either flip or stop. Continuing is bot behavior.
- **Margin stacking without worst-case pending calc causes repeated forced damage**: above 85% no new entries; above 90% immediate relief.
- [WATCH] **Late-NY entries and orphan holds are the system's ugliest leak**: wrong session plus no explicit swing decision. 4/19 `GBP_JPY` zombie carry -654 JPY proved the leak again.
- [WATCH] **Repeated same-pair entries after failure compound losses**: 4/20 `EUR_USD LONG` shelf-holds, 4/21 `AUD_JPY SHORT` three-shot re-fades (-723 JPY), and 4/22 `AUD_JPY LONG` reclaim re-arms all turned edge memory into churn once the first trigger failed.
- [WATCH] **Short-term S receipt farming before an event is action bias, not edge.** On 4/17 the short-term board named 11 ideas, armed 9, entered 5, and still hit only 18% directional accuracy.
- **HIGH score does not mean regime-aligned**: a strong score on the wrong side of the tape is still wrong.
- [WATCH] **AUD_USD remains a pair with no proven edge**: audit pressure or high scores do not turn it into an S-size seat.
- **Tokyo thin / pre-event tight trails are traps**: trail widths below local noise get clipped before the thesis works.
- **Shallow analysis creates one-direction blindness**: if data is good enough to tighten TP, it is good enough to consider the other direction.
- [WATCH] **One-limit-and-done is fake participation**: a live move needs `NOW + RELOAD + SECOND SHOT`, not one distant pending order; 4/18 `EUR_USD LONG` A/B staying unarmed was the same failure in cleaner clothes.
- **Churn burns spread and conviction**: close -> re-enter same thesis without new information is self-inflicted damage.

---

## Active Observations（検証中 — daily-reviewが追記する）

<!-- 各観察: [初出日] 内容。検証: N回確認/N回反証 -->

- [WATCH] [4/22] **When the same audit strongest-unheld seat repeats 3x+ in a few hours and the seat also shows fresh missed-S excursion, the next trader session must arm the honest order or write the exact contradiction.** Rewriting the same opportunity as new prose is deployment failure, not patience. Verified: 1x
- [WATCH] [4/23] **Do not compress live opportunity into one `(pair, direction)` story.** The same pair can hold separate scalp / pullback / swing seats when the trigger, vehicle, or invalidation differs; runtime inventory must preserve those identities instead of forcing one representative seat. Verified: 1x
- [WATCH] [4/23] **Audit narrative seats must survive as seat objects, not pair-direction summaries.** If audit writes two `EUR_USD SHORT` seats with different entries/vehicles, runtime must carry both seat families; `strongest-unheld` should strengthen the best matching seat, not spawn a ghost duplicate. Verified: 1x
- [WATCH] [4/23] **Profitable `LIMIT` vehicles must not be buried by the pair's blended churn record.** `AUD_JPY SHORT` has recent positive `LIMIT` expectancy while its `MARKET` / `STOP-ENTRY` variants are still losing; runtime scoring should judge that seat by the profitable vehicle before pair-direction aggregate suppresses it. Verified: 1x
- [WATCH] [4/23] **Board orderability must come from the same exact pretrade engine that blocks live sends.** If `session_data.py` closes a seat with one brain and `place_trader_order.py` re-judges it with another, underdeployment is structural. Verified: 1x
- [WATCH] [4/23] **Exact pretrade must separate hard blockers from soft vehicle preference.** Geometry floors, live-book drift, and unpaid-unprotected same-pair stacking are real vetoes; `LIMIT vs STOP-ENTRY vs MARKET` preference is trader judgment unless one of those hard blockers exists. Verified: 1x
- [WATCH] [4/23] **The runtime board contract should speak in trader language, not bot language.** Live seats should carry `default_expression / default_orderability`; legacy `execution_style / orderability` may exist only as compatibility aliases for logs and validators, never as the primary source of meaning. Verified: 1x
- [WATCH] [4/23] **Prior board echoes are not fresh seats.** `Podium #...`, `Lane ...`, and `id=board` lines from `state.md` are carry-forward closure records, not new opportunity inventory; re-injecting them into the next session hides live audit/pending/scanner seats behind stale self-copy. Verified: 1x
- [WATCH] [4/23] **Focus-ladder siblings are one carry seat, not three permissions.** `Best expression NOW`, `Primary vehicle`, and `Next fresh risk allowed NOW` can all point at the same live idea; runtime must collapse them into one `state-focus` seat instead of letting the same EUR/USD idea occupy several state lanes. Verified: 1x
- [WATCH] [4/23] **Cross-source agreement is corroboration, not extra seats.** If audit, state focus, and scanner all point at the same live `EUR_USD SHORT MARKET` idea, runtime should keep one seat with corroboration instead of filling the board with three copies of the same trade. Held/pending receipts still stay separate from fresh deployment inventory. Verified: 1x
- [WATCH] [4/23] **The prompt must force a gold inventory, not just a hero seat.** One `Primary vehicle` plus one `Best A/S one print away` is too easy to satisfy with elegant hesitation. In broad tape, the trader should name at least the top 5 executable mines and close each as a receipt or an exact contradiction. Verified: 1x
- [WATCH] [4/23] **Quiet-stable repeat-pressure seats should scout, not keep rewriting the same missed move as trigger prose.** When a seat keeps repeating, missed-S pressure is real, and tape is calm/stable rather than opposed, a small market scout is often cleaner than another no-fill rewrite even if the seat is only B-grade. Verified: 1x
- [WATCH] [4/23] **`has_protection` is not the same as risk reduction.** A same-pair unpaid reload stays blocked if the existing leg has no stop-side protection at or through the entry; TP-only or untouched SL does not justify stacking another market fill. Verified: 1x
- [WATCH] [4/23] **`Inventory lead` from quality-audit is still a live seat when no A/S exists.** If the auditor says `No unheld A/S opportunities` but names an `Inventory lead`, the trader must keep that B-seat in runtime inventory instead of dropping it because there was no hero pick. Verified: 1x
- [WATCH] [4/23] **Send-time exact pretrade must share runtime context, not only runtime geometry.** If live send ignores the current regime, spread, or tape summary that the board used, the same `entry / tp / sl` can still get re-judged under a different market state. Verified: 1x
- [WATCH] [4/22] **If review says a loser was `trigger` or `vehicle` damage while market/structure later recovered, do not kill the direction.** Keep the family alive, but the next seat must change the exact trigger or execution vehicle; copy-pasting the same entry is still blocked. Verified: 1x
- [WATCH] [4/23] **If the board prints multiple non-PASS lanes, one armed receipt is underdeployment unless lane two is explicitly dead.** `trigger-only watch lane` is fake participation; a surviving lane must close as armed `STOP-ENTRY` / `LIMIT` with `id=___` or `dead thesis because ___`. Verified: 1x
- [WATCH] [4/23] **Do not rewrite an already-armed stop as a fake limit in the runtime board.** The board must echo the real live receipt type, and if no market lane is payable it must say why explicitly instead of making the whole tape look passive. Verified: 1x
- [WATCH] [4/23] **Calm microstructure is not automatically bad microstructure.** A short probe that is quiet and spread-stable should be treated as `quiet / stable`, not as automatic friction poison; only unstable spread, true two-way churn, or opposite pressure should kill a market lane by themselves. Verified: 1x
- [WATCH] [4/23] **Fresh audit range boxes must become real `AUDIT_RANGE` LIMIT lanes, not just nice prose in `quality_audit.md`.** If the auditor prints exact buy/sell rails with paid opposite-band TP, the next trader session should surface that box in the fresh-risk board and either arm it or write the exact chart contradiction. Verified: 1x
- [WATCH] [4/23] **Exact-thesis losses need a cool-off window, not a full-day burial.** If the same exact seat just lost, stand down for a few hours and require a materially new print. After that, do not keep the seat dead by date label alone. Verified: 1x
- [WATCH] [4/23] **Recent stop-loss regret is the real noise floor.** If the pair keeps recovering after recent 5-10 pip stop-outs, the next SL must sit outside that historical loss band or the entry must improve. `4x spread` alone is not enough. Verified: 1x
- [WATCH] [4/23] **For `LIMIT` / `STOP-ENTRY`, “trigger not printed yet” is an arm reason, not a death reason.** If the seat is still valid and the only blocker is `has not broken yet` / `needs price improvement`, arm the pending order or write the real contradiction. Also repair the first `SL/TP` draft once against the pair's noise/friction floors before passing on the seat. Verified: 1x
- [WATCH] [4/24] **`REPRICE` belongs in pending LIMIT review when the narrative is alive but the old price is now too far.** It must name the old id, tighter entry, TP, SL, and why the reduced edge still pays after spread and recent noise. Small 2-3 pip tweaks are churn unless they change fill probability without breaking reward/risk. Verified: 1x
- [WATCH] [4/24] **Same-pair opposite-side inventory is a range / hedge structure only if the role map is explicit.** If USD_JPY has both long and short exposure, the handoff must say which side is the hero, which side is hedge or rotation, and the condition that collapses the pair back to one side. Verified: 1x
- [WATCH] [4/24] **Broker-side TP/SL closes are invisible unless the transaction stream is synced.** Slack close receipts should come from OANDA transactions, not from whatever helper happened to initiate the close, and multiple fills should post as one compact close batch. Verified: 1x
- [WATCH] [4/24] **Gold Mine and live-book parser drift can silently disable the trader even when the trade idea is right.** If Gold #1-#5 no longer matches the action board, or quality-audit cannot parse the live `## Positions (Current)` prose, the next session must repair the handoff before judging market quality. Verified: 1x
- [WATCH] [4/23] **EUR_USD can pay as a live direct-USD short even while the paired EUR_JPY cross is still only a managed hold.** The cleaner vehicle can rotate to market first, with the reload limit left working behind it. Verified: 1x
- [WATCH] [4/22] **Today’s losing closes recovered 7/10 within 6h, so the real drag was trigger / vehicle choice, not “no opportunity existed.”** Keep demoting same-day losing `MARKET` / `STOP-ENTRY` lanes and stop confusing repeated proof-chase with active trading. Verified: 1x
- [WATCH] [4/22] **Since the 2026-04-17 discretionary reset, passive `LIMIT` receipts are paying better than fresh `MARKET` / `STOP-ENTRY` proof-chase on the same broad ideas.** When the pair edge is real but the trigger vehicle is still dragging, keep breadth with a better-price limit instead of forcing the reclaim/break print. Verified: 1x
- [WATCH] [4/23] **4/22 `EUR_USD LONG` was only defensible at the exact first-defense shelf, and even there it only scratched.** Three longs still netted -378 JPY once the old shelf was damaged, so repeating the same long family after the first failure was idea recycling, not core-edge execution. Verified: 1x
- [WATCH] [4/23] **4/22 `EUR_JPY SHORT` kept printing correct S-hunt / excavation windows with 47-60 pip favorable excursion and zero deployment.** Spread-heavy PASS is not proof the direction was wrong; it is proof the seat needed a deeper limit or a better-timed retest. Verified: 1x
- [WATCH] [4/23] **4/22 `GBP_JPY LONG` entered the wrong vehicle.** The breakout stop lost -351 JPY, while later `GBP_JPY LONG` and `GBP_USD LONG` misses were often right without deployment. Direction quality survived; spread-heavy trigger quality did not. Verified: 1x
- [WATCH] [4/23] **`S Excavation` handled late `EUR_USD SHORT` correctly: the blocker was the missing rejection print, not the direction.** Once that retest actually printed, the later LIMIT short captured +270 JPY. Verified: 1x
- [WATCH] [4/22] **When `EUR_JPY`, `GBP_JPY`, and `AUD_JPY` all flip from upper-half repair shelves into the same-cycle red-body unwind and the M1 right edge keeps walking lower with no reclaim candle, the honest next cross seat becomes retest-short only.** Buying the first lower rail in that state is just catching traveled unwind, not a fresh defense. Verified: 1x
- [WATCH] [4/22] **When `GBP_USD` accepts above the old lid while `GBP_JPY` is staircasing at the same time, the honest cable expression rotates from stale lid-fade short into hold/long-shelf until 1.3527 actually fails again.** Verified: 1x
- [WATCH] [4/22] **`AUD_JPY LONG` reclaim-stops can be valid live tests and still deserve a fast cut when the first 114.08/10 acceptance fails inside the 15-minute confirmation window.** Base-hold alone was not enough once the first expansion died back into 114.00. Verified: 1x
- [WATCH] [4/22] **When a first-retest `EUR_USD SHORT` fills but M1 immediately expands into a broad USD-offer squeeze and stops it within minutes, that short thesis is dead for the cadence; do not rewrite the same short, wait for first-defense long or a brand-new lower shelf.** Verified: 1x
- [WATCH] [4/21] **`AUD_JPY SHORT` kept positive EV only as Tokyo upper-box range rotation with structural risk.** The 113.700 LIMIT -> 113.580 TP paid +600 JPY after widening the stop to 113.800, but the London market scout + reload after lid acceptance lost -624 JPY. Same pair, different vehicle. Verified: 1x
- [WATCH] [4/21] **Narrative strongest-unheld `GBP_USD LONG` was right multiple times on 4/20 UTC and still went untraded.** `1.35386`, `1.35384`, and `1.35326` all pointed the right way; the miss was reclaim-body timing, not direction. Verified: 1x
- [WATCH] [4/21] **`EUR_JPY LONG` remained a correct pair edge, but not from the first late pullback LIMIT.** The LOW(0) long lost -216 JPY because it bought a stall, while later excavation / narrative longs kept being right without deployment. Entry quality, not pair direction, was the gap. Verified: 1x
- [4/20] **A horizon whose blocker is still alive is excavation, not promoted S.** If the same blocker sentence still explains the seat, keep it in the podium and close the horizon as `dead thesis because no seat cleared promotion gate` instead of pretending S was found. Verified: 1x
- [4/20] **Missed-seat scoring must use best favorable excursion, not the flat close alone.** A seat that traveled 10+ pip then drifted back is still missed edge; UTC close/current alone understates the process loss. Verified: 1x
- [4/20] **4/19 S-hunt proved the read beat the deployment on longer horizons.** `EUR_USD LONG` short-term S had a real LIMIT id=469072 and was directionally right/open, while medium-term `AUD_JPY SHORT` (+10.4 pip BFE) and long-term `GBP_JPY SHORT` (+4.2 pip BFE) died at the promotion gate despite being directionally correct. Verified: 1x
- [WATCH] [4/18] **Audit strongest-unheld `EUR_USD LONG` Edge A / Allocation B @1.1700 was the day's cleanest miss.** NOT held, then moved +63.8 pip. Core edge was right; deployment was absent. Verified: 1x
- [WATCH] [4/16] **LIMIT -> TP auto-fire keeps outperforming screen-time trading.** Preserve this pattern instead of replacing it with chase entries. Verified: 3x
- [CONFIRMED] [4/16] **Counter-S on EUR_JPY keeps beating LOW pretrade.** The recipe is now proven enough to act on with conviction. Verified: 3x

---

## Deprecated（反証済み — 参考のみ）

_(4/23 daily-review for 2026-04-22 UTC: no new deprecations, but 4/22 evidence kept `GBP_JPY LONG` breakout-stop and repeated `AUD_JPY LONG` reclaim-stop behavior in watch-only territory instead of promoting either trigger family.)_

---

## Per-Pair Learnings（ペア別）

### USD_JPY

- [WATCH] Counter-S and ceiling-fade setups still work only when the H4 extreme is real. 4/17 SHORT HIGH(11) lost -67 JPY because it was traded like a strong call inside an already messy session.
- [WATCH] Keep stops wider than spread x3 and respect intervention risk, but 4/22 added the cleaner rule: the 159.320 `USD_JPY SHORT` reclaim-fail probe lost -45 JPY, while many later repaired-box long podium seats were right without deployment. Keep this pair tactical: ceiling fades need exact retest failure, and repaired-box longs need a real lower-rail defense instead of old short memory.

### GBP_USD

- [WATCH] LONG remains the cleaner structural edge; 4/20 strongest-unheld longs were right, and 4/21 late `GBP_USD LONG` repair calls kept being the cleaner untraded payer while the earlier shorts were only tactical, not durable pair-edge proof.
- [WATCH] 4/22 kept scoring `GBP_USD LONG` medium-term windows as CORRECT without deployment. The direction is still real; the missing piece is an honest first-defense trigger, not a new pair thesis.
- [WATCH] 4/17 repeated HIGH(9) shorts for -420 JPY confirmed the hard rule: after 2 cable losses in the same direction, switch pair or stop.
- [WATCH] 4/18 medium-term `GBP_USD SHORT` stayed `STILL PASS` twice and still finished directionally wrong. Cable short remains explanation-rich but low-orderability.
- [WATCH] If a GDP-fade or bearish continuation thesis is real, the first clean failure-break matters. Selling lower again and again after the first failure is just chasing.

### EUR_USD

- [WATCH] EUR_USD LONG is still the best core edge, but 4/17 proved that structural floor alone is not enough inside unresolved M5 squeeze. Need reclaim or cleaner London handoff.
- [WATCH] 4/17 churn around the 1.178 area was expensive: two MEDIUM longs and one HIGH short all lost. Do not flip both directions inside the same unresolved box.
- [WATCH] 4/17 recurring-trader `EUR_USD LONG` -24 JPY was the correct scratch, not a failed core edge. Treat floor LIMIT fills as probes until the reclaim actually prints.
- [WATCH] 4/18 audit strongest-unheld `EUR_USD LONG` Edge A/B @1.1700 went +63.8 pip without deployment. When H4 support and H1 lower-band exhaustion align, leave a real stop-entry / reopen plan instead of prose-only conviction.
- [WATCH] 4/19 short-term S `EUR_USD LONG` had the right operational shape, but 4/21 proved the short side only works as proof-first retest failure: the 1.17655 LIMIT -> TP paid +315 JPY, while event-sensitive or recycled shorts still lost. Treat `EUR_USD SHORT` as tactical vehicle edge, not a broad regime flip.
- [WATCH] 4/22 split the pair cleanly: three longs netted -378 JPY because the old shelf was already damaged, but the later 1.17325 LIMIT short captured +270 JPY. LONG stays core only from rebuilt first-defense; SHORT stays tactical only as retest-failure / lower-shelf rejection.
- [WATCH] Tokyo entry -> London close remains the best expression when the structure is actually clean.

### AUD_USD

- [WATCH] Pair remains no-edge until proven otherwise. 4/21 HIGH(8) `AUD_USD LONG` still lost -85 JPY in 16 minutes, so even cleaner trigger form did not rescue the pair.
- [WATCH] Hard cap remains B-size. No audit miss, score, or scanner note upgrades AUD_USD into S-size.
- [WATCH] Use only when AUD and USD are decisively separated by macro + chart; otherwise pass.

### AUD_JPY

- [WATCH] 4/17 `AUD_JPY LONG` +576 JPY in 137min confirmed the pair's Tokyo continuation edge again.
- When AUD is strongest, JPY is offered, and M5 is band-walking, pretrade LOW is too conservative. Upgrade that condition to tradable B/MEDIUM by default.
- Recurring-trader clean split on 4/17 was asymmetric: Tokyo band-walk LONG +1,120 JPY supported the prior, but pre-Waller upper-box SHORT -210 JPY did not. Keep trusting continuation; treat event-hour fades as watch-only until more proof.
- 4/20 `AUD_JPY SHORT` only paid as Tokyo upper-box range rotation, and 4/21 made the warning harsher: three separate short receipts lost -723 JPY once the first rejection failed. Keep shorts to exact Tokyo LIMIT rotation only and block the second/third fade after acceptance.
- [WATCH] 4/20 `AUD_JPY LONG` NY continuation stop at 114.060 failed -240 JPY. Tokyo band-walk and first-defense still work, but late breakout continuation after the reclaim is watch-only until it wins cleanly.
- [WATCH] 4/22 Tokyo reclaim-stops went 0/2 for -325 JPY. After the first 114.08/10 acceptance fails, the next reclaim is no longer the same edge; wait for a fresh base or leave the pair alone.
- [WATCH] Avoid old churn pattern: first clean long is the money; repeated re-entry after exit burns the edge.

### GBP_JPY

- 4/17 audit strongest-unheld `GBP_JPY LONG` was correct before the later chase. When H4 is early-wave and the audit grades it A/A, waiting for a worse price is opportunity-cost behavior.
- [WATCH] 4/18 `GBP_JPY SHORT` was only valid as an already-triggered late breakdown. After the 214.388 break, fresh chase quality was gone; management mattered more than adding.
- [WATCH] 4/19 reviewed-day loser `GBP_JPY` -654 JPY was zombie inventory, not fresh edge. Once the live M1/M5 ladder flips against the old idea, cut it and re-hunt; do not let older `GBP_JPY LONG` strength or bear-memory keep the seat alive by inertia.
- [WATCH] 4/22 shelf-break STOP at 215.354 lost -351 JPY even though many later `GBP_JPY LONG` windows were directionally right. The long edge survives, but breakout-stop entry at ~3pip spread is worse than honest shallow reloads or exact first-defense limits.
- Long edge remains strong, but 4/21 LOW(3) late-NY shelf rebuy lost -144 JPY in 6 minutes. Tokyo-thin / late-NY LIMIT fills still need wide structure-aware risk and do not justify spread-heavy shelf rebuys once the first defense is already late.

### EUR_JPY

- Trend long remains one of the best live expressions when EUR is strongest and JPY is weak.
- [WATCH] Spread economics still matter: sub-10pip ideas do not belong here, but 20+ pip London/Tokyo continuation does.
- [WATCH] Counter-S at H4/H1 floor remains valid; do not let LOW pretrade suppress the entry when chart alignment is clear.
- [WATCH] 4/17 S-hunt showed a medium-term `EUR_JPY SHORT` capture miss (4/11 directional accuracy, zero deployment). That is process evidence about horizon coverage, not enough to rewrite the pair's confirmed LONG edge.
- [WATCH] 4/20 LOW(0) `EUR_JPY LONG` lost because the first pullback bought a stall, and 4/21 split the lesson cleanly: the early breakout/zombie hold lost -225 JPY, but the later 187.060 quiet-base LIMIT paid +483 JPY. The edge stays long, but only from rebuilt base / first-defense, not stale breakout memory.
- [WATCH] 4/22 early `EUR_JPY SHORT` S-hunt / excavation windows were repeatedly correct by 47-60 pip favorable excursion without deployment. Keep the confirmed LONG edge, but range upper-half shorts are still real tactical payers when H1 is not trend-bull and the limit can actually beat spread friction.

---

## Pretrade Feedback（daily-reviewが自動更新）

### LOWが正しかった場面（避けるべきエントリー）

- [DEPRECATED] [3/27] AUD_JPY/AUD_USD LOW -> total -1,804 JPY. AUD pairs at LOW were correct warnings in that regime.
- [4/20-4/21] `USD_JPY LONG` LOW(2) -> -270 JPY, `EUR_JPY LONG` LOW(0/1) -> -216 / -225 JPY, `AUD_JPY LONG` LOW(3) -> -132 JPY, and `GBP_JPY LONG` LOW(3) -> -144 JPY. LOW was correct when the seat was late, spread-heavy, or still stale breakout memory.
- [4/22] `GBP_JPY LONG` pretrade=LOW(3) -> -351 JPY. Breakout proof did not overcome 3.2pip spread and late shelf quality.

### LOWが間違っていた場面（pretradeが保守的すぎるケース）

- [4/17] `AUD_JPY LONG` pretrade=LOW(3) -> +576 JPY. Tokyo band-walk with AUD strongest / JPY offered overrode the historical caution.
- [4/17] `AUD_JPY LONG` pretrade=LOW(3) -> +1,120 JPY. Same Tokyo continuation regime, bigger size, same lesson: LOW was too conservative when AUD was strongest and JPY was offered.
- [4/7+4/9+4/16] `EUR_JPY LONG` trending/Counter-S override: 9/10 wins, +2,724 JPY, almost all pretrade=LOW.
- [4/21] `EUR_JPY LONG` pretrade=LOW(1) -> +483 JPY. LOW was too conservative once the 187.00/06 base had already rebuilt and the entry used quiet price improvement instead of paid breakout memory.

### HIGHが外れた場面（スコア過信に注意）

- [4/17] `GBP_USD SHORT` pretrade=HIGH(9) -> 3 losses, total -420 JPY. GDP-fade story stayed in the head after the live downside expansion had already weakened.
- [4/17] `AUD_USD SHORT` pretrade=HIGH(9) -> -96 JPY and `AUD_USD LONG` pretrade=HIGH(8) -> -14 JPY. HIGH score on a no-edge pair is still bad business.
- [4/17] `EUR_USD SHORT` pretrade=HIGH(11) -> -67 JPY. Score overstated the fade while price was still trapped in a noisy squeeze near structural support.
- [4/17] `USD_JPY SHORT` pretrade=HIGH(10) -> -96 JPY. Ceiling score did not fix the fact that the live entry was already sell-low in a messy book.
- [4/7+4/9 pattern] HIGH SHORTs in bullish regime already went 5/5 wrong before today. Today's losses confirm the score is direction-agnostic and must be regime-filtered by the trader.
- [4/21] `AUD_USD LONG` pretrade=HIGH(8) -> -85 JPY and one `EUR_USD SHORT` HIGH override / PASS-breach trade -> -124 JPY. HIGH still inflated thin event or no-edge executions even when the story sounded clean.
- [4/22] `USD_JPY SHORT` pretrade=HIGH(8) -> -45 JPY, `EUR_USD SHORT` pretrade=HIGH(7) -> -172 JPY, and `EUR_USD LONG` pretrade=HIGH(7) -> -85 JPY. HIGH kept flattering stale or overpaid triggers; the actual winners came from proof-first LIMIT seats.

### MEDIUM / B scores — recent

- [4/17] `EUR_USD LONG` pretrade=MEDIUM(5) -> losses of -345 JPY and -32 JPY. Structural floor plus squeeze is not enough; require reclaim, acceptance, or London handoff.
- [4/17] `GBP_USD LONG` pretrade=MEDIUM(6) -> wins of +62 JPY and +91 JPY. Medium works when direction is actually aligned with the live turn.
- [4/16] `GBP_USD LONG` pretrade=MEDIUM(4) -> +513 JPY. LIMIT -> TP still beats forced market chasing.
- [4/16] `AUD_JPY LONG` pretrade=MEDIUM(7) -> -332 JPY. H4 overbought warning was valid; in that state prefer deeper pullback LIMIT over immediate market entry.
- [4/21] `AUD_JPY SHORT` pretrade=MEDIUM(4) went 0/3 for -723 JPY. Same score plus the same fade family did not mean three fresh ideas; once the first confirmation failed, the second and third receipts should have been blocked.
- [4/20] `EUR_USD LONG` pretrade=MEDIUM(5) was profitable only at the exact first-defense / reload price. Once the reclaim stalled into squeeze, more buys were churn, not edge.
- [4/19] `GBP_JPY LONG` pretrade=MEDIUM(4) -> -654 JPY in the review report. The loss came from stale 51h hold quality and zombie management, not from proof that MEDIUM should be promoted on `GBP_JPY`.
- [4/22] `AUD_JPY LONG` pretrade=MEDIUM(6) went 0/2 for -325 JPY. Same reclaim family did not become fresh proof just because it rearmed.
- [4/18-4/20] If the seat is real but the live print is wrong, frequency must come from armed `STOP-ENTRY` / structural `LIMIT`, not from forcing a low-quality market fill or leaving a live S as prose-only.
- [4/20] `headwind` is first a size cap, not an automatic flat-book command. If `trending / squeeze / transition` is already paying and spread is still normal, a real seat can still close as `MARKET` scout or `STOP-ENTRY`; passing everything there is opportunity-loss behavior. Verified: 1x
- [4/20] 20-minute cadence punishes visual-only backup lanes. If Tokyo breadth is broad and a backup / rotation seat is still valid, carry it as an armed `STOP-ENTRY` or `LIMIT`; otherwise the next session just rediscovers the same move too late. Verified: 1x
- [4/18] Broad sessions are not one-pair sessions. When Currency Pulse and scanner breadth point the same way across different currencies, keep `PRIMARY / BACKUP / THIRD CURRENCY` lanes alive instead of collapsing everything into one "best seat".

---

## 指標組み合わせの学び

### 効いた組み合わせ

| 日付 | ペア | 組み合わせ | TF | 結果 |
|------|------|-----------|-----|------|
| 3/23 | GBP_USD | M5 StochRSI=0.0 + H1 ADX>25 BULL | M5+H1 | +653円 |
| 4/20 | AUD_JPY | Tokyo upper-box LIMIT + H1 BB-mid structural stop + attached TP | M5+H1 | +600JPY |
| 4/7 | EUR_JPY | H1 ADX>35 + CS alignment + trend continuation | H1+M5 | 6/6 +1,876JPY |
| 4/16 | EUR_JPY | Counter-S at H4/H1 floor + M5 confirmation | H4+H1+M5 | +420JPY |
| 4/17 | AUD_JPY | AUD strongest + JPY offered + M5 band-walk with no retrace | M5+H1 | +576JPY |
| 4/21 | EUR_USD | broken-shelf retest LIMIT + pre-armed TP | M5+H1 | +315JPY |
| 4/22 | EUR_USD | failed-shelf retest LIMIT + attached TP | M5+H1 | +270JPY |

### 効かなかった組み合わせ

| 日付 | ペア | 組み合わせ | 結果 | なぜ |
|------|------|-----------|------|------|
| 4/8 | AUD_JPY | H4/H1 extreme counter vs ADX>35 trend | 0/2 -200JPY | oscillator extreme did not override trend |
| 4/20 | EUR_USD | exact shelf was good, repeated H1-against shelf-hold re-buys were not | -127JPY net | the first-defense edge stayed real, but repeating the same squeeze thesis after it stopped paying was idea recycling |
| 4/17 | GBP_USD | GDP-fade story + repeated sell-lower execution | 3 losses -420JPY | narrative was recycled after edge decayed |
| 4/17 | EUR_USD | H4 floor + M5 squeeze without reclaim | -377JPY | floor was early, box never resolved cleanly |
| 4/17 | AUD_JPY | pre-Waller upper-box Structural-S fade | -210JPY | pair EV was positive, but event timing and repeated short-horizon fades were worse than the Tokyo continuation edge |
| 4/18 | AUD_JPY | late-NY upper-box fade into weekend close | -210JPY | the pair direction could still be right later, but the actual seat was too late and too thin to own |
| 4/22 | GBP_JPY | shelf-break STOP through 3.2pip spread | -351JPY | breakout proof did not overcome spread-heavy trigger quality |
| 4/19 | GBP_JPY | late breakdown carry held 51h after the seat expired | -654JPY | zombie hold outlived the live M1/M5 structure and turned old thesis memory into dead inventory |

### 未検証（今後試す）

- Trend-Dip-S must be separated into unidirectional vs contradictory-flip cases.
- Narrative strongest-unheld A/A picks should be tracked separately from recipe-based scans.
- Repeated `GBP_USD LONG` strongest-unheld continuation calls need explicit stop-entry / first-defense tracking so correct auditor reads stop disappearing into prose.

---

## S-Scan Recipe Scorecard (daily-review tracks)

| Recipe | Fires (all-time) | Entered | Correct (±5pip) | Accuracy | Status |
|--------|-------------------|---------|------------------|----------|--------|
| Structural-S | 23 | 1 | 12/23 | 52% | Watch / degraded but not dead |
| Momentum-S | 55+ | 0 | ~53/55 | ~96% | Confirmed |
| Trend-Dip-S | 18 | 4 | 7/18 | 39% | Watch |
| Squeeze-S | 9 | 1 | 1/9 | 11% | Deprecated |
| Counter-S | 6 | 2 | 5/6 | 83% | Confirmed |
| Post-Catalyst-S | 1 | 0 | 1/1 | 100% | Too small sample |

- [4/22] No formal `s_scan` recipe fires were promoted from `audit_history.jsonl`; review evidence came from narrative strongest-unheld plus `S Hunt` / `S Excavation` receipts.
- [4/22] Narrative strongest-unheld kept landing on `GBP_JPY LONG`, `EUR_USD SHORT`, and `EUR_JPY SHORT`; only late `EUR_USD SHORT` was captured, so the lesson is auditor-value plus deployment gap, not recipe failure.
- Biggest structural miss remains unchanged: Momentum-S is still near-perfect and still almost never entered.

---

## Event Day + Thin Market Rules

- Within 90 minutes of binary event or a stacked late-session macro window: LIMIT / STOP-ENTRY only at structural levels. No market-order range scalps.
- Holiday/thin conditions: fixed structural SL or no SL; tight trails are just noise collection.
- Spread >2x normal: never chase with market order. Wait, use LIMIT, or pass.
- Pre-event backup orders: if the active vehicle rotates away or the invalidation trades before trigger, cancel the old backup instead of letting a stale wide-spread cross fill into the fade.
- Weekend approaching: if holding through the gap, set real structural risk. Do not depend on later restore logic.
- A late-session breakdown that has not paid before the weekend is not a free swing. Re-justify it from the live chart or cut it before it becomes zombie inventory.
- Weekend reopen core-edge rule: if `EUR_USD`-style H4 support + H1 BB-lower reclaim is the strongest unheld A/B seat, leave an explicit stop-entry / reopen trigger. "Watch on open" is not deployment.

---

## メンタル・行動

- **入らないこと自体が最大リスクになり得る。** But "doing something" is not the same as trading well.
- **LIMITは入ったフリになりやすい。** Use LIMIT for structural ambush, not to avoid taking a real view.
- **Conviction = size.** Do not double-discount by scoring a trade high and then sizing it like a fear trade.
- **最低3,000uが trader-owned inventory の基本。** Smaller discretionary size usually means you should not be in the trade.
- **Hero pair集中 + 回転が収益源。** Rotation P&L is the system; random distribution across many pairs is not.
- **2 losses on the same pair/direction is a warning. 3 losses or a fifth re-buy through the same unreclaimed shelf is a stop sign.** 4/21 `AUD_JPY SHORT` hit the stop sign and kept trading anyway.
- **11 short-term S names with 2/11 accuracy is not aggression, it is anxiety.** When short-horizon hunting gets noisy, escalate to medium/long horizon or stand down.
- **A missed core-edge A/B call is still a loss of judgment.** `EUR_USD LONG` +63.8 pip unarmed was not patience, and 4/20 `GBP_USD LONG` strongest-unheld continuation misses were the same mistake in a cleaner market.
- **Letting a winner turn red without a new reason is the same quality failure as entering a stale B seat at market.** Both are avoidable judgment leaks.
- **Quality audit flags are information, not orders.** Audit can reveal opportunity; it cannot replace pair-level judgment.
- **If the market read is good enough to modify TP, it is good enough to ask whether the opposite direction is the better trade now.**
- **Repeat pressure is evidence, not noise.** When audit / missed-seat pressure keeps naming the same pair-direction, do not let a `pass/C` seat die as prose by default. Re-open it as a thin trigger/passive scout unless the contradiction is explicit.
- **A correct unheld lane is still evidence.** 4/22 proved that booked losers can coexist with right missed seats; do not let realized P&L blind the next session to the cleaner undeployed vehicle.
- **Protected inventory is not the same as unpaid naked inventory.** A same-direction add-on can stay alive as `LIMIT` / `STOP-ENTRY` when the live leg is already protected. The ban is on blind averaging-in, not on every layered seat.
- **Stale carry-forward flatness is a real bug, not trader discipline.** If `state.md` still says `Backup vehicle: none` or `Next fresh risk allowed NOW: none` while the live board has a fresh lane, overwrite the stale ladder immediately. Repeating yesterday's flat line is just underdeployment with nicer prose.
- **Lane lines are not summaries, they are closures.** If the board still has lane 2 / lane 3, those lines must name the receipt or the exact dead-thesis contradiction. `none` is just hiding the missed seat.
- **A swallowed validation failure is still a failed session.** If `session_end.py` says `STATE_VALIDATION_FAILED`, the runtime must stop there. Converting that into a quiet `mid_session_check` only teaches the system that underdeployment is acceptable.
- **Stale self-check counts are false memory.** `Entries today: N total` is not harmless wording; it anchors the next session into thinking the book was already hyperactive. Keep the structured live count (`fills / new entry orders / rejects`) or the handoff is lying.
- **The focus ladder can hide missed risk just as much as the lane lines can.** If `Backup vehicle` or `Next fresh risk allowed NOW` still says `none` while the current board has a lane, the handoff is still flat-lie prose even if `Lane 2 / Lane 3` were updated correctly.
- **A board seat is not a live order license.** The exact `entry / tp / sl` must still clear `pretrade_check.py` at send time. If the real stop is inside the pair's noise floor, the helper must block even when the board liked the idea.
- **Self-reported allocation is not a control.** The live order helper must derive `pretrade / allocation / allocation_band` from the exact pretrade result, or stale `--allocation A` prose will keep oversized B/C seats alive.
- **A live pending order can be dirty even when the handoff is clean.** Re-audit trader pending entry orders against exact pretrade geometry at session end; an order like `469447` that no longer clears the stop floor is contaminated risk, not valid inventory.
- **A lane without a shelf-life is just another prose trap.** Every actionable `MARKET / STOP-ENTRY / LIMIT` seat needs `how many 20-minute sessions it survives` or `until what UTC time`. Otherwise the next run rediscovers or forgets it at random.
- **Carry-forward needs a contract, not a vibe.** `MARKET` is this-session-only, `STOP-ENTRY` is short carry, passive `LIMIT` is multi-session while the structural shelf survives, and armed pending entries inherit their actual GTD. Write that explicitly.
- **Duplicate headings are stale-state poison.** A second `## Deepening Pass` or `## Capital Deployment` does not replace the first one; it usually just leaves the parser reading old prose. Rewrite the existing block in place.
- **`20-minute backup trigger armed NOW` is receipt-only.** If there is no real pending entry `id=...`, the line must be `none because ...`. Board-only is not armed.
