# Strategy Memory — プロトレーダーの経験知

**daily-reviewが毎日更新。traderが毎セッション冒頭で読む。**

最終更新: 2026-04-16 (daily-review: +303 JPY, 2/5 WR, R:R 2.2:1)

---

## ⚡ READ THIS FIRST — The #1 profit killer is NOT entering

**Historical data (3/20-4/16):**
- Bad trade avg loss: -354 JPY (B-size). Recoverable in 1 trade
- Drought day (0 entries): -12,569 JPY opportunity cost (entire daily target). NOT recoverable
- Asymmetry: **36:1.** One missed day = 36 bad B-trades worth of damage
- 4/15-16: 5 S-scan signals fired, 4 correct (+~30pip missed). Reason: "waiting for event" / "spread wide" / "theme late"
- **The biggest losses in this system's history came from NOT ENTERING, not from entering wrong**

**When you finish reading the patterns below, your job is to ENTER, not to WAIT.**

---

## Confirmed Patterns（3回以上検証済み — ルールとして扱え）

### ✦ 勝ちパターン（これが金を生む — 読んだら入れ）

- **SIZE IS THE #1 LEVER (500-trade analysis, 3/20-4/16)**:
  - **1-2k units: 330 trades, -23,098 JPY, 52% WR.** Small trades LOSE MONEY in aggregate. Avg win +212 can't offset avg loss -370
  - **3-5k units: 152 trades, +19,226 JPY, 61% WR.** Same trader, same setups — just bigger. Avg win +583 vs avg loss -574 = break-even R:R but WR carries it
  - **9k+ units: 4 trades, +5,921 JPY, 100% WR.** S-conviction sizing WORKS. Every single one won. avg +1,480/trade
  - **Minimum 3,000u enforced.** Below that is proven to be net-negative. If it's not worth 3k, don't enter

- **HOLD TIME: 30-120min = ALL the profit (500-trade analysis)**:
  - <5min: -803 JPY (37% WR) — noise trading, negative EV
  - 5-30min: -5,090 JPY (48% WR) — too impatient, cutting winners before they develop
  - **30-120min: +27,539 JPY (65% WR)** — the ENTIRE system edge lives here
  - 2h+: -21,593 JPY (54% WR) — zombie holds, thesis dead but position alive
  - **Implication**: Enter with 30-120min hold expectation. If flat after 15min → wrong entry (noise zone). If held 2h+ without TP → zombie, cut or swing-commit

- **TIME OF DAY: 08:00 UTC is gold, 22:00 UTC is poison (500-trade analysis)**:
  - Best: 08:00 UTC (17:00 JST) = +7,344 JPY from 37 trades (avg +198)
  - Worst: 22:00 UTC (07:00 JST) = -9,415 JPY from 19 trades (avg -496)
  - 22:00-23:00 UTC combined: **-12,191 JPY** from 37 trades. This 2-hour window erased 6 days of profit
  - 19:00-23:00 UTC = rollover zone, thin liquidity. LIMIT only, B-size max

- **Session dynamics — where the edge actually lives (500-trade analysis, 3/20-4/14)**:
  - **By ENTRY time**: Tokyo +4,997 (119t, 56%WR, avg+42) | London +8,130 (166t, 65%WR, avg+49) | NY -103 (171t, 50%WR, avg-1) | Late NY -11,898 (44t, 36%WR, avg-270)
  - **Tokyo looks bad by CLOSE time (-8,868)** because NY overnight losers get dumped during Tokyo morning. That's an NY position-management problem, not a Tokyo problem
  - **Tokyo entry → London close = avg +347/trade (29t)**: 7× the system average. Tokyo builds the range, London breaks it. If you can read which side the range breaks, positioning in Tokyo before London is the highest-edge play in the system
  - **Late NY (21-00 UTC / 06-09 JST) is the system's worst session**: GBP_USD alone -9,601 from 8 entries (25%WR). Rollover spread + thin liquidity. If the setup is real it'll still be there at Tokyo proper
  - **NY is highest volume (171 entries) for zero return**: The trader is most active here and making no money. Activity ≠ edge
  - **Momentum (30m-2h) is the system's real edge**: Tokyo +4,927(67%WR), London +10,994(75%WR), NY +7,994(61%WR) — profitable in every session. Scalps (<30m) lose in Tokyo(-1,504) AND London(-6,503). The system makes money from reading a 30m-2h directional move correctly, not from 5-minute timing. This isn't about "don't scalp" — it's about recognizing that the edge comes from depth of read, not speed of execution
- **S-conviction discovery = narrative, not scanner**: S-conviction comes from STORY COHERENCE (3+ categories align + chart tells same story + macro confirms), NOT from s_conviction_scan.py recipe matches. Scanner has narrow thresholds (StochRSI ≤0.05/≥0.95) — most S-setups have StochRSI=0.10-0.20 (shallow pullback in strong trend). At any time, 3-5 S-setups exist across 7 pairs. Discovery method: write "I would [DIR] because [story]" for every pair → S emerges when the story coheres. Auditor writes 7-pair conviction map; trader reads and responds. Scanner is supplementary confirmation (Structural-S proven 3/3, Counter-S 4/5, rest noisy or broken)
- **EUR_USD LONGはシステム最強エッジ**: 全期間55%WR, avg+192JPY, total+15,763JPY。迷ったらEUR_USD LONG。H1 ADX>25+EUR strongest+M5 StRSI=0.0は鉄板 (3/31-4/1: 6回連続利確, 4/8: 2/2 wins +4,575JPY, 4/10: +434JPY+228JPY)
- **AUD_JPY両方向で利益**: LONG 64%WR avg+26JPY, SHORT 59%WR avg+54JPY。両方向プラスの安定ペアだが、チャーン(close→re-enter繰返し)で利益を食い潰すリスクあり
- **M5 StochRSI=0.0 + H1 BULL構造不変 → 高確率反発** (3/23 GBP V字+653円, 3/23 EUR BB Mid, 3/26-27複数回, 4/1 EUR_USD 5回連続成功)
- **H1+M5方向一致 + ADX>25 = 高確度エントリー** (3/23 GBP H1+H4 BULL, 3/31-4/1 EUR_USD ADX=46-53で連勝)
- **3/31の回転トレード = 理想形**: 19エントリー, avg 4,737u, +4,591JPY。LIMITでFib pullback待ち→TP→即再エントリー。サイズを恐れず、回転を止めなかった
- **LIMIT + TP/SLで寝ている間に利確**: 3/31 GBP_JPY TP+906JPY, EUR_USD TP+1,622JPY — LIMITが刺さってTPまで到達。離席中の利益はシステムの強み
- **Counter-trade at H4 extreme → bounce capture**: H4 CCI±200/StRSI 0or1 → 反対方向に小ロットで利益 (4/1 USD_JPY SHORT +448JPY, bounce catch)
- **Weekend gap-down averaging-down → SL loss recovery**: 4/13: AUD_JPY LONG 8000u @112.742 held into weekend, SL hit @112.396 = -2,768 JPY. User manually entered 10,000u @111.981 (76pip below original) at Sunday open 06:34 JST, judging the gap-down as excessive. Closed @112.383 = +4,020 JPY. Net +1,252 JPY — SL loss fully recovered + profit. **Key**: (1) new entry at significantly better price (76pip lower), (2) S-size conviction (10,000u), (3) executed at open before bounce, (4) didn't hesitate. This is averaging-down done RIGHT — different price level with conviction, not doubling down at the same level on hope
- **EUR_JPY LONG in trending conditions = highest WR in system**: 69%WR, avg+140JPY, total+1,824JPY. 4/7: 6/6 wins +1,876JPY ALL at pretrade=LOW. When H1 ADX>35 + EUR strongest + JPY weakest, pretrade LOW is wrong — the trend overpowers historical stats. (3/31, 4/6, 4/7: 3+ sessions confirmed)
- **Rotation SHORT within LONG thesis (M5 pullback trading)**: Holding LONG + M5 turns bearish (StRSI↓from 1.0, MACD hist shrinking, lower highs, upper wicks, CCI falling) → SHORT 2000-3000u, TP=M5 support (ATR×0.5-1.0), 15-30min hold. OANDA hedge = LONG stays open, rotation SHORT costs ZERO additional margin. NOT counter-trading — standard pullback trading. 4/8-4/9: 13 entries all LONG, 0 rotation SHORTs. Drawdown periods (EUR_JPY -849JPY, AUD_JPY 111.667→111.550=11.7pip drop, GBP_USD 44.5pip drop to SL) were all capturable as SHORT profits. **If M5 data convinced you to tighten TP, that same data was a SHORT entry signal.** 4/10: USD_JPY SHORT signal identified in Slack (H4 DI->DI+, H1 StRSI=1.0) → not entered → price fell 100+pip to 158.72. Signal seen, analyzed correctly, never traded. pretrade_check wave fix applied (h4+m5 aligned=mid)
- **H4-supported SHORT in bullish macro ≠ counter-trade**: USD_JPY 4/10: H4 DI-=23>DI+=15 (H4 BEARISH) + M5 DI-=38>DI+=12 (M5 BEARISH). Only H1 was still bullish (transitioning). This is NOT counter-trend — H4 supports the SHORT. pretrade_check was classifying this as "small wave" (M5 only). Now classified as "mid wave" (H4+M5 aligned). Macro (JPY weakest) opposes, but that's why it's rotation size (2000-3000u), not swing size

**⚡ You just read 15+ proven winning patterns. Now go find one in today's market and ENTER.**

### ✦ 負けパターン（知っておけ。だが恐れて入らないのは負けパターンより高くつく）

**WARNING: This section has 30+ warnings. Reading them all before trading creates paralysis. The avg loss from a bad B-trade is -354 JPY. The cost of not trading for a day is -12,569 JPY (missed target). Fear of these patterns is 36× more expensive than the patterns themselves.**

- **3-LOSS CIRCUIT BREAKER (4/15: -5,226 JPY day, max 16 consecutive losses)**: After 3 consecutive losses in the same direction → STOP entering that direction this session. 4/15: 21 trades, 13 losses, nearly all LONG. After loss #3, the remaining 10 losses cost -3,200 JPY. The market was selling; the trader kept buying. **3 losses = direction is wrong today. Either flip or stop.**

- **フロー逆行エントリーは必ず負ける**: テクニカルだけでフロー逆方向に入ると勝てない (3/20 USD_JPY 4/5負け, 3/23 GBP SHORT全敗)
- **スパイク慌て損切り = 最大損失源**: 天井/底で成行は最悪。スパイクは戻す確率が高い (3/23 GBP -3,832円, 3/26 パニック-8,041円)
- **小さすぎる利確が利益を食い潰す**: ATR50%未満の利確はスプレッド+手間に見合わない (3/26 勝率65%でNet-583円, 勝ち平均+84円)
- **マージン極限で新規ペア追加 → クローズアウト** (3/26 EUR_JPY追加→全ポジ強制決済)
- **薄商い(祝日/GoodFriday)のタイトSL = 全部狩られる** (4/3: EUR_USD trail 11pip, GBP_USD trail 15pip, AUD_USD SL 10pip → 全滅 -984円。テーゼは全部正解だった)
- **S-convictionをB-sizeで入る = 最大の機会損失**: 3/20-4/3で7回。6,740-13,140JPY失った。pretrade WR数字でビビってsized_down。二重割引するな
- **連敗サーキットブレーカーは同方向のみ**: AUD_JPY SHORT 4連敗 ≠ AUD_JPY LONG禁止。SHORTの連敗はLONGに無関係。方向ごとに判断しろ
- **Margin stacking without worst-case calc → forced close (3x verified)**: 4/1 (-closeout), 4/8 EUR_JPY(-319 at 97%), 4/9 AUD_USD(-1,016 at 98%). Pending LIMITs ignored in margin calculation every time. **Hard rule: margin calc MUST include all pending LIMIT worst-case fills. Above 85% = no new entries. Above 90% = immediate half-close.** Repeat offender — this WILL happen again without discipline
- **Bearish signals seen but never traded (4/8-4/9)**: M5 overbought+div, sellers dominant, lower lows — all noted → only used to tighten TP/SL. Never entered SHORT. If the data is good enough to change TP, it's good enough to trade the other direction. **The problem was not "SHORT is bad" — it was "trader wasn't reading the chart for both directions."**
- **Chart said SHORT, trader said LONG (4/8 USD_JPY)**: H4 DI-=31 + H1 ADX=51 BEAR = clear SHORT chart. Tried LONG 3x (counter-bounce thesis) → all lost (-113JPY). **Trade what the chart shows. Macro is one input, not a veto on chart direction.**
- **Shallow indicator scan = one-direction blindness (4/8-4/9)**: Checking 3 indicators → LONG locked. Checking 6+ indicators → rotation SHORTs become visible. **The fix is depth of analysis, not direction preference. Both directions are valid depending on what the chart says.**
- **GBP_USD repeated entries after failure = compounding losses**: 4/7: 4 entries (3 LONG, 1 SHORT), 1 win +93 JPY, 3 losses totaling -1,099 JPY. 5000u LONG @1.32383 lost -480 JPY in 17 min. After 2 consecutive GBP_USD losses, move to another pair. (4/7: verified 1x, but pattern matches 4/1 GBP_JPY repeated entries)
- **Pretrade score doesn't factor market regime (4/7-4/9, 5 SHORT trades)**: GBP_USD SHORT HIGH(8)→-317, EUR_USD SHORT HIGH(8)→-288, AUD_USD SHORT HIGH(8)→-2,699, USD_JPY SHORT HIGH(10)→-270, EUR_USD SHORT HIGH(10)→-2. 0/5. **These SHORTs lost because the market was bullish in that period, not because SHORTs are inherently bad.** Pretrade needs context-awareness (what regime are we in?) for BOTH directions — a HIGH LONG in a bear market would fail the same way. Don't downgrade all SHORTs; read the chart.

## Active Observations（検証中 — daily-reviewが追記する）

<!-- 各観察: [初出日] 内容。検証: N回確認/N回反証 -->

- [3/23] M5 RSI Regular Bullish Div score=1.0 → 底打ちの決定的シグナル. 検証: 2回確認(3/23 GBP +25pip, 3/27 EUR_USD)
- [3/27] **pretrade LOWの警告は概ね正しい** — ただしEUR_JPY/GBP_JPY trending条件ではLOWが系統的に間違う(4/7, 4/9で確認済み)。トレンドペアではoverrideの余地あり
- [4/3] **Tokyo session trail trap**: ATR×0.7 trail = sub-noise in Tokyo. Hard rules: (1) Tokyo trail min=ATR×1.2. (2) Pre-event=fixed SL only. (3) SL must name structural level. Verified: 1x. Binary "trail or hold" cost 17,844 JPY across 3 incidents.
- [4/7] **Pretrade LOW massively wrong for trending EUR_JPY LONG**: 4/7 EUR_JPY LONG had 6 entries all pretrade=LOW. Result: 6/6 wins, +1,876 JPY. LOW-scored entries across all pairs today: 16 trades, 69% WR, +680 JPY. **Pretrade's historical WR for the pair+direction penalizes trending pairs whose stats are diluted by range-bound sessions.** When H1 ADX>35 + macro CS alignment (EUR strongest, JPY weakest), pretrade LOW should be overridden to at least MEDIUM. Verified: 1x (4/7 EUR_JPY 6/6)
- [4/7] **Pretrade HIGH doesn't mean regime-aligned**: 5/5 HIGH SHORTs lost in bullish regime (4/7+4/9). Pretrade score is direction-agnostic. Same applies to LONGs in bear regime. Verified: 1x
- [4/7] **Churn = spread destruction**: AUD_JPY 3 round-trips -778 JPY + 9.6pip spread burned. Close→re-enter same thesis = don't. First SL = final. 30min cooldown. Verified: 2x (4/6+4/7)
- [4/7-16] **Winners held 48min avg (4/16), losers held 15min avg**: Patience = profit. Consistent across 4 days measured. Don't cut within 15 min unless structural invalidation. Verified: 3x
- [4/8] **Quality audit "S-candidate fix" pressure → oversized loss**: AUD_USD LONG 8000u entered as "quality audit miss fix" → -2,540 JPY. Quality audit flagged AUD_USD as missed S-candidate, trader entered 8000u (~S-size) on a pair with 50%WR LONG all-time. Audit flags are data, not orders. Don't let audit pressure override your own pair-level conviction. AUD_USD LONG avg=-78JPY all-time — this is NOT an S-conviction pair. Verified: 1x
- [4/8] **Counter-trades against H1 ADX>35 BULL = money pit**: H4 StRSI=1.0 does NOT override H1 ADX>35 trend. Swing-size counters lose; M5 rotation SHORTs are different. Verified: 2x
- [4/8-9] **Margin crisis + indicator depth + chart-vs-narrative**: (1) Margin >85% with pending LIMITs → forced close (2x verified). (2) 3 indicators = one-direction blindness, 6+ = both visible (1x). (3) Trade the chart, not the macro narrative (1x). These three combine: shallow scan → one direction → oversize → margin blow. Depth of analysis prevents all three.
- [4/9] **Momentum-S multi-pair = regime signal**: 50+ fires all LONG, EUR_JPY +59pip, GBP_JPY +56pip, AUD_JPY +59pip. ALL NOT_HELD. When Momentum-S fires ≥3 pairs same direction → enter strongest CS pair A-size minimum. Verified: 1x
- [4/10-12] **Rollover_guard restore gap**: restore requires trader session but trader halts weekends. 4/10 AUD_JPY -2,864 JPY. **Weekend approaching = MANDATORY structural SL set.** Verified: 2x
- [4/11] **Weekend Structural-S on JPY crosses = noise**: JPY crosses WRONG on weekends/consolidation (4/11+4/14). Counter-S GBP_JPY ✓. Verified: 2x
- [4/13] **Weekend gap = averaging-down opportunity when thesis intact**: AUD_JPY gapped 76pip, user entered 10,000u @111.981, closed +4,020 JPY. Net +1,252 JPY after old SL hit. Conditions: thesis unchanged + 50+pip gap + enter at open + S-size. Verified: 1x
- [4/13] **Weekend SL slippage**: SL=112.550 filled at 112.380 (-17pip through level). SL doesn't guarantee fill price on gaps. Treat weekend risk as SL×1.5. Verified: 1x
- [4/4] **Conviction-S undersizing = biggest silent profit killer**: 7 trades met S conditions, 5 entered at B-size. 6,740-13,140 JPY thrown away. Fix: "Different lens" in pre-entry format forces checking unused indicator categories. B→S upgrade is where the money is. Verified: 5x
- [4/14] **AUD macro weakness is pair-specific**: NAB=-29 → AUD_JPY TREND-BEAR but AUD_USD barely moved (USD weakest globally). Currency weakness shows up in different pairs depending on OTHER leg. Read CS table before assuming "AUD weak = all AUD pairs SHORT." Verified: 1x
- [4/14] **fib_wave N=BEAR overrides BB lower / StRSI=0.0 🎯 signal**: AUD_JPY M5 at BB lower + StRSI=0.01 fired S-candidate. But fib_wave showed BEAR(q=2.34) at Fib0% (at the low) with N=BEAR — the NEXT wave is also high-quality BEAR. BB lower bounce is real but fleeting within an ongoing bear wave. When N-wave quality ≥2.0, the "extreme oversold" bounce is noise, not a trend entry. Fib wave is the map (where in the wave are you); StRSI is the thermometer (how hot is it now). Thermometer ≠ trade direction. Verified: 1x
- [4/14] **OANDA leverage 20:1 for non-JPY pairs (GBP_USD, EUR_USD), NOT 25:1**: Margin calc = units × (base_CCY_JPY / 20). Using /25 understates margin by 25% → 92% margin on "calculated 84%". Hard rule: For non-JPY pairs, always use /20. Verify with initialMarginRequired from OANDA response. Verified: 1x (GBP_USD 4500u: estimated 38,754 JPY vs actual 48,423 JPY)
- [4/14] **Always_C_hold culminated in GBP_JPY SL hit (-876 JPY)**: 5+ consecutive audit HALF_TP recommendations. Each time held (spread 12-20pip argument). Position SL hit at 214.900. **Auditor was right.** "Spread too wide" was valid short-term but became infinite hold excuse. **If TP doesn't fill within 3 sessions after spread normalizes → close at market.** Verified: 1x (outcome of 5-session pattern)
- [4/14] **Tokyo thin LIMIT fill → instant SL hunt**: GBP_JPY 4000u LIMIT @215.119 filled 01:35Z, SL @214.900 hit 20min later = -876 JPY. SL was 21.9pip = ATR×1.1, too tight for Tokyo. Same pattern as 4/3 Good Friday. **LIMIT in Tokyo thin (00:00-06:00Z) must have SL ≥ ATR×1.5.** Verified: 2x (4/3 + 4/14)
- [4/14] **Counter-S recipe: first entered → WIN +460 JPY**: USD_JPY LONG Counter-S @159.096. Validates Counter-S as entry-worthy at H4 extreme + chart supports counter direction. Cumulative 5 fires, 4 correct = 80%
- [4/15] **NY overnight orphan = worst execution pattern**: 41 trades entered during NY, held overnight, closed during Tokyo morning = -14,094 JPY. The trader enters during NY, doesn't decide at entry whether to hold as swing or cut before NY close, then dumps losers at 10 JST into thin Tokyo liquidity (worst possible time to exit). This is the #1 drag on Tokyo-session P&L and makes Tokyo LOOK unprofitable when it isn't. Fix is at entry: "Am I holding through Tokyo? If yes, structural SL and swing mindset. If no, close before NY end. No orphaning"
- [4/15] **De-botified Pullback Quality**: Removed NOISE/SQUEEZE/DISTRIBUTION scoring + rule tables. Now raw data panel. Trader writes "I see / This tells me / So I'm doing" — forces reading, not label-following
- [4/15] **Trade type awareness — the R:R killer**: GBP_USD 8000u S-Momentum(PPI miss) held 5h40m → -2,583 JPY. Momentum thesis died in 30min but trader checked H1 (ADX=55="intact") instead of M5 (entry TF). Same disease as trail 8pip: managing trades on wrong timeframe. profit_check now shows `held: Xh Ym`. Evaluation block requires `Entry type + Held vs expected + Is thesis still why I'm here?`. Loss management changed from "H1 structure?" to "structure on entry TF?". Track: (1) Momentum trades held past expected window, (2) R:R improvement (currently 0.91, target >1.2)
- [4/15] **Zero-close day = margin-locked opportunity cost**: 4 positions (GBP_USD 4000u + EUR_USD 6000u) consumed 69-89% margin all day. 5 S-scan signals fired, 4 correct. AUD_JPY Structural-S @113.316 moved +16.2pip, AUD_USD Trend-Dip-S @0.71306 moved +11.4pip — combined ~28pip of missed opportunity. **The cost of holding positions isn't just their UPL — it's the S-scan entries you can't take.** When margin is above 70% and positions aren't making money, the opportunity cost of holding exceeds the risk of closing at a small loss. Verified: 1x
- [4/15] **LIMIT management quality improved**: 7+ LIMITs placed, 5 cancelled proactively — wrong regime (AUD_JPY SHORT in TREND-BULL), fill impossible (EUR_JPY 9pip away in thin Tokyo), margin management (EUR_USD worst-case over 85%). No naked LIMITs left overnight. GTD expiry used consistently. Contrast with 4/14 where GBP_JPY LIMIT filled in Tokyo thin → SL hit -876 JPY. Verified: 1x
- [4/15] **Trend-Dip-S improving when unidirectional**: 2/2 correct today (USD_JPY SHORT @158.914 +3pip, AUD_USD LONG @0.71306 +11.4pip). Both were clean single-direction signals, no contradictory flip. Historical 31% accuracy was dragged down by contradictory signals (SHORT then LONG within 90min on same pair). **Trend-Dip-S accuracy by signal type**: unidirectional signals may be more reliable than bidirectional. Track separately. Verified: 1x
- [4/16] **Counter-S EUR_JPY keeps winning at pretrade=LOW**: LONG @187.190 (H4+H1 StRSI=0.00 floor, JPY spike absorbed) → +420 JPY in 32min. pretrade=LOW(1) was wrong again. Counter-S recipe now 5/6 correct (83%). **pretrade LOW + Counter-S recipe = override to at least B.** Verified: 3x (4/7 EUR_JPY 6/6, 4/14 USD_JPY +460, 4/16 EUR_JPY +420)
- [4/16] **Spike-wide spread re-entry = negative EV**: EUR_JPY 2nd LONG @187.190 at Sp=7pip (normal=1.9pip). Lost -158 JPY in 1 min. The 5.1pip extra spread consumed the entire available range to TP (21pip target, 7pip spread = 14pip net, but R:R degraded from 1.5:1 to ~1.1:1). **When spread is >2× normal: LIMIT only, never market order.** Verified: 1x
- [4/16] **Pre-event RANGE scalp in SQUEEZE = negative EV**: EUR_USD LONG @1.18090 (M5 RANGE scalp, StRSI=0.0) → -140 JPY. M5 regime shifted to SQUEEZE within 20min of entry. TP was only 9pip in pre-GDP thin market. Zombie close was disciplined but entry shouldn't have happened. **Within 90min of binary event (GDP, NFP, CPI): LIMIT only at structural level, no market-order RANGE scalps.** Verified: 1x (matches 4/3 pre-NFP pattern = 2x total)
- [4/16] **LIMIT→TP auto-fire = system's best risk-adjusted pattern**: GBP_USD LIMIT @1.35619 filled overnight, TP hit @1.35781 = +513 JPY. No screen time, no decision stress. Cumulative: 3/31 GBP_JPY +906, EUR_USD +1,622. 4/15 GBP_USD +513. All LIMIT→TP. **LIMIT at structural level + TP at ATR×1.0 = best passive income.** Verified: 3x
- [4/16] **AUD_JPY H4 overbought pretrade warning was valid**: LONG @114.036, pretrade warned H4 CCI=88/RSI=75 overbought. SL hit in 6min = -332 JPY. Downsized from S to A-size for this warning, but entry itself questionable. **When pretrade flags H4 overbought, prefer LIMIT at deeper pullback (Fib50%+) over market entry at EMA20.** Verified: 1x

## Deprecated（反証済み — 参考のみ）

_(まだなし — daily-reviewが反証されたパターンをここに移動する)_

---

## Per-Pair Learnings（ペア別）

### USD_JPY
- 東京セッション + JPY最弱時にショートは負けやすかった（3/20: 5本中4本負け）— ただしJPY強い局面ではショートが正解になる。市況次第
- フローとテクニカルが矛盾するとき → チャートの方向を優先しつつフローも考慮。どちらか一方を常に正とするルールは作らない
- SL=4.6pipは止狩りに弱い。スプレッド×3以上を確保
- 介入リスクあり。+20pipで利確検討

### GBP_USD
- ホルムズ系ヘッドラインで+40-90pipの巨大スパイク (3/23)
- N波動targetは信頼できる利確目標
- H1 ADX=26でH4も同方向BULL = 最もクリーンなトレンドペア
- **All-time LONG: 63% WR avg +80 JPY total +4,308 JPY** — strongest LONG edge after EUR_USD. LIMIT→TP auto-fire works well (4/16: +513 JPY overnight). Requires patience (avg winner hold >100 min)
- **SHORT: 25% WR avg -1,027 JPY total -24,639 JPY (sample: 3/17-4/15, predominantly bullish period)** — poor stats but sample is biased by bull market
- Tokyo session trailing stop trap same as EUR_USD: 15pip trail = ATR×0.7 got clipped (4/3 lesson)
- **4/7: repeated entries after failures compound losses.** 4 entries in one day, -1,006 JPY net. After 2 GBP_USD losses, move to another pair
- **Late NY (21-00 UTC) GBP_USD entries = system's worst combination**: 8 entries, -9,601 JPY, 25%WR
- **4/15: dual 2000u positions (4000u total) held all day at 69% margin** — occupied margin without closing or adding. Both WATCH-rated for 4+ audit cycles. UK data binary was the thesis but no resolution realized

### EUR_USD
- スプレッド最小。スキャルプに最適
- VWAP+35pip乖離での追撃 → 過熱からの反転で負け
- ロンドンクローズ(16:00 UTC)で方向変わりやすい
- **All-time LONG 56%WR avg +211 JPY total +18,396 JPY** — best total P&L of any pair/direction by far. EUR_USD LONG is the strongest edge in the system
- **SHORT 48%WR avg -169 JPY total -8,103 JPY (sample: 3/17-4/15, EUR bullish period)** — sample biased by EUR strength
- Trailing stop in Tokyo session (00:00-06:00Z): ATR×0.7 clips positions. Use ATR×1.2 minimum or hard SL only (4/3 lesson)
- **Tokyo entry → London close is EUR_USD's highest-edge play**: EUR_USD entered 00-07 UTC, closed in London = +4,570 JPY (7t)
- **4/15: 6000u dual position held all day** — H1 div=1.2 persisted 8+ audit cycles. NY orphan (3500u entered 19:40Z 4/14, held 9h+). Both positions in loss for most of the day. Always_C_hold soft-matched 4+ cycles. When H1 div persistently above 1.0 and position is in loss, the momentum cooling signal is real — don't hold for "H4 is intact" when H1 is deteriorating

### AUD_USD
- フロー分析が特に効く。STRONG_SHORTで+889円実績 (3/20)
- SHORT 39%WR avg-14JPY, LONG 50%WR avg-78JPY (sample: 3/17-4/9). Both negative but sample is short and biased by AUD-weak period
- **LONG 48%WR avg-123JPY total-2,574JPY | SHORT 38%WR avg-106JPY total-3,078JPY (sample: 3/17-4/9)** — both negative in this sample. AUD_USD requires strong CS + macro alignment for either direction. Stats will shift with market regime
- **4/8: 8000u LONG "quality audit S-candidate fix" → -2,540 JPY** — largest single loss. Audit pressure pushed oversized entry on a pair with no edge. Lesson: audit flags ≠ S-conviction
- **4/9: 7500u LONG S-conviction @0.70474 → -2,699 JPY** — SL hit at 0.70250 (-22pip). Three losses in 2 days: -2,540+−1,016+−2,699 = **-6,255 JPY**. No-edge pair confirmed. **AUD_USD HARD CAP: B-size (1,667u max). S-scan signal does NOT override pair-level no-edge status.**

### AUD_JPY
- スプレッド広い（1.6-1.9pip）。SLは広め必須
- SL=4.6pipで0.2pip差の止狩り→即反発。SL=9pipなら生存
- **LONG 63%WR avg +117 JPY | SHORT 56%WR avg +5 JPY** — LONG clearly dominant. (All-time from daily_review.py 4/13: AUD_JPY LONG 27/43 wins +5,025 JPY, SHORT 36/64 wins +329 JPY)
- SHORT has marginally positive avg but much lower WR (56%) vs LONG (63%). When SHORT thesis is clear, fine to enter but don't over-size.
- **Weekend gap SL slippage**: 4/13 SL=112.550 filled at 112.380 (-17pip through SL). Treat weekend holding risk as SL×1.5.
- 方向転換(LONG↔SHORT)のタイミングに注意。転換直後は負けやすい (3/27: LONG+514→SHORT-792)
- **4/7 churn = profit killer**: 4 trades, 3 round-trips, net -738 JPY + 9.6pip spread. LONG 3500u cut in 6 min (-217), then 5000u held 170 min for +40 (BE SL). Close→re-enter same thesis burns spread. First exit should be final for 30 min
- **Tokyo session = AUD_JPY's natural home**: +3,717 JPY in Tokyo (44t). Sydney+Tokyo overlap provides real directional flow — not a dead market. Momentum trades (30m-2h) work here because AUD has actual institutional flow during Asian hours, unlike GBP/EUR which are range-bound noise until London

### GBP_JPY
- 30-50pipスウィングが普通。痛み上限-40pip
- 25pipの逆行で即切りは早すぎる
- **LONG 65%WR avg+140JPY total+2,794JPY** — strong LONG edge but vulnerable to Tokyo thin SL hunts. Always_C_hold pattern (5+ audit warnings ignored → SL hit -876 JPY on 4/14). LIMIT fills in Tokyo thin → SL ATR×1.5+ minimum

### EUR_JPY
- ボラ大きい。利確遅れやすいので注意
- **LONG 69%WR avg +140 JPY total +1,824 JPY** — highest WR and avg in the system. When EUR strongest + JPY weakest (macro alignment), this pair prints money
- **4/7: 6/6 LONG wins, +1,876 JPY** in one day. All pretrade=LOW yet all won. H1 ADX=35-43 trending strongly. Pretrade undervalues this pair in trending conditions
- Spread 1.7-2.3pip is high. Need 20+ pip target to justify entry. Sub-10pip scalps don't work here
- **Counter-S at H4+H1 StRSI=0.00 floor = high-probability entry**: 4/16 LONG @187.190 +420 JPY (32min hold). JPY spike absorbed, M5 bullish div confirmed. 2nd entry at Sp=7pip lost -158 JPY — **never re-enter Counter-S at spike-wide spread**

---

## Pretrade Feedback（daily-reviewが自動更新）

### LOWが正しかった場面（避けるべきエントリー）

- [3/27] AUD_JPY/AUD_USD LOW → total -1,804 JPY. AUD pairs at LOW = correct warning

### LOWが間違っていた場面（pretradeが保守的すぎるケース）

- **EUR_JPY LONG trending/Counter-S override (4/7+4/9+4/16)**: 9/10 wins, +2,724 JPY, ALL pretrade=LOW. When H1 ADX>35 + CS alignment OR Counter-S recipe at H4 floor, pretrade LOW is systematically wrong. Verified: 3x
- [4/9] GBP_JPY LONG pretrade=LOW(3) → 2/2 wins +319 JPY: LOWs winning in bull regime across multiple pairs
- [4/16] EUR_JPY LONG pretrade=LOW(1) → +420 JPY (Counter-S). 2nd entry LOW(1) → -158 JPY (7pip spread killed it, not the thesis)

### HIGHが外れた場面（スコア過信に注意）

- **HIGH doesn't mean regime-aligned (4/7+4/9 pattern)**: GBP_USD SHORT HIGH(8)→-317, EUR_USD SHORT HIGH(8)→-288, AUD_USD SHORT HIGH(8)→-2,699, USD_JPY SHORT HIGH(10)→-270, EUR_USD SHORT HIGH(10)→-2. **5/5 HIGH SHORTs lost in bullish regime.** Pretrade score is direction-agnostic — it doesn't know the current regime. Same would apply to LONGs in bear regime
- [4/3] EUR_USD LONG HIGH(8)→-170: trail width (ATR×0.7) caused loss, not score

### MEDIUM scores — mixed bag

- **MEDIUM on EUR_USD LONG = reliable** (4/8: 2 trades +4,576 JPY). This pair elevates MEDIUM into wins
- **MEDIUM on AUD_USD = treat as LOW** (4/8: -2,540 JPY). No-edge pair. Pair stats override session score

### B/MEDIUM scores — recent

- [4/13] AUD_JPY SHORT pretrade=MEDIUM(4)/B-conv @112.400 → -152 JPY. B-conviction, B-size (2000u). Score accurately reflected incomplete picture
- [4/14] GBP_JPY LONG pretrade=MEDIUM(4) → -876 JPY. LIMIT fill in Tokyo thin 01:35Z, SL hit. Score was accurate
- [4/14] USD_JPY LONG pretrade=MEDIUM(5) → +460 JPY. Counter-S recipe entry. Score undervalued — H4 extreme + chart alignment was A-quality
- [4/15] GBP_USD LONG pretrade=S(8) → sized A (2000u) — A-sized due to H4 overbought CCI=108 RSI=73. Still open. Downsizing S→A with explicit rationale is OK (unlike double-discount)
- [4/15] AUD_JPY LONG pretrade=B(5) @113.14 LIMIT → cancelled (no fill probability). Score was reasonable for Tokyo thin range-buy
- [4/15] USD_JPY SHORT pretrade=A(6) @159.050 LIMIT → not filled (GTD expired). Correctly sized B due to historical 33%WR WARNING

### MEDIUM scores — 4/16

- AUD_JPY LONG MEDIUM(7) → -332 JPY. H4 overbought warning was valid. SL hit 6min. **MEDIUM + H4 overbought flag = prefer LIMIT at deeper level**
- GBP_USD LONG MEDIUM(4) → +513 JPY. LIMIT→TP overnight. Score undervalued — theme confirmed
- EUR_USD LONG MEDIUM(6) → -140 JPY. Pre-GDP squeeze killed it. Score was reasonable but timing was wrong

### B-score entries

- **B-score in Tokyo = dangerous**: low conviction + thin liquidity + aggressive trail = noise kill (4/3: -146 JPY)
- [4/15-16] B-LIMITs with GTD expiry = zero-cost optionality. Good pattern. 4/16: 10+ LIMITs managed, most cancelled proactively. LIMIT discipline improving

---

## 指標組み合わせの学び

### 効いた組み合わせ

| 日付 | ペア | 組み合わせ | TF | 結果 |
|------|------|-----------|-----|------|
| 3/23 | GBP_USD | M5 StochRSI=0.0 + H1 ADX>25 BULL | M5+H1 | +653円 V字回復 |
| 3/23 | GBP_USD | M5 RSI Bullish Div(score=1.0) + H1構造BULL | M5+H1 | +25pip |
| 3/23 | EUR_USD | M5 BB Mid + StochRSI=0 | M5 | 勝ち |
| 3/23 | GBP_USD | H1 ADX=26 + H4同方向BULL | H1+H4 | クリーントレンド |

### 効かなかった組み合わせ

| 日付 | ペア | 組み合わせ | 結果 | なぜ |
|------|------|-----------|------|------|
| 3/20 | USD_JPY | M1 StochRSI=1.0のみ | 4/5負け | M1極限だけでは根拠不足 |
| 4/8 | AUD_JPY | H4+H1 StRSI=1.0 counter vs ADX>35 BULL | 0/2 -200JPY | osc extreme doesn't override trend |

| 4/7 | EUR_JPY | H1 ADX>35 + StRSI=0.0 + CS alignment | 6/6 +1,876JPY | Trending CS alignment = best edge |
| 4/8 | EUR_USD | H1 ADX>30 + StRSI dip + CS gap>1.0 | 2/2 +4,575JPY | Best 1-day EUR_USD |

### 未検証（今後試す）
- Keltner+BB同時squeeze, Chaikin Vol+ATR変化率, cluster_gap+Ichimoku雲

---

## S-Scan Recipe Scorecard (daily-review tracks)

| Recipe | Fires (all-time) | Entered | Correct (±5pip) | Accuracy | Status |
|--------|-------------------|---------|------------------|----------|--------|
| Structural-S | 23 | 1 | 12/23 | 52% | **Degraded** but improving. 4/15: AUD_JPY LONG @113.316 +16.2pip ✓ (JPY cross worked today). USD pairs reliable (~70%), JPY crosses improving. |
| Momentum-S | 55+ | 0 | ~53/55 | ~96% | Confirmed. No fires today. Still 0 entered = chronic missed opportunity. |
| Trend-Dip-S | 18 | 4 | 7/18 | 39% | **Watch** — 4/15: USD_JPY SHORT +3pip ✓, AUD_USD LONG +11.4pip ✓. 2/2 when unidirectional (no contradictory flip). |
| Squeeze-S | 9 | 1 | 1/9 | 11% | **Deprecated** — 4/15: USD_JPY SHORT @158.828 WRONG (-5.6pip). 1/9 after 9 fires. Remove from S-scan. |
| Counter-S | 6 | 2 | 5/6 | 83% | **Confirmed**. 4/16 EUR_JPY LONG @187.19 +420 JPY. Best entry recipe. Enter at A-size minimum. |
| Post-Catalyst-S | 1 | 0 | 1/1 | 100% | **New** — 4/15: GBP_JPY LONG @215.635 +2.5pip ✓ (marginal). Too small sample. Track. |

**4/15 details**: Structural-S 1/1 (AUD_JPY LONG @113.316 +16.2pip ✓). Trend-Dip-S 2/2 unidirectional. Squeeze-S 0/1 WRONG. Post-Catalyst-S 1/1 marginal. ALL 5 missed (margin lock).
**4/16 details**: No formal s_scan recipe fires in audit_history. But audit flagged AUD_JPY LONG as S_conviction_B_size (04:33Z) — trader entered B-size, audit said should be A/S per 5-category alignment. AUD_JPY went to 114.157 (+22pip from 113.93 LIMIT level). Signal was CORRECT but LIMIT never filled. Counter-S EUR_JPY entered manually → +420 JPY.
**Key finding**: Counter-S is now the highest-accuracy recipe at 83% (5/6) AND the only one consistently entered. Momentum-S still 0 entries despite ~96% accuracy. System's biggest missed money.

**Next review**: Track Trend-Dip-S unidirectional vs bidirectional separately. Counter-S entering 5th verified win — promote to Confirmed.

---

## 両建て戦術メモ

- OANDA v20ヘッジ口座: 大きい方にフルマージン。反対側は追加0
- H1テーゼLONG維持 + M5 StochRSI=1.0でSHORT回転 → プルバックで利確 → LONGだけ残る

---

## Event Day + Squeeze Patterns

- **Pre-event window**: Only the last 90min is true danger zone (4/16: EUR_USD -140 JPY in pre-GDP squeeze). Outside that, trade normally. Trailing stops get clipped by pre-event noise → fixed SL or nothing
- **Pre-event entry**: LIMIT at structural level only. No market-order RANGE scalps within 90min of binary event. Verified: 2x (4/3+4/16)
- **BB Squeeze resolution**: M5 squeeze + M1 3+ directional bodies → 1000-2000u, TP at opposite BB. Fixed SL (no trail). 15-60min hold. Rhythm trade, not conviction trade

---

## Thin Market / Holiday Rules (HARD RULES)

- **Holiday/thin → NO SL or ATR×2.5+**. 4/3: -984 JPY from noise hunts, all theses correct. Spread>2× normal = SLs don't work
- **User "SLいらない" = direct order.** Don't re-add. Don't close on own judgment
- **Pre-event trailing = trap.** Fixed SL or nothing. Trail kills 10h before events

## メンタル・行動

### ✦ 最重要原則：入らないことが最大のリスク

- **BAD TRADE avg cost: -354 JPY. DROUGHT day cost: -12,569 JPY (missed target). Fear is 36× more expensive than action.**
- **「理由があるから入らない」は常に成立する。** 7ペアすべてに「入らない理由」を見つけるのは簡単。全ペアにイベント/スプレッド/テーマ遅延/スクイーズのどれかがある。**入る理由を探せ、入らない理由を探すな。**
- **LIMITは入ったフリ。** LIMIT置いて→分析して→キャンセル→置き直し。これを繰り返してセッション消費するのは「リスクを取らずにトレーダーのフリ」。S/A確信 = マーケットオーダー。LIMITは構造レベルでの待ち伏せだけ

### ✦ 攻めの原則

- **Conviction=Size。二重割引するな。** pretrade_checkのWR%でサイズを落とすな。WRはすでにスコアに織り込み済み。Sと判定したなら30%NAV。pretrade=S(8)→sized_down は金をドブに捨てる行為（過去7回で6,740-13,140JPY損失）
- **最低3,000u。** 1-2kサイズ330本で-23,098 JPY。証明済み。入る価値がないなら入るな
- **S/A conviction = マーケットオーダー。** LIMIT「3pip節約したい」→ 刺さらず20-50pip逃す
- **0%マージンはありえない。** 7ペア×4TF=28の視点がある。全部で何もない日はない。ない＝見てない
- **BがSに昇格するパターンが最重要**: 最初の2指標でB判定→止まるな。別カテゴリの指標を見たらSだった、が過去5回起きてる
- **同一ペアのclose→re-enterを繰り返すな（チャーン）。** 4/7 AUD_JPY: 3回往復でスプレッド9.6pip燃やして-738JPY。最初のSLで決着をつけろ。**4/7 GBP_USDも同パターン: 4エントリー3敗-1,006JPY。2連敗したら別ペアへ**
- **3/31を基準にしろ**: 19エントリー, avg 4,737u, 65%WR, +4,591JPY。これが目指す姿
- **Quality audit flags ≠ trade orders**: 4/8 AUD_USD 8000u "S-candidate fix" → -2,540 JPY. Audit says "you missed this opportunity." That's information, not an instruction to enter. Your own pair-level analysis (AUD_USD 50%WR, avg-78JPY = no edge) should override audit pressure. Audit-driven S-sizing on a losing pair = worst of both worlds

### ✦ 守りの原則（知っていればいい。恐れるな）

- 慌てて損切りが最大の敵。テーゼ判断、金額判断するな。「今フラットならここで入るか？」Noなら閉じろ
- 2連敗→別ペアへ。全ポジ同方向=信者。指標は過去、チャートは今
- 含み益は取れ。「動き切った後」に同方向で入るな。ユーザーがHOLDと言ったらHOLD

## Consolidated Observations (4/13-4/16)
- [4/13-14] **RANGE mechanics**: SL just outside BB boundary (+5pip), not round numbers. Fib N=BEAR overrides LONG LIMITs in squeeze. Chart overrules oscillator extremes during momentum. Verified: 1x each
- [4/15] **Partial close always available**: OANDA supports any unit size. 750u HALF_TP locked +115.5 JPY. Never dismiss HALF_TP as "impractical." Verified: 1x
- [4/15] **Event timing verification**: state.md had wrong time (Bailey@18:00Z vs actual Taylor@19:00Z). Always verify against news_digest. Verified: 1x
