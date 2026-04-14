# Strategy Memory — プロトレーダーの経験知

**daily-reviewが毎日更新。traderが毎セッション冒頭で読む。**

最終更新: 2026-04-14 21:00 UTC (daily-review)

---

## Confirmed Patterns（3回以上検証済み — ルールとして扱え）

### ✦ 勝ちパターン（これが金を生む — 恐怖より先に読め）

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

### ✦ 負けパターン（認識した上で恐れすぎるな）

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
- [4/7] **Pretrade HIGH doesn't mean regime-aligned**: GBP_USD SHORT HIGH(8) → -317 JPY. EUR_USD SHORT HIGH(8) → -288 JPY. Both were in a USD-weak week. **Pretrade score is direction-agnostic — it doesn't know the current regime. A high score on either direction doesn't guarantee the market is going that way.** This applies equally to LONGs in a bear market. Verified: 1x (2 trades same day)
- [4/7] **AUD_JPY churn: 3 round-trips = -778 JPY + 9.6pip spread burned**: LONG 3500u→-217(6min), LONG 5000u→+40(170min BE), SHORT 3500u→-556, SHORT 4500u→-4. Close-and-re-enter on same thesis is spread destruction. First SL should be the final decision. Don't re-enter same pair within 30 min unless thesis genuinely changes. Verified: 1x (matches 4/6 AUD_JPY pattern = 2x total)
- [4/7] **Winners held 127 min avg, losers held 13 min avg**: Patience = profit. Quick cuts on positions that needed time to develop (GBP_USD LONG held 15-17 min before cutting) vs EUR_JPY LONG held 36-225 min for wins. Don't cut within 15 min unless structural invalidation. Verified: 2x (4/7 + 4/8: winners 334min avg vs losers 131min avg)
- [4/8] **Quality audit "S-candidate fix" pressure → oversized loss**: AUD_USD LONG 8000u entered as "quality audit miss fix" → -2,540 JPY. Quality audit flagged AUD_USD as missed S-candidate, trader entered 8000u (~S-size) on a pair with 50%WR LONG all-time. Audit flags are data, not orders. Don't let audit pressure override your own pair-level conviction. AUD_USD LONG avg=-78JPY all-time — this is NOT an S-conviction pair. Verified: 1x
- [4/8] **Counter-trades against H1 ADX>35 BULL = money pit**: AUD_JPY SHORT 1200u (-79JPY) + SHORT 1000u (-121JPY). Both entered on "H4+H1 StRSI=1.0 extreme" counter-thesis. H1 ADX=37-38 DI+ dominant invalidated both within hours. **H4 StRSI=1.0 does NOT override H1 ADX>35 BULL.** The stronger signal is the trend, not the oscillator extreme. Verified: 2x (both AUD_JPY SHORT same day). **Clarification**: This applies to SWING-SIZE counter-trades (4000u+, hours hold). M5 rotation SHORTs (2000-3000u, TP=ATR×0.5, 15-30min hold) within a LONG thesis are standard pullback trading — different concept, different risk
- [4/8] **Margin crisis repeat (97%) → forced close**: EUR_JPY LONG 4500u forced close at -319 JPY. 3 positions stacked + pending AUD_USD LIMIT (7000u×margin) pushed to 97%. Same pattern as 4/1. **Pending LIMITs must be in worst-case calculation. Above 85% = STOP ADDING.** Verified: 2x (4/1 + 4/8)
- [4/9] **Indicator depth determines whether you see both directions**: Checking 3 indicators → one direction locked. Checking 6+ indicators → both directions become visible. 4/8-4/9: 13 LONG, 0 SHORT with shallow scans. **The lesson is "scan deeper so you see the full picture" — not "force SHORTs." In a different market, shallow scan might lock SHORT-only and miss LONGs. Same problem, opposite direction.**
- [4/9] **Macro narrative overrode chart (USD_JPY)**: H4 DI-=31, H1 ADX=51 BEAR = clear SHORT chart. Trader tried LONG 3x (counter-bounce) → all lost. Never traded WITH the chart (SHORT). **The error was ignoring what the chart showed, not "failing to short." In a different regime the chart might say LONG and the lesson is the same: trade the chart.**
- [4/9] **S-scan recipe accuracy divergence — Structural-S best, Squeeze-S worst**: Today's audit_history analysis:
  - Structural-S: 3/3 = 100% (USD_JPY LONG @158.712→158.844 +13.2pip, EUR_JPY LONG @185.151→185.237 +8.6pip, AUD_JPY LONG @111.686→111.749 +6.3pip). All CORRECT
  - Trend-Dip-S: 3/12 = 25%. Fired contradictory signals (EUR_USD SHORT @02:03Z then LONG @03:33Z, USD_JPY LONG @04:33Z then SHORT @05:03Z). Too noisy — fires on both sides within 90 min
  - Squeeze-S: 0/4 = 0%. All 4 signals wrong direction. EUR_USD SHORT @1.16518 → price went UP +9.8pip. USD_JPY LONG @158.86 → stayed flat
  - **Action**: Trust Structural-S signals. Be skeptical of Squeeze-S and Trend-Dip-S that flip direction within 2h. Verified: 1x (19 signals today)
- [4/9] **Recovery day**: Morning SHORTs -2,971 JPY → afternoon LONG pivot net +1,045 JPY. Regime pivot mid-day was the key skill. Margin crisis 3rd time (98%) → promoted to Confirmed Patterns.
- [4/9] **Momentum-S recipe: 50+ fires, all LONG, market moved +50-60pip**: EUR_JPY @185.596→186.186(+59pip), GBP_JPY @213.147→213.708(+56pip), AUD_JPY @111.951→112.538(+59pip). All NOT_HELD. Largest missed opportunity in S-scan history. **Rule: When Momentum-S fires ≥3 pairs in same direction → regime signal. Enter strongest CS pair A-size minimum.** Verified: 1x
- [4/10-12] **Rollover_guard restore = systemic automation gap (2x verified)**: 4/10 AUD_JPY 8000u SL removed for rollover, restore never ran → held weekend with ZERO SL → -2,864 JPY by Monday. Same as 4/3 trail incident. **Root cause**: restore requires trader session but trader halts weekends. **Hard rule: rollover_guard remove + weekend approaching = MANDATORY immediate structural SL set. 45min overdue = close or set manual SL.** Verified: 2x (4/3, 4/12).
- [4/11] **Weekend Structural-S on JPY crosses = noise**: 4 fires Saturday. JPY crosses all WRONG (EUR_JPY -7pip, GBP_JPY -11.8pip). Counter-S GBP_JPY +11.8pip ✓. **JPY crosses on weekends dominated by headlines, not structure.** Now also confirmed 4/14 consolidation day: AUD_JPY -18.6pip, GBP_JPY -7.1pip. Verified: 2x.
- [4/13] **Weekend gap-down = averaging-down opportunity when thesis intact**: AUD_JPY gapped down 76pip over weekend (112.742→~111.98). User entered 10,000u @111.981 at Sunday open, closed @112.383 = +4,020 JPY. Old 8000u SL hit -2,768 JPY → net +1,252 JPY. **Weekend gaps on JPY crosses often overshoot due to thin liquidity + headline panic. If the structural thesis (AUD BULL / JPY WEAK) hasn't changed, the gap is a gift.** Conditions: (1) fundamental thesis unchanged, (2) gap is 50+ pip beyond recent range, (3) enter at open before reversion, (4) size up (S-conviction). Verified: 1x
- [4/13] **Weekend gap causes SL slippage through level**: AUD_JPY 8000u LONG SL was set at 112.550. rollover_guard.py restored protection but weekend gap-down caused fill at 112.380 — **17pip slippage below SL level** (bid 112.39 vs SL 112.550). Loss was -2,768 JPY vs expected ~-1,600 JPY at SL level. **Hard lesson: SL does NOT guarantee fill at your price on weekend gaps.** The SL triggers at the level, but the first available bid is often far below on thin Sunday open. On volatile weekends (geopolitical/tariff headlines), expected SL loss can be 1.5-2× the stated risk. Size accordingly when holding into weekends — treat expected loss as SL_size × 1.5. Verified: 1x
- [4/4] **Conviction-S undersizing = biggest silent profit killer**: 7 trades met S conditions, 5 entered at B-size. 6,740-13,140 JPY thrown away. Fix: "Different lens" in pre-entry format forces checking unused indicator categories. B→S upgrade is where the money is. Verified: 5x
- [4/14] **AUD macro weakness is pair-specific**: NAB=-29 → AUD_JPY TREND-BEAR but AUD_USD barely moved (USD weakest globally). Currency weakness shows up in different pairs depending on OTHER leg. Read CS table before assuming "AUD weak = all AUD pairs SHORT." Verified: 1x
- [4/14] **OANDA leverage 20:1 for non-JPY pairs (GBP_USD, EUR_USD), NOT 25:1**: Margin calc = units × (base_CCY_JPY / 20). Using /25 understates margin by 25% → 92% margin on "calculated 84%". Hard rule: For non-JPY pairs, always use /20. Verify with initialMarginRequired from OANDA response. Verified: 1x (GBP_USD 4500u: estimated 38,754 JPY vs actual 48,423 JPY)
- [4/14] **Always_C_hold culminated in GBP_JPY SL hit (-876 JPY)**: 5+ consecutive audit HALF_TP recommendations. Each time held (spread 12-20pip argument). Position SL hit at 214.900. **Auditor was right.** "Spread too wide" was valid short-term but became infinite hold excuse. **If TP doesn't fill within 3 sessions after spread normalizes → close at market.** Verified: 1x (outcome of 5-session pattern)
- [4/14] **Tokyo thin LIMIT fill → instant SL hunt**: GBP_JPY 4000u LIMIT @215.119 filled 01:35Z, SL @214.900 hit 20min later = -876 JPY. SL was 21.9pip = ATR×1.1, too tight for Tokyo. Same pattern as 4/3 Good Friday. **LIMIT in Tokyo thin (00:00-06:00Z) must have SL ≥ ATR×1.5.** Verified: 2x (4/3 + 4/14)
- [4/14] **Counter-S recipe: first entered → WIN +460 JPY**: USD_JPY LONG Counter-S @159.096. Validates Counter-S as entry-worthy at H4 extreme + chart supports counter direction. Cumulative 5 fires, 4 correct = 80%
- [4/14] **Pullback Quality Check deployed in profit_check.py**: S-conviction TP management upgraded. profit_check now outputs NOISE/SQUEEZE/DISTRIBUTION verdict using 12 underused indicators (ema_slope_20, chaikin_vol, bbw/kc ratio, wick patterns, cluster gaps, ROC, div_score, cross-pair). Existing recommendation logic unchanged — PQ is additive data. Root cause of S-trade profit truncation: 4/7 S-trades captured 25-30pip (trail ATR×1.5), recent S-trades only 12-14pip (trail ATR×0.6). Same conviction, half the profit. PQ enables conviction-aware trail width. Verification pending: track S-trade pip capture before/after

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
- **SHORT: 26% WR avg -1,044 JPY total -24,013 JPY (sample: 3/17-4/9, predominantly bullish period)** — poor stats but sample is biased by bull market. Don't treat as permanent property. In range/bearish regime, GBP_USD SHORT may be valid. Read the chart, not the table.
- **All-time LONG: 56% WR avg +25 JPY total +966 JPY** — modestly profitable but requires patience (avg winner hold >100 min)
- Tokyo session trailing stop trap same as EUR_USD: 15pip trail = ATR×0.7 got clipped (4/3 lesson)
- **4/7: repeated entries after failures compound losses.** 4 entries in one day, -1,006 JPY net. 5000u entry after 2 prior losses = chasing. After 2 GBP_USD losses, move to another pair

### EUR_USD
- スプレッド最小。スキャルプに最適
- VWAP+35pip乖離での追撃 → 過熱からの反転で負け
- ロンドンクローズ(16:00 UTC)で方向変わりやすい
- **All-time LONG 53%WR avg +160 JPY total +11,980 JPY** — best total P&L of any pair/direction by far. EUR_USD LONG is the strongest edge in the system. 4/8: +4,575 JPY (2/2 wins) further confirms.
- **SHORT 51%WR avg -156 JPY total -7,015 JPY (sample: 3/17-4/9, EUR bullish period)** — WR is fine, avg loss is large. Sample biased by EUR strength period. In EUR-weak regime, SHORT edge may appear. Chart-dependent, not pair-property.
- Trailing stop in Tokyo session (00:00-06:00Z): ATR×0.7 clips positions. Use ATR×1.2 minimum or hard SL only (4/3 lesson)

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

### GBP_JPY
- 30-50pipスウィングが普通。痛み上限-40pip
- 25pipの逆行で即切りは早すぎる
- **LONG 65%WR avg+140JPY total+2,794JPY** — strong LONG edge but vulnerable to Tokyo thin SL hunts. Always_C_hold pattern (5+ audit warnings ignored → SL hit -876 JPY on 4/14). LIMIT fills in Tokyo thin → SL ATR×1.5+ minimum

### EUR_JPY
- ボラ大きい。利確遅れやすいので注意
- **LONG 69%WR avg +140 JPY total +1,824 JPY** — highest WR and avg in the system. When EUR strongest + JPY weakest (macro alignment), this pair prints money
- **4/7: 6/6 LONG wins, +1,876 JPY** in one day. All pretrade=LOW yet all won. H1 ADX=35-43 trending strongly. Pretrade undervalues this pair in trending conditions
- Spread 1.7-2.3pip is high. Need 20+ pip target to justify entry. Sub-10pip scalps don't work here

---

## Pretrade Feedback（daily-reviewが自動更新）

### LOWが正しかった場面（避けるべきエントリー）

- [3/27] AUD_JPY/AUD_USD LOW → total -1,804 JPY (3 trades). AUD pairs at LOW = correct warning. AUD_USD SHORT 42%WR, AUD_JPY direction-change both consistently negative

### LOWが間違っていた場面（pretradeが保守的すぎるケース）

- [3/27] AUD_JPY LONG pretrade=LOW → +514円, +657円。AUD_JPY LONGは全期間67%WR。LOWでも勝てる組み合わせ
- [3/27] GBP_USD LONG pretrade=LOW(WR=50%) → まだ保有中、判定保留
- [4/7] **EUR_JPY LONG pretrade=LOW(1) → 6/6 wins, +1,876 JPY** — biggest single-day pretrade miss. All 6 EUR_JPY LONG entries scored LOW. Every one won. H1 ADX=35-43 trending, EUR strongest(+0.42-0.60), JPY weakest(-0.22 to -0.45). **Pretrade LOW in a trending pair with macro alignment = systematically wrong.** This is the strongest evidence yet that pretrade_check needs a "trending override" when H1 ADX>35 + CS alignment
- [4/7] GBP_USD LONG pretrade=LOW(1) → +93 JPY (1/3 LOW entries won). LOW was partly right here — pair was choppy. Not all LOWs are wrong, but EUR_JPY LOW in trend is clearly miscalibrated
- [4/9] **EUR_JPY LONG pretrade=LOW(1) → 2/2 wins +428 JPY**: same pattern as 4/7. Trending pair with CS alignment overrides LOW score. (4/7+4/9: 8/8 EUR_JPY LONG wins all at LOW — pretrade calibration needs trending override)
- [4/9] **GBP_JPY LONG pretrade=LOW(3) → 2/2 wins +319 JPY**: LOWs winning again in bull regime across multiple pairs.

### HIGHが外れた場面（スコア過信に注意）

- [4/3] EUR_USD LONG pretrade=HIGH(score=8) → -170 JPY. Score itself was reasonable (post-NFP USD weak thesis, H1 ADX=59 bull). **Cause was not the entry thesis but the trailing stop calibration**: 11pip trail = ATR×0.7, clipped by Tokyo session (02:22Z) noise before London even opened. Entry thesis was correct — execution/protection was wrong. **Lesson: trail width caused the loss, not the pretrade score. Don't attribute this to score inflation.**
- [4/7] GBP_USD SHORT pretrade=HIGH(8) → -317 JPY. Macro was TACO week, USD structurally weak. HIGH score didn't account for macro regime. **Lesson: HIGH on a direction that fights macro flow = false confidence**
- [4/7] EUR_USD SHORT pretrade=HIGH(8) → -288 JPY. Same macro environment. Both USD-long bets in USD-weak week. **Pretrade needs macro direction filter**
- [4/9] AUD_USD SHORT pretrade=HIGH(8) → -2,699 JPY. Pair has no edge (48%WR). HIGH score on AUD_USD = overconfidence. Score was based on trend alignment that didn't materialize
- [4/9] USD_JPY SHORT pretrade=HIGH(10) → -270 JPY. Perfect 10 score on a SHORT that went against the eventual LONG move. H1 DI flipped during hold. **10/10 is not infallible**
- [4/9] EUR_USD SHORT pretrade=HIGH(10) → -2 JPY. Near-breakeven. **3/3 HIGH SHORTs lost on 4/9 — but this reflects the regime (bullish), not a permanent SHORT bias. In a bearish regime, HIGH LONGs would fail the same way. Pretrade needs regime context for both directions.**

### MEDIUM scores — mixed bag (4/8)

- [4/8] AUD_USD LONG pretrade=MEDIUM(4) → -2,540 JPY. Score was MEDIUM but pair has no edge (50%WR, avg -78JPY). MEDIUM on a historically losing pair = should have been treated as LOW. **Pair-level all-time stats matter more than session-level pretrade score.**
- [4/8] USD_JPY LONG pretrade=MEDIUM(5) → -113 JPY. Counter-trade, small size (1000u), acceptable loss.
- [4/8] EUR_USD LONG pretrade=MEDIUM(6) → +2,200 JPY, +2,376 JPY (2 trades). MEDIUM on EUR_USD LONG = reliable. This pair elevates MEDIUM scores into wins.

### B/MEDIUM scores — recent

- [4/13] AUD_JPY SHORT pretrade=MEDIUM(4)/B-conv @112.400 → -152 JPY (closed 06:54Z, H1 RANGE structure changed). B-conviction, B-size (2000u). Score accurately reflected incomplete picture. Loss was controlled.
- [4/14] GBP_JPY LONG pretrade=MEDIUM(4) → -876 JPY. LIMIT fill in Tokyo thin 01:35Z, SL hit 21.9pip away. ATR×1.1 SL too tight for Tokyo. Score was accurate — MEDIUM correctly reflected incomplete picture.
- [4/14] USD_JPY LONG pretrade=MEDIUM(5) → +460 JPY. Counter-S recipe entry. Score undervalued — H4 extreme + chart alignment was A-quality. Counter-S entries deserve higher weight in pretrade

### B-score losses worth noting

- [4/3] GBP_USD LONG pretrade=B(no formal check) → -146 JPY. Same Tokyo trail trap as EUR_USD. GBP_USD 15pip trail = ATR×0.7, clipped 02:22Z. **B-score entries in Tokyo session are especially dangerous: low conviction + thin liquidity + aggressive trail = high noise-kill probability. Tokyo overnight entries should be sized to survive without trails.**

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
| 3/23 | EUR_USD | VWAP+35pip乖離追撃 | 負け | 乖離大=逆張り指標 |

| 4/7 | GBP_USD | M5 StRSI=0.03 pullback in H1 bull | LONG 5000u -480 JPY(17min) | H1 bull but M1 strong downtrend ignored = premature |
| 4/8 | AUD_JPY | H4+H1 StRSI=1.0 counter SHORT vs H1 ADX=37 DI+ BULL | Counter 0/2 -200 JPY | H4 osc extreme doesn't override H1 ADX>35 trend |

| 4/7 | EUR_JPY | H1 ADX>35 + StRSI=0.0 + CS alignment | M5+H1+Macro | 6/6 wins +1,876 JPY. Trending CS alignment = best edge |
| 4/8 | EUR_USD | H1 ADX>30 + StRSI dip + CS gap>1.0 + Iran risk-ON | M5+H1+Macro | 2/2 wins +4,575 JPY. Best 1-day EUR_USD |

### 未検証（今後試す）
- Keltner幅 + BB幅: 同時squeeze=ブレイク予測精度UP？
- Chaikin Vol + ATR変化率: ボラ急変の早期検知？
- cluster_gap + Ichimoku雲: サポレジの二重確認

---

## S-Scan Recipe Scorecard (daily-review tracks)

| Recipe | Fires (all-time) | Entered | Correct (±5pip) | Accuracy | Status |
|--------|-------------------|---------|------------------|----------|--------|
| Structural-S | 22 | 1 | 11/22 | 50% | **Degraded**. USD pairs reliable (GBP_USD +11.4pip 4/14), JPY crosses wrong on quiet/consolidation days. Split tracking needed. |
| Momentum-S | 55+ | 0 | ~53/55 | ~96% | Confirmed reliable on EUR/GBP USD pairs. 4/14: EUR_USD +6.3pip ✓, GBP_USD +9.1pip ✓. JPY cross signals (EUR_JPY -9.0pip) less reliable. |
| Trend-Dip-S | 16 | 4 | 5/16 | 31% | **Watch** — 4/14: EUR_USD +2.1pip ✓, GBP_JPY -6.6pip ✗. Still noisy, fires contradictory signals. |
| Squeeze-S | 8 | 1 | 1/8 | 12% | **Deprecated** — 1/8 after 8 fires. GBP_JPY LONG entered on Squeeze-S → SL hit -876 JPY. Remove from S-scan. |
| Counter-S | 5 | 1 | 4/5 | 80% | **Watch → Confirmed**. 4/14: USD_JPY LONG entered → +460 JPY WIN. First entered trade = success. Small sample but high accuracy. |

**4/14 details**: Structural-S 1/6 (GBP_USD +11.4pip ✓, AUD_USD -9.5pip ✗, AUD_JPY -18.6pip ✗, GBP_JPY -7.1pip ✗ — JPY crosses all wrong on PPI-wait consolidation day). Momentum-S 3/5 (EUR_USD +6.3pip ✓, GBP_USD +9.1pip ✓, EUR_JPY -9.0pip ✗). Counter-S: GBP_JPY SHORT +6.7pip ✓, GBP_USD SHORT -2.5pip ✗.
**Key finding**: Structural-S accuracy is bimodal — USD pairs (GBP_USD, EUR_USD) are reliable (~70%), JPY crosses (AUD_JPY, GBP_JPY, EUR_JPY) are unreliable (~30%) especially on consolidation/pre-event days.

**Next review**: Squeeze-S deprecated at 0/5. Track Structural-S USD vs JPY cross accuracy separately. Momentum-S → start entering on EUR/GBP USD signals (3 confirmed correct, 0 entered = missed opportunity).

---

## 両建て戦術メモ

- OANDA v20ヘッジ口座: 大きい方にフルマージン。反対側は追加0
- H1テーゼLONG維持 + M5 StochRSI=1.0でSHORT回転 → プルバックで利確 → LONGだけ残る
- ショート側は同量以下。超えたらマージン増加

---

## Event Day Experience (NFP, CPI, FOMC etc.)

**What we've learned about event days:**
- The market doesn't stop moving 8 hours before an event. Small waves, squeezes, and setups still happen
- "Pre-NFP" became a 10-hour standby on 4/3. The real danger window is the last hour, not the whole day
- Thin liquidity (Good Friday, holidays) amplifies spikes on the event candle. The 5-10 min after release are the most dangerous
- Trailing stops get clipped by pre-event noise. Fixed SL survives better close to events

**The thinking**: Before writing "no entries pre-event" in state.md, ask: "How many hours until the event? Is that really a reason to stop trading, or am I just nervous?" Nervousness is not analysis.

---

## Small Wave Patterns (BB Squeeze Resolution)

**Observation**: When all pairs show M5 BB squeeze simultaneously, the squeezes resolve into small moves — usually 5-15 pips. These are tradeable.

**What works**: M5 squeeze + M1 shows 3+ consecutive directional bodies → enter small (1000-2000u). TP at opposite BB band. Fixed SL (no trailing — trail kills small moves before they develop). Typically resolves in 15-60 minutes.

**What doesn't work**: Trailing stops on squeeze trades (4/3 EUR_USD, GBP_USD both clipped by trail). Holding for big targets on a squeeze play (it's a rhythm trade, not a conviction trade).

**When to use**: When margin is underutilized and you're tempted to write "standby." The market is always moving. The question is whether you're watching.

---

## Thin Market / Holiday Rules (HARD RULES)

**4/3 Good Friday: -984 JPY. Every single loss was noise stop hunting. Every thesis was correct.**

- **Holiday/thin liquidity → NO SL**. ATR×1.2 gets hunted when spreads are 2-3× normal. Either ATR×2.5+ or discretionary only
- **User says "SLいらない" → that's a direct order.** Don't re-add. Don't close on your own judgment. The user is managing risk
- **Pre-event trailing stops are a trap.** Trail added at 00:37Z "for NFP protection" killed EUR_USD and GBP_USD at 02:22Z — 10 hours before NFP. Use fixed SL or nothing
- **Spread > 2× normal = the market is telling you SLs don't work.** Listen to the market

## メンタル・行動

### ✦ 攻めの原則（恐怖より先に読め）

- **Conviction=Size。二重割引するな。** pretrade_checkのWR%でサイズを落とすな。WRはすでにスコアに織り込み済み。Sと判定したなら30%NAV。pretrade=S(8)→sized_down は金をドブに捨てる行為（過去7回で6,740-13,140JPY損失）
- **最低2,000u。** 500u/700uは勝ってもスプレッドに食われる。入る価値がないなら入るな
- **S/A conviction = マーケットオーダー。** LIMIT「3pip節約したい」→ 刺さらず20-50pip逃す。これが4/7で0%マージンの原因
- **0%マージンはありえない。** 7ペア×4TF=28の視点がある。全部で何もない日はない。ない＝見てない
- **BがSに昇格するパターンが最重要**: 最初の2指標でB判定→止まるな。別カテゴリの指標を見たらSだった、が過去5回起きてる
- **同一ペアのclose→re-enterを繰り返すな（チャーン）。** 4/7 AUD_JPY: 3回往復でスプレッド9.6pip燃やして-738JPY。最初のSLで決着をつけろ。**4/7 GBP_USDも同パターン: 4エントリー3敗-1,006JPY。2連敗したら別ペアへ**
- **3/31を基準にしろ**: 19エントリー, avg 4,737u, 65%WR, +4,591JPY。これが目指す姿
- **Quality audit flags ≠ trade orders**: 4/8 AUD_USD 8000u "S-candidate fix" → -2,540 JPY. Audit says "you missed this opportunity." That's information, not an instruction to enter. Your own pair-level analysis (AUD_USD 50%WR, avg-78JPY = no edge) should override audit pressure. Audit-driven S-sizing on a losing pair = worst of both worlds

### ✦ 守りの原則（知っていればいい。恐れるな）

- 慌てて損切りが最大の敵。テーゼで判断しろ、金額で判断するな
- 「今フラットだったらここで入るか？」→ Noなら閉じろ
- 1ペアに固執するな。7ペアある。2連敗したら別ペアを探せ
- ヘッドライン相場では追加LONG禁止。30分で局面が変わる
- [4/1] 全ポジ同方向=信者。方向分散しろ
- [4/1] 指標は過去、チャートは今。チャートの形を見ずに指標の数字で判断するのはボット
- [4/1] 含み益は取れ。市場がくれたものは受け取れ
- [4/1] 「動き切った後」に同方向で入るな。次はバウンス
- [4/3] ユーザー指示無視→パニッククローズ→慌てて入り直し = 最悪パターン。**ユーザーがHOLDと言ったらHOLD**

## Consolidated Observations (4/13-4/14)
- [4/13] Counter-S SHORT signal vs M5 raw momentum: expanding bullish bodies band-walking BB upper → counter-signals are noise. Chart overrules oscillator extreme. Verified: 1x
- [4/13] SQUEEZE at top of TREND-BULL: breakout direction = continuation. LIMIT at BB mid (pullback) valid but breakout entry cleaner
- [4/14] **RANGE R:R formula**: RANGE SL should be just outside range boundary (BB upper + 5pip), not round numbers. BB upper 112.795 → SL=112.85 (5pip) vs TP=BB lower 112.70 (9.5pip) = R:R 1.9:1 viable. Round number SL (113.00=16pip) killed valid trade. Verified: 1x
