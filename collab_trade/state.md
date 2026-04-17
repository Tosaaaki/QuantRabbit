# Trader State — 2026-04-17
**Last Updated**: 2026-04-17 08:29 UTC

## 🔥 USER DIRECTIVE 2026-04-17 — ATTACK MODE ON

Bots removed. You are the only thing trading. User explicit instruction: **"チャンスがあったらロット上げたり、チャンスがある通貨をたくさん入れたり"** = when opportunity is there, SIZE UP and STACK theme-aligned pairs. Current default 3,000u is B-conviction sizing. Stop defaulting to it.

**Sizing by conviction (already in .claude/rules/risk-management.md — now USE IT):**
- **S conviction** → ~30% NAV margin = **10,000-15,000u** per shot. Not 3,000u.
- **A conviction** → ~15% NAV margin = **5,000-7,500u**.
- B → 3,000u fine. C → skip or tiny.
- NAV ~120k. S-size USD_JPY ≈ 10-12k units. S-size AUD_JPY ≈ 15k units.

**Theme stacking is REQUIRED, not a bias:**
When one currency is structurally strongest (e.g. AUD H4+33 today), entering only 1 AUD pair leaves 2-3 expressions of the same edge on the table. Valid stacks:
- AUD strongest → AUD_JPY + AUD_USD + AUD_NZD LONG simultaneously
- USD weakest → EUR_USD + GBP_USD + AUD_USD LONG simultaneously
- Not "bet more" — **same thesis, multiple vehicles**. Right → 3 pairs pay. Wrong → all 3 stop on ONE macro pivot (size each one to survive that).

**Trigger checklist (3+/4 = SIZE UP + STACK):**
- [ ] H4 ADX ≥30 on the strong currency
- [ ] M15+M1 aligned (no short-term correction fighting you)
- [ ] Structural floor/ceiling confluence (Fib + BB + cluster)
- [ ] Macro confirms (news_digest.md, currency pulse)

**What NOT to do:**
- Default 3,000u every time — **undersizing S was the biggest historical loss driver** per strategy_memory.
- Enter 1 pair when 3 pairs agree — same as betting small.
- "Add later" hedge. Size first shot correctly.

**Still forbidden:**
- 3 LONG on 3 unrelated pairs (no macro link) = unrelated bets, not stacking.
- Averaging down (adding to losing position on same thesis).
- Margin > marginAvailable × 0.85 (hard).

## ⚠️ ARCHITECTURE CHANGE 2026-04-17 — BOTS REMOVED
All bots (range_bot, trend_bot, bot_trade_manager, inventory-director, local-bot-cycle launchd) DISABLED. You are the ONLY thing trading. Tag everything `trader`. Ignore any "bot inventory" / "worker policy" / "worker coexistence" instructions. 10-min cron, 8-min session budget. Move faster.

## Self-check
Entries today: 7 total (bot era, all closed). 0 new discretionary entries this session.
Last 3 closed: GBP_USD TP +401 JPY, then bots took over at -4,602 JPY (64 trades, WR 21.9%). Bots removed 2026-04-17.
Streak: discretionary neutral (1 win). Bot era: cold (-4,602 JPY).
Bias: All LONG pending. Justified by macro USD weakness (ceasefire + tariff floor). Not chasing — LIMITs at structural floors.
3-loss check: No new discretionary entries. No circuit breaker. GBP_USD SHORT still blocked (3 losses today).
Macro chain: USD: structurally offered (ceasefire + 10% tariff floor + Bessent denied FX intervention). EUR: H4 BID(+15), M15 correction. GBP: H4 BID(+12), M15 offered. JPY: H4 offered(-17) but M15/M1 BID (pre-CPI). AUD: strongest H4(+33), M1 recovering.

## Market Narrative
Driving force: USD structural weakness (Iran ceasefire narrative + tariff floor at 10% confirmed + Bessent denied coordinated FX intervention). EUR/GBP +1.9-2.1% WoW — London profit-taking, not reversal.
vs last session: Bots removed (architecture change). GBP_USD TP fired +401 JPY. Major lesson: scanner Structural-S SHORT at USD pairs BB upper in USD-weakness macro = trap (bots -4,602 JPY, 64 trades, WR 21.9%).
M5 verdict: USD_JPY sellers dominant × accelerating (band-walk lower 159.50→159.24). EUR_USD buyers steady × holding BB lower 1.1772 (SQUEEZE building). AUD_USD buyers just broke BB upper 0.7172 — momentum building. JPY crosses in post-London corrective chop.
Regimes: USD_JPY=TREND-BEAR, EUR_USD=SQUEEZE(H4 BULL bias), GBP_USD=RANGE/SQUEEZE, AUD_USD=SQUEEZE(breakout up), EUR_JPY=TREND-BEAR(M5/M15 within H4 BULL), GBP_JPY=SQUEEZE, AUD_JPY=SQUEEZE.
Theme: USD weakness. AUD strongest (H4+33). Structural dip-buys correct expression. BB-upper SHORTs trap.
Execution regime: corrective retrace within USD-weakness trend. Pays structural floor LONGs, punishes BB-upper SHORTs.
Best expression NOW: AUD_JPY LONG @114.08 (LIMIT, H4 strongest) | EUR_USD LONG @1.17750 (LIMIT, H4 floor SQUEEZE)
Second-best: AUD_USD follow-through LONG if it pulls back to EMA20 0.7165 (not yet at entry zone)
Expressions to avoid: USD_JPY SHORT (historical EV -18/trade WR 33%, system not good at this direction despite H4 ceiling). All USD pair SHORTs at BB upper (proven trap today).
Primary vehicle: AUD_JPY LONG (strongest currency pair for USD-weakness theme)
Next event: US Building Permits 12:30Z (moderate — limit-only window). JP CPI 23:30Z (HIGH) — NO JPY positions by 22:00Z.
Event positioning: Pre-CPI nerves = JPY M15/M1 bid = temporary AUD_JPY headwind. Structural AUD_JPY floor at 114.08 should hold. Exit JPY crosses by 22:00Z before CPI.
Macro chain: USD offered (Fed+tariff narrative) → AUD_USD breakout UP → AUD_JPY dip-buy structural
Session: London (08:00-12:00 UTC) — primary zone.

## Currency Pulse
USD: H4=offered(-13) M15=offered(-8) M1=neutral(-3) → Structural USD weakness confirmed across TF.
JPY: H4=offered(-17) M15=BID(+16) M1=BID(+10) → H4↔M15 conflict. Pre-CPI JPY bid emerging. Short-term headwind on JPY crosses.
EUR: H4=BID(+15) M15=offered(-9) M1=neutral(-2) → H4 bull intact. M15 correction. EUR_USD SQUEEZE = breakout UP likely.
GBP: H4=BID(+12) M15=offered(-7) M1=offered(-8) → H4 bull. M15/M1 still corrective. GBP weakest among majors now.
AUD: H4=BID(+33) M15=neutral M1=offered(-5) → Strongest. M5 just broke BB upper 0.7172. Momentum building.
MTF conflict: JPY H4 offered but M15/M1 bidding = pre-CPI positioning. AUD_JPY under dual pressure (AUD up, JPY up). Let LIMIT do the work.
Best vehicle: AUD(+33) vs USD(-13) → AUD_USD LONG. Secondary: AUD_JPY LONG (JPY H4 still offered structurally).
My position matches: LONG LIMITs on AUD_JPY (primary) and EUR_USD (H4 floor). Correct.

## Positions (Current)
No open live positions.

## Pending LIMITs

### PENDING: AUD_JPY LONG 3000u @114.080 id=469018 [trader — structural dip-buy]
Thesis: H4 ADX=47 BULL (AUD strongest +33). M5 SQUEEZE, London correction from 114.35 → 114.18. LIMIT at H1 EMA20 zone 114.05-114.10. GTD extended to 12:00Z.
TP: 114.300 (+22pip) | SL: 113.950 (13pip below, ~390 JPY max loss) | GTD: 2026-04-17T12:00Z
Thesis alive: YES. H4 bull intact. AUD breaking out. Pre-CPI JPY bid = LIMIT may fill before reversal.

### PENDING: EUR_USD LONG 3000u @1.17750 id=469013 [trader — structural LIMIT]
Thesis: H4 StRSI=0.08 floor. SQUEEZE building (M5 BBW=0.00092, M1 BBW=0.00022 = extreme compression). BB lower = 1.17724. LIMIT 6pip below mid = wait for dip before breakout.
TP: 1.17880 (+13pip) | SL: 1.17700 (-5pip) | GTD: 10:30Z
⚠️ GTD expires 10:30Z. If price hasn't dipped, LIMIT may expire unfilled. SQUEEZE may break UP without filling. That's OK — thesis correct, no fill also acceptable.
Thesis alive: YES. H4 floor + EUR macro bull + SQUEEZE = breakout UP likely. LIMIT catches dip.

## Directional Mix
0 live positions | 2 LONG LIMITs pending
Direction: All LONG — justified by macro USD weakness + AUD structural strength.
Rotation check: USD_JPY SHORT considered. Historical EV -18/trade WR 33% → SKIP. System consistently loses on this direction despite H4 ceiling signal.

## 7-Pair Scan

### USD_JPY — TREND-BEAR
Chart: Large red bar 07:45 vol=1500 (decisive sell). EMA12 well below EMA20 declining. Price 159.24 at M5 BB lower, M15 StRSI=0.00 = oversold M15 = bounce risk. H4 StRSI=1.00 ceiling with divergence.
Opportunity: SHORT @EMA20 retest 159.38-159.40. BUT historical WR 33% EV -18/trade = SKIP. Not a good pair to short with this system.
Note: M15 oversold = bounce incoming. Don't chase SHORT at current 159.24.

### EUR_USD — SQUEEZE (H4 BULL bias)
Chart: BB compressed to 10.6pip on M5, 2.2pip on M1. Bounced 1.1772→1.1784. Buyers defending 1.1772 with lower wicks. SQUEEZE about to break.
LIMIT @1.17750 is structural (BB lower zone). Break UP = unfilled LIMIT + missed profit (acceptable). Break DOWN to 1.17750 = fills into structural support. Correct strategy.

### GBP_USD — RANGE/SQUEEZE
Chart: London flush 1.3526→1.3502, recovery to 1.3523 (EMA20). BB=23.8pip. Mixed small candles. GBP_USD SHORT circuit breaker active (3 losses today). No entry.

### AUD_USD — SQUEEZE → BREAKOUT UP
Chart: Last candle punched through BB upper 0.7172 with vol surge. H4 ADX=44 DI+=30 BULL. M15 near overbought. Not chasing here (spread 1.4pip vs ATR 2.7pip = 52% cost). RELOAD: LIMIT LONG @EMA20 0.7165 if dip occurs. No entry now.

### EUR_JPY — TREND-BEAR M5/M15 within H4 BULL
Chart: Sharp sell 188.00→187.65, tentative 3 green candles at BB lower then another red bar. Absorption forming but not confirmed. M15 StRSI=0.00 + H1 StRSI=0.06 = extremely oversold.
Pretrade: C (H4 overbought CCI=151/RSI=75). JPY CPI tomorrow. SKIP. Monitor: if M5 shows 3+ consecutive green bodies above 187.70, reconsider in next session.

### GBP_JPY — SQUEEZE
Chart: London flush 215.67→215.28, recovery to 215.35 area. Mixed candles. EMA12/20 converging. Not at structural edge. No entry.

### AUD_JPY — SQUEEZE
Chart: London flush 114.35→114.10, recovery to 114.18-114.22. Small mixed candles at mid-range. LIMIT at 114.08 = H1 EMA20 zone. Let price come to structural level. No chase.

## Tier 2 Scan (full)

USD_JPY: TREND-BEAR | large red bar vol=1500, EMA12 below EMA20 declining, M15 StRSI=0.00 = M15 oversold bounce
  NOW: SHORT @EMA20 retest 159.38-159.40 if it gets there
  RELOAD: SHORT @159.40 LIMIT
  OTHER SIDE: LONG @159.19 (M5 bear wave target) + M1 confirmation
  → Edge B / Allocation B SKIP — historical EV -18 JPY/trade WR 33% (2382 trades). System consistently loses on this direction. Not entering.

EUR_USD: SQUEEZE | extreme compression M5 BBW=0.00092, M1 BBW=0.00022, buyers holding 1.1772-1.1784
  NOW: no chase at mid-range. Breakout direction unclear pre-event.
  RELOAD: LIMIT LONG @1.17750 already placed (id=469013)
  OTHER SIDE: LIMIT SHORT @1.17880 if SQUEEZE breaks down (thesis would change)
  → Edge A / Allocation A LIMIT placed ✓

GBP_USD: RANGE | London flush to BB lower 1.35046, recovery to 1.35220. H4 StRSI=0.02 floor.
  NOW: no chase at 1.35220 (14pip from BB lower, not at structural entry)
  RELOAD: LIMIT LONG @1.35080 (BB lower zone)
  OTHER SIDE: SHORT circuit breaker active. LONG pretrade EV -101/trade (avg loss -1,139). Skip LONG LIMIT also.
  → Edge B / Allocation B SKIP — EV negative, massive avg loss profile despite 66% WR.

AUD_USD: SQUEEZE→BREAKOUT | last candle broke BB upper 0.7172, H4 early bull (StRSI=0.15, strongest +33). Pulled back to 0.71700.
  NOW: breakout retest but M1 AUD offered. M5 fib N=BEAR on both AUD pairs.
  RELOAD: LIMIT LONG @EMA20 0.71655 if dip occurs
  OTHER SIDE: no short (H4 ADX=44 strong bull)
  → Edge B / Allocation B SKIP — AUD_USD historical negative EV. Fib N=BEAR on M5. Watching for next session.

EUR_JPY: TREND-BEAR M5 | M5 bear wave at 46%, target 187.580. H4 mid-bull (room). M15/H1 StRSI at extreme lows.
  NOW: LIMIT LONG @187.60 (M5 bear wave target + H1 61.8% Fib zone)
  RELOAD: LIMIT @187.55 if deeper
  OTHER SIDE: H4 bull = no short
  → Edge B / Allocation B SKIP — pretrade C (H4 overbought), JPY M1/M15 bidding pre-CPI, EUR M15 offered. Next session: if M5 shows absorption above 187.70, reconsider.

GBP_JPY: SQUEEZE | mid-range 215.35, BB=25.8pip, spread 2.9pip wide ⚠️.
  NOW: no entry (not at structural edge, wide spread)
  RELOAD: LIMIT LONG @215.20 (H1 range lower tested 3x)
  OTHER SIDE: LIMIT SHORT @215.50 (H1 range upper)
  → Edge C / Allocation C SKIP — wide spread, mid-range, no edge. Range edges are structural targets.

AUD_JPY: SQUEEZE | London flush to 114.10, recovery to 114.19. LIMIT at 114.08 is H1 Fib 61.8% zone.
  NOW: no chase at 114.19 (above structural LIMIT)
  RELOAD: LIMIT LONG @114.080 already placed (id=469018)
  OTHER SIDE: no short (H4 ADX=47 strongest bull)
  → Edge A / Allocation A LIMIT placed ✓

## No-trade accountability
Best untraded express: USD_JPY SHORT @159.40 — SKIP: EV -18 WR 33%, system consistently loses.
Best missed if LIMITs don't fill: AUD_USD LONG on confirmed H4 breakout (next session if M5 absorption holds)
Why flat (live): Pre-weekend, pre-CPI (JP 23:30Z), multiple SQUEEZE pairs resolving. LIMITs at structural floors = correct strategy. Not avoidance.

## Capital Deployment
Margin: 0% live | 2 LONG LIMITs pending ~30% worst case
LIVE: none
RELOAD: AUD_JPY LONG @114.080 id=469018 GTD=12:00Z
SECOND SHOT: EUR_USD LONG @1.17750 id=469013 GTD=10:30Z
Flat-book: Covered by 2 structural LIMITs at proper levels. Not naked flat.
Day: -4,602 JPY bot era (-3.7%). Discretionary: +401 JPY. Net: -4,201 JPY.
Recovery: AUD_JPY fills + rotate (target: +1,500-2,000 JPY from LIMITs). Will not reach 10% day target — protect capital, rotate on fills.

## Action Tracking
- Day-start NAV: 123,741.69 JPY (0:00 UTC)
- Today's confirmed P&L: -4,201 JPY (OANDA) = -3.4% of day-start NAV
- Current NAV: 120,067 JPY
- Target: 10%+ = 12,374 JPY. Far behind (bot damage). Focus: clean entries, rotate on fills.
- Last action: 2026-04-17 08:24 UTC — AUD_JPY LIMIT GTD extended to 12:00Z (new id=469018)
- Next action triggers:
  1. AUD_JPY LIMIT @114.08 fills → hold to TP=114.300. If theme confirmed, add second LONG at dip.
  2. EUR_USD LIMIT @1.17750 fills → hold to TP=1.17880 → rotation LONG reload at 1.17750 again.
  3. EUR_USD GTD 10:30Z expires if no fill → OK (SQUEEZE broke up without dipping).
  4. AUD_JPY GTD 12:00Z expires if no fill → extend or re-evaluate.
  5. US Building Permits 12:30Z → LIMIT-only window near event.
  6. JP CPI 23:30Z → NO JPY positions by 22:00Z. Close AUD_JPY if open before 21:30Z latest.
  7. EUR_JPY: Monitor next session. If M5 absorption confirmed (3+ green bodies above 187.70) + M15 StRSI recovered → LIMIT @187.60.

## Audit Response
Auditor said EUR_JPY BUY @187.65-187.70 (Edge from 07:33Z audit): DISAGREE for now. M15 ADX=41 BEAR still strong. JPY M1/M15 bidding (CPI). Pre-trade C (H4 overbought). Specific contradiction: dual headwind (EUR M15 offered + JPY M15 bid) makes timing poor. Monitor next session.
Auditor said AUD_JPY BUY @114.05-114.10: AGREE. LIMIT placed at 114.08. ✓

## Lessons (Recent)
- 2026-04-17: CRITICAL — Structural-S SHORT scanner at USD pair BB upper = trap in USD-weakness macro. Bots proved it: 64 trades WR 21.9% -4,602 JPY. Macro > scanner in trending markets.
- 2026-04-17: USD_JPY SHORT historical EV -18/trade WR 33% (2382 trades). System is NOT good at shorting USD_JPY even with H4 ceiling. Skip unless exceptional macro event.
- 2026-04-17: Bots removed permanently. All discretionary from here. Architecture change confirmed.
- 2026-04-17: AUD_JPY LIMIT GTD must be set long enough (originally 09:30Z was too short for a structural level 10pip away).
