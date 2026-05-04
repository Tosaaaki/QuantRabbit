# Volume Profile, Market Profile & Order Flow for Retail FX (OANDA) — 2025-2026 State of the Art

## 0. Bottom Line Up Front (Brutal Honesty)

On retail FX you do **not** have real volume. OANDA tick volume = number of price changes per bar, not contracts traded. Aulicino (2003) and the well-known CME Hawkeye / Tom Dante studies showed tick volume on FX correlates ~0.85-0.95 with EBS/CME futures volume on liquid majors during London/NY hours, but the correlation collapses during Asia and around news. So volume profile on FX is **directionally useful as a price-acceptance map**, but anyone selling you "footprint charts" or "true delta" on MT4/OANDA is selling marketing. The signal that survives is **price-time acceptance** (TPO) and **HVN/LVN structure** — not delta, not absorption, not iceberg detection.

Useful for our trader: VPVR, VAH/VAL, POC, naked POC, HVN/LVN, IB, opening type, prior-day/week levels, candle-based order-flow proxies.
Marketing on retail FX: footprint, delta divergence, exhaustion via cumulative volume delta, COT-style retail positioning unless you accept it's a 24-hour-lagged contrarian gauge.

---

## 1. Volume Profile Primitives

| Primitive | Computation from OHLC+tickVol | Use case | Signal strength | False-signal mode |
|---|---|---|---|---|
| **VPVR (Visible Range)** | Bin price range into N price buckets (typ. 50-100). For each bar, distribute its `tickVolume` across buckets the bar's [low,high] touches (uniform or HLC3-weighted). Sum per bucket. | Macro structure on H1/H4/D | Medium-High | Garbage if Asia-only window or thin holiday |
| **VPSV (Session Volume)** | Same but bounded by session: Tokyo 00:00-08:00 UTC, London 07:00-15:00, NY 12:00-20:00 | Intraday playbook | High (London/NY) | Tokyo session is noisy on majors |
| **POC** | argmax bucket of VPVR | Magnet level, mean-reversion target | High | Migrates intra-session — use developing POC live |
| **Value Area (VAH/VAL, 70%)** | Sort buckets desc by volume, accumulate until ≥70% of total, take min/max price of that set. Standard "TPO 70%" rule (Steidlmayer). | Define balance zone | High | 70% is convention not law; try 68% (1σ) too |
| **HVN** | Local maxima in profile (peaks > neighbors by ≥X%) | Acceptance — price chops here | Medium | Lagging |
| **LVN** | Local minima — fast-traverse zones | Breakout acceleration / rejection from far side | High | Single-prints in low-liquidity hours are noise |
| **Single prints** | Buckets touched by exactly 1 TPO/bar | Tail of distribution; gets revisited | Medium-High | Common in Tokyo session — filter by session |
| **Naked POC (NPOC)** | A prior session's POC that has not been retraded since formation | Strong magnet within 3-5 sessions | High | Decays >2 weeks; ignore beyond |

**Computation recipe (Python):**
```python
def volume_profile(df, bins=100):
    pmin, pmax = df.low.min(), df.high.max()
    edges = np.linspace(pmin, pmax, bins+1)
    profile = np.zeros(bins)
    for _, b in df.iterrows():
        lo_i = np.searchsorted(edges, b.low) - 1
        hi_i = np.searchsorted(edges, b.high) - 1
        n = max(hi_i - lo_i + 1, 1)
        profile[lo_i:hi_i+1] += b.tickVolume / n
    poc_i = profile.argmax()
    order = np.argsort(profile)[::-1]
    cum, target = 0, profile.sum()*0.70
    va_set = []
    for i in order:
        cum += profile[i]; va_set.append(i)
        if cum >= target: break
    vah = edges[max(va_set)+1]; val = edges[min(va_set)]
    return profile, edges, edges[poc_i], vah, val
```

---

## 2. Market Profile (TPO) Primitives

TPO replaces volume with **time letters**: each 30-min bracket is a letter (A,B,C…) stamped at every price the bar touched. Volume-agnostic, so it works **better than volume profile on retail FX**.

**FX session mapping:**
- Most FX desks use **London + NY merged session** = 07:00-21:00 UTC, with letters A-Z+a-h on 30-min brackets.
- Some run **separate London (07-15 UTC) and NY (13-21 UTC) profiles** for EUR_JPY/GBP_JPY because Tokyo afternoon adds noise.
- Daily profile reset at 22:00 UTC (NY close / Sydney open).

| Primitive | Definition | Usage |
|---|---|---|
| **Initial Balance (IB)** | High–Low of first two 30-min brackets (A+B) of session. London IB = 07:00-08:00 UTC. NY IB = 13:00-14:00. | Baseline range; breakout = initiative activity |
| **Range Extension (RE)** | Price prints beyond IB | RE up + holds = trend day setup |
| **Day Types** | **Trend day**: open near one extreme, close near the other, IB = small fraction of range. **Normal day**: range = 1-1.5x IB, value balanced. **Normal variation**: 1.5-2x IB with one-sided RE. **Double distribution**: two POCs, gap or single prints between (often London-then-NY profile splits). **Neutral day**: RE both sides, close in middle. | Pre-classify by 11:00 UTC to bias intraday tactics |
| **Open relative to prior value** | OAOR = Open Above Out of Range, OAOY = Above Y'day value but In Range, OARE = Above Range Extension level, OARY = Above Range, In Y'day Value | Strong directional read, see Dalton |
| **Open Type** | **OD** (Open Drive): aggressive directional from bell, no IB rotation back. **OTD** (Open Test Drive): tests opposite side of open, then drives. **OAIR** (Open Auction In Range): rotates inside prior value. **OAOR** (Open Auction Out of Range): rotates outside but accepts/rejects | Highest-conviction read in first 60 min |
| **Value migration** | Today's VA shifted vs yesterday: Higher value = bullish auction; Overlapping = balance; Lower = bearish | Daily bias |

Reference: J. Dalton, *Mind Over Markets* (2nd ed.); *Markets in Profile*. CME's "Profile of a Day" series.

---

## 3. Auction Market Theory (AMT) — Computable Subset

| Concept | Retail FX computable? | How |
|---|---|---|
| Balance vs Imbalance | Yes | VA width vs ATR-N; overlap with prior VA |
| Rotation (price within VA) vs Migration (VA shifting) | Yes | Track VAH/VAL day-over-day |
| Acceptance | Yes | Price spends ≥2 TPO brackets at level |
| Rejection | Yes | Wick + return inside VA within 1-2 brackets |
| Two-time-frame trader | Partially | Compare H1 trend vs M5 rotations |
| Responsive (fade extremes) vs Initiative (break extremes) | Yes | Position relative to prior VAH/VAL + IB |
| **Long-term vs Other-time-frame participation** | No | Requires real volume / COT |

We can implement everything **except** the bit that needs real participant identification.

---

## 4. Order-Flow Proxies on Retail FX

### Tractable on OANDA OHLC + tickVolume

| Proxy | Recipe | Strength | Failure mode |
|---|---|---|---|
| **Tick-volume CVD proxy** | `delta = tickVol * sign(close-open)`; cumulative sum | Medium | Asia hours / news spikes |
| **Pressure ratio** | `(close-low)/(high-low)` per bar (close position) | Medium | Doji bars |
| **Body/Range ratio** | `\|close-open\|/(high-low)` | Medium | — |
| **Wick asymmetry** | `upper_wick - lower_wick` normalized by ATR | Medium-High | Mean-revert traps |
| **Volume climax** | tickVol > 2σ of 50-bar mean + reversal candle | High | News-driven false climax |
| **Liquidity grab / stop run** | New M15 swing high broken by ≤3 ATR, then close back inside, with tickVol spike | High at HVN/swing levels | Random in chop |
| **Absorption** | Tight range bar with elevated tickVol at HVN | Medium | Tick-vol noise in Tokyo |
| **Effort-vs-result** | High tickVol, small body | Medium | — |

### NOT tractable on OANDA without real-volume feed
- True footprint (bid/ask volume per price level)
- True cumulative delta
- Iceberg detection
- Real absorption (need bid-ask depth)
- DOM imbalance — OANDA's order/position book endpoints exist but require special access

---

## 5. Higher-Timeframe Context

| Level | Source | How to use |
|---|---|---|
| Weekly POC / VAH / VAL | D1 bars × tickVol over Mon-Fri | Macro magnet; trade toward unfilled NPOC |
| Daily POC / VAH / VAL | M30 or H1 bars by 22UTC-22UTC day | Today's bias = open relative to Y'day VA |
| Prior Day H/L (PDH/PDL) | Trivial | Strongest intraday levels on FX (Tom Dante, Mike Bellafiore writings) |
| Prior Week H/L (PWH/PWL) | Trivial | Weekly auction extremes |
| Opening drives | First 5/15/30 min | OD identification |
| Trend day flag | If 11:00UTC range > 1.2× 5d-avg-IB and price > IB high | Suspend mean-revert |

**Combination logic:**
```
context = {
  weekly_npoc, weekly_vah, weekly_val,
  daily_poc_y, daily_vah_y, daily_val_y, pdh, pdl,
  open_relation,    # OAOR/OAOY/OARE/OARY
  open_type,        # OD/OTD/OAIR/OAOR
  ib_high, ib_low,
  developing_poc, developing_vah, developing_val
}
```

---

## 6. Signal Strength Ranking (composite, on retail FX, 1-10):
1. PDH/PDL — 9
2. Naked POC <5 sessions old — 8
3. Open Type (OD especially) — 8
4. Weekly VAH/VAL — 8
5. HVN/LVN structure on H1 — 7
6. Daily POC migration — 7
7. IB break + RE — 7
8. Single prints in London/NY — 6
9. Tick-volume climax — 6
10. Tick-volume CVD proxy — 4 (high false-positive in Asia)
11. Retail position book contrarian — 4 (when accessible)

---

## 7. 2024-2026 State-of-the-Art

**Consensus 2025-2026:** Volume profile / TPO on retail FX **is useful as a structural map**, not as an order-flow read. Use it for **levels** (POC, VAH, VAL, NPOC, HVN, LVN), use TPO for **day-type classification**, and treat tick-volume CVD as a low-weight confirmation only.

---

## 8. Implementation Priorities for QuantRabbit

**Phase 1 (immediate, OHLC+tickVol only):**
- VPVR / VPSV with POC, VAH, VAL on D1, H1, M30
- TPO profile with IB, day-type classifier, open type/relation
- PDH/PDL/PWH/PWL/NPOC tracker (5-session lookback)
- HVN/LVN detector

**Phase 2 (low-weight confirmation):**
- Tick-volume CVD proxy
- Wick asymmetry, body/range, close-position bar features
- Volume climax & absorption detector

**Phase 3 (only if data accessible):**
- OANDA position book retail-contrarian (currently 401)

**Skip entirely:**
- Footprint / bid-ask delta (no data)
- Iceberg detection (no data)
- Anything sold as "smart money" indicator

---

## Sources

- https://developer.oanda.com/rest-live-v20/instrument-ep/
- https://www.oanda.com/forex-trading/analysis/open-position-ratios
- https://www.fxcm.com/markets/insights/tick-volume-and-its-relevance-in-forex/
- https://www.tradingview.com/support/solutions/43000502040/
- https://www.cmegroup.com/education/courses/market-profile.html
- J. Dalton, *Mind Over Markets* (2nd ed., Wiley); *Markets in Profile* (Wiley)
- P. Steidlmayer, *Steidlmayer on Markets*
- M. Bellafiore, *The PlayBook* (Wiley)
- Trader Dale, *Volume Profile* (2019, plus 2024 YouTube updates)
