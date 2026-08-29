# Fast-bot shock guard / protective-SL replay — 2026-08-29

## Authority and truth

- Diagnostic shadow evidence only. `execution_authority=NONE`, broker mutation 0, external order attempts 0, external orders 0.
- EUR/USD OANDA M1 bid/ask, 2020-01-01 22:00 UTC through 2026-07-09 plus a GET-only 2026-08-28 UTC slice.
- Historical rows: 2,398,149 (the prior inventory measured 97.901% expected-market-minute coverage). 2026-08-28 rows: 1,260. Combined deduplicated rows: 2,399,409.
- The implementation replay pins ATR to causal completed M5 ATR. This produces 3,020 non-overlapping 60-minute shock episodes; it is not the earlier 733-episode diagnostic whose ATR construction differed. All arms below use this same 3,020-episode set.
- Volume exists in the source rows but was unavailable as a reliable liquidity proxy and was not used. Spread is included through executable bid/ask entry and exit.

## Side-relative episodes

| Measure | Result |
|---|---:|
| Episodes | 3,020 |
| UP | 1,514 |
| DOWN | 1,506 |
| Failed continuation at 5m | 802 (26.56%) |
| 30m retrace >= 50% | 1,151 (38.11%) |

The 2026-08-28 exact central-cell episode was detected at 14:03 UTC: DOWN, -21.0 pips over 15 minutes, completed-M5 ATR 3.207143 pips, not failed at the five-minute rule, not retraced 50% within 30 minutes, and +5.1 pips in the original DOWN direction at 60 minutes. It is therefore a short-horizon continuation episode even though a later local V-shape occurred inside the wider decline.

## Defensive arms

| Arm | Trades | Net pips | PF | Hit | P05 pips | Max loss streak |
|---|---:|---:|---:|---:|---:|---:|
| Immediate continuation baseline | 3,020 | -5,245.300 | 0.786278 | 43.21% | -31.210 | 10 |
| New-entry shock freeze | 0 | 0 | n/a | n/a | n/a | 0 |
| Shock freeze + 50% bot-owned drain proxy | 3,020 | -2,622.650 | 0.786278 | 43.21% | -15.605 | 10 |

The freeze arm is loss avoidance, not a profitable strategy. It rejects 100% of new entries inside the detected shock band and 0% by construction outside the band. The drain arm halves both modeled loss and margin exposure; it is paper/shadow only and never touches manual/tagless inventory. Current `EUR_USD/RANGE_ROTATION/LONG` quarantine cannot be reconstructed from price-only M1 candle files because proposal method labels are absent, so no directional proxy was fabricated.

## Protective-stop geometry

TP is held at the existing 2.4-pip comparison value. Same-minute TP/SL is stop-first. Gap-through exits use executable open and record slippage; no SL is treated as guaranteed.

| Geometry | Net pips | PF | Hit | P05 | SL hit | Max consecutive SL | Post-SL re-entry loss | Median SL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 3.2 | -5,164.700 | 0.273713 | 26.85% | -3.200 | 73.15% | 20 | -5,141.100 | 3.200 |
| ATR 1.0 | -6,242.496 | 0.366533 | 49.83% | -9.475714 | 50.13% | 13 | -4,296.768 | 6.576786 |
| ATR 1.5 | -6,591.709 | 0.404936 | 61.89% | -13.361250 | 37.45% | 7 | -3,350.634 | 9.865179 |
| ATR 2.0 | -6,755.043 | 0.422722 | 68.28% | -16.679643 | 29.21% | 5 | -2,568.100 | 13.153571 |
| Swing + spread | -6,818.200 | 0.443462 | 75.07% | -22.200 | 15.10% | 3 | -1,536.500 | 21.400 |
| Conservative ATR/swing | -6,812.291 | 0.444330 | 75.26% | -22.300 | 14.90% | 3 | -1,469.725 | 21.400 |

Every geometry remains PF < 1 and is ineligible for live promotion. Fixed 3.2 pips has the least standalone net loss but recreates the known noise problem: 73.15% SL hits, 20 consecutive stops, and -5,141.1 pips after automatic same-direction re-entry. The bounded selection rule therefore uses combined trade plus post-SL re-entry loss, then consecutive stops and tail; it selects `CONSERVATIVE_ATR_SWING` for shadow observation only. Its wider distance is coupled to inverse units and cannot be used to increase risk. Automatic re-entry remains prohibited during shock.

## Reproduction

Run `tools/analyze_fast_bot_shock_guard_replay.py` with the seven existing `EUR_USD_M1_BA_2020...20260710` gzip files and the GET-only 2026-08-28 M1 BA file. The tool records every input SHA-256 in its JSON output and does not write policy or call a broker.
