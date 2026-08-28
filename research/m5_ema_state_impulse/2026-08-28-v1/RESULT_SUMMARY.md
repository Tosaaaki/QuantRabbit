# M5 EMA state/impulse replay V1 result

Status: `INTERNAL_HISTORICAL_GATE_FAIL`

The one permitted offline replay used the exact shadow-runtime strategy and
did not change the frozen preregistration or runner. Later data after the
locked validation boundary was not decoded.

## Main result

| Metric | Discovery | Locked validation |
| --- | ---: | ---: |
| RAW proposals | 83,475 | 43,455 |
| Proposals per active UTC day | 535.096 | 543.188 |
| Direction accuracy at six bars | 48.426% | 49.182% |
| RAW mean pips/trade | -0.148 | -0.036 |
| RAW 95% block-bootstrap LCB | -0.239 | -0.114 |
| EXECUTABLE_BASE mean pips/trade | -1.330 | -1.169 |
| ADVERSE_STRESS mean pips/trade | -4.327 | -3.806 |
| RAW positive pairs | 1/3 | 1/3 |
| Terminal open inventory | 0 | 0 |

The proposal rate already exceeds 500 per active day, so the failure is not a
shortage of entries. RAW direction and post-entry expectancy are approximately
zero to slightly negative, and observed execution cost makes the result
materially negative.

The positive RAW median coexists with a negative mean: in discovery, RAW had
11,011 TP exits and 5,322 max-age exits; BASE had 7,689 TP exits and 6,432
max-age exits; ADVERSE had 3,488 TP exits and 7,859 max-age exits. This is
evidence that many small TP wins are outweighed by the unresolved tail, not
evidence that the strategy is profitable.

## Evidence boundary

- Result embedded SHA-256: `143844c771c1076c9ead6d449b3ca1c37861596f1b194b532593bd25aca83209`
- Packet embedded SHA-256: `48364162847b4ec8c13ac0094c9a10775d2a7b0d4014f79d843e484bcdb9d13e`
- Post-boundary rows decoded: 0
- Network attempts: 0
- Credential reads: 0
- External order attempts/orders: 0/0
- Profit proven: false

`LEDGER_MANIFEST.json` binds the uncompressed ledgers to the committed gzip
archives. The frozen preregistration suite passed 24/24 before replay. After
the one-shot lifecycle transition, its precondition test that requires result
files not to exist is expected to fail; the remaining 23 tests continue to
pass without modifying the frozen test bytes.
