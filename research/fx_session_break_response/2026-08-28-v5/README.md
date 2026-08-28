# FX Session Break Response Surface V5

This isolated, offline experiment tested a preregistered 128-cell interaction
surface for London session break acceptance and rejection response on the
sealed OANDA M5 BID/ASK capture. It did not access the network, credentials,
launchd, broker/order endpoints, or Git.

## Outcome

The single replay attempt was rejected before winner selection. Exactly 20 of
128 configs had zero discovery observations, all in constrained
`LONDON_MIDDAY / REJECT_FADE` cells. The preregistered max-T contract required
all 128 configs to have finite, nonzero standardization evidence, so the runner
failed closed with `REJECTED_DISCOVERY_FAMILY_UNSTANDARDIZABLE`.

- Attempt: 1
- Process exit: 1
- Elapsed real time: 7.39 seconds
- Configs with at least one observation: 108/128
- Zero-observation configs: 20/128
- Winner selected: no
- Locked validation rows decoded: 0
- Opened-development/holdout rows decoded: 0
- RAW/BASE/ADVERSE result: not computed; no winner existed
- Profit proven: no
- External orders: 0

V5 must not be rerun or repaired by changing its thresholds after seeing this
result. A successor version may preregister a coarser family or a discovery
density admission stage, while retaining this rejection as immutable evidence.

## Files

- `PREREGISTRATION.json` / `.md`: frozen design and exact 128 configs.
- `replay_session_break_response.py`: the exact runner bytes used by attempt 1.
- `result.json`: fail-closed result with the 20 config identities.
- `evidence_packet.json`: compact content-addressed result receipt.
- `seal_failed_attempt.py`: no-replay evidence sealer for the attempt-1 error.
- `test_replay_session_break_response.py`: focused chronology, formula,
  authority, max-T, mutation, and failure-evidence tests.

Focused tests may be rerun; the replay command must not be rerun for V5.
