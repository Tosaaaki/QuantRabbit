# Operator-alpha fast rotation — 2026-08-11

This research-only packet reconstructs two same-day margin closeouts, the next
four consecutive manual winners, and the immediately reopened manual/unknown
position boundary.  It converts the observed behavior into a falsifiable state
contract without changing live, Paper, broker, order, deployment, or runtime
configuration.

## Direct facts

- Four wins: +532.0000, +809.2680, +1,330.0000, +2,380.8153 JPY.
- Four-win total: +5,052.0833 JPY, 1.9874% of 254,209.0185 JPY.
- Last EUR_USD SHORT: 30,007 units, 1.15520 to 1.15470, 408.218 seconds,
  5.0 pips, +2,380.8153 JPY.
- The 1.15300 TP was created after entry and cancelled when the trade was
  manually closed at 1.15470; the operator did not wait for full TP.
- Close-to-next-entry delays in the win/rotation sequence are 714.703, 40.697,
  94.358, and 16.178 seconds.
- Same-day margin closeouts: -45,720 and -30,480 JPY.  These are failed
  fast-rotation shapes, not planned exits.
- Open entry fill 473207 remains manual/unknown and `NO_TOUCH`.

## X source boundary

The supplied X URL was opened in the actual Chrome extension browser on the
existing connected profile.  The visible target post, parent, author reply,
and image nodes were inspected.  The target post has no attached image or
video; its only image node is the author's profile image.  X API was not used.

The target post contributes a question/measurement structure: completed H4
trend and recent extremes, separate supporting/opposing reasons, entry and skip
conditions, failure precursors, validation fields, and post-trade review.  The
parent's monthly-500,000-JPY and ten-minutes-per-day claims have no supplied
capital/cost/DD/opportunity contract and are not admitted as strategy evidence.

## Four-arm result

All arms use the exact same six entry IDs.  Baseline is broker actual after-cost
P/L.  Counterfactual operator exits use the first post-entry complete S5
side-correct bid/ask close, with an extra adverse half-spread exit stress.
The profit floor, monitoring activation, and maximum hold are descriptive
statistics derived from the four wins; therefore this is in-sample diagnostic
evidence, not validation.

See `verdict_v1.md` and `comparison_report_v1.json` for the result.  The
operator contract cuts diagnostic DD sharply but remains slightly negative;
the X H4 filter selects the two margin-closeout entries and skips the four
winners, so it is directly rejected as an edge signal on this cohort.

## Reproduction

The committed source packet is immutable and can be verified without broker
access:

```bash
PYTHONPATH=src python3 research/operator_alpha_fast_rotation/2026-08-11/acquire_frozen_truth.py --check
python3 research/operator_alpha_fast_rotation/2026-08-11/run_operator_alpha_replay.py
python3 -m unittest research/operator_alpha_fast_rotation/2026-08-11/test_operator_alpha_replay.py -v
python3 research/operator_alpha_fast_rotation/2026-08-11/verify_independent_oracle.py
```

Fresh source acquisition is GET-only and requires an explicit env file:

```bash
PYTHONPATH=src python3 research/operator_alpha_fast_rotation/2026-08-11/acquire_frozen_truth.py --env-file /path/to/.env.local
```

Do not run that command against a different account and treat it as the same
cohort.  Source hashes are frozen in `source_manifest_v1.json`.

## Adoption blocker

This packet lacks decision-time account margin available/used, full inventory,
forecast lineage, and executable unwind evidence.  Initial required-margin
proxies are 85–95% of entry balance on these rows.  Every fusion row therefore
returns `WAIT_EVIDENCE_INCOMPLETE` for live use.  The reusable output is the
operator state/exit/reentry contract and the evidence schema—not live settings.
