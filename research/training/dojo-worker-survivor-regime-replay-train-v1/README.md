# DOJO worker survivors — separate-regime TRAIN replay

Three previously positive worn-TRAIN candidates were frozen and replayed on
the separate 2026-04-01 through 2026-05-01 worn-TRAIN regime, on both fixed
intrabar paths. None survived.

`pullback_a2` completed both paths and lost JPY 187.40 (OHLC) and JPY 195.94
(OLHC) from JPY 200,000. The two tailguard candidates failed closed on a stale
USD/JPY conversion quote at exactly 120 seconds; those four cells are
inconclusive and cannot be counted as survivors. The completed ledgers had no
owner concurrency breach or margin closeout and ended flat.

This run provides a temporal robustness check, not an untouched holdout. Full
ledgers, broker snapshots, preregistration, and the deterministic scoreboard
remain in the external archive `codex-worker-survivor-regime-replay-train-v1`.
