# Loss-close price-action multidimensional sweep

This path is read-only pre-holdout research for the existing SL replacement
task. It does not implement a new X-sourced strategy and does not touch Paper,
live, broker, order, deploy, or holdout paths.

`run_preholdout_stage1.py` joins canonical execution-ledger STOP_LOSS events to
local OANDA S5 bid/ask history, calculates the bounded 27-cell Stage-1 feature
grid, and refuses economic scoring whenever any S5 interval from entry through
the fixed 60-minute unwind is absent. No-quote gaps remain visible and are not
filled forward.

Run from the repository root:

```bash
PYTHONPATH=src python3 research/loss_close_price_action_sweep/run_preholdout_stage1.py \
  --output research/loss_close_price_action_sweep/preholdout_stage1_report.json
```

The output is diagnostic until both TRAIN and embargoed VALIDATION contain at
least 30 identical, strict-S5 events and the connected plateau selector passes.
