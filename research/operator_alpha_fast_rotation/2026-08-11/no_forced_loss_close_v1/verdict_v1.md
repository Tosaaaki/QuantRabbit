# NO_FIXED_SL same-cohort verdict

## Result

No no-fixed-SL arm is eligible for adoption. Several hedging-account arms have
positive **pre-financing** terminal contribution, but every such row carries
unresolved original/hedge inventory and/or unknown financing. Under the
preregistered rule, that is `REJECT`, not a win. Netting-account opposite
orders are reductions with realized original loss, not hedges.

| arm (hedging account) | exec | net pre-financing JPY | PF | max seq DD | max intra-trade DD | unresolved | financing unknown | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_HARD_SL_BASELINE | 50 | -8782.37 | 0.865 | 18849.98 | 8211.57 | 0 | 10 | NOT_EVALUABLE |
| B_NO_SL_NAKED_RETURN_WAIT | 38 | -12933.33 | 0.822 | 30566.30 | 21417.54 | 5 | 19 | REJECT |
| C_NO_SL_DELAYED_ENTRY_EARLY_TP | 50 | 9330.57 | 1.189 | 25501.68 | 20143.63 | 4 | 16 | REJECT |
| D_NO_SL_HEDGE_RETURN_050 | 50 | 23438.43 | 1.421 | 23344.66 | 18965.73 | 5 | 26 | REJECT |
| E_NO_SL_PARTIAL_PROFIT_BE | 50 | 21911.01 | 1.591 | 21807.61 | 15652.04 | 3 | 15 | REJECT |
| F_NO_SL_MULTI_PAIR_ROTATION | 48 | -10409.71 | 0.878 | 37969.25 | 21417.54 | 6 | 22 | REJECT |
| H1_LOCK_AT_ADVERSE_LEVEL_AND_WAIT_050 | 50 | 151.25 | 1.003 | 11854.38 | 7075.96 | 3 | 21 | REJECT |
| H2_HEDGE_TP_KEEP_ORIGINAL_050 | 50 | 23438.43 | 1.421 | 23344.66 | 18965.73 | 5 | 26 | REJECT |
| H3_HEDGE_PARTIAL_TP_REHEDGE_050 | 50 | 45304.00 | 2.052 | 18239.61 | 18533.69 | 7 | 27 | REJECT |
| H4_HEDGE_REVERSAL_CONFIRM_EXIT_050 | 50 | 10831.08 | 1.239 | 26975.66 | 20495.45 | 3 | 21 | REJECT |
| H5_HEDGE_PROFIT_OFFSET_ORIGINAL_BE_050 | 50 | 23438.43 | 1.421 | 23344.66 | 18965.73 | 5 | 26 | REJECT |
| H6_PERSISTENT_TREND_STRESS | 50 | -378133.96 | 0.000 | 378133.96 | 18533.69 | 7 | 50 | REJECT |
| H7_GAP_AND_FINANCING_STRESS | 50 | -390042.67 | 0.000 | 390042.67 | 18533.69 | 7 | 0 | REJECT |

## Interpretation

- The hard-SL comparison lost money, but had zero terminal inventory. It is
  still `NOT_EVALUABLE` for full after-financing comparison on ten trades.
- Delaying entry and taking profit early is the only unhedged positive row,
  matching the manual fast-profit behavior, but four positions remained open
  and sixteen crossed an unknown-financing boundary. The apparent +JPY result
  is therefore rejected rather than promoted.
- H3 at 0.5 ATR reports positive realized+MTM before financing, but leaves
  seven inventories unresolved, has 27 financing-unknown executions and 42
  hedge entries. This is precisely the tail-risk/inventory deferral forbidden
  by the contract.
- Persistent-trend and gap/financing stress turn the selected H3 shape sharply
  negative. No margin closeout occurred at fixed 5,000 units; that is only a
  size-specific observation, not evidence that the mechanism cannot fail.
- The manual wins held for seconds/minutes and rotated after confirmed closes.
  Multi-day recovery waits and terminal inventory are a behavioral mismatch,
  so they do not reproduce `OPERATOR_ALPHA_FAST_ROTATION_V1`.

## Evidence boundary

The repository-required `scripts/replay_exit_workers_groups.py` entrypoint is
absent from HEAD, `origin/codex/qr-python-ecosystem-audit-20260810`, and
`origin/main`. The real-data run here is a research-local M1 bid/ask replay and
must not be represented as the standard QR exit-worker replay. Financing is
not zero-filled, and F cross-pair concurrent portfolio margin is not jointly
simulated; F remains non-adoptable.

## Reproduction

```bash
python3 -m unittest -v research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/test_accounting_oracle.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/run_no_forced_loss_replay.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/verify_independent_oracle.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/build_readback.py
```
