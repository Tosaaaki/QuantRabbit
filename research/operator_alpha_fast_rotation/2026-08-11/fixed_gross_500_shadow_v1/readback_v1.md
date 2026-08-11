# Fixed gross +500 shadow readback

- Environment: research/shadow only; no live/Paper/order/deploy write.
- Adoption: HOLD.
- Fresh broker truth: transaction 473212→473218 realized +560 JPY, but it overlapped still-open 473207; excluded from the clean concurrency=1 success denominator.
- Flat check: one non-shadow/manual-or-unknown trade remained open, so no re-entry is represented as broker-confirmed for that account snapshot.

| Arm | TP / 7 | Unresolved | After-cost terminal JPY | Mean hold sec |
|---|---:|---:|---:|---:|
| ATR_010 | 7 | 0 | 654.10 | 1055.8 |
| ATR_020 | 7 | 0 | 1756.76 | 1096.5 |
| ATR_025 | 7 | 0 | 2317.20 | 1104.4 |
| FIXED_GROSS_500 | 6 | 1 | 2264.64 | 1256.5 |

- Independent oracle: same decision IDs=True; forced end close absent=True; prior four clean manual wins reached fixed +500=True; mean target-touch=405.8s.
- Signal supply: contract is present but remains NOT_EVALUABLE until a frozen out-of-sample S5/M1 signal log exists; it does not extrapolate 100 trades/day or returns.
- Standard replay worker `scripts/replay_exit_workers_groups.py` is absent in this revision, so this uses the research-local S5 bid/ask replay and is not a substitute for that standard worker.
