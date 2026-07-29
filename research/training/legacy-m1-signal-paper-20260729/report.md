# M1Scalper faithful signal port / independent Paper A-B

Generated: 2026-07-29 JST

Project: `dojo-dual-eval`

Authority: Paper only, `live_permission=false`, external broker mutation forbidden,
order authority `NONE`.

## Outcome

The 2025 M1Scalper worker was recovered through its re-enabled snapshot at
commit `d8f751afc`. The strategy source and its historical indicator engine are
frozen in the repository. The strategy source matches SHA-256
`703981f1...37b50e`; the indicator source matches
`ddc8d428...582705`. The only port changes are:

1. Pin the historical parameter file inside this repository.
2. Replace the process wall clock with the completed bar's UTC hour.

The second change is a causal repair. It has the same value in forward operation
and prevents a replay from consulting the replay process's current time.

The port is **not profitable enough for admission**. The earlier result of
`+51 JPY / 6 closes` remains explicitly non-confirmatory.

## Costed causal replay

Five windows different from the earlier 2026-01 tuning/evaluation dates were
run with fixed 1,000-unit maximum, 1.1-pip round-trip cost, no future bars, and
no end-of-replay forced liquidation. These are lineage-unseen diagnostics for
this port, not a globally untouched holdout: the DOJO program has prior use of
historical 2024–2026H1 data.

| Arm | Net JPY | PF | Expectancy JPY | Worst single-window DD JPY | Closes | AI decisions | AI model calls / cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bot-only | -4,713.90 | ~0.629 | -10.18 | 1,809.20 | 463 | 0 | 0 / 0 JPY |
| AI inventory | -89.97 | ~0.439 | -11.25 | 79.75 | 8 | 2,872 | 0 / 0 JPY |

The AI policy suppressed most entries and greatly reduced absolute loss and
drawdown, but it did not produce positive PF or expectancy. Its economic policy
therefore remains `PROVISIONAL_FORWARD_OBSERVATION`; it is not promoted.

## Forward Paper rooms

Started as independent create-once rooms:

- `qr-legacy-m1scalper-bot-only`
  - owner `legacy-paper-m1scalper-bot`
  - operation `dojo-m1scalper-paper:bot-only:v1`
  - fixed 1,000 units
- `qr-legacy-m1scalper-ai-inventory`
  - owner `legacy-paper-m1scalper-ai`
  - operation `dojo-m1scalper-paper:ai-inventory:v1`
  - 1,000-unit ceiling; may reduce to 500, never increase

Each room has its own broker ledger, state, decision ledger, owner, operation
namespace, session directory, and OS lock. A duplicate launch against the
running Bot room failed closed with `M1 Paper room is already running`.

At the first forward observation, the Bot room had one Paper LONG at 1,000
units; the AI room recorded the identical raw signal and suppressed it because
UTC hour 04 was outside its provisional UTC23-long rule. This confirms that the
comparison measures the same base signal before AI inventory decisions.

The eight pre-existing Paper rooms remained running. No stop, restart, config
change, lot increase, live order, broker write, or production mutation was
performed.

## Admission rule

Do not admit M1Scalper or reapply its AI policy based on the prior six trades or
the current forward start. Admission requires multiple lineage-unseen windows,
sufficient closed trades, PF above 1 after costs, positive expectancy after
costs, acceptable drawdown, and prospective Paper confirmation. Current
decision: **continue isolated Paper observation; reject economic promotion**.

## Verification

- Frozen-source identity reconstruction and SHA checks
- Golden signal behavior
- Future-bar invariance and duplicate/rewound-bar rejection
- Authority rejection and 1,000-unit cap
- Independent operation IDs and ledgers
- AI disallowed-session suppression
- Duplicate registry and duplicate-launch fail-closed checks
- 12 targeted tests passed
- Both new screens, child processes, state files, broker ledgers, and decision
  ledgers directly observed
