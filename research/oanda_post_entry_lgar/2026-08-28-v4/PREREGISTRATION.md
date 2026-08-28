# M5_GENERIC_EMA_POST_ENTRY_LGAR_V4 preregistration

This is an offline, zero-order-authority experiment. It asks one narrow question: when the entry source is deliberately simple and cost-blind—EMA3 versus EMA12 on every eligible completed M5 bar—can a finite post-entry state manager improve the same raw proposal stream without hiding spread, slippage, gaps, stale inventory, or terminal loss?

The immutable input is the OANDA M5 BID/ASK capture sealed as `721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb`. Only EUR_USD, USD_JPY, and AUD_USD are in scope. Calibration ends 2024-11-28 04:05 UTC and tuning ends 2025-08-28 04:05 UTC. Earlier invalidated implementation runs exposed opened development, so the final remediation runner is byte-bounded at the tuning boundary and reports development as `INVALIDATED_DIAGNOSTIC_NOT_EVIDENCE`. It decodes neither opened-development nor holdout market fields. The final three months remain untouched holdout.

## Fixed family

- P0: fixed 12-bar time exit.
- P1: fixed 24-bar time exit.
- P2: fixed 48-bar time exit.
- P3: calibration RAW closing-MFE Q40 TP, otherwise 24-bar exit.
- P4: the same frozen Q40 TP, otherwise 48-bar exit.
- P5: FX-LGAR. A high-path-efficiency lot with at least two adverse votes from momentum, prior-rail state, and USD breadth is TRAPPED and unwinds. At 24 bars a non-HARVEST lot unwinds. HARVEST may continue to 48 bars.
- P6: P5 plus a 50% closing-MFE giveback lock after the frozen Q40 TP has first been reached.
- P7: P6 plus a same-signed USD-node basket unwind when STALE/TRAPPED inventory exists and the exact-time aggregate RAW marked PnL is non-negative.

Dynamic exits act only on completed data and fill at the next exact M5 open strictly later than the action time. No fill, feature, or label crosses a missing M5 slot. There is no synthesized candle, price stop, martingale, opposite same-pair hedge, or leverage retuning. Fixed controls are 1,000 units per lot, four lots per pair, absolute net USD-node cap four, gross eight lots, and gross leverage twenty. Each USD-node sign is also capped at four lots; this conservative hard-control prevents a later exit of offsetting inventory from causing the remaining net USD exposure to jump above four.

RAW proposals are always logged before portfolio controls. EXECUTABLE_BASE marked equity is the single common hard-execution ledger for leverage and margin safety, so a BASE breach freezes new execution and unwinds the same lineage for every arm. ADVERSE_STRESS never rejects an entry and never changes that lineage; its first counterfactual margin-closeout or ruin point and marked liquidation value are recorded as an evidence failure.

Every trade has one `signal_id`, policy action time, and exit time shared by RAW_SIGNAL, EXECUTABLE_BASE, and ADVERSE_STRESS. RAW is midpoint economics. BASE applies observed BID/ASK and 0.3 pip slippage per side. ADVERSE applies the same path and 0.9 pip per side. Costs never suppress a source proposal and never choose a policy.

Selection uses tuning RAW only over the fixed eight-policy family. Uncertainty is a deterministic UTC-decision-day cluster bootstrap with Bonferroni correction. The final remediation report shows gross and executable expectancy, tuning-sample paired diagnostics against P2, corrected lower bounds, raw density, pair/month stability, inventory age, terminal liquidation, drawdown, and leverage-guard activity. A positive point estimate with a non-positive corrected lower bound is not evidence of an edge. If tuning unexpectedly passes, a new preregistered version and a fresh evidence window are required; observed development is not reused and holdout is not opened here.
