# M5 EMA state/impulse historical replay — preregistration

Status: **PREREGISTERED — AWAITING INDEPENDENT REVIEW — NO RESULT RUN**

This isolated experiment tests the exact causal proposal rule currently named
`M5_EMA_STATE_IMPULSE_INVENTORY_V1`. The repository search found no historical
result with the exact EMA3/EMA6, three-bar impulse, EMA6 slope, ATR6/spread TP,
six-M5-bar inventory, 1000-unit and three-arm contract. EMA3/EMA12 studies are
not treated as equivalent evidence.

The machine contract is [PREREGISTRATION.json](PREREGISTRATION.json). It freezes
one configuration before any result run. Calibration, discovery and locked
validation end at 2025-08-28 04:05 UTC. The decoder is byte-bound to the sealed
prefixes and must never read, hash, parse or label a later row.

## Fixed hypothesis

Using seven contiguous completed M5 midpoint bars, calculate EMA3 and EMA6 with
the same first-value seed as the runtime. Direction is the sign of EMA3−EMA6.
Emit a cost-independent signal only when three-bar midpoint momentum and the
EMA6 one-step slope agree with direction. Freeze
`TP=max(0.5×ATR6, 1.5×decision spread)`; do not use an individual price stop;
liquidate at six held M5 bars, at a data gap, or at the split terminal.

Every raw signal has one content-addressed `signal_id` and fans out to RAW,
observed BID/ASK, and adverse stress. Spread, slippage and the declared M5
latency envelope are applied after proposal creation and never suppress the raw
proposal.

## Chronology and ambiguity

The source timestamp is candle start. The decision is only after that candle
completes. A source open with the same timestamp as the decision is ineligible;
the fill is the first exact M5 open strictly later. A missing expected fill is
not synthesized. Max-age open liquidation precedes intrabar TP in its due bar.
The adverse fill uses the eligible fill-bar worst executable extreme, so an
adverse position cannot claim a TP in its fill bar.

## Evidence rule

The discovery and validation thresholds are literal in the JSON contract. A
pass means only that the fixed historical hypothesis cleared the internal gate.
It does not prove future profit and grants no live or broker authority. The
runner will not decode the dataset or create results without an independent
review receipt binding the preregistration and runner hashes.
