# Financial Oracle V2 and executable path gate

This research-only checkpoint repairs the frozen 251-trade OANDA cash-flow
label before any exit, hedge, sizing, or multidimensional sweep is evaluated.
It also joins the versioned TP/SL schedule to executable-side OANDA S5 candle
evidence without turning OHLC touches into fictitious fills.

Run from the repository root:

```bash
python3 research/financial_oracle_v2/2026-08-10/build_financial_oracle.py
python3 research/financial_oracle_v2/2026-08-10/test_financial_oracle.py
python3 research/financial_oracle_v2/2026-08-10/verify_independent_oracle.py
python3 research/financial_oracle_v2/2026-08-10/build_path_metrics.py
python3 research/financial_oracle_v2/2026-08-10/test_path_metrics.py
python3 research/financial_oracle_v2/2026-08-10/verify_path_oracle.py
```

Truth boundaries:

- Cash flows are raw OANDA partial reductions, daily financing allocations by
  explicit `tradeID`, and terminal close legs, each counted once.
- Spread/slippage already present in executable broker prices and realized P/L
  are not subtracted twice.
- LONG path evidence uses bid wicks; SHORT uses ask wicks.
- A wick proves only that an executable-side level occurred inside an S5. It
  does not prove the order of TP, SL, protection replacement, fill, or close.
- No-bar endpoints stay unresolved. They are not forward-filled or replaced by
  M1/mid data.
- Concurrent margin is a gross trade-level proxy derived from each entry's
  actual `initialMarginRequired`. Account netting, available margin, and
  external/manual inventory are missing, so it is not account `marginUsed`.

The next phase may replay exit arms only on evidence-eligible rows. Sparse
missingness must remain labelled in the multidimensional cube; it may not be
converted to zero or bypassed by an ALL_TRADES fallback.
