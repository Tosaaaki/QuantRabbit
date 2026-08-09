# Gapless decision-time truth admission

This research path keeps two price sources separate:

- OANDA ledger outcomes remain the executable after-cost truth.
- Dukascopy historical best-bid/best-ask ticks are used only to construct causal pre-entry features.

The OANDA candle `complete` flag does not assert data completeness. An OANDA S5 omission is therefore never relabelled as a no-trade interval. For the feature source, a five-second no-tick bucket can be made explicit only when its complete hourly raw file passed HTTP, LZMA, schema, ordering, duplicate, and bid/ask checks. Every such carried bucket retains `CARRY_RAW_NO_TRADE` lineage.

Run the bounded pipeline:

```bash
research/historical_learning_admission/.venv/bin/python \
  research/historical_learning_gapless_truth/run_pipeline.py all --workers 4
research/historical_learning_admission/.venv/bin/python \
  research/historical_learning_gapless_truth/verify_pipeline.py
```

The raw cache is ignored by Git. Reproducible manifests, audit reports, tests, and final results are committed. The process stops before 2 GiB for review and never exceeds the 5 GiB hard cap.

The frozen input contains 251 labeled episodes, but the explicitly allowed three pairs account for only 146. Reports preserve both denominators; they do not call 146/146 coverage “251/251”.

The replay reuses the already pinned, isolated historical-learning environment at
`research/historical_learning_admission/.venv` and its committed
`requirements.lock`; this path adds no dependency.
