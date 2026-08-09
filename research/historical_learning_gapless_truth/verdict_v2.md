# Gapless historical-learning verdict v2

Decision: **REJECT**. Holdout remained sealed.

The acquisition problem is resolved for the frozen three-pair scope. The cache
contains 413 complete official Dukascopy bid/ask tick hours; five additional
required hours are preregistered New York weekend closures. There are zero
market-open acquisition defects. One hour per pair was downloaded twice and
matched in both raw SHA-256 and decoded row count.

Strict 48-M5 price-action features are available for 141/146 eligible-pair
episodes (96.58%) and 141/251 episodes overall (56.18%). The five eligible
exclusions are all Sunday-reopen lookbacks crossing the scheduled market
closure. The remaining 105 episodes are outside the frozen AUD/JPY, EUR/JPY,
EUR/USD allowlist; they were not silently backfilled with other pairs.

The 16-day and 32-day windows remain below the preregistered fit gate. In the
64-day validation, all-trades earned 15,144.48 JPY with PF 1.585 and max DD
6,794.78 JPY. The price-action filter earned 10,437.10 JPY: incremental net
-4,707.38 JPY, reported paired LCB -142.07 JPY, and independent-bootstrap LCB
-143.12 JPY. It rejected 11 winners worth 9,156.28 JPY while avoiding only five
losers worth 4,448.90 JPY. PF also fell to 1.487; DD improved by only 342.62 JPY.
Margin coverage is incomplete, so ruin and deployability remain unproven.

No multidimensional parameter sweep is opened from this result. The fixed
feature admission failed, and tuning it on validation would convert the run
into a search for a positive period. The next independent hypothesis is an
inventory/concurrency and exposure cap applied to the profitable all-trades
baseline, preregistered separately and judged on incremental net, paired LCB,
DD, and complete margin evidence.
