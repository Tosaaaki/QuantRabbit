# DOJO AI exit TRAIN v1

This directory preserves eight actual historical-candle virtual decisions, but
the run is invalid. The packet included the decision M5 bar's high, low, and
close while the decision was priced at that bar's open. V2 excludes the whole
decision bar. V1 must not be used for policy comparison, holdout, or promotion.

- One position packet was supplied to each fresh context.
- No answer key was supplied until all decisions in that batch were returned.
- The source is receipted OANDA M5 bid/ask history.  Bid/ask spread is present,
  but the scenario generator does not add a separate slippage or financing
  charge, so this is not an all-cost profitability claim.
- Six currency pairs are represented: AUD_USD, USD_JPY, EUR_USD, USD_CHF,
  GBP_USD, and GBP_JPY.
- The recorded arithmetic was AI -51.9 pips, always CUT -63.8, always HOLD
  -29.6, but all comparisons are invalidated by the same-bar leak.

`evidence.json` binds the source manifest, deterministic scenario generator,
packet hashes, prompt families, decisions, revealed branch outcomes, and score.
