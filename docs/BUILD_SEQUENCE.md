# Build Sequence

## Phase 1: Broker Truth And Risk Gateway

1. Import legacy history into `data/legacy_history.db`.
2. Read OANDA account, open trades, pending orders, and live quotes through a read-only client.
3. Validate proposed order intents through `RiskEngine`.
4. Keep live send unavailable until dry-run receipts prove the gateway blocks known loss paths.

## Phase 2: Strategy Mining

1. Mine `pretrade_outcomes`, `seat_outcomes`, `live_trade_log`, `trader_journal`, and daily state files.
2. Promote only pair/direction/vehicle contexts with positive expectancy and bounded loss.
3. Generate strategy candidates as order intents, not broker orders.
4. Treat `RISK_REPAIR_CANDIDATE` as not-live-ready until a dry-run receipt proves loss is capped under 500 JPY.
5. Feed the strategy profile into `risk-dry-run` so history status is enforced, not just reported.

## Phase 3: Execution Gateway

1. Add one live gateway method behind `QR_LIVE_ENABLED=1`.
2. Require current broker snapshot, fresh quote, TP, SL, risk JPY, reward/risk, and owner.
3. Journal every accepted and rejected intent.

## Phase 4: Automation

1. Automation may observe and report first.
2. Automation may propose order intents after Phase 2.
3. Automation may execute only after Phase 3 has a tested live-send path.
