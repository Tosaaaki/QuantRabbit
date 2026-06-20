# Webull OpenAPI Stock Trading Setup

As of 2026-06-20, Webull publishes an official OpenAPI for US market trading, account management, and market data:

- Docs: https://developer.webull.com/apis/docs/
- Getting started: https://developer.webull.com/apis/docs/getting-started/
- SDKs: https://developer.webull.com/apis/docs/sdk/
- Stock trading: https://developer.webull.com/apis/docs/trade-api/stock/
- Authentication: https://developer.webull.com/apis/docs/authentication/overview/

Do not use Webull login passwords, phone numbers, browser cookies, or unofficial reverse-engineered APIs in QuantRabbit. This integration uses only official OpenAPI credentials.

## Credential Flow

1. Apply for Webull OpenAPI access from Webull's OpenAPI management page.
2. Wait for approval. Webull's guide says this typically takes 1-2 business days.
3. Generate an App Key and App Secret in Webull OpenAPI Management.
4. Install the official SDK:

```bash
pip3 install --upgrade webull-openapi-python-sdk
```

5. Put credentials in `.env.local`:

```bash
QR_WEBULL_ENV=test
QR_WEBULL_REGION=us
QR_WEBULL_APP_KEY=...
QR_WEBULL_APP_SECRET=...
QR_WEBULL_ACCOUNT_ID=...
```

Use `QR_WEBULL_ENV=production` only after UAT/test verification is complete. Production order sending additionally requires `QR_WEBULL_LIVE_ENABLED=1`.

## Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli webull-env-check
PYTHONPATH=src python3 -m quant_rabbit.cli webull-account-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli webull-stage-stock-order \
  --symbol AAPL \
  --side BUY \
  --quantity 1 \
  --order-type LIMIT \
  --limit-price 180
```

The stock-order command stages by default and writes:

- `data/webull_stock_order_request.json`
- `docs/webull_stock_order_stage_report.md`

Real order placement is blocked unless all three gates are present:

```bash
QR_WEBULL_LIVE_ENABLED=1 PYTHONPATH=src python3 -m quant_rabbit.cli webull-stage-stock-order \
  --symbol AAPL \
  --side BUY \
  --quantity 1 \
  --order-type LIMIT \
  --limit-price 180 \
  --send \
  --confirm-live
```

## Current Scope

This is a broker adapter and order staging layer, not a full stock strategy engine. It deliberately stays separate from the existing OANDA FX runtime because the current risk engine is pip/JPY/FX-margin based. Stock strategy, equity risk sizing, PDT/cash/margin checks, halt/session handling, and corporate-action filters must be added before autonomous production stock trading.
