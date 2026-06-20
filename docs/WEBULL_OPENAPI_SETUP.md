# Webull OpenAPI Stock Trading Setup

As of 2026-06-20, Webull publishes an official OpenAPI for US market trading, account management, and market data:

- Docs: https://developer.webull.com/apis/docs/
- Getting started: https://developer.webull.com/apis/docs/getting-started/
- SDKs: https://developer.webull.com/apis/docs/sdk/
- Stock trading: https://developer.webull.com/apis/docs/trade-api/stock/
- Authentication: https://developer.webull.com/apis/docs/authentication/overview/

Do not use Webull login passwords, phone numbers, browser cookies, or unofficial reverse-engineered APIs in QuantRabbit. This integration uses only official OpenAPI credentials.

## OpenAPI Application Flow

1. Apply for Webull OpenAPI access from Webull's OpenAPI management page.
2. Wait for approval. Webull's guide says this typically takes 1-2 business days.
3. Generate an App Key and App Secret in Webull OpenAPI Management. Webull's application flow requires SMS/MFA verification during key generation.
4. Install the official SDK:

```bash
pip3 install --upgrade webull-openapi-python-sdk
```

## Secret Storage

Do not put Webull login passwords, phone numbers, browser cookies, or MFA codes in `.env.local`. If an operator needs to keep the Webull login phone/password locally for reference, store it in macOS Keychain:

```bash
scripts/webull-keychain.sh store-login
scripts/webull-keychain.sh status
```

After Webull approves OpenAPI access and you generate App Key/App Secret, store the OpenAPI credentials in Keychain:

```bash
scripts/webull-keychain.sh store-openapi
scripts/webull-keychain.sh status
```

Run QuantRabbit with Keychain-backed OpenAPI env vars:

```bash
scripts/webull-keychain.sh run -- env \
  QR_WEBULL_ENV=test \
  QR_WEBULL_REGION=us \
  PYTHONPATH=src \
  python3 -m quant_rabbit.cli webull-env-check
```

`.env.local` is still supported for local automation, but it should contain only official OpenAPI values, never the Webull account password:

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

If credentials are stored only in Keychain, prefix the same commands with `scripts/webull-keychain.sh run --`.

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
