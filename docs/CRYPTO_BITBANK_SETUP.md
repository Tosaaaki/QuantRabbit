# QuantRabbit Crypto｜bitbank Shadow / Paper Setup

## Safety boundary

This runtime is structurally paper-only:

- `NO_EXECUTE=true`
- `CRYPTO_LIVE_READY=false`
- `WITHDRAWAL_ENABLED=false`
- `CRYPTO_ORDER_AUTHORITY=NONE`
- Public Scanner and Public Stream require no credentials.
- The Private REST adapter exposes only `GET /v1/user/assets`.
- There is no order, cancellation, settlement, withdrawal, or API-permission
  mutation method.

Any unsafe environment value fails closed before the crypto runtime starts.
The crypto package is isolated under `src/quant_rabbit/crypto/`; it does not
modify or reuse the OANDA live path.

## Public Scanner / Shadow / Paper

Run one bounded public-data scan:

```bash
PYTHONPATH=src python3 -m quant_rabbit.crypto.cli scan \
  --output-json data/crypto/latest_scan.json \
  --output-markdown docs/crypto_bitbank_canary_report.md
```

Run a short paper canary with an append-only ledger:

```bash
PYTHONPATH=src python3 -m quant_rabbit.crypto.cli canary \
  --cycles 3 \
  --interval-sec 2 \
  --data-dir data/crypto \
  --report docs/crypto_bitbank_canary_report.md
```

Check a bounded Public Stream subscription:

```bash
PYTHONPATH=src python3 -m quant_rabbit.crypto.cli stream-canary btc_jpy \
  --messages 1 \
  --timeout-sec 15
```

Local runtime artifacts live under ignored `data/crypto/`:

- `latest_scan.json`
- `canary.json`
- `ledger.db`

Verify ledger recovery and integrity with:

```bash
PYTHONPATH=src python3 -m quant_rabbit.crypto.cli ledger-verify \
  --ledger data/crypto/ledger.db
```

## macOS Keychain

The approved local secret path mirrors the existing Webull helper. Store only
a newly issued minimum-permission bitbank key after the earlier plaintext key
has been revoked. Trading and withdrawal permissions must both be disabled.

```bash
scripts/bitbank-keychain.sh store-readonly
scripts/bitbank-keychain.sh status
scripts/bitbank-keychain.sh registry
```

The helper prompts through macOS Keychain, so values are not passed through
command arguments, shell history, repository files, or stdout.

Run the only supported Private REST check:

```bash
scripts/bitbank-keychain.sh run-readonly -- \
  env \
    NO_EXECUTE=true \
    CRYPTO_LIVE_READY=false \
    WITHDRAWAL_ENABLED=false \
    CRYPTO_ORDER_AUTHORITY=NONE \
    PYTHONPATH=src \
    python3 -m quant_rabbit.crypto.cli private-check
```

The check reports only authentication status and the asset record count. It
does not print credentials, asset balances, or secret-bearing request headers.
If either Keychain item is absent, the Private path fails closed while public
Scanner / Shadow / Paper remains available.

## Credential registry (no secret values)

| Field | Value |
|---|---|
| Keychain account | `tossaki` by default; override with `QR_BITBANK_KEYCHAIN_ACCOUNT` |
| API key service | `QuantRabbit.Bitbank.readonly_api_key` |
| API secret service | `QuantRabbit.Bitbank.readonly_api_secret` |
| API type | bitbank Private REST read-only |
| Declared permission | asset read only |
| Trading | disabled |
| Withdrawal | disabled |
| Rotation date | record only after the new key is issued and stored |
| Connection result/date | record only after `private-check` succeeds |

`scripts/bitbank-keychain.sh registry` emits the service/account names,
declared permissions, check time, and `present` flags. It never emits a secret.
