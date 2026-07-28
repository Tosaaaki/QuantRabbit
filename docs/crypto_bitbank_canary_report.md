# QuantRabbit Crypto bitbank canary

- Observed: 2026-07-28T05:23:41.576445+00:00
- Mode: READ_ONLY_SHADOW_PAPER
- Guardian: GREEN
- JPY pairs: 47
- Eligible: 8
- Candidates: 0
- Live/private mutation: disabled

## Candidates

- No candidate cleared the conservative safety buffer.

## Rejections

- `ada_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE
- `xlm_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE
- `xrp_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `sol_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `doge_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `eth_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `btc_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `sui_jpy`: SPOT_LONG_MOMENTUM_NOT_POSITIVE, NET_EDGE_BELOW_SAFETY_BUFFER
- `avax_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `render_jpy`: CIRCUIT_UNKNOWN, DEPTH_NOT_SAMPLED
- `ltc_jpy`: CIRCUIT_UNKNOWN, DEPTH_NOT_SAMPLED
- `boba_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `bcc_jpy`: CIRCUIT_UNKNOWN, DEPTH_NOT_SAMPLED
- `bnb_jpy`: CIRCUIT_UNKNOWN, DEPTH_NOT_SAMPLED
- `mana_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `dai_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `op_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `flr_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `dot_jpy`: CIRCUIT_UNKNOWN, LOW_VOLUME, DEPTH_NOT_SAMPLED
- `trx_jpy`: CIRCUIT_UNKNOWN, DEPTH_NOT_SAMPLED

## Paper KPI

- Net PnL: 0 JPY
- Max DD: 0 JPY
- Fills: 0
- Fees: 0 JPY
- Spread cost: 0 JPY
- Slippage cost: 0 JPY
- Discipline violations: 0

## Canary evidence

- REST cycles: 3
- Public Stream: PASS
- Public Stream messages: 1
- Ledger events: 500
- Ledger hash chain: valid
- Private REST: BLOCKED (rotated Keychain credential absent)
- Keychain services: `QuantRabbit.Bitbank.readonly_api_key`, `QuantRabbit.Bitbank.readonly_api_secret`
- Keychain account: `tossaki`
- Keychain present: `false` / `false`

No market cleared the positive-momentum and net-edge buffer in this window, so zero virtual orders is the expected disciplined result rather than an execution failure.
