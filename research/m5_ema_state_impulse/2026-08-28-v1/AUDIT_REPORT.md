# Pre-replay audit report

Date: 2026-08-28 JST
Scope: repository-local, offline, read-only search plus isolated preregistration
implementation.

## Exact prior evidence search

Exact historical result count: **0**.

The literal runtime identifier appears in the active shadow runtime contract,
README and control records, but no historical result binds all of EMA3/EMA6,
three-bar midpoint momentum, EMA6 slope, ATR6/spread TP, six-M5-bar inventory,
1000 units and the three requested cost arms. Recursive structured-artifact
inspection outside `shadow_runtime` found zero exact parameter dictionaries.
EMA3/EMA12 artifacts were retained as non-equivalent nearby work rather than
silently reused.

## Runtime and dataset bindings

- runtime contract file SHA-256:
  `1d86badc851595b0df60c13c43b3f77dafea78b703490afabc415c29ef708247`
- runtime `paper_execution` canonical SHA-256:
  `5f01a6085b188d578c2054f619e135e76e66b92def15903adb023c99c1c7d50e`
- paper execution source SHA-256:
  `370d30c9456e8e5d548a0ac22d44456124b3f445a33372600473a476d3dc6d12`
- sealed dataset identity:
  `721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb`
- manifest SHA-256:
  `3408963dce76f6c2da5be7f766a48b4e1a91b3fbd03d7082f5369f8e1f2a4a00`
- gap report SHA-256:
  `95e7f222a0579a7339db0c35e8299af9ca2c139425161638176e1eea6dbc32e7`

The three discovery and validation byte-prefix offsets, row counts and hashes
are frozen in `PREREGISTRATION.json`. No candle file was opened by the audit or
test checkpoint.

## Implementation checkpoint

- preregistration file SHA-256:
  `0311d644dcd33cfb642181ab0d4965f74923ee031122a3b856ec187403d3bb35`
- preregistration canonical SHA-256:
  `e0867db99d04e64a01e4ced62b812c4ff09bc7cfe35a7cfbe459fc636819148d`
- runner SHA-256:
  `26564f5e254fdd2a41a135cf50c506f9b045ab397806223ccda0a45d626e5a10`
- tests SHA-256:
  `7573fe6bf023488752cf9aae391709dc9f11178aff08c3caf325e1a47440ab64`
- focused synthetic tests: **24/24 pass**
- Ruff static checks: **pass**
- Python compile: **pass**
- audit-only contract checks: **14/14 pass**

Pre-review evidence state:

- independent review receipt: absent;
- historical replay runs: 0;
- sealed candle rows decoded: 0;
- bytes read after 2025-08-28 boundary: 0;
- result/ledger files created: 0;
- network attempts / credential reads / external order attempts / external
  orders / launchd actions / Git actions: 0 / 0 / 0 / 0 / 0 / 0.

The runner refuses the one-shot path before the candle prefix reader whenever
the independent receipt is absent or hash-mismatched.
