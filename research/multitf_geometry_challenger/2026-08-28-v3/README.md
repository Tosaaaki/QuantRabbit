# MTF FX Causal Geometry V3

`MTF_FX_CAUSAL_GEOMETRY_V3` is an offline, paper-only, opened-development
experiment. It does not connect to OANDA, read credentials, call a broker,
install a service, or create an external order.

The experiment asks a narrower question than the M5 V1/V2 studies: can a slow
H4/H1 state choose direction while a completed M15 geometry event chooses the
entry, so that gross movement is large enough to survive executable FX costs?
The eight-member family is fixed in `preregistration.json` before scoring:

- C0/C1: aligned H4/H1 direction with M15 pullback reclaim, 4h/8h.
- C2/C3: aligned H4/H1 direction, H1 compression, and M15 break acceptance,
  4h/8h.
- C4/C5: H4 direction with an opposite-rail M15 sweep/reclaim, 4h/8h. This
  reproduces the prior diagnostic's only point-estimate-positive BASE family
  without assuming that diagnostic was evidence.
- C6: H4 overextension, H1 deceleration, and M15 sweep/reclaim fade, 8h.
- C7: three-pair USD-star residual, H1 recoupling, and M15 sweep/reclaim, 8h.

All features use exact UTC-aligned completed M5 aggregates. A decision exists
only after an M15 close. A virtual entry uses the first M5 open strictly later
than that decision, not the bar opening at the same timestamp. RAW_SIGNAL is
created without cost. The same trade ID is then valued under observed BID/ASK
plus 0.3 pip/side and 0.9 pip/side slippage. No TP or price SL exists; every lot
has a finite 4h/8h age and is liquidated at a split boundary when earlier.

The historical corpus was already inspected. Therefore this result is a
development diagnosis, never a holdout, profitability proof, admission, or
shadow-promotion decision. USD_JPY-only and session-only variants are excluded
from this family because earlier observation of those slices can only motivate
a separately preregistered N+1 experiment.

Run locally:

```sh
python3 research/multitf_geometry_challenger/2026-08-28-v3/replay_multitf_geometry.py
python3 -m unittest research/multitf_geometry_challenger/2026-08-28-v3/test_multitf_geometry.py
```

Files:

- `preregistration.json`: frozen mechanics, eight configs, evidence boundary.
- `replay_multitf_geometry.py`: deterministic replay and accounting.
- `test_multitf_geometry.py`: chronology, leakage, cost-lineage, and gate tests.
- `result.json`: complete development metrics for every config and arm.
- `evidence_packet.json`: compact hashes and selected diagnostic result.
