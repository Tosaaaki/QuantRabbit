# DOJO first-wave terminal bundle Drive readback

- Checked at UTC: `2026-07-21T21:12:00Z`
- Classification: `AUTHENTICATED_CONNECTOR_EXACT_BYTE_READBACK_DIAGNOSTIC`
- Drive manifests folder: `1PsBMkcVnZ5elZvuzmbzlkE206OqaKMPi`
- Lineage tip after `RESULT_BOUND`: `9a711e11df2b6ac715c5c5ce1cf506ee9ee1f3ea294c0d22e2b4ce0b129a9aa1`
- Terminal handoff receipt: `864da60ce9d07a8579bc55e3d1e0b2a45700b869bf7b13607c65508a0632617b`
- Binding timing: `RETROSPECTIVE_ADMIN_BINDING`

## Five-artifact inventory

| Kind | Drive ID | Size | Local/remote SHA-256 |
|---|---|---:|---|
| RUN | `1JIuEt1dB49XruIhee_pTTiJhrvZJst-J` | 57,863 | `ee5791ff657adeb74ac21ed3f73bceebd1546f5cc7fc31fc463a798870920e8c` |
| EVALUATION | `17hdrRznJ0Ms3ez2xLgPsjEXHgHF9wFE8` | 10,324 | `7fed736924bbf7a81fe95cba23e22530275f4b00a6ff78dc9ef43bef13d30fea` |
| CELLS | `1Nn6qI_inOdXQIGb6pNNQMjHXlPydjtwD` | 61,041 | `cea6de3e3c1999249aec54a052853ffddb8aec877db51b244fd7c773007367a7` |
| SEALED_STUDY | `1jyn9pmoUK5O6Kcoq6dcwLHzYWU642Gnw` | 6,779 | `03bd6d1080d40be3d6d27a4982c1afb1e2c53729da842e4bb08f494aea5b4998` |
| TERMINAL_HANDOFF | `19uCeAFnc1oEpeK-QyCdxG6fJ1iLQMl0r` | 3,652 | `0ddb77d73e39922119b139178c58c5b5362e0a19218a95416848b74ea04b0907` |

The RUN, EVALUATION, and CELLS equality checks were closed by the archive automation. SEALED_STUDY and TERMINAL_HANDOFF were uploaded through the authenticated Google Drive connector, fetched back as raw bytes, and independently hashed; both matched their local files exactly.

## Authority boundary

- `trainer_packet_eligible=false`
- `proof_eligible=false`
- `promotion_eligible=false`
- `live_permission=false`
- `order_authority=NONE`
- `broker_mutation_allowed=false`

This note records an actual connector readback but is not the private same-process typed capability required by `dojo_drive_remote_evidence`. The trainer packet remains fail-closed until a production connector adapter passes those downloaded bytes directly to the validator.
