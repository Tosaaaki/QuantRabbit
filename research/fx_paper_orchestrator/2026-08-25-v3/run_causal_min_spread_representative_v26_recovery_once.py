"""One-shot, paper-only launcher for the authorized V26 recovery.

The coordinator must durably record ``RECOVERY_ATTEMPT_STARTED`` before this
launcher can run.  This file does not alter any strategy field; it installs
only the separately frozen timestamp compatibility parser and then delegates
to the original V26 command-line entry point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_causal_min_spread_representative_v26 as frozen_v26
import run_causal_min_spread_representative_v26_recovery as recovery


ROOT = Path(__file__).resolve().parent
AUTHORIZATION = ROOT / "V26_RECOVERY_AUTHORIZATION.json"
STATE = ROOT / "evidence/orchestrator_state_v2/state.json"
RESULT = ROOT / (
    "evidence/run_causal_min_spread_representative_v26_official_001/"
    "result_causal_min_spread_representative_v26.json"
)
LEDGER = ROOT / (
    "evidence/run_causal_min_spread_representative_v26_official_001/"
    "proposal_ledger_causal_min_spread_representative_v26.jsonl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_one_shot_intent() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if authorization.get("cycle_id") != "V26" or authorization.get("authorized") is not True:
        raise RuntimeError("V26 recovery authorization is absent or invalid")
    if authorization.get("scope") != "ONE_TIMESTAMP_ONLY_PAPER_RECOVERY_ATTEMPT":
        raise RuntimeError("V26 recovery authorization scope changed")
    if authorization.get("recovery_attempt_limit") != 1:
        raise RuntimeError("V26 recovery attempt limit changed")
    if authorization.get("one_shot_launcher_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V26 one-shot launcher hash changed")
    authority = authorization.get("authority", {})
    if authority.get("paper_only") is not True or any(authority.get(key) is not False for key in (
        "live_authority", "broker_account_access", "credential_access", "order_endpoint",
        "deploy", "external_config_mutation",
    )) or authority.get("external_orders") != 0:
        raise RuntimeError("V26 recovery authority is not paper-only/zero-order")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    cycle = state.get("cycles", {}).get("V26", {})
    if cycle.get("status") != "RECOVERY_ATTEMPT_STARTED" \
            or cycle.get("official_attempts") != 1 \
            or cycle.get("recovery_attempts") != 1:
        raise RuntimeError("V26 one-shot recovery intent was not durably registered")
    if cycle.get("recovery_authorization_sha256") != sha256_file(AUTHORIZATION):
        raise RuntimeError("V26 recovery state is not bound to this authorization")
    if RESULT.exists() or LEDGER.exists():
        raise RuntimeError("V26 recovery outputs already exist; second execution forbidden")


def main() -> int:
    validate_one_shot_intent()
    recovery.install_timestamp_compatibility()
    return frozen_v26.main()


if __name__ == "__main__":
    raise SystemExit(main())
