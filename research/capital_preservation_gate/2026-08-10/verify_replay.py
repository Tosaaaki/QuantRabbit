from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / "receipts_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]
report = json.loads((HERE / "report_v1.json").read_text(encoding="utf-8"))


def canonical_sha(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


checks = {
    "one_receipt_per_frozen_decision": len(rows) == 251 == report["episodes"],
    "decision_ids_unique": len({row["decision_id"] for row in rows}) == 251,
    "no_new_exposure": all(not row["new_exposure_permitted"] for row in rows),
    "no_realized_outcome_in_decision": all(row["realized_outcome_used"] is False for row in rows),
    "live_permission_never_granted": all(row["live_permission_granted"] is False for row in rows),
    "missing_not_zero": all(row["missing_stages"] for row in rows),
    "report_actions_match": dict(sorted(Counter(row["action"] for row in rows).items())) == report["actions"],
    "receipt_hash_matches": sha256((HERE / "receipts_v1.jsonl").read_bytes()).hexdigest() == report["receipts_sha256"],
    "each_receipt_self_hash_matches": all(
        row["receipt_sha256"] == canonical_sha({key: value for key, value in row.items() if key != "receipt_sha256"})
        for row in rows
    ),
    "profit_not_claimed": report["profit_generation_status"] == "NOT_PROVEN" and report["market_no_loss_guarantee"] is False,
}
result = {"contract": "CAPITAL_PRESERVATION_GATE_V1_INDEPENDENT_ORACLE", "checks": checks, "passed": sum(checks.values()), "total": len(checks)}
(HERE / "independent_oracle_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not all(checks.values()):
    raise SystemExit(json.dumps(result, ensure_ascii=False))
print(json.dumps(result, ensure_ascii=False))
