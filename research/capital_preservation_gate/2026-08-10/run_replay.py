from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from capital_preservation import DecisionInput, evaluate


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "research/decision_time_execution_evidence/2026-08-10/evidence_ledger_v1.jsonl"
EPISODES = ROOT / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    evidence = read_jsonl(EVIDENCE)
    episodes = {row["episode_id"]: row for row in read_jsonl(EPISODES) if row.get("label_status") == "ACTUAL_AFTER_COST"}
    if len(evidence) != 251 or len(episodes) != 251:
        raise RuntimeError("frozen cohort must remain exactly 251 rows")

    receipts = []
    for row in evidence:
        receipt = evaluate(
            DecisionInput(
                decision_id=row["decision_id"],
                decision_time=row["decision_time"],
                source_sha=row["source_sha"],
                stage_coverage=row["stage_coverage"],
                equity_jpy=None,
                peak_equity_jpy=None,
                daily_gross_loss_spent_jpy=None,
                candidate_loss_bound_jpy=None,
                expected_after_cost_lcb_jpy=None,
            )
        )
        receipts.append(receipt)

    output = HERE / "receipts_v1.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts), encoding="utf-8")
    actions = Counter(row["action"] for row in receipts)
    reasons = Counter(reason for row in receipts for reason in row["reason_codes"])
    legacy_labels = [episodes[row["decision_id"]]["net_jpy"] for row in receipts]
    report = {
        "contract": "CAPITAL_PRESERVATION_GATE_V1",
        "status": "CAPITAL_FLOOR_ACTIVE_EDGE_NOT_PROVEN",
        "source_sha256": {"evidence": file_sha(EVIDENCE), "episodes": file_sha(EPISODES)},
        "episodes": len(receipts),
        "actions": dict(sorted(actions.items())),
        "new_exposure_permitted": sum(row["new_exposure_permitted"] for row in receipts),
        "top_reason_codes": reasons.most_common(),
        "strict_policy_realized_net_jpy": 0.0,
        "strict_policy_realized_max_drawdown_jpy": 0.0,
        "legacy_episode_label_net_jpy_diagnostic_only": sum(legacy_labels),
        "legacy_label_warning": "Known incomplete allocation of DAILY_FINANCING and partial closes; not an accepted financial oracle.",
        "profit_generation_status": "NOT_PROVEN",
        "market_no_loss_guarantee": False,
        "capital_preservation_invariant": "No new exposure is permitted when any required evidence, positive after-cost LCB, or bounded-loss input is missing.",
        "holdout_read": False,
        "live_permission_granted": False,
    }
    report["receipts_sha256"] = file_sha(output)
    (HERE / "report_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
