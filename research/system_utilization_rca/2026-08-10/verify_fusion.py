#!/usr/bin/env python3
"""Independent stdlib oracle for the generated fusion artifacts."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_rows(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inference = load_rows("inference_table_v1.jsonl")
    outcomes = load_rows("outcome_table_v1.jsonl")
    fused = load_rows("fused_decisions_v1.jsonl")
    report = json.loads((HERE / "utilization_report_v1.json").read_text())
    manifest = json.loads((HERE / "run_manifest_v1.json").read_text())
    checks = {
        "outcomes_251": len(outcomes) == 251,
        "fused_251": len(fused) == 251,
        "unique_decisions": len({row["decision_id"] for row in fused}) == 251,
        "separate_outcome_boundary": all("actual_after_cost_net" not in row for row in inference),
        "full_251_sum_checkpoint": abs(sum(row["actual_after_cost_net"] for row in outcomes) - (-18039.7866)) < 1e-6,
        "64d_validation_sum_checkpoint": abs(
            report["fusion"]["QUADRUPLE_64D"]["fused_evidence_admissible"]["all_trades_net_jpy"] - 15144.4802
        ) < 1e-6,
        "no_trade_without_constraints": not any(row["action"] == "TRADE" for row in fused),
        "holdout_unread": report["holdout_read"] is False and manifest["holdout_read"] is False,
        "hashes_match": all(sha(HERE / name) == digest for name, digest in manifest["outputs"].items()),
        "no_profit_attribution": report["profitability_increment_attributed_to_fusion_jpy"] == 0.0,
        "no_all_trades_fallback": report["causal_bottleneck"]["all_trades_fallback_detected"] is False,
    }
    payload = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "ok": all(checks.values())}
    (HERE / "independent_oracle_v1.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
