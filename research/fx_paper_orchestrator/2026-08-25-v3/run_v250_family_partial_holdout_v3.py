from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from run_v250_partial_holdout_v3 import (
    HOLDOUT_END,
    HOLDOUT_START,
    ROOT,
    V250,
    V250_DIR,
    load_common_pair_data,
    manifest_for_local_source,
    sha256_file,
)


def run(input_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    contract = json.loads((V250_DIR / "contract_v250.json").read_text())
    frozen_report = json.loads((V250_DIR / "report_v250_001.json").read_text())
    base_contract = json.loads((ROOT / "research/llm_paper_experiment/2026-08-24-v245/contract_v245.json").read_text())
    if int(contract["candidate_count"]) != 54:
        raise ValueError("frozen family size drift")
    manifest = manifest_for_local_source(input_root)
    pair_data, source_audit = load_common_pair_data(manifest)
    features = list(frozen_report["feature_columns"])
    decision_frames = []
    model_audit = {}
    for timeframe, config in base_contract["workers"].items():
        frames = V250.V249.build_worker_frame(pair_data, timeframe, config, base_contract["execution"])
        for horizon, frame in frames.items():
            worker = f"{timeframe}_H{horizon}"
            eligible = frame[
                (frame["decision_time"] >= HOLDOUT_START)
                & (frame["decision_time"] < HOLDOUT_END)
                & (frame["exit_time"] < HOLDOUT_END)
            ].dropna(subset=features)
            for model_name in ("ridge", "hist_gradient_boosting"):
                key = f"{worker}:{model_name}"
                model_path = ROOT / frozen_report["models"][key]["path"]
                actual_hash = sha256_file(model_path)
                if actual_hash != frozen_report["models"][key]["sha256"]:
                    raise ValueError(f"frozen model hash drift: {key}")
                model_audit[key] = actual_hash
                model = joblib.load(model_path)
                prediction = V250.predict_model(model_name, model, eligible, features)
                for floor in contract["predicted_net_return_floors"]:
                    decision_frames.append(V250.decision_rows(
                        eligible, prediction, model_name, float(floor), "reserved_holdout_partial"
                    ))
    ledger = pd.concat(decision_frames, ignore_index=True)
    ledger = ledger.sort_values(["decision_time", "candidate", "pair"]).reset_index(drop=True)
    ledger_path = output_root / "decision_ledger_v250_family_partial_holdout_v3.jsonl.gz"
    with gzip.open(ledger_path, "wt", encoding="utf-8") as handle:
        for row in ledger.to_dict(orient="records"):
            handle.write(json.dumps({
                key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for key, value in row.items()
            }, separators=(",", ":")) + "\n")

    summaries = {}
    for candidate in sorted(ledger["candidate"].unique()):
        partial = V250.summarize(ledger[ledger["candidate"] == candidate], int(contract["candidate_count"]))
        prior = frozen_report["candidates"].get(candidate, {})
        development = prior.get("development_walk", {})
        validation = prior.get("validation", {})
        partial_lcb = partial.get("family_corrected_lcb")
        summaries[candidate] = {
            "partial_holdout": partial,
            "frozen_validation": validation,
            "frozen_development_walk": development,
            "partial_family_lcb_positive": bool(partial_lcb is not None and partial_lcb > 0),
            "sign_consistent_all_opened_periods": bool(
                validation.get("adverse_mean_return", -math.inf) > 0
                and development.get("adverse_mean_return", -math.inf) > 0
                and partial.get("adverse_mean_return", -math.inf) > 0
            ),
        }
    ranked = sorted(
        summaries,
        key=lambda key: (
            summaries[key]["partial_holdout"].get("family_corrected_lcb")
            if summaries[key]["partial_holdout"].get("family_corrected_lcb") is not None
            else -math.inf,
            summaries[key]["partial_holdout"].get("adverse_mean_return", -math.inf),
        ),
        reverse=True,
    )
    result = {
        "experiment": "V3_FROZEN_V250_FAMILY_PARTIAL_RESERVED_HOLDOUT",
        "evidence_class": "reserved_holdout_opened_partial_not_complete",
        "authority": "local_paper_replay_only_no_credentials_no_broker_no_order_endpoint",
        "registered_candidate_count": int(contract["candidate_count"]),
        "observed_candidate_count": len(summaries),
        "decision_rows": int(len(ledger)),
        "source_audit": source_audit,
        "model_hashes": model_audit,
        "partial_family_lcb_positive_count": sum(
            value["partial_family_lcb_positive"] for value in summaries.values()
        ),
        "sign_consistent_all_opened_periods": [
            key for key, value in summaries.items() if value["sign_consistent_all_opened_periods"]
        ],
        "ranked_partial_holdout": ranked,
        "candidate_summaries": summaries,
        "admitted": [],
        "admission_blockers": [
            "the frozen V250 validation admitted no candidate",
            "the reserved holdout is incomplete after 2026-07-15",
            "opened partial holdout rankings cannot be used to retune this family",
        ],
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "live_authority": False,
        "external_orders": 0,
    }
    result["result_sha256"] = V250.V249.V245.canonical_sha(result)
    result_path = output_root / "result_v250_family_partial_holdout_v3.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    top = []
    for candidate in result["ranked_partial_holdout"][:10]:
        value = result["candidate_summaries"][candidate]
        top.append({"candidate": candidate, **value["partial_holdout"],
                    "sign_consistent": value["sign_consistent_all_opened_periods"]})
    print(json.dumps({
        "registered": result["registered_candidate_count"],
        "observed": result["observed_candidate_count"],
        "partial_family_lcb_positive": result["partial_family_lcb_positive_count"],
        "sign_consistent_count": len(result["sign_consistent_all_opened_periods"]),
        "top": top,
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
