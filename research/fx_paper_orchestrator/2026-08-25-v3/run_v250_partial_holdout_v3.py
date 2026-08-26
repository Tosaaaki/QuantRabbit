from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V250_DIR = ROOT / "research/llm_paper_experiment/2026-08-24-v250"
V252_DIR = ROOT / "research/llm_paper_experiment/2026-08-24-v252"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V250 = import_file("frozen_v250_for_partial_holdout", V250_DIR / "run_expectancy_regression_v250.py")
V252 = import_file("frozen_v252_for_partial_holdout", V252_DIR / "run_validation_account_v252.py")
V245 = V250.V249.V245


PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")
CANDIDATE = "M15_H8:ridge:P0.0"
HOLDOUT_START = pd.Timestamp("2026-05-01T00:00:00Z")
HOLDOUT_END = pd.Timestamp("2026-08-01T00:00:00Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_for_local_source(input_root: Path) -> dict:
    outputs = {}
    for pair in PAIRS:
        files = sorted((input_root / pair).glob(f"{pair}_M5_BA_*.jsonl.gz"))
        if len(files) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(files)}")
        path = files[0].resolve()
        outputs[pair] = {"segments": [{"path": str(path), "sha256": sha256_file(path)}]}
    return {"outputs": outputs}


def load_common_pair_data(manifest: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    raw = {pair: V245.load_pair(manifest, pair) for pair in PAIRS}
    common = raw[PAIRS[0]].index
    union = raw[PAIRS[0]].index
    for pair in PAIRS[1:]:
        common = common.intersection(raw[pair].index)
        union = union.union(raw[pair].index)
    if common.empty or not common.is_monotonic_increasing:
        raise ValueError("empty or non-monotonic common source index")
    pair_data = {pair: raw[pair].loc[common].copy() for pair in PAIRS}
    audit = {
        "raw_rows_by_pair": {pair: int(len(frame)) for pair, frame in raw.items()},
        "common_rows": int(len(common)),
        "union_rows": int(len(union)),
        "common_fraction": float(len(common) / len(union)),
        "first": common[0].isoformat(),
        "last": common[-1].isoformat(),
        "missing_rows_imputed": 0,
    }
    return pair_data, audit


def full_month_multiples(monthly: dict[str, float]) -> dict[str, float]:
    # The source ends on 15 July. May and June are the only complete calendar
    # months inside the frozen May-Jul holdout interval.
    return {month: value for month, value in monthly.items() if month in {"2026-05", "2026-06"}}


def run(input_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    v250_report = json.loads((V250_DIR / "report_v250_001.json").read_text())
    v245_contract = json.loads((ROOT / "research/llm_paper_experiment/2026-08-24-v245/contract_v245.json").read_text())
    v252_contract = json.loads((V252_DIR / "contract_v252.json").read_text())
    v252_report = json.loads((V252_DIR / "report_v252_001.json").read_text())
    actual_decision_path = ROOT / "research/llm_paper_experiment/2026-08-24-v252/actual_llm_decision_v252_001.json"
    actual_decision = json.loads(actual_decision_path.read_text())
    if actual_decision["policy"] != "H8_ONLY" or float(actual_decision["gross_cap"]) != 12.0:
        raise ValueError("frozen actual-LLM policy drift")

    model_path = V250_DIR / "models_v250_001/M15_H8_ridge.joblib"
    claimed_model = v250_report["models"]["M15_H8:ridge"]
    if sha256_file(model_path) != claimed_model["sha256"]:
        raise ValueError("frozen V250 model hash drift")
    model = joblib.load(model_path)
    manifest = manifest_for_local_source(input_root)
    pair_data, source_audit = load_common_pair_data(manifest)

    config = v245_contract["workers"]["M15"]
    frame = V250.V249.build_worker_frame(pair_data, "M15", config, v245_contract["execution"])[8]
    feature_columns = list(v250_report["feature_columns"])
    eligible = frame[
        (frame["decision_time"] >= HOLDOUT_START)
        & (frame["decision_time"] < HOLDOUT_END)
        & (frame["exit_time"] < HOLDOUT_END)
    ].dropna(subset=feature_columns).copy()
    prediction = V250.predict_model("ridge", model, eligible, feature_columns)
    decisions = V250.decision_rows(eligible, prediction, "ridge", 0.0, "reserved_holdout_partial")
    decisions = decisions[decisions["candidate"] == CANDIDATE].copy()
    if decisions.empty:
        raise ValueError("frozen candidate produced no partial-holdout decisions")

    ledger_path = output_root / "decision_ledger_v250_partial_holdout_v3.jsonl.gz"
    with gzip.open(ledger_path, "wt", encoding="utf-8") as handle:
        for row in decisions.sort_values(["decision_time", "pair"]).to_dict(orient="records"):
            handle.write(json.dumps({
                key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for key, value in row.items()
            }, separators=(",", ":")) + "\n")

    end = min(HOLDOUT_END, pair_data[PAIRS[0]].index[-1] + pd.Timedelta(minutes=5))
    capacity = {CANDIDATE: int(v252_report["validation_max_concurrent"][CANDIDATE])}
    weights = {CANDIDATE: 1.0}
    account_results = {}
    for scenario_name in ("normal", "adverse"):
        account_results[scenario_name] = V252.simulate(
            decisions, pair_data, v245_contract["execution"][scenario_name], scenario_name,
            200000.0, 12.0, weights, capacity,
            float(v252_contract["hard_guards"]["pair_gross_leverage"]),
            float(v252_contract["hard_guards"]["gross_leverage"]),
            HOLDOUT_START, end,
        )
        full = full_month_multiples(account_results[scenario_name]["monthly_multiples"])
        account_results[scenario_name]["full_comparable_months"] = full
        account_results[scenario_name]["full_month_minimum_multiple"] = min(full.values()) if full else None
        account_results[scenario_name]["full_months_at_or_above_2x"] = sum(value >= 2.0 for value in full.values())

    signal_summary = V250.summarize(decisions, int(v250_report["candidate_count"]))
    result = {
        "experiment": "V3_FROZEN_V250_PARTIAL_RESERVED_HOLDOUT",
        "evidence_class": "reserved_holdout_opened_partial_not_complete",
        "authority": "local_paper_replay_only_no_credentials_no_broker_no_order_endpoint",
        "candidate": CANDIDATE,
        "frozen_model_sha256": sha256_file(model_path),
        "frozen_actual_llm_decision_sha256": sha256_file(actual_decision_path),
        "frozen_actual_llm_policy": actual_decision,
        "source_audit": source_audit,
        "source_sha256_by_pair": {
            pair: manifest["outputs"][pair]["segments"][0]["sha256"] for pair in PAIRS
        },
        "holdout_contract": {
            "start": HOLDOUT_START.isoformat(), "planned_end": HOLDOUT_END.isoformat(),
            "available_end_exclusive": end.isoformat(),
            "complete": bool(end >= HOLDOUT_END),
            "opened_now": True,
        },
        "decisions": int(len(decisions)),
        "signal_summary": signal_summary,
        "account_results": account_results,
        "monthly_2x_proven": bool(
            end >= HOLDOUT_END
            and all(result["full_month_minimum_multiple"] >= 2.0 for result in account_results.values())
        ),
        "profit_guaranteed": False,
        "admitted": False,
        "admission_blockers": [
            "original V250 family admitted no candidate",
            "reserved holdout source ends before 2026-08-01",
            "a partial opened holdout cannot certify future returns",
            "full comparable months do not meet 2x in both cost arms",
        ],
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "live_authority": False,
        "external_orders": 0,
    }
    result["result_sha256"] = V245.canonical_sha(result)
    result_path = output_root / "result_v250_partial_holdout_v3.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "decisions": result["decisions"],
        "signal_summary": result["signal_summary"],
        "account_results": result["account_results"],
        "monthly_2x_proven": result["monthly_2x_proven"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
