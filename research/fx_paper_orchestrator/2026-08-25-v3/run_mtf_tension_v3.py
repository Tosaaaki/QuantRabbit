from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from indicator_factory_v3 import canonical, metrics, period
from run_v250_partial_holdout_v3 import (
    ROOT, V250, V250_DIR, load_common_pair_data, manifest_for_local_source, sha256_file,
)


WORKERS = (
    "M15_DIRECTION_ALL",
    "M15_H1_AGREE",
    "M15_H4_AGREE",
    "MTF_UNANIMOUS",
    "H4_H1_PULLBACK_CONTINUATION",
)
def model_keys(m15_horizon: int) -> dict[str, str]:
    if m15_horizon not in (8, 16):
        raise ValueError("m15_horizon must be one of the frozen 8 or 16 bar workers")
    return {"M15": f"M15_H{m15_horizon}:ridge", "H1": "H1_H6:ridge", "H4": "H4_H6:ridge"}


def model_directions(frame: pd.DataFrame, model, features: list[str], prefix: str) -> pd.DataFrame:
    valid = frame.dropna(subset=features).copy()
    prediction = np.asarray(model.predict(valid[features]), dtype=float)
    valid[f"{prefix}_pred_long"] = prediction[:, 0]
    valid[f"{prefix}_pred_short"] = prediction[:, 1]
    valid[f"{prefix}_direction"] = np.where(prediction[:, 0] >= prediction[:, 1], 1, -1)
    valid[f"{prefix}_tension"] = prediction[:, 0] - prediction[:, 1]
    return valid


def attach_latest_context(m15: pd.DataFrame, higher: pd.DataFrame, prefix: str) -> pd.DataFrame:
    parts = []
    columns = ["decision_time", f"{prefix}_direction", f"{prefix}_tension"]
    for pair, left in m15.groupby("pair"):
        right = higher[higher["pair"] == pair][columns].sort_values("decision_time")
        parts.append(pd.merge_asof(
            left.sort_values("decision_time"), right, on="decision_time",
            direction="backward", allow_exact_matches=True,
        ))
    return pd.concat(parts, ignore_index=True).sort_values(["decision_time", "pair"]).reset_index(drop=True)


def raw_mid_return(row: pd.Series, pair_data: dict[str, pd.DataFrame], direction: int) -> float:
    entry = pair_data[row["pair"]].loc[row["entry_time"]]
    exit_ = pair_data[row["pair"]].loc[row["exit_time"]]
    entry_mid = (entry["bid_o"] + entry["ask_o"]) * .5
    exit_mid = (exit_["bid_o"] + exit_["ask_o"]) * .5
    return float(exit_mid / entry_mid - 1.0 if direction > 0 else entry_mid / exit_mid - 1.0)


def selected_direction(row: pd.Series, worker: str) -> int | None:
    m15, h1, h4 = int(row["M15_direction"]), int(row["H1_direction"]), int(row["H4_direction"])
    if worker == "M15_DIRECTION_ALL":
        return m15
    if worker == "M15_H1_AGREE":
        return m15 if m15 == h1 else None
    if worker == "M15_H4_AGREE":
        return m15 if m15 == h4 else None
    if worker == "MTF_UNANIMOUS":
        return m15 if m15 == h1 == h4 else None
    if worker == "H4_H1_PULLBACK_CONTINUATION":
        return h1 if h1 == h4 and m15 != h1 else None
    raise ValueError(worker)


def build_records(frame: pd.DataFrame, pair_data: dict[str, pd.DataFrame]) -> list[dict]:
    records = []
    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        for worker in WORKERS:
            direction = selected_direction(series, worker)
            if direction is None:
                continue
            records.append({
                "signal_id": hashlib.sha256(
                    f"{worker}|{series['pair']}|{series['decision_time'].isoformat()}".encode()
                ).hexdigest()[:24],
                "worker": worker, "pair": series["pair"],
                "fill_time": series["entry_time"].isoformat(),
                "exit_time": series["exit_time"].isoformat(),
                "direction": direction,
                "returns": {
                    "RAW_SIGNAL": raw_mid_return(series, pair_data, direction),
                    "EXECUTABLE_BASE": float(
                        series["normal_long"] if direction > 0 else series["normal_short"]
                    ),
                    "ADVERSE_STRESS": float(
                        series["adverse_long"] if direction > 0 else series["adverse_short"]
                    ),
                },
                "tension": {
                    "M15": float(series["M15_tension"]),
                    "H1": float(series["H1_tension"]),
                    "H4": float(series["H4_tension"]),
                },
            })
    return records


def evaluate(records: list[dict], output_root: Path, source_audit: dict, model_audit: dict,
             m15_horizon: int) -> dict:
    family_tests = len(WORKERS)
    z = statistics.NormalDist().inv_cdf(1 - .05 / (2 * family_tests))
    candidates = []
    for worker in WORKERS:
        rows = [row for row in records if row["worker"] == worker]
        periods = {
            p: [row for row in rows if period(row["fill_time"]) == p]
            for p in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC")
        }
        summaries = {
            p: {arm: metrics(xs, arm, z) for arm in (
                "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
            )} for p, xs in periods.items()
        }
        corrected = [
            summaries["TUNING"]["RAW_SIGNAL"]["family_corrected_cluster_lower_bps"],
            summaries["WALK_FORWARD"]["EXECUTABLE_BASE"]["family_corrected_cluster_lower_bps"],
            summaries["WALK_FORWARD"]["ADVERSE_STRESS"]["family_corrected_cluster_lower_bps"],
        ]
        candidates.append({
            "candidate_id": f"FX_MTF_TENSION_{worker}_H{m15_horizon}_V3",
            "worker": worker, "entry_timeframe": "M15", "holding_m15_bars": m15_horizon,
            "periods": summaries,
            "development_admitted": all(value is not None and value > 0 for value in corrected),
            "ranking_floor_bps": min((value for value in corrected if value is not None), default=-math.inf),
        })
    candidates.sort(key=lambda item: (item["development_admitted"], item["ranking_floor_bps"]), reverse=True)
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "signal_ledger_mtf_tension_v3.jsonl"
    ledger_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    payload = {
        "experiment": "FX_MTF_CAUSAL_TENSION_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "signal_generation_uses_predicted_cost_floor": False,
        "raw_signals_suppressed_by_cost": 0,
        "same_signal_id_all_cost_arms": True,
        "family_tests": family_tests,
        "family_corrected_z": z,
        "raw_signal_rows": len(records),
        "candidates": candidates,
        "development_admitted_count": sum(item["development_admitted"] for item in candidates),
        "final_admitted_count": 0,
        "source_audit": source_audit,
        "frozen_model_hashes": model_audit,
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "terminal_inventory_mtm_hidden": False,
        "admission_blockers": [
            "the hierarchy was designed after the 2026 source was opened",
            "portfolio inventory and terminal liquidation are not yet replayed",
            "future holdout has not elapsed",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    (output_root / "result_mtf_tension_v3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def run(input_root: Path, output_root: Path, m15_horizon: int) -> dict:
    frozen_report = json.loads((V250_DIR / "report_v250_001.json").read_text())
    base_contract = json.loads((ROOT / "research/llm_paper_experiment/2026-08-24-v245/contract_v245.json").read_text())
    manifest = manifest_for_local_source(input_root)
    pair_data, source_audit = load_common_pair_data(manifest)
    features = list(frozen_report["feature_columns"])
    frames = {}
    model_audit = {}
    keys = model_keys(m15_horizon)
    for timeframe, worker_key in (("M15", m15_horizon), ("H1", 6), ("H4", 6)):
        frame = V250.V249.build_worker_frame(
            pair_data, timeframe, base_contract["workers"][timeframe], base_contract["execution"]
        )[worker_key]
        model_key = keys[timeframe]
        model_path = ROOT / frozen_report["models"][model_key]["path"]
        model_hash = sha256_file(model_path)
        if model_hash != frozen_report["models"][model_key]["sha256"]:
            raise ValueError(f"model drift: {model_key}")
        model_audit[model_key] = model_hash
        frames[timeframe] = model_directions(frame, joblib.load(model_path), features, timeframe)
    merged = attach_latest_context(frames["M15"], frames["H1"], "H1")
    merged = attach_latest_context(merged, frames["H4"], "H4").dropna(
        subset=["H1_direction", "H4_direction"]
    )
    records = build_records(merged, pair_data)
    return evaluate(records, output_root, source_audit, model_audit, m15_horizon)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m15-horizon", type=int, required=True, choices=(8, 16))
    args = parser.parse_args()
    result = run(args.input_root, args.output_root, args.m15_horizon)
    print(json.dumps({
        "raw_signal_rows": result["raw_signal_rows"],
        "development_admitted": result["development_admitted_count"],
        "final_admitted": result["final_admitted_count"],
        "candidates": result["candidates"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
