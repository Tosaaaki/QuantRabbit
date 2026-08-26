from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_liquid_major_universe_v9 import UNIVERSE
from run_opposite_close_only_v16 import simulate_pair
from run_portfolio_episode_netting_v15 import PERIODS


TUNING = ("2026-03-11", "2026-05-01")


def select_workers(tuning_audit: dict[str, dict[str, dict]]) -> list[str]:
    return sorted(
        pair for pair, arms in tuning_audit.items()
        if arms["RAW_SIGNAL"]["source_signals"] >= 20
        and arms["RAW_SIGNAL"]["sleeve_equity_multiple"] > 1.0
        and arms["ADVERSE_STRESS"]["sleeve_equity_multiple"] > 1.0
    )


def simulate_selected(corpus, rows, enabled, arm, start, end):
    pair_marks, pair_audit = {}, {}
    for pair in sorted(UNIVERSE):
        pair_rows = rows if pair in enabled else []
        pair_marks[pair], pair_audit[pair] = simulate_pair(pair, corpus[pair], pair_rows, arm, start, end)
    common = set.intersection(*(set(values) for values in pair_marks.values()))
    if not common:
        raise ValueError("pair mark timelines have no common timestamps")
    equity_path = [statistics.fmean(pair_marks[pair][stamp] for pair in sorted(UNIVERSE)) for stamp in sorted(common)]
    peak, max_drawdown = equity_path[0], 0.0
    for value in equity_path:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    opens = sum(item["opens"] for item in pair_audit.values())
    closes = sum(item["closes"] for item in pair_audit.values())
    return {
        "equity_multiple": equity_path[-1], "max_drawdown": max_drawdown,
        "enabled_workers": sorted(enabled), "cash_sleeves": len(UNIVERSE) - len(enabled),
        "source_signals": sum(item["source_signals"] for item in pair_audit.values()),
        "position_opens": opens, "position_closes": closes,
        "turnover_nav": (opens + closes) / len(UNIVERSE),
        "ignored_same_direction": sum(item["ignored_same_direction"] for item in pair_audit.values()),
        "opposite_close_only": sum(item["opposite_close_only"] for item in pair_audit.values()),
        "terminal_open_inventory": sum(item["terminal_open_inventory"] for item in pair_audit.values()),
        "pair_audit": pair_audit,
    }


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    rows = [{key: row[key] for key in ("signal_id", "pair", "fill_time", "exit_time", "direction")} for row in raw_source]
    corpus, source_audit = {}, []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    tuning_audit = {
        pair: {
            arm: simulate_pair(pair, corpus[pair], rows, arm, *TUNING)[1]
            for arm in ARMS
        }
        for pair in sorted(UNIVERSE)
    }
    enabled = select_workers(tuning_audit)
    periods = {
        name: {arm: simulate_selected(corpus, rows, enabled, arm, start, end) for arm in ARMS}
        for name, (start, end) in PERIODS.items()
    }
    admitted = len(enabled) >= 2 and all(
        periods[name][arm]["equity_multiple"] > 1.0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    payload = {
        "experiment": "FX_AUCTION_TRAP_TUNING_WORKER_ADMISSION_V17",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "tuning_only_worker_admission",
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "exit_time", "direction"],
        "source_outcome_fields_consumed": False,
        "worker_admission": {
            "tuning_period": list(TUNING), "minimum_signals": 20,
            "rule": "raw sleeve >1 and adverse sleeve >1", "tuning_audit": tuning_audit,
            "enabled_workers": enabled, "disabled_workers": sorted(UNIVERSE - set(enabled)),
            "walk_forward_used": False,
        },
        "allocation": {"weight_per_enabled_pair": 1/7, "cash_for_disabled": True, "gross_leverage_cap": len(enabled)/7},
        "periods": periods, "source_audit": source_audit,
        "cost_suppressed_raw_signals": 0, "same_signal_stream_all_cost_arms": True,
        "development_admitted": admitted, "final_admitted": False,
        "terminal_inventory_mtm_hidden": False, "live_authority": False, "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    result = output_root / "result_tuning_worker_admission_v17.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.source_ledger, args.output_root)
    print(json.dumps({
        "worker_admission": result["worker_admission"], "allocation": result["allocation"],
        "periods": result["periods"], "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
