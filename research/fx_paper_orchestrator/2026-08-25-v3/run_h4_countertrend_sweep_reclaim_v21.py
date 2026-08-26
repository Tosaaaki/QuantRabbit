from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS
from run_tuning_worker_admission_v17 import simulate_selected


H4_BARS = 48
M15_BARS = 3
RAIL_BARS = 12
HORIZON = 192


def signal_at(bars: list[Bar], index: int) -> dict | None:
    """Return a decision from completed bars through index, never from the fill bar."""
    if index < H4_BARS or index + 1 + HORIZON >= len(bars):
        return None
    h4_log_return = math.log(bars[index].mid_c / bars[index - H4_BARS].mid_c)
    direction = 1 if h4_log_return > 0 else -1 if h4_log_return < 0 else 0
    if direction == 0:
        return None
    countertrend_m15 = direction * math.log(bars[index - 1].mid_c / bars[index - 1 - M15_BARS].mid_c)
    if countertrend_m15 >= 0:
        return None
    rail = bars[index - 1 - RAIL_BARS:index - 1]
    if len(rail) != RAIL_BARS:
        return None
    rail_low = min(bar.mid_l for bar in rail)
    rail_high = max(bar.mid_h for bar in rail)
    event = bars[index - 1]
    response = bars[index]
    response_return = math.log(response.mid_c / event.mid_c)
    if direction > 0:
        swept = event.mid_l < rail_low
        reclaimed = response.mid_c > rail_low and response_return > 0
        sweep_distance = rail_low - event.mid_l
        reclaim_distance = response.mid_c - rail_low
    else:
        swept = event.mid_h > rail_high
        reclaimed = response.mid_c < rail_high and response_return < 0
        sweep_distance = event.mid_h - rail_high
        reclaim_distance = rail_high - response.mid_c
    if not (swept and reclaimed):
        return None
    return {
        "direction": direction,
        "h4_log_return": h4_log_return,
        "countertrend_m15_signed_log_return": countertrend_m15,
        "sweep_distance_price": sweep_distance,
        "reclaim_distance_price": reclaim_distance,
        "event_time": event.time,
        "response_completed_time": response.time,
    }


def raw_path_metrics(bars: list[Bar], entry_index: int, direction: int) -> dict:
    entry = bars[entry_index]
    expiry = bars[entry_index + HORIZON]
    path = bars[entry_index:entry_index + HORIZON + 1]
    if direction > 0:
        gross = expiry.mid_c / entry.mid_o - 1.0
        mfe = max(bar.mid_h / entry.mid_o - 1.0 for bar in path)
        mae = min(bar.mid_l / entry.mid_o - 1.0 for bar in path)
    else:
        gross = entry.mid_o / expiry.mid_c - 1.0
        mfe = max(entry.mid_o / bar.mid_l - 1.0 for bar in path)
        mae = min(entry.mid_o / bar.mid_h - 1.0 for bar in path)
    return {"gross_return": gross, "mfe_return": mfe, "mae_return": mae}


def summarize_raw(rows: list[dict], start: str, end: str) -> dict:
    selected = [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]
    gross = [row["raw_path"]["gross_return"] for row in selected]
    mfe = [row["raw_path"]["mfe_return"] for row in selected]
    mae = [row["raw_path"]["mae_return"] for row in selected]
    return {
        "signals": len(selected),
        "mean_gross_return": statistics.fmean(gross) if gross else None,
        "median_gross_return": statistics.median(gross) if gross else None,
        "direction_accuracy": sum(value > 0 for value in gross) / len(gross) if gross else None,
        "mean_mfe_return": statistics.fmean(mfe) if mfe else None,
        "mean_mae_return": statistics.fmean(mae) if mae else None,
        "break_even_roundtrip_cost": statistics.fmean(gross) if gross else None,
    }


def run(input_root: Path, output_root: Path) -> dict:
    corpus, source_audit, rows = {}, [], []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        corpus[pair] = bars
        pair_signals = 0
        for i in range(H4_BARS, len(bars) - HORIZON - 1):
            signal = signal_at(bars, i)
            if signal is None:
                continue
            fill_index = i + 1
            pair_signals += 1
            rows.append({
                "signal_id": f"H4CTSR192::{pair}::{bars[i].time}",
                "pair": pair,
                "decision_time": bars[i].time,
                "fill_time": bars[fill_index].time,
                "exit_time": bars[fill_index + HORIZON].time,
                "direction": signal["direction"],
                "horizon_m5_bars": HORIZON,
                "diagnostics": signal,
                "raw_path": raw_path_metrics(bars, fill_index, signal["direction"]),
            })
        source_audit.append({
            "pair": pair,
            "source_sha256": sha256_file(matches[0]),
            "bars": len(bars),
            "signals": pair_signals,
        })
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    periods = {
        name: {
            "raw_diagnostics": summarize_raw(rows, start, end),
            **{arm: simulate_selected(corpus, rows, set(UNIVERSE), arm, start, end) for arm in ARMS},
        }
        for name, (start, end) in PERIODS.items()
    }
    admitted = all(
        periods[name]["RAW_SIGNAL"]["source_signals"] >= 20
        and periods[name][arm]["equity_multiple"] > 1.0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_h4_countertrend_sweep_reclaim_v21.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_H4_COUNTERTREND_SWEEP_RECLAIM_V21",
        "family": "H4_CONTEXT_SHORT_TIMEFRAME_EXECUTION",
        "family_hypotheses": 1,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "new_h4_context_m15_countertrend_m5_sweep_reclaim_signal_family",
        "indicator": {
            "h4_bars": H4_BARS,
            "m15_bars": M15_BARS,
            "rail_bars": RAIL_BARS,
            "horizon_m5_bars": HORIZON,
            "cost_used_for_signal": False,
            "future_outcome_used_for_signal": False,
        },
        "raw_signals": len(rows),
        "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "periods": periods,
        "source_audit": source_audit,
        "development_admitted": admitted,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "live_authority": False,
        "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_h4_countertrend_sweep_reclaim_v21.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "raw_signals": result["raw_signals"],
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
