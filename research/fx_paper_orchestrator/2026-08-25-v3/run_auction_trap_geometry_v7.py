from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, generate_events, load_bars, pip_size, sha256_file  # noqa: E402


LOOKBACK = 24
HORIZON = 24
PERIODS = {
    "TUNING_DIAGNOSTIC_ONLY": ("2026-03-11", "2026-05-01"),
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
}
ARMS = {
    "RAW_SIGNAL": None,
    "EXECUTABLE_BASE": {"slippage": 0.3, "commission": 0.0, "financing": 0.5},
    "ADVERSE_STRESS": {"slippage": 0.9, "commission": 0.2, "financing": 1.5},
}


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event_energy_ratio(bars: list[Bar], index: int) -> float:
    closes = [bar.mid_c for bar in bars[index - LOOKBACK:index + 1]]
    returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
    baseline = statistics.median(value * value for value in returns[:-1]) or 1e-18
    return returns[-1] * returns[-1] / baseline


def qualifies(bars: list[Bar], event: dict) -> tuple[bool, dict]:
    i = int(event["breakout_index"])
    response = bars[i + 1]
    side = int(event["escape_side"])
    energy_ratio = event_energy_ratio(bars, i)
    response_return = math.log(response.mid_c / bars[i].mid_c)
    diagnostics = {
        "boundary_touch_count": round(float(event["boundary_crowding"]) * LOOKBACK),
        "event_energy_ratio": energy_ratio,
        "escape_scale": float(event["rail_escape_energy"]),
        "response_inside_scale": max(0.0, -float(event["next_boundary_distance"])),
        "response_opposite_log_return": -side * response_return,
        "spread_relaxation_ratio": bars[i].spread_c / max(response.spread_c, 1e-18),
    }
    accepted = (
        "CONFIRMED_REJECTION" in event["workers"]
        and diagnostics["boundary_touch_count"] >= 2
        and energy_ratio >= 1.0
        and diagnostics["escape_scale"] >= 0.10
        and diagnostics["response_inside_scale"] >= 0.10
        and diagnostics["response_opposite_log_return"] > 0.0
        and diagnostics["spread_relaxation_ratio"] >= 1.0
    )
    return accepted, diagnostics


def score(bars: list[Bar], event: dict, arm: str) -> dict | None:
    i = int(event["breakout_index"])
    fill_i, exit_i = i + 2, i + 2 + HORIZON
    if exit_i >= len(bars):
        return None
    direction = -int(event["escape_side"])
    entry, exit_bar = bars[fill_i], bars[exit_i]
    if direction > 0:
        gross = exit_bar.mid_c / entry.mid_o - 1.0
    else:
        gross = entry.mid_o / exit_bar.mid_c - 1.0
    scenario = ARMS[arm]
    if scenario is None:
        net = gross
    else:
        slip = float(scenario["slippage"]) * pip_size(entry.pair)
        if direction > 0:
            net = (exit_bar.bid_c - slip) / (entry.ask_o + slip) - 1.0
        else:
            net = (entry.bid_o - slip) / (exit_bar.ask_c + slip) - 1.0
        elapsed_days = (timestamp(exit_bar.time) - timestamp(entry.time)).total_seconds() / 86400.0
        net -= 2.0 * float(scenario["commission"]) * 1e-4
        net -= float(scenario["financing"]) * 1e-4 * elapsed_days
    path = bars[fill_i:exit_i + 1]
    if direction > 0:
        mfe = max(bar.mid_h / entry.mid_o - 1.0 for bar in path)
        mae = min(bar.mid_l / entry.mid_o - 1.0 for bar in path)
    else:
        mfe = max(entry.mid_o / bar.mid_l - 1.0 for bar in path)
        mae = min(entry.mid_o / bar.mid_h - 1.0 for bar in path)
    return {
        "arm": arm,
        "gross_return": gross,
        "net_return": net,
        "mfe_return": mfe,
        "mae_return": mae,
    }


def summarize(rows: list[dict], start: str, end: str) -> dict:
    selected = [
        row for row in rows
        if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]
    by_arm = {}
    for arm in ARMS:
        values = [row["scores"][arm]["net_return"] for row in selected]
        by_arm[arm] = {
            "signals": len(values),
            "mean_return": statistics.fmean(values) if values else None,
            "median_return": statistics.median(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
            "additive_return": sum(values),
        }
    return {"start": start, "end": end, "arms": by_arm}


def run(input_root: Path, output_root: Path) -> dict:
    files = sorted(input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise ValueError(f"expected exact 28-pair corpus, got {len(files)}")
    proposals = []
    raw_breakouts = 0
    pair_audit = []
    for path in files:
        bars = load_bars(path)
        events = generate_events(bars, LOOKBACK)
        raw_breakouts += len(events)
        accepted = 0
        for event in events:
            is_accepted, diagnostics = qualifies(bars, event)
            if not is_accepted:
                continue
            scored = {arm: score(bars, event, arm) for arm in ARMS}
            if any(value is None for value in scored.values()):
                continue
            accepted += 1
            proposals.append({
                "signal_id": f"ATG24::{event['signal_id']}",
                "pair": event["pair"],
                "breakout_time": event["breakout_time"],
                "response_completed_time": bars[int(event["breakout_index"]) + 1].time,
                "fill_time": bars[int(event["breakout_index"]) + 2].time,
                "exit_time": bars[int(event["breakout_index"]) + 2 + HORIZON].time,
                "direction": -int(event["escape_side"]),
                "horizon_m5_bars": HORIZON,
                "diagnostics": diagnostics,
                "scores": scored,
            })
        pair_audit.append({"pair": bars[0].pair, "source_sha256": sha256_file(path), "bars": len(bars), "signals": accepted})
    proposals.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    periods = {name: summarize(proposals, *bounds) for name, bounds in PERIODS.items()}
    walk = periods["WALK_FORWARD"]["arms"]
    development_admitted = (
        walk["RAW_SIGNAL"]["signals"] >= 20
        and all(walk[arm]["mean_return"] is not None and walk[arm]["mean_return"] > 0 for arm in ARMS)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_auction_trap_geometry_v7.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in proposals), encoding="utf-8")
    payload = {
        "experiment": "FX_AUCTION_TRAP_GEOMETRY_H24_V7",
        "evidence_class": "opened_development_not_future_holdout",
        "raw_breakouts": raw_breakouts,
        "raw_signals": len(proposals),
        "cost_suppressed_raw_signals": 0,
        "same_signal_id_all_cost_arms": True,
        "horizon_m5_bars": HORIZON,
        "periods": periods,
        "development_admitted": development_admitted,
        "final_admitted": False,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "pair_audit": pair_audit,
        "terminal_open_inventory": 0,
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
    result = output_root / "result_auction_trap_geometry_v7.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "raw_breakouts": result["raw_breakouts"],
        "raw_signals": result["raw_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
