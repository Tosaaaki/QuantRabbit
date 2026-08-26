from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from counterparty_response_v4 import FEATURES, counterparty_features


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import (  # noqa: E402
    Bar,
    build_currency_graph_features,
    generate_events,
    load_bars,
    pip_size,
    sha256_file,
)


HORIZONS = (3, 12, 48)
ROLES = ("CONTINUATION_RESPONSE", "FAILED_AUCTION_REVERSAL")
SCENARIOS = {
    "RAW_SIGNAL": None,
    "EXECUTABLE_BASE": {
        "slippage_pips_one_way": 0.3,
        "commission_bps_one_way": 0.0,
        "financing_bps_per_day": 0.5,
    },
    "ADVERSE_STRESS": {
        "slippage_pips_one_way": 0.9,
        "commission_bps_one_way": 0.2,
        "financing_bps_per_day": 1.5,
    },
}


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def score_response(
    bars: list[Bar], event: dict, role: str, horizon: int, arm: str,
) -> dict | None:
    """Score a fixed-horizon response after the response bar is completed.

    Event bar i and response bar i+1 are both complete before the decision.
    The next recorded executable open is i+2. There is no same-bar TP/SL path.
    """
    if role not in ROLES or arm not in SCENARIOS:
        raise ValueError("unknown role or arm")
    i = int(event["breakout_index"])
    fill_i = i + 2
    exit_i = fill_i + horizon
    if exit_i >= len(bars):
        return None
    direction = int(event["escape_side"])
    if role == "FAILED_AUCTION_REVERSAL":
        direction *= -1
    entry, exit_bar = bars[fill_i], bars[exit_i]
    entry_mid, exit_mid = entry.mid_o, exit_bar.mid_c
    raw = exit_mid / entry_mid - 1.0 if direction > 0 else entry_mid / exit_mid - 1.0
    scenario = SCENARIOS[arm]
    if scenario is None:
        net = raw
        slip_pips = 0.0
    else:
        slip_pips = float(scenario["slippage_pips_one_way"])
        slip = slip_pips * pip_size(entry.pair)
        if direction > 0:
            entry_price = entry.ask_o + slip
            exit_price = exit_bar.bid_c - slip
            executable = exit_price / entry_price - 1.0
        else:
            entry_price = entry.bid_o - slip
            exit_price = exit_bar.ask_c + slip
            executable = entry_price / exit_price - 1.0
        elapsed_days = (_timestamp(exit_bar.time) - _timestamp(entry.time)).total_seconds() / 86400.0
        commission = 2.0 * float(scenario["commission_bps_one_way"]) * 1e-4
        financing = float(scenario["financing_bps_per_day"]) * 1e-4 * elapsed_days
        net = executable - commission - financing
    return {
        "type": "SCORE",
        "signal_id": event["signal_id"],
        "pair": entry.pair,
        "role": role,
        "arm": arm,
        "horizon": horizon,
        "breakout_time": event["breakout_time"],
        "response_completed_bar_time": bars[i + 1].time,
        "fill_time": entry.time,
        "exit_time": exit_bar.time,
        "direction": direction,
        "gross_return": raw,
        "net_return": net,
        "slippage_pips_per_side": slip_pips,
    }


def run(input_root: Path, output_root: Path, lookback: int) -> dict:
    files = sorted(input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise ValueError(f"expected exact 28-pair corpus, got {len(files)}")
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "source_ledger_counterparty_v4.jsonl.gz"
    corpus = {path.parent.name: load_bars(path) for path in files}
    graph_features = build_currency_graph_features(corpus)
    event_count = score_count = 0
    pair_audit = []
    with gzip.open(ledger_path, "wt", encoding="utf-8") as ledger:
        for path in files:
            bars = corpus[path.parent.name]
            events = [
                counterparty_features(event, bars, lookback)
                for event in generate_events(bars, lookback, graph_features)
            ]
            event_count += len(events)
            pair_audit.append({
                "pair": bars[0].pair,
                "source_sha256": sha256_file(path),
                "m5_rows": len(bars),
                "events": len(events),
            })
            for event in events:
                public = {key: value for key, value in event.items() if key != "breakout_index"}
                ledger.write(json.dumps({"type": "RAW_EVENT", **public}, sort_keys=True) + "\n")
                for role in ROLES:
                    for horizon in HORIZONS:
                        for arm in SCENARIOS:
                            scored = score_response(bars, event, role, horizon, arm)
                            if scored is not None:
                                ledger.write(json.dumps(scored, sort_keys=True) + "\n")
                                score_count += 1
    payload = {
        "study_id": "FX_COUNTERPARTY_RESPONSE_SOURCE_V4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "opened_development_not_future_holdout",
        "authority": "paper_only_local_bid_ask_no_credentials_no_order_endpoint",
        "timeframe_minutes": 5,
        "lookback_bars": lookback,
        "horizons_m5_bars": list(HORIZONS),
        "roles": list(ROLES),
        "features": list(FEATURES),
        "raw_events": event_count,
        "score_rows": score_count,
        "cost_suppressed_raw_events": 0,
        "same_signal_id_all_cost_arms": True,
        "pair_audit": pair_audit,
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result_path = output_root / "result_source_counterparty_v4.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=24)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root, args.lookback)
    print(json.dumps({key: result[key] for key in (
        "raw_events", "score_rows", "ledger_sha256", "result_sha256",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
