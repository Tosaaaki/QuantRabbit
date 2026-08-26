from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from causal_composite_indicators_v3 import ALL_FEATURES, enrich_event


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import (  # noqa: E402
    aggregate_bars,
    build_currency_graph_features,
    generate_events,
    load_bars,
    score_worker,
    sha256_file,
)


HORIZONS = (3, 6, 12, 24)
ARMS = {"RAW_SIGNAL": 0.0, "EXECUTABLE_BASE": 0.3, "ADVERSE_STRESS": 0.9}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--timeframe-minutes", type=int, required=True)
    args = parser.parse_args()
    files = sorted(args.input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise SystemExit(f"expected exact 28-pair corpus, got {len(files)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_root / "signal_ledger_composite_v3.jsonl.gz"
    corpus = {
        path.parent.name: aggregate_bars(load_bars(path), args.timeframe_minutes)
        for path in files
    }
    graph_features = build_currency_graph_features(corpus)
    raw_events = score_rows = 0
    pair_audit = []
    with gzip.open(ledger_path, "wt", encoding="utf-8") as ledger:
        for path in files:
            bars = corpus[path.parent.name]
            events = [
                enrich_event(event, bars, args.lookback)
                for event in generate_events(bars, args.lookback, graph_features)
            ]
            raw_events += len(events)
            pair_audit.append({
                "pair": bars[0].pair,
                "source_sha256": sha256_file(path),
                "aggregated_rows": len(bars),
                "events": len(events),
            })
            for event in events:
                public = {key: value for key, value in event.items() if key != "breakout_index"}
                ledger.write(json.dumps({"type": "RAW_EVENT", **public}, sort_keys=True) + "\n")
                for worker in event["workers"]:
                    for horizon in HORIZONS:
                        for arm, slippage in ARMS.items():
                            scored = score_worker(bars, event, worker, horizon, slippage)
                            if scored is None:
                                continue
                            ledger.write(json.dumps({"type": "SCORE", "arm": arm, **scored}, sort_keys=True) + "\n")
                            score_rows += 1
    result = {
        "study_id": "FX_CAUSAL_COMPOSITE_INDICATORS_V3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "opened_development_not_future_holdout",
        "authority": "paper_only_existing_local_bid_ask_no_credentials_no_order_endpoint",
        "timeframe_minutes": args.timeframe_minutes,
        "lookback_bars": args.lookback,
        "features": list(ALL_FEATURES),
        "raw_events": raw_events,
        "score_rows": score_rows,
        "pair_audit": pair_audit,
        "ledger_sha256": sha256_file(ledger_path),
        "live_authority": False,
        "external_orders": 0,
    }
    result_path = args.output_root / "result_composite_v3.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
