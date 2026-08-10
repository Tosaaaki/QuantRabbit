#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = ROOT / "data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json"
payload = json.loads(source.read_text())
rows = payload["sample_replay_details"]
checks = {
    "source_sha": hashlib.sha256(source.read_bytes()).hexdigest() == "3ad6ee8feb3db5c79016a13c4ea8f13812b01d0fd9c8616a1dcba1822b15fed6",
    "four_distinct_trades": len({row["trade_id"] for row in rows}) == 4,
    "all_realized_positive": all(float(row["realized_pl_jpy"]) > 0 for row in rows),
    "all_replay_wins": all(row["replay_win"] and not row["replay_loss"] for row in rows),
    "touch_order_causal": all(row["first_entry_touch_utc"] <= row["first_tp_touch_after_entry_utc"] for row in rows),
    "net_reconciles": abs(sum(float(row["realized_pl_jpy"]) for row in rows) - 3255.0938) < 1e-6,
    "vehicle_not_mixed": all(not row["market_close_mixed_in"] and not row["market_or_stop_vehicle_mixed_in"] for row in rows),
}
out = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values())}
(HERE / "independent_oracle_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
if not out["all_passed"]:
    raise SystemExit(json.dumps(out, ensure_ascii=False))
