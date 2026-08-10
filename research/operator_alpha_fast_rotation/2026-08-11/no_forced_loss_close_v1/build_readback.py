#!/usr/bin/env python3
"""Build compact, deterministic readback artifacts from replay outputs."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    comparison_path = ROOT / "comparison_v1.json"
    rows_path = ROOT / "decision_results_v1.jsonl"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['arm']}::{row['account_mode']}"].append(row)

    fields = [
        "arm_account_mode", "scheduled_decisions", "executed", "after_cost_terminal_equity_pre_financing_jpy",
        "after_cost_net_pre_financing_jpy", "profit_factor_pre_financing", "expectancy_pre_financing_jpy",
        "max_realized_sequence_drawdown_jpy", "max_equity_drawdown_within_trade_jpy", "max_open_risk_jpy",
        "minimum_margin_excess_jpy", "peak_broker_margin_jpy", "peak_double_gross_margin_jpy",
        "recovery_rate", "median_recovery_seconds", "max_recovery_seconds", "median_holding_seconds",
        "unrecovered_rate", "unresolved_inventory_count", "margin_closeout_count", "margin_closeout_rate",
        "profit_only_original_close_ratio", "repeated_hedge_count", "netting_reduction_events",
        "estimated_spread_plus_slippage_jpy", "cost_ratio_to_gross_profit", "turnover_notional_jpy",
        "original_realized_jpy", "hedge_realized_jpy", "original_terminal_mtm_jpy", "hedge_terminal_mtm_jpy",
        "unknown_financing_count", "accounting_status", "opportunities_per_observed_weekday",
        "projected_200_trade_net_pre_financing_jpy", "decision",
    ]
    with (ROOT / "comparison_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, summary in sorted(comparison["comparisons"].items()):
            writer.writerow({"arm_account_mode": key, **{field: summary.get(field) for field in fields[1:]}})

    with (ROOT / "mtm_inclusive_equity_curve_v1.jsonl").open("w", encoding="utf-8") as handle:
        for key, summary in sorted(comparison["comparisons"].items()):
            for point in summary["equity_curve"]:
                handle.write(json.dumps({"arm_account_mode": key, "curve_semantics": "sequential terminal realized_plus_final_bid_ask_MTM_pre_financing", **point}, sort_keys=True) + "\n")

    distributions: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        executed = [row for row in group if row.get("executed")]
        recovered = [float(row["recovery_seconds"]) for row in executed if row.get("recovery_seconds") is not None]
        distributions[key] = {
            "executed": len(executed),
            "recovered": len(recovered),
            "unrecovered": sum(bool(row.get("original_open") or row.get("hedge_open")) for row in executed),
            "p50_recovery_seconds": median(recovered) if recovered else None,
            "p90_recovery_seconds": percentile(recovered, 0.90),
            "max_recovery_seconds": max(recovered) if recovered else None,
        }
    (ROOT / "return_time_distribution_v1.json").write_text(json.dumps(distributions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selected = [
        "A_HARD_SL_BASELINE::HEDGING", "B_NO_SL_NAKED_RETURN_WAIT::HEDGING",
        "C_NO_SL_DELAYED_ENTRY_EARLY_TP::HEDGING", "D_NO_SL_HEDGE_RETURN_050::HEDGING",
        "E_NO_SL_PARTIAL_PROFIT_BE::HEDGING", "F_NO_SL_MULTI_PAIR_ROTATION::HEDGING",
        "H1_LOCK_AT_ADVERSE_LEVEL_AND_WAIT_050::HEDGING", "H2_HEDGE_TP_KEEP_ORIGINAL_050::HEDGING",
        "H3_HEDGE_PARTIAL_TP_REHEDGE_050::HEDGING", "H4_HEDGE_REVERSAL_CONFIRM_EXIT_050::HEDGING",
        "H5_HEDGE_PROFIT_OFFSET_ORIGINAL_BE_050::HEDGING", "H6_PERSISTENT_TREND_STRESS::HEDGING",
        "H7_GAP_AND_FINANCING_STRESS::HEDGING",
    ]
    table = [
        "| arm (hedging account) | exec | net pre-financing JPY | PF | max seq DD | max intra-trade DD | unresolved | financing unknown | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in selected:
        row = comparison["comparisons"][key]
        table.append("| " + " | ".join([
            key.split("::", 1)[0], str(row["executed"]), fmt(row["after_cost_net_pre_financing_jpy"]),
            fmt(row["profit_factor_pre_financing"], 3), fmt(row["max_realized_sequence_drawdown_jpy"]),
            fmt(row["max_equity_drawdown_within_trade_jpy"]), str(row["unresolved_inventory_count"]),
            str(row["unknown_financing_count"]), row["decision"],
        ]) + " |")
    verdict = """# NO_FIXED_SL same-cohort verdict

## Result

No no-fixed-SL arm is eligible for adoption. Several hedging-account arms have
positive **pre-financing** terminal contribution, but every such row carries
unresolved original/hedge inventory and/or unknown financing. Under the
preregistered rule, that is `REJECT`, not a win. Netting-account opposite
orders are reductions with realized original loss, not hedges.

""" + "\n".join(table) + """

## Interpretation

- The hard-SL comparison lost money, but had zero terminal inventory. It is
  still `NOT_EVALUABLE` for full after-financing comparison on ten trades.
- Delaying entry and taking profit early is the only unhedged positive row,
  matching the manual fast-profit behavior, but four positions remained open
  and sixteen crossed an unknown-financing boundary. The apparent +JPY result
  is therefore rejected rather than promoted.
- H3 at 0.5 ATR reports positive realized+MTM before financing, but leaves
  seven inventories unresolved, has 27 financing-unknown executions and 42
  hedge entries. This is precisely the tail-risk/inventory deferral forbidden
  by the contract.
- Persistent-trend and gap/financing stress turn the selected H3 shape sharply
  negative. No margin closeout occurred at fixed 5,000 units; that is only a
  size-specific observation, not evidence that the mechanism cannot fail.
- The manual wins held for seconds/minutes and rotated after confirmed closes.
  Multi-day recovery waits and terminal inventory are a behavioral mismatch,
  so they do not reproduce `OPERATOR_ALPHA_FAST_ROTATION_V1`.

## Evidence boundary

The repository-required `scripts/replay_exit_workers_groups.py` entrypoint is
absent from HEAD, `origin/codex/qr-python-ecosystem-audit-20260810`, and
`origin/main`. The real-data run here is a research-local M1 bid/ask replay and
must not be represented as the standard QR exit-worker replay. Financing is
not zero-filled, and F cross-pair concurrent portfolio margin is not jointly
simulated; F remains non-adoptable.

## Reproduction

```bash
python3 -m unittest -v research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/test_accounting_oracle.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/run_no_forced_loss_replay.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/verify_independent_oracle.py
python3 research/operator_alpha_fast_rotation/2026-08-11/no_forced_loss_close_v1/build_readback.py
```
"""
    (ROOT / "verdict_v1.md").write_text(verdict, encoding="utf-8")

    artifact_names = [
        "preregister_v1.json", "no_forced_loss_close_contract_v1.json", "frozen_cohort_v1.json",
        "decision_results_v1.jsonl", "comparison_v1.json", "comparison_v1.csv",
        "mtm_inclusive_equity_curve_v1.jsonl", "return_time_distribution_v1.json",
        "independent_oracle_v1.json", "verdict_v1.md",
    ]
    manifest = {}
    for name in artifact_names:
        payload = (ROOT / name).read_bytes()
        manifest[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    readback = {
        "contract": "NO_FORCED_LOSS_CLOSE_READBACK_V1",
        "generated_artifacts_readable": True,
        "independent_oracle_pass": json.loads((ROOT / "independent_oracle_v1.json").read_text())["pass"],
        "standard_replay_status": comparison["standard_replay_status"],
        "adoption": "REJECT_ALL_NO_FIXED_SL_ARMS",
        "artifact_manifest": manifest,
    }
    (ROOT / "readback_v1.json").write_text(json.dumps(readback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(manifest) + 1, "oracle": readback["independent_oracle_pass"], "adoption": readback["adoption"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
