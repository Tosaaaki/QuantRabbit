"""V27 paper replay for the frozen V26 strategy with complete timestamp compatibility.

The trading hypothesis is unchanged from the unobserved V26 attempt.  This
module supplies only runtime compatibility and cycle identity; the frozen V26
selector, signal contract, portfolio simulator, and rejection policy remain the
strategy implementation.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import run_auction_trap_geometry_v7 as auction_v7
import run_causal_min_spread_representative_v26 as frozen_v26
import run_portfolio_episode_netting_v15 as portfolio_v15


CYCLE_ID = "V27"
EXPERIMENT = "FX_CAUSAL_MIN_SPREAD_REPRESENTATIVE_V27"
RUNTIME_COMPATIBILITY_PROVENANCE = {
    "classification": "NON_STRATEGY_RUNTIME_COMPATIBILITY",
    "changed_strategy_variables": 0,
    "parser_contract": "CANONICAL_UTC_0_TO_9_FRACTIONAL_DIGITS_EXACT_INTEGER_EPOCH_NANOSECONDS",
    "patched_reachable_bindings": [
        "run_causal_min_spread_representative_v26.parse_time",
        "run_portfolio_episode_netting_v15.timestamp",
        "run_auction_trap_geometry_v7.timestamp",
    ],
    "v26_rerun_permitted": False,
}
_UTC_TIMESTAMP = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


@dataclass(frozen=True, order=True)
class EpochNanoseconds:
    value: int

    def __sub__(self, other: "EpochNanoseconds") -> "NanosecondDelta":
        if not isinstance(other, EpochNanoseconds):
            return NotImplemented
        return NanosecondDelta(self.value - other.value)


@dataclass(frozen=True)
class NanosecondDelta:
    value: int

    def total_seconds(self) -> float:
        return self.value / 1_000_000_000


def parse_utc_nanoseconds(value: str) -> EpochNanoseconds:
    """Parse canonical UTC timestamps as exact integer epoch nanoseconds."""
    match = _UTC_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"V27 timestamp is not canonical UTC: {value}")
    fraction = match.group("fraction") or ""
    seconds = datetime.strptime(match.group("head"), "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    epoch_seconds = calendar.timegm(seconds.utctimetuple())
    return EpochNanoseconds(epoch_seconds * 1_000_000_000 + int(fraction.ljust(9, "0") or "0"))


def install_timestamp_compatibility() -> None:
    """Patch every timestamp-parser binding reachable from the V26 call graph."""
    frozen_v26.parse_time = parse_utc_nanoseconds
    portfolio_v15.timestamp = parse_utc_nanoseconds
    auction_v7.timestamp = parse_utc_nanoseconds


def causal_score(row: dict, bars: list, time_index: dict[str, int]) -> float:
    """Delegate the frozen causal spread score without adding strategy inputs."""
    return frozen_v26.causal_score(row, bars, time_index)


def apply_rule(parent_rows: list[dict], corpus: dict[str, list]) -> list[dict]:
    """Delegate the one frozen V26 turnover-reduction rule."""
    return frozen_v26.apply_rule(parent_rows, corpus)


def _relabel_comparison(payload: dict) -> None:
    for period in payload["metric_comparison_vs_v25"].values():
        for arm in period.values():
            for metric in arm.values():
                metric["V27"] = metric.pop("V26")


def _add_margin_metrics(payload: dict) -> None:
    margin_fraction = payload["portfolio"]["rule_max_gross_leverage"]
    for period in payload["periods"].values():
        for arm_name in frozen_v26.ARMS:
            metrics = period[arm_name]
            has_execution = metrics["executed_signals"] > 0
            metrics["max_gross_exposure_nav"] = margin_fraction if has_execution else 0.0
            metrics["max_margin_requirement_jpy_at_1x"] = (
                frozen_v26.INITIAL_EQUITY_JPY * margin_fraction if has_execution else 0.0
            )


def run(input_root: Path, parent_ledger: Path, parent_result: Path, output_root: Path) -> dict:
    install_timestamp_compatibility()
    payload = frozen_v26.run(input_root, parent_ledger, parent_result, output_root)

    old_ledger = output_root / "proposal_ledger_causal_min_spread_representative_v26.jsonl"
    ledger = output_root / "proposal_ledger_causal_min_spread_representative_v27.jsonl"
    old_ledger.replace(ledger)
    payload["cycle_id"] = CYCLE_ID
    payload["experiment"] = EXPERIMENT
    payload["proposal_ledger"] = str(ledger)
    payload["proposal_ledger_sha256"] = frozen_v26.sha256_file(ledger)
    payload["runtime_compatibility_provenance"] = RUNTIME_COMPATIBILITY_PROVENANCE
    _relabel_comparison(payload)
    _add_margin_metrics(payload)
    payload["result_sha256"] = frozen_v26.embedded_hash(payload, "result_sha256")

    result = output_root / "result_causal_min_spread_representative_v27.json"
    result.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "result_causal_min_spread_representative_v26.json").unlink()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_ledger, args.parent_result, args.output_root)
    print(json.dumps({
        "cycle_id": result["cycle_id"],
        "raw_signals": result["raw_signals"],
        "execution_selected_signals": result["execution_selected_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "automatic_rejection": result["automatic_rejection"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
