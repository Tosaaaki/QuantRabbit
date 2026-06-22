#!/usr/bin/env python3
"""Package bid/ask candle replay validation rules for runtime use.

The source report is produced by ``scripts/oanda_history_replay_validate.py``.
This packager refuses to publish partial price-truth runs by default, so a
ranked candle result cannot silently become the runtime evidence base before
all scored samples have OANDA bid/ask truth.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_REPORT = Path("logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json")
DEFAULT_OUTPUT = Path("src/quant_rabbit/bidask_replay_precision_rules.json")

RULE_SECTIONS = (
    "edge_rules",
    "daily_stable_edge_rules",
    "contrarian_edge_rules",
    "daily_stable_contrarian_edge_rules",
    "negative_rules",
    "rejected_sampled_segments",
    "rejected_contrarian_segments",
    "rejected_daily_stability_segments",
)

TRUTH_FIELDS = (
    "status",
    "reason",
    "adoption_level",
    "candidate_rule_validation_blocked",
    "global_currency_validation_blocked",
    "raw_directional_rows",
    "deduped_directional_rows",
    "evaluated_rows",
    "unscorable_no_market_rows",
    "required_min_evaluated_rows",
    "history_files",
    "history_candles",
    "missing_price_truth_samples",
    "missing_price_window_group_count",
    "unscorable_no_market_window_group_count",
    "future_price_truth_window_group_count",
)


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    packaged = package_payload(
        payload,
        source_report=args.source_report,
        allow_partial=bool(args.allow_partial),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packaged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="package even when price_truth_coverage.status is not PRICE_TRUTH_OK",
    )
    return parser.parse_args()


def package_payload(
    payload: dict[str, Any],
    *,
    source_report: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    precision = payload.get("precision_rules")
    if not isinstance(precision, dict):
        raise ValueError("source report has no precision_rules object")
    truth = payload.get("price_truth_coverage")
    if not isinstance(truth, dict):
        raise ValueError("source report has no price_truth_coverage object")
    truth_status = str(truth.get("status") or "").upper()
    if truth_status != "PRICE_TRUTH_OK" and not allow_partial:
        raise ValueError(
            "refusing to package partial bid/ask replay evidence: "
            f"price_truth_coverage.status={truth_status or 'UNKNOWN'}"
        )

    packaged: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": payload.get("generated_at_utc"),
        "generated_from": "scripts/oanda_history_replay_validate.py",
        "packaged_by": "scripts/package_bidask_replay_precision_rules.py",
        "source_report": str(source_report),
        "granularity": payload.get("granularity"),
        "truth_source": payload.get("truth_source"),
        "price_truth_coverage": _copy_fields(truth, TRUTH_FIELDS),
        "forecast_sample_coverage_summary": _forecast_sample_coverage_summary(
            payload.get("forecast_sample_coverage")
        ),
        "selection": copy.deepcopy(precision.get("selection") or {}),
        "adoption_summary": copy.deepcopy(precision.get("adoption_summary") or {}),
    }
    for section in RULE_SECTIONS:
        rows = precision.get(section)
        packaged[section] = copy.deepcopy(rows if isinstance(rows, list) else [])
    return packaged


def _copy_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: copy.deepcopy(payload[field]) for field in fields if field in payload}


def _forecast_sample_coverage_summary(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    under_sampled = raw.get("under_sampled_pair_directions")
    return {
        "min_directional_samples_for_precision_rule": raw.get(
            "min_directional_samples_for_precision_rule"
        ),
        "min_active_days_for_daily_stability": raw.get("min_active_days_for_daily_stability"),
        "pair_count": raw.get("pair_count"),
        "pair_direction_count": raw.get("pair_direction_count"),
        "unscorable_no_market_samples": raw.get("unscorable_no_market_samples"),
        "under_sampled_pair_directions": len(under_sampled) if isinstance(under_sampled, list) else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
