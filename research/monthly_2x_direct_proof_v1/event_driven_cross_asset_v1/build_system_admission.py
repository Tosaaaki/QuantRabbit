#!/usr/bin/env python3
"""Build the preregistered event-driven family System Admission report.

This scanner is deliberately limited to named, non-holdout local artifacts. It
does not fetch data, connect to a broker, run a strategy, or inspect outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

PREREG = HERE / "PREREGISTER_EVENT_DRIVEN_CROSS_ASSET_V1.json"
PARENT = ROOT / "research/monthly_2x_direct_proof_v1/MONTHLY_2X_DIRECT_PROOF_V1.json"
CALENDAR = ROOT / "data/economic_calendar.json"
CROSS_ASSET = ROOT / "data/cross_asset_snapshot.json"
CONTEXT_CHARTS = ROOT / "data/context_asset_charts.json"
EXECUTION_COVERAGE = (
    ROOT / "research/decision_time_execution_evidence/2026-08-10/coverage_report_v1.json"
)
CALENDAR_CODE = ROOT / "src/quant_rabbit/analysis/calendar.py"
CROSS_ASSET_CODE = ROOT / "src/quant_rabbit/analysis/cross_asset.py"
CONTEXT_CODE = ROOT / "src/quant_rabbit/analysis/context_assets.py"
CLI_CODE = ROOT / "src/quant_rabbit/cli.py"

REQUIRED_CONTEXT = {
    "SPX500_USD",
    "XAU_USD",
    "USB02Y_USD",
    "USB10Y_USD",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "sha256": _sha(path)}


def build_report() -> dict[str, Any]:
    prereg = _load(PREREG)
    calendar = _load(CALENDAR)
    cross_asset = _load(CROSS_ASSET)
    context = _load(CONTEXT_CHARTS)
    execution = _load(EXECUTION_COVERAGE)

    events = calendar.get("events") or []
    event_fields = sorted({key for row in events for key in row})
    actual_count = sum(row.get("actual") is not None for row in events)
    forecast_count = sum(row.get("forecast") is not None for row in events)
    timestamps = sorted(row["timestamp_utc"] for row in events if row.get("timestamp_utc"))

    assets = {row.get("instrument") for row in cross_asset.get("assets") or []}
    charts = {
        row.get("pair"): row
        for row in context.get("charts") or []
        if row.get("pair") in REQUIRED_CONTEXT
    }
    view_rows: list[dict[str, Any]] = []
    for pair in sorted(REQUIRED_CONTEXT):
        chart = charts.get(pair) or {}
        granularities: list[str] = []
        recent_counts: list[int] = []
        completed_counts: list[int] = []
        all_sided = True
        for view in chart.get("views") or []:
            candles = view.get("recent_candles") or []
            granularities.append(view.get("granularity"))
            recent_counts.append(len(candles))
            completed_counts.append(sum(row.get("complete") is True for row in candles))
            all_sided = all_sided and bool(candles) and all(
                "bid" in row and "ask" in row for row in candles
            )
        view_rows.append(
            {
                "instrument": pair,
                "granularities": granularities,
                "recent_candle_counts": recent_counts,
                "completed_counts": completed_counts,
                "has_bid_ask_on_every_candle": all_sided,
            }
        )

    cli_code = CLI_CODE.read_text()
    calendar_overwrite_writer = "_write_json(DEFAULT_CALENDAR_SNAPSHOT, calendar.to_dict())" in cli_code

    stage = execution["overall_stage_coverage"]
    gates = {
        "preregistration_precedes_admission_inventory": True,
        "historical_first_published_actuals_with_receipt_time": False,
        "prerelease_consensus_with_observation_time": False,
        "revision_safe_append_only_release_lineage": False,
        "synchronized_cross_asset_event_history": False,
        "cross_asset_bid_ask_or_native_executable_sides": False,
        "required_cross_asset_proxy_presence": REQUIRED_CONTEXT.issubset(assets),
        "reproducible_dxy_definition_present": bool(cross_asset.get("synthetic_dxy")),
        "strict_decision_time_cost_margin_unwind_coverage": (
            stage["slippage_fee_financing"] == execution["episode_count"]
            and stage["margin_exposure_concurrency"] == execution["episode_count"]
            and stage["exit_unwind"] == execution["episode_count"]
        ),
        "holdout_unread": execution["holdout_read"] is False,
        "no_live_paper_broker_order_deploy": (
            execution["live_paper_broker_order_deploy_touched"] is False
        ),
    }
    failed = sorted(key for key, value in gates.items() if not value)

    return {
        "admission_id": "EVENT_DRIVEN_CROSS_ASSET_SYSTEM_ADMISSION_V1",
        "contract": {
            "preregistration": _artifact(PREREG),
            "parent": _artifact(PARENT),
            "contract_id": prereg["contract_id"],
        },
        "inspection_boundary": {
            "local_read_only_named_sources": True,
            "strategy_outcomes_read": False,
            "holdout_read": False,
            "network_data_acquisition": False,
            "live_paper_broker_order_deploy": False,
        },
        "source_inventory": {
            "economic_calendar": {
                **_artifact(CALENDAR),
                "generated_at_utc": calendar.get("generated_at_utc"),
                "source_url": calendar.get("source_url"),
                "event_count": len(events),
                "timestamp_min": timestamps[0] if timestamps else None,
                "timestamp_max": timestamps[-1] if timestamps else None,
                "event_fields": event_fields,
                "actual_non_null_count": actual_count,
                "forecast_non_null_count": forecast_count,
                "provider_receipt_timestamp_field": False,
                "first_published_value_marker": False,
                "revision_lineage": False,
                "append_only_history": False,
                "writer_is_snapshot_overwrite": calendar_overwrite_writer,
            },
            "cross_asset_snapshot": {
                **_artifact(CROSS_ASSET),
                "generated_at_utc": cross_asset.get("generated_at_utc"),
                "granularity": cross_asset.get("granularity"),
                "requested_candle_count": cross_asset.get("candle_count"),
                "stored_asset_rows_are_scalar_aggregates": True,
                "stored_bar_timestamps": False,
                "required_proxy_assets_present": sorted(REQUIRED_CONTEXT & assets),
                "dxy_mode": "SYNTHETIC_CURRENT_AGGREGATE",
            },
            "context_asset_charts": {
                **_artifact(CONTEXT_CHARTS),
                "generated_at_utc": context.get("generated_at_utc"),
                "snapshot_count": 1,
                "view_rows": view_rows,
                "all_required_views_have_bid_ask": bool(view_rows)
                and all(row["has_bid_ask_on_every_candle"] for row in view_rows),
                "append_only_history": False,
            },
            "inherited_execution_coverage": {
                **_artifact(EXECUTION_COVERAGE),
                "episode_count": execution["episode_count"],
                "strict_eligible": execution["strict_eligible"],
                "overall_stage_coverage": stage,
            },
            "lineage_code": [
                _artifact(CALENDAR_CODE),
                _artifact(CROSS_ASSET_CODE),
                _artifact(CONTEXT_CODE),
            ],
        },
        "admission_gates": gates,
        "failed_admission_gates": failed,
        "replay": {
            "started": False,
            "grid_points_evaluated": 0,
            "metrics": {
                "rolling_30d_multiple_after_all_costs": None,
                "train_lcb": None,
                "validation_paired_lcb": None,
                "profit_factor": None,
                "maximum_drawdown_jpy": None,
                "gross_margin_jpy": None,
            },
            "reason": "System Admission failed before any outcome or holdout read.",
        },
        "classification": "NOT_EVALUABLE",
        "parent_target_status": "TARGET_PATH_NOT_YET_PROVEN",
        "dominant_blocker": (
            "No append-only, revision-safe macro release ledger binds first-published actuals "
            "and prerelease consensus to provider receipt times; synchronized executable "
            "cross-asset history and strict cost/margin/unwind evidence are also absent."
        ),
        "next_action": "Execute ACQUISITION_CONTRACT_V1 in a separately authorized phase, then rerun System Admission without reading holdout.",
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    rendered = canonical_json(build_report())
    if args.check:
        if args.check.read_text() != rendered:
            raise SystemExit(f"regeneration mismatch: {args.check}")
        print(f"PASS byte-identical {_sha(args.check)}")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
