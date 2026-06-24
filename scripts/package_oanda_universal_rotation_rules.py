#!/usr/bin/env python3
"""Package mined OANDA rotation report rows for runtime fallback.

The full mining report lives under ``logs/`` and is intentionally not tracked.
Live runtime therefore needs a compact tracked fallback that preserves the
rank-only rule sections consumed by ``forecast_precision`` plus the compact
campaign firepower summary consumed by the profitability and campaign gates.
When a latest report only contains a smaller top-N excerpt while its summary
still proves a broader qualified universe, existing packaged rule rows are kept
so a firepower refresh cannot silently shrink runtime forecast coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_REPORT = Path("logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json")
DEFAULT_OUTPUT = Path("src/quant_rabbit/oanda_universal_rotation_precision_rules.json")

RULE_SECTIONS = (
    "high_precision_directional_selectors",
    "high_precision_inversion_selectors",
    "high_precision_multi_confluences",
    "high_precision_pair_confluences",
    "qualified_directional_selectors",
    "qualified_inversion_selectors",
    "qualified_multi_confluences",
    "qualified_pair_confluences",
)

COUNT_KEYS = (
    "high_precision_directional_selector_count",
    "high_precision_inversion_selector_count",
    "high_precision_multi_confluence_count",
    "high_precision_pair_confluence_count",
    "qualified_directional_selector_count",
    "qualified_inversion_selector_count",
    "qualified_multi_confluence_count",
    "qualified_pair_confluence_count",
)

CONFIG_KEYS = (
    "min_positive_day_rate",
    "min_validation_win_rate",
    "multi_confluence_min_samples",
    "multi_confluence_sizes",
    "inversion_selector_min_samples",
    "inversion_selector_confluence_sizes",
)

ROW_FIELDS = (
    "active_days",
    "confluence",
    "confluence_size",
    "exit_shape",
    "feature_a",
    "feature_b",
    "feature_c",
    "feature_d",
    "feature_e",
    "feature_f",
    "feature_g",
    "feature_h",
    "pair",
    "positive_day_rate",
    "qualification",
    "selected_side",
    "selection_basis",
    "shape",
    "side",
    "source_shape",
    "source_side",
    "source_train_avg_realized_atr",
    "source_train_n",
    "source_train_profit_factor",
    "source_train_win_rate",
    "source_validation_avg_realized_atr",
    "source_validation_avg_realized_pips",
    "source_validation_n",
    "source_validation_profit_factor",
    "source_validation_win_rate",
    "source_validation_win_wilson95_lower",
    "train_n",
    "train_win_rate",
    "validation_avg_realized_atr",
    "validation_avg_realized_pips",
    "validation_inversion_edge_atr",
    "validation_n",
    "validation_profit_factor",
    "validation_win_rate",
    "validation_win_wilson95_lower",
)


def main() -> int:
    args = _parse_args()
    source_reports = args.source_report or [DEFAULT_SOURCE_REPORT]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in source_reports]
    payload = payloads[0] if len(payloads) == 1 else merge_payloads(payloads, source_reports=source_reports)
    packaged = package_payload(payload, source_report=source_reports[0])
    if len(source_reports) > 1:
        packaged["source_report"] = "merged_oanda_universal_rotation_reports"
        packaged["source_reports"] = [str(path) for path in source_reports]
    preserve_path = args.preserve_from or args.output
    if preserve_path.exists():
        try:
            existing = json.loads(preserve_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            existing = {}
        if isinstance(existing, dict):
            report_is_narrower = _packaged_report_is_narrower(packaged, existing)
            preserve_existing_campaign_firepower(
                packaged,
                existing,
                report_is_narrower=report_is_narrower,
            )
            preserve_existing_scope_metadata(
                packaged,
                existing,
                report_is_narrower=report_is_narrower,
            )
            preserve_existing_rule_rows(
                packaged,
                existing,
                report_is_narrower=report_is_narrower,
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
    parser.add_argument(
        "--source-report",
        type=Path,
        action="append",
        help=(
            "mining report to package; may be repeated to merge deterministic pair-shard reports. "
            "Defaults to logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preserve-from",
        type=Path,
        help="existing packaged rules to preserve broader coverage from; defaults to --output",
    )
    return parser.parse_args()


def merge_payloads(payloads: list[dict[str, Any]], *, source_reports: list[Path]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no OANDA universal rotation payloads to merge")
    merged: dict[str, Any] = {
        "generated_at_utc": max(
            str(payload.get("generated_at_utc") or "") for payload in payloads
        ),
        "generated_from": "scripts/oanda_universal_rotation_miner.py",
        "merged_by": "scripts/package_oanda_universal_rotation_rules.py",
        "source_reports": [str(path) for path in source_reports],
    }
    for key in ("source", "truth_source", "contract"):
        value = next((payload.get(key) for payload in payloads if payload.get(key)), None)
        if value is not None:
            merged[key] = value
    config: dict[str, Any] = {}
    for payload in payloads:
        payload_config = payload.get("config")
        if isinstance(payload_config, dict):
            config = _merge_scope_config(payload_config, config)
    if config:
        merged["config"] = config
    for key in ("history_files", "history_pairs", "scored_outcomes", "inversion_scored_outcomes", *COUNT_KEYS):
        merged[key] = sum(_optional_int(payload.get(key)) or 0 for payload in payloads)
    for key in ("history_files_discovered", "history_pairs_discovered"):
        values = [_optional_int(payload.get(key)) for payload in payloads]
        values = [value for value in values if value is not None]
        if values:
            merged[key] = max(values)
    discovered_pairs: list[Any] = []
    selected_pairs: list[Any] = []
    for payload in payloads:
        for pair in payload.get("history_pairs_discovered_order") or []:
            if pair not in discovered_pairs:
                discovered_pairs.append(pair)
        selection = payload.get("history_pair_selection")
        if isinstance(selection, dict):
            for pair in selection.get("selected_pairs") or []:
                if pair not in selected_pairs:
                    selected_pairs.append(pair)
    if discovered_pairs:
        merged["history_pairs_discovered_order"] = discovered_pairs
    if selected_pairs:
        merged["history_pair_selection"] = {
            "selected_pairs": selected_pairs,
            "selected_pair_count": len(selected_pairs),
            "is_partial_pair_scan": len(selected_pairs) < len(discovered_pairs) if discovered_pairs else False,
            "source_report_count": len(payloads),
        }
    for section in RULE_SECTIONS:
        rows: list[Any] = []
        for payload in payloads:
            rows = _merge_rule_rows(list(payload.get(section) or []), rows)
        merged[section] = rows
    campaign_firepower = _merge_campaign_firepower(payloads)
    if campaign_firepower:
        merged["campaign_firepower"] = campaign_firepower
    return merged


def package_payload(payload: dict[str, Any], *, source_report: Path) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    packaged: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": payload.get("generated_at_utc"),
        "generated_from": "scripts/oanda_universal_rotation_miner.py",
        "packaged_by": "scripts/package_oanda_universal_rotation_rules.py",
        "source_report": str(source_report),
        "summary": _summary(payload),
        "config": {key: config[key] for key in CONFIG_KEYS if key in config},
    }
    campaign_firepower = payload.get("campaign_firepower")
    if isinstance(campaign_firepower, dict):
        packaged["campaign_firepower"] = campaign_firepower
    for section in RULE_SECTIONS:
        packaged[section] = [_package_row(row) for row in payload.get(section) or []]
    return packaged


def _merge_campaign_firepower(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    high_precision_vehicles = _unique_firepower_vehicles(payloads, bucket="high_precision")
    evidence_queue_vehicles = _unique_firepower_vehicles(payloads, bucket="evidence_queue")
    high_precision = _merge_firepower_bucket(high_precision_vehicles)
    evidence_queue = _merge_firepower_bucket(evidence_queue_vehicles)
    status = "NO_VERIFIED_FIREPOWER"
    if (high_precision.get("estimated_return_pct_per_active_day_at_observed_frequency") or 0.0) >= 10.0:
        status = "VERIFIED_TARGET_10_ROUTE_ESTIMATED"
    elif (high_precision.get("estimated_return_pct_per_active_day_at_observed_frequency") or 0.0) >= 5.0:
        status = "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED"
    elif high_precision.get("unique_vehicle_count"):
        status = "VERIFIED_EDGE_BUT_DAILY_TARGET_SHORTFALL"
    elif evidence_queue.get("unique_vehicle_count"):
        status = "EVIDENCE_QUEUE_ONLY_NO_VERIFIED_FIREPOWER"
    return {
        "contract": (
            "audit-only merged firepower estimate from pair-shard validation evidence; "
            "it does not grant live permission, size orders, or waive gateway gates"
        ),
        "minimum_return_pct": 5.0,
        "target_return_pct": 10.0,
        "status": status,
        "high_precision": high_precision,
        "evidence_queue": evidence_queue,
    }


def _unique_firepower_vehicles(payloads: list[dict[str, Any]], *, bucket: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    for payload in payloads:
        firepower = payload.get("campaign_firepower")
        if not isinstance(firepower, dict):
            continue
        section = firepower.get(bucket)
        if not isinstance(section, dict):
            continue
        for row in section.get("top_vehicles") or []:
            if not isinstance(row, dict):
                continue
            key = _firepower_vehicle_key(row)
            if key in index_by_key:
                rows[index_by_key[key]] = row
                continue
            index_by_key[key] = len(rows)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("estimated_return_pct_per_active_day_at_observed_frequency") or 0.0),
            float(row.get("validation_expectancy_r") or 0.0),
            float(row.get("validation_win_rate") or 0.0),
        ),
        reverse=True,
    )
    return rows


def _firepower_vehicle_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("vehicle_key"):
        return ("vehicle_key", row.get("vehicle_key"), row.get("confluence"))
    return (
        row.get("pair"),
        row.get("side"),
        row.get("firepower_side"),
        row.get("shape"),
        row.get("exit_shape"),
        row.get("confluence"),
    )


def _merge_firepower_bucket(vehicles: list[dict[str, Any]]) -> dict[str, Any]:
    total_daily_return = sum(
        float(row.get("estimated_return_pct_per_active_day_at_observed_frequency") or 0.0)
        for row in vehicles
    )
    total_attempts = sum(float(row.get("observed_attempts_per_active_day") or 0.0) for row in vehicles)
    weighted_return_numerator = sum(
        float(row.get("estimated_return_pct_per_trade_at_risk_lens") or 0.0)
        * max(float(row.get("observed_attempts_per_active_day") or 0.0), 1.0)
        for row in vehicles
    )
    weighted_return_denominator = sum(
        max(float(row.get("observed_attempts_per_active_day") or 0.0), 1.0)
        for row in vehicles
    )
    weighted_return = (
        weighted_return_numerator / weighted_return_denominator
        if weighted_return_denominator > 0.0
        else 0.0
    )
    return {
        "unique_vehicle_count": len(vehicles),
        "pair_count": len({row.get("pair") for row in vehicles if row.get("pair")}),
        "estimated_return_pct_per_active_day_at_observed_frequency": round(total_daily_return, 6),
        "observed_attempts_per_active_day": round(total_attempts, 6),
        "weighted_return_pct_per_trade_at_risk_lens": round(weighted_return, 6),
        "trades_needed_for_minimum_5pct_at_weighted_expectancy": _trades_needed(5.0, weighted_return),
        "trades_needed_for_target_10pct_at_weighted_expectancy": _trades_needed(10.0, weighted_return),
        "active_days_needed_for_minimum_5pct_at_observed_frequency": _days_needed(5.0, total_daily_return),
        "active_days_needed_for_target_10pct_at_observed_frequency": _days_needed(10.0, total_daily_return),
        "top_vehicles": vehicles,
    }


def _trades_needed(target_return_pct: float, return_per_trade_pct: float) -> int | None:
    if return_per_trade_pct <= 0.0:
        return None
    return int(math.ceil(target_return_pct / return_per_trade_pct))


def _days_needed(target_return_pct: float, return_per_active_day_pct: float) -> float | None:
    if return_per_active_day_pct <= 0.0:
        return None
    return round(target_return_pct / return_per_active_day_pct, 6)


def preserve_existing_rule_rows(
    packaged: dict[str, Any],
    existing: dict[str, Any],
    *,
    report_is_narrower: bool | None = None,
) -> None:
    """Avoid shrinking packaged forecast rules when the source is a top-N excerpt.

    A focused mining run may contain new rows for a narrow pair subset while the
    existing packaged artifact still covers a broader qualified universe. Keep
    the broader rows and merge in the focused rows instead of choosing between
    "new but narrow" and "broad but stale".
    """

    summary = packaged.get("summary")
    if not isinstance(summary, dict):
        return
    narrower_report = (
        _packaged_report_is_narrower(packaged, existing)
        if report_is_narrower is None
        else report_is_narrower
    )
    preservation_metadata = {
        "preserved_from_source_report": existing.get("source_report"),
        "preserved_from_generated_at_utc": existing.get("generated_at_utc"),
        "preserved_during_packaging_source_report": packaged.get("source_report"),
        "preserved_during_packaging_generated_at_utc": packaged.get("generated_at_utc"),
    }
    for section in RULE_SECTIONS:
        current_rows = packaged.get(section)
        existing_rows = existing.get(section)
        if not isinstance(current_rows, list) or not isinstance(existing_rows, list):
            continue
        if narrower_report or _packaged_section_is_narrower(section, packaged, existing):
            packaged[section] = _merge_rule_rows(
                current_rows,
                existing_rows,
                annotate_existing_only=True,
                preservation_metadata=preservation_metadata,
            )
            _preserve_section_summary_count(section, packaged, existing)
            continue
        if len(current_rows) >= len(existing_rows):
            continue
        available_count = _optional_int(summary.get(_count_key_for_section(section)))
        if available_count is None or available_count < len(existing_rows):
            continue
        packaged[section] = _merge_rule_rows(
            current_rows,
            existing_rows,
            annotate_existing_only=True,
            preservation_metadata=preservation_metadata,
        )
        _preserve_section_summary_count(section, packaged, existing)


def preserve_existing_campaign_firepower(
    packaged: dict[str, Any],
    existing: dict[str, Any],
    *,
    report_is_narrower: bool | None = None,
) -> None:
    """Keep broader campaign-firepower proof when packaging a narrower mining run."""

    current_firepower = packaged.get("campaign_firepower")
    existing_firepower = existing.get("campaign_firepower")
    if not isinstance(current_firepower, dict) or not isinstance(existing_firepower, dict):
        return
    narrower_report = (
        _packaged_report_is_narrower(packaged, existing)
        if report_is_narrower is None
        else report_is_narrower
    )
    if not narrower_report:
        return
    if not _campaign_firepower_bucket_shrank(current_firepower, existing_firepower):
        return
    merged_firepower = _merge_existing_campaign_firepower(current_firepower, existing_firepower)
    packaged["campaign_firepower"] = merged_firepower or existing_firepower
    packaged["campaign_firepower_preserved_from_existing"] = True
    packaged["campaign_firepower_preservation_reason"] = (
        "new mining report has a narrower qualified universe or a smaller firepower "
        "bucket; preserving existing broader audit-only firepower instead of "
        "shrinking runtime evidence"
    )
    source_report = existing.get("campaign_firepower_source_report") or existing.get("source_report")
    if source_report:
        packaged["campaign_firepower_source_report"] = source_report


def preserve_existing_scope_metadata(
    packaged: dict[str, Any],
    existing: dict[str, Any],
    *,
    report_is_narrower: bool | None = None,
) -> None:
    """Keep report-scope metadata from narrowing when rows/firepower are preserved."""

    narrower_report = (
        _packaged_report_is_narrower(packaged, existing)
        if report_is_narrower is None
        else report_is_narrower
    )
    if not narrower_report:
        return
    current_summary = packaged.get("summary") if isinstance(packaged.get("summary"), dict) else {}
    existing_summary = existing.get("summary") if isinstance(existing.get("summary"), dict) else {}
    if existing_summary:
        packaged["narrow_source_summary"] = current_summary
        packaged["summary"] = _merge_scope_summary(current_summary, existing_summary)
    current_config = packaged.get("config") if isinstance(packaged.get("config"), dict) else {}
    existing_config = existing.get("config") if isinstance(existing.get("config"), dict) else {}
    if existing_config:
        packaged["narrow_source_config"] = current_config
        packaged["config"] = _merge_scope_config(current_config, existing_config)
    packaged["scope_metadata_preserved_from_existing"] = True
    packaged["scope_metadata_preservation_reason"] = (
        "new mining report has a narrower qualified universe; preserving broader "
        "summary/config scope so audit-only runtime evidence is not downgraded"
    )
    source_report = existing.get("scope_metadata_source_report") or existing.get("source_report")
    if source_report:
        packaged["scope_metadata_source_report"] = source_report


def _packaged_report_is_narrower(packaged: dict[str, Any], existing: dict[str, Any]) -> bool:
    current_summary = packaged.get("summary") if isinstance(packaged.get("summary"), dict) else {}
    existing_summary = existing.get("summary") if isinstance(existing.get("summary"), dict) else {}
    for key in (
        "high_precision_multi_confluence_count",
        "high_precision_pair_confluence_count",
        "qualified_multi_confluence_count",
        "qualified_pair_confluence_count",
    ):
        current_count = _optional_int(current_summary.get(key))
        existing_count = _optional_int(existing_summary.get(key))
        if current_count is None or existing_count is None:
            continue
        if current_count < existing_count:
            return True
    return False


def _packaged_section_is_narrower(
    section: str,
    packaged: dict[str, Any],
    existing: dict[str, Any],
) -> bool:
    current_summary = packaged.get("summary") if isinstance(packaged.get("summary"), dict) else {}
    existing_summary = existing.get("summary") if isinstance(existing.get("summary"), dict) else {}
    key = _count_key_for_section(section)
    current_count = _optional_int(current_summary.get(key))
    existing_count = _optional_int(existing_summary.get(key))
    if existing_count is not None and (current_count is None or current_count < existing_count):
        return True
    return False


def _preserve_section_summary_count(
    section: str,
    packaged: dict[str, Any],
    existing: dict[str, Any],
) -> None:
    summary = packaged.get("summary")
    existing_summary = existing.get("summary")
    if not isinstance(summary, dict) or not isinstance(existing_summary, dict):
        return
    key = _count_key_for_section(section)
    current_count = _optional_int(summary.get(key))
    existing_count = _optional_int(existing_summary.get(key))
    if existing_count is not None and (current_count is None or existing_count > current_count):
        summary[key] = existing_count


def _firepower_unique_vehicle_count(firepower: dict[str, Any]) -> int | None:
    return _firepower_bucket_unique_vehicle_count(firepower, bucket="high_precision")


def _firepower_bucket_unique_vehicle_count(firepower: dict[str, Any], *, bucket: str) -> int | None:
    section = firepower.get(bucket)
    if not isinstance(section, dict):
        return None
    count = _optional_int(section.get("unique_vehicle_count"))
    if count is not None:
        return count
    top = section.get("top_vehicles")
    return len(top) if isinstance(top, list) else None


def _campaign_firepower_bucket_shrank(
    current_firepower: dict[str, Any],
    existing_firepower: dict[str, Any],
) -> bool:
    for bucket in ("high_precision", "evidence_queue"):
        existing_count = _firepower_bucket_unique_vehicle_count(existing_firepower, bucket=bucket)
        if existing_count is None:
            continue
        current_count = _firepower_bucket_unique_vehicle_count(current_firepower, bucket=bucket)
        if current_count is None or current_count < existing_count:
            return True
    return False


def _merge_existing_campaign_firepower(
    current_firepower: dict[str, Any],
    existing_firepower: dict[str, Any],
) -> dict[str, Any] | None:
    merged = _merge_campaign_firepower(
        [
            {"campaign_firepower": existing_firepower},
            {"campaign_firepower": current_firepower},
        ]
    )
    if not any(
        (merged.get(bucket) or {}).get("top_vehicles")
        for bucket in ("high_precision", "evidence_queue")
    ):
        return None
    for key in ("per_trade_risk_pct_lens",):
        if key in current_firepower:
            merged[key] = current_firepower[key]
        elif key in existing_firepower:
            merged[key] = existing_firepower[key]
    return merged


def _merge_rule_rows(
    current_rows: list[Any],
    existing_rows: list[Any],
    *,
    annotate_existing_only: bool = False,
    preservation_metadata: dict[str, Any] | None = None,
) -> list[Any]:
    merged: list[Any] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    for row in existing_rows:
        key = _rule_row_key(row)
        index_by_key[key] = len(merged)
        merged.append(
            _annotated_preserved_row(row, preservation_metadata)
            if annotate_existing_only
            else row
        )
    for row in current_rows:
        key = _rule_row_key(row)
        if key in index_by_key:
            merged[index_by_key[key]] = row
            continue
        index_by_key[key] = len(merged)
        merged.append(row)
    return merged


def _annotated_preserved_row(
    row: Any,
    preservation_metadata: dict[str, Any] | None,
) -> Any:
    if not isinstance(row, dict):
        return row
    metadata = preservation_metadata or {}
    out = dict(row)
    out["preserved_from_existing_packaged_artifact"] = True
    out["preserved_because_narrow_source"] = True
    for key, value in metadata.items():
        if value is not None:
            out.setdefault(key, value)
    return out


def _rule_row_key(row: Any) -> tuple[Any, ...]:
    if not isinstance(row, dict):
        return ("raw", repr(row))
    key_fields = (
        "pair",
        "side",
        "selected_side",
        "source_side",
        "shape",
        "source_shape",
        "exit_shape",
        "confluence",
        "confluence_size",
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "feature_e",
        "feature_f",
        "feature_g",
        "feature_h",
    )
    return tuple((field, _normalise_key_value(row.get(field))) for field in key_fields)


def _normalise_key_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalise_key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _normalise_key_value(item)) for key, item in value.items()))
    return value


def _merge_scope_summary(current: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key in (
        "history_files",
        "history_pairs",
        "history_files_discovered",
        "history_pairs_discovered",
        "scored_outcomes",
        "inversion_scored_outcomes",
        *COUNT_KEYS,
    ):
        current_count = _optional_int(current.get(key))
        existing_count = _optional_int(existing.get(key))
        if existing_count is None:
            continue
        if current_count is None or existing_count > current_count:
            merged[key] = existing.get(key)
    for key, value in existing.items():
        if key not in merged:
            merged[key] = value
    return merged


def _merge_scope_config(current: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in existing.items():
        if key in {"multi_confluence_sizes", "inversion_selector_confluence_sizes"}:
            merged[key] = _merge_numeric_lists(current.get(key), value)
            continue
        if key not in merged or merged.get(key) in (None, "", []):
            merged[key] = value
            continue
        if current.get(key) != value:
            merged[key] = value
    return merged


def _merge_numeric_lists(current: Any, existing: Any) -> list[Any]:
    values: list[Any] = []
    for source in (existing, current):
        if not isinstance(source, list):
            continue
        for item in source:
            if item not in values:
                values.append(item)
    try:
        return sorted(values)
    except TypeError:
        return values


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "history_files",
        "history_pairs",
        "history_files_discovered",
        "history_pairs_discovered",
        "history_pairs_discovered_order",
        "history_pair_selection",
        "scored_outcomes",
        "inversion_scored_outcomes",
    ):
        if key in payload:
            summary[key] = payload[key]
    for key in COUNT_KEYS:
        if key in payload:
            summary[key] = payload[key]
            continue
        section = _section_for_count_key(key)
        summary[key] = len(payload.get(section) or [])
    return summary


def _section_for_count_key(key: str) -> str:
    section = key.removesuffix("_count")
    section = section.replace("selector", "selectors")
    section = section.replace("confluence", "confluences")
    return section


def _count_key_for_section(section: str) -> str:
    return f"{section.removesuffix('s')}_count"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _package_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        field: row[field]
        for field in ROW_FIELDS
        if field in row and row[field] is not None
    }


if __name__ == "__main__":
    raise SystemExit(main())
