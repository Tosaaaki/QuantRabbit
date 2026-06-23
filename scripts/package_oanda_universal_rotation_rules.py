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
    payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    packaged = package_payload(payload, source_report=args.source_report)
    preserve_path = args.preserve_from or args.output
    if preserve_path.exists():
        try:
            existing = json.loads(preserve_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            existing = {}
        if isinstance(existing, dict):
            preserve_existing_rule_rows(packaged, existing)
            preserve_existing_campaign_firepower(packaged, existing)
            preserve_existing_scope_metadata(packaged, existing)
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
        "--preserve-from",
        type=Path,
        help="existing packaged rules to preserve broader coverage from; defaults to --output",
    )
    return parser.parse_args()


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


def preserve_existing_rule_rows(packaged: dict[str, Any], existing: dict[str, Any]) -> None:
    """Avoid shrinking packaged forecast rules when the source is a top-N excerpt.

    A focused mining run may contain new rows for a narrow pair subset while the
    existing packaged artifact still covers a broader qualified universe. Keep
    the broader rows and merge in the focused rows instead of choosing between
    "new but narrow" and "broad but stale".
    """

    summary = packaged.get("summary")
    if not isinstance(summary, dict):
        return
    narrower_report = _packaged_report_is_narrower(packaged, existing)
    for section in RULE_SECTIONS:
        current_rows = packaged.get(section)
        existing_rows = existing.get(section)
        if not isinstance(current_rows, list) or not isinstance(existing_rows, list):
            continue
        if narrower_report:
            packaged[section] = _merge_rule_rows(current_rows, existing_rows)
            continue
        if len(current_rows) >= len(existing_rows):
            continue
        available_count = _optional_int(summary.get(_count_key_for_section(section)))
        if available_count is None or available_count < len(existing_rows):
            continue
        packaged[section] = _merge_rule_rows(current_rows, existing_rows)


def preserve_existing_campaign_firepower(packaged: dict[str, Any], existing: dict[str, Any]) -> None:
    """Keep broader campaign-firepower proof when packaging a narrower mining run."""

    current_firepower = packaged.get("campaign_firepower")
    existing_firepower = existing.get("campaign_firepower")
    if not isinstance(current_firepower, dict) or not isinstance(existing_firepower, dict):
        return
    if not _packaged_report_is_narrower(packaged, existing):
        return
    current_vehicles = _firepower_unique_vehicle_count(current_firepower)
    existing_vehicles = _firepower_unique_vehicle_count(existing_firepower)
    if existing_vehicles is None:
        return
    if current_vehicles is not None and current_vehicles >= existing_vehicles:
        return
    packaged["campaign_firepower"] = existing_firepower
    packaged["campaign_firepower_preserved_from_existing"] = True
    packaged["campaign_firepower_preservation_reason"] = (
        "new mining report has a narrower qualified universe; preserving existing "
        "broader audit-only firepower instead of shrinking runtime evidence"
    )
    if existing.get("source_report"):
        packaged["campaign_firepower_source_report"] = existing.get("source_report")


def preserve_existing_scope_metadata(packaged: dict[str, Any], existing: dict[str, Any]) -> None:
    """Keep report-scope metadata from narrowing when rows/firepower are preserved."""

    if not _packaged_report_is_narrower(packaged, existing):
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
    if existing.get("source_report"):
        packaged["scope_metadata_source_report"] = existing.get("source_report")


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


def _firepower_unique_vehicle_count(firepower: dict[str, Any]) -> int | None:
    high_precision = firepower.get("high_precision")
    if not isinstance(high_precision, dict):
        return None
    count = _optional_int(high_precision.get("unique_vehicle_count"))
    if count is not None:
        return count
    top = high_precision.get("top_vehicles")
    return len(top) if isinstance(top, list) else None


def _merge_rule_rows(current_rows: list[Any], existing_rows: list[Any]) -> list[Any]:
    merged: list[Any] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    for row in existing_rows:
        key = _rule_row_key(row)
        index_by_key[key] = len(merged)
        merged.append(row)
    for row in current_rows:
        key = _rule_row_key(row)
        if key in index_by_key:
            merged[index_by_key[key]] = row
            continue
        index_by_key[key] = len(merged)
        merged.append(row)
    return merged


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
    for key in ("history_pairs", "scored_outcomes", *COUNT_KEYS):
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
    for key in ("history_pairs", "scored_outcomes"):
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
