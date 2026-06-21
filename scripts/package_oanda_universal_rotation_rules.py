#!/usr/bin/env python3
"""Package mined OANDA rotation report rows for runtime fallback.

The full mining report lives under ``logs/`` and is intentionally not tracked.
Live runtime therefore needs a compact tracked fallback that preserves only the
rank-only rule sections consumed by ``forecast_precision``.
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
    for section in RULE_SECTIONS:
        packaged[section] = [_package_row(row) for row in payload.get(section) or []]
    return packaged


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
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
