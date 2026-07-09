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
    "blockers",
    "adoption_level",
    "candidate_rule_validation_blocked",
    "global_currency_validation_blocked",
    "raw_directional_rows",
    "deduped_directional_rows",
    "evaluated_rows",
    "unscorable_no_market_rows",
    "pending_future_truth_rows",
    "required_min_evaluated_rows",
    "history_files",
    "history_candles",
    "missing_price_truth_samples",
    "missing_price_window_group_count",
    "unscorable_no_market_window_group_count",
    "pending_future_truth_window_group_count",
    "future_price_truth_window_group_count",
    "missing_pairs",
    "missing_pair_directions",
    "all_currency_sample_coverage_status",
    "under_sampled_pair_direction_count",
    "under_sampled_pair_directions",
    "under_sampled_missing_evaluated_samples",
    "history_fetch_command",
    "history_fetch_command_count",
    "history_fetch_command_mode",
    "history_fetch_commands",
    "warnings",
)

COVERAGE_DETAIL_LIMIT = 24

COVERAGE_DETAIL_FIELDS = (
    "pair",
    "direction",
    "forecast_samples",
    "forecast_active_days",
    "evaluated_samples",
    "evaluated_active_days",
    "unscorable_no_market_samples",
    "pending_future_truth_samples",
    "missing_price_truth_samples",
    "missing_evaluated_samples",
    "missing_active_days",
    "coverage_gap_reasons",
)

PAIR_COVERAGE_FIELDS = (
    "pair",
    "forecast_samples",
    "evaluated_samples",
    "missing_price_truth_samples",
    "pending_future_truth_samples",
    "unscorable_no_market_samples",
    "missing_evaluated_samples_to_min_directional",
)


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    packaged = package_payload(
        payload,
        source_report=args.source_report,
        allow_partial=bool(args.allow_partial),
    )
    preserve_path = args.preserve_from or args.output
    if preserve_path.exists():
        existing = json.loads(preserve_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            preserve_existing_rule_rows(packaged, existing)
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
        "source_report": _source_report_label(source_report),
        "history_dirs": _history_dirs(payload.get("history_dirs")),
        "source_pair_filter": _string_list(payload.get("pair_filter")),
        "granularity": payload.get("granularity"),
        "truth_source": payload.get("truth_source"),
        "price_truth_coverage": _copy_fields(truth, TRUTH_FIELDS),
        "forecast_sample_coverage_summary": _forecast_sample_coverage_summary(
            payload.get("forecast_sample_coverage"),
            truth,
        ),
        "selection": copy.deepcopy(precision.get("selection") or {}),
        "adoption_summary": copy.deepcopy(precision.get("adoption_summary") or {}),
    }
    for section in RULE_SECTIONS:
        rows = precision.get(section)
        packaged[section] = copy.deepcopy(rows if isinstance(rows, list) else [])
    return packaged


def preserve_existing_rule_rows(packaged: dict[str, Any], existing: dict[str, Any]) -> None:
    """Merge focused bid/ask replay refreshes without dropping untouched pairs.

    A targeted ``--pairs`` validation run is the right operational unit for the
    active board's next lane, but publishing it must not erase unrelated pair
    blockers/support from the broader runtime artifact. Existing rows for
    refreshed pairs are intentionally not preserved when absent from the new
    report, so a pair-scoped refresh can clear stale blockers for that pair.
    When the same refreshed-pair rule is still present but the focused report
    has fewer samples than the existing packaged row, keep the broader row; a
    refresh must not silently downgrade live evidence coverage.
    """

    refreshed_pairs = {
        str(pair).upper()
        for pair in packaged.get("source_pair_filter") or []
        if str(pair).strip()
    }
    if not refreshed_pairs:
        return
    preservation_metadata = {
        "preserved_from_source_report": existing.get("source_report"),
        "preserved_from_generated_at_utc": existing.get("generated_at_utc"),
        "preserved_during_packaging_source_report": packaged.get("source_report"),
        "preserved_during_packaging_generated_at_utc": packaged.get("generated_at_utc"),
    }
    preserved_count = 0
    for section in RULE_SECTIONS:
        current_rows = packaged.get(section)
        existing_rows = existing.get(section)
        if not isinstance(current_rows, list) or not isinstance(existing_rows, list):
            continue
        merged = _merge_rule_rows_for_refreshed_pairs(
            current_rows,
            existing_rows,
            refreshed_pairs=refreshed_pairs,
            preservation_metadata=preservation_metadata,
        )
        packaged[section] = merged
    preserved_count = _preserved_row_count(packaged)
    if preserved_count <= 0:
        return
    packaged["existing_rule_rows_preserved"] = True
    packaged["existing_rule_rows_preserved_count"] = preserved_count
    packaged["existing_rule_rows_preservation_reason"] = (
        "source report is a pair-filtered bid/ask replay refresh; preserved "
        "existing rule rows for pairs outside source_pair_filter or broader "
        "same-pair rule rows when the focused refresh was narrower"
    )
    packaged["existing_rule_rows_preserved_excluding_pairs"] = sorted(refreshed_pairs)
    _preserve_adoption_summary_counts(packaged, existing)


def _merge_rule_rows_for_refreshed_pairs(
    current_rows: list[Any],
    existing_rows: list[Any],
    *,
    refreshed_pairs: set[str],
    preservation_metadata: dict[str, Any],
) -> list[Any]:
    merged: list[Any] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    existing_by_key: dict[tuple[Any, ...], Any] = {}
    for row in existing_rows:
        key = _rule_row_key(row)
        existing_by_key.setdefault(key, row)
    for row in existing_rows:
        if _row_pair(row) in refreshed_pairs:
            continue
        key = _rule_row_key(row)
        index_by_key[key] = len(merged)
        merged.append(_annotated_preserved_row(row, preservation_metadata))
    for row in current_rows:
        key = _rule_row_key(row)
        existing_row = existing_by_key.get(key)
        if (
            _row_pair(row) in refreshed_pairs
            and _row_pair(existing_row) in refreshed_pairs
            and _existing_rule_row_is_stronger(existing_row, row)
        ):
            row = _annotated_preserved_row(
                existing_row,
                {
                    **preservation_metadata,
                    "preserved_because_pair_filtered_source_was_narrower": True,
                },
            )
        if key in index_by_key:
            merged[index_by_key[key]] = row
            continue
        index_by_key[key] = len(merged)
        merged.append(row)
    return merged


def _preserved_row_count(packaged: dict[str, Any]) -> int:
    count = 0
    for section in RULE_SECTIONS:
        rows = packaged.get(section)
        if not isinstance(rows, list):
            continue
        count += sum(
            1
            for row in rows
            if isinstance(row, dict) and row.get("preserved_from_existing_packaged_artifact")
        )
    return count


def _existing_rule_row_is_stronger(existing_row: Any, current_row: Any) -> bool:
    if not isinstance(existing_row, dict) or not isinstance(current_row, dict):
        return False
    existing_rank = _adoption_strength(existing_row)
    current_rank = _adoption_strength(current_row)
    if existing_rank != current_rank:
        return existing_rank > current_rank
    existing_samples = _optional_int(existing_row.get("samples")) or 0
    current_samples = _optional_int(current_row.get("samples")) or 0
    if existing_samples != current_samples:
        return existing_samples > current_samples
    existing_days = _optional_int(existing_row.get("active_days")) or 0
    current_days = _optional_int(current_row.get("active_days")) or 0
    return existing_days > current_days


def _adoption_strength(row: dict[str, Any]) -> int:
    if row.get("live_grade") is True:
        return 3
    status = str(row.get("adoption_status") or row.get("daily_stability_status") or "").upper()
    if "LIVE_GRADE" in status or "LIVE_BLOCK_NEGATIVE_EXPECTANCY" in status:
        return 3
    if "RANK_ONLY" in status:
        return 2
    if status:
        return 1
    return 0


def _row_pair(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    pair = str(row.get("pair") or "").upper().strip()
    return pair or None


def _rule_row_key(row: Any) -> tuple[Any, ...]:
    if not isinstance(row, dict):
        return ("raw", repr(row))
    name = str(row.get("name") or "").strip()
    if name:
        return ("name", name)
    key_fields = (
        "pair",
        "side",
        "direction",
        "forecast_direction",
        "faded_direction",
        "horizon_bucket",
        "confidence_bucket",
        "granularity",
        "optimized_take_profit_pips",
        "optimized_stop_loss_pips",
    )
    return tuple((field, _normalise_key_value(row.get(field))) for field in key_fields)


def _normalise_key_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalise_key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _normalise_key_value(item)) for key, item in value.items()))
    return value


def _annotated_preserved_row(row: Any, preservation_metadata: dict[str, Any]) -> Any:
    if not isinstance(row, dict):
        return row
    out = copy.deepcopy(row)
    out["preserved_from_existing_packaged_artifact"] = True
    out["preserved_because_pair_filtered_source"] = True
    for key, value in preservation_metadata.items():
        if value is not None:
            out.setdefault(key, value)
    return out


def _preserve_adoption_summary_counts(packaged: dict[str, Any], existing: dict[str, Any]) -> None:
    current = packaged.get("adoption_summary")
    prior = existing.get("adoption_summary")
    if not isinstance(current, dict) or not isinstance(prior, dict):
        return
    for key in (
        "live_grade_support_rules",
        "rank_only_support_rules",
        "negative_block_rules",
    ):
        current_value = _optional_int(current.get(key))
        prior_value = _optional_int(prior.get(key))
        if prior_value is not None and (current_value is None or prior_value > current_value):
            current[key] = prior_value
    _sync_adoption_summary_flags(current)


def _sync_adoption_summary_flags(summary: dict[str, Any]) -> None:
    live_count = _optional_int(summary.get("live_grade_support_rules"))
    if live_count is not None:
        summary["has_live_grade_support"] = live_count > 0
    rank_count = _optional_int(summary.get("rank_only_support_rules"))
    if rank_count is not None:
        summary["has_rank_only_support"] = rank_count > 0


def _source_report_label(source_report: Path) -> str:
    if not source_report.is_absolute():
        return source_report.as_posix()
    parts = source_report.parts
    for index, part in enumerate(parts[:-1]):
        if part == "logs" and parts[index + 1] == "reports":
            return Path(*parts[index:]).as_posix()
    return source_report.as_posix()


def _copy_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: copy.deepcopy(payload[field]) for field in fields if field in payload}


def _history_dirs(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _forecast_sample_coverage_summary(raw: Any, truth: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    under_sampled = raw.get("under_sampled_pair_directions")
    under_sampled_rows = under_sampled if isinstance(under_sampled, list) else []
    pairs = raw.get("pairs")
    pair_rows = pairs if isinstance(pairs, list) else []
    under_sampled_details = [
        _copy_fields(item, COVERAGE_DETAIL_FIELDS)
        for item in under_sampled_rows
        if isinstance(item, dict)
    ]
    pair_coverage = [
        _copy_fields(item, PAIR_COVERAGE_FIELDS)
        for item in pair_rows
        if isinstance(item, dict)
    ]
    pending_future = raw.get("pending_future_truth_samples")
    if pending_future is None and isinstance(truth, dict):
        pending_future = truth.get("pending_future_truth_rows")
    return {
        "min_directional_samples_for_precision_rule": raw.get(
            "min_directional_samples_for_precision_rule"
        ),
        "min_active_days_for_daily_stability": raw.get("min_active_days_for_daily_stability"),
        "pair_count": raw.get("pair_count"),
        "pair_direction_count": raw.get("pair_direction_count"),
        "unscorable_no_market_samples": raw.get("unscorable_no_market_samples"),
        "pending_future_truth_samples": pending_future,
        "under_sampled_pair_directions": len(under_sampled_rows),
        "under_sampled_gap_reason_counts": _gap_reason_counts(under_sampled_rows),
        "under_sampled_pair_direction_detail_count": len(under_sampled_details),
        "under_sampled_pair_direction_examples_omitted": max(
            len(under_sampled_details) - COVERAGE_DETAIL_LIMIT,
            0,
        ),
        "under_sampled_pair_direction_details": under_sampled_details,
        "under_sampled_pair_direction_examples": under_sampled_details[:COVERAGE_DETAIL_LIMIT],
        "pair_coverage_count": len(pair_coverage),
        "pair_coverage_examples_omitted": max(len(pair_coverage) - COVERAGE_DETAIL_LIMIT, 0),
        "pair_coverage": pair_coverage,
        "pair_coverage_examples": pair_coverage[:COVERAGE_DETAIL_LIMIT],
    }


def _gap_reason_counts(rows: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        for reason in item.get("coverage_gap_reasons") or []:
            text = str(reason or "").strip()
            if not text:
                continue
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
