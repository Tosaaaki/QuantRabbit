#!/usr/bin/env python3
"""Run the historical-only causal multi-timeframe S5 research grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit import causal_multitf_s5_grid as grid_core  # noqa: E402
from quant_rabbit.fast_bot_historical_s5 import (  # noqa: E402
    build_historical_s5_manifest,
    load_historical_s5_slice,
)


REPORT_CONTRACT = "QR_CAUSAL_MULTITF_S5_GRID_CLI_REPORT_V1"
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_SAFETY_BOUNDARY: dict[str, Any] = {
    "historical_only": True,
    "diagnostic_only": True,
    "shadow_only": True,
    "order_authority": "NONE",
    "live_permission": False,
    "live_order_enabled": False,
    "promotion_allowed": False,
    "automatic_promotion_allowed": False,
    "broker_mutation_allowed": False,
}


class CausalGridCliError(ValueError):
    """Raised when the research CLI cannot preserve its causal boundary."""


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be ISO-8601 UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must be UTC-aware")
    if parsed.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("timestamp offset must be UTC")
    return parsed.astimezone(timezone.utc)


def _parse_pairs(value: str) -> tuple[str, ...]:
    raw = str(value).split(",")
    if not raw or any(not item.strip() for item in raw):
        raise argparse.ArgumentTypeError("--pairs requires a non-empty CSV")
    pairs = tuple(item.strip() for item in raw)
    if any(_PAIR_RE.fullmatch(pair) is None for pair in pairs):
        raise argparse.ArgumentTypeError(
            "pair names must use explicit uppercase AAA_BBB form"
        )
    if len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError("--pairs must not contain duplicates")
    return pairs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        type=_parse_pairs,
        required=True,
        help="Explicit comma-separated research universe; no implicit default.",
    )
    parser.add_argument("--train-from", type=_parse_utc, required=True)
    parser.add_argument("--train-to", type=_parse_utc, required=True)
    parser.add_argument("--validation-from", type=_parse_utc, required=True)
    parser.add_argument("--validation-to", type=_parse_utc, required=True)
    parser.add_argument("--holdout-from", type=_parse_utc, required=True)
    parser.add_argument("--holdout-to", type=_parse_utc, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _build_splits(args: argparse.Namespace) -> tuple[Any, ...]:
    boundaries = (
        args.train_from,
        args.train_to,
        args.validation_from,
        args.validation_to,
        args.holdout_from,
        args.holdout_to,
    )
    if not (
        boundaries[0]
        < boundaries[1]
        <= boundaries[2]
        < boundaries[3]
        <= boundaries[4]
        < boundaries[5]
    ):
        raise CausalGridCliError(
            "split clocks must be positive, ordered, and non-overlapping"
        )
    return (
        grid_core.UtcSplit(
            name="TRAIN",
            from_utc=args.train_from,
            to_utc=args.train_to,
        ),
        grid_core.UtcSplit(
            name="VALIDATION",
            from_utc=args.validation_from,
            to_utc=args.validation_to,
        ),
        grid_core.UtcSplit(
            name="HOLDOUT",
            from_utc=args.holdout_from,
            to_utc=args.holdout_to,
        ),
    )


def _split_receipt(split: object) -> dict[str, str]:
    return {
        "name": str(getattr(split, "name")),
        "from_utc": _iso_utc(getattr(split, "from_utc")),
        "to_utc": _iso_utc(getattr(split, "to_utc")),
    }


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CausalGridCliError("internal split clock lost UTC awareness")
    if value.utcoffset().total_seconds() != 0:
        raise CausalGridCliError("internal split clock is not UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_common_coverage(
    manifest: Mapping[str, Any],
    *,
    available_pairs: Sequence[str],
    requested_from: datetime,
    requested_to: datetime,
) -> None:
    if not available_pairs:
        return
    common_from = _manifest_utc(manifest.get("common_declared_from_utc"))
    common_to = _manifest_utc(manifest.get("common_declared_to_utc"))
    if common_from is None or common_to is None:
        raise CausalGridCliError("available pairs have no common declared interval")
    if requested_from < common_from or requested_to > common_to:
        raise CausalGridCliError(
            "requested train-to-holdout interval exceeds common manifest coverage"
        )


def _manifest_utc(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return _parse_utc(str(value))
    except argparse.ArgumentTypeError as error:
        raise CausalGridCliError("manifest contains an invalid UTC clock") from error


def _manifest_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest.get("contract"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "selection_policy": manifest.get("selection_policy"),
        "selection_is_outcome_blind": manifest.get("selection_is_outcome_blind"),
        "common_declared_from_utc": manifest.get("common_declared_from_utc"),
        "common_declared_to_utc": manifest.get("common_declared_to_utc"),
        "missing_pairs": list(manifest.get("missing_pairs") or ()),
    }


def _source_receipt(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair": source.get("pair"),
        "relative_path": source.get("relative_path"),
        "source_sha256": source.get("source_sha256"),
        "file_sha256": source.get("file_sha256"),
        "source_summary_sha256": source.get("source_summary_sha256"),
        "acquisition_receipt_sha256": source.get("acquisition_receipt_sha256"),
        "acquisition_receipt_proved": source.get("acquisition_receipt_proved"),
    }


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CausalGridCliError("report contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _validate_no_authority_claim(value: object) -> None:
    """Reject a nested core result that contradicts the CLI safety boundary."""

    false_only = {
        "live_permission",
        "live_permission_granted",
        "live_order_enabled",
        "promotion_allowed",
        "automatic_promotion_allowed",
        "broker_mutation_allowed",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in false_only and item is not False:
                raise CausalGridCliError(
                    f"core result contradicts safety boundary at {key_text}"
                )
            if key_text == "order_authority" and item != "NONE":
                raise CausalGridCliError(
                    "core result contradicts safety boundary at order_authority"
                )
            if key_text == "live_side_effects" and item not in ([], (), None):
                raise CausalGridCliError("core result declares live side effects")
            _validate_no_authority_claim(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_no_authority_claim(item)


def _validate_holdout_selection_binding(global_result: Mapping[str, Any]) -> None:
    """Prove that holdout scoring used only the sealed validation winners."""

    selected = global_result.get("selected_arm_ids")
    validation = global_result.get("validation_winner_arm_ids")
    holdout = global_result.get("holdout_evaluated_arm_ids")
    if not all(isinstance(item, list) for item in (selected, validation, holdout)):
        raise CausalGridCliError(
            "core global result is missing sealed holdout selection lists"
        )
    if len(set(validation)) != len(validation):
        raise CausalGridCliError("validation winner list contains duplicates")
    if selected != validation or holdout != validation:
        raise CausalGridCliError(
            "holdout arms differ from the sealed validation winners"
        )
    if global_result.get("holdout_selection_unchanged") is not True:
        raise CausalGridCliError("core did not prove holdout selection invariance")


def _redact_pair_holdout_for_report(
    pair_result: Mapping[str, Any],
    *,
    global_winners: Sequence[str],
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    """Expose holdout evidence only for globally sealed validation winners."""

    winner_set = set(global_winners)
    unavailable = pair_result.get("status") == "UNAVAILABLE"
    visible_metrics: list[dict[str, Any]] = []
    for candidate in pair_result.get("candidate_metrics") or ():
        candidate_id = str(candidate.get("candidate_id") or "")
        split_metrics = candidate.get("metrics_by_split") or {}
        visible_splits = {
            str(name): dict(metric)
            for name, metric in split_metrics.items()
            if str(name).upper() != "HOLDOUT"
            or (candidate_id in winner_set and not unavailable)
        }
        visible_metrics.append({**candidate, "metrics_by_split": visible_splits})
    visible_daily = [
        dict(row)
        for row in pair_result.get("daily_aggregates") or ()
        if str(row.get("split") or "").upper() != "HOLDOUT"
        or (str(row.get("candidate_id") or "") in winner_set and not unavailable)
    ]
    visible_trades = [
        dict(row)
        for row in pair_result.get("trade_rows") or ()
        if str(row.get("split") or "").upper() != "HOLDOUT"
        or (str(row.get("candidate_id") or "") in winner_set and not unavailable)
    ]
    visible_reasons: dict[str, int] = {}
    raw_signal_count = accepted_signal_count = deoverlap_count = 0
    embargoed_signal_count = 0
    for candidate in visible_metrics:
        for metric in candidate["metrics_by_split"].values():
            raw_signal_count += int(metric.get("raw_signal_count", 0))
            accepted_signal_count += int(metric.get("signal_count", 0))
            deoverlap_count += int(metric.get("deoverlap_count", 0))
            embargoed_signal_count += int(metric.get("embargoed_signal_count", 0))
            for reason, count in (metric.get("reason_counts") or {}).items():
                visible_reasons[str(reason)] = visible_reasons.get(
                    str(reason), 0
                ) + int(count)
    safe_aggregation = dict(pair_result.get("aggregation") or {})
    for key in ("trade_row_count", "trade_rows_returned", "trade_rows_omitted"):
        safe_aggregation.pop(key, None)
    return {
        **pair_result,
        "candidate_metrics": visible_metrics,
        "daily_aggregates": visible_daily,
        "trade_rows": visible_trades,
        "signal_rows": [],
        "raw_signal_count": raw_signal_count,
        "accepted_signal_count": accepted_signal_count,
        "deoverlap_count": deoverlap_count,
        "embargoed_signal_count": embargoed_signal_count,
        "reason_counts": dict(sorted(visible_reasons.items())),
        "aggregation": safe_aggregation,
        "holdout": {
            "split": "HOLDOUT",
            "evaluated_arm_ids": [] if unavailable else list(global_winners),
            "intended_global_arm_ids": list(global_winners) if unavailable else [],
            "reselection_performed": False,
            "selection_unchanged": True,
            "selection_receipt_sha256": selection_receipt_sha256,
            "selection_scope": "GLOBAL_VALIDATION_ONLY",
        },
        "report_visibility": "NONWINNER_HOLDOUT_REDACTED",
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _reject_non_finite(value)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise CausalGridCliError("output path must not be a symlink")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    splits = _build_splits(args)
    requested_pairs = tuple(args.pairs)
    manifest = build_historical_s5_manifest(
        args.history_root,
        pairs=requested_pairs,
    )
    unavailable_pairs = tuple(str(pair) for pair in manifest["missing_pairs"])
    unavailable_set = set(unavailable_pairs)
    available_pairs = tuple(
        pair for pair in requested_pairs if pair not in unavailable_set
    )
    _validate_common_coverage(
        manifest,
        available_pairs=available_pairs,
        requested_from=args.train_from,
        requested_to=args.holdout_to,
    )
    source_by_pair = {
        str(row["pair"]): row
        for row in manifest["selected_sources"]
        if isinstance(row, Mapping)
    }

    pair_runs: list[dict[str, Any]] = []
    pair_results: list[Mapping[str, Any]] = []
    for pair in requested_pairs:
        if pair in unavailable_set:
            pair_result = grid_core.run_causal_multitf_s5_grid(
                pair,
                (),
                splits,
                unavailable_pairs=unavailable_pairs,
            )
            if (
                not isinstance(pair_result, Mapping)
                or pair_result.get("status") != "UNAVAILABLE"
            ):
                raise CausalGridCliError(
                    f"core did not preserve unavailable source state for {pair}"
                )
            _validate_no_authority_claim(pair_result)
            pair_results.append(pair_result)
            pair_runs.append(
                {
                    "pair": pair,
                    "source_status": "UNAVAILABLE",
                    "source_receipt": None,
                    "slice_receipt": None,
                    "result": dict(pair_result),
                }
            )
            continue
        source = source_by_pair.get(pair)
        if source is None:
            raise CausalGridCliError(
                f"manifest omitted available source receipt for {pair}"
            )
        loaded = load_historical_s5_slice(
            manifest,
            pair=pair,
            time_from=args.train_from,
            time_to=args.holdout_to,
        )
        slice_receipt = loaded.receipt()
        pair_result = grid_core.run_causal_multitf_s5_grid(
            pair,
            loaded.candles,
            splits,
            unavailable_pairs=unavailable_pairs,
        )
        if not isinstance(pair_result, Mapping):
            raise CausalGridCliError("core pair result must be a mapping")
        _validate_no_authority_claim(pair_result)
        pair_results.append(pair_result)
        pair_runs.append(
            {
                "pair": pair,
                "source_status": "ADMITTED",
                "source_receipt": _source_receipt(source),
                "slice_receipt": slice_receipt,
                "result": dict(pair_result),
            }
        )

    global_result = grid_core.combine_causal_multitf_s5_grid_runs(
        pair_results,
        splits,
    )
    if not isinstance(global_result, Mapping):
        raise CausalGridCliError("core global result must be a mapping")
    _validate_no_authority_claim(global_result)
    _validate_holdout_selection_binding(global_result)
    global_winners = list(global_result["validation_winner_arm_ids"])
    selection_receipt_sha256 = str(global_result.get("selection_receipt_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", selection_receipt_sha256) is None:
        raise CausalGridCliError("core selection receipt SHA is invalid")
    for pair_run in pair_runs:
        pair_run["result"] = _redact_pair_holdout_for_report(
            pair_run["result"],
            global_winners=global_winners,
            selection_receipt_sha256=selection_receipt_sha256,
        )
    split_receipts = [_split_receipt(split) for split in splits]
    scope_body = {
        "requested_pairs": list(requested_pairs),
        "split_receipts": split_receipts,
        "implicit_default_pair_universe_used": False,
        "scope_may_change_after_outcomes": False,
    }
    body: dict[str, Any] = {
        "contract": REPORT_CONTRACT,
        "schema_version": 1,
        "status": "COMPLETED" if available_pairs else "NO_AVAILABLE_PAIR_SOURCE",
        "requested_pairs": list(requested_pairs),
        "available_pairs": list(available_pairs),
        "unavailable_pairs": list(unavailable_pairs),
        "split_receipts": split_receipts,
        "research_scope_receipt": {
            **scope_body,
            "scope_sha256": _canonical_sha(scope_body),
        },
        "manifest_receipt": _manifest_receipt(manifest),
        "pair_runs": pair_runs,
        "global_result": dict(global_result),
        "selection_contract": {
            "selection_source": "VALIDATION_ONLY",
            "holdout_uses_sealed_validation_selection": True,
            "holdout_reselection_allowed": False,
            "scope_may_change_after_outcomes": False,
        },
        **_SAFETY_BOUNDARY,
    }
    report = {**body, "report_sha256": _canonical_sha(body)}
    _atomic_write_json(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError) as error:
        print(
            f"causal multi-TF S5 grid failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    global_result = report["global_result"]
    summary = {
        "status": report["status"],
        "requested_pair_count": len(report["requested_pairs"]),
        "evaluated_pair_count": len(report["available_pairs"]),
        "requested_pair_run_count": len(report["pair_runs"]),
        "unavailable_pairs": report["unavailable_pairs"],
        "selected_arm_ids": global_result.get("selected_arm_ids", []),
        "output": str(args.output),
        "report_sha256": report["report_sha256"],
        "order_authority": "NONE",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
