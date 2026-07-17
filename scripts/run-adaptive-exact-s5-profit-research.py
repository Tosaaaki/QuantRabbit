#!/usr/bin/env python3
"""Run the two-stage exact-S5 cross-sectional research workflow.

``train`` reads only the TRAIN interval and writes a content-addressed family
result plus at most one lock.  ``validate`` refuses to run without that lock
and evaluates only its one frozen spec.  The July 10 holdout boundary is never
read by this tool.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.adaptive_exact_s5_profit_engine import (
    LOCK_CONTRACT,
    PROSPECTIVE_FINAL_FROM_UTC,
    PROSPECTIVE_FINAL_TO_UTC,
    build_prospective_final_test_lock,
    evaluate_locked_spec,
    prepare_exact_s5_series,
    reconcile_prior_anchor_train,
    run_train_research,
)
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


DEFAULT_HOLDOUT_START = datetime(2026, 7, 10, tzinfo=timezone.utc)
FIXED_TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
FIXED_TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
FIXED_VALIDATION_FROM = FIXED_TRAIN_TO
FIXED_VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
INDEPENDENT_PRIOR_ANCHOR_LEDGER_SHA256 = (
    "7326f2bc7505325623f1d991416ed2b515b7b15fc8c0144ea38c5bd5417854cf"
)
MAX_LOOKBACK = timedelta(minutes=720)
WARMUP_SLACK = timedelta(minutes=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _common_arguments(train)
    train.add_argument("--train-from", required=True)
    train.add_argument("--train-to", required=True)
    train.add_argument("--research-output", type=Path, required=True)
    train.add_argument("--reconciliation-output", type=Path, required=True)
    train.add_argument("--lock-output", type=Path, required=True)
    train.add_argument("--prospective-final-lock-output", type=Path, required=True)
    train.add_argument("--cost-stress-pips", type=float, default=0.5)

    reconcile = subparsers.add_parser("reconcile")
    _common_arguments(reconcile)
    reconcile.add_argument("--train-from", required=True)
    reconcile.add_argument("--train-to", required=True)
    reconcile.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    _common_arguments(validate)
    validate.add_argument("--research", type=Path, required=True)
    validate.add_argument("--lock", type=Path, required=True)
    validate.add_argument("--opened-from", required=True)
    validate.add_argument("--opened-to", required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument(
        "--related-approximation-was-previously-inspected",
        action="store_true",
        required=True,
    )

    args = parser.parse_args()
    manifest = _load_object(args.manifest)
    _validate_manifest_scope(manifest)
    holdout_start = _parse_utc(args.holdout_start)
    if holdout_start != DEFAULT_HOLDOUT_START:
        raise ValueError("this research run must preserve the fixed holdout boundary")

    if args.command in {"train", "reconcile"}:
        train_from = _parse_utc(args.train_from)
        train_to = _parse_utc(args.train_to)
        if (train_from, train_to) != (FIXED_TRAIN_FROM, FIXED_TRAIN_TO):
            raise ValueError("TRAIN boundaries must match the frozen research split")
        _validate_read_boundary(
            from_utc=train_from,
            to_utc=train_to,
            holdout_start=holdout_start,
        )
        if args.command == "train":
            _require_absent_outputs(
                args.research_output,
                args.reconciliation_output,
                args.lock_output,
                args.prospective_final_lock_output,
            )
        else:
            _require_absent_outputs(args.output)
        series, receipt_sha = _load_exact_series(
            manifest,
            from_utc=train_from - MAX_LOOKBACK - WARMUP_SLACK,
            to_utc=train_to,
            holdout_start=holdout_start,
        )
        if args.command == "reconcile":
            result = reconcile_prior_anchor_train(
                series,
                train_from_utc=train_from,
                train_to_utc=train_to,
                source_manifest_sha256=str(manifest["manifest_sha256"]),
                slice_receipts_sha256=receipt_sha,
            )
            _atomic_json(args.output, result)
            print(
                json.dumps(
                    {
                        "status": "TRAIN_ANCHOR_RECONCILED",
                        "reconciliation_sha256": result["reconciliation_sha256"],
                        "variants": [
                            {
                                "label": row["label"],
                                "trade_count": row["trade_count"],
                                "net_pips": row["net_pips"],
                            }
                            for row in result["variants"]
                        ],
                        "jul10_17_strategy_evaluated": False,
                        "order_authority": "NONE",
                    },
                    sort_keys=True,
                )
            )
            return 0
        reconciliation = reconcile_prior_anchor_train(
            series,
            train_from_utc=train_from,
            train_to_utc=train_to,
            source_manifest_sha256=str(manifest["manifest_sha256"]),
            slice_receipts_sha256=receipt_sha,
        )
        _require_independent_anchor_agreement(reconciliation)
        research, lock = run_train_research(
            series,
            train_from_utc=train_from,
            train_to_utc=train_to,
            source_manifest_sha256=str(manifest["manifest_sha256"]),
            slice_receipts_sha256=receipt_sha,
            stress_pips_per_trade=args.cost_stress_pips,
        )
        _atomic_json(args.reconciliation_output, reconciliation)
        _atomic_json(args.research_output, research)
        if lock is not None:
            _atomic_json(args.lock_output, lock)
            prospective_lock = build_prospective_final_test_lock(
                lock=lock,
                source_manifest_sha256=str(manifest["manifest_sha256"]),
            )
            _atomic_json(args.prospective_final_lock_output, prospective_lock)
        print(
            json.dumps(
                {
                    "status": "TRAIN_COMPLETE",
                    "research_sha256": research["research_sha256"],
                    "candidate_count": research["candidate_count"],
                    "eligible_candidate_count": research["eligible_candidate_count"],
                    "locked_survivor_spec_id": research["locked_survivor_spec_id"],
                    "lock_written": lock is not None,
                    "jul10_17_strategy_evaluated": False,
                    "jul10_17_byte_unseen_claimed": False,
                    "prospective_final_test_window": [
                        PROSPECTIVE_FINAL_FROM_UTC.isoformat(),
                        PROSPECTIVE_FINAL_TO_UTC.isoformat(),
                    ],
                    "prospective_final_test_state": "UNAVAILABLE_UNOPENED",
                    "order_authority": "NONE",
                },
                sort_keys=True,
            )
        )
        return 0

    lock = _load_object(args.lock)
    research = _load_object(args.research)
    _require_absent_outputs(args.output)
    if lock.get("contract") != LOCK_CONTRACT:
        raise ValueError("validation requires the exact TRAIN lock contract")
    opened_from = _parse_utc(args.opened_from)
    opened_to = _parse_utc(args.opened_to)
    if (opened_from, opened_to) != (FIXED_VALIDATION_FROM, FIXED_VALIDATION_TO):
        raise ValueError("only the frozen VALIDATION interval is accepted")
    _validate_read_boundary(
        from_utc=opened_from,
        to_utc=opened_to,
        holdout_start=holdout_start,
    )
    series, receipt_sha = _load_exact_series(
        manifest,
        from_utc=opened_from - MAX_LOOKBACK - WARMUP_SLACK,
        to_utc=opened_to,
        holdout_start=holdout_start,
    )
    evaluation = evaluate_locked_spec(
        series,
        lock=lock,
        research=research,
        opened_from_utc=opened_from,
        opened_to_utc=opened_to,
        source_manifest_sha256=str(manifest["manifest_sha256"]),
        slice_receipts_sha256=receipt_sha,
        related_approximation_was_previously_inspected=(
            args.related_approximation_was_previously_inspected
        ),
    )
    _atomic_json(args.output, evaluation)
    print(
        json.dumps(
            {
                "status": "LOCKED_REPLICATION_COMPLETE",
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "metrics": evaluation["metrics"],
                "independent_validation_claim_allowed": evaluation[
                    "independent_validation_claim_allowed"
                ],
                "jul10_17_strategy_evaluated": False,
                "jul10_17_byte_unseen_claimed": False,
                "order_authority": "NONE",
            },
            sort_keys=True,
        )
    )
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--holdout-start",
        default=DEFAULT_HOLDOUT_START.isoformat(),
    )


def _load_exact_series(
    manifest: Mapping[str, Any],
    *,
    from_utc: datetime,
    to_utc: datetime,
    holdout_start: datetime,
) -> tuple[dict[str, tuple[Any, ...]], str]:
    _validate_read_boundary(
        from_utc=from_utc,
        to_utc=to_utc,
        holdout_start=holdout_start,
    )
    series: dict[str, tuple[Any, ...]] = {}
    receipts: list[dict[str, Any]] = []
    for pair in DEFAULT_TRADER_PAIRS:
        item = load_historical_s5_slice(
            manifest,
            pair=pair,
            time_from=from_utc,
            time_to=to_utc,
        )
        series[pair] = prepare_exact_s5_series(item.candles)
        receipts.append(item.receipt())
        del item
        gc.collect()
    if tuple(series) != DEFAULT_TRADER_PAIRS:
        raise AssertionError("exact G8 pair order drifted")
    return series, _canonical_sha(receipts)


def _validate_manifest_scope(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    if manifest.get("expected_pairs") != list(DEFAULT_TRADER_PAIRS):
        raise ValueError("manifest does not bind the exact configured 28 pairs")
    if (
        manifest.get("selected_pair_count") != len(DEFAULT_TRADER_PAIRS)
        or manifest.get("expected_pair_count") != len(DEFAULT_TRADER_PAIRS)
        or manifest.get("complete_pair_coverage") is not True
        or manifest.get("missing_pairs") != []
    ):
        raise ValueError("manifest exact-28 coverage is incomplete")
    if (
        manifest.get("historical_only") is not True
        or manifest.get("shadow_only") is not True
        or manifest.get("live_permission") is not False
        or manifest.get("broker_mutation_allowed") is not False
        or manifest.get("order_authority") != "NONE"
    ):
        raise ValueError("manifest authority boundary is invalid")


def _validate_read_boundary(
    *, from_utc: datetime, to_utc: datetime, holdout_start: datetime
) -> None:
    if to_utc <= from_utc:
        raise ValueError("read interval must be positive")
    if to_utc > holdout_start:
        raise ValueError("HOLDOUT access is forbidden")


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.second or parsed.microsecond:
        raise ValueError("research boundaries must be minute aligned")
    return parsed


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _require_absent_outputs(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(
            "research outputs must be clean before computation; refusing stale reuse: "
            + ", ".join(existing)
        )


def _require_independent_anchor_agreement(value: Mapping[str, Any]) -> None:
    variants = value.get("variants")
    if not isinstance(variants, list):
        raise ValueError("anchor reconciliation variants are missing")
    baseline = next(
        (
            row
            for row in variants
            if isinstance(row, Mapping) and row.get("label") == "PRIOR_EXACT_BASELINE"
        ),
        None,
    )
    if (
        baseline is None
        or baseline.get("trade_count") != 524
        or not _close_number(baseline.get("net_pips"), -747.6, tolerance=1e-9)
        or not _close_number(
            baseline.get("profit_factor"), 0.8821117699, tolerance=1e-9
        )
        or baseline.get("compatibility_jsonl_sha256")
        != INDEPENDENT_PRIOR_ANCHOR_LEDGER_SHA256
    ):
        raise ValueError("adaptive anchor does not match the independent calculator")


def _close_number(value: Any, expected: float, *, tolerance: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(float(value) - expected) <= tolerance
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
