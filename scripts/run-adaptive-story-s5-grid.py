#!/usr/bin/env python3
"""Run one immutable phase of the adaptive exact-S5 story research grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import stat
import statistics
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import TZPATH


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit import adaptive_story_s5_grid as story_core  # noqa: E402
from quant_rabbit.fast_bot_historical_s5 import (  # noqa: E402
    HistoricalS5CacheError,
    build_historical_s5_manifest,
    load_historical_s5_slice,
    load_historical_s5_slices,
)


TRAIN_CONTRACT = "QR_ADAPTIVE_STORY_S5_TRAIN_RECEIPT_V2"
VALIDATION_CONTRACT = "QR_ADAPTIVE_STORY_S5_VALIDATION_RECEIPT_V2"
HOLDOUT_CONTRACT = "QR_ADAPTIVE_STORY_S5_HOLDOUT_RECEIPT_V2"
TRAIN_SELECTION_POLICY = (
    "ONE_EXIT_PER_STORY_ONE_SE_COMPLETE_UTC_DAILY_NET_R_"
    "WEIGHTED_TWO_DAY_CLUSTER_ROBUST_SE_ANCHORED_AT_SPLIT_START_"
    "BEST_MEAN_SE_COMPLEXITY_ID_THEN_SIMPLEST_MEAN_SE_ID_V2"
)
VALIDATION_SCREEN_POLICY = (
    "ALL_TRAIN_FIXED_CANDIDATES_GLOBAL_ECONOMIC_SCREEN_"
    "MIN30_RESOLVED_MIN8_DAYS_MIN4_CONTRIBUTING_PAIRS_"
    "POSITIVE_R_PF_ABOVE_ONE_LOOCV_POSITIVE_NO_UNRESOLVED_"
    "TRAIN_VALIDATION_SAME_POSITIVE_SIGN_V2"
)
PORTFOLIO_ALLOCATION = "EQUAL_INITIAL_R_PER_FILLED_TRADE_UNBOUNDED_GROSS_DIAGNOSTIC_V2"
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PAIR_WORKERS = 6
PAIR_EXECUTION_POLICY = "DETERMINISTIC_PROCESS_POOL_INPUT_ORDER_RECONSTRUCTION_MAX_6_V1"
PRICE_ONLY_COST_SCOPE: dict[str, Any] = {
    "economics_basis": (
        "PRICE_ONLY_EXECUTABLE_BID_ASK_BEFORE_FINANCING_AND_COMMISSION"
    ),
    "entry_and_exit_spread": "EXACT_S5_BID_ASK",
    "resting_entry_fill": (
        "FROZEN_TRIGGER_OR_EXECUTABLE_OPEN_GAP_WITH_NO_ADDITIONAL_SLIPPAGE"
    ),
    "market_and_time_exit_fill": "FIRST_ELIGIBLE_EXECUTABLE_S5_OPEN",
    "additional_latency_model": "NONE",
    "additional_slippage_model": "NONE_BEYOND_EXECUTABLE_OPEN_GAP",
    "exact_scope_limit": "RECORDED_S5_TOP_OF_BOOK_BID_ASK_PRICE_PATH_ONLY",
    "order_book_vwap_modeled": False,
    "latency_modeled": False,
    "financing_modeled": False,
    "commission_modeled": False,
    "explicit_commission": "NOT_MODELED",
    "financing_and_swap": "NOT_MODELED",
    "market_impact": "NOT_MODELED",
    "fully_loaded_net_economics": False,
    "live_net_claim_allowed": False,
}
_DEPENDENCY_RELATIVE_PATHS = (
    "scripts/run-adaptive-story-s5-grid.py",
    "src/quant_rabbit/adaptive_story_s5_grid.py",
    "src/quant_rabbit/causal_multitf_s5_grid.py",
    "src/quant_rabbit/fast_bot_historical_s5.py",
    "src/quant_rabbit/instruments.py",
    "src/quant_rabbit/technical_forecast_forward_outcome.py",
)
_AUTHORITY: dict[str, Any] = {
    "historical_only": True,
    "diagnostic_only": True,
    "shadow_only": True,
    "order_authority": "NONE",
    "forward_proof_eligible": False,
    "live_permission": False,
    "live_order_enabled": False,
    "promotion_allowed": False,
    "automatic_promotion_allowed": False,
    "broker_mutation_allowed": False,
}


class AdaptiveStoryCliError(ValueError):
    """Raised when a phase cannot preserve its immutable research scope."""


class AdaptiveStoryCoreCapabilityError(AdaptiveStoryCliError):
    """Raised when the core cannot execute a strict candidate whitelist."""


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


def _parse_run_ids(value: str) -> tuple[str, ...]:
    raw = str(value).split(",")
    if not raw or any(not item.strip() for item in raw):
        raise argparse.ArgumentTypeError("--history-run-ids requires a non-empty CSV")
    run_ids = tuple(item.strip() for item in raw)
    if any(_RUN_ID_RE.fullmatch(run_id) is None for run_id in run_ids):
        raise argparse.ArgumentTypeError(
            "history run IDs must use exact YYYYMMDDTHHMMSSZ form"
        )
    if len(set(run_ids)) != len(run_ids):
        raise argparse.ArgumentTypeError(
            "--history-run-ids must not contain duplicates"
        )
    for run_id in run_ids:
        try:
            parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "history run ID contains an invalid UTC clock"
            ) from error
        if parsed.strftime("%Y%m%dT%H%M%SZ") != run_id:
            raise argparse.ArgumentTypeError("history run ID is not canonical")
    return tuple(sorted(run_ids))


def _parse_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--workers must be an integer") from error
    if not 1 <= workers <= MAX_PAIR_WORKERS:
        raise argparse.ArgumentTypeError(
            f"--workers must be between 1 and {MAX_PAIR_WORKERS}"
        )
    return workers


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)

    train = phases.add_parser("train", help="freeze the ten validation candidates")
    train.add_argument("--history-root", type=Path, required=True)
    train.add_argument("--history-run-ids", type=_parse_run_ids, required=True)
    train.add_argument("--pairs", type=_parse_pairs, required=True)
    train.add_argument("--train-from", type=_parse_utc, required=True)
    train.add_argument("--train-to", type=_parse_utc, required=True)
    train.add_argument("--validation-from", type=_parse_utc, required=True)
    train.add_argument("--validation-to", type=_parse_utc, required=True)
    train.add_argument("--holdout-from", type=_parse_utc, required=True)
    train.add_argument("--holdout-to", type=_parse_utc, required=True)
    train.add_argument("--workers", type=_parse_workers, default=MAX_PAIR_WORKERS)
    train.add_argument("--output", type=Path, required=True)

    validation = phases.add_parser(
        "validation",
        help="run only the ten candidates sealed by the train receipt",
    )
    validation.add_argument("--history-root", type=Path, required=True)
    validation.add_argument("--train-receipt", type=Path, required=True)
    validation.add_argument("--workers", type=_parse_workers, default=MAX_PAIR_WORKERS)
    validation.add_argument("--output", type=Path, required=True)

    holdout = phases.add_parser(
        "holdout",
        help="run only the validation survivors without reselection",
    )
    holdout.add_argument("--history-root", type=Path, required=True)
    holdout.add_argument("--train-receipt", type=Path, required=True)
    holdout.add_argument("--validation-receipt", type=Path, required=True)
    holdout.add_argument("--workers", type=_parse_workers, default=MAX_PAIR_WORKERS)
    holdout.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AdaptiveStoryCliError("internal split clock lost UTC awareness")
    if value.utcoffset().total_seconds() != 0:
        raise AdaptiveStoryCliError("internal split clock is not UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _scope_from_train_args(args: argparse.Namespace) -> dict[str, Any]:
    clocks = (
        args.train_from,
        args.train_to,
        args.validation_from,
        args.validation_to,
        args.holdout_from,
        args.holdout_to,
    )
    if not (clocks[0] < clocks[1] <= clocks[2] < clocks[3] <= clocks[4] < clocks[5]):
        raise AdaptiveStoryCliError(
            "phase clocks must be positive, ordered, and non-overlapping"
        )
    return {
        "pairs": list(args.pairs),
        "history_root": str(args.history_root.expanduser().resolve(strict=True)),
        "history_run_ids": list(args.history_run_ids),
        "splits": {
            "TRAIN": {"from_utc": _iso_utc(clocks[0]), "to_utc": _iso_utc(clocks[1])},
            "VALIDATION": {
                "from_utc": _iso_utc(clocks[2]),
                "to_utc": _iso_utc(clocks[3]),
            },
            "HOLDOUT": {"from_utc": _iso_utc(clocks[4]), "to_utc": _iso_utc(clocks[5])},
        },
        "implicit_pair_universe_used": False,
        "later_phase_scope_override_allowed": False,
    }


def _split_from_scope(scope: Mapping[str, Any], phase: str) -> Any:
    splits = scope.get("splits")
    if not isinstance(splits, Mapping):
        raise AdaptiveStoryCliError("receipt research scope has no split map")
    row = splits.get(phase)
    if not isinstance(row, Mapping):
        raise AdaptiveStoryCliError(f"receipt research scope has no {phase} split")
    start = _receipt_utc(row.get("from_utc"))
    end = _receipt_utc(row.get("to_utc"))
    if start >= end:
        raise AdaptiveStoryCliError("receipt split interval is invalid")
    return story_core.UtcSplit(name=phase, from_utc=start, to_utc=end)


def _receipt_utc(value: object) -> datetime:
    try:
        return _parse_utc(str(value))
    except argparse.ArgumentTypeError as error:
        raise AdaptiveStoryCliError("receipt contains an invalid UTC clock") from error


def _strict_json_loads(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise AdaptiveStoryCliError(f"{label} contains non-finite JSON: {value}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdaptiveStoryCliError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptiveStoryCliError(f"{label} is not strict JSON") from error


def _read_stable_bytes(path: Path) -> bytes:
    """Read one regular path through one FD and reject path/FD substitution.

    The no-follow open and the before/after ``fstat`` checks bind the bytes to
    one inode.  Comparing that same FD with both path observations also closes
    an A -> B -> A rename around ``open``: a descriptor opened on B cannot be
    accepted merely because the pathname points back to A after the swap.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AdaptiveStoryCliError("receipt path stable open failed") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdaptiveStoryCliError("receipt path must be a regular file")
        try:
            path_before = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise AdaptiveStoryCliError(
                "receipt path changed during stable read"
            ) from error
        if not stat.S_ISREG(path_before.st_mode):
            raise AdaptiveStoryCliError("receipt path must be a regular file")
        if _stable_stat_fingerprint(path_before) != _stable_stat_fingerprint(before):
            raise AdaptiveStoryCliError(
                "receipt path identity differs from opened file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise AdaptiveStoryCliError(
                "receipt path changed during stable read"
            ) from error
        if not stat.S_ISREG(path_after.st_mode):
            raise AdaptiveStoryCliError("receipt path must be a regular file")
        fingerprint = _stable_stat_fingerprint(before)
        if (
            _stable_stat_fingerprint(after) != fingerprint
            or _stable_stat_fingerprint(path_after) != fingerprint
            or len(payload) != before.st_size
        ):
            raise AdaptiveStoryCliError("receipt changed during stable read")
        return payload
    finally:
        os.close(descriptor)


def _stable_stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    """Return mutation-sensitive metadata while excluding read-updated atime."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_file_sha256(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise AdaptiveStoryCliError(f"{label} must be a regular file")
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened = os.fstat(handle.fileno())
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_opened = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_opened or identity_opened != identity_after:
        raise AdaptiveStoryCliError(f"{label} changed during stable read")
    return digest.hexdigest()


def _dst_schedule_receipt() -> dict[str, Any]:
    zone_keys = tuple(str(zone) for zone in story_core.SESSION_TIMEZONES)
    expected = ("Europe/London", "America/New_York")
    if zone_keys != expected:
        raise AdaptiveStoryCliError("adaptive story DST timezone scope changed")
    zone_rows: list[dict[str, str]] = []
    for zone_key in zone_keys:
        candidates = [Path(root) / zone_key for root in TZPATH]
        zone_path = next(
            (
                candidate
                for candidate in candidates
                if not candidate.is_symlink() and candidate.is_file()
            ),
            None,
        )
        if zone_path is None:
            raise AdaptiveStoryCliError(
                f"cannot seal exact zoneinfo bytes for {zone_key}"
            )
        zone_rows.append(
            {
                "zone_key": zone_key,
                "tzif_sha256": _stable_file_sha256(
                    zone_path,
                    label=f"zoneinfo dependency {zone_key}",
                ),
            }
        )
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_DST_SCHEDULE_SEAL_V1",
        "timezone_order": list(zone_keys),
        "zoneinfo_files": zone_rows,
        "local_open_hour": story_core.SESSION_LOCAL_OPEN_HOUR,
        "local_open_window_minutes": story_core.SESSION_OPEN_WINDOW_MINUTES,
        "completed_m1_end_clock_window": "LOCAL_(08:00,08:15]",
    }
    return {**body, "dst_schedule_sha256": _canonical_sha(body)}


def _selection_semantic_receipt() -> dict[str, Any]:
    candidates = [
        {
            "candidate_id": row.candidate_id,
            "hypothesis_id": row.hypothesis_id,
            "story_name": row.story_name,
            "exit_policy_id": row.exit_policy_id,
            "contextual_order_policy": row.contextual_order_policy,
            "allowed_order_modes": list(row.allowed_order_modes),
            "max_hold_seconds": row.max_hold_seconds,
            "profit_target_r": row.profit_target_r,
            "trailing_structural": row.trailing_structural,
            "complexity": row.complexity,
        }
        for row in _catalog_rows()
    ]
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_50_CANDIDATE_SELECTION_SEMANTICS_V1",
        "selectable_candidate_count": 50,
        "candidate_rows": candidates,
        "core_story_catalog_policy": story_core.STORY_CATALOG_POLICY_V2,
        "core_truth_policy": story_core.STORY_TRUTH_POLICY_V2,
        "core_story_catalog_sha256": _canonical_sha(
            story_core._story_catalog_receipt_v2()
        ),
        "core_truth_evaluator_sha256": _canonical_sha(
            story_core._truth_evaluator_receipt_v2()
        ),
        "train_selection_policy": TRAIN_SELECTION_POLICY,
        "train_estimator": "COMPLETE_UTC_DAILY_NET_R_ZERO_DAYS_INCLUDED",
        "train_cluster_formula": ("SQRT(B/(B-1)*SUM((S_C-N_C*MEAN_DAILY_R)^2)/N^2)"),
        "train_cluster_unit": "NON_OVERLAPPING_TWO_UTC_CALENDAR_DAY_BLOCK",
        "validation_screen_policy": VALIDATION_SCREEN_POLICY,
        "portfolio_allocation": PORTFOLIO_ALLOCATION,
        "price_only_cost_scope": dict(PRICE_ONLY_COST_SCOPE),
    }
    return {**body, "selection_semantics_sha256": _canonical_sha(body)}


def _evaluator_dependency_seal() -> dict[str, Any]:
    dependency_rows = []
    for relative_path in _DEPENDENCY_RELATIVE_PATHS:
        dependency_rows.append(
            {
                "relative_path": relative_path,
                "sha256": _stable_file_sha256(
                    ROOT / relative_path,
                    label=f"evaluator dependency {relative_path}",
                ),
            }
        )
    dst_schedule = _dst_schedule_receipt()
    selection_semantics = _selection_semantic_receipt()
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_EVALUATOR_DEPENDENCY_SEAL_V1",
        "schema_version": 1,
        "source_dependencies": dependency_rows,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "hexversion": sys.hexversion,
        },
        "dst_schedule": dst_schedule,
        "dst_schedule_sha256": dst_schedule["dst_schedule_sha256"],
        "selection_semantics": selection_semantics,
        "selection_semantics_sha256": selection_semantics["selection_semantics_sha256"],
    }
    return {**body, "evaluator_dependency_sha256": _canonical_sha(body)}


def _validate_evaluator_dependency_seal(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveStoryCliError("receipt has no evaluator dependency seal")
    seal = dict(value)
    claimed = seal.pop("evaluator_dependency_sha256", None)
    if not isinstance(claimed, str) or _SHA_RE.fullmatch(claimed) is None:
        raise AdaptiveStoryCliError("evaluator dependency digest is invalid")
    if _canonical_sha(seal) != claimed:
        raise AdaptiveStoryCliError("evaluator dependency seal digest mismatch")
    if seal.get("contract") != "QR_ADAPTIVE_STORY_EVALUATOR_DEPENDENCY_SEAL_V1":
        raise AdaptiveStoryCliError("evaluator dependency contract mismatch")
    if seal.get("schema_version") != 1:
        raise AdaptiveStoryCliError("evaluator dependency schema mismatch")
    sources = seal.get("source_dependencies")
    if not isinstance(sources, list) or [
        row.get("relative_path") for row in sources if isinstance(row, Mapping)
    ] != list(_DEPENDENCY_RELATIVE_PATHS):
        raise AdaptiveStoryCliError("evaluator source dependency scope mismatch")
    if any(
        not isinstance(row, Mapping)
        or _SHA_RE.fullmatch(str(row.get("sha256") or "")) is None
        for row in sources
    ):
        raise AdaptiveStoryCliError("evaluator source dependency digest is invalid")
    python = seal.get("python")
    if not isinstance(python, Mapping) or any(
        not python.get(field) for field in ("implementation", "version", "cache_tag")
    ):
        raise AdaptiveStoryCliError("evaluator Python dependency is invalid")
    dst = seal.get("dst_schedule")
    if not isinstance(dst, Mapping):
        raise AdaptiveStoryCliError("evaluator DST schedule seal is missing")
    dst_body = dict(dst)
    dst_claimed = dst_body.pop("dst_schedule_sha256", None)
    if (
        _canonical_sha(dst_body) != dst_claimed
        or seal.get("dst_schedule_sha256") != dst_claimed
    ):
        raise AdaptiveStoryCliError("evaluator DST schedule digest mismatch")
    selection = seal.get("selection_semantics")
    if not isinstance(selection, Mapping):
        raise AdaptiveStoryCliError("evaluator selection semantics seal is missing")
    selection_body = dict(selection)
    selection_claimed = selection_body.pop("selection_semantics_sha256", None)
    if (
        _canonical_sha(selection_body) != selection_claimed
        or seal.get("selection_semantics_sha256") != selection_claimed
        or selection.get("selectable_candidate_count") != 50
    ):
        raise AdaptiveStoryCliError("evaluator selection semantics digest mismatch")
    return {**seal, "evaluator_dependency_sha256": claimed}


def _require_current_evaluator_seal(expected: Mapping[str, Any]) -> None:
    validated = _validate_evaluator_dependency_seal(expected)
    if _evaluator_dependency_seal() != validated:
        raise AdaptiveStoryCliError("evaluator dependencies changed during phase")


def _require_prior_evaluator_match(
    receipt: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    prior = _validate_evaluator_dependency_seal(
        receipt.get("evaluator_dependency_seal")
    )
    if prior != dict(current):
        raise AdaptiveStoryCliError(
            "downstream evaluator dependencies differ from the sealed prior phase"
        )


def _validate_no_authority(value: object) -> None:
    false_only = {
        "forward_proof_eligible",
        "live_permission",
        "live_permission_granted",
        "live_order_enabled",
        "promotion_allowed",
        "automatic_promotion_allowed",
        "broker_mutation_allowed",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name in false_only and item is not False:
                raise AdaptiveStoryCliError(
                    f"artifact contradicts authority boundary at {name}"
                )
            if name == "order_authority" and item != "NONE":
                raise AdaptiveStoryCliError("artifact grants order authority")
            if name == "live_side_effects" and item not in (None, [], ()):
                raise AdaptiveStoryCliError("artifact declares live side effects")
            _validate_no_authority(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_no_authority(item)


def _load_receipt(path: Path, *, expected_contract: str) -> dict[str, Any]:
    value = _strict_json_loads(_read_stable_bytes(path), label=str(path))
    if not isinstance(value, dict):
        raise AdaptiveStoryCliError("phase receipt must be an object")
    if value.get("contract") != expected_contract:
        raise AdaptiveStoryCliError("phase receipt contract is invalid")
    expected_phase = {
        TRAIN_CONTRACT: "TRAIN",
        VALIDATION_CONTRACT: "VALIDATION",
        HOLDOUT_CONTRACT: "HOLDOUT",
    }.get(expected_contract)
    if value.get("schema_version") != 2 or value.get("phase") != expected_phase:
        raise AdaptiveStoryCliError("phase receipt schema/phase is invalid")
    claimed = value.get("receipt_sha256")
    if not isinstance(claimed, str) or _SHA_RE.fullmatch(claimed) is None:
        raise AdaptiveStoryCliError("phase receipt digest is invalid")
    body = dict(value)
    del body["receipt_sha256"]
    if _canonical_sha(body) != claimed:
        raise AdaptiveStoryCliError("phase receipt digest mismatch")
    _validate_no_authority(value)
    if value.get("price_only_cost_scope") != PRICE_ONLY_COST_SCOPE:
        raise AdaptiveStoryCliError("phase receipt price-only cost scope mismatch")
    if (
        value.get("fully_loaded_net_economics") is not False
        or value.get("live_net_claim_allowed") is not False
    ):
        raise AdaptiveStoryCliError("phase receipt overclaims net economics")
    if value.get("pair_execution_policy") != PAIR_EXECUTION_POLICY:
        raise AdaptiveStoryCliError("phase receipt pair execution policy mismatch")
    if value.get("integrity_evidence_not_external_authentication") is not True:
        raise AdaptiveStoryCliError("phase receipt overclaims external authentication")
    if not all(value.get(field) == expected for field, expected in _AUTHORITY.items()):
        raise AdaptiveStoryCliError("phase receipt authority contract mismatch")
    return value


def _require_distinct_holdout_paths(
    *,
    train_receipt: Path,
    validation_receipt: Path,
    output: Path,
) -> None:
    """Reject input/output aliases before any downstream phase work."""

    try:
        train_path = train_receipt.expanduser().resolve(strict=True)
        validation_path = validation_receipt.expanduser().resolve(strict=True)
        output_path = output.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise AdaptiveStoryCliError("holdout receipt path resolution failed") from error
    if len({train_path, validation_path, output_path}) != 3:
        raise AdaptiveStoryCliError("holdout receipt paths must be distinct")
    existing_paths = [train_path, validation_path]
    if output_path.exists():
        existing_paths.append(output_path)
    try:
        identities = {
            (value.st_dev, value.st_ino)
            for path in existing_paths
            for value in (os.stat(path, follow_symlinks=False),)
        }
    except OSError as error:
        raise AdaptiveStoryCliError("holdout receipt path identity failed") from error
    if len(identities) != len(existing_paths):
        raise AdaptiveStoryCliError("holdout receipt paths must not alias")


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AdaptiveStoryCliError("phase receipt contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _atomic_publish_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    evaluator_dependency_seal: Mapping[str, Any] | None = None,
) -> None:
    _reject_non_finite(value)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise AdaptiveStoryCliError("output path must not be a symlink")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if _read_stable_bytes(destination) == payload:
            return
        raise AdaptiveStoryCliError("immutable output path already exists")
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
        if evaluator_dependency_seal is not None:
            _require_current_evaluator_seal(evaluator_dependency_seal)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _read_stable_bytes(destination) != payload:
                raise AdaptiveStoryCliError("immutable output publication raced")
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _manifest_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": manifest.get("contract"),
        "schema_version": manifest.get("schema_version"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "selection_policy": manifest.get("selection_policy"),
        "selection_is_outcome_blind": manifest.get("selection_is_outcome_blind"),
        "allowed_run_ids": list(manifest.get("allowed_run_ids") or ()),
        "summary_run_ids": list(manifest.get("summary_run_ids") or ()),
        "run_scope_policy": manifest.get("run_scope_policy"),
        "summary_run_ids_exact_set_proved": manifest.get(
            "summary_run_ids_exact_set_proved"
        ),
        "run_scope_is_outcome_blind": manifest.get("run_scope_is_outcome_blind"),
        "expected_pairs": list(manifest.get("expected_pairs") or ()),
        "selected_pair_count": manifest.get("selected_pair_count"),
        "expected_pair_count": manifest.get("expected_pair_count"),
        "complete_pair_coverage": manifest.get("complete_pair_coverage"),
        "all_selected_sources_acquisition_receipted": manifest.get(
            "all_selected_sources_acquisition_receipted"
        ),
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


def _require_canonical_historical_manifest(value: Mapping[str, Any]) -> None:
    """Run the frozen value through the cache module's canonical validator.

    ``load_historical_s5_slices`` validates the manifest before it handles an
    empty request tuple.  Using that public batch API keeps this CLI exactly in
    sync with ``fast_bot_historical_s5._validate_manifest`` without copying a
    second validator or changing the separately owned cache module.
    """

    try:
        result = load_historical_s5_slices(value, requests=())
    except HistoricalS5CacheError as error:
        raise AdaptiveStoryCliError(
            "frozen historical manifest failed canonical validation"
        ) from error
    if result != ():
        raise AdaptiveStoryCliError(
            "canonical historical manifest validation returned unexpected data"
        )


def _validate_frozen_manifest(
    value: object,
    *,
    history_root: Path,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveStoryCliError("phase receipt has no frozen manifest")
    manifest = dict(value)
    _require_canonical_historical_manifest(manifest)
    claimed = manifest.get("manifest_sha256")
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if (
        not isinstance(claimed, str)
        or _SHA_RE.fullmatch(claimed) is None
        or _canonical_sha(body) != claimed
    ):
        raise AdaptiveStoryCliError("frozen historical manifest digest mismatch")
    if (
        manifest.get("contract") != "QR_FAST_BOT_HISTORICAL_S5_CACHE_MANIFEST_V1"
        or manifest.get("schema_version") != 1
    ):
        raise AdaptiveStoryCliError("frozen historical manifest contract mismatch")
    resolved_root = history_root.expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise AdaptiveStoryCliError("historical root is not a directory")
    if manifest.get("source_root") != str(resolved_root):
        raise AdaptiveStoryCliError("history-root differs from frozen manifest root")
    if scope.get("history_root") != str(resolved_root):
        raise AdaptiveStoryCliError("history-root differs from sealed research scope")
    pairs = scope.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise AdaptiveStoryCliError("sealed pair scope is invalid")
    if manifest.get("expected_pairs") != pairs:
        raise AdaptiveStoryCliError("frozen manifest pair universe changed")
    run_ids = scope.get("history_run_ids")
    if not isinstance(run_ids, list) or not run_ids:
        raise AdaptiveStoryCliError("sealed history run scope is invalid")
    canonical_run_ids = sorted(run_ids)
    if manifest.get("allowed_run_ids") != canonical_run_ids:
        raise AdaptiveStoryCliError("frozen manifest run-id scope changed")
    if manifest.get("summary_run_ids") != canonical_run_ids:
        raise AdaptiveStoryCliError("frozen manifest summary run set changed")
    if (
        manifest.get("run_scope_policy") != "EXPLICIT_ALLOWED_RUN_IDS_EXACT_SET_V1"
        or manifest.get("summary_run_ids_exact_set_proved") is not True
        or manifest.get("run_scope_is_outcome_blind") is not True
    ):
        raise AdaptiveStoryCliError("frozen manifest run-scope policy mismatch")
    if manifest.get("selection_policy") != (
        "EXPLICIT_ALLOWED_RUN_IDS_EXACT_SET_THEN_"
        "SUMMARY_PUBLISHED_ERROR_FREE_WIDEST_DECLARED_WINDOW_"
        "THEN_EARLIEST_FROM_THEN_RELATIVE_PATH_V1"
    ):
        raise AdaptiveStoryCliError("frozen manifest selection policy mismatch")
    if (
        manifest.get("selection_is_outcome_blind") is not True
        or manifest.get("coverage_policy")
        != "SUMMARY_DECLARED_INTERVAL_PLUS_STRICT_FILE_SCAN_V1"
    ):
        raise AdaptiveStoryCliError("frozen manifest evidence policy mismatch")
    if manifest.get("missing_pairs") != []:
        raise AdaptiveStoryCliError("TRAIN requires missing_pairs=[]")
    if manifest.get("complete_pair_coverage") is not True:
        raise AdaptiveStoryCliError("TRAIN requires complete pair coverage")
    if manifest.get("selected_pair_count") != len(pairs):
        raise AdaptiveStoryCliError("frozen manifest selected pair count mismatch")
    if manifest.get("expected_pair_count") != len(pairs):
        raise AdaptiveStoryCliError("frozen manifest expected pair count mismatch")
    if manifest.get("all_selected_sources_acquisition_receipted") is not True:
        raise AdaptiveStoryCliError(
            "TRAIN requires all selected sources acquisition-receipted"
        )
    sources = manifest.get("selected_sources")
    if not isinstance(sources, list):
        raise AdaptiveStoryCliError("frozen manifest source list is invalid")
    if [row.get("pair") for row in sources if isinstance(row, Mapping)] != pairs:
        raise AdaptiveStoryCliError("frozen manifest selected pair set/order changed")
    for source in sources:
        if not isinstance(source, Mapping):
            raise AdaptiveStoryCliError("frozen manifest source is invalid")
        source_body = {
            key: item for key, item in source.items() if key != "source_sha256"
        }
        if source.get("source_sha256") != _canonical_sha(source_body):
            raise AdaptiveStoryCliError("frozen manifest source seal mismatch")
        if (
            source.get("selection_policy") != manifest.get("selection_policy")
            or source.get("coverage_policy") != manifest.get("coverage_policy")
            or source.get("historical_only") is not True
            or source.get("forward_proof_eligible") is not False
        ):
            raise AdaptiveStoryCliError("frozen manifest source policy mismatch")
        if source.get("acquisition_receipt_proved") is not True:
            raise AdaptiveStoryCliError(
                "frozen manifest contains an unreceipted selected source"
            )
        for field in (
            "source_sha256",
            "file_sha256",
            "source_summary_sha256",
            "acquisition_receipt_sha256",
        ):
            if _SHA_RE.fullmatch(str(source.get(field) or "")) is None:
                raise AdaptiveStoryCliError(
                    f"frozen manifest source {field} is invalid"
                )
    _validate_no_authority(manifest)
    manifest_authority = {
        "historical_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "forward_proof_eligible": False,
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    if not all(
        manifest.get(field) == expected
        for field, expected in manifest_authority.items()
    ):
        raise AdaptiveStoryCliError("frozen manifest authority contract mismatch")
    return manifest


def _validate_manifest_coverage(
    manifest: Mapping[str, Any],
    *,
    from_utc: datetime,
    to_utc: datetime,
) -> None:
    common_from = _receipt_utc(manifest.get("common_declared_from_utc"))
    common_to = _receipt_utc(manifest.get("common_declared_to_utc"))
    if from_utc < common_from or to_utc > common_to:
        raise AdaptiveStoryCliError("phase scope exceeds common manifest coverage")


def _catalog_rows() -> tuple[Any, ...]:
    rows = tuple(
        item
        for item in story_core.build_story_vehicle_catalog_v2()
        if not item.no_trade_control
    )
    if len(rows) != 50 or len({item.candidate_id for item in rows}) != 50:
        raise AdaptiveStoryCliError("V2 story catalog is not the fixed 50 candidates")
    return rows


def _result_candidate_ids(result: Mapping[str, Any]) -> tuple[str, ...]:
    rows = result.get("all_trials")
    if not isinstance(rows, list):
        raise AdaptiveStoryCliError("core result has no candidate trial list")
    ids = tuple(
        str(row.get("candidate_id") or "")
        for row in rows
        if isinstance(row, Mapping) and row.get("scorecard_eligible") is True
    )
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise AdaptiveStoryCliError("core result candidate scope is invalid")
    return ids


def _require_result_seal(
    result: Mapping[str, Any], *, digest_field: str, label: str
) -> None:
    claimed = result.get(digest_field)
    if not isinstance(claimed, str) or _SHA_RE.fullmatch(claimed) is None:
        raise AdaptiveStoryCliError(f"{label} has no valid {digest_field}")
    body = {key: item for key, item in result.items() if key != digest_field}
    if _canonical_sha(body) != claimed:
        raise AdaptiveStoryCliError(f"{label} {digest_field} mismatch")


def _expected_split_rows(split: Any) -> list[dict[str, str]]:
    return [
        {
            "name": split.name,
            "from_utc": split.from_utc.isoformat(),
            "to_utc": split.to_utc.isoformat(),
        }
    ]


def _vehicles_for_candidate_ids(candidate_ids: Sequence[str]) -> tuple[Any, ...]:
    by_id = {
        row.candidate_id: row
        for row in story_core.build_story_vehicle_catalog_v2()
        if not row.no_trade_control
    }
    requested = tuple(candidate_ids)
    if any(candidate_id not in by_id for candidate_id in requested):
        raise AdaptiveStoryCliError("phase candidate whitelist is not in V2 catalog")
    catalog_order = tuple(
        row for row in _catalog_rows() if row.candidate_id in requested
    )
    if tuple(row.candidate_id for row in catalog_order) != requested:
        raise AdaptiveStoryCliError("phase candidate whitelist is not in catalog order")
    return catalog_order


def _validate_pair_result_contract(
    result: Mapping[str, Any],
    *,
    pair: str,
    split: Any,
    candidate_ids: Sequence[str],
    expected_statuses: Sequence[str],
) -> None:
    _require_result_seal(result, digest_field="result_sha256", label="pair result")
    vehicles = _vehicles_for_candidate_ids(candidate_ids)
    requested = [row.candidate_id for row in vehicles]
    split_rows = _expected_split_rows(split)
    expected_exact: dict[str, Any] = {
        "contract": story_core.STORY_GRID_CONTRACT_V2,
        "schema_version": 2,
        "pair": pair,
        "story_catalog_policy": story_core.STORY_CATALOG_POLICY_V2,
        "truth_policy": story_core.STORY_TRUTH_POLICY_V2,
        "story_catalog_sha256": _canonical_sha(story_core._story_catalog_receipt_v2()),
        "truth_evaluator_sha256": _canonical_sha(
            story_core._truth_evaluator_receipt_v2()
        ),
        "price_precision_policy": story_core.PRICE_PRECISION_POLICY_V2,
        "price_cost_scope": dict(story_core.PRICE_COST_SCOPE_V2),
        "entry_ttl_boundary": "EXCLUSIVE",
        "intrabar_resting_fill_s5_policy": (
            "NO_TARGET;STOP_RANGE_CHARGED_CONSERVATIVELY;NO_PREFILL_OPEN_GAP"
        ),
        "entry_gap_invalid_geometry_policy": (
            "BROKER_ON_FILL_DEPENDENT_ORDER_LOSS_CANCEL_NO_FILL"
        ),
        "requested_candidate_ids": requested,
        "evaluated_candidate_ids": requested,
        "requested_control_candidate_ids": [],
        "candidate_whitelist_sha256": _canonical_sha(requested),
        "split_receipt": split_rows,
        "split_digest": _canonical_sha(split_rows),
        "daily_aggregates_complete": True,
        "daily_cluster_basis": "ENTRY_UTC_DATE",
        "exit_day_or_mark_to_market_used_for_selection": False,
        "contextual_order_cross_product_forbidden": True,
        "setup_trigger_entry_policy": "T_SETUP_LT_T_TRIGGER_LT_T_ENTRY",
        "quote_observation_policy": (
            "FIRST_REAL_S5_AFTER_TRIGGER_OBSERVES_ONLY;"
            "FOLLOWING_REAL_S5_IS_EARLIEST_FILL"
        ),
    }
    for field, expected in expected_exact.items():
        if result.get(field) != expected:
            raise AdaptiveStoryCliError(f"pair result {field} contract mismatch")
    if result.get("status") not in tuple(expected_statuses):
        raise AdaptiveStoryCliError("pair result status is invalid for source state")
    if result.get("candidate_count") != len(requested):
        raise AdaptiveStoryCliError("pair result candidate count mismatch")
    trials = result.get("all_trials")
    if not isinstance(trials, list) or len(trials) != len(vehicles):
        raise AdaptiveStoryCliError("pair result trial schema is invalid")
    expected_dates = list(_expected_utc_day_labels(split))
    for trial, vehicle in zip(trials, vehicles, strict=True):
        if not isinstance(trial, Mapping):
            raise AdaptiveStoryCliError("pair result trial is invalid")
        trial_exact = {
            "candidate_id": vehicle.candidate_id,
            "hypothesis_id": vehicle.hypothesis_id,
            "story_name": vehicle.story_name,
            "exit_policy_id": vehicle.exit_policy_id,
            "contextual_order_policy": vehicle.contextual_order_policy,
            "allowed_order_modes": list(vehicle.allowed_order_modes),
            "max_hold_seconds": vehicle.max_hold_seconds,
            "profit_target_r": vehicle.profit_target_r,
            "trailing_structural": vehicle.trailing_structural,
            "complexity": vehicle.complexity,
            "no_trade_control": False,
            "scorecard_eligible": True,
        }
        if any(trial.get(field) != expected for field, expected in trial_exact.items()):
            raise AdaptiveStoryCliError("pair result trial metadata mismatch")
        by_split = trial.get("by_split")
        daily_by_split = trial.get("daily_aggregates_by_split")
        if (
            not isinstance(by_split, Mapping)
            or set(by_split) != {split.name}
            or not isinstance(by_split.get(split.name), Mapping)
            or not isinstance(daily_by_split, Mapping)
            or set(daily_by_split) != {split.name}
            or not isinstance(daily_by_split.get(split.name), list)
        ):
            raise AdaptiveStoryCliError("pair result trial split schema mismatch")
        daily = daily_by_split[split.name]
        if [
            row.get("utc_date") for row in daily if isinstance(row, Mapping)
        ] != expected_dates:
            raise AdaptiveStoryCliError("pair result daily aggregate scope mismatch")
    if not all(result.get(field) == expected for field, expected in _AUTHORITY.items()):
        raise AdaptiveStoryCliError("pair result authority contract mismatch")


def _run_core(
    *,
    pair: str,
    candles: Sequence[Any],
    split: Any,
    unavailable_pairs: Sequence[str],
    candidate_ids: Sequence[str] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"unavailable_pairs": tuple(unavailable_pairs)}
    if candidate_ids is not None:
        kwargs["candidate_ids"] = tuple(candidate_ids)
    try:
        result = story_core.run_adaptive_story_s5_grid(
            pair,
            candles,
            (split,),
            **kwargs,
        )
    except TypeError as error:
        if candidate_ids is not None and "candidate_ids" in str(error):
            raise AdaptiveStoryCoreCapabilityError(
                "core candidate whitelist API is required for strict phase execution"
            ) from error
        raise
    if not isinstance(result, dict):
        raise AdaptiveStoryCliError("core phase result must be an object")
    _validate_no_authority(result)
    expected = (
        tuple(item.candidate_id for item in _catalog_rows())
        if candidate_ids is None
        else tuple(candidate_ids)
    )
    actual = _result_candidate_ids(result)
    if actual != expected:
        raise AdaptiveStoryCliError(
            "core calculated or exposed candidates outside phase scope"
        )
    _validate_pair_result_contract(
        result,
        pair=pair,
        split=split,
        candidate_ids=expected,
        expected_statuses=(
            ("UNAVAILABLE",)
            if pair in set(unavailable_pairs)
            else ("COMPLETE", "NO_DATA")
        ),
    )
    return result


def _run_one_pair_phase(
    *,
    manifest: Mapping[str, Any],
    pair: str,
    split: Any,
    candidate_ids: Sequence[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    unavailable = tuple(str(item) for item in manifest.get("missing_pairs") or ())
    unavailable_set = set(unavailable)
    sources = {
        str(row["pair"]): row
        for row in manifest.get("selected_sources") or ()
        if isinstance(row, Mapping)
    }
    if pair in unavailable_set:
        result = _run_core(
            pair=pair,
            candles=(),
            split=split,
            unavailable_pairs=unavailable,
            candidate_ids=candidate_ids,
        )
        if result.get("status") != "UNAVAILABLE":
            raise AdaptiveStoryCliError("core lost unavailable-pair state")
        return (
            {
                "pair": pair,
                "source_status": "UNAVAILABLE",
                "source_receipt": None,
                "slice_receipt": None,
                "result": result,
            },
            result,
        )
    source = sources.get(pair)
    if source is None:
        raise AdaptiveStoryCliError(f"manifest omitted admitted source for {pair}")
    loaded = load_historical_s5_slice(
        manifest,
        pair=pair,
        time_from=split.from_utc,
        time_to=split.to_utc,
    )
    result = _run_core(
        pair=pair,
        candles=loaded.candles,
        split=split,
        unavailable_pairs=unavailable,
        candidate_ids=candidate_ids,
    )
    return (
        {
            "pair": pair,
            "source_status": "ADMITTED",
            "source_receipt": _source_receipt(source),
            "slice_receipt": loaded.receipt(),
            "result": result,
        },
        result,
    )


def _run_one_pair_phase_payload(
    payload: tuple[dict[str, Any], str, Any, tuple[str, ...] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, pair, split, candidate_ids = payload
    return _run_one_pair_phase(
        manifest=manifest,
        pair=pair,
        split=split,
        candidate_ids=candidate_ids,
    )


def _run_pair_phase(
    *,
    manifest: Mapping[str, Any],
    pairs: Sequence[str],
    split: Any,
    candidate_ids: Sequence[str] | None,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_pairs = tuple(pairs)
    if not ordered_pairs:
        raise AdaptiveStoryCliError("pair execution scope cannot be empty")
    if not 1 <= workers <= MAX_PAIR_WORKERS:
        raise AdaptiveStoryCliError("pair worker count is outside the fixed bound")
    frozen_manifest = dict(manifest)
    frozen_ids = tuple(candidate_ids) if candidate_ids is not None else None
    payloads = tuple(
        (frozen_manifest, pair, split, frozen_ids) for pair in ordered_pairs
    )
    if workers == 1 or len(payloads) == 1:
        rows = [_run_one_pair_phase_payload(payload) for payload in payloads]
    else:
        maximum = min(workers, len(payloads), MAX_PAIR_WORKERS)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=maximum,
            mp_context=context,
        ) as executor:
            # executor.map preserves input order; receipts never include worker
            # assignment or completion order, so serial and parallel bytes match.
            rows = list(executor.map(_run_one_pair_phase_payload, payloads))
    artifacts = [artifact for artifact, _result in rows]
    results = [result for _artifact, result in rows]
    if [artifact.get("pair") for artifact in artifacts] != list(ordered_pairs):
        raise AdaptiveStoryCliError("parallel pair execution changed input order")
    return artifacts, results


def _trial_by_id(
    result: Mapping[str, Any], split_name: str
) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in result.get("all_trials") or ():
        if not isinstance(row, Mapping) or row.get("scorecard_eligible") is not True:
            continue
        by_split = row.get("by_split")
        if not isinstance(by_split, Mapping) or not isinstance(
            by_split.get(split_name), Mapping
        ):
            raise AdaptiveStoryCliError("core trial is missing its phase metric")
        rows[str(row["candidate_id"])] = by_split[split_name]
    return rows


def _expected_utc_day_labels(split: Any) -> tuple[str, ...]:
    """Return every UTC calendar day touched by the half-open split."""

    if split.from_utc >= split.to_utc:
        raise AdaptiveStoryCliError("train split interval is invalid")
    cursor = split.from_utc.date()
    final = (split.to_utc - timedelta(microseconds=1)).date()
    labels: list[str] = []
    while cursor <= final:
        labels.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if not labels:
        raise AdaptiveStoryCliError("train split has no UTC day cluster")
    return tuple(labels)


def _daily_train_metrics(
    global_result: Mapping[str, Any],
    split: Any,
) -> dict[str, dict[str, Any]]:
    expected_dates = _expected_utc_day_labels(split)
    rows = global_result.get("candidate_metrics")
    if not isinstance(rows, list):
        raise AdaptiveStoryCliError("train global result has no candidate metrics")
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AdaptiveStoryCliError("train global candidate metric is invalid")
        candidate_id = str(row.get("candidate_id") or "")
        by_split = row.get("by_split")
        if not isinstance(by_split, Mapping):
            raise AdaptiveStoryCliError("train global candidate has no split metrics")
        train = by_split.get("TRAIN")
        if not isinstance(train, Mapping):
            raise AdaptiveStoryCliError("train global candidate has no TRAIN metric")
        daily = train.get("daily_net_r")
        if not isinstance(daily, list):
            raise AdaptiveStoryCliError("train candidate has no daily net-R vector")
        dates: list[str] = []
        values: list[float] = []
        resolved_counts: list[int] = []
        for item in daily:
            if not isinstance(item, Mapping):
                raise AdaptiveStoryCliError("train daily net-R row is invalid")
            utc_date = item.get("utc_date")
            if not isinstance(utc_date, str):
                raise AdaptiveStoryCliError("train daily net-R date is invalid")
            raw_value = item.get("exact_net_r")
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise AdaptiveStoryCliError("train daily net-R is non-finite")
            raw_resolved = item.get("resolved_count")
            if (
                isinstance(raw_resolved, bool)
                or not isinstance(raw_resolved, int)
                or raw_resolved < 0
            ):
                raise AdaptiveStoryCliError("train daily resolved count is invalid")
            value = float(raw_value)
            if raw_resolved == 0 and value != 0.0:
                raise AdaptiveStoryCliError("zero-trade UTC day has non-zero net-R")
            dates.append(utc_date)
            values.append(value)
            resolved_counts.append(raw_resolved)
        if tuple(dates) != expected_dates:
            raise AdaptiveStoryCliError(
                "train daily net-R vector has missing, duplicate, or unordered UTC days"
            )
        if candidate_id in parsed:
            raise AdaptiveStoryCliError("duplicate train global candidate metric")
        mean = statistics.fmean(values)
        blocks: list[dict[str, Any]] = []
        for block_index, offset in enumerate(range(0, len(values), 2)):
            block_values = values[offset : offset + 2]
            block_dates = dates[offset : offset + 2]
            block_mean = statistics.fmean(block_values)
            blocks.append(
                {
                    "block_index": block_index,
                    "utc_dates": block_dates,
                    "contained_day_count": len(block_values),
                    "exact_net_r": sum(block_values),
                    "daily_mean_net_r": block_mean,
                }
            )
        block_count = len(blocks)
        day_count = len(values)
        standard_error = None
        if block_count >= 2:
            centered_cluster_sum_squares = sum(
                (float(block["exact_net_r"]) - int(block["contained_day_count"]) * mean)
                ** 2
                for block in blocks
            )
            cluster_variance = (
                block_count
                / (block_count - 1)
                * centered_cluster_sum_squares
                / (day_count**2)
            )
            if cluster_variance < 0.0 or not math.isfinite(cluster_variance):
                raise AdaptiveStoryCliError("train two-day cluster variance is invalid")
            standard_error = math.sqrt(cluster_variance)
        parsed[candidate_id] = {
            "daily_net_r": [
                {
                    "utc_date": utc_date,
                    "exact_net_r": value,
                    "resolved_count": resolved,
                }
                for utc_date, value, resolved in zip(
                    dates, values, resolved_counts, strict=True
                )
            ],
            "daily_net_r_sha256": _canonical_sha(daily),
            "utc_calendar_day_count": len(values),
            "two_day_blocks": blocks,
            "two_day_block_sizes": [len(block["utc_dates"]) for block in blocks],
            "non_overlapping_two_day_block_cluster_count": block_count,
            "resolved_count": sum(resolved_counts),
            "mean_daily_net_r": mean,
            "two_day_block_cluster_robust_standard_error_daily_r": standard_error,
            "two_day_block_standard_error_daily_r_compatibility_alias": (
                standard_error
            ),
            "cluster_robust_se_formula": (
                "SQRT(B/(B-1)*SUM((S_C-N_C*MEAN_DAILY_R)^2)/N^2)"
            ),
        }
    return parsed


def _pair_cluster_diagnostic(
    pair_results: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    pair_means: list[float] = []
    for result in pair_results:
        if result.get("status") == "UNAVAILABLE":
            continue
        metric = _trial_by_id(result, "TRAIN")[candidate_id]
        resolved = int(metric.get("resolved_count", 0))
        net_r = float(metric.get("exact_net_r", 0.0))
        if resolved > 0:
            pair_means.append(net_r / resolved)
    return {
        "selection_eligible": False,
        "resolved_pair_count": len(pair_means),
        "mean_pair_average_net_r": (
            statistics.fmean(pair_means) if pair_means else None
        ),
        "pair_cluster_standard_error_r": (
            statistics.stdev(pair_means) / math.sqrt(len(pair_means))
            if len(pair_means) >= 2
            else None
        ),
    }


def _train_selection(
    global_result: Mapping[str, Any],
    pair_results: Sequence[Mapping[str, Any]],
    split: Any,
) -> dict[str, Any]:
    catalog = _catalog_rows()
    daily_by_id = _daily_train_metrics(global_result, split)
    expected_ids = tuple(vehicle.candidate_id for vehicle in catalog)
    if tuple(daily_by_id) != expected_ids:
        raise AdaptiveStoryCliError("train global daily candidate scope is invalid")
    metrics: list[dict[str, Any]] = []
    for vehicle in catalog:
        daily = daily_by_id[vehicle.candidate_id]
        metrics.append(
            {
                "candidate_id": vehicle.candidate_id,
                "hypothesis_id": vehicle.hypothesis_id,
                "exit_policy_id": vehicle.exit_policy_id,
                "complexity": int(vehicle.complexity),
                **daily,
                "pair_cluster_diagnostic": _pair_cluster_diagnostic(
                    pair_results,
                    vehicle.candidate_id,
                ),
            }
        )

    selected: list[str] = []
    story_receipts: list[dict[str, Any]] = []
    for hypothesis_id in (f"H{number}" for number in range(21, 31)):
        rows = [row for row in metrics if row["hypothesis_id"] == hypothesis_id]
        defaults = [row for row in rows if row["exit_policy_id"] == "TIME_1H"]
        if len(defaults) != 1 or int(defaults[0]["complexity"]) != min(
            int(row["complexity"]) for row in rows
        ):
            raise AdaptiveStoryCliError(
                "TIME_1H is not the unique preregistered simplest exit"
            )
        sufficient = all(
            int(row["non_overlapping_two_day_block_cluster_count"]) >= 2
            and row["two_day_block_cluster_robust_standard_error_daily_r"] is not None
            for row in rows
        )
        observed_leader = min(
            rows,
            key=lambda row: (
                -float(row["mean_daily_net_r"]),
                float(row["two_day_block_cluster_robust_standard_error_daily_r"])
                if row["two_day_block_cluster_robust_standard_error_daily_r"]
                is not None
                else math.inf,
                int(row["complexity"]),
                str(row["candidate_id"]),
            ),
        )
        if sufficient:
            best = min(
                rows,
                key=lambda row: (
                    -float(row["mean_daily_net_r"]),
                    float(row["two_day_block_cluster_robust_standard_error_daily_r"]),
                    int(row["complexity"]),
                    str(row["candidate_id"]),
                ),
            )
            best_se = float(best["two_day_block_cluster_robust_standard_error_daily_r"])
            threshold = float(best["mean_daily_net_r"]) - best_se
            within = [
                row for row in rows if float(row["mean_daily_net_r"]) >= threshold
            ]
            chosen = min(
                within,
                key=lambda row: (
                    int(row["complexity"]),
                    -float(row["mean_daily_net_r"]),
                    float(row["two_day_block_cluster_robust_standard_error_daily_r"]),
                    str(row["candidate_id"]),
                ),
            )
            basis = "OBSERVED_ONE_SE"
        else:
            best = None
            chosen = defaults[0]
            threshold = None
            basis = "TRAIN_EXIT_DEFAULT_INSUFFICIENT_CLUSTERS"
        selected.append(str(chosen["candidate_id"]))
        story_receipts.append(
            {
                "hypothesis_id": hypothesis_id,
                "best_candidate_id": best["candidate_id"] if best else None,
                "observed_mean_leader_candidate_id": observed_leader["candidate_id"],
                "one_se_threshold_r": threshold,
                "selected_candidate_id": chosen["candidate_id"],
                "selection_basis": basis,
            }
        )
    if len(selected) != 10 or len(set(selected)) != 10:
        raise AdaptiveStoryCliError("train phase did not freeze exactly ten candidates")
    return {
        "policy": TRAIN_SELECTION_POLICY,
        "estimator": "COMPLETE_UTC_DAILY_NET_R_ZERO_DAYS_INCLUDED",
        "reference_mean": "ALL_CALENDAR_DAYS_EQUAL_WEIGHT",
        "standard_error_policy": (
            "WEIGHTED_CLUSTER_ROBUST_TWO_DAY_BLOCK_SE_ANCHORED_AT_SPLIT_START"
        ),
        "non_overlapping_cluster_unit": ("NON_OVERLAPPING_TWO_UTC_CALENDAR_DAY_BLOCK"),
        "cluster_robust_se_formula": (
            "SQRT(B/(B-1)*SUM((S_C-N_C*MEAN_DAILY_R)^2)/N^2)"
        ),
        "final_incomplete_block_included": True,
        "insufficient_cluster_policy": (
            "LESS_THAN_2_OR_UNDEFINED_SE_SELECT_PREREGISTERED_TIME_1H"
        ),
        "best_tie_rule": "MEAN_DAILY_R_DESC_SE_ASC_COMPLEXITY_ASC_ID_ASC",
        "within_one_se_tie_rule": ("COMPLEXITY_ASC_MEAN_DAILY_R_DESC_SE_ASC_ID_ASC"),
        "pair_cluster_diagnostic_selection_eligible": False,
        "candidate_metrics": metrics,
        "story_selections": story_receipts,
        "selected_candidate_ids": selected,
        "selection_uses_validation": False,
        "selection_uses_holdout": False,
    }


def _combine_phase(
    pair_results: Sequence[Mapping[str, Any]],
    split: Any,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    combine = getattr(story_core, "combine_adaptive_story_s5_grid_runs", None)
    if not callable(combine):
        raise AdaptiveStoryCoreCapabilityError(
            "core global combine API is required before train/validation/holdout"
        )
    try:
        result = combine(
            pair_results,
            (split,),
            candidate_ids=tuple(candidate_ids),
        )
    except TypeError as error:
        raise AdaptiveStoryCoreCapabilityError(
            "core global combine candidate whitelist API is incomplete"
        ) from error
    if not isinstance(result, dict):
        raise AdaptiveStoryCliError("core global phase result must be an object")
    _validate_no_authority(result)
    _validate_global_result_contract(
        result,
        pair_results=pair_results,
        split=split,
        candidate_ids=candidate_ids,
    )
    actual = _global_candidate_ids(result)
    if actual != tuple(candidate_ids):
        raise AdaptiveStoryCliError(
            "global combine exposed candidates outside phase scope"
        )
    return result


def _validate_global_result_contract(
    result: Mapping[str, Any],
    *,
    pair_results: Sequence[Mapping[str, Any]],
    split: Any,
    candidate_ids: Sequence[str],
) -> None:
    _require_result_seal(result, digest_field="result_sha256", label="global result")
    vehicles = _vehicles_for_candidate_ids(candidate_ids)
    requested = [row.candidate_id for row in vehicles]
    split_rows = _expected_split_rows(split)
    expected_pairs = sorted(str(row.get("pair") or "") for row in pair_results)
    expected_exact: dict[str, Any] = {
        "contract": story_core.STORY_GRID_COMBINED_CONTRACT_V2,
        "schema_version": 2,
        "status": "COMPLETE",
        "pair_count": len(expected_pairs),
        "pairs": expected_pairs,
        "requested_candidate_ids": requested,
        "evaluated_candidate_ids": requested,
        "candidate_whitelist_sha256": _canonical_sha(requested),
        "story_catalog_policy": story_core.STORY_CATALOG_POLICY_V2,
        "truth_policy": story_core.STORY_TRUTH_POLICY_V2,
        "story_catalog_sha256": _canonical_sha(story_core._story_catalog_receipt_v2()),
        "truth_evaluator_sha256": _canonical_sha(
            story_core._truth_evaluator_receipt_v2()
        ),
        "accepted_pair_run_statuses": list(story_core.PAIR_RUN_ALLOWED_STATUSES),
        "price_precision_policy": story_core.PRICE_PRECISION_POLICY_V2,
        "price_cost_scope": dict(story_core.PRICE_COST_SCOPE_V2),
        "split_receipt": split_rows,
        "split_digest": _canonical_sha(split_rows),
        "daily_cluster_basis": "ENTRY_UTC_DATE",
        "daily_zero_fill_policy": "ALL_SPLIT_CALENDAR_UTC_DAYS",
        "daily_aggregates_source": ("COMPLETE_PAIR_RUN_AGGREGATES_NOT_AUDIT_ROWS"),
        "economic_screen_is_statistical_proof": False,
    }
    for field, expected in expected_exact.items():
        if result.get(field) != expected:
            raise AdaptiveStoryCliError(f"global result {field} contract mismatch")
    rows = result.get("candidate_metrics")
    if not isinstance(rows, list) or len(rows) != len(vehicles):
        raise AdaptiveStoryCliError("global result candidate schema mismatch")
    expected_dates = list(_expected_utc_day_labels(split))
    for row, vehicle in zip(rows, vehicles, strict=True):
        if not isinstance(row, Mapping):
            raise AdaptiveStoryCliError("global candidate metric is invalid")
        row_exact = {
            "candidate_id": vehicle.candidate_id,
            "hypothesis_id": vehicle.hypothesis_id,
            "story_name": vehicle.story_name,
            "exit_policy_id": vehicle.exit_policy_id,
            "no_trade_control": False,
        }
        if any(row.get(field) != expected for field, expected in row_exact.items()):
            raise AdaptiveStoryCliError("global candidate metadata mismatch")
        by_split = row.get("by_split")
        screens = row.get("economic_screen_by_split")
        if (
            not isinstance(by_split, Mapping)
            or set(by_split) != {split.name}
            or not isinstance(by_split.get(split.name), Mapping)
            or not isinstance(screens, Mapping)
            or set(screens) != {split.name}
            or not isinstance(screens.get(split.name), Mapping)
        ):
            raise AdaptiveStoryCliError("global candidate split scope mismatch")
        daily = by_split[split.name].get("daily_net_r")
        if (
            not isinstance(daily, list)
            or [item.get("utc_date") for item in daily if isinstance(item, Mapping)]
            != expected_dates
        ):
            raise AdaptiveStoryCliError("global candidate daily scope mismatch")
    survivors = result.get("economic_survivor_ids")
    if not isinstance(survivors, list) or any(
        survivor not in requested for survivor in survivors
    ):
        raise AdaptiveStoryCliError("global economic survivor scope mismatch")
    if not all(result.get(field) == expected for field, expected in _AUTHORITY.items()):
        raise AdaptiveStoryCliError("global result authority contract mismatch")


def _global_candidate_ids(result: Mapping[str, Any]) -> tuple[str, ...]:
    rows = result.get("candidate_metrics")
    if not isinstance(rows, list):
        raise AdaptiveStoryCliError("global result has no candidate metrics")
    ids = tuple(
        str(row.get("candidate_id") or "") for row in rows if isinstance(row, Mapping)
    )
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise AdaptiveStoryCliError("global candidate metric scope is invalid")
    return ids


def _no_survivor_holdout_skip_result(
    *,
    split: Any,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    split_rows = _expected_split_rows(split)
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_NO_SURVIVOR_HOLDOUT_SKIP_V1",
        "schema_version": 1,
        "status": "NO_VALIDATION_SURVIVORS",
        "phase": "HOLDOUT",
        "core_combiner_invoked": False,
        "skip_reason": "SEALED_VALIDATION_SURVIVOR_SET_EMPTY",
        "source_manifest_sha256": manifest.get("manifest_sha256"),
        "requested_candidate_ids": [],
        "candidate_metrics": [],
        "selected_candidate_ids": [],
        "split_receipt": split_rows,
        "split_digest": _canonical_sha(split_rows),
        "price_only_cost_scope": dict(PRICE_ONLY_COST_SCOPE),
        "fully_loaded_net_economics": False,
        "live_net_claim_allowed": False,
        **_AUTHORITY,
    }
    return {**body, "result_sha256": _canonical_sha(body)}


def _validation_survivors(
    global_result: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> list[str]:
    rows = global_result.get("candidate_metrics")
    assert isinstance(rows, list)
    survivors: list[str] = []
    for row in rows:
        assert isinstance(row, Mapping)
        candidate_id = str(row["candidate_id"])
        screens = row.get("economic_screen_by_split")
        if not isinstance(screens, Mapping):
            raise AdaptiveStoryCliError("global candidate lacks economic screen map")
        screen = screens.get("VALIDATION")
        if not isinstance(screen, Mapping):
            raise AdaptiveStoryCliError("global candidate lacks validation screen")
        gates = screen.get("gates")
        required_gates = (
            "resolved_trade_floor_passed",
            "active_entry_day_floor_passed",
            "contributing_pair_floor_passed",
            "average_net_r_positive",
            "average_daily_net_r_positive",
            "profit_factor_r_above_one",
            "loocv_each_day_removed_total_r_positive",
            "no_unresolved_or_purged",
        )
        if not isinstance(gates, Mapping) or any(
            not isinstance(gates.get(name), bool) for name in required_gates
        ):
            raise AdaptiveStoryCliError("global economic screen gates are invalid")
        eligible = screen.get("eligible")
        if not isinstance(eligible, bool):
            raise AdaptiveStoryCliError("global economic screen eligibility is invalid")
        if eligible != all(bool(gates[name]) for name in required_gates):
            raise AdaptiveStoryCliError("global economic screen gates are inconsistent")
        if screen.get("screen_is_statistical_proof") is not False:
            raise AdaptiveStoryCliError("economic screen claims statistical proof")
        if eligible is True:
            survivors.append(candidate_id)
    claimed = global_result.get("economic_survivor_ids")
    if not isinstance(claimed, list) or claimed != survivors:
        raise AdaptiveStoryCliError("global economic survivor list is inconsistent")
    if any(item not in candidate_ids for item in survivors):
        raise AdaptiveStoryCliError("validation selected an untrained candidate")
    return survivors


def _apply_train_validation_same_sign_gate(
    global_result: Mapping[str, Any],
    candidate_ids: Sequence[str],
    core_survivors: Sequence[str],
    train_means: Mapping[str, float],
) -> tuple[list[str], list[dict[str, Any]]]:
    rows = global_result.get("candidate_metrics")
    if not isinstance(rows, list):
        raise AdaptiveStoryCliError("validation global candidate metrics are invalid")
    by_id = {
        str(row.get("candidate_id") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    core_set = set(core_survivors)
    final: list[str] = []
    gates: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        row = by_id.get(candidate_id)
        if row is None:
            raise AdaptiveStoryCliError("validation candidate metric is missing")
        by_split = row.get("by_split")
        if not isinstance(by_split, Mapping) or not isinstance(
            by_split.get("VALIDATION"), Mapping
        ):
            raise AdaptiveStoryCliError("validation candidate has no phase metric")
        validation_value = by_split["VALIDATION"].get("average_daily_net_r")
        if (
            isinstance(validation_value, bool)
            or not isinstance(validation_value, (int, float))
            or not math.isfinite(float(validation_value))
        ):
            raise AdaptiveStoryCliError("validation mean daily R is invalid")
        validation_mean = float(validation_value)
        train_mean = train_means[candidate_id]
        core_eligible = candidate_id in core_set
        if core_eligible and validation_mean <= 0.0:
            raise AdaptiveStoryCliError(
                "core economic survivor has non-positive validation daily R"
            )
        same_sign_positive = train_mean > 0.0 and validation_mean > 0.0
        selected = core_eligible and same_sign_positive
        if selected:
            final.append(candidate_id)
            rejection_reason = None
        elif not core_eligible:
            rejection_reason = "VALIDATION_GLOBAL_ECONOMIC_SCREEN_FAILED"
        else:
            rejection_reason = "TRAIN_VALIDATION_MEAN_DAILY_R_NOT_BOTH_POSITIVE"
        gates.append(
            {
                "candidate_id": candidate_id,
                "train_mean_daily_net_r": train_mean,
                "validation_mean_daily_net_r": validation_mean,
                "core_validation_economic_screen_eligible": core_eligible,
                "train_validation_same_positive_sign_gate_passed": (same_sign_positive),
                "selected_as_final_survivor": selected,
                "rejection_reason": rejection_reason,
            }
        )
    return final, gates


def _new_portfolio_spec(candidate_ids: Sequence[str]) -> dict[str, Any]:
    ids = list(candidate_ids)
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise AdaptiveStoryCliError("fixed portfolio candidate scope is invalid")
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_FIXED_PORTFOLIO_SPEC_V2",
        "schema_version": 2,
        "allocation_label": PORTFOLIO_ALLOCATION,
        "candidate_ids": ids,
        "candidate_ids_sha256": _canonical_sha(ids),
        "candidate_order_policy": "SEALED_VALIDATION_SURVIVOR_ORDER",
        "all_validation_survivors_included": True,
        "subset_search_allowed": False,
        "weight_search_allowed": False,
        "gross_exposure_limit_applied": False,
        "real_capital_interpretation_allowed": False,
        "price_only_cost_scope": dict(PRICE_ONLY_COST_SCOPE),
        "fully_loaded_net_economics": False,
        **_AUTHORITY,
    }
    return {**body, "portfolio_spec_sha256": _canonical_sha(body)}


def _validate_portfolio_spec(
    value: object,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveStoryCliError("validation receipt has no fixed portfolio spec")
    spec = dict(value)
    claimed = spec.pop("portfolio_spec_sha256", None)
    if not isinstance(claimed, str) or _canonical_sha(spec) != claimed:
        raise AdaptiveStoryCliError("fixed portfolio spec digest mismatch")
    if spec.get("contract") != "QR_ADAPTIVE_STORY_FIXED_PORTFOLIO_SPEC_V2":
        raise AdaptiveStoryCliError("fixed portfolio spec contract is invalid")
    if spec.get("allocation_label") != PORTFOLIO_ALLOCATION:
        raise AdaptiveStoryCliError("fixed portfolio allocation changed")
    if spec.get("candidate_ids") != list(candidate_ids):
        raise AdaptiveStoryCliError("fixed portfolio candidate order changed")
    if spec.get("candidate_ids_sha256") != _canonical_sha(list(candidate_ids)):
        raise AdaptiveStoryCliError("fixed portfolio candidate digest mismatch")
    if any(
        spec.get(name) is not False
        for name in (
            "subset_search_allowed",
            "weight_search_allowed",
            "gross_exposure_limit_applied",
            "real_capital_interpretation_allowed",
        )
    ):
        raise AdaptiveStoryCliError("fixed portfolio permits adaptive capital choices")
    if spec.get("all_validation_survivors_included") is not True:
        raise AdaptiveStoryCliError("fixed portfolio omits validation survivors")
    if spec.get("price_only_cost_scope") != PRICE_ONLY_COST_SCOPE:
        raise AdaptiveStoryCliError("fixed portfolio cost scope changed")
    if spec.get("fully_loaded_net_economics") is not False:
        raise AdaptiveStoryCliError("fixed portfolio claims fully loaded economics")
    complete = {**spec, "portfolio_spec_sha256": claimed}
    _validate_no_authority(complete)
    if not all(
        complete.get(field) == expected for field, expected in _AUTHORITY.items()
    ):
        raise AdaptiveStoryCliError("fixed portfolio authority contract mismatch")
    return complete


def _validate_fixed_portfolio_result(
    value: object,
    *,
    expected_phase: str,
    candidate_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise AdaptiveStoryCliError("phase receipt has no fixed portfolio result")
    result = dict(value)
    _require_result_seal(
        result,
        digest_field="portfolio_result_sha256",
        label="fixed portfolio result",
    )
    if (
        result.get("contract") != "QR_ADAPTIVE_STORY_FIXED_PORTFOLIO_RESULT_V2"
        or result.get("schema_version") != 2
        or result.get("phase") != expected_phase
    ):
        raise AdaptiveStoryCliError("fixed portfolio result contract mismatch")
    spec = _validate_portfolio_spec(result.get("portfolio_spec"), candidate_ids)
    if result.get("portfolio_spec_sha256") != spec["portfolio_spec_sha256"]:
        raise AdaptiveStoryCliError("fixed portfolio result spec binding mismatch")
    if result.get("candidate_count") != len(candidate_ids):
        raise AdaptiveStoryCliError("fixed portfolio result candidate count mismatch")
    if (
        result.get("price_only_cost_scope") != PRICE_ONLY_COST_SCOPE
        or result.get("fully_loaded_net_economics") is not False
        or result.get("live_net_claim_allowed") is not False
    ):
        raise AdaptiveStoryCliError("fixed portfolio result cost scope mismatch")
    if not all(result.get(field) == expected for field, expected in _AUTHORITY.items()):
        raise AdaptiveStoryCliError("fixed portfolio result authority mismatch")
    return result, spec


def _portfolio_metric(
    row: Mapping[str, Any],
    *,
    split: Any,
    expected_dates: Sequence[str],
) -> tuple[list[float], list[int], float, float, int]:
    by_split = row.get("by_split")
    if not isinstance(by_split, Mapping):
        raise AdaptiveStoryCliError("portfolio candidate has no split metrics")
    metric = by_split.get(split.name)
    if not isinstance(metric, Mapping):
        raise AdaptiveStoryCliError("portfolio candidate has no phase metric")
    daily = metric.get("daily_net_r")
    if not isinstance(daily, list):
        raise AdaptiveStoryCliError("portfolio candidate has no daily net-R vector")
    dates: list[str] = []
    values: list[float] = []
    resolved_counts: list[int] = []
    for item in daily:
        if not isinstance(item, Mapping):
            raise AdaptiveStoryCliError("portfolio daily net-R row is invalid")
        utc_date = item.get("utc_date")
        raw_value = item.get("exact_net_r")
        raw_resolved = item.get("resolved_count")
        if not isinstance(utc_date, str):
            raise AdaptiveStoryCliError("portfolio daily UTC date is invalid")
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise AdaptiveStoryCliError("portfolio daily net-R is non-finite")
        if (
            isinstance(raw_resolved, bool)
            or not isinstance(raw_resolved, int)
            or raw_resolved < 0
        ):
            raise AdaptiveStoryCliError("portfolio daily trade count is invalid")
        value = float(raw_value)
        if raw_resolved == 0 and value != 0.0:
            raise AdaptiveStoryCliError("portfolio zero-trade day has non-zero net-R")
        dates.append(utc_date)
        values.append(value)
        resolved_counts.append(raw_resolved)
    if tuple(dates) != tuple(expected_dates):
        raise AdaptiveStoryCliError(
            "portfolio daily vector has missing, duplicate, or unordered UTC days"
        )

    def nonnegative_metric(name: str) -> float:
        value = metric.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise AdaptiveStoryCliError(f"portfolio {name} is invalid")
        return float(value)

    gross_profit = nonnegative_metric("gross_profit_r")
    gross_loss = nonnegative_metric("gross_loss_r")
    claimed_resolved = metric.get("resolved_count")
    if (
        isinstance(claimed_resolved, bool)
        or not isinstance(claimed_resolved, int)
        or claimed_resolved != sum(resolved_counts)
    ):
        raise AdaptiveStoryCliError(
            "portfolio resolved total disagrees with daily vector"
        )
    unresolved = metric.get("unresolved_or_purged_count")
    if (
        isinstance(unresolved, bool)
        or not isinstance(unresolved, int)
        or unresolved < 0
    ):
        raise AdaptiveStoryCliError("portfolio unresolved/purged count is invalid")
    if not math.isclose(
        sum(values),
        gross_profit - gross_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AdaptiveStoryCliError("portfolio net-R disagrees with gross components")
    return values, resolved_counts, gross_profit, gross_loss, unresolved


def _aggregate_fixed_portfolio(
    global_result: Mapping[str, Any],
    split: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_ids = spec.get("candidate_ids")
    if not isinstance(candidate_ids, list):
        raise AdaptiveStoryCliError("fixed portfolio spec has no candidate list")
    expected_dates = _expected_utc_day_labels(split)
    rows = global_result.get("candidate_metrics")
    if not isinstance(rows, list):
        raise AdaptiveStoryCliError("portfolio global result has no candidate metrics")
    by_id = {
        str(row.get("candidate_id") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if any(candidate_id not in by_id for candidate_id in candidate_ids):
        raise AdaptiveStoryCliError("portfolio survivor metric is missing")

    portfolio_daily = [0.0 for _ in expected_dates]
    portfolio_resolved = [0 for _ in expected_dates]
    gross_profit_r = 0.0
    gross_loss_r = 0.0
    unresolved_or_purged_count = 0
    for candidate_id in candidate_ids:
        values, resolved, candidate_profit, candidate_loss, unresolved = (
            _portfolio_metric(
                by_id[candidate_id],
                split=split,
                expected_dates=expected_dates,
            )
        )
        portfolio_daily = [
            aggregate + candidate
            for aggregate, candidate in zip(portfolio_daily, values, strict=True)
        ]
        portfolio_resolved = [
            aggregate + candidate
            for aggregate, candidate in zip(portfolio_resolved, resolved, strict=True)
        ]
        gross_profit_r += candidate_profit
        gross_loss_r += candidate_loss
        unresolved_or_purged_count += unresolved

    total_r = sum(portfolio_daily)
    if not math.isclose(
        total_r,
        gross_profit_r - gross_loss_r,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AdaptiveStoryCliError("fixed portfolio total disagrees with trade P/L")
    daily_profit_r = sum(max(value, 0.0) for value in portfolio_daily)
    daily_loss_r = sum(max(-value, 0.0) for value in portfolio_daily)
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_STORY_FIXED_PORTFOLIO_RESULT_V2",
        "schema_version": 2,
        "phase": split.name,
        "portfolio_spec": dict(spec),
        "portfolio_spec_sha256": spec["portfolio_spec_sha256"],
        "candidate_count": len(candidate_ids),
        "daily_net_r": [
            {
                "utc_date": utc_date,
                "exact_net_r": net_r,
                "resolved_count": resolved,
            }
            for utc_date, net_r, resolved in zip(
                expected_dates,
                portfolio_daily,
                portfolio_resolved,
                strict=True,
            )
        ],
        "utc_calendar_day_count": len(expected_dates),
        "zero_day_count": sum(value == 0.0 for value in portfolio_daily),
        "resolved_trade_count": sum(portfolio_resolved),
        "unresolved_or_purged_count": unresolved_or_purged_count,
        "trades_per_day": sum(portfolio_resolved) / len(expected_dates),
        "total_exact_net_r": total_r,
        "average_daily_net_r": total_r / len(expected_dates),
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "portfolio_profit_factor_r": (
            gross_profit_r / gross_loss_r if gross_loss_r > 0.0 else None
        ),
        "portfolio_profit_factor_r_infinite": (
            gross_loss_r == 0.0 and gross_profit_r > 0.0
        ),
        "portfolio_profit_factor_basis": "FILLED_TRADE_GROSS_PROFIT_LOSS_R",
        "daily_gross_profit_r": daily_profit_r,
        "daily_gross_loss_r": daily_loss_r,
        "daily_portfolio_profit_factor_r": (
            daily_profit_r / daily_loss_r if daily_loss_r > 0.0 else None
        ),
        "daily_portfolio_profit_factor_r_infinite": (
            daily_loss_r == 0.0 and daily_profit_r > 0.0
        ),
        "daily_portfolio_profit_factor_basis": (
            "POSITIVE_AND_NEGATIVE_AGGREGATED_PORTFOLIO_UTC_DAY_NET_R"
        ),
        "individual_candidate_rows_are_secondary": True,
        "subset_or_weight_search_performed": False,
        "price_only_cost_scope": dict(PRICE_ONLY_COST_SCOPE),
        "fully_loaded_net_economics": False,
        "live_net_claim_allowed": False,
        **_AUTHORITY,
    }
    return {**body, "portfolio_result_sha256": _canonical_sha(body)}


def _validate_phase_pair_evidence(
    receipt: Mapping[str, Any],
    *,
    scope: Mapping[str, Any],
    manifest: Mapping[str, Any],
    phase: str,
    candidate_ids: Sequence[str],
) -> tuple[Any, list[Mapping[str, Any]]]:
    """Validate sealed pair evidence for a completed research phase."""

    if phase not in {"TRAIN", "VALIDATION"}:
        raise AdaptiveStoryCliError("unsupported prior phase evidence")
    split = _split_from_scope(scope, phase)
    pairs = scope.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise AdaptiveStoryCliError(f"{phase.lower()} receipt pair scope is invalid")
    artifacts = receipt.get("pair_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(pairs):
        raise AdaptiveStoryCliError(f"{phase.lower()} pair artifact scope is invalid")
    unavailable = set(str(item) for item in manifest.get("missing_pairs") or ())
    source_by_pair = {
        str(row.get("pair") or ""): row
        for row in manifest.get("selected_sources") or ()
        if isinstance(row, Mapping)
    }
    pair_results: list[Mapping[str, Any]] = []
    for pair, artifact in zip(pairs, artifacts, strict=True):
        if not isinstance(artifact, Mapping) or artifact.get("pair") != pair:
            raise AdaptiveStoryCliError(
                f"{phase.lower()} pair artifacts changed pair order or identity"
            )
        expected_unavailable = pair in unavailable
        expected_source_status = "UNAVAILABLE" if expected_unavailable else "ADMITTED"
        if artifact.get("source_status") != expected_source_status:
            raise AdaptiveStoryCliError(
                f"{phase.lower()} pair artifact source status is inconsistent"
            )
        if expected_unavailable:
            if (
                artifact.get("source_receipt") is not None
                or artifact.get("slice_receipt") is not None
            ):
                raise AdaptiveStoryCliError(
                    f"unavailable {phase.lower()} pair carries admitted source evidence"
                )
        else:
            source = source_by_pair.get(pair)
            if source is None or artifact.get("source_receipt") != _source_receipt(
                source
            ):
                raise AdaptiveStoryCliError(
                    f"{phase.lower()} pair artifact source binding is invalid"
                )
            slice_receipt = artifact.get("slice_receipt")
            if not isinstance(slice_receipt, Mapping):
                raise AdaptiveStoryCliError(
                    f"{phase.lower()} pair artifact has no sealed slice"
                )
            _require_result_seal(
                slice_receipt,
                digest_field="slice_sha256",
                label=f"{phase.lower()} slice receipt",
            )
            expected_slice_fields = {
                "contract": "QR_FAST_BOT_HISTORICAL_S5_CACHE_SLICE_V1",
                "schema_version": 1,
                "pair": pair,
                "requested_from_utc": split.from_utc.isoformat(),
                "requested_to_utc": split.to_utc.isoformat(),
                "source_relative_path": source.get("relative_path"),
                "source_file_sha256": source.get("file_sha256"),
                "source_manifest_sha256": manifest.get("manifest_sha256"),
                "acquisition_receipt_proved": True,
                "historical_only": True,
                "diagnostic_only": True,
                "forward_proof_eligible": False,
                "promotion_allowed": False,
                "order_authority": "NONE",
                "shadow_only": True,
                "live_permission": False,
                "broker_mutation_allowed": False,
            }
            if any(
                slice_receipt.get(field) != expected
                for field, expected in expected_slice_fields.items()
            ):
                raise AdaptiveStoryCliError(
                    f"{phase.lower()} pair artifact slice binding is invalid"
                )
        result = artifact.get("result")
        if not isinstance(result, Mapping):
            raise AdaptiveStoryCliError(f"{phase.lower()} pair result is invalid")
        _validate_pair_result_contract(
            result,
            pair=pair,
            split=split,
            candidate_ids=candidate_ids,
            expected_statuses=(
                ("UNAVAILABLE",) if expected_unavailable else ("COMPLETE", "NO_DATA")
            ),
        )
        pair_results.append(result)
    return split, pair_results


def _validate_train_receipt_for_downstream(
    train: Mapping[str, Any],
    *,
    scope: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, float]]:
    """Recompute the TRAIN result and selection from sealed pair evidence."""

    if train.get("status") != "SEALED":
        raise AdaptiveStoryCliError("train receipt status is not sealed")
    expected_catalog_ids = [row.candidate_id for row in _catalog_rows()]
    executed_candidate_ids = train.get("executed_candidate_ids")
    if executed_candidate_ids != expected_catalog_ids:
        raise AdaptiveStoryCliError("train receipt candidate scope is invalid")
    train_split, pair_results = _validate_phase_pair_evidence(
        train,
        scope=scope,
        manifest=manifest,
        phase="TRAIN",
        candidate_ids=expected_catalog_ids,
    )
    provided_global = train.get("global_result")
    if not isinstance(provided_global, Mapping):
        raise AdaptiveStoryCliError("train receipt has no global result")
    _validate_global_result_contract(
        provided_global,
        pair_results=pair_results,
        split=train_split,
        candidate_ids=expected_catalog_ids,
    )
    recomputed_global = _combine_phase(
        pair_results,
        train_split,
        expected_catalog_ids,
    )
    if dict(provided_global) != recomputed_global:
        raise AdaptiveStoryCliError(
            "train global result differs from recomputed pair evidence"
        )
    provided_selection = train.get("selection")
    if not isinstance(provided_selection, Mapping):
        raise AdaptiveStoryCliError("train receipt has no selection")
    recomputed_selection = _train_selection(
        recomputed_global,
        pair_results,
        train_split,
    )
    if dict(provided_selection) != recomputed_selection:
        raise AdaptiveStoryCliError(
            "train selection differs from recomputed phase evidence"
        )
    selected_candidate_ids = recomputed_selection["selected_candidate_ids"]
    if (
        not isinstance(selected_candidate_ids, list)
        or len(selected_candidate_ids) != 10
        or train.get("next_phase_candidate_ids") != selected_candidate_ids
    ):
        raise AdaptiveStoryCliError(
            "train next-phase scope differs from recomputed selection"
        )
    _vehicles_for_candidate_ids(selected_candidate_ids)
    daily_metrics = _daily_train_metrics(recomputed_global, train_split)
    train_means = {
        candidate_id: float(daily_metrics[candidate_id]["mean_daily_net_r"])
        for candidate_id in selected_candidate_ids
    }
    return list(selected_candidate_ids), train_means


def _validate_validation_receipt_for_holdout(
    validation: Mapping[str, Any],
    *,
    scope: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_candidate_ids: Sequence[str],
    train_means: Mapping[str, float],
) -> tuple[list[str], dict[str, Any]]:
    """Rebuild the complete validation decision before any holdout read.

    A top-level receipt digest is not enough: an edited receipt could otherwise
    replace a non-empty winner set with an internally re-sealed empty portfolio
    and make HOLDOUT skip all truth.  Revalidating each sealed pair result,
    rebuilding the global combine, reapplying both screens, and rebuilding the
    fixed portfolio makes the survivor decision a deterministic consequence of
    the validation evidence and the independently supplied TRAIN receipt.
    """

    if validation.get("status") != "SEALED":
        raise AdaptiveStoryCliError("validation receipt status is not sealed")
    input_candidate_ids = validation.get("executed_candidate_ids")
    if not isinstance(input_candidate_ids, list) or len(input_candidate_ids) != 10:
        raise AdaptiveStoryCliError(
            "validation receipt input candidate scope is invalid"
        )
    if input_candidate_ids != list(expected_candidate_ids):
        raise AdaptiveStoryCliError(
            "validation candidate scope differs from sealed train selection"
        )
    _vehicles_for_candidate_ids(input_candidate_ids)
    if tuple(train_means) != tuple(input_candidate_ids) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in train_means.values()
    ):
        raise AdaptiveStoryCliError("recomputed train mean scope is invalid")
    validation_split, pair_results = _validate_phase_pair_evidence(
        validation,
        scope=scope,
        manifest=manifest,
        phase="VALIDATION",
        candidate_ids=input_candidate_ids,
    )

    provided_global = validation.get("global_result")
    if not isinstance(provided_global, Mapping):
        raise AdaptiveStoryCliError("validation receipt has no global result")
    _validate_global_result_contract(
        provided_global,
        pair_results=pair_results,
        split=validation_split,
        candidate_ids=input_candidate_ids,
    )
    recomputed_global = _combine_phase(
        pair_results,
        validation_split,
        input_candidate_ids,
    )
    if dict(provided_global) != recomputed_global:
        raise AdaptiveStoryCliError(
            "validation global result differs from recomputed pair evidence"
        )

    selection = validation.get("selection")
    if not isinstance(selection, Mapping):
        raise AdaptiveStoryCliError("validation receipt has no sealed selection")

    core_survivors = _validation_survivors(
        recomputed_global,
        input_candidate_ids,
    )
    survivors, recomputed_gate_rows = _apply_train_validation_same_sign_gate(
        recomputed_global,
        input_candidate_ids,
        core_survivors,
        train_means,
    )
    portfolio_spec = _new_portfolio_spec(survivors)
    recomputed_portfolio = _aggregate_fixed_portfolio(
        recomputed_global,
        validation_split,
        portfolio_spec,
    )
    provided_portfolio, provided_spec = _validate_fixed_portfolio_result(
        validation.get("fixed_portfolio"),
        expected_phase="VALIDATION",
        candidate_ids=survivors,
    )
    if provided_spec != portfolio_spec or provided_portfolio != recomputed_portfolio:
        raise AdaptiveStoryCliError(
            "validation fixed portfolio differs from recomputed survivors"
        )
    expected_selection = {
        "policy": VALIDATION_SCREEN_POLICY,
        "input_candidate_ids": list(input_candidate_ids),
        "core_validation_economic_survivor_ids": core_survivors,
        "same_sign_policy": (
            "CORE_VALIDATION_ELIGIBLE_AND_TRAIN_MEAN_DAILY_R_POSITIVE"
        ),
        "same_sign_gate_rows": recomputed_gate_rows,
        "selected_candidate_ids": survivors,
        "fixed_portfolio_candidate_ids": survivors,
        "fixed_portfolio_spec_sha256": portfolio_spec["portfolio_spec_sha256"],
        "all_final_same_sign_survivors_included": True,
        "subset_or_weight_search_performed": False,
        "selection_uses_holdout": False,
    }
    if dict(selection) != expected_selection:
        raise AdaptiveStoryCliError(
            "validation selection differs from recomputed screen decision"
        )
    if validation.get("next_phase_candidate_ids") != survivors:
        raise AdaptiveStoryCliError(
            "validation next-phase scope differs from recomputed survivors"
        )
    return survivors, portfolio_spec


def _new_receipt(contract: str, body: Mapping[str, Any]) -> dict[str, Any]:
    complete = {
        "contract": contract,
        "schema_version": 2,
        **dict(body),
        "price_only_cost_scope": dict(PRICE_ONLY_COST_SCOPE),
        "fully_loaded_net_economics": False,
        "live_net_claim_allowed": False,
        "integrity_evidence_not_external_authentication": True,
        **_AUTHORITY,
    }
    _validate_no_authority(complete)
    return {**complete, "receipt_sha256": _canonical_sha(complete)}


def _build_manifest(
    history_root: Path,
    scope: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    pairs = scope.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise AdaptiveStoryCliError("research scope has no pair universe")
    run_ids = scope.get("history_run_ids")
    if not isinstance(run_ids, list) or not run_ids:
        raise AdaptiveStoryCliError("research scope has no acquisition run universe")
    try:
        return build_historical_s5_manifest(
            history_root,
            pairs=tuple(str(item) for item in pairs),
            allowed_run_ids=tuple(str(item) for item in run_ids),
            scan_workers=workers,
        )
    except TypeError as error:
        if "allowed_run_ids" in str(error) or "scan_workers" in str(error):
            raise AdaptiveStoryCoreCapabilityError(
                "historical manifest run-scope/parallel API is required"
            ) from error
        raise


def _verify_manifest_binding(
    manifest: Mapping[str, Any], prior_receipt: Mapping[str, Any]
) -> None:
    prior_manifest = prior_receipt.get("manifest_receipt")
    if not isinstance(prior_manifest, Mapping):
        raise AdaptiveStoryCliError("prior receipt has no manifest binding")
    if _manifest_receipt(manifest) != dict(prior_manifest):
        raise AdaptiveStoryCliError("historical manifest changed after phase sealing")


def _frozen_manifest_from_receipt(
    receipt: Mapping[str, Any],
    *,
    history_root: Path,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _validate_frozen_manifest(
        receipt.get("frozen_manifest"),
        history_root=history_root,
        scope=scope,
    )
    _verify_manifest_binding(manifest, receipt)
    return manifest


def _run_train(
    args: argparse.Namespace,
    evaluator_seal: Mapping[str, Any],
) -> dict[str, Any]:
    scope = _scope_from_train_args(args)
    manifest = _build_manifest(args.history_root, scope, workers=args.workers)
    manifest = _validate_frozen_manifest(
        manifest,
        history_root=args.history_root,
        scope=scope,
    )
    train_split = _split_from_scope(scope, "TRAIN")
    holdout_split = _split_from_scope(scope, "HOLDOUT")
    _validate_manifest_coverage(
        manifest,
        from_utc=train_split.from_utc,
        to_utc=holdout_split.to_utc,
    )
    train_candidate_ids = tuple(item.candidate_id for item in _catalog_rows())
    artifacts, pair_results = _run_pair_phase(
        manifest=manifest,
        pairs=scope["pairs"],
        split=train_split,
        candidate_ids=train_candidate_ids,
        workers=args.workers,
    )
    global_result = _combine_phase(pair_results, train_split, train_candidate_ids)
    selection = _train_selection(global_result, pair_results, train_split)
    scope_body = dict(scope)
    scope = {**scope_body, "scope_sha256": _canonical_sha(scope_body)}
    return _new_receipt(
        TRAIN_CONTRACT,
        {
            "phase": "TRAIN",
            "status": "SEALED",
            "evaluator_dependency_seal": dict(evaluator_seal),
            "research_scope": scope,
            "manifest_receipt": _manifest_receipt(manifest),
            "frozen_manifest": manifest,
            "pair_execution_policy": PAIR_EXECUTION_POLICY,
            "executed_candidate_ids": list(train_candidate_ids),
            "pair_artifacts": artifacts,
            "global_result": global_result,
            "selection": selection,
            "next_phase_candidate_ids": selection["selected_candidate_ids"],
        },
    )


def _scope_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    scope = receipt.get("research_scope")
    if not isinstance(scope, Mapping):
        raise AdaptiveStoryCliError("prior receipt has no research scope")
    claimed = scope.get("scope_sha256")
    body = dict(scope)
    body.pop("scope_sha256", None)
    if not isinstance(claimed, str) or _canonical_sha(body) != claimed:
        raise AdaptiveStoryCliError("research scope digest mismatch")
    return dict(scope)


def _run_validation(
    args: argparse.Namespace,
    evaluator_seal: Mapping[str, Any],
) -> dict[str, Any]:
    train = _load_receipt(args.train_receipt, expected_contract=TRAIN_CONTRACT)
    _require_prior_evaluator_match(train, evaluator_seal)
    scope = _scope_from_receipt(train)
    manifest = _frozen_manifest_from_receipt(
        train,
        history_root=args.history_root,
        scope=scope,
    )
    candidate_ids, train_means = _validate_train_receipt_for_downstream(
        train,
        scope=scope,
        manifest=manifest,
    )
    split = _split_from_scope(scope, "VALIDATION")
    artifacts, pair_results = _run_pair_phase(
        manifest=manifest,
        pairs=scope["pairs"],
        split=split,
        candidate_ids=candidate_ids,
        workers=args.workers,
    )
    global_result = _combine_phase(pair_results, split, candidate_ids)
    core_survivors = _validation_survivors(global_result, candidate_ids)
    survivors, same_sign_gates = _apply_train_validation_same_sign_gate(
        global_result,
        candidate_ids,
        core_survivors,
        train_means,
    )
    portfolio_spec = _new_portfolio_spec(survivors)
    fixed_portfolio = _aggregate_fixed_portfolio(
        global_result,
        split,
        portfolio_spec,
    )
    return _new_receipt(
        VALIDATION_CONTRACT,
        {
            "phase": "VALIDATION",
            "status": "SEALED",
            "evaluator_dependency_seal": dict(evaluator_seal),
            "train_receipt_sha256": train["receipt_sha256"],
            "research_scope": scope,
            "manifest_receipt": _manifest_receipt(manifest),
            "frozen_manifest": manifest,
            "pair_execution_policy": PAIR_EXECUTION_POLICY,
            "executed_candidate_ids": list(candidate_ids),
            "pair_artifacts": artifacts,
            "global_result": global_result,
            "fixed_portfolio": fixed_portfolio,
            "selection": {
                "policy": VALIDATION_SCREEN_POLICY,
                "input_candidate_ids": list(candidate_ids),
                "core_validation_economic_survivor_ids": core_survivors,
                "same_sign_policy": (
                    "CORE_VALIDATION_ELIGIBLE_AND_TRAIN_MEAN_DAILY_R_POSITIVE"
                ),
                "same_sign_gate_rows": same_sign_gates,
                "selected_candidate_ids": survivors,
                "fixed_portfolio_candidate_ids": survivors,
                "fixed_portfolio_spec_sha256": portfolio_spec["portfolio_spec_sha256"],
                "all_final_same_sign_survivors_included": True,
                "subset_or_weight_search_performed": False,
                "selection_uses_holdout": False,
            },
            "next_phase_candidate_ids": survivors,
        },
    )


def _run_holdout(
    args: argparse.Namespace,
    evaluator_seal: Mapping[str, Any],
) -> dict[str, Any]:
    _require_distinct_holdout_paths(
        train_receipt=args.train_receipt,
        validation_receipt=args.validation_receipt,
        output=args.output,
    )
    validation = _load_receipt(
        args.validation_receipt,
        expected_contract=VALIDATION_CONTRACT,
    )
    train = _load_receipt(
        args.train_receipt,
        expected_contract=TRAIN_CONTRACT,
    )
    if validation.get("train_receipt_sha256") != train.get("receipt_sha256"):
        raise AdaptiveStoryCliError(
            "validation does not reference the supplied train receipt"
        )
    _require_prior_evaluator_match(validation, evaluator_seal)
    _require_prior_evaluator_match(train, evaluator_seal)
    scope = _scope_from_receipt(validation)
    train_scope = _scope_from_receipt(train)
    if train_scope != scope:
        raise AdaptiveStoryCliError(
            "validation research scope differs from supplied train receipt"
        )
    manifest = _frozen_manifest_from_receipt(
        validation,
        history_root=args.history_root,
        scope=scope,
    )
    train_manifest = _frozen_manifest_from_receipt(
        train,
        history_root=args.history_root,
        scope=train_scope,
    )
    if train_manifest != manifest:
        raise AdaptiveStoryCliError(
            "validation manifest differs from supplied train receipt"
        )
    train_candidate_ids, train_means = _validate_train_receipt_for_downstream(
        train,
        scope=train_scope,
        manifest=train_manifest,
    )
    split = _split_from_scope(scope, "HOLDOUT")
    candidate_ids, portfolio_spec = _validate_validation_receipt_for_holdout(
        validation,
        scope=scope,
        manifest=manifest,
        expected_candidate_ids=train_candidate_ids,
        train_means=train_means,
    )
    if candidate_ids:
        artifacts, pair_results = _run_pair_phase(
            manifest=manifest,
            pairs=scope["pairs"],
            split=split,
            candidate_ids=candidate_ids,
            workers=args.workers,
        )
        global_result = _combine_phase(pair_results, split, candidate_ids)
    else:
        artifacts = []
        global_result = _no_survivor_holdout_skip_result(
            split=split,
            manifest=manifest,
        )
    if _global_candidate_ids(global_result) != tuple(candidate_ids):
        raise AdaptiveStoryCliError("holdout exposed a validation non-winner")
    fixed_portfolio = _aggregate_fixed_portfolio(
        global_result,
        split,
        portfolio_spec,
    )
    return _new_receipt(
        HOLDOUT_CONTRACT,
        {
            "phase": "HOLDOUT",
            "status": "SEALED" if candidate_ids else "NO_VALIDATION_SURVIVORS",
            "evaluator_dependency_seal": dict(evaluator_seal),
            "validation_receipt_sha256": validation["receipt_sha256"],
            "train_receipt_sha256": validation.get("train_receipt_sha256"),
            "research_scope": scope,
            "manifest_receipt": _manifest_receipt(manifest),
            "frozen_manifest": manifest,
            "pair_execution_policy": PAIR_EXECUTION_POLICY,
            "executed_candidate_ids": list(candidate_ids),
            "pair_artifacts": artifacts,
            "global_result": global_result,
            "primary_result": fixed_portfolio,
            "selection": {
                "source": "SEALED_VALIDATION_SURVIVORS_ONLY",
                "selected_candidate_ids": list(candidate_ids),
                "fixed_portfolio_spec_sha256": portfolio_spec["portfolio_spec_sha256"],
                "fixed_portfolio_is_primary_result": True,
                "reselection_performed": False,
                "subset_or_weight_search_performed": False,
                "non_winner_results_calculated": False,
                "non_winner_results_published": False,
            },
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_seal = _evaluator_dependency_seal()
    if args.phase == "train":
        receipt = _run_train(args, evaluator_seal)
    elif args.phase == "validation":
        receipt = _run_validation(args, evaluator_seal)
    elif args.phase == "holdout":
        receipt = _run_holdout(args, evaluator_seal)
    else:
        raise AdaptiveStoryCliError("unknown phase")
    _require_current_evaluator_seal(evaluator_seal)
    if receipt.get("evaluator_dependency_seal") != evaluator_seal:
        raise AdaptiveStoryCliError("phase receipt lost evaluator dependency seal")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = run(args)
        _atomic_publish_json(
            args.output,
            receipt,
            evaluator_dependency_seal=receipt["evaluator_dependency_seal"],
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            f"adaptive story S5 phase failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    summary = {
        "phase": receipt["phase"],
        "status": receipt["status"],
        "executed_candidate_count": len(receipt["executed_candidate_ids"]),
        "next_phase_candidate_count": len(
            receipt.get("next_phase_candidate_ids") or ()
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "output": str(args.output),
        "order_authority": "NONE",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
