"""Fail-closed DOJO board for edge proof and a monthly 3x hypothesis.

The board deliberately keeps two questions separate:

* Has a worker or AI-supervision lane proved an edge prospectively?
* Is that proved edge's sealed 30-day distribution compatible with 3x?

It is a read-only research classifier.  It never sizes an order, grants live
permission, or treats a target-derived sizing exercise as evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


INPUT_CONTRACT = "QR_DOJO_GOAL_BOARD_INPUT_V1"
OUTPUT_CONTRACT = "QR_DOJO_GOAL_BOARD_V2"
TARGET_MULTIPLE = 3.0
CALENDAR_DAYS = 30
TRADING_DAYS = 22
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_LANES = 256
MAX_LIST_ITEMS = 32
MAX_TEXT_LENGTH = 256

# These are research admission thresholds, not production risk settings.  They
# represent a deliberately demanding definition of "3x compatible": a typical
# stressed month must reach the target, uncertainty must leave at least even
# odds of reaching it, and loss/tail-risk probabilities must remain bounded.
THRESHOLDS: dict[str, float | int] = {
    "stressed_median_multiple_min": TARGET_MULTIPLE,
    "probability_3x_lcb_min": 0.50,
    "probability_losing_month_max": 0.10,
    "probability_drawdown_20pct_max": 0.01,
    "probability_ruin_12m_max": 0.001,
    "normal_mtm_max_drawdown_max": 0.10,
    "stressed_mtm_max_drawdown_max": 0.15,
    "worker_min_active_days": 60,
    "ai_min_active_days": 90,
    "min_prospective_months": 3,
    "post_cost_edge_lcb_exclusive_min": 0.0,
    "absolute_margin_cap_fraction": 0.95,
}

LANE_TYPES = {"WORKER", "AI"}
DECLARED_STATUSES = {"HYPOTHESIS", "EDGE_PROVEN", "INVALID"}
DISTRIBUTION_METHODS = {
    "PROSPECTIVE_FORWARD",
    "WALK_FORWARD",
    "BOOTSTRAP_DIAGNOSTIC",
    "UNAVAILABLE",
}
EDGE_STATUSES = {"INVALID", "HYPOTHESIS", "EDGE_PROVEN"}
GOAL_STATUSES = {"3X_NOT_REACHABLE", "TAIL_ONLY", "GOAL_COMPATIBLE"}

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_CLUSTER_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INSTRUMENT_RE = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")

# No repository evidence artifact currently has a reviewed verifier capable of
# authenticating all forward-cohort, execution-cost, evaluator, and metric
# bindings required by this board.  Promotion therefore remains impossible by
# construction until a verifier is added here in reviewed code.  Merely naming
# a contract in caller JSON can never make that contract trusted.
TRUSTED_PROOF_VERIFIERS: dict[str, Any] = {}
UNVERIFIED_DEPENDENCE_CLUSTER = "UNVERIFIED_DEPENDENCE"


class DojoGoalBoardError(ValueError):
    """Raised when an input is malformed rather than merely unproved."""


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical JSON."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoGoalBoardError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def required_daily_return(*, days: int) -> float:
    """Daily compounded return required to turn 1.0 into 3.0."""

    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise DojoGoalBoardError("days must be a positive integer")
    return TARGET_MULTIPLE ** (1.0 / days) - 1.0


def _reject_json_constant(value: str) -> None:
    raise DojoGoalBoardError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DojoGoalBoardError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def load_goal_board_input(path: Path) -> dict[str, Any]:
    """Load bounded JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DojoGoalBoardError(f"cannot stat input: {exc}") from exc
    if size <= 0:
        raise DojoGoalBoardError("input must not be empty")
    if size > MAX_INPUT_BYTES:
        raise DojoGoalBoardError(
            f"input exceeds {MAX_INPUT_BYTES} byte research-artifact limit"
        )
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except DojoGoalBoardError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DojoGoalBoardError(f"cannot load input JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DojoGoalBoardError("input must be one JSON object")
    return value


def _object(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DojoGoalBoardError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise DojoGoalBoardError(
            f"{path} has invalid keys; missing={missing}, unknown={unknown}"
        )


def _required_optional_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    *,
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        raise DojoGoalBoardError(
            f"{path} has invalid keys; missing={missing}, unknown={unknown}"
        )


def _text(value: Any, *, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_LENGTH:
        raise DojoGoalBoardError(
            f"{path} must be a non-empty string of at most {MAX_TEXT_LENGTH} chars"
        )
    if pattern is not None and pattern.fullmatch(value) is None:
        raise DojoGoalBoardError(f"{path} has a non-canonical value")
    return value


def _boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise DojoGoalBoardError(f"{path} must be a boolean")
    return value


def _number(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoGoalBoardError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DojoGoalBoardError(f"{path} must be a finite number")
    if minimum is not None:
        too_small = result <= minimum if strict_minimum else result < minimum
        if too_small:
            relation = ">" if strict_minimum else ">="
            raise DojoGoalBoardError(f"{path} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise DojoGoalBoardError(f"{path} must be <= {maximum}")
    return result


def _integer(value: Any, *, path: str, minimum: int = 0, maximum: int = 100_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DojoGoalBoardError(f"{path} must be an integer")
    if value < minimum or value > maximum:
        raise DojoGoalBoardError(f"{path} must be between {minimum} and {maximum}")
    return value


def _string_list(value: Any, *, path: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise DojoGoalBoardError(f"{path} must be an array")
    if len(value) > MAX_LIST_ITEMS or (not allow_empty and not value):
        raise DojoGoalBoardError(
            f"{path} must contain {'1..' if not allow_empty else '0..'}"
            f"{MAX_LIST_ITEMS} values"
        )
    result = [_text(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise DojoGoalBoardError(f"{path} must not contain duplicates")
    return result


def _validate_lane(value: Any, *, index: int) -> dict[str, Any]:
    path = f"lanes[{index}]"
    lane = _object(value, path=path)
    _required_optional_keys(
        lane,
        {
            "lane_id",
            "lane_type",
            "status",
            "parent_lane_ids",
            "provenance",
            "risk",
            "margin",
            "correlation_cluster",
            "distribution_30d",
            "sizing",
        },
        {"dependence"},
        path=path,
    )
    lane_id = _text(lane["lane_id"], path=f"{path}.lane_id", pattern=_IDENTIFIER_RE)
    lane_type = _text(lane["lane_type"], path=f"{path}.lane_type")
    if lane_type not in LANE_TYPES:
        raise DojoGoalBoardError(f"{path}.lane_type must be WORKER or AI")
    status = _text(lane["status"], path=f"{path}.status")
    if status not in DECLARED_STATUSES:
        raise DojoGoalBoardError(f"{path}.status is unsupported")
    parent_lane_ids = _string_list(
        lane["parent_lane_ids"],
        path=f"{path}.parent_lane_ids",
        allow_empty=lane_type == "WORKER",
    )
    for parent_id in parent_lane_ids:
        if _IDENTIFIER_RE.fullmatch(parent_id) is None:
            raise DojoGoalBoardError(f"{path}.parent_lane_ids is non-canonical")
        if parent_id == lane_id:
            raise DojoGoalBoardError(f"{path} cannot parent itself")
    if lane_type == "WORKER" and parent_lane_ids:
        raise DojoGoalBoardError(f"{path} WORKER must not declare parents")

    provenance = _object(lane["provenance"], path=f"{path}.provenance")
    _required_optional_keys(
        provenance,
        {
            "valid",
            "prospective",
            "lookahead_free",
            "content_sha256",
            "invalid_reasons",
            "parent_digests",
        },
        {"evidence_path", "evidence_contract"},
        path=f"{path}.provenance",
    )
    valid = _boolean(provenance["valid"], path=f"{path}.provenance.valid")
    prospective = _boolean(
        provenance["prospective"], path=f"{path}.provenance.prospective"
    )
    lookahead_free = _boolean(
        provenance["lookahead_free"], path=f"{path}.provenance.lookahead_free"
    )
    content_sha256 = _text(
        provenance["content_sha256"],
        path=f"{path}.provenance.content_sha256",
        pattern=_SHA256_RE,
    )
    evidence_path_raw = provenance.get("evidence_path")
    evidence_contract_raw = provenance.get("evidence_contract")
    if (evidence_path_raw is None) != (evidence_contract_raw is None):
        raise DojoGoalBoardError(
            f"{path}.provenance evidence_path and evidence_contract must both "
            "be present or both be null"
        )
    evidence_path = (
        None
        if evidence_path_raw is None
        else _text(evidence_path_raw, path=f"{path}.provenance.evidence_path")
    )
    evidence_contract = (
        None
        if evidence_contract_raw is None
        else _text(
            evidence_contract_raw,
            path=f"{path}.provenance.evidence_contract",
            pattern=_IDENTIFIER_RE,
        )
    )
    invalid_reasons = _string_list(
        provenance["invalid_reasons"],
        path=f"{path}.provenance.invalid_reasons",
    )
    parent_digests = _object(
        provenance["parent_digests"], path=f"{path}.provenance.parent_digests"
    )
    if set(parent_digests) != set(parent_lane_ids):
        raise DojoGoalBoardError(
            f"{path}.provenance.parent_digests must bind every parent exactly"
        )
    normalized_parent_digests: dict[str, str] = {}
    for parent_id, digest in parent_digests.items():
        normalized_parent_digests[parent_id] = _text(
            digest,
            path=f"{path}.provenance.parent_digests.{parent_id}",
            pattern=_SHA256_RE,
        )
    if valid and invalid_reasons:
        raise DojoGoalBoardError(
            f"{path}.provenance.valid=true requires no invalid_reasons"
        )
    if not valid and not invalid_reasons:
        raise DojoGoalBoardError(
            f"{path}.provenance.valid=false requires invalid_reasons"
        )
    if status == "INVALID" and valid:
        raise DojoGoalBoardError(f"{path}.status INVALID requires invalid provenance")
    if status != "INVALID" and not valid:
        raise DojoGoalBoardError(f"{path}.invalid provenance requires status INVALID")

    risk = _object(lane["risk"], path=f"{path}.risk")
    _exact_keys(
        risk,
        {
            "mark_to_market",
            "bounded",
            "normal_mtm_max_drawdown_fraction",
            "stressed_mtm_max_drawdown_fraction",
        },
        path=f"{path}.risk",
    )
    mark_to_market = _boolean(
        risk["mark_to_market"], path=f"{path}.risk.mark_to_market"
    )
    bounded = _boolean(risk["bounded"], path=f"{path}.risk.bounded")
    normal_drawdown_raw = risk["normal_mtm_max_drawdown_fraction"]
    normal_drawdown = (
        None
        if normal_drawdown_raw is None
        else _number(
            normal_drawdown_raw,
            path=f"{path}.risk.normal_mtm_max_drawdown_fraction",
            minimum=0.0,
            maximum=1.0,
        )
    )
    stressed_drawdown_raw = risk["stressed_mtm_max_drawdown_fraction"]
    stressed_drawdown = (
        None
        if stressed_drawdown_raw is None
        else _number(
            stressed_drawdown_raw,
            path=f"{path}.risk.stressed_mtm_max_drawdown_fraction",
            minimum=0.0,
            maximum=1.0,
        )
    )
    if (
        stressed_drawdown is not None
        and normal_drawdown is not None
        and stressed_drawdown < normal_drawdown
    ):
        raise DojoGoalBoardError(
            f"{path}.risk stressed drawdown cannot be below normal drawdown"
        )

    margin = _object(lane["margin"], path=f"{path}.margin")
    _exact_keys(
        margin,
        {"peak_usage_fraction", "cap_fraction"},
        path=f"{path}.margin",
    )
    peak_usage_raw = margin["peak_usage_fraction"]
    peak_usage = (
        None
        if peak_usage_raw is None
        else _number(
            peak_usage_raw,
            path=f"{path}.margin.peak_usage_fraction",
            minimum=0.0,
            maximum=1.0,
        )
    )
    cap_fraction = _number(
        margin["cap_fraction"],
        path=f"{path}.margin.cap_fraction",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )

    declared_cluster = _text(
        lane["correlation_cluster"],
        path=f"{path}.correlation_cluster",
        pattern=_CLUSTER_RE,
    )

    dependence_raw = lane.get("dependence")
    dependence: dict[str, str] | None = None
    if dependence_raw is not None:
        dependence_object = _object(dependence_raw, path=f"{path}.dependence")
        _exact_keys(
            dependence_object,
            {"instrument", "strategy_family", "cohort_id"},
            path=f"{path}.dependence",
        )
        dependence = {
            "instrument": _text(
                dependence_object["instrument"],
                path=f"{path}.dependence.instrument",
                pattern=_INSTRUMENT_RE,
            ),
            "strategy_family": _text(
                dependence_object["strategy_family"],
                path=f"{path}.dependence.strategy_family",
                pattern=_IDENTIFIER_RE,
            ),
            "cohort_id": _text(
                dependence_object["cohort_id"],
                path=f"{path}.dependence.cohort_id",
                pattern=_IDENTIFIER_RE,
            ),
        }

    distribution = _object(lane["distribution_30d"], path=f"{path}.distribution_30d")
    _required_optional_keys(
        distribution,
        {
            "method",
            "sample_months",
            "active_days",
            "stressed_median_multiple",
            "probability_3x_lcb",
            "probability_losing_month",
            "probability_drawdown_20pct",
            "probability_ruin_12m",
        },
        {"post_cost_edge_lcb"},
        path=f"{path}.distribution_30d",
    )
    method_raw = distribution["method"]
    method = (
        None
        if method_raw is None
        else _text(method_raw, path=f"{path}.distribution_30d.method")
    )
    if method is not None and method not in DISTRIBUTION_METHODS:
        raise DojoGoalBoardError(f"{path}.distribution_30d.method is unsupported")
    sample_months = (
        None
        if distribution["sample_months"] is None
        else _integer(
            distribution["sample_months"],
            path=f"{path}.distribution_30d.sample_months",
            maximum=1_200,
        )
    )
    active_days = (
        None
        if distribution["active_days"] is None
        else _integer(
            distribution["active_days"],
            path=f"{path}.distribution_30d.active_days",
            maximum=100_000,
        )
    )
    stressed_median = (
        None
        if distribution["stressed_median_multiple"] is None
        else _number(
            distribution["stressed_median_multiple"],
            path=f"{path}.distribution_30d.stressed_median_multiple",
            minimum=0.0,
            strict_minimum=True,
        )
    )
    probabilities = {
        key: (
            None
            if distribution[key] is None
            else _number(
                distribution[key],
                path=f"{path}.distribution_30d.{key}",
                minimum=0.0,
                maximum=1.0,
            )
        )
        for key in (
            "probability_3x_lcb",
            "probability_losing_month",
            "probability_drawdown_20pct",
            "probability_ruin_12m",
        )
    }
    post_cost_edge_lcb_raw = distribution.get("post_cost_edge_lcb")
    post_cost_edge_lcb = (
        None
        if post_cost_edge_lcb_raw is None
        else _number(
            post_cost_edge_lcb_raw,
            path=f"{path}.distribution_30d.post_cost_edge_lcb",
            minimum=-1_000_000.0,
            maximum=1_000_000.0,
        )
    )

    sizing = _object(lane["sizing"], path=f"{path}.sizing")
    _exact_keys(
        sizing,
        {"observed_at_declared_size", "reverse_engineered_from_goal"},
        path=f"{path}.sizing",
    )
    observed_at_declared_size = _boolean(
        sizing["observed_at_declared_size"],
        path=f"{path}.sizing.observed_at_declared_size",
    )
    reverse_engineered = _boolean(
        sizing["reverse_engineered_from_goal"],
        path=f"{path}.sizing.reverse_engineered_from_goal",
    )

    return {
        "lane_id": lane_id,
        "lane_type": lane_type,
        "status": status,
        "parent_lane_ids": parent_lane_ids,
        "provenance": {
            "valid": valid,
            "prospective": prospective,
            "lookahead_free": lookahead_free,
            "content_sha256": content_sha256,
            "evidence_path": evidence_path,
            "evidence_contract": evidence_contract,
            "invalid_reasons": invalid_reasons,
            "parent_digests": normalized_parent_digests,
        },
        "risk": {
            "mark_to_market": mark_to_market,
            "bounded": bounded,
            "normal_mtm_max_drawdown_fraction": normal_drawdown,
            "stressed_mtm_max_drawdown_fraction": stressed_drawdown,
        },
        "margin": {
            "peak_usage_fraction": peak_usage,
            "cap_fraction": cap_fraction,
        },
        "declared_correlation_cluster": declared_cluster,
        "dependence": dependence,
        "distribution_30d": {
            "method": method,
            "sample_months": sample_months,
            "active_days": active_days,
            "stressed_median_multiple": stressed_median,
            "post_cost_edge_lcb": post_cost_edge_lcb,
            **probabilities,
        },
        "sizing": {
            "observed_at_declared_size": observed_at_declared_size,
            "reverse_engineered_from_goal": reverse_engineered,
        },
    }


def validate_goal_board_input(value: Any) -> dict[str, Any]:
    """Validate and normalize the complete input contract."""

    root = _object(value, path="input")
    _exact_keys(root, {"contract", "lanes"}, path="input")
    if root["contract"] != INPUT_CONTRACT:
        raise DojoGoalBoardError(f"input.contract must be {INPUT_CONTRACT}")
    raw_lanes = root["lanes"]
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise DojoGoalBoardError("input.lanes must be a non-empty array")
    if len(raw_lanes) > MAX_LANES:
        raise DojoGoalBoardError(f"input.lanes exceeds {MAX_LANES} lanes")
    lanes = [_validate_lane(item, index=index) for index, item in enumerate(raw_lanes)]
    lane_ids = [lane["lane_id"] for lane in lanes]
    if len(set(lane_ids)) != len(lane_ids):
        raise DojoGoalBoardError("input.lanes must have unique lane_id values")
    return {"contract": INPUT_CONTRACT, "lanes": lanes}


def _untrusted_evidence(
    lane: Mapping[str, Any], status: str, blocker: str, **extra: Any
) -> dict[str, Any]:
    return {
        "trusted": False,
        "status": status,
        "blocker": blocker,
        "artifact_path": lane["provenance"]["evidence_path"],
        "expected_sha256": lane["provenance"]["content_sha256"],
        "actual_sha256": None,
        "contract": lane["provenance"]["evidence_contract"],
        "verified_dependence": None,
        **extra,
    }


def _verify_lane_evidence(
    lane: Mapping[str, Any], *, project_root: Path | None
) -> dict[str, Any]:
    """Authenticate evidence bytes through reviewed code or fail closed.

    A hash supplied beside an artifact by the same JSON author is only an
    integrity hint.  Promotion additionally requires a reviewed contract
    verifier in ``TRUSTED_PROOF_VERIFIERS``.  That registry is intentionally
    empty until an authoritative forward-proof schema exists.
    """

    evidence_path = lane["provenance"]["evidence_path"]
    if evidence_path is None:
        return _untrusted_evidence(
            lane,
            "MISSING",
            "TRUSTED_PROOF_ARTIFACT_REQUIRED",
        )
    if project_root is None:
        return _untrusted_evidence(
            lane,
            "PROJECT_ROOT_UNAVAILABLE",
            "TRUSTED_PROOF_ARTIFACT_REQUIRED",
        )
    pure_path = PurePosixPath(evidence_path)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or "\\" in evidence_path
        or str(pure_path) != evidence_path
    ):
        return _untrusted_evidence(
            lane,
            "UNSAFE_PATH",
            "EVIDENCE_PATH_NOT_PROJECT_RELATIVE",
        )
    try:
        root = project_root.resolve(strict=True)
        resolved = (root / Path(*pure_path.parts)).resolve(strict=True)
        resolved.relative_to(root)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("evidence path is not a regular file")
            if metadata.st_size <= 0 or metadata.st_size > MAX_EVIDENCE_BYTES:
                return _untrusted_evidence(
                    lane,
                    "SIZE_INVALID",
                    "EVIDENCE_ARTIFACT_SIZE_INVALID",
                    size_bytes=metadata.st_size,
                )
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("evidence artifact changed while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise OSError("evidence artifact grew while reading")
            artifact_bytes = b"".join(chunks)
        finally:
            os.close(descriptor)
    except (OSError, RuntimeError, ValueError) as exc:
        return _untrusted_evidence(
            lane,
            "UNREADABLE",
            "EVIDENCE_ARTIFACT_UNREADABLE",
            error=str(exc)[:MAX_TEXT_LENGTH],
        )
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != lane["provenance"]["content_sha256"]:
        return _untrusted_evidence(
            lane,
            "DIGEST_MISMATCH",
            "EVIDENCE_ARTIFACT_DIGEST_MISMATCH",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )
    try:
        artifact = json.loads(
            artifact_bytes.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, DojoGoalBoardError) as exc:
        return _untrusted_evidence(
            lane,
            "MALFORMED",
            "EVIDENCE_ARTIFACT_MALFORMED",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
            error=str(exc)[:MAX_TEXT_LENGTH],
        )
    declared_contract = lane["provenance"]["evidence_contract"]
    if not isinstance(artifact, dict) or artifact.get("contract") != declared_contract:
        return _untrusted_evidence(
            lane,
            "CONTRACT_MISMATCH",
            "EVIDENCE_ARTIFACT_CONTRACT_MISMATCH",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )
    verifier = TRUSTED_PROOF_VERIFIERS.get(declared_contract)
    if verifier is None:
        return _untrusted_evidence(
            lane,
            "UNTRUSTED_CONTRACT",
            "TRUSTED_PROOF_CONTRACT_UNAVAILABLE",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )

    # A future verifier must return metrics and dependence reconstructed from
    # the authenticated artifact, never echo caller-provided fields.  Keep the
    # boundary strict even though no verifier is currently registered.
    try:
        verified = verifier(artifact, lane)
    except Exception as exc:
        return _untrusted_evidence(
            lane,
            "VERIFIER_REJECTED",
            "TRUSTED_PROOF_VERIFIER_REJECTED",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
            error=str(exc)[:MAX_TEXT_LENGTH],
        )
    if not isinstance(verified, dict) or verified.get("trusted") is not True:
        return _untrusted_evidence(
            lane,
            "VERIFIER_REJECTED",
            "TRUSTED_PROOF_VERIFIER_REJECTED",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )
    verified_metrics = verified.get("metrics")
    verified_dependence = verified.get("dependence")
    if (
        not isinstance(verified_metrics, dict)
        or verified_metrics != lane["distribution_30d"]
    ):
        return _untrusted_evidence(
            lane,
            "VERIFIED_METRICS_MISMATCH",
            "TRUSTED_PROOF_METRICS_MISMATCH",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )
    if (
        not isinstance(verified_dependence, dict)
        or verified_dependence != lane["dependence"]
    ):
        return _untrusted_evidence(
            lane,
            "VERIFIED_DEPENDENCE_MISMATCH",
            "TRUSTED_PROOF_DEPENDENCE_MISMATCH",
            actual_sha256=actual_sha256,
            size_bytes=len(artifact_bytes),
        )
    return {
        "trusted": True,
        "status": "VERIFIED",
        "blocker": None,
        "artifact_path": evidence_path,
        "expected_sha256": lane["provenance"]["content_sha256"],
        "actual_sha256": actual_sha256,
        "contract": declared_contract,
        "size_bytes": len(artifact_bytes),
        "verified_dependence": verified_dependence,
        "verified_metrics": verified_metrics,
    }


def _derived_correlation_cluster(evidence: Mapping[str, Any]) -> str:
    dependence = evidence.get("verified_dependence")
    if evidence.get("trusted") is not True or not isinstance(dependence, dict):
        return UNVERIFIED_DEPENDENCE_CLUSTER
    # Only verifier-derived dependence may establish independence.  Hashing
    # avoids letting caller-controlled labels become portfolio group names.
    return f"VERIFIED_{canonical_sha256(dependence)[:24]}"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _base_lane_evaluation(
    lane: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    edge_blockers: list[str] = []
    provenance = lane["provenance"]
    distribution = lane["distribution_30d"]
    risk = lane["risk"]
    margin = lane["margin"]
    sizing = lane["sizing"]

    if lane["status"] == "INVALID" or not provenance["valid"]:
        edge_status = "INVALID"
        edge_blockers.extend(["PROVENANCE_INVALID", *provenance["invalid_reasons"]])
    else:
        if lane["status"] != "EDGE_PROVEN":
            edge_blockers.append("DECLARED_HYPOTHESIS")
        if not provenance["prospective"]:
            edge_blockers.append("PROSPECTIVE_PROVENANCE_REQUIRED")
        if not provenance["lookahead_free"]:
            edge_blockers.append("LOOKAHEAD_FREE_PROVENANCE_REQUIRED")
        if distribution["method"] is None:
            edge_blockers.append("DISTRIBUTION_METHOD_UNKNOWN")
        elif distribution["method"] != "PROSPECTIVE_FORWARD":
            edge_blockers.append("PROSPECTIVE_FORWARD_DISTRIBUTION_REQUIRED")
        if distribution["sample_months"] is None:
            edge_blockers.append("PROSPECTIVE_MONTHS_UNKNOWN")
        elif distribution["sample_months"] < THRESHOLDS["min_prospective_months"]:
            edge_blockers.append("PROSPECTIVE_MONTH_FLOOR_NOT_MET")
        min_active_days = (
            THRESHOLDS["worker_min_active_days"]
            if lane["lane_type"] == "WORKER"
            else THRESHOLDS["ai_min_active_days"]
        )
        if distribution["active_days"] is None:
            edge_blockers.append("ACTIVE_DAYS_UNKNOWN")
        elif distribution["active_days"] < min_active_days:
            edge_blockers.append("ACTIVE_DAY_FLOOR_NOT_MET")
        if evidence.get("trusted") is not True:
            edge_blockers.append(
                str(evidence.get("blocker") or "TRUSTED_PROOF_ARTIFACT_REQUIRED")
            )
        if (
            distribution["post_cost_edge_lcb"] is None
            or distribution["post_cost_edge_lcb"]
            <= THRESHOLDS["post_cost_edge_lcb_exclusive_min"]
        ):
            edge_blockers.append("POST_COST_EDGE_LOWER_BOUND_NOT_POSITIVE")
        if not risk["mark_to_market"]:
            edge_blockers.append("MARK_TO_MARKET_RISK_REQUIRED")
        if not risk["bounded"]:
            edge_blockers.append("BOUNDED_RISK_REQUIRED")
        normal_drawdown = risk["normal_mtm_max_drawdown_fraction"]
        if normal_drawdown is None:
            edge_blockers.append("NORMAL_MTM_DRAWDOWN_UNKNOWN")
        elif normal_drawdown > THRESHOLDS["normal_mtm_max_drawdown_max"]:
            edge_blockers.append("NORMAL_MTM_DRAWDOWN_TOO_HIGH")
        stressed_drawdown = risk["stressed_mtm_max_drawdown_fraction"]
        if stressed_drawdown is None:
            edge_blockers.append("STRESSED_MTM_DRAWDOWN_UNKNOWN")
        elif stressed_drawdown > THRESHOLDS["stressed_mtm_max_drawdown_max"]:
            edge_blockers.append("STRESSED_MTM_DRAWDOWN_TOO_HIGH")
        if margin["cap_fraction"] > THRESHOLDS["absolute_margin_cap_fraction"]:
            edge_blockers.append("MARGIN_CAP_EXCEEDS_ABSOLUTE_BOUND")
        if margin["peak_usage_fraction"] is None:
            edge_blockers.append("MARGIN_PEAK_USAGE_UNKNOWN")
        elif margin["peak_usage_fraction"] > margin["cap_fraction"]:
            edge_blockers.append("MARGIN_CAP_BREACHED")
        if not sizing["observed_at_declared_size"]:
            edge_blockers.append("DECLARED_SIZE_NOT_OBSERVED")
        edge_status = (
            "EDGE_PROVEN"
            if lane["status"] == "EDGE_PROVEN" and not edge_blockers
            else "HYPOTHESIS"
        )

    return {
        "lane_id": lane["lane_id"],
        "lane_type": lane["lane_type"],
        "declared_status": lane["status"],
        "edge_status": edge_status,
        "goal_status": "3X_NOT_REACHABLE",
        "correlation_cluster": _derived_correlation_cluster(evidence),
        "declared_correlation_cluster": lane["declared_correlation_cluster"],
        "dependence": lane["dependence"],
        "parent_lane_ids": list(lane["parent_lane_ids"]),
        "edge_blockers": _dedupe(edge_blockers),
        "goal_blockers": [],
        "distribution_30d": dict(distribution),
        "reverse_engineered_from_goal": sizing["reverse_engineered_from_goal"],
        "evidence_verification": dict(evidence),
    }


def _apply_ai_parent_gate(
    lane: Mapping[str, Any],
    evaluation: dict[str, Any],
    *,
    lanes_by_id: Mapping[str, Mapping[str, Any]],
    evaluations_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    if lane["lane_type"] != "AI":
        return
    invalid_parent = False
    for parent_id in lane["parent_lane_ids"]:
        parent = lanes_by_id.get(parent_id)
        if parent is None:
            evaluation["edge_blockers"].append(f"PARENT_NOT_FOUND:{parent_id}")
            invalid_parent = True
            continue
        if parent["lane_type"] != "WORKER":
            evaluation["edge_blockers"].append(f"PARENT_LANE_NOT_WORKER:{parent_id}")
            invalid_parent = True
        if (
            lane["provenance"]["parent_digests"].get(parent_id)
            != parent["provenance"]["content_sha256"]
        ):
            evaluation["edge_blockers"].append(
                f"PARENT_PROVENANCE_DIGEST_MISMATCH:{parent_id}"
            )
            invalid_parent = True
        if (
            lane["declared_correlation_cluster"]
            != parent["declared_correlation_cluster"]
        ):
            evaluation["edge_blockers"].append(
                f"PARENT_CORRELATION_CLUSTER_MISMATCH:{parent_id}"
            )
            invalid_parent = True
        if lane["dependence"] != parent["dependence"]:
            evaluation["edge_blockers"].append(
                f"PARENT_DEPENDENCE_MISMATCH:{parent_id}"
            )
            invalid_parent = True
        parent_evaluation = evaluations_by_id.get(parent_id)
        if parent_evaluation is None:
            continue
        if parent_evaluation["edge_status"] == "INVALID":
            evaluation["edge_blockers"].append(f"PARENT_INVALID:{parent_id}")
            invalid_parent = True
        elif parent_evaluation["edge_status"] != "EDGE_PROVEN":
            evaluation["edge_blockers"].append(f"PARENT_EDGE_NOT_PROVEN:{parent_id}")
    evaluation["edge_blockers"] = _dedupe(evaluation["edge_blockers"])
    if invalid_parent:
        evaluation["edge_status"] = "INVALID"
    elif evaluation["edge_status"] != "INVALID" and any(
        blocker.startswith("PARENT_EDGE_NOT_PROVEN:")
        for blocker in evaluation["edge_blockers"]
    ):
        evaluation["edge_status"] = "HYPOTHESIS"


def _apply_goal_gate(lane: Mapping[str, Any], evaluation: dict[str, Any]) -> None:
    distribution = lane["distribution_30d"]
    goal_blockers: list[str] = []
    median = distribution["stressed_median_multiple"]
    if median is None:
        goal_blockers.append("STRESSED_MEDIAN_MULTIPLE_UNKNOWN")
    elif median < THRESHOLDS["stressed_median_multiple_min"]:
        goal_blockers.append("STRESSED_MEDIAN_BELOW_3X")
    probability_3x = distribution["probability_3x_lcb"]
    if probability_3x is None:
        goal_blockers.append("PROBABILITY_3X_LCB_UNKNOWN")
    elif probability_3x < THRESHOLDS["probability_3x_lcb_min"]:
        goal_blockers.append("PROBABILITY_3X_LCB_BELOW_FLOOR")
    losing_probability = distribution["probability_losing_month"]
    if losing_probability is None:
        goal_blockers.append("LOSING_MONTH_PROBABILITY_UNKNOWN")
    elif losing_probability > THRESHOLDS["probability_losing_month_max"]:
        goal_blockers.append("LOSING_MONTH_PROBABILITY_TOO_HIGH")
    drawdown_probability = distribution["probability_drawdown_20pct"]
    if drawdown_probability is None:
        goal_blockers.append("DRAWDOWN_20PCT_PROBABILITY_UNKNOWN")
    elif drawdown_probability > THRESHOLDS["probability_drawdown_20pct_max"]:
        goal_blockers.append("DRAWDOWN_20PCT_PROBABILITY_TOO_HIGH")
    ruin_probability = distribution["probability_ruin_12m"]
    if ruin_probability is None:
        goal_blockers.append("RUIN_12M_PROBABILITY_UNKNOWN")
    elif ruin_probability > THRESHOLDS["probability_ruin_12m_max"]:
        goal_blockers.append("RUIN_12M_PROBABILITY_TOO_HIGH")
    if lane["sizing"]["reverse_engineered_from_goal"]:
        goal_blockers.append("TARGET_REVERSE_ENGINEERED_SIZING_NOT_EVIDENCE")
    if evaluation["edge_status"] != "EDGE_PROVEN":
        goal_blockers.insert(0, "EDGE_NOT_PROVEN")
        goal_status = "3X_NOT_REACHABLE"
    elif not goal_blockers:
        goal_status = "GOAL_COMPATIBLE"
    elif probability_3x is not None and probability_3x > 0.0:
        goal_status = "TAIL_ONLY"
    else:
        goal_status = "3X_NOT_REACHABLE"
    evaluation["goal_status"] = goal_status
    evaluation["goal_blockers"] = _dedupe(goal_blockers)


def _representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    edge_rank = {"INVALID": 0, "HYPOTHESIS": 1, "EDGE_PROVEN": 2}
    goal_rank = {"3X_NOT_REACHABLE": 0, "TAIL_ONLY": 1, "GOAL_COMPATIBLE": 2}

    def high_value(value: float | None) -> float:
        return float("-inf") if value is None else value

    def low_value(value: float | None) -> float:
        return float("inf") if value is None else value

    return sorted(
        group,
        key=lambda row: (
            -edge_rank[row["edge_status"]],
            -goal_rank[row["goal_status"]],
            -high_value(row["distribution_30d"]["stressed_median_multiple"]),
            -high_value(row["distribution_30d"]["probability_3x_lcb"]),
            low_value(row["distribution_30d"]["probability_losing_month"]),
            row["lane_id"],
        ),
    )[0]


def build_goal_board(value: Any, *, project_root: Path | None = None) -> dict[str, Any]:
    """Build the sealed read-only board from a strict worker/AI input."""

    normalized = validate_goal_board_input(value)
    lanes = normalized["lanes"]
    lanes_by_id = {lane["lane_id"]: lane for lane in lanes}
    evidence_by_id = {
        lane["lane_id"]: _verify_lane_evidence(lane, project_root=project_root)
        for lane in lanes
    }
    evaluations_by_id: dict[str, dict[str, Any]] = {}

    # Workers are evaluated first because an AI supervision lane is invalid or
    # hypothetical unless every exact parent worker is already trustworthy.
    for lane in sorted(
        lanes, key=lambda row: (row["lane_type"] == "AI", row["lane_id"])
    ):
        evaluation = _base_lane_evaluation(lane, evidence_by_id[lane["lane_id"]])
        if lane["lane_type"] == "AI":
            _apply_ai_parent_gate(
                lane,
                evaluation,
                lanes_by_id=lanes_by_id,
                evaluations_by_id=evaluations_by_id,
            )
        _apply_goal_gate(lane, evaluation)
        evaluations_by_id[lane["lane_id"]] = evaluation

    evaluations = [evaluations_by_id[lane["lane_id"]] for lane in lanes]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        grouped.setdefault(evaluation["correlation_cluster"], []).append(evaluation)

    cluster_rows: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    for cluster in sorted(grouped):
        group = grouped[cluster]
        representative = _representative(group)
        representatives.append(representative)
        suppressed = sorted(
            row["lane_id"]
            for row in group
            if row["lane_id"] != representative["lane_id"]
        )
        cluster_rows.append(
            {
                "correlation_cluster": cluster,
                "lane_ids": sorted(row["lane_id"] for row in group),
                "representative_lane_id": representative["lane_id"],
                "suppressed_correlated_lane_ids": suppressed,
                "edge_status": representative["edge_status"],
                "goal_status": representative["goal_status"],
                "independence_validated": all(
                    row["evidence_verification"]["trusted"] is True for row in group
                ),
                "double_count_prevented": bool(suppressed),
                "return_summation_allowed": False,
            }
        )

    trusted_representatives = [
        row
        for row in representatives
        if row["evidence_verification"]["trusted"] is True
        and row["correlation_cluster"] != UNVERIFIED_DEPENDENCE_CLUSTER
    ]
    trusted_cluster_ids = {
        row["correlation_cluster"] for row in trusted_representatives
    }

    if any(row["edge_status"] == "EDGE_PROVEN" for row in representatives):
        edge_status = "EDGE_PROVEN"
    elif any(row["edge_status"] == "HYPOTHESIS" for row in representatives):
        edge_status = "HYPOTHESIS"
    else:
        edge_status = "INVALID"
    if any(row["goal_status"] == "GOAL_COMPATIBLE" for row in representatives):
        goal_status = "GOAL_COMPATIBLE"
    elif any(row["goal_status"] == "TAIL_ONLY" for row in representatives):
        goal_status = "TAIL_ONLY"
    else:
        goal_status = "3X_NOT_REACHABLE"

    body: dict[str, Any] = {
        "contract": OUTPUT_CONTRACT,
        "schema_version": 2,
        "input_sha256": canonical_sha256(normalized),
        "goal": {
            "target_multiple": TARGET_MULTIPLE,
            "calendar_days": CALENDAR_DAYS,
            "trading_days": TRADING_DAYS,
            "required_calendar_daily_return_fraction": required_daily_return(
                days=CALENDAR_DAYS
            ),
            "required_trading_daily_return_fraction": required_daily_return(
                days=TRADING_DAYS
            ),
        },
        "thresholds": dict(THRESHOLDS),
        "proof_admission": {
            "trusted_proof_contracts": sorted(TRUSTED_PROOF_VERIFIERS),
            "promotion_possible": bool(TRUSTED_PROOF_VERIFIERS),
            "self_asserted_json_can_promote": False,
        },
        "edge_status": edge_status,
        "goal_status": goal_status,
        "lane_evaluations": evaluations,
        "correlation_clusters": cluster_rows,
        "portfolio": {
            "edge_status": edge_status,
            "goal_status": goal_status,
            "aggregation_policy": (
                "BEST_SINGLE_TRUSTED_LANE_PER_VERIFIED_DEPENDENCE_CLUSTER_"
                "NO_SUMMATION"
                if trusted_representatives
                else "NO_TRUSTED_INDEPENDENCE_NO_RETURN_SUMMATION"
            ),
            "included_lane_ids": [row["lane_id"] for row in trusted_representatives],
            "unverified_dependence_lane_ids": sorted(
                row["lane_id"]
                for row in evaluations
                if row["correlation_cluster"] == UNVERIFIED_DEPENDENCE_CLUSTER
            ),
            "suppressed_correlated_lane_ids": sorted(
                lane_id
                for row in cluster_rows
                for lane_id in row["suppressed_correlated_lane_ids"]
            ),
            "independent_correlation_cluster_count": len(trusted_cluster_ids),
            "distribution_summed": False,
        },
        "guarantee": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "sizing_backsolve_allowed": False,
    }
    return {**body, "board_sha256": canonical_sha256(body)}


__all__ = [
    "CALENDAR_DAYS",
    "DojoGoalBoardError",
    "INPUT_CONTRACT",
    "OUTPUT_CONTRACT",
    "TARGET_MULTIPLE",
    "THRESHOLDS",
    "TRADING_DAYS",
    "build_goal_board",
    "canonical_sha256",
    "load_goal_board_input",
    "required_daily_return",
    "validate_goal_board_input",
]
