"""Pure proof-ladder gates for the DOJO inventory-release replay.

The evaluator in this module performs no file, clock, network, broker, or
process I/O.  Callers must first parse and authenticate the replay artifacts,
then pass the complete arm metrics here.  Missing or malformed evidence is a
measurement failure, never an implicit zero or a permissive fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


WINDOWS = ("TRAIN", "VAL", "S5")
POLICIES = ("BASELINE", "CANDIDATE")
COSTS = ("BASE", "STRESS")
INTRABAR_PATHS = ("OHLC", "OLHC")

# These are preregistered proof floors for this experiment, not production
# execution or sizing parameters.  A future experiment must version its
# evaluator rather than silently changing them.
MIN_SETTLEMENTS = 30
MIN_ACTIVE_DAYS = 20
MIN_INDEPENDENT_STRESS_PROFIT_FACTOR = 1.25
PROOF_MANIFEST_CONTRACT = "QR_DOJO_REPLAY_PROOF_MANIFEST_V1"
DECISION_CONTRACT = "QR_DOJO_REPLAY_GATE_DECISION_V2"

_PROOF_MANIFEST_KEYS = frozenset(
    {
        "contract",
        "paper_only",
        "order_authority",
        "live_permission",
        "candidate_id",
        "spec_sha256",
        "policy_sha256",
        "artifact_manifest_sha256",
        "windows",
        "arms",
        "manifest_sha256",
    }
)
_WINDOW_KEYS = frozenset({"from_utc", "to_utc", "source_sha256"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_COUNT_FIELDS = (
    "settlements",
    "active_days",
    "margin_events",
    "ruin_events",
    "unresolved_positions",
    "unresolved_orders",
    "end_of_replay_forced_close_count",
)
_FINITE_FIELDS = (
    "net_jpy",
    "expectancy_jpy",
    "worst_day_jpy",
    "realized_drawdown_jpy",
    "end_of_replay_forced_close_net_jpy",
)
_DEATH_CODE_PRIORITY = {
    "MEASUREMENT": 0,
    "RISK": 1,
    "OVERFIT": 2,
    "INVENTORY": 3,
}


def _canonical_json_value(value: Any) -> Any:
    """Return a deterministic JSON value, including a typed infinity token."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical objects require string keys")
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return {"__qr_float__": "+Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, float) and math.isnan(value):
        return {"__qr_float__": "NaN"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"non-JSON canonical value: {type(value).__name__}")


def canonical_proof_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a proof-manifest body, excluding its self-referential digest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("proof manifest must be an object")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    raw = json.dumps(
        _canonical_json_value(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reason(
    code: str,
    stage: str,
    message: str,
    *,
    window: str | None = None,
    cost: str | None = None,
    intrabar: str | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "stage": stage,
        "window": window,
        "cost": cost,
        "intrabar": intrabar,
        "policy": policy,
        "message": message,
    }


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed


def _binding_from_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {
            "candidate_id": None,
            "spec_sha256": None,
            "policy_sha256": None,
            "artifact_manifest_sha256": None,
            "manifest_sha256": None,
        }
    return {
        field: manifest.get(field) if isinstance(manifest.get(field), str) else None
        for field in (
            "candidate_id",
            "spec_sha256",
            "policy_sha256",
            "artifact_manifest_sha256",
            "manifest_sha256",
        )
    }


def _proof_manifest_reasons(
    manifest: Any,
    *,
    expected_manifest_sha256: Any,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        return [
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest must be an object",
            )
        ]

    keys = frozenset(manifest)
    if keys != _PROOF_MANIFEST_KEYS:
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest has missing or unknown fields",
            )
        )
    if manifest.get("contract") != PROOF_MANIFEST_CONTRACT:
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest contract is invalid",
            )
        )
    if (
        manifest.get("paper_only") is not True
        or manifest.get("order_authority") != "NONE"
        or manifest.get("live_permission") is not False
    ):
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest safety authority is invalid",
            )
        )

    for field in (
        "candidate_id",
        "spec_sha256",
        "policy_sha256",
        "artifact_manifest_sha256",
        "manifest_sha256",
    ):
        if not _valid_sha256(manifest.get(field)):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "BINDING",
                    f"{field} must be a lowercase SHA-256 digest",
                )
            )

    if not _valid_sha256(expected_manifest_sha256):
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "trusted expected_manifest_sha256 is required",
            )
        )
    elif manifest.get("manifest_sha256") != expected_manifest_sha256:
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest does not match trusted digest",
            )
        )

    try:
        canonical_sha256 = canonical_proof_manifest_sha256(manifest)
    except (TypeError, ValueError):
        canonical_sha256 = None
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest is not canonical JSON evidence",
            )
        )
    if (
        canonical_sha256 is not None
        and manifest.get("manifest_sha256") != canonical_sha256
    ):
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "sealed proof manifest canonical digest mismatch",
            )
        )

    windows = manifest.get("windows")
    if not isinstance(windows, Mapping) or frozenset(windows) != frozenset(WINDOWS):
        reasons.append(
            _reason(
                "MEASUREMENT",
                "BINDING",
                "windows must contain exactly TRAIN, VAL, and S5",
            )
        )
        return reasons

    for window in WINDOWS:
        binding = windows.get(window)
        if not isinstance(binding, Mapping) or frozenset(binding) != (_WINDOW_KEYS):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "BINDING",
                    "window binding has missing or unknown fields",
                    window=window,
                )
            )
            continue
        from_utc = _parse_utc(binding.get("from_utc"))
        to_utc = _parse_utc(binding.get("to_utc"))
        if from_utc is None or to_utc is None or from_utc >= to_utc:
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "BINDING",
                    "window bounds must be ordered UTC timestamps",
                    window=window,
                )
            )
        if not _valid_sha256(binding.get("source_sha256")):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "BINDING",
                    "window source_sha256 is invalid",
                    window=window,
                )
            )
    return reasons


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_profit_factor(value: Any) -> bool:
    """Accept finite nonnegative PF or the legitimate positive-infinity case."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return (math.isfinite(number) and number >= 0.0) or (
        math.isinf(number) and number > 0.0
    )


def _arm_key(
    arm: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    values = tuple(arm.get(field) for field in ("window", "policy", "cost", "intrabar"))
    if not all(isinstance(value, str) for value in values):
        return None
    window, policy, cost, intrabar = values
    if (
        window not in WINDOWS
        or policy not in POLICIES
        or cost not in COSTS
        or intrabar not in INTRABAR_PATHS
    ):
        return None
    return window, policy, cost, intrabar


def _validate_metrics(
    metrics: Any,
    *,
    key: tuple[str, str, str, str],
) -> list[dict[str, Any]]:
    window, policy, cost, intrabar = key
    location = {
        "window": window,
        "policy": policy,
        "cost": cost,
        "intrabar": intrabar,
    }
    if not isinstance(metrics, Mapping):
        return [
            _reason(
                "MEASUREMENT",
                window,
                "arm metrics must be an object",
                **location,
            )
        ]

    reasons: list[dict[str, Any]] = []
    for field in _COUNT_FIELDS:
        if not _nonnegative_int(metrics.get(field)):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    window,
                    f"{field} must be a nonnegative integer",
                    **location,
                )
            )
    for field in _FINITE_FIELDS:
        if not _finite_number(metrics.get(field)):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    window,
                    f"{field} must be a finite number",
                    **location,
                )
            )
    if not _valid_profit_factor(metrics.get("profit_factor")):
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                "profit_factor must be nonnegative or positive infinity",
                **location,
            )
        )

    if reasons:
        return reasons

    settlements = int(metrics["settlements"])
    net = float(metrics["net_jpy"])
    expectancy = float(metrics["expectancy_jpy"])
    expected_expectancy = net / settlements if settlements else 0.0
    if not math.isclose(
        expectancy,
        expected_expectancy,
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                "expectancy_jpy does not reconcile to net_jpy / settlements",
                **location,
            )
        )
    if float(metrics["realized_drawdown_jpy"]) < 0.0:
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                "realized_drawdown_jpy must be nonnegative",
                **location,
            )
        )

    profit_factor = float(metrics["profit_factor"])
    if math.isinf(profit_factor) and not (settlements > 0 and net > 0.0):
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                "infinite profit_factor requires resolved positive net",
                **location,
            )
        )
    return reasons


def _complete_arm_index(
    arms: Any,
) -> tuple[
    dict[tuple[str, str, str, str], Mapping[str, Any]],
    list[dict[str, Any]],
]:
    index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    reasons: list[dict[str, Any]] = []
    if not _is_sequence(arms):
        return {}, [
            _reason(
                "MEASUREMENT",
                "INPUT",
                "arms must be a sequence of replay arm objects",
            )
        ]

    for position, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "INPUT",
                    f"arm at index {position} must be an object",
                )
            )
            continue
        key = _arm_key(arm)
        if key is None:
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    "INPUT",
                    f"arm at index {position} has an invalid identity",
                )
            )
            continue
        if key in index:
            reasons.append(
                _reason(
                    "MEASUREMENT",
                    key[0],
                    "duplicate replay arm identity",
                    window=key[0],
                    policy=key[1],
                    cost=key[2],
                    intrabar=key[3],
                )
            )
            continue
        index[key] = arm
        reasons.extend(_validate_metrics(arm.get("metrics"), key=key))

    return index, reasons


def _missing_arm_reasons(
    index: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    windows: Sequence[str],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    expected = {
        (window, policy, cost, intrabar)
        for window in windows
        for policy in POLICIES
        for cost in COSTS
        for intrabar in INTRABAR_PATHS
    }
    for key in sorted(expected - set(index)):
        reasons.append(
            _reason(
                "MEASUREMENT",
                key[0],
                "required replay arm is missing",
                window=key[0],
                policy=key[1],
                cost=key[2],
                intrabar=key[3],
            )
        )
    return reasons


def _metrics(
    index: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    key: tuple[str, str, str, str],
) -> Mapping[str, Any]:
    metrics = index[key]["metrics"]
    assert isinstance(metrics, Mapping)
    return metrics


def _sample_reasons(
    metrics: Mapping[str, Any],
    *,
    window: str,
    policy: str,
    cost: str,
    intrabar: str,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    location = {
        "window": window,
        "policy": policy,
        "cost": cost,
        "intrabar": intrabar,
    }
    if int(metrics["settlements"]) < MIN_SETTLEMENTS:
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                f"settlements are below the fixed floor {MIN_SETTLEMENTS}",
                **location,
            )
        )
    if int(metrics["active_days"]) < MIN_ACTIVE_DAYS:
        reasons.append(
            _reason(
                "MEASUREMENT",
                window,
                f"active_days are below the fixed floor {MIN_ACTIVE_DAYS}",
                **location,
            )
        )
    return reasons


def _forced_close_reasons(
    metrics: Mapping[str, Any],
    *,
    window: str,
    policy: str,
    cost: str,
    intrabar: str,
) -> list[dict[str, Any]]:
    if (
        int(metrics["end_of_replay_forced_close_count"]) == 0
        and float(metrics["end_of_replay_forced_close_net_jpy"]) == 0.0
    ):
        return []
    return [
        _reason(
            "MEASUREMENT",
            window,
            "end-of-replay forced-close activity is forbidden",
            window=window,
            policy=policy,
            cost=cost,
            intrabar=intrabar,
        )
    ]


def _resolved_candidate_end_reasons(
    metrics: Mapping[str, Any],
    *,
    window: str,
    cost: str,
    intrabar: str,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for field in ("unresolved_positions", "unresolved_orders"):
        if int(metrics[field]) != 0:
            reasons.append(
                _reason(
                    "RISK",
                    window,
                    f"candidate {field} must be zero at replay end",
                    window=window,
                    policy="CANDIDATE",
                    cost=cost,
                    intrabar=intrabar,
                )
            )
    return reasons


def _risk_reasons(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    window: str,
    cost: str,
    intrabar: str,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    location = {
        "window": window,
        "policy": "CANDIDATE",
        "cost": cost,
        "intrabar": intrabar,
    }
    if float(candidate["worst_day_jpy"]) < float(baseline["worst_day_jpy"]):
        reasons.append(
            _reason(
                "RISK",
                window,
                "candidate worst day is worse than baseline",
                **location,
            )
        )
    if float(candidate["realized_drawdown_jpy"]) > float(
        baseline["realized_drawdown_jpy"]
    ):
        reasons.append(
            _reason(
                "RISK",
                window,
                "candidate realized drawdown is worse than baseline",
                **location,
            )
        )
    for field in ("margin_events", "ruin_events"):
        if int(candidate[field]) > int(baseline[field]):
            reasons.append(
                _reason(
                    "RISK",
                    window,
                    f"candidate {field} exceed baseline",
                    **location,
                )
            )
    for field in ("unresolved_positions", "unresolved_orders"):
        if int(candidate[field]) > int(baseline[field]):
            reasons.append(
                _reason(
                    "RISK",
                    window,
                    f"candidate {field} exceed baseline",
                    **location,
                )
            )
    return reasons


def _train_reasons(
    index: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for cost in COSTS:
        for intrabar in INTRABAR_PATHS:
            baseline = _metrics(index, ("TRAIN", "BASELINE", cost, intrabar))
            candidate = _metrics(index, ("TRAIN", "CANDIDATE", cost, intrabar))
            for policy, metrics in (
                ("BASELINE", baseline),
                ("CANDIDATE", candidate),
            ):
                reasons.extend(
                    _sample_reasons(
                        metrics,
                        window="TRAIN",
                        policy=policy,
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
                reasons.extend(
                    _forced_close_reasons(
                        metrics,
                        window="TRAIN",
                        policy=policy,
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
            if float(candidate["net_jpy"]) <= float(baseline["net_jpy"]):
                reasons.append(
                    _reason(
                        "INVENTORY",
                        "TRAIN",
                        "candidate net_jpy did not strictly improve baseline",
                        window="TRAIN",
                        policy="CANDIDATE",
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
            if float(candidate["expectancy_jpy"]) <= float(baseline["expectancy_jpy"]):
                reasons.append(
                    _reason(
                        "INVENTORY",
                        "TRAIN",
                        "candidate expectancy_jpy did not strictly improve baseline",
                        window="TRAIN",
                        policy="CANDIDATE",
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
            reasons.extend(
                _risk_reasons(
                    candidate,
                    baseline,
                    window="TRAIN",
                    cost=cost,
                    intrabar=intrabar,
                )
            )
            reasons.extend(
                _resolved_candidate_end_reasons(
                    candidate,
                    window="TRAIN",
                    cost=cost,
                    intrabar=intrabar,
                )
            )
    return reasons


def _proof_reasons(
    index: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for window in ("VAL", "S5"):
        for cost in COSTS:
            for intrabar in INTRABAR_PATHS:
                baseline = _metrics(index, (window, "BASELINE", cost, intrabar))
                candidate = _metrics(index, (window, "CANDIDATE", cost, intrabar))
                for policy, metrics in (
                    ("BASELINE", baseline),
                    ("CANDIDATE", candidate),
                ):
                    reasons.extend(
                        _sample_reasons(
                            metrics,
                            window=window,
                            policy=policy,
                            cost=cost,
                            intrabar=intrabar,
                        )
                    )
                    reasons.extend(
                        _forced_close_reasons(
                            metrics,
                            window=window,
                            policy=policy,
                            cost=cost,
                            intrabar=intrabar,
                        )
                    )
                reasons.extend(
                    _resolved_candidate_end_reasons(
                        candidate,
                        window=window,
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
                reasons.extend(
                    _risk_reasons(
                        candidate,
                        baseline,
                        window=window,
                        cost=cost,
                        intrabar=intrabar,
                    )
                )
                if cost == "BASE":
                    if float(candidate["net_jpy"]) < 0.0:
                        reasons.append(
                            _reason(
                                "OVERFIT",
                                window,
                                "candidate BASE net_jpy is negative",
                                window=window,
                                policy="CANDIDATE",
                                cost=cost,
                                intrabar=intrabar,
                            )
                        )
                    if float(candidate["expectancy_jpy"]) < 0.0:
                        reasons.append(
                            _reason(
                                "OVERFIT",
                                window,
                                "candidate BASE expectancy_jpy is negative",
                                window=window,
                                policy="CANDIDATE",
                                cost=cost,
                                intrabar=intrabar,
                            )
                        )
                    continue

                if float(candidate["profit_factor"]) < (
                    MIN_INDEPENDENT_STRESS_PROFIT_FACTOR
                ):
                    reasons.append(
                        _reason(
                            "OVERFIT",
                            window,
                            "candidate STRESS profit_factor is below the independent gate",
                            window=window,
                            policy="CANDIDATE",
                            cost=cost,
                            intrabar=intrabar,
                        )
                    )
                if float(candidate["net_jpy"]) <= 0.0:
                    reasons.append(
                        _reason(
                            "OVERFIT",
                            window,
                            "candidate STRESS net_jpy is not positive",
                            window=window,
                            policy="CANDIDATE",
                            cost=cost,
                            intrabar=intrabar,
                        )
                    )
                if float(candidate["expectancy_jpy"]) <= 0.0:
                    reasons.append(
                        _reason(
                            "OVERFIT",
                            window,
                            "candidate STRESS expectancy_jpy is not positive",
                            window=window,
                            policy="CANDIDATE",
                            cost=cost,
                            intrabar=intrabar,
                        )
                    )
    return reasons


def _death_code(reasons: Sequence[Mapping[str, Any]]) -> str | None:
    codes = {
        str(reason.get("code"))
        for reason in reasons
        if str(reason.get("code")) in _DEATH_CODE_PRIORITY
    }
    if not codes:
        return None
    return min(codes, key=lambda code: _DEATH_CODE_PRIORITY[code])


def _decision(
    *,
    decision: str,
    binding: Mapping[str, Any],
    manifest_authenticated: bool,
    train_eligible: bool,
    independent_proof_eligible: bool,
    reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized_reasons = [dict(reason) for reason in reasons]
    return {
        "contract": DECISION_CONTRACT,
        "decision": decision,
        "binding": dict(binding),
        "manifest_authenticated": manifest_authenticated,
        # This module is intentionally pure.  A caller-computed content digest
        # authenticates the metric envelope only; it cannot authenticate the
        # filesystem provenance, candidate chain, replay owner, git HEAD, or a
        # future room registry.  Promotion is exclusively a lifecycle
        # controller responsibility.
        "artifact_provenance_authenticated": False,
        "paper_eligible": False,
        "launch_preflight_token_sha256": None,
        "train_eligible": train_eligible,
        "independent_proof_eligible": independent_proof_eligible,
        "proof_eligible": independent_proof_eligible,
        "death_code": _death_code(materialized_reasons),
        "reasons": materialized_reasons,
    }


def evaluate_inventory_release_proof_ladder(
    proof_manifest: Any,
    *,
    expected_manifest_sha256: Any = None,
) -> dict[str, Any]:
    """Evaluate an authenticated TRAIN -> untouched VAL/S5 proof ladder.

    The returned object is deterministic for the same parsed input.  It never
    mutates ``proof_manifest`` and never performs an external side effect.
    A caller must supply the trusted digest obtained from the sealed job or
    candidate ledger.  A self-digest alone is content addressing, not
    authentication, so a naked arm list or an untrusted digest fails closed.
    """

    binding = _binding_from_manifest(proof_manifest)
    binding_reasons = _proof_manifest_reasons(
        proof_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if binding_reasons:
        return _decision(
            decision="MEASUREMENT_BLOCKED",
            binding=binding,
            manifest_authenticated=False,
            train_eligible=False,
            independent_proof_eligible=False,
            reasons=binding_reasons,
        )

    assert isinstance(proof_manifest, Mapping)
    arms = proof_manifest["arms"]
    index, measurement_reasons = _complete_arm_index(arms)
    measurement_reasons.extend(_missing_arm_reasons(index, ("TRAIN",)))
    if measurement_reasons:
        return _decision(
            decision="MEASUREMENT_BLOCKED",
            binding=binding,
            manifest_authenticated=True,
            train_eligible=False,
            independent_proof_eligible=False,
            reasons=measurement_reasons,
        )

    train_reasons = _train_reasons(index)
    if train_reasons:
        return _decision(
            decision="TRAIN_REJECTED",
            binding=binding,
            manifest_authenticated=True,
            train_eligible=False,
            independent_proof_eligible=False,
            reasons=train_reasons,
        )

    proof_measurement_reasons = _missing_arm_reasons(index, ("VAL", "S5"))
    if proof_measurement_reasons:
        return _decision(
            decision="MEASUREMENT_BLOCKED",
            binding=binding,
            manifest_authenticated=True,
            train_eligible=True,
            independent_proof_eligible=False,
            reasons=proof_measurement_reasons,
        )

    proof_reasons = _proof_reasons(index)
    if proof_reasons:
        return _decision(
            decision="PROOF_REJECTED",
            binding=binding,
            manifest_authenticated=True,
            train_eligible=True,
            independent_proof_eligible=False,
            reasons=proof_reasons,
        )

    return _decision(
        decision="PROOF_ELIGIBLE",
        binding=binding,
        manifest_authenticated=True,
        train_eligible=True,
        independent_proof_eligible=True,
        reasons=[],
    )
