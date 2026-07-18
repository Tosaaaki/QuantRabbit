"""Append-only lifecycle for the prospective DOJO worker smoke.

The lifecycle separates three things that older worker experiments mixed:

* a candidate and mechanics lock written before the market window;
* daily, ordered source receipts written after each market day; and
* one full-window result manifest written only after the window matures.

The receipts are deliberately self-attested and never grant promotion or live
authority.  A local hash chain can expose accidental mutation, but only an
external monotonic witness can prove that the history was not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PRECOMMIT_CONTRACT = "QR_DOJO_WORKER_FORWARD_PRECOMMIT_V1"
START_CONTRACT = "QR_DOJO_WORKER_FORWARD_START_V1"
DAY_SEAL_CONTRACT = "QR_DOJO_WORKER_FORWARD_DAY_SEAL_V1"
FINAL_CONTRACT = "QR_DOJO_WORKER_FORWARD_FINAL_V1"
STATUS_CONTRACT = "QR_DOJO_WORKER_FORWARD_STATUS_V1"
EVIDENCE_TIER = "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
INTRABAR_PATHS = ("OHLC", "OLHC")
FINAL_STATES = frozenset({"FINALIZED_PASS", "FINALIZED_FAIL"})
RESULT_STATUSES = frozenset(
    {
        "PASS_POSITIVE_RESOLVED_BALANCE",
        "FAIL_NON_POSITIVE_RESOLVED_BALANCE",
        "FAIL_MARGIN_CLOSEOUT",
        "INVALID_ZERO_TRADES",
        "INVALID_TERMINAL_EXPOSURE",
        "INVALID_RUNNER_FAILURE",
        "INVALID_UNSCOREABLE_TRIAL",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_PRECOMMIT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "experiment_id",
        "state",
        "created_at_utc",
        "window",
        "study",
        "candidate_set",
        "mechanics",
        "source_bindings",
        "thresholds",
        "daily_source_policy",
        "selection_policy",
        "authority",
        "attestations",
        "precommit_sha256",
    }
)
_CANDIDATE_KEYS = frozenset({"candidate_id", "family_id", "config", "config_sha256"})
_DAY_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "pair",
        "granularity",
        "content_sha256",
        "size_bytes",
        "row_count",
        "first_event_utc",
        "last_event_utc",
    }
)
_RESULT_KEYS = frozenset(
    {
        "candidate_id",
        "intrabar",
        "status",
        "ledger_sha256",
        "score_receipt_sha256",
        "entries",
        "resolved_exits",
        "terminal_net_jpy",
        "calendar_30d_multiple",
        "margin_closeouts",
        "terminal_resolved",
        "promotion_eligible",
    }
)


class DojoWorkerForwardError(ValueError):
    """A forward receipt is malformed, late, missing, or inconsistent."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_precommit(spec: Mapping[str, Any], *, now_utc: datetime) -> dict[str, Any]:
    """Build the exact candidate/mechanics lock before the future window."""

    source = _mapping(spec, "precommit spec")
    _exact_keys(
        source,
        {
            "experiment_id",
            "window",
            "candidate_set",
            "mechanics",
            "source_bindings",
            "thresholds",
            "daily_source_policy",
        },
        "precommit spec",
    )
    now = _utc(now_utc, "now_utc")
    experiment_id = _identifier(source["experiment_id"], "experiment_id")
    window = _validate_window(source["window"])
    start = _parse_utc(window["start_utc"], "window.start_utc")
    end = _parse_utc(window["end_utc"], "window.end_utc")
    if now >= start:
        raise DojoWorkerForwardError("forward precommit must precede window start")
    if start.time() != datetime.min.time() or end.time() != datetime.min.time():
        raise DojoWorkerForwardError("forward window must use whole UTC days")
    calendar_days = (end - start).days
    if calendar_days != 14:
        raise DojoWorkerForwardError("worker phase-1 smoke must be exactly 14 days")

    candidate_set = _validate_candidate_set(source["candidate_set"])
    mechanics = _validate_mechanics(source["mechanics"])
    source_bindings = _validate_source_bindings(source["source_bindings"])
    thresholds = _validate_thresholds(source["thresholds"])
    daily_policy = _validate_daily_source_policy(
        source["daily_source_policy"], mechanics=mechanics
    )
    body = {
        "contract": PRECOMMIT_CONTRACT,
        "schema_version": 1,
        "experiment_id": experiment_id,
        "state": "PRECOMMITTED",
        "created_at_utc": _iso(now),
        "window": {**window, "calendar_days": calendar_days},
        "study": {
            "kind": "PROSPECTIVE_HYPOTHESIS_SCREEN",
            "phase": "PHASE_1_14_DAY_SMOKE",
            "confirmatory": False,
            "smoke_only": True,
            "minimum_forward_active_days_for_proof": 60,
            "proof_threshold_met_by_this_window": False,
        },
        "candidate_set": candidate_set,
        "mechanics": mechanics,
        "source_bindings": source_bindings,
        "thresholds": thresholds,
        "daily_source_policy": daily_policy,
        "selection_policy": {
            "candidate_reselection_allowed": False,
            "candidate_parameter_change_allowed": False,
            "scorer_change_allowed": False,
            "intrabar_cherry_pick_allowed": False,
            "all_declared_candidates_remain_in_denominator": True,
            "both_intrabar_paths_required": True,
            "pessimistic_intrabar_is_authoritative": True,
        },
        "authority": _authority(),
        "attestations": _attestations(),
    }
    return {**body, "precommit_sha256": canonical_sha256(body)}


def validate_precommit(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(value, "precommit")
    _exact_keys(artifact, _PRECOMMIT_KEYS, "precommit")
    digest = _sha(artifact["precommit_sha256"], "precommit_sha256")
    body = {key: item for key, item in artifact.items() if key != "precommit_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoWorkerForwardError("precommit digest mismatch")
    if artifact["contract"] != PRECOMMIT_CONTRACT or artifact["schema_version"] != 1:
        raise DojoWorkerForwardError("precommit contract is invalid")
    if artifact["state"] != "PRECOMMITTED":
        raise DojoWorkerForwardError("precommit state is invalid")
    _identifier(artifact["experiment_id"], "experiment_id")
    _parse_utc(artifact["created_at_utc"], "created_at_utc")
    rebuilt = build_precommit(
        {
            "experiment_id": artifact["experiment_id"],
            "window": artifact["window"],
            "candidate_set": {
                key: artifact["candidate_set"][key]
                for key in (
                    "declared_grid_size",
                    "family_denominator",
                    "candidates",
                )
            },
            "mechanics": artifact["mechanics"],
            "source_bindings": artifact["source_bindings"],
            "thresholds": artifact["thresholds"],
            "daily_source_policy": artifact["daily_source_policy"],
        },
        now_utc=_parse_utc(artifact["created_at_utc"], "created_at_utc"),
    )
    # build_precommit adds only deterministic policy fields around caller inputs.
    if rebuilt != artifact:
        raise DojoWorkerForwardError("precommit policy or nested schema drifted")
    return _snapshot(artifact)


def build_start_receipt(
    precommit: Mapping[str, Any], *, now_utc: datetime
) -> dict[str, Any]:
    """Seal the experiment start exactly once before the first observation."""

    lock = validate_precommit(precommit)
    now = _utc(now_utc, "now_utc")
    created = _parse_utc(lock["created_at_utc"], "created_at_utc")
    start = _parse_utc(lock["window"]["start_utc"], "window.start_utc")
    if now < created:
        raise DojoWorkerForwardError("start receipt predates precommit")
    if now >= start:
        raise DojoWorkerForwardError("forward start was missed")
    body = {
        "contract": START_CONTRACT,
        "schema_version": 1,
        "experiment_id": lock["experiment_id"],
        "state": "STARTED",
        "started_at_utc": _iso(now),
        "window_start_utc": lock["window"]["start_utc"],
        "precommit_sha256": lock["precommit_sha256"],
        "previous_receipt_sha256": lock["precommit_sha256"],
        "candidate_count": lock["candidate_set"]["candidate_count"],
        "candidate_set_sha256": lock["candidate_set"]["candidate_set_sha256"],
        "authority": _authority(),
        "attestations": _attestations(),
    }
    return {**body, "start_receipt_sha256": canonical_sha256(body)}


def validate_start_receipt(
    value: Mapping[str, Any], precommit: Mapping[str, Any]
) -> dict[str, Any]:
    lock = validate_precommit(precommit)
    receipt = _mapping(value, "start receipt")
    _exact_keys(
        receipt,
        {
            "contract",
            "schema_version",
            "experiment_id",
            "state",
            "started_at_utc",
            "window_start_utc",
            "precommit_sha256",
            "previous_receipt_sha256",
            "candidate_count",
            "candidate_set_sha256",
            "authority",
            "attestations",
            "start_receipt_sha256",
        },
        "start receipt",
    )
    digest = _sha(receipt["start_receipt_sha256"], "start_receipt_sha256")
    body = {key: item for key, item in receipt.items() if key != "start_receipt_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoWorkerForwardError("start receipt digest mismatch")
    expected = build_start_receipt(
        lock,
        now_utc=_parse_utc(receipt["started_at_utc"], "started_at_utc"),
    )
    if expected != receipt:
        raise DojoWorkerForwardError("start receipt parent or policy drifted")
    return _snapshot(receipt)


def build_day_seal(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    source_manifest: Mapping[str, Any],
    *,
    ordinal: int,
    now_utc: datetime,
) -> dict[str, Any]:
    """Seal one complete UTC day in strict ordinal order."""

    lock = validate_precommit(precommit)
    start_receipt = validate_start_receipt(start_receipt, lock)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise DojoWorkerForwardError("day ordinal must be an integer")
    day_count = lock["window"]["calendar_days"]
    if ordinal < 1 or ordinal > day_count:
        raise DojoWorkerForwardError("day ordinal is outside the window")
    if ordinal == 1:
        if previous_day_seal is not None:
            raise DojoWorkerForwardError("day 1 cannot have a previous day seal")
        previous_sha = start_receipt["start_receipt_sha256"]
    else:
        if previous_day_seal is None:
            raise DojoWorkerForwardError("daily source seal gap detected")
        prior = validate_day_seal(
            previous_day_seal,
            lock,
            start_receipt,
            expected_ordinal=ordinal - 1,
        )
        previous_sha = prior["day_seal_sha256"]

    now = _utc(now_utc, "now_utc")
    window_start = _parse_utc(lock["window"]["start_utc"], "window.start_utc")
    day_start = window_start + timedelta(days=ordinal - 1)
    day_end = day_start + timedelta(days=1)
    grace = timedelta(hours=lock["daily_source_policy"]["seal_grace_hours"])
    if now < day_end:
        raise DojoWorkerForwardError("daily source cannot be sealed before day end")
    if now > day_end + grace:
        raise DojoWorkerForwardError("daily source seal deadline was missed")
    source = _validate_day_source_manifest(
        source_manifest,
        lock=lock,
        day_start=day_start,
        day_end=day_end,
    )
    body = {
        "contract": DAY_SEAL_CONTRACT,
        "schema_version": 1,
        "experiment_id": lock["experiment_id"],
        "state": "COLLECTING" if ordinal < day_count else "MATURED",
        "ordinal": ordinal,
        "day_start_utc": _iso(day_start),
        "day_end_utc": _iso(day_end),
        "sealed_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start_receipt["start_receipt_sha256"],
        "previous_receipt_sha256": previous_sha,
        "source_manifest": source,
        "source_manifest_sha256": canonical_sha256(source),
        "authority": _authority(),
        "attestations": _attestations(),
    }
    return {**body, "day_seal_sha256": canonical_sha256(body)}


def validate_day_seal(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    *,
    expected_ordinal: int | None = None,
) -> dict[str, Any]:
    lock = validate_precommit(precommit)
    start_receipt = validate_start_receipt(start_receipt, lock)
    seal = _mapping(value, "day seal")
    _exact_keys(
        seal,
        {
            "contract",
            "schema_version",
            "experiment_id",
            "state",
            "ordinal",
            "day_start_utc",
            "day_end_utc",
            "sealed_at_utc",
            "precommit_sha256",
            "start_receipt_sha256",
            "previous_receipt_sha256",
            "source_manifest",
            "source_manifest_sha256",
            "authority",
            "attestations",
            "day_seal_sha256",
        },
        "day seal",
    )
    if seal["contract"] != DAY_SEAL_CONTRACT or seal["schema_version"] != 1:
        raise DojoWorkerForwardError("day seal contract is invalid")
    ordinal = seal["ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise DojoWorkerForwardError("day seal ordinal is invalid")
    if expected_ordinal is not None and ordinal != expected_ordinal:
        raise DojoWorkerForwardError("daily source seal gap detected")
    digest = _sha(seal["day_seal_sha256"], "day_seal_sha256")
    body = {key: item for key, item in seal.items() if key != "day_seal_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoWorkerForwardError("day seal digest mismatch")
    if seal["precommit_sha256"] != lock["precommit_sha256"]:
        raise DojoWorkerForwardError("day seal precommit parent drifted")
    if seal["start_receipt_sha256"] != start_receipt["start_receipt_sha256"]:
        raise DojoWorkerForwardError("day seal start parent drifted")
    if canonical_sha256(seal["source_manifest"]) != _sha(
        seal["source_manifest_sha256"], "source_manifest_sha256"
    ):
        raise DojoWorkerForwardError("day source manifest digest mismatch")
    window_start = _parse_utc(lock["window"]["start_utc"], "window.start_utc")
    day_start = window_start + timedelta(days=ordinal - 1)
    day_end = day_start + timedelta(days=1)
    _validate_day_source_manifest(
        seal["source_manifest"],
        lock=lock,
        day_start=day_start,
        day_end=day_end,
    )
    sealed = _parse_utc(seal["sealed_at_utc"], "sealed_at_utc")
    grace = timedelta(hours=lock["daily_source_policy"]["seal_grace_hours"])
    expected_state = (
        "MATURED" if ordinal == lock["window"]["calendar_days"] else "COLLECTING"
    )
    if (
        seal["experiment_id"] != lock["experiment_id"]
        or seal["state"] != expected_state
        or seal["day_start_utc"] != _iso(day_start)
        or seal["day_end_utc"] != _iso(day_end)
        or sealed < day_end
        or sealed > day_end + grace
        or seal["authority"] != _authority()
        or seal["attestations"] != _attestations()
    ):
        raise DojoWorkerForwardError("day seal timing or policy drifted")
    return _snapshot(seal)


def build_final_receipt(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    day_seals: Sequence[Mapping[str, Any]],
    result_manifest: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    """Score the fixed full-window result set after all source days mature."""

    lock = validate_precommit(precommit)
    start_receipt = validate_start_receipt(start_receipt, lock)
    seals = _validate_day_chain(day_seals, lock, start_receipt)
    now = _utc(now_utc, "now_utc")
    end = _parse_utc(lock["window"]["end_utc"], "window.end_utc")
    if now < end:
        raise DojoWorkerForwardError("forward result cannot finalize before maturity")
    results = _validate_result_manifest(result_manifest, lock=lock)
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results["results"]:
        by_candidate.setdefault(row["candidate_id"], {})[row["intrabar"]] = row
    candidate_summaries: list[dict[str, Any]] = []
    passing_candidates: list[str] = []
    open_market_days = sum(
        not seal["source_manifest"]["market_closed"] for seal in seals
    )
    source_coverage_passed = bool(
        open_market_days >= lock["daily_source_policy"]["minimum_open_market_days"]
    )
    for candidate in lock["candidate_set"]["candidates"]:
        candidate_id = candidate["candidate_id"]
        paths = by_candidate[candidate_id]
        pessimistic = min(paths.values(), key=lambda item: item["terminal_net_jpy"])
        both_economic = all(_economic_result_passes(item) for item in paths.values())
        threshold_passed = bool(
            source_coverage_passed
            and both_economic
            and pessimistic["calendar_30d_multiple"]
            >= lock["thresholds"]["minimum_calendar_30d_multiple"]
            and max(item["margin_closeouts"] for item in paths.values())
            <= lock["thresholds"]["maximum_margin_closeouts"]
        )
        if threshold_passed:
            passing_candidates.append(candidate_id)
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "both_intrabar_economic_gate_passed": both_economic,
                "pessimistic_intrabar": pessimistic["intrabar"],
                "pessimistic_terminal_net_jpy": pessimistic["terminal_net_jpy"],
                "pessimistic_calendar_30d_multiple": pessimistic[
                    "calendar_30d_multiple"
                ],
                "smoke_threshold_passed": threshold_passed,
                "promotion_eligible": False,
            }
        )
    state = "FINALIZED_PASS" if passing_candidates else "FINALIZED_FAIL"
    blockers = [
        "PHASE_1_14_DAY_SMOKE_ONLY",
        "MINIMUM_60_FORWARD_ACTIVE_DAYS_NOT_MET",
        "EXTERNAL_MONOTONIC_WITNESS_ABSENT",
        "INDEPENDENT_MARKET_ATTESTATION_ABSENT",
        "DOJO_HAS_NO_LIVE_AUTHORITY",
    ]
    if not source_coverage_passed:
        blockers.append("MINIMUM_OPEN_MARKET_DAY_COVERAGE_NOT_MET")
    body = {
        "contract": FINAL_CONTRACT,
        "schema_version": 1,
        "experiment_id": lock["experiment_id"],
        "state": state,
        "finalized_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start_receipt["start_receipt_sha256"],
        "last_day_seal_sha256": seals[-1]["day_seal_sha256"],
        "daily_seal_count": len(seals),
        "open_market_day_count": open_market_days,
        "minimum_open_market_days": lock["daily_source_policy"][
            "minimum_open_market_days"
        ],
        "source_coverage_passed": source_coverage_passed,
        "result_manifest_sha256": canonical_sha256(results),
        "candidate_summaries": candidate_summaries,
        "smoke_passing_candidate_ids": passing_candidates,
        "smoke_gate_passed": bool(passing_candidates),
        "proof_eligible": False,
        "promotion_eligible": False,
        "evidence_tier": EVIDENCE_TIER,
        "effective_independent_n": 0,
        "goal_status": "3X_NOT_REACHABLE",
        "promotion_blockers": blockers,
        "authority": _authority(),
        "attestations": _attestations(),
    }
    return {**body, "final_receipt_sha256": canonical_sha256(body)}


def validate_result_manifest(
    value: Mapping[str, Any], precommit: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize the exact two-intrabar result denominator for persistence."""

    lock = validate_precommit(precommit)
    return _validate_result_manifest(value, lock=lock)


def audit_lifecycle(run_dir: Path, *, now_utc: datetime) -> dict[str, Any]:
    """Return the fail-closed state without mutating the run directory."""

    now = _utc(now_utc, "now_utc")
    precommit_path = run_dir / "precommit.json"
    if not precommit_path.is_file():
        return _status("ABSENT", ["PRECOMMIT_ABSENT"])
    try:
        precommit = validate_precommit(_read_json(precommit_path))
    except (OSError, DojoWorkerForwardError, json.JSONDecodeError) as exc:
        return _status("INVALIDATED", [f"INVALID_PRECOMMIT:{exc}"])
    start_path = run_dir / "start.json"
    window_start = _parse_utc(precommit["window"]["start_utc"], "window.start_utc")
    if not start_path.is_file():
        return _status(
            "MISSED_START" if now >= window_start else "PRECOMMITTED",
            ["START_RECEIPT_ABSENT"] if now >= window_start else [],
            precommit=precommit,
        )
    try:
        start_receipt = validate_start_receipt(_read_json(start_path), precommit)
    except (OSError, DojoWorkerForwardError, json.JSONDecodeError) as exc:
        return _status("INVALIDATED", [f"INVALID_START:{exc}"], precommit=precommit)
    day_paths = sorted((run_dir / "days").glob("day-*.json"))
    seals: list[dict[str, Any]] = []
    for expected, path in enumerate(day_paths, start=1):
        try:
            seal = validate_day_seal(
                _read_json(path),
                precommit,
                start_receipt,
                expected_ordinal=expected,
            )
        except (OSError, DojoWorkerForwardError, json.JSONDecodeError) as exc:
            return _status(
                "GAP",
                [f"INVALID_DAY_{expected}:{exc}"],
                precommit=precommit,
                sealed_days=len(seals),
            )
        expected_previous = (
            start_receipt["start_receipt_sha256"]
            if expected == 1
            else seals[-1]["day_seal_sha256"]
        )
        if seal["previous_receipt_sha256"] != expected_previous:
            return _status(
                "GAP",
                [f"DAY_{expected}_CHAIN_MISMATCH"],
                precommit=precommit,
                sealed_days=len(seals),
            )
        seals.append(seal)
    expected_next = len(seals) + 1
    if expected_next <= precommit["window"]["calendar_days"]:
        due = window_start + timedelta(days=expected_next)
        grace = timedelta(hours=precommit["daily_source_policy"]["seal_grace_hours"])
        if now > due + grace:
            return _status(
                "GAP",
                [f"DAY_{expected_next}_SEAL_DEADLINE_MISSED"],
                precommit=precommit,
                sealed_days=len(seals),
            )
        return _status(
            "COLLECTING" if now >= window_start else "STARTED",
            [],
            precommit=precommit,
            sealed_days=len(seals),
        )
    final_path = run_dir / "final.json"
    if not final_path.is_file():
        return _status(
            "MATURED",
            ["FINAL_RECEIPT_ABSENT"],
            precommit=precommit,
            sealed_days=len(seals),
        )
    final = _mapping(_read_json(final_path), "final receipt")
    digest = _sha(final.get("final_receipt_sha256"), "final_receipt_sha256")
    final_body = {
        key: item for key, item in final.items() if key != "final_receipt_sha256"
    }
    if canonical_sha256(final_body) != digest or final.get("state") not in FINAL_STATES:
        return _status(
            "INVALIDATED",
            ["FINAL_RECEIPT_INVALID"],
            precommit=precommit,
            sealed_days=len(seals),
        )
    return _status(
        final["state"],
        list(final.get("promotion_blockers") or []),
        precommit=precommit,
        sealed_days=len(seals),
    )


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one mode-0600 JSON artifact without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _validate_candidate_set(value: Any) -> dict[str, Any]:
    source = _mapping(value, "candidate_set")
    _exact_keys(
        source,
        {"declared_grid_size", "family_denominator", "candidates"},
        "candidate_set",
    )
    declared = _integer(source["declared_grid_size"], "declared_grid_size", minimum=1)
    family_denominator = _integer(
        source["family_denominator"], "family_denominator", minimum=1
    )
    raw_candidates = source["candidates"]
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise DojoWorkerForwardError("candidate_set.candidates must be a sequence")
    candidates: list[dict[str, Any]] = []
    ids: set[str] = set()
    families: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        item = _mapping(raw, f"candidate[{index}]")
        _exact_keys(item, _CANDIDATE_KEYS, f"candidate[{index}]")
        candidate_id = _identifier(item["candidate_id"], "candidate_id")
        family_id = _identifier(item["family_id"], "family_id")
        if candidate_id in ids:
            raise DojoWorkerForwardError("candidate id is duplicated")
        ids.add(candidate_id)
        families.add(family_id)
        config = _mapping(item["config"], "candidate config")
        config_copy = _snapshot(config)
        if _sha(item["config_sha256"], "config_sha256") != canonical_sha256(
            config_copy
        ):
            raise DojoWorkerForwardError("candidate config digest mismatch")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "family_id": family_id,
                "config": config_copy,
                "config_sha256": canonical_sha256(config_copy),
            }
        )
    if len(candidates) != declared:
        raise DojoWorkerForwardError("declared grid size differs from candidate count")
    if len(families) != family_denominator:
        raise DojoWorkerForwardError("family denominator differs from fixed families")
    candidates.sort(key=lambda item: item["candidate_id"])
    body = {
        "declared_grid_size": declared,
        "family_denominator": family_denominator,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return {**body, "candidate_set_sha256": canonical_sha256(body)}


def _validate_mechanics(value: Any) -> dict[str, Any]:
    source = _mapping(value, "mechanics")
    _exact_keys(
        source,
        {
            "pairs",
            "granularity",
            "bot_bar",
            "intrabar_paths",
            "initial_balance_jpy",
            "slippage_pips_per_fill",
            "financing_pips_per_day",
            "leverage",
            "period_end_settlement",
            "terminal_score_basis",
        },
        "mechanics",
    )
    pairs = source["pairs"]
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise DojoWorkerForwardError("mechanics.pairs must be a sequence")
    normalized_pairs = sorted({_identifier(item, "pair") for item in pairs})
    if not normalized_pairs or len(normalized_pairs) != len(pairs):
        raise DojoWorkerForwardError("mechanics.pairs is empty or duplicated")
    if source["granularity"] not in {"M1", "S5"}:
        raise DojoWorkerForwardError("mechanics granularity is invalid")
    if source["bot_bar"] not in {"feed", "M1"}:
        raise DojoWorkerForwardError("mechanics bot_bar is invalid")
    if tuple(source["intrabar_paths"]) != INTRABAR_PATHS:
        raise DojoWorkerForwardError("both ordered intrabar paths are required")
    if (
        source["period_end_settlement"]
        != "CANCEL_OWNED_ORDERS_THEN_CLOSE_OWNED_POSITIONS"
    ):
        raise DojoWorkerForwardError("period-end settlement policy is invalid")
    if source["terminal_score_basis"] != "FULLY_RESOLVED_BALANCE":
        raise DojoWorkerForwardError("terminal score basis is invalid")
    numeric = {
        "initial_balance_jpy": _number(
            source["initial_balance_jpy"], "initial_balance_jpy", positive=True
        ),
        "slippage_pips_per_fill": _number(
            source["slippage_pips_per_fill"],
            "slippage_pips_per_fill",
            positive=True,
        ),
        "financing_pips_per_day": _number(
            source["financing_pips_per_day"],
            "financing_pips_per_day",
            positive=True,
        ),
        "leverage": _number(source["leverage"], "leverage", positive=True),
    }
    if not math.isclose(numeric["leverage"], 25.0, rel_tol=0, abs_tol=1e-12):
        raise DojoWorkerForwardError(
            "worker replay requires the VirtualBroker fixed 25x leverage"
        )
    return {
        "pairs": normalized_pairs,
        "granularity": source["granularity"],
        "bot_bar": source["bot_bar"],
        "intrabar_paths": list(INTRABAR_PATHS),
        **numeric,
        "period_end_settlement": source["period_end_settlement"],
        "terminal_score_basis": source["terminal_score_basis"],
    }


def _validate_source_bindings(value: Any) -> dict[str, Any]:
    source = _mapping(value, "source_bindings")
    base_keys = {
        "git_commit",
        "runner_sha256",
        "bot_module_sha256",
        "bot_dependency_sha256",
        "scorer_sha256",
        "precommit_builder_sha256",
    }
    runtime_keys = {
        "python_executable_path",
        "python_executable_sha256",
        "python_version",
    }
    if set(source) not in (base_keys, base_keys | runtime_keys):
        raise DojoWorkerForwardError("source_bindings schema is not exact")
    git_commit = _git_oid_value(source["git_commit"], "git_commit")
    dependencies = _mapping(source["bot_dependency_sha256"], "bot dependencies")
    normalized_dependencies: dict[str, str] = {}
    for path, digest in dependencies.items():
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise DojoWorkerForwardError("bot dependency path is unsafe")
        normalized_dependencies[path] = _sha(digest, f"bot dependency {path}")
    if not normalized_dependencies:
        raise DojoWorkerForwardError("bot dependency closure cannot be empty")
    result = {
        "git_commit": git_commit,
        "runner_sha256": _sha(source["runner_sha256"], "runner_sha256"),
        "bot_module_sha256": _sha(source["bot_module_sha256"], "bot_module_sha256"),
        "bot_dependency_sha256": dict(sorted(normalized_dependencies.items())),
        "scorer_sha256": _sha(source["scorer_sha256"], "scorer_sha256"),
        "precommit_builder_sha256": _sha(
            source["precommit_builder_sha256"], "precommit_builder_sha256"
        ),
    }
    if runtime_keys.issubset(source):
        executable = source["python_executable_path"]
        version = source["python_version"]
        if (
            not isinstance(executable, str)
            or not Path(executable).is_absolute()
            or not executable
            or not isinstance(version, str)
            or not version
        ):
            raise DojoWorkerForwardError("Python runtime binding is invalid")
        result.update(
            {
                "python_executable_path": executable,
                "python_executable_sha256": _sha(
                    source["python_executable_sha256"],
                    "python_executable_sha256",
                ),
                "python_version": version,
            }
        )
    return result


def _validate_thresholds(value: Any) -> dict[str, Any]:
    source = _mapping(value, "thresholds")
    _exact_keys(
        source,
        {
            "minimum_calendar_30d_multiple",
            "maximum_margin_closeouts",
            "zero_trade_policy",
        },
        "thresholds",
    )
    minimum = _number(
        source["minimum_calendar_30d_multiple"],
        "minimum_calendar_30d_multiple",
        positive=True,
    )
    if minimum < 1:
        raise DojoWorkerForwardError("minimum 30d multiple cannot be below one")
    maximum = _integer(
        source["maximum_margin_closeouts"],
        "maximum_margin_closeouts",
        minimum=0,
    )
    if source["zero_trade_policy"] != "FAIL_CLOSED":
        raise DojoWorkerForwardError("zero trade policy must fail closed")
    return {
        "minimum_calendar_30d_multiple": minimum,
        "maximum_margin_closeouts": maximum,
        "zero_trade_policy": "FAIL_CLOSED",
    }


def _validate_daily_source_policy(
    value: Any, *, mechanics: Mapping[str, Any]
) -> dict[str, Any]:
    source = _mapping(value, "daily_source_policy")
    _exact_keys(
        source,
        {
            "source_origin",
            "seal_grace_hours",
            "minimum_open_market_days",
            "expected_source_ids",
            "late_backfill_allowed",
            "market_closed_days_require_explicit_receipt",
        },
        "daily_source_policy",
    )
    origin = _text(source["source_origin"], "source_origin", maximum=200)
    grace = _integer(source["seal_grace_hours"], "seal_grace_hours", minimum=1)
    if grace > 24:
        raise DojoWorkerForwardError("daily source seal grace cannot exceed 24 hours")
    minimum_open_days = _integer(
        source["minimum_open_market_days"],
        "minimum_open_market_days",
        minimum=1,
    )
    if minimum_open_days > 14:
        raise DojoWorkerForwardError("minimum open-market days exceeds the window")
    expected = source["expected_source_ids"]
    if isinstance(expected, (str, bytes)) or not isinstance(expected, Sequence):
        raise DojoWorkerForwardError("expected_source_ids must be a sequence")
    ids = sorted({_identifier(item, "source_id") for item in expected})
    expected_ids = sorted(
        f"{pair}:{mechanics['granularity']}" for pair in mechanics["pairs"]
    )
    if ids != expected_ids or len(ids) != len(expected):
        raise DojoWorkerForwardError("expected source ids do not match mechanics")
    if source["late_backfill_allowed"] is not False:
        raise DojoWorkerForwardError("late backfill must remain forbidden")
    if source["market_closed_days_require_explicit_receipt"] is not True:
        raise DojoWorkerForwardError("market closure must be explicit")
    return {
        "source_origin": origin,
        "seal_grace_hours": grace,
        "minimum_open_market_days": minimum_open_days,
        "expected_source_ids": ids,
        "late_backfill_allowed": False,
        "market_closed_days_require_explicit_receipt": True,
    }


def _validate_day_source_manifest(
    value: Any,
    *,
    lock: Mapping[str, Any],
    day_start: datetime,
    day_end: datetime,
) -> dict[str, Any]:
    source = _mapping(value, "day source manifest")
    _exact_keys(
        source,
        {"market_closed", "closure_reason", "sources"},
        "day source manifest",
    )
    if not isinstance(source["market_closed"], bool):
        raise DojoWorkerForwardError("market_closed must be boolean")
    rows = source["sources"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise DojoWorkerForwardError("day sources must be a sequence")
    if source["market_closed"]:
        reason = _text(source["closure_reason"], "closure_reason", maximum=200)
        if rows:
            raise DojoWorkerForwardError("market-closed day cannot contain sources")
        return {"market_closed": True, "closure_reason": reason, "sources": []}
    if source["closure_reason"] is not None:
        raise DojoWorkerForwardError("open market day cannot declare closure_reason")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(rows):
        item = _mapping(raw, f"source[{index}]")
        _exact_keys(item, _DAY_SOURCE_KEYS, f"source[{index}]")
        source_id = _identifier(item["source_id"], "source_id")
        if source_id in ids:
            raise DojoWorkerForwardError("daily source id is duplicated")
        ids.add(source_id)
        pair, _, granularity = source_id.partition(":")
        if item["pair"] != pair or item["granularity"] != granularity:
            raise DojoWorkerForwardError("daily source identity is inconsistent")
        first = _parse_utc(item["first_event_utc"], "first_event_utc")
        last = _parse_utc(item["last_event_utc"], "last_event_utc")
        if first < day_start or last >= day_end or first > last:
            raise DojoWorkerForwardError("daily source events escape the sealed day")
        normalized.append(
            {
                "source_id": source_id,
                "pair": pair,
                "granularity": granularity,
                "content_sha256": _sha(item["content_sha256"], "content_sha256"),
                "size_bytes": _integer(item["size_bytes"], "size_bytes", minimum=1),
                "row_count": _integer(item["row_count"], "row_count", minimum=1),
                "first_event_utc": _iso(first),
                "last_event_utc": _iso(last),
            }
        )
    expected = set(lock["daily_source_policy"]["expected_source_ids"])
    if ids != expected:
        raise DojoWorkerForwardError("daily source coverage is incomplete")
    normalized.sort(key=lambda item: item["source_id"])
    return {"market_closed": False, "closure_reason": None, "sources": normalized}


def _validate_result_manifest(value: Any, *, lock: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(value, "result manifest")
    _exact_keys(
        source,
        {
            "precommit_sha256",
            "window_start_utc",
            "window_end_utc",
            "results",
        },
        "result manifest",
    )
    if (
        source["precommit_sha256"] != lock["precommit_sha256"]
        or source["window_start_utc"] != lock["window"]["start_utc"]
        or source["window_end_utc"] != lock["window"]["end_utc"]
    ):
        raise DojoWorkerForwardError("result manifest parent or window drifted")
    rows = source["results"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise DojoWorkerForwardError("results must be a sequence")
    expected_candidates = {
        item["candidate_id"] for item in lock["candidate_set"]["candidates"]
    }
    expected_cells = {
        (candidate_id, intrabar)
        for candidate_id in expected_candidates
        for intrabar in INTRABAR_PATHS
    }
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = _mapping(raw, f"result[{index}]")
        _exact_keys(item, _RESULT_KEYS, f"result[{index}]")
        candidate_id = _identifier(item["candidate_id"], "candidate_id")
        intrabar = item["intrabar"]
        key = (candidate_id, intrabar)
        if key not in expected_cells or key in seen:
            raise DojoWorkerForwardError("result cell is unknown or duplicated")
        seen.add(key)
        if item["status"] not in RESULT_STATUSES:
            raise DojoWorkerForwardError("result status is unsupported")
        if item["promotion_eligible"] is not False:
            raise DojoWorkerForwardError("worker smoke result cannot grant promotion")
        terminal_resolved = item["terminal_resolved"]
        if not isinstance(terminal_resolved, bool):
            raise DojoWorkerForwardError("terminal_resolved must be boolean")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "intrabar": intrabar,
                "status": item["status"],
                "ledger_sha256": _sha(item["ledger_sha256"], "ledger_sha256"),
                "score_receipt_sha256": _sha(
                    item["score_receipt_sha256"], "score_receipt_sha256"
                ),
                "entries": _integer(item["entries"], "entries", minimum=0),
                "resolved_exits": _integer(
                    item["resolved_exits"], "resolved_exits", minimum=0
                ),
                "terminal_net_jpy": _number(
                    item["terminal_net_jpy"], "terminal_net_jpy"
                ),
                "calendar_30d_multiple": _number(
                    item["calendar_30d_multiple"],
                    "calendar_30d_multiple",
                    minimum=0,
                ),
                "margin_closeouts": _integer(
                    item["margin_closeouts"], "margin_closeouts", minimum=0
                ),
                "terminal_resolved": terminal_resolved,
                "promotion_eligible": False,
            }
        )
    if seen != expected_cells:
        raise DojoWorkerForwardError("result manifest is incomplete")
    normalized.sort(key=lambda item: (item["candidate_id"], item["intrabar"]))
    return {
        "precommit_sha256": lock["precommit_sha256"],
        "window_start_utc": lock["window"]["start_utc"],
        "window_end_utc": lock["window"]["end_utc"],
        "results": normalized,
    }


def _validate_day_chain(
    values: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DojoWorkerForwardError("day seals must be a sequence")
    if len(values) != lock["window"]["calendar_days"]:
        raise DojoWorkerForwardError("all daily source seals are required")
    seals: list[dict[str, Any]] = []
    previous_sha = start_receipt["start_receipt_sha256"]
    for ordinal, value in enumerate(values, start=1):
        seal = validate_day_seal(value, lock, start_receipt, expected_ordinal=ordinal)
        if seal["previous_receipt_sha256"] != previous_sha:
            raise DojoWorkerForwardError("daily source seal chain is broken")
        previous_sha = seal["day_seal_sha256"]
        seals.append(seal)
    return seals


def _economic_result_passes(value: Mapping[str, Any]) -> bool:
    return bool(
        value["status"] == "PASS_POSITIVE_RESOLVED_BALANCE"
        and value["entries"] > 0
        and value["resolved_exits"] == value["entries"]
        and value["terminal_net_jpy"] > 0
        and value["margin_closeouts"] == 0
        and value["terminal_resolved"] is True
        and value["promotion_eligible"] is False
    )


def _validate_window(value: Any) -> dict[str, str]:
    source = _mapping(value, "window")
    allowed = {"start_utc", "end_utc", "calendar_days"}
    if set(source) not in ({"start_utc", "end_utc"}, allowed):
        raise DojoWorkerForwardError("window schema mismatch")
    start = _parse_utc(source["start_utc"], "window.start_utc")
    end = _parse_utc(source["end_utc"], "window.end_utc")
    if end <= start:
        raise DojoWorkerForwardError("window end must follow start")
    if "calendar_days" in source and source["calendar_days"] != (end - start).days:
        raise DojoWorkerForwardError("window calendar_days mismatch")
    return {"start_utc": _iso(start), "end_utc": _iso(end)}


def _authority() -> dict[str, Any]:
    return {
        "read_only": True,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "promotion_eligible": False,
    }


def _attestations() -> dict[str, Any]:
    return {
        "local_hash_chain_verified": True,
        "external_monotonic_witness_verified": False,
        "independent_market_source_attestation_verified": False,
        "self_attested_only": True,
        "evidence_tier": EVIDENCE_TIER,
    }


def _status(
    state: str,
    blockers: list[str],
    *,
    precommit: Mapping[str, Any] | None = None,
    sealed_days: int = 0,
) -> dict[str, Any]:
    return {
        "contract": STATUS_CONTRACT,
        "state": state,
        "experiment_id": precommit.get("experiment_id") if precommit else None,
        "sealed_days": sealed_days,
        "expected_days": precommit["window"]["calendar_days"] if precommit else 0,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "evidence_tier": EVIDENCE_TIER,
        "blockers": blockers,
    }


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoWorkerForwardError(f"{field} must be an object")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], field: str
) -> None:
    if set(value) != set(expected):
        raise DojoWorkerForwardError(f"{field} schema mismatch")


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DojoWorkerForwardError("artifact is not strict JSON") from exc


def _read_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DojoWorkerForwardError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            DojoWorkerForwardError(f"non-finite JSON number: {token}")
        ),
    )


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoWorkerForwardError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DojoWorkerForwardError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoWorkerForwardError(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise DojoWorkerForwardError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DojoWorkerForwardError(f"{field} is invalid")
    return value


def _text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise DojoWorkerForwardError(f"{field} is invalid")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DojoWorkerForwardError(f"{field} must be a lowercase SHA-256")
    return value


def _git_oid_value(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _GIT_OID.fullmatch(value):
        raise DojoWorkerForwardError(f"{field} must be a lowercase Git object id")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoWorkerForwardError(f"{field} is invalid")
    return value


def _number(
    value: Any,
    field: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoWorkerForwardError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DojoWorkerForwardError(f"{field} must be finite")
    if positive and result <= 0:
        raise DojoWorkerForwardError(f"{field} must be positive")
    if minimum is not None and result < minimum:
        raise DojoWorkerForwardError(f"{field} is below its minimum")
    return result


__all__ = [
    "DAY_SEAL_CONTRACT",
    "DojoWorkerForwardError",
    "EVIDENCE_TIER",
    "FINAL_CONTRACT",
    "PRECOMMIT_CONTRACT",
    "START_CONTRACT",
    "audit_lifecycle",
    "build_day_seal",
    "build_final_receipt",
    "build_precommit",
    "build_start_receipt",
    "canonical_sha256",
    "validate_day_seal",
    "validate_precommit",
    "validate_result_manifest",
    "validate_start_receipt",
    "write_new_json",
]
