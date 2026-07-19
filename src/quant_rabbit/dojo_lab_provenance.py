"""Fail-closed provenance primitives for DOJO strategy experiments.

The historical DOJO artifacts were useful for hypothesis generation, but
their mutable output paths, reused holdouts, and incomplete terminal scoring
make them ineligible for promotion.  This module provides the small set of
rules shared by the lab runners and worker bots:

* strategy ownership follows an order into its resulting trade and is written
  into the virtual-broker ledger;
* trial directories and result files are create-once;
* TRAIN, VAL, and FINAL windows are chronological and disjoint, and a screened
  window cannot be recycled as a holdout;
* a score is valid only after every position and resting order is resolved;
* every ledger byte, replay manifest input, and hardened cost is authenticated;
* monthly normalization uses elapsed calendar time, never active exit days.

Nothing in this module grants live permission.  Historical reruns remain
hypothesis-only even when their economic gate is positive.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from quant_rabbit.virtual_broker import (
    VirtualBroker,
    VirtualBrokerError,
    _strict_json_loads,
)


UTC = timezone.utc
CALENDAR_MONTH_DAYS = 30.0
LEGACY_CONTAMINATION_BLOCKERS = (
    "LEGACY_DOJO_RESULTS_MUTABLE_OR_UNTRACKED",
    "LEGACY_DOJO_HOLDOUT_REUSED_AFTER_SCREENING",
    "LEGACY_DOJO_MULTIPLE_TESTING_300_PLUS_RUNS",
)
_RUN_ID_RE = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{8}")
_TRIAL_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_REGISTRY_ATTR = "_dojo_strategy_ownership_registry"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ZERO_SHA256 = "0" * 64


def _strategy_affecting_event(event: str) -> bool:
    return bool(
        event.startswith("EXIT")
        or event
        in {
            "ORDER_REJECTED_INSUFFICIENT_MARGIN",
            "ORDER_REJECTED_CONCURRENCY_CAP",
            "FILL_MARKET",
            "ORDER_LIMIT",
            "ORDER_STOP",
            "ORDER_CANCEL",
            "ORDER_CANCEL_CONCURRENCY_CAP",
            "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
            "FILL_LIMIT",
            "CLOSE",
            "SET_EXIT",
            "MARGIN_CLOSEOUT",
        }
    )


class DojoLabProvenanceError(ValueError):
    """The experiment cannot produce trustworthy evidence."""


class StrategyOwnershipError(VirtualBrokerError):
    """A strategy attempted to mutate another strategy's order or trade."""


@dataclass(frozen=True)
class WindowSpec:
    """One half-open UTC evaluation interval."""

    role: str
    start: datetime
    end: datetime

    @classmethod
    def from_pair(cls, role: str, value: Sequence[str]) -> "WindowSpec":
        if len(value) != 2:
            raise DojoLabProvenanceError(f"{role} window must contain start/end")
        start = _parse_utc(value[0], field=f"{role}.start")
        end = _parse_utc(value[1], field=f"{role}.end")
        if start >= end:
            raise DojoLabProvenanceError(f"{role} window must have start < end")
        return cls(role=role, start=start, end=end)

    @property
    def calendar_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86_400.0

    def to_dict(self) -> dict[str, str]:
        return {
            "start_utc": self.start.isoformat().replace("+00:00", "Z"),
            "end_utc": self.end.isoformat().replace("+00:00", "Z"),
        }


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DojoLabProvenanceError(f"{field} must be a timestamp string")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise DojoLabProvenanceError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed


def _overlaps(left: WindowSpec, right: WindowSpec) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DojoLabProvenanceError(f"{field} must be a lowercase SHA-256")
    return value


def validate_window_plan(
    windows: Mapping[str, Sequence[str]],
    *,
    screened_windows: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate a three-stage plan and return its canonical representation.

    TRAIN may be used repeatedly for invention.  VAL and FINAL are holdouts;
    neither may overlap any interval that already participated in screening.
    All declared experiment windows must be mutually disjoint.
    """

    required = {"TRAIN", "VAL", "FINAL"}
    if set(windows) != required:
        raise DojoLabProvenanceError(f"window roles must be exactly {sorted(required)}")
    parsed = {role: WindowSpec.from_pair(role, windows[role]) for role in required}
    ordered = [parsed[role] for role in ("TRAIN", "VAL", "FINAL")]
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if _overlaps(left, right):
                raise DojoLabProvenanceError(
                    f"evaluation windows overlap: {left.role}/{right.role}"
                )
    if not (
        parsed["TRAIN"].end <= parsed["VAL"].start
        and parsed["VAL"].end <= parsed["FINAL"].start
    ):
        raise DojoLabProvenanceError(
            "evaluation windows must be chronological: TRAIN <= VAL <= FINAL"
        )

    screened: dict[str, WindowSpec] = {}
    for label, value in (screened_windows or {}).items():
        spec = WindowSpec.from_pair(f"SCREEN:{label}", value)
        screened[label] = spec
        for holdout_role in ("VAL", "FINAL"):
            if _overlaps(spec, parsed[holdout_role]):
                raise DojoLabProvenanceError(
                    f"screened window {label} overlaps holdout {holdout_role}"
                )

    return {
        "policy": "CHRONOLOGICAL_DISJOINT_TRAIN_VAL_FINAL_SCREENED_WINDOWS_EXCLUDED_FROM_HOLDOUT",
        "windows": {role: parsed[role].to_dict() for role in ("TRAIN", "VAL", "FINAL")},
        "screened_windows": {
            label: screened[label].to_dict() for label in sorted(screened)
        },
    }


def reserve_window_plan(
    registry_path: Path | None,
    *,
    run_id: str,
    experiment_id: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably reserve VAL/FINAL, or explicitly downgrade to hypothesis-only.

    The registry is deliberately caller-supplied and global.  A registry under
    an individual run directory cannot detect holdout reuse by another run.
    Records are append-only, hash-chained, locked across processes, and fsynced.
    """

    if registry_path is None:
        return {
            "status": "GLOBAL_REGISTRY_ABSENT",
            "promotion_eligible": False,
            "promotion_blocker": "GLOBAL_WINDOW_RESERVATION_ABSENT",
            "local_reservation_verified": False,
            "external_monotonicity_attested": False,
            "registry_path": None,
        }
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise DojoLabProvenanceError("run_id has invalid format")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise DojoLabProvenanceError("experiment_id is required")
    windows = plan.get("windows")
    if not isinstance(windows, Mapping):
        raise DojoLabProvenanceError("canonical window plan is required")
    holdouts: dict[str, dict[str, str]] = {}
    holdout_specs: dict[str, WindowSpec] = {}
    for role in ("VAL", "FINAL"):
        row = windows.get(role)
        if not isinstance(row, Mapping):
            raise DojoLabProvenanceError(f"window plan is missing {role}")
        pair = (row.get("start_utc"), row.get("end_utc"))
        spec = WindowSpec.from_pair(role, pair)
        holdout_specs[role] = spec
        holdouts[role] = spec.to_dict()
    reserved_at = datetime.now(UTC)
    reserved_before_every_holdout = all(
        reserved_at < spec.start for spec in holdout_specs.values()
    )

    registry_path = registry_path.expanduser().resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - DOJO runs on POSIX
        raise DojoLabProvenanceError(
            "durable reservation requires POSIX locking"
        ) from exc

    with registry_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        previous_sha = _ZERO_SHA256
        prior_records: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise DojoLabProvenanceError(
                    f"reservation registry has a blank record at line {line_number}"
                )
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DojoLabProvenanceError(
                    f"invalid reservation registry JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or set(record) != {
                "seq",
                "reserved_at_utc",
                "experiment_id",
                "run_id",
                "window_plan_sha256",
                "holdouts",
                "prev_sha",
                "sha",
            }:
                raise DojoLabProvenanceError("reservation registry schema mismatch")
            if record["prev_sha"] != previous_sha:
                raise DojoLabProvenanceError("reservation registry hash chain broken")
            claimed_sha = _require_sha256(record["sha"], field="reservation.sha")
            body = {key: value for key, value in record.items() if key != "sha"}
            if claimed_sha != _canonical_sha256(body):
                raise DojoLabProvenanceError(
                    "reservation registry record hash mismatch"
                )
            previous_sha = claimed_sha
            prior_records.append(record)

        for record in prior_records:
            if record.get("run_id") == run_id:
                raise DojoLabProvenanceError("run_id already has a reservation")
            prior_holdouts = record.get("holdouts")
            if not isinstance(prior_holdouts, Mapping):
                raise DojoLabProvenanceError("reservation holdouts are malformed")
            for prior_role in ("VAL", "FINAL"):
                prior_row = prior_holdouts.get(prior_role)
                if not isinstance(prior_row, Mapping):
                    raise DojoLabProvenanceError("reservation holdout is incomplete")
                prior_spec = WindowSpec.from_pair(
                    f"RESERVED:{prior_role}",
                    (prior_row.get("start_utc"), prior_row.get("end_utc")),
                )
                for role, spec in holdout_specs.items():
                    if _overlaps(prior_spec, spec):
                        raise DojoLabProvenanceError(
                            "holdout already reserved: "
                            f"{record.get('experiment_id')}:{prior_role}/{role}"
                        )

        body = {
            "seq": len(prior_records) + 1,
            "reserved_at_utc": reserved_at.isoformat(),
            "experiment_id": experiment_id.strip(),
            "run_id": run_id,
            "window_plan_sha256": _canonical_sha256(plan),
            "holdouts": holdouts,
            "prev_sha": previous_sha,
        }
        record = {**body, "sha": _canonical_sha256(body)}
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return {
        "status": (
            "RESERVED"
            if reserved_before_every_holdout
            else "RESERVED_HISTORICAL_DIAGNOSTIC"
        ),
        "promotion_eligible": False,
        "promotion_blocker": (
            "EXTERNAL_MONOTONIC_RESERVATION_ATTESTATION_ABSENT"
            if reserved_before_every_holdout
            else "HOLDOUT_RESERVED_AFTER_START"
        ),
        "local_reservation_verified": reserved_before_every_holdout,
        "reserved_before_every_holdout": reserved_before_every_holdout,
        "historical_diagnostic_only": not reserved_before_every_holdout,
        "external_monotonicity_attested": False,
        "monotonicity_status": "LOCAL_HASH_CHAIN_NOT_EXTERNALLY_MONOTONIC",
        "registry_path": str(registry_path),
        "reservation_sha256": record["sha"],
        "window_plan_sha256": body["window_plan_sha256"],
        "run_id": run_id,
        "experiment_id": experiment_id.strip(),
        "holdouts": holdouts,
    }


def _reservation_evidence_is_durable(
    evidence: Mapping[str, Any] | None,
    *,
    window_role: str,
    window: WindowSpec,
) -> bool:
    """Re-authenticate a reservation from its global append-only registry."""

    if not evidence or evidence.get("status") != "RESERVED":
        return False
    registry_raw = evidence.get("registry_path")
    target_sha = evidence.get("reservation_sha256")
    if (
        not isinstance(registry_raw, str)
        or not isinstance(target_sha, str)
        or _SHA256_RE.fullmatch(target_sha) is None
    ):
        return False
    registry_path = Path(registry_raw)
    if not registry_path.is_file():
        return False
    try:
        import fcntl

        with registry_path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            previous_sha = _ZERO_SHA256
            target: Mapping[str, Any] | None = None
            for line in handle:
                if not line.strip():
                    return False
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or record.get("prev_sha") != previous_sha
                ):
                    return False
                claimed_sha = record.get("sha")
                if (
                    not isinstance(claimed_sha, str)
                    or _SHA256_RE.fullmatch(claimed_sha) is None
                ):
                    return False
                body = {key: value for key, value in record.items() if key != "sha"}
                if _canonical_sha256(body) != claimed_sha:
                    return False
                previous_sha = claimed_sha
                if claimed_sha == target_sha:
                    target = record
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if target is None:
        return False
    if target.get("run_id") != evidence.get("run_id") or target.get(
        "experiment_id"
    ) != evidence.get("experiment_id"):
        return False
    if target.get("window_plan_sha256") != evidence.get("window_plan_sha256"):
        return False
    if window_role in {"VAL", "FINAL"}:
        holdouts = target.get("holdouts")
        row = holdouts.get(window_role) if isinstance(holdouts, Mapping) else None
        if not isinstance(row, Mapping) or dict(row) != window.to_dict():
            return False
    return True


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{stamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"


def create_run_root(out_root: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Create one immutable run namespace; an existing id is an error."""

    resolved_id = run_id or new_run_id()
    if _RUN_ID_RE.fullmatch(resolved_id) is None:
        raise DojoLabProvenanceError("run_id has invalid format")
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_root = runs_root / resolved_id
    try:
        run_root.mkdir()
    except FileExistsError as exc:
        raise DojoLabProvenanceError(f"run_id already exists: {resolved_id}") from exc
    return resolved_id, run_root


def create_trial_dir(run_root: Path, trial_key: str) -> Path:
    """Create a trial exactly once inside an already-created run root."""

    if _TRIAL_KEY_RE.fullmatch(trial_key) is None:
        raise DojoLabProvenanceError("trial_key has invalid format")
    trials_root = run_root / "trials"
    trials_root.mkdir(exist_ok=True)
    trial = trials_root / trial_key
    try:
        trial.mkdir()
    except FileExistsError as exc:
        raise DojoLabProvenanceError(f"trial already exists: {trial_key}") from exc
    (trial / "inbox").mkdir()
    return trial


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a result once; retries must use a new run id."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise DojoLabProvenanceError(f"result already exists: {path.name}") from exc


def _obsolete_score_session_ledger(
    ledger_path: Path,
    *,
    start_balance_jpy: float,
    window_role: str,
    window: Sequence[str],
    intrabar: str,
    legacy_contaminated: bool,
) -> dict[str, Any]:
    """Score a completed session from terminal executable equity.

    `SESSION_STOP.account.equity_jpy` marks every still-open position at the
    final executable bid/ask.  It therefore participates in the score instead
    of disappearing behind realized-only accounting.
    """

    if not math.isfinite(start_balance_jpy) or start_balance_jpy <= 0:
        raise DojoLabProvenanceError("start balance must be positive and finite")
    spec = WindowSpec.from_pair(window_role, window)
    if intrabar not in {"OHLC", "OLHC"}:
        raise DojoLabProvenanceError("intrabar must be OHLC or OLHC")
    if not ledger_path.is_file():
        raise DojoLabProvenanceError("trial ledger is missing")

    entry_count = 0
    resolved_exit_count = 0
    wins = 0
    closeouts = 0
    realized_from_events = 0.0
    terminal_account: Mapping[str, Any] | None = None
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DojoLabProvenanceError(
                    f"invalid ledger JSON at line {line_number}"
                ) from exc
            event = record.get("event")
            payload = (
                record.get("payload") if isinstance(record.get("payload"), dict) else {}
            )
            if event in {"FILL_MARKET", "FILL_LIMIT"}:
                entry_count += 1
            if isinstance(event, str) and event.startswith(("EXIT", "CLOSE", "MARGIN")):
                value = payload.get("pl_jpy")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    realized_from_events += float(value)
                    resolved_exit_count += 1
                    wins += int(float(value) > 0)
            if event == "MARGIN_CLOSEOUT":
                closeouts += 1
            if event == "SESSION_STOP" and isinstance(payload.get("account"), dict):
                terminal_account = payload["account"]

    if terminal_account is None:
        raise DojoLabProvenanceError("SESSION_STOP terminal account is missing")
    try:
        balance = float(terminal_account["balance_jpy"])
        equity = float(terminal_account["equity_jpy"])
        open_positions = int(terminal_account["open_positions"])
        resting_orders = int(terminal_account["resting_orders"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoLabProvenanceError("terminal account is incomplete") from exc
    if not all(math.isfinite(value) for value in (balance, equity)):
        raise DojoLabProvenanceError("terminal balance/equity must be finite")

    terminal_net = equity - start_balance_jpy
    realized_net = balance - start_balance_jpy
    terminal_unrealized = equity - balance
    total_multiple = max(equity, 0.0) / start_balance_jpy
    monthly_multiple = (
        total_multiple ** (CALENDAR_MONTH_DAYS / spec.calendar_days)
        if total_multiple > 0
        else 0.0
    )
    blockers: list[str] = []
    if entry_count == 0:
        status = "INVALID_ZERO_TRADES"
        blockers.append("ZERO_TRADES")
    elif closeouts:
        status = "FAIL_MARGIN_CLOSEOUT"
        blockers.append("MARGIN_CLOSEOUT_OCCURRED")
    elif terminal_net <= 0:
        status = "FAIL_NON_POSITIVE_TERMINAL_EQUITY"
        blockers.append("NON_POSITIVE_TERMINAL_EQUITY")
    else:
        status = "PASS_POSITIVE_TERMINAL_EQUITY"
    if legacy_contaminated:
        blockers.extend(LEGACY_CONTAMINATION_BLOCKERS)

    economic_gate_passed = entry_count > 0 and terminal_net > 0 and closeouts == 0
    promotion_eligible = economic_gate_passed and not legacy_contaminated
    return {
        "status": status,
        "economic_gate_passed": economic_gate_passed,
        "promotion_eligible": promotion_eligible,
        "evidence_tier": "PROMOTION_CANDIDATE"
        if promotion_eligible
        else "HYPOTHESIS_ONLY",
        "promotion_blockers": list(dict.fromkeys(blockers)),
        "window_role": window_role,
        "window": spec.to_dict(),
        "calendar_days": round(spec.calendar_days, 8),
        "intrabar": intrabar,
        "entries": entry_count,
        "resolved_exits": resolved_exit_count,
        "wins": wins,
        "win_rate_resolved": round(wins / resolved_exit_count, 6)
        if resolved_exit_count
        else None,
        "realized_event_sum_jpy": round(realized_from_events, 2),
        "realized_net_jpy": round(realized_net, 2),
        "terminal_unrealized_jpy": round(terminal_unrealized, 2),
        "terminal_net_jpy": round(terminal_net, 2),
        "final_balance_jpy": round(balance, 2),
        "final_equity_jpy": round(equity, 2),
        "open_positions_marked": open_positions,
        "resting_orders_at_end": resting_orders,
        "total_multiple": round(total_multiple, 8),
        "calendar_30d_multiple": round(monthly_multiple, 8),
        "monthly_multiple": round(monthly_multiple, 8),
        "monthly_multiple_basis": "30_CALENDAR_DAYS_FROM_TERMINAL_EXECUTABLE_EQUITY",
        "margin_closeouts": closeouts,
    }


def _obsolete_combine_intrabar_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require both paths and make the lower terminal result authoritative."""

    by_path = {str(row.get("intrabar")): row for row in results}
    if set(by_path) != {"OHLC", "OLHC"} or len(results) != 2:
        raise DojoLabProvenanceError("every gate requires exactly OHLC and OLHC")
    if any("terminal_net_jpy" not in row for row in by_path.values()):
        raise DojoLabProvenanceError("intrabar result is not scoreable")
    pessimistic_path = min(
        ("OHLC", "OLHC"), key=lambda path: float(by_path[path]["terminal_net_jpy"])
    )
    pessimistic = by_path[pessimistic_path]
    gate_passed = all(bool(row.get("economic_gate_passed")) for row in by_path.values())
    promotion_eligible = gate_passed and all(
        bool(row.get("promotion_eligible")) for row in by_path.values()
    )
    blockers: list[str] = []
    for path in ("OHLC", "OLHC"):
        blockers.extend(
            str(item) for item in by_path[path].get("promotion_blockers", [])
        )
    return {
        "status": "PASS_BOTH_INTRABAR_PATHS" if gate_passed else "FAIL_INTRABAR_GATE",
        "gate_passed": gate_passed,
        "promotion_eligible": promotion_eligible,
        "evidence_tier": "PROMOTION_CANDIDATE"
        if promotion_eligible
        else "HYPOTHESIS_ONLY",
        "promotion_blockers": list(dict.fromkeys(blockers)),
        "pessimistic_intrabar": pessimistic_path,
        "pessimistic_terminal_net_jpy": pessimistic["terminal_net_jpy"],
        "pessimistic_calendar_30d_multiple": pessimistic["calendar_30d_multiple"],
        "intrabar_terminal_net_jpy": {
            path: by_path[path]["terminal_net_jpy"] for path in ("OHLC", "OLHC")
        },
        "intrabar_entries": {
            path: by_path[path]["entries"] for path in ("OHLC", "OLHC")
        },
        "intrabar_margin_closeouts": {
            path: by_path[path]["margin_closeouts"] for path in ("OHLC", "OLHC")
        },
        "intrabar_terminal_resolution": {
            path: bool(by_path[path].get("terminal_resolution_verified"))
            for path in ("OHLC", "OLHC")
        },
    }


def score_session_ledger(
    ledger_path: Path,
    *,
    start_balance_jpy: float,
    window_role: str,
    window: Sequence[str],
    intrabar: str,
    legacy_contaminated: bool,
    expected_pairs: Sequence[str],
    expected_granularity: str,
    expected_bot_bar: str,
    expected_period_end_settlement: bool,
    expected_slippage_pips: float,
    expected_financing_pips_per_day: float,
    expected_bot_module_path: Path,
    expected_bot_module_sha256: str,
    expected_bot_dependency_sha256: Mapping[str, str],
    expected_strategy_owner_id: str,
    expected_bot_config_sha256: str,
    expected_bot_config_length: int,
    reservation_evidence: Mapping[str, Any] | None = None,
    expected_max_concurrent_per_pair: int | None = None,
    expected_global_max_concurrent: int | None = None,
) -> dict[str, Any]:
    """Authenticate and score one fully resolved virtual-market session."""

    if not math.isfinite(start_balance_jpy) or start_balance_jpy <= 0:
        raise DojoLabProvenanceError("start balance must be positive and finite")
    spec = WindowSpec.from_pair(window_role, window)
    if intrabar not in {"OHLC", "OLHC"}:
        raise DojoLabProvenanceError("intrabar must be OHLC or OLHC")
    if not ledger_path.is_file():
        raise DojoLabProvenanceError("trial ledger is missing")
    pairs = sorted(set(expected_pairs))
    if not pairs or any(not isinstance(pair, str) or not pair for pair in pairs):
        raise DojoLabProvenanceError("expected pairs are required")
    if expected_granularity not in {"M1", "S5"}:
        raise DojoLabProvenanceError("expected granularity must be M1 or S5")
    if expected_bot_bar not in {"feed", "M1"}:
        raise DojoLabProvenanceError("expected bot bar must be feed or M1")
    if not isinstance(expected_period_end_settlement, bool):
        raise DojoLabProvenanceError("expected period-end settlement must be boolean")
    for field, value in (
        ("slippage", expected_slippage_pips),
        ("financing", expected_financing_pips_per_day),
    ):
        if not math.isfinite(value) or value <= 0:
            raise DojoLabProvenanceError(
                f"hardened {field} cost must be positive and finite"
            )
    expected_module_sha = _require_sha256(
        expected_bot_module_sha256, field="expected bot module sha"
    )
    expected_dependencies: dict[str, str] = {}
    for path, digest in expected_bot_dependency_sha256.items():
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in expected_dependencies
        ):
            raise DojoLabProvenanceError("expected bot dependency path is unsafe")
        expected_dependencies[path] = _require_sha256(
            digest, field=f"expected bot dependency {path}"
        )
    if not expected_dependencies:
        raise DojoLabProvenanceError("expected bot dependency closure is empty")
    if (
        not isinstance(expected_strategy_owner_id, str)
        or not expected_strategy_owner_id
        or len(expected_strategy_owner_id) > 128
        or any(ord(char) < 33 or ord(char) > 126 for char in expected_strategy_owner_id)
    ):
        raise DojoLabProvenanceError("expected strategy owner id is invalid")
    expected_config_sha = _require_sha256(
        expected_bot_config_sha256, field="expected bot config sha"
    )
    if expected_bot_config_length <= 0:
        raise DojoLabProvenanceError("expected bot config length must be positive")
    if (expected_max_concurrent_per_pair is None) != (
        expected_global_max_concurrent is None
    ):
        raise DojoLabProvenanceError(
            "expected owner concurrency caps must be declared together"
        )
    concurrency_caps_supplied = expected_max_concurrent_per_pair is not None
    if concurrency_caps_supplied and (
        isinstance(expected_max_concurrent_per_pair, bool)
        or not isinstance(expected_max_concurrent_per_pair, int)
        or expected_max_concurrent_per_pair <= 0
        or isinstance(expected_global_max_concurrent, bool)
        or not isinstance(expected_global_max_concurrent, int)
        or expected_global_max_concurrent <= 0
    ):
        raise DojoLabProvenanceError(
            "expected owner concurrency caps must be positive integers"
        )

    entry_count = 0
    resolved_exit_count = 0
    wins = 0
    closeouts = 0
    realized_from_events = 0.0
    terminal_account: Mapping[str, Any] | None = None
    records: list[dict[str, Any]] = []
    previous_sha = _ZERO_SHA256
    active_owner_trades: dict[str, tuple[str, float]] = {}
    seen_owner_trade_ids: set[str] = set()
    resolved_owner_trade_count = 0
    observed_peak_global = 0
    observed_peak_by_pair = {pair: 0 for pair in pairs}

    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DojoLabProvenanceError(
                    f"blank ledger record at line {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DojoLabProvenanceError(
                    f"invalid ledger JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or set(record) != {
                "ts_utc",
                "event",
                "payload",
                "prev_sha",
                "sha",
            }:
                raise DojoLabProvenanceError(
                    f"ledger schema mismatch at line {line_number}"
                )
            _parse_utc(record["ts_utc"], field=f"ledger[{line_number}].ts_utc")
            if not isinstance(record["event"], str) or not isinstance(
                record["payload"], dict
            ):
                raise DojoLabProvenanceError(
                    f"ledger event/payload malformed at line {line_number}"
                )
            if record["prev_sha"] != previous_sha:
                raise DojoLabProvenanceError(
                    f"ledger hash chain broken at line {line_number}"
                )
            claimed_sha = _require_sha256(
                record["sha"], field=f"ledger[{line_number}].sha"
            )
            body = {key: value for key, value in record.items() if key != "sha"}
            if claimed_sha != _canonical_sha256(body):
                raise DojoLabProvenanceError(
                    f"ledger record hash mismatch at line {line_number}"
                )
            previous_sha = claimed_sha
            records.append(record)

            event = record["event"]
            payload = record["payload"]
            if _strategy_affecting_event(event):
                if payload.get("strategy_owner_id") != expected_strategy_owner_id:
                    raise DojoLabProvenanceError(
                        "strategy-affecting ledger event has missing or mismatched owner"
                    )
            if event in {"FILL_MARKET", "FILL_LIMIT"}:
                entry_count += 1
                try:
                    event_slippage = float(payload["slippage_pips"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise DojoLabProvenanceError(
                        "entry is missing hardened slippage evidence"
                    ) from exc
                if not math.isclose(
                    event_slippage,
                    expected_slippage_pips,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise DojoLabProvenanceError(
                        "entry hardened slippage does not match the manifest"
                    )
                if concurrency_caps_supplied:
                    trade_id = payload.get("trade_id")
                    pair = payload.get("pair")
                    if (
                        not isinstance(trade_id, str)
                        or not trade_id
                        or trade_id in seen_owner_trade_ids
                    ):
                        raise DojoLabProvenanceError(
                            "owner concurrency reconstruction has an invalid fill trade id"
                        )
                    if not isinstance(pair, str) or pair not in pairs:
                        raise DojoLabProvenanceError(
                            "owner concurrency reconstruction has an invalid fill pair"
                        )
                    raw_units = payload.get("units")
                    try:
                        fill_units = float(raw_units)
                    except (TypeError, ValueError) as exc:
                        raise DojoLabProvenanceError(
                            "owner concurrency reconstruction has invalid fill units"
                        ) from exc
                    if (
                        isinstance(raw_units, bool)
                        or not math.isfinite(fill_units)
                        or fill_units <= 0
                    ):
                        raise DojoLabProvenanceError(
                            "owner concurrency reconstruction has invalid fill units"
                        )
                    active_pair = (
                        sum(
                            active_pair == pair
                            for active_pair, _ in active_owner_trades.values()
                        )
                        + 1
                    )
                    active_global = len(active_owner_trades) + 1
                    if active_pair > expected_max_concurrent_per_pair:
                        raise DojoLabProvenanceError(
                            "owner per-pair concurrency cap exceeded in ledger"
                        )
                    if active_global > expected_global_max_concurrent:
                        raise DojoLabProvenanceError(
                            "owner global concurrency cap exceeded in ledger"
                        )
                    active_owner_trades[trade_id] = (pair, fill_units)
                    seen_owner_trade_ids.add(trade_id)
                    observed_peak_by_pair[pair] = max(
                        observed_peak_by_pair[pair], active_pair
                    )
                    observed_peak_global = max(observed_peak_global, active_global)
            is_exit = event.startswith("EXIT") or event in {
                "CLOSE",
                "MARGIN_CLOSEOUT",
            }
            if is_exit:
                if concurrency_caps_supplied:
                    trade_id = payload.get("trade_id")
                    if (
                        not isinstance(trade_id, str)
                        or not trade_id
                        or trade_id not in active_owner_trades
                    ):
                        raise DojoLabProvenanceError(
                            "owner concurrency reconstruction has an unmatched exit"
                        )
                    pair, remaining_units = active_owner_trades[trade_id]
                    if event == "CLOSE":
                        raw_close_units = payload.get("units")
                        try:
                            close_units = float(raw_close_units)
                        except (TypeError, ValueError) as exc:
                            raise DojoLabProvenanceError(
                                "owner concurrency reconstruction has invalid close units"
                            ) from exc
                        if (
                            isinstance(raw_close_units, bool)
                            or not math.isfinite(close_units)
                            or close_units <= 0
                        ):
                            raise DojoLabProvenanceError(
                                "owner concurrency reconstruction has invalid close units"
                            )
                        if close_units > remaining_units:
                            raise DojoLabProvenanceError(
                                "owner concurrency reconstruction over-closes a trade"
                            )
                        remaining_units -= close_units
                        if remaining_units > 0:
                            active_owner_trades[trade_id] = (pair, remaining_units)
                        else:
                            del active_owner_trades[trade_id]
                            resolved_owner_trade_count += 1
                    else:
                        del active_owner_trades[trade_id]
                        resolved_owner_trade_count += 1
                try:
                    value = float(payload["pl_jpy"])
                    financing = float(payload["financing_jpy"])
                    exit_slippage = float(payload["slippage_pips"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise DojoLabProvenanceError(
                        "terminal exit cost evidence is incomplete"
                    ) from exc
                if not all(math.isfinite(item) for item in (value, financing)):
                    raise DojoLabProvenanceError("terminal exit values must be finite")
                if financing < 0:
                    raise DojoLabProvenanceError(
                        "terminal financing cannot be negative"
                    )
                if not math.isclose(
                    exit_slippage,
                    expected_slippage_pips,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise DojoLabProvenanceError(
                        "terminal exit is missing hardened exit slippage"
                    )
                realized_from_events += value
                resolved_exit_count += 1
                wins += int(value > 0)
            if event == "MARGIN_CLOSEOUT":
                closeouts += 1
            if event == "SESSION_STOP" and isinstance(payload.get("account"), dict):
                terminal_account = payload["account"]

    if not records:
        raise DojoLabProvenanceError("trial ledger is empty")
    if sum(row["event"] == "SESSION_START" for row in records) != 1 or (
        records[0]["event"] != "SESSION_START"
    ):
        raise DojoLabProvenanceError(
            "ledger requires exactly one initial SESSION_START"
        )
    if sum(row["event"] == "SESSION_STOP" for row in records) != 1 or (
        records[-1]["event"] != "SESSION_STOP"
    ):
        raise DojoLabProvenanceError(
            "ledger requires exactly one terminal SESSION_STOP"
        )

    start_payload = records[0]["payload"]
    if start_payload.get("contract") != "QR_VIRTUAL_MARKET_SESSION_V1":
        raise DojoLabProvenanceError("SESSION_START contract mismatch")
    if start_payload.get("feed") != "replay":
        raise DojoLabProvenanceError("DOJO evidence requires replay feed")
    payload_pairs = sorted(
        pair.strip()
        for pair in str(start_payload.get("pairs", "")).split(",")
        if pair.strip()
    )
    if payload_pairs != pairs:
        raise DojoLabProvenanceError("SESSION_START pair set mismatch")
    try:
        declared_start_balance = float(start_payload["balance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoLabProvenanceError("SESSION_START balance is missing") from exc
    if not math.isclose(
        declared_start_balance, start_balance_jpy, rel_tol=0, abs_tol=1e-8
    ):
        raise DojoLabProvenanceError("SESSION_START balance mismatch")
    if start_payload.get("order_authority") != "NONE":
        raise DojoLabProvenanceError("virtual session has unexpected order authority")

    manifest = start_payload.get("reproducibility_manifest")
    if not isinstance(manifest, dict):
        raise DojoLabProvenanceError("SESSION_START manifest is missing")
    manifest_sha = _require_sha256(
        manifest.get("manifest_sha256"), field="manifest sha"
    )
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest_sha != _canonical_sha256(manifest_body):
        raise DojoLabProvenanceError("reproducibility manifest hash mismatch")
    if start_payload.get("reproducibility_manifest_sha256") != manifest_sha:
        raise DojoLabProvenanceError("SESSION_START manifest digest mismatch")
    if manifest.get("schema") != "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1":
        raise DojoLabProvenanceError("reproducibility manifest schema mismatch")
    if manifest.get("order_authority") != "NONE":
        raise DojoLabProvenanceError("manifest order authority mismatch")
    if manifest.get("resume_snapshot") is not None:
        raise DojoLabProvenanceError("fresh trial unexpectedly resumed a snapshot")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise DojoLabProvenanceError("manifest source binding is missing")
    for key in ("session_script_sha256", "virtual_broker_sha256"):
        _require_sha256(source.get(key), field=f"manifest.source.{key}")
    if not isinstance(source.get("git_head"), str) or not source["git_head"]:
        raise DojoLabProvenanceError("manifest git head is missing")

    replay = manifest.get("replay")
    if not isinstance(replay, dict):
        raise DojoLabProvenanceError("manifest replay binding is missing")
    if replay.get("feed") != "replay" or replay.get("pairs") != pairs:
        raise DojoLabProvenanceError("manifest replay feed/pairs mismatch")
    manifest_window = WindowSpec.from_pair(
        window_role, (replay.get("time_from"), replay.get("time_to"))
    )
    if manifest_window.to_dict() != spec.to_dict():
        raise DojoLabProvenanceError("manifest replay window mismatch")
    if replay.get("granularity") != expected_granularity:
        raise DojoLabProvenanceError("manifest granularity mismatch")
    if replay.get("bot_bar") != expected_bot_bar:
        raise DojoLabProvenanceError("manifest bot cadence mismatch")
    if replay.get("period_end_settlement") is not expected_period_end_settlement:
        raise DojoLabProvenanceError("manifest period-end settlement mismatch")
    if replay.get("intrabar") != intrabar:
        raise DojoLabProvenanceError("manifest intrabar path mismatch")

    costs = manifest.get("costs")
    if not isinstance(costs, dict):
        raise DojoLabProvenanceError("manifest cost binding is missing")
    try:
        manifest_slippage = float(costs["slippage_pips_per_fill"])
        manifest_financing = float(costs["financing_pips_per_day"])
        manifest_leverage = float(costs["leverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoLabProvenanceError("manifest costs are incomplete") from exc
    if not math.isclose(
        manifest_slippage, expected_slippage_pips, rel_tol=0, abs_tol=1e-12
    ) or not math.isclose(
        manifest_financing,
        expected_financing_pips_per_day,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise DojoLabProvenanceError("manifest hardened costs mismatch")
    if manifest_leverage <= 0 or not math.isfinite(manifest_leverage):
        raise DojoLabProvenanceError("manifest leverage is invalid")
    try:
        manifest_initial_balance = float(manifest["initial_balance_jpy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoLabProvenanceError("manifest initial balance is missing") from exc
    if not math.isclose(
        manifest_initial_balance, start_balance_jpy, rel_tol=0, abs_tol=1e-8
    ):
        raise DojoLabProvenanceError("manifest initial balance mismatch")

    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("root"), str):
        raise DojoLabProvenanceError("manifest corpus root is missing")
    corpus_sha = _require_sha256(
        corpus.get("corpus_sha256"), field="manifest.corpus.corpus_sha256"
    )
    corpus_body = {
        key: value for key, value in corpus.items() if key != "corpus_sha256"
    }
    if corpus_sha != _canonical_sha256(corpus_body):
        raise DojoLabProvenanceError("manifest corpus digest mismatch")
    shards = corpus.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DojoLabProvenanceError("manifest corpus shard set is empty")
    shard_paths: set[str] = set()
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise DojoLabProvenanceError("manifest corpus shard schema mismatch")
        path = shard["path"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in shard_paths
        ):
            raise DojoLabProvenanceError("manifest corpus shard path is unsafe")
        shard_paths.add(path)
        if not isinstance(shard["size_bytes"], int) or shard["size_bytes"] <= 0:
            raise DojoLabProvenanceError("manifest corpus shard size is invalid")
        _require_sha256(shard["sha256"], field=f"corpus.shards[{index}].sha256")

    bot = manifest.get("bot")
    if not isinstance(bot, dict) or bot.get("kind") != "custom_module":
        raise DojoLabProvenanceError("manifest bot must be a custom module")
    if (
        Path(str(bot.get("module_path"))).resolve()
        != expected_bot_module_path.resolve()
    ):
        raise DojoLabProvenanceError("manifest bot module path mismatch")
    if bot.get("module_sha256") != expected_module_sha:
        raise DojoLabProvenanceError("manifest bot module digest mismatch")
    if bot.get("class") != "Bot":
        raise DojoLabProvenanceError("manifest bot class mismatch")
    if bot.get("strategy_owner_id") != expected_strategy_owner_id:
        raise DojoLabProvenanceError("manifest strategy owner mismatch")
    if bot.get("dependency_sha256") != dict(sorted(expected_dependencies.items())):
        raise DojoLabProvenanceError("custom bot dependency closure mismatch")
    if bot.get("configuration_bindings") != {
        "DOJO_BOT_CONFIG": {
            "sha256": expected_config_sha,
            "length": expected_bot_config_length,
        }
    }:
        raise DojoLabProvenanceError("manifest bot configuration binding mismatch")

    settlement_records = [
        row for row in records if row["event"] == "PERIOD_END_SETTLEMENT"
    ]
    if expected_period_end_settlement:
        if len(settlement_records) != 1:
            raise DojoLabProvenanceError(
                "period-end settlement requires exactly one receipt"
            )
        settlement = settlement_records[0]["payload"]
        if (
            settlement.get("strategy_owner_id") != expected_strategy_owner_id
            or settlement.get("complete") is not True
            or settlement.get("errors") != []
        ):
            raise DojoLabProvenanceError("period-end settlement is incomplete")
    elif settlement_records:
        raise DojoLabProvenanceError("unexpected period-end settlement receipt")

    comparison_manifest = json.loads(json.dumps(manifest_body))
    comparison_manifest["replay"].pop("intrabar", None)
    intrabar_pair_manifest_sha = _canonical_sha256(comparison_manifest)

    if terminal_account is None:
        raise DojoLabProvenanceError("SESSION_STOP terminal account is missing")
    try:
        balance = float(terminal_account["balance_jpy"])
        equity = float(terminal_account["equity_jpy"])
        open_positions = int(terminal_account["open_positions"])
        resting_orders = int(terminal_account["resting_orders"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoLabProvenanceError("terminal account is incomplete") from exc
    if not all(math.isfinite(value) for value in (balance, equity)):
        raise DojoLabProvenanceError("terminal balance/equity must be finite")
    if concurrency_caps_supplied and len(active_owner_trades) != open_positions:
        raise DojoLabProvenanceError(
            "terminal open positions do not reconcile owner concurrency ledger"
        )

    terminal_resolved = open_positions == 0 and resting_orders == 0
    if terminal_resolved and not math.isclose(balance, equity, rel_tol=0, abs_tol=0.01):
        raise DojoLabProvenanceError(
            "resolved terminal account has inconsistent balance/equity"
        )
    entries_resolved = (
        resolved_owner_trade_count == entry_count
        if concurrency_caps_supplied
        else resolved_exit_count == entry_count
    )
    if terminal_resolved and not entries_resolved:
        raise DojoLabProvenanceError(
            "resolved terminal account does not reconcile entries and exits"
        )
    realized_net = balance - start_balance_jpy
    terminal_unrealized = equity - balance
    reconciliation_tolerance = max(0.05, resolved_exit_count * 0.01)
    if terminal_resolved and not math.isclose(
        realized_from_events,
        realized_net,
        rel_tol=0,
        abs_tol=reconciliation_tolerance,
    ):
        raise DojoLabProvenanceError("terminal balance does not reconcile exit ledger")
    terminal_net = realized_net
    total_multiple = max(balance, 0.0) / start_balance_jpy
    monthly_multiple = (
        total_multiple ** (CALENDAR_MONTH_DAYS / spec.calendar_days)
        if total_multiple > 0
        else 0.0
    )

    blockers: list[str] = []
    if entry_count == 0:
        status = "INVALID_ZERO_TRADES"
        blockers.append("ZERO_TRADES")
    elif not terminal_resolved:
        status = "INVALID_TERMINAL_EXPOSURE"
        blockers.append("OPEN_POSITION_OR_ORDER_AT_SESSION_STOP")
    elif closeouts:
        status = "FAIL_MARGIN_CLOSEOUT"
        blockers.append("MARGIN_CLOSEOUT_OCCURRED")
    elif terminal_net <= 0:
        status = "FAIL_NON_POSITIVE_RESOLVED_BALANCE"
        blockers.append("NON_POSITIVE_RESOLVED_BALANCE")
    else:
        status = "PASS_POSITIVE_RESOLVED_BALANCE"
    if legacy_contaminated:
        blockers.extend(LEGACY_CONTAMINATION_BLOCKERS)
    reservation_ok = _reservation_evidence_is_durable(
        reservation_evidence,
        window_role=window_role,
        window=spec,
    )
    if not reservation_ok:
        blockers.append(
            "GLOBAL_WINDOW_RESERVATION_ABSENT"
            if not reservation_evidence
            or reservation_evidence.get("status") == "GLOBAL_REGISTRY_ABSENT"
            else "GLOBAL_WINDOW_RESERVATION_UNVERIFIED"
        )
    else:
        blockers.append("EXTERNAL_MONOTONIC_RESERVATION_ATTESTATION_ABSENT")

    economic_gate_passed = (
        entry_count > 0
        and terminal_resolved
        and entries_resolved
        and terminal_net > 0
        and closeouts == 0
    )
    local_candidate_eligible = (
        economic_gate_passed and not legacy_contaminated and reservation_ok
    )
    # A local hash chain detects accidental edits and truncation from the
    # observed head, but its owner can still rewrite the complete history.
    # Until an external monotonic witness is integrated, this process cannot
    # honestly mint promotion evidence.
    promotion_eligible = False
    score = {
        "status": status,
        "economic_gate_passed": economic_gate_passed,
        "local_candidate_eligible": local_candidate_eligible,
        "promotion_eligible": promotion_eligible,
        "evidence_tier": "PROMOTION_CANDIDATE"
        if promotion_eligible
        else "HYPOTHESIS_ONLY",
        "promotion_blockers": list(dict.fromkeys(blockers)),
        "window_role": window_role,
        "window": spec.to_dict(),
        "calendar_days": round(spec.calendar_days, 8),
        "intrabar": intrabar,
        "ledger_record_count": len(records),
        "ledger_terminal_sha256": previous_sha,
        "reproducibility_manifest_sha256": manifest_sha,
        "intrabar_pair_manifest_sha256": intrabar_pair_manifest_sha,
        "corpus_manifest_sha256": _canonical_sha256(corpus),
        "corpus_shard_count": len(shards),
        "bot_module_sha256": expected_module_sha,
        "bot_config_sha256": expected_config_sha,
        "hardened_costs": {
            "slippage_pips_per_fill": manifest_slippage,
            "financing_pips_per_day": manifest_financing,
            "leverage": manifest_leverage,
        },
        "reservation_status": (
            reservation_evidence.get("status")
            if reservation_evidence
            else "GLOBAL_REGISTRY_ABSENT"
        ),
        "reservation_monotonicity_status": (
            reservation_evidence.get("monotonicity_status")
            if reservation_evidence
            else "ABSENT"
        ),
        "entries": entry_count,
        "resolved_exits": resolved_exit_count,
        "wins": wins,
        "win_rate_resolved": round(wins / resolved_exit_count, 6)
        if resolved_exit_count
        else None,
        "realized_event_sum_jpy": round(realized_from_events, 2),
        "realized_net_jpy": round(realized_net, 2),
        "terminal_unrealized_jpy": round(terminal_unrealized, 2),
        "terminal_net_jpy": round(terminal_net, 2),
        "final_balance_jpy": round(balance, 2),
        "final_equity_jpy": round(equity, 2),
        "open_positions_at_end": open_positions,
        "resting_orders_at_end": resting_orders,
        "terminal_resolution_verified": terminal_resolved,
        "total_multiple": round(total_multiple, 8),
        "calendar_30d_multiple": round(monthly_multiple, 8),
        "monthly_multiple": round(monthly_multiple, 8),
        "monthly_multiple_basis": "30_CALENDAR_DAYS_FROM_FULLY_RESOLVED_BALANCE",
        "margin_closeouts": closeouts,
    }
    if concurrency_caps_supplied:
        score["owner_concurrency"] = {
            "status": "VERIFIED_FROM_FILL_EXIT_LEDGER",
            "strategy_owner_id": expected_strategy_owner_id,
            "max_concurrent_per_pair": expected_max_concurrent_per_pair,
            "global_max_concurrent": expected_global_max_concurrent,
            "observed_peak_global": observed_peak_global,
            "observed_peak_by_pair": observed_peak_by_pair,
            "resolved_trade_count": resolved_owner_trade_count,
            "terminal_active_positions": len(active_owner_trades),
        }
    return score


def combine_intrabar_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require identical material evidence and both synthetic price paths."""

    by_path = {str(row.get("intrabar")): row for row in results}
    if set(by_path) != {"OHLC", "OLHC"} or len(results) != 2:
        raise DojoLabProvenanceError("every gate requires exactly OHLC and OLHC")
    if any("terminal_net_jpy" not in row for row in by_path.values()):
        raise DojoLabProvenanceError("intrabar result is not scoreable")
    comparison_digests = {
        str(row.get("intrabar_pair_manifest_sha256")) for row in by_path.values()
    }
    if len(comparison_digests) != 1 or "None" in comparison_digests:
        raise DojoLabProvenanceError(
            "OHLC/OLHC manifests differ outside the intrabar path"
        )
    pessimistic_path = min(
        ("OHLC", "OLHC"), key=lambda path: float(by_path[path]["terminal_net_jpy"])
    )
    pessimistic = by_path[pessimistic_path]
    gate_passed = all(bool(row.get("economic_gate_passed")) for row in by_path.values())
    promotion_eligible = gate_passed and all(
        bool(row.get("promotion_eligible")) for row in by_path.values()
    )
    blockers: list[str] = []
    for path in ("OHLC", "OLHC"):
        blockers.extend(
            str(item) for item in by_path[path].get("promotion_blockers", [])
        )
    return {
        "status": "PASS_BOTH_INTRABAR_PATHS" if gate_passed else "FAIL_INTRABAR_GATE",
        "gate_passed": gate_passed,
        "promotion_eligible": promotion_eligible,
        "evidence_tier": "PROMOTION_CANDIDATE"
        if promotion_eligible
        else "HYPOTHESIS_ONLY",
        "promotion_blockers": list(dict.fromkeys(blockers)),
        "intrabar_pair_manifest_sha256": next(iter(comparison_digests)),
        "pessimistic_intrabar": pessimistic_path,
        "pessimistic_terminal_net_jpy": pessimistic["terminal_net_jpy"],
        "pessimistic_calendar_30d_multiple": pessimistic["calendar_30d_multiple"],
        "intrabar_terminal_net_jpy": {
            path: by_path[path]["terminal_net_jpy"] for path in ("OHLC", "OLHC")
        },
        "intrabar_entries": {
            path: by_path[path]["entries"] for path in ("OHLC", "OLHC")
        },
        "intrabar_margin_closeouts": {
            path: by_path[path]["margin_closeouts"] for path in ("OHLC", "OLHC")
        },
    }


def canonical_strategy_owner_id(
    config: Mapping[str, Any], *, namespace: str = "lab", ordinal: int | None = None
) -> str:
    material = {
        key: value
        for key, value in config.items()
        if key not in {"strategy_owner_id", "owner_id"}
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    suffix = f":{ordinal}" if ordinal is not None else ""
    return f"{namespace}{suffix}:{digest}"


def owner_concurrency_caps_from_config(
    config: Mapping[str, Any],
) -> tuple[int, int]:
    """Resolve the exact owner caps consumed by ``bots/lab_bot.py``."""

    if not isinstance(config, Mapping):
        raise DojoLabProvenanceError("bot config must be a mapping")

    def positive_integer(name: str, value: object) -> int:
        if isinstance(value, bool):
            raise DojoLabProvenanceError(f"{name} must be a positive integer")
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DojoLabProvenanceError(f"{name} must be a positive integer") from exc
        if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
            raise DojoLabProvenanceError(f"{name} must be a positive integer")
        return parsed

    raw_pairs = config.get("pairs", ["USD_JPY"])
    if (
        isinstance(raw_pairs, (str, bytes))
        or not isinstance(raw_pairs, Sequence)
        or not raw_pairs
        or any(not isinstance(pair, str) or not pair for pair in raw_pairs)
        or len(set(raw_pairs)) != len(raw_pairs)
    ):
        raise DojoLabProvenanceError(
            "bot concurrency config requires a unique non-empty pair sequence"
        )

    legacy_pair_cap = config.get("max_concurrent")
    explicit_pair_cap = config.get("max_concurrent_per_pair")
    if legacy_pair_cap is not None and explicit_pair_cap is not None:
        if positive_integer("max_concurrent", legacy_pair_cap) != positive_integer(
            "max_concurrent_per_pair", explicit_pair_cap
        ):
            raise DojoLabProvenanceError(
                "max_concurrent and max_concurrent_per_pair must agree"
            )
    pair_cap = positive_integer(
        "max_concurrent_per_pair",
        explicit_pair_cap
        if explicit_pair_cap is not None
        else (legacy_pair_cap if legacy_pair_cap is not None else 3),
    )
    global_cap = positive_integer(
        "global_max_concurrent",
        config.get("global_max_concurrent", pair_cap * len(raw_pairs)),
    )
    return pair_cap, global_cap


class StrategyOwnershipRegistry:
    """One broker-local owner map with ledger enrichment."""

    def __init__(self, broker: VirtualBroker):
        self.__broker = broker
        self._owners: dict[str, object] = {}
        self._submission_stack: list[str] = []
        self._active_order_owner: dict[str, str] = {}
        self._active_trade_owner: dict[str, str] = {}
        self._order_owner_history: dict[str, str] = {}
        self._trade_owner_history: dict[str, str] = {}
        self._entry_caps: dict[str, tuple[int, int]] = {}
        self._rehydrate_from_ledger(broker)
        original_log = broker._log

        def owned_log(event: str, payload: dict[str, Any]) -> None:
            enriched = dict(payload)
            owner = self._owner_for_event(event, enriched)
            declared = enriched.get("strategy_owner_id")
            if declared is not None and declared != owner:
                raise StrategyOwnershipError(
                    "ledger owner does not match broker ownership"
                )
            if owner is not None:
                enriched["strategy_owner_id"] = owner
            original_log(event, enriched)
            self._commit_event(event, enriched, owner)

        broker._log = owned_log  # type: ignore[method-assign]
        broker._entry_admission = self._entry_admission_decision

    def _rehydrate_from_ledger(self, broker: VirtualBroker) -> None:
        """Restore durable owner history before wrapping a reopened broker."""

        try:
            raw_lines = broker.ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise StrategyOwnershipError(
                "broker ownership ledger is unreadable"
            ) from exc
        expected_previous = _ZERO_SHA256
        parsed_records: list[tuple[str, Mapping[str, Any]]] = []
        for line_number, raw_line in enumerate(raw_lines, start=1):
            try:
                record = _strict_json_loads(raw_line)
            except (json.JSONDecodeError, VirtualBrokerError) as exc:
                raise StrategyOwnershipError(
                    f"broker ownership ledger is invalid at line {line_number}"
                ) from exc
            if not isinstance(record, Mapping) or set(record) != {
                "ts_utc",
                "event",
                "payload",
                "prev_sha",
                "sha",
            }:
                raise StrategyOwnershipError(
                    f"broker ownership ledger is invalid at line {line_number}"
                )
            event = record.get("event")
            payload = record.get("payload")
            body = {key: value for key, value in record.items() if key != "sha"}
            if (
                not isinstance(event, str)
                or not isinstance(payload, Mapping)
                or record.get("prev_sha") != expected_previous
                or not isinstance(record.get("sha"), str)
                or record["sha"] != _canonical_sha256(body)
            ):
                raise StrategyOwnershipError(
                    f"broker ownership ledger is invalid at line {line_number}"
                )
            expected_previous = record["sha"]
            parsed_records.append((event, payload))
        if expected_previous != broker._prev_sha:
            raise StrategyOwnershipError(
                "broker ownership ledger terminal hash does not match broker state"
            )
        for event, payload in parsed_records:
            owner = self._identity(payload, "strategy_owner_id")
            if owner is not None:
                self._commit_event(event, payload, owner)

    def reload_durable_state(self) -> None:
        """Rebuild owner maps after a broker transaction rollback."""

        if self._submission_stack:
            raise StrategyOwnershipError(
                "cannot reload ownership during an active submission"
            )
        self._active_order_owner.clear()
        self._active_trade_owner.clear()
        self._order_owner_history.clear()
        self._trade_owner_history.clear()
        self._rehydrate_from_ledger(self.__broker)

    def register_owner(
        self,
        owner_id: str,
        token: object,
        *,
        max_concurrent_per_pair: int | None = None,
        global_max_concurrent: int | None = None,
    ) -> None:
        if not isinstance(owner_id, str) or not owner_id or len(owner_id) > 128:
            raise StrategyOwnershipError(
                "strategy owner id must contain 1..128 characters"
            )
        if any(ord(char) < 33 or ord(char) > 126 for char in owner_id):
            raise StrategyOwnershipError("strategy owner id must be visible ASCII")
        prior = self._owners.get(owner_id)
        if prior is not None and prior is not token:
            raise StrategyOwnershipError(f"duplicate strategy owner id: {owner_id}")
        if (max_concurrent_per_pair is None) != (global_max_concurrent is None):
            raise StrategyOwnershipError(
                "owner concurrency caps must be declared together"
            )
        if max_concurrent_per_pair is not None and global_max_concurrent is not None:
            if (
                isinstance(max_concurrent_per_pair, bool)
                or not isinstance(max_concurrent_per_pair, int)
                or max_concurrent_per_pair <= 0
                or isinstance(global_max_concurrent, bool)
                or not isinstance(global_max_concurrent, int)
                or global_max_concurrent <= 0
            ):
                raise StrategyOwnershipError(
                    "owner concurrency caps must be positive integers"
                )
            self._entry_caps[owner_id] = (
                max_concurrent_per_pair,
                global_max_concurrent,
            )
        self._owners[owner_id] = token

    @contextmanager
    def submission(self, owner_id: str) -> Iterator[None]:
        if owner_id not in self._owners:
            raise StrategyOwnershipError("unregistered strategy owner")
        self._submission_stack.append(owner_id)
        try:
            yield
        finally:
            popped = self._submission_stack.pop()
            if popped != owner_id:
                raise StrategyOwnershipError("strategy submission stack corrupted")

    def _current_owner(self) -> str | None:
        return self._submission_stack[-1] if self._submission_stack else None

    def _entry_admission_decision(
        self, pair: str, side: str, order_id: str | None
    ) -> dict[str, Any] | None:
        """Reject an owner's next fill once its current position cap is full."""

        del side  # The concurrency contract is direction-neutral.
        owner_id = (
            self._current_owner()
            if order_id is None
            else self._active_order_owner.get(order_id)
        )
        if owner_id is None:
            return None
        caps = self._entry_caps.get(owner_id)
        if caps is None:
            return None
        max_per_pair, global_max = caps
        active_trade_ids = self.active_trade_ids(owner_id)
        active_global = len(active_trade_ids)
        active_pair = sum(
            1
            for trade_id in active_trade_ids
            if self.__broker.positions[trade_id].pair == pair
        )
        if active_pair >= max_per_pair:
            return {
                "scope": "PAIR",
                "reason": "OWNER_PAIR_CONCURRENCY_CAP_REACHED",
                "active_pair_positions": active_pair,
                "max_concurrent_per_pair": max_per_pair,
                "active_global_positions": active_global,
                "global_max_concurrent": global_max,
            }
        if active_global >= global_max:
            return {
                "scope": "GLOBAL",
                "reason": "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED",
                "active_pair_positions": active_pair,
                "max_concurrent_per_pair": max_per_pair,
                "active_global_positions": active_global,
                "global_max_concurrent": global_max,
            }
        return None

    @staticmethod
    def _identity(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        return value if isinstance(value, str) and value else None

    def _owner_for_event(self, event: str, payload: Mapping[str, Any]) -> str | None:
        if event == "BOT_LOADED":
            declared = self._identity(payload, "strategy_owner_id")
            return declared if declared in self._owners else None
        if event == "PERIOD_END_SETTLEMENT":
            declared = self._identity(payload, "strategy_owner_id")
            return declared if declared in self._owners else None
        if event in {
            "ORDER_LIMIT",
            "ORDER_STOP",
            "FILL_MARKET",
            "ORDER_REJECTED_INSUFFICIENT_MARGIN",
            "ORDER_REJECTED_CONCURRENCY_CAP",
        }:
            return self._current_owner()
        if event in {
            "FILL_LIMIT",
            "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
            "ORDER_CANCEL",
            "ORDER_CANCEL_CONCURRENCY_CAP",
        }:
            order_id = self._identity(payload, "order_id")
            return self._active_order_owner.get(order_id or "")
        if event.startswith("EXIT") or event in {
            "CLOSE",
            "MARGIN_CLOSEOUT",
            "SET_EXIT",
        }:
            trade_id = self._identity(payload, "trade_id")
            return self._active_trade_owner.get(trade_id or "")
        return None

    def _bind_unique(
        self, mapping: MutableMapping[str, str], identity: str, owner: str
    ) -> None:
        prior = mapping.get(identity)
        if prior is not None and prior != owner:
            raise StrategyOwnershipError(
                f"identity {identity} already belongs to {prior}"
            )
        mapping[identity] = owner

    def _commit_event(
        self, event: str, payload: Mapping[str, Any], owner: str | None
    ) -> None:
        order_id = self._identity(payload, "order_id")
        trade_id = self._identity(payload, "trade_id")
        if event in {"ORDER_LIMIT", "ORDER_STOP"} and owner and order_id:
            self._bind_unique(self._active_order_owner, order_id, owner)
            self._bind_unique(self._order_owner_history, order_id, owner)
        elif event == "FILL_MARKET" and owner and trade_id:
            self._bind_unique(self._active_trade_owner, trade_id, owner)
            self._bind_unique(self._trade_owner_history, trade_id, owner)
        elif event == "FILL_LIMIT" and owner and order_id and trade_id:
            self._active_order_owner.pop(order_id, None)
            self._bind_unique(self._active_trade_owner, trade_id, owner)
            self._bind_unique(self._trade_owner_history, trade_id, owner)
        elif (
            event
            in {
                "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                "ORDER_CANCEL",
                "ORDER_CANCEL_CONCURRENCY_CAP",
            }
            and order_id
        ):
            self._active_order_owner.pop(order_id, None)
        elif (
            event.startswith("EXIT") or event in {"CLOSE", "MARGIN_CLOSEOUT"}
        ) and trade_id:
            if trade_id not in self.__broker.positions:
                self._active_trade_owner.pop(trade_id, None)

    def assert_order_owner(self, order_id: str, owner_id: str) -> None:
        if self._active_order_owner.get(order_id) != owner_id:
            raise StrategyOwnershipError("strategy cannot mutate an unowned order")

    def assert_trade_owner(self, trade_id: str, owner_id: str) -> None:
        if self._active_trade_owner.get(trade_id) != owner_id:
            raise StrategyOwnershipError("strategy cannot mutate an unowned trade")

    def active_trade_ids(
        self, owner_id: str, *, pair: str | None = None
    ) -> tuple[str, ...]:
        return tuple(
            trade_id
            for trade_id, owner in self._active_trade_owner.items()
            if owner == owner_id
            and trade_id in self.__broker.positions
            and (pair is None or self.__broker.positions[trade_id].pair == pair)
        )

    def active_order_ids(
        self, owner_id: str, *, pair: str | None = None
    ) -> tuple[str, ...]:
        return tuple(
            order_id
            for order_id, owner in self._active_order_owner.items()
            if owner == owner_id
            and order_id in self.__broker.orders
            and (pair is None or self.__broker.orders[order_id].pair == pair)
        )

    def historical_trade_owner(self, trade_id: str) -> str | None:
        return self._trade_owner_history.get(trade_id)

    def historical_order_owner(self, order_id: str) -> str | None:
        return self._order_owner_history.get(order_id)


def strategy_ownership_registry(broker: VirtualBroker) -> StrategyOwnershipRegistry:
    existing = getattr(broker, _REGISTRY_ATTR, None)
    if existing is not None:
        if not isinstance(existing, StrategyOwnershipRegistry):
            raise StrategyOwnershipError("broker ownership registry type mismatch")
        return existing
    registry = StrategyOwnershipRegistry(broker)
    setattr(broker, _REGISTRY_ATTR, registry)
    return registry


@dataclass(frozen=True)
class OwnedPositionView:
    """Immutable, owner-filtered position data exposed to a strategy."""

    trade_id: str
    pair: str
    side: str
    units: float
    entry_price: float
    opened_ts: str
    tp_price: float | None
    sl_price: float | None
    current_price: float | None


@dataclass(frozen=True)
class OwnedQuoteView:
    """Immutable executable bid/ask staged for one owner-scoped decision."""

    pair: str
    bid: float
    ask: float
    ts: str


def _owned_components(
    view: "OwnedBrokerView",
) -> tuple[VirtualBroker, StrategyOwnershipRegistry, str]:
    """Internal access that is intentionally absent from the facade API."""

    return (
        object.__getattribute__(view, "_OwnedBrokerView__broker"),
        object.__getattribute__(view, "_OwnedBrokerView__ownership"),
        object.__getattribute__(view, "_OwnedBrokerView__owner_id"),
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


class OwnedBrokerView:
    """Opaque owner-scoped broker facade with immutable query results."""

    __slots__ = ("__broker", "__ownership", "__owner_id", "__registration_token")
    _HIDDEN_STORAGE = frozenset(
        {
            "_OwnedBrokerView__broker",
            "_OwnedBrokerView__ownership",
            "_OwnedBrokerView__registration_token",
        }
    )

    def __init__(
        self,
        broker: VirtualBroker,
        owner_id: str,
        *,
        max_concurrent_per_pair: int | None = None,
        global_max_concurrent: int | None = None,
    ):
        ownership = strategy_ownership_registry(broker)
        token = object()
        ownership.register_owner(
            owner_id,
            token,
            max_concurrent_per_pair=max_concurrent_per_pair,
            global_max_concurrent=global_max_concurrent,
        )
        object.__setattr__(self, "_OwnedBrokerView__broker", broker)
        object.__setattr__(self, "_OwnedBrokerView__ownership", ownership)
        object.__setattr__(self, "_OwnedBrokerView__owner_id", owner_id)
        object.__setattr__(self, "_OwnedBrokerView__registration_token", token)

    def __getattribute__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "_HIDDEN_STORAGE"):
            raise AttributeError("raw broker storage is not exposed")
        return object.__getattribute__(self, name)

    @property
    def owner_id(self) -> str:
        return object.__getattribute__(self, "_OwnedBrokerView__owner_id")

    def account(self) -> Mapping[str, Any]:
        broker, _, _ = _owned_components(self)
        return _deep_freeze(broker.account())

    def jpy_per_quote_unit(self, pair: str) -> float:
        broker, _, _ = _owned_components(self)
        return broker._jpy_per_quote_unit(pair)

    def executable_quote(self, pair: str) -> OwnedQuoteView:
        broker, _, _ = _owned_components(self)
        quote = broker.last_quotes.get(pair)
        if quote is None:
            raise VirtualBrokerError(f"no live quote for {pair}; cannot execute")
        bid, ask, ts = quote
        return OwnedQuoteView(pair=pair, bid=bid, ask=ask, ts=ts)

    def executable_market_entry_price(self, pair: str, side: str) -> float:
        """Return the exact price a market order would receive at this quote.

        Closed-bar callbacks run only after the next executable open has been
        staged.  Exposing this immutable scalar lets a strategy size against
        that current quote (including configured adverse slippage) without
        exposing mutable broker quote storage.
        """

        broker, _, _ = _owned_components(self)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        quote = broker.last_quotes.get(pair)
        if quote is None:
            raise VirtualBrokerError(f"no live quote for {pair}; cannot size")
        bid, ask, _ = quote
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        slip = broker.slippage_pips * pip
        price = ask + slip if side == "LONG" else bid - slip
        price = round(price, 3 if pair.endswith("JPY") else 5)
        if price <= 0:
            raise VirtualBrokerError("slippage produced a non-positive entry price")
        return price

    def position(self, trade_id: str) -> OwnedPositionView | None:
        broker, ownership, owner_id = _owned_components(self)
        if trade_id not in ownership.active_trade_ids(owner_id):
            return None
        position = broker.positions.get(trade_id)
        if position is None:
            return None
        quote = broker.last_quotes.get(position.pair)
        current_price = None
        if quote is not None:
            current_price = quote[0] if position.side == "LONG" else quote[1]
        return OwnedPositionView(
            trade_id=position.trade_id,
            pair=position.pair,
            side=position.side,
            units=position.units,
            entry_price=position.entry_price,
            opened_ts=position.opened_ts,
            tp_price=position.tp_price,
            sl_price=position.sl_price,
            current_price=current_price,
        )

    def market_order(self, *args: Any, **kwargs: Any) -> str:
        broker, ownership, owner_id = _owned_components(self)
        with ownership.submission(owner_id):
            return broker.market_order(*args, **kwargs)

    def limit_order(self, *args: Any, **kwargs: Any) -> str:
        broker, ownership, owner_id = _owned_components(self)
        with ownership.submission(owner_id):
            return broker.limit_order(*args, **kwargs)

    def stop_order(self, *args: Any, **kwargs: Any) -> str:
        broker, ownership, owner_id = _owned_components(self)
        with ownership.submission(owner_id):
            return broker.stop_order(*args, **kwargs)

    def cancel_order(self, order_id: str) -> None:
        broker, ownership, owner_id = _owned_components(self)
        ownership.assert_order_owner(order_id, owner_id)
        broker.cancel_order(order_id)

    def close_trade(self, trade_id: str, units: float | None = None) -> float:
        broker, ownership, owner_id = _owned_components(self)
        ownership.assert_trade_owner(trade_id, owner_id)
        return broker.close_trade(trade_id, units=units)

    def set_exit(
        self,
        trade_id: str,
        tp_price: float | None = None,
        sl_price: float | None = None,
    ) -> None:
        broker, ownership, owner_id = _owned_components(self)
        ownership.assert_trade_owner(trade_id, owner_id)
        broker.set_exit(trade_id, tp_price=tp_price, sl_price=sl_price)

    def active_trade_ids(self, *, pair: str | None = None) -> tuple[str, ...]:
        _, ownership, owner_id = _owned_components(self)
        return ownership.active_trade_ids(owner_id, pair=pair)

    def active_order_ids(self, *, pair: str | None = None) -> tuple[str, ...]:
        _, ownership, owner_id = _owned_components(self)
        return ownership.active_order_ids(owner_id, pair=pair)
