"""Capability-closed room-meta-01 adapter for the economic runner.

The runtime wraps one already sealed declarative strategy generation.  It
never creates a direction or geometry: risk-reducing intents pass through and
new-risk intents retain their upstream order, pair, side, vehicle, TP, SL and
time bounds.  Only admission and units may change according to one sealed
anomaly arm.

The adapter constructs completed H1 bid/ask candles from the runner's exact-28
synchronized quote stream.  Gaps reset the H1 history; no candle is filled or
carried forward.  A compact hash-chain summary retains every upstream
candidate, HOLD, resize and selected replacement.  Economic P/L is computed by
the ordinary portfolio reducer.  The summary is diagnostic until the separate
transcript auditor independently reconstructs every held counterfactual.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_anomaly_admission_controller import (
    EXPERIMENT_ARMS,
    FORMAL_G8_CURRENCIES,
    FORMAL_G8_PAIRS,
    H1_SECONDS,
    allocate_economic_candidates,
    canonical_sha256,
    validate_policy,
)
from quant_rabbit.dojo_tuned_strategy_runtime import (
    SealedTunedStrategyRuntimeFactory,
    build_tuned_strategy_runtime_factory,
    verify_tuned_strategy_runtime_seal,
)


RUNTIME_SEAL_CONTRACT: Final = "QR_DOJO_ANOMALY_ADMISSION_RUNTIME_SEAL_V1"
RUNTIME_STATE_CONTRACT: Final = "QR_DOJO_ANOMALY_ADMISSION_RUNTIME_STATE_V1"
EVIDENCE_SUMMARY_CONTRACT: Final = (
    "QR_DOJO_ANOMALY_ADMISSION_RUNTIME_EVIDENCE_SUMMARY_V1"
)
SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64
MAX_DEPENDENCY_BYTES: Final = 16 * 1024 * 1024
MAX_COUNTERFACTUAL_TAIL: Final = 64
MAX_CAPACITY_SLOTS: Final = 32

_PHASES: Final = {
    "OHLC": ("O", "H", "L", "C"),
    "OLHC": ("O", "L", "H", "C"),
}
_DEPENDENCY_PATHS: Final = (
    "src/quant_rabbit/dojo_anomaly_admission_controller.py",
    "src/quant_rabbit/dojo_anomaly_admission_runtime.py",
    "src/quant_rabbit/dojo_shared_worker_protocol.py",
    "src/quant_rabbit/dojo_tuned_strategy_runtime.py",
)
_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
    "direction_prediction_allowed": False,
    "direction_change_allowed": False,
}
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_COUNT_KEYS: Final = frozenset(
    {"decisions", "upstream_candidates", "selected", "held", "reduced", "warmup_held"}
)
_STATE_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "runtime_binding_sha256",
        "upstream_state",
        "h1_state",
        "counts",
        "evidence_chain_sha256",
        "counterfactual_tail",
    }
)


class DojoAnomalyAdmissionRuntimeError(ValueError):
    """The runtime seal, causal H1 state, or upstream mapping is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoAnomalyAdmissionRuntimeError(
            "value is not strict canonical JSON"
        ) from exc


def _copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _file_sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_DEPENDENCY_BYTES
    ):
        raise DojoAnomalyAdmissionRuntimeError(
            f"runtime dependency is not a bounded single-link file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        while block := os.read(descriptor, 1024 * 1024):
            size += len(block)
            digest.update(block)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identities = {
        (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
        for state in (before, opened, current)
    }
    if len(identities) != 1 or size != before.st_size:
        raise DojoAnomalyAdmissionRuntimeError(
            f"runtime dependency changed while hashing: {path}"
        )
    return size, digest.hexdigest()


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoAnomalyAdmissionRuntimeError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoAnomalyAdmissionRuntimeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DojoAnomalyAdmissionRuntimeError(f"{field} must be positive")
    return result


def build_anomaly_admission_runtime_seal(
    repo_root: Path,
    *,
    upstream_runtime_seal: Mapping[str, Any],
    policy: Mapping[str, Any],
    arm: str,
    capacity_slots: int,
) -> dict[str, Any]:
    """Seal the wrapper, upstream generation, H1 policy, and source bytes."""

    root = Path(repo_root).resolve(strict=True)
    actual_root = Path(__file__).resolve().parents[2]
    if root != actual_root or not root.is_dir():
        raise DojoAnomalyAdmissionRuntimeError(
            "repo_root must be the source tree that loaded this runtime"
        )
    upstream = verify_tuned_strategy_runtime_seal(
        upstream_runtime_seal, repo_root=root
    )
    normalized_policy = validate_policy(policy)
    if arm not in EXPERIMENT_ARMS or arm == "AI_EXIT_CAPITAL_RELEASE":
        raise DojoAnomalyAdmissionRuntimeError(
            "runtime arm must be a direction-neutral room-meta-01 arm"
        )
    slots = _integer(capacity_slots, field="capacity_slots", minimum=1)
    if slots > MAX_CAPACITY_SLOTS:
        raise DojoAnomalyAdmissionRuntimeError("capacity_slots exceeds runtime bound")
    dependencies = []
    for relative_path in _DEPENDENCY_PATHS:
        size, digest = _file_sha256(root / relative_path)
        dependencies.append(
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    body = {
        "contract": RUNTIME_SEAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "runtime_mode": "SEALED_ANOMALY_ADMISSION_OVER_TUNED_STRATEGY",
        "room_id": "room-meta-01",
        "arm": arm,
        "capacity_slots": slots,
        "upstream_ranking_policy": "SEALED_BINDING_ORDER_THEN_INTENT_ORDER",
        "upstream_runtime_binding_sha256": upstream["runtime_binding_sha256"],
        "upstream_runtime_seal": upstream,
        "worker_catalog": _copy(upstream["worker_catalog"]),
        "policy": normalized_policy,
        "policy_sha256": normalized_policy["policy_sha256"],
        "required_completed_h1_bars": max(normalized_policy["lookbacks"].values())
        + 1,
        "formal_pair_universe": list(FORMAL_G8_PAIRS),
        "dependencies": dependencies,
        "dependencies_sha256": canonical_sha256(dependencies),
        "capabilities": {
            "arbitrary_import_allowed": False,
            "broker_handle_available": False,
            "filesystem_available_to_worker": False,
            "network_available_to_worker": False,
            "upstream_direction_or_geometry_change_allowed": False,
            "risk_reducing_intents_pass_through": True,
            "new_risk_admission_and_units_only": True,
        },
        "evidence": {
            "economic_runner_integration_available": True,
            "held_counterfactual_hash_chain_available": True,
            "independent_counterfactual_reexecution_available": False,
            "official_evidence_eligible": False,
            "diagnostic_only": True,
        },
        "authority": _copy(_AUTHORITY),
    }
    return {**body, "runtime_binding_sha256": canonical_sha256(body)}


def verify_anomaly_admission_runtime_seal(
    seal: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    if not isinstance(seal, Mapping):
        raise DojoAnomalyAdmissionRuntimeError("runtime seal must be an object")
    rebuilt = build_anomaly_admission_runtime_seal(
        repo_root,
        upstream_runtime_seal=seal.get("upstream_runtime_seal"),
        policy=seal.get("policy"),
        arm=seal.get("arm"),
        capacity_slots=seal.get("capacity_slots"),
    )
    if _copy(seal) != rebuilt:
        raise DojoAnomalyAdmissionRuntimeError(
            "runtime seal differs from its closed dependency denominator"
        )
    return rebuilt


class _CompletedH1Tracker:
    """Exact-28 H1 builder with discontinuity reset and no interpolation."""

    def __init__(self, *, cadence_seconds: int, required_bars: int, state: Any) -> None:
        cadence = _integer(cadence_seconds, field="cadence_seconds", minimum=1)
        if H1_SECONDS % cadence != 0:
            raise DojoAnomalyAdmissionRuntimeError(
                "runner cadence must divide the H1 boundary"
            )
        self.cadence = cadence
        self.required_bars = _integer(
            required_bars, field="required_bars", minimum=3
        )
        self.expected_subbars = H1_SECONDS // cadence
        self.completed = {pair: [] for pair in FORMAL_G8_PAIRS}
        self.forming: dict[str, Any] | None = None
        self.last_epoch: int | None = None
        self.last_phase: str | None = None
        self.last_intrabar: str | None = None
        if state is not None:
            self._restore(state)

    def _restore(self, value: Any) -> None:
        if not isinstance(value, Mapping) or set(value) != {
            "cadence_seconds",
            "required_bars",
            "completed",
            "forming",
            "last_epoch",
            "last_phase",
            "last_intrabar",
        }:
            raise DojoAnomalyAdmissionRuntimeError("H1 carry schema mismatch")
        if (
            value["cadence_seconds"] != self.cadence
            or value["required_bars"] != self.required_bars
            or set(value["completed"]) != set(FORMAL_G8_PAIRS)
        ):
            raise DojoAnomalyAdmissionRuntimeError("H1 carry binding drifted")
        completed = _copy(value["completed"])
        epochs: list[int] | None = None
        for pair in FORMAL_G8_PAIRS:
            rows = completed[pair]
            if not isinstance(rows, list) or len(rows) > self.required_bars:
                raise DojoAnomalyAdmissionRuntimeError("H1 carry history is invalid")
            pair_epochs = [row.get("close_epoch") for row in rows]
            if epochs is None:
                epochs = pair_epochs
            elif pair_epochs != epochs:
                raise DojoAnomalyAdmissionRuntimeError(
                    "H1 carry histories are unsynchronized"
                )
        self.completed = completed
        self.forming = _copy(value["forming"])
        self.last_epoch = value["last_epoch"]
        self.last_phase = value["last_phase"]
        self.last_intrabar = value["last_intrabar"]

    def _reset_market_history(self) -> None:
        self.completed = {pair: [] for pair in FORMAL_G8_PAIRS}
        self.forming = None

    def consume(self, snapshot: Mapping[str, Any]) -> None:
        epoch = _integer(snapshot.get("epoch"), field="snapshot.epoch", minimum=1)
        phase = snapshot.get("phase")
        intrabar = snapshot.get("intrabar")
        if intrabar not in _PHASES or phase not in _PHASES[intrabar]:
            raise DojoAnomalyAdmissionRuntimeError("snapshot phase/path is invalid")
        quote_rows = snapshot.get("quotes")
        if not isinstance(quote_rows, Sequence) or isinstance(quote_rows, (str, bytes)):
            raise DojoAnomalyAdmissionRuntimeError("snapshot quotes must be an array")
        quotes = {row.get("pair"): row for row in quote_rows if isinstance(row, Mapping)}
        if set(quotes) != set(FORMAL_G8_PAIRS) or len(quote_rows) != len(quotes):
            raise DojoAnomalyAdmissionRuntimeError(
                "anomaly runtime requires the exact-28 quote batch"
            )
        phase_order = _PHASES[intrabar]
        if self.last_intrabar is not None and self.last_intrabar != intrabar:
            raise DojoAnomalyAdmissionRuntimeError("intrabar path drifted")
        if self.last_epoch is None:
            if phase != "O":
                raise DojoAnomalyAdmissionRuntimeError("H1 stream must start at O")
        elif epoch == self.last_epoch:
            if (
                self.last_phase not in phase_order
                or phase_order.index(phase) != phase_order.index(self.last_phase) + 1
            ):
                raise DojoAnomalyAdmissionRuntimeError("phase sequence is non-causal")
        elif epoch > self.last_epoch:
            if self.last_phase != "C" or phase != "O":
                raise DojoAnomalyAdmissionRuntimeError(
                    "new quote epoch did not follow a completed candle"
                )
            if epoch - self.last_epoch != self.cadence:
                self._reset_market_history()
        else:
            raise DojoAnomalyAdmissionRuntimeError("quote epoch moved backwards")

        bucket_start = epoch - (epoch % H1_SECONDS)
        if phase == "O":
            if self.forming is None or self.forming["bucket_start"] != bucket_start:
                if self.forming is not None:
                    self._reset_market_history()
                self.forming = {
                    "bucket_start": bucket_start,
                    "first_epoch": epoch,
                    "subbar_count": 0,
                    "pairs": {},
                }
            for pair in FORMAL_G8_PAIRS:
                bid = _positive(quotes[pair].get("bid"), field=f"{pair}.bid")
                ask = _positive(quotes[pair].get("ask"), field=f"{pair}.ask")
                if ask < bid:
                    raise DojoAnomalyAdmissionRuntimeError("ask crossed below bid")
                current = self.forming["pairs"].get(pair)
                if current is None:
                    self.forming["pairs"][pair] = {
                        "bid_high": bid,
                        "bid_low": bid,
                        "bid_close": bid,
                        "ask_high": ask,
                        "ask_low": ask,
                        "ask_close": ask,
                    }
        if self.forming is None or self.forming["bucket_start"] != bucket_start:
            raise DojoAnomalyAdmissionRuntimeError("H1 forming bucket is missing")
        for pair in FORMAL_G8_PAIRS:
            bid = _positive(quotes[pair].get("bid"), field=f"{pair}.bid")
            ask = _positive(quotes[pair].get("ask"), field=f"{pair}.ask")
            row = self.forming["pairs"][pair]
            row["bid_high"] = max(row["bid_high"], bid)
            row["bid_low"] = min(row["bid_low"], bid)
            row["bid_close"] = bid
            row["ask_high"] = max(row["ask_high"], ask)
            row["ask_low"] = min(row["ask_low"], ask)
            row["ask_close"] = ask
        if phase == "C":
            self.forming["subbar_count"] += 1
            close_epoch = epoch + self.cadence
            if close_epoch == bucket_start + H1_SECONDS:
                complete = (
                    self.forming["first_epoch"] == bucket_start
                    and self.forming["subbar_count"] == self.expected_subbars
                )
                if not complete:
                    self._reset_market_history()
                else:
                    for pair in FORMAL_G8_PAIRS:
                        candle = {
                            "close_epoch": close_epoch,
                            "complete": True,
                            **self.forming["pairs"][pair],
                        }
                        self.completed[pair].append(candle)
                        self.completed[pair] = self.completed[pair][
                            -self.required_bars :
                        ]
                    self.forming = None
        self.last_epoch = epoch
        self.last_phase = str(phase)
        self.last_intrabar = str(intrabar)

    def ready_panel(self) -> tuple[dict[str, Any], int] | None:
        if any(len(rows) < self.required_bars for rows in self.completed.values()):
            return None
        panel = {
            pair: _copy(rows[-self.required_bars :])
            for pair, rows in self.completed.items()
        }
        epochs = {rows[-1]["close_epoch"] for rows in panel.values()}
        if len(epochs) != 1:
            raise DojoAnomalyAdmissionRuntimeError("H1 panel tip is unsynchronized")
        return panel, int(next(iter(epochs)))

    def export_state(self) -> dict[str, Any]:
        return {
            "cadence_seconds": self.cadence,
            "required_bars": self.required_bars,
            "completed": _copy(self.completed),
            "forming": _copy(self.forming),
            "last_epoch": self.last_epoch,
            "last_phase": self.last_phase,
            "last_intrabar": self.last_intrabar,
        }


def _quote_index(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["pair"]: row for row in snapshot["quotes"]}


def _currency_to_jpy(currency: str, quotes: Mapping[str, Mapping[str, Any]]) -> float:
    if currency == "JPY":
        return 1.0
    direct = quotes.get(f"{currency}_JPY")
    inverse = quotes.get(f"JPY_{currency}")
    if direct is not None:
        return (_positive(direct["bid"], field="bid") + _positive(direct["ask"], field="ask")) / 2
    if inverse is not None:
        mid = (_positive(inverse["bid"], field="bid") + _positive(inverse["ask"], field="ask")) / 2
        return 1.0 / mid
    raise DojoAnomalyAdmissionRuntimeError(
        f"exact-28 batch lacks a JPY conversion for {currency}"
    )


def _gross_exposure(
    snapshot: Mapping[str, Any], quotes: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    equity = _positive(snapshot["account"]["equity_jpy"], field="account.equity_jpy")
    result = {currency: 0.0 for currency in FORMAL_G8_CURRENCIES}
    for collection_name in ("positions", "pending_orders"):
        rows = snapshot[collection_name]
        for row in rows:
            base, quote = row["pair"].split("_")
            notional = _positive(row["units"], field=f"{collection_name}.units") * _currency_to_jpy(base, quotes)
            fraction = notional / equity
            result[base] += fraction
            result[quote] += fraction
    return result


def _candidate_rows(
    proposals: Sequence[Mapping[str, Any]],
    *,
    snapshot: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    equity = _positive(snapshot["account"]["equity_jpy"], field="account.equity_jpy")
    candidates: list[dict[str, Any]] = []
    locations: dict[str, tuple[int, int]] = {}
    rank = 0
    for proposal_index, proposal in enumerate(proposals):
        for intent_index, intent in enumerate(proposal["new_risk_intents"]):
            rank += 1
            params = intent["parameters"]
            units = _positive(params["units"], field="intent.units")
            pair = params["pair"]
            candidate_id = f"{proposal['worker_id']}:{intent['intent_id']}"
            if candidate_id in locations:
                raise DojoAnomalyAdmissionRuntimeError(
                    "upstream candidate identity collided"
                )
            base = pair.split("_")[0]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "priority_rank": rank,
                    "strategy_family": proposal["family_id"],
                    "pair": pair,
                    "side": params["side"],
                    "full_size_units": units,
                    "currency_exposure_increment_at_full_size": units
                    * _currency_to_jpy(base, quotes)
                    / equity,
                }
            )
            locations[candidate_id] = (proposal_index, intent_index)
    return candidates, locations


def _filter_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    locations: Mapping[str, tuple[int, int]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = _copy(proposals)
    selected_by_location: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in decisions:
        if row["status"] == "SELECTED":
            selected_by_location[locations[row["candidate_id"]]] = row
    for proposal_index, proposal in enumerate(result):
        filtered = []
        for intent_index, intent in enumerate(proposal["new_risk_intents"]):
            decision = selected_by_location.get((proposal_index, intent_index))
            if decision is None:
                continue
            intent["parameters"]["units"] = decision["allocated_units"]
            filtered.append(intent)
        proposal["new_risk_intents"] = filtered
    return result


class _AnomalyAdmissionRuntime:
    def __init__(
        self,
        *,
        seal: Mapping[str, Any],
        upstream_factory: SealedTunedStrategyRuntimeFactory,
        coordinate: Mapping[str, Any],
        bindings: Sequence[Mapping[str, str]],
        prior_state: Any | None,
    ) -> None:
        cadence = _integer(coordinate.get("bar_seconds"), field="bar_seconds", minimum=1)
        state = None if prior_state is None else _copy(prior_state)
        if state is not None:
            if (
                not isinstance(state, Mapping)
                or set(state) != set(_STATE_KEYS)
                or state.get("contract") != RUNTIME_STATE_CONTRACT
                or state.get("schema_version") != SCHEMA_VERSION
                or state.get("runtime_binding_sha256")
                != seal["runtime_binding_sha256"]
            ):
                raise DojoAnomalyAdmissionRuntimeError("runtime carry binding drifted")
            counts = state.get("counts")
            tail = state.get("counterfactual_tail")
            chain = state.get("evidence_chain_sha256")
            if (
                not isinstance(counts, Mapping)
                or set(counts) != set(_COUNT_KEYS)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in counts.values()
                )
                or counts["selected"] + counts["held"]
                != counts["upstream_candidates"]
                or counts["reduced"] > counts["selected"]
                or counts["warmup_held"] > counts["held"]
                or not isinstance(chain, str)
                or _SHA256_RE.fullmatch(chain) is None
                or not isinstance(tail, list)
                or len(tail) > MAX_COUNTERFACTUAL_TAIL
                or len(tail) > counts["decisions"]
            ):
                raise DojoAnomalyAdmissionRuntimeError(
                    "runtime carry evidence counters are invalid"
                )
            previous: str | None = None
            for index, row in enumerate(tail):
                if (
                    not isinstance(row, Mapping)
                    or not isinstance(row.get("previous_evidence_sha256"), str)
                    or _SHA256_RE.fullmatch(row["previous_evidence_sha256"]) is None
                    or not isinstance(row.get("event_sha256"), str)
                    or _SHA256_RE.fullmatch(row["event_sha256"]) is None
                ):
                    raise DojoAnomalyAdmissionRuntimeError(
                        "runtime carry counterfactual tail is invalid"
                    )
                event = {
                    key: value
                    for key, value in row.items()
                    if key not in {"previous_evidence_sha256", "event_sha256"}
                }
                if row["event_sha256"] != canonical_sha256(
                    {
                        "previous_evidence_sha256": row[
                            "previous_evidence_sha256"
                        ],
                        "event": event,
                    }
                ):
                    raise DojoAnomalyAdmissionRuntimeError(
                        "runtime carry counterfactual hash is invalid"
                    )
                if previous is not None and row["previous_evidence_sha256"] != previous:
                    raise DojoAnomalyAdmissionRuntimeError(
                        "runtime carry counterfactual chain forked"
                    )
                previous = row["event_sha256"]
            if tail:
                if previous != chain:
                    raise DojoAnomalyAdmissionRuntimeError(
                        "runtime carry evidence tip differs from its tail"
                    )
                if counts["decisions"] == len(tail) and tail[0][
                    "previous_evidence_sha256"
                ] != GENESIS_SHA256:
                    raise DojoAnomalyAdmissionRuntimeError(
                        "untruncated runtime evidence does not start at genesis"
                    )
            elif counts["decisions"] != 0 or chain != GENESIS_SHA256:
                raise DojoAnomalyAdmissionRuntimeError(
                    "empty runtime evidence must retain the genesis tip"
                )
        self._seal = _copy(seal)
        self._upstream = upstream_factory(
            coordinate,
            bindings,
            None if state is None else state["upstream_state"],
        )
        self._tracker = _CompletedH1Tracker(
            cadence_seconds=cadence,
            required_bars=seal["required_completed_h1_bars"],
            state=None if state is None else state["h1_state"],
        )
        self._evidence_chain = (
            GENESIS_SHA256 if state is None else state["evidence_chain_sha256"]
        )
        self._counts = (
            {"decisions": 0, "upstream_candidates": 0, "selected": 0, "held": 0, "reduced": 0, "warmup_held": 0}
            if state is None
            else _copy(state["counts"])
        )
        self._tail = [] if state is None else _copy(state["counterfactual_tail"])

    def propose(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        upstream = self._upstream.propose(snapshot)
        self._tracker.consume(snapshot)
        if not any(row["new_risk_intents"] for row in upstream):
            return upstream
        quotes = _quote_index(snapshot)
        candidates, locations = _candidate_rows(
            upstream, snapshot=snapshot, quotes=quotes
        )
        exposure = _gross_exposure(snapshot, quotes)
        phase = snapshot["phase"]
        decision_epoch = (
            int(snapshot["epoch"])
            if phase == "O"
            else int(snapshot["epoch"]) + self._tracker.cadence
            if phase == "C"
            else None
        )
        ready = self._tracker.ready_panel()
        if ready is None or decision_epoch is None:
            decisions = [
                {
                    **candidate,
                    "status": "NOT_SELECTED",
                    "admission_decision": "HOLD",
                    "size_multiplier": 0.0,
                    "allocated_units": 0.0,
                    "selected_slot": None,
                    "reason_codes": [
                        "H1_WARMUP_OR_DISCONTINUITY_HOLD"
                        if ready is None
                        else "UNSUPPORTED_INTRABAR_DECISION_PHASE_HOLD"
                    ],
                }
                for candidate in candidates
            ]
            allocation_sha = None
            latest_h1 = None if ready is None else ready[1]
            self._counts["warmup_held"] += len(candidates)
        else:
            panel, latest_h1 = ready
            allocation = allocate_economic_candidates(
                completed_h1_panel=panel,
                decision_epoch=decision_epoch,
                latest_completed_h1_close_epoch=latest_h1,
                policy=self._seal["policy"],
                arm=self._seal["arm"],
                candidates=candidates,
                currency_gross_exposure_fractions=exposure,
                capacity_slots=self._seal["capacity_slots"],
            )
            decisions = allocation["candidate_decisions"]
            allocation_sha = allocation["allocation_sha256"]
        filtered = _filter_proposals(
            upstream, locations=locations, decisions=decisions
        )
        summary_rows = [
            {
                "candidate_id": row["candidate_id"],
                "priority_rank": row["priority_rank"],
                "pair": row["pair"],
                "side": row["side"],
                "full_size_units": row["full_size_units"],
                "admission_decision": row["admission_decision"],
                "size_multiplier": row["size_multiplier"],
                "allocated_units": row["allocated_units"],
                "reason_codes": row["reason_codes"],
            }
            for row in decisions
        ]
        event = {
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "snapshot_epoch": snapshot["epoch"],
            "snapshot_phase": phase,
            "decision_epoch": decision_epoch,
            "latest_completed_h1_close_epoch": latest_h1,
            "policy_sha256": self._seal["policy_sha256"],
            "arm": self._seal["arm"],
            "allocation_sha256": allocation_sha,
            "currency_gross_exposure_fractions": exposure,
            "candidate_decisions": summary_rows,
            "upstream_proposals_sha256": canonical_sha256(upstream),
            "filtered_proposals_sha256": canonical_sha256(filtered),
        }
        previous_evidence_sha = self._evidence_chain
        event_sha = canonical_sha256(
            {"previous_evidence_sha256": previous_evidence_sha, "event": event}
        )
        self._evidence_chain = event_sha
        self._counts["decisions"] += 1
        self._counts["upstream_candidates"] += len(decisions)
        self._counts["selected"] += sum(row["status"] == "SELECTED" for row in decisions)
        self._counts["held"] += sum(row["status"] != "SELECTED" for row in decisions)
        self._counts["reduced"] += sum(
            row["admission_decision"] == "REDUCE_SIZE" for row in decisions
        )
        self._tail.append(
            {
                **event,
                "previous_evidence_sha256": previous_evidence_sha,
                "event_sha256": event_sha,
            }
        )
        self._tail = self._tail[-MAX_COUNTERFACTUAL_TAIL:]
        return filtered

    def export_admission_evidence(self) -> dict[str, Any]:
        body = {
            "contract": EVIDENCE_SUMMARY_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "runtime_binding_sha256": self._seal["runtime_binding_sha256"],
            "policy_sha256": self._seal["policy_sha256"],
            "arm": self._seal["arm"],
            "counts": _copy(self._counts),
            "evidence_chain_sha256": self._evidence_chain,
            "counterfactual_tail": _copy(self._tail),
            "counterfactual_tail_truncated": self._counts["decisions"] > len(self._tail),
            "runner_integration_complete": True,
            "independent_counterfactual_reexecution_complete": False,
            "official_evidence_eligible": False,
            "authority": _copy(_AUTHORITY),
        }
        return {**body, "evidence_summary_sha256": canonical_sha256(body)}

    def export_state(self) -> dict[str, Any]:
        return {
            "contract": RUNTIME_STATE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "runtime_binding_sha256": self._seal["runtime_binding_sha256"],
            "upstream_state": self._upstream.export_state(),
            "h1_state": self._tracker.export_state(),
            "counts": _copy(self._counts),
            "evidence_chain_sha256": self._evidence_chain,
            "counterfactual_tail": _copy(self._tail),
        }


class SealedAnomalyAdmissionRuntimeFactory:
    """Immutable callable holding one verified wrapper and upstream seal."""

    __slots__ = ("__seal", "__upstream_factory", "__locked")

    def __init__(self, seal: Mapping[str, Any], *, repo_root: Path) -> None:
        verified = verify_anomaly_admission_runtime_seal(seal, repo_root=repo_root)
        object.__setattr__(self, "_SealedAnomalyAdmissionRuntimeFactory__seal", _copy(verified))
        object.__setattr__(
            self,
            "_SealedAnomalyAdmissionRuntimeFactory__upstream_factory",
            build_tuned_strategy_runtime_factory(
                verified["upstream_runtime_seal"], repo_root=repo_root
            ),
        )
        object.__setattr__(self, "_SealedAnomalyAdmissionRuntimeFactory__locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_SealedAnomalyAdmissionRuntimeFactory__locked", False):
            raise AttributeError("an anomaly admission runtime factory is immutable")
        object.__setattr__(self, name, value)

    @property
    def runtime_binding_sha256(self) -> str:
        return str(self.__seal["runtime_binding_sha256"])

    def matches_verified_seal(self, seal: Mapping[str, Any]) -> bool:
        return self.__seal == _copy(seal)

    def __call__(
        self,
        coordinate: Mapping[str, Any],
        bindings: Sequence[Mapping[str, str]],
        prior_state: Any | None,
    ) -> _AnomalyAdmissionRuntime:
        return _AnomalyAdmissionRuntime(
            seal=self.__seal,
            upstream_factory=self.__upstream_factory,
            coordinate=coordinate,
            bindings=bindings,
            prior_state=prior_state,
        )


def build_anomaly_admission_runtime_factory(
    seal: Mapping[str, Any], *, repo_root: Path
) -> SealedAnomalyAdmissionRuntimeFactory:
    return SealedAnomalyAdmissionRuntimeFactory(seal, repo_root=repo_root)


__all__ = [
    "DojoAnomalyAdmissionRuntimeError",
    "EVIDENCE_SUMMARY_CONTRACT",
    "RUNTIME_SEAL_CONTRACT",
    "RUNTIME_STATE_CONTRACT",
    "SealedAnomalyAdmissionRuntimeFactory",
    "build_anomaly_admission_runtime_factory",
    "build_anomaly_admission_runtime_seal",
    "verify_anomaly_admission_runtime_seal",
]
