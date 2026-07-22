"""Causal exact-28 relative-strength alpha for DOJO room-03.

This module intentionally stops at alpha generation.  It ranks G8 currencies
from synchronized, completed H1 mid closes and emits at most one pair/side
candidate at the immediately following M5 open.  It does not inspect anomaly
features, portfolio state, risk budgets, units, margin, or allocator output.
Those concerns belong to separate rooms/adapters and must not change this
room's direction.

The runtime is research-only.  Its decision is not an order and grants no live
or broker authority.
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

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, G8_CURRENCIES


RUNTIME_SEAL_CONTRACT: Final = "QR_DOJO_G8_RELATIVE_STRENGTH_ALPHA_SEAL_V1"
RUNTIME_STATE_CONTRACT: Final = "QR_DOJO_G8_RELATIVE_STRENGTH_ALPHA_STATE_V1"
ALPHA_DECISION_CONTRACT: Final = "QR_DOJO_G8_RELATIVE_STRENGTH_ALPHA_DECISION_V1"
SCHEMA_VERSION: Final = 1
ALGORITHM_REVISION: Final = "EXACT28_COMPLETED_H1_CROSS_SECTIONAL_RANK_NEXT_M5_OPEN_V1"
FORMAL_G8_PAIRS: Final = tuple(sorted(DEFAULT_TRADER_PAIRS))
FORMAL_G8_CURRENCIES: Final = tuple(sorted(G8_CURRENCIES))

# H1 is an exchange-clock boundary, not a tuned market threshold.  If DOJO
# later supports another ranking timeframe, that requires a versioned runtime
# contract rather than changing this constant in-place.
H1_SECONDS: Final = 60 * 60

# This is an artifact-size guard, not a trading parameter.  Three hundred and
# thirty-six H1 closes cover the largest intended 14-day room-03 lookback while
# keeping the uncompressed exact-28 carry below the runner's 1 MiB state
# envelope.  A measured need for longer state should replace the storage
# envelope and contract together rather than silently truncating history.
MAX_LOOKBACK_H1_BARS: Final = 24 * 14

# Dependency bytes are bounded to prevent a malformed source tree from
# amplifying seal construction.  This is not a market or strategy setting.
MAX_DEPENDENCY_BYTES: Final = 16 * 1024 * 1024

_PHASES: Final = {
    "OHLC": ("O", "H", "L", "C"),
    "OLHC": ("O", "L", "H", "C"),
}
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_KEYS: Final = frozenset({"lookback_h1_bars"})
_SEAL_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "algorithm_revision",
        "dojo_room_id",
        "strategy_family",
        "config",
        "config_sha256",
        "formal_pair_universe",
        "formal_currency_universe",
        "input_contract",
        "signal_timing",
        "output_scope",
        "dependencies",
        "dependencies_sha256",
        "authority",
        "runtime_binding_sha256",
    }
)
_DEPENDENCY_PATHS: Final = (
    "src/quant_rabbit/dojo_relative_strength_alpha_runtime.py",
    "src/quant_rabbit/dojo_training_rooms.py",
    "src/quant_rabbit/instruments.py",
)
_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
    "alpha_only": True,
    "portfolio_sizing_allowed": False,
    "anomaly_veto_allowed": False,
}


class DojoRelativeStrengthAlphaRuntimeError(ValueError):
    """The exact-28 stream, seal, config, or causal carry is invalid."""


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
        raise DojoRelativeStrengthAlphaRuntimeError(
            "value is not strict canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the runtime's strict canonical JSON SHA-256."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoRelativeStrengthAlphaRuntimeError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoRelativeStrengthAlphaRuntimeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DojoRelativeStrengthAlphaRuntimeError(f"{field} must be positive")
    return result


def _validate_formal_universe() -> None:
    if len(FORMAL_G8_PAIRS) != 28 or len(set(FORMAL_G8_PAIRS)) != 28:
        raise DojoRelativeStrengthAlphaRuntimeError(
            "relative-strength alpha requires the exact 28-pair G8 universe"
        )
    if len(FORMAL_G8_CURRENCIES) != 8:
        raise DojoRelativeStrengthAlphaRuntimeError(
            "relative-strength alpha requires exactly eight G8 currencies"
        )
    observed_edges: set[frozenset[str]] = set()
    for pair in FORMAL_G8_PAIRS:
        base, quote = pair.split("_")
        if (
            base == quote
            or base not in FORMAL_G8_CURRENCIES
            or quote not in FORMAL_G8_CURRENCIES
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "formal pair universe contains a non-G8 or self pair"
            )
        observed_edges.add(frozenset({base, quote}))
    expected_edges = {
        frozenset({left, right})
        for index, left in enumerate(FORMAL_G8_CURRENCIES)
        for right in FORMAL_G8_CURRENCIES[index + 1 :]
    }
    if observed_edges != expected_edges:
        raise DojoRelativeStrengthAlphaRuntimeError(
            "formal pair universe is not the complete undirected G8 graph"
        )


def _validate_config(config: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(config, Mapping) or set(config) != set(_CONFIG_KEYS):
        raise DojoRelativeStrengthAlphaRuntimeError(
            "alpha config must contain only lookback_h1_bars"
        )
    lookback = _integer(
        config.get("lookback_h1_bars"), field="lookback_h1_bars", minimum=1
    )
    if lookback > MAX_LOOKBACK_H1_BARS:
        raise DojoRelativeStrengthAlphaRuntimeError(
            "lookback_h1_bars exceeds the bounded carry-state envelope"
        )
    return {"lookback_h1_bars": lookback}


def _file_sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_DEPENDENCY_BYTES
    ):
        raise DojoRelativeStrengthAlphaRuntimeError(
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
        raise DojoRelativeStrengthAlphaRuntimeError(
            f"runtime dependency changed while hashing: {path}"
        )
    return size, digest.hexdigest()


def build_relative_strength_alpha_runtime_seal(
    repo_root: Path, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal the alpha config, exact universe, taxonomy, and source bytes."""

    _validate_formal_universe()
    root = Path(repo_root).resolve(strict=True)
    actual_root = Path(__file__).resolve().parents[2]
    if root != actual_root or not root.is_dir():
        raise DojoRelativeStrengthAlphaRuntimeError(
            "repo_root must be the source tree that loaded this runtime"
        )
    validated_config = _validate_config(config)
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
        "algorithm_revision": ALGORITHM_REVISION,
        "dojo_room_id": "room-03",
        "strategy_family": "g8_relative_strength_alpha",
        "config": validated_config,
        "config_sha256": canonical_sha256(validated_config),
        "formal_pair_universe": list(FORMAL_G8_PAIRS),
        "formal_currency_universe": list(FORMAL_G8_CURRENCIES),
        "input_contract": "CAUSAL_EXACT28_COMPLETED_H1_STRENGTH",
        "signal_timing": "COMPLETED_H1_RANK_EXECUTE_NEXT_M5_OPEN",
        "output_scope": "PAIR_SIDE_ALPHA_ONLY_NO_UNITS",
        "dependencies": dependencies,
        "dependencies_sha256": canonical_sha256(dependencies),
        "authority": dict(_AUTHORITY),
    }
    return {**body, "runtime_binding_sha256": canonical_sha256(body)}


def verify_relative_strength_alpha_runtime_seal(
    seal: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Rebuild one seal from current source/config and reject any drift."""

    if not isinstance(seal, Mapping):
        raise DojoRelativeStrengthAlphaRuntimeError("runtime seal must be an object")
    config = seal.get("config")
    if not isinstance(config, Mapping):
        raise DojoRelativeStrengthAlphaRuntimeError("runtime seal config is missing")
    rebuilt = build_relative_strength_alpha_runtime_seal(repo_root, config=config)
    if _canonical_bytes(seal) != _canonical_bytes(rebuilt):
        raise DojoRelativeStrengthAlphaRuntimeError(
            "runtime seal differs from current config or source bytes"
        )
    return rebuilt


class RelativeStrengthAlphaRuntime:
    """Stateful exact-28 H1 alpha generator with no sizing authority."""

    def __init__(
        self,
        *,
        seal: Mapping[str, Any],
        cadence_seconds: int,
        prior_state: Mapping[str, Any] | None = None,
    ) -> None:
        self._seal = _copy(seal)
        if set(self._seal) != set(_SEAL_KEYS):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal schema is invalid"
            )
        if (
            self._seal.get("contract") != RUNTIME_SEAL_CONTRACT
            or self._seal.get("schema_version") != SCHEMA_VERSION
            or self._seal.get("algorithm_revision") != ALGORITHM_REVISION
            or self._seal.get("dojo_room_id") != "room-03"
            or self._seal.get("strategy_family") != "g8_relative_strength_alpha"
            or self._seal.get("formal_pair_universe") != list(FORMAL_G8_PAIRS)
            or self._seal.get("formal_currency_universe") != list(FORMAL_G8_CURRENCIES)
            or self._seal.get("input_contract")
            != "CAUSAL_EXACT28_COMPLETED_H1_STRENGTH"
            or self._seal.get("signal_timing")
            != "COMPLETED_H1_RANK_EXECUTE_NEXT_M5_OPEN"
            or self._seal.get("output_scope") != "PAIR_SIDE_ALPHA_ONLY_NO_UNITS"
            or self._seal.get("authority") != _AUTHORITY
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal identity or capability boundary is invalid"
            )
        binding = self._seal.get("runtime_binding_sha256")
        if not isinstance(binding, str) or _SHA256_RE.fullmatch(binding) is None:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal binding is invalid"
            )
        seal_body = {
            key: value
            for key, value in self._seal.items()
            if key != "runtime_binding_sha256"
        }
        if canonical_sha256(seal_body) != binding:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal content hash is invalid"
            )
        self._lookback = _validate_config(self._seal["config"])["lookback_h1_bars"]
        if self._seal.get("config_sha256") != canonical_sha256(self._seal["config"]):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal config hash is invalid"
            )
        dependencies = self._seal.get("dependencies")
        if not isinstance(dependencies, list) or self._seal.get(
            "dependencies_sha256"
        ) != canonical_sha256(dependencies):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runtime seal dependency hash is invalid"
            )
        self._required_closes = self._lookback + 1
        self._cadence = _integer(cadence_seconds, field="cadence_seconds", minimum=1)
        if H1_SECONDS % self._cadence != 0:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "runner cadence must divide the H1 boundary"
            )
        self._expected_subbars = H1_SECONDS // self._cadence
        self._completed: dict[str, list[dict[str, Any]]] = {
            pair: [] for pair in FORMAL_G8_PAIRS
        }
        self._forming: dict[str, Any] | None = None
        self._last_epoch: int | None = None
        self._last_phase: str | None = None
        self._last_intrabar: str | None = None
        self._last_decision_h1_close_epoch: int | None = None
        self._decision_count = 0
        self._enter_count = 0
        self._hold_counts: dict[str, int] = {}
        if prior_state is not None:
            self._restore(prior_state)

    def _restore(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "contract",
            "schema_version",
            "runtime_binding_sha256",
            "cadence_seconds",
            "completed_h1",
            "forming_h1",
            "last_epoch",
            "last_phase",
            "last_intrabar",
            "last_decision_h1_close_epoch",
            "decision_count",
            "enter_count",
            "hold_counts",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "relative-strength carry schema mismatch"
            )
        if (
            state["contract"] != RUNTIME_STATE_CONTRACT
            or state["schema_version"] != SCHEMA_VERSION
            or state["runtime_binding_sha256"] != self._seal["runtime_binding_sha256"]
            or state["cadence_seconds"] != self._cadence
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "relative-strength carry binding drifted"
            )
        completed = state["completed_h1"]
        if not isinstance(completed, Mapping) or set(completed) != set(FORMAL_G8_PAIRS):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "carry must contain the exact-28 completed-H1 matrix"
            )
        copied_completed = _copy(completed)
        epochs: list[int] | None = None
        for pair in FORMAL_G8_PAIRS:
            rows = copied_completed[pair]
            if not isinstance(rows, list) or len(rows) > self._required_closes:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "completed-H1 carry history exceeds its sealed lookback"
                )
            pair_epochs = []
            for row in rows:
                if not isinstance(row, Mapping) or set(row) != {
                    "close_epoch",
                    "bid_close",
                    "ask_close",
                }:
                    raise DojoRelativeStrengthAlphaRuntimeError(
                        "completed-H1 carry row is invalid"
                    )
                pair_epochs.append(
                    _integer(row["close_epoch"], field="close_epoch", minimum=1)
                )
                bid = _positive(row["bid_close"], field="bid_close")
                ask = _positive(row["ask_close"], field="ask_close")
                if ask < bid:
                    raise DojoRelativeStrengthAlphaRuntimeError(
                        "completed-H1 carry ask crossed below bid"
                    )
            if epochs is None:
                epochs = pair_epochs
            elif pair_epochs != epochs:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "completed-H1 carry histories are unsynchronized"
                )
        if epochs is not None and any(
            later - earlier != H1_SECONDS for earlier, later in zip(epochs, epochs[1:])
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "completed-H1 carry contains a discontinuity"
            )
        forming = state["forming_h1"]
        if forming is not None and not isinstance(forming, Mapping):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "forming-H1 carry must be an object or null"
            )
        copied_forming = _copy(forming)
        if copied_forming is not None:
            if set(copied_forming) != {
                "bucket_start",
                "first_epoch",
                "subbar_count",
                "pairs",
            }:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "forming-H1 carry schema is invalid"
                )
            bucket_start = _integer(
                copied_forming["bucket_start"], field="bucket_start"
            )
            first_epoch = _integer(copied_forming["first_epoch"], field="first_epoch")
            subbar_count = _integer(
                copied_forming["subbar_count"], field="subbar_count"
            )
            pairs = copied_forming["pairs"]
            if (
                bucket_start % H1_SECONDS != 0
                or not bucket_start <= first_epoch < bucket_start + H1_SECONDS
                or subbar_count > self._expected_subbars
                or not isinstance(pairs, Mapping)
                or set(pairs) != set(FORMAL_G8_PAIRS)
            ):
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "forming-H1 carry binding is invalid"
                )
            for pair in FORMAL_G8_PAIRS:
                row = pairs[pair]
                if not isinstance(row, Mapping) or set(row) != {
                    "bid_close",
                    "ask_close",
                }:
                    raise DojoRelativeStrengthAlphaRuntimeError(
                        "forming-H1 carry quote is invalid"
                    )
                bid = _positive(row["bid_close"], field="bid_close")
                ask = _positive(row["ask_close"], field="ask_close")
                if ask < bid:
                    raise DojoRelativeStrengthAlphaRuntimeError(
                        "forming-H1 carry ask crossed below bid"
                    )
        decision_count = _integer(state["decision_count"], field="decision_count")
        enter_count = _integer(state["enter_count"], field="enter_count")
        if enter_count > decision_count:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "carry enter_count exceeds decision_count"
            )
        hold_counts = state["hold_counts"]
        if (
            not isinstance(hold_counts, Mapping)
            or not all(isinstance(key, str) and key for key in hold_counts)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in hold_counts.values()
            )
            or sum(hold_counts.values()) + enter_count != decision_count
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "carry HOLD counters do not reconcile"
            )
        last_epoch = state["last_epoch"]
        last_phase = state["last_phase"]
        last_intrabar = state["last_intrabar"]
        if last_epoch is None:
            if last_phase is not None or last_intrabar is not None:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "empty carry clock is inconsistent"
                )
        else:
            last_epoch = _integer(last_epoch, field="last_epoch")
            if last_intrabar not in _PHASES or last_phase not in _PHASES[last_intrabar]:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "carry clock phase/path is invalid"
                )
            if epochs and epochs[-1] > last_epoch + self._cadence:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "completed-H1 carry is from the future"
                )
            if copied_forming is not None and not (
                copied_forming["bucket_start"]
                <= last_epoch
                < copied_forming["bucket_start"] + H1_SECONDS
            ):
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "forming-H1 carry is not bound to the carry clock"
                )
        last_decision = state["last_decision_h1_close_epoch"]
        if last_decision is not None:
            last_decision = _integer(
                last_decision, field="last_decision_h1_close_epoch", minimum=1
            )
            if last_epoch is None or last_decision > last_epoch:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "last alpha decision is from the future"
                )
        self._completed = copied_completed
        self._forming = copied_forming
        self._last_epoch = last_epoch
        self._last_phase = last_phase
        self._last_intrabar = last_intrabar
        self._last_decision_h1_close_epoch = last_decision
        self._decision_count = decision_count
        self._enter_count = enter_count
        self._hold_counts = dict(hold_counts)

    def _reset_market_history(self) -> None:
        self._completed = {pair: [] for pair in FORMAL_G8_PAIRS}
        self._forming = None

    def _validate_and_advance_clock(
        self, snapshot: Mapping[str, Any]
    ) -> tuple[int, str, str]:
        epoch = _integer(snapshot.get("epoch"), field="snapshot.epoch", minimum=0)
        phase = snapshot.get("phase")
        intrabar = snapshot.get("intrabar")
        if intrabar not in _PHASES or phase not in _PHASES[intrabar]:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "snapshot phase/path is invalid"
            )
        phase_order = _PHASES[intrabar]
        if self._last_intrabar is not None and self._last_intrabar != intrabar:
            raise DojoRelativeStrengthAlphaRuntimeError("intrabar path drifted")
        if self._last_epoch is None:
            if phase != "O":
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "relative-strength stream must start at O"
                )
        elif epoch == self._last_epoch:
            if (
                self._last_phase not in phase_order
                or phase_order.index(phase) != phase_order.index(self._last_phase) + 1
            ):
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "snapshot phase sequence is non-causal"
                )
        elif epoch > self._last_epoch:
            if self._last_phase != "C" or phase != "O":
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "new M5 epoch did not follow a completed candle"
                )
            if epoch - self._last_epoch != self._cadence:
                self._reset_market_history()
        else:
            raise DojoRelativeStrengthAlphaRuntimeError("snapshot epoch moved backward")
        self._last_epoch = epoch
        self._last_phase = str(phase)
        self._last_intrabar = str(intrabar)
        return epoch, str(phase), str(intrabar)

    def _exact_quotes_or_none(
        self, snapshot: Mapping[str, Any]
    ) -> dict[str, Mapping[str, Any]] | None:
        expected = snapshot.get("expected_quote_pairs")
        rows = snapshot.get("quotes")
        if (
            isinstance(expected, (str, bytes))
            or not isinstance(expected, Sequence)
            or set(expected) != set(FORMAL_G8_PAIRS)
            or len(expected) != len(FORMAL_G8_PAIRS)
            or isinstance(rows, (str, bytes))
            or not isinstance(rows, Sequence)
        ):
            return None
        quotes = {row.get("pair"): row for row in rows if isinstance(row, Mapping)}
        if set(quotes) != set(FORMAL_G8_PAIRS) or len(rows) != len(quotes):
            return None
        return quotes

    def _consume_quotes(
        self,
        *,
        epoch: int,
        phase: str,
        quotes: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        bucket_start = epoch - (epoch % H1_SECONDS)
        if phase == "O":
            if self._forming is None or self._forming["bucket_start"] != bucket_start:
                if self._forming is not None:
                    self._reset_market_history()
                self._forming = {
                    "bucket_start": bucket_start,
                    "first_epoch": epoch,
                    "subbar_count": 0,
                    "pairs": {},
                }
            for pair in FORMAL_G8_PAIRS:
                bid = _positive(quotes[pair].get("bid"), field=f"{pair}.bid")
                ask = _positive(quotes[pair].get("ask"), field=f"{pair}.ask")
                if ask < bid:
                    raise DojoRelativeStrengthAlphaRuntimeError(
                        "snapshot ask crossed below bid"
                    )
                if pair not in self._forming["pairs"]:
                    self._forming["pairs"][pair] = {
                        "bid_close": bid,
                        "ask_close": ask,
                    }
        if self._forming is None or self._forming["bucket_start"] != bucket_start:
            return False
        for pair in FORMAL_G8_PAIRS:
            bid = _positive(quotes[pair].get("bid"), field=f"{pair}.bid")
            ask = _positive(quotes[pair].get("ask"), field=f"{pair}.ask")
            if ask < bid:
                raise DojoRelativeStrengthAlphaRuntimeError(
                    "snapshot ask crossed below bid"
                )
            self._forming["pairs"][pair] = {
                "bid_close": bid,
                "ask_close": ask,
            }
        if phase != "C":
            return True
        self._forming["subbar_count"] += 1
        close_epoch = epoch + self._cadence
        if close_epoch != bucket_start + H1_SECONDS:
            return True
        complete = (
            self._forming["first_epoch"] == bucket_start
            and self._forming["subbar_count"] == self._expected_subbars
            and set(self._forming["pairs"]) == set(FORMAL_G8_PAIRS)
        )
        if not complete:
            self._reset_market_history()
            return False
        for pair in FORMAL_G8_PAIRS:
            self._completed[pair].append(
                {
                    "close_epoch": close_epoch,
                    **self._forming["pairs"][pair],
                }
            )
            self._completed[pair] = self._completed[pair][-self._required_closes :]
        self._forming = None
        return True

    def _strength_candidate(self) -> dict[str, Any] | None:
        if any(
            len(self._completed[pair]) < self._required_closes
            for pair in FORMAL_G8_PAIRS
        ):
            return None
        sums = {currency: 0.0 for currency in FORMAL_G8_CURRENCIES}
        counts = {currency: 0 for currency in FORMAL_G8_CURRENCIES}
        close_epochs: set[int] = set()
        for pair in FORMAL_G8_PAIRS:
            rows = self._completed[pair]
            start = rows[-self._required_closes]
            end = rows[-1]
            start_mid = (float(start["bid_close"]) + float(start["ask_close"])) / 2.0
            end_mid = (float(end["bid_close"]) + float(end["ask_close"])) / 2.0
            pair_return = math.log(end_mid / start_mid)
            base, quote = pair.split("_")
            sums[base] += pair_return
            counts[base] += 1
            sums[quote] -= pair_return
            counts[quote] += 1
            close_epochs.add(int(end["close_epoch"]))
        if close_epochs != {next(iter(close_epochs))}:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "completed-H1 rank panel is not synchronized"
            )
        if set(counts.values()) != {7}:
            raise DojoRelativeStrengthAlphaRuntimeError(
                "each G8 currency must receive exactly seven pair votes"
            )
        scores = {
            currency: sums[currency] / counts[currency]
            for currency in FORMAL_G8_CURRENCIES
        }
        ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
        strongest, strongest_score = ranked[0]
        weakest, weakest_score = ranked[-1]
        if not strongest_score > weakest_score:
            return {
                "close_epoch": next(iter(close_epochs)),
                "scores": scores,
                "ranked_currencies": [currency for currency, _score in ranked],
                "pair": None,
                "side": None,
                "strength_dispersion": strongest_score - weakest_score,
            }
        direct = f"{strongest}_{weakest}"
        inverse = f"{weakest}_{strongest}"
        if direct in FORMAL_G8_PAIRS:
            pair, side = direct, "LONG"
        elif inverse in FORMAL_G8_PAIRS:
            pair, side = inverse, "SHORT"
        else:  # pragma: no cover - complete graph validation makes unreachable
            raise DojoRelativeStrengthAlphaRuntimeError(
                "strongest/weakest pair is outside the complete G8 graph"
            )
        return {
            "close_epoch": next(iter(close_epochs)),
            "scores": scores,
            "ranked_currencies": [currency for currency, _score in ranked],
            "pair": pair,
            "side": side,
            "strength_dispersion": strongest_score - weakest_score,
        }

    def _decision(
        self,
        *,
        snapshot_sha256: Any,
        epoch: int,
        phase: str,
        status: str,
        reason_code: str,
        candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(snapshot_sha256, str)
            or _SHA256_RE.fullmatch(snapshot_sha256) is None
        ):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "snapshot_sha256 must be a lowercase SHA-256 digest"
            )
        body = {
            "contract": ALPHA_DECISION_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "runtime_binding_sha256": self._seal["runtime_binding_sha256"],
            "dojo_room_id": "room-03",
            "strategy_family": "g8_relative_strength_alpha",
            "snapshot_sha256": snapshot_sha256,
            "snapshot_epoch": epoch,
            "snapshot_phase": phase,
            "status": status,
            "reason_code": reason_code,
            "pair": None if candidate is None else candidate.get("pair"),
            "side": None if candidate is None else candidate.get("side"),
            "signal_h1_close_epoch": (
                None if candidate is None else candidate.get("close_epoch")
            ),
            "execution_epoch": epoch if status == "ENTER" else None,
            "execution_timing": (
                "NEXT_M5_OPEN_AFTER_COMPLETED_H1" if status == "ENTER" else None
            ),
            "lookback_h1_bars": self._lookback,
            "currency_scores": (None if candidate is None else candidate.get("scores")),
            "ranked_currencies": (
                None if candidate is None else candidate.get("ranked_currencies")
            ),
            "strength_dispersion": (
                None if candidate is None else candidate.get("strength_dispersion")
            ),
            "output_scope": "PAIR_SIDE_ALPHA_ONLY_NO_UNITS",
            "authority": dict(_AUTHORITY),
        }
        result = {**body, "decision_sha256": canonical_sha256(body)}
        self._decision_count += 1
        if status == "ENTER":
            self._enter_count += 1
        else:
            self._hold_counts[reason_code] = self._hold_counts.get(reason_code, 0) + 1
        return result

    def observe(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Consume one M5 phase and return one sealed ENTER/HOLD alpha decision."""

        if not isinstance(snapshot, Mapping):
            raise DojoRelativeStrengthAlphaRuntimeError(
                "worker snapshot must be an object"
            )
        epoch, phase, _intrabar = self._validate_and_advance_clock(snapshot)
        quotes = self._exact_quotes_or_none(snapshot)
        if quotes is None:
            self._reset_market_history()
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="EXACT28_QUOTE_BATCH_MISSING_HOLD",
            )
        continuous = self._consume_quotes(epoch=epoch, phase=phase, quotes=quotes)
        if not continuous:
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="H1_DISCONTINUITY_HOLD",
            )
        if phase != "O":
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="WAIT_FOR_NEXT_M5_OPEN_HOLD",
            )
        candidate = self._strength_candidate()
        if candidate is None:
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="COMPLETED_H1_WARMUP_HOLD",
            )
        if candidate["close_epoch"] != epoch:
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="NOT_IMMEDIATE_POST_H1_M5_OPEN_HOLD",
                candidate=candidate,
            )
        if self._last_decision_h1_close_epoch == candidate["close_epoch"]:
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="H1_ALPHA_ALREADY_EMITTED_HOLD",
                candidate=candidate,
            )
        self._last_decision_h1_close_epoch = int(candidate["close_epoch"])
        if candidate["pair"] is None:
            return self._decision(
                snapshot_sha256=snapshot.get("snapshot_sha256"),
                epoch=epoch,
                phase=phase,
                status="HOLD",
                reason_code="NO_CROSS_SECTIONAL_DISPERSION_HOLD",
                candidate=candidate,
            )
        return self._decision(
            snapshot_sha256=snapshot.get("snapshot_sha256"),
            epoch=epoch,
            phase=phase,
            status="ENTER",
            reason_code="STRONGEST_VERSUS_WEAKEST_COMPLETED_H1_ALPHA",
            candidate=candidate,
        )

    def export_state(self) -> dict[str, Any]:
        """Export bounded causal carry without any portfolio or risk state."""

        return {
            "contract": RUNTIME_STATE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "runtime_binding_sha256": self._seal["runtime_binding_sha256"],
            "cadence_seconds": self._cadence,
            "completed_h1": _copy(self._completed),
            "forming_h1": _copy(self._forming),
            "last_epoch": self._last_epoch,
            "last_phase": self._last_phase,
            "last_intrabar": self._last_intrabar,
            "last_decision_h1_close_epoch": self._last_decision_h1_close_epoch,
            "decision_count": self._decision_count,
            "enter_count": self._enter_count,
            "hold_counts": _copy(self._hold_counts),
        }


__all__ = [
    "ALGORITHM_REVISION",
    "ALPHA_DECISION_CONTRACT",
    "DojoRelativeStrengthAlphaRuntimeError",
    "FORMAL_G8_CURRENCIES",
    "FORMAL_G8_PAIRS",
    "RUNTIME_SEAL_CONTRACT",
    "RUNTIME_STATE_CONTRACT",
    "RelativeStrengthAlphaRuntime",
    "build_relative_strength_alpha_runtime_seal",
    "canonical_sha256",
    "verify_relative_strength_alpha_runtime_seal",
]
