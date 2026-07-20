"""Capability-closed strategy workers for the DOJO long-horizon runner.

The runtime is deliberately a finite built-in catalog, not a plugin surface.
It adapts four already-reviewed ``bots/lab_bot.py`` families to the shared
post-exit snapshot/proposal protocol.  It has no import hook, broker handle,
filesystem handle, network path, allocator authority, or live/order authority.

Every invocation acknowledges every active worker.  A no-action decision is
an explicit proposal with two empty intent arrays.  Causal indicator state is
strict JSON and may only cross a month boundary through the economic runner's
sealed carry artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import bot_config_sha256, validate_bot_config
from quant_rabbit.dojo_lab_provenance import canonical_strategy_owner_id
from quant_rabbit.dojo_strategy_worker_factory import build_baseline_worker_cohort


RUNTIME_SEAL_CONTRACT: Final = "QR_DOJO_BUILTIN_STRATEGY_RUNTIME_SEAL_V1"
RUNTIME_STATE_CONTRACT: Final = "QR_DOJO_BUILTIN_STRATEGY_RUNTIME_STATE_V1"
SCHEMA_VERSION: Final = 1
ALGORITHM_REVISION: Final = "LAB_FAMILY_POST_EXIT_ADAPTER_V1"
GENESIS_SNAPSHOT_SHA256: Final = "0" * 64

# Artifact-shape limits cap memory and dependency amplification; they are not
# market parameters.  A later protocol revision should replace them only if a
# measured sealed artifact exceeds the current envelope.
MAX_DEPENDENCY_BYTES: Final = 8 * 1024 * 1024
MAX_CLOSE_HISTORY: Final = 1441
MAX_DIFF_HISTORY: Final = 360
MAX_WIDTH_HISTORY: Final = 720
MAX_WIDTH_WINDOW: Final = 20

# These market constants reproduce the already-reviewed ``bots/lab_bot.py``
# baseline: 1,441 M1 closes express a rolling 24-hour comparison, 360 M1 moves
# express six-hour efficiency, 720 widths express 12 hours, and Wilder 14 is
# the source strategy's ATR memory.  They are fixed for byte-stable generation
# 1 parity; a trainer may replace them only through a newly sealed algorithm
# revision/config dependency, never by mutating a running replay.
ATR_WILDER_PERIOD: Final = 14.0
COMPRESSION_RANK_CUTOFF: Final = 0.2
COMPRESSION_WARMUP_WIDTHS: Final = 360
COMPRESSION_ENTRY_BUFFER_PIPS: Final = 2.0
SPIKE_RANGE_ATR_MULTIPLE: Final = 2.5
SPREAD_TO_TARGET_CAP: Final = 0.35

# These paths are a closed dependency denominator, not caller-selected input.
# ``bots/lab_bot.py`` is included because this adapter claims lineage from its
# family definitions even though no executable broker object is imported.
_DEPENDENCY_PATHS: Final = (
    "bots/lab_bot.py",
    "src/quant_rabbit/dojo_bot_catalog.py",
    "src/quant_rabbit/dojo_builtin_strategy_runtime.py",
    "src/quant_rabbit/dojo_lab_provenance.py",
    "src/quant_rabbit/dojo_shared_worker_protocol.py",
    "src/quant_rabbit/dojo_strategy_worker_factory.py",
)
_FAMILIES: Final = frozenset(
    {
        "compression_break",
        "daily_break_pullback",
        "range_fade_limit",
        "spike_fade",
    }
)


class DojoBuiltinStrategyRuntimeError(ValueError):
    """A built-in strategy seal, binding, snapshot, or carry state is invalid."""


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
        raise DojoBuiltinStrategyRuntimeError("value is not strict JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > MAX_DEPENDENCY_BYTES
    ):
        raise DojoBuiltinStrategyRuntimeError(
            f"runtime dependency is not a bounded regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if before_identity != after_identity or after_identity != current_identity:
        raise DojoBuiltinStrategyRuntimeError(
            f"runtime dependency changed while hashing: {path}"
        )
    if size != before.st_size:
        raise DojoBuiltinStrategyRuntimeError(
            f"runtime dependency read was incomplete: {path}"
        )
    return size, digest.hexdigest()


def _worker_rows() -> list[dict[str, Any]]:
    cohort = build_baseline_worker_cohort()
    proposals = cohort["trainer_candidate_proposals"]
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        config = validate_bot_config(proposal["config"])
        family = config["signal"]
        if family not in _FAMILIES or proposal["family"] != family:
            raise DojoBuiltinStrategyRuntimeError(
                "baseline cohort contains a non-built-in or mismatched family"
            )
        config_sha = bot_config_sha256(config)
        rows.append(
            {
                "worker_id": proposal["candidate_id"],
                "owner_id": canonical_strategy_owner_id(
                    config, namespace="dojo-long"
                ),
                "family_id": family,
                "config_sha256": config_sha,
                "config": config,
                "algorithm_revision": ALGORITHM_REVISION,
            }
        )
    rows.sort(key=lambda row: row["worker_id"])
    if len(rows) != len(_FAMILIES) or {row["family_id"] for row in rows} != _FAMILIES:
        raise DojoBuiltinStrategyRuntimeError(
            "built-in strategy denominator is not exactly four distinct families"
        )
    return rows


def build_builtin_strategy_runtime_seal(repo_root: Path) -> dict[str, Any]:
    """Seal the exact built-in workers, configs, algorithms, and source bytes."""

    root = Path(repo_root).resolve(strict=True)
    actual_root = Path(__file__).resolve().parents[2]
    if not root.is_dir() or root != actual_root:
        raise DojoBuiltinStrategyRuntimeError(
            "repo_root must be the source tree that loaded this built-in runtime"
        )
    dependencies = []
    for relative_path in _DEPENDENCY_PATHS:
        path = root / relative_path
        size, digest = _file_sha256(path)
        dependencies.append(
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    workers = _worker_rows()
    catalog = [
        {
            key: row[key]
            for key in ("worker_id", "owner_id", "family_id", "config_sha256")
        }
        for row in workers
    ]
    body = {
        "contract": RUNTIME_SEAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "algorithm_revision": ALGORITHM_REVISION,
        "runtime_mode": "SEALED_BUILTIN_MULTI_STRATEGY",
        "worker_count": len(workers),
        "worker_catalog": catalog,
        "worker_catalog_sha256": _sha256(catalog),
        "workers": workers,
        "workers_sha256": _sha256(workers),
        "dependencies": dependencies,
        "dependencies_sha256": _sha256(dependencies),
        "capabilities": {
            "arbitrary_import_allowed": False,
            "broker_handle_available": False,
            "broker_mutation_allowed": False,
            "external_code_loading_allowed": False,
            "filesystem_available_to_worker": False,
            "live_permission": False,
            "network_available_to_worker": False,
            "order_authority": "NONE",
            "proposal_only": True,
        },
        "evidence": {
            "diagnostic_only": True,
            "independent_reexecution_available": False,
            "official_evidence_eligible": False,
            "promotion_eligible": False,
            "three_x_guaranteed": False,
        },
    }
    return {**body, "runtime_binding_sha256": _sha256(body)}


def verify_builtin_strategy_runtime_seal(
    seal: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Rebuild the seal from stable local bytes and require exact equality."""

    if not isinstance(seal, Mapping):
        raise DojoBuiltinStrategyRuntimeError("runtime seal must be an object")
    rebuilt = build_builtin_strategy_runtime_seal(repo_root)
    detached = json.loads(_canonical_bytes(seal).decode("utf-8"))
    if detached != rebuilt:
        raise DojoBuiltinStrategyRuntimeError(
            "runtime seal differs from the current closed dependency denominator"
        )
    return rebuilt


def _empty_pair_state() -> dict[str, Any]:
    return {
        "atr": None,
        "closes": [],
        "diffs_6h": [],
        "widths": [],
        "forming": None,
        "last_closed_epoch": None,
        "day": None,
        "day_high": None,
        "day_low": None,
        "day_observation_count": 0,
        "previous_day_high": None,
        "previous_day_low": None,
    }


def _strict_prior_state(
    value: Any,
    *,
    bindings: Sequence[Mapping[str, str]],
    trade_pairs: Sequence[str],
) -> dict[str, Any]:
    expected_workers = {row["worker_id"]: row for row in bindings}
    if value is None:
        workers = {
            worker_id: {
                "config_sha256": binding["config_sha256"],
                "hold_ack_count": 0,
                "intent_proposal_count": 0,
                "pairs": {pair: _empty_pair_state() for pair in trade_pairs},
            }
            for worker_id, binding in expected_workers.items()
        }
        return {
            "contract": RUNTIME_STATE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "algorithm_revision": ALGORITHM_REVISION,
            "call_count": 0,
            "last_snapshot_sha256": GENESIS_SNAPSHOT_SHA256,
            "last_epoch": None,
            "last_phase": None,
            "last_intrabar": None,
            "last_quote_watermark": 0,
            "trade_pairs": list(trade_pairs),
            "workers": workers,
        }
    if not isinstance(value, Mapping):
        raise DojoBuiltinStrategyRuntimeError("prior worker state must be an object")
    expected_keys = {
        "contract",
        "schema_version",
        "algorithm_revision",
        "call_count",
        "last_snapshot_sha256",
        "last_epoch",
        "last_phase",
        "last_intrabar",
        "last_quote_watermark",
        "trade_pairs",
        "workers",
    }
    if set(value) != expected_keys:
        raise DojoBuiltinStrategyRuntimeError("prior worker state schema is not exact")
    if (
        value["contract"] != RUNTIME_STATE_CONTRACT
        or value["schema_version"] != SCHEMA_VERSION
        or value["algorithm_revision"] != ALGORITHM_REVISION
        or value["trade_pairs"] != list(trade_pairs)
        or not isinstance(value["call_count"], int)
        or isinstance(value["call_count"], bool)
        or value["call_count"] < 0
        or not isinstance(value["last_snapshot_sha256"], str)
        or len(value["last_snapshot_sha256"]) != 64
        or (
            value["last_epoch"] is not None
            and (
                not isinstance(value["last_epoch"], int)
                or isinstance(value["last_epoch"], bool)
                or value["last_epoch"] < 0
            )
        )
        or value["last_phase"] not in {None, "O", "H", "L", "C"}
        or value["last_intrabar"] not in {None, "OHLC", "OLHC"}
        or not isinstance(value["last_quote_watermark"], int)
        or isinstance(value["last_quote_watermark"], bool)
        or value["last_quote_watermark"] < 0
        or not isinstance(value["workers"], Mapping)
        or set(value["workers"]) != set(expected_workers)
    ):
        raise DojoBuiltinStrategyRuntimeError("prior worker state binding drifted")
    # Canonical round-trip rejects non-JSON values before any mutable copy.
    detached = json.loads(_canonical_bytes(value).decode("utf-8"))
    for worker_id, binding in expected_workers.items():
        worker = detached["workers"].get(worker_id)
        if (
            not isinstance(worker, dict)
            or set(worker)
            != {
                "config_sha256",
                "hold_ack_count",
                "intent_proposal_count",
                "pairs",
            }
            or worker["config_sha256"] != binding["config_sha256"]
            or not isinstance(worker["hold_ack_count"], int)
            or not isinstance(worker["intent_proposal_count"], int)
            or worker["hold_ack_count"] < 0
            or worker["intent_proposal_count"] < 0
            or not isinstance(worker["pairs"], dict)
            or set(worker["pairs"]) != set(trade_pairs)
        ):
            raise DojoBuiltinStrategyRuntimeError(
                "prior worker state identity/counter binding drifted"
            )
        for pair_state in worker["pairs"].values():
            _validate_pair_state(pair_state)
    return detached


def _validate_pair_state(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_empty_pair_state()):
        raise DojoBuiltinStrategyRuntimeError("prior pair state schema is not exact")
    for key, maximum in (
        ("closes", MAX_CLOSE_HISTORY),
        ("diffs_6h", MAX_DIFF_HISTORY),
        ("widths", MAX_WIDTH_HISTORY),
    ):
        rows = value[key]
        if (
            not isinstance(rows, list)
            or len(rows) > maximum
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in rows
            )
        ):
            raise DojoBuiltinStrategyRuntimeError(f"invalid prior {key}")
    for key in (
        "atr",
        "day_high",
        "day_low",
        "previous_day_high",
        "previous_day_low",
    ):
        item = value[key]
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise DojoBuiltinStrategyRuntimeError(f"invalid prior {key}")
    if (
        not isinstance(value["day_observation_count"], int)
        or isinstance(value["day_observation_count"], bool)
        or value["day_observation_count"] < 0
        or (value["day"] is not None and not isinstance(value["day"], str))
        or (
            value["last_closed_epoch"] is not None
            and (
                not isinstance(value["last_closed_epoch"], int)
                or isinstance(value["last_closed_epoch"], bool)
                or value["last_closed_epoch"] < 0
            )
        )
    ):
        raise DojoBuiltinStrategyRuntimeError("invalid prior pair clock state")
    forming = value["forming"]
    if forming is not None:
        if not isinstance(forming, Mapping) or set(forming) != {"epoch", "o", "h", "l", "c"}:
            raise DojoBuiltinStrategyRuntimeError("invalid prior forming bar")
        if (
            not isinstance(forming["epoch"], int)
            or isinstance(forming["epoch"], bool)
            or any(
                isinstance(forming[key], bool)
                or not isinstance(forming[key], (int, float))
                or not math.isfinite(float(forming[key]))
                or float(forming[key]) <= 0
                for key in ("o", "h", "l", "c")
            )
        ):
            raise DojoBuiltinStrategyRuntimeError("invalid prior forming bar values")


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _round_price(pair: str, price: float) -> float:
    return round(price, 3 if pair.endswith("JPY") else 5)


def _append_bounded(rows: list[float], value: float, maximum: int) -> None:
    rows.append(float(value))
    if len(rows) > maximum:
        del rows[: len(rows) - maximum]


def _consume_quote_phase(
    state: dict[str, Any], *, epoch: int, phase: str, mid: float
) -> dict[str, float] | None:
    forming = state["forming"]
    if forming is None or forming["epoch"] != epoch:
        if forming is not None:
            raise DojoBuiltinStrategyRuntimeError(
                "new candle arrived before the prior C-phase decision"
            )
        forming = {"epoch": epoch, "o": mid, "h": mid, "l": mid, "c": mid}
        state["forming"] = forming
    forming["h"] = max(float(forming["h"]), mid)
    forming["l"] = min(float(forming["l"]), mid)
    forming["c"] = mid
    if phase != "C":
        return None
    if state["last_closed_epoch"] is not None and epoch <= state["last_closed_epoch"]:
        raise DojoBuiltinStrategyRuntimeError("closed-bar clock did not advance")
    bar = {key: float(forming[key]) for key in ("o", "h", "l", "c")}
    prior_close = state["closes"][-1] if state["closes"] else None
    if prior_close is not None:
        true_range = max(
            bar["h"] - bar["l"],
            abs(bar["h"] - prior_close),
            abs(bar["l"] - prior_close),
        )
        atr = state["atr"]
        state["atr"] = (
            true_range
            if atr is None
            else float(atr) + (true_range - float(atr)) / ATR_WILDER_PERIOD
        )
        _append_bounded(
            state["diffs_6h"], abs(bar["c"] - prior_close), MAX_DIFF_HISTORY
        )
    _append_bounded(state["closes"], bar["c"], MAX_CLOSE_HISTORY)
    if len(state["closes"]) >= MAX_WIDTH_WINDOW:
        recent = state["closes"][-MAX_WIDTH_WINDOW:]
        _append_bounded(
            state["widths"], max(recent) - min(recent), MAX_WIDTH_HISTORY
        )
    day = datetime.fromtimestamp(epoch, timezone.utc).date().isoformat()
    if state["day"] != day:
        if state["day"] is not None and state["day_observation_count"] > 0:
            state["previous_day_high"] = state["day_high"]
            state["previous_day_low"] = state["day_low"]
        state["day"] = day
        state["day_high"] = bar["h"]
        state["day_low"] = bar["l"]
        state["day_observation_count"] = 1
    else:
        state["day_high"] = max(float(state["day_high"]), bar["h"])
        state["day_low"] = min(float(state["day_low"]), bar["l"])
        state["day_observation_count"] += 1
    state["last_closed_epoch"] = epoch
    state["forming"] = None
    return bar


def _quote_index(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["pair"]: row for row in snapshot["quotes"]}


def _quote_to_jpy_rate(
    currency: str, quotes: Mapping[str, Mapping[str, Any]]
) -> float | None:
    if currency == "JPY":
        return 1.0
    direct = quotes.get(f"{currency}_JPY")
    if direct is not None:
        return (float(direct["bid"]) + float(direct["ask"])) / 2.0
    inverse = quotes.get(f"JPY_{currency}")
    if inverse is not None:
        mid = (float(inverse["bid"]) + float(inverse["ask"])) / 2.0
        return 1.0 / mid if mid > 0 else None
    if currency != "USD":
        to_usd = quotes.get(f"{currency}_USD")
        usd_to_jpy = _quote_to_jpy_rate("USD", quotes)
        if to_usd is not None and usd_to_jpy is not None:
            mid = (float(to_usd["bid"]) + float(to_usd["ask"])) / 2.0
            return mid * usd_to_jpy
        from_usd = quotes.get(f"USD_{currency}")
        if from_usd is not None and usd_to_jpy is not None:
            mid = (float(from_usd["bid"]) + float(from_usd["ask"])) / 2.0
            return usd_to_jpy / mid if mid > 0 else None
    return None


def _units(
    config: Mapping[str, Any],
    *,
    pair: str,
    entry_price: float,
    equity_jpy: float,
    quotes: Mapping[str, Mapping[str, Any]],
) -> float | None:
    quote_currency = pair.split("_")[1]
    quote_to_jpy = _quote_to_jpy_rate(quote_currency, quotes)
    if quote_to_jpy is None or quote_to_jpy <= 0 or equity_jpy <= 0:
        return None
    jpy_per_unit = entry_price * quote_to_jpy
    if jpy_per_unit <= 0:
        return None
    value = equity_jpy * float(config["per_pos_lev"]) / jpy_per_unit
    return value if math.isfinite(value) and value > 0 else None


def _intent(
    *,
    binding: Mapping[str, str],
    snapshot: Mapping[str, Any],
    pair: str,
    action: str,
    side: str,
    entry_price: float,
    tp_distance: float,
    sl_distance: float,
    units: float,
    hard_hold_seconds: int,
    reason: str,
) -> dict[str, Any]:
    tp_price = entry_price + tp_distance if side == "LONG" else entry_price - tp_distance
    sl_price = entry_price - sl_distance if side == "LONG" else entry_price + sl_distance
    identity = {
        "worker_id": binding["worker_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "pair": pair,
        "action": action,
        "side": side,
    }
    valid_until = int(snapshot["epoch"]) + hard_hold_seconds
    return {
        "intent_id": "I-" + _sha256(identity)[:24],
        "action": action,
        "parameters": {
            "pair": pair,
            "side": side,
            "units": units,
            "entry_price": _round_price(pair, entry_price),
            "tp_price": _round_price(pair, tp_price),
            "sl_price": _round_price(pair, sl_price),
            "stress_cost_pips": 0.0,
            "hard_max_holding_seconds": hard_hold_seconds,
            "valid_until_epoch": valid_until,
            # The reducer ignores this worker claim and reprices independently.
            "expected_net_edge_jpy": 0.0,
        },
        "reason_code": reason,
    }


def _family_intent(
    *,
    binding: Mapping[str, str],
    config: Mapping[str, Any],
    pair_state: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    pair: str,
    bar: Mapping[str, float],
    quotes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    atr = pair_state["atr"]
    closes = pair_state["closes"]
    if atr is None or float(atr) <= 0 or len(closes) < MAX_CLOSE_HISTORY:
        return None
    pip = _pip(pair)
    if float(atr) / pip < float(config["atr_floor_pips"]):
        return None
    quote = quotes[pair]
    spread = float(quote["ask"]) - float(quote["bid"])
    tp_distance = float(config["tp_atr"]) * float(atr)
    if tp_distance <= 0 or spread > tp_distance * SPREAD_TO_TARGET_CAP:
        return None
    owned_positions = [
        row
        for row in snapshot["positions"]
        if row["worker_id"] == binding["worker_id"] and row["pair"] == pair
    ]
    if len(owned_positions) >= int(config["max_concurrent_per_pair"]):
        return None
    family = binding["family_id"]
    mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
    action = "LIMIT"
    reason = family.upper()
    if family == "compression_break":
        widths = pair_state["widths"]
        if len(widths) < COMPRESSION_WARMUP_WIDTHS:
            return None
        recent = closes[-MAX_WIDTH_WINDOW:]
        width = max(recent) - min(recent)
        rank = sum(float(item) < width for item in widths) / len(widths)
        if rank > COMPRESSION_RANK_CUTOFF:
            return None
        side = "LONG" if closes[-1] > closes[0] else "SHORT"
        entry = (
            max(recent) + COMPRESSION_ENTRY_BUFFER_PIPS * pip
            if side == "LONG"
            else min(recent) - COMPRESSION_ENTRY_BUFFER_PIPS * pip
        )
        action = "STOP"
    elif family == "spike_fade":
        if bar["h"] - bar["l"] < SPIKE_RANGE_ATR_MULTIPLE * float(atr):
            return None
        up_spike = bar["c"] > bar["o"]
        side = "SHORT" if up_spike else "LONG"
        entry = bar["h"] if up_spike else bar["l"]
    elif family == "daily_break_pullback":
        previous_high = pair_state["previous_day_high"]
        previous_low = pair_state["previous_day_low"]
        if previous_high is None or previous_low is None:
            return None
        if float(pair_state["day_high"]) > float(previous_high) and mid > float(previous_high):
            side, entry = "LONG", float(previous_high)
        elif float(pair_state["day_low"]) < float(previous_low) and mid < float(previous_low):
            side, entry = "SHORT", float(previous_low)
        else:
            return None
    elif family == "range_fade_limit":
        diffs = pair_state["diffs_6h"]
        if len(diffs) < MAX_DIFF_HISTORY or len(closes) < MAX_DIFF_HISTORY + 1:
            return None
        path = sum(float(item) for item in diffs)
        efficiency = abs(closes[-1] - closes[-(MAX_DIFF_HISTORY + 1)]) / path if path > 0 else math.inf
        if efficiency > float(config["eff_max"]):
            return None
        mean = sum(float(item) for item in closes[-(MAX_DIFF_HISTORY + 1):]) / (MAX_DIFF_HISTORY + 1)
        side = "SHORT" if mid >= mean else "LONG"
        distance = float(config["fade_atr"]) * float(atr)
        entry = mid + distance if side == "SHORT" else mid - distance
    else:  # pragma: no cover - construction makes this unreachable
        raise DojoBuiltinStrategyRuntimeError("unreachable built-in family")
    # Preserve valid passive geometry after decimal rounding.
    entry = _round_price(pair, entry)
    if action == "LIMIT" and (
        (side == "LONG" and entry > float(quote["ask"]))
        or (side == "SHORT" and entry < float(quote["bid"]))
    ):
        return None
    if action == "STOP" and (
        (side == "LONG" and entry < float(quote["ask"]))
        or (side == "SHORT" and entry > float(quote["bid"]))
    ):
        return None
    units = _units(
        config,
        pair=pair,
        entry_price=entry,
        equity_jpy=float(snapshot["account"]["equity_jpy"]),
        quotes=quotes,
    )
    if units is None:
        return None
    return _intent(
        binding=binding,
        snapshot=snapshot,
        pair=pair,
        action=action,
        side=side,
        entry_price=entry,
        tp_distance=tp_distance,
        sl_distance=float(config["sl_pips"]) * pip,
        units=units,
        hard_hold_seconds=int(config["ceiling_min"]) * 60,
        reason=reason,
    )


class _BuiltinStrategyRuntime:
    def __init__(
        self,
        coordinate: Mapping[str, Any],
        bindings: Sequence[Mapping[str, str]],
        prior_state: Any | None,
    ) -> None:
        rows = _worker_rows()
        by_id = {row["worker_id"]: row for row in rows}
        self._bindings = [dict(row) for row in bindings]
        if not self._bindings:
            raise DojoBuiltinStrategyRuntimeError("active worker set cannot be empty")
        for binding in self._bindings:
            expected = by_id.get(binding["worker_id"])
            if expected is None or any(
                binding[key] != expected[key]
                for key in ("owner_id", "family_id", "config_sha256")
            ):
                raise DojoBuiltinStrategyRuntimeError(
                    "active worker is outside the sealed built-in catalog"
                )
        trade_pairs = coordinate.get("trade_pairs")
        if (
            isinstance(trade_pairs, (str, bytes))
            or not isinstance(trade_pairs, Sequence)
            or not trade_pairs
            or any(not isinstance(pair, str) for pair in trade_pairs)
        ):
            raise DojoBuiltinStrategyRuntimeError(
                "coordinate must expose the sealed tradable-pair mask"
            )
        self._trade_pairs = sorted(set(trade_pairs))
        self._configs = {
            binding["worker_id"]: by_id[binding["worker_id"]]["config"]
            for binding in self._bindings
        }
        self._state = _strict_prior_state(
            prior_state, bindings=self._bindings, trade_pairs=self._trade_pairs
        )

    def propose(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(snapshot, Mapping):
            raise DojoBuiltinStrategyRuntimeError("worker snapshot must be an object")
        snapshot_sha = snapshot.get("snapshot_sha256")
        if not isinstance(snapshot_sha, str) or len(snapshot_sha) != 64:
            raise DojoBuiltinStrategyRuntimeError("snapshot SHA is missing")
        if snapshot_sha == self._state["last_snapshot_sha256"]:
            raise DojoBuiltinStrategyRuntimeError("snapshot replay is forbidden")
        quotes = _quote_index(snapshot)
        if not set(self._trade_pairs).issubset(quotes):
            raise DojoBuiltinStrategyRuntimeError(
                "snapshot does not cover the coordinate tradable pairs"
            )
        epoch = snapshot.get("epoch")
        phase = snapshot.get("phase")
        intrabar = snapshot.get("intrabar")
        watermark = snapshot.get("quote_watermark")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or phase not in {"O", "H", "L", "C"}
            or intrabar not in {"OHLC", "OLHC"}
            or not isinstance(watermark, int)
            or isinstance(watermark, bool)
            or watermark <= self._state["last_quote_watermark"]
        ):
            raise DojoBuiltinStrategyRuntimeError(
                "snapshot clock/phase/watermark is invalid"
            )
        phase_order = ("O", "H", "L", "C") if intrabar == "OHLC" else ("O", "L", "H", "C")
        last_epoch = self._state["last_epoch"]
        last_phase = self._state["last_phase"]
        last_intrabar = self._state["last_intrabar"]
        if last_intrabar is not None and last_intrabar != intrabar:
            raise DojoBuiltinStrategyRuntimeError("intrabar path drifted across carry")
        if last_epoch is None:
            if phase != "O":
                raise DojoBuiltinStrategyRuntimeError(
                    "the built-in runtime must start at an O phase"
                )
        elif epoch == last_epoch:
            if (
                last_phase not in phase_order
                or phase_order.index(phase) != phase_order.index(last_phase) + 1
            ):
                raise DojoBuiltinStrategyRuntimeError(
                    "snapshot phase sequence is non-causal"
                )
        elif epoch > last_epoch:
            if last_phase != "C" or phase != "O":
                raise DojoBuiltinStrategyRuntimeError(
                    "new candle did not follow a completed prior candle"
                )
        else:
            raise DojoBuiltinStrategyRuntimeError("snapshot epoch moved backward")
        proposals = []
        for binding in self._bindings:
            worker_state = self._state["workers"][binding["worker_id"]]
            config = self._configs[binding["worker_id"]]
            risk_reducing = []
            new_risk = []
            owned_position_count = sum(
                row["worker_id"] == binding["worker_id"]
                for row in snapshot["positions"]
            )
            remaining_global_capacity = max(
                0, int(config["global_max_concurrent"]) - owned_position_count
            )
            for pair in self._trade_pairs:
                if pair not in config["pairs"]:
                    continue
                quote = quotes[pair]
                mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
                bar = _consume_quote_phase(
                    worker_state["pairs"][pair], epoch=epoch, phase=phase, mid=mid
                )
                if phase != "C" or bar is None:
                    continue
                for order in snapshot["pending_orders"]:
                    if (
                        order["worker_id"] == binding["worker_id"]
                        and order["pair"] == pair
                    ):
                        risk_reducing.append(
                            {
                                "intent_id": "X-" + order["order_id"][:24],
                                "action": "CANCEL_ORDER",
                                "parameters": {"order_id": order["order_id"]},
                                "reason_code": "REPRICE_CLOSED_BAR",
                            }
                        )
                candidate = _family_intent(
                    binding=binding,
                    config=config,
                    pair_state=worker_state["pairs"][pair],
                    snapshot=snapshot,
                    pair=pair,
                    bar=bar,
                    quotes=quotes,
                )
                if (
                    candidate is not None
                    and len(new_risk) < remaining_global_capacity
                ):
                    new_risk.append(candidate)
            if risk_reducing or new_risk:
                worker_state["intent_proposal_count"] += 1
            else:
                worker_state["hold_ack_count"] += 1
            proposals.append(
                {
                    **binding,
                    "snapshot_sha256": snapshot_sha,
                    "risk_reducing_intents": risk_reducing,
                    "new_risk_intents": new_risk,
                }
            )
        self._state["call_count"] += 1
        self._state["last_snapshot_sha256"] = snapshot_sha
        self._state["last_epoch"] = epoch
        self._state["last_phase"] = phase
        self._state["last_intrabar"] = intrabar
        self._state["last_quote_watermark"] = watermark
        return proposals

    def export_state(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(self._state).decode("utf-8"))


def builtin_strategy_runtime_factory(
    coordinate: Mapping[str, Any],
    bindings: Sequence[Mapping[str, str]],
    prior_state: Any | None,
) -> _BuiltinStrategyRuntime:
    """Resolve only the source-defined built-in runtime; no caller code runs."""

    return _BuiltinStrategyRuntime(coordinate, bindings, prior_state)


__all__ = [
    "ALGORITHM_REVISION",
    "DojoBuiltinStrategyRuntimeError",
    "RUNTIME_SEAL_CONTRACT",
    "RUNTIME_STATE_CONTRACT",
    "build_builtin_strategy_runtime_seal",
    "builtin_strategy_runtime_factory",
    "verify_builtin_strategy_runtime_seal",
]
