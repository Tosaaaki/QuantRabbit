"""Generation-bound declarative strategy runtime for DOJO TRAIN.

This module is the only bridge from trainer-sealed candidate proposals to the
long-horizon economic runner.  It is intentionally not a plugin surface: no
module name, callable, source path, environment variable, or executable code is
accepted from a candidate.  A runtime seal contains only catalog-normalized
JSON configs and source-owned family capability identifiers.

The source algorithms use elapsed-time horizons.  M1 and M5 therefore receive
different history lengths for the same 24h/12h/6h/20m observations.  A factory
holds a canonical private copy of one verified generation seal, so a replay
cannot swap configs or timeframes after dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from quant_rabbit.dojo_bot_catalog import bot_config_sha256, validate_bot_config
from quant_rabbit.dojo_bot_trainer import seal_candidate_proposal
from quant_rabbit.dojo_lab_provenance import canonical_strategy_owner_id
from quant_rabbit.dojo_strategy_catalog_revision_v2 import (
    FAMILY as ASIA_SWEEP_RECLAIM_BE_FAMILY,
    PROPOSAL_CONTRACT as REVISION_V2_PROPOSAL_CONTRACT,
    seal_candidate_proposal as seal_revision_v2_candidate_proposal,
    strategy_config_sha256 as revision_v2_config_sha256,
    validate_asia_sweep_reclaim_be_config,
)


RUNTIME_SEAL_CONTRACT: Final = "QR_DOJO_TUNED_STRATEGY_RUNTIME_SEAL_V2"
RUNTIME_STATE_CONTRACT: Final = "QR_DOJO_TUNED_STRATEGY_RUNTIME_STATE_V2"
SCHEMA_VERSION: Final = 2
ALGORITHM_REVISION: Final = "DECLARATIVE_LAB_FAMILY_POST_EXIT_ADAPTER_V3"
GENESIS_SNAPSHOT_SHA256: Final = "0" * 64
MAX_WORKERS: Final = 32
MAX_DEPENDENCY_BYTES: Final = 8 * 1024 * 1024

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WORKER_ID_RE: Final = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_UTC: Final = timezone.utc
_LONDON: Final = ZoneInfo("Europe/London")
_NEW_YORK: Final = ZoneInfo("America/New_York")

# Source-owned allowlist.  A catalog family is not executable merely because
# its config validates.  Every row below names the exact adapter behavior that
# is sealed per candidate.
_FAMILY_CAPABILITIES: Final = {
    "burst": "TREND_PRIOR_SWING_MARKET_V1",
    "pullback_limit": "TREND_ATR_PULLBACK_LIMIT_V1",
    "compression_break": "SQUEEZE_DIRECTIONAL_STOP_V1",
    "spike_fade": "ONE_BAR_ATR_EXTREME_LIMIT_V1",
    "prev_day_extreme_fade": "NEAREST_PREVIOUS_DAY_EXTREME_LIMIT_V1",
    "round_number_fade": "NEAREST_MAJOR_FIGURE_LIMIT_V1",
    "daily_break_pullback": "PREVIOUS_DAY_BREAK_RETEST_LIMIT_V1",
    "mean_revert_24h": "DEEP_24H_MEAN_REVERSION_MARKET_V1",
    "fade_ladder": "BOUNDED_EFFICIENCY_FADE_LAYER_V1",
    "range_fade_limit": "EFFICIENCY_RANGE_FADE_LIMIT_V1",
    "session_open_range_break": "LONDON_00_08_RANGE_BREAK_MARKET_V1",
    "weekend_gap_recovery": "NY_17_WEEKEND_GAP_RECOVERY_MARKET_V1",
    ASIA_SWEEP_RECLAIM_BE_FAMILY: "ASIA_RANGE_SWEEP_M5_CLOSE_RECLAIM_NEXT_OPEN_BE_V1",
}

_PENDING_FAMILIES: Final = frozenset(
    {
        "pullback_limit",
        "compression_break",
        "spike_fade",
        "prev_day_extreme_fade",
        "round_number_fade",
        "daily_break_pullback",
        "fade_ladder",
        "range_fade_limit",
    }
)

_DEPENDENCY_PATHS: Final = (
    "bots/lab_bot.py",
    "src/quant_rabbit/dojo_bot_catalog.py",
    "src/quant_rabbit/dojo_bot_trainer.py",
    "src/quant_rabbit/dojo_lab_provenance.py",
    "src/quant_rabbit/dojo_shared_worker_protocol.py",
    "src/quant_rabbit/dojo_strategy_catalog_revision_v2.py",
    "src/quant_rabbit/dojo_tuned_strategy_runtime.py",
)

# Time widths are source algorithm definitions, not caller parameters.
_TREND_SECONDS: Final = 24 * 60 * 60
_EFFICIENCY_SECONDS: Final = 6 * 60 * 60
_WIDTH_HISTORY_SECONDS: Final = 12 * 60 * 60
_WIDTH_WINDOW_SECONDS: Final = 20 * 60
_COMPRESSION_WARMUP_SECONDS: Final = 6 * 60 * 60
_ATR_WILDER_SECONDS: Final = 14 * 60
_PRIOR_SWING_SECONDS: Final = 3 * 60
_SESSION_RANGE_END_MINUTE: Final = 8 * 60
_SESSION_ENTRY_END_MINUTE: Final = 11 * 60
_WEEKEND_BOUNDARY_SECONDS: Final = 2 * 60
_DAILY_EDGE_TOLERANCE_SECONDS: Final = 5 * 60
_DAILY_MIN_OBSERVED_SECONDS: Final = 23 * 60 * 60
_DAILY_MAX_GAP_SECONDS: Final = 5 * 60

_COMPRESSION_RANK_CUTOFF: Final = 0.2
_COMPRESSION_ENTRY_BUFFER_PIPS: Final = 2.0
_SPIKE_RANGE_ATR_MULTIPLE: Final = 2.5
_SPREAD_TO_TARGET_CAP: Final = 0.35


class DojoTunedStrategyRuntimeError(ValueError):
    """A tuned runtime seal, candidate, timeframe, or carry is invalid."""


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
        raise DojoTunedStrategyRuntimeError("value is not strict JSON") from exc


def _copy_json(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_sha(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or (not allow_zero and value == GENESIS_SNAPSHOT_SHA256)
    ):
        raise DojoTunedStrategyRuntimeError(f"{field} must be a non-zero SHA-256")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DojoTunedStrategyRuntimeError(f"{field} must be a positive integer")
    return value


def _file_sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > MAX_DEPENDENCY_BYTES
    ):
        raise DojoTunedStrategyRuntimeError(
            f"runtime dependency is not a bounded regular file: {path}"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
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
        raise DojoTunedStrategyRuntimeError(
            f"runtime dependency changed while hashing: {path}"
        )
    return size, digest.hexdigest()


def _ceil_bars(seconds: int, cadence: int) -> int:
    return max(1, (seconds + cadence - 1) // cadence)


def _timeframe_profile(granularity: str, cadence: int) -> dict[str, Any]:
    return {
        "granularity": granularity,
        "cadence_seconds": cadence,
        "trend_horizon_seconds": _TREND_SECONDS,
        "trend_close_count": _TREND_SECONDS // cadence + 1,
        "efficiency_horizon_seconds": _EFFICIENCY_SECONDS,
        "efficiency_diff_count": _EFFICIENCY_SECONDS // cadence,
        "width_history_horizon_seconds": _WIDTH_HISTORY_SECONDS,
        "width_history_count": _WIDTH_HISTORY_SECONDS // cadence,
        "width_window_horizon_seconds": _WIDTH_WINDOW_SECONDS,
        "width_window_count": _WIDTH_WINDOW_SECONDS // cadence,
        "compression_warmup_horizon_seconds": _COMPRESSION_WARMUP_SECONDS,
        "compression_warmup_width_count": _COMPRESSION_WARMUP_SECONDS // cadence,
        "atr_wilder_horizon_seconds": _ATR_WILDER_SECONDS,
        "atr_wilder_period_bars": _ATR_WILDER_SECONDS / cadence,
        "prior_swing_horizon_seconds": _PRIOR_SWING_SECONDS,
        "prior_swing_bar_count": _ceil_bars(_PRIOR_SWING_SECONDS, cadence),
        "session_range_end_minute_london": _SESSION_RANGE_END_MINUTE,
        "session_entry_end_minute_london": _SESSION_ENTRY_END_MINUTE,
        "weekend_boundary_horizon_seconds": _WEEKEND_BOUNDARY_SECONDS,
        "weekend_boundary_bar_count": _ceil_bars(_WEEKEND_BOUNDARY_SECONDS, cadence),
    }


def _timeframe_profiles() -> list[dict[str, Any]]:
    profiles = [_timeframe_profile("M1", 60), _timeframe_profile("M5", 300)]
    if any(
        profiles[0][key] == profiles[1][key]
        for key in (
            "trend_close_count",
            "efficiency_diff_count",
            "width_history_count",
            "width_window_count",
            "compression_warmup_width_count",
            "prior_swing_bar_count",
        )
    ):
        raise DojoTunedStrategyRuntimeError(
            "M1 and M5 elapsed-time profiles collapsed to equal bar counts"
        )
    return profiles


def _seal_runtime_candidate_proposal(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Seal one legacy or explicit revision-V2 proposal without cross-admission."""

    if raw.get("contract") == REVISION_V2_PROPOSAL_CONTRACT:
        proposal = seal_revision_v2_candidate_proposal(raw)
        config = validate_asia_sweep_reclaim_be_config(proposal["config"])
        return proposal, config, revision_v2_config_sha256(config)
    proposal = seal_candidate_proposal(raw)
    config = validate_bot_config(proposal["config"])
    return proposal, config, bot_config_sha256(config)


def _proposal_rows(
    candidate_proposals: Sequence[Mapping[str, Any]],
    *,
    dependency_algorithm_sha256: str,
    timeframe_profiles_sha256: str,
) -> list[dict[str, Any]]:
    if (
        isinstance(candidate_proposals, (str, bytes))
        or not isinstance(candidate_proposals, Sequence)
        or not 1 <= len(candidate_proposals) <= MAX_WORKERS
    ):
        raise DojoTunedStrategyRuntimeError(
            f"candidate_proposals must contain 1..{MAX_WORKERS} sealed proposals"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(candidate_proposals):
        if not isinstance(raw, Mapping) or "proposal_sha256" not in raw:
            raise DojoTunedStrategyRuntimeError(
                f"candidate_proposals[{index}] was not sealed by the trainer builder"
            )
        try:
            proposal, config, config_sha = _seal_runtime_candidate_proposal(raw)
        except ValueError as exc:
            raise DojoTunedStrategyRuntimeError(
                f"candidate_proposals[{index}] is invalid"
            ) from exc
        worker_id = proposal["candidate_id"]
        family = proposal["family"]
        if _WORKER_ID_RE.fullmatch(worker_id) is None:
            raise DojoTunedStrategyRuntimeError(
                "candidate id is not a schedule-safe worker id"
            )
        capability = _FAMILY_CAPABILITIES.get(family)
        if capability is None:
            raise DojoTunedStrategyRuntimeError(
                f"family has no tuned runtime implementation: {family}"
            )
        capability_body = {
            "algorithm_revision": ALGORITHM_REVISION,
            "family_id": family,
            "family_capability": capability,
            "exit_policy": config["exit_policy"],
            "supports_fixed_tp_or_atr_tp": family
            not in {"session_open_range_break", "weekend_gap_recovery"},
            "supports_dynamic_tp_sl": family
            in {"session_open_range_break", "weekend_gap_recovery"},
            "supports_breakeven_overlay": True,
            "supports_atr_trailing_overlay": family
            != ASIA_SWEEP_RECLAIM_BE_FAMILY,
            "m5_close_only": family == ASIA_SWEEP_RECLAIM_BE_FAMILY,
            "proposal_on_next_batch_only": family == ASIA_SWEEP_RECLAIM_BE_FAMILY,
            "dependency_algorithm_sha256": dependency_algorithm_sha256,
            "timeframe_profiles_sha256": timeframe_profiles_sha256,
        }
        rows.append(
            {
                "worker_id": worker_id,
                "owner_id": canonical_strategy_owner_id(
                    config,
                    namespace=(
                        "dojo-long-tuned-v2"
                        if family == ASIA_SWEEP_RECLAIM_BE_FAMILY
                        else "dojo-long-tuned-v1"
                    ),
                ),
                "family_id": family,
                "config_sha256": config_sha,
                "config": config,
                "trainer_proposal_sha256": proposal["proposal_sha256"],
                "algorithm_revision": ALGORITHM_REVISION,
                "algorithm_capability": capability_body,
                "algorithm_capability_sha256": _sha256(capability_body),
            }
        )
    rows.sort(key=lambda row: row["worker_id"])
    worker_ids = [row["worker_id"] for row in rows]
    config_shas = [row["config_sha256"] for row in rows]
    owner_ids = [row["owner_id"] for row in rows]
    if len(set(worker_ids)) != len(rows):
        raise DojoTunedStrategyRuntimeError("duplicate candidate/worker id")
    if len(set(config_shas)) != len(rows) or len(set(owner_ids)) != len(rows):
        raise DojoTunedStrategyRuntimeError(
            "duplicate or normalized-equivalent candidate config is forbidden"
        )
    return rows


def build_tuned_strategy_runtime_seal(
    repo_root: Path,
    *,
    candidate_proposals: Sequence[Mapping[str, Any]],
    generation_ordinal: int,
    generation_binding_sha256: str,
) -> dict[str, Any]:
    """Seal one immutable generation of trainer-produced declarative workers."""

    root = Path(repo_root).resolve(strict=True)
    actual_root = Path(__file__).resolve().parents[2]
    if not root.is_dir() or root != actual_root:
        raise DojoTunedStrategyRuntimeError(
            "repo_root must be the source tree that loaded this tuned runtime"
        )
    ordinal = _positive_integer(generation_ordinal, field="generation_ordinal")
    generation_sha = _strict_sha(
        generation_binding_sha256, field="generation_binding_sha256"
    )
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
    dependencies_sha = _sha256(dependencies)
    algorithm_dependencies = [
        row
        for row in dependencies
        if row["relative_path"]
        in {"bots/lab_bot.py", "src/quant_rabbit/dojo_tuned_strategy_runtime.py"}
    ]
    algorithm_sha = _sha256(
        {
            "algorithm_revision": ALGORITHM_REVISION,
            "dependencies": algorithm_dependencies,
        }
    )
    profiles = _timeframe_profiles()
    profiles_sha = _sha256(profiles)
    workers = _proposal_rows(
        candidate_proposals,
        dependency_algorithm_sha256=algorithm_sha,
        timeframe_profiles_sha256=profiles_sha,
    )
    catalog = [
        {
            key: row[key]
            for key in ("worker_id", "owner_id", "family_id", "config_sha256")
        }
        for row in workers
    ]
    proposals = [
        _seal_runtime_candidate_proposal(raw)[0] for raw in candidate_proposals
    ]
    proposals.sort(key=lambda row: row["candidate_id"])
    supported_families = sorted(
        family
        for family in _FAMILY_CAPABILITIES
        if family != ASIA_SWEEP_RECLAIM_BE_FAMILY
        or any(row["family"] == ASIA_SWEEP_RECLAIM_BE_FAMILY for row in proposals)
    )
    body = {
        "contract": RUNTIME_SEAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "generation_ordinal": ordinal,
        "generation_binding_sha256": generation_sha,
        "algorithm_revision": ALGORITHM_REVISION,
        "algorithm_dependency_sha256": algorithm_sha,
        "runtime_mode": "SEALED_TUNED_DECLARATIVE_MULTI_STRATEGY",
        "worker_count": len(workers),
        "worker_catalog": catalog,
        "worker_catalog_sha256": _sha256(catalog),
        "workers": workers,
        "workers_sha256": _sha256(workers),
        "trainer_candidate_proposals": proposals,
        "trainer_candidate_proposals_sha256": _sha256(proposals),
        "supported_family_allowlist": supported_families,
        "supported_family_allowlist_sha256": _sha256(supported_families),
        "timeframe_profiles": profiles,
        "timeframe_profiles_sha256": profiles_sha,
        "dependencies": dependencies,
        "dependencies_sha256": dependencies_sha,
        "capabilities": {
            "arbitrary_code_allowed": False,
            "arbitrary_import_allowed": False,
            "broker_handle_available": False,
            "broker_mutation_allowed": False,
            "candidate_declared_plugin_allowed": False,
            "external_code_loading_allowed": False,
            "filesystem_available_to_worker": False,
            "live_permission": False,
            "network_available_to_worker": False,
            "order_authority": "NONE",
            "proposal_only": True,
            "explicit_hold_ack_required": True,
            "runtime_seal_mutation_allowed": False,
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


def verify_tuned_strategy_runtime_seal(
    seal: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Rebuild a tuned seal from embedded proposals and current source bytes."""

    if not isinstance(seal, Mapping):
        raise DojoTunedStrategyRuntimeError("runtime seal must be an object")
    try:
        proposals = seal["trainer_candidate_proposals"]
        ordinal = seal["generation_ordinal"]
        generation_sha = seal["generation_binding_sha256"]
    except KeyError as exc:
        raise DojoTunedStrategyRuntimeError(
            "runtime seal schema is incomplete"
        ) from exc
    rebuilt = build_tuned_strategy_runtime_seal(
        repo_root,
        candidate_proposals=proposals,
        generation_ordinal=ordinal,
        generation_binding_sha256=generation_sha,
    )
    detached = _copy_json(seal)
    if detached != rebuilt:
        raise DojoTunedStrategyRuntimeError(
            "tuned runtime seal differs from its generation/config/source denominator"
        )
    return rebuilt


def _empty_pair_state() -> dict[str, Any]:
    return {
        "atr": None,
        "closes": [],
        "diffs": [],
        "widths": [],
        "highs": [],
        "lows": [],
        "forming": None,
        "last_closed_epoch": None,
        "day": None,
        "day_high": None,
        "day_low": None,
        "day_observation_count": 0,
        "previous_day_high": None,
        "previous_day_low": None,
        "daily_true_ranges": [],
        "daily_atr": None,
        "daily_completed_count": 0,
        "daily_previous_close": None,
        "daily_accepted_dates": [],
        "daily_weekday": None,
        "daily_first_epoch": None,
        "daily_last_epoch": None,
        "daily_observed_count": 0,
        "daily_max_gap_seconds": 0,
        "daily_coverage_invalid": False,
        "session_day": None,
        "session_range_high": None,
        "session_range_low": None,
        "session_range_count": 0,
        "session_range_last_epoch": None,
        "session_range_last_minute": None,
        "session_range_contiguous": False,
        "last_bar_epoch": None,
        "last_bar_weekday": None,
        "last_bar_close": None,
        "recent_bar_epochs": [],
        "weekend_friday_close": None,
        "weekend_sunday_open": None,
        "weekend_reference_atr": None,
        "weekend_bar_count": 0,
        "weekend_last_epoch": None,
        "weekend_open_epoch": None,
        "weekend_valid": False,
        "weekend_evaluated": True,
    }


def _empty_worker_pair_state() -> dict[str, Any]:
    return {
        "session_attempted_day": None,
        "weekend_evaluated_open_epoch": None,
        "asia_sweep_reclaim_attempted_day": None,
        "asia_sweep_reclaim_pending": None,
        "position_overlays": {},
    }


def _append_bounded(rows: list[Any], value: Any, maximum: int) -> None:
    rows.append(value)
    if len(rows) > maximum:
        del rows[: len(rows) - maximum]


def _finite_or_none(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise DojoTunedStrategyRuntimeError(f"invalid {field}")
    return float(value)


def _validate_pair_state(value: Any, *, profile: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_empty_pair_state()):
        raise DojoTunedStrategyRuntimeError("prior pair state schema is not exact")
    bounds = {
        "closes": int(profile["trend_close_count"]),
        "diffs": int(profile["efficiency_diff_count"]),
        "widths": int(profile["width_history_count"]),
        "highs": int(profile["prior_swing_bar_count"]),
        "lows": int(profile["prior_swing_bar_count"]),
        "daily_true_ranges": 14,
        "daily_accepted_dates": 14,
        "recent_bar_epochs": int(profile["weekend_boundary_bar_count"]),
    }
    for key, maximum in bounds.items():
        rows = value[key]
        if not isinstance(rows, list) or len(rows) > maximum:
            raise DojoTunedStrategyRuntimeError(f"invalid prior {key}")
    for key in ("closes", "diffs", "widths", "highs", "lows", "daily_true_ranges"):
        for item in value[key]:
            _finite_or_none(item, field=key)
    if any(not isinstance(item, str) for item in value["daily_accepted_dates"]):
        raise DojoTunedStrategyRuntimeError("invalid daily accepted dates")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value["recent_bar_epochs"]
    ):
        raise DojoTunedStrategyRuntimeError("invalid recent bar epochs")
    for key in (
        "atr",
        "day_high",
        "day_low",
        "previous_day_high",
        "previous_day_low",
        "daily_atr",
        "daily_previous_close",
        "session_range_high",
        "session_range_low",
        "last_bar_close",
        "weekend_friday_close",
        "weekend_sunday_open",
        "weekend_reference_atr",
    ):
        _finite_or_none(value[key], field=key)
    for key in (
        "day_observation_count",
        "daily_completed_count",
        "daily_observed_count",
        "daily_max_gap_seconds",
        "session_range_count",
        "weekend_bar_count",
    ):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < 0
        ):
            raise DojoTunedStrategyRuntimeError(f"invalid prior {key}")
    for key in (
        "last_closed_epoch",
        "daily_first_epoch",
        "daily_last_epoch",
        "session_range_last_epoch",
        "last_bar_epoch",
        "weekend_last_epoch",
        "weekend_open_epoch",
    ):
        item = value[key]
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item < 0
        ):
            raise DojoTunedStrategyRuntimeError(f"invalid prior {key}")
    if value["forming"] is not None:
        forming = value["forming"]
        if not isinstance(forming, Mapping) or set(forming) != {
            "epoch",
            "o",
            "h",
            "l",
            "c",
        }:
            raise DojoTunedStrategyRuntimeError("invalid prior forming bar")
        if isinstance(forming["epoch"], bool) or not isinstance(forming["epoch"], int):
            raise DojoTunedStrategyRuntimeError("invalid prior forming epoch")
        for key in ("o", "h", "l", "c"):
            number = _finite_or_none(forming[key], field=f"forming.{key}")
            if number is None or number <= 0:
                raise DojoTunedStrategyRuntimeError("invalid prior forming price")
    for key in (
        "daily_coverage_invalid",
        "session_range_contiguous",
        "weekend_valid",
        "weekend_evaluated",
    ):
        if not isinstance(value[key], bool):
            raise DojoTunedStrategyRuntimeError(f"invalid prior {key}")


def _validate_worker_pair_state(value: Any, *, cadence_seconds: int) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_empty_worker_pair_state()):
        raise DojoTunedStrategyRuntimeError(
            "prior worker-local pair state schema is not exact"
        )
    if value["session_attempted_day"] is not None and not isinstance(
        value["session_attempted_day"], str
    ):
        raise DojoTunedStrategyRuntimeError("invalid session attempt day")
    asia_attempted = value["asia_sweep_reclaim_attempted_day"]
    if asia_attempted is not None and not isinstance(asia_attempted, str):
        raise DojoTunedStrategyRuntimeError("invalid Asia sweep/reclaim attempt day")
    pending = value["asia_sweep_reclaim_pending"]
    if pending is not None:
        if not isinstance(pending, Mapping) or set(pending) != {
            "side",
            "session_day",
            "signal_close_epoch",
            "entry_due_epoch",
            "range_high",
            "range_low",
            "reclaim_close",
            "swept_level",
        }:
            raise DojoTunedStrategyRuntimeError(
                "invalid Asia sweep/reclaim pending schema"
            )
        side = pending["side"]
        session_day = pending["session_day"]
        signal_epoch = pending["signal_close_epoch"]
        due_epoch = pending["entry_due_epoch"]
        if (
            side not in {"LONG", "SHORT"}
            or not isinstance(session_day, str)
            or not session_day
            or isinstance(signal_epoch, bool)
            or not isinstance(signal_epoch, int)
            or signal_epoch < 0
            or isinstance(due_epoch, bool)
            or not isinstance(due_epoch, int)
            or due_epoch != signal_epoch + cadence_seconds
            or asia_attempted != session_day
        ):
            raise DojoTunedStrategyRuntimeError(
                "invalid Asia sweep/reclaim pending identity"
            )
        range_high = _finite_or_none(pending["range_high"], field="range_high")
        range_low = _finite_or_none(pending["range_low"], field="range_low")
        reclaim_close = _finite_or_none(
            pending["reclaim_close"], field="reclaim_close"
        )
        swept_level = _finite_or_none(pending["swept_level"], field="swept_level")
        if (
            range_high is None
            or range_low is None
            or reclaim_close is None
            or swept_level is None
            or range_high <= range_low
            or min(range_low, reclaim_close, swept_level) <= 0
            or (side == "SHORT" and swept_level != range_high)
            or (side == "LONG" and swept_level != range_low)
            or not range_low < reclaim_close < range_high
        ):
            raise DojoTunedStrategyRuntimeError(
                "invalid Asia sweep/reclaim pending prices"
            )
    evaluated = value["weekend_evaluated_open_epoch"]
    if evaluated is not None and (
        isinstance(evaluated, bool) or not isinstance(evaluated, int) or evaluated < 0
    ):
        raise DojoTunedStrategyRuntimeError("invalid weekend evaluation epoch")
    overlays = value["position_overlays"]
    if not isinstance(overlays, Mapping):
        raise DojoTunedStrategyRuntimeError("invalid position overlay state")
    for position_id, overlay in overlays.items():
        if (
            not isinstance(position_id, str)
            or not isinstance(overlay, Mapping)
            or set(overlay)
            != {
                "entry_atr",
                "favorable_extreme",
                "breakeven_done",
            }
        ):
            raise DojoTunedStrategyRuntimeError("invalid position overlay row")
        entry_atr = _finite_or_none(overlay["entry_atr"], field="entry_atr")
        extreme = _finite_or_none(
            overlay["favorable_extreme"], field="favorable_extreme"
        )
        if entry_atr is None or entry_atr <= 0 or extreme is None or extreme <= 0:
            raise DojoTunedStrategyRuntimeError("invalid position overlay values")
        if not isinstance(overlay["breakeven_done"], bool):
            raise DojoTunedStrategyRuntimeError("invalid breakeven overlay flag")


def _strict_prior_state(
    value: Any,
    *,
    runtime_binding_sha256: str,
    granularity: str,
    cadence_seconds: int,
    bindings: Sequence[Mapping[str, str]],
    trade_pairs: Sequence[str],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    expected_workers = {row["worker_id"]: row for row in bindings}
    if value is None:
        return {
            "contract": RUNTIME_STATE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "algorithm_revision": ALGORITHM_REVISION,
            "runtime_binding_sha256": runtime_binding_sha256,
            "granularity": granularity,
            "cadence_seconds": cadence_seconds,
            "call_count": 0,
            "last_snapshot_sha256": GENESIS_SNAPSHOT_SHA256,
            "last_epoch": None,
            "last_phase": None,
            "last_intrabar": None,
            "last_quote_watermark": 0,
            "trade_pairs": list(trade_pairs),
            "market_pairs": {pair: _empty_pair_state() for pair in trade_pairs},
            "workers": {
                worker_id: {
                    "config_sha256": binding["config_sha256"],
                    "hold_ack_count": 0,
                    "intent_proposal_count": 0,
                    "pairs": {pair: _empty_worker_pair_state() for pair in trade_pairs},
                }
                for worker_id, binding in expected_workers.items()
            },
        }
    if not isinstance(value, Mapping):
        raise DojoTunedStrategyRuntimeError("prior worker state must be an object")
    expected_keys = {
        "contract",
        "schema_version",
        "algorithm_revision",
        "runtime_binding_sha256",
        "granularity",
        "cadence_seconds",
        "call_count",
        "last_snapshot_sha256",
        "last_epoch",
        "last_phase",
        "last_intrabar",
        "last_quote_watermark",
        "trade_pairs",
        "market_pairs",
        "workers",
    }
    if set(value) != expected_keys:
        raise DojoTunedStrategyRuntimeError("prior worker state schema is not exact")
    if (
        value["contract"] != RUNTIME_STATE_CONTRACT
        or value["schema_version"] != SCHEMA_VERSION
        or value["algorithm_revision"] != ALGORITHM_REVISION
        or value["runtime_binding_sha256"] != runtime_binding_sha256
        or value["granularity"] != granularity
        or value["cadence_seconds"] != cadence_seconds
        or value["trade_pairs"] != list(trade_pairs)
        or not isinstance(value["market_pairs"], Mapping)
        or set(value["market_pairs"]) != set(trade_pairs)
        or not isinstance(value["workers"], Mapping)
        or set(value["workers"]) != set(expected_workers)
    ):
        raise DojoTunedStrategyRuntimeError("prior worker state binding drifted")
    detached = _copy_json(value)
    if (
        isinstance(detached["call_count"], bool)
        or not isinstance(detached["call_count"], int)
        or detached["call_count"] < 0
        or not isinstance(detached["last_snapshot_sha256"], str)
        or _SHA256_RE.fullmatch(detached["last_snapshot_sha256"]) is None
        or detached["last_phase"] not in {None, "O", "H", "L", "C"}
        or detached["last_intrabar"] not in {None, "OHLC", "OLHC"}
        or isinstance(detached["last_quote_watermark"], bool)
        or not isinstance(detached["last_quote_watermark"], int)
        or detached["last_quote_watermark"] < 0
    ):
        raise DojoTunedStrategyRuntimeError("prior worker clock state is invalid")
    if detached["last_epoch"] is not None and (
        isinstance(detached["last_epoch"], bool)
        or not isinstance(detached["last_epoch"], int)
        or detached["last_epoch"] < 0
    ):
        raise DojoTunedStrategyRuntimeError("prior worker epoch is invalid")
    for pair_state in detached["market_pairs"].values():
        _validate_pair_state(pair_state, profile=profile)
    for worker_id, binding in expected_workers.items():
        worker = detached["workers"][worker_id]
        if not isinstance(worker, Mapping) or set(worker) != {
            "config_sha256",
            "hold_ack_count",
            "intent_proposal_count",
            "pairs",
        }:
            raise DojoTunedStrategyRuntimeError("prior worker schema is not exact")
        if (
            worker["config_sha256"] != binding["config_sha256"]
            or any(
                isinstance(worker[key], bool)
                or not isinstance(worker[key], int)
                or worker[key] < 0
                for key in ("hold_ack_count", "intent_proposal_count")
            )
            or not isinstance(worker["pairs"], Mapping)
            or set(worker["pairs"]) != set(trade_pairs)
        ):
            raise DojoTunedStrategyRuntimeError("prior worker identity drifted")
        for pair_state in worker["pairs"].values():
            _validate_worker_pair_state(
                pair_state, cadence_seconds=cadence_seconds
            )
    return detached


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _round_price(pair: str, price: float) -> float:
    return round(price, 3 if pair.endswith("JPY") else 5)


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
        usd_to_jpy = _quote_to_jpy_rate("USD", quotes)
        direct_usd = quotes.get(f"{currency}_USD")
        inverse_usd = quotes.get(f"USD_{currency}")
        if usd_to_jpy is not None and direct_usd is not None:
            return (
                (float(direct_usd["bid"]) + float(direct_usd["ask"])) / 2.0 * usd_to_jpy
            )
        if usd_to_jpy is not None and inverse_usd is not None:
            mid = (float(inverse_usd["bid"]) + float(inverse_usd["ask"])) / 2.0
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
    quote_to_jpy = _quote_to_jpy_rate(pair.split("_")[1], quotes)
    if quote_to_jpy is None or quote_to_jpy <= 0 or equity_jpy <= 0:
        return None
    jpy_per_unit = entry_price * quote_to_jpy
    value = equity_jpy * float(config["per_pos_lev"]) / jpy_per_unit
    return value if math.isfinite(value) and value > 0 else None


def _daily_coverage_complete(state: Mapping[str, Any], *, cadence: int) -> bool:
    first = state["daily_first_epoch"]
    last = state["daily_last_epoch"]
    if first is None or last is None or state["daily_weekday"] is None:
        return False
    day_start = first - (first % 86400)
    observed_seconds = int(state["daily_observed_count"]) * cadence
    return (
        state["daily_coverage_invalid"] is False
        and int(state["daily_weekday"]) < 4
        and first - day_start <= _DAILY_EDGE_TOLERANCE_SECONDS
        and last >= day_start + 86400 - cadence - _DAILY_EDGE_TOLERANCE_SECONDS
        and observed_seconds >= _DAILY_MIN_OBSERVED_SECONDS
        and int(state["daily_max_gap_seconds"]) <= max(_DAILY_MAX_GAP_SECONDS, cadence)
    )


def _roll_daily_state(
    state: dict[str, Any],
    *,
    epoch: int,
    bar: Mapping[str, float],
    prior_close: float | None,
    cadence: int,
) -> None:
    day = datetime.fromtimestamp(epoch, _UTC).date().isoformat()
    if state["day"] != day:
        if (
            state["day"] is not None
            and state["day_high"] is not None
            and state["day_low"] is not None
            and prior_close is not None
        ):
            state["previous_day_high"] = state["day_high"]
            state["previous_day_low"] = state["day_low"]
            if _daily_coverage_complete(state, cadence=cadence):
                reference = state["daily_previous_close"]
                true_range = max(
                    float(state["day_high"]) - float(state["day_low"]),
                    abs(float(state["day_high"]) - float(reference))
                    if reference is not None
                    else 0.0,
                    abs(float(state["day_low"]) - float(reference))
                    if reference is not None
                    else 0.0,
                )
                _append_bounded(state["daily_true_ranges"], true_range, 14)
                _append_bounded(state["daily_accepted_dates"], state["day"], 14)
                state["daily_completed_count"] += 1
                if state["daily_atr"] is None and len(state["daily_true_ranges"]) == 14:
                    state["daily_atr"] = sum(state["daily_true_ranges"]) / 14.0
                elif state["daily_atr"] is not None:
                    state["daily_atr"] += (
                        true_range - float(state["daily_atr"])
                    ) / 14.0
                state["daily_previous_close"] = prior_close
        state["day"] = day
        state["day_high"] = bar["h"]
        state["day_low"] = bar["l"]
        state["day_observation_count"] = 1
        state["daily_weekday"] = datetime.fromtimestamp(epoch, _UTC).weekday()
        state["daily_first_epoch"] = epoch
        state["daily_last_epoch"] = epoch
        state["daily_observed_count"] = 1
        state["daily_max_gap_seconds"] = 0
        state["daily_coverage_invalid"] = False
        return
    state["day_high"] = max(float(state["day_high"]), bar["h"])
    state["day_low"] = min(float(state["day_low"]), bar["l"])
    state["day_observation_count"] += 1
    last = state["daily_last_epoch"]
    if last is None or epoch <= last or (epoch - last) % cadence != 0:
        state["daily_coverage_invalid"] = True
    else:
        state["daily_observed_count"] += 1
        state["daily_max_gap_seconds"] = max(
            int(state["daily_max_gap_seconds"]), epoch - last
        )
        state["daily_last_epoch"] = epoch


def _update_session_range(
    state: dict[str, Any],
    *,
    epoch: int,
    bar: Mapping[str, float],
    cadence: int,
) -> tuple[str, int]:
    local = datetime.fromtimestamp(epoch, _UTC).astimezone(_LONDON)
    local_day = local.date().isoformat()
    local_minute = local.hour * 60 + local.minute
    if state["session_day"] != local_day:
        state.update(
            {
                "session_day": local_day,
                "session_range_high": None,
                "session_range_low": None,
                "session_range_count": 0,
                "session_range_last_epoch": None,
                "session_range_last_minute": None,
                "session_range_contiguous": False,
            }
        )
    if local_minute < _SESSION_RANGE_END_MINUTE:
        if state["session_range_count"] == 0:
            state["session_range_contiguous"] = local_minute == 0
        elif state["session_range_last_epoch"] is None or (
            epoch != state["session_range_last_epoch"] + cadence
        ):
            state["session_range_contiguous"] = False
        state["session_range_high"] = (
            bar["h"]
            if state["session_range_high"] is None
            else max(float(state["session_range_high"]), bar["h"])
        )
        state["session_range_low"] = (
            bar["l"]
            if state["session_range_low"] is None
            else min(float(state["session_range_low"]), bar["l"])
        )
        state["session_range_count"] += 1
        state["session_range_last_epoch"] = epoch
        state["session_range_last_minute"] = local_minute
    return local_day, local_minute


def _session_range_complete(state: Mapping[str, Any], *, cadence: int) -> bool:
    cadence_minutes = cadence // 60
    expected = _SESSION_RANGE_END_MINUTE // cadence_minutes
    return (
        state["session_range_contiguous"] is True
        and state["session_range_count"] == expected
        and state["session_range_last_minute"]
        == _SESSION_RANGE_END_MINUTE - cadence_minutes
        and state["session_range_high"] is not None
        and state["session_range_low"] is not None
        and float(state["session_range_high"]) > float(state["session_range_low"])
    )


def _update_weekend_gap(
    state: dict[str, Any],
    *,
    epoch: int,
    bar: Mapping[str, float],
    cadence: int,
    boundary_bar_count: int,
    prior_epoch: int | None,
    prior_weekday: int | None,
    prior_close: float | None,
) -> None:
    local = datetime.fromtimestamp(epoch, _UTC).astimezone(_NEW_YORK)
    exact_open = (
        local.weekday() == 6
        and local.hour == 17
        and local.minute == 0
        and local.second == 0
    )
    if exact_open:
        state.update(
            {
                "weekend_friday_close": None,
                "weekend_sunday_open": None,
                "weekend_reference_atr": None,
                "weekend_bar_count": 0,
                "weekend_last_epoch": None,
                "weekend_open_epoch": epoch,
                "weekend_valid": False,
                "weekend_evaluated": True,
            }
        )
        friday = local.date() - timedelta(days=2)
        boundary = datetime(friday.year, friday.month, friday.day, 17, tzinfo=_NEW_YORK)
        expected_last = int(boundary.timestamp()) - cadence
        expected_epochs = [
            expected_last - cadence * offset
            for offset in reversed(range(boundary_bar_count))
        ]
        expected_dates: list[str] = []
        cursor = friday - timedelta(days=1)
        while len(expected_dates) < 14:
            if cursor.weekday() < 4:
                expected_dates.append(cursor.isoformat())
            cursor -= timedelta(days=1)
        expected_dates.reverse()
        if (
            state["recent_bar_epochs"] == expected_epochs
            and state["daily_accepted_dates"] == expected_dates
            and prior_weekday == 4
            and prior_epoch == expected_last
            and prior_close is not None
            and state["daily_atr"] is not None
            and state["daily_completed_count"] >= 14
        ):
            state["weekend_friday_close"] = prior_close
            state["weekend_sunday_open"] = bar["o"]
            state["weekend_reference_atr"] = state["daily_atr"]
            state["weekend_bar_count"] = 1
            state["weekend_last_epoch"] = epoch
            state["weekend_valid"] = True
            state["weekend_evaluated"] = False
        return
    if state["weekend_evaluated"] or state["weekend_bar_count"] <= 0:
        return
    if (
        state["weekend_last_epoch"] is None
        or epoch != state["weekend_last_epoch"] + cadence
    ):
        state["weekend_valid"] = False
        state["weekend_evaluated"] = True
        return
    state["weekend_bar_count"] += 1
    state["weekend_last_epoch"] = epoch


def _consume_quote_phase(
    state: dict[str, Any],
    *,
    epoch: int,
    phase: str,
    mid: float,
    cadence: int,
    profile: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]] | None:
    forming = state["forming"]
    if forming is None or forming["epoch"] != epoch:
        if forming is not None:
            raise DojoTunedStrategyRuntimeError(
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
        raise DojoTunedStrategyRuntimeError("closed-bar clock did not advance")
    bar = {key: float(forming[key]) for key in ("o", "h", "l", "c")}
    context = {
        "prior_high": max(state["highs"]) if state["highs"] else None,
        "prior_low": min(state["lows"]) if state["lows"] else None,
        "prior_epoch": state["last_bar_epoch"],
        "prior_weekday": state["last_bar_weekday"],
        "prior_close": state["last_bar_close"],
    }
    prior_close = state["closes"][-1] if state["closes"] else None
    if prior_close is not None:
        true_range = max(
            bar["h"] - bar["l"],
            abs(bar["h"] - prior_close),
            abs(bar["l"] - prior_close),
        )
        period = float(profile["atr_wilder_period_bars"])
        state["atr"] = (
            true_range
            if state["atr"] is None
            else float(state["atr"]) + (true_range - float(state["atr"])) / period
        )
        _append_bounded(
            state["diffs"],
            abs(bar["c"] - prior_close),
            int(profile["efficiency_diff_count"]),
        )
    _append_bounded(state["closes"], bar["c"], int(profile["trend_close_count"]))
    width_count = int(profile["width_window_count"])
    if len(state["closes"]) >= width_count:
        recent = state["closes"][-width_count:]
        _append_bounded(
            state["widths"],
            max(recent) - min(recent),
            int(profile["width_history_count"]),
        )
    _roll_daily_state(
        state, epoch=epoch, bar=bar, prior_close=prior_close, cadence=cadence
    )
    context["session_day"], context["session_minute"] = _update_session_range(
        state, epoch=epoch, bar=bar, cadence=cadence
    )
    _update_weekend_gap(
        state,
        epoch=epoch,
        bar=bar,
        cadence=cadence,
        boundary_bar_count=int(profile["weekend_boundary_bar_count"]),
        prior_epoch=context["prior_epoch"],
        prior_weekday=context["prior_weekday"],
        prior_close=context["prior_close"],
    )
    _append_bounded(state["highs"], bar["h"], int(profile["prior_swing_bar_count"]))
    _append_bounded(state["lows"], bar["l"], int(profile["prior_swing_bar_count"]))
    _append_bounded(
        state["recent_bar_epochs"],
        epoch,
        int(profile["weekend_boundary_bar_count"]),
    )
    state["last_bar_epoch"] = epoch
    state["last_bar_weekday"] = datetime.fromtimestamp(epoch, _UTC).weekday()
    state["last_bar_close"] = bar["c"]
    state["last_closed_epoch"] = epoch
    state["forming"] = None
    return bar, context


def _new_risk_intent(
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
    tp = entry_price + tp_distance if side == "LONG" else entry_price - tp_distance
    sl = entry_price - sl_distance if side == "LONG" else entry_price + sl_distance
    identity = {
        "worker_id": binding["worker_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "pair": pair,
        "action": action,
        "side": side,
        "entry_price": _round_price(pair, entry_price),
    }
    return {
        "intent_id": "I-" + _sha256(identity)[:24],
        "action": action,
        "parameters": {
            "pair": pair,
            "side": side,
            "units": units,
            "entry_price": _round_price(pair, entry_price),
            "tp_price": _round_price(pair, tp),
            "sl_price": _round_price(pair, sl),
            "stress_cost_pips": 0.0,
            "hard_max_holding_seconds": hard_hold_seconds,
            "valid_until_epoch": int(snapshot["epoch"]) + hard_hold_seconds,
            "expected_net_edge_jpy": 0.0,
        },
        "reason_code": reason,
    }


def _reduction_intent(
    *,
    binding: Mapping[str, str],
    snapshot: Mapping[str, Any],
    action: str,
    target_id: str,
    parameters: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    identity = {
        "worker_id": binding["worker_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "action": action,
        "target_id": target_id,
    }
    return {
        "intent_id": "R-" + _sha256(identity)[:24],
        "action": action,
        "parameters": dict(parameters),
        "reason_code": reason,
    }


def _owned_positions(
    snapshot: Mapping[str, Any], *, worker_id: str, pair: str | None = None
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in snapshot["positions"]
        if row["worker_id"] == worker_id and (pair is None or row["pair"] == pair)
    ]


def _exit_overlay_intents(
    *,
    binding: Mapping[str, str],
    config: Mapping[str, Any],
    pair_state: dict[str, Any],
    worker_pair_state: dict[str, Any],
    snapshot: Mapping[str, Any],
    pair: str,
    bar: Mapping[str, float],
    quote: Mapping[str, Any],
) -> list[dict[str, Any]]:
    positions = _owned_positions(snapshot, worker_id=binding["worker_id"], pair=pair)
    live_ids = {row["position_id"] for row in positions}
    overlays = worker_pair_state["position_overlays"]
    for stale in set(overlays) - live_ids:
        del overlays[stale]
    atr = pair_state["atr"]
    if atr is None or float(atr) <= 0:
        return []
    newly_observed: set[str] = set()
    for position in positions:
        if position["position_id"] not in overlays:
            overlays[position["position_id"]] = {
                "entry_atr": float(atr),
                "favorable_extreme": float(position["entry_price"]),
                "breakeven_done": False,
            }
            newly_observed.add(position["position_id"])
        overlay = overlays[position["position_id"]]
        if position["position_id"] in newly_observed:
            continue
        if position["side"] == "LONG":
            overlay["favorable_extreme"] = max(
                float(overlay["favorable_extreme"]), bar["h"]
            )
        else:
            overlay["favorable_extreme"] = min(
                float(overlay["favorable_extreme"]), bar["l"]
            )
    policy = config["exit_policy"]
    if policy == "FIXED":
        return []
    intents = []
    pip = _pip(pair)
    for position in positions:
        overlay = overlays[position["position_id"]]
        side = position["side"]
        entry = float(position["entry_price"])
        executable = float(quote["bid"] if side == "LONG" else quote["ask"])
        profit = executable - entry if side == "LONG" else entry - executable
        candidate: float | None = None
        if policy == "BREAKEVEN":
            if not overlay["breakeven_done"] and profit >= float(
                config["be_trigger_atr"]
            ) * float(overlay["entry_atr"]):
                candidate = (
                    entry + float(config["be_offset_pips"]) * pip
                    if side == "LONG"
                    else entry - float(config["be_offset_pips"]) * pip
                )
                overlay["breakeven_done"] = True
        elif policy == "ATR_TRAILING":
            extreme = float(overlay["favorable_extreme"])
            favorable = extreme - entry if side == "LONG" else entry - extreme
            if favorable >= float(config["trail_trigger_atr"]) * float(
                overlay["entry_atr"]
            ):
                distance = float(config["trail_distance_atr"]) * float(atr)
                candidate = extreme - distance if side == "LONG" else extreme + distance
        if candidate is None:
            continue
        candidate = _round_price(pair, candidate)
        existing = position["sl_price"]
        tighter = existing is None or (
            candidate > float(existing)
            if side == "LONG"
            else candidate < float(existing)
        )
        if not tighter:
            continue
        if (side == "LONG" and candidate >= executable) or (
            side == "SHORT" and candidate <= executable
        ):
            intents.append(
                _reduction_intent(
                    binding=binding,
                    snapshot=snapshot,
                    action="CLOSE_POSITION",
                    target_id=position["position_id"],
                    parameters={"position_id": position["position_id"], "units": None},
                    reason="EXIT_OVERLAY_ALREADY_CROSSED",
                )
            )
        else:
            intents.append(
                _reduction_intent(
                    binding=binding,
                    snapshot=snapshot,
                    action="TIGHTEN_STOP",
                    target_id=position["position_id"],
                    parameters={
                        "position_id": position["position_id"],
                        "sl_price": candidate,
                    },
                    reason=f"{policy}_OVERLAY",
                )
            )
    return intents


def _target_distance(config: Mapping[str, Any], *, atr: float, pip: float) -> float:
    if config["tp_atr"] is not None:
        return float(config["tp_atr"]) * atr
    if config["tp_pips"] is not None:
        return float(config["tp_pips"]) * pip
    raise DojoTunedStrategyRuntimeError("family has no fixed target distance")


def _stage_asia_sweep_reclaim(
    *,
    pair_state: Mapping[str, Any],
    worker_pair_state: dict[str, Any],
    bar: Mapping[str, float],
    context: Mapping[str, Any],
    epoch: int,
    cadence: int,
) -> None:
    """Record a completed-M5 signal for the next candle only."""

    minute = int(context["session_minute"])
    session_day = str(context["session_day"])
    if (
        not (_SESSION_RANGE_END_MINUTE <= minute < _SESSION_ENTRY_END_MINUTE)
        or worker_pair_state["asia_sweep_reclaim_attempted_day"] == session_day
        or worker_pair_state["asia_sweep_reclaim_pending"] is not None
        or not _session_range_complete(pair_state, cadence=cadence)
    ):
        return
    range_high = float(pair_state["session_range_high"])
    range_low = float(pair_state["session_range_low"])
    reclaim_close = float(bar["c"])
    if not range_low < reclaim_close < range_high:
        return
    swept_high = float(bar["h"]) > range_high
    swept_low = float(bar["l"]) < range_low
    if not swept_high and not swept_low:
        return
    worker_pair_state["asia_sweep_reclaim_attempted_day"] = session_day
    if swept_high == swept_low:
        # A candle that swept both sides has no unambiguous directional thesis.
        return
    side = "SHORT" if swept_high else "LONG"
    worker_pair_state["asia_sweep_reclaim_pending"] = {
        "side": side,
        "session_day": session_day,
        "signal_close_epoch": epoch,
        "entry_due_epoch": epoch + cadence,
        "range_high": range_high,
        "range_low": range_low,
        "reclaim_close": reclaim_close,
        "swept_level": range_high if side == "SHORT" else range_low,
    }


def _asia_sweep_reclaim_pending_intent(
    *,
    binding: Mapping[str, str],
    config: Mapping[str, Any],
    pair_state: Mapping[str, Any],
    worker_pair_state: dict[str, Any],
    snapshot: Mapping[str, Any],
    pair: str,
    quotes: Mapping[str, Mapping[str, Any]],
    allow_new_risk: bool,
) -> dict[str, Any] | None:
    """Consume a staged signal at the exact next M5 O evaluation point."""

    pending = worker_pair_state["asia_sweep_reclaim_pending"]
    if pending is None:
        return None
    worker_pair_state["asia_sweep_reclaim_pending"] = None
    epoch = int(snapshot["epoch"])
    phase = snapshot["phase"]
    session_day = datetime.fromtimestamp(epoch, _UTC).astimezone(_LONDON).date()
    if (
        phase != "O"
        or epoch != int(pending["entry_due_epoch"])
        or session_day.isoformat() != pending["session_day"]
        or not allow_new_risk
    ):
        return None
    atr = pair_state["atr"]
    if atr is None or float(atr) <= 0:
        return None
    atr = float(atr)
    pip = _pip(pair)
    if atr / pip < float(config["atr_floor_pips"]):
        return None
    quote = quotes[pair]
    side = str(pending["side"])
    entry = float(quote["ask"] if side == "LONG" else quote["bid"])
    target_distance = _target_distance(config, atr=atr, pip=pip)
    sl_pips = config["sl_pips"]
    if sl_pips is None:
        return None
    sl_distance = float(sl_pips) * pip
    spread = float(quote["ask"]) - float(quote["bid"])
    if (
        target_distance <= 0
        or sl_distance <= 0
        or spread > target_distance * _SPREAD_TO_TARGET_CAP
    ):
        return None
    entry = _round_price(pair, entry)
    units = _units(
        config,
        pair=pair,
        entry_price=entry,
        equity_jpy=float(snapshot["account"]["equity_jpy"]),
        quotes=quotes,
    )
    if units is None:
        return None
    return _new_risk_intent(
        binding=binding,
        snapshot=snapshot,
        pair=pair,
        action="MARKET",
        side=side,
        entry_price=entry,
        tp_distance=target_distance,
        sl_distance=sl_distance,
        units=units,
        hard_hold_seconds=int(config["ceiling_min"]) * 60,
        reason="ASIA_SWEEP_RECLAIM_BE_NEXT_M5_OPEN",
    )


def _family_intent(
    *,
    binding: Mapping[str, str],
    config: Mapping[str, Any],
    pair_state: dict[str, Any],
    worker_pair_state: dict[str, Any],
    snapshot: Mapping[str, Any],
    pair: str,
    bar: Mapping[str, float],
    context: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    atr = pair_state["atr"]
    if atr is None or float(atr) <= 0:
        return None
    atr = float(atr)
    pip = _pip(pair)
    if atr / pip < float(config["atr_floor_pips"]):
        return None
    quote = quotes[pair]
    mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
    spread = float(quote["ask"]) - float(quote["bid"])
    family = binding["family_id"]
    target_distance: float | None = None
    if family not in {"session_open_range_break", "weekend_gap_recovery"}:
        if len(pair_state["closes"]) < int(profile["trend_close_count"]):
            return None
        target_distance = _target_distance(config, atr=atr, pip=pip)
        if target_distance <= 0 or spread > target_distance * _SPREAD_TO_TARGET_CAP:
            return None
    trend = None
    if pair_state["closes"]:
        trend = (
            "LONG" if pair_state["closes"][-1] > pair_state["closes"][0] else "SHORT"
        )
    action = "LIMIT"
    side: str
    entry: float
    sl_distance = float(config["sl_pips"] or 0.0) * pip
    reason = family.upper()

    if family == "burst":
        if trend is None:
            return None
        prior_high, prior_low = context["prior_high"], context["prior_low"]
        if prior_high is None or prior_low is None:
            return None
        if not (
            (trend == "LONG" and bar["c"] > float(prior_high))
            or (trend == "SHORT" and bar["c"] < float(prior_low))
        ):
            return None
        side = trend
        action = "MARKET"
        entry = float(quote["ask"] if side == "LONG" else quote["bid"])
    elif family == "pullback_limit":
        if trend is None:
            return None
        side = trend
        distance = float(config["pull_atr"]) * atr
        entry = mid - distance if side == "LONG" else mid + distance
    elif family == "compression_break":
        if trend is None:
            return None
        if len(pair_state["widths"]) < int(profile["compression_warmup_width_count"]):
            return None
        recent = pair_state["closes"][-int(profile["width_window_count"]) :]
        width = max(recent) - min(recent)
        rank = sum(float(item) < width for item in pair_state["widths"]) / len(
            pair_state["widths"]
        )
        if rank > _COMPRESSION_RANK_CUTOFF:
            return None
        side = trend
        entry = (
            max(recent) + _COMPRESSION_ENTRY_BUFFER_PIPS * pip
            if side == "LONG"
            else min(recent) - _COMPRESSION_ENTRY_BUFFER_PIPS * pip
        )
        action = "STOP"
    elif family == "spike_fade":
        if bar["h"] - bar["l"] < _SPIKE_RANGE_ATR_MULTIPLE * atr:
            return None
        up = bar["c"] > bar["o"]
        side = "SHORT" if up else "LONG"
        entry = bar["h"] if up else bar["l"]
    elif family == "prev_day_extreme_fade":
        levels = [
            ("SHORT", pair_state["previous_day_high"]),
            ("LONG", pair_state["previous_day_low"]),
        ]
        eligible = [
            (abs(float(level) - mid), candidate_side, float(level))
            for candidate_side, level in levels
            if level is not None and abs(float(level) - mid) <= 40.0 * pip
        ]
        if not eligible:
            return None
        _, side, entry = min(eligible)
    elif family == "round_number_fade":
        step = 0.50 if pair.endswith("JPY") else 0.0050
        above = (math.floor(mid / step) + 1) * step
        below = math.floor(mid / step) * step
        eligible = [
            (abs(level - mid), candidate_side, level)
            for candidate_side, level in (("SHORT", above), ("LONG", below))
            if 3.0 * pip <= abs(level - mid) <= 25.0 * pip
        ]
        if not eligible:
            return None
        _, side, entry = min(eligible)
    elif family == "daily_break_pullback":
        previous_high = pair_state["previous_day_high"]
        previous_low = pair_state["previous_day_low"]
        if previous_high is None or previous_low is None:
            return None
        if float(pair_state["day_high"]) > float(previous_high) and mid > float(
            previous_high
        ):
            side, entry = "LONG", float(previous_high)
        elif float(pair_state["day_low"]) < float(previous_low) and mid < float(
            previous_low
        ):
            side, entry = "SHORT", float(previous_low)
        else:
            return None
    elif family == "mean_revert_24h":
        mean = sum(pair_state["closes"]) / len(pair_state["closes"])
        threshold = float(config["fade_atr"]) * 8.0 * atr
        if mid <= mean - threshold:
            side = "LONG"
        elif mid >= mean + threshold:
            side = "SHORT"
        else:
            return None
        action = "MARKET"
        entry = float(quote["ask"] if side == "LONG" else quote["bid"])
    elif family in {"fade_ladder", "range_fade_limit"}:
        needed = int(profile["efficiency_diff_count"])
        if len(pair_state["diffs"]) < needed or len(pair_state["closes"]) < needed + 1:
            return None
        path = sum(float(item) for item in pair_state["diffs"])
        if path <= 0:
            return None
        efficiency = (
            abs(pair_state["closes"][-1] - pair_state["closes"][-(needed + 1)]) / path
        )
        if efficiency > float(config["eff_max"]):
            return None
        mean = sum(pair_state["closes"][-(needed + 1) :]) / (needed + 1)
        side = "SHORT" if mid >= mean else "LONG"
        open_count = len(
            _owned_positions(snapshot, worker_id=binding["worker_id"], pair=pair)
        )
        multiplier = 2.2 if family == "fade_ladder" and open_count == 1 else 1.0
        distance = float(config["fade_atr"]) * atr * multiplier
        entry = mid + distance if side == "SHORT" else mid - distance
    elif family == "session_open_range_break":
        minute = int(context["session_minute"])
        if not (_SESSION_RANGE_END_MINUTE <= minute < _SESSION_ENTRY_END_MINUTE):
            return None
        if worker_pair_state["session_attempted_day"] == context[
            "session_day"
        ] or not _session_range_complete(
            pair_state, cadence=int(profile["cadence_seconds"])
        ):
            return None
        high = float(pair_state["session_range_high"])
        low = float(pair_state["session_range_low"])
        width = high - low
        buffer = float(config["session_buffer_atr"]) * atr
        if mid > high + buffer:
            side = "LONG"
        elif mid < low - buffer:
            side = "SHORT"
        else:
            return None
        worker_pair_state["session_attempted_day"] = context["session_day"]
        target_distance = float(config["session_tp_range"]) * width
        sl_distance = float(config["session_sl_range"]) * width
        if (
            target_distance <= 0
            or sl_distance <= 0
            or spread > target_distance * _SPREAD_TO_TARGET_CAP
        ):
            return None
        action = "MARKET"
        entry = float(quote["ask"] if side == "LONG" else quote["bid"])
    elif family == "weekend_gap_recovery":
        wait_seconds = int(config["weekend_wait_bars"]) * 60
        wait_bars = _ceil_bars(wait_seconds, int(profile["cadence_seconds"]))
        wait_bars = max(wait_bars, int(profile["weekend_boundary_bar_count"]))
        weekend_open_epoch = pair_state["weekend_open_epoch"]
        if (
            weekend_open_epoch is None
            or worker_pair_state["weekend_evaluated_open_epoch"] == weekend_open_epoch
            or pair_state["weekend_evaluated"]
            or pair_state["weekend_bar_count"] < wait_bars
        ):
            return None
        worker_pair_state["weekend_evaluated_open_epoch"] = weekend_open_epoch
        if not pair_state["weekend_valid"]:
            return None
        friday = pair_state["weekend_friday_close"]
        sunday = pair_state["weekend_sunday_open"]
        reference_atr = pair_state["weekend_reference_atr"]
        if friday is None or sunday is None or reference_atr is None:
            return None
        gap = float(sunday) - float(friday)
        if abs(gap) < float(config["weekend_gap_atr"]) * float(reference_atr):
            return None
        if gap < 0 and float(quote["bid"]) < float(friday):
            side = "LONG"
            entry = float(quote["ask"])
            target_distance = float(friday) - entry
            stop_price = float(sunday) - float(config["weekend_sl_gap"]) * abs(gap)
            sl_distance = entry - stop_price
        elif gap > 0 and float(quote["ask"]) > float(friday):
            side = "SHORT"
            entry = float(quote["bid"])
            target_distance = entry - float(friday)
            stop_price = float(sunday) + float(config["weekend_sl_gap"]) * abs(gap)
            sl_distance = stop_price - entry
        else:
            return None
        if (
            target_distance <= 0
            or sl_distance <= 0
            or spread > target_distance * float(config["weekend_spread_fraction"])
        ):
            return None
        action = "MARKET"
    elif family == ASIA_SWEEP_RECLAIM_BE_FAMILY:
        # This family is staged at C and consumed at the following O in propose().
        return None
    else:  # pragma: no cover - seal construction makes this unreachable
        raise DojoTunedStrategyRuntimeError("unreachable tuned family")

    if target_distance is None:
        raise DojoTunedStrategyRuntimeError("strategy target distance was not resolved")
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
    return _new_risk_intent(
        binding=binding,
        snapshot=snapshot,
        pair=pair,
        action=action,
        side=side,
        entry_price=entry,
        tp_distance=target_distance,
        sl_distance=sl_distance,
        units=units,
        hard_hold_seconds=int(config["ceiling_min"]) * 60,
        reason=reason,
    )


class _TunedStrategyRuntime:
    def __init__(
        self,
        *,
        seal: Mapping[str, Any],
        coordinate: Mapping[str, Any],
        bindings: Sequence[Mapping[str, str]],
        prior_state: Any | None,
    ) -> None:
        granularity = coordinate.get("granularity")
        cadence = coordinate.get("bar_seconds")
        profiles = {row["granularity"]: row for row in seal["timeframe_profiles"]}
        profile = profiles.get(granularity)
        if (
            profile is None
            or isinstance(cadence, bool)
            or not isinstance(cadence, int)
            or cadence != profile["cadence_seconds"]
        ):
            raise DojoTunedStrategyRuntimeError(
                "coordinate timeframe/cadence is outside the sealed runtime"
            )
        rows = {row["worker_id"]: row for row in seal["workers"]}
        self._bindings = [_copy_json(row) for row in bindings]
        if not self._bindings:
            raise DojoTunedStrategyRuntimeError("active worker set cannot be empty")
        for binding in self._bindings:
            expected = rows.get(binding.get("worker_id"))
            if expected is None or any(
                binding.get(key) != expected[key]
                for key in ("owner_id", "family_id", "config_sha256")
            ):
                raise DojoTunedStrategyRuntimeError(
                    "active worker is outside the sealed tuned catalog"
                )
        if any(
            binding["family_id"] == ASIA_SWEEP_RECLAIM_BE_FAMILY
            for binding in self._bindings
        ) and granularity != "M5":
            raise DojoTunedStrategyRuntimeError(
                "asia_sweep_reclaim_be requires sealed M5 coordinates"
            )
        trade_pairs = coordinate.get("trade_pairs")
        if (
            isinstance(trade_pairs, (str, bytes))
            or not isinstance(trade_pairs, Sequence)
            or not trade_pairs
            or any(not isinstance(pair, str) for pair in trade_pairs)
        ):
            raise DojoTunedStrategyRuntimeError(
                "coordinate must expose the sealed tradable-pair mask"
            )
        self._trade_pairs = sorted(set(trade_pairs))
        self._configs = {
            binding["worker_id"]: _copy_json(rows[binding["worker_id"]]["config"])
            for binding in self._bindings
        }
        self._profile = _copy_json(profile)
        self._granularity = str(granularity)
        self._cadence = int(cadence)
        self._runtime_binding_sha256 = str(seal["runtime_binding_sha256"])
        self._state = _strict_prior_state(
            prior_state,
            runtime_binding_sha256=self._runtime_binding_sha256,
            granularity=self._granularity,
            cadence_seconds=self._cadence,
            bindings=self._bindings,
            trade_pairs=self._trade_pairs,
            profile=self._profile,
        )

    def propose(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(snapshot, Mapping):
            raise DojoTunedStrategyRuntimeError("worker snapshot must be an object")
        snapshot_sha = snapshot.get("snapshot_sha256")
        if (
            not isinstance(snapshot_sha, str)
            or _SHA256_RE.fullmatch(snapshot_sha) is None
        ):
            raise DojoTunedStrategyRuntimeError("snapshot SHA is missing")
        if snapshot_sha == self._state["last_snapshot_sha256"]:
            raise DojoTunedStrategyRuntimeError("snapshot replay is forbidden")
        quotes = _quote_index(snapshot)
        if not set(self._trade_pairs).issubset(quotes):
            raise DojoTunedStrategyRuntimeError(
                "snapshot does not cover the coordinate tradable pairs"
            )
        epoch = snapshot.get("epoch")
        phase = snapshot.get("phase")
        intrabar = snapshot.get("intrabar")
        watermark = snapshot.get("quote_watermark")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or phase not in {"O", "H", "L", "C"}
            or intrabar not in {"OHLC", "OLHC"}
            or isinstance(watermark, bool)
            or not isinstance(watermark, int)
            or watermark <= self._state["last_quote_watermark"]
        ):
            raise DojoTunedStrategyRuntimeError(
                "snapshot clock/phase/watermark is invalid"
            )
        phase_order = (
            ("O", "H", "L", "C") if intrabar == "OHLC" else ("O", "L", "H", "C")
        )
        last_epoch = self._state["last_epoch"]
        last_phase = self._state["last_phase"]
        last_intrabar = self._state["last_intrabar"]
        if last_intrabar is not None and last_intrabar != intrabar:
            raise DojoTunedStrategyRuntimeError("intrabar path drifted across carry")
        if last_epoch is None:
            if phase != "O":
                raise DojoTunedStrategyRuntimeError(
                    "the tuned runtime must start at an O phase"
                )
        elif epoch == last_epoch:
            if (
                last_phase not in phase_order
                or phase_order.index(phase) != phase_order.index(last_phase) + 1
            ):
                raise DojoTunedStrategyRuntimeError(
                    "snapshot phase sequence is non-causal"
                )
        elif epoch > last_epoch:
            if (
                (epoch - last_epoch) % self._cadence != 0
                or last_phase != "C"
                or phase != "O"
            ):
                raise DojoTunedStrategyRuntimeError(
                    "new candle did not follow the sealed cadence/completed candle"
                )
        else:
            raise DojoTunedStrategyRuntimeError("snapshot epoch moved backward")

        closed_pairs: dict[str, tuple[dict[str, float], dict[str, Any]]] = {}
        for pair in self._trade_pairs:
            quote = quotes[pair]
            mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
            consumed = _consume_quote_phase(
                self._state["market_pairs"][pair],
                epoch=epoch,
                phase=phase,
                mid=mid,
                cadence=self._cadence,
                profile=self._profile,
            )
            if consumed is not None:
                closed_pairs[pair] = consumed

        proposals = []
        for binding in self._bindings:
            worker_state = self._state["workers"][binding["worker_id"]]
            config = self._configs[binding["worker_id"]]
            risk_reducing: list[dict[str, Any]] = []
            new_risk: list[dict[str, Any]] = []
            owned_positions = _owned_positions(snapshot, worker_id=binding["worker_id"])
            remaining_global = max(
                0, int(config["global_max_concurrent"]) - len(owned_positions)
            )
            for pair in self._trade_pairs:
                if pair not in config["pairs"]:
                    continue
                quote = quotes[pair]
                consumed = closed_pairs.get(pair)
                market_pair_state = self._state["market_pairs"][pair]
                worker_pair_state = worker_state["pairs"][pair]
                pair_positions = _owned_positions(
                    snapshot, worker_id=binding["worker_id"], pair=pair
                )
                if binding["family_id"] == ASIA_SWEEP_RECLAIM_BE_FAMILY:
                    pending_candidate = _asia_sweep_reclaim_pending_intent(
                        binding=binding,
                        config=config,
                        pair_state=market_pair_state,
                        worker_pair_state=worker_pair_state,
                        snapshot=snapshot,
                        pair=pair,
                        quotes=quotes,
                        allow_new_risk=remaining_global > 0
                        and len(pair_positions)
                        < int(config["max_concurrent_per_pair"]),
                    )
                    if pending_candidate is not None:
                        new_risk.append(pending_candidate)
                        remaining_global -= 1
                if phase != "C" or consumed is None:
                    continue
                bar, context = consumed
                risk_reducing.extend(
                    _exit_overlay_intents(
                        binding=binding,
                        config=config,
                        pair_state=market_pair_state,
                        worker_pair_state=worker_pair_state,
                        snapshot=snapshot,
                        pair=pair,
                        bar=bar,
                        quote=quote,
                    )
                )
                if binding["family_id"] in _PENDING_FAMILIES:
                    for order in snapshot["pending_orders"]:
                        if (
                            order["worker_id"] == binding["worker_id"]
                            and order["pair"] == pair
                        ):
                            risk_reducing.append(
                                _reduction_intent(
                                    binding=binding,
                                    snapshot=snapshot,
                                    action="CANCEL_ORDER",
                                    target_id=order["order_id"],
                                    parameters={"order_id": order["order_id"]},
                                    reason="REPRICE_CLOSED_BAR",
                                )
                            )
                if binding["family_id"] == ASIA_SWEEP_RECLAIM_BE_FAMILY:
                    _stage_asia_sweep_reclaim(
                        pair_state=market_pair_state,
                        worker_pair_state=worker_pair_state,
                        bar=bar,
                        context=context,
                        epoch=epoch,
                        cadence=self._cadence,
                    )
                    continue
                if remaining_global <= 0 or len(pair_positions) >= int(
                    config["max_concurrent_per_pair"]
                ):
                    continue
                candidate = _family_intent(
                    binding=binding,
                    config=config,
                    pair_state=market_pair_state,
                    worker_pair_state=worker_pair_state,
                    snapshot=snapshot,
                    pair=pair,
                    bar=bar,
                    context=context,
                    quotes=quotes,
                    profile=self._profile,
                )
                if candidate is not None:
                    new_risk.append(candidate)
                    remaining_global -= 1
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
        return _copy_json(self._state)


class SealedTunedStrategyRuntimeFactory:
    """Callable immutable holder for one verified tuned generation seal."""

    __slots__ = ("__seal", "__locked")

    def __init__(self, seal: Mapping[str, Any]) -> None:
        object.__setattr__(
            self, "_SealedTunedStrategyRuntimeFactory__seal", _copy_json(seal)
        )
        object.__setattr__(self, "_SealedTunedStrategyRuntimeFactory__locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_SealedTunedStrategyRuntimeFactory__locked", False):
            raise AttributeError("a tuned runtime factory seal is immutable")
        object.__setattr__(self, name, value)

    @property
    def runtime_binding_sha256(self) -> str:
        return str(self.__seal["runtime_binding_sha256"])

    def matches_verified_seal(self, seal: Mapping[str, Any]) -> bool:
        return self.__seal == _copy_json(seal)

    def __call__(
        self,
        coordinate: Mapping[str, Any],
        bindings: Sequence[Mapping[str, str]],
        prior_state: Any | None,
    ) -> _TunedStrategyRuntime:
        return _TunedStrategyRuntime(
            seal=self.__seal,
            coordinate=coordinate,
            bindings=bindings,
            prior_state=prior_state,
        )


def build_tuned_strategy_runtime_factory(
    seal: Mapping[str, Any], *, repo_root: Path
) -> SealedTunedStrategyRuntimeFactory:
    """Return a capability-closed factory bound to one verified generation."""

    verified = verify_tuned_strategy_runtime_seal(seal, repo_root=repo_root)
    return SealedTunedStrategyRuntimeFactory(verified)


__all__ = [
    "ALGORITHM_REVISION",
    "DojoTunedStrategyRuntimeError",
    "RUNTIME_SEAL_CONTRACT",
    "RUNTIME_STATE_CONTRACT",
    "SealedTunedStrategyRuntimeFactory",
    "build_tuned_strategy_runtime_factory",
    "build_tuned_strategy_runtime_seal",
    "verify_tuned_strategy_runtime_seal",
]
