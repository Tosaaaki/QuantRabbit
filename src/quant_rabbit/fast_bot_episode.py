"""Point-in-time market episodes for deterministic fast-bot learning.

The ledger records what the bot could know at each closed candle.  It does
not place orders, select risk, or let prose change a rule.  V1 deliberately
covers one narrow parent episode: an M5 attempt to leave a range whose rails
were frozen from the preceding twenty complete M5 candles.

``attempt_direction`` is the direction in which price tried to leave the
range.  It is intentionally different from ``trade_side``:

* UP + ACCEPTED -> LONG
* UP + REJECTED -> SHORT
* DOWN + ACCEPTED -> SHORT
* DOWN + REJECTED -> LONG

The same M5 candle may prove SETUP -> ATTEMPT -> ACCEPTED/REJECTED.  That
compound path is one sealed JSONL event so a process crash cannot leave a
half-written story.  Confirmation always requires a strictly later complete
M1 candle.  Signal emission is not ENTRY; only a future exact execution-truth
binding may add ENTRY/RESOLVED lifecycle events.
"""

from __future__ import annotations

import base64
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.strategy.failed_break_evidence import (
    build_m5_failed_break_evidence,
    verify_m5_failed_break_evidence,
)


REGIME_CONTRACT = "QR_HIERARCHICAL_BOT_REGIME_V1"
EPISODE_EVENT_CONTRACT = "QR_FAST_BOT_EPISODE_EVENT_V1"
EPISODE_STATE_CONTRACT = "QR_FAST_BOT_EPISODE_STATE_V1"
EPISODE_PENDING_CONTRACT = "QR_FAST_BOT_EPISODE_PENDING_V1"
EPISODE_ARCHIVE_OWNER_CONTRACT = "QR_FAST_BOT_EPISODE_ARCHIVE_OWNER_V1"
EPISODE_KIND = "M5_RANGE_BREAK_ATTEMPT"
EPISODE_RULE_VERSION = "M5_RANGE_BREAK_ATTEMPT_V1"
HORIZON_LANE = "M1_EXECUTION_15M_HOLD"
SELECTION_STATUS = "UNSELECTED_COUNTERFACTUAL_PATHS_PRESERVED"
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D": 24 * 60 * 60,
}
OBSERVATION_TIMEFRAMES = {"M1", "M5"}
RANGE_PHASES = {"PRE_RANGE", "RANGE"}
ACTIVE_STATES = {"ATTEMPT", "ACCEPTED", "REJECTED"}
TERMINAL_STATES = {"CONFIRMED", "INVALIDATED", "EXPIRED", "RESOLVED"}
STATES = {"SETUP", *ACTIVE_STATES, *TERMINAL_STATES, "ENTRY"}
ATTEMPT_DIRECTIONS = {"UP", "DOWN", "AMBIGUOUS"}
BRANCH_OUTCOMES = {"PENDING", "ACCEPTED", "REJECTED", "AMBIGUOUS"}
CONFIRMATION_TTL_SECONDS = 15 * 60
LATE_DETECTION_GRACE_SECONDS = 90
MAX_GENESIS_AGE_SECONDS = 10 * 60
BUFFER_RATIO = 0.05
MAX_PRICE_ABS = 1_000_000_000.0
MAX_EVENT_BYTES = 32_768
MAX_EVENTS_PER_UTC_DAY = 24_192
MAX_LEDGER_BYTES = 16 * 1024 * 1024
MAX_LEDGER_EVENTS = 8_192
# Source evidence is deliberately tighter than the hot ledger: one regime may
# be at most 2 MiB compressed / 8 MiB canonical, while the owner-exclusive
# aggregate is bounded to 64 MiB and at most one object per ledger event.
MAX_SOURCE_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_CONTRACT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_ARCHIVE_FILES = MAX_LEDGER_EVENTS
MAX_SOURCE_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024
# Only the untrusted suffix is expanded.  256 MiB is a hard aggregate safety
# ceiling even if many distinct maximum-size regimes arrive in one audit.
MAX_SOURCE_VERIFY_DECOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_STATE_EPISODE_SUMMARIES = 512
# A cycle can add at most 28 rows.  Base64 of 28 x 32 KiB plus the sealed WAL
# envelope remains below this fixed 2 MiB cap.
MAX_PENDING_BYTES = 2 * 1024 * 1024
SOURCE_VERIFIER_VERSION = 1

_EVENT_FIELDS = {
    "contract",
    "schema_version",
    "event_id",
    "episode_id",
    "event_seq",
    "previous_event_sha256",
    "ledger_seq",
    "previous_ledger_event_sha256",
    "generated_at_utc",
    "state",
    "state_entered_at_utc",
    "transition_path",
    "transition_reason",
    "pair",
    "anchor",
    "route",
    "observation",
    "source_m5_evidence",
    "late_detected",
    "diagnostic_only",
    "shadow_only",
    "order_authority",
    "live_permission",
    "broker_mutation_allowed",
    "automatic_rule_change_allowed",
    "promotion_allowed",
    "event_sha256",
}
_ANCHOR_FIELDS = {
    "episode_kind",
    "episode_rule_version",
    "horizon_lane",
    "confirmation_ttl_seconds",
    "setup_candle_utc",
    "attempt_candle_utc",
    "attempt_direction",
    "rail",
    "source_evidence_sha256",
}
_RAIL_FIELDS = {"upper", "lower", "width", "buffer", "buffer_ratio"}
_ROUTE_FIELDS = {
    "branch_outcome",
    "trade_side",
    "candidate_methods",
    "route_family",
    "branch_candle_utc",
    "branch_close",
    "selection_status",
}
_OBSERVATION_FIELDS = {
    "timeframe",
    "candle",
    "candle_close_utc",
    "candle_sha256",
    "regime_contract_sha256",
    "regime_archive_gzip_sha256",
    "vote_reference_side",
    "source_timeframe_votes",
    "source_timeframe_votes_sha256",
    "source_timeframe_clocks",
    "source_timeframe_clocks_sha256",
}
_CANDLE_FIELDS = {"t", "o", "h", "l", "c", "complete"}
_VOTE_FIELDS = {
    "observed_direction",
    "direction_score",
    "phase",
    "readiness",
    "trigger",
    "structure",
    "location",
    "value_zone",
    "extension",
    "evidence_complete",
}
_TRANSITION_REASONS = {
    "M5_RANGE_RAIL_PIERCED_CLOSE_WITHIN_FROZEN_BUFFER",
    "M5_RANGE_BREAK_ACCEPTED_OUTSIDE_FROZEN_RAIL",
    "M5_RANGE_BREAK_REJECTED_BACK_INSIDE_FROZEN_RAIL",
    "M5_BOTH_FROZEN_RAILS_PIERCED_AMBIGUOUS",
    "M5_OPPOSITE_FROZEN_RAIL_PIERCED",
    "M5_BREAK_ATTEMPT_EXPIRED_WITHIN_BUFFER",
    "M5_BREAK_ATTEMPT_STILL_WITHIN_FROZEN_BUFFER",
    "BRANCH_BINDING_INVALID",
    "M1_OPPOSITE_BRANCH_INVALIDATED_EPISODE",
    "STRICTLY_LATER_M1_HELD_BRANCH_AND_FOLLOWED_THROUGH",
    "M1_CONFIRMATION_TTL_EXPIRED",
    "M1_CONFIRMATION_PENDING",
    "M1_SEQUENCE_GAP_UNOBSERVABLE",
    "M5_SEQUENCE_GAP_UNOBSERVABLE",
}


def run_fast_bot_episode_shadow(
    *,
    regime_contract: Mapping[str, Any],
    fast_pair_charts: Mapping[str, Any],
    slow_pair_charts: Mapping[str, Any],
    output_path: Path,
    ledger_path: Path,
    source_archive_dir: Path | None = None,
    now_utc: datetime | None = None,
    processed_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Advance and persist the shadow episode ledger under one file lock.

    A sealed pending batch is durably published before ledger bytes are
    appended.  Startup may therefore complete only the exact byte suffix that
    this writer precommitted; an arbitrary otherwise-valid extension is never
    mistaken for crash recovery.
    """

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    processed_at = _aware_utc(processed_at_utc or now)
    if processed_at < now:
        raise ValueError("episode processing clock precedes its source cycle")
    processing_delay_seconds = (processed_at - now).total_seconds()
    operationally_late = processing_delay_seconds > LATE_DETECTION_GRACE_SECONDS
    archive_dir = source_archive_dir or ledger_path.parent / "fast_bot_episode_sources"
    pending_path = _pending_checkpoint_path(output_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_descriptor = os.open(
        ledger_path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    ledger_stat = os.fstat(ledger_descriptor)
    if not stat.S_ISREG(ledger_stat.st_mode):
        os.close(ledger_descriptor)
        raise ValueError("episode ledger must be a regular file")
    with os.fdopen(ledger_descriptor, "a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "LOCK_BUSY",
                "appended_events": 0,
                "blockers": ["EPISODE_LEDGER_LOCK_BUSY"],
                "state_persisted": False,
                "diagnostic_only": True,
                "shadow_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "broker_mutation_allowed": False,
                "automatic_rule_change_allowed": False,
                "promotion_allowed": False,
            }

        archive_owner_error = _ensure_archive_owner(archive_dir, ledger_path)
        cleanup_error = (
            _cleanup_episode_temps(
                output_path=output_path,
                pending_path=pending_path,
                archive_dir=archive_dir,
            )
            if archive_owner_error is None
            else None
        )
        prior_state, state_read_error = _read_state_checkpoint(output_path)
        pending, pending_read_error = _read_pending_checkpoint(pending_path)
        ledger_size = os.fstat(handle.fileno()).st_size
        ledger_raw = b""
        events: list[dict[str, Any]] = []
        prefix_sizes: list[int] = []
        blockers: list[str] = []
        appended = 0
        head_verified = False
        trusted_source_prefix = 0
        pending_to_clear = False
        recovered_pending = False
        pending_recovery_suffix = b""
        pending_recovery_mode: str | None = None
        status = "VERIFYING"

        if archive_owner_error is not None:
            status = "LEDGER_INVALID"
            blockers.append(archive_owner_error)
        elif cleanup_error is not None:
            status = "LEDGER_INVALID"
            blockers.append(cleanup_error)
        elif state_read_error is not None:
            status = "LEDGER_INVALID"
            blockers.append(state_read_error)
        elif pending_read_error is not None:
            status = "LEDGER_INVALID"
            blockers.append(pending_read_error)
        elif ledger_size > MAX_LEDGER_BYTES:
            status = "LEDGER_INVALID"
            blockers.append("EPISODE_LEDGER_BYTE_CAP_EXCEEDED")
        else:
            handle.seek(0)
            ledger_raw = handle.read(MAX_LEDGER_BYTES + 1)
            if pending is not None:
                (
                    ledger_raw,
                    pending_recovery_mode,
                    pending_recovery_suffix,
                    recovery_error,
                ) = _prepare_pending_recovery(
                    ledger_raw=ledger_raw,
                    state=prior_state,
                    pending=pending,
                )
                if recovery_error is not None:
                    status = "LEDGER_INVALID"
                    blockers.append(recovery_error)
            if len(ledger_raw) > MAX_LEDGER_BYTES:
                status = "LEDGER_INVALID"
                blockers.append("EPISODE_LEDGER_BYTE_CAP_EXCEEDED")
            elif status != "LEDGER_INVALID":
                events, prefix_sizes, read_error = _read_ledger_bytes(ledger_raw)
                if read_error is not None:
                    status = "LEDGER_INVALID"
                    blockers.append(read_error)
                elif pending is not None:
                    checkpoint_valid, checkpoint_error, trusted_source_prefix = (
                        _verify_pending_checkpoint(
                            prior_state,
                            pending=pending,
                            events=events,
                            ledger_raw=ledger_raw,
                            prefix_sizes=prefix_sizes,
                        )
                    )
                    if not checkpoint_valid:
                        status = "LEDGER_INVALID"
                        blockers.append(
                            checkpoint_error
                            or "EPISODE_PENDING_CHECKPOINT_INVALID"
                        )
                else:
                    checkpoint_valid, checkpoint_error, trusted_source_prefix = (
                        _verify_head_checkpoint(
                            prior_state,
                            events=events,
                            ledger_raw=ledger_raw,
                            prefix_sizes=prefix_sizes,
                        )
                    )
                    if not checkpoint_valid:
                        status = "LEDGER_INVALID"
                        blockers.append(
                            checkpoint_error
                            or "EPISODE_LEDGER_HEAD_CHECKPOINT_INVALID"
                        )

            if status != "LEDGER_INVALID":
                ledger_valid, ledger_error = _verify_episode_ledger(
                    events,
                    as_of_utc=processed_at,
                    source_archive_dir=archive_dir,
                    source_verify_start_index=trusted_source_prefix,
                )
                if not ledger_valid:
                    status = "LEDGER_INVALID"
                    blockers.append(
                        str(ledger_error or "EPISODE_LEDGER_INVALID")
                    )
                else:
                    archive_cleanup_error = _cleanup_unreferenced_archives(
                        archive_dir,
                        referenced_digests=_referenced_regime_digests(events),
                    )
                    if archive_cleanup_error is not None:
                        status = "LEDGER_INVALID"
                        blockers.append(archive_cleanup_error)
                    else:
                        if pending_recovery_suffix:
                            handle.seek(0, os.SEEK_END)
                            handle.write(pending_recovery_suffix)
                            handle.flush()
                            os.fsync(handle.fileno())
                            if os.fstat(handle.fileno()).st_size != len(ledger_raw):
                                raise OSError(
                                    "episode pending recovery size mismatch"
                                )
                        if pending_recovery_mode is not None:
                            recovered_pending = True
                            pending_to_clear = True
                            appended = int(
                                pending.get("batch_event_count") or 0
                            )
                            blockers.append(
                                "EPISODE_PENDING_BATCH_RECONCILED"
                            )
                        head_verified = True
                        status = (
                            "RECOVERED_PENDING_BATCH"
                            if recovered_pending
                            else "VERIFIED"
                        )

        if status == "LEDGER_INVALID" or recovered_pending:
            pass
        elif (
            not _sealed_contract_valid(regime_contract, REGIME_CONTRACT)
            or isinstance(regime_contract.get("schema_version"), bool)
            or regime_contract.get("schema_version") != 1
        ):
            status = "INVALID_REGIME_CONTRACT"
            blockers.append("EPISODE_REGIME_CONTRACT_INVALID")
        elif (
            _parse_utc(regime_contract.get("generated_at_utc")) is None
            or _parse_utc(regime_contract.get("generated_at_utc")) > now
        ):
            status = "INVALID_REGIME_CONTRACT"
            blockers.append("EPISODE_REGIME_CONTRACT_CLOCK_INVALID")
        else:
            try:
                new_events, derivation_blockers = _derive_events(
                    regime_contract=regime_contract,
                    fast_pair_charts=fast_pair_charts,
                    slow_pair_charts=slow_pair_charts,
                    existing_events=events,
                    now=now,
                    force_late_detected=operationally_late,
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                new_events = []
                derivation_blockers = [
                    f"EPISODE_INPUT_INVALID:{type(error).__name__.upper()}"
                ]
                status = "INPUT_INVALID"
            else:
                status = "UPDATED" if new_events else "NO_NEW_EVENT"
            blockers.extend(derivation_blockers)
            generated_today = _generated_on_utc_date(events, now.date())
            available = max(0, MAX_EVENTS_PER_UTC_DAY - generated_today)
            if len(new_events) > available:
                new_events = []
                blockers.append("EPISODE_DAILY_EVENT_CAP_REACHED")
                status = "DAILY_CAP_REACHED"
            if len(events) + len(new_events) > MAX_LEDGER_EVENTS:
                new_events = []
                blockers.append("EPISODE_LEDGER_EVENT_CAP_REACHED")
                status = "LEDGER_CAP_REACHED"

            chained_events = _bind_global_ledger_chain(events, new_events)
            encoded_events: list[tuple[dict[str, Any], bytes]] = []
            projected_size = len(ledger_raw)
            for event in chained_events:
                raw = _ledger_line(event)
                if projected_size + len(raw) > MAX_LEDGER_BYTES:
                    blockers.append("EPISODE_LEDGER_BYTE_CAP_REACHED")
                    encoded_events = []
                    break
                encoded_events.append((event, raw))
                projected_size += len(raw)
            if len(encoded_events) != len(chained_events):
                chained_events = []
                encoded_events = []
                status = "LEDGER_CAP_REACHED"

            archive_path: Path | None = None
            archive_created = False
            if chained_events:
                try:
                    archive_path, archive_created = _archive_regime_contract(
                        regime_contract,
                        archive_dir,
                    )
                except (OSError, ValueError) as error:
                    message = str(error).lower()
                    if "aggregate cap" in message:
                        status = "SOURCE_ARCHIVE_CAP_REACHED"
                        blockers.append(
                            "EPISODE_SOURCE_ARCHIVE_AGGREGATE_CAP_REACHED"
                        )
                    else:
                        status = "SOURCE_ARCHIVE_BLOCKED"
                        blockers.append(
                            "EPISODE_SOURCE_ARCHIVE_WRITE_FAILED:"
                            f"{type(error).__name__.upper()}"
                        )
                    chained_events = []
                    encoded_events = []

            candidate_events = [*events, *chained_events]
            candidate_valid = True
            candidate_error: str | None = None
            if chained_events:
                candidate_valid, candidate_error = _verify_episode_ledger(
                    candidate_events,
                    as_of_utc=now,
                    source_archive_dir=archive_dir,
                    source_verify_start_index=len(events),
                )
            if not candidate_valid:
                status = "DERIVATION_INVALID"
                blockers.append(
                    str(candidate_error or "EPISODE_DERIVED_LEDGER_INVALID")
                )
                chained_events = []
                encoded_events = []
                if archive_created and archive_path is not None:
                    cleanup_error = _delete_archive_if_unreferenced(
                        archive_path,
                        referenced_digests=_referenced_regime_digests(events),
                    )
                    if cleanup_error is not None:
                        blockers.append(cleanup_error)
            else:
                if encoded_events:
                    if prior_state is None:
                        zero_state = _build_state(
                            events=events,
                            now=now,
                            processed_at=processed_at,
                            status="VERIFIED",
                            appended=0,
                            blockers=[],
                            ledger_path=ledger_path,
                            source_archive_dir=archive_dir,
                            ledger_event_count=len(events),
                            ledger_size_bytes=len(ledger_raw),
                            ledger_bytes_sha256=hashlib.sha256(ledger_raw).hexdigest(),
                            ledger_tail_sha256=(
                                str(events[-1].get("event_sha256"))
                                if events
                                else None
                            ),
                            ledger_head_verified=True,
                        )
                        _write_json_atomic(
                            output_path,
                            zero_state,
                            max_bytes=MAX_STATE_BYTES,
                        )
                        prior_state = zero_state
                    batch_raw = b"".join(raw for _, raw in encoded_events)
                    pending_contract = _build_pending_checkpoint(
                        base_events=events,
                        base_ledger_raw=ledger_raw,
                        batch_events=chained_events,
                        batch_raw=batch_raw,
                        regime_contract=regime_contract,
                        now=now,
                    )
                    _write_pending_checkpoint(pending_path, pending_contract)
                    pending_to_clear = True
                    handle.seek(0, os.SEEK_END)
                    handle.write(batch_raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                    ledger_raw += batch_raw
                appended = len(encoded_events)
                events.extend(chained_events)
                head_verified = True
                if status not in {
                    "DAILY_CAP_REACHED",
                    "INPUT_INVALID",
                    "LEDGER_CAP_REACHED",
                    "SOURCE_ARCHIVE_CAP_REACHED",
                    "SOURCE_ARCHIVE_BLOCKED",
                    "DERIVATION_INVALID",
                }:
                    status = "UPDATED" if appended else "NO_NEW_EVENT"

        if head_verified:
            checkpoint_count = len(events)
            checkpoint_size = len(ledger_raw)
            checkpoint_ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
            checkpoint_tail = (
                str(events[-1].get("event_sha256")) if events else None
            )
        else:
            (
                checkpoint_count,
                checkpoint_size,
                checkpoint_ledger_sha,
                checkpoint_tail,
            ) = _checkpoint_head(prior_state)
        state = _build_state(
            events=events if head_verified else [],
            now=now,
            processed_at=processed_at,
            status=status,
            appended=appended,
            blockers=blockers,
            ledger_path=ledger_path,
            source_archive_dir=archive_dir,
            ledger_event_count=checkpoint_count,
            ledger_size_bytes=checkpoint_size,
            ledger_bytes_sha256=checkpoint_ledger_sha,
            ledger_tail_sha256=checkpoint_tail,
            ledger_head_verified=head_verified,
        )
        if head_verified:
            _write_json_atomic(
                output_path,
                state,
                max_bytes=MAX_STATE_BYTES,
            )
            if pending_to_clear:
                _unlink_durable(pending_path)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return state


def verify_episode_event(value: object) -> tuple[bool, str | None]:
    """Shallowly verify one sealed event, including its frozen genesis proof."""

    if not isinstance(value, Mapping) or set(value) != _EVENT_FIELDS:
        return False, "EPISODE_EVENT_SCHEMA_INVALID"
    schema_version = value.get("schema_version")
    if (
        value.get("contract") != EPISODE_EVENT_CONTRACT
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        return False, "EPISODE_EVENT_SCHEMA_INVALID"
    stored_sha = value.get("event_sha256")
    if not _sha_text(stored_sha) or stored_sha != _canonical_sha(_without(value, "event_sha256")):
        return False, "EPISODE_EVENT_SHA_MISMATCH"
    if (
        value.get("diagnostic_only") is not True
        or value.get("shadow_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("automatic_rule_change_allowed") is not False
        or value.get("promotion_allowed") is not False
    ):
        return False, "EPISODE_AUTHORITY_BOUNDARY_INVALID"

    pair = value.get("pair")
    seq = value.get("event_seq")
    ledger_seq = value.get("ledger_seq")
    state = value.get("state")
    path = value.get("transition_path")
    if (
        not isinstance(pair, str)
        or pair not in DEFAULT_TRADER_PAIRS
        or isinstance(seq, bool)
        or not isinstance(seq, int)
        or seq < 1
        or isinstance(ledger_seq, bool)
        or not isinstance(ledger_seq, int)
        or ledger_seq < 1
        or (
            value.get("previous_ledger_event_sha256") is not None
            and not _sha_text(value.get("previous_ledger_event_sha256"))
        )
        or state not in STATES
        or not isinstance(path, list)
        or not path
        or any(item not in STATES for item in path)
        or path[-1] != state
        or value.get("transition_reason") not in _TRANSITION_REASONS
        or not isinstance(value.get("late_detected"), bool)
    ):
        return False, "EPISODE_EVENT_SEMANTICS_INVALID"
    try:
        if len(_canonical_json_bytes(value)) > MAX_EVENT_BYTES:
            return False, "EPISODE_EVENT_TOO_LARGE"
    except (TypeError, ValueError, OverflowError):
        return False, "EPISODE_EVENT_SCHEMA_INVALID"

    anchor = value.get("anchor")
    route = value.get("route")
    observation = value.get("observation")
    if not isinstance(anchor, Mapping) or set(anchor) != _ANCHOR_FIELDS:
        return False, "EPISODE_ANCHOR_INVALID"
    if not isinstance(route, Mapping) or set(route) != _ROUTE_FIELDS:
        return False, "EPISODE_ROUTE_INVALID"
    if not isinstance(observation, Mapping) or set(observation) != _OBSERVATION_FIELDS:
        return False, "EPISODE_OBSERVATION_INVALID"
    if (
        anchor.get("episode_kind") != EPISODE_KIND
        or anchor.get("episode_rule_version") != EPISODE_RULE_VERSION
        or anchor.get("horizon_lane") != HORIZON_LANE
        or anchor.get("confirmation_ttl_seconds") != CONFIRMATION_TTL_SECONDS
    ):
        return False, "EPISODE_ANCHOR_INVALID"
    if anchor.get("attempt_direction") not in ATTEMPT_DIRECTIONS:
        return False, "EPISODE_ANCHOR_INVALID"
    rail = anchor.get("rail")
    if not isinstance(rail, Mapping) or set(rail) != _RAIL_FIELDS or not _rail_valid(rail):
        return False, "EPISODE_RAIL_INVALID"
    for field in ("setup_candle_utc", "attempt_candle_utc"):
        if _parse_utc(anchor.get(field)) is None:
            return False, "EPISODE_ANCHOR_INVALID"
    if not _sha_text(anchor.get("source_evidence_sha256")):
        return False, "EPISODE_ANCHOR_INVALID"

    observation_valid, observation_error = _verify_observation(observation)
    if not observation_valid:
        return False, observation_error
    generated = _parse_utc(value.get("generated_at_utc"))
    entered = _parse_utc(value.get("state_entered_at_utc"))
    observed_close = _parse_utc(observation.get("candle_close_utc"))
    if generated is None or entered is None or observed_close is None or generated < observed_close:
        return False, "EPISODE_EVENT_CLOCK_INVALID"
    clocks = observation.get("source_timeframe_clocks")
    parsed_clocks = [
        parsed
        for item in clocks.values() if isinstance(clocks, Mapping)
        for parsed in (_parse_utc(item),)
        if parsed is not None
    ]
    if any(clock > generated for clock in parsed_clocks) or entered > generated:
        return False, "EPISODE_EVENT_CLOCK_INVALID"
    must_be_late = (
        (generated - observed_close).total_seconds() > LATE_DETECTION_GRACE_SECONDS
        or any(clock > observed_close for clock in parsed_clocks)
    )
    if must_be_late and value.get("late_detected") is not True:
        return False, "EPISODE_LOOKAHEAD_NOT_QUARANTINED"
    if seq == 1 and entered != observed_close:
        return False, "EPISODE_GENESIS_STATE_CLOCK_INVALID"

    route_valid, route_error = _verify_route(route, state=state, attempt_direction=str(anchor["attempt_direction"]))
    if not route_valid:
        return False, route_error
    evidence = value.get("source_m5_evidence")
    if seq == 1:
        if not isinstance(evidence, Mapping):
            return False, "EPISODE_SOURCE_EVIDENCE_INVALID"
        evidence_valid, _ = verify_m5_failed_break_evidence(evidence)
        if not evidence_valid or evidence.get("status") != "VALID":
            return False, "EPISODE_SOURCE_EVIDENCE_INVALID"
        if not _evidence_matches_anchor_and_route(
            evidence=evidence,
            anchor=anchor,
            route=route,
            observation=observation,
            state=str(state),
            transition_reason=str(value.get("transition_reason")),
        ):
            return False, "EPISODE_SOURCE_EVIDENCE_INVALID"
    elif evidence is not None:
        return False, "EPISODE_SOURCE_EVIDENCE_MUST_BE_GENESIS_ONLY"

    episode_identity = {"pair": pair, "anchor": dict(anchor)}
    if value.get("episode_id") != _canonical_sha(episode_identity)[:24]:
        return False, "EPISODE_IDENTITY_INVALID"
    event_identity = {
        "episode_id": value.get("episode_id"),
        "event_seq": seq,
        "state": state,
        "observation_candle_utc": observation.get("candle_close_utc"),
        "previous_event_sha256": value.get("previous_event_sha256"),
    }
    if value.get("event_id") != _canonical_sha(event_identity)[:24]:
        return False, "EPISODE_EVENT_IDENTITY_INVALID"
    return True, None


def verify_episode_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime | None = None,
    source_archive_dir: Path | None = None,
) -> tuple[bool, str | None]:
    """Fully verify a ledger without trusting any cached source prefix."""

    return _verify_episode_ledger(
        events,
        as_of_utc=as_of_utc,
        source_archive_dir=source_archive_dir,
        source_verify_start_index=0,
    )


def _verify_episode_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime | None = None,
    source_archive_dir: Path | None = None,
    source_verify_start_index: int = 0,
) -> tuple[bool, str | None]:
    """Verify chains and sources, deeply expanding only the untrusted suffix."""

    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return False, "EPISODE_LEDGER_SCHEMA_INVALID"
    if len(events) > MAX_LEDGER_EVENTS:
        return False, "EPISODE_LEDGER_EVENT_CAP_EXCEEDED"
    if (
        isinstance(source_verify_start_index, bool)
        or not isinstance(source_verify_start_index, int)
        or not 0 <= source_verify_start_index <= len(events)
    ):
        return False, "EPISODE_SOURCE_VERIFY_PREFIX_INVALID"
    as_of = _aware_utc(as_of_utc) if as_of_utc is not None else None
    latest: dict[str, Mapping[str, Any]] = {}
    seen_event_ids: set[str] = set()
    genesis_by_episode: dict[str, Mapping[str, Any]] = {}
    source_cache: dict[str, tuple[str, dict[str, str]]] = {}
    fingerprint_cache: dict[str, str] = {}
    decompressed_total = [0]
    digest_fingerprints: dict[str, str] = {}
    previous_ledger_sha: str | None = None
    if events and source_archive_dir is None:
        return False, "EPISODE_SOURCE_ARCHIVE_REQUIRED"
    if source_archive_dir is not None and source_archive_dir.exists():
        try:
            _archive_inventory(source_archive_dir)
        except (OSError, ValueError) as error:
            if "aggregate cap" in str(error).lower():
                return False, "EPISODE_SOURCE_ARCHIVE_AGGREGATE_CAP_EXCEEDED"
            return False, "EPISODE_SOURCE_ARCHIVE_INVENTORY_INVALID"
    for expected_ledger_seq, raw in enumerate(events, start=1):
        if as_of is not None and isinstance(raw, Mapping):
            generated = _parse_utc(raw.get("generated_at_utc"))
            observation = raw.get("observation")
            observed = _parse_utc(
                observation.get("candle_close_utc")
                if isinstance(observation, Mapping)
                else None
            )
            if (
                generated is not None
                and observed is not None
                and (generated > as_of or observed > as_of)
            ):
                return False, "EPISODE_LEDGER_FUTURE_CLOCK"
        valid, error = verify_episode_event(raw)
        if not valid:
            return False, error
        event = raw
        if (
            event.get("ledger_seq") != expected_ledger_seq
            or event.get("previous_ledger_event_sha256") != previous_ledger_sha
        ):
            return False, "EPISODE_GLOBAL_LEDGER_CHAIN_INVALID"
        observation = event.get("observation")
        digest = str(observation.get("regime_contract_sha256") or "")
        fingerprint = str(observation.get("regime_archive_gzip_sha256") or "")
        prior_fingerprint = digest_fingerprints.get(digest)
        if prior_fingerprint is not None and prior_fingerprint != fingerprint:
            return False, "EPISODE_SOURCE_FINGERPRINT_CONFLICT"
        digest_fingerprints[digest] = fingerprint
        if expected_ledger_seq <= source_verify_start_index:
            membership_valid, membership_error = _verify_regime_fingerprint(
                event,
                archive_dir=source_archive_dir,
                cache=fingerprint_cache,
            )
        else:
            membership_valid, membership_error = _verify_regime_membership(
                event,
                archive_dir=source_archive_dir,
                cache=source_cache,
                fingerprint_cache=fingerprint_cache,
                decompressed_total=decompressed_total,
            )
        if not membership_valid:
            return False, membership_error
        event_id = str(event["event_id"])
        episode_id = str(event["episode_id"])
        if event_id in seen_event_ids:
            return False, "EPISODE_EVENT_ID_DUPLICATE"
        seen_event_ids.add(event_id)
        previous = latest.get(episode_id)
        if previous is None:
            if (
                event.get("event_seq") != 1
                or event.get("previous_event_sha256") is not None
                or not isinstance(event.get("source_m5_evidence"), Mapping)
                or event.get("transition_path")
                not in (
                    ["SETUP", "ATTEMPT"],
                    ["SETUP", "ATTEMPT", "ACCEPTED"],
                    ["SETUP", "ATTEMPT", "REJECTED"],
                    ["SETUP", "ATTEMPT", "INVALIDATED"],
                )
            ):
                return False, "EPISODE_GENESIS_TRANSITION_INVALID"
            genesis_by_episode[episode_id] = event
        else:
            if not _legal_successor(previous, event):
                return False, "EPISODE_TRANSITION_INVALID"
            genesis = genesis_by_episode[episode_id]
            if event.get("anchor") != genesis.get("anchor") or event.get("pair") != genesis.get("pair"):
                return False, "EPISODE_ANCHOR_MUTATED"
            if event.get("late_detected") is False and previous.get("late_detected") is True:
                return False, "EPISODE_LATE_DETECTION_FLAG_REVERSED"
        latest[episode_id] = event
        previous_ledger_sha = str(event["event_sha256"])
    return True, None


def _derive_events(
    *,
    regime_contract: Mapping[str, Any],
    fast_pair_charts: Mapping[str, Any],
    slow_pair_charts: Mapping[str, Any],
    existing_events: Sequence[Mapping[str, Any]],
    now: datetime,
    force_late_detected: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    merged = _merged_pair_views(fast_pair_charts, slow_pair_charts)
    range_rows = _range_rows(regime_contract)
    latest_by_episode = _latest_by_episode(existing_events)
    active_by_pair: dict[str, Mapping[str, Any]] = {}
    for event in latest_by_episode.values():
        if event.get("state") in ACTIVE_STATES:
            pair = str(event.get("pair") or "")
            current = active_by_pair.get(pair)
            if current is None or int(event["event_seq"]) > int(current["event_seq"]):
                active_by_pair[pair] = event
    seen_episode_ids = set(latest_by_episode)
    out: list[dict[str, Any]] = []
    blockers: list[str] = []
    for pair in sorted(set(merged) & set(range_rows)):
        row = range_rows[pair]
        active = active_by_pair.get(pair)
        try:
            if active is not None:
                event, blocker = _advance_episode(
                    previous=active,
                    views=merged[pair],
                    row=row,
                    regime_contract=regime_contract,
                    now=now,
                )
            else:
                event, blocker = _start_episode(
                    pair=pair,
                    views=merged[pair],
                    row=row,
                    regime_contract=regime_contract,
                    now=now,
                )
                if event is not None and event["episode_id"] in seen_episode_ids:
                    event = None
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            event = None
            blocker = f"PAIR_INPUT_INVALID:{type(error).__name__.upper()}"
        if blocker:
            blockers.append(f"{pair}:{blocker}")
        if event is not None:
            if force_late_detected and event.get("late_detected") is not True:
                event_body = _without(event, "event_sha256")
                event_body["late_detected"] = True
                event = {
                    **event_body,
                    "event_sha256": _canonical_sha(event_body),
                }
            out.append(event)
            seen_episode_ids.add(str(event["episode_id"]))
    return out, sorted(set(blockers))


def _start_episode(
    *,
    pair: str,
    views: Mapping[str, Mapping[str, Any]],
    row: Mapping[str, Any],
    regime_contract: Mapping[str, Any],
    now: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    phases = _operating_phases(row)
    if sum(phase in RANGE_PHASES for phase in phases.values()) < 2:
        return None, None
    chart = {"pair": pair, "views": list(views.values())}
    evidence = build_m5_failed_break_evidence(chart)
    evidence_valid, _ = verify_m5_failed_break_evidence(evidence)
    if not evidence_valid or evidence.get("status") != "VALID":
        return None, "M5_BREAK_ATTEMPT_EVIDENCE_UNAVAILABLE"
    candles = evidence.get("candles")
    if not isinstance(candles, list) or len(candles) != 21:
        return None, "M5_BREAK_ATTEMPT_EVIDENCE_UNAVAILABLE"
    current = dict(candles[-1])
    attempt_close = _candle_close(current, "M5")
    if attempt_close is None:
        return None, "M5_ATTEMPT_CANDLE_CLOCK_INVALID"
    age = (now - attempt_close).total_seconds()
    if age < 0.0:
        return None, "M5_ATTEMPT_CANDLE_FUTURE"
    if age > MAX_GENESIS_AGE_SECONDS:
        return None, "M5_ATTEMPT_CANDLE_TOO_OLD"
    rail = _rail_from_evidence(evidence)
    classified = _classify_initial_attempt(current, rail)
    if classified is None:
        return None, None
    attempt_direction, branch = classified
    state = "INVALIDATED" if branch == "AMBIGUOUS" else branch if branch != "PENDING" else "ATTEMPT"
    setup_close = _candle_close(dict(candles[-2]), "M5")
    if setup_close is None:
        return None, "M5_SETUP_CANDLE_CLOCK_INVALID"
    anchor = {
        "episode_kind": EPISODE_KIND,
        "episode_rule_version": EPISODE_RULE_VERSION,
        "horizon_lane": HORIZON_LANE,
        "confirmation_ttl_seconds": CONFIRMATION_TTL_SECONDS,
        "setup_candle_utc": setup_close.isoformat(),
        "attempt_candle_utc": attempt_close.isoformat(),
        "attempt_direction": attempt_direction,
        "rail": rail,
        "source_evidence_sha256": evidence.get("evidence_sha256"),
    }
    route = _route(
        branch=branch,
        attempt_direction=attempt_direction,
        branch_candle=current if branch in {"ACCEPTED", "REJECTED"} else None,
        branch_candle_utc=attempt_close if branch in {"ACCEPTED", "REJECTED"} else None,
    )
    transition_path = ["SETUP", "ATTEMPT"]
    if state in {"ACCEPTED", "REJECTED", "INVALIDATED"}:
        transition_path.append(state)
    reason = {
        "ATTEMPT": "M5_RANGE_RAIL_PIERCED_CLOSE_WITHIN_FROZEN_BUFFER",
        "ACCEPTED": "M5_RANGE_BREAK_ACCEPTED_OUTSIDE_FROZEN_RAIL",
        "REJECTED": "M5_RANGE_BREAK_REJECTED_BACK_INSIDE_FROZEN_RAIL",
        "INVALIDATED": "M5_BOTH_FROZEN_RAILS_PIERCED_AMBIGUOUS",
    }[state]
    return (
        _make_event(
            pair=pair,
            anchor=anchor,
            route=route,
            candle=current,
            timeframe="M5",
            row=row,
            views=views,
            regime_contract=regime_contract,
            now=now,
            state=state,
            transition_path=transition_path,
            transition_reason=reason,
            sequence=1,
            previous=None,
            source_m5_evidence=evidence,
            late_detected=age > LATE_DETECTION_GRACE_SECONDS,
        ),
        None,
    )


def _advance_episode(
    *,
    previous: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    row: Mapping[str, Any],
    regime_contract: Mapping[str, Any],
    now: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    previous_state = str(previous["state"])
    timeframe = "M5" if previous_state == "ATTEMPT" else "M1"
    view = views.get(timeframe)
    candles = _canonical_complete_candles(view)
    previous_close = _parse_utc(previous["observation"]["candle_close_utc"])
    if previous_close is None:
        return None, "EPISODE_PREVIOUS_CLOCK_INVALID"
    next_candle, gap = _next_closed_candle(candles, timeframe=timeframe, after=previous_close, now=now)
    if next_candle is None:
        return None, None
    observed_close = _candle_close(next_candle, timeframe)
    if observed_close is None:
        return None, f"{timeframe}_EPISODE_CANDLE_CLOCK_INVALID"
    anchor = dict(previous["anchor"])
    rail = dict(anchor["rail"])
    route = dict(previous["route"])
    state = previous_state
    reason = ""
    if gap:
        state = "INVALIDATED"
        reason = f"{timeframe}_SEQUENCE_GAP_UNOBSERVABLE"
    elif previous_state == "ATTEMPT":
        direction = str(anchor["attempt_direction"])
        opposite_pierced = (
            direction == "UP" and float(next_candle["l"]) < float(rail["lower"])
        ) or (
            direction == "DOWN" and float(next_candle["h"]) > float(rail["upper"])
        )
        if opposite_pierced:
            state = "INVALIDATED"
            reason = "M5_OPPOSITE_FROZEN_RAIL_PIERCED"
            route = _route(branch="AMBIGUOUS", attempt_direction="AMBIGUOUS")
        else:
            branch = _branch_from_close(direction, float(next_candle["c"]), rail)
            if branch in {"ACCEPTED", "REJECTED"}:
                state = branch
                reason = (
                    "M5_RANGE_BREAK_ACCEPTED_OUTSIDE_FROZEN_RAIL"
                    if branch == "ACCEPTED"
                    else "M5_RANGE_BREAK_REJECTED_BACK_INSIDE_FROZEN_RAIL"
                )
                route = _route(
                    branch=branch,
                    attempt_direction=direction,
                    branch_candle=next_candle,
                    branch_candle_utc=observed_close,
                )
            elif (
                observed_close
                - _parse_utc(anchor["attempt_candle_utc"])
            ).total_seconds() >= CONFIRMATION_TTL_SECONDS:
                state = "EXPIRED"
                reason = "M5_BREAK_ATTEMPT_EXPIRED_WITHIN_BUFFER"
            else:
                reason = "M5_BREAK_ATTEMPT_STILL_WITHIN_FROZEN_BUFFER"
    else:
        state, reason = _confirmation_state(
            previous_state=previous_state,
            anchor=anchor,
            route=route,
            candle=next_candle,
            candle_close=observed_close,
        )
    transition_path = [previous_state] if state == previous_state else [previous_state, state]
    inherited_late = bool(previous.get("late_detected"))
    return (
        _make_event(
            pair=str(previous["pair"]),
            anchor=anchor,
            route=route,
            candle=next_candle,
            timeframe=timeframe,
            row=row,
            views=views,
            regime_contract=regime_contract,
            now=now,
            state=state,
            transition_path=transition_path,
            transition_reason=reason,
            sequence=int(previous["event_seq"]) + 1,
            previous=previous,
            source_m5_evidence=None,
            late_detected=(
                inherited_late
                or (now - observed_close).total_seconds() > LATE_DETECTION_GRACE_SECONDS
            ),
        ),
        None,
    )


def _confirmation_state(
    *,
    previous_state: str,
    anchor: Mapping[str, Any],
    route: Mapping[str, Any],
    candle: Mapping[str, Any],
    candle_close: datetime,
) -> tuple[str, str]:
    direction = str(anchor["attempt_direction"])
    rail = anchor["rail"]
    close = float(candle["c"])
    upper = float(rail["upper"])
    lower = float(rail["lower"])
    buffer = float(rail["buffer"])
    branch_close = _number(route.get("branch_close"))
    branch_at = _parse_utc(route.get("branch_candle_utc"))
    if branch_close is None or branch_at is None or candle_close <= branch_at:
        return "INVALIDATED", "BRANCH_BINDING_INVALID"
    if previous_state == "ACCEPTED":
        invalidated = (direction == "UP" and close < upper - buffer) or (
            direction == "DOWN" and close > lower + buffer
        )
        held = (direction == "UP" and close > upper + buffer) or (
            direction == "DOWN" and close < lower - buffer
        )
    else:
        invalidated = (direction == "UP" and close > upper + buffer) or (
            direction == "DOWN" and close < lower - buffer
        )
        held = (direction == "UP" and close < upper - buffer) or (
            direction == "DOWN" and close > lower + buffer
        )
    if invalidated:
        return "INVALIDATED", "M1_OPPOSITE_BRANCH_INVALIDATED_EPISODE"
    trade_side = str(route.get("trade_side") or "")
    follow_through = (trade_side == "LONG" and close > branch_close) or (
        trade_side == "SHORT" and close < branch_close
    )
    if held and follow_through:
        return "CONFIRMED", "STRICTLY_LATER_M1_HELD_BRANCH_AND_FOLLOWED_THROUGH"
    if (candle_close - branch_at).total_seconds() >= CONFIRMATION_TTL_SECONDS:
        return "EXPIRED", "M1_CONFIRMATION_TTL_EXPIRED"
    return previous_state, "M1_CONFIRMATION_PENDING"


def _make_event(
    *,
    pair: str,
    anchor: Mapping[str, Any],
    route: Mapping[str, Any],
    candle: Mapping[str, Any],
    timeframe: str,
    row: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    regime_contract: Mapping[str, Any],
    now: datetime,
    state: str,
    transition_path: list[str],
    transition_reason: str,
    sequence: int,
    previous: Mapping[str, Any] | None,
    source_m5_evidence: Mapping[str, Any] | None,
    late_detected: bool,
) -> dict[str, Any]:
    observed_close = _candle_close(candle, timeframe)
    if observed_close is None:
        raise ValueError("complete candle has no canonical close clock")
    votes = _canonical_votes(row.get("timeframe_votes"))
    timeframe_clocks = _canonical_timeframe_clocks(
        row.get("source_timeframe_clocks")
    )
    if (
        row.get("source_timeframe_clocks_sha256")
        != _canonical_sha(timeframe_clocks)
        or timeframe_clocks != _timeframe_clocks(views)
    ):
        raise ValueError("episode source clocks do not match the sealed regime row")
    observation = {
        "timeframe": timeframe,
        "candle": dict(candle),
        "candle_close_utc": observed_close.isoformat(),
        "candle_sha256": _canonical_sha(candle),
        "regime_contract_sha256": regime_contract.get("contract_sha256"),
        "regime_archive_gzip_sha256": hashlib.sha256(
            _compressed_regime_contract(regime_contract)
        ).hexdigest(),
        "vote_reference_side": "LONG",
        "source_timeframe_votes": votes,
        "source_timeframe_votes_sha256": _canonical_sha(votes),
        "source_timeframe_clocks": timeframe_clocks,
        "source_timeframe_clocks_sha256": _canonical_sha(timeframe_clocks),
    }
    context_lookahead = any(
        parsed > observed_close
        for value in timeframe_clocks.values()
        for parsed in (_parse_utc(value),)
        if parsed is not None
    )
    episode_identity = {"pair": pair, "anchor": dict(anchor)}
    episode_id = _canonical_sha(episode_identity)[:24]
    previous_sha = previous.get("event_sha256") if previous is not None else None
    event_identity = {
        "episode_id": episode_id,
        "event_seq": sequence,
        "state": state,
        "observation_candle_utc": observed_close.isoformat(),
        "previous_event_sha256": previous_sha,
    }
    state_entered = (
        previous.get("state_entered_at_utc")
        if previous is not None and previous.get("state") == state
        else observed_close.isoformat()
    )
    body = {
        "contract": EPISODE_EVENT_CONTRACT,
        "schema_version": 1,
        "event_id": _canonical_sha(event_identity)[:24],
        "episode_id": episode_id,
        "event_seq": sequence,
        "previous_event_sha256": previous_sha,
        "ledger_seq": 0,
        "previous_ledger_event_sha256": None,
        "generated_at_utc": now.isoformat(),
        "state": state,
        "state_entered_at_utc": state_entered,
        "transition_path": transition_path,
        "transition_reason": transition_reason,
        "pair": pair,
        "anchor": dict(anchor),
        "route": dict(route),
        "observation": observation,
        "source_m5_evidence": dict(source_m5_evidence) if source_m5_evidence is not None else None,
        "late_detected": bool(late_detected or context_lookahead),
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "automatic_rule_change_allowed": False,
        "promotion_allowed": False,
    }
    return {**body, "event_sha256": _canonical_sha(body)}


def _build_state(
    *,
    events: Sequence[Mapping[str, Any]],
    now: datetime,
    processed_at: datetime,
    status: str,
    appended: int,
    blockers: Sequence[str],
    ledger_path: Path,
    source_archive_dir: Path,
    ledger_event_count: int,
    ledger_size_bytes: int,
    ledger_bytes_sha256: str,
    ledger_tail_sha256: str | None,
    ledger_head_verified: bool,
) -> dict[str, Any]:
    processing_delay_seconds = max(0.0, (processed_at - now).total_seconds())
    latest = _latest_by_episode(events)
    ranked_latest = sorted(
        latest.values(),
        key=lambda event: (
            0 if event.get("state") in ACTIVE_STATES else 1,
            -(
                _parse_utc(event.get("generated_at_utc"))
                or now
            ).timestamp(),
            str(event.get("pair") or ""),
            str(event.get("episode_id") or ""),
        ),
    )[:MAX_STATE_EPISODE_SUMMARIES]
    compact = [
        {
            "episode_id": event.get("episode_id"),
            "event_sha256": event.get("event_sha256"),
            "pair": event.get("pair"),
            "state": event.get("state"),
            "attempt_direction": event.get("anchor", {}).get("attempt_direction"),
            "branch_outcome": event.get("route", {}).get("branch_outcome"),
            "trade_side": event.get("route", {}).get("trade_side"),
            "candidate_methods": event.get("route", {}).get("candidate_methods"),
            "observation_candle_utc": event.get("observation", {}).get("candle_close_utc"),
            "late_detected": event.get("late_detected"),
        }
        for event in sorted(
            ranked_latest,
            key=lambda item: (str(item.get("pair") or ""), str(item.get("episode_id") or "")),
        )
    ]
    body = {
        "contract": EPISODE_STATE_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "processed_at_utc": processed_at.isoformat(),
        "processing_delay_seconds": processing_delay_seconds,
        "operationally_late": (
            processing_delay_seconds > LATE_DETECTION_GRACE_SECONDS
        ),
        "status": status,
        "appended_events": appended,
        "ledger_events": len(events),
        "ledger_event_count": ledger_event_count,
        "ledger_size_bytes": ledger_size_bytes,
        "ledger_bytes_sha256": ledger_bytes_sha256,
        "ledger_byte_cap": MAX_LEDGER_BYTES,
        "ledger_event_cap": MAX_LEDGER_EVENTS,
        "ledger_tail_sha256": ledger_tail_sha256,
        "ledger_head_verified": ledger_head_verified,
        "source_verifier_version": SOURCE_VERIFIER_VERSION,
        "source_verified_event_count": (
            ledger_event_count if ledger_head_verified else 0
        ),
        "source_verified_tail_sha256": (
            ledger_tail_sha256 if ledger_head_verified else None
        ),
        "source_fingerprint_index_sha256": (
            _source_fingerprint_index_sha(events)
            if ledger_head_verified
            else None
        ),
        "episodes": len(latest),
        "latest_episode_summaries": len(compact),
        "latest_episodes_truncated": len(compact) < len(latest),
        "latest_episode_summary_cap": MAX_STATE_EPISODE_SUMMARIES,
        "generated_today_events": _generated_on_utc_date(events, now.date()),
        "daily_event_cap": MAX_EVENTS_PER_UTC_DAY,
        "latest_episodes": compact,
        "blockers": list(blockers),
        "episode_scope": EPISODE_KIND,
        "state_machine": [
            "SETUP",
            "ATTEMPT",
            "ACCEPTED|REJECTED",
            "CONFIRMED",
            "ENTRY",
            "RESOLVED|EXPIRED",
        ],
        "accepted_and_rejected_paths_preserved": True,
        "signal_emission_counts_as_entry": False,
        "entry_requires_exact_execution_truth": True,
        "truth_binding_status": "NOT_IMPLEMENTED",
        "promotion_blockers": [
            "EPISODE_TO_EXACT_S5_TRUTH_BINDING_REQUIRED",
            "EPISODE_CLUSTERED_FORWARD_SCORECARD_REQUIRED",
            "CHART_PACKET_TO_REGIME_REPLAY_PROOF_REQUIRED",
            "SEPARATE_ALLOWLISTED_RULE_COMPILER_CONTRACT_REQUIRED",
            "HOT_LEDGER_SEGMENTATION_REQUIRED_BEFORE_BYTE_CAP",
        ],
        "ledger_path": str(ledger_path),
        "source_archive_dir": str(source_archive_dir),
        "source_membership_verified": ledger_head_verified,
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "automatic_rule_change_allowed": False,
        "promotion_allowed": False,
    }
    return _seal(body)


def _route(
    *,
    branch: str,
    attempt_direction: str,
    branch_candle: Mapping[str, Any] | None = None,
    branch_candle_utc: datetime | None = None,
) -> dict[str, Any]:
    trade_side: str | None = None
    methods: list[str]
    route_family: str
    if branch == "ACCEPTED":
        trade_side = "LONG" if attempt_direction == "UP" else "SHORT"
        methods = ["TREND_CONTINUATION"]
        route_family = "BREAKOUT_CONTINUATION"
    elif branch == "REJECTED":
        trade_side = "SHORT" if attempt_direction == "UP" else "LONG"
        methods = ["BREAKOUT_FAILURE", "RANGE_ROTATION"]
        route_family = "RANGE_RECLAIM_OR_BREAKOUT_FAILURE"
    elif branch == "PENDING":
        methods = ["TREND_CONTINUATION", "BREAKOUT_FAILURE", "RANGE_ROTATION"]
        route_family = "UNRESOLVED_BREAK_ATTEMPT"
    else:
        methods = []
        route_family = "AMBIGUOUS"
    return {
        "branch_outcome": branch,
        "trade_side": trade_side,
        "candidate_methods": methods,
        "route_family": route_family,
        "branch_candle_utc": branch_candle_utc.isoformat() if branch_candle_utc else None,
        "branch_close": float(branch_candle["c"]) if branch_candle is not None else None,
        "selection_status": SELECTION_STATUS,
    }


def _classify_initial_attempt(
    candle: Mapping[str, Any],
    rail: Mapping[str, Any],
) -> tuple[str, str] | None:
    up = float(candle["h"]) > float(rail["upper"])
    down = float(candle["l"]) < float(rail["lower"])
    if not up and not down:
        return None
    if up and down:
        return "AMBIGUOUS", "AMBIGUOUS"
    direction = "UP" if up else "DOWN"
    return direction, _branch_from_close(direction, float(candle["c"]), rail)


def _branch_from_close(direction: str, close: float, rail: Mapping[str, Any]) -> str:
    upper = float(rail["upper"])
    lower = float(rail["lower"])
    buffer = float(rail["buffer"])
    if direction == "UP":
        if close > upper + buffer:
            return "ACCEPTED"
        if close < upper - buffer:
            return "REJECTED"
    elif direction == "DOWN":
        if close < lower - buffer:
            return "ACCEPTED"
        if close > lower + buffer:
            return "REJECTED"
    return "PENDING"


def _evidence_matches_anchor_and_route(
    *,
    evidence: Mapping[str, Any],
    anchor: Mapping[str, Any],
    route: Mapping[str, Any],
    observation: Mapping[str, Any],
    state: str,
    transition_reason: str,
) -> bool:
    candles = evidence.get("candles")
    if not isinstance(candles, list) or len(candles) != 21:
        return False
    rail = _rail_from_evidence(evidence)
    current = dict(candles[-1])
    setup_close = _candle_close(dict(candles[-2]), "M5")
    attempt_close = _candle_close(current, "M5")
    classified = _classify_initial_attempt(current, rail)
    if setup_close is None or attempt_close is None or classified is None:
        return False
    direction, branch = classified
    expected_state = "INVALIDATED" if branch == "AMBIGUOUS" else branch if branch != "PENDING" else "ATTEMPT"
    expected_reason = {
        "ATTEMPT": "M5_RANGE_RAIL_PIERCED_CLOSE_WITHIN_FROZEN_BUFFER",
        "ACCEPTED": "M5_RANGE_BREAK_ACCEPTED_OUTSIDE_FROZEN_RAIL",
        "REJECTED": "M5_RANGE_BREAK_REJECTED_BACK_INSIDE_FROZEN_RAIL",
        "INVALIDATED": "M5_BOTH_FROZEN_RAILS_PIERCED_AMBIGUOUS",
    }[expected_state]
    votes = observation.get("source_timeframe_votes")
    operating_range_count = sum(
        isinstance(votes, Mapping)
        and isinstance(votes.get(timeframe), Mapping)
        and str(votes[timeframe].get("phase") or "").upper() in RANGE_PHASES
        for timeframe in ("M5", "M15", "M30")
    )
    route_bound = (
        route.get("branch_candle_utc") == attempt_close.isoformat()
        and route.get("branch_close") == float(current["c"])
        if branch in {"ACCEPTED", "REJECTED"}
        else route.get("branch_candle_utc") is None and route.get("branch_close") is None
    )
    return bool(
        evidence.get("evidence_sha256") == anchor.get("source_evidence_sha256")
        and rail == anchor.get("rail")
        and setup_close.isoformat() == anchor.get("setup_candle_utc")
        and attempt_close.isoformat() == anchor.get("attempt_candle_utc")
        and direction == anchor.get("attempt_direction")
        and branch == route.get("branch_outcome")
        and expected_state == state
        and transition_reason == expected_reason
        and operating_range_count >= 2
        and route_bound
        and observation.get("timeframe") == "M5"
        and observation.get("candle") == current
        and observation.get("candle_close_utc") == attempt_close.isoformat()
    )


def _verify_route(
    route: Mapping[str, Any],
    *,
    state: str,
    attempt_direction: str,
) -> tuple[bool, str | None]:
    branch = route.get("branch_outcome")
    if branch not in BRANCH_OUTCOMES or route.get("selection_status") != SELECTION_STATUS:
        return False, "EPISODE_ROUTE_INVALID"
    if branch in {"ACCEPTED", "REJECTED"} and attempt_direction not in {"UP", "DOWN"}:
        return False, "EPISODE_ROUTE_MAPPING_INVALID"
    expected = _route(branch=str(branch), attempt_direction=attempt_direction)
    for field in ("trade_side", "candidate_methods", "route_family"):
        if route.get(field) != expected.get(field):
            return False, "EPISODE_ROUTE_MAPPING_INVALID"
    branch_at = route.get("branch_candle_utc")
    branch_close = route.get("branch_close")
    if branch in {"ACCEPTED", "REJECTED"}:
        if _parse_utc(branch_at) is None or _number(branch_close) is None:
            return False, "EPISODE_ROUTE_BRANCH_BINDING_INVALID"
        if state not in {branch, "CONFIRMED", "INVALIDATED", "EXPIRED", "ENTRY", "RESOLVED"}:
            return False, "EPISODE_ROUTE_STATE_INVALID"
    else:
        if branch_at is not None or branch_close is not None:
            return False, "EPISODE_ROUTE_BRANCH_BINDING_INVALID"
        if branch == "PENDING" and state not in {"ATTEMPT", "INVALIDATED", "EXPIRED"}:
            return False, "EPISODE_ROUTE_STATE_INVALID"
        if branch == "AMBIGUOUS" and state != "INVALIDATED":
            return False, "EPISODE_ROUTE_STATE_INVALID"
    return True, None


def _verify_observation(value: Mapping[str, Any]) -> tuple[bool, str | None]:
    timeframe = value.get("timeframe")
    candle = value.get("candle")
    if timeframe not in OBSERVATION_TIMEFRAMES or not isinstance(candle, Mapping):
        return False, "EPISODE_OBSERVATION_INVALID"
    canonical = _canonical_candle(candle)
    if canonical is None or canonical != dict(candle):
        return False, "EPISODE_OBSERVATION_CANDLE_INVALID"
    close = _candle_close(canonical, str(timeframe))
    if close is None or value.get("candle_close_utc") != close.isoformat():
        return False, "EPISODE_OBSERVATION_CLOCK_INVALID"
    if value.get("candle_sha256") != _canonical_sha(canonical):
        return False, "EPISODE_OBSERVATION_CANDLE_SHA_MISMATCH"
    if (
        not _sha_text(value.get("regime_contract_sha256"))
        or not _sha_text(value.get("regime_archive_gzip_sha256"))
    ):
        return False, "EPISODE_OBSERVATION_REGIME_BINDING_INVALID"
    votes = value.get("source_timeframe_votes")
    clocks = value.get("source_timeframe_clocks")
    if (
        value.get("vote_reference_side") != "LONG"
        or not isinstance(votes, Mapping)
        or set(votes) != set(TIMEFRAMES)
        or any(not isinstance(item, Mapping) for item in votes.values())
        or any(not _vote_valid(item) for item in votes.values())
        or value.get("source_timeframe_votes_sha256") != _canonical_sha(votes)
        or not isinstance(clocks, Mapping)
        or set(clocks) != set(TIMEFRAMES)
        or any(item is not None and _parse_utc(item) is None for item in clocks.values())
        or clocks.get("M1") is None
        or clocks.get("M5") is None
        or value.get("source_timeframe_clocks_sha256") != _canonical_sha(clocks)
    ):
        return False, "EPISODE_OBSERVATION_VOTES_INVALID"
    return True, None


def _legal_successor(previous: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    previous_state = str(previous["state"])
    state = str(event["state"])
    allowed = {
        "ATTEMPT": {"ATTEMPT", "ACCEPTED", "REJECTED", "INVALIDATED", "EXPIRED"},
        "ACCEPTED": {"ACCEPTED", "CONFIRMED", "INVALIDATED", "EXPIRED"},
        "REJECTED": {"REJECTED", "CONFIRMED", "INVALIDATED", "EXPIRED"},
    }
    previous_close = _parse_utc(previous["observation"]["candle_close_utc"])
    current_close = _parse_utc(event["observation"]["candle_close_utc"])
    expected_timeframe = "M5" if previous_state == "ATTEMPT" else "M1"
    expected_close = (
        previous_close + timedelta(seconds=TIMEFRAME_SECONDS[expected_timeframe])
        if previous_close is not None
        else None
    )
    gap_terminal = (
        state == "INVALIDATED"
        and event.get("transition_reason")
        == f"{expected_timeframe}_SEQUENCE_GAP_UNOBSERVABLE"
    )
    clock_valid = bool(
        current_close is not None
        and expected_close is not None
        and (current_close > expected_close if gap_terminal else current_close == expected_close)
    )
    if (
        previous_state not in allowed
        or state not in allowed[previous_state]
        or event.get("event_seq") != int(previous["event_seq"]) + 1
        or event.get("previous_event_sha256") != previous.get("event_sha256")
        or event.get("source_m5_evidence") is not None
        or previous_close is None
        or current_close is None
        or not clock_valid
        or event["observation"].get("timeframe") != expected_timeframe
    ):
        return False
    expected_path = [previous_state] if state == previous_state else [previous_state, state]
    if event.get("transition_path") != expected_path:
        return False
    expected_entered = (
        previous.get("state_entered_at_utc")
        if state == previous_state
        else current_close.isoformat()
    )
    if event.get("state_entered_at_utc") != expected_entered:
        return False
    semantic_state, semantic_reason, semantic_route = _rederive_successor(previous, event)
    if (
        state != semantic_state
        or event.get("transition_reason") != semantic_reason
        or event.get("route") != semantic_route
    ):
        return False
    previous_branch = previous["route"].get("branch_outcome")
    current_branch = event["route"].get("branch_outcome")
    if previous_state in {"ACCEPTED", "REJECTED"}:
        if current_branch != previous_branch:
            return False
        for field in (
            "trade_side",
            "candidate_methods",
            "route_family",
            "branch_candle_utc",
            "branch_close",
        ):
            if event["route"].get(field) != previous["route"].get(field):
                return False
    return True


def _rederive_successor(
    previous: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    previous_state = str(previous["state"])
    anchor = previous["anchor"]
    rail = anchor["rail"]
    route = dict(previous["route"])
    candle = event["observation"]["candle"]
    candle_close = _parse_utc(event["observation"]["candle_close_utc"])
    if candle_close is None:
        return "INVALIDATED", "BRANCH_BINDING_INVALID", route
    previous_close = _parse_utc(previous["observation"]["candle_close_utc"])
    expected_timeframe = "M5" if previous_state == "ATTEMPT" else "M1"
    expected_close = (
        previous_close + timedelta(seconds=TIMEFRAME_SECONDS[expected_timeframe])
        if previous_close is not None
        else None
    )
    if expected_close is not None and candle_close > expected_close:
        return (
            "INVALIDATED",
            f"{expected_timeframe}_SEQUENCE_GAP_UNOBSERVABLE",
            route,
        )
    if previous_state in {"ACCEPTED", "REJECTED"}:
        state, reason = _confirmation_state(
            previous_state=previous_state,
            anchor=anchor,
            route=route,
            candle=candle,
            candle_close=candle_close,
        )
        return state, reason, route

    direction = str(anchor["attempt_direction"])
    opposite_pierced = (
        direction == "UP" and float(candle["l"]) < float(rail["lower"])
    ) or (
        direction == "DOWN" and float(candle["h"]) > float(rail["upper"])
    )
    if opposite_pierced:
        return (
            "INVALIDATED",
            "M5_OPPOSITE_FROZEN_RAIL_PIERCED",
            _route(branch="AMBIGUOUS", attempt_direction="AMBIGUOUS"),
        )
    branch = _branch_from_close(direction, float(candle["c"]), rail)
    if branch in {"ACCEPTED", "REJECTED"}:
        return (
            branch,
            (
                "M5_RANGE_BREAK_ACCEPTED_OUTSIDE_FROZEN_RAIL"
                if branch == "ACCEPTED"
                else "M5_RANGE_BREAK_REJECTED_BACK_INSIDE_FROZEN_RAIL"
            ),
            _route(
                branch=branch,
                attempt_direction=direction,
                branch_candle=candle,
                branch_candle_utc=candle_close,
            ),
        )
    attempt_at = _parse_utc(anchor.get("attempt_candle_utc"))
    if (
        attempt_at is not None
        and (candle_close - attempt_at).total_seconds() >= CONFIRMATION_TTL_SECONDS
    ):
        return "EXPIRED", "M5_BREAK_ATTEMPT_EXPIRED_WITHIN_BUFFER", route
    return "ATTEMPT", "M5_BREAK_ATTEMPT_STILL_WITHIN_FROZEN_BUFFER", route


def _rail_from_evidence(evidence: Mapping[str, Any]) -> dict[str, float]:
    upper = float(evidence["prior_high"])
    lower = float(evidence["prior_low"])
    width = float(evidence["prior_width"])
    return {
        "upper": upper,
        "lower": lower,
        "width": width,
        "buffer": round(width * BUFFER_RATIO, 10),
        "buffer_ratio": BUFFER_RATIO,
    }


def _rail_valid(value: Mapping[str, Any]) -> bool:
    upper = _number(value.get("upper"))
    lower = _number(value.get("lower"))
    width = _number(value.get("width"))
    buffer = _number(value.get("buffer"))
    return bool(
        upper is not None
        and lower is not None
        and width is not None
        and buffer is not None
        and upper > lower
        and width == round(upper - lower, 10)
        and value.get("buffer_ratio") == BUFFER_RATIO
        and buffer == round(width * BUFFER_RATIO, 10)
    )


def _range_rows(regime_contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in regime_contract.get("rows", []) or []:
        if not isinstance(raw, Mapping):
            continue
        pair = str(raw.get("pair") or "")
        if (
            raw.get("method") == "RANGE_ROTATION"
            and raw.get("side") == "LONG"
            and pair in DEFAULT_TRADER_PAIRS
        ):
            if pair in rows:
                raise ValueError("duplicate episode source regime row")
            votes = raw.get("timeframe_votes")
            clocks = raw.get("source_timeframe_clocks")
            if (
                isinstance(votes, Mapping)
                and set(votes) == set(TIMEFRAMES)
                and isinstance(clocks, Mapping)
                and set(clocks) == set(TIMEFRAMES)
                and raw.get("source_timeframe_clocks_sha256")
                == _canonical_sha(clocks)
            ):
                rows[pair] = raw
    return rows


def _operating_phases(row: Mapping[str, Any]) -> dict[str, str]:
    votes = row.get("timeframe_votes") if isinstance(row.get("timeframe_votes"), Mapping) else {}
    return {
        timeframe: str(votes.get(timeframe, {}).get("phase") or "UNKNOWN").upper()
        if isinstance(votes.get(timeframe), Mapping)
        else "UNKNOWN"
        for timeframe in ("M5", "M15", "M30")
    }


def _canonical_votes(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(TIMEFRAMES):
        raise ValueError("episode source votes must freeze all seven timeframes")
    out: dict[str, dict[str, Any]] = {}
    for timeframe in TIMEFRAMES:
        row = value.get(timeframe)
        if not isinstance(row, Mapping) or not _vote_valid(row):
            raise ValueError("episode source vote must be an object")
        out[timeframe] = dict(row)
    return out


def _timeframe_clocks(
    views: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    clocks: dict[str, str | None] = {}
    for timeframe in TIMEFRAMES:
        candles = _canonical_complete_candles(views.get(timeframe))
        close_times = [
            close
            for candle in candles
            for close in (_candle_close(candle, timeframe),)
            if close is not None
        ]
        clocks[timeframe] = max(close_times).isoformat() if close_times else None
    if clocks["M1"] is None or clocks["M5"] is None:
        raise ValueError("episode source clock manifest requires M1 and M5")
    return clocks


def _canonical_timeframe_clocks(value: object) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or set(value) != set(TIMEFRAMES):
        raise ValueError("episode source clocks must freeze all seven timeframes")
    clocks: dict[str, str | None] = {}
    for timeframe in TIMEFRAMES:
        raw = value.get(timeframe)
        if raw is None:
            clocks[timeframe] = None
            continue
        parsed = _parse_utc(raw)
        if parsed is None:
            raise ValueError("episode source clock must be UTC-aware")
        clocks[timeframe] = parsed.isoformat()
    if clocks["M1"] is None or clocks["M5"] is None:
        raise ValueError("episode source clock manifest requires M1 and M5")
    return clocks


def _merged_pair_views(
    fast_pair_charts: Mapping[str, Any],
    slow_pair_charts: Mapping[str, Any],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    for payload in (slow_pair_charts, fast_pair_charts):
        for chart in payload.get("charts", []) or []:
            if not isinstance(chart, Mapping):
                continue
            pair = str(chart.get("pair") or "")
            if not pair:
                continue
            target = out.setdefault(pair, {})
            for view in chart.get("views", []) or []:
                if not isinstance(view, Mapping):
                    continue
                timeframe = str(view.get("granularity") or "").upper()
                if timeframe in TIMEFRAMES:
                    target[timeframe] = view
    return out


def _canonical_complete_candles(view: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = view.get("recent_candles") if isinstance(view, Mapping) else None
    candles: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping) or item.get("complete") is not True:
            continue
        candle = _canonical_candle(item)
        if candle is not None:
            candles.append(candle)
    candles.sort(key=lambda item: str(item["t"]))
    return candles


def _next_closed_candle(
    candles: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
    after: datetime,
    now: datetime,
) -> tuple[dict[str, Any] | None, bool]:
    expected = after + timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    later = [
        (close, dict(candle))
        for candle in candles
        for close in (_candle_close(candle, timeframe),)
        if close is not None and after < close <= now
    ]
    if not later:
        return None, False
    close, candle = min(later, key=lambda item: item[0])
    if close != expected:
        return candle, True
    return candle, False


def _canonical_candle(value: Mapping[str, Any]) -> dict[str, Any] | None:
    # Live pair-chart candles also carry volume (``v``).  It is intentionally
    # excluded from the episode price predicate, while the sealed observation
    # below is reduced to this exact canonical OHLC schema.
    if not _CANDLE_FIELDS.issubset(set(value)) or value.get("complete") is not True:
        return None
    started = _canonical_timestamp(value.get("t"))
    numbers = [_number(value.get(field)) for field in ("o", "h", "l", "c")]
    if started is None or any(item is None for item in numbers):
        return None
    open_, high, low, close = (round(float(item), 10) for item in numbers)
    if low > min(open_, close) or high < max(open_, close) or high < low:
        return None
    return {
        "t": started,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "complete": True,
    }


def _candle_close(candle: Mapping[str, Any], timeframe: str) -> datetime | None:
    started = _parse_utc(candle.get("t"))
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if started is None or seconds is None:
        return None
    return started + timedelta(seconds=seconds)


def _bind_global_ledger_chain(
    existing_events: Sequence[Mapping[str, Any]],
    new_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    previous_sha = (
        str(existing_events[-1].get("event_sha256")) if existing_events else None
    )
    out: list[dict[str, Any]] = []
    for offset, raw in enumerate(new_events, start=1):
        body = _without(raw, "event_sha256")
        body["ledger_seq"] = len(existing_events) + offset
        body["previous_ledger_event_sha256"] = previous_sha
        event = {**body, "event_sha256": _canonical_sha(body)}
        out.append(event)
        previous_sha = event["event_sha256"]
    return out


def _ledger_line(event: Mapping[str, Any]) -> bytes:
    raw = _canonical_json_bytes(event) + b"\n"
    if len(raw) > MAX_EVENT_BYTES:
        raise ValueError("episode ledger event exceeds the row byte cap")
    return raw


def _archive_regime_contract(
    regime_contract: Mapping[str, Any],
    archive_dir: Path,
) -> tuple[Path, bool]:
    if not _sealed_contract_valid(regime_contract, REGIME_CONTRACT):
        raise ValueError("cannot archive an invalid regime contract")
    raw = _canonical_json_bytes(regime_contract)
    if len(raw) > MAX_SOURCE_CONTRACT_BYTES:
        raise ValueError("episode regime source exceeds the archive byte cap")
    digest = str(regime_contract["contract_sha256"])
    archive_dir.mkdir(parents=True, exist_ok=True)
    if archive_dir.is_symlink():
        raise ValueError("episode source archive root must not be a symlink")
    path = archive_dir / f"{digest}.json.gz"
    compressed = _compressed_regime_contract(regime_contract)
    if len(compressed) > MAX_SOURCE_ARCHIVE_BYTES:
        raise ValueError("compressed episode regime source exceeds the archive cap")
    directory_fd = os.open(
        archive_dir,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    directory_locked = False
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            directory_locked = True
        except BlockingIOError as error:
            raise ValueError("episode source archive lock busy") from error
        existing, existing_error = _read_archive_compressed(archive_dir, digest)
        if existing is not None:
            if existing != compressed:
                raise ValueError("episode source archive digest collision")
            return path, False
        if existing_error != "EPISODE_SOURCE_ARCHIVE_MISSING":
            raise ValueError(existing_error or "episode source archive invalid")

        file_count, total_bytes = _archive_inventory(archive_dir)
        if (
            file_count + 1 > MAX_SOURCE_ARCHIVE_FILES
            or total_bytes + len(compressed) > MAX_SOURCE_ARCHIVE_TOTAL_BYTES
        ):
            raise ValueError("episode source archive aggregate cap reached")
        temp_name = f".{digest}.{os.getpid()}{time.time_ns()}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(descriptor, "wb") as temp_handle:
                descriptor = -1
                temp_handle.write(compressed)
                temp_handle.flush()
                os.fsync(temp_handle.fileno())
            os.rename(
                temp_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        return path, True
    finally:
        if directory_locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _ensure_archive_owner(archive_dir: Path, ledger_path: Path) -> str | None:
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return "EPISODE_SOURCE_ARCHIVE_ROOT_CREATE_FAILED"
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        return "EPISODE_SOURCE_ARCHIVE_ROOT_INVALID"
    owner_path = archive_dir / ".owner.json"
    ledger_identity = _canonical_sha(
        {"ledger_path": str(ledger_path.resolve(strict=False))}
    )
    try:
        directory_fd = os.open(
            archive_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        return "EPISODE_SOURCE_ARCHIVE_ROOT_INVALID"
    directory_locked = False
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            directory_locked = True
        except BlockingIOError:
            return "EPISODE_SOURCE_ARCHIVE_LOCK_BUSY"
        owner, owner_read_error = _read_bounded_regular_json(
            owner_path,
            max_bytes=16 * 1024,
        )
        if owner_read_error != "MISSING":
            if owner_read_error == "SYMLINK":
                return "EPISODE_SOURCE_ARCHIVE_OWNER_SYMLINK"
            if owner_read_error is not None:
                return "EPISODE_SOURCE_ARCHIVE_OWNER_INVALID"
            if (
                not isinstance(owner, Mapping)
                or not _sealed_contract_valid(
                    owner,
                    EPISODE_ARCHIVE_OWNER_CONTRACT,
                )
                or isinstance(owner.get("schema_version"), bool)
                or owner.get("schema_version") != 1
                or owner.get("ledger_identity_sha256") != ledger_identity
                or owner.get("order_authority") != "NONE"
                or owner.get("live_permission") is not False
                or owner.get("broker_mutation_allowed") is not False
            ):
                return "EPISODE_SOURCE_ARCHIVE_OWNER_MISMATCH"
            return None
        removed_owner_temp = False
        for entry in list(archive_dir.iterdir()):
            if re.fullmatch(r"\.\.owner\.json\.[0-9]+\.tmp", entry.name):
                try:
                    entry_stat = entry.lstat()
                    if not (
                        stat.S_ISREG(entry_stat.st_mode)
                        or stat.S_ISLNK(entry_stat.st_mode)
                    ):
                        return "EPISODE_SOURCE_ARCHIVE_OWNER_TEMP_INVALID"
                    os.unlink(entry.name, dir_fd=directory_fd)
                    removed_owner_temp = True
                except OSError:
                    return "EPISODE_SOURCE_ARCHIVE_OWNER_TEMP_CLEANUP_FAILED"
        if removed_owner_temp:
            os.fsync(directory_fd)
        if any(archive_dir.iterdir()):
            return "EPISODE_SOURCE_ARCHIVE_OWNER_MISSING_WITH_CONTENT"
        owner = _seal(
            {
                "contract": EPISODE_ARCHIVE_OWNER_CONTRACT,
                "schema_version": 1,
                "ledger_identity_sha256": ledger_identity,
                "diagnostic_only": True,
                "shadow_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "broker_mutation_allowed": False,
            }
        )
        _write_json_atomic(owner_path, owner, max_bytes=16 * 1024)
        return None
    finally:
        if directory_locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _compressed_regime_contract(regime_contract: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=buffer,
        mtime=0,
    ) as handle:
        handle.write(_canonical_json_bytes(regime_contract))
    return buffer.getvalue()


def _archive_inventory(archive_dir: Path) -> tuple[int, int]:
    if not archive_dir.exists():
        return 0, 0
    if archive_dir.is_symlink():
        raise ValueError("episode source archive root must not be a symlink")
    file_count = 0
    total_bytes = 0
    for path in archive_dir.iterdir():
        if path.name == ".owner.json":
            continue
        if not re.fullmatch(r"[0-9a-f]{64}\.json\.gz", path.name):
            raise ValueError("unexpected episode source archive entry")
        entry_stat = path.lstat()
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError("episode source archive entry is not a regular file")
        size = entry_stat.st_size
        if size <= 0 or size > MAX_SOURCE_ARCHIVE_BYTES:
            raise ValueError("episode source archive entry size is invalid")
        file_count += 1
        total_bytes += size
        if (
            file_count > MAX_SOURCE_ARCHIVE_FILES
            or total_bytes > MAX_SOURCE_ARCHIVE_TOTAL_BYTES
        ):
            raise ValueError("episode source archive aggregate cap exceeded")
    return file_count, total_bytes


def _read_archive_compressed(
    archive_dir: Path,
    digest: str,
) -> tuple[bytes | None, str | None]:
    if not _sha_text(digest):
        return None, "EPISODE_SOURCE_ARCHIVE_DIGEST_INVALID"
    try:
        directory_fd = os.open(
            archive_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        return None, "EPISODE_SOURCE_ARCHIVE_MISSING"
    except OSError:
        return None, "EPISODE_SOURCE_ARCHIVE_ROOT_INVALID"
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                f"{digest}.json.gz",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None, "EPISODE_SOURCE_ARCHIVE_MISSING"
        except OSError:
            return None, "EPISODE_SOURCE_ARCHIVE_OPEN_INVALID"
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_SOURCE_ARCHIVE_BYTES
        ):
            return None, "EPISODE_SOURCE_ARCHIVE_SIZE_INVALID"
        chunks: list[bytes] = []
        remaining = MAX_SOURCE_ARCHIVE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        compressed = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(compressed) != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            return None, "EPISODE_SOURCE_ARCHIVE_CHANGED_DURING_READ"
        return compressed, None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def _read_archived_regime_contract(
    archive_dir: Path,
    digest: str,
    *,
    compressed: bytes | None = None,
    decompressed_budget: int | None = None,
) -> tuple[dict[str, Any] | None, int, str | None]:
    if compressed is None:
        compressed, compressed_error = _read_archive_compressed(
            archive_dir,
            digest,
        )
        if compressed is None:
            return None, 0, compressed_error or "EPISODE_SOURCE_ARCHIVE_INVALID"
    read_limit = MAX_SOURCE_CONTRACT_BYTES
    if decompressed_budget is not None:
        read_limit = min(read_limit, max(0, decompressed_budget))
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
            raw = handle.read(read_limit + 1)
    except (EOFError, OSError):
        return None, 0, "EPISODE_SOURCE_ARCHIVE_GZIP_INVALID"
    if len(raw) > read_limit and read_limit < MAX_SOURCE_CONTRACT_BYTES:
        return (
            None,
            len(raw),
            "EPISODE_SOURCE_VERIFY_DECOMPRESSED_CAP_EXCEEDED",
        )
    if len(raw) > MAX_SOURCE_CONTRACT_BYTES:
        return None, len(raw), "EPISODE_SOURCE_ARCHIVE_DECOMPRESSED_TOO_LARGE"
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None, len(raw), "EPISODE_SOURCE_ARCHIVE_JSON_INVALID"
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        return None, len(raw), "EPISODE_SOURCE_ARCHIVE_CANONICAL_INVALID"
    if (
        not _sealed_contract_valid(value, REGIME_CONTRACT)
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
        or value.get("contract_sha256") != digest
    ):
        return None, len(raw), "EPISODE_SOURCE_ARCHIVE_CONTRACT_INVALID"
    if compressed != _compressed_regime_contract(value):
        return None, len(raw), "EPISODE_SOURCE_ARCHIVE_COMPRESSION_NONCANONICAL"
    return value, len(raw), None


def _verify_regime_fingerprint(
    event: Mapping[str, Any],
    *,
    archive_dir: Path | None,
    cache: dict[str, str],
) -> tuple[bool, str | None]:
    if archive_dir is None:
        return False, "EPISODE_SOURCE_ARCHIVE_REQUIRED"
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        return False, "EPISODE_SOURCE_MEMBERSHIP_INVALID"
    digest = observation.get("regime_contract_sha256")
    expected = observation.get("regime_archive_gzip_sha256")
    if not _sha_text(digest) or not _sha_text(expected):
        return False, "EPISODE_SOURCE_MEMBERSHIP_INVALID"
    cached = cache.get(str(digest))
    if cached is not None:
        if cached != expected:
            return False, "EPISODE_SOURCE_FINGERPRINT_CONFLICT"
        return True, None
    compressed, error = _read_archive_compressed(archive_dir, str(digest))
    if compressed is None:
        return False, error or "EPISODE_SOURCE_ARCHIVE_INVALID"
    actual = hashlib.sha256(compressed).hexdigest()
    if actual != expected:
        return False, "EPISODE_SOURCE_ARCHIVE_GZIP_SHA_MISMATCH"
    cache[str(digest)] = actual
    return True, None


def _verify_regime_membership(
    event: Mapping[str, Any],
    *,
    archive_dir: Path | None,
    cache: dict[str, tuple[str, dict[str, str]]],
    fingerprint_cache: dict[str, str],
    decompressed_total: list[int],
) -> tuple[bool, str | None]:
    if archive_dir is None:
        return False, "EPISODE_SOURCE_ARCHIVE_REQUIRED"
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        return False, "EPISODE_SOURCE_MEMBERSHIP_INVALID"
    digest = observation.get("regime_contract_sha256")
    if not _sha_text(digest):
        return False, "EPISODE_SOURCE_MEMBERSHIP_INVALID"
    fingerprint_valid, fingerprint_error = _verify_regime_fingerprint(
        event,
        archive_dir=archive_dir,
        cache=fingerprint_cache,
    )
    if not fingerprint_valid:
        return False, fingerprint_error
    indexed = cache.get(str(digest))
    if indexed is None:
        remaining_decompressed = (
            MAX_SOURCE_VERIFY_DECOMPRESSED_BYTES - decompressed_total[0]
        )
        if remaining_decompressed <= 0:
            return False, "EPISODE_SOURCE_VERIFY_DECOMPRESSED_CAP_EXCEEDED"
        compressed, compressed_error = _read_archive_compressed(
            archive_dir,
            str(digest),
        )
        if compressed is None:
            return False, compressed_error or "EPISODE_SOURCE_ARCHIVE_INVALID"
        loaded, raw_size, error = _read_archived_regime_contract(
            archive_dir,
            str(digest),
            compressed=compressed,
            decompressed_budget=remaining_decompressed,
        )
        if loaded is None:
            return False, error or "EPISODE_SOURCE_ARCHIVE_INVALID"
        decompressed_total[0] += raw_size
        if decompressed_total[0] > MAX_SOURCE_VERIFY_DECOMPRESSED_BYTES:
            return False, "EPISODE_SOURCE_VERIFY_DECOMPRESSED_CAP_EXCEEDED"
        indexed, index_error = _regime_membership_index(loaded)
        if indexed is None:
            return False, index_error or "EPISODE_SOURCE_ROWS_INVALID"
        cache[str(digest)] = indexed
    regime_generated_text, row_index = indexed
    generated = _parse_utc(event.get("generated_at_utc"))
    regime_generated = _parse_utc(regime_generated_text)
    if generated is None or regime_generated is None or generated != regime_generated:
        return False, "EPISODE_SOURCE_CYCLE_CLOCK_MISMATCH"
    votes = observation.get("source_timeframe_votes")
    clocks = observation.get("source_timeframe_clocks")
    membership_sha = _canonical_sha(
        {
            "timeframe_votes": votes,
            "source_timeframe_clocks": clocks,
            "source_timeframe_clocks_sha256": _canonical_sha(clocks),
            "m1_closed_candle_utc": (
                clocks.get("M1") if isinstance(clocks, Mapping) else None
            ),
        }
    )
    if observation.get("vote_reference_side") != "LONG" or row_index.get(
        str(event.get("pair") or "")
    ) != membership_sha:
        return False, "EPISODE_SOURCE_ROW_MEMBERSHIP_INVALID"
    return True, None


def _regime_membership_index(
    contract: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, str]] | None, str | None]:
    generated = contract.get("generated_at_utc")
    rows = contract.get("rows")
    if not isinstance(generated, str) or not isinstance(rows, list):
        return None, "EPISODE_SOURCE_ROWS_INVALID"
    identities: set[tuple[str, str, str]] = set()
    relevant: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            return None, "EPISODE_SOURCE_ROWS_INVALID"
        identity = (
            str(raw.get("pair") or ""),
            str(raw.get("side") or ""),
            str(raw.get("method") or ""),
        )
        if identity in identities:
            return None, "EPISODE_SOURCE_ROW_DUPLICATE"
        identities.add(identity)
        if identity[1:] != ("LONG", "RANGE_ROTATION"):
            continue
        relevant[identity[0]] = _canonical_sha(
            {
                "timeframe_votes": raw.get("timeframe_votes"),
                "source_timeframe_clocks": raw.get("source_timeframe_clocks"),
                "source_timeframe_clocks_sha256": raw.get(
                    "source_timeframe_clocks_sha256"
                ),
                "m1_closed_candle_utc": raw.get("m1_closed_candle_utc"),
            }
        )
    return (generated, relevant), None


def _read_state_checkpoint(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    value, read_error = _read_bounded_regular_json(
        path,
        max_bytes=MAX_STATE_BYTES,
    )
    if read_error == "MISSING":
        return None, None
    if read_error == "SYMLINK":
        return None, "EPISODE_STATE_CHECKPOINT_SYMLINK"
    if read_error == "TOO_LARGE":
        return None, "EPISODE_STATE_CHECKPOINT_TOO_LARGE"
    if read_error is not None:
        return None, "EPISODE_STATE_CHECKPOINT_INVALID"
    if not isinstance(value, dict):
        return None, "EPISODE_STATE_CHECKPOINT_INVALID"
    return value, None


def _verify_head_checkpoint(
    state: Mapping[str, Any] | None,
    *,
    events: Sequence[Mapping[str, Any]],
    ledger_raw: bytes,
    prefix_sizes: Sequence[int],
) -> tuple[bool, str | None, int]:
    del prefix_sizes
    if state is None:
        if events or ledger_raw:
            return False, "EPISODE_LEDGER_HEAD_CHECKPOINT_MISSING", 0
        return True, None, 0
    descriptor = _head_descriptor(events, ledger_raw)
    if not _state_checkpoint_valid(state):
        return False, "EPISODE_STATE_CHECKPOINT_INVALID", 0
    if not _state_matches_descriptor(state, descriptor):
        return False, "EPISODE_LEDGER_HEAD_CHECKPOINT_MISMATCH", 0
    return True, None, _trusted_source_prefix(state, events)


def _verify_pending_checkpoint(
    state: Mapping[str, Any] | None,
    *,
    pending: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    ledger_raw: bytes,
    prefix_sizes: Sequence[int],
) -> tuple[bool, str | None, int]:
    if not isinstance(state, Mapping) or not _state_checkpoint_valid(state):
        return False, "EPISODE_PENDING_BASE_STATE_INVALID", 0
    base = pending.get("base_head")
    target = pending.get("target_head")
    if not isinstance(base, Mapping) or not isinstance(target, Mapping):
        return False, "EPISODE_PENDING_HEAD_INVALID", 0
    target_descriptor = _head_descriptor(events, ledger_raw)
    if dict(target) != target_descriptor:
        return False, "EPISODE_PENDING_TARGET_MISMATCH", 0

    base_count = int(base.get("event_count") or 0)
    base_size = int(base.get("size_bytes") or 0)
    if base_count > len(events) or base_size > len(ledger_raw):
        return False, "EPISODE_PENDING_BASE_MISMATCH", 0
    expected_prefix_size = prefix_sizes[base_count - 1] if base_count else 0
    base_events = list(events[:base_count])
    if (
        expected_prefix_size != base_size
        or _head_descriptor(base_events, ledger_raw[:base_size]) != dict(base)
    ):
        return False, "EPISODE_PENDING_BASE_MISMATCH", 0
    if _state_matches_descriptor(state, target):
        return True, None, _trusted_source_prefix(state, events)
    if not _state_matches_descriptor(state, base):
        return False, "EPISODE_PENDING_STATE_HEAD_MISMATCH", 0
    return True, None, _trusted_source_prefix(state, base_events)


def _checkpoint_head(
    state: Mapping[str, Any] | None,
) -> tuple[int, int, str, str | None]:
    if not isinstance(state, Mapping) or not _state_checkpoint_valid(state):
        return 0, 0, hashlib.sha256(b"").hexdigest(), None
    count = state.get("ledger_event_count")
    size = state.get("ledger_size_bytes")
    ledger_sha = state.get("ledger_bytes_sha256")
    tail = state.get("ledger_tail_sha256")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not _sha_text(ledger_sha)
        or (tail is not None and not _sha_text(tail))
    ):
        return 0, 0, hashlib.sha256(b"").hexdigest(), None
    return count, size, str(ledger_sha), tail


def _read_ledger_bytes(
    raw: bytes,
) -> tuple[list[dict[str, Any]], list[int], str | None]:
    events: list[dict[str, Any]] = []
    prefix_sizes: list[int] = []
    if not raw:
        return events, prefix_sizes, None
    if not raw.endswith(b"\n"):
        return events, prefix_sizes, "EPISODE_LEDGER_PARTIAL_ROW"
    offset = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line or line in {b"\n", b"\r\n"}:
            return events, prefix_sizes, f"EPISODE_LEDGER_BLANK_LINE_{line_number}"
        if len(line) > MAX_EVENT_BYTES:
            return events, prefix_sizes, f"EPISODE_LEDGER_ROW_TOO_LARGE_LINE_{line_number}"
        try:
            text = line.decode("utf-8")
            value = json.loads(text, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return events, prefix_sizes, f"EPISODE_LEDGER_JSON_INVALID_LINE_{line_number}"
        if not isinstance(value, dict):
            return events, prefix_sizes, f"EPISODE_LEDGER_ROW_INVALID_LINE_{line_number}"
        try:
            canonical = _ledger_line(value)
        except (TypeError, ValueError, OverflowError):
            return events, prefix_sizes, f"EPISODE_LEDGER_ROW_INVALID_LINE_{line_number}"
        if line != canonical:
            return events, prefix_sizes, f"EPISODE_LEDGER_NONCANONICAL_LINE_{line_number}"
        events.append(value)
        offset += len(line)
        prefix_sizes.append(offset)
    return events, prefix_sizes, None


def _state_checkpoint_valid(state: Mapping[str, Any]) -> bool:
    return bool(
        _sealed_contract_valid(state, EPISODE_STATE_CONTRACT)
        and not isinstance(state.get("schema_version"), bool)
        and state.get("schema_version") == 1
        and state.get("diagnostic_only") is True
        and state.get("shadow_only") is True
        and state.get("order_authority") == "NONE"
        and state.get("live_permission") is False
        and state.get("broker_mutation_allowed") is False
        and state.get("automatic_rule_change_allowed") is False
        and state.get("promotion_allowed") is False
    )


def _head_descriptor(
    events: Sequence[Mapping[str, Any]],
    ledger_raw: bytes,
) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "size_bytes": len(ledger_raw),
        "bytes_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "tail_sha256": (
            str(events[-1].get("event_sha256")) if events else None
        ),
    }


def _state_matches_descriptor(
    state: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> bool:
    return bool(
        _state_checkpoint_valid(state)
        and state.get("ledger_head_verified") is True
        and state.get("ledger_event_count") == descriptor.get("event_count")
        and state.get("ledger_size_bytes") == descriptor.get("size_bytes")
        and state.get("ledger_bytes_sha256") == descriptor.get("bytes_sha256")
        and state.get("ledger_tail_sha256") == descriptor.get("tail_sha256")
    )


def _source_fingerprint_index_sha(
    events: Sequence[Mapping[str, Any]],
) -> str:
    rows = sorted(
        {
            (
                str(event.get("observation", {}).get("regime_contract_sha256") or ""),
                str(event.get("observation", {}).get("regime_archive_gzip_sha256") or ""),
            )
            for event in events
            if isinstance(event, Mapping)
            and isinstance(event.get("observation"), Mapping)
        }
    )
    return _canonical_sha(
        [
            {"regime_contract_sha256": digest, "gzip_sha256": fingerprint}
            for digest, fingerprint in rows
        ]
    )


def _trusted_source_prefix(
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> int:
    tail = str(events[-1].get("event_sha256")) if events else None
    if (
        state.get("source_membership_verified") is True
        and state.get("source_verifier_version") == SOURCE_VERIFIER_VERSION
        and state.get("source_verified_event_count") == len(events)
        and state.get("source_verified_tail_sha256") == tail
        and state.get("source_fingerprint_index_sha256")
        == _source_fingerprint_index_sha(events)
    ):
        return len(events)
    return 0


def _pending_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.pending")


def _build_pending_checkpoint(
    *,
    base_events: Sequence[Mapping[str, Any]],
    base_ledger_raw: bytes,
    batch_events: Sequence[Mapping[str, Any]],
    batch_raw: bytes,
    regime_contract: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    target_events = [*base_events, *batch_events]
    target_raw = base_ledger_raw + batch_raw
    source_fingerprints = sorted(
        {
            str(event.get("observation", {}).get("regime_archive_gzip_sha256") or "")
            for event in batch_events
        }
    )
    body = {
        "contract": EPISODE_PENDING_CONTRACT,
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "base_head": _head_descriptor(base_events, base_ledger_raw),
        "batch_event_count": len(batch_events),
        "batch_size_bytes": len(batch_raw),
        "batch_bytes_sha256": hashlib.sha256(batch_raw).hexdigest(),
        "batch_jsonl_base64": base64.b64encode(batch_raw).decode("ascii"),
        "target_head": _head_descriptor(target_events, target_raw),
        "source": {
            "generated_at_utc": regime_contract.get("generated_at_utc"),
            "regime_contract_sha256": regime_contract.get("contract_sha256"),
            "regime_archive_gzip_sha256": (
                source_fingerprints[0] if len(source_fingerprints) == 1 else None
            ),
        },
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "automatic_rule_change_allowed": False,
        "promotion_allowed": False,
    }
    pending = _seal(body)
    if not _pending_contract_valid(pending):
        raise ValueError("episode pending checkpoint is internally invalid")
    return pending


def _read_pending_checkpoint(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    value, read_error = _read_bounded_regular_json(
        path,
        max_bytes=MAX_PENDING_BYTES,
    )
    if read_error == "MISSING":
        return None, None
    if read_error == "SYMLINK":
        return None, "EPISODE_PENDING_CHECKPOINT_SYMLINK"
    if read_error == "TOO_LARGE":
        return None, "EPISODE_PENDING_CHECKPOINT_TOO_LARGE"
    if read_error is not None:
        return None, "EPISODE_PENDING_CHECKPOINT_INVALID"
    if not isinstance(value, dict) or not _pending_contract_valid(value):
        return None, "EPISODE_PENDING_CHECKPOINT_INVALID"
    return value, None


def _pending_contract_valid(value: Mapping[str, Any]) -> bool:
    if (
        not _sealed_contract_valid(value, EPISODE_PENDING_CONTRACT)
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
        or _parse_utc(value.get("created_at_utc")) is None
        or value.get("diagnostic_only") is not True
        or value.get("shadow_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("automatic_rule_change_allowed") is not False
        or value.get("promotion_allowed") is not False
    ):
        return False
    base = value.get("base_head")
    target = value.get("target_head")
    if not _head_descriptor_value_valid(base) or not _head_descriptor_value_valid(target):
        return False
    batch_count = value.get("batch_event_count")
    batch_size = value.get("batch_size_bytes")
    encoded = value.get("batch_jsonl_base64")
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or not 1 <= batch_count <= len(DEFAULT_TRADER_PAIRS)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 0 < batch_size <= len(DEFAULT_TRADER_PAIRS) * MAX_EVENT_BYTES
        or not isinstance(encoded, str)
        or not _sha_text(value.get("batch_bytes_sha256"))
    ):
        return False
    try:
        batch_raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return False
    batch_events, _, read_error = _read_ledger_bytes(batch_raw)
    if (
        read_error is not None
        or len(batch_raw) != batch_size
        or hashlib.sha256(batch_raw).hexdigest() != value.get("batch_bytes_sha256")
        or len(batch_events) != batch_count
        or target.get("event_count") != base.get("event_count") + batch_count
        or target.get("size_bytes") != base.get("size_bytes") + batch_size
        or target.get("tail_sha256")
        != str(batch_events[-1].get("event_sha256"))
    ):
        return False
    source = value.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "generated_at_utc",
            "regime_contract_sha256",
            "regime_archive_gzip_sha256",
        }
        or _parse_utc(source.get("generated_at_utc")) is None
        or not _sha_text(source.get("regime_contract_sha256"))
        or not _sha_text(source.get("regime_archive_gzip_sha256"))
    ):
        return False
    return all(
        event.get("generated_at_utc") == source.get("generated_at_utc")
        and event.get("observation", {}).get("regime_contract_sha256")
        == source.get("regime_contract_sha256")
        and event.get("observation", {}).get("regime_archive_gzip_sha256")
        == source.get("regime_archive_gzip_sha256")
        for event in batch_events
    )


def _head_descriptor_value_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "event_count",
        "size_bytes",
        "bytes_sha256",
        "tail_sha256",
    }:
        return False
    count = value.get("event_count")
    size = value.get("size_bytes")
    tail = value.get("tail_sha256")
    return bool(
        not isinstance(count, bool)
        and isinstance(count, int)
        and 0 <= count <= MAX_LEDGER_EVENTS
        and not isinstance(size, bool)
        and isinstance(size, int)
        and 0 <= size <= MAX_LEDGER_BYTES
        and _sha_text(value.get("bytes_sha256"))
        and ((count == 0 and tail is None) or (count > 0 and _sha_text(tail)))
    )


def _prepare_pending_recovery(
    *,
    ledger_raw: bytes,
    state: Mapping[str, Any] | None,
    pending: Mapping[str, Any],
) -> tuple[bytes, str | None, bytes, str | None]:
    if not isinstance(state, Mapping) or not _state_checkpoint_valid(state):
        return ledger_raw, None, b"", "EPISODE_PENDING_BASE_STATE_INVALID"
    base = pending["base_head"]
    target = pending["target_head"]
    batch_raw = base64.b64decode(str(pending["batch_jsonl_base64"]), validate=True)
    if _state_matches_descriptor(state, target):
        if _raw_matches_descriptor(ledger_raw, target):
            return ledger_raw, "PENDING_ALREADY_COMMITTED", b"", None
        return ledger_raw, None, b"", "EPISODE_PENDING_STATE_AHEAD_OF_LEDGER"
    if not _state_matches_descriptor(state, base):
        return ledger_raw, None, b"", "EPISODE_PENDING_STATE_HEAD_MISMATCH"
    base_size = int(base["size_bytes"])
    if len(ledger_raw) < base_size or not _raw_matches_descriptor(
        ledger_raw[:base_size],
        base,
    ):
        return ledger_raw, None, b"", "EPISODE_PENDING_BASE_BYTES_MISMATCH"
    suffix = ledger_raw[base_size:]
    if len(suffix) > len(batch_raw) or not batch_raw.startswith(suffix):
        return ledger_raw, None, b"", "EPISODE_PENDING_SUFFIX_MISMATCH"
    prospective = ledger_raw + batch_raw[len(suffix) :]
    if len(suffix) < len(batch_raw):
        mode = "PENDING_SUFFIX_COMPLETED"
    else:
        mode = "PENDING_LEDGER_ALREADY_COMPLETE"
    if not _raw_matches_descriptor(prospective, target):
        return ledger_raw, None, b"", "EPISODE_PENDING_TARGET_BYTES_MISMATCH"
    return prospective, mode, batch_raw[len(suffix) :], None


def _raw_matches_descriptor(raw: bytes, descriptor: Mapping[str, Any]) -> bool:
    return bool(
        len(raw) == descriptor.get("size_bytes")
        and hashlib.sha256(raw).hexdigest() == descriptor.get("bytes_sha256")
    )


def _write_pending_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(raw) > MAX_PENDING_BYTES:
        raise ValueError("episode pending checkpoint exceeds byte cap")
    _write_json_atomic(path, value, max_bytes=MAX_PENDING_BYTES)


def _cleanup_episode_temps(
    *,
    output_path: Path,
    pending_path: Path,
    archive_dir: Path,
) -> str | None:
    targets = (
        (
            output_path.parent,
            re.compile(rf"\.{re.escape(output_path.name)}\.[0-9]+\.tmp"),
        ),
        (
            pending_path.parent,
            re.compile(rf"\.{re.escape(pending_path.name)}\.[0-9]+\.tmp"),
        ),
        (
            archive_dir,
            re.compile(
                r"(?:\.[0-9a-f]{64}\.[0-9]+\.tmp|"
                r"\.\.owner\.json\.[0-9]+\.tmp)"
            ),
        ),
    )
    changed: set[Path] = set()
    for parent, pattern in targets:
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            return "EPISODE_TEMP_CLEANUP_ROOT_INVALID"
        try:
            entries = list(parent.iterdir())
        except OSError:
            return "EPISODE_TEMP_CLEANUP_SCAN_FAILED"
        for path in entries:
            if pattern.fullmatch(path.name) is None:
                continue
            try:
                entry_stat = path.lstat()
                if not (
                    stat.S_ISREG(entry_stat.st_mode)
                    or stat.S_ISLNK(entry_stat.st_mode)
                ):
                    return "EPISODE_TEMP_CLEANUP_ENTRY_INVALID"
                path.unlink()
                changed.add(parent)
            except OSError:
                return "EPISODE_TEMP_CLEANUP_FAILED"
    for parent in changed:
        _fsync_directory(parent)
    return None


def _referenced_regime_digests(
    events: Sequence[Mapping[str, Any]],
) -> set[str]:
    return {
        str(event.get("observation", {}).get("regime_contract_sha256"))
        for event in events
        if isinstance(event, Mapping)
        and isinstance(event.get("observation"), Mapping)
        and _sha_text(event.get("observation", {}).get("regime_contract_sha256"))
    }


def _cleanup_unreferenced_archives(
    archive_dir: Path,
    *,
    referenced_digests: set[str],
) -> str | None:
    if not archive_dir.exists():
        return None if not referenced_digests else "EPISODE_SOURCE_ARCHIVE_MISSING"
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        return "EPISODE_SOURCE_ARCHIVE_ROOT_INVALID"
    try:
        directory_fd = os.open(
            archive_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        return "EPISODE_SOURCE_ARCHIVE_ROOT_INVALID"
    changed = False
    directory_locked = False
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            directory_locked = True
        except BlockingIOError:
            return "EPISODE_SOURCE_ARCHIVE_LOCK_BUSY"
        for path in archive_dir.iterdir():
            if path.name == ".owner.json":
                continue
            match = re.fullmatch(r"([0-9a-f]{64})\.json\.gz", path.name)
            if match is None:
                return "EPISODE_SOURCE_ARCHIVE_UNEXPECTED_ENTRY"
            digest = match.group(1)
            try:
                entry_stat = path.lstat()
            except OSError:
                return "EPISODE_SOURCE_ARCHIVE_SCAN_FAILED"
            if digest in referenced_digests:
                if not stat.S_ISREG(entry_stat.st_mode):
                    return "EPISODE_SOURCE_ARCHIVE_ENTRY_INVALID"
                continue
            if not (
                stat.S_ISREG(entry_stat.st_mode)
                or stat.S_ISLNK(entry_stat.st_mode)
            ):
                return "EPISODE_SOURCE_ARCHIVE_ENTRY_INVALID"
            try:
                os.unlink(path.name, dir_fd=directory_fd)
            except OSError:
                return "EPISODE_SOURCE_ARCHIVE_ORPHAN_CLEANUP_FAILED"
            changed = True
        if changed:
            os.fsync(directory_fd)
        return None
    finally:
        if directory_locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _delete_archive_if_unreferenced(
    path: Path,
    *,
    referenced_digests: set[str],
) -> str | None:
    match = re.fullmatch(r"([0-9a-f]{64})\.json\.gz", path.name)
    if match is None or match.group(1) in referenced_digests:
        return None
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
            _fsync_directory(path.parent)
    except OSError:
        return "EPISODE_SOURCE_ARCHIVE_ORPHAN_CLEANUP_FAILED"
    return None


def _unlink_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_bounded_regular_json(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[object | None, str | None]:
    """Read local metadata without following links or blocking on a FIFO."""

    try:
        initial = path.lstat()
    except FileNotFoundError:
        return None, "MISSING"
    except OSError:
        return None, "READ_FAILED"
    if stat.S_ISLNK(initial.st_mode):
        return None, "SYMLINK"
    if not stat.S_ISREG(initial.st_mode):
        return None, "NOT_REGULAR"
    if initial.st_size < 0 or initial.st_size > max_bytes:
        return None, "TOO_LARGE"

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != initial.st_dev
            or before.st_ino != initial.st_ino
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            return None, "CHANGED"
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError:
        return None, "READ_FAILED"
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(raw) > max_bytes:
        return None, "TOO_LARGE"
    if (
        len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        return None, "CHANGED"
    try:
        return (
            json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object),
            None,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, "JSON_INVALID"


def _vote_valid(value: Mapping[str, Any]) -> bool:
    if set(value) != _VOTE_FIELDS:
        return False
    direction_score = value.get("direction_score")
    if (
        isinstance(direction_score, bool)
        or not isinstance(direction_score, int)
        or direction_score not in {-1, 0, 1}
        or not isinstance(value.get("evidence_complete"), bool)
    ):
        return False
    for field in _VOTE_FIELDS - {"direction_score", "evidence_complete"}:
        item = value.get(field)
        if (
            not isinstance(item, str)
            or not item
            or item != item.strip().upper()
            or len(item) > 64
        ):
            return False
    return True


def _generated_on_utc_date(
    events: Sequence[Mapping[str, Any]],
    target: object,
) -> int:
    return sum(
        1
        for event in events
        for generated in (_parse_utc(event.get("generated_at_utc")),)
        if generated is not None and generated.date() == target
    )


def _latest_by_episode(events: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for event in events:
        episode_id = str(event.get("episode_id") or "")
        current = latest.get(episode_id)
        if current is None or int(event.get("event_seq") or 0) > int(current.get("event_seq") or 0):
            latest[episode_id] = event
    return latest


def _sealed_contract_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = value.get("contract_sha256")
    return _sha_text(stored) and stored == _canonical_sha(_without(value, "contract_sha256"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = _without(value, "contract_sha256")
    return {**body, "contract_sha256": _canonical_sha(body)}


def _write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    max_bytes: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("episode state path must not be a symlink")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if max_bytes is not None and len(raw.encode("utf-8")) > max_bytes:
        raise ValueError("episode metadata exceeds its byte cap")
    with temp.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {str(name): item for name, item in value.items() if name != key}


def _canonical_sha(value: object) -> str:
    try:
        raw = _canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError):
        raw = b"INVALID"
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_PRICE_ABS:
        return None
    return parsed


def _canonical_timestamp(value: object) -> str | None:
    parsed = _parse_utc(value)
    return parsed.isoformat().replace("+00:00", "Z") if parsed is not None else None


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "EPISODE_EVENT_CONTRACT",
    "EPISODE_STATE_CONTRACT",
    "run_fast_bot_episode_shadow",
    "verify_episode_event",
    "verify_episode_ledger",
]
