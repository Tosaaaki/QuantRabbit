#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from quant_rabbit.guardian_events import (
    DEFAULT_FRESH_ACTION_THROTTLE_SECONDS,
    DEFAULT_THROTTLE_SECONDS,
    GUARDIAN_ACTIONS,
    SEVERITY_RANK,
    GuardianEvent,
    review_guardian_action_receipt,
)
from quant_rabbit.guardian_tuning_evaluator import (
    EVALUATOR_NAME as TUNING_EVALUATOR_NAME,
    METRIC_NAMES as TUNING_EVALUATOR_METRIC_NAMES,
    OBJECTIVE as TUNING_EVALUATOR_OBJECTIVE,
    PRIMARY_METRIC as TUNING_EVALUATOR_PRIMARY_METRIC,
    SUPPORTED_THRESHOLD_PARAMETERS,
    evaluate_precommitted_threshold_cohort,
    FIXED_ACCEPTANCE_THRESHOLD as TUNING_FIXED_ACCEPTANCE_THRESHOLD,
    source_semantic_digest as tuning_source_semantic_digest,
    validate_source as validate_tuning_source,
)
from quant_rabbit.guardian_tuning_cohort import (
    current_canonical_forward_source_tip,
    current_execution_ledger_anchor,
    validate_canonical_forward_cohort,
)
from quant_rabbit.guardian_tuning_monitor import (
    validate_post_activation_monitor_evidence,
)
from quant_rabbit.guardian_tuning_overrides import (
    apply_accepted_override,
    confirm_accepted_override,
    confirm_post_activation_monitor,
    read_stored_override_record,
    read_validated_kept_predecessor_record,
    runtime_forecast_floor_binding,
    write_terminal_commitment_manifest,
)


MODEL = "gpt-5.5"
ACTIONABLE_ACTIONS = {"TRADE", "ADD", "HARVEST", "REDUCE", "CANCEL_PENDING"}
THESIS_STATES = {"ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY"}
DEFAULT_LIVE_ROOT = Path("/Users/tossaki/App/QuantRabbit-live")
DEFAULT_CODEX_APP_BIN = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
LEGACY_CODEX_APP_BIN = Path("/Applications/Codex.app/Contents/Resources/codex")

# Local compatibility floor for GPT-5.5 support: codex-cli 0.25.0 returns
# "gpt-5.5 model requires a newer version of Codex", while 0.142.4 accepts the
# model. This is a CLI compatibility guard, not market/risk logic.
MIN_GPT55_CODEX_VERSION = (0, 142, 0)
CODEX_MODEL_FAILURE_STATUSES = {"CODEX_MODEL_UNSUPPORTED", "CODEX_CLI_VERSION_UNSUPPORTED"}
CODEX_RUNTIME_FAILURE_STATUSES = {
    "CODEX_TIMEOUT",
    "CODEX_AUTH_OR_SANDBOX_FAILURE",
    "CODEX_USAGE_LIMIT",
    *CODEX_MODEL_FAILURE_STATUSES,
}
CODEX_DIRECT_RESULT_STATUSES = {"CODEX_USAGE_LIMIT", *CODEX_MODEL_FAILURE_STATUSES}

# Guardian wake runs every 30s from a router that just refreshed broker truth;
# older snapshots are likely from a previous operating window and should wait
# for the normal trader refresh instead of prompting GPT from stale quotes.
BROKER_SNAPSHOT_MAX_AGE_SECONDS = 5 * 60

# Codex wake should finish well inside one event-driven review window. A hung
# local CLI must not hold repeated launchd invocations open indefinitely.
CODEX_TIMEOUT_SECONDS = 8 * 60

# Receipt lifecycle follows the hourly trader cadence with sync/launchd grace.
# It is a runtime artifact retention window, not market or risk geometry.
DEFAULT_RECEIPT_TTL_SECONDS = 75 * 60
TERMINAL_RECEIPT_LIFECYCLES = {"CONSUMED", "SUPERSEDED", "EXPIRED", "REJECTED"}

# Failed wake attempts are not reviews. Keep their retry lifecycle separate so
# a repaired Codex binary or a materially changed event can be retried without
# erasing the last accepted review baseline. The cap prevents a malformed GPT
# response from consuming every 30-second launchd tick forever; the TTL opens a
# fresh bounded series for an event that still needs hourly-trader attention.
DEFAULT_RETRY_BASE_SECONDS = 60
DEFAULT_RETRY_MAX_SECONDS = 15 * 60
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_TTL_SECONDS = 30 * 60
DEFAULT_USAGE_LIMIT_RETRY_SECONDS = 90 * 60
DEFAULT_PENDING_DISPATCH_TTL_SECONDS = 2 * 60 * 60
MAX_PENDING_DISPATCHES = 24
MAX_PENDING_SUCCESSORS_PER_DEDUPE = 8
MAX_PENDING_TUNING_WORK_ORDERS = 20
TUNING_REVIEW_BATCH_ITEM_FIELDS = frozenset(
    {"work_order_id", "expected_observation_id", "review"}
)
MAX_TUNING_OBSERVATIONS_PER_WORK_ORDER = 24
MAX_TUNING_TERMINAL_HISTORY = 32
# These are audit-retention caps for superseded experiment contracts.  They
# preserve multiple hourly attempts without letting lifecycle metadata grow
# forever; the monotonic semantic-digest registry remains the no-repeat truth.
MAX_TUNING_STALE_PREPARED_CONTRACTS = 12
MAX_TUNING_ABORTED_EXPERIMENT_CONTRACTS = 24
MAX_TUNING_EXPERIMENT_DIGEST_HISTORY = 10_000
MAX_TUNING_EXPERIMENT_ID_DIGEST_HISTORY = 10_000
MAX_TUNING_OVERRIDE_LIFECYCLE_HEADS = 100
TUNING_QUEUE_SCHEMA_REVISION = 4
# Queue state is a bounded coordination artifact, not an evidence store.  Four
# MiB leaves ample room for every bounded work-order observation while stopping
# a corrupt or hostile file before json.loads allocates an unbounded object.
MAX_TUNING_QUEUE_BYTES = 4 * 1024 * 1024
# The real queue schema is far shallower than 32 levels and stores evidence by
# hash reference.  A 64-K-character per-string ceiling still permits detailed reviews
# while preventing one nested field from consuming the entire queue budget.
MAX_TUNING_QUEUE_JSON_DEPTH = 32
MAX_TUNING_QUEUE_STRING_CHARS = 64 * 1024
MAX_TUNING_SOURCE_BYTES = 4 * 1024 * 1024
MAX_TUNING_EVALUATOR_BYTES = 128 * 1024
MAX_TUNING_RUN_BYTES = 128 * 1024
MAX_TUNING_EVIDENCE_BYTES = 128 * 1024
TUNING_EVALUATOR_RUNNER = "guardian_tuning_evidence_builder_v1"
DEFAULT_DISK_P0_FREE_BYTES = 2 * 1024**3
DEFAULT_DISK_WARNING_FREE_BYTES = 5 * 1024**3

_TUNING_HANDOFF_FAILURE_STATUSES = {
    "WORK_ORDER_READ_FAILED",
    "WORK_ORDER_CONCURRENT_UPDATE",
    "WORK_ORDER_WRITE_FAILED",
    "WORK_ORDER_QUEUE_FULL",
    "STRUCTURED_REVIEW_REQUIRED",
}
_TUNING_BOT_FAMILIES = {
    "trend",
    "mean_reversion",
    "breakout",
    "forecast",
    "execution",
}
_TUNING_ADJUSTMENT_FIELDS = {
    "pair",
    "lane_id",
    "bot_family",
    "parameter",
    "current_value",
    "candidate_value",
    "rationale",
}
_TUNING_ACQUISITION_FIELDS = {
    "action_kind",
    "source_ref",
    "required_new_samples",
    "success_condition",
}
_TUNING_ACQUISITION_ACTION_KINDS = {
    "ADD_PREENTRY_SIGNAL_LOG",
    "BUILD_BID_ASK_REPLAY",
    "COLLECT_FORWARD_ENTRIES",
    "REFRESH_CLOSED_CANDLES",
    "RESOLVE_ATTRIBUTED_OUTCOMES",
}
_TUNING_VAGUE_ACQUISITION_TEXT = {
    "collect more",
    "collect more evidence",
    "later",
    "monitor",
    "observe",
    "tbd",
    "wait",
}
_TUNING_VAGUE_ACQUISITION_PATTERNS = (
    re.compile(r"\b(?:wait|monitor|later|eventually)\b", re.IGNORECASE),
    re.compile(r"\bwhen\s+enough\b", re.IGNORECASE),
)
_TUNING_FORBIDDEN_PARAMETER_TOKENS = {
    "oanda",
    "gateway",
    "permission",
    "live_permission",
    "direct_order",
    "send_order",
    "cancel_order",
    "close_position",
    "risk_gate",
    "blocker",
    "ownership",
    "margin_gate",
    "exposure_gate",
    "risk",
    "max_loss",
    "loss_cap",
    "lot",
    "units",
    "leverage",
    "position_size",
    "size_multiple",
    "stop_loss",
    "take_profit",
    "sl_distance",
    "tp_distance",
    "allocation",
    "risk_fraction",
    "risk_pct",
    "capital_fraction",
    "notional",
    "margin",
    "exposure",
    "quantity",
    "weight",
    "budget",
    "share",
    "portfolio",
    "trade_amount",
    "account",
    "capital",
    "cash",
    "equity",
    "balance",
    "nav",
    "fund",
    "amount",
    "broker",
    "close",
    "drawdown",
    "execute",
    "fill",
    "limit",
    "liquidation",
    "loss",
    "money",
    "open_position",
    "order",
    "pnl",
    "position",
    "profit",
    "return",
    "stop",
    "stopout",
    "submit",
    "trade",
    "wealth",
}
_TUNING_COMMON_SAFE_PARAMETERS = {
    "confirmation_bars",
    "entry_cooldown_bars",
    "entry_score_floor",
    "quality_score_floor",
    "signal_confidence_floor",
    "signal_confirmation_bars",
    "signal_score_floor",
}
_TUNING_SAFE_PARAMETERS_BY_FAMILY = {
    "trend": {
        "adx_score_floor",
        "ema_slope_score_floor",
        "momentum_score_floor",
        "trend_lookback_bars",
        "trend_score_floor",
    },
    "mean_reversion": {
        "band_score_weight",
        "range_lookback_bars",
        "range_score_floor",
        "rsi_score_floor",
        "zscore_score_floor",
    },
    "breakout": {
        "atr_score_floor",
        "breakout_confirmation_bars",
        "breakout_lookback_bars",
        "breakout_score_floor",
        "volatility_score_floor",
    },
    "forecast": {
        "forecast_confidence_floor",
        "forecast_lookback_bars",
        "forecast_score_floor",
        "probability_score_floor",
    },
    "execution": {
        "execution_cost_score_weight",
        "latency_score_weight",
        "slippage_score_weight",
        "spread_score_weight",
        "staleness_score_weight",
    },
}

_EXPLICIT_DISPATCH_REASONS = {
    "NEW_EVENT",
    "SEVERITY_INCREASE",
    "UNKNOWN_GATEWAY_OUTSIDE_ORDER",
    "UNKNOWN_GATEWAY_OUTSIDE_ORDER_STATE_CHANGE",
    "MARGIN_RISK_THRESHOLD_CROSSED",
    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
    "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
    "PRICE_ENTERED_HARVEST_ZONE",
    "FAILED_DISPATCH_RETRY",
}
_MATERIAL_CHANGE_REASONS = {
    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
    "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
    "TECHNICAL_STATE_CHANGE",
    "REGIME_STATE_CHANGE",
    "VOLATILITY_BUCKET_CHANGE",
    "TECHNICAL_FAMILY_STATE_CHANGE",
    "CLOSED_CANDLE_STRUCTURE_CHANGE",
}
_ACTIVE_EXPOSURE_EVENT_TYPES = {
    "HARVEST_ZONE",
    "THESIS_INVALIDATION",
    "MARGIN_PRESSURE",
    "UNKNOWN_ORDER",
    "UNEXPECTED_PROTECTION_MISSING",
    "CONTRACT_HARVEST_TRIGGER",
    "CONTRACT_WOUNDED_TRIGGER",
    "CONTRACT_INVALIDATION_TRIGGER",
    "CONTRACT_EMERGENCY_TRIGGER",
}


@dataclass(frozen=True)
class DispatcherPaths:
    root: Path
    escalation: Path
    events: Path
    event_state: Path
    broker_snapshot: Path
    daily_target_state: Path
    event_report: Path
    prompt_template: Path
    action_receipt: Path
    action_review: Path
    tuning_work_order: Path
    dispatcher_state: Path
    log: Path
    codex_home: Path
    codex_output: Path
    codex_explicit_output: Path
    live_root: Path
    live_lock: Path

    @classmethod
    def from_root(cls, root: Path, *, live_root: Path | None = None) -> "DispatcherPaths":
        live = live_root or Path(os.environ.get("QR_GUARDIAN_WAKE_LIVE_ROOT", str(DEFAULT_LIVE_ROOT)))
        return cls(
            root=root,
            escalation=root / "data" / "guardian_escalation.json",
            events=root / "data" / "guardian_events.json",
            event_state=root / "data" / "guardian_event_state.json",
            broker_snapshot=root / "data" / "broker_snapshot.json",
            daily_target_state=root / "data" / "daily_target_state.json",
            event_report=root / "docs" / "guardian_event_report.md",
            prompt_template=root / "docs" / "guardian_wake_prompt.md",
            action_receipt=root / "data" / "guardian_action_receipt.json",
            action_review=root / "docs" / "guardian_action_review.md",
            tuning_work_order=root / "data" / "guardian_tuning_work_order.json",
            dispatcher_state=root / "data" / "guardian_wake_dispatcher_state.json",
            log=root / "logs" / "guardian_wake_dispatcher.log",
            codex_home=root / "data" / "codex_guardian_home",
            codex_output=root / "data" / "guardian_wake_codex_last_message.md",
            codex_explicit_output=root / "data" / "guardian_wake_codex_explicit_receipt.json",
            live_root=live,
            live_lock=live / ".quant_rabbit_live.lock",
        )


def run_dispatcher(
    *,
    paths: DispatcherPaths,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
    snapshot_refresh_run: Callable[..., Any] | None = None,
    codex_preflight_run: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one dispatcher exclusively across prompt, Codex, and artifact I/O."""

    clock = _utc(now)
    lock_path = paths.root / "logs" / ".guardian_wake_dispatcher.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
    except OSError as exc:
        return {
            "generated_at_utc": clock.isoformat(),
            "model": MODEL,
            "status": "DISPATCHER_LOCK_ERROR",
            "wake_gpt": False,
            "receipt_written": False,
            "action_receipt_path": None,
            "lock": {"path": str(lock_path), "error": f"{type(exc).__name__}: {exc}"},
        }
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "generated_at_utc": clock.isoformat(),
                "model": MODEL,
                "status": "DISPATCHER_LOCKED",
                "wake_gpt": False,
                "receipt_written": False,
                "action_receipt_path": None,
                "lock": {
                    "path": str(lock_path),
                    "status": "HELD_BY_OTHER_DISPATCHER",
                    "contender_pid": os.getpid(),
                },
            }
        return _run_dispatcher_once(
            paths=paths,
            now=clock,
            env=env,
            subprocess_run=subprocess_run,
            snapshot_refresh_run=snapshot_refresh_run,
            codex_preflight_run=codex_preflight_run,
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_handle.close()


def _run_dispatcher_once(
    *,
    paths: DispatcherPaths,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
    snapshot_refresh_run: Callable[..., Any] | None = None,
    codex_preflight_run: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    clock = _utc(now)
    environ = env if env is not None else os.environ
    paths.log.parent.mkdir(parents=True, exist_ok=True)

    escalation = _load_json(paths.escalation)
    events_payload = _load_json(paths.events)
    event_state = _load_json(paths.event_state)
    broker_snapshot = _load_json(paths.broker_snapshot)
    daily_target_state = _load_json(paths.daily_target_state)
    event_report = _read_text(paths.event_report)
    dispatcher_state = _load_json(paths.dispatcher_state)
    dispatcher_state, usage_limit_reclassification = _reclassify_legacy_usage_limit_attempts(
        dispatcher_state,
        now=clock,
        env=environ,
    )
    dispatcher_state, usage_limit_recovery = _release_recovered_usage_limit_backoffs(
        dispatcher_state,
        now=clock,
    )
    dispatcher_state, tuning_queue_recovery = _release_recovered_tuning_queue_backoffs(
        dispatcher_state,
        tuning_work_order=_load_tuning_work_order(paths.tuning_work_order),
        now=clock,
    )
    runtime_disk = _runtime_disk_state(paths.root, environ)

    escalation, events_payload, pending_injection = _inject_pending_dispatches(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
    )
    # Capture a newer same-dedupe observation as a durable successor before a
    # due retry replaces the transient router row with the immutable failed
    # observation.  The retry injection must never erase that successor.
    escalation, events_payload, retry_injection = _inject_due_failed_attempts(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
        env=environ,
    )

    result_base = {
        "generated_at_utc": clock.isoformat(),
        "model": MODEL,
        "paths": _paths_payload(paths),
        "execution_boundary": _execution_boundary(),
        "runtime_disk": runtime_disk,
    }
    if retry_injection:
        result_base["retry_injection"] = retry_injection
    if pending_injection:
        result_base["pending_injection"] = pending_injection
    if usage_limit_reclassification:
        result_base["usage_limit_reclassification"] = usage_limit_reclassification
    if usage_limit_recovery:
        result_base["usage_limit_recovery"] = usage_limit_recovery
    if tuning_queue_recovery:
        result_base["tuning_queue_recovery"] = tuning_queue_recovery
    # Expiry archival writes a new JSON file. Under P0 disk pressure preserve
    # the current receipt as-is and let the first recovered cycle expire it;
    # the disk guard must run before any optional archive/session growth.
    if runtime_disk["status"] != "RUNTIME_DISK_P0":
        _expire_current_receipt_if_needed(paths.action_receipt, now=clock)

    if escalation.get("wake_gpt") is not True:
        result = {
            **result_base,
            "status": "NO_WAKE",
            "wake_gpt": False,
            "reason": "guardian_escalation wake_gpt is not true",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    selected, selection = _select_dispatch_event(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
        env=environ,
    )
    if selected is None:
        result = {
            **result_base,
            "status": "SUPPRESSED",
            "wake_gpt": True,
            "selection": selection,
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    if runtime_disk["status"] == "RUNTIME_DISK_P0":
        result = {
            **result_base,
            "status": "RUNTIME_DISK_P0",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "queued_for_active_trader": True,
            "queue_reason": "runtime free space is below the 2 GiB GPT wake floor",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="runtime free space is below the GPT wake floor",
        )
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result, attempted_event=selected, env=environ)
        _append_log(paths.log, result)
        return result

    lock = _active_live_lock(paths.live_lock)
    if lock["active"]:
        queued = {
            **result_base,
            "status": "QUEUED_FOR_ACTIVE_TRADER",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "lock": lock,
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(paths.escalation, escalation, queued, now=clock, reason="active qr-trader/live gateway lock")
        _write_action_review(paths.action_review, queued, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, queued)
        _append_log(paths.log, queued)
        return queued

    freshness = _broker_snapshot_freshness(broker_snapshot, now=clock, env=environ)
    if not freshness["fresh"]:
        refresh = _refresh_broker_snapshot(
            paths=paths,
            env=environ,
            subprocess_run=snapshot_refresh_run or subprocess.run,
        )
        if refresh["status"] == "REFRESHED":
            broker_snapshot = _load_json(paths.broker_snapshot)
            freshness = _broker_snapshot_freshness(broker_snapshot, now=clock, env=environ)
            freshness["refresh_attempt"] = refresh
        else:
            freshness["refresh_attempt"] = refresh
    if not freshness["fresh"]:
        result = {
            **result_base,
            "status": "BROKER_SNAPSHOT_STALE",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "broker_snapshot_freshness": freshness,
            "queued_for_active_trader": True,
            "queue_reason": "stale broker snapshot; GPT wake not started because stale broker truth must not enter the prompt",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="stale broker snapshot; GPT wake not started",
        )
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    codex_preflight = _run_codex_preflight(
        env=environ,
        subprocess_run=codex_preflight_run or subprocess.run,
    )
    if codex_preflight.get("enabled") and str(codex_preflight.get("status") or "").upper() in CODEX_MODEL_FAILURE_STATUSES:
        result = {
            **result_base,
            "status": codex_preflight["status"],
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "broker_snapshot_freshness": freshness,
            "codex_preflight": codex_preflight,
            "parse": {
                "valid": False,
                "error": codex_preflight["status"],
                "raw_output_excerpt": str(
                    codex_preflight.get("stderr_excerpt") or codex_preflight.get("stdout_excerpt") or ""
                )[:2000],
            },
            "receipt_written": False,
            "action_receipt_path": None,
            "queued_for_active_trader": True,
            "queue_reason": codex_preflight.get("remediation_hint")
            or "Codex CLI/model compatibility failed before guardian wake",
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="Codex CLI/model compatibility failed before guardian wake",
        )
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result, attempted_event=selected, env=environ)
        _append_log(paths.log, result)
        return result

    _prepare_codex_home(paths.codex_home, live_root=paths.live_root)
    prompt = _build_prompt(
        paths=paths,
        selected_event=selected,
        escalation=escalation,
        events_payload=events_payload,
        event_state=event_state,
        broker_snapshot=broker_snapshot,
        daily_target_state=daily_target_state,
        event_report=event_report,
        dispatcher_state=dispatcher_state,
        runtime_disk=runtime_disk,
    )
    codex = _run_codex(
        paths=paths,
        prompt=prompt,
        env=environ,
        subprocess_run=subprocess_run,
        attempt="initial",
        codex_preflight=codex_preflight,
    )

    parsed = _parse_codex_result(codex)
    codex_attempts = [codex]
    repair_attempted = False
    repair_skipped: dict[str, Any] | None = None
    if not parsed["valid"] and parsed.get("error") not in {
        *CODEX_RUNTIME_FAILURE_STATUSES,
    }:
        retry_lock = _active_live_lock(paths.live_lock)
        if retry_lock["active"]:
            queued = {
                **result_base,
                "status": "QUEUED_FOR_ACTIVE_TRADER",
                "wake_gpt": True,
                "selected_event": selected,
                "selection": selection,
                "broker_snapshot_freshness": freshness,
                "codex": codex,
                "codex_attempts": codex_attempts,
                "parse": parsed,
                "receipt_written": False,
                "action_receipt_path": None,
                "lock": retry_lock,
                "queue_reason": "active qr-trader/live gateway lock before parse repair retry",
            }
            _mark_escalation_queued(
                paths.escalation,
                escalation,
                queued,
                now=clock,
                reason="active qr-trader/live gateway lock before parse repair retry",
            )
            _write_action_review(paths.action_review, queued, action_receipt_path=paths.action_receipt, now=clock)
            _record_state(paths.dispatcher_state, dispatcher_state, queued, attempted_event=selected, env=environ)
            _append_log(paths.log, queued)
            return queued
        repair_attempted = True
        repair_prompt = (
            prompt.rstrip()
            + "\n\nReturn only the required JSON object. No markdown. No prose.\n"
        )
        repair_codex = _run_codex(
            paths=paths,
            prompt=repair_prompt,
            env=environ,
            subprocess_run=subprocess_run,
            attempt="repair",
            codex_preflight=codex_preflight,
        )
        codex_attempts.append(repair_codex)
        parsed = _parse_codex_result(repair_codex)
        codex = repair_codex
    review_events_payload = dict(events_payload)
    review_event_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in review_events_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    review_event_by_key[str(selected.get("dedupe_key") or "")] = {
        key: value for key, value in selected.items() if key != "wake_reason_codes"
    }
    review_events_payload["events"] = list(review_event_by_key.values())
    events = _events_from_payload(review_events_payload)
    review = None
    receipt_written = False
    tuning_handoff: dict[str, Any] = {"status": "SKIPPED_NO_ACCEPTED_MATERIAL_EVENT"}
    if parsed["valid"]:
        if not parsed["receipt"].get("dedupe_key") and selected.get("dedupe_key"):
            parsed["receipt"]["dedupe_key"] = selected["dedupe_key"]
        review = review_guardian_action_receipt(parsed["receipt"], events=events, selected_event=selected, now=clock)
        receipt_action = str(parsed["receipt"].get("action") or "").upper()
        if review.get("status") == "ACCEPTED" and receipt_action in GUARDIAN_ACTIONS:
            tuning_handoff = _maybe_write_tuning_work_order(
                path=paths.tuning_work_order,
                selected_event=selected,
                receipt=parsed["receipt"],
                now=clock,
            )
            tuning_handoff_failed_before_receipt = (
                str(tuning_handoff.get("status") or "").upper()
                in _TUNING_HANDOFF_FAILURE_STATUSES
            )
            if not tuning_handoff_failed_before_receipt:
                superseded = _supersede_current_receipt(
                    paths.action_receipt,
                    superseded_by_event_id=str(selected.get("event_id") or ""),
                    now=clock,
                )
                payload = {
                    **review,
                    "source": "guardian_wake_dispatcher",
                    "model": MODEL,
                    "receipt_status": "ACCEPTED",
                    "receipt_lifecycle": "ACTIVE",
                    "expires_at_utc": _receipt_expires_at(clock, env=environ),
                    "consumed_by_trader": False,
                    "superseded_by_event_id": None,
                    "no_direct_oanda": True,
                    "selected_event": selected,
                    "selected_event_id": selected.get("event_id"),
                    "selected_event_dedupe_key": selected.get("dedupe_key"),
                    "dispatcher_status": "RECEIPT_WRITTEN",
                    "receipt_id": parsed["receipt"].get("receipt_id") or review.get("generated_at_utc"),
                }
                if superseded:
                    payload["superseded_previous_receipt"] = superseded
                _write_json(paths.action_receipt, payload)
                _archive_receipt(paths.action_receipt, payload)
                receipt_written = True
        else:
            _remove_non_accepted_current_receipt(paths.action_receipt)
    else:
        _remove_non_accepted_current_receipt(paths.action_receipt)

    tuning_handoff_failed = (
        str(tuning_handoff.get("status") or "").upper()
        in _TUNING_HANDOFF_FAILURE_STATUSES
    )
    if tuning_handoff_failed:
        # A tuning event is not acknowledged until its durable hourly-AI
        # handoff is complete.  Never start an execution cycle from a receipt
        # that this dispatcher is deliberately retaining for retry.
        handoff = {
            "status": "SKIPPED_TUNING_HANDOFF_FAILED",
            "tuning_handoff_status": tuning_handoff.get("status"),
        }
    else:
        handoff = _maybe_gateway_handoff(
            receipt_review=review,
            paths=paths,
            env=environ,
            lock=lock,
        )
    status = (
        "TUNING_HANDOFF_FAILED"
        if tuning_handoff_failed
        else "RECEIPT_WRITTEN"
        if receipt_written
        else parsed["error"]
        if not parsed["valid"] and parsed.get("error") in CODEX_DIRECT_RESULT_STATUSES
        else "PARSE_FAILED"
        if not parsed["valid"]
        else "RECEIPT_EVENT_MISMATCH"
        if review is not None and review.get("status") == "RECEIPT_EVENT_MISMATCH"
        else "RECEIPT_REJECTED"
    )
    if repair_attempted and not parsed["valid"]:
        repair_skipped = {"status": "FAILED_AFTER_ONE_RETRY", "max_retries_per_event": 1}
    result = {
        **result_base,
        "status": status,
        "wake_gpt": True,
        "selected_event": selected,
        "selection": selection,
        "broker_snapshot_freshness": freshness,
        "codex_preflight": codex_preflight,
        "codex": codex,
        "codex_attempts": codex_attempts,
        "parse": parsed,
        "repair_attempted": repair_attempted,
        "repair_result": repair_skipped,
        "receipt_written": receipt_written,
        "action_receipt_path": str(paths.action_receipt) if receipt_written else None,
        "gateway_handoff": handoff,
        "tuning_handoff": tuning_handoff,
        "tuning_handoff_failed": tuning_handoff_failed,
    }
    if review is not None:
        result["receipt_review"] = review
    if status == "RECEIPT_EVENT_MISMATCH":
        result["queued_for_active_trader"] = True
        result["queue_reason"] = "guardian wake receipt did not match selected_event"
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="guardian wake receipt did not match selected_event",
        )
    _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
    _record_state(
        paths.dispatcher_state,
        dispatcher_state,
        result,
        reviewed_event=selected if receipt_written and not tuning_handoff_failed else None,
        attempted_event=selected,
        env=environ,
    )
    _append_log(paths.log, result)
    return result


def _runtime_disk_state(root: Path, env: dict[str, str]) -> dict[str, Any]:
    p0_floor = max(1, int(env.get("QR_GUARDIAN_WAKE_DISK_P0_FREE_BYTES", DEFAULT_DISK_P0_FREE_BYTES)))
    warning_floor = max(
        p0_floor,
        int(env.get("QR_GUARDIAN_WAKE_DISK_WARNING_FREE_BYTES", DEFAULT_DISK_WARNING_FREE_BYTES)),
    )
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return {
            "status": "RUNTIME_DISK_UNKNOWN",
            "path": str(root),
            "p0_free_bytes": p0_floor,
            "warning_free_bytes": warning_floor,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if usage.free < p0_floor:
        status = "RUNTIME_DISK_P0"
    elif usage.free < warning_floor:
        status = "RUNTIME_DISK_WARNING"
    else:
        status = "RUNTIME_DISK_OK"
    return {
        "status": status,
        "path": str(root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 3),
        "p0_free_bytes": p0_floor,
        "warning_free_bytes": warning_floor,
        "gpt_wake_allowed": status != "RUNTIME_DISK_P0",
    }


def _inject_due_failed_attempts(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    attempts = dispatcher_state.get("dispatch_attempts")
    if not isinstance(attempts, dict) or not attempts:
        return escalation, events_payload, None
    current_events = {
        str(item.get("dedupe_key") or ""): item
        for item in events_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    due: list[dict[str, Any]] = []
    for dedupe_key, record in attempts.items():
        if not isinstance(record, dict):
            continue
        stored_event = record.get("event") if isinstance(record.get("event"), dict) else {}
        # Retry the exact failed observation.  A live quote tick changes the
        # current technical event_id/price_zone, but must not reset backoff or
        # the max-attempt budget.  A genuinely new material router escalation
        # is still merged separately and can start its own reviewed series.
        event = dict(stored_event)
        if not event or not event.get("event_id") or not event.get("dedupe_key"):
            continue
        if _retry_suppression(record, event=event, now=now, env=env) is not None:
            continue
        event["wake_reason_codes"] = list(
            dict.fromkeys([*(event.get("wake_reason_codes") or []), "FAILED_DISPATCH_RETRY"])
        )
        due.append(event)
    if not due:
        return escalation, events_payload, None

    payload = dict(escalation)
    review_events = [item for item in payload.get("events_to_review", []) or [] if isinstance(item, dict)]
    review_by_key = {str(item.get("dedupe_key") or ""): item for item in review_events}
    for event in due:
        review_by_key[str(event.get("dedupe_key") or "")] = event
    payload["wake_gpt"] = True
    payload["events_to_review"] = list(review_by_key.values())
    payload["wake_reason_codes"] = sorted(
        {*(str(item) for item in payload.get("wake_reason_codes", []) or []), "FAILED_DISPATCH_RETRY"}
    )

    event_payload = dict(events_payload)
    event_by_key = dict(current_events)
    for event in due:
        event_by_key[str(event.get("dedupe_key") or "")] = {
            key: value for key, value in event.items() if key != "wake_reason_codes"
        }
    event_payload["events"] = list(event_by_key.values())
    return payload, event_payload, {
        "status": "DUE_FAILED_ATTEMPTS_INJECTED",
        "event_ids": [event.get("event_id") for event in due],
        "count": len(due),
    }


def _reclassify_legacy_usage_limit_attempts(
    dispatcher_state: dict[str, Any],
    *,
    now: datetime,
    env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state = dict(dispatcher_state) if isinstance(dispatcher_state, dict) else {}
    last_result = state.get("last_result") if isinstance(state.get("last_result"), dict) else {}
    parse = last_result.get("parse") if isinstance(last_result.get("parse"), dict) else {}
    receipt = parse.get("receipt") if isinstance(parse.get("receipt"), dict) else {}
    evidence = "\n".join(
        str(value or "")
        for value in (
            receipt.get("message"),
            parse.get("raw_output_excerpt"),
            (last_result.get("codex") or {}).get("raw_stdout_excerpt")
            if isinstance(last_result.get("codex"), dict)
            else None,
        )
    )
    source_generated_at = str(last_result.get("generated_at_utc") or "")
    if (
        not source_generated_at
        or state.get("usage_limit_reclassified_source_at") == source_generated_at
        or _codex_usage_limit_failure(
            stderr_text="",
            stdout_text=evidence,
            last_message="",
        )
        is None
    ):
        return state, None

    attempts = state.get("dispatch_attempts") if isinstance(state.get("dispatch_attempts"), dict) else {}
    if not attempts or all(
        isinstance(record, dict) and str(record.get("last_error") or "") == "CODEX_USAGE_LIMIT"
        for record in attempts.values()
    ):
        state["usage_limit_reclassified_source_at"] = source_generated_at
        return state, None

    retry_seconds = max(
        60,
        int(
            env.get(
                "QR_GUARDIAN_WAKE_USAGE_LIMIT_RETRY_SECONDS",
                DEFAULT_USAGE_LIMIT_RETRY_SECONDS,
            )
        ),
    )
    retry_after = now + timedelta(seconds=retry_seconds)
    expires_at = retry_after + timedelta(seconds=retry_seconds)
    changed: list[str] = []
    normalized: dict[str, Any] = {}
    for dedupe_key, value in attempts.items():
        if not isinstance(value, dict):
            normalized[str(dedupe_key)] = value
            continue
        record = dict(value)
        record["last_error"] = "CODEX_USAGE_LIMIT"
        record["last_status"] = "CODEX_USAGE_LIMIT"
        record["retry_after_utc"] = retry_after.isoformat()
        previous_expiry = _parse_utc(record.get("expires_at_utc"))
        record["expires_at_utc"] = max(previous_expiry or expires_at, expires_at).isoformat()
        record["usage_limit_retry_seconds"] = retry_seconds
        record["legacy_usage_limit_reclassified_at_utc"] = now.isoformat()
        normalized[str(dedupe_key)] = record
        changed.append(str(dedupe_key))
    state["dispatch_attempts"] = normalized
    state["usage_limit_reclassified_source_at"] = source_generated_at
    state["usage_limit_reclassified_at_utc"] = now.isoformat()
    return state, {
        "status": "LEGACY_USAGE_LIMIT_RECLASSIFIED",
        "source_generated_at_utc": source_generated_at,
        "retry_after_utc": retry_after.isoformat(),
        "attempt_count": len(changed),
        "dedupe_keys": changed,
    }


def _release_recovered_usage_limit_backoffs(
    previous_state: dict[str, Any],
    *,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Release only quota backoffs proven obsolete by a later accepted wake.

    Codex account capacity is shared across queued guardian events.  Once one
    accepted receipt is written after a CODEX_USAGE_LIMIT failure, retaining a
    separate 90-minute backoff on every sibling event delays market adaptation
    despite direct evidence that capacity recovered.  Ordinary parse/runtime
    failures are intentionally untouched.
    """

    state = dict(previous_state) if isinstance(previous_state, dict) else {}
    reviewed = _accepted_review_records(state.get("reviewed_events"))
    recovered_at: datetime | None = None
    recovery_source: str | None = None
    for dedupe_key, record in reviewed.items():
        if str(record.get("last_status") or "").upper() not in {
            "RECEIPT_WRITTEN",
            "TUNING_HANDOFF_FAILED",
        }:
            continue
        reviewed_at = _parse_utc(record.get("last_reviewed_at_utc"))
        if reviewed_at is None or (recovered_at is not None and reviewed_at <= recovered_at):
            continue
        recovered_at = reviewed_at
        recovery_source = dedupe_key
    if recovered_at is None:
        return state, None

    attempts = state.get("dispatch_attempts")
    if not isinstance(attempts, dict) or not attempts:
        return state, None
    released: list[str] = []
    normalized: dict[str, Any] = {}
    for dedupe_key, value in attempts.items():
        if not isinstance(value, dict):
            normalized[str(dedupe_key)] = value
            continue
        record = dict(value)
        failed_at = _parse_utc(record.get("last_failed_at_utc"))
        quota_failure = str(record.get("last_error") or "").upper() == "CODEX_USAGE_LIMIT"
        already_released_at = _parse_utc(record.get("usage_limit_recovered_at_utc"))
        if not quota_failure or failed_at is None or failed_at >= recovered_at:
            normalized[str(dedupe_key)] = record
            continue
        if (
            str(record.get("last_status") or "").upper() == "CODEX_USAGE_LIMIT_RECOVERED"
            and already_released_at is not None
            and already_released_at >= recovered_at
        ):
            normalized[str(dedupe_key)] = record
            continue
        record["attempt_count"] = 0
        record["retry_after_utc"] = now.isoformat()
        record["retry_budget_exhausted"] = False
        record["last_status"] = "CODEX_USAGE_LIMIT_RECOVERED"
        record["usage_limit_recovered_at_utc"] = recovered_at.isoformat()
        record["usage_limit_recovery_observed_at_utc"] = now.isoformat()
        record["usage_limit_recovery_source_dedupe_key"] = recovery_source
        normalized[str(dedupe_key)] = record
        released.append(str(dedupe_key))
    if not released:
        return state, None

    state["dispatch_attempts"] = normalized
    state["usage_limit_recovered_at_utc"] = recovered_at.isoformat()
    state["usage_limit_recovery_observed_at_utc"] = now.isoformat()
    state["usage_limit_recovery_source_dedupe_key"] = recovery_source
    return state, {
        "status": "CODEX_USAGE_LIMIT_RECOVERED",
        "recovered_at_utc": recovered_at.isoformat(),
        "observed_at_utc": now.isoformat(),
        "source_dedupe_key": recovery_source,
        "released_count": len(released),
        "released_dedupe_keys": released,
    }


def _release_recovered_tuning_queue_backoffs(
    previous_state: dict[str, Any],
    *,
    tuning_work_order: dict[str, Any],
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Release queue-full retries as soon as a durable slot is available."""

    state = dict(previous_state) if isinstance(previous_state, dict) else {}
    if tuning_work_order.get("_read_error"):
        return state, {
            "status": "WORK_ORDER_QUEUE_READ_FAILED",
            "error": tuning_work_order.get("_read_error"),
            "released_count": 0,
        }
    pending, _ = _normalized_tuning_work_order_queue(tuning_work_order)
    if len(pending) >= MAX_PENDING_TUNING_WORK_ORDERS:
        return state, None
    attempts = state.get("dispatch_attempts")
    if not isinstance(attempts, dict) or not attempts:
        return state, None
    released: list[str] = []
    normalized: dict[str, Any] = {}
    for dedupe_key, value in attempts.items():
        if not isinstance(value, dict):
            normalized[str(dedupe_key)] = value
            continue
        record = dict(value)
        if str(record.get("last_error") or "").upper() != "WORK_ORDER_QUEUE_FULL":
            normalized[str(dedupe_key)] = record
            continue
        record["attempt_count"] = 0
        record["retry_after_utc"] = now.isoformat()
        record["retry_budget_exhausted"] = False
        record["last_status"] = "WORK_ORDER_QUEUE_CAPACITY_RECOVERED"
        record["queue_capacity_recovered_at_utc"] = now.isoformat()
        record["queue_pending_count_at_recovery"] = len(pending)
        normalized[str(dedupe_key)] = record
        released.append(str(dedupe_key))
    if not released:
        return state, None
    state["dispatch_attempts"] = normalized
    state["tuning_queue_capacity_recovered_at_utc"] = now.isoformat()
    return state, {
        "status": "WORK_ORDER_QUEUE_CAPACITY_RECOVERED",
        "observed_at_utc": now.isoformat(),
        "pending_count": len(pending),
        "max_pending_count": MAX_PENDING_TUNING_WORK_ORDERS,
        "released_count": len(released),
        "released_dedupe_keys": released,
    }


def _inject_pending_dispatches(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    pending = _active_pending_dispatch_records(
        dispatcher_state.get("pending_dispatches"),
        now=now,
    )
    if not pending:
        return escalation, events_payload, None

    payload = dict(escalation)
    review_events = [
        item for item in payload.get("events_to_review", []) or [] if isinstance(item, dict)
    ]
    review_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in review_events
        if str(item.get("dedupe_key") or "")
    }
    replayed: list[dict[str, Any]] = []
    successor_candidates: list[dict[str, Any]] = []
    for dedupe_key, record in pending.items():
        event = record.get("event") if isinstance(record.get("event"), dict) else {}
        if not event:
            continue
        # A pending row is an immutable observation obligation.  Prefer it
        # even when the transient router payload already contains the same
        # dedupe key.  A materially different current row is retained as a
        # bounded successor and must not reset this row's retry fingerprint or
        # backoff budget.
        current = review_by_key.get(dedupe_key) if isinstance(review_by_key.get(dedupe_key), dict) else {}
        current_fingerprint = _event_material_fingerprint(current) if current else ""
        pending_fingerprint = str(record.get("material_fingerprint") or "")
        if not pending_fingerprint:
            pending_fingerprint = _event_material_fingerprint(event)
        if current and current_fingerprint and current_fingerprint != pending_fingerprint:
            successor_candidates.append(
                {
                    "dedupe_key": dedupe_key,
                    "event": dict(current),
                    "material_fingerprint": current_fingerprint,
                }
            )
        replay_event = dict(event)
        replay_event["wake_reason_codes"] = list(
            dict.fromkeys(
                [
                    *(event.get("wake_reason_codes") or []),
                    *(
                        ["DURABLE_PENDING_SUCCESSOR"]
                        if record.get("promoted_successor") is True
                        else []
                    ),
                ]
            )
        )
        review_by_key[dedupe_key] = replay_event
        replayed.append(replay_event)
    if not replayed:
        return escalation, events_payload, None

    payload["wake_gpt"] = True
    payload["events_to_review"] = list(review_by_key.values())
    payload["wake_reason_codes"] = sorted(
        {
            *(str(item) for item in payload.get("wake_reason_codes", []) or []),
            "DURABLE_PENDING_DISPATCH",
        }
    )
    event_payload = dict(events_payload)
    event_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in event_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    for event in replayed:
        event_by_key[str(event.get("dedupe_key") or "")] = {
            key: value for key, value in event.items() if key != "wake_reason_codes"
        }
    event_payload["events"] = list(event_by_key.values())
    return payload, event_payload, {
        "status": "DURABLE_PENDING_DISPATCHES_INJECTED",
        "event_ids": [event.get("event_id") for event in replayed],
        "count": len(replayed),
        "successor_candidates": successor_candidates,
        "successor_candidate_count": len(successor_candidates),
    }


def _active_pending_successor_records(
    record: dict[str, Any],
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    raw = record.get("successors")
    if not isinstance(raw, list):
        return []
    active: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        fingerprint = str(item.get("material_fingerprint") or "")
        expires_at = _parse_utc(item.get("expires_at_utc"))
        if not event or not fingerprint or fingerprint in seen:
            continue
        if expires_at is None or expires_at <= now:
            continue
        seen.add(fingerprint)
        active.append(dict(item))
    return active[-MAX_PENDING_SUCCESSORS_PER_DEDUPE:]


def _enqueue_pending_dispatch_successor(
    record: dict[str, Any],
    *,
    event: dict[str, Any],
    now: datetime,
    ttl_seconds: int,
) -> tuple[dict[str, Any], bool]:
    """Retain a newer exact observation without replacing the retry head."""

    if not isinstance(event, dict) or not str(event.get("dedupe_key") or "").strip():
        return dict(record), False
    fingerprint = _event_material_fingerprint(event)
    primary_event = record.get("event") if isinstance(record.get("event"), dict) else {}
    primary_fingerprint = str(record.get("material_fingerprint") or "")
    if not primary_fingerprint and primary_event:
        primary_fingerprint = _event_material_fingerprint(primary_event)
    if fingerprint == primary_fingerprint:
        return dict(record), False

    successors = _active_pending_successor_records(record, now=now)
    expires_at = (now + timedelta(seconds=max(60, ttl_seconds))).isoformat()
    for index, successor in enumerate(successors):
        if str(successor.get("material_fingerprint") or "") != fingerprint:
            continue
        refreshed = dict(successor)
        refreshed["last_seen_at_utc"] = now.isoformat()
        refreshed["expires_at_utc"] = expires_at
        successors[index] = refreshed
        updated = dict(record)
        updated["successors"] = successors
        return updated, False

    successors.append(
        {
            "event": dict(event),
            "event_id": event.get("event_id"),
            "material_fingerprint": fingerprint,
            "queued_at_utc": now.isoformat(),
            "last_seen_at_utc": now.isoformat(),
            "expires_at_utc": expires_at,
        }
    )
    overflow = max(0, len(successors) - MAX_PENDING_SUCCESSORS_PER_DEDUPE)
    if overflow:
        successors = successors[-MAX_PENDING_SUCCESSORS_PER_DEDUPE:]
    updated = dict(record)
    updated["successors"] = successors
    if overflow:
        updated["successor_overflow_count"] = int(
            record.get("successor_overflow_count") or 0
        ) + overflow
    return updated, True


def _promote_pending_dispatch_successor(
    record: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | None:
    successors = _active_pending_successor_records(record, now=now)
    if not successors:
        return None
    promoted = dict(successors[0])
    promoted["successors"] = successors[1:]
    promoted["promoted_successor"] = True
    promoted["promoted_at_utc"] = now.isoformat()
    promoted["promoted_from_event_id"] = (
        (record.get("event") or {}).get("event_id")
        if isinstance(record.get("event"), dict)
        else None
    )
    if record.get("successor_overflow_count"):
        promoted["successor_overflow_count"] = record.get("successor_overflow_count")
    return promoted


def _active_pending_dispatch_records(value: Any, *, now: datetime) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for dedupe_key, record in value.items():
        if not isinstance(record, dict):
            continue
        event = record.get("event") if isinstance(record.get("event"), dict) else {}
        expires_at = _parse_utc(record.get("expires_at_utc"))
        normalized = dict(record)
        normalized["successors"] = _active_pending_successor_records(record, now=now)
        if not event or expires_at is None or expires_at <= now:
            promoted = _promote_pending_dispatch_successor(normalized, now=now)
            if promoted is None:
                continue
            normalized = promoted
        active[str(dedupe_key)] = normalized
    return active


def _select_dispatch_event(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    review_events = escalation.get("events_to_review")
    if not isinstance(review_events, list) or not review_events:
        return None, {"status": "NO_EVENTS_TO_REVIEW"}

    reviewed = _accepted_review_records(dispatcher_state.get("reviewed_events"))
    attempts = (
        dispatcher_state.get("dispatch_attempts")
        if isinstance(dispatcher_state.get("dispatch_attempts"), dict)
        else {}
    )
    candidates: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    del events_payload
    for raw_event in review_events:
        if not isinstance(raw_event, dict):
            continue
        # events_to_review is the immutable observation that triggered this
        # wake.  Replacing it with a newer raw quote tick would reset retry
        # identity and could evade backoff indefinitely.
        event = dict(raw_event)
        event["wake_reason_codes"] = list(raw_event.get("wake_reason_codes") or [])
        reasons = {str(item).strip().upper() for item in event.get("wake_reason_codes") or [] if str(item).strip()}
        dedupe_key = str(event.get("dedupe_key") or "").strip()
        if not dedupe_key:
            suppressed.append({"event": event, "reason": "MISSING_DEDUPE_KEY"})
            continue
        if not any(_is_dispatch_reason(reason) for reason in reasons):
            suppressed.append({"event": event, "reason": "NO_DISPATCHABLE_STATE_CHANGE_REASON"})
            continue
        retry_suppression = _retry_suppression(
            attempts.get(dedupe_key) if isinstance(attempts, dict) else None,
            event=event,
            now=now,
            env=env,
        )
        if retry_suppression is not None:
            pending_events.append(event)
            suppressed.append({"event": event, **retry_suppression})
            continue
        record = reviewed.get(dedupe_key) if isinstance(reviewed, dict) else None
        throttle = _dispatch_throttle_seconds(event, env)
        if isinstance(record, dict):
            previous_severity = _severity_rank(record.get("severity"))
            current_severity = _severity_rank(event.get("severity"))
            last_reviewed = _parse_utc(record.get("last_reviewed_at_utc"))
            bypass = _event_bypasses_dispatch_throttle(event, reasons, previous_severity=previous_severity)
            if last_reviewed is not None and not bypass:
                age = (now - last_reviewed).total_seconds()
                if age < throttle:
                    suppressed.append(
                        {
                            "event": event,
                            "reason": "THROTTLED",
                            "age_seconds": age,
                            "throttle_seconds": throttle,
                        }
                    )
                    continue
            if current_severity < previous_severity:
                suppressed.append({"event": event, "reason": "SEVERITY_DECREASED_SINCE_ACCEPTED_REVIEW"})
                continue
            if current_severity == previous_severity and not _accepted_baseline_changed(record, event):
                suppressed.append({"event": event, "reason": "DEDUPE_KEY_ALREADY_REVIEWED"})
                continue
        candidates.append(event)
        pending_events.append(event)

    if not candidates:
        return None, {
            "status": "NO_DISPATCHABLE_EVENT",
            "suppressed": suppressed,
            "pending_events": pending_events,
        }
    candidates.sort(key=_dispatch_priority)
    selected = candidates[0]
    return selected, {
        "status": "SELECTED",
        "suppressed": suppressed,
        "candidate_count": len(candidates),
        "pending_events": pending_events,
        "selected_priority": _dispatch_priority_evidence(selected),
    }


def _dispatch_throttle_seconds(event: dict[str, Any], env: dict[str, str]) -> int:
    if str(event.get("action_hint") or "").upper() in {"TRADE", "ADD"} or str(
        event.get("recommended_review_type") or ""
    ).upper() in {"ENTRY_REVIEW", "ADD_REVIEW"}:
        return int(env.get("QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS", DEFAULT_FRESH_ACTION_THROTTLE_SECONDS))
    return int(env.get("QR_GUARDIAN_EVENT_THROTTLE_SECONDS", DEFAULT_THROTTLE_SECONDS))


def _accepted_review_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): record
        for key, record in value.items()
        if isinstance(record, dict) and record.get("receipt_written") is True
    }


def _event_bypasses_dispatch_throttle(
    event: dict[str, Any],
    reasons: set[str],
    *,
    previous_severity: int,
) -> bool:
    if "DURABLE_PENDING_SUCCESSOR" in reasons:
        return True
    severity_increased = _severity_rank(event.get("severity")) > previous_severity
    return severity_increased and (
        str(event.get("severity") or "").upper() == "P0" or "SEVERITY_INCREASE" in reasons
    )


def _is_dispatch_reason(reason: str) -> bool:
    code = str(reason or "").strip().upper()
    if code in _EXPLICIT_DISPATCH_REASONS:
        return True
    if _is_thesis_degrade_reason(code) or "HARVEST" in code:
        return True
    if "TECHNICAL_STATE_CHANGE" in code:
        return True
    # Future router contracts can name the exact regime/family transition
    # without requiring a dispatcher release. events_to_review remains the
    # deterministic upstream boundary, so this does not wake on raw indicators.
    return any(token in code for token in ("REGIME", "VOLATILITY", "FAMILY", "STRUCTURE"))


def _is_thesis_degrade_reason(reason: str) -> bool:
    match = re.fullmatch(r"THESIS_([A-Z]+)_TO_([A-Z]+)", str(reason or "").upper())
    if not match:
        return False
    rank = {"UNKNOWN": 0, "ALIVE": 1, "WOUNDED": 2, "INVALIDATED": 3, "EMERGENCY": 4}
    return rank.get(match.group(2), -1) > rank.get(match.group(1), -1)


def _event_material_fingerprint(event: dict[str, Any]) -> str:
    material = {
        key: event.get(key)
        for key in (
            "event_id",
            "event_type",
            "pair",
            "direction",
            "thesis",
            "price_zone",
            "severity",
            "recommended_review_type",
            "action_hint",
            "thesis_state",
            "details",
        )
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _event_semantic_state_id(event: dict[str, Any]) -> str:
    """Stable tuning identity, excluding observation-only clock/spread noise.

    Dispatcher retry state still uses ``_event_material_fingerprint`` because
    it must bind to one exact observation.  The hourly tuning queue instead
    groups the market/lane state being reviewed, so regenerating the same
    closed candles or widening only the rollover ask cannot create a new
    obligation.
    """

    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    event_type = str(event.get("event_type") or "").upper()
    semantic: dict[str, Any] = {
        "event_type": event_type,
        "pair": str(event.get("pair") or "").upper(),
        "direction": str(event.get("direction") or "").upper() or None,
        "thesis": event.get("thesis"),
        "action_hint": str(event.get("action_hint") or "").upper(),
        "thesis_state": str(event.get("thesis_state") or "").upper(),
        "dedupe_key": event.get("dedupe_key"),
    }
    if event_type == "TECHNICAL_STATE_CHANGE":
        fingerprint = (
            details.get("material_fingerprint")
            if isinstance(details.get("material_fingerprint"), dict)
            else {}
        )
        watermarks = (
            details.get("closed_candle_watermarks")
            if isinstance(details.get("closed_candle_watermarks"), dict)
            else {}
        )
        semantic["technical_state"] = fingerprint
        semantic["closed_candle_watermarks"] = {
            str(key).upper(): value for key, value in sorted(watermarks.items())
        }
    elif event_type == "FAILED_ACCEPTANCE":
        lane_id = str(details.get("lane_id") or "").strip()
        status = str(details.get("status") or "").strip().upper()
        if lane_id or status:
            semantic["lane_lifecycle"] = {"lane_id": lane_id or None, "status": status or None}
        else:
            # A major figure is the tuning obligation. Bid/ask movement is
            # observation evidence inside that obligation; putting an exact
            # quote in semantic identity churns one queue slot per tick.
            semantic["major_figure"] = _major_figure_identity(event.get("price_zone"))
    else:
        semantic["price_zone"] = event.get("price_zone")
        semantic["stable_details"] = {
            key: details.get(key)
            for key in (
                "trade_id",
                "order_id",
                "lane_id",
                "status",
                "action",
                "owner",
                "position_intent",
            )
            if details.get(key) is not None
        }
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _major_figure_identity(value: Any) -> str | None:
    match = re.search(r"\bmajor\s+figure\s+(-?\d+(?:\.\d+)?)", str(value or ""), re.IGNORECASE)
    return match.group(1) if match else None


def _work_order_semantic_state_id(entry: dict[str, Any]) -> str:
    explicit = str(entry.get("semantic_state_id") or "").strip()
    if explicit:
        return explicit
    selected = entry.get("selected_event") if isinstance(entry.get("selected_event"), dict) else None
    if selected is not None:
        return _event_semantic_state_id(selected)
    return str(entry.get("event_fingerprint") or entry.get("work_order_id") or "")


def _accepted_baseline_changed(record: dict[str, Any], event: dict[str, Any]) -> bool:
    previous_event_id = str(record.get("event_id") or "")
    current_event_id = str(event.get("event_id") or "")
    if previous_event_id and current_event_id and previous_event_id != current_event_id:
        return True
    previous_fingerprint = str(record.get("material_fingerprint") or "")
    # Accepted records written before material baselines were introduced do
    # not carry enough fields for a local comparison. Trust one explicit
    # router material-change reason after the normal review throttle; the new
    # accepted receipt then writes a complete baseline and restores strict
    # fingerprint comparison for subsequent cycles.
    if not previous_fingerprint and _event_is_active_or_material(event):
        return True
    return bool(previous_fingerprint and previous_fingerprint != _event_material_fingerprint(event))


def _retry_suppression(
    record: Any,
    *,
    event: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    if str(record.get("event_id") or "") != str(event.get("event_id") or ""):
        return None
    recorded_fingerprint = str(record.get("material_fingerprint") or "")
    if recorded_fingerprint and recorded_fingerprint != _event_material_fingerprint(event):
        return None
    if str(record.get("last_status") or "").upper() == "RUNTIME_DISK_P0":
        prior_disk = record.get("runtime_disk") if isinstance(record.get("runtime_disk"), dict) else {}
        disk_path = Path(str(prior_disk.get("path") or "."))
        if _runtime_disk_state(disk_path, env).get("status") != "RUNTIME_DISK_P0":
            return None
    prior_binary = record.get("codex_binary_identity")
    current_binary = _codex_binary_identity(env)
    if isinstance(prior_binary, dict) and _binary_identity_key(prior_binary) != _binary_identity_key(current_binary):
        return None
    expires_at = _parse_utc(record.get("expires_at_utc"))
    if expires_at is not None and now >= expires_at:
        return None
    retry_after = _parse_utc(record.get("retry_after_utc"))
    attempt_count = int(record.get("attempt_count") or 0)
    max_attempts = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS)))
    if attempt_count >= max_attempts:
        return {
            "reason": "RETRY_BUDGET_EXHAUSTED",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "retry_after_utc": record.get("retry_after_utc"),
            "expires_at_utc": record.get("expires_at_utc"),
        }
    if retry_after is not None and now < retry_after:
        return {
            "reason": "RETRY_BACKOFF",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "retry_after_utc": retry_after.isoformat(),
            "expires_at_utc": record.get("expires_at_utc"),
        }
    return None


def _codex_binary_identity(env: dict[str, str]) -> dict[str, Any]:
    path = Path(_resolve_codex_bin(env))
    identity: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        return identity
    identity.update(
        {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return identity


def _binary_identity_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(identity.get(key) for key in ("path", "device", "inode", "size", "mtime_ns"))


def _event_has_open_exposure(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").upper()
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    candidate_harvest = bool(
        event_type == "HARVEST_ZONE"
        and str(details.get("lane_id") or "").strip()
        and str(details.get("status") or "").strip()
        and not any(
            str(details.get(key) or "").strip()
            for key in ("trade_id", "position_id", "open_trade_id")
        )
    )
    if candidate_harvest:
        return False
    if event_type in _ACTIVE_EXPOSURE_EVENT_TYPES:
        return True
    stack: list[Any] = [details]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                key_text = str(key).lower()
                if key_text in {"trade_id", "position_id", "open_trade_id"} and str(value or "").strip():
                    return True
                if key_text in {"units", "open_units", "current_units"}:
                    try:
                        if float(value or 0) != 0:
                            return True
                    except (TypeError, ValueError):
                        pass
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _event_is_active_or_material(event: dict[str, Any]) -> bool:
    reasons = {str(item).strip().upper() for item in event.get("wake_reason_codes") or []}
    return _event_has_open_exposure(event) or any(
        reason in _MATERIAL_CHANGE_REASONS
        or _is_thesis_degrade_reason(reason)
        or "HARVEST" in reason
        or "TECHNICAL_STATE_CHANGE" in reason
        or "REGIME" in reason
        or "VOLATILITY" in reason
        or "FAMILY" in reason
        or "STRUCTURE" in reason
        for reason in reasons
    )


def _dispatch_priority(event: dict[str, Any]) -> tuple[Any, ...]:
    open_exposure = _event_has_open_exposure(event)
    p0 = str(event.get("severity") or "").upper() == "P0"
    active_or_material = _event_is_active_or_material(event)
    return (
        -int(open_exposure or p0),
        -int(open_exposure),
        -_severity_rank(event.get("severity")),
        -int(active_or_material),
        str(event.get("pair") or ""),
        str(event.get("event_type") or ""),
        str(event.get("dedupe_key") or ""),
    )


def _dispatch_priority_evidence(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_exposure": _event_has_open_exposure(event),
        "p0": str(event.get("severity") or "").upper() == "P0",
        "active_or_material": _event_is_active_or_material(event),
        "severity": event.get("severity"),
        "alphabetical_tiebreak": [event.get("pair"), event.get("event_type"), event.get("dedupe_key")],
    }


def _broker_snapshot_freshness(payload: dict[str, Any], *, now: datetime, env: dict[str, str]) -> dict[str, Any]:
    raw = payload.get("fetched_at_utc") or (payload.get("account") if isinstance(payload.get("account"), dict) else {}).get(
        "fetched_at_utc"
    )
    fetched = _parse_utc(raw)
    max_age = int(env.get("QR_GUARDIAN_WAKE_MAX_SNAPSHOT_AGE_SECONDS", BROKER_SNAPSHOT_MAX_AGE_SECONDS))
    if fetched is None:
        return {"fresh": False, "status": "MISSING_FETCHED_AT", "max_age_seconds": max_age}
    age = (now - fetched).total_seconds()
    return {
        "fresh": age <= max_age,
        "status": "FRESH" if age <= max_age else "STALE",
        "age_seconds": age,
        "max_age_seconds": max_age,
        "fetched_at_utc": fetched.isoformat(),
    }


def _active_live_lock(lock_dir: Path) -> dict[str, Any]:
    if not lock_dir.exists():
        return {"active": False, "path": str(lock_dir)}
    pid = _read_text(lock_dir / "pid").strip()
    label = _read_text(lock_dir / "command").strip()
    started = _read_text(lock_dir / "started_at_utc").strip()
    active = pid.isdigit() and _pid_running(int(pid))
    command = ""
    if active:
        command = _process_command(int(pid))
    return {
        "active": bool(active),
        "path": str(lock_dir),
        "pid": int(pid) if pid.isdigit() else None,
        "label": label or None,
        "started_at_utc": started or None,
        "process_command": command or None,
    }


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_command(pid: int) -> str:
    try:
        proc = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _mark_escalation_queued(
    path: Path,
    escalation: dict[str, Any],
    result: dict[str, Any],
    *,
    now: datetime,
    reason: str,
) -> None:
    payload = dict(escalation)
    payload["queued_for_active_trader"] = True
    payload["queued_at_utc"] = now.isoformat()
    payload["queue_reason"] = reason
    payload["guardian_wake_dispatcher_status"] = result.get("status")
    payload["guardian_wake_dispatcher_lock"] = result.get("lock")
    payload["guardian_wake_dispatcher_parse_failure"] = result.get("parse_failure")
    _write_json(path, payload)


def _prepare_codex_home(codex_home: Path, *, live_root: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_src = Path.home() / ".codex" / "auth.json"
    auth_dst = codex_home / "auth.json"
    if auth_src.exists() and not auth_dst.exists():
        auth_dst.symlink_to(auth_src)
    config = (
        f'model = "{MODEL}"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        "model_reasoning_effort = \"high\"\n"
        "\n"
        f'[projects."{live_root}"]\n'
        'trust_level = "trusted"\n'
    )
    config_path = codex_home / "config.toml"
    if not config_path.exists() or config_path.read_text() != config:
        config_path.write_text(config)


def _refresh_broker_snapshot(
    *,
    paths: DispatcherPaths,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
) -> dict[str, Any]:
    if env.get("QR_GUARDIAN_WAKE_DISABLE_SNAPSHOT_REFRESH", "0") == "1":
        return {"status": "DISABLED", "reason": "QR_GUARDIAN_WAKE_DISABLE_SNAPSHOT_REFRESH=1"}
    python_bin = env.get("QR_PYTHON", "python3")
    cmd = [
        python_bin,
        "-m",
        "quant_rabbit.cli",
        "broker-snapshot",
        "--output",
        str(paths.broker_snapshot),
    ]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    timeout = int(env.get("QR_GUARDIAN_WAKE_SNAPSHOT_REFRESH_TIMEOUT_SECONDS", "90"))
    try:
        proc = subprocess_run(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=timeout, env=run_env)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "TIMEOUT",
            "command": cmd,
            "timeout_seconds": timeout,
            "stdout_tail": _decoded_tail(exc.stdout),
            "stderr_tail": _decoded_tail(exc.stderr),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "EXCEPTION", "command": cmd, "error": str(exc)}
    return {
        "status": "REFRESHED" if proc.returncode == 0 and paths.broker_snapshot.exists() else "FAILED",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "read_only_broker_operation": True,
    }


def _resolve_codex_bin(env: dict[str, str]) -> str:
    configured = str(env.get("QR_GUARDIAN_WAKE_CODEX_BIN") or "").strip()
    if configured:
        return configured
    # Prefer the desktop-bundled CLI: it is upgraded with the app and can
    # support the requested model while a Homebrew/PATH copy remains older.
    # The app was renamed from Codex.app to ChatGPT.app, so keep the legacy
    # location as a compatibility fallback without selecting it when absent.
    for app_bin in (DEFAULT_CODEX_APP_BIN, LEGACY_CODEX_APP_BIN):
        if app_bin.exists():
            return str(app_bin)
    found = shutil.which("codex")
    if found:
        return found
    return "codex"


def _run_codex_preflight(
    *,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
) -> dict[str, Any]:
    codex_bin = _resolve_codex_bin(env)
    base = {
        "enabled": env.get("QR_GUARDIAN_WAKE_CODEX_PREFLIGHT", "0") == "1",
        "codex_binary_path": codex_bin,
        "requested_model": MODEL,
        "minimum_supported_version": ".".join(str(item) for item in MIN_GPT55_CODEX_VERSION),
    }
    if not base["enabled"]:
        return {**base, "status": "SKIPPED"}
    cmd = [codex_bin, "--version"]
    timeout = int(env.get("QR_GUARDIAN_WAKE_CODEX_PREFLIGHT_TIMEOUT_SECONDS", "10"))
    try:
        proc = subprocess_run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(env))
    except FileNotFoundError as exc:
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc),
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": _decoded_tail(exc.stdout),
            "stderr_excerpt": _decoded_tail(exc.stderr),
            "timeout_seconds": timeout,
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc),
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    stdout = str(getattr(proc, "stdout", "") or "")
    stderr = str(getattr(proc, "stderr", "") or "")
    version = _parse_codex_version(stdout or stderr)
    version_tuple = _codex_version_tuple(version)
    if getattr(proc, "returncode", 1) != 0:
        status = "CODEX_CLI_VERSION_UNSUPPORTED"
    elif version_tuple is None:
        status = "CODEX_CLI_VERSION_UNSUPPORTED"
    elif version_tuple < MIN_GPT55_CODEX_VERSION:
        status = "CODEX_MODEL_UNSUPPORTED"
    else:
        status = "OK"
    result = {
        **base,
        "status": status,
        "command": cmd,
        "returncode": getattr(proc, "returncode", None),
        "codex_version": version,
        "stdout_excerpt": stdout[-1000:],
        "stderr_excerpt": stderr[-1000:],
        "supports_requested_model": status == "OK",
    }
    if status != "OK":
        result["cli_failure_class"] = "CODEX_CLI_VERSION_UNSUPPORTED"
        result["remediation_hint"] = _codex_remediation_hint(status=status, codex_bin=codex_bin)
    return result


def _build_prompt(
    *,
    paths: DispatcherPaths,
    selected_event: dict[str, Any],
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    event_state: dict[str, Any],
    broker_snapshot: dict[str, Any],
    daily_target_state: dict[str, Any],
    event_report: str,
    dispatcher_state: dict[str, Any],
    runtime_disk: dict[str, Any],
) -> str:
    template = _read_text(paths.prompt_template)
    if not template:
        template = "Return exactly one guardian wake JSON receipt."
    prompt_event = _prompt_selected_event(selected_event)
    prompt_payload = {
        "selected_event": prompt_event,
        "guardian_escalation": _compact_escalation(escalation, selected_event=prompt_event),
        "guardian_events": _compact_guardian_events(events_payload, selected_event=prompt_event),
        "guardian_event_state": _compact_event_state(event_state, selected_event=selected_event),
        "broker_snapshot": _compact_broker_snapshot(broker_snapshot, pair=str(selected_event.get("pair") or "")),
        "daily_target_state": _compact_daily_target_state(
            daily_target_state,
            pair=str(selected_event.get("pair") or ""),
        ),
        "dispatcher_state": _compact_dispatcher_state(dispatcher_state, selected_event=selected_event),
        "runtime_disk": runtime_disk,
    }
    selected_event_id = str(selected_event.get("event_id") or "")
    selected_pair = str(selected_event.get("pair") or "")
    selected_side = str(selected_event.get("direction") or "NONE").upper()
    if selected_side not in {"LONG", "SHORT"}:
        selected_side = "NONE"
    return (
        template.rstrip()
        + "\n\n# Authoritative Single Event\n\n"
        + f"Review only event_id={selected_event_id} pair={selected_pair}. "
        + "Do not choose, inspect, or answer for any other pending/reviewed event. "
        + "Your JSON must echo this exact event_id and pair; otherwise it will be rejected. "
        + f"Set side exactly to {selected_side}; side must be LONG, SHORT, or NONE, never UNKNOWN or N/A. "
        + "If this event alone is insufficient, return HOLD or NO_ACTION for this same identity.\n"
        + "\n\nIf the final-message capture is unavailable, also return the same JSON object as the final assistant message. "
        + f"If an explicit output file is available, use this path for the JSON object only: {paths.codex_explicit_output}.\n"
        + "\n\n# Dispatcher Context\n\n"
        + "Use only the following local artifacts. Do not request web, OANDA, broker, or API access.\n\n"
        + "```json\n"
        + _json_dumps(prompt_payload, limit=60_000)
        + "\n```\n\n"
        + "# Selected event report evidence\n\n"
        + _selected_event_report_excerpt(event_report, selected_event=selected_event)
        + "\n"
    )


def _prompt_selected_event(selected_event: dict[str, Any]) -> dict[str, Any]:
    """Preserve one top-level receipt identity and redact nested alternatives."""

    compact: dict[str, Any] = {}
    for key, value in selected_event.items():
        compact[key] = (
            _strip_nested_event_identities(value)
            if isinstance(value, (dict, list))
            else value
        )
    return compact


def _strip_nested_event_identities(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nested_event_identities(item)
            for key, item in value.items()
            if key not in {"event_id", "dedupe_key", "pair"}
        }
    if isinstance(value, list):
        return [_strip_nested_event_identities(item) for item in value]
    return value


def _compact_escalation(
    payload: dict[str, Any],
    *,
    selected_event: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "generated_at_utc",
            "wake_gpt",
            "model_target",
            "wake_policy",
            "execution_boundary",
        )
    }
    compact["wake_reason_codes"] = list(selected_event.get("wake_reason_codes") or [])
    compact["events_to_review"] = [selected_event]
    return compact


def _compact_guardian_events(
    payload: dict[str, Any],
    *,
    selected_event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "events": [selected_event],
        "trigger_contract": _compact_selected_trigger_contract(
            payload.get("trigger_contract"),
            selected_event=selected_event,
        ),
    }


def _compact_selected_trigger_contract(
    payload: Any,
    *,
    selected_event: dict[str, Any],
) -> dict[str, Any] | None:
    """Keep contract context only when the selected event is contract-derived."""

    event_type = str(selected_event.get("event_type") or "").upper()
    if not event_type.startswith("CONTRACT_") or not isinstance(payload, dict):
        return None
    issues = [
        {
            "code": issue.get("code"),
            "severity": issue.get("severity"),
        }
        for issue in payload.get("issues", []) or []
        if isinstance(issue, dict)
    ]
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "age_seconds": payload.get("age_seconds"),
        "entry_count": payload.get("entry_count"),
        "issues": issues[:20],
    }


def _compact_daily_target_state(payload: dict[str, Any], *, pair: str) -> dict[str, Any]:
    """Expose risk-budget scalars without leaking unrelated pair narratives."""

    compact = {
        key: payload.get(key)
        for key in (
            "generated_at_utc",
            "as_of_utc",
            "status",
            "pace_state",
            "current_equity_jpy",
            "realized_pl_jpy",
            "unrealized_pl_jpy",
            "open_risk_jpy",
            "daily_risk_budget_jpy",
            "remaining_risk_budget_jpy",
            "per_trade_risk_budget_jpy",
            "target_return_pct",
            "target_profit_jpy",
            "progress_pct",
        )
    }
    selected_pair = pair.upper()
    positions = [
        item
        for item in payload.get("positions", []) or []
        if isinstance(item, dict)
        and selected_pair
        and str(item.get("pair") or item.get("instrument") or "").upper() == selected_pair
    ]
    raw_unprotected_count = payload.get("unprotected_positions")
    if isinstance(raw_unprotected_count, (int, float)) and not isinstance(raw_unprotected_count, bool):
        global_unprotected_count: int | float | None = raw_unprotected_count
    elif isinstance(raw_unprotected_count, list):
        global_unprotected_count = len(raw_unprotected_count)
    else:
        global_unprotected_count = None
    compact["selected_pair_positions"] = positions
    compact["global_unprotected_position_count"] = global_unprotected_count
    return compact


def _compact_dispatcher_state(
    payload: dict[str, Any],
    *,
    selected_event: dict[str, Any],
) -> dict[str, Any]:
    dedupe_key = str(selected_event.get("dedupe_key") or "")
    attempts = payload.get("dispatch_attempts") if isinstance(payload.get("dispatch_attempts"), dict) else {}
    pending = payload.get("pending_dispatches") if isinstance(payload.get("pending_dispatches"), dict) else {}
    selected_attempt = attempts.get(dedupe_key) if isinstance(attempts.get(dedupe_key), dict) else {}
    selected_pending = pending.get(dedupe_key) if isinstance(pending.get(dedupe_key), dict) else {}
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "last_status": payload.get("last_status"),
        "selected_dispatch_attempt": {
            key: selected_attempt.get(key)
            for key in (
                "attempt_count",
                "max_attempts",
                "last_error",
                "last_status",
                "last_failed_at_utc",
                "retry_after_utc",
                "expires_at_utc",
                "retry_budget_exhausted",
            )
        }
        if selected_attempt
        else None,
        "selected_pending_dispatch": {
            key: selected_pending.get(key)
            for key in (
                "queued_at_utc",
                "last_seen_at_utc",
                "expires_at_utc",
                "material_fingerprint",
            )
        }
        if selected_pending
        else None,
        "usage_limit_recovered_at_utc": payload.get("usage_limit_recovered_at_utc"),
    }


def _compact_event_state(
    payload: dict[str, Any],
    *,
    selected_event: dict[str, Any],
) -> dict[str, Any]:
    dedupe_key = str(selected_event.get("dedupe_key") or "")
    events = payload.get("events") if isinstance(payload.get("events"), dict) else {}
    selected_state = events.get(dedupe_key)
    selected_id = str(selected_event.get("event_id") or "")
    state_id = str(selected_state.get("event_id") or "") if isinstance(selected_state, dict) else ""
    exact_state = isinstance(selected_state, dict) and bool(selected_id) and state_id == selected_id
    state_summary = None
    if exact_state:
        state_summary = {
            key: selected_state.get(key)
            for key in (
                "last_seen_at_utc",
                "last_wake_at_utc",
                "in_harvest_zone",
                "margin_pressure",
                "severity",
                "thesis_state",
            )
        }
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": "EXACT_EVENT_STATE" if exact_state else "CURRENT_STATE_DIFFERS",
        "selected_state": state_summary,
    }


def _selected_event_report_excerpt(
    report: str,
    *,
    selected_event: dict[str, Any],
) -> str:
    text = str(report or "")
    for needle in (str(selected_event.get("event_id") or ""),):
        if not needle:
            continue
        index = text.find(needle)
        if index >= 0:
            start = text.rfind("\n- `", 0, index)
            start = 0 if start < 0 else start + 1
            end = text.find("\n- `", index + len(needle))
            if end < 0:
                end = text.find("\n## ", index + len(needle))
            if end < 0:
                end = min(len(text), index + len(needle) + 1_600)
            return text[start:end][:4_000]
    return "No pair-specific report excerpt; use the authoritative selected_event JSON only."


def _compact_broker_snapshot(payload: dict[str, Any], *, pair: str) -> dict[str, Any]:
    quotes = payload.get("quotes") if isinstance(payload.get("quotes"), dict) else {}
    selected_quotes = {}
    if pair and pair in quotes:
        selected_quotes[pair] = quotes[pair]
    positions = [
        item
        for item in payload.get("positions", []) or []
        if isinstance(item, dict) and (not pair or str(item.get("pair") or item.get("instrument") or "").upper() == pair.upper())
    ]
    orders = [
        item
        for item in payload.get("orders", []) or []
        if isinstance(item, dict) and (not pair or str(item.get("pair") or item.get("instrument") or "").upper() == pair.upper())
    ]
    return {
        "fetched_at_utc": payload.get("fetched_at_utc"),
        "account": payload.get("account"),
        "positions": positions,
        "orders": orders,
        "quotes": selected_quotes,
    }


def _run_codex(
    *,
    paths: DispatcherPaths,
    prompt: str,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
    attempt: str,
    codex_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths.codex_output.parent.mkdir(parents=True, exist_ok=True)
    paths.codex_output.write_text("")
    paths.codex_explicit_output.parent.mkdir(parents=True, exist_ok=True)
    paths.codex_explicit_output.write_text("")
    codex_preflight = codex_preflight if isinstance(codex_preflight, dict) else {}
    codex_bin = str(codex_preflight.get("codex_binary_path") or _resolve_codex_bin(env))
    codex_version = codex_preflight.get("codex_version")
    timeout = int(env.get("QR_GUARDIAN_WAKE_CODEX_TIMEOUT_SECONDS", CODEX_TIMEOUT_SECONDS))
    cmd = [
        codex_bin,
        "exec",
        "-",
        "-m",
        MODEL,
        "-s",
        "read-only",
        "-C",
        str(paths.live_root),
        "--json",
        "--output-last-message",
        str(paths.codex_output),
    ]
    run_env = dict(env)
    run_env["CODEX_HOME"] = str(paths.codex_home)
    run_env["CODEX_DISABLE_UPDATE_CHECK"] = "1"
    session_before = _latest_session_jsonl(paths.codex_home)
    session_before_mtime = _path_mtime(session_before)
    try:
        proc = subprocess_run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=run_env)
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_process_text(exc.stdout)
        stderr_text = _decode_process_text(exc.stderr)
        session_message, session_path = _latest_session_assistant_message(
            paths.codex_home,
            previous=session_before,
            previous_mtime=session_before_mtime,
        )
        explicit_message = _read_text(paths.codex_explicit_output).strip()
        raw_last_message = _read_text(paths.codex_output).strip()
        candidate, source = _first_message_candidate(
            [
                ("output-last-message", raw_last_message),
                ("explicit-output-file", explicit_message),
                ("stdout-assistant", _extract_stdout_assistant_message(stdout_text)),
                ("session-jsonl-assistant", session_message),
            ]
        )
        return {
            "status": "CODEX_TIMEOUT",
            "attempt": attempt,
            "command": cmd,
            "codex_binary_path": codex_bin,
            "codex_version": codex_version,
            "requested_model": MODEL,
            "returncode": None,
            "stderr_tail": stderr_text[-1000:],
            "stdout_tail": stdout_text[-1000:],
            "raw_stdout_excerpt": stdout_text[-2000:],
            "raw_stderr_excerpt": stderr_text[-2000:],
            "output_path": str(paths.codex_output),
            "explicit_output_path": str(paths.codex_explicit_output),
            "session_jsonl_path": str(session_path) if session_path else None,
            "last_message": candidate,
            "last_message_source": source,
            "last_message_file_empty": not raw_last_message,
            "explicit_output_file_empty": not explicit_message,
            "stdout_fallback_used": source == "stdout-assistant",
            "session_jsonl_fallback_used": source == "session-jsonl-assistant",
            "timeout_seconds": timeout,
        }
    except Exception as exc:
        return {
            "status": "CODEX_EXCEPTION",
            "attempt": attempt,
            "command": cmd,
            "codex_binary_path": codex_bin,
            "codex_version": codex_version,
            "requested_model": MODEL,
            "returncode": None,
            "stderr_tail": str(exc),
            "stdout_tail": "",
            "raw_stdout_excerpt": "",
            "raw_stderr_excerpt": str(exc)[-2000:],
            "output_path": str(paths.codex_output),
            "explicit_output_path": str(paths.codex_explicit_output),
            "session_jsonl_path": None,
            "last_message": "",
            "last_message_source": None,
            "last_message_file_empty": True,
            "explicit_output_file_empty": True,
            "stdout_fallback_used": False,
            "session_jsonl_fallback_used": False,
        }
    raw_last_message = _read_text(paths.codex_output)
    explicit_message = _read_text(paths.codex_explicit_output)
    stdout_text = str(getattr(proc, "stdout", "") or "")
    stderr_text = str(getattr(proc, "stderr", "") or "")
    stdout_message = _extract_stdout_assistant_message(stdout_text)
    session_message, session_path = _latest_session_assistant_message(
        paths.codex_home,
        previous=session_before,
        previous_mtime=session_before_mtime,
    )
    last_message_file_empty = not raw_last_message.strip()
    explicit_output_file_empty = not explicit_message.strip()
    last_message, last_message_source = _first_message_candidate(
        [
            ("output-last-message", raw_last_message),
            ("explicit-output-file", explicit_message),
            ("stdout-assistant", stdout_message),
            ("session-jsonl-assistant", session_message),
        ]
    )
    stdout_fallback_used = last_message_source == "stdout-assistant"
    session_jsonl_fallback_used = last_message_source == "session-jsonl-assistant"
    current_invocation_failed = getattr(proc, "returncode", 1) != 0
    model_failure = (
        _codex_model_version_failure(stderr_text=stderr_text, stdout_text=stdout_text, codex_bin=codex_bin)
        if current_invocation_failed
        else None
    )
    usage_limit = _codex_usage_limit_failure(
        stderr_text=stderr_text,
        stdout_text=stdout_text,
        last_message=last_message,
    )
    if getattr(proc, "returncode", 1) == 0 and last_message:
        status = "CODEX_USAGE_LIMIT" if usage_limit is not None else "OK"
    elif model_failure is not None:
        status = model_failure["status"]
    elif usage_limit is not None:
        status = "CODEX_USAGE_LIMIT"
    elif _codex_auth_or_sandbox_failed(getattr(proc, "returncode", 1), stderr_text, stdout_text):
        status = "CODEX_AUTH_OR_SANDBOX_FAILURE"
    elif stdout_text.strip() or session_path is not None:
        status = "CODEX_NO_ASSISTANT_MESSAGE"
    elif getattr(proc, "returncode", 1) == 0 and not last_message:
        status = "CODEX_EMPTY_LAST_MESSAGE"
    else:
        status = "CODEX_FAILED"
    return {
        "status": status,
        "attempt": attempt,
        "command": cmd,
        "codex_binary_path": codex_bin,
        "codex_version": codex_version,
        "requested_model": MODEL,
        "supports_requested_model": status == "OK",
        "remediation_hint": (model_failure or usage_limit or {}).get("remediation_hint"),
        "cli_failure_class": (model_failure or {}).get("cli_failure_class"),
        "returncode": getattr(proc, "returncode", None),
        "stderr_tail": stderr_text[-1000:],
        "stdout_tail": stdout_text[-1000:],
        "raw_stdout_excerpt": stdout_text[-2000:],
        "raw_stderr_excerpt": stderr_text[-2000:],
        "output_path": str(paths.codex_output),
        "explicit_output_path": str(paths.codex_explicit_output),
        "session_jsonl_path": str(session_path) if session_path else None,
        "last_message": last_message,
        "last_message_source": last_message_source,
        "last_message_file_empty": last_message_file_empty,
        "explicit_output_file_empty": explicit_output_file_empty,
        "stdout_fallback_used": stdout_fallback_used,
        "session_jsonl_fallback_used": session_jsonl_fallback_used,
    }


def _parse_codex_receipt(text: str, *, empty_error: str = "CODEX_EMPTY_LAST_MESSAGE") -> dict[str, Any]:
    if not text.strip():
        return {"valid": False, "error": empty_error, "raw_output_excerpt": ""}
    payload = _extract_json_object(text)
    if payload is None:
        return {"valid": False, "error": "CODEX_NO_JSON_RECEIPT", "raw_output_excerpt": text[:2000]}
    if "actions" in payload:
        return {"valid": False, "error": "MULTIPLE_ACTIONS_FIELD_FORBIDDEN", "raw_payload": payload}
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    if not isinstance(receipt, dict):
        return {"valid": False, "error": "RECEIPT_NOT_OBJECT", "raw_payload": payload}
    receipt = dict(receipt)
    issues = _receipt_parse_issues(receipt)
    if issues:
        return {"valid": False, "error": "SCHEMA_INVALID", "issues": issues, "receipt": receipt}
    if "invalidation" not in receipt and "invalidation_evidence" in receipt:
        receipt["invalidation"] = receipt["invalidation_evidence"]
    if "invalidation_evidence" not in receipt and "invalidation" in receipt:
        receipt["invalidation_evidence"] = receipt["invalidation"]
    if "ownership" not in receipt and "manual_system_ownership" in receipt:
        receipt["ownership"] = receipt["manual_system_ownership"]
    receipt["action"] = str(receipt["action"]).upper()
    receipt["thesis_state"] = str(receipt["thesis_state"]).upper()
    receipt["ownership"] = str(receipt.get("ownership") or "UNKNOWN").upper()
    if "side" in receipt:
        receipt["side"] = str(receipt.get("side") or "NONE").upper()
    receipt["gateway_required"] = True
    receipt["no_direct_oanda"] = True
    return {"valid": True, "receipt": receipt}


def _parse_codex_result(codex: dict[str, Any]) -> dict[str, Any]:
    status = str(codex.get("status") or "").upper()
    if status in CODEX_RUNTIME_FAILURE_STATUSES:
        return {
            "valid": False,
            "error": status,
            "raw_output_excerpt": str(codex.get("last_message") or codex.get("raw_stdout_excerpt") or "")[:2000],
            "codex_binary_path": codex.get("codex_binary_path"),
            "codex_version": codex.get("codex_version"),
            "requested_model": codex.get("requested_model") or MODEL,
            "remediation_hint": codex.get("remediation_hint"),
        }
    return _parse_codex_receipt(codex.get("last_message") or "", empty_error=_empty_parse_error(codex))


def _empty_parse_error(codex: dict[str, Any]) -> str:
    status = str(codex.get("status") or "").upper()
    if status in {"CODEX_NO_ASSISTANT_MESSAGE", *CODEX_RUNTIME_FAILURE_STATUSES}:
        return status
    return "CODEX_EMPTY_LAST_MESSAGE"


def _first_message_candidate(candidates: list[tuple[str, str]]) -> tuple[str, str | None]:
    for source, value in candidates:
        text = str(value or "").strip()
        if text:
            return text, source
    return "", None


def _extract_stdout_assistant_message(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and ("action" in payload or "receipt" in payload):
            return stripped
    jsonl_message = _extract_assistant_message_from_jsonl(stripped.splitlines())
    if jsonl_message:
        return jsonl_message
    if stripped.startswith("{") and _extract_json_object(stripped) is not None:
        return stripped
    return _extract_transcript_assistant_message(stripped)


def _latest_session_jsonl(codex_home: Path) -> Path | None:
    sessions = codex_home / "sessions"
    try:
        files = [path for path in sessions.rglob("*.jsonl") if path.is_file()]
    except OSError:
        return None
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def _path_mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _latest_session_assistant_message(
    codex_home: Path,
    *,
    previous: Path | None = None,
    previous_mtime: float | None = None,
) -> tuple[str, Path | None]:
    path = _latest_session_jsonl(codex_home)
    if path is None:
        return "", None
    if previous is not None:
        try:
            if path == previous and previous_mtime is not None and path.stat().st_mtime <= previous_mtime:
                return "", path
        except OSError:
            return "", path
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "", path
    return _extract_assistant_message_from_jsonl(lines), path


def _extract_assistant_message_from_jsonl(lines: list[str]) -> str:
    messages: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _assistant_text_from_event(payload)
        if text:
            messages.append(text)
    return messages[-1].strip() if messages else ""


def _assistant_text_from_event(payload: Any) -> str:
    if isinstance(payload, list):
        for item in reversed(payload):
            text = _assistant_text_from_event(item)
            if text:
                return text
        return ""
    if not isinstance(payload, dict):
        return ""
    role = str(payload.get("role") or "").lower()
    event_type = str(payload.get("type") or payload.get("record_type") or "").lower()
    if role == "assistant" or event_type in {"agent_message", "assistant_message", "final_answer"}:
        return _content_text(payload.get("content") or payload.get("message") or payload.get("text"))
    for key in ("message", "item", "delta", "event"):
        text = _assistant_text_from_event(payload.get(key))
        if text:
            return text
    return ""


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "output_text"):
            text = _content_text(value.get(key))
            if text:
                return text
    return ""


def _extract_transcript_assistant_message(text: str) -> str:
    marker = re.search(r"(?im)^\[[^\n\]]+\]\s+(?:assistant|codex)\s*$", text)
    if not marker:
        return ""
    return text[marker.end() :].strip()


def _parse_codex_version(text: str) -> str | None:
    match = re.search(r"codex(?:-cli)?\s+([0-9]+(?:\.[0-9]+){1,3})", str(text or ""), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9]+(?:\.[0-9]+){1,3})\b", str(text or ""))
    return match.group(1) if match else None


def _codex_version_tuple(version: Any) -> tuple[int, int, int] | None:
    if not version:
        return None
    parts = str(version).strip().split(".")
    try:
        numbers = [int(part) for part in parts[:3]]
    except ValueError:
        return None
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])


def _codex_model_version_failure(*, stderr_text: str, stdout_text: str, codex_bin: str) -> dict[str, str] | None:
    haystack = f"{stderr_text}\n{stdout_text}".lower()
    if not haystack.strip():
        return None
    if MODEL.lower() in haystack and any(token in haystack for token in ("newer version", "upgrade", "requires a newer")):
        return {
            "status": "CODEX_MODEL_UNSUPPORTED",
            "cli_failure_class": "CODEX_CLI_VERSION_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_MODEL_UNSUPPORTED", codex_bin=codex_bin),
        }
    if MODEL.lower() in haystack and any(token in haystack for token in ("unsupported model", "unknown model", "invalid model")):
        return {
            "status": "CODEX_MODEL_UNSUPPORTED",
            "cli_failure_class": "CODEX_MODEL_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_MODEL_UNSUPPORTED", codex_bin=codex_bin),
        }
    if "codex" in haystack and "version" in haystack and any(token in haystack for token in ("unsupported", "too old", "upgrade")):
        return {
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "cli_failure_class": "CODEX_CLI_VERSION_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    return None


def _codex_usage_limit_failure(
    *,
    stderr_text: str,
    stdout_text: str,
    last_message: str,
) -> dict[str, str] | None:
    haystack = f"{stderr_text}\n{stdout_text}\n{last_message}".lower()
    if not any(
        marker in haystack
        for marker in (
            "hit your usage limit",
            "usage limit",
            "purchase more credits",
            "try again at",
        )
    ):
        return None
    return {
        "status": "CODEX_USAGE_LIMIT",
        "remediation_hint": (
            "Codex account usage is temporarily exhausted; keep the exact event in the durable "
            "dispatcher queue and delay the next GPT attempt instead of schema-repair retrying."
        ),
    }


def _codex_remediation_hint(*, status: str, codex_bin: str) -> str:
    if status == "CODEX_MODEL_UNSUPPORTED":
        return (
            f"{MODEL} is not supported by Codex binary {codex_bin}. Upgrade that CLI or set "
            "QR_GUARDIAN_WAKE_CODEX_BIN to a GPT-5.5-capable Codex binary such as "
            f"{DEFAULT_CODEX_APP_BIN}."
        )
    return (
        f"Codex binary {codex_bin} could not prove a supported CLI version. Upgrade it or set "
        f"QR_GUARDIAN_WAKE_CODEX_BIN={DEFAULT_CODEX_APP_BIN}."
    )


def _codex_auth_or_sandbox_failed(returncode: int | None, stderr_text: str, stdout_text: str) -> bool:
    if returncode == 0:
        return False
    haystack = f"{stderr_text}\n{stdout_text}".lower()
    return any(
        token in haystack
        for token in (
            "auth",
            "login",
            "not authenticated",
            "authentication",
            "sandbox",
            "permission denied",
            "operation not permitted",
            "not trusted",
            "approval",
        )
    )


def _decode_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decoded_tail(value: Any, *, limit: int = 1000) -> str:
    return _decode_process_text(value)[-limit:]


def _receipt_parse_issues(receipt: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    action = str(receipt.get("action") or "").upper()
    if action not in GUARDIAN_ACTIONS:
        issues.append({"code": "BAD_ACTION", "message": "action must be exactly one supported guardian action"})
    required = (
        "event_id",
        "new_information",
        "pair",
        "side",
        "thesis_state",
        "reason",
        "harvest_trigger",
        "margin_state",
        "gateway_required",
        "no_direct_oanda",
    )
    for field in required:
        if field not in receipt:
            issues.append({"code": "FIELD_MISSING", "message": f"missing {field}"})
    if "invalidation" not in receipt and "invalidation_evidence" not in receipt:
        issues.append({"code": "FIELD_MISSING", "message": "missing invalidation evidence"})
    if "ownership" not in receipt and "manual_system_ownership" not in receipt:
        issues.append({"code": "FIELD_MISSING", "message": "missing manual/system ownership"})
    if "new_information" in receipt and not isinstance(receipt.get("new_information"), bool):
        issues.append({"code": "BAD_NEW_INFORMATION", "message": "new_information must be boolean"})
    if receipt.get("gateway_required") is not True:
        issues.append({"code": "GATEWAY_REQUIRED", "message": "gateway_required must be true"})
    if receipt.get("no_direct_oanda") is not True:
        issues.append({"code": "NO_DIRECT_OANDA_REQUIRED", "message": "no_direct_oanda must be true"})
    thesis_state = str(receipt.get("thesis_state") or "").upper()
    if thesis_state and thesis_state not in THESIS_STATES:
        issues.append({"code": "BAD_THESIS_STATE", "message": "thesis_state is not supported"})
    ownership = str(receipt.get("ownership") or receipt.get("manual_system_ownership") or "").upper()
    if ownership and ownership not in {"SYSTEM", "OPERATOR_MANUAL", "UNKNOWN"}:
        issues.append({"code": "BAD_OWNERSHIP", "message": "ownership must be SYSTEM, OPERATOR_MANUAL, or UNKNOWN"})
    side = str(receipt.get("side") or "").upper()
    if side and side not in {"LONG", "SHORT", "NONE", "N/A"}:
        issues.append({"code": "BAD_SIDE", "message": "side must be LONG, SHORT, or NONE"})
    return issues


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            found.append(payload)
    return found[-1] if found else None


def _events_from_payload(payload: dict[str, Any]) -> list[GuardianEvent]:
    events = []
    for item in payload.get("events", []) or []:
        if not isinstance(item, dict):
            continue
        fields = {
            "event_id": str(item.get("event_id") or ""),
            "event_type": str(item.get("event_type") or ""),
            "pair": str(item.get("pair") or ""),
            "direction": item.get("direction"),
            "thesis": str(item.get("thesis") or ""),
            "price_zone": str(item.get("price_zone") or ""),
            "severity": str(item.get("severity") or "P2"),
            "recommended_review_type": str(item.get("recommended_review_type") or ""),
            "dedupe_key": str(item.get("dedupe_key") or ""),
            "action_hint": str(item.get("action_hint") or ""),
            "thesis_state": str(item.get("thesis_state") or "UNKNOWN"),
            "detected_at_utc": str(item.get("detected_at_utc") or ""),
            "details": item.get("details") if isinstance(item.get("details"), dict) else {},
        }
        try:
            events.append(GuardianEvent(**fields))
        except TypeError:
            continue
    return events


def _maybe_gateway_handoff(
    *,
    receipt_review: dict[str, Any] | None,
    paths: DispatcherPaths,
    env: dict[str, str],
    lock: dict[str, Any],
) -> dict[str, Any]:
    enabled = env.get("QR_GUARDIAN_WAKE_GATEWAY_HANDOFF", "0") == "1"
    base = {
        "enabled": enabled,
        "default_off_env": "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0",
        "required_gateway": "LiveOrderGateway",
        "required_revalidators": ["guardian-action-cycle", "RiskEngine", "LiveOrderGateway"],
    }
    if not enabled:
        return {**base, "status": "SKIPPED_DEFAULT_OFF"}
    if receipt_review is None or receipt_review.get("status") != "ACCEPTED":
        return {**base, "status": "SKIPPED_RECEIPT_NOT_ACCEPTED"}
    receipt = receipt_review.get("receipt") if isinstance(receipt_review.get("receipt"), dict) else {}
    if receipt.get("gateway_required") is not True:
        return {**base, "status": "SKIPPED_GATEWAY_REQUIRED_FALSE"}
    if receipt.get("new_information") is not True:
        return {**base, "status": "SKIPPED_NO_NEW_INFORMATION"}
    reason = str(receipt.get("reason") or "").lower()
    if "scheduled hour" in reason or "hourly" in reason or receipt.get("schedule_only") is True:
        return {**base, "status": "SKIPPED_SCHEDULED_HOUR_ONLY"}
    if lock.get("active"):
        return {**base, "status": "SKIPPED_LOCK_CONFLICT", "lock": lock}
    live_flags = {
        "QR_LIVE_ENABLED": env.get("QR_LIVE_ENABLED", "0"),
        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": env.get("QR_GUARDIAN_WAKE_GATEWAY_HANDOFF", "0"),
        "QR_GUARDIAN_ACTION_EXECUTE": env.get("QR_GUARDIAN_ACTION_EXECUTE", "0"),
    }
    if live_flags["QR_GUARDIAN_ACTION_EXECUTE"] != "1":
        return {**base, "status": "SKIPPED_ACTION_EXECUTE_DISABLED", "live_flags": live_flags}
    if any(value != "1" for value in live_flags.values()):
        return {**base, "status": "SKIPPED_LIVE_FLAGS_DISABLED", "live_flags": live_flags}
    python_bin = env.get("QR_PYTHON", "python3")
    cmd = [python_bin, "-m", "quant_rabbit.cli", "guardian-action-cycle"]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    try:
        proc = subprocess.run(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=180, env=run_env)
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "ACTION_CYCLE_EXCEPTION", "live_flags": live_flags, "command": cmd, "error": str(exc)}
    return {
        **base,
        "status": "ACTION_CYCLE_CALLED" if proc.returncode == 0 else "ACTION_CYCLE_FAILED",
        "live_flags": live_flags,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def _write_action_review(
    path: Path,
    payload: dict[str, Any],
    *,
    action_receipt_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    clock = _utc(now)
    latest_receipt = _latest_accepted_receipt_payload(action_receipt_path)
    receipt_written = bool(payload.get("receipt_written", False))
    receipt_exists = bool(latest_receipt)
    receipt_reason = _action_receipt_existence_reason(payload, latest_receipt=latest_receipt)
    lines = [
        "# Guardian Action Review",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Dispatcher status: `{payload.get('status')}`",
        f"- Model: `{payload.get('model') or MODEL}`",
        f"- Receipt exists: `{'yes' if receipt_exists else 'no'}`",
        f"- Receipt reason: {receipt_reason}",
        f"- Receipt written: `{receipt_written}`",
        f"- Action receipt path: `{str(action_receipt_path) if receipt_exists and action_receipt_path else payload.get('action_receipt_path') or 'none'}`",
        "",
        "## Latest Accepted Receipt",
        "",
    ]
    if latest_receipt:
        receipt = latest_receipt.get("receipt") if isinstance(latest_receipt.get("receipt"), dict) else latest_receipt
        lines.extend(
            [
                f"- Receipt status: `{latest_receipt.get('receipt_status') or latest_receipt.get('status')}`",
                f"- Lifecycle: `{latest_receipt.get('receipt_lifecycle') or 'ACTIVE'}`",
                f"- Action: `{receipt.get('action') or 'none'}`",
                f"- Event id: `{latest_receipt.get('selected_event_id') or receipt.get('event_id') or 'none'}`",
                f"- Dedupe key: `{latest_receipt.get('selected_event_dedupe_key') or receipt.get('dedupe_key') or 'none'}`",
                f"- Generated at: `{latest_receipt.get('generated_at_utc') or 'none'}`",
                f"- Expires at: `{latest_receipt.get('expires_at_utc') or 'none'}`",
                f"- Consumed by trader: `{latest_receipt.get('consumed_by_trader', False)}`",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Selected Event",
            "",
        ]
    )
    event = payload.get("selected_event") if isinstance(payload.get("selected_event"), dict) else {}
    if event:
        lines.extend(
            [
                f"- Event id: `{event.get('event_id')}`",
                f"- Dedupe key: `{event.get('dedupe_key')}`",
                f"- Type: `{event.get('event_type')}`",
                f"- Pair / side: `{event.get('pair')}` / `{event.get('direction') or event.get('side') or 'N/A'}`",
                f"- Severity: `{event.get('severity')}`",
                f"- Wake reasons: `{', '.join(event.get('wake_reason_codes') or []) or 'none'}`",
            ]
        )
    else:
        lines.append("- none")
    preflight = payload.get("codex_preflight") if isinstance(payload.get("codex_preflight"), dict) else {}
    if preflight:
        lines.extend(
            [
                "",
                "## Codex Preflight",
                "",
                f"- Status: `{preflight.get('status')}` enabled=`{preflight.get('enabled')}`",
                f"- Binary: `{preflight.get('codex_binary_path')}`",
                f"- Version: `{preflight.get('codex_version') or 'unknown'}`",
                f"- Requested model: `{preflight.get('requested_model') or MODEL}`",
                f"- Supports requested model: `{preflight.get('supports_requested_model')}`",
            ]
        )
        if preflight.get("remediation_hint"):
            lines.append(f"- Remediation: {preflight.get('remediation_hint')}")
    attempts = payload.get("codex_attempts") if isinstance(payload.get("codex_attempts"), list) else []
    codex_single = payload.get("codex") if isinstance(payload.get("codex"), dict) else {}
    if attempts or codex_single:
        lines.extend(["", "## Codex Diagnostics", ""])
        for index, attempt in enumerate(attempts or [codex_single], start=1):
            if not isinstance(attempt, dict):
                continue
            lines.extend(
                [
                    f"- Attempt {index}: `{attempt.get('attempt') or 'unknown'}` status=`{attempt.get('status')}` returncode=`{attempt.get('returncode')}`",
                    f"  - binary: `{attempt.get('codex_binary_path') or 'unknown'}` version: `{attempt.get('codex_version') or 'unknown'}` requested model: `{attempt.get('requested_model') or MODEL}`",
                    f"  - output path: `{attempt.get('output_path')}`",
                    f"  - explicit output path: `{attempt.get('explicit_output_path')}`",
                    f"  - session JSONL: `{attempt.get('session_jsonl_path') or 'none'}`",
                    f"  - last-message source: `{attempt.get('last_message_source') or 'none'}`",
                    f"  - last-message empty: `{attempt.get('last_message_file_empty')}` explicit empty: `{attempt.get('explicit_output_file_empty')}` stdout fallback: `{attempt.get('stdout_fallback_used')}` session fallback: `{attempt.get('session_jsonl_fallback_used')}`",
                ]
            )
            stdout_excerpt = str(attempt.get("raw_stdout_excerpt") or attempt.get("stdout_tail") or "").strip()
            stderr_excerpt = str(attempt.get("raw_stderr_excerpt") or attempt.get("stderr_tail") or "").strip()
            if stdout_excerpt:
                lines.append(f"  - stdout excerpt: `{stdout_excerpt[:500]}`")
            if stderr_excerpt:
                lines.append(f"  - stderr excerpt: `{stderr_excerpt[:500]}`")
            if attempt.get("remediation_hint"):
                lines.append(f"  - remediation: {attempt.get('remediation_hint')}")
    lines.extend(["", "## Parse", ""])
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    if parse:
        lines.append(f"- Valid: `{parse.get('valid')}`")
        if parse.get("error"):
            lines.append(f"- Error: `{parse.get('error')}`")
        for issue in parse.get("issues", []) or []:
            lines.append(f"- `{issue.get('code')}` {issue.get('message')}")
    else:
        lines.append("- no Codex output parsed")
    review = payload.get("receipt_review") if isinstance(payload.get("receipt_review"), dict) else {}
    if review:
        lines.extend(["", "## Receipt Review", "", f"- Status: `{review.get('status')}`"])
        receipt = review.get("receipt") if isinstance(review.get("receipt"), dict) else {}
        lines.append(f"- Action: `{receipt.get('action') or 'none'}`")
        lines.append(f"- Event id: `{receipt.get('event_id') or 'none'}`")
        lines.append(f"- Pair / side: `{receipt.get('pair') or 'none'}` / `{receipt.get('side') or 'none'}`")
        lines.append(f"- Dedupe key: `{receipt.get('dedupe_key') or 'none'}`")
        lines.append(f"- Receipt id: `{receipt.get('receipt_id') or review.get('generated_at_utc') or 'none'}`")
        lines.append(f"- Dispatcher status: `{payload.get('status')}`")
        for issue in review.get("issues", []) or []:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    handoff = payload.get("gateway_handoff") if isinstance(payload.get("gateway_handoff"), dict) else {}
    if handoff:
        lines.extend(["", "## Gateway Handoff", "", f"- Status: `{handoff.get('status')}`"])
        lines.append(f"- Required gateway: `{handoff.get('required_gateway')}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Guardian wake dispatcher never calls OANDA.",
            "- Codex runs read-only and writes at most a review/receipt artifact.",
            "- Live orders, cancels, and closes remain gateway-only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _action_receipt_existence_reason(payload: dict[str, Any], *, latest_receipt: dict[str, Any] | None = None) -> str:
    if payload.get("receipt_written"):
        return "accepted schema-valid receipt is bound to selected_event and was written atomically."
    if latest_receipt:
        receipt = latest_receipt.get("receipt") if isinstance(latest_receipt.get("receipt"), dict) else latest_receipt
        action = str(receipt.get("action") or "UNKNOWN").upper()
        event_id = latest_receipt.get("selected_event_id") or receipt.get("event_id") or "unknown"
        return (
            f"latest accepted `{action}` receipt for event `{event_id}` remains "
            f"`{latest_receipt.get('receipt_lifecycle') or 'ACTIVE'}`; dispatcher status "
            f"`{payload.get('status') or 'UNKNOWN'}` does not invalidate it."
        )
    status = str(payload.get("status") or "")
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    review = payload.get("receipt_review") if isinstance(payload.get("receipt_review"), dict) else {}
    if parse.get("error"):
        return f"no receipt because Codex output parse/classification is `{parse.get('error')}`."
    if review.get("status"):
        codes = ", ".join(str(issue.get("code")) for issue in review.get("issues", []) or []) or "no issue codes"
        return f"no receipt because receipt review status is `{review.get('status')}` ({codes})."
    return f"no receipt because dispatcher status is `{status or 'UNKNOWN'}`."


def _receipt_expires_at(now: datetime, *, env: dict[str, str]) -> str:
    ttl = int(env.get("QR_GUARDIAN_ACTION_RECEIPT_TTL_SECONDS", DEFAULT_RECEIPT_TTL_SECONDS))
    return (now + timedelta(seconds=max(0, ttl))).isoformat()


def _receipt_is_accepted(payload: dict[str, Any]) -> bool:
    return str(payload.get("receipt_status") or payload.get("status") or "").upper() == "ACCEPTED"


def _receipt_lifecycle(payload: dict[str, Any]) -> str:
    return str(payload.get("receipt_lifecycle") or ("ACTIVE" if _receipt_is_accepted(payload) else "")).upper()


def _latest_accepted_receipt_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return None
    return payload


def _expire_current_receipt_if_needed(path: Path, *, now: datetime) -> None:
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return
    if _receipt_lifecycle(payload) in TERMINAL_RECEIPT_LIFECYCLES:
        return
    expires_at = _parse_utc(payload.get("expires_at_utc"))
    if expires_at is None or expires_at > now:
        return
    expired = {
        **payload,
        "receipt_lifecycle": "EXPIRED",
        "expired_at_utc": now.isoformat(),
        "lifecycle_reason": "guardian action receipt exceeded expires_at_utc",
    }
    _archive_receipt(path, expired)
    _remove_file(path)


def _supersede_current_receipt(path: Path, *, superseded_by_event_id: str, now: datetime) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return None
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    current_event_id = str(payload.get("selected_event_id") or receipt.get("event_id") or "")
    superseded = {
        **payload,
        "receipt_lifecycle": "SUPERSEDED",
        "superseded_by_event_id": superseded_by_event_id or None,
        "superseded_at_utc": now.isoformat(),
        "consumed_by_trader": bool(payload.get("consumed_by_trader", False)),
    }
    _archive_receipt(path, superseded)
    return {
        "event_id": current_event_id or None,
        "action": receipt.get("action"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "receipt_lifecycle": "SUPERSEDED",
        "archive_path": _archive_path(path, superseded).name,
    }


def _remove_non_accepted_current_receipt(path: Path) -> None:
    payload = _load_json(path)
    if payload and _receipt_is_accepted(payload):
        return
    _remove_file(path)


def _archive_receipt(current_path: Path, payload: dict[str, Any]) -> None:
    archive_path = _archive_path(current_path, payload)
    _write_json(archive_path, payload)


def _archive_path(current_path: Path, payload: dict[str, Any]) -> Path:
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    generated = str(payload.get("generated_at_utc") or datetime.now(timezone.utc).isoformat())
    event_id = str(payload.get("selected_event_id") or receipt.get("event_id") or "unknown")
    lifecycle = str(payload.get("receipt_lifecycle") or "ACTIVE")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{generated}_{event_id}_{lifecycle}").strip("_")
    return current_path.parent / "guardian_action_receipts" / f"{safe}.json"


def _maybe_write_tuning_work_order(
    *,
    path: Path,
    selected_event: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            observation_id = _event_material_fingerprint(selected_event)
            semantic_state_id = _event_semantic_state_id(selected_event)
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "work_order_id": f"guardian-tuning-{semantic_state_id[:20]}",
                "path": str(path),
                "event_fingerprint": observation_id,
                "semantic_state_id": semantic_state_id,
                "observation_id": observation_id,
                "error": "guardian tuning queue lock is held by another lifecycle writer",
                "retry_required": True,
            }
        return _maybe_write_tuning_work_order_locked(
            path=path,
            selected_event=selected_event,
            receipt=receipt,
            now=now,
        )


def transition_tuning_work_order(
    *,
    path: Path,
    work_order_id: str,
    expected_observation_id: str,
    status: str,
    consumed_by: str,
    experiment_id: str,
    experiment_result: str,
    experiment_evidence_ref: str,
    now: datetime,
) -> dict[str, Any]:
    """Atomically complete one reviewed hourly-AI tuning obligation.

    This is the only supported writer for hourly-AI lifecycle transitions. It
    shares the dispatcher's stable lock and CAS path, so neither side can
    overwrite the other's whole-file queue update.
    """

    normalized_status = str(status or "").strip().upper()
    string_fields = {
        "work_order_id": str(work_order_id or "").strip(),
        "expected_observation_id": str(expected_observation_id or "").strip(),
        "consumed_by": str(consumed_by or "").strip(),
        "experiment_id": str(experiment_id or "").strip(),
        "experiment_result": str(experiment_result or "").strip(),
        "experiment_evidence_ref": str(experiment_evidence_ref or "").strip(),
    }
    invalid_fields = [
        key
        for key, value in string_fields.items()
        if not value
        or len(value)
        > (
            2000
            if key == "experiment_result"
            else 1024
            if key == "experiment_evidence_ref"
            else 256
        )
    ]
    if normalized_status not in {"CONSUMED", "SUPERSEDED"} or invalid_fields:
        return {
            "status": "INVALID_TERMINAL_TRANSITION",
            "path": str(path),
            "invalid_fields": invalid_fields,
            "terminal_status": normalized_status or None,
        }
    if now.tzinfo is None:
        return {
            "status": "INVALID_TERMINAL_TRANSITION",
            "path": str(path),
            "invalid_fields": ["now"],
            "terminal_status": normalized_status,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "path": str(path),
                "work_order_id": string_fields["work_order_id"],
                "retry_required": True,
                "error": "guardian tuning queue lock is held by another lifecycle writer",
            }
        return _transition_tuning_work_order_locked(
            path=path,
            work_order_id=string_fields["work_order_id"],
            expected_observation_id=string_fields["expected_observation_id"],
            status=normalized_status,
            consumed_by=string_fields["consumed_by"],
            experiment_id=string_fields["experiment_id"],
            experiment_result=string_fields["experiment_result"],
            experiment_evidence_ref=string_fields["experiment_evidence_ref"],
            now=now,
        )


def _apply_tuning_work_order_review(
    *,
    path: Path,
    pending_entries: list[dict[str, Any]],
    work_order_id: str,
    expected_observation_id: str,
    review: dict[str, Any],
    reviewed_by: str,
    now: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Validate and apply one monotonic review to an in-memory queue."""

    normalized_work_order_id = str(work_order_id or "").strip()
    normalized_observation_id = str(expected_observation_id or "").strip()
    matches = [
        (index, item)
        for index, item in enumerate(pending_entries)
        if str(item.get("work_order_id") or "").strip()
        == normalized_work_order_id
    ]
    if len(matches) != 1:
        return (
            {
                "status": "WORK_ORDER_NOT_FOUND",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "match_count": len(matches),
            },
            pending_entries,
            None,
        )
    match_index, matched = matches[0]
    current = dict(matched)
    latest_observation_id = str(
        current.get("latest_observation_id")
        or current.get("observation_id")
        or current.get("event_fingerprint")
        or ""
    ).strip()
    if latest_observation_id != normalized_observation_id:
        return (
            {
                "status": "WORK_ORDER_OBSERVATION_STALE",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "latest_observation_id": latest_observation_id or None,
                "expected_observation_id": normalized_observation_id,
                "retry_required": True,
            },
            pending_entries,
            current,
        )
    selected_event = (
        current.get("selected_event")
        if isinstance(current.get("selected_event"), dict)
        else {}
    )
    validation = _validate_bot_tuning_review(review, selected_event=selected_event)
    normalized_review = (
        validation.get("review")
        if isinstance(validation.get("review"), dict)
        else {}
    )
    incoming_status = str(normalized_review.get("review_status") or "").upper()
    if (
        validation.get("status") != "VALID"
        or incoming_status
        not in {"TEST_REQUIRED", "NO_CHANGE_INSUFFICIENT_EVIDENCE"}
    ):
        return (
            {
                "status": "STRUCTURED_REVIEW_REQUIRED",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "bot_tuning_review_status": validation.get("status") or "MISSING",
                "issues": validation.get("issues") or [],
                "retry_required": True,
            },
            pending_entries,
            current,
        )
    current_review = (
        current.get("bot_tuning_review")
        if isinstance(current.get("bot_tuning_review"), dict)
        else {}
    )
    current_status = str(current_review.get("review_status") or "").upper()
    current_review_bound_to_latest = (
        str(current.get("latest_reviewed_observation_id") or "")
        == latest_observation_id
    )
    same_review = bool(
        current_status == incoming_status
        and current_status
        in {"TEST_REQUIRED", "NO_CHANGE_INSUFFICIENT_EVIDENCE"}
        and _tuning_review_digest(current_review)
        == _tuning_review_digest(normalized_review)
        and current_review_bound_to_latest
    )
    if same_review:
        return (
            {
                "status": "WORK_ORDER_REVIEW_ALREADY_BOUND",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "observation_id": latest_observation_id,
            },
            pending_entries,
            current,
        )
    current_validation = (
        current.get("bot_tuning_review_validation")
        if isinstance(current.get("bot_tuning_review_validation"), dict)
        else {}
    )
    if (
        current_status == "TEST_REQUIRED"
        and incoming_status == "NO_CHANGE_INSUFFICIENT_EVIDENCE"
        and current_review_bound_to_latest
    ):
        return (
            {
                "status": "WORK_ORDER_REVIEW_CONFLICT",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "current_review_status": current_status,
                "incoming_review_status": incoming_status,
                "retry_required": True,
            },
            pending_entries,
            current,
        )
    if (
        str(current_validation.get("status") or "").upper() == "VALID"
        and current_review_bound_to_latest
        and current.get("review_reacquisition_required_after_abort") is not True
        and not (
            current_status == "NO_CHANGE_INSUFFICIENT_EVIDENCE"
            and incoming_status == "TEST_REQUIRED"
        )
    ):
        return (
            {
                "status": "WORK_ORDER_REVIEW_CONFLICT",
                "path": str(path),
                "work_order_id": normalized_work_order_id,
                "current_review_status": current_status,
                "incoming_review_status": incoming_status,
                "retry_required": True,
            },
            pending_entries,
            current,
        )
    updated = {
        **current,
        "bot_tuning_review": normalized_review,
        "bot_tuning_review_validation": {
            key: value for key, value in validation.items() if key != "review"
        },
        "structured_review_completed_at_utc": now.astimezone(
            timezone.utc
        ).isoformat(),
        "structured_review_completed_by": str(reviewed_by or "").strip(),
        "latest_reviewed_observation_id": latest_observation_id,
    }
    updated_pending = list(pending_entries)
    updated_pending[match_index] = updated
    return (
        {
            "status": "WORK_ORDER_REVIEW_ENRICHED",
            "path": str(path),
            "work_order_id": normalized_work_order_id,
            "observation_id": latest_observation_id,
            "review_digest_sha256": _tuning_review_digest(normalized_review),
        },
        updated_pending,
        updated,
    )


def enrich_tuning_work_order_review(
    *,
    path: Path,
    work_order_id: str,
    expected_observation_id: str,
    review: dict[str, Any],
    reviewed_by: str,
    now: datetime,
) -> dict[str, Any]:
    """Monotonically bind a current observation to one safe structured review.

    A conservative ``NO_CHANGE_INSUFFICIENT_EVIDENCE`` review is durable work:
    it records the exact evidence acquisition step without pretending that an
    experiment is ready.  The same review is idempotent, and the only allowed
    status change for one pending work order is the monotonic
    ``NO_CHANGE_INSUFFICIENT_EVIDENCE`` -> ``TEST_REQUIRED`` upgrade.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "path": str(path),
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        existing = _load_tuning_work_order(path)
        if existing.get("_read_error"):
            return {
                "status": "WORK_ORDER_READ_FAILED",
                "path": str(path),
                "work_order_id": work_order_id,
                "error": existing.get("_read_error"),
                "retry_required": True,
            }
        expected_source_sha256 = existing.get("_queue_source_sha256")
        pending, terminal_history = _normalized_tuning_work_order_queue(existing)
        result, updated_pending, updated = _apply_tuning_work_order_review(
            path=path,
            pending_entries=pending,
            work_order_id=work_order_id,
            expected_observation_id=expected_observation_id,
            review=review,
            reviewed_by=reviewed_by,
            now=now,
        )
        if result.get("status") != "WORK_ORDER_REVIEW_ENRICHED":
            return result
        assert updated is not None
        payload = _tuning_queue_payload(
            primary=updated,
            pending_entries=updated_pending,
            terminal_history=terminal_history,
            experiment_digest_history=_tuning_experiment_digest_history(existing),
            experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
            override_lifecycle_heads=_tuning_override_lifecycle_heads(existing),
        )
        try:
            _write_tuning_queue_json(
                path,
                payload,
                expected_source_sha256=expected_source_sha256,
            )
        except OSError as exc:
            return _tuning_work_order_write_failure(
                path=path,
                work_order_id=work_order_id,
                observation_id=str(result.get("observation_id") or ""),
                semantic_state_id=_work_order_semantic_state_id(updated),
                exc=exc,
            )
        return result


def enrich_tuning_work_order_reviews_batch(
    *,
    path: Path,
    reviews: list[dict[str, Any]],
    reviewed_by: str,
    now: datetime,
) -> dict[str, Any]:
    """Atomically bind a bounded set of current-observation reviews.

    Every item is revalidated against queue bytes loaded under the stable queue
    lock. All mutations remain in memory until one whole-queue CAS write, so a
    stale/conflicting review or a failed write cannot partially bind the batch.
    """

    requested_count = len(reviews) if isinstance(reviews, list) else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "path": str(path),
                "requested_count": requested_count,
                "prevalidated_count": 0,
                "success_count": 0,
                "enriched_count": 0,
                "already_bound_count": 0,
                "failure_count": 1,
                "written_count": 0,
                "queue_write_count": 0,
                "failures": [
                    {
                        "status": "WORK_ORDER_CONCURRENT_UPDATE",
                        "error": "guardian tuning queue lock is held by another lifecycle writer",
                        "retry_required": True,
                    }
                ],
                "results": [],
                "retry_required": True,
            }

        structural_failures: list[dict[str, Any]] = []
        structured: list[dict[str, Any]] = []
        if not isinstance(reviews, list) or not reviews:
            structural_failures.append({"code": "MANIFEST_REVIEWS_REQUIRED"})
        elif len(reviews) > MAX_PENDING_TUNING_WORK_ORDERS:
            structural_failures.append(
                {
                    "code": "MANIFEST_REVIEW_COUNT_EXCEEDED",
                    "review_count": len(reviews),
                    "max_review_count": MAX_PENDING_TUNING_WORK_ORDERS,
                }
            )
        normalized_reviewer = str(reviewed_by or "").strip()
        if not normalized_reviewer or len(normalized_reviewer) > 256:
            structural_failures.append({"code": "REVIEWED_BY_INVALID"})
        if now.tzinfo is None:
            structural_failures.append({"code": "REVIEWED_AT_INVALID"})
        seen_work_order_ids: set[str] = set()
        seen_observation_ids: set[str] = set()
        for index, item in enumerate(reviews if isinstance(reviews, list) else []):
            if not isinstance(item, dict):
                structural_failures.append(
                    {"index": index, "code": "MANIFEST_ITEM_NOT_OBJECT"}
                )
                continue
            unexpected = sorted(set(item) - TUNING_REVIEW_BATCH_ITEM_FIELDS)
            missing = sorted(TUNING_REVIEW_BATCH_ITEM_FIELDS - set(item))
            work_order_id = str(item.get("work_order_id") or "").strip()
            observation_id = str(item.get("expected_observation_id") or "").strip()
            if unexpected or missing:
                structural_failures.append(
                    {
                        "index": index,
                        "code": "MANIFEST_ITEM_FIELDS_INVALID",
                        "work_order_id": work_order_id or None,
                        "unexpected_fields": unexpected,
                        "missing_fields": missing,
                    }
                )
                continue
            if (
                not work_order_id
                or not observation_id
                or len(work_order_id) > 256
                or len(observation_id) > 256
            ):
                structural_failures.append(
                    {
                        "index": index,
                        "code": "MANIFEST_ITEM_ID_INVALID",
                        "work_order_id": work_order_id or None,
                    }
                )
                continue
            if work_order_id in seen_work_order_ids:
                structural_failures.append(
                    {
                        "index": index,
                        "code": "DUPLICATE_WORK_ORDER_ID",
                        "work_order_id": work_order_id,
                    }
                )
                continue
            if observation_id in seen_observation_ids:
                structural_failures.append(
                    {
                        "index": index,
                        "code": "DUPLICATE_OBSERVATION_ID",
                        "work_order_id": work_order_id,
                        "expected_observation_id": observation_id,
                    }
                )
                continue
            seen_work_order_ids.add(work_order_id)
            seen_observation_ids.add(observation_id)
            structured.append(
                {
                    "index": index,
                    "work_order_id": work_order_id,
                    "expected_observation_id": observation_id,
                    "review": item.get("review"),
                }
            )
        if structural_failures:
            return {
                "status": "BATCH_MANIFEST_VALIDATION_FAILED",
                "path": str(path),
                "requested_count": requested_count,
                "prevalidated_count": 0,
                "success_count": 0,
                "enriched_count": 0,
                "already_bound_count": 0,
                "failure_count": len(structural_failures),
                "written_count": 0,
                "queue_write_count": 0,
                "failures": structural_failures,
                "results": [],
            }

        existing = _load_tuning_work_order(path)
        if existing.get("_read_error"):
            failure = {
                "status": "WORK_ORDER_READ_FAILED",
                "path": str(path),
                "error": existing.get("_read_error"),
                "retry_required": True,
            }
            return {
                "status": "WORK_ORDER_READ_FAILED",
                "path": str(path),
                "requested_count": requested_count,
                "prevalidated_count": 0,
                "success_count": 0,
                "enriched_count": 0,
                "already_bound_count": 0,
                "failure_count": 1,
                "written_count": 0,
                "queue_write_count": 0,
                "failures": [failure],
                "results": [],
                "retry_required": True,
            }
        expected_source_sha256 = existing.get("_queue_source_sha256")
        pending, terminal_history = _normalized_tuning_work_order_queue(existing)
        updated_pending = pending
        updated_primary: dict[str, Any] | None = None
        results: list[dict[str, Any]] = []
        validation_failures: list[dict[str, Any]] = []
        for item in structured:
            result, candidate_pending, candidate_primary = (
                _apply_tuning_work_order_review(
                    path=path,
                    pending_entries=updated_pending,
                    work_order_id=item["work_order_id"],
                    expected_observation_id=item["expected_observation_id"],
                    review=item["review"],
                    reviewed_by=normalized_reviewer,
                    now=now,
                )
            )
            status = str(result.get("status") or "")
            if status == "WORK_ORDER_REVIEW_ENRICHED":
                updated_pending = candidate_pending
                updated_primary = candidate_primary
                results.append(result)
            elif status == "WORK_ORDER_REVIEW_ALREADY_BOUND":
                results.append(result)
            else:
                validation_failures.append(
                    {"index": item["index"], **result}
                )
        if validation_failures:
            return {
                "status": "BATCH_MANIFEST_VALIDATION_FAILED",
                "path": str(path),
                "requested_count": requested_count,
                "prevalidated_count": len(results),
                "success_count": 0,
                "enriched_count": 0,
                "already_bound_count": 0,
                "failure_count": len(validation_failures),
                "written_count": 0,
                "queue_write_count": 0,
                "failures": validation_failures,
                "results": [],
            }

        enriched_count = sum(
            result.get("status") == "WORK_ORDER_REVIEW_ENRICHED"
            for result in results
        )
        already_bound_count = sum(
            result.get("status") == "WORK_ORDER_REVIEW_ALREADY_BOUND"
            for result in results
        )
        if enriched_count:
            assert updated_primary is not None
            payload = _tuning_queue_payload(
                primary=updated_primary,
                pending_entries=updated_pending,
                terminal_history=terminal_history,
                experiment_digest_history=_tuning_experiment_digest_history(existing),
                experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
                override_lifecycle_heads=_tuning_override_lifecycle_heads(existing),
            )
            try:
                _write_tuning_queue_json(
                    path,
                    payload,
                    expected_source_sha256=expected_source_sha256,
                )
            except OSError as exc:
                failure = _tuning_work_order_write_failure(
                    path=path,
                    work_order_id="BATCH",
                    observation_id=str(
                        updated_primary.get("latest_observation_id")
                        or updated_primary.get("observation_id")
                        or ""
                    ),
                    semantic_state_id=_work_order_semantic_state_id(updated_primary),
                    exc=exc,
                )
                return {
                    "status": failure["status"],
                    "path": str(path),
                    "requested_count": requested_count,
                    "prevalidated_count": len(results),
                    "success_count": 0,
                    "enriched_count": 0,
                    "already_bound_count": 0,
                    "failure_count": 1,
                    "written_count": 0,
                    "queue_write_count": 0,
                    "failures": [failure],
                    "results": [],
                    "retry_required": True,
                }
        return {
            "status": "BATCH_REVIEW_ENRICHED",
            "path": str(path),
            "requested_count": requested_count,
            "prevalidated_count": len(results),
            "success_count": len(results),
            "enriched_count": enriched_count,
            "already_bound_count": already_bound_count,
            "failure_count": 0,
            "written_count": enriched_count,
            "queue_write_count": 1 if enriched_count else 0,
            "failures": [],
            "results": results,
        }


def prepare_tuning_experiment_contract(
    *,
    path: Path,
    work_order_id: str,
    expected_observation_id: str,
    experiment_id: str,
    cohort_id: str,
    source_watermark: Any,
    sample_count: int,
    evaluator: str,
    source_data_ref: str,
    evaluator_artifact_ref: str,
    primary_metric: str,
    objective: str,
    acceptance_threshold: float,
    metric_names: list[str],
    prepared_by: str,
    now: datetime,
) -> dict[str, Any]:
    """Precommit evaluator, cohort, metric, and threshold before outcomes exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "path": str(path),
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        loaded = _load_tuning_work_order(path)
        if loaded.get("_read_error"):
            return {
                "status": "WORK_ORDER_READ_FAILED",
                "path": str(path),
                "error": loaded.get("_read_error"),
                "retry_required": True,
            }
        expected_source_sha256 = loaded.get("_queue_source_sha256")
        pending, terminal_history = _normalized_tuning_work_order_queue(loaded)
        matches = [
            item
            for item in pending
            if str(item.get("work_order_id") or "") == str(work_order_id or "")
        ]
        if len(matches) != 1:
            return {"status": "WORK_ORDER_NOT_FOUND", "work_order_id": work_order_id}
        current = dict(matches[0])
        latest_observation_id = str(
            current.get("latest_observation_id")
            or current.get("observation_id")
            or current.get("event_fingerprint")
            or ""
        )
        if latest_observation_id != str(expected_observation_id or ""):
            return {
                "status": "WORK_ORDER_OBSERVATION_STALE",
                "work_order_id": work_order_id,
                "latest_observation_id": latest_observation_id or None,
                "retry_required": True,
            }
        review = current.get("bot_tuning_review") if isinstance(current.get("bot_tuning_review"), dict) else {}
        review_validation = current.get("bot_tuning_review_validation") if isinstance(current.get("bot_tuning_review_validation"), dict) else {}
        if (
            str(review_validation.get("status") or "").upper() != "VALID"
            or str(review.get("review_status") or "").upper() != "TEST_REQUIRED"
            or str(current.get("latest_reviewed_observation_id") or "")
            != latest_observation_id
        ):
            return {
                "status": "STRUCTURED_REVIEW_REQUIRED",
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        normalized_metric_names = sorted(
            {
                str(item or "").strip()
                for item in metric_names
                if str(item or "").strip()
            }
        )
        normalized_objective = str(objective or "").strip().upper()
        normalized_acceptance_threshold = _safe_finite_float(acceptance_threshold)
        adjustments = review.get("proposed_adjustments")
        adjustment = (
            adjustments[0]
            if isinstance(adjustments, list) and len(adjustments) == 1
            else {}
        )
        adjustment_parameter = str(adjustment.get("parameter") or "").strip()
        invalid = bool(
            not str(experiment_id or "").strip()
            or not str(cohort_id or "").strip()
            or source_watermark is None
            or source_watermark == ""
            or source_watermark == {}
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or str(evaluator or "").strip() != TUNING_EVALUATOR_NAME
            or not str(prepared_by or "").strip()
            or normalized_metric_names != sorted(TUNING_EVALUATOR_METRIC_NAMES)
            or str(primary_metric or "").strip() != TUNING_EVALUATOR_PRIMARY_METRIC
            or normalized_objective != TUNING_EVALUATOR_OBJECTIVE
            or adjustment_parameter not in SUPPORTED_THRESHOLD_PARAMETERS
            or isinstance(acceptance_threshold, bool)
            or not isinstance(acceptance_threshold, (int, float))
            or normalized_acceptance_threshold is None
            or normalized_acceptance_threshold != TUNING_FIXED_ACCEPTANCE_THRESHOLD
        )
        source_data_validation = _validate_project_artifact_ref(
            queue_path=path,
            artifact_ref=source_data_ref,
            allowed_roots=("data/guardian_tuning_experiment_inputs/data",),
            max_bytes=MAX_TUNING_SOURCE_BYTES,
            require_content_addressed=True,
        )
        evaluator_validation = _validate_project_artifact_ref(
            queue_path=path,
            artifact_ref=evaluator_artifact_ref,
            allowed_roots=("data/guardian_tuning_experiment_inputs/evaluators",),
            max_bytes=MAX_TUNING_EVALUATOR_BYTES,
            require_content_addressed=True,
        )
        source_contract_validation = _validate_prepared_tuning_source(
            queue_path=path,
            source_validation=source_data_validation,
            adjustment=adjustment,
            review=review,
            review_completed_at_utc=current.get(
                "structured_review_completed_at_utc"
            ),
            cohort_id=str(cohort_id or "").strip(),
            source_watermark=source_watermark,
            sample_count=sample_count,
            require_current_tips=True,
        )
        active_parameter_binding: dict[str, Any] | None = None
        if source_contract_validation.get("status") == "VALID":
            try:
                source_identity = source_contract_validation.get("identity")
                if not isinstance(source_identity, dict):
                    raise ValueError("canonical source identity is missing")
                active_parameter_binding = runtime_forecast_floor_binding(
                    lane_id=str(source_identity.get("lane_id") or ""),
                    override_path=path.with_name("guardian_tuning_overrides.json"),
                    queue_path=path,
                )
                reviewed_current = _safe_finite_float(
                    adjustment.get("current_value")
                )
                if (
                    reviewed_current is None
                    or not math.isclose(
                        reviewed_current,
                        float(active_parameter_binding["resolved_value"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    active_parameter_binding = None
            except (KeyError, TypeError, ValueError):
                active_parameter_binding = None
        approved_evaluator_sha256 = _approved_tuning_evaluator_sha256()
        if (
            invalid
            or source_data_validation.get("status") != "VALID"
            or evaluator_validation.get("status") != "VALID"
            or source_contract_validation.get("status") != "VALID"
            or active_parameter_binding is None
            or approved_evaluator_sha256 is None
            or evaluator_validation.get("sha256") != approved_evaluator_sha256
        ):
            return {
                "status": "EXPERIMENT_CONTRACT_INVALID",
                "work_order_id": work_order_id,
                "source_data_validation": source_data_validation,
                "evaluator_validation": evaluator_validation,
                "source_contract_validation": source_contract_validation,
                "active_parameter_binding": active_parameter_binding,
                "retry_required": True,
            }
        if any(
            str(item.get("experiment_id") or "") == str(experiment_id)
            for item in terminal_history
        ) or any(
            str(
                (
                    item.get("prepared_experiment_contract")
                    if isinstance(item.get("prepared_experiment_contract"), dict)
                    else {}
                ).get("experiment_id")
                or ""
            )
            == str(experiment_id)
            for item in pending
            if str(item.get("work_order_id") or "") != str(work_order_id or "")
        ):
            return {
                "status": "EXPERIMENT_ID_ALREADY_USED",
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "retry_required": True,
            }
        material = {
            "semantic_state_id": _work_order_semantic_state_id(current),
            "review_digest_sha256": _tuning_review_digest(review),
            "cohort_id": str(cohort_id).strip(),
            "source_watermark": source_watermark,
            "sample_count": sample_count,
            "evaluator": str(evaluator).strip(),
            "source_data_ref": source_data_validation.get("ref"),
            "evaluator_artifact_ref": evaluator_validation.get("ref"),
            "primary_metric": str(primary_metric).strip(),
            "objective": normalized_objective,
            "acceptance_threshold": normalized_acceptance_threshold,
            "metric_names": normalized_metric_names,
            "active_parameter_binding": active_parameter_binding,
            "source_identity": source_contract_validation.get("identity"),
        }
        digest = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        semantic_digest = _tuning_experiment_semantic_digest(
            adjustment=adjustment,
            source_semantic_digest=source_contract_validation.get(
                "source_semantic_digest"
            ),
            evaluator_artifact_sha256=evaluator_validation.get("sha256"),
            acceptance_threshold=normalized_acceptance_threshold,
        )
        if semantic_digest is None:
            return {
                "status": "EXPERIMENT_CONTRACT_INVALID",
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        digest_history = _tuning_experiment_digest_history(loaded)
        if semantic_digest in digest_history or any(
            str(item.get("experiment_semantic_digest") or "") == semantic_digest
            for item in terminal_history
        ):
            return {
                "status": "EXPERIMENT_ALREADY_EVALUATED",
                "work_order_id": work_order_id,
                "experiment_contract_digest": digest,
                "experiment_semantic_digest": semantic_digest,
                "retry_required": True,
            }
        existing_contract = (
            current.get("prepared_experiment_contract")
            if isinstance(current.get("prepared_experiment_contract"), dict)
            else {}
        )
        if existing_contract:
            if (
                str(existing_contract.get("experiment_id") or "") == experiment_id
                and str(existing_contract.get("experiment_contract_digest") or "") == digest
            ):
                return {
                    "status": "EXPERIMENT_CONTRACT_ALREADY_PREPARED",
                    "work_order_id": work_order_id,
                    "experiment_contract_digest": digest,
                    "prepared_experiment_contract": existing_contract,
                }
            return {
                "status": "EXPERIMENT_CONTRACT_CONFLICT",
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        contract = {
            **material,
            "status": "PREPARED",
            "experiment_id": str(experiment_id).strip(),
            "observation_id": latest_observation_id,
            "experiment_contract_digest": digest,
            "experiment_semantic_digest": semantic_digest,
            "source_semantic_digest": source_contract_validation.get(
                "source_semantic_digest"
            ),
            "evaluator_artifact_sha256": evaluator_validation.get("sha256"),
            "prepared_at_utc": now.astimezone(timezone.utc).isoformat(),
            "prepared_by": str(prepared_by).strip(),
        }
        updated = {**current, "prepared_experiment_contract": contract}
        updated.pop("review_reacquisition_required_after_abort", None)
        updated_pending = [
            updated
            if str(item.get("work_order_id") or "") == str(work_order_id or "")
            else item
            for item in pending
        ]
        payload = _tuning_queue_payload(
            primary=updated,
            pending_entries=updated_pending,
            terminal_history=terminal_history,
            experiment_digest_history=_tuning_experiment_digest_history(loaded),
            experiment_id_digest_history=_tuning_experiment_id_digest_history(loaded),
            override_lifecycle_heads=_tuning_override_lifecycle_heads(loaded),
        )
        try:
            _write_tuning_queue_json(
                path,
                payload,
                expected_source_sha256=expected_source_sha256,
            )
        except OSError as exc:
            return _tuning_work_order_write_failure(
                path=path,
                work_order_id=work_order_id,
                observation_id=latest_observation_id,
                semantic_state_id=_work_order_semantic_state_id(current),
                exc=exc,
            )
        return {
            "status": "EXPERIMENT_CONTRACT_PREPARED",
            "work_order_id": work_order_id,
            "experiment_contract_digest": digest,
            "prepared_experiment_contract": contract,
        }


def abort_tuning_experiment_contract(
    *,
    path: Path,
    work_order_id: str,
    expected_observation_id: str,
    experiment_id: str,
    aborted_by: str,
    reason: str,
    failure_evidence_ref: str,
    now: datetime,
) -> dict[str, Any]:
    """Close a crashed prepared attempt without losing its no-repeat digest."""

    if not str(aborted_by or "").strip() or not str(reason or "").strip():
        return {"status": "EXPERIMENT_ABORT_INVALID", "retry_required": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "WORK_ORDER_CONCURRENT_UPDATE",
                "path": str(path),
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        loaded = _load_tuning_work_order(path)
        if loaded.get("_read_error"):
            return {
                "status": "WORK_ORDER_READ_FAILED",
                "path": str(path),
                "error": loaded.get("_read_error"),
                "retry_required": True,
            }
        expected_source_sha256 = loaded.get("_queue_source_sha256")
        pending, terminal_history = _normalized_tuning_work_order_queue(loaded)
        matches = [
            item
            for item in pending
            if str(item.get("work_order_id") or "") == str(work_order_id or "")
        ]
        if len(matches) != 1:
            return {"status": "WORK_ORDER_NOT_FOUND", "work_order_id": work_order_id}
        current = dict(matches[0])
        observation_id = str(
            current.get("latest_observation_id")
            or current.get("observation_id")
            or current.get("event_fingerprint")
            or ""
        )
        if observation_id != str(expected_observation_id or ""):
            return {
                "status": "WORK_ORDER_OBSERVATION_STALE",
                "work_order_id": work_order_id,
                "latest_observation_id": observation_id,
                "retry_required": True,
            }
        prepared = (
            current.get("prepared_experiment_contract")
            if isinstance(current.get("prepared_experiment_contract"), dict)
            else {}
        )
        archived = [
            dict(item)
            for item in current.get("aborted_experiment_contracts", []) or []
            if isinstance(item, dict)
        ]
        if not prepared:
            prior = next(
                (
                    item
                    for item in archived
                    if str(item.get("experiment_id") or "") == experiment_id
                    and str(item.get("failure_evidence_ref") or "")
                    == failure_evidence_ref
                ),
                None,
            )
            return {
                "status": (
                    "EXPERIMENT_CONTRACT_ALREADY_ABORTED"
                    if prior is not None
                    else "EXPERIMENT_CONTRACT_REQUIRED"
                ),
                "work_order_id": work_order_id,
                "retry_required": prior is None,
            }
        if (
            str(prepared.get("status") or "") != "PREPARED"
            or str(prepared.get("experiment_id") or "") != experiment_id
            or str(prepared.get("observation_id") or "") != observation_id
        ):
            return {
                "status": "EXPERIMENT_CONTRACT_CONFLICT",
                "work_order_id": work_order_id,
                "retry_required": True,
            }
        validation = _validate_project_artifact_ref(
            queue_path=path,
            artifact_ref=failure_evidence_ref,
            allowed_roots=("data/guardian_tuning_experiment_failures",),
            max_bytes=MAX_TUNING_EVIDENCE_BYTES,
            require_content_addressed=True,
        )
        if validation.get("status") != "VALID":
            return {
                "status": "EXPERIMENT_ABORT_EVIDENCE_INVALID",
                "validation": validation,
                "retry_required": True,
            }
        raw, error = _bounded_file_bytes(
            Path(str(validation["path"])),
            max_bytes=MAX_TUNING_EVIDENCE_BYTES,
        )
        try:
            failure = json.loads((raw or b"").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            failure = None
        semantic_digest = str(prepared.get("experiment_semantic_digest") or "")
        expected_failure = {
            "schema_version": 1,
            "status": "ABORTED",
            "work_order_id": work_order_id,
            "observation_id": observation_id,
            "experiment_id": experiment_id,
            "experiment_contract_digest": prepared.get("experiment_contract_digest"),
            "experiment_semantic_digest": semantic_digest,
            "aborted_by": str(aborted_by).strip(),
            "reason": str(reason).strip(),
            "no_live_side_effects": True,
        }
        if error is not None or not isinstance(failure, dict) or any(
            failure.get(key) != value for key, value in expected_failure.items()
        ):
            return {
                "status": "EXPERIMENT_ABORT_EVIDENCE_SCHEMA_MISMATCH",
                "retry_required": True,
            }
        generated_at = _parse_utc(failure.get("generated_at_utc"))
        prepared_at = _parse_utc(prepared.get("prepared_at_utc"))
        if (
            generated_at is None
            or prepared_at is None
            or generated_at < prepared_at
            or generated_at > now.astimezone(timezone.utc) + timedelta(minutes=5)
        ):
            return {
                "status": "EXPERIMENT_ABORT_EVIDENCE_TIME_INVALID",
                "retry_required": True,
            }
        digest_history = _tuning_experiment_digest_history(loaded)
        if semantic_digest not in digest_history:
            if len(digest_history) >= MAX_TUNING_EXPERIMENT_DIGEST_HISTORY:
                return {
                    "status": "EXPERIMENT_DIGEST_HISTORY_FULL",
                    "retry_required": True,
                }
            digest_history.append(semantic_digest)
        archived.append(
            {
                **prepared,
                "status": "ABORTED",
                "aborted_at_utc": generated_at.isoformat(),
                "aborted_by": str(aborted_by).strip(),
                "reason": str(reason).strip(),
                "failure_evidence_ref": failure_evidence_ref,
            }
        )
        updated = {
            **current,
            "aborted_experiment_contracts": archived[
                -MAX_TUNING_ABORTED_EXPERIMENT_CONTRACTS:
            ],
            "bot_tuning_review_validation": {
                "status": "ABORTED_EXPERIMENT_REVIEW_REACQUISITION_REQUIRED",
                "issues": [
                    "bind a materially changed TEST_REQUIRED review before retrying"
                ],
            },
            "review_reacquisition_required_after_abort": True,
        }
        updated.pop("prepared_experiment_contract", None)
        updated.pop("latest_reviewed_observation_id", None)
        updated_pending = [
            updated
            if str(item.get("work_order_id") or "") == str(work_order_id or "")
            else item
            for item in pending
        ]
        payload = _tuning_queue_payload(
            primary=updated,
            pending_entries=updated_pending,
            terminal_history=terminal_history,
            experiment_digest_history=digest_history,
            experiment_id_digest_history=_tuning_experiment_id_digest_history(loaded),
            override_lifecycle_heads=_tuning_override_lifecycle_heads(loaded),
        )
        try:
            _write_tuning_queue_json(
                path,
                payload,
                expected_source_sha256=expected_source_sha256,
            )
        except OSError as exc:
            return _tuning_work_order_write_failure(
                path=path,
                work_order_id=work_order_id,
                observation_id=observation_id,
                semantic_state_id=_work_order_semantic_state_id(current),
                exc=exc,
            )
        return {
            "status": "EXPERIMENT_CONTRACT_ABORTED",
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "experiment_semantic_digest": semantic_digest,
        }


def _transition_tuning_work_order_locked(
    *,
    path: Path,
    work_order_id: str,
    expected_observation_id: str,
    status: str,
    consumed_by: str,
    experiment_id: str,
    experiment_result: str,
    experiment_evidence_ref: str,
    now: datetime,
) -> dict[str, Any]:
    existing = _load_tuning_work_order(path)
    if existing.get("_read_error"):
        return {
            "status": "WORK_ORDER_READ_FAILED",
            "path": str(path),
            "work_order_id": work_order_id,
            "error": existing.get("_read_error"),
            "retry_required": True,
        }
    expected_source_sha256 = existing.get("_queue_source_sha256")
    pending, terminal_history = _normalized_tuning_work_order_queue(existing)
    matches = [
        item
        for item in pending
        if str(item.get("work_order_id") or "").strip() == work_order_id
    ]
    if len(matches) != 1:
        terminal_matches = [
            item
            for item in terminal_history
            if str(item.get("work_order_id") or "").strip() == work_order_id
        ]
        terminal_match = next(
            (
                item
                for item in terminal_matches
                if str(item.get("experiment_id") or "").strip() == experiment_id
            ),
            None,
        )
        recorded = terminal_match or (terminal_matches[0] if terminal_matches else None)
        if recorded is not None:
            recorded_observation_id = str(
                recorded.get("latest_observation_id")
                or recorded.get("observation_id")
                or recorded.get("event_fingerprint")
                or ""
            ).strip()
            idempotent = bool(
                terminal_match is not None
                and recorded_observation_id == expected_observation_id
                and str(recorded.get("status") or "").strip().upper() == status
                and str(recorded.get("consumed_by") or "").strip() == consumed_by
                and str(recorded.get("experiment_result") or "").strip()
                == experiment_result
                and str(recorded.get("experiment_evidence_ref") or "").strip()
                == experiment_evidence_ref
            )
            override_confirmation: dict[str, Any] | None = None
            if idempotent:
                try:
                    override_confirmation = confirm_accepted_override(
                        path=path.with_name("guardian_tuning_overrides.json"),
                        queue_path=path,
                        work_order_id=work_order_id,
                        experiment_id=experiment_id,
                        experiment_result=experiment_result,
                        evidence_ref=experiment_evidence_ref,
                        now=now,
                    )
                except (
                    OSError,
                    OverflowError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    return {
                        "status": "TUNING_OVERRIDE_CONFIRMATION_PENDING",
                        "path": str(path),
                        "work_order_id": work_order_id,
                        "experiment_id": experiment_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "retry_required": True,
                    }
            return {
                "status": (
                    "WORK_ORDER_ALREADY_TERMINAL"
                    if idempotent
                    else "WORK_ORDER_TERMINAL_CONFLICT"
                ),
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "pending_count": len(pending),
                "tuning_override_confirmation": override_confirmation,
                "recorded_terminal": {
                    "status": recorded.get("status"),
                    "observation_id": recorded_observation_id or None,
                    "consumed_by": recorded.get("consumed_by"),
                    "experiment_id": recorded.get("experiment_id"),
                    "experiment_result": recorded.get("experiment_result"),
                    "experiment_evidence_ref": recorded.get("experiment_evidence_ref"),
                },
            }
        return {
            "status": "WORK_ORDER_NOT_FOUND",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "pending_count": len(pending),
        }
    current = dict(matches[0])
    latest_observation_id = str(
        current.get("latest_observation_id")
        or current.get("observation_id")
        or current.get("event_fingerprint")
        or ""
    ).strip()
    latest_reviewed_observation_id = str(
        current.get("latest_reviewed_observation_id") or ""
    ).strip()
    if (
        latest_observation_id != expected_observation_id
        or latest_reviewed_observation_id != expected_observation_id
    ):
        return {
            "status": "WORK_ORDER_OBSERVATION_STALE",
            "path": str(path),
            "work_order_id": work_order_id,
            "expected_observation_id": expected_observation_id,
            "latest_observation_id": latest_observation_id or None,
            "latest_reviewed_observation_id": latest_reviewed_observation_id or None,
            "retry_required": True,
        }
    review_validation = (
        current.get("bot_tuning_review_validation")
        if isinstance(current.get("bot_tuning_review_validation"), dict)
        else {}
    )
    if str(review_validation.get("status") or "").upper() != "VALID":
        return {
            "status": "STRUCTURED_REVIEW_REQUIRED",
            "path": str(path),
            "work_order_id": work_order_id,
            "bot_tuning_review_status": review_validation.get("status") or "MISSING",
            "retry_required": True,
        }
    review = (
        current.get("bot_tuning_review")
        if isinstance(current.get("bot_tuning_review"), dict)
        else {}
    )
    review_status = str(review.get("review_status") or "").strip().upper()
    if review_status != "TEST_REQUIRED":
        return {
            "status": "EVIDENCE_ACQUISITION_REQUIRED",
            "path": str(path),
            "work_order_id": work_order_id,
            "review_status": review_status or "MISSING",
            "retry_required": True,
        }
    prepared_contract = (
        current.get("prepared_experiment_contract")
        if isinstance(current.get("prepared_experiment_contract"), dict)
        else {}
    )
    if (
        str(prepared_contract.get("status") or "") != "PREPARED"
        or str(prepared_contract.get("experiment_id") or "") != experiment_id
        or str(prepared_contract.get("observation_id") or "")
        != expected_observation_id
    ):
        return {
            "status": "EXPERIMENT_CONTRACT_REQUIRED",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "retry_required": True,
        }
    evidence_validation = _validate_tuning_experiment_evidence_ref(
        queue_path=path,
        evidence_ref=experiment_evidence_ref,
        work_order_id=work_order_id,
        observation_id=expected_observation_id,
        experiment_id=experiment_id,
        experiment_result=experiment_result,
        review=review,
        semantic_state_id=_work_order_semantic_state_id(current),
        prepared_contract=prepared_contract,
        work_order_generated_at=current.get("generated_at_utc"),
        review_completed_at_utc=current.get(
            "structured_review_completed_at_utc"
        ),
        now=now,
    )
    if evidence_validation.get("status") != "VALID":
        return {
            "status": "EXPERIMENT_EVIDENCE_INVALID",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_evidence_ref": experiment_evidence_ref,
            "evidence_validation": evidence_validation,
            "retry_required": True,
        }
    experiment_contract_digest = str(
        evidence_validation.get("experiment_contract_digest") or ""
    ).strip()
    experiment_semantic_digest = str(
        evidence_validation.get("experiment_semantic_digest")
        or prepared_contract.get("experiment_semantic_digest")
        or ""
    ).strip()
    digest_history = _tuning_experiment_digest_history(existing)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", experiment_semantic_digest)
        or experiment_semantic_digest in digest_history
        or any(
            str(item.get("experiment_semantic_digest") or "").strip()
            == experiment_semantic_digest
            for item in terminal_history
        )
    ):
        return {
            "status": "EXPERIMENT_ALREADY_EVALUATED",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "experiment_contract_digest": experiment_contract_digest,
            "experiment_semantic_digest": experiment_semantic_digest,
            "retry_required": True,
        }
    if len(digest_history) >= MAX_TUNING_EXPERIMENT_DIGEST_HISTORY:
        return {
            "status": "EXPERIMENT_DIGEST_HISTORY_FULL",
            "path": str(path),
            "work_order_id": work_order_id,
            "retry_required": True,
        }
    experiment_id_history = _tuning_experiment_id_digest_history(existing)
    experiment_id_digest = _experiment_id_digest(experiment_id)
    if len(experiment_id_history) >= MAX_TUNING_EXPERIMENT_ID_DIGEST_HISTORY:
        return {
            "status": "EXPERIMENT_ID_HISTORY_FULL",
            "path": str(path),
            "work_order_id": work_order_id,
            "retry_required": True,
        }
    if (
        experiment_id_digest in experiment_id_history
        or any(
        str(item.get("experiment_id") or "").strip() == experiment_id
        for item in terminal_history
        )
        or any(
            str(item.get("experiment_id") or "").strip() == experiment_id
            for item in _tuning_override_lifecycle_heads(existing)
        )
    ):
        return {
            "status": "EXPERIMENT_ID_ALREADY_USED",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "retry_required": True,
        }

    prospective_override_key = ""
    prior_head_for_successor: dict[str, Any] | None = None
    if status == "CONSUMED" and experiment_result == "ACCEPTED_IMPROVEMENT":
        source_identity = prepared_contract.get("source_identity")
        lane_id = str(
            source_identity.get("lane_id")
            if isinstance(source_identity, dict)
            else ""
        ).strip()
        prospective_override_key = (
            f"{lane_id}|forecast_confidence_floor" if lane_id else ""
        )
        prior_heads = [
            head
            for head in _tuning_override_lifecycle_heads(existing)
            if str(head.get("override_key") or "") == prospective_override_key
        ]
        if len(prior_heads) > 1 or (
            prior_heads
            and str(prior_heads[0].get("status") or "")
            != "MONITORED_KEEP_COMMITTED"
        ):
            return {
                "status": "TUNING_OVERRIDE_PRIOR_MONITOR_NOT_KEPT",
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "override_key": prospective_override_key or None,
                "prior_status": (
                    str(prior_heads[0].get("status") or "")
                    if len(prior_heads) == 1
                    else "AMBIGUOUS"
                ),
                "retry_required": True,
            }
        if prior_heads:
            prior_head_for_successor = prior_heads[0]
            try:
                read_validated_kept_predecessor_record(
                    path=path.with_name("guardian_tuning_overrides.json"),
                    queue_path=path,
                    override_key=prospective_override_key,
                    experiment_id=str(
                        prior_head_for_successor.get("experiment_id") or ""
                    ),
                )
            except (
                OSError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                sqlite3.Error,
            ) as exc:
                return {
                    "status": "TUNING_OVERRIDE_PRIOR_PROVENANCE_INVALID",
                    "path": str(path),
                    "work_order_id": work_order_id,
                    "experiment_id": experiment_id,
                    "override_key": prospective_override_key,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_required": True,
                }

    activation_ledger_anchor: dict[str, Any] = {}
    if status == "CONSUMED" and experiment_result == "ACCEPTED_IMPROVEMENT":
        try:
            activation_ledger_anchor = current_execution_ledger_anchor(
                ledger_path=path.with_name("execution_ledger.db")
            )
        except (OSError, OverflowError, TypeError, ValueError, sqlite3.Error) as exc:
            return {
                "status": "TUNING_OVERRIDE_ACTIVATION_LEDGER_UNAVAILABLE",
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_required": True,
            }

    try:
        override_application = apply_accepted_override(
            path=path.with_name("guardian_tuning_overrides.json"),
            work_order=current,
            prepared_contract=prepared_contract,
            experiment_id=experiment_id,
            experiment_result=(
                experiment_result if status == "CONSUMED" else "SUPERSEDED"
            ),
            evidence_ref=experiment_evidence_ref,
            activation_ledger_anchor=activation_ledger_anchor,
            queue_path=path,
            now=now,
        )
    except (OSError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "TUNING_OVERRIDE_APPLICATION_FAILED",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "error": f"{type(exc).__name__}: {exc}",
            "retry_required": True,
        }

    terminal_entry = {
        **current,
        "status": status,
        "consumed_at_utc": now.astimezone(timezone.utc).isoformat(),
        "consumed_by": consumed_by,
        "experiment_id": experiment_id,
        "experiment_result": experiment_result,
        "experiment_evidence_ref": experiment_evidence_ref,
        "experiment_contract_digest": experiment_contract_digest,
        "experiment_semantic_digest": experiment_semantic_digest,
        "terminal_transition_source": "guardian_tuning_work_order_lifecycle",
        "tuning_override_application": override_application,
    }
    remaining = [
        item
        for item in pending
        if str(item.get("work_order_id") or "").strip() != work_order_id
    ]
    updated_history = _dedupe_terminal_tuning_history(
        [terminal_entry, *terminal_history]
    )
    override_heads = _tuning_override_lifecycle_heads(existing)
    if status == "CONSUMED" and experiment_result == "ACCEPTED_IMPROVEMENT":
        adjustments = review.get("proposed_adjustments")
        adjustment = (
            adjustments[0]
            if isinstance(adjustments, list) and len(adjustments) == 1
            else {}
        )
        source_identity = prepared_contract.get("source_identity")
        active_binding = prepared_contract.get("active_parameter_binding")
        override_key = str(override_application.get("override_key") or "")
        terminal_digest = hashlib.sha256(
            json.dumps(
                terminal_entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if prior_head_for_successor is not None:
            # Stage writes a dormant pending record before either durable
            # terminal write. Re-read the prior immutable chain and current
            # full-ledger monitor truth here so an invalid predecessor cannot
            # mint even an orphan successor commitment manifest.
            try:
                read_validated_kept_predecessor_record(
                    path=path.with_name("guardian_tuning_overrides.json"),
                    queue_path=path,
                    override_key=prospective_override_key,
                    experiment_id=str(
                        prior_head_for_successor.get("experiment_id") or ""
                    ),
                )
            except (
                OSError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                sqlite3.Error,
            ) as exc:
                return {
                    "status": "TUNING_OVERRIDE_PRIOR_PROVENANCE_INVALID",
                    "path": str(path),
                    "work_order_id": work_order_id,
                    "experiment_id": experiment_id,
                    "override_key": prospective_override_key,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_required": True,
                }
        try:
            terminal_record_ref = write_terminal_commitment_manifest(
                queue_path=path,
                terminal=terminal_entry,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "TUNING_OVERRIDE_COMMITMENT_INVALID",
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_required": True,
            }
        head = {
            "status": "ACTIVE_COMMITTED",
            "override_key": override_key,
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "experiment_result": experiment_result,
            "experiment_evidence_ref": experiment_evidence_ref,
            "experiment_contract_digest": experiment_contract_digest,
            "terminal_confirmation_sha256": terminal_digest,
            "terminal_record_ref": terminal_record_ref,
            "pair": str(adjustment.get("pair") or "").upper(),
            "method": str(
                active_binding.get("method")
                if isinstance(active_binding, dict)
                else ""
            ).upper(),
            "lane_id": str(
                source_identity.get("lane_id")
                if isinstance(source_identity, dict)
                else ""
            ),
            "parameter": str(adjustment.get("parameter") or ""),
            "candidate_value": adjustment.get("candidate_value"),
            "activated_at_utc": override_application.get("activated_at_utc"),
            "activation_ledger_anchor": override_application.get(
                "activation_ledger_anchor"
            ),
            "committed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "live_permission_allowed": False,
            "no_direct_oanda": True,
        }
        if (
            not override_key
            or not head["lane_id"]
            or not head["method"]
            or _safe_finite_float(head["candidate_value"]) is None
        ):
            return {
                "status": "TUNING_OVERRIDE_COMMITMENT_INVALID",
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "retry_required": True,
            }
        replaced_heads = [
            item
            for item in override_heads
            if str(item.get("override_key") or "") == override_key
        ]
        if len(replaced_heads) > 1 or (
            replaced_heads
            and str(replaced_heads[0].get("status") or "")
            != "MONITORED_KEEP_COMMITTED"
        ):
            return {
                "status": "TUNING_OVERRIDE_PRIOR_MONITOR_NOT_KEPT",
                "path": str(path),
                "work_order_id": work_order_id,
                "experiment_id": experiment_id,
                "override_key": override_key,
                "prior_status": (
                    str(replaced_heads[0].get("status") or "")
                    if len(replaced_heads) == 1
                    else "AMBIGUOUS"
                ),
                "retry_required": True,
            }
        if replaced_heads:
            # A lifecycle head is unique per override key.  Once a proven KEEP
            # is superseded, its immutable terminal/activation manifests and
            # monotonic experiment registries remain the durable audit trail;
            # keeping the old accepted terminal in the bounded queue history
            # would falsely require a second current head for the same key.
            replaced_head = replaced_heads[0]
            updated_history = [
                item
                for item in updated_history
                if not (
                    str(item.get("work_order_id") or "")
                    == str(replaced_head.get("work_order_id") or "")
                    and str(item.get("experiment_id") or "")
                    == str(replaced_head.get("experiment_id") or "")
                )
            ]
        override_heads = [
            item
            for item in override_heads
            if str(item.get("override_key") or "") != override_key
        ]
        override_heads.append(head)
        if len(override_heads) > MAX_TUNING_OVERRIDE_LIFECYCLE_HEADS:
            return {
                "status": "TUNING_OVERRIDE_COMMITMENT_CAPACITY_FULL",
                "path": str(path),
                "work_order_id": work_order_id,
                "retry_required": True,
            }
    primary = remaining[0] if remaining else terminal_entry
    payload = _tuning_queue_payload(
        primary=primary,
        pending_entries=remaining,
        terminal_history=updated_history,
        experiment_digest_history=[*digest_history, experiment_semantic_digest],
        experiment_id_digest_history=[
            *experiment_id_history,
            experiment_id_digest,
        ],
        override_lifecycle_heads=override_heads,
    )
    try:
        _write_tuning_queue_json(
            path,
            payload,
            expected_source_sha256=expected_source_sha256,
        )
    except OSError as exc:
        return _tuning_work_order_write_failure(
            path=path,
            work_order_id=work_order_id,
            observation_id=expected_observation_id,
            semantic_state_id=_work_order_semantic_state_id(current),
            exc=exc,
        )
    try:
        override_confirmation = confirm_accepted_override(
            path=path.with_name("guardian_tuning_overrides.json"),
            queue_path=path,
            work_order_id=work_order_id,
            experiment_id=experiment_id,
            experiment_result=(
                experiment_result if status == "CONSUMED" else "SUPERSEDED"
            ),
            evidence_ref=experiment_evidence_ref,
            now=now,
        )
    except (OSError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "TUNING_OVERRIDE_CONFIRMATION_PENDING",
            "path": str(path),
            "work_order_id": work_order_id,
            "experiment_id": experiment_id,
            "terminal_status": status,
            "error": f"{type(exc).__name__}: {exc}",
            "retry_required": True,
        }
    return {
        "status": "WORK_ORDER_TERMINAL_WRITTEN",
        "path": str(path),
        "work_order_id": work_order_id,
        "terminal_status": status,
        "experiment_id": experiment_id,
        "pending_count": len(remaining),
        "terminal_history_count": len(updated_history),
        "tuning_override_application": override_application,
        "tuning_override_confirmation": override_confirmation,
    }


def commit_tuning_override_monitor(
    *,
    path: Path,
    override_path: Path,
    override_key: str,
    experiment_id: str,
    monitor_evidence_ref: str,
    decision: str,
    primary_metric_value: float,
    now: datetime,
) -> dict[str, Any]:
    """Commit a first-20 KEEP/QUARANTINE head, then confirm mutable state."""

    normalized_decision = str(decision or "").upper()
    metric = _safe_finite_float(primary_metric_value)
    if (
        normalized_decision not in {"KEEP", "QUARANTINE"}
        or metric is None
        or normalized_decision != ("KEEP" if metric > 0.0 else "QUARANTINE")
    ):
        return {"status": "POST_ACTIVATION_MONITOR_DECISION_INVALID"}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "POST_ACTIVATION_MONITOR_CONCURRENT_UPDATE",
                "retry_required": True,
            }
        existing = _load_tuning_work_order(path)
        if existing.get("_read_error"):
            return {
                "status": "POST_ACTIVATION_MONITOR_QUEUE_INVALID",
                "error": existing.get("_read_error"),
                "retry_required": True,
            }
        try:
            record = read_stored_override_record(
                path=override_path,
                override_key=override_key,
                experiment_id=experiment_id,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "POST_ACTIVATION_MONITOR_OVERRIDE_INVALID",
                "error": f"{type(exc).__name__}: {exc}",
                "retry_required": True,
            }
        validation = validate_post_activation_monitor_evidence(
            queue_path=path,
            ledger_path=path.with_name("execution_ledger.db"),
            evidence_ref=monitor_evidence_ref,
            expected_record=record,
        )
        if (
            validation.get("status") != "VALID"
            or validation.get("decision") != normalized_decision
            or not math.isclose(
                float(validation.get("primary_metric_value")),
                metric,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return {
                "status": "POST_ACTIVATION_MONITOR_EVIDENCE_INVALID",
                "validation": validation,
                "retry_required": True,
            }
        heads = _tuning_override_lifecycle_heads(existing)
        matches = [
            head
            for head in heads
            if str(head.get("override_key") or "") == override_key
            and str(head.get("experiment_id") or "") == experiment_id
        ]
        if len(matches) != 1:
            return {
                "status": "POST_ACTIVATION_MONITOR_HEAD_MISSING",
                "retry_required": True,
            }
        head = matches[0]
        target_status = (
            "MONITORED_KEEP_COMMITTED"
            if normalized_decision == "KEEP"
            else "QUARANTINED_COMMITTED"
        )
        monitored_at = str(
            head.get("monitored_at_utc")
            or now.astimezone(timezone.utc).isoformat()
        )
        updated_head = {
            **head,
            "status": target_status,
            "monitor_decision": normalized_decision,
            "monitor_evidence_ref": monitor_evidence_ref,
            "post_activation_primary_metric": metric,
            "monitored_at_utc": monitored_at,
        }
        if str(head.get("status") or "") != "ACTIVE_COMMITTED" and head != updated_head:
            return {
                "status": "POST_ACTIVATION_MONITOR_COMMITMENT_CONFLICT",
                "retry_required": True,
            }
        if head != updated_head:
            updated_heads = [
                updated_head if item is head else item for item in heads
            ]
            pending, terminal_history = _normalized_tuning_work_order_queue(existing)
            primary = (
                pending[0]
                if pending
                else terminal_history[0]
                if terminal_history
                else _strip_tuning_envelope(existing)
            )
            payload = _tuning_queue_payload(
                primary=primary,
                pending_entries=pending,
                terminal_history=terminal_history,
                experiment_digest_history=_tuning_experiment_digest_history(existing),
                experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
                override_lifecycle_heads=updated_heads,
            )
            try:
                _write_tuning_queue_json(
                    path,
                    payload,
                    expected_source_sha256=existing.get("_queue_source_sha256"),
                )
            except OSError as exc:
                return {
                    "status": "POST_ACTIVATION_MONITOR_QUEUE_WRITE_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_required": True,
                }
        try:
            confirmation = confirm_post_activation_monitor(
                path=override_path,
                queue_path=path,
                override_key=override_key,
                experiment_id=experiment_id,
                monitor_evidence_ref=monitor_evidence_ref,
                decision=normalized_decision,
                primary_metric_value=metric,
                now=now,
            )
        except (OSError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "POST_ACTIVATION_MONITOR_CONFIRMATION_PENDING",
                "error": f"{type(exc).__name__}: {exc}",
                "retry_required": True,
            }
        return {
            "status": "POST_ACTIVATION_MONITOR_COMMITTED",
            "decision": normalized_decision,
            "primary_metric_value": metric,
            "confirmation": confirmation,
        }


def reconcile_tuning_override_monitors(
    *,
    path: Path,
    override_path: Path,
    now: datetime,
) -> dict[str, Any]:
    """Idempotently confirm queue-committed monitor decisions after a crash."""

    loaded = _load_tuning_work_order(path)
    if loaded.get("_read_error"):
        return {
            "status": "POST_ACTIVATION_MONITOR_RECONCILIATION_QUEUE_INVALID",
            "error": loaded.get("_read_error"),
            "retry_required": True,
        }
    heads = [
        head
        for head in _tuning_override_lifecycle_heads(loaded)
        if str(head.get("status") or "")
        in {"MONITORED_KEEP_COMMITTED", "QUARANTINED_COMMITTED"}
    ]
    results: list[dict[str, Any]] = []
    for head in heads:
        results.append(
            commit_tuning_override_monitor(
                path=path,
                override_path=override_path,
                override_key=str(head.get("override_key") or ""),
                experiment_id=str(head.get("experiment_id") or ""),
                monitor_evidence_ref=str(head.get("monitor_evidence_ref") or ""),
                decision=str(head.get("monitor_decision") or ""),
                primary_metric_value=float(
                    head.get("post_activation_primary_metric")
                ),
                now=now,
            )
        )
    failures = [
        result
        for result in results
        if result.get("status") != "POST_ACTIVATION_MONITOR_COMMITTED"
    ]
    return {
        "status": (
            "POST_ACTIVATION_MONITOR_RECONCILIATION_FAILED"
            if failures
            else "POST_ACTIVATION_MONITOR_RECONCILED"
        ),
        "committed_head_count": len(heads),
        "results": results,
        "retry_required": bool(failures),
    }


def _validate_tuning_experiment_evidence_ref(
    *,
    queue_path: Path,
    evidence_ref: str,
    work_order_id: str,
    observation_id: str,
    experiment_id: str,
    experiment_result: str,
    review: dict[str, Any],
    semantic_state_id: str,
    prepared_contract: dict[str, Any],
    work_order_generated_at: Any,
    review_completed_at_utc: Any,
    now: datetime,
) -> dict[str, Any]:
    match = re.fullmatch(r"(.+)#sha256=([0-9a-fA-F]{64})", str(evidence_ref or "").strip())
    if match is None:
        return {
            "status": "INVALID_FORMAT",
            "required_format": "path#sha256=<64 lowercase hex characters>",
        }
    raw_path, expected_sha256 = match.groups()
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return {
            "status": "EVIDENCE_PATH_OUTSIDE_ALLOWED_ROOT",
            "required_root": "data/guardian_tuning_evidence",
        }
    project_root = queue_path.parent.parent.resolve()
    allowed_root = (project_root / "data" / "guardian_tuning_evidence").resolve()
    candidate = project_root / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return {"status": "EVIDENCE_FILE_MISSING", "path": str(candidate)}
    try:
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            return {
                "status": "EVIDENCE_PATH_OUTSIDE_ALLOWED_ROOT",
                "path": str(candidate),
                "required_root": str(allowed_root),
            }
        if not candidate.is_file() or candidate == queue_path.resolve():
            return {"status": "EVIDENCE_FILE_INVALID", "path": str(candidate)}
        raw_evidence, read_error = _bounded_file_bytes(
            candidate,
            max_bytes=MAX_TUNING_EVIDENCE_BYTES,
        )
        if read_error is not None or raw_evidence is None:
            return {
                "status": "EVIDENCE_FILE_READ_FAILED",
                "path": str(candidate),
                "error": read_error,
            }
        observed_sha256 = hashlib.sha256(raw_evidence).hexdigest()
    except OSError as exc:
        return {
            "status": "EVIDENCE_FILE_READ_FAILED",
            "path": str(candidate),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if observed_sha256 != expected_sha256.lower():
        return {
            "status": "EVIDENCE_SHA_MISMATCH",
            "path": str(candidate),
            "expected_sha256": expected_sha256.lower(),
            "observed_sha256": observed_sha256,
        }
    if candidate.stem.lower() != observed_sha256:
        return {
            "status": "EVIDENCE_FILENAME_NOT_CONTENT_ADDRESSED",
            "path": str(candidate),
            "required_stem": observed_sha256,
        }
    try:
        payload = json.loads(raw_evidence.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"status": "EVIDENCE_JSON_INVALID", "path": str(candidate)}
    if not isinstance(payload, dict):
        return {"status": "EVIDENCE_SCHEMA_INVALID", "path": str(candidate)}
    source_artifact_ref = str(payload.get("source_artifact_ref") or "").strip()
    source_validation = _validate_tuning_experiment_run_ref(
        queue_path=queue_path,
        source_artifact_ref=source_artifact_ref,
        work_order_id=work_order_id,
        observation_id=observation_id,
        experiment_id=experiment_id,
        experiment_result=experiment_result,
        review=review,
        semantic_state_id=semantic_state_id,
        prepared_contract=prepared_contract,
        work_order_generated_at=work_order_generated_at,
        review_completed_at_utc=review_completed_at_utc,
        now=now,
    )
    if source_validation.get("status") != "VALID":
        return {
            "status": "SOURCE_EXPERIMENT_INVALID",
            "path": str(candidate),
            "source_validation": source_validation,
        }
    adjustments = review.get("proposed_adjustments")
    adjustment = adjustments[0] if isinstance(adjustments, list) and len(adjustments) == 1 else {}
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "work_order_id": work_order_id,
        "observation_id": observation_id,
        "experiment_id": experiment_id,
        "review_digest_sha256": _tuning_review_digest(review),
        "hypothesis": review.get("hypothesis"),
        "falsifiable_experiment": review.get("falsifiable_experiment"),
        "pair": adjustment.get("pair"),
        "bot_family": adjustment.get("bot_family"),
        "parameter": adjustment.get("parameter"),
        "current_value": adjustment.get("current_value"),
        "candidate_value": adjustment.get("candidate_value"),
        "result": experiment_result,
        "source_artifact_ref": source_artifact_ref,
        "experiment_contract_digest": source_validation.get("experiment_contract_digest"),
        "no_live_side_effects": True,
    }
    mismatched = [
        key for key, expected in expected_fields.items() if payload.get(key) != expected
    ]
    if mismatched:
        return {
            "status": "EVIDENCE_SCHEMA_MISMATCH",
            "path": str(candidate),
            "mismatched_fields": mismatched,
        }
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    work_order_generated = _parse_utc(work_order_generated_at)
    if generated_at is None:
        return {
            "status": "EVIDENCE_GENERATED_AT_INVALID",
            "path": str(candidate),
        }
    if generated_at > now.astimezone(timezone.utc) + timedelta(minutes=5):
        return {
            "status": "EVIDENCE_GENERATED_AT_FUTURE",
            "path": str(candidate),
        }
    if work_order_generated is not None and generated_at < work_order_generated:
        return {
            "status": "EVIDENCE_PREDATES_WORK_ORDER",
            "path": str(candidate),
        }
    return {
        "status": "VALID",
        "path": str(candidate),
        "sha256": observed_sha256,
        "experiment_contract_digest": source_validation.get("experiment_contract_digest"),
        "experiment_semantic_digest": source_validation.get("experiment_semantic_digest"),
    }


def _tuning_review_digest(review: dict[str, Any]) -> str:
    canonical = json.dumps(
        review,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bounded_file_bytes(path: Path, *, max_bytes: int) -> tuple[bytes | None, str | None]:
    try:
        size = path.stat().st_size
        if size < 0 or size > max_bytes:
            return None, f"artifact size {size} exceeds {max_bytes} bytes"
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if len(raw) > max_bytes:
        return None, f"artifact exceeds {max_bytes} bytes while reading"
    return raw, None


def _approved_tuning_evaluator_sha256() -> str | None:
    evaluator = Path(__file__).resolve().with_name("guardian_tuning_metric_evaluator.py")
    raw, error = _bounded_file_bytes(evaluator, max_bytes=MAX_TUNING_EVALUATOR_BYTES)
    if error is not None or raw is None:
        return None
    return hashlib.sha256(raw).hexdigest()


def _validate_prepared_tuning_source(
    *,
    queue_path: Path,
    source_validation: dict[str, Any],
    adjustment: dict[str, Any],
    review: dict[str, Any],
    review_completed_at_utc: Any,
    cohort_id: str,
    source_watermark: Any,
    sample_count: int,
    require_current_tips: bool,
) -> dict[str, Any]:
    if source_validation.get("status") != "VALID":
        return {"status": "SOURCE_ARTIFACT_INVALID"}
    path = Path(str(source_validation.get("path") or ""))
    raw, error = _bounded_file_bytes(path, max_bytes=MAX_TUNING_SOURCE_BYTES)
    if error is not None or raw is None:
        return {"status": "SOURCE_ARTIFACT_READ_FAILED", "error": error}
    try:
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("source must be an object")
        identity = validate_tuning_source(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError, TypeError, ValueError) as exc:
        return {"status": "SOURCE_SCHEMA_INVALID", "error": str(exc)}
    mismatched: list[str] = []
    expected = {
        "cohort_id": cohort_id,
        "source_watermark": source_watermark,
        "sample_count": sample_count,
        "pair": str(adjustment.get("pair") or "").strip().upper(),
        "lane_id": str(adjustment.get("lane_id") or "").strip(),
        "bot_family": str(adjustment.get("bot_family") or "").strip().lower(),
        "parameter": str(adjustment.get("parameter") or "").strip(),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            mismatched.append(key)
    expected_review_completed_at = _parse_utc(review_completed_at_utc)
    source_review_completed_at = _parse_utc(
        (identity.get("validation_contract") or {}).get("review_completed_at_utc")
        if isinstance(identity.get("validation_contract"), dict)
        else None
    )
    if (
        expected_review_completed_at is None
        or source_review_completed_at != expected_review_completed_at
    ):
        mismatched.append("review_completed_at_utc")
    if mismatched:
        return {"status": "SOURCE_CONTRACT_MISMATCH", "mismatched_fields": mismatched}
    project_root = queue_path.parent.parent.resolve()
    if require_current_tips:
        try:
            current_tip = current_canonical_forward_source_tip(
                ledger_path=project_root / "data" / "execution_ledger.db",
                entry_thesis_path=project_root / "data" / "entry_thesis_ledger.jsonl",
                forecast_history_path=project_root / "data" / "forecast_history.jsonl",
            )
        except (OSError, OverflowError, TypeError, ValueError, sqlite3.Error) as exc:
            return {
                "status": "SOURCE_CURRENT_TIP_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            }
        source_tip = payload.get("source_watermark")
        tip_mismatches = [
            key
            for key, value in current_tip.items()
            if not isinstance(source_tip, dict) or source_tip.get(key) != value
        ]
        if tip_mismatches:
            return {
                "status": "SOURCE_NOT_CURRENT_TIP",
                "mismatched_fields": tip_mismatches,
            }
    canonical = validate_canonical_forward_cohort(
        payload,
        ledger_path=project_root / "data" / "execution_ledger.db",
        entry_thesis_path=project_root / "data" / "entry_thesis_ledger.jsonl",
        forecast_history_path=project_root / "data" / "forecast_history.jsonl",
        review=review,
    )
    if canonical.get("status") != "VALID":
        return {
            "status": "SOURCE_CANONICAL_REVALIDATION_FAILED",
            "canonical_validation": canonical,
        }
    return {
        "status": "VALID",
        "source_semantic_digest": tuning_source_semantic_digest(payload),
        "identity": identity,
    }


def _tuning_experiment_semantic_digest(
    *,
    adjustment: dict[str, Any],
    source_semantic_digest: Any,
    evaluator_artifact_sha256: Any,
    acceptance_threshold: Any,
) -> str | None:
    current_value = _safe_finite_float(adjustment.get("current_value"))
    candidate_value = _safe_finite_float(adjustment.get("candidate_value"))
    threshold = _safe_finite_float(acceptance_threshold)
    source_digest = str(source_semantic_digest or "").strip()
    evaluator_digest = str(evaluator_artifact_sha256 or "").strip()
    if (
        current_value is None
        or candidate_value is None
        or threshold is None
        or threshold != TUNING_FIXED_ACCEPTANCE_THRESHOLD
        or not re.fullmatch(r"[0-9a-f]{64}", source_digest)
        or not re.fullmatch(r"[0-9a-f]{64}", evaluator_digest)
    ):
        return None
    material = {
        "pair": str(adjustment.get("pair") or "").upper(),
        "bot_family": str(adjustment.get("bot_family") or "").lower(),
        "parameter": str(adjustment.get("parameter") or ""),
        "current_value": current_value,
        "candidate_value": candidate_value,
        "source_semantic_digest": source_digest,
        "evaluator": TUNING_EVALUATOR_NAME,
        "evaluator_artifact_sha256": evaluator_digest,
        "primary_metric": TUNING_EVALUATOR_PRIMARY_METRIC,
        "objective": TUNING_EVALUATOR_OBJECTIVE,
        "acceptance_threshold": threshold,
        "metric_names": sorted(TUNING_EVALUATOR_METRIC_NAMES),
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_project_artifact_ref(
    *,
    queue_path: Path,
    artifact_ref: Any,
    allowed_roots: tuple[str, ...],
    max_bytes: int = MAX_TUNING_SOURCE_BYTES,
    require_content_addressed: bool = True,
) -> dict[str, Any]:
    match = re.fullmatch(
        r"(.+)#sha256=([0-9a-fA-F]{64})",
        str(artifact_ref or "").strip(),
    )
    if match is None:
        return {"status": "INVALID_ARTIFACT_REF_FORMAT"}
    raw_path, expected_sha256 = match.groups()
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return {"status": "ARTIFACT_PATH_OUTSIDE_PROJECT"}
    project_root = queue_path.parent.parent.resolve()
    try:
        artifact_path = (project_root / relative).resolve(strict=True)
    except (OSError, RuntimeError):
        return {"status": "ARTIFACT_MISSING", "path": str(relative)}
    allowed = False
    for allowed_root in allowed_roots:
        try:
            artifact_path.relative_to((project_root / allowed_root).resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed or not artifact_path.is_file():
        return {"status": "ARTIFACT_PATH_NOT_ALLOWLISTED", "path": str(artifact_path)}
    raw, error = _bounded_file_bytes(artifact_path, max_bytes=max_bytes)
    if error is not None or raw is None:
        return {"status": "ARTIFACT_READ_FAILED", "error": error}
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256.lower():
        return {
            "status": "ARTIFACT_SHA_MISMATCH",
            "expected_sha256": expected_sha256.lower(),
            "observed_sha256": observed_sha256,
        }
    if require_content_addressed and artifact_path.stem.lower() != observed_sha256:
        return {
            "status": "ARTIFACT_FILENAME_NOT_CONTENT_ADDRESSED",
            "path": str(artifact_path),
            "required_stem": observed_sha256,
        }
    return {
        "status": "VALID",
        "path": str(artifact_path),
        "sha256": observed_sha256,
        "ref": str(artifact_ref),
    }


def _validate_tuning_experiment_run_ref(
    *,
    queue_path: Path,
    source_artifact_ref: str,
    work_order_id: str,
    observation_id: str,
    experiment_id: str,
    experiment_result: str,
    review: dict[str, Any],
    semantic_state_id: str,
    prepared_contract: dict[str, Any],
    work_order_generated_at: Any,
    review_completed_at_utc: Any,
    now: datetime,
) -> dict[str, Any]:
    match = re.fullmatch(
        r"(.+)#sha256=([0-9a-fA-F]{64})",
        str(source_artifact_ref or "").strip(),
    )
    if match is None:
        return {"status": "INVALID_SOURCE_REF_FORMAT"}
    raw_path, expected_sha256 = match.groups()
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return {"status": "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT"}
    project_root = queue_path.parent.parent.resolve()
    allowed_root = (project_root / "data" / "guardian_tuning_experiment_runs").resolve()
    try:
        source_path = (project_root / relative).resolve(strict=True)
        source_path.relative_to(allowed_root)
    except (OSError, RuntimeError, ValueError):
        return {
            "status": "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT_OR_MISSING",
            "required_root": str(allowed_root),
        }
    if not source_path.is_file():
        return {"status": "SOURCE_ARTIFACT_NOT_FILE", "path": str(source_path)}
    raw_source, read_error = _bounded_file_bytes(
        source_path,
        max_bytes=MAX_TUNING_RUN_BYTES,
    )
    if read_error is not None or raw_source is None:
        return {"status": "SOURCE_ARTIFACT_READ_FAILED", "error": read_error}
    observed_sha256 = hashlib.sha256(raw_source).hexdigest()
    if observed_sha256 != expected_sha256.lower():
        return {
            "status": "SOURCE_ARTIFACT_SHA_MISMATCH",
            "expected_sha256": expected_sha256.lower(),
            "observed_sha256": observed_sha256,
        }
    if source_path.stem.lower() != observed_sha256:
        return {
            "status": "SOURCE_FILENAME_NOT_CONTENT_ADDRESSED",
            "required_stem": observed_sha256,
        }
    try:
        payload = json.loads(raw_source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError, TypeError, ValueError):
        return {"status": "SOURCE_ARTIFACT_JSON_INVALID"}
    if not isinstance(payload, dict):
        return {"status": "SOURCE_ARTIFACT_SCHEMA_INVALID"}

    if (
        str(prepared_contract.get("status") or "") != "PREPARED"
        or str(prepared_contract.get("experiment_id") or "") != experiment_id
        or str(prepared_contract.get("observation_id") or "") != observation_id
    ):
        return {"status": "SOURCE_EXPERIMENT_CONTRACT_NOT_PREPARED"}
    adjustments = review.get("proposed_adjustments")
    adjustment = (
        adjustments[0]
        if isinstance(adjustments, list) and len(adjustments) == 1
        else {}
    )
    experiment_contract_digest = str(
        prepared_contract.get("experiment_contract_digest") or ""
    )
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "work_order_id": work_order_id,
        "observation_id": observation_id,
        "experiment_id": experiment_id,
        "experiment_contract_digest": experiment_contract_digest,
        "review_digest_sha256": _tuning_review_digest(review),
        "pair": adjustment.get("pair"),
        "bot_family": adjustment.get("bot_family"),
        "parameter": adjustment.get("parameter"),
        "current_value": adjustment.get("current_value"),
        "candidate_value": adjustment.get("candidate_value"),
        "evaluator": TUNING_EVALUATOR_NAME,
        "primary_metric": TUNING_EVALUATOR_PRIMARY_METRIC,
        "objective": TUNING_EVALUATOR_OBJECTIVE,
        "result": experiment_result,
        "no_live_side_effects": True,
    }
    mismatched = [
        key for key, expected in expected_fields.items() if payload.get(key) != expected
    ]
    if mismatched:
        return {
            "status": "SOURCE_ARTIFACT_SCHEMA_MISMATCH",
            "mismatched_fields": mismatched,
        }

    acceptance_threshold = _safe_finite_float(payload.get("acceptance_threshold"))
    if acceptance_threshold != TUNING_FIXED_ACCEPTANCE_THRESHOLD:
        return {"status": "SOURCE_ACCEPTANCE_THRESHOLD_INVALID"}
    source_data_validation = _validate_project_artifact_ref(
        queue_path=queue_path,
        artifact_ref=payload.get("source_data_ref"),
        allowed_roots=("data/guardian_tuning_experiment_inputs/data",),
        max_bytes=MAX_TUNING_SOURCE_BYTES,
        require_content_addressed=True,
    )
    evaluator_validation = _validate_project_artifact_ref(
        queue_path=queue_path,
        artifact_ref=payload.get("evaluator_artifact_ref"),
        allowed_roots=("data/guardian_tuning_experiment_inputs/evaluators",),
        max_bytes=MAX_TUNING_EVALUATOR_BYTES,
        require_content_addressed=True,
    )
    if source_data_validation.get("status") != "VALID":
        return {"status": "SOURCE_DATA_REF_INVALID", "validation": source_data_validation}
    if evaluator_validation.get("status") != "VALID":
        return {"status": "SOURCE_EVALUATOR_REF_INVALID", "validation": evaluator_validation}
    approved_evaluator_sha256 = _approved_tuning_evaluator_sha256()
    if (
        approved_evaluator_sha256 is None
        or evaluator_validation.get("sha256") != approved_evaluator_sha256
    ):
        return {
            "status": "SOURCE_EVALUATOR_NOT_CURRENTLY_APPROVED",
            "approved_evaluator_sha256": approved_evaluator_sha256,
            "evidence_evaluator_sha256": evaluator_validation.get("sha256"),
        }
    source_contract_validation = _validate_prepared_tuning_source(
        queue_path=queue_path,
        source_validation=source_data_validation,
        adjustment=adjustment,
        review=review,
        review_completed_at_utc=review_completed_at_utc,
        cohort_id=str(payload.get("cohort_id") or ""),
        source_watermark=payload.get("source_watermark"),
        sample_count=payload.get("sample_count"),
        require_current_tips=False,
    )
    if source_contract_validation.get("status") != "VALID":
        return {
            "status": "SOURCE_COHORT_REVALIDATION_FAILED",
            "validation": source_contract_validation,
        }
    source_identity = source_contract_validation.get("identity")
    active_parameter_binding = prepared_contract.get("active_parameter_binding")
    bound_value = (
        _safe_finite_float(active_parameter_binding.get("resolved_value"))
        if isinstance(active_parameter_binding, dict)
        else None
    )
    if (
        not isinstance(source_identity, dict)
        or not isinstance(active_parameter_binding, dict)
        or active_parameter_binding.get("parameter") != "forecast_confidence_floor"
        or bound_value is None
        or bound_value != _safe_finite_float(adjustment.get("current_value"))
    ):
        return {"status": "SOURCE_ACTIVE_PARAMETER_BINDING_INVALID"}
    prepared_expected = {
        "review_digest_sha256": _tuning_review_digest(review),
        "cohort_id": payload.get("cohort_id"),
        "source_watermark": payload.get("source_watermark"),
        "sample_count": payload.get("sample_count"),
        "evaluator": TUNING_EVALUATOR_NAME,
        "source_data_ref": source_data_validation.get("ref"),
        "evaluator_artifact_ref": evaluator_validation.get("ref"),
        "evaluator_artifact_sha256": evaluator_validation.get("sha256"),
        "source_semantic_digest": source_contract_validation.get("source_semantic_digest"),
        "primary_metric": TUNING_EVALUATOR_PRIMARY_METRIC,
        "objective": TUNING_EVALUATOR_OBJECTIVE,
        "acceptance_threshold": acceptance_threshold,
        "metric_names": sorted(TUNING_EVALUATOR_METRIC_NAMES),
        "active_parameter_binding": active_parameter_binding,
        "source_identity": source_identity,
    }
    prepared_mismatch = [
        key
        for key, expected in prepared_expected.items()
        if prepared_contract.get(key) != expected
    ]
    if prepared_mismatch:
        return {
            "status": "SOURCE_EXPERIMENT_CONTRACT_MISMATCH",
            "mismatched_fields": prepared_mismatch,
        }
    expected_semantic_digest = _tuning_experiment_semantic_digest(
        adjustment=adjustment,
        source_semantic_digest=source_contract_validation.get(
            "source_semantic_digest"
        ),
        evaluator_artifact_sha256=evaluator_validation.get("sha256"),
        acceptance_threshold=acceptance_threshold,
    )
    if (
        expected_semantic_digest is None
        or prepared_contract.get("experiment_semantic_digest")
        != expected_semantic_digest
    ):
        return {"status": "SOURCE_EXPERIMENT_SEMANTIC_DIGEST_INVALID"}
    contract_material = {
        "semantic_state_id": semantic_state_id,
        "review_digest_sha256": _tuning_review_digest(review),
        "cohort_id": prepared_contract.get("cohort_id"),
        "source_watermark": prepared_contract.get("source_watermark"),
        "sample_count": prepared_contract.get("sample_count"),
        "evaluator": prepared_contract.get("evaluator"),
        "source_data_ref": prepared_contract.get("source_data_ref"),
        "evaluator_artifact_ref": prepared_contract.get("evaluator_artifact_ref"),
        "primary_metric": prepared_contract.get("primary_metric"),
        "objective": prepared_contract.get("objective"),
        "acceptance_threshold": prepared_contract.get("acceptance_threshold"),
        "metric_names": prepared_contract.get("metric_names"),
        "active_parameter_binding": prepared_contract.get(
            "active_parameter_binding"
        ),
        "source_identity": prepared_contract.get("source_identity"),
    }
    expected_contract_digest = hashlib.sha256(
        json.dumps(
            contract_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", experiment_contract_digest)
        or experiment_contract_digest != expected_contract_digest
    ):
        return {"status": "SOURCE_EXPERIMENT_CONTRACT_DIGEST_INVALID"}

    execution_validation = _execute_frozen_tuning_evaluator(
        project_root=project_root,
        source_path=Path(str(source_data_validation["path"])),
        expected_source_sha256=str(source_data_validation["sha256"]),
        evaluator_artifact_sha256=str(evaluator_validation["sha256"]),
        approved_evaluator_sha256=approved_evaluator_sha256,
        adjustment=adjustment,
        prepared_contract=prepared_contract,
    )
    if execution_validation.get("status") != "VALID":
        return {
            "status": "SOURCE_EVALUATOR_REEXECUTION_FAILED",
            "validation": execution_validation,
        }
    evaluation = execution_validation["evaluation"]
    reported_expected = {
        "cohort_id": evaluation.get("cohort_id"),
        "source_watermark": evaluation.get("source_watermark"),
        "sample_count": evaluation.get("sample_count"),
        "baseline_metrics": evaluation.get("baseline_metrics"),
        "candidate_metrics": evaluation.get("candidate_metrics"),
        "acceptance_constraints": evaluation.get("acceptance_constraints"),
    }
    reported_mismatch = [
        key for key, expected in reported_expected.items() if payload.get(key) != expected
    ]
    if reported_mismatch:
        return {
            "status": "SOURCE_REPORTED_METRICS_NOT_REPRODUCIBLE",
            "mismatched_fields": reported_mismatch,
        }
    derived_result = str(evaluation.get("derived_result") or "")
    if experiment_result != derived_result or payload.get("result") != derived_result:
        return {
            "status": "SOURCE_RESULT_NOT_DERIVED_FROM_FROZEN_EVALUATOR",
            "derived_result": derived_result,
            "reported_result": payload.get("result"),
        }
    exit_status = str(payload.get("exit_status") or "").upper()
    if derived_result == "ACCEPTED_IMPROVEMENT" and exit_status != "COMPLETED_SUCCESS":
        return {"status": "SOURCE_EXIT_STATUS_RESULT_MISMATCH"}
    if derived_result == "REJECTED_NO_IMPROVEMENT" and exit_status != "COMPLETED_NO_EDGE":
        return {"status": "SOURCE_EXIT_STATUS_RESULT_MISMATCH"}

    generated_at = _parse_utc(payload.get("generated_at_utc"))
    executed = payload.get("evaluator_execution")
    if not isinstance(executed, dict):
        return {"status": "SOURCE_EVALUATOR_EXECUTION_MISSING"}
    executed_at = _parse_utc(executed.get("executed_at_utc"))
    prepared_at = _parse_utc(prepared_contract.get("prepared_at_utc"))
    work_order_generated = _parse_utc(work_order_generated_at)
    if generated_at is None or executed_at is None or generated_at != executed_at:
        return {"status": "SOURCE_EXECUTION_TIME_INVALID"}
    if prepared_at is None or generated_at < prepared_at:
        return {"status": "SOURCE_PREDATES_PREPARED_CONTRACT"}
    if work_order_generated is not None and generated_at < work_order_generated:
        return {"status": "SOURCE_PREDATES_WORK_ORDER"}
    if generated_at > now.astimezone(timezone.utc) + timedelta(minutes=5):
        return {"status": "SOURCE_GENERATED_AT_FUTURE"}
    expected_execution = {
        "runner": TUNING_EVALUATOR_RUNNER,
        "exit_code": 0,
        "stdout_sha256": execution_validation.get("stdout_sha256"),
        "stderr_sha256": execution_validation.get("stderr_sha256"),
        "source_data_sha256": source_data_validation.get("sha256"),
        "evaluator_artifact_sha256": evaluator_validation.get("sha256"),
        "executed_at_utc": payload.get("generated_at_utc"),
    }
    execution_mismatch = [
        key for key, expected in expected_execution.items() if executed.get(key) != expected
    ]
    if execution_mismatch:
        return {
            "status": "SOURCE_EVALUATOR_EXECUTION_MISMATCH",
            "mismatched_fields": execution_mismatch,
        }
    return {
        "status": "VALID",
        "path": str(source_path),
        "sha256": observed_sha256,
        "payload": payload,
        "experiment_contract_digest": experiment_contract_digest,
        "experiment_semantic_digest": prepared_contract.get(
            "experiment_semantic_digest"
        ),
    }


def _execute_frozen_tuning_evaluator(
    *,
    project_root: Path,
    source_path: Path,
    expected_source_sha256: str,
    evaluator_artifact_sha256: str,
    approved_evaluator_sha256: str,
    adjustment: dict[str, Any],
    prepared_contract: dict[str, Any],
) -> dict[str, Any]:
    """Recompute evidence with trusted declarative code, never data-path Python.

    The evaluator artifact remains part of the provenance contract, but its
    bytes are never imported or executed.  A fresh read must bind that artifact
    to the currently approved repository evaluator before this trusted
    implementation evaluates the content-addressed cohort.
    """

    del project_root
    if (
        not re.fullmatch(r"[0-9a-f]{64}", evaluator_artifact_sha256)
        or evaluator_artifact_sha256 != approved_evaluator_sha256
    ):
        return {
            "status": "EVALUATOR_NOT_CURRENTLY_APPROVED",
            "approved_evaluator_sha256": approved_evaluator_sha256,
            "evidence_evaluator_sha256": evaluator_artifact_sha256,
        }
    try:
        raw_source, source_error = _bounded_file_bytes(
            source_path,
            max_bytes=MAX_TUNING_SOURCE_BYTES,
        )
        if source_error is not None or raw_source is None:
            raise ValueError(source_error or "source read failed")
        observed_source_sha256 = hashlib.sha256(raw_source).hexdigest()
        if observed_source_sha256 != expected_source_sha256:
            raise ValueError("source changed after its content-address validation")
        source_payload = json.loads(raw_source.decode("utf-8"))
        if not isinstance(source_payload, dict):
            raise ValueError("source payload must be an object")
        evaluation = evaluate_precommitted_threshold_cohort(
            source_payload,
            parameter=str(adjustment.get("parameter") or ""),
            current_value=adjustment.get("current_value"),
            candidate_value=adjustment.get("candidate_value"),
            primary_metric=str(prepared_contract.get("primary_metric") or ""),
            objective=str(prepared_contract.get("objective") or ""),
            acceptance_threshold=prepared_contract.get("acceptance_threshold"),
        )
        stdout_raw = (
            json.dumps(evaluation, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (
        OSError,
        OverflowError,
        RuntimeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {"status": "EXECUTION_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    stderr_raw = b""
    if len(stdout_raw) > MAX_TUNING_RUN_BYTES:
        return {"status": "EXECUTION_REJECTED", "reason": "OUTPUT_TOO_LARGE"}
    if not isinstance(evaluation, dict) or evaluation.get("status") != "EVALUATION_COMPLETED":
        return {"status": "OUTPUT_INCOMPLETE"}
    return {
        "status": "VALID",
        "evaluation": evaluation,
        "stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
    }


def _maybe_write_tuning_work_order_locked(
    *,
    path: Path,
    selected_event: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    reasons = sorted(
        {
            str(item).strip().upper()
            for item in selected_event.get("wake_reason_codes") or []
            if _is_tuning_handoff_reason(str(item))
        }
    )
    if not reasons:
        return {"status": "SKIPPED_NON_TUNING_EVENT"}
    observation_id = _event_material_fingerprint(selected_event)
    semantic_state_id = _event_semantic_state_id(selected_event)
    work_order_id = f"guardian-tuning-{semantic_state_id[:20]}"
    existing = _load_tuning_work_order(path)
    if existing.get("_read_error"):
        return {
            "status": "WORK_ORDER_READ_FAILED",
            "work_order_id": work_order_id,
            "path": str(path),
            "event_fingerprint": observation_id,
            "semantic_state_id": semantic_state_id,
            "observation_id": observation_id,
            "error": existing.get("_read_error"),
            "retry_required": True,
        }
    expected_source_sha256 = existing.get("_queue_source_sha256")
    pending_existing, terminal_history = _normalized_tuning_work_order_queue(existing)
    normalization_required = _tuning_queue_requires_normalization(
        existing,
        pending_entries=pending_existing,
        terminal_history=terminal_history,
    )
    incoming_review_validation = _validate_bot_tuning_review(
        receipt.get("bot_tuning_review"),
        selected_event=selected_event,
    )
    matching_pending = next(
        (
            item
            for item in pending_existing
            if _work_order_semantic_state_id(item) == semantic_state_id
        ),
        None,
    )
    if matching_pending is not None:
        updated = dict(matching_pending)
        prior_prepared = (
            updated.get("prepared_experiment_contract")
            if isinstance(updated.get("prepared_experiment_contract"), dict)
            else {}
        )
        changed = False
        observation_changed = False
        review_enriched = False
        observation_review_required = _event_requires_tuning_observation_append(
            selected_event,
            existing_entry=updated,
        )
        if observation_review_required:
            updated, observation_changed = _append_tuning_observation(
                updated,
                selected_event=selected_event,
                observation_id=observation_id,
                observed_at=now,
            )
            changed = changed or observation_changed
        current_validation = (
            updated.get("bot_tuning_review_validation")
            if isinstance(updated.get("bot_tuning_review_validation"), dict)
            else {}
        )
        current_observation_needs_review = bool(
            updated.get("latest_observation_id") == observation_id
            and updated.get("latest_reviewed_observation_id") != observation_id
        )
        current_review = (
            updated.get("bot_tuning_review")
            if isinstance(updated.get("bot_tuning_review"), dict)
            else {}
        )
        incoming_review = (
            incoming_review_validation.get("review")
            if isinstance(incoming_review_validation.get("review"), dict)
            else {}
        )
        review_upgrade = bool(
            str(current_review.get("review_status") or "").upper()
            == "NO_CHANGE_INSUFFICIENT_EVIDENCE"
            and str(incoming_review.get("review_status") or "").upper()
            == "TEST_REQUIRED"
        )
        review_downgrade = bool(
            str(current_validation.get("status") or "").upper() == "VALID"
            and str(current_review.get("review_status") or "").upper()
            == "TEST_REQUIRED"
            and str(incoming_review.get("review_status") or "").upper()
            == "NO_CHANGE_INSUFFICIENT_EVIDENCE"
            and not current_observation_needs_review
        )
        if (
            incoming_review_validation.get("status") == "VALID"
            and not review_downgrade
            and (
                str(current_validation.get("status") or "").upper() != "VALID"
                or current_observation_needs_review
                or review_upgrade
            )
        ):
            updated["bot_tuning_review_validation"] = {
                key: value
                for key, value in incoming_review_validation.items()
                if key != "review"
            }
            updated["bot_tuning_review"] = incoming_review_validation.get("review")
            updated["structured_review_completed_at_utc"] = now.isoformat()
            updated["latest_reviewed_observation_id"] = observation_id
            changed = True
            review_enriched = True
        effective_validation = (
            updated.get("bot_tuning_review_validation")
            if isinstance(updated.get("bot_tuning_review_validation"), dict)
            else {}
        )
        if changed or normalization_required:
            digest_history = _tuning_experiment_digest_history(existing)
            stale_semantic_digest = str(
                prior_prepared.get("experiment_semantic_digest") or ""
            )
            if (
                observation_changed
                and re.fullmatch(r"[0-9a-f]{64}", stale_semantic_digest)
                and stale_semantic_digest not in digest_history
            ):
                if len(digest_history) >= MAX_TUNING_EXPERIMENT_DIGEST_HISTORY:
                    return {
                        "status": "EXPERIMENT_DIGEST_HISTORY_FULL",
                        "path": str(path),
                        "work_order_id": updated.get("work_order_id"),
                        "retry_required": True,
                    }
                digest_history.append(stale_semantic_digest)
            pending_existing = [
                updated if _work_order_semantic_state_id(item) == semantic_state_id else item
                for item in pending_existing
            ]
            payload = _tuning_queue_payload(
                primary=updated,
                pending_entries=pending_existing,
                terminal_history=terminal_history,
                experiment_digest_history=digest_history,
                experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
                override_lifecycle_heads=_tuning_override_lifecycle_heads(existing),
            )
            try:
                _write_tuning_queue_json(
                    path,
                    payload,
                    expected_source_sha256=expected_source_sha256,
                )
            except OSError as exc:
                return _tuning_work_order_write_failure(
                    path=path,
                    work_order_id=updated.get("work_order_id") or work_order_id,
                    observation_id=observation_id,
                    semantic_state_id=semantic_state_id,
                    exc=exc,
                )
        current_observation_needs_review = bool(
            updated.get("latest_observation_id") == observation_id
            and updated.get("latest_reviewed_observation_id") != observation_id
        )
        if review_downgrade:
            return {
                "status": "STRUCTURED_REVIEW_REQUIRED",
                "work_order_id": updated.get("work_order_id") or work_order_id,
                "path": str(path),
                "event_fingerprint": updated.get("event_fingerprint") or observation_id,
                "semantic_state_id": semantic_state_id,
                "observation_id": observation_id,
                "bot_tuning_review_status": "REVIEW_DOWNGRADE_FORBIDDEN",
                "pending_count": len(pending_existing),
                "retry_required": True,
            }
        if incoming_review_validation.get("status") != "VALID":
            return {
                "status": "STRUCTURED_REVIEW_REQUIRED",
                "work_order_id": updated.get("work_order_id") or work_order_id,
                "path": str(path),
                "event_fingerprint": updated.get("event_fingerprint") or observation_id,
                "semantic_state_id": semantic_state_id,
                "observation_id": observation_id,
                "bot_tuning_review_status": incoming_review_validation.get("status") or "MISSING",
                "pending_count": len(pending_existing),
                "retry_required": True,
            }
        if (
            str(effective_validation.get("status") or "").upper() != "VALID"
            or current_observation_needs_review
        ):
            return {
                "status": "STRUCTURED_REVIEW_REQUIRED",
                "work_order_id": updated.get("work_order_id") or work_order_id,
                "path": str(path),
                "event_fingerprint": updated.get("event_fingerprint") or observation_id,
                "semantic_state_id": semantic_state_id,
                "observation_id": observation_id,
                "bot_tuning_review_status": effective_validation.get("status") or "MISSING",
                "pending_count": len(pending_existing),
                "retry_required": True,
            }
        return {
            "status": (
                "WORK_ORDER_OBSERVATION_APPENDED"
                if observation_changed
                else "WORK_ORDER_REVIEW_ENRICHED"
                if review_enriched
                else "WORK_ORDER_QUEUE_MIGRATED"
                if normalization_required
                else "UNCHANGED_IDEMPOTENT"
            ),
            "work_order_id": updated.get("work_order_id") or work_order_id,
            "path": str(path),
            "event_fingerprint": updated.get("event_fingerprint") or observation_id,
            "semantic_state_id": semantic_state_id,
            "observation_id": observation_id,
            "pending_count": len(pending_existing),
        }
    if len(pending_existing) >= MAX_PENDING_TUNING_WORK_ORDERS:
        if normalization_required:
            payload = _tuning_queue_payload(
                primary=pending_existing[0],
                pending_entries=pending_existing,
                terminal_history=terminal_history,
                experiment_digest_history=_tuning_experiment_digest_history(existing),
                experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
                override_lifecycle_heads=_tuning_override_lifecycle_heads(existing),
            )
            try:
                _write_tuning_queue_json(
                    path,
                    payload,
                    expected_source_sha256=expected_source_sha256,
                )
            except OSError as exc:
                return _tuning_work_order_write_failure(
                    path=path,
                    work_order_id=work_order_id,
                    observation_id=observation_id,
                    semantic_state_id=semantic_state_id,
                    exc=exc,
                )
        return {
            "status": "WORK_ORDER_QUEUE_FULL",
            "work_order_id": work_order_id,
            "path": str(path),
            "event_fingerprint": observation_id,
            "semantic_state_id": semantic_state_id,
            "observation_id": observation_id,
            "pending_count": len(pending_existing),
            "max_pending_count": MAX_PENDING_TUNING_WORK_ORDERS,
            "retry_required": True,
        }
    observation = _tuning_observation_record(
        selected_event=selected_event,
        observation_id=observation_id,
        observed_at=now,
    )
    entry: dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "work_order_id": work_order_id,
        "status": "PENDING_HOURLY_AI_REVIEW",
        "source": "guardian_wake_dispatcher",
        "source_receipt_id": receipt.get("receipt_id") or now.isoformat(),
        "selected_event_id": selected_event.get("event_id"),
        "selected_event_dedupe_key": selected_event.get("dedupe_key"),
        # Preserve the historical meaning of event_fingerprint for external
        # readers.  Queue idempotence uses the explicit semantic_state_id.
        "event_fingerprint": observation_id,
        "semantic_state_id": semantic_state_id,
        "observation_id": observation_id,
        "latest_observation_id": observation_id,
        "observation_count": 1,
        "observation_count_total": 1,
        "observations": [observation],
        "material_reason_codes": reasons,
        "selected_event": selected_event,
        "bot_tuning_review_validation": {
            key: value
            for key, value in incoming_review_validation.items()
            if key != "review"
        },
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
        "execution_boundary": {
            "hourly_ai_review_required": True,
            "work_order_never_grants_live_permission": True,
            "live_gateway_and_risk_gates_unchanged": True,
        },
        "lifecycle_contract": {
            "terminal_statuses": ["CONSUMED", "SUPERSEDED"],
            "consume_only_after_falsifiable_review": True,
            "required_terminal_fields": [
                "consumed_at_utc",
                "consumed_by",
                "experiment_id",
                "experiment_result",
            ],
        },
    }
    if incoming_review_validation.get("status") == "VALID":
        entry["bot_tuning_review"] = incoming_review_validation.get("review")
        entry["latest_reviewed_observation_id"] = observation_id
        entry["structured_review_completed_at_utc"] = now.isoformat()

    terminal_match = _latest_terminal_tuning_work_order(
        terminal_history,
        semantic_state_id=semantic_state_id,
    )
    if terminal_match is not None:
        entry["reopened_count"] = int(terminal_match.get("reopened_count") or 0) + 1
        entry["reopened_from_terminal"] = {
            "work_order_id": terminal_match.get("work_order_id"),
            "status": terminal_match.get("status"),
            "consumed_at_utc": terminal_match.get("consumed_at_utc"),
            "experiment_id": terminal_match.get("experiment_id"),
            "experiment_result": terminal_match.get("experiment_result"),
        }

    entries = [entry, *pending_existing]
    deduped_entries: list[dict[str, Any]] = []
    seen_semantics: set[str] = set()
    for item in entries:
        item_semantic = _work_order_semantic_state_id(item)
        if not item_semantic or item_semantic in seen_semantics:
            continue
        seen_semantics.add(item_semantic)
        deduped_entries.append(item)
    payload = _tuning_queue_payload(
        primary=entry,
        pending_entries=deduped_entries,
        terminal_history=terminal_history,
        experiment_digest_history=_tuning_experiment_digest_history(existing),
        experiment_id_digest_history=_tuning_experiment_id_digest_history(existing),
        override_lifecycle_heads=_tuning_override_lifecycle_heads(existing),
    )
    try:
        _write_tuning_queue_json(
            path,
            payload,
            expected_source_sha256=expected_source_sha256,
        )
    except OSError as exc:
        return _tuning_work_order_write_failure(
            path=path,
            work_order_id=work_order_id,
            observation_id=observation_id,
            semantic_state_id=semantic_state_id,
            exc=exc,
        )
    if incoming_review_validation.get("status") != "VALID":
        return {
            "status": "STRUCTURED_REVIEW_REQUIRED",
            "work_order_id": work_order_id,
            "path": str(path),
            "event_fingerprint": observation_id,
            "semantic_state_id": semantic_state_id,
            "observation_id": observation_id,
            "bot_tuning_review_status": incoming_review_validation.get("status"),
            "pending_count": len(deduped_entries),
            "retry_required": True,
        }
    return {
        "status": "WORK_ORDER_WRITTEN",
        "work_order_id": work_order_id,
        "path": str(path),
        "event_fingerprint": observation_id,
        "semantic_state_id": semantic_state_id,
        "observation_id": observation_id,
        "bot_tuning_review_status": incoming_review_validation.get("status"),
        "pending_count": len(deduped_entries),
    }


def _pending_tuning_work_order_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pending, _ = _normalized_tuning_work_order_queue(payload)
    return pending


_REVISIONED_TUNING_QUEUE_ENVELOPE_FIELDS = frozenset(
    {
        "terminal_history",
        "terminal_history_count",
        "experiment_semantic_digest_history",
        "experiment_id_digest_history",
        "override_lifecycle_heads",
    }
)
_REVISIONED_TUNING_QUEUE_ENTRY_FIELDS = frozenset(
    {
        "semantic_state_id",
        "observations",
        "observation_count",
        "observation_count_total",
        "latest_observation_id",
        "last_observed_at_utc",
        "latest_reviewed_observation_id",
        "prepared_experiment_contract",
        "stale_prepared_experiment_contracts",
        "aborted_experiment_contracts",
        "experiment_evidence_ref",
        "experiment_contract_digest",
        "experiment_semantic_digest",
        "terminal_transition_source",
        "legacy_terminal_without_evidence",
        "rejected_terminal_transition",
    }
)


def _is_unversioned_legacy_tuning_queue(payload: dict[str, Any]) -> bool:
    """Recognize only the queue shape emitted before revision markers existed.

    Revision 4 always writes both its envelope fields and per-entry semantic /
    observation fields.  Merely deleting ``queue_schema_revision`` must not
    downgrade those bytes into the evidence-free legacy terminal contract.
    """

    if "queue_schema_revision" in payload:
        return False
    if _REVISIONED_TUNING_QUEUE_ENVELOPE_FIELDS.intersection(payload):
        return False
    candidates: list[dict[str, Any]] = [payload]
    raw_entries = payload.get("work_orders")
    if isinstance(raw_entries, list):
        candidates.extend(item for item in raw_entries if isinstance(item, dict))
    return not any(
        _REVISIONED_TUNING_QUEUE_ENTRY_FIELDS.intersection(item)
        for item in candidates
    )


def _normalized_tuning_work_order_queue(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _flat_tuning_work_order_records(payload)
    pending_by_semantic: dict[str, dict[str, Any]] = {}
    pending_order: list[str] = []
    terminal: list[dict[str, Any]] = []
    legacy_terminal_revision = _is_unversioned_legacy_tuning_queue(payload)
    for raw_item in records:
        item = raw_item
        if _tuning_work_order_has_terminal_evidence(item) and not _tuning_work_order_entry_terminal(item):
            item = _restore_incomplete_terminal_tuning_work_order(item)
        if _tuning_work_order_entry_pending(item):
            semantic = _work_order_semantic_state_id(item)
            if not semantic:
                continue
            normalized = _normalize_pending_tuning_work_order(item, semantic_state_id=semantic)
            if semantic in pending_by_semantic:
                pending_by_semantic[semantic] = _merge_pending_tuning_work_orders(
                    pending_by_semantic[semantic],
                    normalized,
                )
            else:
                pending_by_semantic[semantic] = normalized
                pending_order.append(semantic)
        elif _tuning_work_order_entry_terminal(item):
            if legacy_terminal_revision and not str(
                item.get("terminal_transition_source") or ""
            ).strip():
                item = {
                    **item,
                    "terminal_transition_source": "legacy_pre_evidence_revision",
                    "legacy_terminal_without_evidence": True,
                }
            terminal.append(_strip_tuning_envelope(item))
    pending = [pending_by_semantic[key] for key in pending_order]
    return pending, _dedupe_terminal_tuning_history(terminal)


def _flat_tuning_work_order_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return []
    top = _strip_tuning_envelope(payload)
    raw_entries = payload.get("work_orders") if isinstance(payload.get("work_orders"), list) else []
    entries = [_strip_tuning_envelope(item) for item in raw_entries if isinstance(item, dict)]
    top_identity = _raw_tuning_work_order_identity(top)
    if top_identity:
        matched = False
        for index, item in enumerate(entries):
            if _raw_tuning_work_order_identity(item) == top_identity:
                merged = {**item, **top}
                # Hourly consumers may update either the backward-compatible
                # top mirror or its matching work_orders child.  A complete
                # terminal transition on either surface is authoritative over
                # a stale PENDING mirror; otherwise the top mirror remains the
                # compatibility authority.
                terminal_keys = (
                    "status",
                    "consumed_at_utc",
                    "consumed_by",
                    "experiment_id",
                    "experiment_result",
                    "experiment_evidence_ref",
                    "experiment_contract_digest",
                    "terminal_transition_source",
                )
                top_terminal = _tuning_work_order_entry_terminal(top)
                child_terminal = _tuning_work_order_entry_terminal(item)
                if top_terminal and child_terminal:
                    top_snapshot = {key: top.get(key) for key in terminal_keys}
                    child_snapshot = {key: item.get(key) for key in terminal_keys}
                    if top_snapshot != child_snapshot:
                        for key in terminal_keys:
                            merged.pop(key, None)
                        merged.update(
                            {
                                "status": "PENDING_HOURLY_AI_REVIEW",
                                "live_permission_allowed": False,
                                "no_direct_oanda": True,
                                "preserve_blockers": True,
                                "rejected_terminal_transition": {
                                    "reason": "TOP_CHILD_TERMINAL_CONFLICT",
                                    "top_terminal": top_snapshot,
                                    "child_terminal": child_snapshot,
                                },
                            }
                        )
                        entries[index] = merged
                        matched = True
                        break
                authoritative_terminal: dict[str, Any] | None = None
                if top_terminal:
                    authoritative_terminal = top
                elif child_terminal:
                    authoritative_terminal = item
                elif _tuning_work_order_has_terminal_status(top):
                    authoritative_terminal = top
                elif _tuning_work_order_has_terminal_status(item):
                    authoritative_terminal = item
                if authoritative_terminal is not None:
                    # Never synthesize a complete terminal transition by
                    # unioning complementary, incomplete top/child fields.
                    # One surface must independently carry all four fields.
                    for key in terminal_keys:
                        if key in authoritative_terminal:
                            merged[key] = authoritative_terminal.get(key)
                        else:
                            merged.pop(key, None)
                entries[index] = merged
                matched = True
                break
        if not matched:
            entries.insert(0, top)
    history = payload.get("terminal_history")
    if isinstance(history, list):
        entries.extend(
            _strip_tuning_envelope(item) for item in history if isinstance(item, dict)
        )
    return entries


def _strip_tuning_envelope(value: dict[str, Any]) -> dict[str, Any]:
    envelope_keys = {
        "schema_version",
        "queue_schema_revision",
        "work_orders",
        "pending_count",
        "terminal_history",
        "terminal_history_count",
        "experiment_semantic_digest_history",
        "experiment_id_digest_history",
        "override_lifecycle_heads",
        "_path",
        "_queue_source_sha256",
        "_read_error",
        "_missing",
    }
    return {key: item for key, item in value.items() if key not in envelope_keys}


def _raw_tuning_work_order_identity(entry: dict[str, Any]) -> str:
    return str(entry.get("event_fingerprint") or entry.get("work_order_id") or "")


def _normalize_pending_tuning_work_order(
    entry: dict[str, Any],
    *,
    semantic_state_id: str,
) -> dict[str, Any]:
    normalized = _strip_tuning_envelope(entry)
    observations = _tuning_observations_from_entry(normalized)
    normalized["semantic_state_id"] = semantic_state_id
    normalized["observations"] = observations
    normalized["observation_count"] = len(observations)
    normalized["observation_count_total"] = max(
        int(normalized.get("observation_count_total") or 0),
        len(observations),
    )
    if observations:
        normalized["latest_observation_id"] = observations[-1]["observation_id"]
    normalized["live_permission_allowed"] = False
    normalized["no_direct_oanda"] = True
    normalized["preserve_blockers"] = True
    selected_event = (
        normalized.get("selected_event")
        if isinstance(normalized.get("selected_event"), dict)
        else None
    )
    review_validation = _validate_bot_tuning_review(
        normalized.get("bot_tuning_review"),
        selected_event=selected_event,
    )
    normalized["bot_tuning_review_validation"] = {
        key: value for key, value in review_validation.items() if key != "review"
    }
    if review_validation.get("status") == "VALID":
        normalized["bot_tuning_review"] = review_validation.get("review")
    else:
        normalized.pop("bot_tuning_review", None)
    return normalized


def _merge_pending_tuning_work_orders(
    primary: dict[str, Any],
    duplicate: dict[str, Any],
) -> dict[str, Any]:
    merged = {**duplicate, **primary}
    observations = _dedupe_tuning_observations(
        [*_tuning_observations_from_entry(duplicate), *_tuning_observations_from_entry(primary)]
    )
    merged["observations"] = observations
    merged["observation_count"] = len(observations)
    merged["observation_count_total"] = max(
        int(primary.get("observation_count_total") or 0),
        int(duplicate.get("observation_count_total") or 0),
        len(observations),
    )
    if observations:
        merged["latest_observation_id"] = observations[-1]["observation_id"]
    merged["material_reason_codes"] = sorted(
        {
            str(reason).strip().upper()
            for reason in [
                *(primary.get("material_reason_codes") or []),
                *(duplicate.get("material_reason_codes") or []),
            ]
            if str(reason).strip()
        }
    )
    primary_validation = (
        primary.get("bot_tuning_review_validation")
        if isinstance(primary.get("bot_tuning_review_validation"), dict)
        else {}
    )
    duplicate_validation = (
        duplicate.get("bot_tuning_review_validation")
        if isinstance(duplicate.get("bot_tuning_review_validation"), dict)
        else {}
    )
    if (
        str(primary_validation.get("status") or "").upper() != "VALID"
        and str(duplicate_validation.get("status") or "").upper() == "VALID"
    ):
        merged["bot_tuning_review_validation"] = duplicate_validation
        merged["bot_tuning_review"] = duplicate.get("bot_tuning_review")
    merged["live_permission_allowed"] = False
    merged["no_direct_oanda"] = True
    merged["preserve_blockers"] = True
    return merged


def _tuning_observation_record(
    *,
    selected_event: dict[str, Any],
    observation_id: str,
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "observed_at_utc": observed_at.isoformat(),
        "event_id": selected_event.get("event_id"),
        "price_zone": selected_event.get("price_zone"),
        "selected_event": selected_event,
    }


def _tuning_observations_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    observations = [
        dict(item)
        for item in entry.get("observations", []) or []
        if isinstance(item, dict) and str(item.get("observation_id") or "")
    ]
    selected_event = entry.get("selected_event") if isinstance(entry.get("selected_event"), dict) else {}
    if selected_event:
        observation_id = str(entry.get("observation_id") or entry.get("event_fingerprint") or "")
        if not observation_id:
            observation_id = _event_material_fingerprint(selected_event)
        observed_at = _parse_utc(entry.get("generated_at_utc")) or datetime.now(timezone.utc)
        observations.append(
            _tuning_observation_record(
                selected_event=selected_event,
                observation_id=observation_id,
                observed_at=observed_at,
            )
        )
    return _dedupe_tuning_observations(observations)


def _dedupe_tuning_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in observations:
        observation_id = str(item.get("observation_id") or "")
        if observation_id:
            by_id[observation_id] = dict(item)
    ordered = sorted(
        by_id.values(),
        key=lambda item: (
            _parse_utc(item.get("observed_at_utc")) or datetime.min.replace(tzinfo=timezone.utc),
            str(item.get("observation_id") or ""),
        ),
    )
    return ordered[-MAX_TUNING_OBSERVATIONS_PER_WORK_ORDER:]


def _append_tuning_observation(
    entry: dict[str, Any],
    *,
    selected_event: dict[str, Any],
    observation_id: str,
    observed_at: datetime,
) -> tuple[dict[str, Any], bool]:
    observations = _tuning_observations_from_entry(entry)
    if any(item.get("observation_id") == observation_id for item in observations):
        return entry, False
    updated = dict(entry)
    observations = _dedupe_tuning_observations(
        [
            *observations,
            _tuning_observation_record(
                selected_event=selected_event,
                observation_id=observation_id,
                observed_at=observed_at,
            ),
        ]
    )
    updated["observations"] = observations
    updated["observation_count"] = len(observations)
    updated["observation_count_total"] = int(entry.get("observation_count_total") or len(observations) - 1) + 1
    updated["latest_observation_id"] = observation_id
    updated["last_observed_at_utc"] = observed_at.isoformat()
    prepared_contract = (
        updated.get("prepared_experiment_contract")
        if isinstance(updated.get("prepared_experiment_contract"), dict)
        else None
    )
    if prepared_contract is not None:
        stale_contracts = [
            dict(item)
            for item in updated.get("stale_prepared_experiment_contracts", []) or []
            if isinstance(item, dict)
        ]
        stale_contracts.append(
            {
                **prepared_contract,
                "status": "STALE_OBSERVATION",
                "invalidated_at_utc": observed_at.isoformat(),
                "invalidated_by_observation_id": observation_id,
            }
        )
        updated["stale_prepared_experiment_contracts"] = stale_contracts[
            -MAX_TUNING_STALE_PREPARED_CONTRACTS:
        ]
        updated.pop("prepared_experiment_contract", None)
    return updated, True


def _event_requires_tuning_observation_append(
    event: dict[str, Any],
    *,
    existing_entry: dict[str, Any] | None = None,
) -> bool:
    reasons = {
        str(item).strip().upper()
        for item in event.get("wake_reason_codes") or []
        if str(item).strip()
    }
    event_type = str(event.get("event_type") or "").upper()
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    if event_type == "TECHNICAL_STATE_CHANGE":
        return "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE" in reasons
    if event_type == "FAILED_ACCEPTANCE":
        if str(details.get("lane_id") or "").strip() or str(details.get("status") or "").strip():
            return True
        if "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE" not in reasons:
            return False
        prior_observations = _tuning_observations_from_entry(existing_entry or {})
        prior_event = (
            prior_observations[-1].get("selected_event")
            if prior_observations
            and isinstance(prior_observations[-1].get("selected_event"), dict)
            else {}
        )
        prior_details = (
            prior_event.get("details")
            if isinstance(prior_event.get("details"), dict)
            else {}
        )
        return details.get("bid") != prior_details.get("bid")
    return False


def _tuning_work_order_entry_terminal(entry: dict[str, Any]) -> bool:
    return bool(
        _tuning_work_order_has_terminal_status(entry)
        and all(
            isinstance(entry.get(key), str) and str(entry.get(key) or "").strip()
            for key in (
                "consumed_at_utc",
                "consumed_by",
                "experiment_id",
                "experiment_result",
            )
        )
    )


def _tuning_work_order_has_terminal_status(entry: dict[str, Any]) -> bool:
    return str(entry.get("status") or "").upper() in {"CONSUMED", "SUPERSEDED"}


def _tuning_work_order_has_terminal_evidence(entry: dict[str, Any]) -> bool:
    return bool(
        _tuning_work_order_has_terminal_status(entry)
        or any(
            str(entry.get(key) or "").strip()
            for key in (
                "consumed_at_utc",
                "consumed_by",
                "experiment_id",
                "experiment_result",
            )
        )
    )


def _restore_incomplete_terminal_tuning_work_order(entry: dict[str, Any]) -> dict[str, Any]:
    restored = _strip_tuning_envelope(entry)
    restored["rejected_terminal_transition"] = {
        "status": restored.get("status"),
        "consumed_at_utc": restored.get("consumed_at_utc"),
        "consumed_by": restored.get("consumed_by"),
        "experiment_id": restored.get("experiment_id"),
        "experiment_result": restored.get("experiment_result"),
        "reason": "TERMINAL_FIELDS_INCOMPLETE",
    }
    restored["status"] = "PENDING_HOURLY_AI_REVIEW"
    restored["live_permission_allowed"] = False
    restored["no_direct_oanda"] = True
    restored["preserve_blockers"] = True
    for key in ("consumed_at_utc", "consumed_by", "experiment_id", "experiment_result"):
        restored.pop(key, None)
    return restored


def _dedupe_terminal_tuning_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for raw in entries:
        entry = _strip_tuning_envelope(raw)
        semantic = _work_order_semantic_state_id(entry)
        identity = (
            semantic,
            str(entry.get("status") or "").upper(),
            str(entry.get("experiment_id") or ""),
            str(entry.get("consumed_at_utc") or ""),
            str(entry.get("work_order_id") or entry.get("event_fingerprint") or ""),
        )
        if semantic:
            by_identity[identity] = entry
    ordered = sorted(
        by_identity.values(),
        key=lambda item: (
            _parse_utc(item.get("consumed_at_utc") or item.get("generated_at_utc"))
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )
    return ordered[:MAX_TUNING_TERMINAL_HISTORY]


def _latest_terminal_tuning_work_order(
    entries: list[dict[str, Any]],
    *,
    semantic_state_id: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in entries
            if _work_order_semantic_state_id(item) == semantic_state_id
        ),
        None,
    )


def _tuning_experiment_digest_history(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("experiment_semantic_digest_history")
    history = [
        str(item)
        for item in (raw if isinstance(raw, list) else [])
        if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
    ]
    entries: list[dict[str, Any]] = [payload]
    for key in ("work_orders", "terminal_history"):
        value = payload.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    for entry in entries:
        direct = str(entry.get("experiment_semantic_digest") or "")
        if re.fullmatch(r"[0-9a-f]{64}", direct):
            history.append(direct)
        for key in (
            "stale_prepared_experiment_contracts",
            "aborted_experiment_contracts",
        ):
            value = entry.get(key)
            if not isinstance(value, list):
                continue
            for contract in value:
                digest = str(
                    contract.get("experiment_semantic_digest")
                    if isinstance(contract, dict)
                    else ""
                )
                if re.fullmatch(r"[0-9a-f]{64}", digest):
                    history.append(digest)
    return list(dict.fromkeys(history))


def _experiment_id_digest(experiment_id: object) -> str:
    return hashlib.sha256(
        ("guardian-tuning-experiment-id-v1\0" + str(experiment_id or "")).encode(
            "utf-8"
        )
    ).hexdigest()


def _tuning_experiment_id_digest_history(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("experiment_id_digest_history")
    return list(
        dict.fromkeys(
            str(item)
            for item in (raw if isinstance(raw, list) else [])
            if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
        )
    )


def _tuning_override_lifecycle_heads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("override_lifecycle_heads")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _tuning_queue_payload(
    *,
    primary: dict[str, Any],
    pending_entries: list[dict[str, Any]],
    terminal_history: list[dict[str, Any]],
    experiment_digest_history: list[str] | None = None,
    experiment_id_digest_history: list[str] | None = None,
    override_lifecycle_heads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    flat_pending = [_strip_tuning_envelope(item) for item in pending_entries]
    flat_history = [_strip_tuning_envelope(item) for item in terminal_history]
    return {
        **_strip_tuning_envelope(primary),
        "schema_version": 2,
        "queue_schema_revision": TUNING_QUEUE_SCHEMA_REVISION,
        "work_orders": flat_pending,
        "pending_count": len(flat_pending),
        "terminal_history": flat_history,
        "terminal_history_count": len(flat_history),
        "experiment_semantic_digest_history": list(
            dict.fromkeys(experiment_digest_history or [])
        ),
        "experiment_id_digest_history": list(
            dict.fromkeys(experiment_id_digest_history or [])
        ),
        "override_lifecycle_heads": [
            dict(item) for item in (override_lifecycle_heads or [])
        ],
    }


def _tuning_queue_requires_normalization(
    payload: dict[str, Any],
    *,
    pending_entries: list[dict[str, Any]],
    terminal_history: list[dict[str, Any]],
) -> bool:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return False
    if int(payload.get("queue_schema_revision") or 0) != TUNING_QUEUE_SCHEMA_REVISION:
        return True
    raw_pending = payload.get("work_orders") if isinstance(payload.get("work_orders"), list) else []
    raw_history = payload.get("terminal_history") if isinstance(payload.get("terminal_history"), list) else []
    if int(payload.get("pending_count") or 0) != len(pending_entries):
        return True
    if int(payload.get("terminal_history_count") or 0) != len(terminal_history):
        return True
    if not isinstance(payload.get("experiment_semantic_digest_history"), list):
        return True
    if not isinstance(payload.get("experiment_id_digest_history"), list):
        return True
    if not isinstance(payload.get("override_lifecycle_heads"), list):
        return True
    if len(raw_pending) != len(pending_entries) or len(raw_history) != len(terminal_history):
        return True
    raw_semantics = [
        _work_order_semantic_state_id(item)
        for item in raw_pending
        if isinstance(item, dict)
    ]
    return raw_semantics != [_work_order_semantic_state_id(item) for item in pending_entries]


def _tuning_work_order_write_failure(
    *,
    path: Path,
    work_order_id: Any,
    observation_id: str,
    semantic_state_id: str,
    exc: OSError,
) -> dict[str, Any]:
    return {
        "status": (
            "WORK_ORDER_CONCURRENT_UPDATE"
            if isinstance(exc, _TuningQueueConcurrentUpdateError)
            else "WORK_ORDER_WRITE_FAILED"
        ),
        "work_order_id": work_order_id,
        "path": str(path),
        "event_fingerprint": observation_id,
        "semantic_state_id": semantic_state_id,
        "observation_id": observation_id,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _tuning_work_order_entry_pending(entry: dict[str, Any]) -> bool:
    return bool(
        str(entry.get("status") or "").upper()
        in {"PENDING_HOURLY_AI_REVIEW", "PENDING", "OPEN"}
        and entry.get("consumed_at_utc") in {None, ""}
        and entry.get("live_permission_allowed") is False
        and entry.get("no_direct_oanda") is True
        and entry.get("preserve_blockers") is True
    )


def _is_tuning_handoff_reason(reason: str) -> bool:
    code = str(reason or "").strip().upper()
    return (
        code in _MATERIAL_CHANGE_REASONS
        or "TECHNICAL_STATE_CHANGE" in code
        or "REGIME" in code
        or "VOLATILITY" in code
        or "FAMILY" in code
        or "STRUCTURE" in code
    )


def _unsafe_tuning_instruction(value: Any) -> bool:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return False
    forbidden_phrases = (
        "call oanda",
        "direct oanda",
        "send market",
        "send order",
        "place order",
        "cancel order",
        "close position",
        "bypass gateway",
        "bypass risk",
        "bypass blocker",
        "disable gateway",
        "disable risk",
        "disable blocker",
        "ignore blocker",
        "relax risk gate",
        "live_permission_allowed",
        "no_direct_oanda=false",
    )
    forbidden_patterns = (
        r"\b(?:invoke|call|use|contact|hit)\b.{0,50}\b(?:broker|oanda|trading api|execution client)\b",
        r"\b(?:execute|submit|send|place|open|close|cancel|liquidate|flatten)\b.{0,60}\b(?:order|trade|position|transaction|holding|exposure)\b",
        r"\b(?:order|trade|position|transaction|holding|exposure)\b.{0,50}\b(?:now|immediately|live)\b",
    )
    japanese_objects = (
        "oanda",
        "注文",
        "発注",
        "実口座",
        "成行",
        "ポジション",
        "決済",
        "キャンセル",
        "リスクゲート",
        "ブロッカー",
    )
    japanese_live_actions = (
        "直接",
        "送る",
        "送信",
        "今すぐ",
        "無視",
        "迂回",
        "解除",
        "発注",
        "実口座",
        "成行",
        "決済",
        "キャンセル",
        "建てる",
        "閉じる",
    )
    japanese_execution_instruction = any(token in text for token in japanese_objects) and any(
        token in text for token in japanese_live_actions
    )
    return bool(
        any(phrase in text for phrase in forbidden_phrases)
        or any(re.search(pattern, text) for pattern in forbidden_patterns)
        or japanese_execution_instruction
    )


def _validate_bot_tuning_review(
    value: Any,
    *,
    selected_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if value is None:
        return {"status": "MISSING", "issues": []}
    if not isinstance(value, dict):
        return {"status": "INVALID", "issues": ["bot_tuning_review must be an object"]}
    issues: list[str] = []
    unsafe_issues: list[str] = []
    if value.get("live_permission_allowed") is not False:
        issues.append("bot_tuning_review live_permission_allowed must be false")
    if value.get("no_direct_oanda") is not True:
        issues.append("bot_tuning_review no_direct_oanda must be true")
    if value.get("preserve_blockers") is not True:
        issues.append("bot_tuning_review preserve_blockers must be true")
    if issues:
        return {"status": "INVALID_UNSAFE_BOUNDARY", "issues": issues}
    review_status = str(value.get("review_status") or "").upper()
    if review_status not in {"TEST_REQUIRED", "NO_CHANGE_INSUFFICIENT_EVIDENCE"}:
        issues.append(
            "bot_tuning_review review_status must be TEST_REQUIRED or NO_CHANGE_INSUFFICIENT_EVIDENCE"
        )
    hypothesis = str(value.get("hypothesis") or "").strip()
    if not hypothesis:
        issues.append("bot_tuning_review hypothesis is required")
    elif _unsafe_tuning_instruction(hypothesis):
        unsafe_issues.append("bot_tuning_review hypothesis contains a forbidden execution instruction")
    falsifiable_experiment = str(value.get("falsifiable_experiment") or "").strip()
    if not falsifiable_experiment:
        issues.append("bot_tuning_review falsifiable_experiment is required")
    elif _unsafe_tuning_instruction(falsifiable_experiment):
        unsafe_issues.append(
            "bot_tuning_review falsifiable_experiment contains a forbidden execution instruction"
        )
    affected_pairs = value.get("affected_pairs")
    selected_pair = str((selected_event or {}).get("pair") or "").upper()
    if not isinstance(affected_pairs, list) or not affected_pairs:
        issues.append("bot_tuning_review affected_pairs must be a non-empty list")
        normalized_pairs: list[str] = []
    else:
        normalized_pairs = [
            str(item or "").strip().upper()
            for item in affected_pairs
            if str(item or "").strip()
        ]
        if selected_pair and normalized_pairs != [selected_pair]:
            issues.append("bot_tuning_review affected_pairs must contain only the selected event pair")
    affected_families = value.get("affected_bot_families")
    if not isinstance(affected_families, list) or not affected_families:
        issues.append("bot_tuning_review affected_bot_families must be a non-empty list")
        normalized_families: list[str] = []
    else:
        normalized_families = list(
            dict.fromkeys(
                str(item or "").strip().lower()
                for item in affected_families
                if str(item or "").strip()
            )
        )[:8]
    invalid_families = [
        family for family in normalized_families if family not in _TUNING_BOT_FAMILIES
    ]
    if isinstance(affected_families, list) and not normalized_families:
        issues.append("bot_tuning_review affected_bot_families must name an allowlisted family")
    if invalid_families:
        unsafe_issues.append(
            "bot_tuning_review affected_bot_families contains forbidden families: "
            + ",".join(invalid_families)
        )
    proposed_adjustments = value.get("proposed_adjustments")
    if not isinstance(proposed_adjustments, list):
        issues.append("bot_tuning_review proposed_adjustments must be a list")
        normalized_adjustments: list[dict[str, Any]] = []
    else:
        normalized_adjustments = []
        if len(proposed_adjustments) > 12:
            issues.append("bot_tuning_review proposed_adjustments must contain at most 12 items")
        for index, raw_adjustment in enumerate(proposed_adjustments):
            prefix = f"bot_tuning_review proposed_adjustments[{index}]"
            if not isinstance(raw_adjustment, dict):
                issues.append(f"{prefix} must be an object")
                continue
            unexpected = sorted(set(raw_adjustment) - _TUNING_ADJUSTMENT_FIELDS)
            if unexpected:
                unsafe_issues.append(
                    f"{prefix} contains forbidden fields: {','.join(unexpected)}"
                )
                continue
            adjustment_pair = str(raw_adjustment.get("pair") or "").strip().upper()
            lane_id = str(raw_adjustment.get("lane_id") or "").strip()
            family = str(raw_adjustment.get("bot_family") or "").strip().lower()
            parameter = str(raw_adjustment.get("parameter") or "").strip()
            rationale = str(raw_adjustment.get("rationale") or "").strip()
            if not adjustment_pair or not selected_pair or adjustment_pair != selected_pair:
                unsafe_issues.append(f"{prefix} pair must equal the selected event pair")
            lane_parts = lane_id.split(":")
            if (
                len(lane_parts) != 5
                or any(not part.strip() for part in lane_parts)
                or str(lane_parts[1] if len(lane_parts) > 1 else "").upper()
                != adjustment_pair
                or str(lane_parts[2] if len(lane_parts) > 2 else "").upper()
                not in {"LONG", "SHORT"}
                or str(lane_parts[4] if len(lane_parts) > 4 else "").upper()
                not in {"MARKET", "LIMIT", "STOP"}
            ):
                issues.append(
                    f"{prefix} lane_id must bind desk, selected pair, side, method, and vehicle"
                )
            selected_details = (
                (selected_event or {}).get("details")
                if isinstance((selected_event or {}).get("details"), dict)
                else {}
            )
            selected_lane_id = str(selected_details.get("lane_id") or "").strip()
            if selected_lane_id and lane_id != selected_lane_id:
                unsafe_issues.append(
                    f"{prefix} lane_id must equal the selected event lane"
                )
            if family not in _TUNING_BOT_FAMILIES or family not in normalized_families:
                unsafe_issues.append(f"{prefix} bot_family is not an affected allowlisted family")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,95}", parameter):
                issues.append(f"{prefix} parameter must be a bounded parameter name")
            lowered_parameter = parameter.lower()
            allowed_parameters = (
                _TUNING_COMMON_SAFE_PARAMETERS
                | _TUNING_SAFE_PARAMETERS_BY_FAMILY.get(family, set())
            )
            parameter_allowlisted = lowered_parameter in allowed_parameters
            parameter_evaluable = lowered_parameter in SUPPORTED_THRESHOLD_PARAMETERS
            if not parameter_allowlisted and any(
                token in lowered_parameter for token in _TUNING_FORBIDDEN_PARAMETER_TOKENS
            ):
                unsafe_issues.append(f"{prefix} parameter targets a protected execution boundary")
            if not parameter_allowlisted:
                unsafe_issues.append(f"{prefix} parameter is not an allowlisted technical parameter")
            elif not parameter_evaluable:
                issues.append(
                    f"{prefix} parameter is not supported by the current frozen evaluator"
                )
            current_value = raw_adjustment.get("current_value")
            candidate_value = raw_adjustment.get("candidate_value")
            numeric_values = (current_value, candidate_value)
            numeric_values_valid = not any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in numeric_values
            )
            if not numeric_values_valid:
                issues.append(f"{prefix} current_value and candidate_value must be finite numbers")
            else:
                current_value = float(current_value)
                candidate_value = float(candidate_value)
                if not (0.0 <= current_value <= 1.0) or not (
                    0.0 <= candidate_value <= 1.0
                ):
                    issues.append(
                        f"{prefix} current_value and candidate_value must be within 0..1"
                    )
                elif candidate_value <= current_value:
                    issues.append(
                        f"{prefix} candidate_value must be greater than current_value "
                        "for floor tightening"
                    )
            if not rationale or len(rationale) > 500:
                issues.append(f"{prefix} rationale is required and must be at most 500 characters")
            elif _unsafe_tuning_instruction(rationale):
                unsafe_issues.append(f"{prefix} rationale contains a forbidden execution instruction")
            if not any(
                message.startswith(prefix)
                for message in [*issues, *unsafe_issues]
            ):
                normalized_adjustments.append(
                    {
                        "pair": adjustment_pair,
                        "lane_id": lane_id,
                        "bot_family": family,
                        "parameter": parameter,
                        "current_value": current_value,
                        "candidate_value": candidate_value,
                        "rationale": rationale,
                    }
                )
    if review_status == "NO_CHANGE_INSUFFICIENT_EVIDENCE" and proposed_adjustments:
        issues.append("NO_CHANGE_INSUFFICIENT_EVIDENCE cannot carry proposed_adjustments")
    acquisition_value = value.get("evidence_acquisition")
    normalized_acquisition: dict[str, Any] | None = None
    if acquisition_value is not None:
        prefix = "bot_tuning_review evidence_acquisition"
        if not isinstance(acquisition_value, dict):
            issues.append(f"{prefix} must be an object")
        else:
            unexpected = sorted(set(acquisition_value) - _TUNING_ACQUISITION_FIELDS)
            if unexpected:
                issues.append(f"{prefix} contains unknown fields: {','.join(unexpected)}")
            missing = sorted(_TUNING_ACQUISITION_FIELDS - set(acquisition_value))
            if missing:
                issues.append(f"{prefix} is missing fields: {','.join(missing)}")
            action_kind = str(acquisition_value.get("action_kind") or "").strip().upper()
            source_ref = str(acquisition_value.get("source_ref") or "").strip()
            required_new_samples = acquisition_value.get("required_new_samples")
            success_condition = str(
                acquisition_value.get("success_condition") or ""
            ).strip()
            if action_kind not in _TUNING_ACQUISITION_ACTION_KINDS:
                issues.append(f"{prefix} action_kind is not allowlisted")
            source_parts = Path(source_ref).parts if source_ref else ()
            if (
                not source_ref
                or source_ref.startswith("/")
                or not source_parts
                or source_parts[0] not in {"data", "logs"}
                or ".." in source_parts
                or len(source_ref) > 256
                or re.fullmatch(r"[A-Za-z0-9_./:#-]+", source_ref) is None
            ):
                issues.append(
                    f"{prefix} source_ref must be a bounded project-relative data/ or logs/ reference"
                )
            if (
                isinstance(required_new_samples, bool)
                or not isinstance(required_new_samples, int)
                or not (1 <= required_new_samples <= 1000)
            ):
                issues.append(f"{prefix} required_new_samples must be an integer within 1..1000")
            if (
                len(success_condition) < 24
                or len(success_condition) > 500
                or success_condition.lower() in _TUNING_VAGUE_ACQUISITION_TEXT
                or any(
                    pattern.search(success_condition)
                    for pattern in _TUNING_VAGUE_ACQUISITION_PATTERNS
                )
                or _unsafe_tuning_instruction(success_condition)
            ):
                issues.append(
                    f"{prefix} success_condition must be exact, bounded, and non-executing"
                )
            acquisition_issue_prefix = f"{prefix} "
            if not any(message.startswith(acquisition_issue_prefix) for message in issues):
                normalized_acquisition = {
                    "action_kind": action_kind,
                    "source_ref": source_ref,
                    "required_new_samples": required_new_samples,
                    "success_condition": success_condition,
                }
    if (
        review_status == "NO_CHANGE_INSUFFICIENT_EVIDENCE"
        and normalized_acquisition is None
    ):
        issues.append(
            "NO_CHANGE_INSUFFICIENT_EVIDENCE requires one structured evidence_acquisition"
        )
    if len(normalized_families) != 1:
        issues.append("bot_tuning_review must name exactly one affected bot family")
    if review_status == "TEST_REQUIRED" and len(proposed_adjustments or []) != 1:
        issues.append("TEST_REQUIRED must carry exactly one proposed adjustment")
    if unsafe_issues:
        return {
            "status": "INVALID_UNSAFE_BOUNDARY",
            "issues": [*issues, *unsafe_issues],
        }
    if issues:
        return {"status": "INVALID_INCOMPLETE_REVIEW", "issues": issues}
    review = {
        "review_status": review_status,
        "affected_pairs": normalized_pairs,
        "affected_bot_families": normalized_families,
        "hypothesis": hypothesis,
        "falsifiable_experiment": falsifiable_experiment,
        "proposed_adjustments": normalized_adjustments,
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
    }
    if normalized_acquisition is not None:
        review["evidence_acquisition"] = normalized_acquisition
    return {"status": "VALID", "issues": [], "review": review}


def _record_state(
    path: Path,
    previous_state: dict[str, Any],
    result: dict[str, Any],
    *,
    reviewed_event: dict[str, Any] | None = None,
    attempted_event: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    environ = env if env is not None else os.environ
    state = dict(previous_state) if isinstance(previous_state, dict) else {}
    if "reviewed_events" in state:
        state["reviewed_events"] = _accepted_review_records(state.get("reviewed_events"))
    state["generated_at_utc"] = result.get("generated_at_utc")
    state["last_status"] = result.get("status")
    result_time = _parse_utc(result.get("generated_at_utc")) or datetime.now(timezone.utc)
    pending_dispatches = _active_pending_dispatch_records(
        state.get("pending_dispatches"),
        now=result_time,
    )
    selection = result.get("selection") if isinstance(result.get("selection"), dict) else {}
    pending_ttl = max(
        60,
        int(
            environ.get(
                "QR_GUARDIAN_PENDING_DISPATCH_TTL_SECONDS",
                DEFAULT_PENDING_DISPATCH_TTL_SECONDS,
            )
        ),
    )
    pending_injection = (
        result.get("pending_injection")
        if isinstance(result.get("pending_injection"), dict)
        else {}
    )
    for candidate in pending_injection.get("successor_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        dedupe_key = str(candidate.get("dedupe_key") or "").strip()
        successor_event = (
            candidate.get("event") if isinstance(candidate.get("event"), dict) else {}
        )
        prior = pending_dispatches.get(dedupe_key)
        if not dedupe_key or not successor_event or not isinstance(prior, dict):
            continue
        pending_dispatches[dedupe_key], _ = _enqueue_pending_dispatch_successor(
            prior,
            event=successor_event,
            now=result_time,
            ttl_seconds=pending_ttl,
        )
    for event in selection.get("pending_events", []) or []:
        if not isinstance(event, dict):
            continue
        dedupe_key = str(event.get("dedupe_key") or "").strip()
        if not dedupe_key:
            continue
        prior = pending_dispatches.get(dedupe_key) or {}
        fingerprint = _event_material_fingerprint(event)
        same_observation = str(prior.get("material_fingerprint") or "") == fingerprint
        if prior and not same_observation:
            pending_dispatches[dedupe_key], _ = _enqueue_pending_dispatch_successor(
                prior,
                event=event,
                now=result_time,
                ttl_seconds=pending_ttl,
            )
            continue
        queued_at = (
            prior.get("queued_at_utc")
            if same_observation and prior.get("queued_at_utc")
            else result_time.isoformat()
        )
        pending_record = dict(prior) if isinstance(prior, dict) else {}
        pending_record.update(
            {
                "event": (
                    prior.get("event")
                    if same_observation and isinstance(prior.get("event"), dict)
                    else event
                ),
                "event_id": event.get("event_id"),
                "material_fingerprint": fingerprint,
                "queued_at_utc": queued_at,
                "last_seen_at_utc": result_time.isoformat(),
                "expires_at_utc": (
                    result_time + timedelta(seconds=pending_ttl)
                ).isoformat(),
            }
        )
        pending_record["successors"] = _active_pending_successor_records(
            pending_record,
            now=result_time,
        )
        pending_dispatches[dedupe_key] = pending_record
    state["last_result"] = {
        key: result.get(key)
        for key in (
            "status",
            "generated_at_utc",
            "receipt_written",
            "action_receipt_path",
            "broker_snapshot_freshness",
            "runtime_disk",
            "queued_for_active_trader",
            "queue_reason",
        )
    }
    if isinstance(result.get("selected_event"), dict):
        state["last_result"]["selected_event"] = result.get("selected_event")
    if isinstance(result.get("parse"), dict):
        state["last_result"]["parse"] = result.get("parse")
    if isinstance(result.get("codex_preflight"), dict):
        state["last_result"]["codex_preflight"] = result.get("codex_preflight")
    if isinstance(result.get("codex"), dict):
        codex = result.get("codex") or {}
        state["last_result"]["codex"] = {
            key: codex.get(key)
            for key in (
                "status",
                "attempt",
                "codex_binary_path",
                "codex_version",
                "requested_model",
                "returncode",
                "remediation_hint",
            )
        }
    if isinstance(result.get("parse_failure"), dict):
        state["last_result"]["parse_failure"] = result.get("parse_failure")
    accepted_review = reviewed_event is not None and bool(result.get("receipt_written"))
    if accepted_review:
        reviewed = _accepted_review_records(state.get("reviewed_events"))
        dedupe_key = str(reviewed_event.get("dedupe_key") or "")
        if dedupe_key:
            reviewed[dedupe_key] = {
                "event_id": reviewed_event.get("event_id"),
                "price_zone": reviewed_event.get("price_zone"),
                "details": reviewed_event.get("details") if isinstance(reviewed_event.get("details"), dict) else {},
                "material_fingerprint": _event_material_fingerprint(reviewed_event),
                "severity": reviewed_event.get("severity"),
                "last_reviewed_at_utc": result.get("generated_at_utc"),
                "last_status": result.get("status"),
                "receipt_written": True,
            }
            state["reviewed_events"] = reviewed
            pending_record = pending_dispatches.get(dedupe_key)
            promoted = (
                _promote_pending_dispatch_successor(
                    pending_record,
                    now=result_time,
                )
                if isinstance(pending_record, dict)
                else None
            )
            if promoted is None:
                pending_dispatches.pop(dedupe_key, None)
            else:
                pending_dispatches[dedupe_key] = promoted
                state["last_result"]["promoted_successor"] = {
                    "dedupe_key": dedupe_key,
                    "event_id": promoted.get("event_id"),
                    "material_fingerprint": promoted.get("material_fingerprint"),
                    "remaining_successor_count": len(promoted.get("successors") or []),
                }
            attempts = state.get("dispatch_attempts") if isinstance(state.get("dispatch_attempts"), dict) else {}
            attempts.pop(dedupe_key, None)
            state["dispatch_attempts"] = attempts
            failures = state.get("parse_failures") if isinstance(state.get("parse_failures"), dict) else {}
            failures.pop(dedupe_key, None)
            state["parse_failures"] = failures
    elif attempted_event is not None:
        failure_code = _failed_attempt_code(result)
        dedupe_key = str(attempted_event.get("dedupe_key") or "")
        if dedupe_key and failure_code:
            attempt_record = _next_failed_attempt_record(
                prior=(state.get("dispatch_attempts") or {}).get(dedupe_key)
                if isinstance(state.get("dispatch_attempts"), dict)
                else None,
                event=attempted_event,
                result=result,
                failure_code=failure_code,
                env=environ,
            )
            attempts = state.get("dispatch_attempts") if isinstance(state.get("dispatch_attempts"), dict) else {}
            attempts[dedupe_key] = attempt_record
            state["dispatch_attempts"] = attempts
            state["last_result"]["dispatch_attempt"] = attempt_record
            if _result_has_parse_failure(result):
                parse_failure = {
                    **{
                        key: attempted_event.get(key)
                        for key in (
                            "event_id",
                            "dedupe_key",
                            "pair",
                            "direction",
                            "event_type",
                            "thesis",
                            "price_zone",
                            "severity",
                        )
                    },
                    "material_fingerprint": attempt_record["material_fingerprint"],
                    "last_failed_at_utc": attempt_record["last_failed_at_utc"],
                    "last_error": failure_code,
                    "consecutive_failures": attempt_record["attempt_count"],
                    "retry_after_utc": attempt_record["retry_after_utc"],
                    "expires_at_utc": attempt_record["expires_at_utc"],
                    "queued_for_active_trader_after_failure": attempt_record["retry_budget_exhausted"],
                }
                failures = state.get("parse_failures") if isinstance(state.get("parse_failures"), dict) else {}
                failures[dedupe_key] = parse_failure
                state["parse_failures"] = failures
                state["last_result"]["parse_failure"] = parse_failure
    ordered_pending = sorted(
        pending_dispatches.items(),
        key=lambda item: (
            _dispatch_priority(
                item[1].get("event") if isinstance(item[1].get("event"), dict) else {}
            ),
            str(item[1].get("queued_at_utc") or ""),
        ),
    )[:MAX_PENDING_DISPATCHES]
    state["pending_dispatches"] = dict(ordered_pending)
    state["last_result"]["pending_dispatch_count"] = len(ordered_pending)
    state["last_result"]["pending_successor_count"] = sum(
        len(record.get("successors") or [])
        for _, record in ordered_pending
        if isinstance(record, dict)
    )
    _write_json(path, state)


def _failed_attempt_code(result: dict[str, Any]) -> str | None:
    status = str(result.get("status") or "").upper()
    if bool(result.get("receipt_written")) and status != "TUNING_HANDOFF_FAILED":
        return None
    if status == "TUNING_HANDOFF_FAILED":
        tuning_handoff = (
            result.get("tuning_handoff")
            if isinstance(result.get("tuning_handoff"), dict)
            else {}
        )
        handoff_status = str(tuning_handoff.get("status") or "").upper()
        if handoff_status in _TUNING_HANDOFF_FAILURE_STATUSES:
            return handoff_status
    parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
    parse_error = str(parse.get("error") or "").upper()
    if status in {
        "PARSE_FAILED",
        "RECEIPT_EVENT_MISMATCH",
        "RECEIPT_REJECTED",
        "RUNTIME_DISK_P0",
        "TUNING_HANDOFF_FAILED",
        *CODEX_RUNTIME_FAILURE_STATUSES,
    }:
        return parse_error or status
    # A live lock that appeared after the first malformed response interrupted
    # only the repair retry; the actual GPT attempt still needs backoff state.
    if status == "QUEUED_FOR_ACTIVE_TRADER" and parse.get("valid") is False and result.get("codex"):
        return parse_error or "PARSE_FAILED_BEFORE_QUEUE"
    return None


def _result_has_parse_failure(result: dict[str, Any]) -> bool:
    parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
    status = str(result.get("status") or "").upper()
    return parse.get("valid") is False and status not in CODEX_RUNTIME_FAILURE_STATUSES


def _next_failed_attempt_record(
    *,
    prior: Any,
    event: dict[str, Any],
    result: dict[str, Any],
    failure_code: str,
    env: dict[str, str],
) -> dict[str, Any]:
    failed_at = _parse_utc(result.get("generated_at_utc")) or datetime.now(timezone.utc)
    fingerprint = _event_material_fingerprint(event)
    binary_identity = _codex_binary_identity(env)
    same_series = (
        isinstance(prior, dict)
        and str(prior.get("event_id") or "") == str(event.get("event_id") or "")
        and str(prior.get("material_fingerprint") or "") == fingerprint
        and _binary_identity_key(prior.get("codex_binary_identity") or {}) == _binary_identity_key(binary_identity)
        and (_parse_utc(prior.get("expires_at_utc")) or failed_at) > failed_at
    )
    attempt_count = int(prior.get("attempt_count") or 0) + 1 if same_series else 1
    total_failures = int(prior.get("total_failures") or 0) + 1 if isinstance(prior, dict) else 1
    first_failed = _parse_utc(prior.get("first_failed_at_utc")) if same_series else failed_at
    first_failed = first_failed or failed_at
    base = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_SECONDS)))
    max_backoff = max(base, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_SECONDS", DEFAULT_RETRY_MAX_SECONDS)))
    max_attempts = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS)))
    ttl = max(base, int(env.get("QR_GUARDIAN_WAKE_RETRY_TTL_SECONDS", DEFAULT_RETRY_TTL_SECONDS)))
    usage_limit_retry_seconds: int | None = None
    if failure_code == "CODEX_USAGE_LIMIT":
        usage_limit_retry_seconds = max(
            60,
            int(
                env.get(
                    "QR_GUARDIAN_WAKE_USAGE_LIMIT_RETRY_SECONDS",
                    DEFAULT_USAGE_LIMIT_RETRY_SECONDS,
                )
            ),
        )
        ttl = max(ttl, usage_limit_retry_seconds * 2)
    expires_at = first_failed + timedelta(seconds=ttl)
    backoff = min(max_backoff, base * (2 ** max(0, attempt_count - 1)))
    retry_budget_exhausted = attempt_count >= max_attempts
    retry_delay = usage_limit_retry_seconds or backoff
    retry_after = expires_at if retry_budget_exhausted else min(
        failed_at + timedelta(seconds=retry_delay),
        expires_at,
    )
    record = {
        "event_id": event.get("event_id"),
        "dedupe_key": event.get("dedupe_key"),
        "pair": event.get("pair"),
        "event_type": event.get("event_type"),
        "severity": event.get("severity"),
        "event": event,
        "material_fingerprint": fingerprint,
        "codex_binary_identity": binary_identity,
        "first_failed_at_utc": first_failed.isoformat(),
        "last_failed_at_utc": failed_at.isoformat(),
        "retry_after_utc": retry_after.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "total_failures": total_failures,
        "retry_budget_exhausted": retry_budget_exhausted,
        "last_status": result.get("status"),
        "last_error": failure_code,
        "runtime_disk": result.get("runtime_disk"),
    }
    if usage_limit_retry_seconds is not None:
        record["usage_limit_retry_seconds"] = usage_limit_retry_seconds
    return record


def _paths_payload(paths: DispatcherPaths) -> dict[str, str]:
    return {
        "guardian_escalation": str(paths.escalation),
        "guardian_events": str(paths.events),
        "guardian_event_state": str(paths.event_state),
        "broker_snapshot": str(paths.broker_snapshot),
        "daily_target_state": str(paths.daily_target_state),
        "guardian_event_report": str(paths.event_report),
        "guardian_action_receipt": str(paths.action_receipt),
        "guardian_action_review": str(paths.action_review),
        "guardian_tuning_work_order": str(paths.tuning_work_order),
        "dispatcher_state": str(paths.dispatcher_state),
        "codex_home": str(paths.codex_home),
        "codex_explicit_output": str(paths.codex_explicit_output),
        "live_root": str(paths.live_root),
    }


def _execution_boundary() -> dict[str, bool]:
    return {
        "guardian_never_calls_oanda": True,
        "guardian_never_stages_orders": True,
        "guardian_never_cancels_orders": True,
        "guardian_never_closes_positions": True,
        "guardian_never_edits_broker_state": True,
        "codex_sandbox_read_only": True,
        "live_order_gateway_required_for_execution": True,
    }


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "generated_at_utc": payload.get("generated_at_utc"),
            "status": payload.get("status"),
            "selected_event_id": (payload.get("selected_event") or {}).get("event_id")
            if isinstance(payload.get("selected_event"), dict)
            else None,
            "receipt_written": payload.get("receipt_written", False),
            "runtime_disk_status": (payload.get("runtime_disk") or {}).get("status")
            if isinstance(payload.get("runtime_disk"), dict)
            else None,
            "runtime_disk_free_bytes": (payload.get("runtime_disk") or {}).get("free_bytes")
            if isinstance(payload.get("runtime_disk"), dict)
            else None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with path.open("a") as handle:
        handle.write(line + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


class _TuningQueueConcurrentUpdateError(OSError):
    pass


def _tuning_queue_json_shape_error(value: Any) -> str | None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_TUNING_QUEUE_JSON_DEPTH:
            return "tuning queue JSON exceeds its nesting-depth bound"
        if isinstance(item, str):
            if len(item) > MAX_TUNING_QUEUE_STRING_CHARS:
                return "tuning queue JSON contains an oversized string"
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                return "tuning queue JSON contains a non-UTF-8 string"
            continue
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    return "tuning queue JSON contains a non-string object key"
                if len(key) > MAX_TUNING_QUEUE_STRING_CHARS:
                    return "tuning queue JSON contains an oversized object key"
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError:
                    return "tuning queue JSON contains a non-UTF-8 object key"
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return None


def _load_tuning_work_order(path: Path) -> dict[str, Any]:
    """Strict queue read: corruption is capacity-unknown and must fail closed."""

    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size < 0 or size > MAX_TUNING_QUEUE_BYTES:
                return {
                    "_path": str(path),
                    "_read_error": (
                        "ValueError: tuning queue raw bytes exceed "
                        f"{MAX_TUNING_QUEUE_BYTES}"
                    ),
                    "_queue_source_sha256": None,
                }
            raw = handle.read(MAX_TUNING_QUEUE_BYTES + 1)
    except FileNotFoundError:
        return {
            "_missing": True,
            "_path": str(path),
            "_queue_source_sha256": None,
        }
    except OSError as exc:
        return {
            "_path": str(path),
            "_read_error": f"{type(exc).__name__}: {exc}",
            "_queue_source_sha256": None,
        }
    if len(raw) > MAX_TUNING_QUEUE_BYTES:
        return {
            "_path": str(path),
            "_read_error": (
                "ValueError: tuning queue raw bytes exceed "
                f"{MAX_TUNING_QUEUE_BYTES} while reading"
            ),
            "_queue_source_sha256": None,
        }
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        return {
            "_path": str(path),
            "_read_error": f"{type(exc).__name__}: invalid tuning queue JSON",
            "_queue_source_sha256": source_sha256,
        }
    if not isinstance(payload, dict):
        return {
            "_path": str(path),
            "_read_error": "ValueError: tuning queue root must be a JSON object",
            "_queue_source_sha256": source_sha256,
        }
    shape_error = _tuning_queue_json_shape_error(payload)
    if shape_error is not None:
        return {
            "_path": str(path),
            "_read_error": f"ValueError: {shape_error}",
            "_queue_source_sha256": source_sha256,
        }
    structure_error = _tuning_queue_structure_error(payload)
    if structure_error is not None:
        return {
            "_path": str(path),
            "_read_error": f"ValueError: {structure_error}",
            "_queue_source_sha256": source_sha256,
        }
    terminal_evidence_error = _tuning_queue_terminal_evidence_error(path, payload)
    if terminal_evidence_error is not None:
        return {
            "_path": str(path),
            "_read_error": f"ValueError: {terminal_evidence_error}",
            "_queue_source_sha256": source_sha256,
        }
    return {
        **payload,
        "_path": str(path),
        "_queue_source_sha256": source_sha256,
    }


def _tuning_queue_structure_error(payload: dict[str, Any]) -> str | None:
    queue_revision_present = "queue_schema_revision" in payload
    raw_queue_revision = payload.get("queue_schema_revision")
    queue_revision = (
        raw_queue_revision
        if isinstance(raw_queue_revision, int)
        and not isinstance(raw_queue_revision, bool)
        and raw_queue_revision >= 0
        else 0
    )

    def terminal_contract_error(item: dict[str, Any], label: str) -> str | None:
        if queue_revision < 4 or not _tuning_work_order_entry_terminal(item):
            return None
        source = str(item.get("terminal_transition_source") or "").strip()
        if source == "legacy_pre_evidence_revision" and item.get(
            "legacy_terminal_without_evidence"
        ) is True:
            return None
        if source != "guardian_tuning_work_order_lifecycle":
            return f"{label} must use the lifecycle terminal writer"
        for key in (
            "experiment_evidence_ref",
            "experiment_contract_digest",
            "experiment_semantic_digest",
        ):
            if not isinstance(item.get(key), str) or not str(item.get(key) or "").strip():
                return f"{label}.{key} is required for revision-4 terminal state"
        if not isinstance(item.get("prepared_experiment_contract"), dict):
            return f"{label}.prepared_experiment_contract is required"
        history = payload.get("experiment_semantic_digest_history")
        if (
            isinstance(history, list)
            and item.get("experiment_semantic_digest") not in history
        ):
            return f"{label}.experiment_semantic_digest is missing from durable history"
        id_history = payload.get("experiment_id_digest_history")
        if (
            isinstance(id_history, list)
            and _experiment_id_digest(item.get("experiment_id")) not in id_history
        ):
            return f"{label}.experiment_id is missing from durable identity history"
        return None

    def entry_structure_error(item: dict[str, Any], label: str) -> str | None:
        for key in ("observation_count", "observation_count_total", "reopened_count"):
            value = item.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                return f"{label}.{key} must be a non-negative integer"
        observations = item.get("observations")
        if observations is not None:
            if not isinstance(observations, list):
                return f"{label}.observations must be a list"
            if len(observations) > MAX_TUNING_OBSERVATIONS_PER_WORK_ORDER:
                return f"{label}.observations exceeds its durable bound"
            for observation_index, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    return f"{label}.observations[{observation_index}] must be an object"
                if not str(
                    observation.get("observation_id")
                    or observation.get("event_fingerprint")
                    or ""
                ).strip():
                    return (
                        f"{label}.observations[{observation_index}] "
                        "is missing a durable identity"
                    )
            observation_count = item.get("observation_count")
            if observation_count is not None and observation_count != len(observations):
                return f"{label}.observation_count must equal observations length"
            observation_count_total = item.get("observation_count_total")
            if (
                observation_count_total is not None
                and observation_count_total < len(observations)
            ):
                return (
                    f"{label}.observation_count_total must be at least observations length"
                )
        prepared_contract = item.get("prepared_experiment_contract")
        if prepared_contract is not None and not isinstance(prepared_contract, dict):
            return f"{label}.prepared_experiment_contract must be an object"
        for key, maximum in (
            (
                "stale_prepared_experiment_contracts",
                MAX_TUNING_STALE_PREPARED_CONTRACTS,
            ),
            (
                "aborted_experiment_contracts",
                MAX_TUNING_ABORTED_EXPERIMENT_CONTRACTS,
            ),
        ):
            contracts = item.get(key)
            if contracts is None:
                continue
            if not isinstance(contracts, list):
                return f"{label}.{key} must be a list"
            if len(contracts) > maximum:
                return f"{label}.{key} exceeds its durable bound"
            if any(not isinstance(contract, dict) for contract in contracts):
                return f"{label}.{key} entries must be objects"
        return None

    for key in ("schema_version", "queue_schema_revision", "pending_count", "terminal_history_count"):
        value = payload.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            return f"{key} must be a non-negative integer"
    if queue_revision_present and raw_queue_revision != TUNING_QUEUE_SCHEMA_REVISION:
        return (
            "queue_schema_revision downgrade/unsupported revision is not writable; "
            "only an unversioned legacy queue may migrate"
        )
    if not queue_revision_present and not _is_unversioned_legacy_tuning_queue(payload):
        return (
            "queue_schema_revision is missing from a revisioned queue; "
            "downgrade to the unversioned legacy contract is forbidden"
        )
    if queue_revision == TUNING_QUEUE_SCHEMA_REVISION:
        if payload.get("schema_version") != 2:
            return "revision-4 queue requires schema_version=2"
        for required in (
            "work_orders",
            "pending_count",
            "terminal_history",
            "terminal_history_count",
            "experiment_semantic_digest_history",
            "experiment_id_digest_history",
            "override_lifecycle_heads",
        ):
            if required not in payload:
                return f"revision-4 queue is missing required envelope field {required}"
    digest_history = payload.get("experiment_semantic_digest_history")
    if digest_history is not None:
        if not isinstance(digest_history, list):
            return "experiment_semantic_digest_history must be a list"
        if len(digest_history) > MAX_TUNING_EXPERIMENT_DIGEST_HISTORY:
            return "experiment_semantic_digest_history exceeds its durable bound"
        if len(set(digest_history)) != len(digest_history):
            return "experiment_semantic_digest_history must be unique"
        for index, digest in enumerate(digest_history):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                return f"experiment_semantic_digest_history[{index}] is invalid"
    experiment_id_history = payload.get("experiment_id_digest_history")
    if experiment_id_history is not None:
        if not isinstance(experiment_id_history, list):
            return "experiment_id_digest_history must be a list"
        if len(experiment_id_history) > MAX_TUNING_EXPERIMENT_ID_DIGEST_HISTORY:
            return "experiment_id_digest_history exceeds its durable bound"
        if len(set(experiment_id_history)) != len(experiment_id_history):
            return "experiment_id_digest_history must be unique"
        for index, digest in enumerate(experiment_id_history):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                return f"experiment_id_digest_history[{index}] is invalid"
    override_heads = payload.get("override_lifecycle_heads")
    if override_heads is not None:
        if not isinstance(override_heads, list):
            return "override_lifecycle_heads must be a list"
        if len(override_heads) > MAX_TUNING_OVERRIDE_LIFECYCLE_HEADS:
            return "override_lifecycle_heads exceeds its durable bound"
        seen_override_keys: set[str] = set()
        for index, head in enumerate(override_heads):
            if not isinstance(head, dict):
                return f"override_lifecycle_heads[{index}] must be an object"
            key = str(head.get("override_key") or "")
            if not key or key in seen_override_keys:
                return f"override_lifecycle_heads[{index}] key is missing or duplicated"
            seen_override_keys.add(key)
            if (
                str(head.get("status") or "")
                not in {
                    "ACTIVE_COMMITTED",
                    "MONITORED_KEEP_COMMITTED",
                    "QUARANTINED_COMMITTED",
                }
                or str(head.get("experiment_result") or "")
                != "ACCEPTED_IMPROVEMENT"
                or not str(head.get("work_order_id") or "")
                or not str(head.get("experiment_id") or "")
                or not str(head.get("experiment_evidence_ref") or "")
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(head.get("experiment_contract_digest") or ""),
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(head.get("terminal_confirmation_sha256") or ""),
                )
                or not re.fullmatch(
                    r"data/guardian_tuning_terminal_manifests/[0-9a-f]{64}\.json"
                    r"#sha256=[0-9a-f]{64}",
                    str(head.get("terminal_record_ref") or ""),
                )
                or not str(head.get("pair") or "")
                or not str(head.get("method") or "")
                or not str(head.get("lane_id") or "")
                or str(head.get("parameter") or "")
                != "forecast_confidence_floor"
                or _safe_finite_float(head.get("candidate_value")) is None
                or not _valid_activation_ledger_anchor(
                    head.get("activation_ledger_anchor")
                )
                or _parse_utc(head.get("activated_at_utc")) is None
                or head.get("activated_at_utc")
                != head.get("activation_ledger_anchor", {}).get("captured_at_utc")
                or head.get("live_permission_allowed") is not False
                or head.get("no_direct_oanda") is not True
            ):
                return f"override_lifecycle_heads[{index}] is invalid"
            head_status = str(head.get("status") or "")
            if head_status != "ACTIVE_COMMITTED":
                decision = str(head.get("monitor_decision") or "")
                if (
                    decision not in {"KEEP", "QUARANTINE"}
                    or head_status
                    != (
                        "MONITORED_KEEP_COMMITTED"
                        if decision == "KEEP"
                        else "QUARANTINED_COMMITTED"
                    )
                    or not re.fullmatch(
                        r"data/guardian_tuning_monitor_evidence/[0-9a-f]{64}\.json"
                        r"#sha256=[0-9a-f]{64}",
                        str(head.get("monitor_evidence_ref") or ""),
                    )
                    or _safe_finite_float(head.get("post_activation_primary_metric"))
                    is None
                    or not str(head.get("monitored_at_utc") or "")
                ):
                    return f"override_lifecycle_heads[{index}] monitor commitment is invalid"
    for key in ("work_orders", "terminal_history"):
        if key not in payload:
            continue
        value = payload.get(key)
        if not isinstance(value, list):
            return f"{key} must be a list"
        maximum = (
            MAX_PENDING_TUNING_WORK_ORDERS
            if key == "work_orders"
            else MAX_TUNING_TERMINAL_HISTORY
        )
        if len(value) > maximum:
            return f"{key} exceeds its durable bound"
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                return f"{key}[{index}] must be an object"
            nested_error = entry_structure_error(item, f"{key}[{index}]")
            if nested_error is not None:
                return nested_error
            if not str(item.get("event_fingerprint") or item.get("work_order_id") or "").strip():
                return f"{key}[{index}] is missing a durable identity"
            if key == "work_orders":
                status = str(item.get("status") or "").upper()
                if status not in {
                    "PENDING_HOURLY_AI_REVIEW",
                    "PENDING",
                    "OPEN",
                    "CONSUMED",
                    "SUPERSEDED",
                }:
                    return f"{key}[{index}] has unsupported lifecycle status"
                if status in {"PENDING_HOURLY_AI_REVIEW", "PENDING", "OPEN"} and not (
                    item.get("live_permission_allowed") is False
                    and item.get("no_direct_oanda") is True
                    and item.get("preserve_blockers") is True
                ):
                    return f"{key}[{index}] is missing fail-closed pending boundaries"
                terminal_error = terminal_contract_error(item, f"{key}[{index}]")
                if terminal_error is not None:
                    return terminal_error
            else:
                status = str(item.get("status") or "").upper()
                if status not in {"CONSUMED", "SUPERSEDED"}:
                    return f"{key}[{index}] has unsupported terminal status"
                if not _tuning_work_order_entry_terminal(item):
                    return f"{key}[{index}] is missing complete terminal fields"
                terminal_error = terminal_contract_error(item, f"{key}[{index}]")
                if terminal_error is not None:
                    return terminal_error
    if isinstance(payload.get("work_orders"), list) and payload.get("pending_count") is not None:
        if payload.get("pending_count") != len(payload["work_orders"]):
            return "pending_count must equal work_orders length"
    if (
        isinstance(payload.get("terminal_history"), list)
        and payload.get("terminal_history_count") is not None
    ):
        if payload.get("terminal_history_count") != len(payload["terminal_history"]):
            return "terminal_history_count must equal terminal_history length"
    if isinstance(payload.get("terminal_history"), list) and isinstance(override_heads, list):
        heads_by_experiment = {
            (str(head.get("work_order_id") or ""), str(head.get("experiment_id") or "")): head
            for head in override_heads
            if isinstance(head, dict)
        }
        for terminal in payload["terminal_history"]:
            if (
                not isinstance(terminal, dict)
                or str(terminal.get("status") or "").upper() != "CONSUMED"
                or str(terminal.get("experiment_result") or "")
                != "ACCEPTED_IMPROVEMENT"
            ):
                continue
            head = heads_by_experiment.get(
                (
                    str(terminal.get("work_order_id") or ""),
                    str(terminal.get("experiment_id") or ""),
                )
            )
            terminal_digest = hashlib.sha256(
                json.dumps(
                    terminal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                head is None
                or head.get("terminal_confirmation_sha256") != terminal_digest
                or head.get("experiment_evidence_ref")
                != terminal.get("experiment_evidence_ref")
                or head.get("experiment_contract_digest")
                != terminal.get("experiment_contract_digest")
            ):
                return "accepted terminal is missing its durable override commitment"
    has_explicit_queue = "work_orders" in payload or "terminal_history" in payload
    has_top_identity = bool(
        str(payload.get("event_fingerprint") or payload.get("work_order_id") or "").strip()
    )
    if has_top_identity:
        top_error = entry_structure_error(payload, "top compatibility entry")
        if top_error is not None:
            return top_error
        top_status = str(payload.get("status") or "").upper()
        if top_status not in {
            "PENDING_HOURLY_AI_REVIEW",
            "PENDING",
            "OPEN",
            "CONSUMED",
            "SUPERSEDED",
        }:
            return "top compatibility entry has unsupported lifecycle status"
        if top_status in {"PENDING_HOURLY_AI_REVIEW", "PENDING", "OPEN"} and not (
            payload.get("live_permission_allowed") is False
            and payload.get("no_direct_oanda") is True
            and payload.get("preserve_blockers") is True
        ):
            return "top compatibility entry is missing fail-closed pending boundaries"
        terminal_error = terminal_contract_error(payload, "top compatibility entry")
        if terminal_error is not None:
            return terminal_error
    if not has_explicit_queue and not has_top_identity:
        return "existing tuning queue has neither an entry nor an explicit queue envelope"
    if queue_revision == TUNING_QUEUE_SCHEMA_REVISION:
        if not has_top_identity:
            return "revision-4 queue is missing its top compatibility identity"
        mirror_identities = {
            _raw_tuning_work_order_identity(item)
            for key in ("work_orders", "terminal_history")
            for item in payload.get(key, [])
            if isinstance(item, dict)
        }
        if _raw_tuning_work_order_identity(payload) not in mirror_identities:
            return "revision-4 top compatibility entry has no matching queue record"
    return None


def _tuning_queue_terminal_evidence_error(path: Path, payload: dict[str, Any]) -> str | None:
    if int(payload.get("queue_schema_revision") or 0) < 4:
        return None
    candidates: list[dict[str, Any]] = []
    if _tuning_work_order_entry_terminal(payload):
        candidates.append(payload)
    for key in ("work_orders", "terminal_history"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(
                item
                for item in value
                if isinstance(item, dict)
                and _tuning_work_order_entry_terminal(item)
            )
    checked: set[tuple[str, str, str]] = set()
    for item in candidates:
        source = str(item.get("terminal_transition_source") or "").strip()
        if source == "legacy_pre_evidence_revision":
            continue
        identity = (
            str(item.get("work_order_id") or ""),
            str(item.get("experiment_id") or ""),
            str(item.get("experiment_evidence_ref") or ""),
        )
        if identity in checked:
            continue
        checked.add(identity)
        review = item.get("bot_tuning_review") if isinstance(item.get("bot_tuning_review"), dict) else {}
        prepared_contract = (
            item.get("prepared_experiment_contract")
            if isinstance(item.get("prepared_experiment_contract"), dict)
            else {}
        )
        validation = _validate_tuning_experiment_evidence_ref(
            queue_path=path,
            evidence_ref=str(item.get("experiment_evidence_ref") or ""),
            work_order_id=str(item.get("work_order_id") or ""),
            observation_id=str(
                item.get("latest_observation_id")
                or item.get("observation_id")
                or item.get("event_fingerprint")
                or ""
            ),
            experiment_id=str(item.get("experiment_id") or ""),
            experiment_result=str(item.get("experiment_result") or ""),
            review=review,
            semantic_state_id=_work_order_semantic_state_id(item),
            prepared_contract=prepared_contract,
            work_order_generated_at=item.get("generated_at_utc"),
            review_completed_at_utc=item.get(
                "structured_review_completed_at_utc"
            ),
            now=datetime.now(timezone.utc),
        )
        if validation.get("status") != "VALID":
            return (
                "terminal experiment evidence failed validation: "
                + str(validation.get("status") or "UNKNOWN")
            )
        if str(item.get("experiment_contract_digest") or "") != str(
            validation.get("experiment_contract_digest") or ""
        ):
            return "terminal experiment contract digest does not match evidence"
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_tuning_queue_json(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_source_sha256: Any,
) -> None:
    """Optimistic CAS so an hourly-AI lifecycle write is never overwritten."""

    shape_error = _tuning_queue_json_shape_error(payload)
    if shape_error is not None:
        raise OSError(f"refusing to write invalid guardian tuning queue: {shape_error}")
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(serialized) > MAX_TUNING_QUEUE_BYTES:
        raise OSError(
            "refusing to write guardian tuning queue raw bytes above "
            f"{MAX_TUNING_QUEUE_BYTES} (would write {len(serialized)})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tuning.tmp")
    try:
        tmp.write_bytes(serialized)
        try:
            current_raw = path.read_bytes()
        except FileNotFoundError:
            current_sha256 = None
        except OSError as exc:
            raise OSError(f"cannot re-read tuning queue before replace: {exc}") from exc
        else:
            current_sha256 = hashlib.sha256(current_raw).hexdigest()
        expected = str(expected_source_sha256) if expected_source_sha256 is not None else None
        if current_sha256 != expected:
            raise _TuningQueueConcurrentUpdateError(
                "guardian tuning queue changed after read; reload and retry without overwriting it"
            )
        # The stable queue lock covers compliant hourly-AI lifecycle writers;
        # this SHA comparison is deliberately the last operation before the
        # atomic replace to catch non-locking or legacy writers as well.
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _json_dumps(payload: dict[str, Any], *, limit: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... TRUNCATED ..."


def _parse_utc(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _valid_activation_ledger_anchor(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "ledger_rowid_watermark",
        "ledger_prefix_sha256",
        "execution_ledger_coverage_start_utc",
        "last_oanda_transaction_id",
        "captured_at_utc",
    }:
        return False
    rowid = value.get("ledger_rowid_watermark")
    def normalized_timestamp(raw: object) -> str:
        timestamp = str(raw or "")
        normalized = (
            timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        )
        if "." not in normalized:
            return normalized
        head, tail = normalized.split(".", 1)
        offset_at = next(
            (index for index, char in enumerate(tail) if char in "+-"),
            len(tail),
        )
        fraction, offset = tail[:offset_at], tail[offset_at:]
        return f"{head}.{fraction[:6].ljust(6, '0')}{offset}"

    normalized_coverage = normalized_timestamp(
        value.get("execution_ledger_coverage_start_utc")
    )
    normalized_captured_at = normalized_timestamp(value.get("captured_at_utc"))
    try:
        coverage_at = datetime.fromisoformat(normalized_coverage)
        captured = datetime.fromisoformat(normalized_captured_at)
    except ValueError:
        return False
    return (
        not isinstance(rowid, bool)
        and isinstance(rowid, int)
        and rowid > 0
        and re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("ledger_prefix_sha256") or "")
        )
        is not None
        and str(value.get("last_oanda_transaction_id") or "").isdigit()
        and coverage_at.tzinfo is not None
        and captured.tzinfo is not None
    )


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _severity_rank(value: Any) -> int:
    return SEVERITY_RANK.get(str(value or "P2").upper(), SEVERITY_RANK["P2"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Dispatch one Guardian GPT-5.5 wake through Codex CLI.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--live-root", type=Path, default=Path(os.environ.get("QR_GUARDIAN_WAKE_LIVE_ROOT", str(DEFAULT_LIVE_ROOT))))
    parser.add_argument("--escalation", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--event-state", type=Path, default=None)
    parser.add_argument("--broker-snapshot", type=Path, default=None)
    parser.add_argument("--daily-target-state", type=Path, default=None)
    parser.add_argument("--event-report", type=Path, default=None)
    parser.add_argument("--prompt-template", type=Path, default=None)
    parser.add_argument("--action-receipt", type=Path, default=None)
    parser.add_argument("--action-review", type=Path, default=None)
    parser.add_argument("--tuning-work-order", type=Path, default=None)
    parser.add_argument("--dispatcher-state", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--codex-output", type=Path, default=None)
    parser.add_argument("--codex-explicit-output", type=Path, default=None)
    parser.add_argument("--live-lock", type=Path, default=None)
    return parser.parse_args(argv)


def paths_from_args(args: argparse.Namespace) -> DispatcherPaths:
    paths = DispatcherPaths.from_root(args.root, live_root=args.live_root)
    return DispatcherPaths(
        root=paths.root,
        escalation=args.escalation or paths.escalation,
        events=args.events or paths.events,
        event_state=args.event_state or paths.event_state,
        broker_snapshot=args.broker_snapshot or paths.broker_snapshot,
        daily_target_state=args.daily_target_state or paths.daily_target_state,
        event_report=args.event_report or paths.event_report,
        prompt_template=args.prompt_template or paths.prompt_template,
        action_receipt=args.action_receipt or paths.action_receipt,
        action_review=args.action_review or paths.action_review,
        tuning_work_order=args.tuning_work_order or paths.tuning_work_order,
        dispatcher_state=args.dispatcher_state or paths.dispatcher_state,
        log=args.log or paths.log,
        codex_home=args.codex_home or paths.codex_home,
        codex_output=args.codex_output or paths.codex_output,
        codex_explicit_output=args.codex_explicit_output or paths.codex_explicit_output,
        live_root=paths.live_root,
        live_lock=args.live_lock or paths.live_lock,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_dispatcher(paths=paths_from_args(args))
    print(json.dumps({k: v for k, v in result.items() if k != "codex"}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"CODEX_EXCEPTION"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
