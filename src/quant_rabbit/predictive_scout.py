from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.forecast_precision import (
    bidask_replay_precision_geometry_candidate,
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.models import BrokerSnapshot, OrderIntent, OrderType, Owner, Side, TradeMethod
from quant_rabbit.operator_manual import is_operator_managed_manual_owner
from quant_rabbit.paths import ROOT
from quant_rabbit.risk import MIN_PRODUCTION_LOT_UNITS


DEFAULT_PREDICTIVE_SCOUT_POLICY = ROOT / "config" / "predictive_scout_policy.json"
PREDICTIVE_SCOUT_SOURCE = "BIDASK_REPLAY_PRECISION"
PREDICTIVE_SCOUT_LIVE_ENV = "QR_PREDICTIVE_SCOUT_LIVE_ENABLED"
PREDICTIVE_SCOUT_MAX_TTL_MINUTES = 90
PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY = 8
PREDICTIVE_SCOUT_MAX_CONCURRENT = 2
PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS = 60
PREDICTIVE_SCOUT_LOSS_COOLDOWN_HOURS = 6
PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES = 3
PREDICTIVE_SCOUT_MIN_REPLAY_SAMPLES = 30
PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS = 5
PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR = 1.2
PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE = 2.0 / 3.0
PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS = 30


def predictive_scout_policy(path: Path | None = None) -> dict[str, Any]:
    policy_path = path or DEFAULT_PREDICTIVE_SCOUT_POLICY
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def predictive_scout_live_enabled() -> bool:
    return str(os.environ.get(PREDICTIVE_SCOUT_LIVE_ENV, "0")).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def predictive_scout_claimed(metadata: dict[str, Any] | None) -> bool:
    """Recognize SCOUT identity even when an opt-in marker is stripped.

    Intent JSON crosses model and artifact boundaries.  A missing boolean must
    not downgrade a canonical contrarian SCOUT into an ordinary GTC lane that
    bypasses expiry, sizing, concurrency, and outcome-history checks.
    """

    if not isinstance(metadata, dict):
        return False
    if any(str(key).startswith("predictive_scout") for key in metadata):
        return True
    role = str(metadata.get("campaign_role") or "").upper()
    if role in {
        "BIDASK_REPLAY_CONTRARIAN_SCOUT",
        "BIDASK_REPLAY_PRECISION_SCOUT",
    }:
        return True
    rule = metadata.get("bidask_replay_precision_seed_rule")
    return bool(isinstance(rule, dict) and rule.get("contrarian_edge") is True)


def predictive_scout_geometry_claimed(
    metadata: dict[str, Any] | None,
    *,
    pair: str,
    side: str,
    order_type: str,
    method: str | None,
) -> bool:
    if predictive_scout_claimed(metadata):
        return True
    if not isinstance(metadata, dict):
        return False
    candidate = bidask_replay_precision_geometry_candidate(
        metadata,
        pair=pair,
        side=side,
        order_type=order_type,
        method=method,
    )
    return bool(isinstance(candidate, dict) and candidate.get("contrarian_edge") is True)


def predictive_scout_intent_claimed(intent: OrderIntent) -> bool:
    method = intent.market_context.method.value if intent.market_context is not None else None
    return predictive_scout_geometry_claimed(
        intent.metadata,
        pair=intent.pair,
        side=intent.side.value,
        order_type=intent.order_type.value,
        method=method,
    )


def predictive_scout_broker_raw_claimed(raw: Any) -> bool:
    """Return whether broker extensions identify an exact SCOUT vehicle.

    This public boundary is shared by position-management code so a filled
    forward experiment keeps its entry-time TP geometry.  Broker payloads are
    intentionally checked independently from mutable local intent artifacts.
    """

    return _raw_has_scout_role(raw)


def predictive_scout_metadata_supported(metadata: dict[str, Any]) -> bool:
    if metadata.get("predictive_scout") is not True:
        return False
    if str(metadata.get("predictive_scout_source") or "").upper() != PREDICTIVE_SCOUT_SOURCE:
        return False
    rule = metadata.get("bidask_replay_precision_seed_rule")
    if not isinstance(rule, dict):
        return False
    canonical = canonical_bidask_replay_precision_rule(str(rule.get("name") or ""))
    if canonical is None or not _embedded_rule_matches_canonical(rule, canonical):
        return False
    canonical_digest = bidask_replay_precision_rule_digest(canonical)
    if str(metadata.get("predictive_scout_rule_digest") or "") != canonical_digest:
        return False
    rule_direction = str(rule.get("direction") or "").upper()
    forecast_direction = str(rule.get("forecast_direction") or "").upper()
    return bool(
        metadata.get("predictive_scout_rule_is_vehicle_proof") is False
        and str(metadata.get("predictive_scout_vehicle_proof_status") or "").upper()
        == "UNPROVEN_PASSIVE_LIMIT"
        and str(metadata.get("predictive_scout_hypothesis") or "").upper()
        == "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN"
        and rule.get("contrarian_edge") is True
        and rule_direction in {"UP", "DOWN"}
        and forecast_direction in {"UP", "DOWN"}
        and rule_direction != forecast_direction
        and rule.get("live_grade") is True
        and str(rule.get("adoption_status") or "").upper() == "LIVE_GRADE_DAILY_STABLE"
        and str(rule.get("daily_stability_status") or "").upper() == "DAILY_STABLE"
        and not list(rule.get("adoption_blockers") or [])
    )


def predictive_scout_intent_issues(
    intent: OrderIntent,
    *,
    snapshot: BrokerSnapshot | None,
    data_root: Path,
    validation_time_utc: datetime | None = None,
    policy_path: Path | None = None,
    execution_ledger_db_path: Path | None = None,
) -> list[dict[str, str]]:
    metadata = intent.metadata or {}
    if not predictive_scout_intent_claimed(intent):
        return []
    now = validation_time_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    policy = predictive_scout_policy(policy_path or data_root.parent / "config" / "predictive_scout_policy.json")
    ledger_path = execution_ledger_db_path or (data_root / "execution_ledger.db")
    issues: list[dict[str, str]] = []

    manual_pair_exposure = bool(
        snapshot is not None
        and (
            any(
                str(position.pair or "").upper() == intent.pair.upper()
                and is_operator_managed_manual_owner(position.owner)
                for position in snapshot.positions
            )
            or any(
                str(order.pair or "").upper() == intent.pair.upper()
                and not order.trade_id
                and is_operator_managed_manual_owner(order.owner)
                for order in snapshot.orders
            )
        )
    )
    if manual_pair_exposure:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED",
                f"predictive SCOUT must not enter {intent.pair} while manual/operator-managed "
                "or tagless position/pending-order exposure exists on the same pair, regardless of direction",
            )
        )

    if not _policy_contract_valid(policy):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_POLICY_INVALID",
                "predictive SCOUT policy must be schema v1 forward-evidence-only, LIMIT/1000u, at most two concurrent, capped at eight broker-POST reservations, six-hour post-loss cooldown, negative-vehicle quarantine, and never auto-promote",
            )
        )
    elif policy.get("enabled") is not True:
        issues.append(_issue("PREDICTIVE_SCOUT_POLICY_DISABLED", "predictive SCOUT policy is missing or disabled"))
    if not predictive_scout_live_enabled():
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_LIVE_DISABLED",
                f"{PREDICTIVE_SCOUT_LIVE_ENV}=1 is required for a live SCOUT candidate",
            )
        )
    if not predictive_scout_metadata_supported(metadata):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_RULE_NOT_LIVE_GRADE",
                "SCOUT requires one pre-registered, daily-stable contrarian failure hypothesis; its passive LIMIT vehicle remains unproven until forward exits resolve",
            )
        )
        if metadata.get("predictive_scout") is not True:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_MARKER_REQUIRED",
                    "canonical contrarian SCOUT metadata cannot be downgraded by removing predictive_scout=true",
                )
            )
    else:
        method = intent.market_context.method.value if intent.market_context is not None else None
        canonical_candidate = bidask_replay_precision_geometry_candidate(
            metadata,
            pair=intent.pair,
            side=intent.side.value,
            order_type=intent.order_type.value,
            method=method,
        )
        embedded_rule = metadata.get("bidask_replay_precision_seed_rule")
        embedded_name = str(embedded_rule.get("name") or "") if isinstance(embedded_rule, dict) else ""
        if (
            canonical_candidate is None
            or str(canonical_candidate.get("name") or "") != embedded_name
            or str(canonical_candidate.get("pair") or "").upper() != intent.pair.upper()
            or str(canonical_candidate.get("side") or "").upper() != intent.side.value
        ):
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_CANONICAL_RULE_MISMATCH",
                    "embedded SCOUT rule does not match the current pair/side/forecast confidence/horizon bucket in the configured canonical rule set",
                )
            )
        elif not _canonical_scout_exit_geometry_matches(intent, canonical_candidate):
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_CANONICAL_GEOMETRY_MISMATCH",
                    "SCOUT TP/SL distances must exactly match the canonical bid/ask replay grid; do not mint a new vehicle id by changing the exit shape",
                )
            )
    allowed_sources = {str(item).upper() for item in policy.get("allowed_sources", []) or []}
    if allowed_sources and str(metadata.get("predictive_scout_source") or "").upper() not in allowed_sources:
        issues.append(_issue("PREDICTIVE_SCOUT_SOURCE_NOT_ALLOWED", "SCOUT source is outside the configured allowlist"))
    if intent.order_type != OrderType.LIMIT:
        issues.append(_issue("PREDICTIVE_SCOUT_LIMIT_ONLY", "SCOUT entries must be passive LIMIT orders"))
    scout_method = (
        intent.market_context.method if intent.market_context is not None else None
    )
    if scout_method != TradeMethod.BREAKOUT_FAILURE:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_METHOD_REQUIRED",
                "forecast-failure SCOUT must remain BREAKOUT_FAILURE; relabeling the same selector cannot mint a new failure-memory vehicle",
            )
        )
    if str(metadata.get("desk") or "").strip().lower() != "failure_trader":
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_DESK_REQUIRED",
                "forecast-failure SCOUT must remain on failure_trader; desk relabeling cannot reset cooldown or quarantine",
            )
        )
    if str(metadata.get("campaign_role") or "").upper() != "BIDASK_REPLAY_CONTRARIAN_SCOUT":
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_ROLE_REQUIRED",
                "SCOUT must carry campaign_role=BIDASK_REPLAY_CONTRARIAN_SCOUT so broker truth can enforce the one-active cap",
            )
        )
    expected_units = _positive_int(policy.get("units"), MIN_PRODUCTION_LOT_UNITS)
    if expected_units != MIN_PRODUCTION_LOT_UNITS or abs(int(intent.units)) != MIN_PRODUCTION_LOT_UNITS:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_MIN_LOT_REQUIRED",
                f"SCOUT must use exactly the broker production floor {MIN_PRODUCTION_LOT_UNITS}u",
            )
        )
    if intent.entry is None or not _intent_geometry_valid(intent):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_GEOMETRY_INVALID",
                "SCOUT requires side-consistent entry, TP, and invalidation geometry",
            )
        )
    if not _attached_exit_contract_valid(metadata):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_ATTACHED_EXITS_REQUIRED",
                "SCOUT requires attached technical TP and broker-side SL; timeout or operator close is not the planned exit",
            )
        )
    if metadata.get("predictive_scout_promotion_allowed") is not False:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_AUTO_PROMOTION_FORBIDDEN",
                "forward SCOUT metadata must explicitly forbid automatic promotion",
            )
        )

    expires_at = _parse_utc(metadata.get("predictive_scout_expires_at_utc"))
    created_at = _parse_utc(metadata.get("predictive_scout_generated_at_utc"))
    if expires_at is None or expires_at <= now:
        issues.append(_issue("PREDICTIVE_SCOUT_EXPIRED", "SCOUT prediction TTL is missing or expired"))
    if created_at is None or (expires_at is not None and created_at >= expires_at):
        issues.append(_issue("PREDICTIVE_SCOUT_TIMESTAMP_INVALID", "SCOUT creation/expiry timestamps are invalid"))
    if created_at is not None and created_at > now + timedelta(seconds=PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS):
        issues.append(_issue("PREDICTIVE_SCOUT_CREATED_IN_FUTURE", "SCOUT creation timestamp is ahead of broker validation time"))
    policy_ttl_min = min(
        _positive_int(policy.get("max_ttl_minutes"), PREDICTIVE_SCOUT_MAX_TTL_MINUTES),
        PREDICTIVE_SCOUT_MAX_TTL_MINUTES,
    )
    forecast_horizon_min = _positive_int(metadata.get("forecast_horizon_min"), 0)
    if not str(metadata.get("forecast_cycle_id") or "").strip():
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_FORECAST_CYCLE_MISSING",
                "SCOUT requires a non-empty forecast_cycle_id so one signal cannot be duplicated or counted as independent evidence",
            )
        )
    if forecast_horizon_min <= 0:
        issues.append(_issue("PREDICTIVE_SCOUT_FORECAST_HORIZON_MISSING", "SCOUT requires a positive current forecast horizon"))
    max_ttl_min = min(policy_ttl_min, forecast_horizon_min) if forecast_horizon_min > 0 else 0
    if created_at is not None and expires_at is not None:
        ttl_min = (expires_at - created_at).total_seconds() / 60.0
        if ttl_min <= 0 or ttl_min > max_ttl_min + 1e-9:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_TTL_TOO_LONG",
                    f"SCOUT TTL {ttl_min:.1f}m exceeds current forecast/policy maximum {max_ttl_min}m",
                )
            )
        declared_ttl = _safe_float(metadata.get("predictive_scout_ttl_minutes"), -1.0)
        if declared_ttl <= 0.0 or abs(declared_ttl - ttl_min) > 1e-6:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_TTL_METADATA_MISMATCH",
                    "SCOUT declared TTL must equal the signed creation/expiry interval",
                )
            )

    rule = metadata.get("bidask_replay_precision_seed_rule")
    if isinstance(rule, dict):
        if _nonnegative_int(rule.get("samples")) < _positive_int(
            policy.get("minimum_replay_samples"),
            PREDICTIVE_SCOUT_MIN_REPLAY_SAMPLES,
        ):
            issues.append(_issue("PREDICTIVE_SCOUT_SAMPLE_FLOOR_NOT_MET", "SCOUT replay sample floor is not met"))
        if _safe_float(rule.get("optimized_profit_factor"), 0.0) < _safe_float(
            policy.get("minimum_profit_factor"),
            PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR,
        ):
            issues.append(_issue("PREDICTIVE_SCOUT_PROFIT_FACTOR_NOT_MET", "SCOUT replay profit-factor floor is not met"))
        if _safe_float(rule.get("positive_day_rate"), 0.0) < _safe_float(
            policy.get("minimum_positive_day_rate"),
            PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE,
        ):
            issues.append(_issue("PREDICTIVE_SCOUT_DAILY_STABILITY_NOT_MET", "SCOUT positive-day-rate floor is not met"))
        if _nonnegative_int(rule.get("active_days")) < _positive_int(
            policy.get("minimum_active_days"),
            PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS,
        ):
            issues.append(_issue("PREDICTIVE_SCOUT_ACTIVE_DAY_FLOOR_NOT_MET", "SCOUT active-day floor is not met"))
        if _safe_float(rule.get("optimized_avg_realized_pips"), 0.0) <= 0.0:
            issues.append(_issue("PREDICTIVE_SCOUT_NET_EDGE_NOT_POSITIVE", "SCOUT replay net expectancy must be positive after bid/ask costs"))

    max_daily = min(
        _positive_int(policy.get("max_sent_per_campaign_day"), PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY),
        PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY,
    )
    sent_today = predictive_scout_sent_count(ledger_path, now=now)
    metadata["predictive_scout_sent_today"] = sent_today
    metadata["predictive_scout_max_sent_per_campaign_day"] = max_daily
    if sent_today is None:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_LEDGER_UNAVAILABLE",
                "SCOUT send/loss history cannot be proven from execution_ledger.db",
            )
        )
    elif sent_today >= max_daily:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_DAILY_CAP_REACHED",
                f"SCOUT live-send cap reached for the campaign day ({sent_today}/{max_daily})",
            )
        )

    max_concurrent = min(
        _positive_int(policy.get("max_concurrent"), PREDICTIVE_SCOUT_MAX_CONCURRENT),
        PREDICTIVE_SCOUT_MAX_CONCURRENT,
    )
    concurrent = predictive_scout_concurrent_count(snapshot) if snapshot is not None else None
    metadata["predictive_scout_concurrent"] = concurrent
    metadata["predictive_scout_max_concurrent"] = max_concurrent
    if concurrent is None:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_SNAPSHOT_REQUIRED",
                "fresh broker positions and pending orders are required to prove the one-SCOUT concurrency cap",
            )
        )
    elif concurrent >= max_concurrent:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_CONCURRENT_CAP_REACHED",
                f"another SCOUT position/pending entry is active ({concurrent}/{max_concurrent})",
            )
        )
    broker_last_transaction_id = (
        str(snapshot.account.last_transaction_id or "").strip()
        if snapshot is not None and snapshot.account is not None
        else ""
    )
    ledger_last_transaction_id = _execution_ledger_last_transaction_id(
        ledger_path
    )
    metadata["predictive_scout_broker_last_transaction_id"] = broker_last_transaction_id or None
    metadata["predictive_scout_ledger_last_transaction_id"] = ledger_last_transaction_id
    if (
        not broker_last_transaction_id
        or not ledger_last_transaction_id
        or ledger_last_transaction_id != broker_last_transaction_id
    ):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_LEDGER_NOT_CURRENT",
                "fresh broker lastTransactionID must exactly match execution-ledger sync state before SCOUT staging; sync and rebuild the intent instead of reusing stale loss/count history",
            )
        )
    vehicle_outcomes = predictive_scout_vehicle_outcome_stats(
        ledger_path,
        intent=intent,
    )
    if vehicle_outcomes is None:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_OUTCOME_HISTORY_UNAVAILABLE",
                "SCOUT vehicle outcomes cannot be reconciled from all resolved exits",
            )
        )
    else:
        loss_count = int(vehicle_outcomes["loss_count"])
        net_jpy = float(vehicle_outcomes["net_jpy"])
        last_loss_at = _parse_utc(vehicle_outcomes.get("last_loss_at_utc"))
        metadata["predictive_scout_losing_vehicle_count"] = loss_count
        metadata["predictive_scout_vehicle_net_jpy"] = round(net_jpy, 4)
        metadata["predictive_scout_last_loss_at_utc"] = (
            last_loss_at.isoformat() if last_loss_at is not None else None
        )
        cooldown_hours = max(
            PREDICTIVE_SCOUT_LOSS_COOLDOWN_HOURS,
            _positive_int(
                policy.get("loss_cooldown_hours"),
                PREDICTIVE_SCOUT_LOSS_COOLDOWN_HOURS,
            ),
        )
        if loss_count > 0 and last_loss_at is None:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_LOSS_TIMESTAMP_UNAVAILABLE",
                    "resolved SCOUT loss lacks a durable close timestamp; do not retry until ledger reconciliation repairs it",
                )
            )
        elif last_loss_at is not None and now < last_loss_at + timedelta(hours=cooldown_hours):
            retry_at = last_loss_at + timedelta(hours=cooldown_hours)
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_VEHICLE_LOSS_COOLDOWN",
                    f"exact SCOUT vehicle lost at {last_loss_at.isoformat()}; wait until {retry_at.isoformat()} and require a fresh forecast cycle before retrying",
                )
            )
        loss_limit = min(
            PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES,
            _positive_int(
                policy.get("quarantine_after_resolved_losses"),
                PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES,
            ),
        )
        if loss_count >= loss_limit and net_jpy < 0.0:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_VEHICLE_QUARANTINED_NEGATIVE_NET",
                    f"exact SCOUT vehicle has {loss_count} resolved losses and net {net_jpy:+.1f} JPY; rotate to a materially different selector or exit vehicle instead of repeating the losing shape",
                )
            )
    return issues


def _execution_ledger_last_transaction_id(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT value FROM sync_state WHERE key = 'last_oanda_transaction_id'"
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    value = str(row[0] or "").strip() if row else ""
    return value or None


def predictive_scout_sent_count(db_path: Path, *, now: datetime) -> int | None:
    """Count today's reserved SCOUT broker POST budget, deduplicated by experiment.

    A durable reservation is written immediately before the broker POST.  It
    consumes one daily slot even if the process dies before the final SENT
    receipt is written; otherwise a crash could silently exceed the campaign
    cap.  Legacy sent receipts remain part of the count.
    """

    if not db_path.exists():
        return None
    day = now.astimezone(timezone.utc).date().isoformat()
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            receipt_rows = con.execute(
                """
                SELECT rowid, payload_json
                FROM gateway_receipts
                WHERE sent = 1
                  AND substr(ts_utc, 1, 10) = ?
                """,
                (day,),
            ).fetchall()
            event_rows = con.execute(
                """
                SELECT rowid, event_type, raw_json
                FROM execution_events
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'GATEWAY_ORDER_STAGED')
                  AND substr(ts_utc, 1, 10) = ?
                """,
                (day,),
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    counted: set[str] = set()
    for rowid, raw in receipt_rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if _payload_has_predictive_scout(payload):
            counted.update(_predictive_scout_budget_keys(payload, fallback=f"receipt:{rowid}"))
    for rowid, event_type, raw in event_rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        is_reservation = bool(
            payload.get("predictive_scout_post_reserved") is True
            or str(payload.get("status") or "").upper() == "PREDICTIVE_SCOUT_POST_RESERVED"
        )
        if _payload_has_predictive_scout(payload) and (
            str(event_type) == "GATEWAY_ORDER_SENT" or is_reservation
        ):
            counted.update(_predictive_scout_budget_keys(payload, fallback=f"event:{rowid}"))
    return len(counted)


def predictive_scout_concurrent_count(snapshot: BrokerSnapshot | None) -> int:
    if snapshot is None:
        return 0
    count = 0
    for position in snapshot.positions:
        if predictive_scout_broker_raw_claimed(position.raw):
            count += 1
    for order in snapshot.orders:
        if not order.trade_id and predictive_scout_broker_raw_claimed(order.raw):
            count += 1
    return count


def predictive_scout_broker_vehicle_counts(
    snapshot: BrokerSnapshot | None,
) -> dict[str, int]:
    """Count active broker-visible SCOUTs by stable vehicle id for diagnostics."""

    if snapshot is None:
        return {}
    counts: dict[str, int] = {}
    broker_rows = [
        position.raw
        for position in snapshot.positions
        if predictive_scout_broker_raw_claimed(position.raw)
    ]
    broker_rows.extend(
        order.raw
        for order in snapshot.orders
        if not order.trade_id and predictive_scout_broker_raw_claimed(order.raw)
    )
    for raw in broker_rows:
        vehicle_id = _raw_scout_vehicle_id(raw)
        if not vehicle_id:
            continue
        counts[vehicle_id] = counts.get(vehicle_id, 0) + 1
    return counts


def predictive_scout_broker_signal_ids(
    snapshot: BrokerSnapshot | None,
) -> set[str]:
    """Return exact broker-reflected signal ids for atomic slot reconciliation.

    Vehicle ids are intentionally insufficient here: an older filled position
    and a newer in-flight reservation can share the same vehicle.  Only the
    exact signal id may cancel the corresponding local claim's shadow slot.
    Missing or truncated signal ids stay absent, which is conservative because
    the broker object still contributes to the global concurrent count.
    """

    if snapshot is None:
        return set()
    signal_ids: set[str] = set()
    broker_rows = [
        position.raw
        for position in snapshot.positions
        if predictive_scout_broker_raw_claimed(position.raw)
    ]
    broker_rows.extend(
        order.raw
        for order in snapshot.orders
        if not order.trade_id and predictive_scout_broker_raw_claimed(order.raw)
    )
    for raw in broker_rows:
        signal_id = _raw_scout_signal_id(raw)
        if signal_id:
            signal_ids.add(signal_id)
    return signal_ids


def predictive_scout_vehicle_id(intent: OrderIntent) -> str:
    """Stable failure-memory id for a selector plus executable exit vehicle.

    The full rule digest authenticates the current evidence package, but is
    deliberately excluded here.  Incrementing samples or recomputing PF must
    not erase cooldown/quarantine for the same market selector and TP/SL.
    """

    metadata = intent.metadata or {}
    rule = (
        metadata.get("bidask_replay_precision_seed_rule")
        if isinstance(metadata.get("bidask_replay_precision_seed_rule"), dict)
        else {}
    )
    precision = 3 if intent.pair.endswith("_JPY") else 5
    entry = float(intent.entry or 0.0)
    payload = {
        "source": PREDICTIVE_SCOUT_SOURCE,
        "pair": intent.pair.upper(),
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": abs(int(intent.units)),
        "method": (
            intent.market_context.method.value if intent.market_context is not None else ""
        ),
        "forecast_direction": str(rule.get("forecast_direction") or "").upper(),
        "faded_direction": str(rule.get("faded_direction") or "").upper(),
        "horizon_bucket": str(rule.get("horizon_bucket") or ""),
        "confidence_bucket": str(rule.get("confidence_bucket") or ""),
        "granularity": str(rule.get("granularity") or "").upper(),
        "target_distance": f"{abs(float(intent.tp) - entry):.{precision}f}",
        "stop_distance": f"{abs(entry - float(intent.sl)):.{precision}f}",
    }
    return "psv-" + _stable_digest(payload)[:24]


def predictive_scout_experiment_id(intent: OrderIntent) -> str:
    """Unique id for one forecast cycle's exact forward experiment."""

    metadata = intent.metadata or {}
    payload = {
        "vehicle_id": predictive_scout_vehicle_id(intent),
        "forecast_cycle_id": str(metadata.get("forecast_cycle_id") or ""),
        "generated_at_utc": str(metadata.get("predictive_scout_generated_at_utc") or ""),
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
    }
    return "psx-" + _stable_digest(payload)[:24]


def predictive_scout_signal_id(intent: OrderIntent) -> str:
    """One independent forecast-cycle claim for a stable failure vehicle."""

    metadata = intent.metadata or {}
    payload = {
        "vehicle_id": predictive_scout_vehicle_id(intent),
        "forecast_cycle_id": str(metadata.get("forecast_cycle_id") or "").strip(),
    }
    return "pss-" + _stable_digest(payload)[:24]


def predictive_scout_losing_vehicle_count(
    db_path: Path,
    *,
    intent: OrderIntent,
) -> int | None:
    """Count fully resolved, all-exit net losses for this durable vehicle id."""

    stats = predictive_scout_vehicle_outcome_stats(db_path, intent=intent)
    return None if stats is None else int(stats["loss_count"])


def predictive_scout_vehicle_outcome_stats(
    db_path: Path,
    *,
    intent: OrderIntent,
) -> dict[str, Any] | None:
    extracted = _predictive_scout_forward_outcomes(db_path)
    if extracted is None:
        return None
    state = extracted.get(predictive_scout_vehicle_id(intent))
    outcomes = list(state.get("outcomes") or []) if isinstance(state, dict) else []
    losses = [item for item in outcomes if _safe_float(item.get("net_jpy"), 0.0) < 0.0]
    loss_timestamps = [
        parsed
        for parsed in (_parse_utc(item.get("resolved_at_utc")) for item in losses)
        if parsed is not None
    ]
    return {
        "resolved_count": len(outcomes),
        "loss_count": len(losses),
        "net_jpy": sum(_safe_float(item.get("net_jpy"), 0.0) for item in outcomes),
        "last_loss_at_utc": max(loss_timestamps).isoformat() if loss_timestamps else None,
    }


def predictive_scout_forward_proof(
    db_path: Path,
    *,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate forward expectancy from every fully resolved SCOUT exit."""

    generated_at = datetime.now(timezone.utc).isoformat()
    policy = predictive_scout_policy(policy_path)
    if not _policy_contract_valid(policy):
        return {
            "generated_at_utc": generated_at,
            "status": "POLICY_INVALID",
            "promotion_allowed": False,
            "vehicles": [],
        }
    extracted = _predictive_scout_forward_outcomes(db_path)
    if extracted is None:
        return {
            "generated_at_utc": generated_at,
            "status": "LEDGER_UNAVAILABLE",
            "promotion_allowed": False,
            "vehicles": [],
        }
    min_resolved = max(
        PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS,
        _positive_int(
            policy.get("promotion_min_resolved_exits"),
            PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS,
        ),
    )
    min_days = max(
        PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS,
        _positive_int(policy.get("promotion_min_active_days"), PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS),
    )
    min_pf = max(
        PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR,
        _safe_float(policy.get("promotion_min_profit_factor"), PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR),
    )
    min_positive_day_rate = max(
        PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE,
        _safe_float(
            policy.get("promotion_min_positive_day_rate"),
            PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE,
        ),
    )
    vehicles = [
        _predictive_scout_vehicle_proof_row(
            vehicle_id,
            state,
            min_resolved=min_resolved,
            min_days=min_days,
            min_profit_factor=min_pf,
            min_positive_day_rate=min_positive_day_rate,
        )
        for vehicle_id, state in sorted(extracted.items())
    ]
    any_eligible = any(item["statistically_eligible_for_operator_review"] for item in vehicles)
    return {
        "generated_at_utc": generated_at,
        "status": (
            "PROOF_ELIGIBLE_FOR_OPERATOR_REVIEW"
            if any_eligible
            else ("COLLECTING_FORWARD_EVIDENCE" if vehicles else "NO_SCOUT_SAMPLES")
        ),
        "promotion_allowed": False,
        "future_profit_guaranteed": False,
        "requirements": {
            "minimum_resolved_exits": min_resolved,
            "all_filled_must_be_resolved": True,
            "one_sided_95_mean_lower_jpy_must_exceed": 0.0,
            "minimum_profit_factor": min_pf,
            "minimum_active_days": min_days,
            "minimum_positive_day_rate": min_positive_day_rate,
            "all_exit_reasons_and_financing_included": True,
            "one_broker_order_per_vehicle_forecast_signal": True,
            "every_resolved_trade_has_one_independent_signal": True,
        },
        "vehicles": vehicles,
    }


def _predictive_scout_vehicle_proof_row(
    vehicle_id: str,
    state: dict[str, Any],
    *,
    min_resolved: int,
    min_days: int,
    min_profit_factor: float,
    min_positive_day_rate: float,
) -> dict[str, Any]:
    outcomes = list(state.get("outcomes") or [])
    values = [float(item["net_jpy"]) for item in outcomes]
    resolved_count = len(values)
    wins = sum(1 for value in values if value > 0.0)
    losses = sum(1 for value in values if value < 0.0)
    gross_profit = sum(value for value in values if value > 0.0)
    gross_loss_abs = abs(sum(value for value in values if value < 0.0))
    mean = sum(values) / resolved_count if resolved_count else None
    lower = _one_sided_mean_lower_95(values)
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0.0 else None
    daily_net: dict[str, float] = {}
    exit_reasons: dict[str, int] = {}
    for item in outcomes:
        day = str(item.get("resolved_day_utc") or "unknown")
        daily_net[day] = daily_net.get(day, 0.0) + float(item["net_jpy"])
        for reason in item.get("exit_reasons", []) or ["UNKNOWN"]:
            key = str(reason or "UNKNOWN")
            exit_reasons[key] = exit_reasons.get(key, 0) + 1
    active_days = len(daily_net)
    positive_days = sum(1 for value in daily_net.values() if value > 0.0)
    positive_day_rate = positive_days / active_days if active_days else 0.0
    filled_count = len(state.get("filled_trade_ids") or set())
    unresolved_count = max(0, filled_count - resolved_count)
    all_filled_resolved = filled_count > 0 and unresolved_count == 0
    signal_broker_refs = state.get("signal_broker_refs") or {}
    filled_signal_trade_ids = state.get("filled_signal_trade_ids") or {}
    duplicate_signal_ids = sorted(
        str(signal_id)
        for signal_id, broker_refs in signal_broker_refs.items()
        if len(broker_refs) > 1
    )
    independent_signal_count = len(filled_signal_trade_ids)
    missing_signal_count = max(0, filled_count - independent_signal_count)
    complete_signal_attribution = bool(
        filled_count > 0
        and independent_signal_count == filled_count
        and filled_count == resolved_count
        and all(len(trade_ids) == 1 for trade_ids in filled_signal_trade_ids.values())
        and not duplicate_signal_ids
    )
    profit_factor_pass = gross_loss_abs == 0.0 and gross_profit > 0.0
    if profit_factor is not None:
        profit_factor_pass = profit_factor >= min_profit_factor
    eligible = bool(
        all_filled_resolved
        and complete_signal_attribution
        and resolved_count >= min_resolved
        and lower is not None
        and lower > 0.0
        and profit_factor_pass
        and active_days >= min_days
        and positive_day_rate >= min_positive_day_rate
    )
    return {
        "predictive_scout_vehicle_id": vehicle_id,
        "predictive_scout_rule_digest": state.get("rule_digest"),
        "predictive_scout_rule_name": state.get("rule_name"),
        "pair": state.get("pair"),
        "side": state.get("side"),
        "sent_count": int(state.get("sent_count") or 0),
        "filled_count": filled_count,
        "resolved_count": resolved_count,
        "unresolved_filled_count": unresolved_count,
        "all_filled_resolved": all_filled_resolved,
        "independent_signal_count": independent_signal_count,
        "reserved_signal_count": len(signal_broker_refs),
        "unfilled_signal_count": max(
            0, len(signal_broker_refs) - independent_signal_count
        ),
        "missing_signal_count": missing_signal_count,
        "complete_signal_attribution": complete_signal_attribution,
        "duplicate_signal_count": len(duplicate_signal_ids),
        "duplicate_signal_ids": duplicate_signal_ids,
        "wins": wins,
        "losses": losses,
        "net_jpy": round(sum(values), 4),
        "mean_net_jpy": round(mean, 4) if mean is not None else None,
        "one_sided_95_mean_lower_jpy": round(lower, 4) if lower is not None else None,
        "gross_profit_jpy": round(gross_profit, 4),
        "gross_loss_abs_jpy": round(gross_loss_abs, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "active_days": active_days,
        "positive_days": positive_days,
        "positive_day_rate": round(positive_day_rate, 6),
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "quarantined_negative_vehicle": (
            losses >= PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES
            and sum(values) < 0.0
        ),
        "statistically_eligible_for_operator_review": eligible,
        "automatic_promotion_allowed": False,
    }


def _predictive_scout_forward_outcomes(
    db_path: Path,
) -> dict[str, dict[str, Any]] | None:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            event_columns = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(execution_events)").fetchall()
            }
            client_order_expr = (
                "client_order_id" if "client_order_id" in event_columns else "NULL AS client_order_id"
            )
            gateway_rows = con.execute(
                f"""
                SELECT event_type, order_id, trade_id, {client_order_expr}, raw_json
                FROM execution_events
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'GATEWAY_ORDER_STAGED')
                """
            ).fetchall()
            fill_rows = con.execute(
                f"""
                SELECT order_id, trade_id, {client_order_expr} FROM execution_events
                WHERE event_type = 'ORDER_FILLED' AND trade_id IS NOT NULL AND trade_id != ''
                """
            ).fetchall()
            outcome_rows = con.execute(
                """
                SELECT trade_id, event_type, realized_pl_jpy, financing_jpy, ts_utc, exit_reason
                FROM execution_events
                WHERE event_type IN ('TRADE_REDUCED', 'TRADE_CLOSED')
                  AND trade_id IS NOT NULL AND trade_id != ''
                  AND realized_pl_jpy IS NOT NULL
                """
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    vehicles: dict[str, dict[str, Any]] = {}
    order_to_vehicles: dict[str, set[str]] = {}
    trade_to_vehicles: dict[str, set[str]] = {}
    client_to_vehicles: dict[str, set[str]] = {}
    order_to_signals: dict[str, set[tuple[str, str]]] = {}
    trade_to_signals: dict[str, set[tuple[str, str]]] = {}
    client_to_signals: dict[str, set[tuple[str, str]]] = {}
    counted_signals: set[str] = set()
    for event_type, order_id, trade_id, client_order_id, raw in gateway_rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        for receipt in _predictive_scout_receipts(payload):
            vehicle_id = str(receipt.get("predictive_scout_vehicle_id") or "")
            if not vehicle_id:
                return None
            state = vehicles.setdefault(
                vehicle_id,
                {
                    "sent_count": 0,
                    "filled_trade_ids": set(),
                    "signal_broker_refs": {},
                    "filled_signal_trade_ids": {},
                    "outcomes": [],
                    "rule_digest": receipt.get("predictive_scout_rule_digest"),
                    "rule_name": receipt.get("predictive_scout_rule_name"),
                    "pair": receipt.get("pair"),
                    "side": receipt.get("side"),
                },
            )
            experiment_id = str(receipt.get("predictive_scout_experiment_id") or "")
            signal_id = _predictive_scout_receipt_signal_id(receipt, vehicle_id=vehicle_id)
            is_post_reservation = bool(
                payload.get("predictive_scout_post_reserved") is True
                or str(payload.get("status") or "").upper() == "PREDICTIVE_SCOUT_POST_RESERVED"
            )
            if event_type == "GATEWAY_ORDER_SENT" or is_post_reservation:
                count_key = signal_id or experiment_id
                if not count_key or count_key not in counted_signals:
                    state["sent_count"] = int(state["sent_count"]) + 1
                    if count_key:
                        counted_signals.add(count_key)
            signal_ref = (vehicle_id, signal_id) if signal_id else None
            broker_ref = (
                f"order:{order_id}" if order_id else (f"trade:{trade_id}" if trade_id else None)
            )
            if signal_ref is not None and broker_ref is not None:
                state["signal_broker_refs"].setdefault(signal_id, set()).add(broker_ref)
            if order_id:
                order_to_vehicles.setdefault(str(order_id), set()).add(vehicle_id)
                if signal_ref is not None:
                    order_to_signals.setdefault(str(order_id), set()).add(signal_ref)
            if trade_id:
                trade_text = str(trade_id)
                trade_to_vehicles.setdefault(trade_text, set()).add(vehicle_id)
                if signal_ref is not None:
                    trade_to_signals.setdefault(trade_text, set()).add(signal_ref)
                state["filled_trade_ids"].add(trade_text)
            if client_order_id:
                client_to_vehicles.setdefault(str(client_order_id), set()).add(vehicle_id)
                if signal_ref is not None:
                    client_to_signals.setdefault(str(client_order_id), set()).add(signal_ref)
    for order_id, trade_id, client_order_id in fill_rows:
        vehicle_ids: set[str] = set()
        signal_refs: set[tuple[str, str]] = set()
        if order_id:
            vehicle_ids.update(order_to_vehicles.get(str(order_id), set()))
            signal_refs.update(order_to_signals.get(str(order_id), set()))
        if trade_id:
            vehicle_ids.update(trade_to_vehicles.get(str(trade_id), set()))
            signal_refs.update(trade_to_signals.get(str(trade_id), set()))
        if client_order_id:
            vehicle_ids.update(client_to_vehicles.get(str(client_order_id), set()))
            signal_refs.update(client_to_signals.get(str(client_order_id), set()))
        for vehicle_id in vehicle_ids:
            trade_text = str(trade_id)
            trade_to_vehicles.setdefault(trade_text, set()).add(vehicle_id)
            vehicles[vehicle_id]["filled_trade_ids"].add(trade_text)
        broker_ref = f"order:{order_id}" if order_id else f"trade:{trade_id}"
        for vehicle_id, signal_id in signal_refs:
            trade_to_signals.setdefault(str(trade_id), set()).add((vehicle_id, signal_id))
            vehicles[vehicle_id]["signal_broker_refs"].setdefault(signal_id, set()).add(
                broker_ref
            )
            vehicles[vehicle_id]["filled_signal_trade_ids"].setdefault(
                signal_id, set()
            ).add(str(trade_id))
    per_trade: dict[str, dict[str, Any]] = {}
    for trade_id, event_type, realized_pl, financing, ts_utc, exit_reason in outcome_rows:
        trade_text = str(trade_id)
        if trade_text not in trade_to_vehicles:
            continue
        state = per_trade.setdefault(
            trade_text,
            {
                "net_jpy": 0.0,
                "resolved": False,
                "resolved_at_utc": None,
                "resolved_day_utc": None,
                "exit_reasons": [],
            },
        )
        state["net_jpy"] = (
            float(state["net_jpy"])
            + _safe_float(realized_pl, 0.0)
            + _safe_float(financing, 0.0)
        )
        state["exit_reasons"].append(str(exit_reason or event_type or "UNKNOWN"))
        if str(event_type) == "TRADE_CLOSED":
            state["resolved"] = True
            state["resolved_at_utc"] = str(ts_utc or "") or None
            state["resolved_day_utc"] = str(ts_utc or "")[:10] or "unknown"
    for trade_id, outcome in per_trade.items():
        if not outcome["resolved"]:
            continue
        for vehicle_id in trade_to_vehicles.get(trade_id, set()):
            vehicles[vehicle_id]["outcomes"].append(dict(outcome))
    return vehicles


def _one_sided_mean_lower_95(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean - 1.70 * math.sqrt(max(0.0, variance) / n)


def write_predictive_scout_forward_proof(
    *,
    db_path: Path,
    json_path: Path,
    report_path: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    payload = predictive_scout_forward_proof(db_path, policy_path=policy_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Predictive SCOUT Forward Proof",
        "",
        f"- Status: `{payload.get('status')}`",
        "- Automatic promotion: `false`",
        "- Scope: all fully resolved exits, partial-close aggregation, financing and execution costs",
        "- Meaning: a positive lower bound is statistical forward evidence, not a future-profit guarantee",
        "",
        "## Vehicles",
        "",
    ]
    for item in payload.get("vehicles", []) or []:
        lines.extend(
            [
                f"### {item.get('predictive_scout_vehicle_id')}",
                "",
                f"- Rule: `{item.get('predictive_scout_rule_name')}`",
                f"- Pair/side: `{item.get('pair')} {item.get('side')}`",
                f"- Sent / filled / resolved / unresolved: `{item.get('sent_count')} / {item.get('filled_count')} / {item.get('resolved_count')} / {item.get('unresolved_filled_count')}`",
                f"- Net / mean / one-sided 95% lower: `{item.get('net_jpy')} / {item.get('mean_net_jpy')} / {item.get('one_sided_95_mean_lower_jpy')} JPY`",
                f"- PF / positive-day rate: `{item.get('profit_factor')} / {item.get('positive_day_rate')}`",
                f"- Quarantined negative vehicle: `{item.get('quarantined_negative_vehicle')}`",
                f"- Eligible for operator review: `{item.get('statistically_eligible_for_operator_review')}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def _intent_geometry_valid(intent: OrderIntent) -> bool:
    if intent.entry is None:
        return False
    if intent.side == Side.LONG:
        return intent.tp > intent.entry > intent.sl
    return intent.tp < intent.entry < intent.sl


def _canonical_scout_exit_geometry_matches(
    intent: OrderIntent,
    canonical_candidate: dict[str, Any],
) -> bool:
    if intent.entry is None:
        return False
    try:
        pip_factor = float(instrument_pip_factor(intent.pair))
        expected_tp = float(canonical_candidate["optimized_take_profit_pips"])
        expected_sl = float(canonical_candidate["optimized_stop_loss_pips"])
        actual_tp = abs(float(intent.tp) - float(intent.entry)) * pip_factor
        actual_sl = abs(float(intent.entry) - float(intent.sl)) * pip_factor
    except (KeyError, TypeError, ValueError):
        return False
    # OANDA's 0.1-pip price tick can move a rounded distance by at most half a
    # tick.  Anything beyond that is a different forward vehicle.
    return bool(
        math.isclose(actual_tp, expected_tp, rel_tol=0.0, abs_tol=0.051)
        and math.isclose(actual_sl, expected_sl, rel_tol=0.0, abs_tol=0.051)
    )


def _attached_exit_contract_valid(metadata: dict[str, Any]) -> bool:
    return bool(
        metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").upper() == "ATTACHED_TECHNICAL_TP"
        and str(metadata.get("tp_target_intent") or "").upper() == "HARVEST"
        and str(metadata.get("broker_stop_loss_mode") or "").upper() == "INTENT_SL"
    )


def _embedded_rule_matches_canonical(
    embedded: dict[str, Any],
    canonical: dict[str, Any],
) -> bool:
    immutable_keys = (
        "name",
        "pair",
        "side",
        "direction",
        "forecast_direction",
        "faded_direction",
        "contrarian_edge",
        "horizon_bucket",
        "confidence_bucket",
        "granularity",
        "samples",
        "optimized_take_profit_pips",
        "optimized_stop_loss_pips",
        "optimized_avg_realized_pips",
        "optimized_profit_factor",
        "adoption_status",
        "live_grade",
        "daily_stability_status",
        "active_days",
        "positive_day_rate",
        "min_target_pips",
        "max_target_pips",
        "max_stop_pips",
        "rule_set_generated_at_utc",
        "rule_set_source",
    )
    for key in immutable_keys:
        if canonical.get(key) != embedded.get(key):
            return False
    return list(embedded.get("adoption_blockers") or []) == list(canonical.get("adoption_blockers") or [])


def _policy_contract_valid(policy: dict[str, Any]) -> bool:
    if not policy:
        return False
    try:
        schema_version = int(policy.get("schema_version"))
        units = int(policy.get("units"))
        max_concurrent = int(policy.get("max_concurrent"))
        max_daily = int(policy.get("max_sent_per_campaign_day"))
        max_ttl = int(policy.get("max_ttl_minutes"))
        minimum_samples = int(policy.get("minimum_replay_samples"))
        minimum_active_days = int(policy.get("minimum_active_days"))
        minimum_profit_factor = float(policy.get("minimum_profit_factor"))
        minimum_positive_day_rate = float(policy.get("minimum_positive_day_rate"))
        promotion_min_resolved = int(policy.get("promotion_min_resolved_exits"))
        promotion_min_active_days = int(policy.get("promotion_min_active_days"))
        promotion_min_profit_factor = float(policy.get("promotion_min_profit_factor"))
        promotion_min_positive_day_rate = float(policy.get("promotion_min_positive_day_rate"))
        promotion_confidence = float(policy.get("promotion_one_sided_confidence"))
        loss_cooldown_hours = int(policy.get("loss_cooldown_hours"))
        quarantine_after_losses = int(policy.get("quarantine_after_resolved_losses"))
    except (TypeError, ValueError):
        return False
    order_types = {str(item).upper() for item in policy.get("order_types", []) or []}
    allowed_sources = {str(item).upper() for item in policy.get("allowed_sources", []) or []}
    return bool(
        schema_version == 1
        and str(policy.get("mode") or "").upper() == "FORWARD_EVIDENCE_ONLY"
        and units == MIN_PRODUCTION_LOT_UNITS
        and max_concurrent == PREDICTIVE_SCOUT_MAX_CONCURRENT
        and 0 < max_daily <= PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY
        and 0 < max_ttl <= PREDICTIVE_SCOUT_MAX_TTL_MINUTES
        and order_types == {OrderType.LIMIT.value}
        and allowed_sources == {PREDICTIVE_SCOUT_SOURCE}
        and minimum_samples >= PREDICTIVE_SCOUT_MIN_REPLAY_SAMPLES
        and minimum_active_days >= PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS
        and minimum_profit_factor >= PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR
        and minimum_positive_day_rate >= PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE
        and promotion_min_resolved >= PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS
        and promotion_min_active_days >= PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS
        and promotion_min_profit_factor >= PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR
        and promotion_min_positive_day_rate >= PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE
        and promotion_confidence >= 0.95
        and loss_cooldown_hours >= PREDICTIVE_SCOUT_LOSS_COOLDOWN_HOURS
        and quarantine_after_losses <= PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES
        and quarantine_after_losses > 0
        and policy.get("quarantine_requires_negative_net") is True
        and policy.get("promotion_requires_all_resolved_exit_expectancy_lower_bound_positive") is True
    )


def _payload_has_predictive_scout(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("predictive_scout") is True:
            return True
        return any(_payload_has_predictive_scout(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_has_predictive_scout(value) for value in payload)
    return False


def _predictive_scout_budget_keys(payload: Any, *, fallback: str) -> set[str]:
    signal_ids = {
        str(receipt.get("predictive_scout_signal_id") or "").strip()
        for receipt in _predictive_scout_receipts(payload)
    }
    signal_ids.discard("")
    if signal_ids:
        return {f"signal:{signal_id}" for signal_id in signal_ids}
    experiment_ids = {
        str(receipt.get("predictive_scout_experiment_id") or "").strip()
        for receipt in _predictive_scout_receipts(payload)
    }
    experiment_ids.discard("")
    if experiment_ids:
        return {f"experiment:{experiment_id}" for experiment_id in experiment_ids}
    return {fallback}


def _predictive_scout_receipt_signal_id(
    receipt: dict[str, Any],
    *,
    vehicle_id: str,
) -> str:
    declared = str(receipt.get("predictive_scout_signal_id") or "").strip()
    if declared:
        return declared
    forecast_cycle_id = str(receipt.get("forecast_cycle_id") or "").strip()
    if not forecast_cycle_id:
        return ""
    return "pss-" + _stable_digest(
        {"vehicle_id": vehicle_id, "forecast_cycle_id": forecast_cycle_id}
    )[:24]


def _predictive_scout_receipts(payload: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if (
            payload.get("predictive_scout") is True
            and payload.get("predictive_scout_vehicle_id")
        ):
            receipts.append(payload)
        for value in payload.values():
            receipts.extend(_predictive_scout_receipts(value))
    elif isinstance(payload, list):
        for value in payload:
            receipts.extend(_predictive_scout_receipts(value))
    return receipts


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _raw_has_scout_role(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    for key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, dict):
            continue
        comment = str(extension.get("comment") or "").upper()
        if "ROLE=BIDASK_REPLAY_CONTRARIAN_SCOUT" in comment or "ROLE=BIDASK_REPLAY_PRECISION_SCOUT" in comment:
            return True
    return False


def _raw_scout_vehicle_id(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    for key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, dict):
            continue
        for token in str(extension.get("comment") or "").split():
            if token.lower().startswith("vehicle="):
                vehicle_id = token.split("=", 1)[1].strip()
                if vehicle_id.startswith("psv-"):
                    return vehicle_id
    return None


def _raw_scout_signal_id(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    for key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, dict):
            continue
        extension_id = str(extension.get("id") or "")
        signal_offset = extension_id.rfind("pss-")
        if signal_offset >= 0:
            signal_id = extension_id[signal_offset : signal_offset + 28]
            if signal_id.startswith("pss-"):
                return signal_id
        for token in str(extension.get("comment") or "").split():
            if token.lower().startswith("signal="):
                signal_id = token.split("=", 1)[1].strip()
                if signal_id.startswith("pss-"):
                    return signal_id
    return None


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "severity": "BLOCK"}
