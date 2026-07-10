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
PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY = 30
PREDICTIVE_SCOUT_MAX_CONCURRENT = 2
PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS = 60
PREDICTIVE_SCOUT_LOSS_COOLDOWN_HOURS = 6
PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES = 3
PREDICTIVE_SCOUT_MIN_REPLAY_SAMPLES = 30
PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS = 5
PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR = 1.2
PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE = 2.0 / 3.0
PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS = 30
PREDICTIVE_SCOUT_NORMALIZATION_UNITS = 1000
PREDICTIVE_SCOUT_POLICY_SCHEMA_VERSION = 2
PREDICTIVE_SCOUT_MAX_PER_TRADE_RISK_PCT_NAV = 1.0
PREDICTIVE_SCOUT_MAX_CONCURRENT_RISK_PCT_NAV = 2.0
# These are named forward-evidence risk tiers, not discretionary lot
# multipliers.  Each percentage is applied to fresh broker NAV and converted
# to units from the intent's market-derived stop distance by the caller.  A
# tier may only advance from normalized, exact-vehicle forward outcomes.
PREDICTIVE_SCOUT_RISK_TIER_PCT_NAV = {
    "DISCOVERY": 0.10,
    "EMERGING": 0.25,
    "ESTABLISHED": 0.50,
    "STRONG": 0.75,
    "PROVEN": 1.00,
}


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
                "predictive SCOUT policy must be schema v2 forward-evidence-only with named NAV-risk tiers, no fixed units, at most 1% NAV per trade / 2% concurrent, capped broker-POST reservations, loss cooldown, negative-vehicle quarantine, and no auto-promotion",
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
    if abs(int(intent.units)) < MIN_PRODUCTION_LOT_UNITS:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_MIN_LOT_REQUIRED",
                f"SCOUT NAV-risk sizing resolved below the broker production floor {MIN_PRODUCTION_LOT_UNITS}u",
            )
        )
    elif _policy_contract_valid(policy):
        issues.extend(
            _predictive_scout_risk_plan_issues(
                intent,
                snapshot=snapshot,
                ledger_path=ledger_path,
                policy_path=policy_path
                or data_root.parent / "config" / "predictive_scout_policy.json",
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


def _predictive_scout_risk_plan_issues(
    intent: OrderIntent,
    *,
    snapshot: BrokerSnapshot | None,
    ledger_path: Path,
    policy_path: Path,
) -> list[dict[str, str]]:
    plan = predictive_scout_nav_risk_plan(
        intent,
        snapshot=snapshot,
        execution_ledger_db_path=ledger_path,
        policy_path=policy_path,
    )
    if plan.get("status") != "READY":
        return [
            _issue(
                "PREDICTIVE_SCOUT_NAV_RISK_PLAN_UNAVAILABLE",
                f"SCOUT NAV-risk plan failed closed with status={plan.get('status')}; refresh policy, ledger, and broker NAV before staging",
            )
        ]
    metadata = intent.metadata or {}
    current = {
        "tier": str(plan.get("tier") or "").upper(),
        "nav_jpy": _optional_positive_float(plan.get("nav_jpy")),
        "max_risk_pct_nav": _optional_positive_float(
            plan.get("max_risk_pct_nav")
        ),
        "max_loss_jpy": _optional_positive_float(plan.get("max_loss_jpy")),
    }
    actual = {
        "tier": str(metadata.get("predictive_scout_risk_tier") or "").upper(),
        "nav_jpy": _optional_positive_float(
            metadata.get("predictive_scout_nav_jpy_at_sizing")
        ),
        "max_risk_pct_nav": _optional_positive_float(
            metadata.get("predictive_scout_max_risk_pct_nav")
        ),
        "max_loss_jpy": _optional_positive_float(
            metadata.get("predictive_scout_max_loss_jpy")
        ),
    }
    # NAV is expected to move between intent generation and the gateway's
    # fresh broker snapshot. Requiring byte-for-byte equality would make every
    # otherwise valid SCOUT stale on the next tick. Authenticate the original
    # sizing equation, keep its digest immutable, and then enforce the smaller
    # of the original and current NAV caps against actual planned SL risk.
    declared_equation_valid = bool(
        actual["nav_jpy"] is not None
        and actual["max_risk_pct_nav"] is not None
        and actual["max_loss_jpy"] is not None
        and math.isclose(
            float(actual["max_loss_jpy"]),
            float(actual["nav_jpy"])
            * (float(actual["max_risk_pct_nav"]) / 100.0),
            rel_tol=1e-9,
            abs_tol=1e-4,
        )
    )
    current_contract_match = bool(
        actual["tier"] == current["tier"]
        and actual["max_risk_pct_nav"] is not None
        and current["max_risk_pct_nav"] is not None
        and math.isclose(
            float(actual["max_risk_pct_nav"]),
            float(current["max_risk_pct_nav"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    issues: list[dict[str, str]] = []
    if not declared_equation_valid or not current_contract_match:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_NAV_RISK_PLAN_MISMATCH",
                "SCOUT tier/risk-percent or its original NAV×percent=max-loss equation does not match the current exact-vehicle normalized forward plan",
            )
        )
    planned_risk = _optional_positive_float(
        metadata.get("predictive_scout_planned_initial_risk_jpy")
    )
    fresh_actual_risk = _predictive_scout_intent_initial_risk_jpy(
        intent,
        snapshot,
    )
    current_max_loss = current["max_loss_jpy"]
    declared_max_loss = actual["max_loss_jpy"]
    applicable_max_loss = (
        min(float(current_max_loss), float(declared_max_loss))
        if current_max_loss is not None and declared_max_loss is not None
        else None
    )
    if (
        planned_risk is None
        or applicable_max_loss is None
        or planned_risk > applicable_max_loss + 1e-4
    ):
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_PLANNED_RISK_INVALID",
                "SCOUT requires positive planned initial SL risk no greater than the current NAV-risk tier cap",
            )
        )
    if fresh_actual_risk is None:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_FRESH_ACTUAL_RISK_UNAVAILABLE",
                "SCOUT current units/entry/SL or quote-to-JPY conversion cannot be recomputed from the fresh broker snapshot",
            )
        )
    else:
        metadata["predictive_scout_fresh_actual_initial_risk_jpy"] = round(
            fresh_actual_risk,
            8,
        )
        if (
            applicable_max_loss is None
            or fresh_actual_risk > applicable_max_loss + 1e-4
        ):
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_FRESH_ACTUAL_RISK_CAP_EXCEEDED",
                    "SCOUT fresh units/entry/SL/conversion risk exceeds the smaller of the original and current NAV-tier caps",
                )
            )
        active_risk = predictive_scout_active_initial_risk_jpy(snapshot)
        current_nav = current["nav_jpy"]
        concurrent_pct = _optional_positive_float(
            plan.get("max_concurrent_risk_pct_nav")
        )
        if active_risk is None or current_nav is None or concurrent_pct is None:
            issues.append(
                _issue(
                    "PREDICTIVE_SCOUT_CONCURRENT_RISK_UNAVAILABLE",
                    "SCOUT cannot prove current-NAV aggregate initial risk for active broker SCOUT exposure",
                )
            )
        else:
            concurrent_cap = float(current_nav) * (concurrent_pct / 100.0)
            aggregate_risk = active_risk + fresh_actual_risk
            metadata["predictive_scout_active_initial_risk_jpy"] = round(
                active_risk,
                8,
            )
            metadata["predictive_scout_aggregate_initial_risk_jpy"] = round(
                aggregate_risk,
                8,
            )
            metadata["predictive_scout_concurrent_risk_cap_jpy"] = round(
                concurrent_cap,
                8,
            )
            if aggregate_risk > concurrent_cap + 1e-4:
                issues.append(
                    _issue(
                        "PREDICTIVE_SCOUT_CONCURRENT_NAV_RISK_CAP_EXCEEDED",
                        "active plus candidate SCOUT initial risk exceeds the current-NAV concurrent risk cap",
                    )
                )
    declared_digest = str(
        metadata.get("predictive_scout_sizing_digest") or ""
    ).strip()
    expected_digest = predictive_scout_sizing_digest(intent)
    if declared_digest != expected_digest:
        issues.append(
            _issue(
                "PREDICTIVE_SCOUT_SIZING_DIGEST_MISMATCH",
                "SCOUT units, geometry, or NAV-risk metadata changed after the pre-verifier sizing digest was written",
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


def predictive_scout_active_initial_risk_jpy(
    snapshot: BrokerSnapshot | None,
) -> float | None:
    """Return fresh initial-SL risk for every broker-active SCOUT.

    Missing geometry or conversion fails closed whenever a SCOUT object is
    active.  The 2% aggregate cap must be based on current NAV and current
    quote-to-JPY conversion, not on its entry-time declared percentage.
    """

    if snapshot is None:
        return None
    total = 0.0
    for position in snapshot.positions:
        if not predictive_scout_broker_raw_claimed(position.raw):
            continue
        risk = _predictive_scout_geometry_risk_jpy(
            pair=position.pair,
            units=position.units,
            entry=position.entry_price,
            stop_loss=position.stop_loss,
            snapshot=snapshot,
        )
        if risk is None:
            return None
        total += risk
    for order in snapshot.orders:
        if order.trade_id or not predictive_scout_broker_raw_claimed(order.raw):
            continue
        stop_loss = _raw_nested_positive_float(order.raw, "stopLossOnFill", "price")
        risk = _predictive_scout_geometry_risk_jpy(
            pair=str(order.pair or ""),
            units=order.units,
            entry=order.price,
            stop_loss=stop_loss,
            snapshot=snapshot,
        )
        if risk is None:
            return None
        total += risk
    return total


def _predictive_scout_intent_initial_risk_jpy(
    intent: OrderIntent,
    snapshot: BrokerSnapshot | None,
) -> float | None:
    if snapshot is None:
        return None
    return _predictive_scout_geometry_risk_jpy(
        pair=intent.pair,
        units=intent.units,
        entry=intent.entry,
        stop_loss=intent.sl,
        snapshot=snapshot,
    )


def _predictive_scout_geometry_risk_jpy(
    *,
    pair: str,
    units: Any,
    entry: Any,
    stop_loss: Any,
    snapshot: BrokerSnapshot,
) -> float | None:
    try:
        parsed_units = abs(int(units))
        parsed_entry = float(entry)
        parsed_stop = float(stop_loss)
    except (TypeError, ValueError):
        return None
    conversion = _predictive_scout_quote_to_jpy(pair, snapshot)
    if (
        parsed_units < MIN_PRODUCTION_LOT_UNITS
        or not math.isfinite(parsed_entry)
        or not math.isfinite(parsed_stop)
        or parsed_entry <= 0.0
        or parsed_stop <= 0.0
        or math.isclose(parsed_entry, parsed_stop, rel_tol=0.0, abs_tol=0.0)
        or conversion is None
    ):
        return None
    return abs(parsed_entry - parsed_stop) * parsed_units * conversion


def _predictive_scout_quote_to_jpy(
    pair: str,
    snapshot: BrokerSnapshot,
) -> float | None:
    parts = str(pair or "").upper().split("_", 1)
    if len(parts) != 2:
        return None
    quote_ccy = parts[1]
    if quote_ccy == "JPY":
        return 1.0
    home_conversion = _optional_positive_float(
        snapshot.home_conversions.get(quote_ccy)
    )
    if home_conversion is not None:
        return home_conversion
    conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is None:
        return None
    return _optional_positive_float(max(conversion_quote.bid, conversion_quote.ask))


def _raw_nested_positive_float(raw: Any, *keys: str) -> float | None:
    value = raw
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _optional_positive_float(value)


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


def predictive_scout_sizing_digest(intent: OrderIntent) -> str:
    """Authenticate one pre-verifier NAV-risk sizing decision.

    The stable vehicle id intentionally excludes units so changing NAV cannot
    reset cooldown or quarantine.  This digest does the opposite: it binds the
    exact signal packet to its units, geometry, and material sizing metadata so
    a later gateway cannot silently upsize an AI-verified intent.
    """

    metadata = intent.metadata or {}
    payload = {
        "vehicle_id": predictive_scout_vehicle_id(intent),
        "pair": intent.pair.upper(),
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": abs(int(intent.units)),
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
        "forecast_cycle_id": str(metadata.get("forecast_cycle_id") or ""),
        "forecast_direction": str(metadata.get("forecast_direction") or "").upper(),
        "forecast_confidence": _safe_float(metadata.get("forecast_confidence"), 0.0),
        "forecast_horizon_min": _nonnegative_int(metadata.get("forecast_horizon_min")),
        "source": str(metadata.get("predictive_scout_source") or "").upper(),
        "rule_name": str(metadata.get("predictive_scout_rule_name") or ""),
        "rule_digest": str(metadata.get("predictive_scout_rule_digest") or ""),
        "risk_tier": str(metadata.get("predictive_scout_risk_tier") or "").upper(),
        "nav_jpy": _optional_positive_float(
            metadata.get("predictive_scout_nav_jpy_at_sizing")
        ),
        "max_risk_pct_nav": _optional_positive_float(
            metadata.get("predictive_scout_max_risk_pct_nav")
        ),
        "max_loss_jpy": _optional_positive_float(
            metadata.get("predictive_scout_max_loss_jpy")
        ),
        "planned_initial_risk_jpy": _optional_positive_float(
            metadata.get("predictive_scout_planned_initial_risk_jpy")
        ),
    }
    return "psd-" + _stable_digest(payload)[:24]


def _predictive_scout_receipt_sizing_digest(receipt: dict[str, Any]) -> str:
    """Recompute the sizing digest from a durable gateway receipt."""

    payload = {
        "vehicle_id": str(receipt.get("predictive_scout_vehicle_id") or ""),
        "pair": str(receipt.get("pair") or "").upper(),
        "side": str(receipt.get("side") or "").upper(),
        "order_type": str(receipt.get("order_type") or "").upper(),
        "units": _strict_positive_int(receipt.get("units")),
        "entry": receipt.get("entry"),
        "tp": receipt.get("take_profit"),
        "sl": receipt.get("stop_loss"),
        "forecast_cycle_id": str(receipt.get("forecast_cycle_id") or ""),
        "forecast_direction": str(
            receipt.get("forecast_direction") or ""
        ).upper(),
        "forecast_confidence": _safe_float(
            receipt.get("forecast_confidence"),
            0.0,
        ),
        "forecast_horizon_min": _nonnegative_int(
            receipt.get("forecast_horizon_min")
        ),
        "source": str(receipt.get("predictive_scout_source") or "").upper(),
        "rule_name": str(receipt.get("predictive_scout_rule_name") or ""),
        "rule_digest": str(receipt.get("predictive_scout_rule_digest") or ""),
        "risk_tier": str(
            receipt.get("predictive_scout_risk_tier") or ""
        ).upper(),
        "nav_jpy": _optional_positive_float(
            receipt.get("predictive_scout_nav_jpy_at_sizing")
        ),
        "max_risk_pct_nav": _optional_positive_float(
            receipt.get("predictive_scout_max_risk_pct_nav")
        ),
        "max_loss_jpy": _optional_positive_float(
            receipt.get("predictive_scout_max_loss_jpy")
        ),
        "planned_initial_risk_jpy": _optional_positive_float(
            receipt.get("predictive_scout_planned_initial_risk_jpy")
        ),
    }
    return "psd-" + _stable_digest(payload)[:24]


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
        "units": abs(int(intent.units)),
        "planned_initial_risk_jpy": _optional_positive_float(
            metadata.get("predictive_scout_planned_initial_risk_jpy")
        ),
        "sizing_digest": predictive_scout_sizing_digest(intent),
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
            "one_sided_95_mean_lower_r_must_exceed": 0.0,
            "minimum_profit_factor_r": min_pf,
            "minimum_active_days": min_days,
            "minimum_positive_day_rate_r": min_positive_day_rate,
            "all_exit_reasons_and_financing_included": True,
            "all_resolved_exits_require_initial_risk_normalization": True,
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
    jpy_values = [_safe_float(item.get("net_jpy"), 0.0) for item in outcomes]
    normalized = [
        item
        for item in outcomes
        if _optional_finite_float(item.get("net_r")) is not None
        and _optional_finite_float(item.get("net_jpy_per_1000u")) is not None
    ]
    r_values = [float(item["net_r"]) for item in normalized]
    per_1000_values = [float(item["net_jpy_per_1000u"]) for item in normalized]
    resolved_count = len(jpy_values)
    normalized_count = len(r_values)
    raw_wins = sum(1 for value in jpy_values if value > 0.0)
    raw_losses = sum(1 for value in jpy_values if value < 0.0)
    wins = sum(1 for value in r_values if value > 0.0)
    losses = sum(1 for value in r_values if value < 0.0)
    gross_profit_jpy = sum(value for value in jpy_values if value > 0.0)
    gross_loss_abs_jpy = abs(sum(value for value in jpy_values if value < 0.0))
    gross_profit_r = sum(value for value in r_values if value > 0.0)
    gross_loss_abs_r = abs(sum(value for value in r_values if value < 0.0))
    mean_jpy = sum(jpy_values) / resolved_count if resolved_count else None
    lower_jpy = _one_sided_mean_lower_95(jpy_values)
    mean_r = sum(r_values) / normalized_count if normalized_count else None
    lower_r = _one_sided_mean_lower_95(r_values)
    profit_factor_r = (
        gross_profit_r / gross_loss_abs_r if gross_loss_abs_r > 0.0 else None
    )
    daily_net_r: dict[str, float] = {}
    exit_reasons: dict[str, int] = {}
    for item in normalized:
        day = str(item.get("resolved_day_utc") or "unknown")
        daily_net_r[day] = daily_net_r.get(day, 0.0) + float(item["net_r"])
    for item in outcomes:
        for reason in item.get("exit_reasons", []) or ["UNKNOWN"]:
            key = str(reason or "UNKNOWN")
            exit_reasons[key] = exit_reasons.get(key, 0) + 1
    active_days = len(daily_net_r)
    positive_days = sum(1 for value in daily_net_r.values() if value > 0.0)
    positive_day_rate = positive_days / active_days if active_days else 0.0
    filled_count = len(state.get("filled_trade_ids") or set())
    unresolved_count = max(0, filled_count - resolved_count)
    all_filled_resolved = filled_count > 0 and unresolved_count == 0
    complete_normalization = resolved_count > 0 and normalized_count == resolved_count
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
    profit_factor_pass = gross_loss_abs_r == 0.0 and gross_profit_r > 0.0
    if profit_factor_r is not None:
        profit_factor_pass = profit_factor_r >= min_profit_factor
    eligible = bool(
        all_filled_resolved
        and complete_signal_attribution
        and complete_normalization
        and normalized_count >= min_resolved
        and lower_r is not None
        and lower_r > 0.0
        and profit_factor_pass
        and active_days >= min_days
        and positive_day_rate >= min_positive_day_rate
    )
    risk_tier = _predictive_scout_risk_tier(
        normalized_count=normalized_count,
        net_r=sum(r_values),
        gross_profit_r=gross_profit_r,
        gross_loss_abs_r=gross_loss_abs_r,
        profit_factor_r=profit_factor_r,
        one_sided_95_mean_lower_r=lower_r,
        active_days=active_days,
        positive_day_rate_r=positive_day_rate,
        proven=eligible,
        all_filled_resolved=all_filled_resolved,
        complete_signal_attribution=complete_signal_attribution,
        complete_normalization=complete_normalization,
    )
    return {
        "predictive_scout_vehicle_id": vehicle_id,
        "predictive_scout_rule_digest": state.get("rule_digest"),
        "predictive_scout_rule_name": state.get("rule_name"),
        "pair": state.get("pair"),
        "side": state.get("side"),
        "risk_tier": risk_tier,
        "sent_count": int(state.get("sent_count") or 0),
        "filled_count": filled_count,
        "resolved_count": resolved_count,
        "normalized_resolved_count": normalized_count,
        "normalization_missing_count": max(0, resolved_count - normalized_count),
        "complete_normalization": complete_normalization,
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
        "raw_wins": raw_wins,
        "raw_losses": raw_losses,
        "net_jpy": round(sum(jpy_values), 4),
        "mean_net_jpy": round(mean_jpy, 4) if mean_jpy is not None else None,
        "one_sided_95_mean_lower_jpy": (
            round(lower_jpy, 4) if lower_jpy is not None else None
        ),
        "gross_profit_jpy": round(gross_profit_jpy, 4),
        "gross_loss_abs_jpy": round(gross_loss_abs_jpy, 4),
        "net_r": round(sum(r_values), 6),
        "mean_net_r": round(mean_r, 6) if mean_r is not None else None,
        "one_sided_95_mean_lower_r": (
            round(lower_r, 6) if lower_r is not None else None
        ),
        "gross_profit_r": round(gross_profit_r, 6),
        "gross_loss_abs_r": round(gross_loss_abs_r, 6),
        "profit_factor_r": (
            round(profit_factor_r, 6) if profit_factor_r is not None else None
        ),
        # Backward-compatible key now deliberately points at the normalized-R
        # PF so variable units cannot make a large JPY win dominate the proof.
        "profit_factor": (
            round(profit_factor_r, 6) if profit_factor_r is not None else None
        ),
        "net_jpy_per_1000u": round(sum(per_1000_values), 4),
        "mean_net_jpy_per_1000u": (
            round(sum(per_1000_values) / len(per_1000_values), 4)
            if per_1000_values
            else None
        ),
        "active_days": active_days,
        "positive_days": positive_days,
        "positive_day_rate_r": round(positive_day_rate, 6),
        "positive_day_rate": round(positive_day_rate, 6),
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "quarantined_negative_vehicle": (
            raw_losses >= PREDICTIVE_SCOUT_MAX_NEGATIVE_VEHICLE_LOSSES
            and sum(jpy_values) < 0.0
        ),
        "statistically_eligible_for_operator_review": eligible,
        "automatic_promotion_allowed": False,
    }


def _predictive_scout_risk_tier(
    *,
    normalized_count: int,
    net_r: float,
    gross_profit_r: float,
    gross_loss_abs_r: float,
    profit_factor_r: float | None,
    one_sided_95_mean_lower_r: float | None,
    active_days: int,
    positive_day_rate_r: float,
    proven: bool,
    all_filled_resolved: bool,
    complete_signal_attribution: bool,
    complete_normalization: bool,
) -> str:
    if proven:
        return "PROVEN"
    if not (
        all_filled_resolved
        and complete_signal_attribution
        and complete_normalization
    ):
        return "DISCOVERY"
    pf_1_2 = _profit_factor_meets(
        profit_factor_r,
        gross_profit=gross_profit_r,
        gross_loss_abs=gross_loss_abs_r,
        minimum=1.2,
    )
    established = bool(
        normalized_count >= 10
        and pf_1_2
        and positive_day_rate_r >= PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE
    )
    if (
        normalized_count >= 20
        and one_sided_95_mean_lower_r is not None
        and one_sided_95_mean_lower_r >= 0.0
        and active_days >= 5
    ):
        return "STRONG"
    if established:
        return "ESTABLISHED"
    if (
        normalized_count >= 5
        and net_r > 0.0
        and _profit_factor_meets(
            profit_factor_r,
            gross_profit=gross_profit_r,
            gross_loss_abs=gross_loss_abs_r,
            minimum=1.0,
        )
    ):
        return "EMERGING"
    return "DISCOVERY"


def _profit_factor_meets(
    value: float | None,
    *,
    gross_profit: float,
    gross_loss_abs: float,
    minimum: float,
) -> bool:
    if value is not None:
        return value >= minimum
    return gross_loss_abs == 0.0 and gross_profit > 0.0


def predictive_scout_nav_risk_plan(
    intent: OrderIntent,
    *,
    snapshot: BrokerSnapshot | None,
    execution_ledger_db_path: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Return the fail-closed NAV-loss budget for one exact SCOUT vehicle.

    This function does not calculate units and never grants live permission.
    Its output is the maximum JPY loss the intent generator may convert to
    units from the current stop distance.  Only normalized forward outcomes
    for the stable vehicle id can move the tier above DISCOVERY.
    """

    result: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "tier": None,
        "nav_jpy": None,
        "max_risk_pct_nav": 0.0,
        "max_loss_jpy": 0.0,
        "resolved_count": 0,
        "raw_resolved_count": 0,
        "net_r": 0.0,
        "profit_factor_r": None,
        "one_sided_95_mean_lower_r": None,
        "active_days": 0,
        "positive_day_rate_r": 0.0,
        "complete_signal_attribution": False,
        "complete_normalization": False,
        "max_concurrent_risk_pct_nav": 0.0,
        "vehicle_id": predictive_scout_vehicle_id(intent),
    }
    policy = predictive_scout_policy(policy_path)
    if not _policy_contract_valid(policy):
        result["status"] = "POLICY_INVALID"
        return result
    if policy.get("enabled") is not True:
        result["status"] = "POLICY_DISABLED"
        return result
    account = snapshot.account if snapshot is not None else None
    nav_jpy = _optional_positive_float(getattr(account, "nav_jpy", None))
    if nav_jpy is None:
        result["status"] = "NAV_UNAVAILABLE"
        return result
    result["nav_jpy"] = round(nav_jpy, 4)
    extracted = _predictive_scout_forward_outcomes(execution_ledger_db_path)
    if extracted is None:
        result["status"] = "LEDGER_UNAVAILABLE"
        return result
    vehicle_id = str(result["vehicle_id"])
    state = extracted.get(vehicle_id) or {
        "sent_count": 0,
        "filled_trade_ids": set(),
        "signal_broker_refs": {},
        "filled_signal_trade_ids": {},
        "outcomes": [],
        "pair": intent.pair,
        "side": intent.side.value,
    }
    row = _predictive_scout_vehicle_proof_row(
        vehicle_id,
        state,
        min_resolved=max(
            PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS,
            _positive_int(
                policy.get("promotion_min_resolved_exits"),
                PREDICTIVE_SCOUT_PROMOTION_MIN_RESOLVED_EXITS,
            ),
        ),
        min_days=max(
            PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS,
            _positive_int(
                policy.get("promotion_min_active_days"),
                PREDICTIVE_SCOUT_MIN_ACTIVE_DAYS,
            ),
        ),
        min_profit_factor=max(
            PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR,
            _safe_float(
                policy.get("promotion_min_profit_factor"),
                PREDICTIVE_SCOUT_MIN_PROFIT_FACTOR,
            ),
        ),
        min_positive_day_rate=max(
            PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE,
            _safe_float(
                policy.get("promotion_min_positive_day_rate"),
                PREDICTIVE_SCOUT_MIN_POSITIVE_DAY_RATE,
            ),
        ),
    )
    tier = str(row.get("risk_tier") or "DISCOVERY").upper()
    tier_pct = _policy_risk_tiers(policy).get(tier)
    max_per_trade_pct = _optional_positive_float(
        policy.get("max_per_trade_risk_pct_nav")
    )
    if tier_pct is None or max_per_trade_pct is None:
        result["status"] = "POLICY_INVALID"
        return result
    risk_pct = min(tier_pct, max_per_trade_pct)
    result.update(
        {
            "status": "READY",
            "tier": tier,
            "max_risk_pct_nav": round(risk_pct, 6),
            "max_loss_jpy": round(nav_jpy * (risk_pct / 100.0), 4),
            "resolved_count": int(row.get("normalized_resolved_count") or 0),
            "raw_resolved_count": int(row.get("resolved_count") or 0),
            "net_r": _safe_float(row.get("net_r"), 0.0),
            "profit_factor_r": row.get("profit_factor_r"),
            "one_sided_95_mean_lower_r": row.get(
                "one_sided_95_mean_lower_r"
            ),
            "active_days": int(row.get("active_days") or 0),
            "positive_day_rate_r": _safe_float(
                row.get("positive_day_rate_r"), 0.0
            ),
            "complete_signal_attribution": bool(
                row.get("complete_signal_attribution")
            ),
            "complete_normalization": bool(row.get("complete_normalization")),
            "max_concurrent_risk_pct_nav": round(
                _safe_float(policy.get("max_concurrent_risk_pct_nav"), 0.0),
                6,
            ),
        }
    )
    return result


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
            units_expr = "units" if "units" in event_columns else "NULL AS units"
            gateway_rows = con.execute(
                f"""
                SELECT event_type, order_id, trade_id, {client_order_expr}, raw_json
                FROM execution_events
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'GATEWAY_ORDER_STAGED')
                """
            ).fetchall()
            fill_rows = con.execute(
                f"""
                SELECT order_id, trade_id, {client_order_expr}, {units_expr}, price, raw_json
                FROM execution_events
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
    order_to_normalizers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    trade_to_normalizers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    client_to_normalizers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    trade_normalization: dict[str, dict[str, dict[str, Any]]] = {}
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
            normalizer = _predictive_scout_receipt_normalizer(receipt)
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
                if normalizer is not None:
                    order_to_normalizers.setdefault(str(order_id), []).append(
                        (vehicle_id, normalizer)
                    )
            if trade_id:
                trade_text = str(trade_id)
                trade_to_vehicles.setdefault(trade_text, set()).add(vehicle_id)
                if signal_ref is not None:
                    trade_to_signals.setdefault(trade_text, set()).add(signal_ref)
                if normalizer is not None:
                    trade_to_normalizers.setdefault(trade_text, []).append(
                        (vehicle_id, normalizer)
                    )
                state["filled_trade_ids"].add(trade_text)
            if client_order_id:
                client_to_vehicles.setdefault(str(client_order_id), set()).add(vehicle_id)
                if signal_ref is not None:
                    client_to_signals.setdefault(str(client_order_id), set()).add(signal_ref)
                if normalizer is not None:
                    client_to_normalizers.setdefault(str(client_order_id), []).append(
                        (vehicle_id, normalizer)
                    )
    for order_id, trade_id, client_order_id, fill_units_raw, fill_price_raw, fill_raw in fill_rows:
        vehicle_ids: set[str] = set()
        signal_refs: set[tuple[str, str]] = set()
        normalizer_refs: list[tuple[str, dict[str, Any]]] = []
        if order_id:
            vehicle_ids.update(order_to_vehicles.get(str(order_id), set()))
            signal_refs.update(order_to_signals.get(str(order_id), set()))
            normalizer_refs.extend(order_to_normalizers.get(str(order_id), []))
        if trade_id:
            vehicle_ids.update(trade_to_vehicles.get(str(trade_id), set()))
            signal_refs.update(trade_to_signals.get(str(trade_id), set()))
            normalizer_refs.extend(trade_to_normalizers.get(str(trade_id), []))
        if client_order_id:
            vehicle_ids.update(client_to_vehicles.get(str(client_order_id), set()))
            signal_refs.update(client_to_signals.get(str(client_order_id), set()))
            normalizer_refs.extend(client_to_normalizers.get(str(client_order_id), []))
        fill_units = _predictive_scout_fill_units(fill_units_raw, fill_raw)
        fill_price = _predictive_scout_fill_price(fill_price_raw, fill_raw)
        unique_normalizers: dict[tuple[str, str], dict[str, Any]] = {}
        for vehicle_id, normalizer in normalizer_refs:
            key = (vehicle_id, str(normalizer.get("sizing_digest") or ""))
            previous = unique_normalizers.get(key)
            if previous is None:
                unique_normalizers[key] = normalizer
            elif previous != normalizer:
                unique_normalizers[key] = {"invalid": True}
        for vehicle_id in vehicle_ids:
            trade_text = str(trade_id)
            trade_to_vehicles.setdefault(trade_text, set()).add(vehicle_id)
            vehicles[vehicle_id]["filled_trade_ids"].add(trade_text)
            matching = [
                normalizer
                for (normalizer_vehicle, _digest), normalizer in unique_normalizers.items()
                if normalizer_vehicle == vehicle_id
            ]
            # Conflicting sizing receipts or a fill with unknown units cannot
            # enter normalized proof.  The raw P/L remains attached below for
            # cooldown/quarantine accounting.
            if (
                fill_units is not None
                and fill_price is not None
                and len(matching) == 1
                and matching[0].get("invalid") is not True
            ):
                normalizer = matching[0]
                loss_conversion = _predictive_scout_fill_loss_conversion(
                    str(normalizer.get("pair") or ""),
                    fill_raw,
                )
                stop_loss = _optional_positive_float(normalizer.get("stop_loss"))
                actual_initial_risk = (
                    abs(fill_price - stop_loss) * fill_units * loss_conversion
                    if stop_loss is not None and loss_conversion is not None
                    else None
                )
                record = trade_normalization.setdefault(trade_text, {}).setdefault(
                    vehicle_id,
                    {
                        "filled_units": 0,
                        "initial_risk_jpy": 0.0,
                        "declared_units": int(normalizer["declared_units"]),
                        "sizing_digest": str(normalizer["sizing_digest"]),
                    },
                )
                if (
                    int(record["declared_units"]) == int(normalizer["declared_units"])
                    and str(record["sizing_digest"])
                    == str(normalizer["sizing_digest"])
                    and actual_initial_risk is not None
                    and actual_initial_risk > 0.0
                ):
                    record["filled_units"] = int(record["filled_units"]) + fill_units
                    record["initial_risk_jpy"] = float(
                        record["initial_risk_jpy"]
                    ) + actual_initial_risk
                else:
                    record["invalid"] = True
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
            normalized_outcome = dict(outcome)
            normalization = trade_normalization.get(trade_id, {}).get(vehicle_id)
            if (
                isinstance(normalization, dict)
                and normalization.get("invalid") is not True
                and int(normalization.get("filled_units") or 0) > 0
                and int(normalization.get("filled_units") or 0)
                <= int(normalization.get("declared_units") or 0)
                and float(normalization.get("initial_risk_jpy") or 0.0) > 0.0
            ):
                filled_units = int(normalization["filled_units"])
                initial_risk_jpy = float(normalization["initial_risk_jpy"])
                net_jpy = float(normalized_outcome["net_jpy"])
                normalized_outcome.update(
                    {
                        "normalization_status": "NORMALIZED",
                        "filled_units": filled_units,
                        "initial_risk_jpy": round(initial_risk_jpy, 8),
                        "net_r": net_jpy / initial_risk_jpy,
                        "net_jpy_per_1000u": (
                            net_jpy * PREDICTIVE_SCOUT_NORMALIZATION_UNITS
                            / filled_units
                        ),
                        "sizing_digest": normalization.get("sizing_digest"),
                    }
                )
            else:
                normalized_outcome["normalization_status"] = "MISSING"
                normalized_outcome["net_r"] = None
                normalized_outcome["net_jpy_per_1000u"] = None
            vehicles[vehicle_id]["outcomes"].append(normalized_outcome)
    return vehicles


def _predictive_scout_receipt_normalizer(
    receipt: dict[str, Any],
) -> dict[str, Any] | None:
    declared_units = _strict_positive_int(receipt.get("units"))
    if declared_units is None or declared_units < MIN_PRODUCTION_LOT_UNITS:
        return None
    risk_plan = (
        receipt.get("predictive_scout_risk_plan")
        if isinstance(receipt.get("predictive_scout_risk_plan"), dict)
        else {}
    )
    planned_risk = next(
        (
            parsed
            for parsed in (
                _optional_positive_float(
                    receipt.get("predictive_scout_planned_initial_risk_jpy")
                ),
                _optional_positive_float(receipt.get("planned_initial_risk_jpy")),
                _optional_positive_float(risk_plan.get("planned_initial_risk_jpy")),
            )
            if parsed is not None
        ),
        None,
    )
    sizing_digest = str(
        receipt.get("predictive_scout_sizing_digest")
        or risk_plan.get("sizing_digest")
        or ""
    ).strip()
    declared_signal_id = str(
        receipt.get("predictive_scout_signal_id") or ""
    ).strip()
    expected_signal_id = _predictive_scout_expected_receipt_signal_id(receipt)
    declared_experiment_id = str(
        receipt.get("predictive_scout_experiment_id") or ""
    ).strip()
    expected_experiment_id = _predictive_scout_expected_receipt_experiment_id(
        receipt,
        sizing_digest=sizing_digest,
    )
    if (
        planned_risk is None
        or sizing_digest != _predictive_scout_receipt_sizing_digest(receipt)
        or not expected_signal_id
        or declared_signal_id != expected_signal_id
        or not expected_experiment_id
        or declared_experiment_id != expected_experiment_id
    ):
        return None
    pair = str(receipt.get("pair") or "").upper()
    stop_loss = _optional_positive_float(receipt.get("stop_loss"))
    if not pair or stop_loss is None:
        return None
    return {
        "declared_units": declared_units,
        "planned_initial_risk_jpy": planned_risk,
        "sizing_digest": sizing_digest,
        "pair": pair,
        "stop_loss": stop_loss,
    }


def _predictive_scout_fill_units(value: Any, raw: Any) -> int | None:
    parsed = _strict_nonzero_int_abs(value)
    if parsed is not None:
        return parsed
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    candidates = [payload.get("units")]
    for key in ("orderFillTransaction", "tradeOpened", "tradeReduced"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("units"))
    for candidate in candidates:
        parsed = _strict_nonzero_int_abs(candidate)
        if parsed is not None:
            return parsed
    return None


def _predictive_scout_fill_price(value: Any, raw: Any) -> float | None:
    parsed = _optional_positive_float(value)
    if parsed is not None:
        return parsed
    payload = _json_object(raw)
    if payload is None:
        return None
    candidates = [payload.get("price"), payload.get("fullVWAP")]
    opened = payload.get("tradeOpened")
    if isinstance(opened, dict):
        candidates.append(opened.get("price"))
    for candidate in candidates:
        parsed = _optional_positive_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _predictive_scout_fill_loss_conversion(pair: str, raw: Any) -> float | None:
    if str(pair).upper().endswith("_JPY"):
        return 1.0
    payload = _json_object(raw)
    if payload is None:
        return None
    nested = payload.get("homeConversionFactors")
    nested_loss_quote = (
        nested.get("lossQuoteHome") if isinstance(nested, dict) else None
    )
    candidates = [
        payload.get("lossQuoteHomeConversionFactor"),
        nested_loss_quote.get("factor")
        if isinstance(nested_loss_quote, dict)
        else None,
    ]
    for candidate in candidates:
        parsed = _optional_positive_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _json_object(raw: Any) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _strict_positive_int(value: Any) -> int | None:
    try:
        parsed_float = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(parsed_float)
        or parsed_float <= 0.0
        or not parsed_float.is_integer()
    ):
        return None
    return int(parsed_float)


def _strict_nonzero_int_abs(value: Any) -> int | None:
    """Accept signed broker units only when they are exact non-zero integers."""

    try:
        parsed_float = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(parsed_float)
        or parsed_float == 0.0
        or not parsed_float.is_integer()
    ):
        return None
    return abs(int(parsed_float))


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
                f"- NAV-risk tier: `{item.get('risk_tier')}`",
                f"- Sent / filled / resolved / unresolved: `{item.get('sent_count')} / {item.get('filled_count')} / {item.get('resolved_count')} / {item.get('unresolved_filled_count')}`",
                f"- Normalized / missing: `{item.get('normalized_resolved_count')} / {item.get('normalization_missing_count')}`",
                f"- Raw net JPY / normalized net R / per-1000u: `{item.get('net_jpy')} / {item.get('net_r')} / {item.get('net_jpy_per_1000u')}`",
                f"- Mean R / one-sided 95% lower R: `{item.get('mean_net_r')} / {item.get('one_sided_95_mean_lower_r')}`",
                f"- PF(R) / positive-day rate(R): `{item.get('profit_factor_r')} / {item.get('positive_day_rate_r')}`",
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
        max_concurrent = int(policy.get("max_concurrent"))
        max_per_trade_risk_pct_nav = float(
            policy.get("max_per_trade_risk_pct_nav")
        )
        max_concurrent_risk_pct_nav = float(
            policy.get("max_concurrent_risk_pct_nav")
        )
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
    risk_tiers = _policy_risk_tiers(policy)
    return bool(
        schema_version == PREDICTIVE_SCOUT_POLICY_SCHEMA_VERSION
        and str(policy.get("mode") or "").upper() == "FORWARD_EVIDENCE_ONLY"
        and "units" not in policy
        and risk_tiers == PREDICTIVE_SCOUT_RISK_TIER_PCT_NAV
        and math.isclose(
            max_per_trade_risk_pct_nav,
            PREDICTIVE_SCOUT_MAX_PER_TRADE_RISK_PCT_NAV,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            max_concurrent_risk_pct_nav,
            PREDICTIVE_SCOUT_MAX_CONCURRENT_RISK_PCT_NAV,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
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


def _policy_risk_tiers(policy: dict[str, Any]) -> dict[str, float]:
    raw = policy.get("risk_tiers")
    if not isinstance(raw, dict):
        return {}
    tiers: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            value = value.get("max_risk_pct_nav")
        parsed = _optional_positive_float(value)
        if parsed is None:
            return {}
        tiers[str(name).upper()] = parsed
    return tiers


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
        _predictive_scout_receipt_signal_id(
            receipt,
            vehicle_id=str(receipt.get("predictive_scout_vehicle_id") or ""),
        )
        for receipt in _predictive_scout_receipts(payload)
    }
    signal_ids.discard("")
    if signal_ids:
        return {f"signal:{signal_id}" for signal_id in signal_ids}
    experiment_ids = set()
    for receipt in _predictive_scout_receipts(payload):
        sizing_digest = str(
            receipt.get("predictive_scout_sizing_digest") or ""
        ).strip()
        expected = _predictive_scout_expected_receipt_experiment_id(
            receipt,
            sizing_digest=sizing_digest,
        )
        declared = str(
            receipt.get("predictive_scout_experiment_id") or ""
        ).strip()
        if expected and declared == expected:
            experiment_ids.add(expected)
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
    expected = _predictive_scout_expected_receipt_signal_id(
        receipt,
        vehicle_id=vehicle_id,
    )
    return expected if expected and declared == expected else ""


def _predictive_scout_expected_receipt_signal_id(
    receipt: dict[str, Any],
    *,
    vehicle_id: str | None = None,
) -> str:
    resolved_vehicle_id = str(
        vehicle_id or receipt.get("predictive_scout_vehicle_id") or ""
    ).strip()
    forecast_cycle_id = str(receipt.get("forecast_cycle_id") or "").strip()
    if not resolved_vehicle_id or not forecast_cycle_id:
        return ""
    return "pss-" + _stable_digest(
        {
            "vehicle_id": resolved_vehicle_id,
            "forecast_cycle_id": forecast_cycle_id,
        }
    )[:24]


def _predictive_scout_expected_receipt_experiment_id(
    receipt: dict[str, Any],
    *,
    sizing_digest: str,
) -> str:
    vehicle_id = str(
        receipt.get("predictive_scout_vehicle_id") or ""
    ).strip()
    forecast_cycle_id = str(receipt.get("forecast_cycle_id") or "").strip()
    generated_at = str(
        receipt.get("predictive_scout_generated_at_utc") or ""
    ).strip()
    units = _strict_positive_int(receipt.get("units"))
    planned_risk = _optional_positive_float(
        receipt.get("predictive_scout_planned_initial_risk_jpy")
    )
    if (
        not vehicle_id
        or not forecast_cycle_id
        or not generated_at
        or units is None
        or planned_risk is None
        or not sizing_digest
    ):
        return ""
    payload = {
        "vehicle_id": vehicle_id,
        "forecast_cycle_id": forecast_cycle_id,
        "generated_at_utc": generated_at,
        "entry": receipt.get("entry"),
        "tp": receipt.get("take_profit"),
        "sl": receipt.get("stop_loss"),
        "units": units,
        "planned_initial_risk_jpy": planned_risk,
        "sizing_digest": sizing_digest,
    }
    return "psx-" + _stable_digest(payload)[:24]


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


def _optional_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _optional_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "severity": "BLOCK"}
