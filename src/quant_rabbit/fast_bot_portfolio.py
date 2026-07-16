"""Shadow-only, horizon-aware portfolio planning for fast-bot signals.

The planner deliberately does not choose a single "best" signal.  Every
sealed, eligible path is retained, and deterministic ranking is used only
when an explicit concurrent-risk constraint binds.  The output is diagnostic
evidence; it cannot mutate a broker or grant live permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


PORTFOLIO_CONTRACT = "QR_FAST_BOT_HORIZON_PORTFOLIO_SHADOW_V1"
MAX_SOURCE_SIGNALS = 512
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def build_fast_bot_portfolio(
    sealed_signals: Sequence[Mapping[str, Any]],
    *,
    hedging_enabled: bool,
    constraints: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build a bounded multi-signal shadow portfolio.

    Supported constraints are ``margin_budget_jpy``, ``max_currency_gross``
    (one number or a currency-to-number mapping), and
    ``max_concurrent_signals_per_horizon`` (one integer or a horizon-to-integer
    mapping).  Concurrent-count limits apply inside one horizon lane.  Account
    margin and currency-gross limits aggregate every lane whose half-open
    holding windows actually overlap.
    """

    if isinstance(sealed_signals, (str, bytes)) or len(sealed_signals) > MAX_SOURCE_SIGNALS:
        raise ValueError(f"sealed_signals must contain at most {MAX_SOURCE_SIGNALS} mappings")
    normalized_constraints = _normalize_constraints(constraints)
    generated_at = _aware_utc(now_utc or datetime.now(timezone.utc))

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_seals: set[str] = set()
    for source_index, raw in enumerate(sealed_signals):
        candidate, reasons = _normalize_signal(raw, source_index=source_index)
        if candidate is not None and candidate["source_signal_sha256"] in seen_seals:
            reasons.append("DUPLICATE_SIGNAL_SEAL")
        if reasons or candidate is None:
            rejected.append(
                {
                    "source_index": source_index,
                    "signal_id": _bounded_text(raw.get("signal_id")) if isinstance(raw, Mapping) else None,
                    "signal_key": candidate.get("signal_key") if candidate else None,
                    "selection_status": "SHADOW_REJECTED_INVALID",
                    "reasons": sorted(set(reasons or ["SIGNAL_MAPPING_INVALID"])),
                    "shadow_record_retained": True,
                }
            )
            continue
        seen_seals.add(candidate["source_signal_sha256"])
        eligible.append(candidate)

    ranked = sorted(eligible, key=_rank_key)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        trial = [*selected, candidate]
        evaluation = _evaluate_candidate_constraints(
            trial,
            candidate=candidate,
            constraints=normalized_constraints,
        )
        if evaluation["binding_reasons"]:
            rejected.append(
                {
                    **_public_signal(candidate),
                    "deterministic_rank": rank,
                    "selection_status": "SHADOW_REJECTED_CONSTRAINT",
                    "reasons": evaluation["binding_reasons"],
                    "constraint_evidence": evaluation["evidence"],
                    "shadow_record_retained": True,
                }
            )
            actual = "SHADOW_REJECTED_CONSTRAINT"
        else:
            selected.append(
                {
                    **candidate,
                    "deterministic_rank": rank,
                    "selection_status": "SELECTED_SHADOW",
                    "broker_projection": "NO_OPPOSING_PAIR_SIGNAL",
                }
            )
            actual = "SELECTED_SHADOW"
        audit.append(
            {
                "signal_id": candidate["signal_id"],
                "signal_key": candidate["signal_key"],
                "deterministic_rank": rank,
                "unconstrained_path_status": "SELECTED_SHADOW",
                "constrained_path_status": actual,
                "would_select_if_constraints_unbound": True,
                "selection_changes_if_constraints_unbound": actual != "SELECTED_SHADOW",
                "binding_reasons": evaluation["binding_reasons"],
                "same_pair_or_shared_currency_was_not_a_blanket_rejection": True,
            }
        )

    coexistence, opposing_signal_ids = _opposite_side_coexistence(
        selected,
        hedging_enabled=bool(hedging_enabled),
    )
    for row in selected:
        if row["signal_id"] not in opposing_signal_ids:
            continue
        row["broker_projection"] = (
            "BROKER_HEDGE_CAPABLE_IF_SEPARATELY_PROMOTED"
            if hedging_enabled
            else "VIRTUAL_ONLY_NETTING_ACCOUNT"
        )

    public_selected = [_public_signal(item) for item in selected]
    signals_by_key: dict[str, list[dict[str, Any]]] = {}
    for item in [*public_selected, *rejected]:
        key = item.get("signal_key")
        if isinstance(key, str):
            signals_by_key.setdefault(key, []).append(item)

    body = {
        "contract": PORTFOLIO_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "status": "PLANNED" if eligible else "NO_ELIGIBLE_SIGNAL",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
        "broker_mutation_allowed": False,
        "selection_policy": "ALL_ELIGIBLE_UNLESS_EXPLICIT_OVERLAP_CONSTRAINT_BINDS",
        "top_one_selection": False,
        "risk_scope": {
            "max_concurrent_signals": "SAME_HORIZON_AND_OVERLAPPING_HALF_OPEN_HOLDING_WINDOW",
            "account_margin": "ALL_HORIZONS_WITH_SIMULTANEOUS_HOLDING_WINDOWS",
            "currency_gross_exposure": "ALL_HORIZONS_WITH_SIMULTANEOUS_HOLDING_WINDOWS",
            "correlation_statistics": "HORIZON_SPECIFIC_DIAGNOSTIC_ONLY",
        },
        "source_signal_count": len(sealed_signals),
        "eligible_signal_count": len(eligible),
        "selected_signal_count": len(public_selected),
        "shadow_rejected_count": len(rejected),
        "constraints": normalized_constraints,
        "broker_account_projection": {
            "hedging_enabled": bool(hedging_enabled),
            "opposite_side_mode": (
                "BROKER_HEDGE_CAPABLE_IF_SEPARATELY_PROMOTED"
                if hedging_enabled
                else "VIRTUAL_ONLY_NETTING_ACCOUNT"
            ),
            "broker_independent_shadow_paths_retained": True,
        },
        "source_schema_compatibility": {
            "current_fast_bot_wiring_status": "BLOCKED_PENDING_STANDARDIZED_SHADOW_SIZING_AND_MARGIN",
            "required_signal_fields": [
                "horizon_lane",
                "notional_units",
                "estimated_margin_jpy",
                "entry_or_quote_per_base",
            ],
            "missing_sizing_policy": "SHADOW_REJECT_WITHOUT_FABRICATION",
        },
        "selected": public_selected,
        "shadow_rejected": rejected,
        "eligible_signals_by_key": signals_by_key,
        "counterfactual_path_audit": audit,
        "currency_legs_and_exposure": _currency_exposure(public_selected),
        "concurrent_risk_slices": {
            "global_account_exposure": _global_risk_slices(public_selected),
            "by_horizon": _risk_slices(public_selected),
        },
        "opposite_side_coexistence": coexistence,
        "promotion_contract": {
            "status": "NOT_IMPLEMENTED",
            "live_permission": False,
            "broker_mutation_allowed": False,
            "requires_separate_forward_proof_and_gateway_contract": True,
        },
    }
    return {**body, "contract_sha256": _canonical_sha(body)}


def _normalize_signal(
    raw: Mapping[str, Any],
    *,
    source_index: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, Mapping):
        return None, ["SIGNAL_MAPPING_INVALID"]
    reasons: list[str] = []
    seal = raw.get("signal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        reasons.append("SIGNAL_SEAL_MISSING")
    else:
        body = {key: value for key, value in raw.items() if key != "signal_sha256"}
        if _canonical_sha(body) != seal:
            reasons.append("SIGNAL_SEAL_INVALID")
    if raw.get("shadow_only") is not True:
        reasons.append("SOURCE_NOT_SHADOW_ONLY")
    if raw.get("live_permission") is not False:
        reasons.append("SOURCE_LIVE_PERMISSION_NOT_FALSE")
    if raw.get("broker_mutation_allowed") is not False:
        reasons.append("SOURCE_BROKER_MUTATION_NOT_FALSE")

    signal_id = _bounded_text(raw.get("signal_id"))
    pair = str(raw.get("pair") or "").upper()
    side = str(raw.get("side") or "").upper()
    method = str(raw.get("method") or "").upper()
    horizon = str(raw.get("horizon_lane") or "").upper()
    if not signal_id:
        reasons.append("SIGNAL_ID_MISSING")
    if not _PAIR_RE.fullmatch(pair) or pair[:3] == pair[4:]:
        reasons.append("PAIR_INVALID")
    if side not in {"LONG", "SHORT"}:
        reasons.append("SIDE_INVALID")
    if not _TOKEN_RE.fullmatch(method):
        reasons.append("METHOD_INVALID")
    if not _TOKEN_RE.fullmatch(horizon):
        reasons.append("HORIZON_LANE_INVALID")

    window = raw.get("holding_window") if isinstance(raw.get("holding_window"), Mapping) else {}
    start = _parse_utc(window.get("start_utc") or raw.get("holding_start_utc") or raw.get("generated_at_utc"))
    end = _parse_utc(window.get("end_utc") or raw.get("holding_end_utc"))
    if start is not None and end is None:
        ttl = _nonnegative_float(raw.get("entry_ttl_seconds")) or 0.0
        hold = _positive_float(raw.get("max_hold_seconds"))
        if hold is not None:
            end = start + timedelta(seconds=ttl + hold)
    if start is None or end is None or end <= start:
        reasons.append("HOLDING_WINDOW_INVALID")

    margin = _nonnegative_float(raw.get("estimated_margin_jpy"))
    units = _positive_float(raw.get("notional_units") or raw.get("shadow_notional_units"))
    entry = _positive_float(raw.get("entry"))
    explicit_quote_per_base = _positive_float(raw.get("quote_per_base"))
    if entry is not None and explicit_quote_per_base is not None and not math.isclose(
        entry,
        explicit_quote_per_base,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        reasons.append("ENTRY_QUOTE_PER_BASE_MISMATCH")
    quote_per_base = explicit_quote_per_base if explicit_quote_per_base is not None else entry
    priority = _finite_float(raw.get("portfolio_priority"))
    if priority is None:
        priority = _finite_float(raw.get("regime_score"))
    if margin is None:
        reasons.append("ESTIMATED_MARGIN_JPY_INVALID")
    if units is None:
        reasons.append("NOTIONAL_UNITS_INVALID")
    if quote_per_base is None:
        reasons.append("ENTRY_OR_QUOTE_PER_BASE_INVALID")
    if priority is None:
        reasons.append("PORTFOLIO_PRIORITY_INVALID")
    if reasons:
        return None, reasons

    base, quote = pair.split("_")
    direction = 1.0 if side == "LONG" else -1.0
    assert start is not None and end is not None
    assert margin is not None and units is not None and priority is not None
    assert quote_per_base is not None
    assert isinstance(seal, str)
    return (
        {
            "source_index": source_index,
            "signal_id": signal_id,
            "signal_key": f"{pair}:{side}:{method}:{horizon}",
            "pair": pair,
            "side": side,
            "method": method,
            "horizon_lane": horizon,
            "holding_window": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
            "estimated_margin_jpy": margin,
            "notional_units": units,
            "quote_per_base": quote_per_base,
            "quote_per_base_source": "EXPLICIT_QUOTE_PER_BASE" if explicit_quote_per_base is not None else "ENTRY",
            "portfolio_priority": priority,
            "currency_legs": [
                {"currency": base, "signed_units": direction * units},
                {"currency": quote, "signed_units": -direction * units * quote_per_base},
            ],
            "source_signal_sha256": seal,
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        },
        [],
    )


def _normalize_constraints(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("constraints must be a mapping")
    margin = _positive_float(raw.get("margin_budget_jpy"))
    if margin is None:
        raise ValueError("margin_budget_jpy must be a positive finite number")
    gross = _normalize_positive_limit(raw.get("max_currency_gross"), name="max_currency_gross")
    concurrent = _normalize_positive_integer_limit(
        raw.get("max_concurrent_signals_per_horizon"),
        name="max_concurrent_signals_per_horizon",
    )
    return {
        "margin_budget_jpy": margin,
        "max_currency_gross": gross,
        "max_concurrent_signals_per_horizon": concurrent,
    }


def _normalize_positive_limit(value: Any, *, name: str) -> float | dict[str, float]:
    scalar = _positive_float(value)
    if scalar is not None:
        return scalar
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a positive number or non-empty mapping")
    result: dict[str, float] = {}
    for key, raw_limit in value.items():
        token = str(key).upper()
        limit = _positive_float(raw_limit)
        if (token != "*" and not _TOKEN_RE.fullmatch(token)) or limit is None:
            raise ValueError(f"{name} contains an invalid key or limit")
        result[token] = limit
    return result


def _normalize_positive_integer_limit(value: Any, *, name: str) -> int | dict[str, int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a positive integer or non-empty mapping")
    result: dict[str, int] = {}
    for key, raw_limit in value.items():
        token = str(key).upper()
        if (
            (token != "*" and not _TOKEN_RE.fullmatch(token))
            or not isinstance(raw_limit, int)
            or isinstance(raw_limit, bool)
            or raw_limit <= 0
        ):
            raise ValueError(f"{name} contains an invalid key or limit")
        result[token] = raw_limit
    return result


def _evaluate_candidate_constraints(
    trial: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    horizon = str(candidate["horizon_lane"])
    same_horizon = [row for row in trial if row["horizon_lane"] == horizon]
    horizon_slices = _risk_slices(same_horizon)
    global_slices = _global_risk_slices(trial)
    count_limit = int(_limit_for(constraints["max_concurrent_signals_per_horizon"], horizon))
    margin_limit = float(constraints["margin_budget_jpy"])
    gross_limits = constraints["max_currency_gross"]
    reasons: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for risk_slice in horizon_slices:
        active_ids = risk_slice["active_signal_ids"]
        if candidate["signal_id"] not in active_ids:
            continue
        if risk_slice["concurrent_signal_count"] > count_limit:
            reasons.add("MAX_CONCURRENT_SIGNALS_PER_HORIZON")
            evidence.append(
                {
                    "constraint": "MAX_CONCURRENT_SIGNALS_PER_HORIZON",
                    "observed": risk_slice["concurrent_signal_count"],
                    "limit": count_limit,
                    "at_utc": risk_slice["at_utc"],
                }
            )
    for risk_slice in global_slices:
        active_ids = risk_slice["active_signal_ids"]
        if candidate["signal_id"] not in active_ids:
            continue
        if risk_slice["margin_jpy"] > margin_limit:
            reasons.add("MARGIN_BUDGET_JPY")
            evidence.append(
                {
                    "constraint": "MARGIN_BUDGET_JPY",
                    "observed": risk_slice["margin_jpy"],
                    "limit": margin_limit,
                    "at_utc": risk_slice["at_utc"],
                }
            )
        for currency, exposure in risk_slice["currency_exposure"].items():
            limit = float(_limit_for(gross_limits, currency))
            if exposure["gross_units"] > limit:
                reasons.add(f"MAX_CURRENCY_GROSS:{currency}")
                evidence.append(
                    {
                        "constraint": f"MAX_CURRENCY_GROSS:{currency}",
                        "observed": exposure["gross_units"],
                        "limit": limit,
                        "at_utc": risk_slice["at_utc"],
                    }
                )
    return {"binding_reasons": sorted(reasons), "evidence": evidence}


def _global_risk_slices(signals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = sorted({_parse_utc(row["holding_window"]["start_utc"]) for row in signals})
    result: list[dict[str, Any]] = []
    for point in points:
        assert point is not None
        active = [
            row
            for row in signals
            if _parse_utc(row["holding_window"]["start_utc"]) <= point
            < _parse_utc(row["holding_window"]["end_utc"])
        ]
        result.append(
            {
                "at_utc": point.isoformat(),
                "active_horizon_lanes": sorted({str(row["horizon_lane"]) for row in active}),
                "active_signal_ids": sorted(str(row["signal_id"]) for row in active),
                "concurrent_signal_count": len(active),
                "margin_jpy": round(sum(float(row["estimated_margin_jpy"]) for row in active), 6),
                "currency_exposure": _currency_exposure(active)["by_currency"],
            }
        )
    return result


def _risk_slices(signals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_horizon: dict[str, list[Mapping[str, Any]]] = {}
    for signal in signals:
        by_horizon.setdefault(str(signal["horizon_lane"]), []).append(signal)
    result: list[dict[str, Any]] = []
    for horizon, rows in sorted(by_horizon.items()):
        points = sorted({_parse_utc(row["holding_window"]["start_utc"]) for row in rows})
        for point in points:
            assert point is not None
            active = [
                row
                for row in rows
                if _parse_utc(row["holding_window"]["start_utc"]) <= point
                < _parse_utc(row["holding_window"]["end_utc"])
            ]
            result.append(
                {
                    "horizon_lane": horizon,
                    "at_utc": point.isoformat(),
                    "active_signal_ids": sorted(str(row["signal_id"]) for row in active),
                    "concurrent_signal_count": len(active),
                    "margin_jpy": round(sum(float(row["estimated_margin_jpy"]) for row in active), 6),
                    "currency_exposure": _currency_exposure(active)["by_currency"],
                }
            )
    return result


def _currency_exposure(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    legs: list[dict[str, Any]] = []
    by_currency: dict[str, dict[str, float]] = {}
    for signal in signals:
        for leg in signal["currency_legs"]:
            signed = float(leg["signed_units"])
            currency = str(leg["currency"])
            legs.append(
                {
                    "signal_id": signal["signal_id"],
                    "pair": signal["pair"],
                    "horizon_lane": signal["horizon_lane"],
                    "currency": currency,
                    "signed_units": signed,
                    "gross_units": abs(signed),
                }
            )
            row = by_currency.setdefault(
                currency,
                {"long_units": 0.0, "short_units": 0.0, "gross_units": 0.0, "net_units": 0.0},
            )
            if signed >= 0.0:
                row["long_units"] += signed
            else:
                row["short_units"] += abs(signed)
            row["gross_units"] += abs(signed)
            row["net_units"] += signed
    return {
        "unit_basis": "BASE_UNITS_AND_REFERENCE_ENTRY_QUOTE_UNITS",
        "legs": legs,
        "by_currency": {
            currency: {key: round(value, 6) for key, value in row.items()}
            for currency, row in sorted(by_currency.items())
        },
    }


def _opposite_side_coexistence(
    selected: Sequence[Mapping[str, Any]],
    *,
    hedging_enabled: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    opposing_ids: set[str] = set()
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if left["pair"] != right["pair"] or left["side"] == right["side"]:
                continue
            opposing_ids.update((str(left["signal_id"]), str(right["signal_id"])))
            rows.append(
                {
                    "pair": left["pair"],
                    "signal_ids": [left["signal_id"], right["signal_id"]],
                    "sides": [left["side"], right["side"]],
                    "horizon_lanes": [left["horizon_lane"], right["horizon_lane"]],
                    "same_horizon": left["horizon_lane"] == right["horizon_lane"],
                    "holding_windows_overlap": _windows_overlap(left, right),
                    "shadow_paths_coexist": True,
                    "broker_projection": (
                        "BROKER_HEDGE_CAPABLE_IF_SEPARATELY_PROMOTED"
                        if hedging_enabled
                        else "VIRTUAL_ONLY_NETTING_ACCOUNT"
                    ),
                }
            )
    return rows, opposing_ids


def _windows_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start = _parse_utc(left["holding_window"]["start_utc"])
    left_end = _parse_utc(left["holding_window"]["end_utc"])
    right_start = _parse_utc(right["holding_window"]["start_utc"])
    right_end = _parse_utc(right["holding_window"]["end_utc"])
    assert left_start and left_end and right_start and right_end
    return left_start < right_end and right_start < left_end


def _public_signal(signal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in signal.items()
        if key not in {"source_index"}
    }


def _rank_key(signal: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(signal["portfolio_priority"]),
        str(signal["horizon_lane"]),
        str(signal["pair"]),
        str(signal["side"]),
        str(signal["method"]),
        str(signal["signal_id"]),
        str(signal["source_signal_sha256"]),
    )


def _limit_for(limit: Any, key: str) -> float | int:
    if isinstance(limit, Mapping):
        if key in limit:
            return limit[key]
        if "*" in limit:
            return limit["*"]
        raise ValueError(f"no configured constraint for {key}")
    return limit


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
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


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_float(value: Any) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result > 0.0 else None


def _nonnegative_float(value: Any) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result >= 0.0 else None


def _bounded_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if 0 < len(stripped) <= 256 else None


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
