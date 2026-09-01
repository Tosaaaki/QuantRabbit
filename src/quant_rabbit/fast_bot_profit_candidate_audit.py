"""Read-only, cross-resident audit for fast-bot profit candidates.

The resident collector is commit-addressed, so one market window can be split
across several immutable runtime roots.  Looking at only the newest root can
turn a tiny post-hoc slice into an apparently profitable candidate.  This
module joins every sealed signal/outcome by ``signal_sha256``, evaluates one
finite candidate universe, and refuses to activate any result automatically.

The audit is research evidence only.  It creates no order intent, broker
mutation, live permission, parameter change, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_truth import (
    _fast_bot_outcome_valid_for_signal,
    _fast_bot_signal_valid,
)


AUDIT_CONTRACT = "QR_FAST_BOT_RESIDENT_PROFIT_CANDIDATE_AUDIT_V1"
ENTRY_ARMS = (
    "PASSIVE_NEAR_SIDE",
    "PASSIVE_QUARTER_SPREAD",
    "PASSIVE_MID_SPREAD",
    "PASSIVE_THREE_QUARTER_SPREAD",
)
ATR_FLOORS: tuple[float | None, ...] = (None, 2.0, 3.0, 4.0, 5.0, 6.0)
LANE_FIELDS = ("pair", "side", "method", "horizon_lane")

# This is only an admission floor for a *future* pre-registration.  Passing it
# is not profitability proof.  The final prospective gate remains stricter.
CANDIDATE_ADMISSION_THRESHOLDS = {
    "minimum_samples": 30,
    "minimum_active_days": 3,
    "minimum_profit_factor": 1.25,
    "minimum_pessimistic_expectancy_pips": 0.0,
    "minimum_positive_day_rate": 2.0 / 3.0,
    "maximum_daily_sample_share": 0.70,
}


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def audit_resident_profit_candidates(
    state_root: Path,
    *,
    generated_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Audit every immutable resident root without changing it."""

    generated = _aware_utc(generated_at_utc or datetime.now(timezone.utc))
    roots = sorted(
        path
        for path in state_root.iterdir()
        if path.is_dir() and (path / "ledgers").is_dir()
    )
    source_files: list[dict[str, Any]] = []
    signals_by_sha: dict[str, dict[str, Any]] = {}
    outcomes_by_signal_sha: dict[str, dict[str, Any]] = {}
    signal_id_to_shas: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_signal_rows = 0
    duplicate_outcome_rows = 0
    integrity_errors: list[str] = []

    for root in roots:
        signal_rows, signal_source = _load_jsonl_snapshot(
            state_root,
            root / "ledgers" / "fast_bot_shadow_ledger.jsonl",
        )
        outcome_rows, outcome_source = _load_jsonl_snapshot(
            state_root,
            root / "ledgers" / "fast_bot_outcome_ledger.jsonl",
        )
        source_files.extend((signal_source, outcome_source))
        for row in signal_rows:
            if not _fast_bot_signal_valid(row):
                integrity_errors.append("INVALID_FAST_BOT_SIGNAL")
                continue
            signal_sha = str(row["signal_sha256"])
            signal_id_to_shas[str(row["signal_id"])].add(signal_sha)
            prior = signals_by_sha.get(signal_sha)
            if prior is None:
                signals_by_sha[signal_sha] = row
            elif prior == row:
                duplicate_signal_rows += 1
            else:
                integrity_errors.append("CONFLICTING_SIGNAL_SHA256_ROWS")
        for row in outcome_rows:
            signal_sha = str(row.get("signal_sha256") or "")
            prior = outcomes_by_signal_sha.get(signal_sha)
            if prior is None:
                outcomes_by_signal_sha[signal_sha] = row
            elif prior == row:
                duplicate_outcome_rows += 1
            else:
                integrity_errors.append("CONFLICTING_OUTCOME_SIGNAL_SHA256_ROWS")

    valid_outcomes: dict[str, dict[str, Any]] = {}
    for signal_sha, outcome in outcomes_by_signal_sha.items():
        signal = signals_by_sha.get(signal_sha)
        if signal is None:
            integrity_errors.append("OUTCOME_WITHOUT_MATCHING_SIGNAL")
        elif not _fast_bot_outcome_valid_for_signal(outcome, signal):
            integrity_errors.append("INVALID_FAST_BOT_OUTCOME")
        else:
            valid_outcomes[signal_sha] = outcome

    signal_id_collisions = [
        {"signal_id": signal_id, "distinct_signal_sha256s": sorted(shas)}
        for signal_id, shas in sorted(signal_id_to_shas.items())
        if len(shas) > 1
    ]
    source_bundle_sha = canonical_sha(
        [
            {
                "path": row["path"],
                "file_sha256": row["file_sha256"],
                "bytes": row["bytes"],
                "rows": row["rows"],
            }
            for row in source_files
        ]
    )

    signals = list(signals_by_sha.values())
    lanes = sorted({_lane(signal) for signal in signals})
    candidates: list[dict[str, Any]] = []
    for lane in lanes:
        lane_signals = [signal for signal in signals if _lane(signal) == lane]
        for arm_id in ENTRY_ARMS:
            for atr_floor in ATR_FLOORS:
                cohort = [
                    signal
                    for signal in lane_signals
                    if _atr_admitted(signal, atr_floor)
                ]
                metrics = candidate_metrics(
                    cohort,
                    valid_outcomes,
                    arm_id=arm_id,
                )
                checks = candidate_admission_checks(metrics)
                candidates.append(
                    {
                        "candidate_id": _candidate_id(lane, arm_id, atr_floor),
                        "lane": dict(zip(LANE_FIELDS, lane)),
                        "entry_arm": arm_id,
                        "m5_atr_pips_minimum": atr_floor,
                        "metrics": metrics,
                        "admission_checks": checks,
                        "admission_passed": all(checks.values()),
                    }
                )

    research_leads = [row for row in candidates if row["admission_passed"]]
    ranked = sorted(
        (row for row in candidates if row["metrics"]["filled_signals"] > 0),
        key=lambda row: (
            _sortable_profit_factor(row["metrics"].get("profit_factor")),
            float(row["metrics"].get("net_pips") or 0.0),
            int(row["metrics"].get("filled_signals") or 0),
            row["candidate_id"],
        ),
        reverse=True,
    )
    exact_v2 = next(
        (
            row
            for row in candidates
            if row["lane"]
            == {
                "pair": "USD_JPY",
                "side": "SHORT",
                "method": "RANGE_ROTATION",
                "horizon_lane": "M1_EXECUTION_15M_HOLD",
            }
            and row["entry_arm"] == "PASSIVE_NEAR_SIDE"
            and row["m5_atr_pips_minimum"] == 5.0
        ),
        None,
    )
    aggregate_arms = {
        arm_id: candidate_metrics(signals, valid_outcomes, arm_id=arm_id)
        for arm_id in ENTRY_ARMS
    }

    status = (
        "REJECT_SOURCE_INTEGRITY"
        if integrity_errors
        else "RESEARCH_LEADS_REQUIRE_SEPARATE_PRECOMMITMENT"
        if research_leads
        else "NO_ADMISSIBLE_CANDIDATE"
    )
    body = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated.isoformat(),
        "status": status,
        "source_state_root": str(state_root.resolve()),
        "source_root_count": len(roots),
        "source_files": source_files,
        "source_bundle_sha256": source_bundle_sha,
        "source_integrity_passed": not integrity_errors,
        "source_integrity_errors": sorted(set(integrity_errors)),
        "duplicate_signal_rows_removed": duplicate_signal_rows,
        "duplicate_outcome_rows_removed": duplicate_outcome_rows,
        "signal_id_collision_count": len(signal_id_collisions),
        "signal_id_collisions": signal_id_collisions,
        "unique_sealed_signals": len(signals_by_sha),
        "unique_valid_outcomes": len(valid_outcomes),
        "unresolved_signals": len(signals_by_sha) - len(valid_outcomes),
        "active_signal_days": sorted(
            {str(signal["generated_at_utc"])[:10] for signal in signals}
        ),
        "candidate_universe": {
            "policy": "EXACT_LANE_X_PRECOMMITTED_ENTRY_ARM_X_FIXED_ATR_FLOOR_V1",
            "entry_arms": list(ENTRY_ARMS),
            "m5_atr_pips_floors": list(ATR_FLOORS),
            "candidate_count": len(candidates),
            "multiple_testing_corrected": False,
            "automatic_candidate_activation_allowed": False,
        },
        "candidate_admission_thresholds": dict(CANDIDATE_ADMISSION_THRESHOLDS),
        "research_lead_count": len(research_leads),
        "research_leads": research_leads,
        "top_observed_candidates": ranked[:20],
        "v2_candidate_reassessment": exact_v2,
        "aggregate_entry_arms": aggregate_arms,
        "profitability_claim": (
            "REJECT_SOURCE_INTEGRITY"
            if integrity_errors
            else "NO_ADMISSIBLE_CANDIDATE"
            if not research_leads
            else "RESEARCH_LEADS_ONLY_NOT_PROFITABILITY_PROOF"
        ),
        "shadow_only": True,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "automatic_candidate_activation_allowed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    return seal(body)


def candidate_metrics(
    signals: Sequence[Mapping[str, Any]],
    outcomes_by_signal_sha: Mapping[str, Mapping[str, Any]],
    *,
    arm_id: str,
) -> dict[str, Any]:
    """Return spread-included metrics for one immutable entry arm."""

    resolved = 0
    filled_values: list[float] = []
    daily: defaultdict[str, list[float]] = defaultdict(list)
    for signal in signals:
        outcome = outcomes_by_signal_sha.get(str(signal.get("signal_sha256") or ""))
        if outcome is None:
            continue
        arm = _outcome_arm(outcome, arm_id)
        if arm is None:
            continue
        resolved += 1
        if arm.get("filled") is True:
            realized = _finite(arm.get("realized_pips"))
            if realized is None:
                continue
            filled_values.append(realized)
            daily[str(signal.get("generated_at_utc") or "")[:10]].append(realized)

    wins = [value for value in filled_values if value > 0.0]
    losses = [-value for value in filled_values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else math.inf if gross_profit > 0.0 else None
    )
    positive_day_rate = (
        sum(sum(values) > 0.0 for values in daily.values()) / len(daily)
        if daily
        else 0.0
    )
    maximum_daily_share = (
        max(len(values) for values in daily.values()) / len(filled_values)
        if filled_values
        else 1.0
    )
    pessimistic = _pessimistic_expectancy(wins, losses)
    return {
        "source_signals": len(signals),
        "resolved_signals": resolved,
        "filled_signals": len(filled_values),
        "unfilled_signals": resolved - len(filled_values),
        "active_days": len(daily),
        "wins": len(wins),
        "losses": len(losses),
        "net_pips": round(sum(filled_values), 6),
        "mean_pips_per_fill": (
            round(sum(filled_values) / len(filled_values), 6)
            if filled_values
            else None
        ),
        "profit_factor": (
            round(profit_factor, 6)
            if profit_factor is not None and math.isfinite(profit_factor)
            else "INF" if profit_factor == math.inf else None
        ),
        "pessimistic_expectancy_pips": (
            round(pessimistic, 6) if pessimistic is not None else None
        ),
        "positive_day_rate": round(positive_day_rate, 6),
        "maximum_daily_sample_share": round(maximum_daily_share, 6),
        "daily": [
            {
                "date": day,
                "filled_signals": len(values),
                "net_pips": round(sum(values), 6),
            }
            for day, values in sorted(daily.items())
        ],
        "spread_included": True,
    }


def candidate_admission_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    thresholds = CANDIDATE_ADMISSION_THRESHOLDS
    profit_factor = (
        math.inf
        if metrics.get("profit_factor") == "INF"
        else _finite(metrics.get("profit_factor"))
    )
    pessimistic = _finite(metrics.get("pessimistic_expectancy_pips"))
    return {
        "minimum_samples": int(metrics.get("filled_signals") or 0)
        >= int(thresholds["minimum_samples"]),
        "minimum_active_days": int(metrics.get("active_days") or 0)
        >= int(thresholds["minimum_active_days"]),
        "minimum_profit_factor": profit_factor is not None
        and profit_factor >= float(thresholds["minimum_profit_factor"]),
        "minimum_pessimistic_expectancy_pips": pessimistic is not None
        and pessimistic > float(thresholds["minimum_pessimistic_expectancy_pips"]),
        "minimum_positive_day_rate": float(
            metrics.get("positive_day_rate") or 0.0
        )
        >= float(thresholds["minimum_positive_day_rate"]),
        "maximum_daily_sample_share": float(
            metrics.get("maximum_daily_sample_share") or 1.0
        )
        <= float(thresholds["maximum_daily_sample_share"]),
    }


def render_audit_report(audit: Mapping[str, Any]) -> str:
    near = (audit.get("aggregate_entry_arms") or {}).get("PASSIVE_NEAR_SIDE") or {}
    v2 = ((audit.get("v2_candidate_reassessment") or {}).get("metrics") or {})
    return "\n".join(
        [
            "# Fast Bot Resident Profit Candidate Audit",
            "",
            f"- Generated: `{audit.get('generated_at_utc')}`",
            f"- Status: `{audit.get('status')}`",
            f"- Unique signals / valid outcomes: {audit.get('unique_sealed_signals')} / {audit.get('unique_valid_outcomes')}",
            f"- Active signal days: {', '.join(audit.get('active_signal_days') or []) or 'none'}",
            f"- Near-side filled / net / PF: {near.get('filled_signals')} / {near.get('net_pips')} / {near.get('profit_factor')}",
            f"- V2 filled / active days / net / PF: {v2.get('filled_signals')} / {v2.get('active_days')} / {v2.get('net_pips')} / {v2.get('profit_factor')}",
            f"- Admissible research leads: {audit.get('research_lead_count')}",
            f"- Signal-id collisions across sealed rows: {audit.get('signal_id_collision_count')}",
            "- Automatic candidate activation: `false`",
            f"- Profitability claim: `{audit.get('profitability_claim')}`",
            "- Execution authority: `NONE`",
            "- Broker mutation: `false`",
            "- Live permission: `false`",
            "",
        ]
    )


def _load_jsonl_snapshot(
    state_root: Path,
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = path.read_bytes() if path.exists() else b""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{number}")
        rows.append(value)
    return rows, {
        "path": str(path.relative_to(state_root)),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": len(rows),
    }


def _lane(signal: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(signal.get(field) or "") for field in LANE_FIELDS)  # type: ignore[return-value]


def _atr_admitted(signal: Mapping[str, Any], floor: float | None) -> bool:
    if floor is None:
        return True
    atr = _finite(signal.get("m5_atr_pips"))
    return atr is not None and atr >= floor


def _candidate_id(
    lane: tuple[str, str, str, str],
    arm_id: str,
    atr_floor: float | None,
) -> str:
    atr = "ANY" if atr_floor is None else f"GTE_{atr_floor:g}"
    return ":".join((*lane, arm_id, f"M5_ATR_{atr}"))


def _outcome_arm(
    outcome: Mapping[str, Any],
    arm_id: str,
) -> Mapping[str, Any] | None:
    experiment = outcome.get("entry_experiment")
    arms = experiment.get("arms") if isinstance(experiment, Mapping) else None
    if not isinstance(arms, list):
        return None
    matched = [
        row
        for row in arms
        if isinstance(row, Mapping) and row.get("arm_id") == arm_id
    ]
    return matched[0] if len(matched) == 1 else None


def _pessimistic_expectancy(
    wins: Sequence[float],
    losses: Sequence[float],
) -> float | None:
    total = len(wins) + len(losses)
    if total == 0:
        return None
    z = 1.96
    observed = len(wins) / total
    denominator = 1.0 + z * z / total
    center = observed + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (observed * (1.0 - observed) + z * z / (4.0 * total)) / total
    )
    lower = max(0.0, (center - margin) / denominator)
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return lower * average_win - (1.0 - lower) * average_loss


def _sortable_profit_factor(value: Any) -> float:
    if value == "INF":
        return math.inf
    finite = _finite(value)
    return finite if finite is not None else -math.inf


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "AUDIT_CONTRACT",
    "ATR_FLOORS",
    "CANDIDATE_ADMISSION_THRESHOLDS",
    "ENTRY_ARMS",
    "audit_resident_profit_candidates",
    "candidate_admission_checks",
    "candidate_metrics",
    "canonical_sha",
    "render_audit_report",
    "seal",
]
