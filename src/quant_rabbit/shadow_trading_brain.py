"""Composed shadow trading brain (weakness ledger W25 integration).

This module is the composition that turns the individual shadow guards into
one system.  It runs a full cycle in the pro-trader-brain shape:

  1. Perception   — bind the evidence bundle (packet + proprietary indicators)
  2. Inventory    — three-way reconcile broker/ledger/lane ownership (W24)
  3. Read handoff — LAYER 2 IS RESERVED FOR CODEX'S LIVE AI TRADER.  The brain
                    VALIDATES and CONSUMES a market read; it never authors the
                    intuition.  A deterministic SHADOW_PLACEHOLDER read is the
                    only self-generated form, and it is flagged as such.
  4. Metacognition — weight the read by the supervisor's measured accuracy;
                    families flagged for auto-CAUTION are demoted (W17)
  5. Gates        — pre-entry close-distance and high-cost-window admission
  6. Discipline   — per-currency exposure cap + pre-declared conviction sizing
  7. Seal         — funnel counts, GO risk 0, empty intents, authority NONE

The brain grants no order authority, no live permission, and no broker
mutation.  Sizing output is a NAV risk fraction for the shadow record only;
live execution remains the operator-approved Codex AI trader's job.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.close_distance_gate import evaluate_close_distance_gate
from quant_rabbit.conviction_ladder import allowed_risk_fraction, declared_condition_count
from quant_rabbit.cost_window_mask import evaluate_cost_window
from quant_rabbit.currency_exposure_guard import evaluate_currency_exposure
from quant_rabbit.gate_throughput_slo import build_gate_throughput_slo
from quant_rabbit.portfolio_inventory_reconciliation import build_portfolio_inventory

CONTRACT = "QR_SHADOW_TRADING_BRAIN_CYCLE_V1"
READ_HANDOFF_CONTRACT = "QR_MARKET_READ_HANDOFF_V1"
LAYER2_LIVE_SOURCE = "CODEX_AI_TRADER"
LAYER2_SHADOW_SOURCE = "SHADOW_PLACEHOLDER"
_READ_SOURCES = frozenset({LAYER2_LIVE_SOURCE, LAYER2_SHADOW_SOURCE})
_ACTIONS = frozenset({"GO", "CAUTION", "STOP"})
_REGIMES = frozenset({"TREND", "RANGE", "SQUEEZE", "EVENT", "UNCLEAR"})
_SIDES = frozenset({"LONG", "SHORT"})
_SHA_RE_CHARS = frozenset("0123456789abcdef")


class ShadowBrainError(ValueError):
    """Raised when a brain-cycle input is malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in _SHA_RE_CHARS for char in text):
        raise ShadowBrainError(f"{label} must be a lowercase sha256")
    return text


def validate_market_read_handoff(read: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the LAYER-2 read without authoring it.

    The brain refuses a malformed read fail-closed.  It never fills in a
    missing read: an absent read means the live AI trader has not spoken and
    the cycle proceeds with zero GO candidates, not a fabricated opinion.
    """

    if not isinstance(read, Mapping):
        raise ShadowBrainError("market read must be an object")
    source = str(read.get("read_source") or "")
    if source not in _READ_SOURCES:
        raise ShadowBrainError("market read source is invalid")
    regime = str(read.get("declared_regime") or "").upper()
    if regime not in _REGIMES:
        raise ShadowBrainError("market read declared_regime is invalid")
    rows = read.get("pair_reads")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ShadowBrainError("market read pair_reads must be a list")
    sealed_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ShadowBrainError("pair read row must be an object")
        pair = str(row.get("pair") or "").upper()
        if len(pair.split("_")) != 2:
            raise ShadowBrainError("pair read pair identity is invalid")
        if pair in sealed_rows:
            raise ShadowBrainError(f"duplicate pair read: {pair}")
        action = str(row.get("action") or "").upper()
        if action not in _ACTIONS:
            raise ShadowBrainError(f"pair read action is invalid: {pair}")
        side = str(row.get("side") or "").upper()
        if action == "GO" and side not in _SIDES:
            raise ShadowBrainError(f"GO read requires a side: {pair}")
        narrative = _validated_sha(row.get("narrative_sha256"), f"{pair} narrative")
        predicted = str(row.get("predicted_direction") or "").upper()
        if action == "GO" and predicted != side:
            raise ShadowBrainError(
                f"GO read prediction must match the traded side: {pair}"
            )
        conditions = row.get("conviction_conditions") or []
        checklist = [
            (str(name), bool(met)) for name, met in conditions
        ]
        sealed_rows[pair] = {
            "pair": pair,
            "action": action,
            "side": side if action == "GO" else None,
            "narrative_sha256": narrative,
            "predicted_direction": predicted if action == "GO" else None,
            "conviction_conditions": checklist,
        }
    return {
        "contract": READ_HANDOFF_CONTRACT,
        "read_source": source,
        "authored_by_brain": source == LAYER2_SHADOW_SOURCE,
        "live_authoring_reserved_for": LAYER2_LIVE_SOURCE,
        "declared_regime": regime,
        "pair_reads": sealed_rows,
    }


def run_shadow_brain_cycle(
    *,
    cycle_id: str,
    decision_utc: datetime,
    evidence_packet_sha256: str,
    proprietary_indicator_sha256: str,
    broker_positions: Sequence[Mapping[str, Any]],
    ledger_open_positions: Sequence[Mapping[str, Any]],
    manual_no_touch_ids: Sequence[str],
    nav_account_currency: float,
    broker_snapshot_sha256: str,
    ledger_tip_sha256: str,
    market_read: Mapping[str, Any],
    supervision_scorecard: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    measured_regime: Mapping[str, Any] | None = None,
    family_catalog: Mapping[str, Any] | None = None,
    currency_cap_fraction: float = 0.5,
    pass_rate_floor: float = 0.10,
    today_nav_return_fraction: float = 0.0,
    consecutive_losing_trades: int = 0,
    prior_week_nav_return_fraction: float = 0.0,
) -> dict[str, Any]:
    """Run one fail-closed shadow brain cycle over proposed candidates.

    When a measured regime classification is supplied, the brain trusts the
    MEASUREMENT over the read's declared regime: a family may be admitted
    only if the family catalog lists it as eligible for the measured
    regime x vol cell, and a read whose declared regime contradicts the
    measurement is flagged.  This is the operator's "measure the state, not
    the clock" rule enforced at the composition boundary.
    """

    if decision_utc.tzinfo is None:
        raise ShadowBrainError("decision clock must be timezone-aware")
    decision = decision_utc.astimezone(timezone.utc)

    perception = {
        "evidence_packet_sha256": _validated_sha(
            evidence_packet_sha256, "evidence_packet_sha256"
        ),
        "proprietary_indicator_sha256": _validated_sha(
            proprietary_indicator_sha256, "proprietary_indicator_sha256"
        ),
    }

    measured_regime_label = None
    measured_vol_state = None
    eligible_families: set[str] | None = None
    routing_status = None
    if measured_regime is not None:
        if not isinstance(measured_regime, Mapping):
            raise ShadowBrainError("measured_regime must be an object")
        mbody = {
            key: value
            for key, value in measured_regime.items()
            if key != "classification_sha256"
        }
        if measured_regime.get("classification_sha256") != _canonical_sha(mbody):
            raise ShadowBrainError("measured_regime digest is invalid")
        measured_regime_label = str(measured_regime.get("regime") or "").upper()
        measured_vol_state = str(measured_regime.get("vol_state") or "").upper()
        if family_catalog is not None:
            from quant_rabbit.regime_family_router import (
                RegimeFamilyRouterError,
                route_families,
            )

            try:
                routing = route_families(
                    family_catalog,
                    declared_regime=measured_regime_label,
                    vol_state=measured_vol_state,
                )
            except RegimeFamilyRouterError as error:
                raise ShadowBrainError(
                    f"family catalog routing failed: {error}"
                ) from error
            eligible_families = {str(f) for f in routing["eligible_families"]}
            routing_status = routing["routing_status"]

    inventory = build_portfolio_inventory(
        broker_positions=broker_positions,
        ledger_open_positions=ledger_open_positions,
        manual_no_touch_ids=manual_no_touch_ids,
        nav_account_currency=nav_account_currency,
        as_of_utc=decision,
        broker_snapshot_sha256=broker_snapshot_sha256,
        ledger_tip_sha256=ledger_tip_sha256,
    )
    read = validate_market_read_handoff(market_read)

    # Metacognition: families the supervisor has proven unreliable are
    # demoted before a single candidate is considered.
    auto_caution = set()
    if supervision_scorecard is not None:
        if not isinstance(supervision_scorecard, Mapping):
            raise ShadowBrainError("supervision scorecard must be an object")
        body = {
            key: value
            for key, value in supervision_scorecard.items()
            if key != "scorecard_sha256"
        }
        if supervision_scorecard.get("scorecard_sha256") != _canonical_sha(body):
            raise ShadowBrainError("supervision scorecard digest is invalid")
        auto_caution = {
            str(item)
            for item in supervision_scorecard.get(
                "supervision_auto_caution_required", ()
            )
        }

    open_tradeable = [
        {
            "pair": row["pair"],
            "side": row["side"],
            "nav_exposure_fraction": row["nav_exposure_fraction"],
        }
        for row in inventory["position_rows"]
        if row["ownership"] != "MANUAL_NO_TOUCH"
    ]

    close_ok_by_hold: dict[int, bool] = {}
    cost_decision = evaluate_cost_window(decision)
    candidate_rows: list[dict[str, Any]] = []
    admitted = 0
    for candidate in candidates:
        pair = str(candidate.get("pair") or "").upper()
        side = str(candidate.get("side") or "").upper()
        hold_minutes = candidate.get("hold_minutes")
        family_id = str(candidate.get("family_id") or "").upper()
        refusals: list[str] = []

        if not inventory["reconciled"]:
            refusals.append("INVENTORY_UNRECONCILED")

        # A candidate must self-identify its family, or the supervisor's
        # auto-CAUTION demotion could be evaded by simply omitting it.
        if not family_id:
            refusals.append("UNKNOWN_FAMILY")

        pair_read = read["pair_reads"].get(pair)
        if pair_read is None or pair_read["action"] != "GO":
            refusals.append("NO_GO_READ")
        elif pair_read["side"] != side:
            refusals.append("READ_SIDE_MISMATCH")
        elif family_id and family_id in auto_caution:
            refusals.append("FAMILY_AUTO_CAUTION_UNRELIABLE_SUPERVISOR")

        # Measured regime wins over the declared one: a family may trade only
        # in the regime x vol cell it is catalogued for, as MEASURED.
        if eligible_families is not None and family_id:
            if family_id not in eligible_families:
                refusals.append("FAMILY_NOT_ELIGIBLE_FOR_MEASURED_CELL")

        if not cost_decision.admitted:
            refusals.append(cost_decision.reason)

        if isinstance(hold_minutes, int) and not isinstance(hold_minutes, bool) and hold_minutes > 0:
            if hold_minutes not in close_ok_by_hold:
                close_ok_by_hold[hold_minutes] = evaluate_close_distance_gate(
                    decision, hold_minutes=hold_minutes
                ).admitted
            if not close_ok_by_hold[hold_minutes]:
                refusals.append("HOLD_WOULD_CROSS_NEXT_FX_CLOSE")
        else:
            refusals.append("INVALID_HOLD_MINUTES")

        # A missing/zero/negative exposure must fail closed, not be coerced to
        # a negligible size that silently passes the per-currency cap.
        raw_exposure = candidate.get("nav_exposure_fraction")
        if (
            isinstance(raw_exposure, bool)
            or not isinstance(raw_exposure, (int, float))
            or not float(raw_exposure) > 0.0
        ):
            refusals.append("INVALID_NAV_EXPOSURE_FRACTION")
        elif side in _SIDES:
            exposure = evaluate_currency_exposure(
                open_tradeable,
                {
                    "pair": pair,
                    "side": side,
                    "nav_exposure_fraction": float(raw_exposure),
                },
                currency_cap_fraction=currency_cap_fraction,
            )
            if not exposure.admitted:
                refusals.append(exposure.reason)
        else:
            refusals.append("INVALID_SIDE")

        sizing = None
        if not refusals:
            conditions = pair_read["conviction_conditions"] if pair_read else []
            condition_count = (
                declared_condition_count(conditions) if conditions else 0
            )
            sizing = allowed_risk_fraction(
                conviction_conditions_met=condition_count,
                today_nav_return_fraction=today_nav_return_fraction,
                consecutive_losing_trades=consecutive_losing_trades,
                prior_week_nav_return_fraction=prior_week_nav_return_fraction,
            ).payload()
            if sizing["risk_fraction"] <= 0.0:
                refusals.append(sizing["reason"])
                sizing = None

        is_admitted = not refusals
        admitted += int(is_admitted)
        candidate_rows.append(
            {
                "pair": pair,
                "side": side,
                "family_id": family_id or None,
                "admitted": is_admitted,
                "refusal_reasons": refusals,
                "shadow_risk_fraction": sizing["risk_fraction"] if sizing else 0.0,
                "shadow_intent_only": True,
            }
        )

    funnel = build_gate_throughput_slo(
        [
            {
                "signals_generated": len(candidate_rows),
                "gates": [
                    {
                        "gate_id": "BRAIN_COMPOSED_ADMISSION",
                        "evaluated": len(candidate_rows),
                        "admitted": admitted,
                    }
                ],
            }
        ],
        window_label=f"cycle:{cycle_id}",
        pass_rate_floor=pass_rate_floor,
    )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "cycle_id": str(cycle_id),
        "decision_utc": decision.isoformat(),
        "layers": {
            "1_perception": perception,
            "2_read_handoff": {
                "read_source": read["read_source"],
                "declared_regime": read["declared_regime"],
                "authored_by_brain": read["authored_by_brain"],
                "live_authoring_reserved_for": LAYER2_LIVE_SOURCE,
            },
            "3_metacognition": {
                "supervision_scorecard_present": supervision_scorecard is not None,
                "auto_caution_families": sorted(auto_caution),
                "measured_regime": measured_regime_label,
                "measured_vol_state": measured_vol_state,
                "read_regime_matches_measured": (
                    None
                    if measured_regime_label is None
                    else read["declared_regime"] == measured_regime_label
                ),
                "measured_cell_routing_status": routing_status,
                "measured_cell_eligible_families": (
                    None if eligible_families is None else sorted(eligible_families)
                ),
            },
            "4_discipline": {
                "currency_cap_fraction": float(currency_cap_fraction),
                "daily_stop_and_ladder": "PREDECLARED_CONVICTION_LADDER_V1",
            },
        },
        "inventory_status": inventory["status"],
        "inventory_sha256": inventory["inventory_sha256"],
        "candidate_rows": candidate_rows,
        "admitted_candidate_count": admitted,
        "gate_throughput_slo": {
            "end_to_end_pass_rate": funnel["end_to_end_pass_rate"],
            "floor_breached": funnel["floor_breached"],
            "slo_sha256": funnel["slo_sha256"],
        },
        "go_risk_jpy": 0.0,
        "order_intents": [],
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "layer2_live_decision_owner": LAYER2_LIVE_SOURCE,
    }
    return {**body, "cycle_sha256": _canonical_sha(body)}
