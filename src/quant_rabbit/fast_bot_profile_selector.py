"""Deterministic research-primary and permanent-shadow profile routing.

The selector consumes only sealed profile, activation, and quarantine state.
It never calls an AI model, creates a signal, chooses a side or method, sizes an
order, or grants live permission.  Every catalog lane remains shadow-enabled;
``PRIMARY_RESEARCH`` is only a mutually exclusive research-routing label.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_profile_state import (
    _index_activation_receipts_against_catalog,
    _index_quarantine_records_against_catalog,
)
from quant_rabbit.fast_bot_profiles import (
    AI_ORDER_AUTHORITY,
    ORDER_AUTHORITY,
    FastBotProfile,
    FastBotProfileCatalog,
    LaneKey,
    ProfileContractError,
    canonical_sha256,
    validate_fast_bot_profile_catalog,
)


PROFILE_ROUTING_CONTRACT = "QR_FAST_BOT_PROFILE_ROUTING_V1"

PRIMARY_RESEARCH = "PRIMARY_RESEARCH"
FALLBACK = "FALLBACK"
SHADOW = "SHADOW"
QUARANTINED = "QUARANTINED"
LANE_ROLES = frozenset({PRIMARY_RESEARCH, FALLBACK, SHADOW, QUARANTINED})

PRIMARY_SELECTED = "PRIMARY_SELECTED"
NO_PRIMARY = "NO_PRIMARY"

_AUTHORITY_BODY: dict[str, object] = {
    "ai_order_authority": AI_ORDER_AUTHORITY,
    "order_authority": ORDER_AUTHORITY,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "shadow_only": True,
}


def select_fast_bot_profile_lanes(
    *,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    activation_receipts: Sequence[Mapping[str, Any]] = (),
    quarantine_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve at most one research primary per pair/horizon scope.

    A missing activation receipt or an exhausted sealed fallback path produces
    explicit ``NO_PRIMARY``.  There is no implicit first-profile/default path.
    """

    catalog = validate_fast_bot_profile_catalog(catalog)
    activations = _index_activation_receipts_against_catalog(
        catalog, activation_receipts
    )
    quarantines = _index_quarantine_records_against_catalog(catalog, quarantine_records)
    return _build_routing(catalog, activations, quarantines)


def validate_fast_bot_profile_routing(
    value: object,
    *,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    activation_receipts: Sequence[Mapping[str, Any]] = (),
    quarantine_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild from sealed sources and reject any routing or role mutation."""

    expected = select_fast_bot_profile_lanes(
        catalog=catalog,
        activation_receipts=activation_receipts,
        quarantine_records=quarantine_records,
    )
    return _snapshot_like(value, expected, label="profile routing")


def _build_routing(
    catalog: FastBotProfileCatalog,
    activations: Mapping[tuple[str, str], Mapping[str, Any]],
    quarantines: Mapping[LaneKey, Mapping[str, Any]],
) -> dict[str, Any]:
    profiles_by_id = {profile.profile_id: profile for profile in catalog.profiles}
    lanes = tuple(
        sorted(
            catalog.lanes(),
            key=lambda lane: (lane.pair, lane.horizon_lane, lane.profile_id),
        )
    )
    lanes_by_scope_lists: dict[tuple[str, str], list[LaneKey]] = {}
    for lane in lanes:
        lanes_by_scope_lists.setdefault((lane.pair, lane.horizon_lane), []).append(lane)
    scopes = tuple(sorted(lanes_by_scope_lists))
    lanes_by_scope = {
        scope: tuple(scope_lanes) for scope, scope_lanes in lanes_by_scope_lists.items()
    }
    quarantine_hashes_by_scope: dict[tuple[str, str], list[str]] = {}
    for lane, record in quarantines.items():
        quarantine_hashes_by_scope.setdefault(
            (lane.pair, lane.horizon_lane), []
        ).append(str(record["quarantine_record_sha256"]))

    scope_rows: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    for scope in scopes:
        pair, horizon_lane = scope
        receipt = activations.get(scope)
        chain = (
            tuple(receipt["fallback_chain_profile_ids"]) if receipt is not None else ()
        )
        selected_profile_id = next(
            (
                profile_id
                for profile_id in chain
                if LaneKey(pair, profile_id, horizon_lane) not in quarantines
            ),
            None,
        )
        if selected_profile_id is None:
            scope_status = NO_PRIMARY
            selection_reason = (
                "ALL_SEALED_CANDIDATES_QUARANTINED"
                if receipt is not None
                else "NO_ACTIVATION_RECEIPT"
            )
        else:
            scope_status = PRIMARY_SELECTED
            selection_reason = (
                "ACTIVATED"
                if selected_profile_id == receipt["profile_id"]
                else "SEALED_FALLBACK"
            )

        scope_quarantines = sorted(quarantine_hashes_by_scope.get(scope, ()))
        scope_rows.append(
            {
                "pair": pair,
                "horizon_lane": horizon_lane,
                "status": scope_status,
                "requested_primary_profile_id": (
                    receipt["profile_id"] if receipt is not None else None
                ),
                "selected_primary_profile_id": selected_profile_id,
                "selection_reason": selection_reason,
                "activation_receipt_sha256": (
                    receipt["activation_receipt_sha256"]
                    if receipt is not None
                    else None
                ),
                "fallback_chain_profile_ids": list(chain),
                "fallback_chain_sha256": (
                    receipt["fallback_chain_sha256"] if receipt is not None else None
                ),
                "quarantine_record_sha256s": scope_quarantines,
                **_AUTHORITY_BODY,
            }
        )

        for lane in lanes_by_scope[scope]:
            profile = profiles_by_id[lane.profile_id]
            quarantine = quarantines.get(lane)
            role, lane_reason = _lane_role(
                profile=profile,
                profile_id=lane.profile_id,
                selected_profile_id=selected_profile_id,
                chain=chain,
                quarantined=quarantine is not None,
            )
            lane_rows.append(
                {
                    "pair": lane.pair,
                    "profile_id": lane.profile_id,
                    "profile_sha256": profile.profile_sha256,
                    "horizon_lane": lane.horizon_lane,
                    "role": role,
                    "deployment_role": (
                        PRIMARY_RESEARCH if role == PRIMARY_RESEARCH else SHADOW
                    ),
                    "research_primary_eligible": profile.primary_eligible,
                    "quarantine_status": (
                        QUARANTINED if quarantine is not None else "NOT_QUARANTINED"
                    ),
                    "selection_reason": lane_reason,
                    "shadow_enabled": True,
                    "quarantine_record_sha256": (
                        quarantine["quarantine_record_sha256"]
                        if quarantine is not None
                        else None
                    ),
                    **_AUTHORITY_BODY,
                }
            )

    state_binding = {
        "catalog_sha256": catalog.catalog_sha256,
        "activation_receipt_sha256s": sorted(
            str(receipt["activation_receipt_sha256"])
            for receipt in activations.values()
        ),
        "quarantine_record_sha256s": sorted(
            str(record["quarantine_record_sha256"]) for record in quarantines.values()
        ),
    }
    body = {
        "contract": PROFILE_ROUTING_CONTRACT,
        "catalog_sha256": catalog.catalog_sha256,
        "state_binding_sha256": canonical_sha256(state_binding),
        "scopes": scope_rows,
        "lanes": lane_rows,
        **_AUTHORITY_BODY,
    }
    return {**body, "routing_sha256": canonical_sha256(body)}


def _lane_role(
    *,
    profile: FastBotProfile,
    profile_id: str,
    selected_profile_id: str | None,
    chain: tuple[str, ...],
    quarantined: bool,
) -> tuple[str, str]:
    if quarantined:
        return QUARANTINED, "EXACT_LANE_QUARANTINE"
    if profile_id == selected_profile_id:
        reason = "ACTIVATED" if chain and profile_id == chain[0] else "SEALED_FALLBACK"
        return PRIMARY_RESEARCH, reason
    if profile_id in chain:
        return FALLBACK, "SEALED_FALLBACK_STANDBY"
    if not profile.primary_eligible:
        return SHADOW, "SHADOW_ONLY_PROFILE"
    return SHADOW, "NOT_SELECTED"


def _snapshot_like(value: object, expected: Any, *, label: str) -> Any:
    """Detach untrusted JSON while requiring the exact rebuilt shape/value."""

    if expected.__class__ is dict:
        if not isinstance(value, Mapping):
            raise ProfileContractError(f"{label} must be a mapping")
        try:
            snapshot = dict(value)
        except Exception as exc:
            raise ProfileContractError(f"{label} snapshot is unreadable") from exc
        if any(key.__class__ is not str for key in snapshot):
            raise ProfileContractError(f"{label} keys must be exact strings")
        if frozenset(snapshot) != frozenset(expected):
            raise ProfileContractError(f"{label} has non-canonical keys")
        return {
            key: _snapshot_like(snapshot[key], expected[key], label=f"{label}.{key}")
            for key in expected
        }
    if expected.__class__ is list:
        if value.__class__ is not list:
            raise ProfileContractError(f"{label} must be a JSON list")
        if len(value) != len(expected):
            raise ProfileContractError(f"{label} has non-canonical length")
        return [
            _snapshot_like(item, expected_item, label=f"{label}[]")
            for item, expected_item in zip(value, expected, strict=True)
        ]
    if value.__class__ is not expected.__class__ or value != expected:
        raise ProfileContractError(f"{label} does not match sealed state")
    return value
