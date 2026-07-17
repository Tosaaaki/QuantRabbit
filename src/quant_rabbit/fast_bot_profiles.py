"""Sealed, shadow-only profile primitives for the deterministic fast bot.

This module is intentionally disconnected from the runtime, gateway, broker,
and AI supervisor.  It defines a finite profile registry that later adapters
may consume, but neither a registry entry nor a research-primary designation
grants order or live authority.

Profiles and registries are immutable dataclasses.  Their JSON forms are
strictly canonical and content-addressed so aliases, reordered set-like
fields, unknown keys, and post-seal mutation fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


PROFILE_CONTRACT = "QR_FAST_BOT_PROFILE_V1"
PROFILE_CATALOG_CONTRACT = "QR_FAST_BOT_PROFILE_REGISTRY_V1"
LEGACY_PROFILE_ID = "FAST_BOT_LEGACY_FIXED_V1"
LEGACY_HORIZON_LANE = "M1_EXECUTION_15M_HOLD"

SHADOW_ONLY = "SHADOW_ONLY"
PRIMARY_ELIGIBLE = "PRIMARY_ELIGIBLE"
ACTIVATION_ELIGIBILITIES = frozenset({SHADOW_ONLY, PRIMARY_ELIGIBLE})

AI_ORDER_AUTHORITY = "NONE"
ORDER_AUTHORITY = "NONE"

# These bounds protect local artifact parsing and log surfaces.  They are
# protocol-size limits, not market parameters, and a later schema revision
# should replace them if longer versioned identifiers become necessary.
MAX_TOKEN_CHARS = 96
MAX_IMPLEMENTATION_REF_CHARS = 256
# These registry cardinality limits keep pure validation and routing artifacts
# bounded.  They are operational schema limits, not strategy-selection values;
# a larger finite experiment universe requires an explicit schema revision.
MAX_PROFILES = 64
MAX_HORIZON_LANES_PER_PROFILE = 16
MAX_FALLBACK_EDGES = 4096

_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_IMPLEMENTATION_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_PAIRS = frozenset(DEFAULT_TRADER_PAIRS)

_AUTHORITY_BODY: dict[str, object] = {
    "ai_order_authority": AI_ORDER_AUTHORITY,
    "order_authority": ORDER_AUTHORITY,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "shadow_only": True,
}
_PROFILE_KEYS = frozenset(
    {
        "contract",
        "profile_id",
        "evaluator_id",
        "implementation_ref",
        "supported_pairs",
        "horizon_lanes",
        "activation_eligibility",
        *_AUTHORITY_BODY,
        "profile_sha256",
    }
)
_FALLBACK_EDGE_KEYS = frozenset(
    {"pair", "horizon_lane", "from_profile_id", "to_profile_id"}
)
_CATALOG_KEYS = frozenset(
    {
        "contract",
        "profiles",
        "fallback_edges",
        *_AUTHORITY_BODY,
        "catalog_sha256",
    }
)


class ProfileContractError(ValueError):
    """Raised when a profile artifact is non-canonical or unsafe."""


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical JSON."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProfileContractError("value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class LaneKey:
    """Exact profile lane identity; no pair or token aliases are accepted."""

    pair: str
    profile_id: str
    horizon_lane: str

    def __post_init__(self) -> None:
        _require_pair(self.pair, label="lane pair")
        _require_token(self.profile_id, label="lane profile_id")
        _require_token(self.horizon_lane, label="lane horizon_lane")

    def canonical(self) -> str:
        """Return a lossless, parseable exact identity."""

        return f"{self.pair}:{self.profile_id}:{self.horizon_lane}"

    @classmethod
    def parse(cls, value: object) -> LaneKey:
        """Parse an exact canonical identity without normalization."""

        if value.__class__ is not str:
            raise ProfileContractError("lane identity must be an exact string")
        parts = value.split(":")
        # Three fields are the complete v1 identity.  Adding an axis requires
        # a new contract rather than silently accepting a lossy alias.
        if len(parts) != 3:
            raise ProfileContractError("lane identity must contain three fields")
        lane = cls(*parts)
        if lane.canonical() != value:
            raise ProfileContractError("lane identity is not canonical")
        return lane


@dataclass(frozen=True, slots=True)
class FastBotProfile:
    """One immutable evaluator identity and its finite supported lane set."""

    profile_id: str
    evaluator_id: str
    implementation_ref: str
    supported_pairs: tuple[str, ...]
    horizon_lanes: tuple[str, ...]
    activation_eligibility: str
    profile_sha256: str

    def __post_init__(self) -> None:
        _validate_profile_fields(self)

    @property
    def primary_eligible(self) -> bool:
        return self.activation_eligibility == PRIMARY_ELIGIBLE

    def supports(self, pair: str, horizon_lane: str) -> bool:
        return pair in self.supported_pairs and horizon_lane in self.horizon_lanes

    def to_dict(self) -> dict[str, Any]:
        body = _profile_body(
            profile_id=self.profile_id,
            evaluator_id=self.evaluator_id,
            implementation_ref=self.implementation_ref,
            supported_pairs=self.supported_pairs,
            horizon_lanes=self.horizon_lanes,
            activation_eligibility=self.activation_eligibility,
        )
        return {**body, "profile_sha256": self.profile_sha256}


@dataclass(frozen=True, slots=True, order=True)
class FallbackEdge:
    """One pre-sealed, same-pair and same-horizon fallback transition."""

    pair: str
    horizon_lane: str
    from_profile_id: str
    to_profile_id: str

    def __post_init__(self) -> None:
        _require_pair(self.pair, label="fallback pair")
        _require_token(self.horizon_lane, label="fallback horizon_lane")
        _require_token(self.from_profile_id, label="fallback from_profile_id")
        _require_token(self.to_profile_id, label="fallback to_profile_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "pair": self.pair,
            "horizon_lane": self.horizon_lane,
            "from_profile_id": self.from_profile_id,
            "to_profile_id": self.to_profile_id,
        }


@dataclass(frozen=True, slots=True)
class FastBotProfileCatalog:
    """Immutable, content-addressed profile and fallback registry."""

    profiles: tuple[FastBotProfile, ...]
    fallback_edges: tuple[FallbackEdge, ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        _validate_catalog_components(self.profiles, self.fallback_edges)
        if (
            tuple(sorted(self.profiles, key=lambda item: item.profile_id))
            != self.profiles
        ):
            raise ProfileContractError("catalog profiles are not canonical")
        if tuple(sorted(self.fallback_edges)) != self.fallback_edges:
            raise ProfileContractError("catalog fallback edges are not canonical")
        _require_sha(self.catalog_sha256, label="catalog_sha256")
        if canonical_sha256(self.body_dict()) != self.catalog_sha256:
            raise ProfileContractError("catalog_sha256 mismatch")

    def body_dict(self) -> dict[str, Any]:
        return {
            "contract": PROFILE_CATALOG_CONTRACT,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "fallback_edges": [edge.to_dict() for edge in self.fallback_edges],
            **_AUTHORITY_BODY,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body_dict(), "catalog_sha256": self.catalog_sha256}

    def profile(self, profile_id: str) -> FastBotProfile:
        _require_token(profile_id, label="profile_id")
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise ProfileContractError(f"unknown profile_id: {profile_id}")

    def lanes(self) -> tuple[LaneKey, ...]:
        return tuple(
            sorted(
                LaneKey(pair, profile.profile_id, horizon_lane)
                for profile in self.profiles
                for pair in profile.supported_pairs
                for horizon_lane in profile.horizon_lanes
            )
        )


def build_fast_bot_profile(
    *,
    profile_id: str,
    evaluator_id: str,
    implementation_ref: str,
    supported_pairs: Sequence[str],
    horizon_lanes: Sequence[str],
    activation_eligibility: str,
) -> FastBotProfile:
    """Build one detached canonical profile definition."""

    canonical_pairs = _canonical_string_set(
        supported_pairs,
        validator=lambda value: _require_pair(value, label="supported pair"),
        label="supported_pairs",
    )
    canonical_horizons = _canonical_string_set(
        horizon_lanes,
        validator=lambda value: _require_token(value, label="horizon_lane"),
        label="horizon_lanes",
    )
    body = _profile_body(
        profile_id=profile_id,
        evaluator_id=evaluator_id,
        implementation_ref=implementation_ref,
        supported_pairs=canonical_pairs,
        horizon_lanes=canonical_horizons,
        activation_eligibility=activation_eligibility,
    )
    return FastBotProfile(
        profile_id=profile_id,
        evaluator_id=evaluator_id,
        implementation_ref=implementation_ref,
        supported_pairs=canonical_pairs,
        horizon_lanes=canonical_horizons,
        activation_eligibility=activation_eligibility,
        profile_sha256=canonical_sha256(body),
    )


def build_fallback_edge(
    *,
    pair: str,
    horizon_lane: str,
    from_profile_id: str,
    to_profile_id: str,
) -> FallbackEdge:
    """Build an edge; catalog validation proves its endpoints and acyclicity."""

    return FallbackEdge(pair, horizon_lane, from_profile_id, to_profile_id)


def build_fast_bot_profile_catalog(
    profiles: Sequence[FastBotProfile | Mapping[str, Any]],
    fallback_edges: Sequence[FallbackEdge | Mapping[str, Any]] = (),
) -> FastBotProfileCatalog:
    """Canonicalize and seal a finite profile registry."""

    detached_profiles = tuple(
        sorted(
            (
                _coerce_profile(profile)
                for profile in _snapshot_sequence(profiles, "profiles")
            ),
            key=lambda item: item.profile_id,
        )
    )
    detached_edges = tuple(
        sorted(
            _coerce_fallback_edge(edge)
            for edge in _snapshot_sequence(fallback_edges, "fallback_edges")
        )
    )
    _validate_catalog_components(detached_profiles, detached_edges)
    body = {
        "contract": PROFILE_CATALOG_CONTRACT,
        "profiles": [profile.to_dict() for profile in detached_profiles],
        "fallback_edges": [edge.to_dict() for edge in detached_edges],
        **_AUTHORITY_BODY,
    }
    return FastBotProfileCatalog(
        profiles=detached_profiles,
        fallback_edges=detached_edges,
        catalog_sha256=canonical_sha256(body),
    )


def validate_fast_bot_profile_catalog(value: object) -> FastBotProfileCatalog:
    """Validate an external JSON-like registry and return an immutable copy."""

    if isinstance(value, FastBotProfileCatalog):
        # Round-tripping prevents a forged instance from bypassing the external
        # strict-schema validator while preserving an immutable return type.
        value = value.to_dict()
    snapshot = _snapshot_mapping(value, label="profile catalog")
    _require_exact_keys(snapshot, _CATALOG_KEYS, label="profile catalog")
    if (
        snapshot["contract"].__class__ is not str
        or snapshot["contract"] != PROFILE_CATALOG_CONTRACT
    ):
        raise ProfileContractError("profile catalog contract mismatch")
    _require_authority(snapshot, label="profile catalog")

    profiles_raw = _require_json_list(snapshot["profiles"], label="profiles")
    edges_raw = _require_json_list(snapshot["fallback_edges"], label="fallback_edges")
    profiles = tuple(_profile_from_mapping(item) for item in profiles_raw)
    edges = tuple(_fallback_edge_from_mapping(item) for item in edges_raw)
    if tuple(sorted(profiles, key=lambda item: item.profile_id)) != profiles:
        raise ProfileContractError("catalog profiles are not in canonical order")
    if tuple(sorted(edges)) != edges:
        raise ProfileContractError("fallback edges are not in canonical order")
    _validate_catalog_components(profiles, edges)

    stored_sha = snapshot["catalog_sha256"]
    if stored_sha.__class__ is not str or not _SHA256_RE.fullmatch(stored_sha):
        raise ProfileContractError("catalog_sha256 is invalid")
    body = {key: item for key, item in snapshot.items() if key != "catalog_sha256"}
    if canonical_sha256(body) != stored_sha:
        raise ProfileContractError("catalog_sha256 mismatch")
    return FastBotProfileCatalog(profiles, edges, stored_sha)


def fallback_chain_profile_ids(
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    lane: LaneKey,
) -> tuple[str, ...]:
    """Return the registry-sealed chain beginning at ``lane.profile_id``."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    return _fallback_chain_profile_ids_validated(catalog, lane)


def _fallback_chain_profile_ids_validated(
    catalog: FastBotProfileCatalog,
    lane: LaneKey,
) -> tuple[str, ...]:
    """Resolve a chain from an already validated immutable catalog.

    This helper exists so bounded batch validators do not reparse and rehash
    the complete catalog for every state row.  It is deliberately private;
    public callers must cross ``validate_fast_bot_profile_catalog`` above.
    """

    if catalog.__class__ is not FastBotProfileCatalog:
        raise ProfileContractError("catalog must be a validated profile catalog")
    if lane.__class__ is not LaneKey:
        raise ProfileContractError("lane must be an exact LaneKey")
    profile = catalog.profile(lane.profile_id)
    if not profile.supports(lane.pair, lane.horizon_lane):
        raise ProfileContractError("profile does not support the requested lane")

    outgoing = {
        (edge.pair, edge.horizon_lane, edge.from_profile_id): edge.to_profile_id
        for edge in catalog.fallback_edges
    }
    chain = [lane.profile_id]
    current = lane.profile_id
    while (lane.pair, lane.horizon_lane, current) in outgoing:
        current = outgoing[(lane.pair, lane.horizon_lane, current)]
        if current in chain:
            raise ProfileContractError("fallback cycle")
        chain.append(current)
    return tuple(chain)


def legacy_shadow_only_profile() -> FastBotProfile:
    """Represent the current fixed fast bot without making it primary eligible."""

    return build_fast_bot_profile(
        profile_id=LEGACY_PROFILE_ID,
        evaluator_id="FAST_BOT_LEGACY_FIXED_RULESET_V1",
        implementation_ref="quant_rabbit.fast_bot:build_fast_bot_shadow",
        supported_pairs=DEFAULT_TRADER_PAIRS,
        horizon_lanes=(LEGACY_HORIZON_LANE,),
        activation_eligibility=SHADOW_ONLY,
    )


def build_legacy_shadow_only_catalog() -> FastBotProfileCatalog:
    """Return the pre-validation catalog: legacy fixed logic, shadow only."""

    return build_fast_bot_profile_catalog((legacy_shadow_only_profile(),))


def _profile_body(
    *,
    profile_id: str,
    evaluator_id: str,
    implementation_ref: str,
    supported_pairs: Sequence[str],
    horizon_lanes: Sequence[str],
    activation_eligibility: str,
) -> dict[str, Any]:
    _require_token(profile_id, label="profile_id")
    _require_token(evaluator_id, label="evaluator_id")
    _require_implementation_ref(implementation_ref)
    activation_eligibility = _require_exact_string(
        activation_eligibility, label="activation_eligibility"
    )
    if activation_eligibility not in ACTIVATION_ELIGIBILITIES:
        raise ProfileContractError("unsupported activation_eligibility")
    if not supported_pairs:
        raise ProfileContractError("supported_pairs must not be empty")
    if not horizon_lanes:
        raise ProfileContractError("horizon_lanes must not be empty")
    return {
        "contract": PROFILE_CONTRACT,
        "profile_id": profile_id,
        "evaluator_id": evaluator_id,
        "implementation_ref": implementation_ref,
        "supported_pairs": list(supported_pairs),
        "horizon_lanes": list(horizon_lanes),
        "activation_eligibility": activation_eligibility,
        **_AUTHORITY_BODY,
    }


def _profile_from_mapping(value: object) -> FastBotProfile:
    snapshot = _snapshot_mapping(value, label="profile")
    _require_exact_keys(snapshot, _PROFILE_KEYS, label="profile")
    if (
        snapshot["contract"].__class__ is not str
        or snapshot["contract"] != PROFILE_CONTRACT
    ):
        raise ProfileContractError("profile contract mismatch")
    _require_authority(snapshot, label="profile")
    pairs_raw = _require_json_list(snapshot["supported_pairs"], label="supported_pairs")
    horizons_raw = _require_json_list(snapshot["horizon_lanes"], label="horizon_lanes")
    pairs = tuple(pairs_raw)
    horizons = tuple(horizons_raw)
    _require_canonical_external_string_set(
        pairs,
        validator=lambda item: _require_pair(item, label="supported pair"),
        label="supported_pairs",
    )
    _require_canonical_external_string_set(
        horizons,
        validator=lambda item: _require_token(item, label="horizon_lane"),
        label="horizon_lanes",
    )
    profile = FastBotProfile(
        profile_id=_require_token(snapshot["profile_id"], label="profile_id"),
        evaluator_id=_require_token(snapshot["evaluator_id"], label="evaluator_id"),
        implementation_ref=_require_implementation_ref(snapshot["implementation_ref"]),
        supported_pairs=pairs,
        horizon_lanes=horizons,
        activation_eligibility=_require_exact_string(
            snapshot["activation_eligibility"], label="activation_eligibility"
        ),
        profile_sha256=_require_sha(snapshot["profile_sha256"], label="profile_sha256"),
    )
    return profile


def _coerce_profile(value: FastBotProfile | Mapping[str, Any]) -> FastBotProfile:
    if isinstance(value, FastBotProfile):
        return _profile_from_mapping(value.to_dict())
    return _profile_from_mapping(value)


def _coerce_fallback_edge(value: FallbackEdge | Mapping[str, Any]) -> FallbackEdge:
    if isinstance(value, FallbackEdge):
        return _fallback_edge_from_mapping(value.to_dict())
    return _fallback_edge_from_mapping(value)


def _fallback_edge_from_mapping(value: object) -> FallbackEdge:
    snapshot = _snapshot_mapping(value, label="fallback edge")
    _require_exact_keys(snapshot, _FALLBACK_EDGE_KEYS, label="fallback edge")
    return FallbackEdge(
        pair=snapshot["pair"],
        horizon_lane=snapshot["horizon_lane"],
        from_profile_id=snapshot["from_profile_id"],
        to_profile_id=snapshot["to_profile_id"],
    )


def _validate_profile_fields(profile: FastBotProfile) -> None:
    if profile.supported_pairs.__class__ is not tuple:
        raise ProfileContractError("supported_pairs must be an immutable tuple")
    if profile.horizon_lanes.__class__ is not tuple:
        raise ProfileContractError("horizon_lanes must be an immutable tuple")
    if len(profile.horizon_lanes) > MAX_HORIZON_LANES_PER_PROFILE:
        raise ProfileContractError("profile has too many horizon lanes")
    body = _profile_body(
        profile_id=profile.profile_id,
        evaluator_id=profile.evaluator_id,
        implementation_ref=profile.implementation_ref,
        supported_pairs=profile.supported_pairs,
        horizon_lanes=profile.horizon_lanes,
        activation_eligibility=profile.activation_eligibility,
    )
    _require_canonical_external_string_set(
        profile.supported_pairs,
        validator=lambda item: _require_pair(item, label="supported pair"),
        label="supported_pairs",
    )
    _require_canonical_external_string_set(
        profile.horizon_lanes,
        validator=lambda item: _require_token(item, label="horizon_lane"),
        label="horizon_lanes",
    )
    _require_sha(profile.profile_sha256, label="profile_sha256")
    if canonical_sha256(body) != profile.profile_sha256:
        raise ProfileContractError("profile_sha256 mismatch")


def _validate_catalog_components(
    profiles: tuple[FastBotProfile, ...],
    edges: tuple[FallbackEdge, ...],
) -> None:
    if profiles.__class__ is not tuple or edges.__class__ is not tuple:
        raise ProfileContractError("catalog collections must be immutable tuples")
    if not profiles:
        raise ProfileContractError("profile catalog must not be empty")
    if len(profiles) > MAX_PROFILES:
        raise ProfileContractError("profile catalog exceeds its bounded capacity")
    if len(edges) > MAX_FALLBACK_EDGES:
        raise ProfileContractError("fallback registry exceeds its bounded capacity")
    by_id: dict[str, FastBotProfile] = {}
    for profile in profiles:
        if profile.profile_id in by_id:
            raise ProfileContractError(f"duplicate profile_id: {profile.profile_id}")
        by_id[profile.profile_id] = profile

    outgoing: dict[tuple[str, str, str], str] = {}
    for edge in edges:
        source = by_id.get(edge.from_profile_id)
        target = by_id.get(edge.to_profile_id)
        if source is None or target is None:
            raise ProfileContractError("fallback edge references an unknown profile")
        if not source.primary_eligible or not target.primary_eligible:
            raise ProfileContractError("fallback edge profile is not primary eligible")
        if not source.supports(edge.pair, edge.horizon_lane):
            raise ProfileContractError("fallback source does not support its lane")
        if not target.supports(edge.pair, edge.horizon_lane):
            raise ProfileContractError("fallback target does not support its lane")
        key = (edge.pair, edge.horizon_lane, edge.from_profile_id)
        if key in outgoing:
            raise ProfileContractError("duplicate fallback source")
        outgoing[key] = edge.to_profile_id

    for pair, horizon_lane, starting_profile_id in tuple(outgoing):
        visited: set[str] = set()
        current = starting_profile_id
        while (pair, horizon_lane, current) in outgoing:
            if current in visited:
                raise ProfileContractError("fallback cycle")
            visited.add(current)
            current = outgoing[(pair, horizon_lane, current)]
        if current in visited:
            raise ProfileContractError("fallback cycle")


def _snapshot_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileContractError(f"{label} must be a mapping")
    try:
        snapshot = dict(value)
    except Exception as exc:
        raise ProfileContractError(f"{label} snapshot is unreadable") from exc
    if any(key.__class__ is not str for key in snapshot):
        raise ProfileContractError(f"{label} keys must be exact strings")
    return snapshot


def _snapshot_sequence(value: object, label: str) -> tuple[Any, ...]:
    if value.__class__ not in {list, tuple}:
        raise ProfileContractError(f"{label} must be a list or tuple")
    try:
        return tuple(value)
    except Exception as exc:
        raise ProfileContractError(f"{label} snapshot is unreadable") from exc


def _require_json_list(value: object, *, label: str) -> list[Any]:
    if value.__class__ is not list:
        raise ProfileContractError(f"{label} must be a JSON list")
    return list(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if frozenset(value) != expected:
        raise ProfileContractError(f"{label} has non-canonical keys")


def _require_authority(value: Mapping[str, Any], *, label: str) -> None:
    for key, expected in _AUTHORITY_BODY.items():
        actual = value.get(key)
        if actual.__class__ is not expected.__class__ or actual != expected:
            raise ProfileContractError(f"{label} has unsafe {key}")


def _require_exact_string(value: object, *, label: str) -> str:
    if value.__class__ is not str or not value:
        raise ProfileContractError(f"{label} must be a non-empty exact string")
    return value


def _require_token(value: object, *, label: str) -> str:
    token = _require_exact_string(value, label=label)
    if len(token) > MAX_TOKEN_CHARS or not _TOKEN_RE.fullmatch(token):
        raise ProfileContractError(f"{label} is not canonical")
    return token


def _require_pair(value: object, *, label: str) -> str:
    pair = _require_exact_string(value, label=label)
    if pair not in _KNOWN_PAIRS:
        raise ProfileContractError(f"{label} is not a canonical configured pair")
    return pair


def _require_implementation_ref(value: object) -> str:
    ref = _require_exact_string(value, label="implementation_ref")
    if len(ref) > MAX_IMPLEMENTATION_REF_CHARS or not _IMPLEMENTATION_REF_RE.fullmatch(
        ref
    ):
        raise ProfileContractError("implementation_ref is not canonical")
    return ref


def _require_sha(value: object, *, label: str) -> str:
    if value.__class__ is not str or not _SHA256_RE.fullmatch(value):
        raise ProfileContractError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_string_set(
    value: object,
    *,
    validator: Any,
    label: str,
) -> tuple[str, ...]:
    snapshot = _snapshot_sequence(value, label)
    if not snapshot:
        raise ProfileContractError(f"{label} must not be empty")
    validated = tuple(validator(item) for item in snapshot)
    if len(set(validated)) != len(validated):
        raise ProfileContractError(f"{label} contains duplicates")
    return tuple(sorted(validated))


def _require_canonical_external_string_set(
    value: tuple[str, ...],
    *,
    validator: Any,
    label: str,
) -> None:
    if not value:
        raise ProfileContractError(f"{label} must not be empty")
    validated = tuple(validator(item) for item in value)
    if len(set(validated)) != len(validated):
        raise ProfileContractError(f"{label} contains duplicates")
    if tuple(sorted(validated)) != validated:
        raise ProfileContractError(f"{label} is not in canonical order")
