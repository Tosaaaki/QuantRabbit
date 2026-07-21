"""Fail-closed DOJO strategy-catalog revision for Asia sweep/reclaim research.

The original ``QR_DOJO_BOT_CATALOG_V1`` remains byte- and behavior-stable.
This revision adds exactly one M5-only family and reuses the legacy catalog's
reviewed common risk/exit bounds by validating a ``burst``-shaped surrogate.
The family is deliberately unavailable to the V1 proposal sealer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    CATALOG_CONTRACT as LEGACY_CATALOG_CONTRACT,
    DojoBotCatalogError,
    catalog_manifest as legacy_catalog_manifest,
    validate_bot_config as validate_legacy_bot_config,
)


CATALOG_CONTRACT: Final = "QR_DOJO_STRATEGY_CATALOG_REVISION_V2"
PROPOSAL_CONTRACT: Final = "QR_DOJO_CANDIDATE_PROPOSAL_V2"
SCHEMA_VERSION: Final = 2
FAMILY: Final = "asia_sweep_reclaim_be"
TIMEFRAME: Final = "M5"
ASIA_RANGE_POLICY: Final = "LONDON_0000_0800_EXACT_CLOSED_M5"
ENTRY_POLICY: Final = "NEXT_M5_OPEN_AFTER_SWEEP_CLOSE_RECLAIM"

_IDENTIFIER_RE: Final = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_RAW_PROPOSAL_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "candidate_id",
        "family",
        "hypothesis",
        "config",
        "risk_increase",
    }
)
_SEALED_PROPOSAL_KEYS: Final = _RAW_PROPOSAL_KEYS | frozenset(
    {
        "config_sha256",
        "catalog_contract",
        "catalog_sha256",
        "proposal_sha256",
    }
)


class DojoStrategyCatalogRevisionV2Error(ValueError):
    """The Asia sweep/reclaim config or proposal is outside revision V2."""


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
        raise DojoStrategyCatalogRevisionV2Error(
            "strategy revision value is not strict JSON"
        ) from exc


def _clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def validate_asia_sweep_reclaim_be_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one V2 family against legacy bounds plus fixed BE policy."""

    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise DojoStrategyCatalogRevisionV2Error("strategy config must be an object")
    if config.get("signal") != FAMILY:
        raise DojoStrategyCatalogRevisionV2Error(
            "strategy revision V2 accepts only asia_sweep_reclaim_be"
        )
    if config.get("exit_policy") != "BREAKEVEN":
        raise DojoStrategyCatalogRevisionV2Error(
            "asia_sweep_reclaim_be requires the reviewed BREAKEVEN overlay"
        )
    surrogate = dict(config)
    surrogate["signal"] = "burst"
    try:
        normalized = validate_legacy_bot_config(surrogate)
    except DojoBotCatalogError as exc:
        raise DojoStrategyCatalogRevisionV2Error(
            "asia_sweep_reclaim_be config violates reviewed common bounds"
        ) from exc
    normalized["signal"] = FAMILY
    if normalized["sl_pips"] is None:
        raise DojoStrategyCatalogRevisionV2Error(
            "asia_sweep_reclaim_be requires a finite initial stop"
        )
    if normalized["exit_policy"] != "BREAKEVEN" or any(
        normalized[key] != value for key, value in AUTHORITY_INVARIANTS.items()
    ):
        raise DojoStrategyCatalogRevisionV2Error(
            "strategy exit or authority invariant drifted"
        )
    return normalized


def strategy_config_sha256(config: Mapping[str, Any]) -> str:
    return _sha256(validate_asia_sweep_reclaim_be_config(config))


def catalog_manifest() -> dict[str, Any]:
    legacy = legacy_catalog_manifest()
    return {
        "contract": CATALOG_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "base_catalog_contract": LEGACY_CATALOG_CONTRACT,
        "base_catalog_sha256": _sha256(legacy),
        "supported_families": [FAMILY],
        "family_contract": {
            "family": FAMILY,
            "timeframe": TIMEFRAME,
            "asia_range_policy": ASIA_RANGE_POLICY,
            "reclaim_policy": "STRICT_SWEEP_AND_STRICT_CLOSE_BACK_INSIDE",
            "entry_policy": ENTRY_POLICY,
            "one_attempt_per_london_day": True,
            "exit_policy": "BREAKEVEN",
            "uses_legacy_common_bounds": True,
            "lookahead_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        },
    }


def _proposal_body(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("contract") != PROPOSAL_CONTRACT or value.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise DojoStrategyCatalogRevisionV2Error(
            "strategy proposal contract/version is unsupported"
        )
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or _IDENTIFIER_RE.fullmatch(candidate_id) is None:
        raise DojoStrategyCatalogRevisionV2Error("candidate_id is invalid")
    hypothesis = value.get("hypothesis")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or hypothesis != hypothesis.strip()
        or len(hypothesis) > 1_000
    ):
        raise DojoStrategyCatalogRevisionV2Error("candidate hypothesis is invalid")
    if value.get("family") != FAMILY:
        raise DojoStrategyCatalogRevisionV2Error("candidate family is invalid")
    if value.get("risk_increase") is not False:
        raise DojoStrategyCatalogRevisionV2Error(
            "strategy revision candidate may not increase risk"
        )
    config = validate_asia_sweep_reclaim_be_config(value.get("config"))
    manifest = catalog_manifest()
    return {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "family": FAMILY,
        "hypothesis": hypothesis,
        "config": config,
        "config_sha256": _sha256(config),
        "catalog_contract": CATALOG_CONTRACT,
        "catalog_sha256": _sha256(manifest),
        "risk_increase": False,
    }


def seal_candidate_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Seal or exactly revalidate one revision-V2 candidate proposal."""

    if not isinstance(proposal, Mapping) or any(
        not isinstance(key, str) for key in proposal
    ):
        raise DojoStrategyCatalogRevisionV2Error("candidate proposal must be an object")
    row = _clone(dict(proposal))
    keys = frozenset(row)
    if keys == _RAW_PROPOSAL_KEYS:
        body = _proposal_body(row)
        return {**body, "proposal_sha256": _sha256(body)}
    if keys != _SEALED_PROPOSAL_KEYS:
        raise DojoStrategyCatalogRevisionV2Error(
            "candidate proposal schema is not exact"
        )
    body = _proposal_body(row)
    expected = {**body, "proposal_sha256": _sha256(body)}
    claimed = row.get("proposal_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise DojoStrategyCatalogRevisionV2Error("proposal SHA-256 is invalid")
    if row != expected:
        raise DojoStrategyCatalogRevisionV2Error(
            "sealed strategy proposal differs from revision V2"
        )
    return expected


__all__ = [
    "ASIA_RANGE_POLICY",
    "CATALOG_CONTRACT",
    "DojoStrategyCatalogRevisionV2Error",
    "ENTRY_POLICY",
    "FAMILY",
    "PROPOSAL_CONTRACT",
    "SCHEMA_VERSION",
    "TIMEFRAME",
    "catalog_manifest",
    "seal_candidate_proposal",
    "strategy_config_sha256",
    "validate_asia_sweep_reclaim_be_config",
]
