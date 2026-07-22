"""Bounded session-open range-break catalog revision for DOJO research.

The original :mod:`quant_rabbit.dojo_bot_catalog` deliberately remains
byte- and behavior-stable because the sealed G2 V1 registry binds its catalog
digest.  This explicit opt-in revision accepts the existing
``session_open_range_break`` thesis only when it also declares a finite
``max_initial_stop_pips``.  The tuned runtime must HOLD whenever the rounded
atomic initial stop would exceed that bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import (
    CATALOG_CONTRACT as LEGACY_CATALOG_CONTRACT,
    DojoBotCatalogError,
    bot_risk_policy_manifest,
    bot_risk_policy_sha256,
    catalog_manifest as legacy_catalog_manifest,
    validate_bot_config as validate_legacy_bot_config,
)


CATALOG_CONTRACT: Final = "QR_DOJO_STRATEGY_CATALOG_REVISION_V3"
PROPOSAL_CONTRACT: Final = "QR_DOJO_CANDIDATE_PROPOSAL_V3"
RISK_VECTOR_CONTRACT: Final = "QR_DOJO_SESSION_RANGE_RISK_VECTOR_V1"
SCHEMA_VERSION: Final = 3
FAMILY: Final = "session_open_range_break"
MAX_INITIAL_STOP_PIPS_KEY: Final = "max_initial_stop_pips"

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


class DojoStrategyCatalogRevisionV3Error(ValueError):
    """The bounded session-range config or proposal is outside revision V3."""


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
        raise DojoStrategyCatalogRevisionV3Error(
            "strategy revision value is not strict JSON"
        ) from exc


def _clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise DojoStrategyCatalogRevisionV3Error(f"{field} must be finite")
    number = float(value)
    if positive and number <= 0:
        raise DojoStrategyCatalogRevisionV3Error(f"{field} must be positive")
    return number


def validate_session_open_range_break_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one opt-in session config with a finite atomic stop cap."""

    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise DojoStrategyCatalogRevisionV3Error("strategy config must be an object")
    if config.get("signal") != FAMILY:
        raise DojoStrategyCatalogRevisionV3Error(
            "strategy revision V3 accepts only session_open_range_break"
        )
    if MAX_INITIAL_STOP_PIPS_KEY not in config:
        raise DojoStrategyCatalogRevisionV3Error(
            "session_open_range_break requires max_initial_stop_pips"
        )
    policy = bot_risk_policy_manifest()
    hard_cap = float(policy["hard_envelope"]["max_initial_sl_pips"])
    stop_cap = _finite_number(
        config[MAX_INITIAL_STOP_PIPS_KEY],
        field=MAX_INITIAL_STOP_PIPS_KEY,
        positive=True,
    )
    if stop_cap > hard_cap:
        raise DojoStrategyCatalogRevisionV3Error(
            "max_initial_stop_pips exceeds the repo-owned hard envelope"
        )
    surrogate = dict(config)
    del surrogate[MAX_INITIAL_STOP_PIPS_KEY]
    try:
        normalized = validate_legacy_bot_config(surrogate)
    except DojoBotCatalogError as exc:
        raise DojoStrategyCatalogRevisionV3Error(
            "bounded session config violates reviewed legacy bounds"
        ) from exc
    if normalized["signal"] != FAMILY or normalized["sl_pips"] is not None:
        raise DojoStrategyCatalogRevisionV3Error(
            "bounded session config must retain the dynamic-stop family shape"
        )
    normalized[MAX_INITIAL_STOP_PIPS_KEY] = stop_cap
    return normalized


def strategy_config_sha256(config: Mapping[str, Any]) -> str:
    return _sha256(validate_session_open_range_break_config(config))


def catalog_manifest() -> dict[str, Any]:
    legacy = legacy_catalog_manifest()
    policy = bot_risk_policy_manifest()
    return {
        "contract": CATALOG_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "base_catalog_contract": LEGACY_CATALOG_CONTRACT,
        "base_catalog_sha256": _sha256(legacy),
        "base_risk_policy_sha256": bot_risk_policy_sha256(),
        "supported_families": [FAMILY],
        "family_contract": {
            "family": FAMILY,
            "range_policy": "LONDON_0000_0800_EXACT_CLOSED_BARS",
            "entry_policy": "POST_RANGE_BREAK_MARKET_AT_CURRENT_CAUSAL_SNAPSHOT",
            "dynamic_target": True,
            "dynamic_initial_stop": True,
            "max_initial_stop_pips_required": True,
            "max_initial_stop_pips_hard_max": policy["hard_envelope"][
                "max_initial_sl_pips"
            ],
            "atomic_rounded_stop_must_not_exceed_bound": True,
            "bound_exceeded_action": "HOLD",
            "one_attempt_per_london_day": True,
            "uses_legacy_common_bounds": True,
            "lookahead_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        },
    }


def session_open_range_break_risk_vector(
    config: Mapping[str, Any],
    *,
    stress_slippage_pips_per_fill: float,
) -> dict[str, Any]:
    """Classify the bounded dynamic stop against the repo-owned hard policy."""

    normalized = validate_session_open_range_break_config(config)
    stress = _finite_number(
        stress_slippage_pips_per_fill,
        field="stress_slippage_pips_per_fill",
    )
    if stress < 0:
        raise DojoStrategyCatalogRevisionV3Error(
            "stress_slippage_pips_per_fill must be non-negative"
        )
    policy = bot_risk_policy_manifest()
    envelope = policy["hard_envelope"]
    leverage = float(normalized["per_pos_lev"])
    pair_cap = int(normalized["max_concurrent_per_pair"])
    global_cap = int(normalized["global_max_concurrent"])
    global_gross = leverage * global_cap
    stop_bound = float(normalized[MAX_INITIAL_STOP_PIPS_KEY])
    single_stop_risk = leverage * (stop_bound + stress)
    gross_stop_risk = single_stop_risk * global_cap

    blockers: list[str] = []
    if leverage > float(envelope["max_per_position_leverage"]) + 1e-12:
        blockers.append("PER_POSITION_LEVERAGE_HARD_CAP_EXCEEDED")
    if pair_cap > int(envelope["max_concurrent_per_pair"]):
        blockers.append("PAIR_CONCURRENCY_HARD_CAP_EXCEEDED")
    if global_cap > int(envelope["max_global_concurrent"]):
        blockers.append("GLOBAL_CONCURRENCY_HARD_CAP_EXCEEDED")
    if global_gross > float(envelope["max_global_gross_leverage"]) + 1e-12:
        blockers.append("GLOBAL_GROSS_LEVERAGE_HARD_CAP_EXCEEDED")
    if stop_bound > float(envelope["max_initial_sl_pips"]) + 1e-12:
        blockers.append("INITIAL_SL_PIPS_HARD_CAP_EXCEEDED")
    if single_stop_risk > float(envelope["max_single_stop_risk_index"]) + 1e-12:
        blockers.append("SINGLE_STOP_RISK_INDEX_HARD_CAP_EXCEEDED")
    if gross_stop_risk > float(envelope["max_gross_stop_risk_index"]) + 1e-12:
        blockers.append("GROSS_STOP_RISK_INDEX_HARD_CAP_EXCEEDED")

    body = {
        "contract": RISK_VECTOR_CONTRACT,
        "schema_version": 1,
        "catalog_contract": CATALOG_CONTRACT,
        "catalog_sha256": _sha256(catalog_manifest()),
        "config_sha256": strategy_config_sha256(normalized),
        "base_risk_policy_sha256": bot_risk_policy_sha256(),
        "signal": FAMILY,
        "stress_slippage_pips_per_fill": stress,
        "per_position_leverage": leverage,
        "max_concurrent_per_pair": pair_cap,
        "max_global_concurrent": global_cap,
        "pair_gross_leverage": leverage * pair_cap,
        "global_gross_leverage": global_gross,
        "initial_stop_bound_kind": "DYNAMIC_CONFIG_MAX_ATOMIC_ROUNDED_PIPS",
        "initial_sl_pips": None,
        "max_initial_stop_pips": stop_bound,
        "single_stop_risk_index": single_stop_risk,
        "gross_stop_risk_index": gross_stop_risk,
        "hard_envelope_passed": not blockers,
        "rankable": not blockers,
        "blocker_codes": blockers,
        "absolute_nav_loss_bound_claimed": False,
        "runner_market_receipt_required": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "risk_vector_sha256": _sha256(body)}


def _proposal_body(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("contract") != PROPOSAL_CONTRACT
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise DojoStrategyCatalogRevisionV3Error(
            "strategy proposal contract/version is unsupported"
        )
    candidate_id = value.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or _IDENTIFIER_RE.fullmatch(candidate_id) is None
    ):
        raise DojoStrategyCatalogRevisionV3Error("candidate_id is invalid")
    hypothesis = value.get("hypothesis")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or hypothesis != hypothesis.strip()
        or len(hypothesis) > 1_000
    ):
        raise DojoStrategyCatalogRevisionV3Error("candidate hypothesis is invalid")
    if value.get("family") != FAMILY:
        raise DojoStrategyCatalogRevisionV3Error("candidate family is invalid")
    if value.get("risk_increase") is not False:
        raise DojoStrategyCatalogRevisionV3Error(
            "strategy revision candidate may not increase risk"
        )
    config = validate_session_open_range_break_config(value.get("config"))
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
    """Seal or exactly revalidate one bounded session candidate proposal."""

    if not isinstance(proposal, Mapping) or any(
        not isinstance(key, str) for key in proposal
    ):
        raise DojoStrategyCatalogRevisionV3Error("candidate proposal must be an object")
    row = _clone(dict(proposal))
    keys = frozenset(row)
    if keys == _RAW_PROPOSAL_KEYS:
        body = _proposal_body(row)
        return {**body, "proposal_sha256": _sha256(body)}
    if keys != _SEALED_PROPOSAL_KEYS:
        raise DojoStrategyCatalogRevisionV3Error(
            "candidate proposal schema is not exact"
        )
    body = _proposal_body(row)
    expected = {**body, "proposal_sha256": _sha256(body)}
    claimed = row.get("proposal_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise DojoStrategyCatalogRevisionV3Error("proposal SHA-256 is invalid")
    if row != expected:
        raise DojoStrategyCatalogRevisionV3Error(
            "sealed strategy proposal differs from revision V3"
        )
    return expected


__all__ = [
    "CATALOG_CONTRACT",
    "DojoStrategyCatalogRevisionV3Error",
    "FAMILY",
    "MAX_INITIAL_STOP_PIPS_KEY",
    "PROPOSAL_CONTRACT",
    "RISK_VECTOR_CONTRACT",
    "SCHEMA_VERSION",
    "catalog_manifest",
    "seal_candidate_proposal",
    "session_open_range_break_risk_vector",
    "strategy_config_sha256",
    "validate_session_open_range_break_config",
]
