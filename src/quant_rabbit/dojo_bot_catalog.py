"""Strict declarative configuration catalog for DOJO research bots.

The catalog is deliberately narrower than :mod:`bots.lab_bot`: a trainer may
select a reviewed family and tune only that family's declared parameters.  It
may not inject Python, assign itself an owner, or smuggle live authority into a
configuration.  Normalized configurations always carry explicit research-only
authority fields and are safe to content-address before replay.

The numeric limits below are research-search bounds, not production sizing or
risk settings.  A study may choose a tighter preregistered range, but it may
not widen these bounds without a reviewed catalog revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


CATALOG_CONTRACT = "QR_DOJO_BOT_CATALOG_V1"
RISK_POLICY_CONTRACT = "QR_DOJO_BOT_RISK_POLICY_V1"
RISK_VECTOR_CONTRACT = "QR_DOJO_BOT_RISK_VECTOR_V1"
ORDER_AUTHORITY = "NONE"

EXISTING_FAMILIES = frozenset(
    {
        "burst",
        "pullback_limit",
        "compression_break",
        "spike_fade",
        "prev_day_extreme_fade",
        "round_number_fade",
        "daily_break_pullback",
        "mean_revert_24h",
        "fade_ladder",
        "range_fade_limit",
    }
)
NEW_CANDIDATE_FAMILIES = frozenset(
    {
        "session_open_range_break",
        "weekend_gap_recovery",
    }
)
SUPPORTED_FAMILIES = EXISTING_FAMILIES | NEW_CANDIDATE_FAMILIES

AUTHORITY_INVARIANTS = MappingProxyType(
    {
        "order_authority": ORDER_AUTHORITY,
        "live_permission": False,
        "external_broker_mutation_allowed": False,
    }
)

EXIT_POLICIES = frozenset({"FIXED", "BREAKEVEN", "ATR_TRAILING"})

_PAIR_RE = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")
_MAX_PAIRS = 64

# These are rank-admission limits, not the wider catalog search bounds above.
# A configuration may remain valid research input while being mechanically
# unrankable under this independently versioned policy.
_RISK_MAX_PER_POSITION_LEVERAGE = 5.0
_RISK_MAX_CONCURRENT_PER_PAIR = 2
_RISK_MAX_GLOBAL_CONCURRENT = 4
_RISK_MAX_GLOBAL_GROSS_LEVERAGE = 20.0
_RISK_MAX_INITIAL_SL_PIPS = 25.0
_RISK_MAX_SINGLE_STOP_RISK_INDEX = 126.5
_RISK_MAX_GROSS_STOP_RISK_INDEX = 506.0


class DojoBotCatalogError(ValueError):
    """A proposed bot configuration is outside the reviewed catalog."""


@dataclass(frozen=True)
class _NumberRule:
    minimum: float
    maximum: float
    minimum_inclusive: bool = True
    integer: bool = False


_COMMON_NUMBER_RULES = MappingProxyType(
    {
        "tp_pips": _NumberRule(0.0, 1_000.0, minimum_inclusive=False),
        "tp_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False),
        "sl_pips": _NumberRule(0.0, 2_000.0, minimum_inclusive=False),
        "ceiling_min": _NumberRule(1.0, 10_080.0, integer=True),
        "max_concurrent": _NumberRule(1.0, 16.0, integer=True),
        "max_concurrent_per_pair": _NumberRule(1.0, 16.0, integer=True),
        "global_max_concurrent": _NumberRule(1.0, 128.0, integer=True),
        "per_pos_lev": _NumberRule(0.0, 25.0, minimum_inclusive=False),
        "atr_floor_pips": _NumberRule(0.0, 100.0, minimum_inclusive=False),
    }
)

_EXIT_POLICY_NUMBER_RULES = MappingProxyType(
    {
        "BREAKEVEN": MappingProxyType(
            {
                "be_trigger_atr": _NumberRule(0.5, 5.0),
                "be_offset_pips": _NumberRule(0.0, 2.0),
            }
        ),
        "ATR_TRAILING": MappingProxyType(
            {
                "trail_trigger_atr": _NumberRule(0.5, 5.0),
                "trail_distance_atr": _NumberRule(0.5, 5.0),
            }
        ),
        "FIXED": MappingProxyType({}),
    }
)
_ALL_EXIT_PARAMETER_KEYS = frozenset(
    key for rules in _EXIT_POLICY_NUMBER_RULES.values() for key in rules
)

_FAMILY_NUMBER_RULES = MappingProxyType(
    {
        "burst": MappingProxyType({}),
        "pullback_limit": MappingProxyType(
            {"pull_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False)}
        ),
        "compression_break": MappingProxyType({}),
        "spike_fade": MappingProxyType({}),
        "prev_day_extreme_fade": MappingProxyType({}),
        "round_number_fade": MappingProxyType({}),
        "daily_break_pullback": MappingProxyType({}),
        "mean_revert_24h": MappingProxyType(
            {"fade_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False)}
        ),
        "fade_ladder": MappingProxyType(
            {
                "fade_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False),
                "eff_max": _NumberRule(0.0, 1.0, minimum_inclusive=False),
            }
        ),
        "range_fade_limit": MappingProxyType(
            {
                "fade_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False),
                "eff_max": _NumberRule(0.0, 1.0, minimum_inclusive=False),
            }
        ),
        "session_open_range_break": MappingProxyType(
            {
                "session_buffer_atr": _NumberRule(0.0, 5.0),
                "session_tp_range": _NumberRule(0.0, 20.0, minimum_inclusive=False),
                "session_sl_range": _NumberRule(0.0, 20.0, minimum_inclusive=False),
            }
        ),
        "weekend_gap_recovery": MappingProxyType(
            {
                "weekend_gap_atr": _NumberRule(0.0, 20.0, minimum_inclusive=False),
                "weekend_sl_gap": _NumberRule(0.0, 20.0, minimum_inclusive=False),
                "weekend_wait_bars": _NumberRule(1.0, 1_440.0, integer=True),
                "weekend_spread_fraction": _NumberRule(
                    0.0, 1.0, minimum_inclusive=False
                ),
            }
        ),
    }
)

_COMMON_INPUT_KEYS = frozenset(
    {
        "signal",
        "pairs",
        "tp_pips",
        "tp_atr",
        "sl_pips",
        "ceiling_min",
        "max_concurrent",
        "max_concurrent_per_pair",
        "global_max_concurrent",
        "per_pos_lev",
        "atr_floor_pips",
        "exit_policy",
        *_ALL_EXIT_PARAMETER_KEYS,
        *AUTHORITY_INVARIANTS,
    }
)
_COMMON_REQUIRED_KEYS = frozenset(
    {
        "signal",
        "pairs",
        "sl_pips",
        "ceiling_min",
        "global_max_concurrent",
        "per_pos_lev",
        "atr_floor_pips",
    }
)


def validate_bot_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one declarative DOJO bot configuration.

    Existing families require exactly one take-profit basis.  The session and
    weekend families derive their targets and stops internally, so both common
    TP fields must be omitted or null and ``sl_pips`` must be null.
    ``max_concurrent`` remains an accepted legacy spelling, but the normalized
    result always uses
    ``max_concurrent_per_pair``.  Pair order is normalized because the replay
    contract treats the configured instruments as a set and orders them at its
    own synchronized quote boundary.
    """

    if not isinstance(config, Mapping):
        raise DojoBotCatalogError("bot config must be a JSON object")
    if any(not isinstance(key, str) for key in config):
        raise DojoBotCatalogError("bot config keys must be strings")

    signal = config.get("signal")
    if not isinstance(signal, str) or signal not in SUPPORTED_FAMILIES:
        raise DojoBotCatalogError("signal is not a reviewed DOJO bot family")
    family_rules = _FAMILY_NUMBER_RULES[signal]
    exit_policy = config.get("exit_policy", "FIXED")
    if not isinstance(exit_policy, str) or exit_policy not in EXIT_POLICIES:
        raise DojoBotCatalogError(
            "exit_policy must be FIXED, BREAKEVEN, or ATR_TRAILING"
        )
    exit_rules = _EXIT_POLICY_NUMBER_RULES[exit_policy]
    supplied_exit_parameters = set(config) & _ALL_EXIT_PARAMETER_KEYS
    missing_exit_parameters = sorted(set(exit_rules) - supplied_exit_parameters)
    excess_exit_parameters = sorted(supplied_exit_parameters - set(exit_rules))
    if missing_exit_parameters or excess_exit_parameters:
        raise DojoBotCatalogError(
            "exit overlay schema mismatch; "
            f"missing={missing_exit_parameters}, unknown={excess_exit_parameters}"
        )

    allowed = _COMMON_INPUT_KEYS | set(family_rules)
    unknown = sorted(set(config) - allowed)
    missing = sorted(_COMMON_REQUIRED_KEYS - set(config))
    missing.extend(sorted(set(family_rules) - set(config)))
    if unknown or missing:
        raise DojoBotCatalogError(
            f"bot config schema mismatch; missing={missing}, unknown={unknown}"
        )

    tp_pips_present = config.get("tp_pips") is not None
    tp_atr_present = config.get("tp_atr") is not None
    if signal in NEW_CANDIDATE_FAMILIES:
        if tp_pips_present or tp_atr_present:
            raise DojoBotCatalogError(
                "dynamic-target bot families require tp_pips and tp_atr to be null"
            )
        if config["sl_pips"] is not None:
            raise DojoBotCatalogError(
                "dynamic-stop bot families require sl_pips to be null"
            )
    elif tp_pips_present == tp_atr_present:
        raise DojoBotCatalogError("exactly one of tp_pips or tp_atr is required")

    has_legacy_cap = "max_concurrent" in config
    has_explicit_cap = "max_concurrent_per_pair" in config
    if not has_legacy_cap and not has_explicit_cap:
        raise DojoBotCatalogError(
            "max_concurrent_per_pair (or legacy max_concurrent) is required"
        )
    pair_cap_legacy = (
        _normalize_number(
            config["max_concurrent"],
            _COMMON_NUMBER_RULES["max_concurrent"],
            "max_concurrent",
        )
        if has_legacy_cap
        else None
    )
    pair_cap_explicit = (
        _normalize_number(
            config["max_concurrent_per_pair"],
            _COMMON_NUMBER_RULES["max_concurrent_per_pair"],
            "max_concurrent_per_pair",
        )
        if has_explicit_cap
        else None
    )
    if (
        pair_cap_legacy is not None
        and pair_cap_explicit is not None
        and pair_cap_legacy != pair_cap_explicit
    ):
        raise DojoBotCatalogError(
            "max_concurrent and max_concurrent_per_pair must agree"
        )
    pair_cap = int(
        pair_cap_explicit if pair_cap_explicit is not None else pair_cap_legacy
    )

    pairs = _normalize_pairs(config["pairs"])
    global_cap = int(
        _normalize_number(
            config["global_max_concurrent"],
            _COMMON_NUMBER_RULES["global_max_concurrent"],
            "global_max_concurrent",
        )
    )
    if global_cap > pair_cap * len(pairs):
        raise DojoBotCatalogError(
            "global_max_concurrent exceeds the declared per-pair capacity"
        )
    if signal == "fade_ladder" and pair_cap < 2:
        raise DojoBotCatalogError("fade_ladder requires max_concurrent_per_pair >= 2")

    _validate_authority(config)
    normalized: dict[str, Any] = {
        "signal": signal,
        "pairs": pairs,
        "tp_pips": (
            _normalize_number(
                config["tp_pips"], _COMMON_NUMBER_RULES["tp_pips"], "tp_pips"
            )
            if tp_pips_present
            else None
        ),
        "tp_atr": (
            _normalize_number(
                config["tp_atr"], _COMMON_NUMBER_RULES["tp_atr"], "tp_atr"
            )
            if tp_atr_present
            else None
        ),
        "sl_pips": (
            None
            if config["sl_pips"] is None
            else _normalize_number(
                config["sl_pips"], _COMMON_NUMBER_RULES["sl_pips"], "sl_pips"
            )
        ),
        "ceiling_min": _normalize_number(
            config["ceiling_min"],
            _COMMON_NUMBER_RULES["ceiling_min"],
            "ceiling_min",
        ),
        "max_concurrent_per_pair": pair_cap,
        "global_max_concurrent": global_cap,
        "per_pos_lev": _normalize_number(
            config["per_pos_lev"],
            _COMMON_NUMBER_RULES["per_pos_lev"],
            "per_pos_lev",
        ),
        "atr_floor_pips": _normalize_number(
            config["atr_floor_pips"],
            _COMMON_NUMBER_RULES["atr_floor_pips"],
            "atr_floor_pips",
        ),
        "exit_policy": exit_policy,
    }
    for name, rule in family_rules.items():
        normalized[name] = _normalize_number(config[name], rule, name)
    for name, rule in exit_rules.items():
        normalized[name] = _normalize_number(config[name], rule, name)
    normalized.update(AUTHORITY_INVARIANTS)
    return normalized


def bot_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the strict canonical normalized config."""

    normalized = validate_bot_config(config)
    return _canonical_sha256(normalized, field="normalized bot config")


def bot_risk_policy_manifest() -> dict[str, Any]:
    """Return the repo-owned rank-admission risk policy.

    The policy is intentionally separate from :func:`validate_bot_config`.
    Catalog validation defines what a bounded TRAIN search may propose; this
    policy defines which of those proposals may enter a diagnostic ranking.
    """

    return {
        "contract": RISK_POLICY_CONTRACT,
        "schema_version": 1,
        "hard_envelope": {
            "max_per_position_leverage": _RISK_MAX_PER_POSITION_LEVERAGE,
            "max_concurrent_per_pair": _RISK_MAX_CONCURRENT_PER_PAIR,
            "max_global_concurrent": _RISK_MAX_GLOBAL_CONCURRENT,
            "max_global_gross_leverage": _RISK_MAX_GLOBAL_GROSS_LEVERAGE,
            "max_initial_sl_pips": _RISK_MAX_INITIAL_SL_PIPS,
            "max_single_stop_risk_index": _RISK_MAX_SINGLE_STOP_RISK_INDEX,
            "max_gross_stop_risk_index": _RISK_MAX_GROSS_STOP_RISK_INDEX,
        },
        "fixed_stop_risk_model": {
            "single_stop_risk_index_formula": (
                "per_pos_lev*(sl_pips+stress_slippage_pips_per_fill)"
            ),
            "gross_stop_risk_index_formula": (
                "single_stop_risk_index*global_max_concurrent"
            ),
            "stress_slippage_scope": "ONE_ADVERSE_STOP_EXIT_FILL",
            "price_normalized_nav_loss_requires_runner_market_receipt": True,
        },
        "rankability": {
            "finite_initial_sl_pips_required": True,
            "sl_free_fixed_family_rankable": False,
            "dynamic_initial_stop_rankable": False,
            "dynamic_initial_stop_status": (
                "UNRANKABLE_UNTIL_BOUNDED_ATOMIC_INITIAL_STOP_IS_IMPLEMENTED"
            ),
        },
        "authority": dict(AUTHORITY_INVARIANTS),
    }


def bot_risk_policy_sha256() -> str:
    """Return the canonical digest of the repo-owned risk policy."""

    return _canonical_sha256(bot_risk_policy_manifest(), field="bot risk policy")


def bot_config_risk_vector(
    config: Mapping[str, Any],
    *,
    stress_slippage_pips_per_fill: float,
) -> dict[str, Any]:
    """Mechanically classify one normalized config against the hard envelope.

    The returned pips-times-leverage indices compare configurations that use
    the same pair universe, market window, and stress arm.  Absolute NAV loss
    still requires a runner-owned market-price receipt and is deliberately not
    claimed here.
    """

    normalized = validate_bot_config(config)
    stress_slippage = _non_negative_number(
        stress_slippage_pips_per_fill,
        field="stress_slippage_pips_per_fill",
    )
    per_position_leverage = float(normalized["per_pos_lev"])
    pair_cap = int(normalized["max_concurrent_per_pair"])
    global_cap = int(normalized["global_max_concurrent"])
    pair_gross_leverage = per_position_leverage * pair_cap
    global_gross_leverage = per_position_leverage * global_cap

    envelope_breaches: list[str] = []
    if per_position_leverage > _RISK_MAX_PER_POSITION_LEVERAGE + 1e-12:
        envelope_breaches.append("PER_POSITION_LEVERAGE_HARD_CAP_EXCEEDED")
    if pair_cap > _RISK_MAX_CONCURRENT_PER_PAIR:
        envelope_breaches.append("PAIR_CONCURRENCY_HARD_CAP_EXCEEDED")
    if global_cap > _RISK_MAX_GLOBAL_CONCURRENT:
        envelope_breaches.append("GLOBAL_CONCURRENCY_HARD_CAP_EXCEEDED")
    if global_gross_leverage > _RISK_MAX_GLOBAL_GROSS_LEVERAGE + 1e-12:
        envelope_breaches.append("GLOBAL_GROSS_LEVERAGE_HARD_CAP_EXCEEDED")

    signal = str(normalized["signal"])
    sl_pips = normalized["sl_pips"]
    stop_blockers: list[str] = []
    if signal in NEW_CANDIDATE_FAMILIES:
        stop_bound_kind = "DYNAMIC_UNBOUNDED_BY_CONFIG"
        stop_blockers.append("DYNAMIC_INITIAL_STOP_BOUND_NOT_IMPLEMENTED")
    elif sl_pips is None:
        stop_bound_kind = "MISSING"
        stop_blockers.append("FINITE_INITIAL_SL_BOUND_MISSING")
    else:
        stop_bound_kind = "FIXED_SL_PIPS"

    single_stop_risk_index = (
        per_position_leverage * (float(sl_pips) + stress_slippage)
        if not stop_blockers
        else None
    )
    gross_stop_risk_index = (
        single_stop_risk_index * global_cap
        if single_stop_risk_index is not None
        else None
    )
    if sl_pips is not None and float(sl_pips) > _RISK_MAX_INITIAL_SL_PIPS + 1e-12:
        envelope_breaches.append("INITIAL_SL_PIPS_HARD_CAP_EXCEEDED")
    if (
        single_stop_risk_index is not None
        and single_stop_risk_index > _RISK_MAX_SINGLE_STOP_RISK_INDEX + 1e-12
    ):
        envelope_breaches.append("SINGLE_STOP_RISK_INDEX_HARD_CAP_EXCEEDED")
    if (
        gross_stop_risk_index is not None
        and gross_stop_risk_index > _RISK_MAX_GROSS_STOP_RISK_INDEX + 1e-12
    ):
        envelope_breaches.append("GROSS_STOP_RISK_INDEX_HARD_CAP_EXCEEDED")
    blockers = [*envelope_breaches, *stop_blockers]
    body = {
        "contract": RISK_VECTOR_CONTRACT,
        "schema_version": 1,
        "config_sha256": bot_config_sha256(normalized),
        "risk_policy_contract": RISK_POLICY_CONTRACT,
        "risk_policy_sha256": bot_risk_policy_sha256(),
        "signal": signal,
        "stress_slippage_pips_per_fill": stress_slippage,
        "per_position_leverage": per_position_leverage,
        "max_concurrent_per_pair": pair_cap,
        "max_global_concurrent": global_cap,
        "pair_gross_leverage": pair_gross_leverage,
        "global_gross_leverage": global_gross_leverage,
        "initial_stop_bound_kind": stop_bound_kind,
        "initial_sl_pips": float(sl_pips) if sl_pips is not None else None,
        "single_stop_risk_index": single_stop_risk_index,
        "gross_stop_risk_index": gross_stop_risk_index,
        "hard_envelope_passed": not envelope_breaches,
        "rankable": not blockers,
        "blocker_codes": blockers,
        "absolute_nav_loss_bound_claimed": False,
        "runner_market_receipt_required": True,
        "order_authority": ORDER_AUTHORITY,
        "live_permission": False,
    }
    return {**body, "risk_vector_sha256": _canonical_sha256(body, field="risk vector")}


def catalog_manifest() -> dict[str, Any]:
    """Return a JSON-safe description suitable for trainer preregistration."""

    families = []
    for family in sorted(SUPPORTED_FAMILIES):
        families.append(
            {
                "family_id": family,
                "parameter_keys": sorted(_FAMILY_NUMBER_RULES[family]),
                "implementation_status": (
                    "LAB_BOT_IMPLEMENTED"
                    if family in EXISTING_FAMILIES
                    else "LAB_BOT_IMPLEMENTED_UNTRAINED"
                ),
                "common_exit_contract": (
                    "COMMON_TP_EXACTLY_ONE_SL_OPTIONAL"
                    if family in EXISTING_FAMILIES
                    else "FAMILY_DYNAMIC_TP_SL_COMMON_FIELDS_NULL"
                ),
            }
        )
    return {
        "contract": CATALOG_CONTRACT,
        "schema_version": 1,
        "families": families,
        "risk_policy": bot_risk_policy_manifest(),
        "risk_policy_sha256": bot_risk_policy_sha256(),
        "authority": dict(AUTHORITY_INVARIANTS),
    }


def _canonical_sha256(value: Any, *, field: str) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoBotCatalogError(f"{field} is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _non_negative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoBotCatalogError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DojoBotCatalogError(f"{field} must be finite")
    if number < 0:
        raise DojoBotCatalogError(f"{field} must be non-negative")
    return number


def _normalize_pairs(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > _MAX_PAIRS:
        raise DojoBotCatalogError(f"pairs must be a JSON array of 1..{_MAX_PAIRS}")
    pairs: list[str] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, str) or _PAIR_RE.fullmatch(pair) is None:
            raise DojoBotCatalogError(f"pairs[{index}] is not canonical AAA_BBB")
        if pair[:3] == pair[4:]:
            raise DojoBotCatalogError(f"pairs[{index}] has identical currencies")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise DojoBotCatalogError("pairs must not contain duplicates")
    return sorted(pairs)


def _normalize_number(value: Any, rule: _NumberRule, field: str) -> int | float:
    if rule.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DojoBotCatalogError(f"{field} must be an integer")
        numeric = float(value)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DojoBotCatalogError(f"{field} must be a finite number")
        numeric = float(value)
    if not math.isfinite(numeric):
        raise DojoBotCatalogError(f"{field} must be a finite number")
    below = (
        numeric < rule.minimum if rule.minimum_inclusive else numeric <= rule.minimum
    )
    if below or numeric > rule.maximum:
        left = ">=" if rule.minimum_inclusive else ">"
        raise DojoBotCatalogError(
            f"{field} must be {left} {rule.minimum} and <= {rule.maximum}"
        )
    return int(value) if rule.integer else numeric


def _validate_authority(config: Mapping[str, Any]) -> None:
    for field, expected in AUTHORITY_INVARIANTS.items():
        if field not in config:
            continue
        supplied = config[field]
        if isinstance(expected, bool):
            valid = isinstance(supplied, bool) and supplied is expected
        else:
            valid = isinstance(supplied, str) and supplied == expected
        if not valid:
            raise DojoBotCatalogError(f"{field} cannot widen DOJO authority")


__all__ = [
    "AUTHORITY_INVARIANTS",
    "CATALOG_CONTRACT",
    "DojoBotCatalogError",
    "EXIT_POLICIES",
    "EXISTING_FAMILIES",
    "NEW_CANDIDATE_FAMILIES",
    "ORDER_AUTHORITY",
    "RISK_POLICY_CONTRACT",
    "RISK_VECTOR_CONTRACT",
    "SUPPORTED_FAMILIES",
    "bot_config_risk_vector",
    "bot_config_sha256",
    "bot_risk_policy_manifest",
    "bot_risk_policy_sha256",
    "catalog_manifest",
    "validate_bot_config",
]
