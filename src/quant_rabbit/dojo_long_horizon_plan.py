"""Sealed, fail-closed plan for the worn historical DOJO long-horizon TRAIN.

The plan is deliberately a *plan*, not a replay runner or an evidence scorer.
It freezes the complete historical denominator before any outcome is opened:

* M5: all configured G8 28 pairs for 2020-01 through 2026-06 (78 months);
* M1 precision: core five pairs over the same 78 months, plus all 28 pairs
  over 2025-01 through 2026-06 (18 months);
* independent-month and continuous-account modes;
* OHLC and OLHC intrabar paths under BASE and STRESS costs;
* full portfolio plus pair, family, and currency leave-one-out replays.

Every historical byte in this interval is already worn TRAIN material.  A
perfect diagnostic result cannot promote a strategy, authorize an order, or
be described as prospective proof.  This module has no filesystem, network,
broker, model, order, or tuning side effect.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, G8_CURRENCIES


CONTRACT: Final = "QR_DOJO_LONG_HORIZON_TRAIN_PLAN_V1"
SCHEMA_VERSION: Final = 1
EVIDENCE_TIER: Final = "WORN_HISTORICAL_TRAIN_ONLY"

M5_GRANULARITY: Final = "M5"
M1_GRANULARITY: Final = "M1"
PRICE_COMPONENT: Final = "BID_ASK"
PERIOD_FROM_UTC: Final = "2020-01-01T00:00:00Z"
PERIOD_TO_UTC: Final = "2026-07-01T00:00:00Z"
STARTING_EQUITY_JPY: Final = 200_000

EVALUATION_MODES: Final = ("INDEPENDENT_MONTH", "CONTINUOUS_ACCOUNT")
INTRABAR_PATHS: Final = ("OHLC", "OLHC")
COST_SCENARIOS: Final = ("BASE", "STRESS")
LOPO_STAGES: Final = ("PAIR_LOPO", "FAMILY_LOPO", "CURRENCY_LOPO")

CORE5_PAIRS: Final = (
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_JPY",
)

M5_BINDING_ID: Final = "M5_EXACT28_2020_2026H1"
M1_CORE5_BINDING_ID: Final = "M1_CORE5_2020_2026H1"
M1_FULL28_BINDING_ID: Final = "M1_FULL28_2025_2026H1"
SOURCE_BINDING_IDS: Final = (
    M5_BINDING_ID,
    M1_CORE5_BINDING_ID,
    M1_FULL28_BINDING_ID,
)

IMPLEMENTATION_DIGEST_KEYS: Final = (
    "base_cost_policy_sha256",
    "m1_precision_policy_sha256",
    "replay_engine_sha256",
    "risk_policy_sha256",
    "scorer_sha256",
    "strategy_bundle_sha256",
    "stress_cost_policy_sha256",
    "trainer_sha256",
)

TARGET_MULTIPLE: Final = 3.0
BASE_MAX_DRAWDOWN_FRACTION: Final = 0.10
STRESS_MAX_DRAWDOWN_FRACTION: Final = 0.15
PEAK_MARGIN_FRACTION_MAX: Final = 0.45
LOPO_PROFIT_DROP_FRACTION_MAX: Final = 0.50

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_SHA256: Final = "0" * 64
_FAMILY_ID_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class DojoLongHorizonPlanError(ValueError):
    """The long-horizon TRAIN plan is malformed or no longer sealed."""


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> tuple[str, ...]:
    year, month = start
    result: list[str] = []
    while (year, month) <= end:
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return tuple(result)


M5_MONTHS: Final = _month_range((2020, 1), (2026, 6))
M1_CORE5_MONTHS: Final = M5_MONTHS
M1_FULL28_MONTHS: Final = _month_range((2025, 1), (2026, 6))


def _check_canonical_json(value: Any, *, path: str = "$", depth: int = 0) -> None:
    if depth > 32:
        raise DojoLongHorizonPlanError("canonical JSON nesting is too deep")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoLongHorizonPlanError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoLongHorizonPlanError(
                    f"{path} contains a non-string JSON object key"
                )
            _check_canonical_json(item, path=f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_canonical_json(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    raise DojoLongHorizonPlanError(
        f"{path} contains unsupported canonical JSON type {type(value).__name__}"
    )


def canonical_sha256(value: Any) -> str:
    """Hash strict, deterministic JSON; NaN and ambiguous key coercion fail."""

    _check_canonical_json(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoLongHorizonPlanError(
            f"value is not strict canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _exact_digest_map(
    value: Mapping[str, str],
    *,
    keys: Sequence[str],
    field: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise DojoLongHorizonPlanError(f"{field} must contain exactly {list(keys)}")
    result: dict[str, str] = {}
    for key in keys:
        digest = value[key]
        if (
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or digest == _ZERO_SHA256
        ):
            raise DojoLongHorizonPlanError(
                f"{field}.{key} must be a non-zero lowercase SHA-256"
            )
        result[key] = digest
    return result


def _families(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoLongHorizonPlanError("portfolio_families must be a sequence")
    rows = tuple(value)
    if not 2 <= len(rows) <= 32:
        raise DojoLongHorizonPlanError(
            "portfolio_families must contain between 2 and 32 families"
        )
    for family_id in rows:
        if not isinstance(family_id, str) or _FAMILY_ID_RE.fullmatch(family_id) is None:
            raise DojoLongHorizonPlanError(
                "portfolio family ids must be lowercase canonical identifiers"
            )
    if rows != tuple(sorted(rows)) or len(set(rows)) != len(rows):
        raise DojoLongHorizonPlanError(
            "portfolio_families must be unique and lexicographically sorted"
        )
    return rows


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "evidence_tier": EVIDENCE_TIER,
        "forward_proof_eligible": False,
        "historical_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
        "trainer_may_change_live_configuration": False,
    }


def _month_mode_contracts() -> list[dict[str, Any]]:
    return [
        {
            "mode": "INDEPENDENT_MONTH",
            "account_reset_at_each_month_start": True,
            "month_open_equity_jpy": STARTING_EQUITY_JPY,
            "positions_or_orders_may_cross_month_boundary": False,
            "state_carried_across_months": False,
            "terminal_flat_required_each_month": True,
            "target_denominator": "MONTH_OPEN_EQUITY",
        },
        {
            "mode": "CONTINUOUS_ACCOUNT",
            "account_reset_at_each_month_start": False,
            "initial_equity_jpy": STARTING_EQUITY_JPY,
            "month_end_scoring_uses_mtm_equity": True,
            "positions_or_orders_may_cross_month_boundary": True,
            "state_carried_across_months": True,
            "terminal_flat_required_at_period_end": True,
            "target_denominator": "MONTH_OPEN_MTM_EQUITY",
        },
    ]


def _m1_rectangles(
    source_digests: Mapping[str, str],
    corpus_digests: Mapping[str, str],
) -> list[dict[str, Any]]:
    rectangles = [
        {
            "rectangle_id": M1_CORE5_BINDING_ID,
            "granularity": M1_GRANULARITY,
            "pairs": list(CORE5_PAIRS),
            "months": list(M1_CORE5_MONTHS),
            "pair_count": len(CORE5_PAIRS),
            "month_count": len(M1_CORE5_MONTHS),
            "pair_month_cell_count": len(CORE5_PAIRS) * len(M1_CORE5_MONTHS),
            "source_digest_sha256": source_digests[M1_CORE5_BINDING_ID],
            "corpus_digest_sha256": corpus_digests[M1_CORE5_BINDING_ID],
        },
        {
            "rectangle_id": M1_FULL28_BINDING_ID,
            "granularity": M1_GRANULARITY,
            "pairs": list(DEFAULT_TRADER_PAIRS),
            "months": list(M1_FULL28_MONTHS),
            "pair_count": len(DEFAULT_TRADER_PAIRS),
            "month_count": len(M1_FULL28_MONTHS),
            "pair_month_cell_count": len(DEFAULT_TRADER_PAIRS) * len(M1_FULL28_MONTHS),
            "source_digest_sha256": source_digests[M1_FULL28_BINDING_ID],
            "corpus_digest_sha256": corpus_digests[M1_FULL28_BINDING_ID],
        },
    ]
    return [{**row, "rectangle_sha256": canonical_sha256(row)} for row in rectangles]


def _denominator(families: Sequence[str]) -> dict[str, Any]:
    month_count = len(M5_MONTHS)
    mode_count = len(EVALUATION_MODES)
    path_count = len(INTRABAR_PATHS)
    scenario_count = len(COST_SCENARIOS)
    base_strata = month_count * mode_count * path_count * scenario_count
    pair_lopo = base_strata * len(DEFAULT_TRADER_PAIRS)
    family_lopo = base_strata * len(families)
    currency_lopo = base_strata * len(G8_CURRENCIES)

    m1_pair_month_cells = len(CORE5_PAIRS) * len(M1_CORE5_MONTHS) + len(
        DEFAULT_TRADER_PAIRS
    ) * len(M1_FULL28_MONTHS)
    m1_overlap_cells = len(CORE5_PAIRS) * len(M1_FULL28_MONTHS)
    m1_unique_pair_month_cells = m1_pair_month_cells - m1_overlap_cells
    m1_precision_result_cells = (
        m1_pair_month_cells * mode_count * path_count * scenario_count
    )

    stages = [
        {
            "stage": "PORTFOLIO_MAIN",
            "label_count": 1,
            "result_cell_count": base_strata,
        },
        {
            "stage": "PAIR_LOPO",
            "labels": list(DEFAULT_TRADER_PAIRS),
            "label_count": len(DEFAULT_TRADER_PAIRS),
            "result_cell_count": pair_lopo,
        },
        {
            "stage": "FAMILY_LOPO",
            "labels": list(families),
            "label_count": len(families),
            "result_cell_count": family_lopo,
        },
        {
            "stage": "CURRENCY_LOPO",
            "labels": list(G8_CURRENCIES),
            "label_count": len(G8_CURRENCIES),
            "removed_pair_count_per_label": 7,
            "result_cell_count": currency_lopo,
        },
    ]
    body = {
        "months": list(M5_MONTHS),
        "month_count": month_count,
        "m5_pairs": list(DEFAULT_TRADER_PAIRS),
        "m5_pair_count": len(DEFAULT_TRADER_PAIRS),
        "m5_pair_month_source_cell_count": len(DEFAULT_TRADER_PAIRS) * month_count,
        "evaluation_modes": list(EVALUATION_MODES),
        "intrabar_paths": list(INTRABAR_PATHS),
        "cost_scenarios": list(COST_SCENARIOS),
        "base_month_mode_path_scenario_cell_count": base_strata,
        "portfolio_stages": stages,
        "portfolio_result_cell_count": base_strata
        + pair_lopo
        + family_lopo
        + currency_lopo,
        "m1_precision_rectangle_pair_month_cell_count": m1_pair_month_cells,
        "m1_precision_overlap_pair_month_cell_count": m1_overlap_cells,
        "m1_precision_unique_pair_month_cell_count": m1_unique_pair_month_cells,
        "m1_precision_result_cell_count": m1_precision_result_cells,
        "total_required_result_cell_count": base_strata
        + pair_lopo
        + family_lopo
        + currency_lopo
        + m1_precision_result_cells,
        "missing_or_failed_cell_policy": "COUNT_IN_DENOMINATOR_AND_FAIL_CLOSED",
        "early_stopping_allowed": False,
        "additive_substitute_for_lopo_allowed": False,
    }
    return {**body, "denominator_sha256": canonical_sha256(body)}


def _diagnostic_gates() -> dict[str, Any]:
    body = {
        "objective": "MONTHLY_3X_WORN_TRAIN_DIAGNOSTIC",
        "target_multiple": TARGET_MULTIPLE,
        "target_is_sizing_input": False,
        "target_may_backsolve_risk_or_leverage": False,
        "pessimistic_month_multiple_policy": "MIN_OHLC_OLHC_STRESS",
        "required_month_count": len(M5_MONTHS),
        "required_three_x_month_count": len(M5_MONTHS),
        "required_three_x_hit_rate": 1.0,
        "maximum_losing_month_count": 0,
        "independent_month_target_required": True,
        "continuous_month_target_required": True,
        "continuous_mode_cannot_substitute_for_failed_independent_months": True,
        "base_max_drawdown_fraction_max": BASE_MAX_DRAWDOWN_FRACTION,
        "stress_max_drawdown_fraction_max": STRESS_MAX_DRAWDOWN_FRACTION,
        "peak_margin_fraction_max": PEAK_MARGIN_FRACTION_MAX,
        "margin_closeout_count_max": 0,
        "pair_lopo_profit_drop_fraction_max": LOPO_PROFIT_DROP_FRACTION_MAX,
        "family_lopo_profit_drop_fraction_max": LOPO_PROFIT_DROP_FRACTION_MAX,
        "currency_lopo_profit_drop_fraction_max": LOPO_PROFIT_DROP_FRACTION_MAX,
        "all_paths_scenarios_modes_and_lopo_required": True,
        "missing_cells_pass": False,
        "diagnostic_pass_grants_proof": False,
        "diagnostic_pass_grants_live_permission": False,
        "market_return_guarantee": False,
    }
    return {**body, "diagnostic_gates_sha256": canonical_sha256(body)}


def build_long_horizon_train_plan(
    *,
    portfolio_families: Sequence[str],
    source_digests: Mapping[str, str],
    corpus_digests: Mapping[str, str],
    implementation_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Build the canonical fixed-denominator worn-TRAIN plan.

    Digest mappings are exact: one M5 binding, two M1 rectangle bindings, and
    the complete strategy/replay/trainer/scorer/cost/risk implementation set.
    Unknown keys are rejected rather than silently omitted from the seal.
    """

    families = _families(portfolio_families)
    sources = _exact_digest_map(
        source_digests,
        keys=SOURCE_BINDING_IDS,
        field="source_digests",
    )
    corpora = _exact_digest_map(
        corpus_digests,
        keys=SOURCE_BINDING_IDS,
        field="corpus_digests",
    )
    implementations = _exact_digest_map(
        implementation_digests,
        keys=IMPLEMENTATION_DIGEST_KEYS,
        field="implementation_digests",
    )

    source_body = {
        "price_component": PRICE_COMPONENT,
        "m5": {
            "binding_id": M5_BINDING_ID,
            "granularity": M5_GRANULARITY,
            "from_utc": PERIOD_FROM_UTC,
            "to_utc": PERIOD_TO_UTC,
            "pairs": list(DEFAULT_TRADER_PAIRS),
            "months": list(M5_MONTHS),
            "source_digest_sha256": sources[M5_BINDING_ID],
            "corpus_digest_sha256": corpora[M5_BINDING_ID],
        },
        "m1_precision_rectangles": _m1_rectangles(sources, corpora),
    }
    source_bindings = {
        **source_body,
        "source_binding_sha256": canonical_sha256(source_body),
    }

    family_body = {
        "families": list(families),
        "family_count": len(families),
        "portfolio_policy": "ALL_FAMILIES_SIMULTANEOUS_SHARED_CAPITAL",
        "selection_after_outcome_allowed": False,
    }
    portfolio = {
        **family_body,
        "family_set_sha256": canonical_sha256(family_body),
    }

    implementation_body = {"digests": implementations}
    implementation = {
        **implementation_body,
        "implementation_binding_sha256": canonical_sha256(implementation_body),
    }

    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": EVIDENCE_TIER,
        "period": {
            "from_utc": PERIOD_FROM_UTC,
            "to_utc": PERIOD_TO_UTC,
            "half_open": True,
            "calendar_months": list(M5_MONTHS),
            "month_count": len(M5_MONTHS),
        },
        "source_bindings": source_bindings,
        "implementation_binding": implementation,
        "portfolio": portfolio,
        "evaluation": {
            "modes": _month_mode_contracts(),
            "intrabar_paths": list(INTRABAR_PATHS),
            "cost_scenarios": list(COST_SCENARIOS),
            "one_decision_context_per_timestamp": True,
            "future_candles_visible_to_strategy": False,
            "terminal_mtm_and_flat_settlement_required": True,
        },
        "missing_data_policy": {
            "interpolation_allowed": False,
            "forward_fill_allowed": False,
            "synthetic_candles_allowed": False,
            "denominator_reduction_allowed": False,
            "missing_source_cell_result": "FAILED_CELL_COUNTS_IN_FIXED_DENOMINATOR",
            "known_gap_disclosure": {
                "pair": "NZD_CHF",
                "granularity": M1_GRANULARITY,
                "months": list(_month_range((2024, 1), (2024, 12))),
                "handling": "NOT_IN_EITHER_M1_PRECISION_RECTANGLE_AND_NEVER_IMPUTED",
            },
        },
        "exact_denominator": _denominator(families),
        "monthly_3x_diagnostic_gates": _diagnostic_gates(),
        "authority": _authority(),
        "limitations": [
            "ALL_2020_2026H1_HISTORY_IS_WORN_TRAIN",
            "NO_UNTOUCHED_HISTORICAL_HOLDOUT",
            "M1_IS_LIMITED_TO_TWO_PREDECLARED_PRECISION_RECTANGLES",
            "NZD_CHF_2024_M1_MISSING_AND_NOT_IMPUTED",
            "HISTORICAL_DIAGNOSTIC_CANNOT_GUARANTEE_MONTHLY_3X",
            "FORWARD_PAPER_AND_SEPARATE_PROMOTION_CONTRACT_REQUIRED",
        ],
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def validate_long_horizon_train_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and byte-semantically compare every fixed field and hash."""

    if not isinstance(plan, Mapping):
        raise DojoLongHorizonPlanError("plan must be one JSON object")
    _check_canonical_json(plan)
    expected_top = {
        "contract",
        "schema_version",
        "classification",
        "period",
        "source_bindings",
        "implementation_binding",
        "portfolio",
        "evaluation",
        "missing_data_policy",
        "exact_denominator",
        "monthly_3x_diagnostic_gates",
        "authority",
        "limitations",
        "plan_sha256",
    }
    if set(plan) != expected_top:
        raise DojoLongHorizonPlanError("plan top-level schema is not exact")

    try:
        portfolio = plan["portfolio"]
        source_bindings = plan["source_bindings"]
        implementation = plan["implementation_binding"]
        rectangles = source_bindings["m1_precision_rectangles"]
        sources = {
            M5_BINDING_ID: source_bindings["m5"]["source_digest_sha256"],
            M1_CORE5_BINDING_ID: rectangles[0]["source_digest_sha256"],
            M1_FULL28_BINDING_ID: rectangles[1]["source_digest_sha256"],
        }
        corpora = {
            M5_BINDING_ID: source_bindings["m5"]["corpus_digest_sha256"],
            M1_CORE5_BINDING_ID: rectangles[0]["corpus_digest_sha256"],
            M1_FULL28_BINDING_ID: rectangles[1]["corpus_digest_sha256"],
        }
        family_ids = portfolio["families"]
        implementation_digests = implementation["digests"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DojoLongHorizonPlanError(
            "plan is missing a sealed source, family, or implementation binding"
        ) from exc

    expected = build_long_horizon_train_plan(
        portfolio_families=family_ids,
        source_digests=sources,
        corpus_digests=corpora,
        implementation_digests=implementation_digests,
    )
    if dict(plan) != expected:
        raise DojoLongHorizonPlanError(
            "plan content or canonical hash drifted from the fixed contract"
        )
    return expected


__all__ = [
    "CONTRACT",
    "CORE5_PAIRS",
    "COST_SCENARIOS",
    "DojoLongHorizonPlanError",
    "EVALUATION_MODES",
    "IMPLEMENTATION_DIGEST_KEYS",
    "INTRABAR_PATHS",
    "M1_CORE5_BINDING_ID",
    "M1_FULL28_BINDING_ID",
    "M5_BINDING_ID",
    "M5_MONTHS",
    "SOURCE_BINDING_IDS",
    "build_long_horizon_train_plan",
    "canonical_sha256",
    "validate_long_horizon_train_plan",
]
