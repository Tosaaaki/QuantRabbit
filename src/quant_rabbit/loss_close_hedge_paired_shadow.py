"""Read-only paired diagnostics for reverse-stop and loss-lock hypotheses.

This module extends :mod:`loss_close_paired_shadow` without changing its
17-test checkpoint.  It compares one frozen current-TP/SL control with a
precommitted hedge arm over the same complete S5 bid/ask sequence.  The
calculator is deliberately unable to place orders, read broker state, or
authorize live behavior.

S5 cannot prove the order of two fills inside one candle.  Same-trigger and
dual-leg fills are therefore reported as unresolved and every calculation
remains proof-ineligible.  Bid/ask prices carry spread; the explicit cost
model contains only non-spread fee, slippage, and financing stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

from quant_rabbit.loss_close_paired_shadow import (
    S5BidAskCandle,
    _canonical_s5_candle,
    _first_protection_touch,
    _iso_utc,
    _parse_canonical_utc_parts,
    _s5_candle_issues,
    _validated_s5_candles,
    validate_paired_shadow_state,
)


HEDGE_PAIRED_SHADOW_CONTRACT = "loss_close_hedge_paired_shadow_v1"
FIXED_UNWIND_RULE = "FIXED_COMPLETE_S5_CANDLE_PRECOMMITTED"
_A_SCALES = (0.25, 0.35)
_HYPOTHESIS_A = "A_REVERSE_STOP"
_HYPOTHESIS_B = "B_LOSS_LOCK"
_TIMING_INITIAL = "INITIAL_ENTRY"
_TIMING_SL = "SL_TRIGGER"


@dataclass(frozen=True)
class HedgeCostModel:
    """Non-spread costs, all in JPY and applied exactly once."""

    original_entry_fee_jpy: float
    original_entry_slippage_jpy: float
    original_financing_stress_jpy: float
    baseline_sl_fee_jpy: float
    baseline_sl_slippage_jpy: float
    hedge_entry_fee_jpy: float
    hedge_entry_slippage_jpy: float
    hedge_financing_stress_jpy: float
    original_unwind_fee_jpy: float
    original_unwind_slippage_jpy: float
    hedge_unwind_fee_jpy: float
    hedge_unwind_slippage_jpy: float


@dataclass(frozen=True)
class HedgeExperimentSpec:
    """Frozen inputs for one falsifiable A/B path comparison.

    Prices are executable-side, pre-slippage shadow prices and must be inside
    the supplied complete S5 candle.  Slippage is charged separately by
    :class:`HedgeCostModel`; callers must not also bake it into these prices.
    """

    hypothesis: str
    hedge_timing: str
    hedge_scale: float
    original_entry_timestamp_utc: datetime
    original_entry_price: float
    hedge_entry_timestamp_utc: datetime
    hedge_entry_price: float
    unwind_timestamp_utc: datetime
    original_unwind_price: float | None
    hedge_unwind_price: float
    unwind_rule: str
    initial_equity_jpy: float
    ruin_floor_jpy: float
    margin_rate: float
    margin_closeout_ratio: float
    costs: HedgeCostModel
    holdout_used: bool = False


def score_loss_close_hedge_paired_shadow(
    paired_state: object,
    candles: Sequence[S5BidAskCandle],
    spec: HedgeExperimentSpec,
) -> dict[str, Any]:
    """Compare A or B with the frozen current protection on one S5 path.

    The output can refute a hypothesis on this path.  It cannot establish an
    always-profitable claim, estimate a ruin probability, authenticate S5 or
    broker artifacts, or grant any execution permission.
    """

    state = _freeze_state(paired_state)
    blockers: list[str] = []
    if state is None:
        return _result(blockers=["PAIRED_STATE_SNAPSHOT_UNREADABLE"])
    validation = validate_paired_shadow_state(state)
    if not validation["valid"]:
        return _result(
            state_sha256=validation.get("state_sha256"),
            blockers=[f"INVALID_PAIRED_STATE:{x}" for x in validation["issues"]],
        )
    if spec.__class__ is not HedgeExperimentSpec:
        return _result(
            state_sha256=str(state["state_sha256"]),
            blockers=["INVALID_EXPERIMENT_SPEC"],
        )

    blockers.extend(_spec_issues(spec, units=int(state["units"])))
    frozen_candles, blockers_s5 = _freeze_full_s5(
        candles, pair=str(state["pair"])
    )
    blockers.extend(blockers_s5)
    if blockers:
        return _result(
            state_sha256=str(state["state_sha256"]), blockers=blockers
        )

    entry_candle = _candle_at(frozen_candles, spec.original_entry_timestamp_utc)
    hedge_entry_candle = _candle_at(
        frozen_candles, spec.hedge_entry_timestamp_utc
    )
    unwind_candle = _candle_at(frozen_candles, spec.unwind_timestamp_utc)
    if entry_candle is None:
        blockers.append("ORIGINAL_ENTRY_S5_CANDLE_MISSING")
    if hedge_entry_candle is None:
        blockers.append("HEDGE_ENTRY_S5_CANDLE_MISSING")
    if unwind_candle is None:
        blockers.append("UNWIND_S5_CANDLE_MISSING")
    if blockers:
        return _result(
            state_sha256=str(state["state_sha256"]), blockers=blockers
        )
    assert entry_candle is not None
    assert hedge_entry_candle is not None
    assert unwind_candle is not None

    decision_parts = _parse_canonical_utc_parts(state["decision_timestamp_utc"])
    assert decision_parts is not None
    decision_candle_time = decision_parts[0].replace(
        second=(decision_parts[0].second // 5) * 5, microsecond=0
    )
    decision_slice = tuple(
        candle for candle in frozen_candles if candle.timestamp_utc >= decision_candle_time
    )
    _all, control_candles, control_blockers = _validated_s5_candles(
        decision_slice,
        state=state,
        decision_timestamp_parts=decision_parts,
    )
    if control_blockers:
        return _result(
            state_sha256=str(state["state_sha256"]), blockers=control_blockers
        )
    first_touch = _first_protection_touch(state, control_candles)
    if first_touch is None:
        return _result(
            state_sha256=str(state["state_sha256"]),
            status="PENDING_BASELINE_RESOLUTION",
            blockers=[],
        )
    if first_touch["reason"] != "SL":
        reason = first_touch["reason"]
        return _result(
            state_sha256=str(state["state_sha256"]),
            blockers=[f"BASELINE_FIRST_TOUCH_NOT_UNAMBIGUOUS_SL:{reason}"],
        )
    trigger_time = _parse_candle_time(first_touch["candle_timestamp_utc"])
    assert trigger_time is not None
    trigger_candle = _candle_at(frozen_candles, trigger_time)
    assert trigger_candle is not None

    if spec.unwind_timestamp_utc <= trigger_time:
        blockers.append("UNWIND_MUST_FOLLOW_BASELINE_SL_CANDLE")
    expected_hedge_entry = (
        spec.original_entry_timestamp_utc
        if spec.hedge_timing == _TIMING_INITIAL
        else trigger_time
    )
    if spec.hedge_entry_timestamp_utc != expected_hedge_entry:
        blockers.append("HEDGE_ENTRY_TIMESTAMP_DOES_NOT_MATCH_FROZEN_TIMING")

    side = str(state["side"])
    hedge_side = "SHORT" if side == "LONG" else "LONG"
    blockers.extend(
        _price_binding_issues(
            "ORIGINAL_ENTRY", spec.original_entry_price, entry_candle, side, "ENTRY"
        )
    )
    blockers.extend(
        _price_binding_issues(
            "HEDGE_ENTRY", spec.hedge_entry_price, hedge_entry_candle, hedge_side, "ENTRY"
        )
    )
    blockers.extend(
        _price_binding_issues(
            "HEDGE_UNWIND", spec.hedge_unwind_price, unwind_candle, hedge_side, "EXIT"
        )
    )
    if spec.hypothesis == _HYPOTHESIS_B:
        assert spec.original_unwind_price is not None
        blockers.extend(
            _price_binding_issues(
                "ORIGINAL_UNWIND",
                spec.original_unwind_price,
                unwind_candle,
                side,
                "EXIT",
            )
        )
    sl = float(state["stop_loss"])
    blockers.extend(_price_binding_issues("BASELINE_SL", sl, trigger_candle, side, "EXIT"))
    if blockers:
        return _result(
            state_sha256=str(state["state_sha256"]), blockers=blockers
        )

    units = int(state["units"])
    hedge_units_float = units * spec.hedge_scale
    hedge_units = int(hedge_units_float)
    quote_to_jpy = float(state["quote_to_jpy"])
    original_entry_cost = _sum_costs(
        spec.costs.original_entry_fee_jpy,
        spec.costs.original_entry_slippage_jpy,
    )
    baseline_exit_cost = _sum_costs(
        spec.costs.baseline_sl_fee_jpy,
        spec.costs.baseline_sl_slippage_jpy,
    )
    hedge_entry_cost = _sum_costs(
        spec.costs.hedge_entry_fee_jpy,
        spec.costs.hedge_entry_slippage_jpy,
    )
    hedge_unwind_cost = _sum_costs(
        spec.costs.hedge_unwind_fee_jpy,
        spec.costs.hedge_unwind_slippage_jpy,
    )
    original_unwind_cost = _sum_costs(
        spec.costs.original_unwind_fee_jpy,
        spec.costs.original_unwind_slippage_jpy,
    )

    baseline_gross = _directional_jpy(
        side, spec.original_entry_price, sl, units, quote_to_jpy
    )
    baseline_net = baseline_gross - _sum_costs(
        original_entry_cost,
        baseline_exit_cost,
        spec.costs.original_financing_stress_jpy,
    )

    hedge_gross = _directional_jpy(
        hedge_side,
        spec.hedge_entry_price,
        spec.hedge_unwind_price,
        hedge_units,
        quote_to_jpy,
    )
    if spec.hypothesis == _HYPOTHESIS_A:
        original_gross = baseline_gross
        alternative_net = baseline_net + hedge_gross - _sum_costs(
            hedge_entry_cost,
            hedge_unwind_cost,
            spec.costs.hedge_financing_stress_jpy,
        )
        arm_definition = "ORIGINAL_CLOSES_AT_SL_THEN_SCALED_OPPOSITE_LEG_UNWINDS"
    else:
        assert spec.original_unwind_price is not None
        original_gross = _directional_jpy(
            side,
            spec.original_entry_price,
            spec.original_unwind_price,
            units,
            quote_to_jpy,
        )
        alternative_net = original_gross + hedge_gross - _sum_costs(
            original_entry_cost,
            hedge_entry_cost,
            original_unwind_cost,
            hedge_unwind_cost,
            spec.costs.original_financing_stress_jpy,
            spec.costs.hedge_financing_stress_jpy,
        )
        arm_definition = "ORIGINAL_REMAINS_OPEN_WITH_EQUAL_OPPOSITE_LEG_UNTIL_DUAL_UNWIND"
    if not all(
        math.isfinite(value)
        for value in (baseline_gross, baseline_net, hedge_gross, original_gross, alternative_net)
    ):
        return _result(
            state_sha256=str(state["state_sha256"]),
            blockers=["ECONOMIC_CALCULATION_NON_FINITE"],
        )

    path = tuple(
        candle
        for candle in frozen_candles
        if spec.original_entry_timestamp_utc <= candle.timestamp_utc <= spec.unwind_timestamp_utc
    )
    baseline_path = _equity_path(
        path,
        state=state,
        spec=spec,
        trigger_time=trigger_time,
        hedge_units=hedge_units,
        baseline=True,
        baseline_net=baseline_net,
        alternative_net=alternative_net,
    )
    alternative_path = _equity_path(
        path,
        state=state,
        spec=spec,
        trigger_time=trigger_time,
        hedge_units=hedge_units,
        baseline=False,
        baseline_net=baseline_net,
        alternative_net=alternative_net,
    )
    if not _economic_paths_finite(baseline_path, alternative_path):
        return _result(
            state_sha256=str(state["state_sha256"]),
            blockers=["ECONOMIC_PATH_NON_FINITE"],
        )
    trend = _trend_continuation(
        path,
        hedge_side=hedge_side,
        hedge_entry_time=spec.hedge_entry_timestamp_utc,
        entry_price=spec.hedge_entry_price,
        units=hedge_units,
        quote_to_jpy=quote_to_jpy,
    )
    if not all(math.isfinite(value) for value in trend.values()):
        return _result(
            state_sha256=str(state["state_sha256"]),
            blockers=["TREND_EXCURSION_NON_FINITE"],
        )
    baseline_risk = _risk_metrics(
        baseline_path,
        initial_equity=spec.initial_equity_jpy,
        ruin_floor=spec.ruin_floor_jpy,
        closeout_ratio=spec.margin_closeout_ratio,
    )
    alternative_risk = _risk_metrics(
        alternative_path,
        initial_equity=spec.initial_equity_jpy,
        ruin_floor=spec.ruin_floor_jpy,
        closeout_ratio=spec.margin_closeout_ratio,
    )
    delta_jpy = alternative_net - baseline_net
    diagnostic = (
        "OUTPERFORMS_BASELINE_ON_THIS_PATH"
        if delta_jpy > 0.0
        else "UNDERPERFORMS_BASELINE_ON_THIS_PATH"
        if delta_jpy < 0.0
        else "TIES_BASELINE_ON_THIS_PATH"
    )
    if spec.hedge_timing == _TIMING_INITIAL:
        fill_order = "UNRESOLVED_SAME_S5_DUAL_ENTRY"
    elif spec.hypothesis == _HYPOTHESIS_A:
        fill_order = "UNRESOLVED_SAME_S5_SL_CLOSE_AND_REVERSE_OPEN"
    else:
        fill_order = (
            "UNRESOLVED_SAME_S5_SL_TRIGGER_AND_HEDGE_OPEN_"
            "ORIGINAL_REMAINS_OPEN"
        )
    unwind_fill_order = (
        "SINGLE_HEDGE_LEG_UNWIND"
        if spec.hypothesis == _HYPOTHESIS_A
        else "UNRESOLVED_SAME_S5_DUAL_UNWIND"
    )

    return _result(
        state_sha256=str(state["state_sha256"]),
        status="CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS",
        blockers=[],
        payload={
            "hypothesis": spec.hypothesis,
            "arm_definition": arm_definition,
            "hedge_timing": spec.hedge_timing,
            "hedge_scale": spec.hedge_scale,
            "original_units": units,
            "hedge_units": hedge_units,
            "baseline_first_touch": dict(first_touch),
            "unwind_rule": spec.unwind_rule,
            "unwind_timestamp_utc": _iso_utc(spec.unwind_timestamp_utc),
            "fill_order_status": fill_order,
            "unwind_fill_order_status": unwind_fill_order,
            "baseline": {
                "gross_jpy": baseline_gross,
                "net_jpy": baseline_net,
                "costs": {
                    "original_entry_non_spread_jpy": original_entry_cost,
                    "sl_exit_non_spread_jpy": baseline_exit_cost,
                    "financing_stress_jpy": spec.costs.original_financing_stress_jpy,
                },
                "risk": baseline_risk,
            },
            "alternative": {
                "original_gross_jpy": original_gross,
                "hedge_gross_jpy": hedge_gross,
                "net_jpy": alternative_net,
                "risk": alternative_risk,
            },
            "delta_jpy": delta_jpy,
            "diagnostic_outcome": diagnostic,
            "trend_continuation_after_hedge_entry": trend,
            "cost_model": {
                "spread": "INTRINSIC_EXECUTABLE_BID_ASK_NO_EXTRA_CHARGE",
                "fee": "EXPLICIT_NON_SPREAD_JPY_APPLIED_ONCE",
                "slippage": "EXPLICIT_NON_SPREAD_JPY_APPLIED_ONCE",
                "financing": "EXPLICIT_STRESS_JPY_APPLIED_ONCE",
            },
            "margin_model": "SAME_PAIR_LONGEST_LEG_INCREMENT_PROXY",
            "strategy_hedge_authorized": False,
            "ruin_probability_estimated": False,
            "ruin_metric": "DETERMINISTIC_EQUITY_FLOOR_AND_MARGIN_CLOSEOUT_PROXY_ONLY",
        },
    )


def _spec_issues(spec: HedgeExperimentSpec, *, units: int) -> list[str]:
    issues: list[str] = []
    scale_valid = _exact_positive_float(spec.hedge_scale)
    if not scale_valid:
        issues.append("INVALID_POSITIVE_FLOAT:hedge_scale")
    if spec.hypothesis == _HYPOTHESIS_A:
        if spec.hedge_timing != _TIMING_SL:
            issues.append("HYPOTHESIS_A_REQUIRES_SL_TRIGGER_TIMING")
        if spec.hedge_scale not in _A_SCALES:
            issues.append("HYPOTHESIS_A_SCALE_MUST_BE_0_25_OR_0_35")
        if spec.original_unwind_price is not None:
            issues.append("HYPOTHESIS_A_ORIGINAL_UNWIND_PRICE_MUST_BE_NONE")
    elif spec.hypothesis == _HYPOTHESIS_B:
        if spec.hedge_timing not in (_TIMING_INITIAL, _TIMING_SL):
            issues.append("HYPOTHESIS_B_TIMING_INVALID")
        if spec.hedge_scale != 1.0:
            issues.append("HYPOTHESIS_B_REQUIRES_EQUAL_SCALE")
        if spec.original_unwind_price is None:
            issues.append("HYPOTHESIS_B_REQUIRES_DUAL_UNWIND_PRICE")
    else:
        issues.append("UNKNOWN_HYPOTHESIS")
    if spec.unwind_rule != FIXED_UNWIND_RULE:
        issues.append("UNWIND_RULE_NOT_PRECOMMITTED_FIXED_S5")
    if spec.holdout_used is not False:
        issues.append("HOLDOUT_USE_FORBIDDEN")
    if spec.costs.__class__ is not HedgeCostModel:
        issues.append("INVALID_COST_MODEL")
    else:
        for name, value in vars(spec.costs).items():
            if not _exact_nonnegative_float(value):
                issues.append(f"INVALID_NON_SPREAD_COST:{name}")
    for name in (
        "original_entry_price",
        "hedge_entry_price",
        "hedge_unwind_price",
        "initial_equity_jpy",
        "margin_rate",
        "margin_closeout_ratio",
    ):
        value = getattr(spec, name)
        if not _exact_positive_float(value):
            issues.append(f"INVALID_POSITIVE_FLOAT:{name}")
    if spec.original_unwind_price is not None and not _exact_positive_float(
        spec.original_unwind_price
    ):
        issues.append("INVALID_POSITIVE_FLOAT:original_unwind_price")
    if not _exact_nonnegative_float(spec.ruin_floor_jpy):
        issues.append("INVALID_RUIN_FLOOR")
    elif _exact_positive_float(spec.initial_equity_jpy) and spec.ruin_floor_jpy >= spec.initial_equity_jpy:
        issues.append("RUIN_FLOOR_MUST_BE_BELOW_INITIAL_EQUITY")
    if _exact_positive_float(spec.margin_rate) and spec.margin_rate > 1.0:
        issues.append("MARGIN_RATE_ABOVE_ONE")
    if _exact_positive_float(spec.margin_closeout_ratio) and spec.margin_closeout_ratio > 1.0:
        issues.append("MARGIN_CLOSEOUT_RATIO_ABOVE_ONE")
    for name in (
        "original_entry_timestamp_utc",
        "hedge_entry_timestamp_utc",
        "unwind_timestamp_utc",
    ):
        if not _aligned_s5_time(getattr(spec, name)):
            issues.append(f"INVALID_S5_TIMESTAMP:{name}")
    if scale_valid:
        scaled_units = units * spec.hedge_scale
        if (
            not math.isfinite(scaled_units)
            or scaled_units <= 0.0
            or not scaled_units.is_integer()
        ):
            issues.append("HEDGE_UNITS_NOT_EXACT_POSITIVE_INTEGER")
    return list(dict.fromkeys(issues))


def _freeze_full_s5(
    candles: Sequence[S5BidAskCandle], *, pair: str
) -> tuple[tuple[S5BidAskCandle, ...], list[str]]:
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        return (), ["S5_CANDLES_NOT_SEQUENCE"]
    try:
        values = tuple(candles)
    except Exception:
        return (), ["S5_CANDLES_UNREADABLE"]
    issues: list[str] = []
    by_time: dict[datetime, S5BidAskCandle] = {}
    for index, candle in enumerate(values):
        if candle.__class__ is not S5BidAskCandle:
            issues.append(f"INVALID_S5_CANDLE:{index}")
            continue
        found = _s5_candle_issues(candle, pair=pair)
        issues.extend(f"{x}:{index}" for x in found)
        if found:
            continue
        canonical = _canonical_s5_candle(candle)
        previous = by_time.get(canonical.timestamp_utc)
        if previous is not None and previous != canonical:
            issues.append(f"CONFLICTING_S5_CANDLE:{_iso_utc(canonical.timestamp_utc)}")
        else:
            by_time[canonical.timestamp_utc] = canonical
    ordered = tuple(by_time[key] for key in sorted(by_time))
    if not ordered:
        issues.append("S5_TRUTH_EMPTY")
    for previous, current in zip(ordered, ordered[1:]):
        if (current.timestamp_utc - previous.timestamp_utc).total_seconds() != 5.0:
            issues.append(
                f"S5_TRUTH_GAP:{_iso_utc(previous.timestamp_utc)}->{_iso_utc(current.timestamp_utc)}"
            )
    return ordered, list(dict.fromkeys(issues))


def _equity_path(
    candles: Sequence[S5BidAskCandle],
    *,
    state: Mapping[str, Any],
    spec: HedgeExperimentSpec,
    trigger_time: datetime,
    hedge_units: int,
    baseline: bool,
    baseline_net: float,
    alternative_net: float,
) -> list[dict[str, float | str]]:
    side = str(state["side"])
    hedge_side = "SHORT" if side == "LONG" else "LONG"
    units = int(state["units"])
    q = float(state["quote_to_jpy"])
    original_entry_cost = _sum_costs(
        spec.costs.original_entry_fee_jpy,
        spec.costs.original_entry_slippage_jpy,
        spec.costs.original_financing_stress_jpy,
    )
    hedge_entry_cost = _sum_costs(
        spec.costs.hedge_entry_fee_jpy,
        spec.costs.hedge_entry_slippage_jpy,
        spec.costs.hedge_financing_stress_jpy,
    )
    rows: list[dict[str, float | str]] = []
    for candle in candles:
        at_or_after_sl = candle.timestamp_utc >= trigger_time
        at_or_after_hedge = candle.timestamp_utc >= spec.hedge_entry_timestamp_utc
        is_unwind = candle.timestamp_utc == spec.unwind_timestamp_utc
        original_mark = _mark_price(candle, side)
        hedge_mark = _mark_price(candle, hedge_side)
        original_pnl = _directional_jpy(
            side, spec.original_entry_price, original_mark, units, q
        )
        hedge_pnl = (
            _directional_jpy(
                hedge_side, spec.hedge_entry_price, hedge_mark, hedge_units, q
            )
            if at_or_after_hedge
            else 0.0
        )
        if baseline:
            net = baseline_net if at_or_after_sl else original_pnl - original_entry_cost
            long_units, short_units = _legs(side, units if not at_or_after_sl else 0)
        elif is_unwind:
            net = alternative_net
            long_units, short_units = (0, 0)
        elif spec.hypothesis == _HYPOTHESIS_A:
            if at_or_after_sl:
                net = baseline_net + hedge_pnl - hedge_entry_cost
                long_units, short_units = _legs(hedge_side, hedge_units)
            else:
                net = original_pnl - original_entry_cost
                long_units, short_units = _legs(side, units)
        else:
            net = original_pnl - original_entry_cost
            long_units, short_units = _legs(side, units)
            if at_or_after_hedge:
                net += hedge_pnl - hedge_entry_cost
                h_long, h_short = _legs(hedge_side, hedge_units)
                long_units += h_long
                short_units += h_short
        mid = (candle.bid.close + candle.ask.close) / 2.0
        margin = max(long_units, short_units) * mid * q * spec.margin_rate
        gross_notional = (long_units + short_units) * mid * q
        rows.append(
            {
                "timestamp_utc": _iso_utc(candle.timestamp_utc),
                "equity_jpy": spec.initial_equity_jpy + net,
                "margin_required_jpy": margin,
                "gross_notional_jpy": gross_notional,
            }
        )
    return rows


def _risk_metrics(
    rows: Sequence[Mapping[str, float | str]],
    *,
    initial_equity: float,
    ruin_floor: float,
    closeout_ratio: float,
) -> dict[str, Any]:
    peak = initial_equity
    max_drawdown = 0.0
    min_equity = initial_equity
    peak_margin = 0.0
    peak_gross = 0.0
    ruin_floor_breached = False
    margin_proxy_breached = False
    for row in rows:
        equity = float(row["equity_jpy"])
        margin = float(row["margin_required_jpy"])
        gross = float(row["gross_notional_jpy"])
        peak = max(peak, equity)
        min_equity = min(min_equity, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        peak_margin = max(peak_margin, margin)
        peak_gross = max(peak_gross, gross)
        ruin_floor_breached = ruin_floor_breached or equity <= ruin_floor
        margin_proxy_breached = margin_proxy_breached or (
            margin > 0.0 and equity <= margin * closeout_ratio
        )
    return {
        "min_equity_jpy": min_equity,
        "max_drawdown_jpy": max_drawdown,
        "peak_longest_leg_margin_jpy": peak_margin,
        "peak_gross_notional_jpy": peak_gross,
        "ruin_floor_breached": ruin_floor_breached,
        "margin_closeout_proxy_breached": margin_proxy_breached,
        "probability_estimated": False,
    }


def _economic_paths_finite(
    *paths: Sequence[Mapping[str, float | str]],
) -> bool:
    numeric_fields = ("equity_jpy", "margin_required_jpy", "gross_notional_jpy")
    return all(
        math.isfinite(float(row[field]))
        for path in paths
        for row in path
        for field in numeric_fields
    )


def _trend_continuation(
    candles: Sequence[S5BidAskCandle],
    *,
    hedge_side: str,
    hedge_entry_time: datetime,
    entry_price: float,
    units: int,
    quote_to_jpy: float,
) -> dict[str, float]:
    mfe = 0.0
    mae = 0.0
    for candle in candles:
        if candle.timestamp_utc < hedge_entry_time:
            continue
        executable = candle.ask if hedge_side == "SHORT" else candle.bid
        if hedge_side == "SHORT":
            favorable = entry_price - executable.low
            adverse = executable.high - entry_price
        else:
            favorable = executable.high - entry_price
            adverse = entry_price - executable.low
        mfe = max(mfe, favorable * units * quote_to_jpy)
        mae = max(mae, adverse * units * quote_to_jpy)
    return {
        "hedge_mfe_jpy": mfe,
        "hedge_mae_jpy": mae,
    }


def _price_binding_issues(
    label: str,
    price: float,
    candle: S5BidAskCandle,
    side: str,
    action: str,
) -> list[str]:
    quote = candle.ask if (side == "LONG") == (action == "ENTRY") else candle.bid
    if not quote.low <= price <= quote.high:
        return [f"{label}_PRICE_OUTSIDE_S5_EXECUTABLE_RANGE"]
    return []


def _directional_jpy(
    side: str, start: float, end: float, units: int, quote_to_jpy: float
) -> float:
    delta = end - start if side == "LONG" else start - end
    return delta * units * quote_to_jpy


def _mark_price(candle: S5BidAskCandle, side: str) -> float:
    return candle.bid.close if side == "LONG" else candle.ask.close


def _legs(side: str, units: int) -> tuple[int, int]:
    return (units, 0) if side == "LONG" else (0, units)


def _candle_at(
    candles: Sequence[S5BidAskCandle], timestamp: datetime
) -> S5BidAskCandle | None:
    return next((x for x in candles if x.timestamp_utc == timestamp), None)


def _parse_candle_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _aligned_s5_time(value: object) -> bool:
    return (
        value.__class__ is datetime
        and value.tzinfo is not None
        and value.utcoffset() is not None
        and value.astimezone(timezone.utc) == value
        and value.microsecond == 0
        and value.second % 5 == 0
    )


def _exact_positive_float(value: object) -> bool:
    return value.__class__ is float and math.isfinite(value) and value > 0.0


def _exact_nonnegative_float(value: object) -> bool:
    return value.__class__ is float and math.isfinite(value) and value >= 0.0


def _sum_costs(*values: float) -> float:
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("non-finite cost sum")
    return total


def _freeze_state(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return dict(value)
    except Exception:
        return None


def _result(
    *,
    blockers: Sequence[str],
    state_sha256: str | None = None,
    status: str = "BLOCKED",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contract": HEDGE_PAIRED_SHADOW_CONTRACT,
        "status": status,
        "state_sha256": state_sha256,
        "blockers": list(dict.fromkeys(blockers)),
        "diagnostic_calculation_only": True,
        "always_profit_claim_allowed": False,
        "statistical_claim_allowed": False,
        "proof_eligible": False,
        "artifact_bindings_verified_by_evaluator": False,
        "read_only": True,
        "paper_permission_allowed": False,
        "live_permission_allowed": False,
        "broker_order_allowed": False,
        "deployment_allowed": False,
        "holdout_used": False,
    }
    if payload is not None:
        result.update(payload)
    return result
