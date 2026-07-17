"""Causal cross-sectional FX research on exact S5 bid/ask-derived prices.

The module is intentionally a research/shadow boundary.  It can rank a fixed,
bounded family of deterministic strategies using a TRAIN interval, freeze at
most one survivor, and evaluate that already-frozen survivor later.  It has no
broker client, order, sizing, gateway, or live-promotion surface.

Signals use an ``ExactMinutePoint`` close index.  Execution retains a separate
packed array of every real S5 bid/ask open, because an entry-relative hold can
mature at :05, :10, and so on; a first-open-only minute bar is not exact for
that clock.  No price is synthesized for a missing S5 or minute.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from array import array
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


RESEARCH_CONTRACT = "QR_ADAPTIVE_EXACT_S5_CROSS_SECTIONAL_RESEARCH_V1"
LOCK_CONTRACT = "QR_ADAPTIVE_EXACT_S5_SHADOW_LOCK_V1"
EVALUATION_CONTRACT = "QR_ADAPTIVE_EXACT_S5_LOCKED_EVALUATION_V1"
PROSPECTIVE_FINAL_LOCK_CONTRACT = (
    "QR_ADAPTIVE_EXACT_S5_PROSPECTIVE_FINAL_TEST_LOCK_V1"
)
SPEC_FAMILY_VERSION = "FIXED_192_CROSS_SECTIONAL_SPECS_V1"
EXECUTION_POLICY = (
    "READY_85PCT_FIRST_RAW_S5_BA_ENTRY_HOLD_FROM_ENTRY_"
    "CALENDAR_CONTINUOUS_TIME_CLOSE_SPREAD_INCLUDED_V1"
)
SIGNAL_POLICY = (
    "EXACT_MINUTE_LAST_S5_MID_CLOSE_KEYED_AT_MINUTE_END_"
    "T_T_MINUS_SHORT_T_MINUS_LOOKBACK_V1"
)
AUTHORITY = {
    "historical_only": True,
    "diagnostic_only": True,
    "shadow_only": True,
    "forward_proof_eligible": False,
    "automatic_promotion_allowed": False,
    "promotion_allowed": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
HOLDOUT_DISCLOSURE = {
    "jul10_17_strategy_features_evaluated": False,
    "jul10_17_strategy_outcomes_evaluated": False,
    "jul10_17_byte_unseen_claimed": False,
    "manifest_integrity_scan_includes_later_source_rows": True,
}
PROSPECTIVE_FINAL_FROM_UTC = datetime(2026, 7, 20, tzinfo=timezone.utc)
PROSPECTIVE_FINAL_TO_UTC = datetime(2026, 8, 3, tzinfo=timezone.utc)

_UTC = timezone.utc
_S5_SECONDS = 5
_MINUTE_SECONDS = 60
_VALID_SCORE_FAMILIES = frozenset(
    {"RETURN_PIPS", "RETURN_OVER_SHORT_ABS", "RETURN_MINUS_SHORT"}
)
_VALID_ORIENTATIONS = frozenset({"DIRECT", "INVERSE"})


@dataclass(frozen=True, slots=True)
class ExactMinutePoint:
    """Exact first-open/last-close compaction of real S5 observations."""

    minute_utc: datetime
    first_s5_utc: datetime
    bid_open: float
    ask_open: float
    last_s5_utc: datetime
    bid_close: float
    ask_close: float

    @property
    def mid_close(self) -> float:
        return (self.bid_close + self.ask_close) / 2.0


@dataclass(frozen=True, slots=True)
class ExactS5Series:
    """Signal minute index plus every executable S5 open in packed arrays."""

    points: tuple[ExactMinutePoint, ...]
    minute_epochs: tuple[int, ...]
    s5_epochs: array
    bid_opens: array
    ask_opens: array


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    signal_exact_minute: bool
    signal_max_gap_seconds: int
    require_four_hour_quote: bool
    min_ready_fraction: float
    hold_from_entry: bool
    pre_rank_execution_ready: bool
    require_continuous_fx_week: bool
    execution_max_gap_seconds: int | None

    @property
    def policy_id(self) -> str:
        return "XP-" + _canonical_sha(asdict(self))[:20]


@dataclass(frozen=True, slots=True)
class CrossSectionalSpec:
    score_family: str
    orientation: str
    lookback_minutes: int
    short_minutes: int
    hold_minutes: int
    cadence_minutes: int
    rank_count: int
    dispersion_floor_pips: float

    @property
    def spec_id(self) -> str:
        body = asdict(self)
        return "XS-" + _canonical_sha(body)[:20]


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    pair: str
    side: str
    decision_utc: datetime
    entry_utc: datetime
    exit_utc: datetime
    score: float
    raw_return_pips: float
    entry_bid: float
    entry_ask: float
    exit_bid: float
    exit_ask: float
    gross_mid_pips: float
    round_trip_spread_pips: float
    realized_pips: float
    entry_delay_seconds: int
    exit_delay_seconds: int


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    spec_id: str
    trade_count: int
    active_days: int
    win_count: int
    loss_count: int
    net_pips: float
    mean_pips: float
    profit_factor: float | None
    stress_pips_per_trade: float
    stressed_net_pips: float
    stressed_mean_pips: float
    stressed_profit_factor: float | None
    long_stressed_net_pips: float
    short_stressed_net_pips: float
    leave_best_day_stressed_net_pips: float
    leave_best_pair_stressed_net_pips: float
    positive_day_rate: float
    one_sided_daily_normal_p: float
    holm_adjusted_p: float | None
    max_entry_delay_seconds: int
    max_exit_delay_seconds: int
    train_economic_survivor_eligible: bool
    multiplicity_confirmed: bool


@dataclass(frozen=True, slots=True)
class _ExecutableOpen:
    epoch: int
    bid: float
    ask: float

    @property
    def timestamp_utc(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=_UTC)


def compact_s5_to_exact_minutes(
    candles: Sequence[S5BidAskCandle],
) -> tuple[ExactMinutePoint, ...]:
    """Compact ordered S5 observations without inventing missing prices."""

    return prepare_exact_s5_series(candles).points


def prepare_exact_s5_series(
    candles: Sequence[S5BidAskCandle],
) -> ExactS5Series:
    """Build a closed-minute signal index and retain every exact S5 open."""

    points: list[ExactMinutePoint] = []
    s5_epochs = array("q")
    bid_opens = array("d")
    ask_opens = array("d")
    current_minute: datetime | None = None
    first: S5BidAskCandle | None = None
    last: S5BidAskCandle | None = None
    previous: datetime | None = None

    for candle in candles:
        if candle.__class__ is not S5BidAskCandle:
            raise ValueError("candles must contain exact S5 bid/ask rows")
        timestamp = _aware_utc(candle.timestamp_utc)
        if timestamp.microsecond or int(timestamp.timestamp()) % _S5_SECONDS:
            raise ValueError("S5 timestamp is not exact-grid aligned")
        if previous is not None and timestamp <= previous:
            raise ValueError("S5 candles must be chronological and unique")
        _validate_candle_prices(candle)
        s5_epochs.append(int(timestamp.timestamp()))
        bid_opens.append(float(candle.bid_o))
        ask_opens.append(float(candle.ask_o))
        minute = timestamp.replace(second=0, microsecond=0)
        if current_minute is not None and minute != current_minute:
            assert first is not None and last is not None
            points.append(_minute_point(current_minute, first, last))
            first = None
        current_minute = minute
        if first is None:
            first = candle
        last = candle
        previous = timestamp

    if current_minute is not None:
        assert first is not None and last is not None
        points.append(_minute_point(current_minute, first, last))
    frozen_points = tuple(points)
    return ExactS5Series(
        points=frozen_points,
        minute_epochs=tuple(int(point.minute_utc.timestamp()) for point in frozen_points),
        s5_epochs=s5_epochs,
        bid_opens=bid_opens,
        ask_opens=ask_opens,
    )


def fixed_evaluation_policy_v1() -> EvaluationPolicy:
    """Causal research policy with exact features and calendar continuity."""

    return EvaluationPolicy(
        signal_exact_minute=True,
        signal_max_gap_seconds=300,
        require_four_hour_quote=True,
        min_ready_fraction=0.85,
        hold_from_entry=True,
        pre_rank_execution_ready=True,
        require_continuous_fx_week=True,
        execution_max_gap_seconds=None,
    )


def prior_anchor_audit_policy_v1() -> EvaluationPolicy:
    """Reproduce the previously declared unfiltered anchor exactly."""

    return EvaluationPolicy(
        signal_exact_minute=True,
        signal_max_gap_seconds=300,
        require_four_hour_quote=True,
        min_ready_fraction=0.85,
        hold_from_entry=True,
        pre_rank_execution_ready=True,
        require_continuous_fx_week=False,
        execution_max_gap_seconds=None,
    )


def reconcile_prior_anchor_train(
    series_by_pair: Mapping[str, ExactS5Series | Sequence[ExactMinutePoint]],
    *,
    train_from_utc: datetime,
    train_to_utc: datetime,
    source_manifest_sha256: str,
    slice_receipts_sha256: str,
) -> dict[str, Any]:
    """Explain the provisional-positive/prior-negative anchor contradiction."""

    indexes = _index_series(series_by_pair)
    spec = CrossSectionalSpec(
        score_family="RETURN_OVER_SHORT_ABS",
        orientation="DIRECT",
        lookback_minutes=480,
        short_minutes=60,
        hold_minutes=720,
        cadence_minutes=240,
        rank_count=2,
        dispersion_floor_pips=0.0,
    )
    prior = prior_anchor_audit_policy_v1()
    policies: tuple[tuple[str, EvaluationPolicy], ...] = (
        ("PRIOR_EXACT_BASELINE", prior),
        ("ONLY_SIGNAL_5M_FALLBACK", replace(prior, signal_exact_minute=False)),
        ("ONLY_READY_MIN_4", replace(prior, min_ready_fraction=0.0)),
        ("ONLY_DUE_FROM_DECISION", replace(prior, hold_from_entry=False)),
        (
            "ONLY_POST_RANK_EXECUTION_READY",
            replace(prior, pre_rank_execution_ready=False),
        ),
        (
            "ONLY_EXECUTION_GAP_MAX_5M",
            replace(prior, execution_max_gap_seconds=300),
        ),
        (
            "CAUSAL_CONTINUOUS_FX_WEEK",
            replace(prior, require_continuous_fx_week=True),
        ),
        (
            "CUMULATIVE_PROVISIONAL_POLICY",
            EvaluationPolicy(
                signal_exact_minute=False,
                signal_max_gap_seconds=300,
                require_four_hour_quote=False,
                min_ready_fraction=0.0,
                hold_from_entry=False,
                pre_rank_execution_ready=False,
                require_continuous_fx_week=False,
                execution_max_gap_seconds=300,
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, policy in policies:
        outcomes = _evaluate_indexed(
            indexes,
            spec=spec,
            opened_from_utc=train_from_utc,
            opened_to_utc=train_to_utc,
            policy=policy,
        )
        gross = sum(row.gross_mid_pips for row in outcomes)
        spread = sum(row.round_trip_spread_pips for row in outcomes)
        net = sum(row.realized_pips for row in outcomes)
        if not math.isclose(net, gross - spread, abs_tol=1e-6):
            raise AssertionError("exact entry/exit cost decomposition diverged")
        rows.append(
            {
                "label": label,
                "policy": _policy_payload(policy),
                "decision_count": len({row.decision_utc for row in outcomes}),
                "trade_count": len(outcomes),
                "pair_count": len({row.pair for row in outcomes}),
                "long_count": sum(row.side == "LONG" for row in outcomes),
                "short_count": sum(row.side == "SHORT" for row in outcomes),
                "gross_mid_pips": round(gross, 9),
                "round_trip_spread_pips": round(spread, 9),
                "net_pips": round(net, 9),
                "profit_factor": _rounded_optional(
                    _profit_factor([row.realized_pips for row in outcomes])
                ),
                "max_entry_delay_seconds": max(
                    (row.entry_delay_seconds for row in outcomes), default=0
                ),
                "max_exit_delay_seconds": max(
                    (row.exit_delay_seconds for row in outcomes), default=0
                ),
                "first_ten_trades": [
                    _trade_payload(row) for row in outcomes[:10]
                ],
                "first_ten_compatibility_trades": [
                    _anchor_compatibility_payload(row) for row in outcomes[:10]
                ],
                "compatibility_jsonl_sha256": _jsonl_sha256(
                    [_anchor_compatibility_payload(row) for row in outcomes]
                ),
                "trade_ledger_sha256": _canonical_sha(
                    [_trade_payload(row) for row in outcomes]
                ),
            }
        )
    body: dict[str, Any] = {
        "contract": "QR_ADAPTIVE_EXACT_S5_ANCHOR_RECONCILIATION_V1",
        "schema_version": 1,
        "train_from_utc": _aware_utc(train_from_utc).isoformat(),
        "train_to_utc": _aware_utc(train_to_utc).isoformat(),
        "source_manifest_sha256": source_manifest_sha256,
        "slice_receipts_sha256": slice_receipts_sha256,
        "spec": _spec_payload(spec),
        "configured_pair_count": len(indexes),
        "minimum_ready_pair_count_prior": math.ceil(len(indexes) * 0.85),
        "overlapping_positions_allowed": True,
        "price_rounding_applied": False,
        "extra_cost_beyond_observed_bid_ask_pips": 0.0,
        "variants": rows,
        **HOLDOUT_DISCLOSURE,
        **AUTHORITY,
    }
    return {**body, "reconciliation_sha256": _canonical_sha(body)}


def fixed_strategy_specs_v1() -> tuple[CrossSectionalSpec, ...]:
    """Return the predeclared 192-spec family, independent of prices/outcomes."""

    specs: list[CrossSectionalSpec] = []
    score_shapes: tuple[tuple[str, int, int], ...] = (
        ("RETURN_PIPS", 480, 60),
        ("RETURN_PIPS", 720, 60),
        ("RETURN_OVER_SHORT_ABS", 480, 60),
        ("RETURN_OVER_SHORT_ABS", 480, 240),
        ("RETURN_MINUS_SHORT", 480, 60),
        ("RETURN_MINUS_SHORT", 720, 240),
    )
    for score_family, lookback, short in score_shapes:
        for orientation in ("DIRECT", "INVERSE"):
            for hold in (360, 720):
                for cadence in (60, 240):
                    for rank_count in (1, 2):
                        for dispersion in (0.0, 5.0):
                            specs.append(
                                CrossSectionalSpec(
                                    score_family=score_family,
                                    orientation=orientation,
                                    lookback_minutes=lookback,
                                    short_minutes=short,
                                    hold_minutes=hold,
                                    cadence_minutes=cadence,
                                    rank_count=rank_count,
                                    dispersion_floor_pips=dispersion,
                                )
                            )
    if len(specs) != 192 or len({spec.spec_id for spec in specs}) != 192:
        raise AssertionError("fixed strategy family identity drifted")
    return tuple(specs)


def evaluate_spec(
    series_by_pair: Mapping[str, ExactS5Series | Sequence[ExactMinutePoint]],
    *,
    spec: CrossSectionalSpec,
    opened_from_utc: datetime,
    opened_to_utc: datetime,
    policy: EvaluationPolicy | None = None,
) -> tuple[TradeOutcome, ...]:
    """Evaluate one fixed spec with closed signal data and executable opens.

    A trade is admitted only when both entry and time-close occur before the
    exclusive evaluation end.  This prevents outcome leakage across a split.
    """

    _validate_spec(spec)
    opened_from = _minute_utc(opened_from_utc)
    opened_to = _minute_utc(opened_to_utc)
    if opened_to <= opened_from:
        raise ValueError("evaluation interval must be positive")
    frozen_policy = policy or fixed_evaluation_policy_v1()
    _validate_evaluation_policy(frozen_policy)
    indexes = _index_series(series_by_pair)
    return _evaluate_indexed(
        indexes,
        spec=spec,
        opened_from_utc=opened_from,
        opened_to_utc=opened_to,
        policy=frozen_policy,
    )


def _evaluate_indexed(
    indexes: Mapping[str, ExactS5Series],
    *,
    spec: CrossSectionalSpec,
    opened_from_utc: datetime,
    opened_to_utc: datetime,
    policy: EvaluationPolicy,
) -> tuple[TradeOutcome, ...]:
    opened_from = _minute_utc(opened_from_utc)
    opened_to = _minute_utc(opened_to_utc)
    if len(indexes) < spec.rank_count * 2:
        raise ValueError("insufficient distinct pairs for long/short ranks")

    start_epoch = int(opened_from.timestamp())
    end_epoch = int(opened_to.timestamp())
    first_decision = _ceil_epoch(start_epoch, spec.cadence_minutes * 60)
    hold_seconds = spec.hold_minutes * 60
    required_ready = max(
        spec.rank_count * 2,
        math.ceil(len(indexes) * policy.min_ready_fraction),
    )
    outcomes: list[TradeOutcome] = []

    for decision_epoch in range(
        first_decision, end_epoch, spec.cadence_minutes * 60
    ):
        if decision_epoch + hold_seconds >= end_epoch:
            break
        if policy.require_continuous_fx_week and not _continuous_fx_hold(
            decision_epoch=decision_epoch,
            hold_minutes=spec.hold_minutes,
        ):
            continue
        scored: list[
            tuple[
                str,
                float,
                float,
                _ExecutableOpen | None,
                _ExecutableOpen | None,
                int | None,
            ]
        ] = []
        for pair, index in indexes.items():
            score_row = _score_pair(
                pair,
                index,
                decision_epoch=decision_epoch,
                spec=spec,
                exact_minute=policy.signal_exact_minute,
                max_gap_seconds=policy.signal_max_gap_seconds,
                require_four_hour_quote=policy.require_four_hour_quote,
            )
            if score_row is None:
                continue
            entry: _ExecutableOpen | None = None
            exit_point: _ExecutableOpen | None = None
            exit_boundary: int | None = None
            if policy.pre_rank_execution_ready:
                execution = _execution_for_pair(
                    index,
                    decision_epoch=decision_epoch,
                    hold_seconds=hold_seconds,
                    policy=policy,
                    exclusive_end_epoch=end_epoch,
                )
                if execution is None:
                    continue
                entry, exit_point, exit_boundary = execution
            scored.append(
                (
                    pair,
                    score_row[0],
                    score_row[1],
                    entry,
                    exit_point,
                    exit_boundary,
                )
            )
        if len(scored) < required_ready:
            continue
        raw_values = [row[2] for row in scored]
        if max(raw_values) - min(raw_values) < spec.dispersion_floor_pips:
            continue
        ordered = sorted(scored, key=lambda row: (row[1], row[0]))
        if spec.orientation == "DIRECT":
            short_rows = ordered[: spec.rank_count]
            long_rows = ordered[-spec.rank_count :]
        else:
            long_rows = ordered[: spec.rank_count]
            short_rows = ordered[-spec.rank_count :]
        if {row[0] for row in long_rows} & {row[0] for row in short_rows}:
            raise AssertionError("cross-sectional ranks overlap")
        decision = datetime.fromtimestamp(decision_epoch, tz=_UTC)
        for side, rows in (("SHORT", short_rows), ("LONG", long_rows)):
            for pair, score, raw_return, entry, exit_point, exit_boundary in rows:
                trade = _resolve_time_close(
                    pair,
                    indexes[pair],
                    side=side,
                    decision_utc=decision,
                    hold_minutes=spec.hold_minutes,
                    score=score,
                    raw_return_pips=raw_return,
                    policy=policy,
                    exclusive_end_epoch=end_epoch,
                    prepared_entry=entry,
                    prepared_exit=exit_point,
                    prepared_exit_boundary=exit_boundary,
                )
                if trade is not None:
                    outcomes.append(trade)
    return tuple(outcomes)


def candidate_metrics(
    outcomes: Sequence[TradeOutcome],
    *,
    spec_id: str,
    stress_pips_per_trade: float = 0.5,
    holm_adjusted_p: float | None = None,
) -> CandidateMetrics:
    if not math.isfinite(stress_pips_per_trade) or stress_pips_per_trade < 0:
        raise ValueError("cost stress must be finite and non-negative")
    rows = tuple(outcomes)
    realized = [row.realized_pips for row in rows]
    stressed = [value - stress_pips_per_trade for value in realized]
    daily = _group_sum(rows, stressed, key="day")
    pair = _group_sum(rows, stressed, key="pair")
    side = _group_sum(rows, stressed, key="side")
    net = sum(realized)
    stressed_net = sum(stressed)
    daily_values = list(daily.values())
    p_value = _one_sided_daily_normal_p(daily_values)
    leave_day = stressed_net - max(daily_values, default=0.0)
    leave_pair = stressed_net - max(pair.values(), default=0.0)
    stress_pf = _profit_factor(stressed)
    long_net = side.get("LONG", 0.0)
    short_net = side.get("SHORT", 0.0)
    eligible = (
        len(rows) >= 100
        and len(daily) >= 15
        and stressed_net > 0.0
        and (stress_pf is not None and stress_pf >= 1.05)
        and leave_day > 0.0
        and leave_pair > 0.0
        and long_net > 0.0
        and short_net > 0.0
    )
    adjusted = holm_adjusted_p
    return CandidateMetrics(
        spec_id=spec_id,
        trade_count=len(rows),
        active_days=len(daily),
        win_count=sum(value > 0.0 for value in realized),
        loss_count=sum(value < 0.0 for value in realized),
        net_pips=round(net, 9),
        mean_pips=round(net / len(rows), 9) if rows else 0.0,
        profit_factor=_rounded_optional(_profit_factor(realized)),
        stress_pips_per_trade=stress_pips_per_trade,
        stressed_net_pips=round(stressed_net, 9),
        stressed_mean_pips=(
            round(stressed_net / len(rows), 9) if rows else 0.0
        ),
        stressed_profit_factor=_rounded_optional(stress_pf),
        long_stressed_net_pips=round(long_net, 9),
        short_stressed_net_pips=round(short_net, 9),
        leave_best_day_stressed_net_pips=round(leave_day, 9),
        leave_best_pair_stressed_net_pips=round(leave_pair, 9),
        positive_day_rate=(
            round(sum(value > 0.0 for value in daily_values) / len(daily_values), 9)
            if daily_values
            else 0.0
        ),
        one_sided_daily_normal_p=round(p_value, 12),
        holm_adjusted_p=_rounded_optional(adjusted, digits=12),
        max_entry_delay_seconds=max(
            (row.entry_delay_seconds for row in rows), default=0
        ),
        max_exit_delay_seconds=max(
            (row.exit_delay_seconds for row in rows), default=0
        ),
        train_economic_survivor_eligible=eligible,
        multiplicity_confirmed=adjusted is not None and adjusted <= 0.05,
    )


def run_train_research(
    series_by_pair: Mapping[str, ExactS5Series | Sequence[ExactMinutePoint]],
    *,
    train_from_utc: datetime,
    train_to_utc: datetime,
    source_manifest_sha256: str,
    slice_receipts_sha256: str,
    stress_pips_per_trade: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate the fixed family on TRAIN and freeze zero or one survivor."""

    specs = fixed_strategy_specs_v1()
    policy = fixed_evaluation_policy_v1()
    indexes = _index_series(series_by_pair)
    evaluated: list[tuple[CrossSectionalSpec, tuple[TradeOutcome, ...], CandidateMetrics]] = []
    for spec in specs:
        outcomes = _evaluate_indexed(
            indexes,
            spec=spec,
            opened_from_utc=train_from_utc,
            opened_to_utc=train_to_utc,
            policy=policy,
        )
        metrics = candidate_metrics(
            outcomes,
            spec_id=spec.spec_id,
            stress_pips_per_trade=stress_pips_per_trade,
        )
        evaluated.append((spec, outcomes, metrics))
    adjusted = _holm_adjust(
        [row[2].one_sided_daily_normal_p for row in evaluated]
    )
    evaluated = [
        (
            spec,
            outcomes,
            candidate_metrics(
                outcomes,
                spec_id=spec.spec_id,
                stress_pips_per_trade=stress_pips_per_trade,
                holm_adjusted_p=adjusted[index],
            ),
        )
        for index, (spec, outcomes, _metrics) in enumerate(evaluated)
    ]
    eligible = [row for row in evaluated if row[2].train_economic_survivor_eligible]
    survivor = max(eligible, key=_survivor_rank) if eligible else None

    candidate_rows = [
        {"spec": _spec_payload(spec), "metrics": asdict(metrics)}
        for spec, _outcomes, metrics in evaluated
    ]
    family_sha = _canonical_sha([_spec_payload(spec) for spec in specs])
    body: dict[str, Any] = {
        "contract": RESEARCH_CONTRACT,
        "schema_version": 1,
        "strategy_family_version": SPEC_FAMILY_VERSION,
        "strategy_family_sha256": family_sha,
        "candidate_count": len(specs),
        "train_from_utc": _aware_utc(train_from_utc).isoformat(),
        "train_to_utc": _aware_utc(train_to_utc).isoformat(),
        "source_manifest_sha256": source_manifest_sha256,
        "slice_receipts_sha256": slice_receipts_sha256,
        "signal_policy": SIGNAL_POLICY,
        "execution_policy": EXECUTION_POLICY,
        "evaluation_policy": _policy_payload(policy),
        "cost_stress_pips_per_trade": stress_pips_per_trade,
        "selection_policy": (
            "TRAIN_ONLY_MAX_MIN_STRESSED_NET_LEAVE_BEST_DAY_"
            "LEAVE_BEST_PAIR_THEN_PF_THEN_COUNT_V1"
        ),
        "multiple_testing_policy": (
            "ONE_SIDED_DAILY_NORMAL_APPROXIMATION_HOLM_192_DIAGNOSTIC_V1"
        ),
        "multiple_testing_is_promotion_proof": False,
        "eligible_candidate_count": len(eligible),
        "locked_survivor_spec_id": (
            survivor[0].spec_id if survivor is not None else None
        ),
        "candidates": candidate_rows,
        **HOLDOUT_DISCLOSURE,
        **AUTHORITY,
    }
    research = {**body, "research_sha256": _canonical_sha(body)}
    lock = None
    if survivor is not None:
        spec, outcomes, metrics = survivor
        trade_digest = _canonical_sha([_trade_payload(row) for row in outcomes])
        lock_body: dict[str, Any] = {
            "contract": LOCK_CONTRACT,
            "schema_version": 1,
            "research_sha256": research["research_sha256"],
            "strategy_family_version": SPEC_FAMILY_VERSION,
            "strategy_family_sha256": family_sha,
            "source_manifest_sha256": source_manifest_sha256,
            "slice_receipts_sha256": slice_receipts_sha256,
            "train_from_utc": body["train_from_utc"],
            "train_to_utc": body["train_to_utc"],
            "spec": _spec_payload(spec),
            "evaluation_policy": _policy_payload(policy),
            "train_metrics": asdict(metrics),
            "train_trade_outcomes_sha256": trade_digest,
            "validation_accessed_during_lock": False,
            "locked_for_shadow_replication_only": True,
            **HOLDOUT_DISCLOSURE,
            **AUTHORITY,
        }
        lock = {**lock_body, "lock_sha256": _canonical_sha(lock_body)}
    return research, lock


def evaluate_locked_spec(
    series_by_pair: Mapping[str, ExactS5Series | Sequence[ExactMinutePoint]],
    *,
    lock: Mapping[str, Any],
    research: Mapping[str, Any],
    opened_from_utc: datetime,
    opened_to_utc: datetime,
    source_manifest_sha256: str,
    slice_receipts_sha256: str,
    related_approximation_was_previously_inspected: bool,
) -> dict[str, Any]:
    """Evaluate one immutable TRAIN-locked spec on a later interval.

    The lock alone is a tamper-evidence seal, not a forgery proof: anyone can
    mint a self-consistent lock without running TRAIN.  Requiring the digest
    bound research artifact and its ``locked_survivor_spec_id`` closes that
    gap: the evaluated spec must be the one survivor the TRAIN run froze.
    """

    _validate_lock(lock)
    _validate_research_lock_binding(research=research, lock=lock)
    if source_manifest_sha256 != lock["source_manifest_sha256"]:
        raise ValueError("locked source manifest identity changed")
    opened_from = _aware_utc(opened_from_utc)
    opened_to = _aware_utc(opened_to_utc)
    train_to = datetime.fromisoformat(str(lock["train_to_utc"]))
    if opened_from < train_to or opened_to <= opened_from:
        raise ValueError("locked evaluation interval overlaps TRAIN")
    spec = _spec_from_payload(lock["spec"])
    policy = _policy_from_payload(lock["evaluation_policy"])
    outcomes = evaluate_spec(
        series_by_pair,
        spec=spec,
        opened_from_utc=opened_from,
        opened_to_utc=opened_to,
        policy=policy,
    )
    stress = float(lock["train_metrics"]["stress_pips_per_trade"])
    metrics = candidate_metrics(
        outcomes,
        spec_id=spec.spec_id,
        stress_pips_per_trade=stress,
    )
    body: dict[str, Any] = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "research_sha256": research["research_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "slice_receipts_sha256": slice_receipts_sha256,
        "opened_from_utc": opened_from.isoformat(),
        "opened_to_utc": opened_to.isoformat(),
        "spec": _spec_payload(spec),
        "evaluation_policy": _policy_payload(policy),
        "metrics": asdict(metrics),
        "trade_outcomes_sha256": _canonical_sha(
            [_trade_payload(row) for row in outcomes]
        ),
        "related_approximation_was_previously_inspected": bool(
            related_approximation_was_previously_inspected
        ),
        "independent_validation_claim_allowed": not bool(
            related_approximation_was_previously_inspected
        ),
        **HOLDOUT_DISCLOSURE,
        **AUTHORITY,
    }
    return {**body, "evaluation_sha256": _canonical_sha(body)}


def build_prospective_final_test_lock(
    *, lock: Mapping[str, Any], source_manifest_sha256: str
) -> dict[str, Any]:
    """Predeclare an unavailable future window without reading its prices."""

    _validate_lock(lock)
    if source_manifest_sha256 != lock["source_manifest_sha256"]:
        raise ValueError("prospective lock source manifest identity changed")
    body: dict[str, Any] = {
        "contract": PROSPECTIVE_FINAL_LOCK_CONTRACT,
        "schema_version": 1,
        "shadow_lock_sha256": lock["lock_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "spec": dict(lock["spec"]),
        "evaluation_policy": dict(lock["evaluation_policy"]),
        "opened_from_utc": PROSPECTIVE_FINAL_FROM_UTC.isoformat(),
        "opened_to_utc": PROSPECTIVE_FINAL_TO_UTC.isoformat(),
        "state": "UNAVAILABLE_UNOPENED",
        "strategy_prices_read": False,
        "strategy_outcomes_evaluated": False,
        "selection_use_allowed": False,
        "current_runner_evaluation_allowed": False,
        "earliest_maturity_utc": PROSPECTIVE_FINAL_TO_UTC.isoformat(),
        "manifest_integrity_scan_may_cover_this_clock": False,
        "independent_future_test_claim_requires_new_acquisition": True,
        **AUTHORITY,
    }
    return {**body, "prospective_lock_sha256": _canonical_sha(body)}


def _score_pair(
    pair: str,
    index: ExactS5Series,
    *,
    decision_epoch: int,
    spec: CrossSectionalSpec,
    exact_minute: bool,
    max_gap_seconds: int,
    require_four_hour_quote: bool,
) -> tuple[float, float] | None:
    end = _close_before(
        index,
        decision_epoch,
        exact_minute=exact_minute,
        max_gap_seconds=max_gap_seconds,
    )
    anchor_epoch = decision_epoch - spec.lookback_minutes * 60
    anchor = _close_before(
        index,
        anchor_epoch,
        exact_minute=exact_minute,
        max_gap_seconds=max_gap_seconds,
    )
    short_epoch = decision_epoch - spec.short_minutes * 60
    short = _close_before(
        index,
        short_epoch,
        exact_minute=exact_minute,
        max_gap_seconds=max_gap_seconds,
    )
    four_hour = (
        _close_before(
            index,
            decision_epoch - 240 * 60,
            exact_minute=exact_minute,
            max_gap_seconds=max_gap_seconds,
        )
        if require_four_hour_quote
        else end
    )
    if end is None or anchor is None or short is None or four_hour is None:
        return None
    factor = float(instrument_pip_factor(pair))
    raw = (end.mid_close - anchor.mid_close) * factor
    short_return = (end.mid_close - short.mid_close) * factor
    if spec.score_family == "RETURN_PIPS":
        score = raw
    elif spec.score_family == "RETURN_OVER_SHORT_ABS":
        score = raw / max(abs(short_return), 1.0)
    elif spec.score_family == "RETURN_MINUS_SHORT":
        score = raw - short_return
    else:  # pragma: no cover - guarded by validation
        raise AssertionError("unknown score family")
    if not math.isfinite(score) or not math.isfinite(raw):
        return None
    return score, raw


def _resolve_time_close(
    pair: str,
    index: ExactS5Series,
    *,
    side: str,
    decision_utc: datetime,
    hold_minutes: int,
    score: float,
    raw_return_pips: float,
    policy: EvaluationPolicy,
    exclusive_end_epoch: int,
    prepared_entry: _ExecutableOpen | None = None,
    prepared_exit: _ExecutableOpen | None = None,
    prepared_exit_boundary: int | None = None,
) -> TradeOutcome | None:
    decision_epoch = int(decision_utc.timestamp())
    if prepared_entry is None or prepared_exit is None or prepared_exit_boundary is None:
        execution = _execution_for_pair(
            index,
            decision_epoch=decision_epoch,
            hold_seconds=hold_minutes * 60,
            policy=policy,
            exclusive_end_epoch=exclusive_end_epoch,
        )
        if execution is None:
            return None
        entry, exit_point, exit_boundary = execution
    else:
        entry, exit_point, exit_boundary = (
            prepared_entry,
            prepared_exit,
            prepared_exit_boundary,
        )
    if exit_point.epoch >= exclusive_end_epoch or exit_point.epoch <= entry.epoch:
        return None
    factor = float(instrument_pip_factor(pair))
    entry_mid = (entry.bid + entry.ask) / 2.0
    exit_mid = (exit_point.bid + exit_point.ask) / 2.0
    if side == "LONG":
        gross_mid = (exit_mid - entry_mid) * factor
        realized = (exit_point.bid - entry.ask) * factor
    elif side == "SHORT":
        gross_mid = (entry_mid - exit_mid) * factor
        realized = (entry.bid - exit_point.ask) * factor
    else:
        raise ValueError("side must be LONG or SHORT")
    # Preserve the exact operation order used by the independent stdlib
    # calculator so the compatibility JSONL digest is byte-comparable.
    spread_pips = gross_mid - realized
    return TradeOutcome(
        pair=pair,
        side=side,
        decision_utc=decision_utc,
        entry_utc=entry.timestamp_utc,
        exit_utc=exit_point.timestamp_utc,
        score=score,
        raw_return_pips=raw_return_pips,
        entry_bid=entry.bid,
        entry_ask=entry.ask,
        exit_bid=exit_point.bid,
        exit_ask=exit_point.ask,
        gross_mid_pips=gross_mid,
        round_trip_spread_pips=spread_pips,
        realized_pips=realized,
        entry_delay_seconds=entry.epoch - decision_epoch,
        exit_delay_seconds=exit_point.epoch - exit_boundary,
    )


def _execution_for_pair(
    index: ExactS5Series,
    *,
    decision_epoch: int,
    hold_seconds: int,
    policy: EvaluationPolicy,
    exclusive_end_epoch: int,
) -> tuple[_ExecutableOpen, _ExecutableOpen, int] | None:
    entry = _open_at_or_after(
        index,
        decision_epoch,
        max_gap_seconds=policy.execution_max_gap_seconds,
    )
    if entry is None:
        return None
    exit_boundary = (
        entry.epoch + hold_seconds
        if policy.hold_from_entry
        else decision_epoch + hold_seconds
    )
    exit_point = _open_at_or_after(
        index,
        exit_boundary,
        max_gap_seconds=policy.execution_max_gap_seconds,
    )
    if (
        exit_point is None
        or exit_point.epoch >= exclusive_end_epoch
        or exit_point.epoch <= entry.epoch
    ):
        return None
    return entry, exit_point, exit_boundary


def _index_series(
    series_by_pair: Mapping[str, ExactS5Series | Sequence[ExactMinutePoint]],
) -> dict[str, ExactS5Series]:
    result: dict[str, ExactS5Series] = {}
    for pair, raw_series in sorted(series_by_pair.items()):
        if not isinstance(pair, str):
            raise ValueError("pair identity is invalid")
        instrument_pip_factor(pair)
        if raw_series.__class__ is ExactS5Series:
            series = raw_series
        else:
            points = tuple(raw_series)
            series = ExactS5Series(
                points=points,
                minute_epochs=tuple(
                    int(point.minute_utc.timestamp()) for point in points
                ),
                s5_epochs=array(
                    "q", (int(point.first_s5_utc.timestamp()) for point in points)
                ),
                bid_opens=array("d", (point.bid_open for point in points)),
                ask_opens=array("d", (point.ask_open for point in points)),
            )
        points = series.points
        epochs: list[int] = []
        previous = -1
        for point in points:
            if point.__class__ is not ExactMinutePoint:
                raise ValueError("series point type is invalid")
            _validate_minute_point(point)
            epoch = int(point.minute_utc.timestamp())
            if epoch <= previous:
                raise ValueError("minute points must be chronological and unique")
            epochs.append(epoch)
            previous = epoch
        if tuple(epochs) != series.minute_epochs:
            raise ValueError("minute epoch index diverged from points")
        lengths = (
            len(series.s5_epochs),
            len(series.bid_opens),
            len(series.ask_opens),
        )
        if len(set(lengths)) != 1:
            raise ValueError("exact S5 executable arrays have different lengths")
        previous_s5 = -1
        for index, epoch in enumerate(series.s5_epochs):
            bid = float(series.bid_opens[index])
            ask = float(series.ask_opens[index])
            if epoch <= previous_s5 or epoch % _S5_SECONDS:
                raise ValueError("exact S5 executable clock index is invalid")
            if (
                not math.isfinite(bid)
                or not math.isfinite(ask)
                or bid <= 0.0
                or ask <= 0.0
                or bid > ask
            ):
                raise ValueError("exact S5 executable price is invalid")
            previous_s5 = epoch
        if points:
            result[pair] = series
    return result


def _close_before(
    index: ExactS5Series,
    boundary_epoch: int,
    *,
    exact_minute: bool,
    max_gap_seconds: int,
) -> ExactMinutePoint | None:
    minute_before = ((boundary_epoch - 1) // 60) * 60
    if exact_minute:
        position = bisect.bisect_left(index.minute_epochs, minute_before)
        if (
            position >= len(index.minute_epochs)
            or index.minute_epochs[position] != minute_before
        ):
            return None
    else:
        position = bisect.bisect_right(index.minute_epochs, minute_before) - 1
    if position < 0:
        return None
    point = index.points[position]
    age = boundary_epoch - int(point.last_s5_utc.timestamp())
    if age <= 0 or age > max_gap_seconds:
        return None
    return point


def _open_at_or_after(
    index: ExactS5Series,
    boundary_epoch: int,
    *,
    max_gap_seconds: int | None,
) -> _ExecutableOpen | None:
    position = bisect.bisect_left(index.s5_epochs, boundary_epoch)
    if position >= len(index.s5_epochs):
        return None
    epoch = int(index.s5_epochs[position])
    if max_gap_seconds is not None and epoch - boundary_epoch > max_gap_seconds:
        return None
    return _ExecutableOpen(
        epoch=epoch,
        bid=float(index.bid_opens[position]),
        ask=float(index.ask_opens[position]),
    )


def _continuous_fx_hold(*, decision_epoch: int, hold_minutes: int) -> bool:
    decision = datetime.fromtimestamp(decision_epoch, tz=_UTC)
    status = compute_market_status(decision)
    return bool(
        status.is_fx_open
        and status.minutes_to_next_close is not None
        and status.minutes_to_next_close > hold_minutes + 5
    )


def _group_sum(
    rows: Sequence[TradeOutcome], values: Sequence[float], *, key: str
) -> dict[Any, float]:
    result: dict[Any, float] = {}
    for row, value in zip(rows, values, strict=True):
        if key == "day":
            group: Any = row.decision_utc.date()
        elif key == "pair":
            group = row.pair
        elif key == "side":
            group = row.side
        else:  # pragma: no cover - internal invariant
            raise AssertionError("unknown group")
        result[group] = result.get(group, 0.0) + value
    return result


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    if losses == 0.0:
        return math.inf if gains > 0.0 else None
    return gains / losses


def _one_sided_daily_normal_p(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    sigma = stdev(values)
    mean = fmean(values)
    if sigma == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    z_score = mean / (sigma / math.sqrt(len(values)))
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda row: (row[1], row[0]))
    result = [1.0] * count
    previous = 0.0
    for rank, (index, value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * value)
        previous = max(previous, adjusted)
        result[index] = previous
    return tuple(result)


def _survivor_rank(
    row: tuple[CrossSectionalSpec, tuple[TradeOutcome, ...], CandidateMetrics],
) -> tuple[float, float, int, float, str]:
    spec, _outcomes, metrics = row
    robustness_floor = min(
        metrics.stressed_net_pips,
        metrics.leave_best_day_stressed_net_pips,
        metrics.leave_best_pair_stressed_net_pips,
        metrics.long_stressed_net_pips,
        metrics.short_stressed_net_pips,
    )
    # stressed_profit_factor is None both for an empty candidate and for a
    # loss-free one (infinite PF rounded away); among eligible rows only the
    # loss-free case can occur, and it must rank above any finite PF.
    stressed_pf = metrics.stressed_profit_factor
    if stressed_pf is None:
        stressed_pf = math.inf if metrics.stressed_net_pips > 0.0 else 0.0
    return (
        robustness_floor,
        stressed_pf,
        metrics.trade_count,
        -spec.dispersion_floor_pips,
        spec.spec_id,
    )


def _validate_spec(spec: CrossSectionalSpec) -> None:
    if spec.__class__ is not CrossSectionalSpec:
        raise ValueError("strategy spec type is invalid")
    if spec.score_family not in _VALID_SCORE_FAMILIES:
        raise ValueError("score family is invalid")
    if spec.orientation not in _VALID_ORIENTATIONS:
        raise ValueError("orientation is invalid")
    for value, label in (
        (spec.lookback_minutes, "lookback"),
        (spec.short_minutes, "short lookback"),
        (spec.hold_minutes, "hold"),
        (spec.cadence_minutes, "cadence"),
        (spec.rank_count, "rank count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} is invalid")
    if spec.short_minutes >= spec.lookback_minutes:
        raise ValueError("short lookback must be below long lookback")
    if (
        not math.isfinite(spec.dispersion_floor_pips)
        or spec.dispersion_floor_pips < 0.0
    ):
        raise ValueError("dispersion floor is invalid")


def _validate_evaluation_policy(policy: EvaluationPolicy) -> None:
    if policy.__class__ is not EvaluationPolicy:
        raise ValueError("evaluation policy type is invalid")
    for field in (
        policy.signal_exact_minute,
        policy.require_four_hour_quote,
        policy.hold_from_entry,
        policy.pre_rank_execution_ready,
        policy.require_continuous_fx_week,
    ):
        if field.__class__ is not bool:
            raise ValueError("evaluation policy boolean is invalid")
    if (
        isinstance(policy.signal_max_gap_seconds, bool)
        or not isinstance(policy.signal_max_gap_seconds, int)
        or not 0 <= policy.signal_max_gap_seconds <= 900
    ):
        raise ValueError("evaluation signal gap is invalid")
    if (
        not math.isfinite(policy.min_ready_fraction)
        or not 0.0 <= policy.min_ready_fraction <= 1.0
    ):
        raise ValueError("evaluation ready fraction is invalid")
    if policy.execution_max_gap_seconds is not None and (
        isinstance(policy.execution_max_gap_seconds, bool)
        or not isinstance(policy.execution_max_gap_seconds, int)
        or not 0 <= policy.execution_max_gap_seconds <= 86_400
    ):
        raise ValueError("evaluation execution gap is invalid")


def _validate_research_lock_binding(
    *, research: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    """Refuse a lock that is not the one survivor of a digest-bound TRAIN run."""

    if not isinstance(research, Mapping):
        raise ValueError("research artifact must be an object")
    if (
        research.get("contract") != RESEARCH_CONTRACT
        or research.get("schema_version") != 1
    ):
        raise ValueError("research artifact contract is invalid")
    body = {key: value for key, value in research.items() if key != "research_sha256"}
    if research.get("research_sha256") != _canonical_sha(body):
        raise ValueError("research artifact digest is invalid")
    if lock.get("research_sha256") != research["research_sha256"]:
        raise ValueError("shadow lock is not bound to this research artifact")
    if research.get("locked_survivor_spec_id") != lock["spec"]["spec_id"]:
        raise ValueError("shadow lock spec is not the research TRAIN survivor")
    for key in (
        "strategy_family_version",
        "strategy_family_sha256",
        "source_manifest_sha256",
        "slice_receipts_sha256",
        "train_from_utc",
        "train_to_utc",
    ):
        if research.get(key) != lock.get(key):
            raise ValueError(f"shadow lock/research identity diverged: {key}")


def _validate_lock(lock: Mapping[str, Any]) -> None:
    if not isinstance(lock, Mapping):
        raise ValueError("shadow lock must be an object")
    if lock.get("contract") != LOCK_CONTRACT or lock.get("schema_version") != 1:
        raise ValueError("shadow lock contract is invalid")
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("lock_sha256") != _canonical_sha(body):
        raise ValueError("shadow lock digest is invalid")
    for key, value in AUTHORITY.items():
        if lock.get(key) != value:
            raise ValueError(f"shadow lock authority field is invalid: {key}")
    spec = _spec_from_payload(lock["spec"])
    _validate_spec(spec)
    if spec.spec_id != lock["spec"]["spec_id"]:
        raise ValueError("shadow lock spec identity is invalid")
    if lock.get("validation_accessed_during_lock") is not False:
        raise ValueError("shadow lock claims validation access")
    _policy_from_payload(lock.get("evaluation_policy"))
    for key, value in HOLDOUT_DISCLOSURE.items():
        if lock.get(key) != value:
            raise ValueError(f"shadow lock holdout disclosure is invalid: {key}")


def _spec_payload(spec: CrossSectionalSpec) -> dict[str, Any]:
    return {**asdict(spec), "spec_id": spec.spec_id}


def _spec_from_payload(value: Any) -> CrossSectionalSpec:
    if not isinstance(value, Mapping):
        raise ValueError("strategy spec payload is invalid")
    body = {key: item for key, item in value.items() if key != "spec_id"}
    if set(body) != {
        "score_family",
        "orientation",
        "lookback_minutes",
        "short_minutes",
        "hold_minutes",
        "cadence_minutes",
        "rank_count",
        "dispersion_floor_pips",
    }:
        raise ValueError("strategy spec payload keys are invalid")
    spec = CrossSectionalSpec(**body)
    _validate_spec(spec)
    if value.get("spec_id") != spec.spec_id:
        raise ValueError("strategy spec payload identity is invalid")
    return spec


def _policy_payload(policy: EvaluationPolicy) -> dict[str, Any]:
    _validate_evaluation_policy(policy)
    return {**asdict(policy), "policy_id": policy.policy_id}


def _policy_from_payload(value: Any) -> EvaluationPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation policy payload is invalid")
    body = {key: item for key, item in value.items() if key != "policy_id"}
    if set(body) != {
        "signal_exact_minute",
        "signal_max_gap_seconds",
        "require_four_hour_quote",
        "min_ready_fraction",
        "hold_from_entry",
        "pre_rank_execution_ready",
        "require_continuous_fx_week",
        "execution_max_gap_seconds",
    }:
        raise ValueError("evaluation policy payload keys are invalid")
    policy = EvaluationPolicy(**body)
    _validate_evaluation_policy(policy)
    if value.get("policy_id") != policy.policy_id:
        raise ValueError("evaluation policy payload identity is invalid")
    return policy


def _trade_payload(row: TradeOutcome) -> dict[str, Any]:
    result = asdict(row)
    for key in ("decision_utc", "entry_utc", "exit_utc"):
        result[key] = result[key].isoformat()
    return result


def _anchor_compatibility_payload(row: TradeOutcome) -> dict[str, Any]:
    identity = {
        "decision_utc": _iso_z(row.decision_utc),
        "pair": row.pair,
        "side": row.side,
        "entry_utc": _iso_z(row.entry_utc),
        "exit_utc": _iso_z(row.exit_utc),
    }
    return {
        "trade_id": "EXACT-S5-" + _canonical_sha(identity)[:20],
        **identity,
        "score": row.score,
        "raw_return_pips": row.raw_return_pips,
        "entry_bid": row.entry_bid,
        "entry_ask": row.entry_ask,
        "exit_bid": row.exit_bid,
        "exit_ask": row.exit_ask,
        "gross_mid_pips": row.gross_mid_pips,
        "round_trip_spread_pips": row.round_trip_spread_pips,
        "realized_pips": row.realized_pips,
    }


def _iso_z(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _minute_point(
    minute: datetime, first: S5BidAskCandle, last: S5BidAskCandle
) -> ExactMinutePoint:
    return ExactMinutePoint(
        minute_utc=minute,
        first_s5_utc=_aware_utc(first.timestamp_utc),
        bid_open=float(first.bid_o),
        ask_open=float(first.ask_o),
        last_s5_utc=_aware_utc(last.timestamp_utc),
        bid_close=float(last.bid_c),
        ask_close=float(last.ask_c),
    )


def _validate_candle_prices(candle: S5BidAskCandle) -> None:
    values = (
        candle.bid_o,
        candle.bid_h,
        candle.bid_l,
        candle.bid_c,
        candle.ask_o,
        candle.ask_h,
        candle.ask_l,
        candle.ask_c,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
        raise ValueError("S5 price is invalid")
    if candle.bid_o > candle.ask_o or candle.bid_c > candle.ask_c:
        raise ValueError("S5 bid/ask is crossed")


def _validate_minute_point(point: ExactMinutePoint) -> None:
    minute = _minute_utc(point.minute_utc)
    first = _aware_utc(point.first_s5_utc)
    last = _aware_utc(point.last_s5_utc)
    if point.minute_utc != minute:
        raise ValueError("minute point is not aligned")
    if not (minute <= first <= last < minute + timedelta(minutes=1)):
        raise ValueError("minute point S5 clocks escape the minute")
    values = (point.bid_open, point.ask_open, point.bid_close, point.ask_close)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("minute point price is invalid")
    if point.bid_open > point.ask_open or point.bid_close > point.ask_close:
        raise ValueError("minute point bid/ask is crossed")


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(_UTC)


def _minute_utc(value: datetime) -> datetime:
    aware = _aware_utc(value)
    if aware.second or aware.microsecond:
        raise ValueError("timestamp must be minute aligned")
    return aware


def _ceil_epoch(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rounded_optional(value: float | None, *, digits: int = 9) -> float | None:
    if value is None:
        return None
    if math.isinf(value):
        return None
    return round(value, digits)
