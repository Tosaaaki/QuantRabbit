from __future__ import annotations

import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.adaptive_exact_s5_profit_engine import (
    CrossSectionalSpec,
    ExactMinutePoint,
    candidate_metrics,
    compact_s5_to_exact_minutes,
    evaluate_locked_spec,
    evaluate_spec,
    fixed_strategy_specs_v1,
    prepare_exact_s5_series,
    prior_anchor_audit_policy_v1,
    run_train_research,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


UTC = timezone.utc


def _s5(timestamp: datetime, *, mid: float, spread: float = 0.0001) -> S5BidAskCandle:
    bid = mid - spread / 2.0
    ask = mid + spread / 2.0
    return S5BidAskCandle(
        timestamp_utc=timestamp,
        bid_o=bid,
        bid_h=bid,
        bid_l=bid,
        bid_c=bid,
        ask_o=ask,
        ask_h=ask,
        ask_l=ask,
        ask_c=ask,
    )


def _trend_series(
    *, start: datetime, minutes: int, base: float, slope: float
) -> tuple[ExactMinutePoint, ...]:
    rows: list[ExactMinutePoint] = []
    for index in range(minutes):
        minute = start + timedelta(minutes=index)
        mid = base + index * slope
        spread = 0.0001
        rows.append(
            ExactMinutePoint(
                minute_utc=minute,
                first_s5_utc=minute,
                bid_open=mid - spread / 2.0,
                ask_open=mid + spread / 2.0,
                last_s5_utc=minute + timedelta(seconds=55),
                bid_close=mid - spread / 2.0,
                ask_close=mid + spread / 2.0,
            )
        )
    return tuple(rows)


def _four_pair_trends(
    *, start: datetime, minutes: int
) -> dict[str, tuple[ExactMinutePoint, ...]]:
    return {
        "EUR_USD": _trend_series(
            start=start, minutes=minutes, base=1.10, slope=0.0000015
        ),
        "GBP_USD": _trend_series(
            start=start, minutes=minutes, base=1.25, slope=0.0000008
        ),
        "AUD_USD": _trend_series(
            start=start, minutes=minutes, base=0.75, slope=-0.0000015
        ),
        "NZD_USD": _trend_series(
            start=start, minutes=minutes, base=0.70, slope=-0.0000008
        ),
    }


def _direct_spec(**overrides: object) -> CrossSectionalSpec:
    values: dict[str, object] = {
        "score_family": "RETURN_OVER_SHORT_ABS",
        "orientation": "DIRECT",
        "lookback_minutes": 480,
        "short_minutes": 60,
        "hold_minutes": 360,
        "cadence_minutes": 60,
        "rank_count": 1,
        "dispersion_floor_pips": 0.0,
    }
    values.update(overrides)
    return CrossSectionalSpec(**values)  # type: ignore[arg-type]


def test_compaction_keeps_exact_first_open_last_close_and_missing_minute() -> None:
    start = datetime(2026, 1, 2, 12, tzinfo=UTC)
    candles = (
        _s5(start, mid=1.1000),
        _s5(start + timedelta(seconds=55), mid=1.1004),
        _s5(start + timedelta(minutes=2), mid=1.1008),
    )

    points = compact_s5_to_exact_minutes(candles)

    assert [point.minute_utc for point in points] == [
        start,
        start + timedelta(minutes=2),
    ]
    assert points[0].first_s5_utc == start
    assert points[0].last_s5_utc == start + timedelta(seconds=55)
    assert points[0].mid_close == pytest.approx(1.1004)
    assert points[1].bid_open == pytest.approx(1.10075)


def test_evaluate_direct_uses_spread_included_exact_opens() -> None:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    series = _four_pair_trends(start=start, minutes=2_000)
    opened_from = start + timedelta(minutes=720)
    opened_to = opened_from + timedelta(hours=12)

    direct = evaluate_spec(
        series,
        spec=_direct_spec(),
        opened_from_utc=opened_from,
        opened_to_utc=opened_to,
    )
    inverse = evaluate_spec(
        series,
        spec=_direct_spec(orientation="INVERSE"),
        opened_from_utc=opened_from,
        opened_to_utc=opened_to,
    )

    assert direct
    assert {row.side for row in direct} == {"LONG", "SHORT"}
    assert all(row.realized_pips > 0.0 for row in direct)
    assert all(row.entry_delay_seconds == 0 for row in direct)
    assert all(row.exit_delay_seconds == 0 for row in direct)
    assert inverse
    assert all(row.realized_pips < 0.0 for row in inverse)


def test_fixed_family_has_192_unique_specs_and_contains_original_shape() -> None:
    specs = fixed_strategy_specs_v1()

    assert len(specs) == 192
    assert len({spec.spec_id for spec in specs}) == 192
    assert any(
        spec.score_family == "RETURN_OVER_SHORT_ABS"
        and spec.orientation == "DIRECT"
        and spec.lookback_minutes == 480
        and spec.short_minutes == 60
        and spec.hold_minutes == 720
        and spec.cadence_minutes == 240
        and spec.rank_count == 2
        and spec.dispersion_floor_pips == 0.0
        for spec in specs
    )


def test_entry_relative_hold_uses_raw_s5_inside_due_minute() -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    series = {}
    slopes = {
        "EUR_USD": 0.00002,
        "GBP_USD": 0.00001,
        "AUD_USD": -0.00002,
        "NZD_USD": -0.00001,
    }
    # The prior-anchor policy requires an exact T-4H quote key, so the raw
    # S5 history must extend at least 240 minutes before the first decision.
    for pair, slope in slopes.items():
        candles = []
        for minute in range(250):
            for second in (5, 55):
                timestamp = start + timedelta(minutes=minute, seconds=second)
                candles.append(_s5(timestamp, mid=1.2 + minute * slope))
        series[pair] = prepare_exact_s5_series(tuple(candles))
    spec = CrossSectionalSpec(
        score_family="RETURN_PIPS",
        orientation="DIRECT",
        lookback_minutes=2,
        short_minutes=1,
        hold_minutes=1,
        cadence_minutes=1,
        rank_count=1,
        dispersion_floor_pips=0.0,
    )

    outcomes = evaluate_spec(
        series,
        spec=spec,
        opened_from_utc=start + timedelta(minutes=243),
        opened_to_utc=start + timedelta(minutes=249),
        policy=prior_anchor_audit_policy_v1(),
    )

    assert outcomes
    assert all(row.entry_utc.second == 5 for row in outcomes)
    assert all(row.exit_utc == row.entry_utc + timedelta(minutes=1) for row in outcomes)
    assert all(row.exit_delay_seconds == 0 for row in outcomes)


def test_train_research_locks_only_one_before_locked_replication() -> None:
    source_start = datetime(2026, 1, 1, tzinfo=UTC)
    train_from = datetime(2026, 1, 2, tzinfo=UTC)
    train_to = datetime(2026, 1, 27, tzinfo=UTC)
    validation_from = train_to
    validation_to = datetime(2026, 2, 2, tzinfo=UTC)
    series = _four_pair_trends(
        start=source_start,
        minutes=int((validation_to - source_start).total_seconds() // 60),
    )

    research, lock = run_train_research(
        series,
        train_from_utc=train_from,
        train_to_utc=train_to,
        source_manifest_sha256="a" * 64,
        slice_receipts_sha256="b" * 64,
    )

    assert research["candidate_count"] == 192
    assert research["eligible_candidate_count"] > 0
    assert lock is not None
    assert research["locked_survivor_spec_id"] == lock["spec"]["spec_id"]
    assert lock["validation_accessed_during_lock"] is False
    assert lock["jul10_17_strategy_outcomes_evaluated"] is False
    assert lock["jul10_17_byte_unseen_claimed"] is False
    assert lock["manifest_integrity_scan_includes_later_source_rows"] is True
    assert lock["order_authority"] == "NONE"
    assert lock["live_permission"] is False

    evaluation = evaluate_locked_spec(
        series,
        lock=lock,
        research=research,
        opened_from_utc=validation_from,
        opened_to_utc=validation_to,
        source_manifest_sha256="a" * 64,
        slice_receipts_sha256="c" * 64,
        related_approximation_was_previously_inspected=True,
    )
    assert evaluation["metrics"]["net_pips"] > 0.0
    assert evaluation["research_sha256"] == research["research_sha256"]
    assert evaluation["related_approximation_was_previously_inspected"] is True
    assert evaluation["independent_validation_claim_allowed"] is False
    assert evaluation["order_authority"] == "NONE"


def test_lock_digest_and_manifest_change_fail_closed() -> None:
    source_start = datetime(2026, 1, 1, tzinfo=UTC)
    train_from = datetime(2026, 1, 2, tzinfo=UTC)
    train_to = datetime(2026, 1, 27, tzinfo=UTC)
    validation_to = datetime(2026, 2, 2, tzinfo=UTC)
    series = _four_pair_trends(
        start=source_start,
        minutes=int((validation_to - source_start).total_seconds() // 60),
    )
    research, lock = run_train_research(
        series,
        train_from_utc=train_from,
        train_to_utc=train_to,
        source_manifest_sha256="a" * 64,
        slice_receipts_sha256="b" * 64,
    )
    assert lock is not None

    tampered = dict(lock)
    tampered["live_permission"] = True
    with pytest.raises(ValueError, match="digest"):
        evaluate_locked_spec(
            series,
            lock=tampered,
            research=research,
            opened_from_utc=train_to,
            opened_to_utc=validation_to,
            source_manifest_sha256="a" * 64,
            slice_receipts_sha256="c" * 64,
            related_approximation_was_previously_inspected=True,
        )
    with pytest.raises(ValueError, match="manifest"):
        evaluate_locked_spec(
            series,
            lock=lock,
            research=research,
            opened_from_utc=train_to,
            opened_to_utc=validation_to,
            source_manifest_sha256="d" * 64,
            slice_receipts_sha256="c" * 64,
            related_approximation_was_previously_inspected=True,
        )


def test_minted_lock_without_matching_research_survivor_is_refused() -> None:
    source_start = datetime(2026, 1, 1, tzinfo=UTC)
    train_from = datetime(2026, 1, 2, tzinfo=UTC)
    train_to = datetime(2026, 1, 27, tzinfo=UTC)
    validation_to = datetime(2026, 2, 2, tzinfo=UTC)
    series = _four_pair_trends(
        start=source_start,
        minutes=int((validation_to - source_start).total_seconds() // 60),
    )
    research, lock = run_train_research(
        series,
        train_from_utc=train_from,
        train_to_utc=train_to,
        source_manifest_sha256="a" * 64,
        slice_receipts_sha256="b" * 64,
    )
    assert lock is not None
    survivor_id = lock["spec"]["spec_id"]
    other_spec = next(
        spec for spec in fixed_strategy_specs_v1() if spec.spec_id != survivor_id
    )

    # Mint a self-consistent lock for a spec TRAIN never froze: same family,
    # valid digest, but not the research survivor. The seal alone passes
    # _validate_lock, so only the research binding can refuse it.
    from quant_rabbit.adaptive_exact_s5_profit_engine import (
        _canonical_sha,
        _spec_payload,
    )

    minted_body = {
        key: value for key, value in lock.items() if key != "lock_sha256"
    }
    minted_body["spec"] = _spec_payload(other_spec)
    minted = {**minted_body, "lock_sha256": _canonical_sha(minted_body)}

    with pytest.raises(ValueError, match="survivor"):
        evaluate_locked_spec(
            series,
            lock=minted,
            research=research,
            opened_from_utc=train_to,
            opened_to_utc=validation_to,
            source_manifest_sha256="a" * 64,
            slice_receipts_sha256="c" * 64,
            related_approximation_was_previously_inspected=True,
        )

    # A research artifact whose digest was tampered is refused before use.
    forged_research = dict(research)
    forged_research["locked_survivor_spec_id"] = other_spec.spec_id
    with pytest.raises(ValueError, match="research artifact digest"):
        evaluate_locked_spec(
            series,
            lock=minted,
            research=forged_research,
            opened_from_utc=train_to,
            opened_to_utc=validation_to,
            source_manifest_sha256="a" * 64,
            slice_receipts_sha256="c" * 64,
            related_approximation_was_previously_inspected=True,
        )


def test_cost_stress_and_leave_best_group_are_explicit() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    outcomes = evaluate_spec(
        _four_pair_trends(start=start, minutes=3_000),
        spec=_direct_spec(),
        opened_from_utc=start + timedelta(minutes=720),
        opened_to_utc=start + timedelta(minutes=2_900),
    )

    metrics = candidate_metrics(outcomes, spec_id="fixed", stress_pips_per_trade=0.5)

    assert metrics.stressed_net_pips == pytest.approx(
        metrics.net_pips - 0.5 * metrics.trade_count
    )
    assert metrics.leave_best_day_stressed_net_pips < metrics.stressed_net_pips
    assert metrics.leave_best_pair_stressed_net_pips < metrics.stressed_net_pips


def test_runner_refuses_a_preexisting_stale_lock(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run-adaptive-exact-s5-profit-research.py"
        )
    )
    stale_lock = tmp_path / "stale-lock.json"
    stale_lock.write_text('{"live_permission": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="stale reuse"):
        namespace["_require_absent_outputs"](
            tmp_path / "new-research.json", stale_lock
        )
