from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.dynamic_alloc_worker import compute_scores, normalize_strategy_key


def test_normalize_strategy_key_strips_ephemeral_suffix() -> None:
    assert normalize_strategy_key("scalp_ping_5s_b_live-l6ab7c614") == "scalp_ping_5s_b_live"
    assert normalize_strategy_key("scalp_ping_5s_flow_live-fe90be38") == "scalp_ping_5s_flow_live"
    assert normalize_strategy_key("microranbcc01f5cc") == "MicroRangeBreak"
    assert normalize_strategy_key("microtre584b929c1") == "MicroTrendRetest-long"
    assert normalize_strategy_key("micropul9f1b2a3c") == "MicroPullbackEMA"
    assert normalize_strategy_key("scalpmacdrsi69055fe0") == "scalp_macd_rsi_div_b_live"
    assert normalize_strategy_key("MicroTrendRetest-long") == "MicroTrendRetest-long"


def test_compute_scores_aggregates_ephemeral_tags() -> None:
    rows = [
        ("scalp_ping_5s_b_live-l6ab7c614", "scalp_fast", -1.2, "2026-02-24T00:00:00Z"),
        ("scalp_ping_5s_b_live-l7ab7c615", "scalp_fast", 0.8, "2026-02-24T00:01:00Z"),
        ("scalp_ping_5s_b_live", "scalp_fast", -0.5, "2026-02-24T00:02:00Z"),
    ]
    strategy_scores, _ = compute_scores(rows, min_trades=12, pf_cap=2.0)
    assert "scalp_ping_5s_b_live" in strategy_scores
    assert strategy_scores["scalp_ping_5s_b_live"]["trades"] == 3
    assert strategy_scores["scalp_ping_5s_b_live"]["sum_pips"] == -0.9


def test_compute_scores_prefers_lane_tag_from_entry_thesis_when_trade_rows_are_canonicalized() -> None:
    rows = [
        (
            "RangeFader",
            "scalp",
            -1.2,
            "2026-02-24T00:00:00Z",
            "STOP_LOSS_ORDER",
            -110.0,
            1400,
            "RangeFader",
            "RangeFader",
            '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-neutral-fade"}',
        ),
        (
            "RangeFader",
            "scalp",
            0.8,
            "2026-02-24T00:01:00Z",
            "TAKE_PROFIT_ORDER",
            75.0,
            1400,
            "RangeFader",
            "RangeFader",
            '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-neutral-fade"}',
        ),
    ]

    strategy_scores, _ = compute_scores(rows, min_trades=12, pf_cap=2.0)

    assert "RangeFader-neutral-fade" in strategy_scores
    assert "RangeFader" not in strategy_scores


def test_compute_scores_stronger_loss_penalty_applies() -> None:
    bad_rows = [
        (f"scalp_ping_5s_b_live-l{i:08x}", "scalp_fast", -2.0, f"2026-02-24T00:{i:02d}:00Z", "STOP_LOSS_ORDER")
        for i in range(60)
    ]
    good_rows = [
        ("MicroPullbackEMA", "micro", 2.5, f"2026-02-24T01:{i:02d}:00Z", "TAKE_PROFIT_ORDER")
        for i in range(60)
    ]
    strategy_scores, _ = compute_scores(bad_rows + good_rows, min_trades=24, pf_cap=2.0)

    bad = strategy_scores["scalp_ping_5s_b_live"]
    good = strategy_scores["MicroPullbackEMA"]

    assert 0.30 <= bad["lot_multiplier"] <= 0.45
    assert bad["pf"] == 0.0
    assert bad["avg_pips"] == -2.0
    assert bad["sl_rate"] >= 0.9

    assert good["lot_multiplier"] >= 0.68
    assert good["lot_multiplier"] > bad["lot_multiplier"]
    assert good["pf"] > bad["pf"]
    assert good["allow_loser_block"] is False
    assert good["allow_winner_only"] is False


def test_compute_scores_caps_size_when_realized_jpy_is_negative() -> None:
    rows = []
    # Positive pips with negative realized JPY should still be down-ranked.
    for i in range(40):
        rows.append(
            (
                "scalp_ping_5s_c_live",
                "scalp_fast",
                4.0,
                f"2026-02-24T00:{i:02d}:00Z",
                "TAKE_PROFIT_ORDER",
                -80.0,
                2000,
            )
        )
    strategy_scores, _ = compute_scores(rows, min_trades=16, pf_cap=2.0)
    prof = strategy_scores["scalp_ping_5s_c_live"]
    assert prof["sum_pips"] > 0
    assert prof["sum_realized_jpy"] < 0
    assert prof["lot_multiplier"] <= 0.42


def test_compute_scores_recent_cash_loser_does_not_get_boosted_by_good_pips() -> None:
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(48):
        rows.append(
            (
                "MicroLevelReactor",
                "micro",
                2.2,
                (now - timedelta(minutes=47 - i)).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "TAKE_PROFIT_ORDER",
                -6.0,
                1400,
            )
        )

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=12,
        pf_cap=2.0,
        half_life_hours=18.0,
    )
    prof = strategy_scores["MicroLevelReactor"]
    assert prof["sum_pips"] > 0
    assert prof["sum_realized_jpy"] < 0
    assert prof["realized_jpy_per_1k_units"] <= -4.0
    assert prof["pocket_lot_multiplier_applied"] <= 1.0
    assert prof["lot_multiplier"] <= 0.42


def test_compute_scores_caps_size_when_margin_closeout_rate_is_high() -> None:
    rows = []
    for i in range(30):
        rows.append(
            (
                "scalp_ping_5s_c_live",
                "scalp_fast",
                2.5,
                f"2026-02-24T00:{i:02d}:00Z",
                "MARKET_ORDER_MARGIN_CLOSEOUT",
                -30.0,
                1500,
            )
        )

    strategy_scores, _ = compute_scores(rows, min_trades=12, pf_cap=2.0)
    prof = strategy_scores["scalp_ping_5s_c_live"]
    assert prof["margin_closeout_rate"] >= 0.9
    assert prof["lot_multiplier"] <= 0.5


def test_compute_scores_keeps_strong_winner_above_floor_despite_small_margin_closeout_noise() -> None:
    rows = []
    for i in range(27):
        rows.append(
            (
                "MomentumBurst",
                "micro",
                4.0,
                f"2026-02-24T00:{i:02d}:00Z",
                "TAKE_PROFIT_ORDER",
                90.0,
                1500,
            )
        )
    for i in range(4):
        rows.append(
            (
                "MomentumBurst",
                "micro",
                -1.2,
                f"2026-02-24T01:{i:02d}:00Z",
                "MARKET_ORDER_MARGIN_CLOSEOUT",
                -30.0,
                1500,
            )
        )

    strategy_scores, _ = compute_scores(rows, min_trades=12, pf_cap=2.0)
    prof = strategy_scores["MomentumBurst"]
    assert 0.10 <= prof["margin_closeout_rate"] <= 0.15
    assert prof["sum_realized_jpy"] > 1000.0
    assert prof["lot_multiplier"] >= 0.85


def test_compute_scores_caps_size_below_global_floor_when_market_close_losses_dominate() -> None:
    rows = []
    for i in range(72):
        rows.append(
            (
                "M1Scalper-M1",
                "scalp",
                -1.6,
                f"2026-02-24T00:{i % 60:02d}:00Z",
                "MARKET_ORDER_TRADE_CLOSE",
                -110.0,
                1800,
            )
        )

    strategy_scores, _ = compute_scores(rows, min_trades=24, pf_cap=2.0)
    prof = strategy_scores["M1Scalper-M1"]
    assert prof["market_close_loss_share"] >= 0.9
    assert prof["market_close_loss_rate"] >= 0.9
    assert prof["lot_multiplier"] < 0.45


def test_compute_scores_severe_loser_is_crushed_to_emergency_floor() -> None:
    rows = []
    for i in range(96):
        rows.append(
            (
                "M1Scalper-M1",
                "scalp",
                -1.8,
                f"2026-02-24T{(i // 60):02d}:{(i % 60):02d}:00Z",
                "MARKET_ORDER_TRADE_CLOSE",
                -150.0,
                1800,
            )
        )

    strategy_scores, _ = compute_scores(rows, min_trades=24, pf_cap=2.0)
    prof = strategy_scores["M1Scalper-M1"]
    assert prof["sum_realized_jpy"] <= -10000.0
    assert prof["market_close_loss_share"] >= 0.9
    assert prof["realized_jpy_per_1k_units"] <= -7.0
    assert prof["lot_multiplier"] <= 0.12
