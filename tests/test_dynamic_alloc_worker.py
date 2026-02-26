from __future__ import annotations

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
    scores = compute_scores(rows, min_trades=12, pf_cap=2.0)
    assert "scalp_ping_5s_b_live" in scores
    assert scores["scalp_ping_5s_b_live"]["trades"] == 3
    assert scores["scalp_ping_5s_b_live"]["sum_pips"] == -0.9


def test_compute_scores_stronger_loss_penalty_applies() -> None:
    bad_rows = [
        (f"scalp_ping_5s_b_live-l{i:08x}", "scalp_fast", -2.0, f"2026-02-24T00:{i:02d}:00Z", "STOP_LOSS_ORDER")
        for i in range(60)
    ]
    good_rows = [
        ("MicroPullbackEMA", "micro", 2.5, f"2026-02-24T01:{i:02d}:00Z", "TAKE_PROFIT_ORDER")
        for i in range(60)
    ]
    scores = compute_scores(bad_rows + good_rows, min_trades=24, pf_cap=2.0)

    bad = scores["scalp_ping_5s_b_live"]
    good = scores["MicroPullbackEMA"]

    assert 0.45 <= bad["lot_multiplier"] <= 0.70
    assert bad["pf"] == 0.0
    assert bad["avg_pips"] == -2.0
    assert bad["sl_rate"] >= 0.9

    assert good["lot_multiplier"] > 1.4
    assert good["pf"] == 2.0
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
    scores = compute_scores(rows, min_trades=16, pf_cap=2.0)
    prof = scores["scalp_ping_5s_c_live"]
    assert prof["sum_pips"] > 0
    assert prof["sum_realized_jpy"] < 0
    assert prof["lot_multiplier"] <= 0.7


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

    scores = compute_scores(rows, min_trades=12, pf_cap=2.0)
    prof = scores["scalp_ping_5s_c_live"]
    assert prof["margin_closeout_rate"] >= 0.9
    assert prof["lot_multiplier"] <= 0.5
