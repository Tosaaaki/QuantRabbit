from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from scripts.dynamic_alloc_worker import compute_scores, normalize_strategy_key


def test_normalize_strategy_key_strips_ephemeral_suffix() -> None:
    assert (
        normalize_strategy_key("scalp_ping_5s_b_live-l6ab7c614")
        == "scalp_ping_5s_b_live"
    )
    assert (
        normalize_strategy_key("scalp_ping_5s_flow_live-fe90be38")
        == "scalp_ping_5s_flow_live"
    )
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


def test_compute_scores_prefers_lane_tag_from_entry_thesis_when_trade_rows_are_canonicalized() -> (
    None
):
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
        (
            f"scalp_ping_5s_b_live-l{i:08x}",
            "scalp_fast",
            -2.0,
            f"2026-02-24T00:{i:02d}:00Z",
            "STOP_LOSS_ORDER",
        )
        for i in range(60)
    ]
    good_rows = [
        (
            "MicroPullbackEMA",
            "micro",
            2.5,
            f"2026-02-24T01:{i:02d}:00Z",
            "TAKE_PROFIT_ORDER",
        )
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


def test_compute_scores_emits_setup_override_for_fast_reactive_loser() -> None:
    now = datetime.now(timezone.utc)
    thesis = json.dumps(
        {
            "setup_fingerprint": "VwapRevertS|short|range_fade|tight_fast|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
            "live_setup_context": {
                "flow_regime": "range_fade",
                "microstructure_bucket": "tight_fast",
            },
        },
        ensure_ascii=True,
    )
    rows = [
        (
            "VwapRevertS",
            "scalp",
            -0.8,
            (now - timedelta(minutes=2))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "STOP_LOSS_ORDER",
            -12.0,
            -1200,
            "VwapRevertS",
            "VwapRevertS",
            thesis,
        ),
        (
            "VwapRevertS",
            "scalp",
            -0.6,
            (now - timedelta(minutes=1))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "MARKET_ORDER_TRADE_CLOSE",
            -9.0,
            -1200,
            "VwapRevertS",
            "VwapRevertS",
            thesis,
        ),
    ]

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=12,
        setup_min_trades=4,
        pf_cap=2.0,
    )

    prof = strategy_scores["VwapRevertS"]
    overrides = prof.get("setup_overrides") or []
    loser = next(
        item
        for item in overrides
        if item["setup_fingerprint"]
        == "VwapRevertS|short|range_fade|tight_fast|rsi:overbought|atr:low|gap:up_lean|volatility_compression"
    )

    assert loser["trades"] == 2
    assert loser["lot_multiplier"] < 1.0
    assert loser["sum_realized_jpy"] < 0.0


def test_compute_scores_recent_cash_loser_does_not_get_boosted_by_good_pips() -> None:
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(48):
        rows.append(
            (
                "MicroLevelReactor",
                "micro",
                2.2,
                (now - timedelta(minutes=47 - i))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
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


def test_compute_scores_keeps_strong_winner_above_floor_despite_small_margin_closeout_noise() -> (
    None
):
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


def test_compute_scores_caps_size_below_global_floor_when_market_close_losses_dominate() -> (
    None
):
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


def test_compute_scores_emits_setup_overrides_from_entry_thesis() -> None:
    rows = []
    for i in range(8):
        rows.append(
            (
                "RangeFader-sell-fade",
                "scalp",
                -1.8,
                f"2026-02-24T00:{i:02d}:00Z",
                "MARKET_ORDER_TRADE_CLOSE",
                -95.0,
                1400,
                "RangeFader-sell-fade",
                "RangeFader-sell-fade",
                '{"setup_fingerprint":"RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression","flow_regime":"trend_long","microstructure_bucket":"tight_fast"}',
            )
        )
    for i in range(8):
        rows.append(
            (
                "RangeFader-sell-fade",
                "scalp",
                1.1,
                f"2026-02-24T01:{i:02d}:00Z",
                "TAKE_PROFIT_ORDER",
                60.0,
                1400,
                "RangeFader-sell-fade",
                "RangeFader-sell-fade",
                '{"setup_fingerprint":"RangeFader-sell-fade|short|transition|normal_normal|rsi:mid|atr:mid|gap:up_lean|volatility_compression","flow_regime":"transition","microstructure_bucket":"normal_normal"}',
            )
        )

    strategy_scores, _ = compute_scores(rows, min_trades=12, pf_cap=2.0)

    prof = strategy_scores["RangeFader-sell-fade"]
    assert isinstance(prof.get("setup_overrides"), list)
    loser_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression"
    )
    winner_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "RangeFader-sell-fade|short|transition|normal_normal|rsi:mid|atr:mid|gap:up_lean|volatility_compression"
    )
    assert loser_override["match_dimension"] == "setup_fingerprint"
    assert loser_override["trades"] == 8
    assert loser_override["lot_multiplier"] < winner_override["lot_multiplier"]


def test_compute_scores_emits_four_trade_setup_override_when_strategy_window_is_wider() -> (
    None
):
    rows = []
    for i in range(4):
        rows.append(
            (
                "PrecisionLowVol",
                "scalp",
                -1.7,
                f"2026-02-24T00:{i:02d}:00Z",
                "MARKET_ORDER_TRADE_CLOSE",
                -90.0,
                1400,
                "PrecisionLowVol",
                "PrecisionLowVol",
                '{"setup_fingerprint":"PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression","flow_regime":"range_fade","microstructure_bucket":"unknown"}',
            )
        )
    for i in range(4):
        rows.append(
            (
                "PrecisionLowVol",
                "scalp",
                1.4,
                f"2026-02-24T01:{i:02d}:00Z",
                "TAKE_PROFIT_ORDER",
                75.0,
                1400,
                "PrecisionLowVol",
                "PrecisionLowVol",
                '{"setup_fingerprint":"PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression","flow_regime":"range_fade","microstructure_bucket":"unknown"}',
            )
        )

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=16,
        setup_min_trades=4,
        pf_cap=2.0,
    )

    prof = strategy_scores["PrecisionLowVol"]
    assert isinstance(prof.get("setup_overrides"), list)
    loser_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression"
    )
    winner_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression"
    )
    assert loser_override["trades"] == 4
    assert loser_override["lot_multiplier"] < winner_override["lot_multiplier"]


def test_compute_scores_emits_single_trade_severe_loser_setup_override() -> None:
    rows = [
        (
            "DroughtRevert",
            "scalp",
            -1.5,
            "2026-02-24T00:00:00Z",
            "STOP_LOSS_ORDER",
            -10.57,
            4800,
            "DroughtRevert",
            "DroughtRevert",
            '{"setup_fingerprint":"DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:up_flat|volatility_compression","flow_regime":"range_fade","microstructure_bucket":"unknown"}',
        ),
        (
            "DroughtRevert",
            "scalp",
            1.2,
            "2026-02-24T00:01:00Z",
            "TAKE_PROFIT_ORDER",
            13.47,
            4800,
            "DroughtRevert",
            "DroughtRevert",
            '{"setup_fingerprint":"DroughtRevert|long|range_fade|unknown|rsi:mid|atr:mid|gap:down_strong|volatility_compression","flow_regime":"range_fade","microstructure_bucket":"unknown"}',
        ),
    ]

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=16,
        setup_min_trades=4,
        pf_cap=2.0,
    )

    prof = strategy_scores["DroughtRevert"]
    loser_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:up_flat|volatility_compression"
    )
    assert loser_override["trades"] == 1
    assert loser_override["lot_multiplier"] <= 0.45


def test_compute_scores_crushes_single_trade_strategy_level_severe_loser() -> None:
    rows = [
        (
            "TickImbalance",
            "scalp",
            -1.8,
            "2026-02-24T00:00:00Z",
            "STOP_LOSS_ORDER",
            -69.012,
            3834,
            "TickImbalance",
            "TickImbalance",
            '{"setup_fingerprint":"TickImbalance|short|trend_short|unknown|rsi:ext_oversold|atr:mid|gap:down_extended|trend_ok","flow_regime":"trend_short","microstructure_bucket":"unknown"}',
        ),
    ]

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=8,
        setup_min_trades=2,
        pf_cap=2.0,
        min_lot_multiplier=0.20,
    )

    prof = strategy_scores["TickImbalance"]
    assert prof["trades"] == 1
    assert prof["sum_realized_jpy"] <= -60.0
    assert prof["effective_min_lot_multiplier"] <= 0.16
    assert prof["lot_multiplier"] <= 0.16


def test_compute_scores_crushes_fast_burst_strategy_level_loser() -> None:
    rows = []
    for i in range(4):
        rows.append(
            (
                "WickReversalBlend",
                "scalp",
                -1.3,
                f"2026-02-24T00:0{i}:00Z",
                "MARKET_ORDER_TRADE_CLOSE",
                -18.0,
                1200,
                "WickReversalBlend",
                "WickReversalBlend",
                '{"setup_fingerprint":"WickReversalBlend|short|range_fade|unknown|rsi:overbought|atr:mid|gap:up_flat|trend_ok","flow_regime":"range_fade","microstructure_bucket":"unknown"}',
            )
        )

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=8,
        setup_min_trades=2,
        pf_cap=2.0,
        min_lot_multiplier=0.20,
    )

    prof = strategy_scores["WickReversalBlend"]
    assert prof["trades"] == 4
    assert prof["sum_realized_jpy"] <= -70.0
    assert prof["effective_min_lot_multiplier"] <= 0.18
    assert prof["lot_multiplier"] <= 0.256


def test_compute_scores_emits_low_sample_winner_relief_override() -> None:
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(5):
        rows.append(
            (
                "PrecisionLowVol",
                "scalp",
                -1.8,
                (now - timedelta(minutes=6 - i))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "STOP_LOSS_ORDER",
                -12.0,
                1400,
                "PrecisionLowVol",
                "PrecisionLowVol",
                '{"setup_fingerprint":"PrecisionLowVol|short|range_fade|tight_thin|rsi:mid|atr:mid|gap:down_flat|volatility_compression|align:mixed","flow_regime":"range_fade","microstructure_bucket":"tight_thin"}',
            )
        )
    rows.append(
        (
            "PrecisionLowVol",
            "scalp",
            2.0,
            (now - timedelta(minutes=1))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "TAKE_PROFIT_ORDER",
            1.92,
            1400,
            "PrecisionLowVol",
            "PrecisionLowVol",
            '{"setup_fingerprint":"PrecisionLowVol|long|range_fade|tight_normal|rsi:mid|atr:mid|gap:up_lean|volatility_compression|align:mixed","flow_regime":"range_fade","microstructure_bucket":"tight_normal"}',
        )
    )

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=8,
        setup_min_trades=4,
        pf_cap=2.0,
        min_lot_multiplier=0.20,
    )

    prof = strategy_scores["PrecisionLowVol"]
    winner_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "PrecisionLowVol|long|range_fade|tight_normal|rsi:mid|atr:mid|gap:up_lean|volatility_compression|align:mixed"
    )

    assert winner_override["trades"] == 1
    assert winner_override["sum_realized_jpy"] > 0.0
    assert winner_override["lot_multiplier"] > prof["lot_multiplier"]
    assert 0.70 <= winner_override["lot_multiplier"] <= 0.82


def test_compute_scores_emits_two_trade_winner_relief_near_full_size() -> None:
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(5):
        rows.append(
            (
                "DroughtRevert",
                "scalp",
                -1.7,
                (now - timedelta(minutes=8 - i))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "STOP_LOSS_ORDER",
                -11.5,
                1400,
                "DroughtRevert",
                "DroughtRevert",
                '{"setup_fingerprint":"DroughtRevert|short|range_fade|tight_thin|rsi:mid|atr:mid|gap:up_flat|volatility_compression|align:mixed","flow_regime":"range_fade","microstructure_bucket":"tight_thin"}',
            )
        )
    for i in range(2):
        rows.append(
            (
                "DroughtRevert",
                "scalp",
                2.3,
                (now - timedelta(minutes=1 - i))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "TAKE_PROFIT_ORDER",
                2.24,
                1400,
                "DroughtRevert",
                "DroughtRevert",
                '{"setup_fingerprint":"DroughtRevert|long|range_fade|tight_normal|rsi:mid|atr:mid|gap:down_flat|volatility_compression|align:mixed","flow_regime":"range_fade","microstructure_bucket":"tight_normal"}',
            )
        )

    strategy_scores, _ = compute_scores(
        rows,
        min_trades=8,
        setup_min_trades=4,
        pf_cap=2.0,
        min_lot_multiplier=0.20,
    )

    prof = strategy_scores["DroughtRevert"]
    winner_override = next(
        item
        for item in prof["setup_overrides"]
        if item.get("setup_fingerprint")
        == "DroughtRevert|long|range_fade|tight_normal|rsi:mid|atr:mid|gap:down_flat|volatility_compression|align:mixed"
    )

    assert winner_override["trades"] == 2
    assert winner_override["sum_realized_jpy"] > 4.0
    assert winner_override["lot_multiplier"] > prof["lot_multiplier"]
    assert 0.82 <= winner_override["lot_multiplier"] <= 1.0
