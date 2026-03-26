from __future__ import annotations

from scripts import lane_scoreboard


def test_build_lane_scoreboard_promotes_current_winner_lane() -> None:
    context = {
        "setup_fingerprint": "MomentumBurst|long|transition|tight_fast|reaccel",
        "flow_regime": "transition",
        "microstructure_bucket": "tight_fast",
    }
    summary = {
        "lookback_hours": 6.0,
        "strategies": {
            "MomentumBurst": {
                "pocket": "micro",
                "filled_rate": 0.42,
                "setups": [
                    {
                        "setup_key": context["setup_fingerprint"],
                        "match_dimension": "setup_fingerprint",
                        **context,
                        "attempts": 5,
                        "fills": 5,
                        "filled_rate": 1.0,
                        "attempt_share": 0.08,
                        "fill_share": 0.24,
                        "share_gap": -0.16,
                        "hard_block_rate": 0.0,
                    },
                    {
                        "setup_key": "MomentumBurst|long|transition|normal_fast|continuation",
                        "match_dimension": "setup_fingerprint",
                        "setup_fingerprint": "MomentumBurst|long|transition|normal_fast|continuation",
                        "flow_regime": "transition",
                        "microstructure_bucket": "normal_fast",
                        "attempts": 4,
                        "fills": 1,
                        "filled_rate": 0.25,
                        "attempt_share": 0.12,
                        "fill_share": 0.05,
                        "share_gap": 0.07,
                        "hard_block_rate": 0.22,
                    },
                ],
            }
        },
    }

    payload = lane_scoreboard.build_lane_scoreboard(
        summary,
        trade_metrics_by_setup={
            "MomentumBurst": {
                lane_scoreboard._setup_key("MomentumBurst", context): {
                    "closed_trades": 4,
                    "wins": 4,
                    "losses": 0,
                    "win_rate": 1.0,
                    "realized_jpy": 64.0,
                    "sum_pips": 8.4,
                    "gross_profit_jpy": 64.0,
                    "gross_loss_jpy": 0.0,
                    "profit_factor": 2.4,
                    "avg_realized_jpy": 16.0,
                    "avg_pips": 2.1,
                    "stop_loss_count": 0,
                    "margin_closeout_count": 0,
                    "market_close_loss_count": 0,
                    "stop_loss_like_count": 0,
                    "stop_loss_rate": 0.0,
                    "primary_close_reason": "TAKE_PROFIT_ORDER",
                    "close_reason_counts": {"TAKE_PROFIT_ORDER": 4},
                }
            }
        },
        min_attempts=4,
        setup_min_attempts=2,
        max_units_cut=0.22,
        max_units_boost=0.24,
        max_prob_boost=0.10,
    )

    lane = payload["strategies"]["MomentumBurst"]["lanes"][0]

    assert lane["gate_action"] == "promote"
    assert lane["action"] == "boost_participation"
    assert lane["promotion_gate"]["passed"] is True
    assert lane["lot_multiplier"] > 1.0
    assert lane["probability_boost"] > 0.0


def test_build_lane_scoreboard_quarantines_fresh_stop_loss_lane() -> None:
    context = {
        "setup_fingerprint": "RangeFader|short|trend_long|tight_fast|gap:up_lean",
        "flow_regime": "trend_long",
        "microstructure_bucket": "tight_fast",
    }
    summary = {
        "lookback_hours": 6.0,
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "filled_rate": 0.18,
                "setups": [
                    {
                        "setup_key": context["setup_fingerprint"],
                        "match_dimension": "setup_fingerprint",
                        **context,
                        "attempts": 4,
                        "fills": 3,
                        "filled_rate": 0.75,
                        "attempt_share": 0.18,
                        "fill_share": 0.08,
                        "share_gap": 0.10,
                        "hard_block_rate": 0.28,
                    }
                ],
            }
        },
    }

    payload = lane_scoreboard.build_lane_scoreboard(
        summary,
        trade_metrics_by_setup={
            "RangeFader-sell-fade": {
                lane_scoreboard._setup_key("RangeFader-sell-fade", context): {
                    "closed_trades": 3,
                    "wins": 0,
                    "losses": 3,
                    "win_rate": 0.0,
                    "realized_jpy": -42.0,
                    "sum_pips": -6.8,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 42.0,
                    "profit_factor": 0.0,
                    "avg_realized_jpy": -14.0,
                    "avg_pips": -2.2667,
                    "stop_loss_count": 2,
                    "margin_closeout_count": 0,
                    "market_close_loss_count": 1,
                    "stop_loss_like_count": 2,
                    "stop_loss_rate": 0.6667,
                    "primary_close_reason": "STOP_LOSS_ORDER",
                    "close_reason_counts": {
                        "STOP_LOSS_ORDER": 2,
                        "MARKET_ORDER_TRADE_CLOSE": 1,
                    },
                }
            }
        },
        min_attempts=4,
        setup_min_attempts=2,
        max_units_cut=0.22,
        max_units_boost=0.24,
        max_prob_boost=0.10,
    )

    lane = payload["strategies"]["RangeFader-sell-fade"]["lanes"][0]

    assert lane["gate_action"] == "quarantine"
    assert lane["action"] == "trim_units"
    assert lane["quarantine_gate"]["active"] is True
    assert lane["lot_multiplier"] < 1.0
    assert lane["probability_offset"] < 0.0
