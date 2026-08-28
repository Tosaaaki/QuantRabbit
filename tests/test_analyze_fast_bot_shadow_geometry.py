from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_fast_bot_shadow_geometry",
    ROOT / "tools" / "analyze_fast_bot_shadow_geometry.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candle(at: str, *, bid_o: float, bid_h: float, bid_l: float, bid_c: float) -> S5BidAskCandle:
    spread = 0.00008
    return S5BidAskCandle(
        timestamp_utc=datetime.fromisoformat(at.replace("Z", "+00:00")),
        bid_o=bid_o,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=bid_c,
        ask_o=bid_o + spread,
        ask_h=bid_h + spread,
        ask_l=bid_l + spread,
        ask_c=bid_c + spread,
    )


def _signal() -> dict:
    return {
        "signal_id": "signal-1",
        "signal_sha256": "a" * 64,
        "generated_at_utc": "2026-08-28T12:00:00+00:00",
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "pair": "EUR_USD",
        "side": "LONG",
        "method": "RANGE_ROTATION",
        "entry": 1.0,
        "stop_loss_pips": 3.2,
        "take_profit_pips": 2.4,
        "m5_atr_pips": 4.0,
        "spread_pips": 0.8,
        "regime_score": -2.0,
    }


def test_same_path_fixed_loses_atr_wins_veto_skips_and_lot_scales() -> None:
    signal = _signal()
    candles = [
        _candle("2026-08-28T12:00:05Z", bid_o=0.99994, bid_h=1.00010, bid_l=0.99991, bid_c=1.00005),
        _candle("2026-08-28T12:00:10Z", bid_o=1.00005, bid_h=1.00020, bid_l=0.99968, bid_c=0.99990),
        _candle("2026-08-28T12:00:15Z", bid_o=0.99990, bid_h=1.00031, bid_l=0.99975, bid_c=1.00030),
    ]
    fixed = MODULE.score_path(signal, candles, sl_pips=3.2, tp_pips=2.4)
    atr = MODULE.score_path(signal, candles, sl_pips=4.0, tp_pips=3.0)
    assert fixed["exit_reason"] == "STOP_LOSS"
    assert fixed["realized_pips"] == -3.2
    assert fixed["time_to_stop_seconds"] == 5.0
    assert fixed["mfe_pips"] == 2.0
    assert fixed["mae_pips"] == 3.2
    assert atr["exit_reason"] == "TAKE_PROFIT"
    assert atr["realized_pips"] == 3.0
    specs = {row["policy"]: row for row in MODULE.candidate_specs(signal)}
    assert specs[MODULE.POLICY_ATR_1P0]["sl_pips"] == 4.0
    assert specs[MODULE.POLICY_ATR_1P0]["tp_pips"] == 3.0
    assert specs[MODULE.POLICY_ATR_1P2]["sl_pips"] == 4.8
    assert specs[MODULE.POLICY_ATR_1P2]["tp_pips"] == 3.6
    assert specs[MODULE.POLICY_VETO_1P0]["vetoed"] is True
    assert specs[MODULE.POLICY_VETO_1P2]["vetoed"] is True
    assert specs[MODULE.POLICY_LOT_HALF]["unit_weight"] == 0.5


def test_join_requires_signal_id_and_signal_sha_and_aggregate_separates_weight() -> None:
    signal = _signal()
    outcomes = [
        {"signal_id": "signal-1", "signal_sha256": "a" * 64, "filled": True},
        {"signal_id": "missing", "signal_sha256": "b" * 64, "filled": True},
    ]
    joined, counts = MODULE.join_filled_signals([signal], outcomes)
    assert len(joined) == 1
    assert counts["unmatched_outcomes"] == 1
    summary = MODULE.aggregate([
        {
            "vetoed": False,
            "filled": True,
            "realized_pips": -3.2,
            "weighted_pips": -1.6,
            "exit_reason": "STOP_LOSS",
            "time_to_stop_seconds": 5.0,
            "mfe_pips": 2.0,
            "mae_pips": 3.2,
        }
    ])
    assert summary["net_pips"] == -3.2
    assert summary["weighted_net_pips"] == -1.6
    assert summary["stop_hit_count"] == 1


def test_analysis_preserves_zero_authority_and_inventory_separation() -> None:
    signal = _signal()
    candles = [
        _candle("2026-08-28T12:00:05Z", bid_o=0.99994, bid_h=1.00010, bid_l=0.99991, bid_c=1.00005),
        _candle("2026-08-28T12:00:10Z", bid_o=1.00005, bid_h=1.00020, bid_l=0.99968, bid_c=0.99990),
    ]
    outcome = {
        "signal_id": "signal-1",
        "signal_sha256": "a" * 64,
        "filled": True,
        "exit_reason": "STOP_LOSS",
        "realized_pips": -3.2,
        "truth_chunk_sha256": ["c" * 64],
    }
    result = MODULE.analyze(
        signals=[signal],
        outcomes=[outcome],
        truth_fetcher=lambda _signal, _outcome: (candles, ["c" * 64]),
        generated_at_utc=datetime(2026, 8, 28, tzinfo=timezone.utc),
        signal_ledger_sha256="d" * 64,
        outcome_ledger_sha256="e" * 64,
        release_manifest={"commit": "f" * 40, "source_bundle_sha256": "1" * 64},
        runtime_status={"external_order_attempts": 0, "external_orders": 0, "pid": 123},
    )
    assert result["truth_hash_matches"] == 1
    assert result["authority"]["broker_mutation"] is False
    assert result["authority"]["automatic_parameter_change_allowed"] is False
    assert result["authority"]["inventory_control_evaluated"] is False
