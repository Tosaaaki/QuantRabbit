from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.fast_bot_corrective_challenger import (
    ARM_ORDER,
    aggregate,
    arm_specs,
    build_rows,
    causal_features,
    load_config,
    run_incremental,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "fast_bot_corrective_challenger_v1.json"


def _signal(signal_id: str = "signal-1", *, at: str = "2026-08-28T12:00:00+00:00", atr: float = 4.0) -> dict:
    return {
        "signal_id": signal_id,
        "signal_sha256": (signal_id[-1] if signal_id[-1].isalnum() else "a") * 64,
        "generated_at_utc": at,
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "pair": "EUR_USD",
        "side": "LONG",
        "method": "RANGE_ROTATION",
        "entry": 1.0,
        "stop_loss_pips": 3.2,
        "take_profit_pips": 2.4,
        "m5_atr_pips": atr,
        "spread_pips": 0.8,
        "regime_score": -2.0,
    }


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


def _outcome(signal: dict) -> dict:
    return {
        "signal_id": signal["signal_id"],
        "signal_sha256": signal["signal_sha256"],
        "filled": True,
        "exit_reason": "STOP_LOSS",
        "realized_pips": -3.2,
        "truth_request_from_utc": "2026-08-28T12:00:00+00:00",
        "truth_request_to_utc": "2026-08-28T12:16:30+00:00",
        "truth_chunk_sha256": ["c" * 64],
        "contract_sha256": "d" * 64,
    }


def test_causal_shock_uses_strictly_prior_unique_timestamps() -> None:
    config, _ = load_config(CONFIG_PATH)
    signals = [
        _signal("signal-1", at="2026-08-28T11:40:00+00:00", atr=2.0),
        _signal("signal-2", at="2026-08-28T11:45:00+00:00", atr=2.0),
        _signal("signal-3", at="2026-08-28T11:50:00+00:00", atr=2.0),
        _signal("signal-4", at="2026-08-28T12:00:00+00:00", atr=4.0),
        _signal("signal-5", at="2026-08-28T12:00:00+00:00", atr=4.0),
    ]
    features = causal_features(signals, config)
    assert features["signal-4"]["prior_atr_observations"] == 3
    assert features["signal-4"]["prior_atr_median_pips"] == 2.0
    assert features["signal-4"]["causal_atr_ratio"] == 2.0
    assert features["signal-4"]["vol_shock"] is True
    assert features["signal-5"]["prior_atr_observations"] == 3


def test_arms_are_separate_rr_one_and_veto_worst_lane() -> None:
    config, _ = load_config(CONFIG_PATH)
    signal = _signal(atr=4.0)
    specs = {row["arm_id"]: row for row in arm_specs(signal, {"vol_shock": False}, config)}
    assert tuple(specs) == ARM_ORDER
    atr = specs["ATR_NORMALIZED_GEOMETRY"]
    assert atr["stop_loss_pips"] == 3.0
    assert atr["take_profit_pips"] == 3.0
    assert atr["take_profit_pips"] / atr["stop_loss_pips"] >= 1.0
    assert specs["COMBINED"]["vetoed"] is True
    assert specs["COMBINED"]["veto_reason"] == "WORST_LANE"
    assert specs["EURUSD_RANGE_ROTATION_EXCLUDE"]["vetoed"] is True


def test_build_rows_replays_baseline_and_aggregate_requested_metrics() -> None:
    config, config_sha = load_config(CONFIG_PATH)
    signal = _signal()
    candles = [
        _candle("2026-08-28T12:00:05Z", bid_o=0.99994, bid_h=1.00010, bid_l=0.99991, bid_c=1.00005),
        _candle("2026-08-28T12:00:10Z", bid_o=1.00005, bid_h=1.00020, bid_l=0.99968, bid_c=0.99990),
    ]
    rows = build_rows(
        signal,
        _outcome(signal),
        candles,
        ["c" * 64],
        {"vol_shock": False, "rapid_time_bucket_utc": "NON_SHOCK"},
        config,
        config_sha,
        evaluated_at_utc=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )
    baseline = next(row for row in rows if row["arm_id"] == "BASELINE")
    assert baseline["realized_pips"] == -3.2
    assert baseline["after_cost_net_pips"] == -3.2
    assert baseline["external_orders"] == 0
    summary = aggregate([baseline, {**baseline, "signal_id": "signal-2", "generated_at_utc": "2026-08-28T12:01:00+00:00"}])
    assert summary["win_rate"] == 0.0
    assert summary["max_consecutive_losses"] == 2
    assert summary["tail_5pct_loss_pips"] == -3.2
    assert summary["leftover_inventory"] == 0


def test_incremental_ledger_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    signal = _signal()
    outcome = _outcome(signal)
    shadow = tmp_path / "shadow.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    ledger = tmp_path / "challenger.jsonl"
    scorecard = tmp_path / "scorecard.json"
    shadow.write_text(json.dumps(signal) + "\n")
    outcomes.write_text(json.dumps(outcome) + "\n")
    candles = [
        _candle("2026-08-28T12:00:05Z", bid_o=0.99994, bid_h=1.00010, bid_l=0.99991, bid_c=1.00005),
        _candle("2026-08-28T12:00:10Z", bid_o=1.00005, bid_h=1.00020, bid_l=0.99968, bid_c=0.99990),
    ]
    fetch = lambda _client, _signal, _outcome: (candles, ["c" * 64])
    first = run_incremental(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcomes,
        challenger_ledger_path=ledger,
        scorecard_path=scorecard,
        config_path=CONFIG_PATH,
        client=object(),
        truth_fetcher=fetch,
    )
    second = run_incremental(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcomes,
        challenger_ledger_path=ledger,
        scorecard_path=scorecard,
        config_path=CONFIG_PATH,
        client=object(),
        truth_fetcher=fetch,
    )
    assert first["appended_row_count"] == len(ARM_ORDER)
    assert second["appended_row_count"] == 0
    assert len(ledger.read_text().splitlines()) == len(ARM_ORDER)
    card = json.loads(scorecard.read_text())
    assert card["config_sha256"] == first["config_sha256"]
    assert card["external_order_attempts"] == 0
    assert card["external_orders"] == 0
