from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.fast_bot_corrective_challenger import (
    ARM_ORDER,
    ARM_ORDER_V3,
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
CONFIG_V2_PATH = ROOT / "config" / "fast_bot_corrective_challenger_v2.json"
CONFIG_V3_PATH = ROOT / "config" / "fast_bot_corrective_challenger_v3.json"


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
        "horizon_lane": "M1_EXECUTION_15M_HOLD",
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


def test_lane_reservation_contract_is_immutable(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    config["inventory"]["reservation_seconds"] = 989
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config))
    with pytest.raises(
        ValueError,
        match="corrective challenger lane reservation contract mismatch",
    ):
        load_config(tampered)


def test_v2_is_a_future_only_single_factor_geometry_cohort() -> None:
    config, _ = load_config(CONFIG_V2_PATH)
    preregistration = config["preregistration"]
    assert config["contract"] == "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V2"
    assert preregistration["target_arm_id"] == "ATR_NORMALIZED_GEOMETRY"
    assert preregistration["single_factor_change"] == "GEOMETRY_ONLY_ATR_RR1_BOUNDED"
    assert preregistration["eligibility_cutoff_utc"] == "2026-09-03T09:30:00Z"
    assert preregistration["selection_evidence"]["profitability_claim_allowed"] is False


def test_v3_is_a_future_only_single_factor_entry_confirmation_cohort() -> None:
    config, _ = load_config(CONFIG_V3_PATH)
    preregistration = config["preregistration"]
    assert config["contract"] == "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V3"
    assert preregistration["target_arm_id"] == "M1_TRIGGERED_ONLY"
    assert preregistration["single_factor_change"] == "ENTRY_CONFIRMATION_ONLY_M1_TRIGGERED"
    assert preregistration["eligibility_cutoff_utc"] == "2026-09-03T14:15:00Z"
    assert preregistration["selection_evidence"]["profitability_claim_allowed"] is False


def test_v3_strict_entry_arm_requires_sealed_m1_trigger_at_emission() -> None:
    config, _ = load_config(CONFIG_V3_PATH)
    signal = _signal(at="2026-09-03T14:16:00+00:00")
    unconfirmed = {
        row["arm_id"]: row
        for row in arm_specs(signal, {"vol_shock": False}, config)
    }
    signal["entry_confirmation"] = {
        "contract": "QR_FAST_BOT_ENTRY_CONFIRMATION_V1",
        "policy": "EXECUTION_M1_MUST_BE_TRIGGERED",
        "m1_readiness": "TRIGGERED",
        "m5_readiness": "ARMED",
        "m1_triggered": True,
    }
    confirmed = {
        row["arm_id"]: row
        for row in arm_specs(signal, {"vol_shock": False}, config)
    }

    assert tuple(unconfirmed) == ARM_ORDER_V3
    assert unconfirmed["M1_TRIGGERED_ONLY"]["vetoed"] is True
    assert unconfirmed["M1_TRIGGERED_ONLY"]["veto_reason"] == "M1_NOT_TRIGGERED_AT_EMISSION"
    assert confirmed["M1_TRIGGERED_ONLY"]["vetoed"] is False
    assert confirmed["M1_TRIGGERED_ONLY"]["veto_reason"] is None


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


def test_lane_cooldown_reserves_only_first_signal_in_horizon() -> None:
    config, _ = load_config(CONFIG_PATH)
    signals = [
        _signal("signal-1", at="2026-08-28T12:00:00+00:00"),
        _signal("signal-2", at="2026-08-28T12:01:00+00:00"),
        _signal("signal-3", at="2026-08-28T12:16:29+00:00"),
        _signal("signal-4", at="2026-08-28T12:16:30+00:00"),
    ]
    features = causal_features(signals, config)
    assert features["signal-1"]["lane_cooldown_veto"] is False
    assert features["signal-2"]["lane_cooldown_veto"] is True
    assert features["signal-2"]["lane_reserved_by_signal_id"] == "signal-1"
    assert features["signal-3"]["lane_cooldown_veto"] is True
    assert features["signal-4"]["lane_cooldown_veto"] is False


def test_arms_are_separate_rr_one_and_veto_worst_lane() -> None:
    config, _ = load_config(CONFIG_PATH)
    signal = _signal(atr=4.0)
    specs = {
        row["arm_id"]: row
        for row in arm_specs(
            signal,
            {"vol_shock": False, "lane_cooldown_veto": True},
            config,
        )
    }
    assert tuple(specs) == ARM_ORDER
    atr = specs["ATR_NORMALIZED_GEOMETRY"]
    assert atr["stop_loss_pips"] == 3.0
    assert atr["take_profit_pips"] == 3.0
    assert atr["take_profit_pips"] / atr["stop_loss_pips"] >= 1.0
    assert specs["COMBINED"]["vetoed"] is True
    assert specs["COMBINED"]["veto_reason"] == "WORST_LANE"
    assert specs["LANE_COOLDOWN"]["vetoed"] is True
    assert specs["LANE_COOLDOWN"]["veto_reason"] == "LANE_RESERVED"
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


def test_build_rows_rounds_short_stop_to_executable_price_tick() -> None:
    config, config_sha = load_config(CONFIG_PATH)
    signal = _signal(at="2026-08-31T15:35:32.625774+00:00")
    signal.update(
        {
            "side": "SHORT",
            "entry": 1.16182,
            "stop_loss_pips": 3.2,
            "take_profit_pips": 2.4,
        }
    )
    outcome = {
        **_outcome(signal),
        "truth_request_from_utc": "2026-08-31T15:35:35+00:00",
        "truth_request_to_utc": "2026-08-31T15:52:00+00:00",
    }
    candles = [
        _candle(
            "2026-08-31T15:35:45Z",
            bid_o=1.16170,
            bid_h=1.16182,
            bid_l=1.16168,
            bid_c=1.16180,
        ),
        _candle(
            "2026-08-31T15:48:40Z",
            bid_o=1.16199,
            bid_h=1.16206,
            bid_l=1.16198,
            bid_c=1.16204,
        ),
    ]

    rows = build_rows(
        signal,
        outcome,
        candles,
        ["c" * 64],
        {"vol_shock": False, "rapid_time_bucket_utc": "NON_SHOCK"},
        config,
        config_sha,
        evaluated_at_utc=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    baseline = next(row for row in rows if row["arm_id"] == "BASELINE")
    assert baseline["exit_reason"] == "STOP_LOSS"
    assert baseline["exit_at_utc"] == "2026-08-31T15:48:40+00:00"
    assert baseline["realized_pips"] == -3.2


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


def test_v2_excludes_pre_cutoff_rows_without_broker_read(tmp_path: Path) -> None:
    signal = _signal(at="2026-09-03T09:30:00+00:00")
    shadow = tmp_path / "shadow.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    ledger = tmp_path / "challenger.jsonl"
    scorecard = tmp_path / "scorecard.json"
    shadow.write_text(json.dumps(signal) + "\n")
    outcomes.write_text(json.dumps(_outcome(signal)) + "\n")

    def forbidden_fetch(*_args: object) -> object:
        raise AssertionError("pre-cutoff row attempted a broker read")

    result = run_incremental(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcomes,
        challenger_ledger_path=ledger,
        scorecard_path=scorecard,
        config_path=CONFIG_V2_PATH,
        client=object(),
        truth_fetcher=forbidden_fetch,
    )
    assert result["pre_cutoff_signal_count"] == 1
    assert result["due_signal_count"] == 0
    assert result["processed_signal_count"] == 0
    assert result["best_so_far"] is None
    assert result["external_order_attempts"] == 0


def test_v2_halts_future_reads_after_dual_metric_futility(tmp_path: Path) -> None:
    signals = []
    outcomes = []
    for index in range(10):
        at = f"2026-09-03T09:{31 + index:02d}:00+00:00"
        signal = _signal(f"signal-{index}", at=at, atr=10.0)
        signals.append(signal)
        outcomes.append(_outcome(signal))
    shadow = tmp_path / "shadow.jsonl"
    outcome_path = tmp_path / "outcomes.jsonl"
    ledger = tmp_path / "challenger.jsonl"
    scorecard = tmp_path / "scorecard.json"
    shadow.write_text("".join(json.dumps(row) + "\n" for row in signals))
    outcome_path.write_text("".join(json.dumps(row) + "\n" for row in outcomes))
    fetch_calls = 0

    def fetch(_client: object, signal: dict, _outcome: dict) -> tuple[list[S5BidAskCandle], list[str]]:
        nonlocal fetch_calls
        fetch_calls += 1
        generated = datetime.fromisoformat(signal["generated_at_utc"])
        first = generated.replace(tzinfo=timezone.utc) if generated.tzinfo is None else generated
        return (
            [
                _candle((first.replace(microsecond=0)).isoformat(), bid_o=0.99994, bid_h=1.00010, bid_l=0.99991, bid_c=1.00005),
                _candle((first.replace(microsecond=0) + timedelta(seconds=5)).isoformat(), bid_o=1.00005, bid_h=1.00020, bid_l=0.99968, bid_c=0.99990),
            ],
            ["c" * 64],
        )

    first = run_incremental(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome_path,
        challenger_ledger_path=ledger,
        scorecard_path=scorecard,
        config_path=CONFIG_V2_PATH,
        max_due=12,
        client=object(),
        truth_fetcher=fetch,
    )
    assert fetch_calls == 10
    assert first["status"] == "HALTED_DUAL_METRIC_FUTILITY"
    assert first["collection_control"]["target_filled_count"] == 10

    second = run_incremental(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome_path,
        challenger_ledger_path=ledger,
        scorecard_path=scorecard,
        config_path=CONFIG_V2_PATH,
        client=object(),
        truth_fetcher=fetch,
    )
    assert fetch_calls == 10
    assert second["status"] == "HALTED_DUAL_METRIC_FUTILITY"
    assert second["broker_http_methods_used"] == []
