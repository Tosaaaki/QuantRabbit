from __future__ import annotations

import json
from pathlib import Path

from quant_rabbit.fast_bot import SIGNAL_CONTRACT
from quant_rabbit.fast_bot_corrective_challenger import (
    ARM_ORDER,
    ARM_ORDER_V3,
    ARM_ORDER_V4,
    ROW_CONTRACT,
    canonical_sha,
    load_config,
    seal,
)
from quant_rabbit.fast_bot_knowledge import (
    EPISODE_CONTRACT,
    KNOWLEDGE_CONTRACT,
    _adverse_conditions,
    run_fast_bot_knowledge,
)
from quant_rabbit.fast_bot_truth import OUTCOME_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "fast_bot_corrective_challenger_v1.json"
CONFIG_V2 = ROOT / "config" / "fast_bot_corrective_challenger_v2.json"
CONFIG_V3 = ROOT / "config" / "fast_bot_corrective_challenger_v3.json"
CONFIG_V4 = ROOT / "config" / "fast_bot_corrective_challenger_v4.json"


def _signal(signal_id: str = "a" * 24) -> dict:
    body = {
        "contract": SIGNAL_CONTRACT,
        "schema_version": 3,
        "signal_id": signal_id,
        "generated_at_utc": "2026-08-31T09:00:00+00:00",
        "pair": "USD_JPY",
        "side": "SHORT",
        "method": "TREND_CONTINUATION",
        "horizon_lane": "M1_EXECUTION_15M_HOLD",
        "spread_pips": 0.8,
        "m5_atr_pips": 4.2,
        "regime_score": 2.0,
    }
    return {**body, "signal_sha256": canonical_sha(body)}


def _outcome(signal: dict) -> dict:
    return seal(
        {
            "contract": OUTCOME_CONTRACT,
            "schema_version": 3,
            "signal_id": signal["signal_id"],
            "signal_sha256": signal["signal_sha256"],
            "signal_generated_at_utc": signal["generated_at_utc"],
            "resolved_at_utc": "2026-08-31T09:16:30+00:00",
            "pair": signal["pair"],
            "side": signal["side"],
            "method": signal["method"],
            "filled": True,
            "realized_pips": -4.0,
        }
    )


def _challenger_rows(
    signal: dict,
    config_sha: str,
    *,
    arm_order: tuple[str, ...] = ARM_ORDER,
) -> list[dict]:
    rows = []
    for arm in arm_order:
        cooldown = arm == "LANE_COOLDOWN"
        rows.append(
            seal(
                {
                    "contract": ROW_CONTRACT,
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "arm_id": arm,
                    "signal_id": signal["signal_id"],
                    "signal_sha256": signal["signal_sha256"],
                    "evaluated_at_utc": "2026-08-31T09:16:31+00:00",
                    "generated_at_utc": signal["generated_at_utc"],
                    "pair": signal["pair"],
                    "side": signal["side"],
                    "strategy": signal["method"],
                    "regime_bucket": "REGIME_POSITIVE",
                    "atr_bucket": "ATR_4_TO_LT_5",
                    "spread_bucket": "0.8P",
                    "filled": not cooldown,
                    "vetoed": cooldown,
                    "veto_reason": "LANE_RESERVED" if cooldown else None,
                    "fill_at_utc": None if cooldown else "2026-08-31T09:00:05+00:00",
                    "exit_at_utc": None if cooldown else "2026-08-31T09:02:00+00:00",
                    "exit_reason": "VETOED" if cooldown else "STOP_LOSS_GAP",
                    "after_cost_net_pips": 0.0 if cooldown else -4.0,
                    "realized_pips": 0.0 if cooldown else -4.0,
                    "mfe_pips": 0.0,
                    "mae_pips": 0.0 if cooldown else 4.0,
                    "stop_loss_pips": 3.2,
                    "take_profit_pips": 2.4,
                }
            )
        )
    return rows


def test_v3_knowledge_consumes_complete_strict_confirmation_arm_set(
    tmp_path: Path,
) -> None:
    _, config_sha = load_config(CONFIG_V3)
    signal = _signal("c" * 24)
    signal_body = {key: value for key, value in signal.items() if key != "signal_sha256"}
    signal_body["entry_confirmation"] = {
        "contract": "QR_FAST_BOT_ENTRY_CONFIRMATION_V1",
        "policy": "EXECUTION_M1_MUST_BE_TRIGGERED",
        "m1_readiness": "TRIGGERED",
        "m5_readiness": "ARMED",
        "m1_triggered": True,
    }
    signal = {**signal_body, "signal_sha256": canonical_sha(signal_body)}
    shadow = tmp_path / "shadow.jsonl"
    outcome = tmp_path / "outcome.jsonl"
    challenger = tmp_path / "challenger.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    knowledge = tmp_path / "knowledge.jsonl"
    scorecard = tmp_path / "scorecard.json"
    _write_jsonl(shadow, [signal])
    _write_jsonl(outcome, [_outcome(signal)])
    _write_jsonl(
        challenger,
        _challenger_rows(signal, config_sha, arm_order=ARM_ORDER_V3),
    )

    result = run_fast_bot_knowledge(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome,
        challenger_ledger_path=challenger,
        config_path=CONFIG_V3,
        episode_ledger_path=episodes,
        knowledge_ledger_path=knowledge,
        scorecard_path=scorecard,
    )

    assert result["resolved_episode_count"] == 1
    assert result["missing_complete_counterfactual_count"] == 0
    card = json.loads(scorecard.read_text())
    assert "M1_TRIGGERED_ONLY" in card["arm_metrics"]


def test_v4_knowledge_consumes_complete_method_aware_entry_arm_set(
    tmp_path: Path,
) -> None:
    _, config_sha = load_config(CONFIG_V4)
    signal = _signal("d" * 24)
    signal_body = {
        key: value for key, value in signal.items() if key != "signal_sha256"
    }
    signal = {**signal_body, "signal_sha256": canonical_sha(signal_body)}
    shadow = tmp_path / "shadow.jsonl"
    outcome = tmp_path / "outcome.jsonl"
    challenger = tmp_path / "challenger.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    knowledge = tmp_path / "knowledge.jsonl"
    scorecard = tmp_path / "scorecard.json"
    _write_jsonl(shadow, [signal])
    _write_jsonl(outcome, [_outcome(signal)])
    _write_jsonl(
        challenger,
        _challenger_rows(signal, config_sha, arm_order=ARM_ORDER_V4),
    )

    result = run_fast_bot_knowledge(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome,
        challenger_ledger_path=challenger,
        config_path=CONFIG_V4,
        episode_ledger_path=episodes,
        knowledge_ledger_path=knowledge,
        scorecard_path=scorecard,
    )

    assert result["resolved_episode_count"] == 1
    assert result["missing_complete_counterfactual_count"] == 0
    card = json.loads(scorecard.read_text())
    assert "CAUSAL_ENTRY_EDGE_ONLY" in card["arm_metrics"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_episode_links_one_trade_and_classifies_paired_lane_failure(
    tmp_path: Path,
) -> None:
    _, config_sha = load_config(CONFIG)
    signal = _signal()
    shadow = tmp_path / "shadow.jsonl"
    outcome = tmp_path / "outcome.jsonl"
    challenger = tmp_path / "challenger.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    knowledge = tmp_path / "knowledge.jsonl"
    scorecard = tmp_path / "scorecard.json"
    _write_jsonl(shadow, [signal])
    _write_jsonl(outcome, [_outcome(signal)])
    _write_jsonl(challenger, _challenger_rows(signal, config_sha))

    result = run_fast_bot_knowledge(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome,
        challenger_ledger_path=challenger,
        config_path=CONFIG,
        episode_ledger_path=episodes,
        knowledge_ledger_path=knowledge,
        scorecard_path=scorecard,
    )
    episode = json.loads(episodes.read_text().splitlines()[0])
    assert episode["contract"] == EPISODE_CONTRACT
    assert episode["trade_id"] == signal["signal_id"]
    assert episode["raw_source_refs"]["signal_sha256"] == signal["signal_sha256"]
    assert episode["outcome"]["mfe_pips"] == 0.0
    assert episode["outcome"]["mae_pips"] == 4.0
    assert episode["outcome"]["stop_gap_slippage_like_pips"] == 0.8
    assert episode["expectation_gap"]["actual_minus_planned_take_profit_pips"] == -6.4
    assert episode["failure_classification"]["layer"] == "PORTFOLIO_CONCURRENCY_LAYER"
    assert episode["failure_classification"]["supporting_arm_id"] == "LANE_COOLDOWN"
    assert result["target_net_delta_pips"] == 4.0
    assert result["external_orders"] == 0


def test_run_is_idempotent_and_keeps_adoption_owner_gated(tmp_path: Path) -> None:
    _, config_sha = load_config(CONFIG)
    signal = _signal("b" * 24)
    shadow = tmp_path / "shadow.jsonl"
    outcome = tmp_path / "outcome.jsonl"
    challenger = tmp_path / "challenger.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    knowledge = tmp_path / "knowledge.jsonl"
    scorecard = tmp_path / "scorecard.json"
    _write_jsonl(shadow, [signal])
    _write_jsonl(outcome, [_outcome(signal)])
    _write_jsonl(challenger, _challenger_rows(signal, config_sha))

    kwargs = dict(
        shadow_ledger_path=shadow,
        outcome_ledger_path=outcome,
        challenger_ledger_path=challenger,
        config_path=CONFIG,
        episode_ledger_path=episodes,
        knowledge_ledger_path=knowledge,
        scorecard_path=scorecard,
    )
    first = run_fast_bot_knowledge(**kwargs)
    second = run_fast_bot_knowledge(**kwargs)
    assert first["new_episode_count"] == 1
    assert first["new_knowledge_record_count"] == 1
    assert second["new_episode_count"] == 0
    assert second["new_knowledge_record_count"] == 0
    assert len(episodes.read_text().splitlines()) == 1
    assert len(knowledge.read_text().splitlines()) == 1
    knowledge_row = json.loads(knowledge.read_text().splitlines()[0])
    assert knowledge_row["contract"] == KNOWLEDGE_CONTRACT
    assert knowledge_row["adoption_state"] == "NOT_ADOPTED_OWNER_REVIEW_REQUIRED"
    assert knowledge_row["automatic_adoption_allowed"] is False
    card = json.loads(scorecard.read_text())
    assert card["assessment_status"] == "COLLECTING_FORWARD_EVIDENCE"
    assert card["once_only_activation_ready"] is False
    assert card["positive_profitability_claim_allowed"] is False


def test_v2_knowledge_stops_on_predeclared_dual_metric_futility() -> None:
    config, _ = load_config(CONFIG_V2)
    baseline = {
        "filled_count": 10,
        "net_pips": -10.0,
        "profit_factor": 0.5,
        "max_consecutive_losses": 4,
        "mean_mae_pips": 2.5,
    }
    target = {
        "filled_count": 10,
        "net_pips": -10.0,
        "profit_factor": 0.5,
        "max_consecutive_losses": 4,
        "mean_mae_pips": 2.5,
    }
    result = _adverse_conditions(
        baseline,
        target,
        preregistration=config["preregistration"],
    )
    assert result["early_futility_floor_met"] is True
    assert result["dual_metric_futility_after_early_floor"] is True
    assert result["stop_condition_observed"] is True


def test_early_futility_floor_is_not_itself_a_stop_condition() -> None:
    config, _ = load_config(CONFIG_V2)
    baseline = {
        "filled_count": 10,
        "net_pips": -20.0,
        "profit_factor": 0.6,
        "max_consecutive_losses": 4,
        "mean_mae_pips": 3.0,
    }
    target = {
        "filled_count": 10,
        "net_pips": -10.0,
        "profit_factor": 0.5,
        "max_consecutive_losses": 4,
        "mean_mae_pips": 3.0,
    }

    result = _adverse_conditions(
        baseline,
        target,
        preregistration=config["preregistration"],
    )

    assert result["early_futility_floor_met"] is True
    assert result["dual_metric_futility_after_early_floor"] is False
    assert result["stop_condition_observed"] is False
