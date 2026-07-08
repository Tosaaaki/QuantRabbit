from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_LOCAL = ROOT / ".env.local"
DEFAULT_LEGACY_ARCHIVE = Path(
    os.environ.get(
        "QR_LEGACY_ARCHIVE",
        "/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z",
    )
)
DEFAULT_HISTORY_DB = ROOT / "data" / "legacy_history.db"
DEFAULT_IMPORT_REPORT = ROOT / "docs" / "legacy_import_report.md"
DEFAULT_STRATEGY_PROFILE = ROOT / "data" / "strategy_profile.json"
DEFAULT_STRATEGY_REPORT = ROOT / "docs" / "strategy_mining_report.md"
DEFAULT_MARKET_STORY_PROFILE = ROOT / "data" / "market_story_profile.json"
DEFAULT_MARKET_STORY_REPORT = ROOT / "docs" / "market_story_report.md"
DEFAULT_MARKET_STATUS = ROOT / "data" / "market_status.json"
DEFAULT_MARKET_STATUS_REPORT = ROOT / "docs" / "market_status_report.md"
DEFAULT_CAMPAIGN_PLAN = ROOT / "data" / "daily_campaign_plan.json"
DEFAULT_CAMPAIGN_REPORT = ROOT / "docs" / "daily_campaign_report.md"
DEFAULT_TRADER_SETTINGS = ROOT / "data" / "trader_settings.json"
DEFAULT_TRADER_OVERRIDES = ROOT / "data" / "trader_overrides.json"
DEFAULT_ORDER_INTENTS = ROOT / "data" / "order_intents.json"
DEFAULT_ORDER_INTENT_REPORT = ROOT / "docs" / "order_intents_report.md"
DEFAULT_RECEIPT_PROMOTION_REPORT = ROOT / "docs" / "receipt_promotion_report.md"
DEFAULT_LIVE_ORDER_REQUEST = ROOT / "data" / "live_order_request.json"
DEFAULT_LIVE_ORDER_STAGE_REPORT = ROOT / "docs" / "live_order_stage_report.md"
DEFAULT_TRADER_DECISION = ROOT / "data" / "trader_decision.json"
DEFAULT_TRADER_DECISION_REPORT = ROOT / "docs" / "trader_decision_report.md"
DEFAULT_CODEX_TRADER_DECISION_RESPONSE = ROOT / "data" / "codex_trader_decision_response.json"
DEFAULT_TRADER_DECISION_DRAFT_REPORT = ROOT / "docs" / "trader_decision_draft_report.md"
DEFAULT_MARKET_READ_PREDICTIONS = ROOT / "data" / "market_read_predictions.jsonl"
DEFAULT_MARKET_READ_SCORE_REPORT = ROOT / "docs" / "market_read_score_report.md"
DEFAULT_TRADER_PROMPTS_DIR = ROOT / "docs" / "trader_prompts"
DEFAULT_POSITION_MANAGEMENT = ROOT / "data" / "position_management.json"
DEFAULT_POSITION_MANAGEMENT_REPORT = ROOT / "docs" / "position_management_report.md"
DEFAULT_POSITION_GUARDIAN_MANAGEMENT = ROOT / "data" / "position_guardian_management.json"
DEFAULT_POSITION_GUARDIAN_MANAGEMENT_REPORT = ROOT / "docs" / "position_guardian_management_report.md"
DEFAULT_POSITION_GUARDIAN_EXECUTION = ROOT / "data" / "position_guardian_execution.json"
DEFAULT_POSITION_GUARDIAN_HEARTBEAT = ROOT / "data" / "position_guardian.json"
DEFAULT_GUARDIAN_EVENTS = ROOT / "data" / "guardian_events.json"
DEFAULT_GUARDIAN_EVENT_STATE = ROOT / "data" / "guardian_event_state.json"
DEFAULT_GUARDIAN_ESCALATION = ROOT / "data" / "guardian_escalation.json"
DEFAULT_GUARDIAN_TRIGGER_CONTRACT = ROOT / "data" / "guardian_trigger_contract.json"
DEFAULT_GUARDIAN_TRIGGER_CONTRACT_REPORT = ROOT / "docs" / "guardian_trigger_contract_report.md"
DEFAULT_GUARDIAN_ACTION_RECEIPT = ROOT / "data" / "guardian_action_receipt.json"
DEFAULT_GUARDIAN_EVENT_REPORT = ROOT / "docs" / "guardian_event_report.md"
DEFAULT_GUARDIAN_ACTION_REVIEW = ROOT / "docs" / "guardian_action_review.md"
DEFAULT_GUARDIAN_WAKE_DISPATCHER_STATE = ROOT / "data" / "guardian_wake_dispatcher_state.json"
DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION = ROOT / "data" / "guardian_receipt_consumption.json"
DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION_REPORT = ROOT / "docs" / "guardian_receipt_consumption_report.md"
DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW = ROOT / "data" / "guardian_receipt_operator_review.json"
DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW_REPORT = ROOT / "docs" / "guardian_receipt_operator_review_report.md"
DEFAULT_QR_TRADER_RUN_WATCHDOG = ROOT / "data" / "qr_trader_run_watchdog.json"
DEFAULT_QR_TRADER_RUN_WATCHDOG_REPORT = ROOT / "docs" / "qr_trader_run_watchdog_report.md"
DEFAULT_TRADER_SUPPORT_BOT = ROOT / "data" / "trader_support_bot.json"
DEFAULT_TRADER_SUPPORT_BOT_REPORT = ROOT / "docs" / "trader_support_bot_report.md"
DEFAULT_TRADER_REPAIR_ORCHESTRATOR = ROOT / "data" / "trader_repair_orchestrator.json"
DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT = ROOT / "docs" / "trader_repair_orchestrator_report.md"
DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR = ROOT / "data" / "trader_goal_loop_orchestrator.json"
DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR_REPORT = ROOT / "docs" / "trader_goal_loop_orchestrator_report.md"
DEFAULT_ACTIVE_TRADER_CONTRACT = ROOT / "data" / "active_trader_contract.json"
DEFAULT_ACTIVE_TRADER_CONTRACT_REPORT = ROOT / "docs" / "active_trader_contract.md"
DEFAULT_ACTIVE_OPPORTUNITY_BOARD = ROOT / "data" / "active_opportunity_board.json"
DEFAULT_ACTIVE_OPPORTUNITY_BOARD_REPORT = ROOT / "docs" / "active_opportunity_board.md"
DEFAULT_HARVEST_LIVE_GRADE_PATH = ROOT / "data" / "harvest_live_grade_path.json"
DEFAULT_AS_PROOF_PACK_QUEUE = ROOT / "data" / "as_proof_pack_queue.json"
DEFAULT_AS_LANE_CANDIDATE_BOARD = ROOT / "data" / "as_lane_candidate_board.json"
DEFAULT_PORTFOLIO_4X_PATH_PLANNER = ROOT / "data" / "portfolio_4x_path_planner.json"
DEFAULT_PROFIT_CAPTURE_BOT = ROOT / "data" / "profit_capture_bot.json"
DEFAULT_PROFIT_CAPTURE_BOT_REPORT = ROOT / "docs" / "profit_capture_bot_report.md"
DEFAULT_POSITION_EXECUTION = ROOT / "data" / "position_execution.json"
DEFAULT_POSITION_EXECUTION_REPORT = ROOT / "docs" / "position_execution_report.md"
DEFAULT_ADVERSE_PARTIAL_CLOSE = ROOT / "data" / "adverse_partial_close.json"
DEFAULT_ADVERSE_PARTIAL_CLOSE_REPORT = ROOT / "docs" / "adverse_partial_close_report.md"
DEFAULT_PROFIT_PARTIAL_CLOSE = ROOT / "data" / "profit_partial_close.json"
DEFAULT_PROFIT_PARTIAL_CLOSE_REPORT = ROOT / "docs" / "profit_partial_close_report.md"
DEFAULT_PROFIT_PARTIAL_CLOSE_STATE = ROOT / "data" / "profit_partial_close_state.json"
DEFAULT_PREDICTIVE_LIMIT_ORDERS = ROOT / "data" / "predictive_limit_orders.json"
DEFAULT_BROKER_SNAPSHOT = ROOT / "data" / "broker_snapshot.json"
DEFAULT_OPERATOR_MANUAL_POSITIONS = ROOT / "data" / "operator_manual_positions.json"
DEFAULT_WEBULL_ENV_CHECK = ROOT / "data" / "webull_env_check.json"
DEFAULT_WEBULL_ENV_CHECK_REPORT = ROOT / "docs" / "webull_env_check_report.md"
DEFAULT_WEBULL_ACCOUNT_SNAPSHOT = ROOT / "data" / "webull_account_snapshot.json"
DEFAULT_WEBULL_ACCOUNT_SNAPSHOT_REPORT = ROOT / "docs" / "webull_account_snapshot_report.md"
DEFAULT_WEBULL_STOCK_ORDER_REQUEST = ROOT / "data" / "webull_stock_order_request.json"
DEFAULT_WEBULL_STOCK_ORDER_STAGE_REPORT = ROOT / "docs" / "webull_stock_order_stage_report.md"
DEFAULT_DAILY_TARGET_STATE = ROOT / "data" / "daily_target_state.json"
DEFAULT_DAILY_TARGET_REPORT = ROOT / "docs" / "daily_target_report.md"
DEFAULT_CAPITAL_FLOWS = ROOT / "data" / "capital_flows.json"
DEFAULT_CAPITAL_FLOW_REPORT = ROOT / "docs" / "capital_flow_report.md"
DEFAULT_REPLAY_BACKTEST = ROOT / "data" / "replay_backtest.json"
DEFAULT_REPLAY_BACKTEST_REPORT = ROOT / "docs" / "replay_backtest_report.md"
DEFAULT_AI_TEST_BOT_BACKTEST = ROOT / "data" / "ai_test_bot_backtest.json"
DEFAULT_AI_TEST_BOT_BACKTEST_REPORT = ROOT / "docs" / "ai_test_bot_backtest_report.md"
DEFAULT_OUTCOME_MART = ROOT / "data" / "outcome_mart.json"
DEFAULT_OUTCOME_MART_REPORT = ROOT / "docs" / "outcome_mart_report.md"
DEFAULT_AI_ATTACK_ADVICE = ROOT / "data" / "ai_attack_advice.json"
DEFAULT_AI_ATTACK_ADVICE_REPORT = ROOT / "docs" / "ai_attack_advice_report.md"
DEFAULT_CAPTURE_ECONOMICS = ROOT / "data" / "capture_economics.json"
DEFAULT_CAPTURE_ECONOMICS_REPORT = ROOT / "docs" / "capture_economics_report.md"
DEFAULT_PROFITABILITY_ACCEPTANCE = ROOT / "data" / "profitability_acceptance.json"
DEFAULT_PROFITABILITY_ACCEPTANCE_REPORT = ROOT / "docs" / "profitability_acceptance_report.md"
DEFAULT_PAYOFF_SHAPE_DIAGNOSIS = ROOT / "data" / "payoff_shape_diagnosis.json"
DEFAULT_PAYOFF_SHAPE_DIAGNOSIS_REPORT = ROOT / "docs" / "payoff_shape_diagnosis_report.md"
DEFAULT_MONTH_SCALE_TP_REPLAY_RESIDUALS = ROOT / "data" / "month_scale_tp_replay_residuals.json"
DEFAULT_MONTH_SCALE_TP_REPLAY_RESIDUALS_REPORT = ROOT / "docs" / "month_scale_tp_replay_residuals.md"
DEFAULT_MONTH_SCALE_RESIDUAL_FAMILY_TABLE = ROOT / "data" / "month_scale_residual_family_table.json"
DEFAULT_MONTH_SCALE_RESIDUAL_FAMILY_TABLE_REPORT = ROOT / "docs" / "month_scale_residual_family_table.md"
DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING = (
    ROOT / "logs" / "reports" / "forecast_improvement" / "oanda_universal_rotation_mining_latest.json"
)
DEFAULT_BIDASK_REPLAY_VALIDATION = (
    ROOT / "logs" / "reports" / "forecast_improvement" / "oanda_history_replay_validate_latest.json"
)
DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES = (
    ROOT / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
)


def effective_oanda_universal_rotation_path(latest_path: Path, packaged_path: Path) -> Path:
    """Choose the OANDA rotation evidence file without discarding preserved scope.

    Focused mining runs can update the latest log with a narrow pair subset. The
    packaged runtime artifact may already have merged that focused run while
    preserving broader validated campaign evidence. Prefer it only when it
    explicitly says preserved scope was used and is at least as fresh as latest.
    """

    if _prefer_packaged_oanda_universal_rotation(latest_path, packaged_path):
        return packaged_path
    if latest_path.exists():
        return latest_path
    if packaged_path.exists():
        return packaged_path
    return latest_path


def _prefer_packaged_oanda_universal_rotation(latest_path: Path, packaged_path: Path) -> bool:
    if not packaged_path.exists():
        return False
    if not latest_path.exists():
        return True
    packaged = _load_json_object(packaged_path)
    if packaged is None:
        return False
    if (
        packaged.get("campaign_firepower_preserved_from_existing") is not True
        and packaged.get("scope_metadata_preserved_from_existing") is not True
    ):
        return False
    if not _packaged_source_matches_latest(packaged, packaged_path, latest_path):
        return False
    latest = _load_json_object(latest_path)
    if latest is None:
        return False
    latest_generated_at = _parse_generated_at(latest.get("generated_at_utc"))
    packaged_generated_at = _parse_generated_at(packaged.get("generated_at_utc"))
    if latest_generated_at is None or packaged_generated_at is None:
        return False
    if _packaged_oanda_scope_covers_latest(packaged, latest):
        return True
    return packaged_generated_at >= latest_generated_at


def _packaged_oanda_scope_covers_latest(packaged: dict[str, Any], latest: dict[str, Any]) -> bool:
    packaged_pairs = _oanda_scope_selected_pairs(packaged)
    latest_pairs = _oanda_scope_selected_pairs(latest)
    if packaged_pairs and latest_pairs:
        return packaged_pairs.issuperset(latest_pairs) and len(packaged_pairs) > len(latest_pairs)

    packaged_metrics = _oanda_scope_metrics(packaged)
    latest_metrics = _oanda_scope_metrics(latest)
    comparable_keys = sorted(set(packaged_metrics) & set(latest_metrics))
    if not comparable_keys:
        return False
    if not any(key in comparable_keys for key in ("selected_pair_count", "history_pairs", "history_files")):
        return False
    if any(packaged_metrics[key] < latest_metrics[key] for key in comparable_keys):
        return False
    return any(packaged_metrics[key] > latest_metrics[key] for key in comparable_keys)


def _oanda_scope_selected_pairs(payload: dict[str, Any]) -> set[str]:
    for container in _oanda_scope_containers(payload):
        selection = container.get("history_pair_selection")
        if not isinstance(selection, dict):
            continue
        pairs = selection.get("selected_pairs")
        if not isinstance(pairs, list):
            continue
        selected_pairs = {str(pair).strip().upper() for pair in pairs if str(pair).strip()}
        if selected_pairs:
            return selected_pairs
    return set()


def _oanda_scope_metrics(payload: dict[str, Any]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for container in _oanda_scope_containers(payload):
        selection = container.get("history_pair_selection")
        if isinstance(selection, dict) and "selected_pair_count" not in metrics:
            selected_count = _int_or_none(selection.get("selected_pair_count"))
            if selected_count is None:
                selected_count = len(_oanda_scope_selected_pairs(payload)) or None
            if selected_count is not None:
                metrics["selected_pair_count"] = selected_count
        for key in (
            "history_files",
            "history_files_discovered",
            "history_pairs",
            "history_pairs_discovered",
            "scored_outcomes",
            "inversion_scored_outcomes",
            "high_precision_multi_confluence_count",
            "high_precision_pair_confluence_count",
            "qualified_multi_confluence_count",
            "qualified_pair_confluence_count",
        ):
            value = _int_or_none(container.get(key))
            if value is not None and key not in metrics:
                metrics[key] = value
    return metrics


def _oanda_scope_containers(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary, payload
    return (payload,)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _packaged_source_matches_latest(
    packaged: dict[str, Any],
    packaged_path: Path,
    latest_path: Path,
) -> bool:
    source_report = packaged.get("source_report")
    if not isinstance(source_report, str) or not source_report.strip():
        return False
    source_path = Path(source_report)
    if not source_path.is_absolute():
        source_path = _repo_root_for_packaged_oanda_path(packaged_path) / source_path
    try:
        return source_path.resolve(strict=False) == latest_path.resolve(strict=False)
    except OSError:
        return source_path == latest_path


def _repo_root_for_packaged_oanda_path(packaged_path: Path) -> Path:
    if packaged_path.parent.name == "quant_rabbit" and packaged_path.parent.parent.name == "src":
        return packaged_path.parent.parent.parent
    return packaged_path.parent


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_generated_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
DEFAULT_EXECUTION_TIMING_AUDIT = ROOT / "data" / "execution_timing_audit.json"
DEFAULT_EXECUTION_TIMING_AUDIT_REPORT = ROOT / "docs" / "execution_timing_audit_report.md"
DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE = ROOT / "data" / "tp_progress_harvest_gate_evidence.json"
DEFAULT_TP_PROGRESS_HARVEST_GATE_EVIDENCE_REPORT = ROOT / "docs" / "tp_progress_harvest_gate_evidence_report.md"
DEFAULT_MANUAL_HISTORY_2025 = ROOT / "data" / "manual_history_2025_mining.json"
DEFAULT_OPERATOR_PRECEDENT_AUDIT = ROOT / "data" / "operator_precedent_audit.json"
DEFAULT_OPERATOR_PRECEDENT_AUDIT_REPORT = ROOT / "docs" / "operator_precedent_audit_report.md"
DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT = ROOT / "data" / "manual_market_context_audit.json"
DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT_REPORT = ROOT / "docs" / "manual_market_context_audit_report.md"
DEFAULT_GPT_TRADER_DECISION = ROOT / "data" / "gpt_trader_decision.json"
DEFAULT_GPT_TRADER_DECISION_REPORT = ROOT / "docs" / "gpt_trader_decision_report.md"
DEFAULT_COVERAGE_OPTIMIZATION = ROOT / "data" / "coverage_optimization.json"
DEFAULT_COVERAGE_OPTIMIZATION_REPORT = ROOT / "docs" / "coverage_optimization_report.md"
DEFAULT_POST_TRADE_LEARNING = ROOT / "data" / "post_trade_learning.json"
DEFAULT_POST_TRADE_LEARNING_REPORT = ROOT / "docs" / "post_trade_learning_report.md"
DEFAULT_LEARNING_AUDIT = ROOT / "data" / "learning_audit.json"
DEFAULT_LEARNING_AUDIT_REPORT = ROOT / "docs" / "learning_audit_report.md"
DEFAULT_FORECAST_HISTORY = ROOT / "data" / "forecast_history.jsonl"
DEFAULT_PROJECTION_LEDGER = ROOT / "data" / "projection_ledger.jsonl"
DEFAULT_ENTRY_THESIS_LEDGER = ROOT / "data" / "entry_thesis_ledger.jsonl"
DEFAULT_MEMORY_HEALTH = ROOT / "data" / "memory_health.json"
DEFAULT_MEMORY_HEALTH_REPORT = ROOT / "docs" / "memory_health_report.md"
DEFAULT_SELF_IMPROVEMENT_AUDIT = ROOT / "data" / "self_improvement_audit.json"
DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT = ROOT / "docs" / "self_improvement_audit_report.md"
DEFAULT_SELF_IMPROVEMENT_HISTORY_DB = ROOT / "data" / "self_improvement_history.db"
DEFAULT_EXECUTION_REPLAY = ROOT / "data" / "execution_replay.json"
DEFAULT_EXECUTION_REPLAY_REPORT = ROOT / "docs" / "execution_replay_report.md"
DEFAULT_EXECUTION_LEDGER_DB = ROOT / "data" / "execution_ledger.db"
DEFAULT_EXECUTION_LEDGER_REPORT = ROOT / "docs" / "execution_ledger_report.md"
DEFAULT_VERIFICATION_LEDGER = ROOT / "data" / "verification_ledger.json"
DEFAULT_VERIFICATION_LEDGER_REPORT = ROOT / "docs" / "verification_ledger_report.md"
DEFAULT_DRY_RUN_CERTIFICATION = ROOT / "data" / "dry_run_certification.json"
DEFAULT_DRY_RUN_CERTIFICATION_REPORT = ROOT / "docs" / "dry_run_certification_report.md"
DEFAULT_COMPLETION_STATUS = ROOT / "data" / "completion_status.json"
DEFAULT_COMPLETION_STATUS_REPORT = ROOT / "docs" / "completion_status_report.md"
DEFAULT_PAIR_CHARTS = ROOT / "data" / "pair_charts.json"
DEFAULT_PAIR_CHARTS_REPORT = ROOT / "docs" / "pair_charts_report.md"
DEFAULT_CONTEXT_ASSET_CHARTS = ROOT / "data" / "context_asset_charts.json"
DEFAULT_CONTEXT_ASSET_CHARTS_REPORT = ROOT / "docs" / "context_asset_charts_report.md"
DEFAULT_BROKER_INSTRUMENTS = ROOT / "data" / "broker_instruments.json"
DEFAULT_BROKER_INSTRUMENTS_REPORT = ROOT / "docs" / "broker_instruments_report.md"
DEFAULT_CROSS_ASSET_SNAPSHOT = ROOT / "data" / "cross_asset_snapshot.json"
DEFAULT_CROSS_ASSET_REPORT = ROOT / "docs" / "cross_asset_report.md"
DEFAULT_FLOW_SNAPSHOT = ROOT / "data" / "flow_snapshot.json"
DEFAULT_FLOW_REPORT = ROOT / "docs" / "flow_report.md"
DEFAULT_CURRENCY_STRENGTH = ROOT / "data" / "currency_strength.json"
DEFAULT_CURRENCY_STRENGTH_REPORT = ROOT / "docs" / "currency_strength_report.md"
DEFAULT_LEVELS_SNAPSHOT = ROOT / "data" / "levels_snapshot.json"
DEFAULT_LEVELS_REPORT = ROOT / "docs" / "levels_report.md"
DEFAULT_MARKET_CONTEXT_MATRIX = ROOT / "data" / "market_context_matrix.json"
DEFAULT_MARKET_CONTEXT_MATRIX_REPORT = ROOT / "docs" / "market_context_matrix_report.md"
DEFAULT_CALENDAR_SNAPSHOT = ROOT / "data" / "economic_calendar.json"
DEFAULT_CALENDAR_REPORT = ROOT / "docs" / "economic_calendar_report.md"
DEFAULT_COT_SNAPSHOT = ROOT / "data" / "cot_snapshot.json"
DEFAULT_COT_REPORT = ROOT / "docs" / "cot_report.md"
DEFAULT_OPTION_SKEW = ROOT / "data" / "option_skew_snapshot.json"
DEFAULT_OPTION_SKEW_REPORT = ROOT / "docs" / "option_skew_report.md"
DEFAULT_NEWS_SNAPSHOT = ROOT / "data" / "news_items.json"
DEFAULT_NEWS_DIGEST = ROOT / "logs" / "news_digest.md"
DEFAULT_NEWS_FLOW_LOG = ROOT / "logs" / "news_flow_log.md"
DEFAULT_NEWS_HEALTH = ROOT / "data" / "news_health.json"
DEFAULT_NEWS_HEALTH_REPORT = ROOT / "data" / "news_health_report.md"
# Append-only audit trail of every autotrade-cycle outcome. Each line is one
# JSONL event (one cycle) capturing decision, basket, execution result, and
# resulting trader-owned positions. AGENT_CONTRACT §6 / §11 require this
# trail; without it, post-trade review and `mine-strategy` have no signal.
DEFAULT_TRADER_JOURNAL = ROOT / "logs" / "trader_journal.jsonl"
