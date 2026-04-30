from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
DEFAULT_CAMPAIGN_PLAN = ROOT / "data" / "daily_campaign_plan.json"
DEFAULT_CAMPAIGN_REPORT = ROOT / "docs" / "daily_campaign_report.md"
DEFAULT_ORDER_INTENTS = ROOT / "data" / "order_intents.json"
DEFAULT_ORDER_INTENT_REPORT = ROOT / "docs" / "order_intents_report.md"
DEFAULT_RECEIPT_PROMOTION_REPORT = ROOT / "docs" / "receipt_promotion_report.md"
DEFAULT_LIVE_ORDER_REQUEST = ROOT / "data" / "live_order_request.json"
DEFAULT_LIVE_ORDER_STAGE_REPORT = ROOT / "docs" / "live_order_stage_report.md"
DEFAULT_TRADER_DECISION = ROOT / "data" / "trader_decision.json"
DEFAULT_TRADER_DECISION_REPORT = ROOT / "docs" / "trader_decision_report.md"
DEFAULT_POSITION_MANAGEMENT = ROOT / "data" / "position_management.json"
DEFAULT_POSITION_MANAGEMENT_REPORT = ROOT / "docs" / "position_management_report.md"
DEFAULT_POSITION_EXECUTION = ROOT / "data" / "position_execution.json"
DEFAULT_POSITION_EXECUTION_REPORT = ROOT / "docs" / "position_execution_report.md"
DEFAULT_BROKER_SNAPSHOT = ROOT / "data" / "broker_snapshot.json"
