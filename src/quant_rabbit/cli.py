from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.automation import AutoTradeCycle, DEFAULT_AUTOTRADE_REPORT
from quant_rabbit.ai_test_bot import (
    AITestBotBacktester,
    DEFAULT_MAX_ACTIVE_BUCKETS,
    DEFAULT_MIN_TRAIN_TRADES,
    DEFAULT_SOURCE_TABLES,
    DEFAULT_TRAINING_DAYS,
)
from quant_rabbit.attack_advisor import AttackAdvisor
from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES as DEFAULT_PAIR_CHART_TIMEFRAMES
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.certification import DryRunCertifier
from quant_rabbit.completion import CompletionAuditor
from quant_rabbit.coverage import CoverageOptimizer
from quant_rabbit.execution_replay import ExecutionReplayer
from quant_rabbit.legacy.importer import LegacyImporter
from quant_rabbit.learning import PostTradeLearner
from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    ROOT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_ATTACK_ADVICE_REPORT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_COMPLETION_STATUS,
    DEFAULT_COMPLETION_STATUS_REPORT,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_PAIR_CHARTS_REPORT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_DRY_RUN_CERTIFICATION,
    DEFAULT_DRY_RUN_CERTIFICATION_REPORT,
    DEFAULT_EXECUTION_REPLAY,
    DEFAULT_EXECUTION_REPLAY_REPORT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_HISTORY_DB,
    DEFAULT_IMPORT_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_LEGACY_ARCHIVE,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POST_TRADE_LEARNING,
    DEFAULT_POST_TRADE_LEARNING_REPORT,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_REPLAY_BACKTEST,
    DEFAULT_REPLAY_BACKTEST_REPORT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_STRATEGY_REPORT,
    DEFAULT_TRADER_SETTINGS,
    DEFAULT_TRADER_DECISION,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CROSS_ASSET_REPORT,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_FLOW_REPORT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_CURRENCY_STRENGTH_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_LEVELS_REPORT,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CALENDAR_REPORT,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_COT_REPORT,
    DEFAULT_OPTION_SKEW,
    DEFAULT_OPTION_SKEW_REPORT,
    DEFAULT_NEWS_SNAPSHOT,
    DEFAULT_NEWS_DIGEST,
    DEFAULT_NEWS_FLOW_LOG,
)
from quant_rabbit.gpt_trader import DEFAULT_GPT_MAX_LANES, GPTTraderBrain, StaticTraderProvider
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS_ARG
from quant_rabbit.replay import ReplayBacktester
from quant_rabbit.risk import RiskEngine, RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.strategy.ensemble import CampaignPlanner
from quant_rabbit.strategy.intent_generator import IntentGenerator
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.miner import StrategyMiner
from quant_rabbit.strategy.profile import StrategyProfile
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter
from quant_rabbit.target import DailyTargetLedger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qr-vnext")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-legacy", help="Import legacy archive into the vNext history DB.")
    p_import.add_argument("--archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    p_import.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_import.add_argument("--report", type=Path, default=DEFAULT_IMPORT_REPORT)

    p_snapshot = sub.add_parser("broker-snapshot", help="Read current broker truth without placing orders.")
    p_snapshot.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_snapshot.add_argument("--output", type=Path, default=None)

    p_charts = sub.add_parser("pair-charts", help="Compute multi-timeframe indicator scores per pair.")
    p_charts.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_charts.add_argument("--timeframes", default=",".join(DEFAULT_PAIR_CHART_TIMEFRAMES))
    p_charts.add_argument("--count", type=int, default=200)
    p_charts.add_argument("--output", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_charts.add_argument("--report", type=Path, default=DEFAULT_PAIR_CHARTS_REPORT)

    p_cross = sub.add_parser("cross-asset-snapshot", help="Cross-asset/inter-market snapshot (DXY, US bonds, SPX, Gold, Oil, BTC).")
    p_cross.add_argument("--instruments", default="")  # empty → use defaults
    p_cross.add_argument("--correlation-pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_cross.add_argument("--granularity", default="H1")
    p_cross.add_argument("--count", type=int, default=200)
    p_cross.add_argument("--output", type=Path, default=DEFAULT_CROSS_ASSET_SNAPSHOT)
    p_cross.add_argument("--report", type=Path, default=DEFAULT_CROSS_ASSET_REPORT)

    p_flow = sub.add_parser("flow-snapshot", help="OANDA order book + position book + spread time-series per pair.")
    p_flow.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_flow.add_argument("--top-n", type=int, default=5)
    p_flow.add_argument("--spread-lookback-minutes", type=int, default=60)
    p_flow.add_argument("--output", type=Path, default=DEFAULT_FLOW_SNAPSHOT)
    p_flow.add_argument("--report", type=Path, default=DEFAULT_FLOW_REPORT)

    p_strength = sub.add_parser("currency-strength", help="G8 currency strength meter from a 28-pair matrix.")
    p_strength.add_argument("--granularity", default="H1")
    p_strength.add_argument("--lookback-bars", type=int, default=24)
    p_strength.add_argument("--fetch-count", type=int, default=50)
    p_strength.add_argument("--output", type=Path, default=DEFAULT_CURRENCY_STRENGTH)
    p_strength.add_argument("--report", type=Path, default=DEFAULT_CURRENCY_STRENGTH_REPORT)

    p_levels = sub.add_parser("levels-snapshot", help="Pivots, PDH/PDL/PDC, session ranges, round-numbers per pair.")
    p_levels.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_levels.add_argument("--output", type=Path, default=DEFAULT_LEVELS_SNAPSHOT)
    p_levels.add_argument("--report", type=Path, default=DEFAULT_LEVELS_REPORT)

    p_cal = sub.add_parser("economic-calendar", help="ForexFactory weekly XML feed + per-pair event-window flags.")
    p_cal.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_cal.add_argument("--pre-minutes", type=int, default=30)
    p_cal.add_argument("--post-minutes", type=int, default=30)
    p_cal.add_argument("--impact", default="High,Medium")
    p_cal.add_argument("--no-fetch", action="store_true", help="Skip fetching (for offline tests).")
    p_cal.add_argument("--output", type=Path, default=DEFAULT_CALENDAR_SNAPSHOT)
    p_cal.add_argument("--report", type=Path, default=DEFAULT_CALENDAR_REPORT)

    p_cot = sub.add_parser("cot-snapshot", help="CFTC Commitments of Traders weekly disaggregated report.")
    p_cot.add_argument("--no-fetch", action="store_true")
    p_cot.add_argument("--output", type=Path, default=DEFAULT_COT_SNAPSHOT)
    p_cot.add_argument("--report", type=Path, default=DEFAULT_COT_REPORT)

    p_opt = sub.add_parser("option-skew", help="Option-skew (RR/IV) adapter — placeholder until vendor feed is wired.")
    p_opt.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_opt.add_argument("--tenors", default="1W,1M,3M")
    p_opt.add_argument("--output", type=Path, default=DEFAULT_OPTION_SKEW)
    p_opt.add_argument("--report", type=Path, default=DEFAULT_OPTION_SKEW_REPORT)

    p_news = sub.add_parser("news-snapshot", help="Fetch public news feeds into ignored data/log artifacts.")
    p_news.add_argument("--no-fetch", action="store_true", help="Skip network fetch and emit a missing-feed issue.")
    p_news.add_argument("--lookback-hours", type=int, default=None)
    p_news.add_argument("--max-items", type=int, default=None)
    p_news.add_argument("--digest-items", type=int, default=None)
    p_news.add_argument("--flow-entries", type=int, default=None)
    p_news.add_argument("--output", type=Path, default=DEFAULT_NEWS_SNAPSHOT)
    p_news.add_argument("--digest", type=Path, default=DEFAULT_NEWS_DIGEST)
    p_news.add_argument("--flow-log", type=Path, default=DEFAULT_NEWS_FLOW_LOG)

    p_mine = sub.add_parser("mine-strategy", help="Mine legacy evidence into a strategy profile.")
    p_mine.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_mine.add_argument("--report", type=Path, default=DEFAULT_STRATEGY_REPORT)
    p_mine.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_mine.add_argument("--loss-cap-jpy", type=float, default=None)
    p_mine.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)

    p_story = sub.add_parser("mine-market-stories", help="Mine narrative, market regime, and chart-story evidence.")
    p_story.add_argument("--archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    p_story.add_argument("--report", type=Path, default=DEFAULT_MARKET_STORY_REPORT)
    p_story.add_argument("--profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_story.add_argument("--news-dir", type=Path, default=ROOT / "logs")

    p_campaign = sub.add_parser("plan-campaign", help="Build a multi-desk daily campaign plan.")
    p_campaign.add_argument("--start-balance", type=float, required=True)
    p_campaign.add_argument("--target-return-pct", type=float, default=10.0)
    p_campaign.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_campaign.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_campaign.add_argument("--report", type=Path, default=DEFAULT_CAMPAIGN_REPORT)
    p_campaign.add_argument("--plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)

    p_target = sub.add_parser("daily-target-state", help="Record daily 10%% target progress from broker truth.")
    p_target.add_argument("--start-balance", type=float, default=None)
    p_target.add_argument("--target-return-pct", type=float, default=None)
    p_target.add_argument("--realized-pl", type=float, default=None)
    p_target.add_argument("--daily-risk-budget", type=float, default=None)
    p_target.add_argument(
        "--target-trades-per-day",
        type=int,
        default=None,
        help="Override expected trade pace (per_trade cap = daily_risk_budget / pace).",
    )
    p_target.add_argument("--snapshot", type=Path, default=None)
    p_target.add_argument("--state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_target.add_argument("--report", type=Path, default=DEFAULT_DAILY_TARGET_REPORT)

    p_replay = sub.add_parser("replay-backtest", help="Replay imported legacy days against 10%% target coverage.")
    p_replay.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_replay.add_argument("--start-balance", type=float, required=True)
    p_replay.add_argument("--target-return-pct", type=float, default=10.0)
    p_replay.add_argument("--max-loss", type=float, default=None)
    p_replay.add_argument("--daily-risk-pct", type=float, default=None)
    p_replay.add_argument("--target-trades-per-day", type=int, default=None)
    p_replay.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_replay.add_argument("--max-days", type=int, default=None)
    p_replay.add_argument("--output", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_replay.add_argument("--report", type=Path, default=DEFAULT_REPLAY_BACKTEST_REPORT)

    p_ai_test = sub.add_parser(
        "ai-test-bot-backtest",
        help="Walk-forward backtest an AI-managed parameter policy over imported legacy records.",
    )
    p_ai_test.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_ai_test.add_argument("--start-balance", type=float, required=True)
    p_ai_test.add_argument("--target-return-pct", type=float, default=10.0)
    p_ai_test.add_argument("--max-loss", type=float, default=None)
    p_ai_test.add_argument("--daily-risk-pct", type=float, default=None)
    p_ai_test.add_argument("--target-trades-per-day", type=int, default=None)
    p_ai_test.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_ai_test.add_argument("--training-days", type=int, default=DEFAULT_TRAINING_DAYS)
    p_ai_test.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    p_ai_test.add_argument("--max-active-buckets", type=int, default=DEFAULT_MAX_ACTIVE_BUCKETS)
    p_ai_test.add_argument("--source-tables", default=",".join(DEFAULT_SOURCE_TABLES))
    p_ai_test.add_argument("--no-dedupe-opportunities", action="store_true")
    p_ai_test.add_argument("--max-validation-days", type=int, default=None)
    p_ai_test.add_argument("--output", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_ai_test.add_argument("--report", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST_REPORT)

    p_coverage = sub.add_parser("optimize-coverage", help="Measure live-ready target coverage and emit gap tasks.")
    p_coverage.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_coverage.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_coverage.add_argument("--replay", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_coverage.add_argument("--output", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_coverage.add_argument("--report", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION_REPORT)

    p_attack = sub.add_parser("ai-attack-advice", help="Rank current LIVE_READY lanes using read-only AI parameter advice.")
    p_attack.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_attack.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_attack.add_argument("--ai-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_attack.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_attack.add_argument("--output", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_attack.add_argument("--report", type=Path, default=DEFAULT_AI_ATTACK_ADVICE_REPORT)

    p_exec_replay = sub.add_parser("replay-execution", help="Replay live-ready order receipts over a quote path.")
    p_exec_replay.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_exec_replay.add_argument("--prices", type=Path, required=True)
    p_exec_replay.add_argument("--target-jpy", type=float, default=0.0)
    p_exec_replay.add_argument("--lane-id", default=None)
    p_exec_replay.add_argument("--output", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_exec_replay.add_argument("--report", type=Path, default=DEFAULT_EXECUTION_REPLAY_REPORT)

    p_learn = sub.add_parser("learn-post-trade", help="Create receipt-backed post-trade learning candidates.")
    p_learn.add_argument("--outcome", type=Path, default=None)
    p_learn.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_learn.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_learn.add_argument("--trader-decision", type=Path, default=DEFAULT_TRADER_DECISION)
    p_learn.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_learn.add_argument("--output", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_learn.add_argument("--report", type=Path, default=DEFAULT_POST_TRADE_LEARNING_REPORT)

    p_cert = sub.add_parser("certify-dry-run", help="Certify dry-run receipts before live expansion.")
    p_cert.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_cert.add_argument("--execution-replay", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_cert.add_argument("--post-trade-learning", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_cert.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_cert.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_cert.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_cert.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_cert.add_argument("--output", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION)
    p_cert.add_argument("--report", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION_REPORT)

    p_complete = sub.add_parser("completion-status", help="Summarize blockers to full system completion.")
    p_complete.add_argument("--broker-snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_complete.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_complete.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_complete.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_complete.add_argument("--replay-backtest", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_complete.add_argument("--execution-replay", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_complete.add_argument("--dry-run-certification", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION)
    p_complete.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_complete.add_argument("--output", type=Path, default=DEFAULT_COMPLETION_STATUS)
    p_complete.add_argument("--report", type=Path, default=DEFAULT_COMPLETION_STATUS_REPORT)

    p_gpt = sub.add_parser("gpt-trader-decision", help="Verify a Codex-written trader decision against broker truth.")
    p_gpt.add_argument("--snapshot", type=Path, required=True)
    p_gpt.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_gpt.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    p_gpt.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_gpt.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_gpt.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_gpt.add_argument("--attack-advice", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_gpt.add_argument("--decision-response", type=Path, default=None)
    p_gpt.add_argument("--max-lanes", type=int, default=DEFAULT_GPT_MAX_LANES)
    p_gpt.add_argument("--output", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_gpt.add_argument("--report", type=Path, default=DEFAULT_GPT_TRADER_DECISION_REPORT)

    p_intents = sub.add_parser("generate-intents", help="Generate dry-run order intents from campaign lanes.")
    p_intents.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    p_intents.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_intents.add_argument("--snapshot", type=Path)
    p_intents.add_argument("--output", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_intents.add_argument("--report", type=Path, default=DEFAULT_ORDER_INTENT_REPORT)
    p_intents.add_argument("--max-loss-jpy", type=float, default=None)
    p_intents.add_argument("--max-loss-pct", type=float, default=None)
    p_intents.add_argument("--risk-equity-jpy", type=float, default=None)
    p_intents.add_argument("--max-candidates", type=int, default=56)

    p_promote = sub.add_parser("promote-receipts", help="Promote strategy profiles from dry-run order receipts.")
    p_promote.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_promote.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_promote.add_argument("--output-profile", type=Path, default=None)
    p_promote.add_argument("--report", type=Path, default=DEFAULT_RECEIPT_PROMOTION_REPORT)

    p_stage = sub.add_parser("stage-live-order", help="Stage or explicitly send one live-ready OANDA order.")
    p_stage.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_stage.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_stage.add_argument("--lane-id", default=None)
    p_stage.add_argument("--output", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_stage.add_argument("--report", type=Path, default=DEFAULT_LIVE_ORDER_STAGE_REPORT)
    p_stage.add_argument("--max-loss-jpy", type=float, default=None)
    p_stage.add_argument("--max-loss-pct", type=float, default=None)
    p_stage.add_argument("--risk-equity-jpy", type=float, default=None)
    p_stage.add_argument("--send", action="store_true")
    p_stage.add_argument("--confirm-live", action="store_true")

    p_auto = sub.add_parser("autotrade-cycle", help="Run one safe automated trading cycle.")
    p_auto.add_argument("--send", action="store_true")
    p_auto.add_argument("--report", type=Path, default=DEFAULT_AUTOTRADE_REPORT)
    p_auto.add_argument("--use-gpt-trader", action="store_true")
    p_auto.add_argument("--gpt-decision-response", type=Path, default=None)
    p_auto.add_argument("--gpt-max-lanes", type=int, default=DEFAULT_GPT_MAX_LANES)
    p_auto.add_argument(
        "--reuse-market-artifacts",
        action="store_true",
        help=(
            "Use the existing broker snapshot and order intents for GPT verification/selection; "
            "the live gateway still fetches fresh broker truth before staging or sending."
        ),
    )
    p_auto.add_argument(
        "--refresh-market-story",
        dest="refresh_market_story",
        action="store_true",
        help="Refresh market story from news before scoring lanes (default)",
    )
    p_auto.add_argument(
        "--no-refresh-market-story",
        dest="refresh_market_story",
        action="store_false",
        help="Disable market story refresh for this cycle",
    )
    p_auto.set_defaults(refresh_market_story=True)
    p_auto.add_argument("--market-news-dir", type=Path, default=ROOT / "logs")
    p_auto.add_argument("--max-loss-jpy", type=float, default=None)
    p_auto.add_argument("--max-loss-pct", type=float, default=None)
    p_auto.add_argument("--risk-equity-jpy", type=float, default=None)
    p_auto.add_argument("--trader-settings", type=Path, default=DEFAULT_TRADER_SETTINGS)

    p_risk = sub.add_parser("risk-dry-run", help="Validate an order intent against a JSON snapshot.")
    p_risk.add_argument("--intent", type=Path, required=True)
    p_risk.add_argument("--snapshot", type=Path, required=True)
    p_risk.add_argument("--for-live-send", action="store_true")
    p_risk.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_risk.add_argument("--no-strategy-profile", action="store_true")
    p_risk.add_argument("--max-loss-jpy", type=float, default=None)
    p_risk.add_argument("--max-loss-pct", type=float, default=None)
    p_risk.add_argument("--risk-equity-jpy", type=float, default=None)

    args = parser.parse_args(argv)
    if args.command == "import-legacy":
        summary = LegacyImporter(args.archive, args.db, args.report).run()
        print(
            json.dumps(
                {
                    "archive": str(summary.archive),
                    "db_path": str(summary.db_path),
                    "report_path": str(summary.report_path),
                    "source_files": summary.source_files,
                    "legacy_rows": summary.legacy_rows,
                    "live_trade_events": summary.live_trade_events,
                    "journal_events": summary.journal_events,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-intents":
        summary = IntentGenerator(
            campaign_plan=args.campaign_plan,
            strategy_profile=args.strategy_profile,
            output_path=args.output,
            report_path=args.report,
            max_loss_jpy=_resolve_max_loss_from_args(
                max_loss_jpy=args.max_loss_jpy,
                max_loss_pct=args.max_loss_pct,
                risk_equity_jpy=args.risk_equity_jpy,
                label="generate-intents",
            ),
        ).run(snapshot_path=args.snapshot, max_candidates=args.max_candidates)
        print(
            json.dumps(
                {
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "candidates_seen": summary.candidates_seen,
                    "generated": summary.generated,
                    "needs_snapshot": summary.needs_snapshot,
                    "dry_run_passed": summary.dry_run_passed,
                    "live_ready": summary.live_ready,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "promote-receipts":
        summary = ReceiptPromoter(
            profile_path=args.profile,
            intents_path=args.intents,
            output_profile=args.output_profile,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "profile_path": str(summary.profile_path),
                    "intents_path": str(summary.intents_path),
                    "report_path": str(summary.report_path),
                    "profiles_seen": summary.profiles_seen,
                    "receipts_seen": summary.receipts_seen,
                    "promoted": summary.promoted,
                    "still_blocked": summary.still_blocked,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "stage-live-order":
        try:
            summary = LiveOrderGateway(
                client=OandaExecutionClient(),
                strategy_profile=args.strategy_profile,
                output_path=args.output,
                report_path=args.report,
                live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
                max_loss_jpy=_resolve_max_loss_from_args(
                    max_loss_jpy=args.max_loss_jpy,
                    max_loss_pct=args.max_loss_pct,
                    risk_equity_jpy=args.risk_equity_jpy,
                    label="stage-live-order",
                ),
            ).run(intents_path=args.intents, lane_id=args.lane_id, send=args.send, confirm_live=args.confirm_live)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "lane_id": summary.lane_id,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "sent": summary.sent,
                    "risk_issues": summary.risk_issues,
                    "strategy_issues": summary.strategy_issues,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {"STAGED", "SENT"} else 2
    if args.command == "autotrade-cycle":
        try:
            use_gpt_trader = args.use_gpt_trader or os.environ.get("QR_GPT_TRADER_ENABLED") == "1"
            gpt_provider = _static_gpt_provider(
                decision_response=args.gpt_decision_response,
                required=use_gpt_trader,
            )
            summary = AutoTradeCycle(
                report_path=args.report,
                target_state_path=DEFAULT_DAILY_TARGET_STATE,
                target_report_path=DEFAULT_DAILY_TARGET_REPORT,
                use_gpt_trader=use_gpt_trader,
                gpt_provider=gpt_provider,
                gpt_max_lanes=args.gpt_max_lanes,
                reuse_market_artifacts=args.reuse_market_artifacts,
                trader_settings_path=args.trader_settings,
                refresh_market_story=args.refresh_market_story,
                market_news_root=args.market_news_dir,
                live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
                max_loss_jpy=args.max_loss_jpy,
                max_loss_pct=args.max_loss_pct,
                risk_equity_jpy=args.risk_equity_jpy,
            ).run(send=args.send)
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "report_path": str(summary.report_path),
                    "snapshot_path": str(summary.snapshot_path),
                    "intents_path": str(summary.intents_path),
                    "selected_lane_id": summary.selected_lane_id,
                    "selected_lane_ids": list(summary.selected_lane_ids),
                    "deterministic_lane_id": summary.deterministic_lane_id,
                    "decision_source": summary.decision_source,
                    "sent": summary.sent,
                    "sent_count": summary.sent_count,
                    "positions": summary.positions,
                    "orders": summary.orders,
                    "live_ready": summary.live_ready,
                    "receipt_promotions": summary.receipt_promotions,
                    "canceled_orders": list(summary.canceled_orders),
                    "position_management_action": summary.position_management_action,
                    "position_execution_status": summary.position_execution_status,
                    "position_execution_sent": summary.position_execution_sent,
                    "target_status": summary.target_status,
                    "target_remaining_jpy": summary.target_remaining_jpy,
                    "target_progress_pct": summary.target_progress_pct,
                    "selected_lane_score": summary.selected_lane_score,
                    "selected_lane_size_multiple": summary.selected_lane_size_multiple,
                    "gpt_status": summary.gpt_status,
                    "gpt_action": summary.gpt_action,
                    "gpt_allowed": summary.gpt_allowed,
                    "gpt_issues": summary.gpt_issues,
                    "gpt_error": summary.gpt_error,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {
            "SENT",
            "STAGED",
            "MONITOR_ONLY_EXPOSURE_OPEN",
            "CANCELED_CONTAMINATED_PENDING",
            "POSITION_ACTION_SENT",
            "POSITION_ACTION_STAGED",
            "POSITION_ACTION_BLOCKED",
            "NO_LIVE_READY_INTENT",
            "NO_TRADE",
            "GPT_WAIT",
            "GPT_REQUEST_EVIDENCE",
            "GPT_REJECTED",
            "GPT_DECISION_NOT_PREFILTERED",
            "TARGET_REACHED_PROTECT",
        } else 2
    if args.command == "plan-campaign":
        summary = CampaignPlanner(
            strategy_profile=args.strategy_profile,
            market_story_profile=args.market_story_profile,
            report_path=args.report,
            plan_path=args.plan,
        ).run(start_balance_jpy=args.start_balance, target_return_pct=args.target_return_pct)
        target_summary = DailyTargetLedger(
            state_path=DEFAULT_DAILY_TARGET_STATE,
            report_path=DEFAULT_DAILY_TARGET_REPORT,
            pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
        ).run(start_balance_jpy=args.start_balance, target_return_pct=args.target_return_pct)
        print(
            json.dumps(
                {
                    "report_path": str(summary.report_path),
                    "plan_path": str(summary.plan_path),
                    "target_state_path": str(target_summary.state_path),
                    "target_report_path": str(target_summary.report_path),
                    "target_jpy": summary.target_jpy,
                    "remaining_target_jpy": target_summary.remaining_target_jpy,
                    "lanes": summary.lanes,
                    "actionable_lanes": summary.actionable_lanes,
                    "rejected_lanes": summary.rejected_lanes,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "mine-market-stories":
        summary = MarketStoryMiner(
            archive=args.archive,
            report_path=args.report,
            profile_path=args.profile,
            news_root=args.news_dir,
        ).run()
        print(
            json.dumps(
                {
                    "archive": str(summary.archive),
                    "report_path": str(summary.report_path),
                    "profile_path": str(summary.profile_path),
                    "news_dir": str(args.news_dir),
                    "artifacts": summary.artifacts,
                    "story_lines": summary.story_lines,
                    "pairs": summary.pairs,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "broker-snapshot":
        pairs = [part.strip().upper() for part in args.pairs.split(",") if part.strip()]
        try:
            snapshot = OandaReadOnlyClient().snapshot(pairs)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        text = _snapshot_to_json(snapshot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
            payload = {
                "output_path": str(args.output),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            }
            if snapshot.account is not None:
                payload["account"] = {
                    "nav_jpy": snapshot.account.nav_jpy,
                    "balance_jpy": snapshot.account.balance_jpy,
                    "unrealized_pl_jpy": snapshot.account.unrealized_pl_jpy,
                    "margin_used_jpy": snapshot.account.margin_used_jpy,
                    "margin_available_jpy": snapshot.account.margin_available_jpy,
                    "hedging_enabled": snapshot.account.hedging_enabled,
                }
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(text)
        return 0
    if args.command == "pair-charts":
        from quant_rabbit.analysis.chart_reader import build_pair_chart

        pairs = [part.strip().upper() for part in args.pairs.split(",") if part.strip()]
        timeframes = tuple(part.strip().upper() for part in args.timeframes.split(",") if part.strip())
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2

        charts = []
        for pair in pairs:
            chart = build_pair_chart(pair, client=client, timeframes=timeframes, count=args.count)
            charts.append(chart)

        generated_at = datetime.now(timezone.utc).isoformat()
        chart_payloads = [chart.to_dict() for chart in charts]
        chart_payloads.sort(key=lambda c: max(c["long_score"], c["short_score"]), reverse=True)
        output_payload = {
            "generated_at_utc": generated_at,
            "timeframes": list(timeframes),
            "candle_count": int(args.count),
            "charts": chart_payloads,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# Pair Charts Report",
                "",
                f"- Generated at UTC: `{generated_at}`",
                f"- Timeframes: `{','.join(timeframes)}`",
                f"- Candles per timeframe: `{args.count}`",
                "",
                "## Pair Score Table",
                "",
                "| Pair | Side | Long | Short | Regime | Story |",
                "|---|---|---|---|---|---|",
            ]
            for c in chart_payloads:
                side = "LONG" if c["long_score"] >= c["short_score"] else "SHORT"
                story = c["chart_story"].replace("|", "/")
                lines.append(
                    f"| `{c['pair']}` | `{side}` | `{c['long_score']:.3f}` | `{c['short_score']:.3f}` | `{c['dominant_regime']}` | {story} |"
                )
            lines.extend([
                "",
                "## How To Read",
                "",
                "- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (D>H4>H1>M30>M15>M5>M1).",
                "- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.",
                "- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).",
                "- Pairs are sorted by max(long, short); the top entries are where edges line up.",
            ])
            args.report.write_text("\n".join(lines) + "\n")

        print(json.dumps({
            "output_path": str(args.output) if args.output else None,
            "report_path": str(args.report) if args.report else None,
            "pairs": len(charts),
            "top": [
                {"pair": c["pair"], "side": "LONG" if c["long_score"] >= c["short_score"] else "SHORT",
                 "long": round(c["long_score"], 3), "short": round(c["short_score"], 3), "regime": c["dominant_regime"]}
                for c in chart_payloads[:5]
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "cross-asset-snapshot":
        from quant_rabbit.analysis.cross_asset import (
            DEFAULT_CROSS_ASSET_INSTRUMENTS,
            DEFAULT_CORRELATION_PAIRS,
            build_cross_asset_snapshot,
        )
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        instruments = (
            tuple(p.strip() for p in args.instruments.split(",") if p.strip())
            if args.instruments else DEFAULT_CROSS_ASSET_INSTRUMENTS
        )
        corr_pairs = tuple(p.strip().upper() for p in args.correlation_pairs.split(",") if p.strip())
        snap = build_cross_asset_snapshot(
            client=client, instruments=instruments, correlation_pairs=corr_pairs,
            granularity=args.granularity, count=int(args.count),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Cross-Asset Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Granularity: `{snap.granularity}`",
                f"- Candles per series: `{snap.candle_count}`",
                "",
                "## Synthetic DXY",
            ]
            sd = snap.synthetic_dxy
            if sd:
                lines.append(f"- last={sd.last_value}, Δ24h={sd.change_pct_24h}%, Δ5d={sd.change_pct_5d}%, components={list(sd.components_used)}")
            else:
                lines.append("- (insufficient basket coverage)")
            lines.extend(["", "## Yield Spreads", ""])
            for y in snap.yield_spreads:
                lines.append(f"- **{y.name}**: a={y.a_last}, b={y.b_last}, spread={y.spread_last}, Δ24h={y.spread_change_24h}, issue={y.issue}")
            lines.extend(["", "## Asset Readings", "", "| Instrument | Last | Δ24h% | Δ5d% | Z(60) | RV(60) | Trend |", "|---|---|---|---|---|---|---|"])
            for a in snap.assets:
                lines.append(f"| `{a.instrument}` | {a.last_price} | {a.change_pct_24h} | {a.change_pct_5d} | {a.z_score_60} | {a.realized_vol_60} | {a.trend_label} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output),
            "report_path": str(args.report),
            "assets_fetched": sum(1 for a in snap.assets if a.fetched),
            "issues": list(snap.issues),
            "synthetic_dxy_last": snap.synthetic_dxy.last_value if snap.synthetic_dxy else None,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "flow-snapshot":
        from quant_rabbit.analysis.flow import build_flow_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        snap = build_flow_snapshot(
            client=client, pairs=pairs,
            top_n=int(args.top_n), spread_lookback_minutes=int(args.spread_lookback_minutes),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Flow Snapshot (OrderBook + PositionBook + Spread)",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                "",
                "## Spread State",
                "",
                "| Pair | Current(p) | Median(p) | P90(p) | Max(p) | Samples | Stress |",
                "|---|---|---|---|---|---|---|",
            ]
            for s in snap.spreads:
                lines.append(f"| `{s.instrument}` | {s.current_pips} | {s.median_pips} | {s.p90_pips} | {s.max_pips} | {s.sample_size} | `{s.stress_flag}` |")
            lines.extend(["", "## Position Book Top Clusters", ""])
            for pb in snap.position_books:
                if pb.issue:
                    lines.append(f"- `{pb.instrument}` issue: {pb.issue}")
                    continue
                top_long = ", ".join(f"{b.price}@{b.long_pct:.1f}%" for b in pb.top_long_clusters[:3])
                top_short = ", ".join(f"{b.price}@{b.short_pct:.1f}%" for b in pb.top_short_clusters[:3])
                lines.append(f"- `{pb.instrument}` price={pb.price} long_total={pb.long_total_pct:.1f}% short_total={pb.short_total_pct:.1f}% top_long=[{top_long}] top_short=[{top_short}]")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "pairs": len(pairs), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "currency-strength":
        from quant_rabbit.analysis.strength import build_strength_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        snap = build_strength_snapshot(
            client=client, granularity=args.granularity,
            lookback_bars=int(args.lookback_bars), fetch_count=int(args.fetch_count),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Currency Strength",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Granularity: `{snap.granularity}` over `{snap.lookback_bars}` bars",
                f"- Pairs used: {len(snap.pairs_used)} / missing {len(snap.pairs_missing)}",
                f"- Suggested cross: `{snap.strongest_pair_suggestion}`",
                "",
                "| Rank | Currency | Score (%) |",
                "|---|---|---|",
            ]
            for s in snap.scores:
                lines.append(f"| {s.rank} | `{s.currency}` | {s.score_pct:.3f} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "suggestion": snap.strongest_pair_suggestion,
            "issues": list(snap.issues)[:5],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "levels-snapshot":
        from quant_rabbit.analysis.levels import build_levels_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        snap = build_levels_snapshot(client=client, pairs=pairs)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Levels Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                "",
                "## Per-Pair",
                "",
            ]
            for p in snap.pairs:
                lines.append(f"### `{p.pair}`")
                lines.append(f"- last={p.last_close}, PDH={p.pdh}, PDL={p.pdl}, PDC={p.pdc}, daily_open={p.daily_open}, weekly_open={p.weekly_open}, monthly_open={p.monthly_open}")
                for piv in p.pivots:
                    lines.append(f"- `{piv.style}`: PP={piv.pp}, R1={piv.r1}, R2={piv.r2}, R3={piv.r3}, S1={piv.s1}, S2={piv.s2}, S3={piv.s3}")
                for s in p.sessions:
                    lines.append(f"- session `{s.name}`: H={s.high}, L={s.low}, range={s.range_pips}p")
                rn_near = [r for r in p.round_numbers if abs(r.distance_pips) < 50][:5]
                lines.append(f"- nearby round-numbers: " + ", ".join(f"{r.price}({r.distance_pips:.1f}p)" for r in rn_near))
                lines.append("")
            if snap.issues:
                lines.extend(["## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "pairs": len(snap.pairs), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "news-snapshot":
        from quant_rabbit.analysis.news import (
            DEFAULT_DIGEST_ITEMS,
            DEFAULT_FLOW_ENTRIES,
            DEFAULT_LOOKBACK_HOURS,
            DEFAULT_MAX_ITEMS,
            build_news_snapshot,
            write_news_artifacts,
        )

        snap = build_news_snapshot(
            lookback_hours=args.lookback_hours or DEFAULT_LOOKBACK_HOURS,
            max_items=args.max_items or DEFAULT_MAX_ITEMS,
            fetch=not args.no_fetch,
        )
        write_news_artifacts(
            snap,
            output_path=args.output,
            digest_path=args.digest,
            flow_log_path=args.flow_log,
            digest_items=args.digest_items or DEFAULT_DIGEST_ITEMS,
            flow_entries=args.flow_entries or DEFAULT_FLOW_ENTRIES,
        )
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "digest_path": str(args.digest),
                    "flow_log_path": str(args.flow_log),
                    "items": len(snap.items),
                    "issues": list(snap.issues),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "economic-calendar":
        from quant_rabbit.analysis.calendar import build_calendar_snapshot
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        impact = tuple(p.strip() for p in args.impact.split(",") if p.strip())
        snap = build_calendar_snapshot(
            pairs=pairs, pre_minutes=int(args.pre_minutes), post_minutes=int(args.post_minutes),
            impact_filter=impact, fetch=not args.no_fetch,
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Economic Calendar",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Source: `{snap.source_url}`",
                f"- Events parsed: {len(snap.events)}",
                "",
                "## Pair Windows",
                "",
                "| Pair | In Window | Reason |",
                "|---|---|---|",
            ]
            for w in snap.pair_windows:
                lines.append(f"| `{w.pair}` | {'YES' if w.in_window else 'no'} | {w.reason} |")
            lines.extend(["", "## Upcoming High/Medium Events (first 30)", "", "| Time UTC | Currency | Impact | Title | Forecast | Previous |", "|---|---|---|---|---|---|"])
            for e in list(snap.events)[:30]:
                if e.impact in ("High", "Medium"):
                    lines.append(f"| `{e.timestamp_utc}` | `{e.currency}` | `{e.impact}` | {e.title} | {e.forecast or ''} | {e.previous or ''} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "events": len(snap.events),
            "in_window_pairs": [w.pair for w in snap.pair_windows if w.in_window],
            "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "cot-snapshot":
        from quant_rabbit.analysis.cot import build_cot_snapshot
        snap = build_cot_snapshot(fetch=not args.no_fetch)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# COT Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Source: `{snap.source_url}`",
                f"- Reports: {len(snap.reports)}",
                "",
                "| Currency | Report Date | Lev Long | Lev Short | Lev Net | Δw Lev Net | OI |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in snap.reports:
                lines.append(f"| `{r.currency}` | `{r.report_date}` | {r.leveraged_long} | {r.leveraged_short} | {r.leveraged_net} | {r.week_change_leveraged_net} | {r.open_interest} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "reports": len(snap.reports), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "option-skew":
        from quant_rabbit.analysis.options import build_option_skew_snapshot
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        tenors = tuple(t.strip() for t in args.tenors.split(",") if t.strip())
        snap = build_option_skew_snapshot(pairs=pairs, tenors=tenors)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Option Skew Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Provider: `{snap.provider}`",
                "",
                "| Pair | Tenor | ATM IV | RR 25Δ | BF 25Δ | Source | Issue |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in snap.readings:
                lines.append(f"| `{r.pair}` | `{r.tenor}` | {r.atm_iv} | {r.rr_25d} | {r.bf_25d} | {r.source or ''} | {r.issue or ''} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "readings": len(snap.readings), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "mine-strategy":
        summary = StrategyMiner(
            args.db,
            args.report,
            args.profile,
            loss_cap_jpy=args.loss_cap_jpy,
            target_state_path=args.target_state,
        ).run()
        print(
            json.dumps(
                {
                    "db_path": str(summary.db_path),
                    "report_path": str(summary.report_path),
                    "profile_path": str(summary.profile_path),
                    "profiles": summary.profiles,
                    "blocked": summary.blocked,
                    "candidates": summary.candidates,
                    "risk_repair_candidates": summary.risk_repair_candidates,
                    "mined_missed_edges": summary.mined_missed_edges,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "daily-target-state":
        try:
            summary = DailyTargetLedger(
                state_path=args.state,
                report_path=args.report,
                pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
            ).run(
                start_balance_jpy=args.start_balance,
                target_return_pct=args.target_return_pct,
                realized_pl_jpy=args.realized_pl,
                daily_risk_budget_jpy=args.daily_risk_budget,
                target_trades_per_day=args.target_trades_per_day,
                snapshot_path=args.snapshot,
            )
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "state_path": str(summary.state_path),
                    "report_path": str(summary.report_path),
                    "status": summary.status,
                    "target_jpy": summary.target_jpy,
                    "progress_jpy": summary.progress_jpy,
                    "progress_pct": summary.progress_pct,
                    "remaining_target_jpy": summary.remaining_target_jpy,
                    "remaining_risk_budget_jpy": summary.remaining_risk_budget_jpy,
                    "target_trades_per_day": summary.target_trades_per_day,
                    "target_trades_per_day_source": summary.target_trades_per_day_source,
                    "per_trade_risk_budget_jpy": summary.per_trade_risk_budget_jpy,
                    "unprotected_positions": summary.unprotected_positions,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "replay-backtest":
        summary = ReplayBacktester(
            db_path=args.db,
            output_path=args.output,
            report_path=args.report,
            max_loss_jpy=args.max_loss,
            daily_risk_pct=args.daily_risk_pct,
            target_trades_per_day=args.target_trades_per_day,
            target_state_path=args.target_state,
        ).run(
            start_balance_jpy=args.start_balance,
            target_return_pct=args.target_return_pct,
            max_days=args.max_days,
        )
        print(
            json.dumps(
                {
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "days": summary.days,
                    "target_jpy": summary.target_jpy,
                    "historical_target_hits": summary.historical_target_hits,
                    "evidence_target_covered": summary.evidence_target_covered,
                    "risk_repair_days": summary.risk_repair_days,
                    "missed_edge_days": summary.missed_edge_days,
                    "total_historical_net_jpy": summary.total_historical_net_jpy,
                    "total_risk_capped_net_jpy": summary.total_risk_capped_net_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "ai-test-bot-backtest":
        try:
            source_tables = tuple(item.strip() for item in str(args.source_tables).split(",") if item.strip())
            summary = AITestBotBacktester(
                db_path=args.db,
                output_path=args.output,
                report_path=args.report,
                target_state_path=args.target_state,
                max_loss_jpy=args.max_loss,
                daily_risk_pct=args.daily_risk_pct,
                target_trades_per_day=args.target_trades_per_day,
                training_days=args.training_days,
                min_train_trades=args.min_train_trades,
                max_active_buckets=args.max_active_buckets,
                source_tables=source_tables,
                dedupe_opportunities=not args.no_dedupe_opportunities,
            ).run(
                start_balance_jpy=args.start_balance,
                target_return_pct=args.target_return_pct,
                max_validation_days=args.max_validation_days,
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "validation_days": summary.validation_days,
                    "traded_days": summary.traded_days,
                    "target_hit_days": summary.target_hit_days,
                    "total_managed_net_jpy": summary.total_managed_net_jpy,
                    "profit_factor": summary.profit_factor,
                    "blockers": summary.blockers,
                    "live_permission": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "optimize-coverage":
        summary = CoverageOptimizer(
            intents_path=args.intents,
            target_state_path=args.target_state,
            replay_path=args.replay,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "remaining_target_jpy": summary.remaining_target_jpy,
                    "live_ready_reward_jpy": summary.live_ready_reward_jpy,
                    "sequential_ladder_reward_jpy": summary.sequential_ladder_reward_jpy,
                    "sequential_ladder_steps": summary.sequential_ladder_steps,
                    "potential_reward_jpy": summary.potential_reward_jpy,
                    "live_ready_lanes": summary.live_ready_lanes,
                    "promotion_candidate_lanes": summary.promotion_candidate_lanes,
                    "action_items": summary.action_items,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {"LIVE_READY_COVERAGE_READY", "TARGET_REACHED_PROTECT"} else 2
    if args.command == "replay-execution":
        try:
            summary = ExecutionReplayer(
                intents_path=args.intents,
                price_path=args.prices,
                output_path=args.output,
                report_path=args.report,
            ).run(target_jpy=args.target_jpy, lane_id=args.lane_id)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "orders": summary.orders,
                    "filled": summary.filled,
                    "closed": summary.closed,
                    "target_hit": summary.target_hit,
                    "net_pl_jpy": summary.net_pl_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {"TARGET_HIT", "REPLAY_COMPLETE"} else 2
    if args.command == "learn-post-trade":
        summary = PostTradeLearner(
            outcome_path=args.outcome,
            live_order_path=args.live_order,
            position_execution_path=args.position_execution,
            trader_decision_path=args.trader_decision,
            gpt_decision_path=args.gpt_decision,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "candidates": summary.candidates,
                    "profile_update_candidates": summary.profile_update_candidates,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "READY_FOR_REVIEW" else 2
    if args.command == "certify-dry-run":
        summary = DryRunCertifier(
            coverage_path=args.coverage,
            execution_replay_path=args.execution_replay,
            post_trade_learning_path=args.post_trade_learning,
            order_intents_path=args.order_intents,
            live_order_path=args.live_order,
            position_execution_path=args.position_execution,
            gpt_decision_path=args.gpt_decision,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "checks": summary.checks,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "CERTIFIED" else 2
    if args.command == "completion-status":
        summary = CompletionAuditor(
            broker_snapshot_path=args.broker_snapshot,
            order_intents_path=args.order_intents,
            target_state_path=args.target_state,
            coverage_path=args.coverage,
            replay_backtest_path=args.replay_backtest,
            execution_replay_path=args.execution_replay,
            dry_run_certification_path=args.dry_run_certification,
            live_order_path=args.live_order,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "blockers": summary.blockers,
                    "next_actions": summary.next_actions,
                    "live_ready_lanes": summary.live_ready_lanes,
                    "remaining_target_jpy": summary.remaining_target_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "COMPLETE" else 2
    if args.command == "ai-attack-advice":
        try:
            summary = AttackAdvisor(
                intents_path=args.intents,
                target_state_path=args.target_state,
                ai_backtest_path=args.ai_backtest,
                coverage_path=args.coverage,
                output_path=args.output,
                report_path=args.report,
            ).run()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "live_ready_lanes": summary.live_ready_lanes,
                    "recommended_now_lanes": summary.recommended_now_lanes,
                    "recommended_reward_jpy": summary.recommended_reward_jpy,
                    "coverage_pct": summary.coverage_pct,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.recommended_now_lanes > 0 else 2
    if args.command == "gpt-trader-decision":
        try:
            provider = _static_gpt_provider(
                decision_response=args.decision_response,
                required=True,
            )
            summary = GPTTraderBrain(
                provider=provider,
                intents_path=args.intents,
                campaign_plan_path=args.campaign_plan,
                strategy_profile_path=args.strategy_profile,
                market_story_profile_path=args.market_story_profile,
                target_state_path=args.target_state,
                attack_advice_path=args.attack_advice,
                output_path=args.output,
                report_path=args.report,
                max_lanes=args.max_lanes,
            ).run(snapshot_path=args.snapshot)
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "action": summary.action,
                    "selected_lane_id": summary.selected_lane_id,
                    "selected_lane_ids": list(summary.selected_lane_ids),
                    "allowed": summary.allowed,
                    "issues": summary.issues,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.allowed else 2
    if args.command == "risk-dry-run":
        intent = _intent_from_json(json.loads(args.intent.read_text()))
        snapshot = _snapshot_from_json(json.loads(args.snapshot.read_text()))
        engine = RiskEngine(
            policy=RiskPolicy(
                max_loss_jpy=_resolve_max_loss_from_args(
                    max_loss_jpy=args.max_loss_jpy,
                    max_loss_pct=args.max_loss_pct,
                    risk_equity_jpy=args.risk_equity_jpy,
                    label="risk-dry-run",
                )
            ),
            live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
        )
        decision = engine.validate(intent, snapshot, for_live_send=args.for_live_send)
        strategy_issues = []
        if not args.no_strategy_profile and args.strategy_profile.exists():
            strategy_issues = list(
                StrategyProfile.load(args.strategy_profile).validate(intent, for_live_send=args.for_live_send)
            )
        all_issues = [*decision.issues, *strategy_issues]
        allowed = decision.allowed and not any(issue.severity == "BLOCK" for issue in strategy_issues)
        print(
            json.dumps(
                {
                    "allowed": allowed,
                    "issues": [issue.__dict__ for issue in all_issues],
                    "metrics": decision.metrics.__dict__ if decision.metrics else None,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if allowed else 2
    return 1


def _resolve_max_loss_from_args(
    *,
    max_loss_jpy: float | None,
    max_loss_pct: float | None,
    risk_equity_jpy: float | None,
    label: str,
) -> float:
    # Per AGENT_CONTRACT §3.5: per-trade JPY caps must be equity-derived from
    # the daily target ledger, not from the RiskPolicy literal default
    # (`RiskPolicy.max_loss_jpy = 500.0` is a library default for tests, not a
    # production risk decision). Pull `per_trade_risk_budget_jpy` from
    # `data/daily_target_state.json` first; only fall through to the policy
    # literal when the ledger has no value (first-run / file missing) so the
    # operator gets a non-silent, finite cap and a clear remediation path.
    from quant_rabbit.strategy.intent_generator import _per_trade_risk_from_state
    default_cap = _per_trade_risk_from_state()
    if default_cap is None:
        default_cap = RiskPolicy().max_loss_jpy
    return resolve_max_loss_jpy(
        max_loss_jpy=max_loss_jpy,
        max_loss_pct=max_loss_pct,
        equity_jpy=risk_equity_jpy,
        default_max_loss_jpy=default_cap,
        label=label,
    )


def _snapshot_to_json(snapshot: BrokerSnapshot) -> str:
    payload = {
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "positions": [
            {
                "trade_id": pos.trade_id,
                "pair": pos.pair,
                "side": pos.side.value,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "unrealized_pl_jpy": pos.unrealized_pl_jpy,
                "take_profit": pos.take_profit,
                "stop_loss": pos.stop_loss,
                "owner": pos.owner.value,
            }
            for pos in snapshot.positions
        ],
        "orders": [
            {
                "order_id": order.order_id,
                "pair": order.pair,
                "order_type": order.order_type,
                "trade_id": order.trade_id,
                "price": order.price,
                "state": order.state,
                "units": order.units,
                "owner": order.owner.value,
            }
            for order in snapshot.orders
        ],
        "quotes": {
            pair: {
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in snapshot.quotes.items()
        },
        "home_conversions": snapshot.home_conversions,
    }
    if getattr(snapshot, "account", None) is not None:
        account = snapshot.account
        payload["account"] = {
            "nav_jpy": account.nav_jpy,
            "balance_jpy": account.balance_jpy,
            "unrealized_pl_jpy": account.unrealized_pl_jpy,
            "margin_used_jpy": account.margin_used_jpy,
            "margin_available_jpy": account.margin_available_jpy,
            "pl_jpy": account.pl_jpy,
            "financing_jpy": account.financing_jpy,
            "last_transaction_id": account.last_transaction_id,
            "hedging_enabled": account.hedging_enabled,
            "fetched_at_utc": account.fetched_at_utc.isoformat(),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _intent_from_json(payload: dict) -> OrderIntent:
    return OrderIntent(
        pair=str(payload["pair"]).upper(),
        side=Side.parse(str(payload["side"])),
        order_type=OrderType.parse(str(payload["order_type"])),
        units=int(payload["units"]),
        entry=float(payload["entry"]) if payload.get("entry") is not None else None,
        tp=float(payload["tp"]),
        sl=float(payload["sl"]),
        thesis=str(payload.get("thesis") or ""),
        reason=str(payload.get("reason") or ""),
        owner=Owner(str(payload.get("owner") or Owner.TRADER.value)),
        market_context=_market_context_from_json(payload.get("market_context")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _market_context_from_json(payload: object) -> MarketContext | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("market_context must be an object")
    return MarketContext(
        regime=str(payload.get("regime") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        method=TradeMethod.parse(str(payload.get("method") or "")),
        invalidation=str(payload.get("invalidation") or ""),
        event_risk=str(payload.get("event_risk") or ""),
        session=str(payload.get("session") or ""),
    )


def _snapshot_from_json(payload: dict) -> BrokerSnapshot:
    from quant_rabbit.models import BrokerOrder, BrokerPosition

    positions = []
    for item in payload.get("positions", []) or []:
        positions.append(
            BrokerPosition(
                trade_id=str(item["trade_id"]),
                pair=str(item["pair"]),
                side=Side.parse(str(item["side"])),
                units=int(item["units"]),
                entry_price=float(item["entry_price"]),
                unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
                take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
                stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
                owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            )
        )
    orders = []
    for item in payload.get("orders", []) or []:
        orders.append(
            BrokerOrder(
                order_id=str(item["order_id"]),
                pair=item.get("pair"),
                order_type=str(item.get("order_type") or ""),
                trade_id=item.get("trade_id"),
                price=float(item["price"]) if item.get("price") is not None else None,
                state=item.get("state"),
                units=int(item["units"]) if item.get("units") is not None else None,
                owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            )
        )
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        ts = item.get("timestamp_utc")
        quotes[pair] = Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
        )
    fetched = payload.get("fetched_at_utc")
    account = _account_summary_from_payload(payload.get("account"))
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=tuple(positions),
        orders=tuple(orders),
        quotes=quotes,
        account=account,
        home_conversions={str(k).upper(): float(v) for k, v in (payload.get("home_conversions") or {}).items()},
    )


def _account_summary_from_payload(payload: object):
    from quant_rabbit.models import AccountSummary

    if not isinstance(payload, dict):
        return None
    fetched = payload.get("fetched_at_utc")
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        hedging_enabled=bool(payload.get("hedging_enabled") or False),
        fetched_at_utc=(
            datetime.fromisoformat(fetched) if isinstance(fetched, str) and fetched else datetime.now(timezone.utc)
        ),
    )


def _static_gpt_provider(
    *,
    decision_response: Path | None,
    required: bool,
) -> StaticTraderProvider | None:
    source = decision_response
    if source is None:
        if required:
            raise ValueError("Codex GPT mode requires --decision-response")
        return None
    return StaticTraderProvider(json.loads(source.read_text()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
