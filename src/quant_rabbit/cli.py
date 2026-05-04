from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.automation import AutoTradeCycle, DEFAULT_AUTOTRADE_REPORT
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
)
from quant_rabbit.gpt_trader import GPTTraderBrain, StaticTraderProvider
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
    p_snapshot.add_argument("--pairs", default="USD_JPY,EUR_USD,GBP_USD,AUD_USD,EUR_JPY,GBP_JPY,AUD_JPY")
    p_snapshot.add_argument("--output", type=Path, default=None)

    p_charts = sub.add_parser("pair-charts", help="Compute multi-timeframe indicator scores per pair.")
    p_charts.add_argument("--pairs", default="USD_JPY,EUR_USD,GBP_USD,AUD_USD,EUR_JPY,GBP_JPY,AUD_JPY")
    p_charts.add_argument("--timeframes", default="M5,M15,H1")
    p_charts.add_argument("--count", type=int, default=200)
    p_charts.add_argument("--output", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_charts.add_argument("--report", type=Path, default=DEFAULT_PAIR_CHARTS_REPORT)

    p_mine = sub.add_parser("mine-strategy", help="Mine legacy evidence into a strategy profile.")
    p_mine.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_mine.add_argument("--report", type=Path, default=DEFAULT_STRATEGY_REPORT)
    p_mine.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)

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
    p_replay.add_argument("--max-days", type=int, default=None)
    p_replay.add_argument("--output", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_replay.add_argument("--report", type=Path, default=DEFAULT_REPLAY_BACKTEST_REPORT)

    p_coverage = sub.add_parser("optimize-coverage", help="Measure live-ready target coverage and emit gap tasks.")
    p_coverage.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_coverage.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_coverage.add_argument("--replay", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_coverage.add_argument("--output", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_coverage.add_argument("--report", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION_REPORT)

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
    p_gpt.add_argument("--decision-response", type=Path, default=None)
    p_gpt.add_argument("--max-lanes", type=int, default=8)
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
    p_auto.add_argument("--gpt-max-lanes", type=int, default=8)
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
                    "deterministic_lane_id": summary.deterministic_lane_id,
                    "decision_source": summary.decision_source,
                    "sent": summary.sent,
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
                "- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (H1>M15>M5).",
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
    if args.command == "mine-strategy":
        summary = StrategyMiner(args.db, args.report, args.profile).run()
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
            max_loss_jpy=args.max_loss if args.max_loss is not None else RiskPolicy().max_loss_jpy,
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
    return resolve_max_loss_jpy(
        max_loss_jpy=max_loss_jpy,
        max_loss_pct=max_loss_pct,
        equity_jpy=risk_equity_jpy,
        default_max_loss_jpy=RiskPolicy().max_loss_jpy,
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
