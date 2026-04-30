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
from quant_rabbit.legacy.importer import LegacyImporter
from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_HISTORY_DB,
    DEFAULT_IMPORT_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_LEGACY_ARCHIVE,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_STRATEGY_REPORT,
)
from quant_rabbit.risk import RiskEngine
from quant_rabbit.strategy.ensemble import CampaignPlanner
from quant_rabbit.strategy.intent_generator import IntentGenerator
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.miner import StrategyMiner
from quant_rabbit.strategy.profile import StrategyProfile
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter


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

    p_mine = sub.add_parser("mine-strategy", help="Mine legacy evidence into a strategy profile.")
    p_mine.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_mine.add_argument("--report", type=Path, default=DEFAULT_STRATEGY_REPORT)
    p_mine.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)

    p_story = sub.add_parser("mine-market-stories", help="Mine narrative, market regime, and chart-story evidence.")
    p_story.add_argument("--archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    p_story.add_argument("--report", type=Path, default=DEFAULT_MARKET_STORY_REPORT)
    p_story.add_argument("--profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)

    p_campaign = sub.add_parser("plan-campaign", help="Build a multi-desk daily campaign plan.")
    p_campaign.add_argument("--start-balance", type=float, required=True)
    p_campaign.add_argument("--target-return-pct", type=float, default=10.0)
    p_campaign.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_campaign.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_campaign.add_argument("--report", type=Path, default=DEFAULT_CAMPAIGN_REPORT)
    p_campaign.add_argument("--plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)

    p_intents = sub.add_parser("generate-intents", help="Generate dry-run order intents from campaign lanes.")
    p_intents.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    p_intents.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_intents.add_argument("--snapshot", type=Path)
    p_intents.add_argument("--output", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_intents.add_argument("--report", type=Path, default=DEFAULT_ORDER_INTENT_REPORT)
    p_intents.add_argument("--max-candidates", type=int, default=12)

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
    p_stage.add_argument("--send", action="store_true")
    p_stage.add_argument("--confirm-live", action="store_true")

    p_auto = sub.add_parser("autotrade-cycle", help="Run one safe automated trading cycle.")
    p_auto.add_argument("--send", action="store_true")
    p_auto.add_argument("--report", type=Path, default=DEFAULT_AUTOTRADE_REPORT)

    p_risk = sub.add_parser("risk-dry-run", help="Validate an order intent against a JSON snapshot.")
    p_risk.add_argument("--intent", type=Path, required=True)
    p_risk.add_argument("--snapshot", type=Path, required=True)
    p_risk.add_argument("--for-live-send", action="store_true")
    p_risk.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_risk.add_argument("--no-strategy-profile", action="store_true")

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
        summary = LiveOrderGateway(
            client=OandaExecutionClient(),
            strategy_profile=args.strategy_profile,
            output_path=args.output,
            report_path=args.report,
            live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
        ).run(intents_path=args.intents, lane_id=args.lane_id, send=args.send, confirm_live=args.confirm_live)
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
        summary = AutoTradeCycle(
            report_path=args.report,
            live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
        ).run(send=args.send)
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "report_path": str(summary.report_path),
                    "snapshot_path": str(summary.snapshot_path),
                    "intents_path": str(summary.intents_path),
                    "selected_lane_id": summary.selected_lane_id,
                    "sent": summary.sent,
                    "positions": summary.positions,
                    "orders": summary.orders,
                    "live_ready": summary.live_ready,
                    "canceled_orders": list(summary.canceled_orders),
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
            "NO_LIVE_READY_INTENT",
            "NO_TRADE",
        } else 2
    if args.command == "plan-campaign":
        summary = CampaignPlanner(
            strategy_profile=args.strategy_profile,
            market_story_profile=args.market_story_profile,
            report_path=args.report,
            plan_path=args.plan,
        ).run(start_balance_jpy=args.start_balance, target_return_pct=args.target_return_pct)
        print(
            json.dumps(
                {
                    "report_path": str(summary.report_path),
                    "plan_path": str(summary.plan_path),
                    "target_jpy": summary.target_jpy,
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
        summary = MarketStoryMiner(args.archive, args.report, args.profile).run()
        print(
            json.dumps(
                {
                    "archive": str(summary.archive),
                    "report_path": str(summary.report_path),
                    "profile_path": str(summary.profile_path),
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
        snapshot = OandaReadOnlyClient().snapshot(pairs)
        text = _snapshot_to_json(snapshot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
            print(
                json.dumps(
                    {
                        "output_path": str(args.output),
                        "positions": len(snapshot.positions),
                        "orders": len(snapshot.orders),
                        "quotes": len(snapshot.quotes),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(text)
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
    if args.command == "risk-dry-run":
        intent = _intent_from_json(json.loads(args.intent.read_text()))
        snapshot = _snapshot_from_json(json.loads(args.snapshot.read_text()))
        engine = RiskEngine(live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1")
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
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=tuple(positions),
        orders=tuple(orders),
        quotes=quotes,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
