from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.broker.position_execution import PositionProtectionGateway
from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POSITION_EXECUTION_REPORT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_DECISION,
    DEFAULT_TRADER_DECISION_REPORT,
    DEFAULT_TRADER_SETTINGS,
    ROOT,
)
from quant_rabbit.gpt_trader import GPTTraderBrain, TraderModelProvider
from quant_rabbit.risk import RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.target import DailyTargetLedger, DailyTargetSummary
from quant_rabbit.strategy.intent_generator import IntentGenerator
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.position_manager import PositionManager
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter
from quant_rabbit.strategy.trader_brain import (
    ACTION_NO_TRADE,
    ACTION_SEND_ENTRY,
    LaneScore,
    TraderBrain,
    TraderDecision,
    load_trader_settings,
)


DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"
PENDING_ENTRY_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}

# Per AGENT_CONTRACT §6 / §3.5: structural / contract-named blockers are the
# only hard reasons to keep a LIVE_READY lane out of the GPT prefilter set.
# Anything else (mining edge quality, narrative, capture-rate caution) is a
# discretionary penalty that already sizes the lane down through score →
# size_multiple, and must not be re-stacked as a prose veto. These patterns
# are matched as substrings (case-sensitive) against LaneScore.blockers — they
# come from `_score_lane`, `_discretionary_gate_check`, and
# `_exposure_blockers` in `quant_rabbit.strategy.trader_brain`.
_PREFILTER_HARD_BLOCKER_PATTERNS = (
    # §7 lane completeness — every executable lane must include thesis,
    # context, geometry, and units; missing any is a hard veto.
    "missing trader thesis",
    "missing market context",
    "incomplete market context",
    "missing TP/SL geometry",
    "missing executable units",
    # §11 strategy receipts — profile / lane gating.
    "missing strategy profile",
    "strategy profile is not live-eligible",
    "campaign lane is not executable",
    # §9 lane status — anything not LIVE_READY shouldn't be in the set anyway,
    # but keep an explicit guard.
    "intent status is",
    "receipt is not live-ready",
    # §9 exposure blockers — open or pending exposure must be reconciled
    # before a fresh entry, regardless of GPT discretion.
    "open position exists",
    "pending entry exists",
)


def _is_hard_prefilter_blocker(blocker: str) -> bool:
    return any(pattern in blocker for pattern in _PREFILTER_HARD_BLOCKER_PATTERNS)


def _passes_gpt_prefilter(score: LaneScore) -> bool:
    """Whether this lane is eligible to be picked by GPT.

    Widens beyond ACTION_SEND_ENTRY so that LIVE_READY lanes carrying only
    discretionary penalties (narrative, mined-edge caution, capture rate)
    remain available — per AGENT_CONTRACT §6 those are sized down via
    size_multiple, not blocked in prose. Hard structural blockers
    (`_PREFILTER_HARD_BLOCKER_PATTERNS`) keep the lane out.
    """
    if score.status != "LIVE_READY":
        return False
    if score.action == ACTION_SEND_ENTRY:
        return True
    if score.action != ACTION_NO_TRADE:
        return False
    return not any(_is_hard_prefilter_blocker(b) for b in score.blockers)


@dataclass(frozen=True)
class AutoTradeCycleSummary:
    status: str
    report_path: Path
    snapshot_path: Path
    intents_path: Path
    selected_lane_id: str | None
    deterministic_lane_id: str | None
    sent: bool
    positions: int
    orders: int
    live_ready: int
    decision_source: str = "deterministic"
    selected_lane_score: float | None = None
    selected_lane_size_multiple: float | None = None
    canceled_orders: tuple[str, ...] = ()
    receipt_promotions: int = 0
    position_management_action: str | None = None
    position_execution_status: str | None = None
    position_execution_sent: bool = False
    target_status: str | None = None
    target_remaining_jpy: float | None = None
    target_progress_pct: float | None = None
    gpt_status: str | None = None
    gpt_action: str | None = None
    gpt_allowed: bool | None = None
    gpt_issues: int | None = None
    gpt_error: str | None = None
    gpt_wait_retries: int = 0
    gpt_recovery_source: str | None = None


@dataclass(frozen=True)
class GptHandoffSummary:
    status: str
    action: str | None
    selected_lane_id: str | None
    allowed: bool
    issues: int
    error: str | None = None


class AutoTradeCycle:
    """One safe automated trading cycle.

    The cycle can add only when existing exposure is protected, trader-owned, and
    still inside portfolio risk validation. Pending entry orders remain monitor-only.
    """

    def __init__(
        self,
        *,
        client: OandaExecutionClient | None = None,
        snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        intent_report_path: Path = DEFAULT_ORDER_INTENT_REPORT,
        decision_path: Path = DEFAULT_TRADER_DECISION,
        decision_report_path: Path = DEFAULT_TRADER_DECISION_REPORT,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_management_report_path: Path = DEFAULT_POSITION_MANAGEMENT_REPORT,
        position_execution_path: Path = DEFAULT_POSITION_EXECUTION,
        position_execution_report_path: Path = DEFAULT_POSITION_EXECUTION_REPORT,
        live_order_output_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        live_order_report_path: Path = DEFAULT_LIVE_ORDER_STAGE_REPORT,
        report_path: Path = DEFAULT_AUTOTRADE_REPORT,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        trader_settings_path: Path = DEFAULT_TRADER_SETTINGS,
        receipt_promotion_report_path: Path = DEFAULT_RECEIPT_PROMOTION_REPORT,
        target_state_path: Path | None = None,
        target_report_path: Path | None = None,
        use_gpt_trader: bool = False,
        gpt_provider: TraderModelProvider | None = None,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        gpt_decision_report_path: Path = DEFAULT_GPT_TRADER_DECISION_REPORT,
        gpt_target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        gpt_max_lanes: int = 8,
        gpt_wait_retry_limit: int = 2,
        refresh_market_story: bool = True,
        market_news_root: Path | None = None,
        live_enabled: bool = False,
        max_loss_jpy: float | None = None,
        max_loss_pct: float | None = None,
        risk_equity_jpy: float | None = None,
    ) -> None:
        self.client = client or OandaExecutionClient()
        self.snapshot_path = snapshot_path
        self.intents_path = intents_path
        self.intent_report_path = intent_report_path
        self.decision_path = decision_path
        self.decision_report_path = decision_report_path
        self.position_management_path = position_management_path
        self.position_management_report_path = position_management_report_path
        self.position_execution_path = position_execution_path
        self.position_execution_report_path = position_execution_report_path
        self.live_order_output_path = live_order_output_path
        self.live_order_report_path = live_order_report_path
        self.report_path = report_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.trader_settings_path = trader_settings_path
        self.receipt_promotion_report_path = receipt_promotion_report_path
        self.target_state_path = target_state_path
        self.target_report_path = target_report_path
        self.use_gpt_trader = use_gpt_trader
        self.gpt_provider = gpt_provider
        self.gpt_decision_path = gpt_decision_path
        self.gpt_decision_report_path = gpt_decision_report_path
        self.gpt_target_state_path = gpt_target_state_path
        self.gpt_max_lanes = gpt_max_lanes
        self.gpt_wait_retry_limit = gpt_wait_retry_limit
        self.refresh_market_story = refresh_market_story
        self.market_news_root = market_news_root if market_news_root is not None else ROOT / "logs"
        self.live_enabled = live_enabled
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy

    def run(self, *, send: bool = False) -> AutoTradeCycleSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        pairs = ("AUD_JPY", "AUD_USD", "EUR_JPY", "EUR_USD", "GBP_JPY", "GBP_USD", "USD_JPY")
        snapshot = self.client.snapshot(pairs)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
        target_summary = self._update_target_state(snapshot)
        if self.refresh_market_story:
            self._market_story_miner().run()
        positions = len(snapshot.positions)
        orders = len(snapshot.orders)
        pending_entries = _pending_entry_order_count(snapshot)
        resolved_max_loss_jpy = self._resolve_max_loss_jpy()
        if not positions and not pending_entries and target_summary and target_summary.status == "TARGET_REACHED_PROTECT":
            summary = AutoTradeCycleSummary(
                status="TARGET_REACHED_PROTECT",
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=None,
                deterministic_lane_id=None,
                sent=False,
                positions=positions,
                orders=orders,
                live_ready=0,
                receipt_promotions=0,
                target_status=target_summary.status,
                target_remaining_jpy=target_summary.remaining_target_jpy,
                target_progress_pct=target_summary.progress_pct,
            )
            self._write_report(summary, generated_at)
            return summary

        intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(snapshot_path=self.snapshot_path)
        position_decision = None
        position_execution = None
        if positions:
            decision = self._brain().run(snapshot)
            position_decision = self._position_manager().run(snapshot)
            position_execution = self._position_gateway().run(
                decision=position_decision,
                snapshot=snapshot,
                send=send,
            )
            canceled_orders: list[str] = []
            if (
                pending_entries
                and send
                and self.live_enabled
                and not position_execution.sent
                and position_execution.status == "NO_ACTION"
                and decision.pending_cancel_order_ids
            ):
                for order_id in decision.pending_cancel_order_ids:
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
            if (
                position_execution.sent
                or position_execution.status in {"STAGED", "BLOCKED"}
                or not _portfolio_add_allowed(snapshot)
                or pending_entries
            ):
                status = "MONITOR_ONLY_EXPOSURE_OPEN"
                if canceled_orders:
                    status = "CANCELED_CONTAMINATED_PENDING"
                elif position_execution.sent:
                    status = "POSITION_ACTION_SENT"
                elif position_execution.status == "STAGED":
                    status = "POSITION_ACTION_STAGED"
                elif position_execution.status == "BLOCKED":
                    status = "POSITION_ACTION_BLOCKED"
                summary = AutoTradeCycleSummary(
                    status=status,
                    report_path=self.report_path,
                    snapshot_path=self.snapshot_path,
                    intents_path=self.intents_path,
                    selected_lane_id=None,
                    deterministic_lane_id=None,
                    sent=False,
                    positions=positions,
                    orders=orders,
                    live_ready=intent_summary.live_ready,
                    canceled_orders=tuple(canceled_orders),
                    receipt_promotions=0,
                    position_management_action=position_decision.action,
                    position_execution_status=position_execution.status,
                    position_execution_sent=position_execution.sent,
                    target_status=target_summary.status if target_summary else None,
                    target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                    target_progress_pct=target_summary.progress_pct if target_summary else None,
                )
                self._write_report(summary, generated_at)
                return summary
        if pending_entries:
            decision = self._brain().run(snapshot)
            position_decision = self._position_manager().run(snapshot)
            position_execution = self._position_gateway().run(
                decision=position_decision,
                snapshot=snapshot,
                send=send and positions > 0,
            )
            canceled_orders: list[str] = []
            status = "MONITOR_ONLY_EXPOSURE_OPEN"
            if position_execution.sent:
                status = "POSITION_ACTION_SENT"
            elif position_execution.status == "STAGED":
                status = "POSITION_ACTION_STAGED"
            elif position_execution.status == "BLOCKED":
                status = "POSITION_ACTION_BLOCKED"
            if send and self.live_enabled and positions == 0 and decision.pending_cancel_order_ids:
                for order_id in decision.pending_cancel_order_ids:
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
                status = "CANCELED_CONTAMINATED_PENDING"
            summary = AutoTradeCycleSummary(
                status=status,
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=None,
                deterministic_lane_id=None,
                sent=False,
                positions=positions,
                orders=orders,
                live_ready=intent_summary.live_ready,
                canceled_orders=tuple(canceled_orders),
                receipt_promotions=0,
                position_management_action=position_decision.action,
                position_execution_status=position_execution.status,
                position_execution_sent=position_execution.sent,
                target_status=target_summary.status if target_summary else None,
                target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                target_progress_pct=target_summary.progress_pct if target_summary else None,
            )
            self._write_report(summary, generated_at)
            return summary

        promotion_summary = self._receipt_promoter().run()
        if promotion_summary.promoted:
            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(snapshot_path=self.snapshot_path)
        decision = self._brain().run(snapshot)
        deterministic_lane_id = decision.selected_lane_id if decision.action == ACTION_SEND_ENTRY else None
        selected_lane_id = deterministic_lane_id
        selected_lane_score = decision.selected_lane_score
        selected_lane_size_multiple = decision.selected_lane_size_multiple
        gpt_summary = None
        gpt_wait_retries = 0
        gpt_recovery_source = None

        if selected_lane_id is None:
            if not self.use_gpt_trader:
                summary = AutoTradeCycleSummary(
                    status=decision.action if decision.action != ACTION_SEND_ENTRY else "NO_LIVE_READY_INTENT",
                    report_path=self.report_path,
                    snapshot_path=self.snapshot_path,
                    intents_path=self.intents_path,
                    selected_lane_id=None,
                    deterministic_lane_id=deterministic_lane_id,
                    sent=False,
                    positions=positions,
                    orders=orders,
                    live_ready=intent_summary.live_ready,
                    receipt_promotions=promotion_summary.promoted,
                    position_management_action=position_decision.action if position_decision else None,
                    position_execution_status=position_execution.status if position_execution else None,
                    position_execution_sent=position_execution.sent if position_execution else False,
                    target_status=target_summary.status if target_summary else None,
                    target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                    target_progress_pct=target_summary.progress_pct if target_summary else None,
                )
                self._write_report(summary, generated_at)
                return summary

            gpt_summary = self._run_gpt_handoff()
            if gpt_summary.status != "ACCEPTED" or not gpt_summary.allowed:
                summary = AutoTradeCycleSummary(
                    status="GPT_REJECTED",
                    report_path=self.report_path,
                    snapshot_path=self.snapshot_path,
                    intents_path=self.intents_path,
                    selected_lane_id=None,
                    deterministic_lane_id=deterministic_lane_id,
                    sent=False,
                    positions=positions,
                    orders=orders,
                    live_ready=intent_summary.live_ready,
                    decision_source="gpt_trader",
                    receipt_promotions=promotion_summary.promoted,
                    position_management_action=position_decision.action if position_decision else None,
                    position_execution_status=position_execution.status if position_execution else None,
                    position_execution_sent=position_execution.sent if position_execution else False,
                    target_status=target_summary.status if target_summary else None,
                    target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                    target_progress_pct=target_summary.progress_pct if target_summary else None,
                    gpt_status=gpt_summary.status,
                    gpt_action=gpt_summary.action,
                    gpt_allowed=gpt_summary.allowed,
                    gpt_issues=gpt_summary.issues,
                    gpt_error=gpt_summary.error,
                    gpt_wait_retries=gpt_wait_retries,
                    gpt_recovery_source=gpt_recovery_source,
                )
                self._write_report(summary, generated_at)
                return summary

            if gpt_summary.action == "TRADE" and gpt_summary.selected_lane_id:
                selected_lane_id = gpt_summary.selected_lane_id
                selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                    decision=decision, lane_id=selected_lane_id
                )
            elif gpt_summary.action in {"WAIT", "REQUEST_EVIDENCE"}:
                target_is_pursue = (
                    target_summary is not None
                    and target_summary.status == "PURSUE_TARGET"
                    and target_summary.remaining_target_jpy > 0
                )
                if self.gpt_wait_retry_limit > 0 and target_is_pursue:
                    for attempt in range(1, self.gpt_wait_retry_limit + 1):
                        gpt_wait_retries = attempt
                        intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(
                            snapshot_path=self.snapshot_path,
                            max_candidates=12,
                        )
                        decision = self._brain().run(snapshot)
                        selected_lane_score = decision.selected_lane_score
                        selected_lane_size_multiple = decision.selected_lane_size_multiple
                        if decision.action == ACTION_SEND_ENTRY and decision.selected_lane_id:
                            selected_lane_id = decision.selected_lane_id
                            gpt_summary = GptHandoffSummary(
                                status="ACCEPTED",
                                action="TRADE",
                                selected_lane_id=decision.selected_lane_id,
                                allowed=True,
                                issues=0,
                            )
                            gpt_recovery_source = f"DETERMINISTIC_WAIT_RECOVERY_ATTEMPT_{attempt}"
                            break
                        if attempt == self.gpt_wait_retry_limit:
                            break
                        retry_summary = self._run_gpt_handoff()
                        if retry_summary.status != "ACCEPTED" or not retry_summary.allowed:
                            summary = AutoTradeCycleSummary(
                                status="GPT_REJECTED",
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                deterministic_lane_id=deterministic_lane_id,
                                sent=False,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                decision_source="gpt_trader",
                                receipt_promotions=promotion_summary.promoted,
                                position_management_action=position_decision.action if position_decision else None,
                                position_execution_status=position_execution.status if position_execution else None,
                                position_execution_sent=position_execution.sent if position_execution else False,
                                target_status=target_summary.status if target_summary else None,
                                target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                                target_progress_pct=target_summary.progress_pct if target_summary else None,
                                gpt_status=retry_summary.status,
                                gpt_action=retry_summary.action,
                                gpt_allowed=retry_summary.allowed,
                                gpt_issues=retry_summary.issues,
                                gpt_error=retry_summary.error,
                                gpt_wait_retries=gpt_wait_retries,
                                gpt_recovery_source="GPT_RETRY_REJECTED",
                            )
                            self._write_report(summary, generated_at)
                            return summary
                        if retry_summary.action == "TRADE" and retry_summary.selected_lane_id:
                            selected_lane_id = retry_summary.selected_lane_id
                            selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                                decision=decision, lane_id=selected_lane_id
                            )
                            gpt_summary = retry_summary
                            gpt_recovery_source = f"GPT_RETRY_TRADE_ATTEMPT_{attempt}"
                            break
                        gpt_summary = retry_summary

                    if gpt_summary.action not in {"TRADE", "WAIT", "REQUEST_EVIDENCE"}:
                        summary = AutoTradeCycleSummary(
                            status=f"GPT_{gpt_summary.action or 'NO_TRADE'}",
                            report_path=self.report_path,
                            snapshot_path=self.snapshot_path,
                            intents_path=self.intents_path,
                            selected_lane_id=None,
                            deterministic_lane_id=deterministic_lane_id,
                            sent=False,
                            positions=positions,
                            orders=orders,
                            live_ready=intent_summary.live_ready,
                            decision_source="gpt_trader",
                            receipt_promotions=promotion_summary.promoted,
                            position_management_action=position_decision.action if position_decision else None,
                            position_execution_status=position_execution.status if position_execution else None,
                            position_execution_sent=position_execution.sent if position_execution else False,
                            target_status=target_summary.status if target_summary else None,
                            target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                            target_progress_pct=target_summary.progress_pct if target_summary else None,
                            gpt_status=gpt_summary.status,
                            gpt_action=gpt_summary.action,
                            gpt_allowed=gpt_summary.allowed,
                            gpt_issues=gpt_summary.issues,
                            gpt_error=gpt_summary.error,
                            gpt_wait_retries=gpt_wait_retries,
                            gpt_recovery_source=gpt_recovery_source,
                        )
                        self._write_report(summary, generated_at)
                        return summary

                    if not selected_lane_id:
                        summary = AutoTradeCycleSummary(
                            status=f"GPT_{gpt_summary.action or 'NO_TRADE'}",
                            report_path=self.report_path,
                            snapshot_path=self.snapshot_path,
                            intents_path=self.intents_path,
                            selected_lane_id=None,
                            deterministic_lane_id=deterministic_lane_id,
                            sent=False,
                            positions=positions,
                            orders=orders,
                            live_ready=intent_summary.live_ready,
                            decision_source="gpt_trader",
                            receipt_promotions=promotion_summary.promoted,
                            position_management_action=position_decision.action if position_decision else None,
                            position_execution_status=position_execution.status if position_execution else None,
                            position_execution_sent=position_execution.sent if position_execution else False,
                            target_status=target_summary.status if target_summary else None,
                            target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                            target_progress_pct=target_summary.progress_pct if target_summary else None,
                            gpt_status=gpt_summary.status,
                            gpt_action=gpt_summary.action,
                            gpt_allowed=gpt_summary.allowed,
                            gpt_issues=gpt_summary.issues,
                            gpt_error=gpt_summary.error,
                            gpt_wait_retries=gpt_wait_retries,
                            gpt_recovery_source=gpt_recovery_source,
                        )
                        self._write_report(summary, generated_at)
                        return summary
                else:
                    summary = AutoTradeCycleSummary(
                        status=f"GPT_{gpt_summary.action or 'NO_TRADE'}",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        decision_source="gpt_trader",
                        receipt_promotions=promotion_summary.promoted,
                        position_management_action=position_decision.action if position_decision else None,
                        position_execution_status=position_execution.status if position_execution else None,
                        position_execution_sent=position_execution.sent if position_execution else False,
                        target_status=target_summary.status if target_summary else None,
                        target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                        target_progress_pct=target_summary.progress_pct if target_summary else None,
                        gpt_status=gpt_summary.status,
                        gpt_action=gpt_summary.action,
                        gpt_allowed=gpt_summary.allowed,
                        gpt_issues=gpt_summary.issues,
                        gpt_error=gpt_summary.error,
                        gpt_wait_retries=gpt_wait_retries,
                        gpt_recovery_source=gpt_recovery_source,
                    )
                    self._write_report(summary, generated_at)
                    return summary
            else:
                summary = AutoTradeCycleSummary(
                    status=f"GPT_{gpt_summary.action or 'NO_TRADE'}",
                    report_path=self.report_path,
                    snapshot_path=self.snapshot_path,
                    intents_path=self.intents_path,
                    selected_lane_id=None,
                    deterministic_lane_id=deterministic_lane_id,
                    sent=False,
                    positions=positions,
                    orders=orders,
                    live_ready=intent_summary.live_ready,
                    decision_source="gpt_trader",
                    receipt_promotions=promotion_summary.promoted,
                    position_management_action=position_decision.action if position_decision else None,
                    position_execution_status=position_execution.status if position_execution else None,
                    position_execution_sent=position_execution.sent if position_execution else False,
                    target_status=target_summary.status if target_summary else None,
                    target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                    target_progress_pct=target_summary.progress_pct if target_summary else None,
                    gpt_status=gpt_summary.status,
                    gpt_action=gpt_summary.action,
                    gpt_allowed=gpt_summary.allowed,
                    gpt_issues=gpt_summary.issues,
                    gpt_error=gpt_summary.error,
                    gpt_wait_retries=gpt_wait_retries,
                    gpt_recovery_source=gpt_recovery_source,
                )
                self._write_report(summary, generated_at)
                return summary
        if selected_lane_id:
            if self.use_gpt_trader:
                prefiltered_lane_ids = {
                    item.lane_id for item in decision.scores if _passes_gpt_prefilter(item)
                }
                if gpt_summary is None:
                    gpt_summary = self._run_gpt_handoff()
                if gpt_summary.status != "ACCEPTED" or not gpt_summary.allowed:
                    summary = AutoTradeCycleSummary(
                        status="GPT_REJECTED",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        decision_source="gpt_trader",
                        receipt_promotions=promotion_summary.promoted,
                        position_management_action=position_decision.action if position_decision else None,
                        position_execution_status=position_execution.status if position_execution else None,
                        position_execution_sent=position_execution.sent if position_execution else False,
                        target_status=target_summary.status if target_summary else None,
                        target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                        target_progress_pct=target_summary.progress_pct if target_summary else None,
                        gpt_status=gpt_summary.status,
                        gpt_action=gpt_summary.action,
                        gpt_allowed=gpt_summary.allowed,
                        gpt_issues=gpt_summary.issues,
                        gpt_error=gpt_summary.error,
                        gpt_wait_retries=gpt_wait_retries,
                        gpt_recovery_source=gpt_recovery_source,
                    )
                    self._write_report(summary, generated_at)
                    return summary
                if gpt_summary.action != "TRADE" or not gpt_summary.selected_lane_id:
                    summary = AutoTradeCycleSummary(
                        status=f"GPT_{gpt_summary.action or 'NO_TRADE'}",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        decision_source="gpt_trader",
                        receipt_promotions=promotion_summary.promoted,
                        position_management_action=position_decision.action if position_decision else None,
                        position_execution_status=position_execution.status if position_execution else None,
                        position_execution_sent=position_execution.sent if position_execution else False,
                        target_status=target_summary.status if target_summary else None,
                        target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                        target_progress_pct=target_summary.progress_pct if target_summary else None,
                        gpt_status=gpt_summary.status,
                        gpt_action=gpt_summary.action,
                        gpt_allowed=gpt_summary.allowed,
                        gpt_issues=gpt_summary.issues,
                        gpt_error=gpt_summary.error,
                        gpt_wait_retries=gpt_wait_retries,
                        gpt_recovery_source=gpt_recovery_source,
                    )
                    self._write_report(summary, generated_at)
                    return summary
                if gpt_summary.selected_lane_id not in prefiltered_lane_ids:
                    summary = AutoTradeCycleSummary(
                        status="GPT_DECISION_NOT_PREFILTERED",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        decision_source="gpt_trader",
                        receipt_promotions=promotion_summary.promoted,
                        position_management_action=position_decision.action if position_decision else None,
                        position_execution_status=position_execution.status if position_execution else None,
                        position_execution_sent=position_execution.sent if position_execution else False,
                        target_status=target_summary.status if target_summary else None,
                        target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                        target_progress_pct=target_summary.progress_pct if target_summary else None,
                        gpt_status=gpt_summary.status,
                        gpt_action=gpt_summary.action,
                        gpt_allowed=gpt_summary.allowed,
                        gpt_issues=gpt_summary.issues,
                        gpt_error=gpt_summary.error,
                        gpt_wait_retries=gpt_wait_retries,
                        gpt_recovery_source=gpt_recovery_source,
                    )
                    self._write_report(summary, generated_at)
                    return summary
                if gpt_summary.selected_lane_id:
                    selected_lane_id = gpt_summary.selected_lane_id
                    selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                        decision=decision, lane_id=selected_lane_id
                    )

        order_summary = LiveOrderGateway(
            client=self.client,
            strategy_profile=self.strategy_profile_path,
            output_path=self.live_order_output_path,
            report_path=self.live_order_report_path,
            live_enabled=self.live_enabled,
            max_loss_jpy=resolved_max_loss_jpy,
        ).run(
            intents_path=self.intents_path,
            lane_id=selected_lane_id,
            size_multiple=selected_lane_size_multiple if selected_lane_size_multiple is not None else 1.0,
            send=send,
            confirm_live=send,
        )
        summary = AutoTradeCycleSummary(
            status=order_summary.status,
            report_path=self.report_path,
            snapshot_path=self.snapshot_path,
            intents_path=self.intents_path,
            selected_lane_id=selected_lane_id,
            selected_lane_score=selected_lane_score,
            selected_lane_size_multiple=selected_lane_size_multiple,
            deterministic_lane_id=deterministic_lane_id,
            sent=order_summary.sent,
            positions=positions,
            orders=orders,
            live_ready=intent_summary.live_ready,
            decision_source="gpt_trader" if self.use_gpt_trader else "deterministic",
            receipt_promotions=promotion_summary.promoted,
            position_management_action=position_decision.action if position_decision else None,
            position_execution_status=position_execution.status if position_execution else None,
            position_execution_sent=position_execution.sent if position_execution else False,
            target_status=target_summary.status if target_summary else None,
            target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
            target_progress_pct=target_summary.progress_pct if target_summary else None,
            gpt_status=gpt_summary.status if gpt_summary else None,
            gpt_action=gpt_summary.action if gpt_summary else None,
            gpt_allowed=gpt_summary.allowed if gpt_summary else None,
            gpt_issues=gpt_summary.issues if gpt_summary else None,
            gpt_error=gpt_summary.error if gpt_summary else None,
            gpt_wait_retries=gpt_wait_retries,
            gpt_recovery_source=gpt_recovery_source,
        )
        self._write_report(summary, generated_at)
        return summary

    def _write_report(self, summary: AutoTradeCycleSummary, generated_at: str) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Autotrade Cycle Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Status: `{summary.status}`",
            f"- Positions: `{summary.positions}`",
            f"- Orders: `{summary.orders}`",
            f"- Live-ready intents: `{summary.live_ready}`",
            f"- Receipt promotions: `{summary.receipt_promotions}`",
            f"- Decision source: `{summary.decision_source}`",
            f"- Deterministic lane: `{summary.deterministic_lane_id}`",
            f"- Selected lane: `{summary.selected_lane_id}`",
            f"- Selected lane score: `{summary.selected_lane_score}`",
            f"- Selected lane size multiple: `{summary.selected_lane_size_multiple}`",
            f"- Sent: `{summary.sent}`",
            f"- Canceled orders: `{', '.join(summary.canceled_orders) if summary.canceled_orders else 'none'}`",
            f"- Position management: `{summary.position_management_action or 'none'}`",
            f"- Position execution: `{summary.position_execution_status or 'none'}` sent=`{summary.position_execution_sent}`",
            f"- Daily target: `{summary.target_status or 'not configured'}` remaining=`{summary.target_remaining_jpy}` progress_pct=`{summary.target_progress_pct}`",
            f"- GPT trader: status=`{summary.gpt_status or 'not used'}` action=`{summary.gpt_action}` allowed=`{summary.gpt_allowed}` issues=`{summary.gpt_issues}`",
            f"- GPT error: `{summary.gpt_error or 'none'}`",
            f"- GPT wait recovery attempts: `{summary.gpt_wait_retries}`",
            f"- GPT recovery source: `{summary.gpt_recovery_source or 'none'}`",
            f"- Market story refresh: `{self.refresh_market_story}` (source: `{self.market_news_root}`)",
            "",
            "## Cycle Contract",
            "",
                "- Protected trader-owned positions may add only through portfolio risk validation; pending entries remain monitor-only.",
            "- Open positions are handed to PositionManager first, then the protection gateway may close, repair protection, or tighten SL when the action is risk-reducing.",
            "- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.",
            "- If flat, risk-repair or trigger receipts may promote the strategy profile before TraderBrain compares lanes.",
            "- If the daily target is already reached while flat, the cycle records protection-first no-send status and adds no fresh risk.",
            "- If GPT trader handoff is enabled, the selected lane must also be an accepted GPT `TRADE` decision from the deterministic prefilter set.",
            "- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.",
        ]
        self.report_path.write_text("\n".join(lines) + "\n")

    def _intent_generator(self, max_loss_jpy: float | None = None) -> IntentGenerator:
        return IntentGenerator(
            campaign_plan=self.campaign_plan_path,
            strategy_profile=self.strategy_profile_path,
            output_path=self.intents_path,
            report_path=self.intent_report_path,
            max_loss_jpy=max_loss_jpy,
        )

    def _market_story_miner(self) -> MarketStoryMiner:
        return MarketStoryMiner(
            report_path=DEFAULT_MARKET_STORY_REPORT,
            profile_path=self.market_story_profile_path,
            news_root=self.market_news_root,
        )

    def _update_target_state(self, snapshot) -> DailyTargetSummary | None:
        if self.target_state_path is None:
            return None
        if not self.target_state_path.exists():
            return None
        report_path = self.target_report_path or DEFAULT_DAILY_TARGET_REPORT
        return DailyTargetLedger(
            state_path=self.target_state_path,
            report_path=report_path,
        ).run(snapshot=snapshot)

    def _receipt_promoter(self) -> ReceiptPromoter:
        return ReceiptPromoter(
            profile_path=self.strategy_profile_path,
            intents_path=self.intents_path,
            report_path=self.receipt_promotion_report_path,
        )

    def _brain(self) -> TraderBrain:
        return TraderBrain(
            intents_path=self.intents_path,
            campaign_plan_path=self.campaign_plan_path,
            strategy_profile_path=self.strategy_profile_path,
            market_story_profile_path=self.market_story_profile_path,
            trader_settings_path=self.trader_settings_path,
            output_path=self.decision_path,
            report_path=self.decision_report_path,
        )

    @staticmethod
    def _selected_lane_meta(
        decision: TraderDecision, lane_id: str | None
    ) -> tuple[float | None, float | None]:
        if lane_id is None:
            return None, None
        if decision.selected_lane_id == lane_id and decision.selected_lane_score is not None:
            return decision.selected_lane_score, decision.selected_lane_size_multiple
        for score in decision.scores:
            if score.lane_id == lane_id:
                return score.score, score.size_multiple
        return None, None

    def _run_gpt_handoff(self) -> GptHandoffSummary:
        try:
            summary = self._gpt_brain().run(snapshot_path=self.snapshot_path)
            return GptHandoffSummary(
                status=summary.status,
                action=summary.action,
                selected_lane_id=summary.selected_lane_id,
                allowed=summary.allowed,
                issues=summary.issues,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            return GptHandoffSummary(
                status="ERROR",
                action=None,
                selected_lane_id=None,
                allowed=False,
                issues=1,
                error=str(exc),
            )

    def _gpt_brain(self) -> GPTTraderBrain:
        return GPTTraderBrain(
            provider=self.gpt_provider,
            intents_path=self.intents_path,
            campaign_plan_path=self.campaign_plan_path,
            strategy_profile_path=self.strategy_profile_path,
            market_story_profile_path=self.market_story_profile_path,
            target_state_path=self.gpt_target_state_path,
            output_path=self.gpt_decision_path,
            report_path=self.gpt_decision_report_path,
            max_lanes=self.gpt_max_lanes,
        )

    def _resolve_max_loss_jpy(self) -> float:
        # AGENT_CONTRACT §3.5: per-trade JPY cap is *always* the
        # equity-derived `per_trade_risk_budget_jpy` from the daily-target
        # ledger when that file exists. The operator has only one knob for
        # pacing — `target_trades_per_day` on `daily-target-state` — and the
        # ledger's per-trade value is the authoritative answer for "how much
        # JPY can a single shot risk today." A `max_loss_pct` setting in
        # `trader_settings.json` is a *fallback floor* used only when the
        # ledger is missing or the operator passes an explicit CLI override,
        # otherwise it would silently shadow the per-trade split and let one
        # losing trade spend more than its allotted slice.
        explicit_jpy = self.max_loss_jpy
        explicit_pct = self.max_loss_pct
        if explicit_jpy is not None or explicit_pct is not None:
            risk_equity_jpy = self._risk_equity_jpy_for_pct()
            return resolve_max_loss_jpy(
                max_loss_jpy=explicit_jpy,
                max_loss_pct=explicit_pct,
                equity_jpy=risk_equity_jpy,
                default_max_loss_jpy=RiskPolicy().max_loss_jpy,
                label="autotrade-cycle risk cap",
            )
        ledger_cap = self._daily_risk_budget_jpy_from_target_state()
        if ledger_cap is not None:
            return ledger_cap
        # No CLI override and no ledger — fall through to trader_settings, then
        # policy literal. Preserve first-run / test compatibility: when
        # max_loss_pct is set but equity is unknown, fall back to the policy
        # literal rather than raising. This branch never executes in
        # production because the ledger is always present; it exists for the
        # narrow window before `daily-target-state` first runs.
        settings = load_trader_settings(self.trader_settings_path)
        risk_equity_jpy = self._risk_equity_jpy_for_pct()
        max_loss_jpy = settings.default_max_loss_jpy
        max_loss_pct = settings.default_max_loss_pct
        if max_loss_jpy is None and max_loss_pct is not None and risk_equity_jpy is None:
            max_loss_jpy = RiskPolicy().max_loss_jpy
            max_loss_pct = None
        return resolve_max_loss_jpy(
            max_loss_jpy=max_loss_jpy,
            max_loss_pct=max_loss_pct,
            equity_jpy=risk_equity_jpy,
            default_max_loss_jpy=RiskPolicy().max_loss_jpy,
            label="autotrade-cycle risk cap",
        )

    def _daily_risk_budget_jpy_from_target_state(self) -> float | None:
        """Return the equity-derived **per-trade** cap from the daily-target ledger.

        Per AGENT_CONTRACT §3.5 the per-trade JPY cap is
        `daily_risk_budget_jpy / target_trades_per_day` (i.e.
        `per_trade_risk_budget_jpy`), and that is the value that must flow into
        every intent's `metadata.max_loss_jpy`. Reading the whole-day
        `daily_risk_budget_jpy` here would silently let one losing trade burn
        the entire day's risk budget — exactly the failure mode this split was
        built to remove. Fall back to `daily_risk_budget_jpy` only as a
        last-resort floor for old state files that pre-date the per-trade
        split, and never invent a JPY literal.
        """
        if self.target_state_path is None or not self.target_state_path.exists():
            return None
        try:
            payload = json.loads(self.target_state_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        candidates = (
            payload.get("per_trade_risk_budget_jpy"),
            payload.get("daily_risk_budget_jpy"),
        )
        for raw in candidates:
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _risk_equity_jpy_for_pct(self) -> float | None:
        if self.risk_equity_jpy is not None:
            return self.risk_equity_jpy
        if self.target_state_path is None or not self.target_state_path.exists():
            return None
        try:
            payload = json.loads(self.target_state_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        value = payload.get("current_equity_jpy")
        if value is None:
            value = payload.get("start_balance_jpy")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _position_manager(self) -> PositionManager:
        return PositionManager(
            trader_decision_path=self.decision_path,
            output_path=self.position_management_path,
            report_path=self.position_management_report_path,
        )

    def _position_gateway(self) -> PositionProtectionGateway:
        return PositionProtectionGateway(
            client=self.client,
            output_path=self.position_execution_path,
            report_path=self.position_execution_report_path,
            live_enabled=self.live_enabled,
        )


def _snapshot_to_json(snapshot) -> str:
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
            "hedging_enabled": account.hedging_enabled,
            "fetched_at_utc": account.fetched_at_utc.isoformat(),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _pending_entry_order_count(snapshot) -> int:
    return sum(
        1
        for order in snapshot.orders
        if not order.trade_id and str(order.order_type or "").upper() in PENDING_ENTRY_TYPES
    )


def _portfolio_add_allowed(snapshot) -> bool:
    if not snapshot.positions:
        return False
    return all(
        position.owner.value == "trader" and position.stop_loss is not None and position.take_profit is not None
        for position in snapshot.positions
    )
