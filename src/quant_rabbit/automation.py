from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.broker.position_execution import PositionProtectionGateway
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
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
    DEFAULT_TRADER_JOURNAL,
    DEFAULT_TRADER_SETTINGS,
    ROOT,
)
from quant_rabbit.gpt_trader import DEFAULT_GPT_MAX_LANES, GPTTraderBrain, TraderModelProvider
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.risk import RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.target import DailyTargetLedger, DailyTargetSummary
from quant_rabbit.strategy.intent_generator import IntentGenerationSummary, IntentGenerator, _snapshot_from_json
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.position_manager import PositionManager
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter, ReceiptPromotionSummary
from quant_rabbit.strategy.trader_brain import (
    ACTION_NO_TRADE,
    ACTION_SEND_ENTRY,
    LaneScore,
    TraderBrain,
    TraderDecision,
    load_trader_settings,
)


DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"
DEFAULT_AUTOTRADE_LOCK_DIR = ROOT / ".quant_rabbit_live.lock"
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


def _is_existing_pending_blocker(blocker: str) -> bool:
    return str(blocker).startswith("pending entry exists:")


def _basket_parent_lane_id(lane_id: str | None) -> str | None:
    if not lane_id:
        return None
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
    return lane_id


def _basket_parent_lane_set(lane_ids: tuple[str, ...]) -> set[str]:
    return {
        parent
        for parent in (_basket_parent_lane_id(lane_id) for lane_id in lane_ids)
        if parent
    }


def _passes_basket_prefilter(score: LaneScore, *, allow_existing_pending: bool = False) -> bool:
    if _passes_gpt_prefilter(score):
        return True
    if not allow_existing_pending:
        return False
    if score.status != "LIVE_READY" or score.action != ACTION_NO_TRADE:
        return False
    blockers = [blocker for blocker in score.blockers if not _is_existing_pending_blocker(blocker)]
    return not any(_is_hard_prefilter_blocker(blocker) for blocker in blockers)


def _acquire_autotrade_lock(*, send: bool) -> Path | None:
    """Acquire a nonblocking live-cycle lock for direct CLI sends.

    The shell wrapper also takes this lock and sets QR_AUTOTRADE_LOCK_HELD=1 so
    the in-process guard is reentrant. Direct `autotrade-cycle --send` calls do
    not pass through the wrapper, so this closes the duplicate-send surface.
    """
    if not send or os.environ.get("QR_AUTOTRADE_LOCK_HELD") == "1":
        return None
    lock_dir = Path(os.environ.get("QR_AUTOTRADE_LOCK_DIR") or DEFAULT_AUTOTRADE_LOCK_DIR)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        existing_pid = _lock_pid(lock_dir)
        if existing_pid and _pid_is_running(existing_pid):
            raise RuntimeError(f"another autotrade cycle is already running pid={existing_pid}")
        shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            lock_dir.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(f"failed to acquire autotrade lock: {lock_dir}") from exc
    (lock_dir / "pid").write_text(f"{os.getpid()}\n")
    return lock_dir


def _lock_pid(lock_dir: Path) -> int | None:
    try:
        return int((lock_dir / "pid").read_text().strip())
    except (OSError, TypeError, ValueError):
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
    selected_lane_ids: tuple[str, ...] = ()
    sent_count: int = 0
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
    campaign_exposure_required: bool = False


@dataclass(frozen=True)
class GptHandoffSummary:
    status: str
    action: str | None
    selected_lane_id: str | None
    allowed: bool
    issues: int
    selected_lane_ids: tuple[str, ...] = ()
    cancel_order_ids: tuple[str, ...] = ()
    error: str | None = None


class AutoTradeCycle:
    """One safe automated trading cycle.

    The cycle can add only when existing exposure is protected, trader-owned, and
    still inside portfolio risk validation. Existing trader pending entries are
    basket-counted by the gateway before any additional order is staged or sent.
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
        trader_journal_path: Path = DEFAULT_TRADER_JOURNAL,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        execution_ledger_report_path: Path = DEFAULT_EXECUTION_LEDGER_REPORT,
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
        gpt_attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        gpt_max_lanes: int = DEFAULT_GPT_MAX_LANES,
        gpt_wait_retry_limit: int = 2,
        reuse_market_artifacts: bool = False,
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
        self.trader_journal_path = trader_journal_path
        self.execution_ledger_db_path = execution_ledger_db_path
        self.execution_ledger_report_path = execution_ledger_report_path
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
        self.gpt_attack_advice_path = gpt_attack_advice_path
        self.gpt_max_lanes = gpt_max_lanes
        self.gpt_wait_retry_limit = gpt_wait_retry_limit
        self.reuse_market_artifacts = reuse_market_artifacts
        self.refresh_market_story = refresh_market_story
        self.market_news_root = market_news_root if market_news_root is not None else ROOT / "logs"
        self.live_enabled = live_enabled
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy

    def run(self, *, send: bool = False) -> AutoTradeCycleSummary:
        lock_dir = _acquire_autotrade_lock(send=send)
        try:
            self._sync_execution_ledger()
            summary = self._run(send=send)
            self._record_execution_ledger_receipts()
            self._sync_execution_ledger()
            self._append_trader_journal_entry(summary)
            return summary
        except Exception:
            try:
                self._sync_execution_ledger()
            except Exception:
                pass
            raise
        finally:
            if lock_dir is not None:
                shutil.rmtree(lock_dir, ignore_errors=True)

    def _sync_execution_ledger(self) -> None:
        if not self._execution_ledger_available():
            return
        ExecutionLedger(
            db_path=self.execution_ledger_db_path,
            report_path=self.execution_ledger_report_path,
        ).sync_oanda_transactions(self.client)

    def _record_execution_ledger_receipts(self) -> None:
        if not self._execution_ledger_available():
            return
        ledger = ExecutionLedger(db_path=self.execution_ledger_db_path, report_path=self.execution_ledger_report_path)
        for kind, path in (
            ("live_order", self.live_order_output_path),
            ("position_execution", self.position_execution_path),
        ):
            ledger.record_gateway_receipt(kind=kind, receipt_path=path)

    def _execution_ledger_available(self) -> bool:
        return hasattr(self.client, "account_summary") and hasattr(self.client, "transactions_since_id")

    def _append_trader_journal_entry(self, summary: AutoTradeCycleSummary) -> None:
        """Append one JSONL line to logs/trader_journal.jsonl per cycle.

        AGENT_CONTRACT §6 / §11 require a persistent audit trail of every
        decision, basket selection, and execution outcome. The legacy archive
        carried `logs/trader_journal.jsonl`; vNext rebuilds the writer here so
        post-trade review and `mine-strategy` have something historical to
        learn from. Latest-state files (`data/live_order_request.json`,
        `data/autotrade_cycle_report.json`) get overwritten every cycle and
        cannot serve as a long-term audit trail.

        Best-effort: a journal-write failure must not break the live cycle
        since the broker remains the canonical record either way.
        """
        try:
            entry: dict = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": summary.status,
                "decision_source": summary.decision_source,
                "selected_lane_id": summary.selected_lane_id,
                "selected_lane_ids": list(summary.selected_lane_ids),
                "deterministic_lane_id": summary.deterministic_lane_id,
                "selected_lane_score": summary.selected_lane_score,
                "selected_lane_size_multiple": summary.selected_lane_size_multiple,
                "sent": summary.sent,
                "sent_count": summary.sent_count,
                "positions": summary.positions,
                "orders": summary.orders,
                "live_ready": summary.live_ready,
                "canceled_orders": list(summary.canceled_orders),
                "receipt_promotions": summary.receipt_promotions,
                "position_management_action": summary.position_management_action,
                "position_execution_status": summary.position_execution_status,
                "position_execution_sent": summary.position_execution_sent,
                "target_status": summary.target_status,
                "target_remaining_jpy": summary.target_remaining_jpy,
                "target_progress_pct": summary.target_progress_pct,
                "gpt_status": summary.gpt_status,
                "gpt_action": summary.gpt_action,
                "gpt_allowed": summary.gpt_allowed,
                "gpt_issues": summary.gpt_issues,
                "gpt_error": summary.gpt_error,
                "gpt_recovery_source": summary.gpt_recovery_source,
                "campaign_exposure_required": summary.campaign_exposure_required,
            }
            if summary.sent and self.live_order_output_path.exists():
                try:
                    request_payload = json.loads(self.live_order_output_path.read_text())
                    live_record: dict = {
                        "status": request_payload.get("status"),
                        "lane_id": request_payload.get("lane_id"),
                        "scaled_units": request_payload.get("scaled_units"),
                        "size_multiple": request_payload.get("size_multiple"),
                        "sent": request_payload.get("sent"),
                    }
                    response = request_payload.get("response") or {}
                    if isinstance(response, dict) and response:
                        live_record["response"] = {
                            k: response.get(k)
                            for k in (
                                "status",
                                "trade_id",
                                "fill_price",
                                "fill_units",
                                "reason",
                                "reject_reason",
                            )
                            if response.get(k) is not None
                        }
                    request_orders = request_payload.get("request_orders") or request_payload.get("orders")
                    if isinstance(request_orders, list) and request_orders:
                        live_record["request_orders"] = request_orders
                    entry["live_order"] = live_record
                except (json.JSONDecodeError, OSError):
                    entry["live_order_read_error"] = True
            if self.gpt_decision_path.exists():
                try:
                    gpt_payload = json.loads(self.gpt_decision_path.read_text())
                    issues = gpt_payload.get("verification_issues") or []
                    if issues:
                        entry["verification_issues"] = [
                            {
                                "code": issue.get("code"),
                                "severity": issue.get("severity"),
                                "message": issue.get("message"),
                            }
                            for issue in issues[:10]
                        ]
                except (json.JSONDecodeError, OSError):
                    pass
            if self.snapshot_path.exists():
                try:
                    snapshot_payload = json.loads(self.snapshot_path.read_text())
                    trader_positions = [
                        {
                            "trade_id": str(position.get("trade_id") or ""),
                            "pair": position.get("pair"),
                            "side": position.get("side"),
                            "units": position.get("units"),
                            "entry_price": position.get("entry_price"),
                            "stop_loss": position.get("stop_loss"),
                            "take_profit": position.get("take_profit"),
                            "unrealized_pl_jpy": position.get("unrealized_pl_jpy"),
                        }
                        for position in snapshot_payload.get("positions", [])
                        if position.get("owner") == "trader"
                    ]
                    entry["trader_positions"] = trader_positions
                except (json.JSONDecodeError, OSError):
                    pass
            self.trader_journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trader_journal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            # AGENT_CONTRACT §6: audit trail is required, but a write failure
            # must not block live execution. The broker remains canonical.
            pass

    def _run(self, *, send: bool = False) -> AutoTradeCycleSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        pairs = DEFAULT_TRADER_PAIRS
        if self.reuse_market_artifacts:
            snapshot = self._load_snapshot_artifact()
        else:
            snapshot = self._refresh_snapshot(pairs)
        target_summary = self._update_target_state(snapshot)
        if self.refresh_market_story and not self.reuse_market_artifacts:
            self._market_story_miner().run()
            # Market-story mining can read archive/news artifacts and take
            # longer than RiskPolicy.max_quote_age_seconds. Refresh immediately
            # before intent pricing so risk validation sees broker-current
            # quotes instead of blocking all lanes as STALE_QUOTE.
            snapshot = self._refresh_snapshot(pairs)
            target_summary = self._update_target_state(snapshot) or target_summary
        positions = len(snapshot.positions)
        trader_positions = _trader_position_count(snapshot)
        orders = len(snapshot.orders)
        pending_entries = _pending_entry_order_count(snapshot)
        resolved_max_loss_jpy = self._resolve_max_loss_jpy()
        if not trader_positions and not pending_entries and target_summary and target_summary.status == "TARGET_REACHED_PROTECT":
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

        if self.reuse_market_artifacts:
            intent_summary = self._load_intent_summary_artifact()
        else:
            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(snapshot_path=self.snapshot_path)
        position_decision = None
        position_execution = None
        if trader_positions:
            decision = self._brain().run(snapshot)
            managed_snapshot = _trader_only_snapshot(snapshot)
            position_decision = self._position_manager().run(managed_snapshot)
            position_execution = self._position_gateway().run(
                decision=position_decision,
                snapshot=managed_snapshot,
                send=send,
            )
            canceled_orders: list[str] = []
            canceled_status = "CANCELED_CONTAMINATED_PENDING"
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
            target_open = (
                target_summary is not None
                and target_summary.status == "PURSUE_TARGET"
                and target_summary.remaining_target_jpy > 0
            )
            target_reached = target_summary is not None and target_summary.status == "TARGET_REACHED_PROTECT"
            portfolio_add_allowed = _portfolio_add_allowed(snapshot)
            if (
                pending_entries
                and target_reached
                and send
                and self.live_enabled
                and not position_execution.sent
                and position_execution.status == "NO_ACTION"
                and not canceled_orders
            ):
                for order_id in _trader_pending_entry_order_ids(snapshot):
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
                if canceled_orders:
                    canceled_status = "CANCELED_TARGET_REACHED_PENDING"
            if (
                position_execution.sent
                or position_execution.status in {"STAGED", "BLOCKED"}
                or not portfolio_add_allowed
                or canceled_orders
                or (pending_entries and not target_open)
            ):
                status = "MONITOR_ONLY_EXPOSURE_OPEN"
                if canceled_orders:
                    status = canceled_status
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
            managed_snapshot = _trader_only_snapshot(snapshot)
            position_decision = self._position_manager().run(managed_snapshot)
            position_execution = self._position_gateway().run(
                decision=position_decision,
                snapshot=managed_snapshot,
                send=send and trader_positions > 0,
            )
            canceled_orders: list[str] = []
            status = "MONITOR_ONLY_EXPOSURE_OPEN"
            if position_execution.sent:
                status = "POSITION_ACTION_SENT"
            elif position_execution.status == "STAGED":
                status = "POSITION_ACTION_STAGED"
            elif position_execution.status == "BLOCKED":
                status = "POSITION_ACTION_BLOCKED"
            if send and self.live_enabled and trader_positions == 0 and decision.pending_cancel_order_ids:
                for order_id in decision.pending_cancel_order_ids:
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
                status = "CANCELED_CONTAMINATED_PENDING"
            target_reached = target_summary is not None and target_summary.status == "TARGET_REACHED_PROTECT"
            if not canceled_orders and target_reached and send and self.live_enabled:
                for order_id in _trader_pending_entry_order_ids(snapshot):
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
                if canceled_orders:
                    status = "CANCELED_TARGET_REACHED_PENDING"
            target_open = (
                target_summary is not None
                and target_summary.status == "PURSUE_TARGET"
                and target_summary.remaining_target_jpy > 0
            )
            if not canceled_orders and target_open:
                basket_lane_ids, basket_size_multiples = self._basket_lane_plan(
                    decision=decision,
                    primary_lane_id=None,
                    primary_size_multiple=None,
                    allow_existing_pending=True,
                )
                if basket_lane_ids:
                    if send and self.live_enabled and not self.use_gpt_trader:
                        return self._fresh_entry_gpt_required_summary(
                            generated_at=generated_at,
                            positions=positions,
                            orders=orders,
                            live_ready=intent_summary.live_ready,
                            selected_lane_id=basket_lane_ids[0],
                            selected_lane_ids=basket_lane_ids,
                            deterministic_lane_id=basket_lane_ids[0],
                            canceled_orders=tuple(canceled_orders),
                            target_summary=target_summary,
                            position_decision=position_decision,
                            position_execution=position_execution,
                            decision_source="deterministic_basket_blocked",
                        )
                    gpt_summary = None
                    if send and self.live_enabled and self.use_gpt_trader:
                        gpt_summary = self._run_gpt_handoff()
                        gpt_lane_ids = (
                            gpt_summary.selected_lane_ids
                            or ((gpt_summary.selected_lane_id,) if gpt_summary.selected_lane_id else ())
                        )
                        if (
                            gpt_summary.status == "ACCEPTED"
                            and gpt_summary.allowed
                            and gpt_summary.action == "CANCEL_PENDING"
                        ):
                            canceled_orders.extend(
                                self._cancel_gpt_pending_orders(
                                    gpt_summary,
                                    send=send,
                                    already_canceled=tuple(canceled_orders),
                                )
                            )
                            summary = AutoTradeCycleSummary(
                                status="CANCELED_GPT_PENDING" if canceled_orders else "GPT_CANCEL_PENDING",
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                deterministic_lane_id=basket_lane_ids[0],
                                sent=False,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                selected_lane_ids=basket_lane_ids,
                                canceled_orders=tuple(canceled_orders),
                                receipt_promotions=0,
                                decision_source="gpt_trader",
                                position_management_action=position_decision.action,
                                position_execution_status=position_execution.status,
                                position_execution_sent=position_execution.sent,
                                target_status=target_summary.status if target_summary else None,
                                target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                                target_progress_pct=target_summary.progress_pct if target_summary else None,
                                gpt_status=gpt_summary.status,
                                gpt_action=gpt_summary.action,
                                gpt_allowed=gpt_summary.allowed,
                                gpt_issues=gpt_summary.issues,
                                gpt_error=gpt_summary.error,
                            )
                            self._write_report(summary, generated_at)
                            return summary
                        if (
                            gpt_summary.status != "ACCEPTED"
                            or not gpt_summary.allowed
                            or gpt_summary.action != "TRADE"
                            or not gpt_lane_ids
                        ):
                            summary = AutoTradeCycleSummary(
                                status=(
                                    "GPT_REJECTED"
                                    if gpt_summary.status != "ACCEPTED" or not gpt_summary.allowed
                                    else f"GPT_{gpt_summary.action or 'NO_TRADE'}"
                                ),
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                deterministic_lane_id=basket_lane_ids[0],
                                sent=False,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                selected_lane_ids=basket_lane_ids,
                                canceled_orders=tuple(canceled_orders),
                                receipt_promotions=0,
                                decision_source="gpt_trader",
                                position_management_action=position_decision.action,
                                position_execution_status=position_execution.status,
                                position_execution_sent=position_execution.sent,
                                target_status=target_summary.status if target_summary else None,
                                target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                                target_progress_pct=target_summary.progress_pct if target_summary else None,
                                gpt_status=gpt_summary.status,
                                gpt_action=gpt_summary.action,
                                gpt_allowed=gpt_summary.allowed,
                                gpt_issues=gpt_summary.issues,
                                gpt_error=gpt_summary.error,
                            )
                            self._write_report(summary, generated_at)
                            return summary
                        if not _basket_parent_lane_set(gpt_lane_ids).issubset(_basket_parent_lane_set(basket_lane_ids)):
                            summary = AutoTradeCycleSummary(
                                status="GPT_DECISION_NOT_PREFILTERED",
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                deterministic_lane_id=basket_lane_ids[0],
                                sent=False,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                selected_lane_ids=gpt_lane_ids,
                                canceled_orders=tuple(canceled_orders),
                                receipt_promotions=0,
                                decision_source="gpt_trader",
                                position_management_action=position_decision.action,
                                position_execution_status=position_execution.status,
                                position_execution_sent=position_execution.sent,
                                target_status=target_summary.status if target_summary else None,
                                target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                                target_progress_pct=target_summary.progress_pct if target_summary else None,
                                gpt_status=gpt_summary.status,
                                gpt_action=gpt_summary.action,
                                gpt_allowed=gpt_summary.allowed,
                                gpt_issues=gpt_summary.issues,
                                gpt_error=gpt_summary.error,
                            )
                            self._write_report(summary, generated_at)
                            return summary
                        canceled_orders.extend(
                            self._cancel_gpt_pending_orders(
                                gpt_summary,
                                send=send,
                                already_canceled=tuple(canceled_orders),
                            )
                        )
                        if target_open:
                            basket_lane_ids, basket_size_multiples = self._expanded_gpt_basket_plan(
                                decision=decision,
                                gpt_lane_ids=gpt_lane_ids,
                                allow_existing_pending=True,
                            )
                        else:
                            basket_lane_ids = gpt_lane_ids
                            basket_size_multiples = {
                                lane_id: basket_size_multiples.get(lane_id, 1.0)
                                for lane_id in basket_lane_ids
                            }
                    order_summary = LiveOrderGateway(
                        client=self.client,
                        strategy_profile=self.strategy_profile_path,
                        output_path=self.live_order_output_path,
                        report_path=self.live_order_report_path,
                        live_enabled=self.live_enabled,
                        max_loss_jpy=resolved_max_loss_jpy,
                        portfolio_loss_cap_jpy=self._portfolio_loss_cap_jpy_from_target_state(),
                    ).run_batch(
                        intents_path=self.intents_path,
                        lane_ids=basket_lane_ids,
                        size_multiples=basket_size_multiples,
                        send=send,
                        confirm_live=send,
                    )
                    selected_lane_id = order_summary.lane_id
                    selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                        decision=decision,
                        lane_id=selected_lane_id,
                    )
                    summary = AutoTradeCycleSummary(
                        status=order_summary.status,
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=selected_lane_id,
                        selected_lane_ids=order_summary.lane_ids,
                        selected_lane_score=selected_lane_score,
                        selected_lane_size_multiple=selected_lane_size_multiple,
                        deterministic_lane_id=selected_lane_id,
                        sent=order_summary.sent,
                        sent_count=order_summary.sent_count,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        canceled_orders=tuple(canceled_orders),
                        receipt_promotions=0,
                        decision_source="deterministic_basket",
                        position_management_action=position_decision.action,
                        position_execution_status=position_execution.status,
                        position_execution_sent=position_execution.sent,
                        target_status=target_summary.status if target_summary else None,
                        target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
                        target_progress_pct=target_summary.progress_pct if target_summary else None,
                        gpt_status=gpt_summary.status if gpt_summary else None,
                        gpt_action=gpt_summary.action if gpt_summary else None,
                        gpt_allowed=gpt_summary.allowed if gpt_summary else None,
                        gpt_issues=gpt_summary.issues if gpt_summary else None,
                        gpt_error=gpt_summary.error if gpt_summary else None,
                    )
                    self._write_report(summary, generated_at)
                    return summary
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

        if self.reuse_market_artifacts:
            promotion_summary = self._skipped_receipt_promotion_summary()
        else:
            promotion_summary = self._receipt_promoter().run()
        if promotion_summary.promoted and not self.reuse_market_artifacts:
            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(snapshot_path=self.snapshot_path)
        decision = self._brain().run(snapshot)
        deterministic_lane_id = decision.selected_lane_id if decision.action == ACTION_SEND_ENTRY else None
        selected_lane_id = deterministic_lane_id
        selected_lane_score = decision.selected_lane_score
        selected_lane_size_multiple = decision.selected_lane_size_multiple
        gpt_summary = None
        gpt_selected_lane_ids: tuple[str, ...] = ()
        gpt_wait_retries = 0
        gpt_recovery_source = None
        campaign_exposure_required = _campaign_exposure_required(
            target_summary=target_summary,
            trader_positions=trader_positions,
            pending_entries=pending_entries,
            live_ready=intent_summary.live_ready,
        )

        if selected_lane_id is None and campaign_exposure_required and not self.use_gpt_trader:
            recovery_lane_id = self._campaign_recovery_lane(decision=decision, deterministic_lane_id=None)
            if recovery_lane_id:
                selected_lane_id = recovery_lane_id
                selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                    decision=decision,
                    lane_id=selected_lane_id,
                )
                gpt_recovery_source = "DETERMINISTIC_CAMPAIGN_EXPOSURE_RECOVERY"

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
                    campaign_exposure_required=campaign_exposure_required,
                )
                self._write_report(summary, generated_at)
                return summary

            gpt_summary = self._run_gpt_handoff()
            if gpt_summary.status == "ACCEPTED" and gpt_summary.allowed:
                if gpt_summary.action == "TRADE" and gpt_summary.selected_lane_id:
                    gpt_selected_lane_ids = (
                        gpt_summary.selected_lane_ids
                        or (gpt_summary.selected_lane_id,)
                    )
                    selected_lane_id = gpt_selected_lane_ids[0]
                    selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                        decision=decision,
                        lane_id=selected_lane_id,
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
                            snapshot = self._refresh_snapshot(pairs)
                            target_summary = self._update_target_state(snapshot) or target_summary
                            positions = len(snapshot.positions)
                            orders = len(snapshot.orders)
                            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(
                                snapshot_path=self.snapshot_path,
                                max_candidates=12,
                            )
                            decision = self._brain().run(snapshot)
                            deterministic_lane_id = (
                                decision.selected_lane_id if decision.action == ACTION_SEND_ENTRY else None
                            )
                            if deterministic_lane_id:
                                selected_lane_id = deterministic_lane_id
                                selected_lane_score = decision.selected_lane_score
                                selected_lane_size_multiple = decision.selected_lane_size_multiple
                                gpt_summary = GptHandoffSummary(
                                    status="ACCEPTED",
                                    action="TRADE",
                                    selected_lane_id=deterministic_lane_id,
                                    allowed=True,
                                    issues=0,
                                    selected_lane_ids=(deterministic_lane_id,),
                                )
                                gpt_selected_lane_ids = (deterministic_lane_id,)
                                gpt_recovery_source = f"DETERMINISTIC_WAIT_RECOVERY_ATTEMPT_{attempt}"
                                break
                            if attempt == self.gpt_wait_retry_limit:
                                break
                            retry_summary = self._run_gpt_handoff()
                            gpt_summary = retry_summary
                            if (
                                retry_summary.status == "ACCEPTED"
                                and retry_summary.allowed
                                and retry_summary.action == "TRADE"
                                and retry_summary.selected_lane_id
                            ):
                                gpt_selected_lane_ids = (
                                    retry_summary.selected_lane_ids
                                    or (retry_summary.selected_lane_id,)
                                )
                                selected_lane_id = gpt_selected_lane_ids[0]
                                selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                                    decision=decision,
                                    lane_id=selected_lane_id,
                                )
                                gpt_recovery_source = f"GPT_RETRY_TRADE_ATTEMPT_{attempt}"
                                break

            if selected_lane_id is None and campaign_exposure_required:
                recovery_lane_id = self._campaign_recovery_lane(
                    decision=decision,
                    deterministic_lane_id=deterministic_lane_id,
                )
                if recovery_lane_id:
                    selected_lane_id = recovery_lane_id
                    selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                        decision=decision,
                        lane_id=selected_lane_id,
                    )
                    gpt_recovery_source = (
                        gpt_recovery_source
                        or f"CAMPAIGN_EXPOSURE_RECOVERY_GPT_{gpt_summary.status}_{gpt_summary.action or 'NO_TRADE'}"
                    )

            if selected_lane_id is None:
                status = (
                    "GPT_REJECTED"
                    if gpt_summary.status != "ACCEPTED" or not gpt_summary.allowed
                    else f"GPT_{gpt_summary.action or 'NO_TRADE'}"
                )
                summary = AutoTradeCycleSummary(
                    status=status,
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
                    campaign_exposure_required=campaign_exposure_required,
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
                gpt_trade_accepted = (
                    gpt_summary.status == "ACCEPTED"
                    and gpt_summary.allowed
                    and gpt_summary.action == "TRADE"
                    and bool(gpt_summary.selected_lane_id)
                )
                if gpt_trade_accepted:
                    gpt_selected_lane_ids = (
                        gpt_summary.selected_lane_ids
                        or (gpt_summary.selected_lane_id,)
                    )
                if not gpt_trade_accepted:
                    if campaign_exposure_required:
                        reason = gpt_summary.action or gpt_summary.status or "NO_TRADE"
                        gpt_recovery_source = f"CAMPAIGN_EXPOSURE_RECOVERY_GPT_{reason}"
                    else:
                        status = (
                            "GPT_REJECTED"
                            if gpt_summary.status != "ACCEPTED" or not gpt_summary.allowed
                            else f"GPT_{gpt_summary.action or 'NO_TRADE'}"
                        )
                        summary = AutoTradeCycleSummary(
                            status=status,
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
                            campaign_exposure_required=campaign_exposure_required,
                        )
                        self._write_report(summary, generated_at)
                        return summary
                elif not _basket_parent_lane_set(gpt_selected_lane_ids).issubset(
                    _basket_parent_lane_set(tuple(prefiltered_lane_ids))
                ):
                    if campaign_exposure_required:
                        gpt_recovery_source = "CAMPAIGN_EXPOSURE_RECOVERY_GPT_NOT_PREFILTERED"
                    else:
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
                            campaign_exposure_required=campaign_exposure_required,
                        )
                        self._write_report(summary, generated_at)
                        return summary
                else:
                    selected_lane_id = gpt_selected_lane_ids[0]
                    selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                        decision=decision, lane_id=selected_lane_id
                    )

        target_open = (
            target_summary is not None
            and target_summary.status == "PURSUE_TARGET"
            and target_summary.remaining_target_jpy > 0
        )
        if self.use_gpt_trader and target_open and gpt_selected_lane_ids:
            basket_lane_ids, basket_size_multiples = self._expanded_gpt_basket_plan(
                decision=decision,
                gpt_lane_ids=gpt_selected_lane_ids,
            )
            selected_lane_id = basket_lane_ids[0] if basket_lane_ids else selected_lane_id
            selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                decision=decision,
                lane_id=selected_lane_id,
            )
        else:
            basket_lane_ids = gpt_selected_lane_ids or ((selected_lane_id,) if selected_lane_id else ())
            basket_size_multiples = {
                selected_lane_id: selected_lane_size_multiple if selected_lane_size_multiple is not None else 1.0
            } if selected_lane_id else {}
            for lane_id in basket_lane_ids:
                if lane_id not in basket_size_multiples:
                    _, size_multiple = self._selected_lane_meta(decision=decision, lane_id=lane_id)
                    basket_size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
        if selected_lane_id and not self.use_gpt_trader:
            basket_lane_ids, basket_size_multiples = self._basket_lane_plan(
                decision=decision,
                primary_lane_id=selected_lane_id,
                primary_size_multiple=selected_lane_size_multiple,
            )

        if basket_lane_ids and send and self.live_enabled and not self.use_gpt_trader:
            return self._fresh_entry_gpt_required_summary(
                generated_at=generated_at,
                positions=positions,
                orders=orders,
                live_ready=intent_summary.live_ready,
                selected_lane_id=selected_lane_id,
                selected_lane_ids=basket_lane_ids,
                selected_lane_score=selected_lane_score,
                selected_lane_size_multiple=selected_lane_size_multiple,
                deterministic_lane_id=deterministic_lane_id,
                receipt_promotions=promotion_summary.promoted,
                target_summary=target_summary,
                position_decision=position_decision,
                position_execution=position_execution,
            )

        order_gateway = LiveOrderGateway(
            client=self.client,
            strategy_profile=self.strategy_profile_path,
            output_path=self.live_order_output_path,
            report_path=self.live_order_report_path,
            live_enabled=self.live_enabled,
            max_loss_jpy=resolved_max_loss_jpy,
            portfolio_loss_cap_jpy=self._portfolio_loss_cap_jpy_from_target_state(),
        )
        if len(basket_lane_ids) > 1:
            order_summary = order_gateway.run_batch(
                intents_path=self.intents_path,
                lane_ids=basket_lane_ids,
                size_multiples=basket_size_multiples,
                send=send,
                confirm_live=send,
            )
        else:
            order_summary = order_gateway.run(
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
            selected_lane_ids=order_summary.lane_ids or basket_lane_ids,
            selected_lane_score=selected_lane_score,
            selected_lane_size_multiple=selected_lane_size_multiple,
            deterministic_lane_id=deterministic_lane_id,
            sent=order_summary.sent,
            sent_count=order_summary.sent_count,
            positions=positions,
            orders=orders,
            live_ready=intent_summary.live_ready,
            decision_source=(
                "campaign_exposure_recovery"
                if gpt_recovery_source and "CAMPAIGN_EXPOSURE_RECOVERY" in gpt_recovery_source
                else ("gpt_trader" if self.use_gpt_trader else "deterministic")
            ),
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
            campaign_exposure_required=campaign_exposure_required,
        )
        self._write_report(summary, generated_at)
        return summary

    def _fresh_entry_gpt_required_summary(
        self,
        *,
        generated_at: str,
        positions: int,
        orders: int,
        live_ready: int,
        selected_lane_id: str | None,
        selected_lane_ids: tuple[str, ...],
        deterministic_lane_id: str | None,
        target_summary: DailyTargetSummary | None,
        selected_lane_score: float | None = None,
        selected_lane_size_multiple: float | None = None,
        canceled_orders: tuple[str, ...] = (),
        receipt_promotions: int = 0,
        position_decision=None,
        position_execution=None,
        decision_source: str = "deterministic_blocked",
    ) -> AutoTradeCycleSummary:
        summary = AutoTradeCycleSummary(
            status="GPT_REQUIRED_FOR_LIVE_SEND",
            report_path=self.report_path,
            snapshot_path=self.snapshot_path,
            intents_path=self.intents_path,
            selected_lane_id=selected_lane_id,
            selected_lane_ids=selected_lane_ids,
            selected_lane_score=selected_lane_score,
            selected_lane_size_multiple=selected_lane_size_multiple,
            deterministic_lane_id=deterministic_lane_id,
            sent=False,
            sent_count=0,
            positions=positions,
            orders=orders,
            live_ready=live_ready,
            canceled_orders=canceled_orders,
            receipt_promotions=receipt_promotions,
            decision_source=decision_source,
            position_management_action=position_decision.action if position_decision else None,
            position_execution_status=position_execution.status if position_execution else None,
            position_execution_sent=position_execution.sent if position_execution else False,
            target_status=target_summary.status if target_summary else None,
            target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
            target_progress_pct=target_summary.progress_pct if target_summary else None,
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
            f"- Selected basket lanes: `{', '.join(summary.selected_lane_ids) if summary.selected_lane_ids else 'none'}`",
            f"- Selected lane score: `{summary.selected_lane_score}`",
            f"- Selected lane size multiple: `{summary.selected_lane_size_multiple}`",
            f"- Sent: `{summary.sent}`",
            f"- Sent count: `{summary.sent_count}`",
            f"- Canceled orders: `{', '.join(summary.canceled_orders) if summary.canceled_orders else 'none'}`",
            f"- Position management: `{summary.position_management_action or 'none'}`",
            f"- Position execution: `{summary.position_execution_status or 'none'}` sent=`{summary.position_execution_sent}`",
            f"- Daily target: `{summary.target_status or 'not configured'}` remaining=`{summary.target_remaining_jpy}` progress_pct=`{summary.target_progress_pct}`",
            f"- GPT trader: status=`{summary.gpt_status or 'not used'}` action=`{summary.gpt_action}` allowed=`{summary.gpt_allowed}` issues=`{summary.gpt_issues}`",
            f"- GPT error: `{summary.gpt_error or 'none'}`",
            f"- GPT wait recovery attempts: `{summary.gpt_wait_retries}`",
            f"- GPT recovery source: `{summary.gpt_recovery_source or 'none'}`",
            f"- Campaign exposure required: `{summary.campaign_exposure_required}`",
            f"- Market artifact mode: `{'reuse_existing' if self.reuse_market_artifacts else 'refresh_and_reprice'}`",
            f"- Market story refresh: `{self.refresh_market_story}` (source: `{self.market_news_root}`)",
        ]
        if summary.status == "GPT_REQUIRED_FOR_LIVE_SEND":
            lines.append(
                "- Live entry blocker: `--send` requires `--use-gpt-trader --gpt-decision-response ...`; "
                "deterministic TraderBrain may prefilter but cannot be the live discretionary sender."
            )
        lines.extend(
            [
                "",
                "## Cycle Contract",
                "",
                "- Protected trader-owned positions and trader-owned pending entries may add only through basket portfolio risk validation.",
                "- If basket portfolio validation has no capacity, pending entries remain monitor-only.",
                "- Open positions are handed to PositionManager first, then the protection gateway may close, repair protection, or tighten SL when the action is risk-reducing.",
                "- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.",
                "- A verified GPT `CANCEL_PENDING` cancels only current trader-owned pending entry ids and sends no fresh entry in that cycle.",
                "- If flat, risk-repair or trigger receipts may promote the strategy profile before TraderBrain compares lanes.",
                "- If the daily target is open, the trader is flat, and LIVE_READY lanes survive prefiltering, the cycle must recover to a lane instead of preserving discretionary flatness.",
                "- If the daily target is already reached while flat, the cycle records protection-first no-send status and adds no fresh risk; trader-owned pending entries are canceled instead of left fillable.",
                "- If GPT trader handoff is enabled, the selected lane must also be an accepted GPT `TRADE` decision from the deterministic prefilter set.",
                "- If flat, the cycle refreshes broker truth immediately before pricing intents unless `--reuse-market-artifacts` pins the already generated decision packet; the live gateway still refreshes broker truth before any stage/send.",
            ]
        )
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

    def _refresh_snapshot(self, pairs: tuple[str, ...]):
        snapshot = self.client.snapshot(pairs)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
        return snapshot

    def _load_snapshot_artifact(self):
        if not self.snapshot_path.exists():
            raise ValueError(f"reuse-market-artifacts requires existing broker snapshot: {self.snapshot_path}")
        return _snapshot_from_json(json.loads(self.snapshot_path.read_text()))

    def _load_intent_summary_artifact(self) -> IntentGenerationSummary:
        if not self.intents_path.exists():
            raise ValueError(f"reuse-market-artifacts requires existing order intents: {self.intents_path}")
        payload = json.loads(self.intents_path.read_text())
        results = [item for item in payload.get("results", []) or [] if isinstance(item, dict)]
        return IntentGenerationSummary(
            output_path=self.intents_path,
            report_path=self.intent_report_path,
            candidates_seen=len(results),
            generated=sum(1 for item in results if isinstance(item.get("intent"), dict)),
            needs_snapshot=sum(1 for item in results if item.get("status") == "NEEDS_BROKER_SNAPSHOT"),
            dry_run_passed=sum(1 for item in results if item.get("status") == "DRY_RUN_PASSED"),
            live_ready=sum(1 for item in results if item.get("status") == "LIVE_READY"),
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
            pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
        ).run(snapshot=snapshot)

    def _receipt_promoter(self) -> ReceiptPromoter:
        return ReceiptPromoter(
            profile_path=self.strategy_profile_path,
            intents_path=self.intents_path,
            report_path=self.receipt_promotion_report_path,
        )

    def _skipped_receipt_promotion_summary(self) -> ReceiptPromotionSummary:
        return ReceiptPromotionSummary(
            profile_path=self.strategy_profile_path,
            intents_path=self.intents_path,
            report_path=self.receipt_promotion_report_path,
            profiles_seen=0,
            receipts_seen=0,
            promoted=0,
            still_blocked=0,
        )

    def _brain(self) -> TraderBrain:
        return TraderBrain(
            intents_path=self.intents_path,
            campaign_plan_path=self.campaign_plan_path,
            strategy_profile_path=self.strategy_profile_path,
            market_story_profile_path=self.market_story_profile_path,
            trader_settings_path=self.trader_settings_path,
            target_state_path=self.target_state_path or DEFAULT_DAILY_TARGET_STATE,
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

    @staticmethod
    def _campaign_recovery_lane(*, decision: TraderDecision, deterministic_lane_id: str | None) -> str | None:
        prefiltered = [item for item in decision.scores if _passes_gpt_prefilter(item)]
        if deterministic_lane_id and any(item.lane_id == deterministic_lane_id for item in prefiltered):
            return deterministic_lane_id
        return prefiltered[0].lane_id if prefiltered else None

    @staticmethod
    def _basket_lane_plan(
        *,
        decision: TraderDecision,
        primary_lane_id: str | None,
        primary_size_multiple: float | None,
        allow_existing_pending: bool = False,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        lane_ids: list[str] = []
        size_multiples: dict[str, float] = {}
        parent_lane_ids: set[str] = set()

        def add(lane_id: str | None, size_multiple: float | None) -> None:
            if not lane_id or lane_id in size_multiples:
                return
            parent_lane_id = _basket_parent_lane_id(lane_id)
            if parent_lane_id and parent_lane_id in parent_lane_ids:
                return
            lane_ids.append(lane_id)
            size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
            if parent_lane_id:
                parent_lane_ids.add(parent_lane_id)

        add(primary_lane_id, primary_size_multiple)
        for score in decision.scores:
            if _passes_basket_prefilter(score, allow_existing_pending=allow_existing_pending):
                add(score.lane_id, score.size_multiple)
        return tuple(lane_ids), size_multiples

    @staticmethod
    def _expanded_gpt_basket_plan(
        *,
        decision: TraderDecision,
        gpt_lane_ids: tuple[str, ...],
        allow_existing_pending: bool = False,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        lane_ids: list[str] = []
        size_multiples: dict[str, float] = {}
        parent_lane_ids: set[str] = set()

        def add(lane_id: str | None, size_multiple: float | None = None) -> None:
            if not lane_id or lane_id in size_multiples:
                return
            parent_lane_id = _basket_parent_lane_id(lane_id)
            if parent_lane_id and parent_lane_id in parent_lane_ids:
                return
            if size_multiple is None:
                _, size_multiple = AutoTradeCycle._selected_lane_meta(
                    decision=decision,
                    lane_id=lane_id,
                )
            lane_ids.append(lane_id)
            size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
            if parent_lane_id:
                parent_lane_ids.add(parent_lane_id)

        for lane_id in gpt_lane_ids:
            add(lane_id)
        for score in decision.scores:
            if _passes_basket_prefilter(score, allow_existing_pending=allow_existing_pending):
                add(score.lane_id, score.size_multiple)
        return tuple(lane_ids), size_multiples

    def _run_gpt_handoff(self) -> GptHandoffSummary:
        try:
            summary = self._gpt_brain().run(snapshot_path=self.snapshot_path)
            return GptHandoffSummary(
                status=summary.status,
                action=summary.action,
                selected_lane_id=summary.selected_lane_id,
                allowed=summary.allowed,
                issues=summary.issues,
                selected_lane_ids=summary.selected_lane_ids,
                cancel_order_ids=summary.cancel_order_ids,
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

    def _cancel_gpt_pending_orders(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        send: bool,
        already_canceled: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not send or not self.live_enabled or not gpt_summary.cancel_order_ids:
            return ()
        canceled: list[str] = []
        already = set(already_canceled)
        for order_id in gpt_summary.cancel_order_ids:
            if order_id in already:
                continue
            self.client.cancel_order(order_id)
            canceled.append(order_id)
            already.add(order_id)
        return tuple(canceled)

    def _gpt_brain(self) -> GPTTraderBrain:
        return GPTTraderBrain(
            provider=self.gpt_provider,
            intents_path=self.intents_path,
            campaign_plan_path=self.campaign_plan_path,
            strategy_profile_path=self.strategy_profile_path,
            market_story_profile_path=self.market_story_profile_path,
            target_state_path=self.gpt_target_state_path,
            attack_advice_path=self.gpt_attack_advice_path,
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

    def _portfolio_loss_cap_jpy_from_target_state(self) -> float | None:
        """Return the whole-day cap used for open + pending + basket risk."""
        if self.target_state_path is None or not self.target_state_path.exists():
            return None
        try:
            payload = json.loads(self.target_state_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        raw = payload.get("daily_risk_budget_jpy")
        try:
            value = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

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
    return len(_trader_pending_entry_order_ids(snapshot))


def _trader_pending_entry_order_ids(snapshot) -> tuple[str, ...]:
    return tuple(
        str(order.order_id)
        for order in snapshot.orders
        if not order.trade_id
        and str(order.order_type or "").upper() in PENDING_ENTRY_TYPES
        and order.owner.value not in {"manual", "unknown"}
        and order.order_id
    )


def _campaign_exposure_required(
    *,
    target_summary: DailyTargetSummary | None,
    trader_positions: int,
    pending_entries: int,
    live_ready: int,
) -> bool:
    if target_summary is None:
        return False
    return (
        target_summary.status == "PURSUE_TARGET"
        and target_summary.remaining_target_jpy > 0
        and trader_positions == 0
        and pending_entries == 0
        and live_ready > 0
    )


def _portfolio_add_allowed(snapshot) -> bool:
    trader_positions = tuple(position for position in snapshot.positions if position.owner.value == "trader")
    if not trader_positions:
        return False
    return all(
        position.owner.value == "trader" and position.stop_loss is not None and position.take_profit is not None
        for position in trader_positions
    )


def _trader_position_count(snapshot) -> int:
    return sum(1 for position in snapshot.positions if position.owner.value == "trader")


def _trader_only_snapshot(snapshot):
    return replace(
        snapshot,
        positions=tuple(position for position in snapshot.positions if position.owner.value == "trader"),
    )
