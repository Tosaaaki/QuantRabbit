from __future__ import annotations

import json
import math
import os
import sys
import sqlite3
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.broker.execution import ACTIVE_FX_SESSION_BUCKETS_PER_DAY, LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.broker.position_execution import PositionExecutionSummary, PositionProtectionGateway
from quant_rabbit.analysis.market_status import (
    compute_market_status,
    write_report as write_market_status_report,
    write_snapshot as write_market_status_snapshot,
)
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_LEARNING_AUDIT_REPORT,
    DEFAULT_MARKET_STATUS,
    DEFAULT_MARKET_STATUS_REPORT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_HISTORY_DB,
    DEFAULT_OUTCOME_MART,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POSITION_EXECUTION_REPORT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_POST_TRADE_LEARNING,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_DECISION,
    DEFAULT_TRADER_DECISION_REPORT,
    DEFAULT_TRADER_JOURNAL,
    DEFAULT_TRADER_SETTINGS,
    DEFAULT_VERIFICATION_LEDGER,
    DEFAULT_VERIFICATION_LEDGER_REPORT,
    ROOT,
)
from quant_rabbit.ai_test_bot import AITestBotBacktester
from quant_rabbit.gpt_trader import DEFAULT_GPT_MAX_LANES, GPTTraderBrain, TraderModelProvider
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.learning_audit import LearningAuditor
from quant_rabbit.risk import RiskPolicy, margin_budget_jpy, resolve_max_loss_jpy
from quant_rabbit.snapshot_json import snapshot_order_raw
from quant_rabbit.target import DailyTargetLedger, DailyTargetSummary
from quant_rabbit.strategy.ensemble import CampaignPlanner
from quant_rabbit.strategy.intent_generator import IntentGenerationSummary, IntentGenerator, _snapshot_from_json
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.position_manager import ACTION_REVIEW_EXIT, ManagedPosition, PositionManagementDecision, PositionManager
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter, ReceiptPromotionSummary
from quant_rabbit.strategy.trader_brain import (
    ACTION_NO_TRADE,
    ACTION_SEND_ENTRY,
    LaneScore,
    TraderBrain,
    TraderDecision,
    load_trader_settings,
)
from quant_rabbit.verification_ledger import VerificationLedger


DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"
DEFAULT_AUTOTRADE_LOCK_DIR = ROOT / ".quant_rabbit_live.lock"
PENDING_ENTRY_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
ACCEPTED_GPT_GATEWAY_ACTIONS = frozenset({"TRADE", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE"})
GPT_LIVE_ORDER_ACTIONS = frozenset({"TRADE", "CANCEL_PENDING"})
GPT_POSITION_GATEWAY_ACTIONS = frozenset({"PROTECT", "TIGHTEN_SL", "CLOSE"})

# C-4 margin-aware basket truncation (2026-05-12, repaired 2026-05-15).
# The basket builder stops adding fresh-entry lanes once cumulative
# `LaneScore.estimated_margin_jpy` would exceed the same effective margin
# room used by `RiskEngine` (`min(marginAvailable, NAV * cap - marginUsed)`)
# multiplied by this safety buffer. The buffer is an engineering tolerance
# — not a market-derived value — to leave room for intra-cycle quote drift,
# spread widening, and slippage between the trader_brain margin estimate and
# the LiveOrderGateway's final pre-send revalidation. Using raw
# marginAvailable here was a bug: the discretionary receipt could claim a
# basket fit while the gateway correctly rejected it with
# `BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED`.
MARGIN_AWARE_BASKET_BUFFER = 0.9


def _basket_margin_room_jpy(snapshot: object) -> float | None:
    account = getattr(snapshot, "account", None)
    if account is None:
        return None
    max_margin_pct = RiskPolicy().max_margin_utilization_pct
    if max_margin_pct is None:
        return max(0.0, float(account.margin_available_jpy))
    return max(0.0, margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct))


def _buffered_basket_margin_budget_jpy(
    *,
    margin_room_jpy: float | None,
    margin_available_jpy: float | None,
) -> float | None:
    # `margin_available_jpy` is retained for tests and older callers. Live
    # automation passes `margin_room_jpy`, which is the same effective room
    # used by RiskEngine / LiveOrderGateway.
    base = margin_room_jpy if margin_room_jpy is not None else margin_available_jpy
    if base is None:
        return None
    return max(0.0, float(base)) * MARGIN_AWARE_BASKET_BUFFER


def _position_execution_cycle_status(
    execution: PositionExecutionSummary,
    *,
    fallback: str,
) -> str:
    if execution.sent:
        return "POSITION_ACTION_SENT"
    if execution.status == "STAGED":
        return "POSITION_ACTION_STAGED"
    if execution.status == "BLOCKED":
        return "POSITION_ACTION_BLOCKED"
    if execution.status == "STALE_CLOSE_SATISFIED":
        return "POSITION_ACTION_SATISFIED"
    return fallback


def _snapshot_refresh_pairs(snapshot: object) -> tuple[str, ...]:
    pairs = set(DEFAULT_TRADER_PAIRS)
    pairs.update(str(pair) for pair in getattr(snapshot, "quotes", {}) or {} if pair)
    for position in getattr(snapshot, "positions", ()) or ():
        pair = str(getattr(position, "pair", "") or "")
        if pair:
            pairs.add(pair)
    return tuple(sorted(pairs))


def _projection_atr_pips_by_pair(pair_charts_path: Path) -> dict[str, float]:
    if not pair_charts_path.exists():
        return {}
    try:
        payload = json.loads(pair_charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, float] = {}
    for chart in payload.get("charts", []) or []:
        if not isinstance(chart, dict):
            continue
        pair = str(chart.get("pair") or "")
        if not pair:
            continue
        for view in chart.get("views", []) or []:
            if not isinstance(view, dict) or view.get("granularity") != "H1":
                continue
            try:
                out[pair] = float((view.get("indicators") or {}).get("atr_pips"))
            except (TypeError, ValueError):
                pass
            break
    return out


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Per AGENT_CONTRACT §6 / §3.5: structural / contract-named blockers are the
# only hard reasons to keep a LIVE_READY lane out of the GPT prefilter set.
# Anything else (missing mined history, narrative caution, capture-rate
# caution, campaign-plan drift) is a discretionary penalty that already sizes
# the lane down through score → size_multiple, and must not be re-stacked as a
# prose veto after IntentGenerator has emitted a current LIVE_READY receipt.
# These patterns are matched as substrings (case-sensitive) against
# LaneScore.blockers — they come from `_score_lane`,
# `_discretionary_gate_check`, and `_exposure_blockers` in
# `quant_rabbit.strategy.trader_brain`.
_PREFILTER_HARD_BLOCKER_PATTERNS = (
    # §7 lane completeness — every executable lane must include thesis,
    # context, geometry, and units; missing any is a hard veto.
    "missing trader thesis",
    "missing market context",
    "incomplete market context",
    "missing TP/SL geometry",
    "missing executable units",
    # §11 strategy receipts — explicit non-live eligibility remains hard.
    # A fully LIVE_READY receipt with no mined profile is advisory only under
    # SL-free live mode; otherwise the first valid forecast-first lane for a
    # new pair can never reach the gateway and refresh its own evidence.
    "strategy profile is not live-eligible",
    # §9 lane status — anything not LIVE_READY shouldn't be in the set anyway,
    # but keep an explicit guard.
    "intent status is",
    "receipt is not live-ready",
    # §9 exposure blockers — open or pending exposure must be reconciled
    # before a fresh entry, regardless of GPT discretion.
    "open position exists",
    "pending entry exists",
)


_PREFILTER_HARD_FORECAST_PATTERNS = (
    "forecast up opposes",
    "forecast down opposes",
    "forecast range requires executable",
)

_LOW_CONFIDENCE_FORECAST_RANGE_ORDER_TYPES = {"LIMIT", "LIMIT_ORDER"}


def _is_low_confidence_range_rotation_legacy_blocker(score: LaneScore | None) -> bool:
    if score is None:
        return False
    if score.method != "RANGE_ROTATION":
        return False
    return str(score.order_type or "").upper() in _LOW_CONFIDENCE_FORECAST_RANGE_ORDER_TYPES


def _is_hard_forecast_prefilter_blocker(text: str, *, score: LaneScore | None = None) -> bool:
    # Directional contradiction remains a hard Stage-1 veto. Low-confidence
    # forecast text is softened only for legacy RANGE_ROTATION LIMIT receipts:
    # new TraderBrain code no longer emits that blocker for executable rail
    # geometry, but stale artifacts may still carry it until the next refresh.
    if "forecast confidence" in text:
        return not _is_low_confidence_range_rotation_legacy_blocker(score)
    if any(pattern in text for pattern in _PREFILTER_HARD_FORECAST_PATTERNS):
        return True
    return text.startswith("forecast ") and "has no executable edge" in text


def _is_hard_prefilter_blocker(blocker: str, *, score: LaneScore | None = None) -> bool:
    text = str(blocker).lower()
    if any(pattern in text for pattern in _PREFILTER_HARD_BLOCKER_PATTERNS):
        return True
    return _is_hard_forecast_prefilter_blocker(text, score=score)


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
    return not any(_is_hard_prefilter_blocker(b, score=score) for b in score.blockers)


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


def _recovery_hedge_parent_lane_set(intents_payload: dict) -> set[str]:
    parents: set[str] = set()
    for item in intents_payload.get("results", []) or []:
        if not isinstance(item, dict) or item.get("status") != "LIVE_READY":
            continue
        intent = item.get("intent")
        if not isinstance(intent, dict):
            continue
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        if metadata.get("position_intent") != "HEDGE" or metadata.get("hedge_recovery") is not True:
            continue
        parent = _basket_parent_lane_id(str(item.get("lane_id") or ""))
        if parent:
            parents.add(parent)
    return parents


def _gpt_lanes_pass_prefilter_or_recovery(
    *,
    intents_payload: dict,
    gpt_lane_ids: tuple[str, ...],
    prefiltered_lane_ids: set[str],
) -> tuple[bool, bool]:
    gpt_parent_lanes = _basket_parent_lane_set(gpt_lane_ids)
    if gpt_parent_lanes.issubset(_basket_parent_lane_set(tuple(prefiltered_lane_ids))):
        return True, False
    if gpt_parent_lanes and gpt_parent_lanes.issubset(_recovery_hedge_parent_lane_set(intents_payload)):
        return True, True
    return False, False


def _default_pair_charts_path(campaign_plan_path: Path) -> Path:
    sibling = campaign_plan_path.with_name("pair_charts.json")
    if campaign_plan_path != DEFAULT_CAMPAIGN_PLAN and sibling.exists():
        return sibling
    return DEFAULT_PAIR_CHARTS


def _gpt_sidecar_path(
    *,
    explicit: Path | None,
    gpt_decision_path: Path,
    default_path: Path,
) -> Path:
    if explicit is not None:
        return explicit
    if gpt_decision_path == DEFAULT_GPT_TRADER_DECISION:
        return default_path
    return gpt_decision_path.with_name(default_path.name)


def _attack_sidecar_path(
    *,
    explicit: Path | None,
    attack_advice_path: Path,
    default_path: Path,
) -> Path:
    if explicit is not None:
        return explicit
    if attack_advice_path == DEFAULT_AI_ATTACK_ADVICE:
        return default_path
    return attack_advice_path.with_name(default_path.name)


def _passes_basket_prefilter(score: LaneScore, *, allow_existing_pending: bool = False) -> bool:
    if _passes_gpt_prefilter(score):
        return True
    if not allow_existing_pending:
        return False
    if score.status != "LIVE_READY" or score.action != ACTION_NO_TRADE:
        return False
    blockers = [blocker for blocker in score.blockers if not _is_existing_pending_blocker(blocker)]
    return not any(_is_hard_prefilter_blocker(blocker, score=score) for blocker in blockers)


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


def _running_under_test_harness() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if "unittest" in sys.modules and any(
        name.startswith("unittest.") and name.endswith(("__main__", "main"))
        for name in sys.modules
    ):
        return True
    argv0 = Path(sys.argv[0]).name if sys.argv else ""
    return "unittest" in argv0 or "pytest" in argv0


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
    close_trade_ids: tuple[str, ...] = ()
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
        trader_journal_path: Path | None = None,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        execution_ledger_report_path: Path = DEFAULT_EXECUTION_LEDGER_REPORT,
        report_path: Path = DEFAULT_AUTOTRADE_REPORT,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        pair_charts_path: Path | None = None,
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
        gpt_learning_audit_path: Path | None = None,
        gpt_learning_audit_report_path: Path | None = None,
        gpt_learning_audit_db_path: Path | None = None,
        gpt_self_improvement_audit_path: Path | None = None,
        gpt_verification_ledger_path: Path | None = None,
        gpt_verification_ledger_report_path: Path | None = None,
        gpt_market_status_path: Path | None = None,
        gpt_market_status_report_path: Path | None = None,
        gpt_ai_backtest_path: Path | None = None,
        gpt_outcome_mart_path: Path | None = None,
        gpt_post_trade_learning_path: Path | None = None,
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
        injected_client = client is not None
        explicit_trader_journal_path = trader_journal_path is not None
        explicit_self_improvement_audit_path = gpt_self_improvement_audit_path is not None
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
        self.trader_journal_path = trader_journal_path or DEFAULT_TRADER_JOURNAL
        self._trader_journal_enabled = explicit_trader_journal_path or not injected_client
        self.execution_ledger_db_path = execution_ledger_db_path
        self.execution_ledger_report_path = execution_ledger_report_path
        self.report_path = report_path
        self.campaign_plan_path = campaign_plan_path
        self.pair_charts_path = pair_charts_path or _default_pair_charts_path(campaign_plan_path)
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
        self.gpt_learning_audit_path = _gpt_sidecar_path(
            explicit=gpt_learning_audit_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_LEARNING_AUDIT,
        )
        self.gpt_learning_audit_report_path = _gpt_sidecar_path(
            explicit=gpt_learning_audit_report_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_LEARNING_AUDIT_REPORT,
        )
        self.gpt_learning_audit_db_path = _gpt_sidecar_path(
            explicit=gpt_learning_audit_db_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_EXECUTION_LEDGER_DB,
        )
        self.gpt_self_improvement_audit_path = _gpt_sidecar_path(
            explicit=gpt_self_improvement_audit_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_SELF_IMPROVEMENT_AUDIT,
        )
        self.gateway_self_improvement_audit_path = (
            self.gpt_self_improvement_audit_path
            if explicit_self_improvement_audit_path or not injected_client
            else None
        )
        self.gpt_verification_ledger_path = _gpt_sidecar_path(
            explicit=gpt_verification_ledger_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_VERIFICATION_LEDGER,
        )
        self.gpt_verification_ledger_report_path = _gpt_sidecar_path(
            explicit=gpt_verification_ledger_report_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_VERIFICATION_LEDGER_REPORT,
        )
        self.gpt_market_status_path = _gpt_sidecar_path(
            explicit=gpt_market_status_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_MARKET_STATUS,
        )
        self.gpt_market_status_report_path = _gpt_sidecar_path(
            explicit=gpt_market_status_report_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_MARKET_STATUS_REPORT,
        )
        self.gpt_ai_backtest_path = _attack_sidecar_path(
            explicit=gpt_ai_backtest_path,
            attack_advice_path=gpt_attack_advice_path,
            default_path=DEFAULT_AI_TEST_BOT_BACKTEST,
        )
        self.gpt_outcome_mart_path = _attack_sidecar_path(
            explicit=gpt_outcome_mart_path,
            attack_advice_path=gpt_attack_advice_path,
            default_path=DEFAULT_OUTCOME_MART,
        )
        self.gpt_post_trade_learning_path = _attack_sidecar_path(
            explicit=gpt_post_trade_learning_path,
            attack_advice_path=gpt_attack_advice_path,
            default_path=DEFAULT_POST_TRADE_LEARNING,
        )
        self.gpt_max_lanes = gpt_max_lanes
        self.gpt_wait_retry_limit = gpt_wait_retry_limit
        self.reuse_market_artifacts = reuse_market_artifacts
        self.refresh_market_story = refresh_market_story
        self.market_news_root = market_news_root if market_news_root is not None else ROOT / "logs"
        self.live_enabled = live_enabled
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy
        self._projection_preflight_summary: dict[str, Any] | None = None
        self._ai_test_bot_backtest_refreshed = False
        self._suppress_gateway_receipt_recording = False
        self._stale_gpt_handoff_reason: str | None = None

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
        if self._suppress_gateway_receipt_recording:
            return
        if not self._execution_ledger_available():
            return
        for kind, path in (
            ("live_order", self.live_order_output_path),
            ("position_execution", self.position_execution_path),
        ):
            self._record_execution_ledger_receipt(kind=kind, receipt_path=path)

    def _record_execution_ledger_receipt(self, *, kind: str, receipt_path: Path) -> None:
        if not self._execution_ledger_available():
            return
        ExecutionLedger(
            db_path=self.execution_ledger_db_path,
            report_path=self.execution_ledger_report_path,
        ).record_gateway_receipt(kind=kind, receipt_path=receipt_path)

    def _execution_ledger_available(self) -> bool:
        return hasattr(self.client, "account_summary") and hasattr(self.client, "transactions_since_id")

    def _clear_stale_live_order_artifact(self, *, generated_at: str, cycle_send_requested: bool) -> None:
        """Overwrite previous SENT latest-state files before this cycle decides."""
        if not (self.live_order_output_path.exists() or self.live_order_report_path.exists()):
            return
        result = {
            "generated_at_utc": generated_at,
            "status": "NO_ACTION",
            "lane_id": None,
            "lane_ids": [],
            "requested_units": None,
            "size_multiple": None,
            "scaled_units": None,
            "send_requested": False,
            "cycle_send_requested": cycle_send_requested,
            "sent": False,
            "sent_count": 0,
            "portfolio_position_cap": None,
            "order_request": None,
            "risk_issues": [],
            "strategy_issues": [],
            "reason": "cleared stale latest-state live order artifact before current cycle decision",
        }
        self.live_order_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_order_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.live_order_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_order_report_path.write_text(
            "\n".join(
                [
                    "# Live Order Stage Report",
                    "",
                    f"- Generated at UTC: `{generated_at}`",
                    "- Status: `NO_ACTION`",
                    "- Lane: `None`",
                    "- Lanes: `none`",
                    "- Requested units: `None` size multiple: `None` scaled units:`None`",
                    "- Send requested: `False`",
                    f"- Cycle send requested: `{cycle_send_requested}`",
                    "- Sent: `False`",
                    "- Sent count: `0`",
                    "",
                    "## Order Request",
                    "",
                    "- none",
                    "",
                    "## Issues",
                    "",
                    "- none",
                    "",
                    "## Send Contract",
                    "",
                    "- This report is overwritten by `LiveOrderGateway` when the current cycle actually stages or sends a fresh entry.",
                    "- A stale prior SENT report must not be read as today's live send.",
                ]
            )
            + "\n"
        )

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
        if not self._trader_journal_enabled:
            return
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

    def _run(self, *, send: bool = False, _close_reentry_depth: int = 0) -> AutoTradeCycleSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        self._stale_gpt_handoff_reason = None
        stale_gpt_reason = self._external_gpt_decision_refresh_reason()
        if stale_gpt_reason is not None:
            source_path = getattr(self.gpt_provider, "source_path", None)
            receipt_exists = source_path is not None and Path(source_path).exists()
            if not receipt_exists:
                # No external receipt was ever produced: the playbook handoff
                # contract is broken, so fail loud (§3.5) instead of guessing.
                self._suppress_gateway_receipt_recording = True
                summary = self._stale_gpt_decision_summary(generated_at, stale_gpt_reason)
                self._write_report(summary, generated_at)
                return summary
            # §2/§8: a receipt that was already consumed, already verified as
            # non-TRADE, or predates the current market artifacts is NOT a
            # cycle-stop condition. Continue the cycle deterministically —
            # position management, pending-entry handling, and the campaign
            # exposure occupancy recovery all still run. The stale receipt
            # itself must never reach the gateway: _run_gpt_handoff
            # short-circuits on this reason instead of re-verifying it.
            self._stale_gpt_handoff_reason = stale_gpt_reason
        self._clear_stale_live_order_artifact(generated_at=generated_at, cycle_send_requested=send)
        pairs = DEFAULT_TRADER_PAIRS
        if self.reuse_market_artifacts:
            snapshot = self._load_snapshot_artifact()
            if send and self.live_enabled:
                snapshot = self._refresh_live_position_snapshot(snapshot)
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
        # H (2026-05-13) — Trailing SL pass. Runs ONCE per cycle on
        # trader-owned positions that already carry a broker SL. By
        # construction `apply_trailing_sls` skips every position with
        # `stop_loss is None`, so every SL-free legacy position is
        # mechanically untouchable. Opt out for tests via
        # `QR_DISABLE_TRAILING_SL=1`; production cycles default ON.
        self._maybe_apply_trailing_sls(snapshot, send=send)
        # Projection resolution is learning/audit housekeeping, not intent
        # repricing. Run it even when the decision packet is reused so a gateway
        # cycle cannot preserve expired PENDING forecasts into the next audit.
        self._verify_projection_preflight(snapshot)
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
            self._refresh_campaign_plan(target_summary)
            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(snapshot_path=self.snapshot_path)
        position_decision = None
        position_execution = None
        if trader_positions:
            decision = self._brain().run(snapshot)
            managed_snapshot = _position_management_snapshot(snapshot)
            position_decision = self._position_manager().run(managed_snapshot)
            position_execution = self._position_gateway().run(
                decision=position_decision,
                snapshot=managed_snapshot,
                send=send,
            )
            canceled_orders: list[str] = []
            canceled_status = "CANCELED_CONTAMINATED_PENDING"
            target_open = (
                target_summary is not None
                and target_summary.status == "PURSUE_TARGET"
                and target_summary.remaining_target_jpy > 0
            )
            target_reached = target_summary is not None and target_summary.status == "TARGET_REACHED_PROTECT"
            portfolio_add_allowed = _portfolio_add_allowed(snapshot)
            if (
                pending_entries
                and send
                and self.live_enabled
                and not position_execution.sent
                and position_execution.status == "NO_ACTION"
                and decision.pending_cancel_order_ids
                and not (target_open and portfolio_add_allowed)
            ):
                for order_id in decision.pending_cancel_order_ids:
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
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
            managed_snapshot = _position_management_snapshot(snapshot)
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
            # Target-reached takes precedence over per-cycle contamination
            # (2026-05-12 reorder, see `test_target_reached_cancels_trader_pending_entry`):
            # when the campaign target is hit, the correct dominant
            # signal is "day is done, protect" — labeling the cancel as
            # CONTAM hides the campaign milestone behind a per-cycle
            # lane veto. Both branches still cancel the same trader-owned
            # pending orders; only the status label changes.
            target_reached = target_summary is not None and target_summary.status == "TARGET_REACHED_PROTECT"
            if target_reached and send and self.live_enabled:
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
            # Pending entries are a live thesis, not a one-cycle artifact. If
            # the target is still open and current LIVE_READY lanes exist, let
            # GPT/gateway decide whether to preserve, add, or explicitly cancel
            # them. Preemptively canceling here prevents the market-reading
            # layer from choosing "keep pending + add current basket".
            if not canceled_orders and target_open:
                basket_lane_ids, basket_size_multiples = self._basket_lane_plan(
                    decision=decision,
                    primary_lane_id=None,
                    primary_size_multiple=None,
                    allow_existing_pending=True,
                    margin_room_jpy=_basket_margin_room_jpy(snapshot),
                )
                if not basket_lane_ids and self.use_gpt_trader:
                    gpt_summary = self._run_gpt_handoff()
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
                            deterministic_lane_id=None,
                            sent=False,
                            positions=positions,
                            orders=orders,
                            live_ready=intent_summary.live_ready,
                            selected_lane_ids=(),
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
                    if self.use_gpt_trader:
                        gpt_summary = self._run_gpt_handoff()
                        gpt_lane_ids = (
                            gpt_summary.selected_lane_ids
                            or ((gpt_summary.selected_lane_id,) if gpt_summary.selected_lane_id else ())
                        )
                        if (
                            gpt_summary.status == "ACCEPTED"
                            and gpt_summary.allowed
                            and gpt_summary.action == "CLOSE"
                        ):
                            close_execution = self._close_gpt_trades(gpt_summary, snapshot=snapshot, send=send)
                            return self._continue_after_gpt_close(
                                generated_at=generated_at,
                                send=send,
                                close_execution=close_execution,
                                close_gpt_summary=gpt_summary,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                deterministic_lane_id=basket_lane_ids[0] if basket_lane_ids else None,
                                target_summary=target_summary,
                                canceled_orders=tuple(canceled_orders),
                                close_reentry_depth=_close_reentry_depth,
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
                        gpt_lanes_allowed, gpt_recovery_bypass = _gpt_lanes_pass_prefilter_or_recovery(
                            intents_payload=json.loads(self.intents_path.read_text()),
                            gpt_lane_ids=gpt_lane_ids,
                            prefiltered_lane_ids=set(basket_lane_ids),
                        )
                        if not gpt_lanes_allowed:
                            canceled_orders.extend(
                                self._cancel_gpt_pending_orders(
                                    gpt_summary,
                                    send=send,
                                    already_canceled=tuple(canceled_orders),
                                )
                            )
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
                        if gpt_recovery_bypass:
                            gpt_recovery_source = "RECOVERY_HEDGE_GPT_NOT_PREFILTERED"
                        canceled_orders.extend(
                            self._cancel_gpt_pending_orders(
                                gpt_summary,
                                send=send,
                                already_canceled=tuple(canceled_orders),
                            )
                        )
                        basket_lane_ids, basket_size_multiples = self._expanded_gpt_basket_plan(
                            decision=decision,
                            gpt_lane_ids=gpt_lane_ids,
                            allow_existing_pending=True,
                            margin_room_jpy=_basket_margin_room_jpy(snapshot),
                        )
                    order_summary = LiveOrderGateway(
                        client=self.client,
                        strategy_profile=self.strategy_profile_path,
                        output_path=self.live_order_output_path,
                        report_path=self.live_order_report_path,
                        live_enabled=self.live_enabled,
                        max_loss_jpy=resolved_max_loss_jpy,
                        portfolio_loss_cap_jpy=self._portfolio_loss_cap_jpy_from_target_state(),
                        self_improvement_audit=self.gateway_self_improvement_audit_path,
                        verified_decision_path=self.gpt_decision_path if self.use_gpt_trader else None,
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
                        decision_source="gpt_trader" if gpt_summary else "deterministic_basket",
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
            if (
                not canceled_orders
                and send
                and self.live_enabled
                and trader_positions == 0
                and decision.pending_cancel_order_ids
            ):
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

        if self.reuse_market_artifacts:
            promotion_summary = self._skipped_receipt_promotion_summary()
        else:
            promotion_summary = self._receipt_promoter().run()
        if promotion_summary.promoted and not self.reuse_market_artifacts:
            self._refresh_campaign_plan(target_summary)
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
                # Flat-with-positions GPT path must execute CLOSE / CANCEL_PENDING
                # the same way the basket-with-pending path does upstream. Without
                # these branches, an ACCEPTED CLOSE silently returns GPT_CLOSE and
                # the broker is never asked to retire the named trades.
                if gpt_summary.action == "CLOSE":
                    close_execution = self._close_gpt_trades(gpt_summary, snapshot=snapshot, send=send)
                    return self._continue_after_gpt_close(
                        generated_at=generated_at,
                        send=send,
                        close_execution=close_execution,
                        close_gpt_summary=gpt_summary,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        deterministic_lane_id=deterministic_lane_id,
                        target_summary=target_summary,
                        receipt_promotions=promotion_summary.promoted,
                        campaign_exposure_required=campaign_exposure_required,
                        close_reentry_depth=_close_reentry_depth,
                    )
                if gpt_summary.action == "CANCEL_PENDING":
                    canceled_pending = self._cancel_gpt_pending_orders(gpt_summary, send=send)
                    summary = AutoTradeCycleSummary(
                        status="CANCELED_GPT_PENDING" if canceled_pending else "GPT_CANCEL_PENDING",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        canceled_orders=tuple(canceled_pending),
                        receipt_promotions=promotion_summary.promoted,
                        decision_source="gpt_trader",
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
                        campaign_exposure_required=campaign_exposure_required,
                    )
                    self._write_report(summary, generated_at)
                    return summary
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
                    # Retry only when a current LIVE_READY lane makes flat
                    # exposure itself the failure. With no exposure candidate,
                    # regenerating intents just inflates forecast/projection
                    # ledgers and cannot convert a verified WAIT into a trade.
                    if self.gpt_wait_retry_limit > 0 and target_is_pursue and campaign_exposure_required:
                        for attempt in range(1, self.gpt_wait_retry_limit + 1):
                            gpt_wait_retries = attempt
                            snapshot = self._refresh_snapshot(pairs)
                            target_summary = self._update_target_state(snapshot) or target_summary
                            positions = len(snapshot.positions)
                            orders = len(snapshot.orders)
                            self._refresh_campaign_plan(target_summary)
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
                if intent_summary.live_ready == 0 and gpt_summary.status == "STALE_DECISION":
                    status = "NO_LIVE_READY_INTENT"
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
        if selected_lane_id:
            if self.use_gpt_trader:
                prefiltered_lane_ids = {
                    item.lane_id for item in decision.scores if _passes_gpt_prefilter(item)
                }
                if gpt_summary is None:
                    gpt_summary = self._run_gpt_handoff()
                if (
                    gpt_summary.status == "ACCEPTED"
                    and gpt_summary.allowed
                    and gpt_summary.action == "CLOSE"
                ):
                    close_execution = self._close_gpt_trades(gpt_summary, snapshot=snapshot, send=send)
                    return self._continue_after_gpt_close(
                        generated_at=generated_at,
                        send=send,
                        close_execution=close_execution,
                        close_gpt_summary=gpt_summary,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        deterministic_lane_id=deterministic_lane_id,
                        target_summary=target_summary,
                        receipt_promotions=promotion_summary.promoted,
                        gpt_wait_retries=gpt_wait_retries,
                        gpt_recovery_source=gpt_recovery_source,
                        campaign_exposure_required=campaign_exposure_required,
                        close_reentry_depth=_close_reentry_depth,
                    )
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
                if gpt_trade_accepted:
                    gpt_lanes_allowed, gpt_recovery_bypass = _gpt_lanes_pass_prefilter_or_recovery(
                        intents_payload=json.loads(self.intents_path.read_text()),
                        gpt_lane_ids=gpt_selected_lane_ids,
                        prefiltered_lane_ids=prefiltered_lane_ids,
                    )
                    if gpt_recovery_bypass:
                        gpt_recovery_source = "RECOVERY_HEDGE_GPT_NOT_PREFILTERED"
                    if not gpt_lanes_allowed:
                        if campaign_exposure_required:
                            gpt_recovery_source = "CAMPAIGN_EXPOSURE_RECOVERY_GPT_NOT_PREFILTERED"
                        else:
                            canceled_orders.extend(
                                self._cancel_gpt_pending_orders(
                                    gpt_summary,
                                    send=send,
                                    already_canceled=tuple(canceled_orders),
                                )
                            )
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
                margin_room_jpy=_basket_margin_room_jpy(snapshot),
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
                margin_room_jpy=_basket_margin_room_jpy(snapshot),
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

        if (
            basket_lane_ids
            and trader_positions > 0
            and not _portfolio_entry_capacity_open(snapshot, target_summary)
        ):
            summary = AutoTradeCycleSummary(
                status="MONITOR_ONLY_EXPOSURE_OPEN",
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=selected_lane_id,
                selected_lane_ids=basket_lane_ids,
                selected_lane_score=selected_lane_score,
                selected_lane_size_multiple=selected_lane_size_multiple,
                deterministic_lane_id=deterministic_lane_id,
                sent=False,
                sent_count=0,
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
                campaign_exposure_required=campaign_exposure_required,
            )
            self._write_report(summary, generated_at)
            return summary

        order_gateway = LiveOrderGateway(
            client=self.client,
            strategy_profile=self.strategy_profile_path,
            output_path=self.live_order_output_path,
            report_path=self.live_order_report_path,
            live_enabled=self.live_enabled,
            max_loss_jpy=resolved_max_loss_jpy,
            portfolio_loss_cap_jpy=self._portfolio_loss_cap_jpy_from_target_state(),
            self_improvement_audit=self.gateway_self_improvement_audit_path,
            verified_decision_path=self.gpt_decision_path if self.use_gpt_trader else None,
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

    def _external_gpt_decision_refresh_reason(self) -> str | None:
        if not self.use_gpt_trader:
            return None
        source_path = getattr(self.gpt_provider, "source_path", None)
        if source_path is None:
            return None
        source_path = Path(source_path)
        if not source_path.exists():
            return f"external GPT decision response is missing: {source_path}"
        decision_mtime_ns = source_path.stat().st_mtime_ns
        reusable_gpt = self._load_reusable_verified_gpt_handoff()
        if reusable_gpt is not None:
            consumed_reference_ns = self.gpt_decision_path.stat().st_mtime_ns
            for path, label in self._gpt_consuming_receipts(reusable_gpt.action):
                if self._gpt_receipt_consumes_verified_handoff(
                    path=path,
                    label=label,
                    action=reusable_gpt.action,
                    reference_mtime_ns=consumed_reference_ns,
                    close_trade_ids=reusable_gpt.close_trade_ids,
                ):
                    return (
                        f"external GPT decision response already consumed by {label}; "
                        "refresh broker truth and write one current receipt"
                    )
            return None
        for path, label in (
            (self.snapshot_path, "broker snapshot"),
            (self.intents_path, "order intents"),
        ):
            if path.exists() and path.stat().st_mtime_ns > decision_mtime_ns:
                return (
                    f"external GPT decision response predates {label}; "
                    f"refresh decision from broker truth before gateway handoff: {source_path}"
                )

        gpt_mtime_ns: int | None = None
        if self.gpt_decision_path.exists() and self.gpt_decision_path.stat().st_mtime_ns > decision_mtime_ns:
            gpt_mtime_ns = self.gpt_decision_path.stat().st_mtime_ns
            try:
                payload = json.loads(self.gpt_decision_path.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                payload = {}
            status = str(payload.get("status") or "")
            decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            action = str(decision.get("action") or "")
            if status != "ACCEPTED" or action != "TRADE":
                return (
                    f"external GPT decision response was already verified as "
                    f"{status or 'UNKNOWN'} {action or 'NO_ACTION'}; "
                    "write a fresh receipt before another gateway cycle"
                )

        consumed_reference_ns = gpt_mtime_ns if gpt_mtime_ns is not None else decision_mtime_ns
        for path, label in self._gpt_consuming_receipts("TRADE"):
            if self._gpt_receipt_consumes_verified_handoff(
                path=path,
                label=label,
                action="TRADE",
                reference_mtime_ns=consumed_reference_ns,
                close_trade_ids=(),
            ):
                return (
                    f"external GPT decision response already consumed by {label}; "
                    "refresh broker truth and write one current receipt"
                )
        return None

    def _gpt_consuming_receipts(self, action: str | None) -> tuple[tuple[Path, str], ...]:
        normalized = str(action or "").upper()
        receipts: list[tuple[Path, str]] = []
        if normalized in GPT_LIVE_ORDER_ACTIONS:
            receipts.append((self.live_order_output_path, "live order gateway receipt"))
        if normalized in GPT_POSITION_GATEWAY_ACTIONS:
            receipts.append((self.position_execution_path, "position gateway receipt"))
        if not receipts:
            receipts.extend(
                (
                    (self.live_order_output_path, "live order gateway receipt"),
                    (self.position_execution_path, "position gateway receipt"),
                )
            )
        receipts.append((self.report_path, "autotrade cycle report"))
        return tuple(receipts)

    def _gpt_receipt_consumes_verified_handoff(
        self,
        *,
        path: Path,
        label: str,
        action: str | None,
        reference_mtime_ns: int,
        close_trade_ids: tuple[str, ...],
    ) -> bool:
        if not path.exists() or path.stat().st_mtime_ns <= reference_mtime_ns:
            return False
        normalized = str(action or "").upper()
        if normalized == "CLOSE" and path == self.position_execution_path:
            return self._position_execution_consumes_gpt_close(path, close_trade_ids=close_trade_ids)
        return True

    def _position_execution_consumes_gpt_close(
        self,
        path: Path,
        *,
        close_trade_ids: tuple[str, ...],
    ) -> bool:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return True
        status = str(payload.get("status") or "").upper()
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        wanted = {str(trade_id) for trade_id in close_trade_ids if str(trade_id)}
        for action_payload in actions:
            if not isinstance(action_payload, dict):
                continue
            trade_id = str(action_payload.get("trade_id") or "")
            if wanted and trade_id not in wanted:
                continue
            management_action = str(action_payload.get("management_action") or "").upper()
            reason_text = " ".join(str(reason) for reason in action_payload.get("reasons", [])).lower()
            if status == "STALE_CLOSE_SATISFIED" and management_action == "GPT_CLOSE":
                return True
            if "gpt-close: accepted gpt_trader close receipt passed gate a/b" in reason_text:
                return True
        return False

    def _stale_gpt_decision_summary(self, generated_at: str, reason: str) -> AutoTradeCycleSummary:
        positions = 0
        orders = 0
        live_ready = 0
        target_status = None
        target_remaining_jpy = None
        target_progress_pct = None
        snapshot = None
        try:
            snapshot = self._load_snapshot_artifact()
            positions = len(snapshot.positions)
            orders = len(snapshot.orders)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if snapshot is not None:
            self._verify_projection_preflight(snapshot)
        try:
            intent_summary = self._load_intent_summary_artifact()
            live_ready = intent_summary.live_ready
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if self.target_state_path is not None and self.target_state_path.exists():
            try:
                target_payload = json.loads(self.target_state_path.read_text())
                target_status = target_payload.get("status")
                target_remaining_jpy = _optional_float(target_payload.get("remaining_target_jpy"))
                target_progress_pct = _optional_float(target_payload.get("progress_pct"))
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return AutoTradeCycleSummary(
            status="STALE_GPT_DECISION_REFRESH_REQUIRED",
            report_path=self.report_path,
            snapshot_path=self.snapshot_path,
            intents_path=self.intents_path,
            selected_lane_id=None,
            deterministic_lane_id=None,
            sent=False,
            positions=positions,
            orders=orders,
            live_ready=live_ready,
            decision_source="gpt_trader",
            target_status=target_status,
            target_remaining_jpy=target_remaining_jpy,
            target_progress_pct=target_progress_pct,
            gpt_status="STALE_DECISION",
            gpt_action=None,
            gpt_allowed=False,
            gpt_issues=1,
            gpt_error=reason,
        )

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
        if self._projection_preflight_summary is not None:
            preflight = self._projection_preflight_summary
            lines.append(
                "- Projection preflight: "
                f"status=`{preflight.get('status')}` "
                f"expired_pairs=`{preflight.get('expired_pending_pairs')}` "
                f"pending_pairs=`{preflight.get('pending_pairs')}` "
                f"counts=`{preflight.get('resolution_counts')}` "
                f"error=`{preflight.get('error') or 'none'}`"
            )
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
                "- Open positions are handed to PositionManager first; trader-owned positions may close/repair/tighten when gated, while manual/tagless positions are TP-only.",
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
        data_root = self.target_state_path.parent if self.target_state_path is not None else self.intents_path.parent
        return IntentGenerator(
            campaign_plan=self.campaign_plan_path,
            pair_charts_path=self.pair_charts_path,
            strategy_profile=self.strategy_profile_path,
            output_path=self.intents_path,
            report_path=self.intent_report_path,
            data_root=data_root,
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

    def _refresh_live_position_snapshot(self, snapshot):
        # `--reuse-market-artifacts` pins the decision packet, but position
        # management and close execution must still use current broker truth.
        pairs = _snapshot_refresh_pairs(snapshot) or DEFAULT_TRADER_PAIRS
        return self.client.snapshot(pairs)

    def _verify_projection_preflight(self, snapshot) -> dict[str, Any] | None:
        if os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE", "").strip() not in {
            "1", "true", "TRUE", "yes", "YES",
        }:
            self._projection_preflight_summary = {"status": "DISABLED"}
            return self._projection_preflight_summary
        data_root = self.intents_path.parent
        try:
            from quant_rabbit.strategy.projection_ledger import (
                load_ledger,
                retryable_truth_timeout_pairs,
                verify_pending,
            )

            entries = load_ledger(data_root)
            now = getattr(snapshot, "fetched_at_utc", None)
            if not isinstance(now, datetime):
                now = datetime.now(timezone.utc)
            elif now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            else:
                now = now.astimezone(timezone.utc)
            pending_pairs: set[str] = set()
            expired_pairs: set[str] = set()
            for entry in entries:
                if entry.resolution_status != "PENDING" or not entry.pair:
                    continue
                pending_pairs.add(entry.pair)
                try:
                    emitted_at = datetime.fromisoformat(
                        entry.timestamp_emitted_utc.replace("Z", "+00:00")
                    )
                    window_min = float(entry.resolution_window_min or 0)
                except (TypeError, ValueError):
                    continue
                if (now - emitted_at).total_seconds() / 60.0 >= window_min:
                    expired_pairs.add(entry.pair)
            retry_timeout_pairs = retryable_truth_timeout_pairs(entries)
            pending_pairs_sorted = sorted(pending_pairs)
            expired_pairs_sorted = sorted(expired_pairs)
            verification_pairs_sorted = sorted(pending_pairs | retry_timeout_pairs)
        except Exception as exc:
            self._projection_preflight_summary = {
                "status": "LOAD_FAILED",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            }
            return self._projection_preflight_summary
        if not verification_pairs_sorted:
            self._projection_preflight_summary = {
                "status": "NO_PENDING",
                "expired_pending_pairs": 0,
                "pending_pairs": 0,
                "retryable_timeout_pairs": 0,
            }
            return self._projection_preflight_summary
        quotes_by_pair: dict[str, dict[str, float]] = {}
        for pair, quote in getattr(snapshot, "quotes", {}).items():
            try:
                quotes_by_pair[str(pair)] = {"bid": float(quote.bid), "ask": float(quote.ask)}
            except (TypeError, ValueError):
                continue
        atr_pips_by_pair = _projection_atr_pips_by_pair(self.pair_charts_path)
        candles_by_pair = None
        candle_truth_summary: dict[str, Any] = {
            "candle_counts": {},
            "candle_granularity_counts": {},
            "candle_errors": {},
            "candle_truth_deadline_exceeded": False,
        }
        if hasattr(self.client, "get_json"):
            try:
                from quant_rabbit.projection_truth import (
                    load_projection_candle_truth,
                    projection_candle_truth_summary,
                )

                m1_count = int(os.environ.get("QR_PROJECTION_VERIFY_M1_COUNT", "1500"))
                m5_count = int(os.environ.get("QR_PROJECTION_VERIFY_M5_COUNT", "1500"))
                candle_truth = load_projection_candle_truth(
                    self.client,
                    verification_pairs_sorted,
                    m1_count=m1_count,
                    m5_count=m5_count,
                )
                candles_by_pair = candle_truth.candles_by_pair
                candle_truth_summary = projection_candle_truth_summary(candle_truth)
            except Exception as exc:
                candle_truth_summary["candle_errors"] = {"_loader": f"{type(exc).__name__}: {str(exc)[:160]}"}
                candles_by_pair = None
        try:
            counts = verify_pending(
                data_root,
                quotes_by_pair=quotes_by_pair,
                atr_pips_by_pair=atr_pips_by_pair,
                candles_by_pair=candles_by_pair,
                now=now,
            )
        except Exception as exc:
            self._projection_preflight_summary = {
                "status": "VERIFY_FAILED",
                "expired_pending_pairs": len(expired_pairs_sorted),
                "pending_pairs": len(pending_pairs_sorted),
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                **candle_truth_summary,
            }
            return self._projection_preflight_summary
        self._projection_preflight_summary = {
            "status": "OK",
            "expired_pending_pairs": len(expired_pairs_sorted),
            "pending_pairs": len(pending_pairs_sorted),
            "retryable_timeout_pairs": len(retry_timeout_pairs),
            "resolution_counts": counts,
            **candle_truth_summary,
        }
        return self._projection_preflight_summary

    def _maybe_apply_trailing_sls(self, snapshot, *, send: bool) -> None:
        """H (2026-05-13) — trailing SL pass on trader-owned positions
        that ALREADY have a broker SL set. Skips SL-free positions by
        construction; every existing legacy trade is mechanically
        untouchable. Errors propagate as logs so the broker can be
        temporarily unavailable without stopping the cycle.

        Gated:
          - `QR_DISABLE_TRAILING_SL=1` skips entirely (test escape).
          - Only runs when `send=True` and `self.live_enabled=True`;
            dry-run cycles do not modify broker state.
        """
        if os.environ.get("QR_DISABLE_TRAILING_SL", "").strip() in {
            "1", "true", "TRUE", "yes", "YES",
        }:
            return
        if not send or not self.live_enabled:
            return
        try:
            from quant_rabbit.strategy.trailing_sl import apply_trailing_sls
            pair_charts_payload = (
                json.loads(self.pair_charts_path.read_text())
                if self.pair_charts_path and self.pair_charts_path.exists()
                else {}
            )
            apply_trailing_sls(
                snapshot=snapshot,
                pair_charts_payload=pair_charts_payload,
                broker_client=self.client,
                dry_run=False,
            )
        except Exception:
            # Trailing SL is advisory protection; broker errors during
            # update must not stop the cycle. Position remains at its
            # existing SL.
            return

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
        ledger = DailyTargetLedger(
            state_path=self.target_state_path,
            report_path=report_path,
            pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
            execution_ledger_path=self.execution_ledger_db_path,
        )
        summary = ledger.run(snapshot=snapshot)
        if self._refresh_ai_test_bot_backtest_for_target_pace():
            summary = ledger.run(snapshot=snapshot)
        return summary

    def _refresh_campaign_plan(self, target_summary: DailyTargetSummary | None) -> None:
        if target_summary is None:
            return
        if target_summary.status == "TARGET_REACHED_PROTECT":
            return
        if self.target_state_path is None or not self.target_state_path.exists():
            return
        try:
            target_payload = json.loads(self.target_state_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return
        start_balance = _optional_float(target_payload.get("start_balance_jpy"))
        target_jpy = _optional_float(target_payload.get("target_jpy"))
        if start_balance is None or start_balance <= 0:
            return
        target_return_pct = 10.0
        if target_jpy is not None and target_jpy > 0:
            target_return_pct = (target_jpy / start_balance) * 100.0
        campaign_report_path = (
            DEFAULT_CAMPAIGN_REPORT
            if self.campaign_plan_path == DEFAULT_CAMPAIGN_PLAN
            else self.campaign_plan_path.with_suffix(".md")
        )
        CampaignPlanner(
            strategy_profile=self.strategy_profile_path,
            market_story_profile=self.market_story_profile_path,
            report_path=campaign_report_path,
            plan_path=self.campaign_plan_path,
        ).run(start_balance_jpy=start_balance, target_return_pct=target_return_pct)

    def _refresh_ai_test_bot_backtest_for_target_pace(self) -> bool:
        """Refresh target-band evidence before daily target pace is trusted."""

        if self._ai_test_bot_backtest_refreshed:
            return False
        if _running_under_test_harness() and os.environ.get("QR_REFRESH_AI_BACKTEST_IN_TESTS") != "1":
            return False
        if not DEFAULT_HISTORY_DB.exists():
            return False
        if self.target_state_path is None or not self.target_state_path.exists():
            return False
        try:
            target_payload = json.loads(self.target_state_path.read_text())
            start_balance = float(target_payload.get("start_balance_jpy") or 0.0)
            target_return_pct = float(target_payload.get("target_return_pct") or 10.0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("ai-test-bot backtest refresh requires readable daily target state") from exc
        if start_balance <= 0:
            return False
        try:
            AITestBotBacktester(
                db_path=DEFAULT_HISTORY_DB,
                output_path=DEFAULT_AI_TEST_BOT_BACKTEST,
                report_path=DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
                target_state_path=self.target_state_path,
                execution_ledger_db_path=self.execution_ledger_db_path,
            ).run(
                start_balance_jpy=start_balance,
                target_return_pct=target_return_pct,
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("ai-test-bot backtest refresh failed before daily target pacing") from exc
        self._ai_test_bot_backtest_refreshed = True
        return True

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
            pair_charts_path=self.pair_charts_path,
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
        margin_room_jpy: float | None = None,
        margin_available_jpy: float | None = None,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        lane_ids: list[str] = []
        size_multiples: dict[str, float] = {}
        parent_lane_ids: set[str] = set()
        margin_budget = _buffered_basket_margin_budget_jpy(
            margin_room_jpy=margin_room_jpy,
            margin_available_jpy=margin_available_jpy,
        )
        margin_lookup = {
            score.lane_id: score.estimated_margin_jpy
            for score in decision.scores
            if score.estimated_margin_jpy is not None
        }
        cumulative_margin = 0.0

        def add(lane_id: str | None, size_multiple: float | None) -> bool:
            nonlocal cumulative_margin
            if not lane_id or lane_id in size_multiples:
                return False
            parent_lane_id = _basket_parent_lane_id(lane_id)
            if parent_lane_id and parent_lane_id in parent_lane_ids:
                return False
            # C-4 margin-aware truncation. Apply the buffer-adjusted
            # effective margin room only when we know the lane's estimated
            # margin AND broker truth has surfaced an account figure.
            # Missing data degrades gracefully — without truncation —
            # so unrelated tests and stub fixtures keep working.
            if margin_budget is not None:
                lane_margin = margin_lookup.get(lane_id)
                if lane_margin is not None:
                    if cumulative_margin + lane_margin > margin_budget:
                        return False
                    cumulative_margin += lane_margin
            lane_ids.append(lane_id)
            size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
            if parent_lane_id:
                parent_lane_ids.add(parent_lane_id)
            return True

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
        margin_room_jpy: float | None = None,
        margin_available_jpy: float | None = None,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        lane_ids: list[str] = []
        size_multiples: dict[str, float] = {}
        parent_lane_ids: set[str] = set()
        margin_budget = _buffered_basket_margin_budget_jpy(
            margin_room_jpy=margin_room_jpy,
            margin_available_jpy=margin_available_jpy,
        )
        margin_lookup = {
            score.lane_id: score.estimated_margin_jpy
            for score in decision.scores
            if score.estimated_margin_jpy is not None
        }
        score_lookup = {score.lane_id: score for score in decision.scores}
        cumulative_margin = 0.0

        def add(
            lane_id: str | None,
            size_multiple: float | None = None,
            *,
            require_current_prefilter: bool = False,
        ) -> bool:
            nonlocal cumulative_margin
            if not lane_id or lane_id in size_multiples:
                return False
            if require_current_prefilter:
                score = score_lookup.get(lane_id)
                if score is None or not _passes_basket_prefilter(score, allow_existing_pending=allow_existing_pending):
                    return False
            parent_lane_id = _basket_parent_lane_id(lane_id)
            if parent_lane_id and parent_lane_id in parent_lane_ids:
                return False
            if margin_budget is not None:
                lane_margin = margin_lookup.get(lane_id)
                if lane_margin is not None:
                    if cumulative_margin + lane_margin > margin_budget:
                        return False
                    cumulative_margin += lane_margin
            if size_multiple is None:
                _, size_multiple = AutoTradeCycle._selected_lane_meta(
                    decision=decision,
                    lane_id=lane_id,
                )
            lane_ids.append(lane_id)
            size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
            if parent_lane_id:
                parent_lane_ids.add(parent_lane_id)
            return True

        for lane_id in gpt_lane_ids:
            # GPT receipts can go stale between decision writing and gateway
            # execution. Do not force a now-DRY_RUN_BLOCKED / non-prefiltered
            # lane into LiveOrderGateway.
            add(lane_id, require_current_prefilter=True)
        if lane_ids:
            # The external receipt is the discretionary execution contract.
            # When at least one explicitly selected lane remains current,
            # never append lower-priority deterministic lanes that GPT did
            # not select; otherwise the report can say "only lane X" while
            # the gateway sends lane Y as well.
            return tuple(lane_ids), size_multiples

        # If every GPT-selected lane has gone stale, recover through the
        # current deterministic LIVE_READY basket instead of dead-ending the
        # cycle on an obsolete receipt.
        for score in decision.scores:
            if _passes_basket_prefilter(score, allow_existing_pending=allow_existing_pending):
                add(score.lane_id, score.size_multiple)
        return tuple(lane_ids), size_multiples

    def _run_gpt_handoff(self) -> GptHandoffSummary:
        if self._stale_gpt_handoff_reason:
            # The external receipt was consumed, already verified as
            # non-TRADE, or predates the current market artifacts. It must not
            # be re-verified or handed to the gateway (double-send risk); the
            # cycle continues deterministically and the report's gpt_error
            # tells the scheduled trader to write one current receipt.
            return GptHandoffSummary(
                status="STALE_DECISION",
                action=None,
                selected_lane_id=None,
                allowed=False,
                issues=1,
                error=self._stale_gpt_handoff_reason,
            )
        reusable = self._load_reusable_verified_gpt_handoff()
        if reusable is not None:
            return reusable
        try:
            self._run_market_status_for_gpt_handoff()
            self._run_learning_audit_for_gpt_handoff()
            self._run_verification_ledger_for_gpt_handoff()
            summary = self._gpt_brain().run(snapshot_path=self.snapshot_path)
            return GptHandoffSummary(
                status=summary.status,
                action=summary.action,
                selected_lane_id=summary.selected_lane_id,
                allowed=summary.allowed,
                issues=summary.issues,
                selected_lane_ids=summary.selected_lane_ids,
                cancel_order_ids=summary.cancel_order_ids,
                close_trade_ids=summary.close_trade_ids,
            )
        except (RuntimeError, ValueError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            return GptHandoffSummary(
                status="ERROR",
                action=None,
                selected_lane_id=None,
                allowed=False,
                issues=1,
                error=str(exc),
            )

    def _load_reusable_verified_gpt_handoff(self) -> GptHandoffSummary | None:
        """Reuse a just-accepted gateway verification for a pinned packet.

        `autotrade-cycle --reuse-market-artifacts` is the verifier-to-gateway
        bridge. Once `gpt-trader-decision` has accepted an external gateway
        response, rerunning the verifier against a newer broker snapshot can
        reject the same receipt as stale even though the live gateway will
        fetch fresh broker truth before staging/sending. Reuse is therefore
        allowed only for the same external receipt, the same order-intent
        packet for TRADE receipts, and a still-present LIVE_READY lane for any
        selected fresh-entry lane. CLOSE / CANCEL_PENDING / protection actions
        do not select order-intent lanes, but they still need the same source
        receipt and an unconsumed accepted verifier output.
        """

        if not self.reuse_market_artifacts or not self.use_gpt_trader:
            return None
        source_path = getattr(self.gpt_provider, "source_path", None)
        if source_path is None:
            return None
        source_path = Path(source_path)
        if not source_path.exists() or not self.gpt_decision_path.exists():
            return None
        try:
            if self.gpt_decision_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
                return None
            verified_payload = json.loads(self.gpt_decision_path.read_text())
            source_payload = json.loads(source_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if verified_payload.get("status") != "ACCEPTED":
            return None
        decision = verified_payload.get("decision")
        if not isinstance(decision, dict):
            return None
        action = str(decision.get("action") or "").upper()
        if action not in ACCEPTED_GPT_GATEWAY_ACTIONS:
            return None
        if not self._verified_gpt_decision_matches_source(decision, source_payload):
            return None
        if action == "TRADE" and not self._verified_gpt_order_intents_still_match(verified_payload, decision):
            return None
        selected_lane_id = str(decision.get("selected_lane_id") or "") or None
        selected_lane_ids = self._string_tuple(decision.get("selected_lane_ids"))
        cancel_order_ids = self._string_tuple(decision.get("cancel_order_ids"))
        close_trade_ids = self._string_tuple(decision.get("close_trade_ids"))
        return GptHandoffSummary(
            status="ACCEPTED",
            action=action,
            selected_lane_id=selected_lane_id,
            allowed=True,
            issues=len(verified_payload.get("verification_issues") or []),
            selected_lane_ids=selected_lane_ids,
            cancel_order_ids=cancel_order_ids,
            close_trade_ids=close_trade_ids,
        )

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(item) for item in value if item is not None and str(item))

    @classmethod
    def _verified_gpt_decision_matches_source(cls, decision: dict[str, Any], source: dict[str, Any]) -> bool:
        for key in ("generated_at_utc", "action", "selected_lane_id"):
            left = decision.get(key)
            right = source.get(key)
            if (str(left) if left is not None else None) != (str(right) if right is not None else None):
                return False
        for key in ("selected_lane_ids", "cancel_order_ids", "close_trade_ids"):
            source_values = cls._string_tuple(source.get(key))
            decision_values = cls._string_tuple(decision.get(key))
            if source_values and decision_values != source_values:
                return False
            if not source_values and decision_values:
                return False
        return True

    def _verified_gpt_order_intents_still_match(
        self,
        verified_payload: dict[str, Any],
        decision: dict[str, Any],
    ) -> bool:
        try:
            current_intents = json.loads(self.intents_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        packet = verified_payload.get("input_packet")
        artifact_timestamps = packet.get("artifact_timestamps") if isinstance(packet, dict) else {}
        verified_intents_ts = (
            artifact_timestamps.get("order_intents_generated_at_utc")
            if isinstance(artifact_timestamps, dict)
            else None
        )
        current_intents_ts = current_intents.get("generated_at_utc")
        if verified_intents_ts or current_intents_ts:
            if verified_intents_ts != current_intents_ts:
                return False
        selected_lane_ids = self._string_tuple(decision.get("selected_lane_ids"))
        selected_lane_id = str(decision.get("selected_lane_id") or "")
        if not selected_lane_ids and selected_lane_id:
            selected_lane_ids = (selected_lane_id,)
        if selected_lane_ids:
            live_ready_lane_ids = {
                str(item.get("lane_id") or "")
                for item in current_intents.get("results", []) or []
                if isinstance(item, dict) and item.get("status") == "LIVE_READY"
            }
            if not set(selected_lane_ids).issubset(live_ready_lane_ids):
                return False
        return True

    def _continue_after_gpt_close(
        self,
        *,
        generated_at: str,
        send: bool,
        close_execution: PositionExecutionSummary,
        close_gpt_summary: GptHandoffSummary,
        positions: int,
        orders: int,
        live_ready: int,
        deterministic_lane_id: str | None,
        target_summary: DailyTargetSummary | None,
        canceled_orders: tuple[str, ...] = (),
        receipt_promotions: int = 0,
        gpt_wait_retries: int = 0,
        gpt_recovery_source: str | None = None,
        campaign_exposure_required: bool = False,
        close_reentry_depth: int = 0,
    ) -> AutoTradeCycleSummary:
        """After an accepted GPT CLOSE, finish the cycle as close-only.

        CLOSE still cannot be bundled into a TRADE receipt. The safe shape is
        close first, then wait for the next scheduled cycle so any new entry
        uses a fresh broker snapshot, freshly-priced intents, and its own
        verified GPT TRADE receipt. Same-cycle re-entry after a realized loss
        makes recovery behavior chase the just-invalidated context.
        """
        close_status = _position_execution_cycle_status(close_execution, fallback="GPT_CLOSE")
        close_only = AutoTradeCycleSummary(
            status=close_status,
            report_path=self.report_path,
            snapshot_path=self.snapshot_path,
            intents_path=self.intents_path,
            selected_lane_id=None,
            deterministic_lane_id=deterministic_lane_id,
            sent=False,
            positions=positions,
            orders=orders,
            live_ready=live_ready,
            selected_lane_ids=(),
            canceled_orders=canceled_orders,
            receipt_promotions=receipt_promotions,
            decision_source="gpt_trader",
            position_management_action="GPT_CLOSE",
            position_execution_status=close_execution.status,
            position_execution_sent=close_execution.sent,
            target_status=target_summary.status if target_summary else None,
            target_remaining_jpy=target_summary.remaining_target_jpy if target_summary else None,
            target_progress_pct=target_summary.progress_pct if target_summary else None,
            gpt_status=close_gpt_summary.status,
            gpt_action=close_gpt_summary.action,
            gpt_allowed=close_gpt_summary.allowed,
            gpt_issues=close_gpt_summary.issues,
            gpt_error=close_gpt_summary.error,
            gpt_wait_retries=gpt_wait_retries,
            gpt_recovery_source=gpt_recovery_source,
            campaign_exposure_required=campaign_exposure_required,
        )
        close_satisfied = (
            close_execution.sent
            or close_execution.status == "STAGED"
            or close_execution.status == "STALE_CLOSE_SATISFIED"
        )
        if close_satisfied and close_only.gpt_recovery_source is None:
            close_only = replace(close_only, gpt_recovery_source="POST_CLOSE_REENTRY_DEFERRED")
        self._write_report(close_only, generated_at)
        return close_only

    def _archive_gpt_close_receipt_for_reentry(self) -> None:
        for path in (self.gpt_decision_path, self.gpt_decision_report_path):
            if not path.exists():
                continue
            archive_path = path.with_name(f"{path.stem}.close_reentry{path.suffix}")
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive_path)

    def _run_market_status_for_gpt_handoff(self) -> None:
        status = compute_market_status()
        write_market_status_snapshot(status, self.gpt_market_status_path)
        write_market_status_report(status, self.gpt_market_status_report_path)

    def _run_learning_audit_for_gpt_handoff(self) -> None:
        LearningAuditor(
            db_path=self.gpt_learning_audit_db_path,
            output_path=self.gpt_learning_audit_path,
            report_path=self.gpt_learning_audit_report_path,
        ).run(
            ai_backtest_path=self.gpt_ai_backtest_path,
            outcome_mart_path=self.gpt_outcome_mart_path,
            post_trade_learning_path=self.gpt_post_trade_learning_path,
            ai_attack_advice_path=self.gpt_attack_advice_path,
        )

    def _run_verification_ledger_for_gpt_handoff(self) -> None:
        VerificationLedger(
            db_path=self.gpt_learning_audit_db_path,
            output_path=self.gpt_verification_ledger_path,
            report_path=self.gpt_verification_ledger_report_path,
        ).run(
            snapshot_path=self.snapshot_path,
            order_intents_path=self.intents_path,
            gpt_decision_path=self.gpt_decision_path,
            live_order_path=self.live_order_output_path,
            position_execution_path=self.position_execution_path,
            ai_backtest_path=self.gpt_ai_backtest_path,
            outcome_mart_path=self.gpt_outcome_mart_path,
            post_trade_learning_path=self.gpt_post_trade_learning_path,
            ai_attack_advice_path=self.gpt_attack_advice_path,
            learning_audit_path=self.gpt_learning_audit_path,
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

    def _close_gpt_trades(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        snapshot,
        send: bool,
    ) -> PositionExecutionSummary:
        # Operator-directed market close on trade ids named in
        # decision.close_trade_ids. The verifier (gpt_trader) supplies Gate A/B;
        # live sends refresh broker truth immediately before PositionProtectionGateway
        # checks ownership, live enablement, and receipt/report persistence.
        no_action = PositionExecutionSummary(
            status="NO_ACTION",
            output_path=self.position_execution_path,
            report_path=self.position_execution_report_path,
            sent=False,
            actions=0,
            blocked=0,
        )
        if (
            gpt_summary.status != "ACCEPTED"
            or not gpt_summary.allowed
            or gpt_summary.action != "CLOSE"
        ):
            if gpt_summary.close_trade_ids:
                sys.stderr.write(
                    f"[automation._close_gpt_trades] blocked non-accepted close: "
                    f"status={gpt_summary.status} allowed={gpt_summary.allowed} "
                    f"action={gpt_summary.action} "
                    f"close_trade_ids={list(gpt_summary.close_trade_ids)}\n"
                )
            return no_action
        if not gpt_summary.close_trade_ids:
            sys.stderr.write(
                f"[automation._close_gpt_trades] short-circuit: "
                f"close_trade_ids={list(gpt_summary.close_trade_ids)}\n"
            )
            return no_action
        self._record_execution_ledger_receipt(
            kind="gpt_decision",
            receipt_path=self.gpt_decision_path,
        )
        close_snapshot = snapshot
        if send and self.live_enabled:
            close_snapshot = self._refresh_snapshot(_snapshot_refresh_pairs(snapshot))
            open_trade_ids = {
                str(getattr(position, "trade_id", "") or "")
                for position in getattr(close_snapshot, "positions", ()) or ()
            }
            if all(str(trade_id) not in open_trade_ids for trade_id in gpt_summary.close_trade_ids):
                return self._write_stale_gpt_close_satisfied(gpt_summary, snapshot=close_snapshot)
        decision = self._gpt_close_position_decision(gpt_summary, snapshot=close_snapshot)
        execution = self._position_gateway().run(decision=decision, snapshot=close_snapshot, send=send)
        self._record_execution_ledger_receipt(
            kind="position_execution",
            receipt_path=self.position_execution_path,
        )
        return execution

    def _write_stale_gpt_close_satisfied(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        snapshot,
    ) -> PositionExecutionSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        actions = []
        for trade_id in gpt_summary.close_trade_ids:
            actions.append(
                {
                    "trade_id": str(trade_id),
                    "pair": "",
                    "owner": "",
                    "management_action": "GPT_CLOSE",
                    "request": None,
                    "issues": [
                        {
                            "severity": "INFO",
                            "code": "STALE_CLOSE_ALREADY_ABSENT",
                            "message": (
                                "accepted CLOSE receipt named a trade id that is already absent "
                                "from the refreshed broker snapshot"
                            ),
                        }
                    ],
                    "sent": False,
                    "response": None,
                }
            )
        result = {
            "generated_at_utc": generated_at,
            "status": "STALE_CLOSE_SATISFIED",
            "send_requested": True,
            "sent": False,
            "snapshot_fetched_at_utc": str(getattr(snapshot, "fetched_at_utc", "") or ""),
            "actions": actions,
        }
        self.position_execution_path.parent.mkdir(parents=True, exist_ok=True)
        self.position_execution_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Position Execution Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            "- Status: `STALE_CLOSE_SATISFIED`",
            "- Send requested: `True`",
            "- Sent: `False`",
            f"- Broker snapshot UTC: `{result['snapshot_fetched_at_utc']}`",
            "",
            "## Actions",
            "",
        ]
        for action in actions:
            lines.append(
                f"- `{action['trade_id']}` owner=`{action.get('owner')}` management=`{action['management_action']}` "
                "request=`none` sent=`False`"
            )
            for issue in action.get("issues", []):
                lines.append(f"  - `{issue['severity']}` {issue['code']}: {issue['message']}")
        lines.extend(
            [
                "",
                "## Execution Contract",
                "",
                "- Refreshed broker truth wins over stale local receipts before any market close write.",
                "- A close receipt is satisfied without sending when every named trade id is already absent.",
            ]
        )
        self.position_execution_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.position_execution_report_path.write_text("\n".join(lines) + "\n")
        self._record_execution_ledger_receipt(
            kind="position_execution",
            receipt_path=self.position_execution_path,
        )
        return PositionExecutionSummary(
            status="STALE_CLOSE_SATISFIED",
            output_path=self.position_execution_path,
            report_path=self.position_execution_report_path,
            sent=False,
            actions=0,
            blocked=0,
        )

    def _gpt_close_position_decision(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        snapshot,
    ) -> PositionManagementDecision:
        positions_by_id = {
            str(getattr(position, "trade_id", "")): position
            for position in getattr(snapshot, "positions", ()) or ()
        }
        managed: list[ManagedPosition] = []
        for trade_id in gpt_summary.close_trade_ids:
            position = positions_by_id.get(str(trade_id))
            side = getattr(position, "side", "") if position is not None else ""
            owner = getattr(position, "owner", None) if position is not None else None
            managed.append(
                ManagedPosition(
                    trade_id=str(trade_id),
                    pair=str(getattr(position, "pair", "") if position is not None else ""),
                    side=str(getattr(side, "value", side) or ""),
                    units=int(getattr(position, "units", 0) or 0) if position is not None else 0,
                    action=ACTION_REVIEW_EXIT,
                    unrealized_pl_jpy=(
                        float(getattr(position, "unrealized_pl_jpy", 0.0) or 0.0)
                        if position is not None
                        else 0.0
                    ),
                    remaining_risk_jpy=None,
                    remaining_reward_jpy=None,
                    same_direction_score=None,
                    opposite_direction_score=None,
                    recommended_stop_loss=None,
                    recommended_take_profit=None,
                    reasons=(
                        "gpt-close: accepted gpt_trader CLOSE receipt passed Gate A/B; "
                        "execute only through PositionProtectionGateway",
                    ),
                    owner=str(getattr(owner, "value", owner) or "trader"),
                )
            )
        return PositionManagementDecision(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            snapshot_fetched_at_utc=getattr(snapshot, "fetched_at_utc", None).isoformat()
            if getattr(snapshot, "fetched_at_utc", None) is not None
            else None,
            action="GPT_CLOSE",
            positions=tuple(managed),
        )

    def _gpt_brain(self) -> GPTTraderBrain:
        return GPTTraderBrain(
            provider=self.gpt_provider,
            intents_path=self.intents_path,
            campaign_plan_path=self.campaign_plan_path,
            strategy_profile_path=self.strategy_profile_path,
            market_story_profile_path=self.market_story_profile_path,
            market_status_path=self.gpt_market_status_path,
            target_state_path=self.gpt_target_state_path,
            pair_charts_path=self.pair_charts_path,
            attack_advice_path=self.gpt_attack_advice_path,
            learning_audit_path=self.gpt_learning_audit_path,
            self_improvement_audit_path=self.gpt_self_improvement_audit_path,
            verification_ledger_path=self.gpt_verification_ledger_path,
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
            pair_charts_path=self.pair_charts_path,
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
                "raw": snapshot_order_raw(order.raw),
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
    sl_free_active = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    missing_tp_repair_enabled = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    # Under SL-free mode (user directive 「SLいらない」 / 「損失を出さないで稼ぎまくる」),
    # trader-owned SL=None is intentional. TP-less positions are no-broker-TP
    # runners unless repair is explicitly enabled; margin and gateway risk
    # validation remain the executable add gates.
    return all(
        position.owner.value == "trader"
        and (position.take_profit is not None or (sl_free_active and not missing_tp_repair_enabled))
        and (position.stop_loss is not None or sl_free_active)
        for position in trader_positions
    )


def _portfolio_entry_capacity_open(snapshot, target_summary: DailyTargetSummary | None) -> bool:
    cap = _portfolio_entry_capacity_limit(target_summary)
    occupancy = _trader_position_count(snapshot) + _pending_entry_order_count(snapshot)
    return occupancy < cap


def _portfolio_entry_capacity_limit(target_summary: DailyTargetSummary | None) -> int:
    cap = int(RiskPolicy().max_portfolio_positions)
    target_trades = target_summary.target_trades_per_day if target_summary is not None else None
    if target_trades and target_trades > 0:
        cap = max(cap, math.ceil(target_trades / ACTIVE_FX_SESSION_BUCKETS_PER_DAY))
    return cap


def _trader_position_count(snapshot) -> int:
    return sum(1 for position in snapshot.positions if position.owner.value == "trader")


def _trader_only_snapshot(snapshot):
    return replace(
        snapshot,
        positions=tuple(position for position in snapshot.positions if position.owner.value == "trader"),
    )


def _position_management_snapshot(snapshot):
    return replace(
        snapshot,
        positions=tuple(
            position
            for position in snapshot.positions
            if position.owner.value in {"trader", "manual", "unknown"}
        ),
    )
