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

from quant_rabbit.broker.execution import ACTIVE_FX_SESSION_BUCKETS_PER_DAY, LiveOrderGateway, LiveOrderStageSummary
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
    DEFAULT_AI_ATTACK_ADVICE_REPORT,
    DEFAULT_BROKER_INSTRUMENTS,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_CONTEXT_ASSET_CHARTS,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_LEARNING_AUDIT_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MARKET_STATUS,
    DEFAULT_MARKET_STATUS_REPORT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    DEFAULT_NEWS_HEALTH,
    DEFAULT_NEWS_SNAPSHOT,
    DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    DEFAULT_OPTION_SKEW,
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
    DEFAULT_PREDICTIVE_LIMIT_ORDERS,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_PROFITABILITY_ACCEPTANCE,
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
from quant_rabbit.attack_advisor import AttackAdvisor
from quant_rabbit.ai_test_bot import AITestBotBacktester
from quant_rabbit.gpt_trader import DEFAULT_GPT_MAX_LANES, GPTTraderBrain, TraderModelProvider
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.learning_audit import LearningAuditor
from quant_rabbit.predictive_scout import predictive_scout_geometry_claimed
from quant_rabbit.risk import MARGIN_AWARE_BASKET_BUFFER, RiskPolicy, margin_budget_jpy, resolve_max_loss_jpy
from quant_rabbit.snapshot_json import snapshot_order_raw, snapshot_position_raw
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
    PENDING_ENTRY_REPLACE_SPREAD_MULT,
    TraderBrain,
    TraderDecision,
    load_trader_settings,
    _pending_entry_recent_cancel_regret_supports_preservation,
    _pending_entry_recorded_thesis_horizon_active,
)
from quant_rabbit.verification_ledger import VerificationLedger


DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"
DEFAULT_AUTOTRADE_LOCK_DIR = ROOT / ".quant_rabbit_live.lock"
PENDING_ENTRY_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
ACCEPTED_GPT_GATEWAY_ACTIONS = frozenset({"TRADE", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE"})
# WAIT / REQUEST_EVIDENCE never authorize a broker write, but a freshly
# accepted verifier result still has to reach exactly one autotrade cycle so
# existing-position maintenance and the post-cycle sidecars are not skipped.
# Keep that one-shot cycle outcome set separate from gateway permissions.
ACCEPTED_GPT_VERIFIED_CYCLE_ACTIONS = ACCEPTED_GPT_GATEWAY_ACTIONS | frozenset(
    {"WAIT", "REQUEST_EVIDENCE"}
)
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

# A GPT replacement must improve same-lane fill odds by at least one current
# spread before it is allowed to churn an otherwise equivalent pending order.
# One spread is the minimum market cost/noise unit for changing the trigger;
# smaller reprices are not a meaningful executable improvement.
GPT_PENDING_REPLACEMENT_MIN_FILL_IMPROVEMENT_SPREAD_MULT = 1.0


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


def _close_gate_evidence_status(evidence: dict[str, Any]) -> str:
    """Mirror execution_ledger close-gate normalization before broker close."""

    if evidence.get("gate_a_invalidated") is not True:
        return "BLOCK"
    if evidence.get("same_direction_support_conflict"):
        return "BLOCK"
    if evidence.get("hard_timing_gate_required") is True:
        return "BLOCK"
    if (
        evidence.get("explicit_gate_b_required") is True
        and evidence.get("gate_b_explicit_operator_authorized") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("profitability_p0_context_required") is True
        and evidence.get("profitability_p0_context_cited") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("timing_audit_required") is True
        and evidence.get("timing_evidence_cited") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("gate_b_standing_authorized") is not True
        and evidence.get("gate_b_explicit_operator_authorized") is not True
    ):
        return "BLOCK"
    return "PASS"


def _snapshot_refresh_pairs(snapshot: object) -> tuple[str, ...]:
    pairs = set(DEFAULT_TRADER_PAIRS)
    pairs.update(str(pair) for pair in getattr(snapshot, "quotes", {}) or {} if pair)
    for position in getattr(snapshot, "positions", ()) or ():
        pair = str(getattr(position, "pair", "") or "")
        if pair:
            pairs.add(pair)
    return tuple(sorted(pairs))


def _snapshot_quote_freshness(snapshot: object, *, now_utc: datetime | None = None) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    quotes = getattr(snapshot, "quotes", {}) or {}
    if not quotes:
        return {"status": "NO_QUOTES", "fresh": False, "quote_count": 0}

    fallback_ts = getattr(snapshot, "fetched_at_utc", None)
    if isinstance(fallback_ts, datetime):
        if fallback_ts.tzinfo is None:
            fallback_ts = fallback_ts.replace(tzinfo=timezone.utc)
        else:
            fallback_ts = fallback_ts.astimezone(timezone.utc)
    else:
        fallback_ts = None

    max_age: float | None = None
    oldest_pair = ""
    missing_timestamp_pairs: list[str] = []
    for pair, quote in quotes.items():
        quote_ts = getattr(quote, "timestamp_utc", None)
        if isinstance(quote_ts, datetime):
            if quote_ts.tzinfo is None:
                quote_ts = quote_ts.replace(tzinfo=timezone.utc)
            else:
                quote_ts = quote_ts.astimezone(timezone.utc)
        else:
            quote_ts = fallback_ts
        if quote_ts is None:
            missing_timestamp_pairs.append(str(pair))
            continue
        age = max(0.0, (now - quote_ts).total_seconds())
        if max_age is None or age > max_age:
            max_age = age
            oldest_pair = str(pair)

    max_quote_age = float(RiskPolicy().max_quote_age_seconds)
    refresh_threshold = max(1.0, max_quote_age * 0.5)
    if max_age is None:
        return {
            "status": "NO_QUOTE_TIMESTAMPS",
            "fresh": False,
            "quote_count": len(quotes),
            "missing_timestamp_pairs": missing_timestamp_pairs[:8],
            "max_quote_age_contract_seconds": max_quote_age,
            "refresh_threshold_seconds": refresh_threshold,
        }
    fresh = max_age <= refresh_threshold
    return {
        "status": "FRESH" if fresh else "STALE",
        "fresh": fresh,
        "quote_count": len(quotes),
        "oldest_pair": oldest_pair,
        "max_quote_age_seconds": round(max_age, 3),
        "max_quote_age_contract_seconds": max_quote_age,
        "refresh_threshold_seconds": refresh_threshold,
        "missing_timestamp_pairs": missing_timestamp_pairs[:8],
    }


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


def _normalized_pending_order_type(order_type: str | None) -> str:
    text = str(order_type or "").upper().replace("_", "-")
    if text == "STOP":
        return "STOP-ENTRY"
    return text


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _attached_replacement_stop_loss(intent: dict[str, Any]) -> float | None:
    initial_sl_on = _truthy_env("QR_NEW_ENTRY_INITIAL_SL")
    sl_repair_disabled = _truthy_env("QR_TRADER_DISABLE_SL_REPAIR")
    if initial_sl_on or not sl_repair_disabled:
        return _optional_float(intent.get("sl"))
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    return _optional_float(metadata.get("disaster_sl"))


def _price_drift_pips(pair: str, left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0
    pip_factor = 100 if pair.endswith("_JPY") else 10000
    return abs(float(left) - float(right)) * pip_factor


def _raw_dependent_order_price(order: object, key: str) -> float | None:
    raw = getattr(order, "raw", None)
    if not isinstance(raw, dict):
        return None
    nested = raw.get(key)
    if not isinstance(nested, dict):
        return None
    return _optional_float(nested.get("price"))


def _order_owner_value(order: object) -> str:
    owner = getattr(order, "owner", None)
    return str(getattr(owner, "value", owner) or "").lower()


def _order_side_from_units(order: object) -> str | None:
    units = getattr(order, "units", None)
    try:
        numeric_units = int(units)
    except (TypeError, ValueError):
        return None
    if numeric_units > 0:
        return "LONG"
    if numeric_units < 0:
        return "SHORT"
    return None


def _pending_order_lane_parent(order: object) -> str | None:
    raw = getattr(order, "raw", None)
    if not isinstance(raw, dict):
        return None
    comments: list[str] = []
    for key in ("clientExtensions", "tradeClientExtensions"):
        nested = raw.get(key)
        if isinstance(nested, dict) and nested.get("comment"):
            comments.append(str(nested.get("comment")))
    for comment in comments:
        for token in comment.split():
            if token.startswith("lane="):
                return _basket_parent_lane_id(token[len("lane=") :])
    return None


def _selected_replacement_candidates(
    intents_payload: dict[str, Any],
    lane_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    rows = {
        str(item.get("lane_id") or ""): item
        for item in intents_payload.get("results", []) or []
        if isinstance(item, dict)
    }
    candidates: list[dict[str, Any]] = []
    for lane_id in dict.fromkeys(lane_id for lane_id in lane_ids if lane_id):
        item = rows.get(lane_id)
        if not isinstance(item, dict):
            continue
        intent = item.get("intent")
        if not isinstance(intent, dict):
            continue
        risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
        candidates.append(
            {
                "lane_id": lane_id,
                "parent_lane_id": _basket_parent_lane_id(lane_id),
                "pair": str(intent.get("pair") or ""),
                "side": str(intent.get("side") or "").upper(),
                "order_type": _normalized_pending_order_type(str(intent.get("order_type") or "")),
                "entry": _optional_float(intent.get("entry")),
                "tp": _optional_float(intent.get("tp")),
                "sl": _optional_float(intent.get("sl")),
                "attached_sl": _attached_replacement_stop_loss(intent),
                "spread_pips": _optional_float(risk_metrics.get("spread_pips")),
            }
        )
    return tuple(candidates)


def _pending_entry_invalidated_by_stop(order: object, snapshot: object) -> bool:
    pair = str(getattr(order, "pair", "") or "")
    quote = (getattr(snapshot, "quotes", {}) or {}).get(pair)
    side = _order_side_from_units(order)
    stop_loss = _raw_dependent_order_price(order, "stopLossOnFill")
    if quote is None or side is None or stop_loss is None:
        return False
    bid = _optional_float(getattr(quote, "bid", None))
    ask = _optional_float(getattr(quote, "ask", None))
    if side == "LONG" and bid is not None:
        return bid <= stop_loss
    if side == "SHORT" and ask is not None:
        return ask >= stop_loss
    return False


def _distance_to_fill_pips(
    *,
    pair: str,
    side: str,
    order_type: str,
    entry: float | None,
    snapshot: object,
) -> float | None:
    if entry is None:
        return None
    quote = (getattr(snapshot, "quotes", {}) or {}).get(pair)
    if quote is None:
        return None
    bid = _optional_float(getattr(quote, "bid", None))
    ask = _optional_float(getattr(quote, "ask", None))
    pip_factor = 100 if pair.endswith("_JPY") else 10000
    distance: float | None = None
    if order_type == "LIMIT":
        if side == "LONG" and ask is not None:
            distance = ask - entry
        elif side == "SHORT" and bid is not None:
            distance = entry - bid
    elif order_type == "STOP-ENTRY":
        if side == "LONG" and ask is not None:
            distance = entry - ask
        elif side == "SHORT" and bid is not None:
            distance = bid - entry
    if distance is None:
        return None
    return max(0.0, distance * pip_factor)


def _replacement_materially_improves_fill(
    *,
    order: object,
    candidate: dict[str, Any],
    snapshot: object,
) -> bool:
    spread_pips = _optional_float(candidate.get("spread_pips"))
    if spread_pips is None or spread_pips <= 0:
        return False
    pair = str(candidate.get("pair") or "")
    side = str(candidate.get("side") or "")
    order_type = str(candidate.get("order_type") or "")
    current_distance = _distance_to_fill_pips(
        pair=pair,
        side=side,
        order_type=order_type,
        entry=_optional_float(getattr(order, "price", None)),
        snapshot=snapshot,
    )
    replacement_distance = _distance_to_fill_pips(
        pair=pair,
        side=side,
        order_type=order_type,
        entry=_optional_float(candidate.get("entry")),
        snapshot=snapshot,
    )
    if current_distance is None or replacement_distance is None:
        return False
    improvement = current_distance - replacement_distance
    return improvement >= spread_pips * GPT_PENDING_REPLACEMENT_MIN_FILL_IMPROVEMENT_SPREAD_MULT


def _geometry_reward_risk(
    *,
    side: str,
    entry: float | None,
    tp: float | None,
    sl: float | None,
) -> float | None:
    if entry is None or tp is None or sl is None:
        return None
    if side == "LONG":
        reward = tp - entry
        risk = entry - sl
    elif side == "SHORT":
        reward = entry - tp
        risk = sl - entry
    else:
        return None
    if reward <= 0 or risk <= 0:
        return None
    return reward / risk


def _replacement_degrades_reward_risk(*, order: object, candidate: dict[str, Any]) -> bool:
    side = str(candidate.get("side") or "").upper()
    current_rr = _geometry_reward_risk(
        side=side,
        entry=_optional_float(getattr(order, "price", None)),
        tp=_raw_dependent_order_price(order, "takeProfitOnFill"),
        sl=_raw_dependent_order_price(order, "stopLossOnFill"),
    )
    replacement_rr = _geometry_reward_risk(
        side=side,
        entry=_optional_float(candidate.get("entry")),
        tp=_optional_float(candidate.get("tp")),
        sl=_optional_float(candidate.get("attached_sl")) or _optional_float(candidate.get("sl")),
    )
    if current_rr is None or replacement_rr is None:
        return False
    return replacement_rr < current_rr


def _replacement_geometry_drift_exceeds_tolerance(
    *,
    order: object,
    candidate: dict[str, Any],
) -> bool:
    spread_pips = _optional_float(candidate.get("spread_pips"))
    pair = str(candidate.get("pair") or "")
    if spread_pips is None or spread_pips <= 0 or not pair:
        return False
    tolerance_pips = spread_pips * PENDING_ENTRY_REPLACE_SPREAD_MULT
    price_pairs = (
        (_optional_float(getattr(order, "price", None)), _optional_float(candidate.get("entry"))),
        (_raw_dependent_order_price(order, "takeProfitOnFill"), _optional_float(candidate.get("tp"))),
        (_raw_dependent_order_price(order, "stopLossOnFill"), _optional_float(candidate.get("attached_sl"))),
    )
    return any(_price_drift_pips(pair, left, right) > tolerance_pips for left, right in price_pairs)


def _should_preserve_gpt_trade_cancel(
    *,
    order: object,
    candidates: tuple[dict[str, Any], ...],
    snapshot: object,
) -> bool:
    if getattr(order, "trade_id", None):
        return False
    if _normalized_pending_order_type(str(getattr(order, "order_type", "") or "")) not in {
        "LIMIT",
        "STOP-ENTRY",
        "MARKET-IF-TOUCHED",
        "MARKET-IF-TOUCHED-ORDER",
    }:
        return False
    if _order_owner_value(order) != "trader":
        return False
    parent_lane_id = _pending_order_lane_parent(order)
    if parent_lane_id is None:
        return False
    pair = str(getattr(order, "pair", "") or "")
    side = _order_side_from_units(order)
    order_type = _normalized_pending_order_type(str(getattr(order, "order_type", "") or ""))
    if not pair or side is None:
        return False
    if _pending_entry_invalidated_by_stop(order, snapshot):
        return False
    for candidate in candidates:
        if candidate.get("parent_lane_id") != parent_lane_id:
            continue
        if candidate.get("pair") != pair or candidate.get("side") != side:
            continue
        if candidate.get("order_type") != order_type:
            continue
        if _replacement_materially_improves_fill(order=order, candidate=candidate, snapshot=snapshot):
            if _replacement_degrades_reward_risk(order=order, candidate=candidate):
                return True
            return False
        if _replacement_geometry_drift_exceeds_tolerance(order=order, candidate=candidate):
            return False
        return True
    return False


def _filtered_gpt_trade_cancel_order_ids(
    *,
    client: object,
    intents_path: Path,
    lane_ids: tuple[str, ...],
    cancel_order_ids: tuple[str, ...],
    self_improvement_audit_path: Path | None = None,
) -> tuple[str, ...]:
    if not cancel_order_ids:
        return ()
    force_cancel_order_ids = _self_improvement_pending_cancel_review_order_ids(
        self_improvement_audit_path or (intents_path.parent / DEFAULT_SELF_IMPROVEMENT_AUDIT.name)
    )
    try:
        intents_payload = json.loads(intents_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return cancel_order_ids
    candidates = _selected_replacement_candidates(intents_payload, lane_ids)
    if not candidates:
        return cancel_order_ids
    pairs = tuple(sorted({str(candidate.get("pair") or "") for candidate in candidates if candidate.get("pair")}))
    try:
        snapshot = client.snapshot(pairs)
    except (RuntimeError, ValueError, OSError, sqlite3.Error, json.JSONDecodeError):
        return cancel_order_ids
    orders_by_id = {
        str(getattr(order, "order_id", "") or ""): order
        for order in getattr(snapshot, "orders", ()) or ()
    }
    filtered: list[str] = []
    for order_id in cancel_order_ids:
        if str(order_id) in force_cancel_order_ids:
            filtered.append(str(order_id))
            continue
        order = orders_by_id.get(str(order_id))
        if order is not None:
            if _should_preserve_gpt_trade_cancel(
                order=order,
                candidates=candidates,
                snapshot=snapshot,
            ):
                continue
            if (
                _pending_entry_recent_cancel_regret_supports_preservation(order, None, intents_path.parent)
                and not _gpt_cancel_has_material_same_parent_replacement(
                    order=order,
                    candidates=candidates,
                    snapshot=snapshot,
                )
            ):
                continue
        filtered.append(str(order_id))
    return tuple(filtered)


def _lane_ids_excluding_preserved_pending_parents(
    *,
    snapshot: object,
    lane_ids: tuple[str, ...],
    preserved_order_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if not lane_ids or not preserved_order_ids:
        return lane_ids
    preserved_ids = {str(order_id) for order_id in preserved_order_ids if str(order_id)}
    if not preserved_ids:
        return lane_ids
    preserved_parents: set[str] = set()
    for order in getattr(snapshot, "orders", ()) or ():
        order_id = str(getattr(order, "order_id", "") or "")
        if order_id not in preserved_ids:
            continue
        parent = _pending_order_lane_parent(order)
        if parent:
            preserved_parents.add(parent)
    if not preserved_parents:
        return lane_ids
    return tuple(
        lane_id
        for lane_id in lane_ids
        if _basket_parent_lane_id(lane_id) not in preserved_parents
    )


def _predictive_scout_lane_ids(
    intents_path: Path,
    lane_ids: tuple[str, ...],
) -> set[str]:
    """Identify claimed SCOUT lanes from the executable intent artifact."""

    selected = {str(lane_id) for lane_id in lane_ids if str(lane_id)}
    if not selected:
        return set()
    try:
        payload = json.loads(intents_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return set()
    claimed: set[str] = set()
    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        lane_id = str(result.get("lane_id") or "")
        if lane_id not in selected:
            continue
        intent = result.get("intent")
        if not isinstance(intent, dict):
            continue
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        market_context = (
            intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        )
        if predictive_scout_geometry_claimed(
            metadata,
            pair=str(intent.get("pair") or ""),
            side=str(intent.get("side") or ""),
            order_type=str(intent.get("order_type") or ""),
            method=str(market_context.get("method") or ""),
        ):
            claimed.add(lane_id)
    return claimed


def _fixed_predictive_scout_size_plan(
    *,
    intents_path: Path,
    lane_ids: tuple[str, ...],
    size_multiples: dict[str, float],
    selected_lane_id: str | None,
    selected_lane_size_multiple: float | None,
) -> tuple[dict[str, float], float | None]:
    """Lock post-intent multipliers without forcing a fixed unit count.

    Predictive SCOUT units are already derived from current NAV and canonical
    SL risk in intent generation.  A 1.0 multiplier preserves those exact
    verifier-bound units at the final AI-to-gateway boundary.
    """
    claimed = _predictive_scout_lane_ids(intents_path, lane_ids)
    if not claimed:
        return dict(size_multiples), selected_lane_size_multiple
    normalized = {
        lane_id: (1.0 if lane_id in claimed else multiple)
        for lane_id, multiple in size_multiples.items()
    }
    selected_multiple = 1.0 if selected_lane_id in claimed else selected_lane_size_multiple
    return normalized, selected_multiple


def _gpt_cancel_has_material_same_parent_replacement(
    *,
    order: object,
    candidates: tuple[dict[str, Any], ...],
    snapshot: object,
) -> bool:
    parent_lane_id = _pending_order_lane_parent(order)
    if parent_lane_id is None:
        return False
    pair = str(getattr(order, "pair", "") or "")
    side = _order_side_from_units(order)
    order_type = _normalized_pending_order_type(str(getattr(order, "order_type", "") or ""))
    if not pair or side is None:
        return False
    for candidate in candidates:
        if candidate.get("parent_lane_id") != parent_lane_id:
            continue
        if candidate.get("pair") != pair or candidate.get("side") != side:
            continue
        if candidate.get("order_type") != order_type:
            continue
        if _replacement_materially_improves_fill(order=order, candidate=candidate, snapshot=snapshot):
            return True
    return False


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


def _gpt_defaulted_sidecar_path(
    *,
    value: Path,
    gpt_decision_path: Path,
    default_path: Path,
) -> Path:
    repo_data_default = ROOT / "data" / default_path.name
    explicit = None if value in {default_path, repo_data_default} else value
    return _gpt_sidecar_path(
        explicit=explicit,
        gpt_decision_path=gpt_decision_path,
        default_path=default_path,
    )


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


def _cycle_opportunity_mode_report_lines(intents_path: Path) -> list[str]:
    coverage_path = intents_path.with_name("coverage_optimization.json")
    try:
        payload = json.loads(coverage_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    modes = payload.get("opportunity_modes")
    if not isinstance(modes, dict):
        return []
    runner_diag = (
        payload.get("runner_candidate_diagnostics")
        if isinstance(payload.get("runner_candidate_diagnostics"), dict)
        else {}
    )

    mode_parts: list[str] = []
    issue_parts: list[str] = []
    live_blocker_code_parts: list[str] = []
    for mode in ("HARVEST", "RUNNER", "BALANCED"):
        item = modes.get(mode)
        if not isinstance(item, dict):
            continue
        fields = [
            f"{mode} lanes={_report_value(item.get('lanes'))}",
            f"live_ready={_report_value(item.get('live_ready_lanes'))}",
        ]
        if mode == "RUNNER":
            fields.extend(
                [
                    f"diagnostic_candidates={_report_value(item.get('diagnostic_candidate_lanes'))}",
                    f"demoted_to_harvest={_report_value(item.get('demoted_to_harvest_lanes'))}",
                    f"runner_qualified={_report_value(item.get('runner_qualified_lanes'))}",
                ]
            )
        if item.get("reward_jpy") is not None:
            fields.append(f"reward_jpy={_report_value(item.get('reward_jpy'))}")
        mode_parts.append(" ".join(fields))

        top_codes = _report_count_items(item.get("top_issue_codes"), key="code", limit=4)
        if not top_codes and mode == "RUNNER":
            top_codes = _report_count_items(runner_diag.get("top_issue_codes"), key="code", limit=4)
        if top_codes:
            issue_parts.append(f"{mode}=`{top_codes}`")
        top_live_blocker_codes = _report_count_items(item.get("top_live_blocker_codes"), key="code", limit=4)
        if not top_live_blocker_codes and mode == "RUNNER":
            top_live_blocker_codes = _report_count_items(
                runner_diag.get("top_live_blocker_codes"),
                key="code",
                limit=4,
            )
        if top_live_blocker_codes:
            live_blocker_code_parts.append(f"{mode}=`{top_live_blocker_codes}`")

    lines: list[str] = []
    if mode_parts:
        lines.append(f"- Opportunity modes: {'; '.join(f'`{part}`' for part in mode_parts)}")
    if live_blocker_code_parts:
        lines.append(f"- Opportunity live blocker codes: {'; '.join(live_blocker_code_parts)}")
    if issue_parts:
        lines.append(f"- Opportunity issue codes: {'; '.join(issue_parts)}")

    demotions = _report_count_items(runner_diag.get("top_demotion_reasons"), key="reason", limit=4)
    if demotions:
        lines.append(f"- Runner demotions: `{demotions}`")
    perspective = payload.get("perspective_alignment_diagnostics")
    if isinstance(perspective, dict):
        perspective_parts = _cycle_perspective_alignment_parts(perspective)
        if perspective_parts:
            lines.append(f"- Perspective alignment: {'; '.join(perspective_parts)}")
    return lines


def _cycle_perspective_alignment_parts(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("range_forecast_method_mismatch_top")
    if not isinstance(rows, list):
        rows = []
    parts: list[str] = []
    status = str(payload.get("status") or "").strip()
    mismatch_lanes = payload.get("range_forecast_method_mismatch_lanes")
    if status or mismatch_lanes is not None:
        parts.append(
            f"status={_report_value(status or 'UNKNOWN')} "
            f"range_mismatch_lanes={_report_value(mismatch_lanes)}"
        )
    top_parts: list[str] = []
    for item in _cycle_perspective_alignment_rows(rows):
        pair = str(item.get("pair") or "").strip()
        direction = str(item.get("direction") or "").strip()
        if not pair or not direction:
            continue
        same_codes = _report_count_items(
            item.get("range_rotation_top_live_blocker_codes"),
            key="code",
            limit=3,
        )
        other_dirs = _report_count_items(
            item.get("range_rotation_other_side_directions"),
            key="code",
            limit=3,
        )
        other_codes = _report_count_items(
            item.get("range_rotation_other_side_top_live_blocker_codes"),
            key="code",
            limit=3,
        )
        text = (
            f"{pair} {direction} mismatch={_report_value(item.get('method_mismatch_lanes'))} "
            f"range_lanes={_report_value(item.get('range_rotation_lanes'))}"
        )
        if same_codes:
            text += f" blockers={same_codes}"
        if other_dirs:
            text += f" other_rail={other_dirs}"
        if other_codes:
            text += f" other_blockers={other_codes}"
        top_parts.append(text)
    if top_parts:
        parts.append("top=`" + " | ".join(top_parts) + "`")
    return parts


def _cycle_perspective_alignment_rows(rows: list[Any]) -> list[dict[str, Any]]:
    typed_rows = [item for item in rows if isinstance(item, dict)]
    selected = typed_rows[:3]
    if any(_has_other_side_rotation(item) for item in selected):
        return selected
    for item in typed_rows[3:]:
        if _has_other_side_rotation(item):
            return [*selected, item]
    return selected


def _has_other_side_rotation(item: dict[str, Any]) -> bool:
    try:
        return int(item.get("range_rotation_other_side_lanes") or 0) > 0
    except (TypeError, ValueError):
        return False


def _report_count_items(items: object, *, key: str, limit: int) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        label = str(item.get(key) or "").strip()
        if not label:
            continue
        parts.append(f"{label}:{_report_value(item.get('count'))}")
    return ", ".join(parts)


def _report_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


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
        trader_settings_path: Path | None = None,
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
        gpt_projection_ledger_path: Path | None = None,
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
        self.trader_settings_path = trader_settings_path or DEFAULT_TRADER_SETTINGS
        self.receipt_promotion_report_path = receipt_promotion_report_path
        self.target_state_path = target_state_path
        self.target_report_path = target_report_path
        self.use_gpt_trader = use_gpt_trader
        self.gpt_provider = gpt_provider
        self.gpt_decision_path = gpt_decision_path
        self.gpt_decision_report_path = gpt_decision_report_path
        self.gpt_target_state_path = _gpt_defaulted_sidecar_path(
            value=gpt_target_state_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_DAILY_TARGET_STATE,
        )
        self.gpt_attack_advice_path = _gpt_defaulted_sidecar_path(
            value=gpt_attack_advice_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_AI_ATTACK_ADVICE,
        )
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
        self.gpt_projection_ledger_path = _gpt_sidecar_path(
            explicit=gpt_projection_ledger_path,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_PROJECTION_LEDGER,
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
        self.gpt_context_asset_charts_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_CONTEXT_ASSET_CHARTS,
        )
        self.gpt_broker_instruments_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_BROKER_INSTRUMENTS,
        )
        self.gpt_cross_asset_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_CROSS_ASSET_SNAPSHOT,
        )
        self.gpt_flow_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_FLOW_SNAPSHOT,
        )
        self.gpt_currency_strength_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_CURRENCY_STRENGTH,
        )
        self.gpt_levels_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_LEVELS_SNAPSHOT,
        )
        self.gpt_market_context_matrix_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_MARKET_CONTEXT_MATRIX,
        )
        self.gpt_calendar_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_CALENDAR_SNAPSHOT,
        )
        self.gpt_cot_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_COT_SNAPSHOT,
        )
        self.gpt_option_skew_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_OPTION_SKEW,
        )
        self.gpt_capture_economics_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_CAPTURE_ECONOMICS,
        )
        self.gpt_profitability_acceptance_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_PROFITABILITY_ACCEPTANCE,
        )
        self.gpt_execution_timing_audit_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_EXECUTION_TIMING_AUDIT,
        )
        self.gpt_coverage_optimization_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_COVERAGE_OPTIMIZATION,
        )
        self.gpt_operator_precedent_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_OPERATOR_PRECEDENT_AUDIT,
        )
        self.gpt_manual_market_context_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
        )
        self.gpt_predictive_limits_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_PREDICTIVE_LIMIT_ORDERS,
        )
        self.gpt_news_items_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_NEWS_SNAPSHOT,
        )
        self.gpt_news_health_path = _gpt_sidecar_path(
            explicit=None,
            gpt_decision_path=gpt_decision_path,
            default_path=DEFAULT_NEWS_HEALTH,
        )
        self.gpt_ai_backtest_path = _attack_sidecar_path(
            explicit=gpt_ai_backtest_path,
            attack_advice_path=self.gpt_attack_advice_path,
            default_path=DEFAULT_AI_TEST_BOT_BACKTEST,
        )
        self.gpt_outcome_mart_path = _attack_sidecar_path(
            explicit=gpt_outcome_mart_path,
            attack_advice_path=self.gpt_attack_advice_path,
            default_path=DEFAULT_OUTCOME_MART,
        )
        self.gpt_post_trade_learning_path = _attack_sidecar_path(
            explicit=gpt_post_trade_learning_path,
            attack_advice_path=self.gpt_attack_advice_path,
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
        self._pre_intent_snapshot_refresh_summary: dict[str, Any] | None = None
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
                self._record_execution_ledger_receipts()
            except Exception:
                pass
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
            ("gpt_decision", self.gpt_decision_path),
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
        resolved_max_loss_jpy = self._resolve_max_loss_jpy(snapshot)
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
            refreshed_snapshot = self._refresh_snapshot_before_intent_pricing_if_required(snapshot, pairs)
            if refreshed_snapshot is not snapshot:
                snapshot = refreshed_snapshot
                target_summary = self._update_target_state(snapshot) or target_summary
                positions = len(snapshot.positions)
                trader_positions = _trader_position_count(snapshot)
                orders = len(snapshot.orders)
                pending_entries = _pending_entry_order_count(snapshot)
                resolved_max_loss_jpy = self._resolve_max_loss_jpy(snapshot)
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
                    gpt_lane_ids = (
                        gpt_summary.selected_lane_ids
                        or ((gpt_summary.selected_lane_id,) if gpt_summary.selected_lane_id else ())
                    )
                    if (
                        gpt_summary.status == "ACCEPTED"
                        and gpt_summary.allowed
                        and gpt_summary.action == "CANCEL_PENDING"
                    ):
                        force_cancel_ids = _self_improvement_pending_cancel_review_order_ids(
                            self.gpt_self_improvement_audit_path,
                        )
                        allowed_cancel_ids = tuple(
                            dict.fromkeys(
                                (
                                    *decision.pending_cancel_order_ids,
                                    *(
                                        order_id
                                        for order_id in gpt_summary.cancel_order_ids
                                        if order_id in force_cancel_ids
                                    ),
                                )
                            )
                        )
                        canceled_orders.extend(
                            self._cancel_gpt_pending_orders(
                                gpt_summary,
                                send=send,
                                already_canceled=tuple(canceled_orders),
                                allowed_order_ids=allowed_cancel_ids,
                            )
                        )
                        status = "CANCELED_GPT_PENDING" if canceled_orders else "GPT_CANCEL_PENDING"
                        if (
                            not canceled_orders
                            and gpt_summary.cancel_order_ids
                            and not allowed_cancel_ids
                        ):
                            status = "PENDING_PRESERVED_GPT_CANCEL_NOT_CONTAMINATED"
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
                    if (
                        gpt_summary.status == "ACCEPTED"
                        and gpt_summary.allowed
                        and gpt_summary.action == "TRADE"
                    ):
                        try:
                            intents_payload = json.loads(self.intents_path.read_text())
                        except (OSError, json.JSONDecodeError, ValueError):
                            intents_payload = {}
                        if not isinstance(intents_payload, dict):
                            intents_payload = {}
                        live_ready_lane_ids = {
                            lane_id
                            for item in intents_payload.get("results", []) or []
                            if isinstance(item, dict)
                            for lane_id in (str(item.get("lane_id") or ""),)
                            if lane_id and item.get("status") == "LIVE_READY"
                        }
                        unique_gpt_lane_ids = tuple(dict.fromkeys(gpt_lane_ids))
                        current_gpt_lane_ids = tuple(
                            lane_id for lane_id in unique_gpt_lane_ids if lane_id in live_ready_lane_ids
                        )
                        gpt_lanes_allowed, _ = _gpt_lanes_pass_prefilter_or_recovery(
                            intents_payload=intents_payload,
                            gpt_lane_ids=unique_gpt_lane_ids,
                            prefiltered_lane_ids=live_ready_lane_ids,
                        )
                        if (
                            not unique_gpt_lane_ids
                            or not gpt_lanes_allowed
                            or current_gpt_lane_ids != unique_gpt_lane_ids
                        ):
                            summary = AutoTradeCycleSummary(
                                status="GPT_DECISION_NOT_PREFILTERED" if unique_gpt_lane_ids else "GPT_TRADE",
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                deterministic_lane_id=None,
                                sent=False,
                                positions=positions,
                                orders=orders,
                                live_ready=intent_summary.live_ready,
                                selected_lane_ids=unique_gpt_lane_ids,
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
                        basket_lane_ids = current_gpt_lane_ids
                        basket_size_multiples = {}
                        for lane_id in basket_lane_ids:
                            _, size_multiple = self._selected_lane_meta(decision=decision, lane_id=lane_id)
                            basket_size_multiples[lane_id] = size_multiple if size_multiple is not None else 1.0
                        order_gateway = LiveOrderGateway(
                            client=self.client,
                            strategy_profile=self.strategy_profile_path,
                            output_path=self.live_order_output_path,
                            report_path=self.live_order_report_path,
                            live_enabled=self.live_enabled,
                            max_loss_jpy=resolved_max_loss_jpy,
                            portfolio_loss_cap_jpy=self._portfolio_loss_cap_jpy_from_target_state(),
                            self_improvement_audit=self.gateway_self_improvement_audit_path,
                            verified_decision_path=self.gpt_decision_path,
                            execution_ledger_db_path=self.execution_ledger_db_path,
                            execution_ledger_report_path=self.execution_ledger_report_path,
                        )
                        order_summary, deferred_canceled = self._run_order_batch_with_deferred_gpt_trade_cancels(
                            order_gateway=order_gateway,
                            intents_path=self.intents_path,
                            lane_ids=basket_lane_ids,
                            size_multiples=basket_size_multiples,
                            send=send,
                            gpt_summary=gpt_summary,
                            already_canceled=tuple(canceled_orders),
                        )
                        canceled_orders.extend(deferred_canceled)
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
                            deterministic_lane_id=None,
                            sent=order_summary.sent,
                            sent_count=order_summary.sent_count,
                            positions=positions,
                            orders=orders,
                            live_ready=intent_summary.live_ready,
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
                        basket_lane_ids, basket_size_multiples = self._expanded_gpt_basket_plan(
                            decision=decision,
                            gpt_lane_ids=gpt_lane_ids,
                            allow_existing_pending=True,
                            margin_room_jpy=_basket_margin_room_jpy(snapshot),
                        )
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
                        execution_ledger_db_path=self.execution_ledger_db_path,
                        execution_ledger_report_path=self.execution_ledger_report_path,
                    )
                    order_summary, deferred_canceled = self._run_order_batch_with_deferred_gpt_trade_cancels(
                        order_gateway=order_gateway,
                        intents_path=self.intents_path,
                        lane_ids=basket_lane_ids,
                        size_multiples=basket_size_multiples,
                        send=send,
                        gpt_summary=gpt_summary,
                        already_canceled=tuple(canceled_orders),
                    )
                    canceled_orders.extend(deferred_canceled)
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
                visible_thesis_ids = _pending_cancel_ids_with_visible_current_thesis(
                    snapshot,
                    intents_path=self.intents_path,
                    cancel_order_ids=decision.pending_cancel_order_ids,
                )
                if visible_thesis_ids:
                    status = "PENDING_PRESERVED_CURRENT_THESIS"
                else:
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
            refreshed_snapshot = self._refresh_snapshot_before_intent_pricing_if_required(snapshot, pairs)
            if refreshed_snapshot is not snapshot:
                snapshot = refreshed_snapshot
                target_summary = self._update_target_state(snapshot) or target_summary
                positions = len(snapshot.positions)
                trader_positions = _trader_position_count(snapshot)
                orders = len(snapshot.orders)
                pending_entries = _pending_entry_order_count(snapshot)
                resolved_max_loss_jpy = self._resolve_max_loss_jpy(snapshot)
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
                            refreshed_snapshot = self._refresh_snapshot_before_intent_pricing_if_required(snapshot, pairs)
                            if refreshed_snapshot is not snapshot:
                                snapshot = refreshed_snapshot
                                target_summary = self._update_target_state(snapshot) or target_summary
                                positions = len(snapshot.positions)
                                orders = len(snapshot.orders)
                                resolved_max_loss_jpy = self._resolve_max_loss_jpy(snapshot)
                            intent_summary = self._intent_generator(max_loss_jpy=resolved_max_loss_jpy).run(
                                snapshot_path=self.snapshot_path,
                                max_candidates=12,
                            )
                            decision = self._brain().run(snapshot)
                            deterministic_lane_id = (
                                decision.selected_lane_id if decision.action == ACTION_SEND_ENTRY else None
                            )
                            if deterministic_lane_id:
                                gpt_recovery_source = (
                                    f"DETERMINISTIC_WAIT_RECOVERY_BLOCKED_ATTEMPT_{attempt}"
                                )
                                break
                            if _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary):
                                gpt_recovery_source = (
                                    gpt_recovery_source
                                    or _gpt_campaign_recovery_block_source(gpt_summary)
                                )
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

            if selected_lane_id is None and campaign_exposure_required and intent_summary.live_ready > 0:
                if _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary):
                    gpt_recovery_source = (
                        gpt_recovery_source
                        or _gpt_campaign_recovery_block_source(gpt_summary)
                    )
                else:
                    recovery_lane_id = self._campaign_recovery_lane(
                        decision=decision,
                        deterministic_lane_id=deterministic_lane_id,
                    )
                    if recovery_lane_id:
                        gpt_recovery_source = (
                            gpt_recovery_source
                            or f"CAMPAIGN_EXPOSURE_RECOVERY_GPT_{gpt_summary.status}_{gpt_summary.action or 'NO_TRADE'}"
                        )
                        if not _learning_audit_blocks_recovery_lane(self.gpt_learning_audit_path, recovery_lane_id):
                            selected_lane_id = recovery_lane_id
                            selected_lane_score, selected_lane_size_multiple = self._selected_lane_meta(
                                decision=decision,
                                lane_id=selected_lane_id,
                            )

            if selected_lane_id is None:
                learning_recovery_lane_id = None
                if campaign_exposure_required:
                    learning_recovery_lane_id = self._campaign_recovery_lane(
                        decision=decision,
                        deterministic_lane_id=deterministic_lane_id,
                    )
                if intent_summary.live_ready == 0 and gpt_summary.status == "STALE_DECISION":
                    status = "NO_LIVE_READY_INTENT"
                elif (
                    intent_summary.live_ready == 0
                    and gpt_summary.status == "ACCEPTED"
                    and gpt_summary.action in {"WAIT", "REQUEST_EVIDENCE"}
                ):
                    status = f"GPT_{gpt_summary.action}"
                elif campaign_exposure_required and _learning_audit_blocks_recovery_lane(
                    self.gpt_learning_audit_path,
                    learning_recovery_lane_id or deterministic_lane_id,
                ):
                    status = "LEARNING_AUDIT_BLOCKED"
                    deterministic_lane_id = learning_recovery_lane_id or deterministic_lane_id
                elif _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary):
                    status = _gpt_campaign_recovery_block_status(gpt_summary)
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
                if (
                    gpt_summary.status == "ACCEPTED"
                    and gpt_summary.allowed
                    and gpt_summary.action == "CANCEL_PENDING"
                ):
                    canceled_pending = self._cancel_gpt_pending_orders(
                        gpt_summary,
                        send=send,
                    )
                    summary = AutoTradeCycleSummary(
                        status="CANCELED_GPT_PENDING" if canceled_pending else "GPT_CANCEL_PENDING",
                        report_path=self.report_path,
                        snapshot_path=self.snapshot_path,
                        intents_path=self.intents_path,
                        selected_lane_id=None,
                        selected_lane_ids=(),
                        deterministic_lane_id=deterministic_lane_id,
                        sent=False,
                        sent_count=0,
                        positions=positions,
                        orders=orders,
                        live_ready=intent_summary.live_ready,
                        decision_source="gpt_trader",
                        canceled_orders=tuple(canceled_pending),
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
                        recovery_lane_id = selected_lane_id or deterministic_lane_id
                        if _learning_audit_blocks_recovery_lane(self.gpt_learning_audit_path, recovery_lane_id):
                            summary = AutoTradeCycleSummary(
                                status="LEARNING_AUDIT_BLOCKED",
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                selected_lane_ids=(),
                                selected_lane_score=None,
                                selected_lane_size_multiple=None,
                                deterministic_lane_id=deterministic_lane_id,
                                sent=False,
                                sent_count=0,
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
                        if _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary):
                            gpt_recovery_source = _gpt_campaign_recovery_block_source(gpt_summary)
                            summary = AutoTradeCycleSummary(
                                status=_gpt_campaign_recovery_block_status(gpt_summary),
                                report_path=self.report_path,
                                snapshot_path=self.snapshot_path,
                                intents_path=self.intents_path,
                                selected_lane_id=None,
                                selected_lane_ids=(),
                                selected_lane_score=None,
                                selected_lane_size_multiple=None,
                                deterministic_lane_id=deterministic_lane_id,
                                sent=False,
                                sent_count=0,
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

        # Reassert the verifier-bound SCOUT units at the final AI-trader →
        # gateway boundary. This also protects cycles that consume an
        # external TraderDecision created before TraderBrain learned the
        # post-intent multiplier invariant.
        basket_size_multiples, selected_lane_size_multiple = (
            _fixed_predictive_scout_size_plan(
                intents_path=self.intents_path,
                lane_ids=basket_lane_ids,
                size_multiples=basket_size_multiples,
                selected_lane_id=selected_lane_id,
                selected_lane_size_multiple=selected_lane_size_multiple,
            )
        )

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
            execution_ledger_db_path=self.execution_ledger_db_path,
            execution_ledger_report_path=self.execution_ledger_report_path,
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
            (self.gpt_attack_advice_path, "ai_attack_advice"),
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
        lines.extend(_cycle_opportunity_mode_report_lines(self.intents_path))
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
        if self._pre_intent_snapshot_refresh_summary is not None:
            refresh = self._pre_intent_snapshot_refresh_summary
            freshness = refresh.get("freshness_before_refresh") or refresh
            lines.append(
                "- Pre-intent snapshot refresh: "
                f"status=`{refresh.get('status')}` "
                f"reason=`{refresh.get('reason') or 'none'}` "
                f"oldest_pair=`{freshness.get('oldest_pair') or 'none'}` "
                f"max_quote_age_seconds=`{freshness.get('max_quote_age_seconds')}`"
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
            levels_path=data_root / DEFAULT_LEVELS_SNAPSHOT.name,
            market_context_matrix_path=data_root / DEFAULT_MARKET_CONTEXT_MATRIX.name,
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

    def _refresh_snapshot_before_intent_pricing_if_required(self, snapshot, pairs: tuple[str, ...]):
        freshness = _snapshot_quote_freshness(snapshot)
        if freshness.get("fresh"):
            self._pre_intent_snapshot_refresh_summary = {
                **freshness,
                "status": "SKIPPED",
                "reason": "snapshot_quotes_fresh_for_intent_pricing",
                "snapshot_path": str(self.snapshot_path),
            }
            return snapshot

        refresh_pairs = _snapshot_refresh_pairs(snapshot) or pairs
        refreshed = self._refresh_snapshot(refresh_pairs)
        self._pre_intent_snapshot_refresh_summary = {
            "status": "REFRESHED",
            "reason": "pre_intent_snapshot_stale_after_cycle_preflight",
            "snapshot_path": str(self.snapshot_path),
            "fetched_at_utc": refreshed.fetched_at_utc.isoformat(),
            "positions": len(refreshed.positions),
            "orders": len(refreshed.orders),
            "quotes": len(refreshed.quotes),
            "freshness_before_refresh": freshness,
        }
        return refreshed

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
            target_state_path=self.target_state_path or DEFAULT_DAILY_TARGET_STATE,
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
            self._run_attack_advice_for_gpt_handoff()
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

        `autotrade-cycle --reuse-market-artifacts` is the verifier-to-cycle
        bridge. Once `gpt-trader-decision` has accepted an external response,
        rerunning the verifier against a newer broker snapshot can reject the
        same receipt as stale even though the live gateway will fetch fresh
        broker truth before any staging/sending. Reuse is therefore allowed
        only for the same external receipt, the same order-intent packet for
        TRADE receipts, and a still-present LIVE_READY lane for any selected
        fresh-entry lane. CLOSE / CANCEL_PENDING / protection actions do not
        select order-intent lanes. WAIT / REQUEST_EVIDENCE are not gateway
        permissions at all; they are reusable only as one-shot verified cycle
        outcomes so position maintenance still runs. Every action still needs
        the same source receipt and an unconsumed accepted verifier output.
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
        if action not in ACCEPTED_GPT_VERIFIED_CYCLE_ACTIONS:
            return None
        if not self._verified_gpt_decision_matches_source(decision, source_payload):
            return None
        if action == "TRADE" and not self._verified_gpt_entry_artifacts_still_match(verified_payload, decision):
            return None
        if action in {"WAIT", "REQUEST_EVIDENCE"} and not self._verified_gpt_non_entry_artifacts_still_current(
            source_path
        ):
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

    def _verified_gpt_entry_artifacts_still_match(
        self,
        verified_payload: dict[str, Any],
        decision: dict[str, Any],
    ) -> bool:
        if not self._verified_gpt_attack_advice_still_matches(verified_payload):
            return False
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

    def _verified_gpt_non_entry_artifacts_still_current(self, source_path: Path) -> bool:
        """Do not reuse an old non-entry conclusion after market state changed.

        The shell wrapper performs the same freshness check before composing a
        handoff, but direct CLI callers still need a Python-side fail-closed
        guard. A newer snapshot, intent packet, or attack-advice packet means
        the accepted WAIT / REQUEST_EVIDENCE conclusion belongs to old market
        truth and must fall through to the normal stale-receipt explanation.
        """

        try:
            source_mtime_ns = source_path.stat().st_mtime_ns
            return not any(
                path.exists() and path.stat().st_mtime_ns > source_mtime_ns
                for path in (self.snapshot_path, self.intents_path, self.gpt_attack_advice_path)
            )
        except OSError:
            return False

    def _verified_gpt_attack_advice_still_matches(self, verified_payload: dict[str, Any]) -> bool:
        if not self.gpt_attack_advice_path.exists():
            return True
        try:
            current_attack_advice = json.loads(self.gpt_attack_advice_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        packet = verified_payload.get("input_packet")
        artifact_timestamps = packet.get("artifact_timestamps") if isinstance(packet, dict) else {}
        verified_attack_ts = (
            artifact_timestamps.get("ai_attack_advice_generated_at_utc")
            if isinstance(artifact_timestamps, dict)
            else None
        )
        current_attack_ts = (
            current_attack_advice.get("generated_at_utc") if isinstance(current_attack_advice, dict) else None
        )
        if current_attack_ts:
            return verified_attack_ts == current_attack_ts
        return not verified_attack_ts

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

    def _run_attack_advice_for_gpt_handoff(self) -> None:
        if self.gpt_attack_advice_path != DEFAULT_AI_ATTACK_ADVICE:
            return
        AttackAdvisor(
            intents_path=self.intents_path,
            target_state_path=self.gpt_target_state_path,
            ai_backtest_path=self.gpt_ai_backtest_path,
            outcome_mart_path=self.gpt_outcome_mart_path,
            coverage_path=DEFAULT_COVERAGE_OPTIMIZATION,
            output_path=self.gpt_attack_advice_path,
            report_path=DEFAULT_AI_ATTACK_ADVICE_REPORT,
        ).run()

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

    def _run_order_batch_with_deferred_gpt_trade_cancels(
        self,
        *,
        order_gateway: LiveOrderGateway,
        intents_path: Path,
        lane_ids: tuple[str, ...],
        size_multiples: dict[str, float],
        send: bool,
        gpt_summary: GptHandoffSummary | None,
        already_canceled: tuple[str, ...] = (),
    ):
        replace_order_ids = (
            gpt_summary.cancel_order_ids
            if gpt_summary is not None and gpt_summary.action == "TRADE"
            else ()
        )
        requested_replace_order_ids = replace_order_ids
        if replace_order_ids:
            replace_order_ids = _filtered_gpt_trade_cancel_order_ids(
                client=order_gateway.client,
                intents_path=intents_path,
                lane_ids=lane_ids,
                cancel_order_ids=replace_order_ids,
                self_improvement_audit_path=self.gpt_self_improvement_audit_path,
            )
            if gpt_summary is not None and replace_order_ids != gpt_summary.cancel_order_ids:
                gpt_summary = replace(gpt_summary, cancel_order_ids=replace_order_ids)
        if not replace_order_ids:
            if requested_replace_order_ids:
                remaining_lane_ids = lane_ids
                try:
                    remaining_lane_ids = _lane_ids_excluding_preserved_pending_parents(
                        snapshot=self._load_snapshot_artifact(),
                        lane_ids=lane_ids,
                        preserved_order_ids=requested_replace_order_ids,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    remaining_lane_ids = lane_ids
                if remaining_lane_ids:
                    remaining_size_multiples = {
                        lane_id: size_multiples[lane_id]
                        for lane_id in remaining_lane_ids
                        if lane_id in size_multiples
                    }
                    return (
                        order_gateway.run_batch(
                            intents_path=intents_path,
                            lane_ids=remaining_lane_ids,
                            size_multiples=remaining_size_multiples,
                            send=send,
                            confirm_live=send,
                        ),
                        (),
                    )
                return (
                    self._write_preserved_pending_entry_no_action(
                        lane_ids=lane_ids,
                        cancel_order_ids=requested_replace_order_ids,
                        send=send,
                    ),
                    (),
                )
            return (
                order_gateway.run_batch(
                    intents_path=intents_path,
                    lane_ids=lane_ids,
                    size_multiples=size_multiples,
                    send=send,
                    confirm_live=send,
                ),
                (),
            )

        if not send or not self.live_enabled:
            return (
                order_gateway.run_batch(
                    intents_path=intents_path,
                    lane_ids=lane_ids,
                    size_multiples=size_multiples,
                    ignore_pending_order_ids=replace_order_ids,
                    send=send,
                    confirm_live=send,
                ),
                (),
            )

        preflight = order_gateway.run_batch(
            intents_path=intents_path,
            lane_ids=lane_ids,
            size_multiples=size_multiples,
            ignore_pending_order_ids=replace_order_ids,
            send=False,
            confirm_live=False,
        )
        if preflight.status != "STAGED":
            return preflight, ()

        canceled = self._cancel_gpt_pending_orders(
            gpt_summary,
            send=send,
            already_canceled=already_canceled,
        )
        sent = order_gateway.run_batch(
            intents_path=intents_path,
            lane_ids=lane_ids,
            size_multiples=size_multiples,
            ignore_pending_order_ids=replace_order_ids,
            send=send,
            confirm_live=send,
        )
        return sent, canceled

    def _write_preserved_pending_entry_no_action(
        self,
        *,
        lane_ids: tuple[str, ...],
        cancel_order_ids: tuple[str, ...],
        send: bool,
    ) -> LiveOrderStageSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        primary_lane_id = lane_ids[0] if lane_ids else None
        result = {
            "generated_at_utc": generated_at,
            "status": "NO_ACTION",
            "lane_id": primary_lane_id,
            "lane_ids": list(lane_ids),
            "requested_units": None,
            "size_multiple": None,
            "scaled_units": None,
            "send_requested": send,
            "sent": False,
            "sent_count": 0,
            "portfolio_position_cap": None,
            "order_request": None,
            "risk_issues": [],
            "strategy_issues": [],
            "cancel_order_ids": list(cancel_order_ids),
            "reason": (
                "preserved equivalent trader-owned pending entry named by GPT cancel_order_ids; "
                "no duplicate fresh entry staged"
            ),
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
                    f"- Lane: `{primary_lane_id}`",
                    f"- Lanes: `{', '.join(lane_ids) if lane_ids else 'none'}`",
                    "- Requested units: `None` size multiple: `None` scaled units:`None`",
                    f"- Send requested: `{send}`",
                    "- Sent: `False`",
                    "- Sent count: `0`",
                    f"- Cancel order ids preserved: `{', '.join(cancel_order_ids)}`",
                    "",
                    "## Order Request",
                    "",
                    "- none",
                    "",
                    "## Issues",
                    "",
                    "- none",
                    "",
                    "## Reason",
                    "",
                    "- Equivalent trader-owned pending entry remains active; duplicate fresh entry was not staged.",
                    "",
                    "## Send Contract",
                    "",
                    "- Existing pending broker truth remains the active entry thesis.",
                    "- A fresh entry is staged only after cancellation is still required by current gateway validation.",
                ]
            )
            + "\n"
        )
        return LiveOrderStageSummary(
            status="NO_ACTION",
            lane_id=primary_lane_id,
            output_path=self.live_order_output_path,
            report_path=self.live_order_report_path,
            sent=False,
            risk_issues=0,
            strategy_issues=0,
            sent_count=0,
            lane_ids=lane_ids,
        )

    def _cancel_gpt_pending_orders(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        send: bool,
        already_canceled: tuple[str, ...] = (),
        allowed_order_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        if not send or not self.live_enabled or not gpt_summary.cancel_order_ids:
            return ()
        canceled: list[str] = []
        already = set(already_canceled)
        allowed = set(allowed_order_ids) if allowed_order_ids is not None else None
        force_cancel_order_ids = _self_improvement_pending_cancel_review_order_ids(
            self.gpt_self_improvement_audit_path
            or (self.intents_path.parent / DEFAULT_SELF_IMPROVEMENT_AUDIT.name)
        )
        preserved_current_thesis_ids: set[str] = set()
        preserved_active_recorded_thesis_ids: set[str] = set()
        if gpt_summary.action == "CANCEL_PENDING" and self.intents_path.exists():
            try:
                snapshot = self._load_snapshot_artifact()
                visible_current_thesis_ids = set(
                    _pending_cancel_ids_with_visible_current_thesis(
                        snapshot,
                        intents_path=self.intents_path,
                        cancel_order_ids=gpt_summary.cancel_order_ids,
                    )
                )
                visible_live_ready_thesis_ids = set(
                    _pending_cancel_ids_with_visible_current_thesis(
                        snapshot,
                        intents_path=self.intents_path,
                        cancel_order_ids=gpt_summary.cancel_order_ids,
                        live_ready_only=True,
                    )
                )
                preserved_current_thesis_ids = (
                    visible_current_thesis_ids - force_cancel_order_ids
                ) | (visible_live_ready_thesis_ids & force_cancel_order_ids)
                preserved_active_recorded_thesis_ids = set(
                    _pending_cancel_ids_with_active_recorded_thesis(
                        snapshot,
                        data_root=self.intents_path.parent,
                        cancel_order_ids=gpt_summary.cancel_order_ids,
                    )
                ) - force_cancel_order_ids
            except (OSError, ValueError, json.JSONDecodeError):
                preserved_current_thesis_ids = set()
                preserved_active_recorded_thesis_ids = set()
        for order_id in gpt_summary.cancel_order_ids:
            if order_id in already:
                continue
            if allowed is not None and order_id not in allowed:
                continue
            if order_id in preserved_current_thesis_ids or order_id in preserved_active_recorded_thesis_ids:
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
        close_gate_issue = self._gpt_close_gate_evidence_issue(gpt_summary)
        if close_gate_issue is not None:
            self._record_execution_ledger_receipt(
                kind="gpt_decision",
                receipt_path=self.gpt_decision_path,
            )
            return self._write_gpt_close_gate_evidence_blocked(
                gpt_summary,
                issue=close_gate_issue,
                snapshot=snapshot,
                send=send,
            )
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

    def _gpt_close_gate_evidence_issue(self, gpt_summary: GptHandoffSummary) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.gpt_decision_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return {
                "severity": "BLOCK",
                "code": "GPT_CLOSE_GATE_EVIDENCE_UNREADABLE",
                "message": f"accepted GPT CLOSE receipt is unreadable before broker close: {exc}",
            }
        close_gate_evidence = (
            payload.get("close_gate_evidence")
            if isinstance(payload.get("close_gate_evidence"), list)
            else []
        )
        if not close_gate_evidence:
            return {
                "severity": "BLOCK",
                "code": "GPT_CLOSE_GATE_EVIDENCE_MISSING",
                "message": (
                    "accepted GPT CLOSE receipt has no close_gate_evidence; "
                    "loss-side market close cannot reach PositionProtectionGateway"
                ),
            }
        evidence_by_trade: dict[str, list[dict[str, Any]]] = {}
        for item in close_gate_evidence:
            if not isinstance(item, dict):
                continue
            trade_id = str(item.get("trade_id") or "").strip()
            if not trade_id:
                continue
            evidence_by_trade.setdefault(trade_id, []).append(item)
        missing: list[str] = []
        blocked: list[str] = []
        for trade_id in gpt_summary.close_trade_ids:
            trade_key = str(trade_id)
            evidence_items = evidence_by_trade.get(trade_key) or []
            if not evidence_items:
                missing.append(trade_key)
                continue
            if not any(_close_gate_evidence_status(item) == "PASS" for item in evidence_items):
                blocked.append(trade_key)
        if missing or blocked:
            details: list[str] = []
            if missing:
                details.append("missing evidence for " + ", ".join(missing))
            if blocked:
                details.append("non-PASS evidence for " + ", ".join(blocked))
            return {
                "severity": "BLOCK",
                "code": "GPT_CLOSE_GATE_EVIDENCE_NOT_PASSING",
                "message": (
                    "accepted GPT CLOSE receipt lacks PASS close_gate_evidence for every named trade; "
                    + "; ".join(details)
                ),
            }
        return None

    def _write_gpt_close_gate_evidence_blocked(
        self,
        gpt_summary: GptHandoffSummary,
        *,
        issue: dict[str, Any],
        snapshot,
        send: bool,
    ) -> PositionExecutionSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        positions_by_id = {
            str(getattr(position, "trade_id", "")): position
            for position in getattr(snapshot, "positions", ()) or ()
        }
        actions: list[dict[str, Any]] = []
        for trade_id in gpt_summary.close_trade_ids:
            position = positions_by_id.get(str(trade_id))
            actions.append(
                {
                    "trade_id": str(trade_id),
                    "pair": str(getattr(position, "pair", "") if position is not None else ""),
                    "owner": str(getattr(getattr(position, "owner", ""), "value", getattr(position, "owner", "")) or ""),
                    "management_action": "GPT_CLOSE",
                    "request": None,
                    "issues": [dict(issue)],
                    "sent": False,
                    "response": None,
                }
            )
        result = {
            "generated_at_utc": generated_at,
            "status": "BLOCKED",
            "send_requested": send,
            "sent": False,
            "snapshot_fetched_at_utc": str(getattr(snapshot, "fetched_at_utc", "") or ""),
            "actions": actions,
        }
        self.position_execution_path.parent.mkdir(parents=True, exist_ok=True)
        self.position_execution_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Position Execution Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            "- Status: `BLOCKED`",
            f"- Send requested: `{send}`",
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
            for item in action.get("issues", []):
                lines.append(f"  - `{item['severity']}` {item['code']}: {item['message']}")
        lines.extend(
            [
                "",
                "## Execution Contract",
                "",
                "- Accepted GPT CLOSE receipts must carry PASS close_gate_evidence for every named trade id.",
                "- Missing or non-passing close-gate evidence blocks the broker close before PositionProtectionGateway.",
            ]
        )
        self.position_execution_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.position_execution_report_path.write_text("\n".join(lines) + "\n")
        self._record_execution_ledger_receipt(
            kind="position_execution",
            receipt_path=self.position_execution_path,
        )
        return PositionExecutionSummary(
            status="BLOCKED",
            output_path=self.position_execution_path,
            report_path=self.position_execution_report_path,
            sent=False,
            actions=len(actions),
            blocked=len(actions),
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
            context_asset_charts_path=self.gpt_context_asset_charts_path,
            broker_instruments_path=self.gpt_broker_instruments_path,
            cross_asset_path=self.gpt_cross_asset_path,
            flow_path=self.gpt_flow_path,
            currency_strength_path=self.gpt_currency_strength_path,
            levels_path=self.gpt_levels_path,
            market_context_matrix_path=self.gpt_market_context_matrix_path,
            calendar_path=self.gpt_calendar_path,
            cot_path=self.gpt_cot_path,
            option_skew_path=self.gpt_option_skew_path,
            attack_advice_path=self.gpt_attack_advice_path,
            capture_economics_path=self.gpt_capture_economics_path,
            profitability_acceptance_path=self.gpt_profitability_acceptance_path,
            execution_timing_audit_path=self.gpt_execution_timing_audit_path,
            coverage_optimization_path=self.gpt_coverage_optimization_path,
            learning_audit_path=self.gpt_learning_audit_path,
            verification_ledger_path=self.gpt_verification_ledger_path,
            self_improvement_audit_path=self.gpt_self_improvement_audit_path,
            projection_ledger_path=self.gpt_projection_ledger_path,
            operator_precedent_path=self.gpt_operator_precedent_path,
            manual_market_context_path=self.gpt_manual_market_context_path,
            predictive_limits_path=self.gpt_predictive_limits_path,
            news_items_path=self.gpt_news_items_path,
            news_health_path=self.gpt_news_health_path,
            output_path=self.gpt_decision_path,
            report_path=self.gpt_decision_report_path,
            max_lanes=self.gpt_max_lanes,
        )

    def _resolve_max_loss_jpy(self, snapshot=None) -> float:
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
            risk_equity_jpy = self._risk_equity_jpy_for_pct(snapshot)
            return resolve_max_loss_jpy(
                max_loss_jpy=explicit_jpy,
                max_loss_pct=explicit_pct,
                equity_jpy=risk_equity_jpy,
                default_max_loss_jpy=None,
                label="autotrade-cycle risk cap",
            )
        ledger_cap = self._daily_risk_budget_jpy_from_target_state()
        if ledger_cap is not None:
            return ledger_cap
        # No CLI override and no ledger — fall through to trader_settings, but
        # never invent a JPY literal. If settings specify pct, use target-state
        # equity when present, otherwise the current broker snapshot NAV/balance.
        # Without either source, resolve_max_loss_jpy raises and the cycle fails
        # closed.
        settings = load_trader_settings(self.trader_settings_path)
        risk_equity_jpy = self._risk_equity_jpy_for_pct(snapshot)
        max_loss_jpy = settings.default_max_loss_jpy
        max_loss_pct = settings.default_max_loss_pct
        return resolve_max_loss_jpy(
            max_loss_jpy=max_loss_jpy,
            max_loss_pct=max_loss_pct,
            equity_jpy=risk_equity_jpy,
            default_max_loss_jpy=None,
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

    def _risk_equity_jpy_for_pct(self, snapshot=None) -> float | None:
        if self.risk_equity_jpy is not None:
            return self.risk_equity_jpy
        value = None
        if self.target_state_path is not None and self.target_state_path.exists():
            try:
                payload = json.loads(self.target_state_path.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                payload = {}
            value = payload.get("current_equity_jpy")
            if value is None:
                value = payload.get("start_balance_jpy")
        if value is None and snapshot is not None:
            account = getattr(snapshot, "account", None)
            value = getattr(account, "nav_jpy", None)
            if value is None:
                value = getattr(account, "balance_jpy", None)
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
                "raw": snapshot_position_raw(pos.raw),
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


def _pending_cancel_ids_with_visible_current_thesis(
    snapshot,
    *,
    intents_path: Path,
    cancel_order_ids: tuple[str, ...],
    live_ready_only: bool = False,
) -> tuple[str, ...]:
    cancel_set = {str(order_id) for order_id in cancel_order_ids if str(order_id)}
    if not cancel_set:
        return ()
    try:
        intents_payload = json.loads(intents_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return ()
    current_pair_sides: set[tuple[str, str]] = set()
    for item in intents_payload.get("results", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("intent"), dict):
            continue
        if live_ready_only and str(item.get("status") or "") != "LIVE_READY":
            continue
        intent = item.get("intent") or {}
        current_pair_sides.add(
            (
                str(intent.get("pair") or ""),
                str(intent.get("side") or "").upper(),
            )
        )
    if not current_pair_sides:
        return ()
    preserved: list[str] = []
    for order in snapshot.orders:
        order_id = str(getattr(order, "order_id", "") or "")
        if order_id not in cancel_set:
            continue
        if getattr(order, "trade_id", None):
            continue
        if str(getattr(order, "order_type", "") or "").upper() not in PENDING_ENTRY_TYPES:
            continue
        if _order_owner_value(order) != "trader":
            continue
        pair_side = (
            str(getattr(order, "pair", "") or ""),
            str(_order_side_from_units(order) or "").upper(),
        )
        if pair_side in current_pair_sides:
            preserved.append(order_id)
    return tuple(preserved)


def _pending_cancel_ids_with_active_recorded_thesis(
    snapshot,
    *,
    data_root: Path | None,
    cancel_order_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if data_root is None:
        return ()
    cancel_set = {str(order_id) for order_id in cancel_order_ids if str(order_id)}
    if not cancel_set:
        return ()
    preserved: list[str] = []
    for order in snapshot.orders:
        order_id = str(getattr(order, "order_id", "") or "")
        if order_id not in cancel_set:
            continue
        if getattr(order, "trade_id", None):
            continue
        if str(getattr(order, "order_type", "") or "").upper() not in PENDING_ENTRY_TYPES:
            continue
        if _order_owner_value(order) != "trader":
            continue
        if _pending_entry_recorded_thesis_horizon_active(order, snapshot, data_root):
            preserved.append(order_id)
    return tuple(preserved)


def _pending_order_has_current_pair_side(order: object, intents_payload: dict[str, Any]) -> bool:
    pair_side = (
        str(getattr(order, "pair", "") or ""),
        str(_order_side_from_units(order) or "").upper(),
    )
    if not pair_side[0] or not pair_side[1]:
        return False
    for item in intents_payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent")
        if not isinstance(intent, dict):
            continue
        candidate = (
            str(intent.get("pair") or ""),
            str(intent.get("side") or "").upper(),
        )
        if candidate == pair_side:
            return True
    return False


def _self_improvement_pending_cancel_review_order_ids(path: Path | None) -> set[str]:
    """Return pending entry order ids already flagged for cancel review."""

    if path is None or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return set()
    findings = payload.get("findings") if isinstance(payload, dict) else []
    if isinstance(findings, dict):
        items = findings.values()
    elif isinstance(findings, list):
        items = findings
    else:
        items = []
    order_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("code") != "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED":
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        for order_id in evidence.get("cancel_review_order_ids") or []:
            text = str(order_id or "").strip()
            if text:
                order_ids.add(text)
    return order_ids


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


def _learning_audit_blocks_recovery_lane(path: Path, lane_id: str | None) -> bool:
    if not lane_id or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if payload.get("status") != "LEARNING_AUDIT_BLOCKED":
        return False
    influence = payload.get("learning_influence")
    lanes = influence.get("lanes") if isinstance(influence, dict) else None
    if not isinstance(lanes, list):
        return False
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if not _same_recovery_lane_family(str(lane.get("lane_id") or ""), lane_id):
            continue
        return bool(lane.get("learning_influences"))
    return False


def _same_recovery_lane_family(a: str, b: str | None) -> bool:
    if not a or not b:
        return False
    return a == b or a.startswith(f"{b}:") or b.startswith(f"{a}:")


def _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary: GptHandoffSummary | None) -> bool:
    """Campaign recovery needs a fresh accepted TRADE/ADD receipt."""

    if gpt_summary is None:
        return False
    status = str(gpt_summary.status or "").upper()
    action = str(gpt_summary.action or "").upper()
    error = str(gpt_summary.error or "").lower()
    if not gpt_summary.allowed:
        return True
    if "fresh receipt" in error or "write a fresh receipt" in error:
        return True
    if "guardian receipt" in error and ("unresolved" in error or "block" in error or "normal_routing_allowed=false" in error):
        return True
    if "normal_routing_allowed=false" in error:
        return True
    if status == "STALE_DECISION":
        if action in {"WAIT", "REQUEST_EVIDENCE"}:
            return True
        if action != "TRADE":
            return True
        return "already verified as rejected trade" in error or not gpt_summary.allowed
    if status != "ACCEPTED":
        return True
    if action in {"WAIT", "REQUEST_EVIDENCE"}:
        return True
    if action not in {"TRADE", "ADD"}:
        return True
    if action in {"TRADE", "ADD"} and not gpt_summary.selected_lane_id and not gpt_summary.selected_lane_ids:
        return True
    return False


def _gpt_trade_rejection_blocks_campaign_recovery(gpt_summary: GptHandoffSummary | None) -> bool:
    return _gpt_fresh_entry_receipt_blocks_campaign_recovery(gpt_summary)


def _gpt_campaign_recovery_block_status(gpt_summary: GptHandoffSummary) -> str:
    status = str(gpt_summary.status or "").upper()
    action = str(gpt_summary.action or "").upper()
    error = str(gpt_summary.error or "").lower()
    if status == "STALE_DECISION" and "already verified as rejected trade" in error:
        return "STALE_GPT_DECISION_REFRESH_REQUIRED"
    if action == "TRADE" and (status != "ACCEPTED" or not gpt_summary.allowed):
        return "GPT_REJECTED"
    if not gpt_summary.allowed:
        return "GPT_FRESH_RECEIPT_REQUIRED_FOR_RECOVERY"
    if status == "STALE_DECISION" and action == "WAIT":
        return "STALE_ACCEPTED_WAIT_BLOCKS_CAMPAIGN_RECOVERY"
    if status == "STALE_DECISION" and action == "REQUEST_EVIDENCE":
        return "STALE_ACCEPTED_REQUEST_EVIDENCE_BLOCKS_CAMPAIGN_RECOVERY"
    if action == "WAIT":
        return "ACCEPTED_WAIT_BLOCKS_CAMPAIGN_RECOVERY"
    if action == "REQUEST_EVIDENCE":
        return "ACCEPTED_REQUEST_EVIDENCE_BLOCKS_CAMPAIGN_RECOVERY"
    if "fresh receipt" in error or "write a fresh receipt" in error:
        return "GPT_FRESH_RECEIPT_REQUIRED_FOR_RECOVERY"
    if "guardian receipt" in error or "normal_routing_allowed=false" in error:
        return "GUARDIAN_RECEIPT_BLOCKS_CAMPAIGN_RECOVERY"
    if action not in {"TRADE", "ADD"}:
        return "GPT_FRESH_RECEIPT_REQUIRED_FOR_RECOVERY"
    return "GPT_REJECTED"


def _gpt_campaign_recovery_block_source(gpt_summary: GptHandoffSummary) -> str:
    status = str(gpt_summary.status or "UNKNOWN").upper()
    action = str(gpt_summary.action or "NO_ACTION").upper()
    error = str(gpt_summary.error or "").lower()
    if status == "STALE_DECISION" and "already verified as rejected trade" in error:
        return "CAMPAIGN_EXPOSURE_BLOCKED_GPT_STALE_REJECTED_TRADE"
    if action == "TRADE" and (status != "ACCEPTED" or not gpt_summary.allowed):
        return f"CAMPAIGN_EXPOSURE_BLOCKED_GPT_{status}_{action}"
    if not gpt_summary.allowed:
        return "CAMPAIGN_EXPOSURE_BLOCKED_GPT_NOT_ALLOWED"
    if status == "STALE_DECISION" and action == "WAIT":
        return "CAMPAIGN_EXPOSURE_BLOCKED_STALE_ACCEPTED_WAIT"
    if status == "STALE_DECISION" and action == "REQUEST_EVIDENCE":
        return "CAMPAIGN_EXPOSURE_BLOCKED_STALE_ACCEPTED_REQUEST_EVIDENCE"
    if action in {"WAIT", "REQUEST_EVIDENCE"}:
        return f"CAMPAIGN_EXPOSURE_BLOCKED_ACCEPTED_{action}"
    if "fresh receipt" in error or "write a fresh receipt" in error:
        return "CAMPAIGN_EXPOSURE_BLOCKED_FRESH_RECEIPT_REQUIRED"
    if "guardian receipt" in error or "normal_routing_allowed=false" in error:
        return "CAMPAIGN_EXPOSURE_BLOCKED_GUARDIAN_RECEIPT"
    if status == "STALE_DECISION":
        return "CAMPAIGN_EXPOSURE_BLOCKED_GPT_STALE_NON_TRADE"
    return f"CAMPAIGN_EXPOSURE_BLOCKED_GPT_{status}_{action}"


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
