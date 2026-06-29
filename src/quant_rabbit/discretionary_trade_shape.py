"""Pair-agnostic discretionary trade-shape evaluation.

The 2025 manual USD_JPY record is historical operator precedent, not a
USD_JPY-only rule. This module extracts the reusable shape into read-only
candidate scoring: theme expression, location, thesis state, position-building
style, SL lint state, and pair overlays. It never grants live permission; the
existing intent, risk, verifier, and gateway contracts remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


LOCATION_LOWER = "LOWER"
LOCATION_MIDDLE = "MIDDLE"
LOCATION_UPPER = "UPPER"
LOCATION_UNKNOWN = "UNKNOWN"

TAPE_STATES = {"TREND", "RANGE", "SQUEEZE", "FADE", "ROTATION"}
ENTRY_SHAPES = {"SCOUT", "PULLBACK", "BREAKOUT", "FADE", "RETEST"}
BUILDING_STYLES = {"SINGLE", "BOUNDED_ADVERSE_ADD", "WITH_MOVE_PYRAMID"}
THESIS_STATES = {"ALIVE", "WOUNDED", "INVALIDATED"}
SL_LINT_STATES = {"PASS", "WARN", "BLOCK"}

# Advisory score weights are ordinal explanation weights, not market-derived
# risk thresholds. They only rank already-generated candidates in the audit.
BASE_SHAPE_SCORE = 50
PRECEDENT_MATCH_SCORE = 18
PARTIAL_PRECEDENT_SCORE = 7
LIVE_READY_SCORE = 8
BOUNDED_ADVERSE_ADD_SCORE = 10
WOUNDED_THESIS_PENALTY = -18
INVALIDATED_THESIS_PENALTY = -55
WITH_MOVE_PYRAMID_PENALTY = -70
SL_WARN_PENALTY = -8
SL_BLOCK_PENALTY = -35
PAIR_OVERLAY_LIMIT = 8


@dataclass(frozen=True)
class TradeShape:
    lane_id: str
    status: str
    pair: str
    side: str
    currency_bought: str
    currency_sold: str
    cleanest_pair_expression: str
    session: str
    location_24h: str
    h1_alignment: str
    h4_alignment: str
    tape_state: str
    entry_shape: str
    building_style: str
    thesis_state: str
    sl_lint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "status": self.status,
            "pair": self.pair,
            "side": self.side,
            "currency_bought": self.currency_bought,
            "currency_sold": self.currency_sold,
            "cleanest_pair_expression": self.cleanest_pair_expression,
            "session": self.session,
            "location_24h": self.location_24h,
            "h1_alignment": self.h1_alignment,
            "h4_alignment": self.h4_alignment,
            "tape_state": self.tape_state,
            "entry_shape": self.entry_shape,
            "building_style": self.building_style,
            "thesis_state": self.thesis_state,
            "sl_lint": self.sl_lint,
        }


def evaluate_trade_shape_engine(
    intents_payload: Mapping[str, Any],
    *,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Evaluate every order-intent candidate with the common shape engine."""

    evaluations = [
        evaluation
        for row in intents_payload.get("results", []) or []
        if isinstance(row, Mapping)
        for evaluation in [_evaluate_row(row)]
        if evaluation is not None
    ]
    evaluations.sort(
        key=lambda item: (
            -int(item.get("trade_shape_score") or 0),
            str(item.get("pair") or ""),
            str(item.get("lane_id") or ""),
        )
    )
    if max_candidates is not None:
        evaluations = evaluations[: max(0, int(max_candidates))]
    pair_summaries = _best_by_pair(evaluations)
    shape_matched_live_ready = [
        item
        for item in evaluations
        if item.get("status") == "LIVE_READY"
        and ((item.get("precedent_match") or {}).get("status") == "MATCH")
    ]
    return {
        "status": "TRADE_SHAPE_ENGINE_READY",
        "precedent": generalized_discretionary_precedent(),
        "candidate_count": len(evaluations),
        "candidate_pairs": sorted(pair_summaries),
        "shape_matched_live_ready_lanes": [
            _candidate_brief(item) for item in shape_matched_live_ready[:20]
        ],
        "shape_matched_live_ready_count": len(shape_matched_live_ready),
        "pair_summaries": pair_summaries,
        "pair_evaluations": evaluations,
        "contract": {
            "advisory_only": True,
            "pair_agnostic_core": True,
            "pair_specific_overlays_are_adjustments_only": True,
            "does_not_grant_live_permission": True,
            "does_not_replace_risk_engine": True,
            "does_not_force_usd_jpy_only_trading": True,
        },
    }


def generalized_discretionary_precedent() -> dict[str, Any]:
    """Reusable manual-history lessons independent of the original pair."""

    return {
        "id": "operator_2025_discretionary_trade_shape",
        "source_history_pair": "USD_JPY",
        "pair_agnostic": True,
        "winning_shape": {
            "read_theme_first": True,
            "build_only_when_thesis_alive": True,
            "preferred_building_styles": ["SINGLE", "BOUNDED_ADVERSE_ADD"],
            "blocked_building_styles": ["WITH_MOVE_PYRAMID"],
            "bounded_adverse_add_worked": True,
            "with_move_pyramiding_failed": True,
            "avoid_tight_sl_in_noise": True,
            "harvest_and_housekeeping_mattered": True,
            "margin_closeout_forbidden": True,
            "long_unattended_holds_tail_risk": True,
        },
        "operator_memory": (
            "The 2025 USD_JPY manual history is not a USD_JPY-only rule. It is "
            "operator precedent for a reusable trade shape: read theme, build only "
            "when thesis is alive, prefer bounded adverse add over with-move "
            "pyramid, avoid tight SL in noise, harvest actively, and forbid margin "
            "closeout / unattended carry."
        ),
    }


def evaluate_trade_shape_result(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Public helper for tests and focused diagnostics."""

    return _evaluate_row(row)


def _evaluate_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = _shape_from_row(row)
    if shape is None:
        return None
    risk_bounded = _risk_bounded(row)
    precedent_match = _precedent_match(shape, risk_bounded=risk_bounded)
    pair_adjustments = _pair_specific_adjustments(shape)
    allowed, blocked = _behavior_lists(shape, risk_bounded=risk_bounded)
    core_score = _core_score(shape, precedent_match=precedent_match, risk_bounded=risk_bounded)
    overlay_delta = sum(int(item.get("score_delta") or 0) for item in pair_adjustments)
    score = max(0, min(100, core_score + overlay_delta))
    tradable = (
        shape.status == "LIVE_READY"
        and shape.thesis_state == "ALIVE"
        and shape.building_style != "WITH_MOVE_PYRAMID"
        and shape.sl_lint != "BLOCK"
        and not any(item.get("code") in _hard_behavior_codes() for item in blocked)
    )
    reason = "tradable shape; final authority remains RiskEngine and LiveOrderGateway"
    if not tradable:
        reason = _not_tradable_reason(row, shape, blocked)
    return {
        "lane_id": shape.lane_id,
        "pair": shape.pair,
        "status": shape.status,
        "trade_shape_score": score,
        "core_score_before_overlays": core_score,
        "overlay_score_delta": overlay_delta,
        "precedent_match": precedent_match,
        "trade_shape": shape.as_dict(),
        "pair_specific_adjustments": pair_adjustments,
        "allowed_behaviors": allowed,
        "blocked_behaviors": blocked,
        "tradable": tradable,
        "exact_reason_if_not_tradable": "" if tradable else reason,
    }


def _shape_from_row(row: Mapping[str, Any]) -> TradeShape | None:
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    if not intent:
        return None
    context = intent.get("market_context") if isinstance(intent.get("market_context"), Mapping) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), Mapping) else {}
    pair = str(intent.get("pair") or "").upper().strip()
    side = str(intent.get("side") or "").upper().strip()
    lane_id = str(row.get("lane_id") or "").strip()
    if not pair or side not in {"LONG", "SHORT"} or not lane_id:
        return None
    base, quote = _pair_currencies(pair)
    bought = base if side == "LONG" else quote
    sold = quote if side == "LONG" else base
    session = str(
        context.get("session")
        or metadata.get("session_current_tag")
        or metadata.get("session_bucket")
        or "UNKNOWN"
    )
    h1_regime = _tf_regime(metadata, context, "H1")
    h4_regime = _tf_regime(metadata, context, "H4")
    status = str(row.get("status") or "").upper()
    return TradeShape(
        lane_id=lane_id,
        status=status,
        pair=pair,
        side=side,
        currency_bought=bought,
        currency_sold=sold,
        cleanest_pair_expression=pair,
        session=session,
        location_24h=_location_24h(metadata, context),
        h1_alignment=_alignment(side, h1_regime, "H1"),
        h4_alignment=_alignment(side, h4_regime, "H4"),
        tape_state=_tape_state(context, metadata),
        entry_shape=_entry_shape(intent, context, metadata),
        building_style=_building_style(metadata),
        thesis_state=_thesis_state(row, metadata),
        sl_lint=_sl_lint_state(row, intent, metadata),
    )


def _precedent_match(shape: TradeShape, *, risk_bounded: bool) -> dict[str, Any]:
    reasons: list[str] = []
    blockers: list[str] = []
    if shape.thesis_state == "ALIVE":
        reasons.append("thesis_alive")
    else:
        blockers.append(f"thesis_{shape.thesis_state.lower()}")
    if shape.building_style in {"SINGLE", "BOUNDED_ADVERSE_ADD"}:
        reasons.append(f"building_{shape.building_style.lower()}")
    if shape.building_style == "WITH_MOVE_PYRAMID":
        blockers.append("with_move_pyramid_failed_in_precedent")
    if shape.building_style == "BOUNDED_ADVERSE_ADD" and not risk_bounded:
        blockers.append("bounded_adverse_add_requires_current_risk_bound")
    if shape.sl_lint == "BLOCK":
        blockers.append("sl_lint_block")
    elif shape.sl_lint == "WARN":
        reasons.append("sl_lint_warn_review_required")
    else:
        reasons.append("sl_lint_pass")
    if shape.entry_shape in {"SCOUT", "PULLBACK", "FADE", "RETEST"}:
        reasons.append(f"entry_shape_{shape.entry_shape.lower()}")
    else:
        reasons.append(f"entry_shape_{shape.entry_shape.lower()}_requires_current_breakout_proof")

    if blockers:
        status = "NO_MATCH"
    elif shape.status == "LIVE_READY":
        status = "MATCH"
    else:
        status = "PARTIAL"
    return {
        "status": status,
        "source": "operator_2025_discretionary_trade_shape",
        "pair_agnostic": True,
        "reasons": reasons,
        "blockers": blockers,
    }


def _behavior_lists(
    shape: TradeShape,
    *,
    risk_bounded: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = [
        {
            "code": "READ_THEME_FIRST",
            "behavior": "Use current currency theme and cleanest pair expression before execution filters.",
        },
        {
            "code": "HARVEST_AND_HOUSEKEEP",
            "behavior": "Keep TP/harvest and housekeeping active; do not leave stale exposure unattended.",
        },
    ]
    blocked = [
        {
            "code": "MARGIN_CLOSEOUT_FORBIDDEN",
            "behavior": "Do not use margin pressure or broker liquidation as an exit plan.",
        },
        {
            "code": "UNATTENDED_CARRY_FORBIDDEN",
            "behavior": "Do not carry a decayed thesis without fresh ALIVE evidence and management.",
        },
    ]
    if shape.building_style == "SINGLE":
        allowed.append({"code": "SINGLE_ALLOWED", "behavior": "Single-entry expression may proceed after current gates."})
    elif shape.building_style == "BOUNDED_ADVERSE_ADD":
        if shape.thesis_state == "ALIVE" and risk_bounded:
            allowed.append(
                {
                    "code": "BOUNDED_ADVERSE_ADD_ALLOWED",
                    "behavior": "Bounded adverse add may be considered after current risk and gateway validation.",
                }
            )
        else:
            blocked.append(
                {
                    "code": "BOUNDED_ADVERSE_ADD_BLOCKED",
                    "behavior": "Bounded adverse add requires ALIVE thesis and current bounded-risk proof.",
                }
            )
    elif shape.building_style == "WITH_MOVE_PYRAMID":
        blocked.append(
            {
                "code": "WITH_MOVE_PYRAMID_BLOCKED",
                "behavior": "With-move pyramiding is blocked by the generalized manual precedent.",
            }
        )
    if shape.thesis_state == "INVALIDATED":
        blocked.append(
            {
                "code": "THESIS_INVALIDATED_BLOCKED",
                "behavior": "Do not add risk after thesis invalidation.",
            }
        )
    elif shape.thesis_state == "WOUNDED":
        blocked.append(
            {
                "code": "WOUNDED_THESIS_NEEDS_REPAIR",
                "behavior": "Do not build until the current blocker is cleared or the thesis is refreshed.",
            }
        )
    if shape.sl_lint == "BLOCK":
        blocked.append(
            {
                "code": "SL_LINT_BLOCKED",
                "behavior": "Broker SL shape is blocked until SL lint evidence clears.",
            }
        )
    elif shape.sl_lint == "WARN":
        allowed.append(
            {
                "code": "SL_LINT_WARN_REVIEW",
                "behavior": "Treat SL lint as warning-only audit context until gateway lint runs.",
            }
        )
    return allowed, blocked


def _core_score(
    shape: TradeShape,
    *,
    precedent_match: Mapping[str, Any],
    risk_bounded: bool,
) -> int:
    score = BASE_SHAPE_SCORE
    match_status = str(precedent_match.get("status") or "")
    if match_status == "MATCH":
        score += PRECEDENT_MATCH_SCORE
    elif match_status == "PARTIAL":
        score += PARTIAL_PRECEDENT_SCORE
    if shape.status == "LIVE_READY":
        score += LIVE_READY_SCORE
    if shape.building_style == "BOUNDED_ADVERSE_ADD" and risk_bounded:
        score += BOUNDED_ADVERSE_ADD_SCORE
    if shape.thesis_state == "WOUNDED":
        score += WOUNDED_THESIS_PENALTY
    elif shape.thesis_state == "INVALIDATED":
        score += INVALIDATED_THESIS_PENALTY
    if shape.building_style == "WITH_MOVE_PYRAMID":
        score += WITH_MOVE_PYRAMID_PENALTY
    if shape.sl_lint == "WARN":
        score += SL_WARN_PENALTY
    elif shape.sl_lint == "BLOCK":
        score += SL_BLOCK_PENALTY
    return max(0, min(100, score))


def _pair_specific_adjustments(shape: TradeShape) -> list[dict[str, Any]]:
    pair = shape.pair
    overlays: list[dict[str, Any]] = []
    if pair == "USD_JPY":
        overlays.append(
            _overlay(
                "USD_JPY_MAJOR_FIGURE_INTERVENTION_RISK",
                -4,
                "USD_JPY overlay: major-figure and intervention risk require extra SL/theme care.",
            )
        )
    if pair.endswith("_JPY") or pair.startswith("JPY_"):
        overlays.append(
            _overlay(
                "JPY_THEME_SHARED_RISK_BUDGET",
                -3,
                "JPY-cross overlay: same JPY theme must share risk budget across related expressions.",
            )
        )
    if pair == "GBP_JPY":
        overlays.append(
            _overlay(
                "GBP_JPY_SPREAD_NOISE_PENALTY",
                -5,
                "GBP_JPY overlay: spread/noise penalty adjusts score but does not replace core shape scoring.",
            )
        )
    if pair == "AUD_USD":
        overlays.append(
            _overlay(
                "AUD_USD_NO_EDGE_SIZE_CAP",
                -3,
                "AUD_USD overlay: no-edge size cap keeps this expression scout-sized unless current edge proves itself.",
            )
        )
    if pair == "EUR_USD":
        overlays.append(
            _overlay(
                "EUR_USD_DIRECT_USD_THEME_EXPRESSION",
                3,
                "EUR_USD overlay: direct USD theme expression can receive a small advisory preference.",
            )
        )
    return overlays[:PAIR_OVERLAY_LIMIT]


def _overlay(code: str, score_delta: int, reason: str) -> dict[str, Any]:
    return {
        "code": code,
        "score_delta": score_delta,
        "reason": reason,
        "overlay_only": True,
    }


def _risk_bounded(row: Mapping[str, Any]) -> bool:
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), Mapping) else {}
    if metadata.get("risk_bounded") is True or metadata.get("bounded_risk") is True:
        return True
    status = str(row.get("status") or "").upper()
    if status != "LIVE_READY":
        return False
    if _blocking_issue_codes(row):
        return False
    if metadata.get("same_pair_add_type") and metadata.get("same_pair_add_type") != "AVERAGE_INTO_ADVERSE":
        return False
    if metadata.get("same_pair_add_type") == "AVERAGE_INTO_ADVERSE":
        return metadata.get("tp_atr_pips") is not None or metadata.get("adverse_add_atr_multiple") is not None
    return True


def _not_tradable_reason(
    row: Mapping[str, Any],
    shape: TradeShape,
    blocked: Iterable[Mapping[str, Any]],
) -> str:
    for item in blocked:
        code = str(item.get("code") or "")
        if code in _hard_behavior_codes():
            return str(item.get("behavior") or code)
    if shape.status != "LIVE_READY":
        blockers = [str(item) for item in row.get("live_blockers", []) or [] if item]
        if blockers:
            return blockers[0]
        codes = _all_issue_codes(row)
        if codes:
            return codes[0]
        return f"lane status is {shape.status or 'UNKNOWN'}, not LIVE_READY"
    return "current shape failed generalized discretionary pattern checks"


def _hard_behavior_codes() -> set[str]:
    return {
        "WITH_MOVE_PYRAMID_BLOCKED",
        "BOUNDED_ADVERSE_ADD_BLOCKED",
        "THESIS_INVALIDATED_BLOCKED",
        "SL_LINT_BLOCKED",
    }


def _pair_currencies(pair: str) -> tuple[str, str]:
    if "_" not in pair:
        return pair, "UNKNOWN"
    base, quote = pair.split("_", 1)
    return base or "UNKNOWN", quote or "UNKNOWN"


def _tf_regime(metadata: Mapping[str, Any], context: Mapping[str, Any], tf: str) -> Any:
    lower = tf.lower()
    direct = metadata.get(f"{lower}_regime") or context.get(f"{lower}_regime")
    if direct:
        return direct
    tf_map = metadata.get("tf_regime_map")
    if isinstance(tf_map, Mapping):
        row = tf_map.get(tf)
        if isinstance(row, Mapping):
            return row.get("regime") or row.get("classification") or row.get("state")
    return None


def _location_24h(metadata: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    value = (
        metadata.get("entry_price_percentile_24h")
        or context.get("entry_price_percentile_24h")
        or metadata.get("price_percentile_24h")
        or context.get("price_percentile_24h")
    )
    percentile = _maybe_float(value)
    if percentile is None:
        return LOCATION_UNKNOWN
    # Terciles are descriptive 24h-location buckets matching the requested
    # LOWER/MIDDLE/UPPER shape labels, not new execution thresholds.
    if percentile < (1.0 / 3.0):
        return LOCATION_LOWER
    if percentile > (2.0 / 3.0):
        return LOCATION_UPPER
    return LOCATION_MIDDLE


def _alignment(side: str, regime: Any, tf: str) -> str:
    direction = _trend_direction(regime)
    if direction is None:
        return f"{tf}_UNKNOWN"
    side_direction = "UP" if side == "LONG" else "DOWN"
    relation = "WITH" if side_direction == direction else "AGAINST"
    return f"{relation}_{tf}_TREND"


def _trend_direction(regime: Any) -> str | None:
    text = str(regime or "").upper()
    if "UP" in text:
        return "UP"
    if "DOWN" in text:
        return "DOWN"
    return None


def _tape_state(context: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            metadata.get("tape_state"),
            metadata.get("forecast_direction"),
            metadata.get("range_phase"),
            metadata.get("dominant_regime_state"),
            metadata.get("regime_state"),
            context.get("regime"),
            context.get("method"),
        )
    ).upper()
    if "SQUEEZE" in text:
        return "SQUEEZE"
    if "RANGE_ROTATION" in text or "ROTATION" in text:
        return "ROTATION"
    if "RANGE" in text:
        return "RANGE"
    if "FADE" in text or "FAILURE" in text:
        return "FADE"
    if "TREND" in text or "IMPULSE" in text:
        return "TREND"
    return "RANGE"


def _entry_shape(intent: Mapping[str, Any], context: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("entry_shape") or "").upper()
    if explicit in ENTRY_SHAPES:
        return explicit
    method = str(context.get("method") or metadata.get("method") or "").upper()
    order_type = str(intent.get("order_type") or "").upper()
    campaign_role = str(metadata.get("campaign_role") or "").upper()
    if "SCOUT" in campaign_role:
        return "SCOUT"
    if method == "RANGE_ROTATION":
        return "FADE" if order_type == "LIMIT" else "SCOUT"
    if method == "BREAKOUT_FAILURE":
        return "RETEST" if order_type == "LIMIT" else "FADE"
    if method == "TREND_CONTINUATION":
        return "BREAKOUT" if order_type in {"STOP", "STOP-ENTRY"} else "PULLBACK"
    if order_type == "LIMIT":
        return "PULLBACK"
    if order_type in {"STOP", "STOP-ENTRY"}:
        return "BREAKOUT"
    return "SCOUT"


def _building_style(metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("building_style") or "").upper()
    if explicit in BUILDING_STYLES:
        return explicit
    add_type = str(metadata.get("same_pair_add_type") or "").upper()
    if add_type == "AVERAGE_INTO_ADVERSE":
        return "BOUNDED_ADVERSE_ADD"
    if add_type == "PYRAMID_WITH_MOVE":
        return "WITH_MOVE_PYRAMID"
    return "SINGLE"


def _thesis_state(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("thesis_state") or "").upper()
    if explicit in THESIS_STATES:
        return explicit
    status = str(row.get("status") or "").upper()
    text = " ".join(
        [status]
        + [str(item or "") for item in row.get("live_blockers", []) or []]
        + _all_issue_codes(row)
    ).upper()
    if "INVALIDATED" in text or "THESIS_BROKEN" in text or "RECOMMEND_CLOSE" in text:
        return "INVALIDATED"
    if status == "LIVE_READY":
        return "ALIVE"
    if "BLOCK" in text or "WATCH_ONLY" in text or "NEGATIVE_EXPECTANCY" in text:
        return "WOUNDED"
    return "ALIVE"


def _sl_lint_state(row: Mapping[str, Any], intent: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("sl_lint_status") or metadata.get("sl_lint") or "").upper()
    if explicit in SL_LINT_STATES:
        return explicit
    lint = intent.get("sl_lint") if isinstance(intent.get("sl_lint"), Mapping) else row.get("sl_lint")
    if isinstance(lint, Mapping):
        status = str(lint.get("status") or "").upper()
        if status in SL_LINT_STATES:
            return status
        issues = lint.get("issues") if isinstance(lint.get("issues"), list) else []
        severities = {str(item.get("severity") or "").upper() for item in issues if isinstance(item, Mapping)}
        if "BLOCK" in severities:
            return "BLOCK"
        if "WARN" in severities:
            return "WARN"
    if any(code.startswith("SL_LINT_") for code in _all_issue_codes(row)):
        return "BLOCK"
    if intent.get("sl") is not None or metadata.get("disaster_sl") is not None:
        return "WARN"
    return "PASS"


def _all_issue_codes(row: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    for key in ("risk_issues", "strategy_issues", "live_strategy_issues"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, Mapping) and str(item.get("code") or "").strip():
                codes.append(str(item.get("code")).strip())
    return codes


def _blocking_issue_codes(row: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    for key in ("risk_issues", "strategy_issues", "live_strategy_issues"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("severity") or "").upper() == "BLOCK" and str(item.get("code") or "").strip():
                codes.append(str(item.get("code")).strip())
    return codes


def _best_by_pair(evaluations: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in evaluations:
        pair = str(item.get("pair") or "").upper()
        if not pair:
            continue
        current = out.get(pair)
        if current is None or int(item.get("trade_shape_score") or 0) > int(current.get("trade_shape_score") or 0):
            out[pair] = _candidate_brief(item)
    return out


def _candidate_brief(item: Mapping[str, Any]) -> dict[str, Any]:
    shape = item.get("trade_shape") if isinstance(item.get("trade_shape"), Mapping) else {}
    return {
        "lane_id": item.get("lane_id"),
        "pair": item.get("pair"),
        "status": item.get("status"),
        "trade_shape_score": item.get("trade_shape_score"),
        "precedent_match": item.get("precedent_match"),
        "building_style": shape.get("building_style"),
        "thesis_state": shape.get("thesis_state"),
        "sl_lint": shape.get("sl_lint"),
        "tradable": item.get("tradable"),
        "exact_reason_if_not_tradable": item.get("exact_reason_if_not_tradable"),
    }


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
