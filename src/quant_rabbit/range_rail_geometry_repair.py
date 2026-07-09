from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_FORECAST_PATTERN_REFRESH,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
    DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR_REPORT,
)


SCHEMA_VERSION = "range_rail_geometry_repair_v1"
LONG_DISCOUNT_RAIL_MAX_POSITION = 0.35
SHORT_PREMIUM_RAIL_MIN_POSITION = 0.65
RANGE_REPAIR_ACTION = "RANGE_RAIL_GEOMETRY_REPAIR"


@dataclass(frozen=True)
class RangeRailGeometryRepairSummary:
    status: str
    output_path: Path
    report_path: Path
    target_lane_id: str | None
    live_permission_allowed: bool


class RangeRailGeometryRepair:
    """Consume RANGE rail repair work into a concrete, read-only geometry contract."""

    def __init__(
        self,
        *,
        forecast_pattern_refresh_path: Path = DEFAULT_FORECAST_PATTERN_REFRESH,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        output_path: Path = DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
        report_path: Path = DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "forecast_pattern_refresh": forecast_pattern_refresh_path,
            "active_opportunity_board": active_opportunity_board_path,
            "order_intents": order_intents_path,
        }
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> RangeRailGeometryRepairSummary:
        payload = self.build_payload()
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("range-rail geometry repair must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("range-rail geometry repair must not record live side effects")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
        return RangeRailGeometryRepairSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            target_lane_id=top.get("lane_id"),
            live_permission_allowed=False,
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        order_intents = _order_intents_by_lane(artifacts["order_intents"])
        target = _select_target(
            forecast_pattern_refresh=artifacts["forecast_pattern_refresh"],
            active_board=artifacts["active_opportunity_board"],
        )
        target_lane = _analyze_target(target, order_intents=order_intents)
        actions = _next_actions(target_lane)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": _status(target_lane),
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "top_lane": target_lane,
            "target_lane": target_lane,
            "next_actions": actions,
            "next_contract_prompt": _next_contract_prompt(target_lane, actions),
            "source_artifacts": _source_artifacts(artifacts),
            "do_not_do": _do_not_do(),
        }
        return payload


def _load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_artifact_status": "missing", "_path": str(path), "_sha256": None}
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    result = dict(payload)
    result["_artifact_status"] = "present"
    result["_path"] = str(path)
    result["_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def _select_target(*, forecast_pattern_refresh: dict[str, Any], active_board: dict[str, Any]) -> dict[str, Any]:
    top = forecast_pattern_refresh.get("top_lane") if isinstance(forecast_pattern_refresh.get("top_lane"), dict) else {}
    actions = forecast_pattern_refresh.get("next_actions") if isinstance(forecast_pattern_refresh.get("next_actions"), list) else []
    has_repair_action = any(
        isinstance(row, dict) and row.get("action_type") == RANGE_REPAIR_ACTION for row in actions
    )
    if top and has_repair_action:
        return dict(top)
    board_top = active_board.get("top_lane") if isinstance(active_board.get("top_lane"), dict) else {}
    if top and board_top and _same_lane_shape(top, board_top):
        return dict(top)
    return {}


def _analyze_target(target: dict[str, Any], *, order_intents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not target:
        return {}
    pair = str(target.get("pair") or "UNKNOWN")
    direction = str(target.get("direction") or "UNKNOWN").upper()
    counterpart_from_refresh = (
        target.get("range_rotation_counterpart")
        if isinstance(target.get("range_rotation_counterpart"), dict)
        else {}
    )
    limit_refresh = (
        counterpart_from_refresh.get("limit") if isinstance(counterpart_from_refresh.get("limit"), dict) else {}
    )
    market_refresh = (
        counterpart_from_refresh.get("market") if isinstance(counterpart_from_refresh.get("market"), dict) else {}
    )
    preferred_lane_id = str(
        counterpart_from_refresh.get("preferred_lane_id")
        or limit_refresh.get("lane_id")
        or f"range_trader:{pair}:{direction}:RANGE_ROTATION"
    )
    limit_lane_id = f"range_trader:{pair}:{direction}:RANGE_ROTATION"
    market_lane_id = f"{limit_lane_id}:MARKET"
    limit_intent = _intent_summary(
        order_intents.get(limit_lane_id) or {},
        fallback=limit_refresh,
        fallback_lane_id=limit_lane_id,
    )
    market_intent = _intent_summary(
        order_intents.get(market_lane_id) or {},
        fallback=market_refresh,
        fallback_lane_id=market_lane_id,
    )
    preferred = limit_intent if preferred_lane_id == limit_lane_id or limit_intent.get("present") else market_intent
    range_box = _range_box_geometry(target, direction=direction)
    counterpart_geometry = _counterpart_geometry(preferred, range_box=range_box, direction=direction)
    blockers = _combined_blockers(target, limit_intent, market_intent)
    rail_status = range_box.get("rail_status")
    return {
        "lane_id": target.get("lane_id"),
        "pair": pair,
        "direction": direction,
        "strategy_family": target.get("strategy_family"),
        "vehicle": target.get("vehicle"),
        "status": target.get("status"),
        "blockers": _string_list(target.get("blockers")),
        "range_box": range_box,
        "range_rotation_counterpart": {
            "preferred_lane_id": preferred.get("lane_id"),
            "preferred_order_type": preferred.get("order_type"),
            "limit": limit_intent,
            "market": market_intent,
            "refresh_status": counterpart_from_refresh.get("status"),
            "blocker_codes": _unique(_string_list(limit_intent.get("blocker_codes")) + _string_list(market_intent.get("blocker_codes"))),
        },
        "counterpart_geometry": counterpart_geometry,
        "rail_success_condition": _rail_success_condition(range_box, direction=direction, pair=pair),
        "blocker_preservation": {
            "negative_expectancy_visible": _contains(blockers, "NEGATIVE_EXPECTANCY"),
            "spread_too_wide_not_ignored": _contains(blockers, "SPREAD"),
            "bidask_negative_not_ignored": _contains(blockers, "BIDASK"),
            "range_location_blockers_not_ignored": any(
                code in blockers
                for code in (
                    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                    "EXHAUSTION_RANGE_CHASE",
                    "RANGE_MARKET_NOT_AT_RAIL",
                )
            ),
            "live_permission_allowed": False,
        },
        "repair_status": _repair_status(rail_status, counterpart_geometry),
    }


def _range_box_geometry(target: dict[str, Any], *, direction: str) -> dict[str, Any]:
    forecast = target.get("forecast_range_box") if isinstance(target.get("forecast_range_box"), dict) else {}
    latest = forecast.get("latest") if isinstance(forecast.get("latest"), dict) else {}
    current = _float(latest.get("current_price"))
    low = _float(latest.get("range_low_price"))
    high = _float(latest.get("range_high_price"))
    box_position = _float(forecast.get("box_position"))
    if box_position is None and current is not None and low is not None and high is not None and high > low:
        box_position = round((current - low) / (high - low), 4)
    rail_status = "RANGE_BOX_DATA_INCOMPLETE"
    required_zone = "UNKNOWN"
    threshold: dict[str, float] = {}
    if str(latest.get("direction") or "") != "RANGE":
        rail_status = "LATEST_FORECAST_NOT_RANGE" if latest else "RANGE_BOX_DATA_INCOMPLETE"
    elif box_position is None or low is None or high is None or current is None or high <= low:
        rail_status = "RANGE_BOX_DATA_INCOMPLETE"
    elif direction == "LONG":
        required_zone = "LONG_DISCOUNT_LOWER_RAIL"
        threshold = {"box_position_lte": LONG_DISCOUNT_RAIL_MAX_POSITION}
        rail_status = (
            "RANGE_RAIL_REACHED"
            if box_position <= LONG_DISCOUNT_RAIL_MAX_POSITION
            else "RANGE_RAIL_NOT_REACHED"
        )
    elif direction == "SHORT":
        required_zone = "SHORT_PREMIUM_UPPER_RAIL"
        threshold = {"box_position_gte": SHORT_PREMIUM_RAIL_MIN_POSITION}
        rail_status = (
            "RANGE_RAIL_REACHED"
            if box_position >= SHORT_PREMIUM_RAIL_MIN_POSITION
            else "RANGE_RAIL_NOT_REACHED"
        )
    return {
        "latest": latest,
        "low": low,
        "high": high,
        "current": current,
        "box_position": box_position,
        "required_zone": required_zone,
        "threshold": threshold,
        "rail_status": rail_status,
        "source_status": forecast.get("status"),
    }


def _counterpart_geometry(intent: dict[str, Any], *, range_box: dict[str, Any], direction: str) -> dict[str, Any]:
    low = _float(range_box.get("low"))
    high = _float(range_box.get("high"))
    entry = _float(intent.get("entry_price"))
    tp = _float(intent.get("take_profit_price"))
    sl = _float(intent.get("stop_loss_price"))
    order_type = str(intent.get("order_type") or "UNKNOWN").upper()
    if low is None or high is None or high <= low:
        return {
            "status": "COUNTERPART_BOX_GEOMETRY_INCOMPLETE",
            "geometry_ready": False,
            "reasons": ["RANGE_BOX_DATA_INCOMPLETE"],
        }
    reasons: list[str] = []
    entry_inside = entry is not None and low <= entry <= high
    tp_inside = False
    sl_outside = False
    tp_directional = False
    if direction == "LONG":
        tp_inside = tp is not None and entry is not None and entry < tp <= high
        sl_outside = sl is not None and sl < low
        tp_directional = tp is not None and entry is not None and tp > entry
    elif direction == "SHORT":
        tp_inside = tp is not None and entry is not None and low <= tp < entry
        sl_outside = sl is not None and sl > high
        tp_directional = tp is not None and entry is not None and tp < entry
    if order_type != "LIMIT":
        reasons.append("RANGE_ROTATION_LIMIT_COUNTERPART_REQUIRED")
    if not entry_inside:
        reasons.append("ENTRY_NOT_INSIDE_RANGE_BOX")
    if not tp_inside:
        reasons.append("TP_NOT_INSIDE_RANGE_BOX")
    if not tp_directional:
        reasons.append("TP_NOT_DIRECTIONAL_FROM_ENTRY")
    if not sl_outside:
        reasons.append("SL_NOT_OUTSIDE_RANGE_BOX")
    geometry_ready = not reasons
    return {
        "status": "COUNTERPART_GEOMETRY_READY" if geometry_ready else "COUNTERPART_PRICE_GEOMETRY_INCOMPLETE",
        "geometry_ready": geometry_ready,
        "lane_id": intent.get("lane_id"),
        "order_type": order_type,
        "entry_price": entry,
        "take_profit_price": tp,
        "stop_loss_price": sl,
        "entry_inside_box": entry_inside,
        "tp_inside_box": tp_inside,
        "tp_directional_from_entry": tp_directional,
        "sl_outside_box": sl_outside,
        "reasons": reasons,
    }


def _rail_success_condition(range_box: dict[str, Any], *, direction: str, pair: str) -> dict[str, Any]:
    condition: dict[str, Any] = {
        "pair": pair,
        "direction": direction,
        "required_zone": range_box.get("required_zone"),
        "current_box_position": range_box.get("box_position"),
        "range_low_price": range_box.get("low"),
        "range_high_price": range_box.get("high"),
        "current_price": range_box.get("current"),
        "must_preserve_blockers": [
            "SPREAD_TOO_WIDE",
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "RANGE_ROTATION_BROADER_LOCATION_CHASE",
            "EXHAUSTION_RANGE_CHASE",
        ],
    }
    if direction == "LONG":
        condition["required_box_position_lte"] = LONG_DISCOUNT_RAIL_MAX_POSITION
    elif direction == "SHORT":
        condition["required_box_position_gte"] = SHORT_PREMIUM_RAIL_MIN_POSITION
    return condition


def _next_actions(target_lane: dict[str, Any]) -> list[dict[str, Any]]:
    if not target_lane:
        return []
    lane_id = str(target_lane.get("lane_id") or "")
    range_box = target_lane.get("range_box") if isinstance(target_lane.get("range_box"), dict) else {}
    geometry = (
        target_lane.get("counterpart_geometry")
        if isinstance(target_lane.get("counterpart_geometry"), dict)
        else {}
    )
    actions: list[dict[str, Any]] = []
    if range_box.get("rail_status") == "RANGE_RAIL_NOT_REACHED":
        actions.append(
            _action(
                lane_id,
                "WAIT_FOR_RANGE_RAIL_RECHECK",
                "Recheck only when the current RANGE box reaches the executable broad rail; do not chase midpoint/opposite rail.",
                blockers=["RANGE_RAIL_NOT_REACHED", "RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
                priority=1,
            )
        )
    elif range_box.get("rail_status") == "RANGE_RAIL_REACHED" and geometry.get("geometry_ready"):
        actions.append(
            _action(
                lane_id,
                "RANGE_ROTATION_GEOMETRY_READY_PROOF_BLOCKED",
                "Range rail and LIMIT entry/TP/SL geometry are ready; continue proof/spread/expectancy validation before any live route.",
                blockers=["SPREAD_TOO_WIDE", "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                priority=1,
            )
        )
    elif range_box.get("rail_status") == "RANGE_RAIL_REACHED":
        actions.append(
            _action(
                lane_id,
                "REPRICE_RANGE_ROTATION_COUNTERPART",
                "Rail reached, but LIMIT counterpart entry/TP/SL geometry is incomplete; reprice from current broker truth and range box.",
                blockers=_string_list(geometry.get("reasons")) or ["COUNTERPART_PRICE_GEOMETRY_INCOMPLETE"],
                priority=1,
            )
        )
    else:
        actions.append(
            _action(
                lane_id,
                "REFRESH_FORECAST_RANGE_BOX",
                "Refresh forecast range_low/range_high/current before repairing RANGE_ROTATION geometry.",
                blockers=[str(range_box.get("rail_status") or "RANGE_BOX_DATA_INCOMPLETE")],
                priority=1,
            )
        )
    actions.append(
        _action(
            lane_id,
            "VERIFY_TRIGGER_PROJECTIONS",
            "Resolve expired/pending trigger projections before using them as proof; do not infer hits.",
            blockers=["EXPIRED_PENDING_PROJECTIONS"],
            priority=2,
        )
    )
    actions.append(
        _action(
            lane_id,
            "EXACT_TP_PROOF_COLLECTION",
            "Collect exact attached TAKE_PROFIT_ORDER proof for the selected pair/side/method/vehicle; do not mix market-close losses.",
            blockers=[
                "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            ],
            priority=3,
        )
    )
    return actions


def _status(target_lane: dict[str, Any]) -> str:
    if not target_lane:
        return "NO_RANGE_RAIL_GEOMETRY_REPAIR_TARGET"
    repair_status = target_lane.get("repair_status")
    if repair_status:
        return str(repair_status)
    return "RANGE_RAIL_GEOMETRY_REPAIR_BUILT"


def _repair_status(rail_status: Any, counterpart_geometry: dict[str, Any]) -> str:
    if rail_status in {"RANGE_BOX_DATA_INCOMPLETE", "LATEST_FORECAST_NOT_RANGE"}:
        return "RANGE_RAIL_GEOMETRY_DATA_INCOMPLETE"
    if rail_status == "RANGE_RAIL_NOT_REACHED":
        return "RANGE_RAIL_RECHECK_BUILT"
    if rail_status == "RANGE_RAIL_REACHED" and counterpart_geometry.get("geometry_ready"):
        return "RANGE_RAIL_GEOMETRY_READY_PROOF_BLOCKED"
    if rail_status == "RANGE_RAIL_REACHED":
        return "RANGE_RAIL_GEOMETRY_REPAIR_BUILT"
    return "RANGE_RAIL_GEOMETRY_DATA_INCOMPLETE"


def _next_contract_prompt(target_lane: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    if not target_lane:
        return (
            "No range-rail geometry repair target is current. Rebuild forecast_pattern_refresh from the latest "
            "entry_frequency_recovery and active_opportunity_board; do not send."
        )
    first = actions[0] if actions else {}
    box = target_lane.get("range_box") if isinstance(target_lane.get("range_box"), dict) else {}
    geometry = (
        target_lane.get("counterpart_geometry")
        if isinstance(target_lane.get("counterpart_geometry"), dict)
        else {}
    )
    return (
        "Consume data/range_rail_geometry_repair.json for "
        f"{target_lane.get('lane_id')}: next safe action is {first.get('action_type') or 'KEEP_BLOCKED_WITH_CAUSE'}; "
        f"rail_status={box.get('rail_status')}; box_position={box.get('box_position')}; "
        f"counterpart_geometry_status={geometry.get('status')}. "
        "Preserve spread, bid/ask, negative-expectancy, and range-location blockers. "
        "Do not send, cancel, close, relax gates, or treat rail repair as live permission."
    )


def _action(
    lane_id: str,
    action_type: str,
    description: Any,
    *,
    blockers: list[str],
    priority: int,
) -> dict[str, Any]:
    return {
        "priority": priority,
        "lane_id": lane_id,
        "action_type": action_type,
        "description": str(description or ""),
        "preserve_blockers": _unique(blockers),
        "live_permission_allowed": False,
    }


def _intent_summary(
    row: dict[str, Any],
    *,
    fallback: dict[str, Any],
    fallback_lane_id: str,
) -> dict[str, Any]:
    source = row if row else fallback
    if not source:
        return {
            "present": False,
            "lane_id": fallback_lane_id,
            "status": "MISSING",
            "blocker_codes": [],
            "issue_messages": [],
        }
    intent = source.get("intent") if isinstance(source.get("intent"), dict) else {}
    blockers = _unique(
        _string_list(source.get("live_blocker_codes"))
        + _string_list(source.get("blocker_codes"))
        + _issue_codes(source.get("risk_issues"))
        + _issue_codes(source.get("strategy_issues"))
    )
    return {
        "present": bool(row) or bool(fallback),
        "lane_id": source.get("lane_id") or fallback_lane_id,
        "status": source.get("status"),
        "risk_allowed": bool(source.get("risk_allowed")),
        "blocker_codes": blockers,
        "order_type": intent.get("order_type") or source.get("order_type"),
        "units": _number_or_none(intent.get("units") if intent else source.get("units")),
        "entry_price": _number_or_none(intent.get("entry_price") if intent else source.get("entry_price")),
        "take_profit_price": _number_or_none(
            intent.get("take_profit_price") if intent else source.get("take_profit_price")
        ),
        "stop_loss_price": _number_or_none(
            intent.get("stop_loss_price") if intent else source.get("stop_loss_price")
        ),
        "issue_messages": _issue_messages(source.get("risk_issues"))[:8],
    }


def _order_intents_by_lane(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _list(artifact.get("results")):
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "")
        if lane_id:
            result[lane_id] = row
    return result


def _combined_blockers(target: dict[str, Any], *intents: dict[str, Any]) -> list[str]:
    blockers = _string_list(target.get("blockers"))
    for intent in intents:
        blockers.extend(_string_list(intent.get("blocker_codes")))
    return _unique(blockers)


def _source_artifacts(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "path": artifact.get("_path"),
            "status": artifact.get("_artifact_status"),
            "sha256": artifact.get("_sha256"),
            "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
        }
        for name, artifact in artifacts.items()
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Range Rail Geometry Repair",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        "",
    ]
    top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
    if top:
        box = top.get("range_box") if isinstance(top.get("range_box"), dict) else {}
        geometry = top.get("counterpart_geometry") if isinstance(top.get("counterpart_geometry"), dict) else {}
        condition = top.get("rail_success_condition") if isinstance(top.get("rail_success_condition"), dict) else {}
        lines.extend(
            [
                "## Target",
                "",
                f"- Lane: `{top.get('lane_id')}`",
                f"- Shape: `{top.get('pair')}` / `{top.get('direction')}` / `{top.get('strategy_family')}` / `{top.get('vehicle')}`",
                f"- Rail status: `{box.get('rail_status')}`",
                f"- Box position: `{box.get('box_position')}`",
                f"- Required zone: `{box.get('required_zone')}`",
                f"- Counterpart geometry: `{geometry.get('status')}`",
                f"- Success condition: `{json.dumps(condition, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(["## Next Actions", ""])
    for action in payload.get("next_actions") or []:
        if not isinstance(action, dict):
            continue
        lines.append(f"- P{action.get('priority')} `{action.get('action_type')}`: {action.get('description')}")
    lines.extend(["", "## Safety", ""])
    for item in payload.get("do_not_do") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _do_not_do() -> list[str]:
    return [
        "no live order / cancel / close",
        "no launchd changes",
        "no gate relaxation",
        "do not ignore spread too wide",
        "do not ignore bid/ask negative replay",
        "do not hide negative expectancy",
        "do not chase RANGE midpoint or opposite rail",
        "do not mix market-close losses into TP proof",
        "do not backsolve lot size from the 4x gap",
        "do not infer operator approval",
        "do not print secrets",
    ]


def _same_lane_shape(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("lane_id") and a.get("lane_id") == b.get("lane_id"):
        return True
    return (
        str(a.get("pair") or "").upper(),
        str(a.get("direction") or "").upper(),
        str(a.get("strategy_family") or "").upper(),
    ) == (
        str(b.get("pair") or "").upper(),
        str(b.get("direction") or "").upper(),
        str(b.get("strategy_family") or "").upper(),
    )


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def _issue_codes(value: Any) -> list[str]:
    result: list[str] = []
    for row in _list(value):
        if isinstance(row, dict) and row.get("code"):
            result.append(str(row["code"]))
    return result


def _issue_messages(value: Any) -> list[str]:
    result: list[str] = []
    for row in _list(value):
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").strip()
        message = str(row.get("message") or "").strip()
        if code and message:
            result.append(f"{code}: {message}")
        elif code:
            result.append(code)
    return result


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _contains(values: list[str], needle: str) -> bool:
    needle = needle.upper()
    return any(needle in str(value).upper() for value in values)


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_or_none(value: Any) -> float | int | None:
    number = _float(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return number
