from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_ENTRY_FREQUENCY_RECOVERY,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_FORECAST_PATTERN_REFRESH,
    DEFAULT_FORECAST_PATTERN_REFRESH_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PROJECTION_LEDGER,
)


SCHEMA_VERSION = "forecast_pattern_refresh_v1"
RANGE_FORECAST_BLOCKER = "RANGE_FORECAST_REQUIRES_RANGE_ROTATION"
ENTRY_DROUGHT_BLOCKER = "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH"
NEGATIVE_EXPECTANCY_BLOCKER = "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
TP_PROOF_BLOCKER = "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"
FORECAST_PATTERN_TARGET_ACTIONS = {
    "FORECAST_PATTERN_REFRESH",
    "TRIGGER_PROJECTION_TO_LIMIT_PROOF",
}
TAIL_MAX_BYTES = 18 * 1024 * 1024
TAIL_MAX_LINES = 12000


@dataclass(frozen=True)
class ForecastPatternRefreshSummary:
    status: str
    output_path: Path
    report_path: Path
    target_lane_id: str | None
    live_permission_allowed: bool


class ForecastPatternRefresh:
    """Consume entry-frequency forecast refresh work into concrete range/trigger proof actions."""

    def __init__(
        self,
        *,
        entry_frequency_recovery_path: Path = DEFAULT_ENTRY_FREQUENCY_RECOVERY,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        output_path: Path = DEFAULT_FORECAST_PATTERN_REFRESH,
        report_path: Path = DEFAULT_FORECAST_PATTERN_REFRESH_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "entry_frequency_recovery": entry_frequency_recovery_path,
            "active_trader_contract": active_trader_contract_path,
            "active_opportunity_board": active_opportunity_board_path,
            "order_intents": order_intents_path,
        }
        self.forecast_history_path = forecast_history_path
        self.projection_ledger_path = projection_ledger_path
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> ForecastPatternRefreshSummary:
        payload = self.build_payload()
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("forecast-pattern refresh must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("forecast-pattern refresh must not record live side effects")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
        return ForecastPatternRefreshSummary(
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
            entry_frequency_recovery=artifacts["entry_frequency_recovery"],
            active_contract=artifacts["active_trader_contract"],
            active_board=artifacts["active_opportunity_board"],
        )
        pairs = [str(target.get("pair"))] if target.get("pair") else []
        forecast_rows, forecast_scan = _tail_jsonl_rows(self.forecast_history_path, pairs=pairs)
        projection_rows, projection_scan = _tail_jsonl_rows(self.projection_ledger_path, pairs=pairs)
        target_lane = _analyze_target(
            target,
            order_intents=order_intents,
            forecast_rows=forecast_rows,
            projection_rows=projection_rows,
            now_utc=self.now_utc,
        )
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
            "source_artifacts": _source_artifacts(
                artifacts=artifacts,
                forecast_history_path=self.forecast_history_path,
                projection_ledger_path=self.projection_ledger_path,
                forecast_scan=forecast_scan,
                projection_scan=projection_scan,
            ),
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


def _select_target(
    *,
    entry_frequency_recovery: dict[str, Any],
    active_contract: dict[str, Any],
    active_board: dict[str, Any],
) -> dict[str, Any]:
    entry_top = entry_frequency_recovery.get("top_lane") if isinstance(entry_frequency_recovery.get("top_lane"), dict) else {}
    if entry_top and _has_forecast_refresh_action(entry_frequency_recovery, str(entry_top.get("lane_id") or "")):
        return _lane_base(entry_top)
    contract_state = active_contract.get("current_state") if isinstance(active_contract.get("current_state"), dict) else {}
    contract_board = (
        contract_state.get("active_opportunity_board")
        if isinstance(contract_state.get("active_opportunity_board"), dict)
        else {}
    )
    for source in (contract_board, active_board):
        top = source.get("top_lane") if isinstance(source.get("top_lane"), dict) else {}
        if top and (
            RANGE_FORECAST_BLOCKER in _string_list(top.get("blockers"))
            or ENTRY_DROUGHT_BLOCKER in _string_list(top.get("blockers"))
        ):
            return _lane_base(top)
    return {}


def _has_forecast_refresh_action(entry_frequency_recovery: dict[str, Any], lane_id: str) -> bool:
    if not lane_id:
        return False
    for action in _list(entry_frequency_recovery.get("forecast_pattern_tuning_queue")):
        if not isinstance(action, dict):
            continue
        if action.get("lane_id") == lane_id and action.get("action_type") in FORECAST_PATTERN_TARGET_ACTIONS:
            return True
    return False


def _lane_base(lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_id": str(lane.get("lane_id") or ""),
        "pair": str(lane.get("pair") or "UNKNOWN"),
        "direction": str(lane.get("direction") or lane.get("side") or "UNKNOWN"),
        "strategy_family": str(lane.get("strategy_family") or lane.get("method") or "UNKNOWN"),
        "vehicle": str(lane.get("vehicle") or lane.get("order_type") or "UNKNOWN"),
        "status": str(lane.get("status") or "EVIDENCE_ACQUISITION"),
        "blockers": _string_list(lane.get("blockers")),
        "entry_recovery_history": (
            dict(lane.get("entry_recovery_history"))
            if isinstance(lane.get("entry_recovery_history"), dict)
            else {}
        ),
        "forecast_audit": dict(lane.get("forecast_audit")) if isinstance(lane.get("forecast_audit"), dict) else {},
        "projection_audit": dict(lane.get("projection_audit")) if isinstance(lane.get("projection_audit"), dict) else {},
        "strategy_profile_audit": (
            dict(lane.get("strategy_profile_audit")) if isinstance(lane.get("strategy_profile_audit"), dict) else {}
        ),
        "tp_proof_audit": dict(lane.get("tp_proof_audit")) if isinstance(lane.get("tp_proof_audit"), dict) else {},
    }


def _tail_jsonl_rows(path: Path, *, pairs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {"path": str(path), "status": "missing", "rows_scanned": 0, "rows_matched": 0}
    size = path.stat().st_size
    read_size = min(size, TAIL_MAX_BYTES)
    with path.open("rb") as handle:
        if size > read_size:
            handle.seek(size - read_size)
            handle.readline()
        raw = handle.read()
    pair_set = set(pairs)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for line in raw.decode("utf-8", errors="ignore").splitlines()[-TAIL_MAX_LINES:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        scanned += 1
        if pair_set and str(payload.get("pair") or "") not in pair_set:
            continue
        rows.append(payload)
    return rows, {
        "path": str(path),
        "status": "present",
        "size_bytes": size,
        "tail_bytes_read": read_size,
        "rows_scanned": scanned,
        "rows_matched": len(rows),
        "pair_filter": pairs,
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


def _analyze_target(
    target: dict[str, Any],
    *,
    order_intents: dict[str, dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    now_utc: datetime,
) -> dict[str, Any]:
    if not target:
        return {}
    pair = str(target.get("pair") or "UNKNOWN")
    direction = str(target.get("direction") or "UNKNOWN")
    range_limit_id = f"range_trader:{pair}:{direction}:RANGE_ROTATION"
    range_market_id = f"{range_limit_id}:MARKET"
    range_limit = _intent_summary(order_intents.get(range_limit_id) or {}, fallback_lane_id=range_limit_id)
    range_market = _intent_summary(order_intents.get(range_market_id) or {}, fallback_lane_id=range_market_id)
    original_intent = _intent_summary(order_intents.get(str(target.get("lane_id") or "")) or {})
    forecast = _forecast_box_audit(
        target=target,
        rows=[row for row in forecast_rows if str(row.get("pair") or "") == pair],
    )
    projection = _projection_trigger_audit(
        pair=pair,
        direction=direction,
        rows=[row for row in projection_rows if str(row.get("pair") or "") == pair],
        now_utc=now_utc,
    )
    range_blockers = _unique(range_limit.get("blocker_codes", []) + range_market.get("blocker_codes", []))
    blocker_codes = _unique(_string_list(target.get("blockers")) + range_blockers)
    counterpart_status = _counterpart_status(range_limit=range_limit, range_market=range_market, forecast=forecast)
    return {
        **target,
        "original_intent": original_intent,
        "range_rotation_counterpart": {
            "limit": range_limit,
            "market": range_market,
            "status": counterpart_status,
            "preferred_lane_id": range_limit_id if range_limit.get("present") else None,
            "blocker_codes": range_blockers,
        },
        "forecast_range_box": forecast,
        "projection_trigger_audit": projection,
        "blocker_preservation": {
            "negative_expectancy_visible": _contains(blocker_codes, "NEGATIVE_EXPECTANCY"),
            "spread_too_wide_not_ignored": _contains(blocker_codes, "SPREAD"),
            "bidask_negative_not_ignored": _contains(blocker_codes, "BIDASK"),
            "entry_drought_visible": ENTRY_DROUGHT_BLOCKER in blocker_codes,
            "range_rotation_location_blocker_visible": _contains(blocker_codes, "RANGE_ROTATION"),
            "live_permission_allowed": False,
        },
    }


def _intent_summary(row: dict[str, Any], *, fallback_lane_id: str | None = None) -> dict[str, Any]:
    if not row:
        return {
            "present": False,
            "lane_id": fallback_lane_id,
            "status": "MISSING",
            "blocker_codes": [],
            "issue_messages": [],
        }
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    blockers = _unique(
        _string_list(row.get("live_blocker_codes"))
        + _issue_codes(row.get("risk_issues"))
        + _issue_codes(row.get("strategy_issues"))
    )
    return {
        "present": True,
        "lane_id": row.get("lane_id") or fallback_lane_id,
        "status": row.get("status"),
        "risk_allowed": bool(row.get("risk_allowed")),
        "blocker_codes": blockers,
        "order_type": intent.get("order_type"),
        "units": _number_or_none(intent.get("units")),
        "entry_price": _number_or_none(intent.get("entry_price")),
        "take_profit_price": _number_or_none(intent.get("take_profit_price")),
        "stop_loss_price": _number_or_none(intent.get("stop_loss_price")),
        "issue_messages": _issue_messages(row.get("risk_issues"))[:8],
    }


def _forecast_box_audit(*, target: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_forecast = target.get("forecast_audit") if isinstance(target.get("forecast_audit"), dict) else {}
    latest = top_forecast.get("latest") if isinstance(top_forecast.get("latest"), dict) else {}
    if not latest:
        parsed_rows = [(row, _parse_utc(row.get("timestamp_utc"))) for row in rows]
        parsed_rows = [(row, ts) for row, ts in parsed_rows if ts is not None]
        row, ts = max(parsed_rows, key=lambda item: item[1], default=({}, None))
        latest = _forecast_public_row(row, ts)
    current = _float(latest.get("current_price"))
    low = _float(latest.get("range_low_price"))
    high = _float(latest.get("range_high_price"))
    direction = str(target.get("direction") or "").upper()
    position = None
    if current is not None and low is not None and high is not None and high > low:
        position = round((current - low) / (high - low), 4)
    status = "RANGE_BOX_MISSING"
    action = "refresh forecast range_low/range_high before retuning a range-rotation lane"
    if str(latest.get("direction") or "") == "RANGE" and position is not None:
        if direction == "LONG" and position <= 0.35:
            status = "RANGE_BOX_AT_LONG_DISCOUNT_RAIL"
            action = "range box is near the LONG discount rail; next blockers are spread, proof, and gateway validation"
        elif direction == "SHORT" and position >= 0.65:
            status = "RANGE_BOX_AT_SHORT_PREMIUM_RAIL"
            action = "range box is near the SHORT premium rail; next blockers are spread, proof, and gateway validation"
        else:
            status = "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL"
            action = (
                "wait for broad discount/lower-half rail for LONG or premium/upper-half rail for SHORT; "
                "do not chase the current range midpoint or opposite rail"
            )
    elif latest:
        status = "LATEST_FORECAST_NOT_RANGE"
        action = "forecast is no longer RANGE; rebuild entry_frequency_recovery before changing pattern selection"
    return {
        "status": status,
        "latest": latest,
        "box_position": position,
        "recommended_action": action,
    }


def _projection_trigger_audit(
    *,
    pair: str,
    direction: str,
    rows: list[dict[str, Any]],
    now_utc: datetime,
) -> dict[str, Any]:
    desired = _desired_forecast_direction(direction)
    parsed_rows = [(row, _parse_utc(row.get("timestamp_emitted_utc"))) for row in rows]
    parsed_rows = [(row, ts) for row, ts in parsed_rows if ts is not None]
    desired_rows = [(row, ts) for row, ts in parsed_rows if str(row.get("direction") or "") == desired]
    expired_pending = 0
    for row, ts in parsed_rows:
        if str(row.get("resolution_status") or "") != "PENDING":
            continue
        window = _float(row.get("resolution_window_min")) or 0.0
        if (now_utc - ts).total_seconds() / 60.0 >= window:
            expired_pending += 1
    status_counts = Counter(str(row.get("resolution_status") or "UNKNOWN") for row, _ in parsed_rows)
    status = "NO_TRIGGER_PROJECTION_ROWS"
    action = "record trigger projection evidence before treating a RANGE forecast as actionable"
    if desired_rows and expired_pending:
        status = "TRIGGER_PROJECTIONS_EXPIRED_PENDING_VERIFICATION_REQUIRED"
        action = "run verify-projections, then promote only HIT trigger rows into LIMIT/TP proof; do not infer hits"
    elif desired_rows:
        status = "TRIGGER_PROJECTIONS_PRESENT"
        action = "use side-specific trigger projections as proof-collection candidates, not live permission"
    elif parsed_rows:
        status = "PROJECTION_ROWS_PRESENT_NO_SIDE_TRIGGER"
    return {
        "status": status,
        "pair": pair,
        "desired_direction": desired,
        "rows_in_tail": len(parsed_rows),
        "desired_direction_rows_tail": len(desired_rows),
        "expired_pending_in_tail": expired_pending,
        "status_counts_tail": dict(status_counts),
        "latest_desired_direction_signals": [
            _projection_public_row(row, ts) for row, ts in sorted(desired_rows, key=lambda item: item[1], reverse=True)[:5]
        ],
        "recommended_action": action,
    }


def _counterpart_status(
    *,
    range_limit: dict[str, Any],
    range_market: dict[str, Any],
    forecast: dict[str, Any],
) -> str:
    if not range_limit.get("present") and not range_market.get("present"):
        return "RANGE_ROTATION_COUNTERPART_MISSING"
    blockers = _unique(range_limit.get("blocker_codes", []) + range_market.get("blocker_codes", []))
    if forecast.get("status") == "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL":
        return "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL"
    if any(code in blockers for code in ("RANGE_ROTATION_BROADER_LOCATION_CHASE", "EXHAUSTION_RANGE_CHASE")):
        return "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_LOCATION_OR_EXHAUSTION"
    if any("SPREAD" in code for code in blockers):
        return "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_SPREAD"
    if any("BIDASK" in code or "NEGATIVE_EXPECTANCY" in code for code in blockers):
        return "RANGE_ROTATION_COUNTERPART_PROOF_BLOCKED"
    if range_limit.get("risk_allowed") or range_market.get("risk_allowed"):
        return "RANGE_ROTATION_COUNTERPART_RISK_READY_DRY_RUN_ONLY"
    return "RANGE_ROTATION_COUNTERPART_PRESENT_STILL_BLOCKED"


def _next_actions(target_lane: dict[str, Any]) -> list[dict[str, Any]]:
    if not target_lane:
        return []
    lane_id = str(target_lane.get("lane_id") or "")
    actions: list[dict[str, Any]] = []
    forecast = target_lane.get("forecast_range_box") if isinstance(target_lane.get("forecast_range_box"), dict) else {}
    counterpart = (
        target_lane.get("range_rotation_counterpart")
        if isinstance(target_lane.get("range_rotation_counterpart"), dict)
        else {}
    )
    projection = (
        target_lane.get("projection_trigger_audit")
        if isinstance(target_lane.get("projection_trigger_audit"), dict)
        else {}
    )
    if counterpart.get("status") in {
        "RANGE_ROTATION_COUNTERPART_MISSING",
        "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL",
        "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_LOCATION_OR_EXHAUSTION",
    }:
        actions.append(
            _action(
                lane_id,
                "RANGE_RAIL_GEOMETRY_REPAIR",
                forecast.get("recommended_action")
                or "build executable range rail geometry before collecting live proof",
                blockers=["RANGE_ROTATION_BROADER_LOCATION_CHASE", "EXHAUSTION_RANGE_CHASE", RANGE_FORECAST_BLOCKER],
                priority=1,
            )
        )
    if projection.get("status") == "TRIGGER_PROJECTIONS_EXPIRED_PENDING_VERIFICATION_REQUIRED":
        actions.append(
            _action(
                lane_id,
                "VERIFY_TRIGGER_PROJECTIONS",
                projection.get("recommended_action"),
                blockers=["EXPIRED_PENDING_PROJECTIONS"],
                priority=2,
            )
        )
    tp = target_lane.get("tp_proof_audit") if isinstance(target_lane.get("tp_proof_audit"), dict) else {}
    if tp.get("status") in {"TP_PROOF_FLOOR_GAP", "TP_PROOF_MISSING", "TP_PROOF_COLLECTION_REQUIRED"}:
        actions.append(
            _action(
                lane_id,
                "EXACT_TP_PROOF_COLLECTION",
                "collect exact attached TAKE_PROFIT_ORDER proof for the selected pair/side/method/vehicle; do not mix market-close losses",
                blockers=[TP_PROOF_BLOCKER, NEGATIVE_EXPECTANCY_BLOCKER],
                priority=3,
            )
        )
    if counterpart.get("status") in {
        "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_SPREAD",
        "RANGE_ROTATION_COUNTERPART_PROOF_BLOCKED",
        "RANGE_ROTATION_COUNTERPART_PRESENT_STILL_BLOCKED",
    }:
        actions.append(
            _action(
                lane_id,
                "PRESERVE_SPREAD_AND_EXPECTANCY_BLOCKERS",
                "keep spread, bid/ask, and negative-expectancy blockers visible while collecting lane-local proof",
                blockers=["SPREAD_TOO_WIDE", "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", NEGATIVE_EXPECTANCY_BLOCKER],
                priority=8,
            )
        )
    if not actions:
        actions.append(
            _action(
                lane_id,
                "KEEP_BLOCKED_WITH_CAUSE",
                "forecast-pattern refresh found no narrower safe action; keep current blockers and rerun after fresh market artifacts",
                blockers=_string_list(target_lane.get("blockers"))[:5],
                priority=9,
            )
        )
    return actions


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


def _next_contract_prompt(target_lane: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    if not target_lane:
        return (
            "No forecast-pattern refresh target is current. Rebuild entry_frequency_recovery from the latest "
            "active_opportunity_board; do not send."
        )
    first = actions[0] if actions else {}
    counterpart = target_lane.get("range_rotation_counterpart") or {}
    forecast = target_lane.get("forecast_range_box") or {}
    return (
        "Consume data/forecast_pattern_refresh.json for "
        f"{target_lane.get('lane_id')}: next safe action is {first.get('action_type') or 'KEEP_BLOCKED_WITH_CAUSE'}; "
        f"range_counterpart_status={counterpart.get('status')}; "
        f"forecast_box_status={forecast.get('status')}; "
        f"{first.get('description') or 'keep blockers visible'}. Do not send, cancel, close, relax gates, "
        "or treat forecast refresh as live permission."
    )


def _status(target_lane: dict[str, Any]) -> str:
    if not target_lane:
        return "NO_FORECAST_PATTERN_REFRESH_TARGET"
    forecast = target_lane.get("forecast_range_box") if isinstance(target_lane.get("forecast_range_box"), dict) else {}
    counterpart = (
        target_lane.get("range_rotation_counterpart")
        if isinstance(target_lane.get("range_rotation_counterpart"), dict)
        else {}
    )
    if forecast.get("status") in {"RANGE_BOX_MISSING", "LATEST_FORECAST_NOT_RANGE"}:
        return "FORECAST_PATTERN_REFRESH_DATA_INCOMPLETE"
    if counterpart.get("status") == "RANGE_ROTATION_COUNTERPART_MISSING":
        return "FORECAST_PATTERN_REFRESH_COUNTERPART_MISSING"
    return "FORECAST_PATTERN_REFRESH_BUILT"


def _source_artifacts(
    *,
    artifacts: dict[str, dict[str, Any]],
    forecast_history_path: Path,
    projection_ledger_path: Path,
    forecast_scan: dict[str, Any],
    projection_scan: dict[str, Any],
) -> dict[str, Any]:
    index = {
        name: {
            "path": artifact.get("_path"),
            "status": artifact.get("_artifact_status"),
            "sha256": artifact.get("_sha256"),
            "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
        }
        for name, artifact in artifacts.items()
    }
    index["forecast_history"] = forecast_scan
    index["projection_ledger"] = projection_scan
    index["forecast_history_path"] = str(forecast_history_path)
    index["projection_ledger_path"] = str(projection_ledger_path)
    return index


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Forecast Pattern Refresh",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        "",
        "## Target",
        "",
    ]
    top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
    if top:
        counterpart = top.get("range_rotation_counterpart") if isinstance(top.get("range_rotation_counterpart"), dict) else {}
        forecast = top.get("forecast_range_box") if isinstance(top.get("forecast_range_box"), dict) else {}
        projection = top.get("projection_trigger_audit") if isinstance(top.get("projection_trigger_audit"), dict) else {}
        lines.extend(
            [
                f"- Lane: `{top.get('lane_id')}`",
                f"- Shape: `{top.get('pair')}` / `{top.get('direction')}` / `{top.get('strategy_family')}` / `{top.get('vehicle')}`",
                f"- Range counterpart: `{counterpart.get('status')}` preferred `{counterpart.get('preferred_lane_id')}`",
                f"- Forecast box: `{forecast.get('status')}` position `{forecast.get('box_position')}`",
                f"- Trigger projections: `{projection.get('status')}` desired `{projection.get('desired_direction_rows_tail')}` expired `{projection.get('expired_pending_in_tail')}`",
            ]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    for action in payload.get("next_actions") or []:
        lines.append(
            f"- `{action.get('priority')}` `{action.get('action_type')}` `{action.get('lane_id')}`: "
            f"{action.get('description')}"
        )
    if not payload.get("next_actions"):
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Contract Prompt",
            "",
            str(payload.get("next_contract_prompt") or ""),
            "",
            "## Safety",
            "",
            "This artifact is read-only. It does not authorize live orders, cancels, closes, launchd changes, gate relaxation, 4x deficit lot backsolve, secret disclosure, or inferred operator approval.",
            "",
        ]
    )
    return "\n".join(lines)


def _forecast_public_row(row: dict[str, Any], ts: datetime | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "timestamp_utc": ts.isoformat() if ts else row.get("timestamp_utc"),
        "cycle_id": row.get("cycle_id"),
        "direction": row.get("direction"),
        "confidence": _number_or_none(row.get("confidence")),
        "raw_confidence": _number_or_none(row.get("raw_confidence")),
        "calibration_multiplier": _number_or_none(row.get("calibration_multiplier")),
        "horizon_min": _number_or_none(row.get("horizon_min")),
        "current_price": _number_or_none(row.get("current_price")),
        "target_price": _number_or_none(row.get("target_price")),
        "invalidation_price": _number_or_none(row.get("invalidation_price")),
        "range_low_price": _number_or_none(row.get("range_low_price")),
        "range_high_price": _number_or_none(row.get("range_high_price")),
        "up_score": _number_or_none(row.get("up_score")),
        "down_score": _number_or_none(row.get("down_score")),
        "range_score": _number_or_none(row.get("range_score")),
        "rationale_summary": row.get("rationale_summary"),
    }


def _projection_public_row(row: dict[str, Any], ts: datetime | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "timestamp_emitted_utc": ts.isoformat() if ts else row.get("timestamp_emitted_utc"),
        "cycle_id": row.get("cycle_id"),
        "signal_name": row.get("signal_name"),
        "direction": row.get("direction"),
        "confidence": _number_or_none(row.get("confidence")),
        "lead_time_min": _number_or_none(row.get("lead_time_min")),
        "resolution_window_min": _number_or_none(row.get("resolution_window_min")),
        "resolution_status": row.get("resolution_status"),
        "entry_price": _number_or_none(row.get("entry_price")),
        "predicted_target_price": _number_or_none(row.get("predicted_target_price")),
        "predicted_invalidation_price": _number_or_none(row.get("predicted_invalidation_price")),
    }


def _desired_forecast_direction(side: str) -> str:
    side = str(side or "").upper()
    if side == "LONG":
        return "UP"
    if side == "SHORT":
        return "DOWN"
    return "UNKNOWN"


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item.get("code") if isinstance(item, dict) and item.get("code") else item) for item in value if item]
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    if value in (None, ""):
        return []
    return [str(value)]


def _issue_codes(value: Any) -> list[str]:
    result: list[str] = []
    for item in _list(value):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "").upper()
        if severity and severity not in {"BLOCK", "ERROR"}:
            continue
        code = item.get("code")
        if code:
            result.append(str(code))
    return result


def _issue_messages(value: Any) -> list[str]:
    result: list[str] = []
    for item in _list(value):
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        message = str(item.get("message") or "")
        if code or message:
            result.append(f"{code}: {message}".strip(": "))
    return result


def _contains(values: list[str], needle: str) -> bool:
    return any(needle in value for value in values)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_or_none(value: Any) -> int | float | None:
    num = _float(value)
    if num is None:
        return None
    if num.is_integer():
        return int(num)
    return round(num, 6)


def _do_not_do() -> list[str]:
    return [
        "do_not_send_live_order",
        "do_not_cancel_or_close",
        "do_not_launchd_load_reload",
        "do_not_mutate_broker_state",
        "do_not_relax_gates",
        "do_not_hide_negative_expectancy_spread_or_bidask_negative",
        "do_not_mix_market_close_losses_into_tp_proof",
        "do_not_backsolve_lot_from_4x_deficit",
        "do_not_print_secrets_tokens_credentials",
        "do_not_infer_operator_decision",
    ]
