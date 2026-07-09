from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_ENTRY_FREQUENCY_RECOVERY,
    DEFAULT_ENTRY_FREQUENCY_RECOVERY_REPORT,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_STRATEGY_PROFILE,
)


SCHEMA_VERSION = "entry_frequency_recovery_v1"
ENTRY_DROUGHT_BLOCKER = "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH"
RANGE_FORECAST_BLOCKER = "RANGE_FORECAST_REQUIRES_RANGE_ROTATION"
TP_PROOF_BLOCKER = "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"
NEGATIVE_EXPECTANCY_BLOCKER = "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
TAIL_MAX_BYTES = 18 * 1024 * 1024
TAIL_MAX_LINES = 12000


@dataclass(frozen=True)
class EntryFrequencyRecoverySummary:
    status: str
    output_path: Path
    report_path: Path
    target_lane_id: str | None
    live_permission_allowed: bool


class EntryFrequencyRecovery:
    """Build read-only lane-local diagnosis for profitable lanes that stopped entering."""

    def __init__(
        self,
        *,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        non_eurusd_live_grade_frontier_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        output_path: Path = DEFAULT_ENTRY_FREQUENCY_RECOVERY,
        report_path: Path = DEFAULT_ENTRY_FREQUENCY_RECOVERY_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "active_trader_contract": active_trader_contract_path,
            "active_opportunity_board": active_opportunity_board_path,
            "non_eurusd_live_grade_frontier": non_eurusd_live_grade_frontier_path,
            "order_intents": order_intents_path,
            "strategy_profile": strategy_profile_path,
        }
        self.execution_ledger_db_path = execution_ledger_db_path
        self.forecast_history_path = forecast_history_path
        self.projection_ledger_path = projection_ledger_path
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> EntryFrequencyRecoverySummary:
        payload = self.build_payload()
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("entry-frequency recovery must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("entry-frequency recovery must not record live side effects")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        top = payload.get("target_lanes", [{}])[0] if payload.get("target_lanes") else {}
        return EntryFrequencyRecoverySummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            target_lane_id=top.get("lane_id"),
            live_permission_allowed=False,
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        targets = _collect_entry_recovery_targets(
            active_contract=artifacts["active_trader_contract"],
            active_board=artifacts["active_opportunity_board"],
        )
        pairs = sorted({str(lane.get("pair") or "") for lane in targets if lane.get("pair")})
        forecast_rows, forecast_scan = _tail_jsonl_rows(self.forecast_history_path, pairs=pairs)
        projection_rows, projection_scan = _tail_jsonl_rows(self.projection_ledger_path, pairs=pairs)
        order_intents = _order_intents_by_lane(artifacts["order_intents"])
        strategy_profiles = _strategy_profile_index(artifacts["strategy_profile"])
        execution_summary = _execution_lane_summary(
            self.execution_ledger_db_path,
            lane_ids=[str(lane.get("lane_id") or "") for lane in targets],
        )
        frontier_lane = _frontier_evidence_lane(artifacts["non_eurusd_live_grade_frontier"])

        analyzed_lanes = [
            _analyze_lane(
                lane,
                order_intents=order_intents,
                strategy_profiles=strategy_profiles,
                forecast_rows=forecast_rows,
                projection_rows=projection_rows,
                execution_summary=execution_summary.get(str(lane.get("lane_id") or ""), {}),
                frontier_lane=frontier_lane,
                now_utc=self.now_utc,
            )
            for lane in targets
        ]
        tuning_queue = _tuning_queue(analyzed_lanes)
        status = _status(analyzed_lanes)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "target_lanes": analyzed_lanes,
            "target_lane_count": len(analyzed_lanes),
            "top_lane": analyzed_lanes[0] if analyzed_lanes else {},
            "forecast_pattern_tuning_queue": tuning_queue,
            "next_contract_prompt": _next_contract_prompt(analyzed_lanes, tuning_queue),
            "source_artifacts": _source_artifacts(
                artifacts=artifacts,
                execution_ledger_db_path=self.execution_ledger_db_path,
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


def _collect_entry_recovery_targets(
    *,
    active_contract: dict[str, Any],
    active_board: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    current = active_contract.get("current_state") if isinstance(active_contract.get("current_state"), dict) else {}
    contract_board = (
        current.get("active_opportunity_board")
        if isinstance(current.get("active_opportunity_board"), dict)
        else {}
    )
    for source in (contract_board, active_board):
        top = source.get("top_lane") if isinstance(source.get("top_lane"), dict) else {}
        if top:
            candidates.append(top)
        summary = source.get("entry_recovery_summary") if isinstance(source.get("entry_recovery_summary"), dict) else {}
        for row in _list(summary.get("top_candidates")):
            if isinstance(row, dict):
                candidates.append(row)
        for row in _list(source.get("ranked_active_lanes"))[:25]:
            if isinstance(row, dict):
                candidates.append(row)

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lane in candidates:
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id or lane_id in seen:
            continue
        blockers = _string_list(lane.get("blockers"))
        if lane.get("entry_recovery_candidate") is not True and ENTRY_DROUGHT_BLOCKER not in blockers:
            continue
        result.append(_public_lane_base(lane))
        seen.add(lane_id)
    return sorted(
        result,
        key=lambda lane: (
            0 if lane.get("lane_id") == _top_lane_id(active_contract) else 1,
            -float(((lane.get("entry_recovery_history") or {}).get("closed_pl_jpy") or 0.0)),
            str(lane.get("lane_id") or ""),
        ),
    )[:8]


def _top_lane_id(active_contract: dict[str, Any]) -> str:
    current = active_contract.get("current_state") if isinstance(active_contract.get("current_state"), dict) else {}
    board = current.get("active_opportunity_board") if isinstance(current.get("active_opportunity_board"), dict) else {}
    top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    return str(top.get("lane_id") or "")


def _public_lane_base(lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_id": str(lane.get("lane_id") or ""),
        "pair": str(lane.get("pair") or "UNKNOWN"),
        "direction": str(lane.get("direction") or lane.get("side") or "UNKNOWN"),
        "strategy_family": str(lane.get("strategy_family") or lane.get("method") or "UNKNOWN"),
        "vehicle": str(lane.get("vehicle") or lane.get("order_type") or "UNKNOWN"),
        "status": str(lane.get("status") or "NO_TRADE_WITH_CAUSE"),
        "expected_edge_jpy": _number_or_none(lane.get("expected_edge_jpy")),
        "next_action": str(lane.get("next_action") or ""),
        "blockers": _string_list(lane.get("blockers")),
        "entry_recovery_history": (
            dict(lane.get("entry_recovery_history"))
            if isinstance(lane.get("entry_recovery_history"), dict)
            else {}
        ),
        "local_tp_proof": dict(lane.get("local_tp_proof")) if isinstance(lane.get("local_tp_proof"), dict) else {},
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


def _strategy_profile_index(artifact: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    profiles = {}
    for row in _list(artifact.get("profiles")):
        if not isinstance(row, dict):
            continue
        method = str(row.get("method") or "UNKNOWN")
        profiles[(str(row.get("pair") or ""), str(row.get("direction") or ""), method)] = row
    return profiles


def _execution_lane_summary(path: Path, *, lane_ids: list[str]) -> dict[str, dict[str, Any]]:
    lane_set = {lane_id for lane_id in lane_ids if lane_id}
    if not path.exists() or not lane_set:
        return {}
    placeholders = ",".join("?" for _ in lane_set)
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                f"""
                select lane_id,event_type,ts_utc,realized_pl_jpy
                from execution_events
                where lane_id in ({placeholders})
                  and event_type in ('ORDER_ACCEPTED','ORDER_FILLED','TRADE_CLOSED')
                order by ts_utc asc
                """,
                tuple(lane_set),
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        lane_id = str(row["lane_id"] or "")
        bucket = result.setdefault(
            lane_id,
            {"accepted": 0, "fills": 0, "closed": 0, "closed_pl_jpy": 0.0, "last_event_at_utc": None},
        )
        event_type = str(row["event_type"] or "")
        if event_type == "ORDER_ACCEPTED":
            bucket["accepted"] += 1
        elif event_type == "ORDER_FILLED":
            bucket["fills"] += 1
        elif event_type == "TRADE_CLOSED":
            bucket["closed"] += 1
            bucket["closed_pl_jpy"] = round(float(bucket["closed_pl_jpy"]) + (_float(row["realized_pl_jpy"]) or 0.0), 4)
        bucket["last_event_at_utc"] = row["ts_utc"]
    return result


def _frontier_evidence_lane(frontier: dict[str, Any]) -> dict[str, Any]:
    checks = frontier.get("required_checks") if isinstance(frontier.get("required_checks"), dict) else {}
    lane = checks.get("next_evidence_lane") if isinstance(checks.get("next_evidence_lane"), dict) else {}
    if lane:
        return lane
    top_non = frontier.get("top_non_eurusd_lane") if isinstance(frontier.get("top_non_eurusd_lane"), dict) else {}
    return top_non


def _analyze_lane(
    lane: dict[str, Any],
    *,
    order_intents: dict[str, dict[str, Any]],
    strategy_profiles: dict[tuple[str, str, str], dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    execution_summary: dict[str, Any],
    frontier_lane: dict[str, Any],
    now_utc: datetime,
) -> dict[str, Any]:
    pair = str(lane.get("pair") or "UNKNOWN")
    direction = str(lane.get("direction") or "UNKNOWN")
    strategy = str(lane.get("strategy_family") or "UNKNOWN")
    vehicle = str(lane.get("vehicle") or "UNKNOWN")
    lane_id = str(lane.get("lane_id") or "")
    intent = order_intents.get(lane_id) or {}
    exact_profile = strategy_profiles.get((pair, direction, strategy))
    pair_side_profile = strategy_profiles.get((pair, direction, "UNKNOWN"))
    forecast = _forecast_audit(
        pair=pair,
        direction=direction,
        strategy_family=strategy,
        vehicle=vehicle,
        blockers=_string_list(lane.get("blockers")),
        rows=[row for row in forecast_rows if str(row.get("pair") or "") == pair],
        now_utc=now_utc,
    )
    projection = _projection_audit(
        pair=pair,
        direction=direction,
        rows=[row for row in projection_rows if str(row.get("pair") or "") == pair],
        now_utc=now_utc,
    )
    profile = _profile_audit(exact_profile=exact_profile, pair_side_profile=pair_side_profile)
    tp_proof = _tp_proof_audit(lane, frontier_lane)
    actions = _lane_actions(
        lane=lane,
        current_order_intent=intent,
        forecast=forecast,
        projection=projection,
        profile=profile,
        tp_proof=tp_proof,
        execution_summary=execution_summary,
    )
    return {
        **lane,
        "current_order_intent": _order_intent_summary(intent),
        "execution_ledger_crosscheck": execution_summary,
        "forecast_audit": forecast,
        "projection_audit": projection,
        "strategy_profile_audit": profile,
        "tp_proof_audit": tp_proof,
        "blocker_preservation": {
            "negative_expectancy_visible": NEGATIVE_EXPECTANCY_BLOCKER in _string_list(lane.get("blockers")),
            "spread_or_bidask_not_ignored": any(
                "SPREAD" in code or "BIDASK" in code for code in _string_list(lane.get("blockers"))
            ),
            "entry_drought_visible": ENTRY_DROUGHT_BLOCKER in _string_list(lane.get("blockers")),
            "live_permission_allowed": False,
        },
        "next_tuning_actions": actions,
    }


def _forecast_audit(
    *,
    pair: str,
    direction: str,
    strategy_family: str,
    vehicle: str,
    blockers: list[str],
    rows: list[dict[str, Any]],
    now_utc: datetime,
) -> dict[str, Any]:
    desired = _desired_forecast_direction(direction)
    parsed_rows = [(row, _parse_utc(row.get("timestamp_utc"))) for row in rows]
    parsed_rows = [(row, ts) for row, ts in parsed_rows if ts is not None]
    latest_row, latest_ts = max(parsed_rows, key=lambda item: item[1], default=({}, None))
    counts = Counter(str(row.get("direction") or "UNKNOWN") for row, _ in parsed_rows)
    recent_cutoff = now_utc - timedelta(hours=72)
    recent = [(row, ts) for row, ts in parsed_rows if ts >= recent_cutoff]
    latest_direction = str(latest_row.get("direction") or "")
    latest_age_minutes = (
        round((now_utc - latest_ts).total_seconds() / 60.0, 2) if latest_ts is not None else None
    )
    status = "FORECAST_HISTORY_MISSING_FOR_PAIR"
    if latest_ts is not None:
        if latest_age_minutes is not None and latest_age_minutes > 180:
            status = "FORECAST_HISTORY_STALE"
        elif latest_direction == desired:
            status = "SIDE_SUPPORTED_BY_LATEST_FORECAST"
        elif latest_direction == "RANGE" and RANGE_FORECAST_BLOCKER in blockers:
            status = "RANGE_FORECAST_BLOCKS_NON_RANGE_ENTRY"
        else:
            status = "SIDE_NOT_SUPPORTED_BY_LATEST_FORECAST"
    action = "refresh current forecast telemetry for this pair before changing pattern selection"
    if status == "RANGE_FORECAST_BLOCKS_NON_RANGE_ENTRY":
        action = (
            f"retune {pair} {direction} {strategy_family} {vehicle}: latest forecast is RANGE, so do not "
            "recover a MARKET non-range entry; rotate to RANGE_ROTATION geometry or collect trigger/TP proof first"
        )
    elif status == "SIDE_SUPPORTED_BY_LATEST_FORECAST":
        action = "pair forecast supports the side; next bottleneck is proof/replay/risk, not forecast direction"
    return {
        "status": status,
        "desired_direction": desired,
        "rows_in_tail": len(parsed_rows),
        "recent_rows_72h": len(recent),
        "direction_counts_tail": dict(counts),
        "matching_direction_rows_tail": counts.get(desired, 0),
        "latest": _forecast_public_row(latest_row, latest_ts),
        "latest_age_minutes": latest_age_minutes,
        "recommended_action": action,
    }


def _projection_audit(
    *,
    pair: str,
    direction: str,
    rows: list[dict[str, Any]],
    now_utc: datetime,
) -> dict[str, Any]:
    desired = _desired_forecast_direction(direction)
    parsed_rows = [(row, _parse_utc(row.get("timestamp_emitted_utc"))) for row in rows]
    parsed_rows = [(row, ts) for row, ts in parsed_rows if ts is not None]
    status_counts = Counter(str(row.get("resolution_status") or "UNKNOWN") for row, _ in parsed_rows)
    direction_counts = Counter(str(row.get("direction") or "UNKNOWN") for row, _ in parsed_rows)
    desired_rows = [(row, ts) for row, ts in parsed_rows if str(row.get("direction") or "") == desired]
    directional_rows = [(row, ts) for row, ts in parsed_rows if str(row.get("signal_name") or "") == "directional_forecast"]
    latest_directional, latest_directional_ts = max(directional_rows, key=lambda item: item[1], default=({}, None))
    latest_desired = sorted(desired_rows, key=lambda item: item[1], reverse=True)[:5]
    expired_pending = 0
    for row, ts in parsed_rows:
        if str(row.get("resolution_status") or "") != "PENDING":
            continue
        window = _float(row.get("resolution_window_min")) or 0.0
        if (now_utc - ts).total_seconds() / 60.0 >= window:
            expired_pending += 1
    signal_stats: dict[str, dict[str, Any]] = {}
    for row, _ in parsed_rows:
        name = str(row.get("signal_name") or "UNKNOWN")
        bucket = signal_stats.setdefault(name, {"total": 0, "statuses": {}, "directions": {}})
        bucket["total"] += 1
        status = str(row.get("resolution_status") or "UNKNOWN")
        row_direction = str(row.get("direction") or "UNKNOWN")
        bucket["statuses"][status] = bucket["statuses"].get(status, 0) + 1
        bucket["directions"][row_direction] = bucket["directions"].get(row_direction, 0) + 1
    action = "verify expired projections, then retune the pattern signal with actual HIT/MISS evidence"
    if latest_directional and str(latest_directional.get("direction") or "") == "RANGE" and desired_rows:
        action = (
            "directional_forecast is RANGE while side-specific trigger projections exist; recover via trigger "
            "proof/LIMIT frontier rather than MARKET execution"
        )
    return {
        "status": "PROJECTION_TAIL_SCANNED" if parsed_rows else "PROJECTION_LEDGER_MISSING_FOR_PAIR",
        "desired_direction": desired,
        "rows_in_tail": len(parsed_rows),
        "status_counts_tail": dict(status_counts),
        "direction_counts_tail": dict(direction_counts),
        "desired_direction_rows_tail": len(desired_rows),
        "expired_pending_in_tail": expired_pending,
        "latest_directional_forecast": _projection_public_row(latest_directional, latest_directional_ts),
        "latest_desired_direction_signals": [
            _projection_public_row(row, ts) for row, ts in latest_desired
        ],
        "signal_stats_tail": signal_stats,
        "recommended_action": action,
    }


def _profile_audit(
    *,
    exact_profile: dict[str, Any] | None,
    pair_side_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    exact = exact_profile if isinstance(exact_profile, dict) else {}
    pair_side = pair_side_profile if isinstance(pair_side_profile, dict) else {}
    status = "METHOD_PROFILE_PRESENT" if exact else "METHOD_PROFILE_MISSING"
    if not exact and pair_side:
        status = "PAIR_SIDE_PROFILE_PRESENT_METHOD_PROFILE_MISSING"
    action = "mine/promote method-scoped profile evidence before treating pair-side profit as exact lane proof"
    if exact and (_float(exact.get("order_blocked")) or 0.0) > 0:
        action = "inspect method profile top_block_reasons and retune the pattern/risk fields that still block orders"
    elif exact:
        action = "method profile exists; focus on forecast/proof/replay blockers"
    return {
        "status": status,
        "exact_method_profile": _profile_public_row(exact),
        "pair_side_profile": _profile_public_row(pair_side),
        "recommended_action": action,
    }


def _tp_proof_audit(lane: dict[str, Any], frontier_lane: dict[str, Any]) -> dict[str, Any]:
    proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
    frontier_matches = _same_shape(lane, frontier_lane)
    frontier_tp_count = frontier_lane.get("tp_proof_count") if frontier_matches else None
    frontier_tp_floor = frontier_lane.get("tp_proof_floor") if frontier_matches else None
    trades = _number_or_none(proof.get("capture_take_profit_trades"))
    floor = _number_or_none(proof.get("capture_take_profit_proof_floor"))
    if trades is None:
        trades = _number_or_none(frontier_tp_count)
    if floor is None:
        floor = _number_or_none(frontier_tp_floor)
    remaining = None
    if trades is not None and floor is not None:
        remaining = max(int(floor) - int(trades), 0)
    status = "TP_PROOF_FLOOR_MET" if remaining == 0 else "TP_PROOF_FLOOR_GAP"
    if trades is None:
        status = "TP_PROOF_MISSING"
    return {
        "status": status,
        "scope_key": proof.get("capture_take_profit_scope_key"),
        "tp_trades": trades,
        "tp_losses": _number_or_none(proof.get("capture_take_profit_losses")),
        "tp_expectancy_jpy": _number_or_none(proof.get("capture_take_profit_expectancy_jpy")),
        "tp_proof_floor": floor,
        "remaining_tp_trades": remaining,
        "frontier_same_shape": frontier_matches,
        "recommended_action": (
            "collect exact attached TAKE_PROFIT_ORDER samples for the same pair/side/method/vehicle; "
            "do not mix market-close losses into TP proof"
        ),
    }


def _lane_actions(
    *,
    lane: dict[str, Any],
    current_order_intent: dict[str, Any],
    forecast: dict[str, Any],
    projection: dict[str, Any],
    profile: dict[str, Any],
    tp_proof: dict[str, Any],
    execution_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    lane_id = str(lane.get("lane_id") or "")
    blockers = _string_list(lane.get("blockers"))
    actions: list[dict[str, Any]] = []
    if forecast.get("status") == "RANGE_FORECAST_BLOCKS_NON_RANGE_ENTRY":
        actions.append(
            _action(
                lane_id,
                "FORECAST_PATTERN_REFRESH",
                forecast.get("recommended_action"),
                blockers=[RANGE_FORECAST_BLOCKER],
                priority=1,
            )
        )
    if projection.get("latest_desired_direction_signals"):
        actions.append(
            _action(
                lane_id,
                "TRIGGER_PROJECTION_TO_LIMIT_PROOF",
                projection.get("recommended_action"),
                blockers=[RANGE_FORECAST_BLOCKER],
                priority=2,
            )
        )
    if str(profile.get("status") or "") == "PAIR_SIDE_PROFILE_PRESENT_METHOD_PROFILE_MISSING":
        actions.append(
            _action(
                lane_id,
                "METHOD_SCOPED_PROFILE_PROMOTION",
                profile.get("recommended_action"),
                blockers=["STRATEGY_PROFILE_METHOD_SCOPE_MISSING"],
                priority=3,
            )
        )
    if tp_proof.get("status") in {"TP_PROOF_FLOOR_GAP", "TP_PROOF_MISSING"}:
        actions.append(
            _action(
                lane_id,
                "EXACT_TP_PROOF_COLLECTION",
                tp_proof.get("recommended_action"),
                blockers=[TP_PROOF_BLOCKER, NEGATIVE_EXPECTANCY_BLOCKER],
                priority=4,
            )
        )
    if not current_order_intent:
        actions.append(
            _action(
                lane_id,
                "CURRENT_INTENT_REGEN_REQUIRED",
                "regenerate order_intents after forecast/profile proof changes; do not synthesize a live order",
                blockers=["ENTRY_DROUGHT_RECOVERY_REQUIRES_CURRENT_INTENT"],
                priority=5,
            )
        )
    if not actions:
        actions.append(
            _action(
                lane_id,
                "KEEP_BLOCKED_WITH_CAUSE",
                "entry drought is visible but no narrower safe tuning action was proven from local artifacts",
                blockers=blockers[:5],
                priority=9,
            )
        )
    for action in actions:
        action["execution_history"] = execution_summary or lane.get("entry_recovery_history") or {}
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


def _order_intent_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {"present": False}
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    return {
        "present": True,
        "status": row.get("status"),
        "risk_allowed": bool(row.get("risk_allowed")),
        "live_blocker_codes": _string_list(row.get("live_blocker_codes")),
        "strategy_issue_codes": _issue_codes(row.get("strategy_issues")),
        "risk_issue_codes": _issue_codes(row.get("risk_issues")),
        "method": intent.get("method") or market_context.get("method"),
        "order_type": intent.get("order_type"),
        "event_risk": market_context.get("event_risk"),
        "narrative_excerpt": str(market_context.get("narrative") or "")[:360],
        "chart_story_excerpt": str(market_context.get("chart_story") or "")[:360],
    }


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
        "regime_at_emission": row.get("regime_at_emission"),
        "entry_price": _number_or_none(row.get("entry_price")),
        "predicted_target_price": _number_or_none(row.get("predicted_target_price")),
        "predicted_invalidation_price": _number_or_none(row.get("predicted_invalidation_price")),
    }


def _profile_public_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "pair": row.get("pair"),
        "direction": row.get("direction"),
        "method": row.get("method"),
        "status": row.get("status"),
        "live_n": _number_or_none(row.get("live_n")),
        "live_net_jpy": _number_or_none(row.get("live_net_jpy")),
        "live_avg_jpy": _number_or_none(row.get("live_avg_jpy")),
        "live_worst_jpy": _number_or_none(row.get("live_worst_jpy")),
        "positive_evidence_n": _number_or_none(row.get("positive_evidence_n")),
        "positive_tail_jpy": _number_or_none(row.get("positive_tail_jpy")),
        "order_blocked": _number_or_none(row.get("order_blocked")),
        "required_fix": row.get("required_fix"),
        "top_block_reasons": _string_list(row.get("top_block_reasons"))[:5],
    }


def _tuning_queue(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = [action for lane in lanes for action in _list(lane.get("next_tuning_actions"))]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        key = (str(action.get("lane_id") or ""), str(action.get("action_type") or ""))
        if key not in unique or int(action.get("priority") or 99) < int(unique[key].get("priority") or 99):
            unique[key] = action
    return sorted(unique.values(), key=lambda row: (int(row.get("priority") or 99), str(row.get("lane_id") or "")))


def _next_contract_prompt(lanes: list[dict[str, Any]], tuning_queue: list[dict[str, Any]]) -> str:
    if not lanes:
        return (
            "No entry-frequency recovery target is current. Rebuild active_opportunity_board and "
            "non_eurusd_live_grade_frontier; do not send."
        )
    lane = lanes[0]
    lane_id = str(lane.get("lane_id") or "")
    first = next(
        (
            row
            for row in tuning_queue
            if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id
        ),
        {},
    )
    return (
        "Consume data/entry_frequency_recovery.json for "
        f"{lane.get('lane_id')}: next safe tuning action is {first.get('action_type') or 'KEEP_BLOCKED_WITH_CAUSE'}; "
        f"{first.get('description') or 'keep blockers visible'}. Do not send, cancel, close, relax gates, "
        "or treat entry drought as live permission."
    )


def _status(lanes: list[dict[str, Any]]) -> str:
    if not lanes:
        return "NO_ENTRY_FREQUENCY_RECOVERY_TARGETS"
    if any((lane.get("forecast_audit") or {}).get("status") == "FORECAST_HISTORY_MISSING_FOR_PAIR" for lane in lanes):
        return "ENTRY_FREQUENCY_RECOVERY_DATA_INCOMPLETE"
    return "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT"


def _source_artifacts(
    *,
    artifacts: dict[str, dict[str, Any]],
    execution_ledger_db_path: Path,
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
    index["execution_ledger_db"] = {
        "path": str(execution_ledger_db_path),
        "status": "present" if execution_ledger_db_path.exists() else "missing",
        "size_bytes": execution_ledger_db_path.stat().st_size if execution_ledger_db_path.exists() else None,
    }
    index["forecast_history"] = forecast_scan
    index["projection_ledger"] = projection_scan
    return index


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Entry Frequency Recovery",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Target lanes: `{payload.get('target_lane_count')}`",
        "",
        "## Top Lane",
        "",
    ]
    top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
    if top:
        forecast = top.get("forecast_audit") if isinstance(top.get("forecast_audit"), dict) else {}
        projection = top.get("projection_audit") if isinstance(top.get("projection_audit"), dict) else {}
        profile = top.get("strategy_profile_audit") if isinstance(top.get("strategy_profile_audit"), dict) else {}
        tp_proof = top.get("tp_proof_audit") if isinstance(top.get("tp_proof_audit"), dict) else {}
        lines.extend(
            [
                f"- Lane: `{top.get('lane_id')}`",
                f"- Shape: `{top.get('pair')}` / `{top.get('direction')}` / `{top.get('strategy_family')}` / `{top.get('vehicle')}`",
                f"- Forecast: `{forecast.get('status')}` latest `{(forecast.get('latest') or {}).get('direction')}`",
                f"- Projection: `{projection.get('status')}` desired signals `{len(projection.get('latest_desired_direction_signals') or [])}`",
                f"- Strategy profile: `{profile.get('status')}`",
                f"- TP proof: `{tp_proof.get('status')}` remaining `{tp_proof.get('remaining_tp_trades')}`",
            ]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Tuning Queue", ""])
    for action in payload.get("forecast_pattern_tuning_queue") or []:
        lines.append(
            f"- `{action.get('priority')}` `{action.get('action_type')}` `{action.get('lane_id')}`: "
            f"{action.get('description')}"
        )
    if not payload.get("forecast_pattern_tuning_queue"):
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


def _same_shape(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not right:
        return False
    return (
        str(left.get("pair") or "").upper(),
        str(left.get("direction") or "").upper(),
        str(left.get("strategy_family") or "").upper(),
    ) == (
        str(right.get("pair") or "").upper(),
        str(right.get("direction") or "").upper(),
        str(right.get("strategy_family") or "").upper(),
    )


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


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
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
        "do_not_hide_negative_expectancy_or_bidask_negative",
        "do_not_mix_market_close_losses_into_tp_proof",
        "do_not_backsolve_lot_from_4x_deficit",
        "do_not_print_secrets_tokens_credentials",
        "do_not_infer_operator_decision",
    ]
