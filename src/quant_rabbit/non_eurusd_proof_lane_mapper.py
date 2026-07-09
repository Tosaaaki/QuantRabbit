from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_AS_LANE_CANDIDATE_BOARD,
    DEFAULT_AS_PROOF_PACK_QUEUE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
    DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
    DEFAULT_VERIFICATION_LEDGER,
    ROOT,
)


MAPPER_VERSION = "non_eurusd_proof_lane_mapper_v1"
TARGET_PAIRS = ("USD_CAD", "USD_JPY", "USD_CHF", "AUD_JPY", "GBP_USD")
STATUS_LIVE_READY = "NON_EURUSD_LIVE_READY_FOUND"
STATUS_EVIDENCE_PATH = "NON_EURUSD_EVIDENCE_PATH_FOUND"
STATUS_MAPPING_GAPS = "NON_EURUSD_MAPPING_GAPS_REMAIN"
STATUS_NO_VALID_PATH = "NON_EURUSD_NO_VALID_PATH"

SPREAD_MARKERS = ("SPREAD_TOO_WIDE", "TARGET_TOO_THIN_FOR_SPREAD")
BIDASK_NEGATIVE_MARKERS = ("BIDASK_REPLAY_NEGATIVE", "packaged_bidask_rule_live_block_negative_expectancy")
NEGATIVE_MARKERS = (
    "NEGATIVE_EXPECTANCY",
    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    "MONTH_SCALE_REPLAY_NEGATIVE",
    "MARKET_CLOSE_LEAK",
    "REPLAY_NEGATIVE",
    "BIDASK_REPLAY_NEGATIVE",
)
PROOF_FLOOR_DEFAULT = 20
BROAD_TP_PROOF_NOT_EXACT_VEHICLE = "BROAD_TP_PROOF_NOT_EXACT_VEHICLE"
EXACT_TP_MAPPING_SCOPES = {"EXACT_LANE_ID", "EXACT_FIELDS"}


@dataclass(frozen=True)
class NonEurusdProofLaneMapperSummary:
    status: str
    output_path: Path
    report_path: Path
    next_active_path: str
    live_permission_allowed: bool


class NonEurusdProofLaneMapper:
    """Map non-EUR/USD historical profit evidence to current exact lanes.

    This is a read-only diagnostic mapper. It never stages orders, changes
    gates, or converts historical pair/side-only evidence into live permission.
    """

    def __init__(
        self,
        *,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        lane_candidate_board_path: Path = DEFAULT_AS_LANE_CANDIDATE_BOARD,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        replay_artifact_paths: list[Path] | None = None,
        output_path: Path = DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
        report_path: Path = DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER_REPORT,
        now_utc: datetime | None = None,
        target_pairs: tuple[str, ...] = TARGET_PAIRS,
    ) -> None:
        self.paths = {
            "active_opportunity_board": active_opportunity_board_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "as_proof_pack_queue": proof_pack_queue_path,
            "as_lane_candidate_board": lane_candidate_board_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "order_intents": order_intents_path,
            "verification_ledger": verification_ledger_path,
        }
        self.execution_ledger_db_path = execution_ledger_db_path
        self.replay_artifact_paths = replay_artifact_paths if replay_artifact_paths is not None else _default_replay_artifacts()
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.target_pairs = tuple(pair.upper() for pair in target_pairs)

    def run(self) -> NonEurusdProofLaneMapperSummary:
        payload = self.build_payload()
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("non-EUR/USD proof lane mapper must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("non-EUR/USD proof lane mapper must not record live side effects")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return NonEurusdProofLaneMapperSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            next_active_path=str(payload.get("next_active_path") or ""),
            live_permission_allowed=False,
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        replay_artifacts = {
            _display_path(path): _load_json_artifact(path)
            for path in self.replay_artifact_paths
            if path.name != DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER.name
        }
        active_board = artifacts["active_opportunity_board"]
        current_lanes = _current_non_eurusd_lanes(active_board, target_pairs=self.target_pairs)
        evidence = _collect_profit_evidence(
            current_lanes=current_lanes,
            payoff=artifacts["payoff_shape_diagnosis"],
            proof_queue=artifacts["as_proof_pack_queue"],
            lane_board=artifacts["as_lane_candidate_board"],
            portfolio=artifacts["portfolio_4x_path_planner"],
            order_intents=artifacts["order_intents"],
            replay_artifacts=replay_artifacts,
            execution_ledger_db_path=self.execution_ledger_db_path,
            target_pairs=self.target_pairs,
        )
        mapping = _map_evidence_to_current_lanes(current_lanes, evidence)
        mapped_lanes = _build_mapped_lanes(current_lanes, mapping, target_pairs=self.target_pairs)
        top_candidates = _top_non_eurusd_candidates(current_lanes, mapped_lanes, target_pairs=self.target_pairs)
        lane_mapping_gaps = _lane_mapping_gaps(mapping, mapped_lanes)
        unmapped_profit_evidence = _unmapped_profit_evidence(mapping)
        spread_gaps = _spread_gaps(current_lanes)
        bidask_replay_gaps = _bidask_replay_gaps(current_lanes)
        proof_floor_gaps = _proof_floor_gaps(mapped_lanes)
        next_active_path = _next_active_path(mapped_lanes)
        status = _status(mapped_lanes, next_active_path, lane_mapping_gaps, unmapped_profit_evidence)

        payload = {
            "schema_version": MAPPER_VERSION,
            "status": status,
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "mapped_lanes": mapped_lanes,
            "unmapped_profit_evidence": unmapped_profit_evidence,
            "top_non_eurusd_candidates": top_candidates,
            "lane_mapping_gaps": lane_mapping_gaps,
            "spread_gaps": spread_gaps,
            "bidask_replay_gaps": bidask_replay_gaps,
            "proof_floor_gaps": proof_floor_gaps,
            "next_active_path": next_active_path,
            "do_not_do": _do_not_do(),
            "source_artifacts": _source_artifacts(
                artifacts=artifacts,
                replay_artifacts=replay_artifacts,
                execution_ledger_db_path=self.execution_ledger_db_path,
            ),
            "safety_checks": {
                "pair_side_only_profit_can_be_tp_proof": False,
                "market_close_loss_can_be_tp_proof": False,
                "negative_expectancy_visible": bool(
                    any(_has_marker(lane.get("blockers") or [], NEGATIVE_MARKERS) for lane in current_lanes)
                ),
                "spread_too_wide_visible": bool(spread_gaps),
                "bidask_negative_visible": bool(bidask_replay_gaps),
                "operator_decision_inference_allowed": False,
            },
        }
        return payload


def _default_replay_artifacts() -> list[Path]:
    data_dir = ROOT / "data"
    paths: list[Path] = []
    for pattern in ("*replay*.json", "*proof*.json"):
        for path in data_dir.glob(pattern):
            if path.name in {
                "as_proof_pack_queue.json",
                DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER.name,
            }:
                continue
            if path not in paths:
                paths.append(path)
    return sorted(paths)


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


def _display_path(path: Path) -> str:
    try:
        if path.is_absolute() and path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
    except ValueError:
        pass
    return str(path)


def _current_non_eurusd_lanes(active_board: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    for index, row in enumerate(_list(active_board.get("ranked_active_lanes"))):
        if not isinstance(row, dict):
            continue
        lane = dict(row)
        pair = str(lane.get("pair") or "UNKNOWN").upper()
        if pair == "EUR_USD" or pair == "UNKNOWN":
            continue
        lane["pair"] = pair
        lane["direction"] = str(lane.get("direction") or lane.get("side") or "UNKNOWN").upper()
        lane["strategy_family"] = str(lane.get("strategy_family") or lane.get("method") or "UNKNOWN").upper()
        lane["vehicle"] = _normalize_vehicle(lane.get("vehicle") or lane.get("order_type"))
        lane["entry_type"] = _entry_type_from_vehicle(lane["vehicle"])
        lane["_board_rank_index"] = index
        lane["_is_target_pair"] = pair in target_pairs
        lanes.append(lane)
    return lanes


def _collect_profit_evidence(
    *,
    current_lanes: list[dict[str, Any]],
    payoff: dict[str, Any],
    proof_queue: dict[str, Any],
    lane_board: dict[str, Any],
    portfolio: dict[str, Any],
    order_intents: dict[str, Any],
    replay_artifacts: dict[str, dict[str, Any]],
    execution_ledger_db_path: Path,
    target_pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    evidence.extend(_evidence_from_current_lane_tp(current_lanes))
    evidence.extend(_evidence_from_payoff(payoff, target_pairs=target_pairs))
    evidence.extend(_evidence_from_proof_queue(proof_queue, target_pairs=target_pairs))
    evidence.extend(_evidence_from_lane_board(lane_board, target_pairs=target_pairs))
    evidence.extend(_evidence_from_portfolio(portfolio, target_pairs=target_pairs))
    evidence.extend(_evidence_from_order_intents(order_intents, target_pairs=target_pairs))
    evidence.extend(_evidence_from_replay_artifacts(replay_artifacts, target_pairs=target_pairs))
    evidence.extend(_evidence_from_execution_ledger(execution_ledger_db_path, target_pairs=target_pairs))
    return evidence


def _evidence_from_current_lane_tp(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
        if not proof:
            continue
        trades = _first_int(proof.get("capture_take_profit_trades"), 0)
        expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
        if trades <= 0 and (expectancy is None or expectancy <= 0):
            continue
        rows.append(
            {
                "evidence_id": f"active_board_local_tp:{lane.get('lane_id')}",
                "source": "data/active_opportunity_board.json:ranked_active_lanes.local_tp_proof",
                "source_kind": "active_board_local_tp",
                "evidence_type": "TP_PROOF",
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "side": lane.get("direction"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "entry_type": lane.get("entry_type"),
                "exit_reason": "TAKE_PROFIT_ORDER",
                "trades": trades,
                "wins": _first_int(proof.get("capture_take_profit_wins"), 0),
                "losses": _first_int(proof.get("capture_take_profit_losses"), 0),
                "expectancy_jpy": _json_number_or_none(expectancy),
                "avg_win_jpy": _json_number_or_none(proof.get("capture_take_profit_avg_win_jpy")),
                "avg_loss_jpy": _json_number_or_none(proof.get("capture_take_profit_avg_loss_jpy")),
                "proof_floor": _first_int(proof.get("capture_take_profit_proof_floor"), PROOF_FLOOR_DEFAULT),
                "scope": proof.get("capture_take_profit_scope"),
                "scope_key": proof.get("capture_take_profit_scope_key"),
                "positive": (expectancy or 0.0) > 0,
                "live_permission_allowed": False,
            }
        )
    return rows


def _evidence_from_payoff(artifact: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_map = {
        "harvest_candidates": "HARVEST_PAYOFF_POSITIVE",
        "partial_tp_runner_candidates": "PARTIAL_TP_RUNNER_DIAGNOSTIC",
        "runner_candidates": "RUNNER_DIAGNOSTIC",
    }
    for key, evidence_type in source_map.items():
        for row in _list(artifact.get(key)):
            if not isinstance(row, dict):
                continue
            pair, side, method, vehicle, payoff_shape = _fields_from_row(row)
            if pair not in target_pairs:
                continue
            expectancy = _first_number(
                row.get("take_profit_expectancy_jpy"),
                row.get("overall_expectancy_jpy_per_trade"),
                row.get("runner_tail_estimated_jpy"),
                row.get("partial_tp_runner_tail_jpy"),
            )
            if expectancy is None or expectancy <= 0:
                continue
            rows.append(
                {
                    "evidence_id": f"payoff:{key}:{pair}:{side}:{method}:{len(rows)}",
                    "source": f"data/payoff_shape_diagnosis.json:{key}",
                    "source_kind": "payoff_shape_diagnosis",
                    "evidence_type": evidence_type,
                    "lane_id": row.get("lane_id"),
                    "pair": pair,
                    "side": side,
                    "strategy_family": method,
                    "vehicle": vehicle,
                    "entry_type": _entry_type_from_vehicle(vehicle),
                    "payoff_shape": payoff_shape,
                    "classification": row.get("classification"),
                    "expectancy_jpy": _json_number_or_none(expectancy),
                    "overall_expectancy_jpy_per_trade": _json_number_or_none(
                        row.get("overall_expectancy_jpy_per_trade")
                    ),
                    "proof_gap_trades": _json_number_or_none(row.get("proof_gap_trades")),
                    "month_scale_blocker": row.get("month_scale_blocker")
                    if isinstance(row.get("month_scale_blocker"), dict)
                    else None,
                    "positive": True,
                    "live_permission_allowed": bool(row.get("live_promotion_allowed")) is True,
                }
            )
    return rows


def _evidence_from_proof_queue(artifact: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("queue", "rejected_candidates"):
        for row in _list(artifact.get(key)):
            if not isinstance(row, dict):
                continue
            pair, side, method, vehicle, payoff_shape = _fields_from_row(row)
            if pair not in target_pairs:
                continue
            rows.append(
                {
                    "evidence_id": f"proof_queue:{key}:{row.get('lane_id') or len(rows)}",
                    "source": f"data/as_proof_pack_queue.json:{key}",
                    "source_kind": "as_proof_pack_queue",
                    "evidence_type": "PROOF_QUEUE_CANDIDATE",
                    "lane_id": row.get("lane_id"),
                    "pair": pair,
                    "side": side,
                    "strategy_family": method,
                    "vehicle": vehicle,
                    "entry_type": _entry_type_from_vehicle(vehicle),
                    "payoff_shape": payoff_shape,
                    "proof_classification": row.get("proof_classification"),
                    "can_enter_proof_pack": bool(row.get("can_enter_proof_pack")),
                    "can_create_live_permission": bool(row.get("can_create_live_permission")),
                    "proof_distance": _json_number_or_none(row.get("proof_distance")),
                    "expected_jpy_per_trade": _json_number_or_none(row.get("expected_jpy_per_trade")),
                    "blockers": _string_list(row.get("current_blockers")) + _string_list(row.get("rejection_reasons")),
                    "positive": (_float(row.get("expected_jpy_per_trade")) or 0.0) > 0
                    or bool(row.get("can_enter_proof_pack")),
                    "live_permission_allowed": bool(row.get("can_create_live_permission")),
                }
            )
    return rows


def _evidence_from_lane_board(artifact: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    row = artifact.get("closest_candidate_to_proof_pack")
    if not isinstance(row, dict):
        return []
    pair, side, method, vehicle, payoff_shape = _fields_from_row(row)
    if pair not in target_pairs:
        return []
    return [
        {
            "evidence_id": f"lane_board:closest:{row.get('lane_id') or pair}",
            "source": "data/as_lane_candidate_board.json:closest_candidate_to_proof_pack",
            "source_kind": "as_lane_candidate_board",
            "evidence_type": "LANE_CANDIDATE_BOARD",
            "lane_id": row.get("lane_id"),
            "pair": pair,
            "side": side,
            "strategy_family": method,
            "vehicle": vehicle,
            "entry_type": _entry_type_from_vehicle(vehicle),
            "payoff_shape": payoff_shape,
            "proof_classification": row.get("proof_classification"),
            "can_enter_proof_pack": bool(row.get("can_enter_proof_pack")),
            "proof_distance": _json_number_or_none(row.get("proof_distance")),
            "blockers": _string_list(row.get("current_blockers")),
            "positive": bool(row.get("can_enter_proof_pack")),
            "live_permission_allowed": bool(row.get("can_create_live_permission")),
        }
    ]


def _evidence_from_portfolio(artifact: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _list(artifact.get("candidate_rankings")):
        if not isinstance(row, dict):
            continue
        pair, side, method, vehicle, payoff_shape = _fields_from_row(row)
        if pair not in target_pairs:
            continue
        rows.append(
            {
                "evidence_id": f"portfolio:{row.get('lane_id') or len(rows)}",
                "source": "data/portfolio_4x_path_planner.json:candidate_rankings",
                "source_kind": "portfolio_4x_path_planner",
                "evidence_type": "PORTFOLIO_4X_CANDIDATE",
                "lane_id": row.get("lane_id"),
                "pair": pair,
                "side": side,
                "strategy_family": method,
                "vehicle": vehicle,
                "entry_type": _entry_type_from_vehicle(vehicle),
                "payoff_shape": payoff_shape,
                "proof_classification": row.get("proof_classification"),
                "can_enter_proof_pack": bool(row.get("can_enter_proof_pack")),
                "proof_distance": _json_number_or_none(row.get("proof_distance")),
                "rank_score": _json_number_or_none(row.get("rank_score")),
                "expected_jpy_per_trade": _json_number_or_none(row.get("expected_jpy_per_trade")),
                "math_candidate_eligible": bool(row.get("math_candidate_eligible")),
                "blockers": _string_list(row.get("current_blockers"))
                + _string_list(row.get("math_exclusion_reasons")),
                "positive": (_float(row.get("expected_jpy_per_trade")) or 0.0) > 0
                or bool(row.get("can_enter_proof_pack")),
                "live_permission_allowed": bool(row.get("can_create_live_permission")),
            }
        )
    return rows


def _evidence_from_order_intents(artifact: dict[str, Any], *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _list(artifact.get("results")):
        if not isinstance(row, dict):
            continue
        intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        pair = str(intent.get("pair") or row.get("pair") or "UNKNOWN").upper()
        if pair not in target_pairs:
            continue
        trades = _first_int(metadata.get("capture_take_profit_trades"), 0)
        expectancy = _float(metadata.get("capture_take_profit_expectancy_jpy"))
        if trades <= 0 and (expectancy is None or expectancy <= 0):
            continue
        method = _first_str(intent.get("method"), (intent.get("market_context") or {}).get("method"), row.get("method"))
        vehicle = _normalize_vehicle(_first_str(intent.get("order_type"), row.get("order_type")))
        rows.append(
            {
                "evidence_id": f"order_intent_local_tp:{row.get('lane_id')}",
                "source": "data/order_intents.json:results.intent.metadata",
                "source_kind": "order_intent_local_tp",
                "evidence_type": "TP_PROOF",
                "lane_id": row.get("lane_id"),
                "pair": pair,
                "side": str(intent.get("side") or intent.get("direction") or row.get("side") or "UNKNOWN").upper(),
                "strategy_family": str(method or "UNKNOWN").upper(),
                "vehicle": vehicle,
                "entry_type": _entry_type_from_vehicle(vehicle),
                "exit_reason": "TAKE_PROFIT_ORDER",
                "trades": trades,
                "wins": _first_int(metadata.get("capture_take_profit_wins"), 0),
                "losses": _first_int(metadata.get("capture_take_profit_losses"), 0),
                "expectancy_jpy": _json_number_or_none(expectancy),
                "proof_floor": PROOF_FLOOR_DEFAULT,
                "scope": metadata.get("capture_take_profit_scope"),
                "scope_key": metadata.get("capture_take_profit_scope_key"),
                "blockers": _string_list(row.get("live_blocker_codes")),
                "positive": (expectancy or 0.0) > 0,
                "live_permission_allowed": False,
            }
        )
    return rows


def _evidence_from_replay_artifacts(
    artifacts: dict[str, dict[str, Any]],
    *,
    target_pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, artifact in artifacts.items():
        if artifact.get("_artifact_status") != "present":
            continue
        row = _best_replay_row(artifact)
        pair, side, method, vehicle, payoff_shape = _fields_from_row(row)
        if pair not in target_pairs:
            continue
        blockers = _codes_from_blockers(artifact.get("remaining_blockers")) + _codes_from_blockers(
            artifact.get("proof_queue_blockers_if_positive")
        )
        exact_replay = artifact.get("exact_shape_replay") if isinstance(artifact.get("exact_shape_replay"), dict) else {}
        expectancy = _first_number(
            artifact.get("net_expectancy_after_bidask"),
            artifact.get("net_expectancy_after_bidask_slippage"),
            exact_replay.get("expectancy_jpy"),
        )
        rows.append(
            {
                "evidence_id": f"replay:{source}",
                "source": source,
                "source_kind": "replay_artifact",
                "evidence_type": "BIDASK_OR_REPLAY_PROOF",
                "lane_id": row.get("lane_id") or artifact.get("lane_id"),
                "pair": pair,
                "side": side,
                "strategy_family": method,
                "vehicle": vehicle,
                "entry_type": _entry_type_from_vehicle(vehicle),
                "payoff_shape": payoff_shape,
                "classification": artifact.get("classification") or artifact.get("status"),
                "replay_status": _first_str(
                    artifact.get("status"),
                    artifact.get("bidask_replay_status"),
                    artifact.get("s5_bidask_replay_status"),
                    artifact.get("classification"),
                    "UNKNOWN",
                ),
                "sample_count": _json_number_or_none(
                    _first_number(artifact.get("replay_sample_count"), exact_replay.get("sample_count"))
                ),
                "expectancy_jpy": _json_number_or_none(expectancy),
                "blockers": blockers,
                "positive": expectancy is not None and expectancy > 0,
                "live_permission_allowed": bool(artifact.get("live_permission_allowed")),
            }
        )
    return rows


def _evidence_from_execution_ledger(path: Path, *, target_pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                """
                select pair, side, lane_id, exit_reason,
                       count(*) as trades,
                       sum(realized_pl_jpy) as net_jpy,
                       avg(realized_pl_jpy) as avg_jpy,
                       sum(case when realized_pl_jpy > 0 then 1 else 0 end) as wins,
                       sum(case when realized_pl_jpy < 0 then 1 else 0 end) as losses,
                       max(ts_utc) as latest_ts_utc
                from execution_events
                where event_type = 'TRADE_CLOSED'
                  and pair in ({})
                group by pair, side, lane_id, exit_reason
                """.format(",".join("?" for _ in target_pairs)),
                target_pairs,
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []

    evidence: list[dict[str, Any]] = []
    for row in rows:
        pair = str(row["pair"] or "UNKNOWN").upper()
        side = str(row["side"] or "UNKNOWN").upper()
        lane_id = str(row["lane_id"] or "")
        parsed = _parse_lane_id(lane_id)
        exit_reason = str(row["exit_reason"] or "UNKNOWN").upper()
        net_jpy = _float(row["net_jpy"]) or 0.0
        is_take_profit = exit_reason == "TAKE_PROFIT_ORDER"
        is_market_close = "MARKET_ORDER" in exit_reason and "CLOSE" in exit_reason
        if net_jpy <= 0 and not is_market_close:
            continue
        evidence_type = "EXECUTION_TP_PROOF" if is_take_profit else "MARKET_CLOSE_NOT_TP_PROOF"
        evidence.append(
            {
                "evidence_id": f"execution_ledger:{pair}:{side}:{lane_id or 'NO_LANE'}:{exit_reason}",
                "source": "data/execution_ledger.db:execution_events",
                "source_kind": "execution_ledger",
                "evidence_type": evidence_type,
                "lane_id": lane_id or None,
                "pair": pair,
                "side": side,
                "strategy_family": parsed.get("strategy_family", "UNKNOWN"),
                "vehicle": _normalize_vehicle(parsed.get("vehicle")),
                "entry_type": _entry_type_from_vehicle(parsed.get("vehicle")),
                "exit_reason": exit_reason,
                "trades": int(row["trades"] or 0),
                "wins": int(row["wins"] or 0),
                "losses": int(row["losses"] or 0),
                "net_jpy": _json_number_or_none(net_jpy),
                "avg_jpy": _json_number_or_none(row["avg_jpy"]),
                "latest_ts_utc": row["latest_ts_utc"],
                "positive": net_jpy > 0 and is_take_profit,
                "not_tp_proof_reason": "market_close_is_not_tp_proof" if is_market_close else "",
                "live_permission_allowed": False,
            }
        )
    return evidence


def _map_evidence_to_current_lanes(
    current_lanes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    lanes_by_id = {str(lane.get("lane_id")): lane for lane in current_lanes if lane.get("lane_id")}
    lanes_by_exact = {_exact_lane_key(lane): lane for lane in current_lanes}
    mappings: dict[str, list[dict[str, Any]]] = {str(lane.get("lane_id")): [] for lane in current_lanes}
    partials: dict[str, list[dict[str, Any]]] = {str(lane.get("lane_id")): [] for lane in current_lanes}
    gaps: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []

    for item in evidence:
        if item.get("evidence_type") == "MARKET_CLOSE_NOT_TP_PROOF":
            unmapped.append(_unmapped_item(item, "MARKET_CLOSE_NOT_TP_PROOF", current_lanes))
            if _float(item.get("net_jpy")) is not None and (_float(item.get("net_jpy")) or 0.0) < 0:
                gaps.append(_gap("MARKET_CLOSE_LOSS_NOT_TP_PROOF", item, None))
            continue

        lane_id = str(item.get("lane_id") or "")
        if lane_id and lane_id in lanes_by_id:
            lane = lanes_by_id[lane_id]
            field_gaps = _exact_field_gaps(lane, item)
            if field_gaps:
                gaps.append(_gap("LANE_ID_FIELD_CONFLICT", item, lane, missing_fields=field_gaps))
                continue
            scope_gaps = _local_tp_scope_gaps(item)
            if scope_gaps:
                partial = {**item, "mapping_scope": "PAIR_SIDE_STRATEGY_ONLY", "missing_exact_fields": scope_gaps}
                partials[str(lane["lane_id"])].append(partial)
                gaps.append(_gap("SOURCE_MISSING_EXACT_VEHICLE_OR_ENTRY_TYPE", item, lane, missing_fields=scope_gaps))
                continue
            mappings[str(lane["lane_id"])].append({**item, "mapping_scope": "EXACT_LANE_ID"})
            continue

        item_exact_key = _exact_lane_key(item)
        if _is_exact_key(item_exact_key) and item_exact_key in lanes_by_exact:
            lane = lanes_by_exact[item_exact_key]
            scope_gaps = _local_tp_scope_gaps(item)
            if scope_gaps:
                partial = {**item, "mapping_scope": "PAIR_SIDE_STRATEGY_ONLY", "missing_exact_fields": scope_gaps}
                partials[str(lane["lane_id"])].append(partial)
                gaps.append(_gap("SOURCE_MISSING_EXACT_VEHICLE_OR_ENTRY_TYPE", item, lane, missing_fields=scope_gaps))
                continue
            mappings[str(lane["lane_id"])].append({**item, "mapping_scope": "EXACT_FIELDS"})
            continue

        if _has_pair_side_strategy(item):
            matched = _matching_pair_side_strategy_lanes(current_lanes, item)
            if matched:
                for lane in matched:
                    missing = _missing_exact_fields(item)
                    partial = {**item, "mapping_scope": "PAIR_SIDE_STRATEGY_ONLY", "missing_exact_fields": missing}
                    partials[str(lane["lane_id"])].append(partial)
                    gaps.append(_gap("SOURCE_MISSING_EXACT_VEHICLE_OR_ENTRY_TYPE", item, lane, missing_fields=missing))
                continue

        unmapped.append(_unmapped_item(item, "NO_EXACT_CURRENT_LANE_MATCH", current_lanes))

    return {"mappings": mappings, "partials": partials, "gaps": gaps, "unmapped": unmapped}


def _build_mapped_lanes(
    current_lanes: list[dict[str, Any]],
    mapping: dict[str, Any],
    *,
    target_pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    mapped_rows: list[dict[str, Any]] = []
    exact_map = mapping.get("mappings") if isinstance(mapping.get("mappings"), dict) else {}
    partial_map = mapping.get("partials") if isinstance(mapping.get("partials"), dict) else {}
    for lane in current_lanes:
        lane_id = str(lane.get("lane_id") or "")
        exact_evidence = _unique_evidence(exact_map.get(lane_id) or [])
        partial_evidence = _unique_evidence(partial_map.get(lane_id) or [])
        if not exact_evidence and not partial_evidence:
            continue
        positive_exact = [row for row in exact_evidence if row.get("positive") is True]
        positive_partial = [row for row in partial_evidence if row.get("positive") is True]
        blockers = _string_list(lane.get("blockers"))
        spread_blocked = _spread_blocked(lane)
        bidask_negative = _bidask_negative(lane)
        mapping_gaps = []
        if partial_evidence:
            mapping_gaps.append("SOURCE_EVIDENCE_NOT_EXACT_VEHICLE_ENTRY_TYPE")
        if not positive_exact and positive_partial:
            mapping_gaps.append("POSITIVE_EVIDENCE_PAIR_SIDE_STRATEGY_ONLY")
        if not lane.get("order_intent_status") and "data/order_intents.json" not in _string_list(lane.get("source_refs")):
            mapping_gaps.append("CURRENT_EXECUTABLE_INTENT_MISSING")
        proof_floor = _lane_proof_floor(lane, exact_evidence + partial_evidence)
        if proof_floor.get("broad_method_tp_trades") and not proof_floor.get("exact_vehicle_tp_trades"):
            mapping_gaps.append(BROAD_TP_PROOF_NOT_EXACT_VEHICLE)
        assessment = _promotion_assessment(
            lane,
            positive_exact=positive_exact,
            positive_partial=positive_partial,
            proof_floor=proof_floor,
            mapping_gaps=mapping_gaps,
            spread_blocked=spread_blocked,
            bidask_negative=bidask_negative,
        )
        mapped_rows.append(
            {
                "lane_id": lane_id,
                "pair": lane.get("pair"),
                "side": lane.get("direction"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "entry_type": lane.get("entry_type"),
                "exact_lane_key": _exact_lane_key(lane),
                "active_board_status": lane.get("status"),
                "promotion_assessment": assessment,
                "live_permission_allowed": False,
                "rank_score": _json_number_or_none(lane.get("rank_score")),
                "can_enter_proof_pack": bool(lane.get("can_enter_proof_pack")),
                "proof_status": lane.get("proof_status"),
                "spread_status": lane.get("spread_status"),
                "replay_status": lane.get("replay_status"),
                "blockers": blockers,
                "spread_blocked": spread_blocked,
                "bidask_replay_negative": bidask_negative,
                "negative_expectancy_visible": _has_marker(blockers, NEGATIVE_MARKERS),
                "proof_floor": proof_floor,
                "mapping_gaps": _unique(mapping_gaps),
                "mapped_profit_evidence": _compact_evidence(positive_exact),
                "supporting_partial_evidence": _compact_evidence(positive_partial),
                "all_mapped_evidence_sources": _unique(
                    [str(row.get("source")) for row in exact_evidence + partial_evidence if row.get("source")]
                ),
                "source_refs": _string_list(lane.get("source_refs")),
                "target_pair_priority": _target_pair_priority(str(lane.get("pair") or ""), target_pairs=target_pairs),
                "next_action": _mapper_next_action(lane, assessment, proof_floor, mapping_gaps),
            }
        )
    return sorted(mapped_rows, key=_mapped_lane_sort_key)


def _promotion_assessment(
    lane: dict[str, Any],
    *,
    positive_exact: list[dict[str, Any]],
    positive_partial: list[dict[str, Any]],
    proof_floor: dict[str, Any],
    mapping_gaps: list[str],
    spread_blocked: bool,
    bidask_negative: bool,
) -> str:
    if spread_blocked:
        return "BLOCKED_SPREAD_GAP_VISIBLE"
    if (
        str(lane.get("status") or "") == "LIVE_READY"
        and positive_exact
        and proof_floor.get("met") is True
        and not bidask_negative
        and not mapping_gaps
    ):
        return "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY"
    if positive_exact and str(lane.get("vehicle") or "UNKNOWN") != "UNKNOWN":
        return "EVIDENCE_ACQUISITION_CANDIDATE"
    if positive_partial:
        return "MAPPING_GAPS_REMAIN"
    return "NO_VALID_PROOF_PATH"


def _lane_proof_floor(lane: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
    floor = _first_int(proof.get("capture_take_profit_proof_floor"), None)
    broad_current = _broad_method_tp_trades(proof)
    exact_counts_by_scope: dict[tuple[str, ...], int] = {}
    for row in evidence:
        if row.get("mapping_scope") in EXACT_TP_MAPPING_SCOPES:
            trades = int(_first_int(row.get("trades"), 0) or 0)
            if trades > 0:
                scope = _exact_tp_count_scope(row)
                exact_counts_by_scope[scope] = max(exact_counts_by_scope.get(scope, 0), trades)
        floor = _first_int(floor, row.get("proof_floor"), PROOF_FLOOR_DEFAULT)
    if floor is None:
        floor = PROOF_FLOOR_DEFAULT
    exact_current = sum(exact_counts_by_scope.values())
    current = exact_current
    gap = max(floor - current, 0)
    result = {
        "current_tp_trades": current,
        "required_tp_trades": floor,
        "remaining_tp_trades": gap,
        "met": current >= floor,
        "exact_vehicle_tp_trades": exact_current,
        "proof_scope_status": "EXACT_VEHICLE" if exact_current > 0 else "NO_EXACT_VEHICLE_PROOF",
    }
    if broad_current is not None:
        broad_scope = proof.get("broad_capture_take_profit_scope") or proof.get("capture_take_profit_scope")
        broad_scope_key = proof.get("broad_capture_take_profit_scope_key") or proof.get("capture_take_profit_scope_key")
        result.update(
            {
                "broad_method_tp_trades": broad_current,
                "broad_method_scope": broad_scope,
                "broad_method_scope_key": broad_scope_key,
                "broad_method_not_used_as_exact_vehicle_proof": bool(
                    broad_current > exact_current
                    and (
                        str(broad_scope or "").upper() == "PAIR_SIDE_METHOD"
                        or proof.get("broad_capture_take_profit_not_used_as_exact_vehicle_proof") is True
                    )
                ),
            }
        )
    return result


def _exact_tp_count_scope(row: dict[str, Any]) -> tuple[str, ...]:
    """Collapse duplicate summaries of the same exact TP sample set.

    active_opportunity_board and order_intents both carry local TP summaries
    derived from the same underlying execution ledger. Treating those summaries
    as independent samples overstates proof-floor progress.
    """

    return (
        str(row.get("pair") or "").upper(),
        str(row.get("side") or row.get("direction") or "").upper(),
        str(row.get("strategy_family") or "").upper(),
        str(row.get("vehicle") or "").upper(),
        str(row.get("entry_type") or "").upper(),
        str(row.get("exit_reason") or "TAKE_PROFIT_ORDER").upper(),
    )


def _broad_method_tp_trades(proof: dict[str, Any]) -> int | None:
    broad = _first_int(proof.get("broad_capture_take_profit_trades"), None)
    if broad is not None:
        return broad
    if str(proof.get("capture_take_profit_scope") or "").upper() != "PAIR_SIDE_METHOD":
        return None
    return _first_int(proof.get("capture_take_profit_trades"), None)


def _top_non_eurusd_candidates(
    current_lanes: list[dict[str, Any]],
    mapped_lanes: list[dict[str, Any]],
    *,
    target_pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    mapped_by_id = {row["lane_id"]: row for row in mapped_lanes}
    selected: list[dict[str, Any]] = []
    for lane in sorted(current_lanes, key=lambda item: int(item.get("_board_rank_index") or 999999)):
        pair = str(lane.get("pair") or "")
        if len(selected) >= 25 and pair not in target_pairs:
            continue
        mapped = mapped_by_id.get(str(lane.get("lane_id") or ""))
        selected.append(
            {
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "side": lane.get("direction"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "entry_type": lane.get("entry_type"),
                "active_board_status": lane.get("status"),
                "rank_score": _json_number_or_none(lane.get("rank_score")),
                "proof_status": lane.get("proof_status"),
                "spread_status": lane.get("spread_status"),
                "replay_status": lane.get("replay_status"),
                "blockers": _string_list(lane.get("blockers"))[:12],
                "proof_mapper_assessment": (mapped or {}).get("promotion_assessment", "NO_MAPPED_PROFIT_EVIDENCE"),
                "mapped_positive_evidence_count": len((mapped or {}).get("mapped_profit_evidence") or []),
                "partial_positive_evidence_count": len((mapped or {}).get("supporting_partial_evidence") or []),
            }
        )
        if len(selected) >= 40:
            break
    return selected


def _lane_mapping_gaps(mapping: dict[str, Any], mapped_lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = list(mapping.get("gaps") or [])
    for lane in mapped_lanes:
        for code in _string_list(lane.get("mapping_gaps")):
            gaps.append(
                {
                    "code": code,
                    "lane_id": lane.get("lane_id"),
                    "pair": lane.get("pair"),
                    "side": lane.get("side"),
                    "strategy_family": lane.get("strategy_family"),
                    "vehicle": lane.get("vehicle"),
                    "entry_type": lane.get("entry_type"),
                    "reason": "Mapped evidence is not enough to prove the current exact lane.",
                }
            )
    return _dedupe_gap_rows(gaps)[:80]


def _unmapped_profit_evidence(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in mapping.get("unmapped") or []:
        item = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        if not item:
            continue
        if item.get("positive") is not True and item.get("evidence_type") != "MARKET_CLOSE_NOT_TP_PROOF":
            continue
        rows.append(row)
    return rows[:80]


def _spread_gaps(current_lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in current_lanes:
        if not _spread_blocked(lane):
            continue
        rows.append(_gap_from_lane("SPREAD_GAP_VISIBLE", lane, "Current spread/target blocker remains active."))
    return rows


def _bidask_replay_gaps(current_lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in current_lanes:
        if not _bidask_negative(lane):
            continue
        rows.append(
            _gap_from_lane(
                "BIDASK_REPLAY_NEGATIVE_OR_REFRESH_REQUIRED",
                lane,
                "Current bid/ask replay negative or refresh-required evidence remains active.",
            )
        )
    return rows


def _proof_floor_gaps(mapped_lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in mapped_lanes:
        proof_floor = lane.get("proof_floor") if isinstance(lane.get("proof_floor"), dict) else {}
        if proof_floor.get("met") is True:
            continue
        if int(proof_floor.get("required_tp_trades") or 0) <= 0:
            continue
        rows.append(
            {
                "code": "TP_PROOF_FLOOR_NOT_MET",
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "side": lane.get("side"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "entry_type": lane.get("entry_type"),
                "current_tp_trades": proof_floor.get("current_tp_trades"),
                "required_tp_trades": proof_floor.get("required_tp_trades"),
                "remaining_tp_trades": proof_floor.get("remaining_tp_trades"),
            }
        )
    return rows


def _next_active_path(mapped_lanes: list[dict[str, Any]]) -> str:
    live = [row for row in mapped_lanes if row.get("promotion_assessment") == "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY"]
    if live:
        lane = sorted(live, key=_next_path_sort_key)[0]
        return (
            f"{STATUS_LIVE_READY}: {lane['lane_id']} has diagnostic LIVE_READY mapping only; "
            "live_permission_allowed remains false and gateway/verifier still decide."
        )
    evidence = [row for row in mapped_lanes if row.get("promotion_assessment") == "EVIDENCE_ACQUISITION_CANDIDATE"]
    if not evidence:
        return ""
    lane = sorted(evidence, key=_next_path_sort_key)[0]
    remaining = (lane.get("proof_floor") or {}).get("remaining_tp_trades")
    blockers = ", ".join(_string_list(lane.get("blockers"))[:5])
    return (
        f"{STATUS_EVIDENCE_PATH}: {lane['lane_id']} is the shortest read-only non-EUR/USD proof path; "
        f"remaining_tp_floor={remaining}, blockers={blockers}. Do not send."
    )


def _status(
    mapped_lanes: list[dict[str, Any]],
    next_active_path: str,
    lane_mapping_gaps: list[dict[str, Any]],
    unmapped_profit_evidence: list[dict[str, Any]],
) -> str:
    if any(row.get("promotion_assessment") == "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY" for row in mapped_lanes):
        return STATUS_LIVE_READY
    if next_active_path:
        return STATUS_EVIDENCE_PATH
    if lane_mapping_gaps or unmapped_profit_evidence:
        return STATUS_MAPPING_GAPS
    return STATUS_NO_VALID_PATH


def _mapper_next_action(
    lane: dict[str, Any],
    assessment: str,
    proof_floor: dict[str, Any],
    mapping_gaps: list[str],
) -> str:
    lane_key = _exact_lane_key(lane)
    if assessment == "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY":
        return f"Keep {lane_key} diagnostic only; mapper grants no live permission."
    if assessment == "EVIDENCE_ACQUISITION_CANDIDATE":
        if _bidask_negative(lane):
            return (
                f"Repair bid/ask-negative pattern or vehicle shape for {lane_key}; preserve the negative blocker, "
                "do not repeat exact replay until lane inputs change, and do not send."
            )
        if proof_floor.get("met") is not True:
            return f"Collect {proof_floor.get('remaining_tp_trades')} more exact TP proof sample(s) for {lane_key}."
        return f"Canonicalize exact proof pack for {lane_key}; do not infer live permission."
    if mapping_gaps:
        return f"Do not promote {lane_key}; resolve mapping gaps: {', '.join(mapping_gaps)}."
    return f"No proof path for {lane_key}; preserve blocker cause."


def _source_artifacts(
    *,
    artifacts: dict[str, dict[str, Any]],
    replay_artifacts: dict[str, dict[str, Any]],
    execution_ledger_db_path: Path,
) -> dict[str, Any]:
    entries = {name: _artifact_entry(artifact) for name, artifact in artifacts.items()}
    entries["execution_ledger_db"] = {
        "path": str(execution_ledger_db_path),
        "status": "present" if execution_ledger_db_path.exists() else "missing",
        "size_bytes": execution_ledger_db_path.stat().st_size if execution_ledger_db_path.exists() else None,
    }
    entries["replay_artifacts"] = {name: _artifact_entry(artifact) for name, artifact in replay_artifacts.items()}
    return entries


def _artifact_entry(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact.get("_path"),
        "status": artifact.get("_artifact_status"),
        "sha256": artifact.get("_sha256"),
        "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Non-EUR/USD Proof Lane Mapper",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Mapped lanes: `{len(payload.get('mapped_lanes') or [])}`",
        f"- Unmapped profit evidence rows: `{len(payload.get('unmapped_profit_evidence') or [])}`",
        "",
        "## Next Active Path",
        "",
        payload.get("next_active_path") or "No valid non-EUR/USD evidence path found.",
        "",
        "## Top Non-EUR/USD Candidates",
        "",
    ]
    for lane in (payload.get("top_non_eurusd_candidates") or [])[:12]:
        lines.append(
            f"- `{lane.get('lane_id')}`: `{lane.get('active_board_status')}` / "
            f"`{lane.get('proof_mapper_assessment')}` / spread `{lane.get('spread_status')}` / "
            f"replay `{lane.get('replay_status')}`"
        )
    lines.extend(["", "## Mapped Lanes", ""])
    for lane in (payload.get("mapped_lanes") or [])[:12]:
        proof = lane.get("proof_floor") if isinstance(lane.get("proof_floor"), dict) else {}
        lines.append(
            f"- `{lane.get('lane_id')}`: `{lane.get('promotion_assessment')}`, "
            f"TP proof `{proof.get('current_tp_trades')}/{proof.get('required_tp_trades')}`, "
            f"bidask_negative `{lane.get('bidask_replay_negative')}`, spread_blocked `{lane.get('spread_blocked')}`"
        )
    lines.extend(["", "## Gaps", ""])
    lines.append(f"- Lane mapping gaps: `{len(payload.get('lane_mapping_gaps') or [])}`")
    lines.append(f"- Spread gaps: `{len(payload.get('spread_gaps') or [])}`")
    lines.append(f"- Bid/ask replay gaps: `{len(payload.get('bidask_replay_gaps') or [])}`")
    lines.append(f"- Proof floor gaps: `{len(payload.get('proof_floor_gaps') or [])}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This artifact is read-only. It does not send, cancel, close, change launchd, relax gates, hide negative expectancy, ignore spread blockers, treat market-close loss as TP proof, backsolve lots from a 4x deficit, infer operator approval, or disclose secrets.",
            "",
        ]
    )
    return "\n".join(lines)


def _do_not_do() -> list[str]:
    return [
        "do_not_send_live_order",
        "do_not_cancel_or_close",
        "do_not_launchd_load_reload",
        "do_not_mutate_broker_state",
        "do_not_relax_gates",
        "do_not_hide_negative_expectancy",
        "do_not_ignore_spread_too_wide",
        "do_not_ignore_bidask_replay_negative",
        "do_not_mix_market_close_loss_into_tp_proof",
        "do_not_backsolve_lot_from_4x_deficit",
        "do_not_infer_operator_decision",
        "do_not_print_secrets_tokens_credentials",
    ]


def _best_replay_row(artifact: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in ("target_shape", "required_shape", "requested_shape", "stale_candidate_shape"):
        value = artifact.get(key)
        if isinstance(value, str):
            row.update(_parse_shape(value))
            break
        if isinstance(value, dict):
            row.update(value)
            break
    if artifact.get("lane_id"):
        row["lane_id"] = artifact.get("lane_id")
    for nested_key in ("current_queue_context", "as_candidate"):
        nested = artifact.get(nested_key)
        if isinstance(nested, dict):
            row = {**row, **nested}
            break
    return row


def _fields_from_row(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    shape = row.get("shape_key") or row.get("target_shape") or row.get("required_shape")
    parsed = _parse_shape(shape) if isinstance(shape, str) else {}
    lane_parsed = _parse_lane_id(str(row.get("lane_id") or ""))
    pair = _first_str(row.get("pair"), parsed.get("pair"), lane_parsed.get("pair"), "UNKNOWN").upper()
    side = _first_str(row.get("side"), row.get("direction"), parsed.get("direction"), lane_parsed.get("side"), "UNKNOWN")
    method = _first_str(
        row.get("method"),
        row.get("strategy"),
        row.get("strategy_family"),
        parsed.get("strategy_family"),
        lane_parsed.get("strategy_family"),
        "UNKNOWN",
    )
    vehicle = _normalize_vehicle(
        _first_str(
            row.get("order_type"),
            row.get("entry_type"),
            row.get("vehicle"),
            parsed.get("vehicle"),
            lane_parsed.get("vehicle"),
            "UNKNOWN",
        )
    )
    payoff = _first_str(row.get("exit_shape"), row.get("payoff_shape"), parsed.get("payoff_shape"), "UNKNOWN")
    return pair, str(side).upper(), str(method).upper(), vehicle, str(payoff).upper()


def _parse_lane_id(lane_id: str) -> dict[str, str]:
    parts = str(lane_id or "").split(":")
    if len(parts) < 4:
        return {}
    return {
        "pair": parts[1].upper(),
        "side": parts[2].upper(),
        "strategy_family": parts[3].upper(),
        "vehicle": _normalize_vehicle(parts[4] if len(parts) > 4 else "UNKNOWN"),
    }


def _parse_shape(shape: Any) -> dict[str, str]:
    if not isinstance(shape, str):
        return {}
    parts = shape.split("|")
    parsed: dict[str, str] = {}
    if len(parts) >= 1:
        parsed["pair"] = parts[0].upper()
    if len(parts) >= 2:
        parsed["side"] = parts[1].upper()
    if len(parts) >= 3:
        parsed["strategy_family"] = parts[2].upper()
    if len(parts) >= 4:
        parsed["vehicle"] = _normalize_vehicle(parts[3])
    if len(parts) >= 5:
        parsed["payoff_shape"] = parts[4].upper()
    return parsed


def _exact_lane_key(row: dict[str, Any]) -> str:
    vehicle = _normalize_vehicle(row.get("vehicle"))
    return "|".join(
        [
            str(row.get("pair") or "UNKNOWN").upper(),
            str(row.get("side") or row.get("direction") or "UNKNOWN").upper(),
            str(row.get("strategy_family") or row.get("method") or "UNKNOWN").upper(),
            vehicle,
            _entry_type_from_vehicle(row.get("entry_type") or vehicle),
        ]
    )


def _is_exact_key(key: str) -> bool:
    return "UNKNOWN" not in key.split("|")


def _has_pair_side_strategy(item: dict[str, Any]) -> bool:
    return all(str(item.get(key) or "UNKNOWN") != "UNKNOWN" for key in ("pair", "side", "strategy_family"))


def _matching_pair_side_strategy_lanes(current_lanes: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        lane
        for lane in current_lanes
        if str(lane.get("pair")) == str(item.get("pair"))
        and str(lane.get("direction")) == str(item.get("side"))
        and str(lane.get("strategy_family")) == str(item.get("strategy_family"))
    ]


def _missing_exact_fields(item: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if _normalize_vehicle(item.get("vehicle")) == "UNKNOWN":
        missing.append("vehicle")
    if _entry_type_from_vehicle(item.get("entry_type") or item.get("vehicle")) == "UNKNOWN":
        missing.append("entry_type")
    return missing or ["vehicle_or_entry_type"]


def _local_tp_scope_gaps(item: dict[str, Any]) -> list[str]:
    if item.get("source_kind") not in {"active_board_local_tp", "order_intent_local_tp"}:
        return []
    if str(item.get("evidence_type") or "").upper() != "TP_PROOF":
        return []
    scope = str(item.get("scope") or "").upper()
    if scope in {"PAIR_SIDE_METHOD_VEHICLE", "EXACT_VEHICLE", "EXACT_LANE"}:
        return []
    if scope == "PAIR_SIDE_METHOD":
        return ["vehicle_scope_missing"]
    if not scope or scope.startswith("MISSING"):
        return ["exact_tp_scope_missing"]
    return [f"unsupported_tp_scope:{scope}"]


def _exact_field_gaps(lane: dict[str, Any], item: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    checks = (
        ("pair", lane.get("pair"), item.get("pair")),
        ("side", lane.get("direction"), item.get("side")),
        ("strategy_family", lane.get("strategy_family"), item.get("strategy_family")),
        ("vehicle", lane.get("vehicle"), item.get("vehicle")),
        ("entry_type", lane.get("entry_type"), item.get("entry_type")),
    )
    for name, left, right in checks:
        if right in (None, "", "UNKNOWN"):
            if name in {"vehicle", "entry_type"}:
                gaps.append(f"{name}_missing")
            continue
        if str(left).upper() != str(right).upper():
            gaps.append(f"{name}_mismatch")
    return gaps


def _normalize_vehicle(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    if text in {"STOP-ENTRY", "STOP_ENTRY", "STOP_ORDER"}:
        return "STOP"
    if text in {"LIMIT_ORDER"}:
        return "LIMIT"
    if text in {"MARKET_ORDER"}:
        return "MARKET"
    if text in {"LIMIT", "MARKET", "STOP"}:
        return text
    return "UNKNOWN" if text in {"", "NONE", "NULL"} else text


def _entry_type_from_vehicle(value: Any) -> str:
    vehicle = _normalize_vehicle(value)
    if vehicle == "STOP":
        return "STOP-ENTRY"
    if vehicle in {"LIMIT", "MARKET"}:
        return vehicle
    return "UNKNOWN"


def _spread_blocked(lane: dict[str, Any]) -> bool:
    if str(lane.get("spread_status") or "").upper() == "BLOCKED":
        return True
    return _has_marker(lane.get("blockers") or [], SPREAD_MARKERS)


def _bidask_negative(lane: dict[str, Any]) -> bool:
    if "NEGATIVE" in str(lane.get("replay_status") or "").upper():
        return True
    return _has_marker(lane.get("blockers") or [], BIDASK_NEGATIVE_MARKERS)


def _has_marker(blockers: list[Any], markers: tuple[str, ...]) -> bool:
    return any(any(marker in str(blocker) for marker in markers) for blocker in blockers)


def _gap(
    code: str,
    evidence: dict[str, Any],
    lane: dict[str, Any] | None,
    *,
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "lane_id": lane.get("lane_id") if isinstance(lane, dict) else None,
        "pair": evidence.get("pair"),
        "side": evidence.get("side"),
        "strategy_family": evidence.get("strategy_family"),
        "vehicle": evidence.get("vehicle"),
        "entry_type": evidence.get("entry_type"),
        "evidence_id": evidence.get("evidence_id"),
        "source": evidence.get("source"),
        "missing_fields": missing_fields or [],
        "reason": _gap_reason(code),
    }


def _gap_from_lane(code: str, lane: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "code": code,
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "side": lane.get("direction"),
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "entry_type": lane.get("entry_type"),
        "spread_status": lane.get("spread_status"),
        "replay_status": lane.get("replay_status"),
        "blockers": _string_list(lane.get("blockers"))[:12],
        "reason": reason,
    }


def _gap_reason(code: str) -> str:
    reasons = {
        "LANE_ID_FIELD_CONFLICT": "Evidence lane_id conflicts with one or more current lane fields.",
        "SOURCE_MISSING_EXACT_VEHICLE_OR_ENTRY_TYPE": "Evidence matches pair/side/strategy but lacks exact vehicle or entry_type.",
        "MARKET_CLOSE_LOSS_NOT_TP_PROOF": "Market-close loss must remain negative evidence and cannot be counted as TP proof.",
    }
    return reasons.get(code, "Exact lane mapping gap remains.")


def _unmapped_item(evidence: dict[str, Any], reason_code: str, lanes: list[dict[str, Any]]) -> dict[str, Any]:
    possible = []
    for lane in lanes:
        if str(lane.get("pair")) != str(evidence.get("pair")):
            continue
        if str(lane.get("direction")) != str(evidence.get("side")):
            continue
        possible.append(
            {
                "lane_id": lane.get("lane_id"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "entry_type": lane.get("entry_type"),
                "status": lane.get("status"),
            }
        )
    return {
        "reason_code": reason_code,
        "evidence": _compact_one_evidence(evidence),
        "possible_current_lanes": possible[:10],
        "mapping_allowed_as_tp_proof": False,
    }


def _compact_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_one_evidence(row) for row in rows[:12]]


def _compact_one_evidence(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "evidence_id",
        "source",
        "source_kind",
        "evidence_type",
        "mapping_scope",
        "pair",
        "side",
        "strategy_family",
        "vehicle",
        "entry_type",
        "exit_reason",
        "classification",
        "proof_classification",
        "trades",
        "wins",
        "losses",
        "net_jpy",
        "expectancy_jpy",
        "expected_jpy_per_trade",
        "proof_distance",
        "proof_gap_trades",
        "proof_floor",
        "scope",
        "scope_key",
        "blockers",
        "positive",
        "live_permission_allowed",
        "not_tp_proof_reason",
    )
    return {key: row.get(key) for key in keys if row.get(key) not in (None, [], "")}


def _unique_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("evidence_id") or json.dumps(row, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _dedupe_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = "|".join(
            str(row.get(name) or "")
            for name in ("code", "lane_id", "evidence_id", "pair", "side", "strategy_family", "vehicle")
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _mapped_lane_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _assessment_sort(row.get("promotion_assessment")),
        _next_path_sort_key(row),
    )


def _next_path_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    proof = row.get("proof_floor") if isinstance(row.get("proof_floor"), dict) else {}
    blockers = _string_list(row.get("blockers"))
    return (
        row.get("spread_blocked") is True,
        row.get("promotion_assessment") != "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY",
        row.get("promotion_assessment") != "EVIDENCE_ACQUISITION_CANDIDATE",
        not bool(row.get("can_enter_proof_pack")),
        not _has_replay_or_proof_pack_artifact_source(row),
        str(row.get("vehicle") or "") != "LIMIT",
        _has_marker(blockers, ("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE", "MONTH_SCALE_REPLAY_NEGATIVE")),
        -int(proof.get("current_tp_trades") or 0),
        int(proof.get("remaining_tp_trades") or 999),
        len(blockers),
        -float(row.get("rank_score") or 0.0),
        str(row.get("lane_id") or ""),
    )


def _assessment_sort(value: Any) -> int:
    order = {
        "LIVE_READY_CANDIDATE_DIAGNOSTIC_ONLY": 0,
        "EVIDENCE_ACQUISITION_CANDIDATE": 1,
        "MAPPING_GAPS_REMAIN": 2,
        "BLOCKED_SPREAD_GAP_VISIBLE": 3,
        "NO_VALID_PROOF_PATH": 4,
    }
    return order.get(str(value), 9)


def _has_replay_or_proof_pack_artifact_source(row: dict[str, Any]) -> bool:
    sources = _string_list(row.get("all_mapped_evidence_sources")) + _string_list(row.get("source_refs"))
    return any(("replay" in source or "proof_pack" in source) for source in sources)


def _target_pair_priority(pair: str, *, target_pairs: tuple[str, ...]) -> int:
    try:
        return target_pairs.index(pair)
    except ValueError:
        return len(target_pairs)


def _codes_from_blockers(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            code = item.get("code") or item.get("reason_code") or item.get("check_name")
            if code:
                codes.append(str(code))
        elif item:
            codes.append(str(item))
    return codes


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_str(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _float(value)
        if number is not None:
            return number
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_number_or_none(value: Any) -> int | float | None:
    number = _float(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return round(number, 6)
