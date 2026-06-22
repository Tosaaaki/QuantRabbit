from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    repair_replay_contract_from_payload,
)
from quant_rabbit.models import BrokerPosition, Owner, Quote, Side
from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_PROFIT_CAPTURE_BOT,
    DEFAULT_PROFIT_CAPTURE_BOT_REPORT,
)
from quant_rabbit.strategy.position_manager import (
    ACTION_TAKE_PROFIT_MARKET,
    TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT,
    TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS,
    _executable_profit_pips,
    _indicator_float,
    _pip_factor,
    _position_tp_pips,
    _spread_pips,
    _view_by_timeframe,
)


STATUS_READY = "PROFIT_CAPTURE_READY"
STATUS_WATCH = "PROFIT_CAPTURE_WATCH"
STATUS_BLOCKED = "PROFIT_CAPTURE_BLOCKED"


@dataclass(frozen=True)
class ProfitCaptureSummary:
    status: str
    output_path: Path
    report_path: Path
    metrics: dict[str, Any]
    positions: list[dict[str, Any]]
    blockers: list[dict[str, Any]]


class ProfitCaptureBot:
    """Read-only microscope for TP-progress profit capture.

    The bot mirrors the current PositionManager TP-progress gates and explains
    why each open trader-owned position is bankable, watched, or blocked. It
    never sends closes, changes TP/SL, cancels orders, or loads support agents.
    """

    def __init__(
        self,
        *,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_guardian_management_path: Path = DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        output_path: Path = DEFAULT_PROFIT_CAPTURE_BOT,
        report_path: Path = DEFAULT_PROFIT_CAPTURE_BOT_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.broker_snapshot_path = broker_snapshot_path
        self.pair_charts_path = pair_charts_path
        self.position_management_path = position_management_path
        self.position_guardian_management_path = position_guardian_management_path
        self.execution_timing_audit_path = execution_timing_audit_path
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> ProfitCaptureSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return ProfitCaptureSummary(
            status=payload["status"],
            output_path=self.output_path,
            report_path=self.report_path,
            metrics=payload["metrics"],
            positions=payload["positions"],
            blockers=payload["blockers"],
        )

    def build_payload(self) -> dict[str, Any]:
        broker = _read_json(self.broker_snapshot_path)
        pair_charts = _read_json(self.pair_charts_path)
        position_management = _read_json(self.position_management_path)
        guardian_management = _read_json(self.position_guardian_management_path)
        timing = _read_json(self.execution_timing_audit_path)
        artifact_blockers = _artifact_missing_blockers(
            {
                "broker_snapshot": broker,
                "pair_charts": pair_charts,
                "position_management": position_management,
                "position_guardian_management": guardian_management,
                "execution_timing_audit": timing,
            }
        )

        chart_by_pair = _pair_chart_map(pair_charts)
        position_actions = _position_action_map(position_management)
        guardian_actions = _position_action_map(guardian_management)
        positions = [
            _position_capture_state(
                raw_position=raw,
                quote=_quote_for_pair(broker, str(raw.get("pair") or "")),
                pair_chart=chart_by_pair.get(str(raw.get("pair") or "")),
                position_action=position_actions.get(str(raw.get("trade_id") or "")),
                guardian_action=guardian_actions.get(str(raw.get("trade_id") or "")),
            )
            for raw in broker.get("positions", []) or []
            if isinstance(raw, dict) and str(raw.get("owner") or "").lower() == Owner.TRADER.value
        ]

        history = _historical_capture_summary(timing)
        blockers = artifact_blockers + _blockers(positions=positions, history=history)
        bankable = [item for item in positions if item["gate_status"] == "BANKABLE_NOW"]
        blocked_positions = [item for item in positions if item["gate_status"] == "BLOCKED_MISSING_INPUT"]
        status = STATUS_READY if bankable else STATUS_BLOCKED if blockers else STATUS_WATCH
        metrics = {
            "open_trader_positions": len(positions),
            "bankable_positions": len(bankable),
            "blocked_positions": len(blocked_positions),
            "watch_positions": len(positions) - len(bankable) - len(blocked_positions),
            "historical_missed_loss_closes": history["missed_loss_closes"],
            "historical_estimated_gap_jpy": history["estimated_gap_jpy"],
            "historical_actual_loss_close_pl_jpy": history["actual_loss_close_pl_jpy"],
            "historical_counterfactual_profit_capture_pl_jpy": history[
                "counterfactual_profit_capture_pl_jpy"
            ],
            "historical_counterfactual_profit_capture_delta_jpy": history[
                "counterfactual_profit_capture_delta_jpy"
            ],
            "historical_counterfactual_profit_capture_jpy": history[
                "counterfactual_profit_capture_jpy"
            ],
            "historical_repair_replay_contract": history["repair_replay_contract"],
            "historical_repair_replay_contract_present": history[
                "repair_replay_contract_present"
            ],
            "historical_repair_replay_triggered": history["repair_replay_triggered"],
            "historical_repair_replay_profit_capture_jpy": history[
                "repair_replay_profit_capture_jpy"
            ],
            "historical_repair_replay_counterfactual_pl_jpy": history[
                "repair_replay_counterfactual_pl_jpy"
            ],
            "historical_repair_replay_delta_jpy": history["repair_replay_delta_jpy"],
        }
        return {
            "artifact_paths": {
                "broker_snapshot": str(self.broker_snapshot_path),
                "pair_charts": str(self.pair_charts_path),
                "position_management": str(self.position_management_path),
                "position_guardian_management": str(self.position_guardian_management_path),
                "execution_timing_audit": str(self.execution_timing_audit_path),
            },
            "generated_at_utc": self.now_utc.isoformat(),
            "status": status,
            "metrics": metrics,
            "positions": positions,
            "history": history,
            "blockers": blockers,
            "operator_actions": _operator_actions(status=status, positions=positions, history=history),
            "read_only": True,
            "live_side_effects": [],
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _artifact_missing_blockers(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for name, payload in artifacts.items():
        if payload.get("_missing"):
            blockers.append(
                {
                    "code": f"{name.upper()}_MISSING",
                    "severity": "P0",
                    "message": f"profit-capture diagnosis cannot evaluate current gates: {payload.get('_path')} missing",
                }
            )
    return blockers


def _pair_chart_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    charts = payload.get("charts") if isinstance(payload.get("charts"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for chart in charts:
        if isinstance(chart, dict) and isinstance(chart.get("pair"), str):
            out[chart["pair"]] = chart
    return out


def _position_action_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("trade_id") is not None:
            out[str(row["trade_id"])] = row
    return out


def _quote_for_pair(snapshot: dict[str, Any], pair: str) -> Quote | None:
    quotes = snapshot.get("quotes")
    raw = quotes.get(pair) if isinstance(quotes, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        return Quote(pair=pair, bid=float(raw["bid"]), ask=float(raw["ask"]))
    except (KeyError, TypeError, ValueError):
        return None


def _broker_position(raw: dict[str, Any]) -> BrokerPosition | None:
    try:
        return BrokerPosition(
            trade_id=str(raw["trade_id"]),
            pair=str(raw["pair"]),
            side=Side(str(raw["side"]).upper()),
            units=int(float(raw.get("units") or 0)),
            entry_price=float(raw["entry_price"]),
            unrealized_pl_jpy=float(raw.get("unrealized_pl_jpy") or 0.0),
            take_profit=_float_or_none(raw.get("take_profit")),
            stop_loss=_float_or_none(raw.get("stop_loss")),
            owner=Owner.TRADER,
            raw=raw,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _position_capture_state(
    *,
    raw_position: dict[str, Any],
    quote: Quote | None,
    pair_chart: dict[str, Any] | None,
    position_action: dict[str, Any] | None,
    guardian_action: dict[str, Any] | None,
) -> dict[str, Any]:
    position = _broker_position(raw_position)
    if position is None:
        return {
            "trade_id": raw_position.get("trade_id"),
            "pair": raw_position.get("pair"),
            "gate_status": "BLOCKED_MISSING_INPUT",
            "blocker_codes": ["POSITION_PARSE_FAILED"],
            "reasons": ["position row cannot be parsed into broker position"],
        }

    reasons: list[str] = []
    blocker_codes: list[str] = []
    executable_profit_pips = _executable_profit_pips(position, quote)
    spread_pips = _spread_pips(position.pair, quote)
    m1 = _view_by_timeframe(pair_chart or {}, "M1") if isinstance(pair_chart, dict) else None
    m1_atr_pips = _indicator_float(m1, "atr_pips")
    tp_pips = _position_tp_pips(position)
    noise_floor = None
    if spread_pips is not None and m1_atr_pips is not None:
        noise_floor = max(spread_pips * TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT, m1_atr_pips)
    progress = (
        executable_profit_pips / tp_pips
        if executable_profit_pips is not None and tp_pips is not None and tp_pips > 0
        else None
    )
    trigger = _capture_trigger(position=position, noise_floor_pips=noise_floor, tp_pips=tp_pips)

    if quote is None:
        blocker_codes.append("QUOTE_MISSING")
        reasons.append("TP-progress capture cannot evaluate: quote missing")
    if position.take_profit is None:
        blocker_codes.append("ATTACHED_TP_MISSING")
        reasons.append("TP-progress capture cannot evaluate: attached broker TP missing")
    if not isinstance(pair_chart, dict):
        blocker_codes.append("PAIR_CHART_MISSING")
        reasons.append("TP-progress capture cannot evaluate: pair chart missing")
    if spread_pips is None:
        blocker_codes.append("SPREAD_MISSING")
        reasons.append("TP-progress capture cannot evaluate: spread missing")
    if m1_atr_pips is None:
        blocker_codes.append("M1_ATR_MISSING")
        reasons.append("TP-progress capture cannot evaluate: M1 ATR missing")
    if tp_pips is None or tp_pips <= 0:
        blocker_codes.append("TP_DISTANCE_MISSING")
        reasons.append("TP-progress capture cannot evaluate: attached TP distance unavailable")

    if blocker_codes:
        gate_status = "BLOCKED_MISSING_INPUT"
    elif executable_profit_pips is None or executable_profit_pips <= 0:
        gate_status = "WATCH_NOT_PROFITABLE"
        reasons.append("not profitable on executable bid/ask yet")
    elif noise_floor is not None and executable_profit_pips < noise_floor:
        gate_status = "WATCH_BELOW_NOISE"
        reasons.append(
            f"executable profit {executable_profit_pips:.1f}pip below market noise floor {noise_floor:.1f}pip"
        )
    elif progress is not None and progress < TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS:
        gate_status = "WATCH_BELOW_TP_PROGRESS"
        reasons.append(
            f"TP progress {progress:.0%} below gate {TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS:.0%}"
        )
    else:
        gate_status = "BANKABLE_NOW"
        reasons.append("TP-progress TAKE_PROFIT_MARKET gate is satisfied now")

    action = (position_action or {}).get("action")
    guardian = (guardian_action or {}).get("action")
    if action == ACTION_TAKE_PROFIT_MARKET or guardian == ACTION_TAKE_PROFIT_MARKET:
        reasons.append("current sidecar already emits TAKE_PROFIT_MARKET for this trade")

    return {
        "trade_id": position.trade_id,
        "pair": position.pair,
        "side": position.side.value,
        "units": position.units,
        "unrealized_pl_jpy": _round(position.unrealized_pl_jpy, 3),
        "entry_price": position.entry_price,
        "take_profit": position.take_profit,
        "stop_loss": position.stop_loss,
        "position_management_action": action,
        "guardian_management_action": guardian,
        "gate_status": gate_status,
        "blocker_codes": blocker_codes,
        "executable_profit_pips": _round(executable_profit_pips, 3),
        "spread_pips": _round(spread_pips, 3),
        "m1_atr_pips": _round(m1_atr_pips, 3),
        "noise_floor_pips": _round(noise_floor, 3),
        "attached_tp_pips": _round(tp_pips, 3),
        "tp_progress": _round(progress, 4),
        "tp_progress_gate": TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS,
        "capture_trigger": trigger,
        "reasons": reasons,
    }


def _capture_trigger(
    *,
    position: BrokerPosition,
    noise_floor_pips: float | None,
    tp_pips: float | None,
) -> dict[str, Any] | None:
    if noise_floor_pips is None or tp_pips is None or tp_pips <= 0:
        return None
    required_profit_pips = max(noise_floor_pips, tp_pips * TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS)
    if position.side == Side.LONG:
        price = position.entry_price + required_profit_pips / _pip_factor(position.pair)
        quote_side = "bid"
        comparator = ">="
    else:
        price = position.entry_price - required_profit_pips / _pip_factor(position.pair)
        quote_side = "ask"
        comparator = "<="
    return {
        "quote_side": quote_side,
        "comparator": comparator,
        "price": _round(price, 5),
        "required_profit_pips": _round(required_profit_pips, 3),
    }


def _historical_capture_summary(timing: dict[str, Any]) -> dict[str, Any]:
    summary = timing.get("summary") if isinstance(timing.get("summary"), dict) else {}
    repair_replay_contract = repair_replay_contract_from_payload(timing)
    repair_replay_contract_present = repair_replay_contract == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
    rows = timing.get("loss_close_regrets") if isinstance(timing.get("loss_close_regrets"), list) else []
    top = [
        {
            "trade_id": row.get("trade_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "exit_reason": row.get("exit_reason"),
            "realized_pl_jpy": _round(row.get("realized_pl_jpy"), 3),
            "tp_progress_before_close": _round(row.get("tp_progress_before_loss_close"), 4),
            "counterfactual_exit": row.get("profit_capture_counterfactual_exit"),
            "counterfactual_pips": _round(row.get("profit_capture_counterfactual_pips"), 4),
            "counterfactual_jpy": _round(row.get("profit_capture_counterfactual_jpy"), 3),
            "counterfactual_delta_jpy": _round(
                row.get("profit_capture_counterfactual_net_improvement_jpy"),
                3,
            ),
        }
        for row in rows
        if isinstance(row, dict) and row.get("profit_capture_missed_before_loss_close")
    ][:5]
    top_repair = [
        {
            "trade_id": row.get("trade_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "exit_reason": row.get("exit_reason"),
            "realized_pl_jpy": _round(row.get("realized_pl_jpy"), 3),
            "repair_replay_exit": row.get("repair_replay_exit"),
            "repair_trigger_at_utc": row.get("repair_replay_trigger_at_utc"),
            "repair_profit_pips": _round(row.get("repair_replay_profit_pips"), 4),
            "repair_noise_floor_pips": _round(row.get("repair_replay_noise_floor_pips"), 4),
            "repair_counterfactual_jpy": _round(row.get("repair_replay_counterfactual_jpy"), 3),
            "repair_counterfactual_delta_jpy": _round(
                row.get("repair_replay_counterfactual_net_improvement_jpy"),
                3,
            ),
        }
        for row in rows
        if isinstance(row, dict) and row.get("repair_replay_triggered_before_loss_close")
    ][:5]
    return {
        "generated_at_utc": timing.get("generated_at_utc"),
        "repair_replay_contract": repair_replay_contract,
        "repair_replay_contract_present": repair_replay_contract_present,
        "missed_loss_closes": int(_float(summary.get("loss_closes_profit_capture_missed"))),
        "missed_stop_loss_closes": int(_float(summary.get("stop_loss_closes_profit_capture_missed"))),
        "estimated_gap_jpy": _round(summary.get("loss_close_estimated_capture_gap_jpy"), 3),
        "actual_loss_close_pl_jpy": _round(summary.get("loss_close_actual_pl_jpy"), 3),
        "counterfactual_profit_capture_pl_jpy": _round(
            summary.get("loss_close_counterfactual_profit_capture_pl_jpy"),
            3,
        ),
        "counterfactual_profit_capture_delta_jpy": _round(
            summary.get("loss_close_counterfactual_profit_capture_delta_jpy"),
            3,
        ),
        "counterfactual_profit_capture_jpy": _round(
            summary.get("loss_close_counterfactual_profit_capture_jpy"),
            3,
        ),
        "avg_decision_lag_minutes_after_first_positive": _round(
            summary.get("avg_decision_lag_minutes_after_first_positive"), 3
        ),
        "repair_replay_triggered": int(_float(summary.get("loss_closes_repair_replay_triggered"))),
        "repair_replay_profit_capture_jpy": _round(
            summary.get("loss_close_repair_replay_profit_capture_jpy"),
            3,
        ),
        "repair_replay_counterfactual_pl_jpy": _round(
            summary.get("loss_close_repair_replay_counterfactual_pl_jpy"),
            3,
        ),
        "repair_replay_delta_jpy": _round(
            summary.get("loss_close_repair_replay_delta_jpy"),
            3,
        ),
        "top_misses": top,
        "top_repair_replay_triggers": top_repair,
    }


def _blockers(*, positions: list[dict[str, Any]], history: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    blocked_positions = [item for item in positions if item["gate_status"] == "BLOCKED_MISSING_INPUT"]
    if blocked_positions:
        blockers.append(
            {
                "code": "PROFIT_CAPTURE_INPUT_MISSING",
                "severity": "P0",
                "message": f"{len(blocked_positions)} open trader position(s) cannot evaluate TP-progress capture",
            }
        )
    if history["missed_loss_closes"] > 0:
        blockers.append(
            {
                "code": "HISTORICAL_PROFIT_CAPTURE_MISSED",
                "severity": "P0",
                "message": _historical_miss_message(history),
            }
        )
    return blockers


def _historical_miss_message(history: dict[str, Any]) -> str:
    base = f"{history['missed_loss_closes']} recent loss close(s) missed executable profit capture"
    repair_triggers = int(history.get("repair_replay_triggered") or 0)
    if not history.get("repair_replay_contract_present"):
        return (
            f"{base}; production-gate replay sidecar is stale/missing "
            f"{TP_PROGRESS_REPAIR_REPLAY_CONTRACT}"
        )
    if repair_triggers > 0:
        delta = history.get("repair_replay_delta_jpy")
        if delta is None:
            return f"{base}; production-gate replay triggers={repair_triggers}"
        return f"{base}; production-gate replay triggers={repair_triggers} delta={delta} JPY"
    delta = history.get("counterfactual_profit_capture_delta_jpy")
    if delta is None:
        return base
    return f"{base}; conservative candle counterfactual delta={delta} JPY"


def _operator_actions(
    *,
    status: str,
    positions: list[dict[str, Any]],
    history: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "code": "REFRESH_PROFIT_CAPTURE_BOT",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profit-capture-bot",
            "requires_explicit_operator_approval": False,
            "reason": "refresh read-only TP-progress capture diagnosis",
        }
    ]
    if status == STATUS_READY:
        actions.append(
            {
                "code": "RUN_POSITION_EXECUTION_GATEWAY",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli position-execution",
                "requires_explicit_operator_approval": False,
                "reason": "dry-run the already computed position-management action before any live send",
            }
        )
    if history["missed_loss_closes"] > 0:
        actions.append(
            {
                "code": "REFRESH_EXECUTION_TIMING_AUDIT",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
                "requires_explicit_operator_approval": False,
                "reason": "confirm historical TP-progress misses are shrinking after repairs",
            }
        )
    if any(item["gate_status"] == "BLOCKED_MISSING_INPUT" for item in positions):
        actions.append(
            {
                "code": "REFRESH_POSITION_PACKET",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli cycle-sidecars",
                "requires_explicit_operator_approval": False,
                "reason": "refresh broker snapshot, pair charts, and position-management evidence",
            }
        )
    return actions


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Profit Capture Bot Report",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Read only: `{payload['read_only']}`",
        f"- Live side effects: `{len(payload['live_side_effects'])}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload["metrics"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Open Trader Positions", ""])
    if payload["positions"]:
        lines.extend(
            [
                "| Trade | Pair | Side | Gate | UPL JPY | Profit pips | TP progress | Trigger |",
                "|---|---|---|---|---:|---:|---:|---|",
            ]
        )
        for item in payload["positions"]:
            trigger = item.get("capture_trigger") or {}
            trigger_text = (
                f"{trigger.get('quote_side')} {trigger.get('comparator')} {trigger.get('price')}"
                if trigger
                else "n/a"
            )
            lines.append(
                f"| `{item['trade_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['gate_status']}` | `{item['unrealized_pl_jpy']}` | "
                f"`{item['executable_profit_pips']}` | `{item['tp_progress']}` | `{trigger_text}` |"
            )
            for reason in item["reasons"][:4]:
                lines.append(f"|  |  |  |  |  |  |  | {reason} |")
    else:
        lines.append("- none")
    lines.extend(["", "## Historical Misses", ""])
    history = payload["history"]
    lines.append(f"- Missed loss closes: `{history['missed_loss_closes']}`")
    lines.append(f"- Estimated gap JPY: `{history['estimated_gap_jpy']}`")
    lines.append(f"- Actual loss-close PL JPY: `{history['actual_loss_close_pl_jpy']}`")
    lines.append(
        "- Counterfactual profit-capture PL JPY: "
        f"`{history['counterfactual_profit_capture_pl_jpy']}`"
    )
    lines.append(
        "- Counterfactual profit-capture delta JPY: "
        f"`{history['counterfactual_profit_capture_delta_jpy']}`"
    )
    lines.append(f"- Production-gate replay triggers: `{history['repair_replay_triggered']}`")
    lines.append(f"- Production-gate replay contract: `{history['repair_replay_contract']}`")
    lines.append(
        f"- Production-gate replay contract present: `{history['repair_replay_contract_present']}`"
    )
    lines.append(f"- Production-gate replay delta JPY: `{history['repair_replay_delta_jpy']}`")
    if history["top_repair_replay_triggers"]:
        for item in history["top_repair_replay_triggers"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"{item['exit_reason']} repair_at=`{item.get('repair_trigger_at_utc')}` "
                f"repair_jpy=`{item.get('repair_counterfactual_jpy')}` "
                f"delta=`{item.get('repair_counterfactual_delta_jpy')}`"
            )
    if history["top_misses"]:
        for item in history["top_misses"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"{item['exit_reason']} realized=`{item['realized_pl_jpy']}` "
                f"counterfactual=`{item.get('counterfactual_jpy')}` "
                f"delta=`{item.get('counterfactual_delta_jpy')}`"
            )
    lines.extend(["", "## Blockers", ""])
    if payload["blockers"]:
        for blocker in payload["blockers"]:
            lines.append(f"- `{blocker['severity']}` `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Actions", ""])
    for action in payload["operator_actions"]:
        approval = "yes" if action["requires_explicit_operator_approval"] else "no"
        lines.append(f"- `{action['code']}` approval=`{approval}`: `{action['command']}`")
    lines.append("")
    return "\n".join(lines)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
