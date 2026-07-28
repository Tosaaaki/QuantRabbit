"""Read-only PAPER room inventory and economic visibility.

The reporter reads only local diagnostic PAPER-room files and the immutable
fresh-model handoff.  It never imports a broker client, applies a model action,
or estimates a metric that the source artifacts cannot prove.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from quant_rabbit.dojo_fresh_model_handoff import (
    DojoFreshModelHandoffError,
    handoff_status,
    verify_decision_packet,
    verify_model_response,
)
from quant_rabbit.dojo_paired_model_queue import canonical_sha256

REPORT_CONTRACT: Final = "QR_DOJO_PAPER_INVENTORY_REPORT_V1"
MAX_LEDGER_BYTES: Final = 32 * 1024 * 1024
JST: Final = ZoneInfo("Asia/Tokyo")
REALIZED_EVENTS: Final = frozenset(
    {"CLOSE", "EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT", "FORCED_LIQUIDATION"}
)
FORCED_EVENTS: Final = frozenset({"MARGIN_CLOSEOUT", "FORCED_LIQUIDATION"})


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 2 * 1024 * 1024:
        raise DojoFreshModelHandoffError(f"invalid report JSON source: {path}")
    with resolved.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DojoFreshModelHandoffError(f"report JSON root must be object: {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DojoFreshModelHandoffError(f"{label} must be an aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoFreshModelHandoffError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise DojoFreshModelHandoffError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _active(contract: Mapping[str, Any], now_utc: datetime) -> bool:
    if (
        contract.get("contract") != "QR_VIRTUAL_MARKET_SESSION_V2"
        or contract.get("proof_mode") != "diagnostic"
        or contract.get("proof_eligible") is not False
        or contract.get("authority")
        != {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        }
    ):
        raise DojoFreshModelHandoffError("paper report room authority is invalid")
    source = contract.get("source")
    if not isinstance(source, Mapping):
        raise DojoFreshModelHandoffError("paper report room source is missing")
    start = _parse_utc(source.get("window_start_utc"), "room start")
    end = _parse_utc(source.get("window_end_utc"), "room end")
    return start <= now_utc < end


def _read_verified_ledger(path: Path, expected_tip: str) -> list[dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise DojoFreshModelHandoffError("paper ledger must be a regular file")
    if resolved.stat().st_size > MAX_LEDGER_BYTES:
        raise DojoFreshModelHandoffError("paper ledger exceeds report byte bound")
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DojoFreshModelHandoffError("paper ledger row must be object")
            unsigned = {key: item for key, item in value.items() if key != "sha"}
            if value.get("prev_sha") != previous:
                raise DojoFreshModelHandoffError("paper ledger prev_sha mismatch")
            if value.get("sha") != canonical_sha256(unsigned):
                raise DojoFreshModelHandoffError("paper ledger row hash mismatch")
            previous = str(value["sha"])
            rows.append(value)
    if previous != expected_tip:
        raise DojoFreshModelHandoffError("paper ledger tip does not match snapshot")
    return rows


def _latest_decision(
    runtime_root: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    responses: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for path in sorted((runtime_root / "responses").glob("*.json")):
        response = _read_json(path)
        packet = verify_decision_packet(
            _read_json(
                runtime_root
                / "packets"
                / f"{response.get('decision_packet_sha256')}.json"
            )
        )
        verified = verify_model_response(response, packet)
        responses.append(
            (
                int(packet["rolling_story"]["story_sequence"]),
                packet,
                verified,
            )
        )
    if not responses:
        return {
            "action": "HOLD",
            "reason": "NO_ACCEPTED_PAPER_AI_DECISION",
            "next_review": "HOURLY",
            "packet": None,
            "response_sha256": None,
        }
    _, packet, response = max(responses, key=lambda item: item[0])
    if status["accepted_fresh_model_decision_count"] < 1:
        raise DojoFreshModelHandoffError("response exists without accepted state")
    return {
        "action": response["action"],
        "reason": ",".join(response["reason_ids"]),
        "next_review": response["next_story_content"]["next_review"],
        "packet": packet,
        "response_sha256": response["response_sha256"],
    }


def _previous_room(packet: Mapping[str, Any] | None, room_id: str) -> dict[str, Any] | None:
    if packet is None:
        return None
    rooms = packet.get("snapshot", {}).get("rooms", [])
    if not isinstance(rooms, list):
        return None
    return next(
        (dict(room) for room in rooms if room.get("room_id") == room_id),
        None,
    )


def _sum_or_none(values: Sequence[float | None]) -> float | None:
    return sum(value for value in values if value is not None) if all(
        value is not None for value in values
    ) else None


def _room_report(
    *,
    room_dir: Path,
    contract: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    room_id = str(contract.get("room_id") or "")
    snapshot = _read_json(room_dir / "broker_snapshot.json")
    state = _read_json(room_dir / "state.json")
    account = state.get("account")
    quotes = state.get("quotes")
    positions = snapshot.get("positions")
    orders = snapshot.get("orders")
    if (
        not isinstance(account, Mapping)
        or not isinstance(quotes, Mapping)
        or not isinstance(positions, list)
        or not isinstance(orders, list)
    ):
        raise DojoFreshModelHandoffError("paper report room state is invalid")
    counter_trend_orders = 0
    counter_trend_orders_measured = True
    for raw in orders:
        if not isinstance(raw, Mapping):
            raise DojoFreshModelHandoffError("paper report order is invalid")
        side = str(raw.get("side") or "")
        entry_context = raw.get("entry_context")
        if side not in {"LONG", "SHORT"} or not isinstance(entry_context, Mapping):
            counter_trend_orders_measured = False
            continue
        trend = str(entry_context.get("trend_24h") or "")
        if trend not in {"LONG", "SHORT", "FLAT", "UNKNOWN"}:
            counter_trend_orders_measured = False
            continue
        if trend in {"LONG", "SHORT"} and side != trend:
            counter_trend_orders += 1
    ledger = _read_verified_ledger(
        room_dir / "ledger.jsonl",
        str(snapshot.get("ledger_sha") or ""),
    )
    long_units = 0.0
    short_units = 0.0
    entry_weight = 0.0
    entry_units = 0.0
    mark_weight = 0.0
    unrealized = 0.0
    unrealized_measured = True
    for raw in positions:
        if not isinstance(raw, Mapping):
            raise DojoFreshModelHandoffError("paper report position is invalid")
        side = str(raw.get("side") or "")
        pair = str(raw.get("pair") or "")
        units = _finite(raw.get("units"))
        entry = _finite(raw.get("entry_price"))
        quote = quotes.get(pair)
        if (
            side not in {"LONG", "SHORT"}
            or units is None
            or units <= 0
            or entry is None
            or not isinstance(quote, Mapping)
        ):
            raise DojoFreshModelHandoffError("paper report position fields invalid")
        mark = _finite(quote.get("bid" if side == "LONG" else "ask"))
        if side == "LONG":
            long_units += units
        else:
            short_units += units
        entry_weight += entry * units
        entry_units += units
        if mark is None:
            unrealized_measured = False
        else:
            mark_weight += mark * units
            if pair.endswith("_JPY"):
                unrealized += (mark - entry) * units * (1 if side == "LONG" else -1)
            else:
                unrealized_measured = False
    if not positions:
        configured = list(contract.get("pairs") or [])
        if len(configured) == 1 and isinstance(quotes.get(configured[0]), Mapping):
            quote = quotes[configured[0]]
            bid = _finite(quote.get("bid"))
            ask = _finite(quote.get("ask"))
            current_mark = None if bid is None or ask is None else (bid + ask) / 2
        else:
            current_mark = None
    else:
        current_mark = mark_weight / entry_units if unrealized_measured else None

    realized_values: list[float] = []
    tp_rows: list[Mapping[str, Any]] = []
    normal_loss = 0.0
    forced_loss = 0.0
    explicit_cost_values: list[float] = []
    explicit_cost_complete = True
    for row in ledger:
        event = str(row.get("event") or "")
        payload = row.get("payload")
        payload_map = payload if isinstance(payload, Mapping) else {}
        if event not in REALIZED_EVENTS:
            continue
        pl = _finite(payload_map.get("pl_jpy"))
        if pl is not None:
            realized_values.append(pl)
            if pl < 0:
                if event in FORCED_EVENTS:
                    forced_loss += -pl
                else:
                    normal_loss += -pl
        if event == "EXIT_TP":
            tp_rows.append(payload_map)
        financing = _finite(payload_map.get("financing_jpy"))
        execution = _finite(payload_map.get("execution_cost_jpy"))
        if financing is None and execution is None:
            explicit_cost_complete = False
        else:
            explicit_cost_values.append(abs(financing or 0.0) + abs(execution or 0.0))

    costs = contract.get("costs")
    zero_cost_contract = (
        isinstance(costs, Mapping)
        and _finite(costs.get("financing_pips_per_day")) == 0
        and _finite(costs.get("slippage_pips_per_fill")) == 0
    )
    tp_gross_values: list[float] = []
    tp_gross_measured = True
    for payload in tp_rows:
        gross = _finite(payload.get("gross_pl_jpy"))
        if gross is None and zero_cost_contract:
            gross = _finite(payload.get("pl_jpy"))
        if gross is None:
            tp_gross_measured = False
        else:
            tp_gross_values.append(gross)
    tp_gross = sum(tp_gross_values) if tp_gross_measured else None
    execution_financing = (
        0.0
        if zero_cost_contract
        else (sum(explicit_cost_values) if explicit_cost_complete else None)
    )
    balance = _finite(account.get("balance_jpy"))
    equity = _finite(account.get("equity_jpy"))
    margin_used = _finite(account.get("margin_used_jpy"))
    margin_ratio = _finite(account.get("margin_usage"))
    margin_available = (
        equity - margin_used
        if equity is not None and margin_used is not None
        else None
    )
    previous = _previous_room(decision.get("packet"), room_id)
    previous_balance = None if previous is None else _finite(previous.get("balance_jpy"))
    balance_delta = (
        balance - previous_balance
        if balance is not None and previous_balance is not None
        else None
    )
    bot = contract.get("bot")
    strategy_tags = (
        list(bot.get("strategy_tags") or []) if isinstance(bot, Mapping) else []
    )
    if positions:
        bot_state = "OPEN_POSITION"
    elif orders:
        bot_state = "RESTING_ORDER"
    else:
        bot_state = "FLAT"
    return {
        "room_id": room_id,
        "strategy": ",".join(str(item) for item in strategy_tags) or "未計測",
        "bot_state": bot_state,
        "position_count": len(positions),
        "resting_order_count": len(orders),
        "counter_trend_resting_order_count": (
            counter_trend_orders if counter_trend_orders_measured else None
        ),
        "long_units": long_units,
        "short_units": short_units,
        "net_units": long_units - short_units,
        "avg_entry": entry_weight / entry_units if entry_units else None,
        "current_mark": current_mark,
        "unrealized_pl_jpy": unrealized if unrealized_measured else None,
        "realized_pl_jpy": sum(realized_values),
        "tp_gross_jpy": tp_gross,
        "normal_loss_jpy": normal_loss,
        "forced_liquidation_loss_jpy": forced_loss,
        "execution_financing_jpy": execution_financing,
        "missed_profit_jpy": None,
        "ai_cost_jpy": None,
        "balance_jpy": balance,
        "equity_jpy": equity,
        "margin_used_jpy": margin_used,
        "margin_available_jpy": margin_available,
        "margin_ratio": margin_ratio,
        "last_ai_action": decision["action"],
        "reason": decision["reason"],
        "next_review": decision["next_review"],
        "balance_delta_since_previous_decision_jpy": balance_delta,
        "position_summary": (
            f"L {long_units:.2f} / S {short_units:.2f} / net {long_units-short_units:.2f}"
        ),
        "data_at_utc": _parse_utc(
            state.get("wall_time_utc"), "paper room data time"
        ).isoformat(),
    }


def build_paper_inventory_report(
    *,
    runtime_root: Path,
    rooms_root: Path,
    now_utc: datetime,
) -> dict[str, Any]:
    """Build a fully read-only report from the four active PAPER rooms."""

    if now_utc.tzinfo is None:
        raise DojoFreshModelHandoffError("paper report time must be aware")
    now = now_utc.astimezone(timezone.utc)
    status = handoff_status(runtime_root)
    decision = _latest_decision(runtime_root, status)
    active: list[tuple[Path, dict[str, Any]]] = []
    resolved = rooms_root.resolve(strict=True)
    for contract_path in sorted(resolved.glob("*/*/session_contract.json")):
        contract = _read_json(contract_path)
        if _active(contract, now):
            active.append((contract_path.parent, contract))
    if len(active) != 4:
        raise DojoFreshModelHandoffError(
            f"paper inventory supervisor requires exactly four active rooms, got {len(active)}"
        )
    rooms = [
        _room_report(room_dir=room_dir, contract=contract, decision=decision)
        for room_dir, contract in active
    ]
    data_at = min(_parse_utc(room["data_at_utc"], "room data time") for room in rooms)
    total_unrealized = _sum_or_none([room["unrealized_pl_jpy"] for room in rooms])
    total_realized = _sum_or_none([room["realized_pl_jpy"] for room in rooms])
    total_tp = _sum_or_none([room["tp_gross_jpy"] for room in rooms])
    total_normal_loss = _sum_or_none([room["normal_loss_jpy"] for room in rooms])
    total_forced = _sum_or_none(
        [room["forced_liquidation_loss_jpy"] for room in rooms]
    )
    total_execution = _sum_or_none(
        [room["execution_financing_jpy"] for room in rooms]
    )
    total_missed = _sum_or_none([room["missed_profit_jpy"] for room in rooms])
    total_ai_cost = _sum_or_none([room["ai_cost_jpy"] for room in rooms])
    economic_components = [
        total_tp,
        total_normal_loss,
        total_forced,
        total_missed,
        total_execution,
        total_ai_cost,
    ]
    economic_net = (
        total_tp
        - total_normal_loss
        - total_forced
        - total_missed
        - total_execution
        - total_ai_cost
        if all(value is not None for value in economic_components)
        else None
    )
    total_net_pl = _sum_or_none([total_unrealized, total_realized])
    runtime_health_status = (
        "HEALTHY"
        if status["state"] in {"IDLE_NO_READY_PACKET", "WAITING_FOR_FRESH_TASK"}
        else "HALTED_OR_DEGRADED"
    )
    profitability_status = (
        "UNDETERMINED"
        if total_net_pl is None
        else ("PROFITABLE" if total_net_pl > 0 else "LOSS_OR_FLAT")
    )
    body = {
        "contract": REPORT_CONTRACT,
        "schema_version": 1,
        "title": "PAPER AI inventory supervisor",
        "data_at_utc": data_at.isoformat(),
        "data_at_jst": data_at.astimezone(JST).isoformat(),
        "runtime_state": status["state"],
        "accepted_count": status["accepted_fresh_model_decision_count"],
        "room_count": len(rooms),
        "authority": "NONE",
        "runtime_health_status": runtime_health_status,
        "profitability_status": profitability_status,
        "inventory_observation": {
            "open_position_count": sum(room["position_count"] for room in rooms),
            "resting_order_count": sum(
                room["resting_order_count"] for room in rooms
            ),
            "counter_trend_resting_order_count": _sum_or_none(
                [room["counter_trend_resting_order_count"] for room in rooms]
            ),
        },
        "rooms": sorted(rooms, key=lambda room: room["room_id"]),
        "totals": {
            "balance_jpy": _sum_or_none([room["balance_jpy"] for room in rooms]),
            "nav_jpy": _sum_or_none([room["equity_jpy"] for room in rooms]),
            "net_pl_jpy": total_net_pl,
            "unrealized_pl_jpy": total_unrealized,
            "realized_pl_jpy": total_realized,
            "tp_gross_jpy": total_tp,
            "normal_loss_jpy": total_normal_loss,
            "forced_liquidation_loss_jpy": total_forced,
            "missed_profit_jpy": total_missed,
            "execution_financing_jpy": total_execution,
            "ai_cost_jpy": total_ai_cost,
            "economic_net_jpy": economic_net,
            "economic_result_status": (
                "DETERMINED" if economic_net is not None else "UNDETERMINED"
            ),
            "margin_used_jpy": _sum_or_none(
                [room["margin_used_jpy"] for room in rooms]
            ),
            "margin_available_jpy": _sum_or_none(
                [room["margin_available_jpy"] for room in rooms]
            ),
            "margin_pressure": max(
                (
                    room["margin_ratio"]
                    for room in rooms
                    if room["margin_ratio"] is not None
                ),
                default=None,
            ),
            "previous_balance_delta_jpy": _sum_or_none(
                [
                    room["balance_delta_since_previous_decision_jpy"]
                    for room in rooms
                ]
            ),
        },
        "decision": {
            "action": decision["action"],
            "reason": decision["reason"],
            "next_review": decision["next_review"],
            "response_sha256": decision["response_sha256"],
            "effect": "SHADOW_ONLY_NOT_APPLIED_TO_POSITIONS",
        },
        "economic_formula": (
            "TP粗利益 − 通常損失 − 強制決済損失 − 逃した利益 "
            "− execution/financing − AIコスト"
        ),
        "missing_values_render_as": "未計測",
        "generated_at_utc": now.isoformat(),
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def _fmt(value: Any, *, digits: int = 2, percent: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "未計測"
    if percent:
        return f"{number * 100:.{digits}f}%"
    return f"{number:,.{digits}f}"


def render_paper_inventory_report(report: Mapping[str, Any]) -> str:
    """Render the fixed compact Codex task card and room table."""

    rooms = report["rooms"]
    totals = report["totals"]
    decision = report["decision"]
    lines = [
        "PAPER AI inventory supervisor",
        f"data_at JST: {report['data_at_jst']}",
        (
            f"runtime state: {report['runtime_state']} | accepted count: "
            f"{report['accepted_count']} | room count: {report['room_count']} "
            "| authority NONE"
        ),
        (
            f"稼働評価: {report['runtime_health_status']} | 収益評価: "
            f"{report['profitability_status']} | positions "
            f"{report['inventory_observation']['open_position_count']} | resting "
            f"orders {report['inventory_observation']['resting_order_count']} | "
            "counter-trend resting "
            f"{_fmt(report['inventory_observation']['counter_trend_resting_order_count'], digits=0)}"
        ),
        "",
        (
            "| room / strategy | bot state | long | short | net | avg entry | mark "
            "| unrealized P/L | realized P/L | TP gross | normal loss | "
            "margin used / available / ratio | last AI action | reason | next review |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"
        ),
    ]
    for room in rooms:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{room['room_id']}<br>{room['strategy']}",
                    room["bot_state"],
                    _fmt(room["long_units"]),
                    _fmt(room["short_units"]),
                    _fmt(room["net_units"]),
                    _fmt(room["avg_entry"], digits=5),
                    _fmt(room["current_mark"], digits=5),
                    _fmt(room["unrealized_pl_jpy"]),
                    _fmt(room["realized_pl_jpy"]),
                    _fmt(room["tp_gross_jpy"]),
                    _fmt(room["normal_loss_jpy"]),
                    (
                        f"{_fmt(room['margin_used_jpy'])} / "
                        f"{_fmt(room['margin_available_jpy'])} / "
                        f"{_fmt(room['margin_ratio'], percent=True)}"
                    ),
                    room["last_ai_action"],
                    room["reason"],
                    room["next_review"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "全室合計: balance "
                f"{_fmt(totals['balance_jpy'])} JPY / NAV {_fmt(totals['nav_jpy'])} "
                f"/ net P/L {_fmt(totals['net_pl_jpy'])} / unrealized "
                f"{_fmt(totals['unrealized_pl_jpy'])} / realized "
                f"{_fmt(totals['realized_pl_jpy'])}"
            ),
            (
                "収益分解: TP gross "
                f"{_fmt(totals['tp_gross_jpy'])} − normal loss "
                f"{_fmt(totals['normal_loss_jpy'])} − forced "
                f"{_fmt(totals['forced_liquidation_loss_jpy'])} − missed "
                f"{_fmt(totals['missed_profit_jpy'])} − execution/financing "
                f"{_fmt(totals['execution_financing_jpy'])} − AI cost "
                f"{_fmt(totals['ai_cost_jpy'])} = "
                f"{_fmt(totals['economic_net_jpy'])} "
                f"({totals['economic_result_status']})"
            ),
            (
                "margin pressure "
                f"{_fmt(totals['margin_pressure'], percent=True)} / 前回差 "
                f"{_fmt(totals['previous_balance_delta_jpy'])} JPY"
            ),
            (
                f"判断結果: {decision['action']} — {decision['reason']} "
                f"(shadow only, authority NONE; {decision['effect']})"
            ),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "REPORT_CONTRACT",
    "build_paper_inventory_report",
    "render_paper_inventory_report",
]
