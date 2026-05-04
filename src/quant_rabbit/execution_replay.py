from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import DEFAULT_EXECUTION_REPLAY, DEFAULT_EXECUTION_REPLAY_REPORT, DEFAULT_ORDER_INTENTS


@dataclass(frozen=True)
class ReplayOrderResult:
    lane_id: str
    pair: str
    side: str
    order_type: str
    status: str
    fill_timestamp_utc: str | None
    exit_timestamp_utc: str | None
    exit_reason: str | None
    pl_jpy: float
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionReplaySummary:
    output_path: Path
    report_path: Path
    status: str
    orders: int
    filled: int
    closed: int
    target_hit: bool
    net_pl_jpy: float


class ExecutionReplayer:
    """Replay current order receipts over a supplied tick/quote path."""

    def __init__(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        price_path: Path,
        output_path: Path = DEFAULT_EXECUTION_REPLAY,
        report_path: Path = DEFAULT_EXECUTION_REPLAY_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.price_path = price_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self, *, target_jpy: float = 0.0, lane_id: str | None = None) -> ExecutionReplaySummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents = _load_json(self.intents_path)
        ticks = _load_ticks(self.price_path)
        candidates = [
            item
            for item in intents.get("results", []) or []
            if isinstance(item, dict)
            and isinstance(item.get("intent"), dict)
            and item.get("status") == "LIVE_READY"
            and (lane_id is None or item.get("lane_id") == lane_id)
        ]
        results = tuple(_replay_order(item, ticks) for item in candidates)
        net_pl = _round(sum(item.pl_jpy for item in results))
        target_hit = target_jpy > 0 and net_pl >= target_jpy
        blockers = tuple(_blockers(results, target_jpy, target_hit))
        status = "TARGET_HIT" if target_hit else "REPLAY_COMPLETE"
        if blockers:
            status = "BLOCKED"
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "intents_path": str(self.intents_path),
            "price_path": str(self.price_path),
            "target_jpy": target_jpy,
            "net_pl_jpy": net_pl,
            "target_hit": target_hit,
            "blockers": list(blockers),
            "results": [asdict(item) for item in results],
        }
        self._write_output(payload)
        self._write_report(payload)
        return ExecutionReplaySummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            orders=len(results),
            filled=sum(1 for item in results if item.fill_timestamp_utc is not None),
            closed=sum(1 for item in results if item.exit_timestamp_utc is not None),
            target_hit=target_hit,
            net_pl_jpy=net_pl,
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Execution Replay Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Target: `{payload['target_jpy']:.0f} JPY`",
            f"- Net PnL: `{payload['net_pl_jpy']:.0f} JPY`",
            f"- Target hit: `{payload['target_hit']}`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Orders", ""])
        for item in payload["results"]:
            lines.append(
                f"- `{item['lane_id']}` status=`{item['status']}` fill=`{item['fill_timestamp_utc']}` "
                f"exit=`{item['exit_timestamp_utc']}` reason=`{item['exit_reason']}` pl=`{item['pl_jpy']:.0f}`"
            )
        lines.extend(
            [
                "",
                "## Replay Contract",
                "",
                "- This is an offline quote-path replay. It creates no broker orders.",
                "- Pending entries must fill before TP/SL is evaluated.",
                "- A replay target hit is evidence for certification, not live profit assurance.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _replay_order(result: dict[str, Any], ticks_by_pair: dict[str, list[dict[str, Any]]]) -> ReplayOrderResult:
    lane_id = str(result.get("lane_id") or "")
    intent = result["intent"]
    pair = str(intent.get("pair") or "")
    side = str(intent.get("side") or "")
    order_type = str(intent.get("order_type") or "")
    ticks = ticks_by_pair.get(pair, [])
    entry = _optional_float(intent.get("entry"))
    tp = float(intent.get("tp"))
    sl = float(intent.get("sl"))
    units = int(intent.get("units") or 0)
    filled = False
    fill_ts: str | None = None
    fill_price: float | None = None
    for tick in ticks:
        bid = float(tick["bid"])
        ask = float(tick["ask"])
        ts = str(tick.get("timestamp_utc") or tick.get("timestamp") or "")
        if not filled:
            if _fills(side, order_type, entry, bid, ask):
                filled = True
                fill_ts = ts
                fill_price = _fill_price(side, order_type, entry, bid, ask)
            else:
                continue
        exit_reason = _exit_reason(side, bid, ask, tp, sl)
        if exit_reason:
            pl, blocker = _pl_jpy(
                pair,
                side,
                units,
                fill_price if fill_price is not None else _market_entry(side, bid, ask),
                tp if exit_reason == "TP" else sl,
                ticks_by_pair,
                ts,
            )
            return ReplayOrderResult(
                lane_id=lane_id,
                pair=pair,
                side=side,
                order_type=order_type,
                status="BLOCKED" if blocker else "CLOSED",
                fill_timestamp_utc=fill_ts,
                exit_timestamp_utc=ts,
                exit_reason=exit_reason,
                pl_jpy=pl,
                blockers=(blocker,) if blocker else (),
            )
    return ReplayOrderResult(
        lane_id=lane_id,
        pair=pair,
        side=side,
        order_type=order_type,
        status="OPEN_AT_END" if filled else "NOT_FILLED",
        fill_timestamp_utc=fill_ts,
        exit_timestamp_utc=None,
        exit_reason=None,
        pl_jpy=0.0,
    )


def _fills(side: str, order_type: str, entry: float | None, bid: float, ask: float) -> bool:
    if order_type == "MARKET":
        return True
    if entry is None:
        return False
    if side == "LONG" and order_type == "STOP-ENTRY":
        return ask >= entry
    if side == "SHORT" and order_type == "STOP-ENTRY":
        return bid <= entry
    if side == "LONG" and order_type == "LIMIT":
        return ask <= entry
    if side == "SHORT" and order_type == "LIMIT":
        return bid >= entry
    return False


def _exit_reason(side: str, bid: float, ask: float, tp: float, sl: float) -> str | None:
    if side == "LONG":
        if bid <= sl:
            return "SL"
        if bid >= tp:
            return "TP"
    else:
        if ask >= sl:
            return "SL"
        if ask <= tp:
            return "TP"
    return None


def _pl_jpy(
    pair: str,
    side: str,
    units: int,
    entry: float,
    exit_price: float,
    ticks_by_pair: dict[str, list[dict[str, Any]]],
    timestamp: str,
) -> tuple[float, str | None]:
    pip_factor = _pip_factor(pair)
    pips = (exit_price - entry) * pip_factor if side == "LONG" else (entry - exit_price) * pip_factor
    quote_pl = pips * (units / pip_factor)
    conversion = _quote_to_jpy(pair, quote_pl, ticks_by_pair, timestamp)
    if conversion is None:
        quote_ccy = pair.split("_", 1)[1]
        return 0.0, f"{quote_ccy}_JPY conversion tick is required to replay broker-truth PnL for {pair}"
    return _round(quote_pl * conversion), None


def _market_entry(side: str, bid: float, ask: float) -> float:
    return ask if side == "LONG" else bid


def _fill_price(side: str, order_type: str, entry: float | None, bid: float, ask: float) -> float:
    if order_type == "MARKET":
        return _market_entry(side, bid, ask)
    if entry is not None:
        return entry
    return _market_entry(side, bid, ask)


def _blockers(results: tuple[ReplayOrderResult, ...], target_jpy: float, target_hit: bool) -> list[str]:
    blockers: list[str] = []
    if not results:
        blockers.append("no LIVE_READY order receipts were available to replay")
    for result in results:
        blockers.extend(result.blockers)
    not_filled = sum(1 for item in results if item.status == "NOT_FILLED")
    open_at_end = sum(1 for item in results if item.status == "OPEN_AT_END")
    if not_filled:
        blockers.append(f"{not_filled} orders never filled on supplied quote path")
    if open_at_end:
        blockers.append(f"{open_at_end} orders remained open without TP/SL resolution")
    if target_jpy > 0 and not target_hit:
        blockers.append("replay net PnL did not reach target")
    return blockers


def _load_ticks(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows = payload.get("ticks") if isinstance(payload, dict) else payload
    ticks: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pair = str(row.get("pair") or "")
        if not pair:
            continue
        ticks.setdefault(pair, []).append(row)
    return ticks


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _quote_to_jpy(
    pair: str,
    quote_pl: float,
    ticks_by_pair: dict[str, list[dict[str, Any]]],
    timestamp: str,
) -> float | None:
    quote_ccy = pair.split("_", 1)[1]
    if quote_ccy == "JPY":
        return 1.0
    tick = _conversion_tick_at(ticks_by_pair.get(f"{quote_ccy}_JPY", []), timestamp)
    if tick is None:
        return None
    return float(tick["bid"] if quote_pl >= 0 else tick["ask"])


def _conversion_tick_at(ticks: list[dict[str, Any]], timestamp: str) -> dict[str, Any] | None:
    if not ticks:
        return None
    selected: dict[str, Any] | None = None
    for tick in ticks:
        tick_ts = str(tick.get("timestamp_utc") or tick.get("timestamp") or "")
        if not timestamp or not tick_ts or tick_ts <= timestamp:
            selected = tick
    return selected or ticks[0]


def _round(value: float) -> float:
    return round(value, 4)
