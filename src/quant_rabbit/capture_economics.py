"""Capture-economics audit: is the exit machinery paying for the entries?

Reads trader-attributed realized outcomes from `data/execution_ledger.db`
(same gateway-entry attribution CTE as lane-history scoring, so manual /
tagless closes are excluded) and publishes the payoff arithmetic the daily
5% / 10% campaign actually depends on:

- win rate `p`, average win `W`, average loss `L`, payoff ratio `W/L`
- the breakeven payoff requirement `(1 - p) / p` at the observed win rate
- expectancy per trade in JPY and in % of the campaign per-trade budget
- the same metrics split by exit reason and by ISO week
- pair/side/method repair priorities that separate scoped broker-TP proof from
  MARKET_ORDER_TRADE_CLOSE leakage, so high rotation preserves the paying
  capture shape instead of scaling the lossy close path

This is first an audit surface: it does not select sides or grant permission.
When it reports NEGATIVE_EXPECTANCY with average losses larger than average
wins, intent generation consumes the observed average winner as a temporary
fresh-entry loss cap. That loss-asymmetry guard is the bounded-risk repair for
"one loss erases multiple wins"; it still cannot override forecast, spread,
strategy, margin, or gateway gates. It exists because the 2026-05-14→06-08 ledger showed 55 wins
averaging +376 JPY against 24 losses averaging -1,437 JPY (payoff 0.26 vs
breakeven 0.43 at the observed 70% win rate) — an asymmetry no forecast
hit-rate can outrun. The trader and the operator must see this number move
toward/over breakeven, or the +5% pace/protection marker (§5) has no arithmetic
route on days where a valid edge exists.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import ROOT

DEFAULT_CAPTURE_ECONOMICS = ROOT / "data" / "capture_economics.json"
DEFAULT_CAPTURE_ECONOMICS_REPORT = ROOT / "docs" / "capture_economics_report.md"

# Realized outcomes below this many attributed closes cannot produce a stable
# proportion estimate; the audit still reports them but flags LOW_SAMPLE.
# 20 keeps the binomial standard error under ~11pp at p=0.5 — a documented
# statistical floor, not a tuned market threshold.
MIN_SAMPLE_FOR_VERDICT = 20

# Report/action payloads are read by the trader prompt packet. Keep them short
# so the live cycle sees the repair priorities without drowning out the
# current broker/intent evidence; this is an engineering display cap, not a
# market threshold.
EXIT_REPAIR_ITEM_LIMIT = 4

# Pair/side/method repair rows are compact operator routing evidence, not a
# live-entry permission list. The cap keeps the prompt packet focused while the
# nested metrics above still retain the full realized bucket details.
SEGMENT_REPAIR_PRIORITY_LIMIT = 12
MARKET_CLOSE_LOSS_EXAMPLE_LIMIT = 5

TAKE_PROFIT_EXIT_REASON = "TAKE_PROFIT_ORDER"
MARKET_CLOSE_EXIT_REASON = "MARKET_ORDER_TRADE_CLOSE"
SYSTEM_ATTRIBUTION_SCOPE = "SYSTEM_GATEWAY_ATTRIBUTED_ONLY"

# Use the same statistical floor as the audit verdict for "scoped TP proof";
# RiskEngine and IntentGenerator mirror the live relaxation floor independently
# so manual/replayed receipts remain defended without importing this module.
SCOPED_TP_PROOF_MIN_EXIT_TRADES = MIN_SAMPLE_FOR_VERDICT

_ATTRIBUTED_REALIZED_SQL = """
WITH gateway_entries AS (
    SELECT trade_id, order_id, lane_id
    FROM execution_events
    WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
      AND lane_id IS NOT NULL AND lane_id != ''
),
entries AS (
    SELECT
        e.trade_id,
        COALESCE(NULLIF(MAX(e.pair), ''), '') AS pair,
        COALESCE(NULLIF(MAX(e.side), ''), '') AS side,
        COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS lane_id
    FROM execution_events e
    LEFT JOIN gateway_entries g
      ON (g.trade_id IS NOT NULL AND g.trade_id != '' AND g.trade_id = e.trade_id)
      OR (g.order_id IS NOT NULL AND g.order_id != '' AND g.order_id = e.order_id)
    WHERE e.event_type = 'ORDER_FILLED'
      AND e.trade_id IS NOT NULL AND e.trade_id != ''
    GROUP BY e.trade_id
    HAVING COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) IS NOT NULL
)
-- One row per TRADE outcome, not per close event: a position reduced three
-- times and then closed is one trade whose realized P/L is the SUM of its
-- partial closes. Counting each partial as a separate "win" inflated n and
-- biased the win rate upward (2026-06-10 audit finding). exit_reason and
-- ts_utc come from the FINAL close event of the trade.
SELECT
    MAX(e.ts_utc) AS ts_utc,
    e.trade_id AS trade_id,
    entries.pair,
    entries.side,
    entries.lane_id,
    (
        SELECT e2.exit_reason FROM execution_events e2
        WHERE e2.trade_id = e.trade_id
          AND e2.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
          AND e2.realized_pl_jpy IS NOT NULL
        ORDER BY e2.ts_utc DESC LIMIT 1
    ) AS exit_reason,
    SUM(e.realized_pl_jpy) AS realized_pl_jpy
FROM execution_events e
INNER JOIN entries ON entries.trade_id = e.trade_id
WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
  AND e.realized_pl_jpy IS NOT NULL
GROUP BY e.trade_id, entries.pair, entries.side, entries.lane_id
HAVING SUM(e.realized_pl_jpy) != 0
ORDER BY MAX(e.ts_utc) ASC
"""


@dataclass(frozen=True)
class RealizedOutcome:
    ts_utc: str
    trade_id: str
    pair: str
    side: str
    lane_id: str
    method: str
    exit_reason: str
    realized_pl_jpy: float


@dataclass(frozen=True)
class CaptureEconomicsSummary:
    output_path: Path
    report_path: Path
    status: str
    trades: int
    win_rate: float | None
    payoff_ratio: float | None
    breakeven_payoff: float | None
    expectancy_jpy: float | None


def _bucket_metrics(rows: list[RealizedOutcome]) -> dict[str, Any]:
    wins = [r.realized_pl_jpy for r in rows if r.realized_pl_jpy > 0]
    losses = [r.realized_pl_jpy for r in rows if r.realized_pl_jpy < 0]
    n = len(wins) + len(losses)
    if n == 0:
        return {"trades": 0}
    p = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    breakeven = ((1.0 - p) / p) if p > 0 else None
    expectancy = (sum(wins) + sum(losses)) / n
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(p, 4),
        "avg_win_jpy": round(avg_win, 1),
        "avg_loss_jpy": round(avg_loss, 1),
        "payoff_ratio": round(payoff, 3) if payoff is not None else None,
        "breakeven_payoff_at_win_rate": round(breakeven, 3) if breakeven is not None else None,
        "expectancy_jpy_per_trade": round(expectancy, 1),
        "net_jpy": round(sum(wins) + sum(losses), 1),
    }


def _nested_segment_metrics(
    rows: list[RealizedOutcome],
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[tuple[str, ...], list[RealizedOutcome]] = {}
    for row in rows:
        key = tuple(str(getattr(row, dimension) or "UNKNOWN") for dimension in dimensions)
        groups.setdefault(key, []).append(row)
    for key, bucket in sorted(groups.items()):
        cursor = out
        for part in key[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key[-1]] = _bucket_metrics(bucket)
    return out


def _lane_method(lane_id: str) -> str:
    parts = [part for part in str(lane_id or "").split(":") if part]
    if len(parts) >= 4:
        return parts[3]
    return "UNKNOWN"


def _iso_week(ts_utc: str) -> str:
    try:
        d = datetime.fromisoformat(ts_utc.replace("Z", "+00:00")).date()
    except ValueError:
        return "unknown"
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def _negative_exit_rows(by_exit: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for reason, metrics in by_exit.items():
        if not isinstance(metrics, dict):
            continue
        net = _optional_float(metrics.get("net_jpy"))
        trades = int(metrics.get("trades") or 0)
        if trades <= 0 or net is None or net >= 0:
            continue
        rows.append((reason, metrics))
    return sorted(rows, key=lambda item: float(item[1].get("net_jpy") or 0.0))


def _positive_exit_rows(by_exit: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for reason, metrics in by_exit.items():
        if not isinstance(metrics, dict):
            continue
        net = _optional_float(metrics.get("net_jpy"))
        trades = int(metrics.get("trades") or 0)
        if trades <= 0 or net is None or net <= 0:
            continue
        rows.append((reason, metrics))
    return sorted(rows, key=lambda item: float(item[1].get("net_jpy") or 0.0), reverse=True)


def _capture_repair_summary(
    *,
    status: str,
    overall: dict[str, Any],
    by_exit: dict[str, Any],
) -> dict[str, Any]:
    negative = _negative_exit_rows(by_exit)
    positive = _positive_exit_rows(by_exit)
    payoff = _optional_float(overall.get("payoff_ratio"))
    breakeven = _optional_float(overall.get("breakeven_payoff_at_win_rate"))
    summary: dict[str, Any] = {
        "status": status,
        "payoff_gap_to_breakeven": (
            round(max(0.0, breakeven - payoff), 3)
            if payoff is not None and breakeven is not None
            else None
        ),
        "top_negative_exit_reasons": [
            {
                "exit_reason": reason,
                "trades": int(metrics.get("trades") or 0),
                "net_jpy": metrics.get("net_jpy"),
                "expectancy_jpy_per_trade": metrics.get("expectancy_jpy_per_trade"),
                "win_rate": metrics.get("win_rate"),
                "payoff_ratio": metrics.get("payoff_ratio"),
            }
            for reason, metrics in negative[:EXIT_REPAIR_ITEM_LIMIT]
        ],
        "top_positive_exit_reasons": [
            {
                "exit_reason": reason,
                "trades": int(metrics.get("trades") or 0),
                "net_jpy": metrics.get("net_jpy"),
                "expectancy_jpy_per_trade": metrics.get("expectancy_jpy_per_trade"),
                "win_rate": metrics.get("win_rate"),
                "payoff_ratio": metrics.get("payoff_ratio"),
            }
            for reason, metrics in positive[:EXIT_REPAIR_ITEM_LIMIT]
        ],
    }
    if negative:
        reason, metrics = negative[0]
        summary["dominant_loss_exit_reason"] = reason
        summary["dominant_loss_exit_net_jpy"] = metrics.get("net_jpy")
        summary["dominant_loss_exit_expectancy_jpy_per_trade"] = metrics.get(
            "expectancy_jpy_per_trade"
        )
    if positive:
        reason, metrics = positive[0]
        summary["strongest_positive_exit_reason"] = reason
        summary["strongest_positive_exit_net_jpy"] = metrics.get("net_jpy")
    return summary


def _segment_repair_priorities(rows: list[RealizedOutcome]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[RealizedOutcome]] = {}
    for row in rows:
        key = (row.pair or "UNKNOWN", row.side or "UNKNOWN", row.method or "UNKNOWN")
        groups.setdefault(key, []).append(row)

    items: list[dict[str, Any]] = []
    for (pair, side, method), bucket in groups.items():
        overall = _bucket_metrics(bucket)
        tp_metrics = _bucket_metrics(
            [row for row in bucket if row.exit_reason == TAKE_PROFIT_EXIT_REASON]
        )
        market_close_metrics = _bucket_metrics(
            [row for row in bucket if row.exit_reason == MARKET_CLOSE_EXIT_REASON]
        )
        market_close_loss_rows = [
            row
            for row in bucket
            if row.exit_reason == MARKET_CLOSE_EXIT_REASON and row.realized_pl_jpy < 0
        ]

        tp_trades = int(tp_metrics.get("trades") or 0)
        tp_losses = int(tp_metrics.get("losses") or 0)
        tp_expectancy = _optional_float(tp_metrics.get("expectancy_jpy_per_trade"))
        tp_avg_win = _optional_float(tp_metrics.get("avg_win_jpy"))
        tp_proven = (
            tp_trades >= SCOPED_TP_PROOF_MIN_EXIT_TRADES
            and tp_expectancy is not None
            and tp_expectancy > 0
            and tp_avg_win is not None
            and tp_avg_win > 0
            and tp_losses <= 0
        )
        tp_positive_thin = (
            tp_trades > 0
            and not tp_proven
            and tp_expectancy is not None
            and tp_expectancy > 0
            and tp_avg_win is not None
            and tp_avg_win > 0
            and tp_losses <= 0
        )
        proof_gap = max(0, SCOPED_TP_PROOF_MIN_EXIT_TRADES - tp_trades)

        segment_net = _optional_float(overall.get("net_jpy"))
        market_close_net = _optional_float(market_close_metrics.get("net_jpy"))
        market_close_negative = market_close_net is not None and market_close_net < 0
        segment_negative = segment_net is not None and segment_net < 0

        if tp_proven and market_close_negative:
            priority_class = "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK"
            rank = 0
            next_action = (
                "preserve attached-TP HARVEST entries for this exact shape, but repair "
                "or avoid its MARKET_ORDER_TRADE_CLOSE path before increasing exposure"
            )
        elif market_close_negative and tp_positive_thin:
            priority_class = "COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK"
            rank = 1
            next_action = (
                "collect more scoped broker-TP outcomes and repair MARKET_ORDER_TRADE_CLOSE "
                "leakage before treating this as high-rotation proof"
            )
        elif market_close_negative:
            priority_class = "REPAIR_MARKET_CLOSE_LEAK"
            rank = 2
            next_action = (
                "rank the close provenance for this segment; do not widen fresh risk until "
                "loss-side MARKET_ORDER_TRADE_CLOSE evidence is repaired or explicitly justified"
            )
        elif tp_proven:
            priority_class = "PRESERVE_TP_PROVEN_SHAPE"
            rank = 3
            next_action = (
                "preserve this attached-TP capture shape while keeping forecast, spread, "
                "strategy-profile, margin, and gateway checks active"
            )
        elif tp_positive_thin:
            priority_class = "COLLECT_SCOPED_TP_PROOF"
            rank = 4
            next_action = (
                "treat as evidence-collection candidate: positive broker-TP outcomes exist "
                "but the scoped sample is below the proof floor"
            )
        elif segment_negative:
            priority_class = "AVOID_OR_REPRICE_SEGMENT"
            rank = 5
            next_action = (
                "avoid or reprice this segment until entry/exit geometry produces positive "
                "realized expectancy"
            )
        else:
            priority_class = "MONITOR_LOW_SAMPLE" if int(overall.get("trades") or 0) < MIN_SAMPLE_FOR_VERDICT else "MONITOR"
            rank = 6
            next_action = "monitor; no realized TP proof or market-close repair priority dominates yet"

        items.append(
            {
                "evidence_ref": f"capture:segment:{pair}:{side}:{method}",
                "attribution_scope": SYSTEM_ATTRIBUTION_SCOPE,
                "operator_manual_excluded": True,
                "should_count_against_system_edge": True,
                "pair": pair,
                "side": side,
                "method": method,
                "priority_class": priority_class,
                "next_action": next_action,
                "trades": int(overall.get("trades") or 0),
                "wins": int(overall.get("wins") or 0),
                "losses": int(overall.get("losses") or 0),
                "win_rate": overall.get("win_rate"),
                "expectancy_jpy_per_trade": overall.get("expectancy_jpy_per_trade"),
                "net_jpy": overall.get("net_jpy"),
                "take_profit_trades": tp_trades,
                "take_profit_wins": int(tp_metrics.get("wins") or 0),
                "take_profit_losses": tp_losses,
                "take_profit_expectancy_jpy": tp_metrics.get("expectancy_jpy_per_trade"),
                "take_profit_net_jpy": tp_metrics.get("net_jpy"),
                "take_profit_proof_floor": SCOPED_TP_PROOF_MIN_EXIT_TRADES,
                "take_profit_proof_gap_trades": proof_gap,
                "take_profit_proven": tp_proven,
                "market_close_trades": int(market_close_metrics.get("trades") or 0),
                "market_close_losses": int(market_close_metrics.get("losses") or 0),
                "market_close_loss_net_jpy": round(
                    sum(row.realized_pl_jpy for row in market_close_loss_rows),
                    1,
                ),
                "market_close_expectancy_jpy": market_close_metrics.get(
                    "expectancy_jpy_per_trade"
                ),
                "market_close_net_jpy": market_close_metrics.get("net_jpy"),
                "market_close_loss_trade_ids": [
                    row.trade_id
                    for row in sorted(
                        market_close_loss_rows,
                        key=lambda item: (item.realized_pl_jpy, item.ts_utc),
                    )[:MARKET_CLOSE_LOSS_EXAMPLE_LIMIT]
                ],
                "market_close_loss_examples": _market_close_loss_examples(
                    market_close_loss_rows
                ),
                "_sort_rank": rank,
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, int, str, str, str]:
        market_close_net = _optional_float(item.get("market_close_net_jpy"))
        segment_net = _optional_float(item.get("net_jpy"))
        proof_gap = int(item.get("take_profit_proof_gap_trades") or 0)
        return (
            float(item.get("_sort_rank") or 0),
            market_close_net if market_close_net is not None else 0.0,
            segment_net if segment_net is not None else 0.0,
            proof_gap,
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
        )

    sorted_items = sorted(items, key=_sort_key)
    for item in sorted_items:
        item.pop("_sort_rank", None)
    return {
        "basis": "trader-attributed realized outcomes grouped by pair|side|method",
        "take_profit_exit_reason": TAKE_PROFIT_EXIT_REASON,
        "market_close_exit_reason": MARKET_CLOSE_EXIT_REASON,
        "scoped_tp_proof_min_exit_trades": SCOPED_TP_PROOF_MIN_EXIT_TRADES,
        "total_segments": len(sorted_items),
        "items": sorted_items[:SEGMENT_REPAIR_PRIORITY_LIMIT],
    }


def _capture_action_items(
    *,
    status: str,
    overall: dict[str, Any],
    by_exit: dict[str, Any],
    repair_summary: dict[str, Any],
) -> list[str]:
    if status == "LOW_SAMPLE":
        return ["collect more trader-attributed realized exits before changing exit policy"]
    items: list[str] = []
    payoff = _optional_float(overall.get("payoff_ratio"))
    breakeven = _optional_float(overall.get("breakeven_payoff_at_win_rate"))
    if status == "NEGATIVE_EXPECTANCY":
        if payoff is not None and breakeven is not None:
            items.append(
                "repair exit payoff asymmetry before treating the daily target as arithmetically reachable: "
                f"payoff_ratio={payoff:.3f} breakeven={breakeven:.3f}"
            )
        dominant_reason = str(repair_summary.get("dominant_loss_exit_reason") or "")
        dominant_net = _optional_float(repair_summary.get("dominant_loss_exit_net_jpy"))
        if dominant_reason:
            net_text = f"{dominant_net:.1f} JPY" if dominant_net is not None else "net loss"
            if dominant_reason == "MARKET_ORDER_TRADE_CLOSE":
                items.append(
                    "contain MARKET_ORDER_TRADE_CLOSE drag "
                    f"({net_text}): prefer attached TP, TP-rebalance, profit-side TAKE_PROFIT_MARKET, "
                    "and require hard Gate A/B evidence for loss-side CLOSE"
                )
            else:
                items.append(f"repair dominant negative exit bucket {dominant_reason} ({net_text})")
    strongest_positive = str(repair_summary.get("strongest_positive_exit_reason") or "")
    if strongest_positive:
        items.append(
            f"preserve profitable {strongest_positive} behavior while repairing negative exit buckets"
        )
    return items[:EXIT_REPAIR_ITEM_LIMIT]


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _market_close_loss_examples(rows: list[RealizedOutcome]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.realized_pl_jpy, item.ts_utc))[
        :MARKET_CLOSE_LOSS_EXAMPLE_LIMIT
    ]:
        examples.append(
            {
                "trade_id": row.trade_id,
                "ts_utc": row.ts_utc,
                "lane_id": row.lane_id,
                "pair": row.pair,
                "side": row.side,
                "method": row.method,
                "exit_reason": row.exit_reason,
                "realized_pl_jpy": round(row.realized_pl_jpy, 4),
                "close_family": "SYSTEM_GATEWAY_MARKET_CLOSE",
                "attribution_scope": SYSTEM_ATTRIBUTION_SCOPE,
                "operator_manual_excluded": True,
                "should_count_against_system_edge": True,
            }
        )
    return examples


def build_capture_economics(
    *,
    ledger_path: Path,
    output_path: Path = DEFAULT_CAPTURE_ECONOMICS,
    report_path: Path = DEFAULT_CAPTURE_ECONOMICS_REPORT,
) -> CaptureEconomicsSummary:
    rows: list[RealizedOutcome] = []
    if ledger_path.exists():
        try:
            with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
                rows = [
                    RealizedOutcome(
                        ts_utc=str(ts or ""),
                        trade_id=str(trade_id or ""),
                        pair=str(pair or "UNKNOWN"),
                        side=str(side or "UNKNOWN"),
                        lane_id=str(lane_id or ""),
                        method=_lane_method(str(lane_id or "")),
                        exit_reason=str(reason or "UNKNOWN"),
                        realized_pl_jpy=float(pl),
                    )
                    for ts, trade_id, pair, side, lane_id, reason, pl in conn.execute(
                        _ATTRIBUTED_REALIZED_SQL
                    )
                    if pl is not None
                ]
        except sqlite3.Error:
            rows = []

    overall = _bucket_metrics(rows)
    by_exit: dict[str, Any] = {}
    by_week: dict[str, Any] = {}
    for reason in sorted({r.exit_reason for r in rows}):
        by_exit[reason] = _bucket_metrics([r for r in rows if r.exit_reason == reason])
    for week in sorted({_iso_week(r.ts_utc) for r in rows}):
        by_week[week] = _bucket_metrics([r for r in rows if _iso_week(r.ts_utc) == week])
    by_pair_side_exit = _nested_segment_metrics(rows, ("pair", "side", "exit_reason"))
    by_pair_side_method_exit = _nested_segment_metrics(rows, ("pair", "side", "method", "exit_reason"))

    trades = int(overall.get("trades") or 0)
    payoff = overall.get("payoff_ratio")
    breakeven = overall.get("breakeven_payoff_at_win_rate")
    expectancy = overall.get("expectancy_jpy_per_trade")
    if trades < MIN_SAMPLE_FOR_VERDICT:
        status = "LOW_SAMPLE"
    elif payoff is not None and breakeven is not None and payoff >= breakeven:
        status = "POSITIVE_EXPECTANCY"
    elif payoff is None and expectancy is not None and expectancy > 0:
        # Zero losses in the sample: payoff is undefined (division by the
        # empty loss side) but the expectancy is unambiguously positive.
        status = "POSITIVE_EXPECTANCY"
    else:
        status = "NEGATIVE_EXPECTANCY"
    repair_summary = _capture_repair_summary(status=status, overall=overall, by_exit=by_exit)
    segment_repair_priorities = _segment_repair_priorities(rows)
    action_items = _capture_action_items(
        status=status,
        overall=overall,
        by_exit=by_exit,
        repair_summary=repair_summary,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "status": status,
        "min_sample_for_verdict": MIN_SAMPLE_FOR_VERDICT,
        "overall": overall,
        "by_exit_reason": by_exit,
        "by_pair_side_exit_reason": by_pair_side_exit,
        "by_pair_side_method_exit_reason": by_pair_side_method_exit,
        "by_iso_week": by_week,
        "repair_summary": repair_summary,
        "segment_repair_priorities": segment_repair_priorities,
        "action_items": action_items,
        "note": (
            "Advisory audit (AGENT_CONTRACT §8): payoff_ratio must reach "
            "breakeven_payoff_at_win_rate before the daily 5% floor has an "
            "arithmetic route. When status is NEGATIVE_EXPECTANCY and avg_loss_jpy "
            "exceeds avg_win_jpy, generate-intents caps fresh NEW-entry loss at "
            "the observed average winner until payoff repair clears."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Capture Economics Report",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- Status: `{status}`",
        f"- Trades (trader-attributed, realized): `{trades}`",
    ]
    if trades:
        lines += [
            f"- Win rate: `{(overall.get('win_rate') or 0) * 100:.1f}%`",
            f"- Avg win / avg loss: `{overall.get('avg_win_jpy')}` / `{overall.get('avg_loss_jpy')}` JPY",
            f"- Payoff ratio: `{overall.get('payoff_ratio')}` (breakeven at win rate: `{overall.get('breakeven_payoff_at_win_rate')}`)",
            f"- Expectancy: `{overall.get('expectancy_jpy_per_trade')}` JPY/trade, net `{overall.get('net_jpy')}` JPY",
            "",
            "## Repair Summary",
            "",
            f"- Dominant loss exit: `{repair_summary.get('dominant_loss_exit_reason') or 'none'}` "
            f"net `{repair_summary.get('dominant_loss_exit_net_jpy')}` JPY",
            f"- Strongest positive exit: `{repair_summary.get('strongest_positive_exit_reason') or 'none'}` "
            f"net `{repair_summary.get('strongest_positive_exit_net_jpy')}` JPY",
            f"- Payoff gap to breakeven: `{repair_summary.get('payoff_gap_to_breakeven')}`",
            "",
            "## Segment Repair Priorities",
            "",
            "| pair | side | method | priority | n | TP n/gap | market-close net | net |",
            "|---|---|---|---|---|---|---|---|",
            *[
                (
                    f"| `{item.get('pair')}` | `{item.get('side')}` | `{item.get('method')}` "
                    f"| `{item.get('priority_class')}` | {item.get('trades')} "
                    f"| {item.get('take_profit_trades')}/{item.get('take_profit_proof_gap_trades')} "
                    f"| {item.get('market_close_net_jpy')} | {item.get('net_jpy')} |"
                )
                for item in segment_repair_priorities.get("items", [])
            ],
            "",
            "## Action Items",
            "",
            *[f"- {item}" for item in action_items],
            "",
            "## By exit reason",
            "",
            "| exit_reason | n | win% | avg win | avg loss | net |",
            "|---|---|---|---|---|---|",
        ]
        for reason, m in by_exit.items():
            if not m.get("trades"):
                continue
            lines.append(
                f"| `{reason}` | {m['trades']} | {(m.get('win_rate') or 0) * 100:.0f}% "
                f"| {m.get('avg_win_jpy')} | {m.get('avg_loss_jpy')} | {m.get('net_jpy')} |"
            )
        lines += ["", "## By ISO week", "", "| week | n | win% | payoff | net |", "|---|---|---|---|---|"]
        for week, m in by_week.items():
            if not m.get("trades"):
                continue
            lines.append(
                f"| `{week}` | {m['trades']} | {(m.get('win_rate') or 0) * 100:.0f}% "
                f"| {m.get('payoff_ratio')} | {m.get('net_jpy')} |"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")

    return CaptureEconomicsSummary(
        output_path=output_path,
        report_path=report_path,
        status=status,
        trades=trades,
        win_rate=overall.get("win_rate"),
        payoff_ratio=payoff,
        breakeven_payoff=breakeven,
        expectancy_jpy=overall.get("expectancy_jpy_per_trade"),
    )
