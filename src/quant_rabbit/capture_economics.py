"""Capture-economics audit: is the exit machinery paying for the entries?

Reads trader-attributed realized outcomes from `data/execution_ledger.db`
(same gateway-entry attribution CTE as lane-history scoring, so manual /
tagless closes are excluded) and publishes the payoff arithmetic the daily
5% / 10% campaign actually depends on:

- win rate `p`, average win `W`, average loss `L`, payoff ratio `W/L`
- the breakeven payoff requirement `(1 - p) / p` at the observed win rate
- expectancy per trade in JPY and in % of the campaign per-trade budget
- the same metrics split by exit reason and by ISO week

This is an *audit surface*, not a trade gate: it cannot block lanes or
resize intents. It exists because the 2026-05-14→06-08 ledger showed 55 wins
averaging +376 JPY against 24 losses averaging -1,437 JPY (payoff 0.26 vs
breakeven 0.43 at the observed 70% win rate) — an asymmetry no forecast
hit-rate can outrun. The trader and the operator must see this number move
toward/over breakeven, or the 5% guaranteed floor (§5) has no arithmetic
route.
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
# so the 20-minute cycle sees the repair priorities without drowning out the
# current broker/intent evidence; this is an engineering display cap, not a
# market threshold.
EXIT_REPAIR_ITEM_LIMIT = 4

_ATTRIBUTED_REALIZED_SQL = """
WITH gateway_entries AS (
    SELECT trade_id, order_id, lane_id
    FROM execution_events
    WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
      AND lane_id IS NOT NULL AND lane_id != ''
),
entries AS (
    SELECT e.trade_id
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
    e.pair,
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
GROUP BY e.trade_id, e.pair
HAVING SUM(e.realized_pl_jpy) != 0
ORDER BY MAX(e.ts_utc) ASC
"""


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


def _bucket_metrics(rows: list[tuple[str, str, str, float]]) -> dict[str, Any]:
    wins = [r[3] for r in rows if r[3] > 0]
    losses = [r[3] for r in rows if r[3] < 0]
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


def build_capture_economics(
    *,
    ledger_path: Path,
    output_path: Path = DEFAULT_CAPTURE_ECONOMICS,
    report_path: Path = DEFAULT_CAPTURE_ECONOMICS_REPORT,
) -> CaptureEconomicsSummary:
    rows: list[tuple[str, str, str, float]] = []
    if ledger_path.exists():
        try:
            with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
                rows = [
                    (str(ts or ""), str(pair or ""), str(reason or ""), float(pl))
                    for ts, pair, reason, pl in conn.execute(_ATTRIBUTED_REALIZED_SQL)
                    if pl is not None
                ]
        except sqlite3.Error:
            rows = []

    overall = _bucket_metrics(rows)
    by_exit: dict[str, Any] = {}
    by_week: dict[str, Any] = {}
    for reason in sorted({r[2] for r in rows}):
        by_exit[reason] = _bucket_metrics([r for r in rows if r[2] == reason])
    for week in sorted({_iso_week(r[0]) for r in rows}):
        by_week[week] = _bucket_metrics([r for r in rows if _iso_week(r[0]) == week])

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
        "by_iso_week": by_week,
        "repair_summary": repair_summary,
        "action_items": action_items,
        "note": (
            "Advisory audit (AGENT_CONTRACT §8): payoff_ratio must reach "
            "breakeven_payoff_at_win_rate before the daily 5% floor has an "
            "arithmetic route. Not a trade gate."
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
