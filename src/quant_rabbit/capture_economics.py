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
SELECT e.ts_utc, e.pair, e.exit_reason, e.realized_pl_jpy
FROM execution_events e
INNER JOIN entries ON entries.trade_id = e.trade_id
WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
  AND e.realized_pl_jpy IS NOT NULL
  AND e.realized_pl_jpy != 0
ORDER BY e.ts_utc ASC
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
    if trades < MIN_SAMPLE_FOR_VERDICT:
        status = "LOW_SAMPLE"
    elif payoff is not None and breakeven is not None and payoff >= breakeven:
        status = "POSITIVE_EXPECTANCY"
    else:
        status = "NEGATIVE_EXPECTANCY"

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "status": status,
        "min_sample_for_verdict": MIN_SAMPLE_FOR_VERDICT,
        "overall": overall,
        "by_exit_reason": by_exit,
        "by_iso_week": by_week,
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
