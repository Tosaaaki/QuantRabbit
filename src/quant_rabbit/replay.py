from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from quant_rabbit.paths import (
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_HISTORY_DB,
    DEFAULT_REPLAY_BACKTEST,
    DEFAULT_REPLAY_BACKTEST_REPORT,
)
from quant_rabbit.risk import RiskPolicy


@dataclass(frozen=True)
class ReplayDay:
    session_date: str
    target_jpy: float
    historical_net_jpy: float
    risk_capped_net_jpy: float
    captured_profit_jpy: float
    captured_loss_jpy: float
    pretrade_positive_jpy: float
    missed_positive_jpy: float
    evidence_coverage_jpy: float
    evidence_coverage_pct: float
    trade_count: int
    pretrade_count: int
    missed_count: int
    over_cap_loss_count: int
    worst_loss_jpy: float | None
    status: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ReplayBacktestSummary:
    output_path: Path
    report_path: Path
    days: int
    target_jpy: float
    historical_target_hits: int
    evidence_target_covered: int
    risk_repair_days: int
    missed_edge_days: int
    total_historical_net_jpy: float
    total_risk_capped_net_jpy: float


@dataclass(frozen=True)
class ResolvedReplayLossCap:
    loss_cap_jpy: float
    source: str
    daily_risk_pct: float
    target_trades_per_day: int


class ReplayBacktester:
    """Replay legacy daily evidence against the vNext target and risk contract."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_HISTORY_DB,
        output_path: Path = DEFAULT_REPLAY_BACKTEST,
        report_path: Path = DEFAULT_REPLAY_BACKTEST_REPORT,
        max_loss_jpy: float | None = None,
        daily_risk_pct: float | None = None,
        target_trades_per_day: int | None = None,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    ) -> None:
        self.db_path = db_path
        self.output_path = output_path
        self.report_path = report_path
        self.max_loss_jpy = max_loss_jpy
        self.daily_risk_pct = daily_risk_pct
        self.target_trades_per_day = target_trades_per_day
        self.target_state_path = target_state_path

    def run(
        self,
        *,
        start_balance_jpy: float,
        target_return_pct: float = 10.0,
        max_days: int | None = None,
    ) -> ReplayBacktestSummary:
        target_jpy = round(start_balance_jpy * (target_return_pct / 100.0), 2)
        cap = _resolve_replay_loss_cap(
            start_balance_jpy=start_balance_jpy,
            explicit_max_loss_jpy=self.max_loss_jpy,
            explicit_daily_risk_pct=self.daily_risk_pct,
            explicit_target_trades_per_day=self.target_trades_per_day,
            target_state_path=self.target_state_path,
        )
        days = tuple(self._load_days(target_jpy=target_jpy, max_days=max_days, max_loss_jpy=cap.loss_cap_jpy))
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "generated_at_utc": generated_at,
            "db_path": str(self.db_path),
            "start_balance_jpy": start_balance_jpy,
            "target_return_pct": target_return_pct,
            "target_jpy": target_jpy,
            "max_loss_jpy": cap.loss_cap_jpy,
            "loss_cap_source": cap.source,
            "daily_risk_pct": cap.daily_risk_pct,
            "target_trades_per_day": cap.target_trades_per_day,
            "days": [asdict(day) for day in days],
            "summary": _summary_payload(days, target_jpy),
        }
        self._write_output(payload)
        self._write_report(generated_at, start_balance_jpy, target_return_pct, target_jpy, days, cap)
        summary_payload = payload["summary"]
        return ReplayBacktestSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            days=len(days),
            target_jpy=target_jpy,
            historical_target_hits=int(summary_payload["historical_target_hits"]),
            evidence_target_covered=int(summary_payload["evidence_target_covered"]),
            risk_repair_days=int(summary_payload["risk_repair_days"]),
            missed_edge_days=int(summary_payload["missed_edge_days"]),
            total_historical_net_jpy=float(summary_payload["total_historical_net_jpy"]),
            total_risk_capped_net_jpy=float(summary_payload["total_risk_capped_net_jpy"]),
        )

    def _load_days(self, *, target_jpy: float, max_days: int | None, max_loss_jpy: float) -> Iterable[ReplayDay]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            trade_days = _query_trade_days(conn, max_loss_jpy)
            pretrade = _query_positive_by_day(conn, "pretrade_outcomes")
            missed = _query_positive_by_day(conn, "seat_outcomes")
            live_log = _query_live_log_by_day(conn)

        all_dates = sorted(set(trade_days) | set(pretrade) | set(missed) | set(live_log))
        if max_days is not None:
            all_dates = all_dates[-max_days:]
        for day in all_dates:
            trade = trade_days.get(day, {})
            pre = pretrade.get(day, {})
            miss = missed.get(day, {})
            live = live_log.get(day, {})
            historical_net = _round(_number(trade.get("historical_net_jpy"), live.get("historical_net_jpy"), 0.0))
            risk_capped_net = _round(_number(trade.get("risk_capped_net_jpy"), historical_net))
            captured_profit = _round(_number(trade.get("captured_profit_jpy"), live.get("captured_profit_jpy"), 0.0))
            captured_loss = _round(_number(trade.get("captured_loss_jpy"), live.get("captured_loss_jpy"), 0.0))
            pretrade_positive = _round(float(pre.get("positive_jpy") or 0.0))
            missed_positive = _round(float(miss.get("positive_jpy") or 0.0))
            evidence_coverage = _round(captured_profit + max(pretrade_positive, missed_positive))
            blockers = tuple(
                _blockers(
                    target_jpy=target_jpy,
                    historical_net=historical_net,
                    risk_capped_net=risk_capped_net,
                    evidence_coverage=evidence_coverage,
                    missed_positive=missed_positive,
                    captured_profit=captured_profit,
                    over_cap_loss_count=int(trade.get("over_cap_loss_count") or 0),
                    worst_loss=_optional_float(trade.get("worst_loss_jpy")),
                    max_loss_jpy=max_loss_jpy,
                )
            )
            yield ReplayDay(
                session_date=day,
                target_jpy=target_jpy,
                historical_net_jpy=historical_net,
                risk_capped_net_jpy=risk_capped_net,
                captured_profit_jpy=captured_profit,
                captured_loss_jpy=captured_loss,
                pretrade_positive_jpy=pretrade_positive,
                missed_positive_jpy=missed_positive,
                evidence_coverage_jpy=evidence_coverage,
                evidence_coverage_pct=_round((evidence_coverage / target_jpy) * 100.0) if target_jpy else 0.0,
                trade_count=int(trade.get("trade_count") or live.get("trade_count") or 0),
                pretrade_count=int(pre.get("count") or 0),
                missed_count=int(miss.get("count") or 0),
                over_cap_loss_count=int(trade.get("over_cap_loss_count") or 0),
                worst_loss_jpy=_optional_float(trade.get("worst_loss_jpy")),
                status=_status(
                    target_jpy=target_jpy,
                    historical_net=historical_net,
                    risk_capped_net=risk_capped_net,
                    evidence_coverage=evidence_coverage,
                    blockers=blockers,
                ),
                blockers=blockers,
            )

    def _write_output(self, payload: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        generated_at: str,
        start_balance_jpy: float,
        target_return_pct: float,
        target_jpy: float,
        days: tuple[ReplayDay, ...],
        cap: "ResolvedReplayLossCap",
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        summary = _summary_payload(days, target_jpy)
        lines = [
            "# Replay Backtest Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- History DB: `{self.db_path}`",
            f"- Start balance: `{start_balance_jpy:.0f} JPY`",
            f"- Target: `{target_jpy:.0f} JPY` (`{target_return_pct:.1f}%`)",
            f"- Per-trade loss cap: `{cap.loss_cap_jpy:.0f} JPY` (`{cap.source}`)",
            f"- Days replayed: `{summary['days']}`",
            f"- Historical target hits: `{summary['historical_target_hits']}`",
            f"- Evidence target covered: `{summary['evidence_target_covered']}`",
            f"- Risk repair days: `{summary['risk_repair_days']}`",
            f"- Missed-edge days: `{summary['missed_edge_days']}`",
            f"- Total historical net: `{summary['total_historical_net_jpy']:.0f} JPY`",
            f"- Total risk-capped net: `{summary['total_risk_capped_net_jpy']:.0f} JPY`",
            "",
            "## Replay Contract",
            "",
            "- This is an evidence replay over imported legacy trade outcomes, not tick-level execution replay.",
            "- Losses worse than the current equity-derived per-trade loss cap are marked as risk-repair requirements.",
            "- Positive pretrade and missed-seat outcomes are counted as coverage evidence, not automatic live permission.",
            "- A day is complete only when target coverage becomes executable receipts under broker truth and risk gates.",
            "",
            "## Daily Results",
            "",
        ]
        for day in days:
            lines.append(
                f"- `{day.session_date}` status=`{day.status}` hist=`{day.historical_net_jpy:.0f}` "
                f"risk_capped=`{day.risk_capped_net_jpy:.0f}` coverage=`{day.evidence_coverage_jpy:.0f}` "
                f"coverage_pct=`{day.evidence_coverage_pct:.1f}` trades=`{day.trade_count}`"
            )
            for blocker in day.blockers[:4]:
                lines.append(f"  - blocker: {blocker}")
        self.report_path.write_text("\n".join(lines) + "\n")


def _query_trade_days(conn: sqlite3.Connection, max_loss_jpy: float) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        """
        SELECT
            session_date,
            COUNT(*) trade_count,
            ROUND(SUM(pl), 4) historical_net_jpy,
            ROUND(SUM(CASE WHEN pl < ? THEN ? ELSE pl END), 4) risk_capped_net_jpy,
            ROUND(SUM(CASE WHEN pl > 0 THEN pl ELSE 0 END), 4) captured_profit_jpy,
            ROUND(SUM(CASE WHEN pl < 0 THEN pl ELSE 0 END), 4) captured_loss_jpy,
            SUM(CASE WHEN pl < ? THEN 1 ELSE 0 END) over_cap_loss_count,
            MIN(pl) worst_loss_jpy
        FROM legacy_records
        WHERE source_table='trades' AND session_date IS NOT NULL AND pl IS NOT NULL
        GROUP BY session_date
        """,
        (-max_loss_jpy, -max_loss_jpy, -max_loss_jpy),
    ).fetchall()
    return {str(row["session_date"]): dict(row) for row in rows}


def _resolve_replay_loss_cap(
    *,
    start_balance_jpy: float,
    explicit_max_loss_jpy: float | None,
    explicit_daily_risk_pct: float | None,
    explicit_target_trades_per_day: int | None,
    target_state_path: Path,
) -> ResolvedReplayLossCap:
    """Resolve replay's per-trade loss cap without falling back to a JPY literal.

    Replay is a campaign diagnostic, so its cap must move with the campaign:
    `start_balance * daily_risk_pct / target_trades_per_day`. The persisted
    daily target state supplies the current pacing policy when available; the
    documented RiskPolicy percentages are the ad-hoc/test fallback.
    """
    policy = RiskPolicy()
    state = _load_target_state(target_state_path)
    if explicit_max_loss_jpy is not None:
        if explicit_max_loss_jpy <= 0:
            raise ValueError("replay-backtest --max-loss must be positive")
        daily_risk_pct = _positive_float(
            explicit_daily_risk_pct,
            _state_daily_risk_pct(state),
            policy.daily_risk_pct,
        )
        target_trades = _positive_int(
            explicit_target_trades_per_day,
            state.get("target_trades_per_day"),
            policy.target_trades_per_day,
        )
        return ResolvedReplayLossCap(
            loss_cap_jpy=round(float(explicit_max_loss_jpy), 4),
            source="explicit --max-loss",
            daily_risk_pct=daily_risk_pct,
            target_trades_per_day=target_trades,
        )
    daily_risk_pct = _positive_float(
        explicit_daily_risk_pct,
        _state_daily_risk_pct(state),
        policy.daily_risk_pct,
    )
    target_trades = _positive_int(
        explicit_target_trades_per_day,
        state.get("target_trades_per_day"),
        policy.target_trades_per_day,
    )
    cap = round(start_balance_jpy * (daily_risk_pct / 100.0) / target_trades, 4)
    source = (
        f"equity-derived {daily_risk_pct:g}% daily risk / "
        f"{target_trades} target trades per day"
    )
    return ResolvedReplayLossCap(
        loss_cap_jpy=cap,
        source=source,
        daily_risk_pct=daily_risk_pct,
        target_trades_per_day=target_trades,
    )


def _load_target_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _state_daily_risk_pct(state: dict) -> float | None:
    budget = _optional_positive_float(state.get("daily_risk_budget_jpy"))
    start = _optional_positive_float(state.get("start_balance_jpy"))
    if budget is None or start is None:
        return None
    return round((budget / start) * 100.0, 6)


def _positive_float(*values: object) -> float:
    for value in values:
        parsed = _optional_positive_float(value)
        if parsed is not None:
            return parsed
    raise ValueError("replay-backtest cannot derive a positive daily_risk_pct")


def _positive_int(*values: object) -> int:
    for value in values:
        try:
            parsed = int(value) if value is not None else 0
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    raise ValueError("replay-backtest cannot derive a positive target_trades_per_day")


def _optional_positive_float(value: object) -> float | None:
    try:
        parsed = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _query_positive_by_day(conn: sqlite3.Connection, source_table: str) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        """
        SELECT
            session_date,
            COUNT(*) count,
            ROUND(SUM(CASE WHEN pl > 0 THEN pl ELSE 0 END), 4) positive_jpy
        FROM legacy_records
        WHERE source_table=? AND session_date IS NOT NULL AND pl IS NOT NULL
        GROUP BY session_date
        """,
        (source_table,),
    ).fetchall()
    return {str(row["session_date"]): dict(row) for row in rows}


def _query_live_log_by_day(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        """
        SELECT timestamp_text, pl_jpy
        FROM live_trade_events
        WHERE timestamp_text IS NOT NULL AND pl_jpy IS NOT NULL
        """
    ).fetchall()
    by_day: dict[str, dict[str, float]] = {}
    for row in rows:
        day = _date_from_timestamp(row["timestamp_text"])
        if not day:
            continue
        item = by_day.setdefault(
            day,
            {"trade_count": 0, "historical_net_jpy": 0.0, "captured_profit_jpy": 0.0, "captured_loss_jpy": 0.0},
        )
        pl = float(row["pl_jpy"] or 0.0)
        item["trade_count"] += 1
        item["historical_net_jpy"] += pl
        if pl > 0:
            item["captured_profit_jpy"] += pl
        else:
            item["captured_loss_jpy"] += pl
    for item in by_day.values():
        for key in ("historical_net_jpy", "captured_profit_jpy", "captured_loss_jpy"):
            item[key] = _round(item[key])
    return by_day


def _blockers(
    *,
    target_jpy: float,
    historical_net: float,
    risk_capped_net: float,
    evidence_coverage: float,
    missed_positive: float,
    captured_profit: float,
    over_cap_loss_count: int,
    worst_loss: float | None,
    max_loss_jpy: float,
) -> list[str]:
    blockers: list[str] = []
    if historical_net < target_jpy:
        blockers.append(f"historical net missed target by {target_jpy - historical_net:.0f} JPY")
    if evidence_coverage < target_jpy:
        blockers.append(f"evidence coverage missed target by {target_jpy - evidence_coverage:.0f} JPY")
    if over_cap_loss_count:
        blockers.append(f"{over_cap_loss_count} losses breached current {abs(max_loss_jpy):.0f} JPY cap")
    if worst_loss is not None and worst_loss < -max_loss_jpy:
        blockers.append(f"worst legacy loss {worst_loss:.0f} JPY requires risk repair")
    if missed_positive > captured_profit:
        blockers.append(f"missed-seat positive evidence exceeded captured profit by {missed_positive - captured_profit:.0f} JPY")
    if risk_capped_net > historical_net:
        blockers.append(f"loss cap would have improved day by {risk_capped_net - historical_net:.0f} JPY")
    return blockers


def _status(
    *,
    target_jpy: float,
    historical_net: float,
    risk_capped_net: float,
    evidence_coverage: float,
    blockers: tuple[str, ...],
) -> str:
    if historical_net >= target_jpy:
        return "HISTORICAL_TARGET_HIT"
    if risk_capped_net >= target_jpy:
        return "TARGET_HIT_AFTER_RISK_REPAIR"
    if evidence_coverage >= target_jpy:
        return "EVIDENCE_COVERS_TARGET"
    if any("loss cap" in blocker or "risk repair" in blocker for blocker in blockers):
        return "RISK_REPAIR_REQUIRED"
    if evidence_coverage > 0:
        return "EDGE_CAPTURE_GAP"
    return "NO_EVIDENCE_COVERAGE"


def _summary_payload(days: tuple[ReplayDay, ...], target_jpy: float) -> dict[str, float | int]:
    return {
        "days": len(days),
        "target_jpy": target_jpy,
        "historical_target_hits": sum(1 for day in days if day.historical_net_jpy >= target_jpy),
        "evidence_target_covered": sum(1 for day in days if day.evidence_coverage_jpy >= target_jpy),
        "risk_repair_days": sum(1 for day in days if day.over_cap_loss_count > 0),
        "missed_edge_days": sum(1 for day in days if day.missed_positive_jpy > day.captured_profit_jpy),
        "total_historical_net_jpy": _round(sum(day.historical_net_jpy for day in days)),
        "total_risk_capped_net_jpy": _round(sum(day.risk_capped_net_jpy for day in days)),
    }


def _date_from_timestamp(value: object) -> str | None:
    text = str(value or "")
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _number(*values: object) -> float:
    for value in values:
        if value is None or value == "":
            continue
        return float(value)
    return 0.0


def _round(value: float) -> float:
    return round(value, 4)
