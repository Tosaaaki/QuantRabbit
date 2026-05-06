from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from quant_rabbit.paths import (
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_HISTORY_DB,
)
from quant_rabbit.replay import _resolve_replay_loss_cap


# Research-window defaults. These are data sufficiency / anti-overfit controls,
# not production market thresholds: a policy must have at least several recent
# examples before the test bot allows that bucket, and the active bucket count is
# capped so one validation day cannot become "trade everything that ever won".
DEFAULT_TRAINING_DAYS = 12
DEFAULT_MIN_TRAIN_TRADES = 5
DEFAULT_MAX_ACTIVE_BUCKETS = 4
DEFAULT_SOURCE_TABLES = ("trades",)


@dataclass(frozen=True)
class TestBotBucket:
    source_table: str
    pair: str
    direction: str
    execution_style: str
    allocation_band: str

    @property
    def label(self) -> str:
        return ":".join((self.source_table, self.pair, self.direction, self.execution_style, self.allocation_band))


@dataclass(frozen=True)
class TestBotTrade:
    session_date: str
    source_table: str
    pair: str
    direction: str
    execution_style: str
    allocation_band: str
    pl_jpy: float

    @property
    def bucket(self) -> TestBotBucket:
        return TestBotBucket(
            source_table=self.source_table,
            pair=self.pair,
            direction=self.direction,
            execution_style=self.execution_style,
            allocation_band=self.allocation_band,
        )


@dataclass(frozen=True)
class BucketScore:
    bucket: TestBotBucket
    train_trades: int
    train_net_jpy: float
    train_capped_net_jpy: float
    train_mean_jpy: float
    train_win_rate_pct: float
    train_worst_loss_jpy: float | None


@dataclass(frozen=True)
class TestBotDay:
    session_date: str
    training_start_date: str
    training_end_date: str
    selected_buckets: tuple[str, ...]
    selected_trades: int
    raw_net_jpy: float
    managed_net_jpy: float
    target_hit: bool
    worst_trade_jpy: float | None
    bucket_scores: tuple[BucketScore, ...]


@dataclass(frozen=True)
class AITestBotBacktestSummary:
    output_path: Path
    report_path: Path
    status: str
    validation_days: int
    traded_days: int
    target_hit_days: int
    total_managed_net_jpy: float
    profit_factor: float | None
    blockers: int


class AITestBotBacktester:
    """Walk-forward research bot for AI-managed parameter policy.

    The bot never writes broker state and never calls a model API. It simulates
    an AI trader's parameter discipline by selecting allowed historical buckets
    from a trailing training window, then applying only those buckets to the next
    validation day. Losses are capped by the same equity-derived per-trade cap
    used by replay diagnostics; positive P/L is not inflated.
    """

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_HISTORY_DB,
        output_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        report_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        max_loss_jpy: float | None = None,
        daily_risk_pct: float | None = None,
        target_trades_per_day: int | None = None,
        training_days: int = DEFAULT_TRAINING_DAYS,
        min_train_trades: int = DEFAULT_MIN_TRAIN_TRADES,
        max_active_buckets: int = DEFAULT_MAX_ACTIVE_BUCKETS,
        source_tables: tuple[str, ...] = DEFAULT_SOURCE_TABLES,
    ) -> None:
        if training_days <= 0:
            raise ValueError("ai-test-bot-backtest --training-days must be positive")
        if min_train_trades <= 0:
            raise ValueError("ai-test-bot-backtest --min-train-trades must be positive")
        if max_active_buckets <= 0:
            raise ValueError("ai-test-bot-backtest --max-active-buckets must be positive")
        if not source_tables:
            raise ValueError("ai-test-bot-backtest requires at least one source table")
        self.db_path = db_path
        self.output_path = output_path
        self.report_path = report_path
        self.target_state_path = target_state_path
        self.max_loss_jpy = max_loss_jpy
        self.daily_risk_pct = daily_risk_pct
        self.target_trades_per_day = target_trades_per_day
        self.training_days = training_days
        self.min_train_trades = min_train_trades
        self.max_active_buckets = max_active_buckets
        self.source_tables = tuple(source_tables)

    def run(
        self,
        *,
        start_balance_jpy: float,
        target_return_pct: float = 10.0,
        max_validation_days: int | None = None,
    ) -> AITestBotBacktestSummary:
        if start_balance_jpy <= 0:
            raise ValueError("ai-test-bot-backtest --start-balance must be positive")
        if target_return_pct <= 0:
            raise ValueError("ai-test-bot-backtest --target-return-pct must be positive")
        if max_validation_days is not None and max_validation_days <= 0:
            raise ValueError("ai-test-bot-backtest --max-validation-days must be positive")

        cap = _resolve_replay_loss_cap(
            start_balance_jpy=start_balance_jpy,
            explicit_max_loss_jpy=self.max_loss_jpy,
            explicit_daily_risk_pct=self.daily_risk_pct,
            explicit_target_trades_per_day=self.target_trades_per_day,
            target_state_path=self.target_state_path,
        )
        daily_risk_budget_jpy = round(start_balance_jpy * (cap.daily_risk_pct / 100.0), 4)
        target_jpy = round(start_balance_jpy * (target_return_pct / 100.0), 4)
        trades = tuple(self._load_trades())
        days = sorted({trade.session_date for trade in trades})
        day_results = tuple(
            _walk_forward_days(
                trades=trades,
                days=days,
                target_jpy=target_jpy,
                max_loss_jpy=cap.loss_cap_jpy,
                training_days=self.training_days,
                min_train_trades=self.min_train_trades,
                max_active_buckets=self.max_active_buckets,
                max_validation_days=max_validation_days,
            )
        )
        validation_rows = tuple(row for day in day_results for row in _selected_trades_for_day(trades, day))
        blockers = tuple(
            _certification_blockers(
                day_results=day_results,
                validation_rows=validation_rows,
                target_jpy=target_jpy,
                daily_risk_budget_jpy=daily_risk_budget_jpy,
            )
        )
        gross_profit = _round(sum(_capped_pl(row.pl_jpy, cap.loss_cap_jpy) for row in validation_rows if _capped_pl(row.pl_jpy, cap.loss_cap_jpy) > 0))
        gross_loss = abs(
            _round(sum(_capped_pl(row.pl_jpy, cap.loss_cap_jpy) for row in validation_rows if _capped_pl(row.pl_jpy, cap.loss_cap_jpy) < 0))
        )
        profit_factor = _profit_factor(gross_profit, gross_loss)
        status = _status(blockers, day_results, profit_factor)
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "live_permission": False,
            "db_path": str(self.db_path),
            "source_tables": list(self.source_tables),
            "start_balance_jpy": start_balance_jpy,
            "target_return_pct": target_return_pct,
            "target_jpy": target_jpy,
            "per_trade_loss_cap_jpy": cap.loss_cap_jpy,
            "loss_cap_source": cap.source,
            "daily_risk_budget_jpy": daily_risk_budget_jpy,
            "training_days": self.training_days,
            "min_train_trades": self.min_train_trades,
            "max_active_buckets": self.max_active_buckets,
            "summary": _summary_payload(day_results, validation_rows, target_jpy, gross_profit, gross_loss, profit_factor),
            "blockers": list(blockers),
            "days": [asdict(day) for day in day_results],
        }
        self._write_output(payload)
        self._write_report(payload)
        summary = payload["summary"]
        return AITestBotBacktestSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            validation_days=int(summary["validation_days"]),
            traded_days=int(summary["traded_days"]),
            target_hit_days=int(summary["target_hit_days"]),
            total_managed_net_jpy=float(summary["total_managed_net_jpy"]),
            profit_factor=profit_factor,
            blockers=len(blockers),
        )

    def _load_trades(self) -> Iterable[TestBotTrade]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")
        placeholders = ",".join("?" for _ in self.source_tables)
        query = f"""
            SELECT session_date, source_table, pair, direction, execution_style, allocation_band, pl
            FROM legacy_records
            WHERE source_table IN ({placeholders})
              AND session_date IS NOT NULL
              AND pair IS NOT NULL
              AND direction IS NOT NULL
              AND pl IS NOT NULL
            ORDER BY session_date, source_table, pair, direction
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, self.source_tables).fetchall()
        for row in rows:
            yield TestBotTrade(
                session_date=str(row["session_date"]),
                source_table=str(row["source_table"]),
                pair=str(row["pair"]),
                direction=str(row["direction"]),
                execution_style=_bucket_field(row["execution_style"]),
                allocation_band=_bucket_field(row["allocation_band"]),
                pl_jpy=float(row["pl"]),
            )

    def _write_output(self, payload: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        summary = payload["summary"]
        lines = [
            "# AI Test Bot Backtest Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Live permission: `{payload['live_permission']}`",
            f"- History DB: `{payload['db_path']}`",
            f"- Source tables: `{', '.join(payload['source_tables'])}`",
            f"- Start balance: `{payload['start_balance_jpy']:.0f} JPY`",
            f"- Target: `{payload['target_jpy']:.0f} JPY` (`{payload['target_return_pct']:.1f}%`)",
            f"- Per-trade loss cap: `{payload['per_trade_loss_cap_jpy']:.0f} JPY` (`{payload['loss_cap_source']}`)",
            f"- Training days: `{payload['training_days']}`",
            f"- Min training trades: `{payload['min_train_trades']}`",
            f"- Max active buckets: `{payload['max_active_buckets']}`",
            f"- Validation days: `{summary['validation_days']}`",
            f"- Traded days: `{summary['traded_days']}`",
            f"- Target-hit days: `{summary['target_hit_days']}`",
            f"- Total managed net: `{summary['total_managed_net_jpy']:.0f} JPY`",
            f"- Profit factor: `{summary['profit_factor'] if summary['profit_factor'] is not None else 'n/a'}`",
            f"- Max drawdown: `{summary['max_drawdown_jpy']:.0f} JPY`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Backtest Contract",
                "",
                "- This is an offline research bot. It never places or stages broker orders.",
                "- Bucket selection uses only prior training-window days; validation-day winners cannot select themselves.",
                "- Losses are capped by the equity-derived per-trade cap; wins are not enlarged.",
                "- `live_permission=false` means this receipt can support research, not live execution.",
                "",
                "## Validation Days",
                "",
            ]
        )
        for day in payload["days"]:
            buckets = ", ".join(day["selected_buckets"]) or "none"
            lines.append(
                f"- `{day['session_date']}` net=`{day['managed_net_jpy']:.0f}` raw=`{day['raw_net_jpy']:.0f}` "
                f"trades=`{day['selected_trades']}` target_hit=`{day['target_hit']}` buckets=`{buckets}`"
            )
        self.report_path.write_text("\n".join(lines) + "\n")


def _walk_forward_days(
    *,
    trades: tuple[TestBotTrade, ...],
    days: list[str],
    target_jpy: float,
    max_loss_jpy: float,
    training_days: int,
    min_train_trades: int,
    max_active_buckets: int,
    max_validation_days: int | None,
) -> Iterable[TestBotDay]:
    emitted = 0
    for index, day in enumerate(days):
        if index < training_days:
            continue
        train_days = set(days[index - training_days : index])
        train_rows = tuple(row for row in trades if row.session_date in train_days)
        scores = _select_buckets(
            train_rows,
            max_loss_jpy=max_loss_jpy,
            min_train_trades=min_train_trades,
            max_active_buckets=max_active_buckets,
        )
        selected_keys = {score.bucket for score in scores}
        validation_rows = tuple(row for row in trades if row.session_date == day and row.bucket in selected_keys)
        raw_net = _round(sum(row.pl_jpy for row in validation_rows))
        managed_net = _round(sum(_capped_pl(row.pl_jpy, max_loss_jpy) for row in validation_rows))
        yield TestBotDay(
            session_date=day,
            training_start_date=days[index - training_days],
            training_end_date=days[index - 1],
            selected_buckets=tuple(score.bucket.label for score in scores),
            selected_trades=len(validation_rows),
            raw_net_jpy=raw_net,
            managed_net_jpy=managed_net,
            target_hit=managed_net >= target_jpy,
            worst_trade_jpy=min((row.pl_jpy for row in validation_rows), default=None),
            bucket_scores=scores,
        )
        emitted += 1
        if max_validation_days is not None and emitted >= max_validation_days:
            return


def _select_buckets(
    train_rows: tuple[TestBotTrade, ...],
    *,
    max_loss_jpy: float,
    min_train_trades: int,
    max_active_buckets: int,
) -> tuple[BucketScore, ...]:
    by_bucket: dict[TestBotBucket, list[TestBotTrade]] = {}
    for row in train_rows:
        by_bucket.setdefault(row.bucket, []).append(row)
    scores: list[BucketScore] = []
    for bucket, rows in by_bucket.items():
        if len(rows) < min_train_trades:
            continue
        capped = [_capped_pl(row.pl_jpy, max_loss_jpy) for row in rows]
        capped_net = _round(sum(capped))
        if capped_net <= 0:
            continue
        train_net = _round(sum(row.pl_jpy for row in rows))
        wins = sum(1 for value in capped if value > 0)
        scores.append(
            BucketScore(
                bucket=bucket,
                train_trades=len(rows),
                train_net_jpy=train_net,
                train_capped_net_jpy=capped_net,
                train_mean_jpy=_round(capped_net / len(rows)),
                train_win_rate_pct=_round((wins / len(rows)) * 100.0),
                train_worst_loss_jpy=min((row.pl_jpy for row in rows), default=None),
            )
        )
    return tuple(
        sorted(
            scores,
            key=lambda item: (item.train_mean_jpy, item.train_capped_net_jpy, item.train_trades, item.bucket.label),
            reverse=True,
        )[:max_active_buckets]
    )


def _selected_trades_for_day(trades: tuple[TestBotTrade, ...], day: TestBotDay) -> tuple[TestBotTrade, ...]:
    selected = set(day.selected_buckets)
    return tuple(row for row in trades if row.session_date == day.session_date and row.bucket.label in selected)


def _certification_blockers(
    *,
    day_results: tuple[TestBotDay, ...],
    validation_rows: tuple[TestBotTrade, ...],
    target_jpy: float,
    daily_risk_budget_jpy: float,
) -> Iterable[str]:
    if not day_results:
        yield "no validation days were available after the training window"
        return
    traded_days = sum(1 for day in day_results if day.selected_trades > 0)
    if traded_days == 0:
        yield "AI policy selected no validation trades"
    total_net = _round(sum(day.managed_net_jpy for day in day_results))
    if total_net <= 0:
        yield f"out-of-sample managed net is not positive: {total_net:.0f} JPY"
    gross_profit = _round(sum(row.pl_jpy for row in validation_rows if row.pl_jpy > 0))
    gross_loss = abs(_round(sum(row.pl_jpy for row in validation_rows if row.pl_jpy < 0)))
    if gross_profit <= gross_loss:
        yield "raw selected validation profit does not exceed raw selected losses"
    missed_targets = [day for day in day_results if not day.target_hit]
    if missed_targets:
        yield f"10% target was missed on {len(missed_targets)}/{len(day_results)} validation days"
    worst_day = min((day.managed_net_jpy for day in day_results), default=0.0)
    if worst_day < -daily_risk_budget_jpy:
        yield f"worst validation day {worst_day:.0f} JPY breached daily risk budget {daily_risk_budget_jpy:.0f} JPY"
    if not any(day.selected_buckets for day in day_results):
        yield "no parameter bucket survived the trailing training evidence gate"
    if target_jpy <= 0:
        yield "target_jpy must be positive"


def _summary_payload(
    day_results: tuple[TestBotDay, ...],
    validation_rows: tuple[TestBotTrade, ...],
    target_jpy: float,
    gross_profit: float,
    gross_loss: float,
    profit_factor: float | None,
) -> dict[str, float | int | None]:
    net = _round(sum(day.managed_net_jpy for day in day_results))
    target_hits = sum(1 for day in day_results if day.target_hit)
    return {
        "validation_days": len(day_results),
        "traded_days": sum(1 for day in day_results if day.selected_trades > 0),
        "selected_trades": len(validation_rows),
        "target_hit_days": target_hits,
        "target_hit_rate_pct": _round((target_hits / len(day_results)) * 100.0) if day_results else 0.0,
        "total_raw_net_jpy": _round(sum(day.raw_net_jpy for day in day_results)),
        "total_managed_net_jpy": net,
        "avg_managed_day_jpy": _round(net / len(day_results)) if day_results else 0.0,
        "gross_profit_jpy": gross_profit,
        "gross_loss_jpy": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_jpy": _max_drawdown(tuple(day.managed_net_jpy for day in day_results)),
        "target_jpy": target_jpy,
    }


def _status(
    blockers: tuple[str, ...],
    day_results: tuple[TestBotDay, ...],
    profit_factor: float | None,
) -> str:
    if not blockers:
        return "TARGET_COVERAGE_CERTIFIED"
    total_net = _round(sum(day.managed_net_jpy for day in day_results))
    if total_net > 0 and (profit_factor is None or profit_factor > 1.0):
        return "RESEARCH_PROFITABLE_NOT_CERTIFIED"
    return "BLOCKED"


def _bucket_field(value: object) -> str:
    text = str(value or "").strip().upper()
    return text or "UNSPECIFIED"


def _capped_pl(pl_jpy: float, max_loss_jpy: float) -> float:
    if pl_jpy < -max_loss_jpy:
        return -max_loss_jpy
    return _round(pl_jpy)


def _profit_factor(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss == 0:
        return None
    return _round(gross_profit / gross_loss)


def _max_drawdown(values: tuple[float, ...]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return _round(drawdown)


def _round(value: float) -> float:
    return round(value, 4)
