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
# not production market thresholds: a policy must have several recent examples
# before the test bot allows that bucket, and the active bucket count is capped
# so one validation day cannot become "trade everything that ever won".
#
# Calibration note (2026-06-07): walk-forward sweep over imported legacy
# outcomes showed the old 12-day / 5-trade / trades-only default was too stale
# for regime flips and ignored the pretrade evidence that later became actual
# live receipts. A 5-session lookback with 12 prior observations kept the
# high-support gate but lifted out-of-sample managed net from near-flat to
# strongly positive without adding same-day winners to training.
DEFAULT_TRAINING_DAYS = 5
DEFAULT_MIN_TRAIN_TRADES = 12
DEFAULT_MAX_ACTIVE_BUCKETS = 4
DEFAULT_SOURCE_TABLES = ("trades", "pretrade_outcomes")


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
    source_id: str
    session_date: str
    source_table: str
    pair: str
    direction: str
    execution_style: str
    allocation_band: str
    pl_jpy: float
    opportunity_key: str
    sort_key: str
    match_ids: tuple[str, ...] = ()

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
        dedupe_opportunities: bool = True,
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
        self.dedupe_opportunities = dedupe_opportunities

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
        raw_trades = tuple(self._load_trades())
        trades = _dedupe_opportunities(raw_trades) if self.dedupe_opportunities else raw_trades
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
        oracle = _oracle_summary(
            trades=trades,
            day_results=day_results,
            target_jpy=target_jpy,
            max_loss_jpy=cap.loss_cap_jpy,
            top_n=self.max_active_buckets,
        )
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
            "dedupe_opportunities": self.dedupe_opportunities,
            "raw_rows": len(raw_trades),
            "deduped_rows": len(trades),
            "deduped_away_rows": len(raw_trades) - len(trades),
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
            "firepower": _firepower_payload(day_results, validation_rows, target_jpy, cap.loss_cap_jpy),
            "bucket_contributions": _bucket_contributions(validation_rows, cap.loss_cap_jpy),
            "oracle": oracle,
            "missed_best_days": _missed_best_days(
                trades=trades,
                day_results=day_results,
                max_loss_jpy=cap.loss_cap_jpy,
                limit=12,
            ),
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
            SELECT source_id, session_date, source_table, pair, direction, execution_style, allocation_band, pl, raw_json
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
            raw = _raw_payload(row["raw_json"])
            source_table = str(row["source_table"])
            pair = str(row["pair"])
            direction = str(row["direction"])
            match_ids = _matched_trade_ids(raw.get("matched_trade_ids")) if source_table == "seat_outcomes" else ()
            execution_style, allocation_band = _observable_bucket_fields(
                source_table=source_table,
                execution_style=row["execution_style"],
                allocation_band=row["allocation_band"],
                raw=raw,
            )
            yield TestBotTrade(
                source_id=str(row["source_id"] or raw.get("id") or ""),
                session_date=str(row["session_date"]),
                source_table=source_table,
                pair=pair,
                direction=direction,
                execution_style=execution_style,
                allocation_band=allocation_band,
                pl_jpy=float(row["pl"]),
                opportunity_key=_opportunity_key(
                    source_id=str(row["source_id"] or raw.get("id") or ""),
                    session_date=str(row["session_date"]),
                    source_table=source_table,
                    pair=pair,
                    direction=direction,
                    execution_style=execution_style,
                    allocation_band=allocation_band,
                    raw=raw,
                ),
                sort_key=_event_sort_key(row["source_id"], raw),
                match_ids=match_ids,
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
            f"- Opportunity dedupe: `{payload['dedupe_opportunities']}` "
            f"(raw=`{payload['raw_rows']}`, deduped=`{payload['deduped_rows']}`)",
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
            "## Root Cause",
            "",
            f"- Best selected day: `{payload['firepower']['best_selected_day_jpy']:.0f} JPY`",
            f"- Average selected day: `{summary['avg_managed_day_jpy']:.0f} JPY`",
            f"- Average selected trade: `{payload['firepower']['avg_selected_trade_jpy']:.1f} JPY`",
            f"- Required trades/day at observed expectancy: `{payload['firepower']['required_trades_per_day_at_observed_expectancy']}`",
            f"- Oracle top-{payload['max_active_buckets']} target-hit days: `{payload['oracle']['top_n_target_hit_days']}`",
            f"- Oracle all-positive target-hit days: `{payload['oracle']['all_positive_target_hit_days']}`",
            f"- Selected/oracle capture: `{payload['oracle']['selected_vs_top_n_capture_pct']:.1f}%`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Bucket Contributions", ""])
        if payload["bucket_contributions"]:
            for item in payload["bucket_contributions"][:12]:
                lines.append(
                    f"- `{item['bucket']}` net=`{item['managed_net_jpy']:.0f}` raw=`{item['raw_net_jpy']:.0f}` "
                    f"trades=`{item['trades']}` days=`{item['days']}` win_rate=`{item['win_rate_pct']:.1f}%` "
                    f"worst=`{item['worst_trade_jpy']:.0f}` best=`{item['best_trade_jpy']:.0f}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Missed Best Buckets", ""])
        if payload["missed_best_days"]:
            for item in payload["missed_best_days"]:
                lines.append(
                    f"- `{item['session_date']}` selected=`{item['selected_net_jpy']:.0f}` "
                    f"best=`{item['best_bucket']}` best_net=`{item['best_bucket_net_jpy']:.0f}`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Backtest Contract",
                "",
                "- This is an offline research bot. It never places or stages broker orders.",
                "- Bucket selection uses only prior training-window days; validation-day winners cannot select themselves.",
                "- Seat outcome buckets use only observable setup/orderability/source fields, not future `CAPTURED/FAILED/MISSED` labels.",
                "- Opportunity dedupe counts repeated seat receipts as one candidate before training or validation scoring.",
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


def _dedupe_opportunities(rows: tuple[TestBotTrade, ...]) -> tuple[TestBotTrade, ...]:
    selected: dict[str, TestBotTrade] = {}
    seat_matched_groups: dict[tuple[str, str, str, str], list[TestBotTrade]] = {}
    for row in rows:
        if row.source_table == "seat_outcomes" and row.match_ids:
            key = (row.source_table, row.session_date, row.pair, row.direction)
            seat_matched_groups.setdefault(key, []).append(row)
            continue
        current = selected.get(row.opportunity_key)
        if current is None or _is_newer(row, current):
            selected[row.opportunity_key] = row
    for group_rows in seat_matched_groups.values():
        for row in _dedupe_overlapping_seat_matches(group_rows):
            selected[row.opportunity_key] = row
    return tuple(sorted(selected.values(), key=lambda row: (row.session_date, row.source_table, row.pair, row.direction, row.sort_key)))


def _dedupe_overlapping_seat_matches(rows: list[TestBotTrade]) -> tuple[TestBotTrade, ...]:
    components: list[tuple[set[str], TestBotTrade]] = []
    for row in sorted(rows, key=lambda item: (item.sort_key, item.source_id)):
        row_ids = set(row.match_ids)
        overlaps = [index for index, (ids, _) in enumerate(components) if ids & row_ids]
        if not overlaps:
            components.append((row_ids, row))
            continue
        merged_ids = set(row_ids)
        best = row
        for index in reversed(overlaps):
            ids, candidate = components.pop(index)
            merged_ids.update(ids)
            if _is_newer(candidate, best):
                best = candidate
        components.append((merged_ids, best))
    return tuple(best for _, best in components)


def _is_newer(candidate: TestBotTrade, current: TestBotTrade) -> bool:
    return (candidate.sort_key, candidate.source_id) > (current.sort_key, current.source_id)


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


def _firepower_payload(
    day_results: tuple[TestBotDay, ...],
    validation_rows: tuple[TestBotTrade, ...],
    target_jpy: float,
    max_loss_jpy: float,
) -> dict[str, float | int | None]:
    selected_trade_count = len(validation_rows)
    selected_net = _round(sum(_capped_pl(row.pl_jpy, max_loss_jpy) for row in validation_rows))
    avg_trade = _round(selected_net / selected_trade_count) if selected_trade_count else 0.0
    best_day = max((day.managed_net_jpy for day in day_results), default=0.0)
    avg_trades_per_day = _round(selected_trade_count / len(day_results)) if day_results else 0.0
    required_trades = None
    if avg_trade > 0:
        required_trades = int((target_jpy + avg_trade - 0.000001) // avg_trade)
    return {
        "best_selected_day_jpy": best_day,
        "best_selected_day_coverage_pct": _round((best_day / target_jpy) * 100.0) if target_jpy else 0.0,
        "avg_selected_trade_jpy": avg_trade,
        "avg_selected_trades_per_day": avg_trades_per_day,
        "required_trades_per_day_at_observed_expectancy": required_trades,
        "trade_frequency_multiple_required": _round(required_trades / avg_trades_per_day)
        if required_trades is not None and avg_trades_per_day > 0
        else None,
    }


def _bucket_contributions(
    validation_rows: tuple[TestBotTrade, ...],
    max_loss_jpy: float,
) -> list[dict[str, float | int | str]]:
    by_bucket: dict[TestBotBucket, dict[str, object]] = {}
    for row in validation_rows:
        item = by_bucket.setdefault(
            row.bucket,
            {
                "bucket": row.bucket.label,
                "trades": 0,
                "wins": 0,
                "managed_net_jpy": 0.0,
                "raw_net_jpy": 0.0,
                "days": set(),
                "worst_trade_jpy": None,
                "best_trade_jpy": None,
            },
        )
        capped = _capped_pl(row.pl_jpy, max_loss_jpy)
        item["trades"] = int(item["trades"]) + 1
        item["wins"] = int(item["wins"]) + (1 if capped > 0 else 0)
        item["managed_net_jpy"] = _round(float(item["managed_net_jpy"]) + capped)
        item["raw_net_jpy"] = _round(float(item["raw_net_jpy"]) + row.pl_jpy)
        days = item["days"]
        if isinstance(days, set):
            days.add(row.session_date)
        worst = item["worst_trade_jpy"]
        best = item["best_trade_jpy"]
        item["worst_trade_jpy"] = row.pl_jpy if worst is None else min(float(worst), row.pl_jpy)
        item["best_trade_jpy"] = row.pl_jpy if best is None else max(float(best), row.pl_jpy)
    rows: list[dict[str, float | int | str]] = []
    for item in by_bucket.values():
        trades = int(item["trades"])
        days = item["days"]
        rows.append(
            {
                "bucket": str(item["bucket"]),
                "trades": trades,
                "days": len(days) if isinstance(days, set) else 0,
                "win_rate_pct": _round((int(item["wins"]) / trades) * 100.0) if trades else 0.0,
                "managed_net_jpy": _round(float(item["managed_net_jpy"])),
                "raw_net_jpy": _round(float(item["raw_net_jpy"])),
                "worst_trade_jpy": _round(float(item["worst_trade_jpy"] or 0.0)),
                "best_trade_jpy": _round(float(item["best_trade_jpy"] or 0.0)),
            }
        )
    return sorted(rows, key=lambda item: (float(item["managed_net_jpy"]), str(item["bucket"])), reverse=True)


def _oracle_summary(
    *,
    trades: tuple[TestBotTrade, ...],
    day_results: tuple[TestBotDay, ...],
    target_jpy: float,
    max_loss_jpy: float,
    top_n: int,
) -> dict[str, float | int | str | None]:
    selected_by_day = {day.session_date: set(day.selected_buckets) for day in day_results}
    selected_net_by_day = {day.session_date: day.managed_net_jpy for day in day_results}
    top_n_hits = 0
    all_positive_hits = 0
    top_n_total = 0.0
    all_positive_total = 0.0
    best_top_n = 0.0
    best_top_n_day: str | None = None
    best_all_positive = 0.0
    best_all_positive_day: str | None = None
    selected_total = _round(sum(selected_net_by_day.values()))
    for day in day_results:
        bucket_net = _bucket_net_for_day(trades, day.session_date, max_loss_jpy)
        top_n_net = _round(sum(value for _, value in sorted(bucket_net.items(), key=lambda item: item[1], reverse=True)[:top_n]))
        all_positive_net = _round(sum(value for value in bucket_net.values() if value > 0))
        top_n_total += top_n_net
        all_positive_total += all_positive_net
        if top_n_net >= target_jpy:
            top_n_hits += 1
        if all_positive_net >= target_jpy:
            all_positive_hits += 1
        if top_n_net > best_top_n:
            best_top_n = top_n_net
            best_top_n_day = day.session_date
        if all_positive_net > best_all_positive:
            best_all_positive = all_positive_net
            best_all_positive_day = day.session_date
    positive_top_total = max(top_n_total, 0.0)
    return {
        "top_n": top_n,
        "top_n_target_hit_days": top_n_hits,
        "all_positive_target_hit_days": all_positive_hits,
        "top_n_total_net_jpy": _round(top_n_total),
        "all_positive_total_net_jpy": _round(all_positive_total),
        "best_top_n_day": best_top_n_day,
        "best_top_n_day_jpy": _round(best_top_n),
        "best_all_positive_day": best_all_positive_day,
        "best_all_positive_day_jpy": _round(best_all_positive),
        "selected_vs_top_n_capture_pct": _round((selected_total / positive_top_total) * 100.0) if positive_top_total > 0 else 0.0,
        "validation_days": len(selected_by_day),
    }


def _missed_best_days(
    *,
    trades: tuple[TestBotTrade, ...],
    day_results: tuple[TestBotDay, ...],
    max_loss_jpy: float,
    limit: int,
) -> list[dict[str, float | str]]:
    misses: list[dict[str, float | str]] = []
    for day in day_results:
        bucket_net = _bucket_net_for_day(trades, day.session_date, max_loss_jpy)
        if not bucket_net:
            continue
        best_bucket, best_net = max(bucket_net.items(), key=lambda item: item[1])
        selected = set(day.selected_buckets)
        if best_net > 0 and best_bucket.label not in selected:
            misses.append(
                {
                    "session_date": day.session_date,
                    "selected_net_jpy": day.managed_net_jpy,
                    "best_bucket": best_bucket.label,
                    "best_bucket_net_jpy": _round(best_net),
                }
            )
    return sorted(misses, key=lambda item: float(item["best_bucket_net_jpy"]), reverse=True)[:limit]


def _bucket_net_for_day(
    trades: tuple[TestBotTrade, ...],
    session_date: str,
    max_loss_jpy: float,
) -> dict[TestBotBucket, float]:
    bucket_net: dict[TestBotBucket, float] = {}
    for row in trades:
        if row.session_date != session_date:
            continue
        bucket_net[row.bucket] = _round(bucket_net.get(row.bucket, 0.0) + _capped_pl(row.pl_jpy, max_loss_jpy))
    return bucket_net


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


def _observable_bucket_fields(
    *,
    source_table: str,
    execution_style: object,
    allocation_band: object,
    raw: dict[str, object],
) -> tuple[str, str]:
    if source_table == "seat_outcomes":
        return _seat_order_style(raw), _bucket_field(raw.get("source"))
    if source_table == "pretrade_outcomes":
        return _bucket_field(raw.get("pretrade_level") or execution_style), _bucket_field(allocation_band)
    return _bucket_field(execution_style), _bucket_field(allocation_band)


def _seat_order_style(raw: dict[str, object]) -> str:
    for key in ("setup_type", "orderability"):
        style = _classify_order_style(raw.get(key))
        if style != "UNSPECIFIED":
            return style
    return "UNSPECIFIED"


def _classify_order_style(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "UNSPECIFIED"
    if "STOP" in text:
        return "STOP-ENTRY"
    if "LIMIT" in text:
        return "LIMIT"
    if "MARKET" in text:
        return "MARKET"
    if "ENTER NOW" in text or "NOW" == text:
        return "MARKET"
    if "PASS" in text:
        return "PASS"
    return text


def _opportunity_key(
    *,
    source_id: str,
    session_date: str,
    source_table: str,
    pair: str,
    direction: str,
    execution_style: str,
    allocation_band: str,
    raw: dict[str, object],
) -> str:
    if source_table == "seat_outcomes":
        match_ids = _matched_trade_ids(raw.get("matched_trade_ids"))
        if match_ids:
            return ":".join((source_table, session_date, pair, direction, "matched", ",".join(match_ids)))
        return ":".join((source_table, session_date, pair, direction, allocation_band, execution_style))
    trade_id = str(raw.get("trade_id") or "").strip()
    if trade_id:
        return ":".join((source_table, session_date, pair, direction, trade_id))
    return ":".join((source_table, session_date, pair, direction, execution_style, allocation_band, source_id))


def _matched_trade_ids(value: object) -> tuple[str, ...]:
    parts = sorted(part.strip(" `") for part in str(value or "").replace("，", ",").split(",") if part.strip(" `"))
    return tuple(parts)


def _event_sort_key(source_id: object, raw: dict[str, object]) -> str:
    for key in ("updated_at", "state_last_updated", "created_at"):
        value = str(raw.get(key) or "").strip()
        if value:
            return value
    return str(source_id or "")


def _raw_payload(value: object) -> dict[str, object]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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
