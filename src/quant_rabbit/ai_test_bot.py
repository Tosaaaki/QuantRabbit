from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

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
# Calibration note (2026-06-07): walk-forward sweeps over imported legacy
# outcomes showed the old 12-day / 5-trade / trades-only default was too stale
# for regime flips and ignored the pretrade evidence that later became actual
# live receipts. The first pass moved to 5 sessions / 12 observations / 4 active
# buckets; the next firepower pass found 6 sessions / 10 observations / 6
# buckets preserved PF > 1.9 while lifting selected/oracle capture and best-day
# target coverage without adding same-day winners to training.
DEFAULT_TRAINING_DAYS = 6
DEFAULT_MIN_TRAIN_TRADES = 10
DEFAULT_MAX_ACTIVE_BUCKETS = 6
# Clear-majority evidence floor for research bucket promotion. This is not a
# market-price threshold: live-attributed sweeps showed a bare 50/50 boundary
# still promoted buckets whose capped wins hid raw tail losses, while 55% kept
# enough coverage and improved PF without over-constricting the opportunity set.
DEFAULT_MIN_TRAIN_WIN_RATE_PCT = 55.0
DEFAULT_SOURCE_TABLES = ("trades", "pretrade_outcomes", "seat_outcomes")
EXECUTION_LEDGER_SOURCE_TABLE = "execution_ledger"
DEFAULT_RUNTIME_SOURCE_TABLES = (*DEFAULT_SOURCE_TABLES, EXECUTION_LEDGER_SOURCE_TABLE)
DEFAULT_TARGET_BAND_RETURN_PCTS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
_JST = ZoneInfo("Asia/Tokyo")


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
    actual_trade_ids: tuple[str, ...] = ()

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
        min_train_win_rate_pct: float = DEFAULT_MIN_TRAIN_WIN_RATE_PCT,
        max_active_buckets: int = DEFAULT_MAX_ACTIVE_BUCKETS,
        source_tables: tuple[str, ...] = DEFAULT_SOURCE_TABLES,
        execution_ledger_db_path: Path | None = None,
        dedupe_opportunities: bool = True,
    ) -> None:
        if training_days <= 0:
            raise ValueError("ai-test-bot-backtest --training-days must be positive")
        if min_train_trades <= 0:
            raise ValueError("ai-test-bot-backtest --min-train-trades must be positive")
        if not 0.0 <= min_train_win_rate_pct <= 100.0:
            raise ValueError("ai-test-bot-backtest --min-train-win-rate-pct must be between 0 and 100")
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
        self.min_train_win_rate_pct = float(min_train_win_rate_pct)
        self.max_active_buckets = max_active_buckets
        self.source_tables = tuple(source_tables)
        self.execution_ledger_db_path = execution_ledger_db_path
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
                min_train_win_rate_pct=self.min_train_win_rate_pct,
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
            min_train_trades=self.min_train_trades,
            min_train_win_rate_pct=self.min_train_win_rate_pct,
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
        summary_payload = _summary_payload(day_results, validation_rows, target_jpy, gross_profit, gross_loss, profit_factor)
        firepower = _firepower_payload(day_results, validation_rows, target_jpy, cap.loss_cap_jpy)
        source_contributions = _source_contributions(
            trades=trades,
            day_results=day_results,
            validation_rows=validation_rows,
            max_loss_jpy=cap.loss_cap_jpy,
        )
        target_ceiling = _target_ceiling_payload(
            target_jpy=target_jpy,
            firepower=firepower,
            oracle=oracle,
        )
        target_band = _target_band_payload(
            trades=trades,
            day_results=day_results,
            validation_rows=validation_rows,
            start_balance_jpy=start_balance_jpy,
            max_loss_jpy=cap.loss_cap_jpy,
            top_n=self.max_active_buckets,
            min_train_trades=self.min_train_trades,
            min_train_win_rate_pct=self.min_train_win_rate_pct,
            target_return_pct=target_return_pct,
        )
        action_items = tuple(
            _action_items(
                blockers=blockers,
                firepower=firepower,
                oracle=oracle,
                target_ceiling=target_ceiling,
                target_band=target_band,
                source_contributions=source_contributions,
            )
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "live_permission": False,
            "db_path": str(self.db_path),
            "source_tables": list(self.source_tables),
            "execution_ledger_db": str(self.execution_ledger_db_path) if self.execution_ledger_db_path else None,
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
            "min_train_win_rate_pct": self.min_train_win_rate_pct,
            "max_active_buckets": self.max_active_buckets,
            "summary": summary_payload,
            "firepower": firepower,
            "source_contributions": source_contributions,
            "bucket_contributions": _bucket_contributions(validation_rows, cap.loss_cap_jpy),
            "oracle": oracle,
            "target_ceiling": target_ceiling,
            "target_band": target_band,
            "missed_best_days": _missed_best_days(
                trades=trades,
                day_results=day_results,
                max_loss_jpy=cap.loss_cap_jpy,
                limit=12,
            ),
            "blockers": list(blockers),
            "action_items": list(action_items),
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
        legacy_source_tables = tuple(
            source for source in self.source_tables if source != EXECUTION_LEDGER_SOURCE_TABLE
        )
        if legacy_source_tables and not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")
        if legacy_source_tables:
            placeholders = ",".join("?" for _ in legacy_source_tables)
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
                rows = conn.execute(query, legacy_source_tables).fetchall()
            for row in rows:
                raw = _raw_payload(row["raw_json"])
                source_table = str(row["source_table"])
                pair = str(row["pair"])
                direction = str(row["direction"])
                match_ids = _matched_trade_ids(raw.get("matched_trade_ids")) if source_table == "seat_outcomes" else ()
                actual_trade_ids = _actual_trade_ids(source_table=source_table, raw=raw)
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
                    actual_trade_ids=actual_trade_ids,
                )
        if EXECUTION_LEDGER_SOURCE_TABLE in self.source_tables:
            yield from _execution_ledger_trades(self.execution_ledger_db_path)

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
            f"- Execution ledger DB: `{payload.get('execution_ledger_db') or 'n/a'}`",
            f"- Source tables: `{', '.join(payload['source_tables'])}`",
            f"- Opportunity dedupe: `{payload['dedupe_opportunities']}` "
            f"(raw=`{payload['raw_rows']}`, deduped=`{payload['deduped_rows']}`)",
            f"- Start balance: `{payload['start_balance_jpy']:.0f} JPY`",
            f"- Target: `{payload['target_jpy']:.0f} JPY` (`{payload['target_return_pct']:.1f}%`)",
            f"- Per-trade loss cap: `{payload['per_trade_loss_cap_jpy']:.0f} JPY` (`{payload['loss_cap_source']}`)",
            f"- Training days: `{payload['training_days']}`",
            f"- Min training trades: `{payload['min_train_trades']}`",
            f"- Min training win rate: `{payload['min_train_win_rate_pct']:.1f}%`",
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
            f"- Train-eligible oracle target-hit days: `{payload['oracle']['train_eligible_all_positive_target_hit_days']}`",
            f"- Oracle all-positive target-hit days: `{payload['oracle']['all_positive_target_hit_days']}`",
            f"- Selected/oracle capture: `{payload['oracle']['selected_vs_top_n_capture_pct']:.1f}%`",
            f"- All-positive oracle ceiling: `{payload['target_ceiling']['best_all_positive_day_coverage_pct']:.1f}%`",
            "",
            "## Target Band",
            "",
            f"- Contract floor: `{payload['target_band']['floor_return_pct']:.1f}%`",
            f"- Stretch target: `{payload['target_band']['stretch_return_pct']:.1f}%`",
            f"- Selected-policy attainable: `{_format_optional_pct(payload['target_band']['selected_attainable_return_pct'])}`",
            f"- Train-eligible oracle attainable: `{_format_optional_pct(payload['target_band']['train_eligible_oracle_attainable_return_pct'])}`",
            f"- All-positive oracle attainable: `{_format_optional_pct(payload['target_band']['all_positive_oracle_attainable_return_pct'])}`",
            f"- Status: `{payload['target_band']['status']}`",
        ]
        for item in payload["target_band"]["bands"]:
            lines.append(
                f"- `{item['return_pct']:.1f}%` target=`{item['target_jpy']:.0f}` "
                f"selected_hits=`{item['selected_target_hit_days']}/{item['validation_days']}` "
                f"top_n_oracle_hits=`{item['top_n_target_hit_days']}` "
                f"train_eligible_hits=`{item['train_eligible_all_positive_target_hit_days']}` "
                f"all_positive_hits=`{item['all_positive_target_hit_days']}` "
                f"best_selected_coverage=`{item['best_selected_day_coverage_pct']:.1f}%` "
                f"required_trades_day=`{item['required_trades_per_day_at_observed_expectancy']}`"
            )
        lines.extend(
            [
                "",
                "## Source Contributions",
                "",
            ]
        )
        if payload["source_contributions"]:
            for item in payload["source_contributions"]:
                lines.append(
                    f"- `{item['source_table']}` validation_net=`{item['validation_universe_managed_net_jpy']:.0f}` "
                    f"selected_net=`{item['selected_managed_net_jpy']:.0f}` "
                    f"validation_rows=`{item['validation_universe_trades']}` selected_rows=`{item['selected_trades']}` "
                    f"deduped_rows=`{item['deduped_trades']}` days=`{item['validation_universe_days']}` "
                    f"win_rate=`{item['validation_universe_win_rate_pct']:.1f}%`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Blockers",
                "",
            ]
        )
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Action Items", ""])
        if payload["action_items"]:
            lines.extend(f"- {item}" for item in payload["action_items"])
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
                "- Buckets must also prove majority capped win rate in the training window before promotion.",
                "- Seat outcome buckets use only observable setup/orderability/source fields, not future `CAPTURED/FAILED/MISSED` labels.",
                "- Execution-ledger outcomes require attribution to a sent gateway entry; manual/tagless or otherwise unattributed closes are ignored.",
                "- Execution-ledger buckets use gateway lane desk/strategy as pre-entry evidence; exit reason remains post-trade evidence and is not used as a selection bucket.",
                "- Execution-ledger buckets require raw-positive training net, because hypothetical caps cannot certify old real-exit losses as fixed evidence.",
                "- Opportunity dedupe counts repeated seat receipts and cross-source same-trade-id rows as one candidate before training or validation scoring.",
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
    min_train_win_rate_pct: float,
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
            min_train_win_rate_pct=min_train_win_rate_pct,
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
    min_train_win_rate_pct: float,
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
        train_net = _round(sum(row.pl_jpy for row in rows))
        wins = sum(1 for value in capped if value > 0)
        win_rate_pct = _round((wins / len(rows)) * 100.0)
        if not _training_bucket_is_viable(
            bucket=bucket,
            capped_net_jpy=capped_net,
            raw_net_jpy=train_net,
            train_win_rate_pct=win_rate_pct,
            min_train_win_rate_pct=min_train_win_rate_pct,
        ):
            continue
        scores.append(
            BucketScore(
                bucket=bucket,
                train_trades=len(rows),
                train_net_jpy=train_net,
                train_capped_net_jpy=capped_net,
                train_mean_jpy=_round(capped_net / len(rows)),
                train_win_rate_pct=win_rate_pct,
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
    actual_trade_groups: dict[tuple[str, str, str, str], list[TestBotTrade]] = {}
    seat_matched_groups: dict[tuple[str, str, str, str], list[TestBotTrade]] = {}
    for row in rows:
        if row.source_table == "seat_outcomes" and row.match_ids:
            key = (row.source_table, row.session_date, row.pair, row.direction)
            seat_matched_groups.setdefault(key, []).append(row)
            continue
        if len(row.actual_trade_ids) == 1:
            key = (row.session_date, row.pair, row.direction, row.actual_trade_ids[0])
            actual_trade_groups.setdefault(key, []).append(row)
            continue
        current = selected.get(row.opportunity_key)
        if current is None or _is_newer(row, current):
            selected[row.opportunity_key] = row
    for group_rows in actual_trade_groups.values():
        row = _best_actual_trade_row(group_rows)
        current = selected.get(row.opportunity_key)
        if current is None or _is_newer(row, current):
            selected[row.opportunity_key] = row
    for group_rows in seat_matched_groups.values():
        for row in _dedupe_overlapping_seat_matches(group_rows):
            selected[row.opportunity_key] = row
    return tuple(sorted(selected.values(), key=lambda row: (row.session_date, row.source_table, row.pair, row.direction, row.sort_key)))


def _best_actual_trade_row(rows: list[TestBotTrade]) -> TestBotTrade:
    # Same broker trade can be imported through multiple archive tables.
    # Prefer the row with pre-entry evidence so the backtest does not count the
    # same realized P/L twice while still preserving the most actionable bucket.
    priority = {
        "pretrade_outcomes": 4,
        EXECUTION_LEDGER_SOURCE_TABLE: 3,
        "seat_outcomes": 2,
        "trades": 1,
    }
    return max(
        rows,
        key=lambda row: (
            priority.get(row.source_table, 0),
            row.sort_key,
            row.source_id,
        ),
    )


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


def _target_ceiling_payload(
    *,
    target_jpy: float,
    firepower: dict[str, float | int | None],
    oracle: dict[str, float | int | str | None],
) -> dict[str, float | int | bool | None]:
    best_selected = float(firepower.get("best_selected_day_jpy") or 0.0)
    best_top_n = float(oracle.get("best_top_n_day_jpy") or 0.0)
    best_all_positive = float(oracle.get("best_all_positive_day_jpy") or 0.0)
    best_oracle = max(best_top_n, best_all_positive)
    selected_capture = _round((best_selected / best_oracle) * 100.0) if best_oracle > 0 else 0.0
    return {
        "prediction_only_target_possible": int(oracle.get("all_positive_target_hit_days") or 0) > 0,
        "train_eligible_prediction_target_possible": int(
            oracle.get("train_eligible_all_positive_target_hit_days") or 0
        )
        > 0,
        "top_n_target_possible": int(oracle.get("top_n_target_hit_days") or 0) > 0,
        "best_selected_day_jpy": _round(best_selected),
        "best_top_n_day_jpy": _round(best_top_n),
        "best_all_positive_day_jpy": _round(best_all_positive),
        "best_train_eligible_all_positive_day_jpy": _round(
            float(oracle.get("best_train_eligible_all_positive_day_jpy") or 0.0)
        ),
        "best_all_positive_day_coverage_pct": _round((best_all_positive / target_jpy) * 100.0) if target_jpy else 0.0,
        "oracle_target_gap_jpy": _round(max(target_jpy - best_all_positive, 0.0)),
        "train_eligible_oracle_target_gap_jpy": _round(
            max(target_jpy - float(oracle.get("best_train_eligible_all_positive_day_jpy") or 0.0), 0.0)
        ),
        "selected_best_vs_oracle_best_pct": selected_capture,
    }


def _target_band_payload(
    *,
    trades: tuple[TestBotTrade, ...],
    day_results: tuple[TestBotDay, ...],
    validation_rows: tuple[TestBotTrade, ...],
    start_balance_jpy: float,
    max_loss_jpy: float,
    top_n: int,
    min_train_trades: int,
    min_train_win_rate_pct: float,
    target_return_pct: float,
) -> dict[str, object]:
    bands: list[dict[str, float | int | bool | None]] = []
    for return_pct in _target_band_percentages(target_return_pct):
        band_target_jpy = _round(start_balance_jpy * (return_pct / 100.0))
        selected_hits = sum(1 for day in day_results if day.managed_net_jpy >= band_target_jpy)
        band_firepower = _firepower_payload(day_results, validation_rows, band_target_jpy, max_loss_jpy)
        band_oracle = _oracle_summary(
            trades=trades,
            day_results=day_results,
            target_jpy=band_target_jpy,
            max_loss_jpy=max_loss_jpy,
            top_n=top_n,
            min_train_trades=min_train_trades,
            min_train_win_rate_pct=min_train_win_rate_pct,
        )
        band_ceiling = _target_ceiling_payload(
            target_jpy=band_target_jpy,
            firepower=band_firepower,
            oracle=band_oracle,
        )
        bands.append(
            {
                "return_pct": return_pct,
                "target_jpy": band_target_jpy,
                "validation_days": len(day_results),
                "selected_target_hit_days": selected_hits,
                "selected_target_hit_rate_pct": _round((selected_hits / len(day_results)) * 100.0)
                if day_results
                else 0.0,
                "best_selected_day_jpy": float(band_firepower["best_selected_day_jpy"] or 0.0),
                "best_selected_day_coverage_pct": float(band_firepower["best_selected_day_coverage_pct"] or 0.0),
                "required_trades_per_day_at_observed_expectancy": band_firepower[
                    "required_trades_per_day_at_observed_expectancy"
                ],
                "trade_frequency_multiple_required": band_firepower["trade_frequency_multiple_required"],
                "top_n_target_hit_days": int(band_oracle["top_n_target_hit_days"] or 0),
                "all_positive_target_hit_days": int(band_oracle["all_positive_target_hit_days"] or 0),
                "train_eligible_top_n_target_hit_days": int(
                    band_oracle["train_eligible_top_n_target_hit_days"] or 0
                ),
                "train_eligible_all_positive_target_hit_days": int(
                    band_oracle["train_eligible_all_positive_target_hit_days"] or 0
                ),
                "prediction_only_target_possible": bool(band_ceiling["prediction_only_target_possible"]),
                "train_eligible_prediction_target_possible": bool(
                    band_ceiling["train_eligible_prediction_target_possible"]
                ),
                "top_n_target_possible": bool(band_ceiling["top_n_target_possible"]),
                "oracle_target_gap_jpy": float(band_ceiling["oracle_target_gap_jpy"] or 0.0),
            }
        )
    selected_attainable = _max_band_pct(bands, "selected_target_hit_days")
    top_n_attainable = _max_band_pct(bands, "top_n_target_hit_days")
    train_eligible_attainable = _max_band_pct(bands, "train_eligible_all_positive_target_hit_days")
    all_positive_attainable = _max_band_pct(bands, "all_positive_target_hit_days")
    floor_pct = DEFAULT_TARGET_BAND_RETURN_PCTS[0]
    stretch_pct = DEFAULT_TARGET_BAND_RETURN_PCTS[-1]
    if selected_attainable is not None and selected_attainable >= stretch_pct:
        status = "SELECTED_POLICY_REACHES_STRETCH"
    elif selected_attainable is not None and selected_attainable >= floor_pct:
        status = "SELECTED_POLICY_REACHES_FLOOR_BELOW_STRETCH"
    elif train_eligible_attainable is not None and train_eligible_attainable >= floor_pct:
        status = "TRAIN_ELIGIBLE_ORACLE_ONLY"
    elif all_positive_attainable is not None and all_positive_attainable >= floor_pct:
        status = "HINDSIGHT_ORACLE_ONLY"
    else:
        status = "BELOW_CONTRACT_FLOOR"
    return {
        "floor_return_pct": floor_pct,
        "stretch_return_pct": stretch_pct,
        "status": status,
        "selected_attainable_return_pct": selected_attainable,
        "top_n_oracle_attainable_return_pct": top_n_attainable,
        "train_eligible_oracle_attainable_return_pct": train_eligible_attainable,
        "all_positive_oracle_attainable_return_pct": all_positive_attainable,
        "bands": bands,
    }


def _target_band_percentages(target_return_pct: float) -> tuple[float, ...]:
    values = {round(pct, 4) for pct in DEFAULT_TARGET_BAND_RETURN_PCTS}
    values.add(round(target_return_pct, 4))
    return tuple(sorted(values))


def _max_band_pct(
    bands: list[dict[str, float | int | bool | None]],
    hit_key: str,
) -> float | None:
    values = [float(item["return_pct"]) for item in bands if int(item.get(hit_key) or 0) > 0]
    return max(values) if values else None


def _action_items(
    *,
    blockers: tuple[str, ...],
    firepower: dict[str, float | int | None],
    oracle: dict[str, float | int | str | None],
    target_ceiling: dict[str, float | int | bool | None],
    target_band: dict[str, object],
    source_contributions: list[dict[str, float | int | str]],
) -> Iterable[str]:
    if not blockers:
        return
    for item in source_contributions:
        source_table = str(item["source_table"])
        validation_net = float(item["validation_universe_managed_net_jpy"])
        validation_raw_net = float(item.get("validation_universe_raw_net_jpy") or 0.0)
        selected_raw_net = float(item.get("selected_raw_net_jpy") or 0.0)
        selected_managed_net = float(item.get("selected_managed_net_jpy") or 0.0)
        selected = int(item["selected_trades"])
        validation_rows = int(item["validation_universe_trades"])
        if source_table == "seat_outcomes" and validation_rows and selected == 0 and validation_net < 0:
            yield (
                "seat_outcomes discovery universe is negative and selected no validation trades "
                f"(net {validation_net:.0f} JPY across {validation_rows} receipts); repair discovery filters before increasing live frequency"
            )
        if (
            source_table == EXECUTION_LEDGER_SOURCE_TABLE
            and validation_rows
            and validation_raw_net < 0
            and validation_net > 0
        ):
            yield (
                "execution_ledger is profitable only after applying the hypothetical per-trade loss cap "
                f"(raw {validation_raw_net:.0f} JPY vs managed {validation_net:.0f} JPY); "
                "keep close-discipline repairs in force and do not treat old raw losses as fixed evidence"
            )
        if (
            source_table == EXECUTION_LEDGER_SOURCE_TABLE
            and selected
            and selected_raw_net < 0
            and selected_managed_net > 0
        ):
            yield (
                "selected execution-ledger buckets are still raw-negative after real exits "
                f"(raw {selected_raw_net:.0f} JPY vs managed {selected_managed_net:.0f} JPY); "
                "improve exit timing before scaling the same live buckets"
            )
    if target_ceiling.get("prediction_only_target_possible") is False:
        gap = float(target_ceiling.get("oracle_target_gap_jpy") or 0.0)
        yield (
            "archive opportunity ceiling misses 10% target even with an all-positive oracle "
            f"(gap {gap:.0f} JPY); expand verified opportunity universe/receipt coverage before more prediction tuning"
        )
    elif target_ceiling.get("train_eligible_prediction_target_possible") is False:
        gap = float(target_ceiling.get("train_eligible_oracle_target_gap_jpy") or 0.0)
        yield (
            "hindsight oracle reaches the 10% target only with validation-day or insufficient-history buckets "
            f"(train-eligible gap {gap:.0f} JPY); collect or import pre-entry evidence before counting it as predictable"
        )
    selected_attainable = target_band.get("selected_attainable_return_pct")
    floor_pct = float(target_band.get("floor_return_pct") or DEFAULT_TARGET_BAND_RETURN_PCTS[0])
    stretch_pct = float(target_band.get("stretch_return_pct") or DEFAULT_TARGET_BAND_RETURN_PCTS[-1])
    if selected_attainable is None:
        yield (
            f"selected policy has not reached the {floor_pct:.0f}% contract floor in validation; "
            "prioritize coverage expansion before treating 10% as a calibratable threshold"
        )
    elif float(selected_attainable) < stretch_pct:
        next_pct = min(stretch_pct, float(selected_attainable) + 1.0)
        yield (
            f"selected policy currently reaches {float(selected_attainable):.0f}% of the 5-10% band, not 10%; "
            f"tune the next loop against {next_pct:.0f}% coverage while preserving the {floor_pct:.0f}% floor"
        )
    required = firepower.get("required_trades_per_day_at_observed_expectancy")
    avg_trades = float(firepower.get("avg_selected_trades_per_day") or 0.0)
    if isinstance(required, int) and avg_trades > 0 and required > avg_trades * 3:
        yield (
            f"observed expectancy requires about {required} selected trades/day versus {avg_trades:.1f}; "
            "increase current LIVE_READY opportunity count or per-receipt reward geometry without raising loss caps"
        )
    capture = float(oracle.get("selected_vs_top_n_capture_pct") or 0.0)
    if target_ceiling.get("top_n_target_possible") is True and capture < 70.0:
        yield (
            f"selected policy captures only {capture:.1f}% of top-N oracle; improve bucket/context timing selection"
        )
    elif capture < 50.0:
        yield (
            f"selected policy captures only {capture:.1f}% of top-N oracle, but oracle is also below target; "
            "treat selection tuning as secondary to coverage expansion"
        )


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


def _source_contributions(
    *,
    trades: tuple[TestBotTrade, ...],
    day_results: tuple[TestBotDay, ...],
    validation_rows: tuple[TestBotTrade, ...],
    max_loss_jpy: float,
) -> list[dict[str, float | int | str]]:
    validation_days = {day.session_date for day in day_results}
    selected_ids = {id(row) for row in validation_rows}
    by_source: dict[str, dict[str, object]] = {}
    for row in trades:
        item = by_source.setdefault(
            row.source_table,
            {
                "source_table": row.source_table,
                "deduped_trades": 0,
                "deduped_days": set(),
                "deduped_raw_net_jpy": 0.0,
                "deduped_managed_net_jpy": 0.0,
                "validation_universe_trades": 0,
                "validation_universe_days": set(),
                "validation_universe_wins": 0,
                "validation_universe_raw_net_jpy": 0.0,
                "validation_universe_managed_net_jpy": 0.0,
                "selected_trades": 0,
                "selected_days": set(),
                "selected_raw_net_jpy": 0.0,
                "selected_managed_net_jpy": 0.0,
            },
        )
        capped = _capped_pl(row.pl_jpy, max_loss_jpy)
        item["deduped_trades"] = int(item["deduped_trades"]) + 1
        item["deduped_raw_net_jpy"] = _round(float(item["deduped_raw_net_jpy"]) + row.pl_jpy)
        item["deduped_managed_net_jpy"] = _round(float(item["deduped_managed_net_jpy"]) + capped)
        deduped_days = item["deduped_days"]
        if isinstance(deduped_days, set):
            deduped_days.add(row.session_date)
        if row.session_date in validation_days:
            item["validation_universe_trades"] = int(item["validation_universe_trades"]) + 1
            item["validation_universe_wins"] = int(item["validation_universe_wins"]) + (1 if capped > 0 else 0)
            item["validation_universe_raw_net_jpy"] = _round(float(item["validation_universe_raw_net_jpy"]) + row.pl_jpy)
            item["validation_universe_managed_net_jpy"] = _round(
                float(item["validation_universe_managed_net_jpy"]) + capped
            )
            validation_universe_days = item["validation_universe_days"]
            if isinstance(validation_universe_days, set):
                validation_universe_days.add(row.session_date)
        if id(row) in selected_ids:
            item["selected_trades"] = int(item["selected_trades"]) + 1
            item["selected_raw_net_jpy"] = _round(float(item["selected_raw_net_jpy"]) + row.pl_jpy)
            item["selected_managed_net_jpy"] = _round(float(item["selected_managed_net_jpy"]) + capped)
            selected_days = item["selected_days"]
            if isinstance(selected_days, set):
                selected_days.add(row.session_date)
    rows: list[dict[str, float | int | str]] = []
    for item in by_source.values():
        validation_trades = int(item["validation_universe_trades"])
        validation_wins = int(item["validation_universe_wins"])
        deduped_days = item["deduped_days"]
        validation_days_set = item["validation_universe_days"]
        selected_days_set = item["selected_days"]
        rows.append(
            {
                "source_table": str(item["source_table"]),
                "deduped_trades": int(item["deduped_trades"]),
                "deduped_days": len(deduped_days) if isinstance(deduped_days, set) else 0,
                "deduped_raw_net_jpy": _round(float(item["deduped_raw_net_jpy"])),
                "deduped_managed_net_jpy": _round(float(item["deduped_managed_net_jpy"])),
                "validation_universe_trades": validation_trades,
                "validation_universe_days": len(validation_days_set) if isinstance(validation_days_set, set) else 0,
                "validation_universe_win_rate_pct": _round((validation_wins / validation_trades) * 100.0)
                if validation_trades
                else 0.0,
                "validation_universe_raw_net_jpy": _round(float(item["validation_universe_raw_net_jpy"])),
                "validation_universe_managed_net_jpy": _round(float(item["validation_universe_managed_net_jpy"])),
                "selected_trades": int(item["selected_trades"]),
                "selected_days": len(selected_days_set) if isinstance(selected_days_set, set) else 0,
                "selected_raw_net_jpy": _round(float(item["selected_raw_net_jpy"])),
                "selected_managed_net_jpy": _round(float(item["selected_managed_net_jpy"])),
            }
        )
    return sorted(rows, key=lambda item: str(item["source_table"]))


def _oracle_summary(
    *,
    trades: tuple[TestBotTrade, ...],
    day_results: tuple[TestBotDay, ...],
    target_jpy: float,
    max_loss_jpy: float,
    top_n: int,
    min_train_trades: int,
    min_train_win_rate_pct: float,
) -> dict[str, float | int | str | None]:
    selected_by_day = {day.session_date: set(day.selected_buckets) for day in day_results}
    selected_net_by_day = {day.session_date: day.managed_net_jpy for day in day_results}
    top_n_hits = 0
    all_positive_hits = 0
    train_eligible_top_n_hits = 0
    train_eligible_all_positive_hits = 0
    top_n_total = 0.0
    all_positive_total = 0.0
    train_eligible_top_n_total = 0.0
    train_eligible_all_positive_total = 0.0
    best_top_n = 0.0
    best_top_n_day: str | None = None
    best_all_positive = 0.0
    best_all_positive_day: str | None = None
    best_train_eligible_top_n = 0.0
    best_train_eligible_top_n_day: str | None = None
    best_train_eligible_all_positive = 0.0
    best_train_eligible_all_positive_day: str | None = None
    selected_total = _round(sum(selected_net_by_day.values()))
    for day in day_results:
        bucket_net = _bucket_net_for_day(trades, day.session_date, max_loss_jpy)
        train_eligible_buckets = _train_eligible_buckets(
            trades,
            day,
            max_loss_jpy=max_loss_jpy,
            min_train_trades=min_train_trades,
            min_train_win_rate_pct=min_train_win_rate_pct,
        )
        train_eligible_bucket_net = {
            bucket: value for bucket, value in bucket_net.items() if bucket in train_eligible_buckets
        }
        top_n_net = _round(sum(value for _, value in sorted(bucket_net.items(), key=lambda item: item[1], reverse=True)[:top_n]))
        all_positive_net = _round(sum(value for value in bucket_net.values() if value > 0))
        train_eligible_top_n_net = _round(
            sum(
                value
                for _, value in sorted(train_eligible_bucket_net.items(), key=lambda item: item[1], reverse=True)[:top_n]
            )
        )
        train_eligible_all_positive_net = _round(sum(value for value in train_eligible_bucket_net.values() if value > 0))
        top_n_total += top_n_net
        all_positive_total += all_positive_net
        train_eligible_top_n_total += train_eligible_top_n_net
        train_eligible_all_positive_total += train_eligible_all_positive_net
        if top_n_net >= target_jpy:
            top_n_hits += 1
        if all_positive_net >= target_jpy:
            all_positive_hits += 1
        if train_eligible_top_n_net >= target_jpy:
            train_eligible_top_n_hits += 1
        if train_eligible_all_positive_net >= target_jpy:
            train_eligible_all_positive_hits += 1
        if top_n_net > best_top_n:
            best_top_n = top_n_net
            best_top_n_day = day.session_date
        if all_positive_net > best_all_positive:
            best_all_positive = all_positive_net
            best_all_positive_day = day.session_date
        if train_eligible_top_n_net > best_train_eligible_top_n:
            best_train_eligible_top_n = train_eligible_top_n_net
            best_train_eligible_top_n_day = day.session_date
        if train_eligible_all_positive_net > best_train_eligible_all_positive:
            best_train_eligible_all_positive = train_eligible_all_positive_net
            best_train_eligible_all_positive_day = day.session_date
    positive_top_total = max(top_n_total, 0.0)
    return {
        "top_n": top_n,
        "top_n_target_hit_days": top_n_hits,
        "all_positive_target_hit_days": all_positive_hits,
        "train_eligible_top_n_target_hit_days": train_eligible_top_n_hits,
        "train_eligible_all_positive_target_hit_days": train_eligible_all_positive_hits,
        "top_n_total_net_jpy": _round(top_n_total),
        "all_positive_total_net_jpy": _round(all_positive_total),
        "train_eligible_top_n_total_net_jpy": _round(train_eligible_top_n_total),
        "train_eligible_all_positive_total_net_jpy": _round(train_eligible_all_positive_total),
        "best_top_n_day": best_top_n_day,
        "best_top_n_day_jpy": _round(best_top_n),
        "best_all_positive_day": best_all_positive_day,
        "best_all_positive_day_jpy": _round(best_all_positive),
        "best_train_eligible_top_n_day": best_train_eligible_top_n_day,
        "best_train_eligible_top_n_day_jpy": _round(best_train_eligible_top_n),
        "best_train_eligible_all_positive_day": best_train_eligible_all_positive_day,
        "best_train_eligible_all_positive_day_jpy": _round(best_train_eligible_all_positive),
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


def _train_eligible_buckets(
    trades: tuple[TestBotTrade, ...],
    day: TestBotDay,
    *,
    max_loss_jpy: float,
    min_train_trades: int,
    min_train_win_rate_pct: float,
) -> set[TestBotBucket]:
    by_bucket: dict[TestBotBucket, list[TestBotTrade]] = {}
    for row in trades:
        if not (day.training_start_date <= row.session_date <= day.training_end_date):
            continue
        by_bucket.setdefault(row.bucket, []).append(row)
    eligible: set[TestBotBucket] = set()
    for bucket, rows in by_bucket.items():
        if len(rows) < min_train_trades:
            continue
        capped_net = _round(sum(_capped_pl(row.pl_jpy, max_loss_jpy) for row in rows))
        raw_net = _round(sum(row.pl_jpy for row in rows))
        wins = sum(1 for row in rows if _capped_pl(row.pl_jpy, max_loss_jpy) > 0)
        win_rate_pct = _round((wins / len(rows)) * 100.0)
        if _training_bucket_is_viable(
            bucket=bucket,
            capped_net_jpy=capped_net,
            raw_net_jpy=raw_net,
            train_win_rate_pct=win_rate_pct,
            min_train_win_rate_pct=min_train_win_rate_pct,
        ):
            eligible.add(bucket)
    return eligible


def _training_bucket_is_viable(
    *,
    bucket: TestBotBucket,
    capped_net_jpy: float,
    raw_net_jpy: float,
    train_win_rate_pct: float,
    min_train_win_rate_pct: float,
) -> bool:
    if capped_net_jpy <= 0:
        return False
    if train_win_rate_pct < min_train_win_rate_pct:
        return False
    if bucket.source_table == EXECUTION_LEDGER_SOURCE_TABLE and raw_net_jpy <= 0:
        return False
    return True


def _execution_ledger_trades(db_path: Path | None) -> Iterable[TestBotTrade]:
    if db_path is None or not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
        ).fetchone()
        if not exists:
            return
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
            if row["name"] is not None
        }
        required_columns = {
            "event_uid",
            "ts_utc",
            "event_type",
            "lane_id",
            "order_id",
            "trade_id",
            "pair",
            "side",
            "units",
            "realized_pl_jpy",
            "raw_json",
        }
        if not required_columns <= columns:
            return
        rows = conn.execute(
            """
            WITH gateway_entries AS (
                SELECT
                    trade_id,
                    order_id,
                    lane_id
                FROM execution_events
                WHERE event_type = 'GATEWAY_ORDER_SENT'
                  AND lane_id IS NOT NULL
                  AND lane_id != ''
            ),
            entries AS (
                SELECT
                    e.trade_id,
                    MAX(g.lane_id) AS gateway_lane_id,
                    CASE
                        WHEN MAX(e.units) > 0 THEN 'LONG'
                        WHEN MIN(e.units) < 0 THEN 'SHORT'
                        ELSE NULL
                    END AS position_side
                FROM execution_events e
                LEFT JOIN gateway_entries g
                  ON (
                    g.trade_id IS NOT NULL
                    AND g.trade_id != ''
                    AND g.trade_id = e.trade_id
                  )
                  OR (
                    g.order_id IS NOT NULL
                    AND g.order_id != ''
                    AND g.order_id = e.order_id
                  )
                WHERE e.event_type = 'ORDER_FILLED'
                  AND e.trade_id IS NOT NULL
                  AND e.trade_id != ''
                  AND e.units IS NOT NULL
                GROUP BY e.trade_id
                HAVING gateway_lane_id IS NOT NULL
                   AND gateway_lane_id != ''
            )
            SELECT
                e.event_uid,
                e.ts_utc,
                e.trade_id,
                e.pair,
                e.realized_pl_jpy,
                e.raw_json,
                entries.position_side,
                entries.gateway_lane_id
            FROM execution_events e
            INNER JOIN entries ON entries.trade_id = e.trade_id
            WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
              AND e.ts_utc IS NOT NULL
              AND e.pair IS NOT NULL
              AND e.realized_pl_jpy IS NOT NULL
            ORDER BY e.ts_utc, e.event_uid
            """
        ).fetchall()
    for row in rows:
        pair = _norm(row["pair"])
        direction = _norm(row["position_side"])
        session_date = _execution_session_date(row["ts_utc"])
        if not pair or direction not in {"LONG", "SHORT"} or not session_date:
            continue
        source_id = str(row["event_uid"] or row["trade_id"] or row["ts_utc"] or "")
        raw = _raw_payload(row["raw_json"])
        raw.setdefault("trade_id", str(row["trade_id"] or ""))
        raw.setdefault("gateway_lane_id", str(row["gateway_lane_id"] or ""))
        raw.setdefault("execution_event_uid", source_id)
        raw.setdefault("created_at", str(row["ts_utc"] or ""))
        execution_style, allocation_band = _execution_ledger_bucket_fields(row["gateway_lane_id"])
        yield TestBotTrade(
            source_id=source_id,
            session_date=session_date,
            source_table=EXECUTION_LEDGER_SOURCE_TABLE,
            pair=pair,
            direction=direction,
            execution_style=execution_style,
            allocation_band=allocation_band,
            pl_jpy=float(row["realized_pl_jpy"]),
            opportunity_key=_opportunity_key(
                source_id=source_id,
                session_date=session_date,
                source_table=EXECUTION_LEDGER_SOURCE_TABLE,
                pair=pair,
                direction=direction,
                execution_style=execution_style,
                allocation_band=allocation_band,
                raw=raw,
            ),
            sort_key=str(row["ts_utc"] or source_id),
            actual_trade_ids=_matched_trade_ids(row["trade_id"]),
        )


def _execution_ledger_bucket_fields(lane_id: object) -> tuple[str, str]:
    parts = [part for part in str(lane_id or "").strip().split(":") if part]
    if len(parts) >= 4:
        desk = _lane_bucket_field(parts[0])
        strategy = _lane_bucket_field("_".join(parts[3:]))
        return desk, strategy
    return "UNSPECIFIED", "UNSPECIFIED"


def _lane_bucket_field(value: object) -> str:
    text = _norm(value).replace(":", "_")
    return text or "UNSPECIFIED"


def _execution_session_date(value: object) -> str:
    parsed = _parse_utc_timestamp(value)
    if parsed is not None:
        return parsed.astimezone(_JST).date().isoformat()
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 else ""


def _parse_utc_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    plus_pos = text.rfind("+")
    minus_pos = text.rfind("-", 19)
    tz_pos = max(plus_pos, minus_pos)
    if tz_pos > 19:
        main, suffix = text[:tz_pos], text[tz_pos:]
    else:
        main, suffix = text, ""
    if "." in main:
        head, frac = main.split(".", 1)
        main = f"{head}.{frac[:6]}"
    try:
        parsed = datetime.fromisoformat(f"{main}{suffix}")
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _norm(value: object) -> str:
    return str(value or "").strip().upper().replace("/", "_").replace(" ", "_")


def _opposite_side(value: str) -> str:
    if value == "LONG":
        return "SHORT"
    if value == "SHORT":
        return "LONG"
    return ""


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


def _actual_trade_ids(*, source_table: str, raw: dict[str, object]) -> tuple[str, ...]:
    if source_table == "seat_outcomes":
        return _matched_trade_ids(raw.get("matched_trade_ids"))
    return _matched_trade_ids(raw.get("trade_id"))


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


def _format_optional_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def _round(value: float) -> float:
    return round(value, 4)
