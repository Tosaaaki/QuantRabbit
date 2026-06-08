from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
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
# Cross-pair context overlay controls. These are replay data-sufficiency guards,
# not market-price thresholds: a generalized theme can light up several pairs,
# so it needs broader support than a pair bucket and is capped to one active
# overlay to avoid turning "risk-on" into "trade everything". 2026-06-08
# replay sweeps showed the 60% theme floor admitted a few broad-theme losers;
# 65% kept the same best-day coverage while improving PF and required 5% pace.
CONTEXT_THEME_SOURCE_TABLE = "trades_theme"
CONTEXT_THEME_MIN_TRAIN_TRADES = 20
CONTEXT_THEME_MIN_TRAIN_WIN_RATE_PCT = 65.0
CONTEXT_THEME_MAX_ACTIVE_BUCKETS = 1
_FX_CURRENCIES = frozenset({"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"})
_COMMODITY_LINKED_CURRENCIES = frozenset({"AUD", "CAD", "NZD"})
_EUROPEAN_RISK_CURRENCIES = frozenset({"EUR", "GBP", "CHF"})
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
    extra_buckets: tuple[TestBotBucket, ...] = ()
    context_features: tuple[str, ...] = ()

    @property
    def bucket(self) -> TestBotBucket:
        return TestBotBucket(
            source_table=self.source_table,
            pair=self.pair,
            direction=self.direction,
            execution_style=self.execution_style,
            allocation_band=self.allocation_band,
        )

    @property
    def buckets(self) -> tuple[TestBotBucket, ...]:
        return (self.bucket, *self.extra_buckets)


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
        promotable_source_tables = _promotable_source_tables(self.source_tables)
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
                promotable_source_tables=promotable_source_tables,
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
            promotable_source_tables=promotable_source_tables,
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
        risk_engine_loss_cap = _risk_engine_loss_cap_payload(
            validation_rows=validation_rows,
            max_loss_jpy=cap.loss_cap_jpy,
        )
        target_sizing = _target_sizing_ablation_payload(
            day_results=day_results,
            start_balance_jpy=start_balance_jpy,
            max_loss_jpy=cap.loss_cap_jpy,
            target_return_pct=target_return_pct,
        )
        close_gate_ab = _close_gate_ablation_payload(
            db_path=self.execution_ledger_db_path,
            max_loss_jpy=cap.loss_cap_jpy,
        )
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
            promotable_source_tables=promotable_source_tables,
        )
        action_items = tuple(
            _action_items(
                blockers=blockers,
                firepower=firepower,
                oracle=oracle,
                target_ceiling=target_ceiling,
                target_band=target_band,
                source_contributions=source_contributions,
                risk_engine_loss_cap=risk_engine_loss_cap,
                target_sizing=target_sizing,
                close_gate_ab=close_gate_ab,
            )
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "live_permission": False,
            "db_path": str(self.db_path),
            "source_tables": list(self.source_tables),
            "promotable_source_tables": list(promotable_source_tables),
            "execution_ledger_selection": _execution_ledger_selection_mode(
                self.source_tables,
                promotable_source_tables,
            ),
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
            "context_theme_policy": {
                "source_table": CONTEXT_THEME_SOURCE_TABLE,
                "max_active_buckets": CONTEXT_THEME_MAX_ACTIVE_BUCKETS,
                "min_train_trades": CONTEXT_THEME_MIN_TRAIN_TRADES,
                "min_train_win_rate_pct": CONTEXT_THEME_MIN_TRAIN_WIN_RATE_PCT,
            },
            "context_feature_coverage": _context_feature_coverage(trades),
            "summary": summary_payload,
            "firepower": firepower,
            "mechanism_ablation": {
                "risk_engine_loss_cap": risk_engine_loss_cap,
                "target_sizing": target_sizing,
                "close_gate_ab": close_gate_ab,
            },
            "source_contributions": source_contributions,
            "bucket_contributions": _bucket_contributions(validation_rows, cap.loss_cap_jpy),
            "evidence_bucket_contributions": _evidence_bucket_contributions(validation_rows, cap.loss_cap_jpy),
            "oracle": oracle,
            "target_ceiling": target_ceiling,
            "target_band": target_band,
            "missed_best_days": _missed_best_days(
                trades=trades,
                day_results=day_results,
                max_loss_jpy=cap.loss_cap_jpy,
                promotable_source_tables=promotable_source_tables,
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
                    extra_buckets=_extra_observable_buckets(
                        source_table=source_table,
                        pair=pair,
                        direction=direction,
                    ),
                    context_features=_observable_context_features(raw),
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
            f"- Promotable source tables: `{', '.join(payload['promotable_source_tables'])}`",
            f"- Execution ledger selection: `{payload['execution_ledger_selection']}`",
            f"- Opportunity dedupe: `{payload['dedupe_opportunities']}` "
            f"(raw=`{payload['raw_rows']}`, deduped=`{payload['deduped_rows']}`)",
            f"- Start balance: `{payload['start_balance_jpy']:.0f} JPY`",
            f"- Target: `{payload['target_jpy']:.0f} JPY` (`{payload['target_return_pct']:.1f}%`)",
            f"- Per-trade loss cap: `{payload['per_trade_loss_cap_jpy']:.0f} JPY` (`{payload['loss_cap_source']}`)",
            f"- Training days: `{payload['training_days']}`",
            f"- Min training trades: `{payload['min_train_trades']}`",
            f"- Min training win rate: `{payload['min_train_win_rate_pct']:.1f}%`",
            f"- Max active buckets: `{payload['max_active_buckets']}`",
            f"- Context theme overlay: max=`{payload['context_theme_policy']['max_active_buckets']}` "
            f"min_trades=`{payload['context_theme_policy']['min_train_trades']}` "
            f"min_win_rate=`{payload['context_theme_policy']['min_train_win_rate_pct']:.1f}%`",
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
            f"- Selected-policy best return: `{payload['target_band']['selected_best_return_pct']:.2f}%`",
            f"- Train-eligible oracle attainable: `{_format_optional_pct(payload['target_band']['train_eligible_oracle_attainable_return_pct'])}`",
            f"- Train-eligible oracle best return: `{payload['target_band']['train_eligible_oracle_best_return_pct']:.2f}%`",
            f"- All-positive oracle attainable: `{_format_optional_pct(payload['target_band']['all_positive_oracle_attainable_return_pct'])}`",
            f"- All-positive oracle best return: `{payload['target_band']['all_positive_oracle_best_return_pct']:.2f}%`",
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
        mechanism = payload.get("mechanism_ablation") if isinstance(payload.get("mechanism_ablation"), dict) else {}
        risk_cap = mechanism.get("risk_engine_loss_cap") if isinstance(mechanism.get("risk_engine_loss_cap"), dict) else {}
        target_sizing = mechanism.get("target_sizing") if isinstance(mechanism.get("target_sizing"), dict) else {}
        close_gate = mechanism.get("close_gate_ab") if isinstance(mechanism.get("close_gate_ab"), dict) else {}
        lines.extend(
            [
                "",
                "## Mechanism Ablations",
                "",
                "### RiskEngine Loss Cap",
                "",
                f"- Scope: `{risk_cap.get('scope', 'n/a')}`",
                f"- Raw selected net: `{float(risk_cap.get('raw_selected_net_jpy') or 0.0):.0f} JPY`",
                f"- Managed selected net: `{float(risk_cap.get('managed_selected_net_jpy') or 0.0):.0f} JPY`",
                f"- Managed-minus-raw effect: `{float(risk_cap.get('managed_net_minus_raw_net_jpy') or 0.0):.0f} JPY`",
                f"- Raw PF: `{risk_cap.get('raw_profit_factor') if risk_cap.get('raw_profit_factor') is not None else 'n/a'}`",
                f"- Managed PF: `{risk_cap.get('managed_profit_factor') if risk_cap.get('managed_profit_factor') is not None else 'n/a'}`",
                f"- Interpretation: `{risk_cap.get('interpretation', 'n/a')}`",
                "",
                "### Target-Aware Sizing Diagnostics",
                "",
                f"- Status: `{target_sizing.get('status', 'n/a')}`",
                f"- Diagnostic only: `{target_sizing.get('diagnostic_only', True)}`",
                f"- Best selected day: `{float(target_sizing.get('best_selected_day_jpy') or 0.0):.0f} JPY` "
                f"(`{target_sizing.get('best_selected_day') or 'n/a'}`)",
            ]
        )
        sizing_bands = target_sizing.get("bands") if isinstance(target_sizing.get("bands"), list) else []
        for item in sizing_bands:
            if not isinstance(item, dict):
                continue
            multiplier = item.get("required_size_multiplier")
            multiplier_text = f"{float(multiplier):.4f}" if multiplier is not None else "n/a"
            loss_cap = item.get("scaled_loss_cap_jpy")
            loss_cap_text = f"{float(loss_cap):.0f}" if loss_cap is not None else "n/a"
            scaled_dd = item.get("scaled_max_drawdown_jpy")
            scaled_dd_text = f"{float(scaled_dd):.0f}" if scaled_dd is not None else "n/a"
            lines.append(
                f"- `{float(item.get('return_pct') or 0.0):.1f}%` target=`{float(item.get('target_jpy') or 0.0):.0f}` "
                f"required_size_multiplier=`{multiplier_text}` scaled_loss_cap=`{loss_cap_text}` "
                f"scaled_target_hits=`{int(item.get('scaled_target_hit_days') or 0)}` "
                f"scaled_max_dd=`{scaled_dd_text}` "
                f"status=`{item.get('status', 'n/a')}`"
            )
        lines.extend(
            [
                "",
                "### CLOSE Gate A/B Diagnostics",
                "",
                f"- Status: `{close_gate.get('status', 'n/a')}`",
                f"- Reason: `{close_gate.get('reason', '')}`",
                f"- Close events: `{int(close_gate.get('close_events') or 0)}`",
                f"- Close net: `{float(close_gate.get('close_net_jpy') or 0.0):.0f} JPY`",
                f"- Bot-attributed close events: `{int(close_gate.get('bot_attributed_close_events') or 0)}`",
                f"- Gateway close sent events: `{int(close_gate.get('gateway_close_sent_events') or 0)}`",
                f"- Broker accepted TRADE_CLOSE events: `{int(close_gate.get('broker_trade_close_accept_events') or 0)}`",
                f"- Loss-side market closes: `{int(close_gate.get('loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Gateway loss-side market closes: `{int(close_gate.get('gateway_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('gateway_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Gateway GPT_CLOSE loss-side market closes: `{int(close_gate.get('gateway_gpt_close_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('gateway_gpt_close_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Gateway REVIEW_EXIT loss-side market closes: `{int(close_gate.get('gateway_review_exit_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('gateway_review_exit_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Broker accepted loss-side market closes: `{int(close_gate.get('broker_trade_close_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('broker_trade_close_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Broker accepted without gateway close receipt: `{int(close_gate.get('broker_accepted_without_gateway_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('broker_accepted_without_gateway_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- No close-order provenance loss-side market closes: `{int(close_gate.get('unattributed_loss_side_market_close_count') or 0)}` "
                f"net=`{float(close_gate.get('unattributed_loss_side_market_close_net_jpy') or 0.0):.0f} JPY`",
                f"- Take-profit closes: `{int(close_gate.get('take_profit_close_count') or 0)}` "
                f"net=`{float(close_gate.get('take_profit_close_net_jpy') or 0.0):.0f} JPY`",
            ]
        )
        segments = close_gate.get("segments") if isinstance(close_gate.get("segments"), list) else []
        if segments:
            lines.append("- Exit segments:")
            for segment in segments[:8]:
                if not isinstance(segment, dict):
                    continue
                lines.append(
                    f"  - `{segment.get('event_type')}:{segment.get('exit_reason')}` "
                    f"count=`{int(segment.get('count') or 0)}` net=`{float(segment.get('net_jpy') or 0.0):.0f}` "
                    f"bot_attributed=`{int(segment.get('bot_attributed_count') or 0)}` "
                    f"gateway_close=`{int(segment.get('gateway_close_sent_count') or 0)}` "
                    f"broker_trade_close=`{int(segment.get('broker_trade_close_accepted_count') or 0)}`"
                )
        daily_close_losses = (
            close_gate.get("loss_side_market_close_daily")
            if isinstance(close_gate.get("loss_side_market_close_daily"), list)
            else []
        )
        if daily_close_losses:
            lines.append("- Loss-side market close daily:")
            for item in daily_close_losses[:8]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"  - `{item.get('day')}` count=`{int(item.get('count') or 0)}` "
                    f"net=`{float(item.get('net_jpy') or 0.0):.0f}` "
                    f"gateway_close=`{int(item.get('gateway_close_sent_count') or 0)}` "
                    f"broker_trade_close=`{int(item.get('broker_trade_close_accepted_count') or 0)}` "
                    f"bot_attributed=`{int(item.get('bot_attributed_count') or 0)}`"
                )
        close_examples = (
            close_gate.get("loss_side_market_close_examples")
            if isinstance(close_gate.get("loss_side_market_close_examples"), list)
            else []
        )
        if close_examples:
            lines.append("- Worst loss-side market close examples:")
            for item in close_examples[:8]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"  - `{item.get('ts_utc')}` `{item.get('pair')} {item.get('side')}` "
                    f"trade=`{item.get('trade_id')}` pl=`{float(item.get('pl_jpy') or 0.0):.0f}` "
                    f"gateway_close=`{item.get('gateway_close_sent')}` "
                    f"gateway_reason=`{','.join(item.get('gateway_close_reasons') or [])}` "
                    f"broker_trade_close=`{item.get('broker_trade_close_accepted')}` "
                    f"bot_attributed=`{item.get('bot_attributed')}`"
                )
        lines.extend(
            [
                "",
                "## Context Coverage",
                "",
                f"- Rows with context features: `{payload['context_feature_coverage']['rows_with_features']}`"
                f"/`{payload['context_feature_coverage']['rows']}`",
                f"- Rows with context theme buckets: `{payload['context_feature_coverage']['rows_with_context_theme_buckets']}`",
            ]
        )
        feature_counts = payload["context_feature_coverage"]["feature_counts"]
        if feature_counts:
            for feature, count in feature_counts.items():
                lines.append(f"- `{feature}` rows=`{count}`")
        else:
            lines.append("- no machine-readable context fields found")
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
        evidence_buckets = payload.get("evidence_bucket_contributions")
        lines.extend(["", "## Evidence Bucket Attributions", ""])
        if isinstance(evidence_buckets, list) and evidence_buckets:
            for item in evidence_buckets[:12]:
                if not isinstance(item, dict):
                    continue
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
                "- In mixed-source runtime backtests, execution-ledger rows stay diagnostic-only until actual exits prove a raw-positive promotable policy.",
                "- Cross-pair context theme buckets are a strict one-bucket overlay selected only from pre-entry FX exposure, not from validation-day outcomes.",
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
    promotable_source_tables: tuple[str, ...],
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
            promotable_source_tables=promotable_source_tables,
        )
        selected_keys = {score.bucket for score in scores}
        validation_rows = tuple(
            row
            for row in trades
            if row.session_date == day
            and row.source_table in promotable_source_tables
            and _row_matches_selected_buckets(row, selected_keys)
        )
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
    promotable_source_tables: tuple[str, ...],
) -> tuple[BucketScore, ...]:
    promotable_rows = tuple(row for row in train_rows if row.source_table in promotable_source_tables)
    base_scores = _select_bucket_candidates(
        promotable_rows,
        bucket_getter=lambda row: (row.bucket,),
        max_loss_jpy=max_loss_jpy,
        min_train_trades=min_train_trades,
        min_train_win_rate_pct=min_train_win_rate_pct,
        max_active_buckets=max_active_buckets,
    )
    context_scores = _select_bucket_candidates(
        promotable_rows,
        bucket_getter=_context_theme_buckets,
        max_loss_jpy=max_loss_jpy,
        min_train_trades=CONTEXT_THEME_MIN_TRAIN_TRADES,
        min_train_win_rate_pct=CONTEXT_THEME_MIN_TRAIN_WIN_RATE_PCT,
        max_active_buckets=CONTEXT_THEME_MAX_ACTIVE_BUCKETS,
    )
    return (*base_scores, *context_scores)


def _context_theme_buckets(row: TestBotTrade) -> tuple[TestBotBucket, ...]:
    return tuple(bucket for bucket in row.extra_buckets if bucket.source_table == CONTEXT_THEME_SOURCE_TABLE)


def _select_bucket_candidates(
    train_rows: tuple[TestBotTrade, ...],
    *,
    bucket_getter,
    max_loss_jpy: float,
    min_train_trades: int,
    min_train_win_rate_pct: float,
    max_active_buckets: int,
) -> tuple[BucketScore, ...]:
    by_bucket: dict[TestBotBucket, list[TestBotTrade]] = {}
    for row in train_rows:
        for bucket in bucket_getter(row):
            by_bucket.setdefault(bucket, []).append(row)
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
    return tuple(
        row
        for row in trades
        if row.session_date == day.session_date
        and any(bucket.label in selected for bucket in _selection_buckets_for_row(row))
    )


def _row_matches_selected_buckets(row: TestBotTrade, selected_buckets: set[TestBotBucket]) -> bool:
    return any(bucket in selected_buckets for bucket in _selection_buckets_for_row(row))


def _selection_buckets_for_row(row: TestBotTrade) -> tuple[TestBotBucket, ...]:
    return (row.bucket, *_context_theme_buckets(row))


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
    # Prefer the stable aggregate trade-history bucket for duplicated trade IDs:
    # pretrade/ledger buckets are useful evidence, but they are currently sparse
    # enough to fragment walk-forward support and erase otherwise repeatable
    # pair-direction history.
    priority = {
        "trades": 4,
        "pretrade_outcomes": 3,
        EXECUTION_LEDGER_SOURCE_TABLE: 2,
        "seat_outcomes": 1,
    }
    base = max(
        rows,
        key=lambda row: (
            priority.get(row.source_table, 0),
            row.sort_key,
            row.source_id,
        ),
    )
    return _merge_deduped_evidence_buckets(base, rows)


def _merge_deduped_evidence_buckets(base: TestBotTrade, rows: list[TestBotTrade]) -> TestBotTrade:
    """Keep one P/L row while preserving pre-entry evidence buckets.

    Dedupe must prevent the same broker trade from counting twice, but it must
    not erase the observable pretrade bucket that would have selected the
    trade before entry. The aggregate `trades` row remains the P/L owner; the
    merged buckets are selection/evidence aliases only.
    """

    buckets: list[TestBotBucket] = list(base.extra_buckets)
    features: set[str] = set(base.context_features)
    for row in rows:
        if row is base:
            continue
        for bucket in (row.bucket, *row.extra_buckets):
            if bucket != base.bucket and bucket not in buckets:
                buckets.append(bucket)
        features.update(row.context_features)
    if tuple(buckets) == base.extra_buckets and tuple(sorted(features)) == base.context_features:
        return base
    return replace(base, extra_buckets=tuple(buckets), context_features=tuple(sorted(features)))


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


def _risk_engine_loss_cap_payload(
    *,
    validation_rows: tuple[TestBotTrade, ...],
    max_loss_jpy: float,
) -> dict[str, float | int | str | None]:
    raw_profit = _round(sum(row.pl_jpy for row in validation_rows if row.pl_jpy > 0))
    raw_loss = abs(_round(sum(row.pl_jpy for row in validation_rows if row.pl_jpy < 0)))
    managed_values = tuple(_capped_pl(row.pl_jpy, max_loss_jpy) for row in validation_rows)
    managed_profit = _round(sum(value for value in managed_values if value > 0))
    managed_loss = abs(_round(sum(value for value in managed_values if value < 0)))
    raw_net = _round(raw_profit - raw_loss)
    managed_net = _round(managed_profit - managed_loss)
    effect = _round(managed_net - raw_net)
    if effect > 0:
        interpretation = "LOSS_CAP_HELPED_SELECTED_POLICY"
    elif effect < 0:
        interpretation = "LOSS_CAP_REDUCED_SELECTED_POLICY"
    else:
        interpretation = "LOSS_CAP_NO_EFFECT_IN_SELECTION"
    return {
        "scope": "selected_validation_trades",
        "selected_trades": len(validation_rows),
        "loss_cap_jpy": _round(max_loss_jpy),
        "raw_selected_net_jpy": raw_net,
        "managed_selected_net_jpy": managed_net,
        "managed_net_minus_raw_net_jpy": effect,
        "raw_gross_profit_jpy": raw_profit,
        "raw_gross_loss_jpy": raw_loss,
        "managed_gross_profit_jpy": managed_profit,
        "managed_gross_loss_jpy": managed_loss,
        "raw_profit_factor": _profit_factor(raw_profit, raw_loss),
        "managed_profit_factor": _profit_factor(managed_profit, managed_loss),
        "loss_reduction_jpy": _round(raw_loss - managed_loss),
        "interpretation": interpretation,
    }


def _target_sizing_ablation_payload(
    *,
    day_results: tuple[TestBotDay, ...],
    start_balance_jpy: float,
    max_loss_jpy: float,
    target_return_pct: float,
) -> dict[str, object]:
    best_day = max(day_results, key=lambda item: item.managed_net_jpy, default=None)
    best_day_net = float(best_day.managed_net_jpy) if best_day is not None else 0.0
    bands: list[dict[str, object]] = []
    for return_pct in _target_band_percentages(target_return_pct):
        target_jpy = _round(start_balance_jpy * (return_pct / 100.0))
        already_hit_days = sum(1 for day in day_results if day.managed_net_jpy >= target_jpy)
        multiplier = _ceil_decimal(target_jpy / best_day_net, places=4) if best_day_net > 0 else None
        scaled_loss_cap = _round(max_loss_jpy * float(multiplier)) if multiplier is not None else None
        scaled_day_values = (
            tuple(_round(day.managed_net_jpy * float(multiplier)) for day in day_results)
            if multiplier is not None
            else tuple()
        )
        scaled_target_hit_days = sum(1 for value in scaled_day_values if value >= target_jpy)
        scaled_max_drawdown = _max_drawdown(scaled_day_values) if scaled_day_values else None
        scaled_worst_day = min(scaled_day_values) if scaled_day_values else None
        if already_hit_days > 0:
            status = "ALREADY_HIT"
        elif multiplier is None:
            status = "NO_SELECTED_PROFIT"
        elif float(multiplier) <= 1.1:
            status = "NEAR_MISS_SIZE_TEST"
        elif float(multiplier) <= 1.5:
            status = "MODERATE_SIZE_UP_REQUIRED"
        else:
            status = "MATERIAL_SIZE_UP_REQUIRED"
        bands.append(
            {
                "return_pct": return_pct,
                "target_jpy": target_jpy,
                "already_hit_days": already_hit_days,
                "required_size_multiplier": multiplier,
                "scaled_loss_cap_jpy": scaled_loss_cap,
                "scaled_target_hit_days": scaled_target_hit_days,
                "scaled_max_drawdown_jpy": _round(scaled_max_drawdown) if scaled_max_drawdown is not None else None,
                "scaled_worst_day_jpy": _round(scaled_worst_day) if scaled_worst_day is not None else None,
                "status": status,
            }
        )
    floor_band = next((item for item in bands if float(item["return_pct"]) == DEFAULT_TARGET_BAND_RETURN_PCTS[0]), None)
    stretch_band = next((item for item in bands if float(item["return_pct"]) == DEFAULT_TARGET_BAND_RETURN_PCTS[-1]), None)
    return {
        "status": _target_sizing_status(floor_band=floor_band, stretch_band=stretch_band),
        "diagnostic_only": True,
        "best_selected_day": best_day.session_date if best_day is not None else None,
        "best_selected_day_jpy": _round(best_day_net),
        "selected_best_return_pct": _round((best_day_net / start_balance_jpy) * 100.0)
        if start_balance_jpy
        else 0.0,
        "bands": bands,
    }


def _target_sizing_status(*, floor_band: dict[str, object] | None, stretch_band: dict[str, object] | None) -> str:
    floor_status = str((floor_band or {}).get("status") or "")
    stretch_status = str((stretch_band or {}).get("status") or "")
    if stretch_status == "ALREADY_HIT":
        return "STRETCH_ALREADY_HIT"
    if floor_status == "ALREADY_HIT":
        return "FLOOR_ALREADY_HIT"
    if floor_status == "NEAR_MISS_SIZE_TEST":
        return "FLOOR_NEAR_MISS_SIZE_TEST"
    if floor_status == "MODERATE_SIZE_UP_REQUIRED":
        return "FLOOR_MODERATE_SIZE_UP_REQUIRED"
    if floor_status == "MATERIAL_SIZE_UP_REQUIRED":
        return "FLOOR_MATERIAL_SIZE_UP_REQUIRED"
    return floor_status or "UNAVAILABLE"


def _close_gate_ablation_payload(
    *,
    db_path: Path | None,
    max_loss_jpy: float,
) -> dict[str, object]:
    base: dict[str, object] = {
        "status": "UNAVAILABLE",
        "reason": "",
        "db_path": str(db_path) if db_path else None,
        "close_events": 0,
        "close_net_jpy": 0.0,
        "bot_attributed_close_events": 0,
        "gateway_order_sent_events": 0,
        "gateway_close_sent_events": 0,
        "broker_trade_close_accept_events": 0,
        "broker_trade_close_accept_trade_ids": 0,
        "broker_trade_close_accept_order_ids": 0,
        "loss_side_market_close_count": 0,
        "loss_side_market_close_net_jpy": 0.0,
        "bot_attributed_loss_side_market_close_count": 0,
        "bot_attributed_loss_side_market_close_net_jpy": 0.0,
        "gateway_loss_side_market_close_count": 0,
        "gateway_loss_side_market_close_net_jpy": 0.0,
        "gateway_gpt_close_loss_side_market_close_count": 0,
        "gateway_gpt_close_loss_side_market_close_net_jpy": 0.0,
        "gateway_review_exit_loss_side_market_close_count": 0,
        "gateway_review_exit_loss_side_market_close_net_jpy": 0.0,
        "gateway_other_loss_side_market_close_count": 0,
        "gateway_other_loss_side_market_close_net_jpy": 0.0,
        "broker_trade_close_loss_side_market_close_count": 0,
        "broker_trade_close_loss_side_market_close_net_jpy": 0.0,
        "broker_accepted_without_gateway_loss_side_market_close_count": 0,
        "broker_accepted_without_gateway_loss_side_market_close_net_jpy": 0.0,
        "unattributed_loss_side_market_close_count": 0,
        "unattributed_loss_side_market_close_net_jpy": 0.0,
        "take_profit_close_count": 0,
        "take_profit_close_net_jpy": 0.0,
        "loss_side_market_close_daily": [],
        "loss_side_market_close_examples": [],
        "segments": [],
    }
    if db_path is None:
        base["reason"] = "execution ledger path was not provided"
        return base
    if not db_path.exists():
        base["reason"] = f"execution ledger not found: {db_path}"
        return base
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
            ).fetchone()
            if not table:
                base["reason"] = "execution_events table is missing"
                return base
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
                if row["name"] is not None
            }
            required = {"event_type", "trade_id", "realized_pl_jpy"}
            if not required <= columns:
                base["reason"] = "execution_events lacks close diagnostic columns"
                return base
            gateway_order_sent_events = _count_events(conn, "GATEWAY_ORDER_SENT")
            gateway_close_sent_events = _count_events(conn, "GATEWAY_TRADE_CLOSE_SENT")
            attributed_trade_ids = _gateway_attributed_entry_trade_ids(conn, columns)
            gateway_close_trade_ids = _event_trade_ids(conn, "GATEWAY_TRADE_CLOSE_SENT")
            gateway_close_reasons_by_trade_id = _event_trade_id_reasons(
                conn,
                "GATEWAY_TRADE_CLOSE_SENT",
                columns,
            )
            broker_close_accept = _broker_trade_close_accept_provenance(conn, columns)
            close_rows = _close_event_rows(conn, columns)
    except sqlite3.Error as exc:
        base["reason"] = f"execution ledger unreadable: {exc}"
        return base

    if not close_rows:
        base.update(
            {
                "status": "NO_CLOSES",
                "reason": "execution ledger has no closed/reduced trade events",
                "gateway_order_sent_events": gateway_order_sent_events,
                "gateway_close_sent_events": gateway_close_sent_events,
                "broker_trade_close_accept_events": broker_close_accept["events"],
                "broker_trade_close_accept_trade_ids": len(broker_close_accept["trade_ids"]),
                "broker_trade_close_accept_order_ids": len(broker_close_accept["order_ids"]),
            }
        )
        return base

    segments: dict[tuple[str, str], dict[str, object]] = {}
    close_net = 0.0
    bot_attributed_count = 0
    loss_side_market_count = 0
    loss_side_market_net = 0.0
    bot_loss_market_count = 0
    bot_loss_market_net = 0.0
    gateway_loss_market_count = 0
    gateway_loss_market_net = 0.0
    gateway_gpt_loss_market_count = 0
    gateway_gpt_loss_market_net = 0.0
    gateway_review_exit_loss_market_count = 0
    gateway_review_exit_loss_market_net = 0.0
    gateway_other_loss_market_count = 0
    gateway_other_loss_market_net = 0.0
    broker_loss_market_count = 0
    broker_loss_market_net = 0.0
    broker_without_gateway_loss_market_count = 0
    broker_without_gateway_loss_market_net = 0.0
    unattributed_loss_market_count = 0
    unattributed_loss_market_net = 0.0
    take_profit_count = 0
    take_profit_net = 0.0
    loss_market_examples: list[dict[str, object]] = []
    loss_market_daily: dict[str, dict[str, object]] = {}
    for row in close_rows:
        pl = _maybe_float(row["realized_pl_jpy"])
        if pl is None:
            continue
        close_net += pl
        trade_id = str(row["trade_id"] or "").strip()
        exit_reason = _close_exit_reason(row)
        event_type = _norm(row["event_type"])
        order_id = str(row["order_id"] or "").strip()
        bot_attributed = trade_id in attributed_trade_ids
        gateway_close_sent = trade_id in gateway_close_trade_ids
        gateway_close_reasons = gateway_close_reasons_by_trade_id.get(trade_id, set())
        gateway_gpt_close = "GPT_CLOSE" in gateway_close_reasons
        gateway_review_exit = "REVIEW_EXIT" in gateway_close_reasons and not gateway_gpt_close
        gateway_other_close = gateway_close_sent and not gateway_gpt_close and not gateway_review_exit
        broker_trade_close_accepted = (
            trade_id in broker_close_accept["trade_ids"] or order_id in broker_close_accept["order_ids"]
        )
        close_order_provenance = gateway_close_sent or broker_trade_close_accepted
        if bot_attributed:
            bot_attributed_count += 1
        segment = segments.setdefault(
            (event_type, exit_reason),
            {
                "event_type": event_type,
                "exit_reason": exit_reason,
                "count": 0,
                "net_jpy": 0.0,
                "capped_net_jpy": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "bot_attributed_count": 0,
                "bot_attributed_net_jpy": 0.0,
                "gateway_close_sent_count": 0,
                "gateway_close_sent_net_jpy": 0.0,
                "gateway_gpt_close_count": 0,
                "gateway_gpt_close_net_jpy": 0.0,
                "gateway_review_exit_count": 0,
                "gateway_review_exit_net_jpy": 0.0,
                "gateway_other_close_count": 0,
                "gateway_other_close_net_jpy": 0.0,
                "broker_trade_close_accepted_count": 0,
                "broker_trade_close_accepted_net_jpy": 0.0,
            },
        )
        segment["count"] = int(segment["count"]) + 1
        segment["net_jpy"] = _round(float(segment["net_jpy"]) + pl)
        segment["capped_net_jpy"] = _round(float(segment["capped_net_jpy"]) + _capped_pl(pl, max_loss_jpy))
        segment["win_count"] = int(segment["win_count"]) + (1 if pl > 0 else 0)
        segment["loss_count"] = int(segment["loss_count"]) + (1 if pl < 0 else 0)
        if bot_attributed:
            segment["bot_attributed_count"] = int(segment["bot_attributed_count"]) + 1
            segment["bot_attributed_net_jpy"] = _round(float(segment["bot_attributed_net_jpy"]) + pl)
        if gateway_close_sent:
            segment["gateway_close_sent_count"] = int(segment["gateway_close_sent_count"]) + 1
            segment["gateway_close_sent_net_jpy"] = _round(float(segment["gateway_close_sent_net_jpy"]) + pl)
        if gateway_gpt_close:
            segment["gateway_gpt_close_count"] = int(segment["gateway_gpt_close_count"]) + 1
            segment["gateway_gpt_close_net_jpy"] = _round(float(segment["gateway_gpt_close_net_jpy"]) + pl)
        if gateway_review_exit:
            segment["gateway_review_exit_count"] = int(segment["gateway_review_exit_count"]) + 1
            segment["gateway_review_exit_net_jpy"] = _round(float(segment["gateway_review_exit_net_jpy"]) + pl)
        if gateway_other_close:
            segment["gateway_other_close_count"] = int(segment["gateway_other_close_count"]) + 1
            segment["gateway_other_close_net_jpy"] = _round(float(segment["gateway_other_close_net_jpy"]) + pl)
        if broker_trade_close_accepted:
            segment["broker_trade_close_accepted_count"] = int(segment["broker_trade_close_accepted_count"]) + 1
            segment["broker_trade_close_accepted_net_jpy"] = _round(
                float(segment["broker_trade_close_accepted_net_jpy"]) + pl
            )
        if exit_reason == "TAKE_PROFIT_ORDER":
            take_profit_count += 1
            take_profit_net += pl
        if exit_reason == "MARKET_ORDER_TRADE_CLOSE" and pl < 0:
            loss_side_market_count += 1
            loss_side_market_net += pl
            ts_utc = str(row["ts_utc"] or "")
            day_key = ts_utc[:10] or "UNKNOWN"
            daily = loss_market_daily.setdefault(
                day_key,
                {
                    "day": day_key,
                    "count": 0,
                    "net_jpy": 0.0,
                    "gateway_close_sent_count": 0,
                    "gateway_gpt_close_count": 0,
                    "gateway_review_exit_count": 0,
                    "gateway_other_close_count": 0,
                    "broker_trade_close_accepted_count": 0,
                    "broker_accepted_without_gateway_count": 0,
                    "no_close_order_provenance_count": 0,
                    "bot_attributed_count": 0,
                },
            )
            daily["count"] = int(daily["count"]) + 1
            daily["net_jpy"] = _round(float(daily["net_jpy"]) + pl)
            if gateway_close_sent:
                daily["gateway_close_sent_count"] = int(daily["gateway_close_sent_count"]) + 1
            if gateway_gpt_close:
                daily["gateway_gpt_close_count"] = int(daily["gateway_gpt_close_count"]) + 1
            if gateway_review_exit:
                daily["gateway_review_exit_count"] = int(daily["gateway_review_exit_count"]) + 1
            if gateway_other_close:
                daily["gateway_other_close_count"] = int(daily["gateway_other_close_count"]) + 1
            if broker_trade_close_accepted:
                daily["broker_trade_close_accepted_count"] = int(daily["broker_trade_close_accepted_count"]) + 1
            if broker_trade_close_accepted and not gateway_close_sent:
                daily["broker_accepted_without_gateway_count"] = int(
                    daily["broker_accepted_without_gateway_count"]
                ) + 1
            if not close_order_provenance:
                daily["no_close_order_provenance_count"] = int(daily["no_close_order_provenance_count"]) + 1
            if bot_attributed:
                daily["bot_attributed_count"] = int(daily["bot_attributed_count"]) + 1
            loss_market_examples.append(
                {
                    "ts_utc": ts_utc,
                    "pair": str(row["pair"] or ""),
                    "side": str(row["side"] or ""),
                    "trade_id": trade_id,
                    "lane_id": str(row["lane_id"] or ""),
                    "order_id": str(row["order_id"] or ""),
                    "pl_jpy": _round(pl),
                    "bot_attributed": bot_attributed,
                    "gateway_close_sent": gateway_close_sent,
                    "gateway_close_reasons": sorted(gateway_close_reasons),
                    "broker_trade_close_accepted": broker_trade_close_accepted,
                    "close_order_provenance": close_order_provenance,
                }
            )
            if bot_attributed:
                bot_loss_market_count += 1
                bot_loss_market_net += pl
            if gateway_close_sent:
                gateway_loss_market_count += 1
                gateway_loss_market_net += pl
            if gateway_gpt_close:
                gateway_gpt_loss_market_count += 1
                gateway_gpt_loss_market_net += pl
            if gateway_review_exit:
                gateway_review_exit_loss_market_count += 1
                gateway_review_exit_loss_market_net += pl
            if gateway_other_close:
                gateway_other_loss_market_count += 1
                gateway_other_loss_market_net += pl
            if broker_trade_close_accepted:
                broker_loss_market_count += 1
                broker_loss_market_net += pl
            if broker_trade_close_accepted and not gateway_close_sent:
                broker_without_gateway_loss_market_count += 1
                broker_without_gateway_loss_market_net += pl
            if not close_order_provenance:
                unattributed_loss_market_count += 1
                unattributed_loss_market_net += pl
    segment_rows = sorted(
        (
            {
                **item,
                "net_jpy": _round(float(item["net_jpy"])),
                "capped_net_jpy": _round(float(item["capped_net_jpy"])),
                "bot_attributed_net_jpy": _round(float(item["bot_attributed_net_jpy"])),
                "gateway_close_sent_net_jpy": _round(float(item["gateway_close_sent_net_jpy"])),
                "gateway_gpt_close_net_jpy": _round(float(item["gateway_gpt_close_net_jpy"])),
                "gateway_review_exit_net_jpy": _round(float(item["gateway_review_exit_net_jpy"])),
                "gateway_other_close_net_jpy": _round(float(item["gateway_other_close_net_jpy"])),
                "broker_trade_close_accepted_net_jpy": _round(
                    float(item["broker_trade_close_accepted_net_jpy"])
                ),
            }
            for item in segments.values()
        ),
        key=lambda item: (float(item["net_jpy"]), -int(item["count"])),
    )
    loss_market_daily_rows = sorted(
        (
            {
                **item,
                "net_jpy": _round(float(item["net_jpy"])),
            }
            for item in loss_market_daily.values()
        ),
        key=lambda item: str(item["day"]),
        reverse=True,
    )
    loss_market_example_rows = sorted(
        loss_market_examples,
        key=lambda item: float(item.get("pl_jpy") or 0.0),
    )[:8]
    base.update(
        {
            "status": "MEASURED",
            "reason": "broker-truth execution ledger close diagnostics; not a live permission override",
            "close_events": len(close_rows),
            "close_net_jpy": _round(close_net),
            "bot_attributed_close_events": bot_attributed_count,
            "gateway_order_sent_events": gateway_order_sent_events,
            "gateway_close_sent_events": gateway_close_sent_events,
            "broker_trade_close_accept_events": broker_close_accept["events"],
            "broker_trade_close_accept_trade_ids": len(broker_close_accept["trade_ids"]),
            "broker_trade_close_accept_order_ids": len(broker_close_accept["order_ids"]),
            "loss_side_market_close_count": loss_side_market_count,
            "loss_side_market_close_net_jpy": _round(loss_side_market_net),
            "bot_attributed_loss_side_market_close_count": bot_loss_market_count,
            "bot_attributed_loss_side_market_close_net_jpy": _round(bot_loss_market_net),
            "gateway_loss_side_market_close_count": gateway_loss_market_count,
            "gateway_loss_side_market_close_net_jpy": _round(gateway_loss_market_net),
            "gateway_gpt_close_loss_side_market_close_count": gateway_gpt_loss_market_count,
            "gateway_gpt_close_loss_side_market_close_net_jpy": _round(gateway_gpt_loss_market_net),
            "gateway_review_exit_loss_side_market_close_count": gateway_review_exit_loss_market_count,
            "gateway_review_exit_loss_side_market_close_net_jpy": _round(gateway_review_exit_loss_market_net),
            "gateway_other_loss_side_market_close_count": gateway_other_loss_market_count,
            "gateway_other_loss_side_market_close_net_jpy": _round(gateway_other_loss_market_net),
            "broker_trade_close_loss_side_market_close_count": broker_loss_market_count,
            "broker_trade_close_loss_side_market_close_net_jpy": _round(broker_loss_market_net),
            "broker_accepted_without_gateway_loss_side_market_close_count": broker_without_gateway_loss_market_count,
            "broker_accepted_without_gateway_loss_side_market_close_net_jpy": _round(
                broker_without_gateway_loss_market_net
            ),
            "unattributed_loss_side_market_close_count": unattributed_loss_market_count,
            "unattributed_loss_side_market_close_net_jpy": _round(unattributed_loss_market_net),
            "take_profit_close_count": take_profit_count,
            "take_profit_close_net_jpy": _round(take_profit_net),
            "loss_side_market_close_daily": loss_market_daily_rows,
            "loss_side_market_close_examples": loss_market_example_rows,
            "segments": segment_rows,
        }
    )
    return base


def _count_events(conn: sqlite3.Connection, event_type: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM execution_events WHERE event_type = ?",
        (event_type,),
    ).fetchone()
    return int(row["n"] or 0) if row else 0


def _event_trade_ids(conn: sqlite3.Connection, event_type: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT trade_id
        FROM execution_events
        WHERE event_type = ?
          AND trade_id IS NOT NULL
          AND trade_id != ''
        """,
        (event_type,),
    ).fetchall()
    return {str(row["trade_id"]).strip() for row in rows if str(row["trade_id"] or "").strip()}


def _event_trade_id_reasons(
    conn: sqlite3.Connection,
    event_type: str,
    columns: set[str],
) -> dict[str, set[str]]:
    select_fields = ["trade_id"]
    for column in ("exit_reason", "raw_json"):
        if column in columns:
            select_fields.append(column)
        else:
            select_fields.append(f"NULL AS {column}")
    rows = conn.execute(
        f"""
        SELECT {', '.join(select_fields)}
        FROM execution_events
        WHERE event_type = ?
          AND trade_id IS NOT NULL
          AND trade_id != ''
        """,
        (event_type,),
    ).fetchall()
    out: dict[str, set[str]] = {}
    for row in rows:
        trade_id = str(row["trade_id"] or "").strip()
        if not trade_id:
            continue
        reason = _gateway_close_reason(row)
        out.setdefault(trade_id, set()).add(reason)
    return out


def _gateway_close_reason(row: sqlite3.Row) -> str:
    reason = _norm(row["exit_reason"])
    raw = _raw_payload(row["raw_json"])
    reason_text = " ".join(str(item) for item in raw.get("reasons", []) or []).lower()
    if reason == "GPT_CLOSE" or "gpt-close: accepted gpt_trader close receipt passed gate a/b" in reason_text:
        return "GPT_CLOSE"
    if reason and reason != "UNSPECIFIED":
        return reason
    management_action = _norm(raw.get("management_action"))
    if management_action:
        return management_action
    request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
    if _norm(request.get("type")) == "CLOSE":
        return "CLOSE"
    return "UNSPECIFIED"


def _broker_trade_close_accept_provenance(
    conn: sqlite3.Connection,
    columns: set[str],
) -> dict[str, int | set[str]]:
    select_fields = []
    for column in ("trade_id", "order_id", "exit_reason", "raw_json"):
        if column in columns:
            select_fields.append(column)
        else:
            select_fields.append(f"NULL AS {column}")
    rows = conn.execute(
        f"""
        SELECT {', '.join(select_fields)}
        FROM execution_events
        WHERE event_type = 'ORDER_ACCEPTED'
        """
    ).fetchall()
    trade_ids: set[str] = set()
    order_ids: set[str] = set()
    events = 0
    for row in rows:
        raw = _raw_payload(row["raw_json"])
        trade_close = raw.get("tradeClose") if isinstance(raw.get("tradeClose"), dict) else {}
        reason = _norm(row["exit_reason"]) or _norm(raw.get("reason"))
        if reason != "TRADE_CLOSE" and not trade_close:
            continue
        events += 1
        order_id = str(row["order_id"] or "").strip()
        row_trade_id = str(row["trade_id"] or "").strip()
        close_trade_id = _broker_trade_close_trade_id(trade_close)
        if order_id:
            order_ids.add(order_id)
        if close_trade_id:
            trade_ids.add(close_trade_id)
        elif row_trade_id:
            trade_ids.add(row_trade_id)
    return {"events": events, "trade_ids": trade_ids, "order_ids": order_ids}


def _broker_trade_close_trade_id(trade_close: object) -> str:
    if not isinstance(trade_close, dict):
        return ""
    for key in ("tradeID", "trade_id", "id"):
        trade_id = str(trade_close.get(key) or "").strip()
        if trade_id:
            return trade_id
    return ""


def _gateway_attributed_entry_trade_ids(conn: sqlite3.Connection, columns: set[str]) -> set[str]:
    if "order_id" not in columns or "lane_id" not in columns:
        return set()
    gateway_rows = conn.execute(
        """
        SELECT trade_id, order_id, lane_id
        FROM execution_events
        WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
          AND lane_id IS NOT NULL
          AND lane_id != ''
        """
    ).fetchall()
    gateway_order_ids = {str(row["order_id"]).strip() for row in gateway_rows if str(row["order_id"] or "").strip()}
    gateway_trade_ids = {str(row["trade_id"]).strip() for row in gateway_rows if str(row["trade_id"] or "").strip()}
    gateway_lane_trade_ids = {
        str(row["trade_id"]).strip()
        for row in gateway_rows
        if str(row["trade_id"] or "").strip() and str(row["lane_id"] or "").strip()
    }
    filled_select = ["trade_id", "order_id"]
    if "lane_id" in columns:
        filled_select.append("lane_id")
    else:
        filled_select.append("NULL AS lane_id")
    filled_rows = conn.execute(
        f"""
        SELECT {', '.join(filled_select)}
        FROM execution_events
        WHERE event_type = 'ORDER_FILLED'
          AND trade_id IS NOT NULL
          AND trade_id != ''
        """
    ).fetchall()
    attributed: set[str] = set(gateway_lane_trade_ids)
    for row in filled_rows:
        trade_id = str(row["trade_id"] or "").strip()
        order_id = str(row["order_id"] or "").strip()
        lane_id = str(row["lane_id"] or "").strip()
        if not trade_id:
            continue
        if trade_id in gateway_trade_ids or order_id in gateway_order_ids or lane_id:
            attributed.add(trade_id)
    return attributed


def _close_event_rows(conn: sqlite3.Connection, columns: set[str]) -> list[sqlite3.Row]:
    select_fields = [
        "event_type",
        "trade_id",
        "realized_pl_jpy",
    ]
    for column in ("event_uid", "ts_utc", "pair", "side", "units", "order_id", "lane_id", "exit_reason", "raw_json"):
        if column in columns:
            select_fields.append(column)
        else:
            select_fields.append(f"NULL AS {column}")
    return list(
        conn.execute(
            f"""
            SELECT {', '.join(select_fields)}
            FROM execution_events
            WHERE event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
              AND realized_pl_jpy IS NOT NULL
            ORDER BY ts_utc, event_uid
            """
        ).fetchall()
    )


def _close_exit_reason(row: sqlite3.Row) -> str:
    reason = _norm(row["exit_reason"])
    if reason and reason != "UNSPECIFIED":
        return reason
    raw = _raw_payload(row["raw_json"])
    return _norm(raw.get("reason")) or "UNSPECIFIED"


def _maybe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    promotable_source_tables: tuple[str, ...],
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
            promotable_source_tables=promotable_source_tables,
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
                "best_top_n_day_jpy": float(band_ceiling["best_top_n_day_jpy"] or 0.0),
                "best_all_positive_day_jpy": float(band_ceiling["best_all_positive_day_jpy"] or 0.0),
                "best_train_eligible_all_positive_day_jpy": float(
                    band_ceiling["best_train_eligible_all_positive_day_jpy"] or 0.0
                ),
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
    best_selected_day_jpy = max((float(item["best_selected_day_jpy"] or 0.0) for item in bands), default=0.0)
    best_top_n_day_jpy = max((float(item["best_top_n_day_jpy"] or 0.0) for item in bands), default=0.0)
    best_all_positive_day_jpy = max((float(item["best_all_positive_day_jpy"] or 0.0) for item in bands), default=0.0)
    best_train_eligible_day_jpy = max(
        (float(item["best_train_eligible_all_positive_day_jpy"] or 0.0) for item in bands),
        default=0.0,
    )
    return {
        "floor_return_pct": floor_pct,
        "stretch_return_pct": stretch_pct,
        "status": status,
        "selected_attainable_return_pct": selected_attainable,
        "top_n_oracle_attainable_return_pct": top_n_attainable,
        "train_eligible_oracle_attainable_return_pct": train_eligible_attainable,
        "all_positive_oracle_attainable_return_pct": all_positive_attainable,
        "selected_best_return_pct": _round((best_selected_day_jpy / start_balance_jpy) * 100.0)
        if start_balance_jpy
        else 0.0,
        "top_n_oracle_best_return_pct": _round((best_top_n_day_jpy / start_balance_jpy) * 100.0)
        if start_balance_jpy
        else 0.0,
        "train_eligible_oracle_best_return_pct": _round((best_train_eligible_day_jpy / start_balance_jpy) * 100.0)
        if start_balance_jpy
        else 0.0,
        "all_positive_oracle_best_return_pct": _round((best_all_positive_day_jpy / start_balance_jpy) * 100.0)
        if start_balance_jpy
        else 0.0,
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
    risk_engine_loss_cap: dict[str, float | int | str | None],
    target_sizing: dict[str, object],
    close_gate_ab: dict[str, object],
) -> Iterable[str]:
    if not blockers:
        return
    risk_effect = float(risk_engine_loss_cap.get("managed_net_minus_raw_net_jpy") or 0.0)
    if risk_effect > 0:
        yield (
            "RiskEngine loss-cap ablation is currently positive "
            f"(managed-minus-raw {risk_effect:.0f} JPY); any cap removal/weakening candidate must beat the raw selected baseline, not just move faster"
        )
    elif risk_effect < 0:
        yield (
            "RiskEngine loss-cap ablation currently suppresses selected-policy net "
            f"({risk_effect:.0f} JPY); test a replacement cap surface against drawdown before live promotion"
        )
    yield from _target_sizing_action_items(target_sizing)
    yield from _close_gate_action_items(close_gate_ab)
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
            and selected == 0
            and validation_raw_net < 0
        ):
            yield (
                "execution_ledger is raw-negative and remains diagnostic-only in the mixed-source policy "
                f"(raw {validation_raw_net:.0f} JPY across {validation_rows} closes); "
                "keep improving actual exit timing before promoting ledger buckets into target pacing"
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
    yield (
        "legacy rows do not persist market-context-matrix/news/non-FX context as a dense pre-entry feature set; "
        "store matrix refs, news refs, and gold/oil/context-asset readings on each live entry receipt before treating "
        "non-FX/news prediction as backtest-certified"
    )


def _target_sizing_action_items(payload: dict[str, object]) -> Iterable[str]:
    bands = payload.get("bands") if isinstance(payload.get("bands"), list) else []
    by_pct = {
        float(item.get("return_pct")): item
        for item in bands
        if isinstance(item, dict) and item.get("return_pct") is not None
    }
    floor = by_pct.get(DEFAULT_TARGET_BAND_RETURN_PCTS[0])
    stretch = by_pct.get(DEFAULT_TARGET_BAND_RETURN_PCTS[-1])
    if isinstance(floor, dict) and floor.get("status") in {"NEAR_MISS_SIZE_TEST", "MODERATE_SIZE_UP_REQUIRED"}:
        multiplier = float(floor.get("required_size_multiplier") or 0.0)
        scaled_cap = float(floor.get("scaled_loss_cap_jpy") or 0.0)
        yield (
            f"5% floor is a target-aware sizing candidate: selected best day needs {multiplier:.2f}x size "
            f"(scaled per-trade cap about {scaled_cap:.0f} JPY); verify with margin/drawdown replay before live sizing"
        )
    if isinstance(stretch, dict) and stretch.get("status") == "MATERIAL_SIZE_UP_REQUIRED":
        multiplier = float(stretch.get("required_size_multiplier") or 0.0)
        yield (
            f"10% stretch cannot be treated as a small sizing tweak; selected best day needs {multiplier:.2f}x size, "
            "so coverage/reward geometry must improve before real-time target-chasing is credible"
        )


def _close_gate_action_items(payload: dict[str, object]) -> Iterable[str]:
    status = str(payload.get("status") or "")
    close_events = int(payload.get("close_events") or 0)
    bot_attributed = int(payload.get("bot_attributed_close_events") or 0)
    if status == "MEASURED" and close_events and bot_attributed == 0:
        yield (
            "execution ledger has broker close outcomes but zero gateway-attributed entry closes; "
            "broker truth/live gateway/CLOSE Gate A-B ablation is not attributable until GATEWAY_ORDER_SENT receipts link to fills"
        )
    broker_without_gateway_loss_count = int(
        payload.get("broker_accepted_without_gateway_loss_side_market_close_count") or 0
    )
    broker_without_gateway_loss_net = float(
        payload.get("broker_accepted_without_gateway_loss_side_market_close_net_jpy") or 0.0
    )
    if broker_without_gateway_loss_count:
        yield (
            "broker accepted TRADE_CLOSE orders exist without matching local GATEWAY_TRADE_CLOSE_SENT receipts "
            f"({broker_without_gateway_loss_count} loss-side close(s), net {broker_without_gateway_loss_net:.0f} JPY); "
            "tag whether these were GPT/gateway, operator, or broker-sync-only closes before changing CLOSE Gate A-B policy"
        )
    unattributed_loss_count = int(payload.get("unattributed_loss_side_market_close_count") or 0)
    unattributed_loss_net = float(payload.get("unattributed_loss_side_market_close_net_jpy") or 0.0)
    if unattributed_loss_count:
        yield (
            "loss-side market closes lack both gateway close receipts and broker accepted TRADE_CLOSE provenance "
            f"({unattributed_loss_count} close(s), net {unattributed_loss_net:.0f} JPY); "
            "separate manual/intervention closes from Gate A-B strategy closes before tuning exit permissions"
        )
    gateway_gpt_loss_count = int(payload.get("gateway_gpt_close_loss_side_market_close_count") or 0)
    gateway_gpt_loss_net = float(payload.get("gateway_gpt_close_loss_side_market_close_net_jpy") or 0.0)
    if gateway_gpt_loss_count and gateway_gpt_loss_net < 0:
        yield (
            "CLOSE Gate A-B accepted loss-side market closes are net negative "
            f"({gateway_gpt_loss_count} close(s), {gateway_gpt_loss_net:.0f} JPY); "
            "ablate hard/soft Gate A evidence separately before widening autonomous CLOSE"
        )
    gateway_review_loss_count = int(payload.get("gateway_review_exit_loss_side_market_close_count") or 0)
    gateway_review_loss_net = float(payload.get("gateway_review_exit_loss_side_market_close_net_jpy") or 0.0)
    if gateway_review_loss_count and gateway_review_loss_net < 0:
        yield (
            "legacy gateway REVIEW_EXIT loss-side market closes are net negative "
            f"({gateway_review_loss_count} close(s), {gateway_review_loss_net:.0f} JPY); "
            "do not count them as current Gate A-B evidence, and keep plain auto-close blocked until structural replay proves it"
        )
    market_loss_net = float(payload.get("loss_side_market_close_net_jpy") or 0.0)
    tp_net = float(payload.get("take_profit_close_net_jpy") or 0.0)
    if market_loss_net < 0 and tp_net > 0:
        yield (
            "exit split shows take-profit closes positive while loss-side market closes are negative; "
            "focus on CLOSE trigger attribution/timing rather than blanket TP removal"
        )


def _bucket_contributions(
    validation_rows: tuple[TestBotTrade, ...],
    max_loss_jpy: float,
) -> list[dict[str, float | int | str]]:
    return _bucket_contribution_rows(
        validation_rows,
        max_loss_jpy=max_loss_jpy,
        bucket_getter=lambda row: (row.bucket,),
    )


def _evidence_bucket_contributions(
    validation_rows: tuple[TestBotTrade, ...],
    max_loss_jpy: float,
) -> list[dict[str, float | int | str]]:
    return _bucket_contribution_rows(
        validation_rows,
        max_loss_jpy=max_loss_jpy,
        bucket_getter=lambda row: row.buckets,
    )


def _bucket_contribution_rows(
    validation_rows: tuple[TestBotTrade, ...],
    *,
    max_loss_jpy: float,
    bucket_getter,
) -> list[dict[str, float | int | str]]:
    by_bucket: dict[TestBotBucket, dict[str, object]] = {}
    for row in validation_rows:
        capped = _capped_pl(row.pl_jpy, max_loss_jpy)
        for bucket in bucket_getter(row):
            item = by_bucket.setdefault(
                bucket,
                {
                    "bucket": bucket.label,
                    "trades": 0,
                    "wins": 0,
                    "managed_net_jpy": 0.0,
                    "raw_net_jpy": 0.0,
                    "days": set(),
                    "worst_trade_jpy": None,
                    "best_trade_jpy": None,
                },
            )
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
    promotable_source_tables: tuple[str, ...],
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
    promotable_sources = set(promotable_source_tables)
    for day in day_results:
        bucket_net = _bucket_net_for_day(
            trades,
            day.session_date,
            max_loss_jpy,
            promotable_source_tables=promotable_sources,
        )
        train_eligible_buckets = _train_eligible_buckets(
            trades,
            day,
            max_loss_jpy=max_loss_jpy,
            min_train_trades=min_train_trades,
            min_train_win_rate_pct=min_train_win_rate_pct,
            promotable_source_tables=promotable_sources,
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
    promotable_source_tables: tuple[str, ...],
    limit: int,
) -> list[dict[str, float | str]]:
    misses: list[dict[str, float | str]] = []
    promotable_sources = set(promotable_source_tables)
    for day in day_results:
        bucket_net = _bucket_net_for_day(
            trades,
            day.session_date,
            max_loss_jpy,
            promotable_source_tables=promotable_sources,
        )
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
    promotable_source_tables: set[str] | None = None,
) -> dict[TestBotBucket, float]:
    bucket_net: dict[TestBotBucket, float] = {}
    for row in trades:
        if row.session_date != session_date:
            continue
        if promotable_source_tables is not None and row.source_table not in promotable_source_tables:
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
    promotable_source_tables: set[str],
) -> set[TestBotBucket]:
    by_bucket: dict[TestBotBucket, list[TestBotTrade]] = {}
    for row in trades:
        if not (day.training_start_date <= row.session_date <= day.training_end_date):
            continue
        if row.source_table not in promotable_source_tables:
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


def _promotable_source_tables(source_tables: tuple[str, ...]) -> tuple[str, ...]:
    if EXECUTION_LEDGER_SOURCE_TABLE in source_tables and len(set(source_tables)) > 1:
        return tuple(source for source in source_tables if source != EXECUTION_LEDGER_SOURCE_TABLE)
    return source_tables


def _execution_ledger_selection_mode(
    source_tables: tuple[str, ...],
    promotable_source_tables: tuple[str, ...],
) -> str:
    if EXECUTION_LEDGER_SOURCE_TABLE not in source_tables:
        return "not_requested"
    if EXECUTION_LEDGER_SOURCE_TABLE in promotable_source_tables:
        return "promotable_explicit_execution_ledger_only"
    return "diagnostic_only_mixed_sources"


def _context_feature_coverage(trades: tuple[TestBotTrade, ...]) -> dict[str, object]:
    feature_counts: dict[str, int] = {}
    rows_with_features = 0
    rows_with_context_theme_buckets = 0
    for row in trades:
        if row.context_features:
            rows_with_features += 1
        if row.extra_buckets:
            rows_with_context_theme_buckets += 1
        for feature in row.context_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    return {
        "rows": len(trades),
        "rows_with_features": rows_with_features,
        "rows_with_context_theme_buckets": rows_with_context_theme_buckets,
        "feature_counts": dict(sorted(feature_counts.items())),
    }


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
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
                  AND lane_id IS NOT NULL
                  AND lane_id != ''
            ),
            entries AS (
                SELECT
                    e.trade_id,
                    MAX(g.lane_id) AS gateway_lane_id,
                    CASE
                        WHEN SUM(CASE WHEN UPPER(COALESCE(e.side, '')) = 'LONG' THEN 1 ELSE 0 END) > 0 THEN 'LONG'
                        WHEN SUM(CASE WHEN UPPER(COALESCE(e.side, '')) = 'SHORT' THEN 1 ELSE 0 END) > 0 THEN 'SHORT'
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


def _extra_observable_buckets(
    *,
    source_table: str,
    pair: str,
    direction: str,
) -> tuple[TestBotBucket, ...]:
    if source_table != "trades":
        return ()
    split = _split_fx_pair(pair)
    if split is None or direction not in {"LONG", "SHORT"}:
        return ()
    base, _quote = split
    exposures = dict(_currency_exposures(pair, direction))
    buckets: list[TestBotBucket] = []
    if exposures.get("JPY") == "SHORT" and base != "USD":
        buckets.append(
            TestBotBucket(
                CONTEXT_THEME_SOURCE_TABLE,
                "RISK_ON_JPY_CROSS",
                "LONG",
                "FX_RISK_THEME",
                "ALL",
            )
        )
    if exposures.get("USD") == "SHORT" and base in _COMMODITY_LINKED_CURRENCIES:
        buckets.append(
            TestBotBucket(
                CONTEXT_THEME_SOURCE_TABLE,
                "COMMODITY_FX_USD_WEAK",
                "LONG",
                "FX_RISK_THEME",
                "ALL",
            )
        )
    if exposures.get("USD") == "SHORT" and base in _EUROPEAN_RISK_CURRENCIES:
        buckets.append(
            TestBotBucket(
                CONTEXT_THEME_SOURCE_TABLE,
                "EUROPE_FX_USD_WEAK",
                "LONG",
                "FX_RISK_THEME",
                "ALL",
            )
        )
    if exposures.get("USD") == "LONG" and base in (_COMMODITY_LINKED_CURRENCIES | _EUROPEAN_RISK_CURRENCIES):
        buckets.append(
            TestBotBucket(
                CONTEXT_THEME_SOURCE_TABLE,
                "USD_STRENGTH_AGAINST_RISK",
                "SHORT",
                "FX_RISK_THEME",
                "ALL",
            )
        )
    return tuple(buckets)


def _split_fx_pair(pair: str) -> tuple[str, str] | None:
    parts = _norm(pair).split("_")
    if len(parts) != 2:
        return None
    base, quote = parts
    if base not in _FX_CURRENCIES or quote not in _FX_CURRENCIES:
        return None
    return base, quote


def _currency_exposures(pair: str, direction: str) -> tuple[tuple[str, str], ...]:
    split = _split_fx_pair(pair)
    side = _norm(direction)
    if split is None or side not in {"LONG", "SHORT"}:
        return ()
    base, quote = split
    return ((base, side), (quote, _opposite_side(side)))


def _observable_context_features(raw: dict[str, object]) -> tuple[str, ...]:
    features: set[str] = set()
    if _has_context_value(raw.get("active_headlines")):
        features.add("news_headlines")
    for key in ("event_risk", "regime", "m5_trend", "h1_trend", "entry_type", "session_hour"):
        if _has_context_value(raw.get(key)):
            features.add(key)
    for key in ("dxy", "vix"):
        if _has_context_value(raw.get(key)):
            features.add(key)
    if any("matrix" in str(key).lower() for key in raw):
        features.add("market_context_matrix")
    if any("context_asset" in str(key).lower() for key in raw):
        features.add("context_asset")
    if any(str(key).upper() in {"XAU_USD", "WTICO_USD", "BCO_USD"} for key in raw):
        features.add("non_fx_asset")
    return tuple(sorted(features))


def _has_context_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    text = str(value).strip()
    return bool(text) and text.upper() not in {"NONE", "NULL", "N/A", "UNSPECIFIED"}


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


def _ceil_decimal(value: float, *, places: int) -> float:
    factor = 10**places
    return math.ceil(float(value) * factor) / factor
