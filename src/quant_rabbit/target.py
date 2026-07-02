from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.capital_flows import funding_adjusted_equity, summarize_capital_flows
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.paths import DEFAULT_CAPITAL_FLOWS, DEFAULT_DAILY_TARGET_REPORT, DEFAULT_DAILY_TARGET_STATE
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.snapshot_json import snapshot_payload_order_raw, snapshot_payload_position_raw


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _trader_no_broker_tp_runner(position: BrokerPosition) -> bool:
    return (
        position.owner == Owner.TRADER
        and position.take_profit is None
        and _trader_sl_repair_disabled()
        and not _missing_tp_repair_enabled()
    )


# Rolling target policy constants are product contract values, not tuned market
# thresholds: the top KPI is 4x equity over 30 calendar days. Active-day return
# is reported separately using the cadence-analysis convention of 22 active
# days per 30 calendar days.
ROLLING_30D_POLICY = "ROLLING_30D_4X"
ROLLING_30D_CALENDAR_DAYS = 30
ROLLING_30D_ACTIVE_DAYS = 22
ROLLING_30D_TARGET_MULTIPLIER = 4.0
ROLLING_30D_ON_PACE_TOLERANCE = 0.98
ROLLING_30D_DANGER_DAILY_RETURN_PCT = 10.0


@dataclass(frozen=True)
class TargetPositionRisk:
    trade_id: str
    pair: str
    side: str
    owner: str
    units: int
    unrealized_pl_jpy: float
    protected: bool
    remaining_risk_jpy: float | None
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DailyTargetSnapshot:
    generated_at_utc: str
    start_balance_jpy: float
    target_return_pct: float
    target_jpy: float
    target_profit_jpy: float
    minimum_return_pct: float
    minimum_target_jpy: float
    realized_pl_jpy: float
    unrealized_pl_jpy: float
    account_unrealized_pl_jpy: float
    account_progress_jpy: float
    account_progress_pct: float
    progress_jpy: float
    progress_pct: float
    minimum_progress_pct: float
    remaining_minimum_jpy: float
    remaining_target_jpy: float
    current_equity_jpy: float
    current_equity_raw: float
    capital_flows_30d: float
    capital_flow_count_30d: int
    funding_adjusted_equity: float
    rolling_30d_policy: str
    rolling_30d_start_utc: str
    rolling_30d_end_utc: str
    rolling_30d_elapsed_calendar_days: float
    rolling_30d_remaining_calendar_days: float
    rolling_30d_remaining_active_days: float
    rolling_30d_start_equity: float
    current_equity: float
    rolling_30d_multiplier_raw: float
    rolling_30d_multiplier_funding_adjusted: float
    current_30d_multiplier: float
    remaining_to_4x_raw: float
    remaining_to_4x_funding_adjusted: float
    remaining_to_4x: float
    required_calendar_daily_return_raw: float | None
    required_active_day_return_raw: float | None
    required_calendar_daily_return_funding_adjusted: float | None
    required_active_day_return_funding_adjusted: float | None
    required_calendar_daily_return: float | None
    required_active_day_return: float | None
    performance_basis: str
    sizing_basis: str
    pace_state: str
    capital_flow_issues: tuple[str, ...]
    campaign_day_jst: str
    daily_risk_budget_jpy: float
    daily_risk_pct: float | None
    target_trades_per_day: int
    target_trades_per_day_source: str
    target_trades_per_day_basis_return_pct: float | None
    per_trade_risk_budget_jpy: float
    open_risk_jpy: float
    remaining_risk_budget_jpy: float
    positions: tuple[TargetPositionRisk, ...]
    orders: int
    unprotected_positions: int
    status: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class DailyTargetSummary:
    state_path: Path
    report_path: Path
    status: str
    target_jpy: float
    target_profit_jpy: float
    minimum_target_jpy: float
    progress_jpy: float
    progress_pct: float
    rolling_30d_start_equity: float
    current_equity_raw: float
    capital_flows_30d: float
    funding_adjusted_equity: float
    current_equity: float
    rolling_30d_multiplier_raw: float
    rolling_30d_multiplier_funding_adjusted: float
    current_30d_multiplier: float
    remaining_to_4x_raw: float
    remaining_to_4x_funding_adjusted: float
    remaining_to_4x: float
    required_calendar_daily_return_raw: float | None
    required_active_day_return_raw: float | None
    required_calendar_daily_return_funding_adjusted: float | None
    required_active_day_return_funding_adjusted: float | None
    required_calendar_daily_return: float | None
    required_active_day_return: float | None
    performance_basis: str
    sizing_basis: str
    pace_state: str
    minimum_progress_pct: float
    remaining_minimum_jpy: float
    remaining_target_jpy: float
    remaining_risk_budget_jpy: float
    target_trades_per_day: int
    target_trades_per_day_source: str
    target_trades_per_day_basis_return_pct: float | None
    per_trade_risk_budget_jpy: float
    unprotected_positions: int


@dataclass(frozen=True)
class _BacktestPace:
    trades_per_day: int
    source: str
    basis_return_pct: float | None


_TARGET_SIZING_MULTIPLIER_LIMITS = {
    "NEAR_MISS_SIZE_TEST": 1.1,
    "MODERATE_SIZE_UP_REQUIRED": 1.25,
}


class DailyTargetLedger:
    """Record daily 10% target progress from broker truth and realized PnL."""

    def __init__(
        self,
        *,
        state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        report_path: Path = DEFAULT_DAILY_TARGET_REPORT,
        capital_flows_path: Path = DEFAULT_CAPITAL_FLOWS,
        pace_backtest_path: Path | None = None,
        execution_ledger_path: Path | None = None,
    ) -> None:
        self.state_path = state_path
        self.report_path = report_path
        self.capital_flows_path = capital_flows_path
        self.pace_backtest_path = pace_backtest_path
        self.execution_ledger_path = execution_ledger_path

    def run(
        self,
        *,
        start_balance_jpy: float | None = None,
        target_return_pct: float | None = None,
        target_profit_jpy: float | None = None,
        realized_pl_jpy: float | None = None,
        daily_risk_budget_jpy: float | None = None,
        daily_risk_pct: float | None = None,
        target_trades_per_day: int | None = None,
        snapshot: BrokerSnapshot | None = None,
        snapshot_path: Path | None = None,
        now_utc: datetime | None = None,
    ) -> DailyTargetSummary:
        previous = self._load_previous()
        if snapshot is None and snapshot_path is not None:
            snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text()))

        reference_time = _normalize_utc_now(now_utc)
        campaign_day_jst = _campaign_day_key(reference_time)
        previous_day = _coalesce_campaign_day(previous)
        is_new_campaign_day = previous_day is not None and previous_day != campaign_day_jst
        ledger_realized = (
            None
            if realized_pl_jpy is not None
            else _realized_pl_from_execution_ledger(self.execution_ledger_path, campaign_day_jst)
        )

        # Priority order:
        # 1. Explicit --start-balance argument (caller override).
        # 2. Snapshot.account plus an auditable realized figure (explicit arg or
        #    execution-ledger truth): derive `balance - realized_today` on EVERY
        #    cycle, not only on campaign-day transitions. This makes the start
        #    balance self-healing per §3.5 "no stale persistence": a poisoned
        #    persisted value cannot survive a cycle that has broker truth.
        #    (2026-06-08 incident: start_balance_jpy=222,781 persisted from
        #    ~2026-05-12 while the broker balance was 184,962, so the state
        #    reported -37,818 JPY / -169% "today's" progress even though the
        #    execution ledger truth for the campaign day was +92 JPY.)
        # 3. New campaign day without snapshot.account: roll over previous current_equity.
        # 4. Otherwise reuse previous start_balance.
        snapshot_start_balance = _start_balance_from_snapshot(
            snapshot=snapshot,
            realized_pl_jpy=realized_pl_jpy if realized_pl_jpy is not None else ledger_realized,
        )
        has_audited_realized = realized_pl_jpy is not None or ledger_realized is not None
        start_balance = _coalesce_float(start_balance_jpy)
        if (
            start_balance is None
            and snapshot_start_balance is not None
            and (is_new_campaign_day or has_audited_realized)
        ):
            # Same-day re-derivation requires an audited realized figure;
            # without one, `balance - 0` would wrongly re-anchor the day's
            # target to the current balance and erase visible progress.
            start_balance = snapshot_start_balance
        if start_balance is None and is_new_campaign_day:
            start_balance = _coalesce_float(previous.get("current_equity_jpy"), previous.get("start_balance_jpy"))
        if start_balance is None:
            start_balance = _coalesce_float(previous.get("start_balance_jpy"))
        # First-ever run with no previous state: prefer snapshot-derived value when available.
        if start_balance is None and snapshot_start_balance is not None:
            start_balance = snapshot_start_balance
        if start_balance is None:
            raise ValueError(
                "daily target state requires --start-balance, a previous state file, "
                "or a broker snapshot with account summary on first run"
            )
        explicit_target_profit = _coalesce_float(target_profit_jpy)
        if explicit_target_profit is not None:
            if explicit_target_profit <= 0:
                raise ValueError("daily-target-state --target-profit-jpy must be positive")
            target_pct = (explicit_target_profit / start_balance) * 100.0
        else:
            target_pct = _coalesce_float(target_return_pct, previous.get("target_return_pct"), 10.0)
        if realized_pl_jpy is not None:
            realized = float(realized_pl_jpy)
        elif ledger_realized is not None:
            # Execution-ledger transaction truth wins on every cycle, same-day
            # included. The balance-diff fallback below silently absorbs manual
            # trade P/L, financing postings, deposits, and any historical
            # start-balance poisoning into "today's realized", which both
            # violates feedback_manual_excluded_from_trader_pnl and propagates
            # stale state (§3.5).
            realized = ledger_realized
        elif is_new_campaign_day:
            realized = 0.0
        else:
            realized = (
                _realized_pl_from_snapshot(snapshot=snapshot, start_balance_jpy=start_balance)
                if previous_day is not None
                else None
            )
            if realized is None:
                realized = _coalesce_float(previous.get("realized_pl_jpy"), 0.0)
        # Equity-derived risk budget: per-trade worst-case loss cap is sized to the day's
        # starting equity, not a hardcoded JPY literal. Default uses RiskPolicy.daily_risk_pct
        # (% of starting equity); explicit caller override and previous-day persistence
        # both still win. No silent JPY fallback — if percent and explicit value are both
        # missing, the policy default percent applies but is recorded in the snapshot.
        policy = RiskPolicy()
        # Risk-budget precedence (highest first). NAV%-anchored inputs
        # auto-scale with equity — preferred per feedback_use_nav_percent.md.
        # Absolute JPY paths kept for backward compatibility with smoke
        # scripts and persisted state.
        #   1. CLI --daily-risk-pct (caller wants explicit % of today's NAV)
        #   2. persisted daily_risk_pct from an earlier CLI run — re-applied
        #      to today's NAV so automation cycles stay NAV-anchored without
        #      requiring every cycle to re-pass --daily-risk-pct
        #   3. CLI --daily-risk-budget (caller wants explicit JPY override)
        #   4. previous persisted daily_risk_budget_jpy (carry forward absolute)
        #   5. RiskPolicy.daily_risk_pct (policy default %)
        explicit_pct = _coalesce_float(daily_risk_pct)
        previous_pct = _coalesce_float(previous.get("daily_risk_pct"))
        active_pct: float | None = None
        if explicit_pct is not None and explicit_pct > 0:
            active_pct = explicit_pct
            risk_budget = round(start_balance * (explicit_pct / 100.0), 4)
        elif previous_pct is not None and previous_pct > 0:
            # An earlier CLI invocation set daily_risk_pct. Re-derive from
            # current NAV every cycle so automation honors the operator's
            # NAV% choice without freezing a JPY snapshot.
            active_pct = previous_pct
            risk_budget = round(start_balance * (previous_pct / 100.0), 4)
        else:
            explicit_budget = _coalesce_float(daily_risk_budget_jpy, previous.get("daily_risk_budget_jpy"))
            if explicit_budget is not None:
                risk_budget = explicit_budget
            else:
                if policy.daily_risk_pct is None or policy.daily_risk_pct <= 0:
                    raise ValueError(
                        "daily-target-state: cannot derive daily_risk_budget_jpy. "
                        "Pass --daily-risk-pct or --daily-risk-budget explicitly, "
                        "or set RiskPolicy.daily_risk_pct."
                    )
                risk_budget = round(start_balance * (policy.daily_risk_pct / 100.0), 4)
        # Per-trade risk = day's risk budget / target trade pace. Splitting these
        # keeps a single losing trade from burning the whole day's risk budget,
        # which is what makes "fire many small shots, let winners run" actually
        # behave like that operationally. Per AGENT_CONTRACT §3.5 there is no
        # silent JPY fallback: the trade pace must come from CLI, persisted
        # state, or the documented policy default (RiskPolicy.target_trades_per_day).
        explicit_pace = _coalesce_int(target_trades_per_day)
        pace_source = "cli" if explicit_pace is not None else ""
        pace_basis_return_pct: float | None = None
        if explicit_pace is None:
            previous_pace = _coalesce_int(previous.get("target_trades_per_day"))
            previous_pace_source = str(previous.get("target_trades_per_day_source") or "")
            previous_pace_basis = _coalesce_float(previous.get("target_trades_per_day_basis_return_pct"))
            # Respect a prior CLI-set pace. The operator deliberately chose
            # that pace; the next automation cycle (which calls .run() with
            # no overrides) must not silently revert it via ai_test_bot
            # required-trades — that flips per-trade size back down by an
            # order of magnitude and re-freezes attack-mode entries
            # (regression seen 2026-05-11: pace 10→30, per_trade 1040→346).
            # Detect a prior operator-explicit choice via either the first-hop
            # source ("cli") or the carry-forward marker ("previous_cli").
            # `startswith("cli")` only matches the former, so the protection
            # silently dropped on the next automation cycle and ai_test_bot
            # reclaimed pace=30 (regression seen 2026-05-11 15:46 JST).
            # Preserve operator-explicit pace only within the same campaign day.
            # A new day gets a fresh ai-test-bot firepower read when available,
            # otherwise yesterday's CLI pace can silently become a stale fixed
            # risk divisor and violate the no-default-to-yesterday contract.
            fresh_backtest_after_previous = _backtest_newer_than_previous_state(self.pace_backtest_path, previous)
            previous_was_cli = (
                previous_pace_source == "cli"
                or (previous_pace_source == "previous_cli" and not fresh_backtest_after_previous)
            ) and not is_new_campaign_day
            if previous_pace is not None and previous_was_cli:
                explicit_pace = previous_pace
                pace_source = "previous_cli"
            else:
                backtest_pace = _pace_from_backtest(self.pace_backtest_path)
                if backtest_pace is not None:
                    target_sizing_pace = backtest_pace.source.startswith("ai_test_bot_target_sizing_")
                    explicit_pace = (
                        backtest_pace.trades_per_day
                        if target_sizing_pace
                        else max(backtest_pace.trades_per_day, previous_pace or 0)
                    )
                    pace_source = (
                        backtest_pace.source
                        if explicit_pace == backtest_pace.trades_per_day
                        else "previous_state_above_ai_test_bot"
                    )
                    pace_basis_return_pct = (
                        backtest_pace.basis_return_pct
                        if explicit_pace == backtest_pace.trades_per_day
                        else previous_pace_basis
                    )
                elif previous_pace is not None:
                    explicit_pace = previous_pace
                    pace_source = "previous_state"
                    pace_basis_return_pct = previous_pace_basis
            cap = policy.max_target_trades_per_day
            if (
                explicit_pace is not None
                and cap is not None
                and cap > 0
                and explicit_pace > cap
            ):
                # ai-test-bot.firepower can demand 100+ trades/day when current
                # strategy expectancy is too thin to hit the daily target. That
                # number flowing straight into per_trade_risk_budget_jpy sizes
                # each order into the noise floor; cap to the operator's
                # declared practical maximum so execution stays meaningful.
                # The expectancy gap itself is still surfaced via
                # ai_test_bot.firepower so the operator sees it.
                explicit_pace = int(cap)
                pace_source = (
                    f"{pace_source}_capped"
                    if pace_source
                    else "policy_cap"
                )
        if explicit_pace is None:
            if policy.target_trades_per_day is None or policy.target_trades_per_day <= 0:
                raise ValueError(
                    "daily-target-state: cannot derive target_trades_per_day. "
                    "Pass --target-trades-per-day explicitly, or set "
                    "RiskPolicy.target_trades_per_day."
                )
            explicit_pace = int(policy.target_trades_per_day)
            pace_source = "risk_policy_default"
        per_trade_risk_budget = round(risk_budget / explicit_pace, 4)
        # Per AGENT_CONTRACT §3.5 + feedback_high_conviction_execution.md:
        # if pace × budget drives per-trade below an equity-derived floor,
        # the math has broken — every lane will exceed cap and lock the
        # campaign out. Apply the policy floor (% of starting equity) ONLY
        # when the pace came from automated derivation (backtest / previous
        # state / policy default). Operator-explicit CLI pace is treated as
        # a deliberate override; do not silently mutate it.
        per_trade_floor_applied = False
        # "cli" and "previous_cli" are both operator-explicit pace: the
        # operator's arithmetic persists across automation cycles by design
        # (test_target_trades_per_day_persists_across_runs), so the floor
        # must not silently mutate it on the second cycle either.
        if (
            policy.min_per_trade_risk_pct is not None
            and policy.min_per_trade_risk_pct > 0
            and "cli" not in str(pace_source or "")
        ):
            equity_floor = round(start_balance * (policy.min_per_trade_risk_pct / 100.0), 4)
            if equity_floor > per_trade_risk_budget:
                per_trade_risk_budget = equity_floor
                per_trade_floor_applied = True
                pace_source = (
                    f"{pace_source}_floored_by_min_per_trade_pct"
                    if pace_source
                    else "min_per_trade_pct_floor"
                )

        positions = (
            tuple(_position_risk(position, snapshot.quotes, snapshot.home_conversions) for position in snapshot.positions)
            if snapshot
            else _previous_positions(previous.get("positions"))
        )
        # The daily 10% campaign measures the autonomous trader's progress.
        # Operator-managed manual/tagless positions stay visible in the
        # positions table, but their open P/L must not make the trader look
        # closer to target or further from it.
        unrealized = (
            round(sum(position.unrealized_pl_jpy for position in snapshot.positions if position.owner == Owner.TRADER), 4)
            if snapshot
            else float(previous.get("unrealized_pl_jpy") or 0.0)
        )
        account_unrealized = _account_unrealized_pl(snapshot, previous)
        open_risk = round(
            sum((position.remaining_risk_jpy or 0.0) for position in positions if _counts_against_trader_budget(position)),
            4,
        )
        unprotected = sum(
            1 for position in positions if not position.protected and _counts_against_trader_budget(position)
        )
        progress = round(realized + unrealized, 4)
        target_jpy = round(
            explicit_target_profit
            if explicit_target_profit is not None
            else start_balance * (target_pct / 100.0),
            2,
        )
        # The 10% target remains the campaign objective. The 5% floor is a
        # same-day minimum-progress line requested by the operator; it is
        # derived from the same broker-truth start balance so no JPY literal
        # enters production behavior.
        minimum_pct = min(5.0, target_pct) if target_pct > 0 else 0.0
        minimum_target_jpy = round(start_balance * (minimum_pct / 100.0), 2)
        remaining_target = round(max(0.0, target_jpy - progress), 4)
        remaining_minimum = round(max(0.0, minimum_target_jpy - progress), 4)
        remaining_risk_budget = 0.0 if unprotected else round(max(0.0, risk_budget - open_risk), 4)
        current_equity = _current_equity_jpy(
            snapshot=snapshot,
            start_balance_jpy=start_balance,
            realized_pl_jpy=realized,
            account_unrealized_pl_jpy=account_unrealized,
            fallback_progress_jpy=progress,
            previous=previous,
        )
        rolling_30d = _rolling_30d_policy(
            previous=previous,
            current_equity_jpy=current_equity,
            reference_time=reference_time,
            capital_flows_path=self.capital_flows_path,
        )
        progress_pct = round((progress / target_jpy) * 100.0, 4) if target_jpy else 0.0
        account_progress = round(current_equity - start_balance, 4)
        account_progress_pct = round((account_progress / target_jpy) * 100.0, 4) if target_jpy else 0.0
        minimum_progress_pct = (
            round((progress / minimum_target_jpy) * 100.0, 4)
            if minimum_target_jpy
            else 0.0
        )
        blockers = tuple(_blockers(positions, open_risk=open_risk, risk_budget=risk_budget, remaining_target=remaining_target))
        status = _status(
            progress_jpy=progress,
            target_jpy=target_jpy,
            unprotected_positions=unprotected,
            remaining_risk_budget_jpy=remaining_risk_budget,
        )

        state = DailyTargetSnapshot(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            start_balance_jpy=round(start_balance, 4),
            target_return_pct=round(target_pct, 4),
            target_jpy=target_jpy,
            target_profit_jpy=target_jpy,
            minimum_return_pct=round(minimum_pct, 4),
            minimum_target_jpy=minimum_target_jpy,
            realized_pl_jpy=round(realized, 4),
            unrealized_pl_jpy=unrealized,
            account_unrealized_pl_jpy=account_unrealized,
            account_progress_jpy=account_progress,
            account_progress_pct=account_progress_pct,
            progress_jpy=progress,
            progress_pct=progress_pct,
            minimum_progress_pct=minimum_progress_pct,
            remaining_minimum_jpy=remaining_minimum,
            remaining_target_jpy=remaining_target,
            current_equity_jpy=current_equity,
            current_equity_raw=rolling_30d["current_equity_raw"],
            capital_flows_30d=rolling_30d["capital_flows_30d"],
            capital_flow_count_30d=rolling_30d["capital_flow_count_30d"],
            funding_adjusted_equity=rolling_30d["funding_adjusted_equity"],
            rolling_30d_policy=ROLLING_30D_POLICY,
            rolling_30d_start_utc=rolling_30d["rolling_30d_start_utc"],
            rolling_30d_end_utc=rolling_30d["rolling_30d_end_utc"],
            rolling_30d_elapsed_calendar_days=rolling_30d["rolling_30d_elapsed_calendar_days"],
            rolling_30d_remaining_calendar_days=rolling_30d["rolling_30d_remaining_calendar_days"],
            rolling_30d_remaining_active_days=rolling_30d["rolling_30d_remaining_active_days"],
            rolling_30d_start_equity=rolling_30d["rolling_30d_start_equity"],
            current_equity=rolling_30d["current_equity"],
            rolling_30d_multiplier_raw=rolling_30d["rolling_30d_multiplier_raw"],
            rolling_30d_multiplier_funding_adjusted=rolling_30d["rolling_30d_multiplier_funding_adjusted"],
            current_30d_multiplier=rolling_30d["current_30d_multiplier"],
            remaining_to_4x_raw=rolling_30d["remaining_to_4x_raw"],
            remaining_to_4x_funding_adjusted=rolling_30d["remaining_to_4x_funding_adjusted"],
            remaining_to_4x=rolling_30d["remaining_to_4x"],
            required_calendar_daily_return_raw=rolling_30d["required_calendar_daily_return_raw"],
            required_active_day_return_raw=rolling_30d["required_active_day_return_raw"],
            required_calendar_daily_return_funding_adjusted=rolling_30d[
                "required_calendar_daily_return_funding_adjusted"
            ],
            required_active_day_return_funding_adjusted=rolling_30d[
                "required_active_day_return_funding_adjusted"
            ],
            required_calendar_daily_return=rolling_30d["required_calendar_daily_return"],
            required_active_day_return=rolling_30d["required_active_day_return"],
            performance_basis=rolling_30d["performance_basis"],
            sizing_basis=rolling_30d["sizing_basis"],
            pace_state=rolling_30d["pace_state"],
            capital_flow_issues=rolling_30d["capital_flow_issues"],
            campaign_day_jst=campaign_day_jst,
            daily_risk_budget_jpy=round(risk_budget, 4),
            daily_risk_pct=round(active_pct, 4) if active_pct is not None else None,
            target_trades_per_day=explicit_pace,
            target_trades_per_day_source=pace_source,
            target_trades_per_day_basis_return_pct=round(pace_basis_return_pct, 4)
            if pace_basis_return_pct is not None
            else None,
            per_trade_risk_budget_jpy=per_trade_risk_budget,
            open_risk_jpy=open_risk,
            remaining_risk_budget_jpy=remaining_risk_budget,
            positions=positions,
            orders=len(snapshot.orders) if snapshot else int(previous.get("orders") or 0),
            unprotected_positions=unprotected,
            status=status,
            blockers=blockers,
        )
        self._write_state(state)
        self._write_report(state)
        return DailyTargetSummary(
            state_path=self.state_path,
            report_path=self.report_path,
            status=state.status,
            target_jpy=state.target_jpy,
            target_profit_jpy=state.target_profit_jpy,
            minimum_target_jpy=state.minimum_target_jpy,
            progress_jpy=state.progress_jpy,
            progress_pct=state.progress_pct,
            rolling_30d_start_equity=state.rolling_30d_start_equity,
            current_equity_raw=state.current_equity_raw,
            capital_flows_30d=state.capital_flows_30d,
            funding_adjusted_equity=state.funding_adjusted_equity,
            current_equity=state.current_equity,
            rolling_30d_multiplier_raw=state.rolling_30d_multiplier_raw,
            rolling_30d_multiplier_funding_adjusted=state.rolling_30d_multiplier_funding_adjusted,
            current_30d_multiplier=state.current_30d_multiplier,
            remaining_to_4x_raw=state.remaining_to_4x_raw,
            remaining_to_4x_funding_adjusted=state.remaining_to_4x_funding_adjusted,
            remaining_to_4x=state.remaining_to_4x,
            required_calendar_daily_return_raw=state.required_calendar_daily_return_raw,
            required_active_day_return_raw=state.required_active_day_return_raw,
            required_calendar_daily_return_funding_adjusted=state.required_calendar_daily_return_funding_adjusted,
            required_active_day_return_funding_adjusted=state.required_active_day_return_funding_adjusted,
            required_calendar_daily_return=state.required_calendar_daily_return,
            required_active_day_return=state.required_active_day_return,
            performance_basis=state.performance_basis,
            sizing_basis=state.sizing_basis,
            pace_state=state.pace_state,
            minimum_progress_pct=state.minimum_progress_pct,
            remaining_minimum_jpy=state.remaining_minimum_jpy,
            remaining_target_jpy=state.remaining_target_jpy,
            remaining_risk_budget_jpy=state.remaining_risk_budget_jpy,
            target_trades_per_day=state.target_trades_per_day,
            target_trades_per_day_source=state.target_trades_per_day_source,
            target_trades_per_day_basis_return_pct=state.target_trades_per_day_basis_return_pct,
            per_trade_risk_budget_jpy=state.per_trade_risk_budget_jpy,
            unprotected_positions=state.unprotected_positions,
        )

    def _load_previous(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())

    def _write_state(self, state: DailyTargetSnapshot) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        payload["as_of_utc"] = state.generated_at_utc
        payload["campaign_day"] = state.campaign_day_jst
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, state: DailyTargetSnapshot) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Daily Target Report",
            "",
            f"- Generated at UTC: `{state.generated_at_utc}`",
            f"- Status: `{state.status}`",
            f"- Start equity: `{state.start_balance_jpy:.0f} JPY`",
            f"- Campaign day (JST9): `{state.campaign_day_jst}`",
            f"- Target: `{state.target_jpy:.0f} JPY` (`{state.target_return_pct:.1f}%`)",
            f"- Minimum daily floor: `{state.minimum_target_jpy:.0f} JPY` (`{state.minimum_return_pct:.1f}%`)",
            f"- Realized PnL: `{state.realized_pl_jpy:.0f} JPY`",
            f"- Trader unrealized PnL: `{state.unrealized_pl_jpy:.0f} JPY`",
            f"- Progress: `{state.progress_jpy:.0f} JPY` (`{state.progress_pct:.1f}%` of target)",
            f"- Account unrealized PnL: `{state.account_unrealized_pl_jpy:.0f} JPY` (includes manual/tagless exposure)",
            f"- Account progress: `{state.account_progress_jpy:.0f} JPY` (`{state.account_progress_pct:.1f}%` of target, broker NAV view)",
            "",
            "## Rolling 30D 4X Policy",
            "",
            f"- Policy: `{state.rolling_30d_policy}`",
            f"- Window: `{state.rolling_30d_start_utc}` → `{state.rolling_30d_end_utc}`",
            f"- Rolling 30d start equity: `{state.rolling_30d_start_equity:.0f} JPY`",
            f"- current_equity_raw: `{state.current_equity_raw:.0f} JPY`",
            f"- capital_flows_30d: `{state.capital_flows_30d:.0f} JPY`",
            f"- funding_adjusted_equity: `{state.funding_adjusted_equity:.0f} JPY`",
            f"- rolling_30d_multiplier_raw: `{state.rolling_30d_multiplier_raw:.4f}x`",
            f"- rolling_30d_multiplier_funding_adjusted: `{state.rolling_30d_multiplier_funding_adjusted:.4f}x`",
            f"- remaining_to_4x_raw: `{state.remaining_to_4x_raw:.0f} JPY`",
            f"- remaining_to_4x_funding_adjusted: `{state.remaining_to_4x_funding_adjusted:.0f} JPY`",
            f"- Current 30d multiplier: `{state.current_30d_multiplier:.4f}x` (funding-adjusted)",
            f"- Remaining to 4x: `{state.remaining_to_4x:.0f} JPY` (funding-adjusted)",
            "- required_calendar_daily_return_raw: "
            + (
                f"`{state.required_calendar_daily_return_raw:.4f}%`"
                if state.required_calendar_daily_return_raw is not None
                else "`n/a`"
            ),
            "- required_active_day_return_raw: "
            + (
                f"`{state.required_active_day_return_raw:.4f}%`"
                if state.required_active_day_return_raw is not None
                else "`n/a`"
            ),
            "- required_calendar_daily_return_funding_adjusted: "
            + (
                f"`{state.required_calendar_daily_return_funding_adjusted:.4f}%`"
                if state.required_calendar_daily_return_funding_adjusted is not None
                else "`n/a`"
            ),
            "- required_active_day_return_funding_adjusted: "
            + (
                f"`{state.required_active_day_return_funding_adjusted:.4f}%`"
                if state.required_active_day_return_funding_adjusted is not None
                else "`n/a`"
            ),
            "- Required calendar daily return: "
            + (
                f"`{state.required_calendar_daily_return:.4f}%`"
                if state.required_calendar_daily_return is not None
                else "`n/a`"
            ),
            "- Required active-day return: "
            + (
                f"`{state.required_active_day_return:.4f}%`"
                if state.required_active_day_return is not None
                else "`n/a`"
            ),
            f"- performance_basis: `{state.performance_basis}`",
            f"- sizing_basis: `{state.sizing_basis}`",
            f"- Pace state: `{state.pace_state}`",
            "",
            "## Daily Pace Marker",
            "",
            f"- Minimum-floor progress: `{state.minimum_progress_pct:.1f}%`; remaining floor `{state.remaining_minimum_jpy:.0f} JPY`",
            f"- Remaining target: `{state.remaining_target_jpy:.0f} JPY`",
            f"- Open risk: `{state.open_risk_jpy:.0f} JPY`",
            f"- Remaining risk budget: `{state.remaining_risk_budget_jpy:.0f} JPY`",
            f"- Target trades per day: `{state.target_trades_per_day}` (`{state.target_trades_per_day_source}`)",
            "- Target trade pace basis: "
            + (
                f"`{state.target_trades_per_day_basis_return_pct:.1f}%`"
                if state.target_trades_per_day_basis_return_pct is not None
                else "`n/a`"
            ),
            f"- Per-trade risk cap: `{state.per_trade_risk_budget_jpy:.0f} JPY`",
            f"- Current equity estimate: `{state.current_equity_jpy:.0f} JPY`",
            "",
            "## Blockers",
            "",
        ]
        if state.blockers:
            lines.extend(f"- {blocker}" for blocker in state.blockers)
        else:
            lines.append("- none")
        if state.capital_flow_issues:
            lines.extend(["", "## Capital Flow Issues", ""])
            lines.extend(f"- {issue}" for issue in state.capital_flow_issues)
        lines.extend(["", "## Open Positions", ""])
        if not state.positions:
            lines.append("- none")
        for position in state.positions:
            missing = "/".join(position.missing) if position.missing else "none"
            risk_text = "unknown" if position.remaining_risk_jpy is None else f"{position.remaining_risk_jpy:.0f} JPY"
            lines.append(
                f"- `{position.trade_id}` `{position.pair} {position.side}` owner=`{position.owner}` units=`{position.units}` "
                f"upl=`{position.unrealized_pl_jpy:.0f}` risk=`{risk_text}` missing=`{missing}`"
            )
        lines.extend(
            [
                "",
                "## Target Contract",
                "",
                "- The top KPI is rolling 30-calendar-day 4x equity growth.",
                "- Rolling 30d performance uses funding_adjusted_equity; current_equity_raw remains the broker NAV basis for risk, margin, and sizing.",
                "- Capital flows are recorded as deposits/withdrawals, not trading P/L, and are excluded from funding-adjusted return.",
                "- Backward-compatible required_calendar_daily_return and required_active_day_return are funding-adjusted primary values.",
                "- The +5% daily line is a pace marker, review trigger, and protection milestone; it must not force B/C churn on no-edge days.",
                "- The 10% daily target is extension-only behind the favorable-market gate, not a guaranteed return.",
                "- Unprotected trader-owned or external exposure makes remaining risk budget unavailable; operator-managed manual/tagless exposure is TP-managed only and does not block fresh entries.",
                "- Trader progress excludes operator-managed manual/tagless P/L for risk gating, while account progress shows broker NAV including that exposure.",
                "- Reaching the target switches the system toward protection-first behavior before any new risk is added.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _position_risk(
    position: BrokerPosition,
    quotes: dict[str, Quote],
    home_conversions: dict[str, float] | None = None,
) -> TargetPositionRisk:
    missing = []
    blocking_missing = []
    if position.take_profit is None:
        missing.append("TP")
        if not _trader_no_broker_tp_runner(position):
            blocking_missing.append("TP")
    if position.stop_loss is None:
        # SL-free regime (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
        # 「SLいらない」 / 「損失を出さないで稼ぎまくる」): trader-owned SL=None is
        # intentional. Treat as protected for daily-target accounting so the
        # status stays PURSUE_TARGET and basket entries are not blocked by
        # REPAIR_REQUIRED.
        if not (_trader_sl_repair_disabled() and position.owner == Owner.TRADER):
            missing.append("SL")
            blocking_missing.append("SL")
    remaining_risk = _remaining_risk_jpy(position, quotes, home_conversions or {})
    if position.stop_loss is not None and remaining_risk is None:
        missing.append("JPY_CONVERSION")
        blocking_missing.append("JPY_CONVERSION")
    return TargetPositionRisk(
        trade_id=position.trade_id,
        pair=position.pair,
        side=position.side.value,
        owner=position.owner.value,
        units=position.units,
        unrealized_pl_jpy=round(position.unrealized_pl_jpy, 4),
        protected=not blocking_missing,
        remaining_risk_jpy=remaining_risk,
        missing=tuple(missing),
    )


def _previous_positions(payload: object) -> tuple[TargetPositionRisk, ...]:
    if not isinstance(payload, list):
        return ()
    positions: list[TargetPositionRisk] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        positions.append(
            TargetPositionRisk(
                trade_id=str(item.get("trade_id") or ""),
                pair=str(item.get("pair") or ""),
                side=str(item.get("side") or ""),
                owner=str(item.get("owner") or Owner.UNKNOWN.value),
                units=int(item.get("units") or 0),
                unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
                protected=bool(item.get("protected")),
                remaining_risk_jpy=_optional_float(item.get("remaining_risk_jpy")),
                missing=tuple(str(value) for value in item.get("missing", []) or []),
            )
        )
    return tuple(positions)


def _remaining_risk_jpy(position: BrokerPosition, quotes: dict[str, Quote], home_conversions: dict[str, float]) -> float | None:
    if position.stop_loss is None:
        return None
    factor = _pip_factor(position.pair)
    if position.side == Side.LONG:
        pips = (position.entry_price - position.stop_loss) * factor
    else:
        pips = (position.stop_loss - position.entry_price) * factor
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None:
        return None
    return round(max(0.0, pips) * jpy_per_pip, 4)


def _blockers(
    positions: tuple[TargetPositionRisk, ...],
    *,
    open_risk: float,
    risk_budget: float,
    remaining_target: float,
) -> list[str]:
    blockers: list[str] = []
    for position in positions:
        if not _counts_against_trader_budget(position):
            continue
        if not position.protected:
            blockers.append(
                f"open position {position.trade_id} {position.pair} lacks {'/'.join(position.missing)}; repair before fresh risk"
            )
    if open_risk > risk_budget:
        blockers.append(f"open risk {open_risk:.0f} JPY exceeds daily risk budget {risk_budget:.0f} JPY")
    if remaining_target > 0 and not positions:
        blockers.append(f"remaining target {remaining_target:.0f} JPY still needs live-ready campaign coverage")
    return blockers


def _counts_against_trader_budget(position: TargetPositionRisk) -> bool:
    return position.owner not in {Owner.MANUAL.value, Owner.UNKNOWN.value, Owner.OPERATOR_MANUAL.value}


def _account_unrealized_pl(snapshot: BrokerSnapshot | None, previous: dict[str, Any]) -> float:
    if snapshot is None:
        return float(previous.get("account_unrealized_pl_jpy") or previous.get("unrealized_pl_jpy") or 0.0)
    if snapshot.account is not None:
        return round(float(snapshot.account.unrealized_pl_jpy), 4)
    return round(sum(position.unrealized_pl_jpy for position in snapshot.positions), 4)


def _current_equity_jpy(
    *,
    snapshot: BrokerSnapshot | None,
    start_balance_jpy: float,
    realized_pl_jpy: float,
    account_unrealized_pl_jpy: float,
    fallback_progress_jpy: float,
    previous: dict[str, Any],
) -> float:
    if snapshot is not None and snapshot.account is not None:
        return round(float(snapshot.account.nav_jpy), 4)
    if snapshot is not None:
        return round(start_balance_jpy + realized_pl_jpy + account_unrealized_pl_jpy, 4)
    previous_equity = _coalesce_float(previous.get("current_equity_jpy"))
    if previous_equity is not None:
        return previous_equity
    return round(start_balance_jpy + fallback_progress_jpy, 4)


def _rolling_30d_policy(
    *,
    previous: dict[str, Any],
    current_equity_jpy: float,
    reference_time: datetime,
    capital_flows_path: Path,
) -> dict[str, Any]:
    previous_start = _parse_utc_timestamp(previous.get("rolling_30d_start_utc"))
    previous_equity = _coalesce_float(previous.get("rolling_30d_start_equity"))
    if (
        previous_start is not None
        and previous_equity is not None
        and previous_equity > 0
        and previous_start <= reference_time
        and (reference_time - previous_start) < timedelta(days=ROLLING_30D_CALENDAR_DAYS)
    ):
        start_time = previous_start
        start_equity = previous_equity
    else:
        start_time = reference_time
        start_equity = current_equity_jpy

    end_time = start_time + timedelta(days=ROLLING_30D_CALENDAR_DAYS)
    elapsed_days = max(0.0, (reference_time - start_time).total_seconds() / 86400.0)
    remaining_calendar_days = max(0.0, ROLLING_30D_CALENDAR_DAYS - elapsed_days)
    remaining_active_days = remaining_calendar_days * (
        ROLLING_30D_ACTIVE_DAYS / ROLLING_30D_CALENDAR_DAYS
    )
    target_equity = start_equity * ROLLING_30D_TARGET_MULTIPLIER
    capital_flows = summarize_capital_flows(
        capital_flows_path,
        start_utc=start_time,
        end_utc=reference_time,
    )
    current_equity_raw = round(current_equity_jpy, 4)
    adjusted_equity = funding_adjusted_equity(current_equity_raw, capital_flows.net_amount_jpy)
    multiplier_raw = current_equity_raw / start_equity if start_equity > 0 else 0.0
    multiplier_adjusted = adjusted_equity / start_equity if start_equity > 0 else 0.0
    remaining_to_4x_raw = max(0.0, target_equity - current_equity_raw)
    remaining_to_4x_adjusted = max(0.0, target_equity - adjusted_equity)
    required_calendar_raw = _required_compound_return_pct(
        current_value=current_equity_raw,
        target_value=target_equity,
        remaining_periods=remaining_calendar_days,
    )
    required_active_raw = _required_compound_return_pct(
        current_value=current_equity_raw,
        target_value=target_equity,
        remaining_periods=remaining_active_days,
    )
    required_calendar_adjusted = _required_compound_return_pct(
        current_value=adjusted_equity,
        target_value=target_equity,
        remaining_periods=remaining_calendar_days,
    )
    required_active_adjusted = _required_compound_return_pct(
        current_value=adjusted_equity,
        target_value=target_equity,
        remaining_periods=remaining_active_days,
    )
    expected_multiplier = ROLLING_30D_TARGET_MULTIPLIER ** (
        min(elapsed_days, ROLLING_30D_CALENDAR_DAYS) / ROLLING_30D_CALENDAR_DAYS
    )
    pace_state = _rolling_pace_state(
        current_multiplier=multiplier_adjusted,
        expected_multiplier=expected_multiplier,
        required_calendar_daily_return=required_calendar_adjusted,
        remaining_to_4x=remaining_to_4x_adjusted,
    )
    return {
        "rolling_30d_start_utc": start_time.isoformat(),
        "rolling_30d_end_utc": end_time.isoformat(),
        "rolling_30d_elapsed_calendar_days": round(elapsed_days, 4),
        "rolling_30d_remaining_calendar_days": round(remaining_calendar_days, 4),
        "rolling_30d_remaining_active_days": round(remaining_active_days, 4),
        "rolling_30d_start_equity": round(start_equity, 4),
        "current_equity_raw": current_equity_raw,
        "capital_flows_30d": capital_flows.net_amount_jpy,
        "capital_flow_count_30d": capital_flows.count,
        "funding_adjusted_equity": adjusted_equity,
        "current_equity": adjusted_equity,
        "rolling_30d_multiplier_raw": round(multiplier_raw, 6),
        "rolling_30d_multiplier_funding_adjusted": round(multiplier_adjusted, 6),
        "current_30d_multiplier": round(multiplier_adjusted, 6),
        "remaining_to_4x_raw": round(remaining_to_4x_raw, 4),
        "remaining_to_4x_funding_adjusted": round(remaining_to_4x_adjusted, 4),
        "remaining_to_4x": round(remaining_to_4x_adjusted, 4),
        "required_calendar_daily_return_raw": required_calendar_raw,
        "required_active_day_return_raw": required_active_raw,
        "required_calendar_daily_return_funding_adjusted": required_calendar_adjusted,
        "required_active_day_return_funding_adjusted": required_active_adjusted,
        "required_calendar_daily_return": required_calendar_adjusted,
        "required_active_day_return": required_active_adjusted,
        "performance_basis": "funding_adjusted",
        "sizing_basis": "raw_nav",
        "pace_state": pace_state,
        "capital_flow_issues": capital_flows.issues,
    }


def _required_compound_return_pct(
    *,
    current_value: float,
    target_value: float,
    remaining_periods: float,
) -> float | None:
    if current_value <= 0 or target_value <= 0:
        return None
    if current_value >= target_value:
        return 0.0
    if remaining_periods <= 0:
        return None
    return round(((target_value / current_value) ** (1.0 / remaining_periods) - 1.0) * 100.0, 6)


def _rolling_pace_state(
    *,
    current_multiplier: float,
    expected_multiplier: float,
    required_calendar_daily_return: float | None,
    remaining_to_4x: float,
) -> str:
    if remaining_to_4x <= 0:
        return "AHEAD"
    if current_multiplier >= expected_multiplier:
        return "AHEAD"
    if current_multiplier >= expected_multiplier * ROLLING_30D_ON_PACE_TOLERANCE:
        return "ON_PACE"
    if (
        required_calendar_daily_return is not None
        and required_calendar_daily_return > ROLLING_30D_DANGER_DAILY_RETURN_PCT
    ):
        return "DANGER"
    return "BEHIND"


def _status(
    *,
    progress_jpy: float,
    target_jpy: float,
    unprotected_positions: int,
    remaining_risk_budget_jpy: float,
) -> str:
    if unprotected_positions:
        return "REPAIR_REQUIRED"
    if progress_jpy >= target_jpy:
        return "TARGET_REACHED_PROTECT"
    if remaining_risk_budget_jpy <= 0:
        return "RISK_BUDGET_EXHAUSTED"
    return "PURSUE_TARGET"


def _start_balance_from_snapshot(
    *, snapshot: BrokerSnapshot | None, realized_pl_jpy: float | None
) -> float | None:
    """Today's start balance derived from OANDA broker truth.

    `balance` is cash (excludes unrealized PnL), so on a new campaign day with zero
    realized PnL it equals today's opening cash. When realized PnL is non-zero we
    subtract it so the figure represents the value before today's closed trades.
    """

    if snapshot is None or snapshot.account is None:
        return None
    realized = float(realized_pl_jpy) if realized_pl_jpy is not None else 0.0
    return float(snapshot.account.balance_jpy) - realized


def _realized_pl_from_snapshot(
    *, snapshot: BrokerSnapshot | None, start_balance_jpy: float
) -> float | None:
    """Derive same-day realized P/L from OANDA cash balance.

    OANDA `balance` excludes open P/L, so against a persisted campaign
    start balance it is the broker-truth realized P/L for the day. Without
    this derivation the ledger keeps reporting realized=0 after losing
    closes, which makes the trader continue as if the daily target gap and
    risk budget were still intact.
    """

    if snapshot is None or snapshot.account is None:
        return None
    return float(snapshot.account.balance_jpy) - float(start_balance_jpy)


def _realized_pl_from_execution_ledger(path: Path | None, campaign_day_jst: str) -> float | None:
    """Read closed-trade P/L for the current JST9 campaign day.

    `_campaign_day_key()` is equivalent to the UTC calendar date because the
    campaign resets at 09:00 JST. The execution ledger stores UTC timestamps,
    so filtering by the first 10 chars of `ts_utc` matches the same boundary.

    The figure is authoritative only when the ledger has actually been synced
    during the current campaign day (`sync_state.updated_at_utc`). A ledger
    last synced on an earlier day cannot distinguish "no closes today" from
    "today's closes not yet imported", so it returns None and the caller falls
    back to snapshot/previous-state derivation instead of treating a stale
    zero as broker truth.
    """

    if path is None or not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            sync_row = conn.execute(
                "SELECT updated_at_utc FROM sync_state WHERE key = 'last_oanda_transaction_id'"
            ).fetchone()
            if sync_row is None or not isinstance(sync_row[0], str):
                return None
            try:
                synced_day = _campaign_day_key(datetime.fromisoformat(sync_row[0]))
            except ValueError:
                return None
            if synced_day < campaign_day_jst:
                return None
            row = conn.execute(
                """
                SELECT COALESCE(SUM(COALESCE(realized_pl_jpy, 0.0)), 0.0)
                FROM execution_events
                WHERE event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                  AND substr(ts_utc, 1, 10) = ?
                """,
                (campaign_day_jst,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return 0.0
    return float(row[0] or 0.0)


def _coalesce_float(*values: object) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        return float(value)
    return None


def _coalesce_int(*values: object) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _pace_from_backtest(path: Path | None) -> _BacktestPace | None:
    """Read required daily trade pace from ai-test-bot firepower evidence.

    This is the wiring promised in RiskPolicy's documentation: when observed
    expectancy says the target needs far more attempts than the policy default,
    the ledger records that market/backtest-derived pace instead of carrying a
    stale "10 trades/day" operator default forward. Missing or non-positive
    evidence returns None so the caller can fall back loudly to previous/CLI/
    policy source labels.
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    target_sizing_pace = _pace_from_target_sizing(payload)
    if target_sizing_pace is not None:
        return target_sizing_pace
    target_band_pace = _pace_from_target_band(payload)
    if target_band_pace is not None:
        return target_band_pace
    firepower = payload.get("firepower")
    if not isinstance(firepower, dict):
        return None
    pace = _coalesce_int(firepower.get("required_trades_per_day_at_observed_expectancy"))
    if pace is None:
        return None
    return _BacktestPace(
        trades_per_day=pace,
        source="ai_test_bot_required_trades",
        basis_return_pct=_coalesce_float(payload.get("target_return_pct")),
    )


def _backtest_newer_than_previous_state(path: Path | None, previous: dict[str, Any]) -> bool:
    """Return True when a freshly regenerated backtest should refresh carried pace."""

    if path is None or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    backtest_generated = _parse_utc_timestamp(payload.get("generated_at_utc"))
    previous_generated = _parse_utc_timestamp(previous.get("generated_at_utc"))
    return backtest_generated is not None and previous_generated is not None and backtest_generated > previous_generated


def _pace_from_target_band(payload: dict[str, Any]) -> _BacktestPace | None:
    """Prefer the verified 5-10% target band over the 10% firepower fallback.

    The campaign still records 10% as the stretch objective and 5% as the
    product floor. Pace is different: it should be calibrated to the next
    measurable band. If selected policy reaches 5% but not 6%, the next loop
    should size and pace for 6% coverage instead of thinning every order
    against a currently unreachable 10% firepower number.
    """

    target_band = payload.get("target_band")
    if not isinstance(target_band, dict):
        return None
    bands = [item for item in target_band.get("bands", []) if isinstance(item, dict)]
    if not bands:
        return None

    floor_pct = _coalesce_float(target_band.get("floor_return_pct")) or 5.0
    stretch_pct = _coalesce_float(target_band.get("stretch_return_pct"))
    if stretch_pct is None:
        stretch_pct = _coalesce_float(payload.get("target_return_pct")) or 10.0
    selected_attainable = _coalesce_float(target_band.get("selected_attainable_return_pct"))
    if selected_attainable is None:
        desired_pct = floor_pct
    elif selected_attainable >= stretch_pct:
        desired_pct = stretch_pct
    else:
        desired_pct = min(stretch_pct, selected_attainable + 1.0)

    chosen = _target_band_for_pct(bands, desired_pct)
    if chosen is None:
        return None
    pace = _coalesce_int(chosen.get("required_trades_per_day_at_observed_expectancy"))
    basis_pct = _coalesce_float(chosen.get("return_pct"))
    if pace is None or basis_pct is None:
        return None
    return _BacktestPace(
        trades_per_day=pace,
        source=f"ai_test_bot_target_band_{_pct_source_token(basis_pct)}pct_required_trades",
        basis_return_pct=basis_pct,
    )


def _pace_from_target_sizing(payload: dict[str, Any]) -> _BacktestPace | None:
    target_band = payload.get("target_band")
    if isinstance(target_band, dict) and _coalesce_float(target_band.get("selected_attainable_return_pct")) is not None:
        return None
    mechanism = payload.get("mechanism_ablation")
    if not isinstance(mechanism, dict):
        return None
    sizing = mechanism.get("target_sizing")
    if not isinstance(sizing, dict):
        return None
    if str(sizing.get("status") or "") not in {
        "FLOOR_NEAR_MISS_SIZE_TEST",
        "FLOOR_MODERATE_SIZE_UP_REQUIRED",
    }:
        return None
    bands = [item for item in sizing.get("bands", []) if isinstance(item, dict)]
    floor_pct = _coalesce_float((target_band or {}).get("floor_return_pct")) if isinstance(target_band, dict) else None
    if floor_pct is None:
        floor_pct = 5.0
    floor = _target_band_for_pct(bands, floor_pct)
    if floor is None:
        return None
    floor_status = str(floor.get("status") or "")
    multiplier_limit = _TARGET_SIZING_MULTIPLIER_LIMITS.get(floor_status)
    if multiplier_limit is None:
        return None
    multiplier = _coalesce_float(floor.get("required_size_multiplier"))
    scaled_loss_cap = _coalesce_float(floor.get("scaled_loss_cap_jpy"))
    scaled_target_hits = _coalesce_int(floor.get("scaled_target_hit_days"))
    scaled_max_drawdown = _coalesce_float(floor.get("scaled_max_drawdown_jpy"))
    scaled_worst_day = _coalesce_float(floor.get("scaled_worst_day_jpy"))
    daily_risk_budget = _coalesce_float(payload.get("daily_risk_budget_jpy"))
    if (
        multiplier is None
        or multiplier <= 1.0
        or multiplier > multiplier_limit
        or scaled_loss_cap is None
        or scaled_loss_cap <= 0
        or daily_risk_budget is None
        or daily_risk_budget <= 0
        or scaled_loss_cap > daily_risk_budget
    ):
        return None
    if scaled_target_hits is not None and scaled_target_hits <= 0:
        return None
    if scaled_max_drawdown is not None and scaled_max_drawdown > daily_risk_budget:
        return None
    if scaled_worst_day is not None and scaled_worst_day < -daily_risk_budget:
        return None
    pace = int(daily_risk_budget // scaled_loss_cap)
    if pace <= 0:
        return None
    return _BacktestPace(
        trades_per_day=pace,
        source=f"ai_test_bot_target_sizing_{_pct_source_token(floor_pct)}pct_{_target_sizing_source_token(floor_status)}",
        basis_return_pct=floor_pct,
    )


def _target_sizing_source_token(status: str) -> str:
    if status == "MODERATE_SIZE_UP_REQUIRED":
        return "moderate"
    return "near_miss"


def _target_band_for_pct(bands: list[dict[str, Any]], desired_pct: float) -> dict[str, Any] | None:
    parsed: list[tuple[float, dict[str, Any]]] = []
    for item in bands:
        pct = _coalesce_float(item.get("return_pct"))
        if pct is not None:
            parsed.append((pct, item))
    if not parsed:
        return None
    for pct, item in sorted(parsed, key=lambda row: row[0]):
        if pct + 1e-9 >= desired_pct:
            return item
    return max(parsed, key=lambda row: row[0])[1]


def _pct_source_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _coalesce_campaign_day(payload: dict[str, Any]) -> str | None:
    explicit = payload.get("campaign_day_jst")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    generated = payload.get("generated_at_utc")
    if isinstance(generated, str) and generated.strip():
        try:
            return _campaign_day_key(datetime.fromisoformat(generated))
        except ValueError:
            return None
    return None


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _normalize_utc_now(datetime.fromisoformat(value.strip()))
    except ValueError:
        return None


def _campaign_day_key(value: datetime) -> str:
    jst = _normalize_utc_now(value).astimezone(_jst_timezone())
    return (jst - timedelta(hours=9)).date().isoformat()


def _jst_timezone() -> timezone:
    return timezone(timedelta(hours=9))


def _normalize_utc_now(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _jpy_per_pip(position: BrokerPosition, quotes: dict[str, Quote], home_conversions: dict[str, float]) -> float | None:
    if position.pair.endswith("_JPY"):
        return position.units / 100
    quote_ccy = position.pair.split("_", 1)[1]
    home_conversion = home_conversions.get(quote_ccy)
    if home_conversion is not None and home_conversion > 0:
        return (position.units / _pip_factor(position.pair)) * float(home_conversion)
    conversion_quote = quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is None:
        return None
    return (position.units / _pip_factor(position.pair)) * max(conversion_quote.bid, conversion_quote.ask)


def _snapshot_from_json(payload: dict[str, Any]) -> BrokerSnapshot:
    positions = tuple(
        BrokerPosition(
            trade_id=str(item["trade_id"]),
            pair=str(item["pair"]),
            side=Side.parse(str(item["side"])),
            units=int(item["units"]),
            entry_price=float(item["entry_price"]),
            unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
            take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
            stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            raw=snapshot_payload_position_raw(item),
        )
        for item in payload.get("positions", []) or []
    )
    orders = tuple(
        BrokerOrder(
            order_id=str(item["order_id"]),
            pair=item.get("pair"),
            order_type=str(item.get("order_type") or ""),
            trade_id=item.get("trade_id"),
            price=float(item["price"]) if item.get("price") is not None else None,
            state=item.get("state"),
            units=int(item["units"]) if item.get("units") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            raw=snapshot_payload_order_raw(item),
        )
        for item in payload.get("orders", []) or []
    )
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        ts = item.get("timestamp_utc")
        quotes[pair] = Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
        )
    fetched = payload.get("fetched_at_utc")
    account = _account_summary_from_payload(payload.get("account"))
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=positions,
        orders=orders,
        quotes=quotes,
        account=account,
        home_conversions={str(k).upper(): float(v) for k, v in (payload.get("home_conversions") or {}).items()},
    )


def _account_summary_from_payload(payload: object) -> AccountSummary | None:
    if not isinstance(payload, dict):
        return None
    fetched = payload.get("fetched_at_utc")
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        hedging_enabled=bool(payload.get("hedging_enabled") or False),
        fetched_at_utc=(
            datetime.fromisoformat(fetched) if isinstance(fetched, str) and fetched else datetime.now(timezone.utc)
        ),
    )
