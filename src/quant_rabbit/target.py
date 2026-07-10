from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.capital_flows import funding_adjusted_equity, summarize_capital_flows
from quant_rabbit.execution_ledger import OANDA_TRANSACTION_COVERAGE_START_KEY
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
# OANDA cash/financing fields are persisted to four decimal JPY precision.
# One cent of JPY absorbs only representation/aggregation roundoff; any larger
# gap means the account-wide delta is incomplete and must fail closed.
ACCOUNT_CASH_RECONCILIATION_TOLERANCE_JPY = 0.01


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
    campaign_open_balance_status: str
    campaign_open_balance_source: str
    campaign_open_balance_audited_candidate_jpy: float | None
    campaign_day_account_realized_jpy: float | None
    campaign_day_capital_flows_jpy: float
    campaign_day_broker_capital_flows_jpy: float | None
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
    realized_loss_spent_jpy: float
    daily_loss_capacity_before_open_jpy: float
    daily_risk_pct: float | None
    target_trades_per_day: int
    target_trades_per_day_source: str
    target_trades_per_day_basis_return_pct: float | None
    uncapped_required_trades_per_day: int | None
    uncapped_required_trades_per_day_basis_return_pct: float | None
    selected_basis_uncapped_required_trades_per_day: int | None
    selected_basis_return_pct: float | None
    operating_pace_trades_per_day: int
    automated_operating_cap_trades_per_day: int | None
    observed_trades_per_day: float | None
    observed_expectancy_jpy_per_trade: float | None
    frequency_multiple_required: float | None
    planned_reward_at_operating_pace_jpy: float | None
    stretch_required_minus_operating_gap_trades_per_day: int | None
    selected_required_minus_operating_gap_trades_per_day: int | None
    trade_pace_feasible_within_operating_pace: bool | None
    trade_pace_feasibility: str
    sizing_nav_jpy: float
    base_per_trade_risk_budget_jpy: float
    per_trade_risk_budget_jpy: float
    per_trade_risk_pct_nav: float
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
    realized_loss_spent_jpy: float
    daily_loss_capacity_before_open_jpy: float
    target_trades_per_day: int
    target_trades_per_day_source: str
    target_trades_per_day_basis_return_pct: float | None
    uncapped_required_trades_per_day: int | None
    uncapped_required_trades_per_day_basis_return_pct: float | None
    selected_basis_uncapped_required_trades_per_day: int | None
    selected_basis_return_pct: float | None
    operating_pace_trades_per_day: int
    automated_operating_cap_trades_per_day: int | None
    observed_trades_per_day: float | None
    observed_expectancy_jpy_per_trade: float | None
    frequency_multiple_required: float | None
    planned_reward_at_operating_pace_jpy: float | None
    stretch_required_minus_operating_gap_trades_per_day: int | None
    selected_required_minus_operating_gap_trades_per_day: int | None
    trade_pace_feasible_within_operating_pace: bool | None
    trade_pace_feasibility: str
    sizing_nav_jpy: float
    base_per_trade_risk_budget_jpy: float
    per_trade_risk_budget_jpy: float
    per_trade_risk_pct_nav: float
    unprotected_positions: int


@dataclass(frozen=True)
class _BacktestPace:
    trades_per_day: int
    source: str
    basis_return_pct: float | None


@dataclass(frozen=True)
class _BacktestPaceEvidence:
    """One atomic read of sizing pace plus uncapped firepower visibility."""

    selected_pace: _BacktestPace | None
    selected_basis_uncapped_pace: _BacktestPace | None
    generated_at_utc: str | None
    uncapped_required_trades_per_day: int | None
    uncapped_required_trades_per_day_basis_return_pct: float | None
    observed_trades_per_day: float | None
    observed_expectancy_jpy_per_trade: float | None
    frequency_multiple_required: float | None


@dataclass(frozen=True)
class _AttributedRealized:
    net_jpy: float
    gross_loss_spent_jpy: float
    account_net_jpy: float
    account_capital_flows_jpy: float


@dataclass(frozen=True)
class _CampaignOpenBalance:
    value_jpy: float
    status: str
    source: str
    audited_candidate_jpy: float | None
    blocker: str | None


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
        ledger_realized = _attributed_realized_from_execution_ledger(
            self.execution_ledger_path,
            campaign_day_jst,
            broker_last_transaction_id=(
                snapshot.account.last_transaction_id
                if snapshot is not None and snapshot.account is not None
                else None
            ),
        )
        campaign_day_start_utc = datetime.fromisoformat(
            f"{campaign_day_jst}T00:00:00+00:00"
        )
        campaign_day_capital_flows = summarize_capital_flows(
            self.capital_flows_path,
            start_utc=campaign_day_start_utc,
            end_utc=reference_time,
        )
        capital_flows_reconciled = (
            ledger_realized is not None
            and not campaign_day_capital_flows.issues
            and abs(
                ledger_realized.account_capital_flows_jpy
                - campaign_day_capital_flows.net_amount_jpy
            )
            <= ACCOUNT_CASH_RECONCILIATION_TOLERANCE_JPY
        )
        account_delta_blocker: str | None = None
        if (
            snapshot is not None
            and snapshot.account is not None
            and self.execution_ledger_path is not None
        ):
            if ledger_realized is None:
                account_delta_blocker = (
                    "CAMPAIGN_ACCOUNT_DELTA_AUDIT_UNAVAILABLE: execution-ledger coverage, "
                    "transaction alignment, financing attribution, or supported cash-delta "
                    "evidence is incomplete; fresh-entry capacity stays zero"
                )
            elif campaign_day_capital_flows.issues:
                account_delta_blocker = (
                    "CAMPAIGN_CAPITAL_FLOW_AUDIT_UNAVAILABLE: capital-flow evidence is invalid; "
                    "fresh-entry capacity stays zero"
                )
            elif not capital_flows_reconciled:
                account_delta_blocker = (
                    "CAMPAIGN_CAPITAL_FLOW_MISMATCH: broker TRANSFER_FUNDS total "
                    f"{ledger_realized.account_capital_flows_jpy:.4f} JPY differs from recorded "
                    f"capital flows {campaign_day_capital_flows.net_amount_jpy:.4f} JPY; "
                    "fresh-entry capacity stays zero"
                )
        audited_open_candidate = (
            round(
                float(snapshot.account.balance_jpy)
                - ledger_realized.account_net_jpy
                - campaign_day_capital_flows.net_amount_jpy,
                4,
            )
            if snapshot is not None
            and snapshot.account is not None
            and ledger_realized is not None
            and capital_flows_reconciled
            else None
        )
        campaign_open = _resolve_campaign_open_balance(
            explicit_start_balance_jpy=_coalesce_float(start_balance_jpy),
            previous=previous,
            previous_day=previous_day,
            campaign_day_jst=campaign_day_jst,
            snapshot=snapshot,
            audited_candidate_jpy=audited_open_candidate,
        )
        start_balance = campaign_open.value_jpy
        # Sizing is an instantaneous portfolio decision, so its percentage
        # denominator must be the latest broker NAV rather than the campaign
        # day's opening balance. The opening balance remains the denominator
        # for the whole-day loss budget so losses cannot reset that allowance.
        # Keeping both bases explicit fixes the former `sizing_basis=raw_nav`
        # mismatch where the 1% floor was actually 1% of day-open equity.
        sizing_nav = _sizing_nav_jpy(
            snapshot=snapshot,
            previous=previous,
            fallback_jpy=start_balance,
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
            realized = ledger_realized.net_jpy
        elif is_new_campaign_day:
            realized = 0.0
        elif campaign_open.blocker is not None or account_delta_blocker is not None:
            # A balance difference is account-wide, not trader-attributed: it
            # can contain manual P/L, daily financing, fees, and funding.  If
            # the complete transaction delta is unavailable, preserve the last
            # reported system value for visibility but never relabel the new
            # cash-balance movement as autonomous trader progress.  The opening
            # audit blocker below keeps fresh-entry capacity at zero.
            realized = _coalesce_float(previous.get("realized_pl_jpy"), 0.0)
        else:
            realized = (
                _realized_pl_from_snapshot(snapshot=snapshot, start_balance_jpy=start_balance)
                if previous_day is not None
                else None
            )
            if realized is None:
                realized = _coalesce_float(previous.get("realized_pl_jpy"), 0.0)
        previous_loss_spent = (
            0.0
            if is_new_campaign_day
            else float(previous.get("realized_loss_spent_jpy") or 0.0)
        )
        if ledger_realized is not None:
            # Broker events are authoritative when attribution is available.
            # Sum each losing close/reduction independently so a later winner
            # cannot refill risk already spent earlier in the campaign day.
            realized_loss_spent = max(
                previous_loss_spent,
                ledger_realized.gross_loss_spent_jpy,
            )
        else:
            # A net-only explicit/snapshot value cannot reconstruct losing and
            # winning partials. Preserve the same-day high-water mark and add
            # any currently visible net loss rather than silently refilling it.
            realized_loss_spent = max(previous_loss_spent, max(0.0, -float(realized)))
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
            # An earlier CLI invocation set daily_risk_pct. Re-derive from the
            # campaign opening balance every cycle so automation honors the
            # operator's percentage without letting a loss reset the day cap.
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
        # Read the source packet once even when CLI/previous pace wins. The
        # uncapped expectancy requirement is advisory visibility, so hiding it
        # behind pace-selection precedence would recreate the capped-30-only
        # report that made a 173-trade stretch requirement invisible.
        backtest_pace_evidence = _pace_evidence_from_backtest(self.pace_backtest_path)
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
            fresh_backtest_after_previous = _backtest_evidence_newer_than_previous_state(
                backtest_pace_evidence,
                previous,
            )
            previous_was_cli = (
                previous_pace_source == "cli"
                or (previous_pace_source == "previous_cli" and not fresh_backtest_after_previous)
            ) and not is_new_campaign_day
            if previous_pace is not None and previous_was_cli:
                explicit_pace = previous_pace
                pace_source = "previous_cli"
            else:
                backtest_pace = backtest_pace_evidence.selected_pace
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
        selected_basis_uncapped_required_trades_per_day = (
            backtest_pace_evidence.selected_basis_uncapped_pace.trades_per_day
            if backtest_pace_evidence.selected_basis_uncapped_pace is not None
            else None
        )
        selected_basis_return_pct = (
            backtest_pace_evidence.selected_basis_uncapped_pace.basis_return_pct
            if backtest_pace_evidence.selected_basis_uncapped_pace is not None
            else None
        )
        automated_operating_cap_trades_per_day = (
            int(policy.max_target_trades_per_day)
            if policy.max_target_trades_per_day is not None
            and policy.max_target_trades_per_day > 0
            else None
        )
        stretch_gap = (
            max(0, backtest_pace_evidence.uncapped_required_trades_per_day - explicit_pace)
            if backtest_pace_evidence.uncapped_required_trades_per_day is not None
            else None
        )
        selected_gap = (
            max(0, selected_basis_uncapped_required_trades_per_day - explicit_pace)
            if selected_basis_uncapped_required_trades_per_day is not None
            else None
        )
        planned_reward_at_operating_pace_jpy = (
            round(backtest_pace_evidence.observed_expectancy_jpy_per_trade * explicit_pace, 4)
            if backtest_pace_evidence.observed_expectancy_jpy_per_trade is not None
            else None
        )
        pace_feasible, pace_feasibility = _trade_pace_feasibility(
            required_trades_per_day=backtest_pace_evidence.uncapped_required_trades_per_day,
            operating_pace_trades_per_day=explicit_pace,
            observed_trades_per_day=backtest_pace_evidence.observed_trades_per_day,
            observed_expectancy_jpy_per_trade=(
                backtest_pace_evidence.observed_expectancy_jpy_per_trade
            ),
        )
        per_trade_risk_budget = round(risk_budget / explicit_pace, 4)
        # Per AGENT_CONTRACT §3.5 + feedback_high_conviction_execution.md:
        # if pace × budget drives per-trade below an equity-derived floor,
        # the math has broken — every lane will exceed cap and lock the
        # campaign out. Apply the policy floor (% of current broker NAV) ONLY
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
            equity_floor = round(sizing_nav * (policy.min_per_trade_risk_pct / 100.0), 4)
            if equity_floor > per_trade_risk_budget:
                per_trade_risk_budget = equity_floor
                per_trade_floor_applied = True
                pace_source = (
                    f"{pace_source}_floored_by_min_per_trade_pct"
                    if pace_source
                    else "min_per_trade_pct_floor"
                )
        base_per_trade_risk_budget = per_trade_risk_budget
        daily_loss_capacity_before_open = round(
            max(0.0, risk_budget - realized_loss_spent),
            4,
        )
        if campaign_open.blocker is not None or account_delta_blocker is not None:
            # An unverified/mismatched opening balance cannot authorize a JPY
            # loss allowance.  Keep the provisional target visible for
            # first-run compatibility, but fail closed on fresh-entry capacity
            # until the account-wide delta proves the campaign anchor.
            daily_loss_capacity_before_open = 0.0
        per_trade_risk_budget = round(
            min(base_per_trade_risk_budget, daily_loss_capacity_before_open),
            4,
        )
        per_trade_risk_pct_nav = round(
            (per_trade_risk_budget / sizing_nav) * 100.0,
            6,
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
        remaining_risk_budget = (
            0.0
            if unprotected
            else round(max(0.0, daily_loss_capacity_before_open - open_risk), 4)
        )
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
        blocker_list = _blockers(
            positions,
            open_risk=open_risk,
            risk_budget=daily_loss_capacity_before_open,
            remaining_target=remaining_target,
        )
        if campaign_open.blocker is not None:
            blocker_list.insert(0, campaign_open.blocker)
        if account_delta_blocker is not None:
            blocker_list.insert(0, account_delta_blocker)
        blockers = tuple(blocker_list)
        status = _status(
            progress_jpy=progress,
            target_jpy=target_jpy,
            unprotected_positions=unprotected,
            remaining_risk_budget_jpy=remaining_risk_budget,
        )

        state = DailyTargetSnapshot(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            start_balance_jpy=round(start_balance, 4),
            campaign_open_balance_status=campaign_open.status,
            campaign_open_balance_source=campaign_open.source,
            campaign_open_balance_audited_candidate_jpy=(
                round(campaign_open.audited_candidate_jpy, 4)
                if campaign_open.audited_candidate_jpy is not None
                else None
            ),
            campaign_day_account_realized_jpy=(
                round(ledger_realized.account_net_jpy, 4)
                if ledger_realized is not None
                else None
            ),
            campaign_day_capital_flows_jpy=round(
                campaign_day_capital_flows.net_amount_jpy,
                4,
            ),
            campaign_day_broker_capital_flows_jpy=(
                round(ledger_realized.account_capital_flows_jpy, 4)
                if ledger_realized is not None
                else None
            ),
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
            realized_loss_spent_jpy=round(realized_loss_spent, 4),
            daily_loss_capacity_before_open_jpy=daily_loss_capacity_before_open,
            daily_risk_pct=round(active_pct, 4) if active_pct is not None else None,
            target_trades_per_day=explicit_pace,
            target_trades_per_day_source=pace_source,
            target_trades_per_day_basis_return_pct=round(pace_basis_return_pct, 4)
            if pace_basis_return_pct is not None
            else None,
            uncapped_required_trades_per_day=(
                backtest_pace_evidence.uncapped_required_trades_per_day
            ),
            uncapped_required_trades_per_day_basis_return_pct=(
                round(backtest_pace_evidence.uncapped_required_trades_per_day_basis_return_pct, 4)
                if backtest_pace_evidence.uncapped_required_trades_per_day_basis_return_pct
                is not None
                else None
            ),
            selected_basis_uncapped_required_trades_per_day=(
                selected_basis_uncapped_required_trades_per_day
            ),
            selected_basis_return_pct=(
                round(selected_basis_return_pct, 4)
                if selected_basis_return_pct is not None
                else None
            ),
            operating_pace_trades_per_day=explicit_pace,
            automated_operating_cap_trades_per_day=automated_operating_cap_trades_per_day,
            observed_trades_per_day=(
                round(backtest_pace_evidence.observed_trades_per_day, 4)
                if backtest_pace_evidence.observed_trades_per_day is not None
                else None
            ),
            observed_expectancy_jpy_per_trade=(
                round(backtest_pace_evidence.observed_expectancy_jpy_per_trade, 4)
                if backtest_pace_evidence.observed_expectancy_jpy_per_trade is not None
                else None
            ),
            frequency_multiple_required=(
                round(backtest_pace_evidence.frequency_multiple_required, 4)
                if backtest_pace_evidence.frequency_multiple_required is not None
                else None
            ),
            planned_reward_at_operating_pace_jpy=planned_reward_at_operating_pace_jpy,
            stretch_required_minus_operating_gap_trades_per_day=stretch_gap,
            selected_required_minus_operating_gap_trades_per_day=selected_gap,
            trade_pace_feasible_within_operating_pace=pace_feasible,
            trade_pace_feasibility=pace_feasibility,
            sizing_nav_jpy=round(sizing_nav, 4),
            base_per_trade_risk_budget_jpy=base_per_trade_risk_budget,
            per_trade_risk_budget_jpy=per_trade_risk_budget,
            per_trade_risk_pct_nav=per_trade_risk_pct_nav,
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
            realized_loss_spent_jpy=state.realized_loss_spent_jpy,
            daily_loss_capacity_before_open_jpy=state.daily_loss_capacity_before_open_jpy,
            target_trades_per_day=state.target_trades_per_day,
            target_trades_per_day_source=state.target_trades_per_day_source,
            target_trades_per_day_basis_return_pct=state.target_trades_per_day_basis_return_pct,
            uncapped_required_trades_per_day=state.uncapped_required_trades_per_day,
            uncapped_required_trades_per_day_basis_return_pct=(
                state.uncapped_required_trades_per_day_basis_return_pct
            ),
            selected_basis_uncapped_required_trades_per_day=(
                state.selected_basis_uncapped_required_trades_per_day
            ),
            selected_basis_return_pct=state.selected_basis_return_pct,
            operating_pace_trades_per_day=state.operating_pace_trades_per_day,
            automated_operating_cap_trades_per_day=(
                state.automated_operating_cap_trades_per_day
            ),
            observed_trades_per_day=state.observed_trades_per_day,
            observed_expectancy_jpy_per_trade=state.observed_expectancy_jpy_per_trade,
            frequency_multiple_required=state.frequency_multiple_required,
            planned_reward_at_operating_pace_jpy=state.planned_reward_at_operating_pace_jpy,
            stretch_required_minus_operating_gap_trades_per_day=(
                state.stretch_required_minus_operating_gap_trades_per_day
            ),
            selected_required_minus_operating_gap_trades_per_day=(
                state.selected_required_minus_operating_gap_trades_per_day
            ),
            trade_pace_feasible_within_operating_pace=(
                state.trade_pace_feasible_within_operating_pace
            ),
            trade_pace_feasibility=state.trade_pace_feasibility,
            sizing_nav_jpy=state.sizing_nav_jpy,
            base_per_trade_risk_budget_jpy=state.base_per_trade_risk_budget_jpy,
            per_trade_risk_budget_jpy=state.per_trade_risk_budget_jpy,
            per_trade_risk_pct_nav=state.per_trade_risk_pct_nav,
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
            f"- Campaign opening audit: `{state.campaign_open_balance_status}` "
            f"(`{state.campaign_open_balance_source}`)",
            "- Campaign opening reconstruction: "
            + (
                f"account realized/financing `{state.campaign_day_account_realized_jpy:.0f} JPY`, "
                f"recorded capital flows `{state.campaign_day_capital_flows_jpy:.0f} JPY`, "
                f"broker capital flows `{state.campaign_day_broker_capital_flows_jpy:.0f} JPY`, "
                f"candidate `{state.campaign_open_balance_audited_candidate_jpy:.0f} JPY`"
                if state.campaign_day_account_realized_jpy is not None
                and state.campaign_day_broker_capital_flows_jpy is not None
                and state.campaign_open_balance_audited_candidate_jpy is not None
                else "`pending account-wide delta evidence`"
            ),
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
            f"- Gross realized loss spent: `{state.realized_loss_spent_jpy:.0f} JPY` (wins do not refill it)",
            f"- Loss capacity before open risk: `{state.daily_loss_capacity_before_open_jpy:.0f} JPY`",
            f"- Open risk: `{state.open_risk_jpy:.0f} JPY`",
            f"- Remaining risk budget: `{state.remaining_risk_budget_jpy:.0f} JPY`",
            f"- Target trades per day: `{state.target_trades_per_day}` (`{state.target_trades_per_day_source}`)",
            "- Target trade pace basis: "
            + (
                f"`{state.target_trades_per_day_basis_return_pct:.1f}%`"
                if state.target_trades_per_day_basis_return_pct is not None
                else "`n/a`"
            ),
            f"- Operating pace: `{state.operating_pace_trades_per_day} trades/day`",
            "- Automated operating cap: "
            + (
                f"`{state.automated_operating_cap_trades_per_day} trades/day`"
                if state.automated_operating_cap_trades_per_day is not None
                else "`n/a`"
            ),
            "- Stretch uncapped required pace: "
            + (
                f"`{state.uncapped_required_trades_per_day} trades/day` "
                f"(basis `{state.uncapped_required_trades_per_day_basis_return_pct:.1f}%`)"
                if state.uncapped_required_trades_per_day is not None
                and state.uncapped_required_trades_per_day_basis_return_pct is not None
                else "`n/a`"
            ),
            "- Selected-basis uncapped required pace: "
            + (
                f"`{state.selected_basis_uncapped_required_trades_per_day} trades/day` "
                f"(basis `{state.selected_basis_return_pct:.1f}%`)"
                if state.selected_basis_uncapped_required_trades_per_day is not None
                and state.selected_basis_return_pct is not None
                else "`n/a`"
            ),
            "- Observed selected pace: "
            + (
                f"`{state.observed_trades_per_day:.4f} trades/day`"
                if state.observed_trades_per_day is not None
                else "`n/a`"
            ),
            "- Observed expectancy: "
            + (
                f"`{state.observed_expectancy_jpy_per_trade:+.4f} JPY/trade`"
                if state.observed_expectancy_jpy_per_trade is not None
                else "`n/a`"
            ),
            "- Required frequency multiple vs observed: "
            + (
                f"`{state.frequency_multiple_required:.4f}x`"
                if state.frequency_multiple_required is not None
                else "`n/a`"
            ),
            "- Planned reward at operating pace: "
            + (
                f"`{state.planned_reward_at_operating_pace_jpy:.4f} JPY/day`"
                if state.planned_reward_at_operating_pace_jpy is not None
                else "`n/a`"
            ),
            "- Stretch required-minus-operating gap: "
            + (
                f"`{state.stretch_required_minus_operating_gap_trades_per_day} trades/day`"
                if state.stretch_required_minus_operating_gap_trades_per_day is not None
                else "`n/a`"
            ),
            "- Selected required-minus-operating gap: "
            + (
                f"`{state.selected_required_minus_operating_gap_trades_per_day} trades/day`"
                if state.selected_required_minus_operating_gap_trades_per_day is not None
                else "`n/a`"
            ),
            f"- Trade pace feasibility: `{state.trade_pace_feasibility}` "
            f"(within operating pace=`{state.trade_pace_feasible_within_operating_pace}`)",
            f"- Sizing NAV: `{state.sizing_nav_jpy:.0f} JPY` (latest broker raw NAV)",
            f"- Base per-trade risk cap: `{state.base_per_trade_risk_budget_jpy:.0f} JPY`",
            f"- Per-trade risk cap: `{state.per_trade_risk_budget_jpy:.0f} JPY` "
            f"(`{state.per_trade_risk_pct_nav:.4f}%` of sizing NAV)",
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


def _sizing_nav_jpy(
    *,
    snapshot: BrokerSnapshot | None,
    previous: dict[str, Any],
    fallback_jpy: float,
) -> float:
    """Return the latest positive raw NAV used as the sizing denominator."""

    if snapshot is not None and snapshot.account is not None:
        nav = float(snapshot.account.nav_jpy)
        if nav > 0.0:
            return nav
    previous_nav = _coalesce_float(
        previous.get("current_equity_jpy"),
        previous.get("current_equity_raw"),
    )
    if previous_nav is not None and previous_nav > 0.0:
        return previous_nav
    return float(fallback_jpy)


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


def _resolve_campaign_open_balance(
    *,
    explicit_start_balance_jpy: float | None,
    previous: dict[str, Any],
    previous_day: str | None,
    campaign_day_jst: str,
    snapshot: BrokerSnapshot | None,
    audited_candidate_jpy: float | None,
) -> _CampaignOpenBalance:
    """Resolve one immutable campaign-day opening balance.

    A same-day broker balance includes every later system/manual close,
    financing posting, and funding flow.  It must therefore never replace an
    already fixed opening balance.  A new day may be reconstructed only from a
    complete account-wide realized/financing delta plus the recorded capital
    flows.  Before that evidence is available, preserve first-run compatibility
    with a provisional value but publish a hard audit blocker; the next
    post-ledger-sync refresh may replace that provisional value exactly once.
    """

    if explicit_start_balance_jpy is not None:
        return _CampaignOpenBalance(
            value_jpy=float(explicit_start_balance_jpy),
            status="EXPLICIT",
            source="explicit_start_balance",
            audited_candidate_jpy=audited_candidate_jpy,
            blocker=None,
        )

    previous_start = _coalesce_float(previous.get("start_balance_jpy"))
    previous_status = str(previous.get("campaign_open_balance_status") or "").upper()
    previous_source = str(previous.get("campaign_open_balance_source") or "")
    same_campaign_day = previous_day == campaign_day_jst
    previous_is_provisional = previous_status == "PROVISIONAL_UNAUDITED"
    previous_is_explicit = previous_status == "EXPLICIT" or previous_source == "explicit_start_balance"

    # A legacy state without a campaign-day marker is preserved rather than
    # guessed from today's balance.  Once account-wide evidence exists, compare
    # it for audit, but do not silently mutate the legacy opening.
    fixed_previous = previous_start is not None and (
        (same_campaign_day and not previous_is_provisional)
        or previous_day is None
    )
    if fixed_previous:
        if (
            audited_candidate_jpy is not None
            and round(previous_start, 4) != round(audited_candidate_jpy, 4)
        ):
            return _CampaignOpenBalance(
                value_jpy=previous_start,
                status="AUDITED_MISMATCH",
                source="same_campaign_day_previous_preserved",
                audited_candidate_jpy=audited_candidate_jpy,
                blocker=(
                    "CAMPAIGN_OPEN_BALANCE_MISMATCH: persisted same-day opening "
                    f"{previous_start:.4f} JPY differs from account-wide reconstructed "
                    f"opening {audited_candidate_jpy:.4f} JPY; explicit repair is required"
                ),
            )
        if previous_is_explicit:
            return _CampaignOpenBalance(
                value_jpy=previous_start,
                status="EXPLICIT",
                source="same_campaign_day_previous_explicit",
                audited_candidate_jpy=audited_candidate_jpy,
                blocker=None,
            )
        if audited_candidate_jpy is not None:
            return _CampaignOpenBalance(
                value_jpy=previous_start,
                status="AUDITED_ACCOUNT_DELTA",
                source="same_campaign_day_previous_verified",
                audited_candidate_jpy=audited_candidate_jpy,
                blocker=None,
            )
        if previous_status == "AUDITED_MISMATCH":
            return _CampaignOpenBalance(
                value_jpy=previous_start,
                status="AUDITED_MISMATCH",
                source=previous_source or "same_campaign_day_previous_preserved",
                audited_candidate_jpy=None,
                blocker=(
                    "CAMPAIGN_OPEN_BALANCE_MISMATCH: the persisted opening remains under "
                    "explicit repair; current account-wide delta evidence is unavailable"
                ),
            )
        return _CampaignOpenBalance(
            value_jpy=previous_start,
            status="AUDIT_UNAVAILABLE",
            source="same_campaign_day_previous_preserved_pending_account_delta",
            audited_candidate_jpy=None,
            blocker=(
                "CAMPAIGN_OPEN_BALANCE_AUDIT_UNAVAILABLE: persisted same-day opening "
                "was preserved, but complete current account-wide delta evidence is unavailable; "
                "fresh-entry capacity stays zero"
            ),
        )

    # This is the only implicit path allowed to establish or replace an
    # opening balance: a new day, first run, or a same-day provisional state.
    if audited_candidate_jpy is not None:
        return _CampaignOpenBalance(
            value_jpy=audited_candidate_jpy,
            status="AUDITED_ACCOUNT_DELTA",
            source="broker_balance_minus_account_realized_financing_and_capital_flows",
            audited_candidate_jpy=audited_candidate_jpy,
            blocker=None,
        )

    snapshot_balance = (
        float(snapshot.account.balance_jpy)
        if snapshot is not None and snapshot.account is not None
        else None
    )
    provisional = snapshot_balance
    provisional_source = "broker_snapshot_pending_account_delta"
    if provisional is None and same_campaign_day and previous_start is not None:
        provisional = previous_start
        provisional_source = "same_campaign_day_provisional_preserved"
    if provisional is None and previous_day is not None and previous_day != campaign_day_jst:
        provisional = _coalesce_float(
            previous.get("current_equity_jpy"),
            previous.get("start_balance_jpy"),
        )
        provisional_source = "previous_equity_pending_account_delta"
    if provisional is None:
        raise ValueError(
            "daily target state requires --start-balance, a previous state file, "
            "or a broker snapshot with account summary on first run"
        )
    return _CampaignOpenBalance(
        value_jpy=float(provisional),
        status="PROVISIONAL_UNAUDITED",
        source=provisional_source,
        audited_candidate_jpy=None,
        blocker=(
            "CAMPAIGN_OPEN_BALANCE_UNAUDITED: account-wide realized/financing and "
            "current-day capital-flow evidence is incomplete; fresh-entry capacity stays zero"
        ),
    )


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


def _execution_ledger_covers_campaign_day(
    conn: sqlite3.Connection,
    campaign_day_jst: str,
) -> bool:
    """Prove transaction continuity from at or before the campaign boundary.

    A same-day cold-start baseline updates ``last_oanda_transaction_id`` but
    imports no earlier same-day transactions.  Freshness alone therefore
    cannot prove an account-wide zero delta.  New ledgers persist an explicit
    coverage start; legacy ledgers may be accepted only when durable OANDA
    transaction/event rows predate the boundary.
    """

    campaign_start = datetime.fromisoformat(f"{campaign_day_jst}T00:00:00+00:00")
    marker_row = conn.execute(
        "SELECT value FROM sync_state WHERE key = ?",
        (OANDA_TRANSACTION_COVERAGE_START_KEY,),
    ).fetchone()
    if marker_row is not None:
        coverage_start = _parse_utc_timestamp(marker_row[0])
        # An invalid durable marker is corruption, not permission to use a
        # weaker heuristic.  Likewise a same-day baseline after 00:00 cannot
        # establish what happened between day-open and baseline time.
        return coverage_start is not None and coverage_start <= campaign_start

    candidates: list[datetime] = []
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "oanda_transactions" in tables:
        row = conn.execute(
            "SELECT MIN(time_utc) FROM oanda_transactions WHERE NULLIF(time_utc, '') IS NOT NULL"
        ).fetchone()
        parsed = _parse_utc_timestamp(row[0] if row else None)
        if parsed is not None:
            candidates.append(parsed)

    event_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
    }
    if {"source", "oanda_transaction_id", "ts_utc"}.issubset(event_columns):
        row = conn.execute(
            """
            SELECT MIN(ts_utc)
            FROM execution_events
            WHERE source = 'oanda'
              AND NULLIF(oanda_transaction_id, '') IS NOT NULL
            """
        ).fetchone()
        parsed = _parse_utc_timestamp(row[0] if row else None)
        if parsed is not None:
            candidates.append(parsed)

    return bool(candidates) and min(candidates) <= campaign_start


def _attributed_realized_from_execution_ledger(
    path: Path | None,
    campaign_day_jst: str,
    *,
    broker_last_transaction_id: str | None = None,
) -> _AttributedRealized | None:
    """Read system P/L/loss spend plus account-wide realized/financing.

    System-attributed values drive trader progress and loss capacity.  The
    account-wide net deliberately includes manual/tagless closes and is used
    only to reverse all cash-P/L changes out of broker balance when proving a
    new campaign-day opening.  Mixing these two scopes is what let manual
    profit silently increase the autonomous trader's daily allowance.

    `_campaign_day_key()` is equivalent to the UTC calendar date because the
    campaign resets at 09:00 JST. The execution ledger stores UTC timestamps,
    so filtering by the first 10 chars of `ts_utc` matches the same boundary.

    The figure is authoritative only when the ledger has actually been synced
    during the current campaign day (`sync_state.updated_at_utc`), covers the
    day boundary, and (when supplied) has the same last transaction ID as the
    broker snapshot whose balance is being reversed. A stale, cold-baselined,
    or snapshot-mismatched ledger returns None rather than treating an
    incomplete zero/delta as broker truth.
    """

    if path is None or not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            sync_row = conn.execute(
                "SELECT value, updated_at_utc FROM sync_state WHERE key = 'last_oanda_transaction_id'"
            ).fetchone()
            if sync_row is None or not isinstance(sync_row[1], str):
                return None
            broker_transaction_id = str(broker_last_transaction_id or "").strip()
            if broker_transaction_id and str(sync_row[0] or "").strip() != broker_transaction_id:
                return None
            try:
                synced_day = _campaign_day_key(datetime.fromisoformat(sync_row[1]))
            except ValueError:
                return None
            if synced_day < campaign_day_jst:
                return None
            if not _execution_ledger_covers_campaign_day(conn, campaign_day_jst):
                return None
            row = conn.execute(
                """
                WITH gateway_entries AS (
                    SELECT
                        NULLIF(trade_id, '') AS trade_id,
                        NULLIF(order_id, '') AS order_id,
                        NULLIF(lane_id, '') AS lane_id
                    FROM execution_events
                    WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
                      AND NULLIF(lane_id, '') IS NOT NULL
                ),
                entries AS (
                    SELECT
                        e.trade_id AS trade_id,
                        COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS lane_id,
                        MAX(
                            CASE
                                WHEN LOWER(
                                    COALESCE(
                                        json_extract(e.raw_json, '$.clientExtensions.tag'),
                                        json_extract(e.raw_json, '$.tradeClientExtensions.tag'),
                                        json_extract(e.raw_json, '$.tradeOpened.clientExtensions.tag'),
                                        json_extract(e.raw_json, '$.tradeOpened.tradeClientExtensions.tag'),
                                        ''
                                    )
                                ) IN ('manual', 'operator_manual', 'unknown', 'external')
                                THEN 1 ELSE 0
                            END
                        ) AS manual_owner_marker
                    FROM execution_events AS e
                    LEFT JOIN gateway_entries AS g
                      ON (g.trade_id IS NOT NULL AND g.trade_id = e.trade_id)
                      OR (g.order_id IS NOT NULL AND g.order_id = e.order_id)
                    WHERE e.event_type = 'ORDER_FILLED'
                      AND NULLIF(e.trade_id, '') IS NOT NULL
                    GROUP BY e.trade_id
                    HAVING COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) IS NOT NULL
                       AND manual_owner_marker = 0
                       AND LOWER(COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)))
                           NOT IN ('manual', 'operator_manual', 'unknown', 'external')
                       AND LOWER(COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)))
                           NOT LIKE 'manual:%'
                       AND LOWER(COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)))
                           NOT LIKE 'operator_manual:%'
                       AND LOWER(COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)))
                           NOT LIKE 'unknown:%'
                       AND LOWER(COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)))
                           NOT LIKE 'external:%'
                ),
                close_raw_reconciliation AS (
                    SELECT
                        e.rowid AS event_rowid,
                        e.event_type AS event_type,
                        e.trade_id AS trade_id,
                        e.raw_json AS raw_json,
                        COALESCE(e.realized_pl_jpy, 0.0) AS normalized_realized_jpy,
                        COALESCE(e.financing_jpy, 0.0) AS normalized_financing_jpy,
                        CASE
                            WHEN e.event_type = 'TRADE_CLOSED' THEN (
                                SELECT COUNT(*)
                                FROM json_each(e.raw_json, '$.tradesClosed') AS closed
                                WHERE CAST(json_extract(closed.value, '$.tradeID') AS TEXT) = e.trade_id
                            )
                            WHEN CAST(json_extract(e.raw_json, '$.tradeReduced.tradeID') AS TEXT) = e.trade_id
                                THEN 1
                            ELSE 0
                        END AS matching_component_count,
                        CASE
                            WHEN e.event_type = 'TRADE_CLOSED' THEN COALESCE(
                                (
                                    SELECT CAST(
                                        COALESCE(
                                            json_extract(closed.value, '$.realizedPL'),
                                            CASE
                                                WHEN json_array_length(e.raw_json, '$.tradesClosed') = 1
                                                THEN json_extract(e.raw_json, '$.pl')
                                            END,
                                            0.0
                                        ) AS REAL
                                    )
                                    FROM json_each(e.raw_json, '$.tradesClosed') AS closed
                                    WHERE CAST(json_extract(closed.value, '$.tradeID') AS TEXT) = e.trade_id
                                    LIMIT 1
                                ),
                                0.0
                            )
                            ELSE CAST(
                                COALESCE(
                                    json_extract(e.raw_json, '$.tradeReduced.realizedPL'),
                                    json_extract(e.raw_json, '$.pl'),
                                    0.0
                                ) AS REAL
                            )
                        END AS raw_realized_jpy,
                        CASE
                            WHEN e.event_type = 'TRADE_CLOSED' THEN COALESCE(
                                (
                                    SELECT CAST(
                                        COALESCE(
                                            json_extract(closed.value, '$.financing'),
                                            CASE
                                                WHEN json_array_length(e.raw_json, '$.tradesClosed') = 1
                                                THEN json_extract(e.raw_json, '$.financing')
                                            END,
                                            0.0
                                        ) AS REAL
                                    )
                                    FROM json_each(e.raw_json, '$.tradesClosed') AS closed
                                    WHERE CAST(json_extract(closed.value, '$.tradeID') AS TEXT) = e.trade_id
                                    LIMIT 1
                                ),
                                0.0
                            )
                            ELSE CAST(
                                COALESCE(
                                    json_extract(e.raw_json, '$.tradeReduced.financing'),
                                    json_extract(e.raw_json, '$.financing'),
                                    0.0
                                ) AS REAL
                            )
                        END AS raw_financing_jpy
                    FROM execution_events AS e
                    WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                      AND substr(e.ts_utc, 1, 10) = ?
                ),
                close_raw_audit AS (
                    SELECT COUNT(*) AS invalid_event_count
                    FROM close_raw_reconciliation
                    WHERE json_valid(raw_json) != 1
                       OR COALESCE(json_extract(raw_json, '$.type'), '') != 'ORDER_FILL'
                       OR matching_component_count != 1
                       OR ABS(normalized_realized_jpy - raw_realized_jpy) > ?
                       OR ABS(normalized_financing_jpy - raw_financing_jpy) > ?
                ),
                account_close_events AS (
                    SELECT
                        e.trade_id AS trade_id,
                        COALESCE(e.realized_pl_jpy, 0.0)
                            + COALESCE(e.financing_jpy, 0.0) AS event_pl_jpy
                    FROM execution_events AS e
                    WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                      AND substr(e.ts_utc, 1, 10) = ?
                ),
                financing_transactions AS (
                    SELECT
                        e.rowid AS event_rowid,
                        COALESCE(e.financing_jpy, 0.0) AS account_financing_jpy,
                        e.raw_json AS raw_json
                    FROM execution_events AS e
                    WHERE e.event_type = 'OANDA_TRANSACTION'
                      AND substr(e.ts_utc, 1, 10) = ?
                      AND COALESCE(e.financing_jpy, 0.0) != 0.0
                ),
                transfer_transactions AS (
                    SELECT
                        e.rowid AS event_rowid,
                        CAST(json_extract(e.raw_json, '$.amount') AS REAL) AS amount_jpy
                    FROM execution_events AS e
                    WHERE e.event_type = 'OANDA_TRANSACTION'
                      AND substr(e.ts_utc, 1, 10) = ?
                      AND json_extract(e.raw_json, '$.type') = 'TRANSFER_FUNDS'
                      AND json_type(e.raw_json, '$.amount') IN ('integer', 'real', 'text')
                ),
                financing_components AS (
                    SELECT
                        financing_transactions.event_rowid AS event_rowid,
                        NULLIF(json_extract(open_trade.value, '$.tradeID'), '') AS trade_id,
                        CAST(json_extract(open_trade.value, '$.financing') AS REAL) AS event_pl_jpy
                    FROM financing_transactions
                    JOIN json_each(financing_transactions.raw_json, '$.positionFinancings') AS position
                    JOIN json_each(position.value, '$.openTradeFinancings') AS open_trade
                    WHERE json_type(open_trade.value, '$.tradeID') = 'text'
                      AND json_type(open_trade.value, '$.financing') IN ('integer', 'real', 'text')
                ),
                financing_audit AS (
                    SELECT
                        financing_transactions.event_rowid AS event_rowid,
                        financing_transactions.account_financing_jpy AS account_financing_jpy,
                        COUNT(financing_components.trade_id) AS component_count,
                        COALESCE(SUM(financing_components.event_pl_jpy), 0.0) AS component_financing_jpy
                    FROM financing_transactions
                    LEFT JOIN financing_components
                      ON financing_components.event_rowid = financing_transactions.event_rowid
                    GROUP BY financing_transactions.event_rowid
                ),
                unsupported_cash_event_audit AS (
                    SELECT COUNT(DISTINCT e.rowid) AS unsupported_event_count
                    FROM execution_events AS e
                    WHERE substr(e.ts_utc, 1, 10) = ?
                      AND e.event_type IN (
                          'ORDER_FILLED',
                          'TRADE_CLOSED',
                          'TRADE_REDUCED',
                          'OANDA_TRANSACTION'
                      )
                      AND (
                          ABS(COALESCE(CAST(json_extract(e.raw_json, '$.commission') AS REAL), 0.0)) > 0.0
                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.guaranteedExecutionFee') AS REAL), 0.0)) > 0.0
                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.dividendAdjustment') AS REAL), 0.0)) > 0.0
                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.quoteDividendAdjustment') AS REAL), 0.0)) > 0.0
                          OR (
                              e.event_type = 'OANDA_TRANSACTION'
                              AND (
                                  json_valid(e.raw_json) != 1
                                  OR NULLIF(json_extract(e.raw_json, '$.type'), '') IS NULL
                                  OR (
                                      json_extract(e.raw_json, '$.type') NOT IN (
                                          'DAILY_FINANCING',
                                          'TRANSFER_FUNDS',
                                          'TRADE_CLIENT_EXTENSIONS_MODIFY',
                                          'ORDER_CLIENT_EXTENSIONS_MODIFY'
                                      )
                                      AND (
                                          json_extract(e.raw_json, '$.accountBalance') IS NOT NULL
                                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.amount') AS REAL), 0.0)) > 0.0
                                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.pl') AS REAL), 0.0)) > 0.0
                                          OR ABS(COALESCE(CAST(json_extract(e.raw_json, '$.financing') AS REAL), 0.0)) > 0.0
                                      )
                                  )
                                  OR (
                                      json_extract(e.raw_json, '$.type') = 'TRANSFER_FUNDS'
                                      AND COALESCE(json_type(e.raw_json, '$.amount'), '')
                                          NOT IN ('integer', 'real', 'text')
                                  )
                              )
                          )
                      )
                ),
                system_close_events AS (
                    SELECT account_close_events.event_pl_jpy
                    FROM account_close_events
                    INNER JOIN entries
                      ON entries.trade_id = account_close_events.trade_id
                ),
                system_financing_events AS (
                    SELECT financing_components.event_pl_jpy
                    FROM financing_components
                    INNER JOIN entries
                      ON entries.trade_id = financing_components.trade_id
                ),
                system_realized_events AS (
                    SELECT event_pl_jpy FROM system_close_events
                    UNION ALL
                    SELECT event_pl_jpy FROM system_financing_events
                )
                SELECT
                    COALESCE(SUM(event_pl_jpy), 0.0),
                    COALESCE(SUM(CASE WHEN event_pl_jpy < 0.0 THEN -event_pl_jpy ELSE 0.0 END), 0.0),
                    COALESCE((SELECT SUM(event_pl_jpy) FROM account_close_events), 0.0)
                        + COALESCE((SELECT SUM(account_financing_jpy) FROM financing_transactions), 0.0),
                    COALESCE((SELECT SUM(amount_jpy) FROM transfer_transactions), 0.0),
                    COALESCE(
                        (
                            SELECT SUM(
                                CASE
                                    WHEN component_count = 0
                                      OR ABS(account_financing_jpy - component_financing_jpy) > ?
                                    THEN 1 ELSE 0
                                END
                            )
                            FROM financing_audit
                        ),
                        0
                    ),
                    COALESCE(
                        (SELECT unsupported_event_count FROM unsupported_cash_event_audit),
                        0
                    ) + COALESCE(
                        (SELECT invalid_event_count FROM close_raw_audit),
                        0
                    )
                FROM system_realized_events
                """,
                (
                    campaign_day_jst,
                    ACCOUNT_CASH_RECONCILIATION_TOLERANCE_JPY,
                    ACCOUNT_CASH_RECONCILIATION_TOLERANCE_JPY,
                    campaign_day_jst,
                    campaign_day_jst,
                    campaign_day_jst,
                    campaign_day_jst,
                    ACCOUNT_CASH_RECONCILIATION_TOLERANCE_JPY,
                ),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return _AttributedRealized(0.0, 0.0, 0.0, 0.0)
    # Every non-zero daily-financing transaction must enumerate/reconcile its
    # trade components, and every close/reduction must reconcile normalized
    # values to valid raw ORDER_FILL cash fields with no unsupported non-zero
    # adjustment. Otherwise attribution is unknowable: do not guess an opening
    # balance or loss allowance.
    if int(row[4] or 0) > 0 or int(row[5] or 0) > 0:
        return None
    return _AttributedRealized(
        net_jpy=float(row[0] or 0.0),
        gross_loss_spent_jpy=float(row[1] or 0.0),
        account_net_jpy=float(row[2] or 0.0),
        account_capital_flows_jpy=float(row[3] or 0.0),
    )


def _realized_pl_from_execution_ledger(path: Path | None, campaign_day_jst: str) -> float | None:
    """Compatibility reader for callers that only need attributed net P/L."""

    realized = _attributed_realized_from_execution_ledger(path, campaign_day_jst)
    return realized.net_jpy if realized is not None else None


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


def _empty_pace_evidence() -> _BacktestPaceEvidence:
    return _BacktestPaceEvidence(
        selected_pace=None,
        selected_basis_uncapped_pace=None,
        generated_at_utc=None,
        uncapped_required_trades_per_day=None,
        uncapped_required_trades_per_day_basis_return_pct=None,
        observed_trades_per_day=None,
        observed_expectancy_jpy_per_trade=None,
        frequency_multiple_required=None,
    )


def _pace_from_firepower_payload(payload: dict[str, Any]) -> _BacktestPace | None:
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


def _pace_evidence_from_backtest(path: Path | None) -> _BacktestPaceEvidence:
    """Atomically load selected sizing pace and uncapped stretch firepower.

    The selected pace may target a nearer 5-10% band while top-level firepower
    describes the full stretch target. They remain separate so a bounded
    operating pace cannot hide either the selected-band gap or the stretch gap.
    None of these visibility fields grants or blocks execution.
    """

    if path is None or not path.exists():
        return _empty_pace_evidence()
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return _empty_pace_evidence()
        target_sizing_pace = _pace_from_target_sizing(payload)
        target_band_pace = _pace_from_target_band(payload)
        selected_pace = (
            target_sizing_pace
            or target_band_pace
            or _pace_from_firepower_payload(payload)
        )
        firepower = payload.get("firepower")
        firepower = firepower if isinstance(firepower, dict) else {}
        return _BacktestPaceEvidence(
            selected_pace=selected_pace,
            selected_basis_uncapped_pace=target_band_pace,
            generated_at_utc=(
                str(payload.get("generated_at_utc"))
                if payload.get("generated_at_utc") not in (None, "")
                else None
            ),
            uncapped_required_trades_per_day=_coalesce_int(
                firepower.get("required_trades_per_day_at_observed_expectancy")
            ),
            uncapped_required_trades_per_day_basis_return_pct=_coalesce_float(
                payload.get("target_return_pct")
            ),
            observed_trades_per_day=_coalesce_float(
                firepower.get("avg_selected_trades_per_day")
            ),
            observed_expectancy_jpy_per_trade=_coalesce_float(
                firepower.get("avg_selected_trade_jpy")
            ),
            frequency_multiple_required=_coalesce_float(
                firepower.get("trade_frequency_multiple_required")
            ),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _empty_pace_evidence()


def _pace_from_backtest(path: Path | None) -> _BacktestPace | None:
    """Compatibility reader for the selected pre-cap pace."""

    return _pace_evidence_from_backtest(path).selected_pace


def _backtest_evidence_newer_than_previous_state(
    evidence: _BacktestPaceEvidence,
    previous: dict[str, Any],
) -> bool:
    backtest_generated = _parse_utc_timestamp(evidence.generated_at_utc)
    previous_generated = _parse_utc_timestamp(previous.get("generated_at_utc"))
    return backtest_generated is not None and previous_generated is not None and backtest_generated > previous_generated


def _backtest_newer_than_previous_state(path: Path | None, previous: dict[str, Any]) -> bool:
    """Compatibility wrapper for callers that have only a path."""

    return _backtest_evidence_newer_than_previous_state(
        _pace_evidence_from_backtest(path),
        previous,
    )


def _trade_pace_feasibility(
    *,
    required_trades_per_day: int | None,
    operating_pace_trades_per_day: int,
    observed_trades_per_day: float | None,
    observed_expectancy_jpy_per_trade: float | None,
) -> tuple[bool | None, str]:
    """Classify pace evidence without turning it into a trade blocker."""

    if (
        observed_expectancy_jpy_per_trade is not None
        and observed_expectancy_jpy_per_trade <= 0.0
    ):
        return False, "INFEASIBLE_NON_POSITIVE_EXPECTANCY"
    if required_trades_per_day is None:
        return None, "UNKNOWN_MISSING_FIREPOWER_EVIDENCE"
    if required_trades_per_day > operating_pace_trades_per_day:
        return False, "INFEASIBLE_AT_OPERATING_PACE"
    if observed_trades_per_day is None:
        return True, "OPERATING_PACE_FEASIBLE_OBSERVED_UNKNOWN"
    if observed_trades_per_day + 1e-9 >= required_trades_per_day:
        return True, "OBSERVED_PACE_MEETS_REQUIRED"
    return True, "OPERATING_PACE_FEASIBLE_OBSERVED_SHORT"


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
