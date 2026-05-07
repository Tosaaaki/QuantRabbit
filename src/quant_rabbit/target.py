from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
from quant_rabbit.paths import DEFAULT_DAILY_TARGET_REPORT, DEFAULT_DAILY_TARGET_STATE
from quant_rabbit.risk import RiskPolicy


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
    realized_pl_jpy: float
    unrealized_pl_jpy: float
    progress_jpy: float
    progress_pct: float
    remaining_target_jpy: float
    current_equity_jpy: float
    campaign_day_jst: str
    daily_risk_budget_jpy: float
    target_trades_per_day: int
    target_trades_per_day_source: str
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
    progress_jpy: float
    progress_pct: float
    remaining_target_jpy: float
    remaining_risk_budget_jpy: float
    target_trades_per_day: int
    target_trades_per_day_source: str
    per_trade_risk_budget_jpy: float
    unprotected_positions: int


class DailyTargetLedger:
    """Record daily 10% target progress from broker truth and realized PnL."""

    def __init__(
        self,
        *,
        state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        report_path: Path = DEFAULT_DAILY_TARGET_REPORT,
        pace_backtest_path: Path | None = None,
    ) -> None:
        self.state_path = state_path
        self.report_path = report_path
        self.pace_backtest_path = pace_backtest_path

    def run(
        self,
        *,
        start_balance_jpy: float | None = None,
        target_return_pct: float | None = None,
        realized_pl_jpy: float | None = None,
        daily_risk_budget_jpy: float | None = None,
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

        # Priority order:
        # 1. Explicit --start-balance argument (caller override).
        # 2. New campaign day with snapshot.account: derive from OANDA balance/NAV minus
        #    today's realized PnL so the value reflects today's actual broker truth.
        # 3. New campaign day without snapshot.account: roll over previous current_equity.
        # 4. Otherwise reuse previous start_balance.
        snapshot_start_balance = _start_balance_from_snapshot(
            snapshot=snapshot, realized_pl_jpy=realized_pl_jpy
        )
        start_balance = _coalesce_float(start_balance_jpy)
        if start_balance is None and is_new_campaign_day and snapshot_start_balance is not None:
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
        target_pct = _coalesce_float(target_return_pct, previous.get("target_return_pct"), 10.0)
        if is_new_campaign_day and realized_pl_jpy is None:
            realized = 0.0
        else:
            realized = _coalesce_float(realized_pl_jpy, previous.get("realized_pl_jpy"), 0.0)
        # Equity-derived risk budget: per-trade worst-case loss cap is sized to the day's
        # starting equity, not a hardcoded JPY literal. Default uses RiskPolicy.daily_risk_pct
        # (% of starting equity); explicit caller override and previous-day persistence
        # both still win. No silent JPY fallback — if percent and explicit value are both
        # missing, the policy default percent applies but is recorded in the snapshot.
        policy = RiskPolicy()
        explicit_budget = _coalesce_float(daily_risk_budget_jpy, previous.get("daily_risk_budget_jpy"))
        if explicit_budget is not None:
            risk_budget = explicit_budget
        else:
            if policy.daily_risk_pct is None or policy.daily_risk_pct <= 0:
                raise ValueError(
                    "daily-target-state: cannot derive daily_risk_budget_jpy. "
                    "Pass --daily-risk-budget explicitly, or set RiskPolicy.daily_risk_pct."
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
        if explicit_pace is None:
            backtest_pace = _pace_from_backtest(self.pace_backtest_path)
            previous_pace = _coalesce_int(previous.get("target_trades_per_day"))
            if backtest_pace is not None:
                explicit_pace = max(backtest_pace, previous_pace or 0)
                pace_source = (
                    "ai_test_bot_required_trades"
                    if explicit_pace == backtest_pace
                    else "previous_state_above_ai_test_bot"
                )
            elif previous_pace is not None:
                explicit_pace = previous_pace
                pace_source = "previous_state"
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
        if (
            policy.min_per_trade_risk_pct is not None
            and policy.min_per_trade_risk_pct > 0
            and pace_source != "cli"
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
        open_risk = round(
            sum((position.remaining_risk_jpy or 0.0) for position in positions if _counts_against_trader_budget(position)),
            4,
        )
        unprotected = sum(
            1 for position in positions if not position.protected and _counts_against_trader_budget(position)
        )
        progress = round(realized + unrealized, 4)
        target_jpy = round(start_balance * (target_pct / 100.0), 2)
        remaining_target = round(max(0.0, target_jpy - progress), 4)
        remaining_risk_budget = 0.0 if unprotected else round(max(0.0, risk_budget - open_risk), 4)
        current_equity = round(start_balance + progress, 4)
        progress_pct = round((progress / target_jpy) * 100.0, 4) if target_jpy else 0.0
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
            realized_pl_jpy=round(realized, 4),
            unrealized_pl_jpy=unrealized,
            progress_jpy=progress,
            progress_pct=progress_pct,
            remaining_target_jpy=remaining_target,
            current_equity_jpy=current_equity,
            campaign_day_jst=campaign_day_jst,
            daily_risk_budget_jpy=round(risk_budget, 4),
            target_trades_per_day=explicit_pace,
            target_trades_per_day_source=pace_source,
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
            progress_jpy=state.progress_jpy,
            progress_pct=state.progress_pct,
            remaining_target_jpy=state.remaining_target_jpy,
            remaining_risk_budget_jpy=state.remaining_risk_budget_jpy,
            target_trades_per_day=state.target_trades_per_day,
            target_trades_per_day_source=state.target_trades_per_day_source,
            per_trade_risk_budget_jpy=state.per_trade_risk_budget_jpy,
            unprotected_positions=state.unprotected_positions,
        )

    def _load_previous(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())

    def _write_state(self, state: DailyTargetSnapshot) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

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
            f"- Realized PnL: `{state.realized_pl_jpy:.0f} JPY`",
            f"- Trader unrealized PnL: `{state.unrealized_pl_jpy:.0f} JPY`",
            f"- Progress: `{state.progress_jpy:.0f} JPY` (`{state.progress_pct:.1f}%` of target)",
            f"- Remaining target: `{state.remaining_target_jpy:.0f} JPY`",
            f"- Open risk: `{state.open_risk_jpy:.0f} JPY`",
            f"- Remaining risk budget: `{state.remaining_risk_budget_jpy:.0f} JPY`",
            f"- Target trades per day: `{state.target_trades_per_day}` (`{state.target_trades_per_day_source}`)",
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
                "- The 10% daily target is tracked as a product KPI and execution objective, not a guaranteed return.",
                "- Unprotected trader-owned or external exposure makes remaining risk budget unavailable; operator-managed manual/tagless exposure is observed but not managed by the trader.",
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
    if position.take_profit is None:
        missing.append("TP")
    if position.stop_loss is None:
        # SL-free regime (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
        # 「SLいらない」 / 「損失を出さないで稼ぎまくる」): trader-owned SL=None is
        # intentional. Treat as protected for daily-target accounting so the
        # status stays PURSUE_TARGET and basket entries are not blocked by
        # REPAIR_REQUIRED.
        if not (_trader_sl_repair_disabled() and position.owner == Owner.TRADER):
            missing.append("SL")
    remaining_risk = _remaining_risk_jpy(position, quotes, home_conversions or {})
    if position.stop_loss is not None and remaining_risk is None:
        missing.append("JPY_CONVERSION")
    return TargetPositionRisk(
        trade_id=position.trade_id,
        pair=position.pair,
        side=position.side.value,
        owner=position.owner.value,
        units=position.units,
        unrealized_pl_jpy=round(position.unrealized_pl_jpy, 4),
        protected=not missing,
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
    return position.owner not in {Owner.MANUAL.value, Owner.UNKNOWN.value}


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


def _pace_from_backtest(path: Path | None) -> int | None:
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
    firepower = payload.get("firepower")
    if not isinstance(firepower, dict):
        return None
    return _coalesce_int(firepower.get("required_trades_per_day_at_observed_expectancy"))


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
