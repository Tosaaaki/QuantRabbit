from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import (
    BrokerOrder,
    BrokerSnapshot,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    RiskDecision,
    RiskIssue,
    RiskMetrics,
    Side,
    TradeMethod,
)


@dataclass(frozen=True)
class InstrumentSpec:
    pair: str
    pip_factor: int
    normal_spread_pips: float

    @property
    def pip_size(self) -> float:
        return 1.0 / self.pip_factor


DEFAULT_SPECS: dict[str, InstrumentSpec] = {
    "USD_JPY": InstrumentSpec("USD_JPY", 100, 0.4),
    "EUR_JPY": InstrumentSpec("EUR_JPY", 100, 0.8),
    "GBP_JPY": InstrumentSpec("GBP_JPY", 100, 1.5),
    "AUD_JPY": InstrumentSpec("AUD_JPY", 100, 0.8),
    "EUR_USD": InstrumentSpec("EUR_USD", 10000, 0.5),
    "GBP_USD": InstrumentSpec("GBP_USD", 10000, 0.9),
    "AUD_USD": InstrumentSpec("AUD_USD", 10000, 0.5),
}


@dataclass(frozen=True)
class RiskPolicy:
    # Library default for tests and ad-hoc construction. Production code MUST
    # NOT rely on this literal: target.py derives daily_risk_budget_jpy from
    # equity * daily_risk_pct, intent_generator pulls that into intent.metadata,
    # and validate() prefers metadata over policy when both are present.
    max_loss_jpy: float | None = 500.0
    # Equity-percent cap used by daily-target-state to derive the day's risk
    # budget from starting balance (e.g. 2.0 = 2% of equity per trading day).
    daily_risk_pct: float | None = 2.0
    # Number of independent trade attempts the campaign expects to make in a
    # day. The per-trade JPY cap is daily_risk_budget_jpy / target_trades_per_day,
    # so this knob shapes "few big shots" vs "many small probes".
    #
    # Per AGENT_CONTRACT §3.5:
    # (a) market reality: a realistic FX scalp/swing day fires 5–30 trades
    #     depending on regime, session liquidity, and pacing. 10 is the median
    #     observed in recent trader cycles and matches the "fire many small
    #     shots, let winners run" operating mode the campaign targets.
    # (b) constant rather than derived: this is operator policy (campaign
    #     pacing), not market data. Higher conviction → set lower (e.g. 5).
    #     High-frequency scalp → set higher (e.g. 20). Default 10 makes a
    #     single losing trade burn ≈10% of the day's risk budget, leaving
    #     room for many more attempts.
    # (c) replace via: --target-trades-per-day on daily-target-state, or by
    #     deriving from mining (target_return_pct / (avg_target_R * win_rate))
    #     once that wiring exists in DailyTargetLedger.
    target_trades_per_day: int | None = 10
    min_reward_risk: float = 1.2
    max_quote_age_seconds: int = 20
    max_spread_multiple: float = 2.5
    min_target_spread_multiple: float = 5.0
    min_stop_spread_multiple: float = 5.0
    block_new_entries_with_open_positions: bool = True
    block_new_entries_with_external_risk: bool = True
    block_unprotected_positions: bool = True
    block_new_entries_with_pending_entry_orders: bool = True
    require_live_enabled_for_send: bool = True
    require_market_context_for_live_send: bool = True
    allow_protected_trader_position_adds: bool = False
    # OANDA trades without the vNext trader tag are operator-managed manual
    # exposure. They remain visible in broker truth, but the autonomous trader
    # must not protect, close, or count them against its own entry budget.
    allow_operator_managed_manual_exposure: bool = True
    max_portfolio_positions: int = 4
    max_portfolio_loss_jpy: float | None = None


class RiskEngine:
    def __init__(
        self,
        *,
        policy: RiskPolicy | None = None,
        specs: dict[str, InstrumentSpec] | None = None,
        live_enabled: bool = False,
    ) -> None:
        self.policy = policy or RiskPolicy()
        self.specs = specs or DEFAULT_SPECS
        self.live_enabled = live_enabled

    def validate(self, intent: OrderIntent, snapshot: BrokerSnapshot, *, for_live_send: bool = False) -> RiskDecision:
        issues: list[RiskIssue] = []
        spec = self._spec(intent.pair)
        quote = snapshot.quotes.get(intent.pair)

        if for_live_send and self.policy.require_live_enabled_for_send and not self.live_enabled:
            issues.append(RiskIssue("LIVE_DISABLED", "live execution is disabled; dry-run only"))

        if intent.owner != Owner.TRADER:
            issues.append(RiskIssue("OWNER_NOT_TRADER", f"order owner must be trader, got {intent.owner.value}"))
        if intent.units <= 0:
            issues.append(RiskIssue("BAD_UNITS", f"units must be positive, got {intent.units}"))
        if not intent.thesis.strip():
            issues.append(RiskIssue("MISSING_THESIS", "order intent must carry a non-empty thesis"))
        issues.extend(self._market_context_issues(intent, for_live_send=for_live_send))

        entry_relevant_positions = self._entry_relevant_positions(snapshot)
        portfolio_add_mode = self.policy.allow_protected_trader_position_adds
        if self.policy.block_new_entries_with_open_positions and not portfolio_add_mode:
            for position in entry_relevant_positions:
                issues.append(
                    RiskIssue(
                        "OPEN_POSITION_EXISTS",
                        f"open broker position blocks fresh entry: {position.pair} {position.side.value} "
                        f"id={position.trade_id} {position.units}u; manage exposure before adding risk",
                    )
                )
        elif self.policy.block_new_entries_with_open_positions and portfolio_add_mode:
            if len(entry_relevant_positions) >= self.policy.max_portfolio_positions:
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_POSITION_LIMIT",
                        f"open trader positions {len(entry_relevant_positions)} reached portfolio limit "
                        f"{self.policy.max_portfolio_positions}",
                    )
                )
            for position in entry_relevant_positions:
                if position.owner != Owner.TRADER or position.stop_loss is None or position.take_profit is None:
                    issues.append(
                        RiskIssue(
                            "OPEN_POSITION_EXISTS",
                            f"only protected trader-owned positions can be layered; "
                            f"{position.pair} {position.side.value} id={position.trade_id} is not eligible",
                        )
                    )
                elif (
                    position.pair == intent.pair
                    and position.side != intent.side
                    and not _account_hedging_enabled(snapshot)
                ):
                    issues.append(
                        RiskIssue(
                            "OPPOSING_POSITION_NEEDS_HEDGING",
                            f"fresh {intent.pair} {intent.side.value} entry opposes protected "
                            f"{position.pair} {position.side.value} id={position.trade_id}, but broker snapshot "
                            "does not prove account hedging is enabled",
                        )
                    )

        if self.policy.block_new_entries_with_external_risk:
            for position in entry_relevant_positions:
                if position.owner != Owner.TRADER:
                    issues.append(
                        RiskIssue(
                            "EXTERNAL_RISK_OPEN",
                            f"external/manual risk is open: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u; adopt or close before new entries",
                        )
                    )

        if self.policy.block_unprotected_positions:
            for position in entry_relevant_positions:
                if position.stop_loss is None or position.take_profit is None:
                    missing = []
                    if position.take_profit is None:
                        missing.append("TP")
                    if position.stop_loss is None:
                        missing.append("SL")
                    issues.append(
                        RiskIssue(
                            "UNPROTECTED_POSITION",
                            f"open position lacks {'/'.join(missing)}: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u",
                        )
                    )

        if self.policy.block_new_entries_with_pending_entry_orders:
            for order in snapshot.orders:
                if _is_pending_entry_order(order):
                    issues.append(
                        RiskIssue(
                            "PENDING_ENTRY_ORDER_OPEN",
                            f"pending entry order is already open: {order.pair or '(unknown)'} "
                            f"{order.order_type} id={order.order_id}; resolve it before new entries",
                        )
                    )

        if quote is None:
            issues.append(RiskIssue("MISSING_QUOTE", f"missing live quote for {intent.pair}"))
            return RiskDecision(False, None, tuple(issues))

        quote_age = max(0.0, (datetime.now(timezone.utc) - quote.timestamp_utc).total_seconds())
        if quote_age > self.policy.max_quote_age_seconds:
            issues.append(
                RiskIssue(
                    "STALE_QUOTE",
                    f"{intent.pair} quote is stale: {quote_age:.1f}s > {self.policy.max_quote_age_seconds}s",
                )
            )

        spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
        if spread_pips > spec.normal_spread_pips * self.policy.max_spread_multiple:
            issues.append(
                RiskIssue(
                    "SPREAD_TOO_WIDE",
                    f"{intent.pair} spread {spread_pips:.1f}pip exceeds "
                    f"{self.policy.max_spread_multiple:.1f}x normal {spec.normal_spread_pips:.1f}pip",
                )
            )

        issues.extend(self._entry_contract_issues(intent, quote, spec, spread_pips, for_live_send=for_live_send))
        entry_price = self._entry_price(intent, quote)
        issues.extend(self._conversion_quote_issues(intent.pair, snapshot))
        quote_to_jpy = self._quote_to_jpy(intent.pair, snapshot)
        if quote_to_jpy is None:
            return RiskDecision(False, None, tuple(issues))

        metrics = self._metrics(intent, quote, spec, entry_price, quote_to_jpy)
        if metrics.loss_pips <= 0:
            issues.append(RiskIssue("SL_NOT_LOSS_SIDE", f"SL is not on the loss side for {intent.side.value}"))
        if metrics.reward_pips <= 0:
            issues.append(RiskIssue("TP_NOT_REWARD_SIDE", f"TP is not on the reward side for {intent.side.value}"))
        loss_cap = self._resolved_loss_cap(intent)
        if loss_cap is None:
            issues.append(
                RiskIssue(
                    "LOSS_CAP_MISSING",
                    "no per-trade loss cap available: pass policy.max_loss_jpy or "
                    "intent.metadata['max_loss_jpy'] (equity-derived); refusing to validate without one.",
                )
            )
        elif metrics.risk_jpy > loss_cap:
            issues.append(
                RiskIssue(
                    "LOSS_CAP_EXCEEDED",
                    f"planned worst-case loss {metrics.risk_jpy:.0f} JPY exceeds cap {loss_cap:.0f} JPY",
                )
            )
        if metrics.reward_risk < self.policy.min_reward_risk:
            issues.append(
                RiskIssue(
                    "REWARD_RISK_TOO_LOW",
                    f"planned reward/risk {metrics.reward_risk:.2f}x is below {self.policy.min_reward_risk:.2f}x",
                )
            )
        if metrics.reward_pips < spread_pips * self.policy.min_target_spread_multiple:
            issues.append(
                RiskIssue(
                    "TARGET_TOO_THIN_FOR_SPREAD",
                    f"target {metrics.reward_pips:.1f}pip is less than "
                    f"{self.policy.min_target_spread_multiple:.1f}x spread {spread_pips:.1f}pip",
                )
            )
        if metrics.loss_pips < spread_pips * self.policy.min_stop_spread_multiple:
            issues.append(
                RiskIssue(
                    "STOP_TOO_THIN_FOR_SPREAD",
                    f"stop {metrics.loss_pips:.1f}pip is less than "
                    f"{self.policy.min_stop_spread_multiple:.1f}x spread {spread_pips:.1f}pip",
                )
            )
        if portfolio_add_mode and self.policy.max_portfolio_loss_jpy is not None:
            portfolio_risk, risk_issue = self._open_portfolio_risk_jpy(snapshot)
            if risk_issue:
                issues.append(risk_issue)
            elif portfolio_risk + metrics.risk_jpy > self.policy.max_portfolio_loss_jpy:
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_LOSS_CAP_EXCEEDED",
                        f"open risk {portfolio_risk:.0f} JPY + candidate risk {metrics.risk_jpy:.0f} JPY "
                        f"exceeds portfolio cap {self.policy.max_portfolio_loss_jpy:.0f} JPY",
                    )
                )

        return RiskDecision(
            allowed=not any(issue.severity == "BLOCK" for issue in issues),
            metrics=metrics,
            issues=tuple(issues),
        )

    def _entry_relevant_positions(self, snapshot: BrokerSnapshot) -> tuple[BrokerPosition, ...]:
        if not self.policy.allow_operator_managed_manual_exposure:
            return tuple(snapshot.positions)
        return tuple(position for position in snapshot.positions if not _is_operator_managed_manual(position))

    def _spec(self, pair: str) -> InstrumentSpec:
        try:
            return self.specs[pair]
        except KeyError as exc:
            raise ValueError(f"unsupported instrument: {pair}") from exc

    def _resolved_loss_cap(self, intent: OrderIntent) -> float | None:
        """Return the per-trade loss cap to enforce.

        Resolution order (no JPY literal fallback):
            1. intent.metadata['max_loss_jpy'] — caller (intent generator) injected an
               equity-derived cap for this specific lane.
            2. policy.max_loss_jpy — explicit policy-wide cap from CLI / config.
            3. None — validator emits LOSS_CAP_MISSING and refuses the trade.
        """
        meta = intent.metadata or {}
        cap = meta.get("max_loss_jpy")
        if cap is not None:
            try:
                cap_value = float(cap)
            except (TypeError, ValueError):
                return None
            return cap_value if cap_value > 0 else None
        if self.policy.max_loss_jpy is not None and self.policy.max_loss_jpy > 0:
            return float(self.policy.max_loss_jpy)
        return None

    def _entry_price(self, intent: OrderIntent, quote: Quote) -> float:
        if intent.order_type == OrderType.MARKET:
            return quote.ask if intent.side == Side.LONG else quote.bid
        if intent.entry is not None:
            return float(intent.entry)
        return quote.ask if intent.side == Side.LONG else quote.bid

    def _entry_contract_issues(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        spread_pips: float,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        if intent.order_type == OrderType.MARKET:
            return self._market_entry_issues(intent, quote, spec, spread_pips, for_live_send=for_live_send)
        if intent.entry is None:
            return [RiskIssue("PENDING_ENTRY_REQUIRES_ENTRY", f"{intent.order_type.value} requires an entry price")]
        entry = float(intent.entry)
        issues: list[RiskIssue] = []
        if intent.order_type == OrderType.STOP_ENTRY:
            if intent.side == Side.LONG and entry <= quote.ask:
                issues.append(
                    RiskIssue(
                        "STOP_ENTRY_NOT_ABOVE_MARKET",
                        f"LONG stop-entry must be above current ask: entry={entry} ask={quote.ask}",
                    )
                )
            if intent.side == Side.SHORT and entry >= quote.bid:
                issues.append(
                    RiskIssue(
                        "STOP_ENTRY_NOT_BELOW_MARKET",
                        f"SHORT stop-entry must be below current bid: entry={entry} bid={quote.bid}",
                    )
                )
        elif intent.order_type == OrderType.LIMIT:
            if intent.side == Side.LONG and entry >= quote.ask:
                issues.append(
                    RiskIssue(
                        "LIMIT_ENTRY_NOT_BELOW_MARKET",
                        f"LONG limit must be below current ask: entry={entry} ask={quote.ask}",
                    )
                )
            if intent.side == Side.SHORT and entry <= quote.bid:
                issues.append(
                    RiskIssue(
                        "LIMIT_ENTRY_NOT_ABOVE_MARKET",
                        f"SHORT limit must be above current bid: entry={entry} bid={quote.bid}",
                    )
                )
        return issues

    def _market_entry_issues(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        spread_pips: float,
        *,
        for_live_send: bool,
    ) -> list[RiskIssue]:
        if intent.entry is None:
            return []
        expected = float(intent.entry)
        executable = quote.ask if intent.side == Side.LONG else quote.bid
        drift_pips = abs(expected - executable) * spec.pip_factor
        allowed_drift = max(spread_pips * 2.0, 1.0)
        if drift_pips <= allowed_drift:
            return []
        severity = "BLOCK" if for_live_send else "WARN"
        return [
            RiskIssue(
                "MARKET_ENTRY_DRIFT",
                f"MARKET expected entry is stale versus broker quote: expected={expected} "
                f"executable={executable} drift={drift_pips:.1f}pip > {allowed_drift:.1f}pip",
                severity=severity,
            )
        ]

    def _market_context_issues(self, intent: OrderIntent, *, for_live_send: bool) -> list[RiskIssue]:
        severity = "BLOCK" if for_live_send and self.policy.require_market_context_for_live_send else "WARN"
        context = intent.market_context
        if context is None:
            return [
                RiskIssue(
                    "MISSING_MARKET_CONTEXT",
                    "order intent must state market regime, narrative, chart story, method, and invalidation",
                    severity=severity,
                )
            ]
        issues: list[RiskIssue] = []
        missing = [
            name
            for name, value in (
                ("regime", context.regime),
                ("narrative", context.narrative),
                ("chart_story", context.chart_story),
                ("invalidation", context.invalidation),
            )
            if not value.strip()
        ]
        if missing:
            issues.append(
                RiskIssue(
                    "INCOMPLETE_MARKET_CONTEXT",
                    f"market context is missing {', '.join(missing)}",
                    severity=severity,
                )
            )
        method_issue = self._method_regime_issue(intent, severity)
        if method_issue:
            issues.append(method_issue)
        return issues

    def _method_regime_issue(self, intent: OrderIntent, severity: str) -> RiskIssue | None:
        context = intent.market_context
        if context is None:
            return None
        regime_text = f"{context.regime} {context.chart_story} {context.narrative}".upper()
        method = context.method
        if method == TradeMethod.RANGE_ROTATION and _contains_any(regime_text, ("TREND", "IMPULSE", "BAND WALK")):
            if not _contains_any(regime_text, ("RANGE", "BOX", "RAIL", "ROTATION")):
                return RiskIssue(
                    "METHOD_REGIME_MISMATCH",
                    "range rotation method needs a range/box/rail story, not a one-way trend or impulse",
                    severity=severity,
                )
        if method == TradeMethod.TREND_CONTINUATION and not _contains_any(
            regime_text, ("TREND", "CONTINUATION", "STAIRCASE", "BAND WALK", "LADDER", "BREAKOUT")
        ):
            return RiskIssue(
                "METHOD_REGIME_MISMATCH",
                "trend continuation method needs a trend/continuation chart story",
                severity=severity,
            )
        if method == TradeMethod.BREAKOUT_FAILURE and not _contains_any(
            regime_text, ("FAIL", "REJECT", "RETEST", "RECLAIM", "TRAP", "BREAK")
        ):
            return RiskIssue(
                "METHOD_REGIME_MISMATCH",
                "breakout-failure method needs a failed break, rejection, retest, reclaim, or trapped-side story",
                severity=severity,
            )
        return None

    def _metrics(
        self,
        intent: OrderIntent,
        quote: Quote,
        spec: InstrumentSpec,
        entry_price: float,
        quote_to_jpy: float,
    ) -> RiskMetrics:
        if intent.side == Side.LONG:
            loss_pips = (entry_price - intent.sl) * spec.pip_factor
            reward_pips = (intent.tp - entry_price) * spec.pip_factor
        else:
            loss_pips = (intent.sl - entry_price) * spec.pip_factor
            reward_pips = (entry_price - intent.tp) * spec.pip_factor
        spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
        jpy_per_pip = (intent.units / spec.pip_factor) * quote_to_jpy
        risk_jpy = max(0.0, loss_pips) * jpy_per_pip
        reward_jpy = max(0.0, reward_pips) * jpy_per_pip
        reward_risk = reward_jpy / risk_jpy if risk_jpy > 0 else 0.0
        return RiskMetrics(
            entry_price=entry_price,
            loss_pips=loss_pips,
            reward_pips=reward_pips,
            risk_jpy=risk_jpy,
            reward_jpy=reward_jpy,
            reward_risk=reward_risk,
            spread_pips=spread_pips,
            jpy_per_pip=jpy_per_pip,
        )

    def _conversion_quote_issues(self, pair: str, snapshot: BrokerSnapshot) -> list[RiskIssue]:
        quote_ccy = pair.split("_", 1)[1]
        if quote_ccy == "JPY":
            return []
        if snapshot.home_conversions.get(quote_ccy, 0.0) > 0:
            return []
        conversion_pair = f"{quote_ccy}_JPY"
        conversion_quote = snapshot.quotes.get(conversion_pair)
        if conversion_quote is None:
            return [
                RiskIssue(
                    "MISSING_CONVERSION_QUOTE",
                    f"{conversion_pair} quote is required to compute broker-truth JPY risk for {pair}",
                )
            ]
        issues: list[RiskIssue] = []
        quote_age = max(0.0, (datetime.now(timezone.utc) - conversion_quote.timestamp_utc).total_seconds())
        if quote_age > self.policy.max_quote_age_seconds:
            issues.append(
                RiskIssue(
                    "STALE_CONVERSION_QUOTE",
                    f"{conversion_pair} conversion quote is stale: "
                    f"{quote_age:.1f}s > {self.policy.max_quote_age_seconds}s",
                )
            )
        spec = self._spec(conversion_pair)
        spread_pips = abs(conversion_quote.ask - conversion_quote.bid) * spec.pip_factor
        if spread_pips > spec.normal_spread_pips * self.policy.max_spread_multiple:
            issues.append(
                RiskIssue(
                    "CONVERSION_SPREAD_TOO_WIDE",
                    f"{conversion_pair} conversion spread {spread_pips:.1f}pip exceeds "
                    f"{self.policy.max_spread_multiple:.1f}x normal {spec.normal_spread_pips:.1f}pip",
                )
            )
        return issues

    def _quote_to_jpy(self, pair: str, snapshot: BrokerSnapshot) -> float | None:
        quote_ccy = pair.split("_", 1)[1]
        if quote_ccy == "JPY":
            return 1.0
        home_conversion = snapshot.home_conversions.get(quote_ccy)
        if home_conversion is not None and home_conversion > 0:
            return float(home_conversion)
        conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
        if conversion_quote is not None:
            return max(conversion_quote.bid, conversion_quote.ask)
        return None

    def _open_portfolio_risk_jpy(self, snapshot: BrokerSnapshot) -> tuple[float, RiskIssue | None]:
        total = 0.0
        for position in self._entry_relevant_positions(snapshot):
            if position.stop_loss is None:
                return 0.0, RiskIssue(
                    "PORTFOLIO_RISK_UNKNOWN",
                    f"open position {position.trade_id} has no SL; cannot compute portfolio risk",
                )
            spec = self._spec(position.pair)
            quote_to_jpy = self._quote_to_jpy(position.pair, snapshot)
            if quote_to_jpy is None:
                return 0.0, RiskIssue(
                    "PORTFOLIO_RISK_UNKNOWN",
                    f"missing conversion quote for open position {position.trade_id} {position.pair}",
                )
            if position.side == Side.LONG:
                loss_pips = (position.entry_price - position.stop_loss) * spec.pip_factor
            else:
                loss_pips = (position.stop_loss - position.entry_price) * spec.pip_factor
            jpy_per_pip = (position.units / spec.pip_factor) * quote_to_jpy
            total += max(0.0, loss_pips) * jpy_per_pip
        return total, None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _account_hedging_enabled(snapshot: BrokerSnapshot) -> bool:
    return bool(snapshot.account and snapshot.account.hedging_enabled)


def _is_operator_managed_manual(position: BrokerPosition) -> bool:
    return position.owner in {Owner.MANUAL, Owner.UNKNOWN}


def _is_pending_entry_order(order: BrokerOrder) -> bool:
    if order.trade_id:
        return False
    order_type = order.order_type.upper()
    return order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}


def resolve_max_loss_jpy(
    *,
    max_loss_jpy: float | None,
    max_loss_pct: float | None,
    equity_jpy: float | None,
    default_max_loss_jpy: float | None = None,
    label: str = "max-loss",
) -> float:
    """Resolve a risk cap from explicit JPY value or percentage of equity."""
    if max_loss_jpy is not None:
        if max_loss_jpy <= 0:
            raise ValueError(f"{label}: --max-loss-jpy must be positive")
        return float(max_loss_jpy)
    if max_loss_pct is not None:
        if max_loss_pct <= 0:
            raise ValueError(f"{label}: --max-loss-pct must be positive")
        if equity_jpy is None:
            raise ValueError(f"{label}: --max-loss-pct requires --risk-equity-jpy or a daily target state")
        if equity_jpy <= 0:
            raise ValueError(f"{label}: --risk-equity-jpy must be positive")
        return equity_jpy * (max_loss_pct / 100.0)
    if default_max_loss_jpy is None:
        raise ValueError(
            f"{label}: no risk cap available. Provide --max-loss-jpy, "
            f"--max-loss-pct + --risk-equity-jpy, or have the daily-target ledger emit "
            f"daily_risk_budget_jpy from current equity. No JPY literal fallback."
        )
    return float(default_max_loss_jpy)
