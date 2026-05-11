from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

from .models import (
    AccountSummary,
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
from .instruments import DEFAULT_TRADER_PAIRS, NORMAL_SPREAD_PIPS, instrument_pip_factor

# OANDA Japan retail FX margin in the current account is 25:1 leverage, i.e.
# 4% initial margin. Recent broker truth confirms the same scale: USD_JPY
# 25,000u filled near 155.962 required roughly 155,954 JPY initial margin.
# This is broker/account policy, not market data; replace it with per-instrument
# `/accounts/{id}/instruments` marginRate once that adapter is wired.
OANDA_JP_RETAIL_FX_MARGIN_RATE = 0.04


@dataclass(frozen=True)
class InstrumentSpec:
    pair: str
    pip_factor: int
    normal_spread_pips: float
    margin_rate: float = OANDA_JP_RETAIL_FX_MARGIN_RATE

    @property
    def pip_size(self) -> float:
        return 1.0 / self.pip_factor


DEFAULT_SPECS: dict[str, InstrumentSpec] = {
    pair: InstrumentSpec(pair, instrument_pip_factor(pair), NORMAL_SPREAD_PIPS[pair])
    for pair in DEFAULT_TRADER_PAIRS
}


def estimate_required_margin_jpy(*, units: int, entry_price: float, quote_to_jpy: float, spec: InstrumentSpec) -> float:
    """Estimate initial margin in account JPY for a candidate FX order."""
    return max(0.0, abs(units) * abs(entry_price) * quote_to_jpy * spec.margin_rate)


def margin_budget_jpy(account: AccountSummary, *, max_margin_utilization_pct: float) -> float:
    """Return the smaller of broker free margin and operator utilization headroom."""
    utilization_budget = account.nav_jpy * (max_margin_utilization_pct / 100.0) - account.margin_used_jpy
    return min(account.margin_available_jpy, utilization_budget)


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
    # Fallback number of independent trade attempts the campaign expects to
    # make in a day. DailyTargetLedger's CLI/automation path first reads
    # ai-test-bot firepower evidence; this policy value is used only when that
    # observed-expectancy pace is unavailable.
    #
    # Per AGENT_CONTRACT §3.5:
    # (a) market reality: a realistic FX scalp/swing day fires 5–30 trades
    #     depending on regime, session liquidity, and pacing. 10 is a safe
    #     minimum fallback; current production pace should come from
    #     ai_test_bot_backtest.firepower.required_trades_per_day_at_observed_expectancy.
    # (b) constant rather than derived: this value is only the no-evidence
    #     operator-policy fallback. When backtest firepower is present, the
    #     ledger uses that market/history-derived pace instead.
    # (c) replace via: --target-trades-per-day on daily-target-state, or by
    #     improving ai-test-bot firepower/expectancy wiring.
    target_trades_per_day: int | None = 10
    # Sanity ceiling on the pace divisor used to derive per_trade_risk_budget_jpy.
    # ai-test-bot.firepower can return required_trades_per_day_at_observed_expectancy
    # values into the hundreds when current strategy expectancy is too thin to
    # hit the daily target at any practical pace (e.g. 229 trades/day). Dividing
    # daily_risk_budget_jpy by such a number sizes each order at ~10–20 JPY
    # worst-case loss, which floors out near broker minimum units and silently
    # makes execution operationally meaningless.
    #
    # Per AGENT_CONTRACT §3.5:
    # (a) market reality: a sustained autonomous FX scalp/swing day rarely
    #     supports more than ~30 independent risk-bounded shots before slippage,
    #     spread cost, and decision-quality degradation dominate edge.
    # (b) constant rather than derived: this is the operator's declared maximum
    #     practical attempt count, not market output. Backtest expectancy gaps
    #     are still surfaced in ai_test_bot.firepower so the operator sees the
    #     gap; the cap only prevents the gap from silently sabotaging sizing.
    # (c) replace via: pass --target-trades-per-day on daily-target-state for an
    #     explicit operator pace, or improve strategy expectancy so backtest
    #     pace falls naturally below the cap.
    max_target_trades_per_day: int | None = 30
    # Floor on per_trade_risk_budget as a fraction of starting equity.
    # (a) market reality: even with a tight daily_risk_pct and a high
    #     target_trades_per_day count, the per-trade slice must still
    #     justify a tradable lane size at OANDA's minimum unit (1k base
    #     currency). At ~0.05% of typical retail FX equity (~JPY 200k) this
    #     keeps per_trade ≈ 100 JPY which is the boundary where 1000-unit
    #     lots remain sized within reasonable SL distances.
    # (b) constant rather than derived: this is operator policy preventing
    #     "math break" cycles where pace × budget drives per-trade into
    #     units the broker cannot honor — see
    #     feedback_high_conviction_execution.md and
    #     feedback_basket_and_pace_cap.md.
    # (c) replace via: improve strategy expectancy so backtest firepower
    #     pace falls naturally, or raise daily_risk_pct intentionally.
    min_per_trade_risk_pct: float | None = 0.05
    # Default reward/risk floor for non-range entries.
    # (a) market reality: trend / breakout-failure setups need their TP to clear
    #     spread + slippage by a margin that compensates for losing trades; 1.2R
    #     is the conservative floor where +EV holds at modest hit rate.
    # (b) constant rather than derived: this is operator policy on minimum
    #     trade quality. Per-regime floors override (see range_min_reward_risk).
    # (c) replace via: tune from post-trade-learning hit-rate distribution per
    #     regime if the floor proves systematically too tight or too loose.
    min_reward_risk: float = 1.2
    # Reward/risk floor when the intent's regime context is RANGE.
    # Per AGENT_CONTRACT §3.5: range regimes deserve faster rotation. Hit rate
    # in clean RANGE is materially above trend, so a lower R floor is +EV when
    # the actual move is bounded by the opposite rail. Range geometry already
    # caps TP at the opposing rail (`_range_geometry`), so this floor only
    # gates the loss/reward ratio, not the absolute target distance.
    range_min_reward_risk: float = 0.6
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
    # Operator-set margin utilization ceiling for autonomous entries.
    # (a) market reality: OANDA rejects orders when marginAvailable cannot
    #     cover initial margin; capping marginUsed keeps the rejection in our
    #     risk gate instead of in broker-side order cancellation.
    # (b) constant rather than derived: this is the current operator policy.
    #     92% means the system may use most NAV while leaving 8% headroom for
    #     spread, slippage, and mark-to-market movement.
    # (c) replace via: pass RiskPolicy(max_margin_utilization_pct=...) from
    #     CLI/config when an operator-facing knob is introduced.
    max_margin_utilization_pct: float | None = 92.0
    require_margin_account: bool = True
    allow_protected_trader_position_adds: bool = False
    # OANDA trades without the vNext trader tag are operator-managed manual
    # exposure. They remain visible in broker truth, but the autonomous trader
    # must not protect, close, or count them against its own entry budget.
    allow_operator_managed_manual_exposure: bool = True
    # Concurrent trader-owned positions cap. Default 4 caps simultaneous
    # exposure to ~4 lanes; live env can override via `QR_MAX_PORTFOLIO_POSITIONS`
    # for attack-mode multi-lane participation (`feedback_attack_mode_sizing.md`).
    # NOTE: use `field(default_factory=...)` so the env override is honored at
    # instance construction. With a bare `= int(os.environ.get(...))` the value
    # was frozen at module import — bootstrap-time setdefault arrived too late
    # and the cap stayed at 4 even when QR_MAX_PORTFOLIO_POSITIONS=10 was set
    # (regression seen 2026-05-11 15:44 JST: 48 intents DRY_RUN_BLOCKED with
    # "open trader positions 5 reached portfolio limit 4").
    max_portfolio_positions: int = field(
        default_factory=lambda: int(os.environ.get("QR_MAX_PORTFOLIO_POSITIONS", "4") or "4")
    )
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
            sl_free_active_for_layer = _trader_sl_repair_disabled()
            for position in entry_relevant_positions:
                # SL-free regime: trader-owned positions with TP but no SL are
                # treated as protected for layering. The 「SLいらない」 directive
                # makes SL=None deliberate; TP remains as the harvest protection.
                pos_eligible = (
                    position.owner == Owner.TRADER
                    and position.take_profit is not None
                    and (position.stop_loss is not None or sl_free_active_for_layer)
                )
                if not pos_eligible:
                    issues.append(
                        RiskIssue(
                            "OPEN_POSITION_EXISTS",
                            f"only protected trader-owned positions can be layered; "
                            f"{position.pair} {position.side.value} id={position.trade_id} is not eligible",
                        )
                    )
                elif position.pair == intent.pair and position.side != intent.side and (
                    not _account_hedging_enabled(snapshot) or not _intent_declares_hedge(intent)
                ):
                    issues.append(
                        RiskIssue(
                            "OPPOSING_POSITION_NEEDS_HEDGING",
                            f"fresh {intent.pair} {intent.side.value} entry opposes protected "
                            f"{position.pair} {position.side.value} id={position.trade_id}; "
                            "opposite-side adds require both broker hedging proof and explicit "
                            "intent.metadata['position_intent']='HEDGE'",
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
            sl_free_active = _trader_sl_repair_disabled()
            for position in entry_relevant_positions:
                missing = []
                if position.take_profit is None:
                    missing.append("TP")
                if position.stop_loss is None:
                    if sl_free_active and position.owner == Owner.TRADER:
                        # User directive 「SLいらない」 — trader-owned SL=None is
                        # deliberate and does not block fresh entries. TP-only.
                        pass
                    else:
                        missing.append("SL")
                if missing:
                    issues.append(
                        RiskIssue(
                            "UNPROTECTED_POSITION",
                            f"open position lacks {'/'.join(missing)}: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u",
                        )
                    )

        if self.policy.block_new_entries_with_pending_entry_orders:
            for order in snapshot.orders:
                if _is_pending_entry_order(order) and not _is_operator_managed_manual_order(order):
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

        metrics = self._metrics(intent, quote, spec, entry_price, quote_to_jpy, snapshot)
        issues.extend(self._margin_issues(snapshot, metrics))
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
        elif metrics.risk_jpy > loss_cap and not _trader_sl_repair_disabled():
            # Under SL-free mode (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
            # 「損失を出さないで稼ぎまくる」 2026-05-07), per-trade loss cap is
            # advisory only: sizing is operator-set + margin headroom-bound.
            # The cap stays active when SL-free is off so the conservative
            # SL-anchored regime keeps its 30-trade-divisor protection.
            issues.append(
                RiskIssue(
                    "LOSS_CAP_EXCEEDED",
                    f"planned worst-case loss {metrics.risk_jpy:.0f} JPY exceeds cap {loss_cap:.0f} JPY",
                )
            )
        # Regime-derived reward/risk floor. RANGE regimes are allowed below the
        # default min_reward_risk because rotation geometry caps TP at the
        # opposing rail and high hit-rate range scalps remain +EV at sub-1R.
        # Falls back to default min when regime is missing/unclear so silent
        # data gaps cannot relax the floor (per AGENT_CONTRACT §3.5).
        regime_state = ""
        if intent.metadata:
            raw_state = intent.metadata.get("regime_state")
            if isinstance(raw_state, str):
                regime_state = raw_state.upper()
        active_min_rr = (
            self.policy.range_min_reward_risk
            if "RANGE" in regime_state
            else self.policy.min_reward_risk
        )
        if metrics.reward_risk < active_min_rr:
            issues.append(
                RiskIssue(
                    "REWARD_RISK_TOO_LOW",
                    f"planned reward/risk {metrics.reward_risk:.2f}x is below {active_min_rr:.2f}x"
                    + (f" (regime={regime_state})" if regime_state else ""),
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
                # Under SL-free the per-day loss budget is advisory only —
                # margin-utilization is the real ceiling, not a JPY literal.
                # User 2026-05-08「市況>リスク」+ `feedback_offense_sizing.md`
                # 「loss cap撤廃」+ `feedback_market_over_risk_budget.md`
                # 「含み損%/JPYは判断材料にしない」.
                # Surface the gate as WARN under SL-free so the operator
                # sees the exposure but the cycle isn't blocked.
                severity = "WARN" if _trader_sl_repair_disabled() else "BLOCK"
                issues.append(
                    RiskIssue(
                        "PORTFOLIO_LOSS_CAP_EXCEEDED",
                        f"open risk {portfolio_risk:.0f} JPY + candidate risk {metrics.risk_jpy:.0f} JPY "
                        f"exceeds portfolio cap {self.policy.max_portfolio_loss_jpy:.0f} JPY",
                        severity=severity,
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
        snapshot: BrokerSnapshot,
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
        estimated_margin = estimate_required_margin_jpy(
            units=intent.units,
            entry_price=entry_price,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
        )
        account = snapshot.account
        max_margin_pct = self.policy.max_margin_utilization_pct
        margin_budget = None
        margin_after_utilization = None
        margin_used = None
        margin_available = None
        if account is not None:
            margin_used = account.margin_used_jpy
            margin_available = account.margin_available_jpy
            if max_margin_pct is not None and account.nav_jpy > 0:
                margin_budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
                margin_after_utilization = (account.margin_used_jpy + estimated_margin) / account.nav_jpy * 100.0
        return RiskMetrics(
            entry_price=entry_price,
            loss_pips=loss_pips,
            reward_pips=reward_pips,
            risk_jpy=risk_jpy,
            reward_jpy=reward_jpy,
            reward_risk=reward_risk,
            spread_pips=spread_pips,
            jpy_per_pip=jpy_per_pip,
            estimated_margin_jpy=estimated_margin,
            margin_used_jpy=margin_used,
            margin_available_jpy=margin_available,
            margin_budget_jpy=margin_budget,
            margin_utilization_after_pct=margin_after_utilization,
            max_margin_utilization_pct=max_margin_pct,
        )

    def _margin_issues(self, snapshot: BrokerSnapshot, metrics: RiskMetrics) -> list[RiskIssue]:
        max_margin_pct = self.policy.max_margin_utilization_pct
        if max_margin_pct is None:
            return []
        if max_margin_pct <= 0 or max_margin_pct > 100:
            return [
                RiskIssue(
                    "INVALID_MARGIN_POLICY",
                    f"max_margin_utilization_pct must be within 0-100, got {max_margin_pct}",
                )
            ]
        account = snapshot.account
        if account is None:
            if not self.policy.require_margin_account:
                return []
            return [
                RiskIssue(
                    "MARGIN_ACCOUNT_MISSING",
                    "broker account summary is required to enforce margin availability and utilization cap",
                )
            ]
        issues: list[RiskIssue] = []
        if account.nav_jpy <= 0:
            issues.append(
                RiskIssue(
                    "MARGIN_NAV_INVALID",
                    f"account NAV must be positive to enforce {max_margin_pct:.1f}% margin cap; got {account.nav_jpy:.0f} JPY",
                )
            )
        if account.margin_used_jpy < 0:
            issues.append(
                RiskIssue(
                    "MARGIN_USED_INVALID",
                    f"account margin_used_jpy must be non-negative; got {account.margin_used_jpy:.0f} JPY",
                )
            )
        if account.margin_available_jpy < 0:
            issues.append(
                RiskIssue(
                    "MARGIN_AVAILABLE_INVALID",
                    f"account margin_available_jpy must be non-negative; got {account.margin_available_jpy:.0f} JPY",
                )
            )
        if issues:
            return issues

        budget = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
        cap_jpy = account.nav_jpy * (max_margin_pct / 100.0)
        if metrics.estimated_margin_jpy > account.margin_available_jpy:
            issues.append(
                RiskIssue(
                    "MARGIN_AVAILABLE_EXCEEDED",
                    f"estimated initial margin {metrics.estimated_margin_jpy:.0f} JPY exceeds "
                    f"broker marginAvailable {account.margin_available_jpy:.0f} JPY",
                )
            )
        if budget <= 0 and metrics.estimated_margin_jpy > 0:
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_REACHED",
                    f"current marginUsed {account.margin_used_jpy:.0f} JPY already reaches/exceeds "
                    f"{max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY",
                )
            )
        elif metrics.estimated_margin_jpy > budget:
            after = account.margin_used_jpy + metrics.estimated_margin_jpy
            issues.append(
                RiskIssue(
                    "MARGIN_UTILIZATION_CAP_EXCEEDED",
                    f"candidate margin {metrics.estimated_margin_jpy:.0f} JPY would raise marginUsed to "
                    f"{after:.0f} JPY, above {max_margin_pct:.1f}% NAV cap {cap_jpy:.0f} JPY "
                    f"(remaining budget {budget:.0f} JPY)",
                )
            )
        return issues

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
        sl_free_active = _trader_sl_repair_disabled()
        # Synthetic-SL distance for trader-owned SL-free positions: assume the
        # discretionary close happens within the SL-free invalidation budget
        # (5x M5 ATR ≈ 25 pips on majors). Hard-coded conservative estimate
        # so basket math has a real number; refine later from pair_charts ATR.
        SL_FREE_SYNTHETIC_PIPS = 30.0
        for position in self._entry_relevant_positions(snapshot):
            spec = self._spec(position.pair)
            quote_to_jpy = self._quote_to_jpy(position.pair, snapshot)
            if quote_to_jpy is None:
                return 0.0, RiskIssue(
                    "PORTFOLIO_RISK_UNKNOWN",
                    f"missing conversion quote for open position {position.trade_id} {position.pair}",
                )
            if position.stop_loss is None:
                if sl_free_active and position.owner == Owner.TRADER:
                    loss_pips = SL_FREE_SYNTHETIC_PIPS
                else:
                    return 0.0, RiskIssue(
                        "PORTFOLIO_RISK_UNKNOWN",
                        f"open position {position.trade_id} has no SL; cannot compute portfolio risk",
                    )
            else:
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


def _intent_declares_hedge(intent: OrderIntent) -> bool:
    return str((intent.metadata or {}).get("position_intent") or "").upper() == "HEDGE"


def _is_operator_managed_manual(position: BrokerPosition) -> bool:
    return position.owner in {Owner.MANUAL, Owner.UNKNOWN}


def _is_operator_managed_manual_order(order: BrokerOrder) -> bool:
    return order.owner in {Owner.MANUAL, Owner.UNKNOWN}


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
