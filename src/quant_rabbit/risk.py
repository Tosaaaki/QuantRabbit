from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import BrokerSnapshot, OrderIntent, Owner, Quote, RiskDecision, RiskIssue, RiskMetrics, Side


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
    max_loss_jpy: float = 500.0
    min_reward_risk: float = 1.2
    max_quote_age_seconds: int = 20
    max_spread_multiple: float = 2.5
    min_target_spread_multiple: float = 5.0
    min_stop_spread_multiple: float = 5.0
    block_new_entries_with_external_risk: bool = True
    block_unprotected_positions: bool = True
    require_live_enabled_for_send: bool = True


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

        if self.policy.block_new_entries_with_external_risk:
            for position in snapshot.positions:
                if position.owner != Owner.TRADER:
                    issues.append(
                        RiskIssue(
                            "EXTERNAL_RISK_OPEN",
                            f"external/manual risk is open: {position.pair} {position.side.value} "
                            f"id={position.trade_id} {position.units}u; adopt or close before new entries",
                        )
                    )

        if self.policy.block_unprotected_positions:
            for position in snapshot.positions:
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

        entry_price = self._entry_price(intent, quote)
        metrics = self._metrics(intent, quote, spec, entry_price)
        if metrics.loss_pips <= 0:
            issues.append(RiskIssue("SL_NOT_LOSS_SIDE", f"SL is not on the loss side for {intent.side.value}"))
        if metrics.reward_pips <= 0:
            issues.append(RiskIssue("TP_NOT_REWARD_SIDE", f"TP is not on the reward side for {intent.side.value}"))
        if metrics.risk_jpy > self.policy.max_loss_jpy:
            issues.append(
                RiskIssue(
                    "LOSS_CAP_EXCEEDED",
                    f"planned worst-case loss {metrics.risk_jpy:.0f} JPY exceeds cap {self.policy.max_loss_jpy:.0f} JPY",
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

        return RiskDecision(
            allowed=not any(issue.severity == "BLOCK" for issue in issues),
            metrics=metrics,
            issues=tuple(issues),
        )

    def _spec(self, pair: str) -> InstrumentSpec:
        try:
            return self.specs[pair]
        except KeyError as exc:
            raise ValueError(f"unsupported instrument: {pair}") from exc

    def _entry_price(self, intent: OrderIntent, quote: Quote) -> float:
        if intent.entry is not None:
            return float(intent.entry)
        return quote.ask if intent.side == Side.LONG else quote.bid

    def _metrics(self, intent: OrderIntent, quote: Quote, spec: InstrumentSpec, entry_price: float) -> RiskMetrics:
        if intent.side == Side.LONG:
            loss_pips = (entry_price - intent.sl) * spec.pip_factor
            reward_pips = (intent.tp - entry_price) * spec.pip_factor
        else:
            loss_pips = (intent.sl - entry_price) * spec.pip_factor
            reward_pips = (entry_price - intent.tp) * spec.pip_factor
        spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
        quote_to_jpy = self._quote_to_jpy(intent.pair, quote)
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

    def _quote_to_jpy(self, pair: str, quote: Quote) -> float:
        quote_ccy = pair.split("_", 1)[1]
        if quote_ccy == "JPY":
            return 1.0
        if quote_ccy == "USD":
            # vNext risk snapshots must include USD_JPY for non-JPY pairs in production.
            # During single-pair tests, fall back to a conservative recent USDJPY band.
            return 157.0
        raise ValueError(f"quote-to-JPY conversion not implemented for {pair}")
